# inference.py — Moonshine Streaming ASR inference (native PyTorch).
#
# Usage:
#   python inference.py --audio audio.wav [--weights model.pth] [--tokenizer tokenizer.json]
#
# Long audio:
#   Automatically detected. When audio duration exceeds one safe chunk, the
#   pipeline uses Silero VAD to find natural speech boundaries, splits into
#   chunks, and transcribes each independently.
#   Decoder KV cache is still rebuilt fresh inside model.generate() every call,
#   but long-form mode may carry a tiny encoder-side memory across nearby
#   chunks when the boundary looks continuous.
#   Output includes both a full text string and timestamped segments.

import sys
import torch
import argparse
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from math import gcd
from pathlib import Path
from model import (
    MoonshineStreamingForConditionalGeneration,
    LongFormMemoryState,
    MEMORY_SIMILARITY_THRESHOLD,
    _memory_similarity,
)
from scipy.signal import resample_poly

from config import (
    MoonshineStreamingConfig, InferenceConfig,
    get_device, get_dtype,
    dbg, dbg_tensor, dbg_audio, DEBUG
)

# vad.py lives one directory up (src/) relative to this file (src/moonshine_streaming/)
sys.path.insert(0, str(Path(__file__).parent.parent))
from vad import (
    load_silero_vad, run_vad, smart_chunk,
    derive_chunk_params, free_memory,
    ChunkParams,
)
from longform import (
    LongFormTuning,
    assess_transcript,
    apply_quality_cap,
    choose_better_transcript,
    estimate_decode_budget,
    stitch_segments_text,
    stitch_segments_with_confidence,
)

# ---------------------------------------------------------------------------
# Moonshine-specific chunk parameter derivation
# ---------------------------------------------------------------------------

# Moonshine encoder timing:
#   frame_ms = 5 ms -> 200 frames/sec
#   Conv frontend applies 4x total downsampling -> 50 encoder tokens/sec
_MOONSHINE_ENCODER_TOKENS_PER_SEC = 50.0
_MOONSHINE_MEMORY_RESET_GAP_SEC = 1.5
_MOONSHINE_RESCUE_RIGHT_CONTEXT_SEC = 1.6
_MOONSHINE_RESCUE_RIGHT_CONTEXT_FACTOR = 1.35

# B1: Content-based memory reset threshold.
#
# Two-tier reset logic (both are checked independently):
#   - TIME-based:    reset when VAD gap > _MOONSHINE_MEMORY_RESET_GAP_SEC
#   - CONTENT-based: reset when acoustic similarity falls below this value,
#                    even if the time gap is small (e.g. speaker change with
#                    a brief pause).
#
# This threshold is intentionally lower than MEMORY_SIMILARITY_THRESHOLD
# (the merge gate). The merge gate (0.35) asks "is this similar enough to
# USE the memory?". The reset gate asks "is this so different that carrying
# memory would actively hurt?" — a much weaker bar.
#
# 0.18 was too aggressive in practice: on long narration files, the multi-point
# gate (min of 3 probe cosine similarities) fired on same-speaker pauses between
# sentences, discarding useful acoustic context. Lowering to 0.13 preserves
# memory through natural within-speaker pauses while still resetting cleanly
# at true discontinuities (speaker changes, abrupt topic jumps).
_MOONSHINE_MEMORY_CONTENT_RESET_THRESHOLD = 0.13

# B2: Adaptive memory budget per chunk duration.
#
# The fixed 12-token budget is fine for full-length chunks (~10 s), but
# for short chunks produced by recursive splitting (4-6 s), 12 prepended
# memory tokens represent a disproportionately large fraction of the
# cross-attention context and can dominate decoder attention.
#
# Formula: tokens = clamp(round(dur * TOKENS_PER_SEC), min, max)
# At 10 s → 12 tokens; at 6 s → 8 tokens; at 4 s → 5 tokens.
_MOONSHINE_MEMORY_BUDGET_TOKENS_PER_SEC = 1.2
_MOONSHINE_MEMORY_BUDGET_MIN = 4
_MOONSHINE_MEMORY_BUDGET_MAX = 12  # matches _MEMORY_BUDGET_TOTAL in model.py

# Practical decoder limit: max_position_embeddings=4096 decoder positions.
# Moonshine decodes ~6.5 tokens/sec of speech (empirical from Whisper-class models).
# This gives ~630 seconds of practical maximum before the decoder PE limit is exceeded.
_MOONSHINE_DECODER_TOKENS_PER_SEC = 6.5
_MOONSHINE_LONGFORM = LongFormTuning(
    quality_target_dur=10.0,
    quality_max_dur=12.0,
    min_split_dur=6.0,
    split_overlap_dur=1.25,
    max_split_depth=3,
    decoder_tokens_per_second=3.25,
    decoder_token_buffer=10,
    decoder_min_tokens=20,
    decoder_max_tokens=72,
)


def _moonshine_model_max_seconds() -> float:
    """
    Derive practical max audio duration for one Moonshine forward pass.

    Moonshine's encoder is ergodic (no fixed PE table), so the limit comes from
    the decoder's max_position_embeddings (4096 positions). We estimate how many
    seconds of audio produce 4096 decoder tokens at the observed decode rate.

    Returns seconds of audio that would fill the decoder's position table.
    """
    config = MoonshineStreamingConfig()
    return config.decoder.max_position_embeddings / _MOONSHINE_DECODER_TOKENS_PER_SEC


def _moonshine_chunk_params(device: torch.device, dtype: torch.dtype) -> ChunkParams:
    """
    Compute Moonshine chunk parameters from model config + runtime hardware.
    Called once at the start of transcribe_long().

    Moonshine can technically accept much longer audio than we want to feed it
    in practice. The final clamp keeps chunk sizes inside the quality regime
    where the decoder is still stable on narration.
    """
    config = MoonshineStreamingConfig()
    base_params = derive_chunk_params(
        model_max_encoder_seconds=_moonshine_model_max_seconds(),
        model_dtype=dtype,
        device=device,
        model_hidden_size=config.encoder.hidden_size,
        model_num_layers=config.encoder.num_hidden_layers,
        model_num_heads=config.encoder.num_attention_heads,
    )
    return apply_quality_cap(base_params, _MOONSHINE_LONGFORM)


def _adaptive_memory_budget(duration_sec: float) -> int:
    """
    Derive memory token budget from chunk duration (B2).

    Short chunks should carry fewer memory tokens so the prepended memory does
    not dominate the current chunk's cross-attention context. The cap stays at
    12 to preserve the original total memory ceiling.
    """
    estimated = round(duration_sec * _MOONSHINE_MEMORY_BUDGET_TOKENS_PER_SEC)
    return max(_MOONSHINE_MEMORY_BUDGET_MIN, min(_MOONSHINE_MEMORY_BUDGET_MAX, estimated))


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer(tokenizer_path: str):
    """Load a HuggingFace fast tokenizer from tokenizer.json (requires `tokenizers`)."""
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise ImportError("Install `tokenizers`: pip install tokenizers")

    path = Path(tokenizer_path)
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")

    tokenizer = Tokenizer.from_file(str(path))
    dbg("tokenizer", f"loaded from {path}, vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


def decode_tokens(tokenizer, token_ids: list) -> str:
    """Decode token IDs to text, skipping special tokens."""
    return tokenizer.decode(token_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Load an audio file as mono float32 at target_sr.
    Resamples if the source sample rate differs.
    """
    dbg("load_audio", f"loading {audio_path}")

    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)

    # Stereo -> mono
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
        dbg("load_audio", "converted stereo to mono")

    # Resample if needed.
    # We use polyphase resampling here because it is deterministic, fast, and
    # much more reliable in this environment than the previous librosa path.
    if sr != target_sr:
        print(f"[WARN] Resampling {sr} Hz -> {target_sr} Hz ...")
        try:
            g = gcd(target_sr, sr)
            up, down = target_sr // g, sr // g
            audio = resample_poly(audio, up, down).astype(np.float32)
        except ImportError:
            ratio   = target_sr / sr
            new_len = int(len(audio) * ratio)
            x_old   = np.linspace(0, len(audio) - 1, len(audio))
            x_new   = np.linspace(0, len(audio) - 1, new_len)
            audio   = np.interp(x_new, x_old, audio)

    dbg("load_audio", f"duration={len(audio)/target_sr:.2f}s samples={len(audio)}")
    return audio.astype(np.float32)


def preprocess_audio(
    audio_np:          np.ndarray,
    device:            torch.device,
    dtype:             torch.dtype,
    pad_to_multiple_of: int = 80,
) -> tuple:
    """
    Convert numpy audio to tensor and pad to a multiple of pad_to_multiple_of samples.

    Returns:
        input_values:   [1, padded_len] float tensor
        attention_mask: [1, padded_len] bool tensor (True=valid, False=padding)
    """
    audio_t  = torch.from_numpy(audio_np.copy()).float()
    orig_len = audio_t.shape[0]
    dbg_audio("preprocess/raw", audio_t.unsqueeze(0), 16000)

    # Pad to the nearest multiple of pad_to_multiple_of
    remainder = orig_len % pad_to_multiple_of
    if remainder != 0:
        pad_amount = pad_to_multiple_of - remainder
        audio_t    = F.pad(audio_t, (0, pad_amount))
        dbg("preprocess", f"padded {orig_len} -> {audio_t.shape[0]} (+{pad_amount})")

    padded_len = audio_t.shape[0]

    # Attention mask: True for original samples, False for padding
    attention_mask = torch.zeros(padded_len, dtype=torch.bool)
    attention_mask[:orig_len] = True

    input_values   = audio_t.unsqueeze(0).to(device=device, dtype=dtype)   # [1, padded_len]
    attention_mask = attention_mask.unsqueeze(0).to(device=device)          # [1, padded_len]

    dbg_tensor("preprocess/input_values", input_values)
    dbg_tensor("preprocess/attention_mask", attention_mask.float())

    return input_values, attention_mask


# ---------------------------------------------------------------------------
# Per-chunk inference helper
# ---------------------------------------------------------------------------

def _transcribe_waveform_chunk(
    waveform_np: np.ndarray,
    model:       MoonshineStreamingForConditionalGeneration,
    tokenizer,
    device:      torch.device,
    dtype:       torch.dtype,
    max_new_tokens: int | None = None,
    memory_state: LongFormMemoryState | None = None,
    return_memory_state: bool = False,
    memory_token_budget: int = _MOONSHINE_MEMORY_BUDGET_MAX,
    beam_width: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    eos_min_steps: int = 0,
    return_token_confidences: bool = False,
) -> "str | tuple[str, LongFormMemoryState] | tuple[str, list[float]] | tuple[str, LongFormMemoryState, list[float]]":
    """
    Transcribe a pre-loaded waveform slice (no file I/O).
    Used internally by transcribe_long() for each chunk.

    model.generate() always builds a fresh decoder KV cache, so generated text
    from one chunk does not directly leak into the next chunk. The only
    optional carry-over here is a tiny encoder-side memory, which is safer for
    long-form continuity on low-resource setups.

    return_token_confidences mirrors the same flag on model.generate(). When
    True, per-token softmax top-1 probabilities are returned alongside the text
    so the caller can use them for confidence-weighted stitching decisions.
    """
    input_values, attention_mask = preprocess_audio(
        waveform_np, device, dtype,
        pad_to_multiple_of=InferenceConfig.pad_to_multiple_of,
    )

    with torch.inference_mode():
        result = model.generate(
            input_values=input_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            memory_state=memory_state,
            return_memory_state=return_memory_state,
            memory_token_budget=memory_token_budget,
            beam_width=beam_width,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_min_steps=eos_min_steps,
            return_token_confidences=return_token_confidences,
        )

    # Unpack the variable-length result tuple from model.generate().
    # The four cases match the four return variants documented in generate().
    if return_memory_state and return_token_confidences:
        if isinstance(result, tuple) and len(result) == 3:
            token_ids, next_memory_state, token_confidences = result
            return decode_tokens(tokenizer, token_ids), next_memory_state, token_confidences
        # Beam search fallback: confidence unavailable.
        token_ids, next_memory_state = result
        return decode_tokens(tokenizer, token_ids), next_memory_state, []

    if return_memory_state:
        token_ids, next_memory_state = result
        return decode_tokens(tokenizer, token_ids), next_memory_state

    if return_token_confidences:
        if isinstance(result, tuple) and len(result) == 2:
            token_ids, token_confidences = result
            return decode_tokens(tokenizer, token_ids), token_confidences
        # Beam search fallback.
        return decode_tokens(tokenizer, result), []

    return decode_tokens(tokenizer, result)


def _slice_waveform_interval(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, float]:
    """
    Slice one audio interval from the shared waveform.

    Keeping this in one helper avoids repeating slightly different index math
    across the normal path, retry path, and contextual rescue path.
    """
    start_sample = max(0, int(start_sec * sample_rate))
    end_sample = min(int(end_sec * sample_rate), len(waveform))
    chunk_wav = waveform[start_sample:end_sample]
    duration_sec = (end_sample - start_sample) / sample_rate
    return chunk_wav, duration_sec


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def transcribe(
    audio_path:     str,
    weights_path:   str,
    tokenizer_path: str,
    max_new_tokens: int | None = None,
) -> str:
    """
    Full pipeline: load model -> load audio -> encode -> greedy decode -> text.

    Returns the transcription string.
    """
    device = get_device()
    dtype  = get_dtype(device)
    print(f"[INFO] Device: {device}, dtype: {dtype}")

    config = MoonshineStreamingConfig()
    model  = MoonshineStreamingForConditionalGeneration(config)

    if DEBUG:
        from config import dbg_module
        dbg_module("full_model", model)

    model.load_weights(weights_path, device)
    model.to(device=device, dtype=dtype)
    model.eval()
    print("[INFO] Model ready.")

    if DEBUG:
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        dbg("model_params", f"total={total:,} trainable={trainable:,}")

    tokenizer = load_tokenizer(tokenizer_path)

    audio_np = load_audio(audio_path, target_sr=config.encoder.sample_rate)
    input_values, attention_mask = preprocess_audio(
        audio_np, device, dtype,
        pad_to_multiple_of=InferenceConfig.pad_to_multiple_of,
    )

    print("[INFO] Running inference...")
    with torch.inference_mode():
        token_ids = model.generate(
            input_values=input_values,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
        )

    dbg("generate/result_ids", str(token_ids))
    return decode_tokens(tokenizer, token_ids)


# ---------------------------------------------------------------------------
# Long audio inference
# ---------------------------------------------------------------------------

def transcribe_long(
    audio_path:     str,
    weights_path:   str,
    tokenizer_path: str,
    storage_dir:    str = str(Path(__file__).parent.parent.parent / "storage"),
) -> dict:
    """
    Transcribe a long audio file using VAD-based chunking.

    This function:
        1. Loads the model and tokenizer (once).
        2. Loads the full waveform (once).
        3. Derives chunk parameters from model config + available hardware memory.
        4. If the audio fits in one chunk, runs direct inference.
        5. Otherwise:
           a. Loads Silero VAD (auto-detect local file or download to storage/).
           b. Detects speech boundaries.
           c. Merges boundaries into chunks using derived parameters.
           d. Transcribes each chunk with fresh decoder state, while optionally
              carrying a tiny bounded encoder memory across nearby chunks.
           e. Frees GPU/CPU memory between chunks.

    Args:
        audio_path:     path to any audio file supported by soundfile.
        weights_path:   path to .pth weights file.
        tokenizer_path: path to tokenizer.json.
        storage_dir:    directory to look for / save silero_vad.jit.

    Returns:
        {
            "full_text": str,
            "segments":  [{"start": float, "end": float, "text": str}, ...]
        }
        For short audio, "segments" contains a single entry with start=0.0.
    """
    device = get_device()
    dtype  = get_dtype(device)
    print(f"[INFO] Device: {device}, dtype: {dtype}")

    # Load model once — shared across all chunks
    config = MoonshineStreamingConfig()
    model  = MoonshineStreamingForConditionalGeneration(config)
    model.load_weights(weights_path, device)
    model.to(device=device, dtype=dtype)
    model.eval()
    print("[INFO] Model ready.")

    tokenizer = load_tokenizer(tokenizer_path)

    # Load full waveform once
    waveform       = load_audio(audio_path, target_sr=config.encoder.sample_rate)
    audio_duration = len(waveform) / config.encoder.sample_rate
    print(f"[INFO] Audio duration: {audio_duration:.1f}s")

    # Derive chunk parameters from hardware + model config
    params = _moonshine_chunk_params(device, dtype)
    print(f"[INFO] Chunk params: target={params.target_dur:.1f}s  "
          f"max={params.max_dur:.1f}s  min={params.min_dur:.1f}s  "
          f"overlap_fallback={params.overlap_fallback:.1f}s")

    # Short audio — skip VAD, run single inference
    if audio_duration <= params.max_dur:
        print("[INFO] Audio fits in one chunk — running direct inference.")
        max_new_tokens = estimate_decode_budget(audio_duration, _MOONSHINE_LONGFORM)
        text = _transcribe_waveform_chunk(
            waveform, model, tokenizer, device, dtype,
            max_new_tokens=max_new_tokens,
        )
        return {
            "full_text": text,
            "segments":  [{"start": 0.0, "end": audio_duration, "text": text}],
        }

    # Long audio — VAD + chunking
    print("[INFO] Long audio detected — loading Silero VAD ...")
    vad_model = load_silero_vad(storage_dir, device=device)

    print("[INFO] Running VAD ...")
    vad_segments = run_vad(vad_model, waveform, device=device)
    print(f"[INFO] VAD found {len(vad_segments)} speech segment(s)")

    # Free VAD model — no longer needed
    del vad_model
    free_memory(device)

    chunks = smart_chunk(vad_segments, audio_duration, params)
    print(f"[INFO] Chunked into {len(chunks)} inference chunk(s)")

    sample_rate = config.encoder.sample_rate
    memory_reset_gap = max(_MOONSHINE_MEMORY_RESET_GAP_SEC, params.overlap_fallback * 1.5)

    def _transcribe_controlled(
        chunk_wav: np.ndarray,
        chunk_dur: float,
        incoming_memory: LongFormMemoryState | None,
        start_sec: float,
        end_sec: float,
    ) -> tuple[str, LongFormMemoryState | None, list[float]]:
        """
        Run one chunk with a conservative token budget.

        Moonshine is strongest when it stops close to the spoken content. The
        tighter budget here is a long-form guardrail, not a model change.

        For chunks that still look suspicious after the first pass, we try one
        more decode with a slightly larger token budget and keep the healthier
        transcript. This stays cheap because the retry happens only when the
        text is obviously incomplete or abnormal.

        Both decode attempts receive the SAME incoming memory. That keeps the
        comparison fair and avoids drift where the retry path would see a
        different cross-chunk context than the first attempt.

        If the chunk still looks truncated or under-filled, we allow one more
        rescue pass with a SMALL amount of right acoustic context. The idea is
        simple: many boundary mistakes happen because the decoder stops before
        it has heard enough of the upcoming phrase. Adding a short right tail is
        much cheaper than globally shrinking the whole model window to tiny
        slices, and the extra words can still be deduplicated later by the
        normal stitching logic.

        Confidence tracking:
        Only the primary (first-pass) decode requests per-token confidence scores.
        Retry and rescue passes do not, because those paths select a winner via
        choose_better_transcript() (heuristic comparison), not word-level confidence.
        If retry or rescue wins, we clear the confidence list so the stitcher knows
        no reliable data is available for that segment and falls back to exact-match.

        Returns:
            (text, next_memory_state, token_confidences)
            token_confidences is [] when retry or rescue won, or on error.
        """
        base_budget = estimate_decode_budget(chunk_dur, _MOONSHINE_LONGFORM)
        memory_budget = _adaptive_memory_budget(chunk_dur)

        # Primary decode: request confidence alongside memory state.
        text, next_memory_state, token_confidences = _transcribe_waveform_chunk(
            chunk_wav, model, tokenizer, device, dtype,
            max_new_tokens=base_budget,
            memory_state=incoming_memory,
            return_memory_state=True,
            memory_token_budget=memory_budget,
            return_token_confidences=True,
        )

        assessment = assess_transcript(text, chunk_dur)
        should_retry = (
            assessment.is_degenerate
            or assessment.is_too_short
            or (assessment.looks_truncated and chunk_dur >= 9.0)
        )
        if should_retry:
            expanded_budget = min(96, base_budget + max(8, int(chunk_dur * 0.8)))
            alternate_text = _transcribe_waveform_chunk(
                chunk_wav, model, tokenizer, device, dtype,
                max_new_tokens=expanded_budget,
                memory_state=incoming_memory,
                memory_token_budget=memory_budget,
            )
            best = choose_better_transcript(text, alternate_text, chunk_dur)
            if best != text:
                # Alternate won: its token IDs were never tracked, so no confidence.
                text = best
                token_confidences = []
            else:
                text = best

        final_assessment = assess_transcript(text, chunk_dur)
        needs_right_context_rescue = (
            end_sec < audio_duration
            and (
                final_assessment.is_too_short
                or final_assessment.looks_truncated
            )
        )
        if needs_right_context_rescue:
            rescue_context = min(
                _MOONSHINE_RESCUE_RIGHT_CONTEXT_SEC,
                max(params.overlap_fallback, chunk_dur * 0.10) * _MOONSHINE_RESCUE_RIGHT_CONTEXT_FACTOR,
            )
            rescue_end_sec = min(audio_duration, end_sec + rescue_context)

            # The rescue decode reuses the same LEFT chunk plus a small right
            # tail. We keep the same incoming memory so the left boundary
            # remains stable, but allow a slightly larger budget because the
            # waveform is now longer than the base chunk.
            rescue_wav, rescue_dur = _slice_waveform_interval(
                waveform=waveform,
                sample_rate=sample_rate,
                start_sec=start_sec,
                end_sec=rescue_end_sec,
            )
            rescue_budget = estimate_decode_budget(rescue_dur, _MOONSHINE_LONGFORM)
            rescue_budget = min(104, max(rescue_budget, base_budget + 6))
            rescue_text = _transcribe_waveform_chunk(
                rescue_wav, model, tokenizer, device, dtype,
                max_new_tokens=rescue_budget,
                memory_state=incoming_memory,
                memory_token_budget=_adaptive_memory_budget(rescue_dur),
            )
            best = choose_better_transcript(text, rescue_text, rescue_dur)
            if best != text:
                # Rescue won: confidence from the original chunk no longer applies.
                text = best
                token_confidences = []
            else:
                text = best

        final_assessment = assess_transcript(text, chunk_dur)
        if final_assessment.is_degenerate or final_assessment.is_too_short:
            next_memory_state = None

        free_memory(device)
        return text, next_memory_state, token_confidences

    def _transcribe_interval(
        start_sec: float,
        end_sec: float,
        incoming_memory: LongFormMemoryState | None,
        depth: int = 0,
    ) -> tuple[list[dict], LongFormMemoryState | None]:
        """
        Transcribe one interval and split only when the output still looks
        unhealthy after the controlled retry path.

        This helper is Moonshine-specific because the generic long-form helper
        only returns text, while Moonshine now also needs to thread a bounded
        encoder memory through the left and right children of a split.
        """
        chunk_wav, duration_sec = _slice_waveform_interval(
            waveform=waveform,
            sample_rate=sample_rate,
            start_sec=start_sec,
            end_sec=end_sec,
        )

        text, outgoing_memory, token_confidences = _transcribe_controlled(
            chunk_wav=chunk_wav,
            chunk_dur=duration_sec,
            incoming_memory=incoming_memory,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        text = text.strip()
        assessment = assess_transcript(text, duration_sec)

        should_split = (
            depth < _MOONSHINE_LONGFORM.max_split_depth
            and duration_sec > _MOONSHINE_LONGFORM.min_split_dur
            and (
                assessment.is_degenerate
                or (assessment.is_too_short and duration_sec >= 8.0)
            )
        )
        if not should_split:
            # Attach token_confidences so stitch_segments_with_confidence() can
            # use them for confidence-weighted overlap resolution at boundaries.
            return [
                {
                    "start": start_sec,
                    "end": end_sec,
                    "text": text,
                    "token_confidences": token_confidences,
                }
            ], outgoing_memory

        overlap = min(_MOONSHINE_LONGFORM.split_overlap_dur, duration_sec * 0.15)
        midpoint = (start_sec + end_sec) / 2.0
        left_end = min(end_sec, midpoint + overlap / 2.0)
        right_start = max(start_sec, midpoint - overlap / 2.0)

        left_dur = left_end - start_sec
        right_dur = end_sec - right_start
        if left_dur < _MOONSHINE_LONGFORM.min_split_dur or right_dur < _MOONSHINE_LONGFORM.min_split_dur:
            return [{"start": start_sec, "end": end_sec, "text": text}], outgoing_memory

        print(
            "[INFO] Poor chunk quality detected "
            f"({_fmt_time(start_sec)} - {_fmt_time(end_sec)}); retrying as "
            f"{left_dur:.1f}s + {right_dur:.1f}s"
        )

        left_segments, left_memory = _transcribe_interval(
            start_sec=start_sec,
            end_sec=left_end,
            incoming_memory=incoming_memory,
            depth=depth + 1,
        )
        right_segments, right_memory = _transcribe_interval(
            start_sec=right_start,
            end_sec=end_sec,
            incoming_memory=left_memory,
            depth=depth + 1,
        )
        return left_segments + right_segments, right_memory

    segments: list[dict] = []
    memory_state: LongFormMemoryState | None = None
    prev_chunk_end_sec: float | None = None
    for idx, (start_sec, end_sec) in enumerate(chunks):
        chunk_dur = end_sec - start_sec
        print(f"[INFO] Chunk {idx+1}/{len(chunks)}: "
              f"{_fmt_time(start_sec)} - {_fmt_time(end_sec)} ({chunk_dur:.1f}s)")

        if prev_chunk_end_sec is not None:
            gap_sec = start_sec - prev_chunk_end_sec
            if gap_sec > memory_reset_gap:
                print(
                    "[INFO] Resetting long-form memory before chunk "
                    f"{idx+1} due to speech gap of {gap_sec:.2f}s"
                )
                memory_state = None
            elif memory_state is not None:
                # B1: Content-based reset. A small time gap does not guarantee
                # continuity — speaker changes and hard topic cuts can still
                # happen. We therefore run a cheap pre-check on the current
                # chunk's opening acoustics before deciding whether to preserve
                # the incoming memory.
                preview_wav, _ = _slice_waveform_interval(
                    waveform=waveform,
                    sample_rate=sample_rate,
                    start_sec=start_sec,
                    end_sec=end_sec,
                )
                preview_input, preview_mask = preprocess_audio(
                    preview_wav, device, dtype,
                    pad_to_multiple_of=InferenceConfig.pad_to_multiple_of,
                )
                with torch.inference_mode():
                    preview_encoder, _ = model.encode(preview_input, preview_mask)
                    preview_projected = model.model.decoder.project_encoder_output(preview_encoder)
                preview_similarity = _memory_similarity(memory_state, preview_projected)
                free_memory(device)
                if (
                    preview_similarity is not None
                    and preview_similarity < _MOONSHINE_MEMORY_CONTENT_RESET_THRESHOLD
                ):
                    print(
                        "[INFO] Resetting long-form memory before chunk "
                        f"{idx+1} due to low acoustic continuity ({preview_similarity:.3f})"
                    )
                    memory_state = None

        chunk_segments, memory_state = _transcribe_interval(
            start_sec=start_sec,
            end_sec=end_sec,
            incoming_memory=memory_state,
        )
        segments.extend(chunk_segments)
        prev_chunk_end_sec = end_sec

    full_text = stitch_segments_with_confidence(segments)
    return {"full_text": full_text, "segments": segments}


def _fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS for display."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Print long result
# ---------------------------------------------------------------------------

def print_long_result(result: dict) -> None:
    """Print timestamped segments followed by the full transcription."""
    print()
    for seg in result["segments"]:
        start = _fmt_time(seg["start"])
        end   = _fmt_time(seg["end"])
        print(f"[{start} - {end}] {seg['text']}")

    print()
    print("=" * 60)
    print("Full transcription:")
    print(result["full_text"])
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Moonshine Streaming ASR — Native PyTorch")
    parser.add_argument("--audio",     nargs="+", required=True,
                        help="Path to audio file(s) (.wav, .flac, .mp3)")
    parser.add_argument("--weights",   default=InferenceConfig.weights_path,
                        help=f"Path to .pth weights (default: {InferenceConfig.weights_path})")
    parser.add_argument("--tokenizer", default=InferenceConfig.tokenizer_path,
                        help=f"Path to tokenizer.json (default: {InferenceConfig.tokenizer_path})")
    args = parser.parse_args()

    for name, path in [("weights", args.weights), ("tokenizer", args.tokenizer)]:
        if not Path(path).exists():
            print(f"[ERROR] {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"[INFO] Weights  : {args.weights}")
    print(f"[INFO] Tokenizer: {args.tokenizer}")

    for audio_path in args.audio:
        if not Path(audio_path).exists():
            print(f"[ERROR] audio file not found: {audio_path}", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"Audio: {audio_path}")
        print('='*60)
        try:
            result = transcribe_long(
                audio_path=audio_path,
                weights_path=args.weights,
                tokenizer_path=args.tokenizer,
            )
            print_long_result(result)
        except Exception as e:
            print(f"[ERROR] Failed to process {audio_path}: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
