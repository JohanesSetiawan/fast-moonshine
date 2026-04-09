"""
vad.py — Shared Voice Activity Detection and smart chunking for long audio inference.

Responsibilities:
    1. Load Silero VAD (auto-detect local .jit file, fallback to torch.hub download).
    2. Run VAD on a waveform to get speech segment boundaries.
    3. Derive chunk parameters from model config + runtime hardware (zero hardcoded values).
    4. Merge VAD segments into inference-ready chunks using the derived parameters.

Used by both src/s2t/inference.py and src/moonshine_streaming/inference.py.
"""

import gc
import math
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class ChunkParams:
    """
    Derived runtime parameters for audio chunking.
    All values are in seconds. No value is hardcoded at definition time —
    every field is computed by derive_chunk_params() at runtime.
    """
    target_dur:       float   # aim to fill chunks up to this duration
    max_dur:          float   # hard cap; never exceed this per chunk
    min_dur:          float   # discard chunks shorter than this (e.g., trailing silence)
    overlap_fallback: float   # overlap to keep when no silence gap forces a cut


@dataclass
class Segment:
    """One transcribed chunk with its time boundary in the source audio."""
    start: float   # seconds
    end:   float   # seconds
    text:  str


# ---------------------------------------------------------------------------
# Silero VAD loader
# ---------------------------------------------------------------------------

_SILERO_HUB_REPO  = "snakers4/silero-vad"
_SILERO_HUB_MODEL = "silero_vad"
_SILERO_LOCAL_NAME = "silero_vad.jit"

_SILERO_HUB_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit"
)


def load_silero_vad(storage_dir: str, device: torch.device) -> torch.nn.Module:
    """
    Load the Silero VAD JIT model.

    Resolution order:
        1. <storage_dir>/silero_vad.jit  — existing local file, loaded directly.
        2. torch.hub.load()              — downloads on first call, cached by torch.hub.
           After download, saves a copy to <storage_dir>/silero_vad.jit for future use.

    The model is moved to the given device after loading.

    Args:
        storage_dir: directory to look for / save the JIT file (e.g. project storage/).
        device:      torch.device to place the model on.

    Returns:
        Silero VAD JIT model in eval mode.
    """
    local_path = Path(storage_dir) / _SILERO_LOCAL_NAME

    if local_path.exists():
        print(f"[VAD] Loading Silero VAD from {local_path}")
        vad_model = torch.jit.load(str(local_path), map_location=device)
    else:
        print(f"[VAD] Downloading Silero VAD via torch.hub ...")
        try:
            vad_model, utils = torch.hub.load(
                _SILERO_HUB_REPO,
                _SILERO_HUB_MODEL,
                trust_repo=True,
                verbose=False,
            )
            vad_model = vad_model.to(device)
        except Exception as hub_err:
            # Fallback: direct JIT download (no torch.hub overhead)
            print(f"[VAD] torch.hub failed ({hub_err}), trying direct download ...")
            torch.hub.download_url_to_file(_SILERO_HUB_URL, str(local_path))
            vad_model = torch.jit.load(str(local_path), map_location=device)

        # Persist locally so subsequent runs skip the download
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.jit.save(vad_model, str(local_path))
        print(f"[VAD] Saved to {local_path}")

    vad_model.eval()
    return vad_model


# ---------------------------------------------------------------------------
# Run VAD
# ---------------------------------------------------------------------------

_VAD_SAMPLE_RATE   = 16000          # Silero only supports 16000 and 8000 Hz
_VAD_WINDOW_FRAMES = 512            # required chunk size at 16 kHz
_VAD_THRESHOLD     = 0.5            # speech probability threshold
_VAD_MIN_SPEECH_MS = 250            # merge speech regions shorter than this
_VAD_MIN_SILENCE_MS = 100           # silence shorter than this is bridged
_VAD_PAD_MS        = 30             # padding added to both ends of each speech region


def run_vad(
    vad_model:   torch.nn.Module,
    waveform:    np.ndarray,
    device:      torch.device,
    sample_rate: int = _VAD_SAMPLE_RATE,
) -> List[Tuple[float, float]]:
    """
    Run Silero VAD on a waveform and return speech segment boundaries.

    Args:
        vad_model:   loaded Silero VAD JIT model.
        waveform:    [T_samples] float32 numpy array, range [-1, 1], 16 kHz.
        device:      device the VAD model is on.
        sample_rate: must be 16000 (Silero constraint).

    Returns:
        List of (start_sec, end_sec) tuples for each detected speech region.
        Empty list if no speech detected.

    Note: Silero's built-in get_speech_timestamps() handles windowing, merging,
    and padding internally. We use it here for correctness and simplicity.
    """
    assert sample_rate == _VAD_SAMPLE_RATE, (
        f"Silero VAD requires 16000 Hz, got {sample_rate}"
    )

    # Convert numpy → torch, move to VAD device
    wav_tensor = torch.from_numpy(waveform).float().to(device)

    # Reset VAD model state between calls (stateful LSTM inside)
    vad_model.reset_states()

    # get_speech_timestamps is bundled in utils but we replicate the call
    # directly using the model to avoid torch.hub utils dependency after local load.
    # We compute probabilities per window manually and apply the same logic.
    speech_probs = _compute_speech_probs(vad_model, wav_tensor)

    segments = _probs_to_segments(
        speech_probs,
        sample_rate=sample_rate,
        window_size=_VAD_WINDOW_FRAMES,
        threshold=_VAD_THRESHOLD,
        min_speech_ms=_VAD_MIN_SPEECH_MS,
        min_silence_ms=_VAD_MIN_SILENCE_MS,
        pad_ms=_VAD_PAD_MS,
        audio_len=len(waveform),
    )
    return segments


def _compute_speech_probs(
    vad_model: torch.nn.Module,
    wav:       torch.Tensor,
) -> List[float]:
    """
    Run the VAD model over the waveform in 512-sample windows.
    Returns a list of speech probabilities, one per window.
    """
    probs = []
    total = len(wav)
    with torch.no_grad():
        for start in range(0, total, _VAD_WINDOW_FRAMES):
            chunk = wav[start : start + _VAD_WINDOW_FRAMES]
            # Pad last window if shorter than required
            if len(chunk) < _VAD_WINDOW_FRAMES:
                chunk = torch.nn.functional.pad(chunk, (0, _VAD_WINDOW_FRAMES - len(chunk)))
            prob = vad_model(chunk, _VAD_SAMPLE_RATE).item()
            probs.append(prob)
    return probs


def _probs_to_segments(
    probs:         List[float],
    sample_rate:   int,
    window_size:   int,
    threshold:     float,
    min_speech_ms: int,
    min_silence_ms: int,
    pad_ms:        int,
    audio_len:     int,
) -> List[Tuple[float, float]]:
    """
    Convert per-window speech probabilities into merged speech segments.

    Steps:
        1. Threshold probabilities into a binary speech/silence sequence.
        2. Bridge short silence gaps (< min_silence_ms).
        3. Discard short speech regions (< min_speech_ms).
        4. Add padding on both ends of each region.
        5. Clip to [0, audio_len] and convert to seconds.
    """
    win_dur_ms   = (window_size / sample_rate) * 1000.0
    min_speech_w = max(1, math.ceil(min_speech_ms  / win_dur_ms))
    min_silence_w = max(1, math.ceil(min_silence_ms / win_dur_ms))
    pad_windows  = max(0, round(pad_ms / win_dur_ms))

    # Step 1: threshold
    is_speech = [p >= threshold for p in probs]

    # Step 2: bridge short silence gaps
    i = 0
    while i < len(is_speech):
        if not is_speech[i]:
            j = i
            while j < len(is_speech) and not is_speech[j]:
                j += 1
            silence_len = j - i
            if silence_len < min_silence_w:
                for k in range(i, j):
                    is_speech[k] = True
            i = j
        else:
            i += 1

    # Step 3: collect runs and discard short speech regions
    raw_segments: List[Tuple[int, int]] = []  # (start_window, end_window) inclusive
    i = 0
    while i < len(is_speech):
        if is_speech[i]:
            j = i
            while j < len(is_speech) and is_speech[j]:
                j += 1
            speech_len = j - i
            if speech_len >= min_speech_w:
                raw_segments.append((i, j - 1))
            i = j
        else:
            i += 1

    # Step 4 & 5: add padding and convert to seconds
    max_window = len(probs) - 1
    segments: List[Tuple[float, float]] = []
    for seg_start_w, seg_end_w in raw_segments:
        start_w = max(0, seg_start_w - pad_windows)
        end_w   = min(max_window, seg_end_w + pad_windows)

        start_sample = start_w * window_size
        end_sample   = min((end_w + 1) * window_size, audio_len)

        start_sec = start_sample / sample_rate
        end_sec   = end_sample   / sample_rate
        segments.append((start_sec, end_sec))

    return segments


# ---------------------------------------------------------------------------
# Chunk parameter derivation (zero hardcoded values)
# ---------------------------------------------------------------------------

def derive_chunk_params(
    model_max_encoder_seconds: float,
    model_dtype:               torch.dtype,
    device:                    torch.device,
    model_hidden_size:         int,
    model_num_layers:          int,
    model_num_heads:           int,
    safety_factor:             float = 0.70,
) -> ChunkParams:
    """
    Derive audio chunk parameters entirely from model architecture and runtime hardware.
    No hardcoded duration values — everything is computed.

    Derivation logic:
        1. model_max: seconds that fit within the model's positional/architectural limit.
        2. memory_max: seconds that fit within available free memory (VRAM or RAM),
           estimated from the model's attention matrix size per layer.
        3. safe_max = min(model_max, memory_max) * safety_factor
        4. target_dur   = safe_max * 0.80   (80% of safe ceiling)
        5. max_dur      = safe_max           (100% of safe ceiling)
        6. overlap_fallback = target_dur * 0.06   (6% of target, for forced cuts)
        7. min_dur      = target_dur * 0.04        (4% of target, to discard tiny chunks)

    Args:
        model_max_encoder_seconds: max audio duration derivable from model config
                                   (e.g. MAX_SOURCE_POSITIONS * seconds_per_token).
        model_dtype:               torch.float16 or torch.float32.
        device:                    the inference device.
        model_hidden_size:         encoder hidden_size (e.g. 256 for S2T, 320 for Moonshine).
        model_num_layers:          number of encoder layers.
        model_num_heads:           number of encoder attention heads.
        safety_factor:             fraction of theoretical max to treat as safe ceiling.
                                   Default 0.70 (70%) — leaves headroom for activations,
                                   OS overhead, and KV cache.

    Returns:
        ChunkParams with all fields derived from hardware and model constraints.
    """
    # Bytes per element for the model dtype
    bytes_per_elem = 2 if model_dtype == torch.float16 else 4

    # Estimate available memory
    free_bytes = _get_free_memory_bytes(device)

    # Estimate encoder attention memory per second of audio.
    # For S2T: 1 sec audio -> 25 encoder tokens (40ms each, 4x subsampling at 10ms frames).
    # For Moonshine: 1 sec audio -> 50 encoder tokens (20ms each, 4x at 5ms frames).
    # We use a conservative estimate: tokens_per_sec = 50 (worst case).
    # Attention matrix size per layer: T×T × num_heads × bytes_per_elem
    # Total across all layers: num_layers × T² × num_heads × bytes_per_elem
    # We also account for K, V projections per layer: 2 × T × hidden_size × bytes_per_elem
    tokens_per_sec = 50.0  # conservative upper bound; S2T is 25, Moonshine is 50

    def _memory_for_t_seconds(t: float) -> int:
        """Estimate peak GPU/CPU memory in bytes for encoding t seconds of audio."""
        T = int(t * tokens_per_sec)
        attn_bytes = model_num_layers * T * T * model_num_heads * bytes_per_elem
        kv_bytes   = model_num_layers * 2 * T * model_hidden_size * bytes_per_elem
        return attn_bytes + kv_bytes

    # Binary search for the largest t such that memory_for_t_seconds(t) <= free_bytes
    lo, hi = 1.0, float(model_max_encoder_seconds)
    for _ in range(30):  # 30 bisections -> precision ~1e-9
        mid = (lo + hi) / 2.0
        if _memory_for_t_seconds(mid) <= free_bytes:
            lo = mid
        else:
            hi = mid
    memory_max = lo

    # Take the minimum of model architectural limit and memory limit
    raw_max   = min(model_max_encoder_seconds, memory_max)
    safe_max  = raw_max * safety_factor

    # Ensure a minimum viable chunk size (at least 5 seconds)
    safe_max = max(safe_max, 5.0)

    target_dur       = safe_max * 0.80
    max_dur          = safe_max
    overlap_fallback = max(1.0, target_dur * 0.06)
    min_dur          = max(0.5, target_dur * 0.04)

    return ChunkParams(
        target_dur=round(target_dur, 2),
        max_dur=round(max_dur, 2),
        min_dur=round(min_dur, 2),
        overlap_fallback=round(overlap_fallback, 2),
    )


def _get_free_memory_bytes(device: torch.device) -> int:
    """
    Query available memory for the given device.

    Priority:
        CUDA: torch.cuda.mem_get_info() — exact free VRAM from driver.
        CPU:  psutil.virtual_memory().available — available RAM.
        Fallback (no psutil, no CUDA): conservative 2 GB estimate.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        if hasattr(torch.cuda, "mem_get_info"):
            free, _ = torch.cuda.mem_get_info(device)
            return free
        # Older PyTorch: approximate via reserved/allocated
        props = torch.cuda.get_device_properties(device)
        reserved = torch.cuda.memory_reserved(device)
        return max(0, props.total_memory - reserved)

    # CPU or MPS: query system RAM
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass

    # Absolute fallback: assume 2 GB available
    return 2 * 1024 ** 3


# ---------------------------------------------------------------------------
# Smart chunker
# ---------------------------------------------------------------------------

def smart_chunk(
    vad_segments:  List[Tuple[float, float]],
    audio_duration: float,
    params:        ChunkParams,
) -> List[Tuple[float, float]]:
    """
    Merge VAD speech segments into inference-ready chunks.

    Strategy:
        - Accumulate speech segments until the chunk reaches target_dur.
        - Never exceed max_dur; if a single VAD segment exceeds max_dur,
          force-cut it into smaller overlapping pieces.
        - Discard any resulting chunk shorter than min_dur.
        - The final chunk always extends to audio_duration.

    Args:
        vad_segments:   list of (start_sec, end_sec) from run_vad().
        audio_duration: total audio length in seconds.
        params:         ChunkParams from derive_chunk_params().

    Returns:
        List of (start_sec, end_sec) ready for per-chunk inference.
        If no speech detected, returns the full audio as a single chunk.
    """
    if not vad_segments:
        # No speech detected — return full audio as one chunk
        return [(0.0, audio_duration)]

    chunks: List[Tuple[float, float]] = []
    chunk_start: Optional[float] = None
    chunk_end:   Optional[float] = None

    for seg_start, seg_end in vad_segments:
        seg_dur = seg_end - seg_start

        # Handle over-long individual segment via forced splitting
        if seg_dur > params.max_dur:
            # Flush any accumulated chunk first
            if chunk_start is not None and chunk_end is not None:
                _append_chunk(chunks, chunk_start, chunk_end, params.min_dur)
                chunk_start = chunk_end = None

            # Split the long segment into max_dur-sized pieces with overlap
            pos = seg_start
            while pos < seg_end:
                piece_end = min(pos + params.max_dur, seg_end)
                _append_chunk(chunks, pos, piece_end, params.min_dur)

                # Once we have emitted the tail piece, the long segment is done.
                # This explicit break keeps the control flow simple and avoids
                # reprocessing the last piece again as an accumulated chunk.
                if piece_end >= seg_end:
                    break

                # Overlap: the next piece reuses a small slice of audio from the
                # current tail so the decoder does not lose context at a hard cut.
                next_pos = piece_end - params.overlap_fallback
                if next_pos <= pos:
                    # Degenerate overlap settings must not trap the loop.
                    next_pos = piece_end
                pos = next_pos

            # The segment has already been fully emitted as standalone pieces.
            # There is no remaining tail to carry into the normal accumulation
            # path, so we continue with the next VAD region directly.
            continue

        # Normal segment: try to accumulate
        if chunk_start is None:
            chunk_start = seg_start
            chunk_end   = seg_end
        else:
            prospective_dur = seg_end - chunk_start
            if prospective_dur <= params.max_dur:
                # Fits: extend current chunk
                chunk_end = seg_end
            else:
                # Flush current chunk, start a new one from this segment
                _append_chunk(chunks, chunk_start, chunk_end, params.min_dur)
                chunk_start = seg_start
                chunk_end   = seg_end

        # Emit when we've reached or exceeded target_dur
        if (chunk_end - chunk_start) >= params.target_dur:
            _append_chunk(chunks, chunk_start, chunk_end, params.min_dur)
            chunk_start = chunk_end = None

    # Flush remaining accumulation
    if chunk_start is not None and chunk_end is not None:
        _append_chunk(chunks, chunk_start, chunk_end, params.min_dur)

    if not chunks:
        return [(0.0, audio_duration)]

    # Cover any trailing tail without violating the max-duration contract.
    # The previous implementation always stretched the last chunk to the very
    # end of the file, which could silently undo the max_dur cap we just worked
    # to enforce. For quality-sensitive long-form ASR that behaviour is harmful.
    last_start, last_end = chunks[-1]
    if audio_duration > last_end:
        if (audio_duration - last_start) <= params.max_dur:
            chunks[-1] = (last_start, audio_duration)
        else:
            tail_start = max(last_end - params.overlap_fallback, audio_duration - params.max_dur)
            _append_chunk(chunks, tail_start, audio_duration, params.min_dur)

    return chunks


def _append_chunk(
    chunks:      List[Tuple[float, float]],
    start:       float,
    end:         float,
    min_dur:     float,
) -> None:
    """Append a chunk if it meets the minimum duration requirement."""
    if (end - start) >= min_dur:
        chunks.append((start, end))


# ---------------------------------------------------------------------------
# Post-chunk cleanup helpers
# ---------------------------------------------------------------------------

def free_memory(device: torch.device) -> None:
    """
    Release cached memory between chunks to keep the memory profile flat.
    Calls gc.collect() unconditionally; empties CUDA cache when on GPU.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
