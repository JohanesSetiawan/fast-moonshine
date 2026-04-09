# Moonshine Streaming ASR — Native PyTorch

A native PyTorch implementation of the **Moonshine Streaming Tiny** ASR model with a purpose-built long-form transcription pipeline. No Hugging Face runtime dependency. No training required. All long-form quality improvements are inference-only.

> **Scope:** This README covers only the Moonshine Streaming pipeline under `src/moonshine_streaming/`.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Repository Structure](#2-repository-structure)
3. [Model Architecture](#3-model-architecture)
4. [Inference Flow](#4-inference-flow)
5. [Why Long-Form Transcription Is Hard](#5-why-long-form-transcription-is-hard)
6. [The Long-Form Memory System](#6-the-long-form-memory-system)
7. [Quality Strategies Stack](#7-quality-strategies-stack)
8. [Known Issues and Limitations](#8-known-issues-and-limitations)
9. [Requirements and Installation](#9-requirements-and-installation)
10. [Usage](#10-usage)
11. [Performance Metrics](#11-performance-metrics)
12. [Engineering Notes](#12-engineering-notes)

---

## 1. Overview

### What this is

A self-contained Moonshine Streaming Tiny ASR implementation that can transcribe:

- Short audio clips (seconds) → single forward pass
- Long audio (minutes to hours) → VAD-based chunking pipeline with cross-chunk acoustic memory

The entire implementation is in plain PyTorch. No `transformers`, no `pipeline()`, no cloud API.

### What this is NOT

- **Not a production service.** This is an experimental/research codebase.
- **Not a Hugging Face wrapper.** The model is reimplemented from scratch so the internals are fully visible and controllable.
- **Not streaming output.** Despite the model name, this codebase processes complete audio files. The model *architecture* is streaming-capable (causal encoder), but the pipeline here is batch-mode.
- **Not a multi-model system.** Only Moonshine is used. No secondary ASR, no language model re-ranking, no external correction.

### Primary objective

> Push Moonshine as far as possible for long-form English transcription — preserving low computational cost and maximizing stability and accuracy — without adding new ASR models, without training, and without fine-tuning.

---

## 2. Repository Structure

```
asr-pytorch-exp/
│
├── src/
│   │
│   ├── moonshine_streaming/            # All Moonshine code lives here
│   │   ├── config.py                   # Model config, device selection, debug helpers
│   │   ├── model.py                    # Full native PyTorch encoder-decoder + memory
│   │   ├── inference.py                # Long-form & short-form inference pipeline
│   │   ├── convert.py                  # Weight conversion: HuggingFace .safetensors → .pth
│   │
│   ├── longform.py                     # Shared long-form helpers (budget, health checks, stitching)
│   └── vad.py                          # Silero VAD + smart chunking (shared with s2t/)
│
├── storage/
│   ├── .cache/                         # HuggingFace blob cache (set by convert.py --download)
│   ├── moonshine/                      # Model files — populated by convert.py --download
│   │   ├── model.safetensors           # Original HuggingFace weights (downloaded)
│   │   ├── model.pth                   # Converted weights (native format, used at runtime)
│   │   ├── tokenizer.json              # BPE tokenizer (32768 vocab)
│   │   └── tokenizer_config.json
│   │
│   ├── audios/                         # Test audio files
│   └── silero_vad.jit                  # Silero VAD TorchScript (auto-downloaded if absent)
│
├── requirements.txt
└── README.md
```

---

## 3. Model Architecture

### 3.1 Overview

Moonshine Streaming Tiny is a compact encoder-decoder model (~26M parameters). The encoder is streaming-friendly; the decoder is standard autoregressive with KV caching.

```
MoonshineStreamingForConditionalGeneration
│
├── model : MoonshineStreamingModel
│   │
│   ├── encoder : MoonshineStreamingEncoder
│   │   │
│   │   ├── embedder : MoonshineStreamingEncoderEmbedder
│   │   │   ├── FrameCMVN           per-frame cepstral mean/variance normalization
│   │   │   ├── AsinhCompression    dynamic range compression (arcsinh)
│   │   │   ├── CausalConv1d        stride=2  ─┐ 4× total
│   │   │   ├── CausalConv1d        stride=2  ─┘ downsampling
│   │   │   └── nn.Linear           frame projection: 80 → 320
│   │   │
│   │   ├── EncoderLayer × 6
│   │   │   ├── LayerNorm           (gamma-based, unit_offset init for training stability)
│   │   │   ├── SlidingWindowAttn   local window attention, NO positional embedding
│   │   │   └── MLP                 fc1 → GELU → fc2
│   │   │
│   │   └── LayerNorm               final encoder norm
│   │
│   └── decoder : MoonshineStreamingDecoder
│       │
│       ├── nn.Embedding            token embeddings: 32768 × 320
│       ├── RotaryEmbedding         partial RoPE (factor=0.8, interleaved format)
│       ├── pos_emb                 positional adapter injected into encoder output
│       ├── proj                    encoder projection (Identity for Tiny)
│       │
│       ├── DecoderLayer × 6
│       │   ├── self_attn           causal, RoPE, KV cache grows one step at a time
│       │   ├── encoder_attn        cross-attention to projected encoder, cache fixed at step 0
│       │   └── mlp                 gated SiLU/GLU: fc1 → split → silu(gate) × content → fc2
│       │
│       └── LayerNorm               final decoder norm (no bias)
│
└── proj_out : nn.Linear(320 → 32768, bias=False)     LM head
```

### 3.2 Key design properties

| Property | Value / Behavior |
|---|---|
| Input sample rate | 16 kHz mono |
| Frame size | 80 samples = 5 ms |
| Conv downsampling | 4× → 50 encoder tokens/second |
| Encoder positional embedding | **None inside the encoder stack.** Position is injected into the encoder *output* via `decoder.pos_emb` before cross-attention. This makes the encoder position-invariant and streaming-robust. |
| Decoder positional embedding | Partial RoPE: 32 of 40 head dimensions are rotated (factor=0.8). |
| Sliding window attention | Encoder uses local windows per layer. Layers 0,1,4,5: (16 left, 4 right) = 80ms lookahead. Layers 2,3: (16, 0) = fully causal. |
| Decoder GQA | Tiny uses 8 KV heads, 8 query heads (n_rep=1, no expansion needed). |
| Decode mode | Greedy by default. Beam search available via `beam_width` parameter. |
| Max decoder positions | 4096 tokens → ~630 seconds at ~6.5 tok/s before positional table is exhausted. |

### 3.3 KV Cache

The `KVCache` stores two independent caches per decoder layer:

```
per decoder layer:

    self_cache[layer]
        K : [B, H_kv, T_dec_so_far, head_dim]    grows by 1 each step
        V : [B, H_kv, T_dec_so_far, head_dim]    grows by 1 each step

    cross_cache[layer]
        K : [B, H_kv, T_enc, head_dim]            computed once at step 0, reused
        V : [B, H_kv, T_enc, head_dim]            computed once at step 0, reused
```

**Critical design decision:** The decoder KV cache is **never** carried across chunk boundaries. A fresh cache is built for every chunk. This prevents compounding errors — one bad decode cannot contaminate the next chunk via the self-attention history.

---

## 4. Inference Flow

### 4.1 Short audio (single chunk, no VAD)

```
audio file
    │
    ▼  load_audio()
    │  soundfile + polyphase resample to 16 kHz
    │
    ▼  preprocess_audio()
    │  pad to multiple of 80 samples, build attention mask [1, T]
    │
    ▼  model.generate()
    │  ├── encode(): encoder forward → [B, T_enc, 320]
    │  ├── project_encoder_output(): inject position → [B, T_enc, 320]
    │  └── _decode_greedy(): autoregressive BOS → EOS, with KV cache
    │
    ▼  tokenizer.decode()
    │
    ▼  transcript string
```

### 4.2 Long audio step by step

**Step 1 — Model load (once)**
`MoonshineStreamingForConditionalGeneration` is built from `MoonshineStreamingConfig` and weights are loaded from `model.pth`. Shared across all chunks.

**Step 2 — Audio load (once)**
Full waveform is loaded to RAM as mono float32 at 16 kHz. If source rate differs, `scipy.signal.resample_poly` is used (deterministic, no librosa dependency risk).

**Step 3 — Chunk parameter derivation**
`_moonshine_chunk_params()` computes hardware-aware chunk sizes from model config + available memory, then clamps to the empirically safe quality regime: `target=10s`, `max=12s`.

**Step 4 — Voice Activity Detection**
Silero VAD detects speech boundaries. Short noise segments and long silences are filtered. Returns a list of `(start_sec, end_sec)` speech intervals.

**Step 5 — Smart chunking**
`smart_chunk()` merges VAD intervals into inference chunks respecting target/max/min durations. Chunk boundaries follow speech structure, not wall-clock windows.

**Step 6 — Per-chunk inference loop**

For each chunk:
1. **Time-gap reset check:** if the silence gap since the last chunk exceeds 1.5 s, the memory state is cleared.
2. **Content-based reset check:** if memory exists and the time gap is small, a cheap acoustic similarity probe runs on the opening of the new chunk. If similarity < `0.13`, memory is reset even without a long pause (catches speaker changes, abrupt topic jumps).
3. **`_transcribe_controlled()`:** runs primary decode → selective retry → selective rescue. See §6 and §7 for details.
4. **Recursive re-split:** if the chunk is still unhealthy after controlled decode, it is split at the midpoint with overlap and both halves are decoded independently (up to `max_split_depth=3`).
5. **Memory carry:** the outgoing encoder memory from the accepted decode is carried to the next iteration.

**Step 7 — Stitching**
`stitch_segments_with_confidence()` deduplicates overlapping word sequences at boundaries using exact match + per-token confidence tie-breaking.

**Step 8 — Output**
Returns `{"full_text": str, "segments": [{start, end, text, token_confidences}, ...]}`.

---

## 5. Why Long-Form Transcription Is Hard

### 5.1 The naive approach fails

Feed the whole audio into the encoder in one shot.

**What breaks:**
- Encoder sequence length grows proportionally with duration → quadratic attention memory.
- Decoder positional table is capped at 4096 tokens → hard limit at ~630 seconds.
- At ~6.5 decode tokens/sec, a 28-minute file needs ~10,900 tokens. That is 2.7× the table size.

**Result:** hard OOM or silently corrupt output for audio above ~2 minutes.

---

### 5.2 Naive chunking introduces boundary problems

Split audio into fixed-size windows, transcribe each independently.

**New problems:**

| Problem | Description |
|---|---|
| Boundary artifacts | Words at chunk edges are cut mid-utterance or silently dropped |
| Missing continuity | Each chunk is decoded blind to what came before |
| Repetition loops | Short or ambiguous chunks cause the decoder to hallucinate repetition |
| Over-generation | A generous token budget lets the decoder emit tokens past the end of speech |
| Under-generation | A strict budget cuts off speech that trails into the next chunk |
| Stitching noise | Adjacent chunk texts overlap by a few words → duplicates in final transcript |

---

### 5.3 Why carrying decoder KV cache across chunks is dangerous

It seems natural to seed each chunk with the previous chunk's decoder state. In practice this amplifies errors:

```
chunk N-1 decoder makes one bad word choice
    │
    ▼  KV cache carried to chunk N
    │
    ▼  chunk N self-attention attends to corrupted context
    │
    ▼  chunk N output biased by that error
    │
    ▼  error compounds over the full document → hallucination chain
```

For a small model that already has limited lexical precision, this can produce long runs of confabulated text on difficult audio. A hard reset at every chunk is safer.

**This pipeline never carries decoder KV cache across chunk boundaries.**

---

### 5.4 Why encoder-side memory is safer

Instead of decoder history, a tiny bounded summary of the **previous chunk's projected encoder output** is carried forward. This is acoustically grounded — it represents sound, not words.

An acoustic error does not compound linguistically in the same way a word-level error does. And because the memory bank has a fixed maximum size (12 tokens total), the extra cross-attention cost is O(1) with respect to audio length.

---

## 6. The Long-Form Memory System

This is the most important part of the pipeline. The memory system has been redesigned multiple times; the current version is described in full below.

### 6.1 Overview

```
MEMORY CLASSES IN model.py

MemorySlot                     (frozen dataclass)
    projected_tokens: Tensor   [B, T_slot, D_dec]  — tokens prepended to cross-attention
    anchor_embedding: Tensor   [B, D_dec]           — normalized gate vector

LongFormMemoryState            (frozen dataclass)
    slots: List[MemorySlot]    ordered newest → oldest, up to 3 entries
```

The `LongFormMemoryState` is built at the end of each chunk's decode and optionally merged into the next chunk's encoder context before the decoder cross-attention runs.

---

### 6.2 Memory slot layout (recency decay)

The 12-token budget is split across up to 3 slots, giving older context fewer tokens:

```
slot index   chunk origin   token budget   meaning
──────────   ────────────   ────────────   ───────────────────────────────────────
   0         chunk N-1      6 tokens       most recent, richest, most influential
   1         chunk N-2      4 tokens       two chunks ago, partially compressed
   2         chunk N-3      2 tokens       oldest, most compressed
                            ──────────
                            12 tokens total  ←  same budget as the old single-slot design

Total cross-attention context seen by decoder at chunk N:
  [slot_2 (2 tok)] [slot_1 (4 tok)] [slot_0 (6 tok)] [current chunk (up to ~500 tok)]
   oldest ──────────────────────────────────────────→ current
```

The ordering is oldest-first so that temporal order in the cross-attention sequence is coherent.

---

### 6.3 What each slot contains (dual-window token selection)

Each slot's token budget is split between two sources:

```
PEAK tokens   (33% of budget, rounded up)
    Selected by L2-norm from the ENTIRE chunk's projected encoder output.
    High-norm tokens correspond to strong acoustic events: consonant onsets,
    stressed vowels — the frames most phonetically distinctive.
    Chosen to provide global salience coverage of the chunk.

TAIL tokens   (remaining budget)
    Selected from the temporal TAIL of the chunk (last ~48 encoder tokens ≈ 0.96 s).
    Average-pooled to fit the budget.
    Chosen to provide boundary continuity — the last thing said in the chunk
    is the most likely to overlap acoustically with the opening of the next one.

Combined in a single slot:
    [peak_tokens | tail_tokens]    concatenated along the time dimension
    Anchor = mean(peak + tail), L2-normalized → stored in projected space
```

Example for slot 0 (budget = 6):
- peak_n = ceil(6 × 0.33) = 2 tokens
- tail_n = 6 − 2 = 4 tokens

---

### 6.4 Building memory after each chunk (`_build_longform_memory_state`)

After a chunk is decoded, `_build_longform_memory_state()` is called to produce the outgoing memory:

```
After chunk N decodes:

new_slot = _build_slot(projected_encoder_states, budget=6)
    │
    └── built from chunk N's projected encoder output

previous_state.slots[0] → demoted to slot index 1, budget shrinks 6 → 4
    re-compressed by _select_peak_tokens(old_tokens, new_budget=4)
    new anchor recomputed from compressed tokens

previous_state.slots[1] → demoted to slot index 2, budget shrinks 4 → 2
    re-compressed by _select_peak_tokens(old_tokens, new_budget=2)

previous_state.slots[2] → dropped (exceeds MEMORY_NUM_SLOTS=3)

new LongFormMemoryState = [new_slot, demoted_slot1, demoted_slot2]
```

Re-compression on demotion uses peak selection (highest L2-norm). This retains the most acoustically prominent tokens from the older slot when the budget must shrink.

---

### 6.5 The gate: multi-point similarity (`_memory_similarity`)

Before a chunk's decoder receives any memory, a gate decides whether to merge or discard it:

```
GATE COMPUTATION

Input: LongFormMemoryState, projected encoder states of chunk N

Take first 48 tokens of chunk N's projected output → head window

Divide head window into thirds:
    probe_A = mean(head[0   : T/3])   normalized → "early region"
    probe_B = mean(head[T/2 : T/2 + T/3])  normalized → "mid region"
    probe_C = mean(head[T-T/3 : T])   normalized → "late region"

Compare each probe to slots[0].anchor_embedding via cosine similarity:
    sim_A = cosine(probe_A, anchor)
    sim_B = cosine(probe_B, anchor)
    sim_C = cosine(probe_C, anchor)

Gate score = min(sim_A, sim_B, sim_C)

if gate_score >= 0.35:
    merge memory → prepend all slot tokens to cross-attention context
else:
    discard memory → decode chunk N with no cross-chunk context
```

**Why min of 3 probes, not the mean?**

A mean over the head window can look similar even when only the boundary region overlaps and the rest has diverged (e.g., a speaker changes mid-sentence). By requiring that all three probe points independently meet the threshold, the gate avoids false merges at speaker transitions and abrupt topic boundaries.

**Why gate only against `slots[0]` (the newest slot)?**

The newest slot is the most directly adjacent to the current chunk. Older slots are acoustically further away and are included only if the newest one passes, which means the overall continuity is already confirmed by the strongest signal.

---

### 6.6 Two-tier memory reset at the pipeline level

The gate in `_memory_similarity` (model-level, threshold 0.35) decides whether to *use* memory.

At the pipeline level in `inference.py`, there is an earlier, weaker, two-tier reset:

```
Between chunk N-1 and chunk N:

TIER 1 — Time-based reset (always runs first)
    gap_sec = start_N - end_{N-1}
    if gap_sec > 1.5:
        memory_state = None      ← clear before decoding chunk N
        skip tier 2

TIER 2 — Content-based reset (runs only when gap is small)
    Run encoder on chunk N's opening audio
    Project encoder output to cross-attention space
    compute similarity = _memory_similarity(memory_state, preview_projected)
    if similarity < 0.13:
        memory_state = None      ← clear before decoding chunk N
```

| Threshold | Who checks it | What it asks |
|---|---|---|
| `0.35` (merge gate) | `model.py` / `generate()` | "Is the memory similar enough to *use*?" |
| `0.13` (content reset) | `inference.py` / chunk loop | "Is the memory so *different* that carrying it would actively hurt?" |

The reset threshold (0.13) is intentionally much lower than the merge gate (0.35). It only fires on genuine discontinuities, not on normal within-speaker pauses between sentences.

The value was tuned from 0.18 → 0.13 because the multi-point gate (min of 3 probes) produces lower scores than the old single-mean gate, making 0.18 too aggressive for the new system.

---

### 6.7 Adaptive memory token budget

For short chunks produced by recursive splitting (4–6 s), prepending 12 memory tokens represents a disproportionately large fraction of the cross-attention context and can cause the decoder to over-attend to the memory rather than the current audio.

The budget is scaled down proportionally:

```
memory_budget = clamp( round(chunk_dur × 1.2), min=4, max=12 )

chunk_dur   budget
  10 s   →  12 tokens  (full budget)
   6 s   →   8 tokens
   4 s   →   5 tokens  (floored at 4)
```

The per-slot layout is recomputed for the reduced budget using the same recency-decay ratio (~50% / ~33% / remainder).

---

### 6.8 Memory system — full lifecycle diagram

```
                      ┌──────────────────────────────────┐
                      │           CHUNK  N-1             │
                      └──────────────────────────────────┘
                                       │
                              model.encode()
                                       │
                              [B, T_enc, 320]
                                       │
                        project_encoder_output()
                                       │
                          [B, T_enc, 320]  ← projected space
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
              _select_peak_tokens()          tail window (last 48 tok)
                (highest L2-norm)            adaptive_avg_pool1d → 4 tok
                2 tok (for slot 0)                      │
                       │                                │
                       └───────────────┬────────────────┘
                                       ▼
                              _build_slot(budget=6)
                              projected_tokens: [B, 6, 320]
                              anchor: mean → L2-normalize → [B, 320]
                                       │
                     demote previous slots (compress by peak selection)
                     slot old[0] → slot new[1] (6→4 tok)
                     slot old[1] → slot new[2] (4→2 tok)
                     slot old[2] → dropped
                                       │
                          LongFormMemoryState
                          slots = [new_slot_0, demoted_1, demoted_2]
                                       │
                                       │  (carried to next iteration)
                                       ▼
                      ┌──────────────────────────────────┐
                      │           CHUNK  N               │
                      └──────────────────────────────────┘
                                       │
                      ┌────────────────┴─────────────────┐
                      │      TWO-TIER RESET CHECK         │
                      │                                   │
                      │  gap > 1.5s? → clear memory       │
                      │                   │               │
                      │  otherwise:       │               │
                      │  probe similarity of chunk N head │
                      │  min(3 probes) < 0.13? → clear    │
                      └────────────────┬─────────────────┘
                                       │
                              model.encode()
                                       │
                        project_encoder_output()
                                       │
                        ┌──────────────┴─────────────────┐
                        │     _memory_similarity()        │
                        │                                 │
                        │   probe at T/3, T/2, 2T/3      │
                        │   cosine vs slots[0].anchor     │
                        │   min score ≥ 0.35 ?            │
                        └──────────────┬─────────────────┘
                                       │
                       ┌───────────────┴────────────────┐
                       ▼                                ▼
               score < 0.35                      score ≥ 0.35
            (discard memory)                  _merge_longform_memory()
                       │                                │
                       │                  [slot_2 | slot_1 | slot_0 | current]
                       │                    2 tok   4 tok   6 tok    ~500 tok
                       │                               │
                       └───────────────┬───────────────┘
                                       ▼
                           decoder cross-attention context
                                       │
                              _decode_greedy()
                                       │
                              token IDs → text
```

---

## 7. Quality Strategies Stack

Beyond the memory system, the pipeline applies multiple layers of quality control:

### 7.1 Controlled decode with selective retry

```
_transcribe_controlled(chunk_wav, chunk_dur, incoming_memory, start_sec, end_sec)

1. estimate base_budget = f(chunk_dur)      conservative token budget
   estimate memory_budget = f(chunk_dur)    adaptive memory token count

2. PRIMARY DECODE
   _transcribe_waveform_chunk(
       return_memory_state=True,
       return_token_confidences=True,
   )
   → text, next_memory_state, token_confidences

3. assess_transcript(text, chunk_dur)
   checks: word_count, unique_ratio, looks_truncated, is_too_short, is_degenerate

4. if suspicious:
   EXPANDED RETRY  (same incoming memory, budget += max(8, dur×0.8), cap at 96)
   choose_better_transcript(text, alternate_text)
   if alternate wins → token_confidences = []   (confidence no longer valid)

5. if still truncated/too_short AND right context available:
   RIGHT-CONTEXT RESCUE
   extend audio by up to 1.6 s on the right
   re-decode with adjusted budget
   choose_better_transcript(text, rescue_text)
   if rescue wins → token_confidences = []

6. if final result still degenerate:
   next_memory_state = None   (prevent contamination of downstream chunks)
```

**Why confidence is cleared on retry/rescue win:**
The primary decode path tracks per-token softmax probabilities. Retry and rescue paths do not (they use heuristic comparison). If the primary result loses, its confidence data no longer corresponds to the accepted text, so it is cleared to prevent wrong confidence-weighted stitching decisions.

### 7.2 Repetition loop early-stop

Inside `_decode_greedy()`, after each `argmax` selection:

```
_is_repetition_loop(token_ids, next_id, window=20, max_repeat_count=6)

    recent = token_ids[-20:]
    if recent.count(next_id) + 1 >= 6:
        break          ← stop decode NOW, do not append the token

threshold: 6/20 = 30% repeat rate in a 20-token window
    → only fires on clearly pathological output
    → will not trigger on legitimate repeated words
```

This is an **early-stop**, not a logit constraint. When a small model locks into a repetition loop, blocking the offending token usually just starts a different loop. Stopping early is more reliable.

### 7.3 Confidence-weighted stitching

After all chunks are decoded:

```
stitch_segments_with_confidence(segments)
    │
    for each adjacent pair (left_seg, right_seg):
    │
    ├── find longest common word overlap at left tail / right head
    │
    ├── if both have token_confidences:
    │       left_conf  = mean(left_confidences[-overlap_words * 2:])
    │       right_conf = mean(right_confidences[:overlap_words * 2])
    │
    │       right wins only if right_conf > left_conf + 0.02
    │       otherwise left wins (conservative default)
    │
    └── if either side has no confidence → left always wins (exact match)
```

The 2-tokens-per-word window is a conservative BPE heuristic — most English words are 1–3 tokens, so `overlap_words × 2` covers the boundary region without looking too far into either segment.

### 7.4 Recursive re-splitting

If a chunk is still unhealthy after the controlled decode path:

```
should_split = (
    depth < 3                        max recursion depth
    and duration > 6.0 s             minimum useful chunk size
    and (is_degenerate or is_too_short)
)

if should_split:
    overlap = min(1.25 s, duration × 0.15)
    midpoint = (start + end) / 2
    left  = [start,          midpoint + overlap/2]
    right = [midpoint - overlap/2, end]

    left_segments,  left_memory  = _transcribe_interval(left,  incoming_memory, depth+1)
    right_segments, right_memory = _transcribe_interval(right, left_memory,     depth+1)
```

Left memory is passed into the right child so that boundary continuity is preserved even within the split.

---

## 8. Known Issues and Limitations

### 8.1 Model capability ceiling (cannot be fixed without training)

All remaining errors after pipeline optimization are intrinsic to the Moonshine Tiny weights:

| Error type | Examples from 28-minute test |
|---|---|
| Proper noun instability | `McGivney` → `McGibney` or `McGiffony` |
| Rare word phonetic substitution | `flannel` → `fennel`, `sauerkraut` → `sour crout` |
| Phrase-level acoustic confusion | `stored away` → `saw it away` |
| OOV BPE decoding | Words absent from the BPE training distribution are decoded phonetically |

**The only real fix:** fine-tune on domain-specific audio. All inference-only techniques have been exhausted at the current pipeline maturity level.

---

### 8.2 Proxy metrics, not ground-truth evaluation

WER and CER in this repository are computed against a Gemini-generated reference transcript, not a human-verified ground truth. Relative comparisons between experiment variants are valid; absolute numbers should be interpreted with appropriate caution.

---

### 8.3 No real-time streaming output

The model architecture is streaming-capable (causal encoder, minimal lookahead), but the inference pipeline here loads the full audio before starting. Building a true real-time streaming loop (incremental encode → partial decode → emit text) is not implemented.

---

### 8.4 Full waveform in RAM

The entire waveform is loaded into CPU RAM and kept there for the duration of transcription. GPU/CPU compute memory is freed between chunks (`free_memory(device)`), but the waveform itself stays in RAM.

For a 28-minute file at 16 kHz float32: ~115 MB of RAM for the waveform alone.

---

### 8.5 MP3 read support depends on platform

`soundfile` MP3 support requires libsndfile with MP3 enabled. If it fails:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

---

### 8.6 Single-speaker English narration only (characterized)

All experiments were run on English, single-speaker, audiobook-style narration. Performance on multi-speaker, non-English, noisy, or conversational audio has not been characterized.

---

## 9. Requirements and Installation

### 9.1 System requirements

```
Python  >= 3.11  (developed on 3.14)
CUDA 13.0        recommended (CPU works but is slow on long audio)
```

### 9.2 Python dependencies

```
torch
torchaudio
safetensors
soundfile
numpy
scipy
librosa
tokenizers
psutil
jiwer
```

Install:

```bash
# Clone
git clone <repo-url>
cd asr-pytorch-exp

# Create virtual environment
python -m venv venv

# Activate
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install
pip install -r requirements.txt
```

### 9.3 Getting the model weights

`convert.py` handles both downloading and converting in a single command.

**Option A — Download from HuggingFace and convert automatically (recommended):**

```bash
# Windows
venv\Scripts\python.exe src\moonshine_streaming\convert.py --download

# Linux / macOS
venv/bin/python src/moonshine_streaming/convert.py --download
```

This will:
1. Download `UsefulSensors/moonshine-streaming-tiny` from HuggingFace.
2. Store the HuggingFace blob cache in `storage/.cache/`.
3. Write all model files (safetensors, tokenizer, config) into `storage/moonshine/`.
4. Convert `model.safetensors` → `storage/moonshine/model.pth` automatically.

After this command, the directory layout will be:

```
storage/
├── .cache/              ← HuggingFace blob cache
└── moonshine/
    ├── model.safetensors
    ├── model.pth        ← ready for inference
    ├── tokenizer.json
    └── config.json
```

**Option B — Convert only (already have `model.safetensors` locally):**

```bash
# Windows
venv\Scripts\python.exe src\moonshine_streaming\convert.py \
    --input  storage\moonshine\model.safetensors

# Linux / macOS
venv/bin/python src/moonshine_streaming/convert.py \
    --input  storage/moonshine/model.safetensors
```

**Option C — Custom input and output paths:**

```bash
venv/bin/python src/moonshine_streaming/convert.py \
    --input  path/to/model.safetensors \
    --output path/to/model.pth
```

> **Note:** `--download` requires `huggingface_hub`. Install it with:
> ```bash
> pip install huggingface-hub
> ```

---

## 10. Usage

### 10.1 Command-line inference

The default weights and tokenizer paths are configured in `config.py`. If they exist, you can omit them:

```bash
# Windows — simplest form (defaults apply)
venv\Scripts\python.exe src\moonshine_streaming\inference.py \
    --audio storage\audios\your_audio.wav

# Windows — explicit paths
venv\Scripts\python.exe src\moonshine_streaming\inference.py \
    --audio     storage\audios\your_audio.wav \
    --weights   storage\moonshine\model.pth \
    --tokenizer storage\moonshine\tokenizer.json

# Linux / macOS
venv/bin/python src/moonshine_streaming/inference.py \
    --audio     storage/audios/your_audio.wav \
    --weights   storage/moonshine/model.pth \
    --tokenizer storage/moonshine/tokenizer.json
```

**Multiple files in one call:**

```bash
venv\Scripts\python.exe src\moonshine_streaming\inference.py \
    --audio file1.wav file2.wav file3.mp3
```

### 10.2 What the output looks like

```
[INFO] Device: cuda, dtype: torch.float16
[INFO] Model ready.
[INFO] Audio duration: 1681.2s
[INFO] Chunk params: target=10.0s  max=12.0s  min=6.0s  overlap_fallback=1.2s
[INFO] Long audio detected — loading Silero VAD ...
[INFO] VAD found 599 speech segment(s)
[INFO] Chunked into 156 inference chunk(s)
[INFO] Chunk 1/156: 00:00 - 00:10 (10.4s)
...
[INFO] Resetting long-form memory before chunk 6 due to low acoustic continuity (0.115)
...

[00:00 - 00:10] It was a fine morning in late September...
[00:10 - 00:21] The father turned to look at his son...

============================================================
Full transcription:
It was a fine morning in late September...
============================================================
```

Memory reset lines tell you when the content-based reset fired and what the similarity score was.

### 10.3 Programmatic use

```python
from src.moonshine_streaming.inference import transcribe_long

result = transcribe_long(
    audio_path="path/to/audio.wav",
    weights_path="storage/moonshine/model.pth",
    tokenizer_path="storage/moonshine/tokenizer.json",
)

print(result["full_text"])

for seg in result["segments"]:
    print(f"[{seg['start']:.1f}s – {seg['end']:.1f}s]  {seg['text']}")
    # seg["token_confidences"] is a list[float] of per-token softmax probs
    # (empty list if retry or rescue won for this segment)
```

### 10.4 Short audio only (bypass VAD)

```python
from src.moonshine_streaming.inference import transcribe

text = transcribe(
    audio_path="path/to/short.wav",
    weights_path="storage/moonshine/model.pth",
    tokenizer_path="storage/moonshine/tokenizer.json",
)
```

### 10.5 Enabling debug output

```bash
# Windows
set MOONSHINE_DEBUG=1

# Linux / macOS
export MOONSHINE_DEBUG=1
```

Debug output includes: tensor shapes at each stage, memory state transitions, similarity gate scores, per-chunk token budget decisions, and repetition loop detections.

### 10.6 WER / CER evaluation

```bash
# Hypothesis from a file, reference from stdin
venv\Scripts\python.exe eval_wer.py hypothesis.txt < reference.txt

# Both as files
venv\Scripts\python.exe eval_wer.py hypothesis.txt reference.txt
```

The evaluator normalizes both sides (lowercase, punctuation strip) before computing edit-distance WER and CER.

---

## 11. Performance Metrics

All metrics are proxy measurements against a Gemini-generated reference, not ground-truth.

### Test configuration

| Parameter | Value |
|---|---|
| Audio | `100thestoryofapatriot_10_sinclair_128kb.mp3` |
| Duration | ~28 minutes (1681 s) |
| Content | English audiobook narration (LibriVox), single speaker |
| Reference | Gemini 3.1 Pro transcript |
| Chunks | 156 |
| VAD segments | 599 |

### Result progression

| Configuration | WER | CER |
|---|---|---|
| Initial long-form pipeline (baseline) | 14.83% | 12.30% |
| + shared utilities, VAD fixes, resampling, single-slot memory, shorter chunks, retry/rescue/re-split | 12.66% | 9.91% |
| + multi-slot memory redesign, confidence stitching, repetition early-stop, threshold retuning | **12.04%** | **9.39%** |

### Regression experiments (rolled back)

| Experiment | WER | CER | Why worse |
|---|---|---|---|
| Global right-context + fuzzy stitching | 16.70% | 13.72% | Boundary duplication; fuzzy matching unstable |
| Aggressive single-slot memory (more tokens, lower threshold) | 13.47% | 10.55% | Over-permissive gating → context contamination |
| Beam search + constraints as default path | 13.47% | 10.18% | No quality gain over greedy; added latency |

---

## 12. Engineering Notes

### 12.1 Where to find what

| Question | File → function |
|---|---|
| How does the encoder process audio? | `model.py` → `MoonshineStreamingEncoder.forward()` |
| How does the greedy decode loop work? | `model.py` → `_decode_greedy()` |
| How is the memory slot built from a chunk? | `model.py` → `_build_slot()`, `_build_longform_memory_state()` |
| How is the memory gate computed? | `model.py` → `_memory_similarity()` |
| How is memory prepended to cross-attention? | `model.py` → `_merge_longform_memory()` |
| How is the content-based reset decided? | `inference.py` → chunk loop, `preview_similarity` check |
| How does retry/rescue work? | `inference.py` → `_transcribe_controlled()` |
| How does recursive re-splitting work? | `inference.py` → `_transcribe_interval()` |
| How is the final transcript assembled? | `longform.py` → `stitch_segments_with_confidence()` |
| How are VAD segments merged into chunks? | `vad.py` → `smart_chunk()`, `derive_chunk_params()` |
| Full experiment history and analysis | `src/moonshine_streaming/LONGFORM_TECHNICAL_REPORT.md` |

### 12.2 Code principles

- **No hardcoded words.** The pipeline is general-purpose. All vocabulary decisions come from model weights.
- **No magic numbers without a comment.** Every threshold, budget, and constant has an inline comment explaining why it has that value.
- **No untrained module injection.** A learned layer without training has no alignment with encoder states, decoder states, or the LM head distribution. Inference-only improvements are parameter-free.
- **Bounded memory regardless of audio length.** The memory bank is capped at 12 tokens total. Cross-attention cost is O(1) with respect to audio duration.
- **Spend extra compute only on suspicious chunks.** Retry, rescue, and re-split are selective — triggered by heuristic health checks, not applied globally.

### 12.3 The inference-only ceiling

The best achievable result without training:

> **WER ≈ 12% / CER ≈ 9%** on English audiobook narration (single speaker, clean recording, proxy reference)

All major pipeline-level headroom has been captured. Remaining errors are intrinsic to the Moonshine Tiny weights — proper noun instability, BPE phonetic approximation for OOV words, and acoustic confusion on rare vocabulary.

Further gains require fine-tuning on domain-specific data.
