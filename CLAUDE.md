# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands are run from the project root. The venv is at `venv/`.

```bash
# Activate venv (Windows)
venv\Scripts\activate

# Run Moonshine inference (long or short audio — auto-detected)
venv\Scripts\python.exe src\moonshine_streaming\inference.py --audio <file.wav>

# Download weights from HuggingFace + convert to .pth in one step
venv\Scripts\python.exe src\moonshine_streaming\convert.py --download

# Convert only (already have model.safetensors locally)
venv\Scripts\python.exe src\moonshine_streaming\convert.py --input storage\moonshine\model.safetensors

# Enable debug output (tensor shapes, memory gate scores, budget decisions)
set MOONSHINE_DEBUG=1   # Windows
# export MOONSHINE_DEBUG=1  # Linux/macOS

# WER/CER evaluation against a reference transcript
venv\Scripts\python.exe eval_wer.py hypothesis.txt reference.txt

# Verify a file imports without error (quick syntax check)
venv\Scripts\python.exe -c "import src.moonshine_streaming.model"
```

There are no test suites, linters, or build steps configured in this repository.

## Architecture

### Primary focus

The active codebase is `src/moonshine_streaming/`.

### Module responsibilities

**`src/moonshine_streaming/config.py`**
All model hyperparameters (encoder/decoder dims, sliding window sizes, RoPE config, special token IDs). Also contains device selection (`get_device`, `get_dtype`) and debug helpers (`dbg`, `dbg_tensor`). Debug output is gated on `MOONSHINE_DEBUG` env var, **not** on the module-level `DEBUG = False` constant — that constant is unused at runtime.

**`src/moonshine_streaming/model.py`**
Full native PyTorch encoder-decoder. The most important non-obvious design decisions live here:
- The encoder has **no positional embeddings**. Position is injected into encoder *output* via `decoder.pos_emb` before cross-attention. This keeps the encoder position-invariant and streaming-safe.
- The **long-form memory system** (see below) is entirely implemented here: `MemorySlot`, `LongFormMemoryState`, `_build_slot`, `_build_longform_memory_state`, `_memory_similarity`, `_merge_longform_memory`.
- `generate()` returns 1–3 values depending on `return_memory_state` and `return_token_confidences` flags. Callers must unpack all four variants.
- `_is_repetition_loop` is an early-stop (fires after argmax, before append), not a logit constraint. This distinction matters — blocking the token instead of stopping leads to different loops.

**`src/moonshine_streaming/inference.py`**
The runtime pipeline. Key internal functions:
- `_transcribe_controlled()` — one chunk: primary decode → selective retry → selective rescue. Confidence is cleared (`token_confidences = []`) when retry or rescue wins, so downstream stitching falls back to exact-match.
- `_transcribe_interval()` — wraps controlled decode and adds recursive re-splitting when the result is still unhealthy.
- The chunk loop in `transcribe_long()` runs two independent memory reset checks before each chunk: time-gap reset (>1.5 s gap) and content-based reset (acoustic similarity probe < 0.13).

**`src/longform.py`**
Shared helpers used by the Moonshine pipeline. Contains: `LongFormTuning` (per-model quality knobs), `TranscriptAssessment`, `assess_transcript`, `choose_better_transcript`, `estimate_decode_budget`, `stitch_segments_text`, `stitch_segments_with_confidence`. No model-specific logic here.

**`src/vad.py`**
Silero VAD loading (local `.jit` first, downloads from GitHub if absent), `run_vad`, `smart_chunk`, `derive_chunk_params`.

**`src/moonshine_streaming/convert.py`**
Three entry points: `download()` (HuggingFace → `storage/moonshine/` + `storage/.cache/`), `convert()` (safetensors → .pth), `download_and_convert()`. CLI exposes `--download` and `--input`.

### The long-form memory system

This is the central non-trivial design in the codebase. Understanding it is required before modifying `model.py` or `inference.py`.

**Structure:** `LongFormMemoryState` holds up to 3 `MemorySlot`s, ordered newest→oldest. Each slot has `projected_tokens` (prepended to decoder cross-attention) and `anchor_embedding` (used only for gating, never fed into the model). Total budget is always ≤ 12 tokens regardless of audio length.

**Per-slot token selection (dual-window):** each slot combines peak tokens (highest L2-norm from the full chunk — acoustic salience) and tail tokens (average-pooled from the last ~48 encoder tokens — boundary continuity). The anchor is the mean of the combined set, L2-normalized, in projected space. The anchor and stored tokens are **always in the same projected space** — mixing raw encoder space with projected space was a previous bug.

**Recency decay on demotion:** after each chunk, slot 0 becomes slot 1 (budget 6→4) and slot 1 becomes slot 2 (budget 4→2). Old slot 2 is dropped. Compression on demotion re-runs peak selection on the existing tokens.

**Gate (`_memory_similarity`):** compares the new chunk's opening region (first 48 projected encoder tokens) against `slots[0].anchor_embedding` at three probe positions (first/mid/last third). The **minimum** of the three cosine similarities is the gate score. Threshold 0.35 → merge; below → discard. Using min (not mean) prevents false merges where only part of the boundary looks similar.

**Two-tier pipeline reset (in `inference.py`):** separate from the model-level gate. Time-gap reset (>1.5 s) clears memory before the model even runs. Content-based reset (similarity < 0.13) also clears memory but runs a cheap encoder+projection pass on the incoming chunk first. The 0.13 threshold is intentionally much lower than 0.35 — it only resets on genuine discontinuities, not natural pauses.

### Paths and storage layout

```
storage/
├── .cache/             HuggingFace blob cache (created by convert.py --download)
├── moonshine/
│   ├── model.pth       native weights — what inference.py loads
│   ├── model.safetensors
│   └── tokenizer.json
└── silero_vad.jit      auto-downloaded by vad.py if absent
```

All `storage/` contents are gitignored. Default paths in `InferenceConfig` (in `config.py`) point to `storage/moonshine/model.pth` and `storage/moonshine/tokenizer.json`, resolved relative to `config.py`'s own location.

### Key constraints that must not be violated

1. **Decoder KV cache is never carried across chunk boundaries.** `generate()` always starts fresh. Only the encoder-side `LongFormMemoryState` crosses chunk boundaries.
2. **Memory total is O(1) with respect to audio length.** The 12-token budget ceiling must not grow with duration.
3. **No hardcoded vocabulary.** All lexical decisions come from model weights. No term correction lists.
4. **All improvements are inference-only.** No fine-tuning, no new learned parameters, no training.
