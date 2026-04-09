"""
model.py — Moonshine Streaming Tiny ASR (native PyTorch, no transformers).

Architecture overview:
    MoonshineStreamingForConditionalGeneration
    └── MoonshineStreamingModel
        ├── MoonshineStreamingEncoder
        │   └── MoonshineStreamingEncoderEmbedder    (audio frontend)
        │       ├── MoonshineStreamingFrameCMVN       (per-frame normalization)
        │       ├── MoonshineStreamingAsinhCompression (dynamic range)
        │       ├── MoonshineStreamingCausalConv1d x2 (4x temporal downsampling)
        │       └── nn.Linear                         (frame projection 80→320)
        │   ├── MoonshineStreamingEncoderLayer x6     (sliding window transformer)
        │   │   ├── MoonshineStreamingLayerNorm        (custom, with unit_offset)
        │   │   ├── MoonshineStreamingEncoderAttention (no RoPE, no causal mask)
        │   │   └── MoonshineStreamingEncoderMLP       (fc1→GELU→fc2)
        │   └── MoonshineStreamingLayerNorm            (final encoder norm)
        └── MoonshineStreamingDecoder
            ├── nn.Embedding                          (token embeddings, 32768×320)
            ├── MoonshineStreamingRotaryEmbedding     (partial RoPE, factor=0.8)
            ├── nn.Embedding  (pos_emb)               (positional adapter for encoder)
            ├── nn.Identity   (proj)                  (no-op for Tiny; Linear for larger)
            ├── MoonshineStreamingDecoderLayer x6
            │   ├── self_attn    (causal, with RoPE, KV cache grows per step)
            │   ├── encoder_attn (cross, no RoPE, KV cache fixed after step 0)
            │   └── mlp          (gated SiLU/GLU style)
            └── nn.LayerNorm                          (final decoder norm, no bias)
    └── proj_out: nn.Linear(320, 32768, bias=False)   (LM head)

Key design decisions:
    1. Ergodic encoder: NO positional embeddings anywhere in the encoder.
       Positional information is injected into the ENCODER OUTPUT (not input)
       via `decoder.pos_emb` before cross-attention. This makes the encoder
       position-invariant, enabling robust streaming.

    2. Sliding window attention: Each encoder layer restricts attention to a
       local window [left_window, right_window]. Some layers are fully causal
       (right_window=0), others have limited lookahead (right_window=4 = 80ms).

    3. Custom LayerNorm in encoder: Uses a `gamma` parameter (not `weight`),
       initialized to zero. Effective scale = gamma + 1.0 at init, giving
       training stability without altering initialization magnitude.

    4. Gated MLP in decoder (GLU/SwiGLU style): fc1 outputs 2×intermediate,
       split into content and gate, then silu(gate)×content → fc2. This gives
       better expressiveness than standard FFN at the same parameter count.

    5. Partial RoPE: Only the first `head_dim × partial_rotary_factor = 32` of
       the 40 head dimensions get rotary embedding. The remaining 8 are unrotated.
       This is interleaved format (not split-half), matching HuggingFace weights.

    6. KV Cache (decoder only):
       - Self-attention: K and V grow by one entry per decoding step.
         self_cache[layer] = concat(prev_kv, new_kv) along time dim.
       - Cross-attention: K and V are computed from encoder output at step 0,
         then stored and reused for all subsequent steps (O(enc_len) once).

    7. Audio rate: raw waveform at 16 kHz → frames of 80 samples (5 ms) →
       two stride-2 causal convolutions → 1 encoder token = 320 samples = 20 ms.
       Encoder produces ~50 tokens/second for the attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List

from config import (
    MoonshineStreamingConfig, EncoderConfig, DecoderConfig,
    dbg, dbg_tensor, dbg_layer, DEBUG
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# ---------------------------------------------------------------------------
# Long-form memory
# ---------------------------------------------------------------------------
#
# Core invariants that must never be violated:
#
#   1. NO decoder KV cache across chunks — decoder history is text-conditioned,
#      so one bad chunk would poison all subsequent ones via hallucination chains.
#
#   2. Memory is always encoder-side (acoustic) — grounded in sound, not words.
#      This is much safer: acoustic mistakes don't compound linguistically.
#
#   3. Total cross-attention token budget is O(1) w.r.t. total audio length —
#      critical for low-resource long-form inference.
#
# Memory budget layout (total = MEMORY_BUDGET_TOTAL = 12 tokens):
#
#   The 12 slots are divided across up to MEMORY_NUM_SLOTS=3 recent chunks.
#   Older chunks get fewer tokens (recency decay), inspired by SAM2's
#   temporal memory bank design (num_maskmem=7 with temporal pos encoding).
#   Total cost is the same as before — only the origin of the tokens changes.
#
#   Slot 0 (most recent, chunk N-1): MEMORY_SLOT_SIZES[0] = 6 tokens
#   Slot 1 (chunk N-2):              MEMORY_SLOT_SIZES[1] = 4 tokens
#   Slot 2 (oldest, chunk N-3):      MEMORY_SLOT_SIZES[2] = 2 tokens
#   Sum: 6 + 4 + 2 = 12  ← identical to the old single-slot budget
#
# Within each slot, tokens come from TWO sources (dual-window):
#
#   - TAIL fraction: tokens from the temporal tail of that chunk (continuity).
#   - PEAK fraction: tokens with highest L2-norm in the entire chunk (salience).
#     High-norm encoder tokens correspond acoustically to strong onsets — the
#     frames most likely to encode phonetically distinctive information.
#
# Gate (similarity check) uses THREE probe points (start / mid / end of the
# current chunk's head window) rather than a single mean vector. The minimum
# of the three cosine similarities must exceed the threshold. This prevents
# false merges when only the global mean looks similar but local structure
# diverges — a common failure mode at speaker changes or topic jumps.
#
# Anchor consistency fix (A3):
#   Previously the anchor was computed from raw encoder states while the stored
#   tokens came from projected states — two different representation spaces.
#   Both are now in the same projected space so the gate compares apples to
#   apples.

_MEMORY_BUDGET_TOTAL: int = 12        # total cross-attention tokens for all slots
_MEMORY_NUM_SLOTS:    int = 3         # rolling window depth (SAM2-inspired)
_MEMORY_SLOT_SIZES:   Tuple[int, ...] = (6, 4, 2)  # per-slot budgets, newest first

_MEMORY_TAIL_TOKENS: int = 48         # tail window sampled for slot building (~0.96 s)
_MEMORY_HEAD_TOKENS: int = 48         # head window used for gate comparison (~0.96 s)

# Fraction of each slot's budget drawn from high-salience (peak) tokens;
# remainder comes from the temporal tail for boundary continuity.
_MEMORY_PEAK_FRACTION: float = 0.33   # e.g. slot-0 budget=6 → 2 peak + 4 tail

# Gate: minimum cosine similarity across all three probe points.
# Below this value the memory slot is discarded rather than merged.
_MEMORY_SIMILARITY_THRESHOLD: float = 0.35

# Public alias used by inference.py for content-based reset decisions.
MEMORY_SIMILARITY_THRESHOLD = _MEMORY_SIMILARITY_THRESHOLD


def _memory_slot_sizes(total_budget: int) -> Tuple[int, ...]:
    """
    Derive per-slot budgets for the rolling memory bank.

    Default layout for total_budget=12 is exactly the hand-tuned stable split:
        6 + 4 + 2

    For smaller budgets we keep the same recency-decay spirit without ever
    exceeding the requested total. This lets inference.py shrink memory for
    short chunks while keeping the implementation simple and deterministic.
    """
    total_budget = max(1, int(total_budget))
    if total_budget >= _MEMORY_BUDGET_TOTAL:
        return _MEMORY_SLOT_SIZES

    # Recency-first decay: ~50%, ~33%, remainder.
    slot0 = max(1, round(total_budget * 0.50))
    slot1 = max(0, round(total_budget * 0.33))
    slot2 = max(0, total_budget - slot0 - slot1)

    # Keep ordering newest → oldest and avoid negative / overflow.
    sizes = [slot0, slot1, slot2]
    overflow = sum(sizes) - total_budget
    idx = len(sizes) - 1
    while overflow > 0 and idx >= 0:
        removable = min(overflow, max(0, sizes[idx] - (1 if idx == 0 else 0)))
        sizes[idx] -= removable
        overflow -= removable
        idx -= 1
    return tuple(size for size in sizes if size > 0)


@dataclass(frozen=True)
class MemorySlot:
    """
    One entry in the rolling memory bank — represents one past chunk.

    Fields:
        projected_tokens:
            Compressed encoder tokens in projected (decoder cross-attention)
            space. Already has positional embeddings baked in from the
            previous chunk's `project_encoder_output()` call, so these can
            be prepended to the next chunk's cross-attention context directly.

        anchor_embedding:
            Normalized summary vector derived from the SAME projected tokens.
            Used only as a gate signal — never fed into the model.
            Stored in projected space (A3 fix: consistent with projected_tokens).
    """
    projected_tokens: torch.Tensor   # [B, T_slot, D_dec]
    anchor_embedding: torch.Tensor   # [B, D_dec]  ← projected space (A3 fix)


@dataclass(frozen=True)
class LongFormMemoryState:
    """
    Rolling memory bank: up to MEMORY_NUM_SLOTS recent chunks.

    Replaces the old single-slot design. The decoder now sees tokens from
    multiple past chunks (with recency decay), giving richer temporal context
    at the same total token budget.

    slots[0] = most recent past chunk (largest token allocation)
    slots[1] = two chunks ago
    slots[2] = three chunks ago (smallest allocation)

    The list may have fewer than MEMORY_NUM_SLOTS entries early in a session
    or after a forced reset.
    """
    slots: List[MemorySlot]           # ordered newest → oldest


# ---------------------------------------------------------------------------
# Memory building helpers
# ---------------------------------------------------------------------------

def _select_peak_tokens(
    projected: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """
    Select the n tokens with the highest L2-norm from the projected sequence.

    Why L2-norm as a salience proxy?
        After the encoder + position projection, tokens representing strong
        acoustic events (consonant onsets, stressed vowels) tend to have higher
        activation magnitude. Selecting by norm is parameter-free and cheap —
        just one norm computation and a topk call.

    Args:
        projected: [B, T, D] projected encoder states
        n:         number of tokens to select

    Returns:
        [B, n, D] — the n highest-norm tokens, ordered by their original
        position in the sequence (preserves temporal ordering).
    """
    T = projected.shape[1]
    if T <= n:
        return projected

    # Compute per-token L2 norm: [B, T]
    norms = projected.norm(dim=-1)

    # topk indices (unsorted=False → indices in ascending value order, we want
    # descending norm, so negate). We then re-sort by position so the decoder
    # sees tokens in temporal order — this matters for cross-attention.
    _, top_indices = norms.topk(n, dim=-1, largest=True, sorted=False)  # [B, n]
    top_indices, _ = top_indices.sort(dim=-1)                            # [B, n] positional order

    # Gather selected tokens: [B, n, D]
    idx = top_indices.unsqueeze(-1).expand(-1, -1, projected.shape[-1])
    return projected.gather(dim=1, index=idx)


def _build_slot(
    projected_encoder_states: torch.Tensor,
    slot_budget: int,
    tail_token_count: int = _MEMORY_TAIL_TOKENS,
) -> MemorySlot:
    """
    Build one MemorySlot from the current chunk's projected encoder output.

    Strategy (dual-window, A1+A2):
        - PEAK tokens: top-norm tokens from the entire chunk  → salience
        - TAIL tokens: tokens from the temporal tail          → boundary continuity

    Both sets are in projected space. The anchor is the mean of the combined
    set — also in projected space (A3 fix: no more raw/projected mismatch).

    Args:
        projected_encoder_states: [B, T_enc, D] — already position-projected
        slot_budget:              total tokens for this slot
        tail_token_count:         how many tail tokens to consider before pooling

    Returns:
        MemorySlot ready to be stored in LongFormMemoryState.
    """
    T = projected_encoder_states.shape[1]

    # Split the slot budget between peak and tail.
    # peak_n = ceil(budget * PEAK_FRACTION), at least 1 if budget >= 2.
    peak_n = max(1, round(slot_budget * _MEMORY_PEAK_FRACTION)) if slot_budget >= 2 else 0
    tail_n = slot_budget - peak_n

    tokens_parts: List[torch.Tensor] = []

    # --- PEAK tokens: highest-norm from full chunk ---
    if peak_n > 0:
        peak_tokens = _select_peak_tokens(projected_encoder_states, peak_n)
        tokens_parts.append(peak_tokens)

    # --- TAIL tokens: temporal tail, average-pooled down to tail_n ---
    if tail_n > 0:
        tail_len = min(tail_token_count, T)
        tail_raw = projected_encoder_states[:, -tail_len:, :]   # [B, tail_len, D]
        if tail_raw.shape[1] > tail_n:
            # Average pool: uniform compression within the tail window.
            # Unlike the full-sequence avg pool of the old code, this is
            # applied only to the tail — so it preserves boundary information
            # while still respecting the budget.
            tail_tokens = F.adaptive_avg_pool1d(
                tail_raw.transpose(1, 2),
                output_size=tail_n,
            ).transpose(1, 2)
        else:
            tail_tokens = tail_raw
        tokens_parts.append(tail_tokens)

    # Concatenate peak + tail along the time dimension: [B, slot_budget, D]
    combined = torch.cat(tokens_parts, dim=1) if len(tokens_parts) > 1 else tokens_parts[0]

    # Anchor: mean of combined tokens in projected space (A3 fix).
    anchor = combined.mean(dim=1)                    # [B, D]
    anchor = F.normalize(anchor, dim=-1).detach()

    return MemorySlot(
        projected_tokens=combined.detach(),
        anchor_embedding=anchor,
    )


def _build_longform_memory_state(
    encoder_hidden_states: torch.Tensor,
    projected_encoder_states: torch.Tensor,
    previous_state: Optional["LongFormMemoryState"] = None,
    max_memory_tokens: int = _MEMORY_BUDGET_TOTAL,
    tail_token_count: int = _MEMORY_TAIL_TOKENS,
) -> "LongFormMemoryState":
    """
    Build the new memory state for the CURRENT chunk and return it.

    The new state is constructed BEFORE any memory merge (same contract as
    before): it reflects the current chunk only, so it stays bounded and
    prevents recursive re-packing of old context into future chunks.

    Rolling bank update (A5, SAM2-inspired):
        new slot  = built from current chunk
        old slot 0 → becomes slot 1 (budget shrinks from 6→4)
        old slot 1 → becomes slot 2 (budget shrinks from 4→2)
        old slot 2 → dropped (exceeded MEMORY_NUM_SLOTS)

    Re-compression on demotion:
        When a slot moves from position 0→1, its token budget shrinks.
        We re-select from its existing tokens using L2-norm (peak selection),
        which is cheaper than re-running the full encoder.

    Args:
        encoder_hidden_states:    raw encoder output (used only for legacy compat)
        projected_encoder_states: position-projected encoder output
        previous_state:           the previous LongFormMemoryState (or None)
        max_memory_tokens:        total token budget across all slots (ignored;
                                  layout is determined by _MEMORY_SLOT_SIZES)
        tail_token_count:         tail window size for slot building

    Returns:
        New LongFormMemoryState with slots ordered newest → oldest.
    """
    slot_sizes = _memory_slot_sizes(max_memory_tokens)

    # Build slot for the current chunk using the newest-slot budget.
    new_slot = _build_slot(
        projected_encoder_states=projected_encoder_states,
        slot_budget=slot_sizes[0],
        tail_token_count=tail_token_count,
    )

    # Assemble the demoted slots from the previous state.
    demoted: List[MemorySlot] = []
    if previous_state is not None:
        for old_idx, old_slot in enumerate(previous_state.slots):
            new_idx = old_idx + 1          # slot N-1 becomes slot N
            if new_idx >= len(slot_sizes):
                break                      # oldest slot falls off the bank

            new_budget = slot_sizes[new_idx]
            old_budget = old_slot.projected_tokens.shape[1]

            if old_budget <= new_budget:
                # No compression needed — slot fits in its new, smaller slot.
                # (Shouldn't happen with SLOT_SIZES=[6,4,2] unless budgets were
                #  changed, but defensive is better.)
                demoted.append(old_slot)
            else:
                # Re-compress by selecting the highest-norm tokens.
                # We keep peak selection here for the same reason as in _build_slot:
                # the most acoustically prominent tokens carry the most information
                # when we must discard some.
                compressed_tokens = _select_peak_tokens(old_slot.projected_tokens, new_budget)
                new_anchor = compressed_tokens.mean(dim=1)
                new_anchor = F.normalize(new_anchor, dim=-1).detach()
                demoted.append(MemorySlot(
                    projected_tokens=compressed_tokens.detach(),
                    anchor_embedding=new_anchor,
                ))

    return LongFormMemoryState(slots=[new_slot] + demoted)


# ---------------------------------------------------------------------------
# Memory gate helpers
# ---------------------------------------------------------------------------

def _compute_probe_anchor(
    hidden_states: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor:
    """
    Compute a normalized mean anchor over a slice of encoder hidden states.

    Used to build multiple probe points for the multi-point gate (A4).
    """
    region = hidden_states[:, start:end, :]      # [B, region_len, D]
    anchor = region.mean(dim=1)                   # [B, D]
    return F.normalize(anchor, dim=-1)


def _memory_similarity(
    memory_state: Optional[LongFormMemoryState],
    projected_encoder_states: torch.Tensor,
    head_token_count: int = _MEMORY_HEAD_TOKENS,
) -> Optional[float]:
    """
    Multi-point similarity gate between the most recent memory slot and the
    current chunk's opening (A4).

    Why multi-point instead of a single mean?
        A single mean vector can look similar even when the underlying sequence
        has diverged locally. By probing at three positions (start / mid / end
        of the head window) and taking the MINIMUM similarity, we catch cases
        where only a portion of the boundary is acoustically consistent. This
        prevents false merges at speaker transitions or topic boundaries.

    We gate only against the NEWEST slot (slots[0]) because:
        - It is the most directly adjacent to the current chunk.
        - Older slots are already included if the newest passes the gate.

    Returns:
        Minimum cosine similarity across three probes, or None if no memory.
    """
    if memory_state is None or not memory_state.slots:
        return None

    newest_slot = memory_state.slots[0]
    if newest_slot.projected_tokens.numel() == 0:
        return None

    T_head = min(head_token_count, projected_encoder_states.shape[1])
    if T_head == 0:
        return None

    head = projected_encoder_states[:, :T_head, :]   # [B, T_head, D]
    device = head.device
    dtype  = head.dtype

    # Three probe regions: first third / middle third / last third of head window.
    third = max(1, T_head // 3)
    probes = [
        _compute_probe_anchor(head, 0,               third),           # start
        _compute_probe_anchor(head, T_head // 2,     T_head // 2 + third),  # mid
        _compute_probe_anchor(head, T_head - third,  T_head),          # end
    ]

    ref = newest_slot.anchor_embedding.to(device=device, dtype=dtype)
    ref = F.normalize(ref, dim=-1)

    # Minimum similarity across all three probes — conservative gate.
    similarities = [
        F.cosine_similarity(ref, probe.to(device=device, dtype=dtype), dim=-1).mean().item()
        for probe in probes
    ]
    return min(similarities)


def _merge_longform_memory(
    projected_encoder_states: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor],
    memory_state: LongFormMemoryState,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Prepend all memory bank slots to the current cross-attention context (A5).

    Token layout seen by the decoder cross-attention:
        [slot_2_tokens | slot_1_tokens | slot_0_tokens | current_chunk_tokens]
        (oldest → newest → current)

    Ordering oldest-first keeps temporal order coherent: the decoder reads
    context from furthest-past to current, matching how RoPE encodes time.

    Total prepended tokens = sum of all slot sizes ≤ MEMORY_BUDGET_TOTAL = 12,
    so the extra cross-attention cost is bounded and O(1) w.r.t. audio length.
    """
    device = projected_encoder_states.device
    dtype  = projected_encoder_states.dtype

    # Collect slot tokens oldest → newest (reverse the newest-first list).
    slot_token_list: List[torch.Tensor] = []
    for slot in reversed(memory_state.slots):
        slot_token_list.append(
            slot.projected_tokens.to(device=device, dtype=dtype)
        )

    if not slot_token_list:
        return projected_encoder_states, encoder_attention_mask

    # Concatenate all memory + current: [B, T_mem_total + T_enc, D]
    all_memory = torch.cat(slot_token_list, dim=1)
    merged_states = torch.cat([all_memory, projected_encoder_states], dim=1)

    if encoder_attention_mask is None:
        return merged_states, None

    # All memory positions are valid (no padding in stored slots).
    memory_mask = torch.ones(
        (encoder_attention_mask.shape[0], all_memory.shape[1]),
        device=encoder_attention_mask.device,
        dtype=encoder_attention_mask.dtype,
    )
    merged_mask = torch.cat([memory_mask, encoder_attention_mask], dim=1)
    return merged_states, merged_mask


def _build_blocked_ngram_tokens(token_ids: List[int], ngram_size: int) -> set[int]:
    """
    Return token IDs that would recreate an already seen n-gram suffix.

    This is the classic no-repeat n-gram constraint used in text generation.
    For ASR we keep it small and local. The purpose is not to "stylize" the
    output, but to stop obvious decoder loops such as repeated short clauses.
    """
    if ngram_size <= 1 or len(token_ids) < (ngram_size - 1):
        return set()

    prefix = tuple(token_ids[-(ngram_size - 1):])
    blocked: set[int] = set()
    for idx in range(len(token_ids) - ngram_size + 1):
        ngram = token_ids[idx : idx + ngram_size]
        if tuple(ngram[:-1]) == prefix:
            blocked.add(ngram[-1])
    return blocked


def _apply_decoder_constraints(
    logits: torch.Tensor,
    token_ids: List[int],
    eos_id: int,
    step: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    eos_min_steps: int,
) -> torch.Tensor:
    """
    Apply lightweight inference-only safeguards to decoder logits.

    These constraints are intentionally conservative:
        - repetition penalty nudges the decoder away from reusing the same
          tokens over and over;
        - no-repeat n-gram blocking stops short local loops;
        - EOS gating prevents the model from quitting unrealistically early.

    No weights are changed. Everything here is a decode-time decision only.
    """
    constrained = logits.clone()

    if repetition_penalty > 1.0 and token_ids:
        for token_id in set(token_ids):
            if constrained[token_id] < 0:
                constrained[token_id] *= repetition_penalty
            else:
                constrained[token_id] /= repetition_penalty

    if no_repeat_ngram_size > 1:
        blocked_tokens = _build_blocked_ngram_tokens(token_ids, no_repeat_ngram_size)
        if blocked_tokens:
            blocked_index = torch.tensor(
                sorted(blocked_tokens),
                device=constrained.device,
                dtype=torch.long,
            )
            constrained[blocked_index] = float("-inf")

    if step < eos_min_steps:
        constrained[eos_id] = float("-inf")

    return constrained


def _is_repetition_loop(
    token_ids: List[int],
    token_id: int,
    window: int = 20,
    max_repeat_count: int = 6,
) -> bool:
    """
    Detect whether the decoder has entered a single-token repetition loop.

    This is an EARLY-STOP signal, not a logit-blocking constraint.  It is
    intentionally separate from `_apply_decoder_constraints` because the two
    mechanisms operate at different levels of the decode loop:

        _apply_decoder_constraints → fires BEFORE token selection, nudges or
            blocks specific tokens from the logit vector.

        _is_repetition_loop         → fires AFTER token selection, tells the
            loop to stop entirely if the chosen token has appeared too many
            times recently.

    Why early-stop instead of blocking?
        When a small ASR model locks into a repetition loop, blocking the
        offending token usually just causes the second-best token to be chosen,
        which can start a *different* loop. Stopping early is more reliable: the
        loop is already compromised and further generation will not recover it.

    Conservative defaults:
        window=20, max_repeat_count=6 → a token must fill 30% of the last 20
        positions before triggering a stop. This is clearly pathological for ASR
        (legitimate speech would never repeat one token 30% of the time) and will
        not fire on normal output.

    Args:
        token_ids:        Already generated token history (does NOT yet include
                          `token_id` — the caller appends after this check).
        token_id:         The token just chosen by argmax.
        window:           How many recent history tokens to inspect.
        max_repeat_count: Number of occurrences (including `token_id` itself)
                          that triggers the stop.

    Returns:
        True  → stop decoding now, do NOT append `token_id`.
        False → continue normally.
    """
    recent = token_ids[-window:] if len(token_ids) >= window else token_ids
    # +1 accounts for token_id itself, which is not yet in token_ids.
    return (recent.count(token_id) + 1) >= max_repeat_count


def _length_normalized_score(logprob_sum: float, output_len: int, alpha: float = 0.2) -> float:
    """
    Normalize beam scores so slightly longer candidates are not unfairly punished.

    We keep alpha small because ASR should stay close to the acoustic evidence.
    A larger language-model style length penalty would encourage over-generation.
    """
    denom = max(float(output_len), 1.0) ** alpha
    return logprob_sum / denom

def make_sliding_window_mask(
    seq_len: int,
    left_window: int,
    right_window: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Build an additive attention mask for sliding window encoder attention.

    A query at position q can attend to a key at position k if:
        (q - k) in [0, left_window)   — look left (including self)
        (k - q) in [1, right_window)  — look right (lookahead)

    right_window=0 means fully causal (no lookahead at all).

    Per-layer configuration for Tiny:
        Layers 0, 1, 4, 5: (16, 4) — up to 80 ms lookahead
        Layers 2, 3:        (16, 0) — fully causal

    Returns:
        Tensor of shape [1, 1, seq_len, seq_len] with:
            0.0       where attention is allowed
            -inf      where attention is blocked
        Broadcastable to [batch, heads, seq_len, seq_len].
    """
    # dist[q, k] = q - k; positive means q is to the right of k
    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)  # [T, 1]
    k_pos = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, T]
    dist  = q_pos - k_pos                                       # [T, T]

    # Boolean: True where attention is permitted
    left_ok  = (dist >= 0) & (dist < left_window)
    right_ok = (dist < 0) & (-dist < right_window)   # empty when right_window=0
    can_attend = left_ok | right_ok

    # Convert to additive mask (0.0 = attend, -inf = block)
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    mask[~can_attend] = float("-inf")

    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]


def make_key_padding_mask(
    padding_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Convert a boolean padding mask to an additive attention mask.

    Input:
        padding_mask: [B, T] bool, True = valid token, False = padding

    Returns:
        additive_mask: [B, 1, 1, T] with:
            0.0   for valid tokens
            -inf  for padding tokens

    Note: We use masked_fill instead of (~mask).float() * -inf because
    multiplying 0 by float('inf') gives NaN, which corrupts all attention
    outputs even when all tokens are valid.
    """
    additive = torch.zeros(
        padding_mask.shape,
        device=padding_mask.device,
        dtype=dtype,
    )
    additive = additive.masked_fill(~padding_mask, float("-inf"))
    return additive[:, None, None, :]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand K/V from num_key_value_heads to num_attention_heads (for GQA).

    Input:  [B, num_kv_heads, T, head_dim]
    Output: [B, num_attn_heads, T, head_dim]

    If n_rep == 1 (no GQA, as in Tiny), returns the tensor unchanged.
    """
    if n_rep == 1:
        return hidden_states
    B, num_kv, T, hd = hidden_states.shape
    return (
        hidden_states[:, :, None, :, :]
        .expand(B, num_kv, n_rep, T, hd)
        .reshape(B, num_kv * n_rep, T, hd)
    )


# =============================================================================
# ROTARY POSITION EMBEDDING (RoPE) — decoder self-attention only
# =============================================================================

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims for RoPE using the interleaved format.

    This implementation uses INTERLEAVED pairing (0::2 and 1::2 slices),
    NOT split-half. This MUST match how HuggingFace saved the weights,
    since inv_freq is computed assuming this pairing order.

    For each pair (x[2i], x[2i+1]):
        rotated = (-x[2i+1], x[2i])

    Input/output: [..., dim]
    """
    x1 = x[..., 0::2]   # even indices: 0, 2, 4, ...
    x2 = x[..., 1::2]   # odd  indices: 1, 3, 5, ...
    # Stack along new last dim, then flatten: [..., dim//2] × 2 → [..., dim]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply partial RoPE to query and key tensors.

    With partial_rotary_factor=0.8 and head_dim=40:
        - First 32 dims (40 × 0.8) get rotated
        - Last 8 dims pass through unchanged

    cos/sin shape: [B, 1, T, rotary_dim//2] after unsqueeze (head dim added)
    q/k shape:     [B, num_heads, T, head_dim]

    The rotation formula is: x_rotated = x * cos + rotate_half(x) * sin
    This encodes absolute position as a phase rotation in the frequency domain,
    and has the useful property that q·k depends only on (pos_q - pos_k).
    """
    cos = cos.unsqueeze(1)  # [B, 1, T, rotary_dim//2]
    sin = sin.unsqueeze(1)

    # Interleaved expansion: repeat each cos/sin value twice to align with
    # the interleaved rotate_half. Shape → [B, 1, T, rotary_dim]
    cos = cos[..., :cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., :sin.shape[-1] // 2].repeat_interleave(2, dim=-1)

    rotary_dim = cos.shape[-1]  # 32 for Tiny

    # Separate rotated and pass-through portions
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed = q_rot * cos + rotate_half(q_rot) * sin
    k_embed = k_rot * cos + rotate_half(k_rot) * sin

    return (
        torch.cat([q_embed, q_pass], dim=-1),
        torch.cat([k_embed, k_pass], dim=-1),
    )


class MoonshineStreamingRotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for decoder self-attention.

    Used ONLY in the decoder. The encoder has no positional embeddings at all.

    Configuration for Tiny:
        rope_theta            = 10000
        partial_rotary_factor = 0.8
        head_dim              = 40
        → rotary_dim = int(40 × 0.8) = 32
        → inv_freq shape = [16]  (rotary_dim // 2)

    inv_freq is NOT stored in the checkpoint (persistent=False in HuggingFace),
    so it must be recomputed at init from the formula:
        inv_freq[i] = 1 / (theta ^ (2i / rotary_dim))  for i in [0, rotary_dim//2)

    This gives a geometric progression of frequencies: high-frequency for small i,
    low-frequency for large i. The resulting embeddings allow the model to
    distinguish positions both at fine and coarse granularity.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.head_dim             = config.head_dim
        self.rope_theta           = config.rope_theta
        self.partial_rotary_factor = config.partial_rotary_factor

        rotary_dim = int(self.head_dim * self.partial_rotary_factor)  # 32

        # Compute the inverse frequency vector: shape [rotary_dim // 2] = [16]
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
        )

        # Register as non-persistent buffer: on device, not in state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        dbg("RoPE init", f"head_dim={self.head_dim} rotary_dim={rotary_dim} "
            f"inv_freq.shape={list(inv_freq.shape)}")

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute cos and sin for the given positions.

        Args:
            x:            [B, T, head_dim] — used only for device/dtype reference
            position_ids: [B, T] integer positions

        Returns:
            cos, sin: each [B, T, rotary_dim//2], cast to dtype of x
        """
        # inv_freq: [rotary_dim//2] → [B, rotary_dim//2, 1] for matmul
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )

        # position_ids: [B, T] → [B, 1, T]
        position_ids_expanded = position_ids[:, None, :].float()

        # Outer product via matmul: [B, rotary_dim//2, 1] @ [B, 1, T]
        # → [B, rotary_dim//2, T] → transpose → [B, T, rotary_dim//2]
        with torch.amp.autocast(
            device_type=x.device.type if x.device.type != "mps" else "cpu",
            enabled=False,
        ):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)

        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)

        dbg_tensor("RoPE/cos", cos)
        return cos, sin


# =============================================================================
# KV CACHE
# =============================================================================

class KVCache:
    """
    Key-Value cache for efficient autoregressive decoding.

    Maintains two separate caches per layer:
        self_cache[i]:  Self-attention K/V — grows by one token per step.
                        Shape after t steps: [B, num_kv_heads, t, head_dim]
        cross_cache[i]: Cross-attention K/V — computed ONCE from encoder output
                        at step 0, then reused for all subsequent steps.
                        Shape: [B, num_kv_heads, T_enc, head_dim]

    This reduces decoder complexity from O(t²) to O(t) per step:
        - Without cache: full sequence re-projected every step
        - With cache:    only the new token is projected; history is concatenated

    The cross-attention cache is especially important: without it, the full
    encoder-to-decoder projection (T_enc tokens) would be recomputed every step.
    """

    def __init__(self, num_layers: int):
        self.num_layers  = num_layers
        self.self_cache:  List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers
        self.cross_cache: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * num_layers

    def update_self(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K/V to the self-attention cache and return accumulated K/V.

        First call: stores k_new, v_new directly.
        Subsequent calls: concatenates along time dimension (dim=2).

        Returns the full accumulated (k, v) tensors.
        """
        if self.self_cache[layer_idx] is None:
            self.self_cache[layer_idx] = (k_new, v_new)
        else:
            k_old, v_old = self.self_cache[layer_idx]
            self.self_cache[layer_idx] = (
                torch.cat([k_old, k_new], dim=2),
                torch.cat([v_old, v_new], dim=2),
            )
        return self.self_cache[layer_idx]

    def get_cross(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return the cross-attention cache for a layer, or None if not yet set."""
        return self.cross_cache[layer_idx]

    def set_cross(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """Store the cross-attention K/V (called once at step 0)."""
        self.cross_cache[layer_idx] = (k, v)

    def get_self_seq_len(self) -> int:
        """Return the number of tokens currently cached in self-attention."""
        for c in self.self_cache:
            if c is not None:
                return c[0].shape[2]
        return 0

    def reset(self):
        """Clear all caches (call between utterances)."""
        self.self_cache  = [None] * self.num_layers
        self.cross_cache = [None] * self.num_layers

    def clone(self) -> "KVCache":
        """
        Create a structural copy of the cache for branching decode paths.

        Beam search needs independent cache containers per beam. We only clone
        the lists/tuples, not the underlying tensors, because those tensors are
        treated as immutable history snapshots. This keeps branching cheap.
        """
        cloned = KVCache(self.num_layers)
        cloned.self_cache = [
            None if cache is None else (cache[0], cache[1])
            for cache in self.self_cache
        ]
        cloned.cross_cache = [
            None if cache is None else (cache[0], cache[1])
            for cache in self.cross_cache
        ]
        return cloned


@dataclass
class BeamSearchState:
    """
    One candidate path during small-width beam search.

    Fields:
        token_ids:
            Generated token history excluding EOS, matching the public generate()
            contract used elsewhere in the repository.

        logprob_sum:
            Sum of step log-probabilities for the candidate.

        kv_cache:
            Decoder cache snapshot after consuming the history so far.

        next_input_ids:
            The token to feed on the next decoder step. For unfinished beams
            this is the last chosen token. Finished beams do not use it.

        is_finished:
            Whether the candidate already emitted EOS.
    """

    token_ids: List[int]
    logprob_sum: float
    kv_cache: KVCache
    next_input_ids: Optional[torch.Tensor]
    is_finished: bool


# =============================================================================
# ENCODER COMPONENTS
# =============================================================================

class MoonshineStreamingLayerNorm(nn.Module):
    """
    Custom LayerNorm used in the encoder (NOT the decoder).

    Differs from standard nn.LayerNorm in two ways:
        1. Parameter is named `gamma` (not `weight`), so state dict keys differ.
        2. Effective scale = gamma + unit_offset (default 1.0).
           At initialization, gamma = 0, so scale = 0 + 1 = 1 — same as standard LN.
           This allows the model to learn additive corrections to the identity scale.

    Why not just use nn.LayerNorm?
        - The HuggingFace checkpoint stores these as 'gamma' keys.
        - Loading with nn.LayerNorm would silently skip them (unexpected keys).

    The decoder uses standard nn.LayerNorm with `weight` (no bias), so this
    custom class is ONLY instantiated for encoder components.
    """

    def __init__(self, dim: int, unit_offset: bool = True):
        super().__init__()
        self.unit_offset = float(unit_offset)   # 1.0
        self.ln    = nn.LayerNorm(dim, elementwise_affine=False)
        # Initialized to zeros; effective scale starts at 1.0 due to unit_offset
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln(x)
        return normed * (self.gamma + self.unit_offset)


class MoonshineStreamingFrameCMVN(nn.Module):
    """
    Per-frame Cepstral Mean and Variance Normalization.

    Unlike utterance-level CMVN (which normalizes statistics across the full
    sequence), this normalizes each frame independently:
        centered = x - mean(x, dim=-1)
        output   = centered / (rms(centered) + eps)

    This makes each 5 ms frame have zero mean and unit RMS amplitude,
    regardless of the absolute signal level.

    Input:  [B, num_frames, frame_len]   (frame_len = 80 for 5 ms @ 16 kHz)
    Output: [B, num_frames, frame_len]
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean     = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        rms      = (centered.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        return centered / rms


class MoonshineStreamingAsinhCompression(nn.Module):
    """
    Learnable dynamic range compression using asinh.

    Purpose: reduce the amplitude of loud signals without hard saturation,
    while preserving fine structure in quiet signals. Smoother than log1p
    for negative values.

    Formula: output = asinh(exp(log_k) * x) = asinh(k * x)

    k is the learnable compression strength. Parameterized as log_k to
    ensure k = exp(log_k) > 0 at all times.

    Default init: k_init = 0.75 → log_k ≈ -0.288

    State dict key: 'comp.log_k' (scalar tensor, shape [])
    """

    def __init__(self, k_init: float = 0.75):
        super().__init__()
        self.log_k = nn.Parameter(torch.log(torch.tensor(k_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.asinh(torch.exp(self.log_k) * x)


class MoonshineStreamingCausalConv1d(nn.Conv1d):
    """
    Causal 1D convolution for the audio frontend.

    "Causal" means: output[t] depends only on input[t-kernel_size+1 ... t],
    not on any future input. This is essential for streaming inference.

    Implementation: left-pad by (kernel_size - 1) × dilation samples.
    The standard Conv1d then produces the causal output naturally.

    For Moonshine Tiny:
        conv1: in=320, out=640, kernel=5, stride=2 → left_pad=4
        conv2: in=640, out=320, kernel=5, stride=2 → left_pad=4

    The mask propagation ensures that padded regions are zeroed out
    after convolution, preventing spurious activations.

    State dict: identical to nn.Conv1d (weight[out, in/groups, kernel], bias[out]).
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int = 1,
        dilation:     int = 1,
        bias:         bool = True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size,
            stride=stride, dilation=dilation, bias=bias,
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(
        self,
        x:    torch.Tensor,                   # [B, C, T]
        mask: Optional[torch.Tensor] = None,  # [B, T] bool, True=valid
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x:    [B, C, T] input features
            mask: [B, T] boolean mask (True = valid sample)

        Returns:
            x:    [B, C_out, T_out]  after causal conv (T_out = T // stride)
            mask: [B, T_out] updated mask, or None
        """
        # Apply causal left-padding before the standard convolution
        x = F.pad(x, (self.left_pad, 0))
        x = super().forward(x)

        if mask is not None:
            # Propagate the mask through the conv by applying a ones-kernel
            # to the float mask. This counts how many valid input samples
            # contribute to each output position.
            mask_padded = F.pad(mask.float().unsqueeze(1), (self.left_pad, 0))
            weight_ones  = torch.ones(1, 1, self.kernel_size[0], device=mask.device)
            mask_conv    = F.conv1d(mask_padded, weight_ones, stride=self.stride[0])
            mask_out     = (mask_conv.squeeze(1) > 0)   # [B, T_out]
            x = x * mask_out.unsqueeze(1).to(x.dtype)
            return x, mask_out

        return x, None


class MoonshineStreamingEncoderEmbedder(nn.Module):
    """
    Audio frontend: raw waveform → encoder frame features.

    Full pipeline:
        1. Reshape waveform to frames: [B, audio_len] → [B, num_frames, 80]
           (frame_len = 80 samples = 5 ms at 16 kHz)
        2. Per-frame CMVN: normalize each 5 ms window independently
        3. Asinh compression: learnable dynamic range compression
        4. Linear + SiLU: project frame_len(80) → hidden_size(320)
        5. Transpose to [B, 320, num_frames] for Conv1d
        6. CausalConv1d(stride=2) + SiLU: [B, 320, T] → [B, 640, T//2]
        7. CausalConv1d(stride=2):         [B, 640, T//2] → [B, 320, T//4]
        8. Transpose back: [B, 320, T//4] → [B, T_enc, 320]

    Downsampling: 80 samples/frame × 2 × 2 = 320 samples per encoder token
                  = 20 ms per token at 16 kHz = 50 encoder tokens/second.

    Padding mask is propagated through the conv layers using the mask
    propagation logic in MoonshineStreamingCausalConv1d.forward().
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.frame_len = config.frame_len  # 80 samples

        self.cmvn   = MoonshineStreamingFrameCMVN()
        self.comp   = MoonshineStreamingAsinhCompression()

        # Linear frame projection: 80 → 320, no bias
        self.linear = nn.Linear(self.frame_len, config.hidden_size, bias=False)

        # Two causal convolutions, each with stride 2
        self.conv1 = MoonshineStreamingCausalConv1d(
            config.hidden_size,       # 320
            config.hidden_size * 2,   # 640
            kernel_size=5,
            stride=2,
        )
        self.conv2 = MoonshineStreamingCausalConv1d(
            config.hidden_size * 2,   # 640
            config.hidden_size,       # 320
            kernel_size=5,
            stride=2,
        )

    def forward(
        self,
        input_values: torch.Tensor,                    # [B, audio_len]
        padding_mask: Optional[torch.Tensor] = None,   # [B, audio_len] bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            hidden_states: [B, T_enc, 320]
            padding_mask:  [B, T_enc] bool or None
        """
        B, audio_len = input_values.shape

        # Step 1: Reshape to frames [B, audio_len] → [B, num_frames, 80]
        hidden_states = input_values.reshape(B, -1, self.frame_len)
        dbg_tensor("embedder/frames", hidden_states)

        # Step 2 & 3: Per-frame normalization + compression
        hidden_states = self.cmvn(hidden_states)
        hidden_states = self.comp(hidden_states)
        dbg_tensor("embedder/after_cmvn_comp", hidden_states)

        # Step 4: Linear projection + SiLU: [B, num_frames, 80] → [B, num_frames, 320]
        hidden_states = F.silu(self.linear(hidden_states))
        dbg_tensor("embedder/after_linear", hidden_states)

        # Step 5: Propagate padding mask from audio level to frame level
        if padding_mask is not None:
            # Count valid samples per batch item, convert to frame count
            num_frames = padding_mask.sum(-1) // self.frame_len   # [B]
            T_frames   = hidden_states.shape[1]
            # Build boolean mask at frame level
            padding_mask = (
                torch.arange(T_frames, device=padding_mask.device)[None, :] < num_frames[:, None]
            )   # [B, T_frames]
            # Zero out padding frames
            hidden_states = hidden_states * padding_mask.unsqueeze(-1).to(hidden_states.dtype)

        # Step 6: Transpose for Conv1d: [B, T, 320] → [B, 320, T]
        hidden_states = hidden_states.transpose(1, 2)

        # Step 7: First causal conv + SiLU: [B, 320, T] → [B, 640, T//2]
        hidden_states, padding_mask = self.conv1(hidden_states, padding_mask)
        hidden_states = F.silu(hidden_states)
        dbg_tensor("embedder/after_conv1", hidden_states)

        # Step 8: Second causal conv: [B, 640, T//2] → [B, 320, T//4]
        hidden_states, padding_mask = self.conv2(hidden_states, padding_mask)
        dbg_tensor("embedder/after_conv2", hidden_states)

        # Step 9: Transpose back: [B, 320, T_enc] → [B, T_enc, 320]
        hidden_states = hidden_states.transpose(1, 2)

        return hidden_states, padding_mask


class MoonshineStreamingEncoderMLP(nn.Module):
    """
    Standard two-layer FFN for encoder layers.

    Unlike the gated MLP in the decoder, this is a plain fc1→GELU→fc2.

        fc1: 320 → 1280 (intermediate_size), with bias
        fc2: 1280 → 320, with bias

    Activation: GELU (smooth approximation to ReLU, standard in transformers).

    The decoder uses SiLU-gated MLP; using GELU here in the encoder
    matches the original HuggingFace checkpoint exactly.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)   # 320 → 1280
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)   # 1280 → 320

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class MoonshineStreamingEncoderAttention(nn.Module):
    """
    Multi-head self-attention for encoder layers.

    Key differences from decoder attention:
        - No RoPE: encoder is position-free (ergodic). Positional information
          is injected into the encoder OUTPUT, not the input.
        - Not causal: attention is bidirectional within the sliding window.
          The window restriction is applied externally via attention_mask.
        - No KV cache: the encoder processes the full sequence in one pass.
        - No bias: Q/K/V/O projections all have bias=False.

    The sliding window mask arrives as an additive [1, 1, T, T] tensor from
    MoonshineStreamingEncoder.forward() and is applied inside this module.
    """

    def __init__(self, config: EncoderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx    = layer_idx
        self.num_heads    = config.num_attention_heads       # 8
        self.num_kv_heads = config.num_key_value_heads       # 8 (no GQA in encoder)
        self.head_dim     = config.head_dim                  # 40
        self.n_kv_groups  = self.num_heads // self.num_kv_heads  # 1
        self.scaling      = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,    bias=False)

    def forward(
        self,
        hidden_states:  torch.Tensor,                   # [B, T, 320]
        attention_mask: Optional[torch.Tensor] = None,  # [1, 1, T, T] additive
    ) -> torch.Tensor:
        """
        Returns:
            output: [B, T, 320]
        """
        B, T, _ = hidden_states.shape

        # Project and reshape: [B, T, 320] → [B, H, T, 40]
        q = self.q_proj(hidden_states).view(B, T, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Expand K, V for GQA (no-op for Tiny since num_kv_heads == num_heads)
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q.dtype)

        # Scaled dot-product attention with sliding window mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,  # mask already encodes the causal/window constraint
        )

        # Reshape and project output: [B, H, T, 40] → [B, T, 320]
        attn_out = attn_out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(attn_out)


class MoonshineStreamingEncoderLayer(nn.Module):
    """
    One encoder transformer layer.

    Uses Pre-LayerNorm (same as Whisper, GPT-2, etc.):
        x = x + SelfAttn(LN1(x))
        x = x + MLP(LN2(x))

    Pre-LN is more stable than Post-LN at large depths: gradients flow
    cleanly through the residual branch without vanishing through LN.

    LayerNorm is the custom MoonshineStreamingLayerNorm (key: gamma, not weight).
    """

    def __init__(self, config: EncoderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn               = MoonshineStreamingEncoderAttention(config, layer_idx)
        self.mlp                     = MoonshineStreamingEncoderMLP(config)
        self.input_layernorm         = MoonshineStreamingLayerNorm(config.hidden_size)
        self.post_attention_layernorm = MoonshineStreamingLayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states:  torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention branch with pre-LN and residual
        residual      = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP branch with pre-LN and residual
        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonshineStreamingEncoder(nn.Module):
    """
    Sliding window encoder with ergodic (position-free) design.

    Full pipeline:
        1. Audio frontend (embedder): raw waveform → [B, T_enc, 320]
        2. Build per-layer sliding window masks (from config.sliding_windows)
        3. Pass through 6 encoder layers, each with its own mask
        4. Final LayerNorm

    There are NO positional embeddings in the encoder.
    This is the key property that makes the model "streaming": the encoder's
    representation of a frame doesn't depend on its absolute position in the
    utterance. Position is injected once into the encoder OUTPUT via the
    decoder's pos_emb adapter before cross-attention.

    Sliding window configuration (from config):
        Layers 0, 1, 4, 5: [16, 4]  — 80 ms left + 80 ms lookahead
        Layers 2, 3:        [16, 0]  — 80 ms left only (fully causal)
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config   = config
        self.embedder = MoonshineStreamingEncoderEmbedder(config)
        self.layers   = nn.ModuleList([
            MoonshineStreamingEncoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])
        self.final_norm = MoonshineStreamingLayerNorm(config.hidden_size)

    def forward(
        self,
        input_values:   torch.Tensor,                   # [B, audio_len]
        attention_mask: Optional[torch.Tensor] = None,  # [B, audio_len]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            last_hidden_state: [B, T_enc, 320]
            attention_mask:    [B, T_enc] bool (True=valid), or None
        """
        dbg("encoder.forward", f"input_values shape={list(input_values.shape)}")
        dbg_tensor("encoder/input_values", input_values)

        # Step 1: Audio frontend — waveform → frame features
        hidden_states, attention_mask = self.embedder(input_values, padding_mask=attention_mask)
        dbg_tensor("encoder/after_embedder", hidden_states)
        dbg("encoder/T_enc", str(hidden_states.shape[1]))

        T_enc = hidden_states.shape[1]

        # Step 2: Build per-layer attention masks
        # Each layer has a different sliding window configuration.
        # When no padding, we only need the sliding window mask.
        # When padding exists, we combine it with the key-padding mask.
        per_layer_masks = []
        for layer_idx in range(len(self.layers)):
            left_w, right_w = self.config.sliding_windows[layer_idx]
            sw_mask = make_sliding_window_mask(
                T_enc, left_w, right_w,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if attention_mask is not None:
                # Key-padding mask: block padded key positions in all queries
                key_pad_mask = make_key_padding_mask(attention_mask, sw_mask.dtype)
                sw_mask      = sw_mask + key_pad_mask.to(sw_mask.dtype)
            per_layer_masks.append(sw_mask)

        # Step 3: Forward through encoder layers
        for layer_idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states, per_layer_masks[layer_idx])
            dbg_layer("encoder/layer_out", layer_idx, hidden_states)

        # Step 4: Final LayerNorm
        hidden_states = self.final_norm(hidden_states)
        dbg_tensor("encoder/final_hidden", hidden_states)

        return hidden_states, attention_mask


# =============================================================================
# DECODER COMPONENTS
# =============================================================================

class MoonshineStreamingDecoderMLP(nn.Module):
    """
    Gated MLP (SwiGLU / GLU style) for decoder layers.

    Unlike the encoder's standard fc1→GELU→fc2, this uses a gated formulation:
        [content, gate] = fc1(x).chunk(2, dim=-1)   # split 2560 → 1280 + 1280
        output = fc2(silu(gate) × content)

    This is equivalent to:
        output = down_proj(silu(gate_proj(x)) × up_proj(x))

    The gating mechanism allows the network to selectively suppress or
    amplify information based on the content, providing more expressive
    representations at the same effective parameter count as standard FFN.

    Dimensions for Tiny:
        fc1: 320 → 2560  (intermediate_size × 2)
        fc2: 1280 → 320  (intermediate_size → hidden_size)

    Both layers use bias (default nn.Linear).
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2)  # 320 → 2560
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)       # 1280 → 320

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, T, 320] → [B, T, 2560]
        hidden_states = self.fc1(hidden_states)
        # Split into two halves along the feature dim
        content, gate = hidden_states.chunk(2, dim=-1)   # each [B, T, 1280]
        # Gate with SiLU, then multiply elementwise
        hidden_states = F.silu(gate) * content           # [B, T, 1280]
        # Project back to hidden_size
        hidden_states = self.fc2(hidden_states)          # [B, T, 320]
        return hidden_states


class MoonshineStreamingDecoderAttention(nn.Module):
    """
    Attention module used for both self-attention and cross-attention in the decoder.

    Instantiated twice per decoder layer:
        - self_attn    (is_cross=False): causal self-attention with RoPE and growing KV cache
        - encoder_attn (is_cross=True):  cross-attention from decoder to encoder, no RoPE,
                                         with fixed KV cache (computed once from encoder output)

    KV cache behavior:
        Self-attention:
            At each decoding step, the new K/V for the current token are appended
            to the cache. The full accumulated K/V (all previous + current token)
            are then used in the attention computation. This is O(t) per step.

        Cross-attention:
            At step 0, K and V are computed from encoder_hidden_states and stored.
            From step 1 onward, K and V are retrieved from cache, so the encoder
            projection is never repeated. This saves T_enc operations per step.

    Both projections are bias=False (attention_bias=False in config).
    """

    def __init__(self, config: DecoderConfig, is_cross: bool = False):
        super().__init__()
        self.is_cross     = is_cross
        self.num_heads    = config.num_attention_heads    # 8
        self.num_kv_heads = config.num_key_value_heads    # 8
        self.head_dim     = config.head_dim               # 40
        self.n_kv_groups  = self.num_heads // self.num_kv_heads  # 1

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads    * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size,    bias=False)

    def forward(
        self,
        hidden_states:       torch.Tensor,                              # [B, T_q, 320]
        attention_mask:      Optional[torch.Tensor] = None,            # [B, 1, T_q, T_k] additive
        key_value_states:    Optional[torch.Tensor] = None,            # [B, T_enc, 320] for cross
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # (cos, sin)
        kv_cache:            Optional[KVCache] = None,
        layer_idx:           int = 0,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:       query source [B, T_q, 320]
            attention_mask:      additive mask (0 = attend, -inf = block)
            key_value_states:    encoder output for cross-attention (first step only)
            position_embeddings: (cos, sin) tuple from RotaryEmbedding
            kv_cache:            KVCache object or None
            layer_idx:           which layer's cache to use

        Returns:
            output: [B, T_q, 320]
        """
        B, T_q, _ = hidden_states.shape

        # Query is always projected from hidden_states
        q = self.q_proj(hidden_states).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_cross:
            # ---- Cross-attention: K/V from encoder ----
            # First step (cache miss): compute from encoder output and store.
            # Later steps (cache hit): reuse stored K/V.
            if kv_cache is not None:
                cached = kv_cache.get_cross(layer_idx)
                if cached is not None:
                    k, v = cached   # reuse: O(1)
                else:
                    assert key_value_states is not None, \
                        "key_value_states required for cross-attn cache init"
                    T_enc = key_value_states.shape[1]
                    k = self.k_proj(key_value_states).view(B, T_enc, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v = self.v_proj(key_value_states).view(B, T_enc, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    kv_cache.set_cross(layer_idx, k, v)
            else:
                # No cache: always recompute (used during training/prefill)
                T_enc = key_value_states.shape[1]
                k = self.k_proj(key_value_states).view(B, T_enc, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(key_value_states).view(B, T_enc, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Cross-attention never uses RoPE (encoder output has no position encoding)
            is_causal_flag = False

        else:
            # ---- Self-attention: K/V from current token(s) ----
            k = self.k_proj(hidden_states).view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(hidden_states).view(B, T_q, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Apply partial RoPE to Q and K (first 32 of 40 dims)
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Append to KV cache and return full accumulated K/V
            if kv_cache is not None:
                k, v = kv_cache.update_self(layer_idx, k, v)
                # k/v now span all previous + current tokens

            # is_causal=True only during prefill without attention_mask.
            # During incremental decode (T_q=1), mask is never needed.
            is_causal_flag = (attention_mask is None) and (T_q > 1)

        # GQA expansion (no-op for Tiny: 8/8 = 1)
        k = repeat_kv(k, self.n_kv_groups)
        v = repeat_kv(v, self.n_kv_groups)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=q.dtype)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal_flag,
        )

        # [B, H, T_q, 40] → [B, T_q, 320]
        attn_out = attn_out.transpose(1, 2).reshape(B, T_q, -1)
        return self.o_proj(attn_out)


class MoonshineStreamingDecoderLayer(nn.Module):
    """
    One decoder transformer layer with three sub-blocks.

    Order (Pre-LayerNorm throughout):
        1. Self-attention:   x = x + SelfAttn(LN1(x))   — causal, RoPE, KV cache grows
        2. Cross-attention:  x = x + CrossAttn(LN2(x))  — to encoder, no RoPE, KV cache fixed
        3. MLP:              x = x + MLP(LN3(x))        — gated SiLU

    LayerNorms are standard nn.LayerNorm with bias=False (state dict key: .weight, not .gamma).
    This is the KEY difference from encoder: encoder uses custom LN (key=gamma),
    decoder uses standard LN (key=weight).
    """

    def __init__(self, config: DecoderConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attn    = MoonshineStreamingDecoderAttention(config, is_cross=False)
        self.encoder_attn = MoonshineStreamingDecoderAttention(config, is_cross=True)
        self.mlp          = MoonshineStreamingDecoderMLP(config)

        # Standard LayerNorm, no bias — keys in state dict: '<name>.weight'
        self.input_layernorm          = nn.LayerNorm(config.hidden_size, bias=False)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, bias=False)
        self.final_layernorm          = nn.LayerNorm(config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states:         torch.Tensor,
        self_attn_mask:        Optional[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        encoder_attn_mask:     Optional[torch.Tensor],
        position_embeddings:   Tuple[torch.Tensor, torch.Tensor],
        kv_cache:              Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Sub-block 1: Self-attention
        residual      = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=self_attn_mask,
            position_embeddings=position_embeddings,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )
        hidden_states = residual + hidden_states

        # Sub-block 2: Cross-attention
        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.encoder_attn(
            hidden_states,
            attention_mask=encoder_attn_mask,
            key_value_states=encoder_hidden_states,
            kv_cache=kv_cache,
            layer_idx=self.layer_idx,
        )
        hidden_states = residual + hidden_states

        # Sub-block 3: MLP
        residual      = hidden_states
        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MoonshineStreamingDecoder(nn.Module):
    """
    Autoregressive decoder with cross-attention to encoder output.

    Additional adapter modules (not present in standard transformer decoders):
        pos_emb: Embedding(max_position_embeddings=4096, encoder_hidden_size=320)
            Adds positional information to the encoder output before cross-attention.
            Called once in generate() via project_encoder_output(), not per step.

        proj: Identity (Tiny) or Linear (larger variants)
            Projects encoder_hidden_size → decoder_hidden_size.
            For Tiny, both are 320, so this is a no-op.

    The pos_emb + proj combination is the only place where positional information
    enters the encoder representation. This two-step injection (into the projected
    encoder output) is more efficient than adding positional encodings in the encoder
    itself, because it is done once per utterance rather than once per layer.
    """

    def __init__(self, config: DecoderConfig):
        super().__init__()
        self.config = config

        # Token embedding table: vocab_size × hidden_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        self.layers = nn.ModuleList([
            MoonshineStreamingDecoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

        # Final decoder norm (standard, no bias)
        self.norm = nn.LayerNorm(config.hidden_size, bias=False)

        # Rotary position embedding for self-attention
        self.rotary_emb = MoonshineStreamingRotaryEmbedding(config)

        # Adapter: adds learned positional embedding to encoder output
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.encoder_hidden_size)

        # Adapter: dimension projection (no-op for Tiny)
        if config.encoder_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.encoder_hidden_size, config.hidden_size, bias=False)
        else:
            self.proj = nn.Identity()

    def project_encoder_output(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Inject positional information into encoder output and project dimensions.

        Called ONCE before the generate() loop, not per step.
        The result is passed as `encoder_hidden_states` to all decoder layers.

        Steps:
            1. Create position IDs [0, 1, ..., T_enc-1]
            2. Look up pos_emb for those positions
            3. Add to encoder output: gives each encoder token a position
            4. Optionally project dimension (no-op for Tiny)

        Args:
            encoder_hidden_states: [B, T_enc, 320]

        Returns:
            projected: [B, T_enc, 320] with positional info added
        """
        T_enc   = encoder_hidden_states.shape[1]
        pos_ids = torch.arange(T_enc, device=encoder_hidden_states.device)
        pos_emb = self.pos_emb(pos_ids)   # [T_enc, 320]

        # Add position to encoder output (non-in-place for gradient safety)
        projected = encoder_hidden_states + pos_emb.unsqueeze(0)   # [B, T_enc, 320]
        projected = self.proj(projected)

        dbg_tensor("decoder/projected_encoder", projected)
        return projected

    def forward(
        self,
        input_ids:              torch.Tensor,                   # [B, T_dec]
        encoder_hidden_states:  torch.Tensor,                   # [B, T_enc, 320] (already projected)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # [B, T_enc]
        past_seq_len:           int = 0,                        # tokens already in KV cache
        kv_cache:               Optional[KVCache] = None,
    ) -> torch.Tensor:
        """
        One forward pass of the decoder.

        Args:
            input_ids:              [B, T_dec] token IDs (T_dec=1 in incremental mode)
            encoder_hidden_states:  [B, T_enc, 320] already projected via project_encoder_output()
            encoder_attention_mask: [B, T_enc] True=valid (for cross-attention key masking)
            past_seq_len:           number of tokens already decoded (for correct RoPE positions)
            kv_cache:               KVCache object or None

        Returns:
            hidden_states: [B, T_dec, hidden_size] — feed to proj_out for logits
        """
        B, T_dec = input_ids.shape

        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)   # [B, T_dec, 320]
        dbg_tensor("decoder/embed_tokens", hidden_states)

        # RoPE position IDs: offset by past_seq_len for correct absolute positions
        position_ids = torch.arange(
            past_seq_len, past_seq_len + T_dec,
            device=input_ids.device,
        ).unsqueeze(0).expand(B, -1)   # [B, T_dec]

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        # Causal self-attention mask.
        # During incremental decode (T_dec=1 with cache): no mask needed.
        # During prefill (T_dec > 1, no cache): build upper-triangular causal mask.
        self_attn_mask = None
        if T_dec > 1 and kv_cache is None:
            causal = torch.triu(
                torch.full((T_dec, T_dec), float("-inf"), device=input_ids.device),
                diagonal=1,
            )
            self_attn_mask = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, T, T]

        # Cross-attention padding mask: block padded encoder positions from all decoder queries
        encoder_attn_mask = None
        if encoder_attention_mask is not None:
            encoder_attn_mask = make_key_padding_mask(encoder_attention_mask, hidden_states.dtype)

        # Forward through all decoder layers
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                self_attn_mask=self_attn_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=encoder_attn_mask,
                position_embeddings=position_embeddings,
                kv_cache=kv_cache,
            )
            dbg_layer("decoder/layer_out", layer_idx, hidden_states)

        # Final LayerNorm
        hidden_states = self.norm(hidden_states)
        dbg_tensor("decoder/final_hidden", hidden_states)

        return hidden_states


# =============================================================================
# FULL MODEL
# =============================================================================

class MoonshineStreamingModel(nn.Module):
    """Encoder-decoder backbone without the LM head."""

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.encoder = MoonshineStreamingEncoder(config.encoder)
        self.decoder = MoonshineStreamingDecoder(config.decoder)


class MoonshineStreamingForConditionalGeneration(nn.Module):
    """
    Full ASR model: encoder + decoder + LM head.

    The LM head (proj_out) maps the decoder's hidden states to vocabulary logits.
    In the HuggingFace checkpoint, proj_out.weight is NOT tied to embed_tokens.weight
    (tie_word_embeddings=False), so both are stored separately in the .pth file.

    Public API:
        encode()       — run encoder, return (encoder_hidden_states, mask)
        generate()     — greedy decode with KV cache, return token ID list
        load_weights() — load from .pth file produced by convert.py
    """

    def __init__(self, config: MoonshineStreamingConfig):
        super().__init__()
        self.config   = config
        self.model    = MoonshineStreamingModel(config)
        self.proj_out = nn.Linear(config.decoder.hidden_size, config.decoder.vocab_size, bias=False)

    def encode(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the encoder on raw waveform.

        Returns:
            encoder_hidden_states: [B, T_enc, 320]
            encoder_attention_mask: [B, T_enc] bool or None
        """
        dbg("model.encode", f"input shape={list(input_values.shape)}")
        return self.model.encoder(input_values, attention_mask)

    def _decode_greedy(
        self,
        projected_encoder: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        eos_min_steps: int,
        return_token_confidences: bool = False,
    ) -> "List[int] | Tuple[List[int], List[float]]":
        """
        Decode one utterance with the repository's default fast path.

        Even though this is still "greedy", we optionally apply lightweight
        constraints to keep the decoder from:
            - quitting unrealistically early,
            - falling into short token loops,
            - or repeatedly reusing the same token inventory.

        These controls are inference-only and remain intentionally weak so the
        acoustic evidence still dominates the decision.

        Two optional extensions (both default-off to keep the common path fast):

        return_token_confidences:
            When True, also records the softmax top-1 probability of the chosen
            token at each step and returns it alongside the token IDs.
            The confidence is computed AFTER constraint application, so it
            reflects the model's certainty within the allowed token set, not
            the raw LM head distribution.  This is the right signal for
            downstream stitching: a token that was heavily penalised and still
            won will show low confidence, which is exactly what we want.

            Overhead: one 32768-dim softmax per step when enabled.  On CPU or
            CUDA this is negligible compared to the transformer forward pass.

        Repetition loop early-stop (_is_repetition_loop):
            Fires AFTER argmax, BEFORE appending the new token.  If the chosen
            token has appeared too many times in the recent window, decoding
            stops immediately and the offending token is NOT appended.
            See _is_repetition_loop for the full rationale.

        Returns:
            List[int]                          if return_token_confidences=False
            Tuple[List[int], List[float]]      if return_token_confidences=True
        """
        device = projected_encoder.device
        cfg = self.config
        kv_cache = KVCache(cfg.decoder.num_hidden_layers)
        generated: List[int] = []
        # Allocate only when the caller actually needs confidence data.
        confidences: List[float] = []
        decoder_input_ids = torch.tensor(
            [[cfg.decoder_start_token_id]],
            device=device,
            dtype=torch.long,
        )

        for step in range(max_new_tokens):
            past_seq_len = kv_cache.get_self_seq_len()
            hidden_states = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=projected_encoder,
                encoder_attention_mask=encoder_attention_mask,
                past_seq_len=past_seq_len,
                kv_cache=kv_cache,
            )
            dbg_tensor(f"generate/step{step}/hidden", hidden_states)

            logits = self.proj_out(hidden_states[:, -1, :]).squeeze(0)
            logits = _apply_decoder_constraints(
                logits=logits,
                token_ids=generated,
                eos_id=cfg.eos_token_id,
                step=step,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_min_steps=eos_min_steps,
            )
            next_id = logits.argmax(dim=-1).item()
            dbg("generate", f"step={step} next_token={next_id}")

            if next_id == cfg.eos_token_id:
                dbg("generate", f"EOS at step {step}")
                break

            # Early-stop: single-token repetition loop detected.
            # Stop BEFORE appending so the tail is not contaminated.
            if _is_repetition_loop(generated, next_id):
                dbg("generate", f"repetition loop at step {step}, token={next_id}")
                break

            # Capture decode confidence for the winning token.
            # Deferred to after the guard checks so we never record a token
            # that would not be appended anyway.
            if return_token_confidences:
                prob = logits.softmax(dim=-1)[next_id].item()
                confidences.append(prob)

            generated.append(next_id)
            decoder_input_ids = torch.tensor([[next_id]], device=device, dtype=torch.long)

        if return_token_confidences:
            return generated, confidences
        return generated


    def _decode_beam_search(
        self,
        projected_encoder: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor],
        max_new_tokens: int,
        beam_width: int,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        eos_min_steps: int,
    ) -> List[int]:
        """
        Decode with a very small beam for hard chunks only.

        This implementation is intentionally narrow in scope:
            - batch size is still 1,
            - beam width is expected to stay tiny (for example 2-3),
            - and we reuse the same decoder/cache code as greedy mode.

        That keeps the code aligned with the existing architecture while still
        giving weak chunks a second chance to escape a locally bad greedy token.
        """
        device = projected_encoder.device
        cfg = self.config

        beams = [
            BeamSearchState(
                token_ids=[],
                logprob_sum=0.0,
                kv_cache=KVCache(cfg.decoder.num_hidden_layers),
                next_input_ids=torch.tensor(
                    [[cfg.decoder_start_token_id]],
                    device=device,
                    dtype=torch.long,
                ),
                is_finished=False,
            )
        ]

        for step in range(max_new_tokens):
            expanded: List[BeamSearchState] = []

            for beam in beams:
                if beam.is_finished or beam.next_input_ids is None:
                    expanded.append(beam)
                    continue

                past_seq_len = beam.kv_cache.get_self_seq_len()
                hidden_states = self.model.decoder(
                    input_ids=beam.next_input_ids,
                    encoder_hidden_states=projected_encoder,
                    encoder_attention_mask=encoder_attention_mask,
                    past_seq_len=past_seq_len,
                    kv_cache=beam.kv_cache,
                )

                logits = self.proj_out(hidden_states[:, -1, :]).squeeze(0)
                logits = _apply_decoder_constraints(
                    logits=logits,
                    token_ids=beam.token_ids,
                    eos_id=cfg.eos_token_id,
                    step=step,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    eos_min_steps=eos_min_steps,
                )
                log_probs = F.log_softmax(logits, dim=-1)
                top_k = min(beam_width, log_probs.shape[-1])
                top_values, top_indices = torch.topk(log_probs, k=top_k)

                base_cache = beam.kv_cache.clone()
                for token_logprob, token_id in zip(top_values.tolist(), top_indices.tolist()):
                    is_finished = token_id == cfg.eos_token_id
                    next_token_ids = list(beam.token_ids)
                    if not is_finished:
                        next_token_ids.append(token_id)

                    expanded.append(
                        BeamSearchState(
                            token_ids=next_token_ids,
                            logprob_sum=beam.logprob_sum + float(token_logprob),
                            kv_cache=base_cache.clone(),
                            next_input_ids=(
                                None
                                if is_finished
                                else torch.tensor([[token_id]], device=device, dtype=torch.long)
                            ),
                            is_finished=is_finished,
                        )
                    )

            beams = sorted(
                expanded,
                key=lambda beam: _length_normalized_score(beam.logprob_sum, len(beam.token_ids)),
                reverse=True,
            )[:beam_width]

            if all(beam.is_finished for beam in beams):
                break

        best_beam = max(
            beams,
            key=lambda beam: _length_normalized_score(beam.logprob_sum, len(beam.token_ids)),
        )
        return best_beam.token_ids

    @torch.no_grad()
    def generate(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        memory_state: Optional[LongFormMemoryState] = None,
        return_memory_state: bool = False,
        memory_token_budget: int = _MEMORY_BUDGET_TOTAL,
        beam_width: int = 1,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_min_steps: int = 0,
        return_token_confidences: bool = False,
    ) -> "List[int] | Tuple[List[int], LongFormMemoryState] | Tuple[List[int], List[float]] | Tuple[List[int], LongFormMemoryState, List[float]]":
        """
        Greedy autoregressive decoding with KV cache.

        Algorithm:
            1. Encode input audio once (O(T_enc) total)
            2. Project encoder output once (pos_emb + proj)
            3. Optionally prepend a tiny bounded encoder memory from the
               previous chunk when the acoustic boundary looks continuous.
            4. Initialize KV cache and start from BOS token
            5. Per step: feed one token, append to self-attn cache, read logit,
               take argmax → next token. Stop at EOS or max_new_tokens.

        Complexity per step:
            Self-attn:  O(t) where t = current sequence length
            Cross-attn: O(1) — K/V already cached from step 0
            vs. without cache: O(t²) per step

        Args:
            input_values:   [1, audio_len] waveform (batch size = 1)
            attention_mask: [1, audio_len] or None
            max_new_tokens: limit; defaults to ~6.5 tokens/sec × audio_duration
            memory_state:
                Optional carry-over encoder memory from the previous chunk.
                This memory is bounded, parameter-free, and detached from the
                graph so it stays cheap and stable for long-form inference.
            return_memory_state:
                When True, also return the freshly built memory state derived
                from the CURRENT chunk's encoder tail. The caller can then feed
                it into the next neighbouring chunk.
            beam_width:
                Beam width for selective search. Keep this tiny. `1` means the
                standard greedy path.
            repetition_penalty:
                Soft penalty against reusing previously generated token IDs.
                Values slightly above 1.0 are usually enough for ASR.
            no_repeat_ngram_size:
                Optional local n-gram blocking size. `0` disables it.
            eos_min_steps:
                Minimum decode steps before EOS is allowed.
            return_token_confidences:
                When True, also return a per-token confidence list (List[float])
                alongside the token IDs. Each value is the softmax top-1
                probability of the chosen token AFTER constraint application.
                Only supported on the greedy path (beam_width=1). When beam
                search is used, confidences are silently unavailable and the
                return type falls back to the no-confidence variant.

        Returns (depending on flags):
            return_memory_state=False, return_token_confidences=False:
                List[int]
            return_memory_state=True,  return_token_confidences=False:
                Tuple[List[int], LongFormMemoryState]
            return_memory_state=False, return_token_confidences=True:
                Tuple[List[int], List[float]]
            return_memory_state=True,  return_token_confidences=True:
                Tuple[List[int], LongFormMemoryState, List[float]]

        Long-form note:
            The memory here is intentionally encoder-side only. We do not reuse
            decoder self-attention KV across chunks because that would make the
            next chunk overly sensitive to previous token mistakes. By keeping
            the carry-over anchored in acoustic encoder states, we improve local
            continuity without creating a long hallucination chain.
        """
        device = input_values.device
        cfg    = self.config

        # Estimate token budget from audio length: ~6.5 tokens per second
        if max_new_tokens is None:
            audio_len      = input_values.shape[-1]
            max_new_tokens = max(1, int(audio_len * 6.5 / 16000))
        dbg("generate", f"max_new_tokens={max_new_tokens}")

        # Step 1: Encode audio
        encoder_hidden_states, encoder_attention_mask = self.encode(input_values, attention_mask)
        dbg_tensor("generate/encoder_out", encoder_hidden_states)

        # Step 2: Project encoder output (inject positional embeddings)
        # Done once here, reused for all cross-attention steps
        projected_encoder = self.model.decoder.project_encoder_output(encoder_hidden_states)
        dbg_tensor("generate/projected_encoder", projected_encoder)

        # Build outgoing memory BEFORE any merge so it always reflects ONLY
        # the current chunk — prevents recursive re-packing of old context.
        # The new slot is added to the rolling bank from the previous state.
        new_memory_state = _build_longform_memory_state(
            encoder_hidden_states=encoder_hidden_states,
            projected_encoder_states=projected_encoder,
            previous_state=memory_state,
            max_memory_tokens=memory_token_budget,
        )

        # Gate: only merge incoming memory when the current chunk's opening
        # is acoustically consistent with the most recent stored slot.
        # Uses the multi-point gate (A4): min similarity across 3 probe points.
        # We compare in PROJECTED space so the gate matches the same space used
        # for actual cross-attention memory tokens.
        memory_similarity = _memory_similarity(memory_state, projected_encoder)
        should_merge_memory = (
            memory_similarity is not None
            and memory_similarity >= _MEMORY_SIMILARITY_THRESHOLD
        )
        if should_merge_memory and memory_state is not None:
            projected_encoder, encoder_attention_mask = _merge_longform_memory(
                projected_encoder_states=projected_encoder,
                encoder_attention_mask=encoder_attention_mask,
                memory_state=memory_state,
            )
            dbg("generate/memory", f"merged similarity={memory_similarity:.3f} slots={len(memory_state.slots)}")
        elif memory_similarity is not None:
            dbg("generate/memory", f"discarded similarity={memory_similarity:.3f}")

        # Step 3: Decode. We keep greedy as the default because it is the
        # fastest and most stable path for most chunks. Beam search exists only
        # as a selective rescue tool for hard chunks in long-form inference.
        token_confidences: Optional[List[float]] = None
        if beam_width <= 1:
            decode_result = self._decode_greedy(
                projected_encoder=projected_encoder,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_min_steps=eos_min_steps,
                return_token_confidences=return_token_confidences,
            )
            if return_token_confidences:
                generated, token_confidences = decode_result
            else:
                generated = decode_result
        else:
            # Beam search: per-token confidence is ill-defined across branching
            # paths, so we silently skip it. Callers requesting confidence on the
            # beam path will receive the no-confidence return variant.
            generated = self._decode_beam_search(
                projected_encoder=projected_encoder,
                encoder_attention_mask=encoder_attention_mask,
                max_new_tokens=max_new_tokens,
                beam_width=beam_width,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                eos_min_steps=eos_min_steps,
            )

        dbg("generate", f"total tokens: {len(generated)}")

        # Build return value: variable-length tuple depending on which optional
        # outputs were requested. Ordered: token_ids, memory_state, confidences.
        if return_memory_state and return_token_confidences and token_confidences is not None:
            return generated, new_memory_state, token_confidences
        if return_memory_state:
            return generated, new_memory_state
        if return_token_confidences and token_confidences is not None:
            return generated, token_confidences
        return generated

    def load_weights(self, path: str, device: torch.device):
        """
        Load weights from a .pth file produced by convert.py.

        Expected missing keys (not a problem):
            model.decoder.rotary_emb.inv_freq — registered as non-persistent buffer,
            not stored in the checkpoint; recomputed at init from rope_theta.

        Uses strict=False to tolerate the inv_freq omission. Any other missing
        keys are logged as warnings.
        """
        dbg("load_weights", f"loading from {path}")

        state_dict = torch.load(path, map_location=device, weights_only=True)

        from config import dbg_weights, dbg_missing_keys
        dbg_weights("loaded_state_dict", state_dict)

        model_keys  = set(self.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        dbg_missing_keys(loaded_keys, model_keys)

        missing, unexpected = self.load_state_dict(state_dict, strict=False)

        if DEBUG:
            if missing:
                print(f"[DEBUG] load_state_dict missing ({len(missing)}): {missing}", flush=True)
            if unexpected:
                print(f"[DEBUG] load_state_dict unexpected ({len(unexpected)}): {unexpected}", flush=True)

        # Only warn about truly unexpected missing keys (not inv_freq)
        truly_missing = [k for k in missing if "inv_freq" not in k]
        if truly_missing:
            print(f"[WARN] Truly missing keys (not inv_freq): {truly_missing}")

        print(f"[INFO] Weights loaded from {path}")
        dbg("load_weights", "done")
