"""
longform.py
===========
Shared helpers for long-form ASR orchestration.

Why this module exists:
    The models in this repository are intentionally kept untouched. Long-form
    quality problems are therefore handled one layer above the models:

    1. Cap chunk duration by *quality*, not only by memory.
    2. Estimate a decode budget that is proportional to chunk duration.
    3. Detect obvious repetition / degeneration in the decoded text.
    4. Re-split only the bad chunks into smaller overlapping windows.
    5. Stitch neighbouring chunk texts with overlap-aware deduplication.

The code here is deliberately small and explicit. The goal is not to build a
generic ASR framework, but to give both inference pipelines the same stable
long-form control flow.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

import numpy as np

from vad import ChunkParams


_DOCUMENT_NORMALIZATION_STOPWORDS = {
    "about", "after", "again", "against", "american", "because", "before",
    "being", "between", "called", "could", "every", "first", "friends",
    "great", "heard", "little", "might", "other", "people", "peter",
    "public", "recording", "section", "sections", "should", "still", "their",
    "there", "these", "thing", "those", "through", "under", "until", "which",
    "while", "would", "young",
}


@dataclass(frozen=True)
class LongFormTuning:
    """
    Per-model long-form tuning knobs.

    These values are intentionally practical instead of "fully automatic".
    The repo already derives memory-safe chunk sizes automatically; this layer
    adds a quality ceiling discovered from empirical behaviour on small models.
    """

    quality_target_dur: float
    quality_max_dur: float
    min_split_dur: float
    split_overlap_dur: float
    max_split_depth: int
    decoder_tokens_per_second: float
    decoder_token_buffer: int
    decoder_min_tokens: int
    decoder_max_tokens: int


@dataclass(frozen=True)
class TranscriptAssessment:
    """
    Lightweight quality signal for one decoded chunk.

    This is not a probabilistic confidence estimate. It is a deterministic
    heuristic that helps us choose between multiple transcripts of the same
    audio when we vary chunk size or decode budget.
    """

    score: float
    word_count: int
    unique_ratio: float
    is_degenerate: bool
    looks_truncated: bool
    is_too_short: bool


def apply_quality_cap(base_params: ChunkParams, tuning: LongFormTuning) -> ChunkParams:
    """
    Clamp auto-derived chunk parameters with a model-specific quality ceiling.

    Rationale:
        The existing chunker already does a good job of staying inside hardware
        limits, but that does not mean the model remains accurate near those
        limits. For long-form ASR we prefer a smaller chunk that the model can
        transcribe reliably over a larger chunk that merely "fits in memory".
    """

    max_dur = min(base_params.max_dur, tuning.quality_max_dur)
    target_dur = min(base_params.target_dur, tuning.quality_target_dur, max_dur)
    min_dur = min(base_params.min_dur, max(0.5, tuning.min_split_dur * 0.5))
    overlap_fallback = min(base_params.overlap_fallback, tuning.split_overlap_dur)

    return ChunkParams(
        target_dur=round(target_dur, 2),
        max_dur=round(max_dur, 2),
        min_dur=round(min_dur, 2),
        overlap_fallback=round(max(0.5, overlap_fallback), 2),
    )


def estimate_decode_budget(duration_sec: float, tuning: LongFormTuning) -> int:
    """
    Estimate a conservative decode budget for one chunk.

    The native models expose only simple stop criteria. If the token budget is
    far too generous, the decoder keeps talking after the acoustic evidence is
    already exhausted and the output falls into repetition loops.
    """

    estimated = int(duration_sec * tuning.decoder_tokens_per_second)
    budget = estimated + tuning.decoder_token_buffer
    budget = max(tuning.decoder_min_tokens, budget)
    budget = min(tuning.decoder_max_tokens, budget)
    return budget


def is_degenerate_transcript(text: str, duration_sec: float) -> bool:
    """
    Detect the most common long-form failure mode: repetition / hallucination.

    The heuristic is intentionally conservative:
        - We only inspect fairly long outputs where collapse is meaningful.
        - We look for repeated n-gram runs and very low lexical diversity.
        - We do not try to infer semantic correctness.
    """

    words = _tokenize_words(text)
    word_count = len(words)

    if duration_sec >= 6.0 and word_count <= 2:
        return True
    if word_count < 24:
        return False

    unique_ratio = len(set(words)) / max(word_count, 1)
    max_bigram_run = _max_ngram_repeat(words, 2)
    max_trigram_run = _max_ngram_repeat(words, 3)
    max_fourgram_run = _max_ngram_repeat(words, 4)
    max_word_run = _max_word_repeat(words)

    if unique_ratio < 0.38:
        return True
    if max_word_run >= 5:
        return True
    if max_bigram_run >= 4 or max_trigram_run >= 3 or max_fourgram_run >= 3:
        return True

    return False


def assess_transcript(text: str, duration_sec: float) -> TranscriptAssessment:
    """
    Score one transcript using cheap text-only heuristics.

    The score intentionally rewards:
        - enough lexical content for the chunk duration,
        - reasonable diversity,
        - non-degenerate endings.

    It strongly penalizes:
        - repetition collapse,
        - suspiciously short outputs for non-trivial audio,
        - tails that look cut off mid-thought.
    """

    words = _tokenize_words(text)
    word_count = len(words)
    unique_ratio = len(set(words)) / max(word_count, 1)
    is_degenerate = is_degenerate_transcript(text, duration_sec)

    min_expected_words = max(4, int(duration_sec * 1.25))
    is_too_short = duration_sec >= 6.0 and word_count < min_expected_words
    looks_truncated = _looks_truncated_tail(text)

    useful_words = min(word_count, max(min_expected_words * 2, 8))
    score = useful_words * 0.12
    score += unique_ratio * 4.0
    if is_too_short:
        score -= 3.0
    if looks_truncated:
        score -= 1.5
    if is_degenerate:
        score -= 8.0

    return TranscriptAssessment(
        score=score,
        word_count=word_count,
        unique_ratio=unique_ratio,
        is_degenerate=is_degenerate,
        looks_truncated=looks_truncated,
        is_too_short=is_too_short,
    )


def choose_better_transcript(
    primary_text: str,
    candidate_text: str,
    duration_sec: float,
) -> str:
    """
    Pick the healthier transcript between two attempts of the same audio.

    This helper is intentionally deterministic. It prefers:
        1. non-degenerate over degenerate,
        2. non-truncated over truncated when content is otherwise similar,
        3. the higher overall heuristic score.
    """

    primary = assess_transcript(primary_text, duration_sec)
    candidate = assess_transcript(candidate_text, duration_sec)

    if primary.is_degenerate and not candidate.is_degenerate:
        return candidate_text
    if candidate.is_degenerate and not primary.is_degenerate:
        return primary_text
    if candidate.score > (primary.score + 0.35):
        return candidate_text
    return primary_text


def transcribe_adaptive_chunk(
    waveform: np.ndarray,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
    transcribe_fn: Callable[[np.ndarray, float], str],
    tuning: LongFormTuning,
    log_fn: Callable[[str], None] | None = None,
    depth: int = 0,
) -> list[dict]:
    """
    Transcribe one time interval, re-splitting only when the text collapses.

    This keeps the normal path fast:
        - Good chunk -> one model call.
        - Bad chunk  -> recursively split into two overlapping children.

    The overlap is small on purpose. We only need enough shared audio to make
    stitching robust at the split boundary.
    """

    start_sample = int(start_sec * sample_rate)
    end_sample = min(int(end_sec * sample_rate), len(waveform))
    chunk_wav = waveform[start_sample:end_sample]
    duration_sec = (end_sample - start_sample) / sample_rate

    text = transcribe_fn(chunk_wav, duration_sec).strip()

    should_retry = (
        depth < tuning.max_split_depth
        and duration_sec > tuning.min_split_dur
        and is_degenerate_transcript(text, duration_sec)
    )
    if not should_retry:
        return [{"start": start_sec, "end": end_sec, "text": text}]

    overlap = min(tuning.split_overlap_dur, duration_sec * 0.15)
    midpoint = (start_sec + end_sec) / 2.0
    left_end = min(end_sec, midpoint + overlap / 2.0)
    right_start = max(start_sec, midpoint - overlap / 2.0)

    left_dur = left_end - start_sec
    right_dur = end_sec - right_start
    if left_dur < tuning.min_split_dur or right_dur < tuning.min_split_dur:
        return [{"start": start_sec, "end": end_sec, "text": text}]

    if log_fn is not None:
        log_fn(
            "[INFO] Degenerate chunk detected "
            f"({start_sec:.2f}s - {end_sec:.2f}s); retrying as "
            f"{left_dur:.1f}s + {right_dur:.1f}s"
        )

    left_segments = transcribe_adaptive_chunk(
        waveform=waveform,
        sample_rate=sample_rate,
        start_sec=start_sec,
        end_sec=left_end,
        transcribe_fn=transcribe_fn,
        tuning=tuning,
        log_fn=log_fn,
        depth=depth + 1,
    )
    right_segments = transcribe_adaptive_chunk(
        waveform=waveform,
        sample_rate=sample_rate,
        start_sec=right_start,
        end_sec=end_sec,
        transcribe_fn=transcribe_fn,
        tuning=tuning,
        log_fn=log_fn,
        depth=depth + 1,
    )
    return left_segments + right_segments


def stitch_segments_text(segments: list[dict]) -> str:
    """
    Merge segment texts while removing exact word overlap at boundaries.

    We keep the algorithm intentionally simple and predictable:
        - Sort by time.
        - For each neighbour pair, find the longest exact suffix/prefix match.
        - Drop the duplicated prefix from the later segment.

    This is enough for the small intentional overlaps introduced by adaptive
    re-splitting, and it avoids heavier fuzzy alignment logic.
    """

    if not segments:
        return ""

    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    merged = ""
    for seg in ordered:
        merged = _stitch_pair(merged, seg["text"])
    return merged.strip()


def stitch_segments_with_confidence(segments: list[dict]) -> str:
    """
    Merge segment texts with optional confidence-weighted overlap resolution.

    Drop-in replacement for stitch_segments_text(). Falls back gracefully to
    exact-match behaviour when segments have no "token_confidences" field.

    When confidence data IS present on both sides of an overlap boundary, we
    prefer the surface form from whichever segment decoded that region with
    higher average per-token confidence. This helps resolve cases where two
    slightly different spellings of the same word appear in the overlap — the
    decoder that was more certain is usually more accurate.

    Backward compatible: missing or empty "token_confidences" → left wins
    (same as the existing _stitch_pair behaviour).
    """
    if not segments:
        return ""

    ordered = sorted(segments, key=lambda seg: (seg["start"], seg["end"]))
    merged_text = ""
    merged_confidences: list[float] = []

    for seg in ordered:
        seg_confidences = seg.get("token_confidences") or []
        merged_text, merged_confidences = _stitch_pair_with_confidence(
            left_text=merged_text,
            right_text=seg["text"],
            left_confidences=merged_confidences,
            right_confidences=seg_confidences,
        )

    return merged_text.strip()


def _mean_confidence(token_confidences: list[float], word_count: int) -> float:
    """
    Estimate average decode confidence for the last `word_count` words.

    Token confidences are per BPE subword token, not per word. English averages
    roughly 1-3 tokens per word. We use a window of word_count * 2 tokens as a
    conservative upper bound over the overlap region.

    Returns 0.0 when no confidence data is available so the caller's tie-break
    defaults to the left segment (matching _stitch_pair behaviour).
    """
    if not token_confidences or word_count <= 0:
        return 0.0
    window = min(len(token_confidences), word_count * 2)
    return sum(token_confidences[-window:]) / window


def _stitch_pair_with_confidence(
    left_text: str,
    right_text: str,
    left_confidences: list[float],
    right_confidences: list[float],
    min_overlap_words: int = 3,
    confidence_tolerance: float = 0.02,
) -> tuple[str, list[float]]:
    """
    Join two text fragments with overlap deduplication and confidence tie-breaking.

    Mirrors _stitch_pair() exactly. The only difference is that when both sides
    have confidence data AND the right side is clearly more confident over the
    overlap region, we prefer the right segment's surface form for those words.

    In practice for exact-match overlaps the surface forms are often identical
    (same tokenisation, same capitalisation). The confidence tie-break matters
    most when one side has a rare capitalisation or punctuation variant.

    Returns:
        (stitched_text, accumulated_confidences)

    The accumulated confidence list is a best-effort approximation — BPE tokens
    don't align 1:1 with words — but it is good enough for the next boundary
    decision in the stitching loop.
    """
    left_text = left_text.strip()
    right_text = right_text.strip()

    if not left_text:
        return right_text, list(right_confidences)
    if not right_text:
        return left_text, list(left_confidences)

    left_words  = left_text.split()
    right_words = right_text.split()
    left_norm   = [_normalize_word(w) for w in left_words]
    right_norm  = [_normalize_word(w) for w in right_words]

    max_window = min(32, len(left_words), len(right_words))
    best_overlap = 0
    for size in range(max_window, min_overlap_words - 1, -1):
        if left_norm[-size:] == right_norm[:size]:
            best_overlap = size
            break

    if best_overlap == 0:
        # No overlap: simple concatenation, merge confidence lists.
        return (
            f"{left_text} {right_text}".strip(),
            list(left_confidences) + list(right_confidences),
        )

    # Overlap found. Compare confidence over the overlap region.
    left_conf  = _mean_confidence(left_confidences,  best_overlap)
    right_conf = _mean_confidence(right_confidences, best_overlap)

    # Right wins only when BOTH sides have data and the advantage is clear.
    right_wins = (
        bool(left_confidences)
        and bool(right_confidences)
        and right_conf > left_conf + confidence_tolerance
    )

    if right_wins:
        # Keep left prefix + right's overlap words + right tail.
        overlap_words  = right_words[:best_overlap]
        stitched_words = left_words[:-best_overlap] + overlap_words + right_words[best_overlap:]
        # Drop the left tail tokens that correspond to the overlap region,
        # then append the full right confidences.
        left_prefix_conf = left_confidences[: max(0, len(left_confidences) - best_overlap * 2)]
        combined_conf    = left_prefix_conf + list(right_confidences)
    else:
        # Left wins (default): drop the overlap prefix from the right segment.
        stitched_words   = left_words + right_words[best_overlap:]
        right_tail_start = max(0, len(right_confidences) - best_overlap * 2)
        combined_conf    = list(left_confidences) + list(right_confidences[right_tail_start:])

    return " ".join(stitched_words).strip(), combined_conf


def normalize_document_terms(segments: list[dict]) -> list[dict]:
    """
    Apply conservative repeated-term normalization across the full transcript.

    The goal is not grammar correction. We only try to reduce unstable spelling
    of recurring uncommon terms such as names, where one dominant surface form
    appears multiple times and a close variant appears less often.
    """
    replacements = _build_document_replacements(segments)
    if not replacements:
        return segments

    normalized_segments: list[dict] = []
    for seg in segments:
        normalized_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": _apply_document_replacements(seg["text"], replacements),
        })
    return normalized_segments


def _stitch_pair(left_text: str, right_text: str, min_overlap_words: int = 3) -> str:
    """Join two text fragments and remove duplicated word overlap."""
    left_text = left_text.strip()
    right_text = right_text.strip()
    if not left_text:
        return right_text
    if not right_text:
        return left_text

    left_words = left_text.split()
    right_words = right_text.split()
    left_norm = [_normalize_word(word) for word in left_words]
    right_norm = [_normalize_word(word) for word in right_words]

    max_window = min(32, len(left_words), len(right_words))
    best_overlap = 0
    for size in range(max_window, min_overlap_words - 1, -1):
        if left_norm[-size:] == right_norm[:size]:
            best_overlap = size
            break

    if best_overlap == 0:
        return f"{left_text} {right_text}".strip()

    stitched_words = left_words + right_words[best_overlap:]
    return " ".join(stitched_words).strip()


def _build_document_replacements(segments: list[dict]) -> dict[str, str]:
    """
    Build a small replacement table for repeated uncommon terms.

    We require the dominant form to be clearly more frequent than the variant,
    and we only consider fairly specific words. This keeps the normalizer from
    rewriting common vocabulary.
    """
    token_counts: dict[str, int] = {}
    for seg in segments:
        for token in _extract_document_tokens(seg["text"]):
            token_counts[token] = token_counts.get(token, 0) + 1

    if not token_counts:
        return {}

    candidates = sorted(token_counts.keys(), key=lambda token: (-token_counts[token], token))
    replacements: dict[str, str] = {}
    used_variants: set[str] = set()

    for dominant in candidates:
        dominant_count = token_counts[dominant]
        if dominant in used_variants or dominant_count < 2:
            continue

        for variant in candidates:
            if variant == dominant or variant in used_variants:
                continue
            variant_count = token_counts[variant]
            if variant_count >= dominant_count:
                continue
            if dominant_count < (variant_count + 2):
                continue
            if not _looks_like_same_document_term(dominant, variant):
                continue

            replacements[variant] = dominant
            used_variants.add(variant)

    return replacements


def _extract_document_tokens(text: str) -> list[str]:
    """Extract uncommon word candidates for document-level consistency fixes."""
    tokens = []
    for token in re.findall(r"[A-Za-z][A-Za-z']+", text):
        normalized = token.lower().strip("'")
        if normalized.endswith("'s"):
            normalized = normalized[:-2]
        if len(normalized) < 5:
            continue
        if normalized in _DOCUMENT_NORMALIZATION_STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def _looks_like_same_document_term(left: str, right: str) -> bool:
    """
    Decide whether two rare terms are similar enough to normalize together.

    The matcher is intentionally narrow:
        - same first letter,
        - small length difference,
        - and high string similarity OR a shared long prefix.
    """
    if left[0] != right[0]:
        return False
    if abs(len(left) - len(right)) > 2:
        return False

    ratio = SequenceMatcher(None, left, right).ratio()
    if ratio >= 0.86:
        return True
    if ratio >= 0.74 and len(os.path.commonprefix([left, right])) >= 3:
        return True
    return False


def _apply_document_replacements(text: str, replacements: dict[str, str]) -> str:
    """Replace selected term variants while preserving possessive suffixes."""
    updated = text
    for source, target in replacements.items():
        updated = re.sub(
            rf"\b{re.escape(source)}('s)?\b",
            lambda match: target + (match.group(1) or ""),
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def _tokenize_words(text: str) -> list[str]:
    """Normalize text into comparable word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _normalize_word(word: str) -> str:
    """Normalize one token for overlap matching."""
    normalized = re.sub(r"[^a-z0-9']+", "", word.lower())
    return normalized


def _looks_truncated_tail(text: str) -> bool:
    """
    Detect endings that look cut off mid-thought.

    The heuristic is conservative and tuned for audiobook-style speech:
        - if the text already ends in sentence punctuation, we assume it is fine;
        - otherwise very short tail words are suspicious;
        - some function words are also suspicious as terminal words.
    """

    stripped = text.strip()
    if not stripped:
        return True
    if stripped.endswith((".", "!", "?", '"', "'")):
        return False

    words = stripped.split()
    if not words:
        return True

    last_word = _normalize_word(words[-1])
    if len(last_word) <= 3:
        return True

    if last_word in {
        "a", "an", "and", "as", "at", "but", "for", "from", "if", "in",
        "into", "of", "on", "or", "that", "the", "to", "was", "were", "with",
    }:
        return True

    return False


def _max_word_repeat(words: list[str]) -> int:
    """Return the longest run of the same word repeated consecutively."""
    if not words:
        return 0

    best = 1
    current = 1
    for prev, cur in zip(words, words[1:]):
        if cur == prev:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def _max_ngram_repeat(words: list[str], n: int) -> int:
    """
    Return the longest run of one n-gram repeated back-to-back.

    Example:
        "the story | the story | the story"
        with n=2 -> run length 3
    """

    if len(words) < n * 2:
        return 0

    best = 1
    for start in range(0, len(words) - n):
        gram = words[start : start + n]
        repeats = 1
        cursor = start + n
        while cursor + n <= len(words) and words[cursor : cursor + n] == gram:
            repeats += 1
            cursor += n
        best = max(best, repeats)
    return best
