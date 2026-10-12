"""Deviation scoring and flaw-region detection against a baseline reading.

The participant and the baseline read the same text. Words are matched one to
one, and each word is compared with its baseline twin using speaker-normalized
features. Because every deviation is measured *relative to the participant's own
typical deviation* (median over the whole speech), a naturally faster, quieter
or higher-pitched speaker is not flagged; only local departures are.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from backend.features import FrameFeatures


@dataclass(frozen=True)
class Thresholds:
    """Detection settings. Deviations are relative to the speaker's own typical value."""

    window_words: int = 5  # words per local comparison window (centred on each word)
    pace_log: float = math.log(1.25)  # |log change in speaking time| that counts as a flaw
    pitch_log: float = math.log(0.6)  # pitch spread ratio below this counts as monotone
    energy_db: float = 2.5  # |speech-level change| in dB that counts as a flaw
    pause_excess_s: float = 0.30  # extra silence beyond the expected gap
    min_region_words: int = 3
    merge_gap_words: int = 1


@dataclass
class Region:
    """One detected flaw region (times on the participant's timeline)."""

    flaw_type: str
    start: float
    end: float
    score: float  # peak deviation, in the units of the flaw's metric
    detail: dict = field(default_factory=dict)


def normalize_word(word: str) -> str:
    """Lower-case a word and strip punctuation, for matching across transcripts."""
    return re.sub(r"[^a-z0-9']", "", word.lower())


def match_words(base: list[dict], part: list[dict]) -> tuple[list[tuple[int, int]], list]:
    """Match baseline and participant words one to one.

    Returns (pairs, opcodes). pairs are (baseline_index, participant_index);
    words with no partner (e.g. repeated words) stay unmatched.
    """
    a = [normalize_word(w["word"]) for w in base]
    b = [normalize_word(w["word"]) for w in part]
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    ops = matcher.get_opcodes()
    pairs: list[tuple[int, int]] = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal" or (tag == "replace" and i2 - i1 == j2 - j1):
            pairs.extend(zip(range(i1, i2), range(j1, j2)))
    return pairs, ops


def _spread(feats: FrameFeatures, t0: float, t1: float, min_frames: int = 15) -> float:
    """Pitch spread (P90 - P10, semitones) of voiced frames in [t0, t1]."""
    m = (feats.times >= t0) & (feats.times <= t1)
    st = feats.f0_st[m]
    st = st[np.isfinite(st)]
    return float(np.percentile(st, 90) - np.percentile(st, 10)) if len(st) >= min_frames else float("nan")


SILENCE_REL_DB = -30.0  # frames this far below the speaker's peak level count as silence


def _speech_level(feats: FrameFeatures, t0: float, t1: float, min_frames: int = 20) -> float:
    """Peak speech level (90th percentile, dB re speaker level) of non-silent frames in [t0, t1].

    Silent frames are excluded so that pauses inside the span do not distort
    the level; only the loudness of actual speech is compared.
    """
    m = (feats.times >= t0) & (feats.times <= t1)
    e = feats.energy_rel_db[m]
    e = e[e > SILENCE_REL_DB]
    return float(np.percentile(e, 90)) if len(e) >= min_frames else float("nan")


def _runs(flag: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Contiguous runs of True (allowing small gaps), at least min_len long."""
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        return []
    runs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a + 1 >= min_len]


def _stumbles(part: list[dict], ops: list) -> list[Region]:
    """Repeated words: inserted participant words identical to a neighbouring word."""
    regions = []
    for tag, i1, i2, j1, j2 in ops:
        if tag != "insert":
            continue
        inserted = {normalize_word(part[j]["word"]) for j in range(j1, j2)}
        if len(inserted) != 1:
            continue
        rep = inserted.pop()
        left, right = j1, j2 - 1
        while left > 0 and normalize_word(part[left - 1]["word"]) == rep:
            left -= 1
        while right + 1 < len(part) and normalize_word(part[right + 1]["word"]) == rep:
            right += 1
        if right - left + 1 < 2:
            continue  # not actually a repetition
        regions.append(Region(
            "stumble", part[left]["start"], part[right]["end"], float(j2 - j1),
            {"word": part[left]["word"], "extra_repeats": j2 - j1},
        ))
    return regions


def analyze(
    base_feats: FrameFeatures, base_words: list[dict],
    part_feats: FrameFeatures, part_words: list[dict],
    th: Thresholds = Thresholds(),
) -> list[Region]:
    """Compare a participant reading with the baseline and return flaw regions."""
    pairs, ops = match_words(base_words, part_words)
    regions = _stumbles(part_words, ops)
    m = len(pairs)
    if m < th.window_words:
        return regions
    B = [base_words[i] for i, _ in pairs]
    P = [part_words[j] for _, j in pairs]
    h = th.window_words // 2

    log_rate = np.full(m, np.nan)
    log_spread = np.full(m, np.nan)
    d_energy = np.full(m, np.nan)
    for k in range(m):
        lo, hi = max(0, k - h), min(m - 1, k + h)
        db = sum(w["end"] - w["start"] for w in B[lo : hi + 1])
        dp = sum(w["end"] - w["start"] for w in P[lo : hi + 1])
        if db > 0 and dp > 0:
            log_rate[k] = math.log(dp / db)
        sb = _spread(base_feats, B[lo]["start"], B[hi]["end"])
        sp = _spread(part_feats, P[lo]["start"], P[hi]["end"])
        if sb > 0 and sp > 0:
            log_spread[k] = math.log(sp / sb)
        d_energy[k] = _speech_level(part_feats, P[lo]["start"], P[hi]["end"]) - _speech_level(
            base_feats, B[lo]["start"], B[hi]["end"]
        )

    g_rate, g_spread, g_energy = (float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0
                                  for x in (log_rate, log_spread, d_energy))
    dev_rate, dev_spread, dev_energy = log_rate - g_rate, log_spread - g_spread, d_energy - g_energy

    def add(ftype: str, flag: np.ndarray, dev: np.ndarray, detail_fn) -> None:
        for k0, k1 in _runs(flag, th.min_region_words, th.merge_gap_words):
            score = float(np.nanmax(np.abs(dev[k0 : k1 + 1])))
            regions.append(Region(ftype, P[k0]["start"], P[k1]["end"], score, detail_fn(k0, k1)))

    def pace_detail(k0: int, k1: int) -> dict:
        db = sum(w["end"] - w["start"] for w in B[k0 : k1 + 1])
        dp = sum(w["end"] - w["start"] for w in P[k0 : k1 + 1])
        n = k1 - k0 + 1
        return {"local_ratio": dp / db, "typical_ratio": math.exp(g_rate),
                "baseline_ms_per_word": 1000 * db / n, "participant_ms_per_word": 1000 * dp / n}

    def pitch_detail(k0: int, k1: int) -> dict:
        sb = _spread(base_feats, B[k0]["start"], B[k1]["end"], 5)
        sp = _spread(part_feats, P[k0]["start"], P[k1]["end"], 5)
        return {"spread_baseline_st": sb, "spread_participant_st": sp,
                "ratio": sp / sb if sb else float("nan"), "typical_ratio": math.exp(g_spread)}

    def energy_detail(k0: int, k1: int) -> dict:
        eb = _speech_level(base_feats, B[k0]["start"], B[k1]["end"], 5)
        ep = _speech_level(part_feats, P[k0]["start"], P[k1]["end"], 5)
        return {"delta_db": ep - eb, "typical_delta_db": g_energy}

    add("pace", dev_rate < -th.pace_log, dev_rate, pace_detail)
    add("slow", dev_rate > th.pace_log, dev_rate, pace_detail)
    add("monotone", dev_spread < th.pitch_log, dev_spread, pitch_detail)
    add("volume", np.abs(dev_energy) > th.energy_db, dev_energy, energy_detail)

    for k in range(1, m):  # pauses: excess silence between adjacent matched words
        if pairs[k][0] != pairs[k - 1][0] + 1 or pairs[k][1] != pairs[k - 1][1] + 1:
            continue
        gap_b = max(0.0, B[k]["start"] - B[k - 1]["end"])
        gap_p = max(0.0, P[k]["start"] - P[k - 1]["end"])
        expected = gap_b * math.exp(g_rate)
        if gap_p - expected >= th.pause_excess_s:
            regions.append(Region("pause", P[k - 1]["end"], P[k]["start"], gap_p - expected, {
                "gap_s": gap_p, "expected_gap_s": expected,
                "before": P[k - 1]["word"], "after": P[k]["word"],
            }))
    return sorted(regions, key=lambda r: r.start)