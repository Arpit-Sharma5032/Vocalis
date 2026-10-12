"""Deviation scoring and flaw-region detection against a baseline reading.

The participant and the baseline read the same text. Words are matched one to
one, and each word is compared with its baseline twin using speaker-normalized
features. Because every deviation is measured *relative to the participant's own
typical deviation* (median over the whole speech), a naturally faster, quieter
or higher-pitched speaker is not flagged; only local departures are.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from backend.features import FrameFeatures, refine_word_times
from backend.textutil import ends_phrase, normalize_word


@dataclass(frozen=True)
class Thresholds:
    """Detection settings. Deviations are relative to the speaker's own typical value."""

    window_words: int = 5  # words per local comparison window (centred on each word)
    pace_log: float = math.log(1.25)  # |log change in speaking time| that counts as a flaw
    pitch_log: float = math.log(0.6)  # pitch spread ratio below this counts as monotone
    energy_db: float = 2.5  # |speech-level change| in dB that counts as a flaw
    pause_excess_s: float = 0.30  # extra silence beyond the expected gap, mid-phrase
    pause_punct_excess_s: float = 1.00  # ... at punctuation, where a pause is natural
    min_region_words: int = 3
    merge_gap_words: int = 1
    refine_boundaries: bool = True  # cut words back around silences the aligner swallowed
    noise_k: float = 0.0  # 0 = off. Otherwise a deviation must also exceed noise_k x its own robust spread


@dataclass
class Region:
    """One detected flaw region (times on the participant's timeline)."""

    flaw_type: str
    start: float
    end: float
    score: float  # peak deviation, in the units of the flaw's metric
    detail: dict = field(default_factory=dict)


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


@dataclass
class Deviations:
    """Per-word deviations from the baseline (computed once, reusable for any thresholds)."""

    pairs: list[tuple[int, int]]
    ops: list
    B: list[dict]  # matched baseline words
    P: list[dict]  # matched participant words
    log_rate: np.ndarray  # log(participant / baseline speaking time) in each window
    dev_rate: np.ndarray  # ... minus the speaker's own median
    dev_spread: np.ndarray  # log pitch-spread ratio minus the speaker's median
    dev_energy: np.ndarray  # speech-level change (dB) minus the speaker's median
    g_rate: float
    g_spread: float
    g_energy: float
    base_feats: FrameFeatures
    part_feats: FrameFeatures
    part_words: list[dict]


def compute_deviations(
    base_feats: FrameFeatures, base_words: list[dict],
    part_feats: FrameFeatures, part_words: list[dict], window_words: int = 5,
    refine: bool = True,
) -> Deviations:
    """Match words and measure every word's deviation from its baseline twin."""
    if refine:
        base_words = refine_word_times(base_feats, base_words)
        part_words = refine_word_times(part_feats, part_words)
    pairs, ops = match_words(base_words, part_words)
    B = [base_words[i] for i, _ in pairs]
    P = [part_words[j] for _, j in pairs]
    m, h = len(pairs), window_words // 2
    log_rate = np.full(m, np.nan)
    log_spread = np.full(m, np.nan)
    d_energy = np.full(m, np.nan)
    for k in range(m if m >= window_words else 0):
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
    g = [float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0 for x in (log_rate, log_spread, d_energy)]
    return Deviations(
        pairs, ops, B, P, log_rate, log_rate - g[0], log_spread - g[1], d_energy - g[2],
        g[0], g[1], g[2], base_feats, part_feats, part_words,
    )


def _noise_scale(x: np.ndarray) -> float:
    """Robust spread (1.4826 x MAD) of a deviation series; 0 if too few values."""
    v = x[np.isfinite(x)]
    if len(v) < 10:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


def regions_from_deviations(dev: Deviations, th: Thresholds = Thresholds()) -> list[Region]:
    """Turn per-word deviations into flaw regions using the given thresholds."""
    regions = _stumbles(dev.part_words, dev.ops)
    B, P, m = dev.B, dev.P, len(dev.pairs)
    if len(dev.dev_rate) == 0 or np.isnan(dev.dev_rate).all() and np.isnan(dev.dev_spread).all():
        return sorted(regions, key=lambda r: r.start)

    def add(ftype: str, flag: np.ndarray, deviation: np.ndarray, detail_fn) -> None:
        for k0, k1 in _runs(flag, th.min_region_words, th.merge_gap_words):
            score = float(np.nanmax(np.abs(deviation[k0 : k1 + 1])))
            regions.append(Region(ftype, P[k0]["start"], P[k1]["end"], score, detail_fn(k0, k1)))

    def pace_detail(k0: int, k1: int) -> dict:
        db = sum(w["end"] - w["start"] for w in B[k0 : k1 + 1])
        dp = sum(w["end"] - w["start"] for w in P[k0 : k1 + 1])
        n = k1 - k0 + 1
        return {"local_ratio": dp / db, "typical_ratio": math.exp(dev.g_rate),
                "baseline_ms_per_word": 1000 * db / n, "participant_ms_per_word": 1000 * dp / n}

    def pitch_detail(k0: int, k1: int) -> dict:
        sb = _spread(dev.base_feats, B[k0]["start"], B[k1]["end"], 5)
        sp = _spread(dev.part_feats, P[k0]["start"], P[k1]["end"], 5)
        return {"spread_baseline_st": sb, "spread_participant_st": sp,
                "ratio": sp / sb if sb else float("nan"), "typical_ratio": math.exp(dev.g_spread)}

    def energy_detail(k0: int, k1: int) -> dict:
        eb = _speech_level(dev.base_feats, B[k0]["start"], B[k1]["end"], 5)
        ep = _speech_level(dev.part_feats, P[k0]["start"], P[k1]["end"], 5)
        return {"delta_db": ep - eb, "typical_delta_db": dev.g_energy}

    k_noise = th.noise_k
    pace_thr = max(th.pace_log, k_noise * _noise_scale(dev.dev_rate))
    pitch_thr = min(th.pitch_log, -k_noise * _noise_scale(dev.dev_spread))
    energy_thr = max(th.energy_db, k_noise * _noise_scale(dev.dev_energy))
    add("pace", dev.dev_rate < -pace_thr, dev.dev_rate, pace_detail)
    add("slow", dev.dev_rate > pace_thr, dev.dev_rate, pace_detail)
    add("monotone", dev.dev_spread < pitch_thr, dev.dev_spread, pitch_detail)
    add("volume", np.abs(dev.dev_energy) > energy_thr, dev.dev_energy, energy_detail)

    candidates = []  # pauses: excess silence between adjacent matched words
    for k in range(1, m):
        if dev.pairs[k][0] != dev.pairs[k - 1][0] + 1 or dev.pairs[k][1] != dev.pairs[k - 1][1] + 1:
            continue
        gap_b = max(0.0, B[k]["start"] - B[k - 1]["end"])
        gap_p = max(0.0, P[k]["start"] - P[k - 1]["end"])
        local = np.nanmean(dev.log_rate[k - 1 : k + 1]) if np.isfinite(dev.log_rate[k - 1 : k + 1]).any() else dev.g_rate
        # natural gaps stretch when the local speech is slower, but do not shrink when it is faster
        expected = gap_b * max(1.0, math.exp(local))
        threshold = th.pause_punct_excess_s if ends_phrase(P[k - 1]["word"]) else th.pause_excess_s
        candidates.append((k, gap_p, expected, threshold))
    pause_noise = k_noise * _noise_scale(np.array([g - e for _, g, e, _ in candidates]))
    for k, gap_p, expected, threshold in candidates:
        if gap_p - expected >= max(threshold, pause_noise):
            regions.append(Region("pause", P[k - 1]["end"], P[k]["start"], gap_p - expected, {
                "gap_s": gap_p, "expected_gap_s": expected,
                "before": P[k - 1]["word"], "after": P[k]["word"],
            }))
    return sorted(regions, key=lambda r: r.start)


def analyze(
    base_feats: FrameFeatures, base_words: list[dict],
    part_feats: FrameFeatures, part_words: list[dict],
    th: Thresholds = Thresholds(),
) -> list[Region]:
    """Compare a participant reading with the baseline and return flaw regions."""
    dev = compute_deviations(base_feats, base_words, part_feats, part_words, th.window_words,
                             th.refine_boundaries)
    return regions_from_deviations(dev, th)


def load_thresholds(path: str) -> Thresholds:
    """Load thresholds from a JSON file (missing keys keep their defaults)."""
    import json
    from dataclasses import fields

    raw = json.loads(open(path, encoding="utf-8").read())
    known = {f.name for f in fields(Thresholds)}
    return Thresholds(**{k: v for k, v in raw.items() if k in known})