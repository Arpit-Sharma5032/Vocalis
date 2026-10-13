"""End-to-end analysis: uploaded reading + chosen reference speech -> scored report.

Steps: align the transcript to the audio, extract speaker-normalized features,
compare word by word with the reference reading, detect and explain flaw
regions, and score them with the rubric.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from backend.detection import Thresholds, analyze
from backend.explain import explain
from backend.features import FrameFeatures, extract_features, load_audio, refine_word_times
from backend.scoring_engine import score_reading
from dataset_builder.windows import load_words

SERIES_STEP = 5  # keep every 5th 10 ms frame -> 20 points per second for the dashboard
REFERENCE_SPEECHES = ["kennedy_inaugural", "eisenhower_farewell", "fdr_fireside_01", "lincoln_gettysburg"]
AlignFn = Callable[[str, str], list[dict]]

LIMITATIONS = [
    "Compares your delivery with ONE reference reading of the same text; it cannot judge content, "
    "meaning or emotion.",
    "Flaws shorter than about 4 seconds can be missed; subtle pace changes (under about 1.25x) are not detected.",
    "Repeated words are only visible if the transcript includes them (a verbatim transcript).",
    "Word timings come from forced alignment and can be off by tens of milliseconds, more near long pauses.",
    "Educational prototype; scores are not an official judging result.",
]


@dataclass(frozen=True)
class Paths:
    """Where reference data lives."""

    raw_dir: Path = Path("data/raw_audio")
    align_dir: Path = Path("data/processed_metrics")
    transcript_dir: Path = Path("data/transcripts")


def available_speeches(paths: Paths = Paths()) -> list[dict]:
    """Reference speeches that have audio, transcript and alignment on disk."""
    out = []
    for sid in REFERENCE_SPEECHES:
        files = [paths.raw_dir / f"{sid}.wav", paths.align_dir / f"{sid}_alignment.json",
                 paths.transcript_dir / f"{sid}.txt"]
        if all(f.exists() for f in files):
            words = load_words(files[1])
            out.append({"id": sid, "title": sid.replace("_", " ").title(),
                        "duration_s": round(words[-1]["end"], 1), "words": len(words),
                        "transcript": files[2].read_text(encoding="utf-8").strip()})
    return out


def default_align_fn() -> AlignFn:
    """WhisperX-based aligner (model loaded once, on first use)."""
    from backend.alignment import align_with_model, load_aligner

    model, metadata = load_aligner()
    return lambda wav, text: align_with_model(wav, text, model, metadata)


def _clean(x: float | None, nd: int = 2) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def _series(feats: FrameFeatures) -> dict:
    sl = slice(None, None, SERIES_STEP)
    return {"t": [round(float(t), 2) for t in feats.times[sl]],
            "pitch_st": [_clean(v) for v in feats.f0_st[sl]],
            "energy_db": [_clean(v, 1) for v in feats.energy_rel_db[sl]]}


def analyze_reading(
    audio_path: str, speech_id: str, transcript: str | None = None,
    align_fn: AlignFn | None = None, paths: Paths = Paths(), th: Thresholds = Thresholds(),
) -> dict:
    """Analyze one reading against a reference speech and return a JSON-ready report."""
    base_words = load_words(paths.align_dir / f"{speech_id}_alignment.json")
    base_wav = paths.raw_dir / f"{speech_id}.wav"
    text = transcript.strip() if transcript and transcript.strip() else \
        (paths.transcript_dir / f"{speech_id}.txt").read_text(encoding="utf-8").strip()

    align_fn = align_fn or default_align_fn()
    words = align_fn(audio_path, text)
    usable = [w for w in words if w["start"] is not None and w["end"] is not None]
    dropped = len(words) - len(usable)

    base_feats = extract_features(load_audio(str(base_wav)))
    part_feats = extract_features(load_audio(audio_path))
    if th.refine_boundaries:  # show words where they are actually voiced, not stretched over silences
        base_words = refine_word_times(base_feats, base_words)
        usable = refine_word_times(part_feats, usable)
    regions = analyze(base_feats, base_words, part_feats, usable, th)
    total_baseline = base_words[-1]["end"] - base_words[0]["start"]
    scores = score_reading(regions, total_baseline, th)

    return {
        "speech_id": speech_id,
        "duration_s": round(len(part_feats.times) * 0.01, 1),
        "reference_duration_s": round(base_words[-1]["end"], 1),
        "overall": scores["overall"],
        "dimensions": scores["dimensions"],
        "regions": [
            {"flaw_type": r.flaw_type, "start": round(r.start, 2), "end": round(r.end, 2),
             "penalty": pen, **explain(r)}
            for r, pen in zip(regions, scores["region_penalties"])
        ],
        "series": {"reference": _series(base_feats), "participant": _series(part_feats)},
        "words": {"reference": [{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2)} for w in base_words],
                  "participant": [{"w": w["word"], "s": round(w["start"], 2), "e": round(w["end"], 2)} for w in usable]},
        "thresholds": asdict(th),
        "warnings": ([f"{dropped} words had no timestamp and were ignored."] if dropped else []),
        "limitations": LIMITATIONS,
    }