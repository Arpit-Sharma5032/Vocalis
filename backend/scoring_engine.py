"""Reproducible rubric scoring, tied strictly to measured acoustic quantities.

Every detected flaw region costs penalty points. Each rubric dimension starts at
100 and loses the penalties of its own flaw types; the overall score starts at
100 and loses ALL penalties. The score is a pure function of the detected
regions, so the same audio always gives the same score.

Rubric
------
Dimension              Flaw types      Penalty per region (points)
Pace control           pace, slow      100 x coverage x severity
Vocal variety          monotone        100 x coverage x severity
Loudness consistency   volume          100 x coverage x severity
Pauses & fluency       pause           5 x severity
                       stumble         5 x number of extra repeats

coverage = the region's span in baseline time / total baseline duration.
severity = peak deviation / the detection threshold for that flaw, capped at 3
(1 = just over the threshold, 3 = three times beyond it).
Overall = 100 - sum of all penalties (never below 0).
"""
from __future__ import annotations

from dataclasses import dataclass

from backend.detection import Region, Thresholds

SEVERITY_CAP = 3.0
COVERAGE_DIVISOR = 1.0  # a region covering 10% of the speech at severity 2 costs 20 points
PAUSE_POINTS = 5.0
STUMBLE_POINTS = 5.0

DIMENSIONS = {
    "pace": ("Pace control", ("pace", "slow")),
    "fluency": ("Pauses & fluency", ("pause", "stumble")),
    "pitch": ("Vocal variety", ("monotone",)),
    "volume": ("Loudness consistency", ("volume",)),
}


@dataclass
class RegionPenalty:
    """Penalty attached to one region, with the factors that produced it."""

    penalty: float
    severity: float
    coverage: float


def _threshold(region: Region, th: Thresholds) -> float:
    """Detection threshold (in the region's own units) used to express severity."""
    return {
        "pace": th.pace_log, "slow": th.pace_log, "monotone": abs(th.pitch_log),
        "volume": th.energy_db, "pause": th.pause_excess_s,
    }[region.flaw_type]


def region_penalty(region: Region, total_baseline_s: float, th: Thresholds = Thresholds()) -> RegionPenalty:
    """Penalty points for one region (see the module docstring for the formula)."""
    if region.flaw_type == "stumble":
        return RegionPenalty(STUMBLE_POINTS * region.detail["extra_repeats"], float(region.detail["extra_repeats"]), 0.0)
    severity = min(SEVERITY_CAP, max(1.0, region.score / _threshold(region, th)))
    if region.flaw_type == "pause":
        return RegionPenalty(PAUSE_POINTS * severity, severity, 0.0)
    coverage = region.detail["baseline_span_s"] / total_baseline_s if total_baseline_s > 0 else 0.0
    return RegionPenalty(100.0 * coverage * severity / COVERAGE_DIVISOR, severity, coverage)


def score_reading(regions: list[Region], total_baseline_s: float, th: Thresholds = Thresholds()) -> dict:
    """Score a reading: per-dimension scores, overall score and per-region penalties."""
    penalties = [region_penalty(r, total_baseline_s, th) for r in regions]
    dims = {}
    for key, (label, kinds) in DIMENSIONS.items():
        lost = sum(p.penalty for r, p in zip(regions, penalties) if r.flaw_type in kinds)
        dims[key] = {"label": label, "score": round(max(0.0, 100.0 - lost), 1), "penalty": round(lost, 1),
                     "regions": sum(1 for r in regions if r.flaw_type in kinds)}
    overall = round(max(0.0, 100.0 - sum(p.penalty for p in penalties)), 1)
    return {"overall": overall, "dimensions": dims,
            "region_penalties": [round(p.penalty, 2) for p in penalties]}