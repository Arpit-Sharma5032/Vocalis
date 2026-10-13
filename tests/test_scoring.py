"""The rubric must be bounded, deterministic and monotone in flaw severity."""
import pytest

from backend.detection import Region, Thresholds
from backend.scoring_engine import DIMENSIONS, region_penalty, score_reading

TH = Thresholds()


def _prosodic(ftype, score, span, **extra):
    return Region(ftype, 10.0, 10.0 + span, score, {"baseline_span_s": span, **extra})


def test_no_flaws_scores_100_everywhere():
    result = score_reading([], 100.0)
    assert result["overall"] == 100.0
    assert all(d["score"] == 100.0 for d in result["dimensions"].values())


@pytest.mark.parametrize("ftype,thr", [("pace", TH.pace_log), ("monotone", abs(TH.pitch_log)), ("volume", TH.energy_db)])
def test_penalty_grows_with_severity_and_with_coverage(ftype, thr):
    mild = region_penalty(_prosodic(ftype, 1.2 * thr, 6.0), 100.0).penalty
    harsh = region_penalty(_prosodic(ftype, 2.5 * thr, 6.0), 100.0).penalty
    longer = region_penalty(_prosodic(ftype, 1.2 * thr, 12.0), 100.0).penalty
    assert 0 < mild < harsh and mild < longer


def test_severity_is_capped_so_one_extreme_value_cannot_dominate():
    a = region_penalty(_prosodic("volume", 3 * TH.energy_db, 6.0), 100.0).penalty
    b = region_penalty(_prosodic("volume", 30 * TH.energy_db, 6.0), 100.0).penalty
    assert a == b


def test_pause_and_stumble_penalties():
    short = region_penalty(Region("pause", 1, 2, 0.6, {"gap_s": 0.7}), 100.0).penalty
    long = region_penalty(Region("pause", 1, 4, 2.4, {"gap_s": 2.5}), 100.0).penalty
    assert 0 < short < long
    s1 = region_penalty(Region("stumble", 1, 2, 1.0, {"extra_repeats": 1}), 100.0).penalty
    s3 = region_penalty(Region("stumble", 1, 2, 3.0, {"extra_repeats": 3}), 100.0).penalty
    assert s1 < s3


def test_each_flaw_only_lowers_its_own_dimension():
    result = score_reading([_prosodic("monotone", 2 * abs(TH.pitch_log), 8.0)], 100.0)
    dims = result["dimensions"]
    assert dims["pitch"]["score"] < 100.0
    assert all(dims[k]["score"] == 100.0 for k in DIMENSIONS if k != "pitch")


def test_scores_never_go_below_zero_and_are_deterministic():
    many = [_prosodic("volume", 3 * TH.energy_db, 50.0) for _ in range(5)]
    first, second = score_reading(many, 100.0), score_reading(many, 100.0)
    assert first == second
    assert first["dimensions"]["volume"]["score"] == 0.0
    assert 0.0 <= first["overall"] <= 100.0