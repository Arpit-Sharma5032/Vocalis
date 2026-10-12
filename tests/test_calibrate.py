"""Tests for the calibration helpers."""
import math

import pytest

from evaluation.calibrate import GRIDS, f1_score, pick_best


def test_f1_score_basic_cases():
    assert f1_score(10, 0, 0) == 1.0
    assert f1_score(0, 5, 5) == 0.0
    assert f1_score(0, 0, 0) == 0.0
    assert f1_score(5, 5, 5) == pytest.approx(2 * 5 / (2 * 5 + 5 + 5))


def test_ties_go_to_the_more_conservative_threshold():
    scores = {0.2: 0.8, 0.3: 0.8, 0.4: 0.7}
    assert pick_best(scores, "energy_db") == 0.3  # larger = stricter
    pitch = {math.log(0.5): 0.9, math.log(0.7): 0.9}
    assert pick_best(pitch, "pitch_log") == math.log(0.5)  # more negative = stricter


def test_every_grid_has_several_candidates():
    assert all(len(g) >= 4 for g in GRIDS.values())