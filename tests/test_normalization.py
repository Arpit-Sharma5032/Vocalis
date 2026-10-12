"""Tests for speaker-agnostic normalization."""
import numpy as np

from backend.normalization import (
    cmvn,
    drop_short_runs,
    fix_octave_jumps,
    hz_to_semitones,
    relative_level_db,
)


def test_semitones_are_zero_at_median_and_twelve_per_octave():
    f0 = np.array([100.0, 100.0, 200.0, 100.0, 100.0])
    st = hz_to_semitones(f0, ref_hz=100.0)
    assert st[0] == 0.0 and abs(st[2] - 12.0) < 1e-9


def test_semitone_contour_is_speaker_independent():
    contour = 150.0 + 20.0 * np.sin(np.linspace(0, 6, 200))
    low, high = hz_to_semitones(contour), hz_to_semitones(contour * 1.8)
    assert np.allclose(low, high)  # a higher voice gives the same movement


def test_octave_jump_is_repaired_and_isolated_glitch_dropped():
    st = np.zeros(60)
    st[30] = 12.0  # tracker jumped an octave up
    out = fix_octave_jumps(st)
    assert abs(out[30]) < 1.0
    glitch = np.full(60, np.nan)
    glitch[30] = 14.0  # a lone voiced frame with no neighbours
    assert np.isnan(fix_octave_jumps(glitch)[30])


def test_relative_level_ignores_overall_recording_gain():
    rng = np.random.default_rng(0)
    db = rng.normal(-30, 8, 5000)
    rel_a, _ = relative_level_db(db)
    rel_b, _ = relative_level_db(db - 17.0)  # same speech recorded 17 dB quieter
    assert np.allclose(rel_a, rel_b)


def test_cmvn_gives_zero_mean_unit_variance():
    x = np.random.default_rng(1).normal(5, 3, (200, 13))
    z = cmvn(x)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-6) and np.allclose(z.std(axis=0), 1, atol=1e-3)


def test_run_of_consecutive_octave_errors_at_end_of_voiced_segment_is_repaired():
    st = np.full(60, np.nan)
    st[20:40] = np.linspace(-2.0, 3.0, 20)
    st[37:40] += 12.0  # three octave-error frames at the end of the run
    out = fix_octave_jumps(st)
    assert np.nanmax(np.abs(out[20:40])) < 5.0


def test_short_voiced_bursts_are_dropped_but_long_runs_kept():
    st = np.full(40, np.nan)
    st[5:8] = 1.0  # 3 frames: too short
    st[15:30] = 0.5  # 15 frames: real speech
    out = drop_short_runs(st, min_len=5)
    assert np.isnan(out[5:8]).all() and np.isfinite(out[15:30]).all()