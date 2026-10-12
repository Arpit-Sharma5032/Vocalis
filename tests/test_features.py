"""Tests for frame-level features, pause detection and word statistics."""
import numpy as np

from backend.features import SR, extract_features, find_pauses, word_features


def _voiced(seconds=3.0, base=150.0, wobble=25.0, gain=0.3):
    t = np.arange(int(seconds * SR)) / SR
    f0 = base + wobble * np.sin(2 * np.pi * 1.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (gain * sig / np.max(np.abs(sig))).astype(np.float32)


def test_all_frame_features_share_one_grid():
    f = extract_features(_voiced())
    n = len(f.times)
    assert all(len(x) == n for x in (f.f0_hz, f.f0_st, f.energy_db, f.energy_rel_db, f.mfcc))
    assert abs(f.median_f0_hz - 150.0) < 8.0


def test_gain_changes_absolute_but_not_relative_energy():
    quiet, loud = extract_features(_voiced(gain=0.1)), extract_features(_voiced(gain=0.4))
    assert loud.level_db - quiet.level_db > 10
    assert np.allclose(quiet.energy_rel_db, loud.energy_rel_db, atol=0.5)


def test_pitch_movement_is_the_same_for_a_higher_voice():
    low = extract_features(_voiced(base=130.0, wobble=130.0 * 0.17))
    high = extract_features(_voiced(base=210.0, wobble=210.0 * 0.17))
    std = lambda f: np.nanstd(f.f0_st)
    assert abs(std(low) - std(high)) < 0.4


def test_inserted_silence_is_found_as_a_pause_with_the_right_length():
    voiced = _voiced(2.0)
    audio = np.concatenate([voiced, np.zeros(int(1.2 * SR), dtype=np.float32), voiced])
    pauses = find_pauses(extract_features(audio), min_len_s=0.5)
    assert len(pauses) == 1
    start, end = pauses[0]
    assert abs(start - 2.0) < 0.1 and abs((end - start) - 1.2) < 0.15


def test_word_features_summarize_frames_inside_each_word():
    f = extract_features(_voiced(3.0))
    words = [{"word": "a", "start": 0.2, "end": 1.0}, {"word": "b", "start": 1.5, "end": 2.5}]
    df = word_features(f, words)
    assert list(df["word"]) == ["a", "b"]
    assert df["gap_before"].tolist() == [0.0, 0.5]
    assert (df["f0_std_st"] > 0.5).all() and (df["voiced_frac"] > 0.8).all()