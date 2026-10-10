"""Tests for the perturbation engine: labels must match what was actually done."""
import numpy as np
import parselmouth
import pytest

from dataset_builder.perturb import (
    SEVERITY,
    SR,
    STUMBLE_GAP_S,
    apply_gain,
    apply_monotone,
    apply_pace,
    apply_pause,
    apply_stumble,
    inject,
)


def _voiced_signal(seconds: float = 3.0) -> np.ndarray:
    """Synthetic voiced sound: harmonic tone whose pitch wobbles around 150 Hz."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = 150.0 + 30.0 * np.sin(2 * np.pi * 1.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (0.3 * sig / np.max(np.abs(sig))).astype(np.float32)


def _pitch_std_semitones(audio: np.ndarray, start: float, end: float) -> float:
    pitch = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR).to_pitch(0.01, 75, 500)
    f0 = pitch.selected_array["frequency"]
    times = pitch.xs()
    voiced = f0[(times >= start) & (times <= end) & (f0 > 0)]
    return float(np.std(12 * np.log2(voiced / np.median(voiced))))


def test_pause_label_matches_inserted_length():
    audio = np.ones(SR * 5, dtype=np.float32) * 0.1
    out, (a, b) = apply_pause(audio, at=2.0, seconds=1.5)
    assert len(out) == len(audio) + int(1.5 * SR)
    assert (a, b) == (2.0, 3.5)


def test_gain_keeps_length_and_lowers_level():
    audio = np.ones(SR * 3, dtype=np.float32) * 0.5
    out, _ = apply_gain(audio, 1.0, 2.0, -6.0)
    assert len(out) == len(audio)
    assert out[SR + 100] < audio[SR + 100]


def test_pace_label_end_matches_output_length():
    audio = _voiced_signal(4.0)
    out, (a, b) = apply_pace(audio, 1.0, 3.0, rate=2.0)
    assert len(out) < len(audio)
    assert b == pytest.approx(a + 1.0, abs=0.01)  # 2 s region at 2x -> 1 s


def test_stumble_label_covers_original_plus_repeats():
    audio = np.ones(SR * 3, dtype=np.float32) * 0.1
    out, (a, b) = apply_stumble(audio, 1.0, 1.4, repeats=2)
    expected_extra = 2 * (0.4 + STUMBLE_GAP_S)
    assert len(out) == pytest.approx(len(audio) + expected_extra * SR, abs=2)
    assert (a, b) == pytest.approx((1.0, 1.4 + expected_extra), abs=0.001)


def test_stumble_inserts_audible_silent_gap_before_repeat():
    audio = np.ones(SR * 3, dtype=np.float32) * 0.1
    out, _ = apply_stumble(audio, 1.0, 1.4, repeats=1)
    mid_gap = int((1.4 + STUMBLE_GAP_S / 2) * SR)
    assert out[mid_gap] == 0.0


def test_monotone_reduces_pitch_variation_inside_region_only():
    audio = _voiced_signal(3.0)
    before = _pitch_std_semitones(audio, 0.8, 2.2)
    out, _ = apply_monotone(audio, 0.5, 2.5, strength=1.0)
    after = _pitch_std_semitones(out, 0.8, 2.2)
    assert len(out) == len(audio)
    assert after < 0.3 * before
    assert np.allclose(out[: int(0.4 * SR)], audio[: int(0.4 * SR)])  # untouched outside


def test_severity_is_monotonic_for_every_flaw():
    """Each flaw's parameter must move further from 'normal' as severity rises."""
    normal = {"pace": 1.0, "slow": 1.0, "pause": 0.0, "volume": 0.0, "monotone": 0.0, "stumble": 0}
    for flaw, levels in SEVERITY.items():
        d = [abs(levels[s] - normal[flaw]) for s in ("subtle", "moderate", "severe")]
        assert d == sorted(d) and len(set(d)) == 3, flaw


def test_inject_rejects_unknown_flaw():
    with pytest.raises(ValueError):
        inject(np.zeros(SR, dtype=np.float32), "whisper", "subtle", 0.0, 1.0)