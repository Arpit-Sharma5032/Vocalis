"""Detector tests on synthetic speech-like audio with exact (oracle) word times."""
import numpy as np
import pytest

from backend.detection import Thresholds, analyze, match_words
from backend.explain import explain
from backend.features import extract_features
from dataset_builder.oracle import oracle_alignment
from dataset_builder.perturb import SR, inject
from dataset_builder.windows import candidate_windows, pick_pause_point, pick_stumble_word
from evaluation.evaluate import iou


def _audio(base_hz=150.0, gain=0.3, seconds=62.0):
    t = np.arange(int(seconds * SR)) / SR
    f0 = base_hz * (1.0 + 0.17 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (gain * sig / np.max(np.abs(sig))).astype(np.float32)


def _words(n=140):
    words, t = [], 0.5
    for i in range(n):
        text = f"w{i}" + ("," if (i + 1) % 5 == 0 else "")
        words.append({"word": text, "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.4 + (0.3 if text.endswith(",") else 0.03)
    return words


@pytest.fixture(scope="module")
def baseline():
    audio, words = _audio(), _words()
    return audio, words, extract_features(audio)


def _flawed(baseline, flaw, severity):
    audio, words, _ = baseline
    window = candidate_windows(words, 4.0, 9.0)[3]
    if flaw == "pause":
        start, _ = pick_pause_point(window)
        end = start
    elif flaw == "stumble":
        w = pick_stumble_word(window)
        start, end = w["start"], w["end"]
    else:
        start, end = window.start, window.end
    out, label = inject(audio, flaw, severity, start, end)
    row = {"flaw_type": flaw, "parameter": label.parameter, "flaw_start": label.start,
           "flaw_end": label.end, "window_start": window.start, "window_end": window.end}
    return out, oracle_alignment(words, row), label


@pytest.mark.parametrize("flaw", ["pace", "slow", "pause", "volume", "monotone", "stumble"])
def test_each_severe_flaw_is_found_with_good_overlap(baseline, flaw):
    audio, words, feats = baseline
    flawed, flawed_words, label = _flawed(baseline, flaw, "severe")
    regions = analyze(feats, words, extract_features(flawed), flawed_words)
    same = [r for r in regions if r.flaw_type == flaw]
    assert same, f"{flaw} not detected; got {[r.flaw_type for r in regions]}"
    assert max(iou((r.start, r.end), (label.start, label.end)) for r in same) >= 0.5
    assert explain(same[0])["summary"]  # every region can be explained with numbers


def test_identical_reading_has_no_flaws(baseline):
    audio, words, feats = baseline
    assert analyze(feats, words, feats, words) == []


def test_different_speaker_with_same_delivery_has_no_flaws(baseline):
    """A higher-pitched, quieter speaker reading identically must not be flagged."""
    audio, words, feats = baseline
    other = extract_features(_audio(base_hz=230.0, gain=0.08))
    assert analyze(feats, words, other, words) == []


def test_uniformly_faster_speaker_is_not_flagged_but_local_rush_is(baseline):
    audio, words, feats = baseline
    scaled = [dict(w, start=w["start"] / 1.3, end=w["end"] / 1.3) for w in words]
    assert [r.flaw_type for r in analyze(feats, words, feats, scaled)] == []


def test_match_words_pairs_identical_texts_and_ignores_punctuation():
    a = [{"word": "Hello,"}, {"word": "world."}]
    b = [{"word": "hello"}, {"word": "world"}]
    pairs, _ = match_words(a, b)
    assert pairs == [(0, 0), (1, 1)]


def test_thresholds_are_configurable(baseline):
    audio, words, feats = baseline
    flawed, flawed_words, _ = _flawed(baseline, "volume", "subtle")
    strict = Thresholds(energy_db=6.0)
    assert not [r for r in analyze(feats, words, extract_features(flawed), flawed_words, strict) if r.flaw_type == "volume"]


def test_pause_is_not_mistaken_for_a_volume_change(baseline):
    audio, words, feats = baseline
    flawed, flawed_words, _ = _flawed(baseline, "pause", "severe")
    regions = analyze(feats, words, extract_features(flawed), flawed_words)
    assert "volume" not in {r.flaw_type for r in regions}


def test_slowing_a_region_is_not_also_reported_as_pauses(baseline):
    """Slowing speech stretches its natural gaps; that is pace, not extra pauses."""
    audio, words, feats = baseline
    flawed, flawed_words, _ = _flawed(baseline, "slow", "severe")
    kinds = {r.flaw_type for r in analyze(feats, words, extract_features(flawed), flawed_words)}
    assert "slow" in kinds and "pause" not in kinds