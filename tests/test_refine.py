"""Silence-aware word boundary refinement."""
import numpy as np

from backend.features import SR, extract_features, refine_word_times


def _voiced(seconds, gain=0.3):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 150.0 * (1.0 + 0.17 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum(np.sin(k * phase) / k for k in range(1, 6))
    return (gain * sig / np.max(np.abs(sig))).astype(np.float32)


def _with_silence(before=3.0, silence=1.5, after=3.0):
    audio = np.concatenate([_voiced(before), np.zeros(int(silence * SR), np.float32), _voiced(after)])
    return extract_features(audio)


def test_word_that_swallowed_a_silence_is_cut_back_to_its_voiced_part():
    feats = _with_silence()  # silence from 3.0 to 4.5 s
    words = [{"word": "swallowed", "start": 2.2, "end": 5.5}]  # aligner stretched it over the silence
    out = refine_word_times(feats, words)[0]
    # voiced part before (2.2-3.0 = 0.8 s) vs after (4.5-5.5 = 1.0 s): the longer one is kept
    assert abs(out["start"] - 4.5) < 0.05 and abs(out["end"] - 5.5) < 0.05


def test_normal_words_are_left_exactly_alone():
    feats = _with_silence()
    words = [{"word": "a", "start": 0.5, "end": 1.2}, {"word": "b", "start": 4.7, "end": 5.4}]
    assert refine_word_times(feats, words) == words


def test_word_with_only_a_short_internal_dip_is_not_cut():
    audio = _voiced(4.0)
    audio[int(2.0 * SR): int(2.1 * SR)] = 0.0  # 100 ms dip, shorter than the 200 ms rule
    feats = extract_features(audio)
    words = [{"word": "x", "start": 1.0, "end": 3.0}]
    assert refine_word_times(feats, words) == words