import numpy as np

from dataset_builder.perturb import SR, apply_gain, apply_pause


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