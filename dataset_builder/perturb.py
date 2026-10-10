"""Programmatic flaw injection with exact ground-truth labels.

Each function returns the modified audio and a label whose start/end are
expressed on the *output* timeline (after any length change).
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000

# Severity -> parameter. Larger = more obviously wrong.
SEVERITY = {
    "pace": {"subtle": 1.10, "moderate": 1.30, "severe": 1.60},   # speed-up factor
    "pause": {"subtle": 0.6, "moderate": 1.2, "severe": 2.5},     # seconds of silence
    "volume": {"subtle": -3.0, "moderate": -6.0, "severe": -12.0},  # dB change
}


@dataclass
class FlawLabel:
    """Ground truth for one injected flaw (times in seconds, output timeline)."""

    flaw_type: str
    severity: str
    start: float
    end: float
    parameter: float


def apply_pace(audio: np.ndarray, start: float, end: float, rate: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Speed up (rate > 1) the region [start, end] without changing pitch."""
    s, e = int(start * SR), int(end * SR)
    region = librosa.effects.time_stretch(audio[s:e], rate=rate)
    out = np.concatenate([audio[:s], region, audio[e:]])
    return out, (start, start + len(region) / SR)


def apply_pause(audio: np.ndarray, at: float, seconds: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Insert `seconds` of silence at time `at`."""
    s = int(at * SR)
    silence = np.zeros(int(seconds * SR), dtype=audio.dtype)
    out = np.concatenate([audio[:s], silence, audio[s:]])
    return out, (at, at + seconds)


def apply_gain(audio: np.ndarray, start: float, end: float, db: float) -> tuple[np.ndarray, tuple[float, float]]:
    """Change loudness of [start, end] by `db` decibels."""
    out = audio.copy()
    s, e = int(start * SR), int(end * SR)
    out[s:e] = np.clip(out[s:e] * (10.0 ** (db / 20.0)), -1.0, 1.0)
    return out, (start, end)


def inject(audio: np.ndarray, flaw: str, severity: str, start: float, end: float) -> tuple[np.ndarray, FlawLabel]:
    """Inject one flaw of a given severity into the window [start, end]."""
    param = SEVERITY[flaw][severity]
    if flaw == "pace":
        out, (a, b) = apply_pace(audio, start, end, param)
    elif flaw == "pause":
        out, (a, b) = apply_pause(audio, start, param)
    elif flaw == "volume":
        out, (a, b) = apply_gain(audio, start, end, param)
    else:
        raise ValueError(f"Unknown flaw type: {flaw}")
    return out, FlawLabel(flaw, severity, round(a, 3), round(b, 3), param)


def main() -> None:
    """CLI: python dataset_builder/perturb.py <wav> <flaw> <severity> <start_s> <end_s>"""
    if len(sys.argv) != 6:
        sys.exit(main.__doc__)
    wav, flaw, severity, start, end = sys.argv[1:]
    audio, _ = librosa.load(wav, sr=SR, mono=True)
    out, label = inject(audio, flaw, severity, float(start), float(end))
    stem = Path(wav).stem
    out_wav = Path("data/perturbed") / f"{stem}__{flaw}_{severity}.wav"
    sf.write(out_wav, out, SR)
    out_wav.with_suffix(".json").write_text(json.dumps(asdict(label), indent=2), encoding="utf-8")
    print(f"Wrote {out_wav} with label {label}")


if __name__ == "__main__":
    main()