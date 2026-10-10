"""Programmatic flaw injection with exact ground-truth labels.

Each function returns the modified audio and the flaw region expressed on the
*output* timeline (after any length change), so labels stay exact by construction.

Flaw types:
    pace      speed a region up (rate > 1) using PSOLA, pitch unchanged
    slow      slow a region down (rate < 1) without changing pitch
    pause     insert silence in the middle of a sentence
    volume    change loudness of a region in dB
    monotone  compress the pitch (F0) movement of a region toward flat
    stumble   repeat a word or short phrase, like a verbal stumble
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call

SR = 16000
FADE_S = 0.01  # crossfade length used when splicing processed audio back in
STUMBLE_GAP_S = 0.25  # silence between a word and each repeat (audible hesitation)

# Severity -> parameter. Larger distance from "normal" = more obviously wrong.
SEVERITY = {
    "pace": {"subtle": 1.15, "moderate": 1.50, "severe": 2.00},  # speed-up factor
    "slow": {"subtle": 0.87, "moderate": 0.67, "severe": 0.50},  # slow-down factor
    "pause": {"subtle": 0.6, "moderate": 1.2, "severe": 2.5},  # seconds of silence
    "volume": {"subtle": -3.0, "moderate": -6.0, "severe": -12.0},  # dB change
    "monotone": {"subtle": 0.4, "moderate": 0.7, "severe": 1.0},  # 1.0 = fully flat
    "stumble": {"subtle": 1, "moderate": 2, "severe": 3},  # number of repeats
}


@dataclass
class FlawLabel:
    """Ground truth for one injected flaw (times in seconds, output timeline)."""

    flaw_type: str
    severity: str
    start: float
    end: float
    parameter: float


def _psola_stretch(region: np.ndarray, rate: float) -> np.ndarray:
    """Time-stretch speech by `rate` (>1 faster) using Praat PSOLA, pitch preserved.

    PSOLA works on individual pitch periods, so speech stays far more natural
    than with a phase vocoder, especially at high rates.
    """
    snd = parselmouth.Sound(region.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 75, 500)
    duration_tier = call("Create DurationTier", "stretch", 0.0, snd.xmax)
    call(duration_tier, "Add point", 0.0, 1.0 / rate)  # relative duration: <1 = shorter
    call([duration_tier, manip], "Replace duration tier")
    return call(manip, "Get resynthesis (overlap-add)").values[0].astype(region.dtype)


def apply_pace(
    audio: np.ndarray, start: float, end: float, rate: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Change speed of [start, end] by `rate` (>1 faster, <1 slower), pitch unchanged."""
    s, e = int(start * SR), int(end * SR)
    region = _psola_stretch(audio[s:e], rate)
    out = np.concatenate([audio[:s], region, audio[e:]])
    return out, (start, start + len(region) / SR)


def apply_pause(
    audio: np.ndarray, at: float, seconds: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Insert `seconds` of silence at time `at`."""
    s = int(at * SR)
    silence = np.zeros(int(seconds * SR), dtype=audio.dtype)
    out = np.concatenate([audio[:s], silence, audio[s:]])
    return out, (at, at + seconds)


def apply_gain(
    audio: np.ndarray, start: float, end: float, db: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Change loudness of [start, end] by `db` decibels."""
    out = audio.copy()
    s, e = int(start * SR), int(end * SR)
    out[s:e] = np.clip(out[s:e] * (10.0 ** (db / 20.0)), -1.0, 1.0)
    return out, (start, end)


def _crossfade_splice(original: np.ndarray, processed: np.ndarray, s: int, e: int) -> np.ndarray:
    """Replace original[s:e] with processed[s:e], fading at both edges to avoid clicks."""
    out = original.copy()
    n = min(len(original), len(processed))
    e = min(e, n)
    out[s:e] = processed[s:e]
    fade = min(int(FADE_S * SR), (e - s) // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=original.dtype)
        out[s : s + fade] = original[s : s + fade] * (1 - ramp) + processed[s : s + fade] * ramp
        out[e - fade : e] = processed[e - fade : e] * (1 - ramp[::-1]) + original[e - fade : e] * ramp[::-1]
    return out


def apply_monotone(
    audio: np.ndarray, start: float, end: float, strength: float
) -> tuple[np.ndarray, tuple[float, float]]:
    """Flatten pitch movement in [start, end].

    Each F0 point is pulled toward the region's median in the log (semitone)
    domain: new = median * (f0 / median) ** (1 - strength). strength=1 is fully
    flat; strength=0 leaves the pitch unchanged. Uses Praat's PSOLA resynthesis.
    """
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 75, 500)
    tier = call(manip, "Extract pitch tier")

    n_points = int(call(tier, "Get number of points"))
    times = np.array([call(tier, "Get time from index", i) for i in range(1, n_points + 1)])
    values = np.array([call(tier, "Get value at index", i) for i in range(1, n_points + 1)])

    in_region = (times >= start) & (times <= end)
    if not in_region.any():
        raise ValueError(f"No voiced pitch found between {start}s and {end}s")
    median = np.median(values[in_region])
    new_values = values.copy()
    new_values[in_region] = median * (values[in_region] / median) ** (1.0 - strength)

    new_tier = call("Create PitchTier", "flat", snd.xmin, snd.xmax)
    for t, v in zip(times, new_values):
        call(new_tier, "Add point", float(t), float(v))
    call([new_tier, manip], "Replace pitch tier")
    resynth = call(manip, "Get resynthesis (overlap-add)").values[0].astype(audio.dtype)

    out = _crossfade_splice(audio, resynth, int(start * SR), int(end * SR))
    return out, (start, end)


def apply_stumble(
    audio: np.ndarray, start: float, end: float, repeats: int
) -> tuple[np.ndarray, tuple[float, float]]:
    """Repeat the word(s) in [start, end] `repeats` extra times, like a stumble.

    The label covers the original word plus all repeats.
    """
    s, e = int(start * SR), int(end * SR)
    unit = audio[s:e].copy()
    fade = min(int(FADE_S * SR), len(unit) // 4)  # soften the cut edges to avoid clicks
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, dtype=unit.dtype)
        unit[:fade] *= ramp
        unit[-fade:] *= ramp[::-1]
    gap = np.zeros(int(STUMBLE_GAP_S * SR), dtype=audio.dtype)
    inserted = np.concatenate([np.concatenate([gap, unit]) for _ in range(repeats)])
    out = np.concatenate([audio[:e], inserted, audio[e:]])
    return out, (start, end + len(inserted) / SR)


def inject(
    audio: np.ndarray, flaw: str, severity: str, start: float, end: float
) -> tuple[np.ndarray, FlawLabel]:
    """Inject one flaw of a given severity into the window [start, end]."""
    if flaw not in SEVERITY:
        raise ValueError(f"Unknown flaw type: {flaw}")
    param = SEVERITY[flaw][severity]
    if flaw in ("pace", "slow"):
        out, (a, b) = apply_pace(audio, start, end, param)
    elif flaw == "pause":
        out, (a, b) = apply_pause(audio, start, param)
    elif flaw == "volume":
        out, (a, b) = apply_gain(audio, start, end, param)
    elif flaw == "monotone":
        out, (a, b) = apply_monotone(audio, start, end, param)
    else:  # stumble
        out, (a, b) = apply_stumble(audio, start, end, int(param))
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