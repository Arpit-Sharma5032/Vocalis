"""Turn a detected region into a plain-language causal explanation with its numbers."""
from __future__ import annotations

from backend.detection import Region

ADVICE = {
    "pace": "Slow down here so listeners can follow; rushed phrases lose emphasis.",
    "slow": "Pick up the tempo; sustained slowness can sound hesitant or flat.",
    "monotone": "Vary your pitch on key words; rise or fall on the ideas you want remembered.",
    "volume": "Keep a steady speaking level; project the ending of phrases as well as the start.",
    "pause": "Pause at punctuation or between ideas, not mid-phrase.",
    "stumble": "Plan the phrase ahead; a clean restart is better than repeating the word.",
}


def explain(region: Region) -> dict[str, str]:
    """Return title, a numeric summary of the acoustic deviation, and advice."""
    d, t = region.detail, region.flaw_type
    span = f"{region.start:.2f}-{region.end:.2f} s"
    if t in ("pace", "slow"):
        word = "faster" if t == "pace" else "slower"
        factor = 1 / d["local_ratio"] if t == "pace" else d["local_ratio"]
        title = "Rushed delivery" if t == "pace" else "Unusually slow delivery"
        summary = (
            f"Words took {d['participant_ms_per_word']:.0f} ms each against {d['baseline_ms_per_word']:.0f} ms "
            f"in the reference, about {factor:.2f}x {word}. Your typical ratio over the whole speech is "
            f"{d['typical_ratio']:.2f}, so this stretch departs from your own normal pace."
        )
    elif t == "monotone":
        title = "Flat pitch (monotone)"
        summary = (
            f"Pitch spread (P10-P90) was {d['spread_participant_st']:.1f} semitones against "
            f"{d['spread_baseline_st']:.1f} in the reference, a ratio of {d['ratio']:.2f}. "
            f"Your typical ratio over the whole speech is {d['typical_ratio']:.2f}."
        )
    elif t == "volume":
        direction = "quieter" if d["delta_db"] < 0 else "louder"
        title = "Volume dropped" if d["delta_db"] < 0 else "Volume too high"
        summary = (
            f"Speech level was {abs(d['delta_db']):.1f} dB {direction} than the reference for the same words "
            f"(your typical offset is {d['typical_delta_db']:+.1f} dB)."
        )
    elif t == "pause":
        title = "Unexpected pause"
        summary = (
            f"A {d['gap_s']:.2f} s silence between '{d['before']}' and '{d['after']}', where the reference "
            f"has about {d['expected_gap_s']:.2f} s."
        )
    elif t == "stumble":
        title = "Repeated word"
        summary = f"'{d['word']}' was repeated {d['extra_repeats']} extra time(s)."
    else:
        raise ValueError(f"Unknown flaw type: {t}")
    return {"title": title, "time": span, "summary": summary, "advice": ADVICE[t]}