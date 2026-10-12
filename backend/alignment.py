"""Forced alignment of a known transcript to audio using WhisperX.

Usage:
    python backend/alignment.py data/raw_audio/test_clip.wav data/transcripts/test_clip.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import whisperx

SAMPLE_RATE = 16000
DEVICE = "cpu"


def load_aligner(language: str = "en"):
    """Load the alignment model once so it can be reused for many clips."""
    return whisperx.load_align_model(language_code=language, device=DEVICE)


def align_with_model(audio_path: str, transcript: str, model, metadata) -> list[dict]:
    """Align a known transcript to audio with an already-loaded model.

    Returns a list of dicts with keys 'word', 'start', 'end' (seconds).
    """
    audio = whisperx.load_audio(audio_path)
    duration = len(audio) / SAMPLE_RATE

    # Treat the whole transcript as one segment spanning the full audio.
    segments = [{"text": transcript.strip(), "start": 0.0, "end": duration}]
    result = whisperx.align(
        segments, model, metadata, audio, DEVICE, return_char_alignments=False
    )
    return [
        {"word": w["word"], "start": w.get("start"), "end": w.get("end")}
        for w in result["word_segments"]
    ]


def align_transcript(audio_path: str, transcript: str, language: str = "en") -> list[dict]:
    """Align a known transcript to audio (loads the model for this single call)."""
    model, metadata = load_aligner(language)
    return align_with_model(audio_path, transcript, model, metadata)


def main() -> None:
    """Run alignment from the command line and save a JSON file."""
    if len(sys.argv) != 3:
        sys.exit("Usage: python backend/alignment.py <audio_file> <transcript_file>")
    audio_path, transcript_path = sys.argv[1], sys.argv[2]
    transcript = Path(transcript_path).read_text(encoding="utf-8")

    words = align_transcript(audio_path, transcript)

    out_path = Path("data/processed_metrics") / (Path(audio_path).stem + "_alignment.json")
    out_path.write_text(json.dumps(words, indent=2), encoding="utf-8")

    for w in words:
        start = "?" if w["start"] is None else f"{w['start']:.2f}"
        end = "?" if w["end"] is None else f"{w['end']:.2f}"
        print(f"{start:>7} -> {end:>7}  {w['word']}")
    print(f"\nSaved {len(words)} words to {out_path}")


if __name__ == "__main__":
    main()