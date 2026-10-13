"""FastAPI server: reference speeches in, scored delivery reports out.

Run from the project root:
    uvicorn backend.api:app --reload
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline import Paths, analyze_reading, available_speeches

MAX_UPLOAD_BYTES = 60 * 1024 * 1024
MAX_AUDIO_SECONDS = 360

app = FastAPI(title="Vocalis", description="Contrastive speech analytics (educational prototype)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.state.paths = Paths()
app.state.align_fn = None  # None -> WhisperX is loaded on first request; tests inject a stub


def _to_wav16k(src: Path, dst: Path) -> None:
    """Convert any uploaded audio to 16 kHz mono WAV with FFmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "16000", "-t", str(MAX_AUDIO_SECONDS), str(dst)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not dst.exists():
        raise HTTPException(400, "Could not read that audio file. Try WAV, MP3 or M4A.")


@app.get("/api/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/api/speeches")
def speeches() -> list[dict]:
    """Reference speeches the user can read."""
    return available_speeches(app.state.paths)


@app.get("/api/speeches/{speech_id}/audio")
def reference_audio(speech_id: str) -> FileResponse:
    """Stream the reference recording for playback in the dashboard."""
    if speech_id not in {s["id"] for s in available_speeches(app.state.paths)}:
        raise HTTPException(404, "Unknown speech")
    return FileResponse(app.state.paths.raw_dir / f"{speech_id}.wav", media_type="audio/wav")


@app.post("/api/analyze")
async def analyze(
    speech_id: str = Form(...), audio: UploadFile = File(...), transcript: str | None = Form(None),
) -> dict:
    """Analyze an uploaded reading against a reference speech."""
    if speech_id not in {s["id"] for s in available_speeches(app.state.paths)}:
        raise HTTPException(404, "Unknown speech")
    data = await audio.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Audio file is empty or larger than 60 MB.")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"upload{Path(audio.filename or 'a.wav').suffix or '.wav'}"
        src.write_bytes(data)
        wav = Path(tmp) / "reading.wav"
        _to_wav16k(src, wav)
        try:
            return analyze_reading(str(wav), speech_id, transcript, app.state.align_fn, app.state.paths)
        except ValueError as err:
            raise HTTPException(422, str(err)) from err


_dist = Path("frontend/dist")
if _dist.exists():  # serve the built dashboard from the same server
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")