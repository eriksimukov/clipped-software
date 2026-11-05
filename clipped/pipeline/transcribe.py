"""Transkript mit Wort-Zeitstempeln ueber faster-whisper."""

from __future__ import annotations

import gc
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from clipped.config import Settings
from clipped.pipeline.common import VodPaths, read_json, write_json
from clipped.pipeline.gpu import enable_cuda_dlls

Progress = Callable[[str, float], None]


def load_wav(path: Path) -> np.ndarray:
    """Liest das von ingest erzeugte 16-kHz-Mono-WAV; umgeht den av-Decoder von faster-whisper."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16000 or wav.getsampwidth() != 2:
            raise ValueError(f"Erwarte 16 kHz, Mono, 16 Bit: {path}")
        frames = wav.readframes(wav.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(
    paths: VodPaths,
    settings: Settings,
    progress: Progress = lambda step, value: None,
) -> None:
    """Ueberspringt den Schritt, wenn ein fertiges Transkript vorliegt."""
    if paths.transcript.exists():
        return

    enable_cuda_dlls()
    from faster_whisper import WhisperModel

    duration = read_json(paths.meta)["duration"]
    cfg = settings.transcribe
    model = WhisperModel(
        cfg.model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        download_root=str(settings.paths.data_dir / "models"),
    )
    try:
        raw_segments, _info = model.transcribe(
            load_wav(paths.audio),
            language=settings.clips.language,
            word_timestamps=True,
            vad_filter=True,
            # Verhindert, dass sich Halluzinationen ueber lange Streams fortpflanzen.
            condition_on_previous_text=False,
        )
        segments = []
        for segment in raw_segments:
            segments.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {"start": w.start, "end": w.end, "word": w.word.strip(), "p": w.probability}
                        for w in (segment.words or [])
                    ],
                }
            )
            if duration:
                progress("transcribe", min(segment.end / duration, 1.0))
    finally:
        # Die GPU wird fuer das Sprachmodell gebraucht, also Speicher sofort freigeben.
        del model
        gc.collect()

    write_json(paths.transcript, {"language": settings.clips.language, "segments": segments})
    progress("transcribe", 1.0)
