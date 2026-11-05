"""Prueft, ob alles vorhanden ist, was Clipped zum Laufen braucht.

Aufruf: uv run python scripts/doctor.py
Exit-Code 0 nur, wenn alle Pflichtpruefungen bestanden sind.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipped.config import load_settings  # noqa: E402


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
    )


def check_ffmpeg() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "ffmpeg nicht im PATH"
    filters = run(["ffmpeg", "-hide_banner", "-filters"]).stdout
    if " ass " not in filters:
        return False, "ffmpeg ohne libass (ass-Filter fehlt)"
    return True, "ffmpeg mit libass"


def check_nvenc(encoder: str) -> tuple[bool, str]:
    """Kodiert wirklich ein paar Frames, denn die Encoder-Liste allein beweist nichts."""
    result = run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.2",
            "-c:v", encoder, "-f", "null", "-",
        ]
    )
    if result.returncode != 0:
        return False, f"{encoder} schlaegt fehl: {result.stderr.strip()[:200]}"
    return True, f"{encoder} kodiert"


def check_cuda(settings) -> tuple[bool, str]:
    """Laedt das kleinste Whisper-Modell auf der GPU und transkribiert eine Sekunde Stille."""
    if settings.transcribe.device != "cuda":
        return True, f"Transkription laeuft auf {settings.transcribe.device}"
    try:
        import numpy as np

        from clipped.pipeline.gpu import enable_cuda_dlls

        enable_cuda_dlls()
        from faster_whisper import WhisperModel

        model = WhisperModel(
            "tiny",
            device="cuda",
            compute_type=settings.transcribe.compute_type,
            download_root=str(settings.paths.data_dir / "models"),
        )
        segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en")
        list(segments)
    except Exception as exc:  # noqa: BLE001 - jede Ursache soll im Bericht landen
        return False, f"CUDA/faster-whisper: {exc}"
    return True, "faster-whisper rechnet auf der GPU"


def check_ollama(settings) -> tuple[bool, str]:
    import httpx

    try:
        response = httpx.get(f"{settings.llm.host}/api/tags", timeout=5)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        return False, f"Ollama nicht erreichbar ({exc}); starte 'ollama serve'"
    names = [m["name"] for m in response.json().get("models", [])]
    if settings.llm.model not in names:
        return False, f"Modell fehlt: 'ollama pull {settings.llm.model}' (vorhanden: {names})"
    return True, f"Ollama mit {settings.llm.model}"


def check_data_dir(settings) -> tuple[bool, str]:
    data_dir = settings.paths.data_dir
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(data_dir).free / 1024**3
    except OSError as exc:
        return False, f"Datenordner {data_dir}: {exc}"
    return free_gb >= 50, f"{data_dir} mit {free_gb:.0f} GB frei (mindestens 50 GB empfohlen)"


def main() -> int:
    settings = load_settings()
    checks = {
        "FFmpeg": check_ffmpeg(),
        "NVENC": check_nvenc(settings.render.video_encoder),
        "CUDA": check_cuda(settings),
        "Ollama": check_ollama(settings),
        "Daten": check_data_dir(settings),
    }
    for name, (ok, detail) in checks.items():
        print(f"[{'OK' if ok else 'FEHLT'}] {name}: {detail}")
    return 0 if all(ok for ok, _ in checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
