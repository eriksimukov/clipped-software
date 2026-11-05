"""Schnelle Signale ueber das ganze VOD: eine Interessantheits-Kurve mit einem Wert je Sekunde."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

import numpy as np

from clipped.pipeline.common import VideoInfo, VodPaths, read_json, write_json
from clipped.pipeline.transcribe import load_wav

SAMPLE_RATE = 16000
# Der Chat reagiert verzoegert: Nachrichten kommen etwa so viele Sekunden nach dem Ereignis.
CHAT_DELAY_SECONDS = 8
BASELINE_BLOCK_SECONDS = 300
RMS_BLOCK_SECONDS = 600
WEIGHTS = {"cam": 0.45, "chat": 0.3, "audio": 0.1, "hype": 0.1, "speech": 0.05}
MOTION_FPS = 4
MOTION_WIDTH, MOTION_HEIGHT = 160, 90
CAM_SIDE_MARGIN = 0.15
SCENE_CUT_MIN = 20.0
SCENE_CUT_FACTOR = 4.0

# Nur eindeutige Reaktionen: Fuellwoerter wie "bro" oder "wait" fallen in Streams staendig.
HYPE_RE = re.compile(
    r"\b(oh my god|omg|no way|what the|holy (shit|crap|moly)|oh shit|let'?s go+|insane|craziest|"
    r"wtf|are you kidding|are you serious|i can'?t believe|did you see that)\b",
    re.IGNORECASE,
)


def rms_db_per_second(samples: np.ndarray, rate: int = SAMPLE_RATE) -> np.ndarray:
    seconds = len(samples) // rate
    if seconds == 0:
        return np.zeros(0)
    frames = samples[: seconds * rate].reshape(seconds, rate)
    rms = np.empty(seconds)
    # Abschnittsweise, damit ein langes VOD nicht kurzzeitig mehrere GB zusaetzlich belegt.
    for first in range(0, seconds, RMS_BLOCK_SECONDS):
        block = frames[first : first + RMS_BLOCK_SECONDS]
        rms[first : first + RMS_BLOCK_SECONDS] = np.sqrt(np.mean(block**2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-5))


def spike_score(values: np.ndarray, block: int = BASELINE_BLOCK_SECONDS) -> np.ndarray:
    """0 bis 1: wie stark ein Wert ueber dem Normalniveau seines Abschnitts liegt.

    Blockweise, weil Streamer leiser und lauter werden und ein globaler Massstab
    nur die lautesten Stunden finden wuerde.
    """
    result = np.zeros(len(values), dtype=np.float64)
    for start in range(0, len(values), block):
        part = values[start : start + block]
        median = np.median(part)
        spread = 1.4826 * np.median(np.abs(part - median))
        if spread < 1e-6:
            # Ohne Streuung (z. B. fast leerer Chat) zaehlt die Abweichung vom Mittelwert.
            spread = max(float(np.std(part)), 1e-6)
        result[start : start + block] = np.clip((part - median) / spread, 0, 5) / 5
    return result


def smooth(values: np.ndarray, width: int = 5) -> np.ndarray:
    # Bei kuerzeren Eingaben wuerde mode="same" ein laengeres Array als die Eingabe liefern.
    if len(values) < width:
        return values
    return np.convolve(values, np.ones(width) / width, mode="same")


def chat_rate(messages: list[dict], seconds: int) -> np.ndarray:
    counts = np.zeros(seconds)
    for message in messages:
        second = int(message["t"]) - CHAT_DELAY_SECONDS
        if 0 <= second < seconds:
            counts[second] += 1
    return counts


def word_activity(segments: list[dict], seconds: int) -> tuple[np.ndarray, np.ndarray]:
    """Gibt Woerter je Sekunde und Treffer von Hype-Ausdruecken je Sekunde zurueck."""
    words = np.zeros(seconds)
    hype = np.zeros(seconds)
    for segment in segments:
        for word in segment.get("words", []):
            second = int(word["start"])
            if 0 <= second < seconds:
                words[second] += 1
        if HYPE_RE.search(segment["text"]):
            start = max(int(segment["start"]), 0)
            end = min(int(segment["end"]) + 1, seconds)
            hype[start:end] += 1
    return words, hype


def combine(parts: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    """Gewichtete Summe; fehlende Signale geben ihr Gewicht an die anderen ab."""
    weights = {k: v for k, v in WEIGHTS.items() if k in parts}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    curve = sum(weights[name] * parts[name] for name in weights)
    return curve, weights


def _regions(rect: list[int], info: VideoInfo) -> tuple[np.ndarray, np.ndarray]:
    """Masken im kleinen Bild: Mitte der Facecam und alles Uebrige als Vergleich."""
    scale_x, scale_y = MOTION_WIDTH / info.width, MOTION_HEIGHT / info.height
    x, y, w, h = rect
    cam = np.zeros((MOTION_HEIGHT, MOTION_WIDTH), dtype=bool)
    # Der Rand zeigt bei freigestellten Streamern das Spiel, die Mitte zeigt die Person.
    margin = w * CAM_SIDE_MARGIN
    cam[
        int(y * scale_y) : max(int((y + h) * scale_y), int(y * scale_y) + 1),
        int((x + margin) * scale_x) : max(
            int((x + w - margin) * scale_x), int((x + margin) * scale_x) + 1
        ),
    ] = True
    outside = np.ones((MOTION_HEIGHT, MOTION_WIDTH), dtype=bool)
    outside[
        int(y * scale_y) : int((y + h) * scale_y) + 1, int(x * scale_x) : int((x + w) * scale_x) + 1
    ] = False
    return cam, outside


def cam_motion(
    video: str,
    rect: list[int],
    info: VideoInfo,
    progress: Callable[[float], None] = lambda value: None,
) -> np.ndarray:
    """Wie stark sich der Streamer bewegt, je Sekunde. Das findet Reaktionen besser als Lautstaerke.

    Im Jynxzi-Test lag reiner Trailer-Dialog bei 11 bis 14, echte Reaktionen bei 17 bis 26.
    """
    cam_mask, outside_mask = _regions(rect, info)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video,
        "-vf",
        f"fps={MOTION_FPS},scale={MOTION_WIDTH}:{MOTION_HEIGHT},format=gray",
        "-f",
        "rawvideo",
        "-",
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    size = MOTION_WIDTH * MOTION_HEIGHT
    previous = None
    cam_values: list[float] = []
    outside_values: list[float] = []
    try:
        while True:
            raw = process.stdout.read(size)
            if len(raw) < size:
                break
            frame = (
                np.frombuffer(raw, dtype=np.uint8)
                .reshape(MOTION_HEIGHT, MOTION_WIDTH)
                .astype(np.int16)
            )
            if previous is not None:
                difference = np.abs(frame - previous)
                cam_values.append(float(difference[cam_mask].mean()))
                outside_values.append(float(difference[outside_mask].mean()))
            previous = frame
            if len(cam_values) % (MOTION_FPS * 60) == 0 and info.duration:
                progress(min(len(cam_values) / (MOTION_FPS * info.duration), 1.0))
    finally:
        process.stdout.close()
        process.wait()
    return motion_per_second(np.array(cam_values), np.array(outside_values))


def motion_per_second(cam: np.ndarray, outside: np.ndarray) -> np.ndarray:
    """Szenenschnitte im Spiel aendern bei freigestellten Streamern auch das Cam-Bild."""
    if len(cam) == 0:
        return np.zeros(0)
    limit = max(SCENE_CUT_MIN, SCENE_CUT_FACTOR * float(np.median(outside)))
    cleaned = np.where(outside > limit, np.median(cam), cam)
    return np.array(
        [cleaned[i : i + MOTION_FPS].mean() for i in range(0, len(cleaned), MOTION_FPS)]
    )


def compute_signals(
    paths: VodPaths,
    cam_rect: list[int] | None = None,
    info: VideoInfo | None = None,
    progress: Callable[[float], None] = lambda value: None,
) -> None:
    if paths.signals.exists():
        return
    samples = load_wav(paths.audio)
    segments = read_json(paths.transcript)["segments"]
    messages = read_json(paths.chat)
    loudness = rms_db_per_second(samples)
    del samples
    seconds = len(loudness)

    words, hype = word_activity(segments, seconds)
    parts = {
        "audio": smooth(spike_score(loudness)),
        "hype": smooth(np.minimum(hype, 1.0), width=3),
        "speech": smooth(spike_score(words)),
    }
    has_chat = len(messages) > 0
    if has_chat:
        parts["chat"] = smooth(spike_score(chat_rate(messages, seconds)))
    if cam_rect and info:
        motion = cam_motion(read_json(paths.meta)["video"], cam_rect, info, progress)
        fitted = np.zeros(seconds)
        fitted[: min(len(motion), seconds)] = motion[:seconds]
        parts["cam"] = smooth(spike_score(fitted), width=3)
    curve, weights = combine(parts)
    write_json(
        paths.signals,
        {
            "has_chat": has_chat,
            "weights": weights,
            "curve": [round(float(v), 3) for v in curve],
            "reaction": [round(float(v), 3) for v in parts["cam"]] if "cam" in parts else [],
            "parts": {k: [round(float(v), 3) for v in parts[k]] for k in weights},
        },
    )
