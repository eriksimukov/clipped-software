"""Schnelle Signale ueber das ganze VOD: eine Interessantheits-Kurve mit einem Wert je Sekunde."""

from __future__ import annotations

import re

import numpy as np

from clipped.pipeline.common import VodPaths, read_json, write_json
from clipped.pipeline.transcribe import load_wav

SAMPLE_RATE = 16000
# Der Chat reagiert verzoegert: Nachrichten kommen etwa so viele Sekunden nach dem Ereignis.
CHAT_DELAY_SECONDS = 8
BASELINE_BLOCK_SECONDS = 300
RMS_BLOCK_SECONDS = 600
WEIGHTS = {"audio": 0.35, "chat": 0.35, "hype": 0.2, "speech": 0.1}

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


def combine(parts: dict[str, np.ndarray], has_chat: bool) -> tuple[np.ndarray, dict[str, float]]:
    """Gewichtete Summe; ohne Chat wird dessen Gewicht auf die anderen Signale verteilt."""
    weights = {k: v for k, v in WEIGHTS.items() if has_chat or k != "chat"}
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    curve = sum(weights[name] * parts[name] for name in weights)
    return curve, weights


def compute_signals(paths: VodPaths) -> None:
    if paths.signals.exists():
        return
    samples = load_wav(paths.audio)
    segments = read_json(paths.transcript)["segments"]
    messages = read_json(paths.chat)
    loudness = rms_db_per_second(samples)
    del samples
    seconds = len(loudness)

    words, hype = word_activity(segments, seconds)
    has_chat = len(messages) > 0
    parts = {
        "audio": smooth(spike_score(loudness)),
        "chat": smooth(spike_score(chat_rate(messages, seconds))),
        "hype": smooth(np.minimum(hype, 1.0), width=3),
        "speech": smooth(spike_score(words)),
    }
    curve, weights = combine(parts, has_chat)
    write_json(
        paths.signals,
        {
            "has_chat": has_chat,
            "weights": weights,
            "curve": [round(float(v), 3) for v in curve],
            "parts": {k: [round(float(v), 3) for v in parts[k]] for k in weights},
        },
    )
