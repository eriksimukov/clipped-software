"""Fuehrt Signale, GTA-6-Filter, Fensterbildung und Bewertung zusammen."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np

from clipped.config import Settings
from clipped.pipeline.candidates import build_candidates
from clipped.pipeline.censor import contains_slur
from clipped.pipeline.common import VodPaths, probe_video, read_json, write_json
from clipped.pipeline.facecam import detect_cam, sample_times
from clipped.pipeline.llm import (
    MIN_GTA6_RELEVANCE,
    SLUR_SCORE_FACTOR,
    assess,
    ensure_running,
    final_score,
    streamer_talk,
    unload,
)
from clipped.pipeline.signals import compute_signals
from clipped.pipeline.topic_filter import build_mask

Progress = Callable[[str, float], None]


def select_clips(
    paths: VodPaths, settings: Settings, progress: Progress = lambda step, value: None
) -> list[dict]:
    """Speichert nach jedem Kandidaten, ein Abbruch verliert also keine Bewertungen."""
    progress("signals", 0.0)
    compute_signals(paths)
    meta = read_json(paths.meta)

    if paths.candidates.exists():
        candidates = read_json(paths.candidates)
    else:
        signals = read_json(paths.signals)
        curve = np.array(signals["curve"])
        segments = read_json(paths.transcript)["segments"]
        mask, reason, confirmed = build_mask(meta, segments, len(curve))
        meta["topic_reason"] = reason
        meta["topic_seconds"] = int(mask.sum())
        meta["topic_confirmed"] = confirmed
        if "cam" not in meta and mask.any():
            allowed = np.flatnonzero(mask)
            times = sample_times(float(allowed[0]), float(allowed[-1]))
            meta["cam"] = detect_cam(meta["video"], times, probe_video(Path(meta["video"])))
        write_json(paths.meta, meta)
        candidates = build_candidates(
            segments,
            curve,
            mask,
            settings.clips.min_seconds,
            settings.clips.max_seconds,
            settings.clips.candidates_per_vod,
        )
        for candidate in candidates:
            # Aus der Startzeit, damit eine neue Auswahl keine fertigen Clips ueberschreibt.
            candidate["id"] = f"c{int(candidate['start']):06d}"
            candidate["status"] = "pending"
        write_json(paths.candidates, candidates)

    if any(candidate["status"] == "pending" for candidate in candidates):
        progress("assess", 0.0)
        ensure_running(settings.llm)

    try:
        for index, candidate in enumerate(candidates):
            if candidate["status"] != "pending":
                continue
            try:
                result = assess(
                    candidate,
                    meta.get("streamer", ""),
                    settings.llm,
                    section_confirmed=bool(meta.get("topic_confirmed")),
                )
            except ValueError as exc:
                # Ein Ausreisser soll nicht den ganzen Lauf stoppen; bleibt sichtbar in der Liste.
                candidate.update(status="failed", error=str(exc), score=0.0)
            else:
                candidate["assessment"] = result.model_dump()
                candidate["talk_score"] = streamer_talk(candidate["text"])
                candidate["score"] = final_score(
                    result, candidate["signal_score"], candidate["talk_score"]
                )
                if contains_slur(candidate["text"]):
                    candidate["warnings"] = ["slur"]
                    candidate["score"] = round(candidate["score"] * SLUR_SCORE_FACTOR, 1)
                # Der vom Nutzer gesetzte Zeitbereich steht ueber der Einschaetzung des Modells.
                manual = bool(meta.get("time_range"))
                relevant = manual or result.gta6_relevance >= MIN_GTA6_RELEVANCE
                candidate["status"] = "assessed" if relevant else "off_topic"
            write_json(paths.candidates, candidates)
            progress("assess", (index + 1) / len(candidates))
    finally:
        try:
            unload(settings.llm)
        except httpx.HTTPError:
            pass

    return sorted(candidates, key=lambda c: c.get("score", 0), reverse=True)
