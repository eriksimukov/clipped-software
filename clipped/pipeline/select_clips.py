"""Fuehrt GTA-6-Filter, Facecam, Signale, Moment-Suche und Bewertung zusammen."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np

from clipped.config import Settings
from clipped.pipeline.candidates import build_candidates, candidates_from_moments
from clipped.pipeline.censor import contains_slur
from clipped.pipeline.common import VodPaths, probe_video, read_json, write_json
from clipped.pipeline.facecam import detect_cam, sample_times
from clipped.pipeline.lines import build_lines, reaction_flags
from clipped.pipeline.llm import (
    MIN_GTA6_RELEVANCE,
    SLUR_SCORE_FACTOR,
    assess,
    ensure_running,
    final_score,
    streamer_talk,
    unload,
)
from clipped.pipeline.moments import scan_moments
from clipped.pipeline.signals import compute_signals
from clipped.pipeline.topic_filter import build_mask

Progress = Callable[[str, float], None]

# Ab hier gilt eine Zeile als sichtbare Reaktion des Streamers.
REACTION_FLAG_THRESHOLD = 0.45
# Findet das Modell fast nichts, ergaenzen laute oder bewegte Stellen die Liste.
MIN_MOMENTS = 3
FALLBACK_CANDIDATES = 8
FALLBACK_STRENGTH = 3


def _find_candidates(paths: VodPaths, settings: Settings, progress: Progress) -> list[dict]:
    meta = read_json(paths.meta)
    segments = read_json(paths.transcript)["segments"]
    seconds = int(meta["duration"]) + 1
    mask, reason, confirmed = build_mask(meta, segments, seconds)
    meta["topic_reason"] = reason
    meta["topic_seconds"] = int(mask.sum())
    meta["topic_confirmed"] = confirmed
    if not mask.any():
        write_json(paths.meta, meta)
        return []

    progress("signals", 0.0)
    info = probe_video(Path(meta["video"]))
    if "cam" not in meta:
        allowed = np.flatnonzero(mask)
        times = sample_times(float(allowed[0]), float(allowed[-1]))
        meta["cam"] = detect_cam(meta["video"], times, info)
    write_json(paths.meta, meta)

    compute_signals(paths, (meta["cam"] or {}).get("rect"), info, lambda v: progress("signals", v))
    signals = read_json(paths.signals)
    curve = signals["curve"]
    reaction = signals.get("reaction") or []

    if paths.moments.exists():
        moments = read_json(paths.moments)
        lines = [
            line for line in build_lines(segments) if mask[min(int(line["start"]), seconds - 1)]
        ]
    else:
        progress("scan", 0.0)
        ensure_running(settings.llm)
        lines = [
            line for line in build_lines(segments) if mask[min(int(line["start"]), seconds - 1)]
        ]
        flags = reaction_flags(lines, reaction, REACTION_FLAG_THRESHOLD)
        moments = scan_moments(lines, flags, settings.llm, lambda v: progress("scan", v))
        write_json(paths.moments, moments)

    candidates = candidates_from_moments(
        moments,
        lines,
        curve,
        reaction,
        settings.clips.min_seconds,
        settings.clips.max_seconds,
        settings.clips.candidates_per_vod,
    )
    if len(candidates) < MIN_MOMENTS:
        fallback = build_candidates(
            segments,
            np.array(curve),
            mask[: len(curve)],
            settings.clips.min_seconds,
            settings.clips.max_seconds,
            FALLBACK_CANDIDATES,
        )
        for extra in fallback:
            if all(extra["end"] <= c["start"] or extra["start"] >= c["end"] for c in candidates):
                extra.update(
                    strength=FALLBACK_STRENGTH, reaction_score=extra["signal_score"], story=""
                )
                candidates.append(extra)
    for candidate in candidates:
        # Aus der Startzeit, damit eine neue Auswahl keine fertigen Clips ueberschreibt.
        candidate["id"] = f"c{int(candidate['start']):06d}"
        candidate["status"] = "pending"
    return candidates


def select_clips(
    paths: VodPaths, settings: Settings, progress: Progress = lambda step, value: None
) -> list[dict]:
    """Speichert nach jedem Kandidaten, ein Abbruch verliert also keine Bewertungen."""
    if paths.candidates.exists():
        candidates = read_json(paths.candidates)
    else:
        candidates = _find_candidates(paths, settings, progress)
        write_json(paths.candidates, candidates)
    meta = read_json(paths.meta)
    has_cam = bool((meta.get("cam") or {}).get("rect"))

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
                energy = candidate["reaction_score"] if has_cam else candidate["signal_score"]
                candidate["score"] = final_score(
                    result, energy, candidate["talk_score"], candidate.get("strength", 5)
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
