"""Arbeitet Jobs nacheinander ab und schreibt Fortschritt und Ergebnisse in die Datenbank."""

from __future__ import annotations

import shutil
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from clipped.config import Settings
from clipped.db.store import Store
from clipped.pipeline.common import VideoInfo, VodPaths, probe_video, read_json, write_json
from clipped.pipeline.export import export_paths, export_text
from clipped.pipeline.ingest import ingest, set_time_range
from clipped.pipeline.render import render_clip
from clipped.pipeline.select_clips import select_clips
from clipped.pipeline.spec import ClipSpec, spec_from_candidate
from clipped.pipeline.transcribe import transcribe

PROGRESS_INTERVAL_SECONDS = 0.5


class JobCancelled(Exception):
    pass


def absolute_cam(relative: list[float], info: VideoInfo) -> list[int]:
    x, y, w, h = relative
    return [
        round(x * info.width),
        round(y * info.height),
        round(w * info.width),
        round(h * info.height),
    ]


def relative_cam(rect: list[int] | tuple[int, ...], info: VideoInfo) -> list[float]:
    x, y, w, h = rect
    return [x / info.width, y / info.height, w / info.width, h / info.height]


def clips_from_candidates(candidates: list[dict], meta: dict) -> list[dict]:
    """Abgelehnte Themen und Modellfehler bleiben sichtbar, damit nichts still verschwindet."""
    status_map = {"assessed": "proposed", "off_topic": "off_topic", "failed": "failed"}
    return [
        {
            "id": candidate["id"],
            "status": status_map.get(candidate["status"], "proposed"),
            "score": candidate.get("score", 0),
            "candidate": candidate,
            "spec": spec_from_candidate(candidate, meta).model_dump(),
        }
        for candidate in candidates
    ]


class Worker(threading.Thread):
    def __init__(self, store: Store, load_settings: Callable[[], Settings], data_dir: Path) -> None:
        super().__init__(daemon=True, name="clipped-worker")
        self.store = store
        self.load_settings = load_settings
        self.data_dir = data_dir
        self._wake = threading.Event()
        self._stop_flag = threading.Event()

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop_flag.set()
        self._wake.set()

    def run(self) -> None:
        self.store.requeue_interrupted_jobs()
        while not self._stop_flag.is_set():
            job = self.store.claim_next_job()
            if job is None:
                self._wake.wait(timeout=2)
                self._wake.clear()
                continue
            self.run_job(job)

    def run_job(self, job: dict) -> None:
        handler = {
            "process_vod": self._process_vod,
            "render": self._render,
            "preview": self._preview,
        }[job["kind"]]
        try:
            handler(job, self._progress_callback(job["id"]))
        except JobCancelled:
            self.store.finish_job(job["id"], "cancelled")
            # Ein Abbruch ist kein Fehler: Es bleibt keine Meldung am VOD oder Clip haengen.
            self._mark_failed(job, "", cancelled=True)
        except Exception as exc:  # noqa: BLE001 - jeder Fehler soll in der Oberflaeche ankommen
            traceback.print_exc()
            message = str(exc) or exc.__class__.__name__
            self.store.finish_job(job["id"], "failed", message)
            self._mark_failed(job, message)
        else:
            self.store.finish_job(job["id"], "done")

    def _mark_failed(self, job: dict, message: str, cancelled: bool = False) -> None:
        if job["kind"] == "process_vod":
            self.store.update_vod(
                job["vod_id"], status="cancelled" if cancelled else "failed", error=message
            )
        elif job["kind"] == "render":
            status = "proposed" if cancelled else "render_failed"
            self.store.update_clip(job["vod_id"], job["clip_id"], status=status, error=message)
        elif not cancelled:
            self.store.update_clip(job["vod_id"], job["clip_id"], error=message)

    def _progress_callback(self, job_id: int) -> Callable[[str, float], None]:
        last = {"time": 0.0, "step": ""}

        def report(step: str, value: float) -> None:
            now = time.monotonic()
            if (
                step == last["step"]
                and now - last["time"] < PROGRESS_INTERVAL_SECONDS
                and value < 1
            ):
                return
            last.update(time=now, step=step)
            current = self.store.get_job(job_id)
            if current and current["status"] == "cancelling":
                raise JobCancelled
            self.store.update_job(job_id, step=step, progress=round(value, 4))

        return report

    def _vod_paths(self, vod_id: str) -> VodPaths:
        return VodPaths.for_vod(self.data_dir, vod_id)

    def _process_vod(self, job: dict, progress: Callable[[str, float], None]) -> None:
        settings = self.load_settings()
        vod = self.store.get_vod(job["vod_id"])
        if vod is None:
            return
        self.store.update_vod(vod["id"], status="processing", error="")
        paths = ingest(vod["source"], settings, vod["streamer"] or None, progress)
        set_time_range(paths, vod["time_range"])
        meta = read_json(paths.meta)
        self.store.update_vod(
            vod["id"], title=meta.get("title", ""), streamer=meta.get("streamer", "")
        )

        transcribe(paths, settings, progress)
        candidates = select_clips(paths, settings, progress)

        meta = read_json(paths.meta)
        known_cam = self.store.get_streamer_cam(meta.get("streamer", ""))
        if known_cam:
            info = probe_video(Path(meta["video"]))
            meta["cam"] = {"rect": absolute_cam(known_cam, info), "layout": "split"}
            write_json(paths.meta, meta)
        self.store.replace_proposed_clips(vod["id"], clips_from_candidates(candidates, meta))
        self.store.update_vod(vod["id"], status="ready")

    def _render_context(self, job: dict) -> tuple[Settings, VodPaths, dict, list[dict], ClipSpec]:
        clip = self.store.get_clip(job["vod_id"], job["clip_id"])
        if clip is None:
            raise RuntimeError("Clip nicht mehr vorhanden")
        paths = self._vod_paths(job["vod_id"])
        meta = read_json(paths.meta)
        segments = read_json(paths.transcript)["segments"]
        return self.load_settings(), paths, meta, segments, ClipSpec.model_validate(clip["spec"])

    def _render(self, job: dict, progress: Callable[[str, float], None]) -> None:
        settings, paths, meta, segments, spec = self._render_context(job)
        self.store.update_clip(job["vod_id"], job["clip_id"], status="rendering", error="")
        video_path, text_path = export_paths(self.data_dir, job["vod_id"], spec)
        work_dir = paths.root / "render" / spec.id
        render_clip(
            spec,
            meta,
            segments,
            settings,
            video_path,
            work_dir,
            False,
            lambda v: progress("render", v),
        )
        text_path.write_text(export_text(spec, meta, settings.style.censor), encoding="utf-8")
        shutil.rmtree(work_dir, ignore_errors=True)
        self.store.update_clip(
            job["vod_id"],
            job["clip_id"],
            status="done",
            output_path=str(video_path),
            text_path=str(text_path),
        )

    def _preview(self, job: dict, progress: Callable[[str, float], None]) -> None:
        settings, paths, meta, segments, spec = self._render_context(job)
        self.store.update_clip(job["vod_id"], job["clip_id"], error="")
        video_path = paths.root / "previews" / f"{spec.id}.mp4"
        work_dir = paths.root / "render" / f"{spec.id}-preview"
        render_clip(
            spec,
            meta,
            segments,
            settings,
            video_path,
            work_dir,
            True,
            lambda v: progress("preview", v),
        )
        shutil.rmtree(work_dir, ignore_errors=True)
        self.store.update_clip(job["vod_id"], job["clip_id"], preview_path=str(video_path))
