"""FastAPI-Anwendung. Laeuft nur lokal; fremde Webseiten duerfen sie nicht ansprechen."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError

from clipped.config import ROOT, Settings, load_settings, save_settings
from clipped.db.store import Store
from clipped.jobs.worker import Worker, relative_cam
from clipped.pipeline.common import VodPaths, probe_video, read_json, vod_id_for, write_json
from clipped.pipeline.ingest import is_url
from clipped.pipeline.spec import ClipSpec
from clipped.pipeline.subtitles import apply_text_edits, chunk_words, flatten_words

WEB_DIST = ROOT / "web" / "dist"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "testserver"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".flv", ".ts", ".avi"}
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 180.0
# Diese Zustaende gehoeren dem Worker; waehrenddessen darf der Clip nicht bearbeitet werden.
BUSY_CLIP_STATES = {"approved", "rendering"}


class VodRequest(BaseModel):
    source: str
    streamer: str = ""
    time_range: list[float] | None = None


class RangeRequest(BaseModel):
    time_range: list[float] | None = None


class FolderRequest(BaseModel):
    vod_id: str | None = None


def validate_time_range(time_range: list[float] | None) -> None:
    if time_range is None:
        return
    if len(time_range) != 2 or time_range[0] < 0 or time_range[1] <= time_range[0]:
        raise HTTPException(400, "Der Zeitbereich braucht einen Start vor dem Ende.")


@lru_cache(maxsize=4)
def _load_segments(path: str, _modified: float) -> list[dict]:
    return read_json(Path(path))["segments"]


def read_segments(paths: VodPaths) -> list[dict]:
    """Transkripte langer VODs sind gross; der Editor fragt sie staendig ab."""
    if not paths.transcript.exists():
        return []
    return _load_segments(str(paths.transcript), paths.transcript.stat().st_mtime)


def subtitle_lines(spec: ClipSpec, segments: list[dict]) -> list[dict]:
    words = apply_text_edits(flatten_words(segments, spec.start, spec.end), spec.text_edits)
    return [
        {
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"],
            "text": " ".join(word["word"].strip() for word in chunk),
        }
        for chunk in chunk_words(words)
    ]


def create_app(
    data_dir: Path | None = None,
    settings_loader: Callable[[], Settings] = load_settings,
    settings_path: Path | None = None,
    start_worker: bool = True,
) -> FastAPI:
    data = data_dir or settings_loader().paths.data_dir
    store = Store(data / "clipped.sqlite3")
    worker = Worker(store, settings_loader, data)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if start_worker:
            worker.start()
        yield
        worker.stop()

    app = FastAPI(title="Clipped", lifespan=lifespan)
    app.state.store = store
    app.state.worker = worker

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Sonst koennte jede besuchte Webseite Jobs anstossen oder Dateien lesen."""
        host_header = request.headers.get("host", "")
        if urlparse(f"//{host_header}").hostname not in LOCAL_HOSTS:
            return JSONResponse({"detail": "Nur lokaler Zugriff erlaubt."}, status_code=403)
        origin = request.headers.get("origin")
        if origin and urlparse(origin).netloc != host_header:
            return JSONResponse({"detail": "Fremder Ursprung abgelehnt."}, status_code=403)
        return await call_next(request)

    def vod_paths(vod_id: str) -> VodPaths:
        return VodPaths(data / "vods" / vod_id)

    def require_vod(vod_id: str) -> dict:
        vod = store.get_vod(vod_id)
        if vod is None:
            raise HTTPException(404, "VOD nicht gefunden.")
        return vod

    def require_clip(vod_id: str, clip_id: str) -> dict:
        clip = store.get_clip(vod_id, clip_id)
        if clip is None:
            raise HTTPException(404, "Clip nicht gefunden.")
        return clip

    def read_meta(vod_id: str) -> dict:
        paths = vod_paths(vod_id)
        return read_json(paths.meta) if paths.meta.exists() else {}

    def clip_detail(vod_id: str, clip: dict) -> dict:
        spec = ClipSpec.model_validate(clip["spec"])
        preview = Path(clip["preview_path"]) if clip["preview_path"] else None
        detail = {
            **clip,
            "lines": subtitle_lines(spec, read_segments(vod_paths(vod_id))),
            # Die Datei behaelt ihren Namen; so laedt der Browser nach neuem Rendern nicht die alte.
            "preview_version": preview.stat().st_mtime if preview and preview.is_file() else 0,
        }
        video = read_meta(vod_id).get("video")
        if video and Path(video).is_file():
            info = probe_video(Path(video))
            detail["video"] = {
                "width": info.width,
                "height": info.height,
                "duration": info.duration,
            }
        return detail

    def enqueue(kind: str, vod_id: str, clip_id: str = "") -> None:
        store.enqueue(kind, vod_id, clip_id)
        worker.wake()

    # Status und Jobs

    @app.get("/api/status")
    def status() -> dict:
        settings = settings_loader()
        try:
            response = httpx.get(f"{settings.llm.host}/api/tags", timeout=1.5)
            names = [m["name"] for m in response.json().get("models", [])]
            ollama = {"running": True, "model_ready": settings.llm.model in names}
        except (httpx.HTTPError, ValueError):
            ollama = {"running": False, "model_ready": False}
        return {
            "ollama": ollama,
            "model": settings.llm.model,
            "free_gb": round(shutil.disk_usage(data).free / 1024**3, 1),
            "data_dir": str(data),
            "placeholder_handles": any(
                "deinhandle" in c.handle for c in settings.branding.channels
            ),
        }

    @app.get("/api/jobs")
    def jobs() -> list[dict]:
        return store.list_active_jobs()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: int) -> dict:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Job nicht gefunden.")
        running = store.request_cancel(job_id)
        if not running:
            # Wartender Job: es meldet sich kein Worker mehr, der den Zustand zuruecksetzt.
            if job["kind"] == "process_vod":
                store.update_vod(job["vod_id"], status="cancelled")
            elif job["kind"] == "render":
                store.update_clip(job["vod_id"], job["clip_id"], status="proposed")
        return {"running": running}

    # VODs

    @app.get("/api/vods")
    def list_vods() -> list[dict]:
        result = []
        for vod in store.list_vods():
            clips = store.list_clips(vod["id"])
            result.append(
                {
                    **vod,
                    "clip_count": sum(c["status"] == "proposed" for c in clips),
                    "done_count": sum(c["status"] == "done" for c in clips),
                }
            )
        return result

    @app.post("/api/vods", status_code=201)
    def add_vod(request: VodRequest) -> dict:
        source = request.source.strip().strip('"')
        if not source:
            raise HTTPException(400, "Bitte einen Link oder Dateipfad angeben.")
        if not is_url(source) and not Path(source).is_file():
            raise HTTPException(400, f"Datei nicht gefunden: {source}")
        validate_time_range(request.time_range)
        vod_id = vod_id_for(source)
        store.add_vod(vod_id, source, request.streamer.strip(), request.time_range)
        enqueue("process_vod", vod_id)
        return require_vod(vod_id)

    @app.post("/api/uploads", status_code=201)
    async def upload(request: Request, name: str) -> dict:
        """Roh-Upload statt Formular: so landet nichts im Temp-Ordner auf dem Systemlaufwerk."""
        clean = re.sub(r"[^\w.\- ]", "_", Path(name).name).strip() or "upload.mp4"
        target = data / "uploads" / clean
        if target.suffix.lower() not in VIDEO_SUFFIXES:
            raise HTTPException(400, f"Dateityp nicht unterstützt: {target.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        counter = 1
        while target.exists():
            target = target.with_name(f"{Path(clean).stem}-{counter}{Path(clean).suffix}")
            counter += 1
        partial = target.with_name(target.name + ".part")
        try:
            with partial.open("wb") as handle:
                async for chunk in request.stream():
                    handle.write(chunk)
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)
        return {"path": str(target)}

    @app.get("/api/vods/{vod_id}")
    def get_vod(vod_id: str) -> dict:
        vod = require_vod(vod_id)
        meta = read_meta(vod_id)
        summary_keys = (
            "duration",
            "topic_reason",
            "topic_seconds",
            "chat_messages",
            "cam",
            "platform",
        )
        return {
            **vod,
            "meta": {key: meta.get(key) for key in summary_keys},
            "clips": store.list_clips(vod_id),
        }

    @app.post("/api/vods/{vod_id}/retry")
    def retry_vod(vod_id: str) -> dict:
        require_vod(vod_id)
        store.update_vod(vod_id, status="queued", error="")
        enqueue("process_vod", vod_id)
        return require_vod(vod_id)

    @app.post("/api/vods/{vod_id}/range")
    def set_range(vod_id: str, request: RangeRequest) -> dict:
        require_vod(vod_id)
        validate_time_range(request.time_range)
        store.update_vod(vod_id, time_range=request.time_range, status="queued", error="")
        enqueue("process_vod", vod_id)
        return require_vod(vod_id)

    @app.delete("/api/vods/{vod_id}")
    def delete_vod(vod_id: str) -> dict:
        vod = require_vod(vod_id)
        if any(j["vod_id"] == vod_id for j in store.list_active_jobs()):
            raise HTTPException(409, "Für dieses VOD läuft noch ein Job.")
        store.delete_vod(vod_id)
        shutil.rmtree(data / "vods" / vod_id, ignore_errors=True)
        # Hochgeladene Dateien sind Kopien; Originale an anderer Stelle bleiben unangetastet.
        source = Path(vod["source"])
        if not is_url(vod["source"]) and source.parent == data / "uploads":
            source.unlink(missing_ok=True)
        return {"deleted": vod_id}

    @app.get("/api/vods/{vod_id}/video")
    def source_video(vod_id: str) -> FileResponse:
        meta = read_meta(vod_id)
        video = meta.get("video")
        if not video or not Path(video).is_file():
            raise HTTPException(404, "Quellvideo nicht gefunden.")
        return FileResponse(video)

    @app.get("/api/vods/{vod_id}/frame")
    def frame(vod_id: str, t: float, w: int | None = None) -> Response:
        meta = read_meta(vod_id)
        video = meta.get("video")
        if not video or not Path(video).is_file():
            raise HTTPException(404, "Quellvideo nicht gefunden.")
        scale = ["-vf", f"scale={min(max(w, 64), 1920)}:-2"] if w else []
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{max(t, 0.0):.3f}",
                "-i",
                video,
                "-frames:v",
                "1",
                *scale,
                "-f",
                "image2pipe",
                "-c:v",
                "mjpeg",
                "-q:v",
                "4",
                "-",
            ],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0 or not result.stdout:
            raise HTTPException(500, "Bild konnte nicht gelesen werden.")
        return Response(result.stdout, media_type="image/jpeg")

    @app.get("/api/vods/{vod_id}/signals")
    def signals(vod_id: str, start: float = 0.0, end: float | None = None) -> dict:
        paths = vod_paths(vod_id)
        if not paths.signals.exists():
            return {"curve": [], "start": 0}
        curve = read_json(paths.signals)["curve"]
        first = max(int(start), 0)
        last = len(curve) if end is None else min(int(end) + 1, len(curve))
        return {"curve": curve[first:last], "start": first}

    # Clips

    @app.get("/api/vods/{vod_id}/clips/{clip_id}")
    def get_clip(vod_id: str, clip_id: str) -> dict:
        return clip_detail(vod_id, require_clip(vod_id, clip_id))

    @app.put("/api/vods/{vod_id}/clips/{clip_id}")
    def update_clip(vod_id: str, clip_id: str, body: dict) -> dict:
        clip = require_clip(vod_id, clip_id)
        if clip["status"] in BUSY_CLIP_STATES:
            raise HTTPException(409, "Der Clip wird gerade gerendert.")
        try:
            spec = ClipSpec.model_validate({**body, "id": clip_id})
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_input=False)) from exc

        meta = read_meta(vod_id)
        video = meta.get("video")
        info = probe_video(Path(video)) if video and Path(video).is_file() else None
        limit = info.duration if info else float("inf")
        if spec.start < 0 or spec.end > limit + 0.5:
            raise HTTPException(400, "Der Clip liegt außerhalb des Videos.")
        if not MIN_CLIP_SECONDS <= spec.end - spec.start <= MAX_CLIP_SECONDS:
            raise HTTPException(
                400,
                f"Clips müssen {MIN_CLIP_SECONDS:.0f} bis {MAX_CLIP_SECONDS:.0f} Sekunden lang sein.",
            )
        spec = spec.model_copy(update={"peak": min(max(spec.peak, spec.start), spec.end)})
        if spec.cam_rect and info:
            x, y, w, h = spec.cam_rect
            if x < 0 or y < 0 or w < 16 or h < 16 or x + w > info.width or y + h > info.height:
                raise HTTPException(400, "Der Facecam-Rahmen liegt außerhalb des Bildes.")

        new_spec = spec.model_dump(mode="json")
        fields: dict = {"spec": new_spec}
        if new_spec != clip["spec"]:
            # Alte Vorschau und alte Freigabe gelten fuer den geaenderten Clip nicht mehr.
            fields["preview_path"] = ""
            if clip["status"] in {"done", "render_failed"}:
                fields["status"] = "proposed"
        store.update_clip(vod_id, clip_id, **fields)

        old_cam = clip["spec"].get("cam_rect")
        if spec.cam_rect and list(spec.cam_rect) != old_cam and info:
            _remember_cam(vod_id, clip_id, list(spec.cam_rect), meta, info)
        return clip_detail(vod_id, require_clip(vod_id, clip_id))

    def _remember_cam(vod_id: str, clip_id: str, rect: list[int], meta: dict, info) -> None:
        """Einmal korrigiert gilt fuer das ganze VOD und kuenftige VODs des Streamers."""
        meta["cam"] = {"rect": rect, "layout": "split"}
        write_json(vod_paths(vod_id).meta, meta)
        if meta.get("streamer"):
            store.set_streamer_cam(meta["streamer"], relative_cam(rect, info))
        for other in store.list_clips(vod_id):
            if other["id"] != clip_id and other["status"] not in BUSY_CLIP_STATES | {"done"}:
                store.update_clip(
                    vod_id, other["id"], spec={**other["spec"], "cam_rect": rect}, preview_path=""
                )

    @app.post("/api/vods/{vod_id}/clips/{clip_id}/approve")
    def approve_clip(vod_id: str, clip_id: str) -> dict:
        clip = require_clip(vod_id, clip_id)
        if clip["status"] in BUSY_CLIP_STATES:
            return clip_detail(vod_id, clip)
        store.update_clip(vod_id, clip_id, status="approved", error="")
        enqueue("render", vod_id, clip_id)
        return clip_detail(vod_id, require_clip(vod_id, clip_id))

    def _set_clip_state(vod_id: str, clip_id: str, state: str) -> dict:
        clip = require_clip(vod_id, clip_id)
        if clip["status"] in BUSY_CLIP_STATES:
            raise HTTPException(409, "Der Clip wird gerade gerendert.")
        store.update_clip(vod_id, clip_id, status=state)
        return clip_detail(vod_id, require_clip(vod_id, clip_id))

    @app.post("/api/vods/{vod_id}/clips/{clip_id}/reject")
    def reject_clip(vod_id: str, clip_id: str) -> dict:
        return _set_clip_state(vod_id, clip_id, "rejected")

    @app.post("/api/vods/{vod_id}/clips/{clip_id}/restore")
    def restore_clip(vod_id: str, clip_id: str) -> dict:
        return _set_clip_state(vod_id, clip_id, "proposed")

    @app.post("/api/vods/{vod_id}/clips/{clip_id}/preview")
    def preview_clip(vod_id: str, clip_id: str) -> dict:
        require_clip(vod_id, clip_id)
        enqueue("preview", vod_id, clip_id)
        return {"queued": True}

    def _clip_file(vod_id: str, clip_id: str, field: str) -> FileResponse:
        path = require_clip(vod_id, clip_id)[field]
        if not path or not Path(path).is_file():
            raise HTTPException(404, "Datei nicht vorhanden.")
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/vods/{vod_id}/clips/{clip_id}/preview.mp4")
    def preview_file(vod_id: str, clip_id: str) -> FileResponse:
        return _clip_file(vod_id, clip_id, "preview_path")

    @app.get("/api/vods/{vod_id}/clips/{clip_id}/output.mp4")
    def output_file(vod_id: str, clip_id: str) -> FileResponse:
        return _clip_file(vod_id, clip_id, "output_path")

    # Export

    @app.get("/api/exports")
    def exports() -> list[dict]:
        result = []
        for clip in store.list_finished_clips():
            vod = store.get_vod(clip["vod_id"])
            text_path = Path(clip["text_path"]) if clip["text_path"] else None
            result.append(
                {
                    "vod_id": clip["vod_id"],
                    "vod_title": vod["title"] if vod else "",
                    "id": clip["id"],
                    "score": clip["score"],
                    "spec": clip["spec"],
                    "text": text_path.read_text(encoding="utf-8")
                    if text_path and text_path.is_file()
                    else "",
                    "file": clip["output_path"],
                }
            )
        return result

    @app.post("/api/open-folder")
    def open_folder(request: FolderRequest) -> dict:
        folder = data / "export"
        if request.vod_id:
            require_vod(request.vod_id)
            folder = folder / request.vod_id
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(folder)
        return {"opened": str(folder)}

    # Einstellungen

    @app.get("/api/settings")
    def get_settings() -> dict:
        return settings_loader().model_dump(mode="json")

    @app.put("/api/settings")
    def put_settings(body: dict) -> dict:
        try:
            settings = Settings.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_input=False)) from exc
        save_settings(settings, settings_path)
        return {
            "settings": settings.model_dump(mode="json"),
            "restart_required": settings.paths.data_dir != data,
        }

    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app


def run_server(settings: Settings, port: int) -> None:
    import uvicorn

    app = create_app(settings.paths.data_dir)
    print(f"Clipped läuft auf http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
