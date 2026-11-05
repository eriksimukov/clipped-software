import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipped.api.app import create_app
from clipped.config import Settings, load_settings
from clipped.jobs import worker as worker_module
from clipped.pipeline.common import vod_id_for

HAS_FFMPEG = shutil.which("ffmpeg") is not None
ALL_LOCAL = {"host": "127.0.0.1:8765"}


def _candidate(clip_id="c000010", start=10.0, end=40.0, status="assessed", score=70.0):
    return {
        "id": clip_id,
        "start": start,
        "end": end,
        "peak": 20.0,
        "signal_score": 0.5,
        "text": "hello",
        "status": status,
        "score": score,
        "assessment": {
            "hooks": ["first hook", "second", "third"],
            "title": "A Title",
            "description": "A description",
            "hashtags": ["#fun"],
            "reason": "good",
        },
    }


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings()
    settings.paths.data_dir = tmp_path
    config_path = tmp_path / "settings.toml"
    app = create_app(tmp_path, lambda: settings, config_path, start_worker=False)
    client = TestClient(app, headers=ALL_LOCAL)
    return {
        "client": client,
        "app": app,
        "data": tmp_path,
        "settings": settings,
        "config": config_path,
    }


def _seed_vod(env, video: Path | None = None, candidates=None):
    """Legt ein VOD an, als waere die Pipeline schon durchgelaufen."""
    store = env["app"].state.store
    vod_id = "vod1"
    store.add_vod(vod_id, str(video or "x.mp4"), "Streamer", None)
    store.update_vod(vod_id, status="ready", title="Title")
    root = env["data"] / "vods" / vod_id
    root.mkdir(parents=True, exist_ok=True)
    meta = {"video": str(video or "missing.mp4"), "streamer": "Streamer", "duration": 100.0}
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (root / "candidates.json").write_text(
        json.dumps(candidates or [_candidate()]), encoding="utf-8"
    )
    words = [
        {"start": 11.0 + i, "end": 11.5 + i, "word": w}
        for i, w in enumerate(["hello", "big", "world."])
    ]
    segments = [{"start": 11.0, "end": 14.0, "text": "hello big world.", "words": words}]
    (root / "transcript.json").write_text(json.dumps({"segments": segments}), encoding="utf-8")
    clips = worker_module.clips_from_candidates(candidates or [_candidate()], meta)
    store.replace_proposed_clips(vod_id, clips)
    return vod_id


@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg fehlt")
    path = tmp_path_factory.mktemp("video") / "tiny.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=s=1280x720:r=30:d=60",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=330:sample_rate=48000:duration=60",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path


# Zugriffsschutz


def test_foreign_host_is_rejected(env):
    assert env["client"].get("/api/jobs", headers={"host": "evil.example"}).status_code == 403


def test_foreign_origin_is_rejected(env):
    response = env["client"].post(
        "/api/vods", json={"source": "https://x.test/v"}, headers={"origin": "https://evil.example"}
    )
    assert response.status_code == 403
    assert env["app"].state.store.list_vods() == []


def test_same_origin_is_allowed(env):
    response = env["client"].get("/api/jobs", headers={"origin": "http://127.0.0.1:8765"})
    assert response.status_code == 200


# VODs


def test_add_url_creates_vod_and_job(env):
    response = env["client"].post(
        "/api/vods", json={"source": " https://x.test/v ", "streamer": "Me"}
    )
    assert response.status_code == 201
    assert response.json()["id"] == vod_id_for("https://x.test/v")
    jobs = env["client"].get("/api/jobs").json()
    assert [j["kind"] for j in jobs] == ["process_vod"]


def test_adding_same_vod_twice_does_not_duplicate_job(env):
    for _ in range(2):
        env["client"].post("/api/vods", json={"source": "https://x.test/v"})
    assert len(env["client"].get("/api/jobs").json()) == 1
    assert len(env["client"].get("/api/vods").json()) == 1


def test_link_with_extra_options_is_rejected_and_nothing_is_queued(env):
    response = env["client"].post(
        "/api/vods", json={"source": "https://x.test/v --cookies-from-browser"}
    )
    assert response.status_code == 400
    assert "Leerzeichen" in response.json()["detail"]
    assert env["client"].get("/api/jobs").json() == []


def test_settings_keep_login_paths_and_check_the_file(env, tmp_path):
    settings = env["client"].get("/api/settings").json()
    assert settings["download"] == {"cookies_file": "", "cookies_from_browser": ""}

    settings["download"]["cookies_file"] = str(tmp_path / "fehlt.txt")
    assert env["client"].put("/api/settings", json=settings).status_code == 400

    cookies = tmp_path / "youtube-cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")
    settings["download"]["cookies_file"] = f'"{cookies}"'
    saved = env["client"].put("/api/settings", json=settings).json()["settings"]
    assert saved["download"]["cookies_file"] == str(cookies)


def test_add_missing_file_is_rejected(env):
    response = env["client"].post("/api/vods", json={"source": "C:/does/not/exist.mp4"})
    assert response.status_code == 400


@pytest.mark.parametrize("time_range", [[10, 5], [-1, 5], [1]])
def test_invalid_time_range_is_rejected(env, time_range):
    response = env["client"].post(
        "/api/vods", json={"source": "https://x.test/v", "time_range": time_range}
    )
    assert response.status_code == 400


def test_upload_sanitizes_name_and_avoids_overwrite(env):
    client = env["client"]
    first = client.post("/api/uploads", params={"name": "../../evil name.mp4"}, content=b"one")
    second = client.post("/api/uploads", params={"name": "evil name.mp4"}, content=b"two")
    first_path, second_path = Path(first.json()["path"]), Path(second.json()["path"])
    assert first_path.parent == env["data"] / "uploads"
    assert first_path != second_path
    assert first_path.read_bytes() == b"one"
    assert second_path.read_bytes() == b"two"


def test_upload_rejects_non_video(env):
    response = env["client"].post("/api/uploads", params={"name": "run.exe"}, content=b"x")
    assert response.status_code == 400


def test_delete_vod_removes_work_dir_but_keeps_exports_and_external_source(env, tmp_path):
    external = tmp_path / "original.mp4"
    external.write_bytes(b"x")
    vod_id = _seed_vod(env, external)
    export = env["data"] / "export" / vod_id / "clip.mp4"
    export.parent.mkdir(parents=True)
    export.write_bytes(b"x")
    assert env["client"].delete(f"/api/vods/{vod_id}").status_code == 200
    assert not (env["data"] / "vods" / vod_id).exists()
    assert export.exists()
    assert external.exists()
    assert env["client"].get(f"/api/vods/{vod_id}").status_code == 404


def test_delete_vod_refused_while_job_active(env):
    vod_id = _seed_vod(env)
    env["app"].state.store.enqueue("render", vod_id, "c000010")
    assert env["client"].delete(f"/api/vods/{vod_id}").status_code == 409


# Clips


def test_clip_detail_has_spec_and_subtitle_lines(env):
    vod_id = _seed_vod(env)
    detail = env["client"].get(f"/api/vods/{vod_id}/clips/c000010").json()
    assert detail["spec"]["hook_text"] == "first hook"
    assert detail["spec"]["credit"] == "Streamer"
    assert [line["text"] for line in detail["lines"]] == ["hello big world."]


def test_status_mapping_keeps_rejected_candidates_visible(env):
    candidates = [
        _candidate(),
        _candidate("c000050", 50.0, 80.0, status="off_topic", score=10.0),
        {**_candidate("c000090", 90.0, 120.0), "status": "failed", "score": 0.0},
    ]
    vod_id = _seed_vod(env, candidates=candidates)
    clips = env["client"].get(f"/api/vods/{vod_id}").json()["clips"]
    assert {c["id"]: c["status"] for c in clips} == {
        "c000010": "proposed",
        "c000050": "off_topic",
        "c000090": "failed",
    }
    assert clips[0]["id"] == "c000010"


def test_text_edit_changes_lines(env):
    vod_id = _seed_vod(env)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    spec["text_edits"] = [{"start": 11.0, "end": 13.5, "text": "goodbye world"}]
    detail = env["client"].put(url, json=spec).json()
    assert detail["lines"][0]["text"] == "goodbye world"


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"start": 30.0, "end": 31.0}, 400),
        ({"start": 0.0, "end": 500.0}, 400),
        ({"start": -5.0, "end": 20.0}, 400),
        ({"layout": "banana"}, 422),
        ({"game_center": 2.0}, 422),
    ],
)
def test_invalid_spec_is_rejected(env, tiny_video, change, code):
    vod_id = _seed_vod(env, tiny_video)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    assert env["client"].put(url, json={**spec, **change}).status_code == code


def test_save_response_still_has_video_info(env, tiny_video):
    """Der Editor braucht die Bildmasse auch nach dem Speichern fuer den Facecam-Rahmen."""
    vod_id = _seed_vod(env, tiny_video)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    video = env["client"].put(url, json=spec).json()["video"]
    assert (video["width"], video["height"]) == (1280, 720)


def test_peak_is_clamped_into_clip(env, tiny_video):
    vod_id = _seed_vod(env, tiny_video)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    assert env["client"].put(url, json={**spec, "peak": 55.0}).json()["spec"]["peak"] == 40.0


def test_cam_outside_frame_is_rejected(env, tiny_video):
    vod_id = _seed_vod(env, tiny_video)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    assert env["client"].put(url, json={**spec, "cam_rect": [1200, 0, 300, 200]}).status_code == 400


def test_correcting_cam_updates_other_clips_meta_and_streamer(env, tiny_video):
    candidates = [_candidate(), _candidate("c000050", 45.0, 58.0)]
    vod_id = _seed_vod(env, tiny_video, candidates)
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    env["client"].put(url, json={**spec, "cam_rect": [100, 200, 320, 180], "layout": "split"})

    other = env["client"].get(f"/api/vods/{vod_id}/clips/c000050").json()
    assert other["spec"]["cam_rect"] == [100, 200, 320, 180]
    meta = json.loads((env["data"] / "vods" / vod_id / "meta.json").read_text(encoding="utf-8"))
    assert meta["cam"]["rect"] == [100, 200, 320, 180]
    relative = env["app"].state.store.get_streamer_cam("Streamer")
    assert relative == pytest.approx([100 / 1280, 200 / 720, 320 / 1280, 180 / 720])


def test_changing_finished_clip_needs_new_approval(env):
    vod_id = _seed_vod(env)
    store = env["app"].state.store
    store.update_clip(vod_id, "c000010", status="done", preview_path="x.mp4")
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    detail = env["client"].put(url, json={**spec, "hook_text": "new"}).json()
    assert detail["status"] == "proposed"
    assert detail["preview_path"] == ""


def test_saving_unchanged_clip_keeps_preview_and_status(env):
    vod_id = _seed_vod(env)
    store = env["app"].state.store
    store.update_clip(vod_id, "c000010", status="done", preview_path="x.mp4")
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    detail = env["client"].put(url, json=spec).json()
    assert detail["status"] == "done"
    assert detail["preview_path"] == "x.mp4"


def test_busy_clip_cannot_be_edited(env):
    vod_id = _seed_vod(env)
    url = f"/api/vods/{vod_id}/clips/c000010"
    env["client"].post(f"{url}/approve")
    spec = env["client"].get(url).json()["spec"]
    assert env["client"].put(url, json=spec).status_code == 409
    assert env["client"].post(f"{url}/reject").status_code == 409


def test_reject_and_restore(env):
    vod_id = _seed_vod(env)
    url = f"/api/vods/{vod_id}/clips/c000010"
    assert env["client"].post(f"{url}/reject").json()["status"] == "rejected"
    assert env["client"].post(f"{url}/restore").json()["status"] == "proposed"


def test_new_selection_keeps_edits_and_finished_clips(env):
    vod_id = _seed_vod(
        env,
        candidates=[
            _candidate(),
            _candidate("c000050", 50.0, 80.0),
            _candidate("c000090", 90.0, 120.0),
        ],
    )
    store = env["app"].state.store
    store.update_clip(vod_id, "c000090", status="done")
    url = f"/api/vods/{vod_id}/clips/c000010"
    spec = env["client"].get(url).json()["spec"]
    env["client"].put(url, json={**spec, "hook_text": "edited"})

    meta = {"streamer": "Streamer"}
    store.replace_proposed_clips(
        vod_id,
        worker_module.clips_from_candidates(
            [_candidate(), _candidate("c000200", 200.0, 230.0)], meta
        ),
    )
    clips = {c["id"]: c for c in store.list_clips(vod_id)}
    assert set(clips) == {"c000010", "c000090", "c000200"}
    assert clips["c000010"]["spec"]["hook_text"] == "edited"
    assert clips["c000090"]["status"] == "done"


# Dateien


def test_video_supports_range_requests(env, tiny_video):
    vod_id = _seed_vod(env, tiny_video)
    response = env["client"].get(f"/api/vods/{vod_id}/video", headers={"range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100


def test_frame_returns_jpeg(env, tiny_video):
    vod_id = _seed_vod(env, tiny_video)
    response = env["client"].get(f"/api/vods/{vod_id}/frame", params={"t": 5})
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"


def test_missing_video_gives_404(env):
    vod_id = _seed_vod(env)
    assert env["client"].get(f"/api/vods/{vod_id}/video").status_code == 404
    assert env["client"].get(f"/api/vods/{vod_id}/clips/c000010/preview.mp4").status_code == 404


def test_signals_can_be_sliced(env):
    vod_id = _seed_vod(env)
    root = env["data"] / "vods" / vod_id
    (root / "signals.json").write_text(
        json.dumps({"curve": [0.0, 0.1, 0.2, 0.3, 0.4]}), encoding="utf-8"
    )
    assert env["client"].get(
        f"/api/vods/{vod_id}/signals", params={"start": 1, "end": 2}
    ).json() == {
        "curve": [0.1, 0.2],
        "start": 1,
    }


# Einstellungen


def test_settings_roundtrip_writes_toml(env):
    settings = env["client"].get("/api/settings").json()
    settings["branding"]["watermark"] = "@me"
    settings["branding"]["channels"] = [{"platform": "tiktok", "handle": "@me"}]
    response = env["client"].put("/api/settings", json=settings)
    assert response.status_code == 200
    assert response.json()["restart_required"] is False
    loaded = load_settings(env["config"])
    assert loaded.branding.watermark == "@me"
    assert loaded.branding.channels[0].handle == "@me"


def test_changing_data_dir_asks_for_restart(env, tmp_path):
    settings = env["client"].get("/api/settings").json()
    settings["paths"]["data_dir"] = str(tmp_path / "other")
    assert env["client"].put("/api/settings", json=settings).json()["restart_required"] is True


def test_invalid_settings_are_rejected(env):
    settings = env["client"].get("/api/settings").json()
    settings["style"]["highlight_color"] = "yellow"
    assert env["client"].put("/api/settings", json=settings).status_code == 422
    assert not env["config"].exists()


# Worker


def _run_next_job(env):
    store = env["app"].state.store
    job = store.claim_next_job()
    assert job is not None
    env["app"].state.worker.run_job(job)
    return store.get_job(job["id"])


def test_render_job_exports_video_and_text(env, monkeypatch):
    vod_id = _seed_vod(env)

    def fake_render(spec, meta, segments, settings, out_path, work_dir, preview, on_progress):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"video")
        on_progress(1.0)
        return out_path

    monkeypatch.setattr(worker_module, "render_clip", fake_render)
    env["client"].post(f"/api/vods/{vod_id}/clips/c000010/approve")
    assert _run_next_job(env)["status"] == "done"

    clip = env["client"].get(f"/api/vods/{vod_id}/clips/c000010").json()
    assert clip["status"] == "done"
    assert Path(clip["output_path"]).read_bytes() == b"video"
    exports = env["client"].get("/api/exports").json()
    assert "TITEL\nA Title" in exports[0]["text"]
    assert env["client"].get(f"/api/vods/{vod_id}/clips/c000010/output.mp4").content == b"video"


def test_preview_runs_before_render_and_vod_jobs(env):
    vod_id = _seed_vod(env)
    store = env["app"].state.store
    store.enqueue("process_vod", "other")
    store.enqueue("render", vod_id, "c000010")
    store.enqueue("preview", vod_id, "c000010")
    assert [store.claim_next_job()["kind"] for _ in range(3)] == [
        "preview",
        "render",
        "process_vod",
    ]


def test_failed_render_is_reported_and_can_be_retried(env, monkeypatch):
    vod_id = _seed_vod(env)

    def broken(*_args, **_kwargs):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(worker_module, "render_clip", broken)
    url = f"/api/vods/{vod_id}/clips/c000010"
    env["client"].post(f"{url}/approve")
    job = _run_next_job(env)
    assert job["status"] == "failed" and job["error"] == "kaputt"
    clip = env["client"].get(url).json()
    assert clip["status"] == "render_failed" and clip["error"] == "kaputt"

    assert env["client"].post(f"{url}/approve").json()["status"] == "approved"


def test_cancel_running_job_stops_at_next_progress(env, monkeypatch):
    vod_id = _seed_vod(env)
    store = env["app"].state.store

    def slow(spec, meta, segments, settings, out_path, work_dir, preview, on_progress):
        store.request_cancel(job_id)
        on_progress(1.0)
        raise AssertionError("haette abbrechen muessen")

    monkeypatch.setattr(worker_module, "render_clip", slow)
    env["client"].post(f"/api/vods/{vod_id}/clips/c000010/approve")
    job_id = env["client"].get("/api/jobs").json()[0]["id"]
    assert _run_next_job(env)["status"] == "cancelled"
    clip = env["client"].get(f"/api/vods/{vod_id}/clips/c000010").json()
    assert clip["status"] == "proposed"
    assert clip["error"] == ""


def test_cancel_waiting_render_restores_clip(env):
    vod_id = _seed_vod(env)
    env["client"].post(f"/api/vods/{vod_id}/clips/c000010/approve")
    job_id = env["client"].get("/api/jobs").json()[0]["id"]
    assert env["client"].post(f"/api/jobs/{job_id}/cancel").json() == {"running": False}
    assert env["client"].get("/api/jobs").json() == []
    assert env["client"].get(f"/api/vods/{vod_id}/clips/c000010").json()["status"] == "proposed"


def test_interrupted_jobs_continue_after_restart(env):
    store = env["app"].state.store
    store.enqueue("process_vod", "a")
    store.claim_next_job()
    assert store.requeue_interrupted_jobs() == 1
    assert store.claim_next_job()["vod_id"] == "a"


def test_process_vod_job_failure_is_visible(env, monkeypatch):
    store = env["app"].state.store
    store.add_vod("vod2", "https://x.test/v", "", None)
    store.enqueue("process_vod", "vod2")

    def broken(*_args, **_kwargs):
        raise RuntimeError("Download fehlgeschlagen")

    monkeypatch.setattr(worker_module, "ingest", broken)
    assert _run_next_job(env)["status"] == "failed"
    vod = env["client"].get("/api/vods/vod2").json()
    assert vod["status"] == "failed"
    assert vod["error"] == "Download fehlgeschlagen"
