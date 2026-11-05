"""Rendert einen Clip in allen Layouts und legt Kontaktbogen zum Ansehen ab.

Aufruf: uv run python scripts/render_layout_check.py <vod-id> <ausgabeordner>
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clipped.config import load_settings
from clipped.pipeline.common import VodPaths, probe_video, read_json, write_json
from clipped.pipeline.facecam import detect_cam, sample_times
from clipped.pipeline.render import render_clip
from clipped.pipeline.spec import ClipSpec, spec_from_candidate


def contact_sheet(video: Path, out: Path, count: int = 6) -> None:
    duration = probe_video(video).duration
    times = [0.4, 1.5, duration * 0.3, duration * 0.5, duration * 0.75, duration - 1.0][:count]
    inputs: list[str] = []
    for time_point in times:
        inputs += ["-ss", f"{time_point:.2f}", "-i", str(video)]
    graph = (
        ";".join(f"[{i}:v]scale=300:-2[s{i}]" for i in range(len(times)))
        + ";"
        + "".join(f"[s{i}]" for i in range(len(times)))
        + f"hstack=inputs={len(times)}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            graph,
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )


def main() -> None:
    vod_id, out_dir = sys.argv[1], Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    paths = VodPaths.for_vod(settings.paths.data_dir, vod_id)
    meta = read_json(paths.meta)
    info = probe_video(Path(meta["video"]))

    meta["cam"] = detect_cam(meta["video"], sample_times(0.0, info.duration), info)
    write_json(paths.meta, meta)
    print("Facecam:", meta["cam"])

    segments = read_json(paths.transcript)["segments"]
    candidates = [c for c in read_json(paths.candidates) if c["status"] == "assessed"]
    base: ClipSpec = spec_from_candidate(candidates[0], meta)

    variants = {
        "split": base,
        "gameplay_overlay": base.model_copy(update={"layout": "gameplay"}),
        "gameplay_clean": base.model_copy(
            update={"layout": "gameplay", "cam_rect": None, "teaser": False}
        ),
        "blur": base.model_copy(update={"layout": "blur"}),
    }
    for name, spec in variants.items():
        video = out_dir / f"{name}.mp4"
        started = time.monotonic()
        render_clip(
            spec,
            meta,
            segments,
            settings,
            video,
            paths.root / "render" / f"check-{name}",
            name == "blur",
        )
        seconds = time.monotonic() - started
        duration = probe_video(video).duration
        print(
            f"{name}: {duration:.1f} s Clip in {seconds:.1f} s gerendert ({duration / seconds:.1f}x)"
        )
        contact_sheet(video, out_dir / f"{name}.png")


if __name__ == "__main__":
    main()
