"""Kommandozeile fuer einzelne Pipeline-Schritte."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from clipped.config import load_settings
from clipped.pipeline.common import read_json
from clipped.pipeline.export import export_paths, export_text
from clipped.pipeline.ingest import ingest, set_time_range
from clipped.pipeline.render import render_clip
from clipped.pipeline.select_clips import select_clips
from clipped.pipeline.spec import spec_from_candidate
from clipped.pipeline.transcribe import transcribe


def _print_progress(step: str, value: float) -> None:
    print(f"\r{step}: {value * 100:5.1f}%", end="", flush=True)


def _add_vod_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("source", help="Dateipfad oder Link")
    command.add_argument("--streamer", help="Name fuer den Credit")
    command.add_argument(
        "--range",
        nargs=2,
        type=float,
        metavar=("VON", "BIS"),
        help="Nur diesen Zeitbereich in Sekunden verwenden",
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="clipped")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_vod_arguments(sub.add_parser("prepare", help="VOD importieren und transkribieren"))
    _add_vod_arguments(sub.add_parser("select", help="Zusaetzlich Clips auswaehlen"))
    render = sub.add_parser("render", help="Die besten Clips ohne Pruefung rendern")
    _add_vod_arguments(render)
    render.add_argument("--top", type=int, default=3, help="Anzahl der Clips")
    render.add_argument("--preview", action="store_true", help="Schnelle Vorschau-Qualitaet")
    serve = sub.add_parser("serve", help="Pruef-Oberflaeche starten")
    serve.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    settings = load_settings()
    if args.command == "serve":
        from clipped.api.app import run_server

        run_server(settings, args.port)
        return

    started = time.monotonic()
    paths = ingest(args.source, settings, args.streamer, _print_progress)
    if args.range:
        set_time_range(paths, args.range)
    transcribe(paths, settings, _print_progress)

    if args.command in {"select", "render"}:
        print()
        clips = select_clips(paths, settings, _print_progress)
        print()
        for clip in clips:
            hooks = clip.get("assessment", {}).get("hooks", [])
            print(
                f"{clip.get('score', 0):5.1f}  {clip['start']:7.1f}-{clip['end']:7.1f}  "
                f"{clip['status']}  {hooks[0] if hooks else clip.get('error', '')}"
            )

    if args.command == "render":
        meta = read_json(paths.meta)
        segments = read_json(paths.transcript)["segments"]
        usable = [c for c in clips if c["status"] == "assessed"][: args.top]
        for clip in usable:
            spec = spec_from_candidate(clip, meta)
            video_path, text_path = export_paths(
                settings.paths.data_dir, Path(paths.root).name, spec
            )
            if args.preview:
                video_path = video_path.with_name(video_path.stem + ".preview.mp4")
            render_clip(
                spec,
                meta,
                segments,
                settings,
                video_path,
                paths.root / "render" / spec.id,
                args.preview,
                lambda value, name=spec.id: _print_progress(f"render {name}", value),
            )
            text_path.write_text(export_text(spec, meta), encoding="utf-8")
            print(f"\n{video_path}")

    print(f"\nFertig in {time.monotonic() - started:.0f} s: {paths.root}")


if __name__ == "__main__":
    main()
