"""Kommandozeile fuer einzelne Pipeline-Schritte."""

from __future__ import annotations

import argparse
import time

from clipped.config import load_settings
from clipped.pipeline.ingest import ingest
from clipped.pipeline.transcribe import transcribe


def _print_progress(step: str, value: float) -> None:
    print(f"\r{step}: {value * 100:5.1f}%", end="", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="clipped")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="VOD importieren und transkribieren")
    prep.add_argument("source", help="Dateipfad oder Link")
    prep.add_argument("--streamer", help="Name fuer den Credit")
    args = parser.parse_args()

    settings = load_settings()
    if args.command == "prepare":
        started = time.monotonic()
        paths = ingest(args.source, settings, args.streamer, _print_progress)
        transcribe(paths, settings, _print_progress)
        print(f"\nFertig in {time.monotonic() - started:.0f} s: {paths.root}")


if __name__ == "__main__":
    main()
