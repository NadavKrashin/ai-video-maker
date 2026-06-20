"""Command-line interface: argument parsing and the `main` entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .config import Config
from .constants import VALID_DURATIONS
from .errors import InvalidProjectName, PipelineError
from .logging_setup import logger, setup_logging
from .options import RunOptions
from .runner import Pipeline
from .workspace import PROJECT_ROOT, Workspace


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Local AI video maker (image-to-video via OpenAI + fal.ai).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.json", help="Path to config JSON.")
    p.add_argument("--project", default=None,
                   help="Name of an isolated movie workspace under projects/. "
                        "Each project keeps its own input_images/, frames, clips, "
                        "output, storyboard and state, so separate movies never "
                        "collide. Omit to use the shared top-level folders.")
    p.add_argument("--force", action="store_true", help="Redo completed outputs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work without spending API credits.")
    p.add_argument("--only-style", action="store_true",
                   help="Only style/generate images; skip video generation.")
    p.add_argument("--only-video", action="store_true",
                   help="Only generate videos from existing styled/generated images.")
    p.add_argument("--combine", action="store_true",
                   help="Only combine existing clips/ into output/final_video.mp4 "
                        "(no image/video generation).")
    p.add_argument("--no-combine", action="store_true",
                   help="Skip combining clips into a final video at the end of a run.")
    p.add_argument("--add-audio", action="store_true",
                   help="Force audio on for this run (per-clip SFX + music bed), "
                        "regardless of config.audio_mode.")
    p.add_argument("--no-audio", action="store_true",
                   help="Force audio off for this run, even if config.audio_mode "
                        "is \"post\".")
    p.add_argument("--audio-only", action="store_true",
                   help="Add SFX + music to existing clips/ and rebuild "
                        "output/final_video.mp4 (no image/video generation).")
    p.add_argument("--music-prompt", default=None,
                   help="Override the background-music prompt for this run.")
    p.add_argument("--duration", type=int, choices=sorted(VALID_DURATIONS),
                   help="Clip duration in seconds (5 or 10).")
    p.add_argument("--concurrency", type=int, default=None,
                   help="How many image/clip/SFX API jobs to run in parallel "
                        "(overrides config.max_parallel_requests). 1 = sequential.")
    p.add_argument("--motion-prompt", default=None,
                   help="Override the global motion prompt.")
    p.add_argument("--style-prompt", default=None,
                   help="Override the global style prompt (Mode A).")
    # Mode B
    p.add_argument("--idea", default=None, help="Video idea/prompt (Mode B).")
    p.add_argument("--idea-file", default=None,
                   help="Path to a text file with the idea/source material "
                        "(Mode B). Use this for long or structured pasted data; "
                        "takes precedence over --idea.")
    p.add_argument("--frame-count", type=int, default=None,
                   help="Mode B: number of key frames to generate "
                        "(overrides config default_frame_count). Use 0 to let "
                        "the model choose based on your content.")
    p.add_argument("--from-scratch", action="store_true",
                   help="Use Mode B (generate from an idea).")
    p.add_argument("--create-storyboard", action="store_true",
                   help="Mode B: create the storyboard and stop.")
    p.add_argument("--approve-storyboard", action="store_true",
                   help="Mode B: generate frames/clips from an approved storyboard.")
    p.add_argument("--storyboard-file", default="storyboard/storyboard.json",
                   help="Path to the storyboard JSON (Mode B approval).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Every movie lives in its own workspace under projects/<name>/. Without
    # --project we use projects/default/ so the repo root stays clean.
    try:
        workspace = Workspace.for_project(args.project or "default")
    except InvalidProjectName as exc:
        parser.error(str(exc))

    workspace.mkdirs()

    setup_logging(workspace)
    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Project workspace: %s", workspace.root)

    if args.only_style and args.only_video:
        parser.error("--only-style and --only-video are mutually exclusive.")

    if args.add_audio and args.no_audio:
        parser.error("--add-audio and --no-audio are mutually exclusive.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if args.dry_run:
        logger.info("DRY-RUN: no API credits will be spent.")

    try:
        config = Config.load(config_path)
        Pipeline(config, workspace, RunOptions.from_args(args)).run()
    except PipelineError as exc:
        # Expected, user-facing failures (bad config/storyboard/inputs): report
        # cleanly and exit non-zero instead of dumping a traceback.
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
