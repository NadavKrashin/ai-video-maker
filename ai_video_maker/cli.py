"""Command-line interface: subcommand parsing and the `main` entry point.

The CLI is a thin shell around the pipeline's lifecycle commands — each
subcommand maps 1:1 onto a ``Pipeline.cmd_*`` method (and, later, an API
endpoint):

    init        create the project workspace
    storyboard  style/plan and write the editable storyboard, then stop
    render      generate clips (and missing frames) from the storyboard
    audio       add SFX + music to rendered clips, rebuild the final video
    combine     concatenate the clips into output/final_video.mp4
    status      show where the project stands and what to run next
    run         the whole flow in one go, with confirmation gates

All interactivity lives here (the ``confirm`` callback handed to the
pipeline); the library itself never touches stdin.
"""
from __future__ import annotations

import argparse
import sys
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

_LIFECYCLE = """\
project lifecycle:
  python pipeline.py init myfilm            # create projects/myfilm/, add images
  python pipeline.py storyboard myfilm      # style images + plan clips, stop for review
  python pipeline.py render myfilm          # generate the clips from the storyboard
  python pipeline.py combine myfilm         # stitch clips into output/final_video.mp4
  python pipeline.py status myfilm          # see progress + the suggested next step
  python pipeline.py run myfilm             # everything in one go (with confirmations)
"""


def _add_common_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true",
                   help="Redo completed outputs instead of skipping them.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work without spending API credits.")
    p.add_argument("--concurrency", type=int, default=None,
                   help="How many image/clip/SFX API jobs to run in parallel "
                        "(overrides config.max_parallel_requests). 1 = sequential.")


def _add_audio_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--add-audio", action="store_true",
                   help="Force audio on for this run, regardless of "
                        "config.audio_mode.")
    p.add_argument("--no-audio", action="store_true",
                   help="Force audio off for this run, even if config.audio_mode "
                        "is \"post\".")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Local AI video maker (image-to-video via OpenAI + fal.ai).",
        epilog=_LIFECYCLE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.json", help="Path to config JSON.")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, help_: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(
            name, help=help_, description=help_,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sp.add_argument("project",
                        help="Project name; its workspace is projects/<name>/.")
        return sp

    command("init",
            "Create the project workspace (input_images/, clips/, output/, ...) "
            "and exit. Drop your source images into the printed input_images/ "
            "folder afterwards.")

    sp = command("storyboard",
                 "Create the storyboard and stop for review. By default your "
                 "input images are styled and analysed into per-clip motion "
                 "plans; with --idea/--idea-file the whole storyboard is "
                 "written from scratch by the text model. Edit the saved "
                 "storyboard/storyboard.json freely — `render` uses it as-is.")
    _add_common_flags(sp)
    sp.add_argument("--style-prompt", default=None,
                    help="Override the global style prompt for the styling pass.")
    sp.add_argument("--no-analyze", action="store_true",
                    help="Skip the vision analysis; use the single global motion "
                         "prompt and one duration for every clip.")
    sp.add_argument("--duration", type=int, choices=sorted(VALID_DURATIONS),
                    help="Force every clip to this length (5 or 10 seconds); "
                         "omit to let the planner mix lengths.")
    sp.add_argument("--idea", default=None,
                    help="Generate the storyboard from this idea instead of "
                         "from input images.")
    sp.add_argument("--idea-file", default=None,
                    help="Read the idea/source material from a text file "
                         "(takes precedence over --idea).")
    sp.add_argument("--frame-count", type=int, default=None,
                    help="With --idea: number of key frames (overrides config "
                         "default_frame_count). 0 = let the model decide.")

    sp = command("render",
                 "Generate the video clips from the saved storyboard (any "
                 "missing generated frames are created first). Shows the "
                 "per-clip plan and asks before spending clip credits.")
    _add_common_flags(sp)
    _add_audio_flags(sp)
    sp.add_argument("-y", "--yes", action="store_true",
                    help="Skip the confirmation prompt; render immediately.")
    sp.add_argument("--clip", action="append", metavar="ID",
                    help="Render only this clip (e.g. 003_to_004), regenerating "
                         "it even if it exists. Repeatable.")
    sp.add_argument("--motion-prompt", default=None,
                    help="Override every clip's motion prompt for this run.")
    sp.add_argument("--duration", type=int, choices=sorted(VALID_DURATIONS),
                    help="Force every clip to this length for this run.")

    sp = command("audio",
                 "Add per-clip SFX + a music bed to the rendered clips, then "
                 "rebuild output/final_video.mp4.")
    _add_common_flags(sp)
    sp.add_argument("--music-prompt", default=None,
                    help="Override the background-music prompt for this run.")
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed instead of "
                         "reusing/generating output/music.mp3.")

    sp = command("combine",
                 "Concatenate the storyboard's clips into "
                 "output/final_video.mp4 (adds the music bed when audio is "
                 "enabled). Use --force to rebuild an existing final video.")
    _add_common_flags(sp)
    _add_audio_flags(sp)
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed.")

    command("status",
            "Show the project's progress (frames, storyboard, clips, final "
            "video) and the suggested next command.")

    sp = command("run",
                 "The whole flow in one command: storyboard (reused if already "
                 "saved) -> confirm -> clips -> confirm -> final video.")
    _add_common_flags(sp)
    _add_audio_flags(sp)
    sp.add_argument("-y", "--yes", action="store_true",
                    help="Skip the interactive confirmations; proceed "
                         "automatically.")
    sp.add_argument("--style-prompt", default=None,
                    help="Override the global style prompt.")
    sp.add_argument("--motion-prompt", default=None,
                    help="Override every clip's motion prompt.")
    sp.add_argument("--duration", type=int, choices=sorted(VALID_DURATIONS),
                    help="Force every clip to this length (5 or 10 seconds).")
    sp.add_argument("--no-analyze", action="store_true",
                    help="Skip the vision analysis of the styled frames.")
    sp.add_argument("--no-combine", action="store_true",
                    help="Stop after the clips; don't build the final video.")
    sp.add_argument("--music-prompt", default=None,
                    help="Override the background-music prompt.")
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed.")
    sp.add_argument("--idea", default=None,
                    help="Build the storyboard from this idea instead of from "
                         "input images.")
    sp.add_argument("--idea-file", default=None,
                    help="Read the idea from a text file (overrides --idea).")
    sp.add_argument("--frame-count", type=int, default=None,
                    help="With --idea: number of key frames. 0 = model decides.")

    return p


def _make_confirm(assume_yes: bool):
    """Terminal implementation of the pipeline's confirm gate.

    Auto-proceeds under --yes or when stdin isn't interactive (CI/automation),
    so scripted runs never block.
    """
    def confirm(lines: list[str], question: str) -> bool:
        if assume_yes or not sys.stdin.isatty():
            return True
        print("\n" + "=" * 70)
        for line in lines:
            print(line)
        print("=" * 70)
        try:
            answer = input(question).strip().lower()
        except EOFError:
            return True
        return answer in ("y", "yes")

    return confirm


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Every movie lives in its own workspace under projects/<name>/.
    try:
        workspace = Workspace.for_project(args.project)
    except InvalidProjectName as exc:
        parser.error(str(exc))

    # `init` is the only command allowed to create a project — everything else
    # requires it to exist, so a typo'd name fails loudly instead of silently
    # scaffolding a fresh empty workspace.
    if args.command == "init":
        workspace.mkdirs()
        print(f"Created project '{args.project}' at {workspace.root}")
        print(f"Put your source images in {workspace.input_images_dir}/")
        print(f"then plan the movie with:  python pipeline.py storyboard {args.project}")
        return 0
    if not workspace.root.exists():
        parser.error(
            f"Project '{args.project}' does not exist. Create it first:\n"
            f"  python pipeline.py init {args.project}"
        )
    workspace.mkdirs()  # ensure subdirectories for older/hand-made projects

    setup_logging(workspace)
    load_dotenv(PROJECT_ROOT / ".env")

    logger.info("Project workspace: %s", workspace.root)

    if getattr(args, "add_audio", False) and getattr(args, "no_audio", False):
        parser.error("--add-audio and --no-audio are mutually exclusive.")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if getattr(args, "dry_run", False):
        logger.info("DRY-RUN: no API credits will be spent.")

    try:
        config = Config.load(config_path)
        pipeline = Pipeline(
            config,
            workspace,
            RunOptions.from_args(args),
            confirm=_make_confirm(getattr(args, "yes", False)),
        )
        pipeline.execute(args.command)
    except PipelineError as exc:
        # Expected, user-facing failures (bad config/storyboard/inputs): report
        # cleanly and exit non-zero instead of dumping a traceback.
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
