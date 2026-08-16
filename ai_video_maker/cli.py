"""Command-line interface: subcommand parsing and the `main` entry point.

The CLI is a thin shell around the pipeline's lifecycle commands — each
subcommand maps 1:1 onto a ``Pipeline.cmd_*`` method (and, later, an API
endpoint):

    orders      list the paid web orders waiting in Cloudinary (project-less)
    ingest      download an order's photos into a (new) project
    init        create the project workspace
    storyboard  style/plan and write the editable storyboard, then stop
    render      generate clips (and missing frames) from the storyboard
    audio       add SFX + music to rendered clips, rebuild the final video
    combine     concatenate the clips into output/final_video.mp4
    publish     upload the finished movie into its Cloudinary order folder
    status      show where the project stands and what to run next
    tag         say who is in each frame (identity ground truth for planning)
    feedback    tell the planner what a rendered clip got wrong, and learn from it
    lessons     show/edit what it has learned so far (project-less)
    costs       what each project has cost, and the total (project-less)
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
  python pipeline.py orders                 # list paid web orders in Cloudinary
  python pipeline.py ingest myfilm AM-...   # download an order's photos into a new project
  python pipeline.py init myfilm            # create projects/myfilm/, add images
  python pipeline.py storyboard myfilm      # style images + plan clips, stop for review
  python pipeline.py render myfilm          # generate the clips from the storyboard
  python pipeline.py combine myfilm         # stitch clips into output/final_video.mp4
  python pipeline.py publish myfilm         # deliver it into the order's Cloudinary folder
  python pipeline.py status myfilm          # see progress + the suggested next step
  python pipeline.py run myfilm             # everything in one go (with confirmations)

learning & money:
  python pipeline.py feedback myfilm "the boy teleports across the lawn" --clip 003_to_004
  python pipeline.py lessons                # what the planner has learned so far
  python pipeline.py costs                  # estimated spend, per project and in total
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

    # `orders` is the one project-less command (it inspects Cloudinary, not a
    # workspace), so it bypasses the command() helper and is handled early in
    # main() before any project resolution.
    sub.add_parser(
        "orders",
        help="List the paid web orders waiting in Cloudinary (newest first).",
        description="List the paid web orders waiting in Cloudinary (newest "
                    "first). Each printed folder name starts with the order "
                    "id from the confirmation email; ingest one with "
                    "`python pipeline.py ingest <project> <order-id>`.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # `lessons` and `costs` are project-less too: lessons are studio-wide by
    # design (see feedback.py) and the spending report spans every project.
    sp = sub.add_parser(
        "lessons",
        help="Show (and edit) the rules the planner has learned from clip "
             "feedback.",
        description="Show the rules distilled from clip feedback — they are "
                    "appended to every planning call, so this is the planner's "
                    "learned standing instructions. Rules are studio-wide, not "
                    "per project. Add one by hand with --add, switch a bad one "
                    "off with --disable (kept for the record), or delete it "
                    "with --forget.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("--add", metavar="TEXT", default=None,
                    help="Add a rule by hand (no OpenAI call). Keep it short, "
                         "general and imperative.")
    sp.add_argument("--scope", choices=["motion", "style"], default="motion",
                    help="With --add: which prompt the rule joins — the clip "
                         "planner (motion, default) or image styling (style).")
    sp.add_argument("--disable", metavar="ID", default=None,
                    help="Stop sending this rule to the model (it stays in the "
                         "file).")
    sp.add_argument("--enable", metavar="ID", default=None,
                    help="Send this rule to the model again.")
    sp.add_argument("--forget", metavar="ID", default=None,
                    help="Delete this rule for good.")

    sub.add_parser(
        "costs",
        help="Show what every project has cost so far (estimated).",
        description="Show what each project has cost and the total across all "
                    "of them. Figures are ESTIMATES: every API call is priced "
                    "from the `pricing` block in config.json, not read off a "
                    "provider's invoice. Projects that predate cost tracking "
                    "are valued from the files on disk and marked ~.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sp = sub.add_parser(
        "serve",
        help="Run the admin API server (+ order watcher) for the web admin "
             "panel.",
        description="Run the admin API server: order intake, project status, "
                    "storyboard editing, media serving, and background "
                    "execution of the pipeline steps — everything the "
                    "animoments admin panel talks to. Also starts the "
                    "Cloudinary order watcher (config: watch_*) unless "
                    "--no-watch. Requires ADMIN_API_TOKEN in .env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default 127.0.0.1; use 0.0.0.0 or a "
                         "tunnel to reach it from elsewhere).")
    sp.add_argument("--port", type=int, default=8300, help="Port (default 8300).")
    sp.add_argument("--no-watch", action="store_true",
                    help="Don't start the Cloudinary order watcher.")

    command("init",
            "Create the project workspace (input_images/, clips/, output/, ...) "
            "and exit. Drop your source images into the printed input_images/ "
            "folder afterwards.")

    sp = command("ingest",
                 "Download a paid web order's photos from Cloudinary into the "
                 "project's input_images/, creating the project if needed. "
                 "Photos are saved as 01.jpg, 02.jpg, ... preserving the "
                 "customer's chosen order; already-downloaded files are "
                 "skipped, so an interrupted run can just be re-run.")
    sp.add_argument("order",
                    help="Which order: the order id from the confirmation "
                         "email (AM-...), the full Cloudinary folder name, or "
                         "any unique fragment of it (e.g. the customer's "
                         "name). See `python pipeline.py orders`.")
    sp.add_argument("--force", action="store_true",
                    help="Re-download photos that already exist locally.")
    sp.add_argument("--dry-run", action="store_true",
                    help="List what would be downloaded without downloading.")

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
    sp.add_argument("--replan-all", action="store_true",
                    help="Re-plan EVERY transition, not just the ones whose "
                         "frames changed — the batch to run after correcting "
                         "the cast or the photo tags, since both only reach a "
                         "prompt when its pair is planned again. Hand-written "
                         "motion prompts are replaced.")
    sp.add_argument("--no-tag", action="store_true",
                    help="Skip the pass that proposes who is in each untagged "
                         "photo after planning (one vision call). Tag later "
                         "with `python pipeline.py tag <project>`.")
    sp.add_argument("--replan-clip", action="append", metavar="ID",
                    help="Ask the planner for a fresh motion prompt for this "
                         "transition (e.g. 003_to_004) even though its frames "
                         "didn't change; everything else is kept. If its clip "
                         "is already rendered it gets marked outdated. "
                         "Repeatable.")
    sp.add_argument("--restyle-frame", action="append", metavar="NAME",
                    help="Re-style this frame from scratch (styled png name, "
                         "e.g. beach.png) even though its source is unchanged "
                         "— spends image credits. The adjacent clips are marked "
                         "outdated (never deleted). Repeatable.")
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
    sp.add_argument("--clip", action="append", metavar="ID",
                    help="Redo audio for only this clip (e.g. 003_to_004), even "
                         "if it already has SFX — use after editing its "
                         "sound_prompt in the storyboard. Repeatable.")
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed (music is "
                         "never generated; supply a track or get none).")

    sp = command("combine",
                 "Concatenate the storyboard's clips into "
                 "output/final_video.mp4 (adds the music bed when audio is "
                 "enabled). Use --force to rebuild an existing final video.")
    _add_common_flags(sp)
    _add_audio_flags(sp)
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed.")
    sp.add_argument("--music-url", default=None,
                    help="Download the music bed from a URL (direct audio file "
                         "or an extractable page). You are responsible for "
                         "having the right to use it.")
    sp.add_argument("--credits-photos", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Append the original photos as an end-credits "
                         "montage after the last clip. Overrides the "
                         "credits_photos config key for this run.")
    sp.add_argument("--letter", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Scroll the project's letter.txt (Hebrew-safe) over "
                         "a dark background at the very end, credits-style. "
                         "Overrides the closing_letter config key.")
    sp.add_argument("--intro", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="Prepend the shared intro video (intro.mp4 at the "
                         "repo root; intro_file config key relocates it) "
                         "before everything else, normalized to the movie's "
                         "frame size. Overrides the intro_clip config key "
                         "for this run.")
    sp.add_argument("--final", action="store_true",
                    help="Shorthand for the full presentation: --intro "
                         "--credits-photos in one flag (an explicit "
                         "--no-intro / --no-credits-photos still wins).")

    sp = command("publish",
                 "Upload the finished movie into this project's Cloudinary "
                 "order folder, as final_v1, final_v2, ... Nothing already "
                 "there is ever replaced or deleted: every publish takes the "
                 "next free version number. Asks for confirmation, showing "
                 "the exact name the file will be saved under.")
    sp.add_argument("--dry-run", action="store_true",
                    help="Print the name the movie would be published under "
                         "without uploading anything.")
    sp.add_argument("--yes", "-y", action="store_true",
                    help="Skip the confirmation prompt and publish immediately.")

    command("status",
            "Show the project's progress (frames, storyboard, clips, final "
            "video) and the suggested next command.")

    sp = command("tag",
                 "Propose who is in each styled frame — which cast member "
                 "stands where — for you to correct in the admin panel. "
                 "Identity is what the planner gets wrong on its own (it "
                 "decides from the pixels whether the child in one photo is "
                 "the child from the last one), so tagging turns that guess "
                 "into a fact every later plan is told. One OpenAI vision "
                 "call; frames you have already tagged are left alone unless "
                 "--retag. Nothing is re-planned or re-rendered.")
    sp.add_argument("--retag", action="store_true",
                    help="Redo frames that are already tagged too, replacing "
                         "those tags (your corrections are lost).")
    sp.add_argument("--dry-run", action="store_true",
                    help="Say what would be looked at without spending a call.")
    sp.add_argument("-y", "--yes", action="store_true",
                    help="Skip the confirmation prompt.")

    sp = command("faces",
                 "Cut one canonical styled face per cast member out of the "
                 "frames they appear in, using the positions from `tag`. "
                 "These references are what later stylings of those people "
                 "are matched against, so the same person stays the same "
                 "character from frame to frame. Free — the faces come from "
                 "images already styled. With --audit it also asks the vision "
                 "model which frames draw someone as a DIFFERENT person "
                 "(one paid call); nothing is ever re-styled for you.")
    sp.add_argument("--audit", action="store_true",
                    help="Also compare every tagged appearance of each cast "
                         "member and report the frames where they are drawn "
                         "as somebody else. One OpenAI vision call.")
    sp.add_argument("--dry-run", action="store_true",
                    help="Say what would be cut and compared, spending nothing.")
    sp.add_argument("-y", "--yes", action="store_true",
                    help="Skip the confirmation prompt for --audit.")

    sp = command("feedback",
                 "Tell the planner what a rendered clip got wrong (or right). "
                 "Two things then happen: a vision call WATCHES the clip "
                 "(stills sampled across it, next to its key frames and the "
                 "prompt) and proposes a corrected motion prompt and length "
                 "for it, and your note plus what it saw are distilled into a "
                 "short, general rule added to EVERY future planning call. "
                 "Nothing is applied or re-rendered automatically. Costs two "
                 "small OpenAI calls (--no-watch / --no-learn skip either); "
                 "review or remove the rules with `python pipeline.py "
                 "lessons`.")
    sp.add_argument("note", nargs="?", default="",
                    help="What was wrong (or right) with the clip, in your own "
                         "words. Optional when --clip is given: the reviewer "
                         "will watch it and report on its own.")
    sp.add_argument("--clip", dest="clip_id", metavar="ID", default=None,
                    help="Which transition the feedback is about (e.g. "
                         "003_to_004). Omit for feedback about the movie in "
                         "general.")
    sp.add_argument("--good", action="store_true",
                    help="This clip came out well — learn what to KEEP doing "
                         "(the default is that feedback describes a problem).")
    sp.add_argument("--no-watch", action="store_true",
                    help="Don't let the reviewer look at the rendered clip; "
                         "learn from your note alone. (By default the clip is "
                         "sampled into stills and shown to a vision call, "
                         "which also proposes a corrected prompt and length.)")
    sp.add_argument("--no-learn", action="store_true",
                    help="Record the note only; don't spend an OpenAI call "
                         "turning it into a rule.")
    sp.add_argument("--dry-run", action="store_true",
                    help="Show what would be recorded without writing or "
                         "spending anything.")

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
    sp.add_argument("--music-file", default=None,
                    help="Use this audio file as the music bed.")
    sp.add_argument("--music-url", default=None,
                    help="Download the music bed from a URL (direct audio file "
                         "or an extractable page). You are responsible for "
                         "having the right to use it.")
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


def _shared_config_path(args: argparse.Namespace) -> Path:
    path = Path(args.config)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cmd_orders(args: argparse.Namespace) -> int:
    """Project-less listing of the paid web orders waiting in Cloudinary."""
    from .clients.cloudinary_client import CloudinaryClient
    from .intake import ingested_orders
    from .workspace import PROJECTS_DIR

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        config = Config.load(_shared_config_path(args))
        folders = CloudinaryClient.from_config(config).list_order_folders()
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not folders:
        print("No order folders found in Cloudinary.")
        return 0
    ingested = ingested_orders(PROJECTS_DIR)
    print(f"{len(folders)} order folder(s) in Cloudinary (newest first):")
    for name in folders:
        note = f"  -> ingested as projects/{ingested[name]}" if name in ingested else ""
        print(f"  {name}{note}")
    if len(ingested) < len(folders):
        print("\nDownload one into a new project with:")
        print("  python pipeline.py ingest <project> <order-id>")
    return 0


def _cmd_lessons(args: argparse.Namespace) -> int:
    """Project-less view/edit of the studio's learned planning rules."""
    from .feedback import LessonStore
    from .workspace import lessons_file

    path = lessons_file()
    store = LessonStore(path)
    try:
        if args.add:
            lesson = store.add(args.add, args.scope, origin="manual")
            print(f"Added [{lesson.scope}] {lesson.id}: {lesson.text}")
        for flag, active in (("disable", False), ("enable", True)):
            lesson_id = getattr(args, flag)
            if not lesson_id:
                continue
            lesson = store.update(lesson_id, active=active)
            if lesson is None:
                print(f"error: no lesson {lesson_id}", file=sys.stderr)
                return 1
            print(f"{'Enabled' if active else 'Disabled'} {lesson.id}: {lesson.text}")
        if args.forget:
            if not store.remove(args.forget):
                print(f"error: no lesson {args.forget}", file=sys.stderr)
                return 1
            print(f"Forgot {args.forget}.")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    lessons = store.all()
    if not lessons:
        print(
            "No lessons yet. They are written by `python pipeline.py feedback "
            "<project> \"...\" --clip <ID>` after watching a clip, or by hand "
            "with --add."
        )
        return 0
    print(f"{len(lessons)} lesson(s) in {path}:")
    for lesson in lessons:
        state = "" if lesson.active else "  (off)"
        origin = lesson.source.get("project", lesson.origin)
        print(f"  [{lesson.scope}] {lesson.id}{state}  {lesson.text}")
        print(f"        from {origin} · {lesson.created_at[:19]}")
    print("\nThese are appended to every planning call. Turn one off with:")
    print("  python pipeline.py lessons --disable <ID>")
    return 0


def _cmd_costs(args: argparse.Namespace) -> int:
    """Project-less spending report: per project and studio-wide."""
    from .costs import KIND_LABELS, merge_totals
    from .workspace import PROJECTS_DIR, is_project_dir

    load_dotenv(PROJECT_ROOT / ".env")
    try:
        shared = _shared_config_path(args)
        Config.load(shared)  # fail early on a broken config
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not PROJECTS_DIR.exists():
        print("No projects yet.")
        return 0

    rows: list[tuple[str, dict]] = []
    for path in sorted(PROJECTS_DIR.iterdir()):
        if not is_project_dir(path):
            continue
        ws = Workspace(path)
        try:
            config = Config.load(shared, override_path=path / "config.json")
            report = Pipeline(config, ws, RunOptions()).cost_report()
        except PipelineError as exc:
            print(f"  {path.name}: unreadable ({exc})", file=sys.stderr)
            continue
        rows.append((path.name, report))

    if not rows:
        print("No projects yet.")
        return 0
    totals = merge_totals([r for _, r in rows])
    print("\nEstimated spend per project (priced from config.pricing):\n")
    print(f"  {'project':<28}{'recorded':>10}{'from files':>12}  clips")
    for name, report in rows:
        marker = "~" if report["estimated"] else " "
        print(
            f"  {name:<28}{'$%.2f' % report['total_usd']:>10}"
            f"{'$%.2f' % report['estimated_usd']:>12}{marker} "
            f"{report['clips_rendered']}"
        )
    print(
        f"\n  {'TOTAL':<28}{'$%.2f' % totals['total_usd']:>10}"
        f"{'$%.2f' % totals['estimated_usd']:>12}"
    )
    print("\n  Recorded, by kind:")
    for kind, bucket in totals["by_kind"].items():
        if bucket["count"]:
            print(f"    {KIND_LABELS.get(kind, kind):<26} "
                  f"${bucket['usd']:.2f}  ({bucket['count']} call(s))")
    print(
        "\n  ~ = valued from the files on disk; this project's calls happened "
        "before\n    cost tracking existed (or its ledger is incomplete). "
        "All figures are\n    estimates, not provider invoices."
    )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the admin API server (uvicorn) with the order watcher."""
    load_dotenv(PROJECT_ROOT / ".env")
    try:
        from .server import create_app

        app = create_app(_shared_config_path(args), watch=not args.no_watch)
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # `orders`, `lessons`, `costs` and `serve` are project-less (they span all
    # projects) — handle them before any project resolution (they have no
    # `project` argument).
    if args.command == "orders":
        return _cmd_orders(args)
    if args.command == "lessons":
        return _cmd_lessons(args)
    if args.command == "costs":
        return _cmd_costs(args)
    if args.command == "serve":
        return _cmd_serve(args)

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
    # `ingest` bootstraps a fresh project (init + download) by design.
    if not workspace.root.exists() and args.command != "ingest":
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
        # A projects/<name>/config.json layers per-movie overrides (style
        # prompt, video model, audio settings, ...) over the shared config.
        config = Config.load(config_path, override_path=workspace.root / "config.json")
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
