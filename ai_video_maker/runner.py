"""Pipeline orchestration.

The pipeline is constructed from three explicit inputs — a validated ``Config``,
a ``Workspace`` (all per-movie paths), and ``RunOptions`` (this run's choices).
Nothing here reads global state, argparse, or the terminal, so the same
orchestration can be driven by the CLI or, later, an API request.

The public surface is one method per lifecycle command (``cmd_ingest``,
``cmd_storyboard``, ``cmd_render``, ``cmd_audio``, ``cmd_combine``,
``cmd_publish``, ``cmd_status``, ``cmd_run``),
dispatched via :meth:`Pipeline.execute`. Anything interactive happens through
the injected ``confirm`` callback — the CLI wires it to a terminal prompt, an
API caller simply omits it (every gate auto-proceeds) and drives the steps
individually instead.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from .clients.audio import AudioClient
from .clients.cloudinary_client import (
    CloudinaryClient,
    ingest_filename,
    next_publish_version,
    publish_public_id,
    resolve_order_folder,
)
from .clients.download import download_file
from .clients.openai_client import (
    OpenAIClient,
    ends_offscreen,
    indistinct_epithets,
    is_camera_transition,
    is_clothing_anchored,
    is_unstageable_pair,
)
from .clients.video import VideoClient
from .config import Config
from .costs import KIND_CLIP, KIND_SFX, CostLedger, estimate_usd
from .errors import PipelineCancelled, PipelineError, StoryboardError
from .feedback import (
    MAX_NOTE_CHARS,
    SCOPE_MOTION,
    SCOPE_STYLE,
    VERDICTS,
    FeedbackEntry,
    FeedbackStore,
    LessonStore,
    lesson_prompt_block,
)
from .intake import parse_order_folder, read_order_record, write_order_record
from .logging_setup import logger
from .media.ffmpeg import (
    apply_edge_fades,
    apply_end_fade,
    combine_clips,
    ffprobe_duration,
    find_generated_clips,
    mux_music,
    render_intro_segment,
    render_letter_overlay,
    render_letter_scroll,
    render_photo_still,
    sample_clip_frames,
)
from .media.letter import (
    find_emoji_font,
    find_letter_font,
    letter_state,
    read_letter,
    render_letter_image,
    save_letter,
)
from .media.music_url import fetch_music
from .media.images import (
    SUPPORTED_IMAGE_EXTS,
    list_input_images,
    natural_sort_key,
    slugify_stem,
    verify_dimensions,
)
from .models import (
    Character,
    Frame,
    FramePerson,
    Storyboard,
    Transition,
    identity_fingerprint,
    outdated_identity_plans,
    tagged_people,
)
from .options import RunOptions
from .publish import (
    publish_state,
    published_versions,
    record_publication,
)
from .state import FailedJobStore, StateStore
from .storyboard_html import write_storyboard_preview
from .storyboard_md import write_storyboard_markdown
from .summary import RunSummary
from .workspace import PROJECT_ROOT, Workspace, lessons_file

# Bump when the presentation sections (intro / credits stills / letter) are
# DRAWN differently, so already-rendered ones in output/segments/ are redone
# on the next combine. Their reuse check is otherwise mtime-based, and a code
# change touches no project file: after the emoji and no-grey-photos fixes
# shipped, a re-combine returned the identical old ending.
#   2 - letter emoji via a colour font; photos no longer dimmed under it
_SEGMENT_RENDERER_VERSION = 2

# (info_lines, question) -> proceed? Injected by the CLI as a terminal prompt;
# defaults to always-yes so embedded/API callers never block on stdin.
ConfirmFn = Callable[[list[str], str], bool]

# One planned clip: (start_frame, end_frame, motion_prompt, duration, sound_prompt)
ClipPair = tuple[Path, Path, str, int, str]


def _with_global_motion(global_prompt: str, motion: str) -> str:
    """Prepend the storyboard's global motion guidance to one clip's prompt.

    The global prompt carries whole-movie facts (e.g. "two separate people
    appear throughout; never blend them") so the user doesn't have to repeat
    them in every transition. It rides along verbatim ahead of the per-clip
    action.
    """
    g = " ".join(global_prompt.split())
    if not g:
        return motion
    if g[-1] not in ".!?":
        g += "."
    return f"{g} {motion}"


def _local_time(iso: str) -> str:
    """Render an ISO-8601 UTC timestamp in the machine's local time.

    State timestamps are stored in UTC. Printing them raw next to
    wall-clock expectations reads as three hours in the past on an
    IDT machine, which made an in-flight render look long abandoned.
    """
    if not iso:
        return "unknown"
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return iso[:19]
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone()
    return stamp.strftime("%Y-%m-%d %H:%M:%S")


def _file_sha1(path: Path) -> str:
    """Content hash of a file ("" if missing); used for staleness detection."""
    if not path.exists():
        return ""
    digest = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_credits_and_letter(
    n_photos: int,
    seconds_per_photo: float,
    letter_travel_px: float,
    base_pps: float,
) -> tuple[float, float]:
    """Fit the photo montage and the letter scroll into one shared window.

    Returns (per_photo_seconds, letter_pixels_per_second). The window is the
    LONGER of the two at their configured paces — photos never flash by
    faster than configured and the letter never scrolls faster than
    configured; whichever side is shorter is stretched so both end together.
    """
    credits_len = n_photos * seconds_per_photo
    letter_len = letter_travel_px / base_pps
    section = max(credits_len, letter_len)
    return section / n_photos, letter_travel_px / section


def _consecutive_runs(indices: list[int]) -> list[list[int]]:
    """Group sorted indices into maximal consecutive runs.

    Used to batch adjacent dirty transition pairs into one vision call
    (pairs i and i+1 share a frame, so analysing them together is both
    cheaper and more coherent).
    """
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


class Pipeline:
    def __init__(
        self,
        config: Config,
        workspace: Workspace,
        options: RunOptions,
        confirm: Optional[ConfirmFn] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.options = options
        self.confirm: ConfirmFn = confirm or (lambda lines, question: True)
        # Cooperative cancellation (set by the admin API's job runner; the CLI
        # just uses Ctrl-C). Checked between work items, never mid-API-call.
        self.cancel_event = cancel_event or threading.Event()
        self.dry_run: bool = options.dry_run
        self.force: bool = options.force
        self.duration: int = options.duration or config.duration
        self.state = StateStore(workspace.state_file)
        self.failed = FailedJobStore(workspace.failed_jobs_file)
        self.summary = RunSummary()
        # What this movie has cost so far. Recording is best-effort by
        # design (see costs.py): a run that can't write its ledger still
        # renders.
        self.costs = CostLedger(workspace.costs_file)
        # Lessons learned from earlier renders — studio-wide, so a mistake
        # corrected on one order is corrected for every later one too.
        self.lessons = LessonStore(lessons_file())
        self.feedback_store = FeedbackStore(workspace.feedback_file)
        self.openai = OpenAIClient(config)
        # Every OpenAI call (including the condense/restage/reword ones the
        # runner never sees directly) prices itself into this project's ledger.
        self.openai.on_spend = self._record_cost
        # The state store lets interrupted fal renders resume by request_id
        # instead of re-billing (falreq:<clip> entries).
        self.video_client = VideoClient(
            config, state=self.state,
            should_cancel=self.cancel_event.is_set,
        )
        self.audio_client = AudioClient(config)
        # Audio is on when config.audio_mode == "post", unless overridden by
        # --add-audio / --no-audio for a single run (the `audio` command
        # forces it on).
        if options.no_audio:
            self.audio_enabled = False
        elif options.add_audio:
            self.audio_enabled = True
        else:
            self.audio_enabled = (config.audio_mode or "none").lower() == "post"
        # Resolved when a storyboard is loaded; falls back to config.
        # Concurrency for the I/O-bound generation steps.
        self.concurrency: int = max(
            1, options.concurrency or config.max_parallel_requests
        )
        # Guards summary counters when workers run in parallel (StateStore and
        # FailedJobStore guard themselves).
        self._lock = threading.Lock()

    # ------------------------------ dispatch ----------------------------- #
    def execute(self, command: str) -> None:
        """Run one lifecycle command; flush failure/summary reports after.

        ``status`` is read-only: no summary, and crucially no failure flush
        (flushing a run with zero failures deletes the previous report).
        """
        handlers: dict[str, Callable[[], None]] = {
            "ingest": self.cmd_ingest,
            "storyboard": self.cmd_storyboard,
            "render": self.cmd_render,
            "audio": self.cmd_audio,
            "combine": self.cmd_combine,
            "publish": self.cmd_publish,
            "status": self.cmd_status,
            "tag": self.cmd_tag,
            "feedback": self.cmd_feedback,
            "run": self.cmd_run,
        }
        handler = handlers.get(command)
        if handler is None:
            raise PipelineError(f"Unknown command: {command}")
        if command in ("status", "feedback"):
            # Neither produces pipeline outputs, so neither gets a summary —
            # and crucially neither flushes the failure report, which would
            # delete the record of the last run that actually had problems.
            handler()
            return
        try:
            handler()
        finally:
            self.failed.flush()
            self.summary.print(self.workspace)

    # --------------------------- shared plumbing -------------------------- #
    def _map_parallel(
        self,
        items: list[Any],
        worker: Callable[[Any], None],
        desc: str,
        unit: str = "item",
    ) -> None:
        """Run `worker` over `items`, in parallel unless dry-run/concurrency=1.

        Workers must handle (and record) their own errors; any unexpected
        exception is logged so one bad item can't sink the whole batch.
        """
        if not items:
            return
        self._raise_if_cancelled()

        def guarded(item: Any) -> None:
            # In-flight items finish (their output is paid for and useful);
            # items that haven't started yet are skipped after a cancel.
            if not self.cancel_event.is_set():
                worker(item)

        workers = 1 if self.dry_run else self.concurrency
        if workers <= 1:
            for item in tqdm(items, desc=desc, unit=unit):
                guarded(item)
        else:
            logger.info("%s: %d job(s), %d in parallel", desc, len(items), workers)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(guarded, item) for item in items]
                for fut in tqdm(
                    as_completed(futures), total=len(futures), desc=desc, unit=unit
                ):
                    try:
                        fut.result()
                    except Exception as exc:  # noqa: BLE001 - workers self-report
                        logger.error("Unexpected worker error: %s", exc)
        self._raise_if_cancelled()

    def _raise_if_cancelled(self) -> None:
        """Abort the command cleanly once a cancel has been requested.

        Completed items stay on disk, so the resume logic (existence-based
        skips) makes a later re-run continue from where the cancel landed.
        """
        if self.cancel_event.is_set():
            raise PipelineCancelled(
                "Cancelled by request — finished items are kept; re-run the "
                "command to resume from here."
            )

    def _ask(self, lines: list[str], question: str, decline_log: str) -> bool:
        """Gate on the injected confirm callback; True means proceed.

        Dry-runs always proceed (nothing is spent). A decline logs
        `decline_log` — which should name the command that resumes from here —
        and returns False.
        """
        self._raise_if_cancelled()
        if self.dry_run:
            return True
        if self.confirm(lines, question):
            return True
        logger.info(decline_log)
        return False

    # ------------------------- spending & learning ------------------------ #
    def _record_cost(
        self, kind: str, usd: float, detail: str = "", **extra: Any
    ) -> None:
        """Price one completed API call into this project's ledger.

        A dry-run never reaches a provider, so it must never book money
        either — the guard is here rather than at each call site so no future
        caller can forget it.
        """
        if self.dry_run:
            return
        self.costs.record(kind, usd, detail, **extra)

    def _lesson_texts(self, scope: str = SCOPE_MOTION) -> list[str]:
        """Active lessons for `scope`, newest last, capped for the prompt.

        The cap keeps learned rules from crowding out the instructions they
        refine: when there are more than `max_lessons_in_prompt`, the NEWEST
        win — they encode the most recent correction of the same behaviour.
        """
        if not self.config.learning_enabled:
            return []
        try:
            lessons = self.lessons.active(scope)
        except Exception as exc:  # noqa: BLE001 - learning never sinks a run
            logger.warning("Could not read the lessons store: %s", exc)
            return []
        limit = max(0, self.config.max_lessons_in_prompt)
        return [l.text for l in lessons[-limit:]] if limit else []

    def _next_command(self, command: str, *flags: str) -> str:
        """A copy-pasteable command for this project's next step."""
        return " ".join(
            ["python pipeline.py", command, self.workspace.root.name, *flags]
        )

    # ----------------------------- ingest step ---------------------------- #
    def cmd_ingest(self) -> None:
        """Download a paid web order's photos from Cloudinary into input_images/.

        The animoments frontend uploads each order to its own Cloudinary
        folder after payment, photos named by their position in the movie.
        ``options.order`` accepts the order id from the confirmation email,
        the full folder name, or any unique fragment (e.g. the customer's
        name). Photos are saved as 01.jpg, 02.jpg, ... so the movie keeps the
        customer's chosen order. Resume is existence-based: files already
        downloaded are skipped, so an interrupted run is simply re-run
        (--force re-downloads everything).
        """
        if not self.options.order:
            raise PipelineError(
                "ingest needs an order reference. List the waiting orders "
                "with:  python pipeline.py orders"
            )
        client = CloudinaryClient.from_config(self.config)
        folder = resolve_order_folder(self.options.order, client.list_order_folders())
        assets = client.list_order_assets(folder)
        if not assets:
            raise PipelineError(
                f"Order folder '{folder}' contains no images. Cloudinary was "
                "asked three ways — by the folder's tag, by public_id prefix, "
                "and by asset folder — and all three came back empty, so the "
                "photos are not in that folder. The customer's upload runs "
                "AFTER payment is confirmed, so an order can exist with "
                "nothing behind it."
            )

        targets = [
            (asset, self.workspace.input_images_dir
             / ingest_filename(seq, len(assets), asset.format))
            for seq, asset in enumerate(assets, start=1)
        ]

        # Refuse to mix orders: any input image that isn't one of this order's
        # target names means the project already holds other photos.
        expected = {dst.name for _, dst in targets}
        foreign = [
            p.name
            for p in list_input_images(self.workspace.input_images_dir)
            if p.name not in expected
        ]
        if foreign:
            raise PipelineError(
                f"{self.workspace.input_images_dir} already contains other "
                f"images ({', '.join(foreign[:5])}{'...' if len(foreign) > 5 else ''}) "
                "— ingest into a fresh project instead of mixing orders."
            )

        pending = [
            (asset, dst) for asset, dst in targets
            if self.force or not dst.exists()
        ]
        logger.info(
            "Order folder: %s — %d photo(s), %d to download.",
            folder, len(targets), len(pending),
        )
        if self.dry_run:
            for asset, dst in targets:
                note = "" if (asset, dst) in pending else "  (exists, skipped)"
                logger.info("DRY-RUN: %s -> %s%s", asset.public_id, dst.name, note)
            return
        for asset, dst in tqdm(pending, desc="Downloading photos", unit="photo"):
            download_file(
                asset.url, dst,
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_base_delay_seconds,
                description=f"download {asset.public_id}",
            )
        # Tie the project to its order (order.json): the `orders` listing, the
        # watcher, and the admin API all use this to know the folder is handled.
        write_order_record(
            self.workspace.order_file,
            order_folder=folder,
            photo_count=len(targets),
        )
        self._sync_order_status(folder)
        self._seed_letter_from_order(folder)
        logger.info(
            "Ingested %d photo(s) into %s", len(pending),
            self.workspace.input_images_dir,
        )
        logger.info("Next step:  %s", self._next_command("storyboard"))

    def _sync_order_status(self, folder: str) -> None:
        """Best-effort Firestore write-back after an ingest: status "ingested".

        Every ingest path lands here (CLI, panel button, watcher job), so the
        order ledger reflects progress without any background polling. The
        doc id is the AM-... order id embedded in the folder leaf. A ledger
        hiccup must never fail the ingest itself.
        """
        from .clients.firebase_client import STATUS_INGESTED, FirebaseClient

        if not FirebaseClient.configured(self.config):
            return
        order_id = parse_order_folder(folder)["order_id"]
        try:
            FirebaseClient.from_config(self.config).update_order(
                order_id,
                {"status": STATUS_INGESTED, "project": self.workspace.root.name},
            )
            logger.info("Order %s marked '%s' in Firestore.", order_id, STATUS_INGESTED)
        except Exception as exc:  # noqa: BLE001 - the ledger never sinks an ingest
            logger.warning(
                "Could not update Firestore order %s: %s", order_id, exc
            )

    def _seed_letter_from_order(self, folder: str) -> None:
        """Pre-fill letter.txt with the customer's blessing, if they wrote one.

        The order form already asks for a blessing, and it is exactly what
        the closing letter scrolls — but nothing used to carry it across, so
        the text sat in Firestore and every letter was retyped by hand. This
        only ever SEEDS: an existing letter.txt (someone's edit) is never
        overwritten, and a ledger hiccup never sinks the ingest.
        """
        from .clients.firebase_client import FirebaseClient

        path = self.workspace.letter_file
        if path.exists() or not FirebaseClient.configured(self.config):
            return
        order_id = parse_order_folder(folder)["order_id"]
        try:
            order = FirebaseClient.from_config(self.config).get_order(order_id)
        except Exception as exc:  # noqa: BLE001 - the ledger never sinks an ingest
            logger.warning(
                "Could not read Firestore order %s for its blessing: %s",
                order_id, exc,
            )
            return
        blessing = (order.blessing if order else "").strip()
        if not blessing:
            return
        save_letter(path, blessing)
        logger.info(
            "Closing letter seeded from the order's blessing (%d chars) -> %s "
            "— edit it in the panel, then combine with --letter.",
            len(blessing), path,
        )

    # --------------------------- storyboard step -------------------------- #
    def cmd_storyboard(self) -> None:
        """Create or update the storyboard, then stop for review.

        With --idea/--idea-file the storyboard is written by the text model
        from scratch. Otherwise the images in input_images/ are styled and the
        vision model plans one transition per consecutive pair — but only for
        the pairs that actually changed: transitions whose frames are
        untouched (including your hand edits) are carried over as-is.
        """
        if self.options.idea or self.options.idea_file:
            storyboard = self._create_storyboard_from_idea()
            if storyboard is not None:
                self._announce_storyboard_ready()
            return

        storyboard = self._prepare_mode_a_storyboard()
        if storyboard is not None and not self.dry_run:
            self._announce_storyboard_ready()

    def _prepare_mode_a_storyboard(self) -> Optional[Storyboard]:
        """Style inputs, reconcile the storyboard, save it. None = not enough frames.

        Shared by `storyboard` and `run`: styling (with re-style detection for
        changed sources), keeping unchanged transitions from the saved
        storyboard, re-planning only dirty pairs, and invalidating rendered
        clips whose frames changed.
        """
        images = list_input_images(self.workspace.input_images_dir)
        self.summary.input_count = len(images)
        if not images:
            raise PipelineError(
                f"No supported images found in {self.workspace.input_images_dir}. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}. Add images, or pass "
                "--idea to generate a storyboard from scratch."
            )
        logger.info("Found %d input image(s).", len(images))

        saved = None if self.force else self._load_saved_storyboard_tolerant()
        recorded_sources = {
            f.output_path: f.source_path
            for f in (saved.frames if saved else [])
            if f.source_path
        }
        style_notes = {
            f.output_path: f.style_note
            for f in (saved.frames if saved else [])
            if f.style_note
        }
        frame_pairs = self._style_images(images, recorded_sources, style_notes)
        if len(frame_pairs) < 2:
            logger.warning(
                "Need at least 2 styled images to make a clip; have %d.",
                len(frame_pairs),
            )
            return None

        storyboard, replanned, stale_tids = self._reconcile_storyboard(
            saved, frame_pairs
        )
        if replanned:
            kept = len(storyboard.transitions) - len(replanned)
            logger.info(
                "%s %d transition(s): %s%s",
                "[dry-run] would re-plan" if self.dry_run else "Re-planned",
                len(replanned),
                ", ".join(replanned),
                f" — kept {kept} existing (your edits preserved)" if kept else "",
            )
        else:
            logger.info(
                "Storyboard is up to date. Edit %s to change any clip, or pass "
                "--force to redo styling + analysis from scratch.",
                self.workspace.default_storyboard_json,
            )
        self._mark_stale_clips(stale_tids)
        if not self.dry_run:
            self._save_storyboard(storyboard)
            # The cast exists only now (the planner builds it), and tags
            # point at cast ids — so the first pass at "who is in each photo"
            # can only happen here, at the end of planning. It is a draft for
            # the human review this step already stops for; the tags then
            # feed the re-plan that follows it.
            if self._propose_tags(storyboard):
                self._save_storyboard(storyboard)
        return storyboard

    def _load_saved_storyboard_tolerant(self) -> Optional[Storyboard]:
        path = self.workspace.default_storyboard_json
        if not path.exists():
            return None
        try:
            return Storyboard.load(path)
        except StoryboardError as exc:
            logger.warning("Ignoring unreadable storyboard (%s); re-planning.", exc)
            return None

    def _styled_targets(self, images: list[Path]) -> list[Path]:
        """Map each input image to its styled output path.

        Two naming schemes:
        * Filename-keyed (default): styled_images/<slug-of-input-stem>.png.
          Artifacts follow the input FILE, so adding/removing/reordering
          inputs never misaligns existing styled work — an inserted image
          only costs its own styling plus the two clips around it.
        * Positional (legacy): styled_images/NNN_styled.png — kept for
          projects that already contain such files so nothing is orphaned.
          (To migrate an old project, delete styled_images/ and storyboard/;
          it will re-style everything under the new names.)
        """
        legacy = any(self.workspace.styled_images_dir.glob("*_styled.png"))
        if legacy:
            return [
                self.workspace.styled_images_dir / f"{i:03d}_styled.png"
                for i in range(1, len(images) + 1)
            ]
        targets: list[Path] = []
        first_source: dict[str, str] = {}
        for img in images:
            slug = slugify_stem(img.stem)
            if slug in first_source:
                raise PipelineError(
                    f"Input images {first_source[slug]!r} and {img.name!r} both "
                    f"map to styled name {slug}.png; rename one of them."
                )
            first_source[slug] = img.name
            targets.append(self.workspace.styled_images_dir / f"{slug}.png")
        return targets

    def _style_images(
        self,
        images: list[Path],
        recorded_sources: dict[str, str],
        style_notes: Optional[dict[str, str]] = None,
    ) -> list[tuple[Path, Path]]:
        """Style every input; return ordered (source, styled) pairs on disk.

        An EXISTING styled image is redone when its source file is newer than
        it, or when the saved storyboard records that it was styled from a
        DIFFERENT source (inputs swapped, or shifted in a legacy positional
        project). Because redoing spends image credits, the list is shown and
        gated on confirmation first; declining keeps the old files. --force
        redoes everything without asking.

        ``style_notes`` maps a styled path (workspace-relative) to that
        frame's ``Frame.style_note``, appended to the shared prompt for that
        frame only. A note is NOT a redo trigger — resume here is
        existence-based, so a newly written note takes effect on the next
        explicit ``--restyle-frame``, never as a surprise credit spend.
        """
        style_prompt = self.options.style_prompt or self.config.style_prompt
        # Lessons learned from earlier styled frames ride along with the
        # prompt for the CALL only — the storyboard keeps recording the plain
        # style prompt, so a new lesson never looks like a changed style and
        # never makes existing frames look out of date.
        style_lessons = self._lesson_texts(SCOPE_STYLE)
        if style_lessons:
            logger.info(
                "Applying %d learned styling lesson(s) to this pass.",
                len(style_lessons),
            )
            style_prompt += lesson_prompt_block(style_lessons, SCOPE_STYLE)
        style_notes = style_notes or {}
        targets = self._styled_targets(images)
        jobs = list(zip(images, targets))

        # Explicitly requested re-styles (--restyle-frame / the panel's
        # "Regenerate image" button): redo these frames from scratch even when
        # the source is unchanged, with no confirm gate (an explicit request,
        # like render --clip). A typo must fail loudly, not silently no-op.
        restyle = set(self.options.restyle_frames or [])
        if restyle:
            valid = {dst.name for dst in targets}
            unknown = restyle - valid
            if unknown:
                raise PipelineError(
                    "No such styled frame(s) to regenerate: "
                    f"{', '.join(sorted(unknown))}.\n"
                    "Valid frames: " + ", ".join(sorted(valid))
                )

        redo: dict[Path, str] = {}  # styled path -> reason
        if not self.force:
            for src, dst in jobs:
                if not dst.exists():
                    continue
                if src.stat().st_mtime > dst.stat().st_mtime:
                    redo[dst] = f"source {src.name} is newer"
                    continue
                rel_dst = dst.relative_to(self.workspace.root).as_posix()
                rel_src = src.relative_to(self.workspace.root).as_posix()
                recorded = recorded_sources.get(rel_dst)
                if recorded and recorded != rel_src:
                    redo[dst] = (
                        f"was styled from {Path(recorded).name}, source is now "
                        f"{src.name}"
                    )
        if redo and not self.dry_run:
            lines = [
                f"{len(redo)} existing styled image(s) no longer match their "
                "source and would be re-styled (spends image credits):"
            ] + [f"  {dst.name}: {reason}" for dst, reason in redo.items()]
            if not self._ask(
                lines,
                f"Re-style {len(redo)} image(s) now? [y/N] ",
                "Keeping the existing styled images unchanged.",
            ):
                redo.clear()

        def prompt_for(dst: Path) -> str:
            rel = dst.relative_to(self.workspace.root).as_posix()
            note = style_notes.get(rel, "").strip()
            if not note:
                return style_prompt
            return f"{style_prompt}\n\nFOR THIS IMAGE SPECIFICALLY: {note}"

        def work(job: tuple[Path, Path]) -> None:
            src, dst = job
            job_id = f"style:{dst.name}"

            if dst.exists() and not self.force and dst not in redo and dst.name not in restyle:
                with self._lock:
                    self.summary.styled_skipped += 1
                logger.info("Skip styled (done): %s", dst.name)
                return

            if self.dry_run:
                logger.info("[dry-run] would style %s -> %s", src.name, dst.name)
                with self._lock:
                    self.summary.styled_created += 1
                return

            try:
                self.openai.style_image(src, prompt_for(dst), dst)
                if not verify_dimensions(dst, self.config.target_width, self.config.target_height):
                    # Remove the bad file: leaving it would make the next run
                    # skip this image as "done" (resume is existence-based).
                    dst.unlink(missing_ok=True)
                    raise RuntimeError(f"{dst.name} is not {self.config.target_width}x{self.config.target_height}")
                with self._lock:
                    self.state.set(job_id, "done", output=str(dst))
                    self.summary.styled_created += 1
                logger.info("Styled: %s", dst.name)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.summary.styled_failed += 1
                    self.state.set(job_id, "failed")
                    self.failed.record(job_id, "style", str(exc), source=str(src))

        self._map_parallel(jobs, work, "Styling images", "img")
        if self.dry_run:
            return jobs  # nothing on disk yet; report the plan as-is
        # Only hand back frames that exist: a failed styling must not leak a
        # missing path into transition planning (one bad frame would otherwise
        # crash the vision call and degrade EVERY clip to the generic prompt).
        existing = [(src, dst) for src, dst in jobs if dst.exists()]
        if len(existing) < len(jobs):
            logger.warning(
                "%d image(s) failed to style; planning transitions from the "
                "%d that succeeded.", len(jobs) - len(existing), len(existing),
            )
        return existing

    def _reconcile_storyboard(
        self, saved: Optional[Storyboard], frame_pairs: list[tuple[Path, Path]]
    ) -> tuple[Storyboard, list[str], list[str]]:
        """Merge the saved storyboard with the frames now on disk.

        Returns ``(storyboard, replanned ids, stale ids)``. A pair is
        re-planned when it has no saved transition (a frame was inserted or
        removed next to it) or when one of its styled frames changed after
        the storyboard was written; every other transition is carried over
        verbatim, so hand edits survive. ``stale ids`` are the re-planned
        pairs whose already-rendered clips no longer match the storyboard:
        their frame CONTENT changed, or a real plan replaced a placeholder
        (fallback) prompt the clip was rendered with. Stale clips are only
        MARKED outdated (state ``stale:<clip>``) — never deleted or
        auto-regenerated. (A merely new pairing keeps any existing clip: its
        frames are unchanged, only the plan around them is new.)
        """
        root = self.workspace.root
        hashes = {dst: _file_sha1(dst) for _, dst in frame_pairs}
        # Who a human tagged in each frame, carried over by output path. The
        # frames are rebuilt from disk on every storyboard run, so without
        # this the tagging work would be silently thrown away each time.
        tagged = {
            f.output_path: f.people
            for f in (saved.frames if saved else []) if f.people
        }
        # Hand-written per-frame style notes are rebuilt-from-disk casualties
        # for the same reason, and have to be carried over just as explicitly.
        saved_notes = {
            f.output_path: f.style_note
            for f in (saved.frames if saved else [])
            if f.style_note
        }
        frames = [
            Frame(
                id=self._frame_id(dst),
                description="",
                image_prompt="",
                output_path=dst.relative_to(root).as_posix(),
                source_path=src.relative_to(root).as_posix(),
                styled_hash=hashes[dst],
                people=tagged.get(dst.relative_to(root).as_posix(), []),
                style_note=saved_notes.get(dst.relative_to(root).as_posix(), ""),
            )
            for src, dst in frame_pairs
        ]
        style = self.options.style_prompt or (
            saved.style if saved and saved.style else self.config.style_prompt
        )
        sb_path = self.workspace.default_storyboard_json
        sb_mtime = sb_path.stat().st_mtime if (saved and sb_path.exists()) else 0.0
        styled_paths = [dst for _, dst in frame_pairs]
        recorded_hashes = {
            f.output_path: f.styled_hash
            for f in (saved.frames if saved else [])
            if f.styled_hash
        }

        def frame_changed(p: Path) -> bool:
            if not p.exists():
                return False
            # Content comparison first: mtimes lie (a cloud-sync client
            # re-materializing untouched files once made every frame look
            # "changed" and wiped a project's rendered clips).
            recorded = recorded_hashes.get(p.relative_to(root).as_posix())
            if recorded:
                return hashes.get(p, "") != recorded
            # Storyboards saved before hashes existed: mtime heuristic.
            return p.stat().st_mtime > sb_mtime

        saved_tr = {
            (t.start_frame, t.end_frame): t
            for t in (saved.transitions if saved else [])
        }
        pairs = list(zip(frames, frames[1:]))
        # Explicitly requested re-plans (--replan-clip / the panel's
        # "re-plan prompt" button): planned fresh even though nothing
        # changed. A typo must fail loudly, not silently keep the old plan.
        all_tids = {f"{a.id}_to_{b.id}" for a, b in pairs}
        # --replan-all / the panel's "Re-plan all": every pair is treated as
        # explicitly requested, which is what makes corrected cast epithets
        # and photo tags reach prompts that were planned before them.
        requested = (
            set(all_tids) if self.options.replan_all
            else set(self.options.replan_clips or [])
        )
        unknown = requested - all_tids
        if unknown:
            raise PipelineError(
                f"No such transition(s) to re-plan: {', '.join(sorted(unknown))}.\n"
                "Valid ids: " + ", ".join(f"{a.id}_to_{b.id}" for a, b in pairs)
            )
        dirty: list[int] = []
        stale_tids: list[str] = []
        for i, (a, b) in enumerate(pairs):
            changed = saved is not None and (
                frame_changed(styled_paths[i]) or frame_changed(styled_paths[i + 1])
            )
            prior = saved_tr.get((a.output_path, b.output_path))
            # A transition still carrying the config fallback prompt was never
            # actually planned (the vision call failed — e.g. out of OpenAI
            # quota — and "a planning hiccup never sinks the run" filled it
            # in). It's a placeholder, not a hand edit: re-plan it on every
            # storyboard run until a real plan lands.
            placeholder = (
                prior is not None
                and prior.motion_prompt == self.config.motion_prompt
            )
            if (saved is None or changed or prior is None or placeholder
                    or f"{a.id}_to_{b.id}" in requested):
                dirty.append(i)
            if changed:
                stale_tids.append(f"{a.id}_to_{b.id}")

        # Human tags, resolved to epithets and aligned with the frame list.
        # Built from the cast that is about to be used for planning, so an
        # id nobody has an epithet for simply doesn't appear.
        cast_now = saved.characters if saved else []
        people_by_path = tagged_people(
            Storyboard(
                project_title="", style=style, characters=cast_now,
                frames=frames, transitions=[],
            )
        )
        frame_people = [people_by_path.get(f.output_path) for f in frames]

        plans, cast = self._plan_pairs(
            styled_paths, dirty, style,
            global_context=saved.global_motion_prompt if saved else "",
            cast=cast_now,
            frame_people=frame_people,
        )

        transitions: list[Transition] = []
        replanned: list[str] = []
        for i, (a, b) in enumerate(pairs):
            tid = f"{a.id}_to_{b.id}"
            if i in plans:
                motion, duration, sound = plans[i]
                prior = saved_tr.get((a.output_path, b.output_path))
                # Either half of the plan invalidates the rendered clip: the
                # DURATION alone changing (a re-plan that keeps the wording
                # but stretches the pair from 5s to 10s, which is exactly
                # what confirmed identities can do via swap detection) leaves
                # a 5-second clip on disk for a 10-second plan.
                plan_changed = prior is not None and (
                    motion != prior.motion_prompt or duration != prior.duration
                )
                if plan_changed and (
                    prior.motion_prompt == self.config.motion_prompt
                    or tid in requested
                ):
                    # A genuinely new prompt landed where a clip may already
                    # exist — a real plan replacing a placeholder, or an
                    # explicitly requested re-plan. Flag the clip outdated
                    # downstream (marking only — regeneration stays manual).
                    stale_tids.append(tid)
                transitions.append(
                    Transition(
                        id=tid,
                        start_frame=a.output_path,
                        end_frame=b.output_path,
                        motion_prompt=motion,
                        duration=duration,
                        sound_prompt=sound,
                        output_path=f"clips/{tid}.mp4",
                    )
                )
                replanned.append(tid)
            else:
                transitions.append(saved_tr[(a.output_path, b.output_path)])

        storyboard = Storyboard(
            project_title=saved.project_title if saved else self.workspace.root.name,
            style=style,
            duration_per_clip=self.options.duration
            or (saved.duration_per_clip if saved else self.config.duration),
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            global_motion_prompt=saved.global_motion_prompt if saved else "",
            characters=cast,
            frames=frames,
            transitions=transitions,
        )
        # Stamp each freshly planned prompt with the identity facts it was
        # planned from, so a later tag or renamed cast member shows up as
        # "this prompt doesn't know about that yet" instead of being
        # something you have to remember. Carried-over transitions keep
        # whatever stamp they already had — they were not re-planned.
        for transition in storyboard.transitions:
            if transition.id in replanned:
                transition.planned_identity = identity_fingerprint(
                    storyboard, transition
                )
        return storyboard, replanned, stale_tids

    def _plan_pairs(
        self,
        styled: list[Path],
        dirty: list[int],
        style: str,
        global_context: str = "",
        cast: Optional[list[Character]] = None,
        frame_people: Optional[list[Optional[list[str]]]] = None,
    ) -> tuple[dict[int, tuple[str, int, str]], list[Character]]:
        """Vision-plan the dirty pairs only: {pair index: (motion, dur, sound)}.

        Consecutive dirty pairs are analysed together in one call containing
        just the frames involved, so an inserted image costs one small vision
        request instead of re-analysing the whole movie. Falls back to the
        global motion prompt per pair when analysis is off (--no-analyze),
        during a dry-run, or if a call fails — a planning hiccup never sinks
        the run.

        Also returns the movie's cast list: the one passed in (the saved
        storyboard's), extended by any new people the planning calls
        discovered. Each call receives the cast as it stands so epithets stay
        identical across separately planned segments.
        """
        cast = list(cast or [])
        if not dirty:
            return {}, cast
        fallback = (
            self._motion_prompt(),
            self.options.duration or self.config.duration,
            "",
        )
        if self.dry_run or not self.options.analyze_frames:
            return {i: fallback for i in dirty}, cast
        # Rules distilled from clips that came back wrong on earlier runs.
        # They join the planner's system prompt, so a targeted re-plan of one
        # pair gets exactly the same corrections a full plan does.
        lessons = self._lesson_texts(SCOPE_MOTION)
        if lessons:
            logger.info(
                "Applying %d learned lesson(s) from earlier renders to this plan.",
                len(lessons),
            )
        tags = list(frame_people or [None] * len(styled))
        tagged_count = sum(1 for t in tags if t is not None)
        if tagged_count:
            logger.info(
                "Using confirmed identities for %d of %d frame(s).",
                tagged_count, len(styled),
            )
        plans: dict[int, tuple[str, int, str]] = {}
        for run in _consecutive_runs(dirty):
            segment = styled[run[0]: run[-1] + 2]
            logger.info(
                "Analysing %d frame(s) to plan %d transition(s)...",
                len(segment), len(run),
            )
            try:
                seg_plans, cast = self.openai.analyze_frame_transitions(
                    segment, style,
                    default_duration=self.options.duration,
                    global_context=global_context,
                    cast=cast,
                    lessons=lessons,
                    # Tags for exactly the frames in this segment.
                    frame_people=tags[run[0]: run[-1] + 2],
                )
                for offset, i in enumerate(run):
                    plans[i] = seg_plans[offset]
            except Exception as exc:  # noqa: BLE001 - planning is best-effort
                logger.warning(
                    "Frame analysis failed (%s); using the default motion "
                    "prompt for %d transition(s).", exc, len(run),
                )
                for i in run:
                    plans[i] = fallback
        return plans, cast

    def mark_clips_outdated(self, tids: list[str]) -> list[str]:
        """Public entry point for the admin API; returns the clip files marked.

        Used when the panel saves hand-edited transitions: the same
        "outdated, kept, never auto-rendered" contract as a re-plan.
        """
        return self._mark_stale_clips(tids)

    def _mark_stale_clips(self, stale_tids: list[str]) -> list[str]:
        """Flag rendered clips that no longer match the updated storyboard.

        NEVER deletes and never triggers regeneration: rendered clips cost
        real credits, and any automatic deletion is one auto-confirming
        caller away from a wipeout (the admin API's always-yes confirm once
        deleted 26 rendered clips on a storyboard re-run). Staleness is
        recorded in state (``stale:<clip>``), surfaced by status/snapshot,
        and cleared when the clip is regenerated — redoing a clip is always
        a manual, per-clip decision (``render --clip ID`` / the panel's
        regenerate button).
        """
        stale = [
            clip for tid in stale_tids
            if (clip := self.workspace.clips_dir / f"{tid}.mp4").exists()
        ]
        if not stale:
            return []
        if self.dry_run:
            for clip in stale:
                logger.info("[dry-run] would mark clip %s outdated", clip.name)
            return [clip.name for clip in stale]
        for clip in stale:
            self.state.set(f"stale:{clip.name}", "outdated")
        logger.info(
            "%d rendered clip(s) no longer match the updated storyboard: %s — "
            "they are KEPT and render will not redo them by itself. "
            "Regenerate one with `render %s --clip <ID>` (or the admin "
            "panel's regenerate button) when you want the new plan applied.",
            len(stale), ", ".join(c.name for c in stale),
            self.workspace.root.name,
        )
        return [clip.name for clip in stale]

    # ------------------- storyboard from an idea (Mode B) ----------------- #
    def _resolve_idea(self) -> str:
        """Get the idea text from --idea-file (preferred) or --idea."""
        if self.options.idea_file:
            path = Path(self.options.idea_file)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if not path.exists():
                raise PipelineError(f"--idea-file not found: {path}")
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                raise PipelineError(f"--idea-file is empty: {path}")
            return text
        assert self.options.idea is not None
        return self.options.idea

    def _create_storyboard_from_idea(self) -> Optional[Storyboard]:
        """Write a storyboard from --idea / --idea-file. None on dry-run.

        An existing storyboard is reused (never silently overwritten) unless
        --force is passed.
        """
        sb_path = self.workspace.default_storyboard_json
        if sb_path.exists() and not self.force:
            logger.info(
                "Storyboard already exists at %s; using it. Pass --force to "
                "regenerate it from the idea.", sb_path,
            )
            return Storyboard.load(sb_path)

        idea = self._resolve_idea()
        # Frame count precedence: --frame-count, else config.default_frame_count.
        # A value <= 0 means "let the model choose based on the content".
        frame_count = (
            self.options.frame_count
            if self.options.frame_count is not None
            else self.config.default_frame_count
        )
        count_desc = f"{frame_count} frames" if frame_count and frame_count > 0 else "auto frame count"
        logger.info("Creating storyboard (%s) from idea (%d chars)", count_desc, len(idea))

        if self.dry_run:
            logger.info("[dry-run] would call OpenAI to build a storyboard and "
                        "write %s + %s",
                        self.workspace.default_storyboard_json,
                        self.workspace.storyboard_md)
            return None

        storyboard = self.openai.create_storyboard(
            idea, frame_count, default_duration=self.options.duration
        )
        self._save_storyboard(storyboard)
        return storyboard

    def _save_storyboard(self, storyboard: Storyboard) -> None:
        """Write the storyboard JSON plus its readable md/html views."""
        storyboard.save(self.workspace.default_storyboard_json)
        write_storyboard_markdown(storyboard, self.workspace.storyboard_md)
        write_storyboard_preview(
            storyboard, self.workspace.root, self.workspace.storyboard_preview
        )

    def _announce_storyboard_ready(self) -> None:
        """Tell the user the storyboard is written and how to continue."""
        print("\n" + "=" * 70)
        print("Storyboard ready. Review it:")
        print(f"  open {self.workspace.storyboard_preview}   (visual contact sheet)")
        print(f"  {self.workspace.default_storyboard_json}   (edit clips here)")
        print("\nThen generate the clips with:")
        print(f"  {self._next_command('render')}")
        print("=" * 70 + "\n")

    # ------------------------------ render step --------------------------- #
    def cmd_render(self) -> None:
        """Generate clips (and any missing generated frames) from the storyboard.

        --clip NNN_to_NNN limits the run to the named clip(s) and regenerates
        them even if they exist (that's the point of naming them); their
        SFX/fade state is reset so the redone clips get fresh audio.
        """
        storyboard = self._require_storyboard("render")
        self.summary.input_count = len(storyboard.frames)

        self._generate_frames(storyboard)

        pairs = self._pairs_from_storyboard(storyboard)
        pairs, forced = self._select_clips(pairs)
        self._render_pairs(pairs, forced)

    def _require_storyboard(self, command: str) -> Storyboard:
        path = self.workspace.default_storyboard_json
        if not path.exists():
            raise PipelineError(
                f"No storyboard yet ({path} not found). Create one first:\n"
                f"  {self._next_command('storyboard')}"
            )
        return Storyboard.load(path)

    def _select_clips(
        self, pairs: list[ClipPair]
    ) -> tuple[list[ClipPair], set[str]]:
        """Apply --clip selection. Returns (pairs to process, forced stems)."""
        requested = self.options.clips
        if not requested:
            return pairs, set()
        by_stem = {
            self._clip_name(pair[0], pair[1]).stem: pair for pair in pairs
        }
        wanted = [c.removesuffix(".mp4") for c in requested]
        unknown = [c for c in wanted if c not in by_stem]
        if unknown:
            raise PipelineError(
                f"Unknown clip(s): {', '.join(unknown)}. "
                f"Available: {', '.join(by_stem) or '(none)'}"
            )
        selected = [by_stem[c] for c in wanted]
        return selected, set(wanted)

    def _plan_lines(
        self, pairs: list[ClipPair], forced: set[str]
    ) -> tuple[list[str], int]:
        """Human-readable per-clip plan + how many clips will actually render."""
        lines = ["Clip plan:"]
        to_render = 0
        seconds = 0
        for start, end, motion, duration, _sound in pairs:
            dst = self._clip_name(start, end)
            if dst.exists() and not self.force and dst.stem not in forced:
                status = "done, skip"
            else:
                status = "RENDER"
                to_render += 1
                seconds += duration
            m = motion if len(motion) <= 68 else motion[:65] + "..."
            lines.append(f"  {dst.stem:<12} {duration:>2}s  {status:<10} {m}")
        lines.append(
            f"  -> {to_render} clip(s) to render (~{seconds}s of new video); "
            "this step spends video-provider credits."
        )
        return lines, to_render

    def _render_pairs(self, pairs: list[ClipPair], forced: set[str]) -> None:
        if not pairs:
            logger.warning("No transition pairs to render.")
            return
        plan_lines, to_render = self._plan_lines(pairs, forced)
        for line in plan_lines:
            logger.info("%s", line)
        if to_render == 0 and not self.audio_enabled:
            logger.info("All clips are already rendered; nothing to do.")
            return
        if to_render > 0 and not self._ask(
            plan_lines,
            f"Generate {to_render} clip(s) now? [y/N] ",
            "Clip generation skipped. Continue later with:\n  "
            + self._next_command("render"),
        ):
            return
        self._generate_clips(pairs, forced)

    def _generate_frames(self, storyboard: Storyboard) -> None:
        """Generate any frame that has an image prompt and is missing on disk.

        Image-based frames (styled images) have no image prompt and are never
        touched here; idea-based frames are (re)generated when missing or
        --force.
        """
        todo = [f for f in storyboard.frames if f.image_prompt.strip()]
        if not todo:
            return

        def work(frame: Frame) -> None:
            dst = self.workspace.root / frame.output_path
            job_id = f"frame:{frame.id}"

            if dst.exists() and not self.force:
                with self._lock:
                    self.summary.styled_skipped += 1
                logger.info("Skip frame (done): %s", dst.name)
                return

            # Reinforce style consistency in every prompt.
            full_prompt = (
                f"{frame.image_prompt}\n\nStyle: {storyboard.style}"
            )
            if frame.negative_prompt:
                full_prompt += f"\n\nAvoid: {frame.negative_prompt}"
            if self.config.avoid_text_only_frames:
                full_prompt += (
                    "\n\nIMPORTANT: This must be a full visual scene, NOT a "
                    "title/caption card. Do NOT produce a blank, black, or "
                    "solid-colour background containing only text. (Text is fine "
                    "when it appears naturally within a real scene.)"
                )

            if self.dry_run:
                logger.info("[dry-run] would generate frame %s -> %s", frame.id, dst.name)
                with self._lock:
                    self.summary.styled_created += 1
                return

            try:
                self.openai.generate_image(full_prompt, dst)
                if not verify_dimensions(dst, self.config.target_width, self.config.target_height):
                    dst.unlink(missing_ok=True)  # or the next run skips it as done
                    raise RuntimeError(f"{dst.name} is not {self.config.target_width}x{self.config.target_height}")
                with self._lock:
                    self.state.set(job_id, "done", output=str(dst))
                    self.summary.styled_created += 1
                logger.info("Generated frame: %s", dst.name)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.summary.styled_failed += 1
                    self.state.set(job_id, "failed")
                    self.failed.record(job_id, "frame", str(exc), frame_id=frame.id)

        self._map_parallel(todo, work, "Generating frames", "frame")

    def _pairs_from_storyboard(self, storyboard: Storyboard) -> list[ClipPair]:
        """Build (start, end, motion, duration, sound) clip pairs from a storyboard.

        Surviving frames are paired in order, bridging over any that are missing
        on disk. Each pair takes its motion/duration/sound from the transition
        leaving its start frame (for a bridged pair, that frame's original
        outgoing one); --motion-prompt / --duration override per run.
        """
        transitions = storyboard.transitions or self._derive_transitions(storyboard)
        tr_by_start: dict[str, Transition] = {
            (self.workspace.root / tr.start_frame).name: tr for tr in transitions
        }
        frames_ordered = [self.workspace.root / f.output_path for f in storyboard.frames]

        global_motion = storyboard.global_motion_prompt.strip()
        if len(global_motion.split()) > 25:
            logger.warning(
                "global_motion_prompt is %d words — it is prepended to EVERY "
                "clip's motion prompt, and overlong prompts degrade renders "
                "(word caps exist for a reason). Consider trimming it to a "
                "sentence or two.",
                len(global_motion.split()),
            )

        pairs: list[ClipPair] = []
        for a, b in self._bridge_pairs(frames_ordered):
            tr = tr_by_start.get(a.name)
            motion = self.options.motion_prompt or (
                tr.motion_prompt if tr else storyboard.style
            )
            duration = self.options.duration or (
                tr.duration if tr else storyboard.duration_per_clip
            )
            sound = tr.sound_prompt if tr else ""
            pairs.append((a, b, _with_global_motion(global_motion, motion), duration, sound))
        return pairs

    @staticmethod
    def _derive_transitions(storyboard: Storyboard) -> list[Transition]:
        derived: list[Transition] = []
        frames = storyboard.frames
        for a, b in zip(frames, frames[1:]):
            tid = f"{a.id}_to_{b.id}"
            derived.append(
                Transition(
                    id=tid,
                    start_frame=a.output_path,
                    end_frame=b.output_path,
                    motion_prompt=storyboard.style,
                    duration=storyboard.duration_per_clip,
                    output_path=f"clips/{tid}.mp4",
                )
            )
        return derived

    def _motion_prompt(self) -> str:
        return self.options.motion_prompt or self.config.motion_prompt

    def _bridge_pairs(
        self, ordered: list[Path], *, quiet: bool = False
    ) -> list[tuple[Path, Path]]:
        """Pair consecutive frames, bridging over any that are missing on disk.

        If a frame failed to generate, it is skipped and its nearest existing
        neighbours are paired directly (e.g. frame 4 missing -> ...3->5...), so
        the final video stays continuous instead of leaving a gap. During a
        dry-run the files don't exist yet, so the naive full pairing is used for
        the plan.

        `quiet` is for `snapshot()`, which calls this purely to work out the
        clip list. Status is read on every panel poll, for every project, so
        warning from there filled the log with an alarming line that named
        neither the project nor the frames — one landed in the middle of
        another project's styling run and read as if it belonged to it. The
        condition is reported as DATA instead (`missing_frames`).
        """
        if self.dry_run:
            existing = list(ordered)
        else:
            existing = [p for p in ordered if p.exists()]
            missing = [p.name for p in ordered if not p.exists()]
            if missing and not quiet:
                logger.warning(
                    "%s: %d frame(s) have no styled image (%s); bridging over "
                    "them by pairing the nearest existing neighbours. Those "
                    "photos will NOT appear in the movie.",
                    self.workspace.root.name, len(missing), ", ".join(missing[:8])
                    + (f", …+{len(missing) - 8} more" if len(missing) > 8 else ""),
                )
        return [(existing[i], existing[i + 1]) for i in range(len(existing) - 1)]

    def _generate_clips(self, pairs: list[ClipPair], forced: set[str]) -> None:
        def work(pair: ClipPair) -> None:
            start, end, motion, duration, sound_prompt = pair
            dst = self._clip_name(start, end)
            job_id = f"clip:{dst.name}"
            redo = self.force or dst.stem in forced

            if self.dry_run:
                # Frames may not exist yet during a dry-run (styling was also
                # dry-run), so report the plan without checking for them.
                logger.info(
                    "[dry-run] would render %s (%ss): %s -> %s | motion=%r",
                    dst.name, duration, start.name, end.name, motion,
                )
                with self._lock:
                    self.summary.videos_created += 1
                if self.audio_enabled:
                    logger.info(
                        "[dry-run] would add SFX to %s | sound=%r",
                        dst.name, sound_prompt or self.config.default_sfx_prompt,
                    )
                return

            if dst.exists() and not redo:
                with self._lock:
                    self.summary.videos_skipped += 1
                logger.info("Skip clip (done): %s", dst.name)
            else:
                if not start.exists() or not end.exists():
                    with self._lock:
                        self.summary.videos_failed += 1
                        self.failed.record(
                            job_id, "clip",
                            f"Missing frame(s): {start.name} / {end.name}",
                        )
                    return
                try:
                    self.video_client.generate_clip(
                        start, end, motion, duration, dst,
                        reword=self.openai.reword_motion_prompt,
                    )
                    # Booked on the finished clip, not on submission: a job
                    # that was submitted and billed but never collected shows
                    # up as a pending render (money the ledger can't see yet)
                    # and lands here when it is finally downloaded.
                    self._record_cost(
                        KIND_CLIP,
                        self.config.pricing.clip(duration),
                        dst.name,
                        seconds=duration,
                    )
                    with self._lock:
                        # A fresh clip file invalidates its per-clip audio work:
                        # without this, a regenerated clip would skip SFX/fade
                        # ("done" from the previous file) and come out silent.
                        # It also matches the current storyboard again, so any
                        # "outdated" flag from a re-plan is lifted.
                        self.state.clear(
                            f"sfx:{dst.name}", f"fade:{dst.name}",
                            f"stale:{dst.name}",
                        )
                        self.state.set(job_id, "done", output=str(dst))
                        self.summary.videos_created += 1
                    logger.info("Clip ready: %s", dst.name)
                except PipelineCancelled:
                    # Asked to stop, not a broken clip: don't file it as a
                    # failure (the submitted job is still recoverable).
                    raise
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        self.summary.videos_failed += 1
                        self.state.set(job_id, "failed")
                        self.failed.record(
                            job_id, "clip", str(exc),
                            start=str(start), end=str(end),
                        )
                    return

            # Per-clip SFX/ambient sound (replaces the clip with an audio-bearing
            # version). Tracked separately so it resumes independently of video.
            if self.audio_enabled and dst.exists():
                self._add_sfx(dst, sound_prompt, duration)

        self._map_parallel(list(pairs), work, "Generating clips", "clip")

    @staticmethod
    def _frame_id(frame: Path) -> str:
        """Frame id from a frame filename.

        'img4a.png' -> 'img4a' (filename-keyed); legacy '001_styled.png' ->
        '001'; idea-based '001.png' -> '001'.
        """
        stem = frame.stem
        return stem[: -len("_styled")] if stem.endswith("_styled") else stem

    def _clip_name(self, start: Path, end: Path) -> Path:
        """Map a frame pair to clips/<startid>_to_<endid>.mp4."""
        return self.workspace.clips_dir / (
            f"{self._frame_id(start)}_to_{self._frame_id(end)}.mp4"
        )

    # ------------------------------ audio step ---------------------------- #
    def cmd_audio(self) -> None:
        """Add SFX + music to already-rendered clips, then rebuild the final video.

        Per-clip SFX prompts come from the saved storyboard when there is one;
        otherwise every clip uses config.default_sfx_prompt. --clip limits the
        run to the named clip(s) and REDOES their audio even if marked done —
        that's how an edited sound_prompt is applied to one clip without
        touching the rest.
        """
        self.audio_enabled = True
        clips = self._clips_for_combine()
        if not clips:
            logger.warning("No clips in %s to add audio to.", self.workspace.clips_dir)
            return
        clips = self._select_audio_clips(clips)

        sound_map: dict[str, str] = {}
        sb_path = self.workspace.default_storyboard_json
        if sb_path.exists():
            try:
                sb = Storyboard.load(sb_path)
                for tr in sb.transitions:
                    sound_map[Path(tr.output_path).name] = tr.sound_prompt
                logger.info("Using per-clip sound prompts from %s", sb_path.name)
            except StoryboardError:
                logger.warning("Could not read %s; using default SFX prompt.", sb_path)

        def work(clip: Path) -> None:
            if self.dry_run:
                logger.info("[dry-run] would add SFX to %s", clip.name)
                return
            duration = int(round(ffprobe_duration(clip) or self.duration))
            self._add_sfx(clip, sound_map.get(clip.name, ""), duration)

        self._map_parallel(list(clips), work, "Adding clip SFX", "clip")

        # Rebuild the final video so the new audio is included, then add music.
        self._combine_clips(force_rebuild=True)

    def _select_audio_clips(self, clips: list[Path]) -> list[Path]:
        """Apply --clip selection for the audio step.

        Named clips get their sfx/fade state cleared so the audio is redone,
        not skipped as "done" from an earlier run.
        """
        requested = self.options.clips
        if not requested:
            return clips
        by_stem = {c.stem: c for c in clips}
        wanted = [c.removesuffix(".mp4") for c in requested]
        unknown = [c for c in wanted if c not in by_stem]
        if unknown:
            raise PipelineError(
                f"Unknown clip(s): {', '.join(unknown)}. "
                f"Available: {', '.join(by_stem) or '(none)'}"
            )
        selected = [by_stem[c] for c in wanted]
        if not self.dry_run:
            for clip in selected:
                self.state.clear(f"sfx:{clip.name}", f"fade:{clip.name}")
        logger.info("Redoing audio for %d selected clip(s).", len(selected))
        return selected

    def _add_sfx(self, clip: Path, sound_prompt: str, duration: int) -> None:
        """Run the video->audio model on `clip`, replacing it with a sounded one."""
        job_id = f"sfx:{clip.name}"
        if self.state.is_done(job_id) and not self.force:
            with self._lock:
                self.summary.sfx_skipped += 1
            logger.info("Skip SFX (done): %s", clip.name)
        else:
            prompt = sound_prompt.strip() or self.config.default_sfx_prompt
            try:
                self.audio_client.add_sfx(clip, prompt, duration)
                self._record_cost(KIND_SFX, self.config.pricing.sfx_usd, clip.name)
                with self._lock:
                    self.state.set(job_id, "done")
                    self.summary.sfx_created += 1
                logger.info("SFX added: %s", clip.name)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self.summary.sfx_failed += 1
                    self.state.set(job_id, "failed")
                    self.failed.record(job_id, "sfx", str(exc), clip=str(clip))
                return  # no audio to fade

        # Edge-fade is tracked separately so it can be applied to clips that
        # already have SFX (e.g. from an earlier run) WITHOUT re-paying for it,
        # and so it runs exactly once per clip.
        self._fade_clip(clip)

    def _fade_clip(self, clip: Path) -> None:
        """Apply the boundary edge-fade once; safe to call on any sounded clip."""
        if self.config.sfx_fade_seconds <= 0:
            return
        job_id = f"fade:{clip.name}"
        if self.state.is_done(job_id) and not self.force:
            return
        try:
            apply_edge_fades(clip, self.config.sfx_fade_seconds)
            self.state.set(job_id, "done")
        except Exception as exc:  # noqa: BLE001
            # A fade failure must not discard the (already generated) SFX; leave
            # it unmarked so a later run retries.
            logger.warning("Edge-fade skipped for %s: %s", clip.name, exc)

    # ----------------------------- combine step --------------------------- #
    def cmd_combine(self) -> None:
        """Concatenate the storyboard's clips into the final video."""
        self._combine_clips()

    def _clips_for_combine(self) -> list[Path]:
        """The clips that belong in the final video, in order.

        Derived from the saved storyboard when there is one: existing frames are
        bridge-paired and mapped to their clip files, so stale clips — e.g. a
        bridged 003_to_005.mp4 left over from before frame 004 was fixed, or
        clips from images that were since removed — are never folded into the
        movie. Falls back to the directory listing when no storyboard exists
        (hand-managed clips are still combinable).
        """
        found = find_generated_clips(self.workspace.clips_dir)
        sb_path = self.workspace.default_storyboard_json
        if not sb_path.exists():
            return found
        try:
            storyboard = Storyboard.load(sb_path)
        except StoryboardError as exc:
            logger.warning(
                "Could not read %s (%s); combining every clip in clips/.",
                sb_path.name, exc,
            )
            return found

        frames = [self.workspace.root / f.output_path for f in storyboard.frames]
        expected = [self._clip_name(a, b) for a, b in self._bridge_pairs(frames)]
        clips = [c for c in expected if c.exists()]
        stray = sorted(set(found) - set(expected))
        if stray:
            logger.warning(
                "Ignoring %d clip(s) in %s that don't match the current "
                "storyboard: %s (delete them if they're stale).",
                len(stray), self.workspace.clips_dir.name,
                ", ".join(p.name for p in stray),
            )
        return clips

    def _combine_clips(
        self, force_rebuild: bool = False, confirm: bool = False
    ) -> None:
        """Concatenate the storyboard's clips into output/final_video.mp4.

        When `confirm` is set (the end-of-`run` path) the user is asked first —
        but only once we know there's actually a movie to build, so the prompt
        never appears when there are no clips or the final video is already up
        to date.
        """
        clips = self._clips_for_combine()
        if not clips:
            logger.info("No clips to combine; skipping final video.")
            return

        final_video = self.workspace.final_video
        if self.dry_run:
            logger.info(
                "[dry-run] would combine %d clip(s) into %s",
                len(clips), final_video,
            )
            intro, credits, letter = self._presentation_flags()
            if intro:
                logger.info("[dry-run] would prepend the intro clip from %s",
                            self._intro_source())
            if credits:
                logger.info("[dry-run] would append the original photos as "
                            "an end-credits montage")
            if letter:
                logger.info("[dry-run] would append the scrolling closing "
                            "letter from %s", self.workspace.letter_file)
            return

        if final_video.exists() and not self.force and not force_rebuild:
            logger.info(
                "Final video already exists (use --force to rebuild): %s",
                final_video,
            )
            return

        if confirm and not self._ask(
            [
                "All clips are ready. The final step combines them into "
                f"{final_video.name}.",
            ],
            "Combine clips into the final video now? [y/N] ",
            "Combine skipped. Build the final video later with:\n  "
            + self._next_command("combine"),
        ):
            return

        # Decide the music track BEFORE combining, so every choice happens up
        # front rather than interrupting after the combine runs.
        music_file = self._resolve_music_file() if self.audio_enabled else None

        segments, has_photo_segments = self._presentation_segments(clips)

        logger.info("Combining %d segment(s) into %s", len(segments), final_video)
        try:
            combine_clips(segments, final_video, force_filter=has_photo_segments)
            self.summary.final_video = final_video
            logger.info("Final video ready: %s", final_video)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to combine clips: %s", exc)
            self.failed.record("combine", "combine", str(exc))
            return

        # Lay the chosen music bed under the whole video (ducked under the SFX).
        if self.audio_enabled:
            self._add_music(music_file)

        # Last touch: fade the closing moments to black, audio included.
        if self.config.end_fade_seconds > 0:
            try:
                apply_end_fade(final_video, self.config.end_fade_seconds)
                logger.info(
                    "Faded the last %.1fs to black.", self.config.end_fade_seconds
                )
            except Exception as exc:  # noqa: BLE001 - cosmetic, never fatal
                logger.warning("End fade skipped: %s", exc)

    def _presentation_flags(self) -> tuple[bool, bool, bool]:
        """(intro_clip, credits_photos, closing_letter): CLI wins over config."""
        intro = (
            self.options.intro_clip
            if self.options.intro_clip is not None
            else self.config.intro_clip
        )
        credits = (
            self.options.credits_photos
            if self.options.credits_photos is not None
            else self.config.credits_photos
        )
        letter = (
            self.options.closing_letter
            if self.options.closing_letter is not None
            else self.config.closing_letter
        )
        return intro, credits, letter

    def _original_photo_sources(self) -> list[tuple[str, Path]]:
        """(frame id, original photo path) per storyboard frame, in order.

        Only frames whose recorded source_path still exists count — idea-based
        (Mode B) frames and pre-source_path legacy storyboards yield nothing,
        which callers treat as 'feature not applicable'.
        """
        sb_path = self.workspace.default_storyboard_json
        if not sb_path.exists():
            return []
        try:
            storyboard = Storyboard.load(sb_path)
        except StoryboardError:
            return []
        sources: list[tuple[str, Path]] = []
        for frame in storyboard.frames:
            if not frame.source_path:
                continue
            photo = self.workspace.root / frame.source_path
            if photo.exists():
                fid = self._frame_id(self.workspace.root / frame.output_path)
                sources.append((fid, photo))
        return sources

    def _presentation_segments(self, clips: list[Path]) -> tuple[list[Path], bool]:
        """Wrap the clip list with the optional presentation segments.

        intro_clip prepends the user's own intro video before everything;
        credits_photos appends one still per original photo. Segments are
        cheap local ffmpeg renders, so they're rebuilt on every combine (no
        state tracking to invalidate). Returns the segment list plus whether
        photo segments were added — the concat must then re-encode instead of
        stream-copying, because the stills' encoding differs from the
        provider clips'.
        """
        intro, credits, letter = self._presentation_flags()
        if not (intro or credits or letter) or not clips:
            return clips, False
        sources = self._original_photo_sources() if credits else []
        if credits and not sources:
            logger.warning(
                "credits_photos is on, but this project's storyboard records "
                "no original photos (source_path) — skipping it. (Re-run "
                "`storyboard` on an image-based project to record the "
                "sources.)"
            )
            credits = False
        if intro and not self._intro_source().exists():
            logger.warning(
                "intro_clip is on, but %s does not exist — drop your intro "
                "video there (it's shared by every project; the intro_file "
                "config key relocates it) and re-run combine.",
                self._intro_source(),
            )
            intro = False
        letter_text = self._letter_text() if letter else None
        if letter and letter_text is None:
            letter = False
        if not (intro or credits or letter):
            return clips, False

        seg_dir = self.workspace.output_dir / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)
        width, height = self.config.target_width, self.config.target_height
        segments = list(clips)
        added = False

        if intro:
            section = self._render_intro_segment(seg_dir, width, height)
            if section is not None:
                segments.insert(0, section)
                added = True

        if credits and letter:
            # The letter scrolls OVER the photo montage (dimmed for
            # readability), both paced to end together.
            section = self._render_credits_letter_overlay(
                seg_dir, sources, letter_text, width, height
            )
            if section is not None:
                segments.append(section)
                added = True
        elif credits:
            stills = self._render_credit_stills(
                seg_dir, sources, self.config.credits_seconds_per_photo,
                width, height,
            )
            segments += stills
            added = True
            logger.info("End credits: %d original photo(s) appended.", len(stills))
        elif letter:
            section = self._render_letter_standalone(
                seg_dir, letter_text, width, height
            )
            if section is not None:
                segments.append(section)
                added = True

        return segments, added

    # Segment reuse: a rendered segment is kept as long as it's newer than
    # everything it was built from (its media inputs + the config files that
    # hold sizes/paces/fonts). Delete output/segments/ to force a full redo.
    def _segment_deps(self, *media: Path) -> list[Path]:
        return [
            *media,
            PROJECT_ROOT / "config.json",
            self.workspace.root / "config.json",
        ]

    def _segment_fresh(
        self, dst: Path, media_deps: list[Path], recipe: str = ""
    ) -> bool:
        """Can this rendered section be reused instead of re-rendered?

        Three ways it goes stale, and all three are needed:

        * ``--force`` — the operator asked for a rebuild;
        * one of its inputs is newer than it (the photos, letter.txt, config);
        * the RECIPE changed — the settings and the renderer version that
          produced it. Without this, upgrading the pipeline left every
          project's ending frozen at whatever the old code made: a real
          combine after fixing the letter's emoji and the greyed-out photos
          came back byte-identical, because letter.txt and the photos had
          not been touched.
        """
        if self.force or not dst.exists():
            return False
        mtime = dst.stat().st_mtime
        if any(
            dep.exists() and dep.stat().st_mtime > mtime
            for dep in self._segment_deps(*media_deps)
        ):
            return False
        if not recipe:
            return True
        sidecar = self._segment_recipe_file(dst)
        if not sidecar.exists():
            return False  # made before recipes existed: re-render once
        try:
            return sidecar.read_text(encoding="utf-8").strip() == recipe
        except OSError:
            return False

    @staticmethod
    def _segment_recipe_file(dst: Path) -> Path:
        """Sidecar recording HOW a rendered section was made."""
        return dst.with_suffix(dst.suffix + ".recipe")

    def _record_segment_recipe(self, dst: Path, recipe: str) -> None:
        """Remember the recipe so a settings/code change invalidates it."""
        if not recipe:
            return
        try:
            self._segment_recipe_file(dst).write_text(recipe, encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - caching is never fatal
            logger.debug("Could not record the segment recipe: %s", exc)

    def _segment_recipe(self, kind: str, **parts: Any) -> str:
        """A short fingerprint of everything that shapes a rendered section.

        ``_SEGMENT_RENDERER_VERSION`` is part of it, so a change to HOW
        sections are drawn invalidates them even when every input file is
        untouched — bump it whenever the drawing changes.
        """
        payload = json.dumps(
            {"kind": kind, "renderer": _SEGMENT_RENDERER_VERSION, **parts},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _letter_text(self) -> Optional[str]:
        """The letter's text, or None (with a warning) when there isn't one."""
        path = self.workspace.letter_file
        if not path.exists():
            logger.warning(
                "closing_letter is on, but %s does not exist — write your "
                "letter there (plain text, Hebrew is fine) and re-run combine.",
                path,
            )
            return None
        text = read_letter(path).strip()
        if not text:
            logger.warning("closing_letter: %s is empty; skipping.", path)
            return None
        return text

    def _intro_source(self) -> Path:
        """The shared intro video: config.intro_file, repo-root relative."""
        path = Path(self.config.intro_file)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _render_intro_segment(
        self, seg_dir: Path, width: int, height: int
    ) -> Optional[Path]:
        """The user's intro clip, normalized to the movie's frame size."""
        src = self._intro_source()
        dst = seg_dir / "intro.mp4"
        recipe = self._segment_recipe("intro", src=src.name, size=(width, height))
        if self._segment_fresh(dst, [src], recipe):
            logger.info("Reusing intro clip (unchanged).")
            return dst
        try:
            render_intro_segment(src, dst, width, height)
        except Exception as exc:  # noqa: BLE001 - extras must not kill combine
            logger.error("Intro clip failed (%s); combining without it.", exc)
            self.failed.record("intro", "intro", str(exc))
            return None
        self._record_segment_recipe(dst, recipe)
        logger.info("Intro: %s opens the movie.", src.name)
        return dst

    def _render_credit_stills(
        self,
        seg_dir: Path,
        sources: list[tuple[str, Path]],
        per_photo: float,
        width: int,
        height: int,
    ) -> list[Path]:
        """One still segment per original photo, reusing fresh ones.

        The duration is part of the filename, so a pace change (e.g. the
        letter overlay stretching the montage) naturally misses the cache
        instead of reusing stills of the wrong length.
        """
        stills: list[Path] = []
        rendered = 0
        recipe = self._segment_recipe(
            "credit_still", seconds=round(per_photo, 2), size=(width, height)
        )
        for i, (_fid, photo) in enumerate(sources):
            dst = seg_dir / f"credits_{i:03d}_{per_photo:.2f}s.mp4"
            if not self._segment_fresh(dst, [photo], recipe):
                render_photo_still(photo, dst, width, height, seconds=per_photo)
                self._record_segment_recipe(dst, recipe)
                rendered += 1
            stills.append(dst)
        if rendered < len(stills):
            logger.info(
                "Credit stills: reused %d, rendered %d.",
                len(stills) - rendered, rendered,
            )
        return stills

    def _render_credits_letter_overlay(
        self,
        seg_dir: Path,
        sources: list[tuple[str, Path]],
        text: str,
        width: int,
        height: int,
    ) -> Optional[Path]:
        """The combined ending: letter scrolling over the real-photo montage."""
        dst = seg_dir / "credits_letter.mp4"
        media_deps = [self.workspace.letter_file, *(p for _, p in sources)]
        recipe = self._segment_recipe(
            "credits_letter",
            text=text, photos=len(sources), size=(width, height),
            font=self.config.letter_font_path,
            emoji_font=self.config.letter_emoji_font_path,
            font_size=self.config.letter_font_size,
            pace=self.config.letter_seconds_per_screen,
            per_photo=self.config.credits_seconds_per_photo,
            dim=self.config.letter_overlay_dim,
        )
        if self._segment_fresh(dst, media_deps, recipe):
            logger.info("Reusing credits+letter section (unchanged).")
            return dst
        try:
            font = find_letter_font(self.config.letter_font_path)
            image = render_letter_image(
                text, width, height, font, self.config.letter_font_size,
                pad=False, transparent=True,
                emoji_font_path=find_emoji_font(
                    self.config.letter_emoji_font_path),
            )
            png = seg_dir / "letter.png"
            image.save(png)
            travel = image.height + height  # enters from below, exits above
            per_photo, pps = _fit_credits_and_letter(
                len(sources),
                self.config.credits_seconds_per_photo,
                travel,
                height / self.config.letter_seconds_per_screen,
            )
            stills = self._render_credit_stills(
                seg_dir, sources, per_photo, width, height
            )
            background = seg_dir / "credits_bg.mp4"
            combine_clips(stills, background, force_filter=True)
            render_letter_overlay(
                background, png, dst, pps, self.config.letter_overlay_dim
            )
        except Exception as exc:  # noqa: BLE001 - extras must not kill combine
            logger.error("Credits+letter section failed (%s); combining "
                         "without it.", exc)
            self.failed.record("letter", "letter", str(exc))
            return None
        self._record_segment_recipe(dst, recipe)
        logger.info(
            "Credits + letter: %d photo(s) under a ~%.0fs scrolling letter "
            "(font: %s).", len(sources), travel / pps, Path(font).name,
        )
        return dst

    def _render_letter_standalone(
        self, seg_dir: Path, text: str, width: int, height: int
    ) -> Optional[Path]:
        """The letter alone, scrolling over a dark background."""
        dst = seg_dir / "letter.mp4"
        recipe = self._segment_recipe(
            "letter", text=text, size=(width, height),
            font=self.config.letter_font_path,
            emoji_font=self.config.letter_emoji_font_path,
            font_size=self.config.letter_font_size,
            pace=self.config.letter_seconds_per_screen,
        )
        if self._segment_fresh(dst, [self.workspace.letter_file], recipe):
            logger.info("Reusing closing letter (unchanged).")
            return dst
        try:
            font = find_letter_font(self.config.letter_font_path)
            image = render_letter_image(
                text, width, height, font, self.config.letter_font_size,
                emoji_font_path=find_emoji_font(
                    self.config.letter_emoji_font_path),
            )
            png = seg_dir / "letter.png"
            image.save(png)
            render_letter_scroll(
                png, dst, width, height, image.height,
                pixels_per_second=height / self.config.letter_seconds_per_screen,
            )
        except Exception as exc:  # noqa: BLE001 - extras must not kill combine
            logger.error("Closing letter failed (%s); combining without it.", exc)
            self.failed.record("letter", "letter", str(exc))
            return None
        self._record_segment_recipe(dst, recipe)
        logger.info(
            "Closing letter: %d chars scrolled over ~%.0fs (font: %s).",
            len(text),
            (image.height - height)
            / (height / self.config.letter_seconds_per_screen),
            Path(font).name,
        )
        return dst

    def _resolve_music_file(self) -> Optional[Path]:
        """Decide which music track to lay under the final video.

        The music bed is always a track SUPPLIED by the user — nothing is
        generated. In order: --music-url (downloaded into the custom slot),
        --music-file (must exist), an uploaded custom track
        (output/music_custom.mp3, e.g. from the panel), then a pre-existing
        output/music.mp3 (a track left by an older run or dropped in by
        hand). Returns None to finish the movie without music, which is a
        normal outcome rather than a failure.
        """
        if self.options.music_url:
            # Stored in the custom slot so it behaves exactly like an upload
            # and survives into later combine runs without re-downloading.
            return fetch_music(
                self.options.music_url, self.workspace.custom_music_file
            )
        if self.options.music_file:
            supplied = Path(self.options.music_file).expanduser()
            if not supplied.exists():
                raise PipelineError(f"--music-file not found: {supplied}")
            logger.info("Using music file: %s", supplied)
            return supplied
        custom = self.workspace.custom_music_file
        if custom.exists():
            logger.info("Using uploaded custom music: %s", custom)
            return custom
        default = self.workspace.music_file
        if default.exists():
            logger.info("Reusing music file: %s", default)
            return default
        logger.info(
            "No music track supplied; the movie is built without a music bed. "
            "Upload one in the panel, or pass --music-file PATH."
        )
        return None

    def _add_music(self, music_file: Optional[Path]) -> None:
        """Mix `music_file` over output/final_video.mp4 (louder than the SFX)."""
        if music_file is None:
            return
        job_id = "music:final"
        try:
            mux_music(
                self.workspace.final_video,
                music_file,
                self.config.music_volume,
                self.config.sfx_volume,
                loop=self.config.music_loop,
            )
            self.state.set(job_id, "done")
            self.summary.music_added = True
            logger.info("Music bed added to %s", self.workspace.final_video)
        except Exception as exc:  # noqa: BLE001
            self.state.set(job_id, "failed")
            self.failed.record(job_id, "music", str(exc))

    # ----------------------------- publish step --------------------------- #
    def publish_plan(self, client: Optional[CloudinaryClient] = None) -> dict[str, Any]:
        """What the next publish would upload, and under exactly what name.

        Queries Cloudinary (free, read-only) so the name shown to whoever
        approves the publish is the real one. The version is one past the
        HIGHEST already used — as reported by Cloudinary *and* by this
        project's own published.json, merged — so neither a listing hiccup
        nor a hand-deleted version can make a publish land on a name that was
        used before.
        """
        ws = self.workspace
        order = read_order_record(ws.order_file)
        folder = str((order or {}).get("order_folder", "")).strip()
        if not folder:
            raise PipelineError(
                f"Project '{ws.root.name}' isn't tied to a Cloudinary order "
                "(no order.json), so there is no order folder to publish into. "
                "Only projects created by `ingest` can be published."
            )
        client = client or CloudinaryClient.from_config(self.config)
        remote = client.list_published_videos(folder)
        version = next_publish_version(
            [p.version for p in remote] + published_versions(ws.published_file)
        )
        state = publish_state(ws.published_file, ws.final_video)
        return {
            "order_folder": folder,
            "version": version,
            "public_id": publish_public_id(
                client.orders_folder, folder, client.publish_basename, version
            ),
            "filename": f"{client.publish_basename}_v{version}.mp4",
            "final_video": ws.final_video.exists(),
            "bytes": ws.final_video.stat().st_size if ws.final_video.exists() else 0,
            "published": [
                {
                    "version": p.version, "public_id": p.public_id,
                    "url": p.url, "bytes": p.bytes, "created_at": p.created_at,
                }
                for p in remote
            ],
            # False when the movie on disk is byte-for-byte the one already
            # published: publishing again is allowed (it just makes another
            # version), but whoever approves it should know it adds nothing.
            "changed_since_last": state["changed_since"],
            "latest": state["latest"],
            # Whether the delivered bytes will also be archived locally, so
            # the confirmation can say what actually happens on this machine.
            "keeps_local_copy": self.config.publish_keep_local_copy,
        }

    def cmd_publish(self) -> None:
        """Upload the finished movie into its Cloudinary order folder.

        Delivery, not generation: nothing is created and — by construction —
        nothing in Cloudinary is replaced or removed. Each publish takes the
        next free version (``final_v1``, ``final_v2``, ...), so a movie that
        was re-combined after a fix is delivered alongside its predecessor
        rather than over it.
        """
        ws = self.workspace
        if not ws.final_video.exists():
            raise PipelineError(
                f"No final video to publish ({ws.final_video}). Build it "
                f"first:\n  {self._next_command('combine')}"
            )
        client = CloudinaryClient.from_config(self.config)
        plan = self.publish_plan(client)
        size_mb = plan["bytes"] / (1024 * 1024)

        # The caller (the panel) may pin the name it showed the user for
        # approval. If the world moved between the two — someone published
        # from elsewhere — the approved name is no longer the free one, so
        # stop rather than upload under a name nobody agreed to.
        approved = (self.options.publish_as or "").strip()
        if approved and approved != plan["public_id"]:
            raise PipelineError(
                f"The approved name ({approved}) is no longer the next free "
                f"one — Cloudinary now says {plan['public_id']}. Nothing was "
                "uploaded; publish again to approve the new name."
            )

        if self.dry_run:
            logger.info(
                "DRY-RUN: would upload %s (%.1f MB) to Cloudinary as %s",
                ws.final_video.name, size_mb, plan["public_id"],
            )
            return

        already = (
            [f"Already published: {', '.join('v' + str(p['version']) for p in plan['published'])}."]
            if plan["published"] else ["Nothing has been published for this order yet."]
        )
        unchanged = (
            [] if plan["changed_since_last"] or not plan["latest"]
            else ["NOTE: the final video hasn't changed since v"
                  f"{plan['latest']['version']} — this uploads the same movie again."]
        )
        if not self._ask(
            [
                f"Uploads {ws.final_video} ({size_mb:.1f} MB) to the order's "
                "Cloudinary folder.",
                f"It will be saved as:  {plan['public_id']}.mp4",
                *already,
                *unchanged,
                "Nothing already in Cloudinary is replaced or deleted.",
                *(
                    [f"A copy of exactly what is delivered is kept in "
                     f"{ws.published_dir.relative_to(ws.root)}/."]
                    if self.config.publish_keep_local_copy else []
                ),
            ],
            f"Publish the movie as {plan['filename']}? [y/N] ",
            "Publish skipped. Deliver it later with:\n  "
            + self._next_command("publish"),
        ):
            return

        logger.info(
            "Publishing %s (%.1f MB) as %s …",
            ws.final_video.name, size_mb, plan["public_id"],
        )
        published = client.publish_final_video(
            plan["order_folder"], ws.final_video, plan["version"]
        )
        record_publication(
            ws.published_file,
            order_folder=plan["order_folder"],
            public_id=published.public_id,
            url=published.url,
            version=published.version,
            video=ws.final_video,
            local_file=self._archive_published_movie(published.version),
        )
        logger.info(
            "Published v%d: %s", published.version, published.url or published.public_id,
        )

    def _archive_published_movie(self, version: int) -> str:
        """Keep the delivered bytes as output/published/<basename>_vN.mp4.

        The next combine REPLACES output/final_video.mp4, so without this copy
        the only remaining record of what a customer was sent is the file in
        their Cloudinary folder. Taken after the upload succeeded, and never
        allowed to sink a delivery that already happened: a failure here (disk
        full is the realistic one) is a warning and an empty record entry, not
        an error. Returns the project-relative path, or "" when there is none.
        """
        ws = self.workspace
        if not self.config.publish_keep_local_copy:
            return ""
        dst = ws.published_dir / (
            f"{self.config.cloudinary_publish_basename}_v{version}.mp4"
        )
        if dst.exists():
            # Version numbers are never reused, so this can only be a leftover
            # from a half-finished publish — keep it rather than overwrite it.
            logger.warning("Local copy %s already exists; leaving it untouched.", dst)
            return str(dst.relative_to(ws.root))
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ws.final_video, dst)
        except OSError as exc:
            logger.warning(
                "Published, but could not keep a local copy at %s: %s. The "
                "delivered movie is safe in Cloudinary; only the local "
                "archive is missing.", dst, exc,
            )
            return ""
        logger.info("Kept a copy of exactly what was delivered: %s", dst)
        return str(dst.relative_to(ws.root))

    # ------------------------------ status step --------------------------- #
    def snapshot(self) -> dict[str, Any]:
        """Structured project status: what exists, what's missing, what's next.

        The one source of truth for "where does this project stand" —
        ``cmd_status`` prints it, the admin API returns it as JSON.
        """
        ws = self.workspace

        def _pngs(directory: Path) -> list[Path]:
            if not directory.exists():
                return []
            return sorted(
                (p for p in directory.iterdir()
                 if p.is_file() and p.suffix.lower() == ".png"),
                key=natural_sort_key,
            )

        # A hand-made or partially-synced project may lack subdirectories;
        # status must describe it, not crash on it.
        inputs = (
            list_input_images(ws.input_images_dir)
            if ws.input_images_dir.exists() else []
        )
        styled = _pngs(ws.styled_images_dir)
        generated = _pngs(ws.generated_frames_dir)

        sb_path = ws.default_storyboard_json
        storyboard: Optional[Storyboard] = None
        storyboard_error = ""
        if sb_path.exists():
            try:
                storyboard = Storyboard.load(sb_path)
            except StoryboardError as exc:
                storyboard_error = str(exc)

        changed_frames: list[str] = []
        missing_frames: list[str] = []
        clips: list[dict[str, Any]] = []
        stray: list[str] = []
        missing = 0
        if storyboard is not None:
            sb_mtime = sb_path.stat().st_mtime
            changed_frames = [
                Path(f.output_path).name
                for f in storyboard.frames
                if (ws.root / f.output_path).exists()
                and (ws.root / f.output_path).stat().st_mtime > sb_mtime
            ]
            frames = [ws.root / f.output_path for f in storyboard.frames]
            # Frames the storyboard plans for that have no styled image on
            # disk. They are BRIDGED OVER at render time — the movie stays
            # continuous, but those photos silently do not appear in it, so
            # this has to be visible rather than a line in a log nobody reads.
            missing_frames = [p.name for p in frames if not p.exists()]
            expected = [
                self._clip_name(a, b)
                for a, b in self._bridge_pairs(frames, quiet=True)
            ]
            for clip in expected:
                exists = clip.exists()
                missing += 0 if exists else 1
                clips.append({
                    "id": clip.stem,
                    "file": clip.name,
                    "rendered": exists,
                    "sfx": exists and self.state.is_done(f"sfx:{clip.name}"),
                    # Rendered before its transition was re-planned; kept
                    # as-is until someone regenerates it deliberately.
                    "stale": exists
                    and self.state.status(f"stale:{clip.name}") is not None,
                })
            found = (
                find_generated_clips(ws.clips_dir)
                if ws.clips_dir.exists() else []
            )
            stray = [p.name for p in sorted(set(found) - set(expected))]

        # Delivery state, read from published.json only — snapshot() runs for
        # every project on every panel refresh, so it must never hit the
        # network (the real Cloudinary listing happens in publish_plan()).
        published = publish_state(ws.published_file, ws.final_video)
        order = read_order_record(ws.order_file)
        published["publishable"] = bool((order or {}).get("order_folder"))

        # Tags resolved to epithets, keyed by styled-frame path — the same
        # rosters planning feeds the unstageable-pair gate.
        _snapshot_people = tagged_people(storyboard) if storyboard else {}

        if storyboard is None and not storyboard_error:
            next_step = "storyboard"
        elif storyboard is not None and missing:
            next_step = "render"
        elif storyboard is not None and not ws.final_video.exists():
            next_step = "combine"
        elif published["publishable"] and (
            not published["count"] or published["changed_since"]
        ):
            # A finished movie that has never been delivered — or one rebuilt
            # since the last delivery — still owes the customer a version.
            next_step = "publish"
        else:
            next_step = ""

        return {
            "project": ws.root.name,
            "input_images": [p.name for p in inputs],
            "styled_images": [p.name for p in styled],
            "generated_frames": [p.name for p in generated],
            "storyboard": None if storyboard is None else {
                "frames": len(storyboard.frames),
                "transitions": len(storyboard.transitions),
                "from_idea": any(f.image_prompt.strip() for f in storyboard.frames),
                # Transitions still carrying the config fallback prompt: the
                # planner never succeeded for them (quota/rate-limit failure).
                # Re-running `storyboard` re-plans exactly these.
                "placeholder_transitions": [
                    t.id for t in storyboard.transitions
                    if t.motion_prompt == self.config.motion_prompt
                ],
                # Cast members named by what they were wearing in one photo.
                # New plans can't produce these any more (the planner is
                # forbidden and code re-anchors what slips through), but a
                # cast written before that is frozen on purpose — its wording
                # is baked into prompts already planned — so the panel flags
                # them for a hand edit instead.
                "fragile_epithets": [
                    c.id for c in storyboard.characters
                    if is_clothing_anchored(c.epithet)
                ],
                # Cast members whose epithets cannot be told apart — one a
                # word-subset of the other ("woman with dark hair bun" /
                # "young woman with dark hair bun"). The video model has no
                # anchor to keep them apart, and the pipeline's own swap and
                # mover matching reads them as one person and gives up.
                # Groups of ids, flagged for a hand edit like fragile ones:
                # existing casts are frozen, their wording is already baked
                # into planned prompts.
                "indistinct_epithets": indistinct_epithets(
                    storyboard.characters
                ),
                # Prompts written before the identity facts they should have
                # used: a photo tagged (or re-tagged) since, or a cast member
                # renamed. Both are plan-time inputs that change nothing on
                # their own, so without this the only way to apply them was
                # to re-plan everything and hope.
                "outdated_plans": outdated_identity_plans(storyboard),
                # Prompts whose last beat leaves someone out of the frame.
                # The clip is pinned to the end frame, which shows those
                # people, so it cannot end there — the video model resolves
                # the contradiction with a cut or a teleport. Caught wherever
                # the wording came from: the planner, a restage, or a hand
                # edit.
                "ends_offscreen": [
                    t.id for t in storyboard.transitions
                    if ends_offscreen(t.motion_prompt)
                ],
                # Saved prompts still choreographing people on a pair whose
                # TAGS say the choreography cannot exist — too many movers,
                # or a crowd whose roster changes (is_unstageable_pair, the
                # same gate new plans go through). Rendered, these mush or
                # dissolve; the fix is re-planning the pair, which now
                # yields a deterministic camera transition. A pair already
                # carrying a camera-family prompt is not listed, and
                # untagged frames have no opinion — so pre-existing
                # untagged projects stay silent.
                "unstageable_pairs": [
                    t.id for t in storyboard.transitions
                    if is_unstageable_pair(
                        _snapshot_people.get(t.start_frame),
                        _snapshot_people.get(t.end_frame),
                    )
                    and not is_camera_transition(t.motion_prompt)
                ],
            },
            "storyboard_error": storyboard_error,
            "changed_frames": changed_frames,
            # Storyboard frames with no styled image: bridged over at
            # render time, so these photos are absent from the movie.
            "missing_frames": missing_frames,
            "clips": clips,
            "stray_clips": stray,
            "pending_renders": self.pending_renders(),
            "final_video": ws.final_video.exists(),
            # The movie was built before one of its clips was (re-)rendered,
            # so it is showing an older version of that clip. mtimes are
            # trustworthy here in a way they are not for styled frames: both
            # sides are written locally by this pipeline, and the worst case
            # is a suggestion to re-run a free, local step.
            "final_video_outdated": self._final_video_outdated(),
            # Delivery: which movie versions were published back to the
            # order's Cloudinary folder, and whether the current final video
            # is newer than the last one delivered.
            "published": published,
            # The closing letter is a plain text file someone writes by hand;
            # with the file missing or blank the combine toggle silently
            # produces no letter (see _letter_text), so status and the panel
            # have to be able to say whether there is one at all.
            "letter": letter_state(ws.letter_file),
            # Estimated money spent on this project (see cost_report). Built
            # from what snapshot already knows, so it costs no extra I/O.
            "cost": self.cost_report(
                storyboard=storyboard, clips=clips,
                images=len(styled) + len(generated),
            ),
            # Human judgements of this project's clips, and how many lessons
            # they taught the planner (see feedback.py).
            "feedback": self._feedback_summary(),
            "music": ws.music_file.exists(),
            "custom_music": ws.custom_music_file.exists(),
            "has_failed_jobs": self.failed.path.exists(),
            # The failures themselves, not just "a file exists": the panel had
            # no way to show WHAT went wrong, so a run whose clips all failed
            # was indistinguishable from a clean one without opening the log.
            "failed_jobs": self._recorded_failures(),
            "next_step": next_step,
        }

    # --------------------- who is in each frame (tagging) ------------------ #
    def _propose_tags(self, storyboard: Storyboard) -> int:
        """Fill in who is in each UNTAGGED frame; returns how many were tagged.

        The shared core of `tag` and the first pass that `storyboard` runs at
        the end of planning. Never touches a frame that already carries tags
        (unless ``--retag``): a proposal must never overwrite the human
        corrections that are the whole point of tagging.

        Best-effort by construction — this is a convenience on top of a
        finished plan, so a failure here is a warning, never something that
        loses the storyboard that was just written.
        """
        if not self.options.tag_frames or self.dry_run:
            return 0
        if not storyboard.characters:
            return 0
        root = self.workspace.root
        frames = [f for f in storyboard.frames if (root / f.output_path).exists()]
        targets = (
            frames if self.options.retag else [f for f in frames if not f.people]
        )
        if not targets:
            return 0
        logger.info(
            "Proposing who is in %d untagged frame(s) — a draft to correct.",
            len(targets),
        )
        try:
            proposed = self.openai.identify_people(
                [root / f.output_path for f in targets], storyboard.characters
            )
        except Exception as exc:  # noqa: BLE001 - never lose the plan over this
            logger.warning(
                "Could not propose identities (%s); tag the frames yourself in "
                "the panel, or run `%s`.",
                exc, self._next_command("tag"),
            )
            return 0
        by_path = {f.output_path: people for f, people in zip(targets, proposed)}
        tagged = 0
        for frame in storyboard.frames:
            people = by_path.get(frame.output_path)
            # An empty answer means "I couldn't see anyone" — a question for
            # the human, not a recorded fact that nobody is in the photo.
            if not people:
                continue
            frame.people = [FramePerson(**p) for p in people]
            tagged += 1
        if tagged:
            logger.info(
                "Proposed identities for %d frame(s). CHECK THEM before "
                "re-planning — especially where two people look alike.", tagged,
            )
        return tagged

    def cmd_tag(self) -> None:
        """Propose who is in each frame, for a human to correct.

        Identity is the judgement the planner is worst at, so the fix is to
        let a person state it — but stating it for thirty photos by hand is
        the kind of chore that never gets done. This does the first pass:
        one vision call over the styled frames returns which cast member
        stands where in each, and the panel's tagger is where you fix what it
        got wrong.

        It is a DRAFT, and it behaves like one:

        * frames that already carry tags are left alone (your corrections are
          the whole point; ``--retag`` overrides that deliberately);
        * a frame the model reports nobody in stays untagged rather than
          being recorded as "nobody is here", so a missed frame reads as a
          question, not as an answer;
        * it never touches transitions, so nothing is re-planned or marked
          stale by tagging — the tags feed the NEXT plan (re-plan a clip to
          apply them).
        """
        storyboard = self._require_storyboard("tag")
        if not storyboard.characters:
            raise PipelineError(
                "This project has no cast yet, so there is nobody to tag. Run "
                f"`{self._next_command('storyboard')}` first — the planner "
                "builds the cast — then tag the frames."
            )
        root = self.workspace.root
        frames = [f for f in storyboard.frames if (root / f.output_path).exists()]
        missing = len(storyboard.frames) - len(frames)
        if not frames:
            raise PipelineError(
                "None of the storyboard's styled frames exist on disk yet."
            )
        targets = frames if self.options.retag else [f for f in frames if not f.people]
        if not targets:
            print(
                f"\nAll {len(frames)} frame(s) are already tagged. Use "
                f"`{self._next_command('tag', '--retag')}` to redo them from "
                "scratch (your corrections would be replaced)."
            )
            return
        if self.dry_run:
            logger.info(
                "[dry-run] would identify %d cast member(s) across %d frame(s)",
                len(storyboard.characters), len(targets),
            )
            return
        if not self._ask(
            [
                f"Look at {len(targets)} frame(s) and propose which of the "
                f"{len(storyboard.characters)} cast members is in each, and "
                "where.",
                "One OpenAI vision call. Nothing is re-planned or re-rendered "
                "— the tags feed future planning, and you can correct them "
                "afterwards.",
                *( [f"{len(frames) - len(targets)} already-tagged frame(s) are "
                    "left untouched."] if len(targets) < len(frames) else []),
            ],
            f"Identify people in {len(targets)} frame(s)? [y/N] ",
            "Nothing tagged.",
        ):
            return

        tagged = self._propose_tags(storyboard)
        self._save_storyboard(storyboard)
        epithets = {c.id: c.epithet for c in storyboard.characters}
        print(f"\nTagged {tagged} of {len(targets)} frame(s):")
        for frame in storyboard.frames:
            if not frame.people:
                continue
            who = ", ".join(
                epithets.get(p.id, p.id)
                for p in sorted(frame.people, key=lambda p: p.x)
            )
            print(f"  {Path(frame.output_path).name:<24} {who}")
        if tagged < len(targets):
            print(
                f"  ({len(targets) - tagged} frame(s) came back empty and were "
                "left untagged.)"
            )
        if missing:
            print(f"  ({missing} frame(s) have no styled image yet.)")
        print(
            "\nThis is a DRAFT — check it in the panel's \"Who's in each "
            "photo\" tagger, especially where two people look alike.\n"
            "Tags apply to plans made from now on: re-plan a clip to use them."
        )

    # ------------------------------ feedback ------------------------------ #
    def _feedback_summary(self) -> dict[str, Any]:
        """Per-clip verdicts + counts for status/snapshot (never raises)."""
        try:
            return self.feedback_store.summary()
        except Exception as exc:  # noqa: BLE001 - a note can't break status
            logger.warning("Could not read the feedback file: %s", exc)
            return {"count": 0, "good": 0, "bad": 0, "lessons": 0,
                    "by_transition": {}}

    def record_feedback(self) -> dict[str, Any]:
        """Record one judgement of a rendered clip and learn a rule from it.

        This is the loop the whole feature exists for: the planner writes a
        motion prompt blind (it never sees the clip it caused), so the only
        way it improves is someone watching the result and saying what went
        wrong. There are TWO witnesses to that, and the difference matters:

        * the human, who knows which faults are worth caring about but is
          usually brief ("it looks weird");
        * the reviewer (``feedback_review``), which is handed stills sampled
          across the rendered clip plus the two key frames and the prompt,
          and can name precisely what happened — and propose a corrected
          prompt and length for THIS clip.

        Both accounts go into the distillation, so the rule that comes out
        describes a mechanism rather than a mood. The note is saved verbatim
        with the exact prompt that produced the clip — evidence that must
        survive later storyboard edits, and the only lasting record of what
        the clip contained once it is regenerated over.

        Nothing here changes the movie. The suggested prompt is RETURNED, not
        applied: adopting it is a storyboard edit the user makes (which marks
        the clip outdated), and re-rendering stays the separate, confirmed,
        paid action it has always been.

        Failure modes handled deliberately:

        * both model calls cost credits, so ``feedback_review`` and
          ``feedback_learn`` can each be turned off — the note is still
          recorded either way;
        * if either call fails (quota, network), the note is STILL saved and
          the result says which part didn't happen. Losing what someone wrote
          because a model call timed out is not an acceptable outcome.

        Returns the saved entry, the review, the lessons minted, and any
        errors, for the CLI to print and the API to return.
        """
        options = self.options
        note = (options.feedback_note or "").strip()[:MAX_NOTE_CHARS]
        tid_given = bool((options.feedback_clip or "").strip())
        if not note and not (options.feedback_review and tid_given):
            # A note is the point — unless the user is asking the reviewer to
            # watch a specific clip and report, which is an account of its own.
            raise PipelineError(
                "Feedback needs a note — what was wrong (or right) with the "
                "clip? (Or name a clip and let the reviewer watch it.)"
            )
        verdict = (options.feedback_verdict or "bad").strip().lower()
        if verdict not in VERDICTS:
            raise PipelineError(
                f"Unknown verdict {verdict!r}; use one of: {', '.join(VERDICTS)}"
            )

        tid = (options.feedback_clip or "").strip().removesuffix(".mp4")
        motion_prompt, duration = "", 0
        start_frame = end_frame = None
        if tid:
            # Feedback about a clip has to name a real one: a typo'd id would
            # otherwise teach a lesson attributed to nothing.
            storyboard = self._require_storyboard("feedback")
            transition = next(
                (t for t in storyboard.transitions if t.id == tid), None
            )
            if transition is None:
                raise PipelineError(
                    f"No transition '{tid}' in this project's storyboard.\n"
                    "Valid ids: "
                    + ", ".join(t.id for t in storyboard.transitions)
                )
            # Copied now, on purpose: the storyboard is hand-edited between
            # steps, so "the prompt this clip came from" is only knowable here.
            motion_prompt = _with_global_motion(
                storyboard.global_motion_prompt, transition.motion_prompt
            )
            duration = transition.duration
            start_frame = self.workspace.root / transition.start_frame
            end_frame = self.workspace.root / transition.end_frame

        entry = FeedbackEntry(
            project=self.workspace.root.name,
            transition_id=tid,
            verdict=verdict,
            note=note,
            motion_prompt=motion_prompt,
            duration=duration,
        )

        review: dict[str, Any] = {}
        review_error = ""
        if self.dry_run:
            logger.info("[dry-run] would watch the clip and distil lessons")
        elif options.feedback_review and tid:
            review, review_error = self._review_clip(
                tid, motion_prompt, duration, note, start_frame, end_frame,
                global_context=storyboard.global_motion_prompt,
            )
            entry.ai_observation = review.get("observed", "")
            entry.ai_problems = review.get("problems", [])
            entry.suggested_motion_prompt = review.get(
                "suggested_motion_prompt", ""
            )
            entry.suggested_duration = review.get("suggested_duration", 0)

        learned: list[dict[str, Any]] = []
        learn_error = ""
        if self.dry_run:
            pass  # already reported above
        elif options.feedback_learn:
            try:
                existing = [l.text for l in self.lessons.all() if l.active]
                proposals = self.openai.distill_lesson(
                    note, verdict=verdict, motion_prompt=motion_prompt,
                    duration=duration, existing=existing,
                    observation=review.get("observed", ""),
                    problems=review.get("problems", []),
                )
                for proposal in proposals:
                    lesson = self.lessons.add(
                        proposal["text"], proposal["scope"],
                        origin="feedback",
                        source={
                            "project": self.workspace.root.name,
                            "transition": tid,
                            "verdict": verdict,
                            "feedback_id": entry.id,
                        },
                    )
                    learned.append(lesson.model_dump())
                    logger.info("Learned (%s): %s", lesson.scope, lesson.text)
                if not proposals:
                    logger.info(
                        "Nothing general enough to learn from this note — it is "
                        "recorded, but no rule was added."
                    )
            except Exception as exc:  # noqa: BLE001 - never lose the note
                learn_error = str(exc)
                logger.warning(
                    "Could not distil a lesson (%s); the feedback itself is "
                    "saved and can be turned into a lesson by hand.", exc,
                )

        entry.lesson_ids = [l["id"] for l in learned]
        if not self.dry_run:
            self.feedback_store.add(entry)
        return {
            "feedback": entry.model_dump(),
            "review": review,
            "review_error": review_error,
            "lessons": learned,
            "learn_error": learn_error,
        }

    def _review_clip(
        self,
        tid: str,
        motion_prompt: str,
        duration: int,
        note: str,
        start_frame: Optional[Path],
        end_frame: Optional[Path],
        global_context: str = "",
    ) -> tuple[dict[str, Any], str]:
        """Let the reviewer watch the rendered clip; never raise.

        Returns ``(review, error)`` — an empty review with a spoken reason is
        a normal outcome (the clip was never rendered, ffmpeg is missing, the
        call failed), and the feedback around it must still be saved. The
        sampled stills live in a temporary directory: they exist only for the
        duration of the call, and keeping them would quietly grow the project
        by a few hundred KB per note.
        """
        clip = self.workspace.clips_dir / f"{tid}.mp4"
        if not clip.exists():
            return {}, (
                f"{clip.name} has not been rendered yet, so there was nothing "
                "to watch."
            )
        try:
            with tempfile.TemporaryDirectory(prefix="clip-review-") as tmp:
                frames = sample_clip_frames(
                    clip, Path(tmp), count=self.config.clip_review_frames
                )
                if not frames:
                    return {}, (
                        f"No frames could be read from {clip.name}."
                    )
                logger.info(
                    "Watching %s (%d sampled frame(s))...", clip.name, len(frames)
                )
                review = self.openai.review_clip(
                    frames, motion_prompt, duration or self.config.duration,
                    start_frame=start_frame, end_frame=end_frame,
                    user_note=note, lessons=self._lesson_texts(SCOPE_MOTION),
                    # So its rewrite replaces the per-clip prompt only — the
                    # global part is prepended again at render time.
                    global_context=global_context,
                )
        except Exception as exc:  # noqa: BLE001 - a review never sinks a note
            logger.warning(
                "Could not review %s (%s); the feedback itself is saved.",
                clip.name, exc,
            )
            return {}, str(exc)
        if review.get("problems"):
            logger.info("Reviewer found: %s", "; ".join(review["problems"]))
        else:
            logger.info("Reviewer found no fault with the clip.")
        return review, ""

    def cmd_feedback(self) -> None:
        """CLI face of :meth:`record_feedback` — say it, then show what stuck."""
        result = self.record_feedback()
        entry = result["feedback"]
        target = entry["transition_id"] or "the movie in general"
        print(f"\nFeedback recorded for {target} ({entry['verdict']}).")

        review = result["review"]
        if result["review_error"]:
            print(f"  The clip could not be watched: {result['review_error']}")
        elif review:
            print(f"\n  Watching {target}, the reviewer saw:")
            print(f"    {review['observed']}")
            for problem in review["problems"]:
                print(f"    - {problem}")
            if not review["problems"]:
                print("    - no faults found")
            if review["changes_clip"]:
                # Printed, never applied: adopting it is an edit the user
                # makes, and re-rendering is a separate paid action.
                print(
                    f"\n  It suggests rendering this pair at "
                    f"{review['suggested_duration']}s with:\n"
                    f"    {review['suggested_motion_prompt']}"
                )
                if review["why"]:
                    print(f"    ({review['why']})")
                print(
                    "  To use it: put that prompt (and duration) into the "
                    "transition in\n"
                    f"    {self.workspace.default_storyboard_json}\n"
                    "  then render just this clip:\n    "
                    + self._next_command(
                        'render', '--clip', entry['transition_id'] or 'ID')
                )

        if result["learn_error"]:
            print(
                "  The lesson could not be distilled "
                f"({result['learn_error']}) — the note itself is saved."
            )
        elif result["lessons"]:
            print(f"  {len(result['lessons'])} lesson(s) now apply to every "
                  "future plan:")
            for lesson in result["lessons"]:
                print(f"    [{lesson['scope']}] {lesson['text']}")
        elif self.options.feedback_learn:
            print("  Nothing general enough to turn into a rule — note kept.")
        print(f"  Notes: {self.workspace.feedback_file}")
        print(f"  Lessons: {lessons_file()}")

    def cost_report(
        self,
        storyboard: Optional[Storyboard] = None,
        clips: Optional[list[dict[str, Any]]] = None,
        images: Optional[int] = None,
    ) -> dict[str, Any]:
        """What this project has cost: the ledger, plus a from-disk estimate.

        Two numbers, because they answer different questions:

        * ``total_usd`` — the sum of what was actually recorded while the
          calls happened. Exact about WHAT was bought, priced with
          ``config.pricing``.
        * ``estimated_usd`` — what the artifacts sitting on disk (styled
          frames, rendered clips, SFX passes) would cost to buy today. Every
          project that predates the ledger has a full movie and an empty
          ledger, and reporting those as "$0 spent" would be worse than an
          approximation.

        The panel shows the recorded figure and falls back to the estimate
        when it is clearly the more truthful of the two (``estimated``).
        The optional arguments let ``snapshot`` hand over what it has already
        computed instead of re-reading the project.
        """
        ws = self.workspace
        recorded = self.costs.summary()

        if images is None:
            images = sum(
                1 for directory in (ws.styled_images_dir, ws.generated_frames_dir)
                if directory.exists()
                for p in directory.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            )
        if storyboard is None:
            path = ws.default_storyboard_json
            if path.exists():
                try:
                    storyboard = Storyboard.load(path)
                except StoryboardError:
                    storyboard = None
        seconds_by_clip = {
            Path(t.output_path).stem: t.duration
            for t in (storyboard.transitions if storyboard else [])
        }
        if clips is None:
            found = find_generated_clips(ws.clips_dir) if ws.clips_dir.exists() else []
            clips = [
                {"id": p.stem, "file": p.name, "rendered": True,
                 "sfx": self.state.is_done(f"sfx:{p.name}")}
                for p in found
            ]
        rendered = [c for c in clips if c.get("rendered")]
        clip_seconds = sum(
            seconds_by_clip.get(str(c.get("id", "")), self.config.duration)
            for c in rendered
        )
        estimated = estimate_usd(
            self.config.pricing,
            images=images,
            clip_seconds=clip_seconds,
            sfx_clips=sum(1 for c in rendered if c.get("sfx")),
        )
        return {
            **recorded,
            "estimated_usd": estimated,
            # True when the ledger is missing most of the story (a project
            # made before tracking existed, or one whose ledger was lost):
            # the panel then leads with the estimate and says so.
            "estimated": estimated > recorded["total_usd"] * 1.05,
            "currency": "USD",
            "clips_rendered": len(rendered),
            "clip_seconds": clip_seconds,
            "images": images,
        }

    def _final_video_outdated(self) -> bool:
        """Is output/final_video.mp4 older than a clip it was built from?

        The last "not up to date" gap in the chain: regenerating one clip
        leaves the finished movie silently showing the previous version of
        it, and the fix (another combine) is free and local — so this is a
        hint worth printing, not a gate.
        """
        final = self.workspace.final_video
        if not final.exists() or not self.workspace.clips_dir.exists():
            return False
        try:
            built_at = final.stat().st_mtime
            return any(
                clip.stat().st_mtime > built_at
                for clip in find_generated_clips(self.workspace.clips_dir)
            )
        except OSError:  # a clip vanished mid-check; not worth failing status
            return False

    def _recorded_failures(self) -> list[dict[str, Any]]:
        """The last run's failures, from failed_jobs/failed_jobs.json.

        The file is removed by a clean run, so whatever is here describes the
        most recent run that had problems. Best-effort: a malformed file
        reports nothing rather than breaking status.
        """
        path = self.failed.path
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if not isinstance(data, list):
            return []
        return [
            {
                "id": str(f.get("job_id", "")),
                "kind": str(f.get("kind", "")),
                "error": str(f.get("error", ""))[:400],
                "at": str(f.get("timestamp", "")),
            }
            for f in data if isinstance(f, dict)
        ]

    def pending_renders(self) -> list[dict[str, Any]]:
        """Clip renders that were submitted to fal but never collected.

        The request_id is persisted before the wait and cleared only once the
        mp4 is downloaded, so a surviving ``falreq:`` entry means the job was
        submitted — the money is spent — and its output is still sitting on
        fal's queue. Nothing polls for these in the background: the result is
        fetched on the next ``render`` of that clip, and only while its
        fingerprint still matches the storyboard. Surfacing them is what
        makes an abandoned paid render visible instead of silent.
        """
        pending = []
        for job_id, entry in sorted(self.state.find("falreq:").items()):
            clip = job_id.removeprefix("falreq:")
            pending.append({
                "clip": clip,
                "id": clip.removesuffix(".mp4"),
                "request_id": entry.get("request_id", ""),
                "submitted_at": entry.get("updated_at", ""),
                # Set by the admin server when a running job owns this clip;
                # from here (a one-shot CLI process) there is no way to know.
                "in_flight": False,
                # False once the storyboard moved on: the pending job renders
                # the old plan, so the next render discards it and re-buys.
                "recoverable": self._pending_render_matches(clip, entry),
            })
        return pending

    def _pending_render_matches(self, clip: str, entry: dict[str, Any]) -> bool:
        """Would the next render still be able to reuse this paid job?

        Mirrors VideoClient.fingerprint: the frames, prompt, duration and
        model must all be unchanged since submission. Best-effort — a project
        whose storyboard no longer describes this clip simply reports False.
        """
        fingerprint = entry.get("fingerprint")
        if not fingerprint:
            return False
        path = self.workspace.default_storyboard_json
        if not path.exists():
            return False
        try:
            storyboard = Storyboard.load(path)
        except StoryboardError:
            return False
        stem = clip.removesuffix(".mp4")
        for start, end, motion, duration, _sound in self._pairs_from_storyboard(
            storyboard
        ):
            if self._clip_name(start, end).stem != stem:
                continue
            if not (start.exists() and end.exists()):
                return False
            return self.video_client.fingerprint(
                start, end, motion, duration
            ) == fingerprint
        return False

    def cmd_status(self) -> None:
        """Print where this project stands and what to run next."""
        snap = self.snapshot()
        ws = self.workspace
        line = "=" * 60
        print(f"\n{line}\nPROJECT STATUS: {ws.root.name}\n{line}")

        print(f"  Input images     : {len(snap['input_images'])}")
        print(f"  Styled images    : {len(snap['styled_images'])}")
        if snap["generated_frames"]:
            print(f"  Generated frames : {len(snap['generated_frames'])}")

        if snap["storyboard_error"]:
            print(f"  Storyboard       : UNREADABLE ({snap['storyboard_error']})")
        elif snap["storyboard"] is None:
            print("  Storyboard       : none")
        else:
            sb = snap["storyboard"]
            mode = "from idea" if sb["from_idea"] else "from images"
            print(
                f"  Storyboard       : {sb['frames']} frame(s), "
                f"{sb['transitions']} transition(s) ({mode})"
            )
            if sb["placeholder_transitions"]:
                print(
                    f"  !! {len(sb['placeholder_transitions'])} transition(s) "
                    "still have the generic fallback prompt (planning failed "
                    "- quota/rate limit?). Re-run storyboard to re-plan them."
                )

        if snap["missing_frames"]:
            print(
                f"  !! {len(snap['missing_frames'])} frame(s) have no styled "
                "image: " + ", ".join(snap["missing_frames"][:8])
                + "  (bridged over - these photos are NOT in the movie; "
                "re-style them or remove the photos)"
            )

        if snap["changed_frames"]:
            print(
                "  Changed frames   : " + ", ".join(snap["changed_frames"])
                + "  (newer than the storyboard - run storyboard to re-plan)"
            )

        # Prompts that don't yet know what you told the app about identity.
        outdated_plans = (snap["storyboard"] or {}).get("outdated_plans") or []
        if outdated_plans:
            print(
                f"  !! {len(outdated_plans)} prompt(s) were planned before the "
                "current photo tags / cast names: "
                f"{', '.join(outdated_plans[:6])}"
                f"{' …' if len(outdated_plans) > 6 else ''}\n"
                "     Apply them with:\n     "
                + self._next_command("storyboard", "--replan-all")
            )

        # Saved prompts still choreographing people on pairs whose tags say
        # the choreography cannot exist (too many movers / too crowded).
        unstageable = (snap["storyboard"] or {}).get("unstageable_pairs") or []
        if unstageable:
            print(
                f"  !! {len(unstageable)} prompt(s) stage people on pairs too "
                "crowded to choreograph (renders mush or dissolve): "
                f"{', '.join(unstageable[:6])}"
                f"{' …' if len(unstageable) > 6 else ''}\n"
                "     Re-plan them to get camera transitions instead."
            )

        for clip in snap["clips"]:
            if clip["rendered"]:
                sfx = "sfx ✓" if clip["sfx"] else "silent"
                outdated = (
                    "  !! OUTDATED (storyboard changed - redo with --clip)"
                    if clip.get("stale") else ""
                )
                print(f"    clip {clip['id']:<12} rendered  ({sfx}){outdated}")
            else:
                print(f"    clip {clip['id']:<12} MISSING")
        if snap["stray_clips"]:
            print(f"  Stray clips      : {', '.join(snap['stray_clips'])}")

        # Paid-for renders whose output was never collected. Nothing fetches
        # these in the background — only the next render of that clip does.
        for pending in snap["pending_renders"]:
            if pending["recoverable"]:
                print(
                    f"  !! clip {pending['id']} has a PAID render waiting on "
                    f"the provider (submitted "
                    f"{_local_time(pending['submitted_at'])}). "
                    f"Collect it with:\n     "
                    f"{self._next_command('render', '--clip', pending['id'])}"
                )
            else:
                print(
                    f"  !! clip {pending['id']} has a paid render on the "
                    "provider that NO LONGER matches the storyboard (its "
                    "frames/prompt/duration changed since it was submitted). "
                    "Rendering this clip will pay for a fresh one; the old "
                    "job's output is lost."
                )

        letter = snap["letter"]
        if letter["chars"]:
            print(f"  Closing letter   : {letter['chars']} chars in {ws.letter_file}")
        elif letter["exists"]:
            print(f"  Closing letter   : {ws.letter_file} is empty — nothing to scroll")

        final = ws.final_video
        print(f"  Final video      : {'ready — ' + str(final) if final.exists() else 'not built'}")
        if snap["final_video_outdated"]:
            print(
                "  !! a clip has been re-rendered since the movie was built, "
                "so it still shows the old one.\n     Rebuild it (free, "
                "local):\n     " + self._next_command("combine", "--force")
            )

        # Money. Always described as an estimate: it is priced from
        # config.pricing, never read from a provider's invoice.
        cost = snap["cost"]
        if cost["estimated"]:
            print(
                f"  Spent (est.)     : ${cost['estimated_usd']:.2f} from the "
                f"files on disk (${cost['total_usd']:.2f} recorded — this "
                "project predates cost tracking)"
            )
        elif cost["total_usd"] or cost["entries"]:
            parts = ", ".join(
                f"{kind} ${bucket['usd']:.2f}"
                for kind, bucket in cost["by_kind"].items() if bucket["usd"]
            )
            print(
                f"  Spent (est.)     : ${cost['total_usd']:.2f}"
                + (f"  ({parts})" if parts else "")
            )

        fb = snap["feedback"]
        if fb["count"]:
            print(
                f"  Clip feedback    : {fb['count']} note(s) "
                f"({fb['good']} good / {fb['bad']} bad), "
                f"{fb['lessons']} lesson(s) taught"
            )

        published = snap["published"]
        if published["latest"]:
            note = " (the final video has changed since)" if published["changed_since"] else ""
            print(
                f"  Published        : v{published['latest']['version']} — "
                f"{published['latest']['public_id']}{note}"
            )
            kept = [v for v in published["versions"] if v["local_exists"]]
            if kept:
                print(
                    f"  Delivered copies : {len(kept)} kept in {ws.published_dir} "
                    f"({', '.join(Path(v['local_file']).name for v in kept)})"
                )
        elif published["publishable"] and final.exists():
            print("  Published        : not yet delivered to the order's Cloudinary folder")
        if snap["has_failed_jobs"]:
            print(f"  Failed jobs      : see {self.failed.path}")

        if snap["next_step"]:
            print(f"\n  Next step:\n    {self._next_command(snap['next_step'])}")
        print(line)

    # ------------------------------- one-shot ----------------------------- #
    def cmd_run(self) -> None:
        """The whole flow in one command, gated by confirmation prompts.

        Reuses/reconciles the saved storyboard; otherwise creates one (from
        images, or from --idea when given), then renders and combines.
        Splitting the flow across `storyboard`/`render`/`combine` gives the
        same result with an editable pause between each step.
        """
        if self.options.idea or self.options.idea_file:
            storyboard = self._create_storyboard_from_idea()
            if storyboard is None:  # dry-run: no plan to continue from
                return
        else:
            storyboard = self._prepare_mode_a_storyboard()
            if storyboard is None:
                return

        self.summary.input_count = self.summary.input_count or len(storyboard.frames)
        self._generate_frames(storyboard)
        pairs = self._pairs_from_storyboard(storyboard)
        self._render_pairs(pairs, set())
        if not self.options.no_combine:
            self._combine_clips(confirm=True)
