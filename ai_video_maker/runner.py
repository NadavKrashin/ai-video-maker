"""Pipeline orchestration.

The pipeline is constructed from three explicit inputs — a validated ``Config``,
a ``Workspace`` (all per-movie paths), and ``RunOptions`` (this run's choices).
Nothing here reads global state, argparse, or the terminal, so the same
orchestration can be driven by the CLI or, later, an API request.

The public surface is one method per lifecycle command (``cmd_storyboard``,
``cmd_render``, ``cmd_audio``, ``cmd_combine``, ``cmd_status``, ``cmd_run``),
dispatched via :meth:`Pipeline.execute`. Anything interactive happens through
the injected ``confirm`` callback — the CLI wires it to a terminal prompt, an
API caller simply omits it (every gate auto-proceeds) and drives the steps
individually instead.
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Optional

from tqdm import tqdm

from .clients.audio import AudioClient
from .clients.openai_client import OpenAIClient
from .clients.video import VideoClient
from .config import Config
from .errors import PipelineError, StoryboardError
from .logging_setup import logger
from .media.ffmpeg import (
    apply_edge_fades,
    combine_clips,
    ffprobe_duration,
    find_generated_clips,
    mux_music,
)
from .media.images import (
    SUPPORTED_IMAGE_EXTS,
    list_input_images,
    natural_sort_key,
)
from .media.images import verify_dimensions
from .models import Frame, Storyboard, Transition
from .options import RunOptions
from .state import FailedJobStore, StateStore
from .storyboard_html import write_storyboard_preview
from .storyboard_md import write_storyboard_markdown
from .summary import RunSummary
from .workspace import PROJECT_ROOT, Workspace

# (info_lines, question) -> proceed? Injected by the CLI as a terminal prompt;
# defaults to always-yes so embedded/API callers never block on stdin.
ConfirmFn = Callable[[list[str], str], bool]

# One planned clip: (start_frame, end_frame, motion_prompt, duration, sound_prompt)
ClipPair = tuple[Path, Path, str, int, str]


class Pipeline:
    def __init__(
        self,
        config: Config,
        workspace: Workspace,
        options: RunOptions,
        confirm: Optional[ConfirmFn] = None,
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.options = options
        self.confirm: ConfirmFn = confirm or (lambda lines, question: True)
        self.dry_run: bool = options.dry_run
        self.force: bool = options.force
        self.duration: int = options.duration or config.duration
        self.state = StateStore(workspace.state_file)
        self.failed = FailedJobStore(workspace.failed_jobs_file)
        self.summary = RunSummary()
        self.openai = OpenAIClient(config)
        self.video_client = VideoClient(config)
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
        self._storyboard_music_prompt: str = ""
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
            "storyboard": self.cmd_storyboard,
            "render": self.cmd_render,
            "audio": self.cmd_audio,
            "combine": self.cmd_combine,
            "status": self.cmd_status,
            "run": self.cmd_run,
        }
        handler = handlers.get(command)
        if handler is None:
            raise PipelineError(f"Unknown command: {command}")
        if command == "status":
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
        workers = 1 if self.dry_run else self.concurrency
        if workers <= 1:
            for item in tqdm(items, desc=desc, unit=unit):
                worker(item)
            return

        logger.info("%s: %d job(s), %d in parallel", desc, len(items), workers)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(worker, item) for item in items]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=desc, unit=unit
            ):
                try:
                    fut.result()
                except Exception as exc:  # noqa: BLE001 - workers self-report
                    logger.error("Unexpected worker error: %s", exc)

    def _ask(self, lines: list[str], question: str, decline_log: str) -> bool:
        """Gate on the injected confirm callback; True means proceed.

        Dry-runs always proceed (nothing is spent). A decline logs
        `decline_log` — which should name the command that resumes from here —
        and returns False.
        """
        if self.dry_run:
            return True
        if self.confirm(lines, question):
            return True
        logger.info(decline_log)
        return False

    def _next_command(self, command: str, *flags: str) -> str:
        """A copy-pasteable command for this project's next step."""
        return " ".join(
            ["python pipeline.py", command, self.workspace.root.name, *flags]
        )

    # --------------------------- storyboard step -------------------------- #
    def cmd_storyboard(self) -> None:
        """Create (or refresh) the storyboard, then stop for review.

        With --idea/--idea-file the storyboard is written by the text model
        from scratch (frames get image prompts and are generated at render
        time). Otherwise the images in input_images/ are styled and the vision
        model plans one transition per consecutive pair.
        """
        if self.options.idea or self.options.idea_file:
            storyboard = self._create_storyboard_from_idea()
            if storyboard is not None:
                self._announce_storyboard_ready()
            return

        images = list_input_images(self.workspace.input_images_dir)
        self.summary.input_count = len(images)
        if not images:
            raise PipelineError(
                f"No supported images found in {self.workspace.input_images_dir}. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}. Add images, or pass "
                "--idea to generate a storyboard from scratch."
            )
        logger.info("Found %d input image(s).", len(images))

        styled = self._style_images(images)
        if self.dry_run:
            logger.info(
                "[dry-run] would analyse %d styled frame(s) and write %s + %s",
                len(styled),
                self.workspace.default_storyboard_json,
                self.workspace.storyboard_md,
            )
            return
        if len(styled) < 2:
            logger.warning(
                "Need at least 2 styled images to make a clip; have %d.",
                len(styled),
            )
            return

        existing = self._load_reusable_storyboard(styled)
        if existing is not None:
            logger.info(
                "Storyboard is up to date. Edit %s to change any clip, or pass "
                "--force to redo styling + analysis from scratch.",
                self.workspace.default_storyboard_json,
            )
            # Refresh the readable views so hand-edits to the JSON show up.
            write_storyboard_markdown(existing, self.workspace.storyboard_md)
            write_storyboard_preview(
                existing, self.workspace.root, self.workspace.storyboard_preview
            )
            self._announce_storyboard_ready()
            return

        storyboard = self._build_mode_a_storyboard(styled)
        self._save_storyboard(storyboard)
        self._announce_storyboard_ready()

    def _load_reusable_storyboard(self, styled: list[Path]) -> Optional[Storyboard]:
        """Return the saved storyboard if it still matches `styled`, else None.

        None means "build a fresh one": no saved storyboard, --force, an
        unreadable file, or the styled frames on disk no longer match the
        frames the storyboard was written for (images added/removed).
        """
        path = self.workspace.default_storyboard_json
        if self.force or not path.exists():
            return None
        try:
            storyboard = Storyboard.load(path)
        except StoryboardError as exc:
            logger.warning("Ignoring unreadable storyboard (%s); re-analysing.", exc)
            return None
        current = [p.relative_to(self.workspace.root).as_posix() for p in styled]
        saved = [f.output_path for f in storyboard.frames]
        if saved != current:
            logger.info(
                "Styled frames changed since the storyboard was written "
                "(%d saved vs %d on disk); re-analysing.", len(saved), len(current),
            )
            return None
        logger.info(
            "Using existing storyboard %s — edit it to change any clip, or pass "
            "--force to re-analyse from scratch.", path,
        )
        return storyboard

    def _build_mode_a_storyboard(self, styled: list[Path]) -> Storyboard:
        """Build a storyboard whose frames point at the existing styled images.

        The per-clip motion prompt + duration come from the vision analysis (see
        ``_plan_mode_a_transitions``); the storyboard is always fully populated.
        """
        plans = self._plan_mode_a_transitions(styled)
        style = self.options.style_prompt or self.config.style_prompt
        frames = [
            Frame(
                id=f"{i:03d}",
                description="",
                image_prompt="",
                output_path=p.relative_to(self.workspace.root).as_posix(),
            )
            for i, p in enumerate(styled, start=1)
        ]
        transitions: list[Transition] = []
        for idx, (a, b) in enumerate(zip(frames, frames[1:])):
            motion, duration, sound = plans[idx]
            tid = f"{a.id}_to_{b.id}"
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
        return Storyboard(
            project_title=self.workspace.root.name,
            style=style,
            duration_per_clip=self.options.duration or self.config.duration,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            frames=frames,
            transitions=transitions,
        )

    def _plan_mode_a_transitions(
        self, styled: list[Path]
    ) -> list[tuple[str, int, str]]:
        """Per-pair (motion, duration, sound) plans for the styled frames.

        Uses the vision analysis to tailor each clip. Falls back to the global
        motion prompt and one duration for every clip when analysis is off
        (--no-analyze), during a dry-run (frames aren't on disk yet), or if the
        call fails — so a planning hiccup never sinks the run.
        """
        fallback = (
            self._motion_prompt(),
            self.options.duration or self.config.duration,
            "",
        )
        n = len(styled)
        if self.dry_run or not self.options.analyze_frames:
            return [fallback] * (n - 1)

        style = self.options.style_prompt or self.config.style_prompt
        logger.info("Analysing %d styled frame(s) to plan smooth transitions...", n)
        try:
            return self.openai.analyze_frame_transitions(
                styled, style, default_duration=self.options.duration
            )
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            logger.warning(
                "Frame analysis failed (%s); using the default motion prompt "
                "and one duration for every clip.", exc,
            )
            return [fallback] * (n - 1)

    def _style_images(self, images: list[Path]) -> list[Path]:
        style_prompt = self.options.style_prompt or self.config.style_prompt
        styled = [
            self.workspace.styled_images_dir / f"{i:03d}_styled.png"
            for i in range(1, len(images) + 1)
        ]

        def work(job: tuple[int, Path]) -> None:
            idx, src = job
            dst = styled[idx - 1]
            job_id = f"style:{dst.name}"

            if dst.exists() and not self.force:
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
                self.openai.style_image(src, style_prompt, dst)
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

        self._map_parallel(
            list(enumerate(images, start=1)), work, "Styling images", "img"
        )
        if self.dry_run:
            return styled  # nothing on disk yet; report the plan as-is
        # Only hand back frames that exist: a failed styling must not leak a
        # missing path into transition planning (one bad frame would otherwise
        # crash the vision call and degrade EVERY clip to the generic prompt).
        existing = [p for p in styled if p.exists()]
        if len(existing) < len(styled):
            logger.warning(
                "%d image(s) failed to style; planning transitions from the "
                "%d that succeeded.", len(styled) - len(existing), len(existing),
            )
        return existing

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
        self._storyboard_music_prompt = storyboard.music_prompt or ""

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

        Mode A frames (styled images) have no image prompt and are never
        touched here; Mode B frames are (re)generated when missing or --force.
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
            pairs.append((a, b, motion, duration, sound))
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

    def _bridge_pairs(self, ordered: list[Path]) -> list[tuple[Path, Path]]:
        """Pair consecutive frames, bridging over any that are missing on disk.

        If a frame failed to generate, it is skipped and its nearest existing
        neighbours are paired directly (e.g. frame 4 missing -> ...3->5...), so
        the final video stays continuous instead of leaving a gap. During a
        dry-run the files don't exist yet, so the naive full pairing is used for
        the plan.
        """
        if self.dry_run:
            existing = list(ordered)
        else:
            existing = [p for p in ordered if p.exists()]
            missing = len(ordered) - len(existing)
            if missing:
                logger.warning(
                    "%d frame(s) missing; bridging over them by pairing the "
                    "nearest existing neighbours so the video stays continuous.",
                    missing,
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
                        start, end, motion, duration, dst
                    )
                    with self._lock:
                        # A fresh clip file invalidates its per-clip audio work:
                        # without this, a regenerated clip would skip SFX/fade
                        # ("done" from the previous file) and come out silent.
                        self.state.clear(f"sfx:{dst.name}", f"fade:{dst.name}")
                        self.state.set(job_id, "done", output=str(dst))
                        self.summary.videos_created += 1
                    logger.info("Clip ready: %s", dst.name)
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

    def _clip_name(self, start: Path, end: Path) -> Path:
        """Map a frame pair to clips/<start>_to_<end>.mp4 using leading ids."""
        def stem_id(p: Path) -> str:
            m = re.match(r"(\d+)", p.stem)
            return m.group(1) if m else p.stem
        return self.workspace.clips_dir / f"{stem_id(start)}_to_{stem_id(end)}.mp4"

    # ------------------------------ audio step ---------------------------- #
    def cmd_audio(self) -> None:
        """Add SFX + music to already-rendered clips, then rebuild the final video.

        Per-clip SFX prompts come from the saved storyboard when there is one;
        otherwise every clip uses config.default_sfx_prompt.
        """
        self.audio_enabled = True
        clips = self._clips_for_combine()
        if not clips:
            logger.warning("No clips in %s to add audio to.", self.workspace.clips_dir)
            return

        sound_map: dict[str, str] = {}
        sb_path = self.workspace.default_storyboard_json
        if sb_path.exists():
            try:
                sb = Storyboard.load(sb_path)
                self._storyboard_music_prompt = sb.music_prompt or ""
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

        logger.info("Combining %d clip(s) into %s", len(clips), final_video)
        try:
            combine_clips(clips, final_video)
            self.summary.final_video = final_video
            logger.info("Final video ready: %s", final_video)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to combine clips: %s", exc)
            self.failed.record("combine", "combine", str(exc))
            return

        # Lay the chosen music bed under the whole video (ducked under the SFX).
        if self.audio_enabled:
            self._add_music(music_file)

    def _resolve_music_prompt(self) -> str:
        return (
            self.options.music_prompt
            or self._storyboard_music_prompt
            or self.config.music_prompt
        )

    def _resolve_music_file(self) -> Optional[Path]:
        """Decide which music track to lay under the final video.

        --music-file wins (and must exist). Otherwise the project's
        output/music.mp3 is reused when present (unless --force), and failing
        that a track is generated from the resolved music prompt. Returns None
        to proceed without music.
        """
        if self.options.music_file:
            supplied = Path(self.options.music_file).expanduser()
            if not supplied.exists():
                raise PipelineError(f"--music-file not found: {supplied}")
            logger.info("Using music file: %s", supplied)
            return supplied
        default = self.workspace.music_file
        if default.exists() and not self.force:
            logger.info("Reusing music file: %s", default)
            return default
        return self._generate_music_file(default)

    def _generate_music_file(self, dst: Path) -> Optional[Path]:
        """Generate a music track into `dst` from the resolved prompt.

        Returns `dst` on success, or None when there's no prompt or generation
        fails (the run still finishes, just without a music bed).
        """
        prompt = self._resolve_music_prompt().strip()
        if not prompt:
            logger.info("No music_prompt set; skipping music bed.")
            return None
        try:
            self.audio_client.generate_music(prompt, dst)
            return dst
        except Exception as exc:  # noqa: BLE001
            logger.error("Music generation failed: %s", exc)
            self.failed.record("music:final", "music", str(exc))
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
            )
            self.state.set(job_id, "done")
            self.summary.music_added = True
            logger.info("Music bed added to %s", self.workspace.final_video)
        except Exception as exc:  # noqa: BLE001
            self.state.set(job_id, "failed")
            self.failed.record(job_id, "music", str(exc))

    # ------------------------------ status step --------------------------- #
    def cmd_status(self) -> None:
        """Print where this project stands and what to run next."""
        ws = self.workspace
        line = "=" * 60
        print(f"\n{line}\nPROJECT STATUS: {ws.root.name}\n{line}")

        inputs = list_input_images(ws.input_images_dir)
        styled = sorted(
            (p for p in ws.styled_images_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".png"),
            key=natural_sort_key,
        ) if ws.styled_images_dir.exists() else []
        generated = sorted(
            (p for p in ws.generated_frames_dir.iterdir()
             if p.is_file() and p.suffix.lower() == ".png"),
            key=natural_sort_key,
        ) if ws.generated_frames_dir.exists() else []
        print(f"  Input images     : {len(inputs)}")
        print(f"  Styled images    : {len(styled)}")
        if generated:
            print(f"  Generated frames : {len(generated)}")

        sb_path = ws.default_storyboard_json
        storyboard: Optional[Storyboard] = None
        if sb_path.exists():
            try:
                storyboard = Storyboard.load(sb_path)
                mode = "from idea" if any(
                    f.image_prompt.strip() for f in storyboard.frames
                ) else "from images"
                print(
                    f"  Storyboard       : {len(storyboard.frames)} frame(s), "
                    f"{len(storyboard.transitions)} transition(s) ({mode})"
                )
            except StoryboardError as exc:
                print(f"  Storyboard       : UNREADABLE ({exc})")
        else:
            print("  Storyboard       : none")

        rendered = missing = 0
        if storyboard is not None:
            frames = [ws.root / f.output_path for f in storyboard.frames]
            expected = [self._clip_name(a, b) for a, b in self._bridge_pairs(frames)]
            for clip in expected:
                if clip.exists():
                    rendered += 1
                    sfx = "sfx ✓" if self.state.is_done(f"sfx:{clip.name}") else "silent"
                    print(f"    clip {clip.stem:<12} rendered  ({sfx})")
                else:
                    missing += 1
                    print(f"    clip {clip.stem:<12} MISSING")
            stray = sorted(
                set(find_generated_clips(ws.clips_dir)) - set(expected)
            )
            if stray:
                print(f"  Stray clips      : {', '.join(p.name for p in stray)}")

        final = ws.final_video
        print(f"  Final video      : {'ready — ' + str(final) if final.exists() else 'not built'}")
        if self.failed.path.exists():
            print(f"  Failed jobs      : see {self.failed.path}")

        if storyboard is None:
            hint = self._next_command("storyboard")
        elif missing:
            hint = self._next_command("render")
        elif not final.exists():
            hint = self._next_command("combine")
        else:
            hint = None
        if hint:
            print(f"\n  Next step:\n    {hint}")
        print(line)

    # ------------------------------- one-shot ----------------------------- #
    def cmd_run(self) -> None:
        """The whole flow in one command, gated by confirmation prompts.

        Reuses the saved storyboard when it's still valid; otherwise creates
        one (from images, or from --idea when given), then renders and
        combines. Splitting the flow across `storyboard`/`render`/`combine`
        gives the same result with an editable pause between each step.
        """
        if self.options.idea or self.options.idea_file:
            storyboard = self._create_storyboard_from_idea()
            if storyboard is None:  # dry-run: no plan to continue from
                return
        else:
            images = list_input_images(self.workspace.input_images_dir)
            self.summary.input_count = len(images)
            if not images:
                raise PipelineError(
                    f"No supported images found in {self.workspace.input_images_dir}. "
                    f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
                )
            logger.info("Found %d input image(s).", len(images))
            styled = self._style_images(images)
            if len(styled) < 2:
                logger.warning(
                    "Need at least 2 styled images to make a clip; have %d.",
                    len(styled),
                )
                return
            storyboard = self._load_reusable_storyboard(styled)
            if storyboard is None:
                storyboard = self._build_mode_a_storyboard(styled)
                if not self.dry_run:
                    self._save_storyboard(storyboard)
                    logger.info(
                        "Planned %d transition(s); storyboard written to %s",
                        len(storyboard.transitions), self.workspace.storyboard_md,
                    )

        self.summary.input_count = self.summary.input_count or len(storyboard.frames)
        self._storyboard_music_prompt = storyboard.music_prompt or ""
        self._generate_frames(storyboard)
        pairs = self._pairs_from_storyboard(storyboard)
        self._render_pairs(pairs, set())
        if not self.options.no_combine:
            self._combine_clips(confirm=True)
