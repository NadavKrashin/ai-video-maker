"""Pipeline orchestration.

The pipeline is constructed from three explicit inputs — a validated ``Config``,
a ``Workspace`` (all per-movie paths), and ``RunOptions`` (this run's choices).
Nothing here reads global state or argparse, so the same orchestration can be
driven by the CLI or, later, an API request.
"""
from __future__ import annotations

import re
import sys
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
    verify_dimensions,
)
from .models import Frame, Storyboard, Transition
from .options import RunOptions
from .state import FailedJobStore, StateStore
from .storyboard_md import write_storyboard_markdown
from .summary import RunSummary
from .workspace import PROJECT_ROOT, Workspace


class Pipeline:
    def __init__(
        self, config: Config, workspace: Workspace, options: RunOptions
    ) -> None:
        self.config = config
        self.workspace = workspace
        self.options = options
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
        # --add-audio / --no-audio for a single run.
        if options.no_audio:
            self.audio_enabled = False
        elif options.add_audio or options.audio_only:
            self.audio_enabled = True
        else:
            self.audio_enabled = (config.audio_mode or "none").lower() == "post"
        # Resolved at storyboard-approval time (Mode B); falls back to config.
        self._storyboard_music_prompt: str = ""
        # Concurrency for the I/O-bound generation steps.
        self.concurrency: int = max(
            1, options.concurrency or config.max_parallel_requests
        )
        # Guards summary counters when workers run in parallel (StateStore and
        # FailedJobStore guard themselves).
        self._lock = threading.Lock()

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

    # ------------------------------ Mode A ------------------------------- #
    def run_mode_a(self) -> None:
        logger.info("=== Mode A: image-to-video from input_images/ ===")
        images = list_input_images(self.workspace.input_images_dir)
        self.summary.input_count = len(images)

        if not images:
            raise PipelineError(
                f"No supported images found in {self.workspace.input_images_dir}. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
            )
        logger.info("Found %d input image(s).", len(images))

        styled: list[Path] = []
        if not self.options.only_video:
            styled = self._style_images(images)
        else:
            # --only-video: use existing styled images.
            styled = sorted(
                (p for p in self.workspace.styled_images_dir.iterdir()
                 if p.is_file() and p.suffix.lower() == ".png"),
                key=natural_sort_key,
            )
            logger.info("Using %d existing styled image(s).", len(styled))

        if self.options.only_style:
            logger.info("--only-style set: skipping video generation.")
            return

        if len(styled) < 2:
            logger.warning(
                "Need at least 2 styled images to make a clip; have %d.", len(styled)
            )
            return

        # Plan smooth, per-clip transitions by analysing the styled frames. When
        # that is unavailable (dry-run, --no-analyze, or the call fails) fall back
        # to the single global motion prompt and one duration for every clip.
        storyboard = self._build_mode_a_storyboard(styled)
        if storyboard is not None:
            storyboard.save(self.workspace.default_storyboard_json)
            write_storyboard_markdown(storyboard, self.workspace.storyboard_md)
            logger.info(
                "Planned %d transition(s); storyboard written to %s",
                len(storyboard.transitions), self.workspace.storyboard_md,
            )
            self._generate_clips_with_prompts(self._pairs_from_storyboard(storyboard))
        else:
            self._generate_clips(
                self._bridge_pairs(styled),
                motion_prompt=self._motion_prompt(),
            )

    def _build_mode_a_storyboard(self, styled: list[Path]) -> Optional[Storyboard]:
        """Analyse the styled frames and plan a motion prompt + duration per clip.

        Returns a Storyboard whose frames point at the existing styled images, or
        None to fall back to the old single-motion-prompt behaviour — during a
        dry-run (frames aren't on disk yet), when --no-analyze is set, or if the
        vision call fails (a planning hiccup must not sink the whole run).
        """
        if self.dry_run or not self.options.analyze_frames:
            return None

        style = self.options.style_prompt or self.config.style_prompt
        logger.info("Analysing %d styled frame(s) to plan smooth transitions...", len(styled))
        try:
            plans = self.openai.analyze_frame_transitions(
                styled, style, default_duration=self.options.duration
            )
        except Exception as exc:  # noqa: BLE001 - planning is best-effort
            logger.warning(
                "Frame analysis failed (%s); using the default motion prompt "
                "and one duration for every clip.", exc,
            )
            return None

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
        return styled

    # ------------------------------ Mode B ------------------------------- #
    def run_mode_b(self) -> None:
        logger.info("=== Mode B: generate from scratch ===")

        if self.options.create_storyboard:
            self._create_storyboard()
            return

        if self.options.approve_storyboard:
            self._run_approved_storyboard()
            return

        raise PipelineError(
            "Mode B requires either --create-storyboard (with --idea) or "
            "--approve-storyboard. See README.md."
        )

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
        if self.options.idea:
            return self.options.idea
        raise PipelineError(
            "--create-storyboard requires --idea \"...\" or --idea-file PATH"
        )

    def _create_storyboard(self) -> None:
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
            return

        storyboard = self.openai.create_storyboard(
            idea, frame_count, default_duration=self.options.duration
        )
        storyboard.save(self.workspace.default_storyboard_json)
        write_storyboard_markdown(storyboard, self.workspace.storyboard_md)

        print("\n" + "=" * 70)
        print("Storyboard created. Review storyboard/storyboard.md or "
              "storyboard/storyboard.json,")
        print("edit if needed, then run with --approve-storyboard.")
        print("=" * 70 + "\n")

    def _run_approved_storyboard(self) -> None:
        sb_path = Path(self.options.storyboard_file)
        if not sb_path.is_absolute():
            sb_path = self.workspace.root / sb_path
        storyboard = Storyboard.load(sb_path)
        logger.info(
            "Approved storyboard %r with %d frame(s).",
            storyboard.project_title,
            len(storyboard.frames),
        )
        self.summary.input_count = len(storyboard.frames)
        self._storyboard_music_prompt = storyboard.music_prompt or ""

        if not self.options.only_video:
            self._generate_frames(storyboard)

        if self.options.only_style:
            logger.info("--only-style set: skipping video generation.")
            return

        self._generate_clips_with_prompts(self._pairs_from_storyboard(storyboard))

    def _pairs_from_storyboard(
        self, storyboard: Storyboard
    ) -> list[tuple[Path, Path, str, int, str]]:
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

        pairs: list[tuple[Path, Path, str, int, str]] = []
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

    def _generate_frames(self, storyboard: Storyboard) -> None:
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

        self._map_parallel(
            list(storyboard.frames), work, "Generating frames", "frame"
        )

    # ------------------------- shared video step ------------------------- #
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

    def _generate_clips(
        self, pairs: list[tuple[Path, Path]], motion_prompt: str
    ) -> None:
        # Mode A has no per-clip sound prompt; "" falls back to default_sfx_prompt.
        enriched = [
            (a, b, motion_prompt, self.duration, "") for a, b in pairs
        ]
        self._generate_clips_with_prompts(enriched)

    def _prompt_yes_no(
        self, lines: list[str], question: str, decline_log: str
    ) -> bool:
        """Ask a yes/no question at the terminal; True means proceed.

        Auto-proceeds without prompting for dry-runs, when --yes is set, or when
        stdin isn't interactive (CI/automation), so scripted runs never block.
        A "no" (or anything other than y/yes) logs `decline_log` and returns
        False.
        """
        if self.dry_run or self.options.yes or not sys.stdin.isatty():
            return True

        print("\n" + "=" * 70)
        for line in lines:
            print(line)
        print("=" * 70)
        try:
            answer = input(question).strip().lower()
        except EOFError:
            return True
        if answer in ("y", "yes"):
            return True
        logger.info(decline_log)
        return False

    def _confirm_clip_generation(self) -> bool:
        """Pause after image generation to confirm before generating clips.

        --only-video is an explicit "generate clips now", so it proceeds
        without prompting; otherwise a "no" stops before any clip credits are
        spent and the clips can be generated later with --only-video.
        """
        if self.options.only_video:
            return True
        return self._prompt_yes_no(
            [
                "All images are ready. The next step generates the video "
                "clips, which spends API credits.",
                "You can stop here and generate the clips later by re-running "
                "with --only-video.",
            ],
            "Generate clips now? [y/N] ",
            "Clip generation skipped. Re-run with --only-video to generate the "
            "clips from the existing images.",
        )

    def _confirm_combine(self) -> bool:
        """Pause after clip generation to confirm before building the movie.

        A "no" leaves the clips in place; the final video can be built later by
        re-running with --combine.
        """
        return self._prompt_yes_no(
            [
                "All clips are ready. The final step combines them into "
                f"{self.workspace.final_video.name}.",
                "You can do this later by re-running with --combine.",
            ],
            "Combine clips into the final video now? [y/N] ",
            "Combine skipped. Re-run with --combine to build the final video "
            "from the existing clips.",
        )

    def _generate_clips_with_prompts(
        self, pairs: list[tuple[Path, Path, str, int, str]]
    ) -> None:
        if not pairs:
            logger.warning("No transition pairs to render.")
            return

        if not self._confirm_clip_generation():
            return

        def work(pair: tuple[Path, Path, str, int, str]) -> None:
            start, end, motion, duration, sound_prompt = pair
            dst = self._clip_name(start, end)
            job_id = f"clip:{dst.name}"

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

            if dst.exists() and not self.force:
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

    def _clip_name(self, start: Path, end: Path) -> Path:
        """Map a frame pair to clips/<start>_to_<end>.mp4 using leading ids."""
        def stem_id(p: Path) -> str:
            m = re.match(r"(\d+)", p.stem)
            return m.group(1) if m else p.stem
        return self.workspace.clips_dir / f"{stem_id(start)}_to_{stem_id(end)}.mp4"

    def _combine_clips(
        self, force_rebuild: bool = False, confirm: bool = False
    ) -> None:
        """Concatenate all generated clips into output/final_video.mp4.

        When `confirm` is set (the default end-of-run path) the user is asked
        first — but only once we know there's actually a movie to build, so the
        prompt never appears when there are no clips or the final video is
        already up to date.
        """
        clips = find_generated_clips(self.workspace.clips_dir)
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

        if confirm and not self._confirm_combine():
            return

        # Decide the music track BEFORE combining, so the user makes every
        # choice up front rather than being interrupted after the combine runs.
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

        Interactive (a real terminal, no --yes): ask which file to use,
        defaulting to output/music.mp3. If that file is missing, offer to point
        at another file, generate one with the text->music model, or skip music.
        Non-interactive (--yes / no TTY / dry-run): keep the original behaviour —
        reuse music.mp3 when present, otherwise generate from the prompt.

        Returns a path to a ready music file, or None to proceed without music.
        """
        default = self.workspace.music_file

        if self.dry_run or self.options.yes or not sys.stdin.isatty():
            if default.exists() and not self.force:
                return default
            return self._generate_music_file(default)

        print("\n" + "=" * 70)
        print("Music for the final video.")
        print(f"Default track: {default}"
              f"{'' if default.exists() else '  (not found)'}")
        print("=" * 70)
        while True:
            raw = input(
                "Music file — Enter to use the default, a path to another "
                "file, or 'g' to generate: "
            ).strip()
            if raw.lower() in ("g", "generate"):
                return self._generate_music_file(default)
            candidate = default if not raw else Path(raw).expanduser()
            if candidate.exists():
                logger.info("Using music file: %s", candidate)
                return candidate

            print(f"Not found: {candidate}")
            choice = input(
                "Use [a]nother file, [g]enerate one, or [s]kip music? "
                "[a/g/s]: "
            ).strip().lower()
            if choice in ("g", "generate"):
                return self._generate_music_file(default)
            if choice in ("s", "skip"):
                logger.info("Proceeding without a music bed.")
                return None
            # 'a' / Enter / anything else: loop back and ask for another path.

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

    # --------------------------- audio retrofit -------------------------- #
    def _run_audio_only(self) -> None:
        """Add SFX + music to clips already in clips/, then rebuild the final video.

        Per-clip SFX prompts are taken from --storyboard-file if it exists
        (Mode B), otherwise every clip uses config.default_sfx_prompt.
        """
        clips = find_generated_clips(self.workspace.clips_dir)
        if not clips:
            logger.warning("No clips in %s to add audio to.", self.workspace.clips_dir)
            return

        sound_map: dict[str, str] = {}
        sb_path = Path(self.options.storyboard_file)
        if not sb_path.is_absolute():
            sb_path = self.workspace.root / sb_path
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

    # ------------------------------- run --------------------------------- #
    def run(self) -> None:
        try:
            if self.options.audio_only:
                self._run_audio_only()
                return
            if self.options.combine:
                # Standalone: just stitch the existing clips together.
                self._combine_clips()
                return
            if self.options.from_scratch:
                self.run_mode_b()
            else:
                self.run_mode_a()
            if not self.options.no_combine and not self.options.only_style \
                    and not self.options.create_storyboard:
                self._combine_clips(confirm=True)
        finally:
            self.failed.flush()
            self.summary.print(self.workspace)
