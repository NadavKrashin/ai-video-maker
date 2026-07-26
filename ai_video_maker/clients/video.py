"""Image-to-video generation (fal.ai). Auth via FAL_KEY.

Given a start frame (and, for models that support it, an end frame), fal renders
a short clip that interpolates between them following the motion prompt. The
model id and request shape come entirely from the ``fal_*`` config fields, so
switching models needs no code change.

Clips are the pipeline's most expensive API calls, so they go through fal's
queue API (submit -> persist request_id -> poll) rather than a blocking
subscribe: once a job is submitted the money is spent, and the persisted
request_id (state ``falreq:<clip>``) lets an interrupted run fetch the finished
video later instead of paying for a second render.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config
from ..errors import PipelineCancelled
from ..logging_setup import logger
from ..retry import is_moderation_error, with_reword_recovery
from ..state import StateStore
from .download import download_file
from .fal import FalSession, extract_media_url

# Final fallback when every reword of a motion prompt is still rejected by the
# content checker (with_reword_recovery's last_resort). Deliberately generic:
# no physical-contact verbs, no furniture, no body words — nothing left for the
# checker to misread. The start/end frames still drive the clip, so a bland
# prompt produces a usable (if less directed) result instead of a failed clip.
SAFE_FALLBACK_MOTION_PROMPT = (
    "Smooth, gentle, natural motion carries the first frame into the second. "
    "Everyone moves calmly and naturally in a warm, wholesome moment. The "
    "final frame closely matches the provided end image. No sudden cuts, no "
    "text, no distortion."
)


# Frame hashes, shared across VideoClient instances and keyed on identity +
# mtime + size. The admin panel builds a fresh Pipeline (and VideoClient) for
# every status poll, so the per-instance cache never hit: reporting pending
# renders re-read and re-hashed multi-megabyte frames every few seconds. A
# changed mtime or size only costs a recompute, so a lying mtime is harmless
# here — unlike staleness decisions, which must stay content-based.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _file_hash_cached(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = hashlib.sha1(path.read_bytes()).hexdigest()
    return _HASH_CACHE[key]


class VideoClient:
    """Renders one clip per consecutive frame pair via fal."""

    def __init__(
        self,
        config: Config,
        state: Optional[StateStore] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.config = config
        self.state = state
        # Checked between fal polls so a cancel reaches a clip that is already
        # waiting, not just the gaps between clips.
        self.should_cancel = should_cancel
        self.fal = FalSession(config)
        # Cache uploaded-image URLs so a frame shared by two consecutive clips
        # is only uploaded once per run.
        self._upload_cache: dict[Path, str] = {}
        self._hash_cache: dict[Path, str] = {}

    def _upload(self, path: Path) -> str:
        if path not in self._upload_cache:
            self._upload_cache[path] = self.fal.upload(
                path, description=f"fal upload ({path.name})"
            )
        return self._upload_cache[path]

    def _file_hash(self, path: Path) -> str:
        if path not in self._hash_cache:
            self._hash_cache[path] = _file_hash_cached(path)
        return self._hash_cache[path]

    def fingerprint(
        self, start: Path, end: Path, motion_prompt: str, duration: int
    ) -> str:
        """Identity of one render job, from everything that shapes its output.

        A persisted request_id is only reused when the fingerprint still
        matches — if the user re-styled a frame or edited the prompt between
        runs, the pending job renders the OLD plan and must not be resumed.
        The fingerprint uses the original (pre-reword) prompt, so a job whose
        prompt was reworded by moderation recovery is still recognised.

        Config knobs that reshape the request (negative prompt, cfg_scale,
        extra arguments) are part of the identity too, but only WHEN SET:
        appending them unconditionally would change every fingerprint and
        orphan pending jobs submitted before the knobs existed — a mismatch
        forfeits an already-paid render.
        """
        parts = [
            self.config.fal_model_id,
            str(duration),
            self._file_hash(start),
            self._file_hash(end),
            motion_prompt,
        ]
        if self.config.fal_negative_prompt:
            parts.append(f"neg={self.config.fal_negative_prompt}")
        if self.config.fal_cfg_scale is not None:
            parts.append(f"cfg={self.config.fal_cfg_scale}")
        if self.config.fal_extra_arguments:
            parts.append(
                "extra=" + json.dumps(self.config.fal_extra_arguments,
                                      sort_keys=True)
            )
        material = "|".join(parts)
        return hashlib.sha1(material.encode("utf-8")).hexdigest()

    def _try_resume(
        self, job_key: str, fingerprint: str, dst: Path
    ) -> Optional[dict[str, Any]]:
        """Fetch the result of a previously submitted, still-pending job.

        Returns the fal result dict when the pending request could be
        recovered, None when there is nothing (valid) to resume. Never raises:
        an unrecoverable pending job falls back to a fresh submission.
        """
        if self.state is None:
            return None
        entry = self.state.get(job_key)
        if not entry:
            return None
        request_id = entry.get("request_id")
        if not request_id or entry.get("fingerprint") != fingerprint:
            # The frames/prompt/duration changed since the job was submitted:
            # it renders an outdated plan, so forget it rather than resume it.
            self.state.clear(job_key)
            return None
        logger.info(
            "Resuming pending fal request for %s (request_id=%s) — already "
            "submitted and paid for; fetching instead of resubmitting.",
            dst.name, request_id,
        )
        try:
            return self.fal.wait_for_result(
                self.config.fal_model_id, request_id,
                description=f"fal resume {dst.name}",
                should_cancel=self.should_cancel,
            )
        except PipelineCancelled:
            # A cancel is not a failed recovery: falling through here would
            # submit (and pay for) a brand-new job in response to the user
            # asking to stop. Keep the entry so the next run resumes it.
            raise
        except Exception as exc:  # noqa: BLE001 - resume is best-effort
            logger.warning(
                "Pending fal request for %s could not be recovered (%s) — "
                "submitting a fresh job.",
                dst.name, exc,
            )
            self.state.clear(job_key)
            return None

    def _build_arguments(
        self,
        start_url: str,
        end_url: Optional[str],
        motion_prompt: str,
        duration: int,
    ) -> dict[str, Any]:
        c = self.config
        args: dict[str, Any] = {
            c.fal_start_frame_field: start_url,
            "prompt": motion_prompt,
            "duration": str(duration) if c.fal_duration_as_string else duration,
        }
        # End frame is only sent when the model documents a field for it.
        if end_url and c.fal_end_frame_field:
            args[c.fal_end_frame_field] = end_url
        if c.fal_resolution:
            args["resolution"] = c.fal_resolution
        if c.fal_aspect_ratio:
            args["aspect_ratio"] = c.fal_aspect_ratio
        if c.fal_negative_prompt:
            args["negative_prompt"] = c.fal_negative_prompt
        if c.fal_cfg_scale is not None:
            args["cfg_scale"] = c.fal_cfg_scale
        # Last, so ad-hoc extra arguments can override anything above.
        args.update(c.fal_extra_arguments)
        return args

    def generate_clip(
        self,
        start_frame: Path,
        end_frame: Path,
        motion_prompt: str,
        duration: int,
        dst: Path,
        reword: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Render the start->end clip and download it to `dst`.

        When `reword` is given and fal's content checker rejects the motion
        prompt (content_policy_violation — usually a false positive on
        innocent wording), the prompt is reworded and the job resubmitted, up
        to ``config.moderation_reword_attempts`` times — the same recovery the
        image styling has. The frames are uploaded once; only the prompt
        changes between attempts.

        Each submission's request_id is persisted (state ``falreq:<clip>``)
        before waiting, and cleared once the clip is downloaded. If the wait
        is interrupted — dropped connection, crash, cancellation — the next
        run finds the entry and fetches the already-paid result instead of
        rendering (and billing) the clip again. The entry survives transient
        failures on purpose; it is only dropped when the job itself failed
        (moderation) or no longer matches the storyboard (fingerprint).
        """
        job_key = f"falreq:{dst.name}"
        fingerprint = self.fingerprint(start_frame, end_frame, motion_prompt, duration)

        result = self._try_resume(job_key, fingerprint, dst)
        if result is None:
            start_url = self._upload(start_frame)
            end_url = (
                self._upload(end_frame) if self.config.fal_end_frame_field else None
            )
            logger.info(
                "fal job: %s (model=%s, start=%s%s)",
                dst.name,
                self.config.fal_model_id,
                start_frame.name,
                f", end={end_frame.name}" if end_url else "",
            )

            def run(prompt: str) -> dict[str, Any]:
                arguments = self._build_arguments(start_url, end_url, prompt, duration)
                request_id = self.fal.submit(
                    self.config.fal_model_id, arguments,
                    description=f"fal generate {dst.name}",
                )
                if self.state is not None:
                    self.state.set(
                        job_key, "pending",
                        request_id=request_id, fingerprint=fingerprint,
                    )
                return self.fal.wait_for_result(
                    self.config.fal_model_id, request_id,
                    description=f"fal generate {dst.name}",
                    should_cancel=self.should_cancel,
                )

            try:
                if reword is None:
                    result = run(motion_prompt)
                else:
                    result = with_reword_recovery(
                        run,
                        motion_prompt,
                        reword=reword,
                        attempts=self.config.moderation_reword_attempts,
                        description=f"fal clip {dst.name}",
                        last_resort=SAFE_FALLBACK_MOTION_PROMPT,
                    )
            except Exception as exc:
                if is_moderation_error(exc):
                    # The submitted job was rejected outright — there is no
                    # output to recover later, so drop the pending entry.
                    if self.state is not None:
                        self.state.clear(job_key)
                    if reword is not None:
                        logger.error(
                            "%s: suspect the frames, not the prompt — try "
                            "re-rolling the styling of %s or %s (delete the "
                            "styled image and re-run storyboard), then render "
                            "this clip again.",
                            dst.name, start_frame.name, end_frame.name,
                        )
                # Anything else (network outage, timeout, Ctrl-C-adjacent
                # crash) keeps the entry: the job may still finish server-side
                # and the next run will resume it via the request_id.
                raise
        video_url = extract_media_url(result, ("video",))
        download_file(
            video_url,
            dst,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"fal download -> {dst.name}",
        )
        if self.state is not None:
            self.state.clear(job_key)
