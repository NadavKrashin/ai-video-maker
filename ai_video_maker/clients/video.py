"""Image-to-video generation (fal.ai). Auth via FAL_KEY.

Given a start frame (and, for models that support it, an end frame), fal renders
a short clip that interpolates between them following the motion prompt. The
model id and request shape come entirely from the ``fal_*`` config fields, so
switching models needs no code change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config
from ..logging_setup import logger
from ..retry import with_reword_recovery
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


class VideoClient:
    """Renders one clip per consecutive frame pair via fal."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.fal = FalSession(config)
        # Cache uploaded-image URLs so a frame shared by two consecutive clips
        # is only uploaded once per run.
        self._upload_cache: dict[Path, str] = {}

    def _upload(self, path: Path) -> str:
        if path not in self._upload_cache:
            self._upload_cache[path] = self.fal.upload(
                path, description=f"fal upload ({path.name})"
            )
        return self._upload_cache[path]

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
        """
        start_url = self._upload(start_frame)
        end_url = self._upload(end_frame) if self.config.fal_end_frame_field else None
        logger.info(
            "fal job: %s (model=%s, start=%s%s)",
            dst.name,
            self.config.fal_model_id,
            start_frame.name,
            f", end={end_frame.name}" if end_url else "",
        )

        def run(prompt: str) -> dict[str, Any]:
            arguments = self._build_arguments(start_url, end_url, prompt, duration)
            return self.fal.subscribe(
                self.config.fal_model_id, arguments,
                description=f"fal generate {dst.name}",
            )

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
        video_url = extract_media_url(result, ("video",))
        download_file(
            video_url,
            dst,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"fal download -> {dst.name}",
        )
