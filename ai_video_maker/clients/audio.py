"""Post-generation audio (fal.ai). Auth via FAL_KEY.

Two layers:
  * SFX/ambient — a video->audio model runs on a silent clip and returns the
    SAME clip with synced sound; the clip file is replaced in place.
  * Music — one instrumental track generated from a text prompt, mixed under the
    SFX across the final video (the muxing itself happens in media/ffmpeg.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..logging_setup import logger
from .download import download_file
from .fal import FalSession, extract_media_url


class AudioClient:
    """fal-backed per-clip SFX (video->audio) + a text->music track."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.fal = FalSession(config)

    def _download(self, url: str, dst: Path) -> None:
        download_file(
            url,
            dst,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"audio download -> {dst.name}",
        )

    def add_sfx(self, clip: Path, prompt: str, duration: int) -> None:
        """Generate synced SFX/ambient audio for `clip` and replace it in place."""
        clip_url = self.fal.upload(clip, description=f"audio upload ({clip.name})")
        arguments: dict[str, Any] = {
            "video_url": clip_url,
            "prompt": prompt,
            "negative_prompt": self.config.sfx_negative_prompt,
            "duration": float(duration),
            "num_steps": self.config.sfx_num_steps,
        }
        arguments.update(self.config.sfx_extra_arguments)
        logger.info("SFX job: %s (model=%s)", clip.name, self.config.sfx_model_id)
        result = self.fal.subscribe(
            self.config.sfx_model_id, arguments, description="SFX generate"
        )
        self._download(extract_media_url(result, ("video",)), clip)

    def generate_music(self, prompt: str, dst: Path) -> None:
        """Generate one instrumental track from `prompt` and save it to `dst`."""
        arguments: dict[str, Any] = {"prompt": prompt}
        arguments.update(self.config.music_extra_arguments)
        logger.info("Music job: %s (model=%s)", dst.name, self.config.music_model_id)
        result = self.fal.subscribe(
            self.config.music_model_id, arguments, description="music generate"
        )
        self._download(
            extract_media_url(result, ("audio", "audio_file", "video")), dst
        )
