"""Audio client (post-generation sound). Always uses fal (FAL_KEY).

Two layers: per-clip SFX (video->audio, returns the clip with synced sound) and
a single music bed (text->track). Uses the fal SDK directly so audio works no
matter which provider rendered the silent clips.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..logging_setup import logger
from ..retry import with_retries


def _extract_fal_url(result: Optional[dict[str, Any]], keys: tuple[str, ...]) -> str:
    """Pull a media URL out of a fal result, trying several common shapes."""
    result = result or {}
    for key in (*keys, "url"):
        value = result.get(key)
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
        if isinstance(value, str) and value.startswith("http"):
            return value
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and value[0].get("url")
        ):
            return value[0]["url"]
    raise RuntimeError(f"fal result had no media URL: {result}")


class AudioClient:
    """fal-backed audio generation: per-clip SFX (video->audio) + music (text).

    Uses the fal SDK directly (auth via FAL_KEY) so audio works no matter which
    provider rendered the silent clips.
    """

    provider = "fal-audio"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._sdk = None

    def _ensure_sdk(self):
        if self._sdk is None:
            if not os.environ.get("FAL_KEY"):
                raise RuntimeError(
                    "fal credentials missing for audio. Set FAL_KEY in your .env "
                    "file (get one at https://fal.ai/dashboard/keys)."
                )
            import fal_client  # imported lazily

            self._sdk = fal_client
        return self._sdk

    def _upload(self, path: Path) -> str:
        sdk = self._ensure_sdk()

        def _call() -> str:
            return sdk.upload_file(path)

        return with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"audio upload ({path.name})",
        )

    def _subscribe(self, model_id: str, arguments: dict[str, Any],
                   keys: tuple[str, ...]) -> str:
        sdk = self._ensure_sdk()

        def _call() -> str:
            result = sdk.subscribe(model_id, arguments)
            return _extract_fal_url(result, keys)

        return with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"audio generate ({model_id})",
        )

    def _download(self, url: str, dst: Path) -> None:
        import requests

        dst.parent.mkdir(parents=True, exist_ok=True)

        def _call() -> None:
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dst)

        with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"audio download -> {dst.name}",
        )

    # --- SFX: video -> the same video with synced audio ------------------- #
    def add_sfx(self, clip: Path, prompt: str, duration: int) -> None:
        """Generate synced SFX/ambient audio for `clip` and replace it in place."""
        clip_url = self._upload(clip)
        arguments: dict[str, Any] = {
            "video_url": clip_url,
            "prompt": prompt,
            "negative_prompt": self.config.sfx_negative_prompt,
            "duration": float(duration),
            "num_steps": self.config.sfx_num_steps,
        }
        arguments.update(self.config.sfx_extra_arguments)
        logger.info("SFX job: %s (model=%s)", clip.name, self.config.sfx_model_id)
        url = self._subscribe(self.config.sfx_model_id, arguments, ("video",))
        self._download(url, clip)

    # --- Music: text -> an instrumental track ----------------------------- #
    def generate_music(self, prompt: str, dst: Path) -> None:
        arguments: dict[str, Any] = {"prompt": prompt}
        arguments.update(self.config.music_extra_arguments)
        logger.info("Music job: %s (model=%s)", dst.name, self.config.music_model_id)
        url = self._subscribe(
            self.config.music_model_id, arguments, ("audio", "audio_file", "video")
        )
        self._download(url, dst)
