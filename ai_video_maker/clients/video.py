"""Video clients (isolated). Image-to-video generation.

Two providers are supported, selected by config.video_provider:
  * "fal"        -> fal.ai (recommended for start+end frame interpolation;
                    Kling on fal supports an end/tail frame). Auth: FAL_KEY.
  * "higgsfield" -> Higgsfield. Auth: HF_KEY (or HF_API_KEY + HF_API_SECRET).

Both fal-client and higgsfield-client expose the SAME interface:
  upload_file(path) -> hosted URL, and subscribe(model_id, args) -> result
  dict with result["video"]["url"]. So the shared logic lives in the base
  class below; each subclass only supplies its SDK + credential check.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..logging_setup import logger
from ..retry import with_retries


@dataclass
class VideoBackend:
    """Provider-specific settings, read from config."""
    model_id: str
    start_frame_field: str
    end_frame_field: str
    duration_as_string: bool
    resolution: str
    aspect_ratio: str
    extra_arguments: dict[str, Any]


class SubscribeVideoClient:
    """
    Base image-to-video client for fal-style SDKs (fal-client, higgsfield-client).

    Subclasses implement `_import_sdk()` and `_check_credentials()`.
    """

    provider: str = "video"

    def __init__(self, config: Config, backend: VideoBackend) -> None:
        self.config = config
        self.backend = backend
        self._sdk = None  # lazily imported
        # Cache uploaded-image URLs so a frame shared by two consecutive clips
        # is only uploaded once per run.
        self._upload_cache: dict[Path, str] = {}

    # --- provider hooks (override in subclasses) --------------------------- #
    def _import_sdk(self):  # pragma: no cover - trivial
        raise NotImplementedError

    def _check_credentials(self) -> None:  # pragma: no cover - trivial
        raise NotImplementedError

    def _ensure_sdk(self):
        if self._sdk is None:
            self._check_credentials()
            self._sdk = self._import_sdk()
        return self._sdk

    # --- UPLOAD ------------------------------------------------------------ #
    def _upload(self, path: Path) -> str:
        """Upload a local image and return its hosted URL (cached per run)."""
        if path in self._upload_cache:
            return self._upload_cache[path]
        sdk = self._ensure_sdk()

        def _call() -> str:
            return sdk.upload_file(path)

        url = with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"{self.provider} upload ({path.name})",
        )
        self._upload_cache[path] = url
        return url

    # --- BUILD REQUEST ----------------------------------------------------- #
    def _build_arguments(
        self,
        start_url: str,
        end_url: Optional[str],
        motion_prompt: str,
        duration: int,
    ) -> dict[str, Any]:
        b = self.backend
        duration_value: Any = str(duration) if b.duration_as_string else duration
        args: dict[str, Any] = {
            b.start_frame_field: start_url,  # start frame
            "prompt": motion_prompt,         # motion prompt
            "duration": duration_value,
        }
        # End frame is only sent when the model documents a field for it.
        if end_url and b.end_frame_field:
            args[b.end_frame_field] = end_url
        if b.resolution:
            args["resolution"] = b.resolution
        if b.aspect_ratio:
            args["aspect_ratio"] = b.aspect_ratio
        args.update(b.extra_arguments)
        return args

    # --- SUBMIT + WAIT ----------------------------------------------------- #
    def _generate_video_url(self, arguments: dict[str, Any]) -> str:
        """Submit the job, wait for completion, and return the video URL."""
        sdk = self._ensure_sdk()

        def _call() -> str:
            # subscribe() submits and blocks until the job reaches a terminal
            # state, returning the result dict (or raising on failure).
            result = sdk.subscribe(self.backend.model_id, arguments)
            video = (result or {}).get("video") or {}
            url = video.get("url")
            if not url:
                raise RuntimeError(
                    f"{self.provider} finished without a video URL: {result}"
                )
            return url

        return with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"{self.provider} generate",
        )

    # --- DOWNLOAD ---------------------------------------------------------- #
    def download(self, video_url: str, dst: Path) -> None:
        """Stream the result video to `dst` (atomic via a .part temp file)."""
        import requests

        dst.parent.mkdir(parents=True, exist_ok=True)

        def _call() -> None:
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(video_url, stream=True, timeout=300) as resp:
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
            description=f"{self.provider} download -> {dst.name}",
        )

    # --- High-level convenience -------------------------------------------- #
    def generate_clip(
        self,
        start_frame: Path,
        end_frame: Path,
        motion_prompt: str,
        duration: int,
        dst: Path,
    ) -> None:
        start_url = self._upload(start_frame)
        end_url = self._upload(end_frame) if self.backend.end_frame_field else None
        arguments = self._build_arguments(start_url, end_url, motion_prompt, duration)
        logger.info(
            "%s job: %s (model=%s, start=%s%s)",
            self.provider,
            dst.name,
            self.backend.model_id,
            start_frame.name,
            f", end={end_frame.name}" if end_url else "",
        )
        video_url = self._generate_video_url(arguments)
        self.download(video_url, dst)


class FalClient(SubscribeVideoClient):
    """fal.ai backend. Docs: https://docs.fal.ai — auth via FAL_KEY."""

    provider = "fal"

    def __init__(self, config: Config) -> None:
        super().__init__(
            config,
            VideoBackend(
                model_id=config.fal_model_id,
                start_frame_field=config.fal_start_frame_field,
                end_frame_field=config.fal_end_frame_field,
                duration_as_string=config.fal_duration_as_string,
                resolution=config.fal_resolution,
                aspect_ratio=config.fal_aspect_ratio,
                extra_arguments=config.fal_extra_arguments,
            ),
        )

    def _check_credentials(self) -> None:
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError(
                "fal credentials missing. Set FAL_KEY in your .env file "
                "(get one at https://fal.ai/dashboard/keys)."
            )

    def _import_sdk(self):
        import fal_client  # imported lazily

        return fal_client


class HiggsfieldClient(SubscribeVideoClient):
    """Higgsfield backend. Docs: https://docs.higgsfield.ai — auth via HF_KEY."""

    provider = "higgsfield"

    def __init__(self, config: Config) -> None:
        super().__init__(
            config,
            VideoBackend(
                model_id=config.higgsfield_model_id,
                start_frame_field=config.higgsfield_start_frame_field,
                end_frame_field=config.higgsfield_end_frame_field,
                duration_as_string=config.higgsfield_duration_as_string,
                resolution=config.higgsfield_resolution,
                aspect_ratio=config.higgsfield_aspect_ratio,
                extra_arguments=config.higgsfield_extra_arguments,
            ),
        )

    def _check_credentials(self) -> None:
        if not os.environ.get("HF_KEY") and not (
            os.environ.get("HF_API_KEY") and os.environ.get("HF_API_SECRET")
        ):
            raise RuntimeError(
                "Higgsfield credentials missing. Set HF_KEY "
                '("<api_key>:<api_secret>") or HF_API_KEY + HF_API_SECRET '
                "in your .env file."
            )

    def _import_sdk(self):
        import higgsfield_client  # imported lazily

        return higgsfield_client


def make_video_client(config: Config) -> SubscribeVideoClient:
    """Instantiate the video client selected by config.video_provider."""
    provider = (config.video_provider or "fal").lower()
    if provider == "fal":
        return FalClient(config)
    if provider == "higgsfield":
        return HiggsfieldClient(config)
    raise SystemExit(
        f"Unknown video_provider {provider!r}. Use 'fal' or 'higgsfield'."
    )
