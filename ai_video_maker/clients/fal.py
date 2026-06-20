"""Shared fal.ai access.

Both video (image-to-video) and audio (SFX + music) run on fal, so the SDK
loading, credential check, upload, and job submission live here once. The SDK is
imported lazily so --dry-run / --help work without `fal-client` or `FAL_KEY`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..retry import with_retries


def extract_media_url(result: Optional[dict[str, Any]], keys: tuple[str, ...]) -> str:
    """Pull a media URL out of a fal result, trying several common shapes.

    fal models return the output under different keys (``video``, ``audio``, …)
    and as a dict (``{"url": ...}``), a bare string, or a one-element list.
    """
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


class FalSession:
    """Lazily-loaded fal SDK with credentialed, retrying upload + subscribe."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._sdk = None  # lazily imported

    def _ensure_sdk(self):
        if self._sdk is None:
            if not os.environ.get("FAL_KEY"):
                raise RuntimeError(
                    "fal credentials missing. Set FAL_KEY in your .env file "
                    "(get one at https://fal.ai/dashboard/keys)."
                )
            import fal_client  # imported lazily

            self._sdk = fal_client
        return self._sdk

    def _retry(self, fn, description: str):
        return with_retries(
            fn,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=description,
        )

    def upload(self, path: Path, *, description: str) -> str:
        """Upload a local file and return its hosted URL."""
        sdk = self._ensure_sdk()
        return self._retry(lambda: sdk.upload_file(path), description)

    def subscribe(
        self, model_id: str, arguments: dict[str, Any], *, description: str
    ) -> dict[str, Any]:
        """Submit a job and block until it finishes, returning the result dict."""
        sdk = self._ensure_sdk()
        return self._retry(lambda: sdk.subscribe(model_id, arguments) or {}, description)
