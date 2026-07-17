"""Shared fal.ai access.

Both video (image-to-video) and audio (SFX + music) run on fal, so the SDK
loading, credential check, upload, and job submission live here once. The SDK is
imported lazily so --dry-run / --help work without `fal-client` or `FAL_KEY`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from ..config import Config
from ..logging_setup import logger
from ..retry import with_retries

# Queue polling cadence for submitted jobs. A Kling clip takes single-digit
# minutes; 10s polls are frequent enough without hammering the API, and the
# timeout is generous because a queued job that eventually finishes is money
# already spent — better to wait than to abandon and pay again.
_POLL_INTERVAL_SECONDS = 10.0
_POLL_TIMEOUT_SECONDS = 30 * 60.0


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
        """Submit a job and block until it finishes, returning the result dict.

        WARNING: only suitable for cheap jobs (audio). If the connection drops
        mid-wait, the retry wrapper resubmits the WHOLE job — the first one
        keeps running (and billing) server-side with no handle left to it.
        Expensive jobs (video clips) must use submit() + wait_for_result()
        so a disruption never re-buys work that is already underway.
        """
        sdk = self._ensure_sdk()
        return self._retry(lambda: sdk.subscribe(model_id, arguments) or {}, description)

    def submit(
        self, model_id: str, arguments: dict[str, Any], *, description: str
    ) -> str:
        """Enqueue a job and return its request_id immediately.

        The request_id is the receipt for money being spent: persist it before
        waiting, so a crash or dropped connection can recover the finished
        output later instead of paying for the render again.
        """
        sdk = self._ensure_sdk()
        handle = self._retry(
            lambda: sdk.submit(model_id, arguments), f"{description} (submit)"
        )
        return handle.request_id

    def wait_for_result(
        self, model_id: str, request_id: str, *, description: str
    ) -> dict[str, Any]:
        """Poll a queued job until it finishes and return its result dict.

        Unlike subscribe(), a connection error here only retries the *poll* —
        the job itself is never resubmitted. A failed job surfaces when
        result() raises (e.g. fal's content checker), which the retry wrapper
        classifies as permanent, so moderation errors propagate to the
        reword-recovery layer exactly as before.
        """
        sdk = self._ensure_sdk()
        deadline = time.monotonic() + _POLL_TIMEOUT_SECONDS
        logged_running = False
        while True:
            status = self._retry(
                lambda: sdk.status(model_id, request_id, with_logs=False),
                f"{description} (status)",
            )
            if isinstance(status, sdk.Completed):
                return self._retry(
                    lambda: sdk.result(model_id, request_id) or {},
                    f"{description} (result)",
                )
            if not logged_running:
                logger.info(
                    "%s: request %s is %s — polling every %.0fs",
                    description, request_id, type(status).__name__,
                    _POLL_INTERVAL_SECONDS,
                )
                logged_running = True
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{description}: fal request {request_id} still "
                    f"{type(status).__name__} after "
                    f"{int(_POLL_TIMEOUT_SECONDS)}s — the request_id is "
                    "persisted, so re-running render will try to recover it."
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
