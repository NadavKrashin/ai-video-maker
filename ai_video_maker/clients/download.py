"""Stream a remote file to disk, atomically and with retry."""
from __future__ import annotations

from pathlib import Path

import requests

from ..retry import with_retries

_CHUNK = 1 << 16  # 64 KiB


def download_file(
    url: str, dst: Path, *, max_retries: int, base_delay: float, description: str
) -> None:
    """Download `url` to `dst`, writing to a ``.part`` temp file then renaming.

    The atomic rename means a half-written file is never left at `dst` (so a
    later resume won't mistake a truncated download for a finished one).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    def _call() -> None:
        tmp = dst.with_suffix(dst.suffix + ".part")
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=_CHUNK):
                    if chunk:
                        fh.write(chunk)
        tmp.replace(dst)

    with_retries(
        _call, max_retries=max_retries, base_delay=base_delay, description=description
    )
