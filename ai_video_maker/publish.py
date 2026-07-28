"""The local record of what has been delivered to a customer's order folder.

`pipeline.py publish` uploads ``output/final_video.mp4`` into the project's
Cloudinary order folder as ``final_v1``, ``final_v2``, ... and appends what it
did to ``projects/<name>/published.json``. That file is the answer to three
questions the pipeline has to be able to answer without touching the network:

* has this movie been delivered at all, and where (status / panel / snapshot);
  each delivered version is also archived beside it as
  ``output/published/final_vN.mp4``, because ``output/final_video.mp4`` is
  REPLACED by the next combine — without the copy, the bytes a customer was
  actually sent exist only in Cloudinary;
* which version numbers this machine has already used — merged with what
  Cloudinary reports, so a listing hiccup can never make the next publish
  reuse a number;
* has the movie CHANGED since it was published (a re-combine after the last
  delivery), which is exactly when a v2 is wanted.

The "changed" test compares the final video's size and mtime, not a content
hash: `snapshot()` runs for every project on every panel refresh and hashing a
few hundred MB there would be paid on every poll.

Everything here is pure file I/O so it stays unit-testable; the upload itself
lives in ``clients/cloudinary_client.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def read_publications(path: Path) -> list[dict[str, Any]]:
    """Every recorded publication, oldest first ([] when absent/unreadable)."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    entries = data.get("publications") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def source_fingerprint(video: Path) -> dict[str, Any]:
    """Cheap identity of the movie file: its size and mtime.

    Used to tell "this is the movie that was published" from "it has been
    re-combined since" without hashing hundreds of megabytes on every status
    refresh.
    """
    if not video.is_file():
        return {"bytes": 0, "mtime": 0.0}
    stat = video.stat()
    return {"bytes": stat.st_size, "mtime": round(stat.st_mtime, 3)}


def record_publication(
    path: Path,
    *,
    order_folder: str,
    public_id: str,
    url: str,
    version: int,
    video: Path,
    local_file: str = "",
) -> dict[str, Any]:
    """Append one publication to published.json and return the entry.

    Append-only, mirroring the delivery itself: an earlier version's record is
    never rewritten, so the file is the full delivery history of the movie.
    `local_file` is the project-relative archive copy of exactly what was
    delivered ("" when archiving is off or the copy failed).
    """
    entry = {
        "version": version,
        "public_id": public_id,
        "url": url,
        "order_folder": order_folder,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_fingerprint(video),
        "local_file": local_file,
    }
    publications = read_publications(path) + [entry]
    path.write_text(
        json.dumps({"publications": publications}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return entry


def published_versions(path: Path) -> list[int]:
    """Version numbers this machine has already published."""
    return [
        int(e["version"]) for e in read_publications(path)
        if isinstance(e.get("version"), int)
    ]


def latest_publication(path: Path) -> Optional[dict[str, Any]]:
    """The highest-versioned publication, or None when nothing was published."""
    publications = read_publications(path)
    if not publications:
        return None
    return max(publications, key=lambda e: e.get("version", 0))


def publish_state(path: Path, video: Path) -> dict[str, Any]:
    """Delivery status for `snapshot()` — local only, never hits the network.

    ``changed_since`` is True when the final video on disk is no longer the
    one that was published (re-combined since), i.e. exactly when publishing
    another version is worth offering. ``versions`` lists every delivery,
    newest first, each saying whether its archive copy is still on disk —
    ``output/final_video.mp4`` is REPLACED by the next combine, so the archive
    is the only local copy of what an earlier version actually looked like.
    """
    publications = read_publications(path)
    if not publications:
        return {"count": 0, "latest": None, "versions": [], "changed_since": False}
    project_root = path.parent

    def _view(entry: dict[str, Any]) -> dict[str, Any]:
        local = str(entry.get("local_file", ""))
        return {
            "version": entry.get("version", 0),
            "public_id": entry.get("public_id", ""),
            "url": entry.get("url", ""),
            "published_at": entry.get("published_at", ""),
            "local_file": local,
            # False when archiving was off, the copy failed, or someone
            # cleaned it up later — the panel must not offer a dead download.
            "local_exists": bool(local) and (project_root / local).is_file(),
        }

    latest = max(publications, key=lambda e: e.get("version", 0))
    return {
        "count": len(publications),
        "latest": _view(latest),
        "versions": [
            _view(e)
            for e in sorted(
                publications, key=lambda e: e.get("version", 0), reverse=True
            )
        ],
        "changed_since": (
            video.is_file() and latest.get("source") != source_fingerprint(video)
        ),
    }
