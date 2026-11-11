"""Keep the take a re-render replaces, so the two can be compared — and swapped.

Regenerating a clip overwrote ``clips/<id>.mp4`` in place. If the new take
was worse (and a re-roll of a hard pair often is), the one you already paid
for was gone and the only way back was to pay again and hope. Now the take
being replaced is copied to ``clips_previous/<id>.mp4`` first, with a sidecar
recording what it was: the prompt and length it was rendered from, and the
per-clip state that travels with the file (its SFX, its fade, its model).

Exactly ONE previous take is kept per clip. This is an undo for the last
re-render, not a version history: a folder of every attempt would grow by a
35 MB file per click, and "which of these five was the good one" is not a
question a filename answers.

Restoring SWAPS the two files rather than moving one over the other, so a
restore is itself undoable — nothing in here ever deletes a rendered clip.
That is the same rule as the rest of the pipeline: clips are paid for.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from .logging_setup import logger
from .state import StateStore

# The state entries that describe the clip FILE and so must travel with it.
# `stale:` is deliberately absent: it describes the file's relationship to
# the PLAN, and is recomputed on restore from what the take was rendered from.
_TRAVELLING_STATE = ("clip", "sfx", "fade")


def previous_path(previous_dir: Path, clip_name: str) -> Path:
    return previous_dir / clip_name


def _sidecar(previous_dir: Path, clip_name: str) -> Path:
    return previous_dir / (Path(clip_name).stem + ".json")


def _state_snapshot(state: StateStore, clip_name: str) -> dict[str, Any]:
    snap: dict[str, Any] = {}
    for kind in _TRAVELLING_STATE:
        entry = state.get(f"{kind}:{clip_name}")
        if entry:
            snap[kind] = entry
    return snap


def _apply_state(state: StateStore, clip_name: str, snap: dict[str, Any]) -> None:
    for kind in _TRAVELLING_STATE:
        key = f"{kind}:{clip_name}"
        entry = dict(snap.get(kind) or {})
        if not entry:
            state.clear(key)
            continue
        status = str(entry.pop("status", "done"))
        entry.pop("updated_at", None)
        state.set(key, status, **entry)


def read_info(previous_dir: Path, clip_name: str) -> Optional[dict[str, Any]]:
    """What is known about a clip's previous take, or None if there is none."""
    if not previous_path(previous_dir, clip_name).exists():
        return None
    try:
        data = json.loads(_sidecar(previous_dir, clip_name).read_text("utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def keep_previous_take(
    clip: Path, previous_dir: Path, state: StateStore
) -> bool:
    """Copy `clip` aside before it is re-rendered. Returns True when kept.

    A COPY, not a move: the render that follows may fail, and then the
    current clip has to still be there. Best-effort by design — a full disk
    must cost the undo, never the render the user asked for.
    """
    if not clip.exists():
        return False
    try:
        previous_dir.mkdir(parents=True, exist_ok=True)
        rendered = state.get(f"clip:{clip.name}") or {}
        shutil.copy2(clip, previous_path(previous_dir, clip.name))
        _sidecar(previous_dir, clip.name).write_text(json.dumps({
            "kept_at": time.time(),
            "prompt": rendered.get("prompt"),
            "duration": rendered.get("duration"),
            "model": rendered.get("model"),
            "state": _state_snapshot(state, clip.name),
        }, indent=2), encoding="utf-8")
        return True
    except OSError as exc:
        logger.warning(
            "Could not keep the previous take of %s (%s); rendering anyway.",
            clip.name, exc,
        )
        return False


def restore_previous_take(
    clip: Path,
    previous_dir: Path,
    state: StateStore,
    *,
    planned_prompt: str,
    planned_duration: int,
) -> dict[str, Any]:
    """Swap `clip` with its previous take. Raises FileNotFoundError if none.

    Returns ``{"matches_plan": True | False | None}`` — whether the restored
    take was rendered from the prompt and length the storyboard holds NOW.
    ``None`` means the take predates that record and nobody can tell. A take
    known to differ is marked outdated (never re-rendered): putting an older
    render back is a legitimate choice, but the page must not then imply it
    matches a prompt it was not made from.
    """
    previous = previous_path(previous_dir, clip.name)
    if not previous.exists():
        raise FileNotFoundError(f"No previous take is kept for {clip.stem}.")
    info = read_info(previous_dir, clip.name) or {}

    current_info = {
        "kept_at": time.time(),
        "state": _state_snapshot(state, clip.name),
    }
    rendered = state.get(f"clip:{clip.name}") or {}
    for field in ("prompt", "duration", "model"):
        current_info[field] = rendered.get(field)

    # Three renames through a temp name: at no point does either take exist
    # only in memory, and an interruption leaves both files on disk.
    holding = previous.with_suffix(".swap.mp4")
    if clip.exists():
        os.replace(clip, holding)
    os.replace(previous, clip)
    if holding.exists():
        os.replace(holding, previous)
        _sidecar(previous_dir, clip.name).write_text(
            json.dumps(current_info, indent=2), encoding="utf-8"
        )
    else:
        _sidecar(previous_dir, clip.name).unlink(missing_ok=True)

    _apply_state(state, clip.name, info.get("state") or {})
    # The restored file is older than the movie built from the take it
    # replaces; without a fresh mtime "the final video is out of date" would
    # stay silent, and combine's own freshness check with it.
    now = time.time()
    os.utime(clip, (now, now))

    prompt, duration = info.get("prompt"), info.get("duration")
    if prompt is None or duration is None:
        matches: Optional[bool] = None
    else:
        matches = (
            str(prompt) == planned_prompt and int(duration) == int(planned_duration)
        )
    if matches is False:
        state.set(f"stale:{clip.name}", "outdated")
    elif matches is True:
        state.clear(f"stale:{clip.name}")
    return {"matches_plan": matches}
