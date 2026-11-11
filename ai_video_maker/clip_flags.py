"""A worklist of rendered clips a human has marked as needing a fix.

Deliberately NOT the feedback store, though both amount to "this clip is
bad". They answer different questions and have opposite shapes:

* ``feedback.py`` is EVIDENCE. Each note is pinned to the exact motion prompt
  that produced the clip, it is append-only because a record you can edit is
  not a record, it costs model calls (the reviewer watches the clip, the
  distiller turns the note into a studio-wide rule), and its verdicts are
  ``bad``/``good`` — there is no "never mind", because nothing was learned in
  reverse.
* This is a TO-DO LIST for one movie. You watch twenty-nine clips, six of
  them are wrong, and you need to know which six when you come back. It has
  to be one click, free, and above all UNMARKABLE — the whole point is
  ticking things off once they are fixed, which the evidence log must never
  let you do.

Keeping them separate is what lets each stay honest. Marking a clip teaches
the planner nothing (write feedback for that) and unmarking one erases no
history (there is none to erase).

Nothing here deletes or re-renders a clip: a mark is a note to a human, and
redoing a clip stays the explicit, named, paid action it is everywhere else
in this pipeline.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .logging_setup import logger

# A mark is a pointer, not an essay — the long-form account belongs in
# feedback, where it can actually teach something.
MAX_FLAG_NOTE_CHARS = 500


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClipFlag(BaseModel):
    """One clip a human has marked as needing a fix."""

    transition_id: str
    note: str = ""
    marked_at: str = Field(default_factory=_now)
    # The rendered clip's mtime when the mark was made. A mark is about the
    # clip you WATCHED, so once that clip is re-rendered the mark may be
    # about a file that no longer exists. Clearing it automatically would be
    # this pipeline silently deciding a human's note is finished with, which
    # is the one thing it never does — so the mark stays and is reported as
    # "re-rendered since you marked it" instead. Local mtime is enough here:
    # both sides are on this disk and the cost of being wrong is a label.
    clip_mtime: float = 0.0


class ClipFlagStore:
    """The marks for one project (``projects/<name>/clip_flags.json``)."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, ClipFlag]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A worklist is not worth sinking a run for. Say so and carry on
            # with none, exactly like a missing file.
            logger.warning("Could not read %s (%s); treating it as empty.",
                           self.path, exc)
            return {}
        out: dict[str, ClipFlag] = {}
        for item in raw.get("flags") or []:
            try:
                flag = ClipFlag(**item)
            except Exception:  # noqa: BLE001 - one bad row must not lose the rest
                continue
            if flag.transition_id:
                out[flag.transition_id] = flag
        return out

    def _write(self, flags: dict[str, ClipFlag]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "flags": [
                f.model_dump()
                for f in sorted(flags.values(), key=lambda f: f.transition_id)
            ]
        }
        self.path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def all(self) -> dict[str, ClipFlag]:
        with self._lock:
            return self._read()

    def mark(
        self, transition_id: str, note: str = "", clip_mtime: float = 0.0
    ) -> ClipFlag:
        """Mark a clip (or update its note). Idempotent by transition id."""
        with self._lock:
            flags = self._read()
            flag = ClipFlag(
                transition_id=transition_id,
                note=(note or "").strip()[:MAX_FLAG_NOTE_CHARS],
                clip_mtime=clip_mtime,
            )
            flags[transition_id] = flag
            self._write(flags)
        return flag

    def clear(self, transition_id: str) -> bool:
        """Unmark a clip. False when it wasn't marked (not an error)."""
        with self._lock:
            flags = self._read()
            if transition_id not in flags:
                return False
            del flags[transition_id]
            self._write(flags)
            return True

    def summary(self, clip_mtimes: Optional[dict[str, float]] = None) -> dict[str, Any]:
        """What status and the panel show: the marks, newest render noted.

        ``clip_mtimes`` maps transition id -> the rendered clip's current
        mtime. A mark whose clip is newer than the mark is reported with
        ``rerendered: True`` — still marked, but flagged as possibly already
        dealt with, so the human clears it rather than the code guessing.
        """
        mtimes = clip_mtimes or {}
        items = []
        for flag in sorted(self.all().values(), key=lambda f: f.transition_id):
            current = mtimes.get(flag.transition_id)
            items.append({
                "transition_id": flag.transition_id,
                "note": flag.note,
                "marked_at": flag.marked_at,
                "rerendered": bool(
                    current and flag.clip_mtime and current > flag.clip_mtime
                ),
            })
        return {
            "count": len(items),
            "ids": [i["transition_id"] for i in items],
            "flags": items,
            "rerendered": [i["transition_id"] for i in items if i["rerendered"]],
        }
