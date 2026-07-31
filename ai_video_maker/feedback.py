"""Learning from rendered clips: feedback records and the lessons they distil.

The pipeline plans a motion prompt, Kling renders it, and only then can a
human see whether the plan was any good. Everything the project has learned
that way used to live in CLAUDE.md — rules a person copied into the prompt
files by hand ("no off-screen hands", "arrangement swaps need a crossing").
This module is that loop, automated: watch a clip, say what went wrong, and
the note is distilled into a short reusable RULE that rides along with every
future planning call.

Two stores, deliberately different in scope:

* :class:`FeedbackStore` — per project (``projects/<name>/feedback.json``).
  The raw notes, each pinned to the transition and the exact motion prompt
  that produced the clip. This is the evidence; it is never rewritten.
* :class:`LessonStore` — ONE store for the whole studio
  (``projects/_learning/lessons.json``). A lesson learned on one order is
  worth nothing if the next order has to relearn it, so lessons are global.
  ``_learning`` lives under ``projects/`` because that directory is the
  user-data tree (shared between the dev and production checkouts, never
  touched by a deploy); it is a RESERVED name — see workspace.py — so it can
  never be mistaken for a movie.

Lessons are text, and text is all they are: they are appended to the
planner's system prompt (scope ``motion``) or to the style prompt (scope
``style``). Nothing here edits code or config, so a bad lesson is undone by
switching it off in the panel, never by a rollback.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field

from .logging_setup import logger

# Which stage of the pipeline a lesson steers.
SCOPE_MOTION = "motion"   # the transition planner (what happens in a clip)
SCOPE_STYLE = "style"     # image styling (how a frame looks)
SCOPES = (SCOPE_MOTION, SCOPE_STYLE)

VERDICTS = ("bad", "good")

# A lesson is a rule, not an essay: long ones crowd out the prompt they are
# meant to sharpen (the planner's word budget is already tight).
MAX_LESSON_CHARS = 400
MAX_NOTE_CHARS = 4000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_id() -> str:
    return uuid.uuid4().hex[:8]


class Lesson(BaseModel):
    """One rule learned from a real render, injected into future prompts."""

    id: str = Field(default_factory=_short_id)
    text: str
    # Which prompt it joins: the transition planner or the image styler.
    scope: str = SCOPE_MOTION
    # Off = kept for the record but not sent to the model. A lesson that
    # turns out to be wrong is switched off, never silently deleted, so the
    # history of what was tried survives.
    active: bool = True
    created_at: str = Field(default_factory=_now)
    # "feedback" (distilled from a clip) or "manual" (typed in the panel).
    origin: str = "feedback"
    # Where it came from: {"project", "transition", "verdict", "feedback_id"}.
    source: dict[str, Any] = Field(default_factory=dict)


class FeedbackEntry(BaseModel):
    """One human judgement of one rendered clip."""

    id: str = Field(default_factory=_short_id)
    created_at: str = Field(default_factory=_now)
    project: str = ""
    transition_id: str = ""
    verdict: str = "bad"
    note: str = ""
    # The plan the clip was actually rendered from, copied at feedback time:
    # the storyboard can be edited afterwards, and a note about "this prompt"
    # is worthless once nobody can tell which prompt was meant.
    motion_prompt: str = ""
    duration: int = 0
    # Lessons distilled from this note (empty when learning was skipped or
    # the distillation call failed).
    lesson_ids: list[str] = Field(default_factory=list)


class _JsonListStore:
    """Shared plumbing: a lock-guarded list of models in one JSON file."""

    key = "items"
    model: type[BaseModel] = Lesson

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> list[Any]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); treating it as empty.",
                           self.path, exc)
            return []
        raw = data.get(self.key) if isinstance(data, dict) else data
        if not isinstance(raw, list):
            return []
        out = []
        for item in raw:
            try:
                out.append(self.model(**item))
            except Exception:  # noqa: BLE001 - one bad row can't hide the rest
                continue
        return out

    def _write(self, items: Iterable[Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {self.key: [i.model_dump() for i in items]},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )


class LessonStore(_JsonListStore):
    """The studio's learned rules — global, hand-editable, prompt-bound."""

    key = "lessons"
    model = Lesson

    def all(self) -> list[Lesson]:
        with self._lock:
            return self._read()

    def active(self, scope: str = SCOPE_MOTION) -> list[Lesson]:
        return [l for l in self.all() if l.active and l.scope == scope]

    def add(
        self,
        text: str,
        scope: str = SCOPE_MOTION,
        *,
        origin: str = "manual",
        source: Optional[dict[str, Any]] = None,
    ) -> Lesson:
        """Append a lesson. Blank text and unknown scopes are refused."""
        cleaned = " ".join((text or "").split())[:MAX_LESSON_CHARS]
        if not cleaned:
            raise ValueError("A lesson needs some text.")
        if scope not in SCOPES:
            raise ValueError(f"Unknown lesson scope: {scope!r}")
        lesson = Lesson(
            text=cleaned, scope=scope, origin=origin, source=source or {}
        )
        with self._lock:
            items = self._read()
            items.append(lesson)
            self._write(items)
        return lesson

    def update(
        self,
        lesson_id: str,
        *,
        text: Optional[str] = None,
        active: Optional[bool] = None,
        scope: Optional[str] = None,
    ) -> Optional[Lesson]:
        """Edit one lesson in place; returns it, or None when there is no such id."""
        if scope is not None and scope not in SCOPES:
            raise ValueError(f"Unknown lesson scope: {scope!r}")
        with self._lock:
            items = self._read()
            for lesson in items:
                if lesson.id != lesson_id:
                    continue
                if text is not None:
                    cleaned = " ".join(text.split())[:MAX_LESSON_CHARS]
                    if not cleaned:
                        raise ValueError("A lesson needs some text.")
                    lesson.text = cleaned
                if active is not None:
                    lesson.active = bool(active)
                if scope is not None:
                    lesson.scope = scope
                self._write(items)
                return lesson
        return None

    def remove(self, lesson_id: str) -> bool:
        with self._lock:
            items = self._read()
            keep = [l for l in items if l.id != lesson_id]
            if len(keep) == len(items):
                return False
            self._write(keep)
        return True


class FeedbackStore(_JsonListStore):
    """One project's raw feedback notes (the evidence behind its lessons)."""

    key = "feedback"
    model = FeedbackEntry

    def all(self) -> list[FeedbackEntry]:
        with self._lock:
            return self._read()

    def add(self, entry: FeedbackEntry) -> FeedbackEntry:
        with self._lock:
            items = self._read()
            items.append(entry)
            self._write(items)
        return entry

    def summary(self) -> dict[str, Any]:
        """What the panel shows next to a project: counts and per-clip verdicts."""
        entries = self.all()
        by_transition: dict[str, str] = {}
        for entry in entries:  # later feedback wins
            if entry.transition_id:
                by_transition[entry.transition_id] = entry.verdict
        return {
            "count": len(entries),
            "good": sum(1 for e in entries if e.verdict == "good"),
            "bad": sum(1 for e in entries if e.verdict == "bad"),
            "lessons": sum(len(e.lesson_ids) for e in entries),
            "by_transition": by_transition,
        }


def lesson_prompt_block(lessons: list[str], scope: str = SCOPE_MOTION) -> str:
    """Render lessons as the prompt section that carries them to the model.

    Empty in, empty out — a studio with no lessons yet must add NOTHING to
    the prompt, so the planner behaves exactly as it did before this feature
    existed.
    """
    rules = [" ".join(l.split()) for l in lessons if l and l.strip()]
    if not rules:
        return ""
    if scope == SCOPE_STYLE:
        header = (
            "\n\nLESSONS FROM EARLIER STYLED FRAMES (each one comes from a "
            "real image that came back wrong; they are corrections to the "
            "instructions above and win over them):\n"
        )
    else:
        header = (
            "\n\nLESSONS FROM EARLIER RENDERS (each one comes from a real "
            "clip that came back wrong or right; they are corrections to the "
            "instructions above, so where they disagree with anything earlier, "
            "the lesson wins):\n"
        )
    return header + "\n".join(f"- {rule}" for rule in rules)
