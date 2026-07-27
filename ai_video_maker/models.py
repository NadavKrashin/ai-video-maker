"""Storyboard data models (Mode B). Human-editable JSON maps onto these."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from .errors import StoryboardError


class Frame(BaseModel):
    id: str
    description: str
    image_prompt: str
    negative_prompt: str = ""
    output_path: str
    # Image-based projects: the input image this frame was styled from
    # (workspace-relative). Lets the pipeline detect that a styled file no
    # longer matches its source (inputs swapped/reordered) and re-style it.
    source_path: str = ""
    # SHA-1 of the styled image when the storyboard was saved. Staleness of
    # transitions/clips is decided by comparing content hashes, because file
    # mtimes lie: a cloud-sync client re-materializing untouched files once
    # made every frame look "changed" and wiped a project's rendered clips.
    styled_hash: str = ""


class Character(BaseModel):
    """One recurring person (or animal/vehicle subject) in the movie's cast.

    ``epithet`` is the exact appearance-based phrase every motion prompt uses
    for this person ('the bald man in pink sunglasses'). The planner builds
    the cast on the first full plan and then reuses each epithet VERBATIM in
    every later planning call — including targeted re-plans that only see a
    few frames, which otherwise invent a fresh epithet for someone the rest
    of the movie names differently (the video model then can't tell it's the
    same person from clip to clip). Hand-editable like everything else in
    storyboard.json; edits apply to future planning, not to already-planned
    prompts.
    """

    id: str
    epithet: str
    notes: str = ""


class Transition(BaseModel):
    id: str
    start_frame: str
    end_frame: str
    motion_prompt: str
    duration: int = 5
    # Optional per-clip SFX/ambient guidance for the video->audio step. Empty
    # falls back to config.default_sfx_prompt.
    sound_prompt: str = ""
    output_path: str


class Storyboard(BaseModel):
    project_title: str
    style: str
    duration_per_clip: int = 5
    target_width: int = 1920
    target_height: int = 1080
    concept: str = ""
    scenes: list[str] = Field(default_factory=list)
    # Optional guidance that applies to EVERY clip: prepended to each
    # transition's motion prompt at render time and given to the planner as
    # context (hand-edited between steps). For facts that
    # hold across the whole movie — e.g. "Two different children appear
    # throughout: an older boy with glasses and a younger girl; they are
    # separate people, never blend one into the other." Keep it to a sentence
    # or two: it spends part of every clip's prompt budget.
    global_motion_prompt: str = ""
    # The movie's cast (see Character). Kept stable across storyboard runs;
    # deliberately NOT clip-defining: epithets are baked into motion prompts
    # at planning time, so editing the cast changes future plans only and
    # must not mark rendered clips stale.
    characters: list[Character] = Field(default_factory=list)
    frames: list[Frame]
    transitions: list[Transition] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Storyboard":
        if not path.exists():
            raise StoryboardError(f"Storyboard file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except json.JSONDecodeError as exc:
            raise StoryboardError(f"{path} is not valid JSON: {exc}") from exc
        except ValidationError as exc:
            raise StoryboardError(f"Invalid storyboard JSON ({path}):\n{exc}") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# Transition fields that shape the rendered video. `sound_prompt` is
# deliberately absent: it only feeds the audio step, so editing it must not
# invalidate a clip that is otherwise still correct.
_CLIP_DEFINING_FIELDS = ("motion_prompt", "duration", "start_frame", "end_frame")


def changed_transition_ids(
    old: Optional["Storyboard"], new: "Storyboard"
) -> list[str]:
    """Ids of transitions whose already-rendered clip no longer matches `new`.

    Hand-editing a motion prompt or duration invalidates the clip rendered
    from the old plan in exactly the way a re-plan does, so the panel's save
    can mark those clips outdated instead of leaving the edit invisible to
    everything but the browser tab that made it. Editing the GLOBAL motion
    prompt invalidates every transition, because it is prepended to each
    clip's prompt at render time (`_with_global_motion`).

    Transitions that are new in `new` are never listed: nothing was rendered
    from them, so there is nothing to invalidate.
    """
    if old is None:
        return []
    if old.global_motion_prompt != new.global_motion_prompt:
        return [t.id for t in new.transitions]
    previous = {t.id: t for t in old.transitions}
    return [
        t.id for t in new.transitions
        if (before := previous.get(t.id)) is not None
        and any(getattr(before, f) != getattr(t, f) for f in _CLIP_DEFINING_FIELDS)
    ]
