"""Storyboard data models (Mode B). Human-editable JSON maps onto these."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError

from .errors import StoryboardError


class FramePerson(BaseModel):
    """One cast member, pinned to where they stand in one frame.

    This is ground truth supplied by a human (the panel's "who is in this
    photo" tagger), and it exists because identity is the thing the planner
    gets wrong on its own: it decides from the pixels whether the child in
    frame 7 is the child from frame 6, and when it guesses wrong the video
    model morphs one person into the other.

    ``x``/``y`` are fractions of the frame (0 = left/top, 1 = right/bottom).
    Only ``x`` really matters — it gives the left-to-right order the video
    model maps region-to-region, which is what arrangement-swap detection
    needs — but ``y`` is kept so a marker can be drawn back where it was put.
    """

    id: str          # a Character.id from the movie's cast
    x: float = 0.5
    y: float = 0.5


class Frame(BaseModel):
    id: str
    description: str
    image_prompt: str
    negative_prompt: str = ""
    output_path: str
    # Who is visible in this frame, tagged by hand (or proposed by `tag` and
    # then corrected). Empty means "nobody has said", and the planner falls
    # back to working it out from the image as it always did.
    people: list[FramePerson] = Field(default_factory=list)
    # Image-based projects: the input image this frame was styled from
    # (workspace-relative). Lets the pipeline detect that a styled file no
    # longer matches its source (inputs swapped/reordered) and re-style it.
    source_path: str = ""
    # SHA-1 of the styled image when the storyboard was saved. Staleness of
    # transitions/clips is decided by comparing content hashes, because file
    # mtimes lie: a cloud-sync client re-materializing untouched files once
    # made every frame look "changed" and wiped a project's rendered clips.
    styled_hash: str = ""
    # Optional per-frame styling guidance, appended to the shared style prompt
    # for THIS frame only (hand-edited between steps, carried across
    # reconciles). For photos the global prompt reads wrong: it is written for
    # portraits of people ("the subjects are the picture", "keep the people at
    # least as large as in the source"), which on a scene photo whose only
    # people are a tiny reflection reads as an instruction to promote them to
    # the foreground — a real frame came back with two invented faces standing
    # on a doorstep. Editing this does NOT re-style on its own: styling resume
    # is existence-based, so spend the credit deliberately with
    # `storyboard --restyle-frame <name>`.
    style_note: str = ""
    # Fingerprint of the cast face references this frame was STYLED with
    # (see reference_fingerprint). Comparing it against the references that
    # apply now is what makes "this frame predates your cast faces" visible
    # instead of something you have to remember.
    #
    # An EMPTY stamp means "styled from no references at all", which is true
    # of every frame styled before this existed and of every untagged
    # project — and it is deliberately NOT reported as outdated. Re-styling
    # spends image credits, so a project that was fine yesterday must not
    # wake up demanding money today; the stamp only speaks once a frame has
    # actually been styled WITH references and those references have since
    # moved.
    styled_refs: str = ""


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
    # Which styled frame this person's canonical face reference is cut from
    # (a Frame.output_path). Empty means "let the pipeline choose" — it takes
    # the frame with the fewest people in it, where the face is largest. Set
    # it by hand (or in the panel's Cast editor) when that pick is a bad
    # likeness: the reference is what later stylings and character-aware
    # video models are told this person looks like, so it is worth one
    # glance. A frame this person is not tagged in is ignored rather than
    # obeyed — see media/faces.pick_reference_frame.
    reference_frame: str = ""


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
    # Fingerprint of the identity facts this prompt was PLANNED from: who was
    # tagged in its two frames, in left-to-right order, and what each of them
    # was called at the time. Comparing it against the storyboard as it
    # stands now is what makes "this prompt predates your tags" visible
    # instead of something you have to remember (see identity_fingerprint /
    # outdated_identity_plans). Empty on anything planned before this
    # existed, which reads as "planned from no tags at all" — true, and
    # exactly what makes an untagged project stay quiet.
    planned_identity: str = ""


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


def identity_fingerprint(
    storyboard: "Storyboard", transition: "Transition"
) -> str:
    """What the identity facts for this pair look like RIGHT NOW.

    Covers exactly the inputs a planner uses to decide who is in the clip and
    what to call them: the people tagged in each of the two frames, in
    left-to-right order, and the epithet each of them currently carries. A
    prompt planned when this value was different is a prompt that does not
    know about your latest tag or renamed cast member.

    An untagged frame contributes ``?`` rather than an empty list, so "nobody
    has said who is here" and "this photo has nobody in it" stay different
    facts — and so a plan made before any tagging has a stable, matchable
    value instead of a special case.
    """
    epithets = {c.id: c.epithet for c in storyboard.characters}
    frames = {f.output_path: f for f in storyboard.frames}
    parts: list[str] = []
    for path in (transition.start_frame, transition.end_frame):
        frame = frames.get(path)
        if frame is None or not frame.people:
            parts.append("?")
            continue
        ordered = sorted(frame.people, key=lambda p: p.x)
        parts.append(
            ",".join(f"{p.id}={epithets.get(p.id, '')}" for p in ordered)
        )
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


# What `identity_fingerprint` returns for a pair whose frames are both
# untagged. Anything planned before plans recorded their identity inputs is
# treated as having been planned from this — which is true, and is what keeps
# a project that has never been tagged from reporting every prompt as stale.
def _untagged_fingerprint() -> str:
    return hashlib.sha1("?|?".encode("utf-8")).hexdigest()[:12]


def reference_fingerprint(used: list[tuple[str, str]]) -> str:
    """Identity of one styling's reference set: ``(cast id, content hash)``.

    Content hashes, never mtimes — the same rule the styled frames follow,
    and for the same reason: a cloud-sync client once bumped every mtime and
    the old heuristic wiped a project's clips. Here the cost of a lying mtime
    would be quieter but just as unwelcome, a whole album reporting itself
    stale and inviting the user to re-buy it.

    Order-independent, because which references were attached is what shapes
    the output; the order they were listed in is not.
    """
    material = "|".join(f"{cid}={digest}" for cid, digest in sorted(used))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:12]


def outdated_reference_frames(
    storyboard: "Storyboard", applicable: dict[str, str]
) -> list[str]:
    """Styled frames whose cast references have moved since they were styled.

    ``applicable`` maps a frame's output_path to the fingerprint its
    references would produce TODAY. A frame is listed only when it carries a
    non-empty stamp that disagrees — see ``Frame.styled_refs`` for why an
    empty stamp stays silent.

    Nothing is fixed automatically: re-styling costs image credits, so this
    is a badge and a named ``--restyle-frame`` away, exactly like every other
    staleness in this pipeline.
    """
    return [
        frame.output_path
        for frame in storyboard.frames
        if frame.styled_refs
        and applicable.get(frame.output_path, frame.styled_refs) != frame.styled_refs
    ]


def outdated_identity_plans(storyboard: "Storyboard") -> list[str]:
    """Ids of transitions whose prompt predates the current tags/cast.

    This is the answer to "which prompts don't know what I just told the
    app?". Tagging a photo or renaming a cast member changes nothing on its
    own — both are plan-time inputs — so without this the only way to apply
    them was to re-plan everything and hope, or to remember which pairs the
    person appeared in.
    """
    blank = _untagged_fingerprint()
    return [
        t.id for t in storyboard.transitions
        if identity_fingerprint(storyboard, t) != (t.planned_identity or blank)
    ]


def tagged_people(storyboard: "Storyboard") -> dict[str, list[str]]:
    """Frame output_path -> the epithets standing in it, LEFT TO RIGHT.

    Resolves each frame's tagged cast ids through the movie's cast and
    orders them by their marker's x position, which is the order the video
    model actually works in. Frames nobody has tagged are absent (not
    empty): "no answer" and "nobody is in this photo" have to stay
    distinguishable, or an untagged movie would look like a cast of nobody.

    Unknown ids are dropped — a cast member can be deleted after tagging,
    and a dangling id must not become an epithet-shaped hole in the prompt.
    """
    by_id = {c.id: c.epithet for c in storyboard.characters if c.epithet.strip()}
    out: dict[str, list[str]] = {}
    for frame in storyboard.frames:
        if not frame.people:
            continue
        ordered = sorted(frame.people, key=lambda p: p.x)
        epithets = [by_id[p.id] for p in ordered if p.id in by_id]
        if epithets:
            out[frame.output_path] = epithets
    return out


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
