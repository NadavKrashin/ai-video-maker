"""Cutting a cast member's face out of a styled frame.

The pipeline knows two things it never used together: which styled frames a
person appears in (``Frame.people``, tagged by hand in the panel) and where
their face sits in each one (``FramePerson.x``/``y``, recorded as the CENTRE
OF THE FACE by the same tagger). That is everything needed to cut a canonical
portrait of each character out of work already paid for — no new vision call,
no new UI, no image credits.

Those portraits are the movie's identity anchors, and they serve both halves
of the character-consistency problem:

* styling a frame passes the references for whoever is in it, so the image
  model matches faces it has already drawn instead of re-inventing a cartoon
  of that person from each photo independently;
* rendering a clip can pass them to a video model that accepts character
  elements, so a face survives the transit between two frames.

Everything here is local, free and deterministic, which is why it may be
rebuilt whenever it is out of date — unlike every step that spends money, it
needs no confirmation gate.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from PIL import Image

from ..logging_setup import logger

# Side of the crop for a frame with ONE tagged person, as a fraction of the
# frame's height. A face plus hair, neck and a little shoulder — enough for
# the model to read bone structure, hairline and glasses, which are the
# features the style prompt already treats as identity.
DEFAULT_CROP_FRACTION = 0.5

# Never cut a window smaller than this fraction of the height: past it the
# crop is more JPEG-ish mush than face, and a bad reference is worse than
# none (it teaches the wrong face with full confidence).
_MIN_CROP_FRACTION = 0.16

# References are saved at most this big. They are looked at, not printed, and
# several of them ride along with one API call — a full-height crop of a
# 1080p frame would trade real request weight for detail nothing reads.
DEFAULT_REFERENCE_EDGE = 512


def reference_crop_box(
    width: int,
    height: int,
    x: float,
    y: float,
    people_in_frame: int = 1,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
) -> tuple[int, int, int, int]:
    """The (left, top, right, bottom) box to cut around one tagged face.

    ``x``/``y`` are fractions of the frame, as ``FramePerson`` stores them.

    The size of a face is the one thing the tagger does NOT record, so it is
    estimated from how many people share the frame: a solo portrait fills a
    lot of the height, a group photo divides it up. Dividing by ``sqrt(n)``
    matches how faces actually shrink as a group grows (people spread across
    two dimensions, not one) and degrades gently — a wrong guess here costs
    some context around the face, never the face itself.

    The box is clamped to the image, and clamping SLIDES it rather than
    shrinking it, so a face tagged near an edge still yields a full-size
    square instead of a thin strip.
    """
    fraction = max(_MIN_CROP_FRACTION, crop_fraction / math.sqrt(max(1, people_in_frame)))
    side = int(round(min(height * fraction, float(min(width, height)))))
    side = max(1, side)

    cx = int(round(min(max(x, 0.0), 1.0) * width))
    cy = int(round(min(max(y, 0.0), 1.0) * height))
    half = side // 2

    left = min(max(cx - half, 0), max(0, width - side))
    top = min(max(cy - half, 0), max(0, height - side))
    return left, top, left + side, top + side


def write_face_reference(
    src: Path,
    dst: Path,
    x: float,
    y: float,
    people_in_frame: int = 1,
    crop_fraction: float = DEFAULT_CROP_FRACTION,
    max_edge: int = DEFAULT_REFERENCE_EDGE,
) -> bool:
    """Cut one face out of ``src`` and save it to ``dst``. True when written.

    Never raises: a reference is an optimisation, and a project whose frames
    cannot be read must still style, plan and render exactly as it did
    before. An unreadable frame is a warning and a ``False``.
    """
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            box = reference_crop_box(
                im.width, im.height, x, y, people_in_frame, crop_fraction
            )
            face = im.crop(box)
            if max(face.size) > max_edge:
                face.thumbnail((max_edge, max_edge), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            face.save(dst, format="PNG")
        return True
    except Exception as exc:  # noqa: BLE001 - a reference is never load-bearing
        logger.warning("Could not cut a face reference from %s: %s", src.name, exc)
        return False


def references_for_frame(
    people: list[tuple[str, float]],
    available: set[str],
    appearances: dict[str, int],
    limit: int,
) -> list[str]:
    """Which cast members' faces ride along when this frame is styled.

    ``people`` is ``(cast id, x)`` for everyone tagged in the frame,
    ``available`` the ids that actually have a reference on disk, and
    ``appearances`` how many frames of the whole movie each id appears in.

    The cap matters: references compete with the photograph for the model's
    attention and add real weight to the request, so a nine-person frame must
    not attach nine portraits. When there are more candidates than the cap,
    the ones kept are those who appear in the MOST frames — consistency is
    worth most for the people the audience sees again and again, and least
    for someone who passes through once. Ties break left-to-right, then by
    id, so the choice is stable across runs (an unstable choice would make
    every frame look perpetually out of date).

    This is deliberately the ONE place that decides, called both when styling
    a frame and when asking whether a styled frame is out of date. Two
    implementations of "which references apply here" would drift, and every
    frame would then be reported stale forever.
    """
    candidates = [(cid, x) for cid, x in people if cid in available]
    candidates.sort(key=lambda item: (-appearances.get(item[0], 0), item[1], item[0]))
    return [cid for cid, _ in candidates[: max(0, limit)]]


def elements_for_clip(
    start_people: list[tuple[str, float]],
    end_people: list[tuple[str, float]],
    available: set[str],
    appearances: dict[str, int],
    limit: int,
    allow_partial: bool = False,
) -> list[str]:
    """Which cast members' faces to pin through one clip, as ids.

    Only people tagged in BOTH frames are candidates: an element exists to
    hold one identity steady across the transit, and somebody who leaves or
    arrives has no continuous identity to hold. (They are also exactly the
    people the pipeline stages an exit and an entrance for, precisely so the
    model does NOT try to carry them across.)

    ``limit`` is how many the endpoint accepts. When more people qualify than
    fit, ``allow_partial`` decides between two honest answers, and this is a
    genuinely open question that only a real A/B settles:

    * False (the default) — send none. Pinning one face crisply while three
      others drift can read worse than four drifting together, because the
      eye is drawn to the mismatch.
    * True — send as many as fit, the most-recurring people first, on the
      grounds that a rescued protagonist is worth more than a uniform blur.
    """
    if limit <= 0:
        return []
    ends = {cid for cid, _ in end_people}
    shared = [
        (cid, x) for cid, x in start_people if cid in ends and cid in available
    ]
    if not shared:
        return []
    if len(shared) > limit and not allow_partial:
        return []
    shared.sort(key=lambda item: (-appearances.get(item[0], 0), item[1], item[0]))
    return [cid for cid, _ in shared[:limit]]


def annotate_prompt_with_elements(
    prompt: str, epithets: list[str], template: str
) -> str:
    """Tag each element's epithet in the prompt with its reference token.

    Kling refers to elements positionally ("@Element1"), so a prompt that
    never mentions the token leaves the model to guess which person on screen
    a reference portrait is. This inserts it after the FIRST mention of that
    person's epithet — the prompts already name everyone by a consistent
    appearance phrase, which is what makes the substitution reliable.

    An epithet the prompt does not contain is skipped rather than appended
    somewhere: a token pointing at nobody is worse than no token, and the
    camera-transition prompts deliberately name no one at all.

    ``template`` is a config string with ``{index}`` (1-based). An empty
    template returns the prompt untouched.
    """
    if not template or not epithets:
        return prompt
    out = prompt
    for i, epithet in enumerate(epithets, start=1):
        token = template.format(index=i)
        if not epithet or epithet not in out or token in out:
            continue
        out = out.replace(epithet, f"{epithet} ({token})", 1)
    return out


def pick_reference_frame(
    appearances: list[tuple[str, int]],
    preferred: str = "",
) -> Optional[str]:
    """Which frame to cut a character's canonical portrait from.

    ``appearances`` is ``(frame output_path, how many people are tagged in
    that frame)`` in movie order. The pick is the frame with the FEWEST
    people in it — the one where this person's face is largest and least
    likely to be confused with a neighbour's — with the earliest such frame
    winning a tie, so the answer never wobbles between runs.

    ``preferred`` is the human's own choice (``Character.reference_frame``)
    and wins outright when the person really is tagged in it. A stale
    preference — a frame deleted, or one they were untagged from — falls back
    to the automatic pick rather than leaving the character with no
    reference at all.
    """
    if not appearances:
        return None
    if preferred:
        for path, _ in appearances:
            if path == preferred:
                return path
    # (crowd size, position) — min() is stable, but sorting on the index
    # explicitly is what makes "earliest wins the tie" a promise rather than
    # an implementation detail somebody could optimise away.
    return min(
        enumerate(appearances), key=lambda item: (item[1][1], item[0])
    )[1][0]
