"""OpenAI client (isolated). Image generation/editing + storyboard planning."""
from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from ..config import Config
from ..constants import VALID_DURATIONS
from ..costs import KIND_IMAGE, KIND_TEXT
from ..feedback import (
    MAX_LESSON_CHARS,
    SCOPE_MOTION,
    SCOPES,
    lesson_prompt_block,
)
from ..media.images import (
    encode_image_data_url,
    normalize_image,
    prepare_image_for_upload,
)
from ..logging_setup import logger
from ..models import Character, Frame, Storyboard, Transition
from ..retry import is_output_moderation, with_retries, with_reword_recovery

T = TypeVar("T")

# Longest edge (px) for frames sent to the vision model. Low-detail vision uses
# ~512px anyway, so anything bigger is pure upload weight.
_VISION_MAX_EDGE = 768

# NOTE (user call, 2026-08-14): there is no longer a cap on how many clips
# may be long, and no mechanical word budget for motion prompts. Difficulty
# alone decides 5s vs 10s, and prompt LENGTH is governed by the beat rules in
# _MODE_A_SYSTEM ("say the one thing, then stop") rather than by counting
# words and rewriting what goes over. The counted cap existed because a real
# plan wrote 79-113 words for every 5s clip and an 84-word prompt rendered as
# a whip-pan blur; the replacement for that is a prompt that stays on point,
# not a truncation pass — condensing was reworking prompts that were fine and
# re-opening endings it had just closed. If sprawl comes back, the honest fix
# is sharper beat rules (or a VISIBLE warning), never a silent rewrite.

# Function words dropped when comparing two descriptions of the same person.
# Only articles/prepositions — NEVER adjectives: 'the tall man' and 'the short
# man' must not collapse into the same token set and be read as one person.
_PERSON_STOPWORDS = frozenset({
    "a", "an", "and", "at", "in", "is", "of", "on", "the", "with", "who",
})

# Phrases that show a motion prompt actually MOVES people between sides:
# somebody leaves and comes back, or one person visibly passes the other.
# Used to tell a real swap staging from hold-steady wording, which morphs
# swapped people into each other.
# One person physically passing another: a complete staging on its own.
_PASSING_MARKERS = (
    "crosses", "cross in front", "crossing", "in front of", "behind the",
    "behind her", "behind him", "swap", "swaps", "switch sides",
    "trade places", "trades places", "changes places", "around each other",
)
# Leaving the frame. HALF a staging: the clip is pinned to the end frame, so
# people who walk out have to come back before it ends.
_EXIT_MARKERS = (
    "past the camera", "out of frame", "off frame", "out of the frame",
    "offscreen", "off screen", "off-screen", "out of shot", "exits", "exit ",
    "leaves the frame", "walks out", "steps out",
)
# ...and the other half.
_RETURN_MARKERS = (
    "back in", "back into", "re-enter", "reenter", "re-entering",
    "re-enters", "enters from", "enter from", "steps in from",
    "step in from", "walks in from", "walk in from", "returns",
    "comes back", "come back", "gather again", "gathers again",
    "regroup", "regroups", "re-form", "reform", "re-forms", "reforms",
    "settle again", "stand again", "standing again",
)

# Restage a motion prompt for a pair whose people trade left-right positions.
# Detected in code from the planner's own start_order/end_order lists, because
# the planner keeps writing hold-steady prose for these pairs however the
# system prompt is worded — and hold-steady on a swap is exactly what makes
# the interpolator morph one person into the other.
_RESTAGE_SWAP_SYSTEM = (
    "You fix motion prompts for an image-to-video model that interpolates "
    "between two given frames. The pair you are given has a problem the "
    "prompt ignores: the SAME people appear in both frames but they have "
    "TRADED left-right positions. The model maps the left of the start frame "
    "onto the left of the end frame, so a prompt that leaves them standing "
    "where they are makes each person morph into the other — a real clip "
    "grew hair on a bald man this way. Rewrite the prompt so every person "
    "physically travels to their new side, staged one of two ways: (a) they "
    "walk forward past the camera and out of frame, then step back in one at "
    "a time — say which side each one comes back in from — or (b) one of "
    "them visibly crosses in front of or behind the other. "
    "AN EXIT IS ONLY HALF THE STAGING. The clip is pinned to the end frame, "
    "which SHOWS these people — so a prompt that walks someone out and stops "
    "there describes a clip that cannot exist, and the video model resolves "
    "it with a cut or a teleport. If anyone leaves the frame, the same "
    "prompt must bring them back, and the LAST thing it describes is the "
    "people standing where the end frame shows them. Never end on anyone "
    "being offscreen, leaving, or walking away. "
    "ACCOUNT FOR EVERYONE. You are given the full left-to-right list for "
    "both frames. Every person whose side changes gets named and moved; "
    "people who keep their place can be covered together in a few words "
    "('the other three stay put'). A rewrite that describes one person and "
    "silently forgets the rest of the group is wrong — the model then "
    "rearranges the forgotten ones however it likes, which is the morphing "
    "this is meant to prevent. When the group is large, move them as a group "
    "and name only who ends up somewhere different. "
    "Keep the rest of "
    "the original prompt's intent. Rules that still apply: describe ONLY the "
    "people, never the setting, the background or the location (the two "
    "frames already fix all of that); give every person a concrete physical "
    "verb (walk, step, turn, cross, crouch) and never an abstraction "
    "('squeeze closer', 'share a moment'); refer to each person by their "
    "visible-appearance epithet, never a name or a relationship word, and "
    "never a bare collective 'they' for an action belonging to one of them; "
    "say it in as few words as the staging needs and stop — no scene-setting, no inventory of poses; and END with them standing where the "
    "END frame shows them, each on their end-frame side — never end on "
    "people leaving. Present tense. Return ONLY the rewritten motion prompt, "
    "with no preamble or quotes."
)

# Finish a motion prompt whose last beat leaves someone out of the frame.
# Detected in code (`ends_offscreen`) for the same reason word caps are: the
# rule is in _MODE_A_SYSTEM and the planner writes them anyway — a full
# re-plan of a real 44-pair movie came back with the SAME 29 pairs ending on
# an exit, because nothing rejected them.
_COMPLETE_ENDING_SYSTEM = (
    "You fix motion prompts for an image-to-video model that interpolates "
    "between two given frames (the START and END of one clip). The prompt you "
    "are given has one specific fault: its LAST beat leaves somebody out of "
    "the frame — they walk out, exit past the camera, or end up offscreen — "
    "and nothing brings them back. "
    "THE CLIP IS PINNED TO THE END FRAME, which SHOWS those people standing "
    "in it. A prompt that empties the frame and stops there describes a clip "
    "that cannot exist, so the video model resolves it the only way it can: "
    "with a cut, or by teleporting everyone into place at the last instant. "
    "Rewrite the prompt so it ENDS on the people arranged as the end frame "
    "shows them. Keep the original's staging and its concrete physical verbs "
    "— if people leave the frame, that exit can stay, but the same prompt "
    "must then walk them back in (say which side each returns from) and the "
    "final clause is them standing where they end up. If the exit earns "
    "nothing, drop it instead and let them move directly to their end-frame "
    "positions. Never end on anyone leaving, walking away, or being "
    "offscreen. "
    "Rules that still apply: describe ONLY the people, never the setting, "
    "the background or the location (the two frames already fix all of "
    "that); give everyone who moves a concrete physical verb (walk, step, "
    "turn, cross, crouch) and never an abstraction ('squeeze closer', "
    "'share a moment'); refer to each person by their visible-appearance "
    "epithet, never a name or a relationship word, and never a bare "
    "collective 'they' for an action belonging to one person; keep the pace "
    "unhurried and describe an ARRIVAL rather than a whole route; stay "
    "to the point — as few words as the staging needs, then stop. Present tense. Return ONLY the rewritten "
    "motion prompt, with no preamble or quotes."
)

# Camera transitions: the staging for pairs that cannot be staged through
# the people (user call, 2026-08-13 — "I don't want any clip to have an
# offscreen" — widened 2026-08-14 after reviewing am-130826-pcfd's renders).
# A deliberate, narrow exception to the standing no-camera-moves rule, and
# the honest answer for the pairs that get it: a group of nine cannot walk
# out and walk back in inside 10 seconds, so the alternative is not a better
# staging, it is a cut — or the ghost-dissolve/mushed-bodies render that a
# many-mover choreography actually produced on a real order, while the pairs
# that shipped with this camera prompt were the cleanest in that movie. A
# camera move works where person-staging fails because it is a GLOBAL
# solution: the model never has to map person onto person — it leaves the
# first arrangement behind and arrives on the second. Deterministic on
# purpose — no extra call, and it can never come back still ending
# offscreen. Must contain no _EXIT_MARKERS (pinned by a test): "nobody
# leaves the frame" would ironically trip the very check this exists to
# satisfy. Reached two ways: the offscreen last resort at the bottom of
# _coerce_transition_plans, and the unstageable-pair gate at its top
# (is_unstageable_pair / camera_shift_prompt).
# Appended to the style prompt for ONE retry after an output-stage safety
# rejection (see _image_with_moderation_recovery). Ordinary customer photos
# get refused when they were taken among licensed merchandise: styling asks
# for a CARTOON of the whole scene, so a Mickey-printed popcorn bucket held
# at chest height is a request to DRAW Mickey Mouse — which image models
# refuse on copyright grounds, reported as an unhelpful 'other' category.
# Real case: two frames of order am-160826-vkxq-7, shot inside Disney gift
# shops, while the other 28 photos of the same children styled fine.
#
# It targets the OBJECTS only. The people are explicitly ring-fenced: the
# whole point of styling is likeness, and a "make it safe" rewrite that
# quietly generalises faces would trade one silent failure for a worse one.
_DEBRAND_INSTRUCTION = (
    "IMPORTANT — NO LICENSED CHARACTERS OR BRAND MARKS IN THE OUTPUT. Any "
    "copyrighted or trademarked character, mascot, logo or branded artwork "
    "visible in this photo — printed on cups, buckets, packaging, clothing, "
    "plush toys, signage, posters or shop displays — must be redrawn as a "
    "PLAIN, GENERIC equivalent: the same object in the same place, at the "
    "same size and in the same colours, but decorated with simple unbranded "
    "shapes, stripes or blank surfaces instead. Do not draw any recognisable "
    "licensed character anywhere in the frame, in the foreground or the "
    "background. This changes objects ONLY: the real people are untouched — "
    "their faces, hair, build, glasses and clothing stay exactly as "
    "described above, with the same likeness."
)

CAMERA_SHIFT_MOTION_PROMPT = (
    "The camera moves smoothly and steadily across to the new setting and "
    "settles there, holding on the people standing together exactly as the "
    "end frame shows them, everyone in full view."
)

# The same move for a pair whose END frame shows nobody: "holding on the
# people" would ask the model to keep people in a frame that has none.
CAMERA_SHIFT_TO_SCENE_PROMPT = (
    "The camera moves smoothly and steadily across to the new setting and "
    "settles there, framing the scene exactly as the end frame shows it."
)

# The move for a pair with faces filling BOTH frames (a group on each side).
# The generic travel invites the model to keep the people in shot for the
# whole journey — on a real render (0a_to_0b, walkway selfie -> hallway pose)
# the group turned and walked WITH the camera, and the transit collapsed into
# whip blur with faces smearing into each other: the disguised cut wearing
# the camera prompt as cover. Every camera clip that rendered clean had
# neutral ground mid-transit (an anchor to follow, an empty destination, a
# pan onto scenery), so this wording builds that in: the camera turns AWAY
# from the people first and travels across the surroundings — blur over
# walls reads as camera speed; blur over faces reads as people dissolving —
# and the arrival is a reveal. This is how a real editor cuts a whip-pan.
CAMERA_SHIFT_THROUGH_SCENERY_PROMPT = (
    "Everyone holds still as the camera turns away from them and glides "
    "across the surroundings, then comes to rest on the people exactly as "
    "the end frame shows them."
)


# Reword a prompt that OpenAI's safety filter wrongly flagged. The video pipeline
# never intends harmful content, so flags are typically benign wording the filter
# misreads (e.g. "shot", a body description). We ask the text model to keep the
# scene but make it unambiguously safe-for-work, then retry the image call.
_REWORD_SYSTEM = (
    "You rewrite text-to-image prompts that were incorrectly flagged by an "
    "automated content-safety filter. The prompts are for a wholesome, "
    "safe-for-work cinematic video and contain no harmful intent; the filter is "
    "over-triggering on innocent wording. Rewrite the prompt so it keeps the "
    "SAME scene, subjects, setting, composition, mood and visual style, but "
    "remove or rephrase anything the filter could misread as sexual, violent, "
    "gory, or otherwise unsafe. Make it clearly non-explicit and tasteful: "
    "describe people as fully clothed adults in a wholesome moment, replace "
    "loaded words (e.g. 'shot' -> 'scene', anatomical or suggestive terms -> "
    "neutral ones), and avoid anything that reads as nudity, sexual content, or "
    "graphic violence. Do not add captions or text overlays. Return ONLY the "
    "rewritten image prompt, with no preamble or quotes."
)

# Same idea for MOTION prompts (fal/Kling's content checker flags clip prompts
# the same way — e.g. innocent physical affection near a bed). The rewrite must
# stay a valid image-to-video motion prompt, not become an image prompt.
_REWORD_MOTION_SYSTEM = (
    "You rewrite motion prompts for an image-to-video model that were "
    "incorrectly flagged by an automated content-safety filter. The prompts "
    "describe wholesome, safe-for-work scenes (family videos, everyday "
    "moments) between two given photographs; the filter is over-triggering on "
    "innocent wording. Rewrite the prompt so it keeps the SAME people, "
    "actions and staging, but remove or rephrase anything the filter "
    "could misread as sexual, violent, or otherwise unsafe: make physical "
    "affection unmistakably innocent and familial (a warm hug, leaning on a "
    "shoulder), avoid mentioning beds/lying together/body parts, and replace "
    "loaded words (e.g. 'shot' -> 'scene'). BABIES AND CHILDREN trip the "
    "filter hardest: when a baby or child is in the scene, drop wording about "
    "their body being physically handled — lifted, carried, bounced, lowered, "
    "settled, laid down — and drop bed/crib/blanket mentions; say only that "
    "they share the moment instead (e.g. 'the woman with dark curly hair "
    "and the baby share a joyful moment together'), without describing the "
    "room or where any of it happens. The two input images "
    "already define the scene, so it is always safe to LOSE detail: make each "
    "rewrite noticeably simpler and more generic than the text you were "
    "given, deleting the risky details entirely rather than paraphrasing "
    "them. Identity anchors are NOT risky detail: keep each person's "
    "visible-appearance epithet ('the bald man in pink sunglasses', 'the "
    "younger man with brown hair') INCLUDING the part that tells two "
    "similar people apart — dropping 'in pink sunglasses' where the film "
    "has two bald men makes the model animate the wrong one — and keep any "
    "left/right position or direction wording, plus any exit/re-enter or "
    "trading-places staging — never "
    "collapse named individuals into a collective 'they' or swap their "
    "epithets for names or relationship words. Pace anchors are not risky "
    "detail either: keep wording that limits how far people travel ('the "
    "last few unhurried steps') and never widen it back out into a journey "
    "across a route — that makes the video model sprint them along the "
    "ground. Keep it one to three short "
    "sentences of continuous physical motion "
    "in present tense; no editing effects, no morphing between people. Return "
    "ONLY the rewritten motion prompt, with no preamble or quotes."
)

# --- Identifying the cast in each frame -------------------------------------- #
# A first pass at "who is in this photo", for a human to correct. Kept
# deliberately modest in tone: this is the one judgement the pipeline already
# knows the model is unreliable at, which is why the answer is a draft the
# panel shows for approval rather than something planning consumes directly.
_IDENTIFY_PEOPLE_SYSTEM = (
    "You match faces across a set of photographs of the same family or group, "
    "taken over months or years. You are given a cast — each person with a "
    "short appearance description — and a series of frames. For each frame, "
    "report which of those people are visible and where each one's face is. "
    "Identify people by FACE AND BUILD, never by clothing: the same person "
    "wears something different in every photograph, and two different people "
    "often wear similar things. Children change the most — the same child at "
    "two and at seven looks like two different people, while two siblings at "
    "the same age can look almost identical; use the frame order (the film "
    "runs in order, so ages progress) and any adults beside them as context. "
    "Be honest about uncertainty: a person you are unsure of should be LEFT "
    "OUT rather than guessed, because a human is about to check this and a "
    "missing tag is quicker to add than a wrong one is to notice. Never "
    "invent a person who is not in the given cast."
)

_IDENTIFY_PEOPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_index": {"type": "integer"},
                    "people": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "x": {"type": "number"},
                                "y": {"type": "number"},
                            },
                            "required": ["id", "x", "y"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["frame_index", "people"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["frames"],
    "additionalProperties": False,
}


def _coerce_frame_people(
    data: dict[str, Any], count: int, known_ids: set[str]
) -> list[list[dict[str, Any]]]:
    """Normalise an identify_people response into one clean list per frame.

    Everything questionable is dropped rather than repaired: unknown ids
    (a person the model invented cannot resolve to an epithet), duplicates
    within a frame (the same person cannot stand in two places), and
    out-of-range frame indices. Coordinates are clamped into the frame.
    """
    out: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for entry in (data.get("frames") or []):
        if not isinstance(entry, dict):
            continue
        try:
            index = int(entry.get("frame_index", 0)) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= index < count:
            continue
        seen: set[str] = set()
        for person in (entry.get("people") or []):
            if not isinstance(person, dict):
                continue
            pid = str(person.get("id") or "").strip()
            if pid not in known_ids or pid in seen:
                continue
            seen.add(pid)

            def _fraction(value: Any, fallback: float = 0.5) -> float:
                try:
                    return min(1.0, max(0.0, float(value)))
                except (TypeError, ValueError):
                    return fallback

            out[index].append({
                "id": pid,
                "x": _fraction(person.get("x")),
                "y": _fraction(person.get("y")),
            })
    return out


# --- Watching a rendered clip ----------------------------------------------- #
# The other half of the feedback loop. The planner writes a prompt blind and
# the user reports the result in words; this call actually LOOKS at what came
# back — stills sampled across the clip, next to the two key frames it was
# pinned to and the prompt that produced it — and says what happened. Its job
# is to be specific where a human is brief ("it looks weird"), and to propose
# a concrete replacement prompt the user can accept.
#
# The failure modes listed here are the ones this project has actually seen on
# real renders (they are the reason most rules in _MODE_A_SYSTEM exist), so
# naming them gives the reviewer a vocabulary instead of vague adjectives.
_CLIP_REVIEW_SYSTEM = (
    "You are reviewing one clip produced by an image-to-video model for a "
    "short film. You are shown: the START key frame, the END key frame, and "
    "stills sampled in time order from the rendered clip between them, plus "
    "the motion prompt that produced it and the clip's length. The model was "
    "required to begin exactly on the start frame and end exactly on the end "
    "frame. "
    "Say what ACTUALLY happens in the clip, in plain words, and judge it "
    "against the prompt. Look specifically for the failure modes this kind of "
    "model is prone to: a person changing position without ever walking "
    "there (sliding, drifting, floating, teleporting); one person morphing "
    "into a different-looking person, or a face/body changing identity "
    "mid-clip; people sprinting or skating because the prompt described a "
    "route too long for the clip's length; the shot collapsing into a "
    "disguised cut or whip-pan instead of a continuous action; a person being "
    "moved by hands that are not in frame; someone appearing or disappearing; "
    "hair, glasses or clothing changing; text appearing that was not in the "
    "frames; the clip not landing on the end frame; too many actions crammed "
    "in for the length; or nothing meaningful happening at all. "
    "Report only what you can SEE in the stills. If the clip looks right, say "
    "so — do not invent problems. Sampled stills cannot show smoothness or "
    "flicker, so never guess about those. "
    "Then write a REPLACEMENT motion prompt that would avoid the problems you "
    "found, obeying every rule you have been given for writing motion "
    "prompts, and say which clip length it should be rendered at. If the clip "
    "is fine, repeat the existing prompt unchanged and keep the same length. "
    "Return ONLY JSON of the shape "
    '{"observed": str, "problems": [str, ...], "matches_prompt": bool, '
    '"suggested_motion_prompt": str, "suggested_duration": int, '
    '"why": str}. "observed" is one or two sentences on what happens; '
    '"problems" names each fault in a few words (empty when there are none); '
    '"why" is one sentence on what the new prompt changes and why.'
)

_CLIP_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "observed": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "matches_prompt": {"type": "boolean"},
        "suggested_motion_prompt": {"type": "string"},
        "suggested_duration": {"type": "integer", "enum": sorted(VALID_DURATIONS)},
        "why": {"type": "string"},
    },
    "required": [
        "observed", "problems", "matches_prompt", "suggested_motion_prompt",
        "suggested_duration", "why",
    ],
    "additionalProperties": False,
}

# --- Learning from a rendered clip ----------------------------------------- #
# The feedback loop's one model call: a human watched a clip and said what was
# wrong with it; this turns that into a rule the planner can obey on a pair it
# has never seen. The hard part is GENERALITY — a note about "Michal" on
# "03_to_04" must come back as a rule about people and staging, because the
# next movie has neither. It is also allowed to return nothing: a note that
# teaches nothing beyond one clip ("this one is perfect") must not mint a
# lesson that then rides along with every future planning call forever.
_DISTILL_LESSON_SYSTEM = (
    "You maintain the standing instructions of an automated film planner. The "
    "planner writes a MOTION PROMPT for each pair of still frames; an "
    "image-to-video model then renders the clip between them. A human has "
    "just watched one rendered clip and written what was wrong (or right) "
    "with it. Turn that note into at most two GENERAL RULES the planner "
    "should follow from now on. "
    "A rule must be: (1) GENERAL — it applies to films, people and frames "
    "this planner has never seen, so never name the project, the clip id, "
    "the specific people, or the specific place; write 'a person', 'the "
    "subjects', 'a pair of frames' instead; (2) ACTIONABLE — it tells the "
    "planner what to DO or NOT DO when writing a prompt ('when the two "
    "frames show the same person at different distances, stage the approach "
    "as the last few steps only'), never a description of what went wrong "
    "('the clip looked bad'); (3) SHORT — one sentence, at most 35 words, "
    "imperative mood; (4) ABOUT THE PROMPT, not about the renderer's "
    "settings, the model choice, the pricing or the software. "
    "Choose each rule's scope: 'motion' for anything about what happens in a "
    "clip (movement, staging, pace, identity, beats, wording of the motion "
    "prompt) and 'style' for anything about how a STILL FRAME is drawn "
    "(likeness, framing, cropping, colour, background detail). Most notes "
    "are 'motion'. "
    "If the note is praise, write the rule as the behaviour to KEEP doing, "
    "in the same imperative form. "
    "RETURN NO RULES AT ALL (an empty list) when the note carries nothing "
    "generalisable — it is only about this one clip's content, it is too "
    "vague to act on ('didn't like it'), or it merely repeats an instruction "
    "the planner obviously already has. An empty list is a perfectly good "
    "answer and is much better than a vague rule, because every rule you "
    "write is sent with every future planning call and competes with the "
    "instructions that are already there. "
    "Return ONLY JSON of the shape "
    '{"lessons": [{"scope": "motion"|"style", "text": str}, ...]}.'
)

_LESSON_SCHEMA = {
    "type": "object",
    "properties": {
        "lessons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scope": {"type": "string", "enum": list(SCOPES)},
                    "text": {"type": "string"},
                },
                "required": ["scope", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["lessons"],
    "additionalProperties": False,
}

# Deterministic fallback used when even the reword model call fails: bolt an
# explicit safe-for-work clause onto the prompt so the next attempt has a chance.
_SAFE_SUFFIX = (
    " (Safe-for-work, wholesome, non-explicit scene; fully clothed adults; "
    "no nudity, no sexual content, no gore.)"
)

# --- Storyboard prompting -------------------------------------------------- #
# Lifted out of create_storyboard so the method reads as orchestration, not a
# wall of text. The model returns one JSON object that we post-process into a
# Storyboard (output paths + transitions are added in _assemble_storyboard).
_STORYBOARD_SYSTEM = (
    "You are a film pre-production assistant. Produce a storyboard for a "
    "short cinematic video that will be rendered as a sequence of still "
    "key frames, then animated between consecutive frames. "
    "CRITICAL: keep the same characters, same world, same lighting "
    "language, and same color palette across every frame so the frames "
    "form a continuous, consistent visual flow. Each image_prompt must "
    "be fully self-contained and restate the recurring visual identity "
    "(character looks, wardrobe, environment, palette, lighting) so a "
    "text-to-image model produces consistent results frame to frame. "
    "ADJACENT-FRAME CONTINUITY (most important for smooth transitions): "
    "each frame is animated only into the very next frame, so every "
    "consecutive pair must be CLOSELY related and easy to interpolate "
    "between. Treat consecutive frames as moments a second or two apart "
    "in the SAME shot, not separate cuts: keep the same location, "
    "background, subjects, and overall composition from one frame to the "
    "next, and change only ONE thing at a time by a small, smooth amount "
    "(a slight camera push/pan, a subject moving or turning a little, a "
    "gradual change in light or expression). Avoid hard cuts, teleporting "
    "the camera, swapping the setting, or introducing/removing major "
    "elements between consecutive frames. When the scene genuinely must "
    "change, bridge it gradually across two or three frames (e.g. push in, "
    "pass behind an object, or fade through a doorway) rather than jumping. "
    "In each image_prompt, explicitly describe the frame as a small, "
    "natural continuation of the previous one so the start and end of "
    "every clip share the same framing and content. "
    "PER-CLIP DURATION: for each frame, also pick duration_to_next — the "
    "length in seconds (either 5 or 10) of the clip that animates this "
    "frame into the next one. Default to 5: because consecutive frames are "
    "closely related, 5 seconds is enough for most transitions and keeps the "
    "film pacy. Choose 10 only for a HARD transition — when the people "
    "differ between the two frames, the location or setting changes, a "
    "person's clothing or appearance changes noticeably, or the action needs "
    "two or more sequential beats to play out; squeezing such a change into "
    "5 seconds makes the subject visibly teleport. The last frame's "
    "duration_to_next is ignored. "
    "SOUND: for each frame, also write sound_to_next — a short phrase "
    "describing the diegetic ambient sound and sound effects for the clip "
    "that animates this frame into the next one (e.g. 'waves lapping, gulls "
    "calling, soft wind'). Describe real on-screen/world sounds only — no "
    "music, no speech, no narration."
)

# --- Mode A transition planning (vision) ----------------------------------- #
# Mode A already has the key frames (the user's styled images). Instead of
# inventing frames, we show the model the real frames in order and ask it to plan
# the clip that animates each consecutive pair, so the start/end interpolation is
# smooth and each clip gets an appropriate length.
_MODE_A_SYSTEM = (
    "You are a film director planning how to animate a sequence of existing "
    "still key frames into one continuous short film. You are shown the frames "
    "in order. Each consecutive pair (frame N -> frame N+1) is handed to an "
    "image-to-video model that takes the two frames as the START and END of one "
    "clip and interpolates between them; your motion_prompt tells it what "
    "happens in between. "
    "SUBJECTS ONLY — NEVER DESCRIBE THE SCENE. A motion_prompt is about the "
    "PEOPLE (and any animal or vehicle acting as a character): where they "
    "move, how they move, and where they end up. The two frames already fix "
    "the setting, the light, the outfits and the entire look, and the video "
    "model interpolates all of that by itself — HOW the world gets from one "
    "frame to the other is not your business and does not belong in the "
    "prompt. Never write the background, the location or its name, the "
    "weather, the light, or the time of day ('the green Andean peaks rise "
    "behind them', 'the room gives way to open sky', 'the light shifts to "
    "golden hour', 'the salt flat becomes a mountainside'). Those words buy "
    "nothing — the scenery changes anyway — while spending the clip's word "
    "budget and pulling the model's attention off the one thing it gets "
    "wrong on its own: the people. When the pair's biggest change IS the "
    "setting, the prompt is still only about the people: what they do while "
    "it changes around them. "
    "NAME A CONCRETE PHYSICAL ACTION FOR EVERY PERSON WHO MOVES. The video "
    "model animates BODIES, so every person doing anything gets a verb you "
    "could act out and watch happen: walk, step, stride, back away, turn, "
    "spin, crouch, kneel, sit down, stand up, lie back, lean, bend, stretch, "
    "reach, point, wave, clap, hug, kiss, hold hands, link arms, pick up, "
    "put down, push, pull, carry, throw, catch, climb, jump, hop, skip, "
    "crawl, roll, duck, spin around, dance, run, jog, sprint, slide, ski, "
    "swim, paddle, row, ride, pedal, drive. VAGUE, ABSTRACT OR EMOTIONAL "
    "phrasing is FORBIDDEN — the model cannot animate it, so it fills the "
    "time with a morph, a drift, or a whip-pan: never 'squeeze a little "
    "closer', 'share a moment', 'move together', 'come closer', 'shift', "
    "'adjust', 'settle into', 'engage', 'interact', 'they connect', 'the "
    "mood changes'. A real plan built on 'squeeze a little closer, share a "
    "quiet smile' gave the video model nothing to render at all. "
    "Expressions (smiling, laughing, closing their eyes) are fine as a "
    "detail hung on a body movement, never as the movement itself. If you "
    "cannot name the body movement, you do not yet know what the clip "
    "shows: look at the two frames again and find the physical action that "
    "gets from one to the other. "
    "MATCH THE VERB TO THE REAL PACE AND GROUND. Walk, step and stroll for "
    "calm motion; run, jog and sprint ONLY when the shot genuinely is "
    "someone running (a child charging across a lawn) and the ground "
    "covered is short enough for the clip's length — never as a way to "
    "cross a long route faster; climb, crawl, wade, ski and paddle only "
    "when the frames actually show that terrain or gear. The verb has to be "
    "true to the picture: people on a flat street do not climb, and people "
    "photographed standing still do not sprint. "
    "For EACH consecutive pair, describe WHAT THE SUBJECTS DO to carry the "
    "start frame into the end frame, so the result feels like a scene from a "
    "real movie — never like a slideshow transition. Work in this priority "
    "order: "
    "1. SUBJECT ACTION FIRST. Whenever possible, bridge the frames through the "
    "characters/subjects themselves: they move, turn, walk, gesture, react, "
    "grow, or change in a way that plausibly arrives at the end frame (e.g. "
    "'the small boy pushes off the couch, takes a few unhurried steps "
    "forward and sits down cross-legged, looking up'). Ground "
    "every action in what is actually visible in the two frames. "
    "2. IF THEY CANNOT TRAVEL, THEY STAY PUT. When the two frames are too far "
    "apart for any ordinary path — a different place entirely, or subjects "
    "filling both frames — the people hold roughly steady, but they still "
    "get a real movement: they turn in place, take a half-step, shift their "
    "weight from one foot to the other, tilt their head, raise a hand, "
    "straighten a jacket, turn to look at each other. The frames carry the "
    "change on their own. Write what the PEOPLE do; never narrate the "
    "scenery doing it for them, and never fall back on 'they hold the "
    "moment' — that is not a movement. "
    "3. CAMERA LAST. Reach for camera movement only when the two frames are "
    "too different for subject or world action to connect them — and even "
    "then prefer a motivated camera that follows the action (tracking beside "
    "the character, drifting with their gaze). NEVER write a prompt that is "
    "only camera choreography, and do not default to 'zoom in', 'zoom out', "
    "'pan', or 'pull back'. "
    "STAGE ONLY WHAT THE FRAMING ALLOWS: the clip is pinned to the two "
    "images, so it can never show action that needs the subject to shrink "
    "into the distance and come back — walking across a park or room cannot "
    "be staged when the subject fills BOTH frames; the model fakes it with "
    "an ugly background morph instead. When the setting changes around the "
    "same prominent subject, stage it as either (a) the subject walking "
    "out of frame close past the camera and stepping back in afterwards, "
    "or (b) the subject holding roughly steady — standing, turning in "
    "place, a small gesture — and letting the two frames carry the change. "
    "Describe visible travel only when one of the two frames "
    "already shows the subject small or far away. "
    "EVERY POSITION CHANGE NEEDS A PATH — NOBODY TELEPORTS. When a person "
    "stands somewhere different in the end frame — the other side of the "
    "frame, nearer the camera, a different place — the prompt must give "
    "them an ordinary way to get there ON THEIR OWN FEET, and say which "
    "way they go: they walk, step across, turn, cross in front of or "
    "behind the other person, or leave past the camera and come back in. "
    "NEVER let someone simply BE somewhere new: no appearing, arriving, "
    "materialising, floating, drifting, sliding, being swept along, or "
    "being 'now' in the new place. A prompt that states the new position "
    "without the walk between makes the video model slide the person there "
    "through the air or dissolve them and rebuild them, which reads as a "
    "cut, not a shot. If no ordinary walk fits in the clip's length, then "
    "the path IS the exit past the camera and the walk back in — still a "
    "path, still on their own feet. "
    "PACE AND DISTANCE: a clip can only cover as much ground as its length "
    "allows at a calm, natural pace — roughly a few unhurried steps in 5 "
    "seconds, a short approach in 10. The video model holds the two frames "
    "as fixed endpoints, so a prompt that spans a ROUTE forces it to cover "
    "that whole route within the clip: the people sprint, skate across the "
    "ground, or blur together. Two real clips were ruined exactly this way "
    "— 'stroll side by side along the meadow trail until they stop beside "
    "the bench' and 'stroll down toward a green resort lawn' both rendered "
    "as a frantic sprint. NEVER write travel that names a route or a "
    "destination reached from far off ('walks along the path to ...', "
    "'makes her way across the park to ...', 'heads down toward the ...'). "
    "Write the ARRIVAL instead — only the last few steps and the settling: "
    "'the bald man and the woman with the pink headband take the last few "
    "unhurried steps to the bench and settle onto it side by side'. Name "
    "the pace whenever people move ('unhurried', 'slowly', 'at an easy "
    "pace') and keep the described distance short enough that the pace is "
    "physically possible in the time. A gentler VERB is not enough: "
    "'stroll', 'amble', and 'wander' still tell the model to cover the "
    "whole route, just as 'walk' does — it is the distance that has to "
    "shrink, not the wording. If the two frames really are far apart in "
    "space, that is not a walking shot at all: rate it 4-5 and stage "
    "exit-past-camera plus re-entry, or hold the subject steady (a turn in "
    "place, a small gesture) and let the two frames carry the change. "
    "NO OFF-SCREEN HANDS: every action must be performed by a person "
    "visible in the frames, under their own power. NEVER write actions done "
    "TO the subject by an unseen agent — 'is lifted out', 'is carried', "
    "'is gently set down', 'as if being lifted' — when no such person is "
    "visible: the video model cannot render the invisible helper, so the "
    "subject levitates through the air instead. If a "
    "transition seems to need an off-screen handler, restage it: the "
    "subject moves under their own power (climbs down, stands up, turns) "
    "or holds steady with a small movement of their own. "
    "LAND ON THE END FRAME: each clip stops exactly at its end frame, so the "
    "FINAL clause of every motion_prompt must describe the subject already "
    "in the end frame's pose, activity and SIDE OF THE FRAME, and every "
    "action the "
    "prompt starts must be finished by then. A prompt whose last beat is "
    "people LEAVING (walking out past the camera, turning away) is always "
    "wrong: the clip has to end on the end frame, so the final beat is "
    "always them arriving in it. Before writing, look at the END "
    "frame of the pair and work backwards: what happens so the start frame "
    "arrives exactly there? Never end the prompt mid-action or still inside "
    "the start frame's activity. Keep the landing clause SHORT — the end "
    "frame itself supplies the visual detail, so never append an inventory "
    "of pose, outfit, and expression ('... exactly as seen in the next "
    "frame'). "
    "DIFFERENT PEOPLE: before writing each motion_prompt, check whether the "
    "person or people in the start frame are the SAME individuals as in the "
    "end frame. If anyone differs (e.g. a man in one frame and a woman in the "
    "other, or a person present in only one frame), you MUST stage the change "
    "as separate people sharing one continuous scene: the first person walks "
    "out of frame, turns and moves away, or passes behind a foreground "
    "element, and the other person walks in, is revealed, or was already "
    "present in the background and comes forward. NEVER treat two different "
    "people as one continuous character and NEVER imply a person "
    "transforming, turning into, or becoming someone else — the interpolating "
    "model will morph one face and body into the other, which looks "
    "grotesque. An exit-and-entrance like this is one of the few transitions "
    "that justifies a 10-second clip, so both movements have room to play "
    "out. (The same person at a different age or in different clothes is "
    "still the same person — continuous growth or change is fine there.) "
    "Do not describe editing effects ('crossfade', 'dissolve', 'transition', "
    "'morph') — describe continuous physical motion only. "
    "REFER TO PEOPLE BY APPEARANCE ONLY: the video model sees only the two "
    "images — it knows no names, family roles, or relationships, so those "
    "words are noise it resolves by guessing. Refer to every person, every "
    "time, by a short epithet grounded in their VISIBLE appearance ('the "
    "bald man', 'the younger man with brown hair', 'the woman in the red "
    "dress', 'the small girl with curly hair') and reuse the same epithet "
    "consistently. NEVER use relationship or role words — 'the son', 'the "
    "father', 'mom', 'grandma', 'the couple', 'the family' — and never "
    "proper names: the model cannot tell who is who, so it puts the action "
    "on the wrong person or blends two people (a real clip prompted with "
    "'the son splashes the water' left the model to guess which man that "
    "was). When several people appear, an action belonging to one person "
    "must name that person by epithet — never only a collective 'they', "
    "'the couple', or 'the two of them'. Collective phrasing gives the "
    "video model no anchor to keep the identities apart, and it blends "
    "their faces and hair into each other. Keep each epithet short (2-5 "
    "words) so the word cap still fits. "
    "AN EPITHET MUST DISTINGUISH, NOT MERELY DESCRIBE. Before writing a "
    "pair, list every person visible across the two frames and compare "
    "them. If two of them share the feature you were about to name — two "
    "bald men, two women in sun hats, two men in dark jackets — then that "
    "feature ALONE is FORBIDDEN as an epithet for either one: the video "
    "model cannot tell which you mean, so it puts the action on the wrong "
    "person or blends the two. Name the contrasting detail that separates "
    "them and repeat it in EVERY mention: 'the bald man in pink "
    "sunglasses' vs 'the bald man in the light blue shirt'. Prefer a "
    "difference visible in BOTH frames (clothing colour, eyewear, hair "
    "length, build) over one that only shows in one. A real movie had two "
    "bald men and most of its prompts said just 'the bald man'; the model "
    "mixed them up from clip to clip for the whole film. "
    "CAST LIST: alongside the transitions, output `characters` — the "
    "movie's cast: one entry per distinct person (or recurring "
    "animal/vehicle subject) visible in the frames, each with a short slug "
    "id and the exact epithet to use for them. When the user message "
    "already PROVIDES a cast list, those epithets are FIXED: use each one "
    "verbatim, word for word, in every motion prompt that mentions that "
    "person — never shorten it, reword it, or invent a fresh epithet for "
    "someone already in the cast — and return in `characters` ONLY people "
    "missing from the provided cast (an empty array when nobody is new). "
    "When no cast is provided, build it: choose each epithet by the rules "
    "above (visible appearance only, distinguishing, 2-5 words). "
    "AN EPITHET IS FOR THE WHOLE MOVIE, SO NEVER ANCHOR IT TO CLOTHING. "
    "These films are assembled from photographs taken months or years "
    "apart: the boy in the striped shirt in this frame is in a swimsuit in "
    "the next one, so 'the boy in the striped shirt' stops matching anybody "
    "— the video model hunts for a striped shirt, fails, and puts the "
    "action on whoever is nearest. Build every epithet from what the person "
    "CARRIES between photos: age band and relative size ('the smaller "
    "boy', 'the teenage girl'), hair — colour, length, texture, or its "
    "absence ('the bald man', 'the woman with long curly hair'), facial "
    "hair, glasses, build. NEVER use a garment, its colour, or a hat as the "
    "identifying feature: no 'the boy in the yellow shirt', 'the girl in "
    "the purple dress', 'the man in the blue jacket'. When two people share "
    "every durable trait (two similar boys), separate them by RELATIVE "
    "size or age — 'the taller boy' / 'the smaller boy' — which still works "
    "in every photo. Only when even that fails may clothing appear, and "
    "then only ADDED to a durable core, never instead of it. "
    "Also give each character a `durable_epithet`: the same person written "
    "using ONLY these carried-over features, with no garment word at all. "
    "It is the wording the film falls back on, so make it a drop-in "
    "replacement — same shape, 2-5 words. "
    "EPITHETS MUST BE DISTINCT ACROSS THE WHOLE CAST, not merely within "
    "one pair. Never return two cast entries that differ only by an added "
    "or dropped word — 'woman with dark hair bun' next to 'young woman "
    "with dark hair bun', 'teenage girl with long dark hair' next to "
    "'teenage girl with long dark hair and bun' (both from a real cast): "
    "in any prompt that mentions one of them, the video model cannot tell "
    "which is meant and acts on whoever is nearest, and the pipeline's own "
    "identity matching reads them as one person and gives up. Before "
    "returning the cast, re-read it as a set and separate every pair of "
    "look-alikes by a trait the other genuinely lacks — relative age or "
    "size, hair length or texture, glasses, facial hair, build. "
    "A background crowd that acts as scenery — a full dinner table, "
    "distant swimmers, a busy plaza — may be ONE collective entry ('the "
    "large dinner group') instead of an entry per stranger: individual "
    "entries are for people who recur and carry action. Never give a "
    "collective entry person-level choreography in a motion prompt. "
    "Within one pair you may still add a passing detail inline when it "
    "helps that clip: 'the smaller boy with curly hair, here in a striped "
    "shirt'. The epithet itself stays as it is. "
    "SAY WHERE PEOPLE ARE AND WHICH WAY THEY GO: the model matches regions "
    "of the start frame to regions of the end frame, so screen position is "
    "the strongest anchor available. Whenever anyone enters or leaves, give "
    "their side of the frame and their direction — 'the woman and the man "
    "on the right walk out of frame to the right', 'the bald man in pink "
    "sunglasses turns left'. Name departing people INDIVIDUALLY: a "
    "collective ('the right couple') is a single blob to the model, which "
    "may remove only one of them or fuse them together. A real pair that "
    "morphed under 'the right couple get out of the frame' rendered "
    "cleanly once rewritten as 'the woman and man on the right get out of "
    "the frame ... turn left and walk back to kiss'. "
    "ARRANGEMENT SWAP: when the same people appear in BOTH frames, compare "
    "their left-right arrangement before writing. If they have traded "
    "positions (e.g. the bald man is on the right in the start frame but on "
    "the left in the end frame), NEVER use hold-steady staging ('the world "
    "shifts around them', 'they settle back') — the interpolating model maps "
    "left onto left, so pinning swapped people in place morphs each person "
    "into the other (one real clip grew hair on a bald man mid-shot). Stage "
    "the swap as explicit physical motion instead: prefer an "
    "exit-and-entrance — they step out of frame past the camera, then "
    "re-enter in the new arrangement one at a time, each named by epithet — "
    "or have one person visibly cross in front of or behind the other. The "
    "exit is only HALF the staging: a prompt that ends with people walking "
    "past the camera never reaches the end frame, and the video model — "
    "which must land on that frame regardless — fills the gap with a "
    "whip-pan blur or a hard cut. Always write the walk back in as well. "
    "A worked example, for a couple standing man-left/woman-right in the "
    "start frame and woman-left/man-right in a different place in the end "
    "frame: 'the bearded man on the left and the long-haired woman on the "
    "right walk forward past the camera and out of frame, then step back "
    "in one at a time — she from the left, he from the right — and turn to "
    "face each other.' Note what that prompt never mentions: where any of "
    "it happens. A "
    "position swap is at least difficulty 4; a swap combined with a setting "
    "change is 5. "
    "TOO MANY PEOPLE TO CHOREOGRAPH: exit-and-entrance staging is for a FEW "
    "people — it does not scale. When more than three people would have to "
    "leave or arrive between the two frames, or either frame holds five or "
    "more visible people and the roster or arrangement changes, per-person "
    "choreography cannot fit the beat budget, and the video model answers a "
    "many-body staging with ghost dissolves, bodies blending into each "
    "other, and people vanishing — a real movie staged seven exits and "
    "three entrances in ten seconds and got exactly that. For these pairs — "
    "and only these — write the ONE allowed camera transition instead of "
    "subject choreography: the people simply stay as they are, or keep "
    "doing what they are visibly doing, while the camera moves smoothly "
    "and steadily across to the new setting and settles on exactly what "
    "the end frame shows. When BOTH frames are full of people, have the "
    "camera turn away from them first and travel across the surroundings "
    "before settling — a transit that keeps faces in frame smears them "
    "into each other, while blur over scenery reads as ordinary camera "
    "speed. Name nobody individually in it, stage no exits "
    "and no entrances. This is a deliberate exception to CAMERA LAST, and "
    "code enforces the same rule: a many-mover staging is replaced with a "
    "deterministic camera prompt anyway, so spend your effort on the pairs "
    "that can be staged. "
    "BEAT BUDGET: rate the pair's difficulty BEFORE writing the motion, "
    "because the rating sets the clip's length and the length sets how much "
    "can happen. Difficulty 1-3 = a 5-second clip = exactly ONE continuous "
    "action, ONE sentence ('she crosses the kitchen and sets the tray on "
    "the table' / 'the toddler runs to the patio chair and settles onto "
    "it'). Difficulty 4-5 = up to 10 seconds = at most TWO beats, two "
    "sentences. A prompt with more beats than the clip can hold does not "
    "get compressed — the video model drops or fakes them, typically by "
    "swapping in a different-looking subject mid-clip. "
    "SAY THE ONE THING, THEN STOP. Nothing counts your words, so the "
    "discipline is yours: write the shortest prompt that fully specifies "
    "who moves, how, and where they end up — then end the sentence. Every "
    "extra clause is a chance for the video model to invent a beat you did "
    "not ask for. A prompt that wanders — narrating the setting, listing "
    "each person's pose and expression, restating what the end frame "
    "already shows, adding atmosphere — is WORSE than a short one, not "
    "richer: the two frames already carry all of that, and the words spent "
    "on it pull the model off the people. If a clause does not name a "
    "person, a movement, or where that movement ends, cut it. "
    "COUNT THE BEATS BEFORE YOU WRITE. A beat is any distinct action a "
    "subject performs, and a short prompt can still stack far too many of "
    "them. A real 10-second prompt — 'they laugh, step back from the edge, "
    "and stroll down toward a green resort lawn where they change into "
    "white robes, set backpacks aside, and move together into a relaxed "
    "selfie near lounge chairs' — is only 46 words but SIX beats, and it "
    "rendered as a frantic sprint through all of them instead of the calm "
    "moment it described. Chained verbs are the tell: every extra 'and', "
    "'then', 'where', 'until', or comma-joined verb is another beat. Keep "
    "to ONE (5s) or TWO (10s) and simply drop the rest — the end frame "
    "already shows where everything lands, so the prompt never has to "
    "narrate the way there. Keep every motion_prompt in "
    "present tense; preserve each person's identity, wardrobe, and "
    "environment except for the changes visible between the frames; no hard "
    "cuts, no people who appear in neither frame, no NEW text appearing on "
    "screen. Text that is ALREADY part of a frame — a shop sign, a birthday "
    "banner, a logo on a shirt, a book cover — is part of the scene and "
    "stays: never ask for it to be removed, hidden, or changed, and treat it "
    "like any other scenery. Do not "
    "mention frame numbers or that these are AI-generated images. "
    "SAME PERSON, ONE PROTAGONIST: when both frames show the same individual "
    "— even at a different age, in different clothes, or in a new setting — "
    "write the prompt so there is unmistakably ONE person throughout. Say it "
    "explicitly ('the same woman, now in a winter coat, ...' / 'the same "
    "little boy, now in a blue sweatshirt, ...'), use one consistent noun "
    "phrase with natural singular pronouns matching the person's visible "
    "appearance ('he', 'she'), and NEVER use singular 'they'/'their' — the "
    "video model reads it as several people. Avoid handover phrasing like "
    "'the scene shifts to a man...' that lets the end frame read as a "
    "different individual; the video model will then swap in a new character "
    "instead of keeping one. Never add people who are not visible in the "
    "frames. The same identity rules apply when the main subject is not a "
    "person at all — an animal, a vehicle, a building: one consistent "
    "subject, never a morph into a different one. "
    "DIFFICULTY: for each pair, rate difficulty 1-5 by how much the two "
    "IMAGES differ — judge the pixels, not how gracefully you can word the "
    "motion: 1 = same setting and outfit, one continuous action; 2 = small "
    "change (light, pose, expression, camera); 3 = exactly ONE major change "
    "(the setting changes OR the outfit changes, everything else carries "
    "over); 4 = two major changes at once (setting AND outfit change), or "
    "ANY pair whose transition cannot physically play out in 5 seconds — a "
    "journey that must depart, travel, and settle, INCLUDING a relocation "
    "within the same setting (high chair to couch across one room) when "
    "the subject is prominent in both frames: the exit, the crossing, and "
    "the arrival cannot compress into one action, so a 5-second clip "
    "degrades into a disguised cut — and INCLUDING the same people trading "
    "left-right positions between the frames (the arrangement swap above); "
    "5 = the "
    "frames share almost nothing, the people differ (the "
    "exit-and-entrance above), or a position swap combines with a setting "
    "change. Rate honestly and comparatively: in a typical "
    "photo-album movie (family, travel, an event) most consecutive pairs "
    "change setting or outfit — that alone is a 3, not a 4. Clip length is "
    "derived from this rating and NOTHING ELSE — every pair you rate 4 or 5 "
    "becomes a 10-second clip, with no quota and no second opinion — so an "
    "inflated rating directly makes the movie longer, slower and twice as "
    "expensive on that pair. Rate 4+ only when the pair genuinely cannot "
    "play out in five seconds. "
    "SOUND: also write sound_prompt — a short phrase describing the diegetic "
    "ambient sound and sound effects for that clip (e.g. 'waves lapping, gulls "
    "calling, soft wind'). Real on-screen/world sounds only — no music, no "
    "speech, no narration."
)

_STORYBOARD_NO_TEXT_FRAMES = (
    " Every frame MUST depict a real visual scene with characters "
    "and/or an environment. NEVER create a frame that is only text "
    "on a blank, black, or solid-colour background (title cards, "
    "intro/outro text screens, caption cards, quote cards, credits). "
    "Text is allowed only when it appears naturally on top of a real "
    "scene — it must never be the sole content of a frame."
)

# --- Strict output schemas -------------------------------------------------- #
# Enforced via OpenAI structured outputs (json_schema, strict), so the model
# can't omit fields or return the wrong types. The prose shape description in
# _STORYBOARD_JSON_SHAPE stays for the semantic guidance (id format, when to
# pick 5 vs 10, ...); the schema is the hard guarantee. Coercion in
# _assemble_storyboard / _coerce_transition_plans remains as a final net.
_DURATION_ENUM = sorted(VALID_DURATIONS)

_STORYBOARD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "project_title": {"type": "string"},
        "style": {"type": "string"},
        "concept": {"type": "string"},
        "scenes": {"type": "array", "items": {"type": "string"}},
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "image_prompt": {"type": "string"},
                    "negative_prompt": {"type": "string"},
                    "duration_to_next": {"type": "integer", "enum": _DURATION_ENUM},
                    "sound_to_next": {"type": "string"},
                },
                "required": [
                    "id", "description", "image_prompt", "negative_prompt",
                    "duration_to_next", "sound_to_next",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "project_title", "style", "concept", "scenes", "frames",
    ],
    "additionalProperties": False,
}

_TRANSITIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Property order = generation order, and it matters:
                    # pair_index first so the model anchors on which frame
                    # pair it is describing (a real 20-frame plan slipped by
                    # one pair mid-array; code re-aligns by this index in
                    # _realign_by_pair_index), then difficulty so the beat
                    # budget of the motion prompt can depend on the clip
                    # length that will be derived from it.
                    "pair_index": {"type": "integer"},
                    # The model rates how much the two frames differ; the
                    # DURATION is derived in code (_coerce_transition_plans),
                    # not chosen by the model — prompt-side "prefer 5" biases
                    # produced all-5s and all-10s plans on real projects.
                    "difficulty": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                    # A census of the pixels, not the cast: EVERY visible
                    # human figure in each frame, including background
                    # people nobody tagged. Confirmed tags list who MATTERS,
                    # and a real frame held ten people with two tagged — the
                    # plan staged those two as if they were alone and the
                    # render collapsed the other eight. Code takes the
                    # larger of this count and the roster when deciding
                    # whether a pair can be staged at all
                    # (is_unstageable_pair).
                    "start_heads": {"type": "integer"},
                    "end_heads": {"type": "integer"},
                    # Who stands where, left to right, in each frame. Asked
                    # for as DATA because the planner reliably sees it but
                    # unreliably acts on it: it has twice written
                    # hold-steady prose for a pair whose people trade sides,
                    # which morphs them into each other. Code compares these
                    # two lists (is_arrangement_swap) and restages the
                    # prompt itself when they disagree.
                    "start_order": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "end_order": {
                        "type": "array", "items": {"type": "string"},
                    },
                    "motion_prompt": {"type": "string"},
                    "sound_prompt": {"type": "string"},
                },
                "required": [
                    "pair_index", "difficulty", "start_heads", "end_heads",
                    "start_order", "end_order",
                    "motion_prompt", "sound_prompt",
                ],
                "additionalProperties": False,
            },
        },
        # The movie's cast (see CAST LIST in _MODE_A_SYSTEM): built on the
        # first full plan; later calls receive the existing cast in the
        # instruction and return only people missing from it (usually []).
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "epithet": {"type": "string"},
                    # The same person named ONLY by what they carry from
                    # photo to photo. Code swaps it in when `epithet` turns
                    # out to be anchored to clothing (see
                    # repair_cast_epithets): the cast is used across a whole
                    # album, where the shirt in this frame means nothing.
                    "durable_epithet": {"type": "string"},
                },
                "required": ["id", "epithet", "durable_epithet"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["transitions", "characters"],
    "additionalProperties": False,
}


def _json_schema_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": schema},
    }


def _merge_cast(
    existing: Optional[list[Character]], returned: Any
) -> list[Character]:
    """The storyboard's cast plus genuinely new people the model reported.

    Existing entries are never modified, reordered, or dropped — their
    epithets are already baked into planned motion prompts, so rewriting one
    here would silently split a person's identity across the movie. Returned
    entries are appended only when they carry a usable epithet and neither
    their id nor their epithet collides with the cast (models sometimes
    re-list the provided cast despite being told not to; that must not
    duplicate anyone).
    """
    cast = list(existing or [])
    seen_ids = {c.id for c in cast}
    seen_epithets = {c.epithet.strip().lower() for c in cast}
    for item in returned if isinstance(returned, list) else []:
        if not isinstance(item, dict):
            continue
        epithet = str(item.get("epithet") or "").strip()
        if not epithet or epithet.lower() in seen_epithets:
            continue
        cid = str(item.get("id") or "").strip()
        if not cid:
            cid = "-".join(epithet.lower().split())[:40]
        if cid in seen_ids:
            # Same id but a different epithet: the saved cast wins.
            continue
        cast.append(Character(id=cid, epithet=epithet))
        seen_ids.add(cid)
        seen_epithets.add(epithet.lower())
    return cast


def _people_count(n: int) -> str:
    """"1 person" / "5 people" — the planner reads these sentences."""
    return f"{n} person" if n == 1 else f"{n} people"


def _tagged_people_block(tags: list[Optional[list[str]]]) -> str:
    """Tell the planner who is in each frame, and what that means per pair.

    Two halves, because the tags answer two different questions the planner
    is bad at. The roster fixes WHO (and in what left-to-right order) is in
    each frame. The per-pair verdict then states the consequence that
    actually decides the staging: the same people continuing (animate them
    continuously) or a different cast (stage an exit and an entrance, never
    a continuous morph). Both are stated as fact — a human looked at the
    photographs — so the model is told not to re-decide either one.
    """
    lines = [
        "CONFIRMED IDENTITIES — a human has tagged who is in these frames. "
        "This is FACT and it OVERRULES what you think you see: use exactly "
        "these people and these epithets for the frames listed, never "
        "re-identify anyone, never add a person who is not listed for that "
        "frame, and copy the listed order into start_order/end_order."
    ]
    for i, people in enumerate(tags, start=1):
        if people is None:
            continue
        who = ", ".join(people) if people else "(nobody)"
        lines.append(f"- Frame {i:03d}, left to right: {who}")

    verdicts: list[str] = []
    for i, (start, end) in enumerate(zip(tags, tags[1:]), start=1):
        if start is None or end is None:
            continue
        shared = [p for p in start if p in end]
        gone = [p for p in start if p not in end]
        arrived = [p for p in end if p not in start]
        reason = unstageable_reason(start, end)
        if reason:
            # Do NOT prescribe the exit-and-entrance here: on these pairs it
            # is exactly the choreography that renders as ghost dissolves
            # and mushed bodies. The camera carries them instead. The cause
            # is named by the gate itself, so this sentence can never again
            # explain a crowd rule as if it were a mover count.
            if reason == "movers":
                cause = (
                    f"{_people_count(len(gone) + len(arrived))} leave or "
                    "arrive between these frames — TOO MANY CHANGES TO "
                    "CHOREOGRAPH one at a time"
                )
            else:
                cause = (
                    f"these frames hold {_people_count(max(len(start), len(end)))}"
                    + (
                        " and the group changes"
                        if (gone or arrived)
                        else " and they rearrange"
                    )
                    + " — TOO CROWDED TO CHOREOGRAPH person by person, "
                    "because in a group that size an epithet no longer picks "
                    "anyone out"
                )
            verdict = (
                f"{cause}. Stage no exits and no entrances: everyone stays "
                "as they are, and the camera transition carries the change "
                "(see TOO MANY PEOPLE TO CHOREOGRAPH)"
            )
        elif shared and not gone and not arrived:
            verdict = (
                f"the SAME {'people' if len(shared) > 1 else 'person'} "
                f"({', '.join(shared)}) in both frames — animate them "
                "continuously; they are one and the same, whatever they are "
                "wearing or however much older they look"
            )
        elif not shared:
            verdict = (
                f"COMPLETELY DIFFERENT people ({', '.join(start) or 'nobody'} "
                f"-> {', '.join(end) or 'nobody'}) — they only look alike; "
                "NEVER let one turn into the other. Stage an exit and an "
                "entrance (or a reveal)"
            )
        else:
            parts = []
            if shared:
                parts.append(f"{', '.join(shared)} stay(s)")
            if gone:
                parts.append(f"{', '.join(gone)} leave(s)")
            if arrived:
                parts.append(f"{', '.join(arrived)} arrive(s)")
            verdict = (
                "; ".join(parts)
                + " — whoever stays is continuous, whoever leaves or arrives "
                "must visibly walk out of or into the frame"
            )
        verdicts.append(f"- Pair {i} (frame {i:03d} -> {i + 1:03d}): {verdict}.")
    if verdicts:
        lines.append(
            "What that means for each pair (already worked out for you — "
            "do not second-guess it):"
        )
        lines.extend(verdicts)
    return "\n".join(lines)


# --- Cast epithets that survive the whole movie ----------------------------- #
# An epithet is written once and then used in every prompt that mentions that
# person, across a film assembled from photos taken years apart. So it has to
# name something the PERSON carries, not something they happened to be wearing
# in the frame the cast was built from: "the boy in the striped shirt" is a
# perfect description of one photo and useless in the next one, where he is in
# a swimsuit — the video model then looks for a striped shirt, finds none, and
# puts the action on whoever is nearest.
#
# Prompt guidance alone did not hold this line (the planner still came back
# with "boy in yellow shirt" / "boy in striped shirt" / "girl in purple
# dress"), which is the same failure the arrangement-swap rule hit: the model
# perceives fine but reaches for the easiest discriminator. So code checks.
_GARMENT_WORDS = frozenset({
    "shirt", "t-shirt", "tshirt", "tee", "top", "blouse", "dress", "skirt",
    "trousers", "pants", "jeans", "shorts", "jacket", "coat", "hoodie",
    "sweater", "sweatshirt", "jumper", "cardigan", "vest", "suit", "tie",
    "uniform", "swimsuit", "swimming", "bikini", "trunks", "pyjamas",
    "pajamas", "romper", "onesie", "overalls", "dungarees", "gown", "robe",
    "costume", "outfit", "clothes", "clothing", "tracksuit", "leggings",
    "tights", "socks", "shoes", "sneakers", "trainers", "boots", "sandals",
    "scarf", "apron", "jersey", "kit",
    # Headwear changes photo to photo just like a shirt does.
    "hat", "cap", "beanie", "headband", "bandana", "helmet", "hood",
})

# Traits that stay with a person across years of photographs. Eyewear and
# facial hair are in here deliberately: they are not permanent, but they are
# stable over the span of a single family album and the styling prompt
# already treats eyewear as an identifying feature.
_DURABLE_TRAIT_WORDS = frozenset({
    "bald", "balding", "shaved", "buzz-cut", "hair", "haired", "hairs",
    "curly", "wavy", "straight", "long", "short", "cropped", "ponytail",
    "pigtails", "braids", "braided", "bun", "dreadlocks", "afro", "fringe",
    "blonde", "blond", "brunette", "ginger", "redhead", "red-haired",
    "dark-haired", "brown-haired", "fair-haired", "grey", "gray", "greying",
    "white-haired", "silver",
    "beard", "bearded", "moustache", "mustache", "stubble", "goatee",
    "glasses", "spectacles", "sunglasses", "eyeglasses",
    "freckled", "freckles", "dimpled", "tattooed", "tattoo", "pierced",
    "tall", "taller", "tallest", "small", "smaller", "smallest", "little",
    "tiny", "big", "bigger", "biggest", "slim", "slender", "stocky", "burly",
    "chubby", "plump", "heavyset", "broad-shouldered",
    "older", "oldest", "younger", "youngest", "elderly", "grown", "teenage",
    "teenaged", "middle-aged", "young", "old", "baby", "infant", "newborn",
    "toddler", "elder",
    "dark-skinned", "light-skinned", "pale", "tanned", "freckly",
})

# The bare noun for a person: durable, but on its own it cannot tell two boys
# apart, so it never counts as the distinguishing trait.
_PERSON_NOUNS = frozenset({
    "man", "woman", "boy", "girl", "child", "kid", "baby", "toddler",
    "infant", "person", "guy", "lady", "teenager", "teen", "adult", "dog",
    "cat", "puppy",
})


def _epithet_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z\-]*", (text or "").lower())


def is_clothing_anchored(epithet: str) -> bool:
    """Does this epithet identify someone only by what they are wearing?

    True for "the boy in the striped shirt" and "girl in purple dress";
    False for "the bald man in pink sunglasses" (bald is durable), "the
    taller boy", "the small boy with curly hair" — and False for anything
    with no garment word at all, so a plain "the bald man" is never touched.

    Strict in the safe direction: an epithet that carries ANY durable trait
    is left alone even if it also mentions clothing, because the durable part
    is what keeps working in the next photo.
    """
    tokens = _epithet_tokens(epithet)
    if not tokens:
        return False
    if any(t in _DURABLE_TRAIT_WORDS for t in tokens):
        return False
    return any(t in _GARMENT_WORDS for t in tokens)


def repair_cast_epithets(data: dict[str, Any]) -> list[str]:
    """Replace clothing-only epithets with the durable ones, in place.

    The model returns both a chosen ``epithet`` and a ``durable_epithet``
    describing the same person by features that do not change between
    photos. Code — not the model — decides which one the movie uses: when the
    chosen epithet is anchored to clothing, the durable one takes its place
    here AND in every motion prompt of the same response, so the two never
    disagree.

    Returns a list of human-readable "old -> new" swaps for logging. A
    replacement is refused when it is missing, itself clothing-anchored, or
    would collide with another character's epithet — a duplicate epithet is
    worse than a fragile one, since it makes two people indistinguishable.
    """
    characters = data.get("characters")
    if not isinstance(characters, list):
        return []
    taken = {
        str(c.get("epithet", "")).strip().lower()
        for c in characters if isinstance(c, dict)
    }
    swaps: list[tuple[str, str]] = []
    for item in characters:
        if not isinstance(item, dict):
            continue
        epithet = str(item.get("epithet") or "").strip()
        if not epithet or not is_clothing_anchored(epithet):
            continue
        durable = " ".join(str(item.get("durable_epithet") or "").split())
        if (
            not durable
            or is_clothing_anchored(durable)
            or durable.lower() in taken
        ):
            continue
        taken.discard(epithet.lower())
        taken.add(durable.lower())
        item["epithet"] = durable
        swaps.append((epithet, durable))

    # The prompts in this same response already use the old wording verbatim.
    if swaps:
        transitions = data.get("transitions")
        for tr in transitions if isinstance(transitions, list) else []:
            if not isinstance(tr, dict):
                continue
            for field in ("motion_prompt", "start_order", "end_order"):
                value = tr.get(field)
                if isinstance(value, str):
                    tr[field] = _swap_all(value, swaps)
                elif isinstance(value, list):
                    tr[field] = [
                        _swap_all(v, swaps) if isinstance(v, str) else v
                        for v in value
                    ]
    return [f"{old!r} -> {new!r}" for old, new in swaps]


def indistinct_epithets(characters: Any) -> list[list[str]]:
    """Groups of cast ids whose epithets cannot be told apart.

    The test is operational, not aesthetic: two epithets collide when the
    person-matcher behind swap and mover detection (``_PERSON_MATCH_RULES``)
    would read them as the same person — one a word-subset of the other, or
    nearly all words shared. Such a pair does double damage: the video model
    has no anchor to keep the two people apart (it acts on whoever is
    nearest), and the pipeline's own matching aborts as ambiguous, silently
    disabling swap and mover detection for every frame either person is in.
    A real cast carried "woman with dark hair bun" AND "young woman with
    dark hair bun" — on a 29-entry cast built from a big-family order.

    Merely similar entries ("the taller boy" / "the smaller boy") are NOT
    flagged: they share a noun but the matcher tells them apart, and so can
    the video model. Nothing here rewrites anything — existing casts are
    frozen (their wording is baked into planned prompts), so the groups are
    surfaced for a hand edit in the panel's Cast editor, like fragile
    epithets are.
    """
    entries: list[tuple[str, frozenset[str]]] = []
    for c in characters or []:
        # Both shapes pass through here over time: Character objects from a
        # saved storyboard, plain dicts fresh off the planner.
        get = c.get if isinstance(c, dict) else lambda k, c=c: getattr(c, k, "")
        cid = str(get("id") or "").strip()
        tokens = _person_tokens(str(get("epithet") or ""))
        if cid and tokens:
            entries.append((cid, tokens))

    # Transitive grouping: if A collides with B and B with C, all three are
    # one knot the human untangles together.
    parent = list(range(len(entries)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i][1], entries[j][1]
            if any(rule(a, b) for rule in _PERSON_MATCH_RULES):
                parent[find(j)] = find(i)

    groups: dict[int, list[str]] = {}
    for i, (cid, _) in enumerate(entries):
        groups.setdefault(find(i), []).append(cid)
    return [ids for ids in groups.values() if len(ids) > 1]


def _swap_all(text: str, swaps: list[tuple[str, str]]) -> str:
    """Replace each old epithet with its durable replacement, case-insensitively."""
    for old, new in swaps:
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


def _realign_by_pair_index(items: list[Any], count: int) -> list[Any]:
    """Order transition items by their declared 1-based ``pair_index``.

    On a real 20-frame plan the model slipped one pair mid-array, so a
    transition landed on the wrong frame pair. Each item now declares which
    pair it describes, and that declaration wins over array position. Falls
    back to the given order when the indices are missing, out of range, or
    duplicated — positional order is then the best remaining guess.
    """
    by_index: dict[int, Any] = {}
    for item in items:
        if not isinstance(item, dict):
            return items
        try:
            idx = int(item.get("pair_index"))
        except (TypeError, ValueError):
            return items
        if not 1 <= idx <= count or idx in by_index:
            return items
        by_index[idx] = item
    return [by_index.get(i, {}) for i in range(1, count + 1)]


def _person_tokens(name: Any) -> frozenset[str]:
    """A person description reduced to comparable words.

    Two planning answers describe the same person slightly differently ('the
    bearded young man with short dark hair' in one frame, 'the bearded man'
    in the other), so comparison is on words, minus pure function words.
    """
    if not isinstance(name, str):
        return frozenset()
    words = re.findall(r"[a-z0-9]+", name.lower())
    return frozenset(w for w in words if w not in _PERSON_STOPWORDS)


# How two descriptions of a person are matched, strictest rule first. Order
# matters: 'the tall man in a red shirt' and 'the short man in a red shirt'
# overlap enough to look alike, so they must be paired by EXACT wording
# before any fuzzy rule gets to call them ambiguous (or, worse, the same).
_PERSON_MATCH_RULES = (
    lambda a, b: a == b,
    lambda a, b: a <= b or b <= a,
    lambda a, b: len(a & b) / len(a | b) >= 0.75,
)


def _match_people(
    starts: list[frozenset[str]], ends: list[frozenset[str]]
) -> Optional[list[tuple[int, int]]]:
    """Pair up the same person across the two frames, or None if unclear.

    Each rule runs over everyone still unmatched, and a person who matches
    two candidates under the same rule aborts the whole comparison: a wrong
    pairing invents a swap that isn't there, which buys a 10-second clip and
    restages a prompt that was fine. Ambiguity is always answered with "no
    opinion", never with a guess.
    """
    pairs: list[tuple[int, int]] = []
    matched_starts: set[int] = set()
    matched_ends: set[int] = set()
    for rule in _PERSON_MATCH_RULES:
        for i, s in enumerate(starts):
            if i in matched_starts or not s:
                continue
            found = [
                j for j, e in enumerate(ends)
                if j not in matched_ends and e and rule(s, e)
            ]
            if len(found) > 1:
                return None
            if found:
                matched_starts.add(i)
                matched_ends.add(found[0])
                pairs.append((i, found[0]))
    return pairs


def is_arrangement_swap(start_order: Any, end_order: Any) -> bool:
    """Did the same people trade left-right positions between the frames?

    ``start_order``/``end_order`` are the planner's left-to-right lists of
    who is visible in each frame. Code compares them instead of trusting the
    planner to NOTICE the swap while writing prose: it has now missed one
    twice on real orders (a couple who trade sides between the boat and the
    salt flat got 'step closer together, come to rest side by side'), and a
    swap staged as hold-steady makes the interpolator morph each person into
    the other.

    Only people present in BOTH frames count, at least two of them, and the
    matching must be unambiguous — anything else returns False so the
    pipeline behaves exactly as it did before.
    """
    if not isinstance(start_order, list) or not isinstance(end_order, list):
        return False
    starts = [_person_tokens(n) for n in start_order]
    ends = [_person_tokens(n) for n in end_order]
    if len(starts) < 2 or len(ends) < 2:
        return False

    pairs = _match_people(starts, ends)
    if pairs is None or len(pairs) < 2:
        return False
    # Same people, different left-to-right order.
    pairs.sort()
    end_positions = [j for _, j in pairs]
    return end_positions != sorted(end_positions)


# Where per-person choreography stops being writable at all. Every identity
# rule in this file assumes each mover can be named and given a path inside
# the beat budget (5s = one action, 10s = two beats) — roughly a mover per
# beat. On a real order (am-130826-pcfd: 45 frames, a 29-entry cast, frames
# holding 9-13 people) that assumption broke: plans staged seven exits and
# three entrances in ten seconds and Kling answered with ghost dissolves,
# bodies mushing into each other, and people vanishing — and a sampled
# 4-mover pair (two out, two in) already showed identity churn. The budget
# is 3 movers; the crowd limit exists because past ~4 people the epithets
# stop picking anyone out ("the teenage boy with short brown hair" in a
# 13-person frame points at nobody), so choreography of ANY size mushes.
_MOVER_BUDGET = 3
_CROWD_LIMIT = 4


def _head_count(value: Any) -> int:
    """A planner-reported people count, or 0 when absent or junk."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def unstageable_reason(
    start_order: Any, end_order: Any,
    start_heads: Any = None, end_heads: Any = None,
) -> Optional[str]:
    """WHY this pair can no longer be staged through the people, or None.

    ``"movers"``  — more than ``_MOVER_BUDGET`` people leave or arrive, so
    the choreography cannot fit the beat budget however few people are on
    screen. ``"crowd"`` — the frame is fuller than ``_CROWD_LIMIT`` and the
    group changes at all: past that size the epithets stop picking anyone
    out ("the teenage boy with short brown hair" in a 13-person frame points
    at nobody), so choreography of ANY size mushes.

    The reason exists so the explanation handed to the planner names the
    rule that actually fired. Deriving the wording separately let the two
    drift: a 6-person frame losing ONE person was told "1 people change
    between these frames — TOO MANY TO CHOREOGRAPH", which is both
    ungrammatical and the wrong cause (the crowd is the cause), and
    incoherent reasoning is exactly what erodes compliance on the pairs
    where the planner's own wording still matters.

    A big group that is the SAME on both sides (no movers, no swap) is
    stageable: hold-steady with small gestures works at any size — it is
    exits, entrances and crossings that fall apart in a crowd.

    ``start_heads``/``end_heads`` are the planner's census of every visible
    figure in each frame, tagged or not. They exist because tags list who
    MATTERS, not who is THERE: a real frame held ten people with two tagged,
    and the plan staged those two as if they were alone — the render
    collapsed the other eight. Ambiguity stays conservative, matching the
    swap detector: an unmatchable roster only counts the movers the
    headcount difference proves.
    """
    if not isinstance(start_order, list) or not isinstance(end_order, list):
        return None
    starts = [_person_tokens(n) for n in start_order]
    ends = [_person_tokens(n) for n in end_order]
    pairs = _match_people(starts, ends)
    if pairs is None:
        movers = abs(len(starts) - len(ends))
    else:
        movers = (len(starts) - len(pairs)) + (len(ends) - len(pairs))
    if movers > _MOVER_BUDGET:
        return "movers"
    crowd = max(
        len(starts), len(ends),
        _head_count(start_heads), _head_count(end_heads),
    )
    if crowd <= _CROWD_LIMIT:
        return None
    if movers > 0 or is_arrangement_swap(start_order, end_order):
        return "crowd"
    return None


def is_unstageable_pair(
    start_order: Any, end_order: Any,
    start_heads: Any = None, end_heads: Any = None,
) -> bool:
    """Can this pair no longer be staged through the people themselves?

    See ``unstageable_reason`` — this is that decision as a yes/no. Such a
    pair gets a deterministic camera transition instead of a staged prompt:
    repairing the wording cannot help when the staging itself is impossible.
    """
    return unstageable_reason(
        start_order, end_order, start_heads, end_heads
    ) is not None


def camera_shift_prompt(start_order: Any = None, end_order: Any = None) -> str:
    """The deterministic camera prompt for a pair, picked by its shape.

    Four shapes, all deterministic, all one continuous beat, none containing
    an exit marker (a camera prompt that tripped ``ends_offscreen`` would
    re-enter the very machinery it replaces):

    - end frame shows nobody: travel to the scene itself;
    - a group narrowing to ONE of its own people: everyone holds where they
      are and the camera drifts to that person — the staging the clip
      reviewer independently proposed for a real group-to-single pair whose
      staged prompt had rendered as a ghost dissolve;
    - people on BOTH sides: turn away from them and travel through the
      scenery, so the whip blur lands on walls rather than faces (a real
      generic-worded render dragged the group through the pan and mushed
      them mid-clip);
    - unknown rosters: the standing travel-and-settle wording.
    """
    if not isinstance(end_order, list):
        return CAMERA_SHIFT_MOTION_PROMPT
    if len(end_order) == 0:
        return CAMERA_SHIFT_TO_SCENE_PROMPT
    if (
        len(end_order) == 1
        and isinstance(start_order, list)
        and len(start_order) > 1
    ):
        # Only when the single end person is confidently one of the start
        # people — an anchor the camera can travel toward. A pseudo-entry
        # ("no people", a collective crowd tag) won't match and falls
        # through to the generic wording.
        starts = [_person_tokens(n) for n in start_order]
        if _match_people(starts, [_person_tokens(end_order[0])]):
            who = " ".join(str(end_order[0]).split())
            the = "" if who.lower().startswith("the ") else "the "
            return (
                f"Everyone stays where they are as the camera moves smoothly "
                f"and steadily across to {the}{who}, who fills the frame "
                f"exactly as the end frame shows."
            )
    if (
        len(end_order) >= 2
        and isinstance(start_order, list)
        and len(start_order) >= 1
    ):
        # Faces on both sides: give the transit neutral ground to cross.
        return CAMERA_SHIFT_THROUGH_SCENERY_PROMPT
    return CAMERA_SHIFT_MOTION_PROMPT


# The openings of every deterministic camera prompt this file can write.
# Recognition is deliberately strict-family: a hand-written camera wording
# still gets flagged as unstageable, which is a nudge to re-plan into the
# known-good deterministic wording, not a false alarm.
_CAMERA_PROMPT_OPENINGS = (
    "The camera moves smoothly and steadily across",
    "Everyone stays where they are as the camera",
    "Everyone holds still as the camera",
)


def is_camera_transition(motion_prompt: str) -> bool:
    """Is this one of the deterministic camera-family prompts?"""
    text = (motion_prompt or "").strip()
    return any(text.startswith(o) for o in _CAMERA_PROMPT_OPENINGS)


def _last_index(text: str, markers: tuple[str, ...]) -> int:
    """Where the last of these markers appears, or -1."""
    return max((text.rfind(m) for m in markers), default=-1)


def stages_a_crossing(motion_prompt: str) -> bool:
    """Does the prompt physically move people between sides — and finish?

    Two complete stagings: one person visibly passing another, or an exit
    past the camera FOLLOWED BY a walk back in. An exit on its own is not
    one, which is the whole point: the clip is pinned to the end frame, so a
    prompt that only empties the frame leaves the video model to invent the
    arrival — and a real restage came back as "walks forward past the camera
    and exits left, ending offscreen on the left" for a pair whose end frame
    shows that boy standing in the group.
    """
    text = (motion_prompt or "").lower()
    if any(marker in text for marker in _PASSING_MARKERS):
        return True
    return (
        any(marker in text for marker in _EXIT_MARKERS)
        and any(marker in text for marker in _RETURN_MARKERS)
    )


def ends_offscreen(motion_prompt: str) -> bool:
    """Does the prompt's LAST beat leave someone out of the frame?

    The clip has to land on the end frame, which shows the people in it —
    so "…and exits left, ending offscreen" describes a clip that cannot
    exist. The video model resolves it by cutting or teleporting.
    """
    text = (motion_prompt or "").lower()
    exit_at = _last_index(text, _EXIT_MARKERS)
    if exit_at < 0:
        return False
    return _last_index(text, _RETURN_MARKERS) < exit_at


_STORYBOARD_JSON_SHAPE = (
    "Return ONLY valid JSON with this exact shape:\n"
    "{\n"
    '  "project_title": str,\n'
    '  "style": str,                       // overall visual style sentence\n'
    '  "concept": str,                     // overall concept paragraph\n'
    '  "scenes": [str, ...],               // scene list\n'
    '  "frames": [\n'
    "    {\n"
    '      "id": "001",\n'
    '      "description": str,             // what happens in this frame\n'
    '      "image_prompt": str,            // full detailed image prompt\n'
    '      "negative_prompt": str,         // things to avoid\n'
    '      "duration_to_next": 5 | 10,     // clip seconds into the next frame; prefer 5\n'
    '      "sound_to_next": str            // ambient sound/SFX for that clip; no music, no speech\n'
    "    }, ...\n"
    "  ]\n"
    "}\n"
    "Do not include output_path or transitions; those are added later. "
    "Frame ids must be zero-padded 3-digit strings starting at 001. "
    "duration_to_next must be exactly 5 or 10."
)


class OpenAIClient:
    """Thin wrapper around the OpenAI SDK. Easy to swap models/endpoints."""

    # gpt-image models support these sizes; 16:9 1920x1080 is not native, so we
    # request the closest landscape size and then normalise with Pillow.
    _IMAGE_API_SIZE = "1536x1024"

    def __init__(self, config: Config) -> None:
        self.config = config
        self._client = None  # lazily created
        # Optional spending hook: (kind, usd, detail) -> None, set by the
        # Pipeline to its cost ledger. Every OpenAI call that COMPLETES
        # reports through here, which is why it sits on the client rather
        # than being dotted around the runner: the planner alone makes
        # condense/restage/reword calls the runner never sees.
        self.on_spend: Optional[Callable[[str, float, str], None]] = None

    # --- spending ----------------------------------------------------------- #
    def _spend(self, kind: str, usd: float, detail: str) -> None:
        """Report one paid call. Never raises: billing notes can't fail a run."""
        if self.on_spend is None or usd <= 0:
            return
        try:
            self.on_spend(kind, usd, detail)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not record OpenAI spend: %s", exc)

    def _note_text_usage(self, resp: Any, detail: str) -> None:
        """Price a chat/vision response from the token usage it reports."""
        if self.on_spend is None:
            return
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        pricing = self.config.pricing
        self._spend(
            KIND_TEXT,
            pricing.text(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            ),
            detail,
        )

    def _ensure_client(self):
        if self._client is None:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            from openai import OpenAI  # imported lazily

            self._client = OpenAI(api_key=api_key)
        return self._client

    def _retry(self, fn: Callable[[], T], description: str) -> T:
        return with_retries(
            fn,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=description,
        )

    # --- Mode A: edit an existing image into the target style --------------- #
    def style_image(self, src: Path, style_prompt: str, dst: Path) -> None:
        """Edit `src` into the styled look and write a normalised PNG to `dst`."""
        client = self._ensure_client()
        # Never upload the customer's file as-is: phone originals (iPhone MPO
        # HDR containers, unbaked EXIF rotation, 24MP frames) get rejected by
        # the image API as invalid_image_file. Decode once, send clean PNG.
        upload = prepare_image_for_upload(src)

        def _call(prompt: str) -> bytes:
            image = io.BytesIO(upload)
            image.name = f"{src.stem}.png"  # the SDK infers the mime type from this
            resp = client.images.edit(
                model=self.config.openai_image_model,
                image=image,
                prompt=prompt,
                size=self._IMAGE_API_SIZE,
            )
            # Billed per returned image, so a rewording retry that finally
            # succeeds pays for each attempt that came back — record here,
            # inside the attempt, not once per style_image call.
            self._spend(KIND_IMAGE, self.config.pricing.image(), src.name)
            return base64.b64decode(resp.data[0].b64_json)

        raw = self._image_with_moderation_recovery(
            _call, style_prompt, f"OpenAI style_image({src.name})"
        )
        self._save_normalized(raw, dst)

    # --- Mode B: generate an image from a text prompt ----------------------- #
    def generate_image(self, prompt: str, dst: Path) -> None:
        """Generate an image from `prompt` and write a normalised PNG to `dst`."""
        client = self._ensure_client()

        def _call(prompt: str) -> bytes:
            resp = client.images.generate(
                model=self.config.openai_image_model,
                prompt=prompt,
                size=self._IMAGE_API_SIZE,
            )
            self._spend(KIND_IMAGE, self.config.pricing.image(), dst.name)
            return base64.b64decode(resp.data[0].b64_json)

        raw = self._image_with_moderation_recovery(
            _call, prompt, "OpenAI generate_image"
        )
        self._save_normalized(raw, dst)

    # --- Moderation recovery ----------------------------------------------- #
    def _image_with_moderation_recovery(
        self, call: Callable[[str], bytes], prompt: str, description: str
    ) -> bytes:
        """Run `call(prompt)` with the usual backoff, rewording the prompt and
        re-entering when OpenAI's safety filter rejects it (transient errors
        are still handled by ``with_retries`` inside each attempt).

        An OUTPUT-stage rejection gets ONE targeted retry instead of the
        generic rewording loop. Generic rewording is the wrong tool there — it
        paraphrases and drops "risky detail" from wording the filter never
        objected to, which is why a real order burned four attempts and seven
        minutes per photo achieving nothing. But the prompt is not powerless
        either: it decides what gets DRAWN, and the classifier judges the
        drawing. The one prompt change that can move an output-stage block is
        therefore an instruction about the picture's CONTENT — see
        `_DEBRAND_INSTRUCTION`.
        """
        try:
            return with_reword_recovery(
                lambda p: self._retry(lambda: call(p), description),
                prompt,
                reword=self._reword_prompt_for_safety,
                attempts=self.config.moderation_reword_attempts,
                description=description,
            )
        except Exception as exc:  # noqa: BLE001 - classified here
            if not is_output_moderation(exc):
                raise
            logger.warning(
                "%s: the generated image was refused. Trying once more with "
                "licensed characters and brand marks explicitly excluded — "
                "styling asks for a CARTOON of the whole scene, so a photo "
                "taken among character merchandise is asking the model to "
                "draw that character.",
                description,
            )
            return self._retry(
                lambda: call(f"{prompt}\n\n{_DEBRAND_INSTRUCTION}"), description
            )

    def reword_motion_prompt(self, prompt: str) -> str:
        """Rewrite a clip motion prompt that a video content filter rejected.

        Handed to VideoClient.generate_clip as its `reword` callback, so a
        Kling content_policy_violation gets the same reword-and-retry recovery
        as image styling.
        """
        return self._reword_prompt_for_safety(prompt, system=_REWORD_MOTION_SYSTEM)

    # --- Who is in each frame (a first pass for a human to correct) --------- #
    def identify_people(
        self, frames: list[Path], cast: list[Character]
    ) -> list[list[dict[str, Any]]]:
        """Propose which cast members stand in each frame, and where.

        Returns one list per frame of ``{"id", "x", "y"}`` (fractions of the
        frame). This is a DRAFT for a human to correct in the panel, never
        an answer to act on directly — it is exactly the judgement the
        planner already gets wrong, so shipping it straight into planning
        would just automate the mistake. Its value is saving thirty photos'
        worth of clicking, not being right.

        Only ids from ``cast`` are ever returned; anyone the model invents is
        dropped, since a tag that names nobody in the cast cannot resolve to
        an epithet.
        """
        if not frames or not cast:
            return [[] for _ in frames]
        client = self._ensure_client()
        known = {c.id for c in cast}
        roster = "\n".join(f"- {c.id}: {c.epithet}" for c in cast)
        instruction = (
            f"Here are {len(frames)} frames of one short film, in order. For "
            "each frame, say which of these known people are visible in it "
            "and where each one stands.\n\n"
            f"The cast:\n{roster}\n\n"
            "Return ONLY JSON: {\"frames\": [{\"frame_index\": int, "
            "\"people\": [{\"id\": str, \"x\": number, \"y\": number}]}]}. "
            "frame_index is 1-based and every frame gets an entry (an empty "
            "people list when none of them are visible). x and y are the "
            "CENTRE OF THAT PERSON'S FACE as a fraction of the frame: x=0 is "
            "the left edge, x=1 the right edge, y=0 the top, y=1 the bottom. "
            "Use ONLY the ids listed above — never invent a new person, and "
            "never list the same id twice in one frame. These photographs "
            "were taken over a span of years: the same person changes "
            "clothes, hairstyle and age between frames, so identify them by "
            "face and build, and when you genuinely cannot tell two people "
            "apart, leave the less certain one out rather than guessing."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for i, fp in enumerate(frames, start=1):
            content.append({"type": "text", "text": f"Frame {i:03d}:"})
            content.append({
                "type": "image_url",
                "image_url": {
                    # Faces at "low" detail are what the planner already
                    # works from; matching it keeps this affordable.
                    "url": encode_image_data_url(fp, max_edge=_VISION_MAX_EDGE),
                    "detail": "low",
                },
            })

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": _IDENTIFY_PEOPLE_SYSTEM},
                    {"role": "user", "content": content},
                ],
                response_format=_json_schema_format(
                    "frame_people", _IDENTIFY_PEOPLE_SCHEMA
                ),
            )
            self._note_text_usage(resp, "identify_people")
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI identify_people")
        return _coerce_frame_people(json.loads(raw), len(frames), known)

    # --- Learning: watch the clip that came back --------------------------- #
    def review_clip(
        self,
        clip_frames: list[Path],
        motion_prompt: str,
        duration: int,
        *,
        start_frame: Optional[Path] = None,
        end_frame: Optional[Path] = None,
        user_note: str = "",
        lessons: Optional[list[str]] = None,
        global_context: str = "",
    ) -> dict[str, Any]:
        """Look at a rendered clip and say what actually happened.

        ``clip_frames`` are stills sampled across the clip in time order (see
        ``media/ffmpeg.sample_clip_frames``) — the text model cannot take
        video, and a handful of stills is enough to see sliding, morphing,
        sprinting and disguised cuts.

        Returns ``{observed, problems, matches_prompt, suggested_motion_prompt,
        suggested_duration, why}``. The suggested prompt is held to the same
        rules and word budget the planner works under (the full planning
        rulebook and the learned lessons are handed to the reviewer, and an
        over-budget suggestion is condensed exactly as a planned one would
        be), so accepting it can never smuggle in a prompt the planner itself
        would not have been allowed to write.

        ``user_note`` is what the human said, when they said anything: it
        points the reviewer at what bothered them instead of letting it grade
        the clip on its own priorities. Raises on API failure — the caller
        decides, and here that means keeping the note and reporting that the
        review did not happen.
        """
        if not clip_frames:
            raise ValueError("No frames were sampled from the clip.")
        client = self._ensure_client()

        # The reviewer must judge — and rewrite — under exactly the rules the
        # planner writes under, including everything learned so far.
        system = (
            _MODE_A_SYSTEM
            + lesson_prompt_block(lessons or [], SCOPE_MOTION)
            + "\n\n--- YOUR TASK NOW ---\n"
            + _CLIP_REVIEW_SYSTEM
        )

        instruction = (
            f"Clip length: {duration} seconds. Keep the replacement prompt "
            f"to the point: the {'one action' if duration == min(VALID_DURATIONS) else 'two beats'} "
            f"the clip can hold, and nothing else.\n\n"
            f"The motion prompt that produced this clip:\n{motion_prompt}\n"
        )
        if global_context.strip():
            # The renderer prepends this to every clip, so a replacement that
            # repeated it would double it in the next render — and the panel
            # writes the suggestion straight into the per-clip field.
            instruction += (
                "\nThe opening of that prompt is whole-movie context the "
                "renderer prepends to EVERY clip automatically:\n"
                f"{global_context.strip()}\n"
                "Your replacement prompt must cover only THIS clip's action "
                "and must not restate that context.\n"
            )
        if user_note.strip():
            instruction += (
                "\nWhat the human who watched it said (their complaint is the "
                "priority — confirm it against the stills, and say so if you "
                "cannot see what they describe):\n"
                f"{user_note.strip()}\n"
            )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]

        def _image(path: Path, label: str) -> None:
            content.append({"type": "text", "text": label})
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": encode_image_data_url(path, max_edge=_VISION_MAX_EDGE),
                    "detail": "low",
                },
            })

        if start_frame is not None and start_frame.exists():
            _image(start_frame, "START key frame (the clip had to begin here):")
        if end_frame is not None and end_frame.exists():
            _image(end_frame, "END key frame (the clip had to land here):")
        for i, frame in enumerate(clip_frames, start=1):
            _image(frame, f"Rendered clip, still {i} of {len(clip_frames)}:")

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                response_format=_json_schema_format(
                    "clip_review", _CLIP_REVIEW_SCHEMA
                ),
            )
            self._note_text_usage(resp, "review_clip")
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI review_clip")
        return self._coerce_review(json.loads(raw), motion_prompt, duration)

    def _coerce_review(
        self, data: dict[str, Any], motion_prompt: str, duration: int
    ) -> dict[str, Any]:
        """Normalise a review into a usable suggestion.

        Same division of labour as the planner: the model observes, code
        decides. A suggestion that is empty or at an impossible duration
        would otherwise reach the user as an "improvement" the pipeline
        itself would have rejected. Length is no longer policed here — the
        reviewer writes under the planner's own beat rules (it is handed
        _MODE_A_SYSTEM), and a suggestion is only ever a proposal the human
        applies by hand.
        """
        suggested = " ".join(
            str(data.get("suggested_motion_prompt", "") or "").split()
        )
        new_duration = self._coerce_duration(
            data.get("suggested_duration"), duration
        )
        problems = [
            " ".join(str(p).split()) for p in (data.get("problems") or [])
            if str(p).strip()
        ]
        return {
            "observed": " ".join(str(data.get("observed", "") or "").split()),
            "problems": problems[:6],
            "matches_prompt": bool(data.get("matches_prompt", False)),
            # Falling back to the original prompt keeps "apply the
            # suggestion" a no-op rather than a way to blank a transition.
            "suggested_motion_prompt": suggested or motion_prompt,
            "suggested_duration": new_duration,
            "why": " ".join(str(data.get("why", "") or "").split()),
            # True only when accepting it would actually change the render.
            "changes_clip": bool(
                (suggested and suggested != motion_prompt)
                or new_duration != duration
            ),
        }

    # --- Learning: one clip's feedback -> reusable rules -------------------- #
    def distill_lesson(
        self,
        note: str,
        *,
        verdict: str = "bad",
        motion_prompt: str = "",
        duration: int = 0,
        existing: Optional[list[str]] = None,
        observation: str = "",
        problems: Optional[list[str]] = None,
    ) -> list[dict[str, str]]:
        """Turn what was seen in a rendered clip into general rules.

        Returns ``[{"scope": ..., "text": ...}, ...]`` — possibly empty, which
        is a legitimate answer (see ``_DISTILL_LESSON_SYSTEM``): a note that
        teaches nothing general must not become a rule that then rides along
        with every future planning call.

        ``existing`` are the rules already in force; they are shown to the
        model so it doesn't mint a near-duplicate of a rule the planner
        already has. Raises on API failure — the caller decides what to do,
        and here that means saving the raw note anyway and telling the user
        the distillation did not happen.

        ``observation``/``problems`` are what the reviewer saw in the clip
        itself (see :meth:`review_clip`). Two witnesses beat one: the human
        knows which faults matter and is often brief ("looks weird"), while
        the reviewer has actually looked at the frames and can name the fault
        precisely — and a rule written from both is far likelier to be about
        a mechanism ("a position change with no walk between") than about a
        mood. Either may be absent, and a note alone still works exactly as
        it did before.
        """
        text = (note or "").strip()
        seen = (observation or "").strip()
        if not text and not seen:
            return []
        client = self._ensure_client()
        user = (
            f"The human's verdict: {'this clip is GOOD' if verdict == 'good' else 'this clip is BAD'}.\n"
        )
        user += (
            f"What they wrote:\n{text}\n" if text
            else "They gave no written note — the review below is the only "
                 "account of what happened.\n"
        )
        if seen:
            user += (
                "\nWhat a reviewer saw when it looked at the rendered clip "
                "frame by frame:\n"
                f"{seen}\n"
            )
            faults = [p for p in (problems or []) if str(p).strip()]
            if faults:
                user += (
                    "Faults it identified: "
                    + "; ".join(str(p).strip() for p in faults) + "\n"
                )
            user += (
                "Where the two accounts agree, the fault is real and worth a "
                "rule; where only the reviewer saw something, judge whether it "
                "generalises before writing it down.\n"
            )
        if motion_prompt.strip():
            user += (
                f"\nThe motion prompt that produced the clip"
                f"{f' ({duration}s)' if duration else ''}:\n"
                f"{motion_prompt.strip()}\n"
            )
        if existing:
            user += (
                "\nRules the planner already follows (do not restate these; "
                "return nothing if the note only repeats one):\n"
                + "\n".join(f"- {rule}" for rule in existing)
            )

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": _DISTILL_LESSON_SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format=_json_schema_format("lessons", _LESSON_SCHEMA),
            )
            self._note_text_usage(resp, "distill lesson")
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI distill_lesson")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        out: list[dict[str, str]] = []
        for item in (data.get("lessons") or [])[:2]:
            if not isinstance(item, dict):
                continue
            rule = " ".join(str(item.get("text", "")).split())[:MAX_LESSON_CHARS]
            scope = str(item.get("scope", SCOPE_MOTION))
            if not rule:
                continue
            out.append(
                {"scope": scope if scope in SCOPES else SCOPE_MOTION, "text": rule}
            )
        return out

    def _reword_prompt_for_safety(
        self, prompt: str, system: str = _REWORD_SYSTEM
    ) -> str:
        """Ask the text model to rewrite `prompt` so the safety filter accepts it.

        Falls back to appending an explicit safe-for-work clause if the rewrite
        call itself fails for any reason, so recovery never hard-stops here.
        """
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            )
            self._note_text_usage(resp, "reword prompt")
            reworded = (resp.choices[0].message.content or "").strip()
            if reworded:
                return reworded
        except Exception as exc:  # noqa: BLE001 - never let recovery die here
            logger.warning(
                "Prompt reword via text model failed (%s); falling back to a "
                "safe-for-work suffix",
                exc,
            )
        return prompt + _SAFE_SUFFIX

    def _save_normalized(self, raw_png: bytes, dst: Path) -> None:
        """Write raw bytes to a temp file, then normalise to target size."""
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".raw.png")
        tmp.write_bytes(raw_png)
        try:
            normalize_image(
                tmp, dst, self.config.target_width, self.config.target_height
            )
        finally:
            tmp.unlink(missing_ok=True)

    # --- Mode B: storyboard planning --------------------------------------- #
    def create_storyboard(
        self, idea: str, frame_count: int, default_duration: Optional[int] = None
    ) -> Storyboard:
        """Ask the text model to produce a structured storyboard for `idea`.

        If `frame_count` <= 0, the model chooses the number of frames that best
        fits the provided content instead of a fixed count.

        If `default_duration` is set (5 or 10), every clip is forced to that
        length; otherwise the model picks a per-clip duration (5 or 10) for each
        transition so the video can mix short and long clips.
        """
        client = self._ensure_client()
        system = _STORYBOARD_SYSTEM
        if self.config.avoid_text_only_frames:
            system += _STORYBOARD_NO_TEXT_FRAMES

        if frame_count and frame_count > 0:
            count_instruction = f"Create exactly {frame_count} key frames."
        else:
            count_instruction = (
                "Decide how many key frames best fit the content above and "
                "create that many (use as many as the material naturally needs; "
                "each beat/scene/section in the input should map to one or more "
                "frames). Do not pad to a fixed number."
            )
        user = (
            f"Video idea / source material:\n{idea}\n\n"
            f"{count_instruction} {_STORYBOARD_JSON_SHAPE}"
        )
        if default_duration:
            user += (
                f" Override: use duration_to_next = {default_duration} for every "
                "frame."
            )

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # No explicit temperature: the gpt-5 model line only accepts
                # the default, and the default is fine for this planning work.
                response_format=_json_schema_format("storyboard", _STORYBOARD_SCHEMA),
            )
            self._note_text_usage(resp, "create_storyboard")
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI create_storyboard")
        return self._assemble_storyboard(json.loads(raw), default_duration)

    # --- Mode A: plan smooth transitions from the styled frames ------------- #
    def analyze_frame_transitions(
        self,
        frames: list[Path],
        style_prompt: str,
        default_duration: Optional[int] = None,
        global_context: str = "",
        cast: Optional[list[Character]] = None,
        lessons: Optional[list[str]] = None,
        frame_people: Optional[list[Optional[list[str]]]] = None,
    ) -> tuple[list[tuple[str, int, str]], list[Character]]:
        """Vision-analyze consecutive frames and plan each clip between them.

        Returns ``(plans, cast)``: one ``(motion_prompt, duration,
        sound_prompt)`` per consecutive pair — exactly ``len(frames) - 1``
        items, in frame order — plus the movie's cast list. Durations are
        derived from the model's per-pair difficulty ratings (see
        ``_coerce_transition_plans``). The plans are always fully populated:
        any pair the model omits or returns malformed falls back to
        ``config.motion_prompt`` and the short duration, so the caller can
        rely on the length and types.

        ``cast`` is the storyboard's existing character list. When given, its
        epithets are FIXED for the model (used verbatim in the new prompts —
        the anchor that keeps a targeted re-plan naming people the same way
        the rest of the movie does), and the returned cast is the given one
        plus any genuinely new people the model saw. When None/empty the
        model builds the cast from scratch.

        ``lessons`` are the studio's learned rules (see feedback.py): short
        corrections written from clips that came back wrong, appended to the
        system prompt so the same mistake isn't planned twice. Empty = the
        prompt is byte-for-byte what it was before learning existed.

        ``frame_people`` is one entry per frame: the epithets standing in
        that frame, left to right, as tagged by a human — or None for a
        frame nobody has tagged. It is treated as FACT, not a hint: identity
        is what the planner is worst at (it decides from pixels whether the
        child in one frame is the child from the last one, and a wrong guess
        makes the video model morph them into each other), so where a human
        has said who is who, the model is told not to re-decide, and code
        uses the tags for the same/different and arrangement-swap rulings
        instead of the model's own reading.

        When ``default_duration`` is set (5 or 10) every clip is forced to that
        length instead of one chosen per pair.
        """
        n = len(frames)
        if n < 2:
            return [], list(cast or [])
        client = self._ensure_client()

        instruction = (
            f"Here are {n} key frames of a video, in order. Plan the {n - 1} "
            f"clips that animate each frame into the next. The intended visual "
            f"style is: {style_prompt}\n\n"
            "Return ONLY valid JSON with this exact shape:\n"
            "{\n"
            '  "transitions": [\n'
            '    {"pair_index": int, "difficulty": 1-5, '
            '"start_heads": int, "end_heads": int, '
            '"start_order": [str, ...], "end_order": [str, ...], '
            '"motion_prompt": str, "sound_prompt": str}, ...\n'
            "  ],\n"
            '  "characters": [{"id": str, "epithet": str}, ...]\n'
            "}\n"
            f"The transitions array must have exactly {n - 1} items, in frame "
            "order. pair_index anchors each item to its frames: the item with "
            "pair_index k animates frame k into frame k+1 (pair_index 1 = "
            "frame 001 into 002), and its motion_prompt must END at exactly "
            "what frame k+1 shows. start_heads and end_heads are a CENSUS of "
            "frame k and frame k+1: count every visible human figure, "
            "including background people, crowds, and anyone not in the cast "
            "or the confirmed tags (the tags say who matters; this count "
            "says how full the frame is — both are wanted, and the census is "
            "never limited by the tags). start_order and end_order list every "
            "person visible in frame k and in frame k+1 respectively, LEFT TO "
            "RIGHT as they appear in that image, each by the same epithet the "
            "motion_prompt uses — fill these in by LOOKING at the two images "
            "before writing the motion. Rate difficulty by how much the two "
            "frames differ, per the system instructions; clip lengths are "
            "derived from it, so budget the motion's beats accordingly (1-3: "
            "one action; 4-5: at most two)."
        )
        if default_duration:
            instruction += (
                f" Override: use duration = {default_duration} for every clip."
            )
        if cast:
            # A cast built by an earlier planning call (or hand-edited).
            # Restating it per call is what keeps epithets identical across
            # separately planned pairs — a targeted re-plan sees only its own
            # frames and would otherwise invent a fresh name for someone the
            # rest of the movie calls something else.
            roster = "\n".join(f"- {c.id}: {c.epithet}" for c in cast)
            instruction += (
                "\n\nThe movie's CAST LIST (fixed — use these epithets "
                "verbatim in every motion prompt, and return in `characters` "
                "only people missing from this list):\n" + roster
            )
        else:
            instruction += (
                "\n\nNo cast list exists yet — build `characters` per the "
                "CAST LIST rules."
            )
        # Human-confirmed identities, where they exist, outrank the model's
        # own reading of the frames.
        tags = list(frame_people or [])
        tags += [None] * (n - len(tags))
        if any(t is not None for t in tags):
            instruction += "\n\n" + _tagged_people_block(tags)
        if global_context.strip():
            # The user's whole-movie guidance (storyboard.global_motion_prompt).
            # It is prepended verbatim to every motion prompt at render time,
            # so plans must not contradict it and must not repeat it.
            instruction += (
                "\n\nContext that holds for the WHOLE video (the renderer "
                "already receives it with every clip, so respect it in every "
                "plan but do not restate it in your motion prompts): "
                f"{global_context.strip()}"
            )

        # Learned rules go in the SYSTEM prompt, right after the standing
        # instructions they correct: they are policy, not per-request
        # context, and a targeted re-plan of one pair must get them too.
        system = _MODE_A_SYSTEM + lesson_prompt_block(lessons or [], SCOPE_MOTION)

        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for i, fp in enumerate(frames, start=1):
            content.append({"type": "text", "text": f"Frame {i:03d}:"})
            content.append(
                {
                    "type": "image_url",
                    # "low" detail keeps the per-image token cost small; the model
                    # only needs the gist of each frame to plan the motion. The
                    # frames are downscaled before encoding so a long sequence
                    # stays within the API request-size limit.
                    "image_url": {
                        "url": encode_image_data_url(fp, max_edge=_VISION_MAX_EDGE),
                        "detail": "low",
                    },
                }
            )

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                response_format=_json_schema_format(
                    "transition_plans", _TRANSITIONS_SCHEMA
                ),
            )
            self._note_text_usage(resp, "analyze_frame_transitions")
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI analyze_frame_transitions")
        data = json.loads(raw)
        # A cast epithet is used in every prompt that mentions that person,
        # for the whole movie — so one anchored to this frame's clothing is
        # replaced by the durable wording before anything downstream sees it.
        swapped = repair_cast_epithets(data)
        if swapped:
            logger.info(
                "Re-anchored %d cast epithet(s) that named clothing instead "
                "of the person: %s", len(swapped), "; ".join(swapped),
            )
        plans = self._coerce_transition_plans(
            data, n - 1, default_duration, frame_people=tags,
        )
        return plans, _merge_cast(cast, data.get("characters"))

    def _coerce_transition_plans(
        self, data: dict[str, Any], count: int, default_duration: Optional[int],
        frame_people: Optional[list[Optional[list[str]]]] = None,
    ) -> list[tuple[str, int, str]]:
        """Normalise the model JSON into exactly `count` transition plans.

        The clip length is derived here from the model's per-pair difficulty
        rating, not taken from the model: difficulty 4-5 gets a long clip,
        1-3 a short one, and at most ``_LONG_CLIP_MAX_FRACTION`` of the pairs
        may be long (highest difficulty wins) so long clips stay the
        exception even if the model inflates its ratings.

        Motion prompts over the word budget for their derived duration are
        condensed here for the same reason durations are derived here:
        prompt-side guidance alone hasn't held on real plans.

        ``frame_people`` (human tags, one entry per frame) replaces the
        model's own left-to-right reading wherever a pair's two frames are
        both tagged: swap detection is only as good as the positions it is
        given, and a person who says who stands where is better evidence
        than a model that was asked to look.
        """
        items = _realign_by_pair_index(data.get("transitions") or [], count)

        def orders(i: int) -> tuple[Any, Any]:
            """(start_order, end_order) for pair i — tags first, model second."""
            tags = frame_people or []
            if i + 1 < len(tags) and tags[i] is not None and tags[i + 1] is not None:
                return tags[i], tags[i + 1]
            item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            return item.get("start_order"), item.get("end_order")

        # Pairs whose people trade sides.
        swaps = {i for i in range(count) if is_arrangement_swap(*orders(i))}
        long_indices = (
            set() if default_duration
            else self._select_long_clips(items, count, swaps)
        )
        plans: list[tuple[str, int, str]] = []
        for i in range(count):
            item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            motion = str(item.get("motion_prompt") or "").strip() or self.config.motion_prompt
            duration = default_duration or (
                max(VALID_DURATIONS) if i in long_indices else min(VALID_DURATIONS)
            )
            sound = str(item.get("sound_prompt") or "").strip()
            # Some pairs cannot be staged through the people AT ALL — more
            # movers than the beat budget can name, or a crowd whose roster
            # changes. The planner's choreography for them is noise however
            # well it is worded (verified on real renders: ghost dissolves,
            # mushed bodies, vanished people), so it is not repaired, it is
            # REPLACED with the deterministic camera transition, and the
            # repair machinery below is skipped — a template cannot end
            # offscreen.
            #
            # ALWAYS the short clip (user call, 2026-08-14), whatever the
            # difficulty rating or a --duration override says: this prompt is
            # ONE continuous beat — the camera travels and settles — and ten
            # seconds of it is a slack shot at twice the price. Difficulty
            # rates how hard the pair is to STAGE, which is exactly the
            # question this branch has already answered with "don't".
            if is_unstageable_pair(
                *orders(i), item.get("start_heads"), item.get("end_heads"),
            ):
                logger.info(
                    "Pair %d cannot be staged through its people (too many "
                    "movers or too crowded); using a %ds camera transition.",
                    i + 1, min(VALID_DURATIONS),
                )
                plans.append((
                    camera_shift_prompt(*orders(i)),
                    min(VALID_DURATIONS),
                    sound,
                ))
                continue
            if i in swaps and not stages_a_crossing(motion):
                motion = self._restage_swapped_pair(motion, duration, *orders(i))
            # An exit with no return is HALF a staging, and the clip is pinned
            # to the end frame — so a prompt whose last beat empties the frame
            # describes a clip that cannot exist and the model answers with a
            # cut. Enforced here rather than left to _MODE_A_SYSTEM because a
            # full re-plan of a real 44-pair movie came back with the SAME 29
            # pairs ending on an exit.
            if ends_offscreen(motion):
                motion = self._complete_offscreen_ending(
                    motion, duration, orders(i)[1]
                )
            # NOTHING ships with an offscreen ending. Every rewrite above can
            # decline, and on hard pairs they do — a group of nine cannot walk
            # out and back in inside one clip. Move the camera instead of the
            # people: a narrow, deliberate exception to the no-camera-moves
            # rule, because the real alternative here is a cut, not a better
            # staging.
            if ends_offscreen(motion):
                heads = max(
                    len(o) if isinstance(o, list) else 0 for o in orders(i)
                )
                logger.warning(
                    "Pair %d cannot be staged back into frame (%s); falling "
                    "back to a camera move to the new setting.",
                    i + 1,
                    f"{heads} people" if heads else "people not reported",
                )
                motion = CAMERA_SHIFT_MOTION_PROMPT
            plans.append((motion, duration, sound))
        return plans

    def _restage_swapped_pair(
        self, prompt: str, duration: int,
        start_order: Any, end_order: Any,
    ) -> str:
        """Rewrite a swapped pair's prompt so the people actually change sides.

        Falls back to the original prompt on any failure: a badly staged clip
        still renders, and planning must never hard-stop over it.
        """

        def _names(order: Any) -> str:
            if not isinstance(order, list):
                return "(not reported)"
            return ", ".join(str(n) for n in order) or "(nobody listed)"

        logger.info(
            "Pair's people trade sides but the motion prompt keeps them in "
            "place; restaging it (left-to-right %s -> %s)",
            _names(start_order), _names(end_order),
        )
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": _RESTAGE_SWAP_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Clip length: {duration} seconds. "
                            f"Keep it to the point: {'one action' if duration == min(VALID_DURATIONS) else 'two beats'}, no scene-setting.\n"
                            f"Start frame, left to right: {_names(start_order)}\n"
                            f"End frame, left to right: {_names(end_order)}\n\n"
                            f"{prompt}"
                        ),
                    },
                ],
            )
            self._note_text_usage(resp, "restage swapped pair")
            restaged = (resp.choices[0].message.content or "").strip()
            # Only accept a rewrite that actually fixed the thing it was
            # called for AND still lands on the end frame. An exit with no
            # return is half a staging: it passed the old check and produced
            # "walks forward past the camera and exits left, ending offscreen"
            # for a pair whose end frame shows that boy standing in the group.
            if restaged and stages_a_crossing(restaged):
                if not ends_offscreen(restaged):
                    return restaged
                logger.warning(
                    "Restage ended with someone offscreen, but the clip has to "
                    "land on the end frame; keeping the original"
                )
            else:
                logger.warning(
                    "Restage came back without any crossing; keeping the original"
                )
        except Exception as exc:  # noqa: BLE001 - keep planning alive
            logger.warning(
                "Restaging the swapped pair failed (%s); keeping the original",
                exc,
            )
        return prompt

    def _complete_offscreen_ending(
        self, prompt: str, duration: int, end_order: Any = None,
    ) -> str:
        """Rewrite a prompt whose last beat leaves someone out of the frame.

        Accepts the rewrite only if it actually lands the clip — a rewrite
        that still ends offscreen has fixed nothing, and taking it would
        replace a known-bad prompt with a differently-worded bad one while
        clearing the warning the panel shows. Falls back to the original on
        any failure: a badly staged clip still renders, and planning must
        never hard-stop over it.
        """
        names = (
            ", ".join(str(n) for n in end_order)
            if isinstance(end_order, list) and end_order
            else "(not reported)"
        )
        logger.info(
            "Motion prompt ends with someone offscreen but the clip has to "
            "land on the end frame; rewriting the ending (end frame, left to "
            "right: %s)", names,
        )
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": _COMPLETE_ENDING_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Clip length: {duration} seconds. "
                            f"Keep it to the point: {'one action' if duration == min(VALID_DURATIONS) else 'two beats'}, no scene-setting.\n"
                            f"End frame, left to right: {names}\n\n"
                            f"{prompt}"
                        ),
                    },
                ],
            )
            self._note_text_usage(resp, "complete offscreen ending")
            fixed = (resp.choices[0].message.content or "").strip()
            if fixed and not ends_offscreen(fixed):
                return fixed
            logger.warning(
                "The rewrite still ended offscreen; keeping the original"
            )
        except Exception as exc:  # noqa: BLE001 - keep planning alive
            logger.warning(
                "Completing the offscreen ending failed (%s); keeping the "
                "original", exc,
            )
        return prompt

    def _select_long_clips(
        self, items: list[Any], count: int,
        swaps: Optional[set[int]] = None,
    ) -> set[int]:
        """Pick which pairs get the long duration from their difficulty ratings.

        Difficulty >= 4 gets 10 seconds. EVERY pair that earns it does, with
        no quota (user call, 2026-08-14): the old ceiling capped a movie at
        half long clips, so on a hard order it demoted genuinely hard pairs
        to 5s — where they teleport — and, because the ceiling was applied
        per planning call, re-planning a handful of pairs could hand the same
        pair a different length depending on which others were selected
        alongside it. Rating honestly is the planner's job; if too much comes
        back long, that is a rating problem to fix in the rubric.

        A pair whose people trade sides is forced to at least 4 whatever the
        planner rated it: the staging that keeps them from morphing (walk out
        past the camera, walk back in one at a time) cannot play out in five
        seconds, and the planner has rated real swaps a 2.
        """
        swaps = swaps or set()

        def rating(i: int) -> int:
            item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            try:
                d = int(item.get("difficulty"))
            except (TypeError, ValueError):
                d = 3  # unrated -> ordinary pair
            d = min(5, max(1, d))
            return max(d, 4) if i in swaps else d

        return {i for i in range(count) if rating(i) >= 4}

    def _coerce_duration(self, value: Any, fallback: int) -> int:
        """Return `value` if it is a valid clip duration, else `fallback`."""
        try:
            d = int(value)
        except (TypeError, ValueError):
            return fallback
        return d if d in VALID_DURATIONS else fallback

    def _assemble_storyboard(
        self, data: dict[str, Any], default_duration: Optional[int] = None
    ) -> Storyboard:
        """Normalise the model JSON and attach output paths + transitions.

        Each transition takes the start frame's ``duration_to_next`` (5 or 10),
        unless ``default_duration`` forces every clip to one length.
        """
        frames: list[Frame] = []
        durations: list[int] = []
        sound_prompts: list[str] = []
        for i, fr in enumerate(data.get("frames", []), start=1):
            fid = str(fr.get("id") or f"{i:03d}").zfill(3)
            frames.append(
                Frame(
                    id=fid,
                    description=fr.get("description", ""),
                    image_prompt=fr.get("image_prompt", ""),
                    negative_prompt=fr.get("negative_prompt", ""),
                    output_path=f"generated_frames/{fid}.png",
                )
            )
            durations.append(
                default_duration
                or self._coerce_duration(
                    fr.get("duration_to_next"), self.config.duration
                )
            )
            sound_prompts.append(str(fr.get("sound_to_next", "") or ""))

        transitions: list[Transition] = []
        for idx, (a, b) in enumerate(zip(frames, frames[1:])):
            tid = f"{a.id}_to_{b.id}"
            transitions.append(
                Transition(
                    id=tid,
                    start_frame=a.output_path,
                    end_frame=b.output_path,
                    motion_prompt=self.config.motion_prompt,
                    duration=durations[idx],
                    sound_prompt=sound_prompts[idx],
                    output_path=f"clips/{tid}.mp4",
                )
            )

        return Storyboard(
            project_title=data.get("project_title", "Untitled Project"),
            style=data.get("style", self.config.scratch_style_prompt),
            # Fallback length used only when a transition has no duration of its
            # own; individual clips can differ (see Transition.duration).
            duration_per_clip=default_duration or self.config.duration,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            concept=data.get("concept", ""),
            scenes=list(data.get("scenes", [])),
            frames=frames,
            transitions=transitions,
        )
