"""OpenAI client (isolated). Image generation/editing + storyboard planning."""
from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from ..config import Config
from ..constants import VALID_DURATIONS
from ..media.images import (
    encode_image_data_url,
    normalize_image,
    prepare_image_for_upload,
)
from ..logging_setup import logger
from ..models import Character, Frame, Storyboard, Transition
from ..retry import with_retries, with_reword_recovery

T = TypeVar("T")

# Longest edge (px) for frames sent to the vision model. Low-detail vision uses
# ~512px anyway, so anything bigger is pure upload weight.
_VISION_MAX_EDGE = 768

# Fallback ceiling on how many of a movie's clips may get the long (10s)
# duration; `config.long_clip_max_fraction` overrides it. The planner rates
# each pair's difficulty and code derives durations from it
# (_coerce_transition_plans); the cap keeps the movie from turning long and
# slow when the model inflates its ratings — real plans have come back all-5s
# and all-10s under prompt-side guidance alone. Raised from 1/3 to 1/2 on
# 2026-07-26: hard pairs were being demoted to 5s by the cap and teleporting.
_LONG_CLIP_MAX_FRACTION = 0.5

# Hard word budgets for motion prompts, by clip duration. Kling drops or fakes
# beats it can't fit (a real 84-word 5s prompt rendered as a whip-pan blur —
# a disguised cut), and prompt-side guidance alone doesn't hold the line: a
# real plan under the beat-budget rule still wrote 79-113 words for every 5s
# clip. Over-budget prompts are condensed by a targeted text call in
# _coerce_transition_plans, mirroring how durations are derived in code.
_MOTION_WORD_LIMITS = {min(VALID_DURATIONS): 35, max(VALID_DURATIONS): 60}

# Condense an over-budget motion prompt down to what the clip can hold.
_CONDENSE_MOTION_SYSTEM = (
    "You condense motion prompts for an image-to-video model that "
    "interpolates between two given frames (the START and END of one clip). "
    "The prompt you receive stages more action than the clip's length can "
    "hold; the video model reacts by dropping or faking beats — a whip-pan "
    "blur, a morphed or swapped person. Rewrite the prompt to fit the stated "
    "word limit, keeping only: (1) the single most essential subject action "
    "(a 5-second clip holds exactly ONE continuous action; 10 seconds at "
    "most two beats); (2) identity phrasing: same-person wording ('the same "
    "little boy, now ...') with singular he/she — never 'they' — and, when "
    "several people appear, each person's short distinguishing epithet ('the "
    "bald man in pink sunglasses', 'the younger man with brown hair'). NEVER "
    "shorten an epithet down to a feature two people share — cutting 'the "
    "bald man in pink sunglasses' to 'the bald man' when the film has two "
    "bald men saves three words and makes the clip animate the wrong person. "
    "Cut scenery before identity. Keep any left/right position or direction "
    "wording too ('the man on the right', 'turns left'): that is how the "
    "model knows which figure to move where. Never collapse named "
    "individuals into a collective 'they' or 'the couple', which makes the "
    "video model blend their identities, and if the prompt refers to anyone "
    "by name or by relationship words ('the son', 'the father', 'mom'), "
    "replace those with visible-appearance epithets — the video model knows "
    "only what the frames show; (3) any exit-and-entrance or "
    "position-swap staging (people leaving frame and re-entering, one "
    "crossing past another) — that staging is what prevents the model from "
    "morphing one person into another, so condense around it, never into "
    "everyone holding still; (4) a brief final clause "
    "landing on the same end state the original ends on. Cut secondary "
    "gestures, scenery and wardrobe inventories, and camera directions — the "
    "two frames already carry the visual detail. Collapse route-spanning "
    "travel into the ARRIVAL: 'strolls along the meadow trail until they "
    "stop beside the bench' becomes 'takes the last few unhurried steps to "
    "the bench and settles onto it'. A clip cannot cover a whole route in "
    "its length and the video model compensates by sprinting the people "
    "across the ground, so keep the destination and an unhurried pace but "
    "drop the journey. Every remaining action must "
    "be performed by a visible person under their own power: remove actions "
    "done to the subject by anyone off-screen (being lifted, carried, set "
    "down). No editing terms (crossfade, dissolve, morph, transition). "
    "Present tense. Return ONLY the condensed motion prompt, no preamble or "
    "quotes."
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
    "actions, setting and staging, but remove or rephrase anything the filter "
    "could misread as sexual, violent, or otherwise unsafe: make physical "
    "affection unmistakably innocent and familial (a warm hug, leaning on a "
    "shoulder), avoid mentioning beds/lying together/body parts, and replace "
    "loaded words (e.g. 'shot' -> 'scene'). BABIES AND CHILDREN trip the "
    "filter hardest: when a baby or child is in the scene, drop wording about "
    "their body being physically handled — lifted, carried, bounced, lowered, "
    "settled, laid down — and drop bed/crib/blanket mentions; describe the "
    "moment at scene level instead (e.g. 'the parent and baby share a joyful "
    "moment as the scene moves to the cozy room'). The two input images "
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
    "For EACH consecutive pair, describe what HAPPENS IN THE WORLD to carry the "
    "start frame into the end frame, so the result feels like a scene from a "
    "real movie — never like a slideshow transition. Work in this priority "
    "order: "
    "1. SUBJECT ACTION FIRST. Whenever possible, bridge the frames through the "
    "characters/subjects themselves: they move, turn, walk, gesture, react, "
    "grow, or change in a way that plausibly arrives at the end frame (e.g. "
    "'the boy pushes off the couch and walks toward the sunlit doorway, "
    "stepping into the garden as the room gives way to open sky'). Ground "
    "every action in what is actually visible in the two frames. "
    "2. WORLD FLOW SECOND. If the setting or time changes between the frames, "
    "stage a continuous in-world handover: light shifts, weather rolls in, a "
    "foreground element passes across and reveals the new setting, the "
    "environment transforms around a steady subject. Introduce the next scene "
    "as a continuation of the previous one, not a cut to somewhere else. "
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
    "same prominent subject, stage it as either (a) the subject stepping "
    "out of frame close past the camera and re-entering in the new setting, "
    "or (b) the subject holding roughly steady while the world transforms "
    "around them (light, background, and surroundings flow into the new "
    "scene). Describe visible travel only when one of the two frames "
    "already shows the subject small or far away. "
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
    "exit-past-camera plus re-entry, or hold the subject steady while the "
    "world transforms around them. "
    "NO OFF-SCREEN HANDS: every action must be performed by a person "
    "visible in the frames, under their own power. NEVER write actions done "
    "TO the subject by an unseen agent — 'is lifted out', 'is carried', "
    "'is gently set down', 'as if being lifted' — when no such person is "
    "visible: the video model cannot render the invisible helper, so the "
    "subject levitates through the air instead. If a "
    "transition seems to need an off-screen handler, restage it: the "
    "subject moves under their own power (climbs down, stands up, turns) "
    "or holds steady while the world transforms around them. "
    "LAND ON THE END FRAME: each clip stops exactly at its end frame, so the "
    "FINAL clause of every motion_prompt must describe the subject already "
    "in the end frame's pose, place, and activity, and every action the "
    "prompt starts must be finished by then. Before writing, look at the END "
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
    "above (visible appearance only, distinguishing, 2-5 words), and "
    "prefer traits that stay stable across the whole movie — hair or its "
    "absence, build, age band, eyewear — over clothing that changes "
    "between scenes. When a person's look in one pair has moved on from "
    "their epithet (a new outfit, an older age), keep the epithet and add "
    "the change inline: 'the bald man in pink sunglasses, now in a winter "
    "coat'. "
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
    "or have one person visibly cross in front of or behind the other. A "
    "position swap is at least difficulty 4; a swap combined with a setting "
    "change is 5. "
    "BEAT BUDGET: rate the pair's difficulty BEFORE writing the motion, "
    "because the rating sets the clip's length and the length sets how much "
    "can happen. Difficulty 1-3 = a 5-second clip = exactly ONE continuous "
    "action, ONE sentence, AT MOST 35 WORDS ('she crosses the kitchen and "
    "sets the tray on the table' / 'the toddler runs to the patio chair and "
    "settles onto it'). Difficulty 4-5 = up to 10 seconds = at most TWO "
    "beats, two sentences, AT MOST 60 WORDS. These word caps are hard "
    "limits: an over-cap prompt is mechanically condensed before rendering "
    "and may lose the wrong detail, so stay under the cap yourself. A "
    "prompt with more beats than the clip can hold does not get compressed "
    "— the model drops or fakes the beats, typically by swapping in a "
    "different-looking subject mid-clip. "
    "COUNT THE BEATS BEFORE YOU WRITE — the word cap will NOT do it for "
    "you. A beat is any distinct action a subject performs, and a prompt "
    "can sit comfortably under the word cap while stacking far too many of "
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
    "derived from this rating "
    "(easy pairs get short, pacy clips; only the hardest get long ones), so "
    "do not inflate it. "
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
                    "motion_prompt": {"type": "string"},
                    "sound_prompt": {"type": "string"},
                },
                "required": [
                    "pair_index", "difficulty", "motion_prompt", "sound_prompt",
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
                },
                "required": ["id", "epithet"],
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


def _motion_word_limit(duration: int) -> int:
    """Word budget for a motion prompt of `duration` seconds.

    Unknown durations get the most permissive budget rather than a guess —
    the budget exists to catch clearly overloaded prompts, not to nickel-
    and-dime borderline ones.
    """
    return _MOTION_WORD_LIMITS.get(duration, max(_MOTION_WORD_LIMITS.values()))


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
        """
        return with_reword_recovery(
            lambda p: self._retry(lambda: call(p), description),
            prompt,
            reword=self._reword_prompt_for_safety,
            attempts=self.config.moderation_reword_attempts,
            description=description,
        )

    def reword_motion_prompt(self, prompt: str) -> str:
        """Rewrite a clip motion prompt that a video content filter rejected.

        Handed to VideoClient.generate_clip as its `reword` callback, so a
        Kling content_policy_violation gets the same reword-and-retry recovery
        as image styling.
        """
        return self._reword_prompt_for_safety(prompt, system=_REWORD_MOTION_SYSTEM)

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
            '"motion_prompt": str, "sound_prompt": str}, ...\n'
            "  ],\n"
            '  "characters": [{"id": str, "epithet": str}, ...]\n'
            "}\n"
            f"The transitions array must have exactly {n - 1} items, in frame "
            "order. pair_index anchors each item to its frames: the item with "
            "pair_index k animates frame k into frame k+1 (pair_index 1 = "
            "frame 001 into 002), and its motion_prompt must END at exactly "
            "what frame k+1 shows. Rate difficulty by how much the two frames "
            "differ, per the system instructions; clip lengths are derived "
            "from it, so budget the motion's beats accordingly (1-3: one "
            "action; 4-5: at most two)."
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
                    {"role": "system", "content": _MODE_A_SYSTEM},
                    {"role": "user", "content": content},
                ],
                response_format=_json_schema_format(
                    "transition_plans", _TRANSITIONS_SCHEMA
                ),
            )
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI analyze_frame_transitions")
        data = json.loads(raw)
        plans = self._coerce_transition_plans(data, n - 1, default_duration)
        return plans, _merge_cast(cast, data.get("characters"))

    def _coerce_transition_plans(
        self, data: dict[str, Any], count: int, default_duration: Optional[int]
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
        """
        items = _realign_by_pair_index(data.get("transitions") or [], count)
        long_indices = (
            set() if default_duration
            else self._select_long_clips(items, count)
        )
        plans: list[tuple[str, int, str]] = []
        for i in range(count):
            item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            motion = str(item.get("motion_prompt") or "").strip() or self.config.motion_prompt
            duration = default_duration or (
                max(VALID_DURATIONS) if i in long_indices else min(VALID_DURATIONS)
            )
            if len(motion.split()) > _motion_word_limit(duration):
                motion = self._condense_motion_prompt(motion, duration)
            sound = str(item.get("sound_prompt") or "").strip()
            plans.append((motion, duration, sound))
        return plans

    def _condense_motion_prompt(self, prompt: str, duration: int) -> str:
        """Rewrite an over-budget motion prompt down to the clip's word budget.

        Falls back to the original prompt if the rewrite call fails or comes
        back no shorter — an overloaded prompt still renders (badly), so this
        must never hard-stop planning.
        """
        limit = _motion_word_limit(duration)
        logger.info(
            "Motion prompt is %d words for a %ds clip (budget %d); condensing",
            len(prompt.split()), duration, limit,
        )
        try:
            client = self._ensure_client()
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": _CONDENSE_MOTION_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Clip length: {duration} seconds. "
                            f"Word limit: {limit}.\n\n{prompt}"
                        ),
                    },
                ],
            )
            condensed = (resp.choices[0].message.content or "").strip()
            if condensed and len(condensed.split()) < len(prompt.split()):
                return condensed
        except Exception as exc:  # noqa: BLE001 - keep planning alive
            logger.warning(
                "Condensing motion prompt failed (%s); keeping the original",
                exc,
            )
        return prompt

    def _select_long_clips(self, items: list[Any], count: int) -> set[int]:
        """Pick which pairs get the long duration from their difficulty ratings.

        Difficulty >= 4 qualifies; if more qualify than
        ``config.long_clip_max_fraction`` allows, only the highest-rated
        (earliest on ties) keep the long clip.
        """
        def rating(i: int) -> int:
            item = items[i] if i < len(items) and isinstance(items[i], dict) else {}
            try:
                d = int(item.get("difficulty"))
            except (TypeError, ValueError):
                return 3  # unrated -> ordinary pair, short clip
            return min(5, max(1, d))

        candidates = [i for i in range(count) if rating(i) >= 4]
        fraction = getattr(
            self.config, "long_clip_max_fraction", _LONG_CLIP_MAX_FRACTION
        )
        cap = math.ceil(count * fraction)
        candidates.sort(key=lambda i: (-rating(i), i))
        return set(candidates[:cap])

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
