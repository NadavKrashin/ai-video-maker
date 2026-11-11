"""Named video-model presets — pick a Kling, get a coherent request shape.

Every fal model names its inputs differently: the start frame is ``image_url``
on Kling 2.5 and ``start_image_url`` on Kling 3, the end frame is
``tail_image_url`` on some and ``end_image_url`` on others, and only the newer
ones accept character elements at all. Those were separate config keys the
user had to keep consistent by hand, and getting one wrong does not fail
loudly — it silently drops the end frame, and the movie stops chaining.

A preset ties them together. ``config.video_model`` names one, the preset
fills in every ``fal_*`` field the config does not set explicitly, and one
choice can no longer be half-applied.

The preset also carries `supports_elements`, which is what answers "send the
cast's faces with this render?" — 2.5 cannot take them, 3 can. That decision
belongs to the MODEL, not to a separate switch somebody has to remember to
flip in step with it.

VERIFICATION: `verified` marks a preset whose request shape this project has
actually rendered with: 2.5, and since the first bake-off (2026-11-11) Kling
3.0 Standard, 3.0 Pro and O3 Pro — each accepted the request, returned no
audio and landed on the end frame. That says the REQUEST works, not that the
model is better; face elements and Kling 4 are still untried. So the first
render on a new model should be a single clip — `bakeoff <project>
--clip ID`, which renders one pair on several models side by side without
touching the movie — before a whole movie is committed to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from typing import Optional


@dataclass(frozen=True)
class VideoModel:
    """One selectable video model and the request shape it expects."""

    key: str
    label: str
    model_id: str
    start_frame_field: str
    end_frame_field: str
    usd_per_second: float
    # Can this model be handed reference portraits that pin a character's
    # identity through the shot? False means the cast faces are never sent,
    # however the element fields happen to be configured.
    supports_elements: bool = False
    elements_field: str = ""
    element_image_field: str = ""
    # Some models want each element to carry extra views beside the frontal
    # image and REJECT one without them (Kling 3: "At least one image is
    # required"). When set, the same portrait is sent under this field too.
    element_reference_field: str = ""
    max_elements: int = 0
    element_prompt_template: str = ""
    # Request arguments this model needs to behave like the rest of the
    # pipeline expects — most importantly switching OFF native audio, which
    # Kling 3 generates by default and bills 50% more for. The pipeline adds
    # its own SFX in the audio step.
    extra_arguments: dict[str, Any] = field(default_factory=dict)
    # Has this project actually rendered with this request shape?
    verified: bool = False
    # A preset that must no longer be picked, kept only so a project already
    # pinned to it still LOADS (an unknown name is a ConfigError, which would
    # break that project's status page). Rendering with one is refused.
    retired: bool = False
    # Where the price came from and when — these drift, and a stale number
    # here makes every cost comparison quietly wrong.
    price_note: str = ""
    note: str = ""

    def as_config_fields(self) -> dict[str, object]:
        """The ``fal_*`` config values this preset implies."""
        return {
            "fal_model_id": self.model_id,
            "fal_start_frame_field": self.start_frame_field,
            "fal_end_frame_field": self.end_frame_field,
            "fal_elements_field": self.elements_field,
            "fal_element_image_field": (
                self.element_image_field or "frontal_image_url"
            ),
            "fal_element_reference_field": self.element_reference_field,
            "fal_max_elements": self.max_elements,
            "fal_element_prompt_template": self.element_prompt_template,
            "fal_model_arguments": dict(self.extra_arguments),
        }


# The order here is the order the panel offers them in: the model in
# production use first, then alternatives by price.
#
# Prices and field names below were read from each model's own fal page on
# 2026-11-11 (fal's /pricing page disagreed with the model pages for Kling 3
# and O3; the model pages were taken as authoritative). All are "audio off"
# prices, which is what `extra_arguments` asks for.
VIDEO_MODELS: tuple[VideoModel, ...] = (
    VideoModel(
        key="kling-2.5-turbo-pro",
        label="Kling 2.5 Turbo Pro",
        model_id="fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
        start_frame_field="image_url",
        end_frame_field="tail_image_url",
        usd_per_second=0.07,
        supports_elements=False,
        verified=True,
        price_note="fal model page, read 2026-11-11: $0.35 per 5s clip, $0.07 per extra second",
        note=(
            "The long-standing default. Cheapest, and every movie this studio "
            "has delivered was rendered with it. No character elements: faces "
            "are held only by the two frames and the prompt, which is why "
            "people distort when they travel between them."
        ),
    ),
    VideoModel(
        key="kling-3-standard",
        label="Kling 3.0 Standard",
        model_id="fal-ai/kling-video/v3/standard/image-to-video",
        start_frame_field="start_image_url",
        end_frame_field="end_image_url",
        usd_per_second=0.084,
        supports_elements=True,
        elements_field="elements",
        element_image_field="frontal_image_url",
        element_reference_field="reference_image_urls",
        max_elements=1,
        element_prompt_template="@Element{index}",
        extra_arguments={"generate_audio": False},
        price_note="fal model page, read 2026-11-11: $0.084/s audio off ($0.126/s with audio)",
        verified=True,
        note=(
            "The cheaper tier of Kling 3: 20% more per second than 2.5. "
            "Rendered from here on 2026-11-11 and it lands on the end frame "
            "— but it returns 1280x720, not 1080p, so it is softer than "
            "every other model here. Face elements not yet tried."
        ),
    ),
    VideoModel(
        key="kling-3-pro",
        label="Kling 3.0 Pro",
        model_id="fal-ai/kling-video/v3/pro/image-to-video",
        start_frame_field="start_image_url",
        end_frame_field="end_image_url",
        usd_per_second=0.112,
        supports_elements=True,
        elements_field="elements",
        element_image_field="frontal_image_url",
        element_reference_field="reference_image_urls",
        max_elements=1,
        element_prompt_template="@Element{index}",
        extra_arguments={"generate_audio": False},
        price_note="fal model page, read 2026-11-11: $0.112/s audio off ($0.168/s with audio)",
        verified=True,
        note=(
            "Newer model, 1080p, and it accepts ONE facial element — the "
            "cast face reference — to hold a character's identity through "
            "the shot. 60% more per second than 2.5. Rendered from here on "
            "2026-11-11 and it lands on the end frame; sending a face "
            "element together with the end frame has not been tried yet, so "
            "compare one clip first on a tagged project."
        ),
    ),
    VideoModel(
        key="kling-o3-pro",
        label="Kling O3 Pro",
        model_id="fal-ai/kling-video/o3/pro/image-to-video",
        start_frame_field="image_url",
        end_frame_field="end_image_url",
        usd_per_second=0.112,
        supports_elements=False,
        extra_arguments={"generate_audio": False},
        price_note="fal model page, read 2026-11-11: $0.112/s audio off ($0.14/s with audio)",
        verified=True,
        note=(
            "Kling's first-frame-to-last-frame model, 1080p. Same price as "
            "3.0 Pro; no face elements on this endpoint. Rendered from here "
            "on 2026-11-11 and it lands on the end frame."
        ),
    ),
    VideoModel(
        key="kling-4",
        label="Kling 4.0 (1080p)",
        model_id="kling/v4/image-to-video",
        start_frame_field="start_image_url",
        end_frame_field="end_image_url",
        usd_per_second=0.235,
        supports_elements=False,
        extra_arguments={"resolution": "1080p"},
        price_note="fal model page, read 2026-11-11: $0.235/s at 1080p ($0.157/s at 720p)",
        note=(
            "The newest Kling (October 2026). More than three times the price "
            "of 2.5. Its field names here come from fal's how-to guide, not "
            "its API schema, and whether it returns audio is unknown — the "
            "least certain preset on this list. One clip first, and check "
            "both the end frame and the bill."
        ),
    ),
    VideoModel(
        key="kling-3-turbo-pro",
        label="Kling 3.0 Turbo Pro (retired)",
        model_id="fal-ai/kling-video/v3/turbo/pro/image-to-video",
        start_frame_field="image_url",
        end_frame_field="",
        usd_per_second=0.14,
        supports_elements=False,
        retired=True,
        price_note="fal model page, read 2026-11-11: $0.14/s",
        note=(
            "RETIRED: this endpoint has no end-frame input at all, so a clip "
            "rendered with it does not land on the next photo and the movie "
            "stops chaining. It was listed here from a wrong reading of the "
            "docs. Pick another model."
        ),
    ),
)

_BY_KEY = {m.key: m for m in VIDEO_MODELS}
_BY_MODEL_ID = {m.model_id: m for m in VIDEO_MODELS}


def get_video_model(key: str) -> Optional[VideoModel]:
    """A preset by its short key, or None."""
    return _BY_KEY.get((key or "").strip())


def model_for_id(model_id: str) -> Optional[VideoModel]:
    """The preset matching a raw ``fal_model_id``, or None if it is unknown.

    Used to answer capability questions ("can this model take elements?")
    for a config that names a model id directly rather than a preset —
    which is every project written before presets existed.
    """
    return _BY_MODEL_ID.get((model_id or "").strip())


def model_keys() -> list[str]:
    """Every key a config may name — including retired ones, which must
    still load."""
    return [m.key for m in VIDEO_MODELS]


def selectable_models() -> list[VideoModel]:
    """The models a human may pick: everything that is not retired."""
    return [m for m in VIDEO_MODELS if not m.retired]


# Every config key a preset derives. A `video_model` named in a project's
# config must beat these when they were merely INHERITED from the shared
# config, or picking a model changes the label and nothing else.
DERIVED_CONFIG_KEYS: tuple[str, ...] = tuple(
    VIDEO_MODELS[0].as_config_fields().keys()
)


def movie_costs(clip_seconds: float, current_key: str = "") -> list[dict[str, Any]]:
    """What this many seconds of clips costs on each selectable model.

    The comparison the picker needs: not a price per second, which nobody
    can feel, but what THIS movie costs on each model next to what it costs
    today. Clip rendering only — styling, planning and SFX cost the same
    whichever video model is used.
    """
    seconds = max(0.0, float(clip_seconds or 0.0))
    current = get_video_model(current_key)
    base = current.usd_per_second * seconds if current else None
    rows = []
    for model in selectable_models():
        usd = round(model.usd_per_second * seconds, 2)
        rows.append({
            "key": model.key,
            "label": model.label,
            "usd_per_second": model.usd_per_second,
            "movie_usd": usd,
            "delta_usd": None if base is None else round(usd - base, 2),
            "current": bool(current and model.key == current.key),
            "verified": model.verified,
            "price_note": model.price_note,
        })
    return rows
