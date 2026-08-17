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
actually rendered with. The Kling 3 entries come from fal's published docs
and have not been exercised from here (fal.ai was unreachable when they were
written), so the first render on one should be a single clip — `render
<project> --clip ID` — before a whole movie is committed to it.
"""
from __future__ import annotations

from dataclasses import dataclass
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
    max_elements: int = 0
    element_prompt_template: str = ""
    # Has this project actually rendered with this request shape?
    verified: bool = False
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
            "fal_max_elements": self.max_elements,
            "fal_element_prompt_template": self.element_prompt_template,
        }


# The order here is the order the panel offers them in: the model in
# production use first, then the newer ones.
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
        note=(
            "The long-standing default. Cheapest, and every movie this studio "
            "has delivered was rendered with it. No character elements: faces "
            "are held only by the two frames and the prompt, which is why "
            "people distort when they travel between them."
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
        max_elements=1,
        element_prompt_template="@Element{index}",
        note=(
            "Newer model, better motion coherence, and it accepts ONE facial "
            "element — the cast face reference — to hold a character's "
            "identity through the shot. About 60% more per second than 2.5. "
            "Request shape from fal's docs, not yet rendered from here: try "
            "one clip first and check the end frame is honoured."
        ),
    ),
    VideoModel(
        key="kling-3-turbo-pro",
        label="Kling 3.0 Turbo Pro",
        model_id="fal-ai/kling-video/v3/turbo/pro/image-to-video",
        start_frame_field="start_image_url",
        end_frame_field="end_image_url",
        usd_per_second=0.14,
        supports_elements=True,
        elements_field="elements",
        element_image_field="frontal_image_url",
        max_elements=1,
        element_prompt_template="@Element{index}",
        note=(
            "The 1080p turbo tier of Kling 3. Priced ABOVE plain v3 Pro on "
            "fal, so pick it for quality rather than to save money. Same "
            "unverified caveat as v3 Pro."
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
    return [m.key for m in VIDEO_MODELS]
