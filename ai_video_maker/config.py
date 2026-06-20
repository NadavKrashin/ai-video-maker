"""Validated representation of config.json (pydantic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class Config(BaseModel):
    """Validated representation of config.json."""

    target_width: int = 1920
    target_height: int = 1080
    style_prompt: str
    scratch_style_prompt: str
    motion_prompt: str
    duration: int = 5
    output_format: str = "png"
    default_frame_count: int = 8
    # When True, the storyboard never proposes text-ONLY frames (a title/caption
    # card: text on a blank/black/solid background). Text on top of a real
    # visual scene is allowed; every frame must depict an actual scene.
    avoid_text_only_frames: bool = True

    # Model / endpoint settings (safe to edit).
    openai_image_model: str = "gpt-image-2"
    openai_text_model: str = "gpt-4o"

    # Which image-to-video provider to use: "fal" or "higgsfield".
    video_provider: str = "fal"

    # --- fal.ai (image-to-video) settings. Auth via FAL_KEY. ---
    # Default: Kling v2.1 Pro (image_url + tail_image_url, start->end interpolation).
    # For Kling 3.0 use: model "fal-ai/kling-video/v3/pro/image-to-video",
    # start field "start_image_url", end field "end_image_url".
    fal_model_id: str = "fal-ai/kling-video/v2.1/pro/image-to-video"
    fal_start_frame_field: str = "image_url"
    fal_end_frame_field: str = "tail_image_url"
    fal_duration_as_string: bool = True   # fal Kling uses a string enum ("5"/"10")
    fal_resolution: str = ""
    fal_aspect_ratio: str = ""
    fal_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # --- Higgsfield (image-to-video) settings. Auth via HF_KEY. ---
    # model_id is the path segment of POST https://platform.higgsfield.ai/{model_id}.
    # Browse models at https://cloud.higgsfield.ai/explore.
    higgsfield_model_id: str = "kling-video/v2.1/pro/image-to-video"
    # The request field name for the START frame image URL. Higgsfield/Kling
    # models use "image_url"; some (fal-style) use "start_image_url".
    higgsfield_start_frame_field: str = "image_url"
    # End-frame ("last frame") support is model-dependent and the field name
    # varies per model. Leave empty to send only the start frame + motion prompt.
    # Kling v2.1/v2.5 use "tail_image_url"; some models use "end_image_url".
    higgsfield_end_frame_field: str = "tail_image_url"
    # Higgsfield expects duration as an INTEGER (5/10) for Kling. Set True only
    # if a model you switch to requires a string enum ("5"/"10") instead.
    higgsfield_duration_as_string: bool = False
    # Optional generation args — only sent when non-empty. Some are model-specific.
    higgsfield_resolution: str = ""        # e.g. "720p", "1080p"
    higgsfield_aspect_ratio: str = ""      # e.g. "16:9"
    # Any extra model-specific arguments merged into every request.
    higgsfield_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # --- Audio (post-generation sound). Always uses fal (auth via FAL_KEY), ---
    # regardless of which provider rendered the silent clips. Two layers:
    #   * SFX/ambient: a video->audio model is run on each silent clip; it
    #     returns the SAME clip with synced sound muxed in (replaces the file).
    #   * Music bed: one track generated from `music_prompt`, mixed (ducked) under
    #     the SFX across the whole concatenated final video.
    # Leave audio_mode "none" to keep the old silent behaviour unchanged.
    audio_mode: str = "none"                # "none" | "post"
    # Video->audio model (returns video with synchronized audio).
    sfx_model_id: str = "fal-ai/mmaudio-v2"
    sfx_num_steps: int = 25
    # Used for every Mode A clip, and as the fallback when a Mode B transition
    # has no sound_prompt of its own.
    default_sfx_prompt: str = (
        "Ambient sound and natural sound effects matching the on-screen action. "
        "No music, no speech, no narration."
    )
    sfx_negative_prompt: str = "music, speech, voice, narration, song"
    sfx_extra_arguments: dict[str, Any] = Field(default_factory=dict)
    # Smooth hard cuts between clips: fade each clip's SFX in at the start and
    # out at the end by this many seconds, so the ambience dips gracefully at a
    # cut (the continuous music bed carries the dip) instead of switching
    # abruptly. Sync is preserved (the fade is inside the clip, no overlap).
    # Set 0 to disable.
    sfx_fade_seconds: float = 0.2
    # Music model (text -> a music track). ElevenLabs Music via fal by default;
    # swap for fal-ai/lyria2, cassetteai/music-generator, beatoven/..., etc.
    music_model_id: str = "fal-ai/elevenlabs/music"
    music_prompt: str = (
        "Soft cinematic instrumental underscore, gentle and warm, no vocals."
    )
    music_volume: float = 0.25              # 0..1, ducked under the SFX
    music_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # How many image/clip/SFX API jobs to run at once. These steps are
    # I/O-bound (waiting on the provider), so a small thread pool runs them in
    # parallel. Raise for speed; lower to 1 if you hit provider rate limits
    # (transient 429s are already retried with backoff). CLI: --concurrency.
    max_parallel_requests: int = 4

    # Retry behaviour.
    max_retries: int = 5
    retry_base_delay_seconds: float = 2.0

    @classmethod
    def load(cls, path: Path) -> "Config":
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            return cls(**data)
        except ValidationError as exc:  # pragma: no cover - surfaced to user
            raise SystemExit(f"Invalid config.json:\n{exc}") from exc
