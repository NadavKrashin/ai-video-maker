"""Validated representation of config.json (pydantic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from .errors import ConfigError


class Config(BaseModel):
    """Validated representation of config.json.

    Image generation runs on OpenAI; image-to-video and all audio run on fal.ai.
    The ``fal_*`` fields make the video model swappable without code changes
    (different fal models name their frame fields differently and disagree on
    whether ``duration`` is an int or a string enum).
    """

    target_width: int = 1920
    target_height: int = 1080
    style_prompt: str
    scratch_style_prompt: str
    motion_prompt: str
    duration: int = 5
    default_frame_count: int = 8
    # When True, the storyboard never proposes text-ONLY frames (a title/caption
    # card: text on a blank/black/solid background). Text on top of a real
    # visual scene is allowed; every frame must depict an actual scene.
    avoid_text_only_frames: bool = True

    # OpenAI models (safe to edit).
    openai_image_model: str = "gpt-image-2"
    openai_text_model: str = "gpt-5.1"

    # --- fal.ai image-to-video. Auth via FAL_KEY. ---
    # Default: Kling v2.5 Turbo Pro (image_url + tail_image_url, start->end
    # interpolation — same request shape as v2.1, better and cheaper).
    # For Kling 3.0 use model "fal-ai/kling-video/v3/pro/image-to-video" with
    # start field "start_image_url" and end field "end_image_url".
    fal_model_id: str = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"
    fal_start_frame_field: str = "image_url"
    # End-frame support is model-dependent. Leave empty to send only the start
    # frame + motion prompt. Kling v2.1/v2.5 use "tail_image_url".
    fal_end_frame_field: str = "tail_image_url"
    fal_duration_as_string: bool = True   # fal Kling uses a string enum ("5"/"10")
    fal_resolution: str = ""              # e.g. "720p", "1080p"; only sent when set
    fal_aspect_ratio: str = ""            # e.g. "16:9"; only sent when set
    fal_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # --- Audio (post-generation sound), also fal (auth via FAL_KEY). ---
    # Two layers:
    #   * SFX/ambient: a video->audio model runs on each silent clip and returns
    #     the SAME clip with synced sound muxed in (replaces the file).
    #   * Music bed: one track from `music_prompt`, mixed across the whole
    #     concatenated final video, louder than the per-clip SFX (which is
    #     ducked under it). See music_volume / sfx_volume.
    # Leave audio_mode "none" to keep clips silent.
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
    # Smooth hard cuts: fade each clip's SFX in at the start and out at the end
    # by this many seconds, so the ambience dips gracefully at a cut (the
    # continuous music bed carries the dip) instead of switching abruptly. Sync
    # is preserved (the fade is inside the clip, no overlap). Set 0 to disable.
    sfx_fade_seconds: float = 0.2
    # Music model (text -> a music track). ElevenLabs Music via fal by default;
    # swap for fal-ai/lyria2, cassetteai/music-generator, beatoven/..., etc.
    music_model_id: str = "fal-ai/elevenlabs/music"
    music_prompt: str = (
        "Soft cinematic instrumental underscore, gentle and warm, no vocals."
    )
    # Relative levels when the music bed is mixed with the per-clip SFX. The
    # background music is meant to sit ON TOP of (louder than) the clip SFX, so
    # the SFX is ducked under it. Both are 0..1; what matters is the ratio
    # (music_volume > sfx_volume => music dominates). sfx_volume only applies
    # when a clip actually has SFX; with no SFX the music is the only audio.
    music_volume: float = 0.85              # 0..1, the dominant background bed
    sfx_volume: float = 0.35                # 0..1, clip SFX ducked under music
    # False (default): the track plays once and the rest of the video continues
    # with SFX only. True: the track loops for the whole video length.
    music_loop: bool = False
    music_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # --- Presentation extras (pure ffmpeg, free, both OFF by default) ------- #
    # opening_reveal: the movie opens on the real (unstyled) first photo, holds
    # a beat, then crossfades into the first clip — "the photo comes alive".
    opening_reveal: bool = False
    opening_reveal_hold_seconds: float = 1.6
    # credits_photos: after the last clip, the original photos play as a short
    # end-credits montage (each fitted on a blurred background, fading in/out),
    # so viewers see the real moments behind the animation. Uses the photos
    # recorded in the storyboard (source_path), in movie order.
    credits_photos: bool = False
    credits_seconds_per_photo: float = 1.5
    # intro_clip: prepend the user's own intro video before everything else,
    # normalized to the movie's frame size. The intro is SHARED by all
    # projects: intro_file resolves against the repo root (absolute paths
    # work too), and a per-project config.json can still override it.
    intro_clip: bool = False
    intro_file: str = "intro.mp4"
    # closing_letter: scroll the text of projects/<name>/letter.txt over a
    # dark background at the very end, credits-style. Hebrew/RTL-safe (bidi
    # reordering happens in media/letter.py, not ffmpeg).
    closing_letter: bool = False
    letter_font_path: str = ""       # empty = auto-detect a Hebrew-capable font
    letter_font_size: int = 64
    letter_seconds_per_screen: float = 7.0  # scroll pace per screen height
    # Fade the final video's last moments to black (audio fades with it).
    # Applied after music muxing so the bed fades too. 0 disables.
    end_fade_seconds: float = 1.5

    # How many image/clip/SFX API jobs to run at once. These steps are I/O-bound
    # (waiting on the provider), so a small thread pool runs them in parallel.
    # Raise for speed; lower to 1 if you hit rate limits (transient 429s are
    # already retried with backoff). CLI: --concurrency.
    max_parallel_requests: int = 4

    # Retry behaviour (exponential backoff).
    max_retries: int = 5
    retry_base_delay_seconds: float = 2.0

    # When OpenAI's safety filter wrongly flags an image prompt
    # (`moderation_blocked`), reword the prompt to be unambiguously safe-for-work
    # and try again, up to this many times. Set 0 to disable and fail fast.
    moderation_reword_attempts: int = 3

    @classmethod
    def load(cls, path: Path, override_path: Path | None = None) -> "Config":
        """Load and validate the config, optionally layering project overrides.

        `override_path` (projects/<name>/config.json) is merged key-over-key on
        top of the shared config when it exists, so one movie can pin its own
        style prompt, video model, audio settings, etc. without forking the
        global file.
        """
        data = cls._read_json(path, required=True)
        if override_path is not None:
            overrides = cls._read_json(override_path, required=False)
            data.update(overrides)
        try:
            return cls(**data)
        except ValidationError as exc:
            source = f"{path}" + (
                f" (+ overrides from {override_path})"
                if override_path is not None and override_path.exists() else ""
            )
            raise ConfigError(f"Invalid config ({source}):\n{exc}") from exc

    @staticmethod
    def _read_json(path: Path, *, required: bool) -> dict[str, Any]:
        if not path.exists():
            if required:
                raise ConfigError(f"Config file not found: {path}")
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a JSON object.")
        return data
