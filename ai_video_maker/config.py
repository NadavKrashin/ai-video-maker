"""Validated representation of config.json (pydantic)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError

from .costs import Pricing
from .errors import ConfigError
from .video_models import VideoModel, get_video_model, model_for_id, model_keys


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
    # PREFERRED: name a preset here and every fal_* field below is filled in
    # coherently — see video_models.py for the list ("kling-2.5-turbo-pro",
    # "kling-3-pro", "kling-3-turbo-pro"). Models disagree about what the
    # frame fields are CALLED, and setting the model id without the matching
    # field names does not fail loudly: it silently drops the end frame, and
    # the movie stops landing on its photos. A preset makes that impossible.
    #
    # Any fal_* key set explicitly still wins over the preset, so an
    # experiment needs no new preset and every existing config keeps working
    # untouched. Empty = no preset, exactly the old hand-configured behaviour.
    video_model: str = ""
    # Default: Kling v2.5 Turbo Pro (image_url + tail_image_url, start->end
    # interpolation — same request shape as v2.1, better and cheaper).
    fal_model_id: str = "fal-ai/kling-video/v2.5-turbo/pro/image-to-video"
    fal_start_frame_field: str = "image_url"
    # End-frame support is model-dependent. Leave empty to send only the start
    # frame + motion prompt. Kling v2.1/v2.5 use "tail_image_url".
    fal_end_frame_field: str = "tail_image_url"
    fal_duration_as_string: bool = True   # fal Kling uses a string enum ("5"/"10")
    fal_resolution: str = ""              # e.g. "720p", "1080p"; only sent when set
    fal_aspect_ratio: str = ""            # e.g. "16:9"; only sent when set
    # Sent as "negative_prompt" when non-empty: artifacts the video model
    # should avoid (Kling supports it; harmless noise for models that don't).
    # The shared config.json carries a curated preset targeting the failure
    # modes seen on real renders (face distortion, morphing, on-screen text).
    fal_negative_prompt: str = ""
    # Sent as "cfg_scale" when set (Kling: 0..1, server default 0.5). Higher =
    # stricter prompt adherence but more strain on motion coherence. Leave
    # null to use the provider's default.
    fal_cfg_scale: Optional[float] = None
    fal_extra_arguments: dict[str, Any] = Field(default_factory=dict)

    # --- Character elements (face references sent WITH the clip request) --- #
    # Newer Kling versions accept "elements": reference portraits that pin a
    # character's identity through the shot, which is exactly the failure the
    # cast references exist for — a face distorting while people travel
    # between the two frames.
    #
    # OFF BY DEFAULT, and deliberately shaped like fal_start_frame_field /
    # fal_end_frame_field: every part of the request is named here, so
    # turning this on is a config change once the endpoint's schema has been
    # read, not a code change. It is off because the schema has NOT been
    # verified — see README "Character elements" for the three things to
    # check first, the most important being whether the endpoint accepts
    # elements AND an end frame in the same request. This pipeline's whole
    # contract is that a clip lands exactly on the next styled photo; an
    # endpoint that makes you choose is not a trade-off, it is a different
    # product.
    #
    # The request key for the elements array, e.g. "elements". Empty = never
    # send them.
    fal_elements_field: str = ""
    # The key inside each element object that carries the face image URL,
    # e.g. "frontal_image_url" or "image_url".
    fal_element_image_field: str = "frontal_image_url"
    # How many elements the endpoint accepts. fal's docs describe ONE for the
    # v3 image-to-video endpoints; Kling's own docs describe up to three with
    # start/end frames. 0 keeps the feature off even if a field name is set.
    fal_max_elements: int = 0
    # What happens when a clip has more shared people than fal_max_elements.
    # False (default): send NO elements rather than pin an arbitrary subset —
    # one crisp face among three drifting ones can read worse than four that
    # drift together. True: send as many as fit, most-recurring people first.
    # This is the open question a real A/B on a few clips should settle.
    fal_elements_partial: bool = False
    # How the motion prompt refers to an element, e.g. "@Element{index}"
    # (1-based). When set, each element's epithet in the prompt is annotated
    # with its token, because a model that is not told which person an
    # element IS can only guess. Empty sends the elements unreferenced.
    fal_element_prompt_template: str = ""

    # --- Audio (post-generation sound), also fal (auth via FAL_KEY). ---
    # Two layers:
    #   * SFX/ambient: a video->audio model runs on each silent clip and returns
    #     the SAME clip with synced sound muxed in (replaces the file).
    #   * Music bed: a track SUPPLIED BY THE USER (uploaded in the panel or
    #     --music-file), mixed across the whole concatenated final video,
    #     louder than the per-clip SFX (which is ducked under it). Never
    #     generated. See music_volume / sfx_volume.
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
    # The music bed is never generated — it is always a track the user
    # supplies (uploaded in the panel, or --music-file). There is therefore no
    # music model or music prompt to configure; only how it is mixed.
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

    # --- Presentation extras (pure ffmpeg, free, all OFF by default) -------- #
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
    # Emoji in a letter (a heart is the common one) need their own font: the
    # Hebrew text font has no glyph for them and draws an empty box instead.
    # Empty = auto-detect the system's colour-emoji font.
    letter_emoji_font_path: str = ""
    letter_font_size: int = 64
    letter_seconds_per_screen: float = 7.0  # scroll pace per screen height
    # How much to darken the photo montage under a scrolling letter, 0-1.
    # 0 (the default) plays the photos at their true brightness — the user
    # explicitly does not want them greyed out; the letter's own drop shadow
    # is what keeps the text readable. Raise it only to trade the photos'
    # brightness back for contrast.
    letter_overlay_dim: float = 0.0
    # Fade the final video's last moments to black (audio fades with it).
    # Applied after music muxing so the bed fades too. 0 disables.
    end_fade_seconds: float = 1.5

    # --- Cloudinary order intake (the animoments web frontend) ------------- #
    # After payment the web app uploads each order's photos to its own folder:
    # <cloudinary_orders_folder>/<order-id>_<customer>-<stamp>/ in this cloud.
    # `pipeline.py orders` lists them, `pipeline.py ingest` downloads one into
    # a project. The cloud name is public; the Admin-API key/secret are
    # secrets and come from .env (CLOUDINARY_API_KEY / CLOUDINARY_API_SECRET),
    # never from here.
    cloudinary_cloud_name: str = ""
    cloudinary_orders_folder: str = "video-orders"
    # `pipeline.py publish` uploads the finished movie back INTO the order's
    # own folder, named <basename>_v1, <basename>_v2, ... Publishing never
    # replaces anything: each publish takes the next free version number.
    cloudinary_publish_basename: str = "final"
    # Keep a copy of every delivered version in output/published/ as well.
    # output/final_video.mp4 is rebuilt in place by the next combine, so this
    # is the only local record of what a customer was actually sent — at the
    # cost of one movie's disk space (100-300 MB) per published version.
    publish_keep_local_copy: bool = True

    # --- Firebase order ledger (Firestore) ---------------------------------- #
    # The frontend also writes each paid order to Firestore (collection
    # "orders": customer details, package, Cloudinary folder, status "new").
    # When a service-account key is available (FIREBASE_SERVICE_ACCOUNT in
    # .env, or firebase-service-account.json at the repo root) the watcher
    # tracks orders THERE — the authoritative "someone paid" signal — and
    # writes pipeline progress back into each doc's status; Cloudinary is then
    # only the photo store. Without a key everything falls back to pure
    # Cloudinary folder polling. project id defaults to the key file's own
    # project_id, so these usually stay empty.
    firebase_project_id: str = ""
    firebase_credentials_file: str = ""
    firebase_orders_collection: str = "orders"

    # --- Admin server & order watcher (`pipeline.py serve`) ---------------- #
    # Orders are fetched LIVE whenever the panel asks (/api/orders reads
    # Firestore/Cloudinary on demand), so no background process is needed to
    # see or ingest them — that's the default, manual flow. watch_enabled
    # turns on the optional background watcher, which re-checks every
    # watch_poll_seconds and auto-ingests a new order once its upload is
    # complete (quiet for watch_quiet_minutes — the frontend confirms payment
    # BEFORE photos finish uploading, so "folder exists" != "order
    # complete"), acting while nobody is looking at the panel.
    watch_enabled: bool = False
    watch_poll_seconds: int = 300
    watch_quiet_minutes: float = 10.0
    # With the watcher on, also run `storyboard` right after auto-ingest
    # (spends OpenAI credits — styling ~8-30 images per order — so the
    # storyboard is ready for review by the time you open the admin panel).
    watch_auto_storyboard: bool = True
    # Origins allowed to call the admin API from a browser. The panel is
    # served BY the API process (same origin), so the default is NO
    # cross-origin access at all — the production-safe posture. Only add
    # origins here if the panel is ever hosted elsewhere; the token still
    # gates every request either way.
    admin_cors_origins: list[str] = Field(default_factory=list)

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

    # --- Where person-choreography gives way to a camera move -------------- #
    # A pair whose people cannot be staged one-by-one inside the beat budget
    # gets a deterministic camera transition instead (see unstageable_reason
    # in clients/openai_client.py). These two numbers are that threshold, and
    # they are config rather than constants because the right value is a
    # matter of taste that is only settled by watching renders: raise them to
    # let the planner choreograph more pairs, lower them to hand more pairs to
    # the camera. A per-project config.json can pin its own.
    #
    # mover budget: how many people may leave or arrive and still be staged
    # individually. Roughly one mover per beat (5s = one beat, 10s = two),
    # plus a little slack.
    # crowd limit: above this many visible figures, an epithet stops picking
    # anyone out ("the teenage boy with short brown hair" in a 13-person
    # frame points at nobody), so ANY roster change mushes. A group that is
    # the SAME on both sides is never gated by this — holding steady works at
    # any size.
    #
    # These are the long-standing 3/4. They were briefly tightened to 2/3 on
    # the theory that the pairs sitting just under the line were the ones
    # still mushing, and reverted the same day at the user's request: that
    # was a guess, and no render had been watched to back it. Tightening is
    # now a config edit away — try 2/3 on one project and look at the result
    # before making it everyone's default.
    unstageable_mover_budget: int = 3
    unstageable_crowd_limit: int = 4

    # --- Character consistency (see media/faces.py) ------------------------ #
    # A canonical styled portrait per cast member, cut for free out of frames
    # already styled, using the face positions the panel's tagger records.
    # Two consumers: the image model is shown the people it has already drawn
    # when styling a new frame of them, and a video model that accepts
    # character elements can be shown them at render time.
    #
    # Off by default because it depends on TAGS: an untagged project has no
    # references, gets none, and behaves exactly as it did before. Turning it
    # off entirely also restores the old behaviour on tagged projects.
    cast_references_enabled: bool = True
    # Side of the face crop for a frame with one tagged person, as a fraction
    # of the frame's height; a fuller frame divides it down (see
    # reference_crop_box). Raise for more shoulders and background, lower to
    # tighten onto the face.
    face_reference_crop: float = 0.5
    # How many references may ride along with ONE styling call. They compete
    # with the photo being styled for the model's attention, and a crowded
    # frame would otherwise attach a dozen portraits to a single edit.
    max_style_references: int = 4


    # --- Spending ledger (projects/<name>/logs/costs.json) ----------------- #
    # Unit prices used to value every API call the pipeline makes, so each
    # project can report what it cost. ESTIMATES from these numbers, never
    # billed amounts — correct them here when a provider changes its rates.
    pricing: Pricing = Field(default_factory=Pricing)

    # --- Learning from rendered clips (see feedback.py) -------------------- #
    # Feedback on a clip is distilled into a short rule and appended to the
    # planner's system prompt (and the style prompt) on every later run, so
    # the same mistake isn't planned twice. Off = notes are still recorded,
    # but nothing reaches the model.
    learning_enabled: bool = True
    # How many stills are sampled across a rendered clip when the reviewer
    # watches it (feedback --watch). More frames see more (a morph halfway
    # through a 10s clip is easy to miss with four) at a little more cost per
    # review; they are sent at low detail, so this is cheap either way.
    clip_review_frames: int = 8
    # Ceiling on how many lessons ride along with one call. Lessons compete
    # with the instructions they refine for the model's attention, and a
    # planning call already carries ~20 frames; newest lessons win.
    max_lessons_in_prompt: int = 25

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
        data = apply_video_model_preset(data)
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

    # --- the chosen video model, and what it can do ------------------------ #
    def video_model_preset(self) -> Optional[VideoModel]:
        """The preset behind this config, by name or by model id.

        Falls back to matching ``fal_model_id`` so a config written before
        presets existed still gets the capability answers — which is what
        keeps "does this model take elements?" from becoming a second switch
        somebody has to remember to flip in step with the model.
        """
        return get_video_model(self.video_model) or model_for_id(self.fal_model_id)

    def elements_enabled(self) -> bool:
        """Should cast face references ride along with a clip request?

        The MODEL decides. Kling 2.5 cannot take elements, so it never gets
        them however the element fields happen to be set — that is the whole
        point of tying this to the model rather than to a separate flag that
        could drift out of step with it. An unrecognised model has no opinion
        recorded, so it falls back to the fields, which is what lets a new
        endpoint be tried by hand before it earns a preset.
        """
        configured = bool(self.fal_elements_field) and self.fal_max_elements > 0
        preset = self.video_model_preset()
        if preset is not None and not preset.supports_elements:
            return False
        return configured


def apply_video_model_preset(data: dict[str, Any]) -> dict[str, Any]:
    """Fill in the ``fal_*`` fields implied by ``data['video_model']``.

    Keys already present in ``data`` are left ALONE — an explicitly written
    value always beats the preset, so a per-project experiment needs no new
    preset and no existing config changes behaviour. An unknown preset name
    is left for pydantic-free validation below rather than silently ignored:
    a typo that quietly rendered on the wrong model would be expensive.
    """
    name = str(data.get("video_model") or "").strip()
    if not name:
        return data
    preset = get_video_model(name)
    if preset is None:
        raise ConfigError(
            f"Unknown video_model {name!r}. Available: "
            + ", ".join(model_keys())
        )
    merged = dict(data)
    for key, value in preset.as_config_fields().items():
        merged.setdefault(key, value)
    # Pricing follows the model unless the config states its own — otherwise
    # switching to a model that costs 60% more would keep reporting the old
    # estimate, and the spend figures are the only cost signal there is.
    pricing = dict(merged.get("pricing") or {})
    pricing.setdefault("clip_usd_per_second", preset.usd_per_second)
    merged["pricing"] = pricing
    return merged
