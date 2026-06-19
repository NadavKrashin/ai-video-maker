#!/usr/bin/env python3
"""
AI Video Maker — local automation pipeline.

Two workflows:

  Mode A (default): Image-to-video from images you provide in input_images/.
      * Style every image to a consistent 1920x1080 look (OpenAI image edit).
      * Send each consecutive styled pair to Higgsfield to produce one short
        clip (start frame, plus an optional model-dependent end frame).
        n images -> n-1 clips.

  Mode B (--from-scratch): Generate a video from a raw idea.
      * Ask OpenAI to write a full storyboard (concept, scenes, frames,
        per-frame image prompts, per-transition motion prompts).
      * Save it to storyboard/storyboard.json + .md and STOP for human review.
      * Only after --approve-storyboard: generate frames, then Higgsfield clips.

Final clips are NOT combined — the user assembles them in Premiere Pro.

Run `python pipeline.py --help` for all flags.
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm

# Third-party SDK / HTTP libs are imported lazily inside the client classes so
# that --dry-run and --help work even if credentials/SDKs are not configured.

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_IMAGES_DIR = PROJECT_ROOT / "input_images"
GENERATED_FRAMES_DIR = PROJECT_ROOT / "generated_frames"
STYLED_IMAGES_DIR = PROJECT_ROOT / "styled_images"
STORYBOARD_DIR = PROJECT_ROOT / "storyboard"
CLIPS_DIR = PROJECT_ROOT / "clips"
LOGS_DIR = PROJECT_ROOT / "logs"
FAILED_JOBS_DIR = PROJECT_ROOT / "failed_jobs"

STATE_FILE = LOGS_DIR / "state.json"
FAILED_JOBS_FILE = FAILED_JOBS_DIR / "failed_jobs.json"
DEFAULT_STORYBOARD_JSON = STORYBOARD_DIR / "storyboard.json"
STORYBOARD_MD = STORYBOARD_DIR / "storyboard.md"

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_DURATIONS = {5, 10}

ALL_DIRS = [
    INPUT_IMAGES_DIR,
    GENERATED_FRAMES_DIR,
    STYLED_IMAGES_DIR,
    STORYBOARD_DIR,
    CLIPS_DIR,
    LOGS_DIR,
    FAILED_JOBS_DIR,
]

T = TypeVar("T")
logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging() -> None:
    """Log to both the console and a timestamped file in logs/."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    logfile = LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.debug("Logging initialised -> %s", logfile)


# --------------------------------------------------------------------------- #
# Config (validated with pydantic)
# --------------------------------------------------------------------------- #
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

    # Model / endpoint settings (safe to edit).
    openai_image_model: str = "gpt-image-1"
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


# --------------------------------------------------------------------------- #
# Storyboard data models (Mode B). Human-editable JSON maps onto these.
# --------------------------------------------------------------------------- #
class Frame(BaseModel):
    id: str
    description: str
    image_prompt: str
    negative_prompt: str = ""
    output_path: str


class Transition(BaseModel):
    id: str
    start_frame: str
    end_frame: str
    motion_prompt: str
    duration: int = 5
    output_path: str


class Storyboard(BaseModel):
    project_title: str
    style: str
    duration_per_clip: int = 5
    target_width: int = 1920
    target_height: int = 1080
    concept: str = ""
    scenes: list[str] = Field(default_factory=list)
    frames: list[Frame]
    transitions: list[Transition] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Storyboard":
        if not path.exists():
            raise FileNotFoundError(f"Storyboard file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            return cls(**data)
        except ValidationError as exc:
            raise SystemExit(f"Invalid storyboard JSON ({path}):\n{exc}") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# --------------------------------------------------------------------------- #
# Job state (resume support) and failed-job tracking
# --------------------------------------------------------------------------- #
class StateStore:
    """Persists per-job status to logs/state.json so runs can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, Any] = {"jobs": {}}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
                self._data.setdefault("jobs", {})
            except json.JSONDecodeError:
                logger.warning("Could not parse %s; starting fresh.", path)

    def status(self, job_id: str) -> Optional[str]:
        entry = self._data["jobs"].get(job_id)
        return entry.get("status") if entry else None

    def is_done(self, job_id: str) -> bool:
        return self.status(job_id) == "done"

    def set(self, job_id: str, status: str, **extra: Any) -> None:
        self._data["jobs"][job_id] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **extra,
        }
        self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class FailedJobStore:
    """Collects failed jobs and writes them to failed_jobs/failed_jobs.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.failures: list[dict[str, Any]] = []

    def record(self, job_id: str, kind: str, error: str, **extra: Any) -> None:
        self.failures.append(
            {
                "job_id": job_id,
                "kind": kind,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **extra,
            }
        )
        logger.error("FAILED [%s] %s: %s", kind, job_id, error)

    def flush(self) -> None:
        if not self.failures:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote %d failed job(s) -> %s", len(self.failures), self.path)


# --------------------------------------------------------------------------- #
# Retry helper (exponential backoff)
# --------------------------------------------------------------------------- #
def with_retries(
    func: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    description: str,
) -> T:
    """Call `func` with exponential backoff. Re-raises the last error."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - we want to retry broadly
            last_exc = exc
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


# --------------------------------------------------------------------------- #
# Image utilities (Pillow) — normalise everything to exactly target size
# --------------------------------------------------------------------------- #
def natural_sort_key(path: Path) -> list[Any]:
    """Sort key that orders e.g. img2 before img10 (natural ordering)."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_input_images(directory: Path) -> list[Path]:
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    return sorted(files, key=natural_sort_key)


def normalize_image(src: Path, dst: Path, width: int, height: int) -> None:
    """
    Force `src` to be exactly width x height and write to `dst`.

    Strategy: convert to RGB, then center-crop to the target aspect ratio
    (cover) and resize. This avoids distortion and preserves the center
    composition. If the source is smaller, it is scaled up.
    """
    with Image.open(src) as im:
        im = im.convert("RGB")
        target_ratio = width / height
        src_ratio = im.width / im.height

        if abs(src_ratio - target_ratio) < 1e-3:
            cropped = im
        elif src_ratio > target_ratio:
            # Too wide -> crop the sides.
            new_w = int(round(im.height * target_ratio))
            left = (im.width - new_w) // 2
            cropped = im.crop((left, 0, left + new_w, im.height))
        else:
            # Too tall -> crop top/bottom.
            new_h = int(round(im.width / target_ratio))
            top = (im.height - new_h) // 2
            cropped = im.crop((0, top, im.width, top + new_h))

        resized = cropped.resize((width, height), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, format="PNG")


def verify_dimensions(path: Path, width: int, height: int) -> bool:
    try:
        with Image.open(path) as im:
            return im.size == (width, height)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not verify %s: %s", path, exc)
        return False


# --------------------------------------------------------------------------- #
# OpenAI client (isolated). Image generation/editing + storyboard text.
# --------------------------------------------------------------------------- #
class OpenAIClient:
    """Thin wrapper around the OpenAI SDK. Easy to swap models/endpoints."""

    # gpt-image-1 supports these sizes; 16:9 1920x1080 is not native, so we
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

    # --- Mode A: edit an existing image into the target style --------------- #
    def style_image(self, src: Path, style_prompt: str, dst: Path) -> None:
        """Edit `src` into the styled look and write a normalised PNG to `dst`."""
        client = self._ensure_client()

        def _call() -> bytes:
            with src.open("rb") as fh:
                # NOTE: images.edit applies the prompt to the provided image.
                resp = client.images.edit(
                    model=self.config.openai_image_model,
                    image=fh,
                    prompt=style_prompt,
                    size=self._IMAGE_API_SIZE,
                )
            return base64.b64decode(resp.data[0].b64_json)

        raw = with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"OpenAI style_image({src.name})",
        )
        self._save_normalized(raw, dst)

    # --- Mode B: generate an image from a text prompt ----------------------- #
    def generate_image(self, prompt: str, dst: Path) -> None:
        """Generate an image from `prompt` and write a normalised PNG to `dst`."""
        client = self._ensure_client()

        def _call() -> bytes:
            resp = client.images.generate(
                model=self.config.openai_image_model,
                prompt=prompt,
                size=self._IMAGE_API_SIZE,
            )
            return base64.b64decode(resp.data[0].b64_json)

        raw = with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description="OpenAI generate_image",
        )
        self._save_normalized(raw, dst)

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
    def create_storyboard(self, idea: str, frame_count: int) -> Storyboard:
        """Ask the text model to produce a structured storyboard for `idea`."""
        client = self._ensure_client()

        system = (
            "You are a film pre-production assistant. Produce a storyboard for a "
            "short cinematic video that will be rendered as a sequence of still "
            "key frames, then animated between consecutive frames. "
            "CRITICAL: keep the same characters, same world, same lighting "
            "language, and same color palette across every frame so the frames "
            "form a continuous, consistent visual flow. Each image_prompt must "
            "be fully self-contained and restate the recurring visual identity "
            "(character looks, wardrobe, environment, palette, lighting) so a "
            "text-to-image model produces consistent results frame to frame."
        )
        user = (
            f"Video idea: {idea}\n\n"
            f"Create exactly {frame_count} key frames. Return ONLY valid JSON "
            "with this exact shape:\n"
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
            '      "negative_prompt": str          // things to avoid\n'
            "    }, ...\n"
            "  ]\n"
            "}\n"
            "Do not include output_path or transitions; those are added later. "
            "Frame ids must be zero-padded 3-digit strings starting at 001."
        )

        def _call() -> str:
            resp = client.chat.completions.create(
                model=self.config.openai_text_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            return resp.choices[0].message.content or "{}"

        raw = with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description="OpenAI create_storyboard",
        )
        return self._assemble_storyboard(json.loads(raw))

    def _assemble_storyboard(self, data: dict[str, Any]) -> Storyboard:
        """Normalise the model JSON and attach output paths + transitions."""
        frames_in = data.get("frames", [])
        frames: list[Frame] = []
        for i, fr in enumerate(frames_in, start=1):
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

        transitions: list[Transition] = []
        for a, b in zip(frames, frames[1:]):
            tid = f"{a.id}_to_{b.id}"
            transitions.append(
                Transition(
                    id=tid,
                    start_frame=a.output_path,
                    end_frame=b.output_path,
                    motion_prompt=self.config.motion_prompt,
                    duration=self.config.duration,
                    output_path=f"clips/{tid}.mp4",
                )
            )

        return Storyboard(
            project_title=data.get("project_title", "Untitled Project"),
            style=data.get("style", self.config.scratch_style_prompt),
            duration_per_clip=self.config.duration,
            target_width=self.config.target_width,
            target_height=self.config.target_height,
            concept=data.get("concept", ""),
            scenes=list(data.get("scenes", [])),
            frames=frames,
            transitions=transitions,
        )


# --------------------------------------------------------------------------- #
# Video clients (isolated). Image-to-video generation.
# --------------------------------------------------------------------------- #
#
# Two providers are supported, selected by config.video_provider:
#   * "fal"        -> fal.ai (recommended for start+end frame interpolation;
#                     Kling on fal supports an end/tail frame). Auth: FAL_KEY.
#   * "higgsfield" -> Higgsfield. Auth: HF_KEY (or HF_API_KEY + HF_API_SECRET).
#
# Both fal-client and higgsfield-client expose the SAME interface:
#   upload_file(path) -> hosted URL, and subscribe(model_id, args) -> result
#   dict with result["video"]["url"]. So the shared logic lives in the base
#   class below; each subclass only supplies its SDK + credential check.
# --------------------------------------------------------------------------- #
@dataclass
class VideoBackend:
    """Provider-specific settings, read from config."""
    model_id: str
    start_frame_field: str
    end_frame_field: str
    duration_as_string: bool
    resolution: str
    aspect_ratio: str
    extra_arguments: dict[str, Any]


class SubscribeVideoClient:
    """
    Base image-to-video client for fal-style SDKs (fal-client, higgsfield-client).

    Subclasses implement `_import_sdk()` and `_check_credentials()`.
    """

    provider: str = "video"

    def __init__(self, config: Config, backend: VideoBackend) -> None:
        self.config = config
        self.backend = backend
        self._sdk = None  # lazily imported
        # Cache uploaded-image URLs so a frame shared by two consecutive clips
        # is only uploaded once per run.
        self._upload_cache: dict[Path, str] = {}

    # --- provider hooks (override in subclasses) --------------------------- #
    def _import_sdk(self):  # pragma: no cover - trivial
        raise NotImplementedError

    def _check_credentials(self) -> None:  # pragma: no cover - trivial
        raise NotImplementedError

    def _ensure_sdk(self):
        if self._sdk is None:
            self._check_credentials()
            self._sdk = self._import_sdk()
        return self._sdk

    # --- UPLOAD ------------------------------------------------------------ #
    def _upload(self, path: Path) -> str:
        """Upload a local image and return its hosted URL (cached per run)."""
        if path in self._upload_cache:
            return self._upload_cache[path]
        sdk = self._ensure_sdk()

        def _call() -> str:
            return sdk.upload_file(path)

        url = with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"{self.provider} upload ({path.name})",
        )
        self._upload_cache[path] = url
        return url

    # --- BUILD REQUEST ----------------------------------------------------- #
    def _build_arguments(
        self,
        start_url: str,
        end_url: Optional[str],
        motion_prompt: str,
        duration: int,
    ) -> dict[str, Any]:
        b = self.backend
        duration_value: Any = str(duration) if b.duration_as_string else duration
        args: dict[str, Any] = {
            b.start_frame_field: start_url,  # start frame
            "prompt": motion_prompt,         # motion prompt
            "duration": duration_value,
        }
        # End frame is only sent when the model documents a field for it.
        if end_url and b.end_frame_field:
            args[b.end_frame_field] = end_url
        if b.resolution:
            args["resolution"] = b.resolution
        if b.aspect_ratio:
            args["aspect_ratio"] = b.aspect_ratio
        args.update(b.extra_arguments)
        return args

    # --- SUBMIT + WAIT ----------------------------------------------------- #
    def _generate_video_url(self, arguments: dict[str, Any]) -> str:
        """Submit the job, wait for completion, and return the video URL."""
        sdk = self._ensure_sdk()

        def _call() -> str:
            # subscribe() submits and blocks until the job reaches a terminal
            # state, returning the result dict (or raising on failure).
            result = sdk.subscribe(self.backend.model_id, arguments)
            video = (result or {}).get("video") or {}
            url = video.get("url")
            if not url:
                raise RuntimeError(
                    f"{self.provider} finished without a video URL: {result}"
                )
            return url

        return with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"{self.provider} generate",
        )

    # --- DOWNLOAD ---------------------------------------------------------- #
    def download(self, video_url: str, dst: Path) -> None:
        """Stream the result video to `dst` (atomic via a .part temp file)."""
        import requests

        dst.parent.mkdir(parents=True, exist_ok=True)

        def _call() -> None:
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(video_url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
            tmp.replace(dst)

        with_retries(
            _call,
            max_retries=self.config.max_retries,
            base_delay=self.config.retry_base_delay_seconds,
            description=f"{self.provider} download -> {dst.name}",
        )

    # --- High-level convenience -------------------------------------------- #
    def generate_clip(
        self,
        start_frame: Path,
        end_frame: Path,
        motion_prompt: str,
        duration: int,
        dst: Path,
    ) -> None:
        start_url = self._upload(start_frame)
        end_url = self._upload(end_frame) if self.backend.end_frame_field else None
        arguments = self._build_arguments(start_url, end_url, motion_prompt, duration)
        logger.info(
            "%s job: %s (model=%s, start=%s%s)",
            self.provider,
            dst.name,
            self.backend.model_id,
            start_frame.name,
            f", end={end_frame.name}" if end_url else "",
        )
        video_url = self._generate_video_url(arguments)
        self.download(video_url, dst)


class FalClient(SubscribeVideoClient):
    """fal.ai backend. Docs: https://docs.fal.ai — auth via FAL_KEY."""

    provider = "fal"

    def __init__(self, config: Config) -> None:
        super().__init__(
            config,
            VideoBackend(
                model_id=config.fal_model_id,
                start_frame_field=config.fal_start_frame_field,
                end_frame_field=config.fal_end_frame_field,
                duration_as_string=config.fal_duration_as_string,
                resolution=config.fal_resolution,
                aspect_ratio=config.fal_aspect_ratio,
                extra_arguments=config.fal_extra_arguments,
            ),
        )

    def _check_credentials(self) -> None:
        if not os.environ.get("FAL_KEY"):
            raise RuntimeError(
                "fal credentials missing. Set FAL_KEY in your .env file "
                "(get one at https://fal.ai/dashboard/keys)."
            )

    def _import_sdk(self):
        import fal_client  # imported lazily

        return fal_client


class HiggsfieldClient(SubscribeVideoClient):
    """Higgsfield backend. Docs: https://docs.higgsfield.ai — auth via HF_KEY."""

    provider = "higgsfield"

    def __init__(self, config: Config) -> None:
        super().__init__(
            config,
            VideoBackend(
                model_id=config.higgsfield_model_id,
                start_frame_field=config.higgsfield_start_frame_field,
                end_frame_field=config.higgsfield_end_frame_field,
                duration_as_string=config.higgsfield_duration_as_string,
                resolution=config.higgsfield_resolution,
                aspect_ratio=config.higgsfield_aspect_ratio,
                extra_arguments=config.higgsfield_extra_arguments,
            ),
        )

    def _check_credentials(self) -> None:
        if not os.environ.get("HF_KEY") and not (
            os.environ.get("HF_API_KEY") and os.environ.get("HF_API_SECRET")
        ):
            raise RuntimeError(
                "Higgsfield credentials missing. Set HF_KEY "
                '("<api_key>:<api_secret>") or HF_API_KEY + HF_API_SECRET '
                "in your .env file."
            )

    def _import_sdk(self):
        import higgsfield_client  # imported lazily

        return higgsfield_client


def make_video_client(config: Config) -> SubscribeVideoClient:
    """Instantiate the video client selected by config.video_provider."""
    provider = (config.video_provider or "fal").lower()
    if provider == "fal":
        return FalClient(config)
    if provider == "higgsfield":
        return HiggsfieldClient(config)
    raise SystemExit(
        f"Unknown video_provider {provider!r}. Use 'fal' or 'higgsfield'."
    )


# --------------------------------------------------------------------------- #
# Pipeline orchestration
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    input_count: int = 0
    styled_created: int = 0
    styled_skipped: int = 0
    styled_failed: int = 0
    videos_created: int = 0
    videos_skipped: int = 0
    videos_failed: int = 0

    def print(self) -> None:
        line = "=" * 60
        print(f"\n{line}\nRUN SUMMARY\n{line}")
        print(f"  Input/generated images : {self.input_count}")
        print(f"  Styled/frames created  : {self.styled_created}")
        print(f"  Styled/frames skipped  : {self.styled_skipped}")
        print(f"  Styled/frames failed   : {self.styled_failed}")
        print(f"  Videos created         : {self.videos_created}")
        print(f"  Videos skipped         : {self.videos_skipped}")
        print(f"  Videos failed          : {self.videos_failed}")
        print(f"\n  Output folders:")
        print(f"    Styled images   : {STYLED_IMAGES_DIR}")
        print(f"    Generated frames: {GENERATED_FRAMES_DIR}")
        print(f"    Clips           : {CLIPS_DIR}")
        print(f"    Logs            : {LOGS_DIR}")
        print(f"    Failed jobs     : {FAILED_JOBS_FILE}")
        print(line)


class Pipeline:
    def __init__(self, config: Config, args: argparse.Namespace) -> None:
        self.config = config
        self.args = args
        self.dry_run: bool = args.dry_run
        self.force: bool = args.force
        self.duration: int = args.duration or config.duration
        self.state = StateStore(STATE_FILE)
        self.failed = FailedJobStore(FAILED_JOBS_FILE)
        self.summary = RunSummary()
        self.openai = OpenAIClient(config)
        self.video_client = make_video_client(config)

    # ------------------------------ Mode A ------------------------------- #
    def run_mode_a(self) -> None:
        logger.info("=== Mode A: image-to-video from input_images/ ===")
        images = list_input_images(INPUT_IMAGES_DIR)
        self.summary.input_count = len(images)

        if not images:
            raise SystemExit(
                f"No supported images found in {INPUT_IMAGES_DIR}. "
                f"Supported: {sorted(SUPPORTED_IMAGE_EXTS)}"
            )
        logger.info("Found %d input image(s).", len(images))

        styled: list[Path] = []
        if not self.args.only_video:
            styled = self._style_images(images)
        else:
            # --only-video: use existing styled images.
            styled = sorted(
                (p for p in STYLED_IMAGES_DIR.iterdir()
                 if p.is_file() and p.suffix.lower() == ".png"),
                key=natural_sort_key,
            )
            logger.info("Using %d existing styled image(s).", len(styled))

        if self.args.only_style:
            logger.info("--only-style set: skipping video generation.")
            return

        if len(styled) < 2:
            logger.warning(
                "Need at least 2 styled images to make a clip; have %d.", len(styled)
            )
            return

        self._generate_clips(
            [(styled[i], styled[i + 1]) for i in range(len(styled) - 1)],
            motion_prompt=self._motion_prompt(),
        )

    def _style_images(self, images: list[Path]) -> list[Path]:
        style_prompt = self.args.style_prompt or self.config.style_prompt
        styled: list[Path] = []

        for idx, src in enumerate(
            tqdm(images, desc="Styling images", unit="img"), start=1
        ):
            dst = STYLED_IMAGES_DIR / f"{idx:03d}_styled.png"
            styled.append(dst)
            job_id = f"style:{dst.name}"

            if dst.exists() and not self.force:
                self.summary.styled_skipped += 1
                logger.info("Skip styled (done): %s", dst.name)
                continue

            if self.dry_run:
                logger.info("[dry-run] would style %s -> %s", src.name, dst.name)
                self.summary.styled_created += 1
                continue

            try:
                self.openai.style_image(src, style_prompt, dst)
                if not verify_dimensions(dst, self.config.target_width, self.config.target_height):
                    raise RuntimeError(f"{dst.name} is not {self.config.target_width}x{self.config.target_height}")
                self.state.set(job_id, "done", output=str(dst))
                self.summary.styled_created += 1
                logger.info("Styled: %s", dst.name)
            except Exception as exc:  # noqa: BLE001
                self.summary.styled_failed += 1
                self.state.set(job_id, "failed")
                self.failed.record(job_id, "style", str(exc), source=str(src))

        return styled

    # ------------------------------ Mode B ------------------------------- #
    def run_mode_b(self) -> None:
        logger.info("=== Mode B: generate from scratch ===")

        if self.args.create_storyboard:
            self._create_storyboard()
            return

        if self.args.approve_storyboard:
            self._run_approved_storyboard()
            return

        raise SystemExit(
            "Mode B requires either --create-storyboard (with --idea) or "
            "--approve-storyboard. See README.md."
        )

    def _create_storyboard(self) -> None:
        if not self.args.idea:
            raise SystemExit("--create-storyboard requires --idea \"...\"")

        frame_count = self.config.default_frame_count
        logger.info(
            "Creating storyboard for idea: %r (%d frames)", self.args.idea, frame_count
        )

        if self.dry_run:
            logger.info("[dry-run] would call OpenAI to build a storyboard and "
                        "write %s + %s", DEFAULT_STORYBOARD_JSON, STORYBOARD_MD)
            return

        storyboard = self.openai.create_storyboard(self.args.idea, frame_count)
        storyboard.save(DEFAULT_STORYBOARD_JSON)
        write_storyboard_markdown(storyboard, STORYBOARD_MD)

        print("\n" + "=" * 70)
        print("Storyboard created. Review storyboard/storyboard.md or "
              "storyboard/storyboard.json,")
        print("edit if needed, then run with --approve-storyboard.")
        print("=" * 70 + "\n")

    def _run_approved_storyboard(self) -> None:
        sb_path = Path(self.args.storyboard_file)
        if not sb_path.is_absolute():
            sb_path = PROJECT_ROOT / sb_path
        storyboard = Storyboard.load(sb_path)
        logger.info(
            "Approved storyboard %r with %d frame(s).",
            storyboard.project_title,
            len(storyboard.frames),
        )
        self.summary.input_count = len(storyboard.frames)

        if not self.args.only_video:
            self._generate_frames(storyboard)

        if self.args.only_style:
            logger.info("--only-style set: skipping video generation.")
            return

        # Build transition pairs from the storyboard (per-transition motion).
        pairs: list[tuple[Path, Path, str, int]] = []
        transitions = storyboard.transitions or self._derive_transitions(storyboard)
        for tr in transitions:
            pairs.append(
                (
                    PROJECT_ROOT / tr.start_frame,
                    PROJECT_ROOT / tr.end_frame,
                    self.args.motion_prompt or tr.motion_prompt,
                    self.args.duration or tr.duration,
                )
            )
        self._generate_clips_with_prompts(pairs)

    @staticmethod
    def _derive_transitions(storyboard: Storyboard) -> list[Transition]:
        derived: list[Transition] = []
        frames = storyboard.frames
        for a, b in zip(frames, frames[1:]):
            tid = f"{a.id}_to_{b.id}"
            derived.append(
                Transition(
                    id=tid,
                    start_frame=a.output_path,
                    end_frame=b.output_path,
                    motion_prompt=storyboard.style,
                    duration=storyboard.duration_per_clip,
                    output_path=f"clips/{tid}.mp4",
                )
            )
        return derived

    def _generate_frames(self, storyboard: Storyboard) -> None:
        for frame in tqdm(
            storyboard.frames, desc="Generating frames", unit="frame"
        ):
            dst = PROJECT_ROOT / frame.output_path
            job_id = f"frame:{frame.id}"

            if dst.exists() and not self.force:
                self.summary.styled_skipped += 1
                logger.info("Skip frame (done): %s", dst.name)
                continue

            # Reinforce style consistency in every prompt.
            full_prompt = (
                f"{frame.image_prompt}\n\nStyle: {storyboard.style}"
            )
            if frame.negative_prompt:
                full_prompt += f"\n\nAvoid: {frame.negative_prompt}"

            if self.dry_run:
                logger.info("[dry-run] would generate frame %s -> %s", frame.id, dst.name)
                self.summary.styled_created += 1
                continue

            try:
                self.openai.generate_image(full_prompt, dst)
                if not verify_dimensions(dst, self.config.target_width, self.config.target_height):
                    raise RuntimeError(f"{dst.name} is not {self.config.target_width}x{self.config.target_height}")
                self.state.set(job_id, "done", output=str(dst))
                self.summary.styled_created += 1
                logger.info("Generated frame: %s", dst.name)
            except Exception as exc:  # noqa: BLE001
                self.summary.styled_failed += 1
                self.state.set(job_id, "failed")
                self.failed.record(job_id, "frame", str(exc), frame_id=frame.id)

    # ------------------------- shared video step ------------------------- #
    def _motion_prompt(self) -> str:
        return self.args.motion_prompt or self.config.motion_prompt

    def _generate_clips(
        self, pairs: list[tuple[Path, Path]], motion_prompt: str
    ) -> None:
        enriched = [
            (a, b, motion_prompt, self.duration) for a, b in pairs
        ]
        self._generate_clips_with_prompts(enriched)

    def _generate_clips_with_prompts(
        self, pairs: list[tuple[Path, Path, str, int]]
    ) -> None:
        if not pairs:
            logger.warning("No transition pairs to render.")
            return

        for start, end, motion, duration in tqdm(
            pairs, desc="Generating clips", unit="clip"
        ):
            dst = self._clip_name(start, end)
            job_id = f"clip:{dst.name}"

            if dst.exists() and not self.force:
                self.summary.videos_skipped += 1
                logger.info("Skip clip (done): %s", dst.name)
                continue

            if self.dry_run:
                # Frames may not exist yet during a dry-run (styling was also
                # dry-run), so report the plan without checking for them.
                logger.info(
                    "[dry-run] would render %s (%ss): %s -> %s | motion=%r",
                    dst.name, duration, start.name, end.name, motion,
                )
                self.summary.videos_created += 1
                continue

            if not start.exists() or not end.exists():
                self.summary.videos_failed += 1
                self.failed.record(
                    job_id, "clip",
                    f"Missing frame(s): {start.name} / {end.name}",
                )
                continue

            try:
                self.video_client.generate_clip(start, end, motion, duration, dst)
                self.state.set(job_id, "done", output=str(dst))
                self.summary.videos_created += 1
                logger.info("Clip ready: %s", dst.name)
            except Exception as exc:  # noqa: BLE001
                self.summary.videos_failed += 1
                self.state.set(job_id, "failed")
                self.failed.record(
                    job_id, "clip", str(exc),
                    start=str(start), end=str(end),
                )

    @staticmethod
    def _clip_name(start: Path, end: Path) -> Path:
        """Map a frame pair to clips/<start>_to_<end>.mp4 using leading ids."""
        def stem_id(p: Path) -> str:
            m = re.match(r"(\d+)", p.stem)
            return m.group(1) if m else p.stem
        return CLIPS_DIR / f"{stem_id(start)}_to_{stem_id(end)}.mp4"

    # ------------------------------- run --------------------------------- #
    def run(self) -> None:
        try:
            if self.args.from_scratch:
                self.run_mode_b()
            else:
                self.run_mode_a()
        finally:
            self.failed.flush()
            self.summary.print()


# --------------------------------------------------------------------------- #
# Storyboard markdown rendering
# --------------------------------------------------------------------------- #
def write_storyboard_markdown(storyboard: Storyboard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# {storyboard.project_title}\n")
    lines.append(f"**Style:** {storyboard.style}\n")
    if storyboard.concept:
        lines.append(f"**Concept:** {storyboard.concept}\n")
    lines.append(
        f"**Output:** {storyboard.target_width}x{storyboard.target_height}, "
        f"{storyboard.duration_per_clip}s per clip\n"
    )

    if storyboard.scenes:
        lines.append("## Scenes\n")
        for i, scene in enumerate(storyboard.scenes, start=1):
            lines.append(f"{i}. {scene}")
        lines.append("")

    lines.append("## Frames\n")
    for fr in storyboard.frames:
        lines.append(f"### Frame {fr.id}")
        lines.append(f"- **Description:** {fr.description}")
        lines.append(f"- **Image prompt:** {fr.image_prompt}")
        if fr.negative_prompt:
            lines.append(f"- **Negative prompt:** {fr.negative_prompt}")
        lines.append(f"- **Output:** `{fr.output_path}`")
        lines.append("")

    if storyboard.transitions:
        lines.append("## Transitions (clips)\n")
        for tr in storyboard.transitions:
            lines.append(f"### {tr.id}  ({tr.duration}s)")
            lines.append(f"- **Start:** `{tr.start_frame}`")
            lines.append(f"- **End:** `{tr.end_frame}`")
            lines.append(f"- **Motion prompt:** {tr.motion_prompt}")
            lines.append(f"- **Output:** `{tr.output_path}`")
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Local AI video maker (image-to-video via OpenAI + Higgsfield).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.json", help="Path to config JSON.")
    p.add_argument("--force", action="store_true", help="Redo completed outputs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned work without spending API credits.")
    p.add_argument("--only-style", action="store_true",
                   help="Only style/generate images; skip video generation.")
    p.add_argument("--only-video", action="store_true",
                   help="Only generate videos from existing styled/generated images.")
    p.add_argument("--duration", type=int, choices=sorted(VALID_DURATIONS),
                   help="Clip duration in seconds (5 or 10).")
    p.add_argument("--motion-prompt", default=None,
                   help="Override the global motion prompt.")
    p.add_argument("--style-prompt", default=None,
                   help="Override the global style prompt (Mode A).")
    # Mode B
    p.add_argument("--idea", default=None, help="Video idea/prompt (Mode B).")
    p.add_argument("--from-scratch", action="store_true",
                   help="Use Mode B (generate from an idea).")
    p.add_argument("--create-storyboard", action="store_true",
                   help="Mode B: create the storyboard and stop.")
    p.add_argument("--approve-storyboard", action="store_true",
                   help="Mode B: generate frames/clips from an approved storyboard.")
    p.add_argument("--storyboard-file", default="storyboard/storyboard.json",
                   help="Path to the storyboard JSON (Mode B approval).")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)

    setup_logging()
    load_dotenv(PROJECT_ROOT / ".env")

    if args.only_style and args.only_video:
        parser.error("--only-style and --only-video are mutually exclusive.")

    config = Config.load((PROJECT_ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config))

    if args.dry_run:
        logger.info("DRY-RUN: no API credits will be spent.")

    pipeline = Pipeline(config, args)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
