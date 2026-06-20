"""OpenAI client (isolated). Image generation/editing + storyboard planning."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from ..config import Config
from ..constants import VALID_DURATIONS
from ..media.images import normalize_image
from ..models import Frame, Storyboard, Transition
from ..retry import with_retries

T = TypeVar("T")

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
    "frame into the next one. Choose 10 when the transition covers more "
    "motion or a larger, slower change that needs room to breathe, and 5 "
    "for quick, subtle changes. Vary it across the video; do not make "
    "them all the same. The last frame's duration_to_next is ignored. "
    "SOUND: for each frame, also write sound_to_next — a short phrase "
    "describing the diegetic ambient sound and sound effects for the clip "
    "that animates this frame into the next one (e.g. 'waves lapping, gulls "
    "calling, soft wind'). Describe real on-screen/world sounds only — no "
    "music, no speech, no narration. Also write one music_prompt for the "
    "whole video: a short description of a single instrumental background "
    "track (mood, genre, instrumentation, no vocals)."
)

_STORYBOARD_NO_TEXT_FRAMES = (
    " Every frame MUST depict a real visual scene with characters "
    "and/or an environment. NEVER create a frame that is only text "
    "on a blank, black, or solid-colour background (title cards, "
    "intro/outro text screens, caption cards, quote cards, credits). "
    "Text is allowed only when it appears naturally on top of a real "
    "scene — it must never be the sole content of a frame."
)

_STORYBOARD_JSON_SHAPE = (
    "Return ONLY valid JSON with this exact shape:\n"
    "{\n"
    '  "project_title": str,\n'
    '  "style": str,                       // overall visual style sentence\n'
    '  "concept": str,                     // overall concept paragraph\n'
    '  "scenes": [str, ...],               // scene list\n'
    '  "music_prompt": str,                // one instrumental bed for the whole video, no vocals\n'
    '  "frames": [\n'
    "    {\n"
    '      "id": "001",\n'
    '      "description": str,             // what happens in this frame\n'
    '      "image_prompt": str,            // full detailed image prompt\n'
    '      "negative_prompt": str,         // things to avoid\n'
    '      "duration_to_next": 5 | 10,     // seconds of the clip into the next frame\n'
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

        def _call() -> bytes:
            with src.open("rb") as fh:
                resp = client.images.edit(
                    model=self.config.openai_image_model,
                    image=fh,
                    prompt=style_prompt,
                    size=self._IMAGE_API_SIZE,
                )
            return base64.b64decode(resp.data[0].b64_json)

        raw = self._retry(_call, f"OpenAI style_image({src.name})")
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

        raw = self._retry(_call, "OpenAI generate_image")
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
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            return resp.choices[0].message.content or "{}"

        raw = self._retry(_call, "OpenAI create_storyboard")
        return self._assemble_storyboard(json.loads(raw), default_duration)

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
            music_prompt=str(data.get("music_prompt", "") or ""),
            frames=frames,
            transitions=transitions,
        )
