"""Storyboard data models (Mode B). Human-editable JSON maps onto these."""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from .errors import StoryboardError


class Frame(BaseModel):
    id: str
    description: str
    image_prompt: str
    negative_prompt: str = ""
    output_path: str
    # Image-based projects: the input image this frame was styled from
    # (workspace-relative). Lets the pipeline detect that a styled file no
    # longer matches its source (inputs swapped/reordered) and re-style it.
    source_path: str = ""
    # SHA-1 of the styled image when the storyboard was saved. Staleness of
    # transitions/clips is decided by comparing content hashes, because file
    # mtimes lie: a cloud-sync client re-materializing untouched files once
    # made every frame look "changed" and wiped a project's rendered clips.
    styled_hash: str = ""


class Transition(BaseModel):
    id: str
    start_frame: str
    end_frame: str
    motion_prompt: str
    duration: int = 5
    # Optional per-clip SFX/ambient guidance for the video->audio step. Empty
    # falls back to config.default_sfx_prompt.
    sound_prompt: str = ""
    output_path: str


class Storyboard(BaseModel):
    project_title: str
    style: str
    duration_per_clip: int = 5
    target_width: int = 1920
    target_height: int = 1080
    concept: str = ""
    scenes: list[str] = Field(default_factory=list)
    # Optional global background-music description for the audio step.
    music_prompt: str = ""
    frames: list[Frame]
    transitions: list[Transition] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "Storyboard":
        if not path.exists():
            raise StoryboardError(f"Storyboard file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(**data)
        except json.JSONDecodeError as exc:
            raise StoryboardError(f"{path} is not valid JSON: {exc}") from exc
        except ValidationError as exc:
            raise StoryboardError(f"Invalid storyboard JSON ({path}):\n{exc}") from exc

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
