"""Run-level options for a single pipeline invocation.

This is the seam that decouples orchestration from the CLI: the pipeline reads
its per-run choices from a plain ``RunOptions`` object instead of an
``argparse.Namespace``. The CLI builds one with ``RunOptions.from_args(args)``;
a future API endpoint can build the same object straight from a request body.

Which lifecycle step runs is NOT an option — it's the subcommand, passed to
``Pipeline.execute(command)`` — so these fields are only the knobs a step can
be turned with.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class RunOptions:
    """Everything that varies per run (CLI flags / API request fields)."""

    force: bool = False
    dry_run: bool = False
    concurrency: Optional[int] = None
    duration: Optional[int] = None
    motion_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    music_prompt: Optional[str] = None
    music_file: Optional[str] = None
    # Analyse the styled frames to plan per-clip motion + duration.
    analyze_frames: bool = True
    # Storyboard-from-idea (instead of from input images).
    idea: Optional[str] = None
    idea_file: Optional[str] = None
    frame_count: Optional[int] = None
    # render: limit to (and force-redo) these clips, e.g. ["003_to_004"].
    clips: Optional[list[str]] = None
    # Per-run audio override; neither set -> config.audio_mode decides.
    add_audio: bool = False
    no_audio: bool = False
    # run: stop after the clips, don't build the final video.
    no_combine: bool = False
    # combine: presentation extras. None -> use the config value; True/False
    # -> per-run override (--credits-photos / --no-credits-photos).
    credits_photos: Optional[bool] = None
    closing_letter: Optional[bool] = None
    intro_clip: Optional[bool] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunOptions":
        # Subcommands only define the flags they use, so read defensively.
        def get(name: str, default=None):
            return getattr(args, name, default)

        return cls(
            force=bool(get("force")),
            dry_run=bool(get("dry_run")),
            concurrency=get("concurrency"),
            duration=get("duration"),
            motion_prompt=get("motion_prompt"),
            style_prompt=get("style_prompt"),
            music_prompt=get("music_prompt"),
            music_file=get("music_file"),
            analyze_frames=not get("no_analyze", False),
            idea=get("idea"),
            idea_file=get("idea_file"),
            frame_count=get("frame_count"),
            clips=get("clip"),
            add_audio=bool(get("add_audio")),
            no_audio=bool(get("no_audio")),
            no_combine=bool(get("no_combine")),
            credits_photos=get("credits_photos"),
            closing_letter=get("letter"),
            intro_clip=get("intro"),
        )
