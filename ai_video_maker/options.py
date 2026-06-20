"""Run-level options for a single pipeline invocation.

This is the seam that decouples orchestration from the CLI: the pipeline reads
its per-run choices from a plain ``RunOptions`` object instead of an
``argparse.Namespace``. The CLI builds one with ``RunOptions.from_args(args)``;
a future API endpoint can build the same object straight from a request body.
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
    only_style: bool = False
    only_video: bool = False
    combine: bool = False
    no_combine: bool = False
    add_audio: bool = False
    no_audio: bool = False
    audio_only: bool = False
    music_prompt: Optional[str] = None
    duration: Optional[int] = None
    concurrency: Optional[int] = None
    motion_prompt: Optional[str] = None
    style_prompt: Optional[str] = None
    # Mode B
    idea: Optional[str] = None
    idea_file: Optional[str] = None
    frame_count: Optional[int] = None
    from_scratch: bool = False
    create_storyboard: bool = False
    approve_storyboard: bool = False
    storyboard_file: str = "storyboard/storyboard.json"

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RunOptions":
        return cls(
            force=args.force,
            dry_run=args.dry_run,
            only_style=args.only_style,
            only_video=args.only_video,
            combine=args.combine,
            no_combine=args.no_combine,
            add_audio=args.add_audio,
            no_audio=args.no_audio,
            audio_only=args.audio_only,
            music_prompt=args.music_prompt,
            duration=args.duration,
            concurrency=args.concurrency,
            motion_prompt=args.motion_prompt,
            style_prompt=args.style_prompt,
            idea=args.idea,
            idea_file=args.idea_file,
            frame_count=args.frame_count,
            from_scratch=args.from_scratch,
            create_storyboard=args.create_storyboard,
            approve_storyboard=args.approve_storyboard,
            storyboard_file=args.storyboard_file,
        )
