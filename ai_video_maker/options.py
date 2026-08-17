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
    music_file: Optional[str] = None
    music_url: Optional[str] = None
    # Analyse the styled frames to plan per-clip motion + duration.
    analyze_frames: bool = True
    # Storyboard-from-idea (instead of from input images).
    idea: Optional[str] = None
    idea_file: Optional[str] = None
    frame_count: Optional[int] = None
    # render: limit to (and force-redo) these clips, e.g. ["003_to_004"].
    clips: Optional[list[str]] = None
    # storyboard: force a fresh vision plan for these transitions even though
    # their frames didn't change (e.g. ["003_to_004"]) — the per-clip "write
    # me a new motion prompt" knob. Hand edits elsewhere stay untouched.
    replan_clips: Optional[list[str]] = None
    # storyboard: re-style these frames from scratch (styled png names, e.g.
    # ["beach.png"]) even when the source is unchanged — the per-frame
    # "regenerate this image" knob. Reconcile then marks the adjacent clips
    # outdated (never deletes them).
    restyle_frames: Optional[list[str]] = None
    # ingest: which Cloudinary order to download (order id / folder name /
    # any unique fragment of it).
    order: Optional[str] = None
    # publish: the exact Cloudinary public_id the caller showed the user for
    # approval (e.g. "video-orders/AM-1_Dana-.../final_v2"). The publish
    # refuses to run under any other name, so an approval can never turn into
    # an upload nobody agreed to. None = take the next free version.
    publish_as: Optional[str] = None
    # tag: redo the identity tags on frames that already have them. Off (the
    # default) only fills untagged frames, so a proposal never overwrites a
    # human's correction.
    retag: bool = False
    # storyboard: propose who is in each untagged frame once planning is done
    # (the cast only exists after the first plan, so this is the earliest it
    # can happen). --no-tag skips that call.
    tag_frames: bool = True
    # storyboard: re-plan EVERY pair, not just the dirty ones — the batch you
    # want after correcting the cast or the photo tags, since both only reach
    # a prompt when its pair is planned again. Hand-written prompts are
    # replaced, so it is never implicit.
    replan_all: bool = False
    # storyboard: turn the pairs named by --replan-clip / --replan-all into
    # deterministic CAMERA transitions instead of planning choreography for
    # them. The human override for a pair the gate judged stageable but that
    # mushed when rendered. Free — the wording is a template, so no vision
    # call — and always a 5s clip.
    replan_as_camera: bool = False
    # faces: also ask the vision model which frames draw a cast member as a
    # different person. Off by default because rebuilding the reference sheet
    # is free and this is the one part of it that costs a call.
    audit_faces: bool = False
    # feedback: one human judgement of a rendered clip (see feedback.py).
    # `feedback_clip` is the transition id it is about ("003_to_004"); empty
    # means feedback about the movie in general. `feedback_learn` off records
    # the note without spending an OpenAI call to distil a rule from it.
    feedback_clip: Optional[str] = None
    feedback_note: Optional[str] = None
    feedback_verdict: str = "bad"
    feedback_learn: bool = True
    # Let a vision call WATCH the rendered clip (stills sampled across it)
    # before the rule is written, and propose a corrected prompt/length for
    # this clip. Needs `feedback_clip`; off records only what the human said.
    feedback_review: bool = True
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

        # --final is CLI shorthand for the full presentation package; an
        # explicit per-flag choice (--no-intro etc.) still wins over it.
        final = bool(get("final"))

        def presentation(name: str):
            explicit = get(name)
            if explicit is None and final:
                return True
            return explicit

        return cls(
            force=bool(get("force")),
            dry_run=bool(get("dry_run")),
            concurrency=get("concurrency"),
            duration=get("duration"),
            motion_prompt=get("motion_prompt"),
            style_prompt=get("style_prompt"),
            music_file=get("music_file"),
            music_url=get("music_url"),
            analyze_frames=not get("no_analyze", False),
            idea=get("idea"),
            idea_file=get("idea_file"),
            frame_count=get("frame_count"),
            clips=get("clip"),
            replan_clips=get("replan_clip"),
            restyle_frames=get("restyle_frame"),
            order=get("order"),
            publish_as=get("publish_as"),
            retag=bool(get("retag")),
            tag_frames=not get("no_tag", False),
            replan_all=bool(get("replan_all")),
            replan_as_camera=bool(get("camera")),
            audit_faces=bool(get("audit")),
            feedback_clip=get("clip_id"),
            feedback_note=get("note"),
            # --good flips the verdict; the default is that feedback is a
            # complaint, because that is what people write notes about.
            feedback_verdict="good" if get("good") else "bad",
            feedback_learn=not get("no_learn", False),
            feedback_review=not get("no_watch", False),
            add_audio=bool(get("add_audio")),
            no_audio=bool(get("no_audio")),
            no_combine=bool(get("no_combine")),
            credits_photos=presentation("credits_photos"),
            closing_letter=get("letter"),
            intro_clip=presentation("intro"),
        )
