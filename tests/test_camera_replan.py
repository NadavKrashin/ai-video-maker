"""Forcing a pair to become a camera transition.

The gate already converts pairs it judges unstageable, but it judges from
tags and headcounts and it has never seen a render. This is the human
override for the pair that mushed anyway — and because the camera wording is
a deterministic template, exercising it costs nothing and calls nothing.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ai_video_maker.clients.openai_client import is_camera_transition
from ai_video_maker.models import Character, Frame, FramePerson, Storyboard, Transition
from ai_video_maker.errors import PipelineError


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _frame_pairs(workspace, names):
    now = time.time()
    pairs = []
    for n in names:
        src = _touch(workspace.input_images_dir / f"{n}.jpg")
        dst = _touch(workspace.styled_images_dir / f"{n}.png")
        os.utime(src, (now - 100, now - 100))
        os.utime(dst, (now - 100, now - 100))
        pairs.append((src, dst))
    return pairs


def _saved(workspace, names, people=None, cast=None) -> Storyboard:
    frames = [
        Frame(
            id=n, description="", image_prompt="",
            output_path=f"styled_images/{n}.png",
            source_path=f"input_images/{n}.jpg",
            people=list((people or {}).get(n, [])),
        )
        for n in names
    ]
    transitions = [
        Transition(
            id=f"{a}_to_{b}", start_frame=f"styled_images/{a}.png",
            end_frame=f"styled_images/{b}.png",
            motion_prompt=f"the bald man walks from {a} to {b}", duration=10,
            sound_prompt=f"sound {a}", output_path=f"clips/{a}_to_{b}.mp4",
        )
        for a, b in zip(names, names[1:])
    ]
    sb = Storyboard(
        project_title="t", style="style", characters=list(cast or []),
        frames=frames, transitions=transitions,
    )
    sb.save(workspace.default_storyboard_json)
    return sb


class _NoPlanner:
    """Fails loudly if anything tries to spend a planning call."""

    def plan_transitions(self, *a, **k):  # noqa: ANN002, ANN003
        raise AssertionError("a forced camera transition must not call the planner")


class TestForcingACameraTransition:
    def _run(self, make_pipeline, workspace, **options):
        pipeline = make_pipeline(analyze_frames=False, **options)
        pipeline.__dict__["openai"] = _NoPlanner()
        pairs = _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(
            workspace, ["a", "b", "c"],
            people={
                "a": [FramePerson(id="c1", x=0.3), FramePerson(id="c2", x=0.7)],
                "b": [FramePerson(id="c1", x=0.7), FramePerson(id="c2", x=0.3)],
                "c": [FramePerson(id="c1", x=0.5)],
            },
            cast=[Character(id="c1", epithet="the bald man"),
                  Character(id="c2", epithet="the tall woman")],
        )
        return pipeline._reconcile_storyboard(saved, pairs)

    def test_the_named_pair_becomes_a_camera_transition(
        self, make_pipeline, workspace
    ):
        sb, replanned, _ = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        by_id = {t.id: t for t in sb.transitions}
        assert is_camera_transition(by_id["a_to_b"].motion_prompt)
        assert replanned == ["a_to_b"]

    def test_every_other_pair_is_left_alone(self, make_pipeline, workspace):
        sb, _, _ = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        by_id = {t.id: t for t in sb.transitions}
        assert by_id["b_to_c"].motion_prompt == "the bald man walks from b to c"
        assert by_id["b_to_c"].duration == 10

    def test_it_is_always_a_five_second_clip(self, make_pipeline, workspace):
        # A camera move is ONE continuous beat; ten seconds of it is a slack
        # shot at twice the price. It overrides the saved 10s.
        sb, _, _ = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        assert {t.id: t.duration for t in sb.transitions}["a_to_b"] == 5

    def test_the_sound_prompt_survives(self, make_pipeline, workspace):
        # Sound feeds a different step entirely; a changed picture is no
        # reason to throw away what someone wrote about the audio.
        sb, _, _ = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        assert {t.id: t.sound_prompt for t in sb.transitions}["a_to_b"] == "sound a"

    def test_the_rendered_clip_is_marked_outdated_not_deleted(
        self, make_pipeline, workspace
    ):
        _, _, stale = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        assert "a_to_b" in stale

    def test_it_costs_nothing(self, make_pipeline, workspace):
        # _NoPlanner raises if the planner is touched. Reaching the end of
        # this call at all is the assertion.
        self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b", "b_to_c"], replan_as_camera=True,
        )

    def test_the_wording_suits_the_pair(self, make_pipeline, workspace):
        # People stand on both sides of a_to_b, so the transit needs neutral
        # ground: the turn-away-through-scenery shape, not a pan that drags
        # faces through the blur.
        sb, _, _ = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_as_camera=True,
        )
        prompt = {t.id: t.motion_prompt for t in sb.transitions}["a_to_b"]
        assert "turns away" in prompt

    def test_replan_all_can_be_camera(self, make_pipeline, workspace):
        sb, replanned, _ = self._run(
            make_pipeline, workspace, replan_all=True, replan_as_camera=True,
        )
        assert sorted(replanned) == ["a_to_b", "b_to_c"]
        assert all(is_camera_transition(t.motion_prompt) for t in sb.transitions)

    def test_without_the_flag_nothing_becomes_a_camera_move(
        self, make_pipeline, workspace
    ):
        # The ordinary re-plan path still goes through the planner, so the
        # no-planner stub is what proves the flag is doing the work.
        pipeline = make_pipeline(analyze_frames=False, replan_clips=["a_to_b"])
        pairs = _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(workspace, ["a", "b", "c"])
        sb, replanned, _ = pipeline._reconcile_storyboard(saved, pairs)
        by_id = {t.id: t for t in sb.transitions}
        assert not is_camera_transition(by_id["a_to_b"].motion_prompt)

    def test_a_pair_that_merely_changed_is_never_silently_camera_d(
        self, make_pipeline, workspace
    ):
        # --camera applies to EXPLICITLY named pairs. A pair that is dirty
        # because its photo changed must keep going through the planner.
        pipeline = make_pipeline(
            analyze_frames=False, replan_clips=["a_to_b"], replan_as_camera=True
        )
        pairs = _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(workspace, ["a", "b", "c"])
        # Make c newer than the storyboard so b_to_c is dirty on its own.
        _touch(workspace.styled_images_dir / "c.png", b"changed")
        sb, _, _ = pipeline._reconcile_storyboard(saved, pairs)
        by_id = {t.id: t for t in sb.transitions}
        assert is_camera_transition(by_id["a_to_b"].motion_prompt)
        assert not is_camera_transition(by_id["b_to_c"].motion_prompt)

    def test_an_unknown_id_still_fails_loudly(self, make_pipeline, workspace):
        from ai_video_maker.errors import PipelineError

        pipeline = make_pipeline(
            analyze_frames=False, replan_clips=["nope"], replan_as_camera=True
        )
        pairs = _frame_pairs(workspace, ["a", "b"])
        saved = _saved(workspace, ["a", "b"])
        with pytest.raises(PipelineError):
            pipeline._reconcile_storyboard(saved, pairs)


class TestForcingThePeopleInstead:
    """--stage-people: the mirror override, for a pair the gate over-called.

    Unlike forcing the camera (a template, so free and callless), this one
    PLANS — it just tells the coercion not to throw the choreography away.
    So what these assert is that the request reaches the planner correctly
    and only for pairs that asked for it.
    """

    class _Recorder:
        """Captures what the planner was asked, and plans something visible."""

        def __init__(self):
            self.calls = []

        def analyze_frame_transitions(self, frames, style, **kwargs):
            self.calls.append(kwargs)
            return (
                [("staged people motion", 10, "sound")] * (len(frames) - 1),
                list(kwargs.get("cast") or []),
            )

    def _run(self, make_pipeline, workspace, **options):
        pipeline = make_pipeline(analyze_frames=True, **options)
        recorder = self._Recorder()
        pipeline.__dict__["openai"] = recorder
        pairs = _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(
            workspace, ["a", "b", "c"],
            people={
                "a": [FramePerson(id="c1", x=0.3), FramePerson(id="c2", x=0.7)],
                "b": [FramePerson(id="c1", x=0.7), FramePerson(id="c2", x=0.3)],
                "c": [FramePerson(id="c1", x=0.5)],
            },
            cast=[Character(id="c1", epithet="the bald man"),
                  Character(id="c2", epithet="the tall woman")],
        )
        return pipeline._reconcile_storyboard(saved, pairs), recorder

    def test_the_named_pair_is_flagged_to_the_planner(
        self, make_pipeline, workspace
    ):
        _, rec = self._run(
            make_pipeline, workspace,
            replan_clips=["a_to_b"], replan_stages_people=True,
        )
        assert rec.calls, "the planner was never called"
        assert rec.calls[0]["stage_anyway"] == [True]

    def test_without_the_flag_the_planner_is_told_nothing_of_the_sort(
        self, make_pipeline, workspace
    ):
        _, rec = self._run(make_pipeline, workspace, replan_clips=["a_to_b"])
        assert rec.calls[0]["stage_anyway"] == [False]

    def test_only_the_requested_pair_is_flagged(self, make_pipeline, workspace):
        """A pair dirty because its PHOTO changed must not be force-staged.

        Same rule as the camera override: an approval is for the pairs the
        human named, never for whatever else happened to be dirty.
        """
        pipeline = make_pipeline(
            analyze_frames=True, replan_clips=["a_to_b"],
            replan_stages_people=True,
        )
        rec = self._Recorder()
        pipeline.__dict__["openai"] = rec
        pairs = _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(workspace, ["a", "b", "c"])
        # Dirty b_to_c by changing c's styled image.
        _touch(workspace.styled_images_dir / "c.png", b"changed")
        pipeline._reconcile_storyboard(saved, pairs)
        flags = [f for call in rec.calls for f in call["stage_anyway"]]
        # a_to_b asked for it; b_to_c only became dirty.
        assert flags == [True, False]

    def test_replan_all_can_stage_people(self, make_pipeline, workspace):
        _, rec = self._run(
            make_pipeline, workspace,
            replan_all=True, replan_stages_people=True,
        )
        assert rec.calls[0]["stage_anyway"] == [True, True]

    def test_asking_for_both_overrides_is_refused(self, make_pipeline, workspace):
        """They ask for opposite things; guessing which one wins is worse."""
        with pytest.raises(PipelineError, match="Pass one"):
            self._run(
                make_pipeline, workspace, replan_clips=["a_to_b"],
                replan_as_camera=True, replan_stages_people=True,
            )


class TestCameraTransitionsAreVisibleToThePanel:
    def test_the_snapshot_names_them(self, pipeline, workspace):
        # The panel hides "Use camera move" on a pair that already is one,
        # and recognition is strict-family — so it reads the answer from the
        # server rather than keeping a second copy of the prefix list.
        _frame_pairs(workspace, ["a", "b", "c"])
        saved = _saved(workspace, ["a", "b", "c"])
        saved.transitions[0].motion_prompt = (
            "The camera moves smoothly and steadily across to the terrace."
        )
        saved.save(workspace.default_storyboard_json)
        snap = pipeline.snapshot()
        assert snap["storyboard"]["camera_transitions"] == ["a_to_b"]
