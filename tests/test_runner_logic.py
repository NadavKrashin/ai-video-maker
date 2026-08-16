"""Pipeline pure-logic tests: bridging, pair building, selection, combine list."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from ai_video_maker.errors import PipelineCancelled, PipelineError
from ai_video_maker.models import (
    Character,
    Frame,
    FramePerson,
    Storyboard,
    Transition,
    outdated_identity_plans,
)


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _storyboard(workspace, n_frames: int = 4, durations=None) -> Storyboard:
    frames = [
        Frame(
            id=f"{i:03d}",
            description="",
            image_prompt="",
            output_path=f"styled_images/{i:03d}_styled.png",
        )
        for i in range(1, n_frames + 1)
    ]
    durations = durations or [5] * (n_frames - 1)
    transitions = [
        Transition(
            id=f"{a.id}_to_{b.id}",
            start_frame=a.output_path,
            end_frame=b.output_path,
            motion_prompt=f"motion {a.id}",
            duration=durations[i],
            sound_prompt=f"sound {a.id}",
            output_path=f"clips/{a.id}_to_{b.id}.mp4",
        )
        for i, (a, b) in enumerate(zip(frames, frames[1:]))
    ]
    return Storyboard(
        project_title="t", style="style", frames=frames, transitions=transitions
    )


def _write_frames(workspace, ids):
    for i in ids:
        _touch(workspace.styled_images_dir / f"{i:03d}_styled.png")


class TestBridgePairs:
    def test_all_present(self, pipeline, workspace):
        _write_frames(workspace, [1, 2, 3])
        frames = [workspace.styled_images_dir / f"{i:03d}_styled.png" for i in (1, 2, 3)]
        pairs = pipeline._bridge_pairs(frames)
        assert [(a.name, b.name) for a, b in pairs] == [
            ("001_styled.png", "002_styled.png"),
            ("002_styled.png", "003_styled.png"),
        ]

    def test_missing_middle_is_bridged(self, pipeline, workspace):
        _write_frames(workspace, [1, 3])
        frames = [workspace.styled_images_dir / f"{i:03d}_styled.png" for i in (1, 2, 3)]
        pairs = pipeline._bridge_pairs(frames)
        assert [(a.name, b.name) for a, b in pairs] == [
            ("001_styled.png", "003_styled.png"),
        ]

    def test_dry_run_assumes_all_exist(self, make_pipeline, workspace):
        p = make_pipeline(dry_run=True)
        frames = [workspace.styled_images_dir / f"{i:03d}_styled.png" for i in (1, 2)]
        assert len(p._bridge_pairs(frames)) == 1


class TestBridgingIsAttributedAndVisible:
    """A bridged frame means a photo the customer paid for is not in the movie.

    The warning used to name neither the project nor the frames, and it fired
    from `snapshot()` — which runs for EVERY project on every panel poll. One
    landed in the middle of an unrelated project's styling run and read as if
    it belonged to it.
    """

    def _frames(self, workspace, ids):
        return [workspace.styled_images_dir / f"{i:03d}_styled.png" for i in ids]

    def test_the_warning_names_the_project_and_the_frames(
        self, pipeline, workspace, caplog
    ):
        _write_frames(workspace, [1, 3])
        with caplog.at_level("WARNING"):
            pipeline._bridge_pairs(self._frames(workspace, (1, 2, 3)))
        logged = "\n".join(r.getMessage() for r in caplog.records)
        assert workspace.root.name in logged
        assert "002_styled.png" in logged

    def test_reading_status_says_nothing(self, pipeline, workspace, caplog):
        _write_frames(workspace, [1, 3])
        with caplog.at_level("WARNING"):
            pipeline._bridge_pairs(self._frames(workspace, (1, 2, 3)), quiet=True)
        assert not caplog.records

    def test_nothing_missing_is_silent(self, pipeline, workspace, caplog):
        _write_frames(workspace, [1, 2])
        with caplog.at_level("WARNING"):
            pipeline._bridge_pairs(self._frames(workspace, (1, 2)))
        assert not caplog.records


class TestCancellation:
    """Cooperative cancel: finish the in-flight item, skip the rest, raise."""

    def test_cancel_mid_batch_skips_the_rest_and_raises(self, pipeline):
        done = []

        def worker(i):
            done.append(i)
            if i == 2:
                pipeline.cancel_event.set()  # as the API's cancel would

        with pytest.raises(PipelineCancelled):
            pipeline._map_parallel([1, 2, 3, 4], worker, "t")
        assert done == [1, 2]  # item 3 and 4 never start

    def test_cancel_before_batch_runs_nothing(self, pipeline):
        pipeline.cancel_event.set()
        with pytest.raises(PipelineCancelled):
            pipeline._map_parallel([1, 2], lambda i: pytest.fail("must not run"), "t")

    def test_cancel_blocks_confirm_gates(self, pipeline):
        pipeline.cancel_event.set()
        with pytest.raises(PipelineCancelled):
            pipeline._ask(["info"], "spend money?", "declined")

    def test_no_cancel_no_change(self, pipeline):
        done = []
        pipeline._map_parallel([1, 2], done.append, "t")
        assert done == [1, 2]


class TestPairsFromStoryboard:
    def test_uses_per_transition_plan(self, pipeline, workspace):
        _write_frames(workspace, [1, 2, 3])
        sb = _storyboard(workspace, 3, durations=[5, 10])
        pairs = pipeline._pairs_from_storyboard(sb)
        assert [(m, d, s) for _, _, m, d, s in pairs] == [
            ("motion 001", 5, "sound 001"),
            ("motion 002", 10, "sound 002"),
        ]

    def test_run_options_override(self, make_pipeline, workspace):
        _write_frames(workspace, [1, 2])
        p = make_pipeline(motion_prompt="override", duration=10)
        pairs = p._pairs_from_storyboard(_storyboard(workspace, 2))
        assert pairs[0][2] == "override"
        assert pairs[0][3] == 10

    def test_bridged_pair_inherits_start_transition(self, pipeline, workspace):
        _write_frames(workspace, [1, 3])  # frame 2 missing
        sb = _storyboard(workspace, 3, durations=[5, 10])
        pairs = pipeline._pairs_from_storyboard(sb)
        assert len(pairs) == 1
        start, end, motion, duration, sound = pairs[0]
        assert (start.name, end.name) == ("001_styled.png", "003_styled.png")
        assert motion == "motion 001" and duration == 5 and sound == "sound 001"

    def test_derives_transitions_when_absent(self, pipeline, workspace):
        _write_frames(workspace, [1, 2])
        sb = _storyboard(workspace, 2)
        sb.transitions = []
        pairs = pipeline._pairs_from_storyboard(sb)
        assert pairs[0][2] == "style"  # falls back to the storyboard style
        assert pairs[0][3] == sb.duration_per_clip

    def test_global_motion_prompt_prepended_to_every_clip(
        self, pipeline, workspace
    ):
        _write_frames(workspace, [1, 2, 3])
        sb = _storyboard(workspace, 3)
        sb.global_motion_prompt = "Two separate kids, never merged"
        pairs = pipeline._pairs_from_storyboard(sb)
        assert [m for _, _, m, _, _ in pairs] == [
            "Two separate kids, never merged. motion 001",
            "Two separate kids, never merged. motion 002",
        ]

    def test_global_motion_prompt_rides_along_with_cli_override(
        self, make_pipeline, workspace
    ):
        # --motion-prompt replaces the per-clip action, but whole-movie facts
        # (identities, "never merge them") must still reach the video model.
        _write_frames(workspace, [1, 2])
        p = make_pipeline(motion_prompt="override")
        sb = _storyboard(workspace, 2)
        sb.global_motion_prompt = "One boy throughout."
        pairs = p._pairs_from_storyboard(sb)
        assert pairs[0][2] == "One boy throughout. override"

    def test_empty_global_motion_prompt_changes_nothing(
        self, pipeline, workspace
    ):
        _write_frames(workspace, [1, 2])
        pairs = pipeline._pairs_from_storyboard(_storyboard(workspace, 2))
        assert pairs[0][2] == "motion 001"


class TestSelectClips:
    def _pairs(self, pipeline, workspace, n=3):
        _write_frames(workspace, range(1, n + 1))
        return pipeline._pairs_from_storyboard(_storyboard(workspace, n))

    def test_no_selection_passthrough(self, pipeline, workspace):
        pairs = self._pairs(pipeline, workspace)
        selected, forced = pipeline._select_clips(pairs)
        assert selected == pairs and forced == set()

    def test_selects_and_forces(self, make_pipeline, workspace):
        p = make_pipeline(clips=["002_to_003.mp4"])  # .mp4 suffix tolerated
        pairs = self._pairs(p, workspace)
        selected, forced = p._select_clips(pairs)
        assert len(selected) == 1
        assert selected[0][0].name == "002_styled.png"
        assert forced == {"002_to_003"}

    def test_unknown_clip_raises(self, make_pipeline, workspace):
        p = make_pipeline(clips=["009_to_010"])
        pairs = self._pairs(p, workspace)
        with pytest.raises(PipelineError, match="009_to_010"):
            p._select_clips(pairs)


class TestPlanLines:
    def test_counts_render_vs_skip(self, pipeline, workspace):
        _write_frames(workspace, [1, 2, 3])
        pairs = pipeline._pairs_from_storyboard(_storyboard(workspace, 3))
        _touch(workspace.clips_dir / "001_to_002.mp4")
        lines, to_render = pipeline._plan_lines(pairs, forced=set())
        assert to_render == 1
        assert any("001_to_002" in l and "skip" in l for l in lines)
        assert any("002_to_003" in l and "RENDER" in l for l in lines)

    def test_forced_existing_clip_renders(self, pipeline, workspace):
        _write_frames(workspace, [1, 2])
        pairs = pipeline._pairs_from_storyboard(_storyboard(workspace, 2))
        _touch(workspace.clips_dir / "001_to_002.mp4")
        _, to_render = pipeline._plan_lines(pairs, forced={"001_to_002"})
        assert to_render == 1


class TestClipsForCombine:
    def test_falls_back_to_directory_without_storyboard(self, pipeline, workspace):
        _touch(workspace.clips_dir / "001_to_002.mp4")
        assert [c.name for c in pipeline._clips_for_combine()] == ["001_to_002.mp4"]

    def test_ignores_stray_and_stale_bridged_clips(self, pipeline, workspace):
        _write_frames(workspace, [1, 2, 3])
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        _touch(workspace.clips_dir / "001_to_002.mp4")
        _touch(workspace.clips_dir / "002_to_003.mp4")
        # stale bridged clip from when frame 002 was missing:
        _touch(workspace.clips_dir / "001_to_003.mp4")
        assert [c.name for c in pipeline._clips_for_combine()] == [
            "001_to_002.mp4",
            "002_to_003.mp4",
        ]

    def test_bridged_naming_when_frame_missing(self, pipeline, workspace):
        _write_frames(workspace, [1, 3])  # frame 2 gone
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        _touch(workspace.clips_dir / "001_to_003.mp4")
        _touch(workspace.clips_dir / "001_to_002.mp4")  # stale, frame 2 era
        assert [c.name for c in pipeline._clips_for_combine()] == ["001_to_003.mp4"]


def _make_frame_pairs(workspace, names, age=100.0):
    """Create input+styled files for `names` (slug scheme), aged `age` seconds."""
    now = time.time()
    pairs = []
    for n in names:
        src = _touch(workspace.input_images_dir / f"{n}.jpg")
        dst = _touch(workspace.styled_images_dir / f"{n}.png")
        os.utime(src, (now - age, now - age))
        os.utime(dst, (now - age, now - age))
        pairs.append((src, dst))
    return pairs


def _save_slug_storyboard(workspace, names) -> Storyboard:
    frames = [
        Frame(id=n, description="", image_prompt="",
              output_path=f"styled_images/{n}.png",
              source_path=f"input_images/{n}.jpg")
        for n in names
    ]
    transitions = [
        Transition(id=f"{a}_to_{b}",
                   start_frame=f"styled_images/{a}.png",
                   end_frame=f"styled_images/{b}.png",
                   motion_prompt=f"motion {a}", duration=5,
                   sound_prompt=f"sound {a}",
                   output_path=f"clips/{a}_to_{b}.mp4")
        for a, b in zip(names, names[1:])
    ]
    sb = Storyboard(project_title="t", style="style",
                    global_motion_prompt="two separate kids, never merged",
                    frames=frames, transitions=transitions)
    sb.save(workspace.default_storyboard_json)
    return sb


class TestReconcileStoryboard:
    """The surgical-edit merge: keep untouched transitions, re-plan the rest."""

    def _pipeline(self, make_pipeline):
        # analysis off -> deterministic fallback plans (config.motion_prompt)
        return make_pipeline(analyze_frames=False)

    def _save_with_placeholder(self, workspace, p, names):
        """A saved storyboard whose first transition still has the config
        fallback prompt — i.e. planning failed for it on a previous run."""
        sb = _save_slug_storyboard(workspace, names)
        sb.transitions[0].motion_prompt = p.config.motion_prompt
        sb.save(workspace.default_storyboard_json)
        return Storyboard.load(workspace.default_storyboard_json)

    def test_unchanged_project_keeps_everything(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == [] and stale == []
        assert [t.motion_prompt for t in sb.transitions] == ["motion a", "motion b"]

    def test_placeholder_prompt_is_replanned_not_locked_in(
        self, make_pipeline, workspace
    ):
        # A real order shipped 26/29 generic prompts because planning died on
        # insufficient_quota; re-running storyboard must retry exactly those.
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = self._save_with_placeholder(workspace, p, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b"]  # hand-written "motion b" untouched
        # analysis off -> the re-plan fell back to the same placeholder:
        # nothing improved, so the existing clip must NOT be invalidated
        assert stale == []
        assert sb.transitions[1].motion_prompt == "motion b"

    def test_replanned_placeholder_marks_its_clip_stale(
        self, make_pipeline, workspace
    ):
        # --motion-prompt stands in for a successful vision plan here: the
        # placeholder gets a genuinely new prompt, so its old clip is stale.
        p = make_pipeline(analyze_frames=False, motion_prompt="a real plan")
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = self._save_with_placeholder(workspace, p, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert sb.transitions[0].motion_prompt == "a real plan"
        assert stale == ["a_to_b"]  # confirm-gated deletion downstream
        # user-level fields carried over
        assert sb.global_motion_prompt == "two separate kids, never merged"

    def test_a_replan_that_only_changes_the_duration_marks_the_clip_stale(
        self, make_pipeline, workspace
    ):
        # A 5s clip on disk against a 10s plan is as outdated as a rewritten
        # prompt — and it is exactly what confirmed identities produce, since
        # a detected arrangement swap forces the pair long.
        p = make_pipeline(analyze_frames=False, duration=10)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        # Same wording, different length: only the duration moved.
        p.options.replan_clips = ["a_to_b"]
        p.options.motion_prompt = saved.transitions[0].motion_prompt
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert sb.transitions[0].motion_prompt == saved.transitions[0].motion_prompt
        assert sb.transitions[0].duration == 10
        assert stale == ["a_to_b"]

    def test_a_replan_that_changes_nothing_leaves_the_clip_alone(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        p.options.replan_clips = ["a_to_b"]
        p.options.motion_prompt = saved.transitions[0].motion_prompt
        p.options.duration = saved.transitions[0].duration
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert stale == []

    def test_inserted_frame_replans_only_its_two_pairs(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        _save_slug_storyboard(workspace, ["a", "b", "c"])
        saved = Storyboard.load(workspace.default_storyboard_json)
        pairs = _make_frame_pairs(workspace, ["a", "x", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_x", "x_to_b"]
        assert stale == []  # frames a/b/c themselves are unchanged
        kept = {t.id: t.motion_prompt for t in sb.transitions}
        assert kept["b_to_c"] == "motion b"  # hand-editable transition survived

    def test_requested_replan_forces_fresh_plan_and_marks_stale(
        self, make_pipeline, workspace
    ):
        # The panel's per-clip "re-plan prompt": frames unchanged, but the
        # user wants a new plan for exactly this pair. --motion-prompt stands
        # in for a successful vision plan; the old clip goes stale (marked,
        # never deleted), the neighbour's hand-written prompt survives.
        p = make_pipeline(
            analyze_frames=False, motion_prompt="a fresh plan",
            replan_clips=["a_to_b"],
        )
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b"] and stale == ["a_to_b"]
        assert sb.transitions[0].motion_prompt == "a fresh plan"
        assert sb.transitions[1].motion_prompt == "motion b"

    def test_requested_replan_same_plan_is_not_stale(
        self, make_pipeline, workspace
    ):
        # Re-plan came back identical (or fell back) -> the clip still
        # matches its prompt; don't flag it outdated.
        p = make_pipeline(
            analyze_frames=False, motion_prompt="motion a",
            replan_clips=["a_to_b"],
        )
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b"] and stale == []

    def test_requested_replan_unknown_id_fails_loudly(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False, replan_clips=["nope_to_nada"])
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        with pytest.raises(PipelineError, match="nope_to_nada"):
            p._reconcile_storyboard(saved, pairs)

    def test_removed_frame_replans_the_joined_pair(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        pairs = _make_frame_pairs(workspace, ["a", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_c"] and stale == []
        assert len(sb.transitions) == 1

    def test_restyled_frame_marks_adjacent_pairs_stale(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        # b was re-styled AFTER the storyboard was written
        future = time.time() + 50
        os.utime(workspace.styled_images_dir / "b.png", (future, future))
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b", "b_to_c"]
        assert stale == ["a_to_b", "b_to_c"]

    def test_touched_but_identical_frames_are_not_stale(
        self, make_pipeline, workspace
    ):
        # The real-world wipeout: a sync tool bumped every styled image's
        # mtime without changing content, and mtime-based staleness deleted
        # every rendered clip. Content hashes must shrug that off.
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        first, _, _ = p._reconcile_storyboard(None, pairs)  # records hashes
        for t in first.transitions:  # as if planning had succeeded
            t.motion_prompt = f"planned {t.id}"
        first.save(workspace.default_storyboard_json)
        saved = Storyboard.load(workspace.default_storyboard_json)
        future = time.time() + 50
        for n in ("a", "b", "c"):
            os.utime(workspace.styled_images_dir / f"{n}.png", (future, future))
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == [] and stale == []

    def test_changed_content_is_stale_even_with_old_mtime(
        self, make_pipeline, workspace
    ):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        first, _, _ = p._reconcile_storyboard(None, pairs)
        first.save(workspace.default_storyboard_json)
        saved = Storyboard.load(workspace.default_storyboard_json)
        b = workspace.styled_images_dir / "b.png"
        b.write_bytes(b"restyled content")
        past = time.time() - 50  # mtime says "unchanged"; content says otherwise
        os.utime(b, (past, past))
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b", "b_to_c"]
        assert stale == ["a_to_b", "b_to_c"]

    def test_no_saved_storyboard_plans_all(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(None, pairs)
        assert replanned == ["a_to_b", "b_to_c"] and stale == []
        assert all(t.motion_prompt == p.config.motion_prompt for t in sb.transitions)

    def test_cast_is_carried_over_when_nothing_is_replanned(
        self, make_pipeline, workspace
    ):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        saved.characters = [Character(id="bald-man", epithet="the bald man")]
        sb, replanned, _ = p._reconcile_storyboard(saved, pairs)
        assert replanned == []
        assert [c.epithet for c in sb.characters] == ["the bald man"]

    def test_planning_off_still_preserves_the_cast(self, make_pipeline, workspace):
        # --no-analyze / dry-run take the fallback path; losing the cast there
        # would silently drop the movie's identity anchor.
        p = self._pipeline(make_pipeline)
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        saved.characters = [Character(id="bald-man", epithet="the bald man")]
        pairs = _make_frame_pairs(workspace, ["a", "x", "b", "c"])
        sb, replanned, _ = p._reconcile_storyboard(saved, pairs)
        assert replanned  # the inserted frame's pairs were (fallback-)planned
        assert [c.epithet for c in sb.characters] == ["the bald man"]


class TestCastAcrossPlanningCalls:
    """The cast list is what keeps separately planned pairs naming people the
    same way. A targeted re-plan sees only its own two frames, so without the
    saved cast riding along it invents a fresh epithet for someone the rest of
    the movie already calls something else — and the video model then has no
    way to know it is the same person from clip to clip.
    """

    def _pipeline_with_planner(self, make_pipeline, new_people=None):
        """A pipeline whose vision call is stubbed; records the cast it saw."""
        p = make_pipeline(analyze_frames=True)
        seen: list[list[str]] = []
        # One batch of "new" people per call, so a multi-segment run can show
        # the cast accumulating.
        batches = list(new_people or [])

        def fake_analyze(frames, style, default_duration=None,
                         global_context="", cast=None, lessons=None,
                         frame_people=None):
            seen.append([c.epithet for c in (cast or [])])
            found = batches.pop(0) if batches else []
            merged = list(cast or []) + [
                Character(id=e.replace(" ", "-"), epithet=e) for e in found
            ]
            plans = [("planned motion", 5, "sound")] * (len(frames) - 1)
            return plans, merged

        p.openai.analyze_frame_transitions = fake_analyze
        return p, seen

    def test_saved_cast_is_handed_to_the_planner(self, make_pipeline, workspace):
        p, seen = self._pipeline_with_planner(make_pipeline)
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        saved.characters = [
            Character(id="bald-man", epithet="the bald man in pink sunglasses")
        ]
        pairs = _make_frame_pairs(workspace, ["a", "x", "b", "c"])
        sb, _, _ = p._reconcile_storyboard(saved, pairs)
        assert seen == [["the bald man in pink sunglasses"]]
        assert [c.epithet for c in sb.characters] == [
            "the bald man in pink sunglasses"
        ]

    def test_new_people_reach_the_storyboard(self, make_pipeline, workspace):
        p, _ = self._pipeline_with_planner(
            make_pipeline, new_people=[["the small girl with curly hair"]]
        )
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        sb, _, _ = p._reconcile_storyboard(None, pairs)
        assert [c.epithet for c in sb.characters] == [
            "the small girl with curly hair"
        ]

    def test_cast_accumulates_across_segments_in_one_run(
        self, make_pipeline, workspace
    ):
        # Two non-adjacent dirty pairs = two vision calls. The second must be
        # told what the first discovered, or the same person gets two names.
        p, seen = self._pipeline_with_planner(
            make_pipeline, new_people=[["the bald man"], ["the woman in red"]]
        )
        saved = _save_slug_storyboard(workspace, ["a", "b", "c", "d", "e"])
        pairs = _make_frame_pairs(workspace, ["a", "b", "c", "d", "e"])
        # Dirty the first and last pairs only, leaving a clean pair between.
        for name in ("a", "e"):
            path = workspace.styled_images_dir / f"{name}.png"
            path.write_bytes(b"changed")
        sb, _, _ = p._reconcile_storyboard(saved, pairs)
        assert seen == [[], ["the bald man"]]
        assert [c.epithet for c in sb.characters] == [
            "the bald man", "the woman in red",
        ]

    def test_planning_failure_keeps_the_cast_intact(self, make_pipeline, workspace):
        # "A planning hiccup never sinks the run" — and must not wipe the cast.
        p = make_pipeline(analyze_frames=True)

        def boom(*_a, **_kw):
            raise RuntimeError("vision call failed")

        p.openai.analyze_frame_transitions = boom
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        saved.characters = [Character(id="bald-man", epithet="the bald man")]
        pairs = _make_frame_pairs(workspace, ["a", "x", "b", "c"])
        sb, _, _ = p._reconcile_storyboard(saved, pairs)
        assert [c.epithet for c in sb.characters] == ["the bald man"]


class TestStaleMarking:
    def _pipeline(self, make_pipeline):
        return make_pipeline(analyze_frames=False)

    def test_stale_clips_are_marked_never_deleted(self, make_pipeline, workspace):
        # The admin API's confirm callback auto-answers yes, so any deletion
        # behind a confirm gate is a wipeout waiting to happen (26 rendered
        # clips died this way on a real order). Staleness only marks state.
        p = self._pipeline(make_pipeline)
        clip = _touch(workspace.clips_dir / "a_to_b.mp4")
        p.state.set("sfx:a_to_b.mp4", "done")
        p._mark_stale_clips(["a_to_b", "b_to_c"])  # b_to_c has no file
        assert clip.exists()
        assert p.state.is_done("sfx:a_to_b.mp4")  # audio state untouched too
        assert p.state.status("stale:a_to_b.mp4") == "outdated"
        assert p.state.status("stale:b_to_c.mp4") is None  # nothing to outdate

    def test_stale_mark_survives_until_regeneration(self, make_pipeline, workspace):
        # render must keep skipping an existing stale clip; only an explicit
        # per-clip regeneration (render --clip / panel button) clears the flag.
        p = self._pipeline(make_pipeline)
        _touch(workspace.clips_dir / "a_to_b.mp4")
        p._mark_stale_clips(["a_to_b"])

        rendered = []
        p.video_client = type("Stub", (), {
            "generate_clip":
                lambda self, s, e, m, d, dst, reword=None, elements=None:
                    rendered.append(dst.name) or _touch(dst)
        })()
        start = _touch(workspace.styled_images_dir / "a.png")
        end = _touch(workspace.styled_images_dir / "b.png")
        pair = (start, end, "motion", 5, "")

        p._generate_clips([pair], forced=set())  # exists + not forced -> skip
        assert rendered == []
        assert p.state.status("stale:a_to_b.mp4") == "outdated"

        p._generate_clips([pair], forced={"a_to_b"})  # explicit regenerate
        assert rendered == ["a_to_b.mp4"]
        assert p.state.status("stale:a_to_b.mp4") is None


class TestStyledTargets:
    def test_slug_naming_by_default(self, pipeline, workspace):
        imgs = [_touch(workspace.input_images_dir / n)
                for n in ("My Photo (1).jpg", "img4a.png")]
        targets = pipeline._styled_targets(imgs)
        assert [t.name for t in targets] == ["My_Photo_1.png", "img4a.png"]

    def test_legacy_positional_when_old_files_exist(self, pipeline, workspace):
        _touch(workspace.styled_images_dir / "001_styled.png")
        imgs = [_touch(workspace.input_images_dir / n) for n in ("a.jpg", "b.jpg")]
        targets = pipeline._styled_targets(imgs)
        assert [t.name for t in targets] == ["001_styled.png", "002_styled.png"]

    def test_slug_collision_raises(self, pipeline, workspace):
        imgs = [_touch(workspace.input_images_dir / "a.jpg"),
                _touch(workspace.input_images_dir / "a.png")]
        with pytest.raises(PipelineError, match="rename"):
            pipeline._styled_targets(imgs)


class TestRestyleDetection:
    """_style_images must redo styled files whose sources changed underneath."""

    def test_newer_source_triggers_restyle(self, make_pipeline, workspace):
        p = make_pipeline(dry_run=True)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        src = pairs[0][0]
        future = time.time() + 50
        os.utime(src, (future, future))  # source replaced after styling
        p._style_images([s for s, _ in pairs], recorded_sources={})
        assert p.summary.styled_created == 1  # only the outdated one
        assert p.summary.styled_skipped == 1

    def test_source_mismatch_triggers_restyle(self, make_pipeline, workspace):
        p = make_pipeline(dry_run=True)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        recorded = {"styled_images/a.png": "input_images/other.jpg"}
        p._style_images([s for s, _ in pairs], recorded_sources=recorded)
        assert p.summary.styled_created == 1
        assert p.summary.styled_skipped == 1

    def test_declining_the_gate_keeps_existing_files(
        self, config, workspace
    ):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline
        p = Pipeline(config, workspace, RunOptions(),
                     confirm=lambda lines, question: False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        future = time.time() + 50
        os.utime(pairs[0][0], (future, future))
        result = p._style_images([s for s, _ in pairs], recorded_sources={})
        assert p.summary.styled_skipped == 2  # nothing restyled, nothing spent
        assert len(result) == 2


class TestSelectAudioClips:
    def _clips(self, workspace):
        return [_touch(workspace.clips_dir / n)
                for n in ("a_to_b.mp4", "b_to_c.mp4")]

    def test_no_selection_passthrough(self, pipeline, workspace):
        clips = self._clips(workspace)
        assert pipeline._select_audio_clips(clips) == clips

    def test_selection_clears_audio_state_for_redo(self, make_pipeline, workspace):
        p = make_pipeline(clips=["a_to_b"])
        clips = self._clips(workspace)
        p.state.set("sfx:a_to_b.mp4", "done")
        p.state.set("sfx:b_to_c.mp4", "done")
        selected = p._select_audio_clips(clips)
        assert [c.name for c in selected] == ["a_to_b.mp4"]
        assert not p.state.is_done("sfx:a_to_b.mp4")   # will be redone
        assert p.state.is_done("sfx:b_to_c.mp4")       # untouched

    def test_unknown_clip_raises(self, make_pipeline, workspace):
        p = make_pipeline(clips=["nope_to_nada"])
        with pytest.raises(PipelineError, match="nope_to_nada"):
            p._select_audio_clips(self._clips(workspace))


class TestPresentationSegments:
    """Optional real-photo segments wrapped around the clip list at combine."""

    def _project(self, workspace):
        _make_frame_pairs(workspace, ["a", "b", "c"])
        _save_slug_storyboard(workspace, ["a", "b", "c"])
        return [_touch(workspace.clips_dir / n)
                for n in ("a_to_b.mp4", "b_to_c.mp4")]

    def _stub_renderers(self, monkeypatch):
        """Replace every ffmpeg/PIL renderer with a touch(); count the calls."""
        import ai_video_maker.runner as runner_mod
        calls: list[str] = []

        class FakeImage:
            height = 2000
            def save(self, path):
                _touch(Path(path))

        def stub(name, dst_index):
            def fake(*args, **kwargs):
                calls.append(name)
                _touch(Path(args[dst_index]))
            return fake

        monkeypatch.setattr(runner_mod, "render_photo_still", stub("still", 1))
        monkeypatch.setattr(runner_mod, "render_intro_segment", stub("intro", 1))
        monkeypatch.setattr(runner_mod, "render_letter_scroll", stub("scroll", 1))
        monkeypatch.setattr(runner_mod, "render_letter_overlay", stub("overlay", 2))
        monkeypatch.setattr(runner_mod, "combine_clips", stub("concat", 1))
        monkeypatch.setattr(
            runner_mod, "find_letter_font", lambda explicit="": "/fake/font.ttc"
        )
        monkeypatch.setattr(
            runner_mod, "render_letter_image",
            lambda *a, **k: calls.append("image") or FakeImage(),
        )
        return calls

    def test_off_by_default_passthrough(self, pipeline, workspace):
        clips = self._project(workspace)
        segments, added = pipeline._presentation_segments(clips)
        assert segments == clips and added is False

    def test_credits_appends_one_still_per_photo(
        self, make_pipeline, workspace, monkeypatch
    ):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(credits_photos=True)
        clips = self._project(workspace)
        segments, added = p._presentation_segments(clips)
        assert added is True
        assert segments[: len(clips)] == clips
        assert [s.name for s in segments[len(clips):]] == [
            "credits_000_1.50s.mp4", "credits_001_1.50s.mp4",
            "credits_002_1.50s.mp4",
        ]

    def test_no_recorded_sources_skips_cleanly(
        self, make_pipeline, workspace, monkeypatch
    ):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(credits_photos=True)
        # storyboard whose frames carry no source_path (legacy / Mode B)
        _write_frames(workspace, range(1, 4))
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        clips = [_touch(workspace.clips_dir / "001_to_002.mp4")]
        segments, added = p._presentation_segments(clips)
        assert segments == clips and added is False

    def test_cli_override_beats_config(self, make_pipeline):
        p = make_pipeline(
            credits_photos=False, closing_letter=False, intro_clip=False,
        )
        p.config.credits_photos = True
        p.config.closing_letter = True
        p.config.intro_clip = True
        assert p._presentation_flags() == (False, False, False)

    def _global_intro(self, pipeline, workspace) -> Path:
        """Point the shared intro at a tmp file (never the real repo root)."""
        src = _touch(workspace.root / "shared_intro.mp4")
        pipeline.config.intro_file = str(src)
        return src

    def test_intro_prepends_normalized_segment(
        self, make_pipeline, workspace, monkeypatch
    ):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(intro_clip=True)
        clips = self._project(workspace)
        self._global_intro(p, workspace)
        segments, added = p._presentation_segments(clips)
        assert added is True
        assert segments[0].name == "intro.mp4"
        assert segments[1:] == clips
        assert "intro" in calls

    def test_intro_without_file_skips(self, make_pipeline, workspace, monkeypatch):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(intro_clip=True)
        p.config.intro_file = str(workspace.root / "missing_intro.mp4")
        clips = self._project(workspace)
        segments, added = p._presentation_segments(clips)
        assert segments == clips and added is False
        assert "intro" not in calls

    def test_intro_source_resolves_against_repo_root(self, pipeline):
        from ai_video_maker.workspace import PROJECT_ROOT
        assert pipeline._intro_source() == PROJECT_ROOT / "intro.mp4"
        pipeline.config.intro_file = "assets/opener.mp4"
        assert pipeline._intro_source() == PROJECT_ROOT / "assets/opener.mp4"

    def test_letter_over_credits_merges_into_one_section(
        self, make_pipeline, workspace, monkeypatch
    ):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(credits_photos=True, closing_letter=True)
        clips = self._project(workspace)
        workspace.letter_file.write_text("מתן היקר, אוהבים אותך", encoding="utf-8")
        segments, added = p._presentation_segments(clips)
        assert added is True
        # ONE combined section: the letter scrolls over the montage — no
        # separate stills and no standalone letter segment in the movie.
        assert [s.name for s in segments] == [
            "a_to_b.mp4", "b_to_c.mp4", "credits_letter.mp4",
        ]
        assert "overlay" in calls and "scroll" not in calls

    def test_letter_alone_scrolls_on_dark(self, make_pipeline, workspace, monkeypatch):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(closing_letter=True)
        clips = self._project(workspace)
        workspace.letter_file.write_text("שלום", encoding="utf-8")
        segments, added = p._presentation_segments(clips)
        assert added is True and segments[-1].name == "letter.mp4"
        assert "scroll" in calls and "overlay" not in calls

    def test_letter_without_file_skips(self, make_pipeline, workspace, monkeypatch):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(closing_letter=True)
        clips = self._project(workspace)
        segments, added = p._presentation_segments(clips)
        assert segments == clips and added is False

    def test_second_combine_reuses_fresh_segments(
        self, make_pipeline, workspace, monkeypatch
    ):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(
            intro_clip=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        self._global_intro(p, workspace)
        workspace.letter_file.write_text("שלום", encoding="utf-8")
        first, added = p._presentation_segments(clips)
        assert added is True
        rendered_first = len(calls)
        assert rendered_first > 0
        calls.clear()
        second, added = p._presentation_segments(clips)
        assert second == first and added is True
        assert calls == []  # everything reused, nothing re-rendered

    def test_changed_letter_rebuilds_only_letter_section(
        self, make_pipeline, workspace, monkeypatch
    ):
        calls = self._stub_renderers(monkeypatch)
        p = make_pipeline(
            intro_clip=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        self._global_intro(p, workspace)
        workspace.letter_file.write_text("שלום", encoding="utf-8")
        p._presentation_segments(clips)
        calls.clear()
        future = time.time() + 50
        os.utime(workspace.letter_file, (future, future))  # letter edited
        p._presentation_segments(clips)
        assert "intro" not in calls               # untouched -> reused
        assert "overlay" in calls                 # letter section redone

    def _built(self, p, workspace, clips, monkeypatch):
        """Render every section once, then return the call log, cleared."""
        calls = self._stub_renderers(monkeypatch)
        self._global_intro(p, workspace)
        workspace.letter_file.write_text("שלום", encoding="utf-8")
        p._presentation_segments(clips)
        assert calls  # something really was rendered
        calls.clear()
        return calls

    def test_renderer_version_bump_redoes_the_sections(
        self, make_pipeline, workspace, monkeypatch
    ):
        """The bug the user hit: new code, identical ending.

        The reuse check is mtime-based, and shipping a fix touches no
        project file — so a re-combine after the emoji and no-grey-photos
        fixes returned the OLD ending. The renderer version is part of each
        section's recipe precisely so that can't happen.
        """
        import ai_video_maker.runner as runner_mod

        p = make_pipeline(
            intro_clip=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        calls = self._built(p, workspace, clips, monkeypatch)
        monkeypatch.setattr(
            runner_mod, "_SEGMENT_RENDERER_VERSION",
            runner_mod._SEGMENT_RENDERER_VERSION + 1,
        )
        p._presentation_segments(clips)
        assert "overlay" in calls and "intro" in calls and "still" in calls

    def test_changed_letter_setting_redoes_the_letter_section(
        self, make_pipeline, workspace, monkeypatch
    ):
        # Same trap by another route: config edited, no file touched.
        p = make_pipeline(
            intro_clip=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        calls = self._built(p, workspace, clips, monkeypatch)
        p.config.letter_overlay_dim = 0.5
        p._presentation_segments(clips)
        assert "overlay" in calls
        assert "intro" not in calls  # unrelated section stays cached

    def test_force_rebuilds_every_section(
        self, make_pipeline, workspace, monkeypatch
    ):
        # "Force" on the panel's Combine button rebuilt the final video out
        # of cached sections, which is not what anyone means by force.
        p = make_pipeline(
            intro_clip=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        calls = self._built(p, workspace, clips, monkeypatch)
        p.force = True
        p._presentation_segments(clips)
        assert "overlay" in calls and "intro" in calls and "still" in calls


class TestConsecutiveRuns:
    def test_grouping(self):
        from ai_video_maker.runner import _consecutive_runs
        assert _consecutive_runs([]) == []
        assert _consecutive_runs([2]) == [[2]]
        assert _consecutive_runs([0, 1, 2, 5, 7, 8]) == [[0, 1, 2], [5], [7, 8]]


class TestFitCreditsAndLetter:
    def test_long_letter_stretches_photos(self):
        from ai_video_maker.runner import _fit_credits_and_letter
        # 3 photos * 2.5s = 7.5s, but the letter needs 20s -> photos stretch
        per_photo, pps = _fit_credits_and_letter(3, 2.5, 3000, 150)
        assert per_photo == pytest.approx(20 / 3)
        assert pps == pytest.approx(150)  # letter keeps its configured pace

    def test_long_montage_slows_letter(self):
        from ai_video_maker.runner import _fit_credits_and_letter
        # 10 photos * 2.5s = 25s window, letter would take 10s -> slow it down
        per_photo, pps = _fit_credits_and_letter(10, 2.5, 1500, 150)
        assert per_photo == pytest.approx(2.5)  # photos keep their pace
        assert pps == pytest.approx(1500 / 25)


class TestClipName:
    def test_legacy_styled_suffix_stripped(self, pipeline, workspace):
        a = workspace.styled_images_dir / "001_styled.png"
        b = workspace.styled_images_dir / "002_styled.png"
        assert pipeline._clip_name(a, b).name == "001_to_002.mp4"

    def test_slug_names_used_in_full(self, pipeline, workspace):
        a = workspace.styled_images_dir / "img4.png"
        b = workspace.styled_images_dir / "img4a.png"
        assert pipeline._clip_name(a, b).name == "img4_to_img4a.mp4"

    def test_non_numeric_stem_used_as_is(self, pipeline, workspace):
        a = workspace.styled_images_dir / "alpha.png"
        b = workspace.styled_images_dir / "002.png"
        assert pipeline._clip_name(a, b).name == "alpha_to_002.mp4"


class TestRestyleImages:
    """--restyle-frame / the panel's 'Regenerate image': force one styled frame."""

    def test_restyle_forces_named_frame_only(self, make_pipeline, workspace):
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        images = [src for src, _ in pairs]
        p = make_pipeline(dry_run=True, restyle_frames=["b.png"])
        p._style_images(images, {})
        assert p.summary.styled_created == 1  # b would be re-styled
        assert p.summary.styled_skipped == 2  # a, c reused untouched

    def test_unknown_restyle_frame_raises(self, make_pipeline, workspace):
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        images = [src for src, _ in pairs]
        p = make_pipeline(dry_run=True, restyle_frames=["z.png"])
        with pytest.raises(PipelineError, match="z.png"):
            p._style_images(images, {})


class TestPerFrameStyleNote:
    """Frame.style_note steers ONE frame's styling without touching the rest.

    The shared style prompt is written for portraits of people; on a scene
    photo whose only people are a small reflection it reads as "promote them",
    and a real frame came back with two invented faces. The note is the
    per-frame escape hatch.
    """

    def _capture(self, p):
        """Record (source name, prompt) for every style call; skip the network."""
        calls = []

        class _FakeOpenAI:
            def style_image(self, src, prompt, dst, references=None):
                calls.append((src.name, prompt))
                dst.write_bytes(b"styled")

        p.__dict__["openai"] = _FakeOpenAI()
        return calls

    def _pipeline_with_frames(self, make_pipeline, workspace, names):
        pairs = _make_frame_pairs(workspace, names)
        for _, dst in pairs:
            dst.unlink()  # nothing styled yet -> every frame is styled now
        p = make_pipeline()
        return p, [src for src, _ in pairs], self._capture(p)

    def test_note_is_appended_for_that_frame_only(self, make_pipeline, workspace):
        p, images, calls = self._pipeline_with_frames(
            make_pipeline, workspace, ["a", "b"]
        )
        p._style_images(
            images, {}, {"styled_images/a.png": "keep the reflection a reflection"}
        )
        got = dict(calls)
        assert "keep the reflection a reflection" in got["a.jpg"]
        assert got["a.jpg"].startswith("test style")  # shared prompt still applies
        assert got["b.jpg"] == "test style"  # untouched frame unaffected

    def test_blank_note_leaves_the_prompt_alone(self, make_pipeline, workspace):
        p, images, calls = self._pipeline_with_frames(make_pipeline, workspace, ["a"])
        p._style_images(images, {}, {"styled_images/a.png": "   "})
        assert dict(calls)["a.jpg"] == "test style"

    def test_note_is_not_a_restyle_trigger(self, make_pipeline, workspace):
        """A new note must never spend credits on its own — styling resume is
        existence-based, so it lands on the next explicit --restyle-frame."""
        pairs = _make_frame_pairs(workspace, ["a", "b"])  # both already styled
        p = make_pipeline(dry_run=True)
        p._style_images(
            [s for s, _ in pairs], {}, {"styled_images/a.png": "a fresh note"}
        )
        assert p.summary.styled_created == 0
        assert p.summary.styled_skipped == 2

    def test_restyle_frame_applies_the_note(self, make_pipeline, workspace):
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        p = make_pipeline(restyle_frames=["a.png"])
        calls = self._capture(p)
        p._style_images(
            [s for s, _ in pairs], {}, {"styled_images/a.png": "a reflection, not people"}
        )
        assert [n for n, _ in calls] == ["a.jpg"]  # only the named frame
        assert "a reflection, not people" in calls[0][1]

    def test_note_survives_reconcile(self, make_pipeline, workspace):
        """Frames are rebuilt from disk each run; a hand-written note must not
        be dropped by the next `storyboard`."""
        p = make_pipeline(analyze_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        saved.frames[0].style_note = "keep it a facade shot"
        saved.save(workspace.default_storyboard_json)

        reloaded = Storyboard.load(workspace.default_storyboard_json)
        sb, _, _ = p._reconcile_storyboard(reloaded, pairs)
        notes = {f.output_path: f.style_note for f in sb.frames}
        assert notes["styled_images/a.png"] == "keep it a facade shot"
        assert notes["styled_images/b.png"] == ""


class TestResolveMusicFile:
    """The bed is always a supplied track; nothing is ever generated."""

    def test_prefers_uploaded_custom_music(self, make_pipeline, workspace):
        p = make_pipeline()
        _touch(workspace.music_file, b"older")
        _touch(workspace.custom_music_file, b"custom")
        assert p._resolve_music_file() == workspace.custom_music_file

    def test_custom_music_wins_even_under_force(self, make_pipeline, workspace):
        p = make_pipeline(force=True)
        _touch(workspace.music_file, b"older")
        _touch(workspace.custom_music_file, b"custom")
        assert p._resolve_music_file() == workspace.custom_music_file

    def test_music_file_option_wins_over_everything(self, make_pipeline, workspace, tmp_path):
        supplied = _touch(tmp_path / "mine.mp3", b"mine")
        p = make_pipeline(music_file=str(supplied))
        _touch(workspace.custom_music_file, b"custom")
        assert p._resolve_music_file() == supplied

    def test_missing_music_file_option_is_an_error(self, make_pipeline, tmp_path):
        p = make_pipeline(music_file=str(tmp_path / "nope.mp3"))
        with pytest.raises(PipelineError, match="--music-file not found"):
            p._resolve_music_file()

    def test_existing_track_on_disk_is_reused(self, make_pipeline, workspace):
        p = make_pipeline()
        _touch(workspace.music_file, b"older")
        assert p._resolve_music_file() == workspace.music_file

    def test_no_track_means_no_music_not_generation(self, make_pipeline):
        # Nothing to generate from any more: the movie is simply built silent.
        p = make_pipeline()
        assert p._resolve_music_file() is None


class TestSnapshotMissingFrames:
    """A skipped photo is a photo the customer paid for and will not see.

    Render bridges over a frame with no styled image, so the movie plays
    continuously and simply does not contain it — invisible in the finished
    file, and previously invisible everywhere except one log line.
    """

    def test_a_frame_with_no_styled_image_is_reported(self, pipeline, workspace):
        _write_frames(workspace, [1, 3])  # frame 2 never styled
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        assert pipeline.snapshot()["missing_frames"] == ["002_styled.png"]

    def test_a_complete_storyboard_reports_none(self, pipeline, workspace):
        _write_frames(workspace, [1, 2, 3])
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        assert pipeline.snapshot()["missing_frames"] == []

    def test_no_storyboard_is_not_a_pile_of_missing_frames(self, pipeline):
        assert pipeline.snapshot()["missing_frames"] == []


class TestSnapshotMusic:
    def test_reports_custom_and_generated_music(self, pipeline, workspace):
        snap = pipeline.snapshot()
        assert snap["custom_music"] is False and snap["music"] is False
        _touch(workspace.custom_music_file, b"custom")
        _touch(workspace.music_file, b"generated")
        snap = pipeline.snapshot()
        assert snap["custom_music"] is True and snap["music"] is True


class TestSnapshotLetter:
    """The closing letter's presence has to be visible, not guessed.

    Turning `closing_letter` on without a letter written is a silent no-op
    (combine warns to the log and builds the movie without it), so status
    and the panel need to know whether there is any text to scroll.
    """

    def test_no_letter_file(self, pipeline):
        assert pipeline.snapshot()["letter"] == {"exists": False, "chars": 0}

    def test_written_letter_is_counted(self, pipeline, workspace):
        workspace.letter_file.write_text("dear you\n", encoding="utf-8")
        assert pipeline.snapshot()["letter"] == {"exists": True, "chars": 8}

    def test_blank_letter_file_scrolls_nothing(self, pipeline, workspace):
        workspace.letter_file.write_text("\n  \n", encoding="utf-8")
        assert pipeline.snapshot()["letter"] == {"exists": True, "chars": 0}


class TestSeedLetterFromOrder:
    """Ingest pre-fills letter.txt with the customer's blessing.

    The order form already collects it and it is exactly what the closing
    letter scrolls, but nothing carried it across: the text sat in Firestore
    while every letter was retyped by hand.
    """

    FOLDER = "AM-180726-XY12_Dana-Cohen-18.07.2026_10-30"

    def _stub_firebase(self, monkeypatch, *, configured=True, order=None,
                       explode=False):
        """Stand in for the Firestore ledger; records the id that was read."""
        seen: dict[str, str] = {}

        class _Client:
            @staticmethod
            def configured(config):
                return configured

            @staticmethod
            def from_config(config):
                return _Client()

            def get_order(self, order_id):
                seen["order_id"] = order_id
                if explode:
                    raise RuntimeError("firestore is down")
                return order

        monkeypatch.setattr(
            "ai_video_maker.clients.firebase_client.FirebaseClient", _Client)
        return seen

    def _order(self, blessing: str):
        from ai_video_maker.clients.firebase_client import FirestoreOrder

        return FirestoreOrder(order_id="AM-180726-XY12", blessing=blessing)

    def test_blessing_becomes_the_letter(self, make_pipeline, workspace, monkeypatch):
        seen = self._stub_firebase(monkeypatch, order=self._order("מזל טוב!"))
        p = make_pipeline()
        p._seed_letter_from_order(self.FOLDER)
        assert workspace.letter_file.read_text(encoding="utf-8").strip() == "מזל טוב!"
        # The doc id is the order id from the folder leaf, not the whole leaf.
        assert seen["order_id"] == "AM-180726-XY12"

    def test_existing_letter_is_never_overwritten(self, make_pipeline, workspace,
                                                  monkeypatch):
        workspace.letter_file.write_text("hand-written\n", encoding="utf-8")
        self._stub_firebase(monkeypatch, order=self._order("from the order"))
        make_pipeline()._seed_letter_from_order(self.FOLDER)
        assert workspace.letter_file.read_text(encoding="utf-8") == "hand-written\n"

    def test_order_without_a_blessing_writes_nothing(self, make_pipeline,
                                                     workspace, monkeypatch):
        self._stub_firebase(monkeypatch, order=self._order("   "))
        make_pipeline()._seed_letter_from_order(self.FOLDER)
        assert not workspace.letter_file.exists()

    def test_no_ledger_configured_is_a_no_op(self, make_pipeline, workspace,
                                             monkeypatch):
        self._stub_firebase(monkeypatch, configured=False)
        make_pipeline()._seed_letter_from_order(self.FOLDER)
        assert not workspace.letter_file.exists()

    def test_ledger_failure_never_sinks_the_ingest(self, make_pipeline, workspace,
                                                   monkeypatch):
        self._stub_firebase(monkeypatch, explode=True)
        make_pipeline()._seed_letter_from_order(self.FOLDER)  # must not raise
        assert not workspace.letter_file.exists()


class TestPendingRenders:
    """Submitted-but-uncollected fal jobs — money spent, output not fetched.

    Nothing polls for these in the background: the result is only fetched by
    the next `render` of that clip, and only while the fingerprint still
    matches. They were invisible everywhere before, so an abandoned paid
    render was silent.
    """

    def _pipeline(self, make_pipeline):
        return make_pipeline(analyze_frames=False)

    def _project(self, workspace, p):
        """A 2-frame project; returns the fingerprint render would compute.

        Derived through _pairs_from_storyboard on purpose: that is what
        applies the global motion prompt, so the fingerprint matches the
        prompt generate_clip actually submits.
        """
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        start, end, motion, duration, _ = p._pairs_from_storyboard(sb)[0]
        return p.video_client.fingerprint(start, end, motion, duration)

    def test_no_entries_means_nothing_pending(self, make_pipeline):
        assert self._pipeline(make_pipeline).pending_renders() == []

    def test_matching_fingerprint_is_recoverable(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        fp = self._project(workspace, p)
        p.state.set("falreq:a_to_b.mp4", "pending", request_id="req-1",
                    fingerprint=fp)
        pending = p.pending_renders()
        assert len(pending) == 1
        assert pending[0]["id"] == "a_to_b"
        assert pending[0]["request_id"] == "req-1"
        assert pending[0]["recoverable"] is True

    def test_edited_prompt_makes_the_paid_job_unrecoverable(
        self, make_pipeline, workspace
    ):
        # The money is already spent, but the job renders the OLD plan, so
        # the next render discards it and buys a fresh clip. That has to be
        # visible rather than silent.
        p = self._pipeline(make_pipeline)
        self._project(workspace, p)
        p.state.set("falreq:a_to_b.mp4", "pending", request_id="req-1",
                    fingerprint="a-fingerprint-from-the-old-plan")
        assert p.pending_renders()[0]["recoverable"] is False

    def test_entry_without_a_fingerprint_is_not_recoverable(
        self, make_pipeline, workspace
    ):
        p = self._pipeline(make_pipeline)
        self._project(workspace, p)
        p.state.set("falreq:a_to_b.mp4", "pending", request_id="req-1")
        assert p.pending_renders()[0]["recoverable"] is False

    def test_snapshot_surfaces_pending_renders(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        fp = self._project(workspace, p)
        p.state.set("falreq:a_to_b.mp4", "pending", request_id="req-1",
                    fingerprint=fp)
        assert p.snapshot()["pending_renders"][0]["id"] == "a_to_b"

    def test_collected_render_leaves_nothing_pending(
        self, make_pipeline, workspace
    ):
        # generate_clip clears the entry once the mp4 is downloaded.
        p = self._pipeline(make_pipeline)
        fp = self._project(workspace, p)
        p.state.set("falreq:a_to_b.mp4", "pending", request_id="req-1",
                    fingerprint=fp)
        p.state.clear("falreq:a_to_b.mp4")
        assert p.pending_renders() == []


class TestLocalTime:
    """State timestamps are UTC; showing them raw read as hours in the past."""

    def test_utc_timestamp_is_converted_to_local(self):
        from ai_video_maker.runner import _local_time
        import datetime as dt
        iso = "2026-07-26T12:28:05.514357+00:00"
        expected = (
            dt.datetime.fromisoformat(iso).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        )
        assert _local_time(iso) == expected

    def test_empty_is_unknown(self):
        from ai_video_maker.runner import _local_time
        assert _local_time("") == "unknown"

    def test_unparseable_falls_back_to_the_raw_prefix(self):
        from ai_video_maker.runner import _local_time
        assert _local_time("not a timestamp at all") == "not a timestamp at "


class TestRecordedFailures:
    """snapshot() carries the actual failures, not just "a file exists"."""

    def test_no_file_means_no_failures(self, pipeline):
        assert pipeline.snapshot()["failed_jobs"] == []

    def test_failures_are_surfaced_with_kind_and_error(self, pipeline, workspace):
        pipeline.failed.record("clip:a_to_b", "clip", "fal request still Queued")
        pipeline.failed.flush()
        failures = pipeline.snapshot()["failed_jobs"]
        assert len(failures) == 1
        assert failures[0]["id"] == "clip:a_to_b"
        assert failures[0]["kind"] == "clip"
        assert "still Queued" in failures[0]["error"]

    def test_a_clean_run_clears_them(self, pipeline):
        pipeline.failed.record("clip:a_to_b", "clip", "boom")
        pipeline.failed.flush()
        assert pipeline.snapshot()["failed_jobs"]
        pipeline.failed.failures.clear()
        pipeline.failed.flush()  # removes the stale report
        assert pipeline.snapshot()["failed_jobs"] == []

    def test_malformed_report_does_not_break_status(self, pipeline, workspace):
        path = pipeline.failed.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        assert pipeline.snapshot()["failed_jobs"] == []


class TestFrameTagsSurviveStoryboardRuns:
    """Tagging is hand work; a storyboard re-run must never throw it away."""

    def _tagged_saved(self, workspace, names, tagged_frame="a"):
        sb = _save_slug_storyboard(workspace, names)
        sb.characters = [Character(id="dad", epithet="the bald man")]
        for frame in sb.frames:
            if frame.output_path.endswith(f"{tagged_frame}.png"):
                frame.people = [FramePerson(id="dad", x=0.3, y=0.4)]
        sb.save(workspace.default_storyboard_json)
        return sb

    def test_tags_are_carried_over_by_frame(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False)
        saved = self._tagged_saved(workspace, ["a", "b"])
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        rebuilt, _, _ = p._reconcile_storyboard(saved, pairs)
        by_path = {f.output_path: f for f in rebuilt.frames}
        kept = by_path["styled_images/a.png"].people
        assert [(person.id, person.x) for person in kept] == [("dad", 0.3)]
        assert by_path["styled_images/b.png"].people == []

    def test_tags_follow_the_frame_when_others_are_inserted(
        self, make_pipeline, workspace
    ):
        # Inserting a photo re-plans the pairs around it; the tags belong to
        # the frame, not to its position, so they must not shift.
        p = make_pipeline(analyze_frames=False)
        saved = self._tagged_saved(workspace, ["a", "b"], tagged_frame="b")
        pairs = _make_frame_pairs(workspace, ["a", "a2", "b"])
        rebuilt, _, _ = p._reconcile_storyboard(saved, pairs)
        by_path = {f.output_path: f for f in rebuilt.frames}
        assert [x.id for x in by_path["styled_images/b.png"].people] == ["dad"]
        assert by_path["styled_images/a2.png"].people == []

    def test_the_planner_is_handed_the_tags_as_epithets(
        self, make_pipeline, workspace
    ):
        # A targeted re-plan is exactly the case that used to forget what the
        # rest of the movie knew, so it is the one worth pinning.
        p = make_pipeline(analyze_frames=True, replan_clips=["a_to_b"])
        seen: dict = {}

        def fake_analyze(frames, style, default_duration=None,
                         global_context="", cast=None, lessons=None,
                         frame_people=None):
            seen["frame_people"] = frame_people
            return [("m", 5, "s")] * (len(frames) - 1), list(cast or [])

        p.openai.analyze_frame_transitions = fake_analyze
        saved = self._tagged_saved(workspace, ["a", "b"])
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        p._reconcile_storyboard(saved, pairs)
        # Resolved through the cast, and None (not []) where nobody tagged.
        assert seen["frame_people"] == [["the bald man"], None]


class TestPlanThenTagThenReplan:
    """The three-part flow: plan (+ first tagging) → correct → re-plan.

    The cast only exists after the first plan, and tags point at cast ids, so
    the first tagging pass can only happen at the END of planning. What the
    human corrects in between then reaches the prompts through a re-plan —
    never on its own, because nothing already planned is touched until its
    pair is planned again.
    """

    def _pipeline(self, make_pipeline, proposals=None, **options):
        p = make_pipeline(analyze_frames=False, **options)
        calls: list[list] = []

        def fake_identify(frames, cast):
            calls.append([f.name for f in frames])
            return proposals if proposals is not None else [
                [{"id": "dad", "x": 0.4, "y": 0.5}] for _ in frames
            ]

        p.openai.identify_people = fake_identify
        return p, calls

    def _saved_with_cast(self, workspace, names):
        sb = _save_slug_storyboard(workspace, names)
        sb.characters = [Character(id="dad", epithet="the bald man")]
        sb.save(workspace.default_storyboard_json)
        return Storyboard.load(workspace.default_storyboard_json)

    def test_planning_ends_by_proposing_who_is_in_each_photo(
        self, make_pipeline, workspace
    ):
        p, calls = self._pipeline(make_pipeline)
        _make_frame_pairs(workspace, ["a", "b"])
        self._saved_with_cast(workspace, ["a", "b"])
        p._prepare_mode_a_storyboard()
        assert calls == [["a.png", "b.png"]]
        saved = Storyboard.load(workspace.default_storyboard_json)
        assert [f.people[0].id for f in saved.frames] == ["dad", "dad"]

    def test_frames_already_tagged_are_never_re_proposed(
        self, make_pipeline, workspace
    ):
        # The corrections are the whole point; a later plan must not undo them.
        p, calls = self._pipeline(make_pipeline)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = self._saved_with_cast(workspace, ["a", "b"])
        sb.frames[0].people = [FramePerson(id="dad", x=0.9)]
        sb.save(workspace.default_storyboard_json)
        p._prepare_mode_a_storyboard()
        assert calls == [["b.png"]]  # only the untagged frame
        saved = Storyboard.load(workspace.default_storyboard_json)
        assert saved.frames[0].people[0].x == 0.9  # the human's placement

    def test_the_tag_pass_can_be_skipped(self, make_pipeline, workspace):
        p, calls = self._pipeline(make_pipeline, tag_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        self._saved_with_cast(workspace, ["a", "b"])
        p._prepare_mode_a_storyboard()
        assert calls == []

    def test_a_project_with_no_cast_is_not_tagged(self, make_pipeline, workspace):
        # Nothing to point a tag at until the planner has named people.
        p, calls = self._pipeline(make_pipeline)
        _make_frame_pairs(workspace, ["a", "b"])
        _save_slug_storyboard(workspace, ["a", "b"])  # no characters
        p._prepare_mode_a_storyboard()
        assert calls == []

    def test_a_failed_proposal_never_loses_the_plan(self, make_pipeline, workspace):
        p, _ = self._pipeline(make_pipeline)

        def boom(frames, cast):
            raise RuntimeError("insufficient_quota")

        p.openai.identify_people = boom
        _make_frame_pairs(workspace, ["a", "b"])
        self._saved_with_cast(workspace, ["a", "b"])
        assert p._prepare_mode_a_storyboard() is not None
        assert Storyboard.load(workspace.default_storyboard_json).transitions

    def test_an_empty_answer_leaves_the_frame_untagged(
        self, make_pipeline, workspace
    ):
        # "I couldn't see anyone" is a question for the human, not a record
        # that the photo has nobody in it.
        p, _ = self._pipeline(make_pipeline, proposals=[[], []])
        _make_frame_pairs(workspace, ["a", "b"])
        self._saved_with_cast(workspace, ["a", "b"])
        p._prepare_mode_a_storyboard()
        saved = Storyboard.load(workspace.default_storyboard_json)
        assert all(f.people == [] for f in saved.frames)

    def test_a_dry_run_proposes_nothing(self, make_pipeline, workspace):
        p, calls = self._pipeline(make_pipeline, dry_run=True)
        _make_frame_pairs(workspace, ["a", "b"])
        self._saved_with_cast(workspace, ["a", "b"])
        p._prepare_mode_a_storyboard()
        assert calls == []

    def test_replan_all_replans_every_pair(self, make_pipeline, workspace):
        # The batch that applies corrected tags/epithets: without it, pairs
        # whose frames never changed are carried over verbatim forever.
        p = make_pipeline(analyze_frames=False, motion_prompt="a fresh plan",
                          replan_all=True, tag_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == ["a_to_b", "b_to_c"]
        assert [t.motion_prompt for t in sb.transitions] == [
            "a fresh plan", "a fresh plan",
        ]
        assert stale == ["a_to_b", "b_to_c"]

    def test_without_replan_all_untouched_pairs_are_kept(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False, motion_prompt="a fresh plan",
                          tag_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, _ = p._reconcile_storyboard(saved, pairs)
        assert replanned == []
        assert [t.motion_prompt for t in sb.transitions] == ["motion a", "motion b"]


class TestPlansRecordWhatTheyWerePlannedFrom:
    """The stamp that makes "this prompt predates your tags" visible."""

    def test_a_replanned_pair_records_the_identity_it_used(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False, motion_prompt="fresh",
                          replan_all=True, tag_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        saved.characters = [Character(id="dad", epithet="the bald man")]
        saved.frames[0].people = [FramePerson(id="dad", x=0.3)]
        sb, _, _ = p._reconcile_storyboard(saved, pairs)
        assert sb.transitions[0].planned_identity
        # Planned from exactly what the storyboard now says, so nothing is
        # reported as behind.
        assert outdated_identity_plans(sb) == []

    def test_tagging_after_planning_shows_the_prompt_as_behind(
        self, make_pipeline, workspace
    ):
        # The whole point: the user tags a photo and can SEE which prompts
        # don't know about it yet, instead of having to remember.
        p = make_pipeline(analyze_frames=False, motion_prompt="fresh",
                          replan_all=True, tag_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        saved.characters = [Character(id="dad", epithet="the bald man")]
        sb, _, _ = p._reconcile_storyboard(saved, pairs)
        assert outdated_identity_plans(sb) == []
        sb.frames[0].people = [FramePerson(id="dad", x=0.3)]
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_a_carried_over_pair_keeps_its_old_stamp(
        self, make_pipeline, workspace
    ):
        # It was not re-planned, so its record of what it knew must not be
        # quietly refreshed — that would hide the very thing this reports.
        p = make_pipeline(analyze_frames=False, tag_frames=False)
        pairs = _make_frame_pairs(workspace, ["a", "b"])
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        saved.transitions[0].planned_identity = "deadbeefcafe"
        sb, replanned, _ = p._reconcile_storyboard(saved, pairs)
        assert replanned == []
        assert sb.transitions[0].planned_identity == "deadbeefcafe"

    def test_the_snapshot_reports_them(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.characters = [Character(id="dad", epithet="the bald man")]
        sb.frames[0].people = [FramePerson(id="dad", x=0.3)]
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["outdated_plans"] == ["a_to_b"]


class TestFinalVideoFreshness:
    def test_a_clip_rendered_after_the_movie_flags_it(self, pipeline, workspace):
        import os, time
        _write_frames(workspace, [1, 2])
        clip = _touch(workspace.clips_dir / "001_to_002.mp4")
        final = _touch(workspace.final_video)
        os.utime(final, (time.time() - 60, time.time() - 60))  # built earlier
        assert pipeline._final_video_outdated() is True

    def test_a_movie_newer_than_its_clips_is_fine(self, pipeline, workspace):
        import os, time
        _touch(workspace.clips_dir / "001_to_002.mp4")
        final = _touch(workspace.final_video)
        os.utime(final, (time.time() + 60, time.time() + 60))
        assert pipeline._final_video_outdated() is False

    def test_no_movie_is_not_an_outdated_movie(self, pipeline, workspace):
        _touch(workspace.clips_dir / "001_to_002.mp4")
        assert pipeline._final_video_outdated() is False


class TestOffscreenEndingsAreVisible:
    def test_a_prompt_that_ends_outside_the_frame_is_listed(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.transitions[0].motion_prompt = (
            "The smaller boy with wavy hair near the right walks forward past "
            "the camera and exits left, ending offscreen on the left."
        )
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["ends_offscreen"] == ["a_to_b"]

    def test_a_prompt_that_brings_them_back_is_not_listed(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.transitions[0].motion_prompt = (
            "the bald man walks past the camera, then steps back in from the "
            "right and stands beside the woman with curly hair"
        )
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["ends_offscreen"] == []


class TestIndistinctEpithetsAreVisible:
    """A cast pair the matcher can't tell apart is surfaced, never rewritten.

    Existing casts are frozen (their wording is baked into planned prompts),
    so like fragile epithets these are flagged for a hand edit in the
    panel's Cast editor — a real 29-entry cast carried "woman with dark hair
    bun" AND "young woman with dark hair bun".
    """

    def test_colliding_cast_names_are_listed_as_a_group(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.characters = [
            Character(id="mum", epithet="woman with dark hair bun"),
            Character(id="aunt", epithet="young woman with dark hair bun"),
            Character(id="dad", epithet="the bald man"),
        ]
        sb.save(workspace.default_storyboard_json)
        snap = p.snapshot()["storyboard"]
        assert snap["indistinct_epithets"] == [["mum", "aunt"]]

    def test_a_distinct_cast_reports_nothing(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.characters = [
            Character(id="dad", epithet="the bald man"),
            Character(id="son", epithet="the smaller boy with curly hair"),
        ]
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["indistinct_epithets"] == []


class TestUnstageablePairsAreVisible:
    """A saved prompt staging people on an unstageable pair is surfaced.

    The gate only acts when a pair is (re-)planned, so an existing
    storyboard keeps its many-mover choreography until someone re-plans it —
    this is what tells them where. Untagged frames have no opinion, keeping
    every pre-existing untagged project silent.
    """

    SIX = ["mum", "aunt", "dad", "son", "daughter", "grandpa"]

    def _tagged_storyboard(self, workspace, end_ids):
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        sb.characters = [
            Character(id=i, epithet=f"the {i} person") for i in self.SIX
        ]
        sb.frames[0].people = [
            FramePerson(id=i, x=n / 10) for n, i in enumerate(self.SIX)
        ]
        sb.frames[1].people = [
            FramePerson(id=i, x=n / 10) for n, i in enumerate(end_ids)
        ]
        return sb

    def test_a_staged_prompt_on_an_unstageable_pair_is_listed(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = self._tagged_storyboard(workspace, ["mum"])
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["unstageable_pairs"] == ["a_to_b"]

    def test_a_camera_prompt_is_already_the_answer(
        self, make_pipeline, workspace
    ):
        from ai_video_maker.clients.openai_client import (
            CAMERA_SHIFT_MOTION_PROMPT,
        )
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = self._tagged_storyboard(workspace, ["mum"])
        sb.transitions[0].motion_prompt = CAMERA_SHIFT_MOTION_PROMPT
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["unstageable_pairs"] == []

    def test_untagged_frames_have_no_opinion(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = self._tagged_storyboard(workspace, ["mum"])
        sb.frames[1].people = []  # one side untagged -> silent
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["unstageable_pairs"] == []

    def test_a_small_stageable_pair_is_not_listed(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False)
        _make_frame_pairs(workspace, ["a", "b"])
        sb = self._tagged_storyboard(workspace, self.SIX)
        sb.save(workspace.default_storyboard_json)
        assert p.snapshot()["storyboard"]["unstageable_pairs"] == []
