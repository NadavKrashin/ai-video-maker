"""Pipeline pure-logic tests: bridging, pair building, selection, combine list."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_maker.errors import PipelineError
from ai_video_maker.models import Frame, Storyboard, Transition


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


class TestReusableStoryboard:
    def _styled(self, workspace, n=3):
        _write_frames(workspace, range(1, n + 1))
        return [
            workspace.styled_images_dir / f"{i:03d}_styled.png"
            for i in range(1, n + 1)
        ]

    def test_reuses_matching_storyboard(self, pipeline, workspace):
        styled = self._styled(workspace)
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        assert pipeline._load_reusable_storyboard(styled) is not None

    def test_none_when_absent(self, pipeline, workspace):
        assert pipeline._load_reusable_storyboard(self._styled(workspace)) is None

    def test_none_when_frames_changed(self, pipeline, workspace):
        styled = self._styled(workspace, 4)  # storyboard was made for 3
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        assert pipeline._load_reusable_storyboard(styled) is None

    def test_none_under_force(self, make_pipeline, workspace):
        p = make_pipeline(force=True)
        styled = self._styled(workspace)
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        assert p._load_reusable_storyboard(styled) is None

    def test_none_when_unreadable(self, pipeline, workspace):
        styled = self._styled(workspace)
        workspace.default_storyboard_json.parent.mkdir(parents=True, exist_ok=True)
        workspace.default_storyboard_json.write_text("{not json", encoding="utf-8")
        assert pipeline._load_reusable_storyboard(styled) is None


class TestClipName:
    def test_uses_leading_ids(self, pipeline, workspace):
        a = workspace.styled_images_dir / "001_styled.png"
        b = workspace.styled_images_dir / "002_styled.png"
        assert pipeline._clip_name(a, b).name == "001_to_002.mp4"

    def test_non_numeric_stem_falls_back(self, pipeline, workspace):
        a = workspace.styled_images_dir / "alpha.png"
        b = workspace.styled_images_dir / "002.png"
        assert pipeline._clip_name(a, b).name == "alpha_to_002.mp4"
