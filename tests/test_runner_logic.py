"""Pipeline pure-logic tests: bridging, pair building, selection, combine list."""
from __future__ import annotations

import os
import time
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
    sb = Storyboard(project_title="t", style="style", music_prompt="the tune",
                    frames=frames, transitions=transitions)
    sb.save(workspace.default_storyboard_json)
    return sb


class TestReconcileStoryboard:
    """The surgical-edit merge: keep untouched transitions, re-plan the rest."""

    def _pipeline(self, make_pipeline):
        # analysis off -> deterministic fallback plans (config.motion_prompt)
        return make_pipeline(analyze_frames=False)

    def test_unchanged_project_keeps_everything(self, make_pipeline, workspace):
        p = self._pipeline(make_pipeline)
        pairs = _make_frame_pairs(workspace, ["a", "b", "c"])
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, replanned, stale = p._reconcile_storyboard(saved, pairs)
        assert replanned == [] and stale == []
        assert [t.motion_prompt for t in sb.transitions] == ["motion a", "motion b"]
        assert sb.music_prompt == "the tune"  # user-level fields carried over

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

    def test_invalidate_stale_clips_deletes_and_clears_state(
        self, make_pipeline, workspace
    ):
        p = self._pipeline(make_pipeline)
        clip = _touch(workspace.clips_dir / "a_to_b.mp4")
        p.state.set("sfx:a_to_b.mp4", "done")
        p.state.set("fade:a_to_b.mp4", "done")
        p._invalidate_stale_clips(["a_to_b", "b_to_c"])  # b_to_c has no file
        assert not clip.exists()
        assert not p.state.is_done("sfx:a_to_b.mp4")
        assert not p.state.is_done("fade:a_to_b.mp4")

    def test_invalidate_stale_clips_declined_keeps_clips_and_state(
        self, config, workspace
    ):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline

        asked = []

        def deny(lines, question):
            asked.append(question)
            return False

        p = Pipeline(config, workspace, RunOptions(), confirm=deny)
        clip = _touch(workspace.clips_dir / "a_to_b.mp4")
        p.state.set("sfx:a_to_b.mp4", "done")
        p._invalidate_stale_clips(["a_to_b"])
        assert clip.exists()
        assert p.state.is_done("sfx:a_to_b.mp4")
        assert len(asked) == 1  # rendered clips are never deleted silently

    def test_invalidate_stale_clips_no_files_never_asks(
        self, config, workspace
    ):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline

        def fail_ask(lines, question):  # pragma: no cover - must not be hit
            raise AssertionError("should not ask when nothing exists")

        p = Pipeline(config, workspace, RunOptions(), confirm=fail_ask)
        p._invalidate_stale_clips(["a_to_b", "b_to_c"])  # no clip files


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
        monkeypatch.setattr(runner_mod, "render_opening_reveal", stub("reveal", 2))
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

    def test_reveal_replaces_first_clip(self, make_pipeline, workspace, monkeypatch):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(opening_reveal=True)
        clips = self._project(workspace)
        segments, added = p._presentation_segments(clips)
        assert added is True
        assert segments[0].name == "opening_reveal.mp4"
        assert segments[1:] == clips[1:]

    def test_no_recorded_sources_skips_cleanly(
        self, make_pipeline, workspace, monkeypatch
    ):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(credits_photos=True, opening_reveal=True)
        # storyboard whose frames carry no source_path (legacy / Mode B)
        _write_frames(workspace, range(1, 4))
        _storyboard(workspace, 3).save(workspace.default_storyboard_json)
        clips = [_touch(workspace.clips_dir / "001_to_002.mp4")]
        segments, added = p._presentation_segments(clips)
        assert segments == clips and added is False

    def test_cli_override_beats_config(self, make_pipeline):
        p = make_pipeline(
            credits_photos=False, opening_reveal=False, closing_letter=False,
            intro_clip=False,
        )
        p.config.credits_photos = True
        p.config.opening_reveal = True
        p.config.closing_letter = True
        p.config.intro_clip = True
        assert p._presentation_flags() == (False, False, False, False)

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

    def test_intro_comes_before_the_opening_reveal(
        self, make_pipeline, workspace, monkeypatch
    ):
        self._stub_renderers(monkeypatch)
        p = make_pipeline(intro_clip=True, opening_reveal=True)
        clips = self._project(workspace)
        self._global_intro(p, workspace)
        segments, added = p._presentation_segments(clips)
        assert added is True
        assert [s.name for s in segments[:2]] == [
            "intro.mp4", "opening_reveal.mp4",
        ]
        assert segments[2:] == clips[1:]

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
            opening_reveal=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
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
            opening_reveal=True, credits_photos=True, closing_letter=True
        )
        clips = self._project(workspace)
        workspace.letter_file.write_text("שלום", encoding="utf-8")
        p._presentation_segments(clips)
        calls.clear()
        future = time.time() + 50
        os.utime(workspace.letter_file, (future, future))  # letter edited
        p._presentation_segments(clips)
        assert "reveal" not in calls              # untouched -> reused
        assert "overlay" in calls                 # letter section redone


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
