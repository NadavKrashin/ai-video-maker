"""Marking rendered clips that need redoing — the human's own worklist.

Deliberately not the feedback store (see clip_flags.py): feedback is
append-only evidence that costs model calls and teaches the planner, this is
a list you tick off. What these pin down is the difference — free, mutable,
and inert with respect to the movie itself.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from ai_video_maker.clip_flags import ClipFlagStore
from ai_video_maker.errors import PipelineError


def _clip(workspace, tid: str, data: bytes = b"clip") -> Path:
    workspace.clips_dir.mkdir(parents=True, exist_ok=True)
    path = workspace.clips_dir / f"{tid}.mp4"
    path.write_bytes(data)
    return path


class TestTheStore:
    def test_a_mark_survives_a_reload(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", "teleports", 12.5)
        assert ClipFlagStore(tmp_path / "flags.json").all().keys() == {"003_to_004"}

    def test_marking_twice_updates_rather_than_duplicates(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", "first")
        store.mark("003_to_004", "second")
        flags = store.all()
        assert list(flags) == ["003_to_004"]
        assert flags["003_to_004"].note == "second"

    def test_clearing_is_the_whole_point(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004")
        assert store.clear("003_to_004") is True
        assert store.all() == {}

    def test_clearing_something_unmarked_is_not_an_error(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        assert store.clear("nope") is False

    def test_a_corrupt_file_reads_as_empty_rather_than_raising(self, tmp_path):
        """A worklist must never be the thing that breaks `status`."""
        path = tmp_path / "flags.json"
        path.write_text("{not json", encoding="utf-8")
        assert ClipFlagStore(path).all() == {}

    def test_one_bad_row_does_not_lose_the_others(self, tmp_path):
        path = tmp_path / "flags.json"
        path.write_text(json.dumps({"flags": [
            {"nonsense": True},
            {"transition_id": "004_to_005", "note": "keep me"},
        ]}), encoding="utf-8")
        flags = ClipFlagStore(path).all()
        assert list(flags) == ["004_to_005"]

    def test_a_long_note_is_trimmed_not_refused(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        flag = store.mark("003_to_004", "x" * 5000)
        assert len(flag.note) == 500


class TestReRenderIsReportedNotCleared:
    """A mark is about the clip you watched — but the code never decides
    on your behalf that you are finished with it."""

    def test_an_untouched_clip_is_not_reported_as_rerendered(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", clip_mtime=100.0)
        summary = store.summary({"003_to_004": 100.0})
        assert summary["rerendered"] == []

    def test_a_newer_clip_is_reported(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", clip_mtime=100.0)
        summary = store.summary({"003_to_004": 200.0})
        assert summary["rerendered"] == ["003_to_004"]

    def test_the_mark_itself_survives_a_rerender(self, tmp_path):
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", clip_mtime=100.0)
        store.summary({"003_to_004": 200.0})
        assert "003_to_004" in store.all(), "a re-render must not clear a mark"

    def test_a_mark_from_before_mtimes_were_recorded_is_not_reported(self, tmp_path):
        """clip_mtime 0 means 'unknown', which is not evidence of a re-render."""
        store = ClipFlagStore(tmp_path / "flags.json")
        store.mark("003_to_004", clip_mtime=0.0)
        assert store.summary({"003_to_004": 200.0})["rerendered"] == []


class TestMarkingThroughThePipeline:
    def _pipeline(self, make_pipeline, **options):
        return make_pipeline(**options)

    def test_marking_a_rendered_clip(self, make_pipeline, workspace):
        _clip(workspace, "003_to_004")
        p = make_pipeline(feedback_clip="003_to_004", feedback_note="teleports")
        result = p.mark_clip()
        assert result["marked"] is True
        assert p.clip_flags.all()["003_to_004"].note == "teleports"

    def test_an_unrendered_clip_is_refused(self, make_pipeline, workspace):
        """The mark means "I watched this"; there has to be something to watch."""
        p = make_pipeline(feedback_clip="003_to_004")
        with pytest.raises(PipelineError, match="No rendered clip"):
            p.mark_clip()

    def test_marking_needs_a_clip_id(self, make_pipeline, workspace):
        p = make_pipeline(feedback_note="something")
        with pytest.raises(PipelineError, match="Which clip"):
            p.mark_clip()

    def test_clearing_does_not_need_the_clip_to_exist(self, make_pipeline, workspace):
        """You must be able to tick off a mark whose clip you already deleted."""
        _clip(workspace, "003_to_004")
        p = make_pipeline(feedback_clip="003_to_004")
        p.mark_clip()
        (workspace.clips_dir / "003_to_004.mp4").unlink()
        p2 = make_pipeline(feedback_clip="003_to_004", clear_flag=True)
        assert p2.mark_clip()["was_marked"] is True
        assert p2.clip_flags.all() == {}

    def test_marking_never_touches_the_clip(self, make_pipeline, workspace):
        """The cardinal rule: nothing here deletes or re-renders anything."""
        path = _clip(workspace, "003_to_004", b"the rendered bytes")
        before = (path.read_bytes(), path.stat().st_mtime)
        p = make_pipeline(feedback_clip="003_to_004", feedback_note="bad")
        p.mark_clip()
        assert path.exists()
        assert (path.read_bytes(), path.stat().st_mtime) == before

    def test_marking_teaches_the_planner_nothing(self, make_pipeline, workspace):
        """A worklist is not evidence: no feedback entry, no lesson."""
        _clip(workspace, "003_to_004")
        p = make_pipeline(feedback_clip="003_to_004", feedback_note="bad")
        p.mark_clip()
        assert p.feedback_store.all() == []
        assert not workspace.feedback_file.exists()


class TestMarksAreVisible:
    """The whole value is knowing which clips need fixing, so `status` and
    the snapshot have to say — and each clip has to carry its own mark."""

    def _project(self, make_pipeline, workspace, monkeypatch):
        from ai_video_maker.models import Frame, Storyboard, Transition
        names = ["003", "004", "005"]
        workspace.styled_images_dir.mkdir(parents=True, exist_ok=True)
        for n in names:
            (workspace.styled_images_dir / f"{n}.png").write_bytes(b"x")
        sb = Storyboard(
            project_title="t", style="s",
            frames=[
                Frame(id=n, description="", image_prompt="",
                      output_path=f"styled_images/{n}.png",
                      source_path=f"input_images/{n}.jpg")
                for n in names
            ],
            transitions=[
                Transition(
                    id=f"{a}_to_{b}", start_frame=f"styled_images/{a}.png",
                    end_frame=f"styled_images/{b}.png", motion_prompt="m",
                    duration=5, sound_prompt="",
                    output_path=f"clips/{a}_to_{b}.mp4")
                for a, b in zip(names, names[1:])
            ],
        )
        sb.save(workspace.default_storyboard_json)
        return make_pipeline()

    def test_the_snapshot_names_the_marked_clips(
        self, make_pipeline, workspace, monkeypatch
    ):
        _clip(workspace, "003_to_004")
        p = self._project(make_pipeline, workspace, monkeypatch)
        p.clip_flags.mark("003_to_004", "teleports", 1.0)
        snap = p.snapshot()
        assert snap["clip_flags"]["ids"] == ["003_to_004"]
        assert snap["clip_flags"]["count"] == 1

    def test_each_clip_row_carries_its_own_mark(
        self, make_pipeline, workspace, monkeypatch
    ):
        _clip(workspace, "003_to_004")
        _clip(workspace, "004_to_005")
        p = self._project(make_pipeline, workspace, monkeypatch)
        p.clip_flags.mark("003_to_004", "teleports", 1.0)
        by_id = {c["id"]: c for c in p.snapshot()["clips"]}
        assert by_id["003_to_004"]["needs_fix"] is True
        assert by_id["003_to_004"]["fix_note"] == "teleports"
        assert by_id["004_to_005"]["needs_fix"] is False

    def test_an_unmarked_project_says_nothing(
        self, make_pipeline, workspace, monkeypatch
    ):
        """A project nobody has marked must stay completely silent."""
        _clip(workspace, "003_to_004")
        p = self._project(make_pipeline, workspace, monkeypatch)
        flags = p.snapshot()["clip_flags"]
        assert flags == {"count": 0, "ids": [], "flags": [], "rerendered": []}

    def test_the_snapshot_flags_a_mark_whose_clip_was_rerendered(
        self, make_pipeline, workspace, monkeypatch
    ):
        path = _clip(workspace, "003_to_004")
        p = self._project(make_pipeline, workspace, monkeypatch)
        p.clip_flags.mark("003_to_004", "", path.stat().st_mtime)
        os.utime(path, (time.time() + 60, time.time() + 60))
        assert p.snapshot()["clip_flags"]["rerendered"] == ["003_to_004"]
