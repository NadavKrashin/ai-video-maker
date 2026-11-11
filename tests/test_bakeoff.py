"""`bakeoff`: one pair, several video models, nothing of the movie touched.

The renders are faked — what is pinned here is what the command is ALLOWED
to do: which request each model gets, where the files land, and above all
what it leaves alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ai_video_maker.runner as runner_module
from ai_video_maker.errors import PipelineError
from ai_video_maker.models import Frame, Storyboard, Transition


class _FakeVideoClient:
    """Stands in for VideoClient; records what each model was asked for."""

    calls: list[dict] = []
    fail_for: set[str] = set()

    def __init__(self, config, state=None, should_cancel=None):
        self.config = config

    def generate_clip(self, start, end, motion, duration, dst, reword=None,
                      elements=None):
        key = self.config.video_model
        type(self).calls.append({
            "model": key, "model_id": self.config.fal_model_id,
            "motion": motion, "duration": duration, "dst": dst,
        })
        if key in type(self).fail_for:
            raise RuntimeError("422 Unprocessable: end_image_url not allowed")
        dst.write_bytes(f"video from {key}".encode())


@pytest.fixture
def project(workspace, monkeypatch):
    _FakeVideoClient.calls = []
    _FakeVideoClient.fail_for = set()
    monkeypatch.setattr(runner_module, "VideoClient", _FakeVideoClient)
    for n in ("a", "b", "c"):
        (workspace.styled_images_dir / f"{n}.png").write_bytes(n.encode())
    Storyboard(
        project_title="t", style="style",
        frames=[
            Frame(id=n, description="", image_prompt="",
                  output_path=f"styled_images/{n}.png")
            for n in ("a", "b", "c")
        ],
        transitions=[
            Transition(
                id=f"{x}_to_{y}", start_frame=f"styled_images/{x}.png",
                end_frame=f"styled_images/{y}.png",
                motion_prompt=f"the bald man walks from {x} to {y}",
                duration=5, output_path=f"clips/{x}_to_{y}.mp4",
            )
            for x, y in (("a", "b"), ("b", "c"))
        ],
    ).save(workspace.default_storyboard_json)
    (workspace.clips_dir / "a_to_b.mp4").write_bytes(b"the real clip")
    return workspace


def _run(make_pipeline, **options):
    pipeline = make_pipeline(**options)
    pipeline.cmd_bakeoff()
    return pipeline


class TestBakeoff:
    def test_a_clip_must_be_named(self, make_pipeline, project):
        with pytest.raises(PipelineError, match="--clip"):
            _run(make_pipeline)
        assert _FakeVideoClient.calls == []

    def test_each_model_renders_the_same_pair_with_its_own_request(
        self, make_pipeline, project
    ):
        _run(make_pipeline, clips=["a_to_b"],
             bakeoff_models=["kling-2.5-turbo-pro", "kling-3-pro"])
        calls = {c["model"]: c for c in _FakeVideoClient.calls}
        assert set(calls) == {"kling-2.5-turbo-pro", "kling-3-pro"}
        assert calls["kling-3-pro"]["model_id"].endswith("v3/pro/image-to-video")
        # Like for like: same words, same length, on every model.
        assert calls["kling-3-pro"]["motion"] == calls["kling-2.5-turbo-pro"]["motion"]
        assert {c["duration"] for c in calls.values()} == {5}

    def test_renders_land_in_bakeoff_and_the_movie_is_untouched(
        self, make_pipeline, project
    ):
        pipeline = _run(make_pipeline, clips=["a_to_b"],
                        bakeoff_models=["kling-2.5-turbo-pro", "kling-3-pro"])
        names = sorted(p.name for p in project.bakeoff_dir.glob("*.mp4"))
        assert names == ["a_to_b__kling-2.5-turbo-pro.mp4", "a_to_b__kling-3-pro.mp4"]
        # The real clip, the clips folder and the clip's state are as they were.
        assert (project.clips_dir / "a_to_b.mp4").read_bytes() == b"the real clip"
        assert [p.name for p in project.clips_dir.iterdir()] == ["a_to_b.mp4"]
        assert pipeline.state.get("clip:a_to_b.mp4") is None
        assert pipeline.state.get("stale:a_to_b.mp4") is None
        # ...and the config the pipeline carries is the one it started with.
        assert pipeline.config.video_model == ""

    def test_each_render_records_its_price(self, make_pipeline, project):
        pipeline = _run(make_pipeline, clips=["a_to_b"],
                        bakeoff_models=["kling-2.5-turbo-pro", "kling-3-pro"])
        results = {r["model"]: r for r in pipeline.bakeoff_results()}
        assert results["kling-2.5-turbo-pro"]["usd"] == pytest.approx(0.35)
        assert results["kling-3-pro"]["usd"] == pytest.approx(0.56)
        assert results["kling-3-pro"]["file"] == "a_to_b__kling-3-pro.mp4"
        assert results["kling-3-pro"]["clip"] == "a_to_b"

    def test_one_model_failing_is_a_result_not_a_crash(
        self, make_pipeline, project
    ):
        _FakeVideoClient.fail_for = {"kling-3-pro"}
        pipeline = _run(make_pipeline, clips=["a_to_b"],
                        bakeoff_models=["kling-2.5-turbo-pro", "kling-3-pro"])
        results = {r["model"]: r for r in pipeline.bakeoff_results()}
        assert results["kling-2.5-turbo-pro"]["file"]
        assert results["kling-3-pro"]["file"] == ""
        assert "end_image_url" in results["kling-3-pro"]["error"]
        assert "usd" not in results["kling-3-pro"]  # nothing delivered

    def test_an_existing_render_is_not_bought_twice(self, make_pipeline, project):
        _run(make_pipeline, clips=["a_to_b"], bakeoff_models=["kling-3-pro"])
        _run(make_pipeline, clips=["a_to_b"], bakeoff_models=["kling-3-pro"])
        assert len(_FakeVideoClient.calls) == 1

    def test_a_declined_confirmation_renders_nothing(self, config, project):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline
        seen = []
        pipeline = Pipeline(
            config, project,
            RunOptions(clips=["a_to_b"], bakeoff_models=["kling-3-pro", "kling-4"]),
            confirm=lambda lines, question: seen.append(lines) or False,
        )
        pipeline.cmd_bakeoff()
        assert _FakeVideoClient.calls == []
        assert not project.bakeoff_dir.exists()
        plan = "\n".join(seen[0])
        # The plan names every model and what it costs before any money moves.
        assert "a_to_b__kling-3-pro" in plan and "$0.56" in plan
        assert "a_to_b__kling-4" in plan and "$1.18" in plan

    def test_unknown_and_retired_models_are_refused(self, make_pipeline, project):
        with pytest.raises(PipelineError, match="Unknown model"):
            _run(make_pipeline, clips=["a_to_b"], bakeoff_models=["sora"])
        with pytest.raises(PipelineError, match="retired"):
            _run(make_pipeline, clips=["a_to_b"],
                 bakeoff_models=["kling-3-turbo-pro"])
        assert _FakeVideoClient.calls == []

    def test_a_dry_run_spends_and_writes_nothing(self, make_pipeline, project):
        _run(make_pipeline, clips=["a_to_b"], dry_run=True,
             bakeoff_models=["kling-3-pro"])
        assert _FakeVideoClient.calls == []
        assert not project.bakeoff_dir.exists()


class TestSnapshotCarriesTheComparison:
    def test_costs_come_from_the_planned_seconds(self, make_pipeline, project):
        snap = make_pipeline().snapshot()
        costs = snap["video_model_costs"]
        assert costs["clip_seconds"] == 10
        rows = {r["key"]: r for r in costs["models"]}
        assert rows["kling-2.5-turbo-pro"]["movie_usd"] == pytest.approx(0.70)
        assert rows["kling-4"]["movie_usd"] == pytest.approx(2.35)
        assert snap["bakeoff"] == []
