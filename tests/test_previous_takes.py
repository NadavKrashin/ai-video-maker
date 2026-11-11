"""The take a re-render replaces is kept, and can be swapped back.

Re-rendering overwrote a clip that had been paid for; if the new one was
worse, the only way back was to pay again and hope. What is pinned here is
that nothing is ever lost in either direction: keeping copies, restoring
swaps, and the per-clip state travels with the file it describes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import ai_video_maker.runner as runner_module
from ai_video_maker import previous_takes
from ai_video_maker.errors import PipelineError
from ai_video_maker.models import Frame, Storyboard, Transition
from ai_video_maker.server import JobRunner, create_app
from ai_video_maker.state import StateStore


@pytest.fixture
def clip(tmp_path: Path):
    clips = tmp_path / "clips"
    clips.mkdir()
    path = clips / "a_to_b.mp4"
    path.write_bytes(b"take one")
    return path, tmp_path / "clips_previous", StateStore(tmp_path / "state.json")


class TestKeeping:
    def test_the_current_take_is_copied_not_moved(self, clip):
        path, prev, state = clip
        assert previous_takes.keep_previous_take(path, prev, state)
        assert path.read_bytes() == b"take one"          # a failed render
        assert (prev / "a_to_b.mp4").read_bytes() == b"take one"  # loses nothing

    def test_what_it_was_rendered_from_is_recorded(self, clip):
        path, prev, state = clip
        state.set("clip:a_to_b.mp4", "done", model="m1", prompt="walk", duration=5)
        state.set("sfx:a_to_b.mp4", "done")
        previous_takes.keep_previous_take(path, prev, state)
        info = previous_takes.read_info(prev, "a_to_b.mp4")
        assert (info["prompt"], info["duration"], info["model"]) == ("walk", 5, "m1")
        assert info["state"]["sfx"]["status"] == "done"

    def test_only_the_latest_previous_take_is_kept(self, clip):
        path, prev, state = clip
        previous_takes.keep_previous_take(path, prev, state)
        path.write_bytes(b"take two")
        previous_takes.keep_previous_take(path, prev, state)
        assert [p.name for p in prev.glob("*.mp4")] == ["a_to_b.mp4"]
        assert (prev / "a_to_b.mp4").read_bytes() == b"take two"

    def test_nothing_to_keep_for_a_clip_never_rendered(self, clip, tmp_path):
        _, prev, state = clip
        assert not previous_takes.keep_previous_take(
            tmp_path / "clips" / "x_to_y.mp4", prev, state)
        assert not prev.exists()

    def test_an_unwritable_folder_costs_the_undo_not_the_render(self, clip, tmp_path):
        path, _, state = clip
        blocker = tmp_path / "blocked"
        blocker.write_text("a file where the folder should be")
        assert previous_takes.keep_previous_take(path, blocker, state) is False
        assert path.read_bytes() == b"take one"


class TestRestoring:
    def _rerendered(self, clip, *, old_prompt="walk", new_prompt="walk"):
        path, prev, state = clip
        state.set("clip:a_to_b.mp4", "done", model="m1",
                  prompt=old_prompt, duration=5)
        state.set("sfx:a_to_b.mp4", "done")
        previous_takes.keep_previous_take(path, prev, state)
        # ...the re-render lands:
        path.write_bytes(b"take two")
        state.clear("sfx:a_to_b.mp4")
        state.set("clip:a_to_b.mp4", "done", model="m2",
                  prompt=new_prompt, duration=5)
        return path, prev, state

    def test_the_two_takes_are_swapped(self, clip):
        path, prev, state = self._rerendered(clip)
        previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        assert path.read_bytes() == b"take one"
        assert (prev / "a_to_b.mp4").read_bytes() == b"take two"

    def test_restoring_twice_puts_everything_back(self, clip):
        path, prev, state = self._rerendered(clip)
        for _ in range(2):
            previous_takes.restore_previous_take(
                path, prev, state, planned_prompt="walk", planned_duration=5)
        assert path.read_bytes() == b"take two"
        assert (prev / "a_to_b.mp4").read_bytes() == b"take one"
        assert state.get("clip:a_to_b.mp4")["model"] == "m2"
        assert state.status("sfx:a_to_b.mp4") is None

    def test_the_sound_and_model_state_travel_with_the_file(self, clip):
        path, prev, state = self._rerendered(clip)
        previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        # Take one had SFX baked in and was rendered on m1. If the state said
        # otherwise the audio step would lay a second SFX track over it.
        assert state.is_done("sfx:a_to_b.mp4")
        assert state.get("clip:a_to_b.mp4")["model"] == "m1"

    def test_a_take_from_the_current_prompt_is_not_outdated(self, clip):
        path, prev, state = self._rerendered(clip)
        state.set("stale:a_to_b.mp4", "outdated")
        result = previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        assert result == {"matches_plan": True}
        assert state.status("stale:a_to_b.mp4") is None

    def test_a_take_from_a_different_prompt_is_marked_outdated(self, clip):
        path, prev, state = self._rerendered(clip, old_prompt="run")
        result = previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        assert result == {"matches_plan": False}
        assert state.status("stale:a_to_b.mp4") == "outdated"

    def test_a_take_with_no_record_is_not_guessed_about(self, clip):
        path, prev, state = clip
        previous_takes.keep_previous_take(path, prev, state)  # nothing recorded
        path.write_bytes(b"take two")
        result = previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        assert result == {"matches_plan": None}
        assert state.status("stale:a_to_b.mp4") is None

    def test_the_restored_clip_looks_newer_than_the_movie(self, clip, tmp_path):
        import os
        path, prev, state = self._rerendered(clip)
        old = 1_000_000_000
        os.utime(prev / "a_to_b.mp4", (old, old))
        movie = tmp_path / "final_video.mp4"
        movie.write_bytes(b"movie")
        previous_takes.restore_previous_take(
            path, prev, state, planned_prompt="walk", planned_duration=5)
        assert path.stat().st_mtime >= movie.stat().st_mtime

    def test_restoring_with_nothing_kept_is_an_error_and_changes_nothing(self, clip):
        path, prev, state = clip
        with pytest.raises(FileNotFoundError):
            previous_takes.restore_previous_take(
                path, prev, state, planned_prompt="walk", planned_duration=5)
        assert path.read_bytes() == b"take one"


# ----------------------------------------------------------- in the pipeline

class _FakeVideoClient:
    takes = 0

    def __init__(self, config, state=None, should_cancel=None):
        self.config = config

    def generate_clip(self, start, end, motion, duration, dst, reword=None,
                      elements=None):
        type(self).takes += 1
        dst.write_bytes(f"take {type(self).takes}".encode())


@pytest.fixture
def project(workspace, monkeypatch):
    _FakeVideoClient.takes = 0
    monkeypatch.setattr(runner_module, "VideoClient", _FakeVideoClient)
    for n in ("a", "b"):
        (workspace.styled_images_dir / f"{n}.png").write_bytes(n.encode())
    Storyboard(
        project_title="t", style="style",
        frames=[Frame(id=n, description="", image_prompt="",
                      output_path=f"styled_images/{n}.png") for n in ("a", "b")],
        transitions=[Transition(
            id="a_to_b", start_frame="styled_images/a.png",
            end_frame="styled_images/b.png", motion_prompt="the bald man walks",
            duration=5, output_path="clips/a_to_b.mp4")],
    ).save(workspace.default_storyboard_json)
    return workspace


class TestRenderingKeepsTheTakeItReplaces:
    def _render(self, make_pipeline, **options):
        pipeline = make_pipeline(no_audio=True, **options)
        pipeline.cmd_render()
        return pipeline

    def test_a_first_render_has_no_previous_take(self, make_pipeline, project):
        pipeline = self._render(make_pipeline)
        assert not project.previous_clips_dir.exists()
        assert pipeline.snapshot()["clips"][0]["previous"] is False

    def test_a_re_render_keeps_the_old_take_and_reports_it(
        self, make_pipeline, project
    ):
        self._render(make_pipeline)
        pipeline = self._render(make_pipeline, clips=["a_to_b"])
        assert (project.clips_dir / "a_to_b.mp4").read_bytes() == b"take 2"
        assert (project.previous_clips_dir / "a_to_b.mp4").read_bytes() == b"take 1"
        assert pipeline.snapshot()["clips"][0]["previous"] is True

    def test_previous_takes_are_never_part_of_the_movie(
        self, make_pipeline, project
    ):
        self._render(make_pipeline)
        pipeline = self._render(make_pipeline, clips=["a_to_b"])
        snap = pipeline.snapshot()
        assert [c["id"] for c in snap["clips"]] == ["a_to_b"]
        assert snap["stray_clips"] == []
        assert [p.name for p in project.clips_dir.iterdir()] == ["a_to_b.mp4"]

    def test_restore_puts_take_one_back_and_knows_it_matches(
        self, make_pipeline, project
    ):
        self._render(make_pipeline)
        self._render(make_pipeline, clips=["a_to_b"])
        result = make_pipeline().restore_previous_take("a_to_b")
        assert result == {"clip": "a_to_b", "matches_plan": True}
        assert (project.clips_dir / "a_to_b.mp4").read_bytes() == b"take 1"

    def test_restore_after_a_prompt_edit_marks_the_old_take_outdated(
        self, make_pipeline, project
    ):
        self._render(make_pipeline)
        sb = Storyboard.load(project.default_storyboard_json)
        sb.transitions[0].motion_prompt = "the bald man kneels"
        sb.save(project.default_storyboard_json)
        self._render(make_pipeline, clips=["a_to_b"])
        pipeline = make_pipeline()
        assert pipeline.restore_previous_take("a_to_b")["matches_plan"] is False
        assert pipeline.snapshot()["clips"][0]["stale"] is True

    def test_unknown_clip_and_nothing_kept_are_clear_errors(
        self, make_pipeline, project
    ):
        self._render(make_pipeline)
        with pytest.raises(PipelineError, match="Unknown clip"):
            make_pipeline().restore_previous_take("x_to_y")
        with pytest.raises(PipelineError, match="No previous take"):
            make_pipeline().restore_previous_take("a_to_b")


# ------------------------------------------------------------- over the API

TOKEN = "t" * 32


@pytest.fixture
def api(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
    }))
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
    monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)
    monkeypatch.setattr(JobRunner, "_loop", lambda self: None)
    from ai_video_maker.workspace import Workspace
    ws = Workspace.for_project("matan")
    ws.mkdirs()
    for n in ("a", "b"):
        (ws.styled_images_dir / f"{n}.png").write_bytes(n.encode())
    Storyboard(
        project_title="t", style="style",
        frames=[Frame(id=n, description="", image_prompt="",
                      output_path=f"styled_images/{n}.png") for n in ("a", "b")],
        transitions=[Transition(
            id="a_to_b", start_frame="styled_images/a.png",
            end_frame="styled_images/b.png", motion_prompt="walk",
            duration=5, output_path="clips/a_to_b.mp4")],
    ).save(ws.default_storyboard_json)
    (ws.clips_dir / "a_to_b.mp4").write_bytes(b"take two")
    ws.previous_clips_dir.mkdir()
    (ws.previous_clips_dir / "a_to_b.mp4").write_bytes(b"take one")
    return TestClient(create_app(config_path, watch=False)), ws


def _post(client, path):
    return client.post(path, headers={"Authorization": f"Bearer {TOKEN}"})


class TestRestoreEndpoint:
    URL = "/api/projects/matan/clips/a_to_b/restore-previous"

    def test_it_swaps_and_can_be_undone(self, api):
        client, ws = api
        assert _post(client, self.URL).json()["ok"] is True
        assert (ws.clips_dir / "a_to_b.mp4").read_bytes() == b"take one"
        _post(client, self.URL)
        assert (ws.clips_dir / "a_to_b.mp4").read_bytes() == b"take two"

    def test_it_needs_the_token(self, api):
        client, ws = api
        assert client.post(self.URL).status_code == 401
        assert (ws.clips_dir / "a_to_b.mp4").read_bytes() == b"take two"

    def test_it_is_refused_while_a_job_runs_on_the_project(self, api):
        client, ws = api
        _post(client, "/api/projects/matan/actions/render")
        resp = _post(client, self.URL)
        assert resp.status_code == 409
        assert (ws.clips_dir / "a_to_b.mp4").read_bytes() == b"take two"

    def test_a_clip_with_nothing_kept_is_a_404(self, api):
        client, ws = api
        (ws.previous_clips_dir / "a_to_b.mp4").unlink()
        assert _post(client, self.URL).status_code == 404

    def test_the_previous_take_is_served_as_its_own_media_kind(self, api):
        client, _ = api
        resp = client.get(
            "/api/projects/matan/files/previous/a_to_b.mp4",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200 and resp.content == b"take one"


class TestPulse:
    """The cheap "did anything change?" the panel polls instead of the
    whole project. The one failure that matters is a wrong "unchanged"."""

    URL = "/api/projects/matan/pulse"

    def _version(self, client):
        resp = client.get(self.URL, headers={"Authorization": f"Bearer {TOKEN}"})
        assert resp.status_code == 200
        return resp.json()

    def test_it_is_stable_while_nothing_happens(self, api):
        client, _ = api
        assert self._version(client) == self._version(client)

    def test_it_needs_the_token(self, api):
        client, _ = api
        assert client.get(self.URL).status_code == 401

    def test_a_new_file_moves_it(self, api):
        client, ws = api
        before = self._version(client)["version"]
        (ws.styled_images_dir / "c.png").write_bytes(b"c")
        assert self._version(client)["version"] != before

    def test_a_file_replaced_in_place_moves_it(self, api):
        import os
        client, ws = api
        before = self._version(client)["version"]
        clip = ws.clips_dir / "a_to_b.mp4"
        clip.write_bytes(b"take 2!!")  # same length as before, new content
        stamp = clip.stat().st_mtime + 5
        os.utime(clip, (stamp, stamp))
        assert self._version(client)["version"] != before

    def test_a_removed_file_moves_it(self, api):
        client, ws = api
        before = self._version(client)["version"]
        (ws.styled_images_dir / "a.png").unlink()
        assert self._version(client)["version"] != before

    def test_a_job_starting_moves_it_and_reports_active(self, api):
        client, _ = api
        before = self._version(client)
        assert before["active"] is False
        _post(client, "/api/projects/matan/actions/render")
        after = self._version(client)
        assert after["active"] is True and after["version"] != before["version"]

    def test_a_growing_run_log_does_not_move_it(self, api):
        # Otherwise every line a job prints would trigger a full reload,
        # and this would be a costlier way of doing what it replaced.
        client, ws = api
        before = self._version(client)["version"]
        (ws.logs_dir / "pipeline_20260101.log").write_text("a line\n" * 50)
        assert self._version(client)["version"] == before

    def test_progress_recorded_in_state_moves_it(self, api):
        from ai_video_maker.state import StateStore
        client, ws = api
        before = self._version(client)["version"]
        StateStore(ws.state_file).set("sfx:a_to_b.mp4", "done")
        assert self._version(client)["version"] != before
