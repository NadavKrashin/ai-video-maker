"""Saving a hand-edited storyboard through the admin API.

The panel's "edit prompts -> Save -> generate everything at once" flow depends
on the save marking the clips the edits invalidated: without it a hand edit is
invisible to everything but the browser tab that made it, and the batch render
has no way to know which clips to redo.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.config import Config
from ai_video_maker.models import Storyboard
from ai_video_maker.options import RunOptions
from ai_video_maker.runner import Pipeline
from ai_video_maker.server import _outcome, apply_in_flight, create_app
from ai_video_maker.state import StateStore
from ai_video_maker.workspace import Workspace

TOKEN = "a-perfectly-long-admin-token"
PROJECT = "demo"


def _storyboard(motion: str = "the bald man walks to the door",
                duration: int = 5, sound: str = "footsteps") -> dict:
    return {
        "project_title": "demo", "style": "pixar",
        "frames": [
            {"id": "a", "description": "d", "image_prompt": "",
             "output_path": "styled_images/a.png"},
            {"id": "b", "description": "d", "image_prompt": "",
             "output_path": "styled_images/b.png"},
        ],
        "transitions": [{
            "id": "a_to_b", "start_frame": "styled_images/a.png",
            "end_frame": "styled_images/b.png", "motion_prompt": motion,
            "duration": duration, "sound_prompt": sound,
            "output_path": "clips/a_to_b.mp4",
        }],
    }


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
    }), encoding="utf-8")
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
    monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)

    ws = Workspace.for_project(PROJECT)
    ws.mkdirs()
    ws.default_storyboard_json.write_text(
        json.dumps(_storyboard()), encoding="utf-8")
    # A rendered clip is what staleness is about — an unrendered pair has
    # nothing to invalidate.
    (ws.clips_dir / "a_to_b.mp4").write_bytes(b"fake mp4")
    return TestClient(create_app(config_path, watch=False))


def _put(client: TestClient, body: dict):
    return client.put(
        f"/api/projects/{PROJECT}/storyboard", json=body,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


def _stale() -> bool:
    """Is the rendered clip flagged as no longer matching the storyboard?"""
    ws = Workspace.for_project(PROJECT)
    return StateStore(ws.state_file).status("stale:a_to_b.mp4") is not None


class TestSaveStoryboardMarksOutdated:
    def test_motion_prompt_edit_marks_the_rendered_clip(self, client):
        res = _put(client, _storyboard(motion="the bald man opens the door"))
        assert res.status_code == 200
        assert res.json()["outdated"] == ["a_to_b"]
        assert _stale()

    def test_unchanged_save_marks_nothing(self, client):
        res = _put(client, _storyboard())
        assert res.status_code == 200
        assert res.json()["outdated"] == []
        assert not _stale()

    def test_sound_only_edit_does_not_touch_the_video(self, client):
        res = _put(client, _storyboard(sound="a door creaks"))
        assert res.status_code == 200
        assert res.json()["outdated"] == []
        assert not _stale()

    def test_edit_without_a_rendered_clip_marks_nothing(self, client):
        Workspace.for_project(PROJECT).clips_dir.joinpath("a_to_b.mp4").unlink()
        res = _put(client, _storyboard(motion="something else entirely"))
        assert res.status_code == 200
        # The transition changed, but there is no clip to invalidate.
        assert res.json()["changed"] == ["a_to_b"]
        assert res.json()["outdated"] == []

    def test_edit_is_persisted(self, client):
        _put(client, _storyboard(motion="a brand new prompt", duration=10))
        saved = json.loads(
            Workspace.for_project(PROJECT)
            .default_storyboard_json.read_text(encoding="utf-8")
        )
        assert saved["transitions"][0]["motion_prompt"] == "a brand new prompt"
        assert saved["transitions"][0]["duration"] == 10

    def test_invalid_storyboard_is_rejected_and_changes_nothing(self, client):
        before = Workspace.for_project(PROJECT).default_storyboard_json.read_text(
            encoding="utf-8")
        res = _put(client, {"project_title": "only this"})
        assert res.status_code == 422
        assert Workspace.for_project(PROJECT).default_storyboard_json.read_text(
            encoding="utf-8") == before


class TestSaveStoryboardOrphansPaidRenders:
    """Editing a clip that has an uncollected paid render throws it away.

    The fingerprint stops matching, so the next render buys a fresh clip and
    the submitted one is money gone. The save reports it so the panel can say
    so instead of absorbing the loss silently.
    """

    def _pending(self, fingerprint: str) -> None:
        ws = Workspace.for_project(PROJECT)
        StateStore(ws.state_file).set(
            "falreq:a_to_b.mp4", "pending",
            request_id="req-1", fingerprint=fingerprint,
        )

    def _live_fingerprint(self) -> str:
        """The fingerprint a render of a_to_b would compute right now."""
        ws = Workspace.for_project(PROJECT)
        cfg = Config(style_prompt="s", scratch_style_prompt="s", motion_prompt="m")
        p = Pipeline(cfg, ws, RunOptions())
        start, end, motion, duration, _ = p._pairs_from_storyboard(
            Storyboard.load(ws.default_storyboard_json)
        )[0]
        return p.video_client.fingerprint(start, end, motion, duration)

    def _frames(self) -> None:
        ws = Workspace.for_project(PROJECT)
        for n in ("a", "b"):
            (ws.styled_images_dir / f"{n}.png").write_bytes(f"frame-{n}".encode())

    def test_editing_a_pending_clip_reports_the_loss(self, client):
        self._frames()
        self._pending(self._live_fingerprint())
        res = _put(client, _storyboard(motion="a completely different action"))
        assert res.json()["orphaned_renders"] == ["a_to_b"]

    def test_unrelated_save_keeps_the_paid_render(self, client):
        self._frames()
        self._pending(self._live_fingerprint())
        res = _put(client, _storyboard(sound="only the sound changed"))
        assert res.json()["orphaned_renders"] == []

    def test_already_unrecoverable_render_is_not_reported_again(self, client):
        # It stopped matching before this save, so this edit didn't cost it.
        self._frames()
        self._pending("a-stale-fingerprint")
        res = _put(client, _storyboard(motion="a completely different action"))
        assert res.json()["orphaned_renders"] == []


class TestPendingRendersInFlight:
    """A running job's own clips must not look like abandoned paid work.

    `falreq:` says "submitted, not downloaded" — equally true of a crashed
    run and of the render polling fal right now. Only the job runner can
    tell them apart, so the API marks the running job's clips in_flight.
    Without it every healthy render raised a false alarm.
    """

    def _pending(self, *ids: str) -> list[dict]:
        return [{"id": i, "in_flight": False} for i in ids]

    def test_no_running_job_leaves_everything_stranded(self):
        pending = self._pending("a_to_b", "b_to_c")
        apply_in_flight(pending, None)
        assert [p["in_flight"] for p in pending] == [False, False]

    def test_render_owns_only_the_clips_it_was_given(self):
        pending = self._pending("a_to_b", "b_to_c")
        apply_in_flight(pending, {"clips": ["a_to_b"]})
        # b_to_c was left behind by an earlier run: still stranded money.
        assert [p["in_flight"] for p in pending] == [True, False]

    def test_whole_project_render_owns_every_pending_clip(self):
        pending = self._pending("a_to_b", "b_to_c")
        apply_in_flight(pending, {})
        assert [p["in_flight"] for p in pending] == [True, True]

    def test_empty_clip_list_counts_as_whole_project(self):
        pending = self._pending("a_to_b")
        apply_in_flight(pending, {"clips": []})
        assert pending[0]["in_flight"] is True

    def test_api_reports_not_in_flight_when_idle(self, client):
        ws = Workspace.for_project(PROJECT)
        StateStore(ws.state_file).set(
            "falreq:a_to_b.mp4", "pending", request_id="r", fingerprint="fp")
        res = client.get(
            f"/api/projects/{PROJECT}",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert [p["in_flight"] for p in res.json()["pending_renders"]] == [False]


class TestJobOutcome:
    """A command that survives item failures must not report success.

    `execute()` returning is not proof the work happened: the pipeline keeps
    going when one clip fails so the rest still render. A real render whose
    three clips all timed out on fal reported "done" and looked identical to
    a successful one.
    """

    class _Pipeline:
        """Minimal stand-in exposing what _outcome reads."""

        def __init__(self, failures, created):
            from ai_video_maker.summary import RunSummary
            self.failed = type("F", (), {"failures": failures})()
            self.summary = RunSummary(videos_created=created)

    def _fail(self, kind="clip", error="timed out"):
        return {"job_id": f"{kind}:x", "kind": kind, "error": error}

    def test_clean_run_is_done(self):
        state, error, n = _outcome(self._Pipeline([], 3))
        assert (state, error, n) == ("done", "", 0)

    def test_every_item_failing_is_failed_not_done(self):
        state, error, n = _outcome(self._Pipeline([self._fail()] * 3, 0))
        assert state == "failed"
        assert n == 3
        assert "3 failed" in error and "timed out" in error

    def test_some_produced_some_failed_is_partial(self):
        state, _, n = _outcome(self._Pipeline([self._fail()], 5))
        assert state == "partial" and n == 1

    def test_error_breaks_down_by_kind(self):
        failures = [self._fail("clip"), self._fail("clip"), self._fail("sfx")]
        _, error, _ = _outcome(self._Pipeline(failures, 0))
        assert "2 clip" in error and "1 sfx" in error


class TestMusicFromUrlEndpoint:
    """POST /music/url writes the same file the upload does."""

    def _post(self, client: TestClient, url: str):
        return client.post(
            f"/api/projects/{PROJECT}/music/url", json={"url": url},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )

    def test_fetched_track_lands_in_the_custom_music_slot(self, client, monkeypatch):
        def fake_fetch(url, dst):
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"a-track")
            return dst

        monkeypatch.setattr("ai_video_maker.server.fetch_music", fake_fetch)
        res = self._post(client, "https://x.test/t.mp3")
        assert res.status_code == 200 and res.json()["bytes"] == 7
        assert Workspace.for_project(PROJECT).custom_music_file.read_bytes() == b"a-track"

    def test_bad_url_is_a_400_not_a_500(self, client):
        res = self._post(client, "ftp://x.test/t.mp3")
        assert res.status_code == 400
        assert "music URL" in res.json()["detail"]

    def test_network_failure_is_reported_as_502(self, client, monkeypatch):
        def boom(url, dst):
            raise RuntimeError("connection reset")

        monkeypatch.setattr("ai_video_maker.server.fetch_music", boom)
        res = self._post(client, "https://x.test/t.mp3")
        assert res.status_code == 502
        assert "connection reset" in res.json()["detail"]

    def test_endpoint_requires_the_token(self, client):
        res = client.post(f"/api/projects/{PROJECT}/music/url",
                          json={"url": "https://x.test/t.mp3"})
        assert res.status_code == 401
