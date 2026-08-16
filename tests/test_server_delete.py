"""Deleting a project through the admin API.

The most destructive thing this API can do. Everything else the panel offers
can be redone by spending credits again; this cannot be undone at all —
`projects/` holds paid styled frames and rendered clips, has no backup, and
once held 606 MB that a careless `git checkout` destroyed permanently.

So these tests are almost entirely about the guards REFUSING, and about a
refusal leaving the workspace untouched. The happy path is one test.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.server import create_app
from ai_video_maker.workspace import Workspace

TOKEN = "a-perfectly-long-admin-token"
PROJECT = "demo"


@pytest.fixture
def projects_dir(tmp_path: Path, monkeypatch) -> Path:
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
    return projects


@pytest.fixture
def client(tmp_path: Path, projects_dir: Path, monkeypatch) -> TestClient:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
    }), encoding="utf-8")
    monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)
    ws = Workspace.for_project(PROJECT)
    ws.mkdirs()
    (ws.input_images_dir / "01.jpg").write_bytes(b"photo")
    (ws.input_images_dir / "02.jpg").write_bytes(b"photo")
    (ws.styled_images_dir / "01.png").write_bytes(b"styled")
    (ws.clips_dir / "01_to_02.mp4").write_bytes(b"clip")
    ws.final_video.write_bytes(b"movie")
    return TestClient(create_app(config_path, watch=False))


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _delete(client, name, confirm=None):
    confirm = name if confirm is None else confirm
    return client.delete(
        f"/api/projects/{name}?confirm={confirm}", headers=_auth()
    )


class TestDeletingAProject:
    def test_the_workspace_and_its_contents_are_gone(self, client, projects_dir):
        resp = _delete(client, PROJECT)
        assert resp.status_code == 200
        assert resp.json()["deleted"] == PROJECT
        assert not (projects_dir / PROJECT).exists()

    def test_it_reports_what_it_destroyed(self, client):
        # The response is the last record of paid work that no longer exists,
        # so it has to be specific rather than {"ok": true}.
        contents = _delete(client, PROJECT).json()["contents"]
        assert contents["photos"] == 2
        assert contents["styled frames"] == 1
        assert contents["rendered clips"] == 1
        assert contents["final movies"] == 1

    def test_deleting_needs_a_token_like_everything_else(self, client, projects_dir):
        resp = client.delete(f"/api/projects/{PROJECT}?confirm={PROJECT}")
        assert resp.status_code == 401
        assert (projects_dir / PROJECT).exists()


class TestARefusalLeavesEverythingWhereItIs:
    """Every guard, checked by what SURVIVES it — not just the status code."""

    def test_a_confirm_that_does_not_name_the_project(self, client, projects_dir):
        resp = _delete(client, PROJECT, confirm="")
        assert resp.status_code == 400
        assert (projects_dir / PROJECT).exists()

    def test_a_confirm_naming_a_different_project(self, client, projects_dir):
        resp = _delete(client, PROJECT, confirm="something-else")
        assert resp.status_code == 400
        assert (projects_dir / PROJECT / "input_images" / "01.jpg").exists()

    def test_a_project_that_does_not_exist(self, client):
        assert _delete(client, "no-such-project").status_code == 404

    def test_studio_wide_data_is_not_a_project(self, client, projects_dir):
        # `_learning` holds the lessons learned across every movie ever made.
        learning = projects_dir / "_learning"
        learning.mkdir()
        (learning / "lessons.json").write_text("{}", encoding="utf-8")
        assert _delete(client, "_learning").status_code == 400
        assert (learning / "lessons.json").exists()

    def test_a_name_that_tries_to_escape_the_projects_directory(
        self, client, tmp_path, projects_dir
    ):
        outside = tmp_path / "precious"
        outside.mkdir()
        (outside / "keep.txt").write_text("hi", encoding="utf-8")
        for name in ("..", "../precious", "%2e%2e%2fprecious"):
            # WHICH refusal does not matter — a traversal may never reach the
            # handler at all (".." normalises to the collection route, 405).
            # What matters is that nothing outside projects/ is touched.
            assert not _delete(client, name).is_success
        assert (outside / "keep.txt").exists()
        assert outside.is_dir()

    def test_a_symlinked_workspace_is_refused_not_followed(
        self, client, tmp_path, projects_dir
    ):
        # The dev tree symlinks projects/ at the production checkout; deleting
        # THROUGH a link is exactly how a real 606 MB loss happened.
        real = tmp_path / "elsewhere"
        real.mkdir()
        (real / "clip.mp4").write_bytes(b"paid work")
        (projects_dir / "linked").symlink_to(real, target_is_directory=True)
        assert _delete(client, "linked").status_code == 400
        assert (real / "clip.mp4").exists()

    def test_a_project_with_a_running_job(self, client, projects_dir, monkeypatch):
        # Deleting the files under a live pipeline leaves it writing into a
        # directory that no longer exists.
        monkeypatch.setattr(
            "ai_video_maker.server.JobRunner.list",
            lambda self, project=None, limit=50: [
                SimpleNamespace(command="render", state="running")
            ],
        )
        resp = _delete(client, PROJECT)
        assert resp.status_code == 409
        assert "render" in resp.json()["detail"]
        assert (projects_dir / PROJECT).exists()

    def test_a_queued_job_counts_as_busy_too(self, client, projects_dir, monkeypatch):
        monkeypatch.setattr(
            "ai_video_maker.server.JobRunner.list",
            lambda self, project=None, limit=50: [
                SimpleNamespace(command="storyboard", state="queued")
            ],
        )
        assert _delete(client, PROJECT).status_code == 409
        assert (projects_dir / PROJECT).exists()

    def test_a_finished_job_does_not_block(self, client, projects_dir, monkeypatch):
        monkeypatch.setattr(
            "ai_video_maker.server.JobRunner.list",
            lambda self, project=None, limit=50: [
                SimpleNamespace(command="render", state="done")
            ],
        )
        assert _delete(client, PROJECT).status_code == 200
        assert not (projects_dir / PROJECT).exists()
