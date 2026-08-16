"""The cast-faces surface through the admin API.

The panel is the only place most of this is used, and behind the tunnel there
is no shell — so the two things worth pinning are that `faces` can be reached
at all, and that reaching it can never turn into a credit spend nobody asked
for.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ai_video_maker.server import create_app
from ai_video_maker.workspace import Workspace

TOKEN = "a-perfectly-long-admin-token"
PROJECT = "demo"


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
    # JobRunner.enqueue RUNS the job, so a posted action would really style
    # and really call OpenAI. Capture instead — CI is keyless and must stay
    # incapable of spending anything.
    monkeypatch.setattr(
        "ai_video_maker.server.JobRunner.enqueue",
        lambda self, project, command, options=None: _record(
            project, command, options
        ),
    )
    Workspace.for_project(PROJECT).mkdirs()
    return TestClient(create_app(config_path, watch=False))


ENQUEUED: list[dict] = []


class _Job:
    def summary(self):
        return {"id": "job-1", "status": "queued"}


def _record(project, command, options=None):
    ENQUEUED.append(
        {"project": project, "command": command, "options": options or {}}
    )
    return _Job()


def _post(client: TestClient, options: dict):
    # The action endpoint takes the options at the TOP level of the body,
    # which is what api.js posts.
    ENQUEUED.clear()
    return client.post(
        f"/api/projects/{PROJECT}/actions/faces", json=options,
        headers={"Authorization": f"Bearer {TOKEN}"},
    )


class TestTheFacesActionIsReachable:
    def test_a_plain_rebuild_is_accepted(self, client):
        assert _post(client, {}).status_code == 200
        assert ENQUEUED[0]["command"] == "faces"

    def test_the_audit_flag_is_carried_through(self, client):
        assert _post(client, {"audit_faces": True}).status_code == 200
        assert ENQUEUED[0]["options"]["audit_faces"] is True

    def test_the_audit_is_never_implied(self, client):
        # Rebuilding is free and the audit is not. A request that did not ask
        # for the paid half must never get it.
        _post(client, {})
        assert not ENQUEUED[0]["options"].get("audit_faces")

    def test_it_still_needs_a_token(self, client):
        assert client.post(
            f"/api/projects/{PROJECT}/actions/faces", json={}
        ).status_code in (401, 403)


class TestCastFacesAreServedToThePanel:
    def _face(self) -> Path:
        ws = Workspace.for_project(PROJECT)
        path = ws.cast_refs_dir / "c1.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (10, 20, 30)).save(path)
        return path

    def test_a_reference_can_be_fetched(self, client):
        self._face()
        resp = client.get(
            f"/api/projects/{PROJECT}/files/cast_refs/c1.png?token={TOKEN}"
        )
        assert resp.status_code == 200

    def test_a_missing_reference_is_a_404_not_a_crash(self, client):
        assert client.get(
            f"/api/projects/{PROJECT}/files/cast_refs/nobody.png?token={TOKEN}"
        ).status_code == 404

    def test_the_route_still_refuses_to_walk_out_of_the_directory(self, client):
        self._face()
        assert client.get(
            f"/api/projects/{PROJECT}/files/cast_refs/..%2F..%2Fconfig.json"
            f"?token={TOKEN}"
        ).status_code == 404
