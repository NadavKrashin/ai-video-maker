"""Marking a clip from the panel.

Free, instant and reversible, so unlike every spending action this one does
NOT go through the job runner — there is nothing to queue. What has to hold
is that it cannot reach anything but the worklist: no deletes, no renders,
and no writing outside the project.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

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
    ws = Workspace.for_project(PROJECT)
    ws.mkdirs()
    ws.clips_dir.mkdir(parents=True, exist_ok=True)
    (ws.clips_dir / "003_to_004.mp4").write_bytes(b"rendered")
    return TestClient(create_app(config_path, watch=False))


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _flag(client, clip="003_to_004", **body):
    return client.put(
        f"/api/projects/{PROJECT}/clips/{clip}/flag", json=body, headers=_auth()
    )


def _worklist(tmp_path) -> dict:
    path = tmp_path / "projects" / PROJECT / "clip_flags.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TestMarking:
    def test_a_rendered_clip_can_be_marked(self, client, tmp_path):
        res = _flag(client, marked=True, note="teleports")
        assert res.status_code == 200, res.text
        assert res.json()["marked"] is True
        flags = _worklist(tmp_path)["flags"]
        assert [f["transition_id"] for f in flags] == ["003_to_004"]
        assert flags[0]["note"] == "teleports"

    def test_marking_is_the_default(self, client, tmp_path):
        assert _flag(client).status_code == 200
        assert _worklist(tmp_path)["flags"]

    def test_a_mark_can_be_cleared(self, client, tmp_path):
        _flag(client, marked=True)
        res = _flag(client, marked=False)
        assert res.status_code == 200
        assert _worklist(tmp_path)["flags"] == []

    def test_an_unrendered_clip_is_a_404(self, client):
        res = _flag(client, clip="004_to_005", marked=True)
        assert res.status_code == 404
        assert "render it first" in res.json()["detail"]

    def test_clearing_an_unrendered_clip_still_works(self, client):
        """Ticking off a mark must not depend on the clip still being there."""
        assert _flag(client, clip="004_to_005", marked=False).status_code == 200

    def test_the_clip_file_is_never_touched(self, client, tmp_path):
        path = tmp_path / "projects" / PROJECT / "clips" / "003_to_004.mp4"
        before = path.read_bytes()
        _flag(client, marked=True)
        _flag(client, marked=False)
        assert path.exists() and path.read_bytes() == before


class TestItRefusesJunk:
    @pytest.mark.parametrize("clip_id", ["..", ".", "..%2F..%2Fevil", "a%5Cb"])
    def test_a_traversing_clip_id_writes_nothing_outside(
        self, client, tmp_path, clip_id
    ):
        """Assert what SURVIVES, not the status code: routing rejects some of
        these before the handler sees them and the guard catches the rest,
        and what matters either way is that nothing landed on disk."""
        res = client.put(
            f"/api/projects/{PROJECT}/clips/{clip_id}/flag",
            json={"marked": True}, headers=_auth(),
        )
        assert res.status_code != 200, res.text
        assert not (tmp_path / "projects" / PROJECT / "clip_flags.json").exists()
        assert list((tmp_path / "projects").iterdir()) == [
            tmp_path / "projects" / PROJECT
        ]

    def test_marked_must_be_a_boolean(self, client):
        assert _flag(client, marked="yes").status_code == 422

    def test_the_note_must_be_a_string(self, client):
        assert _flag(client, marked=True, note=5).status_code == 422

    def test_it_needs_a_token(self, client):
        res = client.put(
            f"/api/projects/{PROJECT}/clips/003_to_004/flag", json={"marked": True}
        )
        assert res.status_code in (401, 403)
