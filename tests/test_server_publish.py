"""The admin API's side of publishing: preview the name, then run the job.

The panel must be able to show the EXACT file name before anything is
uploaded, and the name it showed has to be the name the job uses — that pair
is what makes "ask for approval first" mean something.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.clients.cloudinary_client import CloudinaryClient, PublishedVideo
from ai_video_maker.intake import write_order_record
from ai_video_maker.server import create_app
from ai_video_maker.workspace import Workspace

TOKEN = "a-perfectly-long-admin-token"
PROJECT = "demo"
FOLDER = "AM-280726-XY12_Dana-Cohen-28.07.2026_10-30"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class _FakeCloudinary:
    orders_folder = "video-orders"
    publish_basename = "final"

    def __init__(self, versions=()) -> None:
        self._versions = list(versions)

    def list_published_videos(self, folder_leaf):
        return [
            PublishedVideo(
                public_id=f"video-orders/{folder_leaf}/final_v{v}",
                url=f"https://res.cloudinary.com/final_v{v}.mp4", version=v,
            )
            for v in self._versions
        ]


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
    # The queue is exercised here, not the pipeline: a real publish job would
    # reach for Cloudinary credentials this suite deliberately doesn't have.
    monkeypatch.setattr(
        "ai_video_maker.server.JobRunner._run_job", lambda self, job: None
    )

    ws = Workspace.for_project(PROJECT)
    ws.mkdirs()
    ws.final_video.write_bytes(b"a finished movie")
    write_order_record(ws.order_file, order_folder=FOLDER, photo_count=3)
    return TestClient(create_app(config_path, watch=False))


def _preview(client: TestClient):
    return client.get(f"/api/projects/{PROJECT}/publish/preview", headers=AUTH)


class TestPublishPreview:
    def test_returns_the_next_free_name(self, client, monkeypatch):
        monkeypatch.setattr(
            CloudinaryClient, "from_config",
            staticmethod(lambda c: _FakeCloudinary([1, 2])),
        )
        body = _preview(client).json()
        assert body["public_id"] == f"video-orders/{FOLDER}/final_v3"
        assert body["filename"] == "final_v3.mp4"
        assert body["bytes"] == len(b"a finished movie")
        assert [p["version"] for p in body["published"]] == [1, 2]

    def test_requires_the_token(self, client):
        assert client.get(f"/api/projects/{PROJECT}/publish/preview").status_code == 401

    def test_a_project_without_an_order_is_a_clear_400(self, client, monkeypatch):
        Workspace.for_project(PROJECT).order_file.unlink()
        monkeypatch.setattr(
            CloudinaryClient, "from_config", staticmethod(lambda c: _FakeCloudinary()),
        )
        res = _preview(client)
        assert res.status_code == 400
        assert "order" in res.json()["detail"]

    def test_a_cloudinary_outage_reads_as_a_gateway_error(self, client, monkeypatch):
        def _explode(_config):
            raise RuntimeError("cloudinary is down")

        monkeypatch.setattr(CloudinaryClient, "from_config", staticmethod(_explode))
        res = _preview(client)
        assert res.status_code == 502
        assert "cloudinary is down" in res.json()["detail"]


class TestPublishAction:
    def test_publish_is_an_enqueueable_action(self, client):
        # The panel pins the approved name into the job; the runner refuses to
        # upload under any other one (see tests/test_publish.py).
        res = client.post(
            f"/api/projects/{PROJECT}/actions/publish",
            json={"publish_as": f"video-orders/{FOLDER}/final_v1"},
            headers=AUTH,
        )
        assert res.status_code == 200
        assert res.json()["job"]["command"] == "publish"

    def test_publish_options_survive_the_whitelist(self, client):
        from ai_video_maker.server import _ALLOWED_OPTIONS

        assert "publish_as" in _ALLOWED_OPTIONS


class TestArchivedDeliveriesAreDownloadable:
    """The panel offers each delivered version for download from the archive."""

    def test_an_archived_version_is_served(self, client):
        ws = Workspace.for_project(PROJECT)
        ws.published_dir.mkdir(parents=True, exist_ok=True)
        (ws.published_dir / "final_v1.mp4").write_bytes(b"the delivered cut")

        res = client.get(
            f"/api/projects/{PROJECT}/files/published/final_v1.mp4", headers=AUTH
        )
        assert res.status_code == 200 and res.content == b"the delivered cut"

    def test_paths_cannot_escape_the_archive(self, client):
        res = client.get(
            f"/api/projects/{PROJECT}/files/published/..%2Ffinal_video.mp4",
            headers=AUTH,
        )
        assert res.status_code == 404
