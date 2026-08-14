"""/api/orders: the pre-ingest photo count, and ingest that only ingests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.clients.cloudinary_client import OrderAsset
from ai_video_maker.server import create_app

TOKEN = "a-perfectly-long-admin-token"
LEAF = "AM-180726-XY12_Dana-Cohen-18.07.2026_10-30"


class FakeCloudinary:
    """Stands in for the Admin API; records what was asked for."""

    def __init__(self, assets_by_folder, boom=False):
        self.assets_by_folder = assets_by_folder
        self.boom = boom
        self.asset_calls: list[str] = []

    @classmethod
    def install(cls, monkeypatch, assets_by_folder, boom=False):
        fake = cls(assets_by_folder, boom)
        monkeypatch.setattr(
            "ai_video_maker.server.CloudinaryClient.from_config",
            classmethod(lambda _cls, _config: fake),
        )
        return fake

    def list_order_folders(self):
        return sorted(self.assets_by_folder, reverse=True)

    def list_order_assets(self, folder):
        self.asset_calls.append(folder)
        if self.boom:
            raise RuntimeError("cloudinary is down")
        return self.assets_by_folder.get(folder, [])


def _assets(n):
    return [
        OrderAsset(public_id=f"x/{i}", url=f"https://x/{i}", format="jpg",
                   position=i, created_at="2026-07-18T10:30:00Z")
        for i in range(1, n + 1)
    ]


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch) -> TestClient:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
    }), encoding="utf-8")
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
    monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)
    # OFFLINE, always. A service-account file sitting on the dev machine
    # otherwise makes this suite list the REAL Firestore orders — it did,
    # and the assertions started failing against a live customer's name.
    monkeypatch.setattr(
        "ai_video_maker.server.FirebaseClient.configured",
        classmethod(lambda _cls, _config: False),
    )
    return TestClient(create_app(config_path, watch=False))


@pytest.fixture
def enqueued(monkeypatch) -> list[dict]:
    """Capture enqueued jobs INSTEAD of running them.

    The JobRunner starts a job the moment it is enqueued, so letting the
    real one through made this test attempt a live ingest (it tried to
    download from the fake asset host, with retries). What is under test is
    what the endpoint asks for, not what the job then does.
    """
    calls: list[dict] = []

    def fake_enqueue(self, project, command, options, then=None):
        calls.append({"project": project, "command": command,
                      "options": options, "then": then})
        return SimpleNamespace(summary=lambda: {"id": "job-1"}, id="job-1")

    monkeypatch.setattr(
        "ai_video_maker.server.JobRunner.enqueue", fake_enqueue
    )
    return calls


def _auth():
    return {"Authorization": f"Bearer {TOKEN}"}


def _orders(client):
    resp = client.get("/api/orders", headers=_auth())
    assert resp.status_code == 200
    return resp.json()["orders"]


class TestPreIngestPhotoCount:
    """See what is actually in the folder before spending a click on it.

    The frontend confirms payment BEFORE the photos finish uploading, so a
    folder existing has never meant the order is whole.
    """

    def test_an_un_ingested_order_reports_its_cloudinary_photos(
        self, app_client, monkeypatch
    ):
        FakeCloudinary.install(monkeypatch, {LEAF: _assets(12)})
        assert _orders(app_client)[0]["cloudinary_photos"] == 12

    def test_an_empty_folder_reports_zero_not_nothing(
        self, app_client, monkeypatch
    ):
        # 0 and "couldn't count" are different answers: one says the customer
        # has uploaded nothing yet, the other says ask again later.
        FakeCloudinary.install(monkeypatch, {LEAF: []})
        assert _orders(app_client)[0]["cloudinary_photos"] == 0

    def test_a_listing_failure_never_takes_the_page_down(
        self, app_client, monkeypatch
    ):
        FakeCloudinary.install(monkeypatch, {LEAF: _assets(3)}, boom=True)
        rows = _orders(app_client)
        assert rows[0]["cloudinary_photos"] is None
        assert rows[0]["folder"] == LEAF  # the row itself still arrives

    def test_an_ingested_order_is_not_counted_again(
        self, app_client, monkeypatch, tmp_path
    ):
        # Its project snapshot already reports the photos that landed, so
        # counting would spend a Cloudinary call per row on every refresh.
        from ai_video_maker.intake import write_order_record
        project = tmp_path / "projects" / "dana"
        (project / "logs").mkdir(parents=True)
        write_order_record(project / "order.json",
                           order_folder=LEAF, photo_count=12)
        fake = FakeCloudinary.install(monkeypatch, {LEAF: _assets(12)})
        row = _orders(app_client)[0]
        assert row["project"] == "dana"
        assert row["cloudinary_photos"] is None
        assert fake.asset_calls == []


class TestIngestOnlyIngests:
    """Ingest creates the project and pulls the photos — nothing else.

    Storyboarding styles every photo and plans every pair, which spends real
    OpenAI credits, so it stays a separate deliberate click.
    """

    def _ingest(self, client, **body):
        return client.post("/api/orders/ingest",
                           json={"order": LEAF, **body}, headers=_auth())

    def test_ingest_alone_chains_nothing(self, app_client, enqueued):
        assert self._ingest(app_client).status_code == 200
        assert len(enqueued) == 1
        assert enqueued[0]["command"] == "ingest"
        assert enqueued[0]["then"] is None  # no storyboard, no credits

    def test_the_chain_is_still_available_when_asked_for(
        self, app_client, enqueued
    ):
        # The API keeps the capability (the watcher's opt-in auto-storyboard
        # uses the same path); the panel simply stops asking for it.
        assert self._ingest(app_client, storyboard=True).status_code == 200
        assert enqueued[0]["then"] == {"command": "storyboard", "options": {}}
