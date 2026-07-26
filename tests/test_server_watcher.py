"""OrderWatcher decisions against fake Firestore/Cloudinary clients."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ai_video_maker.clients.cloudinary_client import OrderAsset
from ai_video_maker.clients.firebase_client import FirestoreOrder
from ai_video_maker.config import Config
from ai_video_maker.server import JobRunner, OrderWatcher

LEAF = "AM-180726-XY12_Dana-Cohen-18.07.2026_10-30"


class FakeFirebase:
    def __init__(self, orders):
        self.orders = orders
        self.updates: list[tuple[str, dict]] = []

    def list_orders(self):
        return self.orders

    def update_order(self, order_id, fields):
        self.updates.append((order_id, fields))


class FakeCloudinary:
    def __init__(self, assets_by_folder):
        self.assets_by_folder = assets_by_folder

    def list_order_folders(self):
        return sorted(self.assets_by_folder, reverse=True)

    def list_order_assets(self, folder):
        return self.assets_by_folder.get(folder, [])


def _assets(n: int, minutes_ago: float = 60) -> list[OrderAsset]:
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    return [
        OrderAsset(public_id=f"x/{i}", url=f"https://x/{i}", format="jpg",
                   position=i, created_at=stamp)
        for i in range(1, n + 1)
    ]


def _order(status="new", folder=f"video-orders/{LEAF}", photo_count=None):
    return FirestoreOrder(order_id="AM-180726-XY12", customer="Dana Cohen",
                          folder=folder, status=status, photo_count=photo_count)


@pytest.fixture
def watch(tmp_path, monkeypatch):
    """A watcher wired to fakes and a worker-less JobRunner in a tmp projects dir."""
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    # The service-account key that flips the watcher into Firestore mode.
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"project_id": "test"}))
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(key))

    def _make(firebase, cloudinary):
        config = Config(style_prompt="s", scratch_style_prompt="s", motion_prompt="m")
        jobs = JobRunner(tmp_path / "config.json", start=False)
        watcher = OrderWatcher(
            config, tmp_path / "config.json", jobs,
            cloudinary_factory=lambda cfg: cloudinary,
            firebase_factory=lambda cfg: firebase,
        )
        return watcher, jobs, projects

    return _make


def _mark_ingested(projects: Path, leaf: str, project: str) -> None:
    root = projects / project
    root.mkdir()
    (root / "order.json").write_text(json.dumps({"order_folder": leaf}))


class TestFirestorePoll:
    def test_new_complete_order_is_ingested_and_marked(self, watch):
        firebase = FakeFirebase([_order()])
        watcher, jobs, _ = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        assert watcher.poll_once() == [LEAF]
        queued = jobs.list()
        assert [(j.command, j.options["order"]) for j in queued] == [("ingest", LEAF)]
        assert queued[0].then == {"command": "storyboard", "options": {}}
        assert firebase.updates == [("AM-180726-XY12", {"status": "ingesting"})]

    def test_fresh_upload_waits(self, watch):
        firebase = FakeFirebase([_order()])
        watcher, jobs, _ = watch(
            firebase, FakeCloudinary({LEAF: _assets(3, minutes_ago=1)})
        )
        assert watcher.poll_once() == []
        assert jobs.list() == []
        assert firebase.updates == []

    def test_exact_photo_count_skips_quiet_period(self, watch):
        firebase = FakeFirebase([_order(photo_count=3)])
        watcher, jobs, _ = watch(
            firebase, FakeCloudinary({LEAF: _assets(3, minutes_ago=0)})
        )
        assert watcher.poll_once() == [LEAF]

    def test_order_without_folder_is_left_alone(self, watch):
        firebase = FakeFirebase([_order(folder="")])
        watcher, jobs, _ = watch(firebase, FakeCloudinary({}))
        assert watcher.poll_once() == []
        assert firebase.updates == []

    def test_already_ingested_syncs_status_once(self, watch):
        firebase = FakeFirebase([_order(status="ingesting")])
        watcher, jobs, projects = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        _mark_ingested(projects, LEAF, "dana-cohen")
        assert watcher.poll_once() == []
        assert jobs.list() == []
        assert firebase.updates == [
            ("AM-180726-XY12", {"status": "ingested", "project": "dana-cohen"}),
        ]

    def test_later_status_never_downgraded(self, watch):
        firebase = FakeFirebase([_order(status="delivered")])
        watcher, jobs, projects = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        _mark_ingested(projects, LEAF, "dana-cohen")
        assert watcher.poll_once() == []
        assert firebase.updates == []

    def test_failed_ingest_retries_on_next_poll(self, watch):
        # Status was already bumped to "ingesting" but the job died before
        # order.json existed: the next poll must pick the order up again.
        firebase = FakeFirebase([_order(status="ingesting")])
        watcher, jobs, _ = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        assert watcher.poll_once() == [LEAF]

    def test_active_ingest_job_not_requeued(self, watch):
        firebase = FakeFirebase([_order()])
        watcher, jobs, _ = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        assert watcher.poll_once() == [LEAF]
        # Second poll while the (worker-less) job is still queued.
        assert watcher.poll_once() == []
        assert len(jobs.list()) == 1


class TestCloudinaryFallback:
    def test_without_firebase_key_polls_cloudinary(self, watch, monkeypatch):
        monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        monkeypatch.setattr(
            "ai_video_maker.clients.firebase_client.PROJECT_ROOT",
            Path("/nonexistent"),
        )
        firebase = FakeFirebase([])  # must never be consulted
        firebase.list_orders = None  # would raise if called
        watcher, jobs, _ = watch(firebase, FakeCloudinary({LEAF: _assets(3)}))
        assert watcher.poll_once() == [LEAF]
        assert [j.command for j in jobs.list()] == ["ingest"]
