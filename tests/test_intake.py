"""Web-order intake logic: folder parsing, naming, completeness, records."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from ai_video_maker.clients.cloudinary_client import OrderAsset
from ai_video_maker.intake import (
    derive_project_name,
    ingested_orders,
    is_order_complete,
    parse_order_folder,
    read_order_record,
    write_order_record,
)

LEAF = "AM-160726-3BWH_Liat-Heitner-16.07.2026_09-42"


def _asset(created_at: str) -> OrderAsset:
    return OrderAsset(public_id="x/1", url="https://x/1", format="jpg",
                      position=1, created_at=created_at)


class TestParseOrderFolder:
    def test_frontend_format(self):
        parsed = parse_order_folder(LEAF)
        assert parsed == {
            "order_id": "AM-160726-3BWH",
            "customer": "Liat Heitner",
            "stamp": "16.07.2026_09-42",
        }

    def test_unparseable_degrades_to_order_id(self):
        parsed = parse_order_folder("my-manual-folder")
        assert parsed["order_id"] == "my-manual-folder"
        assert parsed["customer"] == ""


class TestDeriveProjectName:
    def test_customer_name_slug(self):
        assert derive_project_name(LEAF, set()) == "liat-heitner"

    def test_collision_gets_suffix(self):
        assert derive_project_name(LEAF, {"liat-heitner"}) == "liat-heitner-2"
        assert derive_project_name(
            LEAF, {"liat-heitner", "liat-heitner-2"}
        ) == "liat-heitner-3"

    def test_hebrew_customer_falls_back_to_order_id(self):
        leaf = "AM-160726-3BWH_ליאת-הייטנר-16.07.2026_09-42"
        assert derive_project_name(leaf, set()) == "am-160726-3bwh"

    def test_garbage_never_returns_empty(self):
        assert derive_project_name("___", set()) != ""


class TestIsOrderComplete:
    NOW = datetime(2026, 7, 16, 12, 0, tzinfo=timezone.utc)

    def test_quiet_order_is_complete(self):
        assets = [_asset("2026-07-16T11:30:00Z"), _asset("2026-07-16T11:45:00Z")]
        assert is_order_complete(assets, quiet_minutes=10, now=self.NOW)

    def test_fresh_upload_is_not_complete(self):
        assets = [_asset("2026-07-16T11:30:00Z"), _asset("2026-07-16T11:55:00Z")]
        assert not is_order_complete(assets, quiet_minutes=10, now=self.NOW)

    def test_empty_folder_is_not_complete(self):
        assert not is_order_complete([], quiet_minutes=10, now=self.NOW)

    def test_unparseable_timestamp_counts_as_fresh(self):
        assets = [_asset("2026-07-16T11:00:00Z"), _asset("not-a-date")]
        assert not is_order_complete(assets, quiet_minutes=10, now=self.NOW)

    def test_boundary_is_complete(self):
        assets = [_asset("2026-07-16T11:50:00Z")]
        assert is_order_complete(assets, quiet_minutes=10, now=self.NOW)


class TestOrderRecords:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "order.json"
        write_order_record(path, order_folder=LEAF, photo_count=8)
        record = read_order_record(path)
        assert record["order_folder"] == LEAF
        assert record["order_id"] == "AM-160726-3BWH"
        assert record["customer"] == "Liat Heitner"
        assert record["photo_count"] == 8

    def test_missing_and_corrupt_return_none(self, tmp_path):
        assert read_order_record(tmp_path / "nope.json") is None
        bad = tmp_path / "order.json"
        bad.write_text("{broken", encoding="utf-8")
        assert read_order_record(bad) is None

    def test_ingested_orders_mapping(self, tmp_path):
        for project, folder in [("liat", LEAF), ("other", "AM-2_X-01.01.2026_00-00")]:
            d = tmp_path / project
            d.mkdir()
            write_order_record(d / "order.json", order_folder=folder, photo_count=1)
        (tmp_path / "manual-project").mkdir()  # no order.json — not listed
        mapping = ingested_orders(tmp_path)
        assert mapping == {LEAF: "liat", "AM-2_X-01.01.2026_00-00": "other"}

    def test_ingested_orders_missing_dir(self, tmp_path):
        assert ingested_orders(tmp_path / "absent") == {}


class TestJobRunnerQueueing:
    def _runner(self, tmp_path):
        from ai_video_maker.server import JobRunner
        return JobRunner(tmp_path / "config.json", start=False)

    def test_duplicate_active_job_is_reused(self, tmp_path):
        runner = self._runner(tmp_path)
        a = runner.enqueue("liat", "render", {})
        b = runner.enqueue("liat", "render", {"force": True})
        assert a.id == b.id  # double-click safe

    def test_option_whitelist(self, tmp_path):
        runner = self._runner(tmp_path)
        job = runner.enqueue("liat", "render", {"clips": ["a_to_b"], "evil": 1})
        assert job.options == {"clips": ["a_to_b"]}

    def test_active_ingest_orders(self, tmp_path):
        runner = self._runner(tmp_path)
        runner.enqueue("liat", "ingest", {"order": "folder-x"})
        assert runner.active_ingest_orders() == {"folder-x"}
