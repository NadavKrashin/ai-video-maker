"""StateStore / FailedJobStore behaviour."""
from __future__ import annotations

from ai_video_maker.state import FailedJobStore, StateStore


class TestStateStore:
    def test_set_and_query(self, tmp_path):
        store = StateStore(tmp_path / "state.json")
        store.set("clip:a.mp4", "done", output="x")
        assert store.is_done("clip:a.mp4")
        assert store.status("clip:a.mp4") == "done"
        assert store.status("unknown") is None

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "state.json"
        StateStore(path).set("a", "done")
        assert StateStore(path).is_done("a")

    def test_clear_removes_and_persists(self, tmp_path):
        path = tmp_path / "state.json"
        store = StateStore(path)
        store.set("sfx:a.mp4", "done")
        store.set("fade:a.mp4", "done")
        store.clear("sfx:a.mp4", "fade:a.mp4", "never-existed")
        assert not store.is_done("sfx:a.mp4")
        assert not StateStore(path).is_done("fade:a.mp4")

    def test_corrupt_file_starts_fresh(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{broken", encoding="utf-8")
        assert StateStore(path).status("x") is None


class TestFailedJobStore:
    def test_flush_writes_failures(self, tmp_path):
        path = tmp_path / "failed.json"
        store = FailedJobStore(path)
        store.record("clip:a", "clip", "boom", start="s.png")
        store.flush()
        assert path.exists()
        assert "boom" in path.read_text(encoding="utf-8")

    def test_clean_flush_removes_stale_report(self, tmp_path):
        path = tmp_path / "failed.json"
        path.write_text("[]", encoding="utf-8")
        FailedJobStore(path).flush()  # no failures this run
        assert not path.exists()

    def test_clean_flush_without_file_is_noop(self, tmp_path):
        FailedJobStore(tmp_path / "failed.json").flush()
