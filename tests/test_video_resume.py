"""VideoClient queue-flow resume: a submitted fal job is money already spent,
so its request_id is persisted and an interrupted render is recovered instead
of re-billed. No network — FalSession and download_file are stubbed."""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_maker.clients import video as video_mod
from ai_video_maker.clients.video import VideoClient
from ai_video_maker.state import StateStore


class FakeFal:
    """Stands in for FalSession: records submits/waits, scripted results."""

    def __init__(self) -> None:
        self.submits: list[tuple[str, str]] = []  # (request_id, prompt)
        self.waits: list[str] = []
        # request_id -> result dict, or an Exception to raise from the wait.
        self.results: dict[str, object] = {}
        self._next = 0

    def upload(self, path: Path, *, description: str) -> str:
        return f"https://fake.fal/{Path(path).name}"

    def submit(self, model_id: str, arguments: dict, *, description: str) -> str:
        request_id = f"req-{self._next}"
        self._next += 1
        self.submits.append((request_id, arguments["prompt"]))
        return request_id

    def wait_for_result(
        self, model_id: str, request_id: str, *, description: str
    ) -> dict:
        self.waits.append(request_id)
        out = self.results.get(
            request_id, {"video": {"url": f"https://fake.fal/{request_id}.mp4"}}
        )
        if isinstance(out, Exception):
            raise out
        return out


@pytest.fixture
def clip_env(config, tmp_path, monkeypatch):
    start = tmp_path / "a.png"
    start.write_bytes(b"frame-a")
    end = tmp_path / "b.png"
    end.write_bytes(b"frame-b")
    dst = tmp_path / "a_to_b.mp4"
    state = StateStore(tmp_path / "state.json")
    client = VideoClient(config, state=state)
    client.fal = FakeFal()
    downloads: list[str] = []

    def fake_download(url, dst_path, **_kw):
        downloads.append(url)
        Path(dst_path).write_bytes(b"mp4")

    monkeypatch.setattr(video_mod, "download_file", fake_download)
    return client, state, start, end, dst, downloads


JOB_KEY = "falreq:a_to_b.mp4"


class TestQueueFlow:
    def test_success_submits_once_and_clears_the_receipt(self, clip_env):
        client, state, start, end, dst, downloads = clip_env
        client.generate_clip(start, end, "walk", 5, dst)
        assert [p for _, p in client.fal.submits] == ["walk"]
        assert downloads and dst.exists()
        assert state.get(JOB_KEY) is None  # receipt cleared after download

    def test_interrupted_wait_keeps_the_request_id(self, clip_env):
        client, state, start, end, dst, _ = clip_env
        client.fal.results["req-0"] = ConnectionError("network died mid-wait")
        with pytest.raises(ConnectionError):
            client.generate_clip(start, end, "walk", 5, dst)
        entry = state.get(JOB_KEY)
        assert entry is not None and entry["request_id"] == "req-0"
        assert entry["status"] == "pending"

    def test_resume_fetches_pending_job_without_resubmitting(self, clip_env):
        client, state, start, end, dst, downloads = clip_env
        fp = client._fingerprint(start, end, "walk", 5)
        state.set(JOB_KEY, "pending", request_id="req-old", fingerprint=fp)
        client.generate_clip(start, end, "walk", 5, dst)
        assert client.fal.submits == []  # nothing re-bought
        assert client.fal.waits == ["req-old"]
        assert downloads == ["https://fake.fal/req-old.mp4"] and dst.exists()
        assert state.get(JOB_KEY) is None

    def test_stale_fingerprint_is_dropped_and_job_resubmitted(self, clip_env):
        client, state, start, end, dst, _ = clip_env
        state.set(JOB_KEY, "pending", request_id="req-old", fingerprint="outdated")
        client.generate_clip(start, end, "walk", 5, dst)
        # The pending job rendered an old plan: never resumed, fresh submit.
        assert "req-old" not in client.fal.waits
        assert len(client.fal.submits) == 1
        assert state.get(JOB_KEY) is None

    def test_unrecoverable_pending_job_falls_back_to_fresh_submit(self, clip_env):
        client, state, start, end, dst, _ = clip_env
        fp = client._fingerprint(start, end, "walk", 5)
        state.set(JOB_KEY, "pending", request_id="req-old", fingerprint=fp)
        client.fal.results["req-old"] = RuntimeError("request expired")
        client.generate_clip(start, end, "walk", 5, dst)
        assert client.fal.waits[0] == "req-old"  # tried the paid job first
        assert len(client.fal.submits) == 1 and dst.exists()

    def test_moderation_failure_clears_the_receipt(self, clip_env):
        client, state, start, end, dst, _ = clip_env
        client.fal.results["req-0"] = RuntimeError("content_policy_violation")
        with pytest.raises(RuntimeError):
            client.generate_clip(start, end, "walk", 5, dst)
        # A rejected job has no recoverable output — nothing to resume.
        assert state.get(JOB_KEY) is None

    def test_reword_recovery_resubmits_with_new_prompt(self, clip_env):
        client, state, start, end, dst, _ = clip_env
        client.fal.results["req-0"] = RuntimeError("content_policy_violation")
        client.generate_clip(
            start, end, "risky wording", 5, dst, reword=lambda p: "safe wording"
        )
        assert [p for _, p in client.fal.submits] == [
            "risky wording", "safe wording",
        ]
        assert dst.exists() and state.get(JOB_KEY) is None

    def test_works_without_a_state_store(self, config, clip_env):
        _, _, start, end, dst, _ = clip_env
        client = VideoClient(config, state=None)
        client.fal = FakeFal()
        client.generate_clip(start, end, "walk", 5, dst)
        assert dst.exists()


class TestFingerprint:
    def test_sensitive_to_every_input(self, config, tmp_path):
        a = tmp_path / "a.png"
        a.write_bytes(b"one")
        b = tmp_path / "b.png"
        b.write_bytes(b"two")
        client = VideoClient(config)
        base = client._fingerprint(a, b, "walk", 5)
        assert client._fingerprint(a, b, "walk", 10) != base
        assert client._fingerprint(a, b, "run", 5) != base
        assert client._fingerprint(b, a, "walk", 5) != base

    def test_stable_for_same_inputs(self, config, tmp_path):
        a = tmp_path / "a.png"
        a.write_bytes(b"one")
        b = tmp_path / "b.png"
        b.write_bytes(b"two")
        assert VideoClient(config)._fingerprint(a, b, "walk", 5) == \
            VideoClient(config)._fingerprint(a, b, "walk", 5)
