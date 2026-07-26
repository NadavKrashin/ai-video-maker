"""Admin API auth: token rules, throttling, and query-token scoping."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.errors import PipelineError
from ai_video_maker.server import (
    _AuthThrottle,
    _install_access_log_redaction,
    _RedactQueryStrings,
    create_app,
)

TOKEN = "a-perfectly-long-admin-token"


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch) -> TestClient:
    """create_app against a tmp config + tmp projects dir, watcher off."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
    }), encoding="utf-8")
    projects = tmp_path / "projects"
    projects.mkdir()
    monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
    monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
    monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)
    return TestClient(create_app(config_path, watch=False))


def _auth(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestTokenRules:
    def test_missing_token_refuses_to_boot(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ADMIN_API_TOKEN", raising=False)
        with pytest.raises(PipelineError, match="ADMIN_API_TOKEN is not set"):
            create_app(tmp_path / "config.json", watch=False)

    def test_short_token_refuses_to_boot(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ADMIN_API_TOKEN", "short")
        with pytest.raises(PipelineError, match="too short"):
            create_app(tmp_path / "config.json", watch=False)

    def test_health_is_open(self, app_client):
        assert app_client.get("/api/health").status_code == 200

    def test_docs_are_disabled(self, app_client):
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert app_client.get(path).status_code == 404, path

    def test_bearer_token_accepted(self, app_client):
        assert app_client.get("/api/jobs", headers=_auth()).status_code == 200

    def test_wrong_bearer_rejected(self, app_client):
        res = app_client.get("/api/jobs", headers=_auth("wrong-token-aaaaaaaa"))
        assert res.status_code == 401

    def test_query_token_rejected_on_api_routes(self, app_client):
        assert app_client.get(f"/api/jobs?token={TOKEN}").status_code == 401

    def test_query_token_accepted_on_media_route(self, app_client, tmp_path):
        clips = tmp_path / "projects" / "proj" / "clips"
        clips.mkdir(parents=True)
        (clips / "a_to_b.mp4").write_bytes(b"fake")
        res = app_client.get(
            f"/api/projects/proj/files/clips/a_to_b.mp4?token={TOKEN}"
        )
        assert res.status_code == 200

    def test_wrong_query_token_rejected_on_media_route(self, app_client, tmp_path):
        clips = tmp_path / "projects" / "proj" / "clips"
        clips.mkdir(parents=True)
        (clips / "a_to_b.mp4").write_bytes(b"fake")
        res = app_client.get(
            "/api/projects/proj/files/clips/a_to_b.mp4?token=wrong-token-aaaa"
        )
        assert res.status_code == 401

    def test_security_headers_present(self, app_client):
        res = app_client.get("/api/health")
        assert res.headers["X-Content-Type-Options"] == "nosniff"
        assert res.headers["X-Frame-Options"] == "DENY"
        assert res.headers["Cache-Control"] == "no-store"


class TestThrottle:
    def test_blocks_after_max_failures_within_window(self):
        throttle = _AuthThrottle(max_failures=3, window_seconds=900)
        for _ in range(3):
            assert not throttle.blocked("1.2.3.4")
            throttle.record_failure("1.2.3.4")
        assert throttle.blocked("1.2.3.4")
        assert not throttle.blocked("5.6.7.8")  # per-address, not global

    def test_success_clears_the_slate(self):
        throttle = _AuthThrottle(max_failures=2, window_seconds=900)
        throttle.record_failure("1.2.3.4")
        throttle.record_failure("1.2.3.4")
        assert throttle.blocked("1.2.3.4")
        throttle.clear("1.2.3.4")
        assert not throttle.blocked("1.2.3.4")

    def test_old_failures_expire(self, monkeypatch):
        throttle = _AuthThrottle(max_failures=2, window_seconds=10)
        clock = iter([0.0, 1.0, 100.0])  # third check is past the window
        monkeypatch.setattr("ai_video_maker.server.time.monotonic", lambda: next(clock))
        throttle.record_failure("1.2.3.4")
        throttle.record_failure("1.2.3.4")
        assert not throttle.blocked("1.2.3.4")

    def test_repeated_bad_tokens_get_429(self, app_client):
        for _ in range(10):  # the default throttle allows 10 failures
            assert app_client.get(
                "/api/jobs", headers=_auth("bad-token-aaaaaaaaaa")
            ).status_code == 401
        res = app_client.get("/api/jobs", headers=_auth("bad-token-aaaaaaaaaa"))
        assert res.status_code == 429
        # even the RIGHT token is locked out until the window passes
        assert app_client.get("/api/jobs", headers=_auth()).status_code == 429


class TestAccessLogRedaction:
    """The media route accepts ?token=, so the access log must never keep it.

    Every <img>/<video> request wrote the full ADMIN_API_TOKEN into
    serve.log in clear text — anyone able to read the file (or a backup)
    had admin access.
    """

    def _record(self, path: str) -> logging.LogRecord:
        rec = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s %s" %d',
            ("127.0.0.1:0", "GET", path, "HTTP/1.1", 200), None,
        )
        _RedactQueryStrings().filter(rec)
        return rec

    def test_token_query_is_stripped(self):
        rec = self._record("/api/projects/x/files/styled/a.png?token=SUPERSECRET")
        assert "SUPERSECRET" not in str(rec.args)
        assert "/api/projects/x/files/styled/a.png?<redacted>" in str(rec.args)

    def test_every_query_is_dropped_not_just_token(self):
        # Whole-query removal, so a future parameter cannot leak by omission.
        rec = self._record("/api/jobs?project=x&token=SECRET&other=1")
        assert "SECRET" not in str(rec.args) and "project=x" not in str(rec.args)

    def test_paths_without_a_query_are_untouched(self):
        rec = self._record("/api/health")
        assert rec.args[2] == "/api/health"

    def test_filter_is_installed_once(self):
        access = logging.getLogger("uvicorn.access")
        before = [f for f in access.filters if isinstance(f, _RedactQueryStrings)]
        _install_access_log_redaction()
        _install_access_log_redaction()
        after = [f for f in access.filters if isinstance(f, _RedactQueryStrings)]
        assert len(after) == max(1, len(before))
