"""The job list survives a restart, and a restart is refused under a job.

Jobs lived only in memory: a deploy or crash erased the queue and the log of
whatever was running. What is pinned here is as much what does NOT happen —
nothing interrupted is ever re-run on its own, because re-running spends
money and a restart is not a person deciding to spend it.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.server import Job, JobRunner, create_app

REPO = Path(__file__).resolve().parents[1]
TOKEN = "t" * 32


def _runner(tmp_path: Path) -> JobRunner:
    return JobRunner(
        tmp_path / "config.json", start=False,
        history_path=tmp_path / "state" / "jobs.json",
    )


def _history(tmp_path: Path) -> list[dict]:
    return json.loads((tmp_path / "state" / "jobs.json").read_text())


class TestHistoryIsWritten:
    def test_a_queued_job_is_on_disk_before_it_runs(self, tmp_path):
        job = _runner(tmp_path).enqueue("matan", "render", {"clips": ["a_to_b"]})
        (record,) = _history(tmp_path)
        assert record["id"] == job.id and record["state"] == "queued"
        assert record["options"] == {"clips": ["a_to_b"]}

    def test_cancelling_is_recorded(self, tmp_path):
        runner = _runner(tmp_path)
        job = runner.enqueue("matan", "render", {})
        runner.cancel(job.id)
        assert _history(tmp_path)[0]["state"] == "cancelled"

    def test_without_a_history_path_nothing_is_written(self, tmp_path):
        JobRunner(tmp_path / "config.json", start=False).enqueue(
            "matan", "render", {})
        assert list(tmp_path.iterdir()) == []

    def test_an_unwritable_history_never_costs_the_job(self, tmp_path):
        blocker = tmp_path / "state"
        blocker.write_text("a file where the folder should be")
        job = _runner(tmp_path).enqueue("matan", "render", {})
        assert job.state == "queued"


class TestRestartRestoresButNeverReruns:
    def _write(self, tmp_path: Path, records: list[dict]) -> None:
        path = tmp_path / "state" / "jobs.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(records))

    def _record(self, **over) -> dict:
        job = Job(id="j1", project="matan", command="render",
                  options={"clips": ["a_to_b"]})
        job.log = ["line one", "line two"]
        return {**job.to_record(), **over}

    def test_a_job_that_was_running_comes_back_interrupted(self, tmp_path):
        self._write(tmp_path, [self._record(
            state="running", started_at="2026-11-11T10:00:00")])
        runner = _runner(tmp_path)
        job = runner.get("j1")
        assert job.state == "interrupted"
        assert "restarted while this job was running" in job.error
        assert "not" in job.error.lower() and "again" in job.error
        # Its log is still there — that is the point of keeping history.
        assert job.log == ["line one", "line two"]
        # ...and the corrected state is written back.
        assert _history(tmp_path)[0]["state"] == "interrupted"

    def test_a_job_still_in_the_queue_is_interrupted_too(self, tmp_path):
        self._write(tmp_path, [self._record(state="queued")])
        job = _runner(tmp_path).get("j1")
        assert job.state == "interrupted"
        assert "waiting in the queue" in job.error

    def test_nothing_interrupted_is_put_back_on_the_queue(self, tmp_path):
        self._write(tmp_path, [
            self._record(state="running"),
            self._record(id="j2", state="queued"),
        ])
        runner = _runner(tmp_path)
        assert runner._queue.empty()
        assert runner.active() == []

    def test_finished_jobs_come_back_as_they_were(self, tmp_path):
        self._write(tmp_path, [self._record(state="done"),
                               self._record(id="j2", state="partial",
                                            error="2 clips failed", failures=2)])
        runner = _runner(tmp_path)
        assert runner.get("j1").state == "done"
        assert runner.get("j2").error == "2 clips failed"

    def test_an_interrupted_job_does_not_block_running_it_again(self, tmp_path):
        self._write(tmp_path, [self._record(state="running")])
        runner = _runner(tmp_path)
        again = runner.enqueue("matan", "render", {"clips": ["a_to_b"]})
        assert again.id != "j1" and again.state == "queued"

    @pytest.mark.parametrize("content", ["{not json", '{"a": 1}', "[1, 2, 3]", ""])
    def test_a_corrupt_history_never_stops_the_server(self, tmp_path, content):
        path = tmp_path / "state" / "jobs.json"
        path.parent.mkdir(parents=True)
        path.write_text(content)
        runner = _runner(tmp_path)
        assert runner.list() == []
        assert runner.enqueue("matan", "render", {}).state == "queued"

    def test_only_the_newest_jobs_and_log_lines_are_kept(self, tmp_path):
        from ai_video_maker.server import _HISTORY_LOG_LINES
        runner = _runner(tmp_path)
        job = runner.enqueue("matan", "render", {})
        job.log = [f"line {n}" for n in range(_HISTORY_LOG_LINES + 50)]
        runner._persist()
        saved = _history(tmp_path)[0]["log"]
        assert len(saved) == _HISTORY_LOG_LINES
        assert saved[-1] == f"line {_HISTORY_LOG_LINES + 49}"


class TestHealthSaysWhetherTheServerIsBusy:
    @pytest.fixture
    def app(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "style_prompt": "s", "scratch_style_prompt": "s",
            "motion_prompt": "m",
        }))
        projects = tmp_path / "projects"
        projects.mkdir()
        monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", projects)
        monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", projects)
        monkeypatch.setenv("ADMIN_API_TOKEN", TOKEN)
        # Capture instead of run: enqueue would otherwise start a real job.
        monkeypatch.setattr(JobRunner, "_loop", lambda self: None)
        return create_app(config_path, watch=False), projects

    def test_idle_then_busy_without_naming_the_work(self, app):
        application, projects = app
        client = TestClient(application)
        assert client.get("/api/health").json() == {
            "ok": True, "busy": False, "active_jobs": 0}
        (projects / "matan").mkdir()
        resp = client.post(
            "/api/projects/matan/actions/render", json={},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert resp.status_code == 200, resp.text
        body = client.get("/api/health").json()
        assert body == {"ok": True, "busy": True, "active_jobs": 1}
        # Unauthenticated, so it must say nothing about WHAT is running.
        assert "matan" not in json.dumps(body)

    def test_the_history_lives_beside_the_lessons(self, app):
        application, projects = app
        (projects / "matan").mkdir()
        TestClient(application).post(
            "/api/projects/matan/actions/render", json={},
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        assert (projects / "_learning" / "jobs.json").exists()


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
class TestDeployRefusesToRestartUnderAJob:
    """deploy.sh, run for real against a fake `curl` and a fake `git`."""

    def _run(self, tmp_path: Path, health: str, **env):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        touched = tmp_path / "git-was-called"
        (bin_dir / "curl").write_text(f"#!/bin/sh\necho '{health}'\n")
        (bin_dir / "git").write_text(f"#!/bin/sh\ntouch '{touched}'\nexit 1\n")
        for tool in ("curl", "git"):
            path = bin_dir / tool
            path.chmod(path.stat().st_mode | stat.S_IEXEC)
        repo = tmp_path / "repo"
        repo.mkdir()
        result = subprocess.run(
            ["bash", str(REPO / "deploy" / "deploy.sh")],
            env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
                 "REPO_DIR": str(repo), **env},
            capture_output=True, text=True,
        )
        return result, touched.exists()

    def test_a_busy_server_stops_the_deploy_before_anything_changes(self, tmp_path):
        result, git_called = self._run(
            tmp_path, '{"ok":true,"busy":true,"active_jobs":1}')
        assert result.returncode == 1
        assert "refusing to restart" in result.stderr
        assert not git_called

    def test_an_idle_server_lets_the_deploy_proceed(self, tmp_path):
        _, git_called = self._run(
            tmp_path, '{"ok":true,"busy":false,"active_jobs":0}')
        assert git_called

    def test_an_unreachable_server_is_not_busy(self, tmp_path):
        _, git_called = self._run(tmp_path, "")
        assert git_called

    def test_force_overrides_the_refusal(self, tmp_path):
        _, git_called = self._run(
            tmp_path, '{"ok":true,"busy":true,"active_jobs":1}',
            FORCE_DEPLOY="1")
        assert git_called
