"""Picking a video model through the admin API.

The endpoint is deliberately narrow — it writes exactly one key into one
project's config.json — because the alternative is opening the whole config
to writes from a browser. These tests pin the narrowness as much as the
feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.config import Config
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
    Workspace.for_project(PROJECT).mkdirs()
    return TestClient(create_app(config_path, watch=False))


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


def _put(client: TestClient, model: str):
    return client.put(
        f"/api/projects/{PROJECT}/video-model", json={"model": model},
        headers=_auth(),
    )


def _project_config() -> dict:
    path = Workspace.for_project(PROJECT).root / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


class TestListingTheModels:
    def test_the_registry_is_offered(self, client):
        body = client.get("/api/video-models", headers=_auth()).json()
        keys = [m["key"] for m in body["models"]]
        assert "kling-2.5-turbo-pro" in keys
        assert "kling-3-pro" in keys

    def test_each_entry_says_whether_it_takes_cast_faces(self, client):
        by_key = {
            m["key"]: m
            for m in client.get("/api/video-models", headers=_auth()).json()["models"]
        }
        assert by_key["kling-2.5-turbo-pro"]["supports_elements"] is False
        assert by_key["kling-3-pro"]["supports_elements"] is True

    def test_it_needs_a_token(self, client):
        assert client.get("/api/video-models").status_code in (401, 403)


class TestPinningAModel:
    def test_a_choice_is_written_to_the_project_config(self, client):
        assert _put(client, "kling-3-pro").status_code == 200
        assert _project_config()["video_model"] == "kling-3-pro"

    def test_the_choice_takes_effect_on_load(self, client, tmp_path):
        _put(client, "kling-3-pro")
        ws = Workspace.for_project(PROJECT)
        config = Config.load(tmp_path / "config.json", ws.root / "config.json")
        assert config.fal_start_frame_field == "start_image_url"
        assert config.fal_end_frame_field == "end_image_url"
        assert config.elements_enabled()

    def test_switching_back_to_two_five_stops_sending_faces(
        self, client, tmp_path
    ):
        _put(client, "kling-3-pro")
        _put(client, "kling-2.5-turbo-pro")
        ws = Workspace.for_project(PROJECT)
        config = Config.load(tmp_path / "config.json", ws.root / "config.json")
        assert not config.elements_enabled()
        assert config.fal_end_frame_field == "tail_image_url"

    def test_an_empty_choice_unpins_rather_than_writing_blank(self, client):
        _put(client, "kling-3-pro")
        assert _put(client, "").status_code == 200
        assert "video_model" not in _project_config()

    def test_an_unknown_model_is_refused_and_changes_nothing(self, client):
        _put(client, "kling-3-pro")
        resp = _put(client, "kling-99")
        assert resp.status_code == 422
        assert "kling-3-pro" in resp.json()["detail"]
        assert _project_config()["video_model"] == "kling-3-pro"

    def test_other_project_settings_are_preserved(self, client):
        ws = Workspace.for_project(PROJECT)
        (ws.root / "config.json").write_text(
            json.dumps({"style_prompt": "a pinned style"}), encoding="utf-8"
        )
        _put(client, "kling-3-pro")
        saved = _project_config()
        assert saved["style_prompt"] == "a pinned style"
        assert saved["video_model"] == "kling-3-pro"

    def test_derived_fields_are_never_frozen_into_the_file(self, client):
        # The preset's field names are resolved at load time. Writing them
        # out would mean a corrected preset could never reach a project that
        # had already been pinned.
        _put(client, "kling-3-pro")
        saved = _project_config()
        for derived in ("fal_model_id", "fal_start_frame_field",
                        "fal_end_frame_field", "fal_elements_field"):
            assert derived not in saved

    def test_a_stale_hand_written_field_is_cleared_on_switching(self, client):
        # Somebody had pinned fal_end_frame_field by hand for the old model.
        # Leaving it would silently break the new model's end frame.
        ws = Workspace.for_project(PROJECT)
        (ws.root / "config.json").write_text(
            json.dumps({"fal_end_frame_field": "tail_image_url"}), encoding="utf-8"
        )
        _put(client, "kling-3-pro")
        assert "fal_end_frame_field" not in _project_config()

    def test_it_needs_a_token(self, client):
        assert client.put(
            f"/api/projects/{PROJECT}/video-model", json={"model": "kling-3-pro"}
        ).status_code in (401, 403)

    def test_it_renders_nothing(self, client, monkeypatch):
        # Switching a model must never enqueue a job — the clips already made
        # keep the model they were made with and are flagged instead.
        enqueued = []
        monkeypatch.setattr(
            "ai_video_maker.server.JobRunner.enqueue",
            lambda self, *a, **k: enqueued.append(a),
        )
        _put(client, "kling-3-pro")
        assert enqueued == []
