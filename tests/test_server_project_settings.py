"""Editing a project's tuning knobs from the panel.

These are the numbers whose right value only a watched render can settle, so
the point of the card is that trying one no longer needs a shell on the mini.
The point of the ALLOWLIST is that nothing else in config becomes editable
from a browser — most of it breaks renders in ways that are hard to diagnose
from there.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.config import Config
from ai_video_maker.project_settings import coerce_settings, setting_keys
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


def _put(client: TestClient, settings: dict):
    return client.put(
        f"/api/projects/{PROJECT}/settings", json={"settings": settings},
        headers=_auth(),
    )


def _saved() -> dict:
    path = Workspace.for_project(PROJECT).project_config_file
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _by_key(resp) -> dict:
    return {s["key"]: s for s in resp.json()["settings"]}


class TestTheAllowlistHolds:
    def test_a_known_knob_is_saved(self, client):
        assert _put(client, {"unstageable_mover_budget": 2}).status_code == 200
        assert _saved()["unstageable_mover_budget"] == 2

    def test_an_unknown_key_is_refused_not_ignored(self, client):
        # Silently dropping it would let the panel believe it saved something.
        resp = _put(client, {"style_prompt": "something else"})
        assert resp.status_code == 422
        assert "not an editable project setting" in resp.json()["detail"]
        assert _saved() == {}

    def test_a_dangerous_key_cannot_ride_along_with_a_valid_one(self, client):
        resp = _put(client, {
            "unstageable_mover_budget": 2, "fal_end_frame_field": "wrong"
        })
        assert resp.status_code == 422
        # Nothing partially applied.
        assert _saved() == {}

    def test_the_registry_and_the_config_agree(self):
        # A knob that is not a real config field would save and then be
        # ignored forever, which is worse than not offering it.
        for key in setting_keys():
            assert key in Config.model_fields


class TestValuesAreValidated:
    def test_a_negative_threshold_is_refused(self, client):
        assert _put(client, {"unstageable_mover_budget": -1}).status_code == 422

    def test_an_absurd_threshold_is_refused(self, client):
        assert _put(client, {"unstageable_crowd_limit": 500}).status_code == 422

    def test_a_string_where_a_number_belongs_is_refused(self, client):
        assert _put(client, {"unstageable_mover_budget": "two"}).status_code == 422

    def test_a_bool_is_not_silently_a_number(self, client):
        # True would otherwise be stored as 1 — a caller mistake worth naming.
        assert _put(client, {"unstageable_mover_budget": True}).status_code == 422

    def test_a_number_where_a_toggle_belongs_is_refused(self, client):
        assert _put(client, {"cast_references_enabled": 1}).status_code == 422

    def test_zero_is_a_legitimate_threshold(self, client):
        # "every pair with any mover gets the camera" is a real thing to try.
        assert _put(client, {"unstageable_mover_budget": 0}).status_code == 200
        assert _saved()["unstageable_mover_budget"] == 0

    def test_settings_must_be_an_object(self, client):
        resp = client.put(
            f"/api/projects/{PROJECT}/settings", json={"settings": [1, 2]},
            headers=_auth(),
        )
        assert resp.status_code == 422


class TestPinningAndUnpinning:
    def test_the_value_takes_effect_on_load(self, client, tmp_path):
        _put(client, {"unstageable_mover_budget": 2, "unstageable_crowd_limit": 3})
        ws = Workspace.for_project(PROJECT)
        config = Config.load(tmp_path / "config.json", ws.project_config_file)
        assert config.unstageable_mover_budget == 2
        assert config.unstageable_crowd_limit == 3

    def test_null_removes_the_override(self, client):
        _put(client, {"unstageable_mover_budget": 2})
        assert _put(client, {"unstageable_mover_budget": None}).status_code == 200
        assert "unstageable_mover_budget" not in _saved()

    def test_unpinning_the_last_key_deletes_the_file(self, client):
        _put(client, {"unstageable_mover_budget": 2})
        _put(client, {"unstageable_mover_budget": None})
        assert not Workspace.for_project(PROJECT).project_config_file.exists()

    def test_other_project_config_is_preserved(self, client):
        ws = Workspace.for_project(PROJECT)
        ws.project_config_file.write_text(
            json.dumps({"video_model": "kling-3-pro", "style_prompt": "mine"}),
            encoding="utf-8",
        )
        _put(client, {"unstageable_mover_budget": 2})
        saved = _saved()
        assert saved["video_model"] == "kling-3-pro"
        assert saved["style_prompt"] == "mine"
        assert saved["unstageable_mover_budget"] == 2

    def test_unpinning_leaves_a_file_that_still_has_other_keys(self, client):
        ws = Workspace.for_project(PROJECT)
        ws.project_config_file.write_text(
            json.dumps({"video_model": "kling-3-pro"}), encoding="utf-8"
        )
        _put(client, {"unstageable_mover_budget": 2})
        _put(client, {"unstageable_mover_budget": None})
        assert _saved() == {"video_model": "kling-3-pro"}

    def test_the_response_reports_what_is_pinned(self, client):
        by_key = _by_key(_put(client, {"unstageable_mover_budget": 2}))
        assert by_key["unstageable_mover_budget"]["pinned"] is True
        assert by_key["unstageable_mover_budget"]["value"] == 2
        # Untouched knobs report the inherited value and say it isn't pinned.
        assert by_key["unstageable_crowd_limit"]["pinned"] is False
        assert by_key["unstageable_crowd_limit"]["value"] == 4


class TestItIsSafeAndVisible:
    def test_it_renders_nothing(self, client, monkeypatch):
        enqueued = []
        monkeypatch.setattr(
            "ai_video_maker.server.JobRunner.enqueue",
            lambda self, *a, **k: enqueued.append(a),
        )
        _put(client, {"unstageable_mover_budget": 2})
        assert enqueued == []

    def test_it_needs_a_token(self, client):
        assert client.put(
            f"/api/projects/{PROJECT}/settings",
            json={"settings": {"unstageable_mover_budget": 2}},
        ).status_code in (401, 403)

    def test_an_unknown_project_is_a_404(self, client):
        assert client.put(
            "/api/projects/nope/settings",
            json={"settings": {"unstageable_mover_budget": 2}},
            headers=_auth(),
        ).status_code == 404

    def test_the_snapshot_carries_the_settings(self, client):
        _put(client, {"cast_references_enabled": False})
        body = client.get(f"/api/projects/{PROJECT}", headers=_auth()).json()
        by_key = {s["key"]: s for s in body["settings"]}
        assert by_key["cast_references_enabled"]["value"] is False
        assert by_key["cast_references_enabled"]["pinned"] is True

    def test_every_knob_explains_itself(self, client):
        body = client.get(f"/api/projects/{PROJECT}", headers=_auth()).json()
        for entry in body["settings"]:
            assert entry["label"] and entry["help"]
            # The panel tells the user what to run next; without it a changed
            # number looks like it did nothing.
            assert entry["applies"] in {"re-plan", "restyle", "render"}


class TestCoerceDirectly:
    def test_null_passes_through_as_an_unpin(self):
        assert coerce_settings({"unstageable_mover_budget": None}) == {
            "unstageable_mover_budget": None
        }

    def test_an_empty_patch_is_fine(self):
        assert coerce_settings({}) == {}

    def test_bools_survive(self):
        assert coerce_settings({"cast_references_enabled": False}) == {
            "cast_references_enabled": False
        }


class TestTheLookFromThePanel:
    """The Look switch (Storyboard step + settings card) through the real API."""

    def test_choosing_cartoon_is_saved_for_the_project(self, client):
        resp = _put(client, {"style_realism": "cartoon"})
        assert resp.status_code == 200
        assert _saved() == {"style_realism": "cartoon"}
        row = _by_key(resp)["style_realism"]
        assert row["value"] == "cartoon" and row["pinned"] is True

    def test_a_made_up_look_is_refused_and_nothing_is_written(self, client):
        resp = _put(client, {"style_realism": "photorealistic"})
        assert resp.status_code == 422
        assert _saved() == {}

    def test_the_project_then_loads_with_that_look(self, client, tmp_path):
        # The same load path every storyboard/restyle job goes through:
        # shared config, then this project's overrides on top.
        _put(client, {"style_realism": "cartoon"})
        ws = Workspace.for_project(PROJECT)
        config = Config.load(tmp_path / "config.json", ws.project_config_file)
        assert config.style_realism == "cartoon"

    def test_the_panel_gets_the_choices_to_render(self, client):
        rows = client.get(f"/api/projects/{PROJECT}", headers=_auth()).json()
        settings = (rows.get("snapshot") or rows).get("settings") or rows.get("settings")
        row = next(s for s in settings if s["key"] == "style_realism")
        assert row["kind"] == "choice"
        assert [c["value"] for c in row["choices"]] == ["cartoon", "lifelike"]
        assert row["value"] == "lifelike" and row["pinned"] is False
