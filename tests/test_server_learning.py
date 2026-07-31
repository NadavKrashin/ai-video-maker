"""The admin API's learning + spending surface.

Covers the two things the panel depends on: feedback that reaches the
lessons store in one request (no job queue — a note must not wait behind a
20-minute render), and cost totals that a page can be built from.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.server import create_app
from ai_video_maker.workspace import Workspace

TOKEN = "a-perfectly-long-admin-token"
PROJECT = "demo"


def _storyboard() -> dict:
    return {
        "project_title": "demo", "style": "pixar",
        "frames": [
            {"id": "a", "description": "d", "image_prompt": "",
             "output_path": "styled_images/a.png"},
            {"id": "b", "description": "d", "image_prompt": "",
             "output_path": "styled_images/b.png"},
        ],
        "transitions": [{
            "id": "a_to_b", "start_frame": "styled_images/a.png",
            "end_frame": "styled_images/b.png",
            "motion_prompt": "the bald man walks to the door",
            "duration": 10, "sound_prompt": "footsteps",
            "output_path": "clips/a_to_b.mp4",
        }],
    }


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

    ws = Workspace.for_project(PROJECT)
    ws.mkdirs()
    ws.default_storyboard_json.write_text(
        json.dumps(_storyboard()), encoding="utf-8")
    (ws.styled_images_dir / "a.png").write_bytes(b"x")
    (ws.styled_images_dir / "b.png").write_bytes(b"x")
    (ws.clips_dir / "a_to_b.mp4").write_bytes(b"fake mp4")
    return TestClient(create_app(config_path, watch=False))


def _auth() -> dict:
    return {"Authorization": f"Bearer {TOKEN}"}


class TestFeedbackEndpoint:
    def test_a_note_is_recorded_and_distilled_in_one_request(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(
            "ai_video_maker.clients.openai_client.OpenAIClient.distill_lesson",
            lambda self, note, **kw: [
                {"scope": "motion", "text": "Give every position change a walk."}
            ],
        )
        res = client.post(
            f"/api/projects/{PROJECT}/feedback",
            json={"clip": "a_to_b", "note": "he slides across the lawn"},
            headers=_auth(),
        )
        assert res.status_code == 200
        body = res.json()
        assert body["feedback"]["transition_id"] == "a_to_b"
        assert [l["text"] for l in body["lessons"]] == [
            "Give every position change a walk."
        ]
        # ...and it is immediately in force for every future plan.
        lessons = client.get("/api/lessons", headers=_auth()).json()["lessons"]
        assert [l["text"] for l in lessons] == [
            "Give every position change a walk."
        ]

    def test_the_note_shows_up_in_the_project_snapshot(self, client):
        client.post(
            f"/api/projects/{PROJECT}/feedback",
            json={"clip": "a_to_b", "note": "he teleports", "learn": False},
            headers=_auth(),
        )
        snap = client.get(f"/api/projects/{PROJECT}", headers=_auth()).json()
        assert snap["feedback"]["by_transition"] == {"a_to_b": "bad"}
        listing = client.get(
            f"/api/projects/{PROJECT}/feedback", headers=_auth()).json()
        assert listing["feedback"][0]["note"] == "he teleports"

    def test_a_blank_note_is_refused(self, client):
        res = client.post(
            f"/api/projects/{PROJECT}/feedback", json={"note": "  "},
            headers=_auth(),
        )
        assert res.status_code == 400

    def test_an_unknown_verdict_is_refused(self, client):
        res = client.post(
            f"/api/projects/{PROJECT}/feedback",
            json={"note": "hm", "verdict": "meh"}, headers=_auth(),
        )
        assert res.status_code == 422

    def test_an_unknown_clip_is_a_client_error(self, client):
        res = client.post(
            f"/api/projects/{PROJECT}/feedback",
            json={"note": "hm", "clip": "x_to_y", "learn": False},
            headers=_auth(),
        )
        assert res.status_code == 400

    def test_feedback_needs_the_token(self, client):
        res = client.post(
            f"/api/projects/{PROJECT}/feedback", json={"note": "hm"})
        assert res.status_code == 401


class TestLessonEndpoints:
    def test_lessons_can_be_written_edited_and_forgotten_by_hand(self, client):
        added = client.post(
            "/api/lessons",
            json={"text": "Never let anyone teleport.", "scope": "motion"},
            headers=_auth(),
        ).json()["lesson"]

        off = client.patch(
            f"/api/lessons/{added['id']}", json={"active": False},
            headers=_auth(),
        ).json()["lesson"]
        assert off["active"] is False

        edited = client.patch(
            f"/api/lessons/{added['id']}", json={"text": "Reworded."},
            headers=_auth(),
        ).json()["lesson"]
        assert edited["text"] == "Reworded."

        assert client.delete(
            f"/api/lessons/{added['id']}", headers=_auth()).status_code == 200
        assert client.get(
            "/api/lessons", headers=_auth()).json()["lessons"] == []

    def test_a_blank_lesson_is_refused(self, client):
        res = client.post("/api/lessons", json={"text": " "}, headers=_auth())
        assert res.status_code == 422

    def test_an_unknown_scope_is_refused(self, client):
        res = client.post(
            "/api/lessons", json={"text": "x", "scope": "sound"},
            headers=_auth(),
        )
        assert res.status_code == 422

    def test_editing_a_missing_lesson_is_a_404(self, client):
        assert client.patch(
            "/api/lessons/nope", json={"active": False},
            headers=_auth()).status_code == 404
        assert client.delete(
            "/api/lessons/nope", headers=_auth()).status_code == 404

    def test_lesson_routes_need_the_token(self, client):
        assert client.get("/api/lessons").status_code == 401
        assert client.post("/api/lessons", json={"text": "x"}).status_code == 401


class TestCostEndpoints:
    def test_the_overview_totals_every_project(self, client):
        body = client.get("/api/costs", headers=_auth()).json()
        row = next(p for p in body["projects"] if p["project"] == PROJECT)
        # Two styled frames + one 10s clip, valued from the files on disk
        # because this project has no ledger yet.
        assert row["estimated_usd"] == 1.08
        assert row["estimated"] is True
        assert body["totals"]["estimated_usd"] == 1.08
        assert body["pricing"]["clip_usd_per_second"] == 0.07

    def test_the_learning_store_is_not_a_project(self, client):
        # projects/_learning holds studio-wide state; it must never appear as
        # a movie with no photos in it.
        client.post("/api/lessons", json={"text": "a rule"}, headers=_auth())
        names = [p["project"]
                 for p in client.get("/api/costs", headers=_auth()).json()["projects"]]
        assert names == [PROJECT]
        listed = client.get("/api/projects", headers=_auth()).json()["projects"]
        assert [p["project"] for p in listed] == [PROJECT]

    def test_a_projects_ledger_lists_the_individual_calls(self, client):
        ws = Workspace.for_project(PROJECT)
        ws.costs_file.parent.mkdir(parents=True, exist_ok=True)
        ws.costs_file.write_text(json.dumps({"entries": [
            {"at": "2026-07-30T10:00:00+00:00", "kind": "clip", "usd": 0.7,
             "detail": "a_to_b.mp4"},
        ]}), encoding="utf-8")
        body = client.get(
            f"/api/projects/{PROJECT}/costs", headers=_auth()).json()
        assert body["summary"]["total_usd"] == 0.7
        assert body["entries"][0]["detail"] == "a_to_b.mp4"
        assert "clip" in body["labels"]

    def test_cost_routes_need_the_token(self, client):
        assert client.get("/api/costs").status_code == 401
        assert client.get(f"/api/projects/{PROJECT}/costs").status_code == 401
