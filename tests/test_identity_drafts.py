"""Unreviewed tag guesses are hints; reviewed tags are facts.

The automatic proposal at the end of `storyboard` used to be handed to the
planner as "a human has tagged this, it OVERRULES what you see". Nobody had
reviewed it. On a real growing-up movie (am-101126-unaq) it labelled the
main child as his younger brother in 14 photos, so the planner was ordered
to stage one boy as two people, and the user rewrote those prompts by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ai_video_maker.clients.openai_client import (
    OpenAIClient,
    _proposed_people_block,
    _tagged_people_block,
)
from ai_video_maker.models import (
    Character,
    FramePerson,
    Storyboard,
    hand_edited_transitions,
    identity_fingerprint,
    outdated_identity_plans,
    tagged_people,
)
from tests.test_runner_logic import _make_frame_pairs, _save_slug_storyboard


def _saved(workspace, names, draft):
    sb = _save_slug_storyboard(workspace, names)
    sb.characters = [Character(id="boy", epithet="the brown-haired boy")]
    for frame in sb.frames:
        frame.people = [FramePerson(id="boy", x=0.5, y=0.5)]
        frame.people_draft = draft
    sb.save(workspace.default_storyboard_json)
    return Storyboard.load(workspace.default_storyboard_json)


def _capture(p):
    seen: dict = {}

    def fake_analyze(frames, style, **kwargs):
        seen.update(kwargs)
        return [("m", 5, "s")] * (len(frames) - 1), list(kwargs.get("cast") or [])

    p.openai.analyze_frame_transitions = fake_analyze
    return seen


class TestGuessesAreHintsNotFacts:
    def test_the_draft_flag_round_trips_and_defaults_to_reviewed(self, workspace):
        sb = _saved(workspace, ["a", "b"], draft=True)
        assert all(f.people_draft for f in sb.frames)
        # Everything saved before the flag existed reads as reviewed, so no
        # existing project changes behaviour.
        raw = json.loads(workspace.default_storyboard_json.read_text())
        for f in raw["frames"]:
            f.pop("people_draft")
        assert not any(Storyboard(**raw).frames[i].people_draft for i in (0, 1))

    def test_a_guess_reaches_the_planner_only_as_a_hint(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=True, replan_all=True, tag_frames=False)
        seen = _capture(p)
        saved = _saved(workspace, ["a", "b"], draft=True)
        p._reconcile_storyboard(saved, _make_frame_pairs(workspace, ["a", "b"]))
        assert seen["frame_people"] == [None, None]  # nothing stated as fact
        assert seen["frame_people_hints"] == [
            ["the brown-haired boy"], ["the brown-haired boy"]
        ]

    def test_a_reviewed_tag_is_still_a_fact(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=True, replan_all=True, tag_frames=False)
        seen = _capture(p)
        saved = _saved(workspace, ["a", "b"], draft=False)
        p._reconcile_storyboard(saved, _make_frame_pairs(workspace, ["a", "b"]))
        assert seen["frame_people"] == [
            ["the brown-haired boy"], ["the brown-haired boy"]
        ]
        assert "frame_people_hints" not in seen  # the call is what it always was

    def test_the_flag_survives_a_storyboard_rebuild(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False, tag_frames=False)
        saved = _saved(workspace, ["a", "b"], draft=True)
        sb, _, _ = p._reconcile_storyboard(
            saved, _make_frame_pairs(workspace, ["a", "b"]))
        assert all(f.people_draft for f in sb.frames)

    def test_code_decisions_ignore_guesses(self, workspace):
        sb = _saved(workspace, ["a", "b"], draft=True)
        assert tagged_people(sb) != {}
        assert tagged_people(sb, include_drafts=False) == {}

    def test_the_hint_is_worded_as_a_guess(self):
        block = _proposed_people_block([["the boy"], None])
        assert "NOBODY HAS CHECKED IT" in block
        assert "TRUST THE PICTURES" in block
        assert "Frame 001" in block and "Frame 002" not in block
        assert "OVERRULES" not in block
        # ...while reviewed tags keep their authority.
        assert "OVERRULES" in _tagged_people_block([["the boy"], ["the boy"]])

    def test_no_hints_means_no_block(self):
        assert _proposed_people_block([None, None]) == ""
        assert _proposed_people_block([]) == ""

    def test_confirming_a_guess_marks_the_plan_as_behind(self, workspace):
        sb = _saved(workspace, ["a", "b"], draft=True)
        t = sb.transitions[0]
        t.planned_identity = identity_fingerprint(sb, t)
        assert outdated_identity_plans(sb) == []
        for f in sb.frames:
            f.people_draft = False
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_the_proposal_itself_is_recorded_as_a_guess(
        self, make_pipeline, workspace
    ):
        p = make_pipeline()
        sb = _saved(workspace, ["a", "b"], draft=False)
        for f in sb.frames:
            f.people = []
        _make_frame_pairs(workspace, ["a", "b"])
        p.openai.identify_people = lambda frames, cast: [
            [{"id": "boy", "x": 0.4, "y": 0.5}] for _ in frames
        ]
        assert p._propose_tags(sb) == 2
        assert all(f.people_draft for f in sb.frames)


class TestThePlannersOriginalIsKept:
    def test_a_fresh_plan_records_what_the_planner_wrote(
        self, make_pipeline, workspace
    ):
        p = make_pipeline(analyze_frames=False, motion_prompt="fresh",
                          replan_all=True, tag_frames=False)
        saved = _save_slug_storyboard(workspace, ["a", "b"])
        sb, _, _ = p._reconcile_storyboard(
            saved, _make_frame_pairs(workspace, ["a", "b"]))
        t = sb.transitions[0]
        assert (t.planned_prompt, t.planned_duration) == (t.motion_prompt, t.duration)
        assert hand_edited_transitions(sb) == []

    def test_a_hand_edit_is_visible_against_it(self, make_pipeline, workspace):
        p = make_pipeline(analyze_frames=False, motion_prompt="fresh",
                          replan_all=True, tag_frames=False)
        saved = _save_slug_storyboard(workspace, ["a", "b", "c"])
        sb, _, _ = p._reconcile_storyboard(
            saved, _make_frame_pairs(workspace, ["a", "b", "c"]))
        sb.transitions[0].motion_prompt = "the boy kicks the ball"
        sb.transitions[1].duration = 10
        assert hand_edited_transitions(sb) == ["a_to_b", "b_to_c"]
        sb.save(workspace.default_storyboard_json)
        snap = make_pipeline(analyze_frames=False).snapshot()["storyboard"]
        assert snap["hand_edited"] == ["a_to_b", "b_to_c"]

    def test_plans_from_before_this_existed_are_never_called_edited(self, workspace):
        sb = _save_slug_storyboard(workspace, ["a", "b"])
        assert hand_edited_transitions(sb) == []


class TestSavingReviewedTags:
    TOKEN = "a-perfectly-long-admin-token"

    @pytest.fixture
    def client(self, tmp_path: Path, monkeypatch, workspace):
        from ai_video_maker.server import create_app
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
        }), encoding="utf-8")
        monkeypatch.setattr("ai_video_maker.server.PROJECTS_DIR", workspace.root.parent)
        monkeypatch.setattr("ai_video_maker.workspace.PROJECTS_DIR", workspace.root.parent)
        monkeypatch.setenv("ADMIN_API_TOKEN", self.TOKEN)
        return TestClient(create_app(config_path, watch=False))

    def test_changing_a_guess_makes_it_reviewed(self, client, workspace):
        sb = _saved(workspace, ["a", "b"], draft=True)
        body = json.loads(sb.model_dump_json())
        body["frames"][0]["people"][0]["x"] = 0.2  # a human moved the marker
        resp = client.put(
            f"/api/projects/{workspace.root.name}/storyboard", json=body,
            headers={"Authorization": f"Bearer {self.TOKEN}"},
        )
        assert resp.status_code == 200
        after = Storyboard.load(workspace.default_storyboard_json)
        assert [f.people_draft for f in after.frames] == [False, True]


class TestClientAcceptsHints:
    def test_the_hint_block_reaches_the_model(self, config, monkeypatch, tmp_path):
        from PIL import Image
        frames = []
        for name in ("a", "b"):
            path = tmp_path / f"{name}.png"
            Image.new("RGB", (64, 36), "white").save(path)
            frames.append(path)
        sent = {}
        reply = json.dumps({"transitions": [{
            "pair_index": 1, "difficulty": 2, "start_heads": 1, "end_heads": 1,
            "same_place": False, "start_order": ["the boy"], "end_order": ["the boy"],
            "motion_prompt": "the boy turns", "sound_prompt": "wind",
        }], "characters": []})

        class _Completions:
            @staticmethod
            def create(**kwargs):
                sent.update(kwargs)
                msg = type("M", (), {"content": reply})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()], "usage": None})()

        fake = type("Cl", (), {"chat": type("Ch", (), {"completions": _Completions()})()})()
        client = OpenAIClient(config)
        monkeypatch.setattr(client, "_ensure_client", lambda: fake)
        client.analyze_frame_transitions(
            frames, "style", frame_people_hints=[["the boy"], None])
        text = " ".join(part["text"] for part in sent["messages"][1]["content"]
                        if part.get("type") == "text")
        assert "PROPOSED IDENTITIES" in text
        assert "CONFIRMED IDENTITIES" not in text
