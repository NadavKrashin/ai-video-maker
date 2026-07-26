"""Storyboard model round-trips, project-name validation, plan coercion."""
from __future__ import annotations

import pytest

from ai_video_maker.clients.openai_client import OpenAIClient
from ai_video_maker.errors import InvalidProjectName, StoryboardError
from ai_video_maker.models import (
    Character,
    Frame,
    Storyboard,
    Transition,
    changed_transition_ids,
)
from ai_video_maker.workspace import Workspace


class TestChangedTransitionIds:
    """Which hand-edits invalidate an already-rendered clip.

    Drives the panel's "save edits, then regenerate exactly those" flow: an
    edited motion prompt must mark its clip outdated, a sound-only edit must
    not (that is the audio step's business).
    """

    def _sb(self, **over) -> Storyboard:
        tr = dict(id="a_to_b", start_frame="styled_images/a.png",
                  end_frame="styled_images/b.png", motion_prompt="walks",
                  duration=5, sound_prompt="birds",
                  output_path="clips/a_to_b.mp4")
        tr.update(over.pop("transition", {}))
        return Storyboard(
            project_title="t", style="s",
            frames=[Frame(id="a", description="d", image_prompt="",
                          output_path="styled_images/a.png")],
            transitions=[Transition(**tr)],
            **over,
        )

    def test_no_change_marks_nothing(self):
        assert changed_transition_ids(self._sb(), self._sb()) == []

    def test_motion_prompt_edit_marks_the_clip(self):
        after = self._sb(transition={"motion_prompt": "runs"})
        assert changed_transition_ids(self._sb(), after) == ["a_to_b"]

    def test_duration_edit_marks_the_clip(self):
        after = self._sb(transition={"duration": 10})
        assert changed_transition_ids(self._sb(), after) == ["a_to_b"]

    def test_frame_swap_marks_the_clip(self):
        after = self._sb(transition={"end_frame": "styled_images/c.png"})
        assert changed_transition_ids(self._sb(), after) == ["a_to_b"]

    def test_sound_prompt_edit_leaves_the_video_alone(self):
        # SFX is muxed in by the audio step; the rendered video is unaffected.
        after = self._sb(transition={"sound_prompt": "waves"})
        assert changed_transition_ids(self._sb(), after) == []

    def test_global_motion_prompt_edit_marks_every_clip(self):
        # It is prepended to every clip's prompt at render time.
        after = self._sb(global_motion_prompt="two separate people")
        assert changed_transition_ids(self._sb(), after) == ["a_to_b"]

    def test_new_transition_is_not_marked(self):
        # Nothing was rendered from a pair that did not exist before.
        before = Storyboard(project_title="t", style="s", frames=[], transitions=[])
        assert changed_transition_ids(before, self._sb()) == []

    def test_cast_edit_leaves_rendered_clips_alone(self):
        # Unlike the global motion prompt, epithets are NOT prepended at
        # render time — they are baked into a motion prompt when the pair is
        # planned. An already-rendered clip therefore still matches its own
        # prompt; picking up a new epithet takes an explicit re-plan.
        after = self._sb(
            characters=[Character(id="bald-man", epithet="the bald man")]
        )
        assert changed_transition_ids(self._sb(), after) == []

    def test_no_previous_storyboard_marks_nothing(self):
        assert changed_transition_ids(None, self._sb()) == []


class TestStoryboardModel:
    def _sb(self) -> Storyboard:
        return Storyboard(
            project_title="t", style="s",
            frames=[Frame(id="001", description="d", image_prompt="p",
                          output_path="generated_frames/001.png")],
        )

    def test_save_load_roundtrip(self, tmp_path):
        path = tmp_path / "sb.json"
        self._sb().save(path)
        loaded = Storyboard.load(path)
        assert loaded.project_title == "t"
        assert loaded.frames[0].id == "001"

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(StoryboardError, match="not found"):
            Storyboard.load(tmp_path / "nope.json")

    def test_load_invalid_json_raises(self, tmp_path):
        path = tmp_path / "sb.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(StoryboardError, match="not valid JSON"):
            Storyboard.load(path)

    def test_load_wrong_shape_raises(self, tmp_path):
        path = tmp_path / "sb.json"
        path.write_text('{"project_title": "x"}', encoding="utf-8")
        with pytest.raises(StoryboardError, match="Invalid storyboard"):
            Storyboard.load(path)


class TestWorkspaceNames:
    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "/", "  "])
    def test_rejects_unsafe_names(self, bad):
        with pytest.raises(InvalidProjectName):
            Workspace.for_project(bad)

    def test_accepts_simple_name(self):
        ws = Workspace.for_project("my-film")
        assert ws.root.name == "my-film"


class TestCoerceTransitionPlans:
    def _client(self, config) -> OpenAIClient:
        return OpenAIClient(config)

    def test_fills_missing_and_malformed(self, config):
        client = self._client(config)
        data = {"transitions": [
            {"motion_prompt": "pan", "difficulty": 5, "sound_prompt": "wind"},
            {"motion_prompt": "", "difficulty": None},  # blank prompt, unrated
        ]}
        plans = client._coerce_transition_plans(data, count=3, default_duration=None)
        assert plans[0] == ("pan", 10, "wind")
        assert plans[1] == (config.motion_prompt, config.duration, "")
        assert plans[2] == (config.motion_prompt, config.duration, "")

    def test_default_duration_overrides_all(self, config):
        client = self._client(config)
        data = {"transitions": [{"motion_prompt": "x", "difficulty": 5}]}
        plans = client._coerce_transition_plans(data, count=1, default_duration=5)
        assert plans[0][1] == 5

    @pytest.mark.parametrize(
        "value,expected", [(5, 5), (10, 10), ("10", 10), (7, 5), (None, 5), ("x", 5)]
    )
    def test_coerce_duration(self, config, value, expected):
        assert self._client(config)._coerce_duration(value, 5) == expected
