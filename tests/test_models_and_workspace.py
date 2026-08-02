"""Storyboard model round-trips, project-name validation, plan coercion."""
from __future__ import annotations

import pytest

from ai_video_maker.clients.openai_client import OpenAIClient
from ai_video_maker.errors import InvalidProjectName, StoryboardError
from ai_video_maker.models import (
    Character,
    Frame,
    FramePerson,
    Storyboard,
    Transition,
    changed_transition_ids,
    identity_fingerprint,
    outdated_identity_plans,
    tagged_people,
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


class TestTaggedPeople:
    """Frame tags resolved into what the planner is actually told."""

    def _sb(self, people, cast=None):
        return Storyboard(
            project_title="t", style="s",
            characters=cast if cast is not None else [
                Character(id="dad", epithet="the bald man"),
                Character(id="son1", epithet="the smaller boy"),
            ],
            frames=[Frame(id="a", description="", image_prompt="",
                          output_path="styled_images/a.png", people=people)],
            transitions=[],
        )

    def test_people_come_back_left_to_right(self):
        # x order is the order the video model works in, so it decides the
        # listing regardless of how the markers were added.
        sb = self._sb([
            FramePerson(id="son1", x=0.8),
            FramePerson(id="dad", x=0.2),
        ])
        assert tagged_people(sb) == {
            "styled_images/a.png": ["the bald man", "the smaller boy"]
        }

    def test_an_untagged_frame_is_absent_not_empty(self):
        # "nobody has said" and "nobody is in this photo" must stay
        # distinguishable, or an untagged movie reads as a cast of nobody.
        assert tagged_people(self._sb([])) == {}

    def test_a_tag_naming_a_deleted_cast_member_is_dropped(self):
        sb = self._sb([FramePerson(id="ghost", x=0.5)])
        assert tagged_people(sb) == {}

    def test_frames_keep_their_tags_through_a_round_trip(self, tmp_path):
        sb = self._sb([FramePerson(id="dad", x=0.25, y=0.4)])
        path = tmp_path / "storyboard.json"
        sb.save(path)
        reloaded = Storyboard.load(path)
        assert reloaded.frames[0].people[0].id == "dad"
        assert reloaded.frames[0].people[0].x == 0.25

    def test_tagging_never_invalidates_a_rendered_clip(self):
        # Tags feed the NEXT plan; they don't change the prompt a clip was
        # rendered from, so they must not mark anything outdated.
        before = self._sb([])
        before.transitions = [Transition(
            id="a_to_b", start_frame="styled_images/a.png",
            end_frame="styled_images/b.png", motion_prompt="m",
            output_path="clips/a_to_b.mp4")]
        after = before.model_copy(deep=True)
        after.frames[0].people = [FramePerson(id="dad", x=0.5)]
        assert changed_transition_ids(before, after) == []


class TestOutdatedPlans:
    """"Which prompts don't know what I just told the app?"

    Tagging a photo and renaming a cast member are both plan-time inputs:
    they change nothing on their own. Without this, the only way to apply
    them was to re-plan everything and hope.
    """

    def _sb(self, people=None, epithet="the bald man"):
        return Storyboard(
            project_title="t", style="s",
            characters=[Character(id="dad", epithet=epithet)],
            frames=[
                Frame(id="a", description="", image_prompt="",
                      output_path="styled_images/a.png",
                      people=people or []),
                Frame(id="b", description="", image_prompt="",
                      output_path="styled_images/b.png"),
            ],
            transitions=[Transition(
                id="a_to_b", start_frame="styled_images/a.png",
                end_frame="styled_images/b.png", motion_prompt="m",
                output_path="clips/a_to_b.mp4")],
        )

    def _stamp(self, sb):
        sb.transitions[0].planned_identity = identity_fingerprint(
            sb, sb.transitions[0])
        return sb

    def test_a_freshly_planned_prompt_is_current(self):
        assert outdated_identity_plans(self._stamp(self._sb())) == []

    def test_tagging_a_photo_makes_its_pairs_outdated(self):
        sb = self._stamp(self._sb())
        sb.frames[0].people = [FramePerson(id="dad", x=0.4)]
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_renaming_a_cast_member_makes_their_pairs_outdated(self):
        sb = self._stamp(self._sb(people=[FramePerson(id="dad", x=0.4)]))
        sb.characters[0].epithet = "the man with grey hair"
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_moving_a_marker_across_another_matters(self):
        # Left-to-right order is what the video model maps region to region,
        # so a swap of the markers is a different plan.
        sb = Storyboard(
            project_title="t", style="s",
            characters=[Character(id="dad", epithet="the bald man"),
                        Character(id="son", epithet="the smaller boy")],
            frames=[
                Frame(id="a", description="", image_prompt="",
                      output_path="styled_images/a.png",
                      people=[FramePerson(id="dad", x=0.2),
                              FramePerson(id="son", x=0.8)]),
                Frame(id="b", description="", image_prompt="",
                      output_path="styled_images/b.png"),
            ],
            transitions=[Transition(
                id="a_to_b", start_frame="styled_images/a.png",
                end_frame="styled_images/b.png", motion_prompt="m",
                output_path="clips/a_to_b.mp4")],
        )
        self._stamp(sb)
        sb.frames[0].people = [FramePerson(id="dad", x=0.8),
                               FramePerson(id="son", x=0.2)]
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_an_untagged_project_never_reports_anything(self):
        # Nothing has been said about identity, so no prompt can be behind —
        # including every project that predates this bookkeeping (their
        # transitions carry no stamp at all).
        sb = self._sb()  # no stamp, no tags
        assert sb.transitions[0].planned_identity == ""
        assert outdated_identity_plans(sb) == []

    def test_an_unstamped_plan_with_tags_is_reported(self):
        # Planned before plans recorded their inputs, but the photos have
        # since been tagged: that prompt cannot have used them.
        sb = self._sb(people=[FramePerson(id="dad", x=0.4)])
        assert outdated_identity_plans(sb) == ["a_to_b"]

    def test_editing_a_motion_prompt_by_hand_is_not_an_identity_change(self):
        sb = self._stamp(self._sb(people=[FramePerson(id="dad", x=0.4)]))
        sb.transitions[0].motion_prompt = "something I wrote myself"
        assert outdated_identity_plans(sb) == []
