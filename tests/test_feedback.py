"""Learning from rendered clips: the stores, and the loop around them.

The value of this feature is entirely in two guarantees:

* a note someone took the trouble to write is NEVER lost — not to a model
  call that failed, not to a typo'd clip id;
* a lesson, once learned, reaches every LATER planning call — including the
  targeted re-plan of a single pair, which is the one that used to forget
  everything the rest of the movie knew.

Both are pinned here, along with the "no lessons changes nothing" invariant:
a studio that has never given feedback must get byte-for-byte the prompt it
got before this feature existed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_maker.errors import PipelineError
from ai_video_maker.feedback import (
    SCOPE_MOTION,
    SCOPE_STYLE,
    FeedbackStore,
    LessonStore,
    lesson_prompt_block,
)
from ai_video_maker.models import Storyboard


def _storyboard(workspace, motion="the bald man walks in", duration=5,
                global_motion="") -> None:
    Storyboard(
        project_title="t", style="s", global_motion_prompt=global_motion,
        frames=[
            {"id": "a", "description": "", "image_prompt": "",
             "output_path": "styled_images/a.png"},
            {"id": "b", "description": "", "image_prompt": "",
             "output_path": "styled_images/b.png"},
        ],
        transitions=[{
            "id": "a_to_b", "start_frame": "styled_images/a.png",
            "end_frame": "styled_images/b.png", "motion_prompt": motion,
            "duration": duration, "output_path": "clips/a_to_b.mp4",
        }],
    ).save(workspace.default_storyboard_json)


@pytest.fixture
def lessons(tmp_path: Path) -> LessonStore:
    return LessonStore(tmp_path / "lessons.json")


class TestLessonStore:
    def test_a_lesson_survives_a_reload(self, tmp_path: Path):
        path = tmp_path / "lessons.json"
        added = LessonStore(path).add("Name the walk that gets them there.")
        reloaded = LessonStore(path).all()
        assert [l.id for l in reloaded] == [added.id]
        assert reloaded[0].scope == SCOPE_MOTION
        assert reloaded[0].active is True

    def test_blank_text_is_refused(self, lessons):
        with pytest.raises(ValueError):
            lessons.add("   ")

    def test_unknown_scopes_are_refused(self, lessons):
        with pytest.raises(ValueError):
            lessons.add("something", "colour-grading")

    def test_text_is_normalised_and_capped(self, lessons):
        lesson = lessons.add("  keep   it\n  short  ")
        assert lesson.text == "keep it short"
        assert len(lessons.add("x" * 900).text) == 400

    def test_disabling_keeps_the_lesson_but_stops_sending_it(self, lessons):
        lesson = lessons.add("Never let anyone teleport.")
        lessons.update(lesson.id, active=False)
        assert [l.id for l in lessons.all()] == [lesson.id]
        assert lessons.active(SCOPE_MOTION) == []

    def test_active_filters_by_scope(self, lessons):
        lessons.add("motion rule", SCOPE_MOTION)
        lessons.add("style rule", SCOPE_STYLE)
        assert [l.text for l in lessons.active(SCOPE_STYLE)] == ["style rule"]

    def test_updating_an_unknown_id_returns_none(self, lessons):
        assert lessons.update("nope", active=False) is None

    def test_forgetting_reports_whether_anything_went(self, lessons):
        lesson = lessons.add("a rule")
        assert lessons.remove(lesson.id) is True
        assert lessons.remove(lesson.id) is False

    def test_a_corrupt_file_reads_as_empty(self, tmp_path: Path):
        path = tmp_path / "lessons.json"
        path.write_text("{oops", encoding="utf-8")
        store = LessonStore(path)
        assert store.all() == []
        store.add("still works")
        assert len(store.all()) == 1


class TestLessonPromptBlock:
    def test_no_lessons_means_no_prompt_change_at_all(self):
        # The whole planner prompt is tuned text; a studio that has never
        # given feedback must get exactly what it got before learning existed.
        assert lesson_prompt_block([]) == ""
        assert lesson_prompt_block(["", "   "]) == ""

    def test_rules_are_listed_under_a_header_that_outranks_the_defaults(self):
        block = lesson_prompt_block(["Never let anyone teleport."])
        assert "- Never let anyone teleport." in block
        assert "the lesson wins" in block

    def test_style_lessons_get_their_own_header(self):
        block = lesson_prompt_block(["Never add hair."], SCOPE_STYLE)
        assert "STYLED FRAMES" in block


class TestRecordFeedback:
    def _pipeline(self, make_pipeline, tmp_path, **options):
        p = make_pipeline(**options)
        p.lessons = LessonStore(tmp_path / "lessons.json")
        return p

    def test_the_note_and_the_prompt_that_caused_it_are_saved(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace, motion="the bald man walks in",
                    global_motion="Two different children appear.")
        p = self._pipeline(
            make_pipeline, tmp_path,
            feedback_clip="a_to_b", feedback_note="he teleports",
            feedback_learn=False,
        )
        result = p.record_feedback()
        entry = result["feedback"]
        assert entry["note"] == "he teleports"
        assert entry["transition_id"] == "a_to_b"
        assert entry["verdict"] == "bad"
        # The prompt is copied at feedback time — including the global part
        # the renderer prepends — because the storyboard is edited later and
        # "the prompt this clip came from" is only knowable now.
        assert entry["motion_prompt"] == (
            "Two different children appear. the bald man walks in"
        )
        assert FeedbackStore(workspace.feedback_file).all()[0].note == "he teleports"

    def test_learning_off_never_calls_the_model(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he teleports", feedback_learn=False,
        )

        def explode(*a, **kw):
            raise AssertionError("distil must not be called with learning off")

        p.openai.distill_lesson = explode
        assert p.record_feedback()["lessons"] == []
        assert p.lessons.all() == []

    def test_a_distilled_rule_is_added_to_the_studio_wide_store(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides across the lawn without walking",
        )
        p.openai.distill_lesson = lambda *a, **kw: [
            {"scope": "motion", "text": "Give every position change a walk."}
        ]
        result = p.record_feedback()
        assert [l["text"] for l in result["lessons"]] == [
            "Give every position change a walk."
        ]
        stored = p.lessons.all()
        assert stored[0].origin == "feedback"
        assert stored[0].source["transition"] == "a_to_b"
        # The feedback entry points at what it taught, so a bad rule can be
        # traced back to the note that caused it.
        assert result["feedback"]["lesson_ids"] == [stored[0].id]

    def test_a_failed_distillation_still_keeps_the_note(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he teleports",
        )

        def boom(*a, **kw):
            raise RuntimeError("insufficient_quota")

        p.openai.distill_lesson = boom
        result = p.record_feedback()
        assert "insufficient_quota" in result["learn_error"]
        assert result["lessons"] == []
        assert FeedbackStore(workspace.feedback_file).all()[0].note == "he teleports"

    def test_nothing_general_enough_is_a_valid_answer(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="lovely",  feedback_verdict="good",
        )
        p.openai.distill_lesson = lambda *a, **kw: []
        result = p.record_feedback()
        assert result["lessons"] == []
        assert result["learn_error"] == ""
        assert FeedbackStore(workspace.feedback_file).all()[0].verdict == "good"

    def test_feedback_about_the_movie_needs_no_clip(
        self, make_pipeline, workspace, tmp_path
    ):
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_note="the movies feel rushed",
            feedback_learn=False,
        )
        entry = p.record_feedback()["feedback"]
        assert entry["transition_id"] == ""
        assert entry["motion_prompt"] == ""

    def test_an_unknown_clip_fails_loudly(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="x_to_y",
            feedback_note="broken", feedback_learn=False,
        )
        with pytest.raises(PipelineError, match="No transition"):
            p.record_feedback()

    def test_an_empty_note_is_refused(self, make_pipeline, tmp_path):
        p = self._pipeline(make_pipeline, tmp_path, feedback_note="  ")
        with pytest.raises(PipelineError, match="needs a note"):
            p.record_feedback()

    def test_an_unknown_verdict_is_refused(self, make_pipeline, tmp_path):
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_note="hm",
            feedback_verdict="meh",
        )
        with pytest.raises(PipelineError, match="verdict"):
            p.record_feedback()

    def test_a_dry_run_writes_nothing(self, make_pipeline, workspace, tmp_path):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, dry_run=True, feedback_clip="a_to_b",
            feedback_note="he teleports",
        )
        p.record_feedback()
        assert not workspace.feedback_file.exists()
        assert p.lessons.all() == []

    def test_the_snapshot_reports_what_people_said(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he teleports", feedback_learn=False,
        )
        p.record_feedback()
        feedback = p.snapshot()["feedback"]
        assert feedback["count"] == 1
        assert feedback["bad"] == 1
        assert feedback["by_transition"] == {"a_to_b": "bad"}


class TestClipReview:
    """The second witness: a vision call that actually watches the clip.

    A human note says which faults matter; the reviewer says precisely what
    happened. Both feed the rule, and the reviewer additionally proposes a
    fix for THIS clip — which is only ever RETURNED. Nothing here may edit
    the storyboard or re-render anything.
    """

    def _pipeline(self, make_pipeline, tmp_path, **options):
        p = make_pipeline(**options)
        p.lessons = LessonStore(tmp_path / "lessons.json")
        return p

    def _rendered(self, workspace, tid="a_to_b"):
        (workspace.clips_dir / f"{tid}.mp4").write_bytes(b"fake mp4")

    def _stub_review(self, pipeline, seen=None, **overrides):
        review = {
            "observed": "the man is in a different place without walking",
            "problems": ["subject slides instead of stepping"],
            "matches_prompt": False,
            "suggested_motion_prompt": "the bald man takes the last few steps",
            "suggested_duration": 5,
            "why": "names the walk",
            "changes_clip": True,
            **overrides,
        }

        def fake_review(frames, motion, duration, **kw):
            if seen is not None:
                seen.update({"frames": list(frames), "motion": motion,
                             "duration": duration, **kw})
            return review

        # Frame sampling is local ffmpeg work; every caller stubs it through
        # monkeypatch (never module-globally — that would leak into the rest
        # of the suite) so these tests need no binary and no real mp4.
        pipeline.openai.review_clip = fake_review
        return review

    def test_the_review_is_saved_with_the_note(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="looks weird", feedback_learn=False,
        )
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])
        self._stub_review(p)
        result = p.record_feedback()
        assert result["review"]["problems"] == ["subject slides instead of stepping"]
        entry = FeedbackStore(workspace.feedback_file).all()[0]
        assert entry.ai_observation.startswith("the man is in a different place")
        assert entry.suggested_motion_prompt == "the bald man takes the last few steps"
        assert entry.suggested_duration == 5
        # The human's own words are untouched by any of it.
        assert entry.note == "looks weird"

    def test_the_reviewer_sees_the_clip_the_prompt_and_the_note(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        _storyboard(workspace, motion="the bald man walks in", duration=10)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides", feedback_learn=False,
        )
        p.lessons.add("Give every position change a walk.")
        seen: dict = {}
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg"), Path("f2.jpg")])
        self._stub_review(p, seen)
        p.record_feedback()
        assert len(seen["frames"]) == 2
        assert seen["motion"] == "the bald man walks in"
        assert seen["duration"] == 10
        assert seen["user_note"] == "he slides"
        # Its rewrite has to obey the rules already learned.
        assert seen["lessons"] == ["Give every position change a walk."]
        assert seen["start_frame"].name == "a.png"
        assert seen["end_frame"].name == "b.png"

    def test_the_reviewer_is_told_which_part_is_prepended_automatically(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        # The clip really was rendered with global + per-clip text, so that
        # is what the reviewer is shown — but its rewrite goes back into the
        # PER-CLIP field, where the global part is prepended again at render
        # time. Without this it would be doubled in the next render.
        _storyboard(workspace, motion="the bald man walks in",
                    global_motion="Two different children appear.")
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides", feedback_learn=False,
        )
        seen: dict = {}
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])
        self._stub_review(p, seen)
        p.record_feedback()
        assert seen["motion"] == (
            "Two different children appear. the bald man walks in"
        )
        assert seen["global_context"] == "Two different children appear."

    def test_both_accounts_reach_the_rule_writer(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="looks weird",
        )
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])
        self._stub_review(p)
        distilled: dict = {}

        def fake_distill(note, **kw):
            distilled.update({"note": note, **kw})
            return [{"scope": "motion", "text": "Name the walk."}]

        p.openai.distill_lesson = fake_distill
        p.record_feedback()
        assert distilled["note"] == "looks weird"
        assert distilled["observation"].startswith("the man is in a different")
        assert distilled["problems"] == ["subject slides instead of stepping"]

    def test_a_failed_review_still_keeps_the_note(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides", feedback_learn=False,
        )
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])

        def boom(*a, **kw):
            raise RuntimeError("insufficient_quota")

        p.openai.review_clip = boom
        result = p.record_feedback()
        assert "insufficient_quota" in result["review_error"]
        assert result["review"] == {}
        assert FeedbackStore(workspace.feedback_file).all()[0].note == "he slides"

    def test_an_unrendered_clip_is_explained_not_an_error(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)  # no clip file on disk
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="the plan looks wrong", feedback_learn=False,
        )
        result = p.record_feedback()
        assert "not been rendered" in result["review_error"]
        assert FeedbackStore(workspace.feedback_file).all()[0].note

    def test_watching_can_be_switched_off(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides", feedback_learn=False,
            feedback_review=False,
        )

        def explode(*a, **kw):
            raise AssertionError("the clip must not be watched")

        p.openai.review_clip = explode
        assert p.record_feedback()["review"] == {}

    def test_a_clipless_note_is_never_reviewed(self, make_pipeline, tmp_path):
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_note="the movies feel rushed",
            feedback_learn=False,
        )

        def explode(*a, **kw):
            raise AssertionError("there is no clip to watch")

        p.openai.review_clip = explode
        assert p.record_feedback()["review"] == {}

    def test_watching_a_clip_needs_no_note(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        # "Just look at it and tell me what's wrong" is a real request.
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b", feedback_note="",
            feedback_learn=False,
        )
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])
        self._stub_review(p)
        result = p.record_feedback()
        assert result["review"]["problems"]
        assert FeedbackStore(workspace.feedback_file).all()[0].note == ""

    def test_a_note_without_a_clip_still_needs_words(
        self, make_pipeline, tmp_path
    ):
        p = self._pipeline(make_pipeline, tmp_path, feedback_note="  ")
        with pytest.raises(PipelineError, match="needs a note"):
            p.record_feedback()

    def test_a_dry_run_watches_nothing(
        self, make_pipeline, workspace, tmp_path
    ):
        _storyboard(workspace)
        self._rendered(workspace)
        p = self._pipeline(
            make_pipeline, tmp_path, dry_run=True, feedback_clip="a_to_b",
            feedback_note="he slides",
        )

        def explode(*a, **kw):
            raise AssertionError("a dry run must not spend a vision call")

        p.openai.review_clip = explode
        assert p.record_feedback()["review"] == {}

    def test_the_suggestion_never_touches_the_storyboard_or_the_clip(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        # The whole safety story of this feature: it proposes, the user
        # disposes. Nothing is edited, nothing is re-rendered, nothing is
        # even marked stale by giving feedback.
        _storyboard(workspace, motion="the bald man walks in", duration=10)
        self._rendered(workspace)
        before = workspace.default_storyboard_json.read_text(encoding="utf-8")
        p = self._pipeline(
            make_pipeline, tmp_path, feedback_clip="a_to_b",
            feedback_note="he slides", feedback_learn=False,
        )
        monkeypatch.setattr("ai_video_maker.runner.sample_clip_frames",
                            lambda *a, **kw: [Path("f1.jpg")])
        self._stub_review(p)
        p.record_feedback()
        assert workspace.default_storyboard_json.read_text(encoding="utf-8") == before
        assert (workspace.clips_dir / "a_to_b.mp4").read_bytes() == b"fake mp4"
        assert p.state.status("stale:a_to_b.mp4") is None


class TestLessonsReachThePlanner:
    def _planner(self, pipeline):
        """Stub the vision call and capture the lessons it was handed."""
        seen: dict = {}

        def fake_analyze(frames, style, default_duration=None,
                         global_context="", cast=None, lessons=None):
            seen["lessons"] = list(lessons or [])
            return [("planned", 5, "sound")] * (len(frames) - 1), list(cast or [])

        pipeline.openai.analyze_frame_transitions = fake_analyze
        return seen

    def _frames(self, workspace) -> list[Path]:
        out = []
        for name in ("a", "b"):
            path = workspace.styled_images_dir / f"{name}.png"
            path.write_bytes(b"x")
            out.append(path)
        return out

    def test_active_lessons_are_handed_to_every_plan(
        self, make_pipeline, workspace, tmp_path
    ):
        p = make_pipeline(analyze_frames=True)
        p.lessons = LessonStore(tmp_path / "lessons.json")
        p.lessons.add("Give every position change a walk.")
        p.lessons.add("Never add hair.", SCOPE_STYLE)  # other scope, other prompt
        seen = self._planner(p)
        p._plan_pairs(self._frames(workspace), [0], "style")
        assert seen["lessons"] == ["Give every position change a walk."]

    def test_disabled_lessons_are_not_sent(
        self, make_pipeline, workspace, tmp_path
    ):
        p = make_pipeline(analyze_frames=True)
        p.lessons = LessonStore(tmp_path / "lessons.json")
        lesson = p.lessons.add("A rule that turned out to be wrong.")
        p.lessons.update(lesson.id, active=False)
        seen = self._planner(p)
        p._plan_pairs(self._frames(workspace), [0], "style")
        assert seen["lessons"] == []

    def test_learning_can_be_switched_off_in_config(
        self, config, workspace, tmp_path
    ):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline

        config = config.model_copy(update={"learning_enabled": False})
        p = Pipeline(config, workspace, RunOptions(analyze_frames=True))
        p.lessons = LessonStore(tmp_path / "lessons.json")
        p.lessons.add("Give every position change a walk.")
        seen = self._planner(p)
        p._plan_pairs(self._frames(workspace), [0], "style")
        assert seen["lessons"] == []

    def test_only_the_newest_lessons_ride_along(
        self, config, workspace, tmp_path
    ):
        from ai_video_maker.options import RunOptions
        from ai_video_maker.runner import Pipeline

        config = config.model_copy(update={"max_lessons_in_prompt": 2})
        p = Pipeline(config, workspace, RunOptions(analyze_frames=True))
        p.lessons = LessonStore(tmp_path / "lessons.json")
        for text in ("oldest", "middle", "newest"):
            p.lessons.add(text)
        seen = self._planner(p)
        p._plan_pairs(self._frames(workspace), [0], "style")
        assert seen["lessons"] == ["middle", "newest"]

    def test_style_lessons_ride_with_the_style_prompt_only(
        self, make_pipeline, workspace, tmp_path, monkeypatch
    ):
        p = make_pipeline()
        p.lessons = LessonStore(tmp_path / "lessons.json")
        p.lessons.add("Never restore a bald man's hair.", SCOPE_STYLE)
        p.lessons.add("Give every position change a walk.", SCOPE_MOTION)
        src = workspace.input_images_dir / "a.jpg"
        src.write_bytes(b"x")
        prompts: list[str] = []
        p.openai.style_image = lambda s, prompt, dst: prompts.append(prompt)
        monkeypatch.setattr("ai_video_maker.runner.verify_dimensions",
                            lambda *a, **kw: True)
        p._style_images([src], {})
        assert "Never restore a bald man's hair." in prompts[0]
        assert "Give every position change a walk." not in prompts[0]
        assert prompts[0].startswith("test style")
