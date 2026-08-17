"""The cast's canonical faces: cutting them, picking them, keeping them fresh.

A reference is derived data — cut for free out of frames the customer already
paid to style, using the face positions the panel's tagger records. So the
invariants these tests pin are mostly about what must NOT happen: no project
may behave differently because a reference is missing, unreadable, or points
at a frame that no longer shows that person.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_video_maker.clients.openai_client import _coerce_face_audit
from ai_video_maker.media.faces import (
    DEFAULT_CROP_FRACTION,
    annotate_prompt_with_elements,
    elements_for_clip,
    pick_reference_frame,
    reference_crop_box,
    references_for_frame,
    write_face_reference,
)
from ai_video_maker.models import (
    Character,
    Frame,
    FramePerson,
    Storyboard,
    outdated_reference_frames,
    reference_fingerprint,
)


def _styled(workspace, name: str, size=(1920, 1080)) -> Path:
    path = workspace.styled_images_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (40, 80, 120)).save(path)
    return path


def _storyboard(frames: list[Frame], characters: list[Character]) -> Storyboard:
    return Storyboard(
        project_title="t", style="s", frames=frames, characters=characters
    )


def _frame(name: str, people: list[FramePerson]) -> Frame:
    return Frame(
        id=name, description="", image_prompt="",
        output_path=f"styled_images/{name}", people=people,
    )


class TestTheCropBox:
    """Where the square lands, given a face centre and a crowd size."""

    def test_a_centred_face_gets_a_centred_square(self):
        left, top, right, bottom = reference_crop_box(1000, 1000, 0.5, 0.5)
        assert right - left == bottom - top
        assert (left + right) // 2 == 500
        assert (top + bottom) // 2 == 500

    def test_a_fuller_frame_gets_a_tighter_crop(self):
        # Faces shrink as a group grows, so the window has to shrink with it.
        solo = reference_crop_box(1920, 1080, 0.5, 0.5, people_in_frame=1)
        group = reference_crop_box(1920, 1080, 0.5, 0.5, people_in_frame=9)
        assert (group[2] - group[0]) < (solo[2] - solo[0])

    def test_the_crop_never_shrinks_to_mush(self):
        # A bad reference is worse than none: it teaches the wrong face with
        # full confidence, so there is a floor however big the crowd is.
        huge = reference_crop_box(1920, 1080, 0.5, 0.5, people_in_frame=400)
        assert (huge[2] - huge[0]) >= int(1080 * 0.16) - 1

    def test_a_face_at_the_edge_slides_the_box_instead_of_shrinking_it(self):
        # Clamping by trimming would hand back a thin strip of cheek.
        corner = reference_crop_box(1920, 1080, 0.0, 0.0)
        centre = reference_crop_box(1920, 1080, 0.5, 0.5)
        assert (corner[2] - corner[0]) == (centre[2] - centre[0])
        assert corner[0] == 0 and corner[1] == 0

    def test_the_box_always_lands_inside_the_image(self):
        for x, y in [(0.0, 0.0), (1.0, 1.0), (0.5, 1.0), (-3.0, 9.0)]:
            left, top, right, bottom = reference_crop_box(800, 600, x, y)
            assert 0 <= left < right <= 800
            assert 0 <= top < bottom <= 600

    def test_a_crop_bigger_than_the_frame_is_capped(self):
        left, top, right, bottom = reference_crop_box(
            100, 100, 0.5, 0.5, crop_fraction=4.0
        )
        assert (left, top, right, bottom) == (0, 0, 100, 100)


class TestCuttingAFace:
    def test_a_reference_is_written_and_bounded(self, workspace):
        src = _styled(workspace, "a.png")
        dst = workspace.cast_refs_dir / "c1.png"
        assert write_face_reference(src, dst, 0.5, 0.4)
        with Image.open(dst) as im:
            assert max(im.size) <= 512
            assert im.size[0] == im.size[1]

    def test_an_unreadable_frame_is_a_warning_not_a_crash(self, workspace):
        # References are an optimisation. A project whose frames cannot be
        # read must still style, plan and render exactly as it did before.
        bad = workspace.styled_images_dir / "broken.png"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"not an image")
        assert not write_face_reference(
            bad, workspace.cast_refs_dir / "c1.png", 0.5, 0.5
        )
        assert not (workspace.cast_refs_dir / "c1.png").exists()


class TestPickingTheFrameToCutFrom:
    def test_the_emptiest_frame_wins(self):
        # Fewest people = the biggest face and the least chance of catching a
        # neighbour's cheek in the crop.
        assert pick_reference_frame(
            [("crowd.png", 9), ("pair.png", 2), ("solo.png", 1)]
        ) == "solo.png"

    def test_ties_go_to_the_earliest_frame_so_the_answer_is_stable(self):
        assert pick_reference_frame(
            [("first.png", 3), ("second.png", 3)]
        ) == "first.png"

    def test_a_human_choice_wins_outright(self):
        assert pick_reference_frame(
            [("crowd.png", 9), ("solo.png", 1)], preferred="crowd.png"
        ) == "crowd.png"

    def test_a_stale_choice_falls_back_rather_than_losing_the_reference(self):
        # The frame was deleted, or they were untagged from it. Obeying a
        # dangling preference would leave this person with no reference at
        # all, which is strictly worse than an automatic pick.
        assert pick_reference_frame(
            [("solo.png", 1)], preferred="deleted.png"
        ) == "solo.png"

    def test_someone_who_appears_nowhere_gets_nothing(self):
        assert pick_reference_frame([]) is None


class TestBuildingTheSheet:
    """Pipeline.build_cast_references over a real (tiny) project."""

    def _project(self, pipeline, workspace):
        _styled(workspace, "a.png")
        _styled(workspace, "b.png")
        frames = [
            _frame("a.png", [FramePerson(id="c1", x=0.3, y=0.4)]),
            _frame("b.png", [
                FramePerson(id="c1", x=0.2, y=0.4),
                FramePerson(id="c2", x=0.7, y=0.4),
            ]),
        ]
        cast = [
            Character(id="c1", epithet="the bald man"),
            Character(id="c2", epithet="the taller boy"),
        ]
        return _storyboard(frames, cast)

    def test_every_tagged_cast_member_gets_a_face(self, pipeline, workspace):
        storyboard = self._project(pipeline, workspace)
        written = pipeline.build_cast_references(storyboard)
        assert set(written) == {"c1", "c2"}
        assert (workspace.cast_refs_dir / "c1.png").exists()
        assert (workspace.cast_refs_dir / "c2.png").exists()

    def test_the_emptiest_appearance_is_the_one_cut(self, pipeline, workspace):
        storyboard = self._project(pipeline, workspace)
        written = pipeline.build_cast_references(storyboard)
        assert written["c1"] == "styled_images/a.png"

    def test_an_untagged_cast_member_is_skipped_silently(
        self, pipeline, workspace
    ):
        storyboard = self._project(pipeline, workspace)
        storyboard.characters.append(Character(id="c3", epithet="the stranger"))
        written = pipeline.build_cast_references(storyboard)
        assert "c3" not in written
        assert not (workspace.cast_refs_dir / "c3.png").exists()

    def test_a_frame_with_no_styled_image_is_not_a_source(
        self, pipeline, workspace
    ):
        storyboard = self._project(pipeline, workspace)
        storyboard.frames.append(
            _frame("never_styled.png", [FramePerson(id="c4", x=0.5, y=0.5)])
        )
        storyboard.characters.append(Character(id="c4", epithet="the aunt"))
        assert "c4" not in pipeline.build_cast_references(storyboard)

    def test_references_for_departed_cast_members_are_dropped(
        self, pipeline, workspace
    ):
        # Otherwise the image model keeps being handed a person to match who
        # is no longer in the movie. Derived data, so deleting is free.
        storyboard = self._project(pipeline, workspace)
        pipeline.build_cast_references(storyboard)
        assert (workspace.cast_refs_dir / "c2.png").exists()

        storyboard.characters = [c for c in storyboard.characters if c.id != "c2"]
        for frame in storyboard.frames:
            frame.people = [p for p in frame.people if p.id != "c2"]
        pipeline.build_cast_references(storyboard)
        assert not (workspace.cast_refs_dir / "c2.png").exists()
        assert (workspace.cast_refs_dir / "c1.png").exists()

    def test_a_hand_picked_reference_frame_is_honoured(
        self, pipeline, workspace
    ):
        storyboard = self._project(pipeline, workspace)
        storyboard.characters[0].reference_frame = "styled_images/b.png"
        assert pipeline.build_cast_references(storyboard)["c1"] == (
            "styled_images/b.png"
        )

    def test_an_untagged_project_produces_nothing_at_all(
        self, pipeline, workspace
    ):
        # The silence rule: a movie nobody has tagged behaves exactly as it
        # did before any of this existed.
        _styled(workspace, "a.png")
        storyboard = _storyboard([_frame("a.png", [])], [])
        assert pipeline.build_cast_references(storyboard) == {}
        assert not workspace.cast_refs_dir.exists()

    def test_the_feature_can_be_switched_off(self, pipeline, workspace):
        storyboard = self._project(pipeline, workspace)
        pipeline.config.cast_references_enabled = False
        assert pipeline.build_cast_references(storyboard) == {}
        assert pipeline.cast_reference_paths(storyboard) == {}

    def test_a_dry_run_writes_nothing(self, make_pipeline, workspace):
        dry = make_pipeline(dry_run=True)
        storyboard = self._project(dry, workspace)
        assert dry.build_cast_references(storyboard) == {}
        assert not workspace.cast_refs_dir.exists()

    def test_reading_the_sheet_lists_only_this_cast(self, pipeline, workspace):
        storyboard = self._project(pipeline, workspace)
        pipeline.build_cast_references(storyboard)
        (workspace.cast_refs_dir / "leftover.png").write_bytes(b"x")
        assert set(pipeline.cast_reference_paths(storyboard)) == {"c1", "c2"}

    def test_reading_a_sheet_that_was_never_built_is_empty(self, pipeline):
        assert pipeline.cast_reference_paths() == {}


class TestWhichReferencesRideAlong:
    """references_for_frame: the one place that decides, called twice.

    Styling asks it what to attach; the staleness check asks it what SHOULD
    be attached. Two implementations would drift and every frame would report
    itself out of date forever, which is why this is a single function.
    """

    def test_only_people_with_a_reference_are_offered(self):
        assert references_for_frame(
            [("c1", 0.2), ("ghost", 0.8)], {"c1"}, {"c1": 3}, 4
        ) == ["c1"]

    def test_the_most_recurring_people_win_the_cap(self):
        # Consistency is worth most for the face the audience keeps seeing.
        people = [("rare", 0.1), ("often", 0.5), ("sometimes", 0.9)]
        assert references_for_frame(
            people, {"rare", "often", "sometimes"},
            {"rare": 1, "often": 20, "sometimes": 7}, 2,
        ) == ["often", "sometimes"]

    def test_ties_break_left_to_right_then_by_id(self):
        people = [("b", 0.9), ("a", 0.1)]
        assert references_for_frame(
            people, {"a", "b"}, {"a": 5, "b": 5}, 4
        ) == ["a", "b"]

    def test_the_cap_is_honoured(self):
        people = [(f"c{i}", i / 10) for i in range(9)]
        chosen = references_for_frame(
            people, {p for p, _ in people}, {}, 3
        )
        assert len(chosen) == 3

    def test_a_zero_cap_attaches_nothing(self):
        assert references_for_frame([("c1", 0.5)], {"c1"}, {"c1": 2}, 0) == []


class TestStylingIsAnchoredToTheCastFaces:
    def _tagged_project(self, pipeline, workspace):
        _styled(workspace, "a.png")
        storyboard = _storyboard(
            [_frame("a.png", [FramePerson(id="c1", x=0.4, y=0.4)])],
            [Character(id="c1", epithet="the bald man")],
        )
        pipeline.build_cast_references(storyboard)
        return storyboard

    def test_a_tagged_frame_plans_its_references(self, pipeline, workspace):
        storyboard = self._tagged_project(pipeline, workspace)
        plan = pipeline.style_reference_plan(storyboard)
        paths, stamp = plan["styled_images/a.png"]
        assert [p.name for p in paths] == ["c1.png"]
        assert stamp

    def test_an_untagged_project_plans_nothing(self, pipeline, workspace):
        _styled(workspace, "a.png")
        storyboard = _storyboard([_frame("a.png", [])], [])
        assert pipeline.style_reference_plan(storyboard) == {}

    def test_no_saved_storyboard_means_no_references(self, pipeline):
        # The first styling of a project: nothing is tagged yet, so this must
        # behave exactly as it did before the feature existed.
        assert pipeline.style_reference_plan(None) == {}

    def test_the_feature_switch_reaches_the_plan(self, pipeline, workspace):
        storyboard = self._tagged_project(pipeline, workspace)
        pipeline.config.cast_references_enabled = False
        assert pipeline.style_reference_plan(storyboard) == {}

    def test_the_references_are_handed_to_the_image_call(
        self, make_pipeline, workspace, monkeypatch
    ):
        pipeline = make_pipeline()
        storyboard = self._tagged_project(pipeline, workspace)
        seen: dict = {}

        def _fake(src, prompt, dst, references=None):
            seen["references"] = references
            dst.write_bytes(b"styled")

        pipeline.__dict__["openai"] = type(
            "F", (), {"style_image": staticmethod(_fake)}
        )()
        monkeypatch.setattr(
            "ai_video_maker.runner.verify_dimensions", lambda *a, **kw: True
        )
        source = workspace.input_images_dir / "a.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x")
        (workspace.styled_images_dir / "a.png").unlink()

        pipeline._style_images(
            [source], {}, None, pipeline.style_reference_plan(storyboard)
        )
        assert seen["references"] and seen["references"][0].name == "c1.png"

    def test_a_frame_styled_without_references_is_never_stamped(
        self, make_pipeline, workspace, monkeypatch
    ):
        pipeline = make_pipeline()
        monkeypatch.setattr(
            "ai_video_maker.runner.verify_dimensions", lambda *a, **kw: True
        )
        pipeline.__dict__["openai"] = type("F", (), {
            "style_image": staticmethod(
                lambda src, prompt, dst, references=None: dst.write_bytes(b"s")
            )
        })()
        source = workspace.input_images_dir / "z.jpg"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"x")
        pipeline._style_images([source], {})
        assert pipeline._styled_ref_stamps == {}


class TestOutdatedStylingIsVisibleButNeverFixed:
    """Re-styling costs image credits, so drift is a badge, not an action."""

    def _frame_with_stamp(self, stamp: str) -> Frame:
        frame = _frame("a.png", [FramePerson(id="c1", x=0.4, y=0.4)])
        frame.styled_refs = stamp
        return frame

    def test_a_frame_whose_references_moved_is_listed(self):
        sb = _storyboard([self._frame_with_stamp("old")], [])
        assert outdated_reference_frames(
            sb, {"styled_images/a.png": "new"}
        ) == ["styled_images/a.png"]

    def test_a_frame_whose_references_match_is_quiet(self):
        sb = _storyboard([self._frame_with_stamp("same")], [])
        assert outdated_reference_frames(sb, {"styled_images/a.png": "same"}) == []

    def test_a_frame_styled_before_references_existed_stays_quiet(self):
        # THE silence rule. Every frame of every project that predates this
        # carries an empty stamp, and an empty stamp is the truth ("styled
        # from no references"), not a demand to spend money re-styling.
        sb = _storyboard([self._frame_with_stamp("")], [])
        assert outdated_reference_frames(sb, {"styled_images/a.png": "new"}) == []

    def test_a_frame_the_plan_says_nothing_about_stays_quiet(self):
        # Everyone in it was untagged since, so there is no reference set to
        # disagree with — that is a tagging question, not a styling one.
        sb = _storyboard([self._frame_with_stamp("old")], [])
        assert outdated_reference_frames(sb, {}) == []

    def test_the_fingerprint_ignores_ordering(self):
        a = reference_fingerprint([("c1", "aaa"), ("c2", "bbb")])
        b = reference_fingerprint([("c2", "bbb"), ("c1", "aaa")])
        assert a == b

    def test_the_fingerprint_notices_a_recut_reference(self):
        before = reference_fingerprint([("c1", "aaa")])
        after = reference_fingerprint([("c1", "zzz")])
        assert before != after

    def test_the_fingerprint_notices_a_person_joining_the_frame(self):
        alone = reference_fingerprint([("c1", "aaa")])
        pair = reference_fingerprint([("c1", "aaa"), ("c2", "bbb")])
        assert alone != pair


class TestTheReferenceInstructionProtectsTheScene:
    """A reference must lend a FACE and nothing else.

    The references are frames of this same movie, full of plausible scenery,
    clothing and poses belonging to a different moment — and the edit
    endpoint will happily blend them in. These photographs are also taken
    years apart, so a reference that overruled the photograph on clothing
    would dress someone in an outfit they are not wearing in this scene.
    """

    def test_the_photograph_stays_the_authority(self):
        from ai_video_maker.clients.openai_client import (
            _STYLE_REFERENCE_INSTRUCTION as text,
        )
        assert "the photograph always" in text and "wins" in text
        assert "Never copy a reference's clothing" in text
        assert "never add a person who appears only in a reference" in text

    def test_identity_features_are_named(self):
        from ai_video_maker.clients.openai_client import (
            _STYLE_REFERENCE_INSTRUCTION as text,
        )
        for feature in ("hairline", "eyewear", "facial hair", "age band"):
            assert feature in text


class TestTheFaceAuditIsConservative:
    """_coerce_face_audit: what survives a model's answer.

    The list this produces invites somebody to spend an image credit per
    frame on it, so an unverifiable answer is dropped rather than repaired.
    """

    ALLOWED = {"c1": {"a.png", "b.png", "c.png"}, "c2": {"d.png", "e.png"}}

    def test_a_clean_finding_survives(self):
        out = _coerce_face_audit(
            {"characters": [
                {"id": "c1", "odd_labels": ["b.png"], "note": "hair added"}
            ]},
            self.ALLOWED,
        )
        assert out == {"c1": {"odd_frames": ["b.png"], "note": "hair added"}}

    def test_a_character_nobody_asked_about_is_dropped(self):
        assert _coerce_face_audit(
            {"characters": [{"id": "ghost", "odd_labels": ["x"], "note": "n"}]},
            self.ALLOWED,
        ) == {}

    def test_a_label_from_another_character_is_dropped(self):
        out = _coerce_face_audit(
            {"characters": [
                {"id": "c1", "odd_labels": ["d.png", "b.png"], "note": ""}
            ]},
            self.ALLOWED,
        )
        assert out["c1"]["odd_frames"] == ["b.png"]

    def test_duplicate_labels_collapse(self):
        out = _coerce_face_audit(
            {"characters": [
                {"id": "c1", "odd_labels": ["b.png", "b.png"], "note": ""}
            ]},
            self.ALLOWED,
        )
        assert out["c1"]["odd_frames"] == ["b.png"]

    def test_flagging_everything_is_not_a_finding(self):
        # "All three crops are the odd one out" says nothing about which
        # frame to redo — it means the character never looked consistent,
        # and the note is the useful half of that.
        out = _coerce_face_audit(
            {"characters": [{
                "id": "c1",
                "odd_labels": ["a.png", "b.png", "c.png"],
                "note": "inconsistent throughout",
            }]},
            self.ALLOWED,
        )
        assert out["c1"]["odd_frames"] == []
        assert out["c1"]["note"] == "inconsistent throughout"

    def test_a_clean_character_is_simply_absent(self):
        assert _coerce_face_audit(
            {"characters": [{"id": "c1", "odd_labels": [], "note": ""}]},
            self.ALLOWED,
        ) == {}

    def test_junk_is_survivable(self):
        assert _coerce_face_audit({}, self.ALLOWED) == {}
        assert _coerce_face_audit({"characters": ["nonsense"]}, self.ALLOWED) == {}
        assert _coerce_face_audit({"characters": None}, self.ALLOWED) == {}


class TestAuditPlumbing:
    def test_only_people_seen_twice_are_worth_comparing(
        self, pipeline, workspace, tmp_path
    ):
        # Consistency is a comparison; one appearance cannot disagree.
        _styled(workspace, "a.png")
        _styled(workspace, "b.png")
        storyboard = _storyboard(
            [
                _frame("a.png", [FramePerson(id="c1", x=0.3, y=0.4),
                                 FramePerson(id="c2", x=0.7, y=0.4)]),
                _frame("b.png", [FramePerson(id="c1", x=0.5, y=0.4)]),
            ],
            [Character(id="c1", epithet="the bald man"),
             Character(id="c2", epithet="the taller boy")],
        )
        groups = pipeline._face_audit_groups(storyboard, tmp_path)
        assert [cid for cid, _, _ in groups] == ["c1"]
        assert len(groups[0][2]) == 2

    def test_an_unstyled_frame_contributes_nothing(
        self, pipeline, workspace, tmp_path
    ):
        _styled(workspace, "a.png")
        storyboard = _storyboard(
            [
                _frame("a.png", [FramePerson(id="c1", x=0.3, y=0.4)]),
                _frame("gone.png", [FramePerson(id="c1", x=0.3, y=0.4)]),
            ],
            [Character(id="c1", epithet="the bald man")],
        )
        assert pipeline._face_audit_groups(storyboard, tmp_path) == []

    def test_a_missing_report_reads_as_empty(self, pipeline):
        assert pipeline.read_face_audit()["characters"] == {}

    def test_a_corrupt_report_reads_as_empty(self, pipeline, workspace):
        workspace.face_audit_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.face_audit_file.write_text("{not json", encoding="utf-8")
        assert pipeline.read_face_audit()["characters"] == {}

    def test_a_saved_report_round_trips(self, pipeline, workspace):
        workspace.face_audit_file.parent.mkdir(parents=True, exist_ok=True)
        workspace.face_audit_file.write_text(
            '{"checked": 2, "characters": {"c1": {"odd_frames": ["b.png"], '
            '"note": "n"}}, "ran_at": "now"}',
            encoding="utf-8",
        )
        report = pipeline.read_face_audit()
        assert report["checked"] == 2
        assert report["characters"]["c1"]["odd_frames"] == ["b.png"]


class TestWhichFacesArePinnedThroughAClip:
    """elements_for_clip: only people who must SURVIVE the transit."""

    APPEARANCES = {"a": 12, "b": 8, "c": 3, "d": 1}

    def test_only_people_in_both_frames_qualify(self):
        # Someone who leaves or arrives has no continuous identity to hold —
        # they are exactly who the pipeline stages an exit for, precisely so
        # the model does NOT try to carry them across.
        assert elements_for_clip(
            [("a", 0.2), ("c", 0.8)], [("a", 0.5)],
            {"a", "c"}, self.APPEARANCES, limit=3,
        ) == ["a"]

    def test_people_without_a_reference_are_skipped(self):
        assert elements_for_clip(
            [("a", 0.2)], [("a", 0.5)], set(), self.APPEARANCES, limit=3
        ) == []

    def test_a_zero_limit_switches_the_feature_off(self):
        assert elements_for_clip(
            [("a", 0.2)], [("a", 0.5)], {"a"}, self.APPEARANCES, limit=0
        ) == []

    def test_too_many_shared_people_sends_none_by_default(self):
        # One crisp face among three drifting ones can read worse than four
        # drifting together, so the conservative default declines.
        shared = [("a", 0.1), ("b", 0.3), ("c", 0.6)]
        assert elements_for_clip(
            shared, shared, {"a", "b", "c"}, self.APPEARANCES, limit=1
        ) == []

    def test_partial_mode_rescues_the_most_recurring_face(self):
        shared = [("a", 0.1), ("b", 0.3), ("c", 0.6)]
        assert elements_for_clip(
            shared, shared, {"a", "b", "c"}, self.APPEARANCES,
            limit=1, allow_partial=True,
        ) == ["a"]

    def test_exactly_at_the_limit_is_fine(self):
        shared = [("a", 0.1), ("b", 0.3)]
        assert set(elements_for_clip(
            shared, shared, {"a", "b"}, self.APPEARANCES, limit=2
        )) == {"a", "b"}


class TestTellingTheModelWhichPersonAnElementIs:
    def test_the_token_lands_after_the_epithet(self):
        out = annotate_prompt_with_elements(
            "the bald man waves as the taller boy runs past",
            ["the bald man", "the taller boy"],
            "@Element{index}",
        )
        assert out == (
            "the bald man (@Element1) waves as the taller boy (@Element2) runs past"
        )

    def test_only_the_first_mention_is_tagged(self):
        out = annotate_prompt_with_elements(
            "the bald man waves, then the bald man sits",
            ["the bald man"], "@Element{index}",
        )
        assert out.count("@Element1") == 1

    def test_an_epithet_the_prompt_never_uses_is_skipped(self):
        # A token pointing at nobody is worse than no token — and the camera
        # transitions deliberately name no one at all.
        prompt = "The camera moves smoothly and steadily across to the terrace"
        assert annotate_prompt_with_elements(
            prompt, ["the bald man"], "@Element{index}"
        ) == prompt

    def test_no_template_means_no_change(self):
        prompt = "the bald man waves"
        assert annotate_prompt_with_elements(prompt, ["the bald man"], "") == prompt


class TestElementsRideOnTheRequestOnlyWhenConfigured:
    # Kling 3 is the element-capable model; the default 2.5 is not, and the
    # class below pins that it stays that way.
    CAPABLE = dict(
        fal_model_id="fal-ai/kling-video/v3/pro/image-to-video",
        fal_elements_field="elements",
        fal_element_image_field="frontal_image_url",
        fal_max_elements=1,
    )

    def _client(self, config, **overrides):
        from ai_video_maker.clients.video import VideoClient
        for key, value in overrides.items():
            setattr(config, key, value)
        return VideoClient(config)

    def test_an_unconfigured_model_never_receives_elements(self, config):
        client = self._client(config)
        args = client._build_arguments("s", "e", "m", 5, ["http://face"])
        assert not any("element" in k.lower() for k in args)

    def test_a_configured_model_receives_them_in_its_own_shape(self, config):
        client = self._client(config, **self.CAPABLE)
        args = client._build_arguments("s", "e", "m", 5, ["http://face"])
        assert args["elements"] == [{"frontal_image_url": "http://face"}]

    def test_the_end_frame_still_rides_alongside_elements(self, config):
        # The contract this pipeline cannot give up: a clip lands exactly on
        # the next styled photo. Elements must never displace that.
        client = self._client(config, **self.CAPABLE)
        args = client._build_arguments("s", "e", "m", 5, ["http://face"])
        assert args[config.fal_end_frame_field] == "e"

    def test_elements_join_the_fingerprint_only_when_present(
        self, config, workspace
    ):
        client = self._client(config, **self.CAPABLE)
        start = _styled(workspace, "s.png", size=(8, 8))
        end = _styled(workspace, "e.png", size=(8, 8))
        face = workspace.cast_refs_dir / "c1.png"
        face.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), (1, 2, 3)).save(face)

        bare = client.fingerprint(start, end, "m", 5)
        # A render submitted before elements existed must stay resumable.
        assert client.fingerprint(start, end, "m", 5, []) == bare
        assert client.fingerprint(start, end, "m", 5, [face]) != bare


def test_the_default_crop_is_a_sane_fraction():
    assert 0.2 <= DEFAULT_CROP_FRACTION <= 1.0
