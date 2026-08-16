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

from ai_video_maker.media.faces import (
    DEFAULT_CROP_FRACTION,
    pick_reference_frame,
    reference_crop_box,
    write_face_reference,
)
from ai_video_maker.models import Character, Frame, FramePerson, Storyboard


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


def test_the_default_crop_is_a_sane_fraction():
    assert 0.2 <= DEFAULT_CROP_FRACTION <= 1.0
