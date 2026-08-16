"""Pure logic of Cloudinary order ingestion (no network)."""
from __future__ import annotations

import pytest

from ai_video_maker.clients.cloudinary_client import (
    OrderAsset,
    asset_position,
    ingest_filename,
    order_id_of,
    resolve_order_folder,
    sort_assets,
)
from ai_video_maker.errors import PipelineError


def _asset(public_id: str, position, created_at: str = "") -> OrderAsset:
    return OrderAsset(
        public_id=public_id, url=f"https://x/{public_id}", format="jpg",
        position=position, created_at=created_at,
    )


class TestAssetPosition:
    def test_context_order_wins(self):
        ctx = {"custom": {"order": "3", "from": "Dana"}}
        assert asset_position("video-orders/AM-1_Dana/9", ctx) == 3

    def test_flat_context_accepted(self):
        assert asset_position("whatever", {"order": "7"}) == 7

    def test_falls_back_to_public_id_trailing_number(self):
        assert asset_position("video-orders/AM-1_Dana/12", None) == 12
        assert asset_position("4", {}) == 4

    def test_non_numeric_context_falls_back(self):
        assert asset_position("video-orders/AM-1/5", {"custom": {"order": "n/a"}}) == 5

    def test_no_position_anywhere(self):
        assert asset_position("video-orders/AM-1/cover", None) is None


class TestSortAssets:
    def test_movie_order(self):
        out = sort_assets([_asset("b", 10), _asset("a", 2), _asset("c", 1)])
        assert [a.position for a in out] == [1, 2, 10]

    def test_unknown_positions_last_by_upload_time(self):
        out = sort_assets([
            _asset("late", None, "2026-07-02"),
            _asset("first", 1),
            _asset("early", None, "2026-07-01"),
        ])
        assert [a.public_id for a in out] == ["first", "early", "late"]


class TestResolveOrderFolder:
    FOLDERS = [
        "AM-20260716-XY12_Dana-Levi-16.07.2026_10-30",
        "AM-20260715-AB34_Noa-Cohen-15.07.2026_09-00",
        "AM-20260715-CD56_Noa-Katz-15.07.2026_11-00",
    ]

    def test_exact_match(self):
        assert resolve_order_folder(self.FOLDERS[0], self.FOLDERS) == self.FOLDERS[0]

    def test_unique_order_id_prefix(self):
        assert resolve_order_folder("AM-20260716-XY12", self.FOLDERS) == self.FOLDERS[0]

    def test_unique_substring_case_insensitive(self):
        assert resolve_order_folder("dana", self.FOLDERS) == self.FOLDERS[0]

    def test_ambiguous_lists_candidates(self):
        with pytest.raises(PipelineError, match="Noa-Cohen"):
            resolve_order_folder("AM-20260715", self.FOLDERS)

    def test_no_match(self):
        with pytest.raises(PipelineError, match="No Cloudinary order folder"):
            resolve_order_folder("AM-19990101-ZZ99", self.FOLDERS)

    def test_a_no_match_names_the_folders_that_do_exist(self):
        # The panel is often the only place this is read, and it has no shell
        # to go looking in — "run pipeline.py orders" is not an answer there.
        with pytest.raises(PipelineError) as caught:
            resolve_order_folder("AM-19990101-ZZ99", self.FOLDERS)
        assert "Dana-Levi" in str(caught.value)

    def test_no_folders_at_all_says_so(self):
        with pytest.raises(PipelineError, match="no order folders"):
            resolve_order_folder("AM-20260716-XY12_Dana", [])


class TestOrderIdSurvivesNameDrift:
    """The ledger's folder name and Cloudinary's can disagree mid-string.

    A real order (AM-160826-VKXQ) was recorded as ``..._00-45_מרגש`` while
    the photos sat in ``..._11-23_מרגש``: same order id, same customer, same
    date — a different TIME. Exact/startswith/contains all need the real
    folder to be at least as long as the query, so all three missed and the
    order could not be ingested at all.
    """

    DOC = "AM-160826-VKXQ_Kfir-Daniel-16.08.2026_00-45_mood"
    REAL = "AM-160826-VKXQ_Kfir-Daniel-16.08.2026_11-23_mood"
    OTHER = "AM-130826-PCFD_Yaron-Litan-13.08.2026_14-36_mood"

    def test_the_order_id_finds_the_folder_the_photos_are_in(self):
        assert resolve_order_folder(self.DOC, [self.OTHER, self.REAL]) == self.REAL

    def test_an_exact_name_still_wins_over_the_fallback(self):
        # Two folders share the order id; the exact name must not be dragged
        # into the ambiguity path.
        also = "AM-160826-VKXQ_Kfir-Daniel-16.08.2026_09-00_mood"
        assert resolve_order_folder(self.DOC, [self.DOC, also]) == self.DOC

    def test_two_folders_for_one_order_is_a_question_not_a_guess(self):
        also = "AM-160826-VKXQ_Kfir-Daniel-16.08.2026_09-00_mood"
        with pytest.raises(PipelineError) as caught:
            resolve_order_folder(self.DOC, [self.REAL, also])
        message = str(caught.value)
        assert self.REAL in message and also in message

    def test_a_different_order_is_never_substituted(self):
        with pytest.raises(PipelineError, match="No Cloudinary order folder"):
            resolve_order_folder(self.DOC, [self.OTHER])

    def test_order_id_is_the_head_of_the_leaf(self):
        assert order_id_of(self.DOC) == "AM-160826-VKXQ"
        assert order_id_of("no-underscores-here") == "no-underscores-here"


class TestIngestFilename:
    def test_zero_padded_min_two_digits(self):
        assert ingest_filename(1, 8, "jpg") == "01.jpg"
        assert ingest_filename(8, 8, "png") == "08.png"

    def test_width_grows_with_order_size(self):
        assert ingest_filename(3, 120, "jpg") == "003.jpg"

    def test_format_normalized_and_defaulted(self):
        assert ingest_filename(2, 9, ".JPG") == "02.jpg"
        assert ingest_filename(2, 9, "") == "02.jpg"
