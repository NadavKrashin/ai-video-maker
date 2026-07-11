"""RunOptions.from_args: the CLI -> options seam."""
from __future__ import annotations

from ai_video_maker.cli import build_parser
from ai_video_maker.options import RunOptions


def _combine_options(*argv: str) -> RunOptions:
    args = build_parser().parse_args(["combine", "proj", *argv])
    return RunOptions.from_args(args)


class TestFinalFlag:
    """--final = --intro --credits-photos in one flag."""

    def test_final_turns_on_intro_and_credits(self):
        opts = _combine_options("--final")
        assert opts.intro_clip is True
        assert opts.credits_photos is True
        assert opts.closing_letter is None  # letter is not part of the package

    def test_explicit_no_flag_beats_final(self):
        opts = _combine_options("--final", "--no-credits-photos")
        assert opts.credits_photos is False
        assert opts.intro_clip is True

    def test_without_final_flags_stay_unset(self):
        opts = _combine_options()
        assert opts.intro_clip is None
        assert opts.credits_photos is None
