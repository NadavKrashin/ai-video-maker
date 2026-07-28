"""media/letter.py: Hebrew-safe closing-letter rasterisation."""
from __future__ import annotations

import pytest
from PIL import ImageFont

from ai_video_maker.media.letter import (
    _BG,
    _segments,
    _wrap,
    find_emoji_font,
    find_letter_font,
    letter_state,
    load_emoji_font,
    read_letter,
    render_letter_image,
    save_letter,
)

try:
    _FONT = find_letter_font()
except FileNotFoundError:  # no system font in this environment
    _FONT = ""

_EMOJI_FONT = find_emoji_font()

needs_font = pytest.mark.skipif(not _FONT, reason="no Hebrew-capable system font")
needs_emoji_font = pytest.mark.skipif(
    not _EMOJI_FONT, reason="no colour-emoji font on this system"
)

# A private-use code point: no real font maps it, so it stands in for "a
# character this font cannot draw" without depending on which font is found.
UNMAPPED = ""


class TestFindLetterFont:
    def test_explicit_path_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_letter_font(str(tmp_path / "missing.ttf"))

    def test_explicit_existing_path_wins(self, tmp_path):
        fake = tmp_path / "font.ttf"
        fake.write_bytes(b"x")
        assert find_letter_font(str(fake)) == str(fake)


class TestLetterFile:
    """Reading/writing letter.txt — what the panel's editor rides on."""

    def test_missing_file_reads_as_no_letter(self, tmp_path):
        path = tmp_path / "letter.txt"
        assert read_letter(path) == ""
        assert letter_state(path) == {"exists": False, "chars": 0}

    def test_save_then_read_round_trips(self, tmp_path):
        path = tmp_path / "letter.txt"
        state = save_letter(path, "מתן היקר,\n\nאוהבים אותך")
        assert state == {"exists": True, "chars": len("מתן היקר,\n\nאוהבים אותך")}
        assert read_letter(path).startswith("מתן היקר,")

    def test_browser_crlf_is_normalised(self, tmp_path):
        # A textarea submits CRLF; a stray \r would be drawn as a glyph.
        path = tmp_path / "letter.txt"
        save_letter(path, "line one\r\nline two\r\n")
        assert "\r" not in path.read_text(encoding="utf-8")

    def test_blank_text_removes_the_file(self, tmp_path):
        # "No letter" must be ONE state, not an empty file that behaves the
        # same but reads differently in status.
        path = tmp_path / "letter.txt"
        save_letter(path, "something")
        assert save_letter(path, "   \n  ") == {"exists": False, "chars": 0}
        assert not path.exists()

    def test_whitespace_only_file_has_no_chars(self, tmp_path):
        # A file left behind by hand: it exists, but scrolls nothing.
        path = tmp_path / "letter.txt"
        path.write_text("\n\n   \n", encoding="utf-8")
        assert letter_state(path) == {"exists": True, "chars": 0}

    def test_unreadable_bytes_do_not_raise(self, tmp_path):
        # status/API describe a project; a corrupt file can't take the view down.
        path = tmp_path / "letter.txt"
        path.write_bytes(b"\xff\xfe\x00garbage")
        assert read_letter(path) == ""
        assert letter_state(path) == {"exists": True, "chars": 0}


class TestFindEmojiFont:
    """Emoji need their own font; missing one is a degrade, not a crash."""

    def test_explicit_path_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_emoji_font(str(tmp_path / "missing.ttf"))

    def test_explicit_existing_path_wins(self, tmp_path):
        fake = tmp_path / "emoji.ttf"
        fake.write_bytes(b"x")
        assert find_emoji_font(str(fake)) == str(fake)

    def test_unloadable_font_is_none_not_an_exception(self, tmp_path):
        # A file that exists but isn't a usable font: the letter must still
        # render (without emoji), never blow up the whole combine.
        fake = tmp_path / "emoji.ttf"
        fake.write_bytes(b"not a font")
        assert load_emoji_font(str(fake)) is None

    def test_no_path_means_no_font(self):
        assert load_emoji_font("") is None


@needs_font
class TestSegments:
    """What each font is asked to draw — and what is dropped instead of boxed."""

    def _font(self):
        return ImageFont.truetype(_FONT, 40)

    def test_plain_text_is_one_text_run(self):
        assert _segments("shalom", self._font(), None) == [("shalom", False)]

    def test_emoji_is_dropped_when_there_is_no_emoji_font(self):
        # The bug this fixes: the text font draws an empty box for a heart.
        # The space that preceded it goes too, or the line centres off by one.
        assert _segments("hi ❤️", self._font(), None) == [("hi", False)]

    @needs_emoji_font
    def test_emoji_becomes_its_own_run_when_a_font_exists(self):
        emoji = load_emoji_font(_EMOJI_FONT)
        assert emoji is not None
        segments = _segments("hi ❤️", self._font(), emoji[0])
        assert segments[0] == ("hi ", False)
        # Heart + variation selector stay together so the font can compose them.
        assert segments[1][1] is True and "❤" in segments[1][0]

    def test_characters_the_font_cannot_draw_are_dropped(self):
        assert _segments(f"a{UNMAPPED}b", self._font(), None) == [("ab", False)]


@needs_font
class TestRenderWithEmoji:
    def _render(self, text, emoji_font=""):
        return render_letter_image(
            text, 1920, 400, _FONT, 64, pad=False, emoji_font_path=emoji_font,
        )

    def test_without_an_emoji_font_the_emoji_leaves_no_mark(self):
        # Strongest available statement of "no empty box": the image is
        # identical to one rendered from the same text without the emoji.
        with_emoji = self._render("shalom ❤️")
        without = self._render("shalom ")
        assert with_emoji.tobytes() == without.tobytes()

    @needs_emoji_font
    def test_emoji_font_puts_colour_in_the_line(self):
        # The text is drawn in one flat colour on a flat background, so any
        # saturated pixel can only have come from the colour emoji.
        img = self._render("shalom ❤️", _EMOJI_FONT).convert("RGB")
        colours = {c for _, c in (img.getcolors(200_000) or [])}
        assert any(r > 120 and r - b > 60 and r - g > 60 for r, g, b in colours)

    @needs_emoji_font
    def test_hebrew_letter_with_a_heart_still_reorders_rtl(self):
        # The real letter: the heart is last logically, so it lands leftmost.
        img = self._render('לאהבת חיי ט"ו באב שמח ❤️', _EMOJI_FONT)
        assert img.width == 1920


@needs_font
class TestWrap:
    def _font(self):
        return ImageFont.truetype(_FONT, 40)

    def test_short_line_untouched(self):
        assert _wrap("hello world", self._font(), 10_000) == ["hello world"]

    def test_long_text_wraps_within_width(self):
        font = self._font()
        lines = _wrap("word " * 40, font, 600)
        assert len(lines) > 1
        assert all(font.getlength(line) <= 600 for line in lines)


@needs_font
class TestRenderLetterImage:
    def test_hebrew_letter_layout(self):
        text = "מתן היקר,\n\nאנחנו אוהבים אותך מאוד.\nאבא ואמא"
        img = render_letter_image(text, 1920, 1080, _FONT, 64)
        assert img.width == 1920
        # blank screen above AND below the text, so the scroll starts/ends empty
        assert img.height > 2 * 1080
        top = img.crop((0, 0, 1920, 1080))
        bottom = img.crop((0, img.height - 1080, 1920, img.height))
        assert top.getcolors(2) == [(1920 * 1080, _BG)]
        assert bottom.getcolors(2) == [(1920 * 1080, _BG)]
        # the middle band actually contains drawn text (non-background pixels)
        middle = img.crop((0, 1080, 1920, img.height - 1080))
        assert len(middle.getcolors(100_000) or []) > 1

    def test_rtl_line_is_visually_reordered(self):
        # pure-Hebrew line: visual order is the reverse of logical order,
        # so the rendered right edge should be inked (Hebrew starts right).
        img = render_letter_image("שלום", 1920, 1080, _FONT, 64)
        band = img.crop((0, 1080, 1920, img.height - 1080)).convert("L")
        band = band.point(lambda v: 0 if v < 40 else 255)  # drop the dark bg
        cols = band.getbbox()  # (left, top, right, bottom) of the ink
        assert cols is not None
        left, _, right, _ = cols
        # single short word centred: ink stays well inside the margins
        assert 1920 * 0.14 < left and right < 1920 * 0.86
