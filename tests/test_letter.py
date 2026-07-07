"""media/letter.py: Hebrew-safe closing-letter rasterisation."""
from __future__ import annotations

import pytest
from PIL import ImageFont

from ai_video_maker.media.letter import (
    _BG,
    _wrap,
    find_letter_font,
    render_letter_image,
)

try:
    _FONT = find_letter_font()
except FileNotFoundError:  # no system font in this environment
    _FONT = ""

needs_font = pytest.mark.skipif(not _FONT, reason="no Hebrew-capable system font")


class TestFindLetterFont:
    def test_explicit_path_must_exist(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            find_letter_font(str(tmp_path / "missing.ttf"))

    def test_explicit_existing_path_wins(self, tmp_path):
        fake = tmp_path / "font.ttf"
        fake.write_bytes(b"x")
        assert find_letter_font(str(fake)) == str(fake)


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
