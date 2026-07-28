"""Closing-letter rendering (Hebrew/RTL-safe).

The letter is rasterised with PIL instead of ffmpeg's drawtext: right-to-left
support in drawtext depends on how the local ffmpeg build was compiled, while
doing the bidi reordering here works everywhere. Hebrew needs no contextual
glyph shaping (unlike Arabic), so reordering each wrapped line to visual order
and drawing it left-to-right renders correctly with any Hebrew-capable font.

The output is one tall image; media/ffmpeg.py scrolls a screen-height window
down it (credits-style) to produce the video segment.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:  # python-bidi >= 0.5 exposes get_display at the top level
    from bidi import get_display
except ImportError:  # pragma: no cover - older python-bidi
    from bidi.algorithm import get_display

# Used in order when config.letter_font_path is empty; first existing wins.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Hebrew.ttc",
    "/System/Library/Fonts/Supplemental/NewPeninimMT.ttc",
    "/System/Library/Fonts/Supplemental/Raanana.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Near-black background + warm off-white text: readable on any display and
# consistent regardless of what the movie's last frame looked like.
_BG = (13, 13, 15)
_FG = (242, 237, 228)


def find_letter_font(explicit: str = "") -> str:
    """Resolve the letter font file: explicit config path, else best system font."""
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"letter_font_path not found: {explicit}")
        return explicit
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "No Hebrew-capable font found on this system; set letter_font_path "
        "in config.json to a .ttf/.ttc that includes Hebrew glyphs."
    )


def read_letter(path: Path) -> str:
    """The project's letter text, or "" when it hasn't been written yet.

    Unreadable bytes are treated as no letter rather than raising: the
    callers (status, the admin panel) are describing a project, and a
    corrupt file must not take the whole view down with it.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def letter_state(path: Path) -> dict:
    """Describe the letter file for status/API: is there text to scroll?

    ``exists`` is about the FILE, ``chars`` about its content — a file of
    blank lines renders nothing, so the two differ and both matter: the
    closing_letter toggle silently does nothing when ``chars`` is 0.
    """
    return {
        "exists": path.is_file(),
        "chars": len(read_letter(path).strip()),
    }


def save_letter(path: Path, text: str) -> dict:
    """Write (or clear) the project's letter; returns the new letter_state.

    Blank text DELETES the file instead of leaving an empty one behind, so
    "no letter" is a single state everywhere rather than two that behave the
    same but read differently in status.
    """
    if not text.strip():
        path.unlink(missing_ok=True)
        return letter_state(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise the line endings a browser textarea submits (CRLF): the
    # renderer splits on lines and a stray \r would be drawn as a glyph.
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(normalised.rstrip() + "\n", encoding="utf-8")
    return letter_state(path)


def _wrap(paragraph: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Word-wrap LOGICAL (unreordered) text to the given pixel width.

    Wrapping must happen before the bidi reorder so RTL lines break at the
    same words a reader would expect; each wrapped line is reordered
    separately afterwards.
    """
    words = paragraph.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if font.getlength(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def render_letter_image(
    text: str,
    width: int,
    screen_height: int,
    font_path: str,
    font_size: int,
    pad: bool = True,
    transparent: bool = False,
) -> Image.Image:
    """Rasterise the letter as one tall image ready to be scrolled.

    With ``pad`` a full blank screen is left above and below the text, so a
    crop-scroll starts empty, rolls the whole letter through, and ends empty.
    ``transparent`` renders text on an alpha-transparent canvas instead of the
    dark background — used when the letter is overlaid on the photo montage
    (the overlay animation provides the entry/exit, so pair it with
    ``pad=False``). Empty lines become paragraph gaps; lines are centred.
    """
    font = ImageFont.truetype(font_path, font_size)
    margin = int(width * 0.14)
    max_line_width = width - 2 * margin
    line_height = int(font_size * 1.6)
    gap_height = int(font_size * 0.9)

    display_lines: list[tuple[str, int]] = []  # (visual-order text, advance)
    for paragraph in text.splitlines():
        if not paragraph.strip():
            display_lines.append(("", gap_height))
            continue
        for line in _wrap(paragraph.strip(), font, max_line_width):
            display_lines.append((get_display(line), line_height))

    text_height = sum(advance for _, advance in display_lines)
    pad_height = screen_height if pad else 0
    total_height = pad_height + text_height + pad_height
    if transparent:
        img = Image.new("RGBA", (width, total_height), (0, 0, 0, 0))
    else:
        img = Image.new("RGB", (width, total_height), _BG)
    draw = ImageDraw.Draw(img)
    y = pad_height
    for line, advance in display_lines:
        if line:
            draw.text((width / 2, y), line, font=font, fill=_FG, anchor="ma")
        y += advance
    return img
