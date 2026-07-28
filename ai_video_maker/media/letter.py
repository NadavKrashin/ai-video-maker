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

import re
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from ..logging_setup import logger

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

# Colour-emoji fonts, in order; first existing wins when
# config.letter_emoji_font_path is empty. A customer's blessing routinely ends
# in a heart, and the Hebrew text font has no glyph for it — a real letter
# rendered "לאהבת חיי טו באב שמח" followed by two empty boxes.
_EMOJI_FONT_CANDIDATES = [
    "/System/Library/Fonts/Apple Color Emoji.ttc",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
    "/usr/share/fonts/google-noto-emoji/NotoColorEmoji.ttf",
    "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
]

# Bitmap emoji fonts only load at the sizes they ship strikes for (Noto Color
# Emoji: 109, Apple Color Emoji: 137/160), so the emoji is drawn at its native
# size and scaled down to sit with the text.
_EMOJI_STRIKE_SIZES = (109, 137, 160, 128, 96, 64, 32)

# Characters handed to the emoji font rather than the text font: the emoji
# blocks plus the joiners/selectors that belong to them (drawn together so the
# font can compose a sequence like a family or a coloured heart).
_EMOJI_RE = re.compile(
    "["
    "\U0001f000-\U0001faff"   # emoji blocks (symbols, faces, objects, flags)
    "☀-➿"           # misc symbols + dingbats (includes ❤ U+2764)
    "⬀-⯿"           # arrows/stars used as emoji
    "←-⇿⌀-⏿■-◿"  # arrows, technical, shapes
    "︎️‍"      # variation selectors + zero-width joiner
    "\U0001f3fb-\U0001f3ff"   # skin-tone modifiers
    "⃣"                  # combining keycap
    "]"
)

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


def find_emoji_font(explicit: str = "") -> str:
    """Resolve a colour-emoji font, or "" when the system has none.

    Unlike the text font this is never fatal: without it the emoji are
    dropped from the letter (see :func:`_segments`) so the movie still ends
    on clean text instead of a row of empty boxes.
    """
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(f"letter_emoji_font_path not found: {explicit}")
        return explicit
    for candidate in _EMOJI_FONT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return ""


def load_emoji_font(path: str) -> Optional[tuple[ImageFont.FreeTypeFont, int]]:
    """Load a colour-emoji font at a size it actually ships, with that size.

    Colour fonts are bitmap strikes: asking for an arbitrary pixel size
    raises OSError('invalid pixel size'), so the caller draws at the native
    size and scales the result.
    """
    if not path:
        return None
    for size in _EMOJI_STRIKE_SIZES:
        try:
            return ImageFont.truetype(path, size), size
        except OSError:
            continue
    logger.warning(
        "Emoji font %s could not be loaded at any known size; "
        "emoji will be dropped from the letter.", path,
    )
    return None


def _missing_glyph_signature(font: ImageFont.FreeTypeFont) -> bytes:
    """What this font draws for a character it has no glyph for.

    FreeType renders the .notdef box, and Pillow exposes no glyph-index API,
    so the box's own bitmap is the reference every character is compared
    against. \\uE000 is a private-use code point no real font maps.
    """
    return _glyph_signature(font, "")


def _glyph_signature(font: ImageFont.FreeTypeFont, ch: str) -> bytes:
    try:
        mask = font.getmask(ch, mode="L")
    except Exception:  # noqa: BLE001 - a font that can't measure can't draw
        return b""
    if not mask.size[0] or not mask.size[1]:
        return b""
    return Image.frombytes("L", mask.size, bytes(mask)).tobytes()


def _drawable(font: ImageFont.FreeTypeFont, ch: str, notdef: bytes) -> bool:
    """Does the font have a real glyph for this character?"""
    if ch.isspace():
        return True
    return _glyph_signature(font, ch) != notdef


def _segments(
    line: str,
    font: ImageFont.FreeTypeFont,
    emoji_font: Optional[ImageFont.FreeTypeFont],
) -> list[tuple[str, bool]]:
    """Split a line into (run, is_emoji) pieces the fonts can actually draw.

    Emoji runs are kept whole so the emoji font can compose sequences (a
    heart plus its variation selector, a joined family). Anything neither
    font can render is dropped rather than drawn as an empty box.
    """
    notdef = _missing_glyph_signature(font)
    out: list[tuple[str, bool]] = []
    for ch in line:
        is_emoji = bool(_EMOJI_RE.match(ch))
        if is_emoji and emoji_font is None:
            continue  # no emoji font on this machine: drop, never box
        if not is_emoji and not _drawable(font, ch, notdef):
            logger.warning(
                "Letter font cannot draw %r (U+%04X); dropping it.",
                ch, ord(ch),
            )
            continue
        if out and out[-1][1] == is_emoji:
            out[-1] = (out[-1][0] + ch, is_emoji)
        else:
            out.append((ch, is_emoji))
    # Dropping an emoji can leave the space that preceded it dangling at the
    # end of the line, which offsets the centring by a space width.
    if out and not out[0][1]:
        out[0] = (out[0][0].lstrip(), False)
    if out and not out[-1][1]:
        out[-1] = (out[-1][0].rstrip(), False)
    return [(run, is_emoji) for run, is_emoji in out if run]


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


def _emoji_size(font_size: int) -> int:
    """How tall an emoji is drawn: a little under the text's cap height."""
    return max(1, int(font_size * 0.95))


def _line_width(
    line: str,
    font: ImageFont.FreeTypeFont,
    emoji_font: Optional[ImageFont.FreeTypeFont],
    font_size: int,
) -> float:
    """Rendered width of a line, counting emoji at their drawn size."""
    box = _emoji_size(font_size)
    total = 0.0
    for run, is_emoji in _segments(line, font, emoji_font):
        total += len(run) * box if is_emoji else font.getlength(run)
    return total


def _wrap(
    paragraph: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    emoji_font: Optional[ImageFont.FreeTypeFont] = None,
    font_size: int = 0,
) -> list[str]:
    """Word-wrap LOGICAL (unreordered) text to the given pixel width.

    Wrapping must happen before the bidi reorder so RTL lines break at the
    same words a reader would expect; each wrapped line is reordered
    separately afterwards.
    """
    size = font_size or int(getattr(font, "size", 0) or 0)

    def width(text: str) -> float:
        return _line_width(text, font, emoji_font, size)

    words = paragraph.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        trial = f"{current} {word}"
        if width(trial) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_line(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    line: str,
    y: int,
    width: int,
    font: ImageFont.FreeTypeFont,
    emoji: Optional[tuple[ImageFont.FreeTypeFont, int]],
    font_size: int,
) -> None:
    """Draw one centred line, switching fonts per run.

    Text runs go through the normal text font; an emoji run is rendered by
    the colour font at its own native strike size and scaled down, because
    bitmap emoji fonts cannot be loaded at an arbitrary size.
    """
    emoji_font = emoji[0] if emoji else None
    segments = _segments(line, font, emoji_font)
    if not segments:
        return
    box = _emoji_size(font_size)
    x = (width - _line_width(line, font, emoji_font, font_size)) / 2
    for run, is_emoji in segments:
        if not is_emoji:
            draw.text((x, y), run, font=font, fill=_FG, anchor="la")
            x += font.getlength(run)
            continue
        strike = emoji[1]  # emoji is not None whenever a run is emoji
        tile = Image.new("RGBA", (strike * len(run) + strike, strike * 2), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text(
            (0, 0), run, font=emoji_font, embedded_color=True, anchor="la",
        )
        cropped = tile.crop(tile.getbbox() or (0, 0, strike, strike))
        scale = box / cropped.height
        scaled = cropped.resize(
            (max(1, int(cropped.width * scale)), box), Image.LANCZOS
        )
        # Sit the emoji on the text's baseline rather than the line's top.
        img.paste(scaled, (int(x), int(y + font_size * 0.15)), scaled)
        x += scaled.width


def render_letter_image(
    text: str,
    width: int,
    screen_height: int,
    font_path: str,
    font_size: int,
    pad: bool = True,
    transparent: bool = False,
    emoji_font_path: str = "",
) -> Image.Image:
    """Rasterise the letter as one tall image ready to be scrolled.

    With ``pad`` a full blank screen is left above and below the text, so a
    crop-scroll starts empty, rolls the whole letter through, and ends empty.
    ``transparent`` renders text on an alpha-transparent canvas instead of the
    dark background — used when the letter is overlaid on the photo montage
    (the overlay animation provides the entry/exit, so pair it with
    ``pad=False``). Empty lines become paragraph gaps; lines are centred.

    ``emoji_font_path`` supplies the colour font for emoji (a heart at the
    end of a blessing is the usual case). Without one the emoji are dropped:
    the text font draws them as empty boxes, which is what a real letter
    shipped with.
    """
    font = ImageFont.truetype(font_path, font_size)
    emoji = load_emoji_font(emoji_font_path)
    emoji_font = emoji[0] if emoji else None
    margin = int(width * 0.14)
    max_line_width = width - 2 * margin
    line_height = int(font_size * 1.6)
    gap_height = int(font_size * 0.9)

    display_lines: list[tuple[str, int]] = []  # (visual-order text, advance)
    for paragraph in text.splitlines():
        if not paragraph.strip():
            display_lines.append(("", gap_height))
            continue
        for line in _wrap(
            paragraph.strip(), font, max_line_width, emoji_font, font_size
        ):
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
            _draw_line(img, draw, line, y, width, font, emoji, font_size)
        y += advance
    return img
