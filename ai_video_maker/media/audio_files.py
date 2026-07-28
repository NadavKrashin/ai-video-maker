"""Recognising an uploaded audio file by its CONTENT, not its name.

The music-bed upload used to accept a file only when its FILENAME ended in a
known audio extension. That broke a real upload from a phone: iOS handed the
browser a track named ``Lights(chosic.com)`` — no ``.mp3`` on the end — and
the server rejected a perfectly good mp3.

The extension was never load-bearing anyway: ffmpeg reads by content, and the
upload is stored under a fixed name (``output/music_custom.mp3``) whatever it
arrived as. So the check is now "does this look like audio at all", answered
from three independent signals — the magic bytes first, then the browser's
declared MIME type, then the extension — and only a file that fails all three
is refused.
"""
from __future__ import annotations

from pathlib import Path

# Container/codec signatures, as (offset, magic) pairs. Deliberately the
# common web-audio set rather than everything ffmpeg can open: this exists to
# refuse a PDF, not to be an exhaustive format database.
_MAGIC: tuple[tuple[int, bytes], ...] = (
    (0, b"ID3"),      # mp3 with an ID3v2 tag (by far the common case)
    (0, b"RIFF"),     # wav (RIFF....WAVE)
    (0, b"OggS"),     # ogg / opus
    (0, b"fLaC"),     # flac
    (0, b"FORM"),     # aiff
    (0, b"\x30\x26\xb2\x75"),  # asf / wma
    (4, b"ftyp"),     # m4a / mp4 audio
)

AUDIO_EXTENSIONS = frozenset(
    {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac",
     ".aif", ".aiff", ".wma", ".mp4", ".webm"}
)


def _has_mpeg_sync(data: bytes) -> bool:
    """A bare mp3/aac stream with no tag: an 11-bit frame-sync word.

    Only the first few bytes are examined — a real stream starts on a frame
    (or an ID3 tag, handled above), and scanning further would start
    matching arbitrary binary noise.
    """
    for i in range(0, min(4, max(0, len(data) - 1))):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            return True
    return False


def looks_like_audio(
    data: bytes, filename: str = "", content_type: str = ""
) -> bool:
    """True when the upload is plausibly an audio file.

    Any one signal is enough. Content is checked first because it is the only
    one a phone cannot get wrong: iOS routinely supplies a name with no
    extension, and some browsers send ``application/octet-stream`` for a file
    they cannot classify.
    """
    if any(data[offset:offset + len(magic)] == magic for offset, magic in _MAGIC):
        return True
    if _has_mpeg_sync(data):
        return True
    if content_type and content_type.split(";")[0].strip().lower().startswith(
        ("audio/", "video/")  # m4a/mp4 audio is often labelled video/mp4
    ):
        return True
    return Path(filename or "").suffix.lower() in AUDIO_EXTENSIONS
