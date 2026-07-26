"""Fetch a music bed from a URL into the project's custom-music slot.

Two shapes of link are supported, because they solve different problems:

* a **direct audio file** (``.../track.mp3``) — streamed straight to disk with
  ``requests``, no extra tooling. This is what royalty-free libraries hand
  out, and it is the dependency-free path.
* a **media page** (a YouTube watch URL and the many other sites yt-dlp
  understands) — the audio stream is extracted and transcoded to mp3 by
  yt-dlp + ffmpeg.

The result always lands at the SAME path an upload writes
(``output/music_custom.mp3``), so everything downstream — resolution,
mixing, the panel's "track uploaded" badge — is unchanged.

LICENSING IS THE CALLER'S PROBLEM AND IT IS A REAL ONE: these movies are
sold to customers, and most music on a video platform is copyrighted, so
embedding it infringes regardless of how the file was obtained. Use the
platform's royalty-free library, CC-licensed tracks, or music you have
licensed. The pipeline cannot check this, so the UI says it where the URL is
entered.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from ..errors import PipelineError
from ..logging_setup import logger

# Extensions that mean "this URL IS the audio file" — fetched directly.
_DIRECT_AUDIO_SUFFIXES = {
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".flac",
}

# A music bed is a few MB; this only stops a mistyped link from filling the
# disk. Checked against Content-Length AND while streaming, because a server
# can lie about (or omit) the header.
_MAX_BYTES = 60 * 1024 * 1024

# Whole-fetch ceilings. A song downloads in seconds; anything far beyond that
# is a wrong link or a site fighting us, and the panel is waiting on it.
_DIRECT_TIMEOUT_SECONDS = 120
_EXTRACT_TIMEOUT_SECONDS = 300


def _validate(url: str) -> str:
    url = (url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise PipelineError(
            f"Not a usable music URL: {url!r}. Give a full http(s) link — "
            "either a direct audio file or a video/track page."
        )
    return url


def looks_like_direct_audio(url: str) -> bool:
    """Does this URL point straight at an audio file (vs a media page)?"""
    return Path(urlparse(url).path).suffix.lower() in _DIRECT_AUDIO_SUFFIXES


def _download_direct(url: str, dst: Path) -> None:
    """Stream an audio file to `dst`, enforcing the size cap as it goes."""
    with requests.get(url, stream=True, timeout=_DIRECT_TIMEOUT_SECONDS) as resp:
        resp.raise_for_status()
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > _MAX_BYTES:
            raise PipelineError(
                f"That track is {int(declared) // (1024 * 1024)} MB; the limit "
                f"is {_MAX_BYTES // (1024 * 1024)} MB."
            )
        written = 0
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("wb") as out:
            for chunk in resp.iter_content(chunk_size=256 * 1024):
                written += len(chunk)
                if written > _MAX_BYTES:
                    out.close()
                    dst.unlink(missing_ok=True)
                    raise PipelineError(
                        "That download exceeded "
                        f"{_MAX_BYTES // (1024 * 1024)} MB and was stopped."
                    )
                out.write(chunk)
    if written == 0:
        dst.unlink(missing_ok=True)
        raise PipelineError(f"Nothing was downloaded from {url}")


def _extract_audio(url: str, dst: Path) -> None:
    """Pull the audio track off a media page via yt-dlp, as mp3."""
    try:
        import yt_dlp  # imported lazily: only this path needs it
    except ImportError as exc:  # pragma: no cover - depends on install
        raise PipelineError(
            "Fetching audio from a page URL needs yt-dlp. Install it with: "
            ".venv/bin/python -m pip install -U yt-dlp"
        ) from exc

    # yt-dlp names the output itself (extension depends on the postprocessor),
    # so work in a temp dir and move the single result into place.
    with tempfile.TemporaryDirectory() as tmp:
        options = {
            "format": "bestaudio/best",
            "outtmpl": str(Path(tmp) / "track.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "noplaylist": True,       # a link inside a playlist means THAT track
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 30,
            "max_filesize": _MAX_BYTES,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])
        except Exception as exc:  # noqa: BLE001 - yt-dlp raises many types
            raise PipelineError(
                f"Could not get audio from {url}: {exc}. If the site changed, "
                "updating yt-dlp usually fixes it: "
                ".venv/bin/python -m pip install -U yt-dlp"
            ) from exc
        produced = sorted(Path(tmp).glob("track.*"))
        if not produced:
            raise PipelineError(f"No audio track came back from {url}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced[0]), dst)


def fetch_music(url: str, dst: Path) -> Path:
    """Download the music bed at `url` to `dst`; returns `dst`.

    Picks the direct-download path for a link that already names an audio
    file, and the yt-dlp extractor for anything else. Raises PipelineError
    with something the user can act on.
    """
    url = _validate(url)
    direct = looks_like_direct_audio(url)
    logger.info(
        "Fetching music bed from %s (%s)",
        url, "direct download" if direct else "audio extraction",
    )
    if direct:
        _download_direct(url, dst)
    else:
        _extract_audio(url, dst)
    logger.info(
        "Music bed saved: %s (%.1f MB)", dst, dst.stat().st_size / (1024 * 1024)
    )
    return dst
