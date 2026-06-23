"""Video assembly utilities (ffmpeg / ffprobe).

Concatenation of clips, audio probing, edge fades and music muxing. Every
function shells out to ffmpeg/ffprobe and is import-safe (the tools are only
required when the function actually runs).
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from ..logging_setup import logger
from .images import natural_sort_key


def _missing_tool_error(tool: str, purpose: str) -> RuntimeError:
    """Build a 'not found on PATH' error with an OS-appropriate install hint."""
    if sys.platform == "win32":
        how = (
            "Install it with `winget install Gyan.FFmpeg` (or "
            "`choco install ffmpeg`), then open a new terminal so it's on PATH"
        )
    elif sys.platform == "darwin":
        how = "Install it with `brew install ffmpeg`"
    else:
        how = "Install it with your package manager (e.g. `apt install ffmpeg`)"
    return RuntimeError(f"{tool} not found on PATH. {how} to {purpose}.")


def find_generated_clips(directory: Path) -> list[Path]:
    """Return the generated transition clips (``<id>_to_<id>.mp4``) in order.

    Only clips matching the pipeline's naming scheme are returned, so the
    combined ``final_video.mp4`` (or any other stray file) is never folded
    back into itself on a re-run.
    """
    clips = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".mp4"
        and re.match(r"\d+_to_\d+$", p.stem)
    ]
    return sorted(clips, key=natural_sort_key)


def combine_clips(clips: list[Path], output: Path) -> None:
    """Concatenate ``clips`` (in the given order) into ``output`` via ffmpeg.

    Uses the concat demuxer with stream copy first (fast, lossless). If that
    fails — e.g. clips with mismatched codecs/parameters — it falls back to
    re-encoding so the join still succeeds.
    """
    if not clips:
        raise ValueError("No clips to combine.")

    if shutil.which("ffmpeg") is None:
        raise _missing_tool_error("ffmpeg", "combine clips")

    output.parent.mkdir(parents=True, exist_ok=True)

    # ffmpeg's concat demuxer reads a list file of `file '<path>'` lines.
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as fh:
        list_path = Path(fh.name)
        for clip in clips:
            safe = str(clip.resolve()).replace("'", r"'\''")
            fh.write(f"file '{safe}'\n")

    copy_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path), "-c", "copy", str(output),
    ]
    reencode_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_path),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(output),
    ]

    try:
        result = subprocess.run(copy_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Stream-copy concat failed, re-encoding instead. ffmpeg said:\n%s",
                result.stderr.strip()[-2000:],
            )
            result = subprocess.run(reencode_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg concat failed:\n{result.stderr.strip()[-2000:]}"
                )
    finally:
        list_path.unlink(missing_ok=True)


def _require_ffmpeg(tool: str = "ffmpeg") -> None:
    if shutil.which(tool) is None:
        raise _missing_tool_error(tool, "add audio")


def ffprobe_duration(path: Path) -> Optional[float]:
    """Return the media duration in seconds, or None if it can't be read."""
    if shutil.which("ffprobe") is None:
        return None
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (TypeError, ValueError):
        return None


def has_audio_stream(path: Path) -> bool:
    """True if `path` contains at least one audio stream."""
    if shutil.which("ffprobe") is None:
        return False
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def apply_edge_fades(clip: Path, fade: float) -> None:
    """Fade the clip's audio in at the start and out at the end, in place.

    Softens the hard cut at each clip boundary without overlapping clips, so the
    total duration and A/V sync are preserved. No-op when the clip has no audio
    or is too short to hold two fades.
    """
    if fade <= 0 or not has_audio_stream(clip):
        return
    duration = ffprobe_duration(clip)
    if not duration or duration <= fade * 2:
        return

    _require_ffmpeg()
    tmp = clip.with_suffix(".faded.mp4")
    out_start = max(0.0, duration - fade)
    cmd = [
        "ffmpeg", "-y", "-i", str(clip),
        "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={out_start:.3f}:d={fade}",
        "-map", "0:v", "-map", "0:a",
        "-c:v", "copy", "-c:a", "aac", str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg edge-fade failed for {clip.name}:\n{result.stderr.strip()[-1500:]}"
        )
    tmp.replace(clip)


def mux_music(
    video: Path, music: Path, music_volume: float, sfx_volume: float = 1.0
) -> None:
    """Mix a (looped) music track into `video`'s audio, in place.

    If the video already has an audio track (e.g. SFX), the music is set to
    `music_volume` and the existing SFX to `sfx_volume`, then the two are mixed
    — so with `music_volume > sfx_volume` the background music sits louder, on
    top of the clip SFX. If the video has no audio, the music becomes the only
    track (at `music_volume`). The music is looped/trimmed to the video length
    either way.
    """
    _require_ffmpeg()
    tmp = video.with_suffix(".muxed.mp4")
    m_vol = max(0.0, min(1.0, music_volume))
    s_vol = max(0.0, min(1.0, sfx_volume))

    if has_audio_stream(video):
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-stream_loop", "-1", "-i", str(music),
            "-filter_complex",
            f"[0:a]volume={s_vol}[s];"
            f"[1:a]volume={m_vol}[m];"
            f"[s][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
            "-map", "0:v", "-map", "[a]",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp),
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video),
            "-stream_loop", "-1", "-i", str(music),
            "-filter:a", f"volume={m_vol}",
            "-map", "0:v", "-map", "1:a",
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(tmp),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg music mux failed:\n{result.stderr.strip()[-2000:]}"
        )
    tmp.replace(video)
