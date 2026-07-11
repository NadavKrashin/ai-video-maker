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
    back into itself on a re-run. Frame ids may be numeric (legacy positional
    naming) or filename slugs, so anything of the form ``X_to_Y.mp4`` counts.
    """
    clips = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".mp4"
        and re.match(r".+_to_.+$", p.stem)
    ]
    return sorted(clips, key=natural_sort_key)


def combine_clips(
    clips: list[Path], output: Path, force_filter: bool = False
) -> None:
    """Concatenate ``clips`` (in the given order) into ``output`` via ffmpeg.

    Uses the concat demuxer with stream copy first (fast, lossless). If that
    fails — e.g. clips with mismatched codecs/parameters — it falls back to
    re-encoding so the join still succeeds.

    When only *some* clips carry audio (e.g. one SFX job failed), the concat
    demuxer can't be used at all — it requires every file to have the same
    streams — so the clips are joined with the concat filter instead, padding
    the silent ones with a generated silent track. ``force_filter`` takes that
    path unconditionally — required when the list mixes provider clips with
    locally rendered segments (photo stills), whose encodings differ in ways
    the demuxer's stream copy would join into a corrupt file rather than
    reject.
    """
    if not clips:
        raise ValueError("No clips to combine.")

    if shutil.which("ffmpeg") is None:
        raise _missing_tool_error("ffmpeg", "combine clips")

    output.parent.mkdir(parents=True, exist_ok=True)

    audio_flags = [has_audio_stream(c) for c in clips]
    if force_filter or (any(audio_flags) and not all(audio_flags)):
        if any(audio_flags) and not all(audio_flags):
            logger.warning(
                "%d of %d clip(s) have no audio; concatenating with silent "
                "padding so the join stays in sync.",
                audio_flags.count(False), len(clips),
            )
        _combine_clips_mixed_audio(clips, audio_flags, output)
        return

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


def _combine_clips_mixed_audio(
    clips: list[Path], audio_flags: list[bool], output: Path
) -> None:
    """Concat filter join for clips where only some have an audio stream.

    Every clip contributes its video; clips without audio get a matching-length
    silent stereo track (anullsrc) so the filter sees a uniform [v][a] pair per
    clip. Always re-encodes (the concat filter can't stream-copy).
    """
    cmd: list[str] = ["ffmpeg", "-y"]
    for clip in clips:
        cmd += ["-i", str(clip)]

    silent_input_index: dict[int, int] = {}
    next_index = len(clips)
    for i, has_audio in enumerate(audio_flags):
        if not has_audio:
            duration = ffprobe_duration(clips[i]) or 5.0
            cmd += [
                "-f", "lavfi", "-t", f"{duration:.3f}",
                "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            ]
            silent_input_index[i] = next_index
            next_index += 1

    pairs = []
    for i, has_audio in enumerate(audio_flags):
        a_index = i if has_audio else silent_input_index[i]
        pairs.append(f"[{i}:v][{a_index}:a]")
    filter_graph = "".join(pairs) + f"concat=n={len(clips)}:v=1:a=1[v][a]"

    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg mixed-audio concat failed:\n{result.stderr.strip()[-2000:]}"
        )


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


def _mux_music_cmd(
    video: Path,
    music: Path,
    dst: Path,
    music_volume: float,
    sfx_volume: float,
    mix_with_existing: bool,
    loop: bool,
) -> list[str]:
    """Build the ffmpeg command for mux_music (pure; unit-testable).

    Looping uses -stream_loop -1 + -shortest (music repeats, trimmed to the
    video). Play-once instead pads the track with trailing silence (apad):
    -shortest still ends the output exactly at the video's length whether the
    music is shorter or longer than the video, and amix keeps two live inputs
    throughout so the SFX level doesn't jump when the music ends.
    """
    music_input = (["-stream_loop", "-1"] if loop else []) + ["-i", str(music)]
    pad = "" if loop else ",apad"
    if mix_with_existing:
        filter_graph = (
            f"[0:a]volume={sfx_volume}[s];"
            f"[1:a]volume={music_volume}{pad}[m];"
            f"[s][m]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
    else:
        filter_graph = f"[1:a]volume={music_volume}{pad}[a]"
    return [
        "ffmpeg", "-y",
        "-i", str(video),
        *music_input,
        "-filter_complex", filter_graph,
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(dst),
    ]


def mux_music(
    video: Path,
    music: Path,
    music_volume: float,
    sfx_volume: float = 1.0,
    loop: bool = False,
) -> None:
    """Mix a music track into `video`'s audio, in place.

    If the video already has an audio track (e.g. SFX), the music is set to
    `music_volume` and the existing SFX to `sfx_volume`, then the two are mixed
    — so with `music_volume > sfx_volume` the background music sits louder, on
    top of the clip SFX. If the video has no audio, the music becomes the only
    track (at `music_volume`). With ``loop`` the track repeats for the whole
    video; otherwise it plays once and the remainder continues with SFX only
    (music longer than the video is trimmed either way).
    """
    _require_ffmpeg()
    # ffprobe decides which mix graph to use below; without it we'd silently
    # take the "no existing audio" branch and drop every clip's SFX.
    _require_ffmpeg("ffprobe")
    tmp = video.with_suffix(".muxed.mp4")
    cmd = _mux_music_cmd(
        video, music, tmp,
        music_volume=max(0.0, min(1.0, music_volume)),
        sfx_volume=max(0.0, min(1.0, sfx_volume)),
        mix_with_existing=has_audio_stream(video),
        loop=loop,
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg music mux failed:\n{result.stderr.strip()[-2000:]}"
        )
    tmp.replace(video)


# --- Real-photo presentation segments (opening reveal / end credits) -------- #
# Rendered locally from the user's original photos; frame rate only needs to be
# sensible, since these segments always go through the re-encoding concat path.
_SEGMENT_FPS = 30


def _photo_fit_filter(width: int, height: int, label_in: str = "0:v") -> str:
    """Fit a photo of any aspect ratio onto a width×height canvas.

    The photo is scaled to *cover* the canvas, blurred, and used as the
    background; the same photo scaled to *fit* is centred on top. Portrait
    photos keep everyone's head — no cropping — with the classic blurred-fill
    look instead of black bars.
    """
    return (
        f"[{label_in}]split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma=40[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        # setsar: odd photo dimensions leave a near-1 fractional sample aspect
        # ratio after scaling, and the concat filter refuses to join that with
        # the clips' exact 1:1.
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def _photo_still_cmd(
    photo: Path, dst: Path, width: int, height: int, seconds: float, fade: float
) -> list[str]:
    """Build the ffmpeg command for one end-credits photo still (pure)."""
    graph = (
        _photo_fit_filter(width, height)
        + f",fade=t=in:st=0:d={fade:.3f}"
        + f",fade=t=out:st={max(0.0, seconds - fade):.3f}:d={fade:.3f}"
        + f",fps={_SEGMENT_FPS},format=yuv420p[v]"
    )
    return [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(photo),
        "-filter_complex", graph, "-map", "[v]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(dst),
    ]


def render_photo_still(
    photo: Path,
    dst: Path,
    width: int,
    height: int,
    seconds: float,
    fade: float = 0.5,
) -> None:
    """Render `photo` as a video still of `seconds` with fade in/out."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _photo_still_cmd(photo, dst, width, height, seconds, fade)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg photo still failed for {photo.name}:\n"
            f"{result.stderr.strip()[-1500:]}"
        )


def _intro_segment_cmd(
    src: Path, dst: Path, width: int, height: int, has_audio: bool
) -> list[str]:
    """Build the ffmpeg command normalizing the user's intro clip (pure).

    The intro can come from anywhere (a phone, an editor), so it's scaled to
    fit inside the movie's width×height (black pads preserve the aspect
    ratio) and re-encoded at the segment fps with square pixels — otherwise
    the concat filter refuses to join it with the provider clips.
    """
    graph = (
        f"[0:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={_SEGMENT_FPS},format=yuv420p[v]"
    )
    maps = ["-map", "[v]"]
    audio_codec: list[str] = []
    if has_audio:
        maps += ["-map", "0:a"]
        audio_codec = ["-c:a", "aac"]
    return [
        "ffmpeg", "-y", "-i", str(src),
        "-filter_complex", graph, *maps,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        *audio_codec, str(dst),
    ]


def render_intro_segment(src: Path, dst: Path, width: int, height: int) -> None:
    """Normalize the user's intro clip to the movie's canvas, into `dst`."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _intro_segment_cmd(src, dst, width, height, has_audio_stream(src))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg intro segment failed for {src.name}:\n"
            f"{result.stderr.strip()[-1500:]}"
        )


def _letter_scroll_cmd(
    image: Path,
    dst: Path,
    width: int,
    height: int,
    image_height: int,
    pixels_per_second: float,
) -> list[str]:
    """Build the ffmpeg command scrolling a tall letter image (pure).

    A screen-height crop window travels down the image at a constant speed;
    on screen the text rolls upward through the frame, credits-style. The
    duration is exactly the travel time from all-blank top padding to
    all-blank bottom padding (min() clamps float rounding on the last frame).
    """
    duration = max(0.5, (image_height - height) / pixels_per_second)
    graph = (
        f"[0:v]fps={_SEGMENT_FPS},"
        f"crop={width}:{height}:0:'min(t*{pixels_per_second:.3f},ih-{height})',"
        f"setsar=1,format=yuv420p[v]"
    )
    return [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{duration:.3f}", "-i", str(image),
        "-filter_complex", graph, "-map", "[v]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", str(dst),
    ]


def render_letter_scroll(
    image: Path,
    dst: Path,
    width: int,
    height: int,
    image_height: int,
    pixels_per_second: float,
) -> None:
    """Render the tall letter image as a scrolling video segment at `dst`."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _letter_scroll_cmd(
        image, dst, width, height, image_height, pixels_per_second
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg letter scroll failed:\n{result.stderr.strip()[-1500:]}"
        )


def _letter_overlay_cmd(
    background: Path, letter_png: Path, dst: Path, pixels_per_second: float
) -> list[str]:
    """Build the ffmpeg command scrolling the letter OVER a video (pure).

    The background (the photo montage) is dimmed with a translucent black
    scrim so the white letter text stays readable above any photo, then the
    transparent letter image slides from below the frame to above it. The
    output lasts exactly as long as the background.
    """
    graph = (
        "[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=black@0.55:t=fill[dim];"
        f"[dim][1:v]overlay=x=(W-w)/2:y='H-t*{pixels_per_second:.3f}',"
        "format=yuv420p[v]"
    )
    return [
        "ffmpeg", "-y",
        "-i", str(background), "-i", str(letter_png),
        "-filter_complex", graph,
        "-map", "[v]", "-map", "0:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", str(dst),
    ]


def render_letter_overlay(
    background: Path, letter_png: Path, dst: Path, pixels_per_second: float
) -> None:
    """Render the letter scrolling over `background` into `dst`."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _letter_overlay_cmd(background, letter_png, dst, pixels_per_second)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg letter overlay failed:\n{result.stderr.strip()[-1500:]}"
        )


def _end_fade_cmd(
    video: Path, dst: Path, start: float, seconds: float, has_audio: bool
) -> list[str]:
    """Build the ffmpeg command fading the video's ending to black (pure)."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"fade=t=out:st={start:.3f}:d={seconds:.3f}",
    ]
    if has_audio:
        cmd += ["-af", f"afade=t=out:st={start:.3f}:d={seconds:.3f}"]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium"]
    if has_audio:
        cmd += ["-c:a", "aac"]
    return cmd + [str(dst)]


def apply_end_fade(video: Path, seconds: float) -> None:
    """Fade the last `seconds` of `video` to black (and its audio to silence),
    in place. Run AFTER the music mux so the music bed fades too."""
    _require_ffmpeg()
    _require_ffmpeg("ffprobe")
    duration = ffprobe_duration(video)
    if not duration or duration <= 2 * seconds:
        logger.warning(
            "End fade skipped: %s is too short (%.1fs).", video.name, duration or 0
        )
        return
    tmp = video.with_suffix(".fade.mp4")
    cmd = _end_fade_cmd(
        video, tmp, duration - seconds, seconds, has_audio_stream(video)
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg end fade failed:\n{result.stderr.strip()[-1500:]}"
        )
    tmp.replace(video)


def _opening_reveal_cmd(
    photo: Path,
    clip: Path,
    dst: Path,
    width: int,
    height: int,
    hold: float,
    fade: float,
    clip_has_audio: bool,
) -> list[str]:
    """Build the ffmpeg command for the photo→clip opening reveal (pure).

    The still runs `hold` seconds, then crossfades over `fade` seconds into
    the clip (xfade needs the still to last hold+fade). The clip's audio, if
    any, is delayed by `hold` so sound begins as the photo comes alive.
    """
    graph = (
        _photo_fit_filter(width, height)
        + f",fps={_SEGMENT_FPS},format=yuv420p,settb=AVTB[ph];"
        f"[1:v]fps={_SEGMENT_FPS},format=yuv420p,setsar=1,settb=AVTB[cl];"
        f"[ph][cl]xfade=transition=fade:duration={fade:.3f}:offset={hold:.3f}[v]"
    )
    maps = ["-map", "[v]"]
    audio_codec: list[str] = []
    if clip_has_audio:
        delay_ms = int(round(hold * 1000))
        graph += f";[1:a]adelay={delay_ms}:all=1[a]"
        maps += ["-map", "[a]"]
        audio_codec = ["-c:a", "aac"]
    return [
        "ffmpeg", "-y",
        "-loop", "1", "-t", f"{hold + fade:.3f}", "-i", str(photo),
        "-i", str(clip),
        "-filter_complex", graph, *maps,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        *audio_codec, str(dst),
    ]


def render_opening_reveal(
    photo: Path,
    clip: Path,
    dst: Path,
    width: int,
    height: int,
    hold: float,
    fade: float = 0.8,
) -> None:
    """Render `photo` held for `hold`s, crossfading into `clip`, to `dst`."""
    _require_ffmpeg()
    _require_ffmpeg("ffprobe")  # audio probing decides the mux graph
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = _opening_reveal_cmd(
        photo, clip, dst, width, height, hold, fade,
        clip_has_audio=has_audio_stream(clip),
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg opening reveal failed:\n{result.stderr.strip()[-1500:]}"
        )
