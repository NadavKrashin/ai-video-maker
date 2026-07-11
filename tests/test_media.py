"""media/: image utilities and clip discovery."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_video_maker.media.ffmpeg import (
    _end_fade_cmd,
    _intro_segment_cmd,
    _letter_overlay_cmd,
    _letter_scroll_cmd,
    _mux_music_cmd,
    _photo_fit_filter,
    _photo_still_cmd,
    find_generated_clips,
)
from ai_video_maker.media.images import (
    encode_image_data_url,
    list_input_images,
    natural_sort_key,
    normalize_image,
    slugify_stem,
    verify_dimensions,
)


def _img(path: Path, size=(100, 60)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path)
    return path


class TestNaturalSort:
    def test_numeric_order(self, tmp_path):
        names = ["img10.png", "img2.png", "img1.png"]
        paths = sorted((tmp_path / n for n in names), key=natural_sort_key)
        assert [p.name for p in paths] == ["img1.png", "img2.png", "img10.png"]


class TestListInputImages:
    def test_filters_and_orders(self, tmp_path):
        for name in ("b2.png", "b10.jpg", "notes.txt", "clip.mp4", "a.webp"):
            (tmp_path / name).write_bytes(b"x")
        assert [p.name for p in list_input_images(tmp_path)] == [
            "a.webp", "b2.png", "b10.jpg",
        ]


class TestNormalizeImage:
    def test_wide_source_cropped_to_target(self, tmp_path):
        src = _img(tmp_path / "wide.png", (3000, 1000))
        dst = tmp_path / "out.png"
        normalize_image(src, dst, 1920, 1080)
        assert verify_dimensions(dst, 1920, 1080)

    def test_tall_source_cropped_to_target(self, tmp_path):
        src = _img(tmp_path / "tall.png", (1000, 3000))
        dst = tmp_path / "out.png"
        normalize_image(src, dst, 1920, 1080)
        assert verify_dimensions(dst, 1920, 1080)


class TestEncodeDataUrl:
    def test_full_resolution_keeps_mime(self, tmp_path):
        src = _img(tmp_path / "a.png")
        assert encode_image_data_url(src).startswith("data:image/png;base64,")

    def test_max_edge_downscales_to_jpeg(self, tmp_path):
        src = _img(tmp_path / "big.png", (1920, 1080))
        url = encode_image_data_url(src, max_edge=768)
        assert url.startswith("data:image/jpeg;base64,")
        assert len(url) < len(encode_image_data_url(src))

    def test_small_image_not_upscaled(self, tmp_path):
        import base64, io
        src = _img(tmp_path / "small.png", (200, 100))
        url = encode_image_data_url(src, max_edge=768)
        raw = base64.b64decode(url.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as im:
            assert im.size == (200, 100)


class TestSlugifyStem:
    def test_keeps_safe_characters(self):
        assert slugify_stem("img4a") == "img4a"
        assert slugify_stem("IMG_2043-final") == "IMG_2043-final"

    def test_replaces_unsafe_characters(self):
        assert slugify_stem("My Photo (1)") == "My_Photo_1"
        assert slugify_stem("קובץ") == "frame"  # nothing safe left -> fallback


class TestMuxMusicCmd:
    def _cmd(self, *, mixed: bool, loop: bool) -> list[str]:
        return _mux_music_cmd(
            Path("v.mp4"), Path("m.mp3"), Path("out.mp4"),
            music_volume=0.85, sfx_volume=0.35,
            mix_with_existing=mixed, loop=loop,
        )

    def test_loop_uses_stream_loop(self):
        cmd = self._cmd(mixed=True, loop=True)
        assert "-stream_loop" in cmd
        assert "apad" not in " ".join(cmd)

    def test_play_once_pads_instead_of_looping(self):
        for mixed in (True, False):
            cmd = self._cmd(mixed=mixed, loop=False)
            assert "-stream_loop" not in cmd
            assert "apad" in " ".join(cmd)
            # -shortest + padded (infinite) audio = output always ends exactly
            # at the video's length, never truncated to a short music track.
            assert "-shortest" in cmd

    def test_mixed_graph_keeps_both_tracks(self):
        cmd = self._cmd(mixed=True, loop=False)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "amix=inputs=2" in graph
        assert "volume=0.35" in graph and "volume=0.85" in graph

    def test_music_only_graph_has_no_mix(self):
        cmd = self._cmd(mixed=False, loop=False)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "amix" not in graph and "volume=0.85" in graph


class TestPhotoSegments:
    def test_fit_filter_pads_with_blur_never_crops_subject(self):
        graph = _photo_fit_filter(1920, 1080)
        assert "gblur" in graph                                # blurred fill
        assert "force_original_aspect_ratio=decrease" in graph  # subject fitted
        assert "overlay=(W-w)/2:(H-h)/2" in graph              # centred
        assert "setsar=1" in graph  # concat refuses mismatched SAR otherwise

    def test_still_cmd_loops_photo_for_duration_with_fades(self):
        cmd = _photo_still_cmd(
            Path("p.jpg"), Path("out.mp4"), 1920, 1080, seconds=2.5, fade=0.5
        )
        assert cmd[cmd.index("-loop") + 1] == "1"
        assert cmd[cmd.index("-t") + 1] == "2.500"
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "fade=t=in" in graph and "fade=t=out:st=2.000" in graph

    def test_letter_scroll_travels_full_image(self):
        # 3240 tall - 1080 window = 2160px of travel at 154.286 px/s -> ~14s
        cmd = _letter_scroll_cmd(
            Path("letter.png"), Path("out.mp4"), 1920, 1080,
            image_height=3240, pixels_per_second=154.286,
        )
        assert cmd[cmd.index("-t") + 1] == "14.000"
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "crop=1920:1080:0:'min(t*154.286,ih-1080)'" in graph
        assert "setsar=1" in graph

    def test_letter_overlay_dims_bg_and_scrolls_up(self):
        cmd = _letter_overlay_cmd(
            Path("bg.mp4"), Path("letter.png"), Path("out.mp4"),
            pixels_per_second=120.0,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "drawbox" in graph and "black@0.55" in graph  # readability scrim
        assert "overlay=x=(W-w)/2:y='H-t*120.000'" in graph  # enters from below
        assert "0:a?" in cmd  # background audio kept if present

    def test_end_fade_cmd_fades_video_and_audio(self):
        cmd = _end_fade_cmd(
            Path("v.mp4"), Path("t.mp4"), start=58.5, seconds=1.5, has_audio=True
        )
        joined = " ".join(cmd)
        assert "fade=t=out:st=58.500:d=1.500" in joined
        assert "afade=t=out:st=58.500:d=1.500" in joined

    def test_end_fade_cmd_silent_video_skips_afade(self):
        cmd = _end_fade_cmd(
            Path("v.mp4"), Path("t.mp4"), start=10.0, seconds=1.5, has_audio=False
        )
        joined = " ".join(cmd)
        assert "afade" not in joined and "-c:a" not in joined

    def test_intro_cmd_fits_and_pads_without_cropping(self):
        cmd = _intro_segment_cmd(
            Path("intro.mp4"), Path("out.mp4"), 1920, 1080, has_audio=True
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        # fitted whole (never cropped), centred on black pads, concat-safe
        assert "force_original_aspect_ratio=decrease" in graph
        assert "pad=1920:1080:(ow-iw)/2:(oh-ih)/2" in graph
        assert "setsar=1" in graph
        assert "-c:a" in cmd and "0:a" in cmd  # intro sound kept

    def test_intro_cmd_silent_source_maps_no_audio(self):
        cmd = _intro_segment_cmd(
            Path("intro.mp4"), Path("out.mp4"), 1920, 1080, has_audio=False
        )
        joined = " ".join(cmd)
        assert "-c:a" not in joined and "0:a" not in joined


class TestFindGeneratedClips:
    def test_matches_only_pipeline_naming(self, tmp_path):
        for name in (
            "001_to_002.mp4", "002_to_003.mp4", "010_to_011.mp4",
            "final_video.mp4", "notes.txt", "001_to_002.mov",
        ):
            (tmp_path / name).write_bytes(b"x")
        assert [p.name for p in find_generated_clips(tmp_path)] == [
            "001_to_002.mp4", "002_to_003.mp4", "010_to_011.mp4",
        ]

    def test_accepts_filename_keyed_clips(self, tmp_path):
        for name in ("img4_to_img4a.mp4", "img4a_to_img5.mp4", "final_video.mp4"):
            (tmp_path / name).write_bytes(b"x")
        assert [p.name for p in find_generated_clips(tmp_path)] == [
            "img4_to_img4a.mp4", "img4a_to_img5.mp4",
        ]
