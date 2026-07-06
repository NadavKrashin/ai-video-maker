"""media/: image utilities and clip discovery."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_video_maker.media.ffmpeg import find_generated_clips
from ai_video_maker.media.images import (
    encode_image_data_url,
    list_input_images,
    natural_sort_key,
    normalize_image,
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
