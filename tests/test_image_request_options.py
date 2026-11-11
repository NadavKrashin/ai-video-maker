"""The canvas, quality and reference labels sent when styling a photo.

The canvas is a true 16:9 one by default (2048x1152); quality and labels
are opt-in, so with nothing configured they add nothing to the request.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from ai_video_maker.clients.openai_client import (
    OpenAIClient,
    reference_label_block,
)
from ai_video_maker.config import Config
from ai_video_maker.media.images import normalize_image

REPO = Path(__file__).resolve().parents[1]


def _config(**overrides) -> Config:
    data = json.loads((REPO / "config.json").read_text())
    data.update(overrides)
    return Config(**data)


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, "PNG")
    return buf.getvalue()


class _Images:
    def __init__(self, size=(1536, 1024)):
        self.calls = []
        self._size = size

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        data = base64.b64encode(_png(*self._size)).decode()
        return SimpleNamespace(data=[SimpleNamespace(b64_json=data)])


def _client(config: Config, images: _Images) -> OpenAIClient:
    client = OpenAIClient(config)
    client._client = SimpleNamespace(images=images)
    return client


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    path = tmp_path / "beach.jpg"
    Image.new("RGB", (800, 1200), "blue").save(path)
    return path


class TestDefaults:
    def test_the_live_config_asks_for_a_true_16_9_canvas(self):
        config = _config()
        assert config.openai_image_size == "2048x1152"
        # ...and so does a config file that does not mention it.
        assert Config.model_fields["openai_image_size"].default == "2048x1152"
        assert config.openai_image_quality == ""
        assert config.style_reference_labels is False

    def test_no_quality_key_is_sent_unless_configured(self, photo, tmp_path):
        images = _Images()
        _client(_config(), images).style_image(photo, "style", tmp_path / "o.png")
        assert images.calls[0]["size"] == "2048x1152"
        assert "quality" not in images.calls[0]

    def test_the_old_canvas_is_still_one_setting_away(self, photo, tmp_path):
        images = _Images()
        config = _config(openai_image_size="1536x1024")
        _client(config, images).style_image(photo, "style", tmp_path / "o.png")
        assert images.calls[0]["size"] == "1536x1024"


class TestCanvasAndQuality:
    def test_configured_size_and_quality_reach_the_request(self, photo, tmp_path):
        images = _Images(size=(2048, 1152))
        config = _config(openai_image_size="2048x1152", openai_image_quality="high")
        _client(config, images).style_image(photo, "style", tmp_path / "o.png")
        assert images.calls[0]["size"] == "2048x1152"
        assert images.calls[0]["quality"] == "high"

    def test_a_16_9_canvas_is_not_cropped_on_the_way_to_1080p(self, tmp_path):
        # A marker row at the very top of the canvas. On the 3:2 canvas the
        # centre-crop to 16:9 removes it; on a 16:9 canvas it survives.
        def top_row_survives(width: int, height: int) -> bool:
            src, dst = tmp_path / f"{width}.png", tmp_path / f"{width}_out.png"
            im = Image.new("RGB", (width, height), "white")
            im.paste("red", (0, 0, width, height // 20))
            im.save(src)
            normalize_image(src, dst, 1920, 1080)
            r, g, b = Image.open(dst).getpixel((960, 2))
            return r > 200 and g < 60 and b < 60

        assert top_row_survives(2048, 1152)
        assert not top_row_survives(1536, 1024)

    @pytest.mark.parametrize("size", [
        "1920x1080",   # 1080 is not a multiple of 16 — the tempting mistake
        "2048",        # not WxH
        "4096x1024",   # 4:1
        "512x512",     # too few pixels
        "3840x2400",   # too many
    ])
    def test_an_invalid_canvas_is_refused_at_load(self, size):
        with pytest.raises(ValueError):
            _config(openai_image_size=size)

    @pytest.mark.parametrize("size", ["1536x1024", "1536x864", "2048x1152", "2560x1440"])
    def test_valid_canvases_load(self, size):
        assert _config(openai_image_size=size).openai_image_size == size

    def test_an_unknown_quality_is_refused(self):
        with pytest.raises(ValueError):
            _config(openai_image_quality="ultra")


class TestReferenceLabels:
    def test_no_labels_means_no_extra_text(self):
        assert reference_label_block(None) == ""
        assert reference_label_block([]) == ""
        assert reference_label_block(["", "  "]) == ""

    def test_labels_are_numbered_from_image_2_in_attachment_order(self):
        block = reference_label_block(["the bald man", "the taller boy"])
        assert "Image 1 is the photograph" in block
        assert "Image 2 is the established face of the bald man" in block
        assert "Image 3 is the established face of the taller boy" in block

    def test_a_blank_label_keeps_the_numbering_of_the_others(self):
        block = reference_label_block(["", "the taller boy"])
        assert "Image 2" not in block
        assert "Image 3 is the established face of the taller boy" in block

    def test_labels_only_ride_along_with_references(self, photo, tmp_path):
        images = _Images()
        client = _client(_config(), images)
        client.style_image(
            photo, "style", tmp_path / "o.png", reference_labels=["the bald man"]
        )
        assert images.calls[0]["prompt"] == "style"

    def test_labels_are_appended_after_the_reference_rules(self, photo, tmp_path):
        ref = tmp_path / "bald.png"
        Image.new("RGB", (256, 256), "pink").save(ref)
        images = _Images()
        _client(_config(), images).style_image(
            photo, "style", tmp_path / "o.png",
            references=[ref], reference_labels=["the bald man"],
        )
        prompt = images.calls[0]["prompt"]
        assert prompt.index("CHARACTER REFERENCES") < prompt.index("WHICH REFERENCE IS WHO")
        assert "Image 2 is the established face of the bald man" in prompt
        assert len(images.calls[0]["image"]) == 2
