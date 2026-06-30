"""Image utilities (Pillow) — normalise everything to exactly target size."""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

from PIL import Image

from ..logging_setup import logger

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_DATA_URL_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def encode_image_data_url(path: Path) -> str:
    """Read `path` and return a base64 ``data:`` URL for the vision API."""
    mime = _DATA_URL_MIME.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def natural_sort_key(path: Path) -> list[Any]:
    """Sort key that orders e.g. img2 before img10 (natural ordering)."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_input_images(directory: Path) -> list[Path]:
    files = [
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    ]
    return sorted(files, key=natural_sort_key)


def normalize_image(src: Path, dst: Path, width: int, height: int) -> None:
    """
    Force `src` to be exactly width x height and write to `dst`.

    Strategy: convert to RGB, then center-crop to the target aspect ratio
    (cover) and resize. This avoids distortion and preserves the center
    composition. If the source is smaller, it is scaled up.
    """
    with Image.open(src) as im:
        im = im.convert("RGB")
        target_ratio = width / height
        src_ratio = im.width / im.height

        if abs(src_ratio - target_ratio) < 1e-3:
            cropped = im
        elif src_ratio > target_ratio:
            # Too wide -> crop the sides.
            new_w = int(round(im.height * target_ratio))
            left = (im.width - new_w) // 2
            cropped = im.crop((left, 0, left + new_w, im.height))
        else:
            # Too tall -> crop top/bottom.
            new_h = int(round(im.width / target_ratio))
            top = (im.height - new_h) // 2
            cropped = im.crop((0, top, im.width, top + new_h))

        resized = cropped.resize((width, height), Image.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, format="PNG")


def verify_dimensions(path: Path, width: int, height: int) -> bool:
    try:
        with Image.open(path) as im:
            return im.size == (width, height)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not verify %s: %s", path, exc)
        return False
