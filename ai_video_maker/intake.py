"""Web-order intake logic shared by `ingest`, `orders`, the watcher, and the API.

An ingested project records which Cloudinary order it came from in
``projects/<name>/order.json`` (see :func:`write_order_record`). That single
file is what ties the two worlds together: the `orders` listing and the
watcher use it to know which folders are already handled, and the admin API
uses it to show order metadata next to a project.

Everything here is pure logic (no network) so it stays unit-testable; the
actual Cloudinary calls live in ``clients/cloudinary_client.py``.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .clients.cloudinary_client import OrderAsset

# `video-orders/<ORDER-ID>_<customer>-<dd.mm.yyyy_HH-MM>[_<music-mood>]` —
# the leaf format the animoments frontend creates (see its App.jsx
# `uploadFolderRef`; newer versions append the chosen music mood).
_FOLDER_RE = re.compile(
    r"^(?P<order_id>[^_]+)_(?P<customer>.*?)-(?P<stamp>\d{2}\.\d{2}\.\d{4}_\d{2}-\d{2})(?:_.*)?$"
)


def parse_order_folder(leaf: str) -> dict[str, str]:
    """Split an order folder's leaf name into order_id / customer / stamp.

    Unparseable names (hand-made folders) degrade gracefully: the whole leaf
    becomes the order_id and the customer stays empty.
    """
    m = _FOLDER_RE.match(leaf)
    if not m:
        return {"order_id": leaf, "customer": "", "stamp": ""}
    return {
        "order_id": m.group("order_id"),
        "customer": m.group("customer").replace("-", " "),
        "stamp": m.group("stamp"),
    }


def derive_project_name(folder_leaf: str, existing: set[str]) -> str:
    """A friendly, unique project name for an order folder.

    Prefers the customer's name (`liat-heitner`); falls back to the order id
    when the name is empty or non-ASCII (project names must be safe path
    segments and the customer may have typed Hebrew). Collisions get a
    numeric suffix — names in `existing` are never reused.
    """
    parsed = parse_order_folder(folder_leaf)
    base = re.sub(r"[^a-z0-9]+", "-", parsed["customer"].lower()).strip("-")
    if not base:
        base = re.sub(r"[^a-z0-9]+", "-", parsed["order_id"].lower()).strip("-")
    if not base:
        base = "order"
    name = base
    counter = 2
    while name in existing:
        name = f"{base}-{counter}"
        counter += 1
    return name


def is_order_complete(
    assets: list[OrderAsset],
    quiet_minutes: float,
    now: Optional[datetime] = None,
    expected_count: Optional[int] = None,
) -> bool:
    """True when an order's upload looks finished.

    The frontend confirms payment BEFORE the photos finish uploading (one at
    a time, with retries), so a folder's existence never means the order is
    complete. When the order ledger knows the exact photo count
    (`expected_count`, from the Firestore order doc), reaching it IS
    completeness. Otherwise the safe signal is a quiet period: no new photo
    for `quiet_minutes`. Assets missing a parseable timestamp count as fresh
    (not complete) rather than risking a half-ingest.
    """
    if not assets:
        return False
    if expected_count is not None and expected_count > 0:
        return len(assets) >= expected_count
    now = now or datetime.now(timezone.utc)
    newest: Optional[datetime] = None
    for asset in assets:
        try:
            uploaded = datetime.fromisoformat(asset.created_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        if uploaded.tzinfo is None:
            uploaded = uploaded.replace(tzinfo=timezone.utc)
        if newest is None or uploaded > newest:
            newest = uploaded
    assert newest is not None
    return (now - newest).total_seconds() >= quiet_minutes * 60


# ------------------------------ order records ------------------------------- #

def write_order_record(path: Path, *, order_folder: str, photo_count: int) -> None:
    """Record which Cloudinary order a project came from (projects/<n>/order.json)."""
    parsed = parse_order_folder(order_folder)
    record = {
        "order_folder": order_folder,
        "order_id": parsed["order_id"],
        "customer": parsed["customer"],
        "photo_count": photo_count,
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def read_order_record(path: Path) -> Optional[dict]:
    """The project's order.json as a dict; None when absent or unreadable."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def ingested_orders(projects_dir: Path) -> dict[str, str]:
    """Map of Cloudinary order folder -> local project name, from order.json files."""
    mapping: dict[str, str] = {}
    if not projects_dir.exists():
        return mapping
    for project in sorted(projects_dir.iterdir()):
        record = read_order_record(project / "order.json")
        if record and record.get("order_folder"):
            mapping[record["order_folder"]] = project.name
    return mapping
