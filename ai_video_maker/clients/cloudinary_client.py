"""Cloudinary order ingestion: find a paid order's photo folder, list its photos.

The animoments web frontend uploads a customer's photos to Cloudinary right
after payment, one folder per order:

    video-orders/<ORDER-ID>_<customer-name>-<dd.mm.yyyy_HH-MM>/
        1.jpg, 2.jpg, ...      # public_id = the photo's position in the movie

Every asset is additionally TAGGED with the folder's leaf name and carries a
``context.order`` value repeating its position, so an order's photos can be
listed by tag — which works in both of Cloudinary's folder modes (legacy
fixed folders and dynamic folders) — with a public_id-prefix listing as the
fallback for old orders that predate tagging.

Auth: the Admin API uses HTTP basic auth with the account's API key/secret,
read from the ``CLOUDINARY_API_KEY`` / ``CLOUDINARY_API_SECRET`` env vars
(.env). The cloud name is public (it appears in the frontend's config) and
lives in config.json (``cloudinary_cloud_name``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests

from ..errors import ConfigError, PipelineError
from ..retry import with_retries

_API_BASE = "https://api.cloudinary.com/v1_1"
_PAGE_SIZE = 500  # Admin API max per page


@dataclass(frozen=True)
class OrderAsset:
    """One customer photo inside a Cloudinary order folder."""

    public_id: str
    url: str
    format: str
    # 1-based position in the movie (from context.order / the public_id's
    # trailing number); None when neither is available.
    position: Optional[int]
    created_at: str = ""


# ------------------------- pure helpers (unit-tested) ------------------------ #

def asset_position(public_id: str, context: Any) -> Optional[int]:
    """The photo's 1-based movie position, from context.order or the public_id.

    The Admin API returns context as ``{"custom": {"order": "3", ...}}``; the
    frontend also names each upload by position, so the public_id's trailing
    number (``video-orders/<folder>/3`` or just ``3``) is the fallback.
    """
    if isinstance(context, dict):
        custom = context.get("custom", context)
        value = custom.get("order") if isinstance(custom, dict) else None
        if value is not None:
            try:
                return int(str(value).strip())
            except ValueError:
                pass
    tail = re.search(r"(\d+)$", public_id.rsplit("/", 1)[-1])
    return int(tail.group(1)) if tail else None


def sort_assets(assets: list[OrderAsset]) -> list[OrderAsset]:
    """Movie order: by position; unknown positions last, by upload time."""
    return sorted(
        assets,
        key=lambda a: (a.position is None, a.position or 0, a.created_at, a.public_id),
    )


def resolve_order_folder(query: str, folders: list[str]) -> str:
    """Match an order reference against the order folder leaf names.

    Accepts the exact folder name, a unique prefix (the natural case: the
    order id ``AM-...`` from the confirmation email), or a unique
    case-insensitive substring (e.g. the customer's name). Ambiguity and
    no-match both fail loudly with the candidates listed.
    """
    if query in folders:
        return query
    matches = [f for f in folders if f.startswith(query)]
    if not matches:
        matches = [f for f in folders if query.lower() in f.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PipelineError(
            f"No Cloudinary order folder matches '{query}'.\n"
            "List the available orders with:  python pipeline.py orders"
        )
    listing = "\n".join(f"  {name}" for name in matches)
    raise PipelineError(
        f"'{query}' matches {len(matches)} order folders — be more specific:\n{listing}"
    )


def ingest_filename(sequence: int, total: int, fmt: str) -> str:
    """Local filename for the photo at 1-based `sequence` of `total`.

    Zero-padded so ``list_input_images``'s ordering matches the customer's
    chosen photo order, with the width sized to the order (min 2 digits).
    """
    width = max(2, len(str(total)))
    ext = (fmt or "jpg").strip().lower().lstrip(".")
    return f"{sequence:0{width}d}.{ext}"


# --------------------------------- client ----------------------------------- #

class CloudinaryClient:
    """Thin Admin-API client for listing and resolving order folders."""

    def __init__(
        self,
        cloud_name: str,
        api_key: str,
        api_secret: str,
        *,
        orders_folder: str = "video-orders",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self.cloud_name = cloud_name
        self.orders_folder = orders_folder.strip("/")
        self._auth = (api_key, api_secret)
        self._max_retries = max_retries
        self._base_delay = base_delay

    @classmethod
    def from_config(cls, config: Any) -> "CloudinaryClient":
        cloud = config.cloudinary_cloud_name or os.environ.get("CLOUDINARY_CLOUD_NAME", "")
        api_key = os.environ.get("CLOUDINARY_API_KEY", "")
        api_secret = os.environ.get("CLOUDINARY_API_SECRET", "")
        missing = [
            name
            for name, value in [
                ("cloudinary_cloud_name (config.json) or CLOUDINARY_CLOUD_NAME", cloud),
                ("CLOUDINARY_API_KEY", api_key),
                ("CLOUDINARY_API_SECRET", api_secret),
            ]
            if not value
        ]
        if missing:
            raise ConfigError(
                "Cloudinary ingestion is not configured — missing: "
                + ", ".join(missing)
                + ". The API key/secret go in .env (Cloudinary console -> "
                "Settings -> API Keys)."
            )
        return cls(
            cloud,
            api_key,
            api_secret,
            orders_folder=config.cloudinary_orders_folder,
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay_seconds,
        )

    # ------------------------------ HTTP core ------------------------------ #

    def _get(self, path: str, params: dict[str, Any], description: str) -> dict[str, Any]:
        url = f"{_API_BASE}/{self.cloud_name}/{path}"

        def _call() -> dict[str, Any]:
            resp = requests.get(url, params=params, auth=self._auth, timeout=60)
            resp.raise_for_status()  # HTTPError carries .response -> 4xx won't be retried
            return resp.json()

        return with_retries(
            _call,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            description=description,
        )

    def _paged(
        self, path: str, params: dict[str, Any], list_key: str, description: str
    ) -> list[dict[str, Any]]:
        """Collect every page of an Admin-API listing (next_cursor pagination)."""
        items: list[dict[str, Any]] = []
        cursor: Optional[str] = None
        while True:
            page_params = dict(params, max_results=_PAGE_SIZE)
            if cursor:
                page_params["next_cursor"] = cursor
            data = self._get(path, page_params, description)
            items.extend(data.get(list_key, []))
            cursor = data.get("next_cursor")
            if not cursor:
                return items

    # ------------------------------- queries ------------------------------- #

    def list_order_folders(self) -> list[str]:
        """Leaf names of every order folder, newest first (names embed a date)."""
        folders = self._paged(
            f"folders/{self.orders_folder}",
            {},
            "folders",
            f"list Cloudinary folders under {self.orders_folder}/",
        )
        return sorted((f["name"] for f in folders), reverse=True)

    def list_order_assets(self, folder_leaf: str) -> list[OrderAsset]:
        """Every photo in one order folder, in movie order.

        Primary: list by tag (the frontend tags each upload with the folder
        leaf; works in both folder modes). Fallback: list by public_id prefix
        (legacy fixed-folder mode, for orders that predate tagging).
        """
        raw = self._paged(
            f"resources/image/tags/{requests.utils.quote(folder_leaf, safe='')}",
            {"context": "true"},
            "resources",
            f"list Cloudinary assets tagged {folder_leaf}",
        )
        if not raw:
            raw = self._paged(
                "resources/image",
                {
                    "type": "upload",
                    "prefix": f"{self.orders_folder}/{folder_leaf}/",
                    "context": "true",
                },
                "resources",
                f"list Cloudinary assets under {self.orders_folder}/{folder_leaf}/",
            )
        assets = [
            OrderAsset(
                public_id=r["public_id"],
                url=r.get("secure_url") or r.get("url", ""),
                format=r.get("format", ""),
                position=asset_position(r["public_id"], r.get("context")),
                created_at=r.get("created_at", ""),
            )
            for r in raw
        ]
        return sort_assets(assets)
