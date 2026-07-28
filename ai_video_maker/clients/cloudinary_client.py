"""Cloudinary order intake and delivery.

*Intake* — the animoments web frontend uploads a customer's photos to
Cloudinary right after payment, one folder per order:

    video-orders/<ORDER-ID>_<customer-name>-<dd.mm.yyyy_HH-MM>/
        1.jpg, 2.jpg, ...      # public_id = the photo's position in the movie

Every asset is additionally TAGGED with the folder's leaf name and carries a
``context.order`` value repeating its position, so an order's photos can be
listed by tag — which works in both of Cloudinary's folder modes (legacy
fixed folders and dynamic folders) — with a public_id-prefix listing as the
fallback for old orders that predate tagging.

*Delivery* — the finished movie is uploaded back into that same order folder
(``publish_final_video``), named ``final_v1``, ``final_v2``, ... Publishing is
strictly ADDITIVE: the version number is one past the highest already there,
every upload carries ``overwrite=false``, and a name that turns out to be
taken fails loudly instead of replacing anything.

Auth: the Admin API uses HTTP basic auth with the account's API key/secret,
read from the ``CLOUDINARY_API_KEY`` / ``CLOUDINARY_API_SECRET`` env vars
(.env); the Upload API is signed with the same secret. The cloud name is
public (it appears in the frontend's config) and lives in config.json
(``cloudinary_cloud_name``).
"""
from __future__ import annotations

import hashlib
import os
import re
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

from ..errors import ConfigError, PipelineError
from ..logging_setup import logger
from ..retry import with_retries

_API_BASE = "https://api.cloudinary.com/v1_1"
_PAGE_SIZE = 500  # Admin API max per page

# Uploads are sent in chunks: Cloudinary caps a single upload request (100 MB
# on most plans) and a finished movie can be bigger, plus a chunked upload
# survives a wobbly connection far better than one long POST.
_CHUNK_BYTES = 20 * 1024 * 1024


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


@dataclass(frozen=True)
class PublishedVideo:
    """One movie version already published into an order folder."""

    public_id: str
    url: str
    version: int          # the N in final_vN
    bytes: int = 0
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


def publish_version(public_id: str, basename: str) -> Optional[int]:
    """The N in a published ``<basename>_vN`` public_id, else None.

    Only the leaf is examined, so it works in both folder modes (the public_id
    carries the folder path in fixed mode and may not in dynamic mode).
    """
    leaf = public_id.rsplit("/", 1)[-1]
    m = re.fullmatch(rf"{re.escape(basename)}_v(\d+)", leaf)
    return int(m.group(1)) if m else None


def next_publish_version(existing: Iterable[int]) -> int:
    """One past the highest version seen — never reuses a number.

    Takes the MAXIMUM rather than the count, so a gap (an old version deleted
    by hand in the Cloudinary console) can never make the next publish land on
    a name that was already used.
    """
    versions = [v for v in existing if isinstance(v, int) and v > 0]
    return max(versions, default=0) + 1


def publish_public_id(
    orders_folder: str, folder_leaf: str, basename: str, version: int
) -> str:
    """Full public_id for a published movie: ``video-orders/<leaf>/final_vN``.

    The path is kept inside the public_id in both folder modes so the delivery
    URL always shows which order the movie belongs to; in dynamic-folder mode
    ``asset_folder`` additionally files it under the order folder in the Media
    Library (see :meth:`CloudinaryClient.publish_final_video`).
    """
    return f"{orders_folder.strip('/')}/{folder_leaf.strip('/')}/{basename}_v{version}"


def upload_signature(params: dict[str, Any], api_secret: str) -> str:
    """Cloudinary's SHA-1 upload signature over the request's own parameters.

    Every parameter that is sent must be signed except ``file``, ``api_key``,
    ``resource_type`` and ``cloud_name`` (they are part of the URL or not
    signed by design) — sorted by name, joined as a query string, with the
    API secret appended.
    """
    signed = {
        k: v for k, v in params.items()
        if k not in ("file", "api_key", "resource_type", "cloud_name", "signature")
        and v not in (None, "")
    }
    payload = "&".join(f"{k}={signed[k]}" for k in sorted(signed))
    return hashlib.sha1(f"{payload}{api_secret}".encode("utf-8")).hexdigest()


def chunk_ranges(total: int, chunk_size: int = _CHUNK_BYTES) -> list[tuple[int, int]]:
    """Inclusive (start, end) byte ranges covering a file of `total` bytes.

    Cloudinary's chunked upload wants ``Content-Range: bytes start-end/total``
    with an INCLUSIVE end. An empty file still yields one (0, 0) range so the
    caller always makes exactly one request per range.
    """
    if total <= 0:
        return [(0, 0)]
    return [
        (start, min(start + chunk_size, total) - 1)
        for start in range(0, total, chunk_size)
    ]


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
        publish_basename: str = "final",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self.cloud_name = cloud_name
        self.orders_folder = orders_folder.strip("/")
        self.publish_basename = publish_basename
        self._api_key = api_key
        self._api_secret = api_secret
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
            publish_basename=config.cloudinary_publish_basename,
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

    # ------------------------------ delivery ------------------------------- #

    def _paged_allowing_404(
        self, path: str, params: dict[str, Any], description: str
    ) -> list[dict[str, Any]]:
        """Listing that treats "no such tag/prefix" as an empty result.

        A folder nobody has published into yet has no video tag at all, and
        Cloudinary answers 404 rather than an empty list. Any other failure
        still propagates: publishing derives the next version number from
        these listings, so a silent error must not look like "nothing there".
        """
        try:
            return self._paged(path, params, "resources", description)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return []
            raise

    def list_published_videos(self, folder_leaf: str) -> list[PublishedVideo]:
        """Movie versions already published into one order folder, oldest first.

        Listed by tag AND by public_id prefix, unioned: the tag listing is the
        one that works in dynamic-folder mode, the prefix listing covers a
        version whose tag was removed in the console. Missing one of these
        would make the next publish reuse a version number, so both are asked
        and the highest wins.
        """
        found: dict[str, dict[str, Any]] = {}
        for raw in (
            self._paged_allowing_404(
                f"resources/video/tags/{requests.utils.quote(folder_leaf, safe='')}",
                {},
                f"list published movies tagged {folder_leaf}",
            )
            + self._paged_allowing_404(
                "resources/video",
                {
                    "type": "upload",
                    "prefix": f"{self.orders_folder}/{folder_leaf}/",
                },
                f"list published movies under {self.orders_folder}/{folder_leaf}/",
            )
        ):
            found[raw["public_id"]] = raw

        published = []
        for public_id, raw in found.items():
            version = publish_version(public_id, self.publish_basename)
            if version is None:
                continue  # something else living in the folder — not ours
            published.append(PublishedVideo(
                public_id=public_id,
                url=raw.get("secure_url") or raw.get("url", ""),
                version=version,
                bytes=int(raw.get("bytes") or 0),
                created_at=raw.get("created_at", ""),
            ))
        return sorted(published, key=lambda p: p.version)

    def folder_mode(self) -> str:
        """The cloud's folder mode: "fixed" or "dynamic".

        It decides how an upload is filed: in fixed mode the folder is part of
        the public_id, in dynamic mode the two are decoupled and the folder
        travels as ``asset_folder``. Best-effort — an account whose key may not
        read settings falls back to "fixed", which still lands the asset at the
        right delivery URL.
        """
        try:
            data = self._get("config", {"settings": "true"}, "read Cloudinary settings")
        except Exception as exc:  # noqa: BLE001 - never block a publish on this
            logger.warning(
                "Could not read the Cloudinary folder mode (%s) — assuming fixed.",
                exc,
            )
            return "fixed"
        settings = data.get("settings") if isinstance(data, dict) else None
        mode = (settings or {}).get("folder_mode", "") if isinstance(settings, dict) else ""
        return str(mode).strip().lower() or "fixed"

    def publish_final_video(
        self, folder_leaf: str, path: Path, version: int
    ) -> PublishedVideo:
        """Upload the finished movie into an order folder as ``final_vN``.

        Additive by construction: the public_id carries an explicit version
        the caller derived from what is already there, and ``overwrite=false``
        means Cloudinary refuses to replace an existing asset rather than
        silently versioning over it. If that name IS taken (someone published
        from another machine in between), the upload returns the existing
        asset untouched — detected here and raised, so nothing is ever lost.
        """
        if not path.is_file():
            raise PipelineError(f"There is no movie to publish at {path}")
        public_id = publish_public_id(
            self.orders_folder, folder_leaf, self.publish_basename, version
        )
        params: dict[str, Any] = {
            "public_id": public_id,
            "overwrite": "false",
            # Same tag the frontend puts on the order's photos, so the whole
            # order (photos + delivered movies) lists together.
            "tags": folder_leaf,
            "context": f"order_folder={folder_leaf}|kind=final_movie",
            "timestamp": str(int(time.time())),
        }
        if self.folder_mode() == "dynamic":
            # Public_id and folder are decoupled in this mode: without these
            # the movie would be delivered from the right URL but sit at the
            # root of the Media Library instead of inside the order.
            params["asset_folder"] = f"{self.orders_folder}/{folder_leaf}"
            params["display_name"] = f"{self.publish_basename}_v{version}"
        params["signature"] = upload_signature(params, self._api_secret)
        params["api_key"] = self._api_key

        result = self._upload_chunked(path, params)
        if result.get("existing"):
            raise PipelineError(
                f"Cloudinary already has an asset named {public_id} — nothing "
                "was uploaded or replaced. Publish again to take the next free "
                "version number."
            )
        return PublishedVideo(
            public_id=str(result.get("public_id") or public_id),
            url=str(result.get("secure_url") or result.get("url") or ""),
            version=version,
            bytes=int(result.get("bytes") or path.stat().st_size),
            created_at=str(result.get("created_at") or ""),
        )

    def _upload_chunked(self, path: Path, params: dict[str, Any]) -> dict[str, Any]:
        """POST a file to the Upload API in chunks; returns the final response.

        Every chunk repeats the same signed parameters and the same
        ``X-Unique-Upload-Id``; Cloudinary assembles them and answers the last
        chunk with the finished asset. Each chunk is retried on its own, which
        is the point of chunking a movie-sized upload at all.
        """
        url = f"{_API_BASE}/{self.cloud_name}/video/upload"
        total = path.stat().st_size
        upload_id = secrets.token_hex(16)
        ranges = chunk_ranges(total)
        result: dict[str, Any] = {}

        with path.open("rb") as handle:
            for index, (start, end) in enumerate(ranges, start=1):
                handle.seek(start)
                blob = handle.read(end - start + 1)

                def _call(blob: bytes = blob, start: int = start, end: int = end):
                    resp = requests.post(
                        url,
                        data=params,
                        files={"file": (path.name, blob, "video/mp4")},
                        headers={
                            "X-Unique-Upload-Id": upload_id,
                            "Content-Range": f"bytes {start}-{end}/{total}",
                        },
                        timeout=600,
                    )
                    resp.raise_for_status()
                    return resp.json()

                if len(ranges) > 1:
                    logger.info(
                        "Uploading chunk %d/%d (%.1f MB)…",
                        index, len(ranges), len(blob) / (1024 * 1024),
                    )
                result = with_retries(
                    _call,
                    max_retries=self._max_retries,
                    base_delay=self._base_delay,
                    description=f"upload {path.name} to Cloudinary",
                )
        if not isinstance(result, dict) or not (
            result.get("secure_url") or result.get("url")
        ):
            raise PipelineError(
                f"Cloudinary accepted the upload but returned no asset URL: {result}"
            )
        return result
