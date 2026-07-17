"""Firestore order tracking (the animoments frontend writes one doc per paid order).

The web frontend saves every completed order to the ``orders`` collection of
its Firebase project (see the frontend's ``src/firebase.js``): customer
contact details, the chosen package/music mood/blessing, the Cloudinary
folder its photos upload into, and a ``status`` that starts as ``"new"``.
The photos themselves still live in Cloudinary — Firestore is the order
*ledger*, so the watcher tracks orders here instead of guessing from folder
listings, and writes the pipeline's progress back into each doc's ``status``
(``new -> ingesting -> ingested``) for the admin panel and the storefront.

Access uses the Firestore REST API with a **service-account** credential
(browser security rules don't apply to it):

1. Firebase console -> Project settings -> Service accounts ->
   "Generate new private key"; save the JSON file.
2. Point ``FIREBASE_SERVICE_ACCOUNT`` in .env at it (or drop it at the repo
   root as ``firebase-service-account.json`` — gitignored).

The REST API (instead of the ``firebase-admin`` SDK) keeps the dependency
footprint at ``google-auth`` + the ``requests`` stack the project already
uses, and keeps the JSON encode/decode pure and unit-testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

from ..errors import ConfigError
from ..retry import with_retries
from ..workspace import PROJECT_ROOT

_FIRESTORE_BASE = "https://firestore.googleapis.com/v1"
_SCOPE = "https://www.googleapis.com/auth/datastore"
_PAGE_SIZE = 300
_DEFAULT_CREDENTIALS_FILE = "firebase-service-account.json"

# Order lifecycle written back into each Firestore doc. The frontend creates
# docs as "new"; the pipeline moves them forward. "ingesting" is intentionally
# still considered pending by the watcher (guarded by the local order.json /
# job queue instead), so a failed ingest self-heals on a later poll rather
# than leaving the order stuck.
STATUS_NEW = "new"
STATUS_INGESTING = "ingesting"
STATUS_INGESTED = "ingested"
PENDING_STATUSES = ("", STATUS_NEW, STATUS_INGESTING)


@dataclass(frozen=True)
class FirestoreOrder:
    """One paid order as recorded by the web frontend."""

    order_id: str          # doc id, "AM-..."
    customer: str = ""     # the customer's display name
    phone: str = ""
    email: str = ""
    package_id: str = ""
    music_mood: str = ""
    blessing: str = ""
    folder: str = ""       # Cloudinary path, e.g. "video-orders/<leaf>"
    status: str = ""
    project: str = ""      # local project name, written back by the pipeline
    photo_count: Optional[int] = None
    created_at: str = ""

    @property
    def folder_leaf(self) -> str:
        """The Cloudinary folder's leaf name (what intake/ingest key on)."""
        return self.folder.rstrip("/").rsplit("/", 1)[-1] if self.folder else ""


# ---------------------- pure helpers (unit-tested) ------------------------- #

def decode_value(value: dict[str, Any]) -> Any:
    """One Firestore REST typed value -> a plain Python value."""
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return value["doubleValue"]
    if "booleanValue" in value:
        return value["booleanValue"]
    if "timestampValue" in value:
        return value["timestampValue"]
    if "nullValue" in value:
        return None
    if "mapValue" in value:
        return decode_fields(value["mapValue"].get("fields", {}))
    if "arrayValue" in value:
        return [decode_value(v) for v in value["arrayValue"].get("values", [])]
    return None  # unknown/unsupported type: better absent than wrong


def decode_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {name: decode_value(value) for name, value in fields.items()}


def encode_value(value: Any) -> dict[str, Any]:
    """A plain Python value -> a Firestore REST typed value."""
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, dict):
        return {"mapValue": {"fields": encode_fields(value)}}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [encode_value(v) for v in value]}}
    raise TypeError(f"Cannot encode {type(value).__name__} for Firestore")


def encode_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {name: encode_value(value) for name, value in fields.items()}


def order_from_document(document: dict[str, Any]) -> FirestoreOrder:
    """A Firestore ``orders`` document -> :class:`FirestoreOrder`.

    Tolerant of missing fields (docs written by older frontend versions):
    everything defaults to empty, and the doc id backs up ``orderId``.
    """
    fields = decode_fields(document.get("fields", {}))
    doc_id = document.get("name", "").rsplit("/", 1)[-1]

    def text(key: str) -> str:
        value = fields.get(key)
        return str(value) if value is not None else ""

    photo_count: Optional[int] = None
    raw_count = fields.get("photoCount")
    if isinstance(raw_count, (int, float)) and raw_count > 0:
        photo_count = int(raw_count)

    return FirestoreOrder(
        order_id=text("orderId") or doc_id,
        customer=text("name"),
        phone=text("phone"),
        email=text("email"),
        package_id=text("packageId"),
        music_mood=text("musicMood"),
        blessing=text("blessing"),
        folder=text("folder"),
        status=text("status"),
        project=text("project"),
        photo_count=photo_count,
        created_at=text("createdAt") or document.get("createTime", ""),
    )


def resolve_credentials_file(config: Any) -> Optional[Path]:
    """Where the service-account JSON lives, or None when not set up.

    Precedence: config ``firebase_credentials_file`` > env
    ``FIREBASE_SERVICE_ACCOUNT`` > env ``GOOGLE_APPLICATION_CREDENTIALS`` >
    ``firebase-service-account.json`` at the repo root. Relative paths
    resolve against the repo root.
    """
    candidates = [
        getattr(config, "firebase_credentials_file", ""),
        os.environ.get("FIREBASE_SERVICE_ACCOUNT", ""),
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        _DEFAULT_CREDENTIALS_FILE,
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
    return None


# --------------------------------- client ----------------------------------- #

class FirebaseClient:
    """Minimal Firestore REST client for the ``orders`` collection."""

    def __init__(
        self,
        project_id: str,
        credentials_file: Path,
        *,
        collection: str = "orders",
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self.project_id = project_id
        self.collection = collection
        self._credentials_file = credentials_file
        self._credentials = None  # built lazily; needs google-auth
        self._max_retries = max_retries
        self._base_delay = base_delay

    # ------------------------------- setup --------------------------------- #

    @staticmethod
    def configured(config: Any) -> bool:
        """True when Firestore order tracking can run (credentials present)."""
        return resolve_credentials_file(config) is not None

    @classmethod
    def from_config(cls, config: Any) -> "FirebaseClient":
        credentials_file = resolve_credentials_file(config)
        if credentials_file is None:
            raise ConfigError(
                "Firebase order tracking is not configured. Download a "
                "service-account key (Firebase console -> Project settings -> "
                "Service accounts -> Generate new private key) and point "
                "FIREBASE_SERVICE_ACCOUNT in .env at it, or save it as "
                f"{_DEFAULT_CREDENTIALS_FILE} at the repo root."
            )
        project_id = getattr(config, "firebase_project_id", "")
        if not project_id:
            import json

            try:
                project_id = json.loads(
                    credentials_file.read_text(encoding="utf-8")
                ).get("project_id", "")
            except (OSError, ValueError):
                project_id = ""
        if not project_id:
            raise ConfigError(
                "Cannot determine the Firebase project id — set "
                "firebase_project_id in config.json (the service-account "
                f"file {credentials_file} has no project_id)."
            )
        return cls(
            project_id,
            credentials_file,
            collection=getattr(config, "firebase_orders_collection", "orders"),
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay_seconds,
        )

    # ------------------------------ HTTP core ------------------------------ #

    def _token(self) -> str:
        # Imported here so the module (and its pure helpers/tests) stays
        # usable without google-auth installed.
        from google.auth.transport.requests import Request
        from google.oauth2 import service_account

        if self._credentials is None:
            self._credentials = service_account.Credentials.from_service_account_file(
                str(self._credentials_file), scopes=[_SCOPE]
            )
        if not self._credentials.valid:
            self._credentials.refresh(Request())
        return self._credentials.token

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Any = None,
        body: Optional[dict[str, Any]] = None,
        description: str,
    ) -> dict[str, Any]:
        url = f"{_FIRESTORE_BASE}/projects/{self.project_id}/databases/(default)/documents/{path}"

        def _call() -> dict[str, Any]:
            resp = requests.request(
                method,
                url,
                params=params,
                json=body,
                headers={"Authorization": f"Bearer {self._token()}"},
                timeout=60,
            )
            resp.raise_for_status()  # HTTPError carries .response -> 4xx won't be retried
            return resp.json()

        return with_retries(
            _call,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            description=description,
        )

    # ------------------------------- queries ------------------------------- #

    def list_orders(self) -> list[FirestoreOrder]:
        """Every order document, newest first."""
        documents: list[dict[str, Any]] = []
        page_token: Optional[str] = None
        while True:
            params: dict[str, Any] = {"pageSize": _PAGE_SIZE}
            if page_token:
                params["pageToken"] = page_token
            data = self._request(
                "GET",
                self.collection,
                params=params,
                description=f"list Firestore {self.collection}",
            )
            documents.extend(data.get("documents", []))
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        orders = [order_from_document(doc) for doc in documents]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def update_order(self, order_id: str, fields: dict[str, Any]) -> None:
        """Merge ``fields`` into one order doc (other fields untouched).

        Fails (409/404 from the precondition) rather than upserting when the
        doc doesn't exist — a pre-Firestore order ingested from Cloudinary
        must not leave a stray skeleton doc in the ledger.
        """
        self._request(
            "PATCH",
            f"{self.collection}/{order_id}",
            # A repeated query key — requests takes a list of pairs.
            params=[("updateMask.fieldPaths", name) for name in fields]
            + [("currentDocument.exists", "true")],
            body={"fields": encode_fields(fields)},
            description=f"update Firestore order {order_id}",
        )
