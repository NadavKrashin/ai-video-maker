"""FastAPI admin server: drive the pipeline remotely (the admin panel).

One process serves four things:

* the **admin panel** — the browser UI in ``admin_ui/`` (its ``dist/`` build
  is mounted at ``/`` when present, so http://host:8300/ is the panel);
* the **admin API** — orders, per-project status (``Pipeline.snapshot()``),
  storyboard read/edit, media files, photo upload, and actions (ingest/
  storyboard/render/audio/combine/publish/run) that run as background jobs;
* the **job runner** — a single worker thread executing one pipeline command
  at a time (an order's steps take minutes and the volume is orders-per-day,
  so serial keeps things simple and safe);
* the **watcher** — tracks new paid orders and auto-ingests (+ optionally
  storyboards) the complete ones, so a new order needs no PC interaction at
  all. With Firebase configured (service-account key, see
  ``clients/firebase_client.py``) the Firestore ``orders`` collection is the
  source of truth and pipeline progress is written back into each order's
  ``status``; otherwise it falls back to polling Cloudinary folders.

Interactivity: pipeline confirm gates auto-proceed here (the API caller made
the decision by pressing the button); nothing blocks on stdin.

Auth: every /api route (except /api/health) requires the ``ADMIN_API_TOKEN``
env value as ``Authorization: Bearer <token>``. Only the media file route
additionally accepts ``?token=<token>`` — ``<img>``/``<video>`` tags can't
send headers. Comparisons are constant-time and repeated failures from one
address are throttled (see ``_AuthThrottle``).
"""
from __future__ import annotations

import dataclasses
import logging
import os
import queue
import secrets
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .clients.cloudinary_client import CloudinaryClient
from .clients.firebase_client import (
    PENDING_STATUSES,
    STATUS_INGESTED,
    STATUS_INGESTING,
    FirebaseClient,
)
from .config import Config
from .costs import KIND_LABELS, merge_totals
from .errors import (
    InvalidProjectName,
    PipelineCancelled,
    PipelineError,
    StoryboardError,
)
from .feedback import (
    MAX_NOTE_CHARS,
    SCOPES,
    VERDICTS,
    FeedbackStore,
    LessonStore,
)
from .intake import (
    derive_project_name,
    ingested_orders,
    is_order_complete,
    parse_order_folder,
    read_order_record,
)
from .logging_setup import logger, setup_logging
from .media.audio_files import looks_like_audio
from .media.letter import read_letter, save_letter
from .media.music_url import fetch_music
from .models import Storyboard, changed_transition_ids
from .options import RunOptions
from .runner import Pipeline
from .workspace import (
    PROJECT_ROOT,
    PROJECTS_DIR,
    Workspace,
    is_project_dir,
    lessons_file,
)

# Pipeline commands the API may enqueue ("ingest" only via /api/orders/ingest),
# and the RunOptions fields a request body may set — every per-run knob the
# CLI has, but still an explicit whitelist so a request can't reach for
# constructor internals.
_ALLOWED_COMMANDS = {
    "ingest", "storyboard", "render", "audio", "combine", "publish", "run",
    # `tag` proposes who is in each frame; it writes only frames[].people,
    # never a transition, so it can't invalidate a rendered clip.
    "tag",
}
_ALLOWED_OPTIONS = {f.name for f in dataclasses.fields(RunOptions)}

# Uploads into input_images/ — the same formats a user would drop there.
_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}

# Custom music-bed uploads are recognised by content (media/audio_files.py):
# ffmpeg reads by content, not extension, and the bytes are stored under a
# fixed name regardless of what was uploaded.

# Per-file cap for photo uploads. Phone photos top out around 15-20 MB; this
# only exists so an unauthenticated-looking-but-authenticated mistake (or a
# stolen token) can't fill the disk through the panel.
_MAX_UPLOAD_BYTES = 40 * 1024 * 1024

# The closing letter is a few paragraphs read off the screen; anything near
# this is already far more than can scroll past in a movie's ending, so the
# cap only stops the text box being used as a file dump.
_MAX_LETTER_CHARS = 20_000

# The token is the only thing between the internet and the pipeline once the
# server sits behind a tunnel — refuse to boot with a guessable one.
_MIN_TOKEN_LENGTH = 16


class _AuthThrottle:
    """Lock out an address after repeated bad tokens (online brute force).

    In-memory and deliberately simple: `max_failures` bad attempts within
    `window_seconds` → 429 for the remainder of the window. A correct token
    clears the address. Behind cloudflared every TCP peer is 127.0.0.1, so the
    caller passes the CF-Connecting-IP header value when present (fine to
    trust: the only way to reach the port from outside IS the tunnel).
    """

    def __init__(self, max_failures: int = 10, window_seconds: float = 900.0) -> None:
        self._max = max_failures
        self._window = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> list[float]:
        kept = [t for t in self._failures.get(key, []) if now - t < self._window]
        if kept:
            self._failures[key] = kept
        else:
            self._failures.pop(key, None)
        return kept

    def blocked(self, key: str) -> bool:
        with self._lock:
            return len(self._prune(key, time.monotonic())) >= self._max

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = time.monotonic()
            self._failures[key] = self._prune(key, now) + [now]

    def clear(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _RedactQueryStrings(logging.Filter):
    """Strip query strings from uvicorn's access log.

    The media route accepts ``?token=`` because <img>/<video> tags cannot send
    headers — which meant every image request wrote the full ADMIN_API_TOKEN
    into the access log in clear text. Anyone able to read the log file (or a
    backup of it) had full admin access. The path is all that is worth
    logging, so the query is dropped wholesale rather than pattern-matched:
    no future query parameter can leak through by being forgotten here.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            path = args[2]
            if "?" in path:
                record.args = (
                    *args[:2], path.split("?", 1)[0] + "?<redacted>", *args[3:],
                )
        return True


def _install_access_log_redaction() -> None:
    """Attach the query-string redactor to uvicorn's access logger (once)."""
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _RedactQueryStrings) for f in access.filters):
        access.addFilter(_RedactQueryStrings())


def _outcome(pipeline: Pipeline) -> tuple[str, str, int]:
    """Turn a completed run into (state, error, failure count).

    A pipeline command deliberately survives individual failures — one clip
    hitting a timeout must not abandon the other nineteen — so `execute()`
    returning is NOT proof the work succeeded. Reporting it as "done"
    regardless made a render whose every clip timed out look exactly like a
    successful one. Anything that produced output alongside failures is
    "partial"; a run that produced nothing at all is "failed".
    """
    failures = list(pipeline.failed.failures)
    if not failures:
        return "done", "", 0
    s = pipeline.summary
    produced = s.styled_created + s.videos_created + s.sfx_created
    kinds: dict[str, int] = {}
    for f in failures:
        kind = f.get("kind", "item")
        kinds[kind] = kinds.get(kind, 0) + 1
    breakdown = ", ".join(f"{n} {kind}" for kind, n in sorted(kinds.items()))
    first = str(failures[0].get("error", ""))[:200]
    return (
        "partial" if produced else "failed",
        f"{len(failures)} failed ({breakdown}). First error: {first}",
        len(failures),
    )


def _project_contents(ws: Workspace) -> dict[str, int]:
    """What a project holds, counted for a deletion's confirmation and log.

    Deliberately counts the things that cost money or cannot be rebuilt —
    styled frames and clips are paid API output, and a delivered movie's
    local archive is the only copy of bytes a customer already has. Cheap
    (a few directory listings), because it runs on the way to `rmtree` and
    is the last chance to say what is about to be destroyed.
    """
    def _count(directory: Path, suffixes: tuple[str, ...]) -> int:
        if not directory.is_dir():
            return 0
        return sum(1 for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in suffixes)

    return {
        "photos": _count(ws.input_images_dir, (".jpg", ".jpeg", ".png", ".webp",
                                               ".heic", ".heif", ".gif", ".bmp")),
        "styled frames": _count(ws.styled_images_dir, (".png", ".jpg", ".jpeg")),
        "rendered clips": _count(ws.clips_dir, (".mp4",)),
        "final movies": int(ws.final_video.is_file()),
        "delivered copies": _count(ws.published_dir, (".mp4",)),
    }


def apply_in_flight(
    pending: list[dict[str, Any]], active_options: Optional[dict[str, Any]]
) -> None:
    """Mark which pending renders a currently-running job already owns.

    A ``falreq:`` entry only records "submitted, not yet downloaded" — which
    is equally true of a run that crashed an hour ago and of the render
    polling fal right now. Only the job runner knows the difference, so it
    is applied here: a clip the active job is handling collects itself and
    is mere progress, while one left behind by an interrupted run is money
    stranded on the provider that only the user can rescue. Reporting both
    the same way turned every healthy render into a false alarm.

    `active_options` is the running render/run job's options, or None when
    no such job is running. A job given an explicit ``clips`` list owns only
    those; one rendering the whole project may own any pending clip.
    """
    owned = (active_options or {}).get("clips")
    for entry in pending:
        entry["in_flight"] = (
            active_options is not None and (not owned or entry["id"] in owned)
        )


# --------------------------------- jobs ------------------------------------- #

@dataclass
class Job:
    id: str
    project: str
    command: str
    options: dict[str, Any]
    # queued | running | cancelling | done | partial | failed | cancelled
    # "partial" = the command ran to the end but some work items failed. The
    # pipeline deliberately keeps going when one clip fails, so without this
    # a render that produced nothing still reported "done" and read as
    # success — a real 3-of-3 timeout looked like a completed render.
    state: str = "queued"
    error: str = ""
    # How many individual items (clips, frames, SFX) failed inside the run.
    failures: int = 0
    log: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    started_at: str = ""
    finished_at: str = ""
    # Enqueued on success — how "ingest then storyboard" chains.
    then: Optional[dict[str, Any]] = None
    # Cooperative cancel flag, shared with the Pipeline while running: set ->
    # the pipeline stops between work items (in-flight API calls finish and
    # their outputs are kept, so a later re-run resumes instead of re-paying).
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "project": self.project, "command": self.command,
            "state": self.state, "error": self.error,
            "failures": self.failures,
            "created_at": self.created_at, "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class _JobLogHandler(logging.Handler):
    """Mirror pipeline log lines into the running job (single-worker safe)."""

    def __init__(self, job: Job, max_lines: int = 1000) -> None:
        super().__init__(level=logging.INFO)
        self._job = job
        self._max = max_lines

    def emit(self, record: logging.LogRecord) -> None:
        self._job.log.append(self.format(record))
        if len(self._job.log) > self._max:
            del self._job.log[: len(self._job.log) - self._max]


class JobRunner:
    """Serial background executor for pipeline commands."""

    def __init__(self, config_path: Path, *, start: bool = True) -> None:
        self._config_path = config_path
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._loop, name="job-runner", daemon=True
        )
        if start:  # tests exercise the queueing logic without a worker
            self._thread.start()

    def enqueue(
        self,
        project: str,
        command: str,
        options: dict[str, Any],
        then: Optional[dict[str, Any]] = None,
    ) -> Job:
        if command not in _ALLOWED_COMMANDS:
            raise PipelineError(f"Command not allowed here: {command}")
        options = {k: v for k, v in options.items() if k in _ALLOWED_OPTIONS}
        with self._lock:
            duplicate = next(
                (j for j in self._jobs.values()
                 if j.project == project and j.command == command
                 and j.state in ("queued", "running")), None,
            )
            if duplicate:
                return duplicate  # idempotent: a double-click doesn't double-run
            job = Job(id=uuid.uuid4().hex[:12], project=project,
                      command=command, options=options, then=then)
            self._jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> Optional[Job]:
        """Request cancellation; returns the job, or None if unknown.

        A queued job is cancelled immediately (the worker skips it when it
        reaches the queue). A running job flips to "cancelling" and its
        cancel_event tells the pipeline to stop between work items — the item
        currently generating finishes and is kept. Finished jobs are left
        untouched (cancelling them is meaningless).
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.state == "queued":
                job.cancel_event.set()
                job.state = "cancelled"
                job.finished_at = _now()
            elif job.state == "running":
                job.cancel_event.set()
                job.state = "cancelling"
            return job

    def list(self, project: Optional[str] = None, limit: int = 50) -> list[Job]:
        with self._lock:
            jobs = [
                j for j in self._jobs.values()
                if project is None or j.project == project
            ]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)[:limit]

    def active_ingest_orders(self) -> set[str]:
        """Order folders with a queued/running ingest — the watcher must not re-queue."""
        with self._lock:
            return {
                j.options.get("order", "")
                for j in self._jobs.values()
                # "cancelling" is still running — the watcher must not
                # re-queue its order until the worker actually lets go.
                if j.command == "ingest"
                and j.state in ("queued", "running", "cancelling")
            }

    def _loop(self) -> None:
        while True:
            self._run_job(self._queue.get())

    def _run_job(self, job: Job) -> None:
        with self._lock:
            if job.state != "queued":  # cancelled while waiting in the queue
                return
            job.state, job.started_at = "running", _now()
        handler = _JobLogHandler(job)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        try:
            workspace = Workspace.for_project(job.project)
            workspace.mkdirs()  # ingest bootstraps new projects
            setup_logging(workspace)
            logger.addHandler(handler)
            config = Config.load(
                self._config_path,
                override_path=workspace.root / "config.json",
            )
            pipeline = Pipeline(
                config, workspace, RunOptions(**job.options),
                cancel_event=job.cancel_event,
            )  # default confirm: always proceed
            pipeline.execute(job.command)
            job.state, job.error, job.failures = _outcome(pipeline)
            if job.state != "done":
                logger.error(
                    "Job %s (%s %s) finished with failures: %s",
                    job.id, job.command, job.project, job.error,
                )
        except PipelineCancelled as exc:
            job.state, job.error = "cancelled", str(exc)
            logger.info("Job %s (%s %s) cancelled.",
                        job.id, job.command, job.project)
        except Exception as exc:  # noqa: BLE001 - jobs must never kill the worker
            job.state, job.error = "failed", str(exc)
            logger.error("Job %s (%s %s) failed: %s",
                         job.id, job.command, job.project, exc)
        finally:
            job.finished_at = _now()
            logger.removeHandler(handler)
        if job.state == "done" and job.then and not job.cancel_event.is_set():
            self.enqueue(
                job.then.get("project", job.project),
                job.then["command"],
                job.then.get("options", {}),
            )


# -------------------------------- watcher ----------------------------------- #

class OrderWatcher:
    """Track paid orders; auto-ingest (+ storyboard) the complete new ones.

    Firestore is the order source when a service-account key is configured
    (the doc is written the moment the customer pays — the authoritative
    signal), with Cloudinary consulted only to check the photos' upload
    progress; pipeline progress is written back into each order doc's
    ``status``. Without Firebase the legacy pure-Cloudinary folder poll runs.

    The client factories are injectable for tests.
    """

    def __init__(
        self,
        config: Config,
        config_path: Path,
        jobs: JobRunner,
        *,
        cloudinary_factory=CloudinaryClient.from_config,
        firebase_factory=FirebaseClient.from_config,
    ) -> None:
        self._config = config
        self._config_path = config_path
        self._jobs = jobs
        self._cloudinary_factory = cloudinary_factory
        self._firebase_factory = firebase_factory
        self._thread = threading.Thread(
            target=self._loop, name="order-watcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        logger.info(
            "Order watcher: polling %s every %ds (quiet period %.0f min).",
            "Firestore" if FirebaseClient.configured(self._config) else "Cloudinary",
            self._config.watch_poll_seconds, self._config.watch_quiet_minutes,
        )
        while True:
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - keep watching through outages
                logger.warning("Order watcher poll failed: %s", exc)
            threading.Event().wait(self._config.watch_poll_seconds)

    def poll_once(self) -> list[str]:
        """One poll; returns the order folders enqueued for ingestion."""
        if FirebaseClient.configured(self._config):
            return self._poll_firestore()
        return self._poll_cloudinary()

    # ------------------------- shared poll pieces -------------------------- #

    def _handled_and_names(self) -> tuple[dict[str, str], set[str], set[str]]:
        handled = ingested_orders(PROJECTS_DIR)  # folder leaf -> project
        active = self._jobs.active_ingest_orders()
        existing_names = {
            p.name for p in PROJECTS_DIR.iterdir() if is_project_dir(p)
        } if PROJECTS_DIR.exists() else set()
        return handled, active, existing_names

    def _enqueue_ingest(self, folder: str, existing_names: set[str]) -> str:
        project = derive_project_name(folder, existing_names)
        existing_names.add(project)
        then = (
            {"command": "storyboard", "options": {}}
            if self._config.watch_auto_storyboard else None
        )
        self._jobs.enqueue(project, "ingest", {"order": folder}, then=then)
        logger.info(
            "Order %s: complete — ingesting as project '%s'%s.",
            folder, project, " + storyboard" if then else "",
        )
        return project

    # ------------------------------ sources -------------------------------- #

    def _poll_firestore(self) -> list[str]:
        firebase = self._firebase_factory(self._config)
        cloudinary = self._cloudinary_factory(self._config)
        handled, active, existing_names = self._handled_and_names()

        enqueued: list[str] = []
        for order in firebase.list_orders():
            leaf = order.folder_leaf
            if not leaf:
                continue  # no photo folder recorded — nothing to ingest yet
            if leaf in handled:
                # Ingested locally (by the watcher, the panel, or the CLI) —
                # make sure the ledger says so. Only pending statuses are
                # bumped: a later stage (e.g. a future "delivered") must
                # never be downgraded back to "ingested".
                if order.status in PENDING_STATUSES:
                    self._update_status(
                        firebase, order.order_id,
                        {"status": STATUS_INGESTED, "project": handled[leaf]},
                    )
                continue
            if order.status not in PENDING_STATUSES or leaf in active:
                continue
            assets = cloudinary.list_order_assets(leaf)
            if not is_order_complete(
                assets, self._config.watch_quiet_minutes,
                expected_count=order.photo_count,
            ):
                logger.info(
                    "Order %s: %d photo(s) but upload still fresh — waiting.",
                    leaf, len(assets),
                )
                continue
            self._enqueue_ingest(leaf, existing_names)
            self._update_status(
                firebase, order.order_id, {"status": STATUS_INGESTING}
            )
            enqueued.append(leaf)
        return enqueued

    def _poll_cloudinary(self) -> list[str]:
        client = self._cloudinary_factory(self._config)
        handled, active, existing_names = self._handled_and_names()
        skip = set(handled) | active

        enqueued: list[str] = []
        for folder in client.list_order_folders():
            if folder in skip:
                continue
            assets = client.list_order_assets(folder)
            if not is_order_complete(assets, self._config.watch_quiet_minutes):
                logger.info(
                    "Order %s: %d photo(s) but upload still fresh — waiting.",
                    folder, len(assets),
                )
                continue
            self._enqueue_ingest(folder, existing_names)
            enqueued.append(folder)
        return enqueued

    @staticmethod
    def _update_status(firebase, order_id: str, fields: dict[str, Any]) -> None:
        # A ledger write-back must never break ingestion itself.
        try:
            firebase.update_order(order_id, fields)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not update Firestore order %s: %s", order_id, exc)


# ---------------------------------- app ------------------------------------- #

def create_app(config_path: Path, *, watch: bool = True) -> FastAPI:
    token = os.environ.get("ADMIN_API_TOKEN", "")
    if not token:
        raise PipelineError(
            "ADMIN_API_TOKEN is not set. Add a long random value to .env — "
            "the admin panel authenticates with it."
        )
    if len(token) < _MIN_TOKEN_LENGTH:
        raise PipelineError(
            f"ADMIN_API_TOKEN is too short ({len(token)} chars, minimum "
            f"{_MIN_TOKEN_LENGTH}). It is the only credential on this API — "
            "generate a strong one: python3 -c \"import secrets; "
            "print(secrets.token_urlsafe(32))\""
        )
    config = Config.load(config_path)
    jobs = JobRunner(config_path)
    throttle = _AuthThrottle()
    # Before anything can serve a request that carries ?token=.
    _install_access_log_redaction()

    app = FastAPI(title="ai-video-maker admin API", docs_url=None, redoc_url=None,
                  openapi_url=None)
    if config.admin_cors_origins:  # same-origin panel needs no CORS at all
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.admin_cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def _client_key(request: Request) -> str:
        # Behind cloudflared the TCP peer is always localhost; the tunnel adds
        # the real address. Direct (LAN) hits fall back to the socket peer.
        return (
            request.headers.get("cf-connecting-ip")
            or (request.client.host if request.client else "unknown")
        )

    def _check_token(request: Request, supplied: str) -> None:
        key = _client_key(request)
        if throttle.blocked(key):
            raise HTTPException(
                status_code=429,
                detail="Too many failed attempts — try again later.",
            )
        if not secrets.compare_digest(supplied, token):
            throttle.record_failure(key)
            raise HTTPException(status_code=401, detail="Bad or missing token")
        throttle.clear(key)

    async def require_token(request: Request) -> None:
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else ""
        _check_token(request, supplied)

    async def require_token_or_query(request: Request) -> None:
        """Media-only variant: <img>/<video> tags can't send headers, so the
        file route also accepts ?token=. Everywhere else the query form is
        rejected — query strings end up in access logs and browser history."""
        auth = request.headers.get("authorization", "")
        supplied = (
            auth[7:] if auth.lower().startswith("bearer ")
            else request.query_params.get("token", "")
        )
        _check_token(request, supplied)

    guarded = [Depends(require_token)]
    media_guarded = [Depends(require_token_or_query)]

    def _workspace(name: str) -> Workspace:
        try:
            ws = Workspace.for_project(name)
        except InvalidProjectName as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not ws.root.exists():
            raise HTTPException(status_code=404, detail=f"No project '{name}'")
        return ws

    def _pipeline(ws: Workspace) -> Pipeline:
        cfg = Config.load(config_path, override_path=ws.root / "config.json")
        return Pipeline(cfg, ws, RunOptions())

    # ------------------------------ endpoints ------------------------------ #

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/orders", dependencies=guarded)
    async def list_orders() -> dict[str, Any]:
        """Paid orders, newest first.

        With Firebase configured the Firestore ledger drives the listing
        (customer/package metadata + status), and Cloudinary folders that
        predate the ledger are appended; otherwise it's the pure Cloudinary
        folder listing.
        """
        ingested = ingested_orders(PROJECTS_DIR)
        # Also keyed by ORDER ID: ingest resolves an order-doc name that has
        # drifted (a recreated folder, a stale timestamp) to the Cloudinary
        # folder that really holds the photos, and order.json then records
        # THAT name. Without this the ledger's row would still look
        # un-ingested and offer a button that builds a second project for an
        # order already done.
        ingested_by_order_id: dict[str, str] = {}
        for handled_folder, handled_project in ingested.items():
            order_id = parse_order_folder(handled_folder)["order_id"]
            if order_id:
                ingested_by_order_id.setdefault(order_id, handled_project)
        pending_ingest = jobs.active_ingest_orders()
        # Each order's most recent ingest job — the panel's success/failure
        # feedback ("the button went back to normal like nothing happened"
        # was a real complaint; a failed ingest must stay visible).
        latest_ingest: dict[str, Job] = {}
        for job in jobs.list():  # newest first
            folder = job.options.get("order", "")
            if job.command == "ingest" and folder and folder not in latest_ingest:
                latest_ingest[folder] = job

        def project_progress(project: str) -> Optional[dict[str, Any]]:
            """The order's real pipeline position, from its project snapshot."""
            try:
                ws = Workspace.for_project(project)
                if not ws.root.exists():
                    return None
                snap = _pipeline(ws).snapshot()
            except Exception:  # noqa: BLE001 - a broken project can't hide its order
                return None
            clips = snap.get("clips") or []
            published = snap.get("published") or {}
            return {
                "photos": len(snap.get("input_images") or []),
                "clips_total": len(clips),
                "clips_rendered": sum(1 for c in clips if c.get("rendered")),
                "clips_stale": sum(1 for c in clips if c.get("stale")),
                "final": bool(snap.get("final_video")),
                # Delivery: the newest movie version sitting in the order's
                # own Cloudinary folder (0 = never published), and whether the
                # movie has been rebuilt since it went there.
                "published": (published.get("latest") or {}).get("version", 0),
                "publish_changed": bool(published.get("changed_since")),
                "next_step": snap.get("next_step", ""),
                "placeholders": len(
                    (snap.get("storyboard") or {}).get("placeholder_transitions")
                    or []
                ),
            }

        def active_job_on(project: str) -> Optional[dict[str, str]]:
            for job in jobs.list(project=project):
                if job.state in ("queued", "running", "cancelling"):
                    return {"command": job.command, "state": job.state}
            return None

        client = CloudinaryClient.from_config(config)

        def cloudinary_photos(folder: str, project: str) -> Optional[int]:
            """How many photos are sitting in the order's folder RIGHT NOW.

            Answers "is this order actually complete?" before anyone spends a
            click on ingest — the frontend confirms payment before the photos
            finish uploading, so a folder existing has never meant the order
            is whole.

            Only for orders not yet ingested: once there is a project, its own
            snapshot reports the photos that actually landed (`progress`), and
            counting again would spend a Cloudinary call per row on every
            refresh for no new information. Best-effort by construction — a
            listing hiccup returns None (the panel shows nothing) and must
            never take the whole orders page down with it.
            """
            if project or not folder:
                return None
            try:
                return len(client.list_order_assets(folder))
            except Exception:  # noqa: BLE001 - a count is never worth a 500
                logger.warning(
                    "Could not count Cloudinary photos for %s", folder,
                    exc_info=True,
                )
                return None

        def row(folder: str) -> dict[str, Any]:
            parsed = parse_order_folder(folder)
            job = latest_ingest.get(folder)
            project = ingested.get(folder, "") or ingested_by_order_id.get(
                parsed["order_id"], ""
            )
            return {
                "folder": folder,
                "order_id": parsed["order_id"],
                "customer": parsed["customer"],
                "uploaded_at": parsed["stamp"],
                "project": project,
                "progress": project_progress(project) if project else None,
                "active_job": active_job_on(project) if project else None,
                "ingesting": folder in pending_ingest,
                "ingest_state": job.state if job else "",
                "ingest_error": job.error if job else "",
                "ingest_job": job.id if job else "",
                # Live Cloudinary count for orders still awaiting ingest.
                "cloudinary_photos": cloudinary_photos(folder, project),
            }

        # The folders Cloudinary really has, read once: the rows below are
        # keyed on them rather than on the name the order doc recorded.
        cloudinary_folders = client.list_order_folders()
        folders_by_order_id: dict[str, list[str]] = {}
        for name in cloudinary_folders:
            folders_by_order_id.setdefault(
                parse_order_folder(name)["order_id"], []
            ).append(name)
        known_folders = set(cloudinary_folders)

        def real_folder(leaf: str) -> str:
            """Where this order's photos actually are, not where the doc says.

            Both systems build the folder leaf from a timestamp of their own,
            so the two can disagree (a real order was recorded as
            ``..._00-45_מרגש`` while its photos sat in ``..._11-23_מרגש``) —
            and then the panel counted photos in a folder that does not
            exist, showed "nothing uploaded yet", and posted a name ingest
            could not resolve. The order id is the stable half of the name,
            so a single folder carrying it IS this order's folder.
            """
            if not leaf or leaf in known_folders:
                return leaf
            same_order = folders_by_order_id.get(
                parse_order_folder(leaf)["order_id"], []
            )
            return same_order[0] if len(same_order) == 1 else leaf

        out: list[dict[str, Any]] = []
        seen_folders: set[str] = set()
        if FirebaseClient.configured(config):
            for order in FirebaseClient.from_config(config).list_orders():
                leaf = real_folder(order.folder_leaf)
                seen_folders.add(leaf)
                entry = row(leaf) if leaf else {
                    "folder": "", "order_id": order.order_id,
                    "customer": order.customer, "uploaded_at": "",
                    "project": "", "progress": None, "active_job": None,
                    "ingesting": False,
                    "ingest_state": "", "ingest_error": "", "ingest_job": "",
                }
                entry.update({
                    "order_id": order.order_id,
                    "customer": order.customer or entry["customer"],
                    "status": order.status,
                    "email": order.email,
                    "phone": order.phone,
                    "package_id": order.package_id,
                    "music_mood": order.music_mood,
                    "blessing": order.blessing,
                    "photo_count": order.photo_count,
                    "created_at": order.created_at,
                    "source": "firestore",
                })
                out.append(entry)

        for folder in cloudinary_folders:
            if folder in seen_folders:
                continue
            out.append({**row(folder), "source": "cloudinary"})
        return {"orders": out}

    @app.post("/api/orders/ingest", dependencies=guarded)
    async def ingest_order(body: dict[str, Any]) -> dict[str, Any]:
        folder = str(body.get("order", "")).strip()
        if not folder:
            raise HTTPException(status_code=400, detail="'order' is required")
        existing = {
            p.name for p in PROJECTS_DIR.iterdir() if is_project_dir(p)
        } if PROJECTS_DIR.exists() else set()
        project = str(body.get("project", "")).strip() or derive_project_name(
            folder, existing
        )
        then = (
            {"command": "storyboard", "options": {}}
            if body.get("storyboard") else None
        )
        job = jobs.enqueue(project, "ingest", {"order": folder}, then=then)
        return {"job": job.summary(), "project": project}

    @app.post("/api/projects", dependencies=guarded)
    async def create_project(body: dict[str, Any]) -> dict[str, Any]:
        """The UI twin of `pipeline.py init`: create an empty workspace."""
        name = str(body.get("name", "")).strip()
        try:
            ws = Workspace.for_project(name)
        except InvalidProjectName as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if ws.root.exists():
            raise HTTPException(
                status_code=409, detail=f"Project '{ws.root.name}' already exists"
            )
        ws.mkdirs()
        return {"ok": True, "project": ws.root.name}

    @app.delete("/api/projects/{name}", dependencies=guarded)
    async def delete_project(name: str, confirm: str = "") -> dict[str, Any]:
        """Delete a whole project workspace — photos, frames, clips, movie.

        The most destructive thing this API can do, and unlike everything
        else here it cannot be undone or re-earned cheaply: the styled frames
        and rendered clips inside cost real API credits, `projects/` has no
        backup, and a delivered movie's local archive lives in there too.
        So it is guarded four ways:

        * the name must be a valid, non-reserved project that exists;
        * ``confirm`` must repeat that name exactly — the same "an approval
          is for one exact name" rule publishing uses, so a stray DELETE
          cannot take a workspace with it;
        * a symlink is refused rather than followed (the dev tree symlinks
          `projects/` at the production checkout — deleting THROUGH one is
          how 606 MB went missing once already);
        * a project with a queued or running job is refused, because
          deleting the files under a running pipeline would leave a job
          writing into a directory that no longer exists.

        What was destroyed is counted first and logged, so the server log
        keeps a record of a deletion nothing else can recover.
        """
        ws = _workspace(name)
        project = ws.root.name
        if confirm != project:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Deleting '{project}' needs confirm={project} — the "
                    "request did not repeat the project name, so nothing was "
                    "deleted."
                ),
            )
        if ws.root.is_symlink() or not is_project_dir(ws.root):
            raise HTTPException(
                status_code=400,
                detail=f"'{project}' is not a deletable project directory.",
            )
        busy = [
            j for j in jobs.list(project=project)
            if j.state in ("queued", "running", "cancelling")
        ]
        if busy:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"'{project}' has a {busy[0].command} job {busy[0].state} "
                    "— cancel it and wait for it to stop before deleting."
                ),
            )
        contents = _project_contents(ws)
        shutil.rmtree(ws.root)
        logger.warning(
            "DELETED project '%s' and everything in it: %s. This is not "
            "recoverable — projects/ has no backup.",
            project,
            ", ".join(f"{n} {kind}" for kind, n in contents.items() if n) or "nothing",
        )
        return {"deleted": project, "contents": contents}

    @app.post("/api/projects/{name}/photos", dependencies=guarded)
    async def upload_photos(
        name: str, files: list[UploadFile] = File(...)
    ) -> dict[str, Any]:
        """Add photos to input_images/ (the UI twin of dropping files there).

        Filenames are kept (movie order = sorted filenames, exactly like the
        CLI workflow); an existing file of the same name is replaced.
        """
        ws = _workspace(name)
        saved: list[str] = []
        for upload in files:
            filename = Path(upload.filename or "").name
            if not filename or Path(filename).suffix.lower() not in _PHOTO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Not an image file: {upload.filename!r} "
                           f"(accepted: {', '.join(sorted(_PHOTO_EXTENSIONS))})",
                )
            data = await upload.read()
            if len(data) > _MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"{filename} is larger than "
                           f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                )
            (ws.input_images_dir / filename).write_bytes(data)
            saved.append(filename)
        return {"saved": saved}

    @app.delete("/api/projects/{name}/photos/{filename}", dependencies=guarded)
    async def delete_photo(name: str, filename: str) -> dict[str, Any]:
        """Remove one INPUT photo (the UI twin of deleting the file).

        Only input_images/ is touchable — styled frames and rendered clips
        are never deleted through the API.
        """
        ws = _workspace(name)
        if Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Not found")
        path = ws.input_images_dir / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        path.unlink()
        return {"ok": True}

    @app.post("/api/projects/{name}/music", dependencies=guarded)
    async def upload_music(
        name: str, file: UploadFile = File(...)
    ) -> dict[str, Any]:
        """Set the music bed (the UI twin of --music-file).

        Music is never generated: an uploaded track is the ONLY way a movie
        gets a music bed. Saved as output/music_custom.mp3, which wins over
        any older output/music.mp3 left on disk.
        """
        ws = _workspace(name)
        data = await file.read()
        if len(data) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Music file is larger than "
                       f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            )
        # Judged by CONTENT first, not by the filename: a phone upload of a
        # real mp3 named "Lights(chosic.com)" — no extension — was rejected
        # by the old extension-only check. See media/audio_files.py.
        if not looks_like_audio(data, file.filename or "", file.content_type or ""):
            raise HTTPException(
                status_code=400,
                detail=f"That doesn't look like an audio file: "
                       f"{file.filename or 'the upload'!r}. Upload an mp3, "
                       f"m4a, wav, ogg or flac.",
            )
        dst = ws.custom_music_file
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return {"ok": True}

    @app.post("/api/projects/{name}/music/url", dependencies=guarded)
    async def fetch_music_from_url(
        name: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Fetch the music bed from a URL instead of uploading a file.

        Same destination as the upload (output/music_custom.mp3), so nothing
        downstream can tell the difference. Runs inline rather than as a job:
        a track is a few seconds' download, and the panel wants to show the
        result immediately.

        Licensing is not — and cannot be — checked here. See media/music_url.
        """
        ws = _workspace(name)
        url = str((body or {}).get("url", "")).strip()
        try:
            fetch_music(url, ws.custom_music_file)
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - network/extractor surface
            raise HTTPException(
                status_code=502, detail=f"Fetching the track failed: {exc}"
            ) from exc
        return {"ok": True, "bytes": ws.custom_music_file.stat().st_size}

    @app.delete("/api/projects/{name}/music", dependencies=guarded)
    async def delete_music(name: str) -> dict[str, Any]:
        """Remove the music bed; the movie then has no music at all."""
        ws = _workspace(name)
        ws.custom_music_file.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/projects", dependencies=guarded)
    async def list_projects() -> dict[str, Any]:
        out = []
        if PROJECTS_DIR.exists():
            for path in sorted(PROJECTS_DIR.iterdir()):
                if not is_project_dir(path):
                    continue
                ws = Workspace(path)
                try:
                    snap = _pipeline(ws).snapshot()
                except Exception as exc:  # noqa: BLE001 - one broken project can't hide the rest
                    snap = {"project": path.name, "error": str(exc)}
                snap["order"] = read_order_record(ws.order_file)
                out.append(snap)
        return {"projects": out}

    def _mark_in_flight(project: str, pending: list[dict[str, Any]]) -> None:
        """Flag pending renders that a RUNNING job is already waiting on."""
        if not pending:
            return
        active = next(
            (j for j in jobs.list(project=project)
             if j.state in ("queued", "running", "cancelling")
             and j.command in ("render", "run")),
            None,
        )
        apply_in_flight(pending, None if active is None else active.options)

    @app.get("/api/projects/{name}", dependencies=guarded)
    async def project_detail(name: str) -> dict[str, Any]:
        ws = _workspace(name)
        # probe_clips: one project's clips get measured on disk (memoised on
        # mtime, so the 3s poll re-probes nothing). The LIST above deliberately
        # doesn't — it would measure every clip of every project per refresh.
        snap = _pipeline(ws).snapshot(probe_clips=True)
        snap["order"] = read_order_record(ws.order_file)
        snap["jobs"] = [j.summary() for j in jobs.list(project=name, limit=10)]
        _mark_in_flight(ws.root.name, snap.get("pending_renders", []))
        sb = ws.default_storyboard_json
        snap["storyboard_json"] = (
            sb.read_text(encoding="utf-8") if sb.exists() else ""
        )
        # The letter's text rides along with the detail view (like the
        # storyboard) so the panel's editor opens filled in; the projects
        # LIST keeps just the summary from snapshot().
        snap["letter_text"] = read_letter(ws.letter_file)
        return snap

    @app.get("/api/projects/{name}/publish/preview", dependencies=guarded)
    async def publish_preview(name: str) -> dict[str, Any]:
        """What publishing this project would upload, and under what name.

        Read-only: it lists the order folder's existing movie versions in
        Cloudinary and derives the next free name. The panel calls it to fill
        the confirmation modal, so the name shown for approval is the real one
        — and the same name is then pinned into the publish job (`publish_as`),
        which refuses to upload under anything else.
        """
        ws = _workspace(name)
        try:
            return _pipeline(ws).publish_plan()
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - Cloudinary/network surface
            raise HTTPException(
                status_code=502, detail=f"Cloudinary lookup failed: {exc}"
            ) from exc

    @app.put("/api/projects/{name}/letter", dependencies=guarded)
    async def save_letter_text(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Write the closing letter (the UI twin of editing letter.txt).

        Combine could already be told to scroll a letter, but nothing could
        WRITE one: the text had to be put on the server's disk by hand, which
        behind the tunnel means a shell on the machine. Saving is free and
        local — the letter is rendered at combine time, so a changed letter
        needs another Combine, never a re-render.
        """
        ws = _workspace(name)
        text = body.get("text", "")
        if not isinstance(text, str):
            raise HTTPException(status_code=422, detail="'text' must be a string")
        if len(text) > _MAX_LETTER_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"The letter is longer than {_MAX_LETTER_CHARS} characters",
            )
        state = save_letter(ws.letter_file, text)
        return {"ok": True, "letter": state}

    @app.put("/api/projects/{name}/storyboard", dependencies=guarded)
    async def save_storyboard(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Overwrite the storyboard, marking clips the edits invalidated.

        The save is compared against what is on disk first: a hand-edited
        motion prompt or duration leaves its already-rendered clip showing
        the OLD plan, so those clips are marked outdated (kept, never
        auto-re-rendered — same contract as a re-plan). The returned
        ``outdated`` list is what the panel offers to regenerate in one go.
        """
        ws = _workspace(name)
        try:
            storyboard = Storyboard(**body)
        except Exception as exc:  # noqa: BLE001 - pydantic validation surface
            raise HTTPException(
                status_code=422, detail=f"Invalid storyboard: {exc}"
            ) from exc
        try:
            previous = (
                Storyboard.load(ws.default_storyboard_json)
                if ws.default_storyboard_json.exists() else None
            )
        except StoryboardError:
            # Unreadable/hand-broken JSON on disk: there is nothing to diff
            # against, so save without inventing staleness.
            previous = None
        changed = changed_transition_ids(previous, storyboard)
        pipeline = _pipeline(ws)
        # A clip with a submitted-but-uncollected fal job is already paid for.
        # Editing its plan means the next render can no longer reuse that job
        # (the fingerprint stops matching), so the money is lost — say so
        # rather than letting it happen silently.
        pending_before = {
            p["id"] for p in pipeline.pending_renders() if p["recoverable"]
        }
        storyboard.save(ws.default_storyboard_json)
        outdated = pipeline.mark_clips_outdated(changed) if changed else []
        orphaned = sorted(pending_before.intersection(changed))
        if orphaned:
            logger.warning(
                "Storyboard save discarded %d already-paid render(s) still "
                "waiting on the provider: %s — their plan changed, so the "
                "next render buys fresh clips.",
                len(orphaned), ", ".join(orphaned),
            )
        return {"ok": True, "frames": len(storyboard.frames),
                "transitions": len(storyboard.transitions),
                "changed": changed,
                "outdated": [c.removesuffix(".mp4") for c in outdated],
                "orphaned_renders": orphaned}

    # --------------------------- spending ---------------------------------- #

    @app.get("/api/costs", dependencies=guarded)
    async def costs_overview() -> dict[str, Any]:
        """What every project has cost, plus the studio-wide total.

        Estimates priced from ``config.pricing`` (see costs.py) — never a
        provider invoice, which is why every figure the panel prints from
        this says "estimated".
        """
        rows: list[dict[str, Any]] = []
        for path in sorted(PROJECTS_DIR.iterdir()) if PROJECTS_DIR.exists() else []:
            if not is_project_dir(path):
                continue
            ws = Workspace(path)
            try:
                report = _pipeline(ws).cost_report()
            except Exception as exc:  # noqa: BLE001 - one broken project can't
                # hide the rest of the studio's spending
                rows.append({"project": path.name, "error": str(exc)})
                continue
            order = read_order_record(ws.order_file)
            rows.append({"project": path.name, "order": order, **report})
        return {
            "projects": rows,
            "totals": merge_totals([r for r in rows if "error" not in r]),
            "labels": KIND_LABELS,
            "pricing": config.pricing.model_dump(),
        }

    @app.get("/api/projects/{name}/costs", dependencies=guarded)
    async def project_costs(name: str, limit: int = 200) -> dict[str, Any]:
        """One project's ledger: the totals plus the individual paid calls."""
        ws = _workspace(name)
        pipeline = _pipeline(ws)
        return {
            "summary": pipeline.cost_report(),
            "entries": pipeline.costs.entries(max(1, min(limit, 1000))),
            "labels": KIND_LABELS,
        }

    # --------------------- feedback & learned lessons ---------------------- #

    @app.get("/api/projects/{name}/feedback", dependencies=guarded)
    async def list_feedback(name: str) -> dict[str, Any]:
        ws = _workspace(name)
        return {
            "feedback": [e.model_dump() for e in FeedbackStore(ws.feedback_file).all()]
        }

    # Deliberately a SYNC handler: distilling a lesson is a blocking OpenAI
    # call, and FastAPI runs sync endpoints in a worker thread — an `async
    # def` would park the whole event loop (and every other panel request)
    # for the length of the call.
    @app.post("/api/projects/{name}/feedback", dependencies=guarded)
    def add_feedback(name: str, body: dict[str, Any]) -> dict[str, Any]:
        """Record what a rendered clip got wrong, watch it, and learn from it.

        Runs inline rather than as a job: it is two short calls, the panel
        wants the review and the suggested prompt back in the same
        interaction, and the serial job runner may be halfway through a
        20-minute render. The note is saved even when the review or the
        distillation fails — see Pipeline.record_feedback.

        The suggested prompt/duration is RETURNED, never applied: adopting it
        is a storyboard edit the panel makes (which marks the clip outdated),
        and re-rendering stays a separate, confirmed, paid action.
        """
        ws = _workspace(name)
        note = str((body or {}).get("note", "")).strip()
        clip = str((body or {}).get("clip", "")).strip()
        # A note is required unless the reviewer is being asked to watch a
        # named clip and report on it.
        review = bool((body or {}).get("review", True))
        if not note and not (review and clip):
            raise HTTPException(
                status_code=400,
                detail="'note' is required (or name a 'clip' to have it watched)",
            )
        if len(note) > MAX_NOTE_CHARS:
            raise HTTPException(
                status_code=413,
                detail=f"The note is longer than {MAX_NOTE_CHARS} characters",
            )
        verdict = str((body or {}).get("verdict", "bad")).lower()
        if verdict not in VERDICTS:
            raise HTTPException(
                status_code=422,
                detail=f"'verdict' must be one of: {', '.join(VERDICTS)}",
            )
        options = RunOptions(
            feedback_clip=clip,
            feedback_note=note,
            feedback_verdict=verdict,
            # Learning is the point, but each step costs an OpenAI call, so
            # the caller can record the note alone.
            feedback_learn=bool((body or {}).get("learn", True)),
            feedback_review=review,
        )
        cfg = Config.load(config_path, override_path=ws.root / "config.json")
        try:
            return Pipeline(cfg, ws, options).record_feedback()
        except PipelineError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/lessons", dependencies=guarded)
    async def list_lessons() -> dict[str, Any]:
        """Every rule the planner has learned, newest last (studio-wide)."""
        return {
            "lessons": [l.model_dump() for l in LessonStore(lessons_file()).all()],
            # So the panel can say "learning is off in config" instead of
            # showing rules that are quietly going nowhere.
            "enabled": config.learning_enabled,
            "max_in_prompt": config.max_lessons_in_prompt,
        }

    @app.get("/api/feedback", dependencies=guarded)
    async def all_feedback(limit: int = 100) -> dict[str, Any]:
        """Recent feedback across every project, newest first."""
        entries: list[dict[str, Any]] = []
        for path in sorted(PROJECTS_DIR.iterdir()) if PROJECTS_DIR.exists() else []:
            if not is_project_dir(path):
                continue
            try:
                store = FeedbackStore(Workspace(path).feedback_file)
                entries.extend(e.model_dump() for e in store.all())
            except Exception:  # noqa: BLE001 - one bad file can't hide the rest
                continue
        entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
        return {"feedback": entries[: max(1, min(limit, 500))]}

    @app.post("/api/lessons", dependencies=guarded)
    async def add_lesson(body: dict[str, Any]) -> dict[str, Any]:
        """Write a rule by hand (free — no model call)."""
        try:
            lesson = LessonStore(lessons_file()).add(
                str((body or {}).get("text", "")),
                str((body or {}).get("scope", "motion")),
                origin="manual",
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"lesson": lesson.model_dump()}

    @app.patch("/api/lessons/{lesson_id}", dependencies=guarded)
    async def update_lesson(lesson_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """Edit a rule's text, scope, or whether it is sent to the model."""
        body = body or {}
        scope = body.get("scope")
        if scope is not None and str(scope) not in SCOPES:
            raise HTTPException(
                status_code=422,
                detail=f"'scope' must be one of: {', '.join(SCOPES)}",
            )
        try:
            lesson = LessonStore(lessons_file()).update(
                lesson_id,
                text=None if body.get("text") is None else str(body["text"]),
                active=None if body.get("active") is None else bool(body["active"]),
                scope=None if scope is None else str(scope),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if lesson is None:
            raise HTTPException(status_code=404, detail="No such lesson")
        return {"lesson": lesson.model_dump()}

    @app.delete("/api/lessons/{lesson_id}", dependencies=guarded)
    async def delete_lesson(lesson_id: str) -> dict[str, Any]:
        if not LessonStore(lessons_file()).remove(lesson_id):
            raise HTTPException(status_code=404, detail="No such lesson")
        return {"ok": True}

    @app.post("/api/projects/{name}/actions/{command}", dependencies=guarded)
    async def run_action(
        name: str, command: str, body: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        ws = _workspace(name)
        if command not in _ALLOWED_COMMANDS or command == "ingest":
            raise HTTPException(status_code=400, detail=f"Unknown action: {command}")
        job = jobs.enqueue(ws.root.name, command, body or {})
        return {"job": job.summary()}

    @app.get("/api/jobs", dependencies=guarded)
    async def list_jobs(project: Optional[str] = None) -> dict[str, Any]:
        return {"jobs": [j.summary() for j in jobs.list(project=project)]}

    @app.get("/api/jobs/{job_id}", dependencies=guarded)
    async def job_detail(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        data = job.summary()
        data["log"] = job.log[-200:]
        return data

    @app.post("/api/jobs/{job_id}/cancel", dependencies=guarded)
    async def cancel_job(job_id: str) -> dict[str, Any]:
        """Cancel a queued job now, or ask a running one to stop.

        A running job stops between work items ("cancelling" until the worker
        confirms); whatever is mid-generation finishes and is kept, so
        re-running the command later resumes rather than re-paying.
        """
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="No such job")
        return {"job": job.summary()}

    _FILE_KINDS = {
        "input": lambda ws: ws.input_images_dir,
        "styled": lambda ws: ws.styled_images_dir,
        "generated": lambda ws: ws.generated_frames_dir,
        "clips": lambda ws: ws.clips_dir,
        "output": lambda ws: ws.output_dir,
        # Archived deliveries (output/published/final_vN.mp4) — its own kind
        # because the file route deliberately refuses path separators.
        "published": lambda ws: ws.published_dir,
        "storyboard": lambda ws: ws.storyboard_dir,
    }

    @app.get("/api/projects/{name}/files/{kind}/{filename}",
             dependencies=media_guarded)
    async def project_file(name: str, kind: str, filename: str) -> FileResponse:
        ws = _workspace(name)
        directory = _FILE_KINDS.get(kind)
        if directory is None or Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Not found")
        path = directory(ws) / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Not found")
        return FileResponse(path)

    @app.post("/api/watch/poll", dependencies=guarded)
    async def poll_now() -> dict[str, Any]:
        """Manual watcher pass — 'check for new orders right now'."""
        watcher = OrderWatcher(config, config_path, jobs)
        return {"enqueued": watcher.poll_once()}

    if watch and config.watch_enabled:
        OrderWatcher(config, config_path, jobs).start()

    # The admin panel itself: admin_ui/dist mounted at / (after the /api
    # routes, so they win). Build it with `npm run build` in admin_ui/.
    dist = PROJECT_ROOT / "admin_ui" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="admin-ui")
    else:
        logger.info(
            "admin_ui/dist not found — serving the API only. Build the panel "
            "with: cd admin_ui && npm install && npm run build"
        )

    return app
