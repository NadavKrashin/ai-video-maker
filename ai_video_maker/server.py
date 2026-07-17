"""FastAPI admin server: drive the pipeline remotely (the animoments admin panel).

One process serves three things:

* the **admin API** — orders, per-project status (``Pipeline.snapshot()``),
  storyboard read/edit, media files, and actions (ingest/storyboard/render/
  audio/combine) that run as background jobs;
* the **job runner** — a single worker thread executing one pipeline command
  at a time (an order's steps take minutes and the volume is orders-per-day,
  so serial keeps things simple and safe);
* the **watcher** — polls Cloudinary for new paid orders and auto-ingests
  (+ optionally storyboards) the ones whose upload has gone quiet, so a new
  order needs no PC interaction at all.

Interactivity: pipeline confirm gates auto-proceed here (the API caller made
the decision by pressing the button); nothing blocks on stdin.

Auth: every /api route (except /api/health) requires the ``ADMIN_API_TOKEN``
env value, as ``Authorization: Bearer <token>`` or ``?token=<token>`` — the
query form exists because ``<img>``/``<video>`` tags can't send headers.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .clients.cloudinary_client import CloudinaryClient
from .config import Config
from .errors import InvalidProjectName, PipelineCancelled, PipelineError
from .intake import (
    derive_project_name,
    ingested_orders,
    is_order_complete,
    parse_order_folder,
    read_order_record,
)
from .logging_setup import logger, setup_logging
from .models import Storyboard
from .options import RunOptions
from .runner import Pipeline
from .workspace import PROJECTS_DIR, Workspace

# Pipeline commands the API may enqueue, and the RunOptions fields a request
# body may set — an explicit whitelist so a request can't reach for
# constructor internals.
_ALLOWED_COMMANDS = {"ingest", "storyboard", "render", "audio", "combine"}
_ALLOWED_OPTIONS = {
    "force", "dry_run", "duration", "motion_prompt", "style_prompt",
    "music_prompt", "music_file", "analyze_frames", "clips", "order",
    "add_audio", "no_audio", "credits_photos", "closing_letter", "intro_clip",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------- jobs ------------------------------------- #

@dataclass
class Job:
    id: str
    project: str
    command: str
    options: dict[str, Any]
    state: str = "queued"  # queued | running | cancelling | done | failed | cancelled
    error: str = ""
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
            job.state = "done"
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
    """Poll Cloudinary; auto-ingest (+ storyboard) complete new orders."""

    def __init__(self, config: Config, config_path: Path, jobs: JobRunner) -> None:
        self._config = config
        self._config_path = config_path
        self._jobs = jobs
        self._thread = threading.Thread(
            target=self._loop, name="order-watcher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def _loop(self) -> None:
        logger.info(
            "Order watcher: polling Cloudinary every %ds (quiet period %.0f min).",
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
        client = CloudinaryClient.from_config(self._config)
        handled = set(ingested_orders(PROJECTS_DIR))
        handled |= self._jobs.active_ingest_orders()
        existing_names = {
            p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()
        } if PROJECTS_DIR.exists() else set()

        enqueued: list[str] = []
        for folder in client.list_order_folders():
            if folder in handled:
                continue
            assets = client.list_order_assets(folder)
            if not is_order_complete(assets, self._config.watch_quiet_minutes):
                logger.info(
                    "Order %s: %d photo(s) but upload still fresh — waiting.",
                    folder, len(assets),
                )
                continue
            project = derive_project_name(folder, existing_names)
            existing_names.add(project)
            then = (
                {"command": "storyboard", "options": {}}
                if self._config.watch_auto_storyboard else None
            )
            self._jobs.enqueue(project, "ingest", {"order": folder}, then=then)
            logger.info(
                "Order %s: complete (%d photos) — ingesting as project '%s'%s.",
                folder, len(assets), project,
                " + storyboard" if then else "",
            )
            enqueued.append(folder)
        return enqueued


# ---------------------------------- app ------------------------------------- #

def create_app(config_path: Path, *, watch: bool = True) -> FastAPI:
    token = os.environ.get("ADMIN_API_TOKEN", "")
    if not token:
        raise PipelineError(
            "ADMIN_API_TOKEN is not set. Add a long random value to .env — "
            "the admin panel authenticates with it."
        )
    config = Config.load(config_path)
    jobs = JobRunner(config_path)

    app = FastAPI(title="ai-video-maker admin API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.admin_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    async def require_token(request: Request) -> None:
        supplied = request.query_params.get("token", "")
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:]
        if supplied != token:
            raise HTTPException(status_code=401, detail="Bad or missing token")

    guarded = [Depends(require_token)]

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
        client = CloudinaryClient.from_config(config)
        ingested = ingested_orders(PROJECTS_DIR)
        pending_ingest = jobs.active_ingest_orders()
        out = []
        for folder in client.list_order_folders():
            parsed = parse_order_folder(folder)
            out.append({
                "folder": folder,
                "order_id": parsed["order_id"],
                "customer": parsed["customer"],
                "uploaded_at": parsed["stamp"],
                "project": ingested.get(folder, ""),
                "ingesting": folder in pending_ingest,
            })
        return {"orders": out}

    @app.post("/api/orders/ingest", dependencies=guarded)
    async def ingest_order(body: dict[str, Any]) -> dict[str, Any]:
        folder = str(body.get("order", "")).strip()
        if not folder:
            raise HTTPException(status_code=400, detail="'order' is required")
        existing = {
            p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()
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

    @app.get("/api/projects", dependencies=guarded)
    async def list_projects() -> dict[str, Any]:
        out = []
        if PROJECTS_DIR.exists():
            for path in sorted(PROJECTS_DIR.iterdir()):
                if not path.is_dir():
                    continue
                ws = Workspace(path)
                try:
                    snap = _pipeline(ws).snapshot()
                except Exception as exc:  # noqa: BLE001 - one broken project can't hide the rest
                    snap = {"project": path.name, "error": str(exc)}
                snap["order"] = read_order_record(ws.order_file)
                out.append(snap)
        return {"projects": out}

    @app.get("/api/projects/{name}", dependencies=guarded)
    async def project_detail(name: str) -> dict[str, Any]:
        ws = _workspace(name)
        snap = _pipeline(ws).snapshot()
        snap["order"] = read_order_record(ws.order_file)
        snap["jobs"] = [j.summary() for j in jobs.list(project=name, limit=10)]
        sb = ws.default_storyboard_json
        snap["storyboard_json"] = (
            sb.read_text(encoding="utf-8") if sb.exists() else ""
        )
        return snap

    @app.put("/api/projects/{name}/storyboard", dependencies=guarded)
    async def save_storyboard(name: str, body: dict[str, Any]) -> dict[str, Any]:
        ws = _workspace(name)
        try:
            storyboard = Storyboard(**body)
        except Exception as exc:  # noqa: BLE001 - pydantic validation surface
            raise HTTPException(
                status_code=422, detail=f"Invalid storyboard: {exc}"
            ) from exc
        storyboard.save(ws.default_storyboard_json)
        return {"ok": True, "frames": len(storyboard.frames),
                "transitions": len(storyboard.transitions)}

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
        "storyboard": lambda ws: ws.storyboard_dir,
    }

    @app.get("/api/projects/{name}/files/{kind}/{filename}", dependencies=guarded)
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

    return app
