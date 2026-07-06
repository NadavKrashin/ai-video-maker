"""Job state (resume support) and failed-job tracking, persisted as JSON."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .logging_setup import logger


class StateStore:
    """Persists per-job status to logs/state.json so runs can resume."""

    def __init__(self, path: Path) -> None:
        self.path = path
        # Guards both the in-memory dict and the file write so parallel workers
        # can record job status without corrupting state.json.
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"jobs": {}}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
                self._data.setdefault("jobs", {})
            except json.JSONDecodeError:
                logger.warning("Could not parse %s; starting fresh.", path)

    def status(self, job_id: str) -> Optional[str]:
        with self._lock:
            entry = self._data["jobs"].get(job_id)
            return entry.get("status") if entry else None

    def is_done(self, job_id: str) -> bool:
        return self.status(job_id) == "done"

    def set(self, job_id: str, status: str, **extra: Any) -> None:
        with self._lock:
            self._data["jobs"][job_id] = {
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                **extra,
            }
            self._flush()

    def clear(self, *job_ids: str) -> None:
        """Forget the given jobs (no-op for ids that aren't recorded).

        Used when an output is regenerated: downstream per-output jobs (e.g. a
        clip's SFX and fade) must run again for the new file, so their "done"
        entries have to go.
        """
        with self._lock:
            removed = False
            for job_id in job_ids:
                if self._data["jobs"].pop(job_id, None) is not None:
                    removed = True
            if removed:
                self._flush()

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )


class FailedJobStore:
    """Collects failed jobs and writes them to failed_jobs/failed_jobs.json."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.failures: list[dict[str, Any]] = []

    def record(self, job_id: str, kind: str, error: str, **extra: Any) -> None:
        with self._lock:
            self.failures.append(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "error": error,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **extra,
                }
            )
        logger.error("FAILED [%s] %s: %s", kind, job_id, error)

    def flush(self) -> None:
        if not self.failures:
            # A clean run removes the previous run's report — otherwise a stale
            # failed_jobs.json sits there looking like it describes this run.
            if self.path.exists():
                self.path.unlink()
                logger.info("No failures this run; removed stale %s", self.path)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.failures, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote %d failed job(s) -> %s", len(self.failures), self.path)
