"""What a project cost: price table + the per-project spending ledger.

Every step of the pipeline buys something — an OpenAI image (styling), an
OpenAI text/vision call (planning), a fal clip render, a fal SFX pass — and
until now the only record of that was the credit-card statement. The ledger
here writes one line per paid call into ``projects/<name>/logs/costs.json``,
so a project can say what it cost and the panel can total the whole studio.

Two honesty rules shape this module:

* **These are ESTIMATES priced from `config.pricing`, not billed amounts.**
  Nothing here talks to a provider's billing API; a line is "this call, at
  the price you configured". Keep the wording "estimated" wherever a number
  reaches the user, and keep the prices in config so they can be corrected
  when a provider's rates change.
* **Recording must never sink a run.** A full disk or an unwritable log is
  not a reason to lose a rendered clip, so every failure here degrades to a
  warning. The ledger is bookkeeping, never a gate.

Costs are recorded when a call SUCCEEDS. A render that was submitted (and
billed) but never collected is therefore missing from the ledger until its
result is downloaded — the same money-is-out-there gap that
``Pipeline.pending_renders()`` surfaces.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .logging_setup import logger

# What a ledger line can be. Kept small and stable — the panel groups by these.
KIND_IMAGE = "image"   # OpenAI image styling / generation
KIND_TEXT = "text"     # OpenAI text + vision calls (planning, condensing, ...)
KIND_CLIP = "clip"     # fal image-to-video render
KIND_SFX = "sfx"       # fal video-to-audio pass
KINDS = (KIND_IMAGE, KIND_TEXT, KIND_CLIP, KIND_SFX)

KIND_LABELS = {
    KIND_IMAGE: "Styled images (OpenAI)",
    KIND_TEXT: "Planning calls (OpenAI)",
    KIND_CLIP: "Clip renders (fal)",
    KIND_SFX: "Sound effects (fal)",
}


class Pricing(BaseModel):
    """Unit prices used to value each API call (config key ``pricing``).

    Defaults reflect the models this project ships with (gpt-image-2,
    gpt-5.1, Kling v2.5 Turbo Pro, mmaudio-v2) as of mid-2026. They are
    list prices, not your invoice: providers change rates, and volume/credit
    deals move them further. Edit config.json when a number drifts — every
    total in the panel is derived from these, including retroactively (the
    ledger stores the price that applied when the call was made, so past
    lines keep their original valuation).
    """

    # One styled/generated frame at 1536x1024.
    openai_image_usd: float = 0.19
    # Text + vision calls are priced from the token usage the API reports.
    openai_text_input_usd_per_1m: float = 1.25
    openai_text_output_usd_per_1m: float = 10.0
    # Kling v2.5 Turbo Pro: $0.35 for a 5s clip, $0.70 for 10s.
    clip_usd_per_second: float = 0.07
    # One video->audio pass over a clip (mmaudio-v2).
    sfx_usd: float = 0.02

    def clip(self, seconds: float) -> float:
        return round(self.clip_usd_per_second * max(0.0, float(seconds)), 6)

    def text(self, input_tokens: float, output_tokens: float) -> float:
        return round(
            (max(0.0, float(input_tokens)) * self.openai_text_input_usd_per_1m
             + max(0.0, float(output_tokens)) * self.openai_text_output_usd_per_1m)
            / 1_000_000,
            6,
        )

    def image(self, count: int = 1) -> float:
        return round(self.openai_image_usd * max(0, int(count)), 6)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_summary() -> dict[str, Any]:
    """The shape :meth:`CostLedger.summary` returns for a project with no spend."""
    return {
        "total_usd": 0.0,
        "by_kind": {kind: {"usd": 0.0, "count": 0} for kind in KINDS},
        "entries": 0,
        "first_at": "",
        "last_at": "",
    }


class CostLedger:
    """Append-only spending log for one project (``logs/costs.json``).

    Thread-safe: clip/image workers run in a pool and all record through the
    same instance. Writes are best-effort — see the module docstring.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._loaded = False

    # --- reading ---------------------------------------------------------- #
    def _load(self) -> None:
        """Read the file once, tolerating anything that isn't a valid ledger."""
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read %s (%s); starting a fresh ledger.",
                           self.path, exc)
            return
        entries = data.get("entries") if isinstance(data, dict) else data
        if isinstance(entries, list):
            self._entries = [e for e in entries if isinstance(e, dict)]

    def entries(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """Recorded lines, newest LAST (chronological); optionally the last N."""
        with self._lock:
            self._load()
            items = list(self._entries)
        return items[-limit:] if limit else items

    def summary(self) -> dict[str, Any]:
        """Totals for this project: overall, per kind, and the time span."""
        out = empty_summary()
        entries = self.entries()
        if not entries:
            return out
        total = 0.0
        for entry in entries:
            kind = str(entry.get("kind", ""))
            usd = float(entry.get("usd", 0.0) or 0.0)
            total += usd
            bucket = out["by_kind"].setdefault(kind, {"usd": 0.0, "count": 0})
            bucket["usd"] = round(bucket["usd"] + usd, 6)
            bucket["count"] += 1
        out["total_usd"] = round(total, 6)
        out["entries"] = len(entries)
        stamps = sorted(str(e.get("at", "")) for e in entries if e.get("at"))
        out["first_at"] = stamps[0] if stamps else ""
        out["last_at"] = stamps[-1] if stamps else ""
        return out

    # --- writing ---------------------------------------------------------- #
    def record(
        self, kind: str, usd: float, detail: str = "", **extra: Any
    ) -> None:
        """Append one paid call. Never raises — bookkeeping can't fail a run."""
        try:
            amount = round(float(usd), 6)
        except (TypeError, ValueError):
            return
        if amount <= 0:
            return
        entry = {
            "at": _now(),
            "kind": kind,
            "usd": amount,
            "detail": detail,
            **extra,
        }
        try:
            with self._lock:
                self._load()
                self._entries.append(entry)
                self._flush()
        except Exception as exc:  # noqa: BLE001 - never break the pipeline
            logger.warning("Could not record cost (%s): %s", kind, exc)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"entries": self._entries}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def estimate_usd(
    pricing: Pricing,
    *,
    images: int = 0,
    clip_seconds: float = 0.0,
    sfx_clips: int = 0,
) -> float:
    """Price the artifacts a project HAS, ignoring the ledger.

    Every project that existed before cost tracking shipped has a full set of
    styled frames and rendered clips and an empty ledger; without this they
    would all read "$0 spent", which is worse than an approximation. The
    panel shows it as an estimate whenever it exceeds what was recorded.
    """
    return round(
        pricing.image(images)
        + pricing.clip(clip_seconds)
        + pricing.sfx_usd * max(0, int(sfx_clips)),
        6,
    )


def merge_totals(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Add up several projects' cost reports into one studio-wide total."""
    total = 0.0
    estimated = 0.0
    by_kind: dict[str, dict[str, Any]] = {
        kind: {"usd": 0.0, "count": 0} for kind in KINDS
    }
    for report in reports:
        total += float(report.get("total_usd", 0.0) or 0.0)
        estimated += float(report.get("estimated_usd", 0.0) or 0.0)
        for kind, bucket in (report.get("by_kind") or {}).items():
            target = by_kind.setdefault(kind, {"usd": 0.0, "count": 0})
            target["usd"] = round(
                target["usd"] + float(bucket.get("usd", 0.0) or 0.0), 6
            )
            target["count"] += int(bucket.get("count", 0) or 0)
    return {
        "total_usd": round(total, 6),
        "estimated_usd": round(estimated, 6),
        "by_kind": by_kind,
        "projects": len(reports),
    }
