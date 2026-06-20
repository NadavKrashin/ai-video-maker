"""Shared logger and logging configuration."""
from __future__ import annotations

import logging
import sys
from datetime import datetime

from .workspace import Workspace

# One shared logger for the whole package (name kept as "pipeline" for
# continuity with existing log files / config).
logger = logging.getLogger("pipeline")


def setup_logging(workspace: Workspace) -> None:
    """Log to both the console and a timestamped file in the workspace's logs/."""
    workspace.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    logfile = (
        workspace.logs_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(logfile, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    logger.debug("Logging initialised -> %s", logfile)
