"""AI Video Maker — local automation pipeline.

Two workflows:

  Mode A (default): Image-to-video from images you provide in input_images/.
  Mode B (--from-scratch): Generate a video from a raw idea via a storyboard.

The pipeline is built from three explicit pieces — ``Config`` (config.json),
``Workspace`` (all per-movie paths), and ``RunOptions`` (one run's choices) —
so it can be driven by the CLI or, later, an API. See README.md.
"""
from __future__ import annotations

from .config import Config
from .options import RunOptions
from .runner import Pipeline
from .workspace import Workspace

__version__ = "0.1.0"

__all__ = ["Config", "Workspace", "RunOptions", "Pipeline", "__version__"]
