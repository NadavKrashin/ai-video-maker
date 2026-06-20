"""Domain exceptions.

Library code raises these instead of calling ``sys.exit`` / ``raise SystemExit``,
so the pipeline can be embedded (e.g. behind an API) without a bad config or
storyboard tearing down the whole process. The CLI catches ``PipelineError`` at
the top level and turns it into a clean non-zero exit.
"""
from __future__ import annotations


class PipelineError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigError(PipelineError):
    """config.json is missing or fails validation."""


class StoryboardError(PipelineError):
    """A storyboard file is missing or fails validation."""


class InvalidProjectName(PipelineError, ValueError):
    """A project name is not a single safe path segment."""
