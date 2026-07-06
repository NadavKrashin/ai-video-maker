"""Shared fixtures: a real Config/Workspace/Pipeline against a tmp dir.

No network, no API keys — the clients are constructed lazily and never used.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ai_video_maker.config import Config
from ai_video_maker.options import RunOptions
from ai_video_maker.runner import Pipeline
from ai_video_maker.workspace import Workspace


@pytest.fixture
def config() -> Config:
    return Config(
        style_prompt="test style",
        scratch_style_prompt="test scratch style",
        motion_prompt="test motion",
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    ws = Workspace(root=tmp_path / "proj")
    ws.mkdirs()
    return ws


@pytest.fixture
def make_pipeline(config: Config, workspace: Workspace):
    def _make(**option_overrides) -> Pipeline:
        options = RunOptions(**option_overrides)
        return Pipeline(config, workspace, options)

    return _make


@pytest.fixture
def pipeline(make_pipeline) -> Pipeline:
    return make_pipeline()
