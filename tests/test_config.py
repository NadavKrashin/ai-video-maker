"""Config loading and per-project override layering."""
from __future__ import annotations

import json

import pytest

from ai_video_maker.config import Config
from ai_video_maker.errors import ConfigError

_BASE = {
    "style_prompt": "base style",
    "scratch_style_prompt": "base scratch",
    "motion_prompt": "base motion",
    "duration": 5,
}


def _write(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestConfigLoad:
    def test_loads_base(self, tmp_path):
        cfg = Config.load(_write(tmp_path / "config.json", _BASE))
        assert cfg.style_prompt == "base style"

    def test_background_watcher_is_opt_in(self, tmp_path):
        # The user explicitly rejected background polling (2026-07-18):
        # orders are fetched live by the panel; auto-ingest is opt-in.
        cfg = Config.load(_write(tmp_path / "config.json", _BASE))
        assert cfg.watch_enabled is False

    def test_missing_base_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            Config.load(tmp_path / "nope.json")

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid JSON"):
            Config.load(path)

    def test_non_object_raises(self, tmp_path):
        path = _write(tmp_path / "config.json", ["not", "an", "object"])
        with pytest.raises(ConfigError, match="JSON object"):
            Config.load(path)


class TestProjectOverrides:
    def test_override_wins_key_by_key(self, tmp_path):
        base = _write(tmp_path / "config.json", _BASE)
        override = _write(
            tmp_path / "project.json", {"style_prompt": "movie style", "duration": 10}
        )
        cfg = Config.load(base, override_path=override)
        assert cfg.style_prompt == "movie style"
        assert cfg.duration == 10
        assert cfg.motion_prompt == "base motion"  # untouched keys kept

    def test_missing_override_is_fine(self, tmp_path):
        base = _write(tmp_path / "config.json", _BASE)
        cfg = Config.load(base, override_path=tmp_path / "absent.json")
        assert cfg.style_prompt == "base style"

    def test_invalid_override_value_names_source(self, tmp_path):
        base = _write(tmp_path / "config.json", _BASE)
        override = _write(tmp_path / "project.json", {"duration": "not-a-number"})
        with pytest.raises(ConfigError, match="overrides"):
            Config.load(base, override_path=override)
