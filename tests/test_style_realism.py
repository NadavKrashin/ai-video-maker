"""The per-project look: lifelike (the approved default) vs classic cartoon.

Moving styling to gpt-image-2.6-sunburst made frames more lifelike than the
gpt-image-2 frames before it, from the SAME style prompt. The user chose that
look for likeness, but customers may want the earlier cartoon, so the look is
a per-project choice on the same model.
"""
from __future__ import annotations

import typing

import pytest

from ai_video_maker.config import Config
from ai_video_maker.project_settings import (
    coerce_settings,
    describe_settings,
    get_setting,
)
from ai_video_maker.style_realism import (
    CARTOON_LOOK_LAYER,
    DEFAULT_STYLE_REALISM,
    STYLE_REALISM_LEVELS,
    with_style_realism,
)

BASE = "Convert this image into an unmistakable Pixar-style cartoon frame."


class TestTheLayer:
    def test_lifelike_is_the_approved_prompt_byte_for_byte(self):
        # The base prompt on this model IS the approved look; the default
        # level must not reword it by so much as a newline.
        assert with_style_realism(BASE, "lifelike") == BASE

    def test_the_default_is_lifelike(self):
        assert DEFAULT_STYLE_REALISM == "lifelike"
        assert Config(style_prompt="s", scratch_style_prompt="s",
                      motion_prompt="m").style_realism == "lifelike"

    def test_cartoon_layers_on_top_of_the_base(self):
        out = with_style_realism(BASE, "cartoon")
        assert out.startswith(BASE)  # every base rule still reaches the model
        assert out.endswith(CARTOON_LOOK_LAYER)

    def test_an_unknown_level_degrades_to_the_approved_look(self):
        assert with_style_realism(BASE, "photo") == BASE

    def test_cartoon_changes_rendering_never_the_people(self):
        # Enlarged eyes, slimmed faces and beautification were rejected long
        # before too much realism was — the extra cartoon must not buy them.
        layer = CARTOON_LOOK_LAYER
        assert "RENDERING ONLY" in layer
        assert "every likeness rule above still governs" in layer
        assert "Do NOT enlarge eyes" in layer
        assert "including baldness" in layer
        assert "exaggerate proportions" in layer

    def test_config_accepts_exactly_the_module_levels(self):
        hint = typing.get_type_hints(Config)["style_realism"]
        assert set(typing.get_args(hint)) == set(STYLE_REALISM_LEVELS)

    def test_config_refuses_anything_else(self):
        with pytest.raises(Exception):
            Config(style_prompt="s", scratch_style_prompt="s",
                   motion_prompt="m", style_realism="photo")


class TestTheProjectSetting:
    def test_it_is_an_editable_choice(self):
        s = get_setting("style_realism")
        assert s is not None and s.kind == "choice"
        assert [v for v, _ in s.choices] == list(STYLE_REALISM_LEVELS)
        assert s.applies == "restyle"

    def test_valid_values_pass_and_others_are_refused(self):
        assert coerce_settings({"style_realism": "cartoon"}) == {
            "style_realism": "cartoon"
        }
        with pytest.raises(ValueError, match="must be one of"):
            coerce_settings({"style_realism": "photo"})

    def test_null_still_unpins(self):
        assert coerce_settings({"style_realism": None}) == {"style_realism": None}

    def test_the_panel_is_given_the_choices(self):
        config = Config(style_prompt="s", scratch_style_prompt="s",
                        motion_prompt="m", style_realism="cartoon")
        row = next(r for r in describe_settings(config, {"style_realism": "cartoon"})
                   if r["key"] == "style_realism")
        assert row["value"] == "cartoon" and row["pinned"] is True
        assert {c["value"] for c in row["choices"]} == set(STYLE_REALISM_LEVELS)

    def test_the_help_says_existing_frames_need_a_restyle(self):
        # Switching looks never silently restyles (that spends credits), so
        # the setting must say how to get existing frames into the new look.
        assert "Force re-style all photos" in get_setting("style_realism").help


class TestStylingUsesTheProjectsLook:
    """The layer reaches the actual styling call — and only that call."""

    def _style(self, make_pipeline, workspace, realism, note=None):
        from tests.test_runner_logic import _make_frame_pairs
        pairs = _make_frame_pairs(workspace, ["a"])
        for _, dst in pairs:
            dst.unlink()
        p = make_pipeline()
        p.config = p.config.model_copy(update={"style_realism": realism})
        calls = []

        class _FakeOpenAI:
            def style_image(self, src, prompt, dst, references=None):
                calls.append(prompt)
                dst.write_bytes(b"styled")

        p.__dict__["openai"] = _FakeOpenAI()
        notes = {"styled_images/a.png": note} if note else {}
        p._style_images([src for src, _ in pairs], {}, notes)
        return calls[0]

    def test_lifelike_sends_the_base_prompt_unchanged(self, make_pipeline, workspace):
        assert self._style(make_pipeline, workspace, "lifelike") == "test style"

    def test_cartoon_sends_the_layer(self, make_pipeline, workspace):
        prompt = self._style(make_pipeline, workspace, "cartoon")
        assert prompt.startswith("test style")
        assert CARTOON_LOOK_LAYER in prompt

    def test_a_frame_note_still_comes_last(self, make_pipeline, workspace):
        # The note is the most specific instruction for that one frame, so
        # it must refine the look rather than be overridden by it.
        prompt = self._style(make_pipeline, workspace, "cartoon",
                             note="keep the reflection a reflection")
        assert prompt.index(CARTOON_LOOK_LAYER) < prompt.index(
            "keep the reflection a reflection")
