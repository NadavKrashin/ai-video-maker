"""Choosing a video model, and what that choice decides for you.

Two things are being pinned here. First, that picking a model sets its whole
request shape at once — the frame fields disagree between Kling versions and
getting one wrong does not fail loudly, it silently drops the end frame and
the movie stops landing on its photos. Second, that whether the cast's faces
are sent is a property of the MODEL: 2.5 cannot take them and never gets
them, 3 can and does. A separate switch would eventually drift out of step
with the model it belongs to.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_video_maker.config import Config, apply_video_model_preset
from ai_video_maker.errors import ConfigError
from ai_video_maker.video_models import (
    VIDEO_MODELS,
    get_video_model,
    model_for_id,
    model_keys,
)


def _config(**data) -> Config:
    base = {"style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m"}
    return Config(**apply_video_model_preset({**base, **data}))


def _load(tmp_path: Path, shared: dict, override: dict | None = None) -> Config:
    shared_path = tmp_path / "config.json"
    shared_path.write_text(json.dumps({
        "style_prompt": "s", "scratch_style_prompt": "s", "motion_prompt": "m",
        **shared,
    }), encoding="utf-8")
    override_path = None
    if override is not None:
        override_path = tmp_path / "project.json"
        override_path.write_text(json.dumps(override), encoding="utf-8")
    return Config.load(shared_path, override_path)


class TestTheRegistryItself:
    def test_every_preset_is_self_consistent(self):
        for model in VIDEO_MODELS:
            assert model.key and model.label and model.model_id
            assert model.start_frame_field
            # Every SELECTABLE model interpolates to an end frame. A preset
            # without one breaks clip chaining, which is the pipeline's
            # contract — and is exactly why one is retired.
            assert model.end_frame_field or model.retired
            assert model.usd_per_second > 0
            assert model.price_note, f"{model.key}: say where the price came from"
            if model.supports_elements:
                assert model.elements_field and model.max_elements > 0
            else:
                assert not model.elements_field and model.max_elements == 0

    def test_keys_are_unique(self):
        assert len(model_keys()) == len(set(model_keys()))

    def test_the_current_production_model_is_offered_and_verified(self):
        model = get_video_model("kling-2.5-turbo-pro")
        assert model is not None and model.verified
        assert model.model_id.endswith("v2.5-turbo/pro/image-to-video")

    def test_an_unrendered_model_is_flagged_and_says_so(self):
        # A request shape that came from documentation, not from a render
        # made here. Saying so is what tells someone to try one clip first.
        model = get_video_model("kling-4")
        assert model is not None and not model.verified
        assert "one clip" in model.note.lower()

    def test_the_models_rendered_in_the_first_bakeoff_are_verified(self):
        for key in ("kling-3-standard", "kling-3-pro", "kling-o3-pro"):
            assert get_video_model(key).verified
        # ...and the one that came back at 720p says so where it is picked.
        assert "720" in get_video_model("kling-3-standard").note

    def test_an_unknown_key_is_simply_absent(self):
        assert get_video_model("kling-99") is None
        assert get_video_model("") is None

    def test_a_raw_model_id_still_resolves_to_its_preset(self):
        # Every project written before presets existed names a model id, and
        # must still get the capability answers.
        found = model_for_id("fal-ai/kling-video/v2.5-turbo/pro/image-to-video")
        assert found is not None and found.key == "kling-2.5-turbo-pro"
        assert model_for_id("something-else") is None


class TestPickingAModelSetsItsWholeShape:
    def test_kling_three_brings_its_own_field_names(self):
        config = _config(video_model="kling-3-pro")
        assert config.fal_model_id.endswith("v3/pro/image-to-video")
        assert config.fal_start_frame_field == "start_image_url"
        assert config.fal_end_frame_field == "end_image_url"

    def test_two_five_brings_the_older_names(self):
        config = _config(video_model="kling-2.5-turbo-pro")
        assert config.fal_start_frame_field == "image_url"
        assert config.fal_end_frame_field == "tail_image_url"

    def test_an_explicit_field_still_beats_the_preset(self):
        # An experiment must not need a new preset, and no existing config
        # may change behaviour because a preset was added.
        config = _config(
            video_model="kling-3-pro", fal_end_frame_field="tail_image_url"
        )
        assert config.fal_end_frame_field == "tail_image_url"
        assert config.fal_start_frame_field == "start_image_url"

    def test_pricing_follows_the_model(self):
        # Switching to a model that costs 60% more must not keep reporting
        # the old estimate — spend figures are the only cost signal there is.
        assert _config(video_model="kling-3-pro").pricing.clip_usd_per_second \
            == pytest.approx(0.112)
        assert _config(video_model="kling-2.5-turbo-pro").pricing \
            .clip_usd_per_second == pytest.approx(0.07)

    def test_explicit_pricing_still_wins(self):
        config = _config(
            video_model="kling-3-pro", pricing={"clip_usd_per_second": 0.2}
        )
        assert config.pricing.clip_usd_per_second == pytest.approx(0.2)

    def test_no_preset_leaves_everything_exactly_as_it_was(self):
        config = _config()
        assert config.video_model == ""
        assert config.fal_start_frame_field == "image_url"
        assert config.fal_end_frame_field == "tail_image_url"

    def test_a_typo_is_refused_loudly(self):
        # Silently rendering a whole order on the wrong model is expensive.
        with pytest.raises(ConfigError) as exc:
            _config(video_model="kling-3")
        assert "kling-3-pro" in str(exc.value)

    def test_a_project_can_pin_its_own_model(self, tmp_path):
        config = _load(tmp_path, {}, {"video_model": "kling-3-pro"})
        assert config.fal_start_frame_field == "start_image_url"

    def test_a_project_override_beats_the_shared_preset(self, tmp_path):
        config = _load(
            tmp_path,
            {"video_model": "kling-2.5-turbo-pro"},
            {"video_model": "kling-3-pro"},
        )
        assert config.fal_model_id.endswith("v3/pro/image-to-video")
        assert config.fal_end_frame_field == "end_image_url"


class TestTheModelDecidesWhetherFacesAreSent:
    """The rule the user asked for: 2.5 off, Kling 3 on."""

    def test_kling_two_five_never_sends_cast_faces(self):
        assert not _config(video_model="kling-2.5-turbo-pro").elements_enabled()

    def test_kling_three_sends_them(self):
        assert _config(video_model="kling-3-pro").elements_enabled()
        assert _config(video_model="kling-3-standard").elements_enabled()

    def test_two_five_refuses_even_when_the_fields_are_forced(self):
        # The whole reason capability lives on the model: a leftover element
        # field from a previous choice must not send faces to a model that
        # cannot read them.
        config = _config(
            video_model="kling-2.5-turbo-pro",
            fal_elements_field="elements",
            fal_max_elements=3,
        )
        assert not config.elements_enabled()

    def test_the_default_config_sends_nothing(self):
        # The shipped default is 2.5, so nothing changes for anyone until
        # they pick a different model.
        assert not _config().elements_enabled()

    def test_an_unknown_model_falls_back_to_its_fields(self):
        # A brand-new endpoint has no preset yet; hand-configuring it must
        # still work, or a preset becomes a prerequisite for experimenting.
        assert not _config(fal_model_id="fal-ai/some/new/model").elements_enabled()
        assert _config(
            fal_model_id="fal-ai/some/new/model",
            fal_elements_field="elements",
            fal_max_elements=1,
        ).elements_enabled()

    def test_a_capable_model_with_the_fields_cleared_sends_nothing(self):
        config = _config(video_model="kling-3-pro", fal_max_elements=0)
        assert not config.elements_enabled()


class TestTheRenderRequestHonoursTheModel:
    def _client(self, config):
        from ai_video_maker.clients.video import VideoClient
        return VideoClient(config)

    def test_faces_are_dropped_for_two_five(self, tmp_path):
        config = _config(video_model="kling-2.5-turbo-pro")
        args = self._client(config)._build_arguments(
            "s", "e", "m", 5, ["http://face"]
        )
        assert not any("element" in k.lower() for k in args)
        assert args["tail_image_url"] == "e"

    def test_faces_are_sent_for_kling_three(self):
        config = _config(video_model="kling-3-pro")
        args = self._client(config)._build_arguments(
            "s", "e", "m", 5, ["http://face"]
        )
        # Kling 3 rejects an element with no reference views, so the same
        # portrait is sent under both fields.
        assert args["elements"] == [{
            "frontal_image_url": "http://face",
            "reference_image_urls": ["http://face"],
        }]
        # And the end frame survives alongside them — the one thing this
        # pipeline cannot trade away.
        assert args["end_image_url"] == "e"
        assert args["start_image_url"] == "s"


REPO = Path(__file__).resolve().parents[1]


class TestPickingAModelActuallyChangesTheModel:
    """The shared config.json spells out fal_model_id (it predates presets).

    "An explicit key beats the preset" then meant a project pinned to Kling 3
    in the panel still rendered on Kling 2.5 — 2.5's endpoint, 2.5's field
    names, 2.5's price — while the page said Kling 3. Tested against the LIVE
    config.json because a hand-made fixture is what hid this.
    """

    def _pinned(self, tmp_path: Path, override: dict) -> Config:
        path = tmp_path / "project.json"
        path.write_text(json.dumps(override), encoding="utf-8")
        return Config.load(REPO / "config.json", path)

    def test_the_live_shared_config_alone_is_still_kling_two_five(self):
        config = Config.load(REPO / "config.json")
        assert config.fal_model_id.endswith("v2.5-turbo/pro/image-to-video")
        assert config.fal_end_frame_field == "tail_image_url"

    def test_a_project_pin_beats_the_inherited_request_shape(self, tmp_path):
        config = self._pinned(tmp_path, {"video_model": "kling-3-pro"})
        assert config.fal_model_id == "fal-ai/kling-video/v3/pro/image-to-video"
        assert config.fal_start_frame_field == "start_image_url"
        assert config.fal_end_frame_field == "end_image_url"
        assert config.pricing.clip_usd_per_second == 0.112
        assert config.fal_model_arguments == {"generate_audio": False}

    def test_the_other_shared_prices_survive_the_pin(self, tmp_path):
        shared = Config.load(REPO / "config.json")
        config = self._pinned(tmp_path, {"video_model": "kling-3-pro"})
        assert config.pricing.openai_image_usd == shared.pricing.openai_image_usd
        assert config.pricing.sfx_usd == shared.pricing.sfx_usd

    def test_keys_the_project_writes_itself_still_win(self, tmp_path):
        config = self._pinned(tmp_path, {
            "video_model": "kling-3-pro",
            "fal_end_frame_field": "my_end",
            "pricing": {"clip_usd_per_second": 0.5},
        })
        assert config.fal_end_frame_field == "my_end"
        assert config.pricing.clip_usd_per_second == 0.5
        assert config.fal_start_frame_field == "start_image_url"

    def test_a_project_without_a_pin_is_untouched(self, tmp_path):
        config = self._pinned(tmp_path, {"duration": 10})
        assert config.fal_model_id.endswith("v2.5-turbo/pro/image-to-video")
        assert config.pricing.clip_usd_per_second == 0.07


class TestModelArguments:
    def _args(self, key: str) -> dict:
        from ai_video_maker.clients.video import VideoClient
        client = VideoClient(_config(video_model=key))
        return client._build_arguments("s", "e", "m", 5)

    def test_kling_three_is_told_not_to_generate_audio(self):
        # It is on by default and bills 50% more; the pipeline adds its own.
        for key in ("kling-3-pro", "kling-3-standard", "kling-o3-pro"):
            assert self._args(key)["generate_audio"] is False

    def test_two_five_gets_no_new_arguments(self):
        assert "generate_audio" not in self._args("kling-2.5-turbo-pro")

    def test_two_five_fingerprints_are_unchanged_by_the_new_field(self, tmp_path):
        # A paid, pending render must stay resumable.
        from ai_video_maker.clients.video import VideoClient
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        a.write_bytes(b"a"); b.write_bytes(b"b")
        config = _config(video_model="kling-2.5-turbo-pro")
        assert config.fal_model_arguments == {}
        with_field = VideoClient(config).fingerprint(a, b, "m", 5)
        import hashlib
        expected = hashlib.sha1("|".join([
            config.fal_model_id, "5",
            hashlib.sha1(b"a").hexdigest(), hashlib.sha1(b"b").hexdigest(), "m",
        ]).encode()).hexdigest()
        assert with_field == expected

    def test_extra_arguments_still_override_the_model(self):
        from ai_video_maker.clients.video import VideoClient
        config = _config(
            video_model="kling-3-pro",
            fal_extra_arguments={"generate_audio": True},
        )
        assert VideoClient(config)._build_arguments("s", "e", "m", 5)[
            "generate_audio"] is True


class TestRetiredPresets:
    def test_turbo_is_retired_because_it_has_no_end_frame(self):
        model = get_video_model("kling-3-turbo-pro")
        assert model.retired and model.end_frame_field == ""

    def test_a_project_pinned_to_it_still_loads(self):
        assert _config(video_model="kling-3-turbo-pro").video_model \
            == "kling-3-turbo-pro"

    def test_it_is_not_offered_and_cannot_be_baked_off(self):
        from ai_video_maker.config import ConfigError, config_for_model
        from ai_video_maker.video_models import selectable_models
        assert "kling-3-turbo-pro" not in [m.key for m in selectable_models()]
        with pytest.raises(ConfigError):
            config_for_model(_config(), "kling-3-turbo-pro")


class TestMovieCosts:
    def test_each_model_prices_the_same_seconds(self):
        from ai_video_maker.video_models import movie_costs
        rows = {r["key"]: r for r in movie_costs(200, "kling-2.5-turbo-pro")}
        assert rows["kling-2.5-turbo-pro"]["movie_usd"] == 14.0
        assert rows["kling-2.5-turbo-pro"]["current"] is True
        assert rows["kling-2.5-turbo-pro"]["delta_usd"] == 0
        assert rows["kling-3-pro"]["movie_usd"] == 22.4
        assert rows["kling-3-pro"]["delta_usd"] == 8.4
        assert rows["kling-4"]["movie_usd"] == 47.0
        assert "kling-3-turbo-pro" not in rows

    def test_an_unknown_current_model_gives_no_deltas(self):
        from ai_video_maker.video_models import movie_costs
        assert all(r["delta_usd"] is None for r in movie_costs(100, ""))


class TestConfigForModel:
    def test_only_the_model_changes(self):
        from ai_video_maker.config import config_for_model
        base = _config(video_model="kling-2.5-turbo-pro",
                       fal_negative_prompt="blurry", max_retries=7)
        other = config_for_model(base, "kling-3-pro")
        assert other.fal_model_id.endswith("v3/pro/image-to-video")
        assert other.pricing.clip_usd_per_second == 0.112
        assert other.fal_negative_prompt == "blurry" and other.max_retries == 7
        assert base.fal_model_id.endswith("v2.5-turbo/pro/image-to-video")
