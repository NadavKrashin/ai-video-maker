"""openai_client._coerce_transition_plans: durations follow the hardness verdict."""
from __future__ import annotations

from ai_video_maker.clients.openai_client import OpenAIClient


def _plans(config, data, count, default_duration=None):
    return OpenAIClient(config)._coerce_transition_plans(
        data, count, default_duration
    )


class TestCoerceTransitionPlans:
    def test_hard_transition_forces_10_over_inconsistent_5(self, config):
        data = {"transitions": [
            {"motion_prompt": "m", "hard_transition": True, "duration": 5,
             "sound_prompt": "s"},
        ]}
        assert _plans(config, data, 1) == [("m", 10, "s")]

    def test_easy_transition_keeps_model_duration(self, config):
        data = {"transitions": [
            {"motion_prompt": "m", "hard_transition": False, "duration": 5,
             "sound_prompt": "s"},
            {"motion_prompt": "m2", "hard_transition": False, "duration": 10,
             "sound_prompt": "s2"},
        ]}
        assert _plans(config, data, 2) == [("m", 5, "s"), ("m2", 10, "s2")]

    def test_default_duration_overrides_hardness(self, config):
        data = {"transitions": [
            {"motion_prompt": "m", "hard_transition": True, "duration": 10,
             "sound_prompt": "s"},
        ]}
        assert _plans(config, data, 1, default_duration=5) == [("m", 5, "s")]

    def test_missing_items_fall_back_to_config(self, config):
        plans = _plans(config, {"transitions": []}, 2)
        assert plans == [
            (config.motion_prompt, config.duration, ""),
            (config.motion_prompt, config.duration, ""),
        ]
