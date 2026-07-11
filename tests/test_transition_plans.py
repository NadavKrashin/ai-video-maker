"""openai_client transition-plan coercion: difficulty-derived durations and
pair_index re-alignment."""
from __future__ import annotations

from ai_video_maker.clients.openai_client import (
    OpenAIClient,
    _realign_by_pair_index,
)


def _plans(config, data, count, default_duration=None):
    return OpenAIClient(config)._coerce_transition_plans(
        data, count, default_duration
    )


def _item(difficulty, motion="m", sound="s"):
    return {"motion_prompt": motion, "difficulty": difficulty,
            "sound_prompt": sound}


def _durations(plans):
    return [d for _, d, _ in plans]


class TestCoerceTransitionPlans:
    def test_difficulty_4_and_5_get_long_clips(self, config):
        data = {"transitions": [_item(1), _item(3), _item(4), _item(5),
                                _item(2), _item(3)]}
        assert _durations(_plans(config, data, 6)) == [5, 5, 10, 10, 5, 5]

    def test_long_clips_capped_at_a_third_highest_difficulty_wins(self, config):
        # 5 of 6 pairs claim to be hard; only ceil(6/3)=2 stay long, and the
        # difficulty-5 pairs outrank the 4s.
        data = {"transitions": [_item(4), _item(5), _item(4), _item(5),
                                _item(4), _item(1)]}
        assert _durations(_plans(config, data, 6)) == [5, 10, 5, 10, 5, 5]

    def test_tie_break_prefers_earlier_pairs(self, config):
        data = {"transitions": [_item(4), _item(4), _item(4)]}
        assert _durations(_plans(config, data, 3)) == [10, 5, 5]

    def test_default_duration_overrides_difficulty(self, config):
        data = {"transitions": [_item(5)]}
        assert _durations(_plans(config, data, 1, default_duration=5)) == [5]

    def test_unrated_or_malformed_pairs_stay_short(self, config):
        data = {"transitions": [{"motion_prompt": "m", "sound_prompt": "s"},
                                {"motion_prompt": "m", "difficulty": "hard",
                                 "sound_prompt": "s"}]}
        assert _durations(_plans(config, data, 2)) == [5, 5]

    def test_missing_items_fall_back_to_config_motion(self, config):
        plans = _plans(config, {"transitions": []}, 2)
        assert plans == [
            (config.motion_prompt, 5, ""),
            (config.motion_prompt, 5, ""),
        ]

    def test_declared_pair_index_wins_over_array_position(self, config):
        # The model slipped: array position 0 describes pair 2 and vice versa.
        data = {"transitions": [
            dict(_item(3, motion="into painting"), pair_index=2),
            dict(_item(3, motion="into park"), pair_index=1),
        ]}
        plans = _plans(config, data, 2)
        assert [m for m, _, _ in plans] == ["into park", "into painting"]


class TestRealignByPairIndex:
    def test_reorders_shuffled_items(self):
        items = [{"pair_index": 2, "motion_prompt": "b"},
                 {"pair_index": 1, "motion_prompt": "a"},
                 {"pair_index": 3, "motion_prompt": "c"}]
        assert [i["motion_prompt"] for i in _realign_by_pair_index(items, 3)] \
            == ["a", "b", "c"]

    def test_missing_declared_pair_becomes_empty_slot(self):
        items = [{"pair_index": 3, "motion_prompt": "c"},
                 {"pair_index": 1, "motion_prompt": "a"}]
        assert _realign_by_pair_index(items, 3)[1] == {}

    def test_falls_back_to_positional_without_indices(self):
        items = [{"motion_prompt": "a"}, {"motion_prompt": "b"}]
        assert _realign_by_pair_index(items, 2) is items

    def test_falls_back_on_duplicate_or_out_of_range_indices(self):
        dup = [{"pair_index": 1}, {"pair_index": 1}]
        assert _realign_by_pair_index(dup, 2) is dup
        oob = [{"pair_index": 0}, {"pair_index": 5}]
        assert _realign_by_pair_index(oob, 2) is oob
