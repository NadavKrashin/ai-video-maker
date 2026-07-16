"""openai_client transition-plan coercion: difficulty-derived durations,
pair_index re-alignment, and motion-prompt word budgets."""
from __future__ import annotations

from ai_video_maker.clients.openai_client import (
    OpenAIClient,
    _motion_word_limit,
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


def _words(n):
    return " ".join(["walks"] * n)


class TestMotionWordBudget:
    """Over-budget motion prompts get condensed; the budget follows the
    clip's DERIVED duration (a real plan wrote 79-113 words for every 5s
    clip despite the prompt-side beat-budget rule)."""

    def _client_with_recorder(self, config, condensed="short"):
        client = OpenAIClient(config)
        calls = []

        def fake_condense(prompt, duration):
            calls.append((prompt, duration))
            return condensed

        client._condense_motion_prompt = fake_condense
        return client, calls

    def test_over_budget_5s_prompt_is_condensed(self, config):
        client, calls = self._client_with_recorder(config)
        data = {"transitions": [_item(1, motion=_words(84))]}
        plans = client._coerce_transition_plans(data, 1, None)
        assert calls == [(_words(84), 5)]
        assert plans[0][0] == "short"

    def test_under_budget_prompt_is_left_alone(self, config):
        client, calls = self._client_with_recorder(config)
        data = {"transitions": [_item(1, motion=_words(35))]}
        plans = client._coerce_transition_plans(data, 1, None)
        assert calls == []
        assert plans[0][0] == _words(35)

    def test_long_clip_gets_the_larger_budget(self, config):
        # 50 words is over the 5s budget but inside the 10s budget.
        client, calls = self._client_with_recorder(config)
        data = {"transitions": [_item(5, motion=_words(50))]}
        plans = client._coerce_transition_plans(data, 1, None)
        assert plans[0][1] == 10
        assert calls == []

    def test_forced_duration_sets_the_budget(self, config):
        # The same 50-word prompt is over budget when --duration 5 forces
        # the clip short.
        client, calls = self._client_with_recorder(config)
        data = {"transitions": [_item(5, motion=_words(50))]}
        client._coerce_transition_plans(data, 1, default_duration=5)
        assert calls == [(_words(50), 5)]

    def test_condense_failure_keeps_the_original_prompt(self, config):
        # No API key / client failure must never hard-stop planning.
        client = OpenAIClient(config)

        def boom():
            raise RuntimeError("no client in tests")

        client._ensure_client = boom
        assert client._condense_motion_prompt(_words(84), 5) == _words(84)


class TestMotionWordLimit:
    def test_budgets_scale_with_duration(self):
        assert _motion_word_limit(5) < _motion_word_limit(10)

    def test_unknown_duration_gets_most_permissive_budget(self):
        assert _motion_word_limit(7) == _motion_word_limit(10)


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
