"""openai_client transition-plan coercion: difficulty-derived durations,
pair_index re-alignment, and motion-prompt word budgets."""
from __future__ import annotations

from ai_video_maker.clients.openai_client import (
    OpenAIClient,
    _merge_cast,
    _motion_word_limit,
    _realign_by_pair_index,
    is_arrangement_swap,
    stages_a_crossing,
)
from ai_video_maker.models import Character


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

    def test_long_clips_capped_by_fraction_highest_difficulty_wins(self, config):
        # 5 of 6 pairs claim to be hard; only ceil(6*0.5)=3 stay long, and the
        # difficulty-5 pairs outrank the 4s (earliest 4 takes the last slot).
        data = {"transitions": [_item(4), _item(5), _item(4), _item(5),
                                _item(4), _item(1)]}
        assert _durations(_plans(config, data, 6)) == [10, 10, 5, 10, 5, 5]

    def test_tie_break_prefers_earlier_pairs(self, config):
        data = {"transitions": [_item(4), _item(4), _item(4)]}
        assert _durations(_plans(config, data, 3)) == [10, 10, 5]

    def test_long_clip_fraction_is_configurable(self, config):
        # The cap is a knob because "how many 10s clips" is a taste/cost call:
        # a real movie had hard pairs demoted to 5s and teleporting.
        data = {"transitions": [_item(5), _item(5), _item(5), _item(5)]}
        config.long_clip_max_fraction = 0.25
        assert _durations(_plans(config, data, 4)) == [10, 5, 5, 5]
        config.long_clip_max_fraction = 1.0
        assert _durations(_plans(config, data, 4)) == [10, 10, 10, 10]

    def test_long_clip_fraction_zero_forces_all_short(self, config):
        data = {"transitions": [_item(5), _item(5)]}
        config.long_clip_max_fraction = 0.0
        assert _durations(_plans(config, data, 2)) == [5, 5]

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


class TestIdentityPromptRules:
    """The planner/condense/reword prompts carry hard-won identity rules
    (appearance-only references, arrangement-swap staging). These strings are
    easy to lose in a prompt rewrite; a real order rendered morphing people
    before they existed, so their presence is pinned here."""

    def test_planner_bans_relationship_words(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "REFER TO PEOPLE BY APPEARANCE ONLY" in s
        assert "'the son'" in s and "'the father'" in s

    def test_planner_has_arrangement_swap_staging(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "ARRANGEMENT SWAP" in s
        assert "left-right arrangement" in s

    def test_condense_preserves_and_fixes_identity_phrasing(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._CONDENSE_MOTION_SYSTEM
        assert "epithet" in s
        assert "relationship words" in s

    def test_reword_keeps_identity_anchors(self):
        from ai_video_maker.clients import openai_client as oc
        assert "Identity anchors" in oc._REWORD_MOTION_SYSTEM


class TestPaceAndBeatPromptRules:
    """Pace/distance and beat-counting rules, pinned like the identity ones.

    Two real clips sprinted because their prompt spanned a route ("stroll
    side by side along the meadow trail until they stop beside the bench"),
    and a third crammed six beats into 46 words — under the 60-word cap, so
    the mechanical check let it through. Only word counts are enforced in
    code; these rules live in the prompts, which makes them easy to lose in
    a rewrite. Presence is pinned here, NOT model behaviour.
    """

    def test_planner_forbids_route_spanning_travel(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "PACE AND DISTANCE" in s
        # The distance must shrink; a gentler verb is explicitly not enough.
        assert "'stroll'" in s
        assert "ARRIVAL" in s

    def test_planner_requires_counting_beats_not_just_words(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "COUNT THE BEATS" in s
        # The worked example is the real 46-word/six-beat prompt.
        assert "46 words" in s

    def test_condense_collapses_routes_into_the_arrival(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._CONDENSE_MOTION_SYSTEM
        assert "ARRIVAL" in s
        assert "sprinting" in s

    def test_reword_keeps_pace_anchors(self):
        from ai_video_maker.clients import openai_client as oc
        assert "Pace anchors" in oc._REWORD_MOTION_SYSTEM


class TestMergeCast:
    """The cast list is the movie's identity anchor across planning calls.

    Reconcile re-plans only dirty pairs, so a later targeted call sees just
    a couple of frames. Feeding it the saved cast keeps its prompts naming
    people exactly as the rest of the movie does — but only if merging never
    rewrites what earlier prompts already baked in.
    """

    def _cast(self, *pairs):
        return [Character(id=i, epithet=e) for i, e in pairs]

    def test_new_people_are_appended(self):
        existing = self._cast(("bald-man", "the bald man in pink sunglasses"))
        merged = _merge_cast(existing, [{"id": "girl", "epithet": "the small girl"}])
        assert [(c.id, c.epithet) for c in merged] == [
            ("bald-man", "the bald man in pink sunglasses"),
            ("girl", "the small girl"),
        ]

    def test_existing_epithets_are_never_rewritten(self):
        # The saved epithet is already inside planned motion prompts; letting
        # a later call "improve" it would split one person's identity across
        # the movie — exactly the drift the cast exists to prevent.
        existing = self._cast(("bald-man", "the bald man in pink sunglasses"))
        merged = _merge_cast(
            existing, [{"id": "bald-man", "epithet": "the man"}]
        )
        assert [c.epithet for c in merged] == ["the bald man in pink sunglasses"]

    def test_recycled_cast_does_not_duplicate_anyone(self):
        # Models re-list the provided cast despite being told not to.
        existing = self._cast(("bald-man", "the bald man"))
        merged = _merge_cast(existing, [{"id": "other", "epithet": "The Bald Man"}])
        assert len(merged) == 1

    def test_missing_id_is_derived_from_the_epithet(self):
        merged = _merge_cast([], [{"epithet": "the woman in the red dress"}])
        assert len(merged) == 1 and merged[0].id

    def test_junk_entries_are_skipped(self):
        merged = _merge_cast([], ["nonsense", {"id": "x"}, {"epithet": "  "}, 7])
        assert merged == []

    def test_no_returned_cast_keeps_the_existing_one(self):
        existing = self._cast(("a", "the tall man"))
        assert _merge_cast(existing, None) == existing


class TestInSceneTextIsPreserved:
    """Text that is IN the photo stays; only generated text is unwanted.

    User call (2026-07-26): a shop sign, a birthday banner or a logo on a
    shirt is part of the scene and must survive the clip — the pipeline may
    only suppress text the video model invents. Easy to undo by tightening
    an artifact list, so both halves are pinned.
    """

    def test_planner_protects_text_already_in_the_frames(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "no NEW text appearing on screen" in s
        assert "birthday banner" in s

    def test_shipped_negative_prompt_targets_overlays_not_scene_text(self):
        # The repo's live config is the one that renders real orders, so it
        # is what this rule has to hold for. A bare "text"/"on-screen text"
        # term would tell the model to erase a sign that is in the photo.
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        negative = json.loads(
            (root / "config.json").read_text(encoding="utf-8")
        )["fal_negative_prompt"]
        terms = {t.strip() for t in negative.split(",")}
        assert "text overlay" in terms and "watermark" in terms
        assert "text" not in terms and "on-screen text" not in terms


MAN = "the bearded young man with short dark hair"
WOMAN = "the young woman with long brown hair"


class TestArrangementSwapDetection:
    """Who stands where is compared in CODE, not left to the planner.

    The planner sees the swap perfectly well and still writes hold-steady
    prose for it — twice on real orders, most recently a couple who trade
    sides between a boat and the salt flat and got "sit side by side, share
    a quick laugh, then both stand up ... step closer together, and come to
    rest standing side by side". Pinned in place, a swapped pair morphs into
    each other, so the planner now reports the left-to-right order of each
    frame as DATA and this comparison decides.
    """

    def test_traded_sides_is_a_swap(self):
        assert is_arrangement_swap([WOMAN, MAN], [MAN, WOMAN])

    def test_same_order_is_not_a_swap(self):
        assert not is_arrangement_swap([WOMAN, MAN], [WOMAN, MAN])

    def test_shorter_description_of_the_same_person_still_matches(self):
        # The planner rarely repeats an epithet word for word between the
        # two lists; a subset of the words is the same person.
        assert is_arrangement_swap(
            [WOMAN, MAN], ["the bearded man", "the woman with brown hair"]
        )

    def test_similar_but_distinct_people_are_not_merged(self):
        # Two men separated only by an adjective: the tokens must not
        # collapse, or an ordinary pair is read as a swap and buys a 10s clip.
        a, b = "the tall man in a red shirt", "the short man in a red shirt"
        assert not is_arrangement_swap([a, b], [a, b])
        assert is_arrangement_swap([a, b], [b, a])

    def test_three_people_reordering_counts(self):
        c = "the small girl with curly hair"
        assert is_arrangement_swap([WOMAN, MAN, c], [WOMAN, c, MAN])

    def test_a_person_only_in_one_frame_cannot_anchor_a_swap(self):
        assert not is_arrangement_swap([MAN], [WOMAN])
        assert not is_arrangement_swap([MAN, WOMAN], [WOMAN])

    def test_ambiguous_or_missing_lists_fall_through(self):
        # Anything unclear must behave exactly as before the guard existed.
        assert not is_arrangement_swap(None, None)
        assert not is_arrangement_swap([], [])
        assert not is_arrangement_swap(["", ""], ["", ""])
        # Two identical descriptions: which one moved is unknowable.
        assert not is_arrangement_swap([MAN, MAN], [MAN, MAN])

    def test_crossing_markers_tell_real_staging_from_hold_steady(self):
        assert not stages_a_crossing(
            "the bearded man and the woman stand up, step closer together, "
            "and come to rest side by side facing forward"
        )
        assert stages_a_crossing(
            "they walk forward past the camera and out of frame, then step "
            "back in one at a time"
        )
        assert stages_a_crossing(
            "the bearded man crosses in front of the woman with brown hair"
        )


class TestSwappedPairsAreForcedLong:
    """A swap can't be staged in five seconds, whatever the planner rated it."""

    def _swap_item(self, difficulty, motion="m"):
        return {
            "motion_prompt": motion, "difficulty": difficulty,
            "sound_prompt": "s",
            "start_order": [WOMAN, MAN], "end_order": [MAN, WOMAN],
        }

    def test_low_rated_swap_still_gets_the_long_clip(self, config, monkeypatch):
        # The real plan rated this pair easy and wrote hold-steady prose.
        monkeypatch.setattr(
            OpenAIClient, "_restage_swapped_pair",
            lambda self, prompt, *a, **k: prompt,
        )
        data = {"transitions": [self._swap_item(2)]}
        assert _durations(_plans(config, data, 1)) == [10]

    def test_non_swapped_pairs_are_unaffected(self, config):
        data = {"transitions": [_item(2), _item(3)]}
        assert _durations(_plans(config, data, 2)) == [5, 5]


class TestSwappedPairsAreRestaged:
    """The prompt itself is rewritten when it leaves swapped people in place."""

    def _data(self, motion, start=None, end=None):
        return {"transitions": [{
            "motion_prompt": motion, "difficulty": 3, "sound_prompt": "s",
            "start_order": start if start is not None else [WOMAN, MAN],
            "end_order": end if end is not None else [MAN, WOMAN],
        }]}

    def _capture(self, monkeypatch):
        calls: list[str] = []

        def fake(self, prompt, duration, start_order, end_order):
            calls.append(prompt)
            return ("the bearded man and the woman walk past the camera and "
                    "out of frame, then step back in one at a time")

        monkeypatch.setattr(OpenAIClient, "_restage_swapped_pair", fake)
        return calls

    def test_hold_steady_prompt_on_a_swap_is_restaged(self, config, monkeypatch):
        calls = self._capture(monkeypatch)
        motion = "they stand up and step closer together, side by side"
        plans = _plans(config, self._data(motion), 1)
        assert calls == [motion]
        assert "past the camera" in plans[0][0]

    def test_prompt_that_already_crosses_is_left_alone(self, config, monkeypatch):
        calls = self._capture(monkeypatch)
        motion = ("the bearded man crosses in front of the woman with brown "
                  "hair and they turn to the camera")
        plans = _plans(config, self._data(motion), 1)
        assert calls == []
        assert plans[0][0] == motion

    def test_no_swap_means_no_restage(self, config, monkeypatch):
        calls = self._capture(monkeypatch)
        motion = "they stand up and step closer together, side by side"
        _plans(config, self._data(motion, [WOMAN, MAN], [WOMAN, MAN]), 1)
        assert calls == []

    def test_restage_failure_keeps_the_original_prompt(self, config, monkeypatch):
        # The rewrite is a network call; planning must survive it failing.
        def boom(self, *a, **k):
            raise RuntimeError("no network")

        monkeypatch.setattr(OpenAIClient, "_ensure_client", boom)
        motion = "they stand up and step closer together, side by side"
        plans = _plans(config, self._data(motion), 1)
        assert plans[0][0] == motion


class TestMotionPromptsDescribeSubjectsOnly:
    """A motion prompt is about the people, never about the scenery.

    User call (2026-07-28): the two frames already fix the setting, and the
    video model interpolates the background on its own, so words spent on
    'the green Andean peaks' buy nothing while eating the word budget and
    pulling attention off the people. What the prompt owes is the SUBJECTS'
    path: an ordinary, walkable route from where each person stands in the
    start frame to where they stand in the end frame.

    The trigger was a real plan for a salt-flat -> Machu Picchu pair (a
    left-right swap AND a setting change): "…squeeze a little closer, share
    a quiet smile, then slowly step toward the camera and past it." It
    pinned swapped people in place (which morphs them) and ended on the exit
    instead of on the end frame.
    """

    def _planner(self) -> str:
        from ai_video_maker.clients import openai_client as oc
        return oc._MODE_A_SYSTEM

    def test_planner_forbids_describing_the_scene(self):
        s = self._planner()
        assert "SUBJECTS ONLY — NEVER DESCRIBE THE SCENE" in s
        # The old "WORLD FLOW" priority actively ASKED for scenery prose
        # ("light shifts, weather rolls in") — it must not come back.
        assert "WORLD FLOW" not in s
        assert "weather rolls in" not in s

    def test_planner_demands_a_walkable_path(self):
        s = self._planner()
        assert "NOBODY TELEPORTS" in s
        assert "ON THEIR OWN FEET" in s
        # The failure modes a position change without a path degrades into.
        for word in ("materialising", "floating", "sliding"):
            assert word in s

    def test_planner_requires_the_walk_back_in_after_an_exit(self):
        # The exit past the camera is half the staging; the plan that broke
        # stopped there, leaving the model to invent the arrival.
        s = self._planner()
        assert "only HALF the staging" in s
        assert "walk back in" in s

    def test_planner_forbids_ending_on_people_leaving(self):
        s = self._planner()
        assert "SIDE OF THE FRAME" in s
        assert "last beat is people LEAVING" in s

    def test_planner_demands_a_concrete_physical_verb(self):
        # "squeeze a little closer" is the canonical failure: the model
        # cannot animate an abstraction, so it morphs or drifts instead.
        s = self._planner()
        assert "NAME A CONCRETE PHYSICAL ACTION FOR EVERY PERSON WHO MOVES" in s
        assert "'squeeze a little closer'" in s
        for verb in ("walk", "climb", "crouch", "sprint", "slide"):
            assert verb in s

    def test_planner_ties_the_verb_to_the_real_pace(self):
        # The verb vocabulary must not become licence to sprint people
        # across a route — the failure that ruined two real clips.
        s = self._planner()
        assert "MATCH THE VERB TO THE REAL PACE AND GROUND" in s
        assert "never as a way to " in s

    def test_condense_keeps_the_verb_instead_of_generalising_it(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._CONDENSE_MOTION_SYSTEM
        assert "AS THE CONCRETE PHYSICAL VERB" in s
        assert "'share a moment'" in s

    def test_condense_drops_scenery_before_anything_a_person_does(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._CONDENSE_MOTION_SYSTEM
        assert "CUT EVERY WORD ABOUT THE SCENE FIRST" in s

    def test_reword_does_not_fall_back_to_describing_the_room(self):
        # The safety rewrite used to suggest "as the scene moves to the cozy
        # room" as its generic escape hatch — scenery prose by another name.
        from ai_video_maker.clients import openai_client as oc
        assert "cozy room" not in oc._REWORD_MOTION_SYSTEM


class TestStylingKeepsWholeSubjectsInFrame:
    """A portrait source must not come back with a head cut off.

    `_IMAGE_API_SIZE` is 1536x1024, so a vertical photo is ALWAYS reshaped
    to 16:9, and the styling prompt's "crop INTO the scene rather than
    invent scenery" bias is what turns that reshape into a crop. Left alone
    it took the height off the top: a real couple on a salt flat came back
    with the man's scalp sliced off at the frame edge. The fix lives in the
    shipped prompt (the one that styles real orders), so pin both halves —
    the no-cut rule and the direction the crop must come from.
    """

    def _prompts(self) -> dict:
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        return json.loads((root / "config.json").read_text(encoding="utf-8"))

    def test_shipped_style_prompt_forbids_cropping_people(self):
        style = self._prompts()["style_prompt"]
        assert "NEVER CUT A PERSON OFF" in style
        # The rule is worthless unless it beats the "keep them large / crop
        # in" bias that caused the crop in the first place.
        assert "OUTRANKS" in style

    def test_shipped_style_prompt_says_where_the_height_comes_off(self):
        style = self._prompts()["style_prompt"]
        assert "PORTRAIT) SOURCE IS THE DANGEROUS CASE" in style
        assert "take it off the BOTTOM" in style
        # Last resort when the people still don't fit: smaller, never cut.
        assert "pull the camera BACK" in style

    def test_shipped_style_prompt_does_not_invent_missing_body_parts(self):
        # A source that is ALREADY cropped must stay cropped — "never cut a
        # head" must not become "hallucinate the top of this head".
        style = self._prompts()["style_prompt"]
        assert "already cuts someone off, keep that framing" in style

    def test_from_idea_frames_keep_headroom_too(self):
        # Generated frames aren't reshaped from a photo, but they feed the
        # same movie and the same complaint.
        scratch = self._prompts()["scratch_style_prompt"]
        assert "never crop a head" in scratch


class TestCastPromptRules:
    """The CAST LIST contract lives in the planner prompt; pin its presence
    the way the other hard-won identity rules are pinned."""

    def test_planner_has_the_cast_list_contract(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "CAST LIST" in s
        assert "verbatim" in s

    def test_cast_is_part_of_the_planner_schema(self):
        from ai_video_maker.clients import openai_client as oc
        props = oc._TRANSITIONS_SCHEMA["properties"]
        assert "characters" in props
        assert "characters" in oc._TRANSITIONS_SCHEMA["required"]


class TestDistinguishingEpithetRules:
    """Epithets must separate similar people, and staging must say where.

    A real 16-clip movie had TWO bald men; 10 of its 16 prompts said only
    "the bald man", and the model mixed them up throughout. Separately, a
    pair that morphed under the collective "the right couple get out of the
    frame" rendered cleanly when the leavers were named individually with
    screen positions. Both live in the prompts, so both are pinned here.
    """

    def test_planner_bans_a_shared_feature_as_an_epithet(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "MUST DISTINGUISH" in s
        assert "two bald men" in s
        assert "in the light blue shirt" in s

    def test_planner_requires_position_and_direction(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._MODE_A_SYSTEM
        assert "SAY WHERE PEOPLE ARE" in s
        assert "the right couple" in s  # the collective that failed
        assert "INDIVIDUALLY" in s

    def test_condense_will_not_shorten_an_epithet_into_ambiguity(self):
        from ai_video_maker.clients import openai_client as oc
        s = oc._CONDENSE_MOTION_SYSTEM
        assert "NEVER" in s and "bald man in pink sunglasses" in s
        assert "Cut scenery before identity." in s

    def test_reword_keeps_the_distinguishing_part(self):
        from ai_video_maker.clients import openai_client as oc
        assert "bald man in pink sunglasses" in oc._REWORD_MOTION_SYSTEM
