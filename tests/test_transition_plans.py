"""openai_client transition-plan coercion: difficulty-derived durations,
pair_index re-alignment, and motion-prompt word budgets."""
from __future__ import annotations

from ai_video_maker.clients.openai_client import (
    _MODE_A_SYSTEM,
    CAMERA_SHIFT_MOTION_PROMPT,
    OpenAIClient,
    _coerce_frame_people,
    _tagged_people_block,
    _merge_cast,
    _motion_word_limit,
    _realign_by_pair_index,
    is_arrangement_swap,
    ends_offscreen,
    is_clothing_anchored,
    repair_cast_epithets,
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


class TestClipReviewCoercion:
    """Guards on what the clip reviewer proposes.

    Same division of labour as planning: the model observes, code decides.
    A suggestion the pipeline itself would have rejected must never reach the
    user as an "improvement".
    """

    def _review(self, config, data, motion="the bald man walks in", duration=5):
        return OpenAIClient(config)._coerce_review(data, motion, duration)

    def test_a_clean_review_passes_through(self, config):
        out = self._review(config, {
            "observed": "he walks in and sits down",
            "problems": [],
            "matches_prompt": True,
            "suggested_motion_prompt": "the bald man walks in",
            "suggested_duration": 5,
            "why": "no change needed",
        })
        assert out["problems"] == []
        assert out["matches_prompt"] is True
        # Nothing would change, so the panel must not offer an "apply".
        assert out["changes_clip"] is False

    def test_an_empty_suggestion_falls_back_to_the_current_prompt(self, config):
        # Never let "apply the suggestion" blank a transition.
        out = self._review(config, {"suggested_motion_prompt": "   ",
                                    "suggested_duration": 5})
        assert out["suggested_motion_prompt"] == "the bald man walks in"
        assert out["changes_clip"] is False

    def test_an_impossible_duration_keeps_the_clip_length(self, config):
        out = self._review(config, {"suggested_motion_prompt": "x",
                                    "suggested_duration": 7}, duration=10)
        assert out["suggested_duration"] == 10

    def test_a_longer_clip_is_a_change_worth_applying(self, config):
        out = self._review(config, {
            "suggested_motion_prompt": "the bald man walks in",
            "suggested_duration": 10,
        }, duration=5)
        assert out["changes_clip"] is True

    def test_an_over_budget_suggestion_is_condensed(self, config, monkeypatch):
        # 5s clips are capped at 35 words; the reviewer's rewrite is held to
        # the same budget a planned prompt is, via the same condense call.
        long_prompt = " ".join(["walks"] * 60)
        monkeypatch.setattr(
            OpenAIClient, "_condense_motion_prompt",
            lambda self, prompt, duration: "the bald man takes a few steps",
        )
        out = self._review(config, {
            "suggested_motion_prompt": long_prompt, "suggested_duration": 5,
        })
        assert out["suggested_motion_prompt"] == "the bald man takes a few steps"

    def test_problems_are_normalised_and_capped(self, config):
        out = self._review(config, {
            "problems": ["  slides   across  ", "", "morphs"] + ["x"] * 8,
            "suggested_motion_prompt": "y", "suggested_duration": 5,
        })
        assert out["problems"][:2] == ["slides across", "morphs"]
        assert len(out["problems"]) == 6

    def test_junk_fields_do_not_crash_the_review(self, config):
        out = self._review(config, {})
        assert out["observed"] == ""
        assert out["suggested_motion_prompt"] == "the bald man walks in"


class TestCastEpithetsSurviveTheMovie:
    """A cast epithet is used across a film built from photos years apart.

    "the boy in the striped shirt" describes one photograph perfectly and
    matches nobody in the next one, where he is in a swimsuit — so the video
    model hunts for a striped shirt, fails, and acts on whoever is nearest.
    Prompt guidance alone did not hold this (a real plan came back with
    "boy in yellow shirt", "boy in striped shirt", "girl in purple dress"),
    so code decides, exactly as it does for arrangement swaps.
    """

    def test_clothing_only_epithets_are_caught(self):
        for epithet in (
            "boy in striped shirt",
            "the girl in the purple dress",
            "man in blue shirt",
            "the woman in the red coat",
            "the kid in the yellow t-shirt",
            "the boy in the baseball cap",
        ):
            assert is_clothing_anchored(epithet), epithet

    def test_durable_epithets_are_left_alone(self):
        for epithet in (
            "the bald man",
            "the bald man in pink sunglasses",   # bald carries it
            "the small boy with curly hair",
            "the taller boy",
            "the woman with long dark hair",
            "the teenage girl",
            "the bearded man in a green jacket",  # beard carries it
        ):
            assert not is_clothing_anchored(epithet), epithet

    def test_an_epithet_with_no_clothing_at_all_is_never_touched(self):
        assert not is_clothing_anchored("the older woman")
        assert not is_clothing_anchored("")

    def _data(self, epithet, durable, motion=None):
        return {
            "characters": [{"id": "son2", "epithet": epithet,
                            "durable_epithet": durable}],
            "transitions": [{
                "pair_index": 1, "difficulty": 2,
                "motion_prompt": motion or f"{epithet} runs to the water",
                "sound_prompt": "waves",
                "start_order": [epithet], "end_order": [epithet],
            }],
        }

    def test_a_clothing_epithet_is_replaced_by_the_durable_one(self):
        data = self._data("boy in striped shirt", "the smaller boy with curly hair")
        swaps = repair_cast_epithets(data)
        assert len(swaps) == 1
        assert data["characters"][0]["epithet"] == "the smaller boy with curly hair"

    def test_the_prompts_of_the_same_plan_are_rewritten_too(self):
        # Otherwise the cast and the prompts would name the same person
        # differently from the very first plan.
        data = self._data("boy in striped shirt", "the smaller boy with curly hair")
        repair_cast_epithets(data)
        tr = data["transitions"][0]
        assert tr["motion_prompt"] == "the smaller boy with curly hair runs to the water"
        assert tr["start_order"] == ["the smaller boy with curly hair"]
        assert tr["end_order"] == ["the smaller boy with curly hair"]

    def test_a_durable_epithet_is_kept_as_written(self):
        data = self._data("the bald man in pink sunglasses", "the bald man")
        assert repair_cast_epithets(data) == []
        assert data["characters"][0]["epithet"] == "the bald man in pink sunglasses"

    def test_a_replacement_that_is_also_clothing_is_refused(self):
        data = self._data("boy in striped shirt", "the boy in the blue shorts")
        assert repair_cast_epithets(data) == []
        assert data["characters"][0]["epithet"] == "boy in striped shirt"

    def test_a_missing_replacement_leaves_the_original(self):
        data = self._data("boy in striped shirt", "")
        assert repair_cast_epithets(data) == []
        assert data["characters"][0]["epithet"] == "boy in striped shirt"

    def test_a_replacement_that_would_duplicate_someone_is_refused(self):
        # Two people sharing one epithet is worse than a fragile epithet:
        # neither can then be told from the other in any prompt.
        data = {
            "characters": [
                {"id": "son1", "epithet": "the smaller boy with curly hair",
                 "durable_epithet": "the smaller boy with curly hair"},
                {"id": "son2", "epithet": "boy in striped shirt",
                 "durable_epithet": "the smaller boy with curly hair"},
            ],
            "transitions": [],
        }
        assert repair_cast_epithets(data) == []
        assert data["characters"][1]["epithet"] == "boy in striped shirt"

    def test_junk_input_does_not_crash_the_repair(self):
        assert repair_cast_epithets({}) == []
        assert repair_cast_epithets({"characters": "nope"}) == []
        assert repair_cast_epithets({"characters": [None, 3]}) == []


class TestCastPromptForbidsClothing:
    """The prompt side of the same rule, pinned against the live text."""

    def test_the_planner_is_told_never_to_anchor_on_clothing(self):
        system = _MODE_A_SYSTEM.lower()
        assert "never anchor it to clothing" in system
        assert "durable_epithet" in system
        # The failing wording from the real plan is named as forbidden.
        assert "the boy in the yellow shirt" in system

    def test_relative_size_is_offered_when_durable_traits_match(self):
        system = _MODE_A_SYSTEM.lower()
        assert "the taller boy" in system and "the smaller boy" in system


class TestConfirmedIdentities:
    """Human tags are FACT: they outrank the model's reading of the frames.

    Identity is what the planner gets wrong on its own — it decides from the
    pixels whether the child in frame 7 is the child from frame 6, and a
    wrong guess is what makes the video model morph one person into another.
    """

    def test_the_roster_of_each_tagged_frame_is_stated(self):
        block = _tagged_people_block([
            ["the bald man", "the smaller boy"],
            ["the bald man"],
        ])
        assert "Frame 001, left to right: the bald man, the smaller boy" in block
        assert "Frame 002, left to right: the bald man" in block
        assert "OVERRULES" in block

    def test_untagged_frames_are_simply_absent(self):
        block = _tagged_people_block([["the bald man"], None])
        assert "Frame 001" in block
        assert "Frame 002" not in block

    def test_the_same_people_are_declared_continuous(self):
        block = _tagged_people_block([["the bald man"], ["the bald man"]])
        assert "SAME person (the bald man)" in block
        assert "animate them continuously" in block

    def test_a_completely_different_cast_forces_an_exit_and_entrance(self):
        # The creepy-morph case: two people who look alike but are not the
        # same human. The planner must never be left to guess this.
        block = _tagged_people_block([["the smaller boy"], ["the taller boy"]])
        assert "COMPLETELY DIFFERENT people" in block
        assert "NEVER let one turn into the other" in block
        assert "exit and an entrance" in block

    def test_a_partial_overlap_names_who_stays_leaves_and_arrives(self):
        block = _tagged_people_block([
            ["the bald man", "the smaller boy"],
            ["the bald man", "the teenage girl"],
        ])
        assert "the bald man stay(s)" in block
        assert "the smaller boy leave(s)" in block
        assert "the teenage girl arrive(s)" in block

    def test_an_empty_frame_is_not_the_same_as_an_untagged_one(self):
        block = _tagged_people_block([[], ["the bald man"]])
        assert "Frame 001, left to right: (nobody)" in block


class TestTagsDriveSwapDetection:
    """Swap detection is only as good as the positions it is given."""

    def _plans(self, config, data, count=1, frame_people=None):
        return OpenAIClient(config)._coerce_transition_plans(
            data, count, None, frame_people=frame_people,
        )

    def _data(self, start, end, motion="they stand together"):
        return {"transitions": [{
            "pair_index": 1, "difficulty": 1, "motion_prompt": motion,
            "sound_prompt": "", "start_order": start, "end_order": end,
        }]}

    def test_tags_reveal_a_swap_the_model_missed(self, config, monkeypatch):
        # The model reported no swap; the human's tags say they traded sides.
        # A swap forces the long clip, which is what makes an exit and
        # re-entry physically possible.
        monkeypatch.setattr(
            OpenAIClient, "_restage_swapped_pair",
            lambda self, prompt, duration, s, e: "one crosses in front of the other",
        )
        plans = self._plans(
            config,
            self._data(["the bald man", "the woman"],
                       ["the bald man", "the woman"]),
            frame_people=[["the bald man", "the woman"],
                          ["the woman", "the bald man"]],
        )
        motion, duration, _ = plans[0]
        assert duration == 10
        assert motion == "one crosses in front of the other"

    def test_tags_overrule_a_swap_the_model_imagined(self, config):
        # The model's own lists claim a swap; the tags say nobody moved.
        plans = self._plans(
            config,
            self._data(["the bald man", "the woman"],
                       ["the woman", "the bald man"]),
            frame_people=[["the bald man", "the woman"],
                          ["the bald man", "the woman"]],
        )
        assert plans[0][1] == 5  # not forced long
        assert plans[0][0] == "they stand together"  # not restaged

    def test_a_half_tagged_pair_falls_back_to_the_model(self, config, monkeypatch):
        monkeypatch.setattr(
            OpenAIClient, "_restage_swapped_pair",
            lambda self, prompt, duration, s, e: "one crosses in front of the other",
        )
        plans = self._plans(
            config,
            self._data(["the bald man", "the woman"],
                       ["the woman", "the bald man"]),
            frame_people=[["the bald man", "the woman"], None],
        )
        assert plans[0][1] == 10  # the model's reading still counts


class TestIdentifyPeopleCoercion:
    """The pre-fill is a draft; everything questionable is dropped, not fixed."""

    def _coerce(self, data, count=2, known=("dad", "son1")):
        return _coerce_frame_people(data, count, set(known))

    def test_people_are_placed_on_their_frame(self):
        out = self._coerce({"frames": [
            {"frame_index": 1, "people": [{"id": "dad", "x": 0.3, "y": 0.4}]},
            {"frame_index": 2, "people": []},
        ]})
        assert out[0] == [{"id": "dad", "x": 0.3, "y": 0.4}]
        assert out[1] == []

    def test_invented_people_are_dropped(self):
        # A tag naming nobody in the cast cannot resolve to an epithet.
        out = self._coerce({"frames": [
            {"frame_index": 1, "people": [{"id": "grandma", "x": 0.5, "y": 0.5}]},
        ]})
        assert out[0] == []

    def test_the_same_person_cannot_stand_in_two_places(self):
        out = self._coerce({"frames": [{"frame_index": 1, "people": [
            {"id": "dad", "x": 0.2, "y": 0.5},
            {"id": "dad", "x": 0.8, "y": 0.5},
        ]}]})
        assert len(out[0]) == 1

    def test_coordinates_are_clamped_into_the_frame(self):
        out = self._coerce({"frames": [{"frame_index": 1, "people": [
            {"id": "dad", "x": 1.7, "y": -3},
        ]}]})
        assert out[0] == [{"id": "dad", "x": 1.0, "y": 0.0}]

    def test_out_of_range_and_junk_entries_are_ignored(self):
        out = self._coerce({"frames": [
            {"frame_index": 9, "people": [{"id": "dad", "x": 0.5, "y": 0.5}]},
            {"frame_index": "x", "people": []},
            "nope",
        ]})
        assert out == [[], []]

    def test_a_missing_response_yields_one_empty_list_per_frame(self):
        assert self._coerce({}) == [[], []]


class TestAnExitIsOnlyHalfTheStaging:
    """A swapped pair must be moved AND land on the end frame.

    The real failure: a five-person family swap was restaged as "The smaller
    boy with wavy hair near the right walks forward past the camera and exits
    left, ending offscreen on the left." It passed the old crossing check
    (it says "past the camera"), but the end frame shows that boy standing in
    the group — so the clip cannot end where the prompt says, and four of the
    five people were never mentioned at all.
    """

    THE_REAL_ONE = (
        "The smaller boy with wavy hair near the right walks forward past the "
        "camera and exits left, ending offscreen on the left."
    )

    def test_the_prompt_that_caused_this_is_rejected(self):
        assert ends_offscreen(self.THE_REAL_ONE)
        assert not stages_a_crossing(self.THE_REAL_ONE)

    def test_an_exit_with_a_return_is_a_complete_staging(self):
        prompt = (
            "the bald man walks past the camera, then steps back in from the "
            "right beside the woman with curly hair"
        )
        assert stages_a_crossing(prompt)
        assert not ends_offscreen(prompt)

    def test_one_person_passing_another_needs_no_exit(self):
        prompt = "the smaller boy crosses in front of the girl and stops on her left"
        assert stages_a_crossing(prompt)
        assert not ends_offscreen(prompt)

    def test_a_return_before_an_exit_still_ends_offscreen(self):
        # Order matters: coming back and then leaving again ends outside.
        prompt = (
            "the taller boy steps back in from the left, then walks out of frame"
        )
        assert ends_offscreen(prompt)

    def test_a_prompt_that_never_leaves_the_frame_is_not_offscreen(self):
        assert not ends_offscreen("the bald man turns to face the camera")

    def test_the_group_regathering_counts_as_a_return(self):
        prompt = (
            "the five of them step forward past the camera, then gather again "
            "with the taller boy beside the man with short dark hair"
        )
        assert stages_a_crossing(prompt)
        assert not ends_offscreen(prompt)


class TestSwappedPairsAreRestagedCompletely:
    def _restage(self, config, reply, monkeypatch):
        client = OpenAIClient(config)

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            @staticmethod
            def create(**kwargs):
                return _Resp()

        class _Client:
            class chat:
                completions = _Completions()

        monkeypatch.setattr(client, "_ensure_client", lambda: _Client())
        return client._restage_swapped_pair(
            "they stand together", 10,
            ["the bald man", "the woman"], ["the woman", "the bald man"],
        )

    def test_a_rewrite_that_ends_offscreen_is_refused(self, config, monkeypatch):
        original = "they stand together"
        assert self._restage(
            config, TestAnExitIsOnlyHalfTheStaging.THE_REAL_ONE, monkeypatch
        ) == original

    def test_a_complete_rewrite_is_accepted(self, config, monkeypatch):
        good = (
            "the bald man walks past the camera and steps back in from the "
            "left; the woman crosses behind him to the right"
        )
        assert self._restage(config, good, monkeypatch) == good


class TestOffscreenEndingsAreRewritten:
    """An exit with no return is enforced in CODE, not just in the prompt.

    The rule lives in _MODE_A_SYSTEM, and the planner writes them anyway: a
    full --replan-all of a real 44-pair movie (am-130826-pcfd) came back with
    the SAME 29 pairs ending on an exit, because nothing rejected them. Only
    the arrangement-swap restage checked, and it gave up 6 times out of 7.
    """

    # Straight off that movie's 0e_to_0f card, planned with correct tags.
    THE_REAL_ONE = (
        "The teenage boy with short brown hair, teenage girl with long dark "
        "hair, woman with long dark hair, and man with short brown hair stand "
        "up, push back their chairs, and walk out of frame to the left as the "
        "remaining five stay seated."
    )
    LANDS = (
        "the teenage boy with short brown hair walks out past the camera and "
        "steps back in from the left beside the woman with long dark hair"
    )
    # The same landing rewrite, still over budget — the walk-back-in costs
    # words, which is exactly why the cap has to run after it.
    LONG_LANDS = LANDS + " " + " ".join(["slowly"] * 60)

    def _data(self, motion, difficulty=1):
        return {"transitions": [{
            "pair_index": 1, "difficulty": difficulty,
            "motion_prompt": motion, "sound_prompt": "",
        }]}

    def _client(self, config, reply, monkeypatch, boom=False):
        client = OpenAIClient(config)

        class _Msg:
            content = reply

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            @staticmethod
            def create(**kwargs):
                if boom:
                    raise RuntimeError("the model is down")
                return _Resp()

        class _Client:
            class chat:
                completions = _Completions()

        monkeypatch.setattr(client, "_ensure_client", lambda: _Client())
        return client

    def test_the_real_prompt_is_detected(self):
        assert ends_offscreen(self.THE_REAL_ONE)

    def test_an_offscreen_ending_is_rewritten(self, config, monkeypatch):
        client = self._client(config, self.LANDS, monkeypatch)
        motion, _, _ = client._coerce_transition_plans(
            self._data(self.THE_REAL_ONE), 1, None
        )[0]
        assert motion == self.LANDS
        assert not ends_offscreen(motion)

    def test_a_landing_prompt_is_left_alone(self, config, monkeypatch):
        """No rewrite call at all when the prompt already lands."""
        good = "the bald man turns to face the camera and smiles"
        calls = []
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: calls.append(p) or p,
        )
        motion, _, _ = _plans(config, self._data(good), 1)[0]
        assert motion == good
        assert calls == []

    def test_a_rewrite_that_still_ends_offscreen_is_refused(
        self, config, monkeypatch
    ):
        """Taking it would clear the panel's warning without fixing the clip."""
        client = self._client(
            config, "they all walk out of frame to the left", monkeypatch
        )
        assert client._complete_offscreen_ending(
            self.THE_REAL_ONE, 10
        ) == self.THE_REAL_ONE

    def test_an_empty_rewrite_is_refused(self, config, monkeypatch):
        client = self._client(config, "   ", monkeypatch)
        assert client._complete_offscreen_ending(
            self.THE_REAL_ONE, 10
        ) == self.THE_REAL_ONE

    def test_a_failed_call_never_sinks_planning(self, config, monkeypatch):
        client = self._client(config, "unused", monkeypatch, boom=True)
        assert client._complete_offscreen_ending(
            self.THE_REAL_ONE, 10
        ) == self.THE_REAL_ONE

    def test_the_rewrite_runs_before_the_word_cap(self, config, monkeypatch):
        """The walk-back-in costs words, so it must be added before trimming."""
        order = []
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: (
                order.append("ending")
                or TestOffscreenEndingsAreRewritten.LONG_LANDS
            ),
        )
        monkeypatch.setattr(
            OpenAIClient, "_condense_motion_prompt",
            lambda self, p, d: order.append("condense") or "short landing prompt",
        )
        long_offscreen = self.THE_REAL_ONE + " " + " ".join(["extra"] * 60)
        _plans(config, self._data(long_offscreen), 1)
        assert order == ["ending", "condense"]

    def _reopening_condense(self, monkeypatch, endings):
        """Condense always re-opens the ending; `endings` feeds the rewriter.

        Measured on am-130826-pcfd: condensing put 7 of the 9 remaining
        endings back offscreen, despite being told to drop a whole exit
        before dropping its return.
        """
        calls = iter(endings)
        seen = []
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: seen.append(p) or next(calls),
        )
        monkeypatch.setattr(
            OpenAIClient, "_condense_motion_prompt",
            lambda self, p, d: "the boy walks out of frame",
        )
        return seen

    def test_condensing_that_reopens_the_ending_is_repaired(
        self, config, monkeypatch
    ):
        """The prompt is short by now, so the walk-back-in has room."""
        short_landing = "the boy steps back in from the left beside the woman"
        seen = self._reopening_condense(
            monkeypatch, [self.LONG_LANDS, short_landing]
        )
        long_offscreen = self.THE_REAL_ONE + " " + " ".join(["extra"] * 60)
        motion, _, _ = _plans(config, self._data(long_offscreen), 1)[0]
        assert motion == short_landing
        assert not ends_offscreen(motion)
        # The repair ran on the CONDENSED text, not the original.
        assert seen[1] == "the boy walks out of frame"

    def test_a_failed_repair_never_keeps_the_over_budget_prompt(
        self, config, monkeypatch, caplog
    ):
        """Never swap one failure for another.

        The long prompt lands but blows the word cap, which renders as a
        rushed blur that reads like a cut — so it is not the answer either.
        The pair falls through to the camera-move last resort instead, and
        the reason is logged on the way past.
        """
        self._reopening_condense(
            monkeypatch, [self.LONG_LANDS, "they all walk out of frame"]
        )
        long_offscreen = self.THE_REAL_ONE + " " + " ".join(["extra"] * 60)
        with caplog.at_level("WARNING"):
            motion, _, _ = _plans(config, self._data(long_offscreen), 1)[0]
        assert motion != self.LONG_LANDS  # the over-budget one is refused
        assert motion == CAMERA_SHIFT_MOTION_PROMPT
        assert not ends_offscreen(motion)
        assert "did not land it" in caplog.text

    def test_no_repair_when_condensing_kept_the_landing(
        self, config, monkeypatch
    ):
        calls = []
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: (
                calls.append(p) or TestOffscreenEndingsAreRewritten.LONG_LANDS
            ),
        )
        monkeypatch.setattr(
            OpenAIClient, "_condense_motion_prompt",
            lambda self, p, d: "the boy steps back in from the left",
        )
        long_offscreen = self.THE_REAL_ONE + " " + " ".join(["extra"] * 60)
        motion, _, _ = _plans(config, self._data(long_offscreen), 1)[0]
        assert motion == "the boy steps back in from the left"
        assert len(calls) == 1  # the first rewrite only, no repair

    def test_the_end_frame_people_are_given_to_the_rewriter(
        self, config, monkeypatch
    ):
        """It has to know where people end up to write them back in."""
        seen = {}
        client = OpenAIClient(config)

        class _Msg:
            content = TestOffscreenEndingsAreRewritten.LANDS

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            usage = None

        class _Completions:
            @staticmethod
            def create(**kwargs):
                seen.update(kwargs)
                return _Resp()

        class _Client:
            class chat:
                completions = _Completions()

        monkeypatch.setattr(client, "_ensure_client", lambda: _Client())
        client._complete_offscreen_ending(
            self.THE_REAL_ONE, 10, ["the bald man", "the woman"]
        )
        user = seen["messages"][1]["content"]
        assert "the bald man, the woman" in user
        assert "Word limit: 60" in user


class TestNothingShipsWithAnOffscreenEnding:
    """User call, 2026-08-13: "I don't want any clip to have an offscreen."

    Every rewrite above may decline, and on hard pairs they do — nine people
    cannot walk out and back in inside one clip. The last resort moves the
    CAMERA instead of the people: a deliberate, narrow exception to the
    no-camera-moves rule, taken because the real alternative on those pairs
    is a cut, not a better staging.
    """

    OFFSCREEN = "the nine of them walk out of frame to the left"

    def _data(self, motion):
        return {"transitions": [{
            "pair_index": 1, "difficulty": 1,
            "motion_prompt": motion, "sound_prompt": "",
        }]}

    def _refuse_everything(self, monkeypatch):
        """Every rewrite declines, exactly as it does on the hard pairs."""
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: p,
        )
        monkeypatch.setattr(
            OpenAIClient, "_condense_motion_prompt", lambda self, p, d: p,
        )

    def test_the_fallback_itself_lands(self):
        """It would be absurd for the last resort to trip the same check."""
        assert not ends_offscreen(CAMERA_SHIFT_MOTION_PROMPT)

    def test_an_unstageable_pair_falls_back_to_a_camera_move(
        self, config, monkeypatch
    ):
        self._refuse_everything(monkeypatch)
        motion, _, _ = _plans(config, self._data(self.OFFSCREEN), 1)[0]
        assert motion == CAMERA_SHIFT_MOTION_PROMPT

    def test_small_groups_get_it_too(self, config, monkeypatch):
        """The >5 case is where it fires most, not a threshold to hide behind.

        Three of six real survivors on am-130826-pcfd had 5, 4 and 3 people;
        gating on group size would have left those clips ending offscreen.
        """
        self._refuse_everything(monkeypatch)
        motion, _, _ = _plans(
            config, self._data("the two of them walk out of frame"), 1
        )[0]
        assert motion == CAMERA_SHIFT_MOTION_PROMPT

    def test_it_is_a_last_resort_not_a_shortcut(self, config, monkeypatch):
        """A prompt the rewrite lands keeps its staging — people, not camera."""
        landed = "the bald man steps back in from the left beside the woman"
        monkeypatch.setattr(
            OpenAIClient, "_complete_offscreen_ending",
            lambda self, p, d, e=None: landed,
        )
        motion, _, _ = _plans(config, self._data(self.OFFSCREEN), 1)[0]
        assert motion == landed

    def test_a_prompt_that_never_left_the_frame_is_untouched(self, config):
        good = "the bald man turns to face the camera and smiles"
        assert _plans(config, self._data(good), 1)[0][0] == good

    def test_no_plan_can_leave_the_coercer_ending_offscreen(
        self, config, monkeypatch
    ):
        """The guarantee, stated as one assertion over a mixed batch."""
        self._refuse_everything(monkeypatch)
        data = {"transitions": [
            {"pair_index": n + 1, "difficulty": 1, "sound_prompt": "",
             "motion_prompt": m}
            for n, m in enumerate([
                self.OFFSCREEN,
                "the boy exits left",
                "she walks out past the camera",
                "the bald man turns and smiles",
            ])
        ]}
        plans = _plans(config, data, 4)
        assert not any(ends_offscreen(m) for m, _, _ in plans)
