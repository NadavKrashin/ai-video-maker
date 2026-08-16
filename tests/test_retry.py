"""retry.py: moderation classification and reword-recovery loop."""
from __future__ import annotations

import pytest

from ai_video_maker import retry
from ai_video_maker.retry import (
    is_moderation_error,
    is_output_moderation,
    is_quota_exhausted_error,
    is_rate_limit_error,
    is_retryable_error,
    moderation_failure_hint,
    with_retries,
    with_reword_recovery,
)

# The real fal/Kling rejection recorded from a failed clip job.
_FAL_MODERATION = (
    "[{'loc': ['body'], 'msg': 'The content could not be processed because it "
    "contained material flagged by a content checker.', "
    "'type': 'content_policy_violation', "
    "'url': 'https://docs.fal.ai/errors#content_policy_violation'}]"
)

# The real OpenAI rejection recorded while styling frame 18 of a live order —
# an ordinary photo of three kids holding popcorn buckets in a Disney store.
_OUTPUT_MODERATION = (
    "Error code: 400 - {'error': {'message': 'Your request was rejected by the "
    "safety system.', 'type': 'image_generation_user_error', 'param': None, "
    "'code': 'moderation_blocked', 'moderation_details': "
    "{'moderation_stage': 'output', 'categories': ['other']}}}"
)


class TestOutputStageModeration:
    """Which stage fired decides whether rewording is worth anything."""

    def test_output_stage_is_recognised(self):
        assert is_output_moderation(RuntimeError(_OUTPUT_MODERATION))

    def test_a_prompt_stage_block_is_not_output_stage(self):
        prompt_stage = _OUTPUT_MODERATION.replace("'output'", "'input'")
        assert is_moderation_error(RuntimeError(prompt_stage))
        assert not is_output_moderation(RuntimeError(prompt_stage))

    def test_fal_rejections_are_never_output_stage(self):
        assert not is_output_moderation(RuntimeError(_FAL_MODERATION))

    def test_the_hint_says_which_lever_moves_it(self):
        hint = moderation_failure_hint(RuntimeError(_OUTPUT_MODERATION))
        assert "Rewording cannot fix" in hint
        assert "Crop" in hint  # what to actually do

    def test_a_non_moderation_error_gets_no_hint(self):
        assert moderation_failure_hint(RuntimeError("500 server error")) == ""


class TestIsModerationError:
    def test_openai_markers(self):
        assert is_moderation_error(RuntimeError("400 moderation_blocked"))
        assert is_moderation_error(RuntimeError("rejected by our safety system"))

    def test_fal_kling_markers(self):
        assert is_moderation_error(RuntimeError(_FAL_MODERATION))

    def test_ordinary_errors_are_not_moderation(self):
        assert not is_moderation_error(RuntimeError("500 internal server error"))
        assert not is_moderation_error(RuntimeError("connection reset by peer"))

    def test_moderation_is_never_plain_retried(self):
        # Reword recovery owns these; the backoff loop must fail fast.
        assert not is_retryable_error(RuntimeError(_FAL_MODERATION))


class _FakeAPIError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


# The real OpenAI 429 recorded from a failed styling batch (5 images/min).
_OPENAI_RATE_LIMIT = _FakeAPIError(
    "Error code: 429 - {'error': {'message': 'Rate limit reached for "
    "gpt-image-2 (for limit gpt-image) ... on input-images per min: Limit 5, "
    "Used 5, Requested 1. Please try again in 12s.', "
    "'code': 'rate_limit_exceeded'}}",
    429,
)


class TestWithRetriesRateLimits:
    """429s get a patient budget: a parallel batch against a per-minute quota
    legitimately waits minutes, which max_retries' backoff can't cover (a real
    order lost images to exhausted retries while merely rate-limited)."""

    @pytest.fixture
    def sleeps(self, monkeypatch):
        recorded: list[float] = []
        monkeypatch.setattr("ai_video_maker.retry.time.sleep", recorded.append)
        return recorded

    def test_rate_limit_classification(self):
        assert is_rate_limit_error(_OPENAI_RATE_LIMIT)
        assert is_retryable_error(_OPENAI_RATE_LIMIT)
        assert not is_rate_limit_error(RuntimeError("500 internal server error"))

    def test_429_survives_beyond_max_retries(self, sleeps):
        calls = []

        def func():
            calls.append(1)
            if len(calls) < 8:  # more failures than max_retries allows
                raise _OPENAI_RATE_LIMIT
            return "ok"

        result = with_retries(
            func, max_retries=3, base_delay=2.0, description="t"
        )
        assert result == "ok" and len(calls) == 8

    def test_429_waits_at_least_the_server_suggested_time(self, sleeps):
        calls = []

        def func():
            calls.append(1)
            if len(calls) == 1:
                raise _OPENAI_RATE_LIMIT  # says "try again in 12s"
            return "ok"

        with_retries(func, max_retries=3, base_delay=2.0, description="t")
        assert sleeps[0] >= 13.0  # suggested 12s + 1, not the 2s backoff

    def test_429_budget_is_finite(self, sleeps):
        def func():
            raise _OPENAI_RATE_LIMIT

        with pytest.raises(_FakeAPIError):
            with_retries(func, max_retries=3, base_delay=0.0, description="t")
        assert len(sleeps) < 20  # gives up eventually, doesn't spin forever

    def test_insufficient_quota_fails_fast_despite_429(self, sleeps):
        # Out-of-credits comes back as HTTP 429 like a rate limit, but waiting
        # never fixes it — a real order burned ~6 min per planning call on it.
        quota = _FakeAPIError(
            "Error code: 429 - {'error': {'message': 'You exceeded your "
            "current quota, please check your plan and billing details.', "
            "'code': 'insufficient_quota'}}",
            429,
        )
        assert is_quota_exhausted_error(quota)
        assert not is_rate_limit_error(quota)
        assert not is_retryable_error(quota)
        calls = []

        def func():
            calls.append(1)
            raise quota

        with pytest.raises(_FakeAPIError):
            with_retries(func, max_retries=5, base_delay=0.0, description="t")
        assert len(calls) == 1 and sleeps == []

    def test_permanent_400_still_fails_fast(self, sleeps):
        invalid = _FakeAPIError(
            "Error code: 400 - {'error': {'code': 'invalid_image_file'}}", 400
        )
        calls = []

        def func():
            calls.append(1)
            raise invalid

        with pytest.raises(_FakeAPIError):
            with_retries(func, max_retries=5, base_delay=0.0, description="t")
        assert len(calls) == 1 and sleeps == []


class TestWithRewordRecovery:
    def test_success_first_try_never_rewords(self):
        rewords = []
        result = with_reword_recovery(
            lambda p: f"ok:{p}", "prompt",
            reword=lambda p: rewords.append(p) or "reworded",
            attempts=2, description="t",
        )
        assert result == "ok:prompt" and rewords == []

    def test_moderation_error_rewords_and_retries(self):
        calls = []

        def run(prompt):
            calls.append(prompt)
            if prompt == "flagged":
                raise RuntimeError(_FAL_MODERATION)
            return f"ok:{prompt}"

        result = with_reword_recovery(
            run, "flagged", reword=lambda p: "safe version",
            attempts=2, description="t",
        )
        assert result == "ok:safe version"
        assert calls == ["flagged", "safe version"]

    def test_non_moderation_error_propagates_untouched(self):
        def run(prompt):
            raise RuntimeError("500 internal server error")

        with pytest.raises(RuntimeError, match="500"):
            with_reword_recovery(
                run, "p", reword=lambda p: pytest.fail("must not reword"),
                attempts=3, description="t",
            )

    def test_exhausted_attempts_raise_last_moderation_error(self):
        rewords = []

        def run(prompt):
            raise RuntimeError(_FAL_MODERATION)

        with pytest.raises(RuntimeError, match="content_policy_violation"):
            with_reword_recovery(
                run, "p",
                reword=lambda p: rewords.append(p) or f"{p}+",
                attempts=2, description="t",
            )
        assert len(rewords) == 2  # exactly `attempts` rewords, then give up

    def test_output_stage_moderation_stops_rewording_immediately(self):
        """The picture was refused, not the words.

        A real order spent ~7 minutes per photo learning this the slow way:
        frames 18 and 19 were refused on all 6 attempts across two runs, each
        carrying a differently reworded prompt. Rewording re-rolls the same
        source image against the same classifier.
        """
        calls, rewords = [], []

        def run(prompt):
            calls.append(prompt)
            raise RuntimeError(_OUTPUT_MODERATION)

        with pytest.raises(RuntimeError, match="moderation_blocked"):
            with_reword_recovery(
                run, "p", reword=lambda p: rewords.append(p) or f"{p}+",
                attempts=3, description="t",
            )
        assert calls == ["p"]  # one attempt, not four
        assert rewords == []

    def test_output_stage_skips_the_generic_fallback_too(self):
        # The fallback is another prompt, so it changes nothing being judged.
        calls = []

        def run(prompt):
            calls.append(prompt)
            raise RuntimeError(_OUTPUT_MODERATION)

        with pytest.raises(RuntimeError):
            with_reword_recovery(
                run, "p", reword=lambda p: "x", attempts=2, description="t",
                last_resort="generic fallback",
            )
        assert calls == ["p"]

    def test_prompt_stage_moderation_still_rewords(self):
        # fal/Kling rejects on the WORDS, where rewording genuinely works —
        # the early exit must not reach it.
        calls = []

        def run(prompt):
            calls.append(prompt)
            if prompt == "flagged":
                raise RuntimeError(_FAL_MODERATION)
            return "ok"

        assert with_reword_recovery(
            run, "flagged", reword=lambda p: "safe", attempts=2, description="t",
        ) == "ok"
        assert calls == ["flagged", "safe"]

    def test_last_resort_tried_after_rewords_exhausted(self):
        calls = []

        def run(prompt):
            calls.append(prompt)
            if prompt != "generic fallback":
                raise RuntimeError(_FAL_MODERATION)
            return f"ok:{prompt}"

        result = with_reword_recovery(
            run, "flagged",
            reword=lambda p: f"{p}+",
            attempts=2, description="t",
            last_resort="generic fallback",
        )
        assert result == "ok:generic fallback"
        assert calls == ["flagged", "flagged+", "flagged++", "generic fallback"]

    def test_last_resort_also_blocked_raises(self):
        calls = []

        def run(prompt):
            calls.append(prompt)
            raise RuntimeError(_FAL_MODERATION)

        with pytest.raises(RuntimeError, match="content_policy_violation"):
            with_reword_recovery(
                run, "p",
                reword=lambda p: f"{p}+",
                attempts=1, description="t",
                last_resort="generic fallback",
            )
        assert calls == ["p", "p+", "generic fallback"]

    def test_last_resort_skipped_when_it_equals_last_attempt(self):
        calls = []

        def run(prompt):
            calls.append(prompt)
            raise RuntimeError(_FAL_MODERATION)

        with pytest.raises(RuntimeError, match="content_policy_violation"):
            with_reword_recovery(
                run, "p",
                reword=lambda p: "generic fallback",
                attempts=1, description="t",
                last_resort="generic fallback",
            )
        assert calls == ["p", "generic fallback"]  # not resubmitted verbatim


# The verbatim rejection fal returned on project Commercial2 at 19:46:19,
# one second before two sibling clips were accepted and rendered fine.
_FAL_BALANCE = (
    "User is locked. Reason: Exhausted balance. "
    "Top up your balance at fal.ai/dashboard/billing."
)


class _Status(Exception):
    def __init__(self, code: int, msg: str = "boom") -> None:
        super().__init__(msg)
        self.status_code = code


class TestBalanceErrors:
    """fal reports "exhausted balance" transiently as well as for real.

    A real batch lost its first clip to this while the next two, submitted a
    second later, went through — it was classified permanent (403) and got
    zero retries. Bounded retries recover the blip without reintroducing the
    minutes-long burn on a genuinely empty account.
    """

    def test_the_real_fal_message_is_recognised(self):
        assert retry.is_balance_error(RuntimeError(_FAL_BALANCE))

    def test_payment_required_status_is_recognised(self):
        assert retry.is_balance_error(_Status(402))

    def test_unrelated_errors_are_not_balance_errors(self):
        for exc in (RuntimeError("connection reset"), _Status(500),
                    RuntimeError("content_policy_violation")):
            assert not retry.is_balance_error(exc)

    def test_balance_errors_are_retryable_despite_being_4xx(self):
        # The old classifier saw 403 -> permanent -> no retry at all.
        exc = _Status(403, _FAL_BALANCE)
        assert retry.is_retryable_error(exc)

    def test_openai_hard_quota_stays_permanent(self):
        # Different beast: waiting never fixes insufficient_quota.
        exc = RuntimeError("Error code: 429 - insufficient_quota")
        assert not retry.is_retryable_error(exc)

    def test_a_transient_lock_recovers_on_the_next_attempt(self, monkeypatch):
        monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) == 1:
                raise _Status(403, _FAL_BALANCE)
            return "clip"

        out = retry.with_retries(
            flaky, max_retries=5, base_delay=0.01, description="fal submit")
        assert out == "clip" and len(calls) == 2

    def test_a_real_empty_balance_gives_up_bounded(self, monkeypatch):
        slept = []
        monkeypatch.setattr(retry.time, "sleep", slept.append)
        calls = []

        def always_broke():
            calls.append(1)
            raise _Status(403, _FAL_BALANCE)

        with pytest.raises(_Status):
            retry.with_retries(always_broke, max_retries=5, base_delay=0.01,
                               description="fal submit")
        assert len(calls) == retry._BALANCE_MAX_RETRIES
        # Bounded and short — not the patient rate-limit budget.
        assert sum(slept) <= retry._BALANCE_MAX_DELAY * retry._BALANCE_MAX_RETRIES

    def test_balance_retries_do_not_consume_the_normal_budget(self, monkeypatch):
        # A balance blip followed by real transient errors must still get the
        # full ordinary retry allowance.
        monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
        calls = []

        def mixed():
            calls.append(1)
            if len(calls) == 1:
                raise _Status(403, _FAL_BALANCE)
            if len(calls) < 4:
                raise RuntimeError("network blip")
            return "ok"

        assert retry.with_retries(mixed, max_retries=4, base_delay=0.01,
                                  description="fal submit") == "ok"
