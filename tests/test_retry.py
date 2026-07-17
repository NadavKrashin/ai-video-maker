"""retry.py: moderation classification and reword-recovery loop."""
from __future__ import annotations

import pytest

from ai_video_maker.retry import (
    is_moderation_error,
    is_quota_exhausted_error,
    is_rate_limit_error,
    is_retryable_error,
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
