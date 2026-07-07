"""retry.py: moderation classification and reword-recovery loop."""
from __future__ import annotations

import pytest

from ai_video_maker.retry import (
    is_moderation_error,
    is_retryable_error,
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
