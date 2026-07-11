"""Retry helper with exponential backoff and retryability classification."""
from __future__ import annotations

import time
from typing import Callable, Optional, TypeVar

from .logging_setup import logger

T = TypeVar("T")


def _http_status(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of an HTTP status code from an exception."""
    code = getattr(exc, "status_code", None)
    if code is None:
        resp = getattr(exc, "response", None)
        code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def is_moderation_error(exc: BaseException) -> bool:
    """True when an error is a content-moderation / safety rejection.

    Covers OpenAI's image/text safety filter and fal's content checker (Kling
    clip generation). These are 4xxs that won't succeed if simply retried with
    the same prompt, but *can* succeed once the prompt is reworded (see
    with_reword_recovery), so callers treat them specially rather than as
    plain permanent failures.
    """
    text = str(exc).lower()
    return (
        # OpenAI
        "moderation_blocked" in text
        or "safety system" in text
        or "safety_violations" in text
        # fal / Kling
        or "content_policy_violation" in text
        or "content checker" in text
        or "content policy" in text
    )


def is_retryable_error(exc: BaseException) -> bool:
    """
    Decide whether an error is worth retrying.

    Permanent client errors (4xx other than 429 rate-limits) will never succeed
    on retry — e.g. OpenAI's 400 `moderation_blocked`, or an invalid request —
    so fail fast. Rate limits (429), server errors (5xx) and network/unknown
    errors are retried. A moderation rejection is reported as non-retryable here
    so the plain backoff loop stops immediately; prompt-rewording recovery is
    handled one level up in the OpenAI client.
    """
    if is_moderation_error(exc):
        return False
    code = _http_status(exc)
    if code is None:
        return True  # network/timeout/unknown -> retry
    if code == 429:
        return True  # rate limited -> retry
    if 400 <= code < 500:
        return False  # other client error -> permanent, don't retry
    return True  # 5xx -> retry


def with_reword_recovery(
    run: Callable[[str], T],
    prompt: str,
    *,
    reword: Callable[[str], str],
    attempts: int,
    description: str,
    last_resort: Optional[str] = None,
) -> T:
    """Run ``run(prompt)``; when a content filter rejects it, reword and retry.

    ``run`` should do its own transient-error backoff (``with_retries``) — a
    moderation rejection is classified non-retryable there, so it surfaces
    here where the prompt can be *changed* instead of resubmitted verbatim.
    Non-moderation errors propagate untouched.

    When ``attempts`` rewords are exhausted and ``last_resort`` is given, one
    final try is made with that fixed prompt — a deliberately bland, generic
    text that carries no flagged wording at all — before the last moderation
    error is raised. This trades prompt fidelity for actually getting output.
    """
    attempt_prompt = prompt
    last_exc: Optional[BaseException] = None
    for round_ in range(attempts + 1):
        try:
            result = run(attempt_prompt)
            if round_:
                logger.info(
                    "%s succeeded with a reworded prompt: %r — the original "
                    "prompt is unchanged in your storyboard/config.",
                    description, attempt_prompt,
                )
            return result
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_moderation_error(exc):
                raise
            last_exc = exc
            if round_ >= attempts:
                break
            logger.warning(
                "%s was blocked by the content filter (reword %d/%d): %s — "
                "rewording the prompt and retrying",
                description, round_ + 1, attempts, exc,
            )
            attempt_prompt = reword(attempt_prompt)
    assert last_exc is not None
    if last_resort and last_resort != attempt_prompt:
        logger.warning(
            "%s still blocked after %d rewording attempts — trying one last "
            "time with a generic fallback prompt",
            description, attempts,
        )
        try:
            result = run(last_resort)
            logger.info(
                "%s succeeded with the generic fallback prompt — the clip "
                "follows the start/end frames but ignores the planned motion "
                "wording.",
                description,
            )
            return result
        except Exception as exc:  # noqa: BLE001 - classified below
            if not is_moderation_error(exc):
                raise
            last_exc = exc
    logger.error(
        "%s still blocked after %d rewording attempts; giving up",
        description, attempts,
    )
    raise last_exc


def with_retries(
    func: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    description: str,
    retryable: Callable[[BaseException], bool] = is_retryable_error,
) -> T:
    """Call `func` with exponential backoff. Re-raises the last error.

    Stops immediately (no further attempts) on permanent errors as judged by
    `retryable`, so e.g. a content-moderation rejection isn't retried 5 times.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - we want to retry broadly
            last_exc = exc
            if not retryable(exc):
                logger.warning(
                    "%s failed with a permanent error (no retry): %s",
                    description,
                    exc,
                )
                break
            if attempt >= max_retries:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "%s failed (attempt %d/%d): %s — retrying in %.1fs",
                description,
                attempt,
                max_retries,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc
