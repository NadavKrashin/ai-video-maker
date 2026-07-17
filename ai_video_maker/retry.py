"""Retry helper with exponential backoff and retryability classification."""
from __future__ import annotations

import random
import re
import time
from typing import Callable, Optional, TypeVar

from .logging_setup import logger

T = TypeVar("T")

# Rate limits get their own, more patient retry budget: the OpenAI image API
# allows only 5 input-images/min while styling submits a whole order's photos
# in parallel, so the tail of the batch legitimately needs minutes of waiting
# — far beyond what max_retries' exponential backoff (~30s total) covers.
_RATE_LIMIT_MAX_RETRIES = 10
_RATE_LIMIT_MAX_DELAY = 60.0


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


def is_quota_exhausted_error(exc: BaseException) -> bool:
    """True when the account is out of credits/budget (OpenAI insufficient_quota).

    Comes back as HTTP 429 like a rate limit, but waiting never fixes it —
    only adding credit does. Must fail fast, not burn the patient rate-limit
    retry budget (a real order spent ~6 minutes per planning call retrying
    this before giving up).
    """
    return "insufficient_quota" in str(exc).lower()


def is_rate_limit_error(exc: BaseException) -> bool:
    """True when an error is a 429 / rate-limit rejection (worth waiting out)."""
    if is_quota_exhausted_error(exc):
        return False
    if _http_status(exc) == 429:
        return True
    text = str(exc).lower()
    return "rate_limit_exceeded" in text or "rate limit" in text


def _suggested_wait_seconds(exc: BaseException) -> Optional[float]:
    """The server's own 'Please try again in Xs' hint, when present."""
    match = re.search(r"try again in (\d+(?:\.\d+)?)\s*s", str(exc), re.IGNORECASE)
    return float(match.group(1)) if match else None


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
    if is_quota_exhausted_error(exc):
        return False  # out of credits: only a billing top-up fixes this
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
                "%s: even the deliberately generic fallback prompt was "
                "blocked — the content checker is almost certainly flagging "
                "the input media (an image), not the prompt text. Rewording "
                "cannot fix this; regenerate the flagged input instead.",
                description,
            )
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

    Rate limits (429) are counted against their own larger budget
    (`_RATE_LIMIT_MAX_RETRIES`) instead of `max_retries`: with per-minute
    quotas and a parallel batch, waiting several minutes is normal, not a
    failure. The wait honours the server's "try again in Xs" hint and adds
    jitter so parallel workers don't stampede the quota in lockstep.
    """
    last_exc: Optional[BaseException] = None
    attempt = 0
    rate_limit_attempt = 0
    while True:
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
            if is_rate_limit_error(exc):
                rate_limit_attempt += 1
                if rate_limit_attempt >= _RATE_LIMIT_MAX_RETRIES:
                    break
                delay = base_delay * (2 ** (rate_limit_attempt - 1))
                suggested = _suggested_wait_seconds(exc)
                if suggested is not None:
                    delay = max(delay, suggested + 1.0)
                delay = min(delay, _RATE_LIMIT_MAX_DELAY)
                delay *= 1.0 + random.random() * 0.25
                logger.warning(
                    "%s hit a rate limit (retry %d/%d) — waiting %.1fs",
                    description,
                    rate_limit_attempt,
                    _RATE_LIMIT_MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)
                continue
            attempt += 1
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
