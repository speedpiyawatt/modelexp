from __future__ import annotations

import random

from tools.weather.retry import RetryPolicy, classify_task_failure, compute_retry_delay_seconds, should_retry_attempt


def test_classify_task_failure_marks_broken_pipe_retryable():
    decision = classify_task_failure(
        exception_type="BrokenPipeError",
        message="BrokenPipeError: [Errno 32] Broken pipe",
        phase="reduce",
    )
    assert decision.error_class == "broken_pipe"
    assert decision.retry_mode == "bounded"


def test_classify_task_failure_marks_deterministic_grid_error_non_retryable():
    decision = classify_task_failure(
        exception_type=None,
        message="Unable to locate KLGA grid cell in reduced GRIB2",
        phase="extract",
    )
    assert decision.error_class == "grid_cell_missing"
    assert decision.retry_mode == "never"


def test_compute_retry_delay_seconds_caps_exponential_backoff():
    policy = RetryPolicy(max_attempts=3, backoff_seconds=2.0, max_backoff_seconds=5.0)
    rng = random.Random(0)
    delay = compute_retry_delay_seconds(attempt=4, policy=policy, rng=rng, jitter_ratio=0.0)
    assert delay == 5.0


def test_should_retry_attempt_honors_single_retry_mode():
    policy = RetryPolicy(max_attempts=3, backoff_seconds=2.0, max_backoff_seconds=30.0)
    decision = classify_task_failure(exception_type=None, message="cfgrib open failed for group x: boom", phase="open")
    assert decision.retry_mode == "single"
    assert should_retry_attempt(attempt=1, policy=policy, decision=decision) is True
    assert should_retry_attempt(attempt=2, policy=policy, decision=decision) is False
