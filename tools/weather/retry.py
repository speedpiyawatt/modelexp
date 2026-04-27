from __future__ import annotations

import random
import re
from dataclasses import dataclass


RETRYABLE_EXCEPTION_TYPES = {
    "BrokenPipeError",
    "ConnectionAbortedError",
    "ConnectionError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "ChunkedEncodingError",
    "ConnectTimeout",
    "OSError",
    "ProtocolError",
    "ReadTimeout",
    "TimeoutError",
    "URLError",
}
NON_RETRYABLE_EXCEPTION_TYPES = {
    "AssertionError",
    "FileNotFoundError",
    "ImportError",
    "KeyError",
    "ModuleNotFoundError",
    "SystemExit",
    "ValueError",
}
RETRYABLE_MESSAGE_PATTERNS = (
    (re.compile(r"broken pipe", re.IGNORECASE), "broken_pipe"),
    (re.compile(r"timed? out", re.IGNORECASE), "timeout"),
    (re.compile(r"connection reset", re.IGNORECASE), "connection_reset"),
    (re.compile(r"connection aborted", re.IGNORECASE), "connection_aborted"),
    (re.compile(r"connection refused", re.IGNORECASE), "connection_refused"),
    (re.compile(r"temporarily unavailable", re.IGNORECASE), "temporarily_unavailable"),
    (re.compile(r"service unavailable|http error 50[234]", re.IGNORECASE), "remote_service_error"),
    (re.compile(r"too many requests|http error 429", re.IGNORECASE), "rate_limited"),
)
NON_RETRYABLE_MESSAGE_PATTERNS = (
    (re.compile(r"missing_required_fields", re.IGNORECASE), "missing_required_fields"),
    (re.compile(r"no_matching_records", re.IGNORECASE), "no_matching_records"),
    (re.compile(r"unable to locate klga grid cell", re.IGNORECASE), "grid_cell_missing"),
    (re.compile(r"did not expose init/valid timestamps", re.IGNORECASE), "timestamp_missing"),
    (re.compile(r"wgrib2 was not found|requires cfgrib|requires xarray", re.IGNORECASE), "dependency_missing"),
    (re.compile(r"no feature rows could be built", re.IGNORECASE), "feature_rows_missing"),
)
AMBIGUOUS_PHASES = {"download", "open", "reduce", "crop", "extract"}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 6
    backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0


@dataclass(frozen=True)
class RetryDecision:
    error_class: str
    retry_mode: str

    @property
    def retryable(self) -> bool:
        return self.retry_mode != "never"


def compute_retry_delay_seconds(
    *,
    attempt: int,
    policy: RetryPolicy,
    jitter_ratio: float = 0.2,
    rng: random.Random | None = None,
) -> float:
    if attempt <= 1:
        return 0.0
    random_source = rng or random
    base_delay = float(policy.backoff_seconds) * (2 ** max(0, attempt - 2))
    capped_delay = min(float(policy.max_backoff_seconds), base_delay)
    jitter = random_source.uniform(max(0.0, 1.0 - jitter_ratio), 1.0 + jitter_ratio)
    return max(0.0, min(float(policy.max_backoff_seconds), capped_delay * jitter))


def classify_task_failure(
    *,
    exception_type: str | None,
    message: str,
    phase: str | None = None,
) -> RetryDecision:
    exc_name = (exception_type or "").strip()
    message_text = (message or "").strip()

    for pattern, error_class in NON_RETRYABLE_MESSAGE_PATTERNS:
        if pattern.search(message_text):
            return RetryDecision(error_class=error_class, retry_mode="never")
    for pattern, error_class in RETRYABLE_MESSAGE_PATTERNS:
        if pattern.search(message_text):
            return RetryDecision(error_class=error_class, retry_mode="bounded")

    if exc_name in NON_RETRYABLE_EXCEPTION_TYPES:
        return RetryDecision(error_class=exc_name.lower(), retry_mode="never")
    if exc_name in RETRYABLE_EXCEPTION_TYPES:
        if exc_name == "OSError" and phase not in AMBIGUOUS_PHASES and "pipe" not in message_text.lower():
            return RetryDecision(error_class="os_error", retry_mode="never")
        return RetryDecision(error_class=exc_name.lower(), retry_mode="bounded")

    lowered = message_text.lower()
    if "cfgrib open failed" in lowered or "wgrib2" in lowered:
        return RetryDecision(error_class="subprocess_or_decode_error", retry_mode="single")
    if phase in AMBIGUOUS_PHASES:
        return RetryDecision(error_class="transient_phase_error", retry_mode="single")
    return RetryDecision(error_class=exc_name.lower() or "unknown_error", retry_mode="never")


def should_retry_attempt(*, attempt: int, policy: RetryPolicy, decision: RetryDecision) -> bool:
    if attempt >= int(policy.max_attempts):
        return False
    if decision.retry_mode == "bounded":
        return True
    if decision.retry_mode == "single":
        return attempt == 1
    return False
