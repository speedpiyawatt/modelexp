#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"
DEFAULT_LOCATION_ID = "KLGA:9:US"
DEFAULT_START_DATE = date(2025, 1, 1)
DEFAULT_DELAY_SECONDS = 1.5
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT_SECONDS = 30
BASE_URL = "https://api.weather.com/v1/location/{location_id}/observations/historical.json"


@dataclass(frozen=True)
class Config:
    api_key: str
    location_id: str
    start_date: date
    end_date: date
    output_dir: Path
    delay_seconds: float
    max_retries: int
    timeout_seconds: int
    skip_http_statuses: frozenset[int]
    force: bool


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Weather.com historical observations for every day in a date range "
            "and save one JSON file per day."
        )
    )
    parser.add_argument(
        "--location-id",
        default=DEFAULT_LOCATION_ID,
        help="Weather.com location id, e.g. KLGA:9:US",
    )
    parser.add_argument(
        "--api-key",
        default=DEFAULT_API_KEY,
        help="Weather.com API key. Defaults to the key observed in page traffic.",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE.isoformat(),
        help="Inclusive start date in YYYY-MM-DD format. Default: 2025-01-01",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date in YYYY-MM-DD format. Default: today",
    )
    parser.add_argument(
        "--output-dir",
        default="output/history",
        help="Directory where per-day JSON files will be written.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Base delay between successful requests. Default: 1.5",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Retries for 429/5xx/network failures. Default: 5",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="HTTP timeout per request. Default: 30",
    )
    parser.add_argument(
        "--skip-http-status",
        type=int,
        action="append",
        default=[],
        help="HTTP status to save as an empty no-data day instead of aborting. Can be repeated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing daily files instead of skipping them.",
    )
    args = parser.parse_args()

    start_date = parse_date(args.start_date, "--start-date")
    end_date = parse_date(args.end_date, "--end-date")
    if end_date < start_date:
        parser.error("--end-date must be on or after --start-date")
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative")
    if args.max_retries < 0:
        parser.error("--max-retries must be non-negative")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")

    return Config(
        api_key=args.api_key,
        location_id=args.location_id,
        start_date=start_date,
        end_date=end_date,
        output_dir=Path(args.output_dir),
        delay_seconds=args.delay_seconds,
        max_retries=args.max_retries,
        timeout_seconds=args.timeout_seconds,
        skip_http_statuses=frozenset(args.skip_http_status),
        force=args.force,
    )


def parse_date(value: str, flag_name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise SystemExit(f"{flag_name} must be YYYY-MM-DD, got {value!r}") from exc


def iter_dates(start_date: date, end_date: date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def build_url(config: Config, day: date) -> str:
    params = urlencode(
        {
            "apiKey": config.api_key,
            "units": "e",
            "startDate": day.strftime("%Y%m%d"),
            "endDate": day.strftime("%Y%m%d"),
        }
    )
    return BASE_URL.format(location_id=config.location_id) + "?" + params


def fetch_json(url: str, timeout_seconds: int) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "User-Agent": "wunderground-history-fetcher/1.0",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def request_with_retries(config: Config, day: date) -> dict[str, Any]:
    url = build_url(config, day)
    attempt = 0
    while True:
        try:
            return fetch_json(url, config.timeout_seconds)
        except HTTPError as exc:
            status = exc.code
            if status in config.skip_http_statuses:
                log(f"{day} got HTTP {status}, writing empty no-data payload")
                return {
                    "observations": [],
                    "fetch_error": {
                        "http_status": status,
                        "reason": str(exc.reason),
                        "url": url,
                    },
                }
            retryable = status == 429 or 500 <= status <= 599
            if retryable and attempt < config.max_retries:
                wait_seconds = backoff_delay(config.delay_seconds, attempt, exc.headers.get("Retry-After"))
                log(
                    f"{day} got HTTP {status}, retrying in {wait_seconds:.1f}s "
                    f"({attempt + 1}/{config.max_retries})"
                )
                time.sleep(wait_seconds)
                attempt += 1
                continue
            raise
        except URLError as exc:
            if attempt < config.max_retries:
                wait_seconds = backoff_delay(config.delay_seconds, attempt)
                log(
                    f"{day} got network error {exc.reason!r}, retrying in {wait_seconds:.1f}s "
                    f"({attempt + 1}/{config.max_retries})"
                )
                time.sleep(wait_seconds)
                attempt += 1
                continue
            raise


def backoff_delay(base_delay: float, attempt: int, retry_after: str | None = None) -> float:
    if retry_after:
        try:
            return max(float(retry_after), base_delay)
        except ValueError:
            pass
    exponential = base_delay * (2 ** attempt)
    jitter = random.uniform(0, max(base_delay, 0.25))
    return exponential + jitter


def output_path(config: Config, day: date) -> Path:
    filename = f"{config.location_id.replace(':', '_')}_{day.isoformat()}.json"
    return config.output_dir / filename


def write_day_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def log(message: str) -> None:
    print(message, file=sys.stderr)


def main() -> int:
    config = parse_args()
    total_days = (config.end_date - config.start_date).days + 1
    processed = 0
    skipped = 0

    for day in iter_dates(config.start_date, config.end_date):
        path = output_path(config, day)
        if path.exists() and not config.force:
            skipped += 1
            log(f"{day} skipped existing file {path}")
            continue

        payload = request_with_retries(config, day)
        write_day_file(path, payload)
        obs_count = len(payload.get("observations", []))
        processed += 1
        log(f"{day} wrote {path} with {obs_count} observations")

        if day != config.end_date:
            time.sleep(config.delay_seconds)

    log(
        f"done: requested={processed}, skipped={skipped}, total_days={total_days}, "
        f"output_dir={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
