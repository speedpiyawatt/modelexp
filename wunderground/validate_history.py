#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable
from zoneinfo import ZoneInfo


PREFIX = "KLGA_9_US_"
TIMEZONE = ZoneInfo("America/New_York")
DEFAULT_HISTORY_DIR = Path(__file__).resolve().parent / "output" / "history"


@dataclass(frozen=True)
class DaySummary:
    day: date
    path: Path
    status_code: int | None
    count: int
    first_local: datetime | None
    last_local: datetime | None
    unique_deltas: tuple[int, ...]
    is_sorted: bool
    all_same_day: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate downloaded Weather.com historical observation files."
    )
    parser.add_argument(
        "--history-dir",
        default=DEFAULT_HISTORY_DIR,
        help="Directory containing per-day JSON history files.",
    )
    parser.add_argument(
        "--recent-days",
        type=int,
        default=2,
        help="Treat the last N dates in the dataset as potentially incomplete recent days.",
    )
    parser.add_argument(
        "--low-count-threshold",
        type=int,
        default=20,
        help="Flag non-DST, non-recent days below this count.",
    )
    parser.add_argument(
        "--high-count-threshold",
        type=int,
        default=48,
        help="Flag days above this count as unusually dense.",
    )
    return parser.parse_args()


def iter_files(history_dir: Path) -> Iterable[Path]:
    yield from sorted(history_dir.glob(f"{PREFIX}*.json"))


def parse_day_from_name(path: Path) -> date:
    return datetime.strptime(path.stem[len(PREFIX) :], "%Y-%m-%d").date()


def load_summary(path: Path) -> DaySummary:
    payload = json.loads(path.read_text())
    obs = payload.get("observations", [])
    times = [
        datetime.fromtimestamp(item["valid_time_gmt"], tz=timezone.utc).astimezone(TIMEZONE)
        for item in obs
        if "valid_time_gmt" in item
    ]
    deltas = [
        int((current - previous).total_seconds() / 60)
        for previous, current in zip(times, times[1:])
    ]
    day = parse_day_from_name(path)
    return DaySummary(
        day=day,
        path=path,
        status_code=payload.get("metadata", {}).get("status_code"),
        count=len(obs),
        first_local=times[0] if times else None,
        last_local=times[-1] if times else None,
        unique_deltas=tuple(sorted(set(deltas))),
        is_sorted=all(current >= previous for previous, current in zip(times, times[1:])),
        all_same_day=all(ts.date() == day for ts in times),
    )


def find_missing_days(days: list[date]) -> list[date]:
    if not days:
        return []
    missing: list[date] = []
    current = min(days)
    end = max(days)
    present = set(days)
    while current <= end:
        if current not in present:
            missing.append(current)
        current += timedelta(days=1)
    return missing


def second_sunday_of_march(year: int) -> date:
    current = date(year, 3, 1)
    while current.weekday() != 6:
        current += timedelta(days=1)
    return current + timedelta(days=7)


def first_sunday_of_november(year: int) -> date:
    current = date(year, 11, 1)
    while current.weekday() != 6:
        current += timedelta(days=1)
    return current


def is_dst_transition_day(day: date) -> bool:
    return day in {second_sunday_of_march(day.year), first_sunday_of_november(day.year)}


def print_section(title: str, rows: list[str]) -> None:
    print(title)
    if not rows:
        print("  none")
        return
    for row in rows:
        print(f"  {row}")


def main() -> int:
    args = parse_args()
    history_dir = Path(args.history_dir)
    files = list(iter_files(history_dir))
    if not files:
        raise SystemExit(f"No history files found in {history_dir}")

    summaries = [load_summary(path) for path in files]
    all_days = [summary.day for summary in summaries]
    missing_days = find_missing_days(all_days)
    recent_cutoff = max(all_days) - timedelta(days=args.recent_days - 1)
    counts = [summary.count for summary in summaries]

    status_issues = [
        f"{s.day} status_code={s.status_code}" for s in summaries if s.status_code != 200
    ]
    timestamp_issues = [
        f"{s.day} sorted={s.is_sorted} all_same_day={s.all_same_day}"
        for s in summaries
        if not s.is_sorted or not s.all_same_day
    ]
    dst_days = [s for s in summaries if is_dst_transition_day(s.day)]
    dst_notes = [
        f"{s.day} count={s.count} deltas={list(s.unique_deltas)[:8]}"
        for s in dst_days
    ]
    low_count_days = [
        f"{s.day} count={s.count}"
        for s in summaries
        if s.count < args.low_count_threshold
        and not is_dst_transition_day(s.day)
        and s.day < recent_cutoff
    ]
    high_count_days = [
        f"{s.day} count={s.count} deltas={list(s.unique_deltas)[:8]}"
        for s in summaries
        if s.count > args.high_count_threshold
    ]
    recent_days = [
        f"{s.day} count={s.count}"
        for s in summaries
        if s.day >= recent_cutoff
    ]

    print("Summary")
    print(f"  files={len(files)}")
    print(f"  first_day={min(all_days)}")
    print(f"  last_day={max(all_days)}")
    print(f"  missing_days={len(missing_days)}")
    print(f"  min_count={min(counts)}")
    print(f"  median_count={median(counts)}")
    print(f"  max_count={max(counts)}")

    print_section("Missing Days", [day.isoformat() for day in missing_days[:20]])
    print_section("Status Issues", status_issues[:20])
    print_section("Timestamp Issues", timestamp_issues[:20])
    print_section("DST Transition Days", dst_notes[:20])
    print_section("Low Count Days", low_count_days[:20])
    print_section("High Count Days", high_count_days[:20])
    print_section("Recent Days", recent_days[:20])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
