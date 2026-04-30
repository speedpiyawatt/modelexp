#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
from typing import Any


DEFAULT_OUTPUT_DIR = pathlib.Path("data/runtime/server_dual_backtest")
RUNNER_VERSION = "server_dual_backtest_v3"


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run production server dual inference over a date range and retain full prediction metadata."
    )
    parser.add_argument("date", nargs="?", type=parse_date, help="Optional single YYYY-MM-DD date.")
    parser.add_argument("--start-date", type=parse_date, default=None)
    parser.add_argument("--end-date", type=parse_date, default=None)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--day-workers", type=int, default=1, help="Number of target dates to run concurrently.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip dates that already completed successfully.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed date instead of stopping.")
    parser.add_argument(
        "--event-bins-mode",
        choices=("polymarket", "standard"),
        default="polymarket",
        help="Use real Polymarket event bins by default; use standard for synthetic ladder backtests.",
    )
    return parser.parse_args()


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise SystemExit(f"end date {end.isoformat()} is before start date {start.isoformat()}")
    days = []
    current = start
    while current <= end:
        days.append(current)
        current += dt.timedelta(days=1)
    return days


def extract_result(stdout: str) -> dict[str, Any] | None:
    marker = "remote_run_root:"
    date_marker = "date:"
    remote_run_root: str | None = None
    target_date: str | None = None
    for line in stdout.splitlines():
        if line.startswith(date_marker):
            target_date = line.split(":", 1)[1].strip()
        if line.startswith(marker):
            remote_run_root = line.split(":", 1)[1].strip()
    if target_date is None or remote_run_root is None:
        return None
    return {"target_date_local": target_date, "remote_run_root": remote_run_root}


def existing_summary_matches(summary: dict[str, Any], *, event_bins_mode: str) -> bool:
    return (
        summary.get("status") == "ok"
        and summary.get("event_bins_mode") == event_bins_mode
        and summary.get("runner_version") == RUNNER_VERSION
    )


def write_summary(path: pathlib.Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def append_manifest(path: pathlib.Path, summary: dict[str, Any]) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")


def failed_summary(target_date: dt.date, *, output_dir: pathlib.Path, message: str) -> dict[str, Any]:
    now = dt.datetime.now(dt.timezone.utc)
    return {
        "target_date_local": target_date.isoformat(),
        "started_at_utc": now.isoformat(),
        "ended_at_utc": now.isoformat(),
        "elapsed_seconds": 0.0,
        "returncode": 1,
        "status": "failed",
        "message": message,
        "runner_version": RUNNER_VERSION,
        "local_run_root": str(output_dir / target_date.isoformat()),
    }


def run_one(target_date: dt.date, *, output_dir: pathlib.Path, event_bins_mode: str) -> dict[str, Any]:
    date_text = target_date.isoformat()
    date_dir = output_dir / date_text
    date_dir.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc)
    command = [sys.executable, "tools/weather/run_server_dual_inference.py", date_text]
    env = {**os.environ, "MODELEXP_COLOR": "never"}
    env.setdefault("MODELEXP_LAMP_SOURCE", "auto")
    env["MODELEXP_EVENT_BINS_MODE"] = event_bins_mode
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    ended = dt.datetime.now(dt.timezone.utc)
    (date_dir / "stdout.txt").write_text(completed.stdout)
    (date_dir / "stderr.txt").write_text(completed.stderr)
    summary = {
        "target_date_local": date_text,
        "command": command,
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "returncode": completed.returncode,
        "status": "ok" if completed.returncode == 0 else "failed",
        "event_bins_mode": event_bins_mode,
        "runner_version": RUNNER_VERSION,
    }
    result = extract_result(completed.stdout)
    if result is not None:
        summary.update(result)
    write_summary(date_dir / "summary.json", summary)
    return summary


def run_indexed(index: int, total: int, day: dt.date, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[{index}/{total}] {day.isoformat()} start", flush=True)
    try:
        summary = run_one(day, output_dir=args.output_dir, event_bins_mode=args.event_bins_mode)
    except Exception as exc:  # noqa: BLE001 - keep date-level failure accounting intact.
        summary = failed_summary(day, output_dir=args.output_dir, message=str(exc))
        write_summary(args.output_dir / day.isoformat() / "summary.json", summary)
    status = summary["status"]
    elapsed = float(summary["elapsed_seconds"])
    remote_root = summary.get("remote_run_root", "n/a")
    print(f"[{index}/{total}] {day.isoformat()} {status} elapsed={elapsed:.1f}s remote_run_root={remote_root}", flush=True)
    return summary


def main() -> int:
    args = parse_args()
    if args.date is not None:
        start = end = args.date
    else:
        if args.start_date is None or args.end_date is None:
            print("provide either a single date or both --start-date and --end-date", file=sys.stderr)
            return 2
        start = args.start_date
        end = args.end_date
    days = date_range(start, end)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_rows_by_date: dict[str, dict[str, Any]] = {}
    manifest_path = args.output_dir / "manifest.jsonl"
    if not args.skip_existing and manifest_path.exists():
        manifest_path.unlink()
    print(f"running {len(days)} dates: {days[0].isoformat()}..{days[-1].isoformat()}")
    print(f"local_output_dir={args.output_dir}")
    print(f"day_workers={max(1, int(args.day_workers))}")
    pending: list[tuple[int, dt.date]] = []
    for index, day in enumerate(days, start=1):
        date_dir = args.output_dir / day.isoformat()
        existing_summary = date_dir / "summary.json"
        if args.skip_existing and existing_summary.exists():
            summary = json.loads(existing_summary.read_text())
            if existing_summary_matches(summary, event_bins_mode=args.event_bins_mode):
                summary = {**summary, "skipped_existing": True}
                run_rows_by_date[day.isoformat()] = summary
                append_manifest(manifest_path, summary)
                print(f"[{index}/{len(days)}] {day.isoformat()} skip existing status=ok event_bins_mode={args.event_bins_mode}")
                continue
        pending.append((index, day))
    max_workers = max(1, int(args.day_workers))
    stopped_after_failure = False
    stop_returncode = 0
    next_pending = 0
    running: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, dt.date]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_pending < len(pending) and len(running) < max_workers:
            index, day = pending[next_pending]
            running[executor.submit(run_indexed, index, len(days), day, args)] = (index, day)
            next_pending += 1
        while running:
            done, _not_done = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                _index, day = running.pop(future)
                try:
                    summary = future.result()
                except Exception as exc:  # noqa: BLE001 - defensive; run_indexed should already catch.
                    summary = failed_summary(day, output_dir=args.output_dir, message=str(exc))
                    write_summary(args.output_dir / day.isoformat() / "summary.json", summary)
                run_rows_by_date[day.isoformat()] = summary
                append_manifest(manifest_path, summary)
                if summary.get("status") != "ok" and not args.keep_going:
                    stopped_after_failure = True
                    stop_returncode = int(summary.get("returncode") or 1)
            if not stopped_after_failure:
                while next_pending < len(pending) and len(running) < max_workers:
                    index, day = pending[next_pending]
                    running[executor.submit(run_indexed, index, len(days), day, args)] = (index, day)
                    next_pending += 1
    run_rows = [run_rows_by_date[day.isoformat()] for day in days if day.isoformat() in run_rows_by_date]
    summary_all = {
        "status": "ok" if not stopped_after_failure and all(row.get("status") == "ok" for row in run_rows) else "failed",
        "start_date": days[0].isoformat(),
        "end_date": days[-1].isoformat(),
        "date_count": len(days),
        "completed_count": len(run_rows),
        "ok_count": sum(1 for row in run_rows if row.get("status") == "ok"),
        "failed_count": sum(1 for row in run_rows if row.get("status") != "ok"),
        "event_bins_mode": args.event_bins_mode,
        "runner_version": RUNNER_VERSION,
        "stopped_after_failure": stopped_after_failure,
        "runs": run_rows,
    }
    write_summary(args.output_dir / "summary.json", summary_all)
    print(f"done status={summary_all['status']} ok={summary_all['ok_count']} failed={summary_all['failed_count']}")
    print(f"summary_path={args.output_dir / 'summary.json'}")
    if stopped_after_failure:
        print("stopped after failed date; rerun with --keep-going to continue", file=sys.stderr)
        return stop_returncode or 1
    return 0 if summary_all["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
