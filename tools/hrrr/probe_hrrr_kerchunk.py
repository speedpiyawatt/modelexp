#!/usr/bin/env python3
"""Evaluate Kerchunk reference generation for HRRR overnight GRIB tasks.

This Phase 4 helper does not replace the production GRIB path. It measures
whether Kerchunk can cheaply build reusable references for the same retained
HRRR files selected by the KLGA overnight task planner.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fetch_hrrr_records import (
    build_remote_paths,
    fetch_content_length,
    fetch_text,
    merged_byte_ranges,
    parse_idx,
    parse_idx_without_content_length,
    selected_ranges_require_content_length,
    wanted_records_by_patterns,
)
from tools.hrrr import build_hrrr_klga_feature_shards as hrrr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2023-02-04", help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2023-02-04", help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument(
        "--selection-mode",
        choices=hrrr.SELECTION_MODES,
        default="overnight_0005",
        help="Retained-cycle policy to evaluate.",
    )
    parser.add_argument(
        "--summary-profile",
        choices=hrrr.SUMMARY_PROFILES,
        default="overnight",
        help="Use overnight to evaluate narrow revision-cycle field plans.",
    )
    parser.add_argument("--source", choices=("aws", "google", "nomads"), default="aws")
    parser.add_argument("--max-tasks", type=int, default=1, help="Maximum HRRR task files to scan.")
    parser.add_argument(
        "--skip-messages",
        type=int,
        default=0,
        help="Passed to kerchunk.grib2.scan_grib. Use a small value for a quick smoke probe; 0 scans the whole file.",
    )
    parser.add_argument(
        "--range-merge-gap-bytes",
        type=int,
        default=hrrr.DEFAULT_RANGE_MERGE_GAP_BYTES,
        help="Byte gap used when estimating selected-record range download spans.",
    )
    parser.add_argument("--output-json", type=Path, help="Optional path for the full probe report.")
    parser.add_argument(
        "--reference-output-dir",
        type=Path,
        help="Optional directory for per-task Kerchunk reference JSON files.",
    )
    return parser.parse_args()


def iter_dates(start: str, end: str) -> Iterable[pd.Timestamp]:
    current = pd.Timestamp(start).normalize()
    stop = pd.Timestamp(end).normalize()
    while current <= stop:
        yield current
        current += pd.Timedelta(days=1)


def build_tasks(args: argparse.Namespace) -> list[hrrr.TaskSpec]:
    original_options = hrrr.RUNTIME_OPTIONS
    try:
        hrrr.RUNTIME_OPTIONS = hrrr.HrrrRuntimeOptions(summary_profile=args.summary_profile)
        tasks: list[hrrr.TaskSpec] = []
        for target_date in iter_dates(args.start_date, args.end_date):
            tasks.extend(hrrr.build_tasks_for_target_date(target_date, selection_mode=args.selection_mode))
        return tasks[: max(0, int(args.max_tasks))]
    finally:
        hrrr.RUNTIME_OPTIONS = original_options


def task_inventory_patterns(task: hrrr.TaskSpec, summary_profile: str) -> list[str]:
    original_options = hrrr.RUNTIME_OPTIONS
    try:
        hrrr.RUNTIME_OPTIONS = hrrr.HrrrRuntimeOptions(summary_profile=summary_profile)
        return [pattern for _, pattern in hrrr.inventory_selection_patterns_for_task(task)]
    finally:
        hrrr.RUNTIME_OPTIONS = original_options


def selected_inventory_summary(
    task: hrrr.TaskSpec,
    *,
    source: str,
    summary_profile: str,
    range_merge_gap_bytes: int,
) -> dict[str, object]:
    grib_url, idx_url = build_remote_paths(
        task.run_date_utc.replace("-", ""),
        task.cycle_hour_utc,
        "surface",
        task.forecast_hour,
        source,
    )

    idx_started = time.perf_counter()
    idx_text = fetch_text(idx_url)
    idx_seconds = time.perf_counter() - idx_started

    records = parse_idx_without_content_length(idx_text)
    patterns = task_inventory_patterns(task, summary_profile)
    selected = wanted_records_by_patterns(records, patterns)
    head_seconds = 0.0
    content_length: int | None = None
    if selected_ranges_require_content_length(records, selected):
        head_started = time.perf_counter()
        content_length = fetch_content_length(grib_url)
        head_seconds = time.perf_counter() - head_started
        records = parse_idx(idx_text, content_length)
        selected = wanted_records_by_patterns(records, patterns)

    spans = merged_byte_ranges(selected, max_gap_bytes=range_merge_gap_bytes)
    selected_bytes = sum(span.byte_length for span in spans)
    return {
        "grib_url": grib_url,
        "idx_url": idx_url,
        "idx_record_count": len(records),
        "selected_record_count": len(selected),
        "merged_range_count": len(spans),
        "selected_range_bytes": selected_bytes,
        "content_length": content_length,
        "timing_idx_fetch_parse_seconds": idx_seconds,
        "timing_head_seconds": head_seconds,
    }


def scan_kerchunk(grib_url: str, *, skip_messages: int) -> dict[str, object]:
    try:
        from kerchunk.grib2 import scan_grib
    except Exception as exc:  # pragma: no cover - exercised only in missing optional envs.
        raise RuntimeError(
            "Missing Kerchunk prototype dependency. Install with "
            "`uv pip install --python .venv/bin/python kerchunk zarr fsspec s3fs ujson`."
        ) from exc

    started = time.perf_counter()
    references = scan_grib(grib_url, storage_options={}, skip=skip_messages)
    scan_seconds = time.perf_counter() - started
    reference_json = json.dumps(references, sort_keys=True, separators=(",", ":"))
    data_variables = sorted(
        {
            key.split("/", 1)[0]
            for reference in references
            for key in reference.get("refs", {})
            if key.endswith("/0.0") and "/" in key and not key.startswith(("latitude/", "longitude/"))
        }
    )
    return {
        "message_reference_count": len(references),
        "reference_json_bytes": len(reference_json.encode("utf-8")),
        "data_variables_sample": data_variables[:50],
        "timing_kerchunk_scan_seconds": scan_seconds,
        "references": references,
    }


def reference_output_path(output_dir: Path, task: hrrr.TaskSpec) -> Path:
    return output_dir / f"{task.key}.kerchunk.json"


def main() -> int:
    args = parse_args()
    tasks = build_tasks(args)
    records: list[dict[str, object]] = []

    for task in tasks:
        inventory = selected_inventory_summary(
            task,
            source=args.source,
            summary_profile=args.summary_profile,
            range_merge_gap_bytes=args.range_merge_gap_bytes,
        )
        scanned = scan_kerchunk(str(inventory["grib_url"]), skip_messages=args.skip_messages)
        references = scanned.pop("references")
        record = {
            "target_date_local": task.target_date_local,
            "task_key": task.key,
            "run_date_utc": task.run_date_utc,
            "cycle_hour_utc": task.cycle_hour_utc,
            "forecast_hour": task.forecast_hour,
            "anchor_cycle_candidate": task.anchor_cycle_candidate,
            "field_profile": "full" if task.anchor_cycle_candidate or args.summary_profile == "full" else "revision",
            **inventory,
            **scanned,
        }
        if args.reference_output_dir is not None:
            args.reference_output_dir.mkdir(parents=True, exist_ok=True)
            path = reference_output_path(args.reference_output_dir, task)
            path.write_text(json.dumps(references, indent=2, sort_keys=True) + "\n")
            record["reference_output_path"] = str(path)
        records.append(record)

    scan_seconds = [float(record["timing_kerchunk_scan_seconds"]) for record in records]
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": args.source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "selection_mode": args.selection_mode,
        "summary_profile": args.summary_profile,
        "skip_messages": args.skip_messages,
        "task_count": len(records),
        "scan_seconds_total": sum(scan_seconds),
        "scan_seconds_median": statistics.median(scan_seconds) if scan_seconds else None,
        "records": records,
    }
    printable = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(printable, indent=2, sort_keys=True))
    for record in records:
        print(
            "{task_key}: selected_records={selected_record_count} merged_ranges={merged_range_count} "
            "kerchunk_messages={message_reference_count} scan_seconds={timing_kerchunk_scan_seconds:.3f}".format(
                **record
            )
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
