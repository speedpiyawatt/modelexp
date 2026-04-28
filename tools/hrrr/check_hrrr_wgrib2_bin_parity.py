#!/usr/bin/env python3
"""Check HRRR wgrib2-bin extraction parity against a reference extractor."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_hrrr_klga_feature_shards as hrrr


def parse_forecast_hours(value: str) -> set[int]:
    hours: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid forecast-hour range: {token}")
            hours.update(range(start, end + 1))
        else:
            hours.add(int(token))
    return hours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced-grib", type=Path, required=True, help="Candidate GRIB2 file to compare.")
    parser.add_argument("--reference-grib", type=Path, help="Optional separate reference GRIB2 file.")
    parser.add_argument("--target-date", required=True, help="Target local date in YYYY-MM-DD.")
    parser.add_argument("--run-date-utc", help="Optional HRRR run date in YYYY-MM-DD. Defaults to any retained run date.")
    parser.add_argument("--cycle-hour-utc", type=int, required=True, help="HRRR cycle hour UTC to compare.")
    parser.add_argument(
        "--forecast-hours",
        default="0-18",
        help="Forecast hours to compare, e.g. 0-18 or 0,3,6. Tasks absent from the target planner are skipped.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=hrrr.SELECTION_MODES,
        default=hrrr.DEFAULT_SELECTION_MODE,
        help="Task-planning selection mode used to find matching tasks.",
    )
    parser.add_argument(
        "--summary-profile",
        choices=hrrr.SUMMARY_PROFILES,
        default="full",
        help="Summary profile to apply during extraction.",
    )
    parser.add_argument(
        "--reference-method",
        choices=("cfgrib", "eccodes"),
        default="eccodes",
        help="Reference extraction backend.",
    )
    parser.add_argument(
        "--candidate-method",
        choices=("wgrib2-bin", "wgrib2-ijbox-bin"),
        default="wgrib2-bin",
        help="Candidate binary extraction backend.",
    )
    parser.add_argument("--numeric-tolerance", type=float, default=1e-4, help="Maximum allowed numeric absolute difference.")
    parser.add_argument("--output-json", type=Path, help="Optional path to write the parity report JSON.")
    parser.add_argument("--include-provenance", action="store_true", help="Also construct provenance rows during comparison.")
    parser.add_argument("--legacy-aliases", action="store_true", help="Include legacy metric aliases in extracted rows.")
    return parser.parse_args()


def matching_tasks(args: argparse.Namespace) -> list[hrrr.TaskSpec]:
    forecast_hours = parse_forecast_hours(str(args.forecast_hours))
    tasks = hrrr.build_tasks_for_target_date(pd.Timestamp(args.target_date), selection_mode=str(args.selection_mode))
    selected = [
        task
        for task in tasks
        if int(task.cycle_hour_utc) == int(args.cycle_hour_utc)
        and int(task.forecast_hour) in forecast_hours
        and (args.run_date_utc is None or task.run_date_utc == args.run_date_utc)
    ]
    selected.sort(key=lambda task: int(task.forecast_hour))
    return selected


def run_extract(
    *,
    method: str,
    reduced_path: Path,
    inventory: list[str],
    task: hrrr.TaskSpec,
    summary_profile: str,
    include_provenance: bool,
    include_legacy_aliases: bool,
) -> hrrr.TaskResult:
    hrrr.RUNTIME_OPTIONS = hrrr.HrrrRuntimeOptions(
        extract_method=method,
        summary_profile=summary_profile,
        skip_provenance=not include_provenance,
    )
    return hrrr.process_reduced_grib(
        reduced_path,
        inventory,
        task,
        hrrr.task_remote_url(task),
        include_legacy_aliases=include_legacy_aliases,
        filter_inventory_to_task_step=True,
        write_provenance=include_provenance,
    )


def compare_rows(reference: dict[str, object], candidate: dict[str, object], *, tolerance: float) -> dict[str, object]:
    numeric_compared = 0
    max_numeric_diff = 0.0
    max_numeric_column: str | None = None
    failures: list[dict[str, object]] = []
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    shared_keys = sorted(reference_keys & candidate_keys)
    for key in shared_keys:
        ref_value = reference.get(key)
        cand_value = candidate.get(key)
        if isinstance(ref_value, (int, float)) and isinstance(cand_value, (int, float)):
            ref_float = float(ref_value)
            cand_float = float(cand_value)
            if not (math.isfinite(ref_float) and math.isfinite(cand_float)):
                numeric_compared += 1
                same_nonfinite = (
                    math.isnan(ref_float)
                    and math.isnan(cand_float)
                    or ref_float == cand_float
                )
                if same_nonfinite:
                    continue
                failures.append(
                    {
                        "column": key,
                        "reference": ref_float,
                        "candidate": cand_float,
                        "abs_diff": math.inf,
                    }
                )
                max_numeric_diff = math.inf
                max_numeric_column = key
                continue
            diff = abs(ref_float - cand_float)
            numeric_compared += 1
            if diff > max_numeric_diff:
                max_numeric_diff = diff
                max_numeric_column = key
            if diff > tolerance:
                failures.append(
                    {
                        "column": key,
                        "reference": ref_float,
                        "candidate": cand_float,
                        "abs_diff": diff,
                    }
                )
    return {
        "numeric_compared": numeric_compared,
        "max_numeric_diff": max_numeric_diff,
        "max_numeric_column": max_numeric_column,
        "numeric_failures": failures,
        "reference_only_columns": sorted(reference_keys - candidate_keys),
        "candidate_only_columns": sorted(candidate_keys - reference_keys),
    }


def compare_summary_rows(
    reference_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
    *,
    target_date: str,
    tolerance: float,
) -> dict[str, object]:
    if not reference_rows or not candidate_rows:
        return {"summary_available": False, "summary_failures": ["missing extracted rows"]}
    reference_summary = hrrr.build_summary_row(target_date, reference_rows)
    candidate_summary = hrrr.build_summary_row(target_date, candidate_rows)
    comparison = compare_rows(reference_summary, candidate_summary, tolerance=tolerance)
    comparison["summary_available"] = True
    return comparison


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    reduced_path = args.reduced_grib
    if not reduced_path.exists():
        print(f"Reduced GRIB does not exist: {reduced_path}", file=sys.stderr)
        return 2
    reference_path = args.reference_grib or reduced_path
    if not reference_path.exists():
        print(f"Reference GRIB does not exist: {reference_path}", file=sys.stderr)
        return 2

    tasks = matching_tasks(args)
    if not tasks:
        print("No matching tasks found for requested target/cycle/forecast-hour selection.", file=sys.stderr)
        return 2

    wgrib2_path = hrrr.ensure_tooling()
    inventory = hrrr.inventory_for_grib(wgrib2_path, reduced_path)
    reference_inventory = inventory if reference_path == reduced_path else hrrr.inventory_for_grib(wgrib2_path, reference_path)
    task_reports: list[dict[str, object]] = []
    reference_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for task in tasks:
        reference = run_extract(
            method=str(args.reference_method),
            reduced_path=reference_path,
            inventory=reference_inventory,
            task=task,
            summary_profile=str(args.summary_profile),
            include_provenance=bool(args.include_provenance),
            include_legacy_aliases=bool(args.legacy_aliases),
        )
        candidate = run_extract(
            method=str(args.candidate_method),
            reduced_path=reduced_path,
            inventory=inventory,
            task=task,
            summary_profile=str(args.summary_profile),
            include_provenance=bool(args.include_provenance),
            include_legacy_aliases=bool(args.legacy_aliases),
        )
        report: dict[str, object] = {
            "task_key": task.key,
            "forecast_hour": int(task.forecast_hour),
            "reference_ok": bool(reference.ok),
            "candidate_ok": bool(candidate.ok),
            "reference_message": reference.message,
            "candidate_message": candidate.message,
            "reference_missing_fields": reference.missing_fields,
            "candidate_missing_fields": candidate.missing_fields,
            "candidate_diagnostics": {
                key: candidate.diagnostics.get(key)
                for key in (
                    "binary_cache_hit",
                    "timing_binary_dump_seconds",
                    "timing_binary_read_seconds",
                    "timing_binary_grid_context_seconds",
                    "binary_byte_count",
                    "binary_grid_shape",
                    "binary_inventory_message_count",
                    "binary_parsed_array_count",
                    "binary_skipped_array_count",
                    "binary_dump_method",
                    "binary_ij_box",
                    "binary_ijbox_cache_hit",
                    "binary_ijbox_cache_key",
                )
            },
        }
        if reference.ok and candidate.ok and reference.row is not None and candidate.row is not None:
            row_comparison = compare_rows(reference.row, candidate.row, tolerance=float(args.numeric_tolerance))
            report.update(row_comparison)
            report["missing_fields_match"] = reference.missing_fields == candidate.missing_fields
            reference_rows.append(reference.row)
            candidate_rows.append(candidate.row)
        task_reports.append(report)

    summary_report = compare_summary_rows(
        reference_rows,
        candidate_rows,
        target_date=str(args.target_date),
        tolerance=float(args.numeric_tolerance),
    )
    failed_tasks = [
        report
        for report in task_reports
        if not report.get("reference_ok")
        or not report.get("candidate_ok")
        or not report.get("missing_fields_match", False)
        or report.get("numeric_failures")
        or report.get("reference_only_columns")
        or report.get("candidate_only_columns")
    ]
    summary_failed = bool(summary_report.get("numeric_failures") or summary_report.get("reference_only_columns") or summary_report.get("candidate_only_columns"))
    report_payload = {
        "status": "fail" if failed_tasks or summary_failed else "pass",
        "reduced_grib": str(reduced_path),
        "reference_grib": str(reference_path),
        "target_date": str(args.target_date),
        "run_date_utc": args.run_date_utc,
        "cycle_hour_utc": int(args.cycle_hour_utc),
        "forecast_hours": sorted(parse_forecast_hours(str(args.forecast_hours))),
        "selection_mode": str(args.selection_mode),
        "summary_profile": str(args.summary_profile),
        "reference_method": str(args.reference_method),
        "candidate_method": str(args.candidate_method),
        "numeric_tolerance": float(args.numeric_tolerance),
        "task_count": len(task_reports),
        "failed_task_count": len(failed_tasks),
        "tasks": task_reports,
        "summary": summary_report,
        "elapsed_seconds": time.perf_counter() - started_at,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report_payload, indent=2, sort_keys=True))
    print(json.dumps(report_payload, indent=2, sort_keys=True))
    return 1 if report_payload["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
