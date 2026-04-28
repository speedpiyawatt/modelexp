#!/usr/bin/env python3
"""Benchmark HRRR binary extractor candidates on cached GRIB artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_hrrr_klga_feature_shards as hrrr
from check_hrrr_wgrib2_bin_parity import compare_rows, compare_summary_rows, parse_forecast_hours


DEFAULT_OUTPUT_ROOT = Path("tools/hrrr/data/runtime/binary_extractor_benchmarks")
BENCHMARK_METHODS = ("cfgrib", "eccodes", "wgrib2-bin", "wgrib2-ijbox-bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reduced-grib", type=Path, required=True, help="Reduced multi-forecast GRIB used by cfgrib/eccodes/wgrib2-bin.")
    parser.add_argument("--direct-grib", type=Path, help="Uncropped selected multi-forecast GRIB used by wgrib2-ijbox-bin.")
    parser.add_argument("--target-date", required=True, help="Target local date in YYYY-MM-DD.")
    parser.add_argument("--run-date-utc", help="Optional HRRR run date in YYYY-MM-DD. Defaults to any retained run date.")
    parser.add_argument("--cycle-hour-utc", type=int, required=True, help="HRRR cycle hour UTC to benchmark.")
    parser.add_argument("--forecast-hours", default="0-18", help="Forecast hours to benchmark, e.g. 0-18 or 0,3,6.")
    parser.add_argument(
        "--selection-mode",
        choices=hrrr.SELECTION_MODES,
        default=hrrr.DEFAULT_SELECTION_MODE,
        help="Task-planning selection mode used to find matching tasks.",
    )
    parser.add_argument("--summary-profile", choices=hrrr.SUMMARY_PROFILES, default="full")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BENCHMARK_METHODS,
        default=("cfgrib", "eccodes", "wgrib2-bin", "wgrib2-ijbox-bin"),
        help="Extractor methods to benchmark.",
    )
    parser.add_argument("--reference-method", choices=("cfgrib", "eccodes"), default="eccodes")
    parser.add_argument("--numeric-tolerance", type=float, default=1e-4)
    parser.add_argument("--runs", type=int, default=1, help="Measured runs per method.")
    parser.add_argument("--include-provenance", action="store_true", help="Also construct provenance rows.")
    parser.add_argument("--legacy-aliases", action="store_true", help="Include legacy metric aliases.")
    parser.add_argument(
        "--keep-binary-cache",
        action="store_true",
        help="Keep the process-local wgrib2 binary array cache across method runs.",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for benchmark JSON/CSV outputs.")
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


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


def clear_wgrib2_binary_cache() -> None:
    with hrrr.WGRIB2_BINARY_ARRAY_CACHE_LOCK:
        hrrr.WGRIB2_BINARY_ARRAY_CACHE.clear()


def ru_maxrss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def numeric(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize_diagnostics(task_reports: list[dict[str, object]]) -> dict[str, object]:
    timing_keys = (
        "timing_cfgrib_open_seconds",
        "timing_direct_extract_seconds",
        "timing_binary_ijbox_context_seconds",
        "timing_binary_dump_seconds",
        "timing_binary_grid_context_seconds",
        "timing_binary_read_seconds",
        "timing_row_build_seconds",
    )
    diagnostics = [report.get("diagnostics") for report in task_reports if isinstance(report.get("diagnostics"), dict)]
    totals = {
        key: sum(numeric(item.get(key)) for item in diagnostics)
        for key in timing_keys
    }
    binary_byte_counts = [
        int(item.get("binary_byte_count") or 0)
        for item in diagnostics
        if item.get("binary_byte_count") is not None and item.get("binary_cache_hit") is False
    ]
    return {
        **totals,
        "binary_temp_bytes_observed": sum(binary_byte_counts),
        "binary_cache_hit_count": sum(1 for item in diagnostics if item.get("binary_cache_hit") is True),
        "binary_cache_miss_count": sum(1 for item in diagnostics if item.get("binary_cache_hit") is False),
        "direct_ijbox_fallback_count": sum(1 for item in diagnostics if item.get("direct_ijbox_fallback_to_reduced") is True),
    }


def run_method(
    *,
    method: str,
    grib_path: Path,
    inventory: list[str],
    tasks: list[hrrr.TaskSpec],
    summary_profile: str,
    include_provenance: bool,
    include_legacy_aliases: bool,
) -> dict[str, object]:
    hrrr.RUNTIME_OPTIONS = hrrr.HrrrRuntimeOptions(
        extract_method=method,
        summary_profile=summary_profile,
        skip_provenance=not include_provenance,
    )
    task_reports: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_rss_mib = ru_maxrss_mib()
    for task in tasks:
        result = hrrr.process_reduced_grib_compat(
            grib_path,
            inventory,
            task,
            hrrr.task_remote_url(task),
            include_legacy_aliases=include_legacy_aliases,
            filter_inventory_to_task_step=True,
            write_provenance=include_provenance,
        )
        task_reports.append(
            {
                "task_key": task.key,
                "forecast_hour": int(task.forecast_hour),
                "ok": bool(result.ok),
                "message": result.message,
                "missing_fields": result.missing_fields,
                "diagnostics": result.diagnostics,
            }
        )
        if result.ok and result.row is not None:
            rows.append(result.row)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    ended_rss_mib = ru_maxrss_mib()
    ok = all(bool(report["ok"]) for report in task_reports)
    summary_row = hrrr.build_summary_row(tasks[0].target_date_local, rows) if rows else None
    return {
        "method": method,
        "ok": ok,
        "task_count": len(task_reports),
        "ok_task_count": sum(1 for report in task_reports if report["ok"]),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "maxrss_mib_before": started_rss_mib,
        "maxrss_mib_after": ended_rss_mib,
        "maxrss_mib_delta": max(0.0, ended_rss_mib - started_rss_mib),
        "diagnostic_totals": summarize_diagnostics(task_reports),
        "tasks": task_reports,
        "rows": rows,
        "summary_row": summary_row,
    }


def compare_to_reference(
    *,
    reference: dict[str, object],
    candidate: dict[str, object],
    target_date: str,
    tolerance: float,
) -> dict[str, object]:
    reference_rows = reference.get("rows") if isinstance(reference.get("rows"), list) else []
    candidate_rows = candidate.get("rows") if isinstance(candidate.get("rows"), list) else []
    row_reports: list[dict[str, object]] = []
    failed_row_count = 0
    max_numeric_diff = 0.0
    max_numeric_column: str | None = None
    for ref_row, cand_row in zip(reference_rows, candidate_rows):
        comparison = compare_rows(ref_row, cand_row, tolerance=tolerance)
        comparison["task_key"] = cand_row.get("task_key")
        row_reports.append(comparison)
        if comparison.get("numeric_failures") or comparison.get("reference_only_columns") or comparison.get("candidate_only_columns"):
            failed_row_count += 1
        if float(comparison.get("max_numeric_diff") or 0.0) > max_numeric_diff:
            max_numeric_diff = float(comparison["max_numeric_diff"])
            max_numeric_column = comparison.get("max_numeric_column") if isinstance(comparison.get("max_numeric_column"), str) else None
    summary = compare_summary_rows(reference_rows, candidate_rows, target_date=target_date, tolerance=tolerance)
    return {
        "row_count_compared": min(len(reference_rows), len(candidate_rows)),
        "failed_row_count": failed_row_count,
        "max_row_numeric_diff": max_numeric_diff,
        "max_row_numeric_column": max_numeric_column,
        "summary": summary,
        "rows": row_reports,
    }


def flatten_run(run: dict[str, object]) -> dict[str, object]:
    totals = run.get("diagnostic_totals") if isinstance(run.get("diagnostic_totals"), dict) else {}
    parity = run.get("parity") if isinstance(run.get("parity"), dict) else {}
    summary = parity.get("summary") if isinstance(parity.get("summary"), dict) else {}
    return {
        "method": run.get("method"),
        "run_index": run.get("run_index"),
        "ok": run.get("ok"),
        "task_count": run.get("task_count"),
        "ok_task_count": run.get("ok_task_count"),
        "wall_seconds": run.get("wall_seconds"),
        "cpu_seconds": run.get("cpu_seconds"),
        "maxrss_mib_after": run.get("maxrss_mib_after"),
        "maxrss_mib_delta": run.get("maxrss_mib_delta"),
        "binary_temp_bytes_observed": totals.get("binary_temp_bytes_observed"),
        "binary_cache_hit_count": totals.get("binary_cache_hit_count"),
        "binary_cache_miss_count": totals.get("binary_cache_miss_count"),
        "timing_cfgrib_open_seconds": totals.get("timing_cfgrib_open_seconds"),
        "timing_direct_extract_seconds": totals.get("timing_direct_extract_seconds"),
        "timing_binary_ijbox_context_seconds": totals.get("timing_binary_ijbox_context_seconds"),
        "timing_binary_dump_seconds": totals.get("timing_binary_dump_seconds"),
        "timing_binary_read_seconds": totals.get("timing_binary_read_seconds"),
        "timing_row_build_seconds": totals.get("timing_row_build_seconds"),
        "failed_row_count": parity.get("failed_row_count"),
        "max_row_numeric_diff": parity.get("max_row_numeric_diff"),
        "max_row_numeric_column": parity.get("max_row_numeric_column"),
        "max_summary_numeric_diff": summary.get("max_numeric_diff"),
        "max_summary_numeric_column": summary.get("max_numeric_column"),
    }


def write_outputs(output_dir: Path, payload: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    run_rows = [flatten_run(run) for run in runs if isinstance(run, dict)]
    (output_dir / "benchmark_runs.json").write_text(json.dumps(runs, indent=2, sort_keys=True) + "\n")
    (output_dir / "benchmark_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    csv_path = output_dir / "benchmark_runs.csv"
    if run_rows:
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)
    else:
        csv_path.write_text("")


def build_summary(args: argparse.Namespace, runs: list[dict[str, object]]) -> dict[str, object]:
    by_method: list[dict[str, object]] = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for run in runs:
        grouped.setdefault(str(run["method"]), []).append(run)
    for method, method_runs in sorted(grouped.items()):
        wall_values = [float(run["wall_seconds"]) for run in method_runs]
        cpu_values = [float(run["cpu_seconds"]) for run in method_runs]
        by_method.append(
            {
                "method": method,
                "runs": len(method_runs),
                "all_ok": all(bool(run.get("ok")) for run in method_runs),
                "median_wall_seconds": statistics.median(wall_values),
                "best_wall_seconds": min(wall_values),
                "median_cpu_seconds": statistics.median(cpu_values),
            }
        )
    return {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target": {
            "reduced_grib": str(args.reduced_grib),
            "direct_grib": str(args.direct_grib) if args.direct_grib is not None else None,
            "target_date": str(args.target_date),
            "run_date_utc": args.run_date_utc,
            "cycle_hour_utc": int(args.cycle_hour_utc),
            "forecast_hours": sorted(parse_forecast_hours(str(args.forecast_hours))),
            "selection_mode": str(args.selection_mode),
            "summary_profile": str(args.summary_profile),
            "reference_method": str(args.reference_method),
            "numeric_tolerance": float(args.numeric_tolerance),
        },
        "methods": by_method,
        "runs": runs,
    }


def main() -> int:
    args = parse_args()
    if not args.reduced_grib.exists():
        raise SystemExit(f"Reduced GRIB does not exist: {args.reduced_grib}")
    if "wgrib2-ijbox-bin" in args.methods and args.direct_grib is None:
        raise SystemExit("--direct-grib is required when benchmarking wgrib2-ijbox-bin")
    if args.direct_grib is not None and not args.direct_grib.exists():
        raise SystemExit(f"Direct GRIB does not exist: {args.direct_grib}")
    tasks = matching_tasks(args)
    if not tasks:
        raise SystemExit("No matching HRRR tasks found for the requested target/cycle/forecast-hour selection.")
    wgrib2_path = hrrr.ensure_tooling()
    reduced_inventory = hrrr.inventory_for_grib(wgrib2_path, args.reduced_grib)
    direct_inventory = hrrr.inventory_for_grib(wgrib2_path, args.direct_grib) if args.direct_grib is not None else None
    output_dir = args.output_dir or default_output_dir()

    runs: list[dict[str, object]] = []
    reference_by_run: dict[int, dict[str, object]] = {}
    requested_methods = list(dict.fromkeys([str(args.reference_method), *[str(method) for method in args.methods]]))
    for run_index in range(1, max(1, int(args.runs)) + 1):
        for method in requested_methods:
            if not args.keep_binary_cache:
                clear_wgrib2_binary_cache()
            grib_path = args.direct_grib if method == "wgrib2-ijbox-bin" and args.direct_grib is not None else args.reduced_grib
            inventory = direct_inventory if method == "wgrib2-ijbox-bin" and direct_inventory is not None else reduced_inventory
            print(f"[benchmark] method={method} run={run_index} tasks={len(tasks)} grib={grib_path}", flush=True)
            run = run_method(
                method=method,
                grib_path=grib_path,
                inventory=inventory,
                tasks=tasks,
                summary_profile=str(args.summary_profile),
                include_provenance=bool(args.include_provenance),
                include_legacy_aliases=bool(args.legacy_aliases),
            )
            run["run_index"] = run_index
            if method == str(args.reference_method):
                reference_by_run[run_index] = run
                run["parity"] = {
                    "reference_method": method,
                    "candidate_method": method,
                    "self_reference": True,
                }
            else:
                reference = reference_by_run.get(run_index)
                if reference is None:
                    raise RuntimeError(f"Reference method {args.reference_method} did not run before {method}")
                run["parity"] = compare_to_reference(
                    reference=reference,
                    candidate=run,
                    target_date=str(args.target_date),
                    tolerance=float(args.numeric_tolerance),
                )
            runs.append(run)
            parity = run.get("parity") if isinstance(run.get("parity"), dict) else {}
            print(
                f"[result] method={method} run={run_index} ok={run['ok']} "
                f"wall={float(run['wall_seconds']):.3f}s cpu={float(run['cpu_seconds']):.3f}s "
                f"max_row_diff={parity.get('max_row_numeric_diff')}",
                flush=True,
            )

    payload = build_summary(args, runs)
    write_outputs(output_dir, payload)
    print(f"output_dir={output_dir}")
    for item in payload["methods"]:
        print(
            f"method={item['method']} median_wall={float(item['median_wall_seconds']):.3f}s "
            f"best_wall={float(item['best_wall_seconds']):.3f}s all_ok={item['all_ok']}"
        )
    return 0 if all(bool(run.get("ok")) for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
