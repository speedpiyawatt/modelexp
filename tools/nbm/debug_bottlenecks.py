#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

TIMING_COLUMNS = (
    "timing_idx_fetch_seconds",
    "timing_idx_parse_seconds",
    "timing_head_seconds",
    "timing_range_download_seconds",
    "timing_crop_seconds",
    "timing_cfgrib_open_seconds",
    "timing_row_build_seconds",
    "timing_cleanup_seconds",
)

DEFAULT_MATRIX_CROP_METHODS = ("small_grib", "auto", "ijsmall_grib")
DEFAULT_MATRIX_WGRIB2_THREADS = (1, 2)
DEFAULT_MATRIX_CROP_PACKINGS = ("same", "complex1", "complex3")


@dataclass(frozen=True)
class BenchmarkConfig:
    lead_workers: int
    crop_method: str
    wgrib2_threads: int | None
    crop_packing: str
    repetition_index: int = 1


@dataclass
class Sample:
    elapsed_seconds: float
    cpu_pct: float
    rss_mb: float
    process_count: int


@dataclass
class RunSummary:
    lead_workers: int
    crop_method: str
    wgrib2_threads: int | None
    crop_packing: str
    wall_seconds: float
    sample_count: int
    avg_cpu_pct: float
    peak_cpu_pct: float
    avg_rss_mb: float
    peak_rss_mb: float
    manifest_rows: int
    ok_rows: int
    error_rows: int
    total_downloaded_mb: float
    total_selected_mb: float
    total_reduced_mb: float
    total_stage_seconds: dict[str, float]
    avg_stage_seconds: dict[str, float]
    p95_stage_seconds: dict[str, float]
    stage_pressure_x: dict[str, float]
    wall_download_mb_s: float | None
    wall_selected_mb_s: float | None
    top_slowest_units: list[dict[str, object]]
    output_dir: str
    repetition_index: int = 1
    config_slug: str = ""
    succeeded: bool = True
    return_code: int = 0
    stdout_log: str = ""
    stderr_log: str = ""
    failure_message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark NBM build bottlenecks for one or more crop configurations."
    )
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    parser.add_argument("--selection-mode", default="overnight_0005")
    parser.add_argument("--workers", type=int, default=1, help="Outer NBM cycle workers.")
    parser.add_argument(
        "--lead-workers",
        type=int,
        action="append",
        required=True,
        help="Lead-worker setting to benchmark. Repeat for comparisons.",
    )
    parser.add_argument(
        "--run-root",
        type=pathlib.Path,
        default=REPO_ROOT / "tmp" / "nbm_debug",
        help="Root directory for benchmark outputs.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=1.0,
        help="Process sampling interval while the builder runs.",
    )
    parser.add_argument(
        "--keep-run-dirs",
        action="store_true",
        help="Keep raw per-run output directories for manual inspection.",
    )
    parser.add_argument(
        "--lead-hour-end",
        type=int,
        default=None,
        help="Optional inclusive upper bound for lead hours during benchmarking.",
    )
    parser.add_argument(
        "--crop-method",
        choices=("auto", "small_grib", "ijsmall_grib"),
        default="auto",
        help="Crop primitive to benchmark in single-run mode.",
    )
    parser.add_argument(
        "--wgrib2-threads",
        type=int,
        help="Optional crop subprocess OpenMP thread override in single-run mode.",
    )
    parser.add_argument(
        "--crop-packings",
        nargs="+",
        default=["same"],
        help="Crop packing variants to benchmark in single-run mode.",
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=float,
        default=10.0,
        help="How often to print live benchmark status while a run is in progress.",
    )
    parser.add_argument(
        "--auto-matrix",
        action="store_true",
        help="Generate and run a full benchmark matrix instead of a single crop-method/thread configuration.",
    )
    parser.add_argument(
        "--matrix-crop-methods",
        nargs="+",
        choices=("auto", "small_grib", "ijsmall_grib"),
        help="Crop methods to sweep in auto-matrix mode.",
    )
    parser.add_argument(
        "--matrix-wgrib2-threads",
        nargs="+",
        type=int,
        help="Crop thread counts to sweep in auto-matrix mode.",
    )
    parser.add_argument(
        "--matrix-crop-packings",
        nargs="+",
        help="Crop packings to sweep in auto-matrix mode.",
    )
    parser.add_argument(
        "--matrix-lead-workers",
        nargs="+",
        type=int,
        help="Lead-worker settings to sweep in auto-matrix mode. Defaults to --lead-workers.",
    )
    parser.add_argument(
        "--matrix-repetitions",
        type=int,
        default=1,
        help="Number of repeated runs per configuration in auto-matrix mode.",
    )
    parser.add_argument(
        "--matrix-max-runs",
        type=int,
        default=48,
        help="Safety cap on the number of generated auto-matrix runs.",
    )
    parser.add_argument(
        "--matrix-baseline",
        help="Optional config slug to use as the comparison baseline in aggregate reports.",
    )
    parser.add_argument(
        "--report-name",
        default="report.md",
        help="Aggregate markdown report filename under --run-root.",
    )
    parser.add_argument(
        "--write-json-summary",
        action="store_true",
        help="Write aggregate JSON summary and matrix metadata beside the markdown report.",
    )
    args = parser.parse_args()
    if args.matrix_repetitions < 1:
        raise SystemExit("--matrix-repetitions must be at least 1.")
    if args.matrix_max_runs is not None and args.matrix_max_runs < 1:
        raise SystemExit("--matrix-max-runs must be at least 1.")
    return args


def thread_label(value: int | None) -> str:
    return "auto" if value is None else str(value)


def slugify_config(config: BenchmarkConfig) -> str:
    base = (
        f"lw{config.lead_workers}_"
        f"{config.crop_method}_"
        f"t{thread_label(config.wgrib2_threads)}_"
        f"{config.crop_packing}"
    )
    if config.repetition_index > 1:
        return f"{base}_r{config.repetition_index}"
    return base


def config_sort_key(config: BenchmarkConfig, crop_method_order: dict[str, int], packing_order: dict[str, int]) -> tuple[int, int, int, int, int]:
    thread_sort = -1 if config.wgrib2_threads is None else config.wgrib2_threads
    return (
        config.lead_workers,
        crop_method_order.get(config.crop_method, 999),
        thread_sort,
        packing_order.get(config.crop_packing, 999),
        config.repetition_index,
    )


def build_benchmark_configs(args: argparse.Namespace) -> list[BenchmarkConfig]:
    if not args.auto_matrix:
        return sorted(
            [
                BenchmarkConfig(
                    lead_workers=lead_workers,
                    crop_method=args.crop_method,
                    wgrib2_threads=args.wgrib2_threads,
                    crop_packing=crop_packing,
                    repetition_index=1,
                )
                for lead_workers in args.lead_workers
                for crop_packing in args.crop_packings
            ],
            key=lambda config: (config.lead_workers, config.crop_packing),
        )

    lead_workers_values = args.matrix_lead_workers or args.lead_workers
    crop_methods = args.matrix_crop_methods or list(DEFAULT_MATRIX_CROP_METHODS)
    thread_values = args.matrix_wgrib2_threads or list(DEFAULT_MATRIX_WGRIB2_THREADS)
    crop_packings = args.matrix_crop_packings or list(DEFAULT_MATRIX_CROP_PACKINGS)
    configs = [
        BenchmarkConfig(
            lead_workers=lead_workers,
            crop_method=crop_method,
            wgrib2_threads=wgrib2_threads,
            crop_packing=crop_packing,
            repetition_index=repetition_index,
        )
        for lead_workers in lead_workers_values
        for crop_method in crop_methods
        for wgrib2_threads in thread_values
        for crop_packing in crop_packings
        for repetition_index in range(1, args.matrix_repetitions + 1)
    ]
    crop_method_order = {value: index for index, value in enumerate(crop_methods)}
    packing_order = {value: index for index, value in enumerate(crop_packings)}
    configs = sorted(configs, key=lambda config: config_sort_key(config, crop_method_order, packing_order))
    if args.matrix_max_runs is not None and len(configs) > args.matrix_max_runs:
        raise SystemExit(
            f"Auto-matrix generated {len(configs)} runs, which exceeds --matrix-max-runs={args.matrix_max_runs}. "
            "Narrow the matrix dimensions or raise the cap explicitly."
        )
    return configs


def write_matrix_metadata(run_root: pathlib.Path, args: argparse.Namespace, configs: list[BenchmarkConfig]) -> pathlib.Path:
    if args.auto_matrix:
        crop_methods = args.matrix_crop_methods or list(DEFAULT_MATRIX_CROP_METHODS)
        thread_values = args.matrix_wgrib2_threads or list(DEFAULT_MATRIX_WGRIB2_THREADS)
        crop_packings = args.matrix_crop_packings or list(DEFAULT_MATRIX_CROP_PACKINGS)
        lead_workers_values = args.matrix_lead_workers or args.lead_workers
    else:
        crop_methods = list(dict.fromkeys(config.crop_method for config in configs))
        thread_values = list(dict.fromkeys(config.wgrib2_threads for config in configs))
        crop_packings = list(dict.fromkeys(config.crop_packing for config in configs))
        lead_workers_values = list(dict.fromkeys(config.lead_workers for config in configs))
    metadata = {
        "auto_matrix": args.auto_matrix,
        "matrix_crop_methods": crop_methods,
        "matrix_wgrib2_threads": thread_values,
        "matrix_crop_packings": crop_packings,
        "matrix_lead_workers": lead_workers_values,
        "matrix_repetitions": args.matrix_repetitions,
        "configs": [{**asdict(config), "config_slug": slugify_config(config)} for config in configs],
    }
    matrix_path = run_root / "matrix.json"
    matrix_path.write_text(json.dumps(metadata, indent=2))
    return matrix_path


def ps_snapshot() -> dict[int, tuple[int, float, float]]:
    command = ["ps", "-axo", "pid=,ppid=,%cpu=,rss="]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    snapshot: dict[int, tuple[int, float, float]] = {}
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) != 4:
            continue
        pid, ppid = int(parts[0]), int(parts[1])
        cpu_pct = float(parts[2])
        rss_mb = float(parts[3]) / 1024.0
        snapshot[pid] = (ppid, cpu_pct, rss_mb)
    return snapshot


def descendant_stats(root_pid: int) -> tuple[float, float, int]:
    snapshot = ps_snapshot()
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _cpu_pct, _rss_mb) in snapshot.items():
            if ppid in descendants and pid not in descendants:
                descendants.add(pid)
                changed = True
    total_cpu_pct = 0.0
    total_rss_mb = 0.0
    live_count = 0
    for pid in descendants:
        item = snapshot.get(pid)
        if item is None:
            continue
        _ppid, cpu_pct, rss_mb = item
        total_cpu_pct += cpu_pct
        total_rss_mb += rss_mb
        live_count += 1
    return total_cpu_pct, total_rss_mb, live_count


def monitor_process(
    process: subprocess.Popen[str],
    *,
    interval_seconds: float,
    samples: list[Sample],
    stop_event: threading.Event,
) -> None:
    started_at = time.perf_counter()
    while not stop_event.is_set():
        if process.poll() is not None:
            break
        try:
            cpu_pct, rss_mb, process_count = descendant_stats(process.pid)
        except Exception:
            cpu_pct, rss_mb, process_count = 0.0, 0.0, 0
        samples.append(
            Sample(
                elapsed_seconds=time.perf_counter() - started_at,
                cpu_pct=cpu_pct,
                rss_mb=rss_mb,
                process_count=process_count,
            )
        )
        stop_event.wait(interval_seconds)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[lower]
    weight = rank - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def summarize_manifest(manifest_paths: list[pathlib.Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in manifest_paths]
    if not frames:
        raise SystemExit("No manifest parquet files were produced.")
    return pd.concat(frames, ignore_index=True)


def slowest_units(manifest: pd.DataFrame, limit: int = 5) -> list[dict[str, object]]:
    timing_subset = manifest.copy()
    timing_subset["total_timing_seconds"] = timing_subset[list(TIMING_COLUMNS)].fillna(0.0).sum(axis=1)
    columns = [
        "init_time_utc",
        "valid_time_utc",
        "forecast_hour",
        "extraction_status",
        "downloaded_range_bytes",
        "selected_record_count",
        "total_timing_seconds",
        *TIMING_COLUMNS,
    ]
    ordered = timing_subset.sort_values("total_timing_seconds", ascending=False).head(limit)
    rows: list[dict[str, object]] = []
    for row in ordered[columns].to_dict(orient="records"):
        rows.append(row)
    return rows


def summarize_run(
    *,
    lead_workers: int,
    crop_method: str,
    wgrib2_threads: int | None,
    crop_packing: str,
    wall_seconds: float,
    samples: list[Sample],
    manifest: pd.DataFrame,
    output_dir: pathlib.Path,
    repetition_index: int = 1,
    config_slug: str = "",
    stdout_path: pathlib.Path | None = None,
    stderr_path: pathlib.Path | None = None,
) -> RunSummary:
    stage_totals = {
        column: float(pd.to_numeric(manifest[column], errors="coerce").fillna(0.0).sum())
        for column in TIMING_COLUMNS
    }
    stage_avgs = {
        column: float(pd.to_numeric(manifest[column], errors="coerce").fillna(0.0).mean())
        for column in TIMING_COLUMNS
    }
    stage_p95 = {
        column: percentile(
            sorted(pd.to_numeric(manifest[column], errors="coerce").fillna(0.0).tolist()),
            0.95,
        )
        for column in TIMING_COLUMNS
    }
    stage_pressure = {
        column: (value / wall_seconds) if wall_seconds > 0 else 0.0
        for column, value in stage_totals.items()
    }
    sample_cpu = [sample.cpu_pct for sample in samples]
    sample_rss = [sample.rss_mb for sample in samples]
    downloaded_mb = float(pd.to_numeric(manifest["downloaded_range_bytes"], errors="coerce").fillna(0.0).sum()) / (
        1024 * 1024
    )
    selected_mb = float(pd.to_numeric(manifest["selected_download_bytes"], errors="coerce").fillna(0.0).sum()) / (
        1024 * 1024
    )
    reduced_mb = float(pd.to_numeric(manifest["reduced_file_size"], errors="coerce").fillna(0.0).sum()) / (
        1024 * 1024
    )
    effective_thread_values = (
        pd.to_numeric(manifest.get("crop_wgrib2_threads"), errors="coerce").dropna()
        if "crop_wgrib2_threads" in manifest
        else pd.Series(dtype="float64")
    )
    effective_wgrib2_threads = int(effective_thread_values.mode().iloc[0]) if not effective_thread_values.empty else wgrib2_threads
    return RunSummary(
        lead_workers=lead_workers,
        crop_method=crop_method,
        wgrib2_threads=effective_wgrib2_threads,
        crop_packing=crop_packing,
        wall_seconds=wall_seconds,
        sample_count=len(samples),
        avg_cpu_pct=float(statistics.fmean(sample_cpu)) if sample_cpu else 0.0,
        peak_cpu_pct=max(sample_cpu) if sample_cpu else 0.0,
        avg_rss_mb=float(statistics.fmean(sample_rss)) if sample_rss else 0.0,
        peak_rss_mb=max(sample_rss) if sample_rss else 0.0,
        manifest_rows=len(manifest),
        ok_rows=int((manifest["extraction_status"] == "ok").sum()),
        error_rows=int((manifest["extraction_status"] != "ok").sum()),
        total_downloaded_mb=downloaded_mb,
        total_selected_mb=selected_mb,
        total_reduced_mb=reduced_mb,
        total_stage_seconds=stage_totals,
        avg_stage_seconds=stage_avgs,
        p95_stage_seconds=stage_p95,
        stage_pressure_x=stage_pressure,
        wall_download_mb_s=(downloaded_mb / wall_seconds) if wall_seconds > 0 else None,
        wall_selected_mb_s=(selected_mb / wall_seconds) if wall_seconds > 0 else None,
        top_slowest_units=slowest_units(manifest),
        output_dir=str(output_dir),
        repetition_index=repetition_index,
        config_slug=config_slug,
        succeeded=True,
        return_code=0,
        stdout_log=str(stdout_path) if stdout_path is not None else "",
        stderr_log=str(stderr_path) if stderr_path is not None else "",
    )


def failed_run_summary(
    *,
    config: BenchmarkConfig,
    wall_seconds: float,
    samples: list[Sample],
    output_dir: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    return_code: int,
) -> RunSummary:
    sample_cpu = [sample.cpu_pct for sample in samples]
    sample_rss = [sample.rss_mb for sample in samples]
    zeros = {column: 0.0 for column in TIMING_COLUMNS}
    return RunSummary(
        lead_workers=config.lead_workers,
        crop_method=config.crop_method,
        wgrib2_threads=config.wgrib2_threads,
        crop_packing=config.crop_packing,
        wall_seconds=wall_seconds,
        sample_count=len(samples),
        avg_cpu_pct=float(statistics.fmean(sample_cpu)) if sample_cpu else 0.0,
        peak_cpu_pct=max(sample_cpu) if sample_cpu else 0.0,
        avg_rss_mb=float(statistics.fmean(sample_rss)) if sample_rss else 0.0,
        peak_rss_mb=max(sample_rss) if sample_rss else 0.0,
        manifest_rows=0,
        ok_rows=0,
        error_rows=0,
        total_downloaded_mb=0.0,
        total_selected_mb=0.0,
        total_reduced_mb=0.0,
        total_stage_seconds=zeros.copy(),
        avg_stage_seconds=zeros.copy(),
        p95_stage_seconds=zeros.copy(),
        stage_pressure_x=zeros.copy(),
        wall_download_mb_s=None,
        wall_selected_mb_s=None,
        top_slowest_units=[],
        output_dir=str(output_dir),
        repetition_index=config.repetition_index,
        config_slug=slugify_config(config),
        succeeded=False,
        return_code=return_code,
        stdout_log=str(stdout_path),
        stderr_log=str(stderr_path),
        failure_message=(
            f"NBM benchmark run failed for {slugify_config(config)}. "
            f"See {stdout_path} and {stderr_path}."
        ),
    )


def build_benchmark_command(
    args: argparse.Namespace,
    *,
    lead_workers: int,
    crop_packing: str,
    output_dir: pathlib.Path,
    scratch_dir: pathlib.Path,
    crop_method: str | None = None,
    wgrib2_threads: int | None = None,
) -> list[str]:
    shared_args = [
        "--start-local-date",
        args.start_local_date,
        "--end-local-date",
        args.end_local_date,
        "--selection-mode",
        args.selection_mode,
        "--output-dir",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--workers",
        str(args.workers),
        "--lead-workers",
        str(lead_workers),
        "--crop-method",
        str(crop_method if crop_method is not None else args.crop_method),
        "--crop-grib-type",
        str(crop_packing),
        "--overwrite",
    ]
    resolved_threads = wgrib2_threads if wgrib2_threads is not None else args.wgrib2_threads
    if resolved_threads is not None:
        shared_args.extend(["--wgrib2-threads", str(resolved_threads)])
    if args.lead_hour_end is None:
        return [
            sys.executable,
            str(SCRIPT_DIR / "build_grib2_features.py"),
            *shared_args,
        ]
    return [
        sys.executable,
        "-c",
        (
            "import pathlib, sys; "
            f"sys.path.insert(0, {str(SCRIPT_DIR)!r}); "
            "import build_grib2_features as mod; "
            f"mod.LEAD_HOURS = list(range(1, {int(args.lead_hour_end) + 1})); "
            "sys.argv = ["
            "'build_grib2_features.py', "
            + ", ".join(repr(item) for item in shared_args)
            + "]; "
            "raise SystemExit(mod.main())"
        ),
    ]


def run_single_benchmark(args: argparse.Namespace, config: BenchmarkConfig) -> RunSummary:
    config_slug = slugify_config(config)
    run_dir = args.run_root / "runs" / config_slug
    if run_dir.exists():
        shutil.rmtree(run_dir)
    output_dir = run_dir / "output"
    scratch_dir = run_dir / "scratch"
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    command = build_benchmark_command(
        args,
        lead_workers=config.lead_workers,
        crop_packing=config.crop_packing,
        output_dir=output_dir,
        scratch_dir=scratch_dir,
        crop_method=config.crop_method,
        wgrib2_threads=config.wgrib2_threads,
    )

    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    print(
        f"[nbm-debug] start config={config_slug} "
        f"lead_workers={config.lead_workers} crop_method={config.crop_method} "
        f"crop_packing={config.crop_packing} wgrib2_threads={thread_label(config.wgrib2_threads)} "
        f"output_dir={output_dir} stdout_log={stdout_path} stderr_log={stderr_path}",
        flush=True,
    )
    started_at = time.perf_counter()
    samples: list[Sample] = []
    stop_event = threading.Event()
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        monitor = threading.Thread(
            target=monitor_process,
            kwargs={
                "process": process,
                "interval_seconds": args.sample_interval_seconds,
                "samples": samples,
                "stop_event": stop_event,
            },
            daemon=True,
        )
        monitor.start()
        next_status_at = time.perf_counter() + max(1.0, float(args.status_interval_seconds))
        while True:
            return_code = process.poll()
            if return_code is not None:
                break
            now = time.perf_counter()
            if now >= next_status_at:
                if samples:
                    latest = samples[-1]
                    print(
                        f"[nbm-debug] running config={config_slug} "
                        f"elapsed={latest.elapsed_seconds:.1f}s "
                        f"cpu_pct={latest.cpu_pct:.1f} rss_mb={latest.rss_mb:.1f} "
                        f"proc_count={latest.process_count}",
                        flush=True,
                    )
                else:
                    print(
                        f"[nbm-debug] running config={config_slug} elapsed={now - started_at:.1f}s "
                        f"status=no_samples_yet",
                        flush=True,
                    )
                next_status_at = now + max(1.0, float(args.status_interval_seconds))
            time.sleep(0.5)
        stop_event.set()
        monitor.join(timeout=5.0)
    wall_seconds = time.perf_counter() - started_at
    print(
        f"[nbm-debug] finished config={config_slug} return_code={return_code} "
        f"wall_seconds={wall_seconds:.2f}",
        flush=True,
    )

    if return_code != 0:
        summary = failed_run_summary(
            config=config,
            wall_seconds=wall_seconds,
            samples=samples,
            output_dir=output_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            return_code=return_code,
        )
        (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
        if not args.keep_run_dirs:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        return summary

    manifest_paths = sorted(output_dir.rglob("*_manifest.parquet"))
    manifest = summarize_manifest(manifest_paths)
    summary = summarize_run(
        lead_workers=config.lead_workers,
        crop_method=config.crop_method,
        wgrib2_threads=config.wgrib2_threads,
        crop_packing=config.crop_packing,
        wall_seconds=wall_seconds,
        samples=samples,
        manifest=manifest,
        output_dir=output_dir,
        repetition_index=config.repetition_index,
        config_slug=config_slug,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    (run_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))
    if not args.keep_run_dirs:
        shutil.rmtree(scratch_dir, ignore_errors=True)
    return summary


def run_benchmark_campaign(args: argparse.Namespace, configs: list[BenchmarkConfig]) -> list[RunSummary]:
    return [run_single_benchmark(args, config) for config in configs]


def stage_total(summary: RunSummary, column: str) -> float:
    return float(summary.total_stage_seconds.get(column, 0.0))


def metric_label(summary: RunSummary) -> str:
    return (
        f"{summary.config_slug or f'lw{summary.lead_workers}'} "
        f"(method={summary.crop_method}, threads={thread_label(summary.wgrib2_threads)}, "
        f"packing={summary.crop_packing}, ok={summary.ok_rows}, err={summary.error_rows})"
    )


def summary_id(summary: RunSummary) -> str:
    return summary.config_slug or (
        f"lw{summary.lead_workers}_{summary.crop_method}_t{thread_label(summary.wgrib2_threads)}_"
        f"{summary.crop_packing}_r{summary.repetition_index}"
    )


def choose_baseline(summaries: list[RunSummary], baseline_slug: str | None) -> RunSummary:
    if baseline_slug:
        for summary in summaries:
            if summary_id(summary) == baseline_slug:
                return summary
        raise SystemExit(f"--matrix-baseline {baseline_slug!r} did not match any generated config slug.")
    successful = [summary for summary in summaries if summary.succeeded and summary.error_rows == 0]
    if not successful:
        successful = [summary for summary in summaries if summary.succeeded]
    if not successful:
        return summaries[0]
    preferred = sorted(
        successful,
        key=lambda summary: (
            0 if summary.crop_method == "small_grib" else 1,
            0 if summary.crop_packing == "same" else 1,
            summary.wgrib2_threads if summary.wgrib2_threads is not None else 999,
            summary.lead_workers,
            summary.repetition_index,
        ),
    )
    return preferred[0]


def rank_lines(
    summaries: list[RunSummary],
    *,
    label: str,
    key_fn,
    formatter,
    include_failed: bool = False,
    limit: int = 5,
) -> list[str]:
    rows = summaries if include_failed else [summary for summary in summaries if summary.succeeded and summary.error_rows == 0]
    ordered = sorted(rows, key=key_fn)
    lines = [f"### {label}", ""]
    if not ordered:
        lines.append("- no fully successful runs")
        lines.append("")
        return lines
    for summary in ordered[:limit]:
        lines.append(f"- {metric_label(summary)}: {formatter(summary)}")
    lines.append("")
    return lines


def best_by_attribute(summaries: list[RunSummary], attribute: str) -> list[str]:
    successful = [summary for summary in summaries if summary.succeeded and summary.error_rows == 0]
    groups: dict[object, list[RunSummary]] = {}
    for summary in successful:
        groups.setdefault(getattr(summary, attribute), []).append(summary)
    lines = [f"### Best By {attribute.replace('_', ' ').title()}", ""]
    if not groups:
        lines.append("- no fully successful runs")
        lines.append("")
        return lines
    for key in sorted(groups, key=lambda value: (str(type(value)), value)):
        best = min(groups[key], key=lambda summary: summary.wall_seconds)
        lines.append(f"- {attribute}={key}: {metric_label(best)} wall_seconds={best.wall_seconds:.2f}")
    lines.append("")
    return lines


def write_campaign_report(
    run_root: pathlib.Path,
    summaries: list[RunSummary],
    *,
    baseline_selector: str | None,
    matrix_meta: dict[str, object],
    report_name: str,
) -> pathlib.Path:
    baseline = choose_baseline(summaries, baseline_selector)
    lines = [
        "# NBM Bottleneck Report",
        "",
        "## Campaign",
        "",
        f"- total_runs: {len(summaries)}",
        f"- successful_runs: {sum(1 for summary in summaries if summary.succeeded)}",
        f"- failed_runs: {sum(1 for summary in summaries if not summary.succeeded)}",
        f"- baseline: {summary_id(baseline)}",
        "",
        "### Matrix",
        "",
    ]
    for key in (
        "matrix_lead_workers",
        "matrix_crop_methods",
        "matrix_wgrib2_threads",
        "matrix_crop_packings",
        "matrix_repetitions",
    ):
        if key in matrix_meta:
            lines.append(f"- {key}: {matrix_meta[key]}")
    lines.extend(["", "## Per-Config Summary", ""])
    for summary in summaries:
        lines.extend(
            [
                f"### {summary.config_slug or f'lead-workers={summary.lead_workers}'}",
                "",
                f"- status: {'ok' if summary.succeeded else 'failed'}",
                f"- lead_workers: {summary.lead_workers}",
                f"- crop_method: {summary.crop_method}",
                f"- crop_packing: {summary.crop_packing}",
                f"- wgrib2_threads: {thread_label(summary.wgrib2_threads)}",
                f"- repetition_index: {summary.repetition_index}",
                f"- wall_seconds: {summary.wall_seconds:.2f}",
                f"- avg_cpu_pct: {summary.avg_cpu_pct:.1f}",
                f"- peak_cpu_pct: {summary.peak_cpu_pct:.1f}",
                f"- avg_rss_mb: {summary.avg_rss_mb:.1f}",
                f"- peak_rss_mb: {summary.peak_rss_mb:.1f}",
                f"- manifest_rows: {summary.manifest_rows}",
                f"- ok_rows: {summary.ok_rows}",
                f"- error_rows: {summary.error_rows}",
                f"- total_downloaded_mb: {summary.total_downloaded_mb:.1f}",
                f"- total_reduced_mb: {summary.total_reduced_mb:.1f}",
                f"- stdout_log: {summary.stdout_log}",
                f"- stderr_log: {summary.stderr_log}",
                f"- wall_download_mb_s: {summary.wall_download_mb_s:.2f}" if summary.wall_download_mb_s is not None else "- wall_download_mb_s: n/a",
                f"- wall_selected_mb_s: {summary.wall_selected_mb_s:.2f}" if summary.wall_selected_mb_s is not None else "- wall_selected_mb_s: n/a",
            ]
        )
        if summary.failure_message:
            lines.append(f"- failure_message: {summary.failure_message}")
        lines.extend(["", "#### Stage Pressure", ""])
        lines.append(
            "Stage pressure is summed per-unit stage time divided by wall time. "
            "Values above 1.0 mean that stage had overlapping work across concurrent leads."
        )
        lines.append("")
        for column, value in sorted(summary.stage_pressure_x.items(), key=lambda item: item[1], reverse=True):
            lines.append(
                f"- {column}: total={summary.total_stage_seconds[column]:.2f}s avg={summary.avg_stage_seconds[column]:.2f}s "
                f"p95={summary.p95_stage_seconds[column]:.2f}s pressure_x={value:.2f}"
            )
        lines.extend(["", "#### Slowest Units", ""])
        if not summary.top_slowest_units:
            lines.append("- none")
        else:
            for unit in summary.top_slowest_units:
                lines.append(
                    f"- forecast_hour={unit.get('forecast_hour')} valid_time_utc={unit.get('valid_time_utc')} "
                    f"status={unit.get('extraction_status')} total={unit.get('total_timing_seconds'):.2f}s "
                    f"download={unit.get('timing_range_download_seconds', 0.0):.2f}s "
                    f"crop={unit.get('timing_crop_seconds', 0.0):.2f}s "
                    f"open={unit.get('timing_cfgrib_open_seconds', 0.0):.2f}s "
                    f"rows={unit.get('timing_row_build_seconds', 0.0):.2f}s"
                )
        lines.append("")

    lines.extend(["## Rankings", ""])
    lines.extend(
        rank_lines(
            summaries,
            label="Fastest Wall Time",
            key_fn=lambda summary: summary.wall_seconds,
            formatter=lambda summary: f"wall_seconds={summary.wall_seconds:.2f}",
        )
    )
    lines.extend(
        rank_lines(
            summaries,
            label="Lowest Crop Time",
            key_fn=lambda summary: stage_total(summary, "timing_crop_seconds"),
            formatter=lambda summary: f"timing_crop_seconds={stage_total(summary, 'timing_crop_seconds'):.2f}",
        )
    )
    lines.extend(
        rank_lines(
            summaries,
            label="Lowest Cfgrib Open Time",
            key_fn=lambda summary: stage_total(summary, "timing_cfgrib_open_seconds"),
            formatter=lambda summary: f"timing_cfgrib_open_seconds={stage_total(summary, 'timing_cfgrib_open_seconds'):.2f}",
        )
    )
    lines.extend(
        rank_lines(
            summaries,
            label="Smallest Reduced Output",
            key_fn=lambda summary: summary.total_reduced_mb,
            formatter=lambda summary: f"total_reduced_mb={summary.total_reduced_mb:.2f}",
        )
    )
    lines.extend(best_by_attribute(summaries, "lead_workers"))
    lines.extend(best_by_attribute(summaries, "crop_method"))
    lines.extend(best_by_attribute(summaries, "wgrib2_threads"))
    lines.extend(best_by_attribute(summaries, "crop_packing"))

    lines.extend(["## Comparison", ""])
    for contender in summaries:
        if contender is baseline:
            continue
        speedup = baseline.wall_seconds / contender.wall_seconds if contender.wall_seconds > 0 else 0.0
        crop_delta = stage_total(contender, "timing_crop_seconds") - stage_total(baseline, "timing_crop_seconds")
        open_delta = stage_total(contender, "timing_cfgrib_open_seconds") - stage_total(baseline, "timing_cfgrib_open_seconds")
        lines.append(
            f"- {summary_id(contender)} vs {summary_id(baseline)}: "
            f"wall speedup={speedup:.2f}x, crop_delta_seconds={crop_delta:.2f}, "
            f"cfgrib_open_delta_seconds={open_delta:.2f}, "
            f"reduced_delta_mb={contender.total_reduced_mb - baseline.total_reduced_mb:.2f}, "
            f"avg_cpu_delta={contender.avg_cpu_pct - baseline.avg_cpu_pct:.1f}, "
            f"peak_rss_delta_mb={contender.peak_rss_mb - baseline.peak_rss_mb:.1f}"
        )
    lines.append("")

    lines.extend(["## Recommended Current Production Candidate", ""])
    successful = [summary for summary in summaries if summary.succeeded and summary.error_rows == 0]
    if successful:
        candidate = min(
            successful,
            key=lambda summary: (
                summary.wall_seconds,
                stage_total(summary, "timing_cfgrib_open_seconds"),
                summary.peak_rss_mb,
            ),
        )
        lines.extend(
            [
                f"- config: {summary_id(candidate)}",
                f"- wall_seconds: {candidate.wall_seconds:.2f}",
                f"- crop_method: {candidate.crop_method}",
                f"- crop_packing: {candidate.crop_packing}",
                f"- wgrib2_threads: {thread_label(candidate.wgrib2_threads)}",
                f"- lead_workers: {candidate.lead_workers}",
                "",
            ]
        )
    else:
        lines.extend(["- no successful runs available", ""])

    report_path = run_root / report_name
    report_path.write_text("\n".join(lines))
    return report_path


def write_comparison_report(run_root: pathlib.Path, summaries: list[RunSummary]) -> pathlib.Path:
    matrix_meta = {
        "matrix_lead_workers": sorted({summary.lead_workers for summary in summaries}),
        "matrix_crop_methods": list(dict.fromkeys(summary.crop_method for summary in summaries)),
        "matrix_wgrib2_threads": list(dict.fromkeys(summary.wgrib2_threads for summary in summaries)),
        "matrix_crop_packings": list(dict.fromkeys(summary.crop_packing for summary in summaries)),
        "matrix_repetitions": max((summary.repetition_index for summary in summaries), default=1),
    }
    return write_campaign_report(
        run_root,
        summaries,
        baseline_selector=None,
        matrix_meta=matrix_meta,
        report_name="report.md",
    )


def write_json_summary(run_root: pathlib.Path, summaries: list[RunSummary]) -> pathlib.Path:
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2))
    return summary_path


def main() -> int:
    args = parse_args()
    args.run_root.mkdir(parents=True, exist_ok=True)
    configs = build_benchmark_configs(args)
    matrix_path = write_matrix_metadata(args.run_root, args, configs)
    summaries = run_benchmark_campaign(args, configs)
    matrix_meta = json.loads(matrix_path.read_text())
    report_path = write_campaign_report(
        args.run_root,
        summaries,
        baseline_selector=args.matrix_baseline,
        matrix_meta=matrix_meta,
        report_name=args.report_name,
    )
    if args.write_json_summary:
        write_json_summary(args.run_root, summaries)
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
