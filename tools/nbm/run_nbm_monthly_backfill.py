#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_nbm_overnight_features
from tools.weather.progress import create_progress_reporter, resolve_progress_mode

DEFAULT_RUN_ROOT = Path("data/runtime/backfill_overnight")
DEFAULT_SELECTION_MODE = "overnight_0005"
DEFAULT_DAY_WORKERS = 1
DEFAULT_MIN_FREE_GB = 5.0
RUN_LOG_LOCK = threading.Lock()
PROGRESS_PREFIX = "[progress]"
PROGRESS_KV_RE = re.compile(r"^(?P<key>[^=\s]+)=(?P<value>.*)$")


@dataclass(frozen=True)
class DayWindow:
    target_date_local: dt.date


@dataclass
class DayProgressState:
    lifecycle_phase: str = "queued"
    details: str = "queued"
    target_date: str = ""
    selected_issue: str | None = None
    expected_leads: int | None = None
    downloaded_children: set[str] | None = None
    child_phases: dict[str, str] | None = None
    child_labels: dict[str, str] | None = None
    retry_count: int = 0
    batch_status: str = "queued"
    extract_status: str = "queued"
    overnight_status: str = "--"
    status: str = "queued"
    started_at: float | None = None
    ended_at: float | None = None

    def __post_init__(self) -> None:
        if self.downloaded_children is None:
            self.downloaded_children = set()
        if self.child_phases is None:
            self.child_phases = {}
        if self.child_labels is None:
            self.child_labels = {}


@dataclass(frozen=True)
class SmartWorkerPlan:
    cpu_cores: int
    day_workers: int
    workers: int
    lead_workers: int
    download_workers: int
    reduce_workers: int
    extract_workers: int
    reduce_queue_size: int | None = None
    extract_queue_size: int | None = None


@dataclass(frozen=True)
class WorkerPlan:
    day_workers: int
    workers: int
    lead_workers: int
    download_workers: int | None
    reduce_workers: int | None
    extract_workers: int | None
    reduce_queue_size: int | None
    extract_queue_size: int | None


@dataclass(frozen=True)
class WorkerPlanBounds:
    day_workers_min: int
    day_workers_max: int
    lead_workers_min: int
    lead_workers_max: int
    reduce_workers_min: int
    reduce_workers_max: int
    extract_workers_min: int
    extract_workers_max: int


@dataclass(frozen=True)
class DayPerformance:
    token: str
    lead_count_completed: int
    raw_build_seconds: float
    cleanup_seconds: float
    full_day_seconds: float
    timing_crop_seconds_median: float | None
    timing_crop_seconds_p95: float | None
    timing_cfgrib_open_seconds_median: float | None
    timing_cfgrib_open_seconds_p95: float | None
    timing_row_metric_seconds_median: float | None
    timing_row_metric_seconds_p95: float | None
    timing_row_provenance_seconds_median: float | None
    timing_row_provenance_seconds_p95: float | None
    cfgrib_open_all_dataset_count_mean: float | None
    cfgrib_filtered_fallback_attempt_count_mean: float | None
    wide_row_count_mean: float | None
    long_row_count_mean: float | None
    provenance_row_count_mean: float | None


@dataclass(frozen=True)
class DayRunResult:
    token: str
    performance: DayPerformance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NBM overnight backfill day by day, skip valid daily outputs already on disk, and rebuild invalid days."
    )
    parser.add_argument("--start-local-date", required=True, help="Inclusive first target local date in YYYY-MM-DD.")
    parser.add_argument("--end-local-date", required=True, help="Inclusive last target local date in YYYY-MM-DD.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--day-workers", type=int, default=DEFAULT_DAY_WORKERS)
    parser.add_argument(
        "--smart-workers",
        action="store_true",
        help="Auto-tune monthly day admission and child worker counts for the available CPU cores.",
    )
    parser.add_argument(
        "--cpu-cores",
        type=int,
        help="Optional CPU core count override for --smart-workers planning. Defaults to os.cpu_count().",
    )
    parser.add_argument(
        "--adaptive-workers",
        action="store_true",
        help="Adapt future day admissions based on measured backfill performance while keeping download concurrency fixed.",
    )
    parser.add_argument(
        "--adaptive-sample-days",
        type=int,
        default=2,
        help="Successful completed days required before the adaptive controller can tune worker counts.",
    )
    parser.add_argument(
        "--adaptive-cooldown-days",
        type=int,
        default=2,
        help="Successful completed days to wait after an adaptive tuning change before another adjustment.",
    )
    parser.add_argument("--adaptive-min-day-workers", type=int)
    parser.add_argument("--adaptive-max-day-workers", type=int)
    parser.add_argument("--adaptive-min-reduce-workers", type=int)
    parser.add_argument("--adaptive-max-reduce-workers", type=int)
    parser.add_argument("--adaptive-min-extract-workers", type=int)
    parser.add_argument("--adaptive-max-extract-workers", type=int)
    parser.add_argument("--adaptive-min-lead-workers", type=int)
    parser.add_argument("--adaptive-max-lead-workers", type=int)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lead-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--reduce-workers", type=int)
    parser.add_argument("--extract-workers", type=int)
    parser.add_argument("--reduce-queue-size", type=int)
    parser.add_argument("--extract-queue-size", type=int)
    parser.add_argument(
        "--crop-method",
        choices=("auto", "small_grib", "ijsmall_grib"),
        default="small_grib",
        help="Raw-builder crop primitive for monthly backfills. Defaults to small_grib based on benchmark results.",
    )
    parser.add_argument(
        "--wgrib2-threads",
        type=int,
        default=1,
        help="OpenMP thread count for raw-builder crop subprocesses. Defaults to 1 for monthly backfill throughput.",
    )
    parser.add_argument(
        "--crop-grib-type",
        default="complex3",
        choices=("same", "complex1", "complex3"),
        help="Raw-builder crop packing for monthly backfills. Defaults to complex3 based on benchmark results.",
    )
    parser.add_argument(
        "--keep-reduced",
        action="store_true",
        help="Retain reduced cropped GRIB2 files to speed reruns/debugging at the cost of extra disk.",
    )
    parser.add_argument(
        "--overnight-subprocess",
        action="store_true",
        help="Run overnight finalization as a subprocess for debugging instead of the default in-process finalizer.",
    )
    parser.add_argument(
        "--overnight-fast",
        action="store_true",
        help="Pass --skip-provenance to the raw builder for production overnight backfills.",
    )
    parser.add_argument(
        "--metric-profile",
        choices=("full", "overnight"),
        default="overnight",
        help=(
            "Raw-builder metric extraction profile. Monthly overnight backfills default to overnight, "
            "which computes only metrics consumed by nbm.overnight.parquet."
        ),
    )
    parser.add_argument(
        "--batch-reduce-mode",
        choices=("off", "cycle"),
        default="off",
        help="Pass through to the raw builder. cycle batches selected lead GRIBs for one crop/cfgrib open per issue.",
    )
    parser.add_argument("--progress-mode", choices=("auto", "dashboard", "log"), default="auto")
    parser.add_argument(
        "--disable-dashboard-hotkeys",
        action="store_true",
        help="Disable interactive dashboard hotkeys such as 'p' for graceful drain-and-pause.",
    )
    parser.add_argument(
        "--pause-control-file",
        help="Optional file path that the monthly runner watches to request graceful drain-and-pause without forwarding that control into child builders.",
    )
    parser.add_argument("--max-task-attempts", type=int)
    parser.add_argument("--retry-backoff-seconds", type=float)
    parser.add_argument("--retry-max-backoff-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild all requested days even when existing outputs validate.")
    parser.add_argument(
        "--keep-temp-on-failure",
        action="store_true",
        help="Keep the failed day temp tree instead of deleting it.",
    )
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def iter_days(start_local_date: str, end_local_date: str) -> list[DayWindow]:
    start_date = parse_local_date(start_local_date)
    end_date = parse_local_date(end_local_date)
    if end_date < start_date:
        raise ValueError("--end-local-date must be on or after --start-local-date")
    return [DayWindow(start_date + dt.timedelta(days=offset)) for offset in range((end_date - start_date).days + 1)]


def day_token(day: DayWindow) -> str:
    return day.target_date_local.isoformat()


def run_command(command: list[str]) -> None:
    emit_runner_log("RUN", " ".join(command))
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True, start_new_session=True)
    except TypeError as exc:
        if "start_new_session" not in str(exc):
            raise
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def emit_runner_log(tag: str, message: str) -> None:
    with RUN_LOG_LOCK:
        print(f"[{tag}] {message}", flush=True)


def free_disk_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def build_smart_worker_plan(args: argparse.Namespace) -> SmartWorkerPlan | None:
    if not bool(getattr(args, "smart_workers", False)):
        return None
    cpu_cores = max(1, int(getattr(args, "cpu_cores", None) or (os.cpu_count() or 8)))
    if str(getattr(args, "batch_reduce_mode", "off")) == "cycle":
        day_workers = cpu_cores
        download_workers = max(4, cpu_cores)
        return SmartWorkerPlan(
            cpu_cores=cpu_cores,
            day_workers=day_workers,
            workers=1,
            lead_workers=download_workers,
            download_workers=download_workers,
            reduce_workers=1,
            extract_workers=1,
            reduce_queue_size=2,
            extract_queue_size=2,
        )
    reserve_cores = 4 if cpu_cores >= 24 else 2
    cpu_budget = max(4, cpu_cores - reserve_cores)
    target_cpu_heavy_per_day = 10 if cpu_cores >= 24 else 8
    day_workers = max(1, cpu_budget // target_cpu_heavy_per_day)
    per_day_budget = max(8, cpu_budget // day_workers)
    workers = 1
    lead_workers = max(8, per_day_budget)
    download_workers = max(8, per_day_budget)
    reduce_workers = max(4, per_day_budget // 2)
    extract_workers = max(4, per_day_budget // 2)
    return SmartWorkerPlan(
        cpu_cores=cpu_cores,
        day_workers=day_workers,
        workers=workers,
        lead_workers=lead_workers,
        download_workers=download_workers,
        reduce_workers=reduce_workers,
        extract_workers=extract_workers,
    )


def apply_smart_worker_plan(args: argparse.Namespace) -> SmartWorkerPlan | None:
    plan = build_smart_worker_plan(args)
    if plan is None:
        return None
    args.day_workers = plan.day_workers
    args.workers = plan.workers
    args.lead_workers = plan.lead_workers
    args.download_workers = plan.download_workers
    args.reduce_workers = plan.reduce_workers
    args.extract_workers = plan.extract_workers
    if plan.reduce_queue_size is not None:
        args.reduce_queue_size = plan.reduce_queue_size
    if plan.extract_queue_size is not None:
        args.extract_queue_size = plan.extract_queue_size
    return plan


def queue_size_for_workers(worker_count: int) -> int:
    return max(1, int(worker_count) * 3)


def initial_worker_plan(args: argparse.Namespace) -> WorkerPlan:
    reduce_workers = getattr(args, "reduce_workers", None)
    extract_workers = getattr(args, "extract_workers", None)
    return WorkerPlan(
        day_workers=max(1, int(getattr(args, "day_workers", DEFAULT_DAY_WORKERS))),
        workers=max(1, int(getattr(args, "workers", 1))),
        lead_workers=max(1, int(getattr(args, "lead_workers", 1))),
        download_workers=(None if getattr(args, "download_workers", None) is None else max(1, int(args.download_workers))),
        reduce_workers=(None if reduce_workers is None else max(1, int(reduce_workers))),
        extract_workers=(None if extract_workers is None else max(1, int(extract_workers))),
        reduce_queue_size=(getattr(args, "reduce_queue_size", None)),
        extract_queue_size=(getattr(args, "extract_queue_size", None)),
    )


def resolved_worker_plan(plan: WorkerPlan) -> WorkerPlan:
    lead_workers = max(1, int(plan.lead_workers))
    reduce_workers = max(1, int(plan.reduce_workers if plan.reduce_workers is not None else lead_workers))
    extract_workers = max(1, int(plan.extract_workers if plan.extract_workers is not None else lead_workers))
    download_workers = None if plan.download_workers is None else max(1, int(plan.download_workers))
    reduce_queue_size = (
        max(1, int(plan.reduce_queue_size))
        if plan.reduce_queue_size is not None
        else queue_size_for_workers(reduce_workers)
    )
    extract_queue_size = (
        max(1, int(plan.extract_queue_size))
        if plan.extract_queue_size is not None
        else queue_size_for_workers(extract_workers)
    )
    return WorkerPlan(
        day_workers=max(1, int(plan.day_workers)),
        workers=max(1, int(plan.workers)),
        lead_workers=lead_workers,
        download_workers=download_workers,
        reduce_workers=reduce_workers,
        extract_workers=extract_workers,
        reduce_queue_size=reduce_queue_size,
        extract_queue_size=extract_queue_size,
    )


def resolve_worker_plan_bounds(args: argparse.Namespace, plan: WorkerPlan) -> WorkerPlanBounds:
    resolved_plan = resolved_worker_plan(plan)
    if str(getattr(args, "batch_reduce_mode", "off")) == "cycle":
        day_workers_min = max(1, int(getattr(args, "adaptive_min_day_workers", None) or 1))
        day_workers_max = max(day_workers_min, int(getattr(args, "adaptive_max_day_workers", None) or resolved_plan.day_workers))
        reduce_max_default = (
            resolved_plan.reduce_workers
            if getattr(args, "reduce_workers", None) is not None
            else 1
        )
        extract_max_default = (
            resolved_plan.extract_workers
            if getattr(args, "extract_workers", None) is not None
            else 1
        )
        reduce_workers_max = max(1, int(getattr(args, "adaptive_max_reduce_workers", None) or reduce_max_default))
        extract_workers_max = max(1, int(getattr(args, "adaptive_max_extract_workers", None) or extract_max_default))
        return WorkerPlanBounds(
            day_workers_min=day_workers_min,
            day_workers_max=day_workers_max,
            lead_workers_min=resolved_plan.lead_workers,
            lead_workers_max=resolved_plan.lead_workers,
            reduce_workers_min=1,
            reduce_workers_max=reduce_workers_max,
            extract_workers_min=1,
            extract_workers_max=extract_workers_max,
        )
    day_max_default = resolved_plan.day_workers + 1
    lead_max_default = resolved_plan.lead_workers
    reduce_max_default = resolved_plan.reduce_workers or resolved_plan.lead_workers
    extract_max_default = resolved_plan.extract_workers or resolved_plan.lead_workers

    day_workers_min = max(1, int(getattr(args, "adaptive_min_day_workers", None) or 1))
    day_workers_max = max(day_workers_min, int(getattr(args, "adaptive_max_day_workers", None) or day_max_default))

    lead_min_default = resolved_plan.lead_workers
    lead_workers_min = max(1, int(getattr(args, "adaptive_min_lead_workers", None) or lead_min_default))
    lead_workers_max = max(lead_workers_min, int(getattr(args, "adaptive_max_lead_workers", None) or lead_max_default))

    reduce_workers_min = max(1, int(getattr(args, "adaptive_min_reduce_workers", None) or 1))
    reduce_workers_max = max(reduce_workers_min, int(getattr(args, "adaptive_max_reduce_workers", None) or reduce_max_default))

    extract_workers_min = max(1, int(getattr(args, "adaptive_min_extract_workers", None) or 1))
    extract_workers_max = max(extract_workers_min, int(getattr(args, "adaptive_max_extract_workers", None) or extract_max_default))

    return WorkerPlanBounds(
        day_workers_min=day_workers_min,
        day_workers_max=day_workers_max,
        lead_workers_min=lead_workers_min,
        lead_workers_max=lead_workers_max,
        reduce_workers_min=reduce_workers_min,
        reduce_workers_max=reduce_workers_max,
        extract_workers_min=extract_workers_min,
        extract_workers_max=extract_workers_max,
    )


def _metric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _series_stat(frame: pd.DataFrame, column: str, *, op: str) -> float | None:
    series = _metric_series(frame, column)
    if series.empty:
        return None
    if op == "median":
        return float(series.median())
    if op == "p95":
        return float(series.quantile(0.95))
    if op == "mean":
        return float(series.mean())
    raise ValueError(f"Unsupported series op: {op}")


def collect_day_performance(*, token: str, raw_dir: Path, raw_build_seconds: float, cleanup_seconds: float, full_day_seconds: float) -> DayPerformance:
    manifest_root = raw_dir / "metadata" / "manifest"
    manifest_paths = sorted(manifest_root.glob("**/*.parquet")) if manifest_root.exists() else []
    if manifest_paths:
        frames = [pd.read_parquet(path) for path in manifest_paths]
        manifest_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        manifest_df = pd.DataFrame()
    if manifest_df.empty:
        lead_count_completed = 0
    elif "extraction_status" in manifest_df:
        lead_count_completed = int((manifest_df["extraction_status"] == "ok").sum())
    else:
        lead_count_completed = int(len(manifest_df))
    return DayPerformance(
        token=token,
        lead_count_completed=lead_count_completed,
        raw_build_seconds=float(raw_build_seconds),
        cleanup_seconds=float(cleanup_seconds),
        full_day_seconds=float(full_day_seconds),
        timing_crop_seconds_median=_series_stat(manifest_df, "timing_crop_seconds", op="median"),
        timing_crop_seconds_p95=_series_stat(manifest_df, "timing_crop_seconds", op="p95"),
        timing_cfgrib_open_seconds_median=_series_stat(manifest_df, "timing_cfgrib_open_seconds", op="median"),
        timing_cfgrib_open_seconds_p95=_series_stat(manifest_df, "timing_cfgrib_open_seconds", op="p95"),
        timing_row_metric_seconds_median=_series_stat(manifest_df, "timing_row_metric_seconds", op="median"),
        timing_row_metric_seconds_p95=_series_stat(manifest_df, "timing_row_metric_seconds", op="p95"),
        timing_row_provenance_seconds_median=_series_stat(manifest_df, "timing_row_provenance_seconds", op="median"),
        timing_row_provenance_seconds_p95=_series_stat(manifest_df, "timing_row_provenance_seconds", op="p95"),
        cfgrib_open_all_dataset_count_mean=_series_stat(manifest_df, "cfgrib_open_all_dataset_count", op="mean"),
        cfgrib_filtered_fallback_attempt_count_mean=_series_stat(manifest_df, "cfgrib_filtered_fallback_attempt_count", op="mean"),
        wide_row_count_mean=_series_stat(manifest_df, "wide_row_count", op="mean"),
        long_row_count_mean=_series_stat(manifest_df, "long_row_count", op="mean"),
        provenance_row_count_mean=_series_stat(manifest_df, "provenance_row_count", op="mean"),
    )


class AdaptiveWorkerController:
    def __init__(self, args: argparse.Namespace, *, initial_plan: WorkerPlan) -> None:
        self.enabled = bool(getattr(args, "adaptive_workers", False))
        self.sample_days = max(1, int(getattr(args, "adaptive_sample_days", 2)))
        self.cooldown_days = max(0, int(getattr(args, "adaptive_cooldown_days", 2)))
        self.bounds = resolve_worker_plan_bounds(args, initial_plan)
        self._plan = resolved_worker_plan(
            WorkerPlan(
                day_workers=max(self.bounds.day_workers_min, initial_plan.day_workers),
                workers=initial_plan.workers,
                lead_workers=max(self.bounds.lead_workers_min, initial_plan.lead_workers),
                download_workers=initial_plan.download_workers,
                reduce_workers=max(self.bounds.reduce_workers_min, initial_plan.reduce_workers or initial_plan.lead_workers),
                extract_workers=max(self.bounds.extract_workers_min, initial_plan.extract_workers or initial_plan.lead_workers),
                reduce_queue_size=initial_plan.reduce_queue_size,
                extract_queue_size=initial_plan.extract_queue_size,
            )
        )
        self._recent_successes: deque[DayPerformance] = deque(maxlen=max(self.sample_days * 3, 8))
        self._failures_since_tune = 0
        self._successful_days_since_tune = 0
        self._cooldown_remaining = 0
        self.last_tune = "init"
        self.tune_reason = "disabled" if not self.enabled else "initial_seed"
        self._stable_baseline: dict[str, float] | None = None
        self._preserve_reduce_queue_size = initial_plan.reduce_queue_size is not None
        self._preserve_extract_queue_size = initial_plan.extract_queue_size is not None

    @property
    def current_plan(self) -> WorkerPlan:
        return self._plan

    @property
    def max_day_workers(self) -> int:
        return self.bounds.day_workers_max if self.enabled else self._plan.day_workers

    def metrics(self) -> dict[str, object]:
        return {
            "Adaptive": "on" if self.enabled else "off",
            "PlanDays": self._plan.day_workers,
            "PlanLead": self._plan.lead_workers,
            "PlanDownload": self._plan.download_workers or self._plan.lead_workers,
            "PlanReduce": self._plan.reduce_workers,
            "PlanExtract": self._plan.extract_workers,
            "PlanRQ": self._plan.reduce_queue_size,
            "PlanEQ": self._plan.extract_queue_size,
            "LastTune": self.last_tune,
            "TuneReason": self.tune_reason,
        }

    def describe_plan(self) -> str:
        return (
            f"plan d={self._plan.day_workers} "
            f"l={self._plan.lead_workers} "
            f"r={self._plan.reduce_workers} "
            f"e={self._plan.extract_workers}"
        )

    def record_failure(self) -> tuple[bool, str]:
        if not self.enabled:
            return False, "adaptive_disabled"
        self._failures_since_tune += 1
        if self._cooldown_remaining > 0:
            self.tune_reason = f"cooldown_{self._cooldown_remaining}"
            return False, self.tune_reason
        if self._failures_since_tune < 2:
            self.tune_reason = "waiting_for_repeated_failures"
            return False, self.tune_reason
        if self._plan.day_workers > self.bounds.day_workers_min:
            self._apply_change("day_workers", self._plan.day_workers - 1, reason="repeated_failures")
            return True, self.tune_reason
        if self._plan.reduce_workers is not None and self._plan.reduce_workers > self.bounds.reduce_workers_min:
            self._apply_change("reduce_workers", self._plan.reduce_workers - 1, reason="repeated_failures")
            return True, self.tune_reason
        if self._plan.extract_workers is not None and self._plan.extract_workers > self.bounds.extract_workers_min:
            self._apply_change("extract_workers", self._plan.extract_workers - 1, reason="repeated_failures")
            return True, self.tune_reason
        self.tune_reason = "repeated_failures_no_room"
        return False, self.tune_reason

    def record_success(self, performance: DayPerformance) -> tuple[bool, str]:
        if not self.enabled:
            return False, "adaptive_disabled"
        self._recent_successes.append(performance)
        self._successful_days_since_tune += 1
        if self._cooldown_remaining > 0:
            self._cooldown_remaining = max(0, self._cooldown_remaining - 1)
            self.tune_reason = f"cooldown_{self._cooldown_remaining}"
            return False, self.tune_reason
        if len(self._recent_successes) < self.sample_days:
            self.tune_reason = f"waiting_for_samples_{len(self._recent_successes)}/{self.sample_days}"
            return False, self.tune_reason
        return self._evaluate_window()

    def _window(self) -> list[DayPerformance]:
        return list(self._recent_successes)[-self.sample_days :]

    def _evaluate_window(self) -> tuple[bool, str]:
        window = self._window()
        cleanup_share = self._avg(
            [
                (item.cleanup_seconds / item.full_day_seconds)
                for item in window
                if item.full_day_seconds > 0
            ]
        )
        crop_median = self._avg([item.timing_crop_seconds_median for item in window if item.timing_crop_seconds_median is not None])
        crop_p95 = self._avg([item.timing_crop_seconds_p95 for item in window if item.timing_crop_seconds_p95 is not None])
        extract_median = self._avg(
            [
                self._extract_total_median(item)
                for item in window
                if self._extract_total_median(item) is not None
            ]
        )
        extract_p95 = self._avg(
            [
                self._extract_total_p95(item)
                for item in window
                if self._extract_total_p95(item) is not None
            ]
        )

        cleanup_dominates = cleanup_share is not None and cleanup_share > 0.20
        crop_dominates = crop_median is not None and extract_median is not None and crop_median > (extract_median * 1.25)
        extract_dominates = crop_median is not None and extract_median is not None and extract_median > (crop_median * 1.25)
        stable_to_increase = self._failures_since_tune == 0 and cleanup_share is not None and cleanup_share < 0.10

        crop_oversaturated = bool(
            crop_p95 is not None
            and self._stable_baseline is not None
            and self._stable_baseline.get("crop_median") is not None
            and self._stable_baseline["crop_median"] > 0
            and crop_p95 > (self._stable_baseline["crop_median"] * 1.75)
        )
        extract_oversaturated = bool(
            extract_p95 is not None
            and self._stable_baseline is not None
            and self._stable_baseline.get("extract_median") is not None
            and self._stable_baseline["extract_median"] > 0
            and extract_p95 > (self._stable_baseline["extract_median"] * 1.75)
        )

        if cleanup_dominates and self._plan.day_workers > self.bounds.day_workers_min:
            self._apply_change("day_workers", self._plan.day_workers - 1, reason="cleanup_share_high")
            return True, self.tune_reason
        if extract_dominates and extract_oversaturated and self._plan.extract_workers is not None and self._plan.extract_workers > self.bounds.extract_workers_min:
            self._apply_change("extract_workers", self._plan.extract_workers - 1, reason="extract_p95_high")
            return True, self.tune_reason
        if crop_dominates and crop_oversaturated and self._plan.reduce_workers is not None and self._plan.reduce_workers > self.bounds.reduce_workers_min:
            self._apply_change("reduce_workers", self._plan.reduce_workers - 1, reason="crop_p95_high")
            return True, self.tune_reason
        if stable_to_increase and not cleanup_dominates and not crop_dominates and not extract_dominates and self._plan.day_workers < self.bounds.day_workers_max:
            self._stable_baseline = {"crop_median": crop_median or 0.0, "extract_median": extract_median or 0.0}
            self._apply_change("day_workers", self._plan.day_workers + 1, reason="cleanup_ok_extract_ok")
            return True, self.tune_reason
        if self._plan.day_workers >= self.bounds.day_workers_max and crop_dominates and not crop_oversaturated and self._plan.reduce_workers is not None and self._plan.reduce_workers < self.bounds.reduce_workers_max:
            self._apply_change("reduce_workers", self._plan.reduce_workers + 1, reason="crop_dominant_room_to_scale")
            return True, self.tune_reason
        if self._plan.day_workers >= self.bounds.day_workers_max and extract_dominates and not extract_oversaturated and self._plan.extract_workers is not None and self._plan.extract_workers < self.bounds.extract_workers_max:
            self._apply_change("extract_workers", self._plan.extract_workers + 1, reason="extract_dominant_room_to_scale")
            return True, self.tune_reason
        if (
            self._plan.day_workers >= self.bounds.day_workers_max
            and self._plan.reduce_workers is not None
            and self._plan.extract_workers is not None
            and self._plan.reduce_workers >= self.bounds.reduce_workers_max
            and self._plan.extract_workers >= self.bounds.extract_workers_max
            and stable_to_increase
            and self._plan.lead_workers < self.bounds.lead_workers_max
        ):
            self._apply_change("lead_workers", self._plan.lead_workers + 1, reason="stage_caps_reached_scale_leads")
            return True, self.tune_reason
        if (
            (crop_oversaturated or extract_oversaturated or self._failures_since_tune > 0)
            and self._plan.reduce_workers is not None
            and self._plan.extract_workers is not None
            and self._plan.reduce_workers <= self.bounds.reduce_workers_min
            and self._plan.extract_workers <= self.bounds.extract_workers_min
            and self._plan.lead_workers > self.bounds.lead_workers_min
        ):
            self._apply_change("lead_workers", self._plan.lead_workers - 1, reason="over_saturated_reduce_leads")
            return True, self.tune_reason

        if stable_to_increase:
            self._stable_baseline = {"crop_median": crop_median or 0.0, "extract_median": extract_median or 0.0}
        self.tune_reason = "window_stable_no_change"
        self.last_tune = "no_change"
        self._failures_since_tune = 0
        return False, self.tune_reason

    def _apply_change(self, field_name: str, new_value: int, *, reason: str) -> None:
        current = getattr(self._plan, field_name)
        if current == new_value:
            self.last_tune = "no_change"
            self.tune_reason = f"{reason}_unchanged"
            return
        updates = {
            "day_workers": self._plan.day_workers,
            "workers": self._plan.workers,
            "lead_workers": self._plan.lead_workers,
            "download_workers": self._plan.download_workers,
            "reduce_workers": self._plan.reduce_workers,
            "extract_workers": self._plan.extract_workers,
            "reduce_queue_size": self._plan.reduce_queue_size,
            "extract_queue_size": self._plan.extract_queue_size,
        }
        updates[field_name] = new_value
        next_plan = WorkerPlan(**updates)
        if field_name == "lead_workers":
            lead_workers = max(self.bounds.lead_workers_min, int(new_value))
            next_plan = WorkerPlan(
                day_workers=next_plan.day_workers,
                workers=next_plan.workers,
                lead_workers=lead_workers,
                download_workers=next_plan.download_workers,
                reduce_workers=next_plan.reduce_workers or lead_workers,
                extract_workers=next_plan.extract_workers or lead_workers,
                reduce_queue_size=next_plan.reduce_queue_size,
                extract_queue_size=next_plan.extract_queue_size,
            )
        resolved = resolved_worker_plan(next_plan)
        resolved = WorkerPlan(
            day_workers=max(self.bounds.day_workers_min, resolved.day_workers),
            workers=resolved.workers,
            lead_workers=max(self.bounds.lead_workers_min, resolved.lead_workers),
            download_workers=self._plan.download_workers,
            reduce_workers=max(self.bounds.reduce_workers_min, resolved.reduce_workers or self.bounds.reduce_workers_min),
            extract_workers=max(self.bounds.extract_workers_min, resolved.extract_workers or self.bounds.extract_workers_min),
            reduce_queue_size=(
                self._plan.reduce_queue_size
                if self._preserve_reduce_queue_size
                else queue_size_for_workers(max(self.bounds.reduce_workers_min, resolved.reduce_workers or self.bounds.reduce_workers_min))
            ),
            extract_queue_size=(
                self._plan.extract_queue_size
                if self._preserve_extract_queue_size
                else queue_size_for_workers(max(self.bounds.extract_workers_min, resolved.extract_workers or self.bounds.extract_workers_min))
            ),
        )
        self._plan = resolved
        self._cooldown_remaining = self.cooldown_days
        self._successful_days_since_tune = 0
        self._failures_since_tune = 0
        self.last_tune = f"{field_name}:{current}->{new_value}"
        self.tune_reason = reason

    @staticmethod
    def _avg(values: list[float | None]) -> float | None:
        filtered = [float(value) for value in values if value is not None]
        if not filtered:
            return None
        return sum(filtered) / len(filtered)

    @staticmethod
    def _extract_total_median(performance: DayPerformance) -> float | None:
        parts = [
            performance.timing_cfgrib_open_seconds_median,
            performance.timing_row_metric_seconds_median,
            performance.timing_row_provenance_seconds_median,
        ]
        filtered = [float(value) for value in parts if value is not None]
        if not filtered:
            return None
        return sum(filtered)

    @staticmethod
    def _extract_total_p95(performance: DayPerformance) -> float | None:
        parts = [
            performance.timing_cfgrib_open_seconds_p95,
            performance.timing_row_metric_seconds_p95,
            performance.timing_row_provenance_seconds_p95,
        ]
        filtered = [float(value) for value in parts if value is not None]
        if not filtered:
            return None
        return sum(filtered)

def effective_progress_mode(args: argparse.Namespace) -> str:
    requested = str(getattr(args, "progress_mode", "auto"))
    if int(getattr(args, "day_workers", DEFAULT_DAY_WORKERS)) > 1 and requested in {"auto", "dashboard"}:
        return "log"
    return requested


def child_progress_mode(args: argparse.Namespace, *, parent_dashboard_active: bool) -> str:
    if parent_dashboard_active:
        return "log"
    return effective_progress_mode(args)


def parse_progress_line(line: str) -> tuple[str, dict[str, str]] | None:
    stripped = line.strip()
    if not stripped.startswith(f"{PROGRESS_PREFIX} "):
        return None
    tokens = stripped.split()
    event_index = next((index for index, token in enumerate(tokens) if token.startswith("event=")), None)
    if event_index is None:
        return None
    event_name = tokens[event_index].split("=", 1)[1]
    payload: dict[str, str] = {}
    current_key: str | None = None
    current_parts: list[str] = []
    for token in tokens[event_index + 1 :]:
        match = PROGRESS_KV_RE.match(token)
        if match:
            if current_key is not None:
                payload[current_key] = " ".join(current_parts)
            current_key = match.group("key")
            current_parts = [match.group("value")]
            continue
        if current_key is not None:
            current_parts.append(token)
    if current_key is not None:
        payload[current_key] = " ".join(current_parts)
    return event_name, payload


def int_or_none(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def float_or_none(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_child_output_level(*, stream_name: str, line: str) -> str:
    if stream_name != "stderr":
        return "info"
    lowered = line.lower()
    if "futurewarning:" in lowered or "userwarning:" in lowered or "runtimewarning:" in lowered:
        return "warn"
    if "warning:" in lowered and "/site-packages/" in lowered:
        return "warn"
    if line.strip().startswith("o = xr.merge("):
        return "warn"
    if is_noisy_cfgrib_grouping_line(line):
        return "info"
    return "error"


def is_noisy_cfgrib_grouping_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    noisy_fragments = (
        "cfgrib.dataset.datasetbuilderror",
        "key present and new value is different",
        "dict_merge(variables, coord_vars)",
        "raise datasetbuilderror",
        "skipping variable:",
        "filter_by_keys",
    )
    if any(fragment in lowered for fragment in noisy_fragments):
        return True
    if "/cfgrib/" in lowered or "\\cfgrib\\" in lowered:
        return True
    if "site-packages/cfgrib" in lowered or "site-packages\\cfgrib" in lowered:
        return True
    return False


def stream_command(
    command: list[str],
    *,
    stdout_handler=None,
    stderr_handler=None,
) -> None:
    emit_runner_log("RUN", " ".join(command))
    kwargs = {
        "cwd": REPO_ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "bufsize": 1,
    }
    try:
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
    except TypeError as exc:
        if "start_new_session" not in str(exc):
            raise
        process = subprocess.Popen(command, **kwargs)

    def _pump(stream, handler) -> None:
        if stream is None:
            return
        try:
            for raw_line in iter(stream.readline, ""):
                line = raw_line.rstrip("\r\n")
                if handler is not None:
                    handler(line)
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    import contextlib

    threads = [
        threading.Thread(target=_pump, args=(process.stdout, stdout_handler), daemon=True),
        threading.Thread(target=_pump, args=(process.stderr, stderr_handler), daemon=True),
    ]
    for thread in threads:
        thread.start()
    return_code = process.wait()
    for thread in threads:
        thread.join(timeout=1.0)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


class DayProgressBridge:
    def __init__(self, reporter, *, day_workers: int, run_root: Path) -> None:
        self.reporter = reporter
        self.day_workers = max(1, int(day_workers))
        self.run_root = run_root
        self._lock = threading.Lock()
        self._days: dict[str, DayProgressState] = {}
        self._extra_metrics: dict[str, object] = {}

    def add_skipped(self, count: int, *, message: str, affects_eta: bool = False) -> None:
        if count <= 0:
            return
        self.reporter.add_skipped(count, message=message, affects_eta=affects_eta)
        self._refresh_metrics()

    def admit_day(self, token: str) -> None:
        with self._lock:
            now = time.perf_counter()
            self._days[token] = DayProgressState(
                lifecycle_phase="queued",
                details="admitted",
                target_date=token,
                started_at=now,
                status="queued",
            )
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=1, status="queued")
        self._sync_batch_day(token)
        self._refresh_metrics()

    def update_day(self, token: str, *, lifecycle_phase: str | None = None, details: str | None = None) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            if lifecycle_phase is not None:
                state.lifecycle_phase = lifecycle_phase
                state.status = self._status_for_phase(lifecycle_phase, details or state.details)
            if details is not None:
                state.details = details
            self._apply_pipeline_status(state, lifecycle_phase=state.lifecycle_phase, details=state.details)
            group_status = f"{state.lifecycle_phase}:{state.details}" if state.details else state.lifecycle_phase
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=1, status=group_status)
        self._sync_batch_day(token)
        self._refresh_metrics()

    def set_day_phase_worker(self, token: str, *, phase: str, details: str) -> None:
        worker_id = self._day_worker_id(token)
        if worker_id in self.reporter.state.workers and self.reporter.state.workers[worker_id].active:
            self.reporter.update_worker(worker_id, phase=phase, details=details, label=token)
        else:
            self.reporter.start_worker(worker_id, label=token, phase=phase, group_id=token, details=details)

    def retire_day_phase_worker(self, token: str, *, message: str | None = None) -> None:
        self.reporter.retire_worker(self._day_worker_id(token), message=message)

    def start_child_worker(self, token: str, *, child_worker: str, label: str, phase: str, details: str | None = None) -> None:
        self._record_child_progress(token, child_worker=child_worker, phase=phase, label=label, details=details)
        self.reporter.start_worker(
            self._child_worker_id(token, child_worker),
            label=label,
            phase=phase,
            group_id=token,
            details=details,
        )
        self._sync_batch_day(token)

    def update_child_worker(
        self,
        token: str,
        *,
        child_worker: str,
        phase: str | None = None,
        label: str | None = None,
        details: str | None = None,
    ) -> None:
        self._record_child_progress(token, child_worker=child_worker, phase=phase, label=label, details=details)
        self.reporter.update_worker(self._child_worker_id(token, child_worker), phase=phase, label=label, details=details)
        self._sync_batch_day(token)

    def set_child_attempt(self, token: str, *, child_worker: str, attempt: int, max_attempts: int) -> None:
        self.reporter.set_worker_attempt(self._child_worker_id(token, child_worker), attempt=attempt, max_attempts=max_attempts)

    def start_transfer(self, token: str, *, child_worker: str, file_label: str, total_bytes: int | None = None) -> None:
        self.reporter.start_transfer(self._child_worker_id(token, child_worker), file_label=file_label, total_bytes=total_bytes)

    def update_transfer(self, token: str, *, child_worker: str, bytes_downloaded: int, total_bytes: int | None = None) -> None:
        self.reporter.update_transfer(
            self._child_worker_id(token, child_worker),
            bytes_downloaded=bytes_downloaded,
            total_bytes=total_bytes,
        )

    def finish_transfer(self, token: str, *, child_worker: str) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            assert state.downloaded_children is not None
            state.downloaded_children.add(self._download_progress_key(state, child_worker))
            state.lifecycle_phase = "download"
            state.status = "download"
        self.reporter.finish_transfer(self._child_worker_id(token, child_worker))
        self._sync_batch_day(token)

    def schedule_retry(
        self,
        token: str,
        *,
        child_worker: str,
        attempt: int,
        max_attempts: int,
        delay_seconds: float,
        message: str,
        error_class: str,
    ) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            state.retry_count += 1
            assert state.child_phases is not None
            state.child_phases[child_worker] = "retry_wait"
        self.reporter.schedule_retry(
            self._child_worker_id(token, child_worker),
            attempt=attempt,
            max_attempts=max_attempts,
            delay_seconds=delay_seconds,
            message=message,
            error_class=error_class,
        )
        self._sync_batch_day(token)

    def start_retry(self, token: str, *, child_worker: str, attempt: int, max_attempts: int) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            assert state.child_phases is not None
            state.child_phases[child_worker] = "retrying"
        self.reporter.start_retry(self._child_worker_id(token, child_worker), attempt=attempt, max_attempts=max_attempts)
        self._sync_batch_day(token)

    def recover_retry(self, token: str, *, child_worker: str, message: str) -> None:
        self.reporter.recover_worker(self._child_worker_id(token, child_worker), message=message)

    def retire_child_worker(self, token: str, *, child_worker: str, message: str | None = None) -> None:
        with self._lock:
            state = self._days.get(token)
            if state is not None and state.child_phases is not None:
                phase = state.child_phases.pop(child_worker, None)
                if self._map_child_phase(phase=phase, details=None) == "download":
                    assert state.downloaded_children is not None
                    state.downloaded_children.add(self._download_progress_key(state, child_worker))
                if state.child_labels is not None:
                    state.child_labels.pop(child_worker, None)
        self.reporter.retire_worker(self._child_worker_id(token, child_worker), message=message)
        self._sync_batch_day(token)

    def retire_child_workers_for_day(self, token: str) -> None:
        prefix = f"{token}/"
        for worker_id in list(self.reporter.state.workers):
            if worker_id.startswith(prefix):
                self.reporter.retire_worker(worker_id)
        with self._lock:
            state = self._days.get(token)
            if state is not None and state.child_phases is not None:
                state.child_phases.clear()
                if state.child_labels is not None:
                    state.child_labels.clear()
        self._sync_batch_day(token)

    def log_event(self, token: str, *, message: str, level: str = "info") -> None:
        self.reporter.log_event(f"{token} {message}", level=level)

    def update_day_group_status(self, token: str, *, status: str) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            state.details = status
            self._apply_pipeline_status(state, lifecycle_phase=state.lifecycle_phase, details=status)
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=1, status=status)
        self._sync_batch_day(token)
        self._refresh_metrics()

    def complete_day(self, token: str, *, performance: DayPerformance | None = None) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            state.lifecycle_phase = "complete"
            state.status = "complete"
            state.batch_status = "done" if state.batch_status != "err" else state.batch_status
            state.extract_status = "done" if state.extract_status != "err" else state.extract_status
            state.overnight_status = "done"
            state.ended_at = time.perf_counter()
        self.retire_day_phase_worker(token, message=f"{token} done")
        self.reporter.upsert_group(token, label=token, total=1, completed=1, failed=0, active=0, status="complete")
        if performance is not None:
            self.reporter.record_batch_timing(**asdict(performance))
        self._sync_batch_day(token)
        with self._lock:
            self._days.pop(token, None)
        self.reporter.record_outcome("completed", message=f"{token} done")
        self._refresh_metrics()

    def fail_day(self, token: str, *, message: str) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            state.lifecycle_phase = "failed"
            state.status = "failed"
            state.batch_status = "err" if state.batch_status not in {"done", "queued"} else state.batch_status
            state.extract_status = "err" if state.extract_status not in {"done", "queued"} else state.extract_status
            state.ended_at = time.perf_counter()
        self.retire_day_phase_worker(token, message=f"{token} {message}")
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=1, active=0, status=message)
        self._sync_batch_day(token)
        with self._lock:
            self._days.pop(token, None)
        self.reporter.record_outcome("failed", message=f"{token} {message}")
        self._refresh_metrics()

    def retire_day(self, token: str, *, message: str | None = None) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            state.lifecycle_phase = "skipped"
            state.status = "skipped"
            state.ended_at = time.perf_counter()
        self.retire_day_phase_worker(token, message=message)
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=0, status=message or "retired")
        self._sync_batch_day(token)
        with self._lock:
            self._days.pop(token, None)
        self._refresh_metrics()

    def refresh_disk(self) -> None:
        self._refresh_metrics()

    def set_plan_metrics(self, metrics: dict[str, object]) -> None:
        with self._lock:
            self._extra_metrics = dict(metrics)
            plan_days = metrics.get("PlanDays")
            if plan_days is not None:
                self.day_workers = max(1, int(plan_days))
        self._refresh_metrics()

    def mark_paused(self, *, reason: str | None = None) -> None:
        self.reporter.mark_paused(reason=reason)
        self._refresh_metrics()

    def close(self, *, status: str) -> None:
        self.reporter.close(status=status)

    def _refresh_metrics(self) -> None:
        with self._lock:
            counts = {"Raw": 0, "Overnight": 0, "Validate": 0, "Cleanup": 0}
            for state in self._days.values():
                key = state.lifecycle_phase.capitalize()
                if key in counts:
                    counts[key] += 1
            active_days = len(self._days)
            free_gb = free_disk_gb(self.run_root)
        total_days = self.reporter.state.total
        queued_days = None if total_days is None else max(0, int(total_days) - self.reporter.state.completed_total - active_days)
        self.reporter.set_metrics(
            DayWorkers=self.day_workers,
            ActiveDays=active_days,
            QueuedDays=queued_days,
            FreeGB=f"{free_gb:.1f}",
            Raw=counts["Raw"],
            Overnight=counts["Overnight"],
            Validate=counts["Validate"],
            Cleanup=counts["Cleanup"],
            **self._extra_metrics,
        )

    def set_day_expected_leads(self, token: str, *, issue: str | None = None, expected_leads: int | None = None) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            if issue:
                state.selected_issue = issue
            if expected_leads is not None:
                state.expected_leads = max(0, int(expected_leads))
        self._sync_batch_day(token)

    def _record_child_progress(
        self,
        token: str,
        *,
        child_worker: str,
        phase: str | None,
        label: str | None = None,
        details: str | None = None,
    ) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(target_date=token))
            if label:
                issue = self._issue_from_label(label)
                if issue is not None:
                    state.selected_issue = issue
            mapped_phase = self._map_child_phase(phase=phase, details=details)
            if mapped_phase is not None:
                state.lifecycle_phase = mapped_phase
                state.status = self._status_for_phase(mapped_phase, details)
            if phase is not None and state.child_phases is not None:
                state.child_phases[child_worker] = phase
            if label is not None and state.child_labels is not None:
                state.child_labels[child_worker] = label
            self._apply_pipeline_status(state, lifecycle_phase=state.lifecycle_phase, details=details or phase or "")

    def _sync_batch_day(self, token: str) -> None:
        with self._lock:
            state = self._days.get(token)
            if state is None:
                return
            phase_counts: dict[str, int] = {}
            for phase in (state.child_phases or {}).values():
                mapped = self._map_child_phase(phase=phase, details=None) or phase or "raw"
                phase_counts[mapped] = phase_counts.get(mapped, 0) + 1
            downloaded_leads = len(state.downloaded_children or set())
            elapsed_seconds = None
            if state.started_at is not None:
                elapsed_seconds = (state.ended_at or time.perf_counter()) - state.started_at
            payload = {
                "lifecycle_phase": state.lifecycle_phase,
                "selected_issue": state.selected_issue,
                "downloaded_leads": downloaded_leads,
                "expected_leads": state.expected_leads,
                "retry_count": state.retry_count,
                "active_child_phase_counts": phase_counts,
                "batch_status": state.batch_status,
                "extract_status": state.extract_status,
                "overnight_status": state.overnight_status,
                "status": state.status,
                "started_at": state.started_at,
                "ended_at": state.ended_at,
                "elapsed_seconds": elapsed_seconds,
            }
        self.reporter.upsert_batch_day(token, **payload)

    @staticmethod
    def _issue_from_label(label: str) -> str | None:
        match = re.search(r"\b(\d{8}T\d{4}Z)\b", label)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _download_progress_key(state: DayProgressState, child_worker: str) -> str:
        label = (state.child_labels or {}).get(child_worker)
        if label:
            return label
        return child_worker

    @staticmethod
    def _map_child_phase(*, phase: str | None, details: str | None) -> str | None:
        text = " ".join(part for part in [phase or "", details or ""] if part).lower()
        if not text:
            return None
        if any(token in text for token in ("download", "idx", "parse_idx", "byte_range", "fetch_idx", "reuse_raw")):
            return "download"
        if any(token in text for token in ("batch_crop", "crop_selected", "reduce", "crop")):
            return "batch_reduce"
        if any(token in text for token in ("batch_open", "batch_build", "open_grouped", "build_rows", "extract")):
            return "batch_extract"
        if any(token in text for token in ("finalize", "batch_cleanup")):
            return "batch_extract"
        return "raw"

    @staticmethod
    def _status_for_phase(lifecycle_phase: str, details: str | None) -> str:
        if lifecycle_phase == "batch_reduce":
            return "batch"
        if lifecycle_phase == "batch_extract":
            return "extract"
        if lifecycle_phase == "download":
            return "download"
        if lifecycle_phase in {"overnight", "validate", "cleanup", "complete", "failed", "queued"}:
            return lifecycle_phase
        return details or lifecycle_phase

    @staticmethod
    def _apply_pipeline_status(state: DayProgressState, *, lifecycle_phase: str, details: str | None) -> None:
        detail_text = (details or "").lower()
        if lifecycle_phase == "download":
            state.batch_status = "queued" if state.batch_status == "queued" else state.batch_status
            state.extract_status = "queued" if state.extract_status == "queued" else state.extract_status
        elif lifecycle_phase == "batch_reduce":
            if "concat" in detail_text:
                state.batch_status = "concat"
            elif "batch_crop" in detail_text or "crop" in detail_text:
                state.batch_status = "crop"
            elif "reuse" in detail_text:
                state.batch_status = "reuse"
            else:
                state.batch_status = "queued"
        elif lifecycle_phase == "batch_extract":
            if "batch_open" in detail_text or "open_grouped" in detail_text:
                state.extract_status = "cfgrib"
            elif "batch_build" in detail_text or "build_rows" in detail_text:
                state.extract_status = "rows"
            elif "cleanup" in detail_text or "finalize" in detail_text:
                state.extract_status = "split"
            else:
                state.extract_status = "queued"
            if state.batch_status not in {"err", "queued"}:
                state.batch_status = "done"
        elif lifecycle_phase == "overnight":
            state.batch_status = "done" if state.batch_status != "err" else state.batch_status
            state.extract_status = "done" if state.extract_status != "err" else state.extract_status
            state.overnight_status = "run"
        elif lifecycle_phase == "validate":
            state.overnight_status = "done"
        elif lifecycle_phase == "cleanup":
            state.overnight_status = "done"

    @staticmethod
    def _child_worker_id(token: str, child_worker: str) -> str:
        return f"{token}/{child_worker}"

    @staticmethod
    def _day_worker_id(token: str) -> str:
        return f"day::{token}"


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def resolve_run_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def day_temp_root(run_root: Path, target_date_local: str) -> Path:
    return run_root / "nbm_tmp" / target_date_local


def day_raw_root(run_root: Path, target_date_local: str) -> Path:
    return day_temp_root(run_root, target_date_local) / "raw"


def day_scratch_root(run_root: Path, target_date_local: str) -> Path:
    return day_temp_root(run_root, target_date_local) / "scratch"


def nbm_day_output_root(run_root: Path, day: DayWindow) -> Path:
    return run_root / "nbm_overnight" / f"target_date_local={day_token(day)}"


def nbm_day_metadata_path(run_root: Path, day: DayWindow) -> Path:
    return nbm_day_output_root(run_root, day) / "nbm.resume.json"


def nbm_day_performance_path(run_root: Path, day: DayWindow) -> Path:
    return nbm_day_output_root(run_root, day) / "nbm.performance.json"


def write_nbm_day_metadata(run_root: Path, day: DayWindow, *, selection_mode: str) -> None:
    metadata_path = nbm_day_metadata_path(run_root, day)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "target_date_local": day_token(day),
                "selection_mode": selection_mode,
            },
            indent=2,
            sort_keys=True,
        )
    )


def write_nbm_day_performance(run_root: Path, day: DayWindow, performance: DayPerformance) -> None:
    performance_path = nbm_day_performance_path(run_root, day)
    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.write_text(json.dumps(asdict(performance), indent=2, sort_keys=True))


def validate_nbm_day(run_root: Path, day: DayWindow, *, selection_mode: str) -> bool:
    root = nbm_day_output_root(run_root, day)
    output_path = root / "nbm.overnight.parquet"
    manifest_path = root / "nbm.overnight.manifest.parquet"
    metadata_path = nbm_day_metadata_path(run_root, day)
    if not output_path.exists() or not manifest_path.exists() or not metadata_path.exists():
        return False
    try:
        manifest_df = pd.read_parquet(manifest_path)
        pd.read_parquet(output_path)
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False
    if manifest_df.empty:
        return False
    if str(metadata.get("target_date_local")) != day_token(day):
        return False
    if str(metadata.get("selection_mode")) != selection_mode:
        return False
    status = str(manifest_df.iloc[0].get("status", ""))
    return status in {"ok", "no_qualifying_issue"}


def build_raw_command(
    args: argparse.Namespace,
    *,
    target_date_local: str,
    raw_dir: Path,
    scratch_dir: Path,
    worker_plan: WorkerPlan | None = None,
    progress_mode: str | None = None,
    include_pause_control_file: bool = False,
) -> list[str]:
    resolved_plan = resolved_worker_plan(worker_plan or initial_worker_plan(args))
    resolved_progress_mode = progress_mode
    if resolved_progress_mode is None:
        requested_mode = str(getattr(args, "progress_mode", "auto"))
        if resolved_plan.day_workers > 1 and requested_mode in {"auto", "dashboard"}:
            resolved_progress_mode = "log"
        else:
            resolved_progress_mode = effective_progress_mode(args)
    command = [
        sys.executable,
        "tools/nbm/build_grib2_features.py",
        "--start-local-date",
        target_date_local,
        "--end-local-date",
        target_date_local,
        "--selection-mode",
        str(args.selection_mode),
        "--output-dir",
        str(raw_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--workers",
        str(resolved_plan.workers),
        "--lead-workers",
        str(resolved_plan.lead_workers),
        "--progress-mode",
        resolved_progress_mode,
        "--metric-profile",
        str(getattr(args, "metric_profile", "overnight")),
        "--batch-reduce-mode",
        str(getattr(args, "batch_reduce_mode", "off")),
    ]
    if resolved_plan.download_workers is not None:
        command.extend(["--download-workers", str(resolved_plan.download_workers)])
    if resolved_plan.reduce_workers is not None:
        command.extend(["--reduce-workers", str(resolved_plan.reduce_workers)])
    if resolved_plan.extract_workers is not None:
        command.extend(["--extract-workers", str(resolved_plan.extract_workers)])
    if resolved_plan.reduce_queue_size is not None:
        command.extend(["--reduce-queue-size", str(resolved_plan.reduce_queue_size)])
    if resolved_plan.extract_queue_size is not None:
        command.extend(["--extract-queue-size", str(resolved_plan.extract_queue_size)])
    crop_method = getattr(args, "crop_method", None)
    if crop_method:
        command.extend(["--crop-method", str(crop_method)])
    wgrib2_threads = getattr(args, "wgrib2_threads", None)
    if wgrib2_threads is not None:
        command.extend(["--wgrib2-threads", str(wgrib2_threads)])
    crop_grib_type = getattr(args, "crop_grib_type", None)
    if crop_grib_type:
        command.extend(["--crop-grib-type", str(crop_grib_type)])
    if getattr(args, "keep_reduced", False):
        command.append("--keep-reduced")
    if getattr(args, "overnight_fast", False):
        command.append("--skip-provenance")
    if args.max_task_attempts is not None:
        command.extend(["--max-task-attempts", str(args.max_task_attempts)])
    if args.retry_backoff_seconds is not None:
        command.extend(["--retry-backoff-seconds", str(args.retry_backoff_seconds)])
    if args.retry_max_backoff_seconds is not None:
        command.extend(["--retry-max-backoff-seconds", str(args.retry_max_backoff_seconds)])
    if include_pause_control_file and args.pause_control_file is not None:
        command.extend(["--pause-control-file", str(args.pause_control_file)])
    if args.overwrite:
        command.append("--overwrite")
    return command


def build_overnight_command(args: argparse.Namespace, *, target_date_local: str, raw_dir: Path, overnight_root: Path) -> list[str]:
    return [
        sys.executable,
        "tools/nbm/build_nbm_overnight_features.py",
        "--features-root",
        str(raw_dir),
        "--output-dir",
        str(overnight_root),
        "--start-local-date",
        target_date_local,
        "--end-local-date",
        target_date_local,
    ]


def build_overnight_in_process(
    *,
    target_date_local: str,
    raw_dir: Path,
    overnight_root: Path,
) -> list[Path]:
    target_date = parse_local_date(target_date_local)
    return build_nbm_overnight_features.build_range(
        features_root=raw_dir,
        output_dir=overnight_root,
        start_local_date=target_date,
        end_local_date=target_date,
        cutoff_local_time=build_nbm_overnight_features.parse_local_time(
            build_nbm_overnight_features.DEFAULT_CUTOFF_LOCAL_TIME
        ),
        progress=None,
    )


def relay_child_progress_line(bridge: DayProgressBridge, token: str, line: str, *, stream_name: str) -> None:
    parsed = parse_progress_line(line)
    if parsed is None:
        if line.startswith(("download_start ", "download_progress ", "download_complete ")):
            return
        if not line.strip():
            return
        level = classify_child_output_level(stream_name=stream_name, line=line)
        bridge.log_event(token, message=f"{stream_name}: {line}", level=level)
        return
    event_name, payload = parsed
    if event_name == "worker_start":
        child_worker = payload.get("worker") or "worker"
        child_phase = payload.get("phase") or "raw"
        lifecycle_phase = bridge._map_child_phase(phase=child_phase, details=payload.get("details")) or "raw"
        bridge.update_day(token, lifecycle_phase=lifecycle_phase, details=payload.get("details") or payload.get("label") or child_phase)
        bridge.start_child_worker(
            token,
            child_worker=child_worker,
            label=payload.get("label") or child_worker,
            phase=child_phase,
            details=payload.get("details"),
        )
        return
    if event_name == "worker_update":
        child_worker = payload.get("worker") or "worker"
        child_phase = payload.get("phase") or "raw"
        lifecycle_phase = bridge._map_child_phase(phase=child_phase, details=payload.get("details")) or "raw"
        bridge.update_day(token, lifecycle_phase=lifecycle_phase, details=payload.get("details") or payload.get("label") or child_phase)
        bridge.update_child_worker(
            token,
            child_worker=child_worker,
            phase=child_phase,
            label=payload.get("label"),
            details=payload.get("details"),
        )
        return
    if event_name == "transfer_start":
        child_worker = payload.get("worker") or "worker"
        bridge.start_transfer(
            token,
            child_worker=child_worker,
            file_label=payload.get("file") or "transfer",
            total_bytes=int_or_none(payload.get("total_bytes")),
        )
        return
    if event_name == "transfer_progress":
        child_worker = payload.get("worker") or "worker"
        bridge.update_day(token, lifecycle_phase="download", details=payload.get("file") or "download")
        bridge.update_child_worker(token, child_worker=child_worker, phase="download", details=payload.get("file") or "download")
        bridge.update_transfer(
            token,
            child_worker=child_worker,
            bytes_downloaded=int_or_none(payload.get("bytes_downloaded")) or 0,
            total_bytes=int_or_none(payload.get("total_bytes")),
        )
        return
    if event_name == "transfer_complete":
        child_worker = payload.get("worker") or "worker"
        bridge.finish_transfer(token, child_worker=child_worker)
        bridge.update_day(token, lifecycle_phase="download", details=payload.get("file") or "download_complete")
        bridge.update_child_worker(token, child_worker=child_worker, phase="download", details=payload.get("file") or "download_complete")
        return
    if event_name == "retry_scheduled":
        bridge.schedule_retry(
            token,
            child_worker=payload.get("worker") or "worker",
            attempt=int_or_none(payload.get("attempt")) or 1,
            max_attempts=int_or_none(payload.get("max_attempts")) or 1,
            delay_seconds=float_or_none(payload.get("delay_seconds")) or 0.0,
            message=payload.get("message") or "retry_scheduled",
            error_class=payload.get("error_class") or "retry",
        )
        return
    if event_name == "retry_started":
        bridge.start_retry(
            token,
            child_worker=payload.get("worker") or "worker",
            attempt=int_or_none(payload.get("attempt")) or 1,
            max_attempts=int_or_none(payload.get("max_attempts")) or 1,
        )
        bridge.update_day(token, lifecycle_phase="download", details="retrying")
        return
    if event_name == "retry_recovered":
        bridge.recover_retry(
            token,
            child_worker=payload.get("worker") or "worker",
            message=payload.get("message") or f"{token} retry recovered",
        )
        bridge.update_day(token, lifecycle_phase="download", details="recovered")
        return
    if event_name == "worker_attempt":
        child_worker = payload.get("worker") or "worker"
        bridge.set_child_attempt(
            token,
            child_worker=child_worker,
            attempt=int_or_none(payload.get("attempt")) or 1,
            max_attempts=int_or_none(payload.get("max_attempts")) or 1,
        )
        bridge.update_day(
            token,
            lifecycle_phase="download",
            details=f"attempt a{int_or_none(payload.get('attempt')) or 1}/{int_or_none(payload.get('max_attempts')) or 1}",
        )
        return
    if event_name == "group":
        status = payload.get("status") or "raw"
        issue = payload.get("group") or payload.get("label")
        bridge.set_day_expected_leads(token, issue=issue, expected_leads=int_or_none(payload.get("total")))
        bridge.update_day(token, lifecycle_phase="raw", details=status)
        bridge.update_day_group_status(token, status=f"raw:{status}")
        return
    if event_name in {"worker_retire", "worker_complete"}:
        child_worker = payload.get("worker") or "worker"
        bridge.retire_child_worker(token, child_worker=child_worker, message=payload.get("message"))
        outcome = payload.get("outcome")
        message = payload.get("message")
        if outcome in {"failed", "skipped"} and message:
            bridge.log_event(token, message=message, level="error" if outcome == "failed" else "warn")
        return
    if event_name in {"message", "outcome"}:
        message = payload.get("message")
        if message:
            bridge.log_event(token, message=message, level=payload.get("level") or "info")


def process_day(
    args: argparse.Namespace,
    day: DayWindow,
    *,
    bridge: DayProgressBridge | None = None,
    worker_plan: WorkerPlan | None = None,
) -> DayRunResult:
    token = day_token(day)
    output_root = nbm_day_output_root(args.run_root, day)
    tmp_root = day_temp_root(args.run_root, token)
    raw_dir = day_raw_root(args.run_root, token)
    scratch_dir = day_scratch_root(args.run_root, token)
    resolved_plan = resolved_worker_plan(worker_plan or initial_worker_plan(args))
    day_started_at = time.perf_counter()

    delete_path(output_root)
    if getattr(args, "keep_reduced", False):
        delete_path(raw_dir)
    else:
        delete_path(tmp_root)

    emit_runner_log("start", f"nbm date={token}")
    try:
        raw_started_at = time.perf_counter()
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="raw", details="start_raw")
            bridge.log_event(token, message=(
                f"plan d={resolved_plan.day_workers} l={resolved_plan.lead_workers} "
                f"r={resolved_plan.reduce_workers} e={resolved_plan.extract_workers}"
            ))
            stream_command(
                build_raw_command(
                    args,
                    target_date_local=token,
                    raw_dir=raw_dir,
                    scratch_dir=scratch_dir,
                    worker_plan=resolved_plan,
                    progress_mode=child_progress_mode(args, parent_dashboard_active=True),
                    include_pause_control_file=False,
                ),
                stdout_handler=lambda line: relay_child_progress_line(bridge, token, line, stream_name="stdout"),
                stderr_handler=lambda line: relay_child_progress_line(bridge, token, line, stream_name="stderr"),
            )
            bridge.retire_child_workers_for_day(token)
        else:
            run_command(
                build_raw_command(
                    args,
                    target_date_local=token,
                    raw_dir=raw_dir,
                    scratch_dir=scratch_dir,
                    worker_plan=resolved_plan,
                )
            )
        raw_build_seconds = time.perf_counter() - raw_started_at
        if getattr(args, "overnight_subprocess", False):
            if bridge is not None:
                bridge.update_day(token, lifecycle_phase="overnight", details="subprocess")
                bridge.set_day_phase_worker(token, phase="overnight", details="subprocess")
                stream_command(
                    build_overnight_command(args, target_date_local=token, raw_dir=raw_dir, overnight_root=args.run_root / "nbm_overnight"),
                    stdout_handler=lambda line: bridge.log_event(token, message=f"overnight stdout: {line}", level="info"),
                    stderr_handler=lambda line: bridge.log_event(
                        token,
                        message=f"overnight stderr: {line}",
                        level=classify_child_output_level(stream_name="stderr", line=line),
                    ),
                )
            else:
                run_command(build_overnight_command(args, target_date_local=token, raw_dir=raw_dir, overnight_root=args.run_root / "nbm_overnight"))
        else:
            if bridge is not None:
                bridge.update_day(token, lifecycle_phase="overnight", details="in_process")
                bridge.set_day_phase_worker(token, phase="overnight", details="in_process")
            build_overnight_in_process(
                target_date_local=token,
                raw_dir=raw_dir,
                overnight_root=args.run_root / "nbm_overnight",
            )
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="validate", details="write_resume")
            bridge.set_day_phase_worker(token, phase="validate", details="write_resume")
        write_nbm_day_metadata(args.run_root, day, selection_mode=str(args.selection_mode))
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="validate", details="validate")
            bridge.set_day_phase_worker(token, phase="validate", details="validate")
        if not validate_nbm_day(args.run_root, day, selection_mode=str(args.selection_mode)):
            raise ValueError(f"NBM validation failed for target_date_local={token}")
    except BaseException:
        if bridge is not None:
            bridge.retire_child_workers_for_day(token)
            bridge.fail_day(token, message="failed")
        if not args.keep_temp_on_failure:
            delete_path(tmp_root)
        raise

    full_day_seconds = time.perf_counter() - day_started_at
    performance = collect_day_performance(
        token=token,
        raw_dir=raw_dir,
        raw_build_seconds=raw_build_seconds,
        cleanup_seconds=0.0,
        full_day_seconds=full_day_seconds,
    )
    cleanup_started_at = time.perf_counter()
    if bridge is not None:
        bridge.update_day(token, lifecycle_phase="cleanup", details="cleanup")
        bridge.set_day_phase_worker(token, phase="cleanup", details="cleanup")
    if getattr(args, "keep_reduced", False):
        delete_path(raw_dir)
    else:
        delete_path(tmp_root)
    cleanup_seconds = time.perf_counter() - cleanup_started_at
    performance = DayPerformance(**{**performance.__dict__, "cleanup_seconds": cleanup_seconds, "full_day_seconds": time.perf_counter() - day_started_at})
    write_nbm_day_performance(args.run_root, day, performance)
    if bridge is not None:
        bridge.complete_day(token, performance=performance)
    emit_runner_log("done", f"nbm date={token}")
    return DayRunResult(token=token, performance=performance)


def run_backfill(args: argparse.Namespace) -> int:
    args.run_root = resolve_run_root(args.run_root)
    args.run_root.mkdir(parents=True, exist_ok=True)
    overnight_root = args.run_root / "nbm_overnight"
    overnight_root.mkdir(parents=True, exist_ok=True)
    smart_plan = apply_smart_worker_plan(args)
    worker_controller = AdaptiveWorkerController(args, initial_plan=initial_worker_plan(args))
    runner_mode = resolve_progress_mode(mode=str(getattr(args, "progress_mode", "auto")))
    if smart_plan is not None:
        emit_runner_log(
            "smart-workers",
            (
                f"cpu_cores={smart_plan.cpu_cores} day_workers={smart_plan.day_workers} "
                f"workers={smart_plan.workers} lead_workers={smart_plan.lead_workers} "
                f"download_workers={smart_plan.download_workers} reduce_workers={smart_plan.reduce_workers} "
                f"extract_workers={smart_plan.extract_workers}"
            ),
        )

    pending_days: list[DayWindow] = []
    skipped_count = 0
    for day in iter_days(args.start_local_date, args.end_local_date):
        token = day_token(day)
        if not args.overwrite and validate_nbm_day(args.run_root, day, selection_mode=str(args.selection_mode)):
            emit_runner_log("skip", f"nbm date={token}")
            skipped_count += 1
            continue
        pending_days.append(day)

    day_workers = worker_controller.current_plan.day_workers
    min_free_gb = float(getattr(args, "min_free_gb", DEFAULT_MIN_FREE_GB))
    stop_admission = threading.Event()
    first_exception: BaseException | None = None
    stop_reason: str | None = None
    previous_handlers: dict[int, object] = {}
    progress_bridge: DayProgressBridge | None = None

    def request_drain(reason: str) -> None:
        nonlocal stop_reason
        if stop_admission.is_set():
            return
        stop_reason = reason
        stop_admission.set()
        emit_runner_log("pause", f"drain requested reason={reason}")

    if runner_mode == "dashboard":
        dashboard_kind = "nbm_batch" if str(getattr(args, "batch_reduce_mode", "off")) == "cycle" else "generic"
        reporter = create_progress_reporter(
            "NBM monthly backfill",
            unit="day",
            total=len(pending_days) + skipped_count,
            mode=str(getattr(args, "progress_mode", "auto")),
            on_pause_request=lambda **kwargs: request_drain(str(kwargs.get("reason", "operator"))),
            enable_dashboard_hotkeys=not bool(getattr(args, "disable_dashboard_hotkeys", False)),
            pause_control_file=getattr(args, "pause_control_file", None),
            dashboard_kind=dashboard_kind,
        )
        progress_bridge = DayProgressBridge(reporter, day_workers=day_workers, run_root=args.run_root)
        progress_bridge.reporter.set_metrics(
            DateRange=f"{args.start_local_date}..{args.end_local_date}",
            BatchMode=str(getattr(args, "batch_reduce_mode", "off")),
            SelectionMode=str(getattr(args, "selection_mode", "")),
            MetricProfile=str(getattr(args, "metric_profile", "")),
            Provenance="off" if bool(getattr(args, "overnight_fast", False)) else "on",
        )
        progress_bridge.set_plan_metrics(worker_controller.metrics())
        if skipped_count:
            progress_bridge.add_skipped(skipped_count, message=f"valid_on_disk={skipped_count}", affects_eta=False)
        if smart_plan is not None:
            progress_bridge.reporter.log_event(
                (
                    f"smart_workers cpu={smart_plan.cpu_cores} days={smart_plan.day_workers} "
                    f"lead={smart_plan.lead_workers} download={smart_plan.download_workers} "
                    f"reduce={smart_plan.reduce_workers} extract={smart_plan.extract_workers}"
                ),
                level="info",
            )
        if worker_controller.enabled:
            progress_bridge.reporter.log_event(
                (
                    f"adaptive_workers enabled sample_days={worker_controller.sample_days} "
                    f"cooldown_days={worker_controller.cooldown_days}"
                ),
                level="info",
            )

    def maybe_pause_requested() -> None:
        if progress_bridge is not None and progress_bridge.reporter.is_pause_requested():
            request_drain(progress_bridge.reporter.state.pause_reason or "operator")
            return
        pause_control_file = getattr(args, "pause_control_file", None)
        if pause_control_file and Path(pause_control_file).exists():
            request_drain("pause_control_file")

    def can_admit_more(active_count: int) -> bool:
        return free_disk_gb(args.run_root) >= min_free_gb

    def install_signal_handlers() -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        def _handler(signum, _frame):
            try:
                signal_name = signal.Signals(signum).name.lower()
            except Exception:
                signal_name = f"signal_{signum}"
            request_drain(signal_name)

        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, _handler)

    def restore_signal_handlers() -> None:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    install_signal_handlers()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_controller.max_day_workers) as executor:
            active_futures: dict[concurrent.futures.Future[DayRunResult], tuple[DayWindow, WorkerPlan]] = {}
            pending_index = 0

            while pending_index < len(pending_days) or active_futures:
                maybe_pause_requested()

                while (
                    pending_index < len(pending_days)
                    and len(active_futures) < worker_controller.current_plan.day_workers
                    and not stop_admission.is_set()
                    and can_admit_more(len(active_futures))
                ):
                    day = pending_days[pending_index]
                    day_plan = worker_controller.current_plan
                    if progress_bridge is not None:
                        progress_bridge.admit_day(day_token(day))
                        progress_bridge.set_plan_metrics(worker_controller.metrics())
                        progress_bridge.refresh_disk()
                    future = executor.submit(process_day, args, day, bridge=progress_bridge, worker_plan=day_plan)
                    active_futures[future] = (day, day_plan)
                    pending_index += 1

                if active_futures:
                    if progress_bridge is not None:
                        progress_bridge.set_plan_metrics(worker_controller.metrics())
                        progress_bridge.refresh_disk()
                    done, _ = concurrent.futures.wait(
                        tuple(active_futures.keys()),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        day, _day_plan = active_futures.pop(future, (None, None))
                        try:
                            result = future.result()
                            tuned, reason = worker_controller.record_success(result.performance)
                            emit_runner_log("adaptive-workers", f"token={result.token} action={'update' if tuned else 'no_change'} reason={reason}")
                            if progress_bridge is not None:
                                progress_bridge.set_plan_metrics(worker_controller.metrics())
                                progress_bridge.reporter.log_event(
                                    f"adaptive_workers {'update' if tuned else 'no_change'} reason={reason}",
                                    level="info",
                                )
                        except BaseException as exc:
                            tuned, reason = worker_controller.record_failure()
                            emit_runner_log("adaptive-workers", f"action={'update' if tuned else 'no_change'} reason={reason}")
                            if progress_bridge is not None:
                                progress_bridge.set_plan_metrics(worker_controller.metrics())
                                progress_bridge.reporter.log_event(
                                    f"adaptive_workers {'update' if tuned else 'no_change'} reason={reason}",
                                    level="info",
                                )
                            if first_exception is None:
                                first_exception = exc
                                request_drain("worker_failure")
                else:
                    if stop_admission.is_set():
                        break
                    if pending_index < len(pending_days) and not can_admit_more(len(active_futures)):
                        request_drain("low_disk")
                        break
                    if progress_bridge is not None:
                        progress_bridge.set_plan_metrics(worker_controller.metrics())
                        progress_bridge.refresh_disk()
                    time.sleep(0.1)

            if first_exception is not None:
                raise first_exception
    finally:
        restore_signal_handlers()

    if progress_bridge is not None:
        if stop_reason is not None and first_exception is None:
            progress_bridge.mark_paused(reason=stop_reason)
            emit_runner_log("paused", f"safe_to_exit reason={stop_reason}")
            progress_bridge.close(status="paused")
        else:
            progress_bridge.close(status="failed" if first_exception is not None else "done")
    elif stop_reason is not None:
        emit_runner_log("paused", f"safe_to_exit reason={stop_reason}")

    return 0


def main() -> int:
    return run_backfill(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
