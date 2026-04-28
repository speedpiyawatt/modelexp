#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import json
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.progress import create_progress_reporter, resolve_progress_mode

DEFAULT_RUN_ROOT = Path("data/runtime/backfill_overnight")
DEFAULT_SELECTION_MODE = "overnight_0005"
DEFAULT_DAY_WORKERS = 1
DEFAULT_RANGE_MERGE_GAP_BYTES = 65536
DEFAULT_BATCH_REDUCE_MODE = "cycle"
DEFAULT_CROP_METHOD = "auto"
DEFAULT_CROP_GRIB_TYPE = "same"
DEFAULT_WGRIB2_THREADS = 1
DEFAULT_EXTRACT_METHOD = "wgrib2-bin"
DEFAULT_SUMMARY_PROFILE = "overnight"
DEFAULT_SKIP_PROVENANCE = True
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
    status: str = "queued"
    started_at: float | None = None
    ended_at: float | None = None


@dataclass(frozen=True)
class DayPerformance:
    token: str
    expected_task_count: int
    completed_task_count: int
    raw_build_seconds: float
    extract_outputs_seconds: float
    cleanup_seconds: float
    full_day_seconds: float
    timing_idx_fetch_seconds_median: float | None
    timing_range_download_seconds_median: float | None
    timing_reduce_seconds_median: float | None
    timing_cfgrib_open_seconds_median: float | None
    timing_row_build_seconds_median: float | None
    timing_cleanup_seconds_median: float | None


@dataclass(frozen=True)
class DayRunResult:
    token: str
    performance: DayPerformance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run HRRR overnight backfill day by day, skip valid daily outputs already on disk, and rebuild invalid days."
    )
    parser.add_argument("--start-local-date", required=True, help="Inclusive first target local date in YYYY-MM-DD.")
    parser.add_argument("--end-local-date", required=True, help="Inclusive last target local date in YYYY-MM-DD.")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    parser.add_argument("--day-workers", type=int, default=DEFAULT_DAY_WORKERS)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--download-workers", type=int)
    parser.add_argument("--reduce-workers", type=int)
    parser.add_argument("--extract-workers", type=int)
    parser.add_argument("--reduce-queue-size", type=int)
    parser.add_argument("--extract-queue-size", type=int)
    parser.add_argument("--range-merge-gap-bytes", type=int, default=DEFAULT_RANGE_MERGE_GAP_BYTES)
    parser.add_argument("--batch-reduce-mode", choices=("off", "cycle"), default=DEFAULT_BATCH_REDUCE_MODE)
    parser.add_argument("--crop-method", choices=("auto", "small_grib", "ijsmall_grib"), default=DEFAULT_CROP_METHOD)
    parser.add_argument("--crop-grib-type", default=DEFAULT_CROP_GRIB_TYPE)
    parser.add_argument("--wgrib2-threads", type=int, default=DEFAULT_WGRIB2_THREADS)
    parser.add_argument("--extract-method", choices=("cfgrib", "eccodes", "wgrib2-bin", "wgrib2-ijbox-bin"), default=DEFAULT_EXTRACT_METHOD)
    parser.add_argument("--summary-profile", choices=("full", "overnight"), default=DEFAULT_SUMMARY_PROFILE)
    parser.add_argument("--skip-provenance", action="store_true", default=DEFAULT_SKIP_PROVENANCE)
    parser.add_argument("--write-provenance", action="store_false", dest="skip_provenance", help="Write provenance parquet output instead of the optimized summary-only default.")
    parser.add_argument("--progress-mode", choices=("auto", "dashboard", "log"), default="auto")
    parser.add_argument(
        "--disable-dashboard-hotkeys",
        action="store_true",
        help="Disable interactive dashboard hotkeys such as 'p' for graceful drain-and-pause.",
    )
    parser.add_argument(
        "--pause-control-file",
        help="Optional file path that the monthly runner watches for graceful drain-and-pause requests.",
    )
    parser.add_argument("--max-task-attempts", type=int)
    parser.add_argument("--retry-backoff-seconds", type=float)
    parser.add_argument("--retry-max-backoff-seconds", type=float)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--keep-reduced", action="store_true", help="Keep reduced GRIB2 files after successful extraction.")
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


def emit_runner_log(tag: str, message: str) -> None:
    with RUN_LOG_LOCK:
        print(f"[{tag}] {message}", flush=True)


def run_command(command: list[str]) -> None:
    emit_runner_log("RUN", " ".join(command))
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True, start_new_session=True)
    except TypeError as exc:
        if "start_new_session" not in str(exc):
            raise
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def stream_command(command: list[str], *, stdout_handler=None, stderr_handler=None) -> None:
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


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def day_temp_root(run_root: Path, target_date_local: str) -> Path:
    return run_root / "hrrr_tmp" / target_date_local


def hrrr_summary_paths(run_root: Path, day: DayWindow) -> dict[str, Path]:
    summary_root = run_root / "hrrr_summary" / f"target_date_local={day_token(day)}"
    state_root = run_root / "hrrr_summary_state" / f"target_date_local={day_token(day)}"
    return {
        "summary_root": summary_root,
        "state_root": state_root,
        "summary": summary_root / "hrrr.overnight.parquet",
        "manifest_json": state_root / "hrrr.manifest.json",
        "manifest_parquet": state_root / "hrrr.manifest.parquet",
        "performance": state_root / "hrrr.performance.json",
    }


def hrrr_day_performance_path(run_root: Path, day: DayWindow) -> Path:
    return hrrr_summary_paths(run_root, day)["performance"]


def validate_hrrr_day(
    run_root: Path,
    day: DayWindow,
    *,
    selection_mode: str,
    extract_method: str = DEFAULT_EXTRACT_METHOD,
    summary_profile: str = DEFAULT_SUMMARY_PROFILE,
    provenance_written: bool = not DEFAULT_SKIP_PROVENANCE,
) -> bool:
    paths = hrrr_summary_paths(run_root, day)
    if not paths["summary"].exists() or not paths["manifest_json"].exists() or not paths["manifest_parquet"].exists():
        return False
    try:
        summary_df = pd.read_parquet(paths["summary"])
        manifest = json.loads(paths["manifest_json"].read_text())
        manifest_df = pd.read_parquet(paths["manifest_parquet"])
    except Exception:
        return False
    if summary_df.empty or len(summary_df) != 1:
        return False
    if str(summary_df.iloc[0].get("target_date_local")) != day_token(day):
        return False
    if not bool(manifest.get("complete")):
        return False
    if str(manifest.get("target_date_local")) != day_token(day):
        return False
    if str(manifest.get("selection_mode")) != selection_mode:
        return False
    if str(manifest.get("extract_method", "cfgrib")) != extract_method:
        return False
    if str(manifest.get("summary_profile", "full")) != summary_profile:
        return False
    if bool(manifest.get("provenance_written", True)) != provenance_written:
        return False
    expected_count = int(manifest.get("expected_task_count", -1))
    completed_count = len(manifest.get("completed_task_keys", []))
    if expected_count < 0 or expected_count != completed_count:
        return False
    if manifest_df.empty:
        return False
    if "selection_mode" in manifest_df.columns and not manifest_df["selection_mode"].eq(selection_mode).all():
        return False
    if "extract_method" in manifest_df.columns and not manifest_df["extract_method"].fillna("cfgrib").eq(extract_method).all():
        return False
    if "summary_profile" in manifest_df.columns and not manifest_df["summary_profile"].fillna("full").eq(summary_profile).all():
        return False
    if "provenance_written" in manifest_df.columns and not manifest_df["provenance_written"].fillna(True).eq(provenance_written).all():
        return False
    return bool((manifest_df["status"] == "ok").all())


def requested_hrrr_contract(args: argparse.Namespace) -> dict[str, object]:
    return {
        "selection_mode": str(args.selection_mode),
        "extract_method": str(getattr(args, "extract_method", None) or DEFAULT_EXTRACT_METHOD),
        "summary_profile": str(getattr(args, "summary_profile", None) or DEFAULT_SUMMARY_PROFILE),
        "provenance_written": not bool(getattr(args, "skip_provenance", DEFAULT_SKIP_PROVENANCE)),
    }


def effective_arg(args: argparse.Namespace, name: str, default: object) -> object:
    value = getattr(args, name, None)
    return default if value is None else value


def child_progress_mode(args: argparse.Namespace, *, parent_dashboard_active: bool) -> str:
    if parent_dashboard_active:
        return "log"
    return str(getattr(args, "progress_mode", "auto"))


def build_hrrr_command(
    args: argparse.Namespace,
    *,
    day: DayWindow,
    tmp_root: Path,
    summary_dir: Path,
    output_dir: Path,
    progress_mode: str | None = None,
    include_pause_control_file: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "tools/hrrr/build_hrrr_klga_feature_shards.py",
        "--start-date",
        day_token(day),
        "--end-date",
        day_token(day),
        "--selection-mode",
        str(args.selection_mode),
        "--download-dir",
        str(tmp_root / "downloads"),
        "--reduced-dir",
        str(tmp_root / "reduced"),
        "--output-dir",
        str(output_dir),
        "--summary-output-dir",
        str(summary_dir),
        "--scratch-dir",
        str(tmp_root / "scratch"),
        "--max-workers",
        str(args.max_workers),
        "--progress-mode",
        progress_mode or str(args.progress_mode),
        "--batch-reduce-mode",
        str(effective_arg(args, "batch_reduce_mode", DEFAULT_BATCH_REDUCE_MODE)),
    ]
    if getattr(args, "download_workers", None) is not None:
        command.extend(["--download-workers", str(args.download_workers)])
    if getattr(args, "reduce_workers", None) is not None:
        command.extend(["--reduce-workers", str(args.reduce_workers)])
    if getattr(args, "extract_workers", None) is not None:
        command.extend(["--extract-workers", str(args.extract_workers)])
    if getattr(args, "reduce_queue_size", None) is not None:
        command.extend(["--reduce-queue-size", str(args.reduce_queue_size)])
    if getattr(args, "extract_queue_size", None) is not None:
        command.extend(["--extract-queue-size", str(args.extract_queue_size)])
    command.extend(["--range-merge-gap-bytes", str(effective_arg(args, "range_merge_gap_bytes", DEFAULT_RANGE_MERGE_GAP_BYTES))])
    command.extend(["--crop-method", str(effective_arg(args, "crop_method", DEFAULT_CROP_METHOD))])
    command.extend(["--crop-grib-type", str(effective_arg(args, "crop_grib_type", DEFAULT_CROP_GRIB_TYPE))])
    command.extend(["--wgrib2-threads", str(effective_arg(args, "wgrib2_threads", DEFAULT_WGRIB2_THREADS))])
    command.extend(["--extract-method", str(effective_arg(args, "extract_method", DEFAULT_EXTRACT_METHOD))])
    command.extend(["--summary-profile", str(effective_arg(args, "summary_profile", DEFAULT_SUMMARY_PROFILE))])
    if bool(effective_arg(args, "skip_provenance", DEFAULT_SKIP_PROVENANCE)):
        command.append("--skip-provenance")
    if getattr(args, "max_task_attempts", None) is not None:
        command.extend(["--max-task-attempts", str(args.max_task_attempts)])
    if getattr(args, "retry_backoff_seconds", None) is not None:
        command.extend(["--retry-backoff-seconds", str(args.retry_backoff_seconds)])
    if getattr(args, "retry_max_backoff_seconds", None) is not None:
        command.extend(["--retry-max-backoff-seconds", str(args.retry_max_backoff_seconds)])
    if include_pause_control_file and getattr(args, "pause_control_file", None) is not None:
        command.extend(["--pause-control-file", str(args.pause_control_file)])
    if getattr(args, "allow_partial", False):
        command.append("--allow-partial")
    if getattr(args, "keep_reduced", False):
        command.append("--keep-reduced")
    return command


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
        elif current_key is not None:
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
    if "cfgrib.dataset.datasetbuilderror" in lowered or "filter_by_keys" in lowered:
        return "info"
    return "error"


class DayProgressBridge:
    def __init__(self, reporter, *, day_workers: int, run_root: Path) -> None:
        self.reporter = reporter
        self.day_workers = max(1, int(day_workers))
        self.run_root = run_root
        self._lock = threading.Lock()
        self._days: dict[str, DayProgressState] = {}

    def admit_day(self, token: str) -> None:
        with self._lock:
            self._days[token] = DayProgressState(started_at=time.perf_counter())
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=1, status="queued")
        self._refresh_metrics()

    def update_day(self, token: str, *, lifecycle_phase: str, details: str) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState(started_at=time.perf_counter()))
            state.lifecycle_phase = lifecycle_phase
            state.details = details
            state.status = lifecycle_phase
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=0, active=1, status=f"{lifecycle_phase}:{details}")
        self._refresh_metrics()

    def set_day_phase_worker(self, token: str, *, phase: str, details: str) -> None:
        worker_id = f"day::{token}"
        if worker_id in self.reporter.state.workers and self.reporter.state.workers[worker_id].active:
            self.reporter.update_worker(worker_id, phase=phase, details=details, label=token)
        else:
            self.reporter.start_worker(worker_id, label=token, phase=phase, group_id=token, details=details)

    def start_child_worker(self, token: str, *, child_worker: str, label: str, phase: str, details: str | None = None) -> None:
        self.reporter.start_worker(f"{token}/{child_worker}", label=label, phase=phase, group_id=token, details=details)
        self.update_day(token, lifecycle_phase=self._map_child_phase(phase=phase, details=details), details=details or phase)

    def update_child_worker(self, token: str, *, child_worker: str, phase: str | None = None, label: str | None = None, details: str | None = None) -> None:
        self.reporter.update_worker(f"{token}/{child_worker}", phase=phase, label=label, details=details)
        self.update_day(token, lifecycle_phase=self._map_child_phase(phase=phase, details=details), details=details or phase or "raw")

    def retire_child_worker(self, token: str, *, child_worker: str, message: str | None = None) -> None:
        self.reporter.retire_worker(f"{token}/{child_worker}", message=message)

    def retire_child_workers_for_day(self, token: str) -> None:
        prefix = f"{token}/"
        for worker_id in list(self.reporter.state.workers):
            if worker_id.startswith(prefix):
                self.reporter.retire_worker(worker_id)

    def start_transfer(self, token: str, *, child_worker: str, file_label: str, total_bytes: int | None = None) -> None:
        self.reporter.start_transfer(f"{token}/{child_worker}", file_label=file_label, total_bytes=total_bytes)

    def update_transfer(self, token: str, *, child_worker: str, bytes_downloaded: int, total_bytes: int | None = None) -> None:
        self.reporter.update_transfer(f"{token}/{child_worker}", bytes_downloaded=bytes_downloaded, total_bytes=total_bytes)

    def finish_transfer(self, token: str, *, child_worker: str) -> None:
        self.reporter.finish_transfer(f"{token}/{child_worker}")

    def set_child_attempt(self, token: str, *, child_worker: str, attempt: int, max_attempts: int) -> None:
        self.reporter.set_worker_attempt(f"{token}/{child_worker}", attempt=attempt, max_attempts=max_attempts)

    def schedule_retry(self, token: str, *, child_worker: str, attempt: int, max_attempts: int, delay_seconds: float, message: str, error_class: str) -> None:
        self.reporter.schedule_retry(
            f"{token}/{child_worker}",
            attempt=attempt,
            max_attempts=max_attempts,
            delay_seconds=delay_seconds,
            message=message,
            error_class=error_class,
        )

    def start_retry(self, token: str, *, child_worker: str, attempt: int, max_attempts: int) -> None:
        self.reporter.start_retry(f"{token}/{child_worker}", attempt=attempt, max_attempts=max_attempts)

    def recover_retry(self, token: str, *, child_worker: str, message: str) -> None:
        self.reporter.recover_worker(f"{token}/{child_worker}", message=message)

    def log_event(self, token: str, *, message: str, level: str = "info") -> None:
        self.reporter.log_event(f"{token} {message}", level=level)

    def complete_day(self, token: str, *, performance: DayPerformance | None = None) -> None:
        with self._lock:
            state = self._days.setdefault(token, DayProgressState())
            state.lifecycle_phase = "complete"
            state.status = "complete"
            state.ended_at = time.perf_counter()
        self.reporter.retire_worker(f"day::{token}", message=f"{token} done")
        self.reporter.upsert_group(token, label=token, total=1, completed=1, failed=0, active=0, status="complete")
        if performance is not None:
            self.reporter.record_batch_timing(**asdict(performance))
        self.reporter.record_outcome("completed", message=f"{token} done")
        with self._lock:
            self._days.pop(token, None)
        self._refresh_metrics()

    def fail_day(self, token: str, *, message: str) -> None:
        with self._lock:
            self._days.pop(token, None)
        self.reporter.retire_worker(f"day::{token}", message=f"{token} {message}")
        self.reporter.upsert_group(token, label=token, total=1, completed=0, failed=1, active=0, status=message)
        self.reporter.record_outcome("failed", message=f"{token} {message}")
        self._refresh_metrics()

    def add_skipped(self, count: int, *, message: str) -> None:
        if count > 0:
            self.reporter.add_skipped(count, message=message, affects_eta=False)
            self._refresh_metrics()

    def mark_paused(self, *, reason: str | None = None) -> None:
        self.reporter.mark_paused(reason=reason)

    def close(self, *, status: str) -> None:
        self.reporter.close(status=status)

    def _refresh_metrics(self) -> None:
        with self._lock:
            active_days = len(self._days)
            counts: dict[str, int] = {"Raw": 0, "Reduce": 0, "Extract": 0, "Validate": 0, "Cleanup": 0}
            for state in self._days.values():
                if state.lifecycle_phase == "validate":
                    counts["Validate"] += 1
                elif state.lifecycle_phase == "cleanup":
                    counts["Cleanup"] += 1
                elif state.lifecycle_phase in {"download", "reduce", "extract", "raw"}:
                    counts["Raw" if state.lifecycle_phase in {"download", "raw"} else state.lifecycle_phase.capitalize()] += 1
        total_days = self.reporter.state.total
        queued_days = None if total_days is None else max(0, int(total_days) - self.reporter.state.completed_total - active_days)
        self.reporter.set_metrics(
            DayWorkers=self.day_workers,
            ActiveDays=active_days,
            QueuedDays=queued_days,
            Raw=counts["Raw"],
            Reduce=counts["Reduce"],
            Extract=counts["Extract"],
            Validate=counts["Validate"],
            Cleanup=counts["Cleanup"],
        )

    @staticmethod
    def _map_child_phase(*, phase: str | None, details: str | None) -> str:
        text = " ".join(part for part in [phase or "", details or ""] if part).lower()
        if any(token in text for token in ("download", "idx", "fetch", "range", "reuse_raw")):
            return "download"
        if any(token in text for token in ("reduce", "crop", "wgrib")):
            return "reduce"
        if any(token in text for token in ("extract", "cfgrib", "open", "row", "build")):
            return "extract"
        return "raw"


def relay_child_progress_line(bridge: DayProgressBridge, token: str, line: str, *, stream_name: str) -> None:
    parsed = parse_progress_line(line)
    if parsed is None:
        if line.startswith(("download_start ", "download_progress ", "download_complete ")) or not line.strip():
            return
        bridge.log_event(token, message=f"{stream_name}: {line}", level=classify_child_output_level(stream_name=stream_name, line=line))
        return
    event_name, payload = parsed
    child_worker = payload.get("worker") or "worker"
    if event_name == "worker_start":
        bridge.start_child_worker(
            token,
            child_worker=child_worker,
            label=payload.get("label") or child_worker,
            phase=payload.get("phase") or "raw",
            details=payload.get("details"),
        )
    elif event_name == "worker_update":
        bridge.update_child_worker(
            token,
            child_worker=child_worker,
            phase=payload.get("phase") or "raw",
            label=payload.get("label"),
            details=payload.get("details"),
        )
    elif event_name == "transfer_start":
        bridge.start_transfer(token, child_worker=child_worker, file_label=payload.get("file") or "transfer", total_bytes=int_or_none(payload.get("total_bytes")))
    elif event_name == "transfer_progress":
        bridge.update_transfer(token, child_worker=child_worker, bytes_downloaded=int_or_none(payload.get("bytes_downloaded")) or 0, total_bytes=int_or_none(payload.get("total_bytes")))
    elif event_name == "transfer_complete":
        bridge.finish_transfer(token, child_worker=child_worker)
    elif event_name == "worker_attempt":
        bridge.set_child_attempt(token, child_worker=child_worker, attempt=int_or_none(payload.get("attempt")) or 1, max_attempts=int_or_none(payload.get("max_attempts")) or 1)
    elif event_name == "retry_scheduled":
        bridge.schedule_retry(
            token,
            child_worker=child_worker,
            attempt=int_or_none(payload.get("attempt")) or 1,
            max_attempts=int_or_none(payload.get("max_attempts")) or 1,
            delay_seconds=float_or_none(payload.get("delay_seconds")) or 0.0,
            message=payload.get("message") or "retry_scheduled",
            error_class=payload.get("error_class") or "retry",
        )
    elif event_name == "retry_started":
        bridge.start_retry(token, child_worker=child_worker, attempt=int_or_none(payload.get("attempt")) or 1, max_attempts=int_or_none(payload.get("max_attempts")) or 1)
    elif event_name == "retry_recovered":
        bridge.recover_retry(token, child_worker=child_worker, message=payload.get("message") or f"{token} retry recovered")
    elif event_name in {"worker_retire", "worker_complete"}:
        bridge.retire_child_worker(token, child_worker=child_worker, message=payload.get("message"))
    elif event_name == "group":
        bridge.update_day(token, lifecycle_phase="raw", details=payload.get("status") or payload.get("label") or "group")
    elif event_name in {"message", "outcome"} and payload.get("message"):
        bridge.log_event(token, message=payload["message"], level=payload.get("level") or "info")


def extract_day_outputs(
    *,
    run_root: Path,
    day: DayWindow,
    tmp_output_dir: Path,
    tmp_summary_dir: Path,
    selection_mode: str,
) -> None:
    paths = hrrr_summary_paths(run_root, day)
    month_id = day.target_date_local.strftime("%Y-%m")
    summary_month_path = tmp_summary_dir / f"{month_id}.parquet"
    manifest_json_path = tmp_output_dir / f"{month_id}.manifest.json"
    manifest_parquet_path = tmp_output_dir / f"{month_id}.manifest.parquet"
    if not summary_month_path.exists() or not manifest_json_path.exists() or not manifest_parquet_path.exists():
        raise ValueError(f"HRRR one-day run missing month outputs for target_date_local={day_token(day)}")

    summary_df = pd.read_parquet(summary_month_path)
    summary_df = summary_df.loc[summary_df["target_date_local"] == day_token(day)].reset_index(drop=True)
    if len(summary_df) != 1:
        raise ValueError(f"HRRR one-day summary row mismatch for target_date_local={day_token(day)}")

    manifest = json.loads(manifest_json_path.read_text())
    manifest_df = pd.read_parquet(manifest_parquet_path).copy()

    paths["summary_root"].mkdir(parents=True, exist_ok=True)
    paths["state_root"].mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(paths["summary"], index=False)

    manifest["target_date_local"] = day_token(day)
    manifest["selection_mode"] = selection_mode
    manifest["summary_parquet_path"] = str(paths["summary"])
    manifest["manifest_json_path"] = str(paths["manifest_json"])
    manifest["manifest_parquet_path"] = str(paths["manifest_parquet"])
    for key in (
        "wide_parquet_path",
        "provenance_parquet_path",
        "summary_output_dir",
        "output_dir",
        "download_dir",
        "reduced_dir",
        "scratch_dir",
    ):
        if key in manifest:
            manifest[key] = None
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True))

    if not manifest_df.empty:
        replacement_paths = {
            "summary_parquet_path": str(paths["summary"]),
            "manifest_json_path": str(paths["manifest_json"]),
            "manifest_parquet_path": str(paths["manifest_parquet"]),
        }
        for column, value in replacement_paths.items():
            if column in manifest_df.columns:
                manifest_df[column] = value
        manifest_df["selection_mode"] = selection_mode
        for column in (
            "wide_parquet_path",
            "provenance_parquet_path",
            "output_dir",
            "summary_output_dir",
            "download_dir",
            "reduced_dir",
            "scratch_dir",
        ):
            if column in manifest_df.columns:
                manifest_df[column] = pd.NA
    manifest_df.to_parquet(paths["manifest_parquet"], index=False)


def _metric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _series_median(frame: pd.DataFrame, column: str) -> float | None:
    series = _metric_series(frame, column)
    if series.empty:
        return None
    return float(series.median())


def collect_day_performance(
    *,
    token: str,
    tmp_output_dir: Path,
    raw_build_seconds: float,
    extract_outputs_seconds: float,
    cleanup_seconds: float,
    full_day_seconds: float,
) -> DayPerformance:
    manifest_paths = sorted(tmp_output_dir.glob("*.manifest.parquet"))
    if manifest_paths:
        frames = [pd.read_parquet(path) for path in manifest_paths]
        manifest_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        manifest_df = pd.DataFrame()
    if manifest_df.empty:
        expected_count = 0
        completed_count = 0
    else:
        expected_count = int(len(manifest_df))
        completed_count = int((manifest_df["status"] == "ok").sum()) if "status" in manifest_df else expected_count
    return DayPerformance(
        token=token,
        expected_task_count=expected_count,
        completed_task_count=completed_count,
        raw_build_seconds=float(raw_build_seconds),
        extract_outputs_seconds=float(extract_outputs_seconds),
        cleanup_seconds=float(cleanup_seconds),
        full_day_seconds=float(full_day_seconds),
        timing_idx_fetch_seconds_median=_series_median(manifest_df, "timing_idx_fetch_seconds"),
        timing_range_download_seconds_median=_series_median(manifest_df, "timing_range_download_seconds"),
        timing_reduce_seconds_median=_series_median(manifest_df, "timing_reduce_seconds"),
        timing_cfgrib_open_seconds_median=_series_median(manifest_df, "timing_cfgrib_open_seconds"),
        timing_row_build_seconds_median=_series_median(manifest_df, "timing_row_build_seconds"),
        timing_cleanup_seconds_median=_series_median(manifest_df, "timing_cleanup_seconds"),
    )


def write_hrrr_day_performance(run_root: Path, day: DayWindow, performance: DayPerformance) -> None:
    path = hrrr_day_performance_path(run_root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(performance), indent=2, sort_keys=True))


def process_day(args: argparse.Namespace, day: DayWindow, *, bridge: DayProgressBridge | None = None) -> DayRunResult:
    token = day_token(day)
    paths = hrrr_summary_paths(args.run_root, day)
    tmp_root = day_temp_root(args.run_root, token)
    tmp_output_dir = tmp_root / "output"
    tmp_summary_dir = tmp_root / "summary"
    day_started_at = time.perf_counter()

    delete_path(paths["summary_root"])
    delete_path(paths["state_root"])
    delete_path(tmp_root)
    emit_runner_log("start", f"hrrr date={token}")
    try:
        raw_started_at = time.perf_counter()
        command = build_hrrr_command(
            args,
            day=day,
            tmp_root=tmp_root,
            summary_dir=tmp_summary_dir,
            output_dir=tmp_output_dir,
            progress_mode=child_progress_mode(args, parent_dashboard_active=bridge is not None),
            include_pause_control_file=False,
        )
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="raw", details="start_raw")
            bridge.set_day_phase_worker(token, phase="raw", details="start_raw")
            stream_command(
                command,
                stdout_handler=lambda line: relay_child_progress_line(bridge, token, line, stream_name="stdout"),
                stderr_handler=lambda line: relay_child_progress_line(bridge, token, line, stream_name="stderr"),
            )
            bridge.retire_child_workers_for_day(token)
        else:
            run_command(command)
        raw_build_seconds = time.perf_counter() - raw_started_at

        extract_started_at = time.perf_counter()
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="validate", details="extract_outputs")
            bridge.set_day_phase_worker(token, phase="validate", details="extract_outputs")
        extract_day_outputs(
            run_root=args.run_root,
            day=day,
            tmp_output_dir=tmp_output_dir,
            tmp_summary_dir=tmp_summary_dir,
            selection_mode=str(args.selection_mode),
        )
        extract_outputs_seconds = time.perf_counter() - extract_started_at
        if bridge is not None:
            bridge.update_day(token, lifecycle_phase="validate", details="validate")
            bridge.set_day_phase_worker(token, phase="validate", details="validate")
        if not validate_hrrr_day(args.run_root, day, **requested_hrrr_contract(args)):
            raise ValueError(f"HRRR validation failed for target_date_local={token}")
    except BaseException:
        if bridge is not None:
            bridge.retire_child_workers_for_day(token)
            bridge.fail_day(token, message="failed")
        if not args.keep_temp_on_failure:
            delete_path(tmp_root)
        raise

    performance = collect_day_performance(
        token=token,
        tmp_output_dir=tmp_output_dir,
        raw_build_seconds=raw_build_seconds,
        extract_outputs_seconds=extract_outputs_seconds,
        cleanup_seconds=0.0,
        full_day_seconds=time.perf_counter() - day_started_at,
    )
    cleanup_started_at = time.perf_counter()
    if bridge is not None:
        bridge.update_day(token, lifecycle_phase="cleanup", details="cleanup")
        bridge.set_day_phase_worker(token, phase="cleanup", details="cleanup")
    delete_path(tmp_root)
    cleanup_seconds = time.perf_counter() - cleanup_started_at
    performance = DayPerformance(**{**performance.__dict__, "cleanup_seconds": cleanup_seconds, "full_day_seconds": time.perf_counter() - day_started_at})
    write_hrrr_day_performance(args.run_root, day, performance)
    if bridge is not None:
        bridge.complete_day(token, performance=performance)
    emit_runner_log("done", f"hrrr date={token}")
    return DayRunResult(token=token, performance=performance)


def run_backfill(args: argparse.Namespace) -> int:
    args.run_root.mkdir(parents=True, exist_ok=True)
    (args.run_root / "hrrr_summary").mkdir(parents=True, exist_ok=True)
    (args.run_root / "hrrr_summary_state").mkdir(parents=True, exist_ok=True)

    days = iter_days(args.start_local_date, args.end_local_date)
    pending_days: list[DayWindow] = []
    skipped_count = 0
    for day in days:
        token = day_token(day)
        if validate_hrrr_day(args.run_root, day, **requested_hrrr_contract(args)):
            emit_runner_log("skip", f"hrrr date={token}")
            skipped_count += 1
            continue
        pending_days.append(day)

    day_workers = max(1, int(getattr(args, "day_workers", DEFAULT_DAY_WORKERS)))
    runner_mode = resolve_progress_mode(mode=str(getattr(args, "progress_mode", "auto")))
    stop_admission = threading.Event()
    first_exception: BaseException | None = None
    stop_reason: str | None = None
    previous_handlers: dict[int, object] = {}
    bridge: DayProgressBridge | None = None

    def request_drain(reason: str) -> None:
        nonlocal stop_reason
        if stop_admission.is_set():
            return
        stop_reason = reason
        stop_admission.set()
        emit_runner_log("pause", f"drain requested reason={reason}")

    if runner_mode == "dashboard":
        reporter = create_progress_reporter(
            "HRRR monthly backfill",
            unit="day",
            total=len(pending_days) + skipped_count,
            mode=str(getattr(args, "progress_mode", "auto")),
            on_pause_request=lambda **kwargs: request_drain(str(kwargs.get("reason", "operator"))),
            enable_dashboard_hotkeys=not bool(getattr(args, "disable_dashboard_hotkeys", False)),
            pause_control_file=getattr(args, "pause_control_file", None),
        )
        bridge = DayProgressBridge(reporter, day_workers=day_workers, run_root=args.run_root)
        reporter.set_metrics(
            DateRange=f"{args.start_local_date}..{args.end_local_date}",
            SelectionMode=str(getattr(args, "selection_mode", "")),
            MaxWorkers=int(getattr(args, "max_workers", 1)),
            DayWorkers=day_workers,
        )
        if skipped_count:
            bridge.add_skipped(skipped_count, message=f"valid_on_disk={skipped_count}")

    def maybe_pause_requested() -> None:
        if bridge is not None and bridge.reporter.is_pause_requested():
            request_drain(bridge.reporter.state.pause_reason or "operator")
            return
        pause_control_file = getattr(args, "pause_control_file", None)
        if pause_control_file and Path(pause_control_file).exists():
            request_drain("pause_control_file")

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
        with concurrent.futures.ThreadPoolExecutor(max_workers=day_workers) as executor:
            active: dict[concurrent.futures.Future[DayRunResult], DayWindow] = {}
            pending_index = 0
            while pending_index < len(pending_days) or active:
                maybe_pause_requested()
                while pending_index < len(pending_days) and len(active) < day_workers and not stop_admission.is_set():
                    day = pending_days[pending_index]
                    if bridge is not None:
                        bridge.admit_day(day_token(day))
                    future = executor.submit(process_day, args, day, bridge=bridge)
                    active[future] = day
                    pending_index += 1
                if active:
                    done, _ = concurrent.futures.wait(
                        tuple(active.keys()),
                        timeout=0.1,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        active.pop(future, None)
                        try:
                            future.result()
                        except BaseException as exc:
                            if first_exception is None:
                                first_exception = exc
                                request_drain("worker_failure")
                else:
                    if stop_admission.is_set():
                        break
                    time.sleep(0.1)
            if first_exception is not None:
                raise first_exception
    finally:
        restore_signal_handlers()

    if bridge is not None:
        if stop_reason is not None and first_exception is None:
            bridge.mark_paused(reason=stop_reason)
            emit_runner_log("paused", f"safe_to_exit reason={stop_reason}")
            bridge.close(status="paused")
        else:
            bridge.close(status="failed" if first_exception is not None else "done")
    elif stop_reason is not None:
        emit_runner_log("paused", f"safe_to_exit reason={stop_reason}")

    return 0


def main() -> int:
    return run_backfill(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
