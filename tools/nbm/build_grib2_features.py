#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "build_grib2_features.py requires xarray. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    import cfgrib  # noqa: F401
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "build_grib2_features.py requires cfgrib and eccodes. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fetch_nbm import GRIB2_BASE, S3HttpClient, iter_dates, list_grib2_cycles, normalize_date, resolve_grib2_key
from tools.weather.location_context import (
    LOCAL_NEIGHBORHOOD_SIZES,
    CropBounds,
    REGIONAL_CROP_BOUNDS,
    SETTLEMENT_LOCATION,
    crop_context_metrics,
    crop_metadata,
    find_nearest_grid_cell,
    infer_north_is_first,
    infer_west_is_first,
    longitude_360_to_180,
    local_context_metrics,
)
from tools.weather.progress import ProgressBar, RunControl, create_progress_reporter, emit_progress_message
from tools.weather.retry import RetryPolicy, classify_task_failure, compute_retry_delay_seconds, should_retry_attempt


NY_TZ = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
DEFAULT_OUTPUT = pathlib.Path("data/nbm/grib2")
LEAD_HOURS = list(range(1, 37))
DEFAULT_SELECTION_MODE = "all"
SELECTION_MODES = (DEFAULT_SELECTION_MODE, "overnight_0005")
DEFAULT_RANGE_MERGE_GAP_BYTES = 4096
DEFAULT_CFGRIB_INDEX_STRATEGY = "temp_dir_per_unit"
OVERNIGHT_INIT_START_HOUR_LOCAL = 18
OVERNIGHT_CUTOFF_LOCAL_TIME = dt.time(hour=0, minute=5)
WIDE_LONG_FIELDS = {"PTYPE", "PWTHER", "TSTM"}
CROP_METHODS = ("auto", "small_grib", "ijsmall_grib")
BATCH_REDUCE_MODES = ("off", "cycle")
DEFAULT_CROP_METHOD = "auto"
DEFAULT_CROP_GRIB_TYPE = "same"
SOURCE_MODEL = "NBM"
SOURCE_PRODUCT = "grib2-core"
SOURCE_VERSION = "nbm-grib2-core-public"
REQUIRED_WIDE_IDENTITY_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "fallback_used_any",
    "station_id",
    "init_time_utc",
    "valid_time_utc",
    "init_time_local",
    "valid_time_local",
    "init_date_local",
    "valid_date_local",
    "forecast_hour",
    "settlement_lat",
    "settlement_lon",
    "crop_top_lat",
    "crop_bottom_lat",
    "crop_left_lon",
    "crop_right_lon",
    "nearest_grid_lat",
    "nearest_grid_lon",
}
CROP_BOUNDS = {
    "top": REGIONAL_CROP_BOUNDS.top,
    "bottom": REGIONAL_CROP_BOUNDS.bottom,
    "left": REGIONAL_CROP_BOUNDS.left,
    "right": REGIONAL_CROP_BOUNDS.right,
}
WGRIB2_ENV_VAR = "WGRIB2_BINARY"
GRID_SHAPE_PATTERNS = (
    re.compile(r"(?:grid:)?\((?P<nx>\d+)\s+x\s+(?P<ny>\d+)\)", re.IGNORECASE),
    re.compile(r"\bnx\s*[=:]\s*(?P<nx>\d+)\b.*?\bny\s*[=:]\s*(?P<ny>\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?P<nx>\d+)\s+by\s+(?P<ny>\d+)\b", re.IGNORECASE),
)
CROP_GRID_CACHE_LOCKS: dict[pathlib.Path, threading.Lock] = {}
CROP_GRID_CACHE_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True)
class CyclePlan:
    init_time_utc: dt.datetime
    init_time_local: dt.datetime
    cycle: str
    selected_target_dates: tuple[dt.date, ...] = ()

    @property
    def init_date_local(self) -> dt.date:
        return self.init_time_local.date()

    @property
    def mode(self) -> str:
        cutoff = self.init_time_local.replace(hour=9, minute=30, second=0, microsecond=0)
        return "premarket" if self.init_time_local < cutoff else "intraday"

    @property
    def cycle_token(self) -> str:
        return self.init_time_utc.strftime("%Y%m%dT%H%MZ")


def lead_hours_for_cycle(cycle_plan: CyclePlan) -> list[int]:
    if not cycle_plan.selected_target_dates:
        return list(LEAD_HOURS)
    selected_dates = set(cycle_plan.selected_target_dates)
    return [
        lead_hour
        for lead_hour in LEAD_HOURS
        if (cycle_plan.init_time_utc + dt.timedelta(hours=lead_hour)).astimezone(NY_TZ).date() in selected_dates
    ]


def total_lead_hours_for_cycles(cycle_plans: list[CyclePlan]) -> int:
    return sum(len(lead_hours_for_cycle(cycle_plan)) for cycle_plan in cycle_plans)


def lead_hours_summary(lead_hours: list[int]) -> str:
    if not lead_hours:
        return "none"
    if lead_hours == list(range(lead_hours[0], lead_hours[-1] + 1)):
        return f"f{lead_hours[0]:03d}-f{lead_hours[-1]:03d}"
    return ",".join(f"f{lead_hour:03d}" for lead_hour in lead_hours)


@dataclass(frozen=True)
class InventoryRecord:
    record_id: str
    offset: int
    init_time_utc: dt.datetime
    short_name: str
    level_text: str
    step_text: str
    extra_text: str
    raw_line: str

    @property
    def is_ensemble(self) -> bool:
        lowered = self.extra_text.lower()
        return "ens std dev" in lowered or "std dev" in lowered


@dataclass(frozen=True)
class SelectedRange:
    record: InventoryRecord
    byte_start: int
    byte_end: int

    @property
    def byte_length(self) -> int:
        return self.byte_end - self.byte_start + 1


@dataclass(frozen=True)
class MergedByteRange:
    byte_start: int
    byte_end: int
    record_count: int

    @property
    def byte_length(self) -> int:
        return self.byte_end - self.byte_start + 1


@dataclass(frozen=True)
class FieldSpec:
    feature_name: str
    short_name: str
    level_preferences: tuple[str, ...]
    output_kind: str
    optional: bool = False


@dataclass
class SelectedField:
    spec: FieldSpec
    records: list[InventoryRecord]
    fallback_used: bool = False
    fallback_source_description: str | None = None


@dataclass
class UnitResult:
    wide_row: dict[str, object] | None
    wide_rows: list[dict[str, object]]
    long_rows: list[dict[str, object]]
    provenance_rows: list[dict[str, object]]
    manifest_row: dict[str, object]


@dataclass(frozen=True)
class DatasetGridContext:
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    nearest: dict[str, object]
    north_is_first: bool


@dataclass(frozen=True)
class CropIjBox:
    i0: int
    i1: int
    j0: int
    j1: int

    @property
    def nx(self) -> int:
        return self.i1 - self.i0 + 1

    @property
    def ny(self) -> int:
        return self.j1 - self.j0 + 1

    def format_i(self) -> str:
        return f"{self.i0}:{self.i1}"

    def format_j(self) -> str:
        return f"{self.j0}:{self.j1}"

    def as_text(self) -> str:
        return f"{self.format_i()} {self.format_j()}"


@dataclass(frozen=True)
class CropGridCacheEntry:
    signature: str
    region: str
    grid_shape: tuple[int, int]
    north_is_first: bool
    west_is_first: bool
    crop_bounds: CropBounds
    ij_box: CropIjBox


@dataclass(frozen=True)
class CropExecutionResult:
    command: str
    method_used: str
    crop_grid_cache_key: str | None
    crop_grid_cache_hit: bool
    crop_ij_box: str | None
    crop_wgrib2_threads: int


@dataclass
class NbmPipelineItem:
    cycle_plan: CyclePlan
    lead_hour: int
    attempt_count: int = 1
    current_crop_bounds: CropBounds | None = None
    raw_path: pathlib.Path | None = None
    idx_path: pathlib.Path | None = None
    reduced_path: pathlib.Path | None = None
    raw_url: str | None = None
    selected_fields: list[SelectedField] = field(default_factory=list)
    manifest_row: dict[str, object] = field(default_factory=dict)


@dataclass
class NbmBatchPipelineItem:
    cycle_plan: CyclePlan
    lead_items: list[NbmPipelineItem]
    current_crop_bounds: CropBounds
    batch_raw_path: pathlib.Path
    batch_reduced_path: pathlib.Path
    batch_cfgrib_index_dir: pathlib.Path
    attempt_count: int = 1
    concat_seconds: float = 0.0
    crop_seconds: float = 0.0
    crop_result: CropExecutionResult | None = None


@dataclass
class CycleAggregationState:
    cycle_plan: CyclePlan
    expected_leads: int
    results_by_lead: dict[int, UnitResult] = field(default_factory=dict)
    completed_count: int = 0
    failed_count: int = 0
    written: bool = False


@dataclass(frozen=True)
class PhaseConcurrencyLimits:
    download_workers: int
    reduce_workers: int
    extract_workers: int
    download_semaphore: threading.Semaphore
    reduce_semaphore: threading.Semaphore
    extract_semaphore: threading.Semaphore


IDX_RE = re.compile(
    r"^(?P<record>\d+(?:\.\d+)?):(?P<offset>\d+):d=(?P<init>\d{10}):"
    r"(?P<short>[^:]+):(?P<level>[^:]+):(?P<step>[^:]+):?(?P<extra>.*)$"
)

FIELD_SPECS = (
    FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
    FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
    FieldSpec("rh", "RH", ("2 m above ground",), "wide"),
    FieldSpec("tmax", "TMAX", ("2 m above ground",), "wide", optional=True),
    FieldSpec("tmin", "TMIN", ("2 m above ground",), "wide", optional=True),
    FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
    FieldSpec("wdir", "WDIR", ("10 m above ground",), "wide"),
    FieldSpec("gust", "GUST", ("10 m above ground",), "wide"),
    FieldSpec(
        "tcdc",
        "TCDC",
        ("surface", "entire atmosphere (considered as a single layer)", "reserved"),
        "wide",
    ),
    FieldSpec("dswrf", "DSWRF", ("surface",), "wide"),
    FieldSpec("apcp", "APCP", ("surface",), "wide"),
    FieldSpec("vrate", "VRATE", ("entire atmosphere (considered as a single layer)",), "wide"),
    FieldSpec("pcpdur", "PCPDUR", ("surface",), "wide", optional=True),
    FieldSpec("vis", "VIS", ("surface",), "wide"),
    FieldSpec("ceil", "CEIL", ("cloud ceiling",), "wide"),
    FieldSpec("cape", "CAPE", ("surface",), "wide"),
    FieldSpec("thunc", "THUNC", ("entire atmosphere",), "wide", optional=True),
    FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
    FieldSpec("pwther", "PWTHER", ("surface - reserved",), "long", optional=True),
    FieldSpec("tstm", "TSTM", ("surface",), "long", optional=True),
)
OPTIONAL_FEATURE_NAMES = tuple(spec.feature_name for spec in FIELD_SPECS if spec.optional)
LONG_FEATURES_MIRRORED_TO_WIDE = frozenset({"ptype", "pwther", "tstm"})


def normalize_identifier(value: object) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


FEATURE_SPEC_BY_NAME = {spec.feature_name: spec for spec in FIELD_SPECS}
FEATURE_IDENTIFIER_ALIASES: dict[str, set[str]] = {
    "tmp": {"TMP", "2T", "T2M", "2 METRE TEMPERATURE"},
    "dpt": {"DPT", "2D", "D2M", "2 METRE DEWPOINT TEMPERATURE"},
    "rh": {"RH", "2R", "R2", "2 METRE RELATIVE HUMIDITY"},
    "tmax": {"TMAX", "MX2T", "MAXIMUM TEMPERATURE AT 2 METRES"},
    "tmin": {"TMIN", "MN2T", "MINIMUM TEMPERATURE AT 2 METRES"},
    "wind": {"WIND", "10SI", "SI10", "10 METRE WIND SPEED"},
    "wdir": {"WDIR", "10WDIR", "WDIR10", "10 METRE WIND DIRECTION"},
    "gust": {"GUST", "I10FG", "INSTANTANEOUS 10 METRE WIND GUST"},
    "tcdc": {"TCDC", "TCC", "TOTAL CLOUD COVER"},
    "dswrf": {"DSWRF", "SDSWRF", "SURFACE DOWNWARD SHORT-WAVE RADIATION FLUX"},
    "apcp": {"APCP", "TP", "TOTAL PRECIPITATION"},
    "vrate": {"VRATE", "VENTILATION RATE"},
    "pcpdur": {"PCPDUR", "PRECIPITATION DURATION"},
    "vis": {"VIS", "VISIBILITY"},
    "ceil": {"CEIL", "CEILING"},
    "cape": {"CAPE", "CONVECTIVE AVAILABLE POTENTIAL ENERGY"},
    "thunc": {"THUNC", "THUNDERSTORM COVERAGE"},
    "ptype": {"PTYPE", "PRECIPITATION TYPE"},
    "pwther": {"PWTHER", "PRESENT WEATHER"},
    "tstm": {"TSTM", "THUNDERSTORM PROBABILITY"},
}
FEATURE_IDENTIFIER_ALIASES = {
    feature_name: {normalize_identifier(alias) for alias in aliases}
    for feature_name, aliases in FEATURE_IDENTIFIER_ALIASES.items()
}

GROUP_FILTERS = (
    {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2},
    {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 10},
    {"typeOfLevel": "heightAboveGround", "stepType": "max", "level": 2},
    {"typeOfLevel": "heightAboveGround", "stepType": "min", "level": 2},
    {"typeOfLevel": "surface", "stepType": "instant"},
    {"typeOfLevel": "surface", "stepType": "accum"},
    {"typeOfLevel": "surface", "stepType": "avg"},
    {"typeOfLevel": "cloudCeiling", "stepType": "instant"},
    {"typeOfLevel": "atmosphereSingleLayer", "stepType": "instant"},
    {"typeOfLevel": "atmosphere", "stepType": "instant"},
)


def default_post_crop_manifest_counters() -> dict[str, object]:
    return {
        "cfgrib_open_all_dataset_count": 0,
        "cfgrib_filtered_fallback_open_count": 0,
        "cfgrib_filtered_fallback_attempt_count": 0,
        "cfgrib_opened_dataset_count": 0,
        "timing_row_geometry_seconds": 0.0,
        "timing_row_metric_seconds": 0.0,
        "timing_row_provenance_seconds": 0.0,
        "wide_row_count": 0,
        "long_row_count": 0,
        "provenance_row_count": 0,
        "provenance_written": True,
    }


def default_crop_manifest_fields() -> dict[str, object]:
    return {
        "crop_method_used": None,
        "crop_grid_cache_key": None,
        "crop_grid_cache_hit": False,
        "crop_ij_box": None,
        "crop_wgrib2_threads": None,
    }


def provenance_enabled(args: argparse.Namespace) -> bool:
    return not bool(getattr(args, "skip_provenance", False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill NBM GRIB2 files, subset them around KLGA/NYC, and build Parquet feature tables."
    )
    parser.add_argument("--start-local-date", required=True, help="First America/New_York date in YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--end-local-date", required=True, help="Last America/New_York date in YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--region", default="co", choices=["ak", "co", "gu", "hi", "pr"], help="NBM region code")
    parser.add_argument("--top", type=float, default=CROP_BOUNDS["top"], help="Crop top latitude")
    parser.add_argument("--bottom", type=float, default=CROP_BOUNDS["bottom"], help="Crop bottom latitude")
    parser.add_argument("--left", type=float, default=CROP_BOUNDS["left"], help="Crop left longitude")
    parser.add_argument("--right", type=float, default=CROP_BOUNDS["right"], help="Crop right longitude")
    parser.add_argument(
        "--crop-method",
        choices=CROP_METHODS,
        default=DEFAULT_CROP_METHOD,
        help="Crop primitive. Use auto to prefer cached -ijsmall_grib and fall back to -small_grib when needed.",
    )
    parser.add_argument(
        "--wgrib2-threads",
        type=int,
        help="Optional OpenMP thread count for crop subprocesses. Defaults to a safe auto policy based on active parallelism.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT, help="Root output directory")
    parser.add_argument("--scratch-dir", type=pathlib.Path, help="Optional scratch root for raw/reduced GRIBs and cfgrib indexes")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent cycle workers")
    parser.add_argument("--lead-workers", type=int, default=1, help="Concurrent lead workers within each cycle")
    parser.add_argument(
        "--download-workers",
        type=int,
        help="Concurrent lead downloads within a cycle. Defaults to lead-workers.",
    )
    parser.add_argument(
        "--reduce-workers",
        type=int,
        help="Concurrent crop/reduce steps within a cycle. Defaults to lead-workers.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        help="Concurrent open/extract steps within a cycle. Defaults to lead-workers.",
    )
    parser.add_argument(
        "--reduce-queue-size",
        type=int,
        help="Optional max size for the download->reduce queue. Defaults to 3 x reduce-workers.",
    )
    parser.add_argument(
        "--extract-queue-size",
        type=int,
        help="Optional max size for the reduce->extract queue. Defaults to 3 x extract-workers.",
    )
    parser.add_argument(
        "--range-merge-gap-bytes",
        type=int,
        default=DEFAULT_RANGE_MERGE_GAP_BYTES,
        help="Merge selected byte ranges when the gap between them is at most this many bytes.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=DEFAULT_SELECTION_MODE,
        help="Cycle-planning mode. Use overnight_0005 to keep only overnight cycles eligible by the 00:05 America/New_York cutoff.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded selected-record .grib2 and .idx files after success",
    )
    parser.add_argument("--keep-reduced", action="store_true", help="Keep reduced cropped GRIB2 files after success")
    parser.add_argument("--write-long", action="store_true", help="Write optional long-format Parquet output")
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Do not construct or write provenance rows. Intended for production overnight backfills where provenance is unused.",
    )
    parser.add_argument(
        "--metric-profile",
        choices=("full", "overnight"),
        default="full",
        help=(
            "Metric extraction profile. full writes the complete raw wide metric family; "
            "overnight computes only nearest values plus the nb/crop metrics consumed by nbm.overnight.parquet."
        ),
    )
    parser.add_argument(
        "--batch-reduce-mode",
        choices=BATCH_REDUCE_MODES,
        default="off",
        help=(
            "Batch crop/cfgrib work after selected-record downloads. off keeps the legacy per-lead "
            "reduce/extract path; cycle concatenates all selected lead GRIBs for a cycle, crops once, "
            "opens once with cfgrib, then splits rows back to per-lead results."
        ),
    )
    parser.add_argument("--crop-grib-type", default=DEFAULT_CROP_GRIB_TYPE, help=argparse.SUPPRESS)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print the cycle/lead plan without downloading data")
    parser.add_argument(
        "--progress-mode",
        choices=("auto", "dashboard", "log"),
        default="auto",
        help="Progress rendering mode. Use log to disable the live terminal dashboard even when stdout is a TTY.",
    )
    parser.add_argument(
        "--disable-dashboard-hotkeys",
        action="store_true",
        help="Disable interactive dashboard hotkeys such as 'p' for graceful drain-and-pause.",
    )
    parser.add_argument(
        "--pause-control-file",
        help="Optional file path to watch for graceful drain-and-pause requests. Create the file with `touch` to pause.",
    )
    parser.add_argument("--max-task-attempts", type=int, default=6, help="Maximum attempts per lead task including the first try.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="Base backoff in seconds for task retries.")
    parser.add_argument("--retry-max-backoff-seconds", type=float, default=30.0, help="Maximum backoff in seconds for task retries.")
    return parser.parse_args()


def ensure_runtime_dependencies() -> None:
    if resolve_wgrib2_binary() is None:
        raise SystemExit(
            f"wgrib2 is required. Install it on PATH, set {WGRIB2_ENV_VAR}, or use an active environment that provides it."
        )


def _resolve_phase_cap(requested: int | None, *, default: int, outer_limit: int) -> int:
    candidate = default if requested is None else int(requested)
    return max(1, candidate)


def build_phase_concurrency_limits(*, lead_workers: int, args: argparse.Namespace) -> PhaseConcurrencyLimits:
    outer_limit = max(1, int(lead_workers))
    download_workers = _resolve_phase_cap(
        getattr(args, "download_workers", None),
        default=outer_limit,
        outer_limit=outer_limit,
    )
    reduce_workers = _resolve_phase_cap(
        getattr(args, "reduce_workers", None),
        default=outer_limit,
        outer_limit=outer_limit,
    )
    extract_workers = _resolve_phase_cap(
        getattr(args, "extract_workers", None),
        default=outer_limit,
        outer_limit=outer_limit,
    )
    return PhaseConcurrencyLimits(
        download_workers=download_workers,
        reduce_workers=reduce_workers,
        extract_workers=extract_workers,
        download_semaphore=threading.BoundedSemaphore(download_workers),
        reduce_semaphore=threading.BoundedSemaphore(reduce_workers),
        extract_semaphore=threading.BoundedSemaphore(extract_workers),
    )


def resolve_pipeline_queue_size(requested: int | None, *, downstream_workers: int) -> int:
    default_size = max(1, int(downstream_workers) * 3)
    candidate = default_size if requested is None else int(requested)
    return max(1, candidate)


@contextlib.contextmanager
def phase_gate(
    *,
    semaphore: threading.Semaphore,
    reporter,
    worker_id: str,
    wait_phase: str,
    active_phase: str,
    details: str,
):
    if reporter is not None:
        reporter.update_worker(worker_id, phase=wait_phase, details=details)
    semaphore.acquire()
    try:
        if reporter is not None:
            reporter.update_worker(worker_id, phase=active_phase, details=details)
        yield
    finally:
        semaphore.release()


def resolve_wgrib2_binary() -> str | None:
    candidate = shutil.which("wgrib2")
    if candidate:
        return candidate
    env_override = os.environ.get(WGRIB2_ENV_VAR)
    if env_override:
        override_path = pathlib.Path(env_override).expanduser()
        if override_path.exists():
            return str(override_path)
    conda_prefix = os.environ.get("CONDA_PREFIX")
    executable_root = pathlib.Path(sys.executable).resolve().parent
    search_roots = [conda_prefix, executable_root.parent, pathlib.Path.home() / ".local"]
    for root in search_roots:
        if not root:
            continue
        path = pathlib.Path(root) / "bin" / "wgrib2"
        if path.exists():
            return str(path)
    return None


def utc_days_for_local_window(start_local: dt.date, end_local: dt.date) -> list[dt.date]:
    return iter_dates(start_local - dt.timedelta(days=1), end_local + dt.timedelta(days=1))


def overnight_cutoff_timestamp(target_date_local: dt.date) -> dt.datetime:
    return dt.datetime.combine(target_date_local, OVERNIGHT_CUTOFF_LOCAL_TIME, tzinfo=NY_TZ)


def is_overnight_cycle_for_window(init_time_local: dt.datetime, *, start_local: dt.date, end_local: dt.date) -> bool:
    for target_date_local in iter_dates(start_local, end_local):
        window_start = dt.datetime.combine(
            target_date_local - dt.timedelta(days=1),
            dt.time(hour=OVERNIGHT_INIT_START_HOUR_LOCAL),
            tzinfo=NY_TZ,
        )
        cutoff_ts = overnight_cutoff_timestamp(target_date_local)
        if window_start <= init_time_local <= cutoff_ts:
            return True
    return False


def eligible_overnight_window(
    target_date_local: dt.date,
) -> tuple[dt.datetime, dt.datetime]:
    window_start = dt.datetime.combine(
        target_date_local - dt.timedelta(days=1),
        dt.time(hour=OVERNIGHT_INIT_START_HOUR_LOCAL),
        tzinfo=NY_TZ,
    )
    cutoff_ts = overnight_cutoff_timestamp(target_date_local)
    return window_start, cutoff_ts


def discover_available_cycles(
    start_local: dt.date,
    end_local: dt.date,
    *,
    client: S3HttpClient,
    progress: ProgressBar | None = None,
) -> list[CyclePlan]:
    plans: list[CyclePlan] = []
    seen: set[tuple[dt.date, str]] = set()
    for index, utc_day in enumerate(utc_days_for_local_window(start_local, end_local), start=1):
        if progress is not None:
            progress.update(completed=index - 1, stage="discover", status=f"utc_day={utc_day.isoformat()}")
        for cycle in list_grib2_cycles(utc_day, client):
            init_time_utc = dt.datetime.combine(utc_day, dt.time(hour=int(cycle)), tzinfo=UTC)
            init_time_local = init_time_utc.astimezone(NY_TZ)
            key = (utc_day, cycle)
            if key in seen:
                continue
            seen.add(key)
            plans.append(CyclePlan(init_time_utc=init_time_utc, init_time_local=init_time_local, cycle=cycle))
        if progress is not None:
            progress.advance(stage="discover", status=f"utc_day={utc_day.isoformat()} done")
    return sorted(plans, key=lambda item: item.init_time_utc)


def discover_cycle_plans(
    start_local: dt.date,
    end_local: dt.date,
    *,
    client: S3HttpClient,
    progress: ProgressBar | None = None,
    selection_mode: str = DEFAULT_SELECTION_MODE,
) -> list[CyclePlan]:
    available_cycles = discover_available_cycles(start_local, end_local, client=client, progress=progress)
    if selection_mode == "overnight_0005":
        selected_target_dates_by_issue: dict[tuple[dt.datetime, str], list[dt.date]] = {}
        selected_cycle_by_issue: dict[tuple[dt.datetime, str], CyclePlan] = {}
        for target_date_local in iter_dates(start_local, end_local):
            window_start, cutoff_ts = eligible_overnight_window(target_date_local)
            eligible_cycles = [
                cycle_plan
                for cycle_plan in available_cycles
                if window_start <= cycle_plan.init_time_local <= cutoff_ts
            ]
            if not eligible_cycles:
                continue
            selected_cycle = eligible_cycles[-1]
            issue_key = (selected_cycle.init_time_utc, selected_cycle.cycle)
            selected_cycle_by_issue[issue_key] = selected_cycle
            selected_target_dates_by_issue.setdefault(issue_key, []).append(target_date_local)
        plans = [
            CyclePlan(
                init_time_utc=cycle_plan.init_time_utc,
                init_time_local=cycle_plan.init_time_local,
                cycle=cycle_plan.cycle,
                selected_target_dates=tuple(sorted(selected_target_dates_by_issue[issue_key])),
            )
            for issue_key, cycle_plan in sorted(selected_cycle_by_issue.items(), key=lambda item: item[1].init_time_utc)
        ]
        return plans
    return [
        cycle_plan
        for cycle_plan in available_cycles
        if start_local <= cycle_plan.init_time_local.date() <= end_local
    ]


def stdout_is_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def parse_idx_lines(text: str) -> list[InventoryRecord]:
    records: list[InventoryRecord] = []
    for line in text.splitlines():
        match = IDX_RE.match(line.strip())
        if not match:
            continue
        info = match.groupdict()
        init_time = dt.datetime.strptime(info["init"], "%Y%m%d%H").replace(tzinfo=UTC)
        records.append(
            InventoryRecord(
                record_id=info["record"],
                offset=int(info["offset"]),
                init_time_utc=init_time,
                short_name=info["short"],
                level_text=info["level"],
                step_text=info["step"],
                extra_text=info["extra"].strip(":"),
                raw_line=line.strip(),
            )
        )
    return records


def _record_matches_level(record: InventoryRecord, level_text: str) -> bool:
    return record.level_text == level_text


def select_inventory_records(records: list[InventoryRecord]) -> tuple[list[SelectedField], list[str], list[str]]:
    selected: list[SelectedField] = []
    warnings: list[str] = []
    missing_required: list[str] = []

    for spec in FIELD_SPECS:
        matching = [record for record in records if record.short_name == spec.short_name and not record.is_ensemble]
        if spec.output_kind == "wide":
            chosen: InventoryRecord | None = None
            fallback_used = False
            fallback_source_description: str | None = None
            for level_text in spec.level_preferences:
                exact_matches = [record for record in matching if _record_matches_level(record, level_text)]
                exact_matches = [record for record in exact_matches if not record.extra_text]
                if exact_matches:
                    chosen = exact_matches[0]
                    break
                if exact_matches := [record for record in matching if _record_matches_level(record, level_text)]:
                    chosen = exact_matches[0]
                    fallback_used = True
                    fallback_source_description = (
                        f"matched level '{level_text}' using non-empty extra_text '{chosen.extra_text}'"
                    )
                    break
            if chosen is None:
                message = f"missing field {spec.short_name} levels={','.join(spec.level_preferences)}"
                if spec.optional:
                    warnings.append(message)
                    continue
                missing_required.append(message)
                continue
            preferred_level = spec.level_preferences[0]
            if chosen.level_text != preferred_level:
                fallback_used = True
                fallback_source_description = f"used fallback level '{chosen.level_text}' instead of '{preferred_level}'"
            selected.append(
                SelectedField(
                    spec=spec,
                    records=[chosen],
                    fallback_used=fallback_used,
                    fallback_source_description=fallback_source_description,
                )
            )
            continue

        long_records: list[InventoryRecord] = []
        fallback_used = False
        fallback_source_description: str | None = None
        for level_text in spec.level_preferences:
            level_records = [record for record in matching if _record_matches_level(record, level_text)]
            if level_records:
                long_records = level_records
                if level_text != spec.level_preferences[0]:
                    fallback_used = True
                    fallback_source_description = (
                        f"used fallback level '{level_text}' instead of '{spec.level_preferences[0]}'"
                    )
                break
        if long_records:
            selected.append(
                SelectedField(
                    spec=spec,
                    records=long_records,
                    fallback_used=fallback_used,
                    fallback_source_description=fallback_source_description,
                )
            )
        elif spec.optional:
            warnings.append(f"missing field {spec.short_name} levels={','.join(spec.level_preferences)}")
        else:
            missing_required.append(f"missing field {spec.short_name} levels={','.join(spec.level_preferences)}")

    return selected, warnings, missing_required


def inventory_subset_text(selected: list[SelectedField]) -> str:
    lines: list[str] = []
    for field in selected:
        for record in field.records:
            lines.append(record.raw_line)
    return "\n".join(lines) + ("\n" if lines else "")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_command(command: list[str], *, input_text: str | None = None, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        input=input_text,
        text=True,
        capture_output=True,
        check=True,
        env=env,
    )
    return completed.stdout if completed.stdout else completed.stderr


def build_selected_ranges(
    records: list[InventoryRecord],
    selected: list[SelectedField],
    *,
    content_length: int,
) -> list[SelectedRange]:
    indexed_ranges: list[SelectedRange] = []
    selected_raw_lines = {
        record.raw_line
        for field in selected
        for record in field.records
    }
    for index, record in enumerate(records):
        if record.raw_line not in selected_raw_lines:
            continue
        next_offset = records[index + 1].offset if index + 1 < len(records) else content_length
        indexed_ranges.append(
            SelectedRange(
                record=record,
                byte_start=record.offset,
                byte_end=next_offset - 1,
            )
        )
    return indexed_ranges


def selected_ranges_require_content_length(
    records: list[InventoryRecord],
    selected: list[SelectedField],
) -> bool:
    if not records:
        return False
    selected_raw_lines = {
        record.raw_line
        for field in selected
        for record in field.records
    }
    return records[-1].raw_line in selected_raw_lines


def merge_selected_ranges(
    ranges: list[SelectedRange],
    *,
    max_gap_bytes: int,
) -> list[MergedByteRange]:
    if not ranges:
        return []
    merged: list[MergedByteRange] = []
    current_start = ranges[0].byte_start
    current_end = ranges[0].byte_end
    current_count = 1
    allowed_gap = max(0, int(max_gap_bytes))
    for item in ranges[1:]:
        gap_bytes = item.byte_start - current_end - 1
        if gap_bytes <= allowed_gap:
            current_end = max(current_end, item.byte_end)
            current_count += 1
            continue
        merged.append(
            MergedByteRange(
                byte_start=current_start,
                byte_end=current_end,
                record_count=current_count,
            )
        )
        current_start = item.byte_start
        current_end = item.byte_end
        current_count = 1
    merged.append(
        MergedByteRange(
            byte_start=current_start,
            byte_end=current_end,
            record_count=current_count,
        )
    )
    return merged


def stage_elapsed_seconds(started_at: float) -> float:
    return round(max(0.0, time.perf_counter() - started_at), 6)


def should_reuse_cached_raw_grib(
    *,
    raw_path: pathlib.Path,
    idx_was_present: bool,
    overwrite: bool,
    expected_selected_bytes: int,
) -> bool:
    if overwrite or not idx_was_present or not raw_path.exists():
        return False
    try:
        return raw_path.stat().st_size == expected_selected_bytes
    except OSError:
        return False


def scratch_root(args: argparse.Namespace) -> pathlib.Path | None:
    value = getattr(args, "scratch_dir", None)
    if value is None:
        return None
    return pathlib.Path(value)


def raw_root(args: argparse.Namespace) -> pathlib.Path:
    root = scratch_root(args)
    if root is not None:
        return root / "raw"
    return args.output_dir / "raw"


def reduced_root(args: argparse.Namespace, cycle_plan: CyclePlan) -> pathlib.Path:
    root = scratch_root(args)
    base = (root / "reduced") if root is not None else (args.output_dir / "reduced")
    return base / f"init_date_local={cycle_plan.init_date_local.isoformat()}"


def cfgrib_index_base_dir(args: argparse.Namespace) -> pathlib.Path | None:
    root = scratch_root(args)
    if root is None:
        return None
    return root / "cfgrib_index"


def reduced_signature_path(reduced_path: pathlib.Path) -> pathlib.Path:
    return reduced_path.with_suffix(f"{reduced_path.suffix}.reuse.json")


def selected_records_identity_hash(selected: list[SelectedField]) -> str:
    payload = [
        record.raw_line
        for field in selected
        for record in field.records
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=False).encode("utf-8")).hexdigest()


def build_reduced_reuse_signature(
    *,
    raw_path: pathlib.Path,
    cycle_plan: CyclePlan,
    lead_hour: int,
    crop_bounds: CropBounds,
    region: str,
    idx_sha256: str | None,
    selected_records_hash: str | None,
) -> str:
    payload = {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "cycle_token": cycle_plan.cycle_token,
        "lead_hour": int(lead_hour),
        "region": str(region),
        "raw_file_name": raw_path.name,
        "raw_file_size": raw_path.stat().st_size if raw_path.exists() else None,
        "idx_sha256": idx_sha256,
        "selected_records_hash": selected_records_hash,
        "crop_bounds": {
            "top": float(crop_bounds.top),
            "bottom": float(crop_bounds.bottom),
            "left": float(crop_bounds.left),
            "right": float(crop_bounds.right),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def reduced_grib_reusable(*, reduced_path: pathlib.Path, signature: str) -> bool:
    if not reduced_path.exists():
        return False
    signature_path = reduced_signature_path(reduced_path)
    if not signature_path.exists():
        return False
    try:
        payload = json.loads(signature_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("signature") == signature


def write_reduced_reuse_signature(reduced_path: pathlib.Path, signature: str) -> None:
    reduced_signature_path(reduced_path).write_text(json.dumps({"signature": signature}, sort_keys=True))


def cfgrib_index_cache_dir(
    *,
    args: argparse.Namespace,
    reduced_path: pathlib.Path,
    reuse_signature: str | None,
) -> pathlib.Path | None:
    if not getattr(args, "keep_reduced", False) or not reuse_signature:
        return None
    base_dir = cfgrib_index_base_dir(args)
    if base_dir is None:
        base_dir = reduced_path.parent / ".cfgrib_index"
    return base_dir / reuse_signature


def crop_grid_cache_root(args: argparse.Namespace, *, reduced_path: pathlib.Path) -> pathlib.Path:
    root = scratch_root(args)
    if root is not None:
        return root / "crop_grid_cache"
    return reduced_path.parent / ".crop_grid_cache"


def crop_grid_cache_lock(cache_path: pathlib.Path) -> threading.Lock:
    with CROP_GRID_CACHE_LOCKS_GUARD:
        lock = CROP_GRID_CACHE_LOCKS.get(cache_path)
        if lock is None:
            lock = threading.Lock()
            CROP_GRID_CACHE_LOCKS[cache_path] = lock
        return lock


def normalize_crop_longitudes_for_grid(bounds: CropBounds, lon_grid: np.ndarray) -> tuple[float, float]:
    if float(np.nanmax(lon_grid)) > 180:
        return float(bounds.left), float(bounds.right)
    return float(longitude_360_to_180(bounds.left)), float(longitude_360_to_180(bounds.right))


def grid_signature_payload(
    *,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    north_is_first: bool,
    west_is_first: bool,
) -> dict[str, object]:
    lat_flat = lat_grid.reshape(-1)
    lon_flat = lon_grid.reshape(-1)
    mid_index = int(lat_flat.size // 2) if lat_flat.size else 0
    return {
        "grid_shape": [int(lat_grid.shape[0]), int(lat_grid.shape[1])],
        "north_is_first": bool(north_is_first),
        "west_is_first": bool(west_is_first),
        "lat_sample": [
            float(lat_flat[0]),
            float(lat_flat[mid_index]),
            float(lat_flat[-1]),
        ],
        "lon_sample": [
            float(lon_flat[0]),
            float(lon_flat[mid_index]),
            float(lon_flat[-1]),
        ],
        "lat_min": float(np.nanmin(lat_grid)),
        "lat_max": float(np.nanmax(lat_grid)),
        "lon_min": float(np.nanmin(lon_grid)),
        "lon_max": float(np.nanmax(lon_grid)),
    }


def build_crop_grid_cache_key(
    *,
    region: str,
    bounds: CropBounds,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    north_is_first: bool,
    west_is_first: bool,
) -> str:
    payload = {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "region": str(region),
        "crop_bounds": {
            "top": float(bounds.top),
            "bottom": float(bounds.bottom),
            "left": float(bounds.left),
            "right": float(bounds.right),
        },
        **grid_signature_payload(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            north_is_first=north_is_first,
            west_is_first=west_is_first,
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def crop_grid_cache_path(args: argparse.Namespace, *, reduced_path: pathlib.Path, cache_key: str) -> pathlib.Path:
    root = crop_grid_cache_root(args, reduced_path=reduced_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.json"


def crop_grid_negative_cache_path(args: argparse.Namespace, *, reduced_path: pathlib.Path, cache_key: str) -> pathlib.Path:
    root = crop_grid_cache_root(args, reduced_path=reduced_path)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{cache_key}.unsupported.json"


def crop_grid_index_dir(args: argparse.Namespace, *, reduced_path: pathlib.Path, cache_key: str) -> pathlib.Path:
    return crop_grid_cache_root(args, reduced_path=reduced_path) / ".cfgrib_index" / cache_key


def raw_grid_identity_payload(
    *,
    raw_path: pathlib.Path,
    region: str,
    bounds: CropBounds,
) -> dict[str, object]:
    sample = b""
    try:
        with raw_path.open("rb") as handle:
            sample = handle.read(4096)
    except OSError:
        sample = b""
    return {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "region": str(region),
        "raw_file_name": raw_path.name,
        "raw_file_size": raw_path.stat().st_size if raw_path.exists() else None,
        "raw_prefix_sha256": hashlib.sha256(sample).hexdigest(),
        "crop_bounds": {
            "top": float(bounds.top),
            "bottom": float(bounds.bottom),
            "left": float(bounds.left),
            "right": float(bounds.right),
        },
    }


def raw_grid_identity_key(
    *,
    raw_path: pathlib.Path,
    region: str,
    bounds: CropBounds,
) -> str:
    payload = raw_grid_identity_payload(raw_path=raw_path, region=region, bounds=bounds)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def raw_grid_identity_cache_path(
    args: argparse.Namespace,
    *,
    raw_path: pathlib.Path,
    reduced_path: pathlib.Path,
    bounds: CropBounds,
) -> pathlib.Path:
    cache_root = crop_grid_cache_root(args, reduced_path=reduced_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"raw_identity_{raw_grid_identity_key(raw_path=raw_path, region=args.region, bounds=bounds)}.json"


def load_raw_grid_identity_cache(cache_path: pathlib.Path) -> str | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    cache_key = payload.get("cache_key")
    if not cache_key:
        return None
    return str(cache_key)


def write_raw_grid_identity_cache(cache_path: pathlib.Path, *, cache_key: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps({"cache_key": cache_key}, sort_keys=True))
    tmp_path.replace(cache_path)


def serialize_crop_grid_cache_entry(entry: CropGridCacheEntry) -> str:
    payload = {
        "signature": entry.signature,
        "region": entry.region,
        "grid_shape": [int(entry.grid_shape[0]), int(entry.grid_shape[1])],
        "north_is_first": bool(entry.north_is_first),
        "west_is_first": bool(entry.west_is_first),
        "ij_box": {
            "i0": int(entry.ij_box.i0),
            "i1": int(entry.ij_box.i1),
            "j0": int(entry.ij_box.j0),
            "j1": int(entry.ij_box.j1),
        },
        "crop_bounds": {
            "top": float(entry.crop_bounds.top),
            "bottom": float(entry.crop_bounds.bottom),
            "left": float(entry.crop_bounds.left),
            "right": float(entry.crop_bounds.right),
        },
        "created_by": SOURCE_VERSION,
    }
    return json.dumps(payload, sort_keys=True)


def parse_crop_grid_cache_entry(text: str) -> CropGridCacheEntry:
    payload = json.loads(text)
    ij_box = payload["ij_box"]
    grid_shape = payload["grid_shape"]
    crop_bounds = payload["crop_bounds"]
    return CropGridCacheEntry(
        signature=str(payload["signature"]),
        region=str(payload["region"]),
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        north_is_first=bool(payload["north_is_first"]),
        west_is_first=bool(payload["west_is_first"]),
        crop_bounds=CropBounds(
            top=float(crop_bounds["top"]),
            bottom=float(crop_bounds["bottom"]),
            left=float(crop_bounds["left"]),
            right=float(crop_bounds["right"]),
        ),
        ij_box=CropIjBox(
            i0=int(ij_box["i0"]),
            i1=int(ij_box["i1"]),
            j0=int(ij_box["j0"]),
            j1=int(ij_box["j1"]),
        ),
    )


def write_crop_grid_cache_entry(cache_path: pathlib.Path, entry: CropGridCacheEntry) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(serialize_crop_grid_cache_entry(entry))
    tmp_path.replace(cache_path)


def load_crop_grid_cache_entry(cache_path: pathlib.Path) -> CropGridCacheEntry | None:
    if not cache_path.exists():
        return None
    try:
        return parse_crop_grid_cache_entry(cache_path.read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def load_crop_grid_negative_cache(cache_path: pathlib.Path) -> str | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable_negative_cache"
    reason = payload.get("reason")
    return str(reason) if reason else "ijsmall_grib_disabled"


def write_crop_grid_negative_cache(cache_path: pathlib.Path, *, reason: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps({"reason": str(reason), "created_by": SOURCE_VERSION}, sort_keys=True))
    tmp_path.replace(cache_path)


def representative_grid_context_for_path(
    path: pathlib.Path,
    *,
    index_dir: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    datasets = open_grouped_datasets(path, index_dir=index_dir, selected=None)
    try:
        if not datasets:
            raise ValueError(f"No datasets available to infer crop grid for {path}")
        first_dataset = datasets[0]
        if not first_dataset.data_vars:
            raise ValueError(f"Representative dataset for {path} had no data variables")
        first_data_array = next(iter(first_dataset.data_vars.values()))
        lat_grid, lon_grid = extract_lat_lon(first_data_array)
        north_is_first = infer_north_is_first(lat_grid)
        west_is_first = infer_west_is_first(lon_grid)
        return lat_grid, lon_grid, north_is_first, west_is_first
    finally:
        for dataset in datasets:
            dataset.close()


def crop_ij_box_from_grid(
    *,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    bounds: CropBounds,
    north_is_first: bool,
    west_is_first: bool,
) -> CropIjBox:
    left, right = normalize_crop_longitudes_for_grid(bounds, lon_grid)
    mask = (
        np.isfinite(lat_grid)
        & np.isfinite(lon_grid)
        & (lat_grid >= float(bounds.bottom))
        & (lat_grid <= float(bounds.top))
        & (lon_grid >= left)
        & (lon_grid <= right)
    )
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Crop bounds do not include any grid points for ijsmall_grib")
    row_min = int(rows.min())
    row_max = int(rows.max())
    col_min = int(cols.min())
    col_max = int(cols.max())
    nrows, ncols = lat_grid.shape
    if west_is_first:
        i0 = col_min + 1
        i1 = col_max + 1
    else:
        i0 = ncols - col_max
        i1 = ncols - col_min
    if north_is_first:
        j0 = nrows - row_max
        j1 = nrows - row_min
    else:
        j0 = row_min + 1
        j1 = row_max + 1
    return CropIjBox(i0=int(i0), i1=int(i1), j0=int(j0), j1=int(j1))


def resolve_crop_grid_cache_entry(
    *,
    args: argparse.Namespace,
    raw_path: pathlib.Path,
    reduced_path: pathlib.Path,
    bounds: CropBounds,
) -> tuple[CropGridCacheEntry, bool]:
    cache_root = crop_grid_cache_root(args, reduced_path=reduced_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    identity_cache_path = raw_grid_identity_cache_path(
        args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        bounds=bounds,
    )
    cached_identity_key = load_raw_grid_identity_cache(identity_cache_path)
    if cached_identity_key:
        cache_path = crop_grid_cache_path(args, reduced_path=reduced_path, cache_key=cached_identity_key)
        cached_entry = load_crop_grid_cache_entry(cache_path)
        if cached_entry is not None and cached_entry.signature == cached_identity_key:
            return cached_entry, True
    probe_index_dir = crop_grid_index_dir(
        args,
        reduced_path=reduced_path,
        cache_key=f"probe_{raw_grid_identity_key(raw_path=raw_path, region=args.region, bounds=bounds)}",
    )
    probe_index_dir.mkdir(parents=True, exist_ok=True)
    lat_grid, lon_grid, north_is_first, west_is_first = representative_grid_context_for_path(raw_path, index_dir=probe_index_dir)
    cache_key = build_crop_grid_cache_key(
        region=args.region,
        bounds=bounds,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        north_is_first=north_is_first,
        west_is_first=west_is_first,
    )
    cache_path = crop_grid_cache_path(args, reduced_path=reduced_path, cache_key=cache_key)
    lock = crop_grid_cache_lock(cache_path)
    with lock:
        cached = load_crop_grid_cache_entry(cache_path)
        if cached is not None and cached.signature == cache_key:
            return cached, True
        ij_box = crop_ij_box_from_grid(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            bounds=bounds,
            north_is_first=north_is_first,
            west_is_first=west_is_first,
        )
        entry = CropGridCacheEntry(
            signature=cache_key,
            region=str(args.region),
            grid_shape=(int(lat_grid.shape[0]), int(lat_grid.shape[1])),
            north_is_first=bool(north_is_first),
            west_is_first=bool(west_is_first),
            crop_bounds=bounds,
            ij_box=ij_box,
        )
        write_crop_grid_cache_entry(cache_path, entry)
        write_raw_grid_identity_cache(identity_cache_path, cache_key=cache_key)
        return entry, False


def crop_wgrib2_thread_count(args: argparse.Namespace) -> int:
    explicit = getattr(args, "wgrib2_threads", None)
    if explicit is not None:
        return max(1, int(explicit))
    outer_parallelism = max(1, int(getattr(args, "workers", 1)))
    reduce_parallelism = max(1, int(getattr(args, "reduce_workers", 1) or 1))
    return 1 if (outer_parallelism > 1 or reduce_parallelism > 1) else 2


def crop_wgrib2_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(crop_wgrib2_thread_count(args))
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    return env


def read_wgrib2_grid_shape(path: pathlib.Path, *, env: dict[str, str] | None = None) -> tuple[int, int]:
    wgrib2_binary = resolve_wgrib2_binary()
    if wgrib2_binary is None:
        raise SystemExit("wgrib2 is required but could not be located.")
    output = run_command([wgrib2_binary, str(path), "-grid"], env=env)
    normalized_output = output.replace("\n", " ")
    match = None
    for pattern in GRID_SHAPE_PATTERNS:
        match = pattern.search(normalized_output)
        if match is not None:
            break
    if match is None:
        excerpt = normalized_output[:500].strip()
        raise ValueError(f"Unable to parse wgrib2 -grid output for {path}: {excerpt}")
    return int(match.group("nx")), int(match.group("ny"))


def crop_selected_grib2(
    *,
    args: argparse.Namespace,
    raw_path: pathlib.Path,
    reduced_path: pathlib.Path,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> CropExecutionResult:
    wgrib2_binary = resolve_wgrib2_binary()
    if wgrib2_binary is None:
        raise SystemExit("wgrib2 is required but could not be located.")
    crop_bounds = CropBounds(top=float(top), bottom=float(bottom), left=float(left), right=float(right))
    wgrib2_threads = crop_wgrib2_thread_count(args)
    crop_env = crop_wgrib2_env(args)

    def _run_small_grib() -> CropExecutionResult:
        crop_command = [
            wgrib2_binary,
            str(raw_path),
            "-set_grib_type",
            str(getattr(args, "crop_grib_type", DEFAULT_CROP_GRIB_TYPE)),
            "-small_grib",
            f"{left}:{right}",
            f"{bottom}:{top}",
            str(reduced_path),
        ]
        run_command(crop_command, env=crop_env)
        return CropExecutionResult(
            command=" ".join(crop_command),
            method_used="small_grib",
            crop_grid_cache_key=None,
            crop_grid_cache_hit=False,
            crop_ij_box=None,
            crop_wgrib2_threads=wgrib2_threads,
        )

    method = str(getattr(args, "crop_method", DEFAULT_CROP_METHOD))
    if method == "small_grib":
        return _run_small_grib()

    cache_key: str | None = None
    cache_hit = False
    crop_ij_box: CropIjBox | None = None
    try:
        cache_entry, cache_hit = resolve_crop_grid_cache_entry(
            args=args,
            raw_path=raw_path,
            reduced_path=reduced_path,
            bounds=crop_bounds,
        )
        cache_key = cache_entry.signature
        crop_ij_box = cache_entry.ij_box
    except Exception:
        if method == "ijsmall_grib":
            raise

    if crop_ij_box is None:
        return _run_small_grib()
    negative_cache_path = crop_grid_negative_cache_path(args, reduced_path=reduced_path, cache_key=cache_key or "")
    if method == "auto" and cache_key is not None and load_crop_grid_negative_cache(negative_cache_path) is not None:
        return _run_small_grib()

    crop_command = [
        wgrib2_binary,
        str(raw_path),
        "-set_grib_type",
        str(getattr(args, "crop_grib_type", DEFAULT_CROP_GRIB_TYPE)),
        "-ijsmall_grib",
        crop_ij_box.format_i(),
        crop_ij_box.format_j(),
        str(reduced_path),
    ]
    try:
        run_command(crop_command, env=crop_env)
        actual_grid_shape = read_wgrib2_grid_shape(reduced_path, env=crop_env)
        expected_grid_shape = (crop_ij_box.nx, crop_ij_box.ny)
        if actual_grid_shape != expected_grid_shape:
            raise ValueError(
                f"ijsmall_grib crop produced grid {actual_grid_shape} but expected {expected_grid_shape}"
            )
        return CropExecutionResult(
            command=" ".join(crop_command),
            method_used="ijsmall_grib",
            crop_grid_cache_key=cache_key,
            crop_grid_cache_hit=cache_hit,
            crop_ij_box=crop_ij_box.as_text(),
            crop_wgrib2_threads=wgrib2_threads,
        )
    except Exception:
        if method == "ijsmall_grib":
            raise
        if cache_key is not None:
            write_crop_grid_negative_cache(negative_cache_path, reason="ijsmall_grib_crop_or_validation_failed")
        return _run_small_grib()


def coerce_crop_execution_result(result: object, *, args: argparse.Namespace) -> CropExecutionResult:
    if isinstance(result, CropExecutionResult):
        return result
    return CropExecutionResult(
        command=str(result),
        method_used="small_grib" if str(getattr(args, "crop_method", DEFAULT_CROP_METHOD)) == "small_grib" else "legacy_stub",
        crop_grid_cache_key=None,
        crop_grid_cache_hit=False,
        crop_ij_box=None,
        crop_wgrib2_threads=crop_wgrib2_thread_count(args),
    )


def default_batch_manifest_fields() -> dict[str, object]:
    return {
        "batch_reduce_mode": "off",
        "batch_raw_file_path": None,
        "batch_reduced_file_path": None,
        "batch_lead_count": 1,
        "batch_concat_seconds": 0.0,
        "batch_crop_seconds": 0.0,
        "batch_cfgrib_open_seconds": 0.0,
        "batch_row_build_seconds": 0.0,
        "batch_timing_policy": "per_lead",
    }


def selected_required_group_filter_indexes(selected: list[SelectedField] | None) -> set[int]:
    if not selected:
        return set(range(len(GROUP_FILTERS)))
    return set(selected_required_features_by_group_filter_index(selected))


def selected_required_features_by_group_filter_index(selected: list[SelectedField]) -> dict[int, set[str]]:
    required_features: dict[int, set[str]] = {}
    for selected_field in selected:
        for record in selected_field.records:
            type_of_level = expected_type_of_level(record.level_text)
            step_type = expected_step_type(record.step_text)
            level_value = expected_level_value(record.level_text)
            for filter_index, filter_keys in enumerate(GROUP_FILTERS):
                if filter_keys.get("typeOfLevel") != type_of_level:
                    continue
                if filter_keys.get("stepType") != step_type:
                    continue
                if "level" in filter_keys:
                    if level_value is None:
                        continue
                    try:
                        if int(filter_keys["level"]) != int(level_value):
                            continue
                    except (TypeError, ValueError):
                        continue
                required_features.setdefault(filter_index, set()).add(selected_field.spec.feature_name)
    return required_features


def open_grouped_datasets(
    path: pathlib.Path,
    *,
    index_dir: pathlib.Path | None = None,
    selected: list[SelectedField] | None = None,
    stats: dict[str, object] | None = None,
) -> list[xr.Dataset]:
    datasets: list[xr.Dataset] = []
    seen_signatures: set[tuple[tuple[str, str | None, str | None, str | None, str | None], ...]] = set()
    if stats is not None:
        stats.update(
            {
                "cfgrib_open_all_dataset_count": 0,
                "cfgrib_filtered_fallback_open_count": 0,
                "cfgrib_filtered_fallback_attempt_count": 0,
                "cfgrib_opened_dataset_count": 0,
            }
        )
    if index_dir is not None:
        index_dir.mkdir(parents=True, exist_ok=True)

    def _data_array_level_value_for_signature(data_array: xr.DataArray) -> object | None:
        level_value = data_array.attrs.get("GRIB_level")
        if level_value is not None:
            return level_value
        type_of_level = data_array.attrs.get("GRIB_typeOfLevel")
        for coord_name in filter(None, [type_of_level, "heightAboveGround", "surface", "cloudCeiling", "atmosphere", "atmosphereSingleLayer"]):
            coord = data_array.coords.get(str(coord_name))
            if coord is not None and coord.size:
                value = np.asarray(coord.values).reshape(-1)[0]
                return value.item() if hasattr(value, "item") else value
        return None

    def dataset_signature(dataset: xr.Dataset) -> tuple[tuple[str, str | None, str | None, str | None, str | None], ...]:
        return tuple(
            sorted(
                (
                    str(var_name),
                    data_array.attrs.get("GRIB_shortName"),
                    data_array.attrs.get("GRIB_typeOfLevel"),
                    data_array.attrs.get("GRIB_stepType"),
                    str(_data_array_level_value_for_signature(data_array)),
                )
                for var_name, data_array in dataset.data_vars.items()
            )
        )

    xarray_options = {"use_new_combine_kwarg_defaults": True} if hasattr(xr, "set_options") else {}

    def _open_dataset_with_filters(filter_keys: dict[str, object]) -> xr.Dataset:
        indexpath = ""
        if index_dir is not None:
            key = "__".join(f"{name}={filter_keys[name]}" for name in sorted(filter_keys))
            indexpath = str(index_dir / f"{key}.idx")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*default value for compat will change from compat='no_conflicts' to compat='override'.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"cfgrib\.xarray_store",
            )
            with xr.set_options(**xarray_options) if xarray_options else contextlib.nullcontext():
                return xr.open_dataset(
                    path,
                    engine="cfgrib",
                    backend_kwargs={"filter_by_keys": filter_keys, "indexpath": indexpath},
                )

    def _open_all_datasets() -> list[xr.Dataset]:
        indexpath = str(index_dir / "all.idx") if index_dir is not None else ""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*default value for compat will change from compat='no_conflicts' to compat='override'.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                module=r"cfgrib\.xarray_store",
            )
            with xr.set_options(**xarray_options) if xarray_options else contextlib.nullcontext():
                return cfgrib.open_datasets(str(path), indexpath=indexpath)

    def _dataset_matches_filter(dataset: xr.Dataset, filter_keys: dict[str, object]) -> bool:
        if not dataset.data_vars:
            return False
        first_data_array = next(iter(dataset.data_vars.values()))
        if first_data_array.attrs.get("GRIB_typeOfLevel") != filter_keys.get("typeOfLevel"):
            return False
        if first_data_array.attrs.get("GRIB_stepType") != filter_keys.get("stepType"):
            return False
        if "level" not in filter_keys:
            return True
        level_value = first_data_array.attrs.get("GRIB_level")
        if level_value is None:
            coord = dataset.coords.get(str(filter_keys["typeOfLevel"]))
            if coord is not None and coord.size:
                level_value = np.asarray(coord.values).reshape(-1)[0]
        try:
            return int(level_value) == int(filter_keys["level"])
        except (TypeError, ValueError):
            return False

    covered_filter_indexes: set[int] = set()
    required_filter_indexes = selected_required_group_filter_indexes(selected)
    required_features_by_filter_index = selected_required_features_by_group_filter_index(selected or [])
    selected_fields_by_feature = selected_records_by_feature_name(selected or [])
    observed_features_by_filter_index: dict[int, set[str]] = {}

    def _record_observed_features(dataset: xr.Dataset, filter_index: int) -> None:
        observed = observed_features_by_filter_index.setdefault(filter_index, set())
        for var_name, data_array in dataset.data_vars.items():
            feature_name = canonical_feature_name(var_name, data_array)
            if feature_name is None:
                continue
            required_features = required_features_by_filter_index.get(filter_index)
            if required_features and feature_name not in required_features:
                continue
            selected_field = selected_fields_by_feature.get(feature_name)
            if selected_field is not None and not matching_selected_records_for_data_array(selected_field.records, data_array):
                continue
            observed.add(feature_name)

    def _filter_is_fully_covered(filter_index: int) -> bool:
        if filter_index not in covered_filter_indexes:
            return False
        required_features = required_features_by_filter_index.get(filter_index)
        if not required_features:
            return True
        return required_features <= observed_features_by_filter_index.get(filter_index, set())

    try:
        opened_datasets = _open_all_datasets()
        if stats is not None:
            stats["cfgrib_open_all_dataset_count"] = len(opened_datasets)
        for dataset in opened_datasets:
            if not dataset.data_vars:
                dataset.close()
                continue
            signature = dataset_signature(dataset)
            if signature in seen_signatures:
                dataset.close()
                continue
            datasets.append(dataset)
            seen_signatures.add(signature)
            for filter_index, filter_keys in enumerate(GROUP_FILTERS):
                if _dataset_matches_filter(dataset, filter_keys):
                    covered_filter_indexes.add(filter_index)
                    _record_observed_features(dataset, filter_index)
    except Exception:
        pass
    for filter_index, filter_keys in enumerate(GROUP_FILTERS):
        if filter_index not in required_filter_indexes:
            continue
        if _filter_is_fully_covered(filter_index):
            continue
        if stats is not None:
            stats["cfgrib_filtered_fallback_attempt_count"] = int(stats["cfgrib_filtered_fallback_attempt_count"]) + 1
        try:
            dataset = _open_dataset_with_filters(filter_keys)
        except Exception:
            continue
        if dataset.data_vars:
            signature = dataset_signature(dataset)
            if signature not in seen_signatures:
                datasets.append(dataset)
                if stats is not None:
                    stats["cfgrib_filtered_fallback_open_count"] = int(stats["cfgrib_filtered_fallback_open_count"]) + 1
                seen_signatures.add(signature)
                _record_observed_features(dataset, filter_index)
            else:
                _record_observed_features(dataset, filter_index)
                dataset.close()
        else:
            dataset.close()
    if stats is not None:
        stats["cfgrib_opened_dataset_count"] = len(datasets)
    return datasets


def extract_lat_lon(data_array: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat_coord = data_array.coords.get("latitude")
    if lat_coord is None:
        lat_coord = data_array.coords.get("lat")
    lon_coord = data_array.coords.get("longitude")
    if lon_coord is None:
        lon_coord = data_array.coords.get("lon")
    if lat_coord is None or lon_coord is None:
        raise ValueError("cfgrib dataset is missing latitude/longitude coordinates")
    lat_values = np.asarray(lat_coord.values)
    lon_values = np.asarray(lon_coord.values)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        return lat_grid, lon_grid
    return lat_values, lon_values


def spatial_dims(data_array: xr.DataArray) -> tuple[str, ...]:
    names = []
    for name in data_array.dims:
        lowered = name.lower()
        if lowered in {"latitude", "longitude", "lat", "lon", "x", "y"}:
            names.append(name)
    return tuple(names)


def select_data_slice(data_array: xr.DataArray) -> tuple[xr.DataArray, dict[str, object]]:
    spatial = spatial_dims(data_array)
    non_spatial = [dim for dim in data_array.dims if dim not in spatial]
    selection = {}
    for dim in non_spatial:
        selection[dim] = 0
    selected = data_array.isel(selection) if selection else data_array
    if selected.ndim != 2:
        raise ValueError(f"Expected a 2D grid after selecting non-spatial dims, got ndim={selected.ndim}")
    metadata = {}
    for dim, index in selection.items():
        coord = selected.coords.get(dim)
        if coord is not None and coord.size:
            metadata[dim] = coord.values[index].item() if hasattr(coord.values[index], "item") else coord.values[index]
        else:
            metadata[dim] = index
    return selected, metadata


def pick_2d_slice(data_array: xr.DataArray) -> tuple[np.ndarray, dict[str, object]]:
    selected, metadata = select_data_slice(data_array)
    return np.asarray(selected.values, dtype="float64"), metadata


def extract_coord_scalar(data_array: xr.DataArray, coord_name: str) -> object | None:
    coord = data_array.coords.get(coord_name)
    if coord is None or coord.size == 0:
        return None
    values = np.asarray(coord.values)
    if values.size == 0:
        return None
    scalar = values.reshape(-1)[0]
    return scalar.item() if hasattr(scalar, "item") else scalar


def coerce_datetime_utc(value: object) -> dt.datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.to_pydatetime()


def parse_step_text_end_hour(step_text: str) -> int | None:
    range_match = re.search(r"(?P<start>\d+)-(?P<end>\d+) hour", step_text)
    if range_match:
        return int(range_match.group("end"))
    single_match = re.search(r"(?P<end>\d+) hour", step_text)
    if single_match:
        return int(single_match.group("end"))
    return None


def extract_actual_valid_time(
    data_array: xr.DataArray,
    *,
    cycle_plan: CyclePlan,
    lead_hour: int,
    primary_record: InventoryRecord | None,
) -> dt.datetime:
    selected, _ = select_data_slice(data_array)
    return extract_actual_valid_time_from_slice(
        selected,
        cycle_plan=cycle_plan,
        lead_hour=lead_hour,
        primary_record=primary_record,
    )


def extract_actual_valid_time_from_slice(
    selected: xr.DataArray,
    *,
    cycle_plan: CyclePlan,
    lead_hour: int,
    primary_record: InventoryRecord | None,
) -> dt.datetime:
    valid_time = extract_coord_scalar(selected, "valid_time")
    if valid_time is not None:
        return coerce_datetime_utc(valid_time)

    time_value = extract_coord_scalar(selected, "time")
    step_value = extract_coord_scalar(selected, "step")
    if time_value is not None and step_value is not None:
        base_time = coerce_datetime_utc(time_value)
        return (pd.Timestamp(base_time) + pd.to_timedelta(step_value)).to_pydatetime()

    if primary_record is not None:
        end_hour = parse_step_text_end_hour(primary_record.step_text)
        if end_hour is not None:
            return cycle_plan.init_time_utc + dt.timedelta(hours=end_hour)

    return cycle_plan.init_time_utc + dt.timedelta(hours=lead_hour)


def dataset_grid_context(
    dataset: xr.Dataset,
    *,
    station_lat: float,
    station_lon: float,
) -> DatasetGridContext:
    if not dataset.data_vars:
        raise ValueError("Dataset has no data variables")
    first_data_array = next(iter(dataset.data_vars.values()))
    lat_grid, lon_grid = extract_lat_lon(first_data_array)
    nearest = find_nearest_grid_cell(lat_grid, lon_grid, station_lat=station_lat, station_lon=station_lon)
    return DatasetGridContext(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        nearest=nearest,
        north_is_first=infer_north_is_first(lat_grid),
    )


def data_array_grid_context(
    data_array: xr.DataArray,
    *,
    station_lat: float,
    station_lon: float,
) -> DatasetGridContext:
    lat_grid, lon_grid = extract_lat_lon(data_array)
    nearest = find_nearest_grid_cell(lat_grid, lon_grid, station_lat=station_lat, station_lon=station_lon)
    return DatasetGridContext(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        nearest=nearest,
        north_is_first=infer_north_is_first(lat_grid),
    )


def data_array_grid_cache_key(data_array: xr.DataArray) -> tuple[object, ...]:
    lat_coord = data_array.coords.get("latitude")
    if lat_coord is None:
        lat_coord = data_array.coords.get("lat")
    lon_coord = data_array.coords.get("longitude")
    if lon_coord is None:
        lon_coord = data_array.coords.get("lon")
    if lat_coord is None or lon_coord is None:
        return (id(data_array),)
    lat_values = np.asarray(lat_coord.values)
    lon_values = np.asarray(lon_coord.values)
    lat_pointer = int(lat_values.__array_interface__["data"][0]) if lat_values.size else 0
    lon_pointer = int(lon_values.__array_interface__["data"][0]) if lon_values.size else 0
    def _coord_fingerprint(values: np.ndarray) -> tuple[object, ...]:
        if values.size == 0:
            return (0, None, None)
        flat = values.reshape(-1)
        return (
            int(values.size),
            float(flat[0]),
            float(flat[-1]),
        )

    return (
        tuple(lat_coord.dims),
        lat_values.shape,
        lat_values.strides,
        lat_pointer,
        _coord_fingerprint(lat_values),
        tuple(lon_coord.dims),
        lon_values.shape,
        lon_values.strides,
        lon_pointer,
        _coord_fingerprint(lon_values),
    )


def grid_metrics_from_slice(
    selected: xr.DataArray,
    *,
    slice_metadata: dict[str, object],
    grid_context: DatasetGridContext,
    requested_metric_names: set[str] | None = None,
) -> dict[str, object]:
    values = np.asarray(selected.values, dtype="float64")
    row = int(grid_context.nearest["grid_row"])
    col = int(grid_context.nearest["grid_col"])
    if requested_metric_names is None:
        sample_metrics = local_context_metrics(
            values,
            row=row,
            col=col,
            north_is_first=grid_context.north_is_first,
            window_sizes=LOCAL_NEIGHBORHOOD_SIZES,
        )
        crop_metrics = crop_context_metrics(values)
    else:
        sample_value = values[row, col]
        sample_metrics = {
            "sample_value": None if not np.isfinite(sample_value) else float(sample_value),
        }
        crop_names = {name for name in requested_metric_names if name.startswith("crop_")}
        crop_metrics = selected_crop_context_metrics(values, requested_names=crop_names)
        for window_size in LOCAL_NEIGHBORHOOD_SIZES:
            prefix = f"nb{window_size}"
            requested_for_window = {name for name in requested_metric_names if name.startswith(f"{prefix}_")}
            if requested_for_window:
                sample_metrics.update(
                    selected_neighborhood_metrics(
                        values,
                        row=row,
                        col=col,
                        window_size=window_size,
                        north_is_first=grid_context.north_is_first,
                        requested_names=requested_for_window,
                    )
                )
    result = {
        "sample_value": sample_metrics["sample_value"],
        "grid_row": row,
        "grid_col": col,
        "grid_lat": grid_context.nearest["grid_lat"],
        "grid_lon": grid_context.nearest["grid_lon"],
        "nearest_grid_lat": grid_context.nearest["grid_lat"],
        "nearest_grid_lon": float(longitude_360_to_180(grid_context.nearest["grid_lon"])),
        **crop_metrics,
        **sample_metrics,
    }
    result.update({f"slice_{key}": value for key, value in slice_metadata.items()})
    return result


def nearest_grid_metrics(
    data_array: xr.DataArray,
    *,
    station_lat: float,
    station_lon: float,
) -> dict[str, object]:
    selected, slice_metadata = select_data_slice(data_array)
    lat_grid, lon_grid = extract_lat_lon(data_array)
    nearest = find_nearest_grid_cell(lat_grid, lon_grid, station_lat=station_lat, station_lon=station_lon)
    context = DatasetGridContext(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        nearest=nearest,
        north_is_first=infer_north_is_first(lat_grid),
    )
    return grid_metrics_from_slice(selected, slice_metadata=slice_metadata, grid_context=context)


def selected_summary_values(values: np.ndarray, *, requested_stats: set[str]) -> dict[str, float | None]:
    if not requested_stats:
        return {}
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {name: None for name in requested_stats}
    outputs: dict[str, float | None] = {}
    if "mean" in requested_stats:
        outputs["mean"] = float(np.nanmean(finite))
    if "min" in requested_stats:
        outputs["min"] = float(np.nanmin(finite))
    if "max" in requested_stats:
        outputs["max"] = float(np.nanmax(finite))
    if "std" in requested_stats:
        outputs["std"] = float(np.nanstd(finite))
    return outputs


def selected_crop_context_metrics(values: np.ndarray, *, requested_names: set[str]) -> dict[str, float | None]:
    requested_stats = {name.replace("crop_", "") for name in requested_names}
    stats = selected_summary_values(values, requested_stats=requested_stats)
    return {f"crop_{name}": value for name, value in stats.items()}


def selected_neighborhood_metrics(
    values: np.ndarray,
    *,
    row: int,
    col: int,
    window_size: int,
    north_is_first: bool,
    requested_names: set[str],
) -> dict[str, float | None]:
    prefix = f"nb{window_size}"
    radius = window_size // 2
    row_start = max(0, row - radius)
    row_end = min(values.shape[0], row + radius + 1)
    col_start = max(0, col - radius)
    col_end = min(values.shape[1], col + radius + 1)
    neighborhood = values[row_start:row_end, col_start:col_end]
    requested_stats = {
        name[len(prefix) + 1 :]
        for name in requested_names
        if name in {f"{prefix}_mean", f"{prefix}_min", f"{prefix}_max", f"{prefix}_std"}
    }
    stats = selected_summary_values(neighborhood, requested_stats=requested_stats)
    outputs = {f"{prefix}_{name}": value for name, value in stats.items()}
    if f"{prefix}_gradient_west_east" in requested_names or f"{prefix}_gradient_south_north" in requested_names:
        west_mean = selected_summary_values(neighborhood[:, 0], requested_stats={"mean"}).get("mean")
        east_mean = selected_summary_values(neighborhood[:, -1], requested_stats={"mean"}).get("mean")
        top_mean = selected_summary_values(neighborhood[0, :], requested_stats={"mean"}).get("mean")
        bottom_mean = selected_summary_values(neighborhood[-1, :], requested_stats={"mean"}).get("mean")
        north_mean = top_mean if north_is_first else bottom_mean
        south_mean = bottom_mean if north_is_first else top_mean
        if f"{prefix}_gradient_west_east" in requested_names:
            outputs[f"{prefix}_gradient_west_east"] = (
                None if west_mean is None or east_mean is None else east_mean - west_mean
            )
        if f"{prefix}_gradient_south_north" in requested_names:
            outputs[f"{prefix}_gradient_south_north"] = (
                None if north_mean is None or south_mean is None else north_mean - south_mean
            )
    return outputs


def data_array_identifiers(var_name: str, data_array: xr.DataArray) -> set[str]:
    values = {
        var_name,
        data_array.name,
        data_array.attrs.get("GRIB_shortName"),
        data_array.attrs.get("GRIB_cfVarName"),
        data_array.attrs.get("GRIB_name"),
        data_array.attrs.get("long_name"),
        data_array.attrs.get("standard_name"),
    }
    return {normalize_identifier(value) for value in values if value is not None}


def canonical_feature_name(var_name: str, data_array: xr.DataArray) -> str | None:
    identifiers = data_array_identifiers(var_name, data_array)
    for feature_name, aliases in FEATURE_IDENTIFIER_ALIASES.items():
        if identifiers & aliases:
            return feature_name
    return None


def canonical_feature_mapping(var_name: str, data_array: xr.DataArray) -> tuple[str | None, bool, str | None]:
    identifiers = data_array_identifiers(var_name, data_array)
    for feature_name, aliases in FEATURE_IDENTIFIER_ALIASES.items():
        matched = identifiers & aliases
        if not matched:
            continue
        fallback_used = normalize_identifier(var_name) not in {normalize_identifier(feature_name), FEATURE_SPEC_BY_NAME[feature_name].short_name}
        fallback_source_description = None
        if fallback_used:
            fallback_source_description = f"mapped data variable '{var_name}' using alias '{sorted(matched)[0]}'"
        return feature_name, fallback_used, fallback_source_description
    return None, False, None


def canonical_feature_name_for_record(record: InventoryRecord) -> str | None:
    record_identifiers = {normalize_identifier(record.short_name)}
    for feature_name, aliases in FEATURE_IDENTIFIER_ALIASES.items():
        if record_identifiers & aliases:
            return feature_name
    return None


def expected_type_of_level(level_text: str) -> str | None:
    mapping = {
        "2 m above ground": "heightAboveGround",
        "10 m above ground": "heightAboveGround",
        "surface": "surface",
        "surface - reserved": "surface",
        "reserved": "surface",
        "cloud ceiling": "cloudCeiling",
        "entire atmosphere (considered as a single layer)": "atmosphereSingleLayer",
        "entire atmosphere": "atmosphere",
    }
    return mapping.get(level_text)


def expected_level_value(level_text: str) -> int | None:
    mapping = {
        "2 m above ground": 2,
        "10 m above ground": 10,
    }
    return mapping.get(level_text)


def expected_step_type(step_text: str) -> str:
    lowered = step_text.lower()
    if " acc " in f" {lowered} " or "acc fcst" in lowered:
        return "accum"
    if " avg " in f" {lowered} " or "avg fcst" in lowered:
        return "avg"
    if " max " in f" {lowered} " or "max fcst" in lowered:
        return "max"
    if " min " in f" {lowered} " or "min fcst" in lowered:
        return "min"
    return "instant"


def data_array_level_value(data_array: xr.DataArray, *, type_of_level: str | None) -> int | None:
    for coord_name in filter(None, [type_of_level, "heightAboveGround", "surface", "cloudCeiling", "atmosphere", "atmosphereSingleLayer"]):
        value = extract_coord_scalar(data_array, coord_name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    attr_level = data_array.attrs.get("GRIB_level")
    if attr_level is not None:
        try:
            return int(attr_level)
        except (TypeError, ValueError):
            return None
    return None


def record_matches_data_array(record: InventoryRecord, data_array: xr.DataArray) -> bool:
    record_feature_name = canonical_feature_name_for_record(record)
    if record_feature_name is not None:
        if canonical_feature_name(str(data_array.name), data_array) != record_feature_name:
            return False
    else:
        short_name = normalize_identifier(data_array.attrs.get("GRIB_shortName"))
        if short_name and short_name != normalize_identifier(record.short_name):
            return False
    step_type = data_array.attrs.get("GRIB_stepType")
    expected_step = expected_step_type(record.step_text)
    if step_type is not None and step_type != expected_step:
        return False
    expected_level = expected_level_value(record.level_text)
    if expected_level is not None:
        type_of_level = data_array.attrs.get("GRIB_typeOfLevel")
        actual_level = data_array_level_value(data_array, type_of_level=type_of_level)
        if actual_level is not None and actual_level != expected_level:
            return False
    return True


def matching_selected_records_for_data_array(
    records: list[InventoryRecord],
    data_array: xr.DataArray,
) -> list[InventoryRecord]:
    return [record for record in records if record_matches_data_array(record, data_array)]


def matching_records_for_forecast_hour(
    records: list[InventoryRecord],
    forecast_hour: int,
) -> list[InventoryRecord]:
    return [
        record
        for record in records
        if parse_step_text_end_hour(record.step_text) == forecast_hour
    ]


def selected_records_by_feature_name(selected: list[SelectedField]) -> dict[str, SelectedField]:
    mapping: dict[str, SelectedField] = {}
    for field in selected:
        mapping[field.spec.feature_name] = field
    return mapping


def infer_selected_long_field(
    *,
    data_array: xr.DataArray,
    selected: list[SelectedField],
    present_features_for_valid_time: set[str],
    allow_unknown_long_inference: bool,
) -> tuple[str | None, SelectedField | None, bool, str | None]:
    if not allow_unknown_long_inference:
        return None, None, False, None
    identifiers = data_array_identifiers(str(data_array.name), data_array)
    if "UNKNOWN" not in identifiers:
        return None, None, False, None
    unresolved = [
        field
        for field in selected
        if field.spec.output_kind == "long"
        and field.spec.feature_name not in present_features_for_valid_time
    ]
    if len(unresolved) != 1:
        return None, None, False, None
    selected_field = unresolved[0]
    return (
        selected_field.spec.feature_name,
        selected_field,
        True,
        f"inferred feature '{selected_field.spec.feature_name}' from unknown cfgrib variable using remaining selected inventory",
    )


def eligible_optional_feature_names(selected: list[SelectedField]) -> set[str]:
    return {
        field.spec.feature_name
        for field in selected
        if field.spec.optional
    }


def make_base_time_columns(
    cycle_plan: CyclePlan,
    valid_time_utc: dt.datetime,
    requested_lead_hour: int,
    *,
    crop_bounds: CropBounds | None = None,
    nearest_grid_lat: float | None = None,
    nearest_grid_lon: float | None = None,
) -> dict[str, object]:
    valid_time_local = valid_time_utc.astimezone(NY_TZ)
    forecast_hour = int(round((valid_time_utc - cycle_plan.init_time_utc).total_seconds() / 3600.0))
    return {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "init_time_utc": cycle_plan.init_time_utc.isoformat(),
        "init_time_local": cycle_plan.init_time_local.isoformat(),
        "init_date_local": cycle_plan.init_time_local.date().isoformat(),
        "valid_time_utc": valid_time_utc.isoformat(),
        "valid_time_local": valid_time_local.isoformat(),
        "valid_date_local": valid_time_local.date().isoformat(),
        "forecast_hour": forecast_hour,
        "lead_hour": forecast_hour,
        "requested_lead_hour": requested_lead_hour,
        "mode": cycle_plan.mode,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "station_lat": SETTLEMENT_LOCATION.lat,
        "station_lon": SETTLEMENT_LOCATION.lon,
        **crop_metadata(crop_bounds),
        "settlement_station_id": SETTLEMENT_LOCATION.station_id,
        "settlement_lat": SETTLEMENT_LOCATION.lat,
        "settlement_lon": SETTLEMENT_LOCATION.lon,
        "nearest_grid_lat": nearest_grid_lat,
        "nearest_grid_lon": nearest_grid_lon,
    }


def record_metadata_dict(record: InventoryRecord | None) -> dict[str, object]:
    if record is None:
        return {
            "short_name": None,
            "level_text": None,
            "step_text": None,
            "inventory_line": None,
        }
    return {
        "short_name": record.short_name,
        "level_text": record.level_text,
        "step_text": record.step_text,
        "inventory_line": record.raw_line,
    }


def flatten_value(value: object) -> object:
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def json_list(values: list[str]) -> str:
    return json.dumps(values)


def validate_required_wide_row(row: dict[str, object]) -> None:
    missing = sorted(column for column in REQUIRED_WIDE_IDENTITY_COLUMNS if row.get(column) is None)
    if missing:
        raise ValueError(f"Wide row is missing required canonical identity columns: {', '.join(missing)}")


def write_wide_feature_metrics(
    wide_row_metrics: dict[str, object],
    *,
    field_name: str,
    metrics: dict[str, object],
    metric_names: set[str] | None = None,
) -> None:
    wide_row_metrics[field_name] = metrics["sample_value"]
    metric_names = metric_names or {
        "crop_mean",
        "crop_min",
        "crop_max",
        "crop_std",
        "nb3_mean",
        "nb3_min",
        "nb3_max",
        "nb3_std",
        "nb3_gradient_west_east",
        "nb3_gradient_south_north",
        "nb7_mean",
        "nb7_min",
        "nb7_max",
        "nb7_std",
        "nb7_gradient_west_east",
        "nb7_gradient_south_north",
    }
    for metric_name in sorted(metric_names):
        if metric_name in metrics:
            wide_row_metrics[f"{field_name}_{metric_name}"] = metrics[metric_name]
    if "grid_row" not in wide_row_metrics:
        wide_row_metrics["grid_row"] = metrics["grid_row"]
        wide_row_metrics["grid_col"] = metrics["grid_col"]
        wide_row_metrics["nearest_grid_lat"] = metrics["nearest_grid_lat"]
        wide_row_metrics["nearest_grid_lon"] = metrics["nearest_grid_lon"]
        wide_row_metrics["grid_lat"] = metrics["grid_lat"]
        wide_row_metrics["grid_lon"] = metrics["grid_lon"]


def build_rows_from_datasets(
    *,
    datasets: list[xr.Dataset],
    selected: list[SelectedField],
    cycle_plan: CyclePlan,
    lead_hour: int,
    crop_bounds: CropBounds | None = None,
    allow_unknown_long_inference: bool = True,
    write_long: bool = True,
    write_provenance: bool = True,
    metric_profile: str = "full",
    stats: dict[str, object] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if not datasets:
        raise ValueError("No grouped cfgrib datasets could be opened from the reduced file")

    wide_rows_by_valid_time: dict[dt.datetime, dict[str, object]] = {}
    wide_row_fallback_by_valid_time: dict[dt.datetime, bool] = {}
    long_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    selected_by_feature = selected_records_by_feature_name(selected)
    eligible_optional_features = eligible_optional_feature_names(selected)
    present_features_by_valid_time: dict[dt.datetime, set[str]] = {}
    base_columns_cache: dict[tuple[dt.datetime, float | None, float | None], dict[str, object]] = {}
    grid_context_cache: dict[tuple[object, ...], DatasetGridContext] = {}
    timing_row_geometry_seconds = 0.0
    timing_row_metric_seconds = 0.0
    timing_row_provenance_seconds = 0.0

    def _metric_names_for_field(field_name: str) -> set[str] | None:
        if metric_profile == "full":
            return None
        required_by_field = {
            "tmax": {"crop_max", "nb3_max", "nb7_max"},
            "tmin": {"crop_min", "nb3_min", "nb7_min"},
            "tmp": {"crop_mean", "nb3_mean", "nb7_mean"},
            "tcdc": {"crop_mean"},
            "dswrf": {"crop_max"},
            "wind": {"nb7_mean"},
        }
        return required_by_field.get(field_name, set())

    def _wide_metric_values(metrics: dict[str, object], metric_names: set[str] | None) -> dict[str, object]:
        if metric_names is not None:
            return {"sample_value": metrics.get("sample_value"), **{name: metrics.get(name) for name in metric_names}}
        return {
            "sample_value": metrics.get("sample_value"),
            "crop_mean": metrics.get("crop_mean"),
            "crop_min": metrics.get("crop_min"),
            "crop_max": metrics.get("crop_max"),
            "crop_std": metrics.get("crop_std"),
            "nb3_mean": metrics.get("nb3_mean"),
            "nb3_min": metrics.get("nb3_min"),
            "nb3_max": metrics.get("nb3_max"),
            "nb3_std": metrics.get("nb3_std"),
            "nb3_gradient_west_east": metrics.get("nb3_gradient_west_east"),
            "nb3_gradient_south_north": metrics.get("nb3_gradient_south_north"),
            "nb7_mean": metrics.get("nb7_mean"),
            "nb7_min": metrics.get("nb7_min"),
            "nb7_max": metrics.get("nb7_max"),
            "nb7_std": metrics.get("nb7_std"),
            "nb7_gradient_west_east": metrics.get("nb7_gradient_west_east"),
            "nb7_gradient_south_north": metrics.get("nb7_gradient_south_north"),
        }

    def _existing_wide_metric_values(
        wide_row_metrics: dict[str, object],
        *,
        field_name: str,
        metric_names: set[str] | None,
    ) -> dict[str, object]:
        if metric_names is not None:
            return {
                "sample_value": wide_row_metrics.get(field_name),
                **{name: wide_row_metrics.get(f"{field_name}_{name}") for name in metric_names},
            }
        return {
            "sample_value": wide_row_metrics.get(field_name),
            "crop_mean": wide_row_metrics.get(f"{field_name}_crop_mean"),
            "crop_min": wide_row_metrics.get(f"{field_name}_crop_min"),
            "crop_max": wide_row_metrics.get(f"{field_name}_crop_max"),
            "crop_std": wide_row_metrics.get(f"{field_name}_crop_std"),
            "nb3_mean": wide_row_metrics.get(f"{field_name}_nb3_mean"),
            "nb3_min": wide_row_metrics.get(f"{field_name}_nb3_min"),
            "nb3_max": wide_row_metrics.get(f"{field_name}_nb3_max"),
            "nb3_std": wide_row_metrics.get(f"{field_name}_nb3_std"),
            "nb3_gradient_west_east": wide_row_metrics.get(f"{field_name}_nb3_gradient_west_east"),
            "nb3_gradient_south_north": wide_row_metrics.get(f"{field_name}_nb3_gradient_south_north"),
            "nb7_mean": wide_row_metrics.get(f"{field_name}_nb7_mean"),
            "nb7_min": wide_row_metrics.get(f"{field_name}_nb7_min"),
            "nb7_max": wide_row_metrics.get(f"{field_name}_nb7_max"),
            "nb7_std": wide_row_metrics.get(f"{field_name}_nb7_std"),
            "nb7_gradient_west_east": wide_row_metrics.get(f"{field_name}_nb7_gradient_west_east"),
            "nb7_gradient_south_north": wide_row_metrics.get(f"{field_name}_nb7_gradient_south_north"),
        }

    def _base_columns(
        actual_valid_time_utc: dt.datetime,
        *,
        nearest_grid_lat: float | None,
        nearest_grid_lon: float | None,
    ) -> dict[str, object]:
        key = (actual_valid_time_utc, nearest_grid_lat, nearest_grid_lon)
        cached = base_columns_cache.get(key)
        if cached is None:
            cached = make_base_time_columns(
                cycle_plan,
                actual_valid_time_utc,
                requested_lead_hour=lead_hour,
                crop_bounds=crop_bounds,
                nearest_grid_lat=nearest_grid_lat,
                nearest_grid_lon=nearest_grid_lon,
            )
            base_columns_cache[key] = cached
        return cached.copy()

    def _append_provenance_row(
        *,
        actual_valid_time_utc: dt.datetime,
        nearest_grid_lat: float | None,
        nearest_grid_lon: float | None,
        feature_name: str,
        data_var_name: str | None,
        source_variable_name: str | None,
        metadata: dict[str, object] | None,
        data_array: xr.DataArray | None,
        present_directly: bool,
        derived: bool,
        missing_optional: bool,
        derivation_method: str | None,
        source_feature_names: list[str],
        fallback_used: bool,
        fallback_source_description: str | None,
        notes: str | None,
        extra_metadata: dict[str, object] | None = None,
        units: str | None = None,
    ) -> None:
        nonlocal timing_row_provenance_seconds
        if not write_provenance:
            return
        started_at = time.perf_counter()
        row = {
            **_base_columns(
                actual_valid_time_utc,
                nearest_grid_lat=nearest_grid_lat,
                nearest_grid_lon=nearest_grid_lon,
            ),
            "feature_name": feature_name,
            "data_var_name": data_var_name,
            "source_variable_name": source_variable_name,
            "grib_short_name": metadata["short_name"] if metadata else None,
            "grib_level_text": metadata["level_text"] if metadata else None,
            "grib_step_text": metadata["step_text"] if metadata else None,
            "inventory_line": metadata["inventory_line"] if metadata else None,
            "type_of_level": data_array.attrs.get("GRIB_typeOfLevel") if data_array is not None else None,
            "step_type": data_array.attrs.get("GRIB_stepType") if data_array is not None else None,
            "grib_name": data_array.attrs.get("GRIB_name") if data_array is not None else None,
            "units": units if units is not None else (data_array.attrs.get("units") if data_array is not None else None),
            "source_product": SOURCE_PRODUCT,
            "source_version": SOURCE_VERSION,
            "present_directly": present_directly,
            "derived": derived,
            "missing_optional": missing_optional,
            "derivation_method": derivation_method,
            "source_feature_names": json_list(source_feature_names),
            "fallback_used": fallback_used,
            "fallback_source_description": fallback_source_description,
            "notes": notes,
        }
        if extra_metadata:
            row.update(extra_metadata)
        provenance_rows.append(row)
        timing_row_provenance_seconds += stage_elapsed_seconds(started_at)

    for dataset in datasets:
        for var_name, data_array in dataset.data_vars.items():
            field_name, alias_fallback_used, alias_fallback_description = canonical_feature_mapping(var_name, data_array)
            if field_name is None and not allow_unknown_long_inference:
                continue
            spec = FEATURE_SPEC_BY_NAME.get(field_name) if field_name else None
            selected_field = selected_by_feature.get(field_name) if field_name else None
            matching_records = (
                matching_selected_records_for_data_array(selected_field.records, data_array)
                if selected_field
                else []
            )
            primary_record = matching_records[0] if matching_records else None
            fallback_used = bool((selected_field and selected_field.fallback_used) or alias_fallback_used)
            fallback_parts = [
                selected_field.fallback_source_description if selected_field else None,
                alias_fallback_description,
            ]
            fallback_source_description = "; ".join(part for part in fallback_parts if part) or None
            if field_name and selected_field is None:
                continue
            if field_name and selected_field and not matching_records:
                continue

            try:
                selected_slice, slice_metadata = select_data_slice(data_array)
            except ValueError:
                continue

            actual_valid_time_utc = extract_actual_valid_time_from_slice(
                selected_slice,
                cycle_plan=cycle_plan,
                lead_hour=lead_hour,
                primary_record=primary_record,
            )
            forecast_hour = int(round((actual_valid_time_utc - cycle_plan.init_time_utc).total_seconds() / 3600.0))
            if lead_hour < 0 and matching_records:
                matching_records = matching_records_for_forecast_hour(matching_records, forecast_hour)
                if not matching_records:
                    continue
                primary_record = matching_records[0]
            if field_name is None:
                (
                    field_name,
                    selected_field,
                    inferred_fallback_used,
                    inferred_fallback_description,
                ) = infer_selected_long_field(
                    data_array=data_array,
                    selected=selected,
                    present_features_for_valid_time=present_features_by_valid_time.get(actual_valid_time_utc, set()),
                    allow_unknown_long_inference=allow_unknown_long_inference,
                )
                if field_name:
                    spec = FEATURE_SPEC_BY_NAME.get(field_name)
                    matching_records = selected_field.records if selected_field else []
                    if lead_hour < 0 and matching_records:
                        matching_records = matching_records_for_forecast_hour(matching_records, forecast_hour)
                    primary_record = matching_records[0] if matching_records else None
                    if not matching_records:
                        continue
                    alias_fallback_used = alias_fallback_used or inferred_fallback_used
                    if inferred_fallback_description:
                        alias_fallback_description = (
                            f"{alias_fallback_description}; {inferred_fallback_description}"
                            if alias_fallback_description
                            else inferred_fallback_description
                        )
                    fallback_used = bool((selected_field and selected_field.fallback_used) or alias_fallback_used)
                    fallback_parts = [
                        selected_field.fallback_source_description if selected_field else None,
                        alias_fallback_description,
                    ]
                    fallback_source_description = "; ".join(part for part in fallback_parts if part) or None

            if field_name is None or spec is None:
                continue

            grid_context_key = data_array_grid_cache_key(data_array)
            grid_context = grid_context_cache.get(grid_context_key)
            if grid_context is None:
                geometry_started_at = time.perf_counter()
                grid_context = data_array_grid_context(
                    data_array,
                    station_lat=SETTLEMENT_LOCATION.lat,
                    station_lon=SETTLEMENT_LOCATION.lon,
                )
                grid_context_cache[grid_context_key] = grid_context
                timing_row_geometry_seconds += stage_elapsed_seconds(geometry_started_at)

            metric_started_at = time.perf_counter()
            metric_names = _metric_names_for_field(field_name)
            metrics = grid_metrics_from_slice(
                selected_slice,
                slice_metadata=slice_metadata,
                grid_context=grid_context,
                requested_metric_names=metric_names,
            )
            timing_row_metric_seconds += stage_elapsed_seconds(metric_started_at)

            if spec.output_kind == "wide":
                wide_row_metrics = wide_rows_by_valid_time.setdefault(actual_valid_time_utc, {})
                if field_name in wide_row_metrics:
                    existing_metrics = _existing_wide_metric_values(
                        wide_row_metrics,
                        field_name=field_name,
                        metric_names=metric_names,
                    )
                    if _wide_metric_values(metrics, metric_names) == existing_metrics:
                        continue
                    raise ValueError(
                        f"Ambiguous duplicate dataset for selected wide feature '{field_name}' at {actual_valid_time_utc.isoformat()}"
                    )
                wide_row_fallback_by_valid_time[actual_valid_time_utc] = (
                    wide_row_fallback_by_valid_time.get(actual_valid_time_utc, False) or fallback_used
                )
                present_features_by_valid_time.setdefault(actual_valid_time_utc, set()).add(field_name)
                write_wide_feature_metrics(wide_row_metrics, field_name=field_name, metrics=metrics, metric_names=metric_names)
                metadata = record_metadata_dict(primary_record)
                _append_provenance_row(
                    actual_valid_time_utc=actual_valid_time_utc,
                    nearest_grid_lat=metrics["nearest_grid_lat"],
                    nearest_grid_lon=metrics["nearest_grid_lon"],
                    feature_name=field_name,
                    data_var_name=var_name,
                    source_variable_name=var_name,
                    metadata=metadata,
                    data_array=data_array,
                    present_directly=True,
                    derived=False,
                    missing_optional=False,
                    derivation_method=None,
                    source_feature_names=[],
                    fallback_used=fallback_used,
                    fallback_source_description=fallback_source_description,
                    notes=primary_record.extra_text if primary_record else None,
                )
            elif spec.output_kind == "long":
                if field_name in LONG_FEATURES_MIRRORED_TO_WIDE:
                    wide_row_metrics = wide_rows_by_valid_time.setdefault(actual_valid_time_utc, {})
                    if field_name in wide_row_metrics:
                        existing_metrics = _existing_wide_metric_values(
                            wide_row_metrics,
                            field_name=field_name,
                            metric_names=metric_names,
                        )
                        if _wide_metric_values(metrics, metric_names) != existing_metrics:
                            raise ValueError(
                                f"Ambiguous duplicate dataset for selected mirrored feature '{field_name}' at {actual_valid_time_utc.isoformat()}"
                            )
                    else:
                        write_wide_feature_metrics(wide_row_metrics, field_name=field_name, metrics=metrics, metric_names=metric_names)
                    wide_row_fallback_by_valid_time[actual_valid_time_utc] = (
                        wide_row_fallback_by_valid_time.get(actual_valid_time_utc, False) or fallback_used
                    )
                present_features_by_valid_time.setdefault(actual_valid_time_utc, set()).add(field_name)
                record_lines = "; ".join(record.raw_line for record in matching_records)
                metadata = record_metadata_dict(primary_record)
                extra_metadata = {
                    key.replace("slice_", ""): flatten_value(value)
                    for key, value in metrics.items()
                    if key.startswith("slice_")
                }
                if write_long:
                    long_row = {
                        **_base_columns(
                            actual_valid_time_utc,
                            nearest_grid_lat=metrics["nearest_grid_lat"],
                            nearest_grid_lon=metrics["nearest_grid_lon"],
                        ),
                        "feature_name": field_name,
                        "data_var_name": var_name,
                        "sample_value": metrics["sample_value"],
                        "inventory_line": record_lines or None,
                        "type_of_level": data_array.attrs.get("GRIB_typeOfLevel"),
                        "step_type": data_array.attrs.get("GRIB_stepType"),
                        "grib_name": data_array.attrs.get("GRIB_name"),
                        "units": data_array.attrs.get("units"),
                        "source_product": SOURCE_PRODUCT,
                        "source_version": SOURCE_VERSION,
                        "fallback_used": fallback_used,
                        "fallback_source_description": fallback_source_description,
                        **extra_metadata,
                    }
                    for metric_name in (
                        "crop_mean",
                        "crop_min",
                        "crop_max",
                        "crop_std",
                        "nb3_mean",
                        "nb3_min",
                        "nb3_max",
                        "nb3_std",
                        "nb3_gradient_west_east",
                        "nb3_gradient_south_north",
                        "nb7_mean",
                        "nb7_min",
                        "nb7_max",
                        "nb7_std",
                        "nb7_gradient_west_east",
                        "nb7_gradient_south_north",
                    ):
                        long_row[metric_name] = metrics.get(metric_name)
                    long_rows.append(long_row)
                metadata["inventory_line"] = metadata["inventory_line"] or record_lines or None
                _append_provenance_row(
                    actual_valid_time_utc=actual_valid_time_utc,
                    nearest_grid_lat=metrics["nearest_grid_lat"],
                    nearest_grid_lon=metrics["nearest_grid_lon"],
                    feature_name=field_name,
                    data_var_name=var_name,
                    source_variable_name=var_name,
                    metadata=metadata,
                    data_array=data_array,
                    present_directly=True,
                    derived=False,
                    missing_optional=False,
                    derivation_method=None,
                    source_feature_names=[],
                    fallback_used=fallback_used,
                    fallback_source_description=fallback_source_description,
                    notes=None,
                    extra_metadata=extra_metadata,
                )

    if not wide_rows_by_valid_time and not long_rows:
        raise ValueError("No feature rows could be built from the reduced GRIB2 datasets")

    wide_rows: list[dict[str, object]] = []
    for actual_valid_time_utc in sorted(wide_rows_by_valid_time):
        wide_row_metrics = wide_rows_by_valid_time[actual_valid_time_utc]
        wide_row = _base_columns(
            actual_valid_time_utc,
            nearest_grid_lat=wide_row_metrics.get("nearest_grid_lat"),
            nearest_grid_lon=wide_row_metrics.get("nearest_grid_lon"),
        )
        wide_row.update(wide_row_metrics)
        wide_row["fallback_used_any"] = wide_row_fallback_by_valid_time.get(actual_valid_time_utc, False)

        if wide_row.get("wind") is not None and wide_row.get("wdir") is not None:
            angle = math.radians(float(wide_row["wdir"]))
            speed = float(wide_row["wind"])
            wide_row["u10"] = -speed * math.sin(angle)
            wide_row["v10"] = -speed * math.cos(angle)
            for feature_name, value, method in (
                ("u10", wide_row["u10"], "zonal wind from scalar speed and meteorological direction"),
                ("v10", wide_row["v10"], "meridional wind from scalar speed and meteorological direction"),
            ):
                _append_provenance_row(
                    actual_valid_time_utc=actual_valid_time_utc,
                    nearest_grid_lat=wide_row_metrics.get("nearest_grid_lat"),
                    nearest_grid_lon=wide_row_metrics.get("nearest_grid_lon"),
                    feature_name=feature_name,
                    data_var_name=None,
                    source_variable_name=None,
                    metadata=None,
                    data_array=None,
                    present_directly=False,
                    derived=True,
                    missing_optional=False,
                    derivation_method=method,
                    source_feature_names=["wind", "wdir"],
                    fallback_used=False,
                    fallback_source_description=None,
                    notes=f"Derived value={value}",
                    units="m s-1",
                )

        missing_optional_features = sorted(eligible_optional_features - present_features_by_valid_time.get(actual_valid_time_utc, set()))
        wide_row["missing_optional_any"] = bool(missing_optional_features)
        wide_row["missing_optional_fields_count"] = len(missing_optional_features)
        for feature_name in missing_optional_features:
            _append_provenance_row(
                actual_valid_time_utc=actual_valid_time_utc,
                nearest_grid_lat=wide_row_metrics.get("nearest_grid_lat"),
                nearest_grid_lon=wide_row_metrics.get("nearest_grid_lon"),
                feature_name=feature_name,
                data_var_name=None,
                source_variable_name=None,
                metadata=None,
                data_array=None,
                present_directly=False,
                derived=False,
                missing_optional=True,
                derivation_method=None,
                source_feature_names=[],
                fallback_used=False,
                fallback_source_description=None,
                notes="Optional feature family missing for this valid time in the reduced NBM inventory.",
            )

        validate_required_wide_row(wide_row)
        wide_rows.append(wide_row)

    if stats is not None:
        stats.update(
            {
                "wide_row_count": len(wide_rows),
                "long_row_count": len(long_rows),
                "provenance_row_count": len(provenance_rows),
                "provenance_written": bool(write_provenance),
                "timing_row_geometry_seconds": timing_row_geometry_seconds,
                "timing_row_metric_seconds": timing_row_metric_seconds,
                "timing_row_provenance_seconds": timing_row_provenance_seconds,
            }
        )

    return wide_rows, long_rows, provenance_rows


def deterministic_file_id(cycle_plan: CyclePlan) -> str:
    return cycle_plan.init_time_utc.strftime("%Y%m%dT%H%MZ")


def manifest_path(output_dir: pathlib.Path, cycle_plan: CyclePlan) -> pathlib.Path:
    return (
        output_dir
        / "metadata"
        / "manifest"
        / f"init_date_local={cycle_plan.init_date_local.isoformat()}"
        / f"mode={cycle_plan.mode}"
        / f"cycle_{deterministic_file_id(cycle_plan)}_manifest.parquet"
    )


def cycle_already_complete(output_dir: pathlib.Path, cycle_plan: CyclePlan) -> bool:
    path = manifest_path(output_dir, cycle_plan)
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path)
    except Exception:
        return False
    if df.empty or not bool((df["extraction_status"] == "ok").all()):
        return False
    if "reduced_retained" in df and df["reduced_retained"].fillna(False).any():
        retained = df.loc[df["reduced_retained"].fillna(False), "reduced_file_path"].dropna().tolist()
        if retained and not all(pathlib.Path(value).exists() for value in retained):
            return False
    for column in ("wide_output_paths", "long_output_paths", "provenance_output_paths"):
        if column not in df:
            continue
        for token in df[column].dropna().tolist():
            if not token:
                continue
            if not all(pathlib.Path(part).exists() for part in token.split(";") if part):
                return False
    return True


def partition_file_path(
    *,
    root: pathlib.Path,
    cycle_plan: CyclePlan,
    suffix: str,
    valid_date_local: str,
    init_date_local: str,
    mode: str,
) -> pathlib.Path:
    return (
        root
        / f"valid_date_local={valid_date_local}"
        / f"init_date_local={init_date_local}"
        / f"mode={mode}"
        / f"cycle_{deterministic_file_id(cycle_plan)}_{suffix}.parquet"
    )


def write_partitioned_table(
    df: pd.DataFrame,
    *,
    root: pathlib.Path,
    cycle_plan: CyclePlan,
    suffix: str,
) -> dict[tuple[str, str, str], str]:
    if df.empty:
        return {}
    paths: dict[tuple[str, str, str], str] = {}
    grouped = df.groupby(["valid_date_local", "init_date_local", "mode"], dropna=False)
    for (valid_date_local, init_date_local, mode), part_df in grouped:
        path = partition_file_path(
            root=root,
            cycle_plan=cycle_plan,
            suffix=suffix,
            valid_date_local=str(valid_date_local),
            init_date_local=str(init_date_local),
            mode=str(mode),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        part_df.to_parquet(path, index=False)
        paths[(str(valid_date_local), str(init_date_local), str(mode))] = str(path)
    return paths


def manifest_row_output_paths(
    manifest_row: dict[str, object],
    row_group: list[dict[str, object]],
    path_map: dict[tuple[str, str, str], str],
) -> str:
    if not row_group:
        return ""
    values: list[str] = []
    for row in row_group:
        key = (
            str(row["valid_date_local"]),
            str(row["init_date_local"]),
            str(row["mode"]),
        )
        path = path_map.get(key)
        if path and path not in values:
            values.append(path)
    if values:
        return ";".join(values)
    key = (
        str(manifest_row["valid_date_local"]),
        str(manifest_row["init_date_local"]),
        str(manifest_row["mode"]),
    )
    return path_map.get(key, "")


def write_cycle_outputs(
    output_dir: pathlib.Path,
    cycle_plan: CyclePlan,
    results: list[UnitResult],
    *,
    write_long: bool,
    write_provenance: bool = True,
) -> None:
    wide_rows = [row for result in results for row in result.wide_rows]
    long_rows = [row for result in results for row in result.long_rows]
    provenance_rows = [row for result in results for row in result.provenance_rows]
    manifest_rows = [result.manifest_row for result in results]

    wide_paths = write_partitioned_table(
        pd.DataFrame.from_records(wide_rows),
        root=output_dir / "features" / "wide",
        cycle_plan=cycle_plan,
        suffix="wide",
    )
    long_paths = (
        write_partitioned_table(
            pd.DataFrame.from_records(long_rows),
            root=output_dir / "features" / "long",
            cycle_plan=cycle_plan,
            suffix="long",
        )
        if write_long
        else {}
    )
    provenance_paths = (
        write_partitioned_table(
            pd.DataFrame.from_records(provenance_rows),
            root=output_dir / "metadata" / "provenance",
            cycle_plan=cycle_plan,
            suffix="provenance",
        )
        if write_provenance
        else {}
    )

    manifest_df = pd.DataFrame.from_records(manifest_rows)
    manifest_df["provenance_written"] = bool(write_provenance)
    manifest_df["wide_output_paths"] = [
        manifest_row_output_paths(result.manifest_row, result.wide_rows, wide_paths)
        for result in results
    ]
    manifest_df["long_output_paths"] = [
        manifest_row_output_paths(result.manifest_row, result.long_rows, long_paths)
        for result in results
    ]
    manifest_df["provenance_output_paths"] = [
        manifest_row_output_paths(result.manifest_row, result.provenance_rows, provenance_paths)
        for result in results
    ]
    out_path = manifest_path(output_dir, cycle_plan)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_parquet(out_path, index=False)


def concatenate_grib_files(paths: list[pathlib.Path], destination: pathlib.Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    with destination.open("wb") as out_handle:
        for path in paths:
            with path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)
    return stage_elapsed_seconds(started_at)


def combine_selected_fields_for_batch(lead_items: list[NbmPipelineItem]) -> list[SelectedField]:
    by_feature: dict[str, SelectedField] = {}
    for item in lead_items:
        for field in item.selected_fields:
            existing = by_feature.get(field.spec.feature_name)
            if existing is None:
                by_feature[field.spec.feature_name] = SelectedField(
                    spec=field.spec,
                    records=list(field.records),
                    fallback_used=field.fallback_used,
                    fallback_source_description=field.fallback_source_description,
                )
                continue
            existing.records.extend(field.records)
            existing.fallback_used = existing.fallback_used or field.fallback_used
            if not existing.fallback_source_description:
                existing.fallback_source_description = field.fallback_source_description
    return [by_feature[spec.feature_name] for spec in FIELD_SPECS if spec.feature_name in by_feature]


def expanded_2d_datasets(datasets: list[xr.Dataset]) -> list[xr.Dataset]:
    expanded: list[xr.Dataset] = []
    for dataset in datasets:
        for var_name, data_array in dataset.data_vars.items():
            spatial = set(spatial_dims(data_array))
            non_spatial_dims = [dim for dim in data_array.dims if dim not in spatial]
            if not non_spatial_dims:
                expanded.append(data_array.to_dataset(name=var_name))
                continue
            dim_ranges = [range(int(data_array.sizes[dim])) for dim in non_spatial_dims]
            for indexes in itertools.product(*dim_ranges):
                selection = dict(zip(non_spatial_dims, indexes))
                sliced = data_array.isel(selection)
                if sliced.ndim != 2:
                    raise ValueError(
                        f"Expanded {var_name} with selection {selection} to ndim={sliced.ndim}, expected 2."
                    )
                expanded.append(sliced.to_dataset(name=var_name))
    return expanded


def rows_by_forecast_hour(rows: list[dict[str, object]]) -> dict[int, list[dict[str, object]]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        forecast_hour = int(row["forecast_hour"])
        grouped.setdefault(forecast_hour, []).append(row)
    return grouped


def filter_rows_to_selected_target_dates(
    rows: list[dict[str, object]],
    selected_target_dates: tuple[dt.date, ...],
) -> list[dict[str, object]]:
    if not selected_target_dates:
        return rows
    target_dates = {target_date.isoformat() for target_date in selected_target_dates}
    return [
        row
        for row in rows
        if str(row.get("valid_date_local")) in target_dates
    ]


def validate_batch_row_forecast_hours(
    *,
    expected_leads: set[int],
    wide_by_lead: dict[int, list[dict[str, object]]],
    long_by_lead: dict[int, list[dict[str, object]]],
    provenance_by_lead: dict[int, list[dict[str, object]]],
) -> None:
    unexpected = (set(wide_by_lead) | set(long_by_lead) | set(provenance_by_lead)) - expected_leads
    if unexpected:
        raise ValueError(
            "Batch extract produced row forecast_hour values outside admitted leads: "
            f"unexpected={sorted(unexpected)} expected={sorted(expected_leads)}"
        )


def batch_artifact_paths(args: argparse.Namespace, cycle_plan: CyclePlan) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = scratch_root(args) or args.output_dir
    batch_root = root / "batch" / f"init_date_local={cycle_plan.init_date_local.isoformat()}"
    batch_raw_path = batch_root / f"{cycle_plan.cycle_token}.selected_multilead.grib2"
    batch_reduced_path = batch_root / f"{cycle_plan.cycle_token}.selected_multilead.subset.grib2"
    batch_cfgrib_index_dir = batch_root / f"{cycle_plan.cycle_token}.cfgrib_index"
    return batch_raw_path, batch_reduced_path, batch_cfgrib_index_dir


def _process_unit_once(
    *,
    args: argparse.Namespace,
    client: S3HttpClient,
    cycle_plan: CyclePlan,
    lead_hour: int,
    phase_limits: PhaseConcurrencyLimits,
    progress: ProgressBar | None = None,
    reporter=None,
    attempt: int = 1,
    max_attempts: int = 1,
) -> UnitResult:
    worker_id = threading.current_thread().name
    worker_label = f"{cycle_plan.cycle_token} f{lead_hour:03d}"
    if reporter is not None:
        reporter.start_worker(
            worker_id,
            label=worker_label,
            phase="idx",
            group_id=cycle_plan.cycle_token,
            details="fetch_idx",
        )
        reporter.set_worker_attempt(worker_id, attempt=attempt, max_attempts=max_attempts)
    current_crop_bounds = CropBounds(top=args.top, bottom=args.bottom, left=args.left, right=args.right)
    key = resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, lead_hour, args.region)
    raw_url = f"{GRIB2_BASE}/{key}"
    idx_key = f"{key}.idx"
    progress_label = f"{cycle_plan.cycle_token}:f{lead_hour:03d}"
    raw_dir = raw_root(args)
    reduced_dir = reduced_root(args, cycle_plan)
    raw_path = raw_dir / key
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path = raw_dir / idx_key
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    reduced_path = reduced_dir / f"{pathlib.Path(key).stem}.subset.grib2"
    reduced_path.parent.mkdir(parents=True, exist_ok=True)
    idx_progress_label = f"{progress_label}:idx" if progress_label else None
    range_merge_gap_bytes = max(0, int(getattr(args, "range_merge_gap_bytes", DEFAULT_RANGE_MERGE_GAP_BYTES)))
    idx_was_present = idx_path.exists()
    if idx_was_present and not args.overwrite:
        idx_started_at = time.perf_counter()
        if progress is not None:
            progress.update(stage="idx", status=f"lead=f{lead_hour:03d} reuse_idx")
        if reporter is not None:
            reporter.update_worker(worker_id, phase="idx", details="reuse_idx")
        idx_text = idx_path.read_text()
        idx_fetch_seconds = stage_elapsed_seconds(idx_started_at)
    else:
        idx_url = f"{GRIB2_BASE}/{idx_key}"
        idx_started_at = time.perf_counter()
        if progress is not None:
            progress.update(stage="idx", status=f"lead=f{lead_hour:03d} fetch_idx")
        if reporter is not None:
            reporter.update_worker(worker_id, phase="idx", details="fetch_idx")
        if idx_progress_label and (reporter is None or reporter.mode == "log"):
            emit_progress_message(f"download_start label={idx_progress_label} file={idx_path.name} total=unknown")
        idx_text = client.fetch_text(idx_url)
        idx_path.write_text(idx_text)
        idx_fetch_seconds = stage_elapsed_seconds(idx_started_at)
        if idx_progress_label and (reporter is None or reporter.mode == "log"):
            final_mb = idx_path.stat().st_size / (1024 * 1024)
            emit_progress_message(f"download_complete label={idx_progress_label} file={idx_path.name} size_mb={final_mb:.1f}")
    idx_hash = sha256_text(idx_text)
    parse_started_at = time.perf_counter()
    if progress is not None:
        progress.update(stage="parse", status=f"lead=f{lead_hour:03d} parse_idx")
    if reporter is not None:
        reporter.update_worker(worker_id, phase="parse", details="parse_idx")
    inventory_records = parse_idx_lines(idx_text)
    selected_fields, warnings, missing_required = select_inventory_records(inventory_records)
    idx_parse_seconds = stage_elapsed_seconds(parse_started_at)
    # Treat availability as the issue/init time so historical rebuilds remain
    # eligible under the overnight 00:05 cutoff instead of using wall-clock run time.
    processed_at = cycle_plan.init_time_utc.isoformat()
    selected_record_count = sum(len(field.records) for field in selected_fields)

    manifest_row = {
        **make_base_time_columns(
            cycle_plan,
            cycle_plan.init_time_utc + dt.timedelta(hours=lead_hour),
            lead_hour,
            crop_bounds=current_crop_bounds,
        ),
        "source_url": raw_url,
        "raw_file_path": str(raw_path),
        "idx_file_path": str(idx_path),
        "download_mode": "byte_range_selected_records",
        "remote_file_size": None,
        "raw_file_name": raw_path.name,
        "raw_file_size": 0,
        "selected_record_count": selected_record_count,
        "selected_records_hash": selected_records_identity_hash(selected_fields) if selected_fields else None,
        "selected_download_bytes": 0,
        "downloaded_range_bytes": 0,
        "merged_range_count": 0,
        "head_used": False,
        "range_merge_gap_bytes": range_merge_gap_bytes,
        "reduced_file_path": str(reduced_path),
        "reduced_file_size": None,
        "idx_sha256": idx_hash,
        "extraction_status": "ok",
        "subset_command": None,
        "processed_timestamp_utc": processed_at,
        "timing_idx_fetch_seconds": idx_fetch_seconds,
        "timing_idx_parse_seconds": idx_parse_seconds,
        "timing_head_seconds": 0.0,
        "timing_range_download_seconds": 0.0,
        "timing_crop_seconds": 0.0,
        "timing_cfgrib_open_seconds": 0.0,
        "timing_row_build_seconds": 0.0,
        "timing_cleanup_seconds": 0.0,
        "raw_deleted": False,
        "idx_deleted": False,
        "reduced_deleted": False,
        "reduced_retained": bool(getattr(args, "keep_reduced", False)),
        "reduced_reused": False,
        "reduced_reuse_signature": None,
        "cfgrib_index_strategy": DEFAULT_CFGRIB_INDEX_STRATEGY,
        **default_crop_manifest_fields(),
        **default_batch_manifest_fields(),
        **default_post_crop_manifest_counters(),
        "provenance_written": provenance_enabled(args),
        "scratch_dir": str(scratch_root(args)) if scratch_root(args) is not None else None,
        "warnings": "; ".join(warnings + missing_required),
        "attempt_count": attempt,
        "retried": False,
        "retry_recovered": False,
        "final_error_class": None,
        "last_error_message": None,
    }

    if missing_required:
        manifest_row["extraction_status"] = "error:missing_required_fields"
        return UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row=manifest_row,
        )

    if selected_record_count == 0:
        manifest_row["extraction_status"] = "no_matching_records"
        return UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row=manifest_row,
        )

    try:
        remote_file_size: int | None = None
        if selected_ranges_require_content_length(inventory_records, selected_fields):
            with phase_gate(
                semaphore=phase_limits.download_semaphore,
                reporter=reporter,
                worker_id=worker_id,
                wait_phase="download_wait",
                active_phase="download",
                details="inspect_remote_size",
            ):
                head_started_at = time.perf_counter()
                if progress is not None:
                    progress.update(stage="download", status=f"lead=f{lead_hour:03d} inspect_remote_size")
                remote_file_size = client.fetch_content_length(raw_url)
                manifest_row["timing_head_seconds"] = stage_elapsed_seconds(head_started_at)
                manifest_row["head_used"] = True
        selected_ranges = build_selected_ranges(
            inventory_records,
            selected_fields,
            content_length=remote_file_size if remote_file_size is not None else 0,
        )
        manifest_row["remote_file_size"] = remote_file_size
        manifest_row["selected_record_count"] = len(selected_ranges)
        manifest_row["selected_download_bytes"] = sum(item.byte_length for item in selected_ranges)
        merged_ranges = merge_selected_ranges(
            selected_ranges,
            max_gap_bytes=range_merge_gap_bytes,
        )
        manifest_row["merged_range_count"] = len(merged_ranges)
        manifest_row["downloaded_range_bytes"] = sum(item.byte_length for item in merged_ranges)
        if should_reuse_cached_raw_grib(
            raw_path=raw_path,
            idx_was_present=idx_was_present,
            overwrite=args.overwrite,
            expected_selected_bytes=int(manifest_row["downloaded_range_bytes"]),
        ):
            if progress is not None:
                progress.update(stage="download", status=f"lead=f{lead_hour:03d} reuse_raw")
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="reuse_raw")
            manifest_row["raw_file_size"] = raw_path.stat().st_size
        else:
            with phase_gate(
                semaphore=phase_limits.download_semaphore,
                reporter=reporter,
                worker_id=worker_id,
                wait_phase="download_wait",
                active_phase="download",
                details="byte_range_download",
            ):
                download_started_at = time.perf_counter()
                if progress is not None:
                    progress.update(stage="download", status=f"lead=f{lead_hour:03d} byte_range_download")
                if reporter is not None:
                    reporter.start_transfer(
                        worker_id,
                        file_label=raw_path.name,
                        total_bytes=int(manifest_row["downloaded_range_bytes"]) or None,
                    )
                download_kwargs = {
                    "url": raw_url,
                    "ranges": [(item.byte_start, item.byte_end) for item in merged_ranges],
                    "destination": raw_path,
                    "overwrite": args.overwrite,
                    "progress_label": progress_label if (reporter is None or reporter.mode == "log") else None,
                    "progress_callback": (
                        (lambda downloaded, total, _elapsed: reporter.update_transfer(
                            worker_id,
                            bytes_downloaded=downloaded,
                            total_bytes=total,
                        ))
                        if reporter is not None
                        else None
                    ),
                }
                try:
                    client.download_byte_ranges(**download_kwargs)
                except TypeError as exc:
                    if "progress_callback" not in str(exc):
                        raise
                    download_kwargs.pop("progress_callback", None)
                    client.download_byte_ranges(**download_kwargs)
                manifest_row["timing_range_download_seconds"] = stage_elapsed_seconds(download_started_at)
                manifest_row["raw_file_size"] = raw_path.stat().st_size
                if reporter is not None:
                    reporter.finish_transfer(worker_id)
        reuse_signature: str | None = None
        if getattr(args, "keep_reduced", False):
            reuse_signature = build_reduced_reuse_signature(
                raw_path=raw_path,
                cycle_plan=cycle_plan,
                lead_hour=lead_hour,
                crop_bounds=current_crop_bounds,
                region=args.region,
                idx_sha256=idx_hash,
                selected_records_hash=manifest_row["selected_records_hash"],
            )
            manifest_row["reduced_reuse_signature"] = reuse_signature
        if reuse_signature is not None and not args.overwrite and reduced_grib_reusable(reduced_path=reduced_path, signature=reuse_signature):
            manifest_row["reduced_reused"] = True
            manifest_row["subset_command"] = "reuse_reduced_grib"
            manifest_row["crop_method_used"] = "reused"
            manifest_row["reduced_file_size"] = reduced_path.stat().st_size if reduced_path.exists() else None
            if reporter is not None:
                reporter.update_worker(worker_id, phase="crop", details="reuse_reduced")
        else:
            with phase_gate(
                semaphore=phase_limits.reduce_semaphore,
                reporter=reporter,
                worker_id=worker_id,
                wait_phase="reduce_wait",
                active_phase="crop",
                details="crop_selected_grib2",
            ):
                crop_started_at = time.perf_counter()
                if progress is not None:
                    progress.update(stage="crop", status=f"lead=f{lead_hour:03d} crop_selected_grib2")
                crop_result = coerce_crop_execution_result(
                    crop_selected_grib2(
                        args=args,
                        raw_path=raw_path,
                        reduced_path=reduced_path,
                        left=args.left,
                        right=args.right,
                        bottom=args.bottom,
                        top=args.top,
                    ),
                    args=args,
                )
                manifest_row["subset_command"] = crop_result.command
                manifest_row["crop_method_used"] = crop_result.method_used
                manifest_row["crop_grid_cache_key"] = crop_result.crop_grid_cache_key
                manifest_row["crop_grid_cache_hit"] = crop_result.crop_grid_cache_hit
                manifest_row["crop_ij_box"] = crop_result.crop_ij_box
                manifest_row["crop_wgrib2_threads"] = crop_result.crop_wgrib2_threads
                manifest_row["reduced_file_size"] = reduced_path.stat().st_size if reduced_path.exists() else None
                if reuse_signature is not None:
                    write_reduced_reuse_signature(reduced_path, reuse_signature)
                manifest_row["timing_crop_seconds"] = stage_elapsed_seconds(crop_started_at)
        with phase_gate(
            semaphore=phase_limits.extract_semaphore,
            reporter=reporter,
            worker_id=worker_id,
            wait_phase="extract_wait",
            active_phase="open",
            details="open_grouped_datasets",
        ):
            cfgrib_started_at = time.perf_counter()
            if progress is not None:
                progress.update(stage="extract", status=f"lead=f{lead_hour:03d} open_grouped_datasets")
            cfgrib_parent = cfgrib_index_base_dir(args)
            persistent_cfgrib_index_dir = cfgrib_index_cache_dir(
                args=args,
                reduced_path=reduced_path,
                reuse_signature=reuse_signature,
            )
            if persistent_cfgrib_index_dir is not None:
                persistent_cfgrib_index_dir.mkdir(parents=True, exist_ok=True)
                cfgrib_index_dir = persistent_cfgrib_index_dir
                manifest_row["cfgrib_index_strategy"] = "persistent_cache_per_reduced_signature"
            else:
                if cfgrib_parent is not None:
                    cfgrib_parent.mkdir(parents=True, exist_ok=True)
                cfgrib_index_dir = pathlib.Path(
                    tempfile.mkdtemp(
                        prefix=f"nbm_cfgrib_{lead_hour:03d}_",
                        dir=str(cfgrib_parent) if cfgrib_parent is not None else None,
                    )
                )
            datasets: list[xr.Dataset] = []
            try:
                open_stats: dict[str, object] = {}
                datasets = open_grouped_datasets(
                    reduced_path,
                    index_dir=cfgrib_index_dir,
                    selected=selected_fields,
                    stats=open_stats,
                )
                manifest_row.update(open_stats)
                manifest_row["timing_cfgrib_open_seconds"] = stage_elapsed_seconds(cfgrib_started_at)
                row_build_started_at = time.perf_counter()
                if progress is not None:
                    progress.update(stage="extract", status=f"lead=f{lead_hour:03d} build_rows")
                if reporter is not None:
                    reporter.update_worker(worker_id, phase="extract", details="build_rows")
                row_stats: dict[str, object] = {}
                wide_rows, long_rows, provenance_rows = build_rows_from_datasets(
                    datasets=datasets,
                    selected=selected_fields,
                    cycle_plan=cycle_plan,
                    lead_hour=lead_hour,
                    crop_bounds=current_crop_bounds,
                    allow_unknown_long_inference=(
                        int(manifest_row["downloaded_range_bytes"]) == int(manifest_row["selected_download_bytes"])
                    ),
                    write_long=bool(getattr(args, "write_long", False)),
                    write_provenance=not bool(getattr(args, "skip_provenance", False)),
                    metric_profile=str(getattr(args, "metric_profile", "full")),
                    stats=row_stats,
                )
                manifest_row.update(row_stats)
                manifest_row["timing_row_build_seconds"] = stage_elapsed_seconds(row_build_started_at)
            finally:
                for dataset in datasets:
                    dataset.close()
                if persistent_cfgrib_index_dir is None:
                    shutil.rmtree(cfgrib_index_dir, ignore_errors=True)
        primary_wide_row = wide_rows[0] if wide_rows else None
        manifest_row.update(
            {
                "valid_time_utc": primary_wide_row["valid_time_utc"] if primary_wide_row else manifest_row["valid_time_utc"],
                "valid_time_local": primary_wide_row["valid_time_local"] if primary_wide_row else manifest_row["valid_time_local"],
                "valid_date_local": primary_wide_row["valid_date_local"] if primary_wide_row else manifest_row["valid_date_local"],
                "fallback_used_any": bool(primary_wide_row.get("fallback_used_any", False)) if primary_wide_row else False,
            }
        )
        if len(wide_rows) > 1:
            manifest_row["warnings"] = "; ".join(
                filter(None, [manifest_row["warnings"], f"multiple_valid_times={len(wide_rows)}"])
            )
        cleanup_started_at = time.perf_counter()
        if reporter is not None:
            reporter.update_worker(worker_id, phase="cleanup", details="cleanup")
        if not getattr(args, "keep_downloads", False):
            raw_path.unlink(missing_ok=True)
            idx_path.unlink(missing_ok=True)
            manifest_row["raw_deleted"] = True
            manifest_row["idx_deleted"] = True
        if not getattr(args, "keep_reduced", False):
            reduced_path.unlink(missing_ok=True)
            reduced_signature_path(reduced_path).unlink(missing_ok=True)
            manifest_row["reduced_deleted"] = True
            manifest_row["reduced_retained"] = False
        manifest_row["timing_cleanup_seconds"] = stage_elapsed_seconds(cleanup_started_at)
        if progress is not None:
            progress.update(stage="write", status=f"lead=f{lead_hour:03d} unit_ready")
        return UnitResult(
            wide_row=primary_wide_row,
            wide_rows=wide_rows,
            long_rows=long_rows,
            provenance_rows=provenance_rows,
            manifest_row=manifest_row,
        )
    except Exception as exc:
        if progress is not None:
            progress.update(stage="error", status=f"lead=f{lead_hour:03d} {type(exc).__name__}")
        if reporter is not None:
            reporter.update_worker(worker_id, phase="cleanup", details=type(exc).__name__)
        cleanup_started_at = time.perf_counter()
        if not getattr(args, "keep_downloads", False):
            raw_path.unlink(missing_ok=True)
            idx_path.unlink(missing_ok=True)
            manifest_row["idx_deleted"] = True
        if reduced_path.exists() and not getattr(args, "keep_reduced", False):
            reduced_path.unlink(missing_ok=True)
            reduced_signature_path(reduced_path).unlink(missing_ok=True)
            manifest_row["reduced_deleted"] = True
        manifest_row["timing_cleanup_seconds"] = stage_elapsed_seconds(cleanup_started_at)
        manifest_row["raw_deleted"] = not raw_path.exists()
        manifest_row["extraction_status"] = f"error:{type(exc).__name__}"
        error_message = str(exc)
        manifest_row["last_error_message"] = error_message
        manifest_row["warnings"] = "; ".join(filter(None, [manifest_row["warnings"], error_message]))
        return UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row=manifest_row,
        )


def _sleep_with_retry_refresh(delay_seconds: float, reporter) -> None:
    remaining = max(0.0, float(delay_seconds))
    while remaining > 0:
        sleep_seconds = min(0.2, remaining)
        time.sleep(sleep_seconds)
        remaining -= sleep_seconds
        if reporter is not None:
            reporter.refresh(force=True)


def process_unit(
    *,
    args: argparse.Namespace,
    client: S3HttpClient,
    cycle_plan: CyclePlan,
    lead_hour: int,
    phase_limits: PhaseConcurrencyLimits | None = None,
    progress: ProgressBar | None = None,
    reporter=None,
) -> UnitResult:
    if phase_limits is None:
        phase_limits = build_phase_concurrency_limits(
            lead_workers=max(1, int(getattr(args, "lead_workers", 1))),
            args=args,
        )
    max_attempts = max(1, int(getattr(args, "max_task_attempts", 6)))
    retry_policy = RetryPolicy(
        max_attempts=max_attempts,
        backoff_seconds=float(getattr(args, "retry_backoff_seconds", 2.0)),
        max_backoff_seconds=float(getattr(args, "retry_max_backoff_seconds", 30.0)),
    )
    worker_id = threading.current_thread().name
    worker_label = f"{cycle_plan.cycle_token} f{lead_hour:03d}"
    last_result: UnitResult | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        if reporter is not None and attempt > 1:
            reporter.start_retry(worker_id, attempt=attempt, max_attempts=retry_policy.max_attempts)
        result = _process_unit_once(
            args=args,
            client=client,
            cycle_plan=cycle_plan,
            lead_hour=lead_hour,
            phase_limits=phase_limits,
            progress=progress,
            reporter=reporter,
            attempt=attempt,
            max_attempts=retry_policy.max_attempts,
        )
        last_result = result
        result.manifest_row["attempt_count"] = attempt
        result.manifest_row["retried"] = attempt > 1

        extraction_status = str(result.manifest_row.get("extraction_status", ""))
        success = extraction_status == "ok"
        skipped = extraction_status == "no_matching_records"
        if success:
            result.manifest_row["retry_recovered"] = attempt > 1
            result.manifest_row["final_error_class"] = None
            if reporter is not None:
                if attempt > 1:
                    reporter.recover_worker(worker_id, message=f"{worker_label} recovered a{attempt}/{retry_policy.max_attempts}")
                reporter.complete_worker(worker_id, message=f"{worker_label} ok")
            return result
        if skipped:
            result.manifest_row["retry_recovered"] = False
            result.manifest_row["final_error_class"] = None
            if reporter is not None:
                reporter.complete_worker(worker_id, message=f"{worker_label} no_matching_records", outcome="skipped")
            return result

        current_phase = None
        if reporter is not None:
            worker = reporter.state.workers.get(worker_id)
            current_phase = worker.phase if worker is not None else None
        message = str(result.manifest_row.get("warnings") or extraction_status or "unknown error")
        result.manifest_row["last_error_message"] = message
        exception_type = extraction_status.split("error:", 1)[1] if extraction_status.startswith("error:") else None
        decision = classify_task_failure(exception_type=exception_type, message=message, phase=current_phase)
        result.manifest_row["final_error_class"] = decision.error_class
        if should_retry_attempt(attempt=attempt, policy=retry_policy, decision=decision):
            delay_seconds = compute_retry_delay_seconds(attempt=attempt + 1, policy=retry_policy)
            if reporter is not None:
                reporter.schedule_retry(
                    worker_id,
                    attempt=attempt + 1,
                    max_attempts=retry_policy.max_attempts,
                    delay_seconds=delay_seconds,
                    message=message,
                    error_class=decision.error_class,
                )
            _sleep_with_retry_refresh(delay_seconds, reporter)
            continue

        result.manifest_row["retry_recovered"] = False
        if reporter is not None:
            reporter.fail_worker(worker_id, message=f"{worker_label} {message}")
        return result

    assert last_result is not None
    return last_result


def process_cycle(
    *,
    args: argparse.Namespace,
    client: S3HttpClient,
    cycle_plan: CyclePlan,
    phase_limits: PhaseConcurrencyLimits | None = None,
    reporter=None,
) -> list[UnitResult]:
    lead_workers = max(1, int(getattr(args, "lead_workers", 1)))
    lead_hours = lead_hours_for_cycle(cycle_plan)
    if phase_limits is None:
        phase_limits = build_phase_concurrency_limits(lead_workers=lead_workers, args=args)
    if reporter is None or reporter.mode == "log":
        emit_progress_message(
            f"cycle_start init_time_utc={cycle_plan.init_time_utc.isoformat()} "
            f"init_time_local={cycle_plan.init_time_local.isoformat()} leads={len(lead_hours)}"
        )
    progress = None
    if reporter is not None:
        reporter.upsert_group(
            cycle_plan.cycle_token,
            label=cycle_plan.cycle_token,
            total=len(lead_hours),
            completed=0,
            failed=0,
            status="queued",
        )
        reporter.log_event(f"{cycle_plan.cycle_token} queued leads={len(lead_hours)}")
    results_by_lead: dict[int, UnitResult] = {}
    ok_count = 0
    failed_count = 0
    if lead_workers == 1:
        for lead_index, lead_hour in enumerate(lead_hours, start=1):
            if reporter is not None:
                reporter.upsert_group(
                    cycle_plan.cycle_token,
                    completed=ok_count,
                    failed=failed_count,
                    status=f"active f{lead_hour:03d}",
                )
            result = process_unit(
                args=args,
                client=client,
                cycle_plan=cycle_plan,
                lead_hour=lead_hour,
                phase_limits=phase_limits,
                progress=progress,
                reporter=reporter,
            )
            results_by_lead[lead_hour] = result
            if result.manifest_row["extraction_status"] == "ok":
                ok_count += 1
            else:
                failed_count += 1
            if reporter is not None:
                reporter.upsert_group(
                    cycle_plan.cycle_token,
                    completed=ok_count,
                    failed=failed_count,
                    status=f"completed={lead_index}/{len(lead_hours)}",
                )
            if (reporter is None or reporter.mode == "log") and (
                lead_index % 6 == 0 or lead_index == len(lead_hours) or result.manifest_row["extraction_status"] != "ok"
            ):
                emit_progress_message(
                    f"lead_checkpoint init_time_utc={cycle_plan.init_time_utc.isoformat()} "
                    f"completed={lead_index}/{len(lead_hours)} lead=f{lead_hour:03d} "
                    f"result={result.manifest_row['extraction_status']}"
                )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=lead_workers) as executor:
            futures = {
                executor.submit(
                    process_unit,
                    args=args,
                    client=client,
                    cycle_plan=cycle_plan,
                    lead_hour=lead_hour,
                    phase_limits=phase_limits,
                    progress=None,
                    reporter=reporter,
                ): lead_hour
                for lead_hour in lead_hours
            }
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                lead_hour = futures[future]
                result = future.result()
                results_by_lead[lead_hour] = result
                if result.manifest_row["extraction_status"] == "ok":
                    ok_count += 1
                else:
                    failed_count += 1
                if reporter is not None:
                    reporter.upsert_group(
                        cycle_plan.cycle_token,
                        completed=ok_count,
                        failed=failed_count,
                        status=f"completed={completed_count}/{len(lead_hours)}",
                    )
                if (reporter is None or reporter.mode == "log") and (
                    completed_count % 6 == 0 or completed_count == len(lead_hours) or result.manifest_row["extraction_status"] != "ok"
                ):
                    emit_progress_message(
                        f"lead_checkpoint init_time_utc={cycle_plan.init_time_utc.isoformat()} "
                        f"completed={completed_count}/{len(lead_hours)} lead=f{lead_hour:03d} "
                        f"result={result.manifest_row['extraction_status']}"
                    )
    if reporter is not None:
        reporter.upsert_group(
            cycle_plan.cycle_token,
            completed=ok_count,
            failed=failed_count,
            status="ready_to_write",
        )
    return [results_by_lead[lead_hour] for lead_hour in lead_hours]


_ORIGINAL_PROCESS_CYCLE = process_cycle


def _run_pipeline_legacy(args: argparse.Namespace, client: S3HttpClient | None = None) -> int:
    start_local = normalize_date(args.start_local_date)
    end_local = normalize_date(args.end_local_date)
    if end_local < start_local:
        raise SystemExit("--end-local-date must be on or after --start-local-date")

    lead_workers = max(1, int(getattr(args, "lead_workers", 1)))
    shared_phase_limits = build_phase_concurrency_limits(lead_workers=lead_workers, args=args)
    client = client or S3HttpClient(pool_maxsize=max(8, args.workers * lead_workers * 2))
    discovery_days = utc_days_for_local_window(start_local, end_local)
    cycle_plans = discover_cycle_plans(
        start_local,
        end_local,
        client=client,
        progress=None,
        selection_mode=getattr(args, "selection_mode", DEFAULT_SELECTION_MODE),
    )
    if not cycle_plans:
        raise SystemExit("No GRIB2 cycles found for the requested local-date window.")
    planned_lead_count = total_lead_hours_for_cycles(cycle_plans)

    run_control = RunControl()
    reporter = create_progress_reporter(
        "NBM build",
        unit="lead",
        total=planned_lead_count,
        mode=getattr(args, "progress_mode", "auto"),
        stream=sys.stdout,
        on_pause_request=lambda **kwargs: run_control.request_pause(reason=str(kwargs.get("reason", "operator"))),
        enable_dashboard_hotkeys=not bool(getattr(args, "disable_dashboard_hotkeys", False)),
        pause_control_file=getattr(args, "pause_control_file", None),
    )
    reporter.set_metrics(
        cycles_total=len(cycle_plans),
        active_cycles=0,
        completed_cycles=0,
        skipped_cycles=0,
        planned_leads=planned_lead_count,
        download_workers=_resolve_phase_cap(getattr(args, "download_workers", None), default=lead_workers, outer_limit=lead_workers),
        reduce_workers=_resolve_phase_cap(getattr(args, "reduce_workers", None), default=lead_workers, outer_limit=lead_workers),
        extract_workers=_resolve_phase_cap(getattr(args, "extract_workers", None), default=lead_workers, outer_limit=lead_workers),
    )
    reporter.log_event(f"discovered cycles={len(cycle_plans)} planned_leads={planned_lead_count}")
    if reporter.mode == "log":
        emit_progress_message(f"cycles={len(cycle_plans)} planned_leads={planned_lead_count}")
    if args.dry_run:
        planned = 0
        for cycle_plan in cycle_plans:
            lead_hours = lead_hours_for_cycle(cycle_plan)
            target_dates_text = ""
            if cycle_plan.selected_target_dates:
                target_dates_text = " " + ",".join(
                    [
                        f"target_date_local={target_date_local.isoformat()}"
                        for target_date_local in cycle_plan.selected_target_dates
                    ]
                )
            print(
                "plan:",
                cycle_plan.init_time_utc.isoformat(),
                cycle_plan.init_time_local.isoformat(),
                f"mode={cycle_plan.mode}",
                f"leads={lead_hours_summary(lead_hours)}",
                target_dates_text.strip(),
            )
            planned += len(lead_hours)
        print(f"planned_units={planned}")
        reporter.close(status=f"planned_units={planned}")
        return 0

    ensure_runtime_dependencies()

    active_cycles = 0
    completed_cycles = 0
    skipped_cycles = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        pending_cycle_plans: list[CyclePlan] = []

        def submit_cycle(cycle_plan: CyclePlan) -> concurrent.futures.Future[list[UnitResult]]:
            future = executor.submit(
                process_cycle,
                args=args,
                client=client,
                cycle_plan=cycle_plan,
                phase_limits=shared_phase_limits,
                reporter=reporter,
            )
            return future

        for cycle_plan in cycle_plans:
            lead_count = len(lead_hours_for_cycle(cycle_plan))
            if lead_count == 0:
                skipped_cycles += 1
                reporter.add_skipped(0, message=f"{cycle_plan.cycle_token} skipped no_target_day_leads")
                reporter.upsert_group(cycle_plan.cycle_token, label=cycle_plan.cycle_token, total=0, status="skipped_no_target_day_leads")
                continue
            if not args.overwrite and cycle_already_complete(args.output_dir, cycle_plan):
                skipped_cycles += 1
                reporter.add_skipped(
                    lead_count,
                    message=f"{cycle_plan.cycle_token} skipped manifest_complete",
                )
                reporter.set_metrics(
                    cycles_total=len(cycle_plans),
                    active_cycles=active_cycles,
                    completed_cycles=completed_cycles,
                    skipped_cycles=skipped_cycles,
                    planned_leads=planned_lead_count,
                )
                reporter.log_event(
                    f"skip_cycle init_time_utc={cycle_plan.init_time_utc.isoformat()} reason=manifest_complete",
                    level="info",
                )
                if reporter.mode == "log":
                    emit_progress_message(
                    f"skip_cycle init_time_utc={cycle_plan.init_time_utc.isoformat()} reason=manifest_complete"
                    )
                continue
            pending_cycle_plans.append(cycle_plan)

        futures: dict[concurrent.futures.Future[list[UnitResult]], CyclePlan] = {}
        pending_cycle_index = 0
        initial_submit = min(len(pending_cycle_plans), max(1, int(args.workers)))
        for _ in range(initial_submit):
            cycle_plan = pending_cycle_plans[pending_cycle_index]
            future = submit_cycle(cycle_plan)
            futures[future] = cycle_plan
            pending_cycle_index += 1
            active_cycles += 1
            reporter.set_metrics(
                cycles_total=len(cycle_plans),
                active_cycles=active_cycles,
                completed_cycles=completed_cycles,
                skipped_cycles=skipped_cycles,
                planned_leads=planned_lead_count,
            )
            reporter.upsert_group(cycle_plan.cycle_token, label=cycle_plan.cycle_token, total=len(lead_hours_for_cycle(cycle_plan)), status="submitted")

        while futures:
            done, _ = concurrent.futures.wait(futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                cycle_plan = futures.pop(future)
                reporter.upsert_group(cycle_plan.cycle_token, status="write_outputs")
                results = future.result()
                write_cycle_outputs(
                    args.output_dir,
                    cycle_plan,
                    results,
                    write_long=bool(getattr(args, "write_long", False)),
                    write_provenance=not bool(getattr(args, "skip_provenance", False)),
                )
                ok_count = sum(result.manifest_row["extraction_status"] == "ok" for result in results)
                active_cycles = max(0, active_cycles - 1)
                completed_cycles += 1
                reporter.set_metrics(
                    cycles_total=len(cycle_plans),
                    active_cycles=active_cycles,
                    completed_cycles=completed_cycles,
                    skipped_cycles=skipped_cycles,
                    planned_leads=planned_lead_count,
                )
                reporter.upsert_group(
                    cycle_plan.cycle_token,
                    completed=ok_count,
                    failed=len(results) - ok_count,
                    status="complete",
                )
                if reporter.mode == "log":
                    emit_progress_message(
                        f"completed_cycle init_time_utc={cycle_plan.init_time_utc.isoformat()} "
                        f"mode={cycle_plan.mode} ok_units={ok_count}/{len(results)}"
                    )
                if pending_cycle_index < len(pending_cycle_plans) and not run_control.pause_requested():
                    next_cycle_plan = pending_cycle_plans[pending_cycle_index]
                    next_future = submit_cycle(next_cycle_plan)
                    futures[next_future] = next_cycle_plan
                    pending_cycle_index += 1
                    active_cycles += 1
                    reporter.set_metrics(
                        cycles_total=len(cycle_plans),
                        active_cycles=active_cycles,
                        completed_cycles=completed_cycles,
                        skipped_cycles=skipped_cycles,
                        planned_leads=planned_lead_count,
                    )
                    reporter.upsert_group(next_cycle_plan.cycle_token, label=next_cycle_plan.cycle_token, total=len(lead_hours_for_cycle(next_cycle_plan)), status="submitted")

    if run_control.pause_requested():
        run_control.mark_paused(reason=run_control.pause_reason or "operator")
        reporter.mark_paused(reason=run_control.pause_reason or "operator")
    reporter.close(status="paused" if run_control.is_paused() else f"cycles={completed_cycles}")

    return 0


def run_pipeline(args: argparse.Namespace, client: S3HttpClient | None = None) -> int:
    if process_cycle is not _ORIGINAL_PROCESS_CYCLE:
        return _run_pipeline_legacy(args, client=client)

    start_local = normalize_date(args.start_local_date)
    end_local = normalize_date(args.end_local_date)
    if end_local < start_local:
        raise SystemExit("--end-local-date must be on or after --start-local-date")

    lead_workers = max(1, int(getattr(args, "lead_workers", 1)))
    batch_reduce_mode = str(getattr(args, "batch_reduce_mode", "off"))
    phase_limits = build_phase_concurrency_limits(lead_workers=lead_workers, args=args)
    client = client or S3HttpClient(pool_maxsize=max(8, lead_workers * max(1, args.workers) * 2))
    cycle_plans = discover_cycle_plans(
        start_local,
        end_local,
        client=client,
        progress=None,
        selection_mode=getattr(args, "selection_mode", DEFAULT_SELECTION_MODE),
    )
    if not cycle_plans:
        raise SystemExit("No GRIB2 cycles found for the requested local-date window.")
    planned_lead_count = total_lead_hours_for_cycles(cycle_plans)

    run_control = RunControl()
    reporter = create_progress_reporter(
        "NBM build",
        unit="lead",
        total=planned_lead_count,
        mode=getattr(args, "progress_mode", "auto"),
        stream=sys.stdout,
        on_pause_request=lambda **kwargs: run_control.request_pause(reason=str(kwargs.get("reason", "operator"))),
        enable_dashboard_hotkeys=not bool(getattr(args, "disable_dashboard_hotkeys", False)),
        pause_control_file=getattr(args, "pause_control_file", None),
    )

    phase_activity_lock = threading.Lock()
    phase_activity = {"download": 0, "reduce": 0, "extract": 0}

    def update_phase_activity(phase_name: str, delta: int) -> None:
        with phase_activity_lock:
            phase_activity[phase_name] = max(0, phase_activity[phase_name] + delta)

    def current_phase_activity() -> tuple[int, int, int]:
        with phase_activity_lock:
            return phase_activity["download"], phase_activity["reduce"], phase_activity["extract"]

    def update_pipeline_metrics(*, active_cycles: int, completed_cycles: int, skipped_cycles: int, reduce_queue_obj, extract_queue_obj) -> None:
        active_download, active_reduce, active_extract = current_phase_activity()
        reporter.set_metrics(
            cycles_total=len(cycle_plans),
            active_cycles=active_cycles,
            completed_cycles=completed_cycles,
            skipped_cycles=skipped_cycles,
            planned_leads=planned_lead_count,
            download_workers=phase_limits.download_workers,
            reduce_workers=phase_limits.reduce_workers,
            extract_workers=phase_limits.extract_workers,
            reduce_queued=reduce_queue_obj.qsize(),
            extract_queued=extract_queue_obj.qsize(),
            active_download=active_download,
            active_reduce=active_reduce,
            active_extract=active_extract,
        )

    reporter.log_event(f"discovered cycles={len(cycle_plans)} planned_leads={planned_lead_count}")
    if reporter.mode == "log":
        emit_progress_message(f"cycles={len(cycle_plans)} planned_leads={planned_lead_count}")
    if args.dry_run:
        planned = 0
        for cycle_plan in cycle_plans:
            lead_hours = lead_hours_for_cycle(cycle_plan)
            target_dates_text = ""
            if cycle_plan.selected_target_dates:
                target_dates_text = " " + ",".join(
                    [f"target_date_local={target_date_local.isoformat()}" for target_date_local in cycle_plan.selected_target_dates]
                )
            print(
                "plan:",
                cycle_plan.init_time_utc.isoformat(),
                cycle_plan.init_time_local.isoformat(),
                f"mode={cycle_plan.mode}",
                f"selection_mode={getattr(args, 'selection_mode', DEFAULT_SELECTION_MODE)}",
                f"leads={lead_hours_summary(lead_hours)}",
                target_dates_text.strip(),
            )
            planned += len(lead_hours)
        print(f"planned_units={planned}")
        reporter.close(status=f"planned_units={planned}")
        return 0

    ensure_runtime_dependencies()

    cycle_worker_limit = max(1, int(getattr(args, "workers", 1)))
    active_cycles = 0
    completed_cycles = 0
    skipped_cycles = 0
    active_cycle_plans: list[CyclePlan] = []
    pending_cycle_plans: list[CyclePlan] = []
    aggregations: dict[str, CycleAggregationState] = {}
    for cycle_plan in cycle_plans:
        lead_count = len(lead_hours_for_cycle(cycle_plan))
        if lead_count == 0:
            skipped_cycles += 1
            reporter.add_skipped(0, message=f"{cycle_plan.cycle_token} skipped no_target_day_leads")
            reporter.upsert_group(cycle_plan.cycle_token, label=cycle_plan.cycle_token, total=0, status="skipped_no_target_day_leads")
            continue
        if not args.overwrite and cycle_already_complete(args.output_dir, cycle_plan):
            skipped_cycles += 1
            reporter.add_skipped(lead_count, message=f"{cycle_plan.cycle_token} skipped manifest_complete")
            reporter.upsert_group(cycle_plan.cycle_token, label=cycle_plan.cycle_token, total=lead_count, status="skipped")
            continue
        aggregations[cycle_plan.cycle_token] = CycleAggregationState(cycle_plan=cycle_plan, expected_leads=lead_count)
        reporter.upsert_group(cycle_plan.cycle_token, label=cycle_plan.cycle_token, total=lead_count, completed=0, failed=0, status="queued")
        pending_cycle_plans.append(cycle_plan)

    download_queue: queue.Queue[NbmPipelineItem | None] = queue.Queue()
    reduce_queue: queue.Queue[NbmPipelineItem | NbmBatchPipelineItem | None] = queue.Queue(
        maxsize=resolve_pipeline_queue_size(getattr(args, "reduce_queue_size", None), downstream_workers=phase_limits.reduce_workers)
    )
    extract_queue: queue.Queue[NbmPipelineItem | NbmBatchPipelineItem | None] = queue.Queue(
        maxsize=resolve_pipeline_queue_size(getattr(args, "extract_queue_size", None), downstream_workers=phase_limits.extract_workers)
    )
    completion_queue: queue.Queue[tuple[str, int, UnitResult]] = queue.Queue()
    batch_download_lock = threading.Lock()
    batch_downloaded_by_cycle: dict[str, list[NbmPipelineItem]] = {}
    batch_disabled_cycles: set[str] = set()
    update_pipeline_metrics(
        active_cycles=active_cycles,
        completed_cycles=completed_cycles,
        skipped_cycles=skipped_cycles,
        reduce_queue_obj=reduce_queue,
        extract_queue_obj=extract_queue,
    )

    def make_initial_item(cycle_plan: CyclePlan, lead_hour: int) -> NbmPipelineItem:
        current_crop_bounds = CropBounds(top=args.top, bottom=args.bottom, left=args.left, right=args.right)
        key = resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, lead_hour, args.region)
        raw_url = f"{GRIB2_BASE}/{key}"
        idx_key = f"{key}.idx"
        raw_dir = raw_root(args)
        reduced_dir = reduced_root(args, cycle_plan)
        raw_path = raw_dir / key
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path = raw_dir / idx_key
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path = reduced_dir / f"{pathlib.Path(key).stem}.subset.grib2"
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        return NbmPipelineItem(
            cycle_plan=cycle_plan,
            lead_hour=lead_hour,
            attempt_count=1,
            current_crop_bounds=current_crop_bounds,
            raw_path=raw_path,
            idx_path=idx_path,
            reduced_path=reduced_path,
            raw_url=raw_url,
        )

    def finalize_nbm_failure(item: NbmPipelineItem, exc: Exception, *, phase: str) -> UnitResult:
        manifest_row = dict(item.manifest_row)
        manifest_row.setdefault("extraction_status", f"error:{type(exc).__name__}")
        manifest_row["extraction_status"] = f"error:{type(exc).__name__}"
        manifest_row["last_error_message"] = str(exc)
        manifest_row["warnings"] = "; ".join(filter(None, [str(manifest_row.get("warnings") or ""), str(exc)]))
        cleanup_started_at = time.perf_counter()
        if not getattr(args, "keep_downloads", False):
            if item.raw_path is not None:
                item.raw_path.unlink(missing_ok=True)
            if item.idx_path is not None:
                item.idx_path.unlink(missing_ok=True)
            manifest_row["raw_deleted"] = True
            manifest_row["idx_deleted"] = True
        if item.reduced_path is not None and item.reduced_path.exists() and not getattr(args, "keep_reduced", False):
            item.reduced_path.unlink(missing_ok=True)
            reduced_signature_path(item.reduced_path).unlink(missing_ok=True)
            manifest_row["reduced_deleted"] = True
        manifest_row["timing_cleanup_seconds"] = stage_elapsed_seconds(cleanup_started_at)
        manifest_row["attempt_count"] = item.attempt_count
        manifest_row["retried"] = item.attempt_count > 1
        manifest_row["retry_recovered"] = False
        manifest_row["final_error_class"] = classify_task_failure(
            exception_type=type(exc).__name__,
            message=str(exc),
            phase=phase,
        ).error_class
        return UnitResult(wide_row=None, wide_rows=[], long_rows=[], provenance_rows=[], manifest_row=manifest_row)

    def run_download_stage(item: NbmPipelineItem, *, worker_id: str) -> NbmPipelineItem | UnitResult:
        cycle_plan = item.cycle_plan
        lead_hour = item.lead_hour
        worker_label = f"{cycle_plan.cycle_token} f{lead_hour:03d}"
        progress_label = f"{cycle_plan.cycle_token}:f{lead_hour:03d}"
        current_crop_bounds = item.current_crop_bounds
        assert current_crop_bounds is not None
        assert item.raw_path is not None and item.idx_path is not None and item.reduced_path is not None and item.raw_url is not None
        if reporter is not None:
            reporter.start_worker(worker_id, label=worker_label, phase="download", group_id=cycle_plan.cycle_token, details="idx")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=max(1, int(getattr(args, "max_task_attempts", 6))))
        key = resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, lead_hour, args.region)
        idx_key = f"{key}.idx"
        raw_url = item.raw_url
        idx_progress_label = f"{progress_label}:idx" if progress_label else None
        range_merge_gap_bytes = max(0, int(getattr(args, "range_merge_gap_bytes", DEFAULT_RANGE_MERGE_GAP_BYTES)))
        idx_was_present = item.idx_path.exists()
        if idx_was_present and not args.overwrite:
            idx_started_at = time.perf_counter()
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="reuse_idx")
            idx_text = item.idx_path.read_text()
            idx_fetch_seconds = stage_elapsed_seconds(idx_started_at)
        else:
            idx_url = f"{GRIB2_BASE}/{idx_key}"
            idx_started_at = time.perf_counter()
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="fetch_idx")
            if idx_progress_label and (reporter is None or reporter.mode == "log"):
                emit_progress_message(f"download_start label={idx_progress_label} file={item.idx_path.name} total=unknown")
            idx_text = client.fetch_text(idx_url)
            item.idx_path.write_text(idx_text)
            idx_fetch_seconds = stage_elapsed_seconds(idx_started_at)
            if idx_progress_label and (reporter is None or reporter.mode == "log"):
                final_mb = item.idx_path.stat().st_size / (1024 * 1024)
                emit_progress_message(f"download_complete label={idx_progress_label} file={item.idx_path.name} size_mb={final_mb:.1f}")
        idx_hash = sha256_text(idx_text)
        if reporter is not None:
            reporter.update_worker(worker_id, phase="download", details="parse_idx")
        parse_started_at = time.perf_counter()
        inventory_records = parse_idx_lines(idx_text)
        selected_fields, warnings_list, missing_required = select_inventory_records(inventory_records)
        idx_parse_seconds = stage_elapsed_seconds(parse_started_at)
        processed_at = cycle_plan.init_time_utc.isoformat()
        selected_record_count = sum(len(field.records) for field in selected_fields)
        manifest_row = {
            **make_base_time_columns(cycle_plan, cycle_plan.init_time_utc + dt.timedelta(hours=lead_hour), lead_hour, crop_bounds=current_crop_bounds),
            "source_url": raw_url,
            "raw_file_path": str(item.raw_path),
            "idx_file_path": str(item.idx_path),
            "download_mode": "byte_range_selected_records",
            "remote_file_size": None,
            "raw_file_name": item.raw_path.name,
            "raw_file_size": 0,
            "selected_record_count": selected_record_count,
            "selected_records_hash": selected_records_identity_hash(selected_fields) if selected_fields else None,
            "selected_download_bytes": 0,
            "downloaded_range_bytes": 0,
            "merged_range_count": 0,
            "head_used": False,
            "range_merge_gap_bytes": range_merge_gap_bytes,
            "reduced_file_path": str(item.reduced_path),
            "reduced_file_size": None,
            "idx_sha256": idx_hash,
            "extraction_status": "ok",
            "subset_command": None,
            "processed_timestamp_utc": processed_at,
            "timing_idx_fetch_seconds": idx_fetch_seconds,
            "timing_idx_parse_seconds": idx_parse_seconds,
            "timing_head_seconds": 0.0,
            "timing_range_download_seconds": 0.0,
            "timing_crop_seconds": 0.0,
            "timing_cfgrib_open_seconds": 0.0,
            "timing_row_build_seconds": 0.0,
            "timing_cleanup_seconds": 0.0,
            "raw_deleted": False,
            "idx_deleted": False,
            "reduced_deleted": False,
            "reduced_retained": bool(getattr(args, "keep_reduced", False)),
            "reduced_reused": False,
            "reduced_reuse_signature": None,
            "cfgrib_index_strategy": DEFAULT_CFGRIB_INDEX_STRATEGY,
            **default_crop_manifest_fields(),
            **default_batch_manifest_fields(),
            **default_post_crop_manifest_counters(),
            "provenance_written": provenance_enabled(args),
            "scratch_dir": str(scratch_root(args)) if scratch_root(args) is not None else None,
            "warnings": "; ".join(warnings_list + missing_required),
            "attempt_count": item.attempt_count,
            "retried": item.attempt_count > 1,
            "retry_recovered": False,
            "final_error_class": None,
            "last_error_message": None,
        }
        item.manifest_row = manifest_row
        item.selected_fields = selected_fields
        if missing_required:
            manifest_row["extraction_status"] = "error:missing_required_fields"
            return UnitResult(wide_row=None, wide_rows=[], long_rows=[], provenance_rows=[], manifest_row=manifest_row)
        if selected_record_count == 0:
            manifest_row["extraction_status"] = "no_matching_records"
            return UnitResult(wide_row=None, wide_rows=[], long_rows=[], provenance_rows=[], manifest_row=manifest_row)
        remote_file_size: int | None = None
        if selected_ranges_require_content_length(inventory_records, selected_fields):
            head_started_at = time.perf_counter()
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="inspect_remote_size")
            remote_file_size = client.fetch_content_length(raw_url)
            manifest_row["timing_head_seconds"] = stage_elapsed_seconds(head_started_at)
            manifest_row["head_used"] = True
        selected_ranges = build_selected_ranges(inventory_records, selected_fields, content_length=remote_file_size if remote_file_size is not None else 0)
        manifest_row["remote_file_size"] = remote_file_size
        manifest_row["selected_record_count"] = len(selected_ranges)
        manifest_row["selected_download_bytes"] = sum(item_range.byte_length for item_range in selected_ranges)
        merged_ranges = merge_selected_ranges(selected_ranges, max_gap_bytes=range_merge_gap_bytes)
        manifest_row["merged_range_count"] = len(merged_ranges)
        manifest_row["downloaded_range_bytes"] = sum(item_range.byte_length for item_range in merged_ranges)
        if should_reuse_cached_raw_grib(
            raw_path=item.raw_path,
            idx_was_present=idx_was_present,
            overwrite=args.overwrite,
            expected_selected_bytes=int(manifest_row["downloaded_range_bytes"]),
        ):
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="reuse_raw")
            manifest_row["raw_file_size"] = item.raw_path.stat().st_size
        else:
            if reporter is not None:
                reporter.update_worker(worker_id, phase="download", details="byte_range_download")
                reporter.start_transfer(
                    worker_id,
                    file_label=item.raw_path.name,
                    total_bytes=int(manifest_row["downloaded_range_bytes"]) or None,
                )
            download_started_at = time.perf_counter()
            download_kwargs = {
                "url": raw_url,
                "ranges": [(item_range.byte_start, item_range.byte_end) for item_range in merged_ranges],
                "destination": item.raw_path,
                "overwrite": args.overwrite,
                "progress_label": progress_label if (reporter is None or reporter.mode == "log") else None,
                "progress_callback": (
                    (lambda downloaded, total, _elapsed: reporter.update_transfer(worker_id, bytes_downloaded=downloaded, total_bytes=total))
                    if reporter is not None
                    else None
                ),
            }
            try:
                client.download_byte_ranges(**download_kwargs)
            except TypeError as exc:
                if "progress_callback" not in str(exc):
                    raise
                download_kwargs.pop("progress_callback", None)
                client.download_byte_ranges(**download_kwargs)
            manifest_row["timing_range_download_seconds"] = stage_elapsed_seconds(download_started_at)
            manifest_row["raw_file_size"] = item.raw_path.stat().st_size
            if reporter is not None:
                reporter.finish_transfer(worker_id)
        return item

    def run_reduce_stage(item: NbmPipelineItem, *, worker_id: str) -> NbmPipelineItem:
        assert item.raw_path is not None and item.reduced_path is not None and item.current_crop_bounds is not None
        reuse_signature: str | None = None
        if getattr(args, "keep_reduced", False):
            reuse_signature = build_reduced_reuse_signature(
                raw_path=item.raw_path,
                cycle_plan=item.cycle_plan,
                lead_hour=item.lead_hour,
                crop_bounds=item.current_crop_bounds,
                region=args.region,
                idx_sha256=item.manifest_row.get("idx_sha256"),
                selected_records_hash=item.manifest_row.get("selected_records_hash"),
            )
            item.manifest_row["reduced_reuse_signature"] = reuse_signature
        if reuse_signature is not None and not args.overwrite and reduced_grib_reusable(reduced_path=item.reduced_path, signature=reuse_signature):
            item.manifest_row["reduced_reused"] = True
            item.manifest_row["subset_command"] = "reuse_reduced_grib"
            item.manifest_row["crop_method_used"] = "reused"
            item.manifest_row["reduced_file_size"] = item.reduced_path.stat().st_size if item.reduced_path.exists() else None
            if reporter is not None:
                reporter.start_worker(worker_id, label=f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d}", phase="reduce", group_id=item.cycle_plan.cycle_token, details="reuse_reduced")
                reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=max(1, int(getattr(args, "max_task_attempts", 6))))
            return item
        if reporter is not None:
            reporter.start_worker(worker_id, label=f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d}", phase="reduce", group_id=item.cycle_plan.cycle_token, details="crop_selected_grib2")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=max(1, int(getattr(args, "max_task_attempts", 6))))
        crop_started_at = time.perf_counter()
        crop_result = coerce_crop_execution_result(
            crop_selected_grib2(
                args=args,
                raw_path=item.raw_path,
                reduced_path=item.reduced_path,
                left=args.left,
                right=args.right,
                bottom=args.bottom,
                top=args.top,
            ),
            args=args,
        )
        item.manifest_row["subset_command"] = crop_result.command
        item.manifest_row["crop_method_used"] = crop_result.method_used
        item.manifest_row["crop_grid_cache_key"] = crop_result.crop_grid_cache_key
        item.manifest_row["crop_grid_cache_hit"] = crop_result.crop_grid_cache_hit
        item.manifest_row["crop_ij_box"] = crop_result.crop_ij_box
        item.manifest_row["crop_wgrib2_threads"] = crop_result.crop_wgrib2_threads
        item.manifest_row["reduced_file_size"] = item.reduced_path.stat().st_size if item.reduced_path.exists() else None
        if reuse_signature is not None:
            write_reduced_reuse_signature(item.reduced_path, reuse_signature)
        item.manifest_row["timing_crop_seconds"] = stage_elapsed_seconds(crop_started_at)
        return item

    def run_extract_stage(item: NbmPipelineItem, *, worker_id: str) -> UnitResult:
        assert item.reduced_path is not None and item.current_crop_bounds is not None
        if reporter is not None:
            reporter.start_worker(worker_id, label=f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d}", phase="extract", group_id=item.cycle_plan.cycle_token, details="open_grouped_datasets")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=max(1, int(getattr(args, "max_task_attempts", 6))))
        cfgrib_started_at = time.perf_counter()
        cfgrib_parent = cfgrib_index_base_dir(args)
        persistent_cfgrib_index_dir = cfgrib_index_cache_dir(
            args=args,
            reduced_path=item.reduced_path,
            reuse_signature=item.manifest_row.get("reduced_reuse_signature"),
        )
        if persistent_cfgrib_index_dir is not None:
            persistent_cfgrib_index_dir.mkdir(parents=True, exist_ok=True)
            cfgrib_index_dir = persistent_cfgrib_index_dir
            item.manifest_row["cfgrib_index_strategy"] = "persistent_cache_per_reduced_signature"
        else:
            if cfgrib_parent is not None:
                cfgrib_parent.mkdir(parents=True, exist_ok=True)
            cfgrib_index_dir = pathlib.Path(
                tempfile.mkdtemp(
                    prefix=f"nbm_cfgrib_{item.lead_hour:03d}_",
                    dir=str(cfgrib_parent) if cfgrib_parent is not None else None,
                )
            )
        datasets: list[xr.Dataset] = []
        try:
            open_stats: dict[str, object] = {}
            datasets = open_grouped_datasets(
                item.reduced_path,
                index_dir=cfgrib_index_dir,
                selected=item.selected_fields,
                stats=open_stats,
            )
            item.manifest_row.update(open_stats)
            item.manifest_row["timing_cfgrib_open_seconds"] = stage_elapsed_seconds(cfgrib_started_at)
            row_build_started_at = time.perf_counter()
            if reporter is not None:
                reporter.update_worker(worker_id, phase="extract", details="build_rows")
            row_stats: dict[str, object] = {}
            wide_rows, long_rows, provenance_rows = build_rows_from_datasets(
                datasets=datasets,
                selected=item.selected_fields,
                cycle_plan=item.cycle_plan,
                lead_hour=item.lead_hour,
                crop_bounds=item.current_crop_bounds,
                allow_unknown_long_inference=(
                    int(item.manifest_row["downloaded_range_bytes"]) == int(item.manifest_row["selected_download_bytes"])
                ),
                write_long=bool(getattr(args, "write_long", False)),
                write_provenance=not bool(getattr(args, "skip_provenance", False)),
                metric_profile=str(getattr(args, "metric_profile", "full")),
                stats=row_stats,
            )
            item.manifest_row.update(row_stats)
            item.manifest_row["timing_row_build_seconds"] = stage_elapsed_seconds(row_build_started_at)
        finally:
            for dataset in datasets:
                dataset.close()
            if persistent_cfgrib_index_dir is None:
                shutil.rmtree(cfgrib_index_dir, ignore_errors=True)
        primary_wide_row = wide_rows[0] if wide_rows else None
        item.manifest_row.update(
            {
                "valid_time_utc": primary_wide_row["valid_time_utc"] if primary_wide_row else item.manifest_row["valid_time_utc"],
                "valid_time_local": primary_wide_row["valid_time_local"] if primary_wide_row else item.manifest_row["valid_time_local"],
                "valid_date_local": primary_wide_row["valid_date_local"] if primary_wide_row else item.manifest_row["valid_date_local"],
                "fallback_used_any": bool(primary_wide_row.get("fallback_used_any", False)) if primary_wide_row else False,
            }
        )
        if len(wide_rows) > 1:
            item.manifest_row["warnings"] = "; ".join(filter(None, [item.manifest_row["warnings"], f"multiple_valid_times={len(wide_rows)}"]))
        cleanup_started_at = time.perf_counter()
        if reporter is not None:
            reporter.update_worker(worker_id, phase="finalize", details="cleanup")
        if not getattr(args, "keep_downloads", False):
            assert item.raw_path is not None and item.idx_path is not None
            item.raw_path.unlink(missing_ok=True)
            item.idx_path.unlink(missing_ok=True)
            item.manifest_row["raw_deleted"] = True
            item.manifest_row["idx_deleted"] = True
        if not getattr(args, "keep_reduced", False):
            item.reduced_path.unlink(missing_ok=True)
            reduced_signature_path(item.reduced_path).unlink(missing_ok=True)
            item.manifest_row["reduced_deleted"] = True
            item.manifest_row["reduced_retained"] = False
        item.manifest_row["timing_cleanup_seconds"] = stage_elapsed_seconds(cleanup_started_at)
        return UnitResult(
            wide_row=primary_wide_row,
            wide_rows=wide_rows,
            long_rows=long_rows,
            provenance_rows=provenance_rows,
            manifest_row=item.manifest_row,
        )

    def make_batch_item(cycle_plan: CyclePlan, lead_items: list[NbmPipelineItem]) -> NbmBatchPipelineItem:
        batch_raw_path, batch_reduced_path, batch_cfgrib_index_dir = batch_artifact_paths(args, cycle_plan)
        return NbmBatchPipelineItem(
            cycle_plan=cycle_plan,
            lead_items=sorted(lead_items, key=lambda item: item.lead_hour),
            current_crop_bounds=CropBounds(top=args.top, bottom=args.bottom, left=args.left, right=args.right),
            batch_raw_path=batch_raw_path,
            batch_reduced_path=batch_reduced_path,
            batch_cfgrib_index_dir=batch_cfgrib_index_dir,
        )

    def apply_batch_timing_to_manifest_rows(
        batch_item: NbmBatchPipelineItem,
        *,
        open_seconds: float = 0.0,
        row_build_seconds: float = 0.0,
    ) -> None:
        lead_count = max(1, len(batch_item.lead_items))
        for index, lead_item in enumerate(batch_item.lead_items):
            row = lead_item.manifest_row
            row["batch_reduce_mode"] = "cycle"
            row["batch_raw_file_path"] = str(batch_item.batch_raw_path)
            row["batch_reduced_file_path"] = str(batch_item.batch_reduced_path)
            row["batch_lead_count"] = lead_count
            row["reduced_file_path"] = str(batch_item.batch_reduced_path)
            row["batch_timing_policy"] = "apportioned_legacy_first_row_total"
            row["batch_concat_seconds"] = batch_item.concat_seconds if index == 0 else 0.0
            row["batch_crop_seconds"] = batch_item.crop_seconds if index == 0 else 0.0
            row["batch_cfgrib_open_seconds"] = open_seconds if index == 0 else 0.0
            row["batch_row_build_seconds"] = row_build_seconds if index == 0 else 0.0
            row["timing_crop_seconds"] = round(batch_item.crop_seconds / lead_count, 6)
            row["timing_cfgrib_open_seconds"] = round(open_seconds / lead_count, 6)
            row["timing_row_build_seconds"] = round(row_build_seconds / lead_count, 6)

    def run_batch_reduce_stage(batch_item: NbmBatchPipelineItem, *, worker_id: str) -> NbmBatchPipelineItem:
        if reporter is not None:
            reporter.start_worker(
                worker_id,
                label=f"{batch_item.cycle_plan.cycle_token} batch",
                phase="reduce",
                group_id=batch_item.cycle_plan.cycle_token,
                details=f"batch_crop leads={len(batch_item.lead_items)}",
            )
            reporter.set_worker_attempt(worker_id, attempt=batch_item.attempt_count, max_attempts=max_attempts)
        batch_item.batch_raw_path.parent.mkdir(parents=True, exist_ok=True)
        batch_item.batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_item.batch_raw_path.unlink(missing_ok=True)
        batch_item.batch_reduced_path.unlink(missing_ok=True)
        raw_paths = [item.raw_path for item in batch_item.lead_items if item.raw_path is not None]
        if len(raw_paths) != len(batch_item.lead_items):
            raise ValueError("Batch reduce received a lead item without raw_path")
        batch_item.concat_seconds = concatenate_grib_files(raw_paths, batch_item.batch_raw_path)
        crop_started_at = time.perf_counter()
        batch_item.crop_result = coerce_crop_execution_result(
            crop_selected_grib2(
                args=args,
                raw_path=batch_item.batch_raw_path,
                reduced_path=batch_item.batch_reduced_path,
                left=args.left,
                right=args.right,
                bottom=args.bottom,
                top=args.top,
            ),
            args=args,
        )
        batch_item.crop_seconds = stage_elapsed_seconds(crop_started_at)
        for lead_item in batch_item.lead_items:
            row = lead_item.manifest_row
            row["subset_command"] = batch_item.crop_result.command
            row["crop_method_used"] = batch_item.crop_result.method_used
            row["crop_grid_cache_key"] = batch_item.crop_result.crop_grid_cache_key
            row["crop_grid_cache_hit"] = batch_item.crop_result.crop_grid_cache_hit
            row["crop_ij_box"] = batch_item.crop_result.crop_ij_box
            row["crop_wgrib2_threads"] = batch_item.crop_result.crop_wgrib2_threads
            row["reduced_file_size"] = batch_item.batch_reduced_path.stat().st_size if batch_item.batch_reduced_path.exists() else None
        apply_batch_timing_to_manifest_rows(batch_item)
        return batch_item

    def run_batch_extract_stage(batch_item: NbmBatchPipelineItem, *, worker_id: str) -> list[UnitResult]:
        if reporter is not None:
            reporter.start_worker(
                worker_id,
                label=f"{batch_item.cycle_plan.cycle_token} batch",
                phase="extract",
                group_id=batch_item.cycle_plan.cycle_token,
                details="batch_open_grouped_datasets",
            )
            reporter.set_worker_attempt(worker_id, attempt=batch_item.attempt_count, max_attempts=max_attempts)
        datasets: list[xr.Dataset] = []
        try:
            open_stats: dict[str, object] = {}
            cfgrib_started_at = time.perf_counter()
            datasets = open_grouped_datasets(
                batch_item.batch_reduced_path,
                index_dir=batch_item.batch_cfgrib_index_dir,
                selected=combine_selected_fields_for_batch(batch_item.lead_items),
                stats=open_stats,
            )
            open_seconds = stage_elapsed_seconds(cfgrib_started_at)
            row_build_started_at = time.perf_counter()
            if reporter is not None:
                reporter.update_worker(worker_id, phase="extract", details="batch_build_rows")
            row_stats: dict[str, object] = {}
            expanded_datasets = expanded_2d_datasets(datasets)
            wide_rows, long_rows, provenance_rows = build_rows_from_datasets(
                datasets=expanded_datasets,
                selected=combine_selected_fields_for_batch(batch_item.lead_items),
                cycle_plan=batch_item.cycle_plan,
                lead_hour=-1,
                crop_bounds=batch_item.current_crop_bounds,
                allow_unknown_long_inference=True,
                write_long=bool(getattr(args, "write_long", False)),
                write_provenance=not bool(getattr(args, "skip_provenance", False)),
                metric_profile=str(getattr(args, "metric_profile", "full")),
                stats=row_stats,
            )
            row_build_seconds = stage_elapsed_seconds(row_build_started_at)
        finally:
            for dataset in datasets:
                dataset.close()
            shutil.rmtree(batch_item.batch_cfgrib_index_dir, ignore_errors=True)

        wide_rows = filter_rows_to_selected_target_dates(wide_rows, batch_item.cycle_plan.selected_target_dates)
        long_rows = filter_rows_to_selected_target_dates(long_rows, batch_item.cycle_plan.selected_target_dates)
        provenance_rows = filter_rows_to_selected_target_dates(provenance_rows, batch_item.cycle_plan.selected_target_dates)
        wide_by_lead = rows_by_forecast_hour(wide_rows)
        long_by_lead = rows_by_forecast_hour(long_rows)
        provenance_by_lead = rows_by_forecast_hour(provenance_rows)
        validate_batch_row_forecast_hours(
            expected_leads={int(lead_item.lead_hour) for lead_item in batch_item.lead_items},
            wide_by_lead=wide_by_lead,
            long_by_lead=long_by_lead,
            provenance_by_lead=provenance_by_lead,
        )
        lead_count = max(1, len(batch_item.lead_items))
        apply_batch_timing_to_manifest_rows(batch_item, open_seconds=open_seconds, row_build_seconds=row_build_seconds)
        results: list[UnitResult] = []
        cleanup_started_at = time.perf_counter()
        if reporter is not None:
            reporter.update_worker(worker_id, phase="finalize", details="batch_cleanup")
        for lead_item in batch_item.lead_items:
            lead_wide_rows = wide_by_lead.get(int(lead_item.lead_hour), [])
            lead_long_rows = long_by_lead.get(int(lead_item.lead_hour), [])
            lead_provenance_rows = provenance_by_lead.get(int(lead_item.lead_hour), [])
            primary_wide_row = lead_wide_rows[0] if lead_wide_rows else None
            lead_item.manifest_row.update(open_stats)
            lead_item.manifest_row.update(
                {
                    "wide_row_count": len(lead_wide_rows),
                    "long_row_count": len(lead_long_rows),
                    "provenance_row_count": len(lead_provenance_rows),
                    "provenance_written": provenance_enabled(args),
                    "timing_row_geometry_seconds": round(float(row_stats.get("timing_row_geometry_seconds", 0.0)) / lead_count, 6),
                    "timing_row_metric_seconds": round(float(row_stats.get("timing_row_metric_seconds", 0.0)) / lead_count, 6),
                    "timing_row_provenance_seconds": round(float(row_stats.get("timing_row_provenance_seconds", 0.0)) / lead_count, 6),
                    "valid_time_utc": primary_wide_row["valid_time_utc"] if primary_wide_row else lead_item.manifest_row["valid_time_utc"],
                    "valid_time_local": primary_wide_row["valid_time_local"] if primary_wide_row else lead_item.manifest_row["valid_time_local"],
                    "valid_date_local": primary_wide_row["valid_date_local"] if primary_wide_row else lead_item.manifest_row["valid_date_local"],
                    "fallback_used_any": bool(primary_wide_row.get("fallback_used_any", False)) if primary_wide_row else False,
                }
            )
            if not lead_wide_rows:
                lead_item.manifest_row["extraction_status"] = "error:no_batch_rows_for_lead"
                lead_item.manifest_row["warnings"] = "; ".join(
                    filter(None, [lead_item.manifest_row.get("warnings"), "batch extract produced no rows for lead"])
                )
            results.append(
                UnitResult(
                    wide_row=primary_wide_row,
                    wide_rows=lead_wide_rows,
                    long_rows=lead_long_rows,
                    provenance_rows=lead_provenance_rows,
                    manifest_row=lead_item.manifest_row,
                )
            )

        if not getattr(args, "keep_downloads", False):
            for lead_item in batch_item.lead_items:
                if lead_item.raw_path is not None:
                    lead_item.raw_path.unlink(missing_ok=True)
                if lead_item.idx_path is not None:
                    lead_item.idx_path.unlink(missing_ok=True)
                lead_item.manifest_row["raw_deleted"] = True
                lead_item.manifest_row["idx_deleted"] = True
            batch_item.batch_raw_path.unlink(missing_ok=True)
        if not getattr(args, "keep_reduced", False):
            batch_item.batch_reduced_path.unlink(missing_ok=True)
            for lead_item in batch_item.lead_items:
                lead_item.manifest_row["reduced_deleted"] = True
                lead_item.manifest_row["reduced_retained"] = False
        cleanup_seconds = stage_elapsed_seconds(cleanup_started_at)
        for lead_item in batch_item.lead_items:
            lead_item.manifest_row["timing_cleanup_seconds"] = round(cleanup_seconds / lead_count, 6)
        return results

    def finalize_batch_failure(batch_item: NbmBatchPipelineItem, exc: Exception, *, phase: str) -> list[UnitResult]:
        results: list[UnitResult] = []
        for lead_item in batch_item.lead_items:
            lead_item.attempt_count = batch_item.attempt_count
            results.append(finalize_nbm_failure(lead_item, exc, phase=phase))
        if not getattr(args, "keep_downloads", False):
            batch_item.batch_raw_path.unlink(missing_ok=True)
        if not getattr(args, "keep_reduced", False):
            batch_item.batch_reduced_path.unlink(missing_ok=True)
        return results

    max_attempts = max(1, int(getattr(args, "max_task_attempts", 6)))

    admitted_terminals = 0

    def activate_cycle(cycle_plan: CyclePlan) -> None:
        nonlocal active_cycles, admitted_terminals
        lead_hours = lead_hours_for_cycle(cycle_plan)
        active_cycle_plans.append(cycle_plan)
        active_cycles += 1
        admitted_terminals += len(lead_hours)
        reporter.upsert_group(cycle_plan.cycle_token, status="submitted")
        for lead_hour in lead_hours:
            download_queue.put(make_initial_item(cycle_plan, lead_hour))

    def maybe_retry_stage(item: NbmPipelineItem, *, worker_id: str, stage_name: str, exc: Exception) -> bool:
        decision = classify_task_failure(exception_type=type(exc).__name__, message=str(exc), phase=stage_name)
        if not should_retry_attempt(
            attempt=item.attempt_count,
            policy=RetryPolicy(
                max_attempts=max_attempts,
                backoff_seconds=float(getattr(args, "retry_backoff_seconds", 2.0)),
                max_backoff_seconds=float(getattr(args, "retry_max_backoff_seconds", 30.0)),
            ),
            decision=decision,
        ):
            return False
        next_attempt = item.attempt_count + 1
        delay_seconds = compute_retry_delay_seconds(
            attempt=next_attempt,
            policy=RetryPolicy(
                max_attempts=max_attempts,
                backoff_seconds=float(getattr(args, "retry_backoff_seconds", 2.0)),
                max_backoff_seconds=float(getattr(args, "retry_max_backoff_seconds", 30.0)),
            ),
        )
        if reporter is not None:
            reporter.schedule_retry(
                worker_id,
                attempt=next_attempt,
                max_attempts=max_attempts,
                delay_seconds=delay_seconds,
                message=str(exc),
                error_class=decision.error_class,
            )
        _sleep_with_retry_refresh(delay_seconds, reporter)
        item.attempt_count = next_attempt
        return True

    def record_terminal_result(item: NbmPipelineItem, result: UnitResult) -> None:
        completion_queue.put((item.cycle_plan.cycle_token, item.lead_hour, result))

    def record_terminal_results(results: list[UnitResult]) -> None:
        for result in results:
            cycle_token = str(result.manifest_row["init_time_utc"])
            # Use the canonical cycle token from the result's init timestamp.
            init_ts = pd.Timestamp(result.manifest_row["init_time_utc"])
            completion_queue.put((init_ts.strftime("%Y%m%dT%H%MZ"), int(result.manifest_row["lead_hour"]), result))

    def disable_cycle_batching(cycle_token: str) -> list[NbmPipelineItem]:
        with batch_download_lock:
            batch_disabled_cycles.add(cycle_token)
            return batch_downloaded_by_cycle.pop(cycle_token, [])

    def enqueue_downloaded_item(item: NbmPipelineItem) -> None:
        if batch_reduce_mode != "cycle":
            reduce_queue.put(item)
            return
        cycle_token = item.cycle_plan.cycle_token
        batch_to_enqueue: NbmBatchPipelineItem | None = None
        direct_items: list[NbmPipelineItem] = []
        with batch_download_lock:
            if cycle_token in batch_disabled_cycles:
                direct_items.append(item)
            else:
                downloaded = batch_downloaded_by_cycle.setdefault(cycle_token, [])
                downloaded.append(item)
                expected = aggregations[cycle_token].expected_leads
                if len(downloaded) == expected:
                    batch_to_enqueue = make_batch_item(item.cycle_plan, downloaded)
                    batch_downloaded_by_cycle.pop(cycle_token, None)
        for direct_item in direct_items:
            reduce_queue.put(direct_item)
        if batch_to_enqueue is not None:
            reduce_queue.put(batch_to_enqueue)

    def disable_batch_and_flush_downloaded(cycle_token: str) -> None:
        for downloaded_item in disable_cycle_batching(cycle_token):
            reduce_queue.put(downloaded_item)

    def download_worker(index: int) -> None:
        worker_id = f"download_{index}"
        while True:
            item = download_queue.get()
            if item is None:
                download_queue.task_done()
                break
            update_phase_activity("download", 1)
            try:
                while True:
                    try:
                        stage_result = run_download_stage(item, worker_id=worker_id)
                        if reporter is not None and item.attempt_count > 1:
                            reporter.recover_worker(worker_id, message=f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d} download recovered")
                        if isinstance(stage_result, UnitResult):
                            if batch_reduce_mode == "cycle":
                                disable_batch_and_flush_downloaded(item.cycle_plan.cycle_token)
                            record_terminal_result(item, stage_result)
                        else:
                            enqueue_downloaded_item(stage_result)
                            update_pipeline_metrics(active_cycles=active_cycles, completed_cycles=completed_cycles, skipped_cycles=skipped_cycles, reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                        break
                    except Exception as exc:
                        if maybe_retry_stage(item, worker_id=worker_id, stage_name="download", exc=exc):
                            continue
                        if batch_reduce_mode == "cycle":
                            disable_batch_and_flush_downloaded(item.cycle_plan.cycle_token)
                        record_terminal_result(item, finalize_nbm_failure(item, exc, phase="download"))
                        break
            finally:
                update_phase_activity("download", -1)
                update_pipeline_metrics(active_cycles=active_cycles, completed_cycles=completed_cycles, skipped_cycles=skipped_cycles, reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                if reporter is not None:
                    reporter.retire_worker(worker_id)
                download_queue.task_done()

    def reduce_worker(index: int) -> None:
        worker_id = f"reduce_{index}"
        while True:
            item = reduce_queue.get()
            if item is None:
                reduce_queue.task_done()
                break
            update_phase_activity("reduce", 1)
            try:
                while True:
                    try:
                        next_item = (
                            run_batch_reduce_stage(item, worker_id=worker_id)
                            if isinstance(item, NbmBatchPipelineItem)
                            else run_reduce_stage(item, worker_id=worker_id)
                        )
                        if reporter is not None and item.attempt_count > 1:
                            label = f"{item.cycle_plan.cycle_token} batch" if isinstance(item, NbmBatchPipelineItem) else f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d}"
                            reporter.recover_worker(worker_id, message=f"{label} reduce recovered")
                        extract_queue.put(next_item)
                        update_pipeline_metrics(active_cycles=active_cycles, completed_cycles=completed_cycles, skipped_cycles=skipped_cycles, reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                        break
                    except Exception as exc:
                        if maybe_retry_stage(item, worker_id=worker_id, stage_name="reduce", exc=exc):
                            continue
                        if isinstance(item, NbmBatchPipelineItem):
                            record_terminal_results(finalize_batch_failure(item, exc, phase="reduce"))
                        else:
                            record_terminal_result(item, finalize_nbm_failure(item, exc, phase="reduce"))
                        break
            finally:
                update_phase_activity("reduce", -1)
                update_pipeline_metrics(active_cycles=active_cycles, completed_cycles=completed_cycles, skipped_cycles=skipped_cycles, reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                if reporter is not None:
                    reporter.retire_worker(worker_id)
                reduce_queue.task_done()

    def extract_worker(index: int) -> None:
        worker_id = f"extract_{index}"
        while True:
            item = extract_queue.get()
            if item is None:
                extract_queue.task_done()
                break
            update_phase_activity("extract", 1)
            try:
                while True:
                    try:
                        if isinstance(item, NbmBatchPipelineItem):
                            results = run_batch_extract_stage(item, worker_id=worker_id)
                            if reporter is not None and item.attempt_count > 1:
                                reporter.recover_worker(worker_id, message=f"{item.cycle_plan.cycle_token} batch extract recovered")
                            if item.attempt_count > 1:
                                for result in results:
                                    result.manifest_row["retry_recovered"] = True
                            record_terminal_results(results)
                        else:
                            result = run_extract_stage(item, worker_id=worker_id)
                            if reporter is not None and item.attempt_count > 1:
                                reporter.recover_worker(worker_id, message=f"{item.cycle_plan.cycle_token} f{item.lead_hour:03d} extract recovered")
                            if item.attempt_count > 1:
                                result.manifest_row["retry_recovered"] = True
                            record_terminal_result(item, result)
                        break
                    except Exception as exc:
                        if maybe_retry_stage(item, worker_id=worker_id, stage_name="extract", exc=exc):
                            continue
                        if isinstance(item, NbmBatchPipelineItem):
                            record_terminal_results(finalize_batch_failure(item, exc, phase="extract"))
                        else:
                            record_terminal_result(item, finalize_nbm_failure(item, exc, phase="extract"))
                        break
            finally:
                update_phase_activity("extract", -1)
                update_pipeline_metrics(active_cycles=active_cycles, completed_cycles=completed_cycles, skipped_cycles=skipped_cycles, reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                if reporter is not None:
                    reporter.retire_worker(worker_id)
                extract_queue.task_done()

    for cycle_plan in pending_cycle_plans[:cycle_worker_limit]:
        activate_cycle(cycle_plan)
    pending_cycle_index = min(len(pending_cycle_plans), cycle_worker_limit)
    update_pipeline_metrics(
        active_cycles=active_cycles,
        completed_cycles=completed_cycles,
        skipped_cycles=skipped_cycles,
        reduce_queue_obj=reduce_queue,
        extract_queue_obj=extract_queue,
    )

    download_threads = [threading.Thread(target=download_worker, args=(index,), daemon=True) for index in range(phase_limits.download_workers)]
    reduce_threads = [threading.Thread(target=reduce_worker, args=(index,), daemon=True) for index in range(phase_limits.reduce_workers)]
    extract_threads = [threading.Thread(target=extract_worker, args=(index,), daemon=True) for index in range(phase_limits.extract_workers)]
    worker_threads = download_threads + reduce_threads + extract_threads
    for thread in worker_threads:
        thread.start()

    completed_terminals = 0

    def pause_drain_complete() -> bool:
        active_download, active_reduce, active_extract = current_phase_activity()
        return (
            active_cycles == 0
            and completed_terminals >= admitted_terminals
            and completion_queue.empty()
            and download_queue.empty()
            and reduce_queue.empty()
            and extract_queue.empty()
            and active_download == 0
            and active_reduce == 0
            and active_extract == 0
        )

    try:
        while True:
            if run_control.pause_requested() and pause_drain_complete():
                run_control.mark_paused(reason="operator")
                reporter.mark_paused(reason="operator")
                break
            if completed_terminals >= admitted_terminals and pending_cycle_index >= len(pending_cycle_plans) and active_cycles == 0:
                break
            try:
                cycle_token, lead_hour, result = completion_queue.get(timeout=0.1)
            except queue.Empty:
                if reporter is not None and run_control.pause_requested():
                    reporter.refresh(force=True)
                continue
            completed_terminals += 1
            aggregation = aggregations[cycle_token]
            aggregation.results_by_lead[lead_hour] = result
            aggregation.completed_count += 1
            extraction_status = str(result.manifest_row.get("extraction_status") or "")
            if extraction_status == "ok":
                reporter.record_outcome("completed", message=f"{cycle_token} f{lead_hour:03d} ok")
            elif extraction_status == "no_matching_records":
                reporter.record_outcome("skipped", message=f"{cycle_token} f{lead_hour:03d} no_matching_records")
            else:
                aggregation.failed_count += 1
                reporter.record_outcome("failed", message=f"{cycle_token} f{lead_hour:03d} {extraction_status}")
            reporter.upsert_group(
                cycle_token,
                completed=aggregation.completed_count - aggregation.failed_count,
                failed=aggregation.failed_count,
                status=f"completed={aggregation.completed_count}/{aggregation.expected_leads}",
            )
            if aggregation.completed_count == aggregation.expected_leads and not aggregation.written:
                reporter.upsert_group(cycle_token, status="write_outputs")
                ordered_results = [
                    aggregation.results_by_lead[lead_hour_key]
                    for lead_hour_key in lead_hours_for_cycle(aggregation.cycle_plan)
                ]
                write_cycle_outputs(
                    args.output_dir,
                    aggregation.cycle_plan,
                    ordered_results,
                    write_long=bool(getattr(args, "write_long", False)),
                    write_provenance=not bool(getattr(args, "skip_provenance", False)),
                )
                aggregation.written = True
                active_cycles = max(0, active_cycles - 1)
                completed_cycles += 1
                reporter.upsert_group(
                    cycle_token,
                    completed=aggregation.expected_leads - aggregation.failed_count,
                    failed=aggregation.failed_count,
                    status="complete",
                )
                if reporter.mode == "log":
                    emit_progress_message(
                        f"completed_cycle init_time_utc={aggregation.cycle_plan.init_time_utc.isoformat()} "
                        f"mode=stage_pipeline ok_units={aggregation.expected_leads - aggregation.failed_count}/{aggregation.expected_leads}"
                    )
                if pending_cycle_index < len(pending_cycle_plans) and not run_control.pause_requested():
                    activate_cycle(pending_cycle_plans[pending_cycle_index])
                    pending_cycle_index += 1
            update_pipeline_metrics(
                active_cycles=active_cycles,
                completed_cycles=completed_cycles,
                skipped_cycles=skipped_cycles,
                reduce_queue_obj=reduce_queue,
                extract_queue_obj=extract_queue,
            )
    finally:
        for _ in download_threads:
            download_queue.put(None)
        for _ in reduce_threads:
            reduce_queue.put(None)
        for _ in extract_threads:
            extract_queue.put(None)
        download_queue.join()
        reduce_queue.join()
        extract_queue.join()
        for thread in worker_threads:
            thread.join()
        reporter.close(status="paused" if run_control.is_paused() else f"cycles={completed_cycles}")
    return 0


def main() -> int:
    return run_pipeline(parse_args())


if __name__ == "__main__":
    sys.exit(main())
