#!/usr/bin/env python3
"""Build monthly KLGA HRRR feature shards from hourly 2D CONUS files."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import statistics
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fetch_hrrr_records import (
    build_remote_paths,
    download_subset_for_inventory_patterns,
    manifest_inventory_lines,
)
from tools.weather.location_context import (
    CropBounds,
    REGIONAL_CROP_BOUNDS,
    SETTLEMENT_LOCATION,
    crop_context_metrics,
    crop_metadata,
    find_nearest_grid_cell,
    infer_north_is_first,
    infer_west_is_first,
    local_context_metrics,
    longitude_360_to_180,
    settlement_longitude_360,
    settlement_metadata,
)
from tools.weather.progress import ProgressBar, RunControl, create_progress_reporter
from tools.weather.retry import RetryPolicy, classify_task_failure, compute_retry_delay_seconds, should_retry_attempt


DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2026-04-12"
DEFAULT_DOWNLOAD_DIR = Path("data/runtime/downloads")
DEFAULT_REDUCED_DIR = Path("data/runtime/reduced")
DEFAULT_OUTPUT_DIR = Path("data/runtime/features/hourly_surface_klga")
DEFAULT_SUMMARY_OUTPUT_DIR = Path("data/runtime/features/overnight_summary_klga")
DEFAULT_MAX_WORKERS = 2
DEFAULT_RANGE_MERGE_GAP_BYTES = 64 * 1024
DEFAULT_CFGRIB_INDEX_STRATEGY = "temp_dir_per_task"
DEFAULT_SELECTION_MODE = "all"
SELECTION_MODES = (DEFAULT_SELECTION_MODE, "overnight_0005")
BATCH_REDUCE_MODES = ("off", "cycle")
CROP_METHODS = ("auto", "small_grib", "ijsmall_grib")
EXTRACT_METHODS = ("cfgrib", "eccodes")
SUMMARY_PROFILES = ("full", "overnight")
DEFAULT_CROP_METHOD = "small_grib"
DEFAULT_CROP_GRIB_TYPE = "same"
DEFAULT_EXTRACT_METHOD = "cfgrib"
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_SOURCE_MODEL = "HRRR"
DEFAULT_SOURCE_PRODUCT = "wrfsfcf"
DEFAULT_SOURCE_VERSION = "hrrr-conus-wrfsfcf-public"
DEFAULT_SLICE_POLICY = "overnight_local_v1"
OVERNIGHT_INIT_START_HOUR_LOCAL = 18
OVERNIGHT_INIT_END_HOUR_LOCAL = 0
OVERNIGHT_VALID_END_HOUR_LOCAL = 21
OVERNIGHT_CYCLE_COUNT = 4
OVERNIGHT_REVISION_CYCLE_COUNT = max(0, OVERNIGHT_CYCLE_COUNT - 1)
OVERNIGHT_VALIDATION_CYCLE_COUNT = 2
OVERNIGHT_VALIDATION_REVISION_CYCLE_COUNT = max(0, OVERNIGHT_VALIDATION_CYCLE_COUNT - 1)
OVERNIGHT_AVAILABILITY_CUTOFF_MINUTE_LOCAL = 5
FULL_LENGTH_CYCLE_HOURS_UTC = {0, 6, 12, 18}
WGRIB2_CANDIDATES = ["wgrib2", str(Path.home() / ".local/bin/wgrib2")]
GRID_SHAPE_PATTERNS = (
    re.compile(r"(?:grid:)?\((?P<nx>\d+)\s+x\s+(?P<ny>\d+)\)", re.IGNORECASE),
    re.compile(r"\bnx\s*[=:]\s*(?P<nx>\d+)\b.*?\bny\s*[=:]\s*(?P<ny>\d+)\b", re.IGNORECASE),
    re.compile(r"\b(?P<nx>\d+)\s+by\s+(?P<ny>\d+)\b", re.IGNORECASE),
)
SPATIAL_DIM_NAMES = {"latitude", "longitude", "lat", "lon", "x", "y"}
NY_TZ = ZoneInfo(DEFAULT_TIMEZONE)
INVENTORY_RE = re.compile(
    r"^\d+:\d+:d=(?P<init>\d{10}):(?P<short>[^:]+):(?P<level>[^:]+):(?P<step>[^:]+):?(?P<extra>.*)$"
)
INVENTORY_SELECTION_PATTERN_SPECS = (
    ("tmp_2m_k", r":TMP:2 m above ground:"),
    ("dpt_2m_k", r":DPT:2 m above ground:"),
    ("rh_2m_pct", r":RH:2 m above ground:"),
    ("ugrd_10m_ms", r":UGRD:10 m above ground:"),
    ("vgrd_10m_ms", r":VGRD:10 m above ground:"),
    ("gust_surface_ms", r":GUST:surface:"),
    ("surface_pressure_pa", r":PRES:surface:"),
    ("mslma_pa", r":MSLMA:mean sea level:"),
    ("visibility_m", r":VIS:surface:"),
    ("lcdc_low_pct", r":LCDC:low cloud layer:"),
    ("mcdc_mid_pct", r":MCDC:middle cloud layer:"),
    ("hcdc_high_pct", r":HCDC:high cloud layer:"),
    ("tcdc_entire_pct", r":TCDC:entire atmosphere:"),
    ("dswrf_surface_w_m2", r":DSWRF:surface:"),
    ("dlwrf_surface_w_m2", r":DLWRF:surface:"),
    ("apcp_surface_kg_m2", r":APCP:surface:"),
    ("prate_surface_kg_m2_s", r":PRATE:surface:"),
    ("hpbl_m", r":HPBL:surface:"),
    ("pwat_entire_atmosphere_kg_m2", r":PWAT:entire atmosphere \(considered as a single layer\):"),
    ("cape_surface_j_kg", r":CAPE:surface:"),
    ("cin_surface_j_kg", r":CIN:surface:"),
    ("refc_entire_atmosphere", r":REFC:entire atmosphere:"),
    ("ltng_entire_atmosphere", r":LTNG:entire atmosphere:"),
    ("dpt_upper_support", r":DPT:(1000 mb|925 mb|850 mb|700 mb):"),
    ("tmp_upper_support", r":TMP:(1000 mb|925 mb|850 mb|700 mb):"),
    ("ugrd_upper_support", r":UGRD:(1000 mb|925 mb|850 mb|700 mb):"),
    ("vgrd_upper_support", r":VGRD:(1000 mb|925 mb|850 mb|700 mb):"),
    ("hgt_upper_support", r":HGT:(1000 mb|925 mb|850 mb|700 mb):"),
    ("rh_upper_direct", r":RH:(1000 mb|925 mb|850 mb|700 mb):"),
    ("spfh_upper_direct", r":SPFH:(1000 mb|925 mb|850 mb|700 mb):"),
)
COMPILED_INVENTORY_SELECTION_PATTERNS = tuple((name, re.compile(pattern)) for name, pattern in INVENTORY_SELECTION_PATTERN_SPECS)
REVISION_FIELD_PREFIXES = (
    "tmp_2m_k",
    "tcdc_entire_pct",
    "dswrf_surface_w_m2",
    "pwat_entire_atmosphere_kg_m2",
    "hpbl_m",
    "mslma_pa",
)
REVISION_INVENTORY_SELECTION_PATTERN_SPECS = tuple(
    (name, pattern)
    for name, pattern in INVENTORY_SELECTION_PATTERN_SPECS
    if name in REVISION_FIELD_PREFIXES
)
COMPILED_REVISION_INVENTORY_SELECTION_PATTERNS = tuple(
    (name, re.compile(pattern)) for name, pattern in REVISION_INVENTORY_SELECTION_PATTERN_SPECS
)
CROP_GRID_CACHE_LOCKS: dict[Path, threading.Lock] = {}
CROP_GRID_CACHE_LOCKS_GUARD = threading.Lock()
CANONICAL_WIDE_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "fallback_used_any",
    "station_id",
    "forecast_hour",
    "init_time_utc",
    "init_time_local",
    "init_date_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "settlement_lat",
    "settlement_lon",
    "crop_top_lat",
    "crop_bottom_lat",
    "crop_left_lon",
    "crop_right_lon",
    "nearest_grid_lat",
    "nearest_grid_lon",
}
CANONICAL_PROVENANCE_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "init_time_utc",
    "init_time_local",
    "init_date_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "nearest_grid_lat",
    "nearest_grid_lon",
    "feature_name",
    "fallback_used",
    "fallback_source_description",
}


@dataclass(frozen=True)
class GroupSpec:
    name: str
    filter_by_keys: dict[str, object]
    vars_map: dict[str, str]
    inventory_checks: tuple[str, ...]


@dataclass(frozen=True)
class TaskSpec:
    target_date_local: str
    run_date_utc: str
    cycle_hour_utc: int
    forecast_hour: int
    init_time_utc: str
    init_time_local: str
    valid_time_utc: str
    valid_time_local: str
    init_date_local: str
    valid_date_local: str
    init_hour_local: int
    valid_hour_local: int
    cycle_rank_desc: int
    selected_for_summary: bool
    anchor_cycle_candidate: bool
    slice_policy: str = DEFAULT_SLICE_POLICY

    @property
    def key(self) -> str:
        return (
            f"{self.target_date_local}__"
            f"{self.run_date_utc}_t{self.cycle_hour_utc:02d}_f{self.forecast_hour:02d}"
        )


@dataclass
class TaskResult:
    ok: bool
    task_key: str
    row: dict[str, object] | None
    provenance_rows: list[dict[str, object]]
    missing_fields: list[str]
    message: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class HrrrPipelineItem:
    task: TaskSpec
    attempt_count: int = 1
    raw_path: Path | None = None
    raw_manifest_path: Path | None = None
    raw_selection_manifest_path: Path | None = None
    reduced_path: Path | None = None
    grib_url: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
    reduced_inventory: list[str] | None = None
    batch_reduced_path: Path | None = None
    batch_reduced_inventory: list[str] | None = None


@dataclass(frozen=True)
class PhaseConcurrencyLimits:
    download_workers: int
    reduce_workers: int
    extract_workers: int
    download_semaphore: threading.Semaphore
    reduce_semaphore: threading.Semaphore
    extract_semaphore: threading.Semaphore


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
    grid_shape: tuple[int, int]
    north_is_first: bool
    west_is_first: bool
    crop_bounds: CropBounds
    ij_box: CropIjBox


@dataclass(frozen=True)
class CropExecutionResult:
    selected_lines: list[str]
    matched_names: set[str]
    inventory_seconds: float
    reduce_seconds: float
    command: str
    method_used: str
    crop_grid_cache_key: str | None
    crop_grid_cache_hit: bool
    crop_ij_box: str | None
    crop_wgrib2_threads: int
    crop_fallback_reason: str | None = None


@dataclass(frozen=True)
class HrrrRuntimeOptions:
    crop_method: str = DEFAULT_CROP_METHOD
    crop_grib_type: str = DEFAULT_CROP_GRIB_TYPE
    wgrib2_threads: int | None = None
    max_workers: int = DEFAULT_MAX_WORKERS
    reduce_workers: int | None = None
    extract_method: str = DEFAULT_EXTRACT_METHOD
    summary_profile: str = "full"
    skip_provenance: bool = False


RUNTIME_OPTIONS = HrrrRuntimeOptions()


GROUP_SPECS = [
    GroupSpec(
        name="near_surface_2m",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        vars_map={"t2m": "tmp_2m_k", "d2m": "dpt_2m_k", "r2": "rh_2m_pct"},
        inventory_checks=(
            ":TMP:2 m above ground:",
            ":DPT:2 m above ground:",
            ":RH:2 m above ground:",
        ),
    ),
    GroupSpec(
        name="near_surface_10m",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
        vars_map={"u10": "ugrd_10m_ms", "v10": "vgrd_10m_ms"},
        inventory_checks=(
            ":UGRD:10 m above ground:",
            ":VGRD:10 m above ground:",
        ),
    ),
    GroupSpec(
        name="surface_instant",
        filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"},
        vars_map={
            "gust": "gust_surface_ms",
            "sp": "surface_pressure_pa",
            "vis": "visibility_m",
            "prate": "prate_surface_kg_m2_s",
            "cape": "cape_surface_j_kg",
            "cin": "cin_surface_j_kg",
            "sdswrf": "dswrf_surface_w_m2",
            "sdlwrf": "dlwrf_surface_w_m2",
            "blh": "hpbl_m",
        },
        inventory_checks=(
            ":GUST:surface:",
            ":PRES:surface:",
            ":VIS:surface:",
            ":PRATE:surface:",
            ":CAPE:surface:",
            ":CIN:surface:",
            ":DSWRF:surface:",
            ":DLWRF:surface:",
            ":HPBL:surface:",
        ),
    ),
    GroupSpec(
        name="surface_accum",
        filter_by_keys={"typeOfLevel": "surface", "stepType": "accum"},
        vars_map={"tp": "apcp_surface_kg_m2"},
        inventory_checks=(":APCP:surface:",),
    ),
    GroupSpec(
        name="mean_sea",
        filter_by_keys={"typeOfLevel": "meanSea", "stepType": "instant"},
        vars_map={"mslma": "mslma_pa"},
        inventory_checks=(":MSLMA:mean sea level:",),
    ),
    GroupSpec(
        name="low_cloud",
        filter_by_keys={"typeOfLevel": "lowCloudLayer", "stepType": "instant"},
        vars_map={"lcc": "lcdc_low_pct"},
        inventory_checks=(":LCDC:low cloud layer:",),
    ),
    GroupSpec(
        name="mid_cloud",
        filter_by_keys={"typeOfLevel": "middleCloudLayer", "stepType": "instant"},
        vars_map={"mcc": "mcdc_mid_pct"},
        inventory_checks=(":MCDC:middle cloud layer:",),
    ),
    GroupSpec(
        name="high_cloud",
        filter_by_keys={"typeOfLevel": "highCloudLayer", "stepType": "instant"},
        vars_map={"hcc": "hcdc_high_pct"},
        inventory_checks=(":HCDC:high cloud layer:",),
    ),
    GroupSpec(
        name="total_cloud",
        filter_by_keys={"typeOfLevel": "atmosphere", "stepType": "instant", "shortName": "tcc"},
        vars_map={"tcc": "tcdc_entire_pct"},
        inventory_checks=(":TCDC:entire atmosphere:",),
    ),
    GroupSpec(
        name="reflectivity",
        filter_by_keys={"typeOfLevel": "atmosphere", "stepType": "instant", "shortName": "refc"},
        vars_map={"refc": "refc_entire_atmosphere"},
        inventory_checks=(":REFC:entire atmosphere:",),
    ),
    GroupSpec(
        name="lightning",
        filter_by_keys={"typeOfLevel": "atmosphere", "stepType": "instant", "shortName": "ltng"},
        vars_map={"ltng": "ltng_entire_atmosphere"},
        inventory_checks=(":LTNG:entire atmosphere:",),
    ),
    GroupSpec(
        name="pwat",
        filter_by_keys={"typeOfLevel": "atmosphereSingleLayer", "stepType": "instant", "shortName": "pwat"},
        vars_map={"pwat": "pwat_entire_atmosphere_kg_m2"},
        inventory_checks=(":PWAT:entire atmosphere (considered as a single layer):",),
    ),
]

ISOBARIC_LEVELS = (1000, 925, 850, 700)
REQUESTED_FIELD_PREFIXES = [
    "tmp_2m_k",
    "dpt_2m_k",
    "rh_2m_pct",
    "ugrd_10m_ms",
    "vgrd_10m_ms",
    "gust_surface_ms",
    "surface_pressure_pa",
    "mslma_pa",
    "visibility_m",
    "lcdc_low_pct",
    "mcdc_mid_pct",
    "hcdc_high_pct",
    "tcdc_entire_pct",
    "dswrf_surface_w_m2",
    "dlwrf_surface_w_m2",
    "apcp_surface_kg_m2",
    "prate_surface_kg_m2_s",
    "hpbl_m",
    "pwat_entire_atmosphere_kg_m2",
    "cape_surface_j_kg",
    "cin_surface_j_kg",
    "refc_entire_atmosphere",
    "ltng_entire_atmosphere",
]
for level in ISOBARIC_LEVELS:
    REQUESTED_FIELD_PREFIXES.extend(
        [
            f"tmp_{level}mb_k",
            f"ugrd_{level}mb_ms",
            f"vgrd_{level}mb_ms",
            f"rh_{level}mb_pct",
            f"spfh_{level}mb_kg_kg",
            f"hgt_{level}mb_gpm",
        ]
    )

DIRECT_INVENTORY_PATTERNS: dict[str, re.Pattern[str]] = {
    "tmp_2m_k": re.compile(r":TMP:2 m above ground:"),
    "dpt_2m_k": re.compile(r":DPT:2 m above ground:"),
    "rh_2m_pct": re.compile(r":RH:2 m above ground:"),
    "ugrd_10m_ms": re.compile(r":UGRD:10 m above ground:"),
    "vgrd_10m_ms": re.compile(r":VGRD:10 m above ground:"),
    "gust_surface_ms": re.compile(r":GUST:surface:"),
    "surface_pressure_pa": re.compile(r":PRES:surface:"),
    "mslma_pa": re.compile(r":MSLMA:mean sea level:"),
    "visibility_m": re.compile(r":VIS:surface:"),
    "lcdc_low_pct": re.compile(r":LCDC:low cloud layer:"),
    "mcdc_mid_pct": re.compile(r":MCDC:middle cloud layer:"),
    "hcdc_high_pct": re.compile(r":HCDC:high cloud layer:"),
    "tcdc_entire_pct": re.compile(r":TCDC:entire atmosphere:"),
    "dswrf_surface_w_m2": re.compile(r":DSWRF:surface:"),
    "dlwrf_surface_w_m2": re.compile(r":DLWRF:surface:"),
    "apcp_surface_kg_m2": re.compile(r":APCP:surface:"),
    "prate_surface_kg_m2_s": re.compile(r":PRATE:surface:"),
    "hpbl_m": re.compile(r":HPBL:surface:"),
    "pwat_entire_atmosphere_kg_m2": re.compile(r":PWAT:entire atmosphere \(considered as a single layer\):"),
    "cape_surface_j_kg": re.compile(r":CAPE:surface:"),
    "cin_surface_j_kg": re.compile(r":CIN:surface:"),
    "refc_entire_atmosphere": re.compile(r":REFC:entire atmosphere:"),
    "ltng_entire_atmosphere": re.compile(r":LTNG:entire atmosphere:"),
}
for level in ISOBARIC_LEVELS:
    DIRECT_INVENTORY_PATTERNS[f"tmp_{level}mb_k"] = re.compile(rf":TMP:{level} mb:")
    DIRECT_INVENTORY_PATTERNS[f"ugrd_{level}mb_ms"] = re.compile(rf":UGRD:{level} mb:")
    DIRECT_INVENTORY_PATTERNS[f"vgrd_{level}mb_ms"] = re.compile(rf":VGRD:{level} mb:")
    DIRECT_INVENTORY_PATTERNS[f"rh_{level}mb_pct"] = re.compile(rf":RH:{level} mb:")
    DIRECT_INVENTORY_PATTERNS[f"spfh_{level}mb_kg_kg"] = re.compile(rf":SPFH:{level} mb:")
    DIRECT_INVENTORY_PATTERNS[f"hgt_{level}mb_gpm"] = re.compile(rf":HGT:{level} mb:")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR, help="Directory for raw full GRIB2 downloads.")
    parser.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR, help="Directory for reduced local GRIB2 subsets.")
    parser.add_argument("--scratch-dir", type=Path, help="Optional scratch root for raw/reduced GRIBs and cfgrib indexes.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for monthly parquet shards and manifests.")
    parser.add_argument(
        "--summary-output-dir",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_DIR,
        help="Directory for monthly overnight summary parquet shards.",
    )
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers for download/subset/extract tasks.")
    parser.add_argument(
        "--download-workers",
        type=int,
        help="Optional cap for concurrent download/subset phases. Defaults to max-workers.",
    )
    parser.add_argument(
        "--reduce-workers",
        type=int,
        help="Optional cap for concurrent reduce phases. Defaults to min(3, max-workers).",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        help="Optional cap for concurrent open/extract phases. Defaults to min(4, max-workers).",
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
        help="Maximum byte gap to merge between selected HRRR ranges before downloading.",
    )
    parser.add_argument(
        "--batch-reduce-mode",
        choices=BATCH_REDUCE_MODES,
        default="off",
        help=(
            "Batch crop/cfgrib work after selected-record downloads. off keeps the legacy per-task "
            "reduce/extract path; cycle concatenates all downloaded forecast-hour subsets for a retained "
            "HRRR cycle, crops once, then extracts each task from the multi-step reduced GRIB."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default=DEFAULT_SELECTION_MODE,
        help="Task-planning mode. Use overnight_0005 to keep only the anchor overnight cycle plus the latest revision cycle.",
    )
    parser.add_argument(
        "--crop-method",
        choices=CROP_METHODS,
        default=DEFAULT_CROP_METHOD,
        help="Crop primitive. Use auto to prefer cached -ijsmall_grib and fall back to -small_grib when needed.",
    )
    parser.add_argument(
        "--crop-grib-type",
        default=DEFAULT_CROP_GRIB_TYPE,
        help="Value passed to wgrib2 -set_grib_type before crop. Use same to preserve current encoding.",
    )
    parser.add_argument(
        "--wgrib2-threads",
        type=int,
        help="OMP thread count for wgrib2 crop operations. Defaults to 1 under parallel reduce, otherwise 2.",
    )
    parser.add_argument(
        "--extract-method",
        choices=EXTRACT_METHODS,
        default=DEFAULT_EXTRACT_METHOD,
        help="GRIB extraction backend. cfgrib is the reference path; eccodes is an opt-in direct message-iterator prototype.",
    )
    parser.add_argument(
        "--summary-profile",
        choices=SUMMARY_PROFILES,
        default="full",
        help="Summary extraction profile. full preserves current behavior; overnight records production intent for optimized overnight runs.",
    )
    parser.add_argument(
        "--skip-provenance",
        action="store_true",
        help="Do not construct or write provenance rows. Intended for production overnight backfills where provenance is unused.",
    )
    parser.add_argument("--keep-reduced", action="store_true", help="Keep reduced GRIB2 files after successful extraction.")
    parser.add_argument("--keep-downloads", action="store_true", help="Keep downloaded full GRIB2 files after successful extraction.")
    parser.add_argument(
        "--write-legacy-aliases",
        action="store_true",
        help="Include compatibility alias columns such as *_nearest, *_3x3_*, and *_7x7_* in the wide output.",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Exit 0 even if some tasks fail.")
    parser.add_argument("--limit-tasks", type=int, help="Optional cap for smoke tests.")
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
    parser.add_argument("--max-task-attempts", type=int, default=6, help="Maximum attempts per HRRR task including the first try.")
    parser.add_argument("--retry-backoff-seconds", type=float, default=2.0, help="Base backoff in seconds for HRRR task retries.")
    parser.add_argument("--retry-max-backoff-seconds", type=float, default=30.0, help="Maximum backoff in seconds for HRRR task retries.")
    return parser.parse_args()


def resolve_wgrib2() -> str:
    for candidate in WGRIB2_CANDIDATES:
        resolved = shutil.which(candidate) if candidate == "wgrib2" else candidate
        if resolved and Path(resolved).exists():
            return resolved
    raise RuntimeError("wgrib2 was not found in PATH or ~/.local/bin/wgrib2")


def _resolve_phase_cap(requested: int | None, *, default: int, outer_limit: int) -> int:
    candidate = default if requested is None else int(requested)
    return max(1, min(candidate, outer_limit))


def build_phase_concurrency_limits(*, max_workers: int, args: argparse.Namespace) -> PhaseConcurrencyLimits:
    outer_limit = max(1, int(max_workers))
    download_workers = _resolve_phase_cap(
        getattr(args, "download_workers", None),
        default=outer_limit,
        outer_limit=outer_limit,
    )
    reduce_workers = _resolve_phase_cap(
        getattr(args, "reduce_workers", None),
        default=min(3, outer_limit),
        outer_limit=outer_limit,
    )
    extract_workers = _resolve_phase_cap(
        getattr(args, "extract_workers", None),
        default=min(4, outer_limit),
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


def ensure_tooling() -> str:
    missing: list[str] = []
    wgrib2_path = None
    try:
        wgrib2_path = resolve_wgrib2()
    except RuntimeError:
        missing.append("wgrib2")

    try:
        import cfgrib  # noqa: F401
    except Exception:
        missing.append("cfgrib")

    try:
        import eccodes  # noqa: F401
    except Exception:
        missing.append("eccodes")

    try:
        import pyarrow  # noqa: F401
    except Exception:
        missing.append("pyarrow")

    if missing:
        raise RuntimeError(f"Missing required tooling: {', '.join(missing)}")
    assert wgrib2_path is not None
    return wgrib2_path


def parse_date(value: str) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def iter_dates(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start_date
    while current <= end_date:
        yield current
        current += pd.Timedelta(days=1)


def isoformat_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat()


def isoformat_local(value: dt.datetime) -> str:
    return value.astimezone(NY_TZ).isoformat()


def max_forecast_hour_for_cycle(cycle_hour_utc: int) -> int:
    return 48 if cycle_hour_utc in FULL_LENGTH_CYCLE_HOURS_UTC else 18


def target_day_expected_local_times(target_date_local: pd.Timestamp) -> list[dt.datetime]:
    start_local = dt.datetime.combine(target_date_local.date(), dt.time(0, 0), tzinfo=NY_TZ)
    end_local = dt.datetime.combine(target_date_local.date(), dt.time(OVERNIGHT_VALID_END_HOUR_LOCAL, 0), tzinfo=NY_TZ)
    expected: list[dt.datetime] = []
    current_utc = start_local.astimezone(dt.timezone.utc)
    end_utc = end_local.astimezone(dt.timezone.utc)
    while current_utc <= end_utc:
        local = current_utc.astimezone(NY_TZ)
        if local.date() == target_date_local.date() and local.hour <= OVERNIGHT_VALID_END_HOUR_LOCAL:
            expected.append(local)
        current_utc += dt.timedelta(hours=1)
    return expected


def target_day_expected_hours(target_date_local: pd.Timestamp) -> set[int]:
    return {local.hour for local in target_day_expected_local_times(target_date_local)}


def target_day_expected_slots(target_date_local: pd.Timestamp) -> set[str]:
    return {local.isoformat() for local in target_day_expected_local_times(target_date_local)}


def choose_anchor_cycle(
    cycle_slots: dict[tuple[str, int], set[str]],
    cycle_rank: dict[tuple[str, int], int],
    expected_slots: set[str],
) -> tuple[tuple[str, int], bool]:
    full_coverage = [key for key, slots in cycle_slots.items() if expected_slots <= slots]
    if full_coverage:
        full_coverage.sort(key=lambda item: cycle_rank[item])
        return full_coverage[0], True
    fallback = max(
        cycle_slots,
        key=lambda item: (len(cycle_slots[item] & expected_slots), -cycle_rank[item]),
    )
    return fallback, False


def select_anchor_cycle(tasks: list[TaskSpec], expected_slots: set[str]) -> tuple[tuple[str, int], bool, dict[tuple[str, int], set[str]]]:
    cycle_slots: dict[tuple[str, int], set[str]] = {}
    cycle_rank: dict[tuple[str, int], int] = {}
    for task in tasks:
        key = (task.run_date_utc, task.cycle_hour_utc)
        cycle_slots.setdefault(key, set()).add(task.valid_time_local)
        cycle_rank[key] = task.cycle_rank_desc
    anchor_key, has_full_coverage = choose_anchor_cycle(cycle_slots, cycle_rank, expected_slots)
    return anchor_key, has_full_coverage, cycle_slots


def build_tasks_for_target_date(target_date_local: pd.Timestamp, selection_mode: str = DEFAULT_SELECTION_MODE) -> list[TaskSpec]:
    target_date = target_date_local.date()
    window_start = dt.datetime.combine(
        target_date - dt.timedelta(days=1),
        dt.time(OVERNIGHT_INIT_START_HOUR_LOCAL, 0),
        tzinfo=NY_TZ,
    )
    window_end = dt.datetime.combine(
        target_date,
        dt.time(OVERNIGHT_INIT_END_HOUR_LOCAL, 0),
        tzinfo=NY_TZ,
    )
    availability_cutoff = dt.datetime.combine(
        target_date,
        dt.time(0, OVERNIGHT_AVAILABILITY_CUTOFF_MINUTE_LOCAL),
        tzinfo=NY_TZ,
    )

    available_cycles: list[dt.datetime] = []
    current = window_end
    while current >= window_start:
        if current <= availability_cutoff:
            available_cycles.append(current)
        current -= dt.timedelta(hours=1)

    target_start_local = dt.datetime.combine(target_date, dt.time(0, 0), tzinfo=NY_TZ)
    target_end_local = dt.datetime.combine(target_date, dt.time(OVERNIGHT_VALID_END_HOUR_LOCAL, 0), tzinfo=NY_TZ)
    expected_slots = target_day_expected_slots(target_date_local)

    cycle_task_map: dict[tuple[str, int], list[TaskSpec]] = {}
    all_cycle_rank: dict[tuple[str, int], int] = {}
    for cycle_rank_desc, init_local in enumerate(available_cycles):
        init_utc = init_local.astimezone(dt.timezone.utc)
        run_date_utc = init_utc.date().isoformat()
        cycle_hour_utc = init_utc.hour
        cycle_key = (run_date_utc, cycle_hour_utc)
        all_cycle_rank[cycle_key] = cycle_rank_desc
        max_forecast_hour = max_forecast_hour_for_cycle(cycle_hour_utc)
        for forecast_hour in range(0, max_forecast_hour + 1):
            valid_utc = init_utc + dt.timedelta(hours=forecast_hour)
            valid_local = valid_utc.astimezone(NY_TZ)
            if valid_local.date() != target_date:
                continue
            if valid_local < target_start_local or valid_local > target_end_local:
                continue
            cycle_task_map.setdefault(cycle_key, []).append(
                TaskSpec(
                    target_date_local=target_date.isoformat(),
                    run_date_utc=run_date_utc,
                    cycle_hour_utc=cycle_hour_utc,
                    forecast_hour=forecast_hour,
                    init_time_utc=isoformat_utc(init_utc),
                    init_time_local=isoformat_local(init_local),
                    valid_time_utc=isoformat_utc(valid_utc),
                    valid_time_local=isoformat_local(valid_local),
                    init_date_local=init_local.date().isoformat(),
                    valid_date_local=valid_local.date().isoformat(),
                    init_hour_local=init_local.hour,
                    valid_hour_local=valid_local.hour,
                    cycle_rank_desc=cycle_rank_desc,
                    selected_for_summary=True,
                    anchor_cycle_candidate=False,
                )
            )

    if not cycle_task_map:
        return []

    cycle_slots = {
        cycle_key: {task.valid_time_local for task in cycle_tasks}
        for cycle_key, cycle_tasks in cycle_task_map.items()
    }
    anchor_cycle, _ = choose_anchor_cycle(cycle_slots, all_cycle_rank, expected_slots)

    retained_cycle_keys: list[tuple[str, int]] = [anchor_cycle]
    if selection_mode == "overnight_0005":
        revision_cycle_count = OVERNIGHT_VALIDATION_REVISION_CYCLE_COUNT
        max_cycle_count = OVERNIGHT_VALIDATION_CYCLE_COUNT
    else:
        revision_cycle_count = OVERNIGHT_REVISION_CYCLE_COUNT
        max_cycle_count = OVERNIGHT_CYCLE_COUNT
    latest_revision_cycle_keys = sorted(
        (cycle_key for cycle_key in cycle_task_map if cycle_key != anchor_cycle),
        key=lambda cycle_key: all_cycle_rank[cycle_key],
    )[:revision_cycle_count]
    retained_cycle_keys.extend(latest_revision_cycle_keys)
    retained_cycle_keys = sorted(
        set(retained_cycle_keys),
        key=lambda cycle_key: all_cycle_rank[cycle_key],
    )[:max_cycle_count]

    retained_tasks: list[TaskSpec] = []
    retained_cycle_rank = {cycle_key: rank for rank, cycle_key in enumerate(retained_cycle_keys)}
    for cycle_key in retained_cycle_keys:
        for task in cycle_task_map[cycle_key]:
            retained_tasks.append(
                TaskSpec(
                    **{
                        **task.__dict__,
                        "cycle_rank_desc": retained_cycle_rank[cycle_key],
                        "anchor_cycle_candidate": cycle_key == anchor_cycle,
                    }
                )
            )

    retained_tasks.sort(key=lambda item: (item.cycle_rank_desc, item.forecast_hour))
    return retained_tasks


def build_all_tasks(start_date: pd.Timestamp, end_date: pd.Timestamp, selection_mode: str = DEFAULT_SELECTION_MODE) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for date in iter_dates(start_date, end_date):
        tasks.extend(build_tasks_for_target_date(date, selection_mode=selection_mode))
    return tasks


def month_id_for_task(task: TaskSpec) -> str:
    return task.target_date_local[:7]


def crop_spec() -> tuple[str, str]:
    return (
        f"{REGIONAL_CROP_BOUNDS.left}:{REGIONAL_CROP_BOUNDS.right}",
        f"{REGIONAL_CROP_BOUNDS.bottom}:{REGIONAL_CROP_BOUNDS.top}",
    )


def inventory_selection_patterns() -> list[tuple[str, str]]:
    return list(INVENTORY_SELECTION_PATTERN_SPECS)


def task_field_profile(task: TaskSpec | None) -> str:
    if (
        task is not None
        and RUNTIME_OPTIONS.summary_profile == "overnight"
        and not bool(task.anchor_cycle_candidate)
    ):
        return "revision"
    return "full"


def inventory_selection_patterns_for_task(task: TaskSpec | None) -> list[tuple[str, str]]:
    if task_field_profile(task) == "revision":
        return list(REVISION_INVENTORY_SELECTION_PATTERN_SPECS)
    return inventory_selection_patterns()


def requested_field_prefixes_for_task(task: TaskSpec | None) -> list[str]:
    if task_field_profile(task) == "revision":
        return list(REVISION_FIELD_PREFIXES)
    return list(REQUESTED_FIELD_PREFIXES)


def compiled_inventory_selection_patterns_for_task(
    task: TaskSpec | None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if task_field_profile(task) == "revision":
        return COMPILED_REVISION_INVENTORY_SELECTION_PATTERNS
    return COMPILED_INVENTORY_SELECTION_PATTERNS


def select_inventory_lines(inventory_lines: list[str], *, task: TaskSpec | None = None) -> tuple[list[str], set[str]]:
    selected: list[str] = []
    matched_names: set[str] = set()
    compiled_patterns = compiled_inventory_selection_patterns_for_task(task)
    for line in inventory_lines:
        for name, pattern in compiled_patterns:
            if pattern.search(line):
                selected.append(line)
                matched_names.add(name)
                break
    return selected, matched_names


def run_command(
    args: list[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, input=input_text, text=True, capture_output=True, env=env)


def inventory_for_grib(wgrib2_path: str, grib_path: Path) -> list[str]:
    result = run_command([wgrib2_path, str(grib_path), "-s"])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"wgrib2 inventory failed for {grib_path}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def crop_grid_cache_root(reduced_path: Path) -> Path:
    return reduced_path.parent / ".crop_grid"


def first_cached_crop_grid_entry(reduced_path: Path) -> CropGridCacheEntry | None:
    cache_root = crop_grid_cache_root(reduced_path)
    if not cache_root.exists():
        return None
    for cache_path in sorted(cache_root.glob("*.json")):
        if cache_path.name.startswith("raw_identity_") or cache_path.name.endswith(".unsupported.json"):
            continue
        cached = load_crop_grid_cache_entry(cache_path)
        if cached is not None:
            return cached
    return None


def crop_grid_cache_lock(cache_path: Path) -> threading.Lock:
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
        "lat_sample": [float(lat_flat[0]), float(lat_flat[mid_index]), float(lat_flat[-1])],
        "lon_sample": [float(lon_flat[0]), float(lon_flat[mid_index]), float(lon_flat[-1])],
        "lat_min": float(np.nanmin(lat_grid)),
        "lat_max": float(np.nanmax(lat_grid)),
        "lon_min": float(np.nanmin(lon_grid)),
        "lon_max": float(np.nanmax(lon_grid)),
    }


def build_crop_grid_cache_key(
    *,
    bounds: CropBounds,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    north_is_first: bool,
    west_is_first: bool,
) -> str:
    payload = {
        "source_model": DEFAULT_SOURCE_MODEL,
        "source_product": DEFAULT_SOURCE_PRODUCT,
        "source_version": DEFAULT_SOURCE_VERSION,
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


def serialize_crop_grid_cache_entry(entry: CropGridCacheEntry) -> str:
    payload = {
        "signature": entry.signature,
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
        "created_by": DEFAULT_SOURCE_VERSION,
    }
    return json.dumps(payload, sort_keys=True)


def parse_crop_grid_cache_entry(text: str) -> CropGridCacheEntry:
    payload = json.loads(text)
    ij_box = payload["ij_box"]
    grid_shape = payload["grid_shape"]
    crop_bounds = payload["crop_bounds"]
    return CropGridCacheEntry(
        signature=str(payload["signature"]),
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


def load_crop_grid_cache_entry(cache_path: Path) -> CropGridCacheEntry | None:
    if not cache_path.exists():
        return None
    try:
        return parse_crop_grid_cache_entry(cache_path.read_text())
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def write_crop_grid_cache_entry(cache_path: Path, entry: CropGridCacheEntry) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(serialize_crop_grid_cache_entry(entry))
    tmp_path.replace(cache_path)


def load_crop_grid_negative_cache(cache_path: Path) -> str | None:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return "unreadable_negative_cache"
    reason = payload.get("reason")
    return str(reason) if reason else "ijsmall_grib_disabled"


def write_crop_grid_negative_cache(cache_path: Path, *, reason: str) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
    tmp_path.write_text(json.dumps({"reason": str(reason), "created_by": DEFAULT_SOURCE_VERSION}, sort_keys=True))
    tmp_path.replace(cache_path)


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
    i0, i1 = (col_min + 1, col_max + 1) if west_is_first else (ncols - col_max, ncols - col_min)
    j0, j1 = (nrows - row_max, nrows - row_min) if north_is_first else (row_min + 1, row_max + 1)
    return CropIjBox(i0=int(i0), i1=int(i1), j0=int(j0), j1=int(j1))


def representative_grid_context_for_path(
    grib_path: Path,
    grib_inventory: list[str],
    *,
    cfgrib_index_dir: Path,
) -> tuple[np.ndarray, np.ndarray, bool, bool]:
    group_datasets = open_reduced_grib_group_datasets(
        grib_path,
        grib_inventory,
        cfgrib_index_dir=cfgrib_index_dir,
    )
    try:
        if not group_datasets:
            raise ValueError(f"No datasets available to infer crop grid for {grib_path}")
        for _, dataset in group_datasets:
            if not dataset.data_vars:
                continue
            first_data_array = next(iter(dataset.data_vars.values()))
            lat_grid, lon_grid = extract_lat_lon(first_data_array)
            return lat_grid, lon_grid, infer_north_is_first(lat_grid), infer_west_is_first(lon_grid)
        raise ValueError(f"No data variables available to infer crop grid for {grib_path}")
    finally:
        close_group_datasets(group_datasets)


def resolve_crop_grid_cache_entry(
    *,
    full_path: Path,
    reduced_path: Path,
    selected_inventory_lines: list[str],
    bounds: CropBounds,
) -> tuple[CropGridCacheEntry, bool]:
    cache_root = crop_grid_cache_root(reduced_path)
    cache_root.mkdir(parents=True, exist_ok=True)
    probe_index_dir = cache_root / ".cfgrib_index" / f"probe_{reduced_path.stem}"
    probe_index_dir.mkdir(parents=True, exist_ok=True)
    lat_grid, lon_grid, north_is_first, west_is_first = representative_grid_context_for_path(
        full_path,
        selected_inventory_lines,
        cfgrib_index_dir=probe_index_dir,
    )
    cache_key = build_crop_grid_cache_key(
        bounds=bounds,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        north_is_first=north_is_first,
        west_is_first=west_is_first,
    )
    cache_path = cache_root / f"{cache_key}.json"
    with crop_grid_cache_lock(cache_path):
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
            grid_shape=(int(lat_grid.shape[0]), int(lat_grid.shape[1])),
            north_is_first=bool(north_is_first),
            west_is_first=bool(west_is_first),
            crop_bounds=bounds,
            ij_box=ij_box,
        )
        write_crop_grid_cache_entry(cache_path, entry)
        return entry, False


def active_runtime_options(options: HrrrRuntimeOptions | None = None) -> HrrrRuntimeOptions:
    return RUNTIME_OPTIONS if options is None else options


def crop_wgrib2_thread_count(options: HrrrRuntimeOptions | None = None) -> int:
    options = active_runtime_options(options)
    if options.wgrib2_threads is not None:
        return max(1, int(options.wgrib2_threads))
    outer_parallelism = max(1, int(options.max_workers))
    reduce_parallelism = max(1, int(options.reduce_workers or 1))
    return 1 if (outer_parallelism > 1 or reduce_parallelism > 1) else 2


def crop_wgrib2_env(options: HrrrRuntimeOptions | None = None) -> dict[str, str]:
    options = active_runtime_options(options)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(crop_wgrib2_thread_count(options))
    env["OMP_WAIT_POLICY"] = "PASSIVE"
    return env


def read_wgrib2_grid_shape(wgrib2_path: str, path: Path, *, env: dict[str, str] | None = None) -> tuple[int, int]:
    result = run_command([wgrib2_path, str(path), "-grid"], env=env)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"wgrib2 -grid failed for {path}")
    normalized_output = result.stdout.replace("\n", " ")
    for pattern in GRID_SHAPE_PATTERNS:
        match = pattern.search(normalized_output)
        if match is not None:
            return int(match.group("nx")), int(match.group("ny"))
    excerpt = normalized_output[:500].strip()
    raise ValueError(f"Unable to parse wgrib2 -grid output for {path}: {excerpt}")


def reduce_grib2(
    wgrib2_path: str,
    full_path: Path,
    reduced_path: Path,
    *,
    selected_inventory_lines: list[str] | None = None,
    task: TaskSpec | None = None,
    options: HrrrRuntimeOptions | None = None,
) -> CropExecutionResult:
    options = active_runtime_options(options)
    if selected_inventory_lines is None:
        inventory_started_at = time.perf_counter()
        inventory_lines = inventory_for_grib(wgrib2_path, full_path)
        inventory_seconds = time.perf_counter() - inventory_started_at
        selected_lines, matched_names = select_inventory_lines(inventory_lines, task=task)
    else:
        inventory_seconds = 0.0
        selected_lines = [line for line in selected_inventory_lines if line.strip()]
        _, matched_names = select_inventory_lines(selected_lines, task=task)
    if not selected_lines:
        raise RuntimeError(f"No requested records found in {full_path.name}")

    reduced_path.parent.mkdir(parents=True, exist_ok=True)
    lon_spec, lat_spec = crop_spec()
    crop_env = crop_wgrib2_env(options)
    wgrib2_threads = crop_wgrib2_thread_count(options)

    def _run_crop(command: list[str], *, method: str, **extra: object) -> CropExecutionResult:
        reduce_started_at = time.perf_counter()
        result = run_command(command, input_text="\n".join(selected_lines) + "\n", env=crop_env)
        reduce_seconds = time.perf_counter() - reduce_started_at
        if result.returncode != 0 or not reduced_path.exists():
            raise RuntimeError(result.stderr.strip() or f"wgrib2 subsetting failed for {full_path.name}")
        if reduced_path.stat().st_size <= 0:
            raise RuntimeError(f"Reduced GRIB2 is empty for {full_path.name}")
        return CropExecutionResult(
            selected_lines=selected_lines,
            matched_names=matched_names,
            inventory_seconds=inventory_seconds,
            reduce_seconds=reduce_seconds,
            command=" ".join(command),
            method_used=method,
            crop_grid_cache_key=extra.get("crop_grid_cache_key") if isinstance(extra.get("crop_grid_cache_key"), str) else None,
            crop_grid_cache_hit=bool(extra.get("crop_grid_cache_hit")),
            crop_ij_box=extra.get("crop_ij_box") if isinstance(extra.get("crop_ij_box"), str) else None,
            crop_wgrib2_threads=wgrib2_threads,
            crop_fallback_reason=extra.get("crop_fallback_reason") if isinstance(extra.get("crop_fallback_reason"), str) else None,
        )

    def _small_grib(*, fallback_reason: str | None = None) -> CropExecutionResult:
        command = [
            wgrib2_path,
            "-i",
            str(full_path),
            "-set_grib_type",
            str(options.crop_grib_type),
            "-small_grib",
            lon_spec,
            lat_spec,
            str(reduced_path),
        ]
        return _run_crop(command, method="small_grib", crop_fallback_reason=fallback_reason)

    if options.crop_method == "small_grib":
        return _small_grib()

    cached_entry = first_cached_crop_grid_entry(reduced_path)
    if cached_entry is not None:
        negative_cache_path = crop_grid_cache_root(reduced_path) / f"{cached_entry.signature}.unsupported.json"
        negative_cache_reason = load_crop_grid_negative_cache(negative_cache_path)
        if options.crop_method == "ijsmall_grib" or negative_cache_reason is None:
            try:
                command = [
                    wgrib2_path,
                    "-i",
                    str(full_path),
                    "-set_grib_type",
                    str(options.crop_grib_type),
                    "-ijsmall_grib",
                    cached_entry.ij_box.format_i(),
                    cached_entry.ij_box.format_j(),
                    str(reduced_path),
                ]
                result = _run_crop(
                    command,
                    method="ijsmall_grib",
                    crop_grid_cache_key=cached_entry.signature,
                    crop_grid_cache_hit=True,
                    crop_ij_box=cached_entry.ij_box.as_text(),
                )
                actual_grid_shape = read_wgrib2_grid_shape(wgrib2_path, reduced_path, env=crop_env)
                expected_grid_shape = (cached_entry.ij_box.nx, cached_entry.ij_box.ny)
                if actual_grid_shape != expected_grid_shape:
                    raise ValueError(f"ijsmall_grib crop produced grid {actual_grid_shape} but expected {expected_grid_shape}")
                return result
            except Exception as exc:
                if options.crop_method == "ijsmall_grib":
                    raise
                write_crop_grid_negative_cache(negative_cache_path, reason="ijsmall_grib_crop_or_validation_failed")
                reduced_path.unlink(missing_ok=True)
                return _small_grib(fallback_reason=str(exc)[:300])

    try:
        cache_entry, cache_hit = resolve_crop_grid_cache_entry(
            full_path=full_path,
            reduced_path=reduced_path,
            selected_inventory_lines=selected_lines,
            bounds=REGIONAL_CROP_BOUNDS,
        )
        negative_cache_path = crop_grid_cache_root(reduced_path) / f"{cache_entry.signature}.unsupported.json"
        negative_cache_reason = load_crop_grid_negative_cache(negative_cache_path)
        if options.crop_method == "auto" and negative_cache_reason is not None:
            return _small_grib(fallback_reason=negative_cache_reason)
        command = [
            wgrib2_path,
            "-i",
            str(full_path),
            "-set_grib_type",
            str(options.crop_grib_type),
            "-ijsmall_grib",
            cache_entry.ij_box.format_i(),
            cache_entry.ij_box.format_j(),
            str(reduced_path),
        ]
        result = _run_crop(
            command,
            method="ijsmall_grib",
            crop_grid_cache_key=cache_entry.signature,
            crop_grid_cache_hit=cache_hit,
            crop_ij_box=cache_entry.ij_box.as_text(),
        )
        actual_grid_shape = read_wgrib2_grid_shape(wgrib2_path, reduced_path, env=crop_env)
        expected_grid_shape = (cache_entry.ij_box.nx, cache_entry.ij_box.ny)
        if actual_grid_shape != expected_grid_shape:
            raise ValueError(f"ijsmall_grib crop produced grid {actual_grid_shape} but expected {expected_grid_shape}")
        return result
    except Exception as exc:
        if options.crop_method == "ijsmall_grib":
            raise
        try:
            if "cache_entry" in locals():
                negative_cache_path = crop_grid_cache_root(reduced_path) / f"{cache_entry.signature}.unsupported.json"
                write_crop_grid_negative_cache(negative_cache_path, reason="ijsmall_grib_crop_or_validation_failed")
        except Exception:
            pass
        reduced_path.unlink(missing_ok=True)
        return _small_grib(fallback_reason=str(exc)[:300])


def path_for_raw(download_dir: Path, task: TaskSpec) -> Path:
    return download_dir / f"hrrr.{task.run_date_utc.replace('-', '')}.t{task.cycle_hour_utc:02d}z.wrfsfcf{task.forecast_hour:02d}.grib2"


def path_for_raw_manifest(download_dir: Path, task: TaskSpec) -> Path:
    return download_dir / f"hrrr.{task.run_date_utc.replace('-', '')}.t{task.cycle_hour_utc:02d}z.wrfsfcf{task.forecast_hour:02d}.manifest.csv"


def path_for_raw_selection_manifest(download_dir: Path, task: TaskSpec) -> Path:
    return download_dir / f"hrrr.{task.run_date_utc.replace('-', '')}.t{task.cycle_hour_utc:02d}z.wrfsfcf{task.forecast_hour:02d}.selection.csv"


def path_for_reduced(reduced_dir: Path, task: TaskSpec) -> Path:
    return reduced_dir / f"hrrr.{task.run_date_utc.replace('-', '')}.t{task.cycle_hour_utc:02d}z.wrfsfcf{task.forecast_hour:02d}.reduced.grib2"


def open_group_dataset(reduced_path: Path, group: GroupSpec, *, indexpath: str) -> xr.Dataset:
    return xr.open_dataset(
        reduced_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": group.filter_by_keys, "indexpath": indexpath},
    )


def open_group_dataset_with_optional_index(
    reduced_path: Path,
    group: GroupSpec,
    *,
    indexpath: str,
) -> xr.Dataset:
    try:
        return open_group_dataset(reduced_path, group, indexpath=indexpath)
    except TypeError as exc:
        # Test doubles and older call sites may still expose the legacy two-argument signature.
        if "indexpath" not in str(exc):
            raise
        return open_group_dataset(reduced_path, group)


def timestamp_to_utc_string(value: object) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def timestamp_to_local_strings(value: object, timezone_name: str) -> tuple[str, str]:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(timezone_name)
    return local.isoformat(), local.date().isoformat()


def spatial_dims(data_array: xr.DataArray) -> tuple[str, ...]:
    return tuple(dim for dim in data_array.dims if dim.lower() in SPATIAL_DIM_NAMES)


def _target_step_timedelta(task: TaskSpec) -> pd.Timedelta:
    return pd.Timedelta(hours=int(task.forecast_hour))


def _select_task_coordinate(data_array: xr.DataArray, task: TaskSpec) -> tuple[xr.DataArray, bool]:
    target_step = _target_step_timedelta(task)
    for coord_name in ("step",):
        if coord_name in data_array.coords:
            coord = data_array.coords[coord_name]
            dims = tuple(dim for dim in coord.dims if dim in data_array.dims)
            if len(dims) == 1:
                values = pd.to_timedelta(np.asarray(coord.values).ravel())
                matches = np.flatnonzero(values == target_step)
                if len(matches) == 1:
                    return data_array.isel({dims[0]: int(matches[0])}), True
    if "valid_time" in data_array.coords:
        coord = data_array.coords["valid_time"]
        dims = tuple(dim for dim in coord.dims if dim in data_array.dims)
        if len(dims) == 1:
            target_valid = pd.Timestamp(task.valid_time_utc)
            if target_valid.tzinfo is None:
                target_valid = target_valid.tz_localize("UTC")
            else:
                target_valid = target_valid.tz_convert("UTC")
            values = pd.to_datetime(np.asarray(coord.values).ravel(), utc=True)
            matches = np.flatnonzero(values == target_valid)
            if len(matches) == 1:
                return data_array.isel({dims[0]: int(matches[0])}), True
    return data_array, False


def select_2d_slice(
    data_array: xr.DataArray,
    task: TaskSpec | None = None,
    *,
    require_task_match: bool = False,
) -> xr.DataArray:
    matched_task_coordinate = False
    if task is not None:
        data_array, matched_task_coordinate = _select_task_coordinate(data_array, task)
    non_spatial = [dim for dim in data_array.dims if dim not in spatial_dims(data_array)]
    if require_task_match and task is not None and non_spatial and not matched_task_coordinate:
        raise ValueError(
            f"Unable to select forecast_hour={task.forecast_hour} from non-spatial dims={non_spatial} "
            "using step or valid_time coordinates."
        )
    selection = {dim: 0 for dim in non_spatial}
    selected = data_array.isel(selection) if selection else data_array
    if selected.ndim != 2:
        raise ValueError(f"Expected 2D grid after selecting non-spatial dims, got ndim={selected.ndim}")
    return selected


def extract_lat_lon(selected: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:
    lat_coord = selected.coords.get("latitude")
    if lat_coord is None:
        lat_coord = selected.coords.get("lat")
    lon_coord = selected.coords.get("longitude")
    if lon_coord is None:
        lon_coord = selected.coords.get("lon")
    if lat_coord is None or lon_coord is None:
        raise ValueError("cfgrib dataset is missing latitude/longitude coordinates")
    lat_values = np.asarray(lat_coord.values, dtype=float)
    lon_values = np.asarray(lon_coord.values, dtype=float)
    if lat_values.ndim == 1 and lon_values.ndim == 1:
        lon_grid, lat_grid = np.meshgrid(lon_values, lat_values)
        return lat_grid, lon_grid
    return lat_values, lon_values


def finite_stat(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def validate_required_columns(payload: dict[str, object], required: set[str], *, label: str) -> None:
    missing = sorted(column for column in required if column not in payload)
    if missing:
        raise ValueError(f"{label} is missing required canonical columns: {', '.join(missing)}")


def canonical_metric_columns(prefix: str, metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {
        prefix: finite_stat(metrics["sample_value"]),
        f"{prefix}_crop_mean": finite_stat(metrics["crop_mean"]),
        f"{prefix}_crop_min": finite_stat(metrics["crop_min"]),
        f"{prefix}_crop_max": finite_stat(metrics["crop_max"]),
        f"{prefix}_crop_std": finite_stat(metrics["crop_std"]),
        f"{prefix}_nb3_mean": finite_stat(metrics["nb3_mean"]),
        f"{prefix}_nb3_min": finite_stat(metrics["nb3_min"]),
        f"{prefix}_nb3_max": finite_stat(metrics["nb3_max"]),
        f"{prefix}_nb3_std": finite_stat(metrics["nb3_std"]),
        f"{prefix}_nb3_gradient_west_east": finite_stat(metrics["nb3_gradient_west_east"]),
        f"{prefix}_nb3_gradient_south_north": finite_stat(metrics["nb3_gradient_south_north"]),
        f"{prefix}_nb7_mean": finite_stat(metrics["nb7_mean"]),
        f"{prefix}_nb7_min": finite_stat(metrics["nb7_min"]),
        f"{prefix}_nb7_max": finite_stat(metrics["nb7_max"]),
        f"{prefix}_nb7_std": finite_stat(metrics["nb7_std"]),
        f"{prefix}_nb7_gradient_west_east": finite_stat(metrics["nb7_gradient_west_east"]),
        f"{prefix}_nb7_gradient_south_north": finite_stat(metrics["nb7_gradient_south_north"]),
    }


def legacy_metric_aliases(prefix: str, canonical_metrics: dict[str, float | None]) -> dict[str, float | None]:
    return {
        f"{prefix}_nearest": canonical_metrics[prefix],
        f"{prefix}_3x3_mean": canonical_metrics[f"{prefix}_nb3_mean"],
        f"{prefix}_3x3_min": canonical_metrics[f"{prefix}_nb3_min"],
        f"{prefix}_3x3_max": canonical_metrics[f"{prefix}_nb3_max"],
        f"{prefix}_3x3_std": canonical_metrics[f"{prefix}_nb3_std"],
        f"{prefix}_3x3_west_east_grad": canonical_metrics[f"{prefix}_nb3_gradient_west_east"],
        f"{prefix}_3x3_south_north_grad": canonical_metrics[f"{prefix}_nb3_gradient_south_north"],
        f"{prefix}_7x7_mean": canonical_metrics[f"{prefix}_nb7_mean"],
        f"{prefix}_7x7_min": canonical_metrics[f"{prefix}_nb7_min"],
        f"{prefix}_7x7_max": canonical_metrics[f"{prefix}_nb7_max"],
        f"{prefix}_7x7_std": canonical_metrics[f"{prefix}_nb7_std"],
        f"{prefix}_7x7_west_east_grad": canonical_metrics[f"{prefix}_nb7_gradient_west_east"],
        f"{prefix}_7x7_south_north_grad": canonical_metrics[f"{prefix}_nb7_gradient_south_north"],
    }


def feature_metrics(
    prefix: str,
    values: np.ndarray,
    *,
    grid_row: int,
    grid_col: int,
    north_is_first: bool,
    include_legacy_aliases: bool = False,
) -> dict[str, float | None]:
    metrics = local_context_metrics(values, row=grid_row, col=grid_col, north_is_first=north_is_first)
    metrics.update(crop_context_metrics(values))
    canonical = canonical_metric_columns(prefix, metrics)
    if include_legacy_aliases:
        canonical.update(legacy_metric_aliases(prefix, canonical))
    return canonical


def provenance_identity(row: dict[str, object]) -> dict[str, object]:
    return {
        "source_model": row["source_model"],
        "source_product": row["source_product"],
        "source_version": row["source_version"],
        "station_id": row["station_id"],
        "init_time_utc": row["init_time_utc"],
        "init_time_local": row["init_time_local"],
        "init_date_local": row["init_date_local"],
        "valid_time_utc": row["valid_time_utc"],
        "valid_time_local": row["valid_time_local"],
        "valid_date_local": row["valid_date_local"],
        "forecast_hour": row["forecast_hour"],
        "nearest_grid_lat": row["nearest_grid_lat"],
        "nearest_grid_lon": row["nearest_grid_lon"],
    }


def populate_task_metadata(row: dict[str, object], task: TaskSpec) -> None:
    row.update(
        {
            "target_date_local": task.target_date_local,
            "slice_policy": task.slice_policy,
            "run_date_utc": task.run_date_utc,
            "cycle_hour_utc": task.cycle_hour_utc,
            "forecast_hour": task.forecast_hour,
            "init_time_utc": task.init_time_utc,
            "init_time_local": task.init_time_local,
            "init_date_local": task.init_date_local,
            "valid_time_utc": task.valid_time_utc,
            "valid_time_local": task.valid_time_local,
            "valid_date_local": task.valid_date_local,
            "init_hour_local": task.init_hour_local,
            "valid_hour_local": task.valid_hour_local,
            "cycle_rank_desc": task.cycle_rank_desc,
            "selected_for_summary": task.selected_for_summary,
            "anchor_cycle_candidate": task.anchor_cycle_candidate,
        }
    )


def rh_from_temp_and_dewpoint_k(temp_k: np.ndarray, dewpoint_k: np.ndarray) -> np.ndarray:
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    es = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    e = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    rh = 100.0 * (e / es)
    return np.clip(rh, 0.0, 100.0)


def parse_inventory_line(line: str) -> dict[str, str] | None:
    match = INVENTORY_RE.match(line.strip())
    if not match:
        return None
    return match.groupdict()


def find_inventory_line(prefix: str, inventory_lines: list[str]) -> str | None:
    pattern = DIRECT_INVENTORY_PATTERNS.get(prefix)
    if pattern is None:
        return None
    candidates = [line for line in inventory_lines if pattern.search(line)]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    signatures: set[tuple[str, str, str, str] | str] = set()
    for line in candidates:
        metadata = parse_inventory_line(line)
        if metadata is None:
            signatures.add(line)
            continue
        signatures.add((metadata["short"], metadata["level"], metadata["step"], metadata["extra"]))
    if prefix == "apcp_surface_kg_m2":
        exact_hour_accum = []
        hourly_accum = []
        cumulative = []
        for line in candidates:
            metadata = parse_inventory_line(line)
            if metadata is None:
                continue
            if re.match(r"^\d+\s+hour\s+acc\s+fcst$", metadata["step"]):
                exact_hour_accum.append(line)
            if re.match(r"^\d+-\d+\s+hour\s+acc\s+fcst$", metadata["step"]) and not metadata["step"].startswith("0-"):
                hourly_accum.append(line)
            if metadata["step"].startswith("0-") and metadata["step"].endswith("hour acc fcst"):
                cumulative.append(line)
        if len(exact_hour_accum) == 1:
            return exact_hour_accum[0]
        if len(hourly_accum) == 1:
            return hourly_accum[0]
        if len(cumulative) == 1:
            return cumulative[0]
    if len(signatures) > 1:
        raise ValueError(f"Ambiguous inventory lines for {prefix}: {candidates}")
    return candidates[0]


def direct_prefix_for_inventory_line(line: str) -> str | None:
    metadata = parse_inventory_line(line)
    if metadata is not None and metadata["short"] == "DPT" and metadata["level"].endswith(" mb"):
        level_text = metadata["level"].split()[0]
        try:
            level = int(level_text)
        except ValueError:
            return None
        if level in ISOBARIC_LEVELS:
            return f"dpt_{level}mb_k_support"
    for prefix, pattern in DIRECT_INVENTORY_PATTERNS.items():
        if pattern.search(line):
            return prefix
    return None


def inventory_line_forecast_hour(line: str) -> int | None:
    metadata = parse_inventory_line(line)
    if metadata is None:
        return None
    step = metadata["step"].strip().lower()
    if step in {"anl", "analysis"}:
        return 0
    match = re.match(r"(?P<hour>\d+)\s+hour\s+fcst", step)
    if match:
        return int(match.group("hour"))
    match = re.match(r"\d+-(?P<hour>\d+)\s+hour\s+(acc|ave)\s+fcst", step)
    if match:
        return int(match.group("hour"))
    match = re.match(r"\d+-(?P<day>\d+)\s+day\s+(acc|ave)\s+fcst", step)
    if match:
        return int(match.group("day")) * 24
    return None


def inventory_lines_for_task(inventory_lines: list[str], task: TaskSpec) -> list[str]:
    task_hour = int(task.forecast_hour)
    filtered = [line for line in inventory_lines if inventory_line_forecast_hour(line) == task_hour]
    return filtered or inventory_lines


def json_list(values: list[str]) -> str:
    return json.dumps(values, sort_keys=True)


def provenance_row(
    *,
    task_key: str,
    row_identity: dict[str, object],
    feature_name: str,
    output_column_base: str,
    grib_short_name: str | None,
    grib_level_text: str | None,
    grib_type_of_level: str | None,
    grib_step_type: str | None,
    grib_step_text: str | None,
    source_inventory_line: str | None,
    units: str | None,
    present_directly: bool,
    derived: bool,
    derivation_method: str | None,
    source_feature_names: list[str] | None,
    missing_optional: bool,
    fallback_used: bool,
    fallback_source_description: str | None,
    notes: str | None,
) -> dict[str, object]:
    row = {
        "task_key": task_key,
        **row_identity,
        "feature_name": feature_name,
        "output_column_base": output_column_base,
        "grib_short_name": grib_short_name,
        "grib_level_text": grib_level_text,
        "grib_type_of_level": grib_type_of_level,
        "grib_step_type": grib_step_type,
        "grib_step_text": grib_step_text,
        "source_inventory_line": source_inventory_line,
        "units": units,
        "present_directly": present_directly,
        "derived": derived,
        "derivation_method": derivation_method,
        "source_feature_names": json_list(source_feature_names or []),
        "missing_optional": missing_optional,
        "fallback_used": fallback_used,
        "fallback_source_description": fallback_source_description,
        "notes": notes,
    }
    validate_required_columns(row, CANONICAL_PROVENANCE_COLUMNS, label=f"HRRR provenance row {feature_name}")
    return row


def direct_provenance_row(
    *,
    task_key: str,
    row_identity: dict[str, object],
    feature_name: str,
    data_array: xr.DataArray,
    group: GroupSpec,
    inventory_line: str | None,
) -> dict[str, object]:
    inventory_meta = parse_inventory_line(inventory_line or "")
    return provenance_row(
        task_key=task_key,
        row_identity=row_identity,
        feature_name=feature_name,
        output_column_base=feature_name,
        grib_short_name=str(data_array.attrs.get("GRIB_shortName") or (inventory_meta or {}).get("short") or data_array.name),
        grib_level_text=(inventory_meta or {}).get("level"),
        grib_type_of_level=str(data_array.attrs.get("GRIB_typeOfLevel") or group.filter_by_keys.get("typeOfLevel") or ""),
        grib_step_type=str(data_array.attrs.get("GRIB_stepType") or group.filter_by_keys.get("stepType") or ""),
        grib_step_text=(inventory_meta or {}).get("step"),
        source_inventory_line=inventory_line,
        units=str(data_array.attrs.get("units") or ""),
        present_directly=True,
        derived=False,
        derivation_method=None,
        source_feature_names=[],
        missing_optional=False,
        fallback_used=False,
        fallback_source_description=None,
        notes=(inventory_meta or {}).get("extra") or None,
    )


def direct_eccodes_provenance_row(
    *,
    task_key: str,
    row_identity: dict[str, object],
    feature_name: str,
    inventory_line: str | None,
    units: str | None,
) -> dict[str, object]:
    inventory_meta = parse_inventory_line(inventory_line or "")
    return provenance_row(
        task_key=task_key,
        row_identity=row_identity,
        feature_name=feature_name,
        output_column_base=feature_name,
        grib_short_name=(inventory_meta or {}).get("short"),
        grib_level_text=(inventory_meta or {}).get("level"),
        grib_type_of_level=None,
        grib_step_type=None,
        grib_step_text=(inventory_meta or {}).get("step"),
        source_inventory_line=inventory_line,
        units=units,
        present_directly=True,
        derived=False,
        derivation_method=None,
        source_feature_names=[],
        missing_optional=False,
        fallback_used=False,
        fallback_source_description=None,
        notes=(inventory_meta or {}).get("extra") or "extracted with direct ecCodes prototype",
    )


def derived_provenance_row(
    *,
    task_key: str,
    row_identity: dict[str, object],
    feature_name: str,
    units: str | None,
    derivation_method: str,
    source_feature_names: list[str],
    notes: str | None = None,
) -> dict[str, object]:
    return provenance_row(
        task_key=task_key,
        row_identity=row_identity,
        feature_name=feature_name,
        output_column_base=feature_name,
        grib_short_name=None,
        grib_level_text=None,
        grib_type_of_level=None,
        grib_step_type=None,
        grib_step_text=None,
        source_inventory_line=None,
        units=units,
        present_directly=False,
        derived=True,
        derivation_method=derivation_method,
        source_feature_names=source_feature_names,
        missing_optional=False,
        fallback_used=False,
        fallback_source_description=None,
        notes=notes,
    )


def missing_provenance_row(
    *,
    task_key: str,
    row_identity: dict[str, object],
    feature_name: str,
) -> dict[str, object]:
    return provenance_row(
        task_key=task_key,
        row_identity=row_identity,
        feature_name=feature_name,
        output_column_base=feature_name,
        grib_short_name=None,
        grib_level_text=None,
        grib_type_of_level=None,
        grib_step_type=None,
        grib_step_text=None,
        source_inventory_line=None,
        units=None,
        present_directly=False,
        derived=False,
        derivation_method=None,
        source_feature_names=[],
        missing_optional=True,
        fallback_used=False,
        fallback_source_description=None,
        notes="Requested feature family was not present in the reduced GRIB inventory.",
    )


def convert_family_units(
    row: dict[str, object],
    *,
    source_base: str,
    target_base: str,
    transform,
) -> None:
    for key, value in list(row.items()):
        if key == source_base or key.startswith(f"{source_base}_"):
            if isinstance(value, (int, float)) and value is not None:
                row[key.replace(source_base, target_base, 1)] = transform(float(value))


def _is_temperature_delta_suffix(suffix: str) -> bool:
    return suffix.endswith("_std") or suffix.endswith("_gradient_west_east") or suffix.endswith("_gradient_south_north")


def add_temperature_conversions(row: dict[str, object], provenance_rows: dict[str, dict[str, object]]) -> None:
    for base in list(REQUESTED_FIELD_PREFIXES):
        if not (base.startswith("tmp_") or base.startswith("dpt_")) or row.get(base) is None:
            continue
        target_base = base.replace("_k", "_f", 1)
        for key, value in list(row.items()):
            if key != base and not key.startswith(f"{base}_"):
                continue
            if not isinstance(value, (int, float)) or value is None:
                continue
            suffix = key[len(base) :]
            if suffix and _is_temperature_delta_suffix(suffix):
                converted = float(value) * 9.0 / 5.0
            else:
                converted = (float(value) - 273.15) * 9.0 / 5.0 + 32.0
            row[key.replace(base, target_base, 1)] = converted
        provenance_rows[target_base] = derived_provenance_row(
            task_key=str(row["task_key"]),
            row_identity=provenance_identity(row),
            feature_name=target_base,
            units="degF",
            derivation_method="kelvin_to_fahrenheit",
            source_feature_names=[base],
        )


def add_wind_derivatives(
    row: dict[str, object],
    provenance_rows: dict[str, dict[str, object]],
    support_arrays: dict[str, np.ndarray],
    *,
    grid_row: int,
    grid_col: int,
    north_is_first: bool,
) -> None:
    u_grid = support_arrays.get("ugrd_10m_ms")
    v_grid = support_arrays.get("vgrd_10m_ms")
    if u_grid is None or v_grid is None:
        return

    speed_ms = np.hypot(u_grid, v_grid)
    row.update(
        feature_metrics(
            "wind_10m_speed_ms",
            speed_ms,
            grid_row=grid_row,
            grid_col=grid_col,
            north_is_first=north_is_first,
        )
    )
    convert_family_units(
        row,
        source_base="wind_10m_speed_ms",
        target_base="wind_10m_speed_mph",
        transform=lambda value: value * 2.2369362920544,
    )

    direction_deg = (np.degrees(np.arctan2(-u_grid, -v_grid)) + 360.0) % 360.0
    row.update(
        feature_metrics(
            "wind_10m_direction_deg",
            direction_deg,
            grid_row=grid_row,
            grid_col=grid_col,
            north_is_first=north_is_first,
        )
    )

    for feature_name, units, method in (
        ("wind_10m_speed_ms", "m s-1", "sqrt(u10^2 + v10^2)"),
        ("wind_10m_speed_mph", "mph", "sqrt(u10^2 + v10^2) converted from m s-1 to mph"),
        ("wind_10m_direction_deg", "degrees", "meteorological wind direction from u10 and v10"),
    ):
        if feature_name in row:
            provenance_rows[feature_name] = derived_provenance_row(
                task_key=str(row["task_key"]),
                row_identity=provenance_identity(row),
                feature_name=feature_name,
                units=units,
                derivation_method=method,
                source_feature_names=["ugrd_10m_ms", "vgrd_10m_ms"],
            )


def missing_prefixes_from_row(row: dict[str, object], *, task: TaskSpec | None = None) -> list[str]:
    return [prefix for prefix in requested_field_prefixes_for_task(task) if row.get(prefix) is None]


def isobaric_specs() -> list[GroupSpec]:
    specs: list[GroupSpec] = []
    for level in ISOBARIC_LEVELS:
        specs.append(
            GroupSpec(
                name=f"isobaric_{level}",
                filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": level},
                vars_map={
                    "t": f"tmp_{level}mb_k",
                    "u": f"ugrd_{level}mb_ms",
                    "v": f"vgrd_{level}mb_ms",
                    "gh": f"hgt_{level}mb_gpm",
                    "dpt": f"dpt_{level}mb_k_support",
                    "r": f"rh_{level}mb_pct",
                    "q": f"spfh_{level}mb_kg_kg",
                },
                inventory_checks=(
                    f":TMP:{level} mb:",
                    f":UGRD:{level} mb:",
                    f":VGRD:{level} mb:",
                    f":HGT:{level} mb:",
                    f":DPT:{level} mb:",
                    f":RH:{level} mb:",
                    f":SPFH:{level} mb:",
                ),
            )
        )
    return specs


def open_reduced_grib_group_datasets(
    reduced_path: Path,
    reduced_inventory: list[str],
    *,
    diagnostics: dict[str, object] | None = None,
    cfgrib_index_dir: Path,
) -> list[tuple[GroupSpec, xr.Dataset]]:
    all_groups = GROUP_SPECS + isobaric_specs()
    inventory_blob = "\n".join(reduced_inventory)
    group_datasets: list[tuple[GroupSpec, xr.Dataset]] = []
    try:
        for group in all_groups:
            if not any(check in inventory_blob for check in group.inventory_checks):
                continue
            try:
                cfgrib_open_started_at = time.perf_counter()
                ds = open_group_dataset_with_optional_index(
                    reduced_path,
                    group,
                    indexpath=str(cfgrib_index_dir / f"{group.name}.idx"),
                )
                if diagnostics is not None:
                    diagnostics["timing_cfgrib_open_seconds"] = float(diagnostics.get("timing_cfgrib_open_seconds", 0.0)) + (
                        time.perf_counter() - cfgrib_open_started_at
                    )
                group_datasets.append((group, ds))
            except Exception as exc:
                raise RuntimeError(f"cfgrib open failed for group {group.name}: {exc}") from exc
    except Exception:
        close_group_datasets(group_datasets)
        raise
    return group_datasets


def close_group_datasets(group_datasets: Iterable[tuple[GroupSpec, xr.Dataset]]) -> None:
    for _group, ds in group_datasets:
        with contextlib.suppress(Exception):
            ds.close()


def eccodes_grid_arrays(gid: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import eccodes

    nx = int(eccodes.codes_get(gid, "Nx"))
    ny = int(eccodes.codes_get(gid, "Ny"))
    values = np.asarray(eccodes.codes_get_values(gid), dtype=float).reshape(ny, nx)
    lat_grid = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype=float).reshape(ny, nx)
    lon_grid = np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype=float).reshape(ny, nx)
    return values, lat_grid, lon_grid


def build_task_result_with_eccodes(
    reduced_path: Path,
    reduced_inventory: list[str],
    task: TaskSpec,
    grib_url: str,
    *,
    diagnostics: dict[str, object] | None = None,
    include_legacy_aliases: bool = False,
    filter_inventory_to_task_step: bool = False,
    write_provenance: bool = True,
) -> TaskResult:
    import eccodes

    diagnostics = dict(diagnostics or default_task_diagnostics(task))
    diagnostics["extract_method"] = "eccodes"
    diagnostics.setdefault("timing_cfgrib_open_seconds", 0.0)
    row: dict[str, object] = {
        "task_key": task.key,
        "source_model": DEFAULT_SOURCE_MODEL,
        "source_product": DEFAULT_SOURCE_PRODUCT,
        "source_version": DEFAULT_SOURCE_VERSION,
        "fallback_used_any": False,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "grib_url": grib_url,
        **settlement_metadata(),
        **crop_metadata(),
        "target_lat": SETTLEMENT_LOCATION.lat,
        "target_lon": settlement_longitude_360(),
    }
    populate_task_metadata(row, task)

    task_inventory = inventory_lines_for_task(reduced_inventory, task) if filter_inventory_to_task_step else reduced_inventory
    task_inventory_set = set(task_inventory)
    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    grid_row: int | None = None
    grid_col: int | None = None
    north_is_first: bool | None = None
    support_arrays: dict[str, np.ndarray] = {}
    provenance_rows: dict[str, dict[str, object]] = {}
    message_count = 0
    used_message_count = 0
    started_at = time.perf_counter()

    try:
        with reduced_path.open("rb") as handle:
            while True:
                gid = eccodes.codes_grib_new_from_file(handle)
                if gid is None:
                    break
                try:
                    inventory_line = reduced_inventory[message_count] if message_count < len(reduced_inventory) else None
                    message_count += 1
                    if inventory_line is None:
                        continue
                    if filter_inventory_to_task_step and inventory_line not in task_inventory_set:
                        continue
                    prefix = direct_prefix_for_inventory_line(inventory_line)
                    if prefix is None:
                        continue
                    values, lat_grid, lon_grid = eccodes_grid_arrays(gid)
                    used_message_count += 1
                    if latitude is None or longitude is None:
                        latitude = lat_grid
                        longitude = lon_grid
                        nearest = find_nearest_grid_cell(
                            latitude,
                            longitude,
                            station_lat=SETTLEMENT_LOCATION.lat,
                            station_lon=SETTLEMENT_LOCATION.lon,
                        )
                        grid_row = int(nearest["grid_row"])
                        grid_col = int(nearest["grid_col"])
                        north_is_first = infer_north_is_first(latitude)
                        row["grid_row"] = grid_row
                        row["grid_col"] = grid_col
                        row["nearest_grid_lat"] = float(nearest["grid_lat"])
                        row["nearest_grid_lon"] = float(longitude_360_to_180(nearest["grid_lon"]))
                        row["grid_lat"] = float(nearest["grid_lat"])
                        row["grid_lon"] = float(nearest["grid_lon"])

                    units: str | None
                    try:
                        units = str(eccodes.codes_get(gid, "units"))
                    except Exception:
                        units = None
                    support_arrays[prefix] = values
                    if prefix.endswith("_support"):
                        continue
                    assert grid_row is not None and grid_col is not None and north_is_first is not None
                    row.update(
                        feature_metrics(
                            prefix,
                            values,
                            grid_row=grid_row,
                            grid_col=grid_col,
                            north_is_first=north_is_first,
                            include_legacy_aliases=include_legacy_aliases,
                        )
                    )
                    if write_provenance:
                        provenance_rows[prefix] = direct_eccodes_provenance_row(
                            task_key=task.key,
                            row_identity=provenance_identity(row),
                            feature_name=prefix,
                            inventory_line=inventory_line,
                            units=units,
                        )
                finally:
                    eccodes.codes_release(gid)
    except Exception as exc:
        diagnostics["timing_direct_extract_seconds"] = time.perf_counter() - started_at
        return TaskResult(False, task.key, None, [], [], f"ecCodes direct extraction failed: {exc}", diagnostics)

    diagnostics["timing_direct_extract_seconds"] = time.perf_counter() - started_at
    diagnostics["direct_message_count"] = message_count
    diagnostics["direct_used_message_count"] = used_message_count

    if latitude is None or longitude is None or grid_row is None or grid_col is None or north_is_first is None:
        return TaskResult(False, task.key, None, [], [], "Unable to locate KLGA grid cell in reduced GRIB2", diagnostics)
    if "init_time_utc" not in row or "valid_time_utc" not in row or "init_time_local" not in row:
        return TaskResult(False, task.key, None, [], [], "Reduced GRIB2 did not expose init/valid timestamps", diagnostics)

    row_build_started_at = time.perf_counter()
    for level in ISOBARIC_LEVELS:
        rh_prefix = f"rh_{level}mb_pct"
        if row.get(rh_prefix) is None:
            temp = support_arrays.get(f"tmp_{level}mb_k")
            dewpoint = support_arrays.get(f"dpt_{level}mb_k_support")
            if temp is not None and dewpoint is not None:
                rh = rh_from_temp_and_dewpoint_k(temp, dewpoint)
                row.update(
                    feature_metrics(
                        rh_prefix,
                        rh,
                        grid_row=grid_row,
                        grid_col=grid_col,
                        north_is_first=north_is_first,
                        include_legacy_aliases=include_legacy_aliases,
                    )
                )
                if write_provenance:
                    provenance_rows[rh_prefix] = derived_provenance_row(
                        task_key=task.key,
                        row_identity=provenance_identity(row),
                        feature_name=rh_prefix,
                        units="%",
                        derivation_method="relative_humidity_from_temperature_and_dewpoint",
                        source_feature_names=[f"tmp_{level}mb_k", f"dpt_{level}mb_k_support"],
                        notes=f"Derived because direct {rh_prefix} was not present.",
                    )

    add_temperature_conversions(row, provenance_rows if write_provenance else {})
    add_wind_derivatives(
        row,
        provenance_rows if write_provenance else {},
        support_arrays,
        grid_row=grid_row,
        grid_col=grid_col,
        north_is_first=north_is_first,
    )
    diagnostics["timing_row_build_seconds"] = float(diagnostics.get("timing_row_build_seconds", 0.0)) + (
        time.perf_counter() - row_build_started_at
    )

    missing_fields = missing_prefixes_from_row(row, task=task)
    row["missing_optional_any"] = bool(missing_fields)
    row["missing_optional_fields_count"] = len(missing_fields)
    for prefix in missing_fields:
        if write_provenance:
            provenance_rows.setdefault(
                prefix,
                missing_provenance_row(
                    task_key=task.key,
                    row_identity=provenance_identity(row),
                    feature_name=prefix,
                ),
            )

    validate_required_columns(row, CANONICAL_WIDE_COLUMNS, label=f"HRRR wide row {task.key}")
    return TaskResult(
        True,
        task.key,
        row,
        sorted(provenance_rows.values(), key=lambda item: item["feature_name"]) if write_provenance else [],
        missing_fields,
        None,
        diagnostics,
    )


def build_task_result_from_open_datasets(
    group_datasets: list[tuple[GroupSpec, xr.Dataset]],
    reduced_inventory: list[str],
    task: TaskSpec,
    grib_url: str,
    *,
    diagnostics: dict[str, object] | None = None,
    include_legacy_aliases: bool = False,
    filter_inventory_to_task_step: bool = False,
    write_provenance: bool = True,
) -> TaskResult:
    diagnostics = dict(diagnostics or default_task_diagnostics(task))
    row: dict[str, object] = {
        "task_key": task.key,
        "source_model": DEFAULT_SOURCE_MODEL,
        "source_product": DEFAULT_SOURCE_PRODUCT,
        "source_version": DEFAULT_SOURCE_VERSION,
        "fallback_used_any": False,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "grib_url": grib_url,
        **settlement_metadata(),
        **crop_metadata(),
        "target_lat": SETTLEMENT_LOCATION.lat,
        "target_lon": settlement_longitude_360(),
    }
    populate_task_metadata(row, task)

    latitude: np.ndarray | None = None
    longitude: np.ndarray | None = None
    grid_row: int | None = None
    grid_col: int | None = None
    north_is_first: bool | None = None
    support_arrays: dict[str, np.ndarray] = {}
    provenance_rows: dict[str, dict[str, object]] = {}
    task_inventory = inventory_lines_for_task(reduced_inventory, task) if filter_inventory_to_task_step else reduced_inventory
    inventory_blob = "\n".join(task_inventory)

    for group, ds in group_datasets:
        if not any(check in inventory_blob for check in group.inventory_checks):
            continue
        try:
            row_build_started_at = time.perf_counter()
            for ds_var_name, prefix in group.vars_map.items():
                if ds_var_name not in ds.data_vars:
                    continue
                selected = select_2d_slice(
                    ds[ds_var_name],
                    task=task,
                    require_task_match=filter_inventory_to_task_step,
                )
                values = np.asarray(selected.values, dtype=float)
                if latitude is None or longitude is None:
                    latitude, longitude = extract_lat_lon(selected)
                    nearest = find_nearest_grid_cell(
                        latitude,
                        longitude,
                        station_lat=SETTLEMENT_LOCATION.lat,
                        station_lon=SETTLEMENT_LOCATION.lon,
                    )
                    grid_row = int(nearest["grid_row"])
                    grid_col = int(nearest["grid_col"])
                    north_is_first = infer_north_is_first(latitude)
                    row["grid_row"] = grid_row
                    row["grid_col"] = grid_col
                    row["nearest_grid_lat"] = float(nearest["grid_lat"])
                    row["nearest_grid_lon"] = float(longitude_360_to_180(nearest["grid_lon"]))
                    row["grid_lat"] = float(nearest["grid_lat"])
                    row["grid_lon"] = float(nearest["grid_lon"])

                if prefix.endswith("_support"):
                    support_arrays[prefix] = values
                    continue

                assert grid_row is not None and grid_col is not None and north_is_first is not None
                row.update(
                    feature_metrics(
                        prefix,
                        values,
                        grid_row=grid_row,
                        grid_col=grid_col,
                        north_is_first=north_is_first,
                        include_legacy_aliases=include_legacy_aliases,
                    )
                )
                support_arrays[prefix] = values
                inventory_line = find_inventory_line(prefix, task_inventory)
                if write_provenance and "init_time_utc" in row and "valid_time_utc" in row:
                    provenance_rows[prefix] = direct_provenance_row(
                        task_key=task.key,
                        row_identity=provenance_identity(row),
                        feature_name=prefix,
                        data_array=selected,
                        group=group,
                        inventory_line=inventory_line,
                    )
            diagnostics["timing_row_build_seconds"] = float(diagnostics.get("timing_row_build_seconds", 0.0)) + (
                time.perf_counter() - row_build_started_at
            )
        except Exception as exc:
            return TaskResult(False, task.key, None, [], [], f"Failed to extract group {group.name}: {exc}", diagnostics)

    if latitude is None or longitude is None or grid_row is None or grid_col is None:
        return TaskResult(False, task.key, None, [], [], "Unable to locate KLGA grid cell in reduced GRIB2", diagnostics)
    if "init_time_utc" not in row or "valid_time_utc" not in row or "init_time_local" not in row:
        return TaskResult(False, task.key, None, [], [], "Reduced GRIB2 did not expose init/valid timestamps", diagnostics)

    for level in ISOBARIC_LEVELS:
        rh_prefix = f"rh_{level}mb_pct"
        if row.get(rh_prefix) is None:
            temp = support_arrays.get(f"tmp_{level}mb_k")
            dewpoint = support_arrays.get(f"dpt_{level}mb_k_support")
            if temp is not None and dewpoint is not None:
                rh = rh_from_temp_and_dewpoint_k(temp, dewpoint)
                row.update(
                    feature_metrics(
                        rh_prefix,
                        rh,
                        grid_row=grid_row,
                        grid_col=grid_col,
                        north_is_first=north_is_first,
                        include_legacy_aliases=include_legacy_aliases,
                    )
                )
                if write_provenance:
                    provenance_rows[rh_prefix] = derived_provenance_row(
                        task_key=task.key,
                        row_identity=provenance_identity(row),
                        feature_name=rh_prefix,
                        units="%",
                        derivation_method="relative_humidity_from_temperature_and_dewpoint",
                        source_feature_names=[f"tmp_{level}mb_k", f"dpt_{level}mb_k_support"],
                        notes=f"Derived because direct {rh_prefix} was not present.",
                    )

    add_temperature_conversions(row, provenance_rows if write_provenance else {})
    add_wind_derivatives(
        row,
        provenance_rows if write_provenance else {},
        support_arrays,
        grid_row=grid_row,
        grid_col=grid_col,
        north_is_first=north_is_first,
    )

    missing_fields = missing_prefixes_from_row(row, task=task)
    row["missing_optional_any"] = bool(missing_fields)
    row["missing_optional_fields_count"] = len(missing_fields)

    for prefix in missing_fields:
        if write_provenance:
            provenance_rows.setdefault(
                prefix,
                missing_provenance_row(
                    task_key=task.key,
                    row_identity=provenance_identity(row),
                    feature_name=prefix,
                ),
            )

    validate_required_columns(row, CANONICAL_WIDE_COLUMNS, label=f"HRRR wide row {task.key}")

    return TaskResult(
        True,
        task.key,
        row,
        sorted(provenance_rows.values(), key=lambda item: item["feature_name"]) if write_provenance else [],
        missing_fields,
        None,
        diagnostics,
    )


def process_reduced_grib(
    reduced_path: Path,
    reduced_inventory: list[str],
    task: TaskSpec,
    grib_url: str,
    *,
    cfgrib_index_dir: Path | None = None,
    diagnostics: dict[str, object] | None = None,
    include_legacy_aliases: bool = False,
    filter_inventory_to_task_step: bool = False,
    write_provenance: bool = True,
) -> TaskResult:
    diagnostics = dict(diagnostics or default_task_diagnostics(task))
    if RUNTIME_OPTIONS.extract_method == "eccodes":
        return build_task_result_with_eccodes(
            reduced_path,
            reduced_inventory,
            task,
            grib_url,
            diagnostics=diagnostics,
            include_legacy_aliases=include_legacy_aliases,
            filter_inventory_to_task_step=filter_inventory_to_task_step,
            write_provenance=write_provenance,
        )
    diagnostics["extract_method"] = "cfgrib"
    local_index_dir = cfgrib_index_dir
    created_index_dir = False
    if local_index_dir is None:
        local_index_dir = Path(tempfile.mkdtemp(prefix=f"hrrr_cfgrib_{task.forecast_hour:02d}_"))
        created_index_dir = True
    group_datasets: list[tuple[GroupSpec, xr.Dataset]] = []
    try:
        try:
            group_datasets = open_reduced_grib_group_datasets(
                reduced_path,
                reduced_inventory,
                diagnostics=diagnostics,
                cfgrib_index_dir=local_index_dir,
            )
        except Exception as exc:
            return TaskResult(False, task.key, None, [], [], str(exc), diagnostics)
        return build_task_result_from_open_datasets(
            group_datasets,
            reduced_inventory,
            task,
            grib_url,
            diagnostics=diagnostics,
            include_legacy_aliases=include_legacy_aliases,
            filter_inventory_to_task_step=filter_inventory_to_task_step,
            write_provenance=write_provenance,
        )
    finally:
        close_group_datasets(group_datasets)
        if created_index_dir and local_index_dir is not None:
            shutil.rmtree(local_index_dir, ignore_errors=True)


def task_remote_url(task: TaskSpec, source: str = "google") -> str:
    date_token = task.run_date_utc.replace("-", "")
    grib_url, _ = build_remote_paths(date_token, task.cycle_hour_utc, "surface", task.forecast_hour, source)
    return grib_url


def default_task_diagnostics(task: TaskSpec) -> dict[str, object]:
    return {
        "task_key": task.key,
        "field_profile": task_field_profile(task),
        "extract_method": RUNTIME_OPTIONS.extract_method,
        "head_used": False,
        "remote_file_size": None,
        "selected_record_count": 0,
        "merged_range_count": 0,
        "downloaded_range_bytes": 0,
        "timing_idx_fetch_seconds": 0.0,
        "timing_idx_parse_seconds": 0.0,
        "timing_head_seconds": 0.0,
        "timing_range_download_seconds": 0.0,
        "timing_wgrib_inventory_seconds": 0.0,
        "timing_reduce_seconds": 0.0,
        "crop_method": None,
        "crop_command": None,
        "crop_grid_cache_key": None,
        "crop_grid_cache_hit": False,
        "crop_ij_box": None,
        "crop_wgrib2_threads": None,
        "crop_fallback_reason": None,
        "timing_cfgrib_open_seconds": 0.0,
        "timing_row_build_seconds": 0.0,
        "timing_cleanup_seconds": 0.0,
        "reduced_reused": False,
        "reduced_reuse_signature": None,
        "cfgrib_index_strategy": DEFAULT_CFGRIB_INDEX_STRATEGY,
        "scratch_dir": None,
        "attempt_count": 1,
        "retried": False,
        "retry_recovered": False,
        "final_error_class": None,
        "last_error_type": None,
        "last_error_message": None,
        "raw_file_path": None,
        "raw_file_size": None,
        "raw_manifest_path": None,
        "raw_selection_manifest_path": None,
        "reduced_file_path": None,
        "reduced_file_size": None,
        "grib_url": None,
        "summary_profile": RUNTIME_OPTIONS.summary_profile,
        "provenance_written": not RUNTIME_OPTIONS.skip_provenance,
    }


def reduced_signature_path(reduced_path: Path) -> Path:
    return reduced_path.with_suffix(f"{reduced_path.suffix}.reuse.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reduced_inventory_path(reduced_path: Path) -> Path:
    return reduced_path.with_suffix(f"{reduced_path.suffix}.inventory.json")


def selection_manifest_signature(path: Path) -> str | None:
    try:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    except OSError:
        return None
    if len(lines) < 2:
        return None
    return lines[1]


def build_reduced_reuse_signature(*, task: TaskSpec, raw_path: Path, selection_manifest_path: Path | None = None) -> str:
    payload = {
        "source_model": DEFAULT_SOURCE_MODEL,
        "source_product": DEFAULT_SOURCE_PRODUCT,
        "source_version": DEFAULT_SOURCE_VERSION,
        "task_key": task.key,
        "raw_file_name": raw_path.name,
        "raw_file_size": raw_path.stat().st_size if raw_path.exists() else None,
        "selection_manifest_signature": (
            selection_manifest_signature(selection_manifest_path) if selection_manifest_path is not None else None
        ),
        "crop_bounds": {
            "top": REGIONAL_CROP_BOUNDS.top,
            "bottom": REGIONAL_CROP_BOUNDS.bottom,
            "left": REGIONAL_CROP_BOUNDS.left,
            "right": REGIONAL_CROP_BOUNDS.right,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def reduced_grib_reusable(*, reduced_path: Path, signature: str) -> bool:
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


def write_reduced_reuse_signature(reduced_path: Path, signature: str) -> None:
    reduced_signature_path(reduced_path).write_text(json.dumps({"signature": signature}, sort_keys=True))


def load_reduced_inventory(*, reduced_path: Path, signature: str) -> list[str] | None:
    path = reduced_inventory_path(reduced_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("signature") != signature:
        return None
    inventory = payload.get("inventory_lines")
    if not isinstance(inventory, list):
        return None
    return [str(line) for line in inventory if str(line).strip()]


def write_reduced_inventory(reduced_path: Path, signature: str, inventory_lines: list[str]) -> None:
    reduced_inventory_path(reduced_path).write_text(
        json.dumps({"signature": signature, "inventory_lines": list(inventory_lines)}, sort_keys=True)
    )


def manifest_selected_inventory_lines(task: TaskSpec, manifest_path: Path) -> list[str]:
    run_stamp = f"d={pd.Timestamp(task.init_time_utc).tz_convert('UTC').strftime('%Y%m%d%H')}"
    return manifest_inventory_lines(manifest_path, run_stamp=run_stamp)


def reduce_grib2_for_task(
    wgrib2_path: str,
    raw_path: Path,
    reduced_path: Path,
    *,
    task: TaskSpec,
    raw_manifest_path: Path,
) -> CropExecutionResult:
    del raw_manifest_path
    return reduce_grib2(wgrib2_path, raw_path, reduced_path, task=task)


def cycle_key_for_task(task: TaskSpec) -> tuple[str, int]:
    return task.run_date_utc, int(task.cycle_hour_utc)


def batch_artifact_paths(reduced_dir: Path, task: TaskSpec) -> tuple[Path, Path]:
    cycle_root = reduced_dir / "batch" / f"hrrr.{task.run_date_utc.replace('-', '')}.t{task.cycle_hour_utc:02d}z"
    return (
        cycle_root / "selected_multiforecast.grib2",
        cycle_root / "selected_multiforecast.reduced.grib2",
    )


def concatenate_files(paths: list[Path], destination: Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    with destination.open("wb") as out_handle:
        for path in paths:
            with path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)
    return time.perf_counter() - started_at


def reduce_grib2_for_batch(
    wgrib2_path: str,
    raw_paths: list[Path],
    batch_raw_path: Path,
    batch_reduced_path: Path,
    *,
    task: TaskSpec,
) -> tuple[list[str], set[str], float, float, float, CropExecutionResult]:
    concat_seconds = concatenate_files(raw_paths, batch_raw_path)
    crop_result = reduce_grib2(
        wgrib2_path,
        batch_raw_path,
        batch_reduced_path,
        task=task,
    )
    return (
        crop_result.selected_lines,
        crop_result.matched_names,
        crop_result.inventory_seconds,
        crop_result.reduce_seconds,
        concat_seconds,
        crop_result,
    )


def download_task_subset(
    task: TaskSpec,
    *,
    raw_path: Path,
    raw_manifest_path: Path,
    raw_selection_manifest_path: Path,
    range_merge_gap_bytes: int,
    overwrite: bool,
    reporter=None,
    worker_id: str | None = None,
) -> object:
    fetch_kwargs = {
        "date": task.run_date_utc.replace("-", ""),
        "cycle": task.cycle_hour_utc,
        "product": "surface",
        "forecast_hour": task.forecast_hour,
        "source": "google",
        "patterns": [pattern for _, pattern in inventory_selection_patterns_for_task(task)],
        "subset_path": raw_path,
        "manifest_path": raw_manifest_path,
        "selection_manifest_path": raw_selection_manifest_path,
        "overwrite": overwrite,
    }
    try:
        return download_subset_for_inventory_patterns(
            **fetch_kwargs,
            range_merge_gap_bytes=range_merge_gap_bytes,
            progress_callback=(
                (
                    lambda event_name, payload: _report_hrrr_download_event(
                        reporter=reporter,
                        worker_id=str(worker_id),
                        event_name=event_name,
                        payload=payload,
                    )
                )
                if reporter is not None and worker_id is not None
                else None
            ),
        )
    except TypeError as exc:
        if "range_merge_gap_bytes" not in str(exc) and "progress_callback" not in str(exc):
            raise
        return download_subset_for_inventory_patterns(**fetch_kwargs)


def _process_task_once(
    task: TaskSpec,
    *,
    wgrib2_path: str,
    download_dir: Path,
    reduced_dir: Path,
    phase_limits: PhaseConcurrencyLimits,
    keep_downloads: bool,
    keep_reduced: bool,
    include_legacy_aliases: bool,
    range_merge_gap_bytes: int = DEFAULT_RANGE_MERGE_GAP_BYTES,
    scratch_dir: Path | None = None,
    overwrite: bool = False,
    reporter=None,
    attempt: int = 1,
    max_attempts: int = 1,
) -> TaskResult:
    raw_path = path_for_raw(download_dir, task)
    raw_manifest_path = path_for_raw_manifest(download_dir, task)
    raw_selection_manifest_path = path_for_raw_selection_manifest(download_dir, task)
    reduced_path = path_for_reduced(reduced_dir, task)
    grib_url = task_remote_url(task, "google")
    diagnostics = default_task_diagnostics(task)
    diagnostics["scratch_dir"] = str(scratch_dir) if scratch_dir is not None else None
    cfgrib_index_dir: Path | None = None
    worker_id = threading.current_thread().name
    worker_label = f"{task.target_date_local} c{task.cycle_hour_utc:02d} f{task.forecast_hour:02d}"
    diagnostics["attempt_count"] = attempt
    diagnostics["retried"] = attempt > 1
    diagnostics["retry_recovered"] = False
    diagnostics["final_error_class"] = None
    diagnostics["last_error_type"] = None
    diagnostics["last_error_message"] = None
    diagnostics["raw_file_path"] = str(raw_path)
    diagnostics["raw_manifest_path"] = str(raw_manifest_path)
    diagnostics["raw_selection_manifest_path"] = str(raw_selection_manifest_path)
    diagnostics["reduced_file_path"] = str(reduced_path)
    diagnostics["grib_url"] = grib_url

    try:
        if reporter is not None:
            reporter.start_worker(
                worker_id,
                label=worker_label,
                phase="download",
                group_id=month_id_for_task(task),
                details="fetch_subset",
            )
            reporter.set_worker_attempt(worker_id, attempt=attempt, max_attempts=max_attempts)
        fetch_kwargs = {
            "date": task.run_date_utc.replace("-", ""),
            "cycle": task.cycle_hour_utc,
            "product": "surface",
            "forecast_hour": task.forecast_hour,
            "source": "google",
            "patterns": [pattern for _, pattern in inventory_selection_patterns_for_task(task)],
            "subset_path": raw_path,
            "manifest_path": raw_manifest_path,
            "selection_manifest_path": raw_selection_manifest_path,
            "overwrite": overwrite,
        }
        with phase_gate(
            semaphore=phase_limits.download_semaphore,
            reporter=reporter,
            worker_id=worker_id,
            wait_phase="download_wait",
            active_phase="download",
            details="fetch_subset",
        ):
            try:
                fetch_result = download_subset_for_inventory_patterns(
                    **fetch_kwargs,
                    range_merge_gap_bytes=range_merge_gap_bytes,
                    progress_callback=(
                        (
                            lambda event_name, payload: _report_hrrr_download_event(
                                reporter=reporter,
                                worker_id=worker_id,
                                event_name=event_name,
                                payload=payload,
                            )
                        )
                        if reporter is not None
                        else None
                    ),
                )
            except TypeError as exc:
                if "range_merge_gap_bytes" not in str(exc) and "progress_callback" not in str(exc):
                    raise
                fetch_result = download_subset_for_inventory_patterns(**fetch_kwargs)
        if isinstance(fetch_result, Path):
            class _LegacyFetchResult:
                head_used = False
                remote_file_size = None
                selected_record_count = 0
                merged_range_count = 0
                downloaded_range_bytes = 0
                timing_idx_fetch_seconds = 0.0
                timing_idx_parse_seconds = 0.0
                timing_head_seconds = 0.0
                timing_range_download_seconds = 0.0
            fetch_result = _LegacyFetchResult()
        diagnostics.update(
            {
                "head_used": fetch_result.head_used,
                "remote_file_size": fetch_result.remote_file_size,
                "selected_record_count": fetch_result.selected_record_count,
                "merged_range_count": fetch_result.merged_range_count,
                "downloaded_range_bytes": fetch_result.downloaded_range_bytes,
                "timing_idx_fetch_seconds": fetch_result.timing_idx_fetch_seconds,
                "timing_idx_parse_seconds": fetch_result.timing_idx_parse_seconds,
                "timing_head_seconds": fetch_result.timing_head_seconds,
                "timing_range_download_seconds": fetch_result.timing_range_download_seconds,
                "raw_file_size": raw_path.stat().st_size if raw_path.exists() else None,
            }
        )
        reuse_signature: str | None = None
        if keep_reduced:
            reuse_signature = build_reduced_reuse_signature(
                task=task,
                raw_path=raw_path,
                selection_manifest_path=raw_selection_manifest_path,
            )
            diagnostics["reduced_reuse_signature"] = reuse_signature
        if keep_reduced and not overwrite and reuse_signature is not None and reduced_grib_reusable(reduced_path=reduced_path, signature=reuse_signature):
            diagnostics["reduced_reused"] = True
            with phase_gate(
                semaphore=phase_limits.reduce_semaphore,
                reporter=reporter,
                worker_id=worker_id,
                wait_phase="reduce_wait",
                active_phase="reduce",
                details="reuse_reduced",
            ):
                reduced_inventory = load_reduced_inventory(reduced_path=reduced_path, signature=reuse_signature)
                if reduced_inventory is None:
                    inventory_started_at = time.perf_counter()
                    reduced_inventory = inventory_for_grib(wgrib2_path, reduced_path)
                    inventory_seconds = time.perf_counter() - inventory_started_at
                else:
                    inventory_seconds = 0.0
                reduce_seconds = 0.0
        else:
            with phase_gate(
                semaphore=phase_limits.reduce_semaphore,
                reporter=reporter,
                worker_id=worker_id,
                wait_phase="reduce_wait",
                active_phase="reduce",
                details="reduce_grib2",
            ):
                crop_result = reduce_grib2_for_task(
                    wgrib2_path,
                    raw_path,
                    reduced_path,
                    task=task,
                    raw_manifest_path=raw_manifest_path,
                )
                reduced_inventory = crop_result.selected_lines
                inventory_seconds = crop_result.inventory_seconds
                reduce_seconds = crop_result.reduce_seconds
                diagnostics.update(
                    {
                        "crop_method": crop_result.method_used,
                        "crop_command": crop_result.command,
                        "crop_grid_cache_key": crop_result.crop_grid_cache_key,
                        "crop_grid_cache_hit": crop_result.crop_grid_cache_hit,
                        "crop_ij_box": crop_result.crop_ij_box,
                        "crop_wgrib2_threads": crop_result.crop_wgrib2_threads,
                        "crop_fallback_reason": crop_result.crop_fallback_reason,
                    }
                )
                if keep_reduced and reuse_signature is not None:
                    write_reduced_reuse_signature(reduced_path, reuse_signature)
                    write_reduced_inventory(reduced_path, reuse_signature, reduced_inventory)
        diagnostics["reduced_file_size"] = reduced_path.stat().st_size if reduced_path.exists() else None
        diagnostics["timing_wgrib_inventory_seconds"] = inventory_seconds
        diagnostics["timing_reduce_seconds"] = reduce_seconds
        cfgrib_parent = (scratch_dir / "cfgrib_index") if scratch_dir is not None else None
        if cfgrib_parent is not None:
            cfgrib_parent.mkdir(parents=True, exist_ok=True)
        cfgrib_index_dir = Path(
            tempfile.mkdtemp(
                prefix=f"hrrr_cfgrib_{task.forecast_hour:02d}_",
                dir=str(cfgrib_parent) if cfgrib_parent is not None else None,
            )
        )
        with phase_gate(
            semaphore=phase_limits.extract_semaphore,
            reporter=reporter,
            worker_id=worker_id,
            wait_phase="extract_wait",
            active_phase="open",
            details="process_reduced_grib",
        ):
            try:
                result = process_reduced_grib(
                    reduced_path,
                    reduced_inventory,
                    task,
                    grib_url,
                    cfgrib_index_dir=cfgrib_index_dir,
                    diagnostics=diagnostics,
                    include_legacy_aliases=include_legacy_aliases,
                    write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                )
            except TypeError as exc:
                if "cfgrib_index_dir" not in str(exc) and "diagnostics" not in str(exc):
                    raise
                result = process_reduced_grib(
                    reduced_path,
                    reduced_inventory,
                    task,
                    grib_url,
                    include_legacy_aliases=include_legacy_aliases,
                    write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                )
    except Exception as exc:
        diagnostics["last_error_type"] = type(exc).__name__
        diagnostics["last_error_message"] = str(exc)
        return TaskResult(False, task.key, None, [], [], str(exc), diagnostics)
    finally:
        cleanup_started_at = time.perf_counter()
        if reporter is not None:
            reporter.update_worker(worker_id, phase="finalize", details="cleanup")
        if cfgrib_index_dir is not None:
            shutil.rmtree(cfgrib_index_dir, ignore_errors=True)
        if reduced_path.exists() and not keep_reduced:
            reduced_path.unlink()
            reduced_signature_path(reduced_path).unlink(missing_ok=True)
            reduced_inventory_path(reduced_path).unlink(missing_ok=True)
        if raw_path.exists() and not keep_downloads:
            raw_path.unlink()
        if raw_manifest_path.exists() and not keep_downloads:
            raw_manifest_path.unlink()
        if raw_selection_manifest_path.exists() and not keep_downloads:
            raw_selection_manifest_path.unlink()
        diagnostics["timing_cleanup_seconds"] = time.perf_counter() - cleanup_started_at

    return result


def _report_hrrr_download_event(*, reporter, worker_id: str, event_name: str, payload: dict[str, object]) -> None:
    if reporter is None:
        return
    if event_name == "start":
        reporter.update_worker(worker_id, phase="download", details="download_subset")
        reporter.start_transfer(
            worker_id,
            file_label=str(payload.get("file_label") or "subset.grib2"),
            total_bytes=int(payload["total_bytes"]) if payload.get("total_bytes") is not None else None,
        )
        return
    if event_name == "progress":
        reporter.update_transfer(
            worker_id,
            bytes_downloaded=int(payload.get("downloaded_bytes") or 0),
            total_bytes=int(payload["total_bytes"]) if payload.get("total_bytes") is not None else None,
        )
        return
    if event_name == "complete":
        reporter.finish_transfer(worker_id)


def process_task(
    task: TaskSpec,
    *,
    wgrib2_path: str,
    download_dir: Path,
    reduced_dir: Path,
    phase_limits: PhaseConcurrencyLimits | None = None,
    keep_downloads: bool,
    keep_reduced: bool,
    include_legacy_aliases: bool,
    range_merge_gap_bytes: int = DEFAULT_RANGE_MERGE_GAP_BYTES,
    scratch_dir: Path | None = None,
    overwrite: bool = False,
    reporter=None,
    max_attempts: int = 6,
    retry_backoff_seconds: float = 2.0,
    retry_max_backoff_seconds: float = 30.0,
) -> TaskResult:
    if phase_limits is None:
        phase_limits = build_phase_concurrency_limits(
            max_workers=1,
            args=argparse.Namespace(download_workers=None, reduce_workers=None, extract_workers=None),
        )
    retry_policy = RetryPolicy(
        max_attempts=max(1, int(max_attempts)),
        backoff_seconds=float(retry_backoff_seconds),
        max_backoff_seconds=float(retry_max_backoff_seconds),
    )
    worker_id = threading.current_thread().name
    worker_label = f"{task.target_date_local} c{task.cycle_hour_utc:02d} f{task.forecast_hour:02d}"
    last_result: TaskResult | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        if reporter is not None and attempt > 1:
            reporter.start_retry(worker_id, attempt=attempt, max_attempts=retry_policy.max_attempts)
        result = _process_task_once(
            task,
            wgrib2_path=wgrib2_path,
            download_dir=download_dir,
            reduced_dir=reduced_dir,
            phase_limits=phase_limits,
            keep_downloads=keep_downloads,
            keep_reduced=keep_reduced,
            include_legacy_aliases=include_legacy_aliases,
            range_merge_gap_bytes=range_merge_gap_bytes,
            scratch_dir=scratch_dir,
            overwrite=overwrite,
            reporter=reporter,
            attempt=attempt,
            max_attempts=retry_policy.max_attempts,
        )
        last_result = result
        result.diagnostics["attempt_count"] = attempt
        result.diagnostics["retried"] = attempt > 1

        if result.ok and result.row is not None:
            result.diagnostics["retry_recovered"] = attempt > 1
            result.diagnostics["final_error_class"] = None
            if reporter is not None:
                if attempt > 1:
                    reporter.recover_worker(worker_id, message=f"{worker_label} recovered a{attempt}/{retry_policy.max_attempts}")
                reporter.complete_worker(worker_id, message=f"{worker_label} ok")
            return result

        current_phase = None
        if reporter is not None:
            worker = reporter.state.workers.get(worker_id)
            current_phase = worker.phase if worker is not None else None
        message = result.message or "unknown error"
        result.diagnostics["last_error_message"] = message
        decision = classify_task_failure(
            exception_type=str(result.diagnostics.get("last_error_type") or ""),
            message=message,
            phase=current_phase,
        )
        result.diagnostics["final_error_class"] = decision.error_class
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
            remaining = max(0.0, float(delay_seconds))
            while remaining > 0:
                sleep_seconds = min(0.2, remaining)
                time.sleep(sleep_seconds)
                remaining -= sleep_seconds
                if reporter is not None:
                    reporter.refresh(force=True)
            continue

        result.diagnostics["retry_recovered"] = False
        if reporter is not None:
            reporter.fail_worker(worker_id, message=f"{worker_label} {message}")
        return result

    assert last_result is not None
    return last_result


def manifest_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.manifest.json"


def manifest_parquet_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.manifest.parquet"


def parquet_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.parquet"


def provenance_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.provenance.parquet"


def row_buffer_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.rows.jsonl"


def provenance_buffer_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.provenance.rows.jsonl"


def summary_parquet_path(output_dir: Path, month_id: str) -> Path:
    return output_dir / f"{month_id}.parquet"


def new_manifest(path: Path, month_id: str, expected_task_keys: list[str], *, keep_downloads: bool, keep_reduced: bool) -> dict[str, object]:
    return {
        "month": month_id,
        "expected_task_count": len(expected_task_keys),
        "expected_task_keys": expected_task_keys,
        "completed_task_keys": [],
        "failure_reasons": {},
        "missing_fields": {},
        "source_model": DEFAULT_SOURCE_MODEL,
        "source_product": DEFAULT_SOURCE_PRODUCT,
        "source_version": DEFAULT_SOURCE_VERSION,
        "wide_parquet_path": str(parquet_path(path.parent, month_id)),
        "provenance_path": str(provenance_path(path.parent, month_id)),
        "summary_parquet_path": None,
        "provenance_written": not RUNTIME_OPTIONS.skip_provenance,
        "extract_method": RUNTIME_OPTIONS.extract_method,
        "summary_profile": RUNTIME_OPTIONS.summary_profile,
        "manifest_parquet_path": str(manifest_parquet_path(path.parent, month_id)),
        "manifest_json_path": str(path),
        "row_buffer_path": str(row_buffer_path(path.parent, month_id)),
        "provenance_buffer_path": str(provenance_buffer_path(path.parent, month_id)),
        "keep_downloads": keep_downloads,
        "keep_reduced": keep_reduced,
        "task_diagnostics": {},
        "complete": False,
    }


def manifest_matches_current_run(manifest: dict[str, object], expected_task_keys: list[str]) -> bool:
    return (
        list(manifest.get("expected_task_keys", [])) == list(expected_task_keys)
        and str(manifest.get("extract_method", DEFAULT_EXTRACT_METHOD)) == RUNTIME_OPTIONS.extract_method
        and str(manifest.get("summary_profile", "full")) == RUNTIME_OPTIONS.summary_profile
        and bool(manifest.get("provenance_written", True)) == (not RUNTIME_OPTIONS.skip_provenance)
    )


def load_manifest(path: Path, month_id: str, expected_task_keys: list[str], *, keep_downloads: bool, keep_reduced: bool) -> dict[str, object]:
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.setdefault("source_model", DEFAULT_SOURCE_MODEL)
        manifest.setdefault("source_product", DEFAULT_SOURCE_PRODUCT)
        manifest.setdefault("source_version", DEFAULT_SOURCE_VERSION)
        manifest.setdefault("summary_parquet_path", None)
        manifest.setdefault("provenance_written", True)
        manifest.setdefault("extract_method", DEFAULT_EXTRACT_METHOD)
        manifest.setdefault("summary_profile", "full")
        manifest.setdefault("manifest_parquet_path", str(manifest_parquet_path(path.parent, month_id)))
        manifest.setdefault("manifest_json_path", str(path))
        manifest.setdefault("task_diagnostics", {})
        if manifest_matches_current_run(manifest, expected_task_keys):
            return manifest
        row_buffer_path(path.parent, month_id).unlink(missing_ok=True)
        provenance_buffer_path(path.parent, month_id).unlink(missing_ok=True)
    return new_manifest(
        path,
        month_id,
        expected_task_keys,
        keep_downloads=keep_downloads,
        keep_reduced=keep_reduced,
    )


def save_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["manifest_json_path"] = str(path)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_row_buffer(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[str(row["task_key"])] = row
    return rows


def load_provenance_buffer(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    if not path.exists():
        return rows
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[f"{row['task_key']}|{row['feature_name']}"] = row
    return rows


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def append_jsonl_batch(path: Path, payloads: Iterable[dict[str, object]]) -> None:
    rows = [json.dumps(payload, sort_keys=True) + "\n" for payload in payloads]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.writelines(rows)


def manifest_records(month_id: str, manifest: dict[str, object]) -> list[dict[str, object]]:
    completed = set(manifest.get("completed_task_keys", []))
    failures = dict(manifest.get("failure_reasons", {}))
    missing_fields = dict(manifest.get("missing_fields", {}))
    task_diagnostics = dict(manifest.get("task_diagnostics", {}))
    records: list[dict[str, object]] = []
    for task_key in manifest.get("expected_task_keys", []):
        diagnostics = dict(task_diagnostics.get(task_key, {}))
        records.append(
            {
                "month": month_id,
                "task_key": task_key,
                "status": "ok" if task_key in completed else ("error" if task_key in failures else "pending"),
                "failure_reason": failures.get(task_key),
                "missing_fields": json_list(sorted(missing_fields.get(task_key, []))),
                "source_model": manifest.get("source_model", DEFAULT_SOURCE_MODEL),
                "source_product": manifest.get("source_product", DEFAULT_SOURCE_PRODUCT),
                "source_version": manifest.get("source_version", DEFAULT_SOURCE_VERSION),
                "wide_parquet_path": manifest.get("wide_parquet_path"),
                "provenance_path": manifest.get("provenance_path"),
                "summary_parquet_path": manifest.get("summary_parquet_path"),
                "manifest_json_path": manifest.get("manifest_json_path", manifest.get("manifest_path")),
                "keep_downloads": bool(manifest.get("keep_downloads", False)),
                "keep_reduced": bool(manifest.get("keep_reduced", False)),
                "complete": bool(manifest.get("complete", False)),
                **diagnostics,
            }
        )
    return records


def write_manifest_parquet(output_dir: Path, month_id: str, manifest: dict[str, object]) -> None:
    path = manifest_parquet_path(output_dir, month_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(manifest_records(month_id, manifest)).to_parquet(path, index=False)


def persist_month_manifest(
    *,
    output_dir: Path,
    month_id: str,
    manifest_file: Path,
    manifest: dict[str, object],
    completed: set[str],
    failures: dict[str, str],
    missing_fields: dict[str, list[str]],
    task_diagnostics: dict[str, dict[str, object]],
) -> None:
    manifest["completed_task_keys"] = sorted(completed)
    manifest["failure_reasons"] = failures
    manifest["missing_fields"] = missing_fields
    manifest["task_diagnostics"] = task_diagnostics
    save_manifest(manifest_file, manifest)
    write_manifest_parquet(output_dir, month_id, manifest)


def cycle_group_key(row: dict[str, object]) -> tuple[str, int]:
    return str(row["run_date_utc"]), int(row["cycle_hour_utc"])


def cycle_group_slots(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["valid_time_local"]) for row in rows}


def coerce_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return numeric


def hour_value(rows_by_hour: dict[int, dict[str, object]], hour: int, column: str) -> float | None:
    row = rows_by_hour.get(hour)
    if row is None:
        return None
    return coerce_float(row.get(column))


def rows_in_half_open_local_hour_range(
    rows: list[dict[str, object]],
    *,
    start_hour: int,
    end_hour: int,
) -> list[dict[str, object]]:
    return [row for row in rows if start_hour <= int(row["valid_hour_local"]) < end_hour]


def summarize_hourly_rows(rows: list[dict[str, object]], column: str, reducer: str) -> float | None:
    values = [coerce_float(row.get(column)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    if reducer == "max":
        return max(finite)
    if reducer == "min":
        return min(finite)
    if reducer == "sum":
        return float(sum(finite))
    if reducer == "mean":
        return float(sum(finite) / len(finite))
    raise ValueError(f"Unsupported reducer {reducer}")


def wind_speed_from_uv(row: dict[str, object], u_key: str, v_key: str) -> float | None:
    u = coerce_float(row.get(u_key))
    v = coerce_float(row.get(v_key))
    if u is None or v is None:
        return None
    return float(np.hypot(u, v))


def wind_direction_from_uv(row: dict[str, object], u_key: str, v_key: str) -> float | None:
    u = coerce_float(row.get(u_key))
    v = coerce_float(row.get(v_key))
    if u is None or v is None:
        return None
    speed = float(np.hypot(u, v))
    if speed <= 0.0:
        return None
    return float((270.0 - np.degrees(np.arctan2(v, u))) % 360.0)


def summarize_bool_any(rows: list[dict[str, object]], column: str) -> bool | None:
    values = [coerce_float(row.get(column)) for row in rows]
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return any(value > 0.0 for value in finite)


def build_revision_features(
    anchor_summary: dict[str, object],
    lag_summaries: dict[int, dict[str, object]],
) -> dict[str, float | None]:
    revision_bases = {
        "hrrr_temp_2m_day_max_k",
        "hrrr_temp_2m_09_local_k",
        "hrrr_temp_2m_12_local_k",
        "hrrr_temp_2m_15_local_k",
        "hrrr_tcdc_day_mean_pct",
        "hrrr_dswrf_day_max_w_m2",
        "hrrr_pwat_day_mean_kg_m2",
        "hrrr_hpbl_day_max_m",
        "hrrr_mslp_day_mean_pa",
    }
    features: dict[str, float | None] = {}
    for lag, lag_summary in lag_summaries.items():
        suffix = f"_rev_{lag}cycle"
        for base_name in revision_bases:
            anchor_value = coerce_float(anchor_summary.get(base_name))
            lag_value = coerce_float(lag_summary.get(base_name))
            if anchor_value is None or lag_value is None:
                features[f"{base_name}{suffix}"] = None
            else:
                features[f"{base_name}{suffix}"] = float(anchor_value - lag_value)
    return features


def cycle_summary_features(rows: list[dict[str, object]]) -> dict[str, object]:
    rows_sorted = sorted(rows, key=lambda item: str(item["valid_time_utc"]))
    rows_by_hour = {int(row["valid_hour_local"]): row for row in rows_sorted}
    morning_rows = rows_in_half_open_local_hour_range(rows_sorted, start_hour=6, end_hour=12)
    afternoon_rows = rows_in_half_open_local_hour_range(rows_sorted, start_hour=12, end_hour=18)
    night_rows = [row for row in rows_sorted if int(row["valid_hour_local"]) <= 6]

    summary = {
        "hrrr_temp_2m_06_local_k": hour_value(rows_by_hour, 6, "tmp_2m_k"),
        "hrrr_temp_2m_09_local_k": hour_value(rows_by_hour, 9, "tmp_2m_k"),
        "hrrr_temp_2m_12_local_k": hour_value(rows_by_hour, 12, "tmp_2m_k"),
        "hrrr_temp_2m_15_local_k": hour_value(rows_by_hour, 15, "tmp_2m_k"),
        "hrrr_temp_2m_18_local_k": hour_value(rows_by_hour, 18, "tmp_2m_k"),
        "hrrr_dewpoint_2m_06_local_k": hour_value(rows_by_hour, 6, "dpt_2m_k"),
        "hrrr_dewpoint_2m_09_local_k": hour_value(rows_by_hour, 9, "dpt_2m_k"),
        "hrrr_dewpoint_2m_12_local_k": hour_value(rows_by_hour, 12, "dpt_2m_k"),
        "hrrr_dewpoint_2m_15_local_k": hour_value(rows_by_hour, 15, "dpt_2m_k"),
        "hrrr_dewpoint_2m_18_local_k": hour_value(rows_by_hour, 18, "dpt_2m_k"),
        "hrrr_rh_2m_06_local_pct": hour_value(rows_by_hour, 6, "rh_2m_pct"),
        "hrrr_rh_2m_09_local_pct": hour_value(rows_by_hour, 9, "rh_2m_pct"),
        "hrrr_rh_2m_12_local_pct": hour_value(rows_by_hour, 12, "rh_2m_pct"),
        "hrrr_rh_2m_15_local_pct": hour_value(rows_by_hour, 15, "rh_2m_pct"),
        "hrrr_rh_2m_18_local_pct": hour_value(rows_by_hour, 18, "rh_2m_pct"),
        "hrrr_u10m_09_local_ms": hour_value(rows_by_hour, 9, "ugrd_10m_ms"),
        "hrrr_u10m_12_local_ms": hour_value(rows_by_hour, 12, "ugrd_10m_ms"),
        "hrrr_u10m_15_local_ms": hour_value(rows_by_hour, 15, "ugrd_10m_ms"),
        "hrrr_u10m_18_local_ms": hour_value(rows_by_hour, 18, "ugrd_10m_ms"),
        "hrrr_v10m_09_local_ms": hour_value(rows_by_hour, 9, "vgrd_10m_ms"),
        "hrrr_v10m_12_local_ms": hour_value(rows_by_hour, 12, "vgrd_10m_ms"),
        "hrrr_v10m_15_local_ms": hour_value(rows_by_hour, 15, "vgrd_10m_ms"),
        "hrrr_v10m_18_local_ms": hour_value(rows_by_hour, 18, "vgrd_10m_ms"),
        "hrrr_wind_10m_09_local_speed_ms": hour_value(rows_by_hour, 9, "wind_10m_speed_ms"),
        "hrrr_wind_10m_12_local_speed_ms": hour_value(rows_by_hour, 12, "wind_10m_speed_ms"),
        "hrrr_wind_10m_15_local_speed_ms": hour_value(rows_by_hour, 15, "wind_10m_speed_ms"),
        "hrrr_wind_10m_18_local_speed_ms": hour_value(rows_by_hour, 18, "wind_10m_speed_ms"),
        "hrrr_wind_10m_09_local_direction_deg": wind_direction_from_uv(rows_by_hour.get(9, {}), "ugrd_10m_ms", "vgrd_10m_ms"),
        "hrrr_wind_10m_12_local_direction_deg": wind_direction_from_uv(rows_by_hour.get(12, {}), "ugrd_10m_ms", "vgrd_10m_ms"),
        "hrrr_wind_10m_15_local_direction_deg": wind_direction_from_uv(rows_by_hour.get(15, {}), "ugrd_10m_ms", "vgrd_10m_ms"),
        "hrrr_wind_10m_18_local_direction_deg": wind_direction_from_uv(rows_by_hour.get(18, {}), "ugrd_10m_ms", "vgrd_10m_ms"),
        "hrrr_mslp_09_local_pa": hour_value(rows_by_hour, 9, "mslma_pa"),
        "hrrr_mslp_12_local_pa": hour_value(rows_by_hour, 12, "mslma_pa"),
        "hrrr_mslp_15_local_pa": hour_value(rows_by_hour, 15, "mslma_pa"),
        "hrrr_surface_pressure_09_local_pa": hour_value(rows_by_hour, 9, "surface_pressure_pa"),
        "hrrr_tcdc_day_mean_pct": summarize_hourly_rows(rows_sorted, "tcdc_entire_pct", "mean"),
        "hrrr_tcdc_morning_mean_pct": summarize_hourly_rows(morning_rows, "tcdc_entire_pct", "mean"),
        "hrrr_tcdc_afternoon_mean_pct": summarize_hourly_rows(afternoon_rows, "tcdc_entire_pct", "mean"),
        "hrrr_tcdc_day_max_pct": summarize_hourly_rows(rows_sorted, "tcdc_entire_pct", "max"),
        "hrrr_mcdc_day_mean_pct": summarize_hourly_rows(rows_sorted, "mcdc_mid_pct", "mean"),
        "hrrr_hcdc_day_mean_pct": summarize_hourly_rows(rows_sorted, "hcdc_high_pct", "mean"),
        "hrrr_mcdc_afternoon_mean_pct": summarize_hourly_rows(afternoon_rows, "mcdc_mid_pct", "mean"),
        "hrrr_hcdc_afternoon_mean_pct": summarize_hourly_rows(afternoon_rows, "hcdc_high_pct", "mean"),
        "hrrr_dswrf_day_max_w_m2": summarize_hourly_rows(rows_sorted, "dswrf_surface_w_m2", "max"),
        "hrrr_pwat_day_mean_kg_m2": summarize_hourly_rows(rows_sorted, "pwat_entire_atmosphere_kg_m2", "mean"),
        "hrrr_hpbl_day_max_m": summarize_hourly_rows(rows_sorted, "hpbl_m", "max"),
        "hrrr_temp_2m_day_max_k": summarize_hourly_rows(rows_sorted, "tmp_2m_k", "max"),
        "hrrr_temp_2m_day_mean_k": summarize_hourly_rows(rows_sorted, "tmp_2m_k", "mean"),
        "hrrr_rh_2m_day_min_pct": summarize_hourly_rows(rows_sorted, "rh_2m_pct", "min"),
        "hrrr_wind_10m_day_max_ms": summarize_hourly_rows(rows_sorted, "wind_10m_speed_ms", "max"),
        "hrrr_gust_day_max_ms": summarize_hourly_rows(rows_sorted, "gust_surface_ms", "max"),
        "hrrr_lcdc_morning_mean_pct": summarize_hourly_rows(morning_rows, "lcdc_low_pct", "mean"),
        "hrrr_dlwrf_night_mean_w_m2": summarize_hourly_rows(night_rows, "dlwrf_surface_w_m2", "mean"),
        "hrrr_apcp_day_total_kg_m2": summarize_hourly_rows(rows_sorted, "apcp_surface_kg_m2", "sum"),
        "hrrr_cape_day_max_j_kg": summarize_hourly_rows(rows_sorted, "cape_surface_j_kg", "max"),
        "hrrr_cape_afternoon_max_j_kg": summarize_hourly_rows(afternoon_rows, "cape_surface_j_kg", "max"),
        "hrrr_cin_day_min_j_kg": summarize_hourly_rows(rows_sorted, "cin_surface_j_kg", "min"),
        "hrrr_cin_afternoon_min_j_kg": summarize_hourly_rows(afternoon_rows, "cin_surface_j_kg", "min"),
        "hrrr_refc_day_max": summarize_hourly_rows(rows_sorted, "refc_entire_atmosphere", "max"),
        "hrrr_ltng_day_max": summarize_hourly_rows(rows_sorted, "ltng_entire_atmosphere", "max"),
        "hrrr_ltng_day_any": summarize_bool_any(rows_sorted, "ltng_entire_atmosphere"),
        "hrrr_mslp_day_mean_pa": summarize_hourly_rows(rows_sorted, "mslma_pa", "mean"),
        "hrrr_temp_1000mb_day_mean_k": summarize_hourly_rows(rows_sorted, "tmp_1000mb_k", "mean"),
        "hrrr_temp_925mb_day_mean_k": summarize_hourly_rows(rows_sorted, "tmp_925mb_k", "mean"),
        "hrrr_temp_850mb_day_mean_k": summarize_hourly_rows(rows_sorted, "tmp_850mb_k", "mean"),
        "hrrr_rh_925mb_day_mean_pct": summarize_hourly_rows(rows_sorted, "rh_925mb_pct", "mean"),
        "hrrr_temp_2m_k_nb3_day_mean": summarize_hourly_rows(rows_sorted, "tmp_2m_k_nb3_mean", "mean"),
        "hrrr_temp_2m_k_crop_day_mean": summarize_hourly_rows(rows_sorted, "tmp_2m_k_crop_mean", "mean"),
        "hrrr_tcdc_entire_pct_crop_day_mean": summarize_hourly_rows(rows_sorted, "tcdc_entire_pct_crop_mean", "mean"),
        "hrrr_dswrf_surface_w_m2_crop_day_max": summarize_hourly_rows(rows_sorted, "dswrf_surface_w_m2_crop_max", "max"),
        "hrrr_pwat_entire_atmosphere_kg_m2_nb7_day_mean": summarize_hourly_rows(rows_sorted, "pwat_entire_atmosphere_kg_m2_nb7_mean", "mean"),
        "hrrr_u925_day_mean_ms": summarize_hourly_rows(rows_sorted, "ugrd_925mb_ms", "mean"),
        "hrrr_v925_day_mean_ms": summarize_hourly_rows(rows_sorted, "vgrd_925mb_ms", "mean"),
        "hrrr_u850_day_mean_ms": summarize_hourly_rows(rows_sorted, "ugrd_850mb_ms", "mean"),
        "hrrr_v850_day_mean_ms": summarize_hourly_rows(rows_sorted, "vgrd_850mb_ms", "mean"),
        "hrrr_hgt_925_day_mean_gpm": summarize_hourly_rows(rows_sorted, "hgt_925mb_gpm", "mean"),
        "hrrr_hgt_700_day_mean_gpm": summarize_hourly_rows(rows_sorted, "hgt_700mb_gpm", "mean"),
    }

    wind_850_values = [wind_speed_from_uv(row, "ugrd_850mb_ms", "vgrd_850mb_ms") for row in rows_sorted]
    wind_850_values = [value for value in wind_850_values if value is not None]
    summary["hrrr_wind_850mb_speed_day_mean_ms"] = (
        float(sum(wind_850_values) / len(wind_850_values)) if wind_850_values else None
    )
    return summary


def build_summary_row(target_date_local: str, rows: list[dict[str, object]]) -> dict[str, object]:
    expected_slots = target_day_expected_slots(pd.Timestamp(target_date_local))
    checkpoint_hours = {6, 9, 12, 15, 18, 21}
    retained_task_keys = sorted(str(row["task_key"]) for row in rows)
    cycle_groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    cycle_rank: dict[tuple[str, int], int] = {}
    for row in rows:
        key = cycle_group_key(row)
        cycle_groups.setdefault(key, []).append(row)
        cycle_rank[key] = int(row["cycle_rank_desc"])
    cycle_slots = {key: cycle_group_slots(cycle_rows) for key, cycle_rows in cycle_groups.items()}
    anchor_key, has_full_coverage = choose_anchor_cycle(cycle_slots, cycle_rank, expected_slots)
    anchor_rows = sorted(cycle_groups[anchor_key], key=lambda item: str(item["valid_time_utc"]))
    coverage_hours = {int(row["valid_hour_local"]) for row in anchor_rows}
    covered_checkpoint_count = len(coverage_hours & checkpoint_hours)
    anchor_summary = cycle_summary_features(anchor_rows)

    lag_summaries: dict[int, dict[str, object]] = {}
    non_anchor_keys = sorted(
        (key for key in cycle_groups if key != anchor_key),
        key=lambda key: cycle_rank[key],
    )[:OVERNIGHT_REVISION_CYCLE_COUNT]
    for lag, lag_key in enumerate(non_anchor_keys, start=1):
        lag_summaries[lag] = cycle_summary_features(cycle_groups[lag_key])

    summary_row: dict[str, object] = {
        "target_date_local": target_date_local,
        "anchor_run_date_utc": anchor_key[0],
        "anchor_cycle_hour_utc": anchor_key[1],
        "anchor_init_time_utc": anchor_rows[0]["init_time_utc"],
        "anchor_init_time_local": anchor_rows[0]["init_time_local"],
        "retained_cycle_count": len(cycle_groups),
        "first_valid_hour_local": min(coverage_hours) if coverage_hours else None,
        "last_valid_hour_local": max(coverage_hours) if coverage_hours else None,
        "covered_hour_count": len(coverage_hours),
        "covered_checkpoint_count": covered_checkpoint_count,
        "retained_task_keys_json": json.dumps(retained_task_keys),
        "coverage_end_hour_local": max(coverage_hours) if coverage_hours else None,
        "has_full_day_21_local_coverage": has_full_coverage,
        "missing_checkpoint_count": sum(
            1
            for checkpoint in (
                "hrrr_temp_2m_06_local_k",
                "hrrr_temp_2m_09_local_k",
                "hrrr_temp_2m_12_local_k",
                "hrrr_temp_2m_15_local_k",
                "hrrr_temp_2m_18_local_k",
                "hrrr_dewpoint_2m_06_local_k",
                "hrrr_dewpoint_2m_09_local_k",
                "hrrr_dewpoint_2m_12_local_k",
                "hrrr_dewpoint_2m_15_local_k",
                "hrrr_dewpoint_2m_18_local_k",
                "hrrr_rh_2m_06_local_pct",
                "hrrr_rh_2m_09_local_pct",
                "hrrr_rh_2m_12_local_pct",
                "hrrr_rh_2m_15_local_pct",
                "hrrr_rh_2m_18_local_pct",
            )
            if anchor_summary.get(checkpoint) is None
        ),
    }
    summary_row.update(anchor_summary)
    summary_row.update(build_revision_features(anchor_summary, lag_summaries))
    return summary_row


def write_summary_month(summary_output_dir: Path, month_id: str, row_buffer: dict[str, dict[str, object]]) -> None:
    summary_output_dir.mkdir(parents=True, exist_ok=True)
    by_target_date: dict[str, list[dict[str, object]]] = {}
    for row in row_buffer.values():
        by_target_date.setdefault(str(row["target_date_local"]), []).append(row)
    summary_rows = [build_summary_row(target_date_local, rows) for target_date_local, rows in sorted(by_target_date.items())]
    pd.DataFrame.from_records(summary_rows).to_parquet(summary_parquet_path(summary_output_dir, month_id), index=False)


def finalize_month(
    output_dir: Path,
    summary_output_dir: Path,
    month_id: str,
    row_buffer: dict[str, dict[str, object]],
    provenance_buffer: dict[str, dict[str, object]],
    manifest: dict[str, object],
    write_provenance: bool = True,
) -> None:
    wide_df = pd.DataFrame(sorted(row_buffer.values(), key=lambda row: row["task_key"]))
    wide_df.to_parquet(parquet_path(output_dir, month_id), index=False)
    if write_provenance:
        provenance_df = pd.DataFrame(
            sorted(
                provenance_buffer.values(),
                key=lambda row: (row["task_key"], row["feature_name"]),
            )
        )
        provenance_df.to_parquet(provenance_path(output_dir, month_id), index=False)
        manifest["provenance_path"] = str(provenance_path(output_dir, month_id))
    else:
        provenance_path(output_dir, month_id).unlink(missing_ok=True)
        manifest["provenance_path"] = None
    write_summary_month(summary_output_dir, month_id, row_buffer)
    manifest["summary_parquet_path"] = str(summary_parquet_path(summary_output_dir, month_id))
    manifest["provenance_written"] = bool(write_provenance)
    manifest["extract_method"] = RUNTIME_OPTIONS.extract_method
    manifest["summary_profile"] = RUNTIME_OPTIONS.summary_profile
    manifest["complete"] = True
    manifest["failure_reasons"] = {}
    save_manifest(manifest_path(output_dir, month_id), manifest)
    write_manifest_parquet(output_dir, month_id, manifest)
    row_buffer_path(output_dir, month_id).unlink(missing_ok=True)
    provenance_buffer_path(output_dir, month_id).unlink(missing_ok=True)


def month_is_complete(output_dir: Path, month_id: str, manifest: dict[str, object], expected_task_keys: list[str]) -> bool:
    provenance_ok = (
        not bool(manifest.get("provenance_written", True))
        or provenance_path(output_dir, month_id).exists()
    )
    return (
        bool(manifest.get("complete"))
        and manifest_matches_current_run(manifest, expected_task_keys)
        and parquet_path(output_dir, month_id).exists()
        and provenance_ok
        and manifest.get("summary_parquet_path")
        and Path(str(manifest["summary_parquet_path"])).exists()
        and manifest_parquet_path(output_dir, month_id).exists()
    )


_ORIGINAL_PROCESS_TASK = process_task


def _run_month_legacy(
    month_id: str,
    tasks: list[TaskSpec],
    *,
    wgrib2_path: str,
    output_dir: Path,
    summary_output_dir: Path,
    download_dir: Path,
    reduced_dir: Path,
    max_workers: int,
    keep_downloads: bool,
    keep_reduced: bool,
    download_workers: int | None = None,
    reduce_workers: int | None = None,
    extract_workers: int | None = None,
    reduce_queue_size: int | None = None,
    extract_queue_size: int | None = None,
    range_merge_gap_bytes: int = DEFAULT_RANGE_MERGE_GAP_BYTES,
    batch_reduce_mode: str = "off",
    include_legacy_aliases: bool = False,
    scratch_dir: Path | None = None,
    progress_mode: str = "auto",
    max_task_attempts: int = 6,
    retry_backoff_seconds: float = 2.0,
    retry_max_backoff_seconds: float = 30.0,
    enable_dashboard_hotkeys: bool = True,
    pause_control_file: str | None = None,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_task_keys = [task.key for task in tasks]
    manifest_file = manifest_path(output_dir, month_id)
    manifest = load_manifest(
        manifest_file,
        month_id,
        expected_task_keys,
        keep_downloads=keep_downloads,
        keep_reduced=keep_reduced,
    )

    if month_is_complete(output_dir, month_id, manifest, expected_task_keys):
        print(f"[skip-month] {month_id} already complete")
        return 0, 0

    row_buffer_file = row_buffer_path(output_dir, month_id)
    provenance_buffer_file = provenance_buffer_path(output_dir, month_id)
    completed = set(manifest.get("completed_task_keys", []))
    pending = [task for task in tasks if task.key not in completed]
    phase_limits = build_phase_concurrency_limits(
        max_workers=max_workers,
        args=argparse.Namespace(
            download_workers=download_workers,
            reduce_workers=reduce_workers,
            extract_workers=extract_workers,
        ),
    )
    row_buffer = load_row_buffer(row_buffer_file)
    provenance_buffer = load_provenance_buffer(provenance_buffer_file)
    failures = dict(manifest.get("failure_reasons", {}))
    missing_fields = dict(manifest.get("missing_fields", {}))
    task_diagnostics = dict(manifest.get("task_diagnostics", {}))

    wrote = 0
    failed = 0
    run_control = RunControl()
    reporter = (
        create_progress_reporter(
            f"HRRR {month_id}",
            unit="task",
            total=len(pending),
            mode=progress_mode,
            stream=sys.stdout,
            on_pause_request=lambda **kwargs: run_control.request_pause(reason=str(kwargs.get("reason", "operator"))),
            enable_dashboard_hotkeys=enable_dashboard_hotkeys,
            pause_control_file=pause_control_file,
        )
        if pending
        else None
    )
    if reporter is not None:
        retained_cycles = len({(task.run_date_utc, task.cycle_hour_utc) for task in pending})
        reporter.set_metrics(
            month=month_id,
            retained_cycles=retained_cycles,
            max_workers=max_workers,
            download_workers=phase_limits.download_workers,
            reduce_workers=phase_limits.reduce_workers,
            extract_workers=phase_limits.extract_workers,
        )
        reporter.upsert_group(month_id, label=month_id, total=len(pending), completed=0, failed=0, status="queued")

    try:
        if reporter is not None:
            reporter.log_event(f"{month_id} pending_tasks={len(pending)}")
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            pending_futures: dict[object, TaskSpec] = {}
            task_iter = iter(pending)

            for _ in range(max(1, max_workers)):
                if run_control.pause_requested():
                    break
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                if reporter is not None:
                    reporter.upsert_group(month_id, completed=wrote, failed=failed, status=f"submit fh={task.forecast_hour:02d}")
                future = executor.submit(
                    process_task,
                    task,
                    wgrib2_path=wgrib2_path,
                    download_dir=download_dir,
                    reduced_dir=reduced_dir,
                    phase_limits=phase_limits,
                    keep_downloads=keep_downloads,
                    keep_reduced=keep_reduced,
                    include_legacy_aliases=include_legacy_aliases,
                    range_merge_gap_bytes=range_merge_gap_bytes,
                    scratch_dir=scratch_dir,
                    overwrite=False,
                    reporter=reporter,
                    max_attempts=max_task_attempts,
                    retry_backoff_seconds=retry_backoff_seconds,
                    retry_max_backoff_seconds=retry_max_backoff_seconds,
                )
                pending_futures[future] = task

            while pending_futures:
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending_futures.pop(future)
                    if reporter is not None:
                        reporter.upsert_group(month_id, completed=wrote, failed=failed, status=f"active fh={task.forecast_hour:02d}")
                    result = future.result()
                    task_diagnostics[result.task_key] = result.diagnostics
                    if result.ok and result.row is not None:
                        append_jsonl_batch(row_buffer_file, [result.row])
                        row_buffer[result.task_key] = result.row
                        if result.provenance_rows:
                            append_jsonl_batch(provenance_buffer_file, result.provenance_rows)
                            for provenance_entry in result.provenance_rows:
                                provenance_buffer[f"{provenance_entry['task_key']}|{provenance_entry['feature_name']}"] = provenance_entry
                        completed.add(result.task_key)
                        failures.pop(result.task_key, None)
                        missing_fields[result.task_key] = result.missing_fields
                        wrote += 1
                    else:
                        failures[result.task_key] = result.message or "unknown error"
                        failed += 1

                    if reporter is not None:
                        reporter.upsert_group(
                            month_id,
                            completed=wrote,
                            failed=failed,
                            status=(
                                f"checkpoint fh={task.forecast_hour:02d} "
                                f"result={'ok' if result.ok and result.row is not None else 'error'}"
                            ),
                        )

                    if not run_control.pause_requested():
                        try:
                            next_task = next(task_iter)
                            if reporter is not None:
                                reporter.upsert_group(month_id, completed=wrote, failed=failed, status=f"submit fh={next_task.forecast_hour:02d}")
                            next_future = executor.submit(
                                process_task,
                                next_task,
                                wgrib2_path=wgrib2_path,
                                download_dir=download_dir,
                                reduced_dir=reduced_dir,
                                phase_limits=phase_limits,
                                keep_downloads=keep_downloads,
                                keep_reduced=keep_reduced,
                                include_legacy_aliases=include_legacy_aliases,
                                range_merge_gap_bytes=range_merge_gap_bytes,
                                scratch_dir=scratch_dir,
                                overwrite=False,
                                reporter=reporter,
                                max_attempts=max_task_attempts,
                                retry_backoff_seconds=retry_backoff_seconds,
                                retry_max_backoff_seconds=retry_max_backoff_seconds,
                            )
                            pending_futures[next_future] = next_task
                        except StopIteration:
                            pass
    except BaseException:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        raise
    finally:
        if reporter is not None:
            reporter.upsert_group(month_id, completed=wrote, failed=failed, status="finalize")
            if run_control.pause_requested():
                run_control.mark_paused(reason=run_control.pause_reason or "operator")
                reporter.mark_paused(reason=run_control.pause_reason or "operator")
            reporter.close(status="paused" if run_control.is_paused() else f"month={month_id} wrote={wrote} failed={failed}")

    if len(completed) == len(expected_task_keys):
        manifest["complete"] = True
        manifest["completed_task_keys"] = sorted(completed)
        manifest["failure_reasons"] = {}
        manifest["missing_fields"] = missing_fields
        manifest["task_diagnostics"] = task_diagnostics
        finalize_month(
            output_dir,
            summary_output_dir,
            month_id,
            row_buffer,
            provenance_buffer,
            manifest,
            write_provenance=not RUNTIME_OPTIONS.skip_provenance,
        )
        provenance_label = manifest.get("provenance_path") or "skipped"
        print(
            f"[month-done] {month_id} rows={len(row_buffer)} parquet={parquet_path(output_dir, month_id)} "
            f"provenance={provenance_label} summary={summary_parquet_path(summary_output_dir, month_id)}"
        )
    else:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        print(
            f"[month-incomplete] {month_id} completed={len(completed)} expected={len(expected_task_keys)} "
            f"failures={len(failures)}",
            file=sys.stderr,
        )

    return wrote, failed


def _run_month_batch_cycle(
    month_id: str,
    tasks: list[TaskSpec],
    *,
    wgrib2_path: str,
    output_dir: Path,
    summary_output_dir: Path,
    download_dir: Path,
    reduced_dir: Path,
    max_workers: int,
    keep_downloads: bool,
    keep_reduced: bool,
    download_workers: int | None = None,
    reduce_workers: int | None = None,
    extract_workers: int | None = None,
    reduce_queue_size: int | None = None,
    extract_queue_size: int | None = None,
    range_merge_gap_bytes: int = DEFAULT_RANGE_MERGE_GAP_BYTES,
    include_legacy_aliases: bool = False,
    scratch_dir: Path | None = None,
    progress_mode: str = "auto",
    max_task_attempts: int = 6,
    retry_backoff_seconds: float = 2.0,
    retry_max_backoff_seconds: float = 30.0,
    run_control: RunControl | None = None,
    enable_dashboard_hotkeys: bool = True,
    pause_control_file: str | None = None,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_task_keys = [task.key for task in tasks]
    manifest_file = manifest_path(output_dir, month_id)
    manifest = load_manifest(
        manifest_file,
        month_id,
        expected_task_keys,
        keep_downloads=keep_downloads,
        keep_reduced=keep_reduced,
    )
    manifest["batch_reduce_mode"] = "cycle"
    if month_is_complete(output_dir, month_id, manifest, expected_task_keys):
        print(f"[skip-month] {month_id} already complete")
        return 0, 0

    row_buffer_file = row_buffer_path(output_dir, month_id)
    provenance_buffer_file = provenance_buffer_path(output_dir, month_id)
    completed = set(manifest.get("completed_task_keys", []))
    pending = [task for task in tasks if task.key not in completed]
    row_buffer = load_row_buffer(row_buffer_file)
    provenance_buffer = load_provenance_buffer(provenance_buffer_file)
    failures = dict(manifest.get("failure_reasons", {}))
    missing_fields = dict(manifest.get("missing_fields", {}))
    task_diagnostics = dict(manifest.get("task_diagnostics", {}))
    run_control = run_control or RunControl()
    reporter = (
        create_progress_reporter(
            f"HRRR {month_id}",
            unit="task",
            total=len(pending),
            mode=progress_mode,
            stream=sys.stdout,
            on_pause_request=lambda **kwargs: run_control.request_pause(reason=str(kwargs.get("reason", "operator"))),
            enable_dashboard_hotkeys=enable_dashboard_hotkeys,
            pause_control_file=pause_control_file,
        )
        if pending
        else None
    )

    wrote = 0
    failed = 0
    retry_policy = RetryPolicy(
        max_attempts=max(1, int(max_task_attempts)),
        backoff_seconds=float(retry_backoff_seconds),
        max_backoff_seconds=float(retry_max_backoff_seconds),
    )
    phase_limits = build_phase_concurrency_limits(
        max_workers=max_workers,
        args=argparse.Namespace(download_workers=download_workers, reduce_workers=reduce_workers, extract_workers=extract_workers),
    )
    reduce_queue_limit = resolve_pipeline_queue_size(reduce_queue_size, downstream_workers=phase_limits.reduce_workers)
    extract_queue_limit = resolve_pipeline_queue_size(extract_queue_size, downstream_workers=phase_limits.extract_workers)

    def sleep_for_retry(*, worker_id: str, attempt: int, phase: str, exc: Exception) -> bool:
        decision = classify_task_failure(exception_type=type(exc).__name__, message=str(exc), phase=phase)
        if not should_retry_attempt(attempt=attempt, policy=retry_policy, decision=decision):
            return False
        next_attempt = attempt + 1
        delay_seconds = compute_retry_delay_seconds(attempt=next_attempt, policy=retry_policy)
        if reporter is not None:
            reporter.schedule_retry(
                worker_id,
                attempt=next_attempt,
                max_attempts=retry_policy.max_attempts,
                delay_seconds=delay_seconds,
                message=str(exc),
                error_class=decision.error_class,
            )
        remaining = max(0.0, float(delay_seconds))
        while remaining > 0:
            sleep_seconds = min(0.2, remaining)
            time.sleep(sleep_seconds)
            remaining -= sleep_seconds
            if reporter is not None:
                reporter.refresh(force=True)
        return True

    cycle_tasks: dict[tuple[str, int], list[TaskSpec]] = {}
    for task in pending:
        cycle_tasks.setdefault(cycle_key_for_task(task), []).append(task)
    cycle_states: dict[tuple[str, int], dict[str, object]] = {}
    pending_downloads: list[tuple[tuple[str, int], HrrrPipelineItem]] = []
    for cycle_key, group_tasks in sorted(cycle_tasks.items()):
        group_tasks = sorted(group_tasks, key=lambda item: item.forecast_hour)
        first_task = group_tasks[0]
        if reporter is not None:
            reporter.upsert_group(
                f"{cycle_key[0]}T{cycle_key[1]:02d}",
                label=f"{cycle_key[0]} t{cycle_key[1]:02d}z",
                total=len(group_tasks),
                completed=0,
                failed=0,
                status="queued",
            )
            reporter.set_metrics(
                month=month_id,
                retained_cycles=len(cycle_tasks),
                max_workers=max_workers,
                download_workers=phase_limits.download_workers,
                reduce_workers=phase_limits.reduce_workers,
                extract_workers=phase_limits.extract_workers,
                batch_reduce_mode="cycle",
            )
        items = [HrrrPipelineItem(task=task, attempt_count=1) for task in group_tasks]
        for item in items:
            item.raw_path = path_for_raw(download_dir, item.task)
            item.raw_manifest_path = path_for_raw_manifest(download_dir, item.task)
            item.raw_selection_manifest_path = path_for_raw_selection_manifest(download_dir, item.task)
            item.reduced_path = path_for_reduced(reduced_dir, item.task)
            item.grib_url = task_remote_url(item.task, "google")
            item.diagnostics = default_task_diagnostics(item.task)
            item.diagnostics["scratch_dir"] = str(scratch_dir) if scratch_dir is not None else None
            item.diagnostics["raw_file_path"] = str(item.raw_path)
            item.diagnostics["raw_manifest_path"] = str(item.raw_manifest_path)
            item.diagnostics["raw_selection_manifest_path"] = str(item.raw_selection_manifest_path)
            item.diagnostics["reduced_file_path"] = str(item.reduced_path)
            item.diagnostics["grib_url"] = item.grib_url
            item.diagnostics["batch_reduce_mode"] = "cycle"
            pending_downloads.append((cycle_key, item))
        cycle_states[cycle_key] = {
            "first_task": first_task,
            "items": items,
            "remaining_downloads": len(items),
            "reduce_submitted": False,
            "extract_submitted": False,
            "done": False,
        }

    def download_cycle_item(cycle_key: tuple[str, int], item: HrrrPipelineItem) -> HrrrPipelineItem:
        worker_id = f"batch_download_{cycle_key[0]}T{cycle_key[1]:02d}_f{item.task.forecast_hour:02d}"
        try:
            if reporter is not None:
                reporter.start_worker(
                    worker_id,
                    label=f"{item.task.target_date_local} c{item.task.cycle_hour_utc:02d} f{item.task.forecast_hour:02d}",
                    phase="download",
                    group_id=f"{cycle_key[0]}T{cycle_key[1]:02d}",
                    details="fetch_subset",
                )
                reporter.set_worker_attempt(worker_id, attempt=1, max_attempts=retry_policy.max_attempts)
            attempt = 1
            while True:
                item.attempt_count = attempt
                item.diagnostics["attempt_count"] = attempt
                item.diagnostics["retried"] = attempt > 1
                if reporter is not None:
                    reporter.set_worker_attempt(worker_id, attempt=attempt, max_attempts=retry_policy.max_attempts)
                try:
                    fetch_result = download_task_subset(
                        item.task,
                        raw_path=item.raw_path,
                        raw_manifest_path=item.raw_manifest_path,
                        raw_selection_manifest_path=item.raw_selection_manifest_path,
                        range_merge_gap_bytes=range_merge_gap_bytes,
                        overwrite=False,
                        reporter=reporter,
                        worker_id=worker_id,
                    )
                    break
                except Exception as exc:
                    item.diagnostics["last_error_type"] = type(exc).__name__
                    item.diagnostics["last_error_message"] = str(exc)
                    if sleep_for_retry(worker_id=worker_id, attempt=attempt, phase="download", exc=exc):
                        attempt += 1
                        continue
                    raise
            item.diagnostics.update(
                {
                    "head_used": getattr(fetch_result, "head_used", False),
                    "remote_file_size": getattr(fetch_result, "remote_file_size", None),
                    "selected_record_count": getattr(fetch_result, "selected_record_count", 0),
                    "merged_range_count": getattr(fetch_result, "merged_range_count", 0),
                    "downloaded_range_bytes": getattr(fetch_result, "downloaded_range_bytes", 0),
                    "timing_idx_fetch_seconds": getattr(fetch_result, "timing_idx_fetch_seconds", 0.0),
                    "timing_idx_parse_seconds": getattr(fetch_result, "timing_idx_parse_seconds", 0.0),
                    "timing_head_seconds": getattr(fetch_result, "timing_head_seconds", 0.0),
                    "timing_range_download_seconds": getattr(fetch_result, "timing_range_download_seconds", 0.0),
                    "raw_file_size": item.raw_path.stat().st_size if item.raw_path and item.raw_path.exists() else None,
                    "retry_recovered": attempt > 1,
                    "final_error_class": None,
                }
            )
            if reporter is not None:
                reporter.retire_worker(worker_id)
            return item
        except Exception as exc:
            if reporter is not None:
                reporter.fail_worker(worker_id, message=str(exc))
            raise

    def reduce_cycle(
        cycle_key: tuple[str, int],
        ok_items: list[HrrrPipelineItem],
        first_task: TaskSpec,
    ) -> tuple[tuple[str, int], list[HrrrPipelineItem], Path, Path, list[str]]:
        batch_raw_path, batch_reduced_path = batch_artifact_paths(reduced_dir, first_task)
        reduce_started_at = time.perf_counter()
        reduce_worker_id = f"batch_reduce_{cycle_key[0]}T{cycle_key[1]:02d}"
        if reporter is not None:
            reporter.upsert_group(f"{cycle_key[0]}T{cycle_key[1]:02d}", status="batch_reduce")
            reporter.start_worker(
                reduce_worker_id,
                label=f"{cycle_key[0]} t{cycle_key[1]:02d}z",
                phase="reduce",
                group_id=f"{cycle_key[0]}T{cycle_key[1]:02d}",
                details="batch_reduce",
            )
            reporter.set_worker_attempt(reduce_worker_id, attempt=1, max_attempts=retry_policy.max_attempts)
        try:
            reduce_attempt = 1
            while True:
                if reporter is not None:
                    reporter.set_worker_attempt(reduce_worker_id, attempt=reduce_attempt, max_attempts=retry_policy.max_attempts)
                try:
                    reduced_inventory, _, inventory_seconds, reduce_seconds, concat_seconds, crop_result = reduce_grib2_for_batch(
                        wgrib2_path,
                        [item.raw_path for item in ok_items if item.raw_path is not None],
                        batch_raw_path,
                        batch_reduced_path,
                        task=first_task,
                    )
                    break
                except Exception as exc:
                    for item in ok_items:
                        item.diagnostics["last_error_type"] = type(exc).__name__
                        item.diagnostics["last_error_message"] = str(exc)
                    if sleep_for_retry(worker_id=reduce_worker_id, attempt=reduce_attempt, phase="reduce", exc=exc):
                        reduce_attempt += 1
                        continue
                    raise
            batch_reduce_seconds = time.perf_counter() - reduce_started_at
            task_count = max(1, len(ok_items))
            for item in ok_items:
                item.batch_reduced_path = batch_reduced_path
                item.batch_reduced_inventory = reduced_inventory
                item.diagnostics["batch_raw_file_path"] = str(batch_raw_path)
                item.diagnostics["batch_reduced_file_path"] = str(batch_reduced_path)
                item.diagnostics["batch_lead_count"] = len(ok_items)
                item.diagnostics["batch_concat_seconds"] = concat_seconds
                item.diagnostics["batch_reduce_seconds"] = batch_reduce_seconds
                item.diagnostics["batch_cycle_reduce_seconds"] = reduce_seconds
                item.diagnostics["batch_cycle_inventory_seconds"] = inventory_seconds
                item.diagnostics["batch_timing_policy"] = "cycle_total_plus_apportioned_task"
                item.diagnostics["timing_wgrib_inventory_seconds"] = round(inventory_seconds / task_count, 6)
                item.diagnostics["timing_reduce_seconds"] = round(reduce_seconds / task_count, 6)
                item.diagnostics["crop_method"] = crop_result.method_used
                item.diagnostics["crop_command"] = crop_result.command
                item.diagnostics["crop_grid_cache_key"] = crop_result.crop_grid_cache_key
                item.diagnostics["crop_grid_cache_hit"] = crop_result.crop_grid_cache_hit
                item.diagnostics["crop_ij_box"] = crop_result.crop_ij_box
                item.diagnostics["crop_wgrib2_threads"] = crop_result.crop_wgrib2_threads
                item.diagnostics["crop_fallback_reason"] = crop_result.crop_fallback_reason
                item.diagnostics["retry_recovered"] = bool(item.diagnostics.get("retry_recovered")) or reduce_attempt > 1
                item.diagnostics["reduced_file_path"] = str(batch_reduced_path)
                item.diagnostics["reduced_file_size"] = batch_reduced_path.stat().st_size if batch_reduced_path.exists() else None
            if reporter is not None:
                reporter.retire_worker(reduce_worker_id)
            return cycle_key, ok_items, batch_raw_path, batch_reduced_path, reduced_inventory
        except Exception as exc:
            if reporter is not None:
                reporter.fail_worker(reduce_worker_id, message=str(exc))
            raise

    def extract_cycle(
        cycle_key: tuple[str, int],
        ok_items: list[HrrrPipelineItem],
        batch_raw_path: Path,
        batch_reduced_path: Path,
        reduced_inventory: list[str],
    ) -> tuple[tuple[str, int], list[TaskResult]]:
        cfgrib_parent = (scratch_dir / "cfgrib_index") if scratch_dir is not None else None
        if cfgrib_parent is not None:
            cfgrib_parent.mkdir(parents=True, exist_ok=True)
        cfgrib_index_dir = Path(
            tempfile.mkdtemp(
                prefix=f"hrrr_batch_cfgrib_{cycle_key[0]}T{cycle_key[1]:02d}_",
                dir=str(cfgrib_parent) if cfgrib_parent is not None else None,
            )
        )
        extract_worker_id = f"batch_extract_{cycle_key[0]}T{cycle_key[1]:02d}"
        if reporter is not None:
            reporter.upsert_group(f"{cycle_key[0]}T{cycle_key[1]:02d}", status="batch_extract")
            reporter.start_worker(
                extract_worker_id,
                label=f"{cycle_key[0]} t{cycle_key[1]:02d}z",
                phase="extract",
                group_id=f"{cycle_key[0]}T{cycle_key[1]:02d}",
                details="batch_extract",
            )
            reporter.set_worker_attempt(extract_worker_id, attempt=1, max_attempts=retry_policy.max_attempts)
        group_datasets: list[tuple[GroupSpec, xr.Dataset]] = []
        results: list[TaskResult] = []
        try:
            if RUNTIME_OPTIONS.extract_method == "eccodes":
                for item in ok_items:
                    extract_attempt = 1
                    item_extract_worker_id = f"batch_extract_{cycle_key[0]}T{cycle_key[1]:02d}_f{item.task.forecast_hour:02d}"
                    while True:
                        item.attempt_count = max(item.attempt_count, extract_attempt)
                        item.diagnostics["attempt_count"] = max(int(item.diagnostics.get("attempt_count") or 1), extract_attempt)
                        item.diagnostics["retried"] = bool(item.diagnostics.get("retried")) or extract_attempt > 1
                        try:
                            result = process_reduced_grib(
                                batch_reduced_path,
                                reduced_inventory,
                                item.task,
                                item.grib_url or task_remote_url(item.task, "google"),
                                diagnostics=item.diagnostics,
                                include_legacy_aliases=include_legacy_aliases,
                                filter_inventory_to_task_step=True,
                                write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                            )
                            result.diagnostics["retry_recovered"] = bool(result.diagnostics.get("retry_recovered")) or extract_attempt > 1
                            results.append(result)
                            break
                        except Exception as exc:
                            item.diagnostics["last_error_type"] = type(exc).__name__
                            item.diagnostics["last_error_message"] = str(exc)
                            if sleep_for_retry(worker_id=item_extract_worker_id, attempt=extract_attempt, phase="extract", exc=exc):
                                extract_attempt += 1
                                continue
                            raise
                return cycle_key, results

            shared_open_ok = False
            extract_attempt = 1
            while True:
                for item in ok_items:
                    item.attempt_count = max(item.attempt_count, extract_attempt)
                    item.diagnostics["attempt_count"] = max(int(item.diagnostics.get("attempt_count") or 1), extract_attempt)
                    item.diagnostics["retried"] = bool(item.diagnostics.get("retried")) or extract_attempt > 1
                if reporter is not None:
                    reporter.set_worker_attempt(extract_worker_id, attempt=extract_attempt, max_attempts=retry_policy.max_attempts)
                open_diagnostics: dict[str, object] = {"timing_cfgrib_open_seconds": 0.0}
                try:
                    group_datasets = open_reduced_grib_group_datasets(
                        batch_reduced_path,
                        reduced_inventory,
                        diagnostics=open_diagnostics,
                        cfgrib_index_dir=cfgrib_index_dir,
                    )
                    shared_open_seconds = float(open_diagnostics["timing_cfgrib_open_seconds"])
                    task_open_seconds = round(shared_open_seconds / max(1, len(ok_items)), 6)
                    for item in ok_items:
                        item.diagnostics["batch_cycle_cfgrib_open_seconds"] = shared_open_seconds
                        item.diagnostics["batch_timing_policy"] = "cycle_total_plus_apportioned_task"
                        item.diagnostics["timing_cfgrib_open_seconds"] = task_open_seconds
                        item.diagnostics["retry_recovered"] = bool(item.diagnostics.get("retry_recovered")) or extract_attempt > 1
                    shared_open_ok = True
                    break
                except Exception as exc:
                    for item in ok_items:
                        item.diagnostics["last_error_type"] = type(exc).__name__
                        item.diagnostics["last_error_message"] = str(exc)
                    if sleep_for_retry(worker_id=extract_worker_id, attempt=extract_attempt, phase="extract", exc=exc):
                        extract_attempt += 1
                        continue
                    for item in ok_items:
                        results.append(TaskResult(False, item.task.key, None, [], [], str(exc), item.diagnostics))
                        if reporter is not None:
                            reporter.record_outcome("failed", message=f"{item.task.key} {str(exc) or 'unknown error'}")
                    break

            if shared_open_ok:
                for item in ok_items:
                    extract_attempt = 1
                    item_extract_worker_id = f"batch_extract_{cycle_key[0]}T{cycle_key[1]:02d}_f{item.task.forecast_hour:02d}"
                    while True:
                        item.attempt_count = max(item.attempt_count, extract_attempt)
                        item.diagnostics["attempt_count"] = max(int(item.diagnostics.get("attempt_count") or 1), extract_attempt)
                        item.diagnostics["retried"] = bool(item.diagnostics.get("retried")) or extract_attempt > 1
                        try:
                            result = build_task_result_from_open_datasets(
                                group_datasets,
                                item.batch_reduced_inventory or [],
                                item.task,
                                item.grib_url or task_remote_url(item.task, "google"),
                                diagnostics=item.diagnostics,
                                include_legacy_aliases=include_legacy_aliases,
                                filter_inventory_to_task_step=True,
                                write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                            )
                            result.diagnostics["retry_recovered"] = bool(result.diagnostics.get("retry_recovered")) or extract_attempt > 1
                            results.append(result)
                            break
                        except Exception as exc:
                            item.diagnostics["last_error_type"] = type(exc).__name__
                            item.diagnostics["last_error_message"] = str(exc)
                            if sleep_for_retry(worker_id=item_extract_worker_id, attempt=extract_attempt, phase="extract", exc=exc):
                                extract_attempt += 1
                                continue
                            raise
        finally:
            close_group_datasets(group_datasets)
            shutil.rmtree(cfgrib_index_dir, ignore_errors=True)

            cleanup_started_at = time.perf_counter()
            if not keep_reduced:
                batch_reduced_path.unlink(missing_ok=True)
                reduced_inventory_path(batch_reduced_path).unlink(missing_ok=True)
                reduced_signature_path(batch_reduced_path).unlink(missing_ok=True)
            if not keep_downloads:
                batch_raw_path.unlink(missing_ok=True)
                for item in ok_items:
                    if item.raw_path is not None:
                        item.raw_path.unlink(missing_ok=True)
                    if item.raw_manifest_path is not None:
                        item.raw_manifest_path.unlink(missing_ok=True)
                    if item.raw_selection_manifest_path is not None:
                        item.raw_selection_manifest_path.unlink(missing_ok=True)
            cleanup_seconds = time.perf_counter() - cleanup_started_at
            for item in ok_items:
                item.diagnostics["timing_cleanup_seconds"] = cleanup_seconds
            if reporter is not None:
                reporter.retire_worker(extract_worker_id)
        return cycle_key, results

    def submit_ready_reduces(
        reduce_ready: list[tuple[tuple[str, int], list[HrrrPipelineItem], TaskSpec]],
        active_reduces: dict[object, tuple[tuple[str, int], list[HrrrPipelineItem]]],
        reduce_executor: ThreadPoolExecutor,
    ) -> None:
        while reduce_ready and len(active_reduces) < phase_limits.reduce_workers and len(extract_ready) < extract_queue_limit:
            cycle_key, ok_items, first_task = reduce_ready.pop(0)
            future = reduce_executor.submit(reduce_cycle, cycle_key, ok_items, first_task)
            active_reduces[future] = (cycle_key, ok_items)

    def submit_ready_extracts(
        extract_ready: list[tuple[tuple[str, int], list[HrrrPipelineItem], Path, Path, list[str]]],
        active_extracts: dict[object, tuple[tuple[str, int], list[HrrrPipelineItem]]],
        extract_executor: ThreadPoolExecutor,
    ) -> None:
        while extract_ready and len(active_extracts) < phase_limits.extract_workers:
            cycle_key, ok_items, batch_raw_path, batch_reduced_path, reduced_inventory = extract_ready.pop(0)
            future = extract_executor.submit(extract_cycle, cycle_key, ok_items, batch_raw_path, batch_reduced_path, reduced_inventory)
            active_extracts[future] = (cycle_key, ok_items)

    try:
        active_downloads: dict[object, tuple[tuple[str, int], HrrrPipelineItem]] = {}
        active_reduces: dict[object, tuple[tuple[str, int], list[HrrrPipelineItem]]] = {}
        active_extracts: dict[object, tuple[tuple[str, int], list[HrrrPipelineItem]]] = {}
        reduce_ready: list[tuple[tuple[str, int], list[HrrrPipelineItem], TaskSpec]] = []
        extract_ready: list[tuple[tuple[str, int], list[HrrrPipelineItem], Path, Path, list[str]]] = []
        with ThreadPoolExecutor(max_workers=phase_limits.download_workers) as download_executor, ThreadPoolExecutor(
            max_workers=phase_limits.reduce_workers
        ) as reduce_executor, ThreadPoolExecutor(max_workers=phase_limits.extract_workers) as extract_executor:
            while (
                pending_downloads
                or active_downloads
                or reduce_ready
                or active_reduces
                or extract_ready
                or active_extracts
            ):
                while (
                    pending_downloads
                    and len(active_downloads) < phase_limits.download_workers
                    and len(reduce_ready) < reduce_queue_limit
                    and not run_control.pause_requested()
                ):
                    cycle_key, item = pending_downloads.pop(0)
                    if reporter is not None:
                        reporter.upsert_group(f"{cycle_key[0]}T{cycle_key[1]:02d}", status="download")
                    future = download_executor.submit(download_cycle_item, cycle_key, item)
                    active_downloads[future] = (cycle_key, item)

                submit_ready_reduces(reduce_ready, active_reduces, reduce_executor)
                submit_ready_extracts(extract_ready, active_extracts, extract_executor)

                active_futures = tuple(active_downloads.keys()) + tuple(active_reduces.keys()) + tuple(active_extracts.keys())
                if not active_futures:
                    if run_control.pause_requested():
                        break
                    time.sleep(0.1)
                    continue

                done_futures, _ = wait(active_futures, timeout=0.1, return_when=FIRST_COMPLETED)
                if not done_futures:
                    continue

                for future in list(done_futures):
                    if future in active_downloads:
                        cycle_key, item = active_downloads.pop(future)
                        state = cycle_states[cycle_key]
                        state["remaining_downloads"] = int(state["remaining_downloads"]) - 1
                        try:
                            future.result()
                            failures.pop(item.task.key, None)
                        except Exception as exc:
                            result = TaskResult(False, item.task.key, None, [], [], str(exc), item.diagnostics)
                            failures[result.task_key] = result.message or "unknown error"
                            task_diagnostics[result.task_key] = result.diagnostics
                            failed += 1
                        if int(state["remaining_downloads"]) == 0 and not bool(state["reduce_submitted"]):
                            items = state["items"]
                            ok_items = [
                                candidate
                                for candidate in items
                                if candidate.task.key not in failures
                                and candidate.raw_path is not None
                                and candidate.raw_path.exists()
                            ]
                            state["reduce_submitted"] = True
                            if ok_items:
                                reduce_ready.append((cycle_key, ok_items, state["first_task"]))
                            else:
                                state["done"] = True

                submit_ready_reduces(reduce_ready, active_reduces, reduce_executor)
                for future in list(done_futures):
                    if future in active_reduces:
                        cycle_key, ok_items = active_reduces.pop(future)
                        try:
                            extract_ready.append(future.result())
                        except Exception as exc:
                            for item in ok_items:
                                result = TaskResult(False, item.task.key, None, [], [], str(exc), item.diagnostics)
                                failures[result.task_key] = result.message or "unknown error"
                                task_diagnostics[result.task_key] = result.diagnostics
                                failed += 1
                            cycle_states[cycle_key]["done"] = True

                submit_ready_extracts(extract_ready, active_extracts, extract_executor)
                for future in list(done_futures):
                    if future in active_extracts:
                        cycle_key, ok_items = active_extracts.pop(future)
                        cycle_key_result, results = future.result()
                        for result in results:
                            task_diagnostics[result.task_key] = result.diagnostics
                            if result.ok and result.row is not None:
                                append_jsonl_batch(row_buffer_file, [result.row])
                                row_buffer[result.task_key] = result.row
                                if result.provenance_rows:
                                    append_jsonl_batch(provenance_buffer_file, result.provenance_rows)
                                    for provenance_entry in result.provenance_rows:
                                        provenance_buffer[f"{provenance_entry['task_key']}|{provenance_entry['feature_name']}"] = provenance_entry
                                completed.add(result.task_key)
                                failures.pop(result.task_key, None)
                                missing_fields[result.task_key] = result.missing_fields
                                wrote += 1
                                if reporter is not None:
                                    reporter.record_outcome("completed", message=f"{result.task_key} ok")
                            else:
                                failures[result.task_key] = result.message or "unknown error"
                                failed += 1
                                if reporter is not None:
                                    reporter.record_outcome(
                                        "failed",
                                        message=f"{result.task_key} {result.message or 'unknown error'}",
                                    )
                        if reporter is not None:
                            reporter.upsert_group(
                                f"{cycle_key_result[0]}T{cycle_key_result[1]:02d}",
                                completed=sum(1 for item in ok_items if item.task.key in completed),
                                failed=sum(1 for item in ok_items if item.task.key in failures),
                                status="done",
                            )
                        cycle_states[cycle_key]["done"] = True
    except BaseException:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        raise
    finally:
        if reporter is not None:
            if run_control.pause_requested():
                run_control.mark_paused(reason=run_control.pause_reason or "operator")
                reporter.mark_paused(reason=run_control.pause_reason or "operator")
            reporter.close(status="paused" if run_control.is_paused() else f"month={month_id} wrote={wrote} failed={failed}")

    if len(completed) == len(expected_task_keys):
        manifest["complete"] = True
        manifest["completed_task_keys"] = sorted(completed)
        manifest["failure_reasons"] = {}
        manifest["missing_fields"] = missing_fields
        manifest["task_diagnostics"] = task_diagnostics
        finalize_month(
            output_dir,
            summary_output_dir,
            month_id,
            row_buffer,
            provenance_buffer,
            manifest,
            write_provenance=not RUNTIME_OPTIONS.skip_provenance,
        )
        provenance_label = manifest.get("provenance_path") or "skipped"
        print(
            f"[month-done] {month_id} rows={len(row_buffer)} parquet={parquet_path(output_dir, month_id)} "
            f"provenance={provenance_label} summary={summary_parquet_path(summary_output_dir, month_id)}"
        )
    else:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        print(
            f"[month-incomplete] {month_id} completed={len(completed)} expected={len(expected_task_keys)} "
            f"failures={len(failures)}",
            file=sys.stderr,
        )
    return wrote, failed


def run_month(
    month_id: str,
    tasks: list[TaskSpec],
    *,
    wgrib2_path: str,
    output_dir: Path,
    summary_output_dir: Path,
    download_dir: Path,
    reduced_dir: Path,
    max_workers: int,
    keep_downloads: bool,
    keep_reduced: bool,
    download_workers: int | None = None,
    reduce_workers: int | None = None,
    extract_workers: int | None = None,
    reduce_queue_size: int | None = None,
    extract_queue_size: int | None = None,
    range_merge_gap_bytes: int = DEFAULT_RANGE_MERGE_GAP_BYTES,
    batch_reduce_mode: str = "off",
    include_legacy_aliases: bool = False,
    scratch_dir: Path | None = None,
    progress_mode: str = "auto",
    max_task_attempts: int = 6,
    retry_backoff_seconds: float = 2.0,
    retry_max_backoff_seconds: float = 30.0,
    run_control: RunControl | None = None,
    enable_dashboard_hotkeys: bool = True,
    pause_control_file: str | None = None,
) -> tuple[int, int]:
    if batch_reduce_mode == "cycle":
        return _run_month_batch_cycle(
            month_id,
            tasks,
            wgrib2_path=wgrib2_path,
            output_dir=output_dir,
            summary_output_dir=summary_output_dir,
            download_dir=download_dir,
            reduced_dir=reduced_dir,
            max_workers=max_workers,
            keep_downloads=keep_downloads,
            keep_reduced=keep_reduced,
            download_workers=download_workers,
            reduce_workers=reduce_workers,
            extract_workers=extract_workers,
            reduce_queue_size=reduce_queue_size,
            extract_queue_size=extract_queue_size,
            range_merge_gap_bytes=range_merge_gap_bytes,
            include_legacy_aliases=include_legacy_aliases,
            scratch_dir=scratch_dir,
            progress_mode=progress_mode,
            max_task_attempts=max_task_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_max_backoff_seconds=retry_max_backoff_seconds,
            run_control=run_control,
            enable_dashboard_hotkeys=enable_dashboard_hotkeys,
            pause_control_file=pause_control_file,
        )
    if process_task is not _ORIGINAL_PROCESS_TASK:
        return _run_month_legacy(
            month_id,
            tasks,
            wgrib2_path=wgrib2_path,
            output_dir=output_dir,
            summary_output_dir=summary_output_dir,
            download_dir=download_dir,
            reduced_dir=reduced_dir,
            max_workers=max_workers,
            keep_downloads=keep_downloads,
            keep_reduced=keep_reduced,
            download_workers=download_workers,
            reduce_workers=reduce_workers,
            extract_workers=extract_workers,
            reduce_queue_size=reduce_queue_size,
            extract_queue_size=extract_queue_size,
            range_merge_gap_bytes=range_merge_gap_bytes,
            include_legacy_aliases=include_legacy_aliases,
            scratch_dir=scratch_dir,
            progress_mode=progress_mode,
            max_task_attempts=max_task_attempts,
            retry_backoff_seconds=retry_backoff_seconds,
            retry_max_backoff_seconds=retry_max_backoff_seconds,
            enable_dashboard_hotkeys=enable_dashboard_hotkeys,
            pause_control_file=pause_control_file,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_task_keys = [task.key for task in tasks]
    manifest_file = manifest_path(output_dir, month_id)
    manifest = load_manifest(
        manifest_file,
        month_id,
        expected_task_keys,
        keep_downloads=keep_downloads,
        keep_reduced=keep_reduced,
    )
    if month_is_complete(output_dir, month_id, manifest, expected_task_keys):
        print(f"[skip-month] {month_id} already complete")
        return 0, 0

    row_buffer_file = row_buffer_path(output_dir, month_id)
    provenance_buffer_file = provenance_buffer_path(output_dir, month_id)
    completed = set(manifest.get("completed_task_keys", []))
    pending = [task for task in tasks if task.key not in completed]
    phase_limits = build_phase_concurrency_limits(
        max_workers=max_workers,
        args=argparse.Namespace(download_workers=download_workers, reduce_workers=reduce_workers, extract_workers=extract_workers),
    )
    row_buffer = load_row_buffer(row_buffer_file)
    provenance_buffer = load_provenance_buffer(provenance_buffer_file)
    failures = dict(manifest.get("failure_reasons", {}))
    missing_fields = dict(manifest.get("missing_fields", {}))
    task_diagnostics = dict(manifest.get("task_diagnostics", {}))

    wrote = 0
    failed = 0
    run_control = run_control or RunControl()
    reporter = (
        create_progress_reporter(
            f"HRRR {month_id}",
            unit="task",
            total=len(pending),
            mode=progress_mode,
            stream=sys.stdout,
            on_pause_request=lambda **kwargs: run_control.request_pause(reason=str(kwargs.get("reason", "operator"))),
            enable_dashboard_hotkeys=enable_dashboard_hotkeys,
            pause_control_file=pause_control_file,
        )
        if pending
        else None
    )

    phase_activity_lock = threading.Lock()
    phase_activity = {"download": 0, "reduce": 0, "extract": 0}

    def update_phase_activity(phase_name: str, delta: int) -> None:
        with phase_activity_lock:
            phase_activity[phase_name] = max(0, phase_activity[phase_name] + delta)

    def current_phase_activity() -> tuple[int, int, int]:
        with phase_activity_lock:
            return phase_activity["download"], phase_activity["reduce"], phase_activity["extract"]

    def update_pipeline_metrics(*, reduce_queue_obj, extract_queue_obj) -> None:
        if reporter is None:
            return
        active_download, active_reduce, active_extract = current_phase_activity()
        reporter.set_metrics(
            month=month_id,
            retained_cycles=len({(task.run_date_utc, task.cycle_hour_utc) for task in pending}),
            max_workers=max_workers,
            download_workers=phase_limits.download_workers,
            reduce_workers=phase_limits.reduce_workers,
            extract_workers=phase_limits.extract_workers,
            reduce_queued=reduce_queue_obj.qsize(),
            extract_queued=extract_queue_obj.qsize(),
            active_download=active_download,
            active_reduce=active_reduce,
            active_extract=active_extract,
        )

    if reporter is not None:
        reporter.upsert_group(month_id, label=month_id, total=len(pending), completed=0, failed=0, status="queued")

    if not pending:
        if reporter is not None:
            reporter.close(status=f"month={month_id} wrote=0 failed=0")
        return 0, 0

    download_queue: queue.Queue[HrrrPipelineItem | None] = queue.Queue()
    reduce_queue: queue.Queue[HrrrPipelineItem | None] = queue.Queue(
        maxsize=resolve_pipeline_queue_size(reduce_queue_size, downstream_workers=phase_limits.reduce_workers)
    )
    extract_queue: queue.Queue[HrrrPipelineItem | None] = queue.Queue(
        maxsize=resolve_pipeline_queue_size(extract_queue_size, downstream_workers=phase_limits.extract_workers)
    )
    completion_queue: queue.Queue[TaskResult] = queue.Queue()
    update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)

    retry_policy = RetryPolicy(
        max_attempts=max(1, int(max_task_attempts)),
        backoff_seconds=float(retry_backoff_seconds),
        max_backoff_seconds=float(retry_max_backoff_seconds),
    )

    def make_initial_item(task: TaskSpec) -> HrrrPipelineItem:
        raw_path = path_for_raw(download_dir, task)
        raw_manifest_path = path_for_raw_manifest(download_dir, task)
        raw_selection_manifest_path = path_for_raw_selection_manifest(download_dir, task)
        reduced_path = path_for_reduced(reduced_dir, task)
        diagnostics = default_task_diagnostics(task)
        diagnostics["scratch_dir"] = str(scratch_dir) if scratch_dir is not None else None
        diagnostics["attempt_count"] = 1
        diagnostics["retried"] = False
        diagnostics["retry_recovered"] = False
        diagnostics["final_error_class"] = None
        diagnostics["last_error_type"] = None
        diagnostics["last_error_message"] = None
        diagnostics["raw_file_path"] = str(raw_path)
        diagnostics["raw_manifest_path"] = str(raw_manifest_path)
        diagnostics["raw_selection_manifest_path"] = str(raw_selection_manifest_path)
        diagnostics["reduced_file_path"] = str(reduced_path)
        grib_url = task_remote_url(task, "google")
        diagnostics["grib_url"] = grib_url
        return HrrrPipelineItem(
            task=task,
            attempt_count=1,
            raw_path=raw_path,
            raw_manifest_path=raw_manifest_path,
            raw_selection_manifest_path=raw_selection_manifest_path,
            reduced_path=reduced_path,
            grib_url=grib_url,
            diagnostics=diagnostics,
        )

    def finalize_failure(item: HrrrPipelineItem, exc: Exception, *, phase: str) -> TaskResult:
        diagnostics = dict(item.diagnostics)
        diagnostics["attempt_count"] = item.attempt_count
        diagnostics["retried"] = item.attempt_count > 1
        diagnostics["retry_recovered"] = False
        diagnostics["last_error_type"] = type(exc).__name__
        diagnostics["last_error_message"] = str(exc)
        diagnostics["final_error_class"] = classify_task_failure(exception_type=type(exc).__name__, message=str(exc), phase=phase).error_class
        cleanup_started_at = time.perf_counter()
        if item.reduced_path is not None and item.reduced_path.exists() and not keep_reduced:
            item.reduced_path.unlink(missing_ok=True)
            reduced_signature_path(item.reduced_path).unlink(missing_ok=True)
            reduced_inventory_path(item.reduced_path).unlink(missing_ok=True)
        if item.raw_path is not None and item.raw_path.exists() and not keep_downloads:
            item.raw_path.unlink(missing_ok=True)
        if item.raw_manifest_path is not None and item.raw_manifest_path.exists() and not keep_downloads:
            item.raw_manifest_path.unlink(missing_ok=True)
        if item.raw_selection_manifest_path is not None and item.raw_selection_manifest_path.exists() and not keep_downloads:
            item.raw_selection_manifest_path.unlink(missing_ok=True)
        diagnostics["timing_cleanup_seconds"] = time.perf_counter() - cleanup_started_at
        return TaskResult(False, item.task.key, None, [], [], str(exc), diagnostics)

    def maybe_retry(item: HrrrPipelineItem, *, worker_id: str, phase: str, exc: Exception) -> bool:
        decision = classify_task_failure(exception_type=type(exc).__name__, message=str(exc), phase=phase)
        if not should_retry_attempt(attempt=item.attempt_count, policy=retry_policy, decision=decision):
            return False
        next_attempt = item.attempt_count + 1
        delay_seconds = compute_retry_delay_seconds(attempt=next_attempt, policy=retry_policy)
        if reporter is not None:
            reporter.schedule_retry(
                worker_id,
                attempt=next_attempt,
                max_attempts=retry_policy.max_attempts,
                delay_seconds=delay_seconds,
                message=str(exc),
                error_class=decision.error_class,
            )
        remaining = max(0.0, float(delay_seconds))
        while remaining > 0:
            sleep_seconds = min(0.2, remaining)
            time.sleep(sleep_seconds)
            remaining -= sleep_seconds
            if reporter is not None:
                reporter.refresh(force=True)
        item.attempt_count = next_attempt
        item.diagnostics["attempt_count"] = next_attempt
        item.diagnostics["retried"] = next_attempt > 1
        return True

    def run_download_stage(item: HrrrPipelineItem, *, worker_id: str) -> HrrrPipelineItem:
        task = item.task
        if reporter is not None:
            reporter.start_worker(worker_id, label=f"{task.target_date_local} c{task.cycle_hour_utc:02d} f{task.forecast_hour:02d}", phase="download", group_id=month_id, details="fetch_subset")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=retry_policy.max_attempts)
        fetch_kwargs = {
            "date": task.run_date_utc.replace("-", ""),
            "cycle": task.cycle_hour_utc,
            "product": "surface",
            "forecast_hour": task.forecast_hour,
            "source": "google",
            "patterns": [pattern for _, pattern in inventory_selection_patterns_for_task(task)],
            "subset_path": item.raw_path,
            "manifest_path": item.raw_manifest_path,
            "selection_manifest_path": item.raw_selection_manifest_path,
            "overwrite": False,
        }
        try:
            fetch_result = download_subset_for_inventory_patterns(
                **fetch_kwargs,
                range_merge_gap_bytes=range_merge_gap_bytes,
                progress_callback=(
                    (lambda event_name, payload: _report_hrrr_download_event(reporter=reporter, worker_id=worker_id, event_name=event_name, payload=payload))
                    if reporter is not None
                    else None
                ),
            )
        except TypeError as exc:
            if "range_merge_gap_bytes" not in str(exc) and "progress_callback" not in str(exc):
                raise
            fetch_result = download_subset_for_inventory_patterns(**fetch_kwargs)
        if isinstance(fetch_result, Path):
            class _LegacyFetchResult:
                head_used = False
                remote_file_size = None
                selected_record_count = 0
                merged_range_count = 0
                downloaded_range_bytes = 0
                timing_idx_fetch_seconds = 0.0
                timing_idx_parse_seconds = 0.0
                timing_head_seconds = 0.0
                timing_range_download_seconds = 0.0
            fetch_result = _LegacyFetchResult()
        item.diagnostics.update(
            {
                "head_used": fetch_result.head_used,
                "remote_file_size": fetch_result.remote_file_size,
                "selected_record_count": fetch_result.selected_record_count,
                "merged_range_count": fetch_result.merged_range_count,
                "downloaded_range_bytes": fetch_result.downloaded_range_bytes,
                "timing_idx_fetch_seconds": fetch_result.timing_idx_fetch_seconds,
                "timing_idx_parse_seconds": fetch_result.timing_idx_parse_seconds,
                "timing_head_seconds": fetch_result.timing_head_seconds,
                "timing_range_download_seconds": fetch_result.timing_range_download_seconds,
                "raw_file_size": item.raw_path.stat().st_size if item.raw_path.exists() else None,
            }
        )
        return item

    def run_reduce_stage(item: HrrrPipelineItem, *, worker_id: str) -> HrrrPipelineItem:
        task = item.task
        if reporter is not None:
            reporter.start_worker(worker_id, label=f"{task.target_date_local} c{task.cycle_hour_utc:02d} f{task.forecast_hour:02d}", phase="reduce", group_id=month_id, details="reduce_grib2")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=retry_policy.max_attempts)
        reuse_signature: str | None = None
        if keep_reduced:
            reuse_signature = build_reduced_reuse_signature(
                task=task,
                raw_path=item.raw_path,
                selection_manifest_path=item.raw_selection_manifest_path,
            )
            item.diagnostics["reduced_reuse_signature"] = reuse_signature
        if keep_reduced and reuse_signature is not None and reduced_grib_reusable(reduced_path=item.reduced_path, signature=reuse_signature):
            item.diagnostics["reduced_reused"] = True
            inventory = load_reduced_inventory(reduced_path=item.reduced_path, signature=reuse_signature)
            if inventory is None:
                inventory_started_at = time.perf_counter()
                inventory = inventory_for_grib(wgrib2_path, item.reduced_path)
                item.diagnostics["timing_wgrib_inventory_seconds"] = time.perf_counter() - inventory_started_at
            else:
                item.diagnostics["timing_wgrib_inventory_seconds"] = 0.0
            item.reduced_inventory = inventory
            item.diagnostics["timing_reduce_seconds"] = 0.0
            return item
        crop_result = reduce_grib2_for_task(
            wgrib2_path,
            item.raw_path,
            item.reduced_path,
            task=task,
            raw_manifest_path=item.raw_manifest_path,
        )
        reduced_inventory = crop_result.selected_lines
        inventory_seconds = crop_result.inventory_seconds
        reduce_seconds = crop_result.reduce_seconds
        if keep_reduced and reuse_signature is not None:
            write_reduced_reuse_signature(item.reduced_path, reuse_signature)
            write_reduced_inventory(item.reduced_path, reuse_signature, reduced_inventory)
        item.reduced_inventory = reduced_inventory
        item.diagnostics["reduced_file_size"] = item.reduced_path.stat().st_size if item.reduced_path.exists() else None
        item.diagnostics["timing_wgrib_inventory_seconds"] = inventory_seconds
        item.diagnostics["timing_reduce_seconds"] = reduce_seconds
        item.diagnostics["crop_method"] = crop_result.method_used
        item.diagnostics["crop_command"] = crop_result.command
        item.diagnostics["crop_grid_cache_key"] = crop_result.crop_grid_cache_key
        item.diagnostics["crop_grid_cache_hit"] = crop_result.crop_grid_cache_hit
        item.diagnostics["crop_ij_box"] = crop_result.crop_ij_box
        item.diagnostics["crop_wgrib2_threads"] = crop_result.crop_wgrib2_threads
        item.diagnostics["crop_fallback_reason"] = crop_result.crop_fallback_reason
        return item

    def run_extract_stage(item: HrrrPipelineItem, *, worker_id: str) -> TaskResult:
        task = item.task
        if reporter is not None:
            reporter.start_worker(worker_id, label=f"{task.target_date_local} c{task.cycle_hour_utc:02d} f{task.forecast_hour:02d}", phase="extract", group_id=month_id, details="process_reduced_grib")
            reporter.set_worker_attempt(worker_id, attempt=item.attempt_count, max_attempts=retry_policy.max_attempts)
        cfgrib_parent = (scratch_dir / "cfgrib_index") if scratch_dir is not None else None
        if cfgrib_parent is not None:
            cfgrib_parent.mkdir(parents=True, exist_ok=True)
        cfgrib_index_dir = Path(tempfile.mkdtemp(prefix=f"hrrr_cfgrib_{task.forecast_hour:02d}_", dir=str(cfgrib_parent) if cfgrib_parent is not None else None))
        try:
            try:
                result = process_reduced_grib(
                    item.reduced_path,
                    item.reduced_inventory,
                    task,
                    item.grib_url,
                    cfgrib_index_dir=cfgrib_index_dir,
                    diagnostics=item.diagnostics,
                    include_legacy_aliases=include_legacy_aliases,
                    write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                )
            except TypeError as exc:
                if "cfgrib_index_dir" not in str(exc) and "diagnostics" not in str(exc):
                    raise
                result = process_reduced_grib(
                    item.reduced_path,
                    item.reduced_inventory,
                    task,
                    item.grib_url,
                    include_legacy_aliases=include_legacy_aliases,
                    write_provenance=not RUNTIME_OPTIONS.skip_provenance,
                )
        finally:
            shutil.rmtree(cfgrib_index_dir, ignore_errors=True)
        cleanup_started_at = time.perf_counter()
        if item.reduced_path.exists() and not keep_reduced:
            item.reduced_path.unlink(missing_ok=True)
            reduced_signature_path(item.reduced_path).unlink(missing_ok=True)
            reduced_inventory_path(item.reduced_path).unlink(missing_ok=True)
        if item.raw_path.exists() and not keep_downloads:
            item.raw_path.unlink(missing_ok=True)
        if item.raw_manifest_path.exists() and not keep_downloads:
            item.raw_manifest_path.unlink(missing_ok=True)
        if item.raw_selection_manifest_path.exists() and not keep_downloads:
            item.raw_selection_manifest_path.unlink(missing_ok=True)
        result.diagnostics["timing_cleanup_seconds"] = time.perf_counter() - cleanup_started_at
        return result

    seeded_count = 0
    pending_index = 0

    def admit_task(task: TaskSpec) -> None:
        nonlocal seeded_count
        download_queue.put(make_initial_item(task))
        seeded_count += 1

    initial_admit = min(len(pending), max(1, max_workers))
    for _ in range(initial_admit):
        admit_task(pending[pending_index])
        pending_index += 1

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
                        reduce_queue.put(run_download_stage(item, worker_id=worker_id))
                        if reporter is not None and item.attempt_count > 1:
                            reporter.recover_worker(worker_id, message=f"{item.task.key} download recovered")
                        update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                        break
                    except Exception as exc:
                        if maybe_retry(item, worker_id=worker_id, phase="download", exc=exc):
                            continue
                        completion_queue.put(finalize_failure(item, exc, phase="download"))
                        break
            finally:
                update_phase_activity("download", -1)
                update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
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
                        extract_queue.put(run_reduce_stage(item, worker_id=worker_id))
                        if reporter is not None and item.attempt_count > 1:
                            reporter.recover_worker(worker_id, message=f"{item.task.key} reduce recovered")
                        update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                        break
                    except Exception as exc:
                        if maybe_retry(item, worker_id=worker_id, phase="reduce", exc=exc):
                            continue
                        completion_queue.put(finalize_failure(item, exc, phase="reduce"))
                        break
            finally:
                update_phase_activity("reduce", -1)
                update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
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
                        result = run_extract_stage(item, worker_id=worker_id)
                        if reporter is not None and item.attempt_count > 1:
                            reporter.recover_worker(worker_id, message=f"{item.task.key} extract recovered")
                        result.diagnostics["attempt_count"] = item.attempt_count
                        result.diagnostics["retried"] = item.attempt_count > 1
                        result.diagnostics["retry_recovered"] = item.attempt_count > 1
                        completion_queue.put(result)
                        break
                    except Exception as exc:
                        if maybe_retry(item, worker_id=worker_id, phase="extract", exc=exc):
                            continue
                        completion_queue.put(finalize_failure(item, exc, phase="extract"))
                        break
            finally:
                update_phase_activity("extract", -1)
                update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
                if reporter is not None:
                    reporter.retire_worker(worker_id)
                extract_queue.task_done()

    download_threads = [threading.Thread(target=download_worker, args=(index,), daemon=True) for index in range(phase_limits.download_workers)]
    reduce_threads = [threading.Thread(target=reduce_worker, args=(index,), daemon=True) for index in range(phase_limits.reduce_workers)]
    extract_threads = [threading.Thread(target=extract_worker, args=(index,), daemon=True) for index in range(phase_limits.extract_workers)]
    for thread in download_threads + reduce_threads + extract_threads:
        thread.start()

    def pause_drain_complete(processed: int) -> bool:
        active_download, active_reduce, active_extract = current_phase_activity()
        return (
            processed >= seeded_count
            and completion_queue.empty()
            and download_queue.empty()
            and reduce_queue.empty()
            and extract_queue.empty()
            and active_download == 0
            and active_reduce == 0
            and active_extract == 0
        )

    try:
        processed = 0
        while True:
            if run_control.pause_requested() and pause_drain_complete(processed):
                if reporter is not None:
                    reporter.mark_paused(reason="operator")
                run_control.mark_paused(reason="operator")
                break
            if processed >= len(pending):
                break
            try:
                result = completion_queue.get(timeout=0.1)
            except queue.Empty:
                if reporter is not None and run_control.pause_requested():
                    reporter.refresh(force=True)
                continue
            processed += 1
            task_diagnostics[result.task_key] = result.diagnostics
            if result.ok and result.row is not None:
                append_jsonl_batch(row_buffer_file, [result.row])
                row_buffer[result.task_key] = result.row
                if result.provenance_rows:
                    append_jsonl_batch(provenance_buffer_file, result.provenance_rows)
                    for provenance_entry in result.provenance_rows:
                        provenance_buffer[f"{provenance_entry['task_key']}|{provenance_entry['feature_name']}"] = provenance_entry
                completed.add(result.task_key)
                failures.pop(result.task_key, None)
                missing_fields[result.task_key] = result.missing_fields
                wrote += 1
                if reporter is not None:
                    reporter.record_outcome("completed", message=f"{result.task_key} ok")
            else:
                failures[result.task_key] = result.message or "unknown error"
                failed += 1
                if reporter is not None:
                    reporter.record_outcome("failed", message=f"{result.task_key} {result.message or 'unknown error'}")
            if reporter is not None:
                reporter.upsert_group(month_id, completed=wrote, failed=failed, status=f"completed={processed}/{len(pending)}")
            if pending_index < len(pending) and not run_control.pause_requested():
                admit_task(pending[pending_index])
                pending_index += 1
            update_pipeline_metrics(reduce_queue_obj=reduce_queue, extract_queue_obj=extract_queue)
    except BaseException:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        raise
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
        for thread in download_threads + reduce_threads + extract_threads:
            thread.join()
        if reporter is not None:
            reporter.upsert_group(month_id, completed=wrote, failed=failed, status="finalize")
            reporter.close(status="paused" if run_control.is_paused() else f"month={month_id} wrote={wrote} failed={failed}")

    if len(completed) == len(expected_task_keys):
        manifest["complete"] = True
        manifest["completed_task_keys"] = sorted(completed)
        manifest["failure_reasons"] = {}
        manifest["missing_fields"] = missing_fields
        manifest["task_diagnostics"] = task_diagnostics
        finalize_month(
            output_dir,
            summary_output_dir,
            month_id,
            row_buffer,
            provenance_buffer,
            manifest,
            write_provenance=not RUNTIME_OPTIONS.skip_provenance,
        )
        provenance_label = manifest.get("provenance_path") or "skipped"
        print(
            f"[month-done] {month_id} rows={len(row_buffer)} parquet={parquet_path(output_dir, month_id)} "
            f"provenance={provenance_label} summary={summary_parquet_path(summary_output_dir, month_id)}"
        )
    else:
        manifest["complete"] = False
        persist_month_manifest(
            output_dir=output_dir,
            month_id=month_id,
            manifest_file=manifest_file,
            manifest=manifest,
            completed=completed,
            failures=failures,
            missing_fields=missing_fields,
            task_diagnostics=task_diagnostics,
        )
        print(
            f"[month-incomplete] {month_id} completed={len(completed)} expected={len(expected_task_keys)} "
            f"failures={len(failures)}",
            file=sys.stderr,
        )
    return wrote, failed


def main() -> int:
    global RUNTIME_OPTIONS
    args = parse_args()
    RUNTIME_OPTIONS = HrrrRuntimeOptions(
        crop_method=str(args.crop_method),
        crop_grib_type=str(args.crop_grib_type),
        wgrib2_threads=args.wgrib2_threads,
        max_workers=max(1, int(args.max_workers)),
        reduce_workers=args.reduce_workers,
        extract_method=str(args.extract_method),
        summary_profile=str(args.summary_profile),
        skip_provenance=bool(args.skip_provenance),
    )
    wgrib2_path = ensure_tooling()
    scratch_dir = args.scratch_dir
    download_dir = (scratch_dir / "downloads") if scratch_dir is not None else args.download_dir
    reduced_dir = (scratch_dir / "reduced") if scratch_dir is not None else args.reduced_dir

    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    if end_date < start_date:
        print("End date must be on or after start date.", file=sys.stderr)
        return 2

    tasks = build_all_tasks(start_date, end_date, selection_mode=args.selection_mode)
    if args.limit_tasks is not None:
        tasks = tasks[: args.limit_tasks]
    if not tasks:
        print("No tasks to process.", file=sys.stderr)
        return 0

    tasks_by_month: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        tasks_by_month.setdefault(month_id_for_task(task), []).append(task)

    total_wrote = 0
    total_failed = 0
    run_control = RunControl()
    for month_id in sorted(tasks_by_month):
        wrote, failed = run_month(
            month_id,
            tasks_by_month[month_id],
            wgrib2_path=wgrib2_path,
            output_dir=args.output_dir,
            summary_output_dir=args.summary_output_dir,
            download_dir=download_dir,
            reduced_dir=reduced_dir,
            max_workers=args.max_workers,
            download_workers=args.download_workers,
            reduce_workers=args.reduce_workers,
            extract_workers=args.extract_workers,
            reduce_queue_size=args.reduce_queue_size,
            extract_queue_size=args.extract_queue_size,
            keep_downloads=args.keep_downloads,
            keep_reduced=args.keep_reduced,
            range_merge_gap_bytes=max(0, int(args.range_merge_gap_bytes)),
            batch_reduce_mode=args.batch_reduce_mode,
            include_legacy_aliases=args.write_legacy_aliases,
            scratch_dir=scratch_dir,
            progress_mode=args.progress_mode,
            max_task_attempts=args.max_task_attempts,
            retry_backoff_seconds=args.retry_backoff_seconds,
            retry_max_backoff_seconds=args.retry_max_backoff_seconds,
            run_control=run_control,
            enable_dashboard_hotkeys=not bool(getattr(args, "disable_dashboard_hotkeys", False)),
            pause_control_file=getattr(args, "pause_control_file", None),
        )
        total_wrote += wrote
        total_failed += failed
        if run_control.is_paused():
            print(f"[paused] month={month_id} safe_to_exit=true", file=sys.stderr)
            return 0

    print(
        f"[done] wrote={total_wrote} failed={total_failed} "
        f"output_dir={args.output_dir} summary_output_dir={args.summary_output_dir}"
    )
    if total_failed and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
