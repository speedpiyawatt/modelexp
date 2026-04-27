#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import pathlib
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_grib2_features as nbm


TIMING_COLUMNS = (
    "timing_range_download_seconds",
    "timing_crop_seconds",
    "timing_cfgrib_open_seconds",
    "timing_row_build_seconds",
)

COMPARE_KEY = "valid_time_utc"
IGNORE_COMPARE_COLUMNS = {
    "requested_lead_hour",
}
DIAGNOSTIC_COMPARE_COLUMNS = {
    "fallback_used_any",
    "missing_optional_any",
    "missing_optional_fields_count",
}


@dataclass
class BatchResult:
    cycle_token: str
    init_time_utc: str
    lead_count: int
    separated_row_count: int
    separated_coalesced_row_count: int
    batch_row_count: int
    separated_crop_seconds: float
    separated_cfgrib_open_seconds: float
    separated_row_build_seconds: float
    separated_processing_seconds: float
    batch_concat_seconds: float
    batch_crop_seconds: float
    batch_cfgrib_open_seconds: float
    batch_row_build_seconds: float
    batch_processing_seconds: float
    crop_speedup_x: float | None
    crop_open_speedup_x: float | None
    processing_speedup_x: float | None
    row_count_match: bool
    value_mismatch_count: int
    core_value_mismatch_count: int
    diagnostic_mismatch_count: int
    value_mismatch_examples: list[dict[str, Any]]
    core_value_mismatch_examples: list[dict[str, Any]]
    diagnostic_mismatch_examples: list[dict[str, Any]]
    missing_batch_keys: int
    extra_batch_keys: int
    separated_coalesce_conflict_count: int
    separated_coalesce_conflict_examples: list[dict[str, Any]]
    cfgrib_open_all_dataset_count: int
    cfgrib_filtered_fallback_open_count: int
    cfgrib_filtered_fallback_attempt_count: int
    cfgrib_opened_dataset_count: int
    batch_raw_path: str
    batch_reduced_path: str
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current per-lead NBM crop/cfgrib processing against a local "
            "multi-lead selected-GRIB batch for the same selected records."
        )
    )
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    parser.add_argument("--selection-mode", default="overnight_0005", choices=nbm.SELECTION_MODES)
    parser.add_argument("--run-root", type=pathlib.Path, default=REPO_ROOT / "data" / "runtime" / "nbm_multilead_benchmark")
    parser.add_argument("--region", default="co", choices=["ak", "co", "gu", "hi", "pr"])
    parser.add_argument("--top", type=float, default=nbm.CROP_BOUNDS["top"])
    parser.add_argument("--bottom", type=float, default=nbm.CROP_BOUNDS["bottom"])
    parser.add_argument("--left", type=float, default=nbm.CROP_BOUNDS["left"])
    parser.add_argument("--right", type=float, default=nbm.CROP_BOUNDS["right"])
    parser.add_argument("--crop-method", default="small_grib", choices=nbm.CROP_METHODS)
    parser.add_argument("--crop-grib-type", default="complex3")
    parser.add_argument("--wgrib2-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1, help="Separated baseline cycle workers.")
    parser.add_argument("--lead-workers", type=int, default=1, help="Separated baseline lead workers.")
    parser.add_argument("--metric-profile", default="overnight", choices=("full", "overnight"))
    parser.add_argument(
        "--max-days",
        type=int,
        default=31,
        help="Safety cap for benchmark window length. Increase intentionally for larger validation runs.",
    )
    parser.add_argument("--reuse-separated", action="store_true", help="Reuse an existing separated baseline under --run-root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite benchmark output directories.")
    parser.add_argument("--keep-batch-gribs", action="store_true", help="Keep batch raw/reduced GRIBs after the benchmark.")
    parser.add_argument("--report-name", default="multi_lead_benchmark.json")
    return parser.parse_args()


def validate_date_window(args: argparse.Namespace) -> None:
    start = nbm.normalize_date(args.start_local_date)
    end = nbm.normalize_date(args.end_local_date)
    if end < start:
        raise SystemExit("--end-local-date must be on or after --start-local-date.")
    days = (end - start).days + 1
    max_days = max(1, int(getattr(args, "max_days", 31)))
    if days > max_days:
        raise SystemExit(f"Refusing {days}-day benchmark window because --max-days={max_days}.")


def run_separated_baseline(args: argparse.Namespace, *, output_dir: pathlib.Path, scratch_dir: pathlib.Path) -> dict[str, Any]:
    if output_dir.exists() and args.reuse_separated:
        return {"reused": True, "return_code": 0, "wall_seconds": 0.0, "command": []}
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(
            f"{output_dir} already exists. Pass --overwrite for a fresh separated baseline "
            "or --reuse-separated to benchmark existing kept raw files."
        )
    if args.overwrite:
        shutil.rmtree(output_dir, ignore_errors=True)
        shutil.rmtree(scratch_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "build_grib2_features.py"),
        "--start-local-date",
        args.start_local_date,
        "--end-local-date",
        args.end_local_date,
        "--selection-mode",
        args.selection_mode,
        "--region",
        args.region,
        "--top",
        str(args.top),
        "--bottom",
        str(args.bottom),
        "--left",
        str(args.left),
        "--right",
        str(args.right),
        "--output-dir",
        str(output_dir),
        "--scratch-dir",
        str(scratch_dir),
        "--workers",
        str(args.workers),
        "--lead-workers",
        str(args.lead_workers),
        "--crop-method",
        args.crop_method,
        "--crop-grib-type",
        args.crop_grib_type,
        "--wgrib2-threads",
        str(args.wgrib2_threads),
        "--metric-profile",
        args.metric_profile,
        "--keep-downloads",
        "--keep-reduced",
        "--skip-provenance",
        "--progress-mode",
        "log",
        "--overwrite",
    ]
    started_at = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, text=True)
    wall_seconds = round(time.perf_counter() - started_at, 6)
    if completed.returncode != 0:
        raise SystemExit(f"Separated baseline failed with return code {completed.returncode}.")
    return {"reused": False, "return_code": completed.returncode, "wall_seconds": wall_seconds, "command": command}


def load_parquet_glob(root: pathlib.Path, pattern: str) -> pd.DataFrame:
    paths = sorted(root.glob(pattern))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True, sort=False)


def manifest_path(value: Any) -> pathlib.Path:
    path = pathlib.Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def selected_fields_for_idx(idx_path: pathlib.Path) -> list[nbm.SelectedField]:
    idx_text = idx_path.read_text()
    records = nbm.parse_idx_lines(idx_text)
    selected, warnings, missing_required = nbm.select_inventory_records(records)
    if missing_required:
        raise ValueError(f"{idx_path} missing required selected records: {'; '.join(missing_required)}")
    if not selected:
        raise ValueError(f"{idx_path} did not select any records; warnings={warnings}")
    return selected


def combine_selected_fields(idx_paths: list[pathlib.Path]) -> list[nbm.SelectedField]:
    by_feature: dict[str, nbm.SelectedField] = {}
    for idx_path in idx_paths:
        for field in selected_fields_for_idx(idx_path):
            existing = by_feature.get(field.spec.feature_name)
            if existing is None:
                by_feature[field.spec.feature_name] = nbm.SelectedField(
                    spec=field.spec,
                    records=list(field.records),
                    fallback_used=field.fallback_used,
                    fallback_source_description=field.fallback_source_description,
                )
            else:
                existing.records.extend(field.records)
                existing.fallback_used = existing.fallback_used or field.fallback_used
                if not existing.fallback_source_description:
                    existing.fallback_source_description = field.fallback_source_description
    ordered: list[nbm.SelectedField] = []
    for spec in nbm.FIELD_SPECS:
        field = by_feature.get(spec.feature_name)
        if field is not None:
            ordered.append(field)
    return ordered


def cycle_plan_from_manifest(group: pd.DataFrame) -> nbm.CyclePlan:
    first = group.sort_values("lead_hour").iloc[0]
    init_time_utc = pd.Timestamp(first["init_time_utc"]).to_pydatetime()
    if init_time_utc.tzinfo is None:
        init_time_utc = init_time_utc.replace(tzinfo=nbm.UTC)
    init_time_utc = init_time_utc.astimezone(nbm.UTC)
    init_time_local = init_time_utc.astimezone(nbm.NY_TZ)
    return nbm.CyclePlan(
        init_time_utc=init_time_utc,
        init_time_local=init_time_local,
        cycle=f"{init_time_utc.hour:02d}",
        selected_target_dates=tuple(sorted(pd.to_datetime(group["valid_date_local"]).dt.date.unique())),
    )


def benchmark_args(args: argparse.Namespace, *, scratch_dir: pathlib.Path) -> argparse.Namespace:
    return argparse.Namespace(
        top=args.top,
        bottom=args.bottom,
        left=args.left,
        right=args.right,
        region=args.region,
        crop_method=args.crop_method,
        crop_grib_type=args.crop_grib_type,
        wgrib2_threads=args.wgrib2_threads,
        workers=1,
        lead_workers=1,
        reduce_workers=1,
        scratch_dir=scratch_dir,
        output_dir=args.run_root,
        keep_reduced=True,
        overwrite=True,
    )


def concatenate_files(paths: list[pathlib.Path], destination: pathlib.Path) -> float:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    with destination.open("wb") as out:
        for path in paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    return round(time.perf_counter() - started_at, 6)


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def values_equal(left: Any, right: Any) -> bool:
    left = normalize_value(left)
    right = normalize_value(right)
    if left is None and right is None:
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isfinite(left) and not np.isfinite(right):
            return True
        return abs(float(left) - float(right)) <= 1e-9
    return str(left) == str(right)


def coalesce_by_valid_time(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if df.empty or COMPARE_KEY not in df.columns:
        return df.copy(), 0
    rows: list[dict[str, Any]] = []
    conflict_count = 0
    for key, group in df.groupby(COMPARE_KEY, dropna=False, sort=True):
        row: dict[str, Any] = {COMPARE_KEY: key}
        for column in group.columns:
            if column == COMPARE_KEY:
                continue
            values = [normalize_value(value) for value in group[column].tolist()]
            present = [value for value in values if value is not None]
            if not present:
                row[column] = None
                continue
            first = present[0]
            if any(not values_equal(first, value) for value in present[1:]):
                conflict_count += 1
            row[column] = first
        rows.append(row)
    return pd.DataFrame.from_records(rows), conflict_count


def coalesce_by_valid_time_with_examples(df: pd.DataFrame) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    if df.empty or COMPARE_KEY not in df.columns:
        return df.copy(), 0, []
    rows: list[dict[str, Any]] = []
    conflict_count = 0
    examples: list[dict[str, Any]] = []
    for key, group in df.groupby(COMPARE_KEY, dropna=False, sort=True):
        row: dict[str, Any] = {COMPARE_KEY: key}
        for column in group.columns:
            if column == COMPARE_KEY:
                continue
            values = [normalize_value(value) for value in group[column].tolist()]
            present = [value for value in values if value is not None]
            if not present:
                row[column] = None
                continue
            first = present[0]
            if any(not values_equal(first, value) for value in present[1:]):
                conflict_count += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "valid_time_utc": str(key),
                            "column": column,
                            "values": [value for value in present[:10]],
                            "row_count": len(group),
                        }
                    )
            row[column] = first
        rows.append(row)
    return pd.DataFrame.from_records(rows), conflict_count, examples


def compare_wide_rows(separated_df: pd.DataFrame, batch_df: pd.DataFrame) -> dict[str, Any]:
    separated_coalesced, conflict_count, conflict_examples = coalesce_by_valid_time_with_examples(separated_df)
    batch_coalesced, _, _ = coalesce_by_valid_time_with_examples(batch_df)
    if separated_coalesced.empty or batch_coalesced.empty or COMPARE_KEY not in separated_coalesced.columns or COMPARE_KEY not in batch_coalesced.columns:
        return {
            "row_count_match": len(separated_coalesced) == len(batch_coalesced),
            "value_mismatch_count": 0 if len(separated_coalesced) == len(batch_coalesced) else 1,
            "core_value_mismatch_count": 0 if len(separated_coalesced) == len(batch_coalesced) else 1,
            "diagnostic_mismatch_count": 0,
            "value_mismatch_examples": [],
            "core_value_mismatch_examples": [],
            "diagnostic_mismatch_examples": [],
            "missing_batch_keys": len(separated_coalesced),
            "extra_batch_keys": len(batch_coalesced),
            "separated_coalesced_row_count": len(separated_coalesced),
            "separated_coalesce_conflict_count": conflict_count,
            "separated_coalesce_conflict_examples": conflict_examples,
        }
    separated_by_key = separated_coalesced.set_index(COMPARE_KEY, drop=False)
    batch_by_key = batch_coalesced.set_index(COMPARE_KEY, drop=False)
    separated_keys = set(separated_by_key.index)
    batch_keys = set(batch_by_key.index)
    common_keys = sorted(separated_keys & batch_keys)
    common_columns = sorted(
        (set(separated_coalesced.columns) & set(batch_coalesced.columns))
        - {COMPARE_KEY}
        - IGNORE_COMPARE_COLUMNS
    )
    mismatch_count = 0
    core_mismatch_count = 0
    diagnostic_mismatch_count = 0
    mismatch_examples: list[dict[str, Any]] = []
    core_mismatch_examples: list[dict[str, Any]] = []
    diagnostic_mismatch_examples: list[dict[str, Any]] = []
    for key in common_keys:
        left_row = separated_by_key.loc[key]
        right_row = batch_by_key.loc[key]
        if isinstance(left_row, pd.DataFrame):
            left_row = left_row.iloc[0]
        if isinstance(right_row, pd.DataFrame):
            right_row = right_row.iloc[0]
        for column in common_columns:
            if not values_equal(left_row[column], right_row[column]):
                mismatch_count += 1
                is_diagnostic = column in DIAGNOSTIC_COMPARE_COLUMNS
                if is_diagnostic:
                    diagnostic_mismatch_count += 1
                else:
                    core_mismatch_count += 1
                example = {
                    "valid_time_utc": str(key),
                    "column": column,
                    "separated": normalize_value(left_row[column]),
                    "batch": normalize_value(right_row[column]),
                }
                if len(mismatch_examples) < 20:
                    mismatch_examples.append(example)
                if is_diagnostic and len(diagnostic_mismatch_examples) < 20:
                    diagnostic_mismatch_examples.append(example)
                if not is_diagnostic and len(core_mismatch_examples) < 20:
                    core_mismatch_examples.append(example)
    missing = len(separated_keys - batch_keys)
    extra = len(batch_keys - separated_keys)
    return {
        "row_count_match": len(separated_coalesced) == len(batch_coalesced),
        "value_mismatch_count": mismatch_count,
        "core_value_mismatch_count": core_mismatch_count,
        "diagnostic_mismatch_count": diagnostic_mismatch_count,
        "value_mismatch_examples": mismatch_examples,
        "core_value_mismatch_examples": core_mismatch_examples,
        "diagnostic_mismatch_examples": diagnostic_mismatch_examples,
        "missing_batch_keys": missing,
        "extra_batch_keys": extra,
        "separated_coalesced_row_count": len(separated_coalesced),
        "separated_coalesce_conflict_count": conflict_count,
        "separated_coalesce_conflict_examples": conflict_examples,
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def expanded_2d_datasets(datasets: list[Any]) -> list[Any]:
    """Split cfgrib datasets with step/time dimensions into 2D variable slices.

    The production row builder intentionally selects index 0 for non-spatial
    dimensions. For this benchmark we need to test one crop/open over many
    leads, so a cfgrib variable shaped like (step, y, x) must be expanded into
    one 2D DataArray per step before delegating to the existing row builder.
    """
    expanded: list[Any] = []
    for dataset in datasets:
        for var_name, data_array in dataset.data_vars.items():
            spatial = set(nbm.spatial_dims(data_array))
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


def benchmark_batch_for_cycle(
    args: argparse.Namespace,
    *,
    cycle_token: str,
    group: pd.DataFrame,
    wide_df: pd.DataFrame,
    batch_dir: pathlib.Path,
) -> BatchResult:
    ordered = group.sort_values("lead_hour")
    raw_paths = [manifest_path(value) for value in ordered["raw_file_path"].tolist()]
    idx_paths = [manifest_path(value) for value in ordered["idx_file_path"].tolist()]
    missing_paths = [str(path) for path in raw_paths + idx_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"Kept separated raw/idx files are missing: {missing_paths[:5]}")

    cycle_plan = cycle_plan_from_manifest(ordered)
    selected = combine_selected_fields(idx_paths)
    cycle_dir = batch_dir / cycle_token
    shutil.rmtree(cycle_dir, ignore_errors=True)
    raw_path = cycle_dir / f"{cycle_token}.selected_multilead.grib2"
    reduced_path = cycle_dir / f"{cycle_token}.selected_multilead.subset.grib2"
    scratch_dir = cycle_dir / "scratch"
    crop_bounds = nbm.CropBounds(top=args.top, bottom=args.bottom, left=args.left, right=args.right)

    concat_seconds = concatenate_files(raw_paths, raw_path)
    crop_started_at = time.perf_counter()
    crop_result = nbm.crop_selected_grib2(
        args=benchmark_args(args, scratch_dir=scratch_dir),
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )
    crop_seconds = nbm.stage_elapsed_seconds(crop_started_at)

    datasets = []
    open_stats: dict[str, object] = {}
    try:
        open_started_at = time.perf_counter()
        datasets = nbm.open_grouped_datasets(
            reduced_path,
            index_dir=scratch_dir / "cfgrib_index",
            selected=selected,
            stats=open_stats,
        )
        open_seconds = nbm.stage_elapsed_seconds(open_started_at)
        row_started_at = time.perf_counter()
        expanded_datasets = expanded_2d_datasets(datasets)
        batch_rows, _, _ = nbm.build_rows_from_datasets(
            datasets=expanded_datasets,
            selected=selected,
            cycle_plan=cycle_plan,
            lead_hour=-1,
            crop_bounds=crop_bounds,
            allow_unknown_long_inference=True,
            write_long=False,
            write_provenance=False,
            metric_profile=args.metric_profile,
        )
        row_seconds = nbm.stage_elapsed_seconds(row_started_at)
    finally:
        for dataset in datasets:
            dataset.close()

    batch_df = pd.DataFrame.from_records(batch_rows)
    separated_cycle_df = wide_df[wide_df["init_time_utc"].astype(str) == cycle_plan.init_time_utc.isoformat()].copy()
    compare = compare_wide_rows(separated_cycle_df, batch_df)

    separated_crop = float(pd.to_numeric(ordered["timing_crop_seconds"], errors="coerce").fillna(0.0).sum())
    separated_open = float(pd.to_numeric(ordered["timing_cfgrib_open_seconds"], errors="coerce").fillna(0.0).sum())
    separated_row = float(pd.to_numeric(ordered["timing_row_build_seconds"], errors="coerce").fillna(0.0).sum())
    separated_processing = separated_crop + separated_open + separated_row
    batch_processing = crop_seconds + open_seconds + row_seconds

    if not args.keep_batch_gribs:
        raw_path.unlink(missing_ok=True)
        reduced_path.unlink(missing_ok=True)

    return BatchResult(
        cycle_token=cycle_token,
        init_time_utc=cycle_plan.init_time_utc.isoformat(),
        lead_count=len(ordered),
        separated_row_count=len(separated_cycle_df),
        separated_coalesced_row_count=int(compare["separated_coalesced_row_count"]),
        batch_row_count=len(batch_df),
        separated_crop_seconds=round(separated_crop, 6),
        separated_cfgrib_open_seconds=round(separated_open, 6),
        separated_row_build_seconds=round(separated_row, 6),
        separated_processing_seconds=round(separated_processing, 6),
        batch_concat_seconds=concat_seconds,
        batch_crop_seconds=crop_seconds,
        batch_cfgrib_open_seconds=open_seconds,
        batch_row_build_seconds=row_seconds,
        batch_processing_seconds=round(batch_processing, 6),
        crop_speedup_x=safe_ratio(separated_crop, crop_seconds),
        crop_open_speedup_x=safe_ratio(separated_crop + separated_open, crop_seconds + open_seconds),
        processing_speedup_x=safe_ratio(separated_processing, batch_processing),
        row_count_match=bool(compare["row_count_match"]),
        value_mismatch_count=int(compare["value_mismatch_count"]),
        core_value_mismatch_count=int(compare["core_value_mismatch_count"]),
        diagnostic_mismatch_count=int(compare["diagnostic_mismatch_count"]),
        value_mismatch_examples=list(compare["value_mismatch_examples"]),
        core_value_mismatch_examples=list(compare["core_value_mismatch_examples"]),
        diagnostic_mismatch_examples=list(compare["diagnostic_mismatch_examples"]),
        missing_batch_keys=int(compare["missing_batch_keys"]),
        extra_batch_keys=int(compare["extra_batch_keys"]),
        separated_coalesce_conflict_count=int(compare["separated_coalesce_conflict_count"]),
        separated_coalesce_conflict_examples=list(compare["separated_coalesce_conflict_examples"]),
        cfgrib_open_all_dataset_count=int(open_stats.get("cfgrib_open_all_dataset_count") or 0),
        cfgrib_filtered_fallback_open_count=int(open_stats.get("cfgrib_filtered_fallback_open_count") or 0),
        cfgrib_filtered_fallback_attempt_count=int(open_stats.get("cfgrib_filtered_fallback_attempt_count") or 0),
        cfgrib_opened_dataset_count=int(open_stats.get("cfgrib_opened_dataset_count") or 0),
        batch_raw_path=str(raw_path),
        batch_reduced_path=str(reduced_path),
        error=None,
    )


def summarize_results(results: list[BatchResult], separated_run: dict[str, Any]) -> dict[str, Any]:
    ok_results = [result for result in results if result.error is None]
    totals: dict[str, Any] = {
        "cycle_count": len(results),
        "ok_cycle_count": len(ok_results),
        "lead_count": sum(result.lead_count for result in ok_results),
        "separated_wall_seconds": separated_run["wall_seconds"],
        "separated_crop_seconds": round(sum(result.separated_crop_seconds for result in ok_results), 6),
        "separated_cfgrib_open_seconds": round(sum(result.separated_cfgrib_open_seconds for result in ok_results), 6),
        "separated_row_build_seconds": round(sum(result.separated_row_build_seconds for result in ok_results), 6),
        "separated_processing_seconds": round(sum(result.separated_processing_seconds for result in ok_results), 6),
        "batch_concat_seconds": round(sum(result.batch_concat_seconds for result in ok_results), 6),
        "batch_crop_seconds": round(sum(result.batch_crop_seconds for result in ok_results), 6),
        "batch_cfgrib_open_seconds": round(sum(result.batch_cfgrib_open_seconds for result in ok_results), 6),
        "batch_row_build_seconds": round(sum(result.batch_row_build_seconds for result in ok_results), 6),
        "batch_processing_seconds": round(sum(result.batch_processing_seconds for result in ok_results), 6),
        "value_mismatch_count": sum(result.value_mismatch_count for result in ok_results),
        "core_value_mismatch_count": sum(result.core_value_mismatch_count for result in ok_results),
        "diagnostic_mismatch_count": sum(result.diagnostic_mismatch_count for result in ok_results),
        "missing_batch_keys": sum(result.missing_batch_keys for result in ok_results),
        "extra_batch_keys": sum(result.extra_batch_keys for result in ok_results),
        "separated_coalesce_conflict_count": sum(result.separated_coalesce_conflict_count for result in ok_results),
    }
    totals["crop_speedup_x"] = safe_ratio(totals["separated_crop_seconds"], totals["batch_crop_seconds"])
    totals["crop_open_speedup_x"] = safe_ratio(
        totals["separated_crop_seconds"] + totals["separated_cfgrib_open_seconds"],
        totals["batch_crop_seconds"] + totals["batch_cfgrib_open_seconds"],
    )
    totals["processing_speedup_x"] = safe_ratio(totals["separated_processing_seconds"], totals["batch_processing_seconds"])
    return totals


def main() -> int:
    args = parse_args()
    validate_date_window(args)
    separated_output = args.run_root / "separated" / "output"
    separated_scratch = args.run_root / "separated" / "scratch"
    batch_dir = args.run_root / "batch"
    if args.overwrite:
        shutil.rmtree(batch_dir, ignore_errors=True)
    batch_dir.mkdir(parents=True, exist_ok=True)

    separated_run = run_separated_baseline(args, output_dir=separated_output, scratch_dir=separated_scratch)
    manifest_df = load_parquet_glob(separated_output, "metadata/manifest/**/*.parquet")
    if manifest_df.empty:
        raise SystemExit(f"No separated manifest parquet found under {separated_output}.")
    ok_manifest = manifest_df[manifest_df["extraction_status"] == "ok"].copy()
    if len(ok_manifest) != len(manifest_df):
        status_counts = manifest_df["extraction_status"].value_counts(dropna=False).to_dict()
        raise SystemExit(f"Separated baseline has non-ok lead rows; refusing partial benchmark: {status_counts}")
    if ok_manifest.empty:
        raise SystemExit("Separated baseline produced no ok manifest rows.")
    wide_df = load_parquet_glob(separated_output, "features/wide/**/*.parquet")
    if wide_df.empty:
        raise SystemExit(f"No separated wide parquet found under {separated_output}.")

    results: list[BatchResult] = []
    for init_time_utc, group in ok_manifest.groupby("init_time_utc", sort=True):
        cycle_token = pd.Timestamp(init_time_utc).strftime("%Y%m%dT%H%MZ")
        try:
            result = benchmark_batch_for_cycle(
                args,
                cycle_token=cycle_token,
                group=group,
                wide_df=wide_df,
                batch_dir=batch_dir,
            )
        except Exception as exc:
            result = BatchResult(
                cycle_token=cycle_token,
                init_time_utc=str(init_time_utc),
                lead_count=len(group),
                separated_row_count=0,
                separated_coalesced_row_count=0,
                batch_row_count=0,
                separated_crop_seconds=0.0,
                separated_cfgrib_open_seconds=0.0,
                separated_row_build_seconds=0.0,
                separated_processing_seconds=0.0,
                batch_concat_seconds=0.0,
                batch_crop_seconds=0.0,
                batch_cfgrib_open_seconds=0.0,
                batch_row_build_seconds=0.0,
                batch_processing_seconds=0.0,
                crop_speedup_x=None,
                crop_open_speedup_x=None,
                processing_speedup_x=None,
                row_count_match=False,
                value_mismatch_count=0,
                core_value_mismatch_count=0,
                diagnostic_mismatch_count=0,
                value_mismatch_examples=[],
                core_value_mismatch_examples=[],
                diagnostic_mismatch_examples=[],
                missing_batch_keys=0,
                extra_batch_keys=0,
                separated_coalesce_conflict_count=0,
                separated_coalesce_conflict_examples=[],
                cfgrib_open_all_dataset_count=0,
                cfgrib_filtered_fallback_open_count=0,
                cfgrib_filtered_fallback_attempt_count=0,
                cfgrib_opened_dataset_count=0,
                batch_raw_path="",
                batch_reduced_path="",
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)

    report = {
        "config": {
            "start_local_date": args.start_local_date,
            "end_local_date": args.end_local_date,
            "selection_mode": args.selection_mode,
            "crop_method": args.crop_method,
            "crop_grib_type": args.crop_grib_type,
            "wgrib2_threads": args.wgrib2_threads,
            "metric_profile": args.metric_profile,
            "separated_workers": args.workers,
            "separated_lead_workers": args.lead_workers,
        },
        "separated_run": separated_run,
        "summary": summarize_results(results, separated_run),
        "cycles": [asdict(result) for result in results],
    }
    report_path = args.run_root / args.report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"report_path={report_path}")
    return 0 if all(result.error is None for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
