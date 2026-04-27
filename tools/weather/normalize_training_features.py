#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.canonical_feature_schema import (
    CANONICAL_PROVENANCE_COLUMNS,
    OPTIONAL_PASSTHROUGH_COLUMNS,
    RUNTIME_IDENTITY_COLUMNS,
    RUNTIME_METADATA_COLUMNS,
    RUNTIME_SPATIAL_COLUMNS,
    SPATIAL_STAT_SUFFIXES,
    canonical_base_for,
    canonical_wide_columns,
    source_base_mapping,
)


DEFAULT_OUTPUT_DIR = pathlib.Path("data/features/canonical")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw NBM or HRRR parquet outputs into the canonical training schema.")
    parser.add_argument("--source", choices=("nbm", "hrrr", "lamp"), required=True, help="Raw source schema to normalize.")
    parser.add_argument("--wide", type=pathlib.Path, required=True, help="Raw wide parquet input path.")
    parser.add_argument("--provenance", type=pathlib.Path, required=True, help="Raw provenance parquet input path.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory for canonical parquet outputs.",
    )
    return parser.parse_args()


def _initialize_frame(index: pd.Index, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(index=index, columns=columns)


def _copy_columns(source_df: pd.DataFrame, target_df: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in source_df.columns and column in target_df.columns:
            target_df[column] = source_df[column]


def _copy_metric_family(source_df: pd.DataFrame, target_df: pd.DataFrame, raw_base: str, canonical_base: str) -> None:
    for suffix in SPATIAL_STAT_SUFFIXES:
        raw_column = f"{raw_base}{suffix}"
        canonical_column = f"{canonical_base}{suffix}"
        if raw_column in source_df.columns and canonical_column in target_df.columns:
            target_df[canonical_column] = source_df[raw_column]


def _lamp_temperature_f_to_k(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return (numeric - 32.0) * (5.0 / 9.0) + 273.15


def _knots_to_ms(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric * 0.514444


def _hundreds_of_feet_to_m(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric * 100.0 * 0.3048


def _miles_to_m(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric * 1609.344


def _normalize_lamp_wide(source_df: pd.DataFrame) -> pd.DataFrame:
    normalized = _initialize_frame(source_df.index, canonical_wide_columns())
    _copy_columns(source_df, normalized, RUNTIME_IDENTITY_COLUMNS)
    _copy_columns(source_df, normalized, RUNTIME_METADATA_COLUMNS)
    _copy_columns(source_df, normalized, OPTIONAL_PASSTHROUGH_COLUMNS)

    if "tmp" in source_df.columns:
        normalized["temp_2m_k"] = _lamp_temperature_f_to_k(source_df["tmp"])
    if "dpt" in source_df.columns:
        normalized["dewpoint_2m_k"] = _lamp_temperature_f_to_k(source_df["dpt"])
    if "wdr" in source_df.columns:
        normalized["wind_10m_direction_deg"] = pd.to_numeric(source_df["wdr"], errors="coerce")
    if "wsp" in source_df.columns:
        normalized["wind_10m_speed_ms"] = _knots_to_ms(source_df["wsp"])
    if "wgs" in source_df.columns:
        normalized["gust_10m_ms"] = _knots_to_ms(source_df["wgs"])
    if "cig" in source_df.columns:
        normalized["ceiling_m"] = _hundreds_of_feet_to_m(source_df["cig"])
    if "vis" in source_df.columns:
        normalized["visibility_m"] = _miles_to_m(source_df["vis"])
    if "typ" in source_df.columns:
        normalized["ptype_code"] = source_df["typ"]

    return normalized.reset_index(drop=True)


def _normalize_wide(source_df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    if source == "lamp":
        return _normalize_lamp_wide(source_df)
    target_columns = canonical_wide_columns()
    normalized = _initialize_frame(source_df.index, target_columns)
    _copy_columns(source_df, normalized, RUNTIME_IDENTITY_COLUMNS)
    _copy_columns(source_df, normalized, RUNTIME_SPATIAL_COLUMNS)
    _copy_columns(source_df, normalized, RUNTIME_METADATA_COLUMNS)
    _copy_columns(source_df, normalized, OPTIONAL_PASSTHROUGH_COLUMNS)

    for raw_base, canonical_base in source_base_mapping(source).items():
        _copy_metric_family(source_df, normalized, raw_base, canonical_base)
        if raw_base in source_df.columns and canonical_base in normalized.columns:
            normalized[canonical_base] = source_df[raw_base]

    return normalized.reset_index(drop=True)


def _decode_feature_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]


def _encode_feature_list(feature_names: list[str]) -> str:
    return json.dumps(feature_names)


def _normalize_source_feature_names(source: str, series: pd.Series) -> pd.Series:
    def transform(value: object) -> str:
        mapped: list[str] = []
        for raw_name in _decode_feature_list(value):
            mapped.append(canonical_base_for(source, raw_name) or raw_name)
        return _encode_feature_list(mapped)

    return series.map(transform)

def _normalize_provenance(source_df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    normalized = _initialize_frame(source_df.index, list(CANONICAL_PROVENANCE_COLUMNS))
    identity_columns = (
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
    )
    _copy_columns(source_df, normalized, identity_columns)
    raw_feature_names = source_df["feature_name"] if "feature_name" in source_df.columns else pd.Series(pd.NA, index=source_df.index)
    normalized["raw_feature_name"] = raw_feature_names
    normalized["feature_name"] = raw_feature_names.map(lambda value: canonical_base_for(source, str(value)) if pd.notna(value) else pd.NA)
    normalized["feature_name"] = normalized["feature_name"].where(normalized["feature_name"].notna(), raw_feature_names)

    normalized["present_directly"] = (
        source_df["present_directly"].fillna(False).astype(bool)
        if "present_directly" in source_df.columns
        else pd.Series([True] * len(source_df), index=source_df.index, dtype=bool)
    )
    normalized["derived"] = (
        source_df["derived"].fillna(False).astype(bool)
        if "derived" in source_df.columns
        else pd.Series([False] * len(source_df), index=source_df.index, dtype=bool)
    )
    normalized["missing_optional"] = (
        source_df["missing_optional"].fillna(False).astype(bool)
        if "missing_optional" in source_df.columns
        else pd.Series([False] * len(source_df), index=source_df.index, dtype=bool)
    )
    normalized["derivation_method"] = source_df["derivation_method"] if "derivation_method" in source_df.columns else pd.NA
    if "source_feature_names" in source_df.columns:
        normalized["source_feature_names"] = _normalize_source_feature_names(source, source_df["source_feature_names"])
    else:
        normalized["source_feature_names"] = _encode_feature_list([])
    normalized["fallback_used"] = (
        source_df["fallback_used"].fillna(False).astype(bool)
        if "fallback_used" in source_df.columns
        else pd.Series([False] * len(source_df), index=source_df.index, dtype=bool)
    )
    normalized["fallback_source_description"] = (
        source_df["fallback_source_description"] if "fallback_source_description" in source_df.columns else pd.NA
    )
    normalized["grib_short_name"] = source_df["grib_short_name"] if "grib_short_name" in source_df.columns else pd.NA
    normalized["grib_level_text"] = source_df["grib_level_text"] if "grib_level_text" in source_df.columns else pd.NA
    normalized["grib_type_of_level"] = (
        source_df["grib_type_of_level"]
        if "grib_type_of_level" in source_df.columns
        else source_df["type_of_level"] if "type_of_level" in source_df.columns else pd.NA
    )
    normalized["grib_step_type"] = (
        source_df["grib_step_type"]
        if "grib_step_type" in source_df.columns
        else source_df["step_type"] if "step_type" in source_df.columns else pd.NA
    )
    normalized["grib_step_text"] = source_df["grib_step_text"] if "grib_step_text" in source_df.columns else pd.NA
    normalized["inventory_line"] = (
        source_df["inventory_line"]
        if "inventory_line" in source_df.columns
        else source_df["source_inventory_line"] if "source_inventory_line" in source_df.columns else pd.NA
    )
    normalized["units"] = source_df["units"] if "units" in source_df.columns else pd.NA
    normalized["notes"] = source_df["notes"] if "notes" in source_df.columns else pd.NA
    return normalized.reset_index(drop=True)


def normalize_nbm_wide_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_wide(df_raw, source="nbm")


def normalize_hrrr_wide_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_wide(df_raw, source="hrrr")


def normalize_nbm_provenance_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_provenance(df_raw, source="nbm")


def normalize_hrrr_provenance_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_provenance(df_raw, source="hrrr")


def normalize_lamp_wide_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_wide(df_raw, source="lamp")


def normalize_lamp_provenance_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    return _normalize_provenance(df_raw, source="lamp")


def _write_outputs(*, source: str, output_dir: pathlib.Path, wide_df: pd.DataFrame, provenance_df: pd.DataFrame) -> tuple[pathlib.Path, pathlib.Path]:
    target_dir = output_dir / f"source={source}"
    target_dir.mkdir(parents=True, exist_ok=True)
    wide_path = target_dir / "wide.parquet"
    provenance_path = target_dir / "provenance.parquet"
    wide_df.to_parquet(wide_path, index=False)
    provenance_df.to_parquet(provenance_path, index=False)
    return wide_path, provenance_path


def main() -> int:
    args = parse_args()
    wide_df = pd.read_parquet(args.wide)
    provenance_df = pd.read_parquet(args.provenance)
    if args.source == "nbm":
        canonical_wide = normalize_nbm_wide_to_canonical(wide_df)
        canonical_provenance = normalize_nbm_provenance_to_canonical(provenance_df)
    elif args.source == "hrrr":
        canonical_wide = normalize_hrrr_wide_to_canonical(wide_df)
        canonical_provenance = normalize_hrrr_provenance_to_canonical(provenance_df)
    else:
        canonical_wide = normalize_lamp_wide_to_canonical(wide_df)
        canonical_provenance = normalize_lamp_provenance_to_canonical(provenance_df)
    wide_path, provenance_path = _write_outputs(
        source=args.source,
        output_dir=args.output_dir,
        wide_df=canonical_wide,
        provenance_df=canonical_provenance,
    )
    print(f"[ok] source={args.source} wide={wide_path} provenance={provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
