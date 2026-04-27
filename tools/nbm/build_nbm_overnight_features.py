#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.progress import ProgressBar


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_FEATURES_ROOT = pathlib.Path("data/nbm/grib2")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "runtime" / "overnight"
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"
CHECKPOINT_HOURS = (6, 9, 12, 15, 18, 21)

CHECKPOINT_SPECS: tuple[tuple[int, str, str], ...] = (
    (6, "tmp", "nbm_temp_2m_06_local_k"),
    (9, "tmp", "nbm_temp_2m_09_local_k"),
    (12, "tmp", "nbm_temp_2m_12_local_k"),
    (15, "tmp", "nbm_temp_2m_15_local_k"),
    (18, "tmp", "nbm_temp_2m_18_local_k"),
    (21, "tmp", "nbm_temp_2m_21_local_k"),
    (9, "dpt", "nbm_dewpoint_2m_09_local_k"),
    (15, "dpt", "nbm_dewpoint_2m_15_local_k"),
    (9, "rh", "nbm_rh_2m_09_local_pct"),
    (15, "rh", "nbm_rh_2m_15_local_pct"),
    (9, "wind", "nbm_wind_10m_speed_09_local_ms"),
    (15, "wind", "nbm_wind_10m_speed_15_local_ms"),
    (9, "wdir", "nbm_wind_10m_direction_09_local_deg"),
    (15, "wdir", "nbm_wind_10m_direction_15_local_deg"),
)

REQUIRED_WIDE_INPUT_COLUMNS: tuple[str, ...] = (
    "init_time_utc",
    "valid_time_utc",
    "init_time_local",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "tmp",
    "dpt",
    "rh",
    "wind",
    "wdir",
    "tmax",
    "tmax_nb3_max",
    "tmax_nb7_max",
    "tmax_crop_max",
    "tmin",
    "tmin_nb3_min",
    "tmin_nb7_min",
    "tmin_crop_min",
    "gust",
    "tcdc",
    "dswrf",
    "apcp",
    "pcpdur",
    "vis",
    "ceil",
    "cape",
    "pwther",
    "tstm",
    "ptype",
    "thunc",
    "vrate",
    "tmp_nb3_mean",
    "tmp_nb7_mean",
    "tmp_crop_mean",
    "tcdc_crop_mean",
    "dswrf_crop_max",
    "wind_nb7_mean",
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "target_date_local",
    "selected_init_time_utc",
    "selected_init_time_local",
    "selection_cutoff_local",
    "selected_issue_age_minutes",
    "forecast_hour_min",
    "forecast_hour_max",
    "target_day_row_count",
    "coverage_complete",
    "missing_checkpoint_count",
    "missing_required_feature_count",
    "nbm_temp_2m_day_max_k",
    "nbm_temp_2m_day_mean_k",
    "nbm_native_tmax_2m_day_max_k",
    "nbm_native_tmax_2m_nb3_day_max_k",
    "nbm_native_tmax_2m_nb7_day_max_k",
    "nbm_native_tmax_2m_crop_day_max_k",
    "nbm_native_tmin_2m_day_min_k",
    "nbm_native_tmin_2m_nb3_day_min_k",
    "nbm_native_tmin_2m_nb7_day_min_k",
    "nbm_native_tmin_2m_crop_day_min_k",
    "nbm_gust_10m_day_max_ms",
    "nbm_tcdc_morning_mean_pct",
    "nbm_tcdc_day_mean_pct",
    "nbm_dswrf_day_max_w_m2",
    "nbm_apcp_day_total_kg_m2",
    "nbm_pcpdur_day_total_h",
    "nbm_pcpdur_day_max_h",
    "nbm_pcpdur_morning_total_h",
    "nbm_visibility_day_min_m",
    "nbm_ceiling_morning_min_m",
    "nbm_cape_day_max_j_kg",
    "nbm_pwther_code_day_mode",
    "nbm_pwther_nonzero_hour_count",
    "nbm_pwther_any_flag",
    "nbm_tstm_day_max_pct",
    "nbm_tstm_day_mean_pct",
    "nbm_tstm_any_flag",
    "nbm_ptype_code_day_mode",
    "nbm_ptype_nonzero_hour_count",
    "nbm_ptype_any_flag",
    "nbm_thunc_day_max_code",
    "nbm_thunc_day_mean_code",
    "nbm_thunc_nonzero_hour_count",
    "nbm_vrate_day_max",
    "nbm_vrate_day_mean",
    "nbm_temp_2m_nb3_day_mean_k",
    "nbm_temp_2m_nb7_day_mean_k",
    "nbm_temp_2m_crop_day_mean_k",
    "nbm_tcdc_crop_day_mean_pct",
    "nbm_dswrf_crop_day_max_w_m2",
    "nbm_wind_10m_speed_nb7_day_mean_ms",
    *(output for _, _, output in CHECKPOINT_SPECS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one-row-per-day NBM overnight summary features for KLGA.")
    parser.add_argument("--features-root", type=pathlib.Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME)
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def parse_local_time(value: str) -> dt.time:
    return dt.time.fromisoformat(value)


def iter_local_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    return [start_date + dt.timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def parse_local_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(NY_TZ)
    return timestamp.tz_convert(NY_TZ)


def parse_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def split_manifest_paths(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part for part in str(value).split(";") if part]


def partition_value_from_path(path: pathlib.Path, key: str) -> str | None:
    prefix = f"{key}="
    for part in path.parts:
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def discover_issue_catalog(features_root: pathlib.Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for manifest_path in sorted((features_root / "metadata" / "manifest").rglob("*.parquet")):
        manifest_df = pd.read_parquet(manifest_path)
        if manifest_df.empty:
            continue
        for _, manifest_row in manifest_df.iterrows():
            if str(manifest_row.get("extraction_status", "")).lower() != "ok":
                continue
            for wide_path in split_manifest_paths(manifest_row.get("wide_output_paths")):
                if wide_path in seen_paths:
                    continue
                path = pathlib.Path(wide_path)
                if not path.is_absolute():
                    repo_relative_path = (REPO_ROOT / path).resolve()
                    manifest_relative_path = (manifest_path.parent / path).resolve()
                    if repo_relative_path.exists():
                        path = repo_relative_path
                    else:
                        path = manifest_relative_path
                if not path.exists():
                    continue
                seen_paths.add(str(path))
                valid_date_local = partition_value_from_path(path, "valid_date_local") or str(manifest_row["valid_date_local"])
                rows.append(
                    {
                        "wide_path": str(path),
                        "source_model": manifest_row.get("source_model", "NBM"),
                        "source_product": manifest_row.get("source_product", "grib2-core"),
                        "source_version": manifest_row.get("source_version", "nbm-grib2-core-public"),
                        "station_id": manifest_row.get("station_id", "KLGA"),
                        "init_time_utc": parse_utc_timestamp(manifest_row["init_time_utc"]),
                        "init_time_local": parse_local_timestamp(manifest_row["init_time_local"]),
                        "processed_timestamp_utc": (
                            parse_utc_timestamp(manifest_row["processed_timestamp_utc"])
                            if manifest_row.get("processed_timestamp_utc") is not None and not pd.isna(manifest_row.get("processed_timestamp_utc"))
                            else pd.NaT
                        ),
                        "valid_date_local": valid_date_local,
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "wide_path",
                "source_model",
                "source_product",
                "source_version",
                "station_id",
                "init_time_utc",
                "init_time_local",
                "processed_timestamp_utc",
                "valid_date_local",
            ]
        )
    catalog = pd.DataFrame.from_records(rows)
    catalog["available_time_local"] = catalog["processed_timestamp_utc"].map(
        lambda value: parse_utc_timestamp(value).tz_convert(NY_TZ) if not pd.isna(value) else pd.NaT
    )
    fallback_mask = catalog["available_time_local"].isna()
    catalog.loc[fallback_mask, "available_time_local"] = catalog.loc[fallback_mask, "init_time_local"]
    return catalog.sort_values(["valid_date_local", "available_time_local", "init_time_local"]).reset_index(drop=True)


def selection_cutoff_timestamp(target_date_local: dt.date, cutoff_local_time: dt.time) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(target_date_local, cutoff_local_time, tzinfo=NY_TZ))


def select_issue_for_target(
    issue_catalog: pd.DataFrame,
    *,
    target_date_local: dt.date,
    cutoff_local_time: dt.time,
) -> tuple[pd.Series | None, pd.Timestamp]:
    cutoff_ts = selection_cutoff_timestamp(target_date_local, cutoff_local_time)
    catalog = issue_catalog.copy()
    if "available_time_local" not in catalog.columns:
        catalog["available_time_local"] = catalog["init_time_local"]
    eligible = catalog.loc[
        (catalog["valid_date_local"] == target_date_local.isoformat())
        & (catalog["available_time_local"] <= cutoff_ts)
    ].sort_values(["available_time_local", "init_time_local"])
    if eligible.empty:
        return None, cutoff_ts
    return eligible.iloc[-1], cutoff_ts


def available_wide_columns(wide_path: str | pathlib.Path) -> set[str]:
    return set(pq.read_schema(wide_path).names)


def load_issue_frame(wide_path: str | pathlib.Path) -> pd.DataFrame:
    wide_path = pathlib.Path(wide_path)
    available_columns = available_wide_columns(wide_path)
    projected_columns = [column for column in REQUIRED_WIDE_INPUT_COLUMNS if column in available_columns]
    df = pd.read_parquet(wide_path, columns=projected_columns).copy()
    if "init_time_utc" in df.columns:
        df["init_time_utc"] = df["init_time_utc"].map(parse_utc_timestamp)
    if "valid_time_utc" in df.columns:
        df["valid_time_utc"] = df["valid_time_utc"].map(parse_utc_timestamp)
    if "init_time_local" in df.columns:
        df["init_time_local"] = df["init_time_local"].map(parse_local_timestamp)
    if "valid_time_local" in df.columns:
        df["valid_time_local"] = df["valid_time_local"].map(parse_local_timestamp)
    return df.sort_values("valid_time_local").reset_index(drop=True)


def target_day_frame(issue_df: pd.DataFrame, *, target_date_local: dt.date) -> pd.DataFrame:
    if issue_df.empty:
        return issue_df.copy()
    return issue_df.loc[issue_df["valid_date_local"] == target_date_local.isoformat()].sort_values("valid_time_local").reset_index(drop=True)


def _float_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def summarize_hourly_rows(day_df: pd.DataFrame, column: str, reducer: str) -> float | None:
    if column not in day_df.columns or day_df.empty:
        return None
    values = pd.to_numeric(day_df[column], errors="coerce").dropna()
    if values.empty:
        return None
    if reducer == "max":
        return float(values.max())
    if reducer == "min":
        return float(values.min())
    if reducer == "mean":
        return float(values.mean())
    if reducer == "sum":
        return float(values.sum())
    raise ValueError(f"Unsupported reducer {reducer}")


def numeric_series(day_df: pd.DataFrame, column: str) -> pd.Series:
    if column not in day_df.columns or day_df.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(day_df[column], errors="coerce").dropna()


def observed_numeric(day_df: pd.DataFrame, column: str) -> bool:
    return not numeric_series(day_df, column).empty


def stable_numeric_mode(day_df: pd.DataFrame, column: str) -> float | None:
    values = numeric_series(day_df, column)
    if values.empty:
        return None
    counts = values.value_counts(dropna=True)
    max_count = counts.max()
    return float(counts[counts == max_count].index.min())


def nonzero_count(day_df: pd.DataFrame, column: str) -> int | None:
    values = numeric_series(day_df, column)
    if values.empty:
        return None
    return int((values != 0).sum())


def any_nonzero_flag(day_df: pd.DataFrame, column: str) -> float | None:
    count = nonzero_count(day_df, column)
    if count is None:
        return None
    return 1.0 if count > 0 else 0.0


def target_day_row_count(day_df: pd.DataFrame) -> int:
    if day_df.empty:
        return 0
    if "forecast_hour" not in day_df.columns:
        return int(len(day_df))
    forecast_hours = pd.to_numeric(day_df["forecast_hour"], errors="coerce").dropna()
    if forecast_hours.empty:
        return int(len(day_df))
    return int(forecast_hours.nunique())


def checkpoint_features(day_df: pd.DataFrame) -> tuple[dict[str, object], int]:
    outputs = {output: None for _, _, output in CHECKPOINT_SPECS}
    missing = 0
    if day_df.empty:
        return outputs, len(CHECKPOINT_SPECS)
    day_df = day_df.copy()
    day_df["valid_hour_local"] = pd.to_datetime(day_df["valid_time_local"]).dt.hour
    for hour, raw_column, output_name in CHECKPOINT_SPECS:
        row = day_df.loc[day_df["valid_hour_local"] == hour]
        if row.empty:
            missing += 1
            continue
        if raw_column in row.columns:
            non_null_values = pd.to_numeric(row[raw_column], errors="coerce").dropna()
            outputs[output_name] = None if non_null_values.empty else float(non_null_values.iloc[0])
        else:
            outputs[output_name] = None
        if outputs[output_name] is None:
            missing += 1
    return outputs, missing


def build_summary_row(issue_row: pd.Series, day_df: pd.DataFrame, *, target_date_local: dt.date, cutoff_ts: pd.Timestamp) -> dict[str, object]:
    morning_df = day_df.loc[pd.to_datetime(day_df["valid_time_local"]).dt.hour.between(6, 12)] if not day_df.empty else day_df
    checkpoint_values, missing_checkpoint_count = checkpoint_features(day_df)
    missing_required_feature_count = sum(1 for column in ("tmax", "tmin") if not observed_numeric(day_df, column))
    init_time_local = pd.Timestamp(issue_row["init_time_local"])
    row = {
        "source_model": issue_row["source_model"],
        "source_product": issue_row["source_product"],
        "source_version": issue_row["source_version"],
        "station_id": issue_row["station_id"],
        "target_date_local": target_date_local.isoformat(),
        "selected_init_time_utc": pd.Timestamp(issue_row["init_time_utc"]).isoformat(),
        "selected_init_time_local": init_time_local.isoformat(),
        "selection_cutoff_local": cutoff_ts.isoformat(),
        "selected_issue_age_minutes": float((cutoff_ts - init_time_local).total_seconds() / 60.0),
        "forecast_hour_min": _float_or_none(day_df["forecast_hour"].min()) if not day_df.empty else None,
        "forecast_hour_max": _float_or_none(day_df["forecast_hour"].max()) if not day_df.empty else None,
        "target_day_row_count": target_day_row_count(day_df),
        "coverage_complete": missing_checkpoint_count == 0 and missing_required_feature_count == 0,
        "missing_checkpoint_count": int(missing_checkpoint_count),
        "missing_required_feature_count": int(missing_required_feature_count),
        "nbm_temp_2m_day_max_k": summarize_hourly_rows(day_df, "tmp", "max"),
        "nbm_temp_2m_day_mean_k": summarize_hourly_rows(day_df, "tmp", "mean"),
        "nbm_native_tmax_2m_day_max_k": summarize_hourly_rows(day_df, "tmax", "max"),
        "nbm_native_tmax_2m_nb3_day_max_k": summarize_hourly_rows(day_df, "tmax_nb3_max", "max"),
        "nbm_native_tmax_2m_nb7_day_max_k": summarize_hourly_rows(day_df, "tmax_nb7_max", "max"),
        "nbm_native_tmax_2m_crop_day_max_k": summarize_hourly_rows(day_df, "tmax_crop_max", "max"),
        "nbm_native_tmin_2m_day_min_k": summarize_hourly_rows(day_df, "tmin", "min"),
        "nbm_native_tmin_2m_nb3_day_min_k": summarize_hourly_rows(day_df, "tmin_nb3_min", "min"),
        "nbm_native_tmin_2m_nb7_day_min_k": summarize_hourly_rows(day_df, "tmin_nb7_min", "min"),
        "nbm_native_tmin_2m_crop_day_min_k": summarize_hourly_rows(day_df, "tmin_crop_min", "min"),
        "nbm_gust_10m_day_max_ms": summarize_hourly_rows(day_df, "gust", "max"),
        "nbm_tcdc_morning_mean_pct": summarize_hourly_rows(morning_df, "tcdc", "mean"),
        "nbm_tcdc_day_mean_pct": summarize_hourly_rows(day_df, "tcdc", "mean"),
        "nbm_dswrf_day_max_w_m2": summarize_hourly_rows(day_df, "dswrf", "max"),
        "nbm_apcp_day_total_kg_m2": summarize_hourly_rows(day_df, "apcp", "sum"),
        "nbm_pcpdur_day_total_h": summarize_hourly_rows(day_df, "pcpdur", "sum"),
        "nbm_pcpdur_day_max_h": summarize_hourly_rows(day_df, "pcpdur", "max"),
        "nbm_pcpdur_morning_total_h": summarize_hourly_rows(morning_df, "pcpdur", "sum"),
        "nbm_visibility_day_min_m": summarize_hourly_rows(day_df, "vis", "min"),
        "nbm_ceiling_morning_min_m": summarize_hourly_rows(morning_df, "ceil", "min"),
        "nbm_cape_day_max_j_kg": summarize_hourly_rows(day_df, "cape", "max"),
        "nbm_pwther_code_day_mode": stable_numeric_mode(day_df, "pwther"),
        "nbm_pwther_nonzero_hour_count": nonzero_count(day_df, "pwther"),
        "nbm_pwther_any_flag": any_nonzero_flag(day_df, "pwther"),
        "nbm_tstm_day_max_pct": summarize_hourly_rows(day_df, "tstm", "max"),
        "nbm_tstm_day_mean_pct": summarize_hourly_rows(day_df, "tstm", "mean"),
        "nbm_tstm_any_flag": any_nonzero_flag(day_df, "tstm"),
        "nbm_ptype_code_day_mode": stable_numeric_mode(day_df, "ptype"),
        "nbm_ptype_nonzero_hour_count": nonzero_count(day_df, "ptype"),
        "nbm_ptype_any_flag": any_nonzero_flag(day_df, "ptype"),
        "nbm_thunc_day_max_code": summarize_hourly_rows(day_df, "thunc", "max"),
        "nbm_thunc_day_mean_code": summarize_hourly_rows(day_df, "thunc", "mean"),
        "nbm_thunc_nonzero_hour_count": nonzero_count(day_df, "thunc"),
        "nbm_vrate_day_max": summarize_hourly_rows(day_df, "vrate", "max"),
        "nbm_vrate_day_mean": summarize_hourly_rows(day_df, "vrate", "mean"),
        "nbm_temp_2m_nb3_day_mean_k": summarize_hourly_rows(day_df, "tmp_nb3_mean", "mean"),
        "nbm_temp_2m_nb7_day_mean_k": summarize_hourly_rows(day_df, "tmp_nb7_mean", "mean"),
        "nbm_temp_2m_crop_day_mean_k": summarize_hourly_rows(day_df, "tmp_crop_mean", "mean"),
        "nbm_tcdc_crop_day_mean_pct": summarize_hourly_rows(day_df, "tcdc_crop_mean", "mean"),
        "nbm_dswrf_crop_day_max_w_m2": summarize_hourly_rows(day_df, "dswrf_crop_max", "max"),
        "nbm_wind_10m_speed_nb7_day_mean_ms": summarize_hourly_rows(day_df, "wind_nb7_mean", "mean"),
    }
    row.update(checkpoint_values)
    return row


def output_paths(output_dir: pathlib.Path, target_date_local: dt.date) -> tuple[pathlib.Path, pathlib.Path]:
    root = output_dir / f"target_date_local={target_date_local.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    return root / "nbm.overnight.parquet", root / "nbm.overnight.manifest.parquet"


def failure_manifest_row(*, target_date_local: dt.date, cutoff_ts: pd.Timestamp, overnight_path: pathlib.Path, manifest_path: pathlib.Path) -> dict[str, object]:
    return {
        "source_model": "NBM",
        "source_product": "grib2-core",
        "source_version": "nbm-grib2-core-public",
        "station_id": "KLGA",
        "target_date_local": target_date_local.isoformat(),
        "selection_cutoff_local": cutoff_ts.isoformat(),
        "status": "no_qualifying_issue",
        "warning": "No qualifying NBM issue was available at or before the overnight local cutoff.",
        "overnight_output_path": str(overnight_path),
        "manifest_output_path": str(manifest_path),
    }


def success_manifest_row(*, row: dict[str, object], overnight_path: pathlib.Path, manifest_path: pathlib.Path) -> dict[str, object]:
    return {
        "source_model": row["source_model"],
        "source_product": row["source_product"],
        "source_version": row["source_version"],
        "station_id": row["station_id"],
        "target_date_local": row["target_date_local"],
        "selection_cutoff_local": row["selection_cutoff_local"],
        "status": "ok",
        "coverage_complete": row["coverage_complete"],
        "missing_checkpoint_count": row["missing_checkpoint_count"],
        "missing_required_feature_count": row["missing_required_feature_count"],
        "overnight_output_path": str(overnight_path),
        "manifest_output_path": str(manifest_path),
    }


def build_for_date(
    *,
    issue_catalog: pd.DataFrame,
    target_date_local: dt.date,
    cutoff_local_time: dt.time,
    output_dir: pathlib.Path,
    progress: ProgressBar | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    overnight_path, manifest_path = output_paths(output_dir, target_date_local)
    if progress is not None:
        progress.update(stage="select", status=f"target_date_local={target_date_local.isoformat()}")
    selected, cutoff_ts = select_issue_for_target(issue_catalog, target_date_local=target_date_local, cutoff_local_time=cutoff_local_time)
    if selected is None:
        if progress is not None:
            progress.update(stage="write", status=f"target_date_local={target_date_local.isoformat()} no_qualifying_issue")
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(overnight_path, index=False)
        pd.DataFrame.from_records(
            [failure_manifest_row(target_date_local=target_date_local, cutoff_ts=cutoff_ts, overnight_path=overnight_path, manifest_path=manifest_path)]
        ).to_parquet(manifest_path, index=False)
        return overnight_path, manifest_path

    if progress is not None:
        progress.update(stage="load", status=f"target_date_local={target_date_local.isoformat()} load_issue_frame")
    day_df = target_day_frame(load_issue_frame(selected["wide_path"]), target_date_local=target_date_local)
    if progress is not None:
        progress.update(stage="summarize", status=f"target_date_local={target_date_local.isoformat()} build_summary")
    row = build_summary_row(selected, day_df, target_date_local=target_date_local, cutoff_ts=cutoff_ts)
    if progress is not None:
        progress.update(stage="write", status=f"target_date_local={target_date_local.isoformat()} write_outputs")
    pd.DataFrame.from_records([row], columns=OUTPUT_COLUMNS).to_parquet(overnight_path, index=False)
    pd.DataFrame.from_records([success_manifest_row(row=row, overnight_path=overnight_path, manifest_path=manifest_path)]).to_parquet(manifest_path, index=False)
    return overnight_path, manifest_path


def build_range(
    *,
    features_root: pathlib.Path,
    output_dir: pathlib.Path,
    start_local_date: dt.date,
    end_local_date: dt.date,
    cutoff_local_time: dt.time,
    progress: ProgressBar | None = None,
) -> list[pathlib.Path]:
    issue_catalog = discover_issue_catalog(features_root)
    written: list[pathlib.Path] = []
    target_dates = iter_local_dates(start_local_date, end_local_date)
    for target_date_local in target_dates:
        overnight_path, manifest_path = build_for_date(
            issue_catalog=issue_catalog,
            target_date_local=target_date_local,
            cutoff_local_time=cutoff_local_time,
            output_dir=output_dir,
            progress=progress,
        )
        written.extend([overnight_path, manifest_path])
        if progress is not None:
            progress.advance(stage="complete", status=f"target_date_local={target_date_local.isoformat()} done")
    return written


def main() -> int:
    args = parse_args()
    target_dates = iter_local_dates(parse_local_date(args.start_local_date), parse_local_date(args.end_local_date))
    progress = ProgressBar(len(target_dates), label="NBM overnight", unit="date")
    written = build_range(
        features_root=args.features_root,
        output_dir=args.output_dir,
        start_local_date=parse_local_date(args.start_local_date),
        end_local_date=parse_local_date(args.end_local_date),
        cutoff_local_time=parse_local_time(args.cutoff_local_time),
        progress=progress,
    )
    progress.close(stage="finalize", status=f"wrote_files={len(written)}")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
