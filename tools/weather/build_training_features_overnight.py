#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import tempfile
import sys
from zoneinfo import ZoneInfo

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.training_features_overnight_contract import REGISTRY_ROWS, registry_by_name, registry_columns


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_LABELS_PATH = pathlib.Path("wunderground/output/tables/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("wunderground/output/tables/wu_obs_intraday.parquet")
DEFAULT_NBM_ROOT = pathlib.Path("tools/nbm/data/runtime/overnight")
DEFAULT_LAMP_ROOT = pathlib.Path("tools/lamp/data/runtime/overnight")
DEFAULT_HRRR_ROOT = pathlib.Path("data/runtime/features/overnight_summary_klga")
DEFAULT_OUTPUT_PATH = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight")
DEFAULT_STATION_ID = "KLGA"
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the merged overnight KLGA training table from frozen source-aware daily contracts.")
    parser.add_argument("--labels-path", type=pathlib.Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument(
        "--label-history-path",
        type=pathlib.Path,
        default=None,
        help="Optional labels_daily parquet with support history for previous-day WU features. Defaults to --labels-path.",
    )
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--nbm-root", type=pathlib.Path, default=DEFAULT_NBM_ROOT)
    parser.add_argument("--lamp-root", type=pathlib.Path, default=DEFAULT_LAMP_ROOT)
    parser.add_argument("--hrrr-root", type=pathlib.Path, default=DEFAULT_HRRR_ROOT)
    parser.add_argument("--output-path", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--manifest-output-path", type=pathlib.Path, default=None)
    parser.add_argument("--start-local-date", default=None)
    parser.add_argument("--end-local-date", default=None)
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME)
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty required inputs and emit an empty schema-only output.")
    return parser.parse_args()


def _read_parquet(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _require_frame(name: str, path: pathlib.Path, df: pd.DataFrame, *, allow_empty: bool) -> None:
    if allow_empty:
        return
    if not path.exists():
        raise FileNotFoundError(f"{name} input does not exist: {path}")
    if df.empty:
        raise ValueError(f"{name} input is empty: {path}")


def _read_named_parquets(root: pathlib.Path, filename: str) -> pd.DataFrame:
    if root.is_file():
        return _read_parquet(root)
    if not root.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(root.rglob(filename))]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _read_hrrr_summaries(root: pathlib.Path) -> pd.DataFrame:
    if root.is_file():
        return _read_parquet(root)
    if not root.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(root.rglob("*.parquet")) if not path.name.endswith(".manifest.parquet")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def iter_local_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    return [start_date + dt.timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def filter_target_date_window(df: pd.DataFrame, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if df.empty or "target_date_local" not in df.columns:
        return df.copy()
    start_token = start_date.isoformat()
    end_token = end_date.isoformat()
    return df.loc[(df["target_date_local"] >= start_token) & (df["target_date_local"] <= end_token)].reset_index(drop=True)


def output_paths_for_date(output_dir: pathlib.Path, target_date_local: str) -> tuple[pathlib.Path, pathlib.Path]:
    root = output_dir / f"target_date_local={target_date_local}"
    return root / "part.parquet", root / "manifest.json"


def _write_atomic_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp_path = pathlib.Path(handle.name)
    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_atomic_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w") as handle:
        temp_path = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def build_output_manifest_for_date(
    *,
    target_date_local: str,
    output_path: pathlib.Path,
    row_count: int,
    station_id: str,
    cutoff_local_time: str,
    output_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "status": "ok",
        "target_date_local": target_date_local,
        "station_id": station_id,
        "cutoff_local_time": cutoff_local_time,
        "row_count": int(row_count),
        "column_count": int(len(output_df.columns)),
        "output_path": str(output_path),
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def validate_date_output(
    *,
    output_path: pathlib.Path,
    manifest_path: pathlib.Path,
    target_date_local: str,
    allow_empty: bool,
) -> bool:
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        df = pd.read_parquet(output_path)
    except Exception:
        return False
    if manifest.get("status") != "ok":
        return False
    if manifest.get("target_date_local") != target_date_local:
        return False
    if not allow_empty and df.empty:
        return False
    if int(manifest.get("row_count", -1)) != int(len(df)):
        return False
    if list(df.columns) != registry_columns():
        return False
    if df.empty:
        return allow_empty
    if len(df) != 1:
        return False
    target_dates = df["target_date_local"].astype(str)
    if not (target_dates == target_date_local).all():
        return False
    return True


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    return series.astype("boolean").fillna(False)


def _numeric_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def kelvin_to_f(value: object) -> float | None:
    numeric = _numeric_or_none(value)
    if numeric is None:
        return None
    return (numeric - 273.15) * (9.0 / 5.0) + 32.0


def cutoff_timestamp(target_date_local: str, cutoff_local_time: str) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(dt.date.fromisoformat(target_date_local), dt.time.fromisoformat(cutoff_local_time), tzinfo=NY_TZ))


def _window_stats(history: pd.DataFrame, *, cutoff_ts: pd.Timestamp, hours: int) -> dict[str, float | None]:
    recent = history.loc[(history["valid_time_local"] > cutoff_ts - pd.Timedelta(hours=hours)) & (history["valid_time_local"] <= cutoff_ts)]
    compare = history.loc[history["valid_time_local"] <= cutoff_ts - pd.Timedelta(hours=hours)]
    compare_row = compare.iloc[-1] if not compare.empty else None
    last_row = history.iloc[-1] if not history.empty else None
    result: dict[str, float | None] = {
        f"wu_temp_change_{hours}h_f": None,
    }
    if last_row is not None and compare_row is not None:
        last_temp = _numeric_or_none(last_row.get("temp_f"))
        compare_temp = _numeric_or_none(compare_row.get("temp_f"))
        if last_temp is not None and compare_temp is not None:
            result[f"wu_temp_change_{hours}h_f"] = float(last_temp - compare_temp)
    if hours == 3:
        if last_row is not None and compare_row is not None:
            last_dewpoint = _numeric_or_none(last_row.get("dewpoint_f"))
            compare_dewpoint = _numeric_or_none(compare_row.get("dewpoint_f"))
            if last_dewpoint is not None and compare_dewpoint is not None:
                result["wu_dewpoint_change_3h_f"] = float(last_dewpoint - compare_dewpoint)
            last_pressure = _numeric_or_none(last_row.get("pressure_in"))
            compare_pressure = _numeric_or_none(compare_row.get("pressure_in"))
            if last_pressure is not None and compare_pressure is not None:
                result["wu_pressure_change_3h"] = float(last_pressure - compare_pressure)
        result["wu_wind_speed_mean_3h"] = _numeric_or_none(recent["wind_speed_mph"].mean()) if not recent.empty else None
    if hours == 6:
        result["wu_wind_gust_max_6h"] = _numeric_or_none(recent["wind_gust_mph"].max()) if not recent.empty else None
        result["wu_visibility_min_6h"] = _numeric_or_none(recent["visibility"].min()) if not recent.empty else None
        precip = pd.to_numeric(recent["precip_hrly_in"], errors="coerce").fillna(0.0) if not recent.empty else pd.Series(dtype="float64")
        result["wu_precip_total_6h"] = float(precip.sum()) if not precip.empty else None
    return result


def build_wu_cutoff_features(
    labels_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    *,
    cutoff_local_time: str,
    label_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if labels_df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id"])
    if obs_df.empty:
        rows = []
        for _, label_row in labels_df.iterrows():
            rows.append(
                {
                    "target_date_local": label_row["target_date_local"],
                    "station_id": label_row["station_id"],
                    "meta_wu_obs_available": False,
                    "meta_wu_last_obs_time_local": None,
                }
            )
        return pd.DataFrame.from_records(rows)

    obs = obs_df.copy()
    obs["valid_time_local"] = pd.to_datetime(obs["valid_time_local"], utc=True).dt.tz_convert(NY_TZ)
    obs = obs.sort_values(["station_id", "valid_time_local"]).reset_index(drop=True)
    effective_history_df = label_history_df if label_history_df is not None and not label_history_df.empty else labels_df
    labels_by_key = effective_history_df.set_index(["target_date_local", "station_id"])
    rows: list[dict[str, object]] = []

    for _, label_row in labels_df.sort_values(["station_id", "target_date_local"]).iterrows():
        target_date_local = str(label_row["target_date_local"])
        station_id = str(label_row["station_id"])
        cutoff_ts = cutoff_timestamp(target_date_local, cutoff_local_time)
        station_obs = obs.loc[(obs["station_id"] == station_id) & (obs["valid_time_local"] <= cutoff_ts)].copy()
        row: dict[str, object] = {
            "target_date_local": target_date_local,
            "station_id": station_id,
            "meta_wu_obs_available": not station_obs.empty,
            "meta_wu_last_obs_time_local": None,
            "wu_last_temp_f": None,
            "wu_last_dewpoint_f": None,
            "wu_last_rh_pct": None,
            "wu_last_pressure_in": None,
            "wu_last_wind_speed_mph": None,
            "wu_last_wind_dir_deg": None,
            "wu_last_wind_gust_mph": None,
            "wu_last_visibility": None,
            "wu_last_cloud_cover_code": None,
            "wu_last_wx_phrase": None,
            "wu_last_precip_hrly_in": None,
            "wu_prev_day_final_tmax_f": None,
            "wu_prev_day_final_tmin_f": None,
            "wu_prev_day_total_precip_in": None,
            "wu_temp_change_1h_f": None,
            "wu_temp_change_3h_f": None,
            "wu_temp_change_6h_f": None,
            "wu_dewpoint_change_3h_f": None,
            "wu_pressure_change_3h": None,
            "wu_wind_speed_mean_3h": None,
            "wu_wind_gust_max_6h": None,
            "wu_visibility_min_6h": None,
            "wu_precip_total_6h": None,
        }
        if not station_obs.empty:
            last_obs = station_obs.iloc[-1]
            row.update(
                {
                    "meta_wu_last_obs_time_local": last_obs["valid_time_local"].isoformat(),
                    "wu_last_temp_f": _numeric_or_none(last_obs.get("temp_f")),
                    "wu_last_dewpoint_f": _numeric_or_none(last_obs.get("dewpoint_f")),
                    "wu_last_rh_pct": _numeric_or_none(last_obs.get("rh_pct")),
                    "wu_last_pressure_in": _numeric_or_none(last_obs.get("pressure_in")),
                    "wu_last_wind_speed_mph": _numeric_or_none(last_obs.get("wind_speed_mph")),
                    "wu_last_wind_dir_deg": _numeric_or_none(last_obs.get("wind_dir_deg")),
                    "wu_last_wind_gust_mph": _numeric_or_none(last_obs.get("wind_gust_mph")),
                    "wu_last_visibility": _numeric_or_none(last_obs.get("visibility")),
                    "wu_last_cloud_cover_code": last_obs.get("cloud_cover_code"),
                    "wu_last_wx_phrase": last_obs.get("wx_phrase"),
                    "wu_last_precip_hrly_in": _numeric_or_none(last_obs.get("precip_hrly_in")),
                }
            )
            for hours in (1, 3, 6):
                row.update(_window_stats(station_obs, cutoff_ts=cutoff_ts, hours=hours))

        previous_date_local = (dt.date.fromisoformat(target_date_local) - dt.timedelta(days=1)).isoformat()
        previous_key = (previous_date_local, station_id)
        if previous_key in labels_by_key.index:
            previous = labels_by_key.loc[previous_key]
            if isinstance(previous, pd.DataFrame):
                previous = previous.iloc[0]
            row["wu_prev_day_final_tmax_f"] = _numeric_or_none(previous.get("label_final_tmax_f"))
            row["wu_prev_day_final_tmin_f"] = _numeric_or_none(previous.get("label_final_tmin_f"))
            row["wu_prev_day_total_precip_in"] = _numeric_or_none(previous.get("label_total_precip_in"))
        rows.append(row)

    return pd.DataFrame.from_records(rows)


def transform_nbm_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id", "meta_nbm_available"])
    rename_map = {
        "source_model": "meta_nbm_source_model",
        "source_product": "meta_nbm_source_product",
        "source_version": "meta_nbm_source_version",
        "selected_init_time_utc": "meta_nbm_selected_init_time_utc",
        "selected_init_time_local": "meta_nbm_selected_init_time_local",
        "selected_issue_age_minutes": "meta_nbm_selected_issue_age_minutes",
        "target_day_row_count": "meta_nbm_target_day_row_count",
        "missing_checkpoint_count": "meta_nbm_missing_checkpoint_count",
        "missing_required_feature_count": "meta_nbm_missing_required_feature_count",
        "coverage_complete": "meta_nbm_coverage_complete",
    }
    out = df.rename(columns=rename_map).copy()
    out["meta_nbm_available"] = True
    keep = ["target_date_local", "station_id", "meta_nbm_available", *rename_map.values(), *[column for column in out.columns if column.startswith("nbm_")]]
    return out.loc[:, [column for column in keep if column in out.columns]].drop_duplicates(subset=["target_date_local", "station_id"])


def transform_lamp_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id", "meta_lamp_available"])
    rename_map = {
        "source_model": "meta_lamp_source_model",
        "source_product": "meta_lamp_source_product",
        "source_version": "meta_lamp_source_version",
        "selected_init_time_utc": "meta_lamp_selected_init_time_utc",
        "selected_init_time_local": "meta_lamp_selected_init_time_local",
        "previous_init_time_utc": "meta_lamp_previous_init_time_utc",
        "previous_init_time_local": "meta_lamp_previous_init_time_local",
        "revision_available": "meta_lamp_revision_available",
        "missing_optional_any": "meta_lamp_missing_optional_any",
        "missing_optional_fields_count": "meta_lamp_missing_optional_fields_count",
        "coverage_complete": "meta_lamp_coverage_complete",
        "missing_checkpoint_count": "meta_lamp_missing_checkpoint_count",
    }
    out = df.rename(columns=rename_map).copy()
    out["meta_lamp_available"] = True
    feature_columns = []
    for column in out.columns:
        if column in {"target_date_local", "station_id", "selection_cutoff_local", *rename_map.values(), "meta_lamp_available"}:
            continue
        feature_columns.append(f"lamp_{column}")
    feature_rename = {
        column: f"lamp_{column}"
        for column in out.columns
        if column not in {"target_date_local", "station_id", "selection_cutoff_local", *rename_map.values(), "meta_lamp_available"}
    }
    out = out.rename(columns=feature_rename)
    keep = ["target_date_local", "station_id", "meta_lamp_available", *rename_map.values(), *feature_columns]
    return out.loc[:, [column for column in keep if column in out.columns]].drop_duplicates(subset=["target_date_local", "station_id"])


def transform_hrrr_daily(df: pd.DataFrame, *, station_id: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id", "meta_hrrr_available"])
    out = df.copy()
    out["station_id"] = out.get("station_id", station_id)
    out["meta_hrrr_available"] = True
    out["meta_hrrr_source_model"] = out.get("source_model", "HRRR")
    out["meta_hrrr_source_product"] = out.get("source_product", "wrfsfcf")
    out["meta_hrrr_source_version"] = out.get("source_version", "hrrr-conus-wrfsfcf-public")
    rename_map = {
        "anchor_init_time_utc": "meta_hrrr_anchor_init_time_utc",
        "anchor_init_time_local": "meta_hrrr_anchor_init_time_local",
        "retained_cycle_count": "meta_hrrr_retained_cycle_count",
        "first_valid_hour_local": "meta_hrrr_first_valid_hour_local",
        "last_valid_hour_local": "meta_hrrr_last_valid_hour_local",
        "covered_hour_count": "meta_hrrr_covered_hour_count",
        "covered_checkpoint_count": "meta_hrrr_covered_checkpoint_count",
        "coverage_end_hour_local": "meta_hrrr_coverage_end_hour_local",
        "has_full_day_21_local_coverage": "meta_hrrr_has_full_day_21_local_coverage",
        "missing_checkpoint_count": "meta_hrrr_missing_checkpoint_count",
    }
    out = out.rename(columns=rename_map)
    keep = [
        "target_date_local",
        "station_id",
        "meta_hrrr_available",
        "meta_hrrr_source_model",
        "meta_hrrr_source_product",
        "meta_hrrr_source_version",
        *rename_map.values(),
        *[column for column in out.columns if column.startswith("hrrr_")],
    ]
    return out.loc[:, [column for column in keep if column in out.columns]].drop_duplicates(subset=["target_date_local", "station_id"])


def apply_registry_layout(df: pd.DataFrame) -> pd.DataFrame:
    registry = registry_by_name()
    ordered = df.reindex(columns=registry_columns(), fill_value=pd.NA).copy()
    for column, spec in registry.items():
        if spec["dtype"] == "bool":
            ordered[column] = _coerce_bool_series(ordered[column])
    return ordered


def build_training_features_overnight(
    *,
    labels_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    nbm_daily_df: pd.DataFrame,
    lamp_daily_df: pd.DataFrame,
    hrrr_daily_df: pd.DataFrame,
    cutoff_local_time: str,
    station_id: str,
    label_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if labels_df.empty:
        return apply_registry_layout(pd.DataFrame(columns=registry_columns()))

    base = labels_df.copy()
    if "station_id" in base.columns:
        base = base.loc[base["station_id"].astype(str) == str(station_id)].reset_index(drop=True)
    if base.empty:
        return apply_registry_layout(pd.DataFrame(columns=registry_columns()))
    if label_history_df is not None and not label_history_df.empty and "station_id" in label_history_df.columns:
        label_history_df = label_history_df.loc[label_history_df["station_id"].astype(str) == str(station_id)].reset_index(drop=True)
    if not obs_df.empty and "station_id" in obs_df.columns:
        obs_df = obs_df.loc[obs_df["station_id"].astype(str) == str(station_id)].reset_index(drop=True)

    base["target_date_local"] = base["target_date_local"].astype(str)
    base["station_id"] = base["station_id"].astype(str)
    base["selection_cutoff_local"] = base["target_date_local"].map(lambda value: cutoff_timestamp(value, cutoff_local_time).isoformat())

    merged = base.merge(
        build_wu_cutoff_features(
            base,
            obs_df,
            cutoff_local_time=cutoff_local_time,
            label_history_df=label_history_df,
        ),
        on=["target_date_local", "station_id"],
        how="left",
    )
    merged = merged.merge(transform_nbm_daily(nbm_daily_df), on=["target_date_local", "station_id"], how="left")
    merged = merged.merge(transform_lamp_daily(lamp_daily_df), on=["target_date_local", "station_id"], how="left")
    merged = merged.merge(transform_hrrr_daily(hrrr_daily_df, station_id=station_id), on=["target_date_local", "station_id"], how="left")

    for column in ("meta_nbm_available", "meta_lamp_available", "meta_hrrr_available", "meta_wu_obs_available", "meta_lamp_revision_available", "meta_lamp_missing_optional_any", "meta_nbm_coverage_complete", "meta_lamp_coverage_complete", "meta_hrrr_has_full_day_21_local_coverage"):
        if column in merged.columns:
            merged[column] = merged[column].astype("boolean").fillna(False)

    nbm_tmax_k = pd.to_numeric(merged.get("nbm_temp_2m_day_max_k"), errors="coerce")
    lamp_tmax_f = pd.to_numeric(merged.get("lamp_day_tmp_max_f_forecast"), errors="coerce")
    hrrr_tmax_k = pd.to_numeric(merged.get("hrrr_temp_2m_day_max_k"), errors="coerce")
    merged["nbm_minus_lamp_day_max_f"] = ((nbm_tmax_k - 273.15) * (9.0 / 5.0) + 32.0) - lamp_tmax_f
    merged["nbm_minus_hrrr_day_max_k"] = nbm_tmax_k - hrrr_tmax_k

    return apply_registry_layout(merged.drop_duplicates(subset=["target_date_local", "station_id"]).reset_index(drop=True))


def main() -> int:
    args = parse_args()
    if args.output_dir is not None and args.output_path != DEFAULT_OUTPUT_PATH:
        raise ValueError("Use either --output-path or --output-dir, not both.")
    if (args.start_local_date is None) ^ (args.end_local_date is None):
        raise ValueError("Provide both --start-local-date and --end-local-date when using daily output mode.")
    labels_df = _read_parquet(args.labels_path)
    label_history_path = args.label_history_path or args.labels_path
    label_history_df = _read_parquet(label_history_path)
    obs_df = _read_parquet(args.obs_path)
    nbm_daily_df = _read_named_parquets(args.nbm_root, "nbm.overnight.parquet")
    lamp_daily_df = _read_named_parquets(args.lamp_root, "lamp.overnight.parquet")
    hrrr_daily_df = _read_hrrr_summaries(args.hrrr_root)
    _require_frame("labels_daily", args.labels_path, labels_df, allow_empty=args.allow_empty)
    _require_frame("label_history", label_history_path, label_history_df, allow_empty=args.allow_empty)
    _require_frame("wu_obs_intraday", args.obs_path, obs_df, allow_empty=args.allow_empty)

    if args.output_dir is None:
        output_df = build_training_features_overnight(
            labels_df=labels_df,
            label_history_df=label_history_df,
            obs_df=obs_df,
            nbm_daily_df=nbm_daily_df,
            lamp_daily_df=lamp_daily_df,
            hrrr_daily_df=hrrr_daily_df,
            cutoff_local_time=args.cutoff_local_time,
            station_id=args.station_id,
        )
        if output_df.empty and not args.allow_empty:
            raise ValueError("training_features_overnight build produced zero rows; rerun with --allow-empty only for schema/bootstrap workflows")
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(args.output_path, index=False)
        print(args.output_path)
        return 0

    start_date = parse_local_date(args.start_local_date)
    end_date = parse_local_date(args.end_local_date)
    if end_date < start_date:
        raise ValueError("--end-local-date must be on or after --start-local-date")

    written: list[pathlib.Path] = []
    for target_date in iter_local_dates(start_date, end_date):
        target_token = target_date.isoformat()
        day_labels_df = filter_target_date_window(labels_df, start_date=target_date, end_date=target_date)
        day_label_history_df = filter_target_date_window(label_history_df, start_date=target_date - dt.timedelta(days=1), end_date=target_date)
        day_obs_df = obs_df.copy()
        if not day_obs_df.empty and "date_local" in day_obs_df.columns:
            day_obs_df = day_obs_df.loc[
                (day_obs_df["date_local"] >= (target_date - dt.timedelta(days=1)).isoformat())
                & (day_obs_df["date_local"] <= target_token)
            ].reset_index(drop=True)
        day_nbm_daily_df = filter_target_date_window(nbm_daily_df, start_date=target_date, end_date=target_date)
        day_lamp_daily_df = filter_target_date_window(lamp_daily_df, start_date=target_date, end_date=target_date)
        day_hrrr_daily_df = filter_target_date_window(hrrr_daily_df, start_date=target_date, end_date=target_date)
        output_df = build_training_features_overnight(
            labels_df=day_labels_df,
            label_history_df=day_label_history_df,
            obs_df=day_obs_df,
            nbm_daily_df=day_nbm_daily_df,
            lamp_daily_df=day_lamp_daily_df,
            hrrr_daily_df=day_hrrr_daily_df,
            cutoff_local_time=args.cutoff_local_time,
            station_id=args.station_id,
        )
        if output_df.empty and not args.allow_empty:
            raise ValueError(
                f"training_features_overnight build produced zero rows for target_date_local={target_token}; "
                "rerun with --allow-empty only for schema/bootstrap workflows"
            )
        output_path, manifest_path = output_paths_for_date(args.output_dir, target_token)
        manifest = build_output_manifest_for_date(
            target_date_local=target_token,
            output_path=output_path,
            row_count=len(output_df),
            station_id=args.station_id,
            cutoff_local_time=args.cutoff_local_time,
            output_df=output_df,
        )
        _write_atomic_parquet(output_df, output_path)
        _write_atomic_json(manifest_path, manifest)
        if not validate_date_output(
            output_path=output_path,
            manifest_path=manifest_path,
            target_date_local=target_token,
            allow_empty=args.allow_empty,
        ):
            raise ValueError(f"daily merged output validation failed for target_date_local={target_token}")
        written.extend([output_path, manifest_path])

    if args.manifest_output_path is not None:
        summary_payload = {
            "status": "ok",
            "start_local_date": start_date.isoformat(),
            "end_local_date": end_date.isoformat(),
            "target_date_count": len(iter_local_dates(start_date, end_date)),
            "written_paths": [str(path) for path in written],
            "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_atomic_json(args.manifest_output_path, summary_payload)
        print(args.manifest_output_path)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
