from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.build_training_features_overnight import build_wu_cutoff_features

from .contracts import DEFAULT_CUTOFF_LOCAL_TIME, DEFAULT_STATION_ID, audit_training_features, cutoff_timestamp, kelvin_to_fahrenheit


DEFAULT_LABELS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/wunderground/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/wunderground/wu_obs_intraday.parquet")
DEFAULT_NBM_ROOT = pathlib.Path("data/runtime/backfill_overnight/nbm_overnight")
DEFAULT_LAMP_ROOT = pathlib.Path("experiments/no_hrrr_model/data/runtime/lamp_overnight")
DEFAULT_OUTPUT_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet")
DEFAULT_MANIFEST_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-HRRR overnight KLGA training features.")
    parser.add_argument("--labels-path", type=pathlib.Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--nbm-root", type=pathlib.Path, default=DEFAULT_NBM_ROOT)
    parser.add_argument("--lamp-root", type=pathlib.Path, default=DEFAULT_LAMP_ROOT)
    parser.add_argument("--output-path", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-output-path", type=pathlib.Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME)
    parser.add_argument("--start-local-date")
    parser.add_argument("--end-local-date")
    parser.add_argument("--allow-missing-lamp", action="store_true", help="Permit successful output when every row lacks LAMP.")
    return parser.parse_args()


def read_parquet(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def read_named_parquets(root: pathlib.Path, filename: str) -> pd.DataFrame:
    if root.is_file():
        return read_parquet(root)
    if not root.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path).dropna(axis=1, how="all") for path in sorted(root.rglob(filename))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_date_window(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if df.empty or "target_date_local" not in df.columns:
        return df.copy()
    out = df.copy()
    out["target_date_local"] = out["target_date_local"].astype(str)
    if start_date is not None:
        out = out.loc[out["target_date_local"] >= start_date]
    if end_date is not None:
        out = out.loc[out["target_date_local"] <= end_date]
    return out.reset_index(drop=True)


def transform_nbm_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id", "meta_nbm_available"])
    duplicate_count = int(df.duplicated(subset=["target_date_local", "station_id"]).sum())
    if duplicate_count:
        raise ValueError(f"NBM input has duplicate target_date_local/station_id rows: {duplicate_count}")
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
    keep = ["target_date_local", "station_id", "meta_nbm_available", *rename_map.values()]
    keep.extend(column for column in out.columns if column.startswith("nbm_"))
    return out.loc[:, [column for column in keep if column in out.columns]].drop_duplicates(subset=["target_date_local", "station_id"])


def transform_lamp_daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target_date_local", "station_id", "meta_lamp_available"])
    duplicate_count = int(df.duplicated(subset=["target_date_local", "station_id"]).sum())
    if duplicate_count:
        raise ValueError(f"LAMP input has duplicate target_date_local/station_id rows: {duplicate_count}")
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
    passthrough = {"target_date_local", "station_id", "selection_cutoff_local", *rename_map.values(), "meta_lamp_available"}
    out = df.rename(columns=rename_map).copy()
    out["meta_lamp_available"] = True
    feature_rename = {column: f"lamp_{column}" for column in out.columns if column not in passthrough}
    out = out.rename(columns=feature_rename)
    keep = ["target_date_local", "station_id", "meta_lamp_available", *rename_map.values()]
    keep.extend(column for column in out.columns if column.startswith("lamp_"))
    return out.loc[:, [column for column in keep if column in out.columns]].drop_duplicates(subset=["target_date_local", "station_id"])


def build_training_features(
    *,
    labels_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    nbm_df: pd.DataFrame,
    lamp_df: pd.DataFrame,
    station_id: str = DEFAULT_STATION_ID,
    cutoff_local_time: str = DEFAULT_CUTOFF_LOCAL_TIME,
    label_history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = labels_df.copy()
    if base.empty:
        return pd.DataFrame()
    duplicate_label_count = int(base.duplicated(subset=["target_date_local", "station_id"]).sum())
    if duplicate_label_count:
        raise ValueError(f"labels input has duplicate target_date_local/station_id rows: {duplicate_label_count}")
    base = base.loc[base["station_id"].astype(str) == str(station_id)].copy()
    base["target_date_local"] = base["target_date_local"].astype(str)
    base["station_id"] = base["station_id"].astype(str)
    base = base.rename(columns={"label_final_tmax_f": "final_tmax_f", "label_final_tmin_f": "final_tmin_f"})
    base["selection_cutoff_local"] = base["target_date_local"].map(lambda value: cutoff_timestamp(value, cutoff_local_time).isoformat())

    if not obs_df.empty and "station_id" in obs_df.columns:
        obs_df = obs_df.loc[obs_df["station_id"].astype(str) == str(station_id)].copy()

    wu_features = build_wu_cutoff_features(
        labels_df.loc[labels_df["station_id"].astype(str) == str(station_id)].copy(),
        obs_df,
        cutoff_local_time=cutoff_local_time,
        label_history_df=label_history_df if label_history_df is not None else labels_df,
    )
    merged = base.merge(wu_features, on=["target_date_local", "station_id"], how="left")
    merged = merged.merge(transform_nbm_daily(nbm_df), on=["target_date_local", "station_id"], how="left")
    merged = merged.merge(transform_lamp_daily(lamp_df), on=["target_date_local", "station_id"], how="left")

    for column in ("meta_wu_obs_available", "meta_nbm_available", "meta_lamp_available", "meta_nbm_coverage_complete", "meta_lamp_coverage_complete", "meta_lamp_revision_available", "meta_lamp_missing_optional_any"):
        if column in merged.columns:
            merged[column] = merged[column].astype("boolean").fillna(False)

    nbm_tmax_k = merged["nbm_temp_2m_day_max_k"] if "nbm_temp_2m_day_max_k" in merged.columns else pd.Series([pd.NA] * len(merged), index=merged.index)
    lamp_tmax_f = merged["lamp_day_tmp_max_f_forecast"] if "lamp_day_tmp_max_f_forecast" in merged.columns else pd.Series([pd.NA] * len(merged), index=merged.index)
    merged["nbm_tmax_open_f"] = pd.to_numeric(nbm_tmax_k, errors="coerce").map(kelvin_to_fahrenheit)
    merged["lamp_tmax_open_f"] = pd.to_numeric(lamp_tmax_f, errors="coerce")
    merged["anchor_tmax_f"] = 0.5 * merged["nbm_tmax_open_f"] + 0.5 * merged["lamp_tmax_open_f"]
    merged["nbm_minus_lamp_tmax_f"] = merged["nbm_tmax_open_f"] - merged["lamp_tmax_open_f"]
    merged["target_residual_f"] = pd.to_numeric(merged["final_tmax_f"], errors="coerce") - merged["anchor_tmax_f"]
    merged["model_training_eligible"] = (
        merged["meta_nbm_available"].astype("boolean").fillna(False)
        & merged["meta_lamp_available"].astype("boolean").fillna(False)
        & pd.to_numeric(merged["nbm_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(merged["lamp_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(merged["final_tmax_f"], errors="coerce").notna()
    )

    forbidden = [column for column in merged.columns if column.startswith(("hrrr_", "meta_hrrr_"))]
    if forbidden:
        merged = merged.drop(columns=forbidden)
    return merged.drop_duplicates(subset=["target_date_local", "station_id"]).reset_index(drop=True)


def write_manifest(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    labels_df = filter_date_window(read_parquet(args.labels_path), args.start_local_date, args.end_local_date)
    obs_df = read_parquet(args.obs_path)
    nbm_df = filter_date_window(read_named_parquets(args.nbm_root, "nbm.overnight.parquet"), args.start_local_date, args.end_local_date)
    lamp_df = filter_date_window(read_named_parquets(args.lamp_root, "lamp.overnight.parquet"), args.start_local_date, args.end_local_date)
    output_df = build_training_features(
        labels_df=labels_df,
        obs_df=obs_df,
        nbm_df=nbm_df,
        lamp_df=lamp_df,
        station_id=args.station_id,
        cutoff_local_time=args.cutoff_local_time,
    )
    audit = audit_training_features(output_df, cutoff_local_time=args.cutoff_local_time)
    fatal_errors = list(audit.errors)
    if args.allow_missing_lamp:
        print("--allow-missing-lamp is deprecated; missing-source rows are retained with model_training_eligible=False.")
    if fatal_errors:
        raise ValueError("; ".join(fatal_errors))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_parquet(args.output_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "row_count": int(len(output_df)),
        "column_count": int(len(output_df.columns)),
        "station_id": args.station_id,
        "cutoff_local_time": args.cutoff_local_time,
        "labels_path": str(args.labels_path),
        "obs_path": str(args.obs_path),
        "nbm_root": str(args.nbm_root),
        "lamp_root": str(args.lamp_root),
        "output_path": str(args.output_path),
        "audit_warnings": list(audit.warnings),
    }
    write_manifest(args.manifest_output_path, manifest)
    print(args.output_path)
    print(args.manifest_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
