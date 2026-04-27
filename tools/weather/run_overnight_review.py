#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import random
import sys
from typing import Any

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.audit_training_features_overnight import build_audit as build_merged_audit
from tools.weather.audit_training_features_overnight_normalized import build_audit as build_normalized_audit
from tools.weather.build_training_features_overnight import build_training_features_overnight
from tools.weather.build_training_features_overnight_normalized import load_vocabularies, normalize_training_features_overnight
from tools.weather.training_features_overnight_contract import registry_columns as merged_registry_columns
from wunderground.build_training_tables import build_training_tables


DEFAULT_HISTORY_DIR = REPO_ROOT / "wunderground" / "output" / "history"
DEFAULT_HRRR_SUMMARY_PATH = REPO_ROOT / "data" / "runtime" / "features" / "overnight_summary_klga" / "2025-04.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tools" / "weather" / "data" / "runtime" / "review"
DEFAULT_VOCAB_PATH = REPO_ROOT / "tools" / "weather" / "training_feature_vocabularies.json"
DEFAULT_RANDOM_SEED = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local overnight-stack review across WU history, merged tables, and normalization.")
    parser.add_argument("--history-dir", type=pathlib.Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--hrrr-summary-path", type=pathlib.Path, default=DEFAULT_HRRR_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--vocab-path", type=pathlib.Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    return parser.parse_args()


def _iso_dates(values: pd.Series) -> list[str]:
    return sorted({str(value) for value in values.dropna().astype(str)})


def select_review_dates(labels_df: pd.DataFrame, *, random_seed: int) -> dict[str, list[str]]:
    rng = random.Random(random_seed)
    dates = pd.to_datetime(labels_df["target_date_local"])
    warm = _iso_dates(labels_df.loc[dates.dt.month.isin((6, 7, 8)), "target_date_local"])
    cold = _iso_dates(labels_df.loc[dates.dt.month.isin((12, 1, 2)), "target_date_local"])
    obs_counts = pd.to_numeric(labels_df["label_obs_count"], errors="coerce")
    anomalies = labels_df.assign(label_obs_count_numeric=obs_counts).sort_values("label_obs_count_numeric")
    low = anomalies.head(1)["target_date_local"].astype(str).tolist()
    high = anomalies.tail(1)["target_date_local"].astype(str).tolist()
    return {
        "fixed": ["2025-04-11", "2025-03-09", "2025-11-02"],
        "warm_random": rng.sample(warm, k=min(3, len(warm))),
        "cold_random": rng.sample(cold, k=min(3, len(cold))),
        "anomalous_obs_count": [*low, *high],
    }


def inspect_dates(merged_df: pd.DataFrame, obs_df: pd.DataFrame, labels_df: pd.DataFrame, review_dates: list[str]) -> list[dict[str, Any]]:
    obs = obs_df.copy()
    obs["valid_time_local"] = pd.to_datetime(obs["valid_time_local"], utc=True).dt.tz_convert("America/New_York")
    labels = labels_df.set_index(["target_date_local", "station_id"])
    inspections: list[dict[str, Any]] = []

    for target_date_local in review_dates:
        row_df = merged_df.loc[merged_df["target_date_local"] == target_date_local]
        if row_df.empty:
            inspections.append({"target_date_local": target_date_local, "status": "missing_row"})
            continue
        row = row_df.iloc[0]
        cutoff = pd.Timestamp(str(row["selection_cutoff_local"]))
        prior_obs = obs.loc[(obs["station_id"] == str(row["station_id"])) & (obs["valid_time_local"] <= cutoff)].sort_values("valid_time_local")
        last_obs_time = prior_obs["valid_time_local"].iloc[-1].isoformat() if not prior_obs.empty else None
        previous_date_local = (dt.date.fromisoformat(target_date_local) - dt.timedelta(days=1)).isoformat()
        previous_label = labels.loc[(previous_date_local, str(row["station_id"]))] if (previous_date_local, str(row["station_id"])) in labels.index else None
        if isinstance(previous_label, pd.DataFrame):
            previous_label = previous_label.iloc[0]
        inspections.append(
            {
                "target_date_local": target_date_local,
                "selection_cutoff_local": str(row["selection_cutoff_local"]),
                "meta_wu_obs_available": bool(row["meta_wu_obs_available"]),
                "meta_nbm_available": bool(row["meta_nbm_available"]),
                "meta_lamp_available": bool(row["meta_lamp_available"]),
                "meta_hrrr_available": bool(row["meta_hrrr_available"]),
                "wu_last_obs_time_local": row["meta_wu_last_obs_time_local"],
                "wu_last_obs_matches_latest_pre_cutoff": row["meta_wu_last_obs_time_local"] == last_obs_time,
                "label_final_tmax_f": row["label_final_tmax_f"],
                "wu_prev_day_final_tmax_f": row["wu_prev_day_final_tmax_f"],
                "expected_prev_day_final_tmax_f": None if previous_label is None else float(previous_label["label_final_tmax_f"]),
                "prev_day_match": previous_label is None or row["wu_prev_day_final_tmax_f"] == float(previous_label["label_final_tmax_f"]),
            }
        )
    return inspections


def build_fuzz_source_rows(*, random_seed: int, row_count: int = 40) -> pd.DataFrame:
    rng = random.Random(random_seed)
    rows: list[dict[str, Any]] = []
    for index in range(row_count):
        row = {column: None for column in merged_registry_columns()}
        row["target_date_local"] = f"2026-02-{(index % 20) + 1:02d}"
        row["station_id"] = "KLGA"
        row["selection_cutoff_local"] = f"{row['target_date_local']}T00:05:00-05:00"
        row["label_final_tmax_f"] = rng.choice([28.0, 44.0, 61.0, 79.0])
        row["label_final_tmin_f"] = rng.choice([15.0, 29.0, 41.0])
        row["label_market_bin"] = f"{int(row['label_final_tmax_f'])}F"
        row["label_obs_count"] = rng.choice([2, 3, 23, 24, 25, 31])
        row["label_first_obs_time_local"] = f"{row['target_date_local']}T00:00:00-05:00"
        row["label_last_obs_time_local"] = f"{row['target_date_local']}T23:00:00-05:00"
        row["label_total_precip_in"] = rng.choice([0.0, 0.1, 0.75])
        for meta_name in (
            "meta_wu_obs_available",
            "meta_nbm_available",
            "meta_lamp_available",
            "meta_hrrr_available",
            "meta_nbm_coverage_complete",
            "meta_lamp_coverage_complete",
            "meta_hrrr_has_full_day_21_local_coverage",
        ):
            row[meta_name] = rng.choice([True, False])
        row["wu_last_cloud_cover_code"] = rng.choice(["CLR", "BKN", "OV", "", None, "ZZ"])
        row["wu_last_wx_phrase"] = rng.choice(["Haze", "Cloudy", "Volcanic dust", "Thunderstorm", "", None])
        row["lamp_cld_code_at_09"] = rng.choice(["BKN", None, "UNK"])
        row["lamp_obv_code_at_09"] = rng.choice(["HZ", "FG", None, "ZZ"])
        row["lamp_typ_code_at_09"] = rng.choice(["R", "S", None, "QQ"])
        row["lamp_morning_cld_mode"] = rng.choice(["SCT", None, "BAD"])
        row["lamp_day_precip_type_mode"] = rng.choice(["R", None, "BAD"])
        row["nbm_temp_2m_day_max_k"] = rng.choice([289.0, 294.0, None])
        row["hrrr_temp_2m_day_max_k"] = rng.choice([288.0, 293.0, None])
        row["nbm_minus_hrrr_day_max_k"] = None if row["nbm_temp_2m_day_max_k"] is None or row["hrrr_temp_2m_day_max_k"] is None else row["nbm_temp_2m_day_max_k"] - row["hrrr_temp_2m_day_max_k"]
        row["nbm_gust_10m_day_max_ms"] = rng.choice([0.0, 4.5, 12.0, 22.0])
        row["hrrr_wind_10m_day_max_ms"] = rng.choice([0.0, 3.0, 10.0, 18.0])
        row["nbm_apcp_day_total_kg_m2"] = rng.choice([0.0, 12.7, 25.4, 50.8])
        row["hrrr_apcp_day_total_kg_m2"] = rng.choice([0.0, 12.7, 25.4, 50.8])
        row["nbm_visibility_day_min_m"] = rng.choice([400.0, 1609.344, 8046.72])
        row["nbm_ceiling_morning_min_m"] = rng.choice([60.96, 304.8, 914.4])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def summarize_fuzz(normalized_df: pd.DataFrame, audit_summary: dict[str, Any]) -> dict[str, Any]:
    sample_columns = ["wu_last_cloud_cover_id", "wu_last_weather_family_id", "lamp_cloud_cover_09_local_id", "lamp_weather_09_local_id", "lamp_precip_type_09_local_id"]
    available_columns = [column for column in sample_columns if column in normalized_df.columns]
    return {
        "row_count": int(len(normalized_df)),
        "checks": audit_summary["checks"],
        "category_sample": normalized_df.loc[:, available_columns].head(10).to_dict(orient="records"),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    labels_df, obs_df = build_training_tables(args.history_dir)
    hrrr_df = pd.read_parquet(args.hrrr_summary_path) if args.hrrr_summary_path.exists() else pd.DataFrame()
    merged_df = build_training_features_overnight(
        labels_df=labels_df,
        obs_df=obs_df,
        nbm_daily_df=pd.DataFrame(),
        lamp_daily_df=pd.DataFrame(),
        hrrr_daily_df=hrrr_df,
        cutoff_local_time="00:05",
        station_id="KLGA",
    )
    merged_summary, merged_metrics = build_merged_audit(merged_df)

    review_date_groups = select_review_dates(labels_df, random_seed=args.random_seed)
    review_dates = [date for group in review_date_groups.values() for date in group]
    inspections = inspect_dates(merged_df, obs_df, labels_df, review_dates)

    vocabularies = load_vocabularies(args.vocab_path)
    fuzz_source_df = build_fuzz_source_rows(random_seed=args.random_seed)
    fuzz_normalized_df = normalize_training_features_overnight(fuzz_source_df, vocabularies)
    fuzz_summary, fuzz_metrics = build_normalized_audit(fuzz_normalized_df)

    duplicate_fuzz_df = fuzz_normalized_df.copy()
    if not duplicate_fuzz_df.empty:
        duplicate_fuzz_df.loc[:, "target_date_local"] = duplicate_fuzz_df.iloc[0]["target_date_local"]
        duplicate_fuzz_df.loc[:, "station_id"] = duplicate_fuzz_df.iloc[0]["station_id"]
    duplicate_summary, _ = build_normalized_audit(duplicate_fuzz_df)
    empty_summary, _ = build_normalized_audit(pd.DataFrame(columns=fuzz_normalized_df.columns))

    report = {
        "wu_labels_row_count": int(len(labels_df)),
        "wu_obs_row_count": int(len(obs_df)),
        "merged_row_count": int(len(merged_df)),
        "merged_audit_checks": merged_summary["checks"],
        "review_dates": review_date_groups,
        "date_inspections": inspections,
        "fuzz_summary": summarize_fuzz(fuzz_normalized_df, fuzz_summary),
        "duplicate_fuzz_checks": duplicate_summary["checks"],
        "empty_fuzz_checks": empty_summary["checks"],
    }

    labels_df.to_parquet(args.output_dir / "wu_labels_daily.parquet", index=False)
    obs_df.to_parquet(args.output_dir / "wu_obs_intraday.parquet", index=False)
    merged_df.to_parquet(args.output_dir / "training_features_overnight.parquet", index=False)
    fuzz_normalized_df.to_parquet(args.output_dir / "training_features_overnight_normalized_fuzz.parquet", index=False)
    merged_metrics.to_csv(args.output_dir / "training_features_overnight.column_metrics.csv", index=False)
    fuzz_metrics.to_csv(args.output_dir / "training_features_overnight_normalized_fuzz.column_metrics.csv", index=False)
    (args.output_dir / "overnight_review_report.json").write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    print(args.output_dir / "overnight_review_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
