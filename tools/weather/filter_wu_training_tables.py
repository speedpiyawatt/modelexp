#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import pandas as pd


DEFAULT_LABELS_PATH = pathlib.Path("wunderground/output/tables/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("wunderground/output/tables/wu_obs_intraday.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("tools/weather/data/runtime/contract_test/wu_tables")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter Wunderground training tables into a target-date window while preserving support history for overnight contract tests.")
    parser.add_argument("--labels-path", type=pathlib.Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def filter_target_labels(labels_df: pd.DataFrame, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if labels_df.empty:
        return labels_df.copy()
    mask = (labels_df["target_date_local"] >= start_date.isoformat()) & (labels_df["target_date_local"] <= end_date.isoformat())
    return labels_df.loc[mask].sort_values(["target_date_local", "station_id"]).reset_index(drop=True)


def filter_label_history(labels_df: pd.DataFrame, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if labels_df.empty:
        return labels_df.copy()
    history_start = (start_date - dt.timedelta(days=1)).isoformat()
    history_end = end_date.isoformat()
    mask = (labels_df["target_date_local"] >= history_start) & (labels_df["target_date_local"] <= history_end)
    return labels_df.loc[mask].sort_values(["target_date_local", "station_id"]).reset_index(drop=True)


def filter_obs_support(obs_df: pd.DataFrame, *, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    if obs_df.empty:
        return obs_df.copy()
    support_start = (start_date - dt.timedelta(days=1)).isoformat()
    support_end = end_date.isoformat()
    mask = (obs_df["date_local"] >= support_start) & (obs_df["date_local"] <= support_end)
    return obs_df.loc[mask].sort_values(["date_local", "station_id", "valid_time_local"]).reset_index(drop=True)


def build_manifest(
    *,
    start_date: dt.date,
    end_date: dt.date,
    labels_df: pd.DataFrame,
    label_history_df: pd.DataFrame,
    obs_df: pd.DataFrame,
) -> dict[str, object]:
    return {
        "start_local_date": start_date.isoformat(),
        "end_local_date": end_date.isoformat(),
        "label_row_count": int(len(labels_df)),
        "label_history_row_count": int(len(label_history_df)),
        "obs_row_count": int(len(obs_df)),
        "label_history_start_local_date": (start_date - dt.timedelta(days=1)).isoformat(),
    }


def main() -> int:
    args = parse_args()
    start_date = parse_local_date(args.start_local_date)
    end_date = parse_local_date(args.end_local_date)
    labels_df = pd.read_parquet(args.labels_path)
    obs_df = pd.read_parquet(args.obs_path)

    filtered_labels_df = filter_target_labels(labels_df, start_date=start_date, end_date=end_date)
    filtered_label_history_df = filter_label_history(labels_df, start_date=start_date, end_date=end_date)
    filtered_obs_df = filter_obs_support(obs_df, start_date=start_date, end_date=end_date)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_output_path = args.output_dir / "labels_daily.parquet"
    label_history_output_path = args.output_dir / "labels_history.parquet"
    obs_output_path = args.output_dir / "wu_obs_intraday.parquet"
    manifest_output_path = args.output_dir / "manifest.json"

    filtered_labels_df.to_parquet(labels_output_path, index=False)
    filtered_label_history_df.to_parquet(label_history_output_path, index=False)
    filtered_obs_df.to_parquet(obs_output_path, index=False)
    manifest_output_path.write_text(
        json.dumps(
            build_manifest(
                start_date=start_date,
                end_date=end_date,
                labels_df=filtered_labels_df,
                label_history_df=filtered_label_history_df,
                obs_df=filtered_obs_df,
            ),
            indent=2,
            sort_keys=True,
        )
    )

    print(labels_output_path)
    print(label_history_output_path)
    print(obs_output_path)
    print(manifest_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
