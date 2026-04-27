#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import sys


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from tools.weather.filter_wu_training_tables import (
    build_manifest as build_wu_window_manifest,
    filter_label_history,
    filter_obs_support,
    filter_target_labels,
    parse_local_date,
)
from tools.weather.run_server_overnight_stage import (
    DEFAULT_VOCAB_PATH,
    build_short_window_review,
    filter_window,
    load_named_parquets,
    source_review_ok,
    write_json,
)
from tools.weather.build_training_features_overnight_normalized import load_vocabularies
from wunderground.build_training_tables import build_training_tables, write_outputs


DEFAULT_HISTORY_DIR = pathlib.Path("wunderground/output/history")
DEFAULT_OUTPUT_ROOT = pathlib.Path("tools/weather/data/runtime/contract_test_2025-04")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a month-scoped overnight contract-test review from Wunderground labels plus source-aware overnight feature tables.")
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    parser.add_argument("--history-dir", type=pathlib.Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-root", type=pathlib.Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vocab-path", type=pathlib.Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--skip-wu-build", action="store_true", help="Reuse an existing <output-root>/wu_tables_full instead of rebuilding full Wunderground tables.")
    return parser.parse_args()


def build_or_load_wu_full_tables(*, history_dir: pathlib.Path, output_root: pathlib.Path, skip_build: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    wu_full_dir = output_root / "wu_tables_full"
    labels_path = wu_full_dir / "labels_daily.parquet"
    obs_path = wu_full_dir / "wu_obs_intraday.parquet"
    if skip_build:
        return pd.read_parquet(labels_path), pd.read_parquet(obs_path)
    labels_df, obs_df = build_training_tables(history_dir)
    write_outputs(output_dir=wu_full_dir, labels_df=labels_df, obs_df=obs_df)
    return labels_df, obs_df


def write_filtered_wu_tables(
    *,
    output_root: pathlib.Path,
    start_local_date: str,
    end_local_date: str,
    labels_df: pd.DataFrame,
    obs_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_date = parse_local_date(start_local_date)
    end_date = parse_local_date(end_local_date)
    filtered_labels_df = filter_target_labels(labels_df, start_date=start_date, end_date=end_date)
    filtered_label_history_df = filter_label_history(labels_df, start_date=start_date, end_date=end_date)
    filtered_obs_df = filter_obs_support(obs_df, start_date=start_date, end_date=end_date)

    wu_dir = output_root / "wu_tables"
    wu_dir.mkdir(parents=True, exist_ok=True)
    filtered_labels_df.to_parquet(wu_dir / "labels_daily.parquet", index=False)
    filtered_label_history_df.to_parquet(wu_dir / "labels_history.parquet", index=False)
    filtered_obs_df.to_parquet(wu_dir / "wu_obs_intraday.parquet", index=False)
    write_json(
        wu_dir / "manifest.json",
        build_wu_window_manifest(
            start_date=start_date,
            end_date=end_date,
            labels_df=filtered_labels_df,
            label_history_df=filtered_label_history_df,
            obs_df=filtered_obs_df,
        ),
    )
    return filtered_labels_df, filtered_label_history_df, filtered_obs_df


def main() -> int:
    args = parse_args()
    full_labels_df, full_obs_df = build_or_load_wu_full_tables(
        history_dir=args.history_dir,
        output_root=args.output_root,
        skip_build=args.skip_wu_build,
    )
    labels_df, label_history_df, obs_df = write_filtered_wu_tables(
        output_root=args.output_root,
        start_local_date=args.start_local_date,
        end_local_date=args.end_local_date,
        labels_df=full_labels_df,
        obs_df=full_obs_df,
    )

    nbm_df = load_named_parquets(args.output_root / "nbm" / "overnight", "nbm.overnight.parquet")
    lamp_df = load_named_parquets(args.output_root / "lamp" / "overnight", "lamp.overnight.parquet")
    hrrr_df = filter_window(
        load_named_parquets(args.output_root / "hrrr" / "summary", "*.parquet"),
        start_date=args.start_local_date,
        end_date=args.end_local_date,
    )
    vocabularies = load_vocabularies(args.vocab_path)

    review = build_short_window_review(
        labels_df=labels_df,
        label_history_df=label_history_df,
        obs_df=obs_df,
        nbm_df=nbm_df,
        lamp_df=lamp_df,
        hrrr_df=hrrr_df,
        vocabularies=vocabularies,
        output_dir=args.output_root / "training",
    )
    write_json(args.output_root / "training" / "review.json", review)
    print(args.output_root / "training" / "review.json")
    return 0 if source_review_ok(review) else 1


if __name__ == "__main__":
    raise SystemExit(main())
