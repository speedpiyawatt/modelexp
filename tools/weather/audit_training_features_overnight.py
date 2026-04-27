#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.training_features_overnight_contract import ALLOWED_PREFIXES, REGISTRY_ROWS, registry_by_name, registry_columns


DEFAULT_INPUT_PATH = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("tools/weather/data/runtime/training/audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the frozen overnight training table against the checked-in contract registry.")
    parser.add_argument("--input-path", type=pathlib.Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def numeric_variance(series: pd.Series) -> float | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 2:
        return None
    return float(numeric.var())


def build_audit(df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    registry = registry_by_name()
    required_core_columns = [row["column_name"] for row in REGISTRY_ROWS if row["freeze_level"] == "core_required"]
    missing_core_columns = [column for column in required_core_columns if column not in df.columns]
    missing_registry_columns = [column for column in registry_columns() if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in registry]
    prefix_violations = [
        column
        for column in df.columns
        if column not in {"target_date_local", "station_id", "selection_cutoff_local"}
        and not column.startswith(ALLOWED_PREFIXES)
    ]
    duplicate_key_count = int(df.duplicated(subset=["target_date_local", "station_id"]).sum()) if {"target_date_local", "station_id"} <= set(df.columns) else len(df)

    metrics_rows: list[dict[str, object]] = []
    for row in REGISTRY_ROWS:
        column = str(row["column_name"])
        if column not in df.columns:
            metrics_rows.append(
                {
                    "column_name": column,
                    "freeze_level": row["freeze_level"],
                    "present": False,
                    "null_rate": None,
                    "availability_rate": None,
                    "variance": None,
                }
            )
            continue
        series = df[column]
        null_rate = float(series.isna().mean()) if len(series) else 0.0
        metrics_rows.append(
            {
                "column_name": column,
                "freeze_level": row["freeze_level"],
                "present": True,
                "null_rate": null_rate,
                "availability_rate": float(1.0 - null_rate),
                "variance": numeric_variance(series) if str(row["dtype"]).startswith(("float", "int")) else None,
            }
        )

    summary = {
        "row_count": int(len(df)),
        "registry_column_count": len(registry_columns()),
        "table_column_count": int(len(df.columns)),
        "missing_core_columns": missing_core_columns,
        "missing_registry_columns": missing_registry_columns,
        "extra_columns": extra_columns,
        "prefix_violations": prefix_violations,
        "duplicate_key_count": duplicate_key_count,
        "checks": {
            "non_empty_table": len(df) > 0,
            "core_columns_present": not missing_core_columns,
            "registry_match": not missing_registry_columns and not extra_columns,
            "prefix_policy_ok": not prefix_violations,
            "unique_target_date_station_key": duplicate_key_count == 0,
        },
    }
    return summary, pd.DataFrame.from_records(metrics_rows)


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    summary, metrics_df = build_audit(df)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "training_features_overnight.audit.json"
    metrics_path = args.output_dir / "training_features_overnight.column_metrics.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    metrics_df.to_csv(metrics_path, index=False)
    print(summary_path)
    print(metrics_path)
    return 0 if all(summary["checks"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
