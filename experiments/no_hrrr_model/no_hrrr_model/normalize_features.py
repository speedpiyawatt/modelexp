from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import pathlib

import pandas as pd

from .contracts import audit_training_features, knots_to_mph, kelvin_to_fahrenheit, mps_to_mph


DEFAULT_INPUT_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet")
DEFAULT_OUTPUT_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_MANIFEST_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.manifest.json")
NON_FEATURE_CATEGORICAL_PREFIXES = ("label_",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize no-HRRR overnight training features for modeling.")
    parser.add_argument("--input-path", type=pathlib.Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-output-path", type=pathlib.Path, default=DEFAULT_MANIFEST_PATH)
    return parser.parse_args()


def normalize_features(df: pd.DataFrame, vocabularies: dict[str, list[str]] | None = None) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    out = df.copy()
    for column in list(out.columns):
        if column.startswith("nbm_") and column.endswith("_k"):
            out[f"{column[:-2]}_f"] = pd.to_numeric(out[column], errors="coerce").map(kelvin_to_fahrenheit)
        if column.startswith("nbm_") and column.endswith("_ms"):
            out[f"{column[:-3]}_mph"] = pd.to_numeric(out[column], errors="coerce").map(mps_to_mph)
        if column.startswith("lamp_") and "_kt_" in column:
            out[column.replace("_kt_", "_mph_")] = pd.to_numeric(out[column], errors="coerce").map(knots_to_mph)

    if "target_date_local" in out.columns:
        dates = pd.to_datetime(out["target_date_local"], errors="coerce")
        out["season_month"] = dates.dt.month
        out["day_of_year"] = dates.dt.dayofyear
        radians = 2.0 * 3.141592653589793 * out["day_of_year"] / 366.0
        out["day_of_year_sin"] = radians.map(lambda value: pd.NA if pd.isna(value) else math.sin(value))
        out["day_of_year_cos"] = radians.map(lambda value: pd.NA if pd.isna(value) else math.cos(value))

    categorical_columns = [
        column
        for column in out.columns
        if out[column].dtype == "object"
        and column not in {"target_date_local", "station_id", "selection_cutoff_local"}
        and not column.startswith(NON_FEATURE_CATEGORICAL_PREFIXES)
    ]
    learned_vocabularies: dict[str, list[str]] = {}
    for column in categorical_columns:
        string_values = out[column].astype("string")
        if vocabularies is not None and column in vocabularies:
            mapping = {value: index for index, value in enumerate(vocabularies[column])}
            out[f"{column}_code"] = string_values.map(mapping).fillna(-1).astype("int64")
            continue
        codes, uniques = pd.factorize(string_values, sort=True)
        learned_vocabularies[column] = [str(value) for value in uniques.tolist()]
        out[f"{column}_code"] = codes
        out.loc[out[column].isna(), f"{column}_code"] = -1
    if vocabularies is not None:
        learned_vocabularies.update(vocabularies)
    return out, learned_vocabularies


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    audit_training_features(df).raise_for_errors()
    out, vocabularies = normalize_features(df)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "categorical_vocabularies": vocabularies,
    }
    args.manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_path)
    print(args.manifest_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
