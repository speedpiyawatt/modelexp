from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .distribution import expected_temperature, quantiles_to_degree_ladder
from .event_bins import load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_MODELS_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/models")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/predictions")
DEFAULT_CALIBRATION_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/rolling_origin_calibration_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate no-HRRR final-Tmax probabilities for a normalized feature row.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-date-local", required=True)
    parser.add_argument("--station-id", default="KLGA")
    parser.add_argument("--event-bin", action="append", default=[], help="Optional event bin label. May be repeated.")
    parser.add_argument("--event-bins-path", type=pathlib.Path, default=None, help="Optional JSON or text file containing event-bin labels.")
    parser.add_argument("--max-so-far-f", type=float, default=None)
    parser.add_argument("--calibration-path", type=pathlib.Path, default=None, help="Optional calibration manifest with final-quantile offsets.")
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())


def prepare_features(row: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = row.loc[:, feature_columns].copy()
    for column in x.columns:
        if pd.api.types.is_bool_dtype(x[column].dtype):
            x[column] = x[column].astype("int8")
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x


def select_row(df: pd.DataFrame, *, target_date_local: str, station_id: str) -> pd.DataFrame:
    row = df.loc[(df["target_date_local"].astype(str) == target_date_local) & (df["station_id"].astype(str) == station_id)].copy()
    if row.empty:
        raise ValueError(f"no feature row found for target_date_local={target_date_local} station_id={station_id}")
    if len(row) > 1:
        raise ValueError(f"multiple feature rows found for target_date_local={target_date_local} station_id={station_id}")
    if "model_prediction_available" in row.columns:
        available = bool(row["model_prediction_available"].astype("boolean").fillna(False).iloc[0])
    elif "model_training_eligible" in row.columns:
        available = bool(row["model_training_eligible"].astype("boolean").fillna(False).iloc[0])
    else:
        available = True
    if not available:
        raise ValueError(f"feature row is not model eligible for target_date_local={target_date_local} station_id={station_id}")
    return row


def print_prediction_summary(payload: dict[str, object], output_path: pathlib.Path) -> None:
    print(f"prediction_path={output_path}")
    print(f"status={payload['status']} target_date_local={payload['target_date_local']} station_id={payload['station_id']}")
    print(f"expected_final_tmax_f={float(payload['expected_final_tmax_f']):.2f}")
    print(f"anchor_tmax_f={float(payload['anchor_tmax_f']):.2f}")
    final_quantiles = payload.get("final_tmax_quantiles_f", {})
    if isinstance(final_quantiles, dict):
        labels = (("0.05", "q05"), ("0.1", "q10"), ("0.25", "q25"), ("0.5", "q50"), ("0.75", "q75"), ("0.9", "q90"), ("0.95", "q95"))
        values = " ".join(f"{label}={float(final_quantiles[key]):.2f}" for key, label in labels if key in final_quantiles)
        if values:
            print(f"final_tmax_quantiles_f {values}")
    event_bins = payload.get("event_bins", [])
    if isinstance(event_bins, list) and event_bins:
        print("event_bins")
        for row in event_bins:
            if isinstance(row, dict):
                print(f"  {row.get('bin')}: {float(row.get('probability', 0.0)):.4f}")


def main() -> int:
    args = parse_args()
    feature_manifest = load_json(args.models_dir / "feature_manifest.json")
    feature_columns = list(feature_manifest["feature_columns"])
    df = pd.read_parquet(args.features_path)
    row = select_row(df, target_date_local=args.target_date_local, station_id=args.station_id)
    x = prepare_features(row, feature_columns)
    anchor_tmax_f = float(pd.to_numeric(row["anchor_tmax_f"], errors="coerce").iloc[0])
    calibration_path = args.calibration_path
    if calibration_path is None and DEFAULT_CALIBRATION_PATH.exists():
        calibration_path = DEFAULT_CALIBRATION_PATH
    offsets: dict[str, float] = {}
    if calibration_path is not None:
        offsets = {str(key): float(value) for key, value in load_json(calibration_path).get("offsets_f", {}).items()}

    residual_quantiles: dict[float, float] = {}
    final_quantiles: dict[float, float] = {}
    raw_final_values = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster = lgb.Booster(model_file=str(args.models_dir / f"residual_quantile_{tag}.txt"))
        residual = float(booster.predict(x, num_iteration=booster.best_iteration)[0])
        residual_quantiles[float(quantile)] = residual
        raw_final_values.append(anchor_tmax_f + residual + offsets.get(tag, 0.0))

    rearranged_final_values = np.maximum.accumulate(np.asarray(raw_final_values, dtype=float))
    for quantile, value in zip(DEFAULT_QUANTILES, rearranged_final_values):
        final_quantiles[float(quantile)] = float(value)

    ladder = quantiles_to_degree_ladder(final_quantiles)
    bin_records: list[dict[str, object]] = []
    event_bin_labels = list(args.event_bin)
    if args.event_bins_path is not None:
        event_bin_labels.extend(load_event_bin_labels(args.event_bins_path))
    if event_bin_labels:
        bins = [parse_event_bin(label) for label in event_bin_labels]
        bin_records = map_ladder_to_bins(ladder, bins, max_so_far_f=args.max_so_far_f).to_dict("records")

    payload = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_date_local": args.target_date_local,
        "station_id": args.station_id,
        "anchor_tmax_f": anchor_tmax_f,
        "expected_final_tmax_f": expected_temperature(ladder),
        "residual_quantiles_f": {str(key): value for key, value in residual_quantiles.items()},
        "final_tmax_quantiles_f": {str(key): value for key, value in final_quantiles.items()},
        "degree_ladder": ladder.to_dict("records"),
        "event_bins": bin_records,
        "event_bin_labels": event_bin_labels,
        "max_so_far_f": args.max_so_far_f,
        "calibration_path": str(calibration_path) if calibration_path is not None else None,
        "calibration_offsets_f": offsets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"prediction_{args.station_id}_{args.target_date_local}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print_prediction_summary(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
