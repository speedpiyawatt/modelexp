from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluate import mae, prepare_features, rmse
from .train_quantile_models import DEFAULT_QUANTILES, pinball_loss, quantile_tag, select_feature_columns


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation")
RANDOM_SEED = 20260425


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin validation for the no-HRRR residual quantile model.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def train_quantile_predict(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, quantile: float) -> np.ndarray:
    train_data = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }
    booster = lgb.train(params, train_data, num_boost_round=250)
    return booster.predict(valid_x)


def evaluate_split(df: pd.DataFrame, feature_columns: list[str], *, train_end: str, valid_start: str, valid_end: str) -> tuple[dict[str, object], pd.DataFrame]:
    train_df = df.loc[df["target_date_local"].astype(str) <= train_end].copy()
    valid_df = df.loc[(df["target_date_local"].astype(str) >= valid_start) & (df["target_date_local"].astype(str) <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"empty rolling split train_end={train_end} valid_start={valid_start} valid_end={valid_end}")
    train_x = prepare_features(train_df, feature_columns)
    valid_x = prepare_features(valid_df, feature_columns)
    train_y_residual = pd.to_numeric(train_df["target_residual_f"], errors="coerce")
    valid_y_residual = pd.to_numeric(valid_df["target_residual_f"], errors="coerce")
    valid_y_final = pd.to_numeric(valid_df["final_tmax_f"], errors="coerce")
    anchor = pd.to_numeric(valid_df["anchor_tmax_f"], errors="coerce").to_numpy(float)
    residual_predictions: dict[str, np.ndarray] = {}
    final_predictions: dict[str, np.ndarray] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        residual_pred = train_quantile_predict(train_x, train_y_residual, valid_x, quantile)
        residual_predictions[tag] = residual_pred
        final_predictions[tag] = anchor + residual_pred
    final_matrix = np.column_stack([final_predictions[quantile_tag(q)] for q in DEFAULT_QUANTILES])
    final_matrix = np.maximum.accumulate(final_matrix, axis=1)
    for index, quantile in enumerate(DEFAULT_QUANTILES):
        final_predictions[quantile_tag(quantile)] = final_matrix[:, index]
    q50 = final_predictions["q50"]
    q05 = final_predictions["q05"]
    q95 = final_predictions["q95"]
    predictions = valid_df[["target_date_local", "station_id", "final_tmax_f", "target_residual_f", "anchor_tmax_f", "nbm_tmax_open_f", "lamp_tmax_open_f", "nbm_minus_lamp_tmax_f"]].copy()
    predictions["train_end"] = train_end
    predictions["valid_start"] = valid_start
    predictions["valid_end"] = valid_end
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        predictions[f"pred_residual_{tag}_f"] = residual_predictions[tag]
        predictions[f"pred_tmax_{tag}_f"] = final_predictions[tag]
    metrics = {
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(valid_df)),
        "residual_q50_mae_f": mae(valid_y_residual, residual_predictions["q50"]),
        "residual_q50_rmse_f": rmse(valid_y_residual, residual_predictions["q50"]),
        "final_tmax_q50_mae_f": mae(valid_y_final, q50),
        "final_tmax_q50_rmse_f": rmse(valid_y_final, q50),
        "fixed_anchor_mae_f": mae(valid_y_final, valid_df["anchor_tmax_f"]),
        "fixed_anchor_rmse_f": rmse(valid_y_final, valid_df["anchor_tmax_f"]),
        "q05_q95_coverage": float(np.mean((valid_y_final.to_numpy(float) >= q05) & (valid_y_final.to_numpy(float) <= q95))),
        "q05_q95_mean_width_f": float(np.mean(q95 - q05)),
        "q50_pinball_loss": pinball_loss(valid_y_residual.to_numpy(float), residual_predictions["q50"], 0.50),
    }
    return metrics, predictions


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features_path)
    df = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    feature_columns = select_feature_columns(df)
    splits = [
        ("2023-12-31", "2024-01-01", "2024-12-31"),
        ("2024-12-31", "2025-01-01", "2025-12-31"),
    ]
    rows = []
    prediction_frames = []
    for train_end, valid_start, valid_end in splits:
        metrics, predictions = evaluate_split(df, feature_columns, train_end=train_end, valid_start=valid_start, valid_end=valid_end)
        rows.append(metrics)
        prediction_frames.append(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "rolling_origin_metrics.csv"
    predictions_path = args.output_dir / "rolling_origin_predictions.parquet"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "output_path": str(output_path),
        "predictions_path": str(predictions_path),
        "split_count": len(rows),
        "feature_count": len(feature_columns),
    }
    (args.output_dir / "rolling_origin_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output_path)
    print(args.output_dir / "rolling_origin_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
