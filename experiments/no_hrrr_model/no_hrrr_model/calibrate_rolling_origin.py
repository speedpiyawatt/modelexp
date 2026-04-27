from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .calibrate_quantiles import apply_offsets, coverage_rows, fit_offsets
from .evaluate import mae, rmse


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/rolling_origin_predictions.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit rolling-origin calibration on earlier holdouts and test on later holdouts.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--calibration-valid-end", default="2024-12-31")
    parser.add_argument("--test-valid-start", default="2025-01-01")
    return parser.parse_args()


def interval_score(df: pd.DataFrame, *, prefix: str, lower_tag: str, upper_tag: str, alpha: float) -> float:
    y = pd.to_numeric(df["final_tmax_f"], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(df[f"{prefix}_tmax_{lower_tag}_f"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(df[f"{prefix}_tmax_{upper_tag}_f"], errors="coerce").to_numpy(float)
    width = upper - lower
    lower_penalty = (2.0 / alpha) * np.maximum(lower - y, 0.0)
    upper_penalty = (2.0 / alpha) * np.maximum(y - upper, 0.0)
    return float(np.mean(width + lower_penalty + upper_penalty))


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.predictions_path)
    calibration_df = df.loc[df["target_date_local"].astype(str) <= args.calibration_valid_end].copy()
    test_df = df.loc[df["target_date_local"].astype(str) >= args.test_valid_start].copy()
    if calibration_df.empty or test_df.empty:
        raise ValueError("rolling-origin calibration requires non-empty calibration and test slices")
    offsets = fit_offsets(calibration_df)
    calibrated_test = apply_offsets(test_df, offsets)
    coverage = pd.DataFrame(coverage_rows(calibrated_test))
    summary = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predictions_path": str(args.predictions_path),
        "calibration_valid_end": args.calibration_valid_end,
        "test_valid_start": args.test_valid_start,
        "calibration_row_count": int(len(calibration_df)),
        "test_row_count": int(len(test_df)),
        "offsets_f": offsets,
        "uncalibrated_q50_mae_f": mae(test_df["final_tmax_f"], test_df["pred_tmax_q50_f"]),
        "calibrated_q50_mae_f": mae(calibrated_test["final_tmax_f"], calibrated_test["calibrated_pred_tmax_q50_f"]),
        "uncalibrated_q50_rmse_f": rmse(test_df["final_tmax_f"], test_df["pred_tmax_q50_f"]),
        "calibrated_q50_rmse_f": rmse(calibrated_test["final_tmax_f"], calibrated_test["calibrated_pred_tmax_q50_f"]),
        "uncalibrated_q05_q95_interval_score": interval_score(test_df, prefix="pred", lower_tag="q05", upper_tag="q95", alpha=0.10),
        "calibrated_q05_q95_interval_score": interval_score(calibrated_test, prefix="calibrated_pred", lower_tag="q05", upper_tag="q95", alpha=0.10),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "rolling_origin_calibration_manifest.json"
    coverage_path = args.output_dir / "rolling_origin_calibrated_coverage.csv"
    predictions_path = args.output_dir / "rolling_origin_predictions_calibrated_test.parquet"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    coverage.to_csv(coverage_path, index=False)
    calibrated_test.to_parquet(predictions_path, index=False)
    print(manifest_path)
    print(coverage_path)
    print(predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
