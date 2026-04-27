from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .evaluate import coverage_rate
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/validation_predictions.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit simple empirical quantile offsets for no-HRRR final-Tmax quantiles.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def fit_offsets(df: pd.DataFrame) -> dict[str, float]:
    y = pd.to_numeric(df["final_tmax_f"], errors="coerce").to_numpy(float)
    offsets: dict[str, float] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        pred = pd.to_numeric(df[f"pred_tmax_{tag}_f"], errors="coerce").to_numpy(float)
        residual = y - pred
        keep = np.isfinite(residual)
        offsets[tag] = float(np.quantile(residual[keep], quantile))
    return offsets


def apply_offsets(df: pd.DataFrame, offsets: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    columns = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        column = f"calibrated_pred_tmax_{tag}_f"
        out[column] = pd.to_numeric(out[f"pred_tmax_{tag}_f"], errors="coerce") + float(offsets[tag])
        columns.append(column)
    out.loc[:, columns] = np.maximum.accumulate(out[columns].to_numpy(float), axis=1)
    return out


def coverage_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for prefix in ("pred", "calibrated_pred"):
        for lower_tag, upper_tag in (("q05", "q95"), ("q10", "q90"), ("q25", "q75")):
            lower = df[f"{prefix}_tmax_{lower_tag}_f"].to_numpy(float)
            upper = df[f"{prefix}_tmax_{upper_tag}_f"].to_numpy(float)
            rows.append(
                {
                    "prediction_set": prefix,
                    "interval": f"{lower_tag}_{upper_tag}",
                    "coverage": coverage_rate(df["final_tmax_f"], lower, upper),
                    "mean_width_f": float(np.mean(upper - lower)),
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.predictions_path)
    offsets = fit_offsets(df)
    calibrated = apply_offsets(df, offsets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_path = args.output_dir / "validation_predictions_calibrated.parquet"
    calibrated.to_parquet(calibrated_path, index=False)
    coverage_path = args.output_dir / "calibrated_quantile_coverage.csv"
    pd.DataFrame(coverage_rows(calibrated)).to_csv(coverage_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "method": "empirical_validation_quantile_offsets",
        "predictions_path": str(args.predictions_path),
        "calibrated_predictions_path": str(calibrated_path),
        "coverage_path": str(coverage_path),
        "row_count": int(len(df)),
        "offsets_f": offsets,
    }
    manifest_path = args.output_dir / "calibration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(coverage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
