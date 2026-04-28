from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .calibrate_quantiles import apply_offsets, fit_offsets
from .calibrate_rolling_origin import score_predictions
from .model_config import DEFAULT_MODEL_CANDIDATE_ID
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet")
DEFAULT_MODEL_SELECTION_SUMMARY_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/model_selection/rolling_origin_model_selection_summary.csv")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/ensemble_diagnostics")

KEY_COLUMNS = [
    "target_date_local",
    "station_id",
    "train_end",
    "valid_start",
    "valid_end",
    "final_tmax_f",
    "target_residual_f",
    "anchor_tmax_f",
    "nbm_tmax_open_f",
    "lamp_tmax_open_f",
    "nbm_minus_lamp_tmax_f",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 7 small-ensemble diagnostics for no-HRRR rolling-origin predictions.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--model-selection-summary-path", type=pathlib.Path, default=DEFAULT_MODEL_SELECTION_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default=DEFAULT_MODEL_CANDIDATE_ID)
    parser.add_argument("--calibration-valid-end", default="2024-12-31")
    parser.add_argument("--test-valid-start", default="2025-01-01")
    parser.add_argument("--ensemble-sizes", default="2,3,5")
    return parser.parse_args()


def parse_ensemble_sizes(value: str) -> list[int]:
    sizes = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes:
        raise ValueError("ensemble sizes must contain at least one integer")
    if any(size < 2 for size in sizes):
        raise ValueError("ensemble sizes must be at least 2")
    return sizes


def ranked_candidates(summary_df: pd.DataFrame) -> list[str]:
    required = {"candidate_id", "weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"}
    missing = required - set(summary_df.columns)
    if missing:
        raise ValueError(f"model selection summary missing required columns: {sorted(missing)}")
    ranked = summary_df.sort_values(["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"])
    return [str(candidate_id) for candidate_id in ranked["candidate_id"].tolist()]


def ensemble_specs(ranked: list[str], *, selected_candidate_id: str, sizes: list[int]) -> list[dict[str, object]]:
    if selected_candidate_id not in ranked:
        raise ValueError(f"selected candidate {selected_candidate_id} not present in ranked candidates")
    specs: list[dict[str, object]] = [
        {
            "method_id": "selected_single",
            "member_candidate_ids": [selected_candidate_id],
        }
    ]
    for size in sizes:
        if len(ranked) >= size:
            specs.append(
                {
                    "method_id": f"top{size}_quantile_mean",
                    "member_candidate_ids": ranked[:size],
                }
            )
    return specs


def build_ensemble_frame(predictions: pd.DataFrame, *, member_candidate_ids: list[str], method_id: str) -> pd.DataFrame:
    work = predictions.loc[predictions["candidate_id"].astype(str).isin(member_candidate_ids)].copy()
    if work.empty:
        raise ValueError(f"no predictions for ensemble {method_id}: {member_candidate_ids}")
    member_count = int(work["candidate_id"].nunique())
    if member_count != len(member_candidate_ids):
        present = sorted(work["candidate_id"].astype(str).unique().tolist())
        raise ValueError(f"ensemble {method_id} missing members; expected={member_candidate_ids} present={present}")
    duplicate_counts = work.groupby([*KEY_COLUMNS, "candidate_id"], dropna=False).size()
    if bool((duplicate_counts > 1).any()):
        raise ValueError(f"ensemble {method_id} has duplicate candidate rows for at least one target key")
    key_member_counts = work.groupby(KEY_COLUMNS, dropna=False)["candidate_id"].nunique()
    missing_key_count = int((key_member_counts != len(member_candidate_ids)).sum())
    if missing_key_count:
        raise ValueError(f"ensemble {method_id} has {missing_key_count} target keys without all requested members")
    quantile_columns = [f"pred_tmax_{quantile_tag(quantile)}_f" for quantile in DEFAULT_QUANTILES]
    grouped = work.groupby(KEY_COLUMNS, dropna=False, sort=False)[quantile_columns].mean().reset_index()
    matrix = grouped[quantile_columns].to_numpy(float)
    matrix = np.maximum.accumulate(matrix, axis=1)
    grouped.loc[:, quantile_columns] = matrix
    grouped["method_id"] = method_id
    grouped["member_count"] = member_count
    grouped["member_candidate_ids"] = ",".join(member_candidate_ids)
    return grouped


def evaluate_method(frame: pd.DataFrame, *, method_id: str, calibration_valid_end: str, test_valid_start: str) -> tuple[dict[str, object], pd.DataFrame, dict[str, float]]:
    calibration_df = frame.loc[frame["target_date_local"].astype(str) <= calibration_valid_end].copy()
    test_df = frame.loc[frame["target_date_local"].astype(str) >= test_valid_start].copy()
    if calibration_df.empty or test_df.empty:
        raise ValueError(f"ensemble method {method_id} requires non-empty calibration and test slices")
    offsets = fit_offsets(calibration_df)
    calibrated_test = apply_offsets(test_df, offsets)
    metrics = score_predictions(calibrated_test, method_id=method_id, prediction_set="calibrated_pred")
    metrics["member_count"] = int(frame["member_count"].iloc[0])
    metrics["member_candidate_ids"] = str(frame["member_candidate_ids"].iloc[0])
    metrics["calibration_row_count"] = int(len(calibration_df))
    metrics["test_row_count"] = int(len(test_df))
    calibrated_test["method_id"] = method_id
    calibrated_test["member_count"] = int(frame["member_count"].iloc[0])
    calibrated_test["member_candidate_ids"] = str(frame["member_candidate_ids"].iloc[0])
    return metrics, calibrated_test, offsets


def main() -> int:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions_path)
    if predictions.empty:
        raise ValueError("ensemble diagnostics require non-empty rolling-origin predictions")
    summary = pd.read_csv(args.model_selection_summary_path)
    ranked = ranked_candidates(summary)
    specs = ensemble_specs(ranked, selected_candidate_id=str(args.candidate_id), sizes=parse_ensemble_sizes(args.ensemble_sizes))

    metrics_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    method_configs: dict[str, object] = {}
    for spec in specs:
        method_id = str(spec["method_id"])
        member_candidate_ids = [str(candidate_id) for candidate_id in spec["member_candidate_ids"]]
        frame = build_ensemble_frame(predictions, member_candidate_ids=member_candidate_ids, method_id=method_id)
        metrics, calibrated_test, offsets = evaluate_method(
            frame,
            method_id=method_id,
            calibration_valid_end=str(args.calibration_valid_end),
            test_valid_start=str(args.test_valid_start),
        )
        metrics_rows.append(metrics)
        prediction_frames.append(calibrated_test)
        method_configs[method_id] = {
            "member_candidate_ids": member_candidate_ids,
            "offsets_f": offsets,
        }

    summary_df = pd.DataFrame(metrics_rows).sort_values(["event_bin_nll", "degree_ladder_nll"]).reset_index(drop=True)
    selected_method_id = str(summary_df.iloc[0]["method_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ensemble_diagnostics_summary.csv"
    predictions_path = args.output_dir / "ensemble_diagnostics_predictions.parquet"
    manifest_path = args.output_dir / "ensemble_diagnostics_manifest.json"
    summary_df.to_csv(summary_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predictions_path": str(args.predictions_path),
        "model_selection_summary_path": str(args.model_selection_summary_path),
        "output_dir": str(args.output_dir),
        "candidate_id": str(args.candidate_id),
        "calibration_valid_end": str(args.calibration_valid_end),
        "test_valid_start": str(args.test_valid_start),
        "selected_method_id": selected_method_id,
        "selected_by": ["event_bin_nll", "degree_ladder_nll"],
        "summary_path": str(summary_path),
        "ensemble_predictions_path": str(predictions_path),
        "methods": method_configs,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(summary_path)
    print(predictions_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
