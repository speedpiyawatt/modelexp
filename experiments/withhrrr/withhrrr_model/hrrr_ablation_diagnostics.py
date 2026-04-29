from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import pandas as pd

from .model_config import DEFAULT_MODEL_CANDIDATE_ID, candidate_by_id
from .rolling_origin_model_select import (
    DEFAULT_FEATURES_PATH,
    DEFAULT_OUTPUT_DIR as DEFAULT_MODEL_SELECTION_DIR,
    evaluate_candidate_split,
    leakage_findings,
    load_splits,
    select_feature_columns,
    weighted_mean,
)


DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/hrrr_ablation")
HRRR_EXACT_FEATURES = {
    "hrrr_tmax_open_f",
    "hrrr_minus_lamp_tmax_f",
    "hrrr_minus_nbm_tmax_f",
    "abs_hrrr_minus_lamp_tmax_f",
    "abs_hrrr_minus_nbm_tmax_f",
    "anchor_equal_3way_tmax_f",
    "hrrr_above_nbm_lamp_range_f",
    "hrrr_below_nbm_lamp_range_f",
    "hrrr_outside_nbm_lamp_range_f",
    "hrrr_hotter_than_lamp_3f",
    "hrrr_colder_than_lamp_3f",
    "hrrr_hotter_than_nbm_3f",
    "hrrr_colder_than_nbm_3f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run with-HRRR rolling-origin ablation with HRRR feature columns dropped.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-selection-dir", type=pathlib.Path, default=DEFAULT_MODEL_SELECTION_DIR)
    parser.add_argument("--splits-path", type=pathlib.Path, default=None)
    parser.add_argument("--candidate-id", default=DEFAULT_MODEL_CANDIDATE_ID)
    return parser.parse_args()


def drop_hrrr_feature_columns(feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if column not in HRRR_EXACT_FEATURES
        and not column.startswith("hrrr_")
        and not column.startswith("meta_hrrr_")
    ]


def summarize_fold_metrics(metrics_df: pd.DataFrame, *, model_variant: str) -> dict[str, object]:
    row: dict[str, object] = {
        "model_variant": model_variant,
        "fold_count": int(len(metrics_df)),
        "validation_row_count": int(metrics_df["validation_row_count"].sum()),
    }
    for column in (
        "degree_ladder_nll",
        "degree_ladder_brier",
        "degree_ladder_rps",
        "event_bin_nll",
        "event_bin_brier",
        "final_tmax_q50_mae_f",
        "final_tmax_q50_rmse_f",
        "q05_q95_coverage",
    ):
        row[f"weighted_mean_{column}"] = weighted_mean(metrics_df, column)
    return row


def reference_metrics(model_selection_dir: pathlib.Path, *, candidate_id: str) -> pd.DataFrame:
    path = model_selection_dir / "rolling_origin_model_selection_fold_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(path)
    if "candidate_id" not in metrics.columns:
        return pd.DataFrame()
    return metrics.loc[metrics["candidate_id"].astype(str) == str(candidate_id)].copy()


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.features_path)
    df = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    full_feature_columns = select_feature_columns(df)
    ablated_feature_columns = drop_hrrr_feature_columns(full_feature_columns)
    leakage = leakage_findings(ablated_feature_columns)
    if leakage:
        raise ValueError(f"leakage-prone ablation feature columns selected: {leakage[:10]}")
    if len(ablated_feature_columns) >= len(full_feature_columns):
        raise ValueError("HRRR ablation did not drop any feature columns")

    candidate = candidate_by_id(args.candidate_id)
    splits = load_splits(args.splits_path)
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for train_end, valid_start, valid_end in splits:
        metrics, predictions = evaluate_candidate_split(
            df,
            ablated_feature_columns,
            candidate,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        metrics["model_variant"] = "without_hrrr_features"
        fold_rows.append(metrics)
        prediction_frames.append(predictions)

    ablation_metrics = pd.DataFrame(fold_rows)
    comparison_rows = [summarize_fold_metrics(ablation_metrics, model_variant="without_hrrr_features")]
    reference = reference_metrics(args.model_selection_dir, candidate_id=args.candidate_id)
    if not reference.empty:
        reference = reference.copy()
        reference["model_variant"] = "with_hrrr_features_reference"
        comparison_rows.insert(0, summarize_fold_metrics(reference, model_variant="with_hrrr_features_reference"))
    comparison = pd.DataFrame(comparison_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = args.output_dir / "hrrr_ablation_fold_metrics.csv"
    comparison_path = args.output_dir / "hrrr_ablation_comparison.csv"
    predictions_path = args.output_dir / "hrrr_ablation_predictions.parquet"
    manifest_path = args.output_dir / "hrrr_ablation_manifest.json"
    ablation_metrics.to_csv(fold_metrics_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "candidate_id": args.candidate_id,
        "full_feature_count": len(full_feature_columns),
        "ablated_feature_count": len(ablated_feature_columns),
        "dropped_feature_count": len(full_feature_columns) - len(ablated_feature_columns),
        "dropped_feature_examples": [column for column in full_feature_columns if column not in ablated_feature_columns][:50],
        "fold_metrics_path": str(fold_metrics_path),
        "comparison_path": str(comparison_path),
        "predictions_path": str(predictions_path),
        "selected_by": ["event_bin_nll", "degree_ladder_nll"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
