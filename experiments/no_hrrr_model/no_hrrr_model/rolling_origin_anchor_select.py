from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .rolling_origin_model_select import (
    DEFAULT_CANDIDATES,
    DEFAULT_FEATURES_PATH,
    DEFAULT_MODEL_CANDIDATE_ID,
    DEFAULT_SPLITS,
    candidate_by_id,
    evaluate_candidate_split,
    leakage_findings,
    load_splits,
    summarize_candidates,
)
from .train_quantile_models import select_feature_columns


DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/anchor_selection")
DEFAULT_WEIGHT_GRID: tuple[float, ...] = tuple(round(value / 10.0, 1) for value in range(0, 11))
RIDGE_L2 = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin anchor selection for no-HRRR residual quantile models.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--splits-path", type=pathlib.Path, default=None)
    return parser.parse_args()


def anchor_candidate_specs() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for weight in DEFAULT_WEIGHT_GRID:
        candidates.append({"anchor_candidate_id": f"fixed_nbm_weight_{weight:.1f}", "anchor_type": "fixed_weight", "weight": weight})
    candidates.extend(
        [
            {"anchor_candidate_id": "segmented_season_train_mae", "anchor_type": "segmented_season"},
            {"anchor_candidate_id": "segmented_month_train_mae", "anchor_type": "segmented_month"},
            {"anchor_candidate_id": "segmented_disagreement_train_mae", "anchor_type": "segmented_disagreement"},
            {"anchor_candidate_id": "ridge_linear_anchor", "anchor_type": "ridge_linear"},
        ]
    )
    return candidates


def fixed_anchor(df: pd.DataFrame, weight: float) -> pd.Series:
    nbm = pd.to_numeric(df["nbm_tmax_open_f"], errors="coerce")
    lamp = pd.to_numeric(df["lamp_tmax_open_f"], errors="coerce")
    return weight * nbm + (1.0 - weight) * lamp


def anchor_mae(df: pd.DataFrame, weight: float) -> float:
    pred = fixed_anchor(df, weight)
    observed = pd.to_numeric(df["final_tmax_f"], errors="coerce")
    return float((observed - pred).abs().mean())


def best_weight_for_subset(df: pd.DataFrame, *, fallback_weight: float) -> float:
    if df.empty:
        return fallback_weight
    scored = [(anchor_mae(df, weight), weight) for weight in DEFAULT_WEIGHT_GRID]
    finite_scores = [(score, weight) for score, weight in scored if np.isfinite(score)]
    if not finite_scores:
        return fallback_weight
    return float(min(finite_scores, key=lambda item: (item[0], abs(item[1] - 0.5)))[1])


def season_segment(df: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(df["target_date_local"]).dt.month
    return months.between(4, 10).map({True: "warm_apr_oct", False: "cool_nov_mar"})


def month_segment(df: pd.DataFrame) -> pd.Series:
    months = pd.to_datetime(df["target_date_local"]).dt.month
    return months.map(lambda month: f"month_{int(month):02d}" if pd.notna(month) else np.nan)


def disagreement_segment(df: pd.DataFrame) -> pd.Series:
    abs_disagreement = pd.to_numeric(df["nbm_minus_lamp_tmax_f"], errors="coerce").abs()
    return pd.cut(
        abs_disagreement,
        bins=[-np.inf, 2.0, 5.0, np.inf],
        labels=["lt_2f", "2_to_5f", "gte_5f"],
        right=False,
    ).astype(str)


def segment_weights(train_df: pd.DataFrame, segment_values: pd.Series) -> dict[str, float]:
    fallback = best_weight_for_subset(train_df, fallback_weight=0.5)
    weights: dict[str, float] = {"__fallback__": fallback}
    segment_labels = segment_values.astype(str)
    for segment in sorted(str(value) for value in segment_values.dropna().unique()):
        subset = train_df.loc[segment_labels == segment]
        weights[segment] = best_weight_for_subset(subset, fallback_weight=fallback)
    return weights


def apply_segmented_anchor(df: pd.DataFrame, segment_values: pd.Series, weights: dict[str, float]) -> pd.Series:
    anchor = pd.Series(index=df.index, dtype=float)
    fallback = float(weights["__fallback__"])
    segment_labels = segment_values.astype(str)
    for segment in segment_labels.unique():
        weight = float(weights.get(str(segment), fallback))
        mask = segment_labels == str(segment)
        anchor.loc[mask] = fixed_anchor(df.loc[mask], weight)
    return anchor


def ridge_design_matrix(df: pd.DataFrame) -> np.ndarray:
    nbm = pd.to_numeric(df["nbm_tmax_open_f"], errors="coerce").to_numpy(float)
    lamp = pd.to_numeric(df["lamp_tmax_open_f"], errors="coerce").to_numpy(float)
    disagreement = pd.to_numeric(df["nbm_minus_lamp_tmax_f"], errors="coerce").abs().to_numpy(float)
    months = pd.to_datetime(df["target_date_local"]).dt.month.to_numpy(float)
    radians = 2.0 * np.pi * (months - 1.0) / 12.0
    return np.column_stack([np.ones(len(df)), nbm, lamp, disagreement, np.sin(radians), np.cos(radians)])


def fit_ridge_anchor(train_df: pd.DataFrame, *, l2: float = RIDGE_L2) -> np.ndarray:
    x = ridge_design_matrix(train_df)
    y = pd.to_numeric(train_df["final_tmax_f"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if int(keep.sum()) < x.shape[1]:
        raise ValueError("not enough finite rows to fit ridge anchor")
    penalty = np.eye(x.shape[1]) * float(l2)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x[keep].T @ x[keep] + penalty, x[keep].T @ y[keep])


def apply_ridge_anchor(df: pd.DataFrame, coefficients: np.ndarray) -> pd.Series:
    return pd.Series(ridge_design_matrix(df) @ coefficients, index=df.index, dtype=float)


def apply_anchor(df: pd.DataFrame, anchor: pd.Series) -> pd.DataFrame:
    out = df.copy()
    out["anchor_tmax_f"] = pd.to_numeric(anchor, errors="coerce")
    out["target_residual_f"] = pd.to_numeric(out["final_tmax_f"], errors="coerce") - out["anchor_tmax_f"]
    return out


def transformed_fold_df(
    df: pd.DataFrame,
    spec: dict[str, object],
    *,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    fold_df = df.loc[(df["target_date_local"].astype(str) <= train_end) | ((df["target_date_local"].astype(str) >= valid_start) & (df["target_date_local"].astype(str) <= valid_end))].copy()
    train_mask = fold_df["target_date_local"].astype(str) <= train_end
    train_df = fold_df.loc[train_mask].copy()
    anchor_type = str(spec["anchor_type"])
    metadata: dict[str, object] = {"anchor_type": anchor_type}
    if anchor_type == "fixed_weight":
        weight = float(spec["weight"])
        metadata["weight"] = weight
        return apply_anchor(fold_df, fixed_anchor(fold_df, weight)), metadata
    if anchor_type == "segmented_season":
        train_segments = season_segment(train_df)
        weights = segment_weights(train_df, train_segments)
        metadata["segment_weights"] = weights
        return apply_anchor(fold_df, apply_segmented_anchor(fold_df, season_segment(fold_df), weights)), metadata
    if anchor_type == "segmented_month":
        train_segments = month_segment(train_df)
        weights = segment_weights(train_df, train_segments)
        metadata["segment_weights"] = weights
        return apply_anchor(fold_df, apply_segmented_anchor(fold_df, month_segment(fold_df), weights)), metadata
    if anchor_type == "segmented_disagreement":
        train_segments = disagreement_segment(train_df)
        weights = segment_weights(train_df, train_segments)
        metadata["segment_weights"] = weights
        return apply_anchor(fold_df, apply_segmented_anchor(fold_df, disagreement_segment(fold_df), weights)), metadata
    if anchor_type == "ridge_linear":
        coefficients = fit_ridge_anchor(train_df)
        metadata["coefficients"] = [float(value) for value in coefficients]
        metadata["features"] = ["intercept", "nbm_tmax_open_f", "lamp_tmax_open_f", "abs_nbm_minus_lamp_tmax_f", "month_sin", "month_cos"]
        metadata["l2"] = RIDGE_L2
        return apply_anchor(fold_df, apply_ridge_anchor(fold_df, coefficients)), metadata
    raise ValueError(f"unknown anchor_type: {anchor_type}")


def main() -> int:
    args = parse_args()
    splits = load_splits(args.splits_path)
    df = pd.read_parquet(args.features_path)
    df = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    feature_columns = select_feature_columns(df)
    leakage = leakage_findings(feature_columns)
    if leakage:
        raise ValueError(f"leakage-prone feature columns selected: {leakage[:10]}")
    model_candidate = candidate_by_id(DEFAULT_MODEL_CANDIDATE_ID, DEFAULT_CANDIDATES)

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    anchor_metadata_rows: list[dict[str, object]] = []
    for spec in anchor_candidate_specs():
        anchor_candidate_id = str(spec["anchor_candidate_id"])
        for train_end, valid_start, valid_end in splits:
            fold_df, metadata = transformed_fold_df(df, spec, train_end=train_end, valid_start=valid_start, valid_end=valid_end)
            candidate = dict(model_candidate)
            candidate["candidate_id"] = anchor_candidate_id
            metrics, predictions = evaluate_candidate_split(
                fold_df,
                feature_columns,
                candidate,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
            metrics["anchor_type"] = spec["anchor_type"]
            predictions["anchor_candidate_id"] = anchor_candidate_id
            fold_rows.append(metrics)
            prediction_frames.append(predictions)
            anchor_metadata_rows.append(
                {
                    "anchor_candidate_id": anchor_candidate_id,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "metadata": json.dumps(metadata, sort_keys=True),
                }
            )

    metrics_df = pd.DataFrame(fold_rows)
    summary_df = summarize_candidates(metrics_df).rename(columns={"candidate_id": "anchor_candidate_id"})
    selected_anchor_id = str(summary_df.iloc[0]["anchor_candidate_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = args.output_dir / "rolling_origin_anchor_selection_fold_metrics.csv"
    summary_path = args.output_dir / "rolling_origin_anchor_selection_summary.csv"
    predictions_path = args.output_dir / "rolling_origin_anchor_selection_predictions.parquet"
    metadata_path = args.output_dir / "rolling_origin_anchor_selection_anchor_metadata.csv"
    manifest_path = args.output_dir / "rolling_origin_anchor_selection_manifest.json"
    metrics_df.to_csv(fold_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    pd.DataFrame(anchor_metadata_rows).to_csv(metadata_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "output_dir": str(args.output_dir),
        "fold_metrics_path": str(fold_metrics_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "metadata_path": str(metadata_path),
        "selected_anchor_candidate_id": selected_anchor_id,
        "selected_by": ["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"],
        "anchor_candidate_count": len(anchor_candidate_specs()),
        "split_count": len(splits),
        "splits": [{"train_end": train_end, "valid_start": valid_start, "valid_end": valid_end} for train_end, valid_start, valid_end in splits],
        "model_candidate": model_candidate,
        "feature_count": len(feature_columns),
        "leakage_check": {"status": "ok", "finding_count": 0},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(fold_metrics_path)
    print(summary_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
