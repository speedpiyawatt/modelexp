from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .evaluate import (
    DEFAULT_REPRESENTATIVE_EVENT_BINS,
    build_degree_ladder_diagnostics,
    build_event_bin_diagnostics,
    mae,
    prepare_features,
    representative_event_bins,
    rmse,
)
from .model_config import DEFAULT_CANDIDATES, DEFAULT_MODEL_CANDIDATE_ID, DEFAULT_QUANTILES, RANDOM_SEED, candidate_by_id
from .train_quantile_models import pinball_loss, quantile_tag, select_feature_columns


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation/model_selection")

DEFAULT_SPLITS: tuple[tuple[str, str, str], ...] = (
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-12-31"),
)
FORBIDDEN_EXACT_FEATURES = {
    "target_date_local",
    "station_id",
    "selection_cutoff_local",
    "final_tmax_f",
    "final_tmin_f",
    "target_residual_f",
    "model_training_eligible",
}
FORBIDDEN_FEATURE_PREFIXES = ("label_", "hrrr_", "meta_hrrr_")
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "_time_utc_code",
    "_time_local_code",
    "_source_model_code",
    "_source_product_code",
    "_source_version_code",
    "market",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin model selection for no-HRRR residual quantile models.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-config-path", type=pathlib.Path, default=None)
    parser.add_argument("--splits-path", type=pathlib.Path, default=None)
    return parser.parse_args()


def load_candidates(path: pathlib.Path | None) -> list[dict[str, object]]:
    if path is None:
        return [dict(candidate) for candidate in DEFAULT_CANDIDATES]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        candidates = payload.get("candidates")
    else:
        candidates = payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate config must be a non-empty list or contain a non-empty candidates list")
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or "candidate_id" not in candidate or "params" not in candidate:
            raise ValueError("each candidate must contain candidate_id and params")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
    return candidates


def load_splits(path: pathlib.Path | None) -> list[tuple[str, str, str]]:
    if path is None:
        return list(DEFAULT_SPLITS)
    payload = json.loads(path.read_text())
    items = payload.get("splits") if isinstance(payload, dict) else payload
    if not isinstance(items, list) or not items:
        raise ValueError("splits config must be a non-empty list or contain a non-empty splits list")
    splits: list[tuple[str, str, str]] = []
    for item in items:
        if isinstance(item, dict):
            train_end = item.get("train_end")
            valid_start = item.get("valid_start")
            valid_end = item.get("valid_end")
        elif isinstance(item, list | tuple) and len(item) == 3:
            train_end, valid_start, valid_end = item
        else:
            raise ValueError("each split must be a dict with train_end/valid_start/valid_end or a 3-item list")
        if train_end is None or valid_start is None or valid_end is None:
            raise ValueError("each split must include train_end, valid_start, and valid_end")
        train_end_date = dt.date.fromisoformat(str(train_end))
        valid_start_date = dt.date.fromisoformat(str(valid_start))
        valid_end_date = dt.date.fromisoformat(str(valid_end))
        if train_end_date >= valid_start_date:
            raise ValueError("each split must satisfy train_end < valid_start to avoid validation leakage")
        if valid_start_date > valid_end_date:
            raise ValueError("each split must satisfy valid_start <= valid_end")
        splits.append((str(train_end), str(valid_start), str(valid_end)))
    return splits


def leakage_findings(feature_columns: list[str]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for column in feature_columns:
        if column in FORBIDDEN_EXACT_FEATURES:
            findings.append({"feature": column, "reason": "forbidden exact feature"})
        if column.startswith(FORBIDDEN_FEATURE_PREFIXES):
            findings.append({"feature": column, "reason": "forbidden prefix"})
        for token in FORBIDDEN_FEATURE_SUBSTRINGS:
            if token in column:
                findings.append({"feature": column, "reason": f"forbidden substring {token}"})
    return findings


def lightgbm_params(candidate: dict[str, object], quantile: float) -> dict[str, object]:
    base = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }
    params = candidate.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"candidate {candidate.get('candidate_id')} params must be a dict")
    base.update(params)
    return base


def train_quantile_prediction(
    *,
    candidate: dict[str, object],
    quantile: float,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
) -> np.ndarray:
    train_data = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    booster = lgb.train(
        lightgbm_params(candidate, quantile),
        train_data,
        num_boost_round=int(candidate.get("num_boost_round", 250)),
    )
    return booster.predict(valid_x)


def evaluate_candidate_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    candidate: dict[str, object],
    *,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_df = df.loc[df["target_date_local"].astype(str) <= train_end].copy()
    valid_df = df.loc[(df["target_date_local"].astype(str) >= valid_start) & (df["target_date_local"].astype(str) <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"empty split for train_end={train_end} valid_start={valid_start} valid_end={valid_end}")

    train_x = prepare_features(train_df, feature_columns)
    valid_x = prepare_features(valid_df, feature_columns)
    train_y_residual = pd.to_numeric(train_df["target_residual_f"], errors="coerce")
    valid_y_residual = pd.to_numeric(valid_df["target_residual_f"], errors="coerce")
    valid_y_final = pd.to_numeric(valid_df["final_tmax_f"], errors="coerce")
    anchor = pd.to_numeric(valid_df["anchor_tmax_f"], errors="coerce").to_numpy(float)

    predictions = valid_df[
        [
            "target_date_local",
            "station_id",
            "final_tmax_f",
            "target_residual_f",
            "anchor_tmax_f",
            "nbm_tmax_open_f",
            "lamp_tmax_open_f",
            "nbm_minus_lamp_tmax_f",
        ]
    ].copy()
    predictions["candidate_id"] = str(candidate["candidate_id"])
    predictions["train_end"] = train_end
    predictions["valid_start"] = valid_start
    predictions["valid_end"] = valid_end

    residual_predictions: dict[str, np.ndarray] = {}
    final_predictions: dict[str, np.ndarray] = {}
    pinball_rows: dict[str, float] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        residual_pred = train_quantile_prediction(
            candidate=candidate,
            quantile=quantile,
            train_x=train_x,
            train_y=train_y_residual,
            valid_x=valid_x,
        )
        residual_predictions[tag] = residual_pred
        final_predictions[tag] = anchor + residual_pred
        pinball_rows[f"{tag}_pinball_loss"] = pinball_loss(valid_y_residual.to_numpy(float), residual_pred, quantile)

    final_matrix = np.column_stack([final_predictions[quantile_tag(q)] for q in DEFAULT_QUANTILES])
    final_matrix = np.maximum.accumulate(final_matrix, axis=1)
    for index, quantile in enumerate(DEFAULT_QUANTILES):
        tag = quantile_tag(quantile)
        final_predictions[tag] = final_matrix[:, index]
        predictions[f"pred_residual_{tag}_f"] = residual_predictions[tag]
        predictions[f"pred_tmax_{tag}_f"] = final_predictions[tag]

    degree_scores, _, _, pit_diagnostics, degree_metrics = build_degree_ladder_diagnostics(predictions)
    event_scores, _, _, event_metrics = build_event_bin_diagnostics(predictions)
    q05 = final_predictions["q05"]
    q95 = final_predictions["q95"]
    q50 = final_predictions["q50"]
    metrics = {
        "candidate_id": str(candidate["candidate_id"]),
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(valid_df)),
        "final_tmax_q50_mae_f": mae(valid_y_final, q50),
        "final_tmax_q50_rmse_f": rmse(valid_y_final, q50),
        "residual_q50_mae_f": mae(valid_y_residual, residual_predictions["q50"]),
        "residual_q50_rmse_f": rmse(valid_y_residual, residual_predictions["q50"]),
        "fixed_anchor_mae_f": mae(valid_y_final, valid_df["anchor_tmax_f"]),
        "fixed_anchor_rmse_f": rmse(valid_y_final, valid_df["anchor_tmax_f"]),
        "q05_q95_coverage": float(np.mean((valid_y_final.to_numpy(float) >= q05) & (valid_y_final.to_numpy(float) <= q95))),
        "q05_q95_mean_width_f": float(np.mean(q95 - q05)),
        "degree_ladder_nll": degree_metrics["mean_negative_log_likelihood"],
        "degree_ladder_brier": degree_metrics["mean_brier_score"],
        "degree_ladder_rps": degree_metrics["mean_ranked_probability_score"],
        "degree_ladder_observed_probability": degree_metrics["mean_observed_probability"],
        "event_bin_nll": event_metrics["mean_negative_log_likelihood"],
        "event_bin_brier": event_metrics["mean_brier_score"],
        "event_bin_observed_probability": event_metrics["mean_observed_bin_probability"],
        "pit_mean": float(pit_diagnostics.loc[pit_diagnostics["slice"] == "overall", "pit_mean"].iloc[0]),
        "pit_std": float(pit_diagnostics.loc[pit_diagnostics["slice"] == "overall", "pit_std"].iloc[0]),
        "zero_degree_observed_probability_count": int((degree_scores["observed_probability"] <= 0.0).sum()),
        "zero_event_observed_probability_count": int((event_scores["observed_bin_probability"] <= 0.0).sum()),
        **pinball_rows,
    }
    return metrics, predictions


def weighted_mean(group: pd.DataFrame, column: str) -> float:
    weights = pd.to_numeric(group["validation_row_count"], errors="coerce").to_numpy(float)
    values = pd.to_numeric(group[column], errors="coerce").to_numpy(float)
    keep = np.isfinite(weights) & np.isfinite(values) & (weights > 0)
    if not bool(keep.any()):
        return float("nan")
    return float(np.average(values[keep], weights=weights[keep]))


def summarize_candidates(metrics_df: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics_df.groupby("candidate_id", dropna=False)
    rows = []
    metric_columns = [
        "degree_ladder_nll",
        "degree_ladder_brier",
        "degree_ladder_rps",
        "event_bin_nll",
        "event_bin_brier",
        "final_tmax_q50_mae_f",
        "final_tmax_q50_rmse_f",
        "q05_q95_coverage",
        *[f"{quantile_tag(quantile)}_pinball_loss" for quantile in DEFAULT_QUANTILES],
    ]
    for candidate_id, group in grouped:
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "fold_count": int(len(group)),
            "validation_row_count": int(group["validation_row_count"].sum()),
        }
        for column in metric_columns:
            if column not in group.columns:
                continue
            row[f"weighted_mean_{column}"] = weighted_mean(group, column)
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"std_{column}"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    candidates = load_candidates(args.candidate_config_path)
    splits = load_splits(args.splits_path)
    df = pd.read_parquet(args.features_path)
    df = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    feature_columns = select_feature_columns(df)
    leakage = leakage_findings(feature_columns)
    if leakage:
        raise ValueError(f"leakage-prone feature columns selected: {leakage[:10]}")

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        for train_end, valid_start, valid_end in splits:
            metrics, predictions = evaluate_candidate_split(
                df,
                feature_columns,
                candidate,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
            fold_rows.append(metrics)
            prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(fold_rows)
    summary_df = summarize_candidates(metrics_df)
    selected_candidate_id = str(summary_df.iloc[0]["candidate_id"])
    selected_candidate = next(candidate for candidate in candidates if str(candidate["candidate_id"]) == selected_candidate_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = args.output_dir / "rolling_origin_model_selection_fold_metrics.csv"
    summary_path = args.output_dir / "rolling_origin_model_selection_summary.csv"
    predictions_path = args.output_dir / "rolling_origin_model_selection_predictions.parquet"
    stability_path = args.output_dir / "rolling_origin_model_selection_stability.csv"
    manifest_path = args.output_dir / "rolling_origin_model_selection_manifest.json"

    metrics_df.to_csv(fold_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(predictions_path, index=False)
    stability_df = summary_df.filter(regex=r"^(candidate_id|std_)")
    stability_df.to_csv(stability_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "output_dir": str(args.output_dir),
        "fold_metrics_path": str(fold_metrics_path),
        "summary_path": str(summary_path),
        "predictions_path": str(predictions_path),
        "stability_path": str(stability_path),
        "selected_candidate_id": selected_candidate_id,
        "default_model_candidate_id": DEFAULT_MODEL_CANDIDATE_ID,
        "selected_by": ["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"],
        "selected_candidate": selected_candidate,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "split_count": len(splits),
        "splits": [{"train_end": train_end, "valid_start": valid_start, "valid_end": valid_end} for train_end, valid_start, valid_end in splits],
        "feature_count": len(feature_columns),
        "leakage_check": {"status": "ok", "finding_count": 0},
        "representative_event_bin_labels": list(DEFAULT_REPRESENTATIVE_EVENT_BINS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(fold_metrics_path)
    print(summary_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
