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
from .source_trust import (
    apply_anchor_and_residual,
    fit_ridge_4way_anchor,
    high_disagreement_mask,
    source_trust_feature_subset,
    training_weights,
)
from .train_quantile_models import pinball_loss, quantile_tag, select_feature_columns


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection")

DEFAULT_SPLITS: tuple[tuple[str, str, str], ...] = (
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2025-12-31"),
)
DEFAULT_ANCHOR_GRID: tuple[str, ...] = (
    "equal_3way",
    "hourly_native_lamp_hrrr",
    "source_median_4way",
    "source_trimmed_mean_4way",
    "hourly_native_lamp",
    "native_lamp",
    "current_50_50",
    "ridge_4way_anchor",
)
SOURCE_TRUST_WEIGHT_PROFILES: tuple[str, ...] = (
    "high_disagreement_weighted",
    "native_warm_hrrr_cold_specialist",
    "native_cold_hrrr_warm_specialist",
    "hrrr_outlier_specialist",
)
SOURCE_TRUST_PROMOTION_MAX_OVERALL_REGRESSION = 0.005
SOURCE_TRUST_PROMOTION_MIN_HIGH_DISAGREEMENT_IMPROVEMENT = 0.02
FORBIDDEN_EXACT_FEATURES = {
    "target_date_local",
    "station_id",
    "selection_cutoff_local",
    "final_tmax_f",
    "final_tmin_f",
    "target_residual_f",
    "model_training_eligible",
    "model_prediction_available",
}
FORBIDDEN_FEATURE_PREFIXES = ("label_",)
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "_time_utc_code",
    "_time_local_code",
    "_source_model_code",
    "_source_product_code",
    "_source_version_code",
    "market",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-origin model selection for with-HRRR residual quantile models.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-config-path", type=pathlib.Path, default=None, help="Backwards-compatible path. Model-candidate configs contain params; candidate-spec configs contain model_candidate_id/anchor/profile fields.")
    parser.add_argument("--model-candidate-config-path", type=pathlib.Path, default=None)
    parser.add_argument("--candidate-spec-config-path", type=pathlib.Path, default=None)
    parser.add_argument("--splits-path", type=pathlib.Path, default=None)
    return parser.parse_args()


def load_config_payload(path: pathlib.Path | None) -> object | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def config_items(payload: object | None) -> list[object]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        for key in ("candidate_specs", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return []
    if isinstance(payload, list):
        return payload
    return []


def looks_like_model_candidate_config(payload: object | None) -> bool:
    items = config_items(payload)
    return bool(items) and all(isinstance(item, dict) and "candidate_id" in item and "params" in item for item in items)


def looks_like_candidate_spec_config(payload: object | None) -> bool:
    items = config_items(payload)
    return bool(items) and all(
        isinstance(item, dict) and ("model_candidate_id" in item or ("candidate_id" in item and "params" not in item))
        for item in items
    )


def load_candidates(path: pathlib.Path | None) -> list[dict[str, object]]:
    payload = load_config_payload(path)
    if payload is None:
        candidates = [dict(candidate) for candidate in DEFAULT_CANDIDATES]
    elif isinstance(payload, dict):
        candidates = payload.get("candidates")
    else:
        candidates = payload
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate config must be a non-empty list or contain a non-empty candidates list")
    return validate_model_candidates(candidates)


def validate_model_candidates(candidates: list[object]) -> list[dict[str, object]]:
    validated: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or "candidate_id" not in candidate or "params" not in candidate:
            raise ValueError("each candidate must contain candidate_id and params")
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in seen_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id}")
        seen_ids.add(candidate_id)
        validated.append(dict(candidate))
    return validated


def candidate_spec_id(spec: dict[str, object]) -> str:
    tokens = [
        str(spec["model_candidate_id"]),
        f"anchor={spec['anchor_policy']}",
        f"features={spec['feature_profile']}",
        f"weights={spec['weight_profile']}",
    ]
    if bool(spec.get("meta_residual")):
        tokens.append("meta=source_trust")
    return "__".join(tokens)


def default_candidate_specs(model_candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    seen: set[str] = set()

    def add_spec(
        *,
        model_candidate_id: str,
        anchor_policy: str = "equal_3way",
        feature_profile: str = "global_all_features",
        weight_profile: str = "unweighted",
        meta_residual: bool = False,
    ) -> None:
        spec = {
            "model_candidate_id": model_candidate_id,
            "anchor_policy": anchor_policy,
            "feature_profile": feature_profile,
            "weight_profile": weight_profile,
            "meta_residual": bool(meta_residual),
        }
        spec_id = candidate_spec_id(spec)
        if spec_id in seen:
            return
        spec["candidate_id"] = spec_id
        seen.add(spec_id)
        specs.append(spec)

    model_candidate_ids = [str(candidate["candidate_id"]) for candidate in model_candidates]
    for model_candidate_id in model_candidate_ids:
        add_spec(model_candidate_id=model_candidate_id)

    if DEFAULT_MODEL_CANDIDATE_ID not in model_candidate_ids:
        return specs

    for anchor_policy in DEFAULT_ANCHOR_GRID:
        add_spec(
            model_candidate_id=DEFAULT_MODEL_CANDIDATE_ID,
            anchor_policy=anchor_policy,
            feature_profile="source_trust_all_features",
        )

    for weight_profile in SOURCE_TRUST_WEIGHT_PROFILES:
        add_spec(
            model_candidate_id=DEFAULT_MODEL_CANDIDATE_ID,
            anchor_policy="equal_3way",
            feature_profile=weight_profile,
            weight_profile=weight_profile,
        )

    add_spec(
        model_candidate_id=DEFAULT_MODEL_CANDIDATE_ID,
        anchor_policy="equal_3way",
        feature_profile="source_trust_all_features",
        weight_profile="unweighted",
        meta_residual=True,
    )
    return specs


def validate_candidate_specs(items: list[object]) -> list[dict[str, object]]:
    if not isinstance(items, list) or not items:
        raise ValueError("candidate spec config must be a non-empty list")
    specs: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each candidate spec must be a dict")
        if "model_candidate_id" not in item:
            model_candidate_id = item.get("candidate_id")
            if model_candidate_id is None:
                raise ValueError("candidate spec must include model_candidate_id or candidate_id")
            item = {
                "model_candidate_id": str(model_candidate_id),
                "anchor_policy": item.get("anchor_policy", "equal_3way"),
                "feature_profile": item.get("feature_profile", "global_all_features"),
                "weight_profile": item.get("weight_profile", "unweighted"),
                "meta_residual": bool(item.get("meta_residual", False)),
            }
        spec = dict(item)
        spec.setdefault("anchor_policy", "equal_3way")
        spec.setdefault("feature_profile", "global_all_features")
        spec.setdefault("weight_profile", "unweighted")
        spec.setdefault("meta_residual", False)
        spec["candidate_id"] = str(spec.get("candidate_id") or candidate_spec_id(spec))
        if str(spec["candidate_id"]) in seen_ids:
            raise ValueError(f"duplicate candidate spec candidate_id: {spec['candidate_id']}")
        seen_ids.add(str(spec["candidate_id"]))
        specs.append(spec)
    return specs


def load_candidate_specs(path: pathlib.Path | None, model_candidates: list[dict[str, object]]) -> list[dict[str, object]]:
    payload = load_config_payload(path)
    if payload is None:
        return default_candidate_specs(model_candidates)
    if isinstance(payload, dict):
        items = payload.get("candidate_specs") if "candidate_specs" in payload else payload.get("candidates")
    else:
        items = payload
    if not isinstance(items, list) or not items:
        raise ValueError("candidate spec config must be a non-empty list or contain candidates/candidate_specs")
    return validate_candidate_specs(items)


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


def train_quantile_booster(
    *,
    candidate: dict[str, object],
    quantile: float,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    train_weight: pd.Series | None = None,
) -> lgb.Booster:
    train_data = lgb.Dataset(train_x, label=train_y, weight=train_weight, free_raw_data=False)
    booster = lgb.train(
        lightgbm_params(candidate, quantile),
        train_data,
        num_boost_round=int(candidate.get("num_boost_round", 250)),
    )
    return booster


def train_meta_residual_correction(
    *,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    valid_x: pd.DataFrame,
    train_weight: pd.Series | None = None,
) -> np.ndarray:
    train_data = lgb.Dataset(train_x, label=train_y, weight=train_weight, free_raw_data=False)
    params = {
        "objective": "regression_l2",
        "metric": "rmse",
        "learning_rate": 0.025,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
        "lambda_l2": 8.0,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }
    booster = lgb.train(params, train_data, num_boost_round=180)
    return booster.predict(valid_x)


def evaluate_candidate_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    model_candidates_by_id: dict[str, dict[str, object]],
    candidate_spec: dict[str, object],
    *,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    train_df = df.loc[df["target_date_local"].astype(str) <= train_end].copy()
    valid_df = df.loc[(df["target_date_local"].astype(str) >= valid_start) & (df["target_date_local"].astype(str) <= valid_end)].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError(f"empty split for train_end={train_end} valid_start={valid_start} valid_end={valid_end}")

    model_candidate_id = str(candidate_spec["model_candidate_id"])
    candidate = model_candidates_by_id[model_candidate_id]
    anchor_policy = str(candidate_spec["anchor_policy"])
    feature_profile = str(candidate_spec["feature_profile"])
    weight_profile = str(candidate_spec["weight_profile"])
    candidate_id = str(candidate_spec["candidate_id"])
    ridge_metadata = fit_ridge_4way_anchor(train_df) if anchor_policy == "ridge_4way_anchor" else None
    train_df = apply_anchor_and_residual(train_df, anchor_policy=anchor_policy, ridge_metadata=ridge_metadata)
    valid_df = apply_anchor_and_residual(valid_df, anchor_policy=anchor_policy, ridge_metadata=ridge_metadata)
    selected_feature_columns = source_trust_feature_subset(feature_columns, feature_profile)
    train_x = prepare_features(train_df, selected_feature_columns)
    valid_x = prepare_features(valid_df, selected_feature_columns)
    train_y_residual = pd.to_numeric(train_df["target_residual_f"], errors="coerce")
    valid_y_residual = pd.to_numeric(valid_df["target_residual_f"], errors="coerce")
    valid_y_final = pd.to_numeric(valid_df["final_tmax_f"], errors="coerce")
    anchor = pd.to_numeric(valid_df["anchor_tmax_f"], errors="coerce").to_numpy(float)
    train_anchor = pd.to_numeric(train_df["anchor_tmax_f"], errors="coerce").to_numpy(float)
    train_weight = training_weights(train_df, weight_profile)

    prediction_columns = [
        "target_date_local",
        "station_id",
        "final_tmax_f",
        "target_residual_f",
        "anchor_tmax_f",
        "nbm_tmax_open_f",
        "lamp_tmax_open_f",
        "hrrr_tmax_open_f",
        "anchor_equal_3way_tmax_f",
        "nbm_minus_lamp_tmax_f",
        "hrrr_minus_lamp_tmax_f",
        "hrrr_minus_nbm_tmax_f",
        "abs_hrrr_minus_lamp_tmax_f",
        "abs_hrrr_minus_nbm_tmax_f",
        "hrrr_above_nbm_lamp_range_f",
        "hrrr_below_nbm_lamp_range_f",
        "hrrr_outside_nbm_lamp_range_f",
        "hrrr_hotter_than_lamp_3f",
        "hrrr_colder_than_lamp_3f",
        "hrrr_hotter_than_nbm_3f",
        "hrrr_colder_than_nbm_3f",
        "nbm_native_tmax_2m_day_max_f",
        "source_spread_f",
        "source_median_tmax_f",
        "warmest_source",
        "coldest_source",
        "native_minus_hrrr_f",
        "hrrr_minus_source_median_f",
        "native_minus_source_median_f",
        "source_trimmed_mean_tmax_f",
        "native_minus_lamp_tmax_f",
        "native_minus_nbm_tmax_f",
        "lamp_minus_hrrr_tmax_f",
        "abs_native_minus_lamp_tmax_f",
        "abs_native_minus_nbm_tmax_f",
        "abs_native_minus_hrrr_f",
        "abs_lamp_minus_hrrr_tmax_f",
        "source_rank_nbm",
        "source_rank_native",
        "source_rank_lamp",
        "source_rank_hrrr",
        "source_disagreement_regime",
    ]
    predictions = valid_df[[column for column in prediction_columns if column in valid_df.columns]].copy()
    predictions["candidate_id"] = candidate_id
    predictions["model_candidate_id"] = model_candidate_id
    predictions["anchor_policy"] = anchor_policy
    predictions["feature_profile"] = feature_profile
    predictions["weight_profile"] = weight_profile
    predictions["meta_residual"] = bool(candidate_spec.get("meta_residual", False))
    predictions["train_end"] = train_end
    predictions["valid_start"] = valid_start
    predictions["valid_end"] = valid_end

    residual_predictions: dict[str, np.ndarray] = {}
    train_residual_predictions: dict[str, np.ndarray] = {}
    final_predictions: dict[str, np.ndarray] = {}
    pinball_rows: dict[str, float] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster = train_quantile_booster(
            candidate=candidate,
            quantile=quantile,
            train_x=train_x,
            train_y=train_y_residual,
            train_weight=train_weight,
        )
        residual_pred = booster.predict(valid_x)
        train_residual_predictions[tag] = booster.predict(train_x)
        residual_predictions[tag] = residual_pred
        final_predictions[tag] = anchor + residual_pred
        pinball_rows[f"{tag}_pinball_loss"] = pinball_loss(valid_y_residual.to_numpy(float), residual_pred, quantile)

    meta_correction = np.zeros(len(valid_df), dtype=float)
    if bool(candidate_spec.get("meta_residual", False)):
        meta_train_x = train_x.copy()
        meta_valid_x = valid_x.copy()
        meta_train_x["base_pred_q50_f"] = train_anchor + train_residual_predictions["q50"]
        meta_train_x["base_residual_q50_f"] = train_residual_predictions["q50"]
        meta_valid_x["base_pred_q50_f"] = anchor + residual_predictions["q50"]
        meta_valid_x["base_residual_q50_f"] = residual_predictions["q50"]
        meta_y = pd.to_numeric(train_df["final_tmax_f"], errors="coerce").to_numpy(float) - meta_train_x["base_pred_q50_f"].to_numpy(float)
        meta_correction = train_meta_residual_correction(
            train_x=meta_train_x,
            train_y=pd.Series(meta_y, index=train_df.index),
            valid_x=meta_valid_x,
            train_weight=train_weight,
        )
        for tag in list(final_predictions):
            final_predictions[tag] = final_predictions[tag] + meta_correction
        predictions["pred_meta_residual_correction_f"] = meta_correction

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
        "candidate_id": candidate_id,
        "model_candidate_id": model_candidate_id,
        "anchor_policy": anchor_policy,
        "feature_profile": feature_profile,
        "weight_profile": weight_profile,
        "meta_residual": bool(candidate_spec.get("meta_residual", False)),
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
        "fixed_equal_3way_anchor_mae_f": mae(valid_y_final, valid_df["anchor_equal_3way_tmax_f"]),
        "fixed_equal_3way_anchor_rmse_f": rmse(valid_y_final, valid_df["anchor_equal_3way_tmax_f"]),
        "hrrr_only_mae_f": mae(valid_y_final, valid_df["hrrr_tmax_open_f"]),
        "hrrr_only_rmse_f": rmse(valid_y_final, valid_df["hrrr_tmax_open_f"]),
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
        "high_disagreement_validation_row_count": int(high_disagreement_mask(valid_df).sum()),
        **pinball_rows,
    }
    if ridge_metadata is not None:
        metrics["ridge_anchor_coefficients"] = json.dumps(ridge_metadata["coefficients"])
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
        "fixed_anchor_mae_f",
        "fixed_equal_3way_anchor_mae_f",
        "hrrr_only_mae_f",
        "q05_q95_coverage",
        *[f"{quantile_tag(quantile)}_pinball_loss" for quantile in DEFAULT_QUANTILES],
    ]
    for candidate_id, group in grouped:
        first = group.iloc[0]
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "model_candidate_id": first.get("model_candidate_id"),
            "anchor_policy": first.get("anchor_policy"),
            "feature_profile": first.get("feature_profile"),
            "weight_profile": first.get("weight_profile"),
            "meta_residual": bool(first.get("meta_residual", False)),
            "fold_count": int(len(group)),
            "validation_row_count": int(group["validation_row_count"].sum()),
            "high_disagreement_validation_row_count": int(group.get("high_disagreement_validation_row_count", pd.Series(0, index=group.index)).sum()),
        }
        for column in metric_columns:
            if column not in group.columns:
                continue
            row[f"weighted_mean_{column}"] = weighted_mean(group, column)
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"std_{column}"] = float(group[column].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"]).reset_index(drop=True)


def source_slice_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if predictions_df.empty:
        return pd.DataFrame(rows)
    for candidate_id, group in predictions_df.groupby("candidate_id", dropna=False):
        regimes = group.get("source_disagreement_regime", pd.Series("unknown", index=group.index)).astype(str)
        masks: dict[str, pd.Series] = {
            "high_disagreement": high_disagreement_mask(group),
            "native_warm_hrrr_cold": regimes == "native_warm_hrrr_cold",
            "native_cold_hrrr_warm": regimes == "native_cold_hrrr_warm",
            "hrrr_hot_outlier": regimes == "hrrr_hot_outlier",
            "hrrr_cold_outlier": regimes == "hrrr_cold_outlier",
            "broad_disagreement": regimes == "broad_disagreement",
            "tight_consensus": regimes == "tight_consensus",
        }
        for regime in sorted(regimes.dropna().unique()):
            masks[f"regime:{regime}"] = regimes == regime
        for slice_name, mask in masks.items():
            subset = group.loc[mask].copy()
            if subset.empty:
                continue
            degree_scores, _, _, _, degree_metrics = build_degree_ladder_diagnostics(subset)
            event_scores, _, _, event_metrics = build_event_bin_diagnostics(subset)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "slice": slice_name,
                    "row_count": int(subset[["target_date_local", "station_id"]].drop_duplicates().shape[0]),
                    "event_bin_nll": event_metrics["mean_negative_log_likelihood"],
                    "event_bin_brier": event_metrics["mean_brier_score"],
                    "event_bin_observed_probability": event_metrics["mean_observed_bin_probability"],
                    "degree_ladder_nll": degree_metrics["mean_negative_log_likelihood"],
                    "degree_ladder_rps": degree_metrics["mean_ranked_probability_score"],
                    "degree_ladder_observed_probability": degree_metrics["mean_observed_probability"],
                    "q50_mae_f": mae(subset["final_tmax_f"], subset["pred_tmax_q50_f"]),
                    "q50_rmse_f": rmse(subset["final_tmax_f"], subset["pred_tmax_q50_f"]),
                    "zero_degree_observed_probability_count": int((degree_scores["observed_probability"] <= 0.0).sum()),
                    "zero_event_observed_probability_count": int((event_scores["observed_bin_probability"] <= 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def select_candidate(summary_df: pd.DataFrame, slice_metrics_df: pd.DataFrame) -> str:
    sorted_summary = summary_df.sort_values(["weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"]).reset_index(drop=True)
    best = sorted_summary.iloc[0]
    best_candidate_id = str(best["candidate_id"])
    best_overall_event_nll = float(best["weighted_mean_event_bin_nll"])
    high = slice_metrics_df.loc[slice_metrics_df["slice"] == "high_disagreement"].copy() if not slice_metrics_df.empty else pd.DataFrame()
    if high.empty:
        return best_candidate_id
    best_high = high.loc[high["candidate_id"].astype(str) == best_candidate_id]
    if best_high.empty:
        return best_candidate_id
    best_high_event_nll = float(best_high["event_bin_nll"].iloc[0])
    candidates = sorted_summary.loc[
        pd.to_numeric(sorted_summary["weighted_mean_event_bin_nll"], errors="coerce")
        <= best_overall_event_nll + SOURCE_TRUST_PROMOTION_MAX_OVERALL_REGRESSION
    ].copy()
    eligible_rows = []
    for _, candidate in candidates.iterrows():
        candidate_id = str(candidate["candidate_id"])
        candidate_high = high.loc[high["candidate_id"].astype(str) == candidate_id]
        if candidate_high.empty:
            continue
        high_improvement = best_high_event_nll - float(candidate_high["event_bin_nll"].iloc[0])
        if candidate_id == best_candidate_id or high_improvement >= SOURCE_TRUST_PROMOTION_MIN_HIGH_DISAGREEMENT_IMPROVEMENT:
            row = candidate.to_dict()
            row["high_disagreement_event_nll"] = float(candidate_high["event_bin_nll"].iloc[0])
            row["high_disagreement_improvement_vs_best_overall"] = high_improvement
            eligible_rows.append(row)
    if not eligible_rows:
        return best_candidate_id
    eligible = pd.DataFrame(eligible_rows).sort_values(
        ["high_disagreement_event_nll", "weighted_mean_event_bin_nll", "weighted_mean_degree_ladder_nll"]
    )
    return str(eligible.iloc[0]["candidate_id"])


def main() -> int:
    args = parse_args()
    if args.candidate_config_path is not None and (
        args.model_candidate_config_path is not None or args.candidate_spec_config_path is not None
    ):
        raise ValueError("--candidate-config-path cannot be combined with --model-candidate-config-path or --candidate-spec-config-path")

    model_candidate_config_path = args.model_candidate_config_path
    candidate_spec_config_path = args.candidate_spec_config_path
    if args.candidate_config_path is not None:
        legacy_payload = load_config_payload(args.candidate_config_path)
        if looks_like_model_candidate_config(legacy_payload):
            model_candidate_config_path = args.candidate_config_path
        elif looks_like_candidate_spec_config(legacy_payload):
            candidate_spec_config_path = args.candidate_config_path
        else:
            raise ValueError(
                "--candidate-config-path must contain either model candidates with params "
                "or candidate specs with model_candidate_id/anchor/profile fields"
            )

    model_candidates = load_candidates(model_candidate_config_path)
    candidate_specs = load_candidate_specs(candidate_spec_config_path, model_candidates)
    model_candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in model_candidates}
    for spec in candidate_specs:
        if str(spec["model_candidate_id"]) not in model_candidates_by_id:
            raise ValueError(f"unknown model_candidate_id in candidate spec: {spec['model_candidate_id']}")
    splits = load_splits(args.splits_path)
    df = pd.read_parquet(args.features_path)
    df = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    feature_columns = select_feature_columns(df)
    leakage = leakage_findings(feature_columns)
    if leakage:
        raise ValueError(f"leakage-prone feature columns selected: {leakage[:10]}")

    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for candidate_spec in candidate_specs:
        for train_end, valid_start, valid_end in splits:
            metrics, predictions = evaluate_candidate_split(
                df,
                feature_columns,
                model_candidates_by_id,
                candidate_spec,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
            fold_rows.append(metrics)
            prediction_frames.append(predictions)

    metrics_df = pd.DataFrame(fold_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    summary_df = summarize_candidates(metrics_df)
    slice_metrics_df = source_slice_summary(predictions_df)
    selected_candidate_id = select_candidate(summary_df, slice_metrics_df)
    selected_spec = next(spec for spec in candidate_specs if str(spec["candidate_id"]) == selected_candidate_id)
    selected_model_candidate = model_candidates_by_id[str(selected_spec["model_candidate_id"])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_metrics_path = args.output_dir / "rolling_origin_model_selection_fold_metrics.csv"
    summary_path = args.output_dir / "rolling_origin_model_selection_summary.csv"
    slice_metrics_path = args.output_dir / "rolling_origin_model_selection_source_slices.csv"
    predictions_path = args.output_dir / "rolling_origin_model_selection_predictions.parquet"
    stability_path = args.output_dir / "rolling_origin_model_selection_stability.csv"
    manifest_path = args.output_dir / "rolling_origin_model_selection_manifest.json"

    metrics_df.to_csv(fold_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    slice_metrics_df.to_csv(slice_metrics_path, index=False)
    predictions_df.to_parquet(predictions_path, index=False)
    stability_df = summary_df.filter(regex=r"^(candidate_id|std_)")
    stability_df.to_csv(stability_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "output_dir": str(args.output_dir),
        "fold_metrics_path": str(fold_metrics_path),
        "summary_path": str(summary_path),
        "source_slice_metrics_path": str(slice_metrics_path),
        "predictions_path": str(predictions_path),
        "stability_path": str(stability_path),
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_spec": selected_spec,
        "selected_model_candidate_id": str(selected_spec["model_candidate_id"]),
        "selected_anchor_policy": str(selected_spec["anchor_policy"]),
        "selected_feature_profile": str(selected_spec["feature_profile"]),
        "selected_weight_profile": str(selected_spec["weight_profile"]),
        "selected_meta_residual": bool(selected_spec.get("meta_residual", False)),
        "default_model_candidate_id": DEFAULT_MODEL_CANDIDATE_ID,
        "selected_by": [
            "weighted_mean_event_bin_nll",
            "weighted_mean_degree_ladder_nll",
            "or_within_0_005_overall_and_improves_high_disagreement_event_nll_by_0_02",
        ],
        "selected_model_candidate": selected_model_candidate,
        "candidate_count": len(candidate_specs),
        "candidate_specs": candidate_specs,
        "model_candidates": model_candidates,
        "split_count": len(splits),
        "splits": [{"train_end": train_end, "valid_start": valid_start, "valid_end": valid_end} for train_end, valid_start, valid_end in splits],
        "feature_count": len(feature_columns),
        "leakage_check": {"status": "ok", "finding_count": 0},
        "representative_event_bin_labels": list(DEFAULT_REPRESENTATIVE_EVENT_BINS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(fold_metrics_path)
    print(summary_path)
    print(slice_metrics_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
