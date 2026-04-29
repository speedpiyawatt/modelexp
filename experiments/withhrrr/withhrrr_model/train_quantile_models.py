from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .model_config import DEFAULT_MODEL_CANDIDATE_ID, DEFAULT_QUANTILES, RANDOM_SEED, candidate_by_id, selected_model_candidate
from .source_trust import apply_anchor_and_residual, fit_ridge_4way_anchor, source_trust_feature_subset, training_weights


DEFAULT_INPUT_PATH = pathlib.Path(
    "experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet"
)
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/models")
DEFAULT_MODEL_SELECTION_MANIFEST_PATH = pathlib.Path(
    "experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_manifest.json"
)
DEFAULT_VALIDATION_FRACTION = 0.20

IDENTITY_COLUMNS = {"target_date_local", "station_id", "selection_cutoff_local"}
NON_FEATURE_COLUMNS = {
    *IDENTITY_COLUMNS,
    "final_tmax_f",
    "final_tmin_f",
    "label_final_tmax_f",
    "label_final_tmin_f",
    "target_residual_f",
    "model_training_eligible",
    "model_prediction_available",
    "warmest_source",
    "coldest_source",
    "source_disagreement_regime",
}
NON_FEATURE_SUBSTRINGS = (
    "market",
    "_time_utc_code",
    "_time_local_code",
    "_source_model_code",
    "_source_product_code",
    "_source_version_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train with-HRRR residual LightGBM quantile models.")
    parser.add_argument("--input-path", type=pathlib.Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-selection-manifest-path", type=pathlib.Path, default=DEFAULT_MODEL_SELECTION_MANIFEST_PATH)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--min-train-rows", type=int, default=100)
    return parser.parse_args()


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def _has_converted_counterpart(column: str, columns: set[str]) -> bool:
    if column.startswith(("nbm_", "hrrr_")) and column.endswith("_k") and f"{column[:-2]}_f" in columns:
        return True
    if column.startswith(("nbm_", "hrrr_")) and column.endswith("_ms") and f"{column[:-3]}_mph" in columns:
        return True
    if column.startswith("hrrr_") and column.endswith("_kg_m2") and f"{column[:-6]}_in" in columns:
        return True
    if column.startswith("lamp_") and "_kt_" in column and column.replace("_kt_", "_mph_") in columns:
        return True
    return False


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    columns = set(df.columns)
    feature_columns: list[str] = []
    for column in df.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if column.startswith("label_"):
            continue
        if any(token in column for token in NON_FEATURE_SUBSTRINGS):
            continue
        if _has_converted_counterpart(column, columns):
            continue
        dtype = df[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or pd.api.types.is_numeric_dtype(dtype):
            feature_columns.append(column)
    return feature_columns


def prepare_model_frame(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    eligible = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    if eligible.empty:
        raise ValueError("no model-eligible rows found")
    y = pd.to_numeric(eligible["target_residual_f"], errors="coerce")
    valid_target = y.notna()
    eligible = eligible.loc[valid_target].copy()
    y = y.loc[valid_target]
    x = eligible.loc[:, feature_columns].copy()
    for column in x.columns:
        if pd.api.types.is_bool_dtype(x[column].dtype):
            x[column] = x[column].astype("int8")
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x, y


def train_validation_split(
    df: pd.DataFrame,
    x: pd.DataFrame,
    y: pd.Series,
    *,
    validation_fraction: float,
    min_train_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str, str]:
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be greater than 0 and less than 0.5")
    ordered = df.loc[y.index].sort_values("target_date_local")
    split_index = int(np.floor(len(ordered) * (1.0 - validation_fraction)))
    if split_index < min_train_rows:
        raise ValueError(f"not enough training rows before validation split: {split_index} < {min_train_rows}")
    if split_index >= len(ordered):
        raise ValueError("validation split produced zero validation rows")
    train_index = ordered.index[:split_index]
    valid_index = ordered.index[split_index:]
    return (
        x.loc[train_index],
        x.loc[valid_index],
        y.loc[train_index],
        y.loc[valid_index],
        str(ordered.loc[train_index, "target_date_local"].max()),
        str(ordered.loc[valid_index, "target_date_local"].min()),
    )


def train_one_quantile(
    *,
    selected_candidate: dict[str, object],
    quantile: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    train_weight: pd.Series | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    candidate_params = selected_candidate["params"]
    if not isinstance(candidate_params, dict):
        raise ValueError(f"selected candidate {DEFAULT_MODEL_CANDIDATE_ID} params must be a dict")
    train_data = lgb.Dataset(x_train, label=y_train, weight=train_weight, free_raw_data=False)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data, free_raw_data=False)
    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }
    params.update(candidate_params)
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=int(selected_candidate.get("num_boost_round", 250)),
        valid_sets=[valid_data],
        valid_names=["validation"],
    )
    train_pred = booster.predict(x_train)
    valid_pred = booster.predict(x_valid)
    metrics = {
        "best_iteration": int(booster.current_iteration()),
        "train_pinball_loss": pinball_loss(y_train.to_numpy(), train_pred, quantile),
        "validation_pinball_loss": pinball_loss(y_valid.to_numpy(), valid_pred, quantile),
    }
    return booster, metrics


def train_meta_residual_model(
    *,
    x_train: pd.DataFrame,
    y_train_final: pd.Series,
    base_q50_final_train: np.ndarray,
    base_q50_residual_train: np.ndarray,
    train_weight: pd.Series | None = None,
) -> tuple[lgb.Booster, dict[str, float]]:
    meta_x = x_train.copy()
    meta_x["base_pred_q50_f"] = base_q50_final_train
    meta_x["base_residual_q50_f"] = base_q50_residual_train
    meta_y = pd.to_numeric(y_train_final, errors="coerce").to_numpy(float) - base_q50_final_train
    train_data = lgb.Dataset(meta_x, label=meta_y, weight=train_weight, free_raw_data=False)
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
    }
    booster = lgb.train(params, train_data, num_boost_round=180)
    pred = booster.predict(meta_x)
    return booster, {
        "train_correction_mae_f": float(np.mean(np.abs(meta_y - pred))),
        "train_correction_rmse_f": float(np.sqrt(np.mean((meta_y - pred) ** 2))),
    }


def feature_prefix_counts(feature_columns: list[str]) -> dict[str, int]:
    prefixes = ("wu_", "nbm_", "lamp_", "hrrr_", "meta_")
    return {prefix: sum(1 for column in feature_columns if column.startswith(prefix)) for prefix in prefixes}


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def selected_training_spec(manifest_path: pathlib.Path) -> dict[str, object]:
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
        spec = payload.get("selected_candidate_spec")
        if isinstance(spec, dict):
            return {
                "model_candidate_id": str(spec.get("model_candidate_id", DEFAULT_MODEL_CANDIDATE_ID)),
                "anchor_policy": str(spec.get("anchor_policy", "equal_3way")),
                "feature_profile": str(spec.get("feature_profile", "global_all_features")),
                "weight_profile": str(spec.get("weight_profile", "unweighted")),
                "meta_residual": bool(spec.get("meta_residual", False)),
                "source_manifest_path": str(manifest_path),
            }
    selected_candidate = selected_model_candidate()
    return {
        "model_candidate_id": str(selected_candidate["candidate_id"]),
        "anchor_policy": "equal_3way",
        "feature_profile": "global_all_features",
        "weight_profile": "unweighted",
        "meta_residual": False,
        "source_manifest_path": None,
    }


def main() -> int:
    args = parse_args()
    training_spec = selected_training_spec(args.model_selection_manifest_path)
    selected_candidate = candidate_by_id(str(training_spec["model_candidate_id"]))
    selected_params = selected_candidate["params"]
    if not isinstance(selected_params, dict):
        raise ValueError(f"selected candidate {training_spec['model_candidate_id']} params must be a dict")
    df = pd.read_parquet(args.input_path)
    ridge_metadata = fit_ridge_4way_anchor(df) if training_spec["anchor_policy"] == "ridge_4way_anchor" else None
    df = apply_anchor_and_residual(df, anchor_policy=str(training_spec["anchor_policy"]), ridge_metadata=ridge_metadata)
    feature_columns = source_trust_feature_subset(select_feature_columns(df), str(training_spec["feature_profile"]))
    if not any(column.startswith("hrrr_") for column in feature_columns):
        raise ValueError("feature selection did not include any HRRR columns")
    x, y = prepare_model_frame(df, feature_columns)
    x_train, x_valid, y_train, y_valid, train_end_date, validation_start_date = train_validation_split(
        df,
        x,
        y,
        validation_fraction=args.validation_fraction,
        min_train_rows=args.min_train_rows,
    )
    train_weight = training_weights(df.loc[x_train.index], str(training_spec["weight_profile"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_metrics: dict[str, dict[str, float]] = {}
    model_paths: dict[str, str] = {}
    train_predictions: dict[str, np.ndarray] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster, metrics = train_one_quantile(
            selected_candidate=selected_candidate,
            quantile=quantile,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            train_weight=train_weight,
        )
        train_predictions[tag] = booster.predict(x_train)
        model_path = args.output_dir / f"residual_quantile_{tag}.txt"
        booster.save_model(model_path)
        model_metrics[tag] = metrics
        model_paths[tag] = str(model_path)
        print(f"{tag} validation_pinball_loss={metrics['validation_pinball_loss']:.6f} best_iteration={metrics['best_iteration']}")

    meta_residual_metrics = None
    if bool(training_spec["meta_residual"]):
        train_anchor = pd.to_numeric(df.loc[x_train.index, "anchor_tmax_f"], errors="coerce").to_numpy(float)
        train_final = pd.to_numeric(df.loc[x_train.index, "final_tmax_f"], errors="coerce")
        meta_booster, meta_residual_metrics = train_meta_residual_model(
            x_train=x_train,
            y_train_final=train_final,
            base_q50_final_train=train_anchor + train_predictions["q50"],
            base_q50_residual_train=train_predictions["q50"],
            train_weight=train_weight,
        )
        meta_model_path = args.output_dir / "meta_residual_correction.txt"
        meta_booster.save_model(meta_model_path)
        model_paths["meta_residual_correction"] = str(meta_model_path)

    feature_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_prefix_counts": feature_prefix_counts(feature_columns),
        "excluded_columns": sorted(column for column in df.columns if column not in feature_columns),
        "anchor_policy": training_spec["anchor_policy"],
        "feature_profile": training_spec["feature_profile"],
        "weight_profile": training_spec["weight_profile"],
        "meta_residual": training_spec["meta_residual"],
        "ridge_anchor_metadata": ridge_metadata,
        "model_selection_manifest_path": training_spec["source_manifest_path"],
        "meta_residual_model_path": model_paths.get("meta_residual_correction"),
    }
    training_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "target_column": "target_residual_f",
        "quantiles": list(DEFAULT_QUANTILES),
        "model_candidate_id": str(training_spec["model_candidate_id"]),
        "model_candidate": selected_candidate,
        "anchor_policy": training_spec["anchor_policy"],
        "feature_profile": training_spec["feature_profile"],
        "weight_profile": training_spec["weight_profile"],
        "meta_residual": training_spec["meta_residual"],
        "ridge_anchor_metadata": ridge_metadata,
        "model_selection_manifest_path": training_spec["source_manifest_path"],
        "meta_residual_metrics": meta_residual_metrics,
        "training_procedure": "fixed_num_boost_round_no_inner_early_stopping",
        "lightgbm_params": selected_params,
        "num_boost_round": int(selected_candidate.get("num_boost_round", 250)),
        "row_count_total": int(len(df)),
        "row_count_eligible": int(df["model_training_eligible"].astype("boolean").fillna(False).sum()),
        "row_count_used": int(len(x)),
        "train_row_count": int(len(x_train)),
        "validation_row_count": int(len(x_valid)),
        "train_end_date": train_end_date,
        "validation_start_date": validation_start_date,
        "random_seed": RANDOM_SEED,
        "model_paths": model_paths,
        "metrics": model_metrics,
    }
    write_json(args.output_dir / "feature_manifest.json", feature_manifest)
    write_json(args.output_dir / "training_manifest.json", training_manifest)
    print(args.output_dir / "feature_manifest.json")
    print(args.output_dir / "training_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
