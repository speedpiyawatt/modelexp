from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/models")
DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
DEFAULT_VALIDATION_FRACTION = 0.20
RANDOM_SEED = 20260425

IDENTITY_COLUMNS = {"target_date_local", "station_id", "selection_cutoff_local"}
NON_FEATURE_COLUMNS = {
    *IDENTITY_COLUMNS,
    "final_tmax_f",
    "final_tmin_f",
    "target_residual_f",
    "model_training_eligible",
}
NON_FEATURE_SUBSTRINGS = (
    "_time_utc_code",
    "_time_local_code",
    "_source_model_code",
    "_source_product_code",
    "_source_version_code",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train no-HRRR residual LightGBM quantile models.")
    parser.add_argument("--input-path", type=pathlib.Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--validation-fraction", type=float, default=DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--min-train-rows", type=int, default=100)
    return parser.parse_args()


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    errors = y_true - y_pred
    return float(np.mean(np.maximum(quantile * errors, (quantile - 1.0) * errors)))


def _has_converted_counterpart(column: str, columns: set[str]) -> bool:
    if column.startswith("nbm_") and column.endswith("_k") and f"{column[:-2]}_f" in columns:
        return True
    if column.startswith("nbm_") and column.endswith("_ms") and f"{column[:-3]}_mph" in columns:
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
    quantile: float,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> tuple[lgb.Booster, dict[str, float]]:
    train_data = lgb.Dataset(x_train, label=y_train, free_raw_data=False)
    valid_data = lgb.Dataset(x_valid, label=y_valid, reference=train_data, free_raw_data=False)
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
    booster = lgb.train(
        params,
        train_data,
        num_boost_round=1200,
        valid_sets=[valid_data],
        valid_names=["validation"],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    train_pred = booster.predict(x_train, num_iteration=booster.best_iteration)
    valid_pred = booster.predict(x_valid, num_iteration=booster.best_iteration)
    metrics = {
        "best_iteration": int(booster.best_iteration or booster.current_iteration()),
        "train_pinball_loss": pinball_loss(y_train.to_numpy(), train_pred, quantile),
        "validation_pinball_loss": pinball_loss(y_valid.to_numpy(), valid_pred, quantile),
    }
    return booster, metrics


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    feature_columns = select_feature_columns(df)
    if not feature_columns:
        raise ValueError("no numeric feature columns selected")
    x, y = prepare_model_frame(df, feature_columns)
    x_train, x_valid, y_train, y_valid, train_end_date, validation_start_date = train_validation_split(
        df,
        x,
        y,
        validation_fraction=args.validation_fraction,
        min_train_rows=args.min_train_rows,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_metrics: dict[str, dict[str, float]] = {}
    model_paths: dict[str, str] = {}
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster, metrics = train_one_quantile(
            quantile=quantile,
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
        )
        model_path = args.output_dir / f"residual_quantile_{tag}.txt"
        booster.save_model(model_path)
        model_metrics[tag] = metrics
        model_paths[tag] = str(model_path)
        print(f"{tag} validation_pinball_loss={metrics['validation_pinball_loss']:.6f} best_iteration={metrics['best_iteration']}")

    feature_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "excluded_columns": sorted(column for column in df.columns if column not in feature_columns),
    }
    training_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_path": str(args.input_path),
        "output_dir": str(args.output_dir),
        "target_column": "target_residual_f",
        "quantiles": list(DEFAULT_QUANTILES),
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
