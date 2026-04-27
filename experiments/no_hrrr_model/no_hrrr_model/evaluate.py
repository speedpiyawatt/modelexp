from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .train_quantile_models import DEFAULT_QUANTILES, pinball_loss


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet")
DEFAULT_MODELS_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/models")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-HRRR residual quantile models and baselines.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def quantile_tag(quantile: float) -> str:
    return f"q{int(round(quantile * 100)):02d}"


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    errors = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(errors * errors)))


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = df.loc[:, feature_columns].copy()
    for column in x.columns:
        if pd.api.types.is_bool_dtype(x[column].dtype):
            x[column] = x[column].astype("int8")
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x


def load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())


def write_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fit_linear_blend(train_df: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [
            np.ones(len(train_df)),
            pd.to_numeric(train_df["nbm_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(train_df["lamp_tmax_open_f"], errors="coerce").to_numpy(float),
        ]
    )
    y = pd.to_numeric(train_df["final_tmax_f"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if int(keep.sum()) < 3:
        raise ValueError("not enough rows to fit linear NBM/LAMP blend")
    coef, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    return coef


def predict_linear_blend(df: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    x = np.column_stack(
        [
            np.ones(len(df)),
            pd.to_numeric(df["nbm_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(df["lamp_tmax_open_f"], errors="coerce").to_numpy(float),
        ]
    )
    return x @ coef


def train_direct_absolute_lgbm(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame) -> np.ndarray:
    train_data = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.035,
        "num_leaves": 15,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "seed": 20260425,
    }
    booster = lgb.train(params, train_data, num_boost_round=250)
    return booster.predict(valid_x)


def coverage_rate(y_true: pd.Series, lower: np.ndarray, upper: np.ndarray) -> float:
    values = y_true.to_numpy(float)
    return float(np.mean((values >= lower) & (values <= upper)))


def summarize_point_metrics(df: pd.DataFrame, prediction_column: str) -> dict[str, float]:
    return {
        "mae_f": mae(df["final_tmax_f"], df[prediction_column]),
        "rmse_f": rmse(df["final_tmax_f"], df[prediction_column]),
    }


def main() -> int:
    args = parse_args()
    feature_manifest = load_json(args.models_dir / "feature_manifest.json")
    training_manifest = load_json(args.models_dir / "training_manifest.json")
    feature_columns = list(feature_manifest["feature_columns"])
    validation_start_date = str(training_manifest["validation_start_date"])

    df = pd.read_parquet(args.features_path)
    eligible = df.loc[df["model_training_eligible"].astype("boolean").fillna(False)].copy()
    train_df = eligible.loc[eligible["target_date_local"].astype(str) < validation_start_date].copy()
    valid_df = eligible.loc[eligible["target_date_local"].astype(str) >= validation_start_date].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("evaluation requires non-empty train and validation slices")

    train_x = prepare_features(train_df, feature_columns)
    valid_x = prepare_features(valid_df, feature_columns)
    valid_y_final = pd.to_numeric(valid_df["final_tmax_f"], errors="coerce")
    valid_y_residual = pd.to_numeric(valid_df["target_residual_f"], errors="coerce")

    prediction_df = valid_df[["target_date_local", "station_id", "final_tmax_f", "target_residual_f", "anchor_tmax_f", "nbm_tmax_open_f", "lamp_tmax_open_f", "nbm_minus_lamp_tmax_f"]].copy()
    pinball_rows: list[dict[str, object]] = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster = lgb.Booster(model_file=str(args.models_dir / f"residual_quantile_{tag}.txt"))
        residual_pred = booster.predict(valid_x, num_iteration=booster.best_iteration)
        final_pred = pd.to_numeric(valid_df["anchor_tmax_f"], errors="coerce").to_numpy(float) + residual_pred
        prediction_df[f"pred_residual_{tag}_f"] = residual_pred
        prediction_df[f"pred_tmax_{tag}_raw_f"] = final_pred
        prediction_df[f"pred_tmax_{tag}_f"] = final_pred
        pinball_rows.append(
            {
                "quantile": quantile,
                "tag": tag,
                "residual_pinball_loss": pinball_loss(valid_y_residual.to_numpy(float), residual_pred, quantile),
                "final_tmax_pinball_loss": pinball_loss(valid_y_final.to_numpy(float), final_pred, quantile),
            }
        )

    final_quantile_columns = [f"pred_tmax_{quantile_tag(q)}_f" for q in DEFAULT_QUANTILES]
    raw_final_quantile_columns = [f"pred_tmax_{quantile_tag(q)}_raw_f" for q in DEFAULT_QUANTILES]
    crossing_rows = []
    for lower_column, upper_column in zip(raw_final_quantile_columns, raw_final_quantile_columns[1:]):
        diff = prediction_df[upper_column] - prediction_df[lower_column]
        crossing_rows.append(
            {
                "lower": lower_column.replace("pred_tmax_", "").replace("_raw_f", ""),
                "upper": upper_column.replace("pred_tmax_", "").replace("_raw_f", ""),
                "crossing_count": int((diff < 0).sum()),
                "zero_width_count": int((diff == 0).sum()),
                "min_raw_width_f": float(diff.min()),
                "mean_raw_width_f": float(diff.mean()),
            }
        )
    rearranged = np.maximum.accumulate(prediction_df[final_quantile_columns].to_numpy(float), axis=1)
    for index, column in enumerate(final_quantile_columns):
        prediction_df[f"{column[:-2]}_rearranged_f"] = rearranged[:, index]
        prediction_df[column] = rearranged[:, index]
    prediction_df["baseline_nbm_tmax_f"] = prediction_df["nbm_tmax_open_f"]
    prediction_df["baseline_lamp_tmax_f"] = prediction_df["lamp_tmax_open_f"]
    prediction_df["baseline_anchor_tmax_f"] = prediction_df["anchor_tmax_f"]
    linear_coef = fit_linear_blend(train_df)
    prediction_df["baseline_linear_blend_tmax_f"] = predict_linear_blend(valid_df, linear_coef)
    prediction_df["baseline_direct_lgbm_tmax_f"] = train_direct_absolute_lgbm(
        train_x,
        pd.to_numeric(train_df["final_tmax_f"], errors="coerce"),
        valid_x,
    )

    baseline_rows = []
    for name, column in (
        ("nbm_only", "baseline_nbm_tmax_f"),
        ("lamp_only", "baseline_lamp_tmax_f"),
        ("fixed_50_50_anchor", "baseline_anchor_tmax_f"),
        ("linear_nbm_lamp_blend", "baseline_linear_blend_tmax_f"),
        ("direct_absolute_lgbm", "baseline_direct_lgbm_tmax_f"),
        ("residual_quantile_q50", "pred_tmax_q50_f"),
    ):
        baseline_rows.append({"model": name, **summarize_point_metrics(prediction_df, column)})

    coverage_rows = []
    for lower_tag, upper_tag in (("q05", "q95"), ("q10", "q90"), ("q25", "q75")):
        coverage_rows.append(
            {
                "interval": f"{lower_tag}_{upper_tag}",
                "coverage": coverage_rate(prediction_df["final_tmax_f"], prediction_df[f"pred_tmax_{lower_tag}_f"].to_numpy(float), prediction_df[f"pred_tmax_{upper_tag}_f"].to_numpy(float)),
                "mean_width_f": float((prediction_df[f"pred_tmax_{upper_tag}_f"] - prediction_df[f"pred_tmax_{lower_tag}_f"]).mean()),
            }
        )

    warm_mask = pd.to_datetime(prediction_df["target_date_local"]).dt.month.between(4, 10)
    season_rows = []
    for name, mask in (("warm_apr_oct", warm_mask), ("cool_nov_mar", ~warm_mask)):
        subset = prediction_df.loc[mask]
        if not subset.empty:
            season_rows.append({"season": name, "row_count": int(len(subset)), **summarize_point_metrics(subset, "pred_tmax_q50_f")})

    abs_disagreement = pd.to_numeric(prediction_df["nbm_minus_lamp_tmax_f"], errors="coerce").abs()
    disagreement_rows = []
    for name, mask in (
        ("lt_2f", abs_disagreement < 2.0),
        ("2_to_5f", (abs_disagreement >= 2.0) & (abs_disagreement < 5.0)),
        ("gte_5f", abs_disagreement >= 5.0),
    ):
        subset = prediction_df.loc[mask]
        if not subset.empty:
            disagreement_rows.append({"disagreement_bucket": name, "row_count": int(len(subset)), **summarize_point_metrics(subset, "pred_tmax_q50_f")})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_overall = {
        "status": "ok",
        "validation_row_count": int(len(prediction_df)),
        "validation_start_date": validation_start_date,
        "validation_end_date": str(prediction_df["target_date_local"].max()),
        "residual_q50_mae_f": mae(prediction_df["target_residual_f"], prediction_df["pred_residual_q50_f"]),
        "residual_q50_rmse_f": rmse(prediction_df["target_residual_f"], prediction_df["pred_residual_q50_f"]),
        "final_tmax_q50_mae_f": mae(prediction_df["final_tmax_f"], prediction_df["pred_tmax_q50_f"]),
        "final_tmax_q50_rmse_f": rmse(prediction_df["final_tmax_f"], prediction_df["pred_tmax_q50_f"]),
    }

    write_json(args.output_dir / "metrics_overall.json", metrics_overall)
    pd.DataFrame(baseline_rows).to_csv(args.output_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(pinball_rows).to_csv(args.output_dir / "pinball_loss.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "quantile_coverage.csv", index=False)
    pd.DataFrame(crossing_rows).to_csv(args.output_dir / "quantile_crossing.csv", index=False)
    pd.DataFrame(season_rows).to_csv(args.output_dir / "metrics_by_season.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(args.output_dir / "metrics_by_nbm_lamp_disagreement.csv", index=False)
    prediction_df.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    evaluation_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "models_dir": str(args.models_dir),
        "output_dir": str(args.output_dir),
        "validation_row_count": int(len(prediction_df)),
        "linear_blend_coefficients": [float(value) for value in linear_coef],
        "outputs": [
            "metrics_overall.json",
            "baseline_comparison.csv",
            "pinball_loss.csv",
            "quantile_coverage.csv",
            "quantile_crossing.csv",
            "metrics_by_season.csv",
            "metrics_by_nbm_lamp_disagreement.csv",
            "validation_predictions.parquet",
        ],
    }
    write_json(args.output_dir / "evaluation_manifest.json", evaluation_manifest)
    print(args.output_dir / "metrics_overall.json")
    print(args.output_dir / "baseline_comparison.csv")
    print(args.output_dir / "evaluation_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
