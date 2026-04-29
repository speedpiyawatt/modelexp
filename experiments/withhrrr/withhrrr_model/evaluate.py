from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .distribution import quantiles_to_degree_ladder
from .event_bins import EventBin, load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from .source_disagreement import DISAGREEMENT_WIDENING_REGIMES, add_source_disagreement_features
from .train_quantile_models import DEFAULT_QUANTILES, pinball_loss


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet")
DEFAULT_MODELS_DIR = pathlib.Path("experiments/withhrrr/data/runtime/models")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation")
EPSILON = 1e-12
DEFAULT_REPRESENTATIVE_EVENT_BINS: tuple[str, ...] = (
    "35F or below",
    "36-37F",
    "38-39F",
    "40-41F",
    "42-43F",
    "44-45F",
    "46-47F",
    "48-49F",
    "50-51F",
    "52-53F",
    "54-55F",
    "56-57F",
    "58-59F",
    "60-61F",
    "62-63F",
    "64-65F",
    "66-67F",
    "68-69F",
    "70-71F",
    "72-73F",
    "74-75F",
    "76-77F",
    "78-79F",
    "80-81F",
    "82-83F",
    "84-85F",
    "86-87F",
    "88-89F",
    "90-91F",
    "92-93F",
    "94F or higher",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate with-HRRR residual quantile models and baselines.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--representative-event-bins-path", type=pathlib.Path, default=None)
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


def fit_linear_three_way_blend(train_df: pd.DataFrame) -> np.ndarray:
    x = np.column_stack(
        [
            np.ones(len(train_df)),
            pd.to_numeric(train_df["nbm_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(train_df["lamp_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(train_df["hrrr_tmax_open_f"], errors="coerce").to_numpy(float),
        ]
    )
    y = pd.to_numeric(train_df["final_tmax_f"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if int(keep.sum()) < 4:
        raise ValueError("not enough rows to fit linear NBM/LAMP/HRRR blend")
    coef, *_ = np.linalg.lstsq(x[keep], y[keep], rcond=None)
    return coef


def predict_linear_three_way_blend(df: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    x = np.column_stack(
        [
            np.ones(len(df)),
            pd.to_numeric(df["nbm_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(df["lamp_tmax_open_f"], errors="coerce").to_numpy(float),
            pd.to_numeric(df["hrrr_tmax_open_f"], errors="coerce").to_numpy(float),
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


def prediction_quantiles(row: pd.Series) -> dict[float, float]:
    return {quantile: float(row[f"pred_tmax_{quantile_tag(quantile)}_f"]) for quantile in DEFAULT_QUANTILES}


def ordered_ladder_bounds(prediction_df: pd.DataFrame) -> tuple[int, int]:
    values = pd.to_numeric(prediction_df["final_tmax_f"], errors="coerce").dropna().to_numpy(float)
    quantile_columns = [f"pred_tmax_{quantile_tag(quantile)}_f" for quantile in DEFAULT_QUANTILES]
    predicted = prediction_df[quantile_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float).ravel()
    combined = np.concatenate([values, predicted[np.isfinite(predicted)]])
    if len(combined) == 0:
        raise ValueError("cannot build ladder bounds without finite temperatures")
    return int(np.floor(np.nanmin(combined) - 8.0)), int(np.ceil(np.nanmax(combined) + 8.0))


def ladder_for_prediction(row: pd.Series, *, min_temp_f: int, max_temp_f: int) -> pd.DataFrame:
    return quantiles_to_degree_ladder(
        prediction_quantiles(row),
        min_temp_f=min_temp_f,
        max_temp_f=max_temp_f,
    )


def ranked_probability_score(ladder: pd.DataFrame, observed_temp_f: int) -> float:
    degrees = pd.to_numeric(ladder["temp_f"]).astype(int).to_numpy()
    probabilities = pd.to_numeric(ladder["probability"], errors="coerce").fillna(0.0).to_numpy(float)
    predicted_cdf = np.cumsum(probabilities)
    observed_cdf = (degrees >= int(observed_temp_f)).astype(float)
    if len(degrees) <= 1:
        return float(np.mean((predicted_cdf - observed_cdf) ** 2))
    return float(np.mean((predicted_cdf - observed_cdf) ** 2))


def interval_hit(row: pd.Series, lower_tag: str, upper_tag: str) -> bool:
    observed = float(row["final_tmax_f"])
    return float(row[f"pred_tmax_{lower_tag}_f"]) <= observed <= float(row[f"pred_tmax_{upper_tag}_f"])


def row_numeric(row: pd.Series, column: str) -> float:
    if column not in row.index:
        return float("nan")
    value = pd.to_numeric(row[column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def build_degree_ladder_diagnostics(prediction_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    min_temp_f, max_temp_f = ordered_ladder_bounds(prediction_df)
    rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    per_degree_rows: list[dict[str, object]] = []
    for _, row in prediction_df.iterrows():
        ladder = ladder_for_prediction(row, min_temp_f=min_temp_f, max_temp_f=max_temp_f)
        observed_temp = int(round(float(row["final_tmax_f"])))
        probabilities = pd.to_numeric(ladder["probability"], errors="coerce").fillna(0.0)
        degrees = pd.to_numeric(ladder["temp_f"]).astype(int)
        observed_probability = float(probabilities.loc[degrees == observed_temp].sum())
        modal_index = int(probabilities.to_numpy(float).argmax())
        modal_temp = int(degrees.iloc[modal_index])
        modal_probability = float(probabilities.iloc[modal_index])
        one_hot = (degrees == observed_temp).astype(float).to_numpy()
        brier = float(np.sum((probabilities.to_numpy(float) - one_hot) ** 2))
        cdf_below = float(probabilities.loc[degrees < observed_temp].sum())
        pit_mid = cdf_below + 0.5 * observed_probability
        rows.append(
            {
                "target_date_local": row["target_date_local"],
                "station_id": row["station_id"],
                "final_tmax_f": float(row["final_tmax_f"]),
                "observed_temp_f": observed_temp,
                "observed_probability": observed_probability,
                "negative_log_likelihood": float(-np.log(max(observed_probability, EPSILON))),
                "brier_score": brier,
                "ranked_probability_score": ranked_probability_score(ladder, observed_temp),
                "pit_lower": cdf_below,
                "pit_mid": pit_mid,
                "pit_upper": cdf_below + observed_probability,
                "modal_temp_f": modal_temp,
                "modal_probability": modal_probability,
                "modal_correct": modal_temp == observed_temp,
                "q05_q95_hit": interval_hit(row, "q05", "q95"),
                "q10_q90_hit": interval_hit(row, "q10", "q90"),
                "q25_q75_hit": interval_hit(row, "q25", "q75"),
                "nbm_lamp_abs_disagreement_f": abs(float(row["nbm_minus_lamp_tmax_f"])),
                "hrrr_lamp_abs_disagreement_f": abs(row_numeric(row, "hrrr_minus_lamp_tmax_f")),
                "hrrr_nbm_abs_disagreement_f": abs(row_numeric(row, "hrrr_minus_nbm_tmax_f")),
                "hrrr_minus_lamp_tmax_f": row_numeric(row, "hrrr_minus_lamp_tmax_f"),
                "hrrr_minus_nbm_tmax_f": row_numeric(row, "hrrr_minus_nbm_tmax_f"),
            }
        )
        reliability_rows.append(
            {
                "target_date_local": row["target_date_local"],
                "confidence": modal_probability,
                "correct": modal_temp == observed_temp,
            }
        )
        for degree, probability in zip(degrees, probabilities):
            per_degree_rows.append(
                {
                    "target_date_local": row["target_date_local"],
                    "temp_f": int(degree),
                    "probability": float(probability),
                    "observed": int(degree) == observed_temp,
                }
            )
    diagnostics = pd.DataFrame(rows)
    reliability_input = pd.DataFrame(reliability_rows)
    reliability_input["confidence_bucket"] = pd.cut(
        reliability_input["confidence"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    ).astype(str)
    reliability = (
        reliability_input.groupby("confidence_bucket", dropna=False)
        .agg(
            row_count=("correct", "size"),
            mean_confidence=("confidence", "mean"),
            observed_accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    per_degree_input = pd.DataFrame(per_degree_rows)
    per_degree_input["probability_bucket"] = pd.cut(
        per_degree_input["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    ).astype(str)
    per_degree_reliability = (
        per_degree_input.groupby("probability_bucket", dropna=False)
        .agg(
            row_count=("observed", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_frequency=("observed", "mean"),
        )
        .reset_index()
    )
    pit = build_pit_diagnostics(diagnostics)
    metrics = {
        "status": "ok",
        "row_count": int(len(diagnostics)),
        "ladder_min_temp_f": min_temp_f,
        "ladder_max_temp_f": max_temp_f,
        "mean_negative_log_likelihood": float(diagnostics["negative_log_likelihood"].mean()),
        "mean_brier_score": float(diagnostics["brier_score"].mean()),
        "mean_ranked_probability_score": float(diagnostics["ranked_probability_score"].mean()),
        "mean_observed_probability": float(diagnostics["observed_probability"].mean()),
        "modal_accuracy": float(diagnostics["modal_correct"].mean()),
        "q05_q95_coverage": float(diagnostics["q05_q95_hit"].mean()),
        "q10_q90_coverage": float(diagnostics["q10_q90_hit"].mean()),
        "q25_q75_coverage": float(diagnostics["q25_q75_hit"].mean()),
    }
    return diagnostics, reliability, per_degree_reliability, pit, metrics


def pit_summary(df: pd.DataFrame, *, slice_name: str) -> dict[str, object]:
    pit = pd.to_numeric(df["pit_mid"], errors="coerce").dropna()
    if pit.empty:
        return {
            "slice": slice_name,
            "row_count": 0,
            "pit_mean": np.nan,
            "pit_std": np.nan,
            "pit_below_0_1": np.nan,
            "pit_above_0_9": np.nan,
        }
    return {
        "slice": slice_name,
        "row_count": int(len(pit)),
        "pit_mean": float(pit.mean()),
        "pit_std": float(pit.std(ddof=0)),
        "pit_below_0_1": float((pit < 0.1).mean()),
        "pit_above_0_9": float((pit > 0.9).mean()),
    }


def build_pit_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    work = diagnostics.copy()
    dates = pd.to_datetime(work["target_date_local"])
    abs_disagreement = pd.to_numeric(work["nbm_lamp_abs_disagreement_f"], errors="coerce")
    hrrr_lamp_abs = pd.to_numeric(work.get("hrrr_lamp_abs_disagreement_f"), errors="coerce")
    hrrr_nbm_abs = pd.to_numeric(work.get("hrrr_nbm_abs_disagreement_f"), errors="coerce")
    hrrr_lamp_diff = pd.to_numeric(work.get("hrrr_minus_lamp_tmax_f"), errors="coerce")
    hrrr_nbm_diff = pd.to_numeric(work.get("hrrr_minus_nbm_tmax_f"), errors="coerce")
    rows = [pit_summary(work, slice_name="overall")]
    for name, mask in (
        ("warm_apr_oct", dates.dt.month.between(4, 10)),
        ("cool_nov_mar", ~dates.dt.month.between(4, 10)),
        ("nbm_lamp_abs_disagreement_lt_2f", abs_disagreement < 2.0),
        ("nbm_lamp_abs_disagreement_2_to_5f", (abs_disagreement >= 2.0) & (abs_disagreement < 5.0)),
        ("nbm_lamp_abs_disagreement_gte_5f", abs_disagreement >= 5.0),
        ("hrrr_lamp_abs_disagreement_lt_2f", hrrr_lamp_abs < 2.0),
        ("hrrr_lamp_abs_disagreement_2_to_5f", (hrrr_lamp_abs >= 2.0) & (hrrr_lamp_abs < 5.0)),
        ("hrrr_lamp_abs_disagreement_gte_5f", hrrr_lamp_abs >= 5.0),
        ("hrrr_nbm_abs_disagreement_lt_2f", hrrr_nbm_abs < 2.0),
        ("hrrr_nbm_abs_disagreement_2_to_5f", (hrrr_nbm_abs >= 2.0) & (hrrr_nbm_abs < 5.0)),
        ("hrrr_nbm_abs_disagreement_gte_5f", hrrr_nbm_abs >= 5.0),
        ("hrrr_hotter_than_lamp_3f", hrrr_lamp_diff >= 3.0),
        ("hrrr_colder_than_lamp_3f", hrrr_lamp_diff <= -3.0),
        ("hrrr_hotter_than_nbm_3f", hrrr_nbm_diff >= 3.0),
        ("hrrr_colder_than_nbm_3f", hrrr_nbm_diff <= -3.0),
    ):
        subset = work.loc[mask]
        if not subset.empty:
            rows.append(pit_summary(subset, slice_name=name))
    return pd.DataFrame(rows)


def representative_event_bins(labels: list[str] | None = None) -> list[EventBin]:
    source_labels = labels if labels is not None else list(DEFAULT_REPRESENTATIVE_EVENT_BINS)
    return [parse_event_bin(label) for label in source_labels]


def observed_event_bin_name(observed_temp_f: int, bins: list[EventBin]) -> str:
    matches = [event_bin.name for event_bin in bins if event_bin.contains(observed_temp_f)]
    if len(matches) != 1:
        raise ValueError(f"observed temperature {observed_temp_f} matched {len(matches)} representative bins")
    return matches[0]


def build_event_bin_diagnostics(prediction_df: pd.DataFrame, *, bins: list[EventBin] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    min_temp_f, max_temp_f = ordered_ladder_bounds(prediction_df)
    bins = bins if bins is not None else representative_event_bins()
    score_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    for _, row in prediction_df.iterrows():
        ladder = ladder_for_prediction(row, min_temp_f=min_temp_f, max_temp_f=max_temp_f)
        mapped = map_ladder_to_bins(ladder, bins)
        observed_temp = int(round(float(row["final_tmax_f"])))
        observed_bin = observed_event_bin_name(observed_temp, bins)
        probabilities = pd.to_numeric(mapped["probability"], errors="coerce").fillna(0.0).to_numpy(float)
        labels = mapped["bin"].astype(str).tolist()
        observed = np.array([label == observed_bin for label in labels], dtype=float)
        observed_probability = float(probabilities[observed.astype(bool)].sum())
        score_rows.append(
            {
                "target_date_local": row["target_date_local"],
                "station_id": row["station_id"],
                "final_tmax_f": float(row["final_tmax_f"]),
                "observed_bin": observed_bin,
                "observed_bin_probability": observed_probability,
                "negative_log_likelihood": float(-np.log(max(observed_probability, EPSILON))),
                "brier_score": float(np.sum((probabilities - observed) ** 2)),
            }
        )
        for label, probability, is_observed in zip(labels, probabilities, observed):
            probability_rows.append(
                {
                    "target_date_local": row["target_date_local"],
                    "bin": label,
                    "probability": float(probability),
                    "observed": bool(is_observed),
                }
            )
    scores = pd.DataFrame(score_rows)
    probabilities_df = pd.DataFrame(probability_rows)
    bin_metrics = (
        probabilities_df.assign(squared_error=(probabilities_df["probability"] - probabilities_df["observed"].astype(float)) ** 2)
        .groupby("bin", dropna=False)
        .agg(
            row_count=("observed", "size"),
            observed_frequency=("observed", "mean"),
            mean_predicted_probability=("probability", "mean"),
            brier_score=("squared_error", "mean"),
        )
        .reset_index()
    )
    reliability_input = probabilities_df.copy()
    reliability_input["probability_bucket"] = pd.cut(
        reliability_input["probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    ).astype(str)
    reliability = (
        reliability_input.groupby("probability_bucket", dropna=False)
        .agg(
            row_count=("observed", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_frequency=("observed", "mean"),
        )
        .reset_index()
    )
    metrics = {
        "status": "ok",
        "row_count": int(len(scores)),
        "bin_count": int(len(bins)),
        "bins": [event_bin.name for event_bin in bins],
        "mean_negative_log_likelihood": float(scores["negative_log_likelihood"].mean()),
        "mean_brier_score": float(scores["brier_score"].mean()),
        "mean_observed_bin_probability": float(scores["observed_bin_probability"].mean()),
    }
    return scores, bin_metrics, reliability, metrics


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

    diagnostic_columns = [
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
        "source_disagreement_regime",
    ]
    prediction_df = valid_df[[column for column in diagnostic_columns if column in valid_df.columns]].copy()
    prediction_df = add_source_disagreement_features(prediction_df)
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
    prediction_df["baseline_hrrr_tmax_f"] = prediction_df["hrrr_tmax_open_f"]
    prediction_df["baseline_anchor_tmax_f"] = prediction_df["anchor_tmax_f"]
    prediction_df["baseline_equal_3way_anchor_tmax_f"] = prediction_df["anchor_equal_3way_tmax_f"]
    linear_coef = fit_linear_blend(train_df)
    prediction_df["baseline_linear_blend_tmax_f"] = predict_linear_blend(valid_df, linear_coef)
    linear_three_way_coef = fit_linear_three_way_blend(train_df)
    prediction_df["baseline_linear_3way_blend_tmax_f"] = predict_linear_three_way_blend(valid_df, linear_three_way_coef)
    prediction_df["baseline_direct_lgbm_tmax_f"] = train_direct_absolute_lgbm(
        train_x,
        pd.to_numeric(train_df["final_tmax_f"], errors="coerce"),
        valid_x,
    )

    baseline_rows = []
    for name, column in (
        ("nbm_only", "baseline_nbm_tmax_f"),
        ("lamp_only", "baseline_lamp_tmax_f"),
        ("hrrr_only", "baseline_hrrr_tmax_f"),
        ("fixed_50_50_anchor", "baseline_anchor_tmax_f"),
        ("fixed_equal_3way_anchor", "baseline_equal_3way_anchor_tmax_f"),
        ("linear_nbm_lamp_blend", "baseline_linear_blend_tmax_f"),
        ("linear_nbm_lamp_hrrr_blend", "baseline_linear_3way_blend_tmax_f"),
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

    hrrr_lamp_abs_disagreement = pd.to_numeric(prediction_df["hrrr_minus_lamp_tmax_f"], errors="coerce").abs()
    hrrr_nbm_abs_disagreement = pd.to_numeric(prediction_df["hrrr_minus_nbm_tmax_f"], errors="coerce").abs()
    hrrr_lamp_disagreement_rows = []
    hrrr_nbm_disagreement_rows = []
    for rows, disagreement in (
        (hrrr_lamp_disagreement_rows, hrrr_lamp_abs_disagreement),
        (hrrr_nbm_disagreement_rows, hrrr_nbm_abs_disagreement),
    ):
        for name, mask in (
            ("lt_2f", disagreement < 2.0),
            ("2_to_5f", (disagreement >= 2.0) & (disagreement < 5.0)),
            ("gte_5f", disagreement >= 5.0),
        ):
            subset = prediction_df.loc[mask]
            if not subset.empty:
                rows.append({"disagreement_bucket": name, "row_count": int(len(subset)), **summarize_point_metrics(subset, "pred_tmax_q50_f")})

    hrrr_lamp_direction_rows = []
    hrrr_nbm_direction_rows = []
    direction_specs = (
        (
            hrrr_lamp_direction_rows,
            pd.to_numeric(prediction_df["hrrr_minus_lamp_tmax_f"], errors="coerce"),
            (
                ("hrrr_hotter_than_lamp_3f", lambda diff: diff >= 3.0),
                ("hrrr_colder_than_lamp_3f", lambda diff: diff <= -3.0),
                ("hrrr_lamp_within_3f", lambda diff: diff.abs() < 3.0),
            ),
        ),
        (
            hrrr_nbm_direction_rows,
            pd.to_numeric(prediction_df["hrrr_minus_nbm_tmax_f"], errors="coerce"),
            (
                ("hrrr_hotter_than_nbm_3f", lambda diff: diff >= 3.0),
                ("hrrr_colder_than_nbm_3f", lambda diff: diff <= -3.0),
                ("hrrr_nbm_within_3f", lambda diff: diff.abs() < 3.0),
            ),
        ),
    )
    for rows, diff, specs in direction_specs:
        for name, mask_fn in specs:
            subset = prediction_df.loc[mask_fn(diff)]
            if not subset.empty:
                rows.append({"direction_slice": name, "row_count": int(len(subset)), **summarize_point_metrics(subset, "pred_tmax_q50_f")})

    degree_scores, modal_reliability, degree_reliability, pit_diagnostics, degree_metrics = build_degree_ladder_diagnostics(prediction_df)
    representative_bin_labels = load_event_bin_labels(args.representative_event_bins_path) if args.representative_event_bins_path is not None else list(DEFAULT_REPRESENTATIVE_EVENT_BINS)
    event_scores, event_bin_metrics, event_reliability, event_metrics = build_event_bin_diagnostics(
        prediction_df,
        bins=representative_event_bins(representative_bin_labels),
    )
    score_regimes = prediction_df[["target_date_local", "station_id", "source_disagreement_regime"]].copy()
    degree_with_regime = degree_scores.merge(score_regimes, on=["target_date_local", "station_id"], how="left")
    event_with_regime = event_scores.merge(score_regimes, on=["target_date_local", "station_id"], how="left")
    source_regime_rows = []
    source_masks = {
        "high_disagreement": degree_with_regime["source_disagreement_regime"].astype(str).isin(DISAGREEMENT_WIDENING_REGIMES),
        "native_warm_hrrr_cold": degree_with_regime["source_disagreement_regime"].astype(str) == "native_warm_hrrr_cold",
    }
    for regime in sorted(degree_with_regime["source_disagreement_regime"].dropna().astype(str).unique()):
        source_masks[f"regime:{regime}"] = degree_with_regime["source_disagreement_regime"].astype(str) == regime
    for slice_name, degree_mask in source_masks.items():
        degree_subset = degree_with_regime.loc[degree_mask]
        if degree_subset.empty:
            continue
        keys = degree_subset[["target_date_local", "station_id"]].drop_duplicates()
        event_subset = event_with_regime.merge(keys, on=["target_date_local", "station_id"], how="inner")
        pred_subset = prediction_df.merge(keys, on=["target_date_local", "station_id"], how="inner")
        source_regime_rows.append(
            {
                "slice": slice_name,
                "row_count": int(len(keys)),
                "event_bin_nll": float(event_subset["negative_log_likelihood"].mean()),
                "event_bin_brier": float(event_subset["brier_score"].mean()),
                "degree_ladder_nll": float(degree_subset["negative_log_likelihood"].mean()),
                "degree_ladder_rps": float(degree_subset["ranked_probability_score"].mean()),
                "q50_mae_f": mae(pred_subset["final_tmax_f"], pred_subset["pred_tmax_q50_f"]),
                "q50_rmse_f": rmse(pred_subset["final_tmax_f"], pred_subset["pred_tmax_q50_f"]),
                "event_bin_observed_probability": float(event_subset["observed_bin_probability"].mean()),
            }
        )

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
        "degree_ladder_mean_negative_log_likelihood": degree_metrics["mean_negative_log_likelihood"],
        "degree_ladder_mean_brier_score": degree_metrics["mean_brier_score"],
        "degree_ladder_mean_ranked_probability_score": degree_metrics["mean_ranked_probability_score"],
        "event_bin_mean_negative_log_likelihood": event_metrics["mean_negative_log_likelihood"],
        "event_bin_mean_brier_score": event_metrics["mean_brier_score"],
    }

    write_json(args.output_dir / "metrics_overall.json", metrics_overall)
    pd.DataFrame(baseline_rows).to_csv(args.output_dir / "baseline_comparison.csv", index=False)
    pd.DataFrame(pinball_rows).to_csv(args.output_dir / "pinball_loss.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(args.output_dir / "quantile_coverage.csv", index=False)
    pd.DataFrame(crossing_rows).to_csv(args.output_dir / "quantile_crossing.csv", index=False)
    pd.DataFrame(season_rows).to_csv(args.output_dir / "metrics_by_season.csv", index=False)
    pd.DataFrame(disagreement_rows).to_csv(args.output_dir / "metrics_by_nbm_lamp_disagreement.csv", index=False)
    pd.DataFrame(hrrr_lamp_disagreement_rows).to_csv(args.output_dir / "metrics_by_hrrr_lamp_disagreement.csv", index=False)
    pd.DataFrame(hrrr_nbm_disagreement_rows).to_csv(args.output_dir / "metrics_by_hrrr_nbm_disagreement.csv", index=False)
    pd.DataFrame(hrrr_lamp_direction_rows).to_csv(args.output_dir / "metrics_by_hrrr_lamp_direction.csv", index=False)
    pd.DataFrame(hrrr_nbm_direction_rows).to_csv(args.output_dir / "metrics_by_hrrr_nbm_direction.csv", index=False)
    pd.DataFrame(source_regime_rows).to_csv(args.output_dir / "metrics_by_source_disagreement_regime.csv", index=False)
    degree_scores.to_csv(args.output_dir / "degree_ladder_scores.csv", index=False)
    modal_reliability.to_csv(args.output_dir / "degree_ladder_modal_reliability.csv", index=False)
    degree_reliability.to_csv(args.output_dir / "degree_ladder_reliability.csv", index=False)
    pit_diagnostics.to_csv(args.output_dir / "pit_diagnostics.csv", index=False)
    event_scores.to_csv(args.output_dir / "event_bin_scores.csv", index=False)
    event_bin_metrics.to_csv(args.output_dir / "event_bin_metrics.csv", index=False)
    event_reliability.to_csv(args.output_dir / "event_bin_reliability.csv", index=False)
    write_json(args.output_dir / "degree_ladder_metrics.json", degree_metrics)
    write_json(args.output_dir / "event_bin_metrics.json", event_metrics)
    prediction_df.to_parquet(args.output_dir / "validation_predictions.parquet", index=False)
    evaluation_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "features_path": str(args.features_path),
        "models_dir": str(args.models_dir),
        "output_dir": str(args.output_dir),
        "validation_row_count": int(len(prediction_df)),
        "representative_event_bin_labels": representative_bin_labels,
        "linear_blend_coefficients": [float(value) for value in linear_coef],
        "linear_three_way_blend_coefficients": [float(value) for value in linear_three_way_coef],
        "outputs": [
            "metrics_overall.json",
            "baseline_comparison.csv",
            "pinball_loss.csv",
            "quantile_coverage.csv",
            "quantile_crossing.csv",
            "metrics_by_season.csv",
            "metrics_by_nbm_lamp_disagreement.csv",
            "metrics_by_hrrr_lamp_disagreement.csv",
            "metrics_by_hrrr_nbm_disagreement.csv",
            "metrics_by_hrrr_lamp_direction.csv",
            "metrics_by_hrrr_nbm_direction.csv",
            "metrics_by_source_disagreement_regime.csv",
            "degree_ladder_scores.csv",
            "degree_ladder_modal_reliability.csv",
            "degree_ladder_reliability.csv",
            "degree_ladder_metrics.json",
            "pit_diagnostics.csv",
            "event_bin_scores.csv",
            "event_bin_metrics.csv",
            "event_bin_metrics.json",
            "event_bin_reliability.csv",
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
