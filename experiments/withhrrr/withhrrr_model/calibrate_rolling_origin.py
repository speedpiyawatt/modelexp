from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections.abc import Callable

import numpy as np
import pandas as pd

from .calibrate_quantiles import apply_offsets, coverage_rows, fit_offsets
from .evaluate import build_degree_ladder_diagnostics, build_event_bin_diagnostics, mae, rmse
from .model_config import DEFAULT_MODEL_CANDIDATE_ID
from .source_disagreement import source_disagreement_regime_series
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet")
DEFAULT_MODEL_SELECTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_manifest.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/calibration_selection")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit rolling-origin calibration on earlier holdouts and test on later holdouts.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="auto")
    parser.add_argument("--calibration-valid-end", default="2024-12-31")
    parser.add_argument("--test-valid-start", default="2025-01-01")
    parser.add_argument("--min-segment-rows", type=int, default=60)
    return parser.parse_args()


def resolve_candidate_id(candidate_id: str) -> str:
    if candidate_id != "auto":
        return candidate_id
    if DEFAULT_MODEL_SELECTION_MANIFEST_PATH.exists():
        payload = json.loads(DEFAULT_MODEL_SELECTION_MANIFEST_PATH.read_text())
        selected = payload.get("selected_candidate_id")
        if isinstance(selected, str) and selected:
            return selected
    return DEFAULT_MODEL_CANDIDATE_ID


def interval_score(df: pd.DataFrame, *, prefix: str, lower_tag: str, upper_tag: str, alpha: float) -> float:
    y = pd.to_numeric(df["final_tmax_f"], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(df[f"{prefix}_tmax_{lower_tag}_f"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(df[f"{prefix}_tmax_{upper_tag}_f"], errors="coerce").to_numpy(float)
    width = upper - lower
    lower_penalty = (2.0 / alpha) * np.maximum(lower - y, 0.0)
    upper_penalty = (2.0 / alpha) * np.maximum(y - upper, 0.0)
    return float(np.mean(width + lower_penalty + upper_penalty))


def date_month(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["target_date_local"], errors="coerce").dt.month


def season_segments(df: pd.DataFrame) -> pd.Series:
    month = date_month(df)
    return pd.Series(np.where(month.isin([5, 6, 7, 8, 9]), "warm", "cool"), index=df.index, dtype="string")


def disagreement_segments(df: pd.DataFrame) -> pd.Series:
    disagreement = pd.to_numeric(df["nbm_minus_lamp_tmax_f"], errors="coerce").abs()
    return disagreement_bucket(disagreement, index=df.index)


def disagreement_bucket(disagreement: pd.Series, *, index: pd.Index) -> pd.Series:
    values = np.select(
        [disagreement.isna(), disagreement < 2.0, disagreement < 5.0],
        ["unknown", "under_2f", "2_to_5f"],
        default="5f_or_more",
    )
    return pd.Series(values, index=index, dtype="string")


def month_segments(df: pd.DataFrame) -> pd.Series:
    return date_month(df).map(lambda value: f"month_{int(value):02d}" if pd.notna(value) else "month_unknown").astype("string")


def _numeric_difference(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def hrrr_lamp_disagreement_segments(df: pd.DataFrame) -> pd.Series:
    return disagreement_bucket(_numeric_difference(df, "hrrr_minus_lamp_tmax_f").abs(), index=df.index)


def hrrr_nbm_disagreement_segments(df: pd.DataFrame) -> pd.Series:
    return disagreement_bucket(_numeric_difference(df, "hrrr_minus_nbm_tmax_f").abs(), index=df.index)


def direction_segments(df: pd.DataFrame, column: str, *, hot_name: str, cold_name: str) -> pd.Series:
    difference = _numeric_difference(df, column)
    values = np.select([difference.isna(), difference >= 3.0, difference <= -3.0], ["unknown", hot_name, cold_name], default="within_3f")
    return pd.Series(values, index=df.index, dtype="string")


def hrrr_lamp_direction_segments(df: pd.DataFrame) -> pd.Series:
    return direction_segments(
        df,
        "hrrr_minus_lamp_tmax_f",
        hot_name="hrrr_hotter_than_lamp_3f",
        cold_name="hrrr_colder_than_lamp_3f",
    )


def hrrr_nbm_direction_segments(df: pd.DataFrame) -> pd.Series:
    return direction_segments(
        df,
        "hrrr_minus_nbm_tmax_f",
        hot_name="hrrr_hotter_than_nbm_3f",
        cold_name="hrrr_colder_than_nbm_3f",
    )


def source_disagreement_regime_segments(df: pd.DataFrame) -> pd.Series:
    return source_disagreement_regime_series(df)


SEGMENTERS = {
    "season": season_segments,
    "disagreement": disagreement_segments,
    "month": month_segments,
    "hrrr_lamp_disagreement": hrrr_lamp_disagreement_segments,
    "hrrr_nbm_disagreement": hrrr_nbm_disagreement_segments,
    "hrrr_lamp_direction": hrrr_lamp_direction_segments,
    "hrrr_nbm_direction": hrrr_nbm_direction_segments,
    "source_disagreement_regime": source_disagreement_regime_segments,
}


def fit_segmented_offsets(
    calibration_df: pd.DataFrame,
    *,
    segment_name: str,
    min_segment_rows: int,
) -> dict[str, object]:
    if segment_name not in SEGMENTERS:
        raise ValueError(f"unknown segment_name: {segment_name}")
    global_offsets = fit_offsets(calibration_df)
    segments = SEGMENTERS[segment_name](calibration_df)
    segment_offsets: dict[str, dict[str, float]] = {}
    segment_counts: dict[str, int] = {}
    for segment_value, group_index in calibration_df.groupby(segments, dropna=False).groups.items():
        segment_key = str(segment_value)
        group = calibration_df.loc[group_index]
        segment_counts[segment_key] = int(len(group))
        if len(group) >= min_segment_rows:
            segment_offsets[segment_key] = fit_offsets(group)
    return {
        "segment_name": segment_name,
        "min_segment_rows": int(min_segment_rows),
        "global_offsets_f": global_offsets,
        "segment_offsets_f": segment_offsets,
        "segment_counts": segment_counts,
    }


def apply_segmented_offsets(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    segment_name = str(config["segment_name"])
    global_offsets = config["global_offsets_f"]
    segment_offsets = config["segment_offsets_f"]
    if not isinstance(global_offsets, dict) or not isinstance(segment_offsets, dict):
        raise ValueError("segmented offset config must contain offset dictionaries")
    out = df.copy()
    segments = SEGMENTERS[segment_name](out)
    calibrated_columns: list[str] = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        column = f"calibrated_pred_tmax_{tag}_f"
        values = pd.to_numeric(out[f"pred_tmax_{tag}_f"], errors="coerce").to_numpy(float)
        offsets = np.array(
            [
                float(segment_offsets.get(str(segment), global_offsets).get(tag, global_offsets[tag]))
                if isinstance(segment_offsets.get(str(segment), global_offsets), dict)
                else float(global_offsets[tag])
                for segment in segments
            ],
            dtype=float,
        )
        out[column] = values + offsets
        calibrated_columns.append(column)
    out.loc[:, calibrated_columns] = np.maximum.accumulate(out[calibrated_columns].to_numpy(float), axis=1)
    return out


def fit_shrunk_segmented_offsets(
    calibration_df: pd.DataFrame,
    *,
    segment_name: str,
    min_segment_rows: int = 30,
    full_weight_rows: int = 120,
) -> dict[str, object]:
    if segment_name not in SEGMENTERS:
        raise ValueError(f"unknown segment_name: {segment_name}")
    global_offsets = fit_offsets(calibration_df)
    segments = SEGMENTERS[segment_name](calibration_df)
    segment_offsets: dict[str, dict[str, float]] = {}
    raw_segment_offsets: dict[str, dict[str, float]] = {}
    segment_counts: dict[str, int] = {}
    segment_weights: dict[str, float] = {}
    for segment_value, group_index in calibration_df.groupby(segments, dropna=False).groups.items():
        segment_key = str(segment_value)
        group = calibration_df.loc[group_index]
        count = int(len(group))
        segment_counts[segment_key] = count
        if count < min_segment_rows:
            continue
        raw_offsets = fit_offsets(group)
        weight = min(1.0, count / float(full_weight_rows))
        segment_weights[segment_key] = float(weight)
        raw_segment_offsets[segment_key] = raw_offsets
        segment_offsets[segment_key] = {
            tag: float(global_offsets[tag] + weight * (raw_offsets[tag] - global_offsets[tag]))
            for tag in global_offsets
        }
    return {
        "segment_name": segment_name,
        "min_segment_rows": int(min_segment_rows),
        "full_weight_rows": int(full_weight_rows),
        "global_offsets_f": global_offsets,
        "segment_offsets_f": segment_offsets,
        "raw_segment_offsets_f": raw_segment_offsets,
        "segment_counts": segment_counts,
        "segment_weights": segment_weights,
        "shrinkage": "global_plus_min_1_count_over_full_weight_rows",
    }


def conformal_quantile(nonconformity: np.ndarray, alpha: float) -> float:
    values = np.sort(nonconformity[np.isfinite(nonconformity)])
    if len(values) == 0:
        return 0.0
    rank = int(np.ceil((len(values) + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), len(values))
    return float(values[rank - 1])


def fit_conformal_intervals(calibration_df: pd.DataFrame) -> dict[str, object]:
    y = pd.to_numeric(calibration_df["final_tmax_f"], errors="coerce").to_numpy(float)
    median_offsets = fit_offsets(calibration_df)
    intervals = {
        ("q05", "q95"): 0.10,
        ("q10", "q90"): 0.20,
        ("q25", "q75"): 0.50,
    }
    adjustments: dict[str, float] = {}
    for (lower_tag, upper_tag), alpha in intervals.items():
        lower = pd.to_numeric(calibration_df[f"pred_tmax_{lower_tag}_f"], errors="coerce").to_numpy(float)
        upper = pd.to_numeric(calibration_df[f"pred_tmax_{upper_tag}_f"], errors="coerce").to_numpy(float)
        adjustments[f"{lower_tag}_{upper_tag}"] = conformal_quantile(np.maximum(lower - y, y - upper), alpha)
    return {"median_offsets_f": median_offsets, "interval_adjustments_f": adjustments}


def apply_conformal_intervals(df: pd.DataFrame, config: dict[str, object]) -> pd.DataFrame:
    out = df.copy()
    median_offsets = config["median_offsets_f"]
    adjustments = config["interval_adjustments_f"]
    if not isinstance(median_offsets, dict) or not isinstance(adjustments, dict):
        raise ValueError("conformal config must contain median_offsets_f and interval_adjustments_f")
    pair_adjustments = {
        "q05": -float(adjustments["q05_q95"]),
        "q95": float(adjustments["q05_q95"]),
        "q10": -float(adjustments["q10_q90"]),
        "q90": float(adjustments["q10_q90"]),
        "q25": -float(adjustments["q25_q75"]),
        "q75": float(adjustments["q25_q75"]),
        "q50": float(median_offsets["q50"]),
    }
    calibrated_columns: list[str] = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        column = f"calibrated_pred_tmax_{tag}_f"
        out[column] = pd.to_numeric(out[f"pred_tmax_{tag}_f"], errors="coerce") + pair_adjustments[tag]
        calibrated_columns.append(column)
    out.loc[:, calibrated_columns] = np.maximum.accumulate(out[calibrated_columns].to_numpy(float), axis=1)
    return out


def as_scoring_frame(df: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = df.copy()
    if prefix == "pred":
        return out
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        out[f"pred_tmax_{tag}_f"] = out[f"{prefix}_tmax_{tag}_f"]
    return out


def score_predictions(df: pd.DataFrame, *, method_id: str, prediction_set: str) -> dict[str, object]:
    scoring_df = as_scoring_frame(df, prefix=prediction_set)
    _, _, _, pit_diagnostics, degree_metrics = build_degree_ladder_diagnostics(scoring_df)
    _, _, _, event_metrics = build_event_bin_diagnostics(scoring_df)
    return {
        "method_id": method_id,
        "prediction_set": prediction_set,
        "row_count": int(len(df)),
        "degree_ladder_nll": degree_metrics["mean_negative_log_likelihood"],
        "degree_ladder_brier": degree_metrics["mean_brier_score"],
        "degree_ladder_rps": degree_metrics["mean_ranked_probability_score"],
        "degree_ladder_observed_probability": degree_metrics["mean_observed_probability"],
        "event_bin_nll": event_metrics["mean_negative_log_likelihood"],
        "event_bin_brier": event_metrics["mean_brier_score"],
        "event_bin_observed_probability": event_metrics["mean_observed_bin_probability"],
        "pit_mean": float(pit_diagnostics.loc[pit_diagnostics["slice"] == "overall", "pit_mean"].iloc[0]),
        "pit_std": float(pit_diagnostics.loc[pit_diagnostics["slice"] == "overall", "pit_std"].iloc[0]),
        "q50_mae_f": mae(scoring_df["final_tmax_f"], scoring_df["pred_tmax_q50_f"]),
        "q50_rmse_f": rmse(scoring_df["final_tmax_f"], scoring_df["pred_tmax_q50_f"]),
        "q05_q95_coverage": float(np.mean((pd.to_numeric(scoring_df["final_tmax_f"], errors="coerce").to_numpy(float) >= pd.to_numeric(scoring_df["pred_tmax_q05_f"], errors="coerce").to_numpy(float)) & (pd.to_numeric(scoring_df["final_tmax_f"], errors="coerce").to_numpy(float) <= pd.to_numeric(scoring_df["pred_tmax_q95_f"], errors="coerce").to_numpy(float)))),
        "q05_q95_interval_score": interval_score(scoring_df, prefix="pred", lower_tag="q05", upper_tag="q95", alpha=0.10),
        "q10_q90_interval_score": interval_score(scoring_df, prefix="pred", lower_tag="q10", upper_tag="q90", alpha=0.20),
        "q25_q75_interval_score": interval_score(scoring_df, prefix="pred", lower_tag="q25", upper_tag="q75", alpha=0.50),
    }


def calibration_sort_key(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["selection_penalty"] = np.where(out["method_id"].astype(str) == "uncalibrated", 1, 0)
    return out.sort_values(["event_bin_nll", "degree_ladder_nll", "selection_penalty"]).drop(columns=["selection_penalty"]).reset_index(drop=True)


def calibration_methods(calibration_df: pd.DataFrame, *, min_segment_rows: int) -> list[tuple[str, dict[str, object], Callable[..., pd.DataFrame]]]:
    global_offsets = fit_offsets(calibration_df)
    return [
        ("global_offsets", {"offsets_f": global_offsets}, lambda df: apply_offsets(df, global_offsets)),
        (
            "source_disagreement_regime_offsets",
            fit_shrunk_segmented_offsets(
                calibration_df,
                segment_name="source_disagreement_regime",
                min_segment_rows=30,
                full_weight_rows=120,
            ),
            apply_segmented_offsets,
        ),
        ("season_offsets", fit_segmented_offsets(calibration_df, segment_name="season", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("disagreement_offsets", fit_segmented_offsets(calibration_df, segment_name="disagreement", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("hrrr_lamp_disagreement_offsets", fit_segmented_offsets(calibration_df, segment_name="hrrr_lamp_disagreement", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("hrrr_nbm_disagreement_offsets", fit_segmented_offsets(calibration_df, segment_name="hrrr_nbm_disagreement", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("hrrr_lamp_direction_offsets", fit_segmented_offsets(calibration_df, segment_name="hrrr_lamp_direction", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("hrrr_nbm_direction_offsets", fit_segmented_offsets(calibration_df, segment_name="hrrr_nbm_direction", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("month_offsets", fit_segmented_offsets(calibration_df, segment_name="month", min_segment_rows=min_segment_rows), apply_segmented_offsets),
        ("conformal_intervals", fit_conformal_intervals(calibration_df), apply_conformal_intervals),
    ]


def apply_method_config(df: pd.DataFrame, method_id: str, config: dict[str, object]) -> pd.DataFrame:
    if method_id == "global_offsets":
        offsets = config.get("offsets_f")
        if not isinstance(offsets, dict):
            raise ValueError("global_offsets config must contain offsets_f")
        return apply_offsets(df, {str(key): float(value) for key, value in offsets.items()})
    if method_id.endswith("_offsets") and "segment_name" in config:
        return apply_segmented_offsets(df, config)
    if method_id == "conformal_intervals":
        return apply_conformal_intervals(df, config)
    if method_id in {"none", "uncalibrated"}:
        return df.copy()
    raise ValueError(f"unknown calibration method_id: {method_id}")


def main() -> int:
    args = parse_args()
    candidate_id = resolve_candidate_id(str(args.candidate_id))
    df = pd.read_parquet(args.predictions_path)
    if "candidate_id" in df.columns:
        df = df.loc[df["candidate_id"].astype(str) == candidate_id].copy()
    if df.empty:
        raise ValueError(f"no rolling-origin predictions for candidate_id={candidate_id}")
    calibration_df = df.loc[df["target_date_local"].astype(str) <= args.calibration_valid_end].copy()
    test_df = df.loc[df["target_date_local"].astype(str) >= args.test_valid_start].copy()
    if calibration_df.empty or test_df.empty:
        raise ValueError("rolling-origin calibration requires non-empty calibration and test slices")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    method_rows = [score_predictions(test_df, method_id="uncalibrated", prediction_set="pred")]
    coverage_frames: list[pd.DataFrame] = []
    method_manifests: dict[str, object] = {}
    for method_id, config, _applier in calibration_methods(calibration_df, min_segment_rows=args.min_segment_rows):
        calibrated_test = apply_method_config(test_df, method_id, config)
        method_rows.append(score_predictions(calibrated_test, method_id=method_id, prediction_set="calibrated_pred"))
        coverage_df = pd.DataFrame(coverage_rows(calibrated_test))
        coverage_df.insert(0, "method_id", method_id)
        coverage_frames.append(coverage_df)
        predictions_path = args.output_dir / f"rolling_origin_predictions_{method_id}_test.parquet"
        calibrated_test.to_parquet(predictions_path, index=False)
        method_manifests[method_id] = {"config": config, "predictions_path": str(predictions_path)}

    summary_df = calibration_sort_key(pd.DataFrame(method_rows))
    coverage = pd.concat(coverage_frames, ignore_index=True) if coverage_frames else pd.DataFrame()
    selected_method_id = str(summary_df.iloc[0]["method_id"])
    if selected_method_id == "uncalibrated":
        selected_method_id = "none"
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predictions_path": str(args.predictions_path),
        "candidate_id": candidate_id,
        "calibration_valid_end": args.calibration_valid_end,
        "test_valid_start": args.test_valid_start,
        "calibration_row_count": int(len(calibration_df)),
        "test_row_count": int(len(test_df)),
        "selected_method_id": selected_method_id,
        "selected_by": ["event_bin_nll", "degree_ladder_nll"],
        "methods": method_manifests,
    }
    manifest_path = args.output_dir / "rolling_origin_calibration_manifest.json"
    coverage_path = args.output_dir / "rolling_origin_calibrated_coverage.csv"
    summary_path = args.output_dir / "rolling_origin_calibration_summary.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    summary_df.to_csv(summary_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    print(manifest_path)
    print(summary_path)
    print(coverage_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
