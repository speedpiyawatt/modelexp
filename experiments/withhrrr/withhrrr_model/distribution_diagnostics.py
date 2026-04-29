from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .calibrate_ladder import selected_quantile_calibrated_frame
from .distribution import degree_ladder_from_quantiles, normal_iqr_degree_ladder, smoothed_degree_ladder
from .evaluate import (
    EPSILON,
    observed_event_bin_name,
    ordered_ladder_bounds,
    ranked_probability_score,
    representative_event_bins,
)
from .event_bins import map_ladder_to_bins
from .model_config import DEFAULT_MODEL_CANDIDATE_ID
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet")
DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json")
DEFAULT_MODEL_SELECTION_SUMMARY_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_summary.csv")
DEFAULT_MODEL_SELECTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_manifest.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 6 with-HRRR quantile crossing and distribution-shape diagnostics.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--quantile-calibration-manifest-path", type=pathlib.Path, default=DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH)
    parser.add_argument("--model-selection-summary-path", type=pathlib.Path, default=DEFAULT_MODEL_SELECTION_SUMMARY_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="auto")
    parser.add_argument("--distribution-test-valid-start", default="2025-01-01")
    parser.add_argument("--crossing-penalty-weight", type=float, default=0.10)
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


def raw_final_quantile_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    anchor = pd.to_numeric(out["anchor_tmax_f"], errors="coerce").to_numpy(float)
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        residual_column = f"pred_residual_{tag}_f"
        raw_column = f"raw_tmax_{tag}_f"
        if residual_column in out.columns:
            out[raw_column] = anchor + pd.to_numeric(out[residual_column], errors="coerce").to_numpy(float)
        else:
            out[raw_column] = pd.to_numeric(out[f"pred_tmax_{tag}_f"], errors="coerce")
    return out


def slice_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    dates = pd.to_datetime(df["target_date_local"], errors="coerce")
    disagreement = pd.to_numeric(df["nbm_minus_lamp_tmax_f"], errors="coerce").abs()
    return [
        ("overall", pd.Series(True, index=df.index)),
        ("warm_apr_oct", dates.dt.month.between(4, 10)),
        ("cool_nov_mar", ~dates.dt.month.between(4, 10)),
        ("nbm_lamp_abs_disagreement_lt_2f", disagreement < 2.0),
        ("nbm_lamp_abs_disagreement_2_to_5f", (disagreement >= 2.0) & (disagreement < 5.0)),
        ("nbm_lamp_abs_disagreement_gte_5f", disagreement >= 5.0),
    ]


def crossing_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    work = raw_final_quantile_frame(df)
    rows: list[dict[str, object]] = []
    group_columns = ["candidate_id", "train_end", "valid_start", "valid_end"]
    for group_key, group in work.groupby(group_columns, dropna=False):
        candidate_id, train_end, valid_start, valid_end = group_key
        for slice_name, mask in slice_masks(group):
            subset = group.loc[mask]
            if subset.empty:
                continue
            any_crossing = np.zeros(len(subset), dtype=bool)
            for lower, upper in zip(DEFAULT_QUANTILES, DEFAULT_QUANTILES[1:]):
                lower_tag = quantile_tag(lower)
                upper_tag = quantile_tag(upper)
                diff = pd.to_numeric(subset[f"raw_tmax_{upper_tag}_f"], errors="coerce").to_numpy(float) - pd.to_numeric(subset[f"raw_tmax_{lower_tag}_f"], errors="coerce").to_numpy(float)
                finite = np.isfinite(diff)
                crossing = finite & (diff < 0.0)
                any_crossing |= crossing
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "train_end": train_end,
                        "valid_start": valid_start,
                        "valid_end": valid_end,
                        "slice": slice_name,
                        "quantile_pair": f"{lower_tag}_{upper_tag}",
                        "row_count": int(len(subset)),
                        "finite_width_count": int(finite.sum()),
                        "crossing_count": int(crossing.sum()),
                        "crossing_rate": float(crossing.sum() / finite.sum()) if bool(finite.any()) else np.nan,
                        "zero_width_count": int((finite & (diff == 0.0)).sum()),
                        "min_raw_width_f": float(np.nanmin(diff)) if bool(finite.any()) else np.nan,
                        "mean_raw_width_f": float(np.nanmean(diff)) if bool(finite.any()) else np.nan,
                    }
                )
            finite_any = np.zeros(len(subset), dtype=bool)
            for lower, upper in zip(DEFAULT_QUANTILES, DEFAULT_QUANTILES[1:]):
                lower_tag = quantile_tag(lower)
                upper_tag = quantile_tag(upper)
                diff = pd.to_numeric(subset[f"raw_tmax_{upper_tag}_f"], errors="coerce").to_numpy(float) - pd.to_numeric(subset[f"raw_tmax_{lower_tag}_f"], errors="coerce").to_numpy(float)
                finite_any |= np.isfinite(diff)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "train_end": train_end,
                    "valid_start": valid_start,
                    "valid_end": valid_end,
                    "slice": slice_name,
                    "quantile_pair": "any_adjacent",
                    "row_count": int(len(subset)),
                    "finite_width_count": int(finite_any.sum()),
                    "crossing_count": int(any_crossing.sum()),
                    "crossing_rate": float(any_crossing.sum() / finite_any.sum()) if bool(finite_any.any()) else np.nan,
                    "zero_width_count": np.nan,
                    "min_raw_width_f": np.nan,
                    "mean_raw_width_f": np.nan,
                }
            )
    return pd.DataFrame(rows)


def prediction_quantiles(row: pd.Series) -> dict[float, float]:
    return {float(quantile): float(row[f"pred_tmax_{quantile_tag(quantile)}_f"]) for quantile in DEFAULT_QUANTILES}


def smooth_ladder(ladder: pd.DataFrame) -> pd.DataFrame:
    return smoothed_degree_ladder(ladder)


def normal_iqr_ladder(row: pd.Series, *, min_temp_f: int, max_temp_f: int) -> pd.DataFrame:
    quantiles = {
        float(quantile): float(row[f"pred_tmax_{quantile_tag(quantile)}_f"])
        for quantile in DEFAULT_QUANTILES
        if f"pred_tmax_{quantile_tag(quantile)}_f" in row.index
    }
    return normal_iqr_degree_ladder(quantiles, min_temp_f=min_temp_f, max_temp_f=max_temp_f)


def ladder_for_method(row: pd.Series, *, method_id: str, min_temp_f: int, max_temp_f: int) -> pd.DataFrame:
    return degree_ladder_from_quantiles(prediction_quantiles(row), method_id=method_id, min_temp_f=min_temp_f, max_temp_f=max_temp_f)


def score_distribution_method(prediction_df: pd.DataFrame, *, method_id: str) -> dict[str, object]:
    min_temp_f, max_temp_f = ordered_ladder_bounds(prediction_df)
    degree_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    event_bins = representative_event_bins()
    for _, row in prediction_df.iterrows():
        ladder = ladder_for_method(row, method_id=method_id, min_temp_f=min_temp_f, max_temp_f=max_temp_f)
        observed_temp = int(round(float(row["final_tmax_f"])))
        probabilities = pd.to_numeric(ladder["probability"], errors="coerce").fillna(0.0).to_numpy(float)
        degrees = pd.to_numeric(ladder["temp_f"]).astype(int).to_numpy()
        observed = (degrees == observed_temp).astype(float)
        observed_probability = float(probabilities[degrees == observed_temp].sum())
        degree_rows.append(
            {
                "observed_probability": observed_probability,
                "negative_log_likelihood": float(-np.log(max(observed_probability, EPSILON))),
                "brier_score": float(np.sum((probabilities - observed) ** 2)),
                "ranked_probability_score": ranked_probability_score(ladder, observed_temp),
            }
        )
        mapped = map_ladder_to_bins(ladder, event_bins)
        labels = mapped["bin"].astype(str).tolist()
        mapped_probabilities = pd.to_numeric(mapped["probability"], errors="coerce").fillna(0.0).to_numpy(float)
        observed_bin = observed_event_bin_name(observed_temp, event_bins)
        event_observed = np.array([label == observed_bin for label in labels], dtype=float)
        observed_bin_probability = float(mapped_probabilities[event_observed.astype(bool)].sum())
        event_rows.append(
            {
                "observed_bin_probability": observed_bin_probability,
                "negative_log_likelihood": float(-np.log(max(observed_bin_probability, EPSILON))),
                "brier_score": float(np.sum((mapped_probabilities - event_observed) ** 2)),
            }
        )
    degree_df = pd.DataFrame(degree_rows)
    event_df = pd.DataFrame(event_rows)
    return {
        "method_id": method_id,
        "row_count": int(len(prediction_df)),
        "degree_ladder_nll": float(degree_df["negative_log_likelihood"].mean()),
        "degree_ladder_brier": float(degree_df["brier_score"].mean()),
        "degree_ladder_rps": float(degree_df["ranked_probability_score"].mean()),
        "degree_ladder_observed_probability": float(degree_df["observed_probability"].mean()),
        "event_bin_nll": float(event_df["negative_log_likelihood"].mean()),
        "event_bin_brier": float(event_df["brier_score"].mean()),
        "event_bin_observed_probability": float(event_df["observed_bin_probability"].mean()),
    }


def crossing_candidate_summary(crossing: pd.DataFrame, model_selection_summary_path: pathlib.Path, *, penalty_weight: float) -> pd.DataFrame:
    overall = crossing.loc[(crossing["slice"] == "overall") & (crossing["quantile_pair"] == "any_adjacent")].copy()
    grouped = (
        overall.groupby("candidate_id", dropna=False)
        .agg(row_count=("row_count", "sum"), crossing_count=("crossing_count", "sum"))
        .reset_index()
    )
    grouped["crossing_rate"] = grouped["crossing_count"] / grouped["row_count"]
    if model_selection_summary_path.exists():
        summary = pd.read_csv(model_selection_summary_path)
        keep_columns = [
            column
            for column in (
                "candidate_id",
                "weighted_mean_event_bin_nll",
                "weighted_mean_degree_ladder_nll",
                "weighted_mean_final_tmax_q50_mae_f",
            )
            if column in summary.columns
        ]
        grouped = grouped.merge(summary[keep_columns], on="candidate_id", how="left")
        if "weighted_mean_event_bin_nll" in grouped.columns:
            grouped["crossing_penalized_event_bin_nll"] = grouped["weighted_mean_event_bin_nll"] + float(penalty_weight) * grouped["crossing_rate"]
            grouped = grouped.sort_values(["crossing_penalized_event_bin_nll", "weighted_mean_degree_ladder_nll"], na_position="last")
        else:
            grouped = grouped.sort_values("crossing_rate")
    else:
        grouped = grouped.sort_values("crossing_rate")
    return grouped.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    candidate_id = resolve_candidate_id(str(args.candidate_id))
    df = pd.read_parquet(args.predictions_path)
    if df.empty:
        raise ValueError("distribution diagnostics require non-empty rolling-origin predictions")
    crossing = crossing_diagnostics(df)
    candidate_crossing = crossing_candidate_summary(crossing, args.model_selection_summary_path, penalty_weight=args.crossing_penalty_weight)
    selected_df = df.loc[df["candidate_id"].astype(str) == candidate_id].copy()
    if selected_df.empty:
        raise ValueError(f"no predictions for candidate_id={candidate_id}")
    calibration_manifest = json.loads(args.quantile_calibration_manifest_path.read_text())
    distribution_test_df = selected_df.loc[selected_df["target_date_local"].astype(str) >= args.distribution_test_valid_start].copy()
    if distribution_test_df.empty:
        raise ValueError("distribution method comparison requires a non-empty test slice")
    calibrated_df = selected_quantile_calibrated_frame(distribution_test_df, calibration_manifest)
    methods = ["interpolation_tail", "interpolation_no_tail", "smoothed_interpolation_tail", "normal_iqr"]
    distribution_summary = pd.DataFrame([score_distribution_method(calibrated_df, method_id=method_id) for method_id in methods]).sort_values(["event_bin_nll", "degree_ladder_nll"]).reset_index(drop=True)
    selected_method_id = str(distribution_summary.iloc[0]["method_id"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crossing_path = args.output_dir / "quantile_crossing_diagnostics.csv"
    candidate_crossing_path = args.output_dir / "quantile_crossing_candidate_summary.csv"
    distribution_path = args.output_dir / "distribution_method_comparison.csv"
    manifest_path = args.output_dir / "distribution_diagnostics_manifest.json"
    crossing.to_csv(crossing_path, index=False)
    candidate_crossing.to_csv(candidate_crossing_path, index=False)
    distribution_summary.to_csv(distribution_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predictions_path": str(args.predictions_path),
        "quantile_calibration_manifest_path": str(args.quantile_calibration_manifest_path),
        "candidate_id": candidate_id,
        "crossing_diagnostics_path": str(crossing_path),
        "crossing_candidate_summary_path": str(candidate_crossing_path),
        "crossing_penalty_weight": float(args.crossing_penalty_weight),
        "distribution_method_comparison_path": str(distribution_path),
        "distribution_test_valid_start": args.distribution_test_valid_start,
        "distribution_test_row_count": int(len(calibrated_df)),
        "selected_distribution_method_id": selected_method_id,
        "selected_by": ["event_bin_nll", "degree_ladder_nll"],
        "monotone_safety_guard": "raw quantile crossing is diagnosed here; prediction/evaluation outputs use monotone rearrangement via cumulative maximum before ladder construction.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(crossing_path)
    print(distribution_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
