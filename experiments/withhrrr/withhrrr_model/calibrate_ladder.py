from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .calibrate_rolling_origin import apply_method_config, as_scoring_frame
from .evaluate import EPSILON, observed_event_bin_name, ordered_ladder_bounds, ranked_probability_score, representative_event_bins
from .event_bins import map_ladder_to_bins
from .model_config import DEFAULT_MODEL_CANDIDATE_ID
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag
from .distribution import degree_ladder_from_quantiles


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet")
DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json")
DEFAULT_DISTRIBUTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics/distribution_diagnostics_manifest.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/ladder_calibration")
FALLBACK_DISTRIBUTION_METHOD_ID = "interpolation_tail"
DistributionMethod = str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit final 1F probability-ladder calibration on rolling-origin predictions.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--quantile-calibration-manifest-path", type=pathlib.Path, default=DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH)
    parser.add_argument("--distribution-manifest-path", type=pathlib.Path, default=DEFAULT_DISTRIBUTION_MANIFEST_PATH)
    parser.add_argument("--distribution-method", default="auto", choices=("auto", "interpolation_tail", "interpolation_no_tail", "smoothed_interpolation_tail", "normal_iqr"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default=DEFAULT_MODEL_CANDIDATE_ID)
    parser.add_argument("--calibration-valid-end", default="2024-12-31")
    parser.add_argument("--test-valid-start", default="2025-01-01")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--shrinkage", default="0.25,0.50,0.75,1.00")
    return parser.parse_args()


def parse_shrinkage_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("shrinkage grid must contain at least one value")
    if any(item < 0.0 or item > 1.0 for item in grid):
        raise ValueError("shrinkage values must be between 0 and 1")
    return grid


def selected_quantile_calibrated_frame(df: pd.DataFrame, manifest: dict[str, object]) -> pd.DataFrame:
    method_id = str(manifest.get("selected_method_id", "none"))
    methods = manifest.get("methods", {})
    if method_id in {"none", "uncalibrated"}:
        return df.copy()
    if not isinstance(methods, dict) or method_id not in methods:
        raise ValueError(f"quantile calibration manifest does not contain selected method: {method_id}")
    method = methods[method_id]
    if not isinstance(method, dict) or not isinstance(method.get("config"), dict):
        raise ValueError(f"quantile calibration method {method_id} must contain a config")
    calibrated = apply_method_config(df, method_id, method["config"])
    return as_scoring_frame(calibrated, prefix="calibrated_pred")


def prediction_quantiles(row: pd.Series) -> dict[float, float]:
    return {float(quantile): float(row[f"pred_tmax_{quantile_tag(quantile)}_f"]) for quantile in DEFAULT_QUANTILES}


def selected_distribution_method(method_arg: str, manifest_path: pathlib.Path) -> DistributionMethod:
    if method_arg != "auto":
        return method_arg
    if not manifest_path.exists():
        return FALLBACK_DISTRIBUTION_METHOD_ID
    manifest = json.loads(manifest_path.read_text())
    method_id = manifest.get("selected_distribution_method_id")
    if not isinstance(method_id, str):
        raise ValueError(f"distribution manifest does not contain selected_distribution_method_id: {manifest_path}")
    if method_id not in {"interpolation_tail", "interpolation_no_tail", "smoothed_interpolation_tail", "normal_iqr"}:
        raise ValueError(f"unsupported selected distribution method {method_id!r} in {manifest_path}")
    return method_id


def ladder_records(prediction_df: pd.DataFrame, *, distribution_method: DistributionMethod) -> pd.DataFrame:
    min_temp_f, max_temp_f = ordered_ladder_bounds(prediction_df)
    records: list[dict[str, object]] = []
    for _, row in prediction_df.iterrows():
        ladder = degree_ladder_from_quantiles(
            prediction_quantiles(row),
            method_id=distribution_method,
            min_temp_f=min_temp_f,
            max_temp_f=max_temp_f,
        )
        observed_temp = int(round(float(row["final_tmax_f"])))
        for ladder_row in ladder.itertuples(index=False):
            records.append(
                {
                    "target_date_local": row["target_date_local"],
                    "station_id": row["station_id"],
                    "final_tmax_f": float(row["final_tmax_f"]),
                    "observed_temp_f": observed_temp,
                    "temp_f": int(ladder_row.temp_f),
                    "probability": float(ladder_row.probability),
                    "observed": int(ladder_row.temp_f) == observed_temp,
                }
            )
    return pd.DataFrame(records)


def probability_bucket(probabilities: pd.Series, *, bucket_count: int) -> pd.Series:
    bins = np.linspace(0.0, 1.0, bucket_count + 1)
    return pd.cut(pd.to_numeric(probabilities, errors="coerce").fillna(0.0), bins=bins, include_lowest=True, right=True).astype(str)


def fit_bucket_reliability(records: pd.DataFrame, *, bucket_count: int) -> pd.DataFrame:
    work = records.copy()
    work["probability_bucket"] = probability_bucket(work["probability"], bucket_count=bucket_count)
    grouped = (
        work.groupby("probability_bucket", dropna=False)
        .agg(
            row_count=("observed", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_frequency=("observed", "mean"),
        )
        .reset_index()
    )
    mean_pred = pd.to_numeric(grouped["mean_predicted_probability"], errors="coerce").to_numpy(float)
    observed = pd.to_numeric(grouped["observed_frequency"], errors="coerce").to_numpy(float)
    raw_factor = np.divide(observed, mean_pred, out=np.ones_like(observed), where=mean_pred > EPSILON)
    grouped["raw_factor"] = np.clip(raw_factor, 0.05, 20.0)
    return grouped


def apply_bucket_reliability(records: pd.DataFrame, reliability: pd.DataFrame, *, bucket_count: int, shrinkage: float) -> pd.DataFrame:
    work = records.copy()
    factors = reliability.copy()
    factors["factor"] = 1.0 + float(shrinkage) * (pd.to_numeric(factors["raw_factor"], errors="coerce").fillna(1.0) - 1.0)
    factors["probability_bucket"] = factors["probability_bucket"].astype(str)
    work["probability_bucket"] = probability_bucket(work["probability"], bucket_count=bucket_count)
    work = work.merge(factors[["probability_bucket", "factor"]], on="probability_bucket", how="left")
    work["factor"] = pd.to_numeric(work["factor"], errors="coerce").fillna(1.0).clip(lower=0.0)
    work["calibrated_probability"] = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0) * work["factor"]
    totals = work.groupby(["target_date_local", "station_id"], dropna=False)["calibrated_probability"].transform("sum")
    original = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0)
    work["probability"] = np.where(totals > 0.0, work["calibrated_probability"] / totals, original)
    return work.drop(columns=["factor", "calibrated_probability"])


def score_ladder_records(records: pd.DataFrame, *, method_id: str) -> dict[str, object]:
    degree_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    event_bins = representative_event_bins()
    for (target_date, station_id), group in records.groupby(["target_date_local", "station_id"], sort=False, dropna=False):
        ladder = group[["temp_f", "probability"]].copy()
        observed_temp = int(group["observed_temp_f"].iloc[0])
        probabilities = pd.to_numeric(ladder["probability"], errors="coerce").fillna(0.0).to_numpy(float)
        degrees = pd.to_numeric(ladder["temp_f"]).astype(int).to_numpy()
        observed = (degrees == observed_temp).astype(float)
        observed_probability = float(probabilities[degrees == observed_temp].sum())
        cdf_below = float(probabilities[degrees < observed_temp].sum())
        degree_rows.append(
            {
                "target_date_local": target_date,
                "station_id": station_id,
                "observed_probability": observed_probability,
                "negative_log_likelihood": float(-np.log(max(observed_probability, EPSILON))),
                "brier_score": float(np.sum((probabilities - observed) ** 2)),
                "ranked_probability_score": ranked_probability_score(ladder, observed_temp),
                "pit_mid": cdf_below + 0.5 * observed_probability,
            }
        )
        mapped = map_ladder_to_bins(ladder, event_bins)
        labels = mapped["bin"].astype(str).tolist()
        mapped_probs = pd.to_numeric(mapped["probability"], errors="coerce").fillna(0.0).to_numpy(float)
        observed_bin = observed_event_bin_name(observed_temp, event_bins)
        event_observed = np.array([label == observed_bin for label in labels], dtype=float)
        observed_bin_probability = float(mapped_probs[event_observed.astype(bool)].sum())
        event_rows.append(
            {
                "target_date_local": target_date,
                "station_id": station_id,
                "observed_bin_probability": observed_bin_probability,
                "negative_log_likelihood": float(-np.log(max(observed_bin_probability, EPSILON))),
                "brier_score": float(np.sum((mapped_probs - event_observed) ** 2)),
            }
        )
    degree_df = pd.DataFrame(degree_rows)
    event_df = pd.DataFrame(event_rows)
    return {
        "method_id": method_id,
        "row_count": int(len(degree_df)),
        "degree_ladder_nll": float(degree_df["negative_log_likelihood"].mean()),
        "degree_ladder_brier": float(degree_df["brier_score"].mean()),
        "degree_ladder_rps": float(degree_df["ranked_probability_score"].mean()),
        "degree_ladder_observed_probability": float(degree_df["observed_probability"].mean()),
        "event_bin_nll": float(event_df["negative_log_likelihood"].mean()),
        "event_bin_brier": float(event_df["brier_score"].mean()),
        "event_bin_observed_probability": float(event_df["observed_bin_probability"].mean()),
        "pit_mean": float(degree_df["pit_mid"].mean()),
        "pit_std": float(degree_df["pit_mid"].std(ddof=0)),
    }


def main() -> int:
    args = parse_args()
    df = pd.read_parquet(args.predictions_path)
    if "candidate_id" in df.columns:
        df = df.loc[df["candidate_id"].astype(str) == str(args.candidate_id)].copy()
    if df.empty:
        raise ValueError(f"no rolling-origin predictions for candidate_id={args.candidate_id}")
    manifest = json.loads(args.quantile_calibration_manifest_path.read_text())
    calibration_df = df.loc[df["target_date_local"].astype(str) <= args.calibration_valid_end].copy()
    test_df = df.loc[df["target_date_local"].astype(str) >= args.test_valid_start].copy()
    if calibration_df.empty or test_df.empty:
        raise ValueError("ladder calibration requires non-empty calibration and test slices")
    calibration_frame = selected_quantile_calibrated_frame(calibration_df, manifest)
    test_frame = selected_quantile_calibrated_frame(test_df, manifest)
    distribution_method = selected_distribution_method(args.distribution_method, args.distribution_manifest_path)
    calibration_records = ladder_records(calibration_frame, distribution_method=distribution_method)
    test_records = ladder_records(test_frame, distribution_method=distribution_method)
    reliability = fit_bucket_reliability(calibration_records, bucket_count=args.bucket_count)
    summary_rows = [score_ladder_records(test_records, method_id="quantile_calibrated_ladder")]
    calibrated_records: dict[str, pd.DataFrame] = {}
    for shrinkage in parse_shrinkage_grid(args.shrinkage):
        method_id = f"bucket_reliability_s{shrinkage:.2f}".replace(".", "_")
        adjusted = apply_bucket_reliability(test_records, reliability, bucket_count=args.bucket_count, shrinkage=shrinkage)
        calibrated_records[method_id] = adjusted
        summary_rows.append(score_ladder_records(adjusted, method_id=method_id))

    summary = pd.DataFrame(summary_rows).sort_values(["event_bin_nll", "degree_ladder_nll"]).reset_index(drop=True)
    selected_method_id = str(summary.iloc[0]["method_id"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ladder_calibration_summary.csv"
    reliability_path = args.output_dir / "ladder_calibration_bucket_reliability.csv"
    manifest_path = args.output_dir / "ladder_calibration_manifest.json"
    test_records_path = args.output_dir / "ladder_calibration_test_ladders.parquet"
    summary.to_csv(summary_path, index=False)
    reliability.to_csv(reliability_path, index=False)
    if selected_method_id in calibrated_records:
        output_records = calibrated_records[selected_method_id]
    else:
        output_records = test_records
    output_records.to_parquet(test_records_path, index=False)
    output_manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "predictions_path": str(args.predictions_path),
        "quantile_calibration_manifest_path": str(args.quantile_calibration_manifest_path),
        "distribution_manifest_path": str(args.distribution_manifest_path),
        "distribution_method": distribution_method,
        "candidate_id": args.candidate_id,
        "calibration_valid_end": args.calibration_valid_end,
        "test_valid_start": args.test_valid_start,
        "calibration_row_count": int(calibration_frame[["target_date_local", "station_id"]].drop_duplicates().shape[0]),
        "test_row_count": int(test_frame[["target_date_local", "station_id"]].drop_duplicates().shape[0]),
        "bucket_count": int(args.bucket_count),
        "shrinkage_grid": parse_shrinkage_grid(args.shrinkage),
        "selected_method_id": selected_method_id,
        "selected_by": ["event_bin_nll", "degree_ladder_nll"],
        "summary_path": str(summary_path),
        "bucket_reliability_path": str(reliability_path),
        "test_ladders_path": str(test_records_path),
    }
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(summary_path)
    print(reliability_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
