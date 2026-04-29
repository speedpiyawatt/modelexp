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
from .source_disagreement import DISAGREEMENT_WIDENING_REGIMES, source_disagreement_regime_series
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag
from .distribution import degree_ladder_from_quantiles


DEFAULT_PREDICTIONS_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet")
DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json")
DEFAULT_DISTRIBUTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics/distribution_diagnostics_manifest.json")
DEFAULT_MODEL_SELECTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_manifest.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/ladder_calibration")
FALLBACK_DISTRIBUTION_METHOD_ID = "normal_iqr"
DistributionMethod = str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit final 1F probability-ladder calibration on rolling-origin predictions.")
    parser.add_argument("--predictions-path", type=pathlib.Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--quantile-calibration-manifest-path", type=pathlib.Path, default=DEFAULT_QUANTILE_CALIBRATION_MANIFEST_PATH)
    parser.add_argument("--distribution-manifest-path", type=pathlib.Path, default=DEFAULT_DISTRIBUTION_MANIFEST_PATH)
    parser.add_argument("--distribution-method", default="auto", choices=("auto", "interpolation_tail", "interpolation_no_tail", "smoothed_interpolation_tail", "normal_iqr"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="auto")
    parser.add_argument("--calibration-valid-end", default="2024-12-31")
    parser.add_argument("--test-valid-start", default="2025-01-01")
    parser.add_argument("--bucket-count", type=int, default=10)
    parser.add_argument("--shrinkage", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--disagreement-widening", default="0.0,0.5,1.0,1.5")
    parser.add_argument("--widening-max-overall-event-nll-regression", type=float, default=0.005)
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


def parse_shrinkage_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("shrinkage grid must contain at least one value")
    if any(item < 0.0 or item > 1.0 for item in grid):
        raise ValueError("shrinkage values must be between 0 and 1")
    return grid


def parse_widening_grid(value: str) -> list[float]:
    grid = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not grid:
        raise ValueError("widening grid must contain at least one value")
    if any(item < 0.0 for item in grid):
        raise ValueError("widening values must be non-negative")
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
    regimes = source_disagreement_regime_series(prediction_df)
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
                    "pred_tmax_q50_f": float(row["pred_tmax_q50_f"]),
                    "source_disagreement_regime": str(regimes.loc[row.name]),
                    "temp_f": int(ladder_row.temp_f),
                    "probability": float(ladder_row.probability),
                    "observed": int(ladder_row.temp_f) == observed_temp,
                }
            )
    return pd.DataFrame(records)


def gaussian_kernel(width_f: float) -> np.ndarray:
    width = float(width_f)
    if width <= 0.0:
        return np.asarray([1.0], dtype=float)
    radius = max(1, int(np.ceil(width * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / width) ** 2)
    return kernel / kernel.sum()


def widen_ladder_records(records: pd.DataFrame, *, widening_f: float) -> pd.DataFrame:
    if widening_f <= 0.0:
        return records.copy()
    kernel = gaussian_kernel(widening_f)
    out_frames: list[pd.DataFrame] = []
    for (_target_date, _station_id), group in records.groupby(["target_date_local", "station_id"], sort=False, dropna=False):
        work = group.copy()
        regime = str(work["source_disagreement_regime"].iloc[0]) if "source_disagreement_regime" in work.columns else "unknown"
        if regime in DISAGREEMENT_WIDENING_REGIMES:
            probabilities = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0).to_numpy(float)
            radius = len(kernel) // 2
            padded = np.pad(probabilities, (radius, radius), mode="edge")
            widened = np.convolve(padded, kernel, mode="valid")
            total = float(widened.sum())
            if total > 0.0:
                work["probability"] = widened / total
        out_frames.append(work)
    return pd.concat(out_frames, ignore_index=True) if out_frames else records.copy()


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


def metric_summary_for_records(records: pd.DataFrame, *, method_id: str, slice_name: str) -> dict[str, object]:
    if records.empty:
        return {
            "method_id": method_id,
            "slice": slice_name,
            "row_count": 0,
            "degree_ladder_nll": np.nan,
            "degree_ladder_brier": np.nan,
            "degree_ladder_rps": np.nan,
            "event_bin_nll": np.nan,
            "event_bin_brier": np.nan,
            "q50_mae_f": np.nan,
            "q50_rmse_f": np.nan,
            "event_bin_observed_probability": np.nan,
        }
    scored = score_ladder_records(records, method_id=method_id)
    per_row = records.groupby(["target_date_local", "station_id"], dropna=False).first().reset_index()
    observed = pd.to_numeric(per_row["final_tmax_f"], errors="coerce").to_numpy(float)
    predicted = pd.to_numeric(per_row["pred_tmax_q50_f"], errors="coerce").to_numpy(float)
    return {
        "method_id": method_id,
        "slice": slice_name,
        "row_count": scored["row_count"],
        "degree_ladder_nll": scored["degree_ladder_nll"],
        "degree_ladder_brier": scored["degree_ladder_brier"],
        "degree_ladder_rps": scored["degree_ladder_rps"],
        "event_bin_nll": scored["event_bin_nll"],
        "event_bin_brier": scored["event_bin_brier"],
        "q50_mae_f": float(np.mean(np.abs(predicted - observed))),
        "q50_rmse_f": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "event_bin_observed_probability": scored["event_bin_observed_probability"],
    }


def score_records_by_disagreement_slice(records: pd.DataFrame, *, method_id: str) -> pd.DataFrame:
    rows = [metric_summary_for_records(records, method_id=method_id, slice_name="overall")]
    if "source_disagreement_regime" in records.columns:
        regimes = records["source_disagreement_regime"].astype(str)
        high_mask = regimes.isin(DISAGREEMENT_WIDENING_REGIMES)
        native_warm_mask = regimes == "native_warm_hrrr_cold"
        rows.append(metric_summary_for_records(records.loc[high_mask].copy(), method_id=method_id, slice_name="high_disagreement"))
        rows.append(metric_summary_for_records(records.loc[native_warm_mask].copy(), method_id=method_id, slice_name="native_warm_hrrr_cold"))
        for regime in sorted(regimes.dropna().unique()):
            rows.append(metric_summary_for_records(records.loc[regimes == regime].copy(), method_id=method_id, slice_name=f"regime:{regime}"))
    return pd.DataFrame(rows)


def select_ladder_method(summary: pd.DataFrame, slice_metrics: pd.DataFrame, *, max_overall_regression: float) -> str:
    sorted_summary = summary.sort_values(["event_bin_nll", "degree_ladder_nll"]).reset_index(drop=True)
    best_by_overall = str(sorted_summary.iloc[0]["method_id"])
    non_widening = sorted_summary.loc[~sorted_summary["method_id"].astype(str).str.startswith("source_disagreement_widen_")]
    if non_widening.empty:
        return best_by_overall
    baseline = non_widening.iloc[0]
    baseline_method_id = str(baseline["method_id"])
    baseline_event_nll = float(baseline["event_bin_nll"])
    high = slice_metrics.loc[slice_metrics["slice"] == "high_disagreement"].copy()
    if high.empty:
        return baseline_method_id
    baseline_high = high.loc[high["method_id"] == baseline_method_id]
    if baseline_high.empty:
        baseline_high = high.loc[high["method_id"] == "quantile_calibrated_ladder"]
    if baseline_high.empty:
        return baseline_method_id
    baseline_high_event_nll = float(baseline_high["event_bin_nll"].iloc[0])
    widening_candidates = sorted_summary.loc[sorted_summary["method_id"].astype(str).str.startswith("source_disagreement_widen_")]
    for _, candidate in widening_candidates.iterrows():
        candidate_method_id = str(candidate["method_id"])
        candidate_event_nll = float(candidate["event_bin_nll"])
        if candidate_event_nll > baseline_event_nll + float(max_overall_regression):
            continue
        candidate_high = high.loc[high["method_id"] == candidate_method_id]
        if candidate_high.empty:
            continue
        high_improvement = baseline_high_event_nll - float(candidate_high["event_bin_nll"].iloc[0])
        if high_improvement > 0.0:
            return candidate_method_id
    return baseline_method_id


def main() -> int:
    args = parse_args()
    candidate_id = resolve_candidate_id(str(args.candidate_id))
    df = pd.read_parquet(args.predictions_path)
    if "candidate_id" in df.columns:
        df = df.loc[df["candidate_id"].astype(str) == candidate_id].copy()
    if df.empty:
        raise ValueError(f"no rolling-origin predictions for candidate_id={candidate_id}")
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
    slice_metric_frames = [score_records_by_disagreement_slice(test_records, method_id="quantile_calibrated_ladder")]
    calibrated_records: dict[str, pd.DataFrame] = {}
    for shrinkage in parse_shrinkage_grid(args.shrinkage):
        method_id = f"bucket_reliability_s{shrinkage:.2f}".replace(".", "_")
        adjusted = apply_bucket_reliability(test_records, reliability, bucket_count=args.bucket_count, shrinkage=shrinkage)
        calibrated_records[method_id] = adjusted
        summary_rows.append(score_ladder_records(adjusted, method_id=method_id))
        slice_metric_frames.append(score_records_by_disagreement_slice(adjusted, method_id=method_id))
    for widening in parse_widening_grid(args.disagreement_widening):
        if widening <= 0.0:
            continue
        method_id = f"source_disagreement_widen_{widening:.2f}f".replace(".", "_")
        widened = widen_ladder_records(test_records, widening_f=widening)
        calibrated_records[method_id] = widened
        summary_rows.append(score_ladder_records(widened, method_id=method_id))
        slice_metric_frames.append(score_records_by_disagreement_slice(widened, method_id=method_id))

    summary = pd.DataFrame(summary_rows).sort_values(["event_bin_nll", "degree_ladder_nll"]).reset_index(drop=True)
    slice_metrics = pd.concat(slice_metric_frames, ignore_index=True) if slice_metric_frames else pd.DataFrame()
    selected_method_id = select_ladder_method(
        summary,
        slice_metrics,
        max_overall_regression=args.widening_max_overall_event_nll_regression,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "ladder_calibration_summary.csv"
    slice_metrics_path = args.output_dir / "ladder_calibration_disagreement_slices.csv"
    reliability_path = args.output_dir / "ladder_calibration_bucket_reliability.csv"
    manifest_path = args.output_dir / "ladder_calibration_manifest.json"
    test_records_path = args.output_dir / "ladder_calibration_test_ladders.parquet"
    summary.to_csv(summary_path, index=False)
    slice_metrics.to_csv(slice_metrics_path, index=False)
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
        "candidate_id": candidate_id,
        "calibration_valid_end": args.calibration_valid_end,
        "test_valid_start": args.test_valid_start,
        "calibration_row_count": int(calibration_frame[["target_date_local", "station_id"]].drop_duplicates().shape[0]),
        "test_row_count": int(test_frame[["target_date_local", "station_id"]].drop_duplicates().shape[0]),
        "bucket_count": int(args.bucket_count),
        "shrinkage_grid": parse_shrinkage_grid(args.shrinkage),
        "disagreement_widening_grid_f": parse_widening_grid(args.disagreement_widening),
        "disagreement_widening_regimes": sorted(DISAGREEMENT_WIDENING_REGIMES),
        "widening_max_overall_event_nll_regression": float(args.widening_max_overall_event_nll_regression),
        "selected_method_id": selected_method_id,
        "selected_by": [
            "event_bin_nll",
            "degree_ladder_nll",
            "source_disagreement_widening_requires_high_disagreement_improvement_when_overall_regresses",
        ],
        "summary_path": str(summary_path),
        "disagreement_slice_metrics_path": str(slice_metrics_path),
        "bucket_reliability_path": str(reliability_path),
        "test_ladders_path": str(test_records_path),
    }
    manifest_path.write_text(json.dumps(output_manifest, indent=2, sort_keys=True) + "\n")
    print(manifest_path)
    print(summary_path)
    print(slice_metrics_path)
    print(reliability_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
