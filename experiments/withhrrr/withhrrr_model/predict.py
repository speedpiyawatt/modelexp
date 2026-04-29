from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import pandas as pd

from .distribution import degree_ladder_from_quantiles, expected_temperature
from .event_bins import load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from .train_quantile_models import DEFAULT_QUANTILES, quantile_tag


DEFAULT_FEATURES_PATH = pathlib.Path("experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet")
DEFAULT_MODELS_DIR = pathlib.Path("experiments/withhrrr/data/runtime/models")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/predictions")
DEFAULT_CALIBRATION_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json")
DEFAULT_DISTRIBUTION_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics/distribution_diagnostics_manifest.json")
FALLBACK_DISTRIBUTION_METHOD_ID = "interpolation_tail"
DISTRIBUTION_METHOD_IDS = ["auto", "interpolation_tail", "interpolation_no_tail", "smoothed_interpolation_tail", "normal_iqr"]
ANCHOR_POLICY = "feature_anchor_tmax_f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate with-HRRR final-Tmax probabilities for a normalized feature row.")
    parser.add_argument("--features-path", type=pathlib.Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--models-dir", type=pathlib.Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-date-local", required=True)
    parser.add_argument("--station-id", default="KLGA")
    parser.add_argument("--event-bin", action="append", default=[], help="Optional event bin label. May be repeated.")
    parser.add_argument("--event-bins-path", type=pathlib.Path, default=None, help="Optional JSON or text file containing event-bin labels.")
    parser.add_argument("--max-so-far-f", type=float, default=None)
    parser.add_argument("--calibration-path", type=pathlib.Path, default=None, help="Optional calibration manifest with final-quantile offsets.")
    parser.add_argument("--no-calibration", action="store_true", help="Disable automatic quantile calibration offsets.")
    parser.add_argument(
        "--distribution-method",
        choices=DISTRIBUTION_METHOD_IDS,
        default="auto",
        help="Internal 1F ladder construction method. auto uses the selected Phase 6 manifest when present.",
    )
    parser.add_argument(
        "--distribution-manifest-path",
        type=pathlib.Path,
        default=None,
        help="Optional Phase 6 distribution manifest used when --distribution-method=auto.",
    )
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())


def prepare_features(row: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    x = row.loc[:, feature_columns].copy()
    for column in x.columns:
        if pd.api.types.is_bool_dtype(x[column].dtype):
            x[column] = x[column].astype("int8")
        else:
            x[column] = pd.to_numeric(x[column], errors="coerce")
    return x


def select_row(df: pd.DataFrame, *, target_date_local: str, station_id: str) -> pd.DataFrame:
    row = df.loc[(df["target_date_local"].astype(str) == target_date_local) & (df["station_id"].astype(str) == station_id)].copy()
    if row.empty:
        raise ValueError(f"no feature row found for target_date_local={target_date_local} station_id={station_id}")
    if len(row) > 1:
        raise ValueError(f"multiple feature rows found for target_date_local={target_date_local} station_id={station_id}")
    if "model_prediction_available" in row.columns:
        available = bool(row["model_prediction_available"].astype("boolean").fillna(False).iloc[0])
    elif "model_training_eligible" in row.columns:
        available = bool(row["model_training_eligible"].astype("boolean").fillna(False).iloc[0])
    else:
        available = True
    if not available:
        raise ValueError(f"feature row is not model eligible for target_date_local={target_date_local} station_id={station_id}")
    return row


def _single_row_value(row: pd.DataFrame | None, column: str) -> object | None:
    if row is None or column not in row.columns or row.empty:
        return None
    return row[column].iloc[0]


def _calibration_segment(row: pd.DataFrame | None, segment_name: str) -> str | None:
    if segment_name == "season":
        value = _single_row_value(row, "target_date_local")
        month = pd.to_datetime(value, errors="coerce").month if value is not None else None
        if month is None or pd.isna(month):
            return None
        return "warm" if int(month) in {5, 6, 7, 8, 9} else "cool"
    if segment_name == "month":
        value = _single_row_value(row, "target_date_local")
        month = pd.to_datetime(value, errors="coerce").month if value is not None else None
        if month is None or pd.isna(month):
            return "month_unknown"
        return f"month_{int(month):02d}"
    if segment_name == "disagreement":
        value = pd.to_numeric(_single_row_value(row, "nbm_minus_lamp_tmax_f"), errors="coerce")
        if pd.isna(value):
            return None
        disagreement = abs(float(value))
        if disagreement < 2.0:
            return "under_2f"
        if disagreement < 5.0:
            return "2_to_5f"
        return "5f_or_more"
    return None


def calibration_offsets(payload: dict[str, object], row: pd.DataFrame | None = None) -> dict[str, float]:
    offsets = payload.get("offsets_f")
    if isinstance(offsets, dict):
        return {str(key): float(value) for key, value in offsets.items()}
    selected_method_id = payload.get("selected_method_id")
    methods = payload.get("methods")
    if isinstance(selected_method_id, str) and isinstance(methods, dict):
        selected = methods.get(selected_method_id)
        if isinstance(selected, dict):
            config = selected.get("config")
            if isinstance(config, dict):
                method_offsets = config.get("offsets_f")
                if isinstance(method_offsets, dict):
                    return {str(key): float(value) for key, value in method_offsets.items()}
                global_offsets = config.get("global_offsets_f")
                segment_offsets = config.get("segment_offsets_f")
                segment_name = config.get("segment_name")
                if isinstance(global_offsets, dict):
                    if isinstance(segment_offsets, dict) and isinstance(segment_name, str):
                        segment = _calibration_segment(row, segment_name)
                        selected_offsets = segment_offsets.get(str(segment)) if segment is not None else None
                        if isinstance(selected_offsets, dict):
                            return {str(key): float(value) for key, value in selected_offsets.items()}
                    return {str(key): float(value) for key, value in global_offsets.items()}
                interval_adjustments = config.get("interval_adjustments_f")
                if isinstance(interval_adjustments, dict):
                    pair_adjustments = {
                        "q05": -float(interval_adjustments["q05_q95"]),
                        "q95": float(interval_adjustments["q05_q95"]),
                        "q10": -float(interval_adjustments["q10_q90"]),
                        "q90": float(interval_adjustments["q10_q90"]),
                        "q25": -float(interval_adjustments["q25_q75"]),
                        "q75": float(interval_adjustments["q25_q75"]),
                    }
                    median_offsets = config.get("median_offsets_f")
                    if isinstance(median_offsets, dict):
                        pair_adjustments["q50"] = float(median_offsets.get("q50", 0.0))
                    else:
                        pair_adjustments["q50"] = 0.0
                    return pair_adjustments
    return {}


def selected_distribution_method(
    method_arg: str,
    *,
    manifest_path: pathlib.Path | None = None,
) -> tuple[str, pathlib.Path | None]:
    if method_arg != "auto":
        return method_arg, None
    selected_path = manifest_path
    if selected_path is None and DEFAULT_DISTRIBUTION_MANIFEST_PATH.exists():
        selected_path = DEFAULT_DISTRIBUTION_MANIFEST_PATH
    if selected_path is None:
        return FALLBACK_DISTRIBUTION_METHOD_ID, None
    payload = load_json(selected_path)
    method_id = payload.get("selected_distribution_method_id")
    if not isinstance(method_id, str):
        raise ValueError(f"distribution manifest does not contain selected_distribution_method_id: {selected_path}")
    if method_id not in DISTRIBUTION_METHOD_IDS or method_id == "auto":
        raise ValueError(f"unsupported selected distribution method {method_id!r} in {selected_path}")
    return method_id, selected_path


def print_prediction_summary(payload: dict[str, object], output_path: pathlib.Path) -> None:
    print(f"prediction_path={output_path}")
    print(f"status={payload['status']} target_date_local={payload['target_date_local']} station_id={payload['station_id']}")
    print(f"expected_final_tmax_f={float(payload['expected_final_tmax_f']):.2f}")
    print(f"anchor_tmax_f={float(payload['anchor_tmax_f']):.2f}")
    print(f"distribution_method={payload['distribution_method']}")
    final_quantiles = payload.get("final_tmax_quantiles_f", {})
    if isinstance(final_quantiles, dict):
        labels = (("0.05", "q05"), ("0.1", "q10"), ("0.25", "q25"), ("0.5", "q50"), ("0.75", "q75"), ("0.9", "q90"), ("0.95", "q95"))
        values = " ".join(f"{label}={float(final_quantiles[key]):.2f}" for key, label in labels if key in final_quantiles)
        if values:
            print(f"final_tmax_quantiles_f {values}")
    event_bins = payload.get("event_bins", [])
    if isinstance(event_bins, list) and event_bins:
        print("event_bins")
        for row in event_bins:
            if isinstance(row, dict):
                print(f"  {row.get('bin')}: {float(row.get('probability', 0.0)):.4f}")


def main() -> int:
    args = parse_args()
    feature_manifest = load_json(args.models_dir / "feature_manifest.json")
    feature_columns = list(feature_manifest["feature_columns"])
    df = pd.read_parquet(args.features_path)
    row = select_row(df, target_date_local=args.target_date_local, station_id=args.station_id)
    x = prepare_features(row, feature_columns)
    anchor_tmax_f = float(pd.to_numeric(row["anchor_tmax_f"], errors="coerce").iloc[0])
    calibration_path = args.calibration_path
    if args.no_calibration:
        calibration_path = None
    elif calibration_path is None and DEFAULT_CALIBRATION_PATH.exists():
        calibration_path = DEFAULT_CALIBRATION_PATH
    offsets: dict[str, float] = {}
    if calibration_path is not None:
        offsets = calibration_offsets(load_json(calibration_path), row=row)
    distribution_method, distribution_manifest_path = selected_distribution_method(
        args.distribution_method,
        manifest_path=args.distribution_manifest_path,
    )

    residual_quantiles: dict[float, float] = {}
    final_quantiles: dict[float, float] = {}
    raw_final_values = []
    for quantile in DEFAULT_QUANTILES:
        tag = quantile_tag(quantile)
        booster = lgb.Booster(model_file=str(args.models_dir / f"residual_quantile_{tag}.txt"))
        residual = float(booster.predict(x, num_iteration=booster.best_iteration)[0])
        residual_quantiles[float(quantile)] = residual
        raw_final_values.append(anchor_tmax_f + residual + offsets.get(tag, 0.0))

    rearranged_final_values = np.maximum.accumulate(np.asarray(raw_final_values, dtype=float))
    for quantile, value in zip(DEFAULT_QUANTILES, rearranged_final_values):
        final_quantiles[float(quantile)] = float(value)

    ladder = degree_ladder_from_quantiles(final_quantiles, method_id=distribution_method)
    bin_records: list[dict[str, object]] = []
    event_bin_labels = list(args.event_bin)
    if args.event_bins_path is not None:
        event_bin_labels.extend(load_event_bin_labels(args.event_bins_path))
    if event_bin_labels:
        bins = [parse_event_bin(label) for label in event_bin_labels]
        bin_records = map_ladder_to_bins(ladder, bins, max_so_far_f=args.max_so_far_f).to_dict("records")

    payload = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_date_local": args.target_date_local,
        "station_id": args.station_id,
        "anchor_tmax_f": anchor_tmax_f,
        "anchor_policy": ANCHOR_POLICY,
        "expected_final_tmax_f": expected_temperature(ladder),
        "residual_quantiles_f": {str(key): value for key, value in residual_quantiles.items()},
        "final_tmax_quantiles_f": {str(key): value for key, value in final_quantiles.items()},
        "degree_ladder": ladder.to_dict("records"),
        "event_bins": bin_records,
        "event_bin_labels": event_bin_labels,
        "max_so_far_f": args.max_so_far_f,
        "calibration_path": str(calibration_path) if calibration_path is not None else None,
        "calibration_enabled": calibration_path is not None,
        "calibration_offsets_f": offsets,
        "distribution_method": distribution_method,
        "distribution_manifest_path": str(distribution_manifest_path) if distribution_manifest_path is not None else None,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"prediction_{args.station_id}_{args.target_date_local}.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print_prediction_summary(payload, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
