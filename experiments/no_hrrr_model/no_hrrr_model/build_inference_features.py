from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from .build_training_features import build_training_features, filter_date_window, read_named_parquets, read_parquet
from .normalize_features import normalize_features


DEFAULT_LABEL_HISTORY_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/wunderground/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/wunderground/wu_obs_intraday.parquet")
DEFAULT_NBM_ROOT = pathlib.Path("data/runtime/backfill_overnight/nbm_overnight")
DEFAULT_LAMP_ROOT = pathlib.Path("experiments/no_hrrr_model/data/runtime/lamp_overnight")
DEFAULT_NORMALIZED_MANIFEST_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.manifest.json")
DEFAULT_FEATURE_MANIFEST_PATH = pathlib.Path("experiments/no_hrrr_model/data/runtime/models/feature_manifest.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/no_hrrr_model/data/runtime/predictions/features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one normalized no-HRRR inference feature row from local source artifacts.")
    parser.add_argument("--target-date-local", required=True)
    parser.add_argument("--station-id", default="KLGA")
    parser.add_argument("--label-history-path", type=pathlib.Path, default=DEFAULT_LABEL_HISTORY_PATH)
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--nbm-root", type=pathlib.Path, default=DEFAULT_NBM_ROOT)
    parser.add_argument("--lamp-root", type=pathlib.Path, default=DEFAULT_LAMP_ROOT)
    parser.add_argument("--normalized-manifest-path", type=pathlib.Path, default=DEFAULT_NORMALIZED_MANIFEST_PATH)
    parser.add_argument("--feature-manifest-path", type=pathlib.Path, default=DEFAULT_FEATURE_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())


def dummy_label_row(*, target_date_local: str, station_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target_date_local": target_date_local,
                "station_id": station_id,
                "label_final_tmax_f": np.nan,
                "label_final_tmin_f": np.nan,
                "label_market_bin": None,
                "label_obs_count": 0,
                "label_first_obs_time_local": None,
                "label_last_obs_time_local": None,
                "label_total_precip_in": np.nan,
                "label_source_file": None,
            }
        ]
    )


def filter_label_history_for_inference(label_history_df: pd.DataFrame, *, target_date_local: str, station_id: str) -> pd.DataFrame:
    if label_history_df.empty or "target_date_local" not in label_history_df.columns:
        return label_history_df.copy()
    out = label_history_df.copy()
    out = out.loc[out["target_date_local"].astype(str) < str(target_date_local)]
    if "station_id" in out.columns:
        out = out.loc[out["station_id"].astype(str) == str(station_id)]
    return out.reset_index(drop=True)


def main() -> int:
    args = parse_args()
    label_history_df = filter_label_history_for_inference(
        read_parquet(args.label_history_path),
        target_date_local=args.target_date_local,
        station_id=args.station_id,
    )
    labels_df = dummy_label_row(target_date_local=args.target_date_local, station_id=args.station_id)
    obs_df = read_parquet(args.obs_path)
    nbm_df = filter_date_window(read_named_parquets(args.nbm_root, "nbm.overnight.parquet"), args.target_date_local, args.target_date_local)
    lamp_df = filter_date_window(read_named_parquets(args.lamp_root, "lamp.overnight.parquet"), args.target_date_local, args.target_date_local)
    unnormalized = build_training_features(
        labels_df=labels_df,
        obs_df=obs_df,
        nbm_df=nbm_df,
        lamp_df=lamp_df,
        station_id=args.station_id,
        label_history_df=label_history_df,
    )
    if unnormalized.empty:
        raise ValueError(f"failed to build inference row for {args.station_id} {args.target_date_local}")
    unnormalized["model_prediction_available"] = (
        unnormalized["meta_nbm_available"].astype("boolean").fillna(False)
        & unnormalized["meta_lamp_available"].astype("boolean").fillna(False)
        & pd.to_numeric(unnormalized["nbm_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(unnormalized["lamp_tmax_open_f"], errors="coerce").notna()
    )
    if not bool(unnormalized["model_prediction_available"].iloc[0]):
        raise ValueError(f"NBM/LAMP anchors are unavailable for {args.station_id} {args.target_date_local}")

    normalized_manifest = load_json(args.normalized_manifest_path)
    vocabularies = {str(key): [str(value) for value in values] for key, values in normalized_manifest.get("categorical_vocabularies", {}).items()}
    normalized, _ = normalize_features(unnormalized, vocabularies=vocabularies)
    feature_columns = list(load_json(args.feature_manifest_path)["feature_columns"])
    for column in feature_columns:
        if column not in normalized.columns:
            normalized[column] = pd.NA

    output_root = args.output_dir / f"target_date_local={args.target_date_local}"
    output_root.mkdir(parents=True, exist_ok=True)
    unnormalized_path = output_root / "no_hrrr.inference_features.parquet"
    normalized_path = output_root / "no_hrrr.inference_features_normalized.parquet"
    manifest_path = output_root / "no_hrrr.inference_features.manifest.json"
    unnormalized.to_parquet(unnormalized_path, index=False)
    normalized.to_parquet(normalized_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_date_local": args.target_date_local,
        "station_id": args.station_id,
        "model_prediction_available": bool(unnormalized["model_prediction_available"].iloc[0]),
        "unnormalized_path": str(unnormalized_path),
        "normalized_path": str(normalized_path),
        "feature_count": len(feature_columns),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(normalized_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
