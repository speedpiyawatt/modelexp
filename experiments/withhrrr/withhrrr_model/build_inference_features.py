from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import numpy as np
import pandas as pd

from tools.weather.build_training_features_overnight import (
    build_training_features_overnight,
    filter_target_date_window,
    _read_hrrr_summaries,
    _read_named_parquets,
    _read_parquet,
)
from tools.weather.build_training_features_overnight_normalized import (
    load_vocabularies,
    normalize_training_features_overnight,
)

from .prepare_training_features import build_training_table


DEFAULT_LABEL_HISTORY_PATH = pathlib.Path("wunderground/output/tables/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("wunderground/output/tables/wu_obs_intraday.parquet")
DEFAULT_NBM_ROOT = pathlib.Path("data/runtime/backfill_overnight/nbm_overnight")
DEFAULT_LAMP_ROOT = pathlib.Path("experiments/no_hrrr_model/data/runtime/lamp_overnight")
DEFAULT_HRRR_ROOT = pathlib.Path("experiments/withhrrr/data/runtime/source/hrrr_summary")
DEFAULT_FEATURE_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/models/feature_manifest.json")
DEFAULT_VOCAB_PATH = pathlib.Path("tools/weather/training_feature_vocabularies.json")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/predictions/features")
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one normalized with-HRRR inference feature row from local source artifacts.")
    parser.add_argument("--target-date-local", required=True)
    parser.add_argument("--station-id", default="KLGA")
    parser.add_argument("--label-history-path", type=pathlib.Path, default=DEFAULT_LABEL_HISTORY_PATH)
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--nbm-root", type=pathlib.Path, default=DEFAULT_NBM_ROOT)
    parser.add_argument("--lamp-root", type=pathlib.Path, default=DEFAULT_LAMP_ROOT)
    parser.add_argument("--hrrr-root", type=pathlib.Path, default=DEFAULT_HRRR_ROOT)
    parser.add_argument("--feature-manifest-path", type=pathlib.Path, default=DEFAULT_FEATURE_MANIFEST_PATH)
    parser.add_argument("--vocab-path", type=pathlib.Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME)
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


def source_available(df: pd.DataFrame, column: str) -> bool:
    return bool(df[column].astype("boolean").fillna(False).iloc[0]) if column in df.columns and not df.empty else False


def prediction_available(model_ready: pd.DataFrame) -> pd.Series:
    return (
        model_ready["meta_nbm_available"].astype("boolean").fillna(False)
        & model_ready["meta_lamp_available"].astype("boolean").fillna(False)
        & model_ready["meta_hrrr_available"].astype("boolean").fillna(False)
        & pd.to_numeric(model_ready["nbm_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(model_ready["lamp_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(model_ready["hrrr_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(model_ready["anchor_tmax_f"], errors="coerce").notna()
    )


def main() -> int:
    args = parse_args()
    feature_manifest = load_json(args.feature_manifest_path)
    anchor_policy = str(feature_manifest.get("anchor_policy", "equal_3way"))
    ridge_metadata = feature_manifest.get("ridge_anchor_metadata") if isinstance(feature_manifest.get("ridge_anchor_metadata"), dict) else None
    label_history_df = filter_label_history_for_inference(
        _read_parquet(args.label_history_path),
        target_date_local=args.target_date_local,
        station_id=args.station_id,
    )
    labels_df = dummy_label_row(target_date_local=args.target_date_local, station_id=args.station_id)
    obs_df = _read_parquet(args.obs_path)
    nbm_df = filter_target_date_window(
        _read_named_parquets(args.nbm_root, "nbm.overnight.parquet"),
        start_date=dt.date.fromisoformat(args.target_date_local),
        end_date=dt.date.fromisoformat(args.target_date_local),
    )
    lamp_df = filter_target_date_window(
        _read_named_parquets(args.lamp_root, "lamp.overnight.parquet"),
        start_date=dt.date.fromisoformat(args.target_date_local),
        end_date=dt.date.fromisoformat(args.target_date_local),
    )
    hrrr_df = filter_target_date_window(
        _read_hrrr_summaries(args.hrrr_root),
        start_date=dt.date.fromisoformat(args.target_date_local),
        end_date=dt.date.fromisoformat(args.target_date_local),
    )
    unnormalized = build_training_features_overnight(
        labels_df=labels_df,
        obs_df=obs_df,
        nbm_daily_df=nbm_df,
        lamp_daily_df=lamp_df,
        hrrr_daily_df=hrrr_df,
        cutoff_local_time=args.cutoff_local_time,
        station_id=args.station_id,
        label_history_df=label_history_df,
    )
    if unnormalized.empty:
        raise ValueError(f"failed to build inference row for {args.station_id} {args.target_date_local}")

    normalized = normalize_training_features_overnight(unnormalized, load_vocabularies(args.vocab_path))
    model_ready = build_training_table(normalized, hrrr=None, anchor_policy=anchor_policy, ridge_metadata=ridge_metadata)
    model_ready["model_prediction_available"] = prediction_available(model_ready)
    if not bool(model_ready["model_prediction_available"].iloc[0]):
        availability = {
            "meta_nbm_available": source_available(model_ready, "meta_nbm_available"),
            "meta_lamp_available": source_available(model_ready, "meta_lamp_available"),
            "meta_hrrr_available": source_available(model_ready, "meta_hrrr_available"),
        }
        raise ValueError(f"with-HRRR prediction sources unavailable for {args.station_id} {args.target_date_local}: {availability}")

    feature_columns = list(feature_manifest["feature_columns"])
    for column in feature_columns:
        if column not in model_ready.columns:
            model_ready[column] = pd.NA

    output_root = args.output_dir / f"target_date_local={args.target_date_local}"
    output_root.mkdir(parents=True, exist_ok=True)
    unnormalized_path = output_root / "withhrrr.inference_features.parquet"
    normalized_path = output_root / "withhrrr.inference_features_normalized.parquet"
    manifest_path = output_root / "withhrrr.inference_features.manifest.json"
    unnormalized.to_parquet(unnormalized_path, index=False)
    model_ready.to_parquet(normalized_path, index=False)
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_date_local": args.target_date_local,
        "station_id": args.station_id,
        "model_prediction_available": bool(model_ready["model_prediction_available"].iloc[0]),
        "source_availability": {
            "meta_nbm_available": source_available(model_ready, "meta_nbm_available"),
            "meta_lamp_available": source_available(model_ready, "meta_lamp_available"),
            "meta_hrrr_available": source_available(model_ready, "meta_hrrr_available"),
        },
        "unnormalized_path": str(unnormalized_path),
        "normalized_path": str(normalized_path),
        "feature_count": len(feature_columns),
        "anchor_policy": anchor_policy,
        "feature_profile": feature_manifest.get("feature_profile"),
        "weight_profile": feature_manifest.get("weight_profile"),
        "meta_residual": feature_manifest.get("meta_residual"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(normalized_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
