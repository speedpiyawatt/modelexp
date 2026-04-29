from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

import pandas as pd

from .source_disagreement import add_source_disagreement_features
from .source_trust import SOURCE_TRUST_FEATURE_COLUMNS, add_source_trust_features, apply_anchor_policy


DEFAULT_BASE_PATH = pathlib.Path(
    "experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.parquet"
)
DEFAULT_HRRR_ROOT = pathlib.Path("experiments/withhrrr/data/runtime/source/hrrr_summary")
DEFAULT_OUTPUT_PATH = pathlib.Path(
    "experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet"
)
DEFAULT_MANIFEST_PATH = pathlib.Path(
    "experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.manifest.json"
)
DEFAULT_ANCHOR_POLICY = "equal_3way"
ANCHOR_POLICIES = (
    "current_50_50",
    "hourly_native_lamp",
    "hourly_native_lamp_hrrr",
    "native_lamp",
    "equal_3way",
    "source_median_4way",
    "source_trimmed_mean_4way",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the with-HRRR model training table.")
    parser.add_argument("--base-path", type=pathlib.Path, default=DEFAULT_BASE_PATH)
    parser.add_argument("--hrrr-root", type=pathlib.Path, default=DEFAULT_HRRR_ROOT)
    parser.add_argument("--output-path", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest-output-path", type=pathlib.Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument(
        "--anchor-policy",
        choices=ANCHOR_POLICIES,
        default=DEFAULT_ANCHOR_POLICY,
        help="Anchor formula used for the residual target. Defaults to the selected equal NBM-hourly/LAMP/HRRR anchor.",
    )
    return parser.parse_args()


def read_hrrr_summaries(root: pathlib.Path) -> pd.DataFrame:
    paths = sorted(root.glob("target_date_local=*/hrrr.overnight.parquet"))
    if not paths:
        raise FileNotFoundError(f"no HRRR summary parquet files found under {root}")
    frames = [pd.read_parquet(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise ValueError(f"HRRR summary parquet files were all empty under {root}")
    df = pd.concat(frames, ignore_index=True)
    if "target_date_local" not in df.columns:
        raise ValueError("HRRR summaries are missing target_date_local")
    return df.drop_duplicates(subset=["target_date_local"]).reset_index(drop=True)


def kelvin_to_f(series: pd.Series) -> pd.Series:
    return (pd.to_numeric(series, errors="coerce") - 273.15) * 9.0 / 5.0 + 32.0


def ms_to_mph(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") * 2.2369362920544


def kg_m2_to_in(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce") / 25.4


def normalize_hrrr_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in list(out.columns):
        if not column.startswith("hrrr_"):
            continue
        if column.endswith("_k"):
            out[f"{column[:-2]}_f"] = kelvin_to_f(out[column])
        elif column.endswith("_ms"):
            out[f"{column[:-3]}_mph"] = ms_to_mph(out[column])
        elif column.endswith("_kg_m2"):
            out[f"{column[:-6]}_in"] = kg_m2_to_in(out[column])
    return out


def transform_hrrr_for_join(hrrr: pd.DataFrame) -> pd.DataFrame:
    out = hrrr.copy()
    out["target_date_local"] = out["target_date_local"].astype(str)
    out["meta_hrrr_available"] = True
    out["meta_hrrr_anchor_init_time_utc"] = out.get("anchor_init_time_utc")
    out["meta_hrrr_anchor_init_time_local"] = out.get("anchor_init_time_local")
    rename_map = {
        "retained_cycle_count": "meta_hrrr_retained_cycle_count",
        "first_valid_hour_local": "meta_hrrr_first_valid_hour_local",
        "last_valid_hour_local": "meta_hrrr_last_valid_hour_local",
        "covered_hour_count": "meta_hrrr_covered_hour_count",
        "covered_checkpoint_count": "meta_hrrr_covered_checkpoint_count",
        "coverage_end_hour_local": "meta_hrrr_coverage_end_hour_local",
        "has_full_day_21_local_coverage": "meta_hrrr_has_full_day_21_local_coverage",
        "missing_checkpoint_count": "meta_hrrr_missing_checkpoint_count",
    }
    out = out.rename(columns={key: value for key, value in rename_map.items() if key in out.columns})
    out = normalize_hrrr_columns(out)
    keep = [
        "target_date_local",
        "meta_hrrr_available",
        "meta_hrrr_anchor_init_time_utc",
        "meta_hrrr_anchor_init_time_local",
        *rename_map.values(),
        *[column for column in out.columns if column.startswith("hrrr_")],
    ]
    return out.loc[:, [column for column in keep if column in out.columns]]


def _numeric_column(df: pd.DataFrame, *candidates: str) -> pd.Series:
    for column in candidates:
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(pd.NA, index=df.index, dtype="Float64")


def _bool_column(df: pd.DataFrame, column: str, *, default: bool = False) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="boolean")
    return df[column].astype("boolean").fillna(default)


def _prepare_base_columns(base: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    if "label_final_tmax_f" not in base.columns and "final_tmax_f" in base.columns:
        base["label_final_tmax_f"] = base["final_tmax_f"]
    if "label_final_tmin_f" not in base.columns and "final_tmin_f" in base.columns:
        base["label_final_tmin_f"] = base["final_tmin_f"]
    if "final_tmax_f" not in base.columns and "label_final_tmax_f" in base.columns:
        base["final_tmax_f"] = base["label_final_tmax_f"]
    if "final_tmin_f" not in base.columns and "label_final_tmin_f" in base.columns:
        base["final_tmin_f"] = base["label_final_tmin_f"]

    if "nbm_tmax_open_f" not in base.columns:
        if "nbm_temp_2m_day_max_f" in base.columns:
            base["nbm_tmax_open_f"] = _numeric_column(base, "nbm_temp_2m_day_max_f")
        else:
            base["nbm_tmax_open_f"] = kelvin_to_f(_numeric_column(base, "nbm_temp_2m_day_max_k"))
    if "nbm_native_tmax_2m_day_max_f" not in base.columns and "nbm_native_tmax_2m_day_max_k" in base.columns:
        base["nbm_native_tmax_2m_day_max_f"] = kelvin_to_f(_numeric_column(base, "nbm_native_tmax_2m_day_max_k"))
    if "lamp_tmax_open_f" not in base.columns:
        base["lamp_tmax_open_f"] = _numeric_column(
            base,
            "lamp_day_temp_max_f",
            "lamp_day_tmp_max_f_forecast",
        )
    if "anchor_tmax_f" not in base.columns:
        base["anchor_tmax_f"] = 0.5 * _numeric_column(base, "nbm_tmax_open_f") + 0.5 * _numeric_column(base, "lamp_tmax_open_f")
    if "nbm_minus_lamp_tmax_f" not in base.columns:
        base["nbm_minus_lamp_tmax_f"] = _numeric_column(base, "nbm_tmax_open_f") - _numeric_column(base, "lamp_tmax_open_f")
    return base


def _prepare_hrrr_disagreement_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "hrrr_tmax_open_f" not in out.columns:
        if "hrrr_temp_2m_day_max_f" in out.columns:
            out["hrrr_tmax_open_f"] = _numeric_column(out, "hrrr_temp_2m_day_max_f")
        else:
            out["hrrr_tmax_open_f"] = kelvin_to_f(_numeric_column(out, "hrrr_temp_2m_day_max_k"))

    hrrr = _numeric_column(out, "hrrr_tmax_open_f")
    lamp = _numeric_column(out, "lamp_tmax_open_f")
    nbm = _numeric_column(out, "nbm_tmax_open_f")
    out["hrrr_minus_lamp_tmax_f"] = hrrr - lamp
    out["hrrr_minus_nbm_tmax_f"] = hrrr - nbm
    out["abs_hrrr_minus_lamp_tmax_f"] = out["hrrr_minus_lamp_tmax_f"].abs()
    out["abs_hrrr_minus_nbm_tmax_f"] = out["hrrr_minus_nbm_tmax_f"].abs()
    out["anchor_equal_3way_tmax_f"] = (hrrr + lamp + nbm) / 3.0

    lower_guidance = pd.concat([lamp, nbm], axis=1).min(axis=1)
    upper_guidance = pd.concat([lamp, nbm], axis=1).max(axis=1)
    out["hrrr_above_nbm_lamp_range_f"] = (hrrr - upper_guidance).clip(lower=0.0)
    out["hrrr_below_nbm_lamp_range_f"] = (lower_guidance - hrrr).clip(lower=0.0)
    out["hrrr_outside_nbm_lamp_range_f"] = out["hrrr_above_nbm_lamp_range_f"] + out["hrrr_below_nbm_lamp_range_f"]

    out["hrrr_hotter_than_lamp_3f"] = (out["hrrr_minus_lamp_tmax_f"] >= 3.0).astype("boolean")
    out["hrrr_colder_than_lamp_3f"] = (out["hrrr_minus_lamp_tmax_f"] <= -3.0).astype("boolean")
    out["hrrr_hotter_than_nbm_3f"] = (out["hrrr_minus_nbm_tmax_f"] >= 3.0).astype("boolean")
    out["hrrr_colder_than_nbm_3f"] = (out["hrrr_minus_nbm_tmax_f"] <= -3.0).astype("boolean")
    out = add_source_disagreement_features(out)
    return add_source_trust_features(out)


def build_training_table(
    base: pd.DataFrame,
    hrrr: pd.DataFrame | None = None,
    *,
    anchor_policy: str = DEFAULT_ANCHOR_POLICY,
    ridge_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    if base.empty:
        raise ValueError("base normalized training table is empty")
    base = _prepare_base_columns(base)
    required_base = {
        "target_date_local",
        "station_id",
        "label_final_tmax_f",
        "anchor_tmax_f",
    }
    missing = sorted(required_base - set(base.columns))
    if missing:
        raise ValueError(f"base table missing required columns: {missing}")

    work = base
    work["target_date_local"] = work["target_date_local"].astype(str)
    if not any(column.startswith("hrrr_") for column in work.columns):
        if hrrr is None:
            raise ValueError("base table has no HRRR columns and no HRRR summaries were provided")
        hrrr_join = transform_hrrr_for_join(hrrr)
        merged = work.merge(hrrr_join, on="target_date_local", how="left")
    else:
        merged = work.copy()
        if "meta_hrrr_available" not in merged.columns:
            merged["meta_hrrr_available"] = True
    merged["meta_hrrr_available"] = merged["meta_hrrr_available"].astype("boolean").fillna(False)
    merged = _prepare_hrrr_disagreement_columns(merged)
    merged = apply_anchor_policy(merged, anchor_policy=anchor_policy, ridge_metadata=ridge_metadata)

    label = pd.to_numeric(merged["label_final_tmax_f"], errors="coerce")
    anchor = pd.to_numeric(merged["anchor_tmax_f"], errors="coerce")
    residual = label - anchor
    existing_residual = pd.to_numeric(merged.get("target_residual_f"), errors="coerce")
    if "target_residual_f" in merged.columns and anchor_policy == "current_50_50":
        mismatch = (existing_residual - residual).abs() > 1e-6
        if bool(mismatch.fillna(False).any()):
            raise ValueError(f"target_residual_f formula mismatch rows: {int(mismatch.sum())}")
    else:
        merged["target_residual_f"] = residual

    required_sources = (
        _bool_column(merged, "meta_nbm_available", default=True)
        & _bool_column(merged, "meta_lamp_available", default=True)
        & _bool_column(merged, "meta_hrrr_available", default=False)
    )
    finite_core = (
        pd.to_numeric(merged["label_final_tmax_f"], errors="coerce").notna()
        & pd.to_numeric(merged["anchor_tmax_f"], errors="coerce").notna()
        & pd.to_numeric(merged["nbm_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(merged["lamp_tmax_open_f"], errors="coerce").notna()
        & pd.to_numeric(merged["hrrr_tmax_open_f"], errors="coerce").notna()
    )
    if "model_training_eligible" in merged.columns:
        previous_eligible = merged["model_training_eligible"].astype("boolean").fillna(False)
        merged["model_training_eligible"] = previous_eligible & required_sources & finite_core
    else:
        merged["model_training_eligible"] = required_sources & finite_core
    return merged.reset_index(drop=True)


def prefix_counts(columns: list[str]) -> dict[str, int]:
    prefixes = ("label_", "meta_", "wu_", "nbm_", "lamp_", "hrrr_")
    counts: dict[str, int] = {}
    for prefix in prefixes:
        counts[prefix] = sum(1 for column in columns if column.startswith(prefix))
    counts["other"] = sum(1 for column in columns if not column.startswith(prefixes))
    return counts


def build_manifest(
    *,
    base_path: pathlib.Path,
    hrrr_root: pathlib.Path,
    output_path: pathlib.Path,
    df: pd.DataFrame,
    anchor_policy: str,
) -> dict[str, object]:
    dates = df["target_date_local"].astype(str)
    eligible = df["model_training_eligible"].astype("boolean").fillna(False)
    hrrr_available = df["meta_hrrr_available"].astype("boolean").fillna(False)
    return {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_path": str(base_path),
        "hrrr_root": str(hrrr_root),
        "output_path": str(output_path),
        "anchor_policy": anchor_policy,
        "anchor_columns": sorted(column for column in df.columns if column.startswith("anchor_") and column.endswith("_tmax_f")),
        "source_trust_feature_count": int(sum(1 for column in SOURCE_TRUST_FEATURE_COLUMNS if column in df.columns)),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "target_date_min": str(dates.min()) if len(df) else None,
        "target_date_max": str(dates.max()) if len(df) else None,
        "target_date_unique_count": int(dates.nunique()),
        "model_training_eligible_count": int(eligible.sum()),
        "hrrr_available_count": int(hrrr_available.sum()),
        "column_prefix_counts": prefix_counts(list(df.columns)),
    }


def main() -> int:
    args = parse_args()
    base = pd.read_parquet(args.base_path)
    hrrr = None if any(column.startswith("hrrr_") for column in base.columns) else read_hrrr_summaries(args.hrrr_root)
    output = build_training_table(base, hrrr, anchor_policy=args.anchor_policy)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_path, index=False)
    manifest = build_manifest(
        base_path=args.base_path,
        hrrr_root=args.hrrr_root,
        output_path=args.output_path,
        df=output,
        anchor_policy=args.anchor_policy,
    )
    args.manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output_path)
    print(args.manifest_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
