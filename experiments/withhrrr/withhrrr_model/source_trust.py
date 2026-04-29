from __future__ import annotations

import numpy as np
import pandas as pd

SOURCE_COLUMNS = {
    "nbm": "nbm_tmax_open_f",
    "native": "nbm_native_tmax_2m_day_max_f",
    "lamp": "lamp_tmax_open_f",
    "hrrr": "hrrr_tmax_open_f",
}

SOURCE_TRUST_FEATURE_COLUMNS = {
    "native_minus_lamp_tmax_f",
    "native_minus_nbm_tmax_f",
    "lamp_minus_hrrr_tmax_f",
    "abs_native_minus_lamp_tmax_f",
    "abs_native_minus_nbm_tmax_f",
    "abs_native_minus_hrrr_f",
    "abs_lamp_minus_hrrr_tmax_f",
    "source_trimmed_mean_tmax_f",
    "source_rank_nbm",
    "source_rank_native",
    "source_rank_lamp",
    "source_rank_hrrr",
    "source_nbm_warmest",
    "source_native_warmest",
    "source_lamp_warmest",
    "source_hrrr_warmest",
    "source_nbm_coldest",
    "source_native_coldest",
    "source_lamp_coldest",
    "source_hrrr_coldest",
    "wu_last_temp_minus_nbm_f",
    "wu_last_temp_minus_native_f",
    "wu_last_temp_minus_lamp_f",
    "wu_last_temp_minus_hrrr_f",
    "target_month",
    "target_dayofyear_sin",
    "target_dayofyear_cos",
}
SOURCE_TRUST_FEATURE_PROFILES = {
    "global_all_features",
    "source_trust_all_features",
    "high_disagreement_weighted",
    "native_warm_hrrr_cold_specialist",
    "native_cold_hrrr_warm_specialist",
    "hrrr_outlier_specialist",
}


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def source_value_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: _numeric(df, column) for name, column in SOURCE_COLUMNS.items()}, index=df.index)


def add_source_trust_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sources = source_value_frame(out)
    complete = sources.notna().all(axis=1)
    source_max = sources.max(axis=1)
    source_min = sources.min(axis=1)
    out["source_trimmed_mean_tmax_f"] = ((sources.sum(axis=1) - source_max - source_min) / 2.0).where(complete)

    out["native_minus_lamp_tmax_f"] = sources["native"] - sources["lamp"]
    out["native_minus_nbm_tmax_f"] = sources["native"] - sources["nbm"]
    out["lamp_minus_hrrr_tmax_f"] = sources["lamp"] - sources["hrrr"]
    out["abs_native_minus_lamp_tmax_f"] = out["native_minus_lamp_tmax_f"].abs()
    out["abs_native_minus_nbm_tmax_f"] = out["native_minus_nbm_tmax_f"].abs()
    out["abs_native_minus_hrrr_f"] = (sources["native"] - sources["hrrr"]).abs()
    out["abs_lamp_minus_hrrr_tmax_f"] = out["lamp_minus_hrrr_tmax_f"].abs()

    ranks = sources.rank(axis=1, method="min", ascending=True)
    for source_name in SOURCE_COLUMNS:
        out[f"source_rank_{source_name}"] = ranks[source_name].where(complete)
        out[f"source_{source_name}_warmest"] = (sources[source_name] == source_max).where(complete).astype("boolean")
        out[f"source_{source_name}_coldest"] = (sources[source_name] == source_min).where(complete).astype("boolean")

    wu_last_temp = _numeric(out, "wu_last_temp_f")
    out["wu_last_temp_minus_nbm_f"] = wu_last_temp - sources["nbm"]
    out["wu_last_temp_minus_native_f"] = wu_last_temp - sources["native"]
    out["wu_last_temp_minus_lamp_f"] = wu_last_temp - sources["lamp"]
    out["wu_last_temp_minus_hrrr_f"] = wu_last_temp - sources["hrrr"]

    if "target_date_local" in out.columns:
        dates = pd.to_datetime(out["target_date_local"], errors="coerce")
        month = dates.dt.month
        dayofyear = dates.dt.dayofyear
        out["target_month"] = month.astype("Float64")
        out["target_dayofyear_sin"] = np.sin(2.0 * np.pi * dayofyear / 366.0)
        out["target_dayofyear_cos"] = np.cos(2.0 * np.pi * dayofyear / 366.0)
    else:
        out["target_month"] = np.nan
        out["target_dayofyear_sin"] = np.nan
        out["target_dayofyear_cos"] = np.nan
    return out


def fixed_anchor_values(df: pd.DataFrame, anchor_policy: str) -> pd.Series:
    sources = source_value_frame(df)
    if anchor_policy == "current_50_50":
        return 0.5 * sources["nbm"] + 0.5 * sources["lamp"]
    if anchor_policy == "hourly_native_lamp":
        return (sources["nbm"] + sources["native"] + sources["lamp"]) / 3.0
    if anchor_policy == "hourly_native_lamp_hrrr":
        return (sources["nbm"] + sources["native"] + sources["lamp"] + sources["hrrr"]) / 4.0
    if anchor_policy == "native_lamp":
        return 0.5 * sources["native"] + 0.5 * sources["lamp"]
    if anchor_policy == "equal_3way":
        return (sources["nbm"] + sources["lamp"] + sources["hrrr"]) / 3.0
    if anchor_policy == "source_median_4way":
        return sources.median(axis=1).where(sources.notna().all(axis=1))
    if anchor_policy == "source_trimmed_mean_4way":
        complete = sources.notna().all(axis=1)
        return ((sources.sum(axis=1) - sources.max(axis=1) - sources.min(axis=1)) / 2.0).where(complete)
    raise ValueError(f"unknown fixed anchor policy: {anchor_policy}")


def ridge_design_matrix(df: pd.DataFrame) -> np.ndarray:
    sources = source_value_frame(df)
    return np.column_stack(
        [
            np.ones(len(df), dtype=float),
            sources["nbm"].to_numpy(float),
            sources["native"].to_numpy(float),
            sources["lamp"].to_numpy(float),
            sources["hrrr"].to_numpy(float),
        ]
    )


def fit_ridge_4way_anchor(df: pd.DataFrame, *, alpha: float = 5.0) -> dict[str, object]:
    x = ridge_design_matrix(df)
    y = pd.to_numeric(df["final_tmax_f"], errors="coerce").to_numpy(float)
    keep = np.isfinite(x).all(axis=1) & np.isfinite(y)
    if int(keep.sum()) < x.shape[1]:
        raise ValueError("not enough finite rows to fit ridge_4way_anchor")
    x_keep = x[keep]
    y_keep = y[keep]
    penalty = np.eye(x_keep.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x_keep.T @ x_keep + penalty, x_keep.T @ y_keep)
    return {
        "anchor_policy": "ridge_4way_anchor",
        "alpha": float(alpha),
        "row_count": int(keep.sum()),
        "terms": ["intercept", "nbm_tmax_open_f", "nbm_native_tmax_2m_day_max_f", "lamp_tmax_open_f", "hrrr_tmax_open_f"],
        "coefficients": [float(value) for value in coefficients],
    }


def predict_ridge_4way_anchor(df: pd.DataFrame, metadata: dict[str, object]) -> pd.Series:
    coefficients = np.asarray(metadata.get("coefficients", []), dtype=float)
    if coefficients.shape != (5,):
        raise ValueError("ridge_4way_anchor metadata must contain five coefficients")
    values = ridge_design_matrix(df) @ coefficients
    return pd.Series(values, index=df.index, dtype=float)


def apply_anchor_policy(
    df: pd.DataFrame,
    *,
    anchor_policy: str,
    ridge_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for policy in (
        "current_50_50",
        "hourly_native_lamp",
        "hourly_native_lamp_hrrr",
        "native_lamp",
        "equal_3way",
        "source_median_4way",
        "source_trimmed_mean_4way",
    ):
        out[f"anchor_{policy}_tmax_f"] = fixed_anchor_values(out, policy)
    if anchor_policy == "ridge_4way_anchor":
        if ridge_metadata is None:
            raise ValueError("ridge_4way_anchor requires ridge_metadata")
        out["anchor_ridge_4way_anchor_tmax_f"] = predict_ridge_4way_anchor(out, ridge_metadata)
    elif anchor_policy not in {
        "current_50_50",
        "hourly_native_lamp",
        "hourly_native_lamp_hrrr",
        "native_lamp",
        "equal_3way",
        "source_median_4way",
        "source_trimmed_mean_4way",
    }:
        raise ValueError(f"unknown anchor_policy={anchor_policy!r}")
    out["anchor_tmax_f"] = pd.to_numeric(out[f"anchor_{anchor_policy}_tmax_f"], errors="coerce")
    out["anchor_policy"] = anchor_policy
    native = _numeric(out, "nbm_native_tmax_2m_day_max_f")
    nbm = _numeric(out, "nbm_tmax_open_f")
    out["nbm_native_tmax_minus_anchor_f"] = native - pd.to_numeric(out["anchor_tmax_f"], errors="coerce")
    out["nbm_native_tmax_minus_nbm_tmax_f"] = native - nbm
    out["abs_nbm_native_tmax_minus_anchor_f"] = out["nbm_native_tmax_minus_anchor_f"].abs()
    out["abs_nbm_native_tmax_minus_nbm_tmax_f"] = out["nbm_native_tmax_minus_nbm_tmax_f"].abs()
    out["nbm_native_tmax_above_anchor_2f"] = (out["nbm_native_tmax_minus_anchor_f"] >= 2.0).astype("boolean")
    out["nbm_native_tmax_below_anchor_2f"] = (out["nbm_native_tmax_minus_anchor_f"] <= -2.0).astype("boolean")
    out["nbm_native_tmax_above_hourly_nbm_2f"] = (out["nbm_native_tmax_minus_nbm_tmax_f"] >= 2.0).astype("boolean")
    out["nbm_native_tmax_below_hourly_nbm_2f"] = (out["nbm_native_tmax_minus_nbm_tmax_f"] <= -2.0).astype("boolean")
    return out


def apply_anchor_and_residual(
    df: pd.DataFrame,
    *,
    anchor_policy: str,
    ridge_metadata: dict[str, object] | None = None,
) -> pd.DataFrame:
    out = apply_anchor_policy(df, anchor_policy=anchor_policy, ridge_metadata=ridge_metadata)
    out["target_residual_f"] = pd.to_numeric(out["final_tmax_f"], errors="coerce") - pd.to_numeric(out["anchor_tmax_f"], errors="coerce")
    return out


def high_disagreement_mask(df: pd.DataFrame) -> pd.Series:
    regimes = df.get("source_disagreement_regime", pd.Series("unknown", index=df.index)).astype(str)
    return regimes.isin(
        {
            "broad_disagreement",
            "native_warm_hrrr_cold",
            "native_cold_hrrr_warm",
            "hrrr_hot_outlier",
            "hrrr_cold_outlier",
        }
    )


def training_weights(df: pd.DataFrame, profile: str) -> pd.Series | None:
    if profile in {"none", "", "unweighted"}:
        return None
    weights = pd.Series(1.0, index=df.index, dtype=float)
    regimes = df.get("source_disagreement_regime", pd.Series("unknown", index=df.index)).astype(str)
    if profile == "high_disagreement_weighted":
        weights.loc[high_disagreement_mask(df)] = 2.0
    elif profile == "native_warm_hrrr_cold_specialist":
        weights.loc[regimes == "native_warm_hrrr_cold"] = 4.0
        weights.loc[high_disagreement_mask(df) & (regimes != "native_warm_hrrr_cold")] = 1.5
    elif profile == "native_cold_hrrr_warm_specialist":
        weights.loc[regimes == "native_cold_hrrr_warm"] = 4.0
        weights.loc[high_disagreement_mask(df) & (regimes != "native_cold_hrrr_warm")] = 1.5
    elif profile == "hrrr_outlier_specialist":
        weights.loc[regimes.isin(["hrrr_hot_outlier", "hrrr_cold_outlier"])] = 4.0
        weights.loc[high_disagreement_mask(df) & ~regimes.isin(["hrrr_hot_outlier", "hrrr_cold_outlier"])] = 1.5
    else:
        raise ValueError(f"unknown training weight profile: {profile}")
    return weights


def source_trust_feature_subset(feature_columns: list[str], profile: str) -> list[str]:
    if profile == "global_all_features":
        return [column for column in feature_columns if column not in SOURCE_TRUST_FEATURE_COLUMNS]
    if profile in SOURCE_TRUST_FEATURE_PROFILES:
        return list(feature_columns)
    raise ValueError(f"unknown feature_profile: {profile}")
