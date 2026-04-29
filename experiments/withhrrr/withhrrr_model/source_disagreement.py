from __future__ import annotations

import numpy as np
import pandas as pd


SOURCE_COLUMNS = {
    "nbm": "nbm_tmax_open_f",
    "native_nbm": "nbm_native_tmax_2m_day_max_f",
    "lamp": "lamp_tmax_open_f",
    "hrrr": "hrrr_tmax_open_f",
}

DISAGREEMENT_WIDENING_REGIMES = {
    "broad_disagreement",
    "native_warm_hrrr_cold",
    "native_cold_hrrr_warm",
    "hrrr_hot_outlier",
    "hrrr_cold_outlier",
}


def _numeric_source(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _source_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: _numeric_source(df, column) for name, column in SOURCE_COLUMNS.items()}, index=df.index)


def source_disagreement_features(df: pd.DataFrame) -> pd.DataFrame:
    sources = _source_frame(df)
    complete = sources.notna().all(axis=1)
    source_max = sources.max(axis=1)
    source_min = sources.min(axis=1)
    source_median = sources.median(axis=1)
    spread = source_max - source_min

    out = pd.DataFrame(index=df.index)
    out["source_spread_f"] = spread.where(complete)
    out["source_median_tmax_f"] = source_median.where(complete)
    out["warmest_source"] = sources.fillna(-np.inf).idxmax(axis=1).where(complete, "unknown").astype("string")
    out["coldest_source"] = sources.fillna(np.inf).idxmin(axis=1).where(complete, "unknown").astype("string")
    out["native_minus_hrrr_f"] = (sources["native_nbm"] - sources["hrrr"]).where(complete)
    out["hrrr_minus_source_median_f"] = (sources["hrrr"] - source_median).where(complete)
    out["native_minus_source_median_f"] = (sources["native_nbm"] - source_median).where(complete)

    native_warm_hrrr_cold = (
        complete
        & (out["native_minus_hrrr_f"] >= 3.0)
        & (out["native_minus_source_median_f"] >= 1.0)
        & (out["hrrr_minus_source_median_f"] <= -1.0)
    )
    native_cold_hrrr_warm = (
        complete
        & (out["native_minus_hrrr_f"] <= -3.0)
        & (out["native_minus_source_median_f"] <= -1.0)
        & (out["hrrr_minus_source_median_f"] >= 1.0)
    )
    other_sources = sources.drop(columns=["hrrr"])
    hrrr_hot_outlier = complete & ((sources["hrrr"] - other_sources.max(axis=1)) >= 3.0)
    hrrr_cold_outlier = complete & ((other_sources.min(axis=1) - sources["hrrr"]) >= 3.0)
    broad_disagreement = complete & (spread >= 4.0)
    moderate_disagreement = complete & (spread >= 2.0) & (spread < 4.0)
    tight_consensus = complete & (spread < 2.0)

    out["source_disagreement_regime"] = pd.Series(
        np.select(
            [
                native_warm_hrrr_cold,
                native_cold_hrrr_warm,
                hrrr_hot_outlier,
                hrrr_cold_outlier,
                broad_disagreement,
                moderate_disagreement,
                tight_consensus,
            ],
            [
                "native_warm_hrrr_cold",
                "native_cold_hrrr_warm",
                "hrrr_hot_outlier",
                "hrrr_cold_outlier",
                "broad_disagreement",
                "moderate_disagreement",
                "tight_consensus",
            ],
            default="unknown",
        ),
        index=df.index,
        dtype="string",
    )
    return out


def add_source_disagreement_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    features = source_disagreement_features(out)
    for column in features.columns:
        out[column] = features[column]
    return out


def source_disagreement_regime_series(df: pd.DataFrame) -> pd.Series:
    if "source_disagreement_regime" in df.columns:
        return df["source_disagreement_regime"].fillna("unknown").astype("string")
    return source_disagreement_features(df)["source_disagreement_regime"]
