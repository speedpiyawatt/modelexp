from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_STATION_ID = "KLGA"
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"

IDENTITY_COLUMNS: tuple[str, ...] = (
    "target_date_local",
    "station_id",
    "selection_cutoff_local",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    "final_tmax_f",
    "final_tmin_f",
    "nbm_tmax_open_f",
    "lamp_tmax_open_f",
    "anchor_tmax_f",
    "nbm_minus_lamp_tmax_f",
    "target_residual_f",
    "meta_nbm_available",
    "meta_lamp_available",
    "meta_wu_obs_available",
    "model_training_eligible",
)

FORBIDDEN_PREFIXES: tuple[str, ...] = ("hrrr_", "meta_hrrr_")


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def cutoff_timestamp(target_date_local: str, cutoff_local_time: str = DEFAULT_CUTOFF_LOCAL_TIME) -> pd.Timestamp:
    local_date = dt.date.fromisoformat(str(target_date_local))
    local_time = dt.time.fromisoformat(cutoff_local_time)
    return pd.Timestamp(dt.datetime.combine(local_date, local_time, tzinfo=NY_TZ))


def _as_local_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(NY_TZ)
    return timestamp.tz_convert(NY_TZ)


def audit_training_features(
    df: pd.DataFrame,
    *,
    cutoff_local_time: str = DEFAULT_CUTOFF_LOCAL_TIME,
    residual_tolerance: float = 1e-6,
) -> AuditResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df.empty:
        errors.append("training table is empty")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        errors.append(f"missing required columns: {missing}")

    forbidden = [column for column in df.columns if column.startswith(FORBIDDEN_PREFIXES)]
    if forbidden:
        errors.append(f"HRRR columns are not allowed in this experiment: {forbidden}")

    if all(column in df.columns for column in ("target_date_local", "station_id")):
        duplicate_count = int(df.duplicated(subset=["target_date_local", "station_id"]).sum())
        if duplicate_count:
            errors.append(f"duplicate target_date_local/station_id rows: {duplicate_count}")

    if "model_training_eligible" in df.columns:
        eligible = df["model_training_eligible"].astype("boolean").fillna(False)
    else:
        eligible = pd.Series([True] * len(df), index=df.index, dtype="boolean")

    for column in ("final_tmax_f",):
        if column in df.columns:
            missing_count = int(pd.to_numeric(df[column], errors="coerce").isna().sum())
            if missing_count:
                errors.append(f"{column} has {missing_count} missing or non-numeric values")

    for column in ("anchor_tmax_f", "target_residual_f"):
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            missing_count = int(values.loc[eligible].isna().sum())
            if missing_count:
                errors.append(f"{column} has {missing_count} missing or non-numeric values among model-eligible rows")

    if all(column in df.columns for column in ("target_residual_f", "final_tmax_f", "anchor_tmax_f")):
        expected = pd.to_numeric(df["final_tmax_f"], errors="coerce") - pd.to_numeric(df["anchor_tmax_f"], errors="coerce")
        actual = pd.to_numeric(df["target_residual_f"], errors="coerce")
        mismatch = ((actual - expected).abs() > residual_tolerance) & eligible
        mismatch = mismatch.fillna(True)
        if bool(mismatch.any()):
            errors.append(f"target_residual_f formula mismatch rows: {int(mismatch.sum())}")

    if all(column in df.columns for column in ("selection_cutoff_local", "target_date_local")):
        bad_rows = 0
        for _, row in df.iterrows():
            try:
                actual = _as_local_timestamp(row["selection_cutoff_local"])
                expected = cutoff_timestamp(str(row["target_date_local"]), cutoff_local_time)
            except Exception:
                bad_rows += 1
                continue
            if actual != expected:
                bad_rows += 1
        if bad_rows:
            errors.append(f"selection_cutoff_local is not {cutoff_local_time} America/New_York for {bad_rows} rows")

    if "meta_lamp_available" in df.columns and not bool(df["meta_lamp_available"].astype("boolean").fillna(False).any()):
        warnings.append("no rows have LAMP available")
    if "meta_nbm_available" in df.columns and not bool(df["meta_nbm_available"].astype("boolean").fillna(False).any()):
        warnings.append("no rows have NBM available")

    return AuditResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings))


def numeric_or_nan(value: object) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if not pd.isna(numeric) else float("nan")


def kelvin_to_fahrenheit(value: object) -> float:
    numeric = numeric_or_nan(value)
    return (numeric - 273.15) * 9.0 / 5.0 + 32.0 if np.isfinite(numeric) else float("nan")


def mps_to_mph(value: object) -> float:
    numeric = numeric_or_nan(value)
    return numeric * 2.2369362921 if np.isfinite(numeric) else float("nan")


def knots_to_mph(value: object) -> float:
    numeric = numeric_or_nan(value)
    return numeric * 1.150779448 if np.isfinite(numeric) else float("nan")
