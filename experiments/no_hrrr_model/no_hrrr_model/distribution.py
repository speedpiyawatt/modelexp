from __future__ import annotations

import math

import numpy as np
import pandas as pd


def final_tmax_quantiles(anchor_tmax_f: float, residual_quantiles: dict[float, float]) -> dict[float, float]:
    return {float(q): float(anchor_tmax_f) + float(value) for q, value in sorted(residual_quantiles.items())}


def quantiles_to_degree_ladder(
    quantiles: dict[float, float],
    *,
    min_temp_f: int | None = None,
    max_temp_f: int | None = None,
    rearrange_crossing: bool = True,
    allocate_tail_mass: bool = True,
) -> pd.DataFrame:
    if not quantiles:
        raise ValueError("quantiles must be non-empty")
    qs = np.array(sorted(float(q) for q in quantiles), dtype=float)
    values = np.array([float(quantiles[q]) for q in qs], dtype=float)
    if np.any(~np.isfinite(qs)) or np.any(~np.isfinite(values)):
        raise ValueError("quantile levels and values must be finite")
    if np.any((qs <= 0.0) | (qs >= 1.0)):
        raise ValueError("quantile levels must be between 0 and 1")
    if np.any(np.diff(values) < 0):
        if not rearrange_crossing:
            raise ValueError("quantile values must be monotone nondecreasing")
        values = np.maximum.accumulate(values)
    if min_temp_f is None:
        min_temp_f = int(np.floor(np.nanmin(values) - 8.0))
    if max_temp_f is None:
        max_temp_f = int(np.ceil(np.nanmax(values) + 8.0))
    if max_temp_f < min_temp_f:
        raise ValueError("max_temp_f must be on or above min_temp_f")
    degrees = np.arange(min_temp_f, max_temp_f + 1, dtype=int)
    if np.nanmax(values) == np.nanmin(values):
        probs = np.zeros(len(degrees), dtype=float)
        point = int(round(float(values[0])))
        if point < min_temp_f:
            probs[0] = 1.0
        elif point > max_temp_f:
            probs[-1] = 1.0
        else:
            probs[degrees == point] = 1.0
        return pd.DataFrame({"temp_f": degrees, "probability": probs})
    unique_values, inverse = np.unique(values, return_inverse=True)
    unique_qs = np.zeros(len(unique_values), dtype=float)
    for index, q in zip(inverse, qs):
        unique_qs[index] = max(unique_qs[index], q)
    upper_edges = degrees + 0.5
    cdf_upper = np.interp(upper_edges, unique_values, unique_qs, left=unique_qs[0], right=unique_qs[-1])
    lower_edges = degrees - 0.5
    cdf_lower = np.interp(lower_edges, unique_values, unique_qs, left=unique_qs[0], right=unique_qs[-1])
    probs = np.maximum(cdf_upper - cdf_lower, 0.0)
    if allocate_tail_mass:
        lower_tail = degrees < unique_values[0]
        upper_tail = degrees > unique_values[-1]
        if not bool(lower_tail.any()):
            lower_tail = degrees == degrees[0]
        if not bool(upper_tail.any()):
            upper_tail = degrees == degrees[-1]
        probs[lower_tail] += unique_qs[0] / float(lower_tail.sum())
        probs[upper_tail] += (1.0 - unique_qs[-1]) / float(upper_tail.sum())
    total = probs.sum()
    if total <= 0:
        probs[:] = 1.0 / len(probs)
    else:
        probs = probs / total
    return pd.DataFrame({"temp_f": degrees, "probability": probs})


def smoothed_degree_ladder(ladder: pd.DataFrame) -> pd.DataFrame:
    out = ladder.copy()
    probabilities = pd.to_numeric(out["probability"], errors="coerce").fillna(0.0).to_numpy(float)
    if len(probabilities) > 1:
        smoothed = np.convolve(probabilities, np.array([0.25, 0.50, 0.25]), mode="same")
        smoothed[0] += 0.25 * probabilities[0]
        smoothed[-1] += 0.25 * probabilities[-1]
        total = smoothed.sum()
        if total > 0.0:
            probabilities = smoothed / total
    out["probability"] = probabilities
    return out


def normal_iqr_degree_ladder(
    quantiles: dict[float, float],
    *,
    min_temp_f: int | None = None,
    max_temp_f: int | None = None,
) -> pd.DataFrame:
    if not quantiles:
        raise ValueError("quantiles must be non-empty")
    q = {float(key): float(value) for key, value in quantiles.items()}
    required = [0.05, 0.25, 0.5, 0.75, 0.95]
    missing = [key for key in required if key not in q]
    if missing:
        raise ValueError(f"normal_iqr distribution requires quantiles: {missing}")
    values = np.asarray([q[key] for key in sorted(q)], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("quantile values must be finite")
    values = np.maximum.accumulate(values)
    if min_temp_f is None:
        min_temp_f = int(np.floor(np.nanmin(values) - 8.0))
    if max_temp_f is None:
        max_temp_f = int(np.ceil(np.nanmax(values) + 8.0))
    if max_temp_f < min_temp_f:
        raise ValueError("max_temp_f must be on or above min_temp_f")
    degrees = np.arange(min_temp_f, max_temp_f + 1, dtype=int)
    q05, q25, q50, q75, q95 = q[0.05], q[0.25], q[0.5], q[0.75], q[0.95]
    scales = [
        (q75 - q25) / 1.3489795003921634,
        (q95 - q05) / 3.289707253902945,
    ]
    finite_scales = [value for value in scales if np.isfinite(value) and value > 0.0]
    sigma = max(float(np.nanmedian(finite_scales)), 0.75) if finite_scales else 0.75
    edges = np.concatenate([[degrees[0] - 0.5], degrees + 0.5])
    z = (edges - q50) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    probabilities = np.maximum(np.diff(cdf), 0.0)
    total = probabilities.sum()
    if total > 0.0:
        probabilities = probabilities / total
    else:
        probabilities[:] = 1.0 / len(probabilities)
    return pd.DataFrame({"temp_f": degrees, "probability": probabilities})


def degree_ladder_from_quantiles(
    quantiles: dict[float, float],
    *,
    method_id: str = "interpolation_tail",
    min_temp_f: int | None = None,
    max_temp_f: int | None = None,
) -> pd.DataFrame:
    if method_id == "interpolation_tail":
        return quantiles_to_degree_ladder(quantiles, min_temp_f=min_temp_f, max_temp_f=max_temp_f, allocate_tail_mass=True)
    if method_id == "interpolation_no_tail":
        return quantiles_to_degree_ladder(quantiles, min_temp_f=min_temp_f, max_temp_f=max_temp_f, allocate_tail_mass=False)
    if method_id == "smoothed_interpolation_tail":
        return smoothed_degree_ladder(
            quantiles_to_degree_ladder(quantiles, min_temp_f=min_temp_f, max_temp_f=max_temp_f, allocate_tail_mass=True)
        )
    if method_id == "normal_iqr":
        return normal_iqr_degree_ladder(quantiles, min_temp_f=min_temp_f, max_temp_f=max_temp_f)
    raise ValueError(f"unknown distribution method_id: {method_id}")


def expected_temperature(ladder: pd.DataFrame) -> float:
    return float((pd.to_numeric(ladder["temp_f"]) * pd.to_numeric(ladder["probability"])).sum())
