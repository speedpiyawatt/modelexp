from __future__ import annotations

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


def expected_temperature(ladder: pd.DataFrame) -> float:
    return float((pd.to_numeric(ladder["temp_f"]) * pd.to_numeric(ladder["probability"])).sum())
