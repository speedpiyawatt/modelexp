from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


EPSILON = 1e-12
HIGH_RISK_REGIMES = {
    "broad_disagreement",
    "native_warm_hrrr_cold",
    "native_cold_hrrr_warm",
    "hrrr_hot_outlier",
    "hrrr_cold_outlier",
}


def risk_level(regime: str | None, spread_f: float | None) -> str:
    if regime is None or str(regime) == "unknown":
        return "unknown"
    try:
        spread = float(spread_f) if spread_f is not None else math.nan
    except (TypeError, ValueError):
        spread = math.nan
    if math.isfinite(spread) and spread >= 6.0:
        return "very_high"
    if str(regime) in HIGH_RISK_REGIMES:
        return "high"
    if str(regime) == "moderate_disagreement":
        return "medium"
    if str(regime) == "tight_consensus":
        return "low"
    return "unknown"


def with_hrrr_weight(candidate_id: str, *, regime: str | None, risk: str | None) -> float:
    regime = str(regime or "unknown")
    risk = str(risk or "unknown")
    if candidate_id == "always_no_hrrr":
        return 0.0
    if candidate_id == "always_with_hrrr":
        return 1.0
    if candidate_id == "with_hrrr_except_native_cold_hrrr_warm":
        return 0.0 if regime == "native_cold_hrrr_warm" else 1.0
    if candidate_id == "with_hrrr_only_high_or_very_high_disagreement":
        return 1.0 if risk in {"high", "very_high"} else 0.0
    if candidate_id in {"probability_blend_by_regime", "expected_tmax_blend_by_regime"}:
        return {
            "native_cold_hrrr_warm": 0.25,
            "hrrr_cold_outlier": 0.25,
            "hrrr_hot_outlier": 0.65,
            "native_warm_hrrr_cold": 0.65,
            "broad_disagreement": 0.50,
            "moderate_disagreement": 0.50,
            "tight_consensus": 0.35,
        }.get(regime, 0.50)
    raise ValueError(f"unknown dual guard candidate_id: {candidate_id}")


def blend_probability(no_hrrr_probability: float, with_hrrr_probability: float, *, weight: float) -> float:
    weight = min(1.0, max(0.0, float(weight)))
    return (1.0 - weight) * float(no_hrrr_probability) + weight * float(with_hrrr_probability)


def blend_expected(no_hrrr_expected_f: float, with_hrrr_expected_f: float, *, weight: float) -> float:
    return blend_probability(no_hrrr_expected_f, with_hrrr_expected_f, weight=weight)


def normalize_event_bins(event_bins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(float(row.get("probability", 0.0)) for row in event_bins)
    if total <= 0.0:
        return event_bins
    return [{**row, "probability": float(row.get("probability", 0.0)) / total} for row in event_bins]


def blend_event_bins(
    no_hrrr_bins: list[dict[str, Any]],
    with_hrrr_bins: list[dict[str, Any]],
    *,
    weight: float,
) -> list[dict[str, Any]]:
    no_by_label = {str(row["bin"]): float(row["probability"]) for row in no_hrrr_bins}
    with_by_label = {str(row["bin"]): float(row["probability"]) for row in with_hrrr_bins}
    rows = []
    for label, no_probability in no_by_label.items():
        if label not in with_by_label:
            continue
        rows.append(
            {
                "bin": label,
                "probability": blend_probability(no_probability, with_by_label[label], weight=weight),
            }
        )
    return normalize_event_bins(rows)


def apply_guard_to_prediction_payloads(
    no_hrrr: Mapping[str, Any],
    with_hrrr: Mapping[str, Any],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    disagreement = with_hrrr.get("source_disagreement")
    if not isinstance(disagreement, Mapping):
        disagreement = {}
    regime = str(disagreement.get("source_disagreement_regime") or "unknown")
    spread = disagreement.get("source_spread_f")
    risk = str(disagreement.get("source_disagreement_risk_level") or risk_level(regime, spread))
    weight = with_hrrr_weight(candidate_id, regime=regime, risk=risk)
    if candidate_id == "expected_tmax_blend_by_regime":
        probability_weight = 1.0 if regime != "native_cold_hrrr_warm" else 0.0
        expected_weight = weight
    else:
        probability_weight = weight
        expected_weight = weight
    event_bins = blend_event_bins(
        list(no_hrrr.get("event_bins", [])),
        list(with_hrrr.get("event_bins", [])),
        weight=probability_weight,
    )
    return {
        "candidate_id": candidate_id,
        "source_disagreement_regime": regime,
        "source_disagreement_risk_level": risk,
        "with_hrrr_probability_weight": probability_weight,
        "with_hrrr_expected_weight": expected_weight,
        "expected_final_tmax_f": blend_expected(
            float(no_hrrr["expected_final_tmax_f"]),
            float(with_hrrr["expected_final_tmax_f"]),
            weight=expected_weight,
        ),
        "anchor_tmax_f": blend_expected(
            float(no_hrrr.get("anchor_tmax_f", 0.0)),
            float(with_hrrr.get("anchor_tmax_f", 0.0)),
            weight=expected_weight,
        ),
        "event_bins": event_bins,
    }
