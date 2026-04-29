from __future__ import annotations

from experiments.no_hrrr_model.no_hrrr_model.model_config import (  # noqa: F401
    DEFAULT_CANDIDATES as NO_HRRR_DEFAULT_CANDIDATES,
    DEFAULT_QUANTILES,
    RANDOM_SEED,
)


DEFAULT_MODEL_CANDIDATE_ID = "very_regularized_min_leaf70_lgbm_350"
HRRR_CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "candidate_id": "hrrr_shallow_stronger_l2_lgbm_350",
        "description": "HRRR-aware shallow model with the selected shallow structure, more rounds, and stronger L2 shrinkage for the wider feature set.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 35,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "lambda_l1": 0.10,
            "lambda_l2": 6.0,
        },
        "num_boost_round": 350,
    },
    {
        "candidate_id": "hrrr_shallow_min_leaf45_lgbm_350",
        "description": "HRRR-aware shallow model with slightly higher leaf size to damp sparse HRRR disagreement regimes.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 45,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "lambda_l1": 0.15,
            "lambda_l2": 5.0,
        },
        "num_boost_round": 350,
    },
    {
        "candidate_id": "hrrr_medium_min_leaf60_lgbm_300",
        "description": "HRRR-aware medium candidate allowing limited interaction capacity while regularizing leaves and feature subsampling.",
        "params": {
            "learning_rate": 0.030,
            "num_leaves": 11,
            "max_depth": 4,
            "min_data_in_leaf": 60,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "lambda_l1": 0.20,
            "lambda_l2": 6.0,
        },
        "num_boost_round": 300,
    },
)
NEARBY_TUNING_CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "candidate_id": "nearby_vreg_leaf90_lgbm_400",
        "description": "Focused nearby-station tuning around the current very-regularized winner with larger leaf size for the wider feature set.",
        "params": {
            "learning_rate": 0.022,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 90,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.25,
            "lambda_l2": 7.0,
        },
        "num_boost_round": 400,
    },
    {
        "candidate_id": "nearby_vreg_leaf70_l2_8_lgbm_400",
        "description": "Focused nearby-station tuning with current leaf size, slower learning, and stronger L2.",
        "params": {
            "learning_rate": 0.022,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 70,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.25,
            "lambda_l2": 8.0,
        },
        "num_boost_round": 400,
    },
    {
        "candidate_id": "nearby_vreg_leaf80_frac060_lgbm_450",
        "description": "Focused nearby-station tuning with stronger feature subsampling and more rounds.",
        "params": {
            "learning_rate": 0.020,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 80,
            "feature_fraction": 0.60,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.25,
            "lambda_l2": 7.0,
        },
        "num_boost_round": 450,
    },
    {
        "candidate_id": "nearby_shallow_leaf70_leaves11_lgbm_350",
        "description": "Focused nearby-station tuning allowing slightly more interaction capacity while keeping high leaf regularization.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 11,
            "max_depth": 4,
            "min_data_in_leaf": 70,
            "feature_fraction": 0.65,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.20,
            "lambda_l2": 7.0,
        },
        "num_boost_round": 350,
    },
    {
        "candidate_id": "nearby_vreg_leaf100_lgbm_350",
        "description": "Focused nearby-station tuning with very high leaf size to resist sparse nearby interactions.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.25,
            "lambda_l2": 8.0,
        },
        "num_boost_round": 350,
    },
)
DEFAULT_CANDIDATES: tuple[dict[str, object], ...] = (*NO_HRRR_DEFAULT_CANDIDATES, *HRRR_CANDIDATES, *NEARBY_TUNING_CANDIDATES)


def candidate_by_id(
    candidate_id: str,
    candidates: list[dict[str, object]] | tuple[dict[str, object], ...] = DEFAULT_CANDIDATES,
) -> dict[str, object]:
    for candidate in candidates:
        if str(candidate["candidate_id"]) == candidate_id:
            return dict(candidate)
    raise ValueError(f"unknown candidate_id: {candidate_id}")


def selected_model_candidate() -> dict[str, object]:
    return candidate_by_id(DEFAULT_MODEL_CANDIDATE_ID)
