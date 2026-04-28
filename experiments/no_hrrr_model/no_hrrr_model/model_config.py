from __future__ import annotations


DEFAULT_QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
RANDOM_SEED = 20260425
DEFAULT_MODEL_CANDIDATE_ID = "very_regularized_lgbm_350"

DEFAULT_CANDIDATES: tuple[dict[str, object], ...] = (
    {
        "candidate_id": "current_lgbm_fixed_250_no_inner_es",
        "description": "Current no-HRRR LightGBM residual quantile parameters scored with fixed 250 rounds and no inner early stopping.",
        "params": {
            "learning_rate": 0.035,
            "num_leaves": 15,
            "min_data_in_leaf": 25,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
        },
        "num_boost_round": 250,
    },
    {
        "candidate_id": "regularized_shallow_lgbm_300",
        "description": "Shallower regularized LightGBM with smaller leaves, explicit max depth, L1/L2 penalties, and modest subsampling.",
        "params": {
            "learning_rate": 0.030,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 35,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "lambda_l1": 0.10,
            "lambda_l2": 3.0,
        },
        "num_boost_round": 300,
    },
    {
        "candidate_id": "regularized_moderate_lgbm_300",
        "description": "Moderate-depth regularized LightGBM preserving current leaf count with stronger leaf-size and L2 controls.",
        "params": {
            "learning_rate": 0.030,
            "num_leaves": 15,
            "max_depth": 4,
            "min_data_in_leaf": 35,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "lambda_l1": 0.05,
            "lambda_l2": 3.0,
        },
        "num_boost_round": 300,
    },
    {
        "candidate_id": "high_min_leaf_lgbm_250",
        "description": "Current learning-rate scale with higher minimum leaf size and stronger L2 regularization.",
        "params": {
            "learning_rate": 0.035,
            "num_leaves": 11,
            "max_depth": 4,
            "min_data_in_leaf": 60,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.05,
            "lambda_l2": 4.0,
        },
        "num_boost_round": 250,
    },
    {
        "candidate_id": "very_regularized_lgbm_350",
        "description": "Most conservative candidate with low learning rate, shallow trees, high leaf size, and strong L1/L2 penalties.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 7,
            "max_depth": 3,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.70,
            "bagging_fraction": 0.70,
            "bagging_freq": 1,
            "lambda_l1": 0.20,
            "lambda_l2": 5.0,
        },
        "num_boost_round": 350,
    },
    {
        "candidate_id": "slower_regularized_lgbm_350",
        "description": "Lower learning-rate candidate with moderate leaves, max-depth cap, high leaf size, and strong L2 penalty.",
        "params": {
            "learning_rate": 0.025,
            "num_leaves": 21,
            "max_depth": 5,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.75,
            "bagging_fraction": 0.75,
            "bagging_freq": 1,
            "lambda_l1": 0.05,
            "lambda_l2": 5.0,
        },
        "num_boost_round": 350,
    },
)


def candidate_by_id(candidate_id: str, candidates: list[dict[str, object]] | tuple[dict[str, object], ...] = DEFAULT_CANDIDATES) -> dict[str, object]:
    for candidate in candidates:
        if str(candidate["candidate_id"]) == candidate_id:
            return dict(candidate)
    raise ValueError(f"unknown candidate_id: {candidate_id}")


def selected_model_candidate() -> dict[str, object]:
    return candidate_by_id(DEFAULT_MODEL_CANDIDATE_ID)
