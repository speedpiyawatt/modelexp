from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "tools" / "weather" / "audit_training_features_overnight.py"
CONTRACT_PATH = ROOT / "tools" / "weather" / "training_features_overnight_contract.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_module = load_module("training_features_overnight_audit_test", AUDIT_PATH)
contract = load_module("training_features_overnight_contract_audit_test", CONTRACT_PATH)


def test_build_audit_accepts_registry_conformant_table():
    row = {column: None for column in contract.registry_columns()}
    row["target_date_local"] = "2026-04-12"
    row["station_id"] = "KLGA"
    row["selection_cutoff_local"] = "2026-04-12T00:05:00-04:00"
    row["label_final_tmax_f"] = 72.0
    row["label_final_tmin_f"] = 55.0
    row["label_market_bin"] = "72F"
    row["label_obs_count"] = 25
    row["label_first_obs_time_local"] = "2026-04-12T00:00:00-04:00"
    row["label_last_obs_time_local"] = "2026-04-12T23:00:00-04:00"
    row["label_total_precip_in"] = 0.0
    row["meta_wu_obs_available"] = True
    row["meta_nbm_available"] = True
    row["meta_lamp_available"] = False
    row["meta_hrrr_available"] = True
    row["wu_last_temp_f"] = 55.0
    row["nbm_temp_2m_day_max_k"] = 294.0
    row["hrrr_temp_2m_day_max_k"] = 293.0
    row["nbm_minus_hrrr_day_max_k"] = 1.0

    df = pd.DataFrame.from_records([row])
    summary, metrics = audit_module.build_audit(df)

    assert summary["checks"]["core_columns_present"] is True
    assert summary["checks"]["registry_match"] is True
    assert summary["checks"]["prefix_policy_ok"] is True
    assert summary["checks"]["unique_target_date_station_key"] is True
    candidate_row = metrics.loc[metrics["column_name"] == "nbm_minus_hrrr_day_max_k"].iloc[0]
    assert candidate_row["freeze_level"] == "optional_candidate"
    assert candidate_row["availability_rate"] == 1.0


def test_build_audit_flags_prefix_and_key_violations():
    df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "selection_cutoff_local": "2026-04-12T00:05:00-04:00",
                "bad_column": 1,
            },
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "selection_cutoff_local": "2026-04-12T00:05:00-04:00",
                "bad_column": 2,
            },
        ]
    )

    summary, _ = audit_module.build_audit(df)

    assert summary["checks"]["prefix_policy_ok"] is False
    assert summary["checks"]["unique_target_date_station_key"] is False
    assert "bad_column" in summary["extra_columns"]


def test_build_audit_rejects_zero_row_outputs():
    df = pd.DataFrame(columns=contract.registry_columns())
    summary, _ = audit_module.build_audit(df)
    assert summary["checks"]["non_empty_table"] is False
