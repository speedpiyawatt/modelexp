from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
FILTER_PATH = ROOT / "tools" / "weather" / "filter_wu_training_tables.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wu_filter = load_module("weather_filter_wu_training_tables_test", FILTER_PATH)


def test_filter_wu_training_tables_keeps_month_spine_and_support_history():
    labels_df = pd.DataFrame.from_records(
        [
            {"target_date_local": "2025-03-31", "station_id": "KLGA", "label_final_tmax_f": 58.0},
            {"target_date_local": "2025-04-01", "station_id": "KLGA", "label_final_tmax_f": 61.0},
            {"target_date_local": "2025-04-02", "station_id": "KLGA", "label_final_tmax_f": 64.0},
            {"target_date_local": "2025-05-01", "station_id": "KLGA", "label_final_tmax_f": 70.0},
        ]
    )
    obs_df = pd.DataFrame.from_records(
        [
            {"station_id": "KLGA", "date_local": "2025-03-30", "valid_time_local": "2025-03-30T23:00:00-04:00"},
            {"station_id": "KLGA", "date_local": "2025-03-31", "valid_time_local": "2025-03-31T23:00:00-04:00"},
            {"station_id": "KLGA", "date_local": "2025-04-01", "valid_time_local": "2025-04-01T00:00:00-04:00"},
            {"station_id": "KLGA", "date_local": "2025-04-02", "valid_time_local": "2025-04-02T00:00:00-04:00"},
            {"station_id": "KLGA", "date_local": "2025-05-01", "valid_time_local": "2025-05-01T00:00:00-04:00"},
        ]
    )

    start_date = wu_filter.parse_local_date("2025-04-01")
    end_date = wu_filter.parse_local_date("2025-04-02")

    filtered_labels_df = wu_filter.filter_target_labels(labels_df, start_date=start_date, end_date=end_date)
    filtered_label_history_df = wu_filter.filter_label_history(labels_df, start_date=start_date, end_date=end_date)
    filtered_obs_df = wu_filter.filter_obs_support(obs_df, start_date=start_date, end_date=end_date)

    assert filtered_labels_df["target_date_local"].tolist() == ["2025-04-01", "2025-04-02"]
    assert filtered_label_history_df["target_date_local"].tolist() == ["2025-03-31", "2025-04-01", "2025-04-02"]
    assert filtered_obs_df["date_local"].tolist() == ["2025-03-31", "2025-04-01", "2025-04-02"]


def test_filter_wu_training_tables_manifest_reports_support_counts():
    start_date = wu_filter.parse_local_date("2025-04-01")
    end_date = wu_filter.parse_local_date("2025-04-30")

    manifest = wu_filter.build_manifest(
        start_date=start_date,
        end_date=end_date,
        labels_df=pd.DataFrame(index=range(30)),
        label_history_df=pd.DataFrame(index=range(31)),
        obs_df=pd.DataFrame(index=range(120)),
    )

    assert manifest["start_local_date"] == "2025-04-01"
    assert manifest["end_local_date"] == "2025-04-30"
    assert manifest["label_row_count"] == 30
    assert manifest["label_history_row_count"] == 31
    assert manifest["label_history_start_local_date"] == "2025-03-31"
