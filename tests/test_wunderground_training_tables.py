from __future__ import annotations

import datetime as dt
import importlib.util
import json
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "wunderground" / "build_training_tables.py"
NY_TZ = ZoneInfo("America/New_York")


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


wu_tables = load_module("wu_training_tables_test", BUILD_PATH)


def epoch_seconds(local_time: str) -> int:
    timestamp = pd.Timestamp(local_time).tz_convert("UTC")
    return int(timestamp.timestamp())


def write_history_file(path: pathlib.Path, observations: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": {"station_id": "KLGA"}, "observations": observations}
    path.write_text(json.dumps(payload))


def test_build_wu_obs_intraday_and_labels_daily(tmp_path):
    history_dir = tmp_path / "history"
    write_history_file(
        history_dir / "KLGA_9_US_2026-04-11.json",
        [
            {"valid_time_gmt": epoch_seconds("2026-04-11T12:00:00-04:00"), "temp": 65, "dewPt": 50, "rh": 55, "pressure": 30.10, "wspd": 8, "wdir": 180, "gust": 12, "vis": 10, "clds": "SCT", "wx_phrase": "Fair", "precip_hrly": 0.0, "snow_hrly": 0.0},
            {"valid_time_gmt": epoch_seconds("2026-04-11T13:00:00-04:00"), "temp": 67, "dewPt": 51, "rh": 53, "pressure": 30.12, "wspd": 9, "wdir": 190, "gust": 13, "vis": 10, "clds": "SCT", "wx_phrase": "Fair", "precip_hrly": 0.0, "snow_hrly": 0.0},
            {"valid_time_gmt": epoch_seconds("2026-04-11T15:00:00-04:00"), "temp": 66, "dewPt": 52, "rh": 58, "pressure": 30.08, "wspd": 7, "wdir": 200, "gust": 11, "vis": 9, "clds": "BKN", "wx_phrase": "Cloudy", "precip_hrly": 0.1, "snow_hrly": 0.0},
        ],
    )
    write_history_file(
        history_dir / "KLGA_9_US_2026-04-12.json",
        [
            {"valid_time_gmt": epoch_seconds("2026-04-12T00:00:00-04:00"), "temp": 56, "dewPt": 49, "rh": 63, "pressure": 30.20, "wspd": 6, "wdir": 210, "gust": None, "vis": 8, "clds": "OVC", "wx_phrase": "Cloudy", "precip_hrly": 0.0, "snow_hrly": 0.0},
            {"valid_time_gmt": epoch_seconds("2026-04-12T01:00:00-04:00"), "temp": 57, "dewPt": 48, "rh": 60, "pressure": 30.22, "wspd": 5, "wdir": 220, "gust": None, "vis": 8, "clds": "OVC", "wx_phrase": "Cloudy", "precip_hrly": 0.0, "snow_hrly": 0.0},
        ],
    )

    obs_df = wu_tables.build_wu_obs_intraday(history_dir)
    labels_df = wu_tables.build_labels_daily(obs_df)

    assert list(obs_df["date_local"].unique()) == ["2026-04-11", "2026-04-12"]
    row_1300 = obs_df.loc[obs_df["valid_time_local"] == "2026-04-11T13:00:00-04:00"].iloc[0]
    assert row_1300["max_so_far_f"] == 67.0
    assert row_1300["warming_rate_1h_f"] == 2.0

    label_row = labels_df.loc[labels_df["target_date_local"] == "2026-04-11"].iloc[0]
    assert label_row["label_final_tmax_f"] == 67.0
    assert label_row["label_final_tmin_f"] == 65.0
    assert label_row["label_market_bin"] == "67F"
    assert label_row["label_total_precip_in"] == 0.1


def test_write_outputs_writes_expected_paths(tmp_path):
    labels_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-11",
                "station_id": "KLGA",
                "label_final_tmax_f": 67.0,
                "label_final_tmin_f": 65.0,
                "label_market_bin": "67F",
                "label_obs_count": 3,
                "label_first_obs_time_local": "2026-04-11T12:00:00-04:00",
                "label_last_obs_time_local": "2026-04-11T15:00:00-04:00",
                "label_total_precip_in": 0.1,
                "label_source_file": "KLGA_9_US_2026-04-11.json",
            }
        ]
    )
    obs_df = pd.DataFrame.from_records(
        [
            {
                "station_id": "KLGA",
                "source_file": "KLGA_9_US_2026-04-11.json",
                "valid_time_utc": "2026-04-11T16:00:00+00:00",
                "valid_time_local": "2026-04-11T12:00:00-04:00",
                "date_local": "2026-04-11",
                "temp_f": 65.0,
                "dewpoint_f": 50.0,
                "rh_pct": 55.0,
                "pressure_in": 30.10,
                "wind_speed_mph": 8.0,
                "wind_dir_deg": 180.0,
                "wind_gust_mph": 12.0,
                "visibility": 10.0,
                "cloud_cover_code": "SCT",
                "wx_phrase": "Fair",
                "precip_hrly_in": 0.0,
                "snow_hrly_in": 0.0,
                "max_so_far_f": 65.0,
                "warming_rate_1h_f": None,
                "warming_rate_3h_f": None,
            }
        ]
    )
    labels_path, obs_path = wu_tables.write_outputs(output_dir=tmp_path, labels_df=labels_df, obs_df=obs_df)
    assert labels_path.exists()
    assert obs_path.exists()


def test_build_training_tables_returns_labels_and_obs_from_history_dir(tmp_path):
    history_dir = tmp_path / "history"
    write_history_file(
        history_dir / "KLGA_9_US_2026-04-13.json",
        [
            {"valid_time_gmt": epoch_seconds("2026-04-13T00:00:00-04:00"), "temp": 55, "dewPt": 45, "rh": 60, "pressure": 30.05, "wspd": 5, "wdir": 180, "gust": 9, "vis": 10, "clds": "CLR", "wx_phrase": "Fair", "precip_hrly": 0.0, "snow_hrly": 0.0},
            {"valid_time_gmt": epoch_seconds("2026-04-13T13:00:00-04:00"), "temp": 68, "dewPt": 50, "rh": 55, "pressure": 30.02, "wspd": 8, "wdir": 200, "gust": 12, "vis": 10, "clds": "SCT", "wx_phrase": "Fair", "precip_hrly": 0.0, "snow_hrly": 0.0},
        ],
    )

    labels_df, obs_df = wu_tables.build_training_tables(history_dir)

    assert len(labels_df) == 1
    assert len(obs_df) == 2
    assert labels_df.iloc[0]["target_date_local"] == "2026-04-13"
    assert labels_df.iloc[0]["label_final_tmax_f"] == 68.0
