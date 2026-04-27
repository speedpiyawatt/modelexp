from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "tools" / "weather" / "build_training_features_overnight_normalized.py"
SOURCE_CONTRACT_PATH = ROOT / "tools" / "weather" / "training_features_overnight_contract.py"
NORMALIZED_CONTRACT_PATH = ROOT / "tools" / "weather" / "training_features_overnight_normalized_contract.py"
VOCAB_PATH = ROOT / "tools" / "weather" / "training_feature_vocabularies.json"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("training_features_overnight_normalized_builder_test", BUILD_PATH)
source_contract = load_module("training_features_overnight_source_contract_normalized_test", SOURCE_CONTRACT_PATH)
normalized_contract = load_module("training_features_overnight_normalized_contract_test", NORMALIZED_CONTRACT_PATH)


def build_source_row() -> dict[str, object]:
    row = {column: None for column in source_contract.registry_columns()}
    row.update(
        {
            "target_date_local": "2026-04-12",
            "station_id": "KLGA",
            "selection_cutoff_local": "2026-04-12T00:05:00-04:00",
            "label_final_tmax_f": 72.0,
            "label_final_tmin_f": 55.0,
            "label_market_bin": "72F",
            "label_obs_count": 25,
            "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
            "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
            "label_total_precip_in": 0.25,
            "meta_wu_obs_available": True,
            "meta_nbm_available": True,
            "meta_lamp_available": True,
            "meta_hrrr_available": True,
            "meta_nbm_coverage_complete": True,
            "meta_lamp_coverage_complete": False,
            "meta_hrrr_has_full_day_21_local_coverage": True,
            "wu_last_temp_f": 55.0,
            "wu_last_pressure_in": 30.12,
            "wu_last_visibility": 8.0,
            "wu_last_cloud_cover_code": "OVC",
            "wu_last_wx_phrase": "Haze",
            "nbm_temp_2m_day_max_k": 294.261111,
            "nbm_gust_10m_day_max_ms": 10.0,
            "nbm_apcp_day_total_kg_m2": 12.7,
            "nbm_visibility_day_min_m": 8046.72,
            "nbm_ceiling_morning_min_m": 914.4,
            "lamp_tmp_f_at_09": 60.0,
            "lamp_wsp_kt_at_09": 10.0,
            "lamp_cig_hundreds_ft_at_09": 8.0,
            "lamp_vis_miles_at_09": 2.0,
            "lamp_cld_code_at_09": "BKN",
            "lamp_obv_code_at_09": "HZ",
            "lamp_typ_code_at_09": "R",
            "lamp_morning_cld_mode": "SCT",
            "lamp_day_precip_type_mode": "R",
            "lamp_day_tmp_max_f_forecast": 68.0,
            "lamp_rev_day_tmp_max_f": 2.0,
            "hrrr_temp_2m_day_max_k": 293.261111,
            "hrrr_wind_10m_day_max_ms": 11.0,
            "hrrr_apcp_day_total_kg_m2": 25.4,
            "hrrr_temp_850mb_day_mean_k": 276.0,
            "hrrr_wind_850mb_speed_day_mean_ms": 15.0,
            "nbm_minus_lamp_day_max_f": 2.0,
            "nbm_minus_hrrr_day_max_k": 1.0,
        }
    )
    return row


def test_normalize_training_features_overnight_converts_units_and_encodes_categories():
    vocab = json.loads(VOCAB_PATH.read_text())
    source_df = pd.DataFrame.from_records([build_source_row()])

    normalized = builder.normalize_training_features_overnight(source_df, vocab)

    assert list(normalized.columns) == normalized_contract.registry_columns()
    row = normalized.iloc[0]
    assert row["nbm_temp_2m_day_max_f"] == pytest.approx(70.0, abs=1e-3)
    assert row["nbm_gust_10m_day_max_mph"] == pytest.approx(22.369, abs=1e-3)
    assert row["nbm_apcp_day_total_in"] == pytest.approx(0.5, abs=1e-6)
    assert row["nbm_visibility_day_min_mi"] == pytest.approx(5.0, abs=1e-6)
    assert row["nbm_ceiling_morning_min_ft"] == pytest.approx(3000.0, abs=1e-6)
    assert row["wu_last_visibility_mi"] == pytest.approx(8.0)
    assert row["wu_last_pressure_inhg"] == pytest.approx(30.12)
    assert row["wu_last_cloud_cover_id"] == vocab["cloud_cover"]["OVC"]
    assert row["wu_last_weather_family_id"] == vocab["weather_family"]["HAZE"]
    assert row["lamp_temp_09_local_f"] == pytest.approx(60.0)
    assert row["lamp_wind_speed_09_local_mph"] == pytest.approx(11.50779448, abs=1e-6)
    assert row["lamp_ceiling_09_local_ft"] == pytest.approx(800.0)
    assert row["lamp_cloud_cover_09_local_id"] == vocab["cloud_cover"]["BKN"]
    assert row["lamp_weather_09_local_id"] == vocab["weather_family"]["HAZE"]
    assert row["lamp_precip_type_09_local_id"] == vocab["precip_type"]["R"]
    assert row["lamp_morning_cloud_cover_mode_id"] == vocab["cloud_cover"]["SCT"]
    assert row["lamp_day_precip_type_mode_id"] == vocab["precip_type"]["R"]
    assert row["hrrr_temp_2m_day_max_f"] == pytest.approx(68.2, abs=1e-3)
    assert row["hrrr_wind_10m_day_max_mph"] == pytest.approx(24.606, abs=1e-3)
    assert row["hrrr_apcp_day_total_in"] == pytest.approx(1.0, abs=1e-6)
    assert row["hrrr_temp_850mb_day_mean_f"] == pytest.approx(37.13, abs=1e-2)
    assert row["hrrr_wind_850mb_speed_day_mean_mph"] == pytest.approx(33.554, abs=1e-3)
    assert row["nbm_minus_hrrr_day_max_f"] == pytest.approx(1.8, abs=1e-3)
    assert bool(row["meta_lamp_available"]) is True
    assert bool(row["meta_lamp_coverage_complete"]) is False


def test_normalize_training_features_overnight_uses_missing_and_unknown_buckets():
    vocab = json.loads(VOCAB_PATH.read_text())
    source_row = build_source_row()
    source_row["wu_last_wx_phrase"] = "Volcanic dust"
    source_row["lamp_cld_code_at_09"] = None
    source_df = pd.DataFrame.from_records([source_row])

    normalized = builder.normalize_training_features_overnight(source_df, vocab)
    row = normalized.iloc[0]

    assert row["wu_last_weather_family_id"] == vocab["weather_family"]["__UNK__"]
    assert row["lamp_cloud_cover_09_local_id"] == vocab["cloud_cover"]["__MISSING__"]


def test_validate_merged_input_rejects_missing_empty_and_missing_columns(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        builder.validate_merged_input(pd.DataFrame(), input_path=tmp_path / "missing.parquet", allow_empty=False)

    empty_path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=source_contract.registry_columns()).to_parquet(empty_path)
    with pytest.raises(ValueError):
        builder.validate_merged_input(pd.DataFrame(columns=source_contract.registry_columns()), input_path=empty_path, allow_empty=False)

    bad_df = pd.DataFrame.from_records([{"target_date_local": "2026-04-12"}])
    bad_path = tmp_path / "bad.parquet"
    bad_df.to_parquet(bad_path)
    with pytest.raises(ValueError):
        builder.validate_merged_input(bad_df, input_path=bad_path, allow_empty=True)


def test_validate_date_output_accepts_valid_daily_shard(tmp_path: pathlib.Path):
    vocab = json.loads(VOCAB_PATH.read_text())
    source_df = pd.DataFrame.from_records([build_source_row()])
    output_df = builder.normalize_training_features_overnight(source_df, vocab)
    output_path, manifest_path = builder.output_paths_for_date(tmp_path, "2026-04-12")
    manifest = builder.build_date_manifest(
        target_date_local="2026-04-12",
        output_path=output_path,
        output_df=output_df,
        vocabularies=vocab,
    )
    builder._write_atomic_parquet(output_df, output_path)
    builder._write_atomic_json(manifest_path, manifest)

    assert builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local="2026-04-12",
        allow_empty=False,
    )


def test_validate_date_output_rejects_row_count_mismatch(tmp_path: pathlib.Path):
    output_df = pd.DataFrame(columns=normalized_contract.registry_columns())
    output_path, manifest_path = builder.output_paths_for_date(tmp_path, "2026-04-12")
    builder._write_atomic_parquet(output_df, output_path)
    builder._write_atomic_json(
        manifest_path,
        {
            "status": "ok",
            "normalization_version": builder.NORMALIZATION_VERSION,
            "target_date_local": "2026-04-12",
            "row_count": 3,
            "column_count": len(output_df.columns),
        },
    )

    assert not builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local="2026-04-12",
        allow_empty=True,
    )
