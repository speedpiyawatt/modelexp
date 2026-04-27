from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "tools" / "weather" / "build_training_features_overnight.py"
CONTRACT_PATH = ROOT / "tools" / "weather" / "training_features_overnight_contract.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("training_features_overnight_builder_test", BUILD_PATH)
contract = load_module("training_features_overnight_contract_test", CONTRACT_PATH)


def test_build_training_features_overnight_uses_label_spine_and_frozen_registry():
    labels_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-11",
                "station_id": "KLGA",
                "label_final_tmax_f": 67.0,
                "label_final_tmin_f": 49.0,
                "label_market_bin": "67F",
                "label_obs_count": 24,
                "label_first_obs_time_local": "2026-04-11T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-11T23:00:00-04:00",
                "label_total_precip_in": 0.1,
            },
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "label_final_tmax_f": 72.0,
                "label_final_tmin_f": 55.0,
                "label_market_bin": "72F",
                "label_obs_count": 25,
                "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
                "label_total_precip_in": 0.0,
            },
        ]
    )
    obs_df = pd.DataFrame.from_records(
        [
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-11T21:00:00-04:00",
                "temp_f": 58.0,
                "dewpoint_f": 48.0,
                "rh_pct": 65.0,
                "pressure_in": 30.10,
                "wind_speed_mph": 10.0,
                "wind_dir_deg": 200.0,
                "wind_gust_mph": 15.0,
                "visibility": 10.0,
                "cloud_cover_code": "BKN",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.0,
            },
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-11T23:00:00-04:00",
                "temp_f": 56.0,
                "dewpoint_f": 47.0,
                "rh_pct": 68.0,
                "pressure_in": 30.12,
                "wind_speed_mph": 8.0,
                "wind_dir_deg": 210.0,
                "wind_gust_mph": 12.0,
                "visibility": 9.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.0,
            },
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-12T00:00:00-04:00",
                "temp_f": 55.0,
                "dewpoint_f": 46.0,
                "rh_pct": 70.0,
                "pressure_in": 30.15,
                "wind_speed_mph": 7.0,
                "wind_dir_deg": 220.0,
                "wind_gust_mph": 10.0,
                "visibility": 8.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.1,
            },
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-12T00:10:00-04:00",
                "temp_f": 60.0,
                "dewpoint_f": 49.0,
                "rh_pct": 66.0,
                "pressure_in": 30.20,
                "wind_speed_mph": 6.0,
                "wind_dir_deg": 230.0,
                "wind_gust_mph": 9.0,
                "visibility": 8.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.0,
            },
        ]
    )
    nbm_daily_df = pd.DataFrame.from_records(
        [
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "target_date_local": "2026-04-12",
                "selected_init_time_utc": "2026-04-12T04:00:00+00:00",
                "selected_init_time_local": "2026-04-12T00:00:00-04:00",
                "selected_issue_age_minutes": 5.0,
                "target_day_row_count": 6,
                "missing_checkpoint_count": 0,
                "missing_required_feature_count": 0,
                "coverage_complete": True,
                "nbm_temp_2m_day_max_k": 294.261111,
                "nbm_temp_2m_day_mean_k": 289.0,
                "nbm_native_tmax_2m_day_max_k": 295.0,
                "nbm_native_tmax_2m_nb3_day_max_k": 295.4,
                "nbm_native_tmax_2m_nb7_day_max_k": 295.8,
                "nbm_native_tmax_2m_crop_day_max_k": 296.2,
                "nbm_native_tmin_2m_day_min_k": 274.0,
                "nbm_native_tmin_2m_nb3_day_min_k": 273.6,
                "nbm_native_tmin_2m_nb7_day_min_k": 273.2,
                "nbm_native_tmin_2m_crop_day_min_k": 272.8,
                "nbm_gust_10m_day_max_ms": 12.0,
                "nbm_tcdc_morning_mean_pct": 45.0,
                "nbm_tcdc_day_mean_pct": 42.0,
                "nbm_dswrf_day_max_w_m2": 500.0,
                "nbm_apcp_day_total_kg_m2": 0.1,
                "nbm_pcpdur_day_total_h": 3.0,
                "nbm_pcpdur_day_max_h": 2.0,
                "nbm_pcpdur_morning_total_h": 3.0,
                "nbm_visibility_day_min_m": 9000.0,
                "nbm_ceiling_morning_min_m": 900.0,
                "nbm_cape_day_max_j_kg": 50.0,
                "nbm_pwther_code_day_mode": 0.0,
                "nbm_pwther_nonzero_hour_count": 3,
                "nbm_pwther_any_flag": 1.0,
                "nbm_tstm_day_max_pct": 35.0,
                "nbm_tstm_day_mean_pct": 10.0,
                "nbm_tstm_any_flag": 1.0,
                "nbm_ptype_code_day_mode": 0.0,
                "nbm_ptype_nonzero_hour_count": 3,
                "nbm_ptype_any_flag": 1.0,
                "nbm_thunc_day_max_code": 2.0,
                "nbm_thunc_day_mean_code": 0.67,
                "nbm_thunc_nonzero_hour_count": 3,
                "nbm_vrate_day_max": 250.0,
                "nbm_vrate_day_mean": 166.7,
                "nbm_temp_2m_nb3_day_mean_k": 289.2,
                "nbm_temp_2m_nb7_day_mean_k": 289.4,
                "nbm_temp_2m_crop_day_mean_k": 289.8,
                "nbm_tcdc_crop_day_mean_pct": 47.0,
                "nbm_dswrf_crop_day_max_w_m2": 520.0,
                "nbm_wind_10m_speed_nb7_day_mean_ms": 8.5,
                "nbm_temp_2m_06_local_k": 281.0,
                "nbm_temp_2m_09_local_k": 285.0,
                "nbm_temp_2m_12_local_k": 289.0,
                "nbm_temp_2m_15_local_k": 294.261111,
                "nbm_temp_2m_18_local_k": 292.0,
                "nbm_temp_2m_21_local_k": 288.0,
                "nbm_dewpoint_2m_09_local_k": 279.0,
                "nbm_dewpoint_2m_15_local_k": 280.0,
                "nbm_rh_2m_09_local_pct": 55.0,
                "nbm_rh_2m_15_local_pct": 48.0,
                "nbm_wind_10m_speed_09_local_ms": 8.0,
                "nbm_wind_10m_speed_15_local_ms": 10.0,
                "nbm_wind_10m_direction_09_local_deg": 220.0,
                "nbm_wind_10m_direction_15_local_deg": 240.0,
            }
        ]
    )
    lamp_daily_df = pd.DataFrame.from_records(
        [
            {
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "station_id": "KLGA",
                "target_date_local": "2026-04-12",
                "selected_init_time_utc": "2026-04-12T04:00:00+00:00",
                "selected_init_time_local": "2026-04-12T00:00:00-04:00",
                "previous_init_time_utc": "2026-04-12T03:45:00+00:00",
                "previous_init_time_local": "2026-04-11T23:45:00-04:00",
                "revision_available": True,
                "missing_optional_any": False,
                "missing_optional_fields_count": 0,
                "coverage_complete": True,
                "missing_checkpoint_count": 0,
                "day_tmp_max_f_forecast": 68.0,
                "day_tmp_min_f_forecast": 54.0,
                "day_tmp_range_f_forecast": 14.0,
                "day_tmp_argmax_local_hour": 15,
                "morning_cld_mode": "BKN",
                "morning_cig_min_hundreds_ft": 8.0,
                "morning_vis_min_miles": 2.0,
                "morning_obv_any": True,
                "morning_ifr_like_any": True,
                "day_p01_max_pct": 20.0,
                "day_p06_max_pct": 30.0,
                "day_p12_max_pct": 40.0,
                "day_pos_max_pct": 30.0,
                "day_poz_max_pct": 12.0,
                "day_precip_type_any": True,
                "day_precip_type_mode": "R",
                "rev_day_tmp_max_f": 2.0,
                "rev_day_p01_max_pct": 5.0,
                "rev_day_pos_max_pct": 4.0,
                "rev_morning_cig_min_hundreds_ft": -2.0,
                "rev_morning_vis_min_miles": -1.0,
                "tmp_f_at_06": 54.0,
                "tmp_f_at_09": 60.0,
                "tmp_f_at_12": 65.0,
                "tmp_f_at_15": 68.0,
                "tmp_f_at_18": 66.0,
                "tmp_f_at_21": 61.0,
                "dpt_f_at_06": 46.0,
                "dpt_f_at_09": 47.0,
                "dpt_f_at_12": 50.0,
                "dpt_f_at_15": 52.0,
                "dpt_f_at_18": 51.0,
                "dpt_f_at_21": 49.0,
                "wsp_kt_at_06": 8.0,
                "wsp_kt_at_09": 10.0,
                "wsp_kt_at_12": 12.0,
                "wsp_kt_at_15": 13.0,
                "wsp_kt_at_18": 11.0,
                "wsp_kt_at_21": 9.0,
                "wdr_deg_at_06": 220.0,
                "wdr_deg_at_09": 230.0,
                "wdr_deg_at_12": 240.0,
                "wdr_deg_at_15": 250.0,
                "wdr_deg_at_18": 255.0,
                "wdr_deg_at_21": 245.0,
                "cld_code_at_06": "BKN",
                "cld_code_at_09": "OVC",
                "cld_code_at_12": "BKN",
                "cld_code_at_15": "SCT",
                "cld_code_at_18": "SCT",
                "cld_code_at_21": "FEW",
                "cig_hundreds_ft_at_06": 12.0,
                "cig_hundreds_ft_at_09": 8.0,
                "cig_hundreds_ft_at_12": 10.0,
                "cig_hundreds_ft_at_15": 20.0,
                "cig_hundreds_ft_at_18": 25.0,
                "cig_hundreds_ft_at_21": 30.0,
                "vis_miles_at_06": 6.0,
                "vis_miles_at_09": 2.0,
                "vis_miles_at_12": 4.0,
                "vis_miles_at_15": 6.0,
                "vis_miles_at_18": 6.0,
                "vis_miles_at_21": 6.0,
                "obv_code_at_06": None,
                "obv_code_at_09": "HZ",
                "obv_code_at_12": None,
                "obv_code_at_15": None,
                "obv_code_at_18": None,
                "obv_code_at_21": None,
                "typ_code_at_06": None,
                "typ_code_at_09": "R",
                "typ_code_at_12": "R",
                "typ_code_at_15": None,
                "typ_code_at_18": None,
                "typ_code_at_21": None,
                "rev_tmp_f_at_06": 1.0,
                "rev_tmp_f_at_09": 2.0,
                "rev_tmp_f_at_12": 2.0,
                "rev_tmp_f_at_15": 2.0,
                "rev_tmp_f_at_18": 1.0,
                "rev_tmp_f_at_21": 1.0,
            }
        ]
    )
    hrrr_daily_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-12",
                "anchor_init_time_utc": "2026-04-12T00:00:00+00:00",
                "anchor_init_time_local": "2026-04-11T20:00:00-04:00",
                "retained_cycle_count": 4,
                "first_valid_hour_local": 6,
                "last_valid_hour_local": 21,
                "covered_hour_count": 16,
                "covered_checkpoint_count": 6,
                "coverage_end_hour_local": 21,
                "has_full_day_21_local_coverage": True,
                "missing_checkpoint_count": 0,
                "hrrr_temp_2m_06_local_k": 281.0,
                "hrrr_temp_2m_09_local_k": 284.0,
                "hrrr_temp_2m_12_local_k": 288.0,
                "hrrr_temp_2m_15_local_k": 293.261111,
                "hrrr_temp_2m_18_local_k": 291.0,
                "hrrr_dewpoint_2m_06_local_k": 275.0,
                "hrrr_dewpoint_2m_09_local_k": 277.0,
                "hrrr_dewpoint_2m_12_local_k": 279.0,
                "hrrr_dewpoint_2m_15_local_k": 280.0,
                "hrrr_dewpoint_2m_18_local_k": 278.0,
                "hrrr_rh_2m_06_local_pct": 61.0,
                "hrrr_rh_2m_09_local_pct": 57.0,
                "hrrr_rh_2m_12_local_pct": 52.0,
                "hrrr_rh_2m_15_local_pct": 49.0,
                "hrrr_rh_2m_18_local_pct": 54.0,
                "hrrr_u10m_09_local_ms": 4.0,
                "hrrr_u10m_12_local_ms": 5.0,
                "hrrr_u10m_15_local_ms": 6.0,
                "hrrr_u10m_18_local_ms": 4.5,
                "hrrr_v10m_09_local_ms": 2.0,
                "hrrr_v10m_12_local_ms": 2.5,
                "hrrr_v10m_15_local_ms": 3.0,
                "hrrr_v10m_18_local_ms": 2.0,
                "hrrr_wind_10m_09_local_speed_ms": 4.472136,
                "hrrr_wind_10m_12_local_speed_ms": 5.59017,
                "hrrr_wind_10m_15_local_speed_ms": 6.708204,
                "hrrr_wind_10m_18_local_speed_ms": 4.924429,
                "hrrr_wind_10m_09_local_direction_deg": 243.434949,
                "hrrr_wind_10m_12_local_direction_deg": 243.434949,
                "hrrr_wind_10m_15_local_direction_deg": 243.434949,
                "hrrr_wind_10m_18_local_direction_deg": 246.037511,
                "hrrr_mslp_09_local_pa": 101400.0,
                "hrrr_mslp_12_local_pa": 101350.0,
                "hrrr_mslp_15_local_pa": 101300.0,
                "hrrr_surface_pressure_09_local_pa": 100950.0,
                "hrrr_temp_2m_day_max_k": 293.261111,
                "hrrr_temp_2m_day_mean_k": 288.0,
                "hrrr_rh_2m_day_min_pct": 45.0,
                "hrrr_wind_10m_day_max_ms": 11.0,
                "hrrr_gust_day_max_ms": 13.0,
                "hrrr_tcdc_day_mean_pct": 40.0,
                "hrrr_tcdc_morning_mean_pct": 45.0,
                "hrrr_tcdc_afternoon_mean_pct": 32.0,
                "hrrr_tcdc_day_max_pct": 75.0,
                "hrrr_lcdc_morning_mean_pct": 22.0,
                "hrrr_mcdc_day_mean_pct": 28.0,
                "hrrr_hcdc_day_mean_pct": 35.0,
                "hrrr_mcdc_afternoon_mean_pct": 24.0,
                "hrrr_hcdc_afternoon_mean_pct": 30.0,
                "hrrr_dswrf_day_max_w_m2": 510.0,
                "hrrr_dlwrf_night_mean_w_m2": 300.0,
                "hrrr_apcp_day_total_kg_m2": 0.0,
                "hrrr_cape_day_max_j_kg": 425.0,
                "hrrr_cape_afternoon_max_j_kg": 425.0,
                "hrrr_cin_day_min_j_kg": -35.0,
                "hrrr_cin_afternoon_min_j_kg": -35.0,
                "hrrr_refc_day_max": 27.0,
                "hrrr_ltng_day_max": 1.0,
                "hrrr_ltng_day_any": True,
                "hrrr_mslp_day_mean_pa": 101340.0,
                "hrrr_pwat_day_mean_kg_m2": 20.0,
                "hrrr_hpbl_day_max_m": 500.0,
                "hrrr_temp_1000mb_day_mean_k": 282.0,
                "hrrr_temp_925mb_day_mean_k": 279.0,
                "hrrr_temp_850mb_day_mean_k": 276.0,
                "hrrr_rh_925mb_day_mean_pct": 62.0,
                "hrrr_u925_day_mean_ms": 8.0,
                "hrrr_v925_day_mean_ms": 2.0,
                "hrrr_u850_day_mean_ms": 10.0,
                "hrrr_v850_day_mean_ms": 1.0,
                "hrrr_hgt_925_day_mean_gpm": 760.0,
                "hrrr_hgt_700_day_mean_gpm": 3110.0,
                "hrrr_temp_2m_k_nb3_day_mean": 288.2,
                "hrrr_temp_2m_k_crop_day_mean": 288.7,
                "hrrr_tcdc_entire_pct_crop_day_mean": 42.0,
                "hrrr_dswrf_surface_w_m2_crop_day_max": 520.0,
                "hrrr_pwat_entire_atmosphere_kg_m2_nb7_day_mean": 21.0,
                "hrrr_wind_850mb_speed_day_mean_ms": 15.0,
                "hrrr_temp_2m_day_max_k_rev_1cycle": 1.0,
                "hrrr_temp_2m_09_local_k_rev_1cycle": 1.0,
                "hrrr_temp_2m_12_local_k_rev_1cycle": 1.0,
                "hrrr_temp_2m_15_local_k_rev_1cycle": 1.0,
                "hrrr_tcdc_day_mean_pct_rev_1cycle": -2.0,
                "hrrr_dswrf_day_max_w_m2_rev_1cycle": 10.0,
                "hrrr_pwat_day_mean_kg_m2_rev_1cycle": 1.0,
                "hrrr_hpbl_day_max_m_rev_1cycle": 20.0,
                "hrrr_mslp_day_mean_pa_rev_1cycle": 30.0,
                "hrrr_temp_2m_day_max_k_rev_2cycle": 1.5,
                "hrrr_temp_2m_09_local_k_rev_2cycle": 1.5,
                "hrrr_temp_2m_12_local_k_rev_2cycle": 1.5,
                "hrrr_temp_2m_15_local_k_rev_2cycle": 1.5,
                "hrrr_tcdc_day_mean_pct_rev_2cycle": -3.0,
                "hrrr_dswrf_day_max_w_m2_rev_2cycle": 12.0,
                "hrrr_pwat_day_mean_kg_m2_rev_2cycle": 1.2,
                "hrrr_hpbl_day_max_m_rev_2cycle": 25.0,
                "hrrr_mslp_day_mean_pa_rev_2cycle": 45.0,
                "hrrr_temp_2m_day_max_k_rev_3cycle": 2.0,
                "hrrr_temp_2m_09_local_k_rev_3cycle": 2.0,
                "hrrr_temp_2m_12_local_k_rev_3cycle": 2.0,
                "hrrr_temp_2m_15_local_k_rev_3cycle": 2.0,
                "hrrr_tcdc_day_mean_pct_rev_3cycle": -4.0,
                "hrrr_dswrf_day_max_w_m2_rev_3cycle": 15.0,
                "hrrr_pwat_day_mean_kg_m2_rev_3cycle": 1.5,
                "hrrr_hpbl_day_max_m_rev_3cycle": 30.0,
                "hrrr_mslp_day_mean_pa_rev_3cycle": 55.0,
            }
        ]
    )

    merged = builder.build_training_features_overnight(
        labels_df=labels_df,
        obs_df=obs_df,
        nbm_daily_df=nbm_daily_df,
        lamp_daily_df=lamp_daily_df,
        hrrr_daily_df=hrrr_daily_df,
        cutoff_local_time="00:05",
        station_id="KLGA",
    )

    assert list(merged.columns) == contract.registry_columns()
    assert list(merged["target_date_local"]) == ["2026-04-11", "2026-04-12"]

    first_row = merged.loc[merged["target_date_local"] == "2026-04-11"].iloc[0]
    second_row = merged.loc[merged["target_date_local"] == "2026-04-12"].iloc[0]

    assert bool(first_row["meta_nbm_available"]) is False
    assert bool(second_row["meta_nbm_available"]) is True
    assert int(second_row["meta_nbm_missing_required_feature_count"]) == 0
    assert bool(second_row["meta_lamp_available"]) is True
    assert bool(second_row["meta_hrrr_available"]) is True
    assert second_row["meta_wu_last_obs_time_local"] == "2026-04-12T00:00:00-04:00"
    assert int(second_row["meta_hrrr_first_valid_hour_local"]) == 6
    assert int(second_row["meta_hrrr_last_valid_hour_local"]) == 21
    assert int(second_row["meta_hrrr_covered_hour_count"]) == 16
    assert int(second_row["meta_hrrr_covered_checkpoint_count"]) == 6
    assert second_row["wu_prev_day_final_tmax_f"] == 67.0
    assert second_row["wu_last_temp_f"] == 55.0
    assert second_row["nbm_native_tmax_2m_day_max_k"] == pytest.approx(295.0)
    assert second_row["nbm_native_tmin_2m_day_min_k"] == pytest.approx(274.0)
    assert second_row["nbm_pcpdur_day_total_h"] == pytest.approx(3.0)
    assert second_row["nbm_pwther_any_flag"] == pytest.approx(1.0)
    assert second_row["nbm_tstm_day_max_pct"] == pytest.approx(35.0)
    assert second_row["nbm_ptype_nonzero_hour_count"] == pytest.approx(3)
    assert second_row["nbm_vrate_day_max"] == pytest.approx(250.0)
    assert second_row["lamp_day_tmp_max_f_forecast"] == 68.0
    assert second_row["hrrr_temp_2m_day_max_k"] == pytest.approx(293.261111)
    assert second_row["nbm_minus_lamp_day_max_f"] == pytest.approx(2.0, abs=1e-3)
    assert second_row["nbm_minus_hrrr_day_max_k"] == pytest.approx(1.0)


def test_build_training_features_overnight_uses_label_history_for_first_target_day_previous_label_features():
    labels_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "label_final_tmax_f": 72.0,
                "label_final_tmin_f": 55.0,
                "label_market_bin": "72F",
                "label_obs_count": 25,
                "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
                "label_total_precip_in": 0.0,
            }
        ]
    )
    label_history_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-11",
                "station_id": "KLGA",
                "label_final_tmax_f": 67.0,
                "label_final_tmin_f": 49.0,
                "label_market_bin": "67F",
                "label_obs_count": 24,
                "label_first_obs_time_local": "2026-04-11T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-11T23:00:00-04:00",
                "label_total_precip_in": 0.1,
            },
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "label_final_tmax_f": 72.0,
                "label_final_tmin_f": 55.0,
                "label_market_bin": "72F",
                "label_obs_count": 25,
                "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
                "label_total_precip_in": 0.0,
            },
        ]
    )
    obs_df = pd.DataFrame.from_records(
        [
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-11T23:55:00-04:00",
                "temp_f": 56.0,
                "dewpoint_f": 47.0,
                "rh_pct": 68.0,
                "pressure_in": 30.12,
                "wind_speed_mph": 8.0,
                "wind_dir_deg": 210.0,
                "wind_gust_mph": 12.0,
                "visibility": 9.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.0,
            },
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-12T00:00:00-04:00",
                "temp_f": 55.0,
                "dewpoint_f": 46.0,
                "rh_pct": 70.0,
                "pressure_in": 30.15,
                "wind_speed_mph": 7.0,
                "wind_dir_deg": 220.0,
                "wind_gust_mph": 10.0,
                "visibility": 8.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.1,
            },
        ]
    )

    merged = builder.build_training_features_overnight(
        labels_df=labels_df,
        label_history_df=label_history_df,
        obs_df=obs_df,
        nbm_daily_df=pd.DataFrame(),
        lamp_daily_df=pd.DataFrame(),
        hrrr_daily_df=pd.DataFrame(),
        cutoff_local_time="00:05",
        station_id="KLGA",
    )

    row = merged.iloc[0]
    assert row["target_date_local"] == "2026-04-12"
    assert row["wu_prev_day_final_tmax_f"] == 67.0
    assert row["wu_prev_day_final_tmin_f"] == 49.0
    assert row["wu_prev_day_total_precip_in"] == 0.1


def test_build_training_features_overnight_filters_to_requested_station():
    labels_df = pd.DataFrame.from_records(
        [
            {
                "target_date_local": "2026-04-12",
                "station_id": "KLGA",
                "label_final_tmax_f": 72.0,
                "label_final_tmin_f": 55.0,
                "label_market_bin": "72F",
                "label_obs_count": 25,
                "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
                "label_total_precip_in": 0.0,
            },
            {
                "target_date_local": "2026-04-12",
                "station_id": "KJFK",
                "label_final_tmax_f": 70.0,
                "label_final_tmin_f": 54.0,
                "label_market_bin": "70F",
                "label_obs_count": 25,
                "label_first_obs_time_local": "2026-04-12T00:00:00-04:00",
                "label_last_obs_time_local": "2026-04-12T23:00:00-04:00",
                "label_total_precip_in": 0.0,
            },
        ]
    )
    label_history_df = pd.DataFrame.from_records(
        [
            {"target_date_local": "2026-04-11", "station_id": "KLGA", "label_final_tmax_f": 67.0, "label_final_tmin_f": 49.0},
            {"target_date_local": "2026-04-11", "station_id": "KJFK", "label_final_tmax_f": 66.0, "label_final_tmin_f": 48.0},
        ]
    )
    obs_df = pd.DataFrame.from_records(
        [
            {
                "station_id": "KLGA",
                "valid_time_local": "2026-04-12T00:00:00-04:00",
                "temp_f": 55.0,
                "dewpoint_f": 46.0,
                "rh_pct": 70.0,
                "pressure_in": 30.15,
                "wind_speed_mph": 7.0,
                "wind_dir_deg": 220.0,
                "wind_gust_mph": 10.0,
                "visibility": 8.0,
                "cloud_cover_code": "OVC",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.1,
            },
            {
                "station_id": "KJFK",
                "valid_time_local": "2026-04-12T00:00:00-04:00",
                "temp_f": 53.0,
                "dewpoint_f": 45.0,
                "rh_pct": 71.0,
                "pressure_in": 30.10,
                "wind_speed_mph": 6.0,
                "wind_dir_deg": 210.0,
                "wind_gust_mph": 9.0,
                "visibility": 8.0,
                "cloud_cover_code": "BKN",
                "wx_phrase": "Cloudy",
                "precip_hrly_in": 0.0,
            },
        ]
    )

    merged = builder.build_training_features_overnight(
        labels_df=labels_df,
        label_history_df=label_history_df,
        obs_df=obs_df,
        nbm_daily_df=pd.DataFrame(),
        lamp_daily_df=pd.DataFrame(),
        hrrr_daily_df=pd.DataFrame(),
        cutoff_local_time="00:05",
        station_id="KLGA",
    )

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["station_id"] == "KLGA"
    assert row["target_date_local"] == "2026-04-12"


def test_require_frame_rejects_missing_or_empty_required_inputs(tmp_path: pathlib.Path):
    missing_path = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        builder._require_frame("labels_daily", missing_path, pd.DataFrame(), allow_empty=False)

    existing_path = tmp_path / "empty.parquet"
    pd.DataFrame().to_parquet(existing_path)
    with pytest.raises(ValueError):
        builder._require_frame("labels_daily", existing_path, pd.DataFrame(), allow_empty=False)


def test_validate_date_output_accepts_valid_daily_shard(tmp_path: pathlib.Path):
    output_df = pd.DataFrame.from_records([{column: None for column in contract.registry_columns()}])
    output_df.loc[0, "target_date_local"] = "2026-04-12"
    output_df.loc[0, "station_id"] = "KLGA"
    output_df.loc[0, "selection_cutoff_local"] = "2026-04-12T00:05:00-04:00"
    output_df.loc[0, "label_final_tmax_f"] = 72.0
    output_df.loc[0, "label_final_tmin_f"] = 55.0
    output_df.loc[0, "label_market_bin"] = "72F"
    output_df.loc[0, "label_obs_count"] = 24
    output_df.loc[0, "label_first_obs_time_local"] = "2026-04-12T00:00:00-04:00"
    output_df.loc[0, "label_last_obs_time_local"] = "2026-04-12T23:00:00-04:00"
    output_df.loc[0, "meta_wu_obs_available"] = True
    output_df = builder.apply_registry_layout(output_df)

    output_path, manifest_path = builder.output_paths_for_date(tmp_path, "2026-04-12")
    manifest = builder.build_output_manifest_for_date(
        target_date_local="2026-04-12",
        output_path=output_path,
        row_count=len(output_df),
        station_id="KLGA",
        cutoff_local_time="00:05",
        output_df=output_df,
    )
    builder._write_atomic_parquet(output_df, output_path)
    builder._write_atomic_json(manifest_path, manifest)

    assert builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local="2026-04-12",
        allow_empty=False,
    )


def test_validate_date_output_rejects_manifest_mismatch(tmp_path: pathlib.Path):
    output_df = pd.DataFrame(columns=contract.registry_columns())
    output_path, manifest_path = builder.output_paths_for_date(tmp_path, "2026-04-12")
    builder._write_atomic_parquet(output_df, output_path)
    builder._write_atomic_json(
        manifest_path,
        {
            "status": "ok",
            "target_date_local": "2026-04-13",
            "row_count": 0,
        },
    )

    assert not builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local="2026-04-12",
        allow_empty=True,
    )
