from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
NORMALIZE_PATH = ROOT / "tools" / "weather" / "normalize_training_features.py"
SCHEMA_PATH = ROOT / "tools" / "weather" / "canonical_feature_schema.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


normalize_features = load_module("normalize_training_features_test", NORMALIZE_PATH)
canonical_schema = load_module("canonical_feature_schema_test", SCHEMA_PATH)


def nbm_wide_rows() -> list[dict[str, object]]:
    base = {
        "source_model": "NBM",
        "source_product": "grib2-core",
        "source_version": "nbm-grib2-core-public",
        "fallback_used_any": False,
        "missing_optional_any": True,
        "missing_optional_fields_count": 3,
        "station_id": "KLGA",
        "init_time_utc": "2026-04-11T12:00:00+00:00",
        "init_time_local": "2026-04-11T08:00:00-04:00",
        "init_date_local": "2026-04-11",
        "settlement_lat": 40.7769,
        "settlement_lon": -73.8740,
        "crop_top_lat": 43.5,
        "crop_bottom_lat": 39.0,
        "crop_left_lon": 282.5,
        "crop_right_lon": 289.5,
        "nearest_grid_lat": 40.7769,
        "nearest_grid_lon": -73.874,
        "mode": "premarket",
        "tmp": 280.0,
        "tmp_crop_mean": 281.0,
        "tmp_nb3_mean": 280.5,
        "tmp_nb7_std": 1.2,
        "wind": 8.0,
        "wind_nb3_mean": 8.5,
        "wdir": 240.0,
        "u10": 6.0,
        "v10": 5.0,
        "gust": 10.0,
        "tcdc": 55.0,
        "dswrf": 100.0,
        "apcp": 0.2,
        "vrate": 15.0,
        "vis": 12000.0,
        "ceil": 900.0,
        "cape": 50.0,
        "thunc": 1,
    }
    return [
        {
            **base,
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
        },
        {
            **base,
            "valid_time_utc": "2026-04-11T14:00:00+00:00",
            "valid_time_local": "2026-04-11T10:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 2,
            "tmp": 282.0,
            "u10": 7.0,
            "v10": 4.0,
        },
    ]


def hrrr_wide_rows() -> list[dict[str, object]]:
    return [
        {
            "task_key": "2026-04-11__2026-04-11_t12_f01",
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "fallback_used_any": False,
            "missing_optional_any": False,
            "missing_optional_fields_count": 0,
            "station_id": "KLGA",
            "target_date_local": "2026-04-11",
            "slice_policy": "overnight_local_v1",
            "run_date_utc": "2026-04-11",
            "cycle_hour_utc": 12,
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_date_local": "2026-04-11",
            "init_hour_local": 8,
            "valid_hour_local": 9,
            "cycle_rank_desc": 0,
            "selected_for_summary": True,
            "anchor_cycle_candidate": True,
            "forecast_hour": 1,
            "settlement_lat": 40.7769,
            "settlement_lon": -73.8740,
            "crop_top_lat": 43.5,
            "crop_bottom_lat": 39.0,
            "crop_left_lon": 282.5,
            "crop_right_lon": 289.5,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "tmp_2m_k": 281.0,
            "tmp_2m_k_crop_mean": 281.5,
            "tmp_2m_k_nb3_mean": 281.2,
            "dpt_2m_k": 275.0,
            "ugrd_10m_ms": 5.0,
            "ugrd_10m_ms_nb3_mean": 5.1,
            "vgrd_10m_ms": 2.0,
            "wind_10m_speed_ms": 5.385,
            "wind_10m_direction_deg": 248.0,
            "gust_surface_ms": 8.0,
            "surface_pressure_pa": 101325.0,
            "mslma_pa": 101500.0,
            "visibility_m": 10000.0,
            "tcdc_entire_pct": 40.0,
            "lcdc_low_pct": 20.0,
            "mcdc_mid_pct": 30.0,
            "hcdc_high_pct": 10.0,
            "dswrf_surface_w_m2": 120.0,
            "dlwrf_surface_w_m2": 300.0,
            "apcp_surface_kg_m2": 0.0,
            "prate_surface_kg_m2_s": 0.0,
            "hpbl_m": 400.0,
            "pwat_entire_atmosphere_kg_m2": 20.0,
            "cape_surface_j_kg": 100.0,
            "cin_surface_j_kg": -20.0,
            "tmp_850mb_k": 270.0,
            "rh_850mb_pct": 60.0,
            "ugrd_850mb_ms": 12.0,
            "vgrd_850mb_ms": 3.0,
            "spfh_850mb_kg_kg": 0.004,
            "hgt_850mb_gpm": 1500.0,
        }
    ]


def nbm_provenance_rows() -> list[dict[str, object]]:
    return [
        {
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "tmp",
            "grib_short_name": "TMP",
            "grib_level_text": "2 m above ground",
            "grib_step_text": "1 hour fcst",
            "inventory_line": "1:0:d=2026041112:TMP:2 m above ground:1 hour fcst:",
            "type_of_level": "heightAboveGround",
            "step_type": "instant",
            "units": "K",
            "present_directly": True,
            "derived": False,
            "missing_optional": False,
            "derivation_method": None,
            "source_feature_names": "[]",
            "fallback_used": False,
            "fallback_source_description": None,
        },
        {
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "u10",
            "present_directly": False,
            "derived": True,
            "missing_optional": False,
            "derivation_method": "zonal wind from scalar speed and meteorological direction",
            "source_feature_names": json.dumps(["wind", "wdir"]),
            "fallback_used": False,
            "fallback_source_description": None,
            "units": "m s-1",
        },
        {
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "tmax",
            "present_directly": False,
            "derived": False,
            "missing_optional": True,
            "derivation_method": None,
            "source_feature_names": "[]",
            "fallback_used": False,
            "fallback_source_description": None,
            "units": None,
        },
    ]


def hrrr_provenance_rows() -> list[dict[str, object]]:
    return [
        {
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "tmp_2m_k",
            "grib_short_name": "TMP",
            "grib_level_text": "2 m above ground",
            "grib_type_of_level": "heightAboveGround",
            "grib_step_type": "instant",
            "grib_step_text": "1 hour fcst",
            "source_inventory_line": "1:0:d=2026041112:TMP:2 m above ground:1 hour fcst:",
            "units": "K",
            "present_directly": True,
            "derived": False,
            "derivation_method": None,
            "source_feature_names": "[]",
            "missing_optional": False,
            "fallback_used": False,
            "fallback_source_description": None,
            "notes": None,
        },
        {
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "wind_10m_speed_ms",
            "present_directly": False,
            "derived": True,
            "derivation_method": "sqrt(u10^2 + v10^2)",
            "source_feature_names": json.dumps(["ugrd_10m_ms", "vgrd_10m_ms"]),
            "missing_optional": False,
            "fallback_used": False,
            "fallback_source_description": None,
            "units": "m s-1",
            "notes": None,
        },
        {
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-11T12:00:00+00:00",
            "init_time_local": "2026-04-11T08:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_time_utc": "2026-04-11T13:00:00+00:00",
            "valid_time_local": "2026-04-11T09:00:00-04:00",
            "valid_date_local": "2026-04-11",
            "forecast_hour": 1,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "feature_name": "cape_surface_j_kg",
            "present_directly": False,
            "derived": False,
            "derivation_method": None,
            "source_feature_names": "[]",
            "missing_optional": True,
            "fallback_used": False,
            "fallback_source_description": None,
            "units": None,
            "notes": "Requested feature family was not present in the reduced GRIB inventory.",
        },
    ]


def lamp_wide_rows() -> list[dict[str, object]]:
    return [
        {
            "source_model": "LAMP",
            "source_product": "lav",
            "source_version": "lamp-station-text-v1",
            "fallback_used_any": False,
            "missing_optional_any": True,
            "missing_optional_fields_count": 2,
            "station_id": "KLGA",
            "init_time_utc": "2026-04-12T04:30:00+00:00",
            "valid_time_utc": "2026-04-12T05:00:00+00:00",
            "init_time_local": "2026-04-12T00:30:00-04:00",
            "valid_time_local": "2026-04-12T01:00:00-04:00",
            "init_date_local": "2026-04-12",
            "valid_date_local": "2026-04-12",
            "forecast_hour": 1,
            "tmp": 50,
            "dpt": 42,
            "wdr": 240,
            "wsp": 12,
            "wgs": 20,
            "cig": 35,
            "vis": 6,
            "typ": "R",
        }
    ]


def lamp_provenance_rows() -> list[dict[str, object]]:
    return [
        {
            "source_model": "LAMP",
            "source_product": "lav",
            "source_version": "lamp-station-text-v1",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-12T04:30:00+00:00",
            "init_time_local": "2026-04-12T00:30:00-04:00",
            "init_date_local": "2026-04-12",
            "valid_time_utc": "2026-04-12T05:00:00+00:00",
            "valid_time_local": "2026-04-12T01:00:00-04:00",
            "valid_date_local": "2026-04-12",
            "forecast_hour": 1,
            "feature_name": "tmp",
            "raw_feature_name": "TMP",
            "present_directly": True,
            "derived": False,
            "missing_optional": False,
            "derivation_method": None,
            "source_feature_names": "[]",
            "fallback_used": False,
            "fallback_source_description": None,
            "units": "F",
            "bulletin_type": "standard",
            "bulletin_version": "current",
            "bulletin_source_path": "/tmp/lmp.t0430z.lavtxt.ascii",
            "archive_member": None,
        },
        {
            "source_model": "LAMP",
            "source_product": "lav",
            "source_version": "lamp-station-text-v1",
            "station_id": "KLGA",
            "init_time_utc": "2026-04-12T04:30:00+00:00",
            "init_time_local": "2026-04-12T00:30:00-04:00",
            "init_date_local": "2026-04-12",
            "valid_time_utc": "2026-04-12T05:00:00+00:00",
            "valid_time_local": "2026-04-12T01:00:00-04:00",
            "valid_date_local": "2026-04-12",
            "forecast_hour": 1,
            "feature_name": "p06",
            "raw_feature_name": "P06",
            "present_directly": False,
            "derived": False,
            "missing_optional": True,
            "derivation_method": None,
            "source_feature_names": "[]",
            "fallback_used": False,
            "fallback_source_description": None,
            "units": None,
            "bulletin_type": "standard;extended",
            "bulletin_version": "current",
            "bulletin_source_path": "/tmp/lmp.t0430z.lavtxt.ascii;/tmp/lmp.t0430z.lavtxt_ext.ascii",
            "archive_member": None,
        },
    ]


def test_wide_normalizers_emit_shared_canonical_column_set_and_preserve_rows():
    nbm_raw = pd.DataFrame.from_records(nbm_wide_rows())
    hrrr_raw = pd.DataFrame.from_records(hrrr_wide_rows())

    nbm_canonical = normalize_features.normalize_nbm_wide_to_canonical(nbm_raw)
    hrrr_canonical = normalize_features.normalize_hrrr_wide_to_canonical(hrrr_raw)

    assert list(nbm_canonical.columns) == list(hrrr_canonical.columns)
    assert len(nbm_canonical) == len(nbm_raw)
    assert len(hrrr_canonical) == len(hrrr_raw)
    assert list(nbm_canonical["valid_time_utc"]) == list(nbm_raw["valid_time_utc"])
    assert nbm_canonical.iloc[0]["temp_2m_k"] == nbm_raw.iloc[0]["tmp"]
    assert hrrr_canonical.iloc[0]["temp_2m_k"] == hrrr_raw.iloc[0]["tmp_2m_k"]
    assert nbm_canonical.iloc[0]["wind_10m_u_ms"] == nbm_raw.iloc[0]["u10"]
    assert hrrr_canonical.iloc[0]["wind_10m_u_ms"] == hrrr_raw.iloc[0]["ugrd_10m_ms"]
    assert pd.isna(nbm_canonical.iloc[0]["cin_surface_j_kg"])
    assert nbm_canonical.iloc[0]["mode"] == "premarket"
    assert pd.isna(hrrr_canonical.iloc[0]["mode"])


def test_provenance_normalizers_map_semantics_and_lineage():
    nbm_raw = pd.DataFrame.from_records(nbm_provenance_rows())
    hrrr_raw = pd.DataFrame.from_records(hrrr_provenance_rows())

    nbm_canonical = normalize_features.normalize_nbm_provenance_to_canonical(nbm_raw)
    hrrr_canonical = normalize_features.normalize_hrrr_provenance_to_canonical(hrrr_raw)

    assert list(nbm_canonical.columns) == list(canonical_schema.CANONICAL_PROVENANCE_COLUMNS)
    assert list(hrrr_canonical.columns) == list(canonical_schema.CANONICAL_PROVENANCE_COLUMNS)

    nbm_tmp = nbm_canonical.loc[nbm_canonical["raw_feature_name"] == "tmp"].iloc[0]
    assert nbm_tmp["feature_name"] == "temp_2m_k"
    assert nbm_tmp["grib_type_of_level"] == "heightAboveGround"
    assert json.loads(nbm_tmp["source_feature_names"]) == []

    nbm_u10 = nbm_canonical.loc[nbm_canonical["raw_feature_name"] == "u10"].iloc[0]
    assert nbm_u10["feature_name"] == "wind_10m_u_ms"
    assert json.loads(nbm_u10["source_feature_names"]) == ["wind_10m_speed_ms", "wind_10m_direction_deg"]

    hrrr_speed = hrrr_canonical.loc[hrrr_canonical["raw_feature_name"] == "wind_10m_speed_ms"].iloc[0]
    assert json.loads(hrrr_speed["source_feature_names"]) == ["wind_10m_u_ms", "wind_10m_v_ms"]

    hrrr_missing = hrrr_canonical.loc[hrrr_canonical["raw_feature_name"] == "cape_surface_j_kg"].iloc[0]
    assert bool(hrrr_missing["missing_optional"]) is True


def test_cli_materializes_column_compatible_canonical_parquet(tmp_path):
    nbm_dir = tmp_path / "nbm"
    hrrr_dir = tmp_path / "hrrr"
    output_dir = tmp_path / "canonical"
    nbm_dir.mkdir()
    hrrr_dir.mkdir()

    pd.DataFrame.from_records(nbm_wide_rows()).to_parquet(nbm_dir / "wide.parquet", index=False)
    pd.DataFrame.from_records(nbm_provenance_rows()).to_parquet(nbm_dir / "provenance.parquet", index=False)
    pd.DataFrame.from_records(hrrr_wide_rows()).to_parquet(hrrr_dir / "wide.parquet", index=False)
    pd.DataFrame.from_records(hrrr_provenance_rows()).to_parquet(hrrr_dir / "provenance.parquet", index=False)

    subprocess.run(
        [
            sys.executable,
            str(NORMALIZE_PATH),
            "--source",
            "nbm",
            "--wide",
            str(nbm_dir / "wide.parquet"),
            "--provenance",
            str(nbm_dir / "provenance.parquet"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            sys.executable,
            str(NORMALIZE_PATH),
            "--source",
            "hrrr",
            "--wide",
            str(hrrr_dir / "wide.parquet"),
            "--provenance",
            str(hrrr_dir / "provenance.parquet"),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        cwd=ROOT,
    )

    nbm_wide = pd.read_parquet(output_dir / "source=nbm" / "wide.parquet")
    hrrr_wide = pd.read_parquet(output_dir / "source=hrrr" / "wide.parquet")
    nbm_prov = pd.read_parquet(output_dir / "source=nbm" / "provenance.parquet")
    hrrr_prov = pd.read_parquet(output_dir / "source=hrrr" / "provenance.parquet")

    assert list(nbm_wide.columns) == list(hrrr_wide.columns)
    assert list(nbm_prov.columns) == list(hrrr_prov.columns)


def test_lamp_normalizers_emit_canonical_rows_with_unit_conversions():
    lamp_raw = pd.DataFrame.from_records(lamp_wide_rows())
    lamp_prov_raw = pd.DataFrame.from_records(lamp_provenance_rows())

    lamp_canonical = normalize_features.normalize_lamp_wide_to_canonical(lamp_raw)
    lamp_prov = normalize_features.normalize_lamp_provenance_to_canonical(lamp_prov_raw)

    assert lamp_canonical.iloc[0]["forecast_hour"] == 1
    assert abs(lamp_canonical.iloc[0]["temp_2m_k"] - 283.15) < 1e-6
    assert abs(lamp_canonical.iloc[0]["dewpoint_2m_k"] - 278.7055555555555) < 1e-6
    assert abs(lamp_canonical.iloc[0]["wind_10m_speed_ms"] - (12 * 0.514444)) < 1e-6
    assert abs(lamp_canonical.iloc[0]["gust_10m_ms"] - (20 * 0.514444)) < 1e-6
    assert abs(lamp_canonical.iloc[0]["ceiling_m"] - (35 * 100 * 0.3048)) < 1e-6
    assert abs(lamp_canonical.iloc[0]["visibility_m"] - (6 * 1609.344)) < 1e-6
    assert lamp_canonical.iloc[0]["ptype_code"] == "R"
    assert list(lamp_prov.columns) == list(canonical_schema.CANONICAL_PROVENANCE_COLUMNS)
    assert lamp_prov.iloc[0]["feature_name"] == "temp_2m_k"
    assert lamp_prov.iloc[1]["feature_name"] == "p06"
    assert bool(lamp_prov.iloc[1]["missing_optional"]) is True
