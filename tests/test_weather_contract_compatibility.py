from __future__ import annotations

import importlib.util
import pathlib
import sys

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
COMPARE_PATH = ROOT / "tools" / "weather" / "compare_weather_contracts.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compare_contracts = load_module("compare_weather_contracts_test", COMPARE_PATH)


def write_parquet(path: pathlib.Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_parquet(path, index=False)


def synced_wide_row(source_model: str, source_product: str, source_version: str) -> dict[str, object]:
    return {
        "source_model": source_model,
        "source_product": source_product,
        "source_version": source_version,
        "fallback_used_any": False,
        "missing_optional_any": False,
        "missing_optional_fields_count": 0,
        "station_id": "KLGA",
        "init_time_utc": "2026-04-11T12:00:00+00:00",
        "valid_time_utc": "2026-04-11T13:00:00+00:00",
        "init_time_local": "2026-04-11T08:00:00-04:00",
        "valid_time_local": "2026-04-11T09:00:00-04:00",
        "init_date_local": "2026-04-11",
        "valid_date_local": "2026-04-11",
        "forecast_hour": 1,
        "settlement_lat": 40.7769,
        "settlement_lon": -73.8740,
        "crop_top_lat": 43.5,
        "crop_bottom_lat": 39.0,
        "crop_left_lon": 282.5,
        "crop_right_lon": 289.5,
        "nearest_grid_lat": 40.7769,
        "nearest_grid_lon": -73.874,
        "tmp_nb3_mean": 12.0,
        "tmp_nb7_mean": 13.0,
        "tmp_crop_mean": 14.0,
    }


def synced_provenance_row(source_model: str, source_product: str, source_version: str) -> dict[str, object]:
    return {
        "source_model": source_model,
        "source_product": source_product,
        "source_version": source_version,
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
        "fallback_used": False,
        "fallback_source_description": None,
        "present_directly": True,
        "derived": False,
        "missing_optional": False,
        "derivation_method": None,
        "source_feature_names": "[]",
        "units": "K",
    }


def manifest_row(source_model: str, source_product: str, source_version: str, status: str = "ok") -> dict[str, object]:
    return {
        "source_model": source_model,
        "source_product": source_product,
        "source_version": source_version,
        "status": status,
        "task_key": "2026-04-11_t12_f01",
        "wide_parquet_path": f"/tmp/{source_model.lower()}_wide.parquet",
        "provenance_path": f"/tmp/{source_model.lower()}_provenance.parquet",
        "failure_reason": None,
        "missing_fields": "[]",
    }


def lamp_wide_row() -> dict[str, object]:
    return {
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
    }


def lamp_provenance_row() -> dict[str, object]:
    return {
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
        "fallback_used": False,
        "fallback_source_description": None,
        "present_directly": True,
        "derived": False,
        "missing_optional": False,
        "derivation_method": None,
        "source_feature_names": "[]",
        "bulletin_type": "standard",
        "bulletin_version": "current",
        "bulletin_source_path": "/tmp/lmp.t0430z.lavtxt.ascii",
        "archive_member": None,
        "units": "F",
    }


def lamp_manifest_row() -> dict[str, object]:
    return {
        "source_model": "LAMP",
        "source_product": "lav",
        "source_version": "lamp-station-text-v1",
        "status": "ok",
        "task_key": "2026-04-12T0430Z",
        "wide_output_path": "/tmp/lamp.wide.parquet",
        "provenance_output_path": "/tmp/lamp.provenance.parquet",
        "warnings": "Missing optional LAMP labels in 38/38 forecast rows; max_missing_optional_fields_count=9.",
    }


def test_compatibility_report_marks_synced_artifacts(tmp_path):
    nbm_wide = tmp_path / "nbm" / "wide.parquet"
    nbm_prov = tmp_path / "nbm" / "provenance.parquet"
    nbm_manifest = tmp_path / "nbm" / "manifest.parquet"
    hrrr_wide = tmp_path / "hrrr" / "wide.parquet"
    hrrr_prov = tmp_path / "hrrr" / "provenance.parquet"
    hrrr_manifest = tmp_path / "hrrr" / "manifest.parquet"

    write_parquet(nbm_wide, [synced_wide_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_prov, [synced_provenance_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_manifest, [manifest_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(hrrr_wide, [synced_wide_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_prov, [synced_provenance_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_manifest, [manifest_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])

    args = type(
        "Args",
        (),
        {
            "nbm_wide": nbm_wide,
            "nbm_provenance": nbm_prov,
            "nbm_manifest": nbm_manifest,
            "hrrr_wide": hrrr_wide,
            "hrrr_provenance": hrrr_prov,
            "hrrr_manifest": hrrr_manifest,
        },
    )()
    report = compare_contracts.build_report(args)
    assert report["verdict"]["hrrr_fully_aligned"] is True
    assert report["verdict"]["raw_runtime_contract_shared"] is True
    assert report["hrrr"]["wide"]["status"] == "synced"
    assert report["hrrr"]["provenance"]["status"] == "synced"
    assert report["hrrr"]["manifest"]["status"] == "synced"
    assert report["verdict"]["overall_status"] == "synced"


def test_compatibility_report_flags_legacy_aliases_and_missing_fields(tmp_path):
    nbm_wide = tmp_path / "nbm" / "wide.parquet"
    nbm_prov = tmp_path / "nbm" / "provenance.parquet"
    nbm_manifest = tmp_path / "nbm" / "manifest.parquet"
    hrrr_wide = tmp_path / "hrrr" / "wide.parquet"
    hrrr_prov = tmp_path / "hrrr" / "provenance.parquet"
    hrrr_manifest = tmp_path / "hrrr" / "manifest.parquet"

    write_parquet(nbm_wide, [synced_wide_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_prov, [synced_provenance_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_manifest, [manifest_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    bad_hrrr_wide = synced_wide_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")
    bad_hrrr_wide["nearest_grid_lon"] = 286.126
    bad_hrrr_wide["tmp_2m_k_nearest"] = bad_hrrr_wide["tmp_nb3_mean"]
    del bad_hrrr_wide["source_version"]
    del bad_hrrr_wide["missing_optional_any"]
    write_parquet(hrrr_wide, [bad_hrrr_wide])
    bad_hrrr_prov = synced_provenance_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")
    del bad_hrrr_prov["nearest_grid_lon"]
    del bad_hrrr_prov["fallback_used"]
    del bad_hrrr_prov["source_feature_names"]
    write_parquet(hrrr_prov, [bad_hrrr_prov])
    write_parquet(hrrr_manifest, [{"source_model": "HRRR", "status": "ok"}])

    args = type(
        "Args",
        (),
        {
            "nbm_wide": nbm_wide,
            "nbm_provenance": nbm_prov,
            "nbm_manifest": nbm_manifest,
            "hrrr_wide": hrrr_wide,
            "hrrr_provenance": hrrr_prov,
            "hrrr_manifest": hrrr_manifest,
        },
    )()
    report = compare_contracts.build_report(args)
    assert report["verdict"]["hrrr_fully_aligned"] is False
    assert report["hrrr"]["wide"]["status"] == "divergent"
    assert "source_version" in report["hrrr"]["wide"]["missing_columns"]
    assert report["hrrr"]["wide"]["legacy_alias_columns"] == ["tmp_2m_k_nearest"]
    assert report["hrrr"]["wide"]["nearest_grid_lon_convention"] == "0-360"
    assert report["hrrr"]["provenance"]["status"] == "partially synced"
    assert report["hrrr"]["manifest"]["status"] == "partially synced"
    assert "wide_output_path" in report["hrrr"]["manifest"]["missing_columns"]
    assert report["verdict"]["overall_status"] == "not fully aligned"


def test_compatibility_report_fails_shared_verdict_when_nbm_is_divergent(tmp_path):
    nbm_wide = tmp_path / "nbm" / "wide.parquet"
    nbm_prov = tmp_path / "nbm" / "provenance.parquet"
    nbm_manifest = tmp_path / "nbm" / "manifest.parquet"
    hrrr_wide = tmp_path / "hrrr" / "wide.parquet"
    hrrr_prov = tmp_path / "hrrr" / "provenance.parquet"
    hrrr_manifest = tmp_path / "hrrr" / "manifest.parquet"

    write_parquet(nbm_wide, [{"source_model": "NBM"}])
    write_parquet(nbm_prov, [synced_provenance_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_manifest, [manifest_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(hrrr_wide, [synced_wide_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_prov, [synced_provenance_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_manifest, [manifest_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])

    args = type(
        "Args",
        (),
        {
            "nbm_wide": nbm_wide,
            "nbm_provenance": nbm_prov,
            "nbm_manifest": nbm_manifest,
            "hrrr_wide": hrrr_wide,
            "hrrr_provenance": hrrr_prov,
            "hrrr_manifest": hrrr_manifest,
        },
    )()
    report = compare_contracts.build_report(args)

    assert report["nbm"]["wide"]["status"] == "divergent"
    assert report["verdict"]["raw_runtime_contract_shared"] is False
    assert report["verdict"]["hrrr_fully_aligned"] is True
    assert report["verdict"]["overall_status"] == "not fully aligned"


def test_compatibility_report_accepts_lamp_when_provided(tmp_path):
    nbm_wide = tmp_path / "nbm" / "wide.parquet"
    nbm_prov = tmp_path / "nbm" / "provenance.parquet"
    nbm_manifest = tmp_path / "nbm" / "manifest.parquet"
    hrrr_wide = tmp_path / "hrrr" / "wide.parquet"
    hrrr_prov = tmp_path / "hrrr" / "provenance.parquet"
    hrrr_manifest = tmp_path / "hrrr" / "manifest.parquet"
    lamp_wide = tmp_path / "lamp" / "wide.parquet"
    lamp_prov = tmp_path / "lamp" / "provenance.parquet"
    lamp_manifest = tmp_path / "lamp" / "manifest.parquet"

    write_parquet(nbm_wide, [synced_wide_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_prov, [synced_provenance_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(nbm_manifest, [manifest_row("NBM", "grib2-core", "nbm-grib2-core-public")])
    write_parquet(hrrr_wide, [synced_wide_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_prov, [synced_provenance_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(hrrr_manifest, [manifest_row("HRRR", "wrfsfcf", "hrrr-conus-wrfsfcf-public")])
    write_parquet(lamp_wide, [lamp_wide_row()])
    write_parquet(lamp_prov, [lamp_provenance_row()])
    write_parquet(lamp_manifest, [lamp_manifest_row()])

    args = type(
        "Args",
        (),
        {
            "nbm_wide": nbm_wide,
            "nbm_provenance": nbm_prov,
            "nbm_manifest": nbm_manifest,
            "hrrr_wide": hrrr_wide,
            "hrrr_provenance": hrrr_prov,
            "hrrr_manifest": hrrr_manifest,
            "lamp_wide": lamp_wide,
            "lamp_provenance": lamp_prov,
            "lamp_manifest": lamp_manifest,
        },
    )()
    report = compare_contracts.build_report(args)

    assert report["lamp"]["wide"]["status"] == "synced"
    assert report["lamp"]["provenance"]["status"] == "synced"
    assert report["lamp"]["manifest"]["status"] == "synced"
    assert report["verdict"]["raw_runtime_contract_shared"] is True
    assert report["verdict"]["overall_status"] == "synced"
