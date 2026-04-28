from __future__ import annotations

import argparse
import concurrent.futures
import io
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import xarray as xr


ROOT = pathlib.Path(__file__).resolve().parents[1]
HRRR_DIR = ROOT / "tools" / "hrrr"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HRRR_DIR) not in sys.path:
    sys.path.insert(0, str(HRRR_DIR))

import tools.weather.progress as progress_mod

HRRR_PIPELINE_PATH = HRRR_DIR / "build_hrrr_klga_feature_shards.py"
HRRR_FETCH_PATH = HRRR_DIR / "fetch_hrrr_records.py"
HRRR_MONTHLY_BACKFILL_PATH = HRRR_DIR / "run_hrrr_monthly_backfill.py"
HRRR_BENCHMARK_PATH = HRRR_DIR / "benchmark_hrrr_sources.py"
HRRR_BINARY_BENCHMARK_PATH = HRRR_DIR / "benchmark_hrrr_binary_extractors.py"
HRRR_SUMMARIZE_DIAGNOSTICS_PATH = HRRR_DIR / "summarize_hrrr_diagnostics.py"
HRRR_WGRIB2_BIN_PARITY_PATH = HRRR_DIR / "check_hrrr_wgrib2_bin_parity.py"
LOCATION_CONTEXT_PATH = ROOT / "tools" / "weather" / "location_context.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


location_context = load_module("weather_location_context_test", LOCATION_CONTEXT_PATH)
hrrr_fetch = load_module("hrrr_fetch_test", HRRR_FETCH_PATH)
hrrr_pipeline = load_module("hrrr_pipeline_test", HRRR_PIPELINE_PATH)
hrrr_monthly_backfill = load_module("hrrr_monthly_backfill_test", HRRR_MONTHLY_BACKFILL_PATH)
hrrr_benchmark = load_module("hrrr_benchmark_test", HRRR_BENCHMARK_PATH)
hrrr_binary_benchmark = load_module("hrrr_binary_benchmark_test", HRRR_BINARY_BENCHMARK_PATH)
hrrr_diagnostics_summary = load_module("hrrr_diagnostics_summary_test", HRRR_SUMMARIZE_DIAGNOSTICS_PATH)
hrrr_wgrib2_bin_parity = load_module("hrrr_wgrib2_bin_parity_test", HRRR_WGRIB2_BIN_PARITY_PATH)


def make_lat_lon_grid() -> tuple[np.ndarray, np.ndarray]:
    lat_axis = np.array([41.25, 41.10, 40.95, 40.7769, 40.60, 40.45, 40.30], dtype=float)
    lon_axis = np.array([285.80, 285.95, 286.05, 286.1260, 286.22, 286.35, 286.50], dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    return lat_grid, lon_grid


def test_parse_args_accepts_pause_control_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hrrr_klga_feature_shards.py",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-01",
            "--pause-control-file",
            "/tmp/hrrr.pause",
        ],
    )

    args = hrrr_pipeline.parse_args()

    assert args.pause_control_file == "/tmp/hrrr.pause"


def test_parse_args_accepts_batch_reduce_mode(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_hrrr_klga_feature_shards.py",
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-01",
            "--batch-reduce-mode",
            "cycle",
        ],
    )

    args = hrrr_pipeline.parse_args()

    assert args.batch_reduce_mode == "cycle"


def test_benchmark_build_remote_paths_for_source_google_and_azure():
    google_grib, google_idx = hrrr_benchmark.build_remote_paths_for_source("20260101", 6, "surface", 3, "google")
    azure_grib, azure_idx = hrrr_benchmark.build_remote_paths_for_source("20260101", 6, "surface", 3, "azure")

    assert google_grib == "https://storage.googleapis.com/high-resolution-rapid-refresh/hrrr.20260101/conus/hrrr.t06z.wrfsfcf03.grib2"
    assert google_idx == f"{google_grib}.idx"
    assert azure_grib == "https://noaahrrr.blob.core.windows.net/hrrr/hrrr.20260101/conus/hrrr.t06z.wrfsfcf03.grib2"
    assert azure_idx == f"{azure_grib}.idx"


def test_benchmark_parse_args_accepts_sources_and_runs(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_hrrr_sources.py",
            "--date",
            "20260101",
            "--cycle",
            "6",
            "--forecast-hour",
            "3",
            "--sources",
            "google",
            "azure",
            "--runs",
            "2",
        ],
    )

    args = hrrr_benchmark.parse_args()

    assert args.sources == ["google", "azure"]
    assert args.runs == 2


def test_wgrib2_bin_parity_parse_forecast_hours():
    assert hrrr_wgrib2_bin_parity.parse_forecast_hours("0-3,6,9") == {0, 1, 2, 3, 6, 9}


def test_row_col_slices_from_ij_box_round_trips_crop_box():
    ij_box = hrrr_pipeline.CropIjBox(i0=1436, i1=1666, j0=612, j1=821)
    row_slice, col_slice = hrrr_pipeline.row_col_slices_from_ij_box(
        ij_box=ij_box,
        grid_shape=(1059, 1799),
        north_is_first=False,
        west_is_first=True,
    )

    assert row_slice == slice(611, 821)
    assert col_slice == slice(1435, 1666)
    assert row_slice.stop - row_slice.start == ij_box.ny
    assert col_slice.stop - col_slice.start == ij_box.nx


def test_wgrib2_bin_parity_compare_rows_reports_numeric_failures():
    comparison = hrrr_wgrib2_bin_parity.compare_rows(
        {"a": 1.0, "b": 2.0, "same": "x"},
        {"a": 1.00001, "b": 2.1, "same": "x", "extra": 3.0},
        tolerance=1e-4,
    )

    assert comparison["numeric_compared"] == 2
    assert comparison["max_numeric_column"] == "b"
    assert comparison["numeric_failures"] == [
        {"column": "b", "reference": 2.0, "candidate": 2.1, "abs_diff": pytest.approx(0.1)}
    ]
    assert comparison["candidate_only_columns"] == ["extra"]


def test_wgrib2_bin_parity_compare_rows_reports_nonfinite_mismatch():
    comparison = hrrr_wgrib2_bin_parity.compare_rows(
        {"finite": 1.0, "both_nan": float("nan")},
        {"finite": float("nan"), "both_nan": float("nan")},
        tolerance=1e-4,
    )

    assert comparison["numeric_compared"] == 2
    assert len(comparison["numeric_failures"]) == 1
    failure = comparison["numeric_failures"][0]
    assert failure["column"] == "finite"
    assert failure["reference"] == 1.0
    assert np.isnan(failure["candidate"])
    assert failure["abs_diff"] == float("inf")


def test_hrrr_monthly_backfill_parse_args_accepts_wgrib2_bin(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hrrr_monthly_backfill.py",
            "--start-local-date",
            "2026-04-11",
            "--end-local-date",
            "2026-04-11",
            "--extract-method",
            "wgrib2-ijbox-bin",
        ],
    )

    args = hrrr_monthly_backfill.parse_args()

    assert args.extract_method == "wgrib2-ijbox-bin"


def test_hrrr_binary_benchmark_parse_args_accepts_direct_method(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_hrrr_binary_extractors.py",
            "--reduced-grib",
            str(tmp_path / "reduced.grib2"),
            "--direct-grib",
            str(tmp_path / "direct.grib2"),
            "--target-date",
            "2026-04-11",
            "--run-date-utc",
            "2026-04-11",
            "--cycle-hour-utc",
            "4",
            "--forecast-hours",
            "0-2",
            "--methods",
            "wgrib2-bin",
            "wgrib2-ijbox-bin",
            "--runs",
            "2",
        ],
    )

    args = hrrr_binary_benchmark.parse_args()

    assert args.reduced_grib == tmp_path / "reduced.grib2"
    assert args.direct_grib == tmp_path / "direct.grib2"
    assert args.methods == ["wgrib2-bin", "wgrib2-ijbox-bin"]
    assert args.runs == 2


def test_hrrr_binary_benchmark_flattens_run_diagnostics():
    row = hrrr_binary_benchmark.flatten_run(
        {
            "method": "wgrib2-bin",
            "run_index": 1,
            "ok": True,
            "task_count": 1,
            "ok_task_count": 1,
            "wall_seconds": 1.2,
            "cpu_seconds": 0.9,
            "maxrss_mib_after": 128.0,
            "maxrss_mib_delta": 4.0,
            "diagnostic_totals": {
                "binary_temp_bytes_observed": 100,
                "binary_cache_hit_count": 1,
                "binary_cache_miss_count": 1,
                "timing_binary_dump_seconds": 0.2,
                "timing_binary_read_seconds": 0.3,
            },
            "parity": {
                "failed_row_count": 0,
                "max_row_numeric_diff": 1e-5,
                "max_row_numeric_column": "tmp_2m_f",
                "summary": {
                    "max_numeric_diff": 2e-5,
                    "max_numeric_column": "hrrr_tmp_2m_day_mean_f",
                },
            },
        }
    )

    assert row["method"] == "wgrib2-bin"
    assert row["binary_temp_bytes_observed"] == 100
    assert row["max_row_numeric_diff"] == 1e-5
    assert row["max_summary_numeric_column"] == "hrrr_tmp_2m_day_mean_f"


def test_run_month_legacy_pause_control_stops_additional_task_submission(tmp_path, monkeypatch):
    tasks = [
        hrrr_pipeline.TaskSpec(
            target_date_local="2023-01-01",
            run_date_utc="2023-01-01",
            cycle_hour_utc=0,
            forecast_hour=5,
            init_time_utc="2023-01-01T00:00:00Z",
            init_time_local="2022-12-31T19:00:00-05:00",
            valid_time_utc="2023-01-01T05:00:00Z",
            valid_time_local="2023-01-01T00:00:00-05:00",
            init_date_local="2022-12-31",
            valid_date_local="2023-01-01",
            init_hour_local=19,
            valid_hour_local=0,
            cycle_rank_desc=0,
            selected_for_summary=True,
            anchor_cycle_candidate=True,
        ),
        hrrr_pipeline.TaskSpec(
            target_date_local="2023-01-01",
            run_date_utc="2023-01-01",
            cycle_hour_utc=1,
            forecast_hour=4,
            init_time_utc="2023-01-01T01:00:00Z",
            init_time_local="2022-12-31T20:00:00-05:00",
            valid_time_utc="2023-01-01T05:00:00Z",
            valid_time_local="2023-01-01T00:00:00-05:00",
            init_date_local="2022-12-31",
            valid_date_local="2023-01-01",
            init_hour_local=20,
            valid_hour_local=0,
            cycle_rank_desc=1,
            selected_for_summary=True,
            anchor_cycle_candidate=False,
        ),
    ]
    processed_tasks: list[str] = []
    pause_once = {"value": False}

    def fake_process_task(task, **kwargs):
        reporter = kwargs.get("reporter")
        processed_tasks.append(task.key)
        if reporter is not None and not pause_once["value"]:
            pause_once["value"] = True
            reporter.request_pause(reason="operator")
        return hrrr_pipeline.TaskResult(
            ok=True,
            task_key=task.key,
            row={"task_key": task.key},
            provenance_rows=[],
            missing_fields=[],
            diagnostics={},
        )

    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
        pause_control_file=str(tmp_path / "pause.request"),
    )

    assert wrote == 1
    assert failed == 0
    assert processed_tasks == [tasks[0].key]


def make_data_array(
    values: np.ndarray,
    *,
    short_name: str,
    type_of_level: str,
    step_type: str,
    units: str,
) -> xr.DataArray:
    lat_grid, lon_grid = make_lat_lon_grid()
    return xr.DataArray(
        values,
        dims=("y", "x"),
        coords={
            "latitude": (("y", "x"), lat_grid),
            "longitude": (("y", "x"), lon_grid),
            "time": pd.Timestamp("2023-01-01T12:00:00Z"),
            "valid_time": pd.Timestamp("2023-01-01T12:00:00Z"),
        },
        attrs={
            "GRIB_shortName": short_name,
            "GRIB_typeOfLevel": type_of_level,
            "GRIB_stepType": step_type,
            "units": units,
        },
    )


def build_test_datasets() -> dict[str, xr.Dataset]:
    base = np.arange(49, dtype=float).reshape(7, 7)
    return {
        "near_surface_2m": xr.Dataset(
            {
                "t2m": make_data_array(base + 273.15, short_name="t2m", type_of_level="heightAboveGround", step_type="instant", units="K"),
                "d2m": make_data_array(base + 268.15, short_name="d2m", type_of_level="heightAboveGround", step_type="instant", units="K"),
                "r2": make_data_array(base + 10.0, short_name="r2", type_of_level="heightAboveGround", step_type="instant", units="%"),
            }
        ),
        "near_surface_10m": xr.Dataset(
            {
                "u10": make_data_array(base / 10.0, short_name="u10", type_of_level="heightAboveGround", step_type="instant", units="m s-1"),
                "v10": make_data_array(base / 20.0, short_name="v10", type_of_level="heightAboveGround", step_type="instant", units="m s-1"),
            }
        ),
        "pwat": xr.Dataset(
            {
                "pwat": make_data_array(base + 5.0, short_name="pwat", type_of_level="atmosphereSingleLayer", step_type="instant", units="kg m-2"),
            }
        ),
        "isobaric_850": xr.Dataset(
            {
                "t": make_data_array(base + 260.0, short_name="t", type_of_level="isobaricInhPa", step_type="instant", units="K"),
                "u": make_data_array(base / 8.0, short_name="u", type_of_level="isobaricInhPa", step_type="instant", units="m s-1"),
                "v": make_data_array(base / 9.0, short_name="v", type_of_level="isobaricInhPa", step_type="instant", units="m s-1"),
                "gh": make_data_array(base + 1500.0, short_name="gh", type_of_level="isobaricInhPa", step_type="instant", units="gpm"),
                "dpt": make_data_array(base + 255.0, short_name="dpt", type_of_level="isobaricInhPa", step_type="instant", units="K"),
            }
        ),
    }


def make_task(
    *,
    target_date_local: str = "2023-01-01",
    run_date_utc: str = "2023-01-01",
    cycle_hour_utc: int = 12,
    forecast_hour: int = 0,
    init_time_utc: str | None = None,
    valid_time_utc: str | None = None,
    init_time_local: str | None = None,
    valid_time_local: str | None = None,
    cycle_rank_desc: int = 0,
    anchor_cycle_candidate: bool = True,
) -> object:
    init_time_utc = init_time_utc or f"{run_date_utc}T{cycle_hour_utc:02d}:00:00+00:00"
    init_ts_utc = pd.Timestamp(init_time_utc)
    valid_ts_utc = pd.Timestamp(valid_time_utc) if valid_time_utc else init_ts_utc + pd.Timedelta(hours=forecast_hour)
    init_ts_local = pd.Timestamp(init_time_local) if init_time_local else init_ts_utc.tz_convert(hrrr_pipeline.DEFAULT_TIMEZONE)
    valid_ts_local = pd.Timestamp(valid_time_local) if valid_time_local else valid_ts_utc.tz_convert(hrrr_pipeline.DEFAULT_TIMEZONE)
    return hrrr_pipeline.TaskSpec(
        target_date_local=target_date_local,
        run_date_utc=run_date_utc,
        cycle_hour_utc=cycle_hour_utc,
        forecast_hour=forecast_hour,
        init_time_utc=init_ts_utc.isoformat(),
        init_time_local=init_ts_local.isoformat(),
        valid_time_utc=valid_ts_utc.isoformat(),
        valid_time_local=valid_ts_local.isoformat(),
        init_date_local=init_ts_local.date().isoformat(),
        valid_date_local=valid_ts_local.date().isoformat(),
        init_hour_local=int(init_ts_local.hour),
        valid_hour_local=int(valid_ts_local.hour),
        cycle_rank_desc=cycle_rank_desc,
        selected_for_summary=True,
        anchor_cycle_candidate=anchor_cycle_candidate,
    )


def test_reduce_grib2_for_task_does_not_swallow_internal_typeerror(monkeypatch, tmp_path):
    def fake_reduce_grib2(_wgrib2_path, _raw_path, _reduced_path, *, task):
        raise TypeError(f"internal task failure for {task.key}")

    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)

    with pytest.raises(TypeError, match="internal task failure"):
        hrrr_pipeline.reduce_grib2_for_task(
            "wgrib2",
            tmp_path / "raw.grib2",
            tmp_path / "reduced.grib2",
            task=make_task(),
            raw_manifest_path=tmp_path / "raw.manifest.csv",
        )


def test_process_reduced_grib_compat_does_not_swallow_internal_typeerror(monkeypatch, tmp_path):
    calls = 0

    def fake_process_reduced_grib(_reduced_path, _inventory, _task, _url, *, diagnostics=None, write_provenance=True):
        nonlocal calls
        calls += 1
        assert diagnostics == {"source": "test"}
        assert write_provenance is False
        raise TypeError("internal diagnostics failure")

    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    with pytest.raises(TypeError, match="internal diagnostics failure"):
        hrrr_pipeline.process_reduced_grib_compat(
            tmp_path / "reduced.grib2",
            [],
            make_task(),
            "https://example.com/hrrr.grib2",
            cfgrib_index_dir=tmp_path / "idx",
            diagnostics={"source": "test"},
            write_provenance=False,
        )
    assert calls == 1


def test_process_reduced_grib_compat_omits_unsupported_optional_kwargs(monkeypatch, tmp_path):
    received: dict[str, object] = {}

    def fake_process_reduced_grib(_reduced_path, _inventory, task, _url, *, include_legacy_aliases=False):
        received["task_key"] = task.key
        received["include_legacy_aliases"] = include_legacy_aliases
        return hrrr_pipeline.TaskResult(
            True,
            task.key,
            make_minimal_hrrr_output_row_from_task(task),
            [],
            [],
            None,
            {},
        )

    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)
    task = make_task()

    result = hrrr_pipeline.process_reduced_grib_compat(
        tmp_path / "reduced.grib2",
        [],
        task,
        "https://example.com/hrrr.grib2",
        cfgrib_index_dir=tmp_path / "idx",
        diagnostics={"source": "test"},
        include_legacy_aliases=True,
        write_provenance=False,
    )

    assert result.ok is True
    assert received == {"task_key": task.key, "include_legacy_aliases": True}


def make_summary_input_row_from_task(task, *, base_temp: float = 300.0) -> dict[str, object]:
    hour = task.valid_hour_local
    return {
        "task_key": task.key,
        "target_date_local": task.target_date_local,
        "run_date_utc": task.run_date_utc,
        "cycle_hour_utc": task.cycle_hour_utc,
        "forecast_hour": task.forecast_hour,
        "init_time_utc": task.init_time_utc,
        "init_time_local": task.init_time_local,
        "valid_time_utc": task.valid_time_utc,
        "valid_time_local": task.valid_time_local,
        "init_date_local": task.init_date_local,
        "valid_date_local": task.valid_date_local,
        "init_hour_local": task.init_hour_local,
        "valid_hour_local": task.valid_hour_local,
        "cycle_rank_desc": task.cycle_rank_desc,
        "selected_for_summary": task.selected_for_summary,
        "anchor_cycle_candidate": task.anchor_cycle_candidate,
        "tmp_2m_k": base_temp + hour,
        "dpt_2m_k": base_temp - 10.0 + hour,
        "rh_2m_pct": 50.0 + (hour % 10),
        "ugrd_10m_ms": 2.0 + hour,
        "vgrd_10m_ms": 1.0 + (hour / 2.0),
        "wind_10m_speed_ms": 5.0 + (hour % 3),
        "gust_surface_ms": 7.0 + (hour % 2),
        "tcdc_entire_pct": 20.0 + hour,
        "lcdc_low_pct": 10.0 + (hour % 5),
        "mcdc_mid_pct": 15.0 + hour,
        "hcdc_high_pct": 5.0 + hour,
        "dswrf_surface_w_m2": float(hour * 10),
        "dlwrf_surface_w_m2": 300.0 + hour,
        "apcp_surface_kg_m2": 0.1,
        "surface_pressure_pa": 101000.0 + hour,
        "mslma_pa": 101200.0 + hour,
        "cape_surface_j_kg": float(hour * 25),
        "cin_surface_j_kg": float(-hour),
        "refc_entire_atmosphere": float(hour / 2.0),
        "ltng_entire_atmosphere": float(1 if hour in {15, 18} else 0),
        "pwat_entire_atmosphere_kg_m2": 20.0 + (hour / 10.0),
        "hpbl_m": 100.0 + hour,
        "tmp_1000mb_k": base_temp - 15.0,
        "tmp_925mb_k": base_temp - 20.0,
        "tmp_850mb_k": base_temp - 25.0,
        "rh_925mb_pct": 60.0,
        "ugrd_925mb_ms": 8.0,
        "vgrd_925mb_ms": 3.0,
        "ugrd_850mb_ms": 10.0,
        "vgrd_850mb_ms": 0.0,
        "hgt_925mb_gpm": 780.0,
        "hgt_700mb_gpm": 3120.0,
        "tmp_2m_k_nb3_mean": base_temp + hour + 0.5,
        "tmp_2m_k_crop_mean": base_temp + hour + 1.0,
        "tcdc_entire_pct_crop_mean": 22.0 + hour,
        "dswrf_surface_w_m2_crop_max": float(hour * 10 + 5),
        "pwat_entire_atmosphere_kg_m2_nb7_mean": 19.5 + (hour / 10.0),
    }


def make_minimal_hrrr_output_row_from_task(task) -> dict[str, object]:
    row = make_summary_input_row_from_task(task)
    row.update(
        {
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "fallback_used_any": False,
            "station_id": "KLGA",
            "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
            "settlement_lat": 40.7769,
            "settlement_lon": -73.8740,
            "crop_top_lat": 43.5,
            "crop_bottom_lat": 39.0,
            "crop_left_lon": 282.5,
            "crop_right_lon": 289.5,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "missing_optional_any": False,
            "missing_optional_fields_count": 0,
        }
    )
    return row


def test_merged_byte_ranges_merges_touching_overlapping_and_small_gaps():
    records = [
        hrrr_fetch.IdxRecord(1, 0, "d=2023010112", "TMP", "2 m above ground", "anl", 99),
        hrrr_fetch.IdxRecord(2, 100, "d=2023010112", "DPT", "2 m above ground", "anl", 199),
        hrrr_fetch.IdxRecord(3, 240, "d=2023010112", "RH", "2 m above ground", "anl", 299),
        hrrr_fetch.IdxRecord(4, 300, "d=2023010112", "UGRD", "10 m above ground", "anl", 399),
    ]

    spans = hrrr_fetch.merged_byte_ranges(records, max_gap_bytes=32)

    assert [(span.start, span.end) for span in spans] == [(0, 199), (240, 399)]


def test_merged_byte_ranges_keeps_large_gaps_separate():
    records = [
        hrrr_fetch.IdxRecord(1, 0, "d=2023010112", "TMP", "2 m above ground", "anl", 99),
        hrrr_fetch.IdxRecord(2, 200, "d=2023010112", "DPT", "2 m above ground", "anl", 299),
    ]

    spans = hrrr_fetch.merged_byte_ranges(records, max_gap_bytes=64)

    assert [(span.start, span.end) for span in spans] == [(0, 99), (200, 299)]


def test_download_selected_records_uses_merged_ranges_and_preserves_selection_manifest(tmp_path, monkeypatch):
    selected = [
        hrrr_fetch.IdxRecord(1, 0, "d=2023010112", "TMP", "2 m above ground", "anl", 99),
        hrrr_fetch.IdxRecord(2, 100, "d=2023010112", "DPT", "2 m above ground", "anl", 199),
        hrrr_fetch.IdxRecord(3, 70000, "d=2023010112", "RH", "2 m above ground", "anl", 70049),
    ]
    calls: list[tuple[int, int]] = []

    def fake_download_range(_url, start, end):
        calls.append((start, end))
        return bytes(end - start + 1)

    monkeypatch.setattr(hrrr_fetch, "download_range", fake_download_range)
    subset_path = tmp_path / "subset.grib2"
    manifest_path = tmp_path / "subset.manifest.csv"
    selection_manifest_path = tmp_path / "subset.selection.csv"
    out = hrrr_fetch.download_selected_records(
        grib_url="https://example.com/hrrr.grib2",
        idx_url="https://example.com/hrrr.grib2.idx",
        subset_path=subset_path,
        manifest_path=manifest_path,
        selection_manifest_path=selection_manifest_path,
        patterns=["TMP", "DPT", "RH"],
        selected=selected,
        overwrite=True,
    )

    assert out == subset_path
    assert calls == [(0, 199), (70000, 70049)]
    assert subset_path.stat().st_size == 250
    manifest_text = manifest_path.read_text()
    assert "record_number" in manifest_text
    assert "1,0,99,TMP" in manifest_text
    assert "2,100,199,DPT" in manifest_text
    assert "3,70000,70049,RH" in manifest_text
    assert hrrr_fetch.manifest_matches_selection(
        selection_manifest_path,
        expected_signature=hrrr_fetch.selection_signature(
            grib_url="https://example.com/hrrr.grib2",
            patterns=["TMP", "DPT", "RH"],
            selected=selected,
        ),
    )


def test_process_reduced_grib_emits_canonical_schema_and_provenance(monkeypatch):
    datasets = build_test_datasets()

    def fake_open_group_dataset(_path, group):
        return datasets[group.name].copy(deep=True)

    monkeypatch.setattr(hrrr_pipeline, "open_group_dataset", fake_open_group_dataset)

    inventory = [
        "1:0:d=2023010112:TMP:2 m above ground:anl:",
        "2:100:d=2023010112:DPT:2 m above ground:anl:",
        "3:200:d=2023010112:RH:2 m above ground:anl:",
        "4:300:d=2023010112:UGRD:10 m above ground:anl:",
        "5:400:d=2023010112:VGRD:10 m above ground:anl:",
        "6:500:d=2023010112:PWAT:entire atmosphere (considered as a single layer):anl:",
        "7:600:d=2023010112:TMP:850 mb:anl:",
        "8:700:d=2023010112:UGRD:850 mb:anl:",
        "9:800:d=2023010112:VGRD:850 mb:anl:",
        "10:900:d=2023010112:HGT:850 mb:anl:",
        "11:1000:d=2023010112:DPT:850 mb:anl:",
    ]
    task = make_task()

    result = hrrr_pipeline.process_reduced_grib(pathlib.Path("/tmp/fake.grib2"), inventory, task, "https://example.com/file.grib2")

    assert result.ok
    assert result.row is not None
    row = result.row
    assert row["source_model"] == "HRRR"
    assert row["source_product"] == "wrfsfcf"
    assert row["source_version"] == "hrrr-conus-wrfsfcf-public"
    assert row["fallback_used_any"] is False
    assert row["station_id"] == "KLGA"
    assert row["init_time_local"].endswith("-05:00")
    assert row["init_date_local"] == "2023-01-01"
    assert row["valid_date_local"] == "2023-01-01"
    assert row["settlement_lat"] == pytest.approx(40.7769)
    assert row["settlement_lon"] == pytest.approx(-73.8740)
    assert row["nearest_grid_lat"] == pytest.approx(40.7769)
    assert row["nearest_grid_lon"] == pytest.approx(-73.8740)

    assert row["tmp_2m_k"] == pytest.approx(datasets["near_surface_2m"]["t2m"].values[3, 3])
    assert row["tmp_2m_k_crop_mean"] == pytest.approx(np.mean(datasets["near_surface_2m"]["t2m"].values))
    assert row["tmp_2m_k_nb3_mean"] == pytest.approx(np.mean(datasets["near_surface_2m"]["t2m"].values[2:5, 2:5]))
    assert row["tmp_2m_k_nb7_mean"] == pytest.approx(np.mean(datasets["near_surface_2m"]["t2m"].values))
    assert "tmp_2m_k_nearest" not in row
    assert "tmp_2m_k_3x3_mean" not in row
    assert "tmp_2m_k_7x7_mean" not in row
    assert row["rh_850mb_pct"] is not None
    assert row["tmp_2m_f"] == pytest.approx((row["tmp_2m_k"] - 273.15) * 9.0 / 5.0 + 32.0)
    assert row["tmp_2m_f_crop_std"] == pytest.approx(row["tmp_2m_k_crop_std"] * 9.0 / 5.0)
    assert row["tmp_2m_f_nb3_std"] == pytest.approx(row["tmp_2m_k_nb3_std"] * 9.0 / 5.0)
    assert row["tmp_2m_f_nb7_std"] == pytest.approx(row["tmp_2m_k_nb7_std"] * 9.0 / 5.0)
    assert row["tmp_2m_f_nb3_gradient_west_east"] == pytest.approx(row["tmp_2m_k_nb3_gradient_west_east"] * 9.0 / 5.0)
    assert row["tmp_2m_f_nb3_gradient_south_north"] == pytest.approx(row["tmp_2m_k_nb3_gradient_south_north"] * 9.0 / 5.0)
    assert row["dpt_2m_f_crop_std"] == pytest.approx(row["dpt_2m_k_crop_std"] * 9.0 / 5.0)
    wind_speed_grid = np.hypot(datasets["near_surface_10m"]["u10"].values, datasets["near_surface_10m"]["v10"].values)
    assert row["wind_10m_speed_ms"] == pytest.approx(wind_speed_grid[3, 3])
    assert row["wind_10m_speed_ms_crop_mean"] == pytest.approx(np.mean(wind_speed_grid))
    assert row["wind_10m_speed_ms_nb3_mean"] == pytest.approx(np.mean(wind_speed_grid[2:5, 2:5]))
    assert row["wind_10m_speed_ms_nb3_std"] == pytest.approx(np.std(wind_speed_grid[2:5, 2:5]))
    assert "wind_10m_speed_ms_3x3_std" not in row
    assert row["wind_10m_speed_mph_crop_mean"] == pytest.approx(row["wind_10m_speed_ms_crop_mean"] * 2.2369362920544)
    assert row["wind_10m_direction_deg_nb7_mean"] is not None
    assert row["missing_optional_any"] is True
    assert row["missing_optional_fields_count"] == len(result.missing_fields)

    provenance = {entry["feature_name"]: entry for entry in result.provenance_rows}
    assert provenance["tmp_2m_k"]["source_version"] == "hrrr-conus-wrfsfcf-public"
    assert provenance["tmp_2m_k"]["nearest_grid_lat"] == pytest.approx(40.7769)
    assert provenance["tmp_2m_k"]["nearest_grid_lon"] == pytest.approx(-73.8740)
    assert provenance["tmp_2m_k"]["forecast_hour"] == 0
    assert provenance["tmp_2m_k"]["fallback_used"] is False
    assert provenance["tmp_2m_k"]["fallback_source_description"] is None
    assert provenance["tmp_2m_k"]["present_directly"] is True
    assert provenance["tmp_2m_k"]["derived"] is False
    assert provenance["rh_850mb_pct"]["derived"] is True
    assert json.loads(provenance["rh_850mb_pct"]["source_feature_names"]) == ["tmp_850mb_k", "dpt_850mb_k_support"]
    assert provenance["tmp_2m_f"]["derived"] is True
    assert json.loads(provenance["tmp_2m_f"]["source_feature_names"]) == ["tmp_2m_k"]
    assert provenance["wind_10m_speed_ms"]["derived"] is True
    assert provenance["cape_surface_j_kg"]["missing_optional"] is True


def test_process_reduced_grib_can_emit_legacy_aliases_when_requested(monkeypatch):
    datasets = build_test_datasets()

    def fake_open_group_dataset(_path, group):
        return datasets[group.name].copy(deep=True)

    monkeypatch.setattr(hrrr_pipeline, "open_group_dataset", fake_open_group_dataset)
    inventory = [
        "1:0:d=2023010112:TMP:2 m above ground:anl:",
        "2:100:d=2023010112:DPT:2 m above ground:anl:",
        "3:200:d=2023010112:RH:2 m above ground:anl:",
    ]
    task = make_task()
    result = hrrr_pipeline.process_reduced_grib(
        pathlib.Path("/tmp/fake.grib2"),
        inventory,
        task,
        "https://example.com/file.grib2",
        include_legacy_aliases=True,
    )
    assert result.ok
    assert result.row is not None
    assert result.row["tmp_2m_k_nearest"] == result.row["tmp_2m_k"]
    assert result.row["tmp_2m_k_3x3_mean"] == result.row["tmp_2m_k_nb3_mean"]


def test_select_2d_slice_uses_task_step_for_batch_grib():
    lat_grid, lon_grid = make_lat_lon_grid()
    values = np.stack(
        [
            np.full((7, 7), 280.0),
            np.full((7, 7), 285.0),
        ],
        axis=0,
    )
    data_array = xr.DataArray(
        values,
        dims=("step", "y", "x"),
        coords={
            "step": pd.to_timedelta([0, 1], unit="h"),
            "latitude": (("y", "x"), lat_grid),
            "longitude": (("y", "x"), lon_grid),
        },
    )
    task = make_task(forecast_hour=1)

    selected = hrrr_pipeline.select_2d_slice(data_array, task=task)

    assert selected.ndim == 2
    assert float(selected.values[3, 3]) == pytest.approx(285.0)


def test_select_2d_slice_requires_task_match_for_batch_grib():
    lat_grid, lon_grid = make_lat_lon_grid()
    values = np.stack(
        [
            np.full((7, 7), 280.0),
            np.full((7, 7), 285.0),
        ],
        axis=0,
    )
    data_array = xr.DataArray(
        values,
        dims=("step", "y", "x"),
        coords={
            "step": pd.to_timedelta([0, 1], unit="h"),
            "latitude": (("y", "x"), lat_grid),
            "longitude": (("y", "x"), lon_grid),
        },
    )
    task = make_task(forecast_hour=2)

    with pytest.raises(ValueError, match="Unable to select forecast_hour=2"):
        hrrr_pipeline.select_2d_slice(data_array, task=task, require_task_match=True)


def test_process_reduced_grib_rejects_ambiguous_inventory_provenance(monkeypatch):
    datasets = build_test_datasets()

    def fake_open_group_dataset(_path, group):
        return datasets[group.name].copy(deep=True)

    monkeypatch.setattr(hrrr_pipeline, "open_group_dataset", fake_open_group_dataset)

    inventory = [
        "1:0:d=2023010112:TMP:2 m above ground:anl:",
        "2:100:d=2023010112:TMP:2 m above ground:1 hour fcst:",
        "3:200:d=2023010112:DPT:2 m above ground:anl:",
        "4:300:d=2023010112:RH:2 m above ground:anl:",
    ]
    task = make_task()

    result = hrrr_pipeline.process_reduced_grib(pathlib.Path("/tmp/fake.grib2"), inventory, task, "https://example.com/file.grib2")

    assert result.ok is False
    assert result.message is not None
    assert "Ambiguous inventory lines for tmp_2m_k" in result.message


def test_open_reduced_grib_group_datasets_closes_partial_opens_on_failure(monkeypatch, tmp_path):
    closed: list[str] = []

    class FakeDataset:
        def __init__(self, name):
            self.name = name

        def close(self):
            closed.append(self.name)

    def fake_open_group_dataset(_path, group, *, indexpath):
        if group.name == "near_surface_10m":
            raise RuntimeError("boom")
        return FakeDataset(group.name)

    monkeypatch.setattr(hrrr_pipeline, "open_group_dataset", fake_open_group_dataset)
    inventory = [
        "1:0:d=2023010112:TMP:2 m above ground:anl:",
        "2:100:d=2023010112:UGRD:10 m above ground:anl:",
    ]

    with pytest.raises(RuntimeError, match="cfgrib open failed for group near_surface_10m"):
        hrrr_pipeline.open_reduced_grib_group_datasets(
            pathlib.Path("/tmp/fake.grib2"),
            inventory,
            diagnostics={},
            cfgrib_index_dir=tmp_path,
        )

    assert closed == ["near_surface_2m"]


def test_find_inventory_line_for_apcp_prefers_hourly_accum_over_day_accum():
    inventory_lines = [
        "32:1685494:d=2025041100:APCP:surface:0-1 day acc fcst:",
        "33:1758471:d=2025041100:APCP:surface:23-24 hour acc fcst:",
    ]

    selected = hrrr_pipeline.find_inventory_line("apcp_surface_kg_m2", inventory_lines)

    assert selected == "33:1758471:d=2025041100:APCP:surface:23-24 hour acc fcst:"


def test_find_inventory_line_for_apcp_prefers_hourly_accum_over_cumulative_hour_accum():
    inventory_lines = [
        "32:1685494:d=2025041100:APCP:surface:0-2 hour acc fcst:",
        "33:1758471:d=2025041100:APCP:surface:1-2 hour acc fcst:",
        "34:1800000:d=2025041100:APCP:surface:2 hour acc fcst:",
    ]

    selected = hrrr_pipeline.find_inventory_line("apcp_surface_kg_m2", inventory_lines)

    assert selected == "34:1800000:d=2025041100:APCP:surface:2 hour acc fcst:"


def test_inventory_line_forecast_hour_handles_day_accumulation():
    assert hrrr_pipeline.inventory_line_forecast_hour(
        "32:1685494:d=2025041105:APCP:surface:0-0 day acc fcst:"
    ) == 0
    assert hrrr_pipeline.inventory_line_forecast_hour(
        "33:1685494:d=2025041105:APCP:surface:0-1 day acc fcst:"
    ) == 24


def test_run_month_writes_wide_and_provenance_artifacts(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]

    def fake_process_task(task, **_kwargs):
        row = {
            "task_key": task.key,
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "fallback_used_any": False,
            "station_id": "KLGA",
            "target_date_local": task.target_date_local,
            "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
            "forecast_hour": task.forecast_hour,
            "run_date_utc": task.run_date_utc,
            "cycle_hour_utc": task.cycle_hour_utc,
            "init_time_utc": task.init_time_utc,
            "init_time_local": task.init_time_local,
            "init_date_local": task.init_date_local,
            "valid_time_utc": task.valid_time_utc,
            "valid_time_local": task.valid_time_local,
            "valid_date_local": task.valid_date_local,
            "init_hour_local": task.init_hour_local,
            "valid_hour_local": task.valid_hour_local,
            "cycle_rank_desc": task.cycle_rank_desc,
            "selected_for_summary": task.selected_for_summary,
            "anchor_cycle_candidate": task.anchor_cycle_candidate,
            "settlement_lat": 40.7769,
            "settlement_lon": -73.8740,
            "crop_top_lat": 43.5,
            "crop_bottom_lat": 39.0,
            "crop_left_lon": 282.5,
            "crop_right_lon": 289.5,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "tmp_2m_k": 290.0 + task.forecast_hour,
            "tmp_2m_k_crop_mean": 289.0,
            "tmp_2m_k_nb3_mean": 290.0,
            "tmp_2m_k_nb7_mean": 291.0,
            "missing_optional_any": False,
            "missing_optional_fields_count": 0,
        }
        provenance_rows = [
            {
                "task_key": task.key,
                "source_model": "HRRR",
                "source_product": "wrfsfcf",
                "source_version": "hrrr-conus-wrfsfcf-public",
                "station_id": "KLGA",
                "init_time_utc": row["init_time_utc"],
                "init_time_local": row["init_time_local"],
                "init_date_local": row["init_date_local"],
                "valid_time_utc": row["valid_time_utc"],
                "valid_time_local": row["valid_time_local"],
                "valid_date_local": row["valid_date_local"],
                "forecast_hour": row["forecast_hour"],
                "nearest_grid_lat": row["nearest_grid_lat"],
                "nearest_grid_lon": row["nearest_grid_lon"],
                "feature_name": "tmp_2m_k",
                "output_column_base": "tmp_2m_k",
                "grib_short_name": "TMP",
                "grib_level_text": "2 m above ground",
                "grib_type_of_level": "heightAboveGround",
                "grib_step_type": "instant",
                "grib_step_text": "anl",
                "source_inventory_line": "1:0:d=2023010112:TMP:2 m above ground:anl:",
                "units": "K",
                "present_directly": True,
                "derived": False,
                "derivation_method": None,
                "source_feature_names": "[]",
                "missing_optional": False,
                "fallback_used": False,
                "fallback_source_description": None,
                "notes": None,
            }
        ]
        return hrrr_pipeline.TaskResult(True, task.key, row, provenance_rows, [], None)

    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )

    assert wrote == 2
    assert failed == 0
    wide_path = tmp_path / "2023-01.parquet"
    provenance_path = tmp_path / "2023-01.provenance.parquet"
    summary_path = tmp_path / "summary" / "2023-01.parquet"
    manifest_parquet_path = tmp_path / "2023-01.manifest.parquet"
    manifest = json.loads((tmp_path / "2023-01.manifest.json").read_text())

    assert wide_path.exists()
    assert provenance_path.exists()
    assert summary_path.exists()
    assert manifest_parquet_path.exists()
    assert manifest["complete"] is True
    assert manifest["wide_parquet_path"] == str(wide_path)
    assert manifest["provenance_path"] == str(provenance_path)
    assert manifest["summary_parquet_path"] == str(summary_path)
    assert manifest["manifest_parquet_path"] == str(manifest_parquet_path)
    assert not (tmp_path / "2023-01.rows.jsonl").exists()
    assert not (tmp_path / "2023-01.provenance.rows.jsonl").exists()


def test_run_month_batch_reduce_cycle_crops_once_for_cycle(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    reduce_calls: list[list[pathlib.Path]] = []
    open_calls: list[pathlib.Path] = []
    build_calls: list[tuple[pathlib.Path, str]] = []
    fake_group_datasets = []

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.forecast_hour}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, raw_paths, _batch_raw_path, batch_reduced_path):
        reduce_calls.append(list(raw_paths))
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (
            [
                "1:0:d=2023010112:TMP:2 m above ground:anl:",
                "2:100:d=2023010112:TMP:2 m above ground:1 hour fcst:",
            ],
            {"tmp_2m_k"},
            0.01,
            0.02,
            0.03,
        )

    def fake_open_reduced_grib_group_datasets(reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        open_calls.append(reduced_path)
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.123
        return fake_group_datasets

    def fake_build_task_result_from_open_datasets(group_datasets, _inventory, task, _grib_url, **kwargs):
        assert group_datasets is fake_group_datasets
        assert kwargs["filter_inventory_to_task_step"] is True
        build_calls.append((pathlib.Path(kwargs["diagnostics"]["batch_reduced_file_path"]), task.key))
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        progress_mode="log",
    )

    assert wrote == 2
    assert failed == 0
    assert len(reduce_calls) == 1
    assert len(reduce_calls[0]) == 2
    assert len(open_calls) == 1
    assert len(build_calls) == 2
    assert len({path for path, _task_key in build_calls}) == 1
    manifest = json.loads((tmp_path / "2023-01.manifest.json").read_text())
    assert manifest["batch_reduce_mode"] == "cycle"
    manifest_df = pd.read_parquet(tmp_path / "2023-01.manifest.parquet")
    assert set(manifest_df["batch_reduce_mode"]) == {"cycle"}
    assert set(manifest_df["batch_cycle_cfgrib_open_seconds"]) == {0.123}
    assert set(manifest_df["timing_cfgrib_open_seconds"]) == {0.0615}
    assert manifest_df["timing_cleanup_seconds"].notna().all()


def test_run_month_batch_reduce_cycle_uses_download_workers(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=hour, valid_time_utc=f"2023-01-01T{12 + hour:02d}:00:00+00:00")
        for hour in range(4)
    ]
    active_downloads = 0
    max_active_downloads = 0
    lock = threading.Lock()

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        nonlocal active_downloads, max_active_downloads
        with lock:
            active_downloads += 1
            max_active_downloads = max(max_active_downloads, active_downloads)
        time.sleep(0.05)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.forecast_hour}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        with lock:
            active_downloads -= 1
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, _raw_paths, _batch_raw_path, batch_reduced_path):
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.01, 0.02, 0.03)

    def fake_open_reduced_grib_group_datasets(_reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.01
        return []

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=4,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        download_workers=4,
        reduce_workers=1,
        extract_workers=1,
        progress_mode="log",
    )

    assert wrote == 4
    assert failed == 0
    assert max_active_downloads > 1


def test_run_month_batch_reduce_cycle_caps_reduce_workers(tmp_path, monkeypatch):
    tasks = [
        make_task(run_date_utc="2023-01-01", cycle_hour_utc=0, forecast_hour=0, valid_time_utc="2023-01-01T00:00:00+00:00"),
        make_task(run_date_utc="2023-01-01", cycle_hour_utc=1, forecast_hour=0, valid_time_utc="2023-01-01T01:00:00+00:00"),
    ]
    active_reduces = 0
    max_active_reduces = 0
    lock = threading.Lock()

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.cycle_hour_utc}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, _raw_paths, _batch_raw_path, batch_reduced_path):
        nonlocal active_reduces, max_active_reduces
        with lock:
            active_reduces += 1
            max_active_reduces = max(max_active_reduces, active_reduces)
        time.sleep(0.05)
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        with lock:
            active_reduces -= 1
        return (["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.01, 0.02, 0.03)

    def fake_open_reduced_grib_group_datasets(_reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.01
        return []

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=4,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        download_workers=4,
        reduce_workers=1,
        extract_workers=1,
        progress_mode="log",
    )

    assert wrote == 2
    assert failed == 0
    assert max_active_reduces == 1


def test_run_month_batch_reduce_cycle_caps_extract_workers(tmp_path, monkeypatch):
    tasks = [
        make_task(run_date_utc="2023-01-01", cycle_hour_utc=0, forecast_hour=0, valid_time_utc="2023-01-01T00:00:00+00:00"),
        make_task(run_date_utc="2023-01-01", cycle_hour_utc=1, forecast_hour=0, valid_time_utc="2023-01-01T01:00:00+00:00"),
    ]
    active_extracts = 0
    max_active_extracts = 0
    lock = threading.Lock()

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.cycle_hour_utc}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, _raw_paths, _batch_raw_path, batch_reduced_path):
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.01, 0.02, 0.03)

    def fake_open_reduced_grib_group_datasets(_reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        nonlocal active_extracts, max_active_extracts
        with lock:
            active_extracts += 1
            max_active_extracts = max(max_active_extracts, active_extracts)
        time.sleep(0.05)
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.01
        with lock:
            active_extracts -= 1
        return []

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=4,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        download_workers=4,
        reduce_workers=2,
        extract_workers=1,
        progress_mode="log",
    )

    assert wrote == 2
    assert failed == 0
    assert max_active_extracts == 1


def test_run_month_batch_reduce_cycle_retries_manifest_failures(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    month_id = "2023-01"
    manifest = {
        "month": month_id,
        "expected_task_count": 2,
        "expected_task_keys": [task.key for task in tasks],
        "completed_task_keys": [],
        "failure_reasons": {task.key: "previous failure" for task in tasks},
        "missing_fields": {},
        "source_model": "HRRR",
        "source_product": "wrfsfcf",
        "source_version": "hrrr-conus-wrfsfcf-public",
        "wide_parquet_path": str(tmp_path / f"{month_id}.parquet"),
        "provenance_path": str(tmp_path / f"{month_id}.provenance.parquet"),
        "summary_parquet_path": None,
        "manifest_parquet_path": str(tmp_path / f"{month_id}.manifest.parquet"),
        "row_buffer_path": str(tmp_path / f"{month_id}.rows.jsonl"),
        "provenance_buffer_path": str(tmp_path / f"{month_id}.provenance.rows.jsonl"),
        "keep_downloads": False,
        "keep_reduced": False,
        "complete": False,
    }
    (tmp_path / f"{month_id}.manifest.json").write_text(json.dumps(manifest))
    reduce_calls: list[list[pathlib.Path]] = []

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.forecast_hour}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, raw_paths, _batch_raw_path, batch_reduced_path):
        reduce_calls.append(list(raw_paths))
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (
            [
                "1:0:d=2023010112:TMP:2 m above ground:anl:",
                "2:100:d=2023010112:TMP:2 m above ground:1 hour fcst:",
            ],
            {"tmp_2m_k"},
            0.01,
            0.02,
            0.03,
        )

    def fake_open_reduced_grib_group_datasets(_reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.01
        return []

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        month_id,
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        progress_mode="log",
    )

    assert wrote == 2
    assert failed == 0
    assert len(reduce_calls) == 1
    manifest_after = json.loads((tmp_path / f"{month_id}.manifest.json").read_text())
    assert manifest_after["failure_reasons"] == {}


def test_run_month_batch_reduce_cycle_retries_transient_download(tmp_path, monkeypatch):
    task = make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00")
    attempts = {"download": 0}

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(_task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        attempts["download"] += 1
        if attempts["download"] == 1:
            raise TimeoutError("temporary")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(b"raw")
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, _raw_paths, _batch_raw_path, batch_reduced_path):
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.01, 0.02, 0.03)

    def fake_open_reduced_grib_group_datasets(_reduced_path, _inventory, *, diagnostics=None, **_kwargs):
        if diagnostics is not None:
            diagnostics["timing_cfgrib_open_seconds"] = 0.01
        return []

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None, kwargs["diagnostics"])

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        [task],
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        progress_mode="log",
        max_task_attempts=2,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
    )

    assert wrote == 1
    assert failed == 0
    assert attempts["download"] == 2
    manifest_df = pd.read_parquet(tmp_path / "2023-01.manifest.parquet")
    assert bool(manifest_df.loc[0, "retry_recovered"]) is True


def test_run_month_batch_reduce_cycle_shared_open_failure_marks_cycle_tasks_failed(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    build_calls: list[str] = []

    class FakeFetchResult:
        head_used = False
        remote_file_size = None
        selected_record_count = 2
        merged_range_count = 1
        downloaded_range_bytes = 20
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_task_subset(task, *, raw_path, raw_manifest_path, raw_selection_manifest_path, **_kwargs):
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(f"raw-{task.forecast_hour}".encode("ascii"))
        raw_manifest_path.write_text("manifest")
        raw_selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2_for_batch(_wgrib2_path, _raw_paths, _batch_raw_path, batch_reduced_path):
        batch_reduced_path.parent.mkdir(parents=True, exist_ok=True)
        batch_reduced_path.write_bytes(b"batch-reduced")
        return (
            [
                "1:0:d=2023010112:TMP:2 m above ground:anl:",
                "2:100:d=2023010112:TMP:2 m above ground:1 hour fcst:",
            ],
            {"tmp_2m_k"},
            0.01,
            0.02,
            0.03,
        )

    def fake_open_reduced_grib_group_datasets(*_args, **_kwargs):
        raise RuntimeError("shared cfgrib failed")

    def fake_build_task_result_from_open_datasets(_group_datasets, _inventory, task, _grib_url, **_kwargs):
        build_calls.append(task.key)
        return hrrr_pipeline.TaskResult(True, task.key, make_minimal_hrrr_output_row_from_task(task), [], [], None)

    monkeypatch.setattr(hrrr_pipeline, "download_task_subset", fake_download_task_subset)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2_for_batch", fake_reduce_grib2_for_batch)
    monkeypatch.setattr(hrrr_pipeline, "open_reduced_grib_group_datasets", fake_open_reduced_grib_group_datasets)
    monkeypatch.setattr(hrrr_pipeline, "build_task_result_from_open_datasets", fake_build_task_result_from_open_datasets)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        batch_reduce_mode="cycle",
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
    )

    assert wrote == 0
    assert failed == 2
    assert build_calls == []
    manifest_after = json.loads((tmp_path / "2023-01.manifest.json").read_text())
    assert manifest_after["complete"] is False
    assert set(manifest_after["failure_reasons"]) == {task.key for task in tasks}
    assert all("shared cfgrib failed" in reason for reason in manifest_after["failure_reasons"].values())


def test_run_month_defers_manifest_flush_until_finalize(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    save_calls: list[str] = []
    parquet_calls: list[str] = []

    def fake_process_task(task, **_kwargs):
        row = {
            "task_key": task.key,
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "fallback_used_any": False,
            "station_id": "KLGA",
            "target_date_local": task.target_date_local,
            "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
            "forecast_hour": task.forecast_hour,
            "run_date_utc": task.run_date_utc,
            "cycle_hour_utc": task.cycle_hour_utc,
            "init_time_utc": task.init_time_utc,
            "init_time_local": task.init_time_local,
            "init_date_local": task.init_date_local,
            "valid_time_utc": task.valid_time_utc,
            "valid_time_local": task.valid_time_local,
            "valid_date_local": task.valid_date_local,
            "init_hour_local": task.init_hour_local,
            "valid_hour_local": task.valid_hour_local,
            "cycle_rank_desc": task.cycle_rank_desc,
            "selected_for_summary": task.selected_for_summary,
            "anchor_cycle_candidate": task.anchor_cycle_candidate,
            "settlement_lat": 40.7769,
            "settlement_lon": -73.8740,
            "crop_top_lat": 43.5,
            "crop_bottom_lat": 39.0,
            "crop_left_lon": 282.5,
            "crop_right_lon": 289.5,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "tmp_2m_k": 290.0 + task.forecast_hour,
            "tmp_2m_k_crop_mean": 289.0,
            "tmp_2m_k_nb3_mean": 290.0,
            "tmp_2m_k_nb7_mean": 291.0,
            "missing_optional_any": False,
            "missing_optional_fields_count": 0,
        }
        return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None)

    original_save_manifest = hrrr_pipeline.save_manifest
    original_write_manifest_parquet = hrrr_pipeline.write_manifest_parquet

    def recording_save_manifest(path, manifest):
        save_calls.append(str(path))
        return original_save_manifest(path, manifest)

    def recording_write_manifest_parquet(output_dir, month_id, manifest):
        parquet_calls.append(f"{output_dir}/{month_id}")
        return original_write_manifest_parquet(output_dir, month_id, manifest)

    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)
    monkeypatch.setattr(hrrr_pipeline, "save_manifest", recording_save_manifest)
    monkeypatch.setattr(hrrr_pipeline, "write_manifest_parquet", recording_write_manifest_parquet)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )

    assert wrote == 2
    assert failed == 0
    assert len(save_calls) == 1
    assert len(parquet_calls) == 1


def test_run_month_resumes_from_existing_buffers(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    month_id = "2023-01"
    output_dir = tmp_path
    first_task = tasks[0]
    second_task = tasks[1]

    manifest = {
        "month": month_id,
        "expected_task_count": 2,
        "expected_task_keys": [task.key for task in tasks],
        "completed_task_keys": [first_task.key],
        "failure_reasons": {},
        "missing_fields": {first_task.key: []},
        "source_model": "HRRR",
        "source_product": "wrfsfcf",
        "source_version": "hrrr-conus-wrfsfcf-public",
        "wide_parquet_path": str(output_dir / f"{month_id}.parquet"),
        "provenance_path": str(output_dir / f"{month_id}.provenance.parquet"),
        "summary_parquet_path": None,
        "manifest_parquet_path": str(output_dir / f"{month_id}.manifest.parquet"),
        "row_buffer_path": str(output_dir / f"{month_id}.rows.jsonl"),
        "provenance_buffer_path": str(output_dir / f"{month_id}.provenance.rows.jsonl"),
        "keep_downloads": False,
        "keep_reduced": False,
        "complete": False,
    }
    (output_dir / f"{month_id}.manifest.json").write_text(json.dumps(manifest))

    first_row = {
        "task_key": first_task.key,
        "source_model": "HRRR",
        "source_product": "wrfsfcf",
        "source_version": "hrrr-conus-wrfsfcf-public",
        "fallback_used_any": False,
        "station_id": "KLGA",
        "target_date_local": first_task.target_date_local,
        "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
        "forecast_hour": first_task.forecast_hour,
        "run_date_utc": first_task.run_date_utc,
        "cycle_hour_utc": first_task.cycle_hour_utc,
        "init_time_utc": first_task.init_time_utc,
        "init_time_local": first_task.init_time_local,
        "init_date_local": first_task.init_date_local,
        "valid_time_utc": first_task.valid_time_utc,
        "valid_time_local": first_task.valid_time_local,
        "valid_date_local": first_task.valid_date_local,
        "init_hour_local": first_task.init_hour_local,
        "valid_hour_local": first_task.valid_hour_local,
        "cycle_rank_desc": first_task.cycle_rank_desc,
        "selected_for_summary": first_task.selected_for_summary,
        "anchor_cycle_candidate": first_task.anchor_cycle_candidate,
        "settlement_lat": 40.7769,
        "settlement_lon": -73.8740,
        "crop_top_lat": 43.5,
        "crop_bottom_lat": 39.0,
        "crop_left_lon": 282.5,
        "crop_right_lon": 289.5,
        "nearest_grid_lat": 40.7769,
        "nearest_grid_lon": -73.874,
        "tmp_2m_k": 290.0,
        "missing_optional_any": False,
        "missing_optional_fields_count": 0,
    }
    (output_dir / f"{month_id}.rows.jsonl").write_text(json.dumps(first_row) + "\n")

    first_provenance = {
        "task_key": first_task.key,
        "source_model": "HRRR",
        "source_product": "wrfsfcf",
        "source_version": "hrrr-conus-wrfsfcf-public",
        "station_id": "KLGA",
        "init_time_utc": first_row["init_time_utc"],
        "init_time_local": first_row["init_time_local"],
        "init_date_local": first_row["init_date_local"],
        "valid_time_utc": first_row["valid_time_utc"],
        "valid_time_local": first_row["valid_time_local"],
        "valid_date_local": first_row["valid_date_local"],
        "forecast_hour": first_row["forecast_hour"],
        "nearest_grid_lat": first_row["nearest_grid_lat"],
        "nearest_grid_lon": first_row["nearest_grid_lon"],
        "feature_name": "tmp_2m_k",
        "output_column_base": "tmp_2m_k",
        "grib_short_name": "TMP",
        "grib_level_text": "2 m above ground",
        "grib_type_of_level": "heightAboveGround",
        "grib_step_type": "instant",
        "grib_step_text": "anl",
        "source_inventory_line": "1:0:d=2023010112:TMP:2 m above ground:anl:",
        "units": "K",
        "present_directly": True,
        "derived": False,
        "derivation_method": None,
        "source_feature_names": "[]",
        "missing_optional": False,
        "fallback_used": False,
        "fallback_source_description": None,
        "notes": None,
    }
    (output_dir / f"{month_id}.provenance.rows.jsonl").write_text(json.dumps(first_provenance) + "\n")

    def fake_process_task(task, **_kwargs):
        assert task.key == second_task.key
        row = dict(first_row)
        row["task_key"] = task.key
        row["forecast_hour"] = task.forecast_hour
        row["valid_time_utc"] = task.valid_time_utc
        row["valid_time_local"] = task.valid_time_local
        row["valid_date_local"] = task.valid_date_local
        row["valid_hour_local"] = task.valid_hour_local
        row["tmp_2m_k"] = 291.0
        provenance = dict(first_provenance)
        provenance["task_key"] = task.key
        provenance["valid_time_utc"] = row["valid_time_utc"]
        provenance["valid_time_local"] = row["valid_time_local"]
        provenance["valid_date_local"] = row["valid_date_local"]
        provenance["forecast_hour"] = row["forecast_hour"]
        return hrrr_pipeline.TaskResult(True, task.key, row, [provenance], [], None)

    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)

    wrote, failed = hrrr_pipeline.run_month(
        month_id,
        tasks,
        wgrib2_path="wgrib2",
        output_dir=output_dir,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )

    assert wrote == 1
    assert failed == 0
    wide_df = pd.read_parquet(output_dir / f"{month_id}.parquet")
    provenance_df = pd.read_parquet(output_dir / f"{month_id}.provenance.parquet")
    summary_df = pd.read_parquet(tmp_path / "summary" / f"{month_id}.parquet")
    saved_manifest = json.loads((output_dir / f"{month_id}.manifest.json").read_text())

    assert list(wide_df["task_key"]) == [first_task.key, second_task.key]
    assert sorted(provenance_df["task_key"].tolist()) == [first_task.key, second_task.key]
    assert saved_manifest["complete"] is True
    assert (output_dir / f"{month_id}.manifest.parquet").exists()
    assert len(summary_df) == 1
    assert sorted(saved_manifest["completed_task_keys"]) == [first_task.key, second_task.key]
    assert not (output_dir / f"{month_id}.rows.jsonl").exists()
    assert not (output_dir / f"{month_id}.provenance.rows.jsonl").exists()


def test_run_month_incomplete_flushes_manifest_once(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    save_calls: list[str] = []
    parquet_calls: list[str] = []

    def fake_process_task(task, **_kwargs):
        if task.forecast_hour == 0:
            row = {
                "task_key": task.key,
                "source_model": "HRRR",
                "source_product": "wrfsfcf",
                "source_version": "hrrr-conus-wrfsfcf-public",
                "fallback_used_any": False,
                "station_id": "KLGA",
                "target_date_local": task.target_date_local,
                "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
                "forecast_hour": task.forecast_hour,
                "run_date_utc": task.run_date_utc,
                "cycle_hour_utc": task.cycle_hour_utc,
                "init_time_utc": task.init_time_utc,
                "init_time_local": task.init_time_local,
                "init_date_local": task.init_date_local,
                "valid_time_utc": task.valid_time_utc,
                "valid_time_local": task.valid_time_local,
                "valid_date_local": task.valid_date_local,
                "init_hour_local": task.init_hour_local,
                "valid_hour_local": task.valid_hour_local,
                "cycle_rank_desc": task.cycle_rank_desc,
                "selected_for_summary": task.selected_for_summary,
                "anchor_cycle_candidate": task.anchor_cycle_candidate,
                "settlement_lat": 40.7769,
                "settlement_lon": -73.8740,
                "crop_top_lat": 43.5,
                "crop_bottom_lat": 39.0,
                "crop_left_lon": 282.5,
                "crop_right_lon": 289.5,
                "nearest_grid_lat": 40.7769,
                "nearest_grid_lon": -73.874,
                "tmp_2m_k": 290.0,
                "tmp_2m_k_crop_mean": 289.0,
                "tmp_2m_k_nb3_mean": 290.0,
                "tmp_2m_k_nb7_mean": 291.0,
                "missing_optional_any": False,
                "missing_optional_fields_count": 0,
            }
            return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None)
        return hrrr_pipeline.TaskResult(False, task.key, None, [], [], "boom")

    original_save_manifest = hrrr_pipeline.save_manifest
    original_write_manifest_parquet = hrrr_pipeline.write_manifest_parquet

    def recording_save_manifest(path, manifest):
        save_calls.append(str(path))
        return original_save_manifest(path, manifest)

    def recording_write_manifest_parquet(output_dir, month_id, manifest):
        parquet_calls.append(f"{output_dir}/{month_id}")
        return original_write_manifest_parquet(output_dir, month_id, manifest)

    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)
    monkeypatch.setattr(hrrr_pipeline, "save_manifest", recording_save_manifest)
    monkeypatch.setattr(hrrr_pipeline, "write_manifest_parquet", recording_write_manifest_parquet)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )

    assert wrote == 1
    assert failed == 1
    assert len(save_calls) == 1
    assert len(parquet_calls) == 1


def test_run_month_progress_counts_successes_and_failures(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00"),
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
    ]
    reporter_instances: list[object] = []

    class RecordingReporter:
        def __init__(self, title, *, unit="task", total=None, mode="auto", stream=None, is_tty=None, **_kwargs):
            self.title = title
            self.unit = unit
            self.total = total
            self.mode = "log"
            self.groups = []
            self.metrics = []
            self.events = []
            self.closed = False
            reporter_instances.append(self)

        def set_metrics(self, **metrics):
            self.metrics.append(metrics)

        def upsert_group(self, group_id, **kwargs):
            self.groups.append((group_id, kwargs))

        def log_event(self, message, *, level="info"):
            self.events.append((level, message))

        def close(self, *, status=None):
            self.closed = True
            self.events.append(("close", status))

    def fake_process_task(task, **_kwargs):
        if task.forecast_hour == 0:
            row = {
                "task_key": task.key,
                "source_model": "HRRR",
                "source_product": "wrfsfcf",
                "source_version": "hrrr-conus-wrfsfcf-public",
                "fallback_used_any": False,
                "station_id": "KLGA",
                "target_date_local": task.target_date_local,
                "slice_policy": hrrr_pipeline.DEFAULT_SLICE_POLICY,
                "forecast_hour": task.forecast_hour,
                "run_date_utc": task.run_date_utc,
                "cycle_hour_utc": task.cycle_hour_utc,
                "init_time_utc": task.init_time_utc,
                "init_time_local": task.init_time_local,
                "init_date_local": task.init_date_local,
                "valid_time_utc": task.valid_time_utc,
                "valid_time_local": task.valid_time_local,
                "valid_date_local": task.valid_date_local,
                "init_hour_local": task.init_hour_local,
                "valid_hour_local": task.valid_hour_local,
                "cycle_rank_desc": task.cycle_rank_desc,
                "selected_for_summary": task.selected_for_summary,
                "anchor_cycle_candidate": task.anchor_cycle_candidate,
                "settlement_lat": 40.7769,
                "settlement_lon": -73.8740,
                "crop_top_lat": 43.5,
                "crop_bottom_lat": 39.0,
                "crop_left_lon": 282.5,
                "crop_right_lon": 289.5,
                "nearest_grid_lat": 40.7769,
                "nearest_grid_lon": -73.874,
                "tmp_2m_k": 290.0,
                "tmp_2m_k_crop_mean": 289.0,
                "tmp_2m_k_nb3_mean": 290.0,
                "tmp_2m_k_nb7_mean": 291.0,
                "missing_optional_any": False,
                "missing_optional_fields_count": 0,
            }
            return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None)
        return hrrr_pipeline.TaskResult(False, task.key, None, [], [], "boom")

    monkeypatch.setattr(hrrr_pipeline, "create_progress_reporter", RecordingReporter)
    monkeypatch.setattr(hrrr_pipeline, "process_task", fake_process_task)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        progress_mode="log",
    )

    assert wrote == 1
    assert failed == 1
    assert len(reporter_instances) == 1
    reporter = reporter_instances[0]
    assert reporter.total == 2
    assert reporter.title == "HRRR 2023-01"
    assert reporter.unit == "task"
    assert reporter.closed is True
    assert any(group_id == "2023-01" for group_id, _ in reporter.groups)


def test_build_phase_concurrency_limits_defaults_and_clamps():
    args = argparse.Namespace(download_workers=None, reduce_workers=None, extract_workers=None)
    limits = hrrr_pipeline.build_phase_concurrency_limits(max_workers=6, args=args)
    assert limits.download_workers == 6
    assert limits.reduce_workers == 3
    assert limits.extract_workers == 4

    explicit = argparse.Namespace(download_workers=10, reduce_workers=0, extract_workers=2)
    limits = hrrr_pipeline.build_phase_concurrency_limits(max_workers=5, args=explicit)
    assert limits.download_workers == 5
    assert limits.reduce_workers == 1
    assert limits.extract_workers == 2


def test_process_task_phase_caps_limit_download_reduce_and_extract(tmp_path, monkeypatch):
    task = make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00")
    phase_limits = hrrr_pipeline.build_phase_concurrency_limits(
        max_workers=3,
        args=argparse.Namespace(download_workers=2, reduce_workers=1, extract_workers=1),
    )

    phase_counts = defaultdict(int)
    phase_peaks = defaultdict(int)
    phase_lock = threading.Lock()

    def run_in_phase(phase_name: str, fn):
        with phase_lock:
            phase_counts[phase_name] += 1
            phase_peaks[phase_name] = max(phase_peaks[phase_name], phase_counts[phase_name])
        try:
            time.sleep(0.05)
            return fn()
        finally:
            with phase_lock:
                phase_counts[phase_name] -= 1

    class FakeFetchResult:
        head_used = False
        remote_file_size = 10
        selected_record_count = 1
        merged_range_count = 1
        downloaded_range_bytes = 10
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_subset_for_inventory_patterns(*, subset_path, manifest_path, selection_manifest_path, **_kwargs):
        def inner():
            subset_path.parent.mkdir(parents=True, exist_ok=True)
            subset_path.write_bytes(b"subset")
            manifest_path.write_text("manifest")
            selection_manifest_path.write_text("selection")
            return FakeFetchResult()
        return run_in_phase("download", inner)

    def fake_reduce_grib2(_wgrib2_path, _raw_path, reduced_path):
        def inner():
            reduced_path.parent.mkdir(parents=True, exist_ok=True)
            reduced_path.write_bytes(b"reduced")
            return (["inventory"], "cmd", 0.01, 0.02)
        return run_in_phase("reduce", inner)

    def fake_process_reduced_grib(_reduced_path, _reduced_inventory, task, _grib_url, **_kwargs):
        def inner():
            return hrrr_pipeline.TaskResult(
                ok=True,
                task_key=task.key,
                row={"task_key": task.key},
                provenance_rows=[],
                missing_fields=[],
                diagnostics={},
            )
        return run_in_phase("extract", inner)

    monkeypatch.setattr(hrrr_pipeline, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(hrrr_pipeline, "build_reduced_reuse_signature", lambda **_kwargs: "sig")
    monkeypatch.setattr(hrrr_pipeline, "reduced_grib_reusable", lambda **_kwargs: False)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                hrrr_pipeline.process_task,
                task,
                wgrib2_path="wgrib2",
                download_dir=tmp_path / "downloads",
                reduced_dir=tmp_path / "reduced",
                phase_limits=phase_limits,
                keep_downloads=False,
                keep_reduced=False,
                include_legacy_aliases=False,
                reporter=None,
                max_attempts=1,
                retry_backoff_seconds=0.0,
                retry_max_backoff_seconds=0.0,
            )
            for task in [
                make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
                make_task(forecast_hour=2, valid_time_utc="2023-01-01T14:00:00+00:00"),
                make_task(forecast_hour=3, valid_time_utc="2023-01-01T15:00:00+00:00"),
            ]
        ]
        results = [future.result() for future in futures]

    assert all(result.ok for result in results)
    assert phase_peaks["download"] <= 2
    assert phase_peaks["reduce"] <= 1
    assert phase_peaks["extract"] <= 1


def test_process_task_reused_reduced_inventory_respects_reduce_cap(tmp_path, monkeypatch):
    phase_limits = hrrr_pipeline.build_phase_concurrency_limits(
        max_workers=3,
        args=argparse.Namespace(download_workers=3, reduce_workers=1, extract_workers=3),
    )
    phase_counts = defaultdict(int)
    phase_peaks = defaultdict(int)
    phase_lock = threading.Lock()

    def run_in_phase(phase_name: str, fn):
        with phase_lock:
            phase_counts[phase_name] += 1
            phase_peaks[phase_name] = max(phase_peaks[phase_name], phase_counts[phase_name])
        try:
            time.sleep(0.05)
            return fn()
        finally:
            with phase_lock:
                phase_counts[phase_name] -= 1

    class FakeFetchResult:
        head_used = False
        remote_file_size = 10
        selected_record_count = 1
        merged_range_count = 1
        downloaded_range_bytes = 10
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_subset_for_inventory_patterns(*, subset_path, manifest_path, selection_manifest_path, **_kwargs):
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_bytes(b"subset")
        manifest_path.write_text("manifest")
        selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_inventory_for_grib(_wgrib2_path, _reduced_path):
        return run_in_phase("reduce", lambda: ["inventory"])

    def fake_process_reduced_grib(_reduced_path, _reduced_inventory, task, _grib_url, **_kwargs):
        return hrrr_pipeline.TaskResult(
            ok=True,
            task_key=task.key,
            row={"task_key": task.key},
            provenance_rows=[],
            missing_fields=[],
            diagnostics={},
        )

    monkeypatch.setattr(hrrr_pipeline, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(hrrr_pipeline, "build_reduced_reuse_signature", lambda **_kwargs: "sig")
    monkeypatch.setattr(hrrr_pipeline, "reduced_grib_reusable", lambda **_kwargs: True)
    monkeypatch.setattr(hrrr_pipeline, "inventory_for_grib", fake_inventory_for_grib)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    tasks = [
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
        make_task(forecast_hour=2, valid_time_utc="2023-01-01T14:00:00+00:00"),
        make_task(forecast_hour=3, valid_time_utc="2023-01-01T15:00:00+00:00"),
    ]
    reduced_dir = tmp_path / "reduced"
    download_dir = tmp_path / "downloads"
    for task in tasks:
        reduced_path = hrrr_pipeline.path_for_reduced(reduced_dir, task)
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_bytes(b"reused")
        hrrr_pipeline.write_reduced_reuse_signature(reduced_path, "sig")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                hrrr_pipeline.process_task,
                task,
                wgrib2_path="wgrib2",
                download_dir=download_dir,
                reduced_dir=reduced_dir,
                phase_limits=phase_limits,
                keep_downloads=True,
                keep_reduced=True,
                include_legacy_aliases=False,
                reporter=None,
                max_attempts=1,
                retry_backoff_seconds=0.0,
                retry_max_backoff_seconds=0.0,
            )
            for task in tasks
        ]
        results = [future.result() for future in futures]

    assert all(result.ok for result in results)
    assert phase_peaks["reduce"] <= 1


def test_run_month_pause_stops_new_task_admission_and_persists_incomplete_manifest(tmp_path, monkeypatch):
    tasks = [
        make_task(forecast_hour=1, valid_time_utc="2023-01-01T13:00:00+00:00"),
        make_task(forecast_hour=2, valid_time_utc="2023-01-01T14:00:00+00:00"),
    ]

    class FakeFetchResult:
        head_used = False
        remote_file_size = 10
        selected_record_count = 1
        merged_range_count = 1
        downloaded_range_bytes = 10
        timing_idx_fetch_seconds = 0.0
        timing_idx_parse_seconds = 0.0
        timing_head_seconds = 0.0
        timing_range_download_seconds = 0.0

    def fake_download_subset_for_inventory_patterns(*, subset_path, manifest_path, selection_manifest_path, **_kwargs):
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_bytes(b"subset")
        manifest_path.write_text("manifest")
        selection_manifest_path.write_text("selection")
        return FakeFetchResult()

    def fake_reduce_grib2(_wgrib2_path, _raw_path, reduced_path):
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_bytes(b"reduced")
        return (["inventory"], "cmd", 0.01, 0.02)

    def fake_process_reduced_grib(_reduced_path, _reduced_inventory, task, _grib_url, **_kwargs):
        return hrrr_pipeline.TaskResult(
            ok=True,
            task_key=task.key,
            row={"task_key": task.key},
            provenance_rows=[],
            missing_fields=[],
            diagnostics={},
        )

    monkeypatch.setattr(hrrr_pipeline, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(hrrr_pipeline, "build_reduced_reuse_signature", lambda **_kwargs: "sig")
    monkeypatch.setattr(hrrr_pipeline, "reduced_grib_reusable", lambda **_kwargs: False)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    reporter_instances = []

    def auto_pause_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs)
        original_start_worker = reporter.start_worker
        triggered = {"value": False}

        def start_worker(*start_args, **start_kwargs):
            original_start_worker(*start_args, **start_kwargs)
            if not triggered["value"]:
                triggered["value"] = True
                reporter.request_pause(reason="operator")

        reporter.start_worker = start_worker  # type: ignore[method-assign]
        reporter_instances.append(reporter)
        return reporter

    monkeypatch.setattr(hrrr_pipeline, "create_progress_reporter", auto_pause_reporter)

    wrote, failed = hrrr_pipeline.run_month(
        "2023-01",
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
    )

    manifest = json.loads((tmp_path / "2023-01.manifest.json").read_text())
    assert wrote == 1
    assert failed == 0
    assert manifest["complete"] is False
    assert manifest["completed_task_keys"] == [tasks[0].key]
    assert reporter_instances[0].is_paused() is True


def test_run_month_skip_does_not_create_progress_bar(tmp_path, monkeypatch):
    tasks = [make_task(forecast_hour=0, valid_time_utc="2023-01-01T12:00:00+00:00")]
    month_id = "2023-01"

    manifest = {
        "month": month_id,
        "expected_task_count": 1,
        "expected_task_keys": [tasks[0].key],
        "completed_task_keys": [tasks[0].key],
        "failure_reasons": {},
        "missing_fields": {tasks[0].key: []},
        "source_model": "HRRR",
        "source_product": "wrfsfcf",
        "source_version": "hrrr-conus-wrfsfcf-public",
        "wide_parquet_path": str(tmp_path / f"{month_id}.parquet"),
        "provenance_path": str(tmp_path / f"{month_id}.provenance.parquet"),
        "summary_parquet_path": str(tmp_path / "summary" / f"{month_id}.parquet"),
        "manifest_parquet_path": str(tmp_path / f"{month_id}.manifest.parquet"),
        "row_buffer_path": str(tmp_path / f"{month_id}.rows.jsonl"),
        "provenance_buffer_path": str(tmp_path / f"{month_id}.provenance.rows.jsonl"),
        "keep_downloads": False,
        "keep_reduced": False,
        "complete": True,
    }
    (tmp_path / f"{month_id}.manifest.json").write_text(json.dumps(manifest))
    (tmp_path / f"{month_id}.parquet").write_text("")
    (tmp_path / f"{month_id}.provenance.parquet").write_text("")
    (tmp_path / f"{month_id}.manifest.parquet").write_text("")

    created = []

    class FailIfConstructedReporter:
        def __init__(self, *args, **kwargs):
            created.append((args, kwargs))
            raise AssertionError("Reporter should not be created for skipped months")

    monkeypatch.setattr(hrrr_pipeline, "create_progress_reporter", FailIfConstructedReporter)

    wrote, failed = hrrr_pipeline.run_month(
        month_id,
        tasks,
        wgrib2_path="wgrib2",
        output_dir=tmp_path,
        summary_output_dir=tmp_path / "summary",
        download_dir=tmp_path / "_downloads",
        reduced_dir=tmp_path / "_reduced",
        max_workers=1,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        progress_mode="log",
    )

    assert wrote == 0
    assert failed == 0
    assert created == []


def test_process_task_cleanup_flags_control_intermediate_retention(tmp_path, monkeypatch):
    task = make_task()

    def fake_download_subset_for_inventory_patterns(
        *,
        date,
        cycle,
        product,
        forecast_hour,
        source,
        patterns,
        subset_path,
        manifest_path,
        selection_manifest_path,
        overwrite=False,
    ):
        del date, cycle, product, forecast_hour, source, patterns, overwrite
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("raw")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("manifest")
        selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        selection_manifest_path.write_text("selection")
        return subset_path

    def fake_reduce_grib2(_wgrib2_path, _raw_path, reduced_path):
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_text("reduced")
        return ["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}

    def fake_process_reduced_grib(_reduced_path, _inventory, task, _url, *, include_legacy_aliases=False):
        row = {
            "task_key": task.key,
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "station_id": "KLGA",
            "init_time_utc": "2023-01-01T12:00:00+00:00",
            "valid_time_utc": "2023-01-01T12:00:00+00:00",
        }
        return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None)

    monkeypatch.setattr(
        hrrr_pipeline,
        "download_subset_for_inventory_patterns",
        fake_download_subset_for_inventory_patterns,
    )
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    download_dir = tmp_path / "downloads"
    reduced_dir = tmp_path / "reduced"

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )
    assert result.ok
    assert not hrrr_pipeline.path_for_raw(download_dir, task).exists()
    assert not hrrr_pipeline.path_for_reduced(reduced_dir, task).exists()

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=True,
        keep_reduced=True,
        include_legacy_aliases=False,
    )
    assert result.ok
    assert hrrr_pipeline.path_for_raw(download_dir, task).exists()
    assert hrrr_pipeline.path_for_reduced(reduced_dir, task).exists()


def test_process_task_reuses_reduced_grib_and_tracks_scratch_dir(tmp_path, monkeypatch):
    task = make_task()
    scratch_dir = tmp_path / "scratch"
    download_dir = scratch_dir / "downloads"
    reduced_dir = scratch_dir / "reduced"
    reduce_calls: list[Path] = []
    inventory_calls: list[Path] = []

    def fake_download_subset_for_inventory_patterns(
        *,
        date,
        cycle,
        product,
        forecast_hour,
        source,
        patterns,
        subset_path,
        manifest_path,
        selection_manifest_path,
        overwrite=False,
    ):
        del date, cycle, product, forecast_hour, source, patterns, overwrite
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("raw")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("manifest")
        selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        selection_manifest_path.write_text("selection")
        return subset_path

    def fake_reduce_grib2(_wgrib2_path, raw_path, reduced_path):
        reduce_calls.append(raw_path)
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_text("reduced")
        return ["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.1, 0.2

    def fake_inventory_for_grib(_wgrib2_path, reduced_path):
        inventory_calls.append(reduced_path)
        return ["1:0:d=2023010112:TMP:2 m above ground:anl:"]

    def fake_process_reduced_grib(_reduced_path, _inventory, task, _url, *, include_legacy_aliases=False, diagnostics=None, **kwargs):
        del kwargs
        row = {
            "task_key": task.key,
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "station_id": "KLGA",
            "init_time_utc": "2023-01-01T12:00:00+00:00",
            "valid_time_utc": "2023-01-01T12:00:00+00:00",
        }
        return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None, diagnostics or {})

    monkeypatch.setattr(
        hrrr_pipeline,
        "download_subset_for_inventory_patterns",
        fake_download_subset_for_inventory_patterns,
    )
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "inventory_for_grib", fake_inventory_for_grib)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    first = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=True,
        keep_reduced=True,
        include_legacy_aliases=False,
        scratch_dir=scratch_dir,
    )
    second = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=True,
        keep_reduced=True,
        include_legacy_aliases=False,
        scratch_dir=scratch_dir,
    )

    reduced_path = hrrr_pipeline.path_for_reduced(reduced_dir, task)
    assert first.ok
    assert second.ok
    assert first.diagnostics["reduced_reused"] is False
    assert second.diagnostics["reduced_reused"] is True
    assert second.diagnostics["scratch_dir"] == str(scratch_dir)
    assert second.diagnostics["cfgrib_index_strategy"] == hrrr_pipeline.DEFAULT_CFGRIB_INDEX_STRATEGY
    assert reduced_path.exists()
    assert hrrr_pipeline.reduced_signature_path(reduced_path).exists()
    assert hrrr_pipeline.reduced_inventory_path(reduced_path).exists()
    assert len(reduce_calls) == 1
    assert inventory_calls == []


def test_hrrr_reduced_reuse_signature_changes_with_selection_manifest_or_size(tmp_path):
    raw_path = tmp_path / "raw.grib2"
    raw_path.write_bytes(b"a" * 8)
    selection_manifest_path = tmp_path / "raw.selection.csv"
    selection_manifest_path.write_text("selection_signature\nsig-a\n")
    task = make_task()
    first = hrrr_pipeline.build_reduced_reuse_signature(
        task=task,
        raw_path=raw_path,
        selection_manifest_path=selection_manifest_path,
    )
    selection_manifest_path.write_text("selection_signature\nsig-b\n")
    second = hrrr_pipeline.build_reduced_reuse_signature(
        task=task,
        raw_path=raw_path,
        selection_manifest_path=selection_manifest_path,
    )
    raw_path.write_bytes(b"b" * 9)
    third = hrrr_pipeline.build_reduced_reuse_signature(
        task=task,
        raw_path=raw_path,
        selection_manifest_path=selection_manifest_path,
    )
    assert first != second
    assert second != third


def test_summarize_hrrr_diagnostics_reports_timings_and_size_ratio(tmp_path, capsys):
    manifest_path = tmp_path / "2023-01.manifest.parquet"
    pd.DataFrame(
        [
            {
                "timing_range_download_seconds": 1.0,
                "timing_wgrib_inventory_seconds": 2.0,
                "timing_reduce_seconds": 3.0,
                "timing_cfgrib_open_seconds": 4.0,
                "raw_file_size": 100.0,
                "reduced_file_size": 25.0,
            },
            {
                "timing_range_download_seconds": 2.0,
                "timing_wgrib_inventory_seconds": 4.0,
                "timing_reduce_seconds": 6.0,
                "timing_cfgrib_open_seconds": 8.0,
                "raw_file_size": 200.0,
                "reduced_file_size": 50.0,
            },
        ]
    ).to_parquet(manifest_path, index=False)

    monkey_args = argparse.Namespace(path=tmp_path, glob="*.manifest.parquet")
    original_parse_args = hrrr_diagnostics_summary.parse_args
    try:
        hrrr_diagnostics_summary.parse_args = lambda: monkey_args
        assert hrrr_diagnostics_summary.main() == 0
    finally:
        hrrr_diagnostics_summary.parse_args = original_parse_args

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_count"] == 1
    assert payload["metrics"]["timing_reduce_seconds"]["median"] == 4.5
    assert payload["size_ratio"]["median_reduced_to_raw"] == 0.25


def test_process_task_without_keep_reduced_skips_reuse_signature(tmp_path, monkeypatch):
    task = make_task()
    build_calls = {"count": 0}

    def fake_download_subset_for_inventory_patterns(*, subset_path, manifest_path, selection_manifest_path, **_kwargs):
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("raw")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "grib_url,record_number,byte_start,byte_end,variable,level,valid_desc\n"
            "https://example,1,0,99,TMP,2 m above ground,anl\n"
        )
        selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        selection_manifest_path.write_text("selection_signature\nsig\n")
        return subset_path

    def fake_build_reduced_reuse_signature(**_kwargs):
        build_calls["count"] += 1
        return "sig"

    def fake_reduce_grib2(_wgrib2_path, _raw_path, reduced_path, *, selected_inventory_lines=None):
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_text("reduced")
        assert selected_inventory_lines is None
        return (["1:0:d=2023010112:TMP:2 m above ground:anl"], {"tmp_2m_k"}, 0.0, 0.1)

    def fake_process_reduced_grib(_reduced_path, _inventory, task, _url, **_kwargs):
        return hrrr_pipeline.TaskResult(True, task.key, {"task_key": task.key}, [], [], None, {})

    monkeypatch.setattr(hrrr_pipeline, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(hrrr_pipeline, "build_reduced_reuse_signature", fake_build_reduced_reuse_signature)
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=tmp_path / "downloads",
        reduced_dir=tmp_path / "reduced",
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
    )

    assert result.ok
    assert build_calls["count"] == 0
    assert result.diagnostics.get("reduced_reuse_signature") is None


def test_process_task_overwrite_disables_reduced_reuse(tmp_path, monkeypatch):
    task = make_task()
    scratch_dir = tmp_path / "scratch"
    download_dir = scratch_dir / "downloads"
    reduced_dir = scratch_dir / "reduced"
    reduce_calls: list[Path] = []

    def fake_download_subset_for_inventory_patterns(
        *,
        date,
        cycle,
        product,
        forecast_hour,
        source,
        patterns,
        subset_path,
        manifest_path,
        selection_manifest_path,
        overwrite=False,
    ):
        del date, cycle, product, forecast_hour, source, patterns
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        subset_path.write_text("raw-overwrite" if overwrite else "raw")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text("manifest")
        selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        selection_manifest_path.write_text("selection")
        return subset_path

    def fake_reduce_grib2(_wgrib2_path, raw_path, reduced_path):
        reduce_calls.append(raw_path)
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_text("reduced")
        return ["1:0:d=2023010112:TMP:2 m above ground:anl:"], {"tmp_2m_k"}, 0.1, 0.2

    def fake_process_reduced_grib(_reduced_path, _inventory, task, _url, *, include_legacy_aliases=False, diagnostics=None, **kwargs):
        del kwargs
        row = {
            "task_key": task.key,
            "source_model": "HRRR",
            "source_product": "wrfsfcf",
            "source_version": "hrrr-conus-wrfsfcf-public",
            "station_id": "KLGA",
            "init_time_utc": "2023-01-01T12:00:00+00:00",
            "valid_time_utc": "2023-01-01T12:00:00+00:00",
        }
        return hrrr_pipeline.TaskResult(True, task.key, row, [], [], None, diagnostics or {})

    monkeypatch.setattr(
        hrrr_pipeline,
        "download_subset_for_inventory_patterns",
        fake_download_subset_for_inventory_patterns,
    )
    monkeypatch.setattr(hrrr_pipeline, "reduce_grib2", fake_reduce_grib2)
    monkeypatch.setattr(hrrr_pipeline, "process_reduced_grib", fake_process_reduced_grib)

    first = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=True,
        keep_reduced=True,
        include_legacy_aliases=False,
        scratch_dir=scratch_dir,
        overwrite=False,
    )
    second = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=download_dir,
        reduced_dir=reduced_dir,
        keep_downloads=True,
        keep_reduced=True,
        include_legacy_aliases=False,
        scratch_dir=scratch_dir,
        overwrite=True,
    )

    assert first.ok
    assert second.ok
    assert first.diagnostics["reduced_reused"] is False
    assert second.diagnostics["reduced_reused"] is False
    assert len(reduce_calls) == 2


def test_build_tasks_for_target_date_uses_local_overnight_window_across_dst():
    est_tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2023-01-15"))
    edt_tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2023-07-15"))
    spring_tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2024-03-10"))
    fall_tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2024-11-03"))

    for tasks in (est_tasks, edt_tasks, spring_tasks, fall_tasks):
        assert tasks
        assert all(task.target_date_local == tasks[0].target_date_local for task in tasks)
        assert max(task.cycle_rank_desc for task in tasks) <= 3
        assert max(task.valid_hour_local for task in tasks) <= 21
        assert min(task.valid_hour_local for task in tasks) >= 0
        retained_cycles = sorted({(task.run_date_utc, task.cycle_hour_utc, task.anchor_cycle_candidate) for task in tasks})
        assert len(retained_cycles) == 4
        assert sum(1 for _, _, is_anchor in retained_cycles if is_anchor) == 1

    spring_expected = hrrr_pipeline.target_day_expected_hours(pd.Timestamp("2024-03-10"))
    assert 2 not in spring_expected
    fall_expected = hrrr_pipeline.target_day_expected_hours(pd.Timestamp("2024-11-03"))
    assert 1 in fall_expected
    fall_expected_slots = hrrr_pipeline.target_day_expected_slots(pd.Timestamp("2024-11-03"))
    assert len(fall_expected_slots) > len(fall_expected)


def test_build_tasks_for_target_date_month_boundary_uses_target_month():
    tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2023-02-01"))
    assert tasks
    assert all(task.target_date_local == "2023-02-01" for task in tasks)
    assert hrrr_pipeline.month_id_for_task(tasks[0]) == "2023-02"
    assert any(task.init_date_local == "2023-01-31" for task in tasks)


def test_build_tasks_for_target_date_selects_real_anchor_cycle_with_full_coverage():
    tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2026-04-11"))
    expected_slots = hrrr_pipeline.target_day_expected_slots(pd.Timestamp("2026-04-11"))
    cycle_groups: dict[tuple[str, int], list[object]] = {}
    for task in tasks:
        cycle_groups.setdefault((task.run_date_utc, task.cycle_hour_utc), []).append(task)

    anchor_cycles = [cycle_key for cycle_key, cycle_tasks in cycle_groups.items() if any(task.anchor_cycle_candidate for task in cycle_tasks)]
    assert anchor_cycles == [("2026-04-11", 0)]
    anchor_slots = {task.valid_time_local for task in cycle_groups[anchor_cycles[0]]}
    assert expected_slots <= anchor_slots


def test_build_tasks_for_target_date_overnight_mode_reduces_retained_cycles():
    tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2026-04-11"), selection_mode="overnight_0005")
    retained_cycles = sorted({(task.run_date_utc, task.cycle_hour_utc, task.anchor_cycle_candidate) for task in tasks})

    assert len(retained_cycles) == 2
    assert sum(1 for _, _, is_anchor in retained_cycles if is_anchor) == 1
    assert retained_cycles[0][:2] == ("2026-04-11", 0)


def test_build_summary_row_uses_real_planner_anchor_cycle():
    tasks = hrrr_pipeline.build_tasks_for_target_date(pd.Timestamp("2026-04-11"))
    rows = [make_summary_input_row_from_task(task) for task in tasks]
    summary = hrrr_pipeline.build_summary_row("2026-04-11", rows)
    anchor_task = next(task for task in tasks if task.anchor_cycle_candidate)
    assert summary["anchor_run_date_utc"] == anchor_task.run_date_utc
    assert summary["anchor_cycle_hour_utc"] == anchor_task.cycle_hour_utc
    assert summary["has_full_day_21_local_coverage"] is True
    assert summary["hrrr_temp_2m_day_max_k"] == pytest.approx(321.0)


def test_build_summary_row_expands_new_fields_with_half_open_windows():
    rows = []
    for hour in [6, 9, 12, 15, 18]:
        task = make_task(
            target_date_local="2026-07-11",
            run_date_utc="2026-07-11",
            cycle_hour_utc=0,
            forecast_hour=hour,
            valid_time_local=f"2026-07-11T{hour:02d}:00:00-04:00",
            valid_time_utc=f"2026-07-11T{hour+4:02d}:00:00+00:00",
        )
        row = make_summary_input_row_from_task(task, base_temp=290.0)
        row["tcdc_entire_pct"] = float(hour)
        row["mcdc_mid_pct"] = float(hour + 100)
        row["hcdc_high_pct"] = float(hour + 200)
        row["ltng_entire_atmosphere"] = {6: None, 9: 0.0, 12: None, 15: 2.0, 18: 0.0}[hour]
        rows.append(row)

    summary = hrrr_pipeline.build_summary_row("2026-07-11", rows)

    assert summary["first_valid_hour_local"] == 6
    assert summary["last_valid_hour_local"] == 18
    assert summary["covered_hour_count"] == 5
    assert summary["covered_checkpoint_count"] == 5
    assert summary["hrrr_dewpoint_2m_06_local_k"] == pytest.approx(286.0)
    assert summary["hrrr_dewpoint_2m_12_local_k"] == pytest.approx(292.0)
    assert summary["hrrr_dewpoint_2m_15_local_k"] == pytest.approx(295.0)
    assert summary["hrrr_dewpoint_2m_18_local_k"] == pytest.approx(298.0)
    assert summary["hrrr_rh_2m_06_local_pct"] == pytest.approx(56.0)
    assert summary["hrrr_rh_2m_12_local_pct"] == pytest.approx(52.0)
    assert summary["hrrr_rh_2m_15_local_pct"] == pytest.approx(55.0)
    assert summary["hrrr_rh_2m_18_local_pct"] == pytest.approx(58.0)
    assert summary["hrrr_tcdc_morning_mean_pct"] == pytest.approx((6.0 + 9.0) / 2.0)
    assert summary["hrrr_tcdc_afternoon_mean_pct"] == pytest.approx((12.0 + 15.0) / 2.0)
    assert summary["hrrr_tcdc_day_max_pct"] == pytest.approx(18.0)
    assert summary["hrrr_mcdc_day_mean_pct"] == pytest.approx((106.0 + 109.0 + 112.0 + 115.0 + 118.0) / 5.0)
    assert summary["hrrr_hcdc_afternoon_mean_pct"] == pytest.approx((212.0 + 215.0) / 2.0)
    assert summary["hrrr_ltng_day_max"] == pytest.approx(2.0)
    assert summary["hrrr_ltng_day_any"] is True
    assert summary["hrrr_mslp_day_mean_pa"] == pytest.approx((101206.0 + 101209.0 + 101212.0 + 101215.0 + 101218.0) / 5.0)
    assert summary["hrrr_surface_pressure_09_local_pa"] == pytest.approx(101009.0)
    assert summary["hrrr_temp_1000mb_day_mean_k"] == pytest.approx(275.0)
    assert summary["hrrr_u925_day_mean_ms"] == pytest.approx(8.0)
    assert summary["hrrr_hgt_700_day_mean_gpm"] == pytest.approx(3120.0)
    assert summary["hrrr_wind_10m_09_local_speed_ms"] == pytest.approx(5.0)
    assert summary["hrrr_u10m_15_local_ms"] == pytest.approx(17.0)
    assert summary["hrrr_v10m_18_local_ms"] == pytest.approx(10.0)
    expected_direction = (270.0 - np.degrees(np.arctan2(5.5, 11.0))) % 360.0
    assert summary["hrrr_wind_10m_09_local_direction_deg"] == pytest.approx(expected_direction)


def test_build_summary_row_ltng_day_any_propagates_missing_and_false():
    rows_missing = []
    rows_false = []
    for hour in [9, 15]:
        task = make_task(
            target_date_local="2026-07-12",
            run_date_utc="2026-07-12",
            cycle_hour_utc=0,
            forecast_hour=hour,
            valid_time_local=f"2026-07-12T{hour:02d}:00:00-04:00",
            valid_time_utc=f"2026-07-12T{hour+4:02d}:00:00+00:00",
        )
        row_missing = make_summary_input_row_from_task(task)
        row_missing["ltng_entire_atmosphere"] = None
        rows_missing.append(row_missing)

        row_false = make_summary_input_row_from_task(task)
        row_false["ltng_entire_atmosphere"] = 0.0
        rows_false.append(row_false)

    summary_missing = hrrr_pipeline.build_summary_row("2026-07-12", rows_missing)
    summary_false = hrrr_pipeline.build_summary_row("2026-07-12", rows_false)

    assert summary_missing["hrrr_ltng_day_max"] is None
    assert summary_missing["hrrr_ltng_day_any"] is None
    assert summary_false["hrrr_ltng_day_max"] == pytest.approx(0.0)
    assert summary_false["hrrr_ltng_day_any"] is False


def test_build_summary_row_emits_operational_coverage_fields():
    rows = []
    for hour in [6, 9, 15, 21]:
        valid_time_local = pd.Timestamp(f"2026-07-13T{hour:02d}:00:00-04:00")
        task = make_task(
            target_date_local="2026-07-13",
            run_date_utc="2026-07-13",
            cycle_hour_utc=0,
            forecast_hour=hour,
            valid_time_local=valid_time_local.isoformat(),
            valid_time_utc=valid_time_local.tz_convert("UTC").isoformat(),
        )
        rows.append(make_summary_input_row_from_task(task))

    summary = hrrr_pipeline.build_summary_row("2026-07-13", rows)

    assert summary["first_valid_hour_local"] == 6
    assert summary["last_valid_hour_local"] == 21
    assert summary["covered_hour_count"] == 4
    assert summary["covered_checkpoint_count"] == 4
    assert summary["coverage_end_hour_local"] == 21


def test_build_revision_features_has_exact_whitelist():
    anchor = {
        "hrrr_temp_2m_day_max_k": 1.0,
        "hrrr_temp_2m_09_local_k": 2.0,
        "hrrr_temp_2m_12_local_k": 3.0,
        "hrrr_temp_2m_15_local_k": 4.0,
        "hrrr_tcdc_day_mean_pct": 5.0,
        "hrrr_dswrf_day_max_w_m2": 6.0,
        "hrrr_pwat_day_mean_kg_m2": 7.0,
        "hrrr_hpbl_day_max_m": 8.0,
        "hrrr_mslp_day_mean_pa": 9.0,
        "hrrr_cape_day_max_j_kg": 10.0,
    }
    lag_summaries = {1: {key: 0.0 for key in anchor}}

    revisions = hrrr_pipeline.build_revision_features(anchor, lag_summaries)

    assert sorted(revisions) == sorted(
        [
            "hrrr_temp_2m_day_max_k_rev_1cycle",
            "hrrr_temp_2m_09_local_k_rev_1cycle",
            "hrrr_temp_2m_12_local_k_rev_1cycle",
            "hrrr_temp_2m_15_local_k_rev_1cycle",
            "hrrr_tcdc_day_mean_pct_rev_1cycle",
            "hrrr_dswrf_day_max_w_m2_rev_1cycle",
            "hrrr_pwat_day_mean_kg_m2_rev_1cycle",
            "hrrr_hpbl_day_max_m_rev_1cycle",
            "hrrr_mslp_day_mean_pa_rev_1cycle",
        ]
    )


def test_build_summary_row_marks_incomplete_coverage_when_no_full_day_cycle():
    rows = []
    for hour in [6, 9, 12]:
        rows.append(
            {
                "task_key": f"2023-01-01__2023-01-01_t00_f{hour:02d}",
                "target_date_local": "2023-01-01",
                "run_date_utc": "2023-01-01",
                "cycle_hour_utc": 0,
                "forecast_hour": hour,
                "init_time_utc": "2023-01-01T00:00:00+00:00",
                "init_time_local": "2022-12-31T19:00:00-05:00",
                "valid_time_utc": f"2023-01-01T{hour:02d}:00:00+00:00",
                "valid_time_local": f"2023-01-01T{hour:02d}:00:00-05:00",
                "init_date_local": "2022-12-31",
                "valid_date_local": "2023-01-01",
                "init_hour_local": 19,
                "valid_hour_local": hour,
                "cycle_rank_desc": 0,
                "selected_for_summary": True,
                "anchor_cycle_candidate": False,
                "tmp_2m_k": 290.0,
            }
        )

    summary = hrrr_pipeline.build_summary_row("2023-01-01", rows)
    assert summary["has_full_day_21_local_coverage"] is False
    assert summary["coverage_end_hour_local"] == 12


def test_open_group_dataset_reads_real_sample_if_available():
    pytest.importorskip("cfgrib")
    sample_path = HRRR_DIR / "out" / "hrrr.t12z.wrfsfcf00.grib2.subset.grib2"
    if not sample_path.exists():
        pytest.skip("Local HRRR sample GRIB2 is not available.")

    ds = hrrr_pipeline.open_group_dataset(sample_path, hrrr_pipeline.GROUP_SPECS[0])
    try:
        assert {"t2m", "d2m", "r2"} <= set(ds.data_vars)
        assert ds.sizes["y"] > 0
        assert ds.sizes["x"] > 0
    finally:
        ds.close()


def test_process_task_retries_transient_failure_then_recovers(monkeypatch, tmp_path):
    task = make_task(forecast_hour=5)
    calls = {"count": 0}

    def fake_process_task_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return hrrr_pipeline.TaskResult(
                False,
                task.key,
                None,
                [],
                [],
                "BrokenPipeError: [Errno 32] Broken pipe",
                {"last_error_type": "BrokenPipeError"},
            )
        return hrrr_pipeline.TaskResult(
            True,
            task.key,
            {"task_key": task.key, "valid_time_utc": task.valid_time_utc},
            [],
            [],
            None,
            {},
        )

    monkeypatch.setattr(hrrr_pipeline, "_process_task_once", fake_process_task_once)
    monkeypatch.setattr(hrrr_pipeline.time, "sleep", lambda _seconds: None)
    reporter = hrrr_pipeline.create_progress_reporter("HRRR 2023-01", unit="task", total=1, mode="log", stream=io.StringIO(), is_tty=False)

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=tmp_path / "downloads",
        reduced_dir=tmp_path / "reduced",
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        reporter=reporter,
        max_attempts=3,
    )

    assert result.ok is True
    assert calls["count"] == 2
    assert result.diagnostics["attempt_count"] == 2
    assert result.diagnostics["retry_recovered"] is True


def test_process_task_does_not_retry_deterministic_failure(monkeypatch, tmp_path):
    task = make_task(forecast_hour=6)
    calls = {"count": 0}

    def fake_process_task_once(*args, **kwargs):
        calls["count"] += 1
        return hrrr_pipeline.TaskResult(
            False,
            task.key,
            None,
            [],
            [],
            "Unable to locate KLGA grid cell in reduced GRIB2",
            {"last_error_type": None},
        )

    monkeypatch.setattr(hrrr_pipeline, "_process_task_once", fake_process_task_once)
    monkeypatch.setattr(hrrr_pipeline.time, "sleep", lambda _seconds: None)
    reporter = hrrr_pipeline.create_progress_reporter("HRRR 2023-01", unit="task", total=1, mode="log", stream=io.StringIO(), is_tty=False)

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=tmp_path / "downloads",
        reduced_dir=tmp_path / "reduced",
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        reporter=reporter,
        max_attempts=3,
    )

    assert result.ok is False
    assert calls["count"] == 1


def test_process_task_records_failure_context(monkeypatch, tmp_path):
    task = make_task(forecast_hour=7)

    def fake_process_task_once(*args, **kwargs):
        return hrrr_pipeline.TaskResult(
            False,
            task.key,
            None,
            [],
            [],
            "BrokenPipeError: [Errno 32] Broken pipe",
            {
                "last_error_type": "BrokenPipeError",
                "last_error_message": None,
                "raw_file_path": str(tmp_path / "downloads" / "raw.grib2"),
                "reduced_file_path": str(tmp_path / "reduced" / "reduced.grib2"),
                "grib_url": "https://example.test/file.grib2",
            },
        )

    monkeypatch.setattr(hrrr_pipeline, "_process_task_once", fake_process_task_once)
    monkeypatch.setattr(hrrr_pipeline.time, "sleep", lambda _seconds: None)

    result = hrrr_pipeline.process_task(
        task,
        wgrib2_path="wgrib2",
        download_dir=tmp_path / "downloads",
        reduced_dir=tmp_path / "reduced",
        keep_downloads=False,
        keep_reduced=False,
        include_legacy_aliases=False,
        reporter=None,
        max_attempts=3,
    )

    assert result.ok is False
    assert result.diagnostics["attempt_count"] == 3
    assert result.diagnostics["final_error_class"] == "broken_pipe"
    assert "BrokenPipeError" in result.diagnostics["last_error_message"]
    assert result.diagnostics["raw_file_path"].endswith("raw.grib2")


def test_hrrr_monthly_backfill_iter_days():
    windows = hrrr_monthly_backfill.iter_days("2024-02-28", "2024-03-01")

    assert [window.target_date_local.isoformat() for window in windows] == [
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
    ]


def test_hrrr_monthly_backfill_runs_daily_commands_extracts_day_output_and_cleans_tmp(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    removed: list[pathlib.Path] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        start_date = command[command.index("--start-date") + 1]
        month_id = start_date[:7]
        target_date_local = start_date
        output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        summary_dir = pathlib.Path(command[command.index("--summary-output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame([{"target_date_local": target_date_local, "status": "ok"}])
        summary_df.to_parquet(summary_dir / f"{month_id}.parquet", index=False)
        manifest = {
            "complete": True,
            "expected_task_count": 1,
            "completed_task_keys": ["task_a"],
            "summary_parquet_path": str(summary_dir / f"{month_id}.parquet"),
            "target_date_local": target_date_local,
            "selection_mode": "overnight_0005",
            "extract_method": "wgrib2-bin",
            "summary_profile": "overnight",
            "provenance_written": False,
        }
        (output_dir / f"{month_id}.manifest.json").write_text(json.dumps(manifest))
        pd.DataFrame(
            [
                {
                    "status": "ok",
                    "summary_parquet_path": str(summary_dir / f"{month_id}.parquet"),
                    "selection_mode": "overnight_0005",
                    "extract_method": "wgrib2-bin",
                    "summary_profile": "overnight",
                    "provenance_written": False,
                }
            ]
        ).to_parquet(output_dir / f"{month_id}.manifest.parquet", index=False)
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(hrrr_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(hrrr_monthly_backfill.shutil, "rmtree", lambda path, ignore_errors=True: removed.append(path))

    valid_summary_root = tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-01"
    valid_state_root = tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01"
    valid_summary_root.mkdir(parents=True, exist_ok=True)
    valid_state_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01", "status": "ok"}]).to_parquet(
        valid_summary_root / "hrrr.overnight.parquet",
        index=False,
    )
    manifest = {
        "complete": True,
        "expected_task_count": 1,
        "completed_task_keys": ["task_a"],
        "target_date_local": "2024-01-01",
        "selection_mode": "overnight_0005",
        "extract_method": "wgrib2-bin",
        "summary_profile": "overnight",
        "provenance_written": False,
    }
    (valid_state_root / "hrrr.manifest.json").write_text(json.dumps(manifest))
    pd.DataFrame([{"status": "ok", "selection_mode": "overnight_0005", "extract_method": "wgrib2-bin", "summary_profile": "overnight", "provenance_written": False}]).to_parquet(
        valid_state_root / "hrrr.manifest.parquet",
        index=False,
    )

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-02",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        max_workers=4,
        download_workers=4,
        reduce_workers=2,
        extract_workers=2,
        reduce_queue_size=6,
        extract_queue_size=8,
        progress_mode="log",
        pause_control_file="/tmp/hrrr.pause",
        max_task_attempts=5,
        retry_backoff_seconds=1.5,
        retry_max_backoff_seconds=9.0,
        allow_partial=True,
        keep_temp_on_failure=False,
    )

    assert hrrr_monthly_backfill.run_backfill(args) == 0
    assert len(commands) == 1
    assert commands[0][1] == "tools/hrrr/build_hrrr_klga_feature_shards.py"
    assert "--reduce-queue-size" in commands[0]
    assert "--extract-queue-size" in commands[0]
    assert "--allow-partial" in commands[0]
    assert "--max-task-attempts" in commands[0]
    assert "--retry-backoff-seconds" in commands[0]
    assert "--retry-max-backoff-seconds" in commands[0]
    assert "--pause-control-file" not in commands[0]
    assert (tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-01" / "hrrr.overnight.parquet").exists()
    assert (tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01" / "hrrr.manifest.json").exists()
    assert (tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01" / "hrrr.manifest.parquet").exists()
    assert (tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-02" / "hrrr.overnight.parquet").exists()
    assert (tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-02" / "hrrr.manifest.json").exists()
    assert (tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-02" / "hrrr.manifest.parquet").exists()
    performance = json.loads(
        (tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-02" / "hrrr.performance.json").read_text()
    )
    assert performance["token"] == "2024-01-02"
    assert "full_day_seconds" in performance
    assert removed == [
        tmp_path / "runtime" / "hrrr_tmp" / "2024-01-02",
    ]


def test_hrrr_monthly_backfill_rebuilds_day_when_selection_mode_changes(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        start_date = command[command.index("--start-date") + 1]
        month_id = start_date[:7]
        output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        summary_dir = pathlib.Path(command[command.index("--summary-output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"target_date_local": start_date, "status": "ok"}]).to_parquet(summary_dir / f"{month_id}.parquet", index=False)
        (output_dir / f"{month_id}.manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "expected_task_count": 1,
                    "completed_task_keys": ["task_a"],
                    "target_date_local": start_date,
                    "selection_mode": "overnight_0005",
                    "extract_method": "wgrib2-bin",
                    "summary_profile": "overnight",
                    "provenance_written": False,
                }
            )
        )
        pd.DataFrame(
            [
                {
                    "status": "ok",
                    "selection_mode": "overnight_0005",
                    "extract_method": "wgrib2-bin",
                    "summary_profile": "overnight",
                    "provenance_written": False,
                }
            ]
        ).to_parquet(output_dir / f"{month_id}.manifest.parquet", index=False)
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(hrrr_monthly_backfill.subprocess, "run", fake_run)

    valid_summary_root = tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-01"
    valid_state_root = tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01"
    valid_summary_root.mkdir(parents=True, exist_ok=True)
    valid_state_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01", "status": "ok"}]).to_parquet(
        valid_summary_root / "hrrr.overnight.parquet",
        index=False,
    )
    (valid_state_root / "hrrr.manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_task_count": 1,
                "completed_task_keys": ["task_a"],
                "target_date_local": "2024-01-01",
                "selection_mode": "all",
            }
        )
    )
    pd.DataFrame([{"status": "ok", "selection_mode": "all"}]).to_parquet(
        valid_state_root / "hrrr.manifest.parquet",
        index=False,
    )

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        max_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        allow_partial=False,
        keep_temp_on_failure=False,
    )

    assert hrrr_monthly_backfill.run_backfill(args) == 0
    assert [command[1] for command in commands] == ["tools/hrrr/build_hrrr_klga_feature_shards.py"]


def test_hrrr_monthly_backfill_rebuilds_day_when_manifest_parquet_is_empty(tmp_path):
    summary_root = tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-01"
    state_root = tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01"
    summary_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01", "status": "ok"}]).to_parquet(summary_root / "hrrr.overnight.parquet", index=False)
    (state_root / "hrrr.manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_task_count": 1,
                "completed_task_keys": ["task_a"],
                "target_date_local": "2024-01-01",
                "selection_mode": "overnight_0005",
            }
        )
    )
    pd.DataFrame([], columns=["status", "selection_mode"]).to_parquet(state_root / "hrrr.manifest.parquet", index=False)

    assert hrrr_monthly_backfill.validate_hrrr_day(
        tmp_path / "runtime",
        hrrr_monthly_backfill.DayWindow(dt.date(2024, 1, 1)),
        selection_mode="overnight_0005",
    ) is False


def test_hrrr_monthly_validation_rejects_wrong_extract_contract(tmp_path):
    summary_root = tmp_path / "runtime" / "hrrr_summary" / "target_date_local=2024-01-01"
    state_root = tmp_path / "runtime" / "hrrr_summary_state" / "target_date_local=2024-01-01"
    summary_root.mkdir(parents=True, exist_ok=True)
    state_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01", "status": "ok"}]).to_parquet(
        summary_root / "hrrr.overnight.parquet",
        index=False,
    )
    (state_root / "hrrr.manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "expected_task_count": 1,
                "completed_task_keys": ["task_a"],
                "target_date_local": "2024-01-01",
                "selection_mode": "overnight_0005",
                "extract_method": "eccodes",
                "summary_profile": "overnight",
                "provenance_written": False,
            }
        )
    )
    pd.DataFrame(
        [
            {
                "status": "ok",
                "selection_mode": "overnight_0005",
                "extract_method": "eccodes",
                "summary_profile": "overnight",
                "provenance_written": False,
            }
        ]
    ).to_parquet(state_root / "hrrr.manifest.parquet", index=False)

    day = hrrr_monthly_backfill.DayWindow(dt.date(2024, 1, 1))

    assert hrrr_monthly_backfill.validate_hrrr_day(
        tmp_path / "runtime",
        day,
        selection_mode="overnight_0005",
        extract_method="eccodes",
        summary_profile="overnight",
        provenance_written=False,
    ) is True
    assert hrrr_monthly_backfill.validate_hrrr_day(
        tmp_path / "runtime",
        day,
        selection_mode="overnight_0005",
        extract_method="wgrib2-bin",
        summary_profile="overnight",
        provenance_written=False,
    ) is False


def test_hrrr_monthly_parse_args_accepts_day_workers_and_dashboard_hotkey_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hrrr_monthly_backfill.py",
            "--start-local-date",
            "2024-01-01",
            "--end-local-date",
            "2024-01-02",
            "--day-workers",
            "3",
            "--disable-dashboard-hotkeys",
        ],
    )

    args = hrrr_monthly_backfill.parse_args()

    assert args.day_workers == 3
    assert args.disable_dashboard_hotkeys is True
    assert args.batch_reduce_mode == "cycle"
    assert args.range_merge_gap_bytes == 65536
    assert args.crop_method == "auto"
    assert args.crop_grib_type == "same"
    assert args.wgrib2_threads == 1
    assert args.extract_method == "wgrib2-bin"
    assert args.summary_profile == "overnight"
    assert args.skip_provenance is True


def test_hrrr_monthly_parse_args_can_write_provenance(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_hrrr_monthly_backfill.py",
            "--start-local-date",
            "2024-01-01",
            "--end-local-date",
            "2024-01-01",
            "--write-provenance",
        ],
    )

    args = hrrr_monthly_backfill.parse_args()

    assert args.skip_provenance is False


def test_hrrr_monthly_build_command_has_single_python_executable_and_optional_parent_pause_forwarding(tmp_path):
    args = argparse.Namespace(
        selection_mode="overnight_0005",
        max_workers=4,
        download_workers=4,
        reduce_workers=2,
        extract_workers=2,
        reduce_queue_size=6,
        extract_queue_size=8,
        progress_mode="dashboard",
        pause_control_file="/tmp/hrrr.pause",
        max_task_attempts=5,
        retry_backoff_seconds=1.5,
        retry_max_backoff_seconds=9.0,
        allow_partial=True,
        keep_reduced=True,
        batch_reduce_mode="cycle",
    )
    day = hrrr_monthly_backfill.DayWindow(dt.date(2024, 1, 2))

    command = hrrr_monthly_backfill.build_hrrr_command(
        args,
        day=day,
        tmp_root=tmp_path / "tmp",
        summary_dir=tmp_path / "summary",
        output_dir=tmp_path / "output",
        progress_mode="log",
        include_pause_control_file=False,
    )

    assert command[:2] == [sys.executable, "tools/hrrr/build_hrrr_klga_feature_shards.py"]
    assert command.count(sys.executable) == 1
    assert command[command.index("--progress-mode") + 1] == "log"
    assert command[command.index("--batch-reduce-mode") + 1] == "cycle"
    assert command[command.index("--range-merge-gap-bytes") + 1] == "65536"
    assert command[command.index("--crop-method") + 1] == "auto"
    assert command[command.index("--crop-grib-type") + 1] == "same"
    assert command[command.index("--wgrib2-threads") + 1] == "1"
    assert command[command.index("--extract-method") + 1] == "wgrib2-bin"
    assert command[command.index("--summary-profile") + 1] == "overnight"
    assert "--skip-provenance" in command
    assert "--pause-control-file" not in command
    assert "--allow-partial" in command
    assert "--keep-reduced" in command

    command_with_pause = hrrr_monthly_backfill.build_hrrr_command(
        args,
        day=day,
        tmp_root=tmp_path / "tmp",
        summary_dir=tmp_path / "summary",
        output_dir=tmp_path / "output",
        include_pause_control_file=True,
    )

    assert command_with_pause[command_with_pause.index("--pause-control-file") + 1] == "/tmp/hrrr.pause"


def test_hrrr_monthly_backfill_day_workers_run_days_concurrently(tmp_path, monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_process_day(args, day, *, bridge=None):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        token = hrrr_monthly_backfill.day_token(day)
        performance = hrrr_monthly_backfill.DayPerformance(
            token=token,
            expected_task_count=1,
            completed_task_count=1,
            raw_build_seconds=0.01,
            extract_outputs_seconds=0.01,
            cleanup_seconds=0.01,
            full_day_seconds=0.03,
            timing_idx_fetch_seconds_median=None,
            timing_range_download_seconds_median=None,
            timing_reduce_seconds_median=None,
            timing_cfgrib_open_seconds_median=None,
            timing_row_build_seconds_median=None,
            timing_cleanup_seconds_median=None,
        )
        return hrrr_monthly_backfill.DayRunResult(token=token, performance=performance)

    monkeypatch.setattr(hrrr_monthly_backfill, "process_day", fake_process_day)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-03",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        max_workers=4,
        progress_mode="log",
        pause_control_file=None,
        disable_dashboard_hotkeys=False,
        keep_temp_on_failure=False,
    )

    assert hrrr_monthly_backfill.run_backfill(args) == 0
    assert max_active == 2


def test_hrrr_monthly_backfill_pause_file_drains_active_day_and_stops_new_admission(tmp_path, monkeypatch):
    pause_file = tmp_path / "hrrr.pause"
    processed: list[str] = []

    def fake_process_day(args, day, *, bridge=None):
        token = hrrr_monthly_backfill.day_token(day)
        processed.append(token)
        pause_file.write_text("pause")
        time.sleep(0.02)
        performance = hrrr_monthly_backfill.DayPerformance(
            token=token,
            expected_task_count=1,
            completed_task_count=1,
            raw_build_seconds=0.01,
            extract_outputs_seconds=0.01,
            cleanup_seconds=0.01,
            full_day_seconds=0.03,
            timing_idx_fetch_seconds_median=None,
            timing_range_download_seconds_median=None,
            timing_reduce_seconds_median=None,
            timing_cfgrib_open_seconds_median=None,
            timing_row_build_seconds_median=None,
            timing_cleanup_seconds_median=None,
        )
        return hrrr_monthly_backfill.DayRunResult(token=token, performance=performance)

    monkeypatch.setattr(hrrr_monthly_backfill, "process_day", fake_process_day)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-03",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=1,
        max_workers=4,
        progress_mode="log",
        pause_control_file=str(pause_file),
        disable_dashboard_hotkeys=False,
        keep_temp_on_failure=False,
    )

    assert hrrr_monthly_backfill.run_backfill(args) == 0
    assert processed == ["2024-01-01"]


def test_hrrr_monthly_backfill_worker_failure_drains_active_days_and_raises_first_exception(tmp_path, monkeypatch):
    started: list[str] = []
    release_second = threading.Event()

    def fake_process_day(args, day, *, bridge=None):
        token = hrrr_monthly_backfill.day_token(day)
        started.append(token)
        if token == "2024-01-01":
            raise RuntimeError("first day failed")
        release_second.wait(timeout=1.0)
        performance = hrrr_monthly_backfill.DayPerformance(
            token=token,
            expected_task_count=1,
            completed_task_count=1,
            raw_build_seconds=0.01,
            extract_outputs_seconds=0.01,
            cleanup_seconds=0.01,
            full_day_seconds=0.03,
            timing_idx_fetch_seconds_median=None,
            timing_range_download_seconds_median=None,
            timing_reduce_seconds_median=None,
            timing_cfgrib_open_seconds_median=None,
            timing_row_build_seconds_median=None,
            timing_cleanup_seconds_median=None,
        )
        return hrrr_monthly_backfill.DayRunResult(token=token, performance=performance)

    monkeypatch.setattr(hrrr_monthly_backfill, "process_day", fake_process_day)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-03",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        max_workers=4,
        progress_mode="log",
        pause_control_file=None,
        disable_dashboard_hotkeys=False,
        keep_temp_on_failure=False,
    )

    try:
        with pytest.raises(RuntimeError, match="first day failed"):
            hrrr_monthly_backfill.run_backfill(args)
    finally:
        release_second.set()
    assert "2024-01-03" not in started


def test_hrrr_process_day_keep_temp_on_failure_preserves_failed_tmp_root(tmp_path, monkeypatch):
    def fake_run_command(command):
        output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        tmp_root = output_dir.parent
        tmp_root.mkdir(parents=True, exist_ok=True)
        (tmp_root / "debug.txt").write_text("keep me")
        raise RuntimeError("child failed")

    monkeypatch.setattr(hrrr_monthly_backfill, "run_command", fake_run_command)

    day = hrrr_monthly_backfill.DayWindow(dt.date(2024, 1, 1))
    args = argparse.Namespace(
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        max_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        allow_partial=False,
        keep_reduced=False,
        keep_temp_on_failure=True,
    )

    with pytest.raises(RuntimeError, match="child failed"):
        hrrr_monthly_backfill.process_day(args, day)
    assert (tmp_path / "runtime" / "hrrr_tmp" / "2024-01-01" / "debug.txt").exists()


def test_hrrr_process_day_failure_removes_tmp_root_by_default(tmp_path, monkeypatch):
    def fake_run_command(command):
        output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        tmp_root = output_dir.parent
        tmp_root.mkdir(parents=True, exist_ok=True)
        (tmp_root / "debug.txt").write_text("remove me")
        raise RuntimeError("child failed")

    monkeypatch.setattr(hrrr_monthly_backfill, "run_command", fake_run_command)

    day = hrrr_monthly_backfill.DayWindow(dt.date(2024, 1, 1))
    args = argparse.Namespace(
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        max_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        allow_partial=False,
        keep_reduced=False,
        keep_temp_on_failure=False,
    )

    with pytest.raises(RuntimeError, match="child failed"):
        hrrr_monthly_backfill.process_day(args, day)
    assert not (tmp_path / "runtime" / "hrrr_tmp" / "2024-01-01").exists()


def test_hrrr_monthly_backfill_dashboard_forces_child_log_and_relays_progress(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    reporters = []

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        commands.append(command)
        start_date = command[command.index("--start-date") + 1]
        month_id = start_date[:7]
        output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
        summary_dir = pathlib.Path(command[command.index("--summary-output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"target_date_local": start_date, "status": "ok"}]).to_parquet(summary_dir / f"{month_id}.parquet", index=False)
        (output_dir / f"{month_id}.manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "expected_task_count": 1,
                    "completed_task_keys": ["task_a"],
                    "target_date_local": start_date,
                    "selection_mode": "overnight_0005",
                    "extract_method": "wgrib2-bin",
                    "summary_profile": "overnight",
                    "provenance_written": False,
                }
            )
        )
        pd.DataFrame(
            [
                {
                    "status": "ok",
                    "selection_mode": "overnight_0005",
                    "extract_method": "wgrib2-bin",
                    "summary_profile": "overnight",
                    "provenance_written": False,
                    "timing_range_download_seconds": 0.2,
                    "timing_reduce_seconds": 0.3,
                    "timing_cfgrib_open_seconds": 0.4,
                }
            ]
        ).to_parquet(output_dir / f"{month_id}.manifest.parquet", index=False)
        stdout_handler("[progress] HRRR event=worker_start worker=download_1 label=2024-01-01 c00 f01 phase=download details=byte_range_download")
        stdout_handler("[progress] HRRR event=transfer_start worker=download_1 file=hrrr.grib2 total_bytes=1024")
        stdout_handler("[progress] HRRR event=transfer_progress worker=download_1 file=hrrr.grib2 bytes_downloaded=1024 total_bytes=1024")
        stdout_handler("[progress] HRRR event=transfer_complete worker=download_1 file=hrrr.grib2 bytes_downloaded=1024 total_bytes=1024")
        stdout_handler("[progress] HRRR event=worker_retire worker=download_1 label=2024-01-01 c00 f01")
        stderr_handler("child plain stderr")

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(hrrr_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(hrrr_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(hrrr_monthly_backfill, "stream_command", fake_stream_command)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        max_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        progress_mode="dashboard",
        pause_control_file=str(tmp_path / "runner.pause"),
        disable_dashboard_hotkeys=True,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        allow_partial=False,
        keep_reduced=False,
        keep_temp_on_failure=False,
    )

    assert hrrr_monthly_backfill.run_backfill(args) == 0
    assert reporters[0].mode == "dashboard"
    assert commands[0][commands[0].index("--progress-mode") + 1] == "log"
    assert "--pause-control-file" not in commands[0]
    worker = reporters[0].state.workers["2024-01-01/download_1"]
    assert worker.transfer is not None
    assert worker.transfer.bytes_downloaded == 1024
    assert reporters[0].state.groups["2024-01-01"].status == "complete"
    messages = [event.message for event in reporters[0].state.recent_events]
    assert any("2024-01-01 stderr: child plain stderr" == message for message in messages)
