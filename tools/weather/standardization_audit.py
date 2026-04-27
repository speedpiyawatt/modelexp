#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib.util
import inspect
import json
import pathlib
import sys
import tempfile
from typing import Iterator

import numpy as np
import pandas as pd
import xarray as xr


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HRRR_DIR = REPO_ROOT / "tools" / "hrrr"
NBM_DIR = REPO_ROOT / "tools" / "nbm"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HRRR_DIR) not in sys.path:
    sys.path.insert(0, str(HRRR_DIR))
if str(NBM_DIR) not in sys.path:
    sys.path.insert(0, str(NBM_DIR))

from tools.weather import location_context
from tools.weather.canonical_feature_schema import (
    CANONICAL_PROVENANCE_COLUMNS,
    CANONICAL_WIDE_COLUMNS,
    RUNTIME_IDENTITY_COLUMNS,
    RUNTIME_SPATIAL_COLUMNS,
)
from tools.weather.normalize_training_features import (
    normalize_hrrr_provenance_to_canonical,
    normalize_hrrr_wide_to_canonical,
    normalize_nbm_provenance_to_canonical,
    normalize_nbm_wide_to_canonical,
)


OUTPUT_MATRIX_NAME = "compatibility_matrix.json"
OUTPUT_REVIEW_NAME = "compatibility_review.md"

HRRR_PIPELINE_PATH = HRRR_DIR / "build_hrrr_klga_feature_shards.py"
NBM_PIPELINE_PATH = NBM_DIR / "build_grib2_features.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


hrrr = load_module("weather_standardization_hrrr", HRRR_PIPELINE_PATH)
nbm = load_module("weather_standardization_nbm", NBM_PIPELINE_PATH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HRRR and NBM weather-feature standardization status.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=SCRIPT_DIR, help="Directory for compatibility outputs.")
    return parser.parse_args()


def make_lat_lon_grid() -> tuple[np.ndarray, np.ndarray]:
    lat_axis = np.array([41.25, 41.10, 40.95, 40.7769, 40.60, 40.45, 40.30], dtype=float)
    lon_axis = np.array([285.80, 285.95, 286.05, 286.1260, 286.22, 286.35, 286.50], dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    return lat_grid, lon_grid


def make_lat_lon_grid_for_shape(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    full_lat, full_lon = make_lat_lon_grid()
    if height <= full_lat.shape[0] and width <= full_lat.shape[1]:
        row_start = max(0, (full_lat.shape[0] - height) // 2)
        col_start = max(0, (full_lat.shape[1] - width) // 2)
        return (
            full_lat[row_start : row_start + height, col_start : col_start + width],
            full_lon[row_start : row_start + height, col_start : col_start + width],
        )

    lat_axis = np.linspace(41.25, 40.30, height, dtype=float)
    lon_axis = np.linspace(285.80, 286.50, width, dtype=float)
    lon_grid, lat_grid = np.meshgrid(lon_axis, lat_axis)
    return lat_grid, lon_grid


def make_hrrr_data_array(
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


def build_hrrr_test_datasets() -> dict[str, xr.Dataset]:
    base = np.arange(49, dtype=float).reshape(7, 7)
    return {
        "near_surface_2m": xr.Dataset(
            {
                "t2m": make_hrrr_data_array(base + 273.15, short_name="t2m", type_of_level="heightAboveGround", step_type="instant", units="K"),
                "d2m": make_hrrr_data_array(base + 268.15, short_name="d2m", type_of_level="heightAboveGround", step_type="instant", units="K"),
                "r2": make_hrrr_data_array(base + 10.0, short_name="r2", type_of_level="heightAboveGround", step_type="instant", units="%"),
            }
        ),
        "near_surface_10m": xr.Dataset(
            {
                "u10": make_hrrr_data_array(base / 10.0, short_name="u10", type_of_level="heightAboveGround", step_type="instant", units="m s-1"),
                "v10": make_hrrr_data_array(base / 20.0, short_name="v10", type_of_level="heightAboveGround", step_type="instant", units="m s-1"),
            }
        ),
        "pwat": xr.Dataset(
            {
                "pwat": make_hrrr_data_array(base + 5.0, short_name="pwat", type_of_level="atmosphereSingleLayer", step_type="instant", units="kg m-2"),
            }
        ),
        "isobaric_850": xr.Dataset(
            {
                "t": make_hrrr_data_array(base + 260.0, short_name="t", type_of_level="isobaricInhPa", step_type="instant", units="K"),
                "u": make_hrrr_data_array(base / 8.0, short_name="u", type_of_level="isobaricInhPa", step_type="instant", units="m s-1"),
                "v": make_hrrr_data_array(base / 9.0, short_name="v", type_of_level="isobaricInhPa", step_type="instant", units="m s-1"),
                "gh": make_hrrr_data_array(base + 1500.0, short_name="gh", type_of_level="isobaricInhPa", step_type="instant", units="gpm"),
                "dpt": make_hrrr_data_array(base + 255.0, short_name="dpt", type_of_level="isobaricInhPa", step_type="instant", units="K"),
            }
        ),
    }


def make_nbm_inventory_line(record_id: int, short_name: str, level_text: str, step_text: str, extra_text: str = "") -> str:
    line = f"{record_id}:{record_id * 100}:d=2026010115:{short_name}:{level_text}:{step_text}:"
    if extra_text:
        line = f"{line}{extra_text}"
    return line


def make_nbm_raw_payload(size: int = 1400) -> bytes:
    chunks = bytearray()
    for index in range(size):
        chunks.append(index % 251)
    return bytes(chunks)


def make_nbm_dataset(var_name: str, short_name: str, values: np.ndarray, *, units: str = "unitless") -> xr.Dataset:
    lat_grid, lon_grid = make_lat_lon_grid_for_shape(values.shape[0], values.shape[1])
    dataset = xr.Dataset(
        {
            var_name: xr.DataArray(
                values,
                dims=("y", "x"),
                coords={
                    "latitude": (("y", "x"), lat_grid),
                    "longitude": (("y", "x"), lon_grid),
                    "time": pd.Timestamp("2026-01-01T15:00:00Z"),
                    "valid_time": pd.Timestamp("2026-01-01T16:00:00Z"),
                },
                attrs={
                    "GRIB_shortName": short_name,
                    "GRIB_typeOfLevel": "heightAboveGround" if short_name in {"TMP", "WIND", "WDIR"} else "surface",
                    "GRIB_stepType": "instant",
                    "GRIB_name": short_name,
                    "units": units,
                },
            )
        }
    )
    return dataset


@contextlib.contextmanager
def patched_attr(module, name: str, value) -> Iterator[None]:
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


def build_hrrr_sample_result() -> hrrr.TaskResult:
    datasets = build_hrrr_test_datasets()

    def fake_open_group_dataset(_path, group):
        return datasets[group.name].copy(deep=True)

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
    task = hrrr.TaskSpec(
        target_date_local="2023-01-01",
        run_date_utc="2023-01-01",
        cycle_hour_utc=12,
        forecast_hour=0,
        init_time_utc="2023-01-01T12:00:00+00:00",
        init_time_local="2023-01-01T07:00:00-05:00",
        valid_time_utc="2023-01-01T12:00:00+00:00",
        valid_time_local="2023-01-01T07:00:00-05:00",
        init_date_local="2023-01-01",
        valid_date_local="2023-01-01",
        init_hour_local=7,
        valid_hour_local=7,
        cycle_rank_desc=0,
        selected_for_summary=True,
        anchor_cycle_candidate=True,
    )

    with patched_attr(hrrr, "open_group_dataset", fake_open_group_dataset):
        result = hrrr.process_reduced_grib(pathlib.Path("/tmp/fake.grib2"), inventory, task, "https://example.com/file.grib2")
    if not result.ok or result.row is None:
        raise RuntimeError(f"Failed to build synthetic HRRR sample result: {result.message}")
    return result


def build_hrrr_sample_outputs() -> dict[str, object]:
    sample_result = build_hrrr_sample_result()
    task = hrrr.TaskSpec(
        target_date_local="2023-01-01",
        run_date_utc="2023-01-01",
        cycle_hour_utc=12,
        forecast_hour=0,
        init_time_utc="2023-01-01T12:00:00+00:00",
        init_time_local="2023-01-01T07:00:00-05:00",
        valid_time_utc="2023-01-01T12:00:00+00:00",
        valid_time_local="2023-01-01T07:00:00-05:00",
        init_date_local="2023-01-01",
        valid_date_local="2023-01-01",
        init_hour_local=7,
        valid_hour_local=7,
        cycle_rank_desc=0,
        selected_for_summary=True,
        anchor_cycle_candidate=True,
    )

    def fake_process_task(_task, **_kwargs):
        return hrrr.TaskResult(True, task.key, dict(sample_result.row), list(sample_result.provenance_rows), list(sample_result.missing_fields), None)

    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        with patched_attr(hrrr, "process_task", fake_process_task):
            wrote, failed = hrrr.run_month(
                "2023-01",
                [task],
                wgrib2_path="wgrib2",
                output_dir=output_dir,
                summary_output_dir=output_dir / "summary",
                download_dir=output_dir / "_downloads",
                reduced_dir=output_dir / "_reduced",
                max_workers=1,
                keep_downloads=False,
                keep_reduced=False,
            )
        if wrote != 1 or failed != 0:
            raise RuntimeError("Synthetic HRRR audit month failed to finalize.")
        wide_df = pd.read_parquet(output_dir / "2023-01.parquet")
        provenance_df = pd.read_parquet(output_dir / "2023-01.provenance.parquet")
        manifest_json = json.loads((output_dir / "2023-01.manifest.json").read_text())
        manifest_df = pd.read_parquet(output_dir / "2023-01.manifest.parquet")
        return {
            "wide_df": wide_df,
            "provenance_df": provenance_df,
            "wide_row": wide_df.iloc[0].to_dict(),
            "provenance_row": provenance_df.iloc[0].to_dict(),
            "manifest_row": manifest_df.iloc[0].to_dict(),
            "manifest_json": manifest_json,
        }


class FakeGrib2Client:
    def __init__(self, payloads: dict[str, bytes | str]):
        self.payloads = payloads

    @staticmethod
    def _key_from_url(url: str) -> str:
        return url.split(".amazonaws.com/", 1)[1]

    def fetch_text(self, url: str) -> str:
        payload = self.payloads[self._key_from_url(url)]
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        return str(payload)

    def fetch_content_length(self, url: str) -> int:
        payload = self.payloads[self._key_from_url(url)]
        if isinstance(payload, bytes):
            return len(payload)
        return len(str(payload).encode("utf-8"))

    def download_byte_ranges(
        self,
        *,
        url: str,
        ranges: list[tuple[int, int]],
        destination: pathlib.Path,
        overwrite: bool,
        progress_label: str | None = None,
    ) -> pathlib.Path:
        del overwrite, progress_label
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads[self._key_from_url(url)]
        if not isinstance(payload, bytes):
            payload = str(payload).encode("utf-8")
        with destination.open("wb") as handle:
            for start, end in ranges:
                handle.write(payload[start : end + 1])
        return destination

    def download_key(self, *, base_url: str, key: str, output_dir: pathlib.Path, overwrite: bool, preserve_tree: bool) -> pathlib.Path:
        del base_url, overwrite
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / key if preserve_tree else output_dir / pathlib.Path(key).name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads[key]
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload)
        return path


def build_nbm_sample_outputs() -> dict[str, object]:
    idx_text = "\n".join(
        [
            make_nbm_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(3, "RH", "2 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(4, "WIND", "10 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(5, "WDIR", "10 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(6, "GUST", "10 m above ground", "1 hour fcst"),
            make_nbm_inventory_line(7, "TCDC", "surface", "1 hour fcst"),
            make_nbm_inventory_line(8, "DSWRF", "surface", "1 hour fcst"),
            make_nbm_inventory_line(9, "APCP", "surface", "0-1 hour acc fcst"),
            make_nbm_inventory_line(10, "VIS", "surface", "1 hour fcst"),
            make_nbm_inventory_line(11, "CEIL", "cloud ceiling", "1 hour fcst"),
            make_nbm_inventory_line(12, "CAPE", "surface", "1 hour fcst"),
            make_nbm_inventory_line(13, "VRATE", "entire atmosphere (considered as a single layer)", "1 hour fcst"),
        ]
    )
    cycle_plan = nbm.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=nbm.NY_TZ),
        cycle="15",
    )
    key = nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_nbm_raw_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=None,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    def fake_open_grouped_datasets(_path):
        return [
            make_nbm_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float), units="K"),
            make_nbm_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float), units="m s-1"),
            make_nbm_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float), units="degrees"),
        ]

    with tempfile.TemporaryDirectory() as temp_dir:
        args.output_dir = pathlib.Path(temp_dir)
        with patched_attr(nbm, "crop_selected_grib2", fake_crop_selected_grib2), patched_attr(nbm, "open_grouped_datasets", fake_open_grouped_datasets):
            result = nbm.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)
            nbm.write_cycle_outputs(args.output_dir, cycle_plan, [result], write_long=False)
        manifest_df = pd.read_parquet(nbm.manifest_path(args.output_dir, cycle_plan))
        return {
            "wide_df": pd.DataFrame.from_records(result.wide_rows),
            "provenance_df": pd.DataFrame.from_records(result.provenance_rows),
            "wide_row": result.wide_rows[0],
            "provenance_row": result.provenance_rows[0],
            "manifest_row": manifest_df.iloc[0].to_dict(),
        }


def normalize_hrrr_manifest_semantics(manifest_row: dict[str, object], manifest_json: dict[str, object] | None = None) -> dict[str, bool]:
    manifest_json = manifest_json or {}
    return {
        "tracks_expected_state": "expected_task_count" in manifest_json and "completed_task_keys" in manifest_json,
        "tracks_failure_state": "status" in manifest_row or "failure_reasons" in manifest_json,
        "tracks_missing_fields": "missing_fields" in manifest_row or "missing_fields" in manifest_json,
        "tracks_output_paths": (
            ("wide_parquet_path" in manifest_row and "provenance_path" in manifest_row)
            or ("wide_parquet_path" in manifest_json and "provenance_path" in manifest_json)
        ),
        "tracks_cleanup_state": (
            ("keep_downloads" in manifest_row and "keep_reduced" in manifest_row)
            or ("keep_downloads" in manifest_json and "keep_reduced" in manifest_json)
        ),
    }


def normalize_nbm_manifest_semantics(manifest: dict[str, object]) -> dict[str, bool]:
    return {
        "tracks_expected_state": "extraction_status" in manifest,
        "tracks_failure_state": "extraction_status" in manifest,
        "tracks_missing_fields": "warnings" in manifest,
        "tracks_output_paths": any(key.endswith("_output_paths") for key in manifest),
        "tracks_cleanup_state": any(key in manifest for key in ("raw_deleted", "idx_deleted", "reduced_deleted", "reduced_retained")),
    }


def classify_presence(nbm_ok: bool, hrrr_ok: bool, *, notes: str = "", allowed_difference: bool = False) -> dict[str, str]:
    if nbm_ok and hrrr_ok:
        classification = "synced"
    elif allowed_difference:
        classification = "allowed_difference"
    else:
        classification = "drift"
    return {
        "nbm_status": "present" if nbm_ok else "missing",
        "hrrr_status": "present" if hrrr_ok else "missing",
        "classification": classification,
        "notes": notes,
    }


def row_has_metric_family(row: dict[str, object], base: str) -> bool:
    required = [
        base,
        f"{base}_crop_mean",
        f"{base}_crop_min",
        f"{base}_crop_max",
        f"{base}_crop_std",
        f"{base}_nb3_mean",
        f"{base}_nb3_min",
        f"{base}_nb3_max",
        f"{base}_nb3_std",
        f"{base}_nb3_gradient_west_east",
        f"{base}_nb3_gradient_south_north",
        f"{base}_nb7_mean",
        f"{base}_nb7_min",
        f"{base}_nb7_max",
        f"{base}_nb7_std",
        f"{base}_nb7_gradient_west_east",
        f"{base}_nb7_gradient_south_north",
    ]
    return all(key in row for key in required)


def same_location_context_source() -> bool:
    nbm_source = inspect.getsourcefile(nbm.find_nearest_grid_cell)
    hrrr_source = inspect.getsourcefile(hrrr.find_nearest_grid_cell)
    expected = str((REPO_ROOT / "tools" / "weather" / "location_context.py").resolve())
    return nbm_source == hrrr_source == expected


def compare_contracts() -> tuple[list[dict[str, str]], str]:
    hrrr_outputs = build_hrrr_sample_outputs()
    nbm_outputs = build_nbm_sample_outputs()

    hrrr_row = hrrr_outputs["wide_row"]
    nbm_row = nbm_outputs["wide_row"]
    hrrr_provenance = hrrr_outputs["provenance_row"]
    nbm_provenance = nbm_outputs["provenance_row"]
    hrrr_manifest = hrrr_outputs["manifest_row"]
    nbm_manifest = nbm_outputs["manifest_row"]
    hrrr_manifest_json = hrrr_outputs["manifest_json"]

    nbm_canonical_wide = normalize_nbm_wide_to_canonical(nbm_outputs["wide_df"])
    nbm_canonical_provenance = normalize_nbm_provenance_to_canonical(nbm_outputs["provenance_df"])
    hrrr_canonical_wide = normalize_hrrr_wide_to_canonical(hrrr_outputs["wide_df"])
    hrrr_canonical_provenance = normalize_hrrr_provenance_to_canonical(hrrr_outputs["provenance_df"])

    rows: list[dict[str, str]] = []

    def add(area: str, item: str, nbm_status: str, hrrr_status: str, classification: str, notes: str) -> None:
        rows.append(
            {
                "area": area,
                "item": item,
                "nbm_status": nbm_status,
                "hrrr_status": hrrr_status,
                "classification": classification,
                "notes": notes,
            }
        )

    add(
        "raw_runtime_contract",
        "shared_location_context",
        "shared" if same_location_context_source() else "not_shared",
        "shared" if same_location_context_source() else "not_shared",
        "synced" if same_location_context_source() else "drift",
        "Both pipelines should resolve spatial helpers from tools/weather/location_context.py.",
    )

    core_identity_columns = set(RUNTIME_IDENTITY_COLUMNS)
    add(
        "raw_runtime_contract",
        "runtime_identity_columns",
        "present" if core_identity_columns <= set(nbm_row) else "missing",
        "present" if core_identity_columns <= set(hrrr_row) else "missing",
        "synced" if core_identity_columns <= set(nbm_row) and core_identity_columns <= set(hrrr_row) else "drift",
        "Both live raw pipelines must carry the shared runtime row-identity columns, including source_version.",
    )
    add(
        "raw_runtime_contract",
        "source_model_product_values",
        f"{nbm_row.get('source_model')}/{nbm_row.get('source_product')}",
        f"{hrrr_row.get('source_model')}/{hrrr_row.get('source_product')}",
        "allowed_difference",
        "Model/product identifiers are expected to differ by source.",
    )
    add(
        "raw_runtime_contract",
        "source_version_field",
        "present" if "source_version" in nbm_row else "missing",
        "present" if "source_version" in hrrr_row else "missing",
        "synced" if "source_version" in nbm_row and "source_version" in hrrr_row else "drift",
        "Both live raw pipelines now expose source_version on the wide rows.",
    )
    add(
        "raw_runtime_contract",
        "source_specific_mode_metadata",
        "present" if "mode" in nbm_row else "missing",
        "present" if "mode" in hrrr_row else "missing",
        "allowed_difference",
        "NBM exposes premarket/intraday mode as optional metadata; HRRR intentionally keeps it out of raw runtime identity.",
    )

    spatial_columns = set(RUNTIME_SPATIAL_COLUMNS)
    add(
        "raw_runtime_contract",
        "runtime_spatial_metadata",
        "present" if spatial_columns <= set(nbm_row) else "missing",
        "present" if spatial_columns <= set(hrrr_row) else "missing",
        "synced" if spatial_columns <= set(nbm_row) and spatial_columns <= set(hrrr_row) else "drift",
        "Spatial join metadata should be present on both wide outputs.",
    )
    add(
        "raw_runtime_contract",
        "raw_spatial_suffix_contract",
        "present" if row_has_metric_family(nbm_row, "tmp") else "missing",
        "present" if row_has_metric_family(hrrr_row, "tmp_2m_k") else "missing",
        "synced" if row_has_metric_family(nbm_row, "tmp") and row_has_metric_family(hrrr_row, "tmp_2m_k") else "drift",
        "Raw outputs keep source-specific base names but share the unsuffixed nearest plus crop, nb3, and nb7 spatial suffix pattern.",
    )
    add(
        "raw_runtime_contract",
        "missing_optional_summary_columns",
        "present" if {"missing_optional_any", "missing_optional_fields_count"} <= set(nbm_row) else "missing",
        "present" if {"missing_optional_any", "missing_optional_fields_count"} <= set(hrrr_row) else "missing",
        "synced" if {"missing_optional_any", "missing_optional_fields_count"} <= set(nbm_row) and {"missing_optional_any", "missing_optional_fields_count"} <= set(hrrr_row) else "drift",
        "Both raw pipelines now expose row-level missing-optional summaries.",
    )
    add(
        "raw_runtime_contract",
        "raw_namespace_is_source_aware",
        "tmp/tmp-like",
        "tmp_2m_k/explicit-level",
        "allowed_difference",
        "Raw NBM and raw HRRR remain source-aware; the model-facing standard now lives in the canonical training schema instead of in the raw wide tables.",
    )

    add(
        "canonical_training_schema",
        "schema_module_present",
        "present" if CANONICAL_WIDE_COLUMNS else "missing",
        "present" if CANONICAL_WIDE_COLUMNS else "missing",
        "synced",
        "Canonical training wide columns are defined centrally in tools/weather/canonical_feature_schema.py.",
    )
    add(
        "canonical_training_schema",
        "wide_column_compatibility",
        "present" if set(CANONICAL_WIDE_COLUMNS) <= set(nbm_canonical_wide.columns) else "missing",
        "present" if set(CANONICAL_WIDE_COLUMNS) <= set(hrrr_canonical_wide.columns) else "missing",
        "synced" if list(nbm_canonical_wide.columns) == list(hrrr_canonical_wide.columns) else "drift",
        "The NBM and HRRR adapters now emit one stable canonical wide column layout with nullable passthrough metadata such as mode.",
    )
    add(
        "canonical_training_schema",
        "provenance_column_compatibility",
        "present" if set(CANONICAL_PROVENANCE_COLUMNS) <= set(nbm_canonical_provenance.columns) else "missing",
        "present" if set(CANONICAL_PROVENANCE_COLUMNS) <= set(hrrr_canonical_provenance.columns) else "missing",
        "synced" if list(nbm_canonical_provenance.columns) == list(hrrr_canonical_provenance.columns) else "drift",
        "The provenance adapters now emit one stable canonical lineage schema for downstream debugging and feature-traceability.",
    )
    add(
        "canonical_training_schema",
        "shared_feature_mapping_examples",
        "present" if {"temp_2m_k", "wind_10m_speed_ms", "wind_10m_u_ms", "wind_10m_v_ms"} <= set(nbm_canonical_wide.columns) else "missing",
        "present" if {"temp_2m_k", "wind_10m_speed_ms", "wind_10m_u_ms", "wind_10m_v_ms"} <= set(hrrr_canonical_wide.columns) else "missing",
        "synced"
        if not nbm_canonical_wide.empty
        and not hrrr_canonical_wide.empty
        and pd.notna(nbm_canonical_wide.iloc[0]["temp_2m_k"])
        and pd.notna(hrrr_canonical_wide.iloc[0]["temp_2m_k"])
        else "drift",
        "Shared physical features such as temperature and 10 m wind now normalize into the same explicit-unit namespace.",
    )

    provenance_identity = {"source_model", "source_product", "source_version", "station_id", "init_time_utc", "valid_time_utc", "feature_name", "units"}
    add(
        "provenance_minimum_contract",
        "core_provenance_identity",
        "present" if provenance_identity <= set(nbm_provenance) else "missing",
        "present" if provenance_identity <= set(hrrr_provenance) else "missing",
        "synced" if provenance_identity <= set(nbm_provenance) and provenance_identity <= set(hrrr_provenance) else "drift",
        "Both provenance outputs should expose feature identity, timestamps, and units.",
    )
    add(
        "provenance_minimum_contract",
        "grib_source_metadata",
        "present" if {"grib_short_name", "grib_level_text", "step_type"} <= set(nbm_provenance) else "missing",
        "present" if {"grib_short_name", "grib_level_text", "grib_step_type"} <= set(hrrr_provenance) else "missing",
        "synced" if {"grib_short_name", "grib_level_text", "step_type"} <= set(nbm_provenance) and {"grib_short_name", "grib_level_text", "grib_step_type"} <= set(hrrr_provenance) else "drift",
        "Both provenance tables expose GRIB short-name/level/step metadata, though field names differ slightly.",
    )
    add(
        "provenance_minimum_contract",
        "direct_derived_missing_semantics",
        "present" if {"present_directly", "derived", "missing_optional"} <= set(nbm_provenance) else "missing",
        "present" if {"present_directly", "derived", "missing_optional"} <= set(hrrr_provenance) else "missing",
        "synced"
        if {"present_directly", "derived", "missing_optional"} <= set(nbm_provenance)
        and {"present_directly", "derived", "missing_optional"} <= set(hrrr_provenance)
        else "drift",
        "Both raw provenance outputs now expose direct, derived, and missing-optional semantics.",
    )
    add(
        "provenance_minimum_contract",
        "derivation_lineage_fields",
        "present" if {"derivation_method", "source_feature_names", "fallback_source_description"} <= set(nbm_provenance) else "missing",
        "present" if {"derivation_method", "source_feature_names"} <= set(hrrr_provenance) else "missing",
        "synced"
        if {"derivation_method", "source_feature_names", "fallback_source_description"} <= set(nbm_provenance)
        and {"derivation_method", "source_feature_names"} <= set(hrrr_provenance)
        else "drift",
        "Both pipelines now expose derivation lineage and fallback descriptions through the raw provenance layer.",
    )
    add(
        "provenance_minimum_contract",
        "inventory_line_reference",
        "present" if "inventory_line" in nbm_provenance else "missing",
        "present" if "source_inventory_line" in hrrr_provenance else "missing",
        "synced" if "inventory_line" in nbm_provenance and "source_inventory_line" in hrrr_provenance else "drift",
        "Both pipelines retain a source inventory line reference for direct fields.",
    )

    nbm_manifest_semantics = normalize_nbm_manifest_semantics(nbm_manifest)
    hrrr_manifest_semantics = normalize_hrrr_manifest_semantics(hrrr_manifest, hrrr_manifest_json)
    add(
        "operational_manifest_contract",
        "manifest_parquet_artifact",
        "present",
        "present",
        "synced",
        "Both live pipelines write a Parquet manifest artifact; HRRR also retains an auxiliary JSON sidecar for month-level recovery state.",
    )
    for item, note in (
        ("tracks_failure_state", "Both manifests should expose failure/error state."),
        ("tracks_missing_fields", "Both manifests should expose missing/partial extraction information."),
        ("tracks_output_paths", "Both manifests should expose output artifact paths."),
        ("tracks_cleanup_state", "Both manifests should record or imply cleanup/retention policy."),
    ):
        add(
            "operational_manifest_contract",
            item,
            "present" if nbm_manifest_semantics[item] else "missing",
            "present" if hrrr_manifest_semantics[item] else "missing",
            "synced" if nbm_manifest_semantics[item] and hrrr_manifest_semantics[item] else "drift",
            note,
        )
    add(
        "operational_manifest_contract",
        "expected_completed_state",
        "present" if nbm_manifest_semantics["tracks_expected_state"] else "missing",
        "present" if hrrr_manifest_semantics["tracks_expected_state"] else "missing",
        "allowed_difference",
        "HRRR manifest JSON tracks month-level expected/completed task state; NBM manifest rows remain per-unit operational records.",
    )
    add(
        "operational_manifest_contract",
        "manifest_json_sidecar",
        "absent",
        "present",
        "allowed_difference",
        "HRRR keeps an auxiliary JSON manifest sidecar for resumable month builds; NBM does not need the same recovery artifact.",
    )

    add(
        "artifact_policy",
        "parquet_first_persistent_outputs",
        "present",
        "present",
        "synced",
        "Both production paths persist tabular outputs and treat GRIB as processing artifacts by default.",
    )
    add(
        "artifact_policy",
        "keep_flags_debug_only",
        "present",
        "present",
        "synced",
        "Both pipelines expose keep flags for raw/reduced artifacts while defaulting to cleanup.",
    )
    add(
        "artifact_policy",
        "nbm_long_output",
        "present",
        "absent",
        "allowed_difference",
        "NBM supports optional long-format output; HRRR intentionally keeps long output out of scope.",
    )

    drift_count = sum(row["classification"] == "drift" for row in rows)
    allowed_count = sum(row["classification"] == "allowed_difference" for row in rows)
    if drift_count:
        verdict = "not yet standardized"
    elif allowed_count:
        verdict = "standardized with accepted differences"
    else:
        verdict = "standardized"

    synced_items = [row for row in rows if row["classification"] == "synced"]
    allowed_items = [row for row in rows if row["classification"] == "allowed_difference"]
    drift_items = [row for row in rows if row["classification"] == "drift"]

    def bullet_lines(items: list[dict[str, str]]) -> str:
        if not items:
            return "- none"
        return "\n".join(f"- `{item['area']}.{item['item']}`: {item['notes']}" for item in items)

    review = "\n".join(
        [
            "# HRRR vs NBM Standardization Review",
            "",
            f"Verdict: **{verdict}**",
            "",
            "## Direct Answers",
            f"- Shared raw runtime contract: {'yes' if not any(row['area'] == 'raw_runtime_contract' and row['classification'] == 'drift' for row in rows) else 'no'}",
            f"- Canonical training schema available: {'yes' if not any(row['area'] == 'canonical_training_schema' and row['classification'] == 'drift' for row in rows) else 'no'}",
            f"- Provenance minimum synchronized: {'yes' if not any(row['area'] == 'provenance_minimum_contract' and row['classification'] == 'drift' for row in rows) else 'no'}",
            f"- Operational manifest minimum synchronized: {'yes' if not any(row['area'] == 'operational_manifest_contract' and row['classification'] == 'drift' for row in rows) else 'no'}",
            "",
            "## Synced Areas",
            bullet_lines(synced_items),
            "",
            "## Allowed Differences",
            bullet_lines(allowed_items),
            "",
            "## Drift Requiring Follow-up",
            bullet_lines(drift_items),
            "",
            "## Summary",
            f"- synced: {len(synced_items)}",
            f"- allowed_difference: {len(allowed_items)}",
            f"- drift: {len(drift_items)}",
        ]
    )
    return rows, review


def write_outputs(output_dir: pathlib.Path, matrix: list[dict[str, str]], review: str) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / OUTPUT_MATRIX_NAME
    review_path = output_dir / OUTPUT_REVIEW_NAME
    matrix_path.write_text(json.dumps(matrix, indent=2, sort_keys=True))
    review_path.write_text(review)
    return matrix_path, review_path


def main() -> int:
    args = parse_args()
    matrix, review = compare_contracts()
    matrix_path, review_path = write_outputs(args.output_dir, matrix, review)
    drift_count = sum(row["classification"] == "drift" for row in matrix)
    print(f"[ok] wrote {matrix_path} and {review_path} drift={drift_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
