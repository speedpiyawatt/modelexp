#!/usr/bin/env python3
"""Probe HRRR-Zarr archive coverage for the KLGA overnight feature contract.

This is a Phase 4 prototype helper. It intentionally starts with metadata and
coverage checks before trying to replace the GRIB path. The default probe uses
only HTTPS requests against the public ``hrrrzarr`` S3 endpoint, so it does not
require optional zarr/s3fs dependencies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hrrr import build_hrrr_klga_feature_shards as hrrr


S3_ENDPOINT = "https://hrrrzarr.s3.amazonaws.com"


@dataclass(frozen=True)
class ZarrField:
    feature_name: str
    level: str
    variable: str
    notes: str = ""

    @property
    def array_path(self) -> str:
        return f"{self.level}/{self.variable}/{self.level}/{self.variable}"


ZARR_FIELDS: tuple[ZarrField, ...] = (
    ZarrField("tmp_2m_k", "2m_above_ground", "TMP"),
    ZarrField("dpt_2m_k", "2m_above_ground", "DPT"),
    ZarrField("rh_2m_pct", "2m_above_ground", "RH"),
    ZarrField("ugrd_10m_ms", "10m_above_ground", "UGRD"),
    ZarrField("vgrd_10m_ms", "10m_above_ground", "VGRD"),
    ZarrField("gust_surface_ms", "surface", "GUST"),
    ZarrField("surface_pressure_pa", "surface", "PRES"),
    ZarrField("mslma_pa", "mean_sea_level", "MSLMA"),
    ZarrField("visibility_m", "surface", "VIS"),
    ZarrField("lcdc_low_pct", "low_cloud_layer", "LCDC"),
    ZarrField("mcdc_mid_pct", "middle_cloud_layer", "MCDC"),
    ZarrField("hcdc_high_pct", "high_cloud_layer", "HCDC"),
    ZarrField("tcdc_entire_pct", "entire_atmosphere", "TCDC"),
    ZarrField("dswrf_surface_w_m2", "surface", "DSWRF", "Known HRRR-Zarr constant-field gap risk at night."),
    ZarrField("dlwrf_surface_w_m2", "surface", "DLWRF"),
    ZarrField("apcp_surface_kg_m2", "surface", "APCP_acc_fcst", "Cumulative APCP; forecast hour 0 may be absent."),
    ZarrField("apcp_1hr_surface_kg_m2", "surface", "APCP_1hr_acc_fcst", "Hourly APCP fallback candidate."),
    ZarrField("prate_surface_kg_m2_s", "surface", "PRATE"),
    ZarrField("hpbl_m", "surface", "HPBL"),
    ZarrField("pwat_entire_atmosphere_kg_m2", "entire_atmosphere_single_layer", "PWAT"),
    ZarrField("cape_surface_j_kg", "surface", "CAPE"),
    ZarrField("cin_surface_j_kg", "surface", "CIN"),
    ZarrField("refc_entire_atmosphere", "entire_atmosphere", "REFC"),
    ZarrField("ltng_entire_atmosphere", "entire_atmosphere", "LTNG"),
)

for level in hrrr.ISOBARIC_LEVELS:
    level_name = f"{level}mb"
    ZARR_FIELDS += (
        ZarrField(f"tmp_{level}mb_k", level_name, "TMP"),
        ZarrField(f"dpt_{level}mb_k_support", level_name, "DPT", "Support field for derived RH fallback."),
        ZarrField(f"ugrd_{level}mb_ms", level_name, "UGRD"),
        ZarrField(f"vgrd_{level}mb_ms", level_name, "VGRD"),
        ZarrField(f"rh_{level}mb_pct", level_name, "RH"),
        ZarrField(f"spfh_{level}mb_kg_kg", level_name, "SPFH"),
        ZarrField(f"hgt_{level}mb_gpm", level_name, "HGT"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2023-02-04", help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2023-02-04", help="Inclusive target local date in YYYY-MM-DD.")
    parser.add_argument(
        "--selection-mode",
        choices=hrrr.SELECTION_MODES,
        default="overnight_0005",
        help="Retained-cycle policy to test.",
    )
    parser.add_argument(
        "--summary-profile",
        choices=hrrr.SUMMARY_PROFILES,
        default="overnight",
        help="Use overnight to check revision cycles against the narrow revision field set.",
    )
    parser.add_argument(
        "--field-scope",
        choices=("profile", "all"),
        default="profile",
        help="profile checks fields required by each task profile; all checks every mapped field for every task.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--output-json", type=Path, help="Optional path for the full probe report.")
    return parser.parse_args()


def iter_dates(start: str, end: str) -> Iterable[pd.Timestamp]:
    current = pd.Timestamp(start).normalize()
    stop = pd.Timestamp(end).normalize()
    while current <= stop:
        yield current
        current += pd.Timedelta(days=1)


def field_names_for_task(task: hrrr.TaskSpec, *, summary_profile: str, field_scope: str) -> set[str]:
    if field_scope == "all":
        return {field.feature_name for field in ZARR_FIELDS}
    original_options = hrrr.RUNTIME_OPTIONS
    try:
        hrrr.RUNTIME_OPTIONS = hrrr.HrrrRuntimeOptions(summary_profile=summary_profile)
        names = set(hrrr.requested_field_prefixes_for_task(task))
        if any(name.startswith("rh_") and name.endswith("mb_pct") for name in names):
            names.update(f"dpt_{level}mb_k_support" for level in hrrr.ISOBARIC_LEVELS)
        if "apcp_surface_kg_m2" in names:
            names.add("apcp_1hr_surface_kg_m2")
        return names
    finally:
        hrrr.RUNTIME_OPTIONS = original_options


def zarr_root_prefix(task: hrrr.TaskSpec) -> str:
    date_token = task.run_date_utc.replace("-", "")
    return f"sfc/{date_token}/{date_token}_{task.cycle_hour_utc:02d}z_fcst.zarr"


def fetch_json(url: str, *, timeout_seconds: float) -> tuple[int, dict[str, object] | None, str | None]:
    try:
        response = requests.get(url, timeout=timeout_seconds)
    except requests.RequestException as exc:
        return 0, None, str(exc)
    if response.status_code != 200:
        return response.status_code, None, response.text[:200]
    try:
        return response.status_code, response.json(), None
    except ValueError as exc:
        return response.status_code, None, str(exc)


def load_array_metadata(
    array_prefix: str,
    *,
    timeout_seconds: float,
    metadata_cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    if array_prefix in metadata_cache:
        return metadata_cache[array_prefix]
    zarray_url = f"{S3_ENDPOINT}/{array_prefix}/.zarray"
    status_code, metadata, error = fetch_json(zarray_url, timeout_seconds=timeout_seconds)
    payload: dict[str, object] = {
        "status_code": status_code,
        "exists": metadata is not None,
        "shape": None,
        "chunks": None,
        "dtype": None,
    }
    if error is not None:
        payload["error"] = error
    if metadata is not None:
        payload["shape"] = list(metadata.get("shape") or [])
        payload["chunks"] = list(metadata.get("chunks") or [])
        payload["dtype"] = metadata.get("dtype")
    metadata_cache[array_prefix] = payload
    return payload


def check_field(
    task: hrrr.TaskSpec,
    field: ZarrField,
    *,
    timeout_seconds: float,
    metadata_cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    root = zarr_root_prefix(task)
    array_prefix = f"{root}/{field.array_path}"
    metadata = load_array_metadata(array_prefix, timeout_seconds=timeout_seconds, metadata_cache=metadata_cache)
    result: dict[str, object] = {
        "feature_name": field.feature_name,
        "level": field.level,
        "variable": field.variable,
        "array_prefix": array_prefix,
        "status_code": metadata["status_code"],
        "exists": metadata["exists"],
        "forecast_hour": task.forecast_hour,
        "covered": False,
        "shape": metadata["shape"],
        "chunks": metadata["chunks"],
        "dtype": metadata["dtype"],
        "notes": field.notes,
    }
    if "error" in metadata:
        result["error"] = metadata["error"]
    if not bool(metadata["exists"]):
        return result

    shape = list(metadata["shape"] or [])
    chunks = list(metadata["chunks"] or [])
    if shape and isinstance(shape[0], int):
        result["covered"] = int(task.forecast_hour) < int(shape[0])
    else:
        result["covered"] = True

    chunk_key: str | None = None
    if chunks and len(chunks) == 3:
        chunk_key = f"{int(task.forecast_hour) // int(chunks[0])}.0.0"
        result["sample_chunk_key"] = chunk_key
    return result


def main() -> int:
    args = parse_args()
    field_by_name = {field.feature_name: field for field in ZARR_FIELDS}
    records: list[dict[str, object]] = []
    metadata_cache: dict[str, dict[str, object]] = {}

    for target_date in iter_dates(args.start_date, args.end_date):
        tasks = hrrr.build_tasks_for_target_date(target_date, selection_mode=args.selection_mode)
        for task in tasks:
            for field_name in sorted(field_names_for_task(task, summary_profile=args.summary_profile, field_scope=args.field_scope)):
                field = field_by_name.get(field_name)
                if field is None:
                    records.append(
                        {
                            "target_date_local": task.target_date_local,
                            "task_key": task.key,
                            "feature_name": field_name,
                            "exists": False,
                            "covered": False,
                            "error": "No HRRR-Zarr mapping defined.",
                        }
                    )
                    continue
                checked = check_field(
                    task,
                    field,
                    timeout_seconds=args.timeout_seconds,
                    metadata_cache=metadata_cache,
                )
                checked.update(
                    {
                        "target_date_local": task.target_date_local,
                        "task_key": task.key,
                        "run_date_utc": task.run_date_utc,
                        "cycle_hour_utc": task.cycle_hour_utc,
                        "anchor_cycle_candidate": task.anchor_cycle_candidate,
                    }
                )
                records.append(checked)

    missing = [record for record in records if not bool(record.get("exists"))]
    uncovered = [record for record in records if bool(record.get("exists")) and not bool(record.get("covered"))]
    unique_missing = sorted({str(record.get("array_prefix") or record.get("feature_name")) for record in missing})
    unique_uncovered = sorted(
        {
            f"{record.get('array_prefix')} f{int(record.get('forecast_hour', -1)):02d} shape={record.get('shape')}"
            for record in uncovered
        }
    )
    report = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": "https://hrrrzarr.s3.amazonaws.com",
        "start_date": args.start_date,
        "end_date": args.end_date,
        "selection_mode": args.selection_mode,
        "summary_profile": args.summary_profile,
        "field_scope": args.field_scope,
        "record_count": len(records),
        "unique_array_count": len(metadata_cache),
        "missing_count": len(missing),
        "unique_missing_count": len(unique_missing),
        "unique_missing": unique_missing,
        "uncovered_count": len(uncovered),
        "unique_uncovered_count": len(unique_uncovered),
        "unique_uncovered": unique_uncovered,
        "records": records,
    }
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key != "records"
            },
            indent=2,
            sort_keys=True,
        )
    )
    if missing:
        print("\nMissing mappings or arrays:")
        for item in unique_missing[:50]:
            print(f"- {item}")
    if uncovered:
        print("\nExisting arrays without requested forecast-hour coverage:")
        for item in unique_uncovered[:50]:
            print(f"- {item}")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output_json}")
    return 1 if missing or uncovered else 0


if __name__ == "__main__":
    raise SystemExit(main())
