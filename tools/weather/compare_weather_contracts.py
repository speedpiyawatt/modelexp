from __future__ import annotations

import argparse
import json
from pathlib import Path
import pathlib
import sys

import pandas as pd

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.canonical_feature_schema import RUNTIME_IDENTITY_COLUMNS, RUNTIME_METADATA_COLUMNS, RUNTIME_SPATIAL_COLUMNS

RAW_RUNTIME_WIDE_METADATA_COLUMNS = set(RUNTIME_METADATA_COLUMNS)
GRID_SOURCE_WIDE_COLUMNS = set(RUNTIME_IDENTITY_COLUMNS + RUNTIME_SPATIAL_COLUMNS + ("fallback_used_any",))
LAMP_WIDE_COLUMNS = set(RUNTIME_IDENTITY_COLUMNS + ("fallback_used_any",))
GRID_PROVENANCE_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "init_time_utc",
    "init_time_local",
    "init_date_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "nearest_grid_lat",
    "nearest_grid_lon",
    "feature_name",
    "fallback_used",
    "fallback_source_description",
    "present_directly",
    "derived",
    "missing_optional",
    "derivation_method",
    "source_feature_names",
}
LAMP_PROVENANCE_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "init_time_utc",
    "init_time_local",
    "init_date_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "feature_name",
    "raw_feature_name",
    "fallback_used",
    "fallback_source_description",
    "present_directly",
    "derived",
    "missing_optional",
    "derivation_method",
    "source_feature_names",
    "bulletin_type",
    "bulletin_version",
    "bulletin_source_path",
    "archive_member",
}
MANIFEST_REQUIRED_COLUMNS = {"source_model", "source_product", "source_version"}
MANIFEST_TASK_IDENTIFIER_COLUMNS = {"task_key", "lead_hour"}
MANIFEST_STATUS_COLUMNS = {"status", "extraction_status"}
MANIFEST_WIDE_PATH_COLUMNS = {"wide_parquet_path", "wide_output_path", "wide_output_paths"}
MANIFEST_PROVENANCE_PATH_COLUMNS = {"provenance_path", "provenance_output_path", "provenance_output_paths"}
MANIFEST_STATE_COLUMNS = {"failure_reason", "warnings", "missing_fields"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nbm-wide", type=Path, required=True)
    parser.add_argument("--nbm-provenance", type=Path, required=True)
    parser.add_argument("--nbm-manifest", type=Path, required=True)
    parser.add_argument("--hrrr-wide", type=Path, required=True)
    parser.add_argument("--hrrr-provenance", type=Path, required=True)
    parser.add_argument("--hrrr-manifest", type=Path, required=True)
    parser.add_argument("--lamp-wide", type=Path)
    parser.add_argument("--lamp-provenance", type=Path)
    parser.add_argument("--lamp-manifest", type=Path)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser.parse_args()


def legacy_alias_columns(columns: list[str]) -> list[str]:
    return sorted(column for column in columns if "_nearest" in column or "_3x3_" in column or "_7x7_" in column)


def longitude_convention(series: pd.Series) -> str:
    finite = pd.to_numeric(series, errors="coerce").dropna()
    if finite.empty:
        return "unknown"
    return "0-360" if finite.max() > 180 else "-180..180"


def classify_surface(*, missing: list[str], legacy_aliases: list[str], longitude: str, require_wgs84: bool = True) -> str:
    problems = 0
    if missing:
        problems += 1
    if legacy_aliases:
        problems += 1
    if require_wgs84 and longitude != "-180..180":
        problems += 1
    if problems == 0:
        return "synced"
    if problems == 1:
        return "partially synced"
    return "divergent"


def analyze_wide(path: Path, *, source: str) -> dict[str, object]:
    df = pd.read_parquet(path)
    columns = list(df.columns)
    required_columns = GRID_SOURCE_WIDE_COLUMNS if source in {"nbm", "hrrr"} else LAMP_WIDE_COLUMNS
    require_wgs84 = source in {"nbm", "hrrr"}
    missing = sorted((required_columns | RAW_RUNTIME_WIDE_METADATA_COLUMNS) - set(columns))
    legacy = legacy_alias_columns(columns)
    return {
        "path": str(path),
        "rows": len(df),
        "missing_columns": missing,
        "legacy_alias_columns": legacy,
        "nearest_grid_lon_convention": longitude_convention(df["nearest_grid_lon"]) if "nearest_grid_lon" in df.columns else "missing",
        "status": classify_surface(
            missing=missing,
            legacy_aliases=legacy,
            longitude=longitude_convention(df["nearest_grid_lon"]) if "nearest_grid_lon" in df.columns else "missing",
            require_wgs84=require_wgs84,
        ),
    }


def analyze_provenance(path: Path, *, source: str) -> dict[str, object]:
    df = pd.read_parquet(path)
    required = GRID_PROVENANCE_COLUMNS if source in {"nbm", "hrrr"} else LAMP_PROVENANCE_COLUMNS
    missing = sorted(required - set(df.columns))
    return {
        "path": str(path),
        "rows": len(df),
        "missing_columns": missing,
        "status": "synced" if not missing else "partially synced",
    }


def analyze_manifest(path: Path) -> dict[str, object]:
    df = pd.read_parquet(path)
    columns = set(df.columns)
    missing: list[str] = []
    if MANIFEST_REQUIRED_COLUMNS - columns:
        missing.extend(sorted(MANIFEST_REQUIRED_COLUMNS - columns))
    if not (MANIFEST_TASK_IDENTIFIER_COLUMNS & columns):
        missing.append("task_identifier")
    if not (MANIFEST_STATUS_COLUMNS & columns):
        missing.append("status_field")
    if not (MANIFEST_WIDE_PATH_COLUMNS & columns):
        missing.append("wide_output_path")
    if not (MANIFEST_PROVENANCE_PATH_COLUMNS & columns):
        missing.append("provenance_output_path")
    if not (MANIFEST_STATE_COLUMNS & columns):
        missing.append("state_tracking")
    return {
        "path": str(path),
        "rows": len(df),
        "missing_columns": missing,
        "status": "synced" if not missing else "partially synced",
    }


def source_report(*, name: str, wide: Path, provenance: Path, manifest: Path) -> dict[str, object]:
    return {
        "wide": analyze_wide(wide, source=name),
        "provenance": analyze_provenance(provenance, source=name),
        "manifest": analyze_manifest(manifest),
    }


def build_report(args: argparse.Namespace) -> dict[str, object]:
    report = {
        "canonical_training_schema": "tools.weather.canonical_feature_schema",
        "nbm": source_report(name="nbm", wide=args.nbm_wide, provenance=args.nbm_provenance, manifest=args.nbm_manifest),
        "hrrr": source_report(name="hrrr", wide=args.hrrr_wide, provenance=args.hrrr_provenance, manifest=args.hrrr_manifest),
    }
    optional_sources: list[str] = []
    lamp_wide = getattr(args, "lamp_wide", None)
    lamp_provenance = getattr(args, "lamp_provenance", None)
    lamp_manifest = getattr(args, "lamp_manifest", None)
    if lamp_wide and lamp_provenance and lamp_manifest:
        report["lamp"] = source_report(name="lamp", wide=lamp_wide, provenance=lamp_provenance, manifest=lamp_manifest)
        optional_sources.append("lamp")
    hrrr_statuses = [report["hrrr"]["wide"]["status"], report["hrrr"]["provenance"]["status"], report["hrrr"]["manifest"]["status"]]
    nbm_statuses = [report["nbm"]["wide"]["status"], report["nbm"]["provenance"]["status"], report["nbm"]["manifest"]["status"]]
    raw_runtime_statuses = [
        report["nbm"]["wide"]["status"],
        report["nbm"]["provenance"]["status"],
        report["hrrr"]["wide"]["status"],
        report["hrrr"]["provenance"]["status"],
    ]
    all_statuses = nbm_statuses + hrrr_statuses
    for source_name in optional_sources:
        source_statuses = [
            report[source_name]["wide"]["status"],
            report[source_name]["provenance"]["status"],
            report[source_name]["manifest"]["status"],
        ]
        raw_runtime_statuses.extend(source_statuses[:2])
        all_statuses.extend(source_statuses)
    report["verdict"] = {
        "raw_runtime_contract_shared": all(status == "synced" for status in raw_runtime_statuses),
        "hrrr_fully_aligned": all(status == "synced" for status in hrrr_statuses),
        "overall_status": "synced" if all(status == "synced" for status in all_statuses) else "not fully aligned",
    }
    report["comparison_matrix"] = [
        {
            "surface": "wide",
            "nbm_status": report["nbm"]["wide"]["status"],
            "hrrr_status": report["hrrr"]["wide"]["status"],
            "remediation_action": (
                "remove legacy aliases, add missing columns, and normalize nearest_grid_lon"
                if report["nbm"]["wide"]["status"] != "synced" or report["hrrr"]["wide"]["status"] != "synced"
                else ""
            ),
        },
        {
            "surface": "provenance",
            "nbm_status": report["nbm"]["provenance"]["status"],
            "hrrr_status": report["hrrr"]["provenance"]["status"],
            "remediation_action": (
                "add missing canonical identity and fallback metadata"
                if report["nbm"]["provenance"]["status"] != "synced" or report["hrrr"]["provenance"]["status"] != "synced"
                else ""
            ),
        },
        {
            "surface": "manifest",
            "nbm_status": report["nbm"]["manifest"]["status"],
            "hrrr_status": report["hrrr"]["manifest"]["status"],
            "remediation_action": (
                "write manifest artifacts with source metadata, task/status identifiers, output paths, and state tracking"
                if report["nbm"]["manifest"]["status"] != "synced" or report["hrrr"]["manifest"]["status"] != "synced"
                else ""
            ),
        },
    ]
    if "lamp" in report:
        report["comparison_matrix"].append(
            {
                "surface": "lamp",
                "nbm_status": report["lamp"]["wide"]["status"],
                "hrrr_status": report["lamp"]["provenance"]["status"],
                "remediation_action": (
                    "add missing LAMP runtime identity, provenance semantics, or manifest state fields"
                    if any(
                        report["lamp"][surface]["status"] != "synced" for surface in ("wide", "provenance", "manifest")
                    )
                    else ""
                ),
            }
        )
    return report


def report_markdown(report: dict[str, object]) -> str:
    lines = [
        "# Weather Contract Compatibility Report",
        "",
        f"- Canonical training schema: `{report['canonical_training_schema']}`",
        f"- Verdict: `{report['verdict']['overall_status']}`",
        f"- HRRR fully aligned: `{report['verdict']['hrrr_fully_aligned']}`",
        f"- Shared raw runtime contract: `{report['verdict']['raw_runtime_contract_shared']}`",
        "",
        "| Surface | NBM | HRRR | Remediation |",
        "| --- | --- | --- | --- |",
    ]
    for row in report["comparison_matrix"]:
        lines.append(f"| {row['surface']} | {row['nbm_status']} | {row['hrrr_status']} | {row['remediation_action']} |")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(report_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
