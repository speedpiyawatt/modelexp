#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lamp.parse_lamp_ascii import (
    DEFAULT_OUTPUT_DIR as DEFAULT_PARSED_DIR,
    SOURCE_MODEL,
    SOURCE_PRODUCT,
    SOURCE_VERSION,
    discover_ascii_inputs,
    merge_station_frames,
    parse_bulletin_file,
)
from tools.weather.location_context import SETTLEMENT_LOCATION
from tools.weather.progress import ProgressBar


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "runtime" / "features"
PROGRESS_RENDER_EVERY = 100
CURATED_LABELS = (
    "TMP",
    "DPT",
    "WDR",
    "WSP",
    "WGS",
    "PPO",
    "P01",
    "P06",
    "P12",
    "PCO",
    "PC1",
    "LP1",
    "LC1",
    "CP1",
    "CC1",
    "POZ",
    "POS",
    "TYP",
    "CLD",
    "CIG",
    "CCG",
    "VIS",
    "CVS",
    "OBV",
)
OPTIONAL_LABELS = {"P06", "P12", "LP1", "LC1", "CP1", "CC1", "POZ", "POS", "TYP"}
CATEGORICAL_WIDE_LABELS = {"WGS", "PCO", "PC1", "LC1", "CC1", "TYP", "CLD", "OBV"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KLGA LAMP parquet features from raw ASCII bulletins.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="Raw ASCII files or directories.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station-id", default=SETTLEMENT_LOCATION.station_id)
    parser.add_argument("--write-long", action="store_true", help="Write the merged raw long parquet artifact.")
    return parser.parse_args()


def cycle_token_from_init_time(init_time_utc: str) -> str:
    timestamp = pd.Timestamp(init_time_utc)
    return timestamp.strftime("%H%M")


def issue_partition(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    if df.empty:
        return groups
    for init_time_utc, frame in df.groupby("init_time_utc", sort=True):
        groups.append((str(init_time_utc), frame.sort_values(["valid_time_utc", "raw_label"]).reset_index(drop=True)))
    return groups


def base_wide_identity(init_time_utc: pd.Timestamp, valid_time_utc: pd.Timestamp, *, forecast_hour: int) -> dict[str, object]:
    init_local = init_time_utc.tz_convert(NY_TZ)
    valid_local = valid_time_utc.tz_convert(NY_TZ)
    return {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "fallback_used_any": False,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "station_lat": SETTLEMENT_LOCATION.lat,
        "station_lon": SETTLEMENT_LOCATION.lon,
        "settlement_station_id": SETTLEMENT_LOCATION.station_id,
        "settlement_lat": SETTLEMENT_LOCATION.lat,
        "settlement_lon": SETTLEMENT_LOCATION.lon,
        "init_time_utc": init_time_utc.isoformat(),
        "init_time_local": init_local.isoformat(),
        "init_date_local": init_local.date().isoformat(),
        "valid_time_utc": valid_time_utc.isoformat(),
        "valid_time_local": valid_local.isoformat(),
        "valid_date_local": valid_local.date().isoformat(),
        "forecast_hour": forecast_hour,
    }


def label_column_name(label: str) -> str:
    return label.lower()


def wide_value_for_label(label: str, value: object) -> object:
    if value is None:
        return None
    if label in CATEGORICAL_WIDE_LABELS:
        return str(value)
    return value


def optional_provenance_row(
    *,
    init_time: pd.Timestamp,
    valid_time: pd.Timestamp,
    forecast_hour: int,
    raw_label: str,
    bulletin_types: str,
    bulletin_versions: str,
    bulletin_source_paths: str,
    archive_members: str,
) -> dict[str, object]:
    return {
        **base_wide_identity(init_time, valid_time, forecast_hour=forecast_hour),
        "feature_name": label_column_name(raw_label),
        "raw_feature_name": raw_label,
        "present_directly": False,
        "derived": False,
        "missing_optional": True,
        "derivation_method": None,
        "source_feature_names": json.dumps([]),
        "fallback_used": False,
        "fallback_source_description": None,
        "grib_short_name": None,
        "grib_level_text": None,
        "grib_type_of_level": None,
        "grib_step_type": None,
        "grib_step_text": None,
        "inventory_line": None,
        "units": None,
        "notes": "Optional LAMP label was not present in the parsed bulletin rows for this valid time.",
        "bulletin_type": bulletin_types,
        "bulletin_version": bulletin_versions,
        "bulletin_source_path": bulletin_source_paths,
        "archive_member": archive_members,
    }


def issue_warning_messages(*, wide_df: pd.DataFrame, issue_df: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    if issue_df.empty:
        warnings.append("No KLGA rows parsed from the selected input files.")
        return warnings
    if "missing_optional_any" not in wide_df.columns or "missing_optional_fields_count" not in wide_df.columns:
        return warnings
    missing_rows = wide_df.loc[wide_df["missing_optional_any"].fillna(False).astype(bool)]
    if missing_rows.empty:
        return warnings
    max_missing = int(pd.to_numeric(missing_rows["missing_optional_fields_count"], errors="coerce").fillna(0).max())
    warnings.append(
        f"Missing optional LAMP labels in {len(missing_rows)}/{len(wide_df)} forecast rows; max_missing_optional_fields_count={max_missing}."
    )
    return warnings


def build_issue_outputs(issue_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wide_rows: list[dict[str, object]] = []
    provenance_rows: list[dict[str, object]] = []
    long_df = issue_df.sort_values(["valid_time_utc", "raw_label"]).reset_index(drop=True)

    for valid_time_utc, frame in long_df.groupby("valid_time_utc", sort=True):
        init_time = pd.Timestamp(frame["init_time_utc"].iloc[0])
        valid_time = pd.Timestamp(valid_time_utc)
        forecast_hour = int(frame["forecast_hour"].iloc[0])
        row = base_wide_identity(init_time, valid_time, forecast_hour=forecast_hour)
        present_labels = set()
        for _, record in frame.iterrows():
            label = str(record["raw_label"])
            if label not in CURATED_LABELS:
                continue
            column_name = label_column_name(label)
            row[column_name] = wide_value_for_label(label, record["value"])
            present_labels.add(label)
            provenance_rows.append(
                {
                    **base_wide_identity(init_time, valid_time, forecast_hour=forecast_hour),
                    "feature_name": column_name,
                    "raw_feature_name": label,
                    "present_directly": True,
                    "derived": False,
                    "missing_optional": False,
                    "derivation_method": None,
                    "source_feature_names": json.dumps([]),
                    "fallback_used": False,
                    "fallback_source_description": None,
                    "grib_short_name": None,
                    "grib_level_text": None,
                    "grib_type_of_level": None,
                    "grib_step_type": None,
                    "grib_step_text": None,
                    "inventory_line": None,
                    "units": record["units"],
                    "notes": None,
                    "bulletin_type": record["bulletin_type"],
                    "bulletin_version": record["bulletin_version"],
                    "bulletin_source_path": record["bulletin_source_path"],
                    "archive_member": record["archive_member"],
                }
            )
        missing_optional = sorted(OPTIONAL_LABELS - present_labels)
        bulletin_types = ";".join(sorted(frame["bulletin_type"].dropna().astype(str).unique()))
        bulletin_versions = ";".join(sorted(frame["bulletin_version"].dropna().astype(str).unique()))
        bulletin_source_paths = ";".join(sorted(frame["bulletin_source_path"].dropna().astype(str).unique()))
        archive_members = ";".join(sorted(frame["archive_member"].dropna().astype(str).unique()))
        for label in missing_optional:
            provenance_rows.append(
                optional_provenance_row(
                    init_time=init_time,
                    valid_time=valid_time,
                    forecast_hour=forecast_hour,
                    raw_label=label,
                    bulletin_types=bulletin_types,
                    bulletin_versions=bulletin_versions,
                    bulletin_source_paths=bulletin_source_paths,
                    archive_members=archive_members,
                )
            )
        row["missing_optional_any"] = bool(missing_optional)
        row["missing_optional_fields_count"] = len(missing_optional)
        wide_rows.append(row)

    return (
        pd.DataFrame.from_records(wide_rows),
        long_df,
        pd.DataFrame.from_records(provenance_rows),
    )


def parquet_safe_long_df(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    safe_df["value_numeric"] = pd.to_numeric(safe_df["value"], errors="coerce")
    safe_df["value"] = safe_df["value"].map(lambda value: None if value is None else str(value))
    return safe_df


def manifest_row_for_issue(
    *,
    issue_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    provenance_df: pd.DataFrame,
    wide_path: pathlib.Path,
    long_path: pathlib.Path | None,
    provenance_path: pathlib.Path,
    manifest_path: pathlib.Path,
) -> dict[str, object]:
    init_time = pd.Timestamp(issue_df["init_time_utc"].iloc[0])
    cycle = init_time.strftime("%H%M")
    init_local = init_time.tz_convert(NY_TZ)
    warnings = issue_warning_messages(wide_df=wide_df, issue_df=issue_df)
    return {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "task_key": f"{init_time.date().isoformat()}T{cycle}Z",
        "status": "ok",
        "extraction_status": "ok",
        "processed_timestamp_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "init_time_utc": init_time.isoformat(),
        "init_time_local": init_local.isoformat(),
        "init_date_local": init_local.date().isoformat(),
        "cycle": cycle,
        "bulletin_types": ";".join(sorted(issue_df["bulletin_type"].dropna().astype(str).unique())),
        "raw_input_paths": ";".join(sorted(issue_df["bulletin_source_path"].dropna().astype(str).unique())),
        "archive_members": ";".join(sorted(issue_df["archive_member"].dropna().astype(str).unique())),
        "warnings": ";".join(warnings),
        "long_row_count": int(len(issue_df)),
        "wide_row_count": int(issue_df["valid_time_utc"].nunique()),
        "provenance_row_count": int(len(provenance_df)),
        "wide_output_path": str(wide_path),
        "long_output_path": str(long_path) if long_path is not None else None,
        "provenance_output_path": str(provenance_path),
        "manifest_parquet_path": str(manifest_path),
    }


def output_paths(output_dir: pathlib.Path, init_time_utc: str) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    timestamp = pd.Timestamp(init_time_utc)
    date_token = timestamp.date().isoformat()
    cycle = timestamp.strftime("%H%M")
    root = output_dir / f"station_id={SETTLEMENT_LOCATION.station_id}" / f"date_utc={date_token}" / f"cycle={cycle}"
    root.mkdir(parents=True, exist_ok=True)
    return (
        root / "lamp.wide.parquet",
        root / "lamp.long.parquet",
        root / "lamp.provenance.parquet",
        root / "lamp.manifest.parquet",
    )


def update_progress_sparse(
    progress: ProgressBar,
    *,
    completed: int,
    total: int,
    stage: str,
    status: str,
) -> None:
    if total <= PROGRESS_RENDER_EVERY or completed == total or completed % PROGRESS_RENDER_EVERY == 0:
        current_completed = int(getattr(progress, "completed", completed - 1))
        progress.advance(max(0, completed - current_completed), stage=stage, status=status)
    else:
        progress.update(completed=completed, stage=stage, status=status)


def build_from_inputs(inputs: list[pathlib.Path], *, station_id: str, output_dir: pathlib.Path, write_long: bool) -> list[pathlib.Path]:
    parse_progress = ProgressBar(len(inputs), label="LAMP parse", unit="input") if inputs else None
    frames = []
    for input_index, path in enumerate(inputs, start=1):
        if parse_progress is not None:
            parse_progress.update(stage="parse", status=f"input={path.name}")
        frames.append(parse_bulletin_file(path, station_id=station_id))
        if parse_progress is not None:
            update_progress_sparse(
                parse_progress,
                completed=input_index,
                total=len(inputs),
                stage="parse",
                status=f"input={path.name} done",
            )
    if parse_progress is not None:
        parse_progress.close(stage="merge", status=f"parsed_inputs={len(inputs)}")
    merged = merge_station_frames(frames)
    if merged.empty or "station_id" not in merged.columns:
        raise SystemExit(f"No parsed LAMP rows found for station_id={station_id}")
    merged = merged.loc[merged["station_id"] == station_id].reset_index(drop=True)
    if merged.empty:
        raise SystemExit(f"No parsed LAMP rows found for station_id={station_id}")

    issues = issue_partition(merged)
    written_paths: list[pathlib.Path] = []
    progress = ProgressBar(len(issues), label="LAMP features", unit="issue")
    for issue_index, (init_time_utc, issue_df) in enumerate(issues, start=1):
        progress.update(stage="build", status=f"issue={init_time_utc} wide_long_provenance")
        wide_df, long_df, provenance_df = build_issue_outputs(issue_df)
        wide_path, long_path, provenance_path, manifest_path = output_paths(output_dir, init_time_utc)
        progress.update(stage="write", status=f"issue={init_time_utc} wide")
        wide_df.to_parquet(wide_path, index=False)
        written_paths.append(wide_path)
        if write_long:
            progress.update(stage="write", status=f"issue={init_time_utc} long")
            parquet_safe_long_df(long_df).to_parquet(long_path, index=False)
            written_paths.append(long_path)
        else:
            long_path = None
        progress.update(stage="write", status=f"issue={init_time_utc} provenance")
        provenance_df.to_parquet(provenance_path, index=False)
        written_paths.append(provenance_path)
        manifest_df = pd.DataFrame.from_records(
            [
                manifest_row_for_issue(
                    issue_df=issue_df,
                    wide_df=wide_df,
                    provenance_df=provenance_df,
                    wide_path=wide_path,
                    long_path=long_path,
                    provenance_path=provenance_path,
                    manifest_path=manifest_path,
                )
            ]
        )
        progress.update(stage="write", status=f"issue={init_time_utc} manifest")
        manifest_df.to_parquet(manifest_path, index=False)
        written_paths.append(manifest_path)
        update_progress_sparse(
            progress,
            completed=issue_index,
            total=len(issues),
            stage="complete",
            status=f"issue={init_time_utc} done",
        )
    progress.close(stage="finalize", status=f"wrote_files={len(written_paths)}")
    return written_paths


def main() -> int:
    args = parse_args()
    inputs = discover_ascii_inputs(args.inputs)
    written = build_from_inputs(inputs, station_id=args.station_id, output_dir=args.output_dir, write_long=args.write_long)
    print(f"[ok] files={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
