#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import re
import sys
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.location_context import SETTLEMENT_LOCATION


UTC = dt.timezone.utc
NY_TZ = ZoneInfo("America/New_York")
SOURCE_MODEL = "LAMP"
SOURCE_PRODUCT = "lavtxt"
SOURCE_VERSION = "lamp-station-ascii-public"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "runtime" / "parsed"
HEADER_RE = re.compile(
    r"^\s*(?P<station>[A-Z0-9]{4})\s+GFS LAMP GUIDANCE\s+"
    r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\s+(?P<hhmm>\d{4})\s+UTC\s*$"
)
ROW_LABEL_RE = re.compile(r"^\s*(?P<label>[A-Z0-9]{2,4})\s")
NUMERIC_LABELS = {
    "TMP",
    "DPT",
    "WDR",
    "WSP",
    "PPO",
    "P01",
    "P06",
    "P12",
    "CIG",
    "CCG",
    "VIS",
    "CVS",
    "LP1",
    "CP1",
    "POZ",
    "POS",
}
MAYBE_NUMERIC_LABELS = {"WGS"}
LABEL_UNITS = {
    "TMP": "F",
    "DPT": "F",
    "WDR": "tens_of_degrees",
    "WSP": "kt",
    "WGS": "kt_or_code",
    "PPO": "pct",
    "P01": "pct",
    "P06": "pct",
    "P12": "pct",
    "P01": "pct",
    "PCO": "yes_no",
    "PC1": "yes_no",
    "LP1": "pct",
    "LC1": "yes_no",
    "CP1": "pct",
    "CC1": "yes_no",
    "POZ": "pct",
    "POS": "pct",
    "TYP": "code",
    "CLD": "code",
    "CIG": "ordinal_code",
    "CCG": "ordinal_code",
    "VIS": "ordinal_code",
    "CVS": "ordinal_code",
    "OBV": "code",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse NOAA LAMP ASCII bulletins into long-form parquet.")
    parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="ASCII bulletin files or directories.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR, help="Parsed parquet output directory.")
    parser.add_argument("--station-id", default=None, help="Optional station filter, for example KLGA.")
    return parser.parse_args()


def discover_ascii_inputs(paths: Iterable[pathlib.Path]) -> list[pathlib.Path]:
    discovered: list[pathlib.Path] = []
    for path in paths:
        if path.is_dir():
            discovered.extend(sorted(candidate for candidate in path.rglob("*.ascii") if candidate.is_file()))
        elif path.is_file():
            discovered.append(path)
    unique: dict[str, pathlib.Path] = {}
    for path in discovered:
        unique[str(path.resolve())] = path
    return list(unique.values())


def issue_time_from_header_line(line: str) -> dt.datetime | None:
    match = HEADER_RE.match(line.rstrip())
    if not match:
        return None
    info = match.groupdict()
    hour = int(info["hhmm"][:2])
    minute = int(info["hhmm"][2:])
    return dt.datetime(
        year=int(info["year"]),
        month=int(info["month"]),
        day=int(info["day"]),
        hour=hour,
        minute=minute,
        tzinfo=UTC,
    )


def infer_bulletin_type(path: pathlib.Path, forecast_hours: list[int]) -> str:
    lowered = path.name.lower()
    if "lavtxt_ext" in lowered:
        return "extended"
    if "lavtxt" in lowered and path.name.endswith(".ascii") and max(forecast_hours, default=0) <= 25:
        return "standard"
    if forecast_hours and min(forecast_hours) >= 26:
        return "extended"
    if forecast_hours and max(forecast_hours) > 25:
        return "full"
    return "standard"


def parse_fixed_width_cells(line: str, *, count: int | None = None) -> tuple[str, list[str | None]]:
    normalized = line.rstrip("\n")
    label = normalized[:4].strip()
    # LAMP rows place the first value field after a separating spacer column.
    body = normalized[5:] if len(normalized) > 5 else ""
    if count is not None:
        body = body.ljust(count * 3)
        width = count * 3
    else:
        width = len(body)
        remainder = width % 3
        if remainder:
            width += 3 - remainder
        body = body.ljust(width)
    cells = [body[index : index + 3].strip() or None for index in range(0, width, 3)]
    return label, cells


def forecast_hours_for_block(row_map: dict[str, str], issue_time_utc: dt.datetime) -> tuple[list[int], list[int]]:
    if "HR" in row_map:
        _, hr_cells = parse_fixed_width_cells(row_map["HR"])
        forecast_hours = [int(token) for token in hr_cells if token]
    elif "UTC" in row_map:
        _, utc_cells = parse_fixed_width_cells(row_map["UTC"])
        forecast_hours = list(range(1, len([token for token in utc_cells if token]) + 1))
    else:
        raise ValueError(f"{issue_time_utc.isoformat()} block is missing a UTC row")
    _, utc_cells = parse_fixed_width_cells(row_map["UTC"], count=len(forecast_hours))
    utc_hours = [int(token) for token in utc_cells if token]
    return forecast_hours, utc_hours


def first_valid_time_utc(issue_time_utc: dt.datetime) -> dt.datetime:
    return issue_time_utc.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)


def valid_times_for_forecast_hours(issue_time_utc: dt.datetime, forecast_hours: list[int]) -> list[dt.datetime]:
    first_valid = first_valid_time_utc(issue_time_utc)
    return [first_valid + dt.timedelta(hours=forecast_hour - 1) for forecast_hour in forecast_hours]


def parse_value_token(label: str, token: str) -> object | None:
    if token in {"", "X"}:
        return None
    if label in NUMERIC_LABELS:
        if set(token) == {"9"}:
            return None
        return int(token)
    if label in MAYBE_NUMERIC_LABELS:
        if token == "NG":
            return token
        if set(token) == {"9"}:
            return None
        return int(token)
    if set(token) == {"9"}:
        return None
    return token


def parse_station_block(
    *,
    header_line: str,
    body_lines: list[str],
    source_path: pathlib.Path,
    source_url: str | None = None,
    archive_member: str | None = None,
) -> list[dict[str, object]]:
    match = HEADER_RE.match(header_line.rstrip())
    if not match:
        return []
    station_id = match.group("station")
    issue_time_utc = issue_time_from_header_line(header_line)
    if issue_time_utc is None:
        return []

    row_map: dict[str, str] = {}
    for line in body_lines:
        if not line.strip():
            continue
        label_match = ROW_LABEL_RE.match(line)
        if not label_match:
            continue
        row_map[label_match.group("label")] = line.rstrip("\n")

    if "UTC" not in row_map:
        return []

    forecast_hours, utc_hours = forecast_hours_for_block(row_map, issue_time_utc)
    valid_times_utc = valid_times_for_forecast_hours(issue_time_utc, forecast_hours)
    bulletin_type = infer_bulletin_type(source_path, forecast_hours)

    rows: list[dict[str, object]] = []
    for label, line in row_map.items():
        if label in {"DT", "HR", "UTC"}:
            continue
        _, value_cells = parse_fixed_width_cells(line, count=len(forecast_hours))
        for index, forecast_hour in enumerate(forecast_hours):
            token = value_cells[index] if index < len(value_cells) else None
            if token is None:
                continue
            parsed_value = parse_value_token(label, token)
            if parsed_value is None:
                continue
            valid_time_utc = valid_times_utc[index]
            valid_time_local = valid_time_utc.astimezone(NY_TZ)
            issue_time_local = issue_time_utc.astimezone(NY_TZ)
            rows.append(
                {
                    "source_model": SOURCE_MODEL,
                    "source_product": SOURCE_PRODUCT,
                    "source_version": SOURCE_VERSION,
                    "station_id": station_id,
                    "bulletin_type": bulletin_type,
                    "bulletin_version": None,
                    "bulletin_source_path": str(source_path),
                    "bulletin_source_url": source_url,
                    "archive_member": archive_member,
                    "raw_label": label,
                    "raw_value": token,
                    "value": parsed_value,
                    "units": LABEL_UNITS.get(label),
                    "init_time_utc": issue_time_utc.isoformat(),
                    "init_time_local": issue_time_local.isoformat(),
                    "init_date_local": issue_time_local.date().isoformat(),
                    "valid_time_utc": valid_time_utc.isoformat(),
                    "valid_time_local": valid_time_local.isoformat(),
                    "valid_date_local": valid_time_local.date().isoformat(),
                    "forecast_hour": forecast_hour,
                    "utc_hour_token": utc_hours[index] if index < len(utc_hours) else None,
                    "station_lat": SETTLEMENT_LOCATION.lat if station_id == SETTLEMENT_LOCATION.station_id else None,
                    "station_lon": SETTLEMENT_LOCATION.lon if station_id == SETTLEMENT_LOCATION.station_id else None,
                    "settlement_station_id": SETTLEMENT_LOCATION.station_id,
                    "settlement_lat": SETTLEMENT_LOCATION.lat,
                    "settlement_lon": SETTLEMENT_LOCATION.lon,
                }
            )
    return rows


def parse_bulletin_text(
    text: str,
    *,
    source_path: pathlib.Path,
    source_url: str | None = None,
    archive_member: str | None = None,
    station_id: str | None = None,
) -> pd.DataFrame:
    blocks: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current_lines: list[str] = []
    for line in text.splitlines():
        header_match = HEADER_RE.match(line.rstrip())
        if header_match:
            if current_header is not None:
                blocks.append((current_header, current_lines))
            if station_id is None or header_match.group("station") == station_id:
                current_header = line
            else:
                current_header = None
            current_lines = []
            continue
        if current_header is not None:
            current_lines.append(line)
    if current_header is not None:
        blocks.append((current_header, current_lines))

    rows: list[dict[str, object]] = []
    for header_line, body_lines in blocks:
        rows.extend(
            parse_station_block(
                header_line=header_line,
                body_lines=body_lines,
                source_path=source_path,
                source_url=source_url,
                archive_member=archive_member,
            )
        )
    return pd.DataFrame.from_records(rows)


def parse_bulletin_file(path: pathlib.Path, *, station_id: str | None = None) -> pd.DataFrame:
    return parse_bulletin_text(path.read_text(), source_path=path, station_id=station_id)


def merge_station_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    materialized = [frame for frame in frames if not frame.empty]
    if not materialized:
        return pd.DataFrame()
    merged = pd.concat(materialized, ignore_index=True)
    bulletin_priority = {"standard": 0, "full": 1, "extended": 2}
    merged["_priority"] = merged["bulletin_type"].map(lambda value: bulletin_priority.get(str(value), 9))
    merged = merged.sort_values(
        by=["station_id", "init_time_utc", "forecast_hour", "raw_label", "_priority", "bulletin_source_path"]
    )
    merged = merged.drop_duplicates(
        subset=["station_id", "init_time_utc", "forecast_hour", "raw_label"],
        keep="last",
    )
    return merged.drop(columns="_priority").reset_index(drop=True)


def parquet_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.columns:
        if safe[column].dtype == object:
            safe[column] = safe[column].map(lambda value: None if pd.isna(value) else str(value))
    return safe


def write_parsed_output(df: pd.DataFrame, *, output_dir: pathlib.Path, station_id: str | None) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    token = station_id or "all"
    path = output_dir / f"lamp_parsed_station={token}.parquet"
    parquet_safe_df(df).to_parquet(path, index=False)
    return path


def main() -> int:
    args = parse_args()
    inputs = discover_ascii_inputs(args.inputs)
    frames = [parse_bulletin_file(path, station_id=args.station_id) for path in inputs]
    merged = merge_station_frames(frames)
    output_path = write_parsed_output(merged, output_dir=args.output_dir, station_id=args.station_id)
    print(f"[ok] rows={len(merged)} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
