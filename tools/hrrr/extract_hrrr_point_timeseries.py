#!/usr/bin/env python3
"""Extract HRRR point rows for KLGA or another target point into CSV."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import math
import sys
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

import eccodes

from fetch_hrrr_records import (
    IdxRecord,
    build_remote_paths,
    download_file,
    fetch_text,
    parse_idx,
)
from hrrr_fields import PRODUCT_FIELD_SPECS, FieldSpec


DEFAULT_KLGA_LAT = 40.7769
DEFAULT_KLGA_LON = -73.8740
DEFAULT_TIMEZONE = "America/New_York"


@dataclass
class FileResult:
    ok: bool
    row: dict[str, object] | None
    grib_url: str
    message: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="Start run date in YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument("--end-date", required=True, help="End run date in YYYY-MM-DD or YYYYMMDD.")
    parser.add_argument(
        "--product",
        required=True,
        choices=sorted(PRODUCT_FIELD_SPECS),
        help="HRRR product family to extract.",
    )
    parser.add_argument(
        "--cycle-hours",
        default="0-23",
        help="UTC cycle hours, comma-separated and/or ranges, e.g. 0-23 or 0,6,12,18.",
    )
    parser.add_argument(
        "--forecast-hours",
        required=True,
        help="Forecast file steps, comma-separated and/or ranges; subhourly uses 15-minute steps, e.g. 0-18 or 0-72.",
    )
    parser.add_argument(
        "--source",
        default="aws",
        choices=["aws", "google", "nomads"],
        help="Archive source to use.",
    )
    parser.add_argument("--station-id", default="KLGA", help="Point identifier to store in the CSV.")
    parser.add_argument("--lat", type=float, default=DEFAULT_KLGA_LAT, help="Target latitude.")
    parser.add_argument("--lon", type=float, default=DEFAULT_KLGA_LON, help="Target longitude.")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="Local timezone for local-day columns.")
    parser.add_argument("--output-path", required=True, help="Output CSV or CSV.GZ path.")
    parser.add_argument(
        "--download-dir",
        default="data/runtime/downloads",
        help="Directory for temporary or cached full-file GRIB downloads.",
    )
    parser.add_argument(
        "--keep-downloads",
        action="store_true",
        help="Keep downloaded GRIB files in --download-dir for reuse across reruns.",
    )
    parser.add_argument("--append", action="store_true", help="Append to an existing output file.")
    parser.add_argument("--max-workers", type=int, default=4, help="Parallel workers for remote fetches.")
    parser.add_argument("--limit-files", type=int, help="Optional cap for quick tests.")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Exit 0 even if some files are skipped or fail during backfill.",
    )
    return parser.parse_args()


def parse_date(value: str) -> dt.date:
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unsupported date format: {value}")


def parse_int_ranges(value: str) -> list[int]:
    values: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range: {token}")
            values.update(range(start, end + 1))
            continue
        values.add(int(token))
    if not values:
        raise ValueError("No integer values were provided.")
    return sorted(values)


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def open_csv(path: Path, append: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        mode = "at" if append else "wt"
        return gzip.open(path, mode, newline="")
    mode = "a" if append else "w"
    return path.open(mode, newline="")


def normalize_lon(lon: float) -> float:
    return lon if lon >= 0 else lon + 360.0


def signed_lon(lon: float) -> float:
    return lon - 360.0 if lon > 180.0 else lon


def record_matches_spec(record: IdxRecord, spec: FieldSpec) -> bool:
    if record.variable != spec.variable or record.level != spec.level:
        return False
    valid_desc = f" {record.valid_desc.lower()} "
    if spec.mode == "instant":
        return " ave " not in valid_desc and " acc " not in valid_desc
    if spec.mode == "accum":
        return " acc " in valid_desc
    return False


def select_records(records: Iterable[IdxRecord], specs: list[FieldSpec]) -> dict[str, IdxRecord]:
    selected: dict[str, IdxRecord] = {}
    for spec in specs:
        for record in records:
            if record_matches_spec(record, spec):
                selected[spec.column] = record
                break
    return selected


def parse_grib_datetimes(handle) -> tuple[dt.datetime, dt.datetime]:
    init_date = eccodes.codes_get(handle, "dataDate")
    init_time = eccodes.codes_get(handle, "dataTime")
    valid_date = eccodes.codes_get(handle, "validityDate")
    valid_time = eccodes.codes_get(handle, "validityTime")
    init_dt = dt.datetime.strptime(f"{init_date}{init_time:04d}", "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
    valid_dt = dt.datetime.strptime(f"{valid_date}{valid_time:04d}", "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
    return init_dt, valid_dt


def k_to_f(value: float | None) -> float | None:
    if value is None:
        return None
    return (value - 273.15) * 9.0 / 5.0 + 32.0


def ms_to_mph(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 2.2369362920544


def pa_to_hpa(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0


def wind_speed(u: float | None, v: float | None) -> float | None:
    if u is None or v is None:
        return None
    return math.hypot(u, v)


def wind_direction_from_uv(u: float | None, v: float | None) -> float | None:
    if u is None or v is None:
        return None
    return math.degrees(math.atan2(-u, -v)) % 360.0


def base_row(
    station_id: str,
    lat: float,
    lon: float,
    timezone_name: str,
    product: str,
    source: str,
    grib_url: str,
    forecast_hour: int,
) -> dict[str, object]:
    return {
        "station_id": station_id,
        "target_lat": lat,
        "target_lon": lon,
        "timezone": timezone_name,
        "product": product,
        "source": source,
        "grib_url": grib_url,
        "forecast_hour_file": forecast_hour,
    }


def enrich_row(row: dict[str, object], timezone_name: str) -> None:
    local_tz = ZoneInfo(timezone_name)
    init_dt = row.get("init_time_utc")
    valid_dt = row.get("valid_time_utc")
    if isinstance(init_dt, dt.datetime):
        init_local = init_dt.astimezone(local_tz)
        row["init_time_local"] = init_local.isoformat()
        row["init_date_local"] = init_local.date().isoformat()
        row["init_hour_local"] = init_local.hour
    if isinstance(valid_dt, dt.datetime):
        valid_local = valid_dt.astimezone(local_tz)
        row["valid_time_local"] = valid_local.isoformat()
        row["valid_date_local"] = valid_local.date().isoformat()
        row["valid_hour_local"] = valid_local.hour

    row["tmp_2m_f"] = k_to_f(row.get("tmp_2m_k"))
    row["dpt_2m_f"] = k_to_f(row.get("dpt_2m_k"))
    row["gust_surface_mph"] = ms_to_mph(row.get("gust_surface_ms"))
    row["surface_pressure_hpa"] = pa_to_hpa(row.get("surface_pressure_pa"))
    row["wind_speed_10m_ms"] = wind_speed(row.get("ugrd_10m_ms"), row.get("vgrd_10m_ms"))
    row["wind_speed_10m_mph"] = ms_to_mph(row.get("wind_speed_10m_ms"))
    row["wind_dir_10m_deg"] = wind_direction_from_uv(row.get("ugrd_10m_ms"), row.get("vgrd_10m_ms"))

    if isinstance(init_dt, dt.datetime) and isinstance(valid_dt, dt.datetime):
        row["lead_minutes"] = int((valid_dt - init_dt).total_seconds() // 60)


def serialize_row(row: dict[str, object]) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in row.items():
        if isinstance(value, dt.datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value
    return serialized


def local_download_path(download_dir: Path, source: str, run_date: dt.date, cycle_hour: int, product: str, forecast_hour: int) -> Path:
    token = "wrfsfcf" if product == "surface" else "wrfsubhf"
    filename = f"{source}_hrrr.{run_date:%Y%m%d}.t{cycle_hour:02d}z.{token}{forecast_hour:02d}.grib2"
    return download_dir / filename


def read_local_range(handle, start: int, end: int) -> bytes:
    handle.seek(start)
    return handle.read(end - start + 1)


def process_file(
    run_date: dt.date,
    cycle_hour: int,
    forecast_hour: int,
    product: str,
    source: str,
    station_id: str,
    lat: float,
    lon: float,
    timezone_name: str,
    download_dir: Path,
    keep_downloads: bool,
) -> FileResult:
    grib_url, idx_url = build_remote_paths(run_date.strftime("%Y%m%d"), cycle_hour, product, forecast_hour, source)
    specs = PRODUCT_FIELD_SPECS[product]
    local_path = local_download_path(download_dir, source, run_date, cycle_hour, product, forecast_hour)

    try:
        idx_text = fetch_text(idx_url)
    except urllib.error.HTTPError as exc:
        return FileResult(ok=False, row=None, grib_url=grib_url, message=f"HTTP {exc.code}")
    except Exception as exc:
        return FileResult(ok=False, row=None, grib_url=grib_url, message=str(exc))

    try:
        download_file(grib_url, local_path, overwrite=False)
    except Exception as exc:
        return FileResult(ok=False, row=None, grib_url=grib_url, message=str(exc))

    try:
        records = parse_idx(idx_text, local_path.stat().st_size)
        selected = select_records(records, specs)
        if not selected:
            return FileResult(ok=False, row=None, grib_url=grib_url, message="no requested fields in archive file")

        row = base_row(station_id, lat, lon, timezone_name, product, source, grib_url, forecast_hour)
        target_lon = normalize_lon(lon)

        with local_path.open("rb") as local_handle:
            for spec in specs:
                record = selected.get(spec.column)
                if record is None:
                    continue
                payload = read_local_range(local_handle, record.byte_start, record.byte_end)
                handle = eccodes.codes_new_from_message(payload)
                try:
                    nearest = eccodes.codes_grib_find_nearest(handle, lat, target_lon)[0]
                    row[spec.column] = float(nearest.value)
                    if "init_time_utc" not in row:
                        init_dt, valid_dt = parse_grib_datetimes(handle)
                        row["init_time_utc"] = init_dt
                        row["valid_time_utc"] = valid_dt
                        row["step_range"] = eccodes.codes_get(handle, "stepRange")
                        row["grid_lat"] = float(nearest.lat)
                        row["grid_lon"] = signed_lon(float(nearest.lon))
                        row["grid_distance_km"] = float(nearest.distance)
                finally:
                    eccodes.codes_release(handle)

        if "init_time_utc" not in row:
            return FileResult(ok=False, row=None, grib_url=grib_url, message="missing GRIB timing metadata")

        enrich_row(row, timezone_name)
        return FileResult(ok=True, row=serialize_row(row), grib_url=grib_url)
    except Exception as exc:
        return FileResult(ok=False, row=None, grib_url=grib_url, message=str(exc))
    finally:
        if local_path.exists() and not keep_downloads:
            local_path.unlink()


def iter_tasks(
    start_date: dt.date,
    end_date: dt.date,
    cycle_hours: list[int],
    forecast_hours: list[int],
    limit_files: int | None,
) -> Iterable[tuple[dt.date, int, int]]:
    emitted = 0
    for run_date in daterange(start_date, end_date):
        for cycle_hour in cycle_hours:
            for forecast_hour in forecast_hours:
                if limit_files is not None and emitted >= limit_files:
                    return
                emitted += 1
                yield (run_date, cycle_hour, forecast_hour)


def task_count(
    start_date: dt.date,
    end_date: dt.date,
    cycle_hours: list[int],
    forecast_hours: list[int],
    limit_files: int | None,
) -> int:
    days = (end_date - start_date).days + 1
    total = days * len(cycle_hours) * len(forecast_hours)
    if limit_files is not None:
        return min(total, limit_files)
    return total


def csv_columns(product: str) -> list[str]:
    field_columns = [spec.column for spec in PRODUCT_FIELD_SPECS[product]]
    return [
        "station_id",
        "target_lat",
        "target_lon",
        "grid_lat",
        "grid_lon",
        "grid_distance_km",
        "timezone",
        "product",
        "source",
        "grib_url",
        "forecast_hour_file",
        "step_range",
        "lead_minutes",
        "init_time_utc",
        "init_time_local",
        "init_date_local",
        "init_hour_local",
        "valid_time_utc",
        "valid_time_local",
        "valid_date_local",
        "valid_hour_local",
        *field_columns,
        "tmp_2m_f",
        "dpt_2m_f",
        "surface_pressure_hpa",
        "wind_speed_10m_ms",
        "wind_speed_10m_mph",
        "wind_dir_10m_deg",
        "gust_surface_mph",
    ]


def main() -> int:
    args = parse_args()
    try:
        start_date = parse_date(args.start_date)
        end_date = parse_date(args.end_date)
        cycle_hours = parse_int_ranges(args.cycle_hours)
        forecast_hours = parse_int_ranges(args.forecast_hours)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if end_date < start_date:
        print("End date must be on or after start date.", file=sys.stderr)
        return 2

    total_tasks = task_count(start_date, end_date, cycle_hours, forecast_hours, args.limit_files)
    if total_tasks == 0:
        print("No tasks to process.", file=sys.stderr)
        return 0

    output_path = Path(args.output_path)
    download_dir = Path(args.download_dir)
    needs_header = not args.append or not output_path.exists()
    columns = csv_columns(args.product)

    processed = 0
    wrote = 0
    failed = 0
    with open_csv(output_path, args.append) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if needs_header:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
            task_iter = iter(iter_tasks(start_date, end_date, cycle_hours, forecast_hours, args.limit_files))
            futures: dict[object, tuple[dt.date, int, int]] = {}

            for _ in range(max(1, args.max_workers)):
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                future = executor.submit(
                    process_file,
                    run_date=task[0],
                    cycle_hour=task[1],
                    forecast_hour=task[2],
                    product=args.product,
                    source=args.source,
                    station_id=args.station_id,
                    lat=args.lat,
                    lon=args.lon,
                    timezone_name=args.timezone,
                    download_dir=download_dir,
                    keep_downloads=args.keep_downloads,
                )
                futures[future] = task

            while futures:
                future = next(as_completed(futures))
                del futures[future]
                processed += 1
                result = future.result()
                if result.ok and result.row is not None:
                    writer.writerow(result.row)
                    wrote += 1
                else:
                    failed += 1
                    print(f"[skip] {result.grib_url} -> {result.message}", file=sys.stderr)

                try:
                    task = next(task_iter)
                    next_future = executor.submit(
                        process_file,
                        run_date=task[0],
                        cycle_hour=task[1],
                        forecast_hour=task[2],
                        product=args.product,
                        source=args.source,
                        station_id=args.station_id,
                        lat=args.lat,
                        lon=args.lon,
                        timezone_name=args.timezone,
                        download_dir=download_dir,
                        keep_downloads=args.keep_downloads,
                    )
                    futures[next_future] = task
                except StopIteration:
                    pass

                if processed % 25 == 0 or processed == total_tasks:
                    print(
                        f"[progress] processed={processed} wrote={wrote} failed={failed} total={total_tasks}",
                        file=sys.stderr,
                    )

    print(f"[done] wrote={wrote} failed={failed} path={output_path}")
    if failed and not args.allow_partial:
        print(
            "Backfill completed with skipped or failed files. "
            "Re-run with --allow-partial only if you intentionally want incomplete output.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
