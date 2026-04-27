#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import math
import numpy as np
import pathlib
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd
from pyproj import Transformer

try:
    import rasterio
    from rasterio.errors import RasterioIOError
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "sample_cog_point.py requires rasterio. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional import path
    pa = None
    pq = None

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.location_context import (
    LOCAL_NEIGHBORHOOD_SIZES,
    REGIONAL_CROP_BOUNDS,
    SETTLEMENT_LOCATION,
    crop_context_metrics,
    crop_metadata,
    longitude_360_to_180,
    local_context_metrics,
)

FILE_RE = re.compile(
    r"^(?P<version>blendv[\d.]+)_(?P<domain>[^_]+)_(?P<variable>.+?)_"
    r"(?P<init>\d{4}-\d{2}-\d{2}T\d{2}:\d{2})_"
    r"(?P<valid>\d{4}-\d{2}-\d{2}T\d{2}:\d{2})\.tif$"
)
NY_TZ = ZoneInfo("America/New_York")
DEFAULT_INPUT = pathlib.Path("data/nbm/cog")
DEFAULT_OUTPUT = pathlib.Path("data/features/klga_nbm_features.csv")
CHUNK_ROWS = 5000
SAMPLE_PROGRESS_EVERY = 250
SOURCE_MODEL = "NBM"
SOURCE_PRODUCT = "cog"
REQUIRED_WIDE_IDENTITY_COLUMNS = {
    "source_model",
    "source_product",
    "source_version",
    "fallback_used_any",
    "station_id",
    "init_time_utc",
    "valid_time_utc",
    "init_time_local",
    "valid_time_local",
    "init_date_local",
    "valid_date_local",
    "forecast_hour",
    "settlement_lat",
    "settlement_lon",
    "crop_top_lat",
    "crop_bottom_lat",
    "crop_left_lon",
    "crop_right_lon",
    "nearest_grid_lat",
    "nearest_grid_lon",
}


@dataclass(frozen=True)
class ParsedFile:
    path: pathlib.Path
    version: str
    domain: str
    variable: str
    init_time_utc: dt.datetime
    valid_time_utc: dt.datetime
    lead_hours: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample a point from downloaded NOAA NBM COG files and build a feature table."
    )
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        default=DEFAULT_INPUT,
        help="Root directory containing downloaded NBM COG files",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Wide output path",
    )
    parser.add_argument(
        "--long-output",
        type=pathlib.Path,
        help="Optional long-table output path",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "parquet"],
        default="csv",
        help="Output format for wide output and optional long output",
    )
    parser.add_argument(
        "--station-id",
        default=SETTLEMENT_LOCATION.station_id,
        help="Settlement station identifier to stamp into the output",
    )
    parser.add_argument("--lat", type=float, default=SETTLEMENT_LOCATION.lat, help="Settlement latitude in decimal degrees")
    parser.add_argument("--lon", type=float, default=SETTLEMENT_LOCATION.lon, help="Settlement longitude in decimal degrees")
    parser.add_argument(
        "--variables",
        nargs="+",
        help="Optional variable filter, e.g. temp dewpoint windspd maxt",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent file sampling workers",
    )
    return parser.parse_args()


def validate_output_path(path: pathlib.Path, output_format: str, label: str) -> None:
    if output_format == "csv" and path.suffix and path.suffix.lower() != ".csv":
        raise SystemExit(f"{label} must end with .csv when --output-format=csv: {path}")
    if output_format == "parquet" and path.suffix and path.suffix.lower() != ".parquet":
        raise SystemExit(f"{label} must end with .parquet when --output-format=parquet: {path}")


def validate_settlement_target(args: argparse.Namespace) -> None:
    if args.station_id != SETTLEMENT_LOCATION.station_id:
        raise SystemExit(
            f"--station-id must remain the settlement station {SETTLEMENT_LOCATION.station_id}: {args.station_id}"
        )
    if not math.isclose(args.lat, SETTLEMENT_LOCATION.lat, abs_tol=1e-6):
        raise SystemExit(f"--lat must remain the settlement latitude {SETTLEMENT_LOCATION.lat}: {args.lat}")
    if not math.isclose(args.lon, SETTLEMENT_LOCATION.lon, abs_tol=1e-6):
        raise SystemExit(f"--lon must remain the settlement longitude {SETTLEMENT_LOCATION.lon}: {args.lon}")


def parse_file_metadata(path: pathlib.Path) -> ParsedFile | None:
    match = FILE_RE.match(path.name)
    if not match:
        return None
    info = match.groupdict()
    init_time = dt.datetime.strptime(info["init"], "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)
    valid_time = dt.datetime.strptime(info["valid"], "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc)
    return ParsedFile(
        path=path,
        version=info["version"],
        domain=info["domain"],
        variable=info["variable"],
        init_time_utc=init_time,
        valid_time_utc=valid_time,
        lead_hours=(valid_time - init_time).total_seconds() / 3600.0,
    )


def _dataset_transformers(dataset: rasterio.DatasetReader):
    to_dataset = None
    to_wgs84 = None
    if dataset.crs and str(dataset.crs).upper() != "EPSG:4326":
        to_dataset = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
    return to_dataset, to_wgs84


def _transform_lon_lat(transformer: Transformer | None, lon: float, lat: float) -> tuple[float, float]:
    if transformer is None:
        return lon, lat
    return transformer.transform(lon, lat)


def crop_window_for_dataset(dataset: rasterio.DatasetReader) -> rasterio.windows.Window:
    to_dataset, _ = _dataset_transformers(dataset)
    left = longitude_360_to_180(REGIONAL_CROP_BOUNDS.left)
    right = longitude_360_to_180(REGIONAL_CROP_BOUNDS.right)
    corners = [
        _transform_lon_lat(to_dataset, left, REGIONAL_CROP_BOUNDS.top),
        _transform_lon_lat(to_dataset, right, REGIONAL_CROP_BOUNDS.top),
        _transform_lon_lat(to_dataset, left, REGIONAL_CROP_BOUNDS.bottom),
        _transform_lon_lat(to_dataset, right, REGIONAL_CROP_BOUNDS.bottom),
    ]
    rows: list[int] = []
    cols: list[int] = []
    for x, y in corners:
        row, col = dataset.index(x, y)
        rows.append(row)
        cols.append(col)
    row_start = max(0, min(rows))
    row_end = min(dataset.height, max(rows) + 1)
    col_start = max(0, min(cols))
    col_end = min(dataset.width, max(cols) + 1)
    return rasterio.windows.Window(
        col_off=col_start,
        row_off=row_start,
        width=max(1, col_end - col_start),
        height=max(1, row_end - row_start),
    )


def validate_required_wide_columns(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_WIDE_IDENTITY_COLUMNS - set(df.columns))
    if missing:
        raise SystemExit(f"Wide output is missing required canonical identity columns: {', '.join(missing)}")


def discover_files(input_dir: pathlib.Path, variables: set[str] | None) -> list[ParsedFile]:
    parsed: list[ParsedFile] = []
    skipped = 0
    for path in sorted(input_dir.rglob("*.tif")):
        parsed_file = parse_file_metadata(path)
        if parsed_file is None:
            skipped += 1
            continue
        if variables and parsed_file.variable not in variables:
            continue
        parsed.append(parsed_file)
    if skipped:
        print(f"skipped_unmatched_files={skipped}", file=sys.stderr)
    if not parsed:
        raise SystemExit(f"No matching .tif files found under {input_dir}")
    return parsed


def sample_file(parsed_file: ParsedFile, station_id: str, lat: float, lon: float) -> dict[str, object]:
    try:
        with rasterio.open(parsed_file.path) as dataset:
            to_dataset, to_wgs84 = _dataset_transformers(dataset)
            x, y = _transform_lon_lat(to_dataset, lon, lat)
            row, col = dataset.index(x, y)
            if row < 0 or col < 0 or row >= dataset.height or col >= dataset.width:
                raise ValueError(f"Sample point outside raster bounds: row={row} col={col}")
            radius = max(LOCAL_NEIGHBORHOOD_SIZES) // 2
            row_start = max(0, row - radius)
            row_end = min(dataset.height, row + radius + 1)
            col_start = max(0, col - radius)
            col_end = min(dataset.width, col + radius + 1)
            window = rasterio.windows.Window(
                col_off=col_start,
                row_off=row_start,
                width=col_end - col_start,
                height=row_end - row_start,
            )
            values = dataset.read(1, window=window, boundless=False).astype("float64")
            nodata = dataset.nodata
            if nodata is not None:
                values[np.isclose(values, float(nodata))] = np.nan
            crop_window = crop_window_for_dataset(dataset)
            crop_values = dataset.read(1, window=crop_window, boundless=False).astype("float64")
            if nodata is not None:
                crop_values[np.isclose(crop_values, float(nodata))] = np.nan
            local_metrics = local_context_metrics(
                values,
                row=row - row_start,
                col=col - col_start,
                north_is_first=dataset.transform.e < 0,
                window_sizes=LOCAL_NEIGHBORHOOD_SIZES,
            )
            crop_metrics = crop_context_metrics(crop_values)
            grid_x, grid_y = dataset.xy(row, col)
            nearest_grid_lon, nearest_grid_lat = _transform_lon_lat(to_wgs84, float(grid_x), float(grid_y))
            init_local = parsed_file.init_time_utc.astimezone(NY_TZ)
            valid_local = parsed_file.valid_time_utc.astimezone(NY_TZ)
            return {
                "source_model": SOURCE_MODEL,
                "source_product": SOURCE_PRODUCT,
                "source_version": parsed_file.version,
                "fallback_used_any": False,
                "station_id": station_id,
                "station_lat": lat,
                "station_lon": lon,
                **crop_metadata(),
                "settlement_station_id": SETTLEMENT_LOCATION.station_id,
                "settlement_lat": SETTLEMENT_LOCATION.lat,
                "settlement_lon": SETTLEMENT_LOCATION.lon,
                "init_time_utc": parsed_file.init_time_utc.isoformat(),
                "valid_time_utc": parsed_file.valid_time_utc.isoformat(),
                "init_time_local": init_local.isoformat(),
                "valid_time_local": valid_local.isoformat(),
                "init_date_local": init_local.date().isoformat(),
                "valid_date_local": valid_local.date().isoformat(),
                "forecast_hour": parsed_file.lead_hours,
                "lead_hours": parsed_file.lead_hours,
                "version": parsed_file.version,
                "domain": parsed_file.domain,
                "variable": parsed_file.variable,
                "sample_value": local_metrics["sample_value"],
                "crop_mean": crop_metrics["crop_mean"],
                "crop_min": crop_metrics["crop_min"],
                "crop_max": crop_metrics["crop_max"],
                "crop_std": crop_metrics["crop_std"],
                "nb3_mean": local_metrics["nb3_mean"],
                "nb3_min": local_metrics["nb3_min"],
                "nb3_max": local_metrics["nb3_max"],
                "nb3_std": local_metrics["nb3_std"],
                "nb3_gradient_west_east": local_metrics["nb3_gradient_west_east"],
                "nb3_gradient_south_north": local_metrics["nb3_gradient_south_north"],
                "nb7_mean": local_metrics["nb7_mean"],
                "nb7_min": local_metrics["nb7_min"],
                "nb7_max": local_metrics["nb7_max"],
                "nb7_std": local_metrics["nb7_std"],
                "nb7_gradient_west_east": local_metrics["nb7_gradient_west_east"],
                "nb7_gradient_south_north": local_metrics["nb7_gradient_south_north"],
                "grid_row": row,
                "grid_col": col,
                "nearest_grid_lat": float(nearest_grid_lat),
                "nearest_grid_lon": float(nearest_grid_lon),
                "grid_x": float(grid_x),
                "grid_y": float(grid_y),
                "source_path": str(parsed_file.path),
            }
    except RasterioIOError as exc:
        raise SystemExit(f"Failed to read raster file {parsed_file.path}: {exc}") from exc


def flush_chunk(records: list[dict[str, object]], chunk_dir: pathlib.Path, chunk_index: int) -> pathlib.Path:
    chunk_path = chunk_dir / f"chunk_{chunk_index:05d}.parquet"
    pd.DataFrame.from_records(records).to_parquet(chunk_path, index=False)
    return chunk_path


def collect_chunk_paths(args: argparse.Namespace, parsed_files: list[ParsedFile]) -> list[pathlib.Path]:
    start_time = time.perf_counter()
    chunk_paths: list[pathlib.Path] = []
    buffer: list[dict[str, object]] = []
    chunk_index = 0
    long_writer = None
    long_schema = None

    with tempfile.TemporaryDirectory(prefix="nbm-sample-") as tempdir:
        chunk_dir = pathlib.Path(tempdir)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(sample_file, parsed_file, args.station_id, args.lat, args.lon)
                for parsed_file in parsed_files
            ]
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                buffer.append(future.result())
                if idx % SAMPLE_PROGRESS_EVERY == 0:
                    elapsed = max(0.001, time.perf_counter() - start_time)
                    print(
                        f"sampling_progress processed={idx}/{len(parsed_files)} rate_fps={idx / elapsed:.2f}",
                        file=sys.stderr,
                    )
                if len(buffer) >= CHUNK_ROWS:
                    batch_df = pd.DataFrame.from_records(buffer)
                    chunk_path = chunk_dir / f"chunk_{chunk_index:05d}.parquet"
                    batch_df.to_parquet(chunk_path, index=False)
                    chunk_paths.append(chunk_path)
                    if args.long_output and args.output_format == "parquet":
                        if pq is None or pa is None:
                            raise SystemExit(
                                "Parquet output requires pyarrow. Install dependencies with `pip install -r requirements.txt`."
                            )
                        if long_writer is None:
                            args.long_output.parent.mkdir(parents=True, exist_ok=True)
                            table = pa.Table.from_pandas(batch_df, preserve_index=False)
                            long_schema = table.schema
                            long_writer = pq.ParquetWriter(args.long_output, long_schema)
                            long_writer.write_table(table)
                        else:
                            long_writer.write_table(pa.Table.from_pandas(batch_df, schema=long_schema, preserve_index=False))
                    chunk_index += 1
                    buffer = []
        if buffer:
            batch_df = pd.DataFrame.from_records(buffer)
            chunk_path = chunk_dir / f"chunk_{chunk_index:05d}.parquet"
            batch_df.to_parquet(chunk_path, index=False)
            chunk_paths.append(chunk_path)
            if args.long_output and args.output_format == "parquet":
                if pq is None or pa is None:
                    raise SystemExit(
                        "Parquet output requires pyarrow. Install dependencies with `pip install -r requirements.txt`."
                    )
                if long_writer is None:
                    args.long_output.parent.mkdir(parents=True, exist_ok=True)
                    table = pa.Table.from_pandas(batch_df, preserve_index=False)
                    long_schema = table.schema
                    long_writer = pq.ParquetWriter(args.long_output, long_schema)
                    long_writer.write_table(table)
                else:
                    long_writer.write_table(pa.Table.from_pandas(batch_df, schema=long_schema, preserve_index=False))

        if long_writer is not None:
            long_writer.close()

        persisted_paths: list[pathlib.Path] = []
        persisted_dir = args.output.parent / ".tmp_nbm_chunks"
        if persisted_dir.exists():
            for old_chunk in persisted_dir.glob("*.parquet"):
                old_chunk.unlink()
        persisted_dir.mkdir(parents=True, exist_ok=True)
        for chunk_path in chunk_paths:
            target = persisted_dir / chunk_path.name
            chunk_path.replace(target)
            persisted_paths.append(target)
        return persisted_paths


def load_long_dataframe(chunk_paths: list[pathlib.Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in chunk_paths]
    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.sort_values(
        ["init_time_utc", "valid_time_utc", "variable", "source_path"]
    ).reset_index(drop=True)
    return long_df


def write_dataframe(df: pd.DataFrame, path: pathlib.Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def write_outputs(args: argparse.Namespace, chunk_paths: list[pathlib.Path]) -> None:
    long_df = load_long_dataframe(chunk_paths)
    index_cols = [
        "source_model",
        "source_product",
        "source_version",
        "fallback_used_any",
        "station_id",
        "station_lat",
        "station_lon",
        "settlement_station_id",
        "settlement_lat",
        "settlement_lon",
        "crop_top_lat",
        "crop_bottom_lat",
        "crop_left_lon",
        "crop_right_lon",
        "crop_top",
        "crop_bottom",
        "crop_left",
        "crop_right",
        "init_time_utc",
        "valid_time_utc",
        "init_time_local",
        "valid_time_local",
        "init_date_local",
        "valid_date_local",
        "forecast_hour",
        "lead_hours",
        "version",
        "domain",
        "grid_row",
        "grid_col",
        "nearest_grid_lat",
        "nearest_grid_lon",
        "grid_x",
        "grid_y",
    ]
    value_cols = [
        "sample_value",
        "crop_mean",
        "crop_min",
        "crop_max",
        "crop_std",
        "nb3_mean",
        "nb3_min",
        "nb3_max",
        "nb3_std",
        "nb3_gradient_west_east",
        "nb3_gradient_south_north",
        "nb7_mean",
        "nb7_min",
        "nb7_max",
        "nb7_std",
        "nb7_gradient_west_east",
        "nb7_gradient_south_north",
    ]
    wide_df = (
        long_df.pivot_table(
            index=index_cols,
            columns="variable",
            values=value_cols,
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["init_time_utc", "valid_time_utc"])
    )
    flattened_columns: list[str] = []
    for column in wide_df.columns:
        if not isinstance(column, tuple):
            flattened_columns.append(str(column))
            continue
        metric_name, variable_name = column
        if not variable_name:
            flattened_columns.append(str(metric_name))
            continue
        if metric_name == "sample_value":
            flattened_columns.append(str(variable_name))
            continue
        flattened_columns.append(f"{variable_name}_{metric_name}")
    wide_df.columns = flattened_columns
    validate_required_wide_columns(wide_df)

    write_dataframe(wide_df, args.output, args.output_format)
    print(f"saved_wide={args.output}")

    if args.long_output:
        if args.output_format == "csv":
            write_dataframe(long_df, args.long_output, args.output_format)
        print(f"saved_long={args.long_output}")

    chunk_dir = args.output.parent / ".tmp_nbm_chunks"
    if chunk_dir.exists():
        for chunk_path in chunk_dir.glob("*.parquet"):
            chunk_path.unlink()
        chunk_dir.rmdir()


def main() -> int:
    args = parse_args()
    validate_output_path(args.output, args.output_format, "--output")
    if args.long_output:
        validate_output_path(args.long_output, args.output_format, "--long-output")
    validate_settlement_target(args)
    variables = set(args.variables or [])
    parsed_files = discover_files(args.input_dir, variables or None)
    chunk_paths = collect_chunk_paths(args, parsed_files)
    write_outputs(args, chunk_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
