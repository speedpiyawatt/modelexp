from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SettlementLocation:
    station_id: str
    lat: float
    lon: float


@dataclass(frozen=True)
class CropBounds:
    top: float
    bottom: float
    left: float
    right: float


SETTLEMENT_LOCATION = SettlementLocation(station_id="KLGA", lat=40.7769, lon=-73.8740)
REGIONAL_CROP_BOUNDS = CropBounds(top=43.5, bottom=39.0, left=282.5, right=289.5)
LOCAL_NEIGHBORHOOD_SIZES = (3, 7)


def settlement_longitude_360() -> float:
    return SETTLEMENT_LOCATION.lon + 360.0 if SETTLEMENT_LOCATION.lon < 0 else SETTLEMENT_LOCATION.lon


def settlement_metadata() -> dict[str, object]:
    return {
        "station_id": SETTLEMENT_LOCATION.station_id,
        "settlement_station_id": SETTLEMENT_LOCATION.station_id,
        "settlement_lat": SETTLEMENT_LOCATION.lat,
        "settlement_lon": SETTLEMENT_LOCATION.lon,
        "settlement_lon_360": settlement_longitude_360(),
    }


def crop_metadata(bounds: CropBounds | None = None) -> dict[str, object]:
    bounds = bounds or REGIONAL_CROP_BOUNDS
    return {
        "crop_top_lat": bounds.top,
        "crop_bottom_lat": bounds.bottom,
        "crop_left_lon": bounds.left,
        "crop_right_lon": bounds.right,
        "crop_top": bounds.top,
        "crop_bottom": bounds.bottom,
        "crop_left": bounds.left,
        "crop_right": bounds.right,
    }


def longitude_360_to_180(lon: float) -> float:
    if lon > 180:
        return lon - 360.0
    return lon


def normalize_longitude_for_grid(lon_values: np.ndarray, station_lon: float) -> float:
    if np.nanmax(lon_values) > 180 and station_lon < 0:
        return station_lon + 360.0
    return station_lon


def find_nearest_grid_cell(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    *,
    station_lat: float,
    station_lon: float,
) -> dict[str, object]:
    station_lon_normalized = normalize_longitude_for_grid(lon_grid, station_lon)
    distances = (lat_grid - station_lat) ** 2 + (lon_grid - station_lon_normalized) ** 2
    row, col = np.unravel_index(np.nanargmin(distances), distances.shape)
    return {
        "grid_row": int(row),
        "grid_col": int(col),
        "grid_lat": float(lat_grid[row, col]),
        "grid_lon": float(lon_grid[row, col]),
    }


def infer_north_is_first(lat_values: np.ndarray) -> bool:
    if lat_values.ndim == 1:
        return float(np.nanmean(lat_values[:1])) >= float(np.nanmean(lat_values[-1:]))
    return float(np.nanmean(lat_values[0])) >= float(np.nanmean(lat_values[-1]))


def infer_west_is_first(lon_values: np.ndarray) -> bool:
    if lon_values.ndim == 1:
        return float(np.nanmean(lon_values[:1])) <= float(np.nanmean(lon_values[-1:]))
    return float(np.nanmean(lon_values[:, 0])) <= float(np.nanmean(lon_values[:, -1]))


def _finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def _mean_or_none(values: np.ndarray) -> float | None:
    finite = _finite_values(values)
    if finite.size == 0:
        return None
    return float(np.nanmean(finite))


def summarize_values(values: np.ndarray) -> dict[str, float | None]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {
        "mean": float(np.nanmean(finite)),
        "min": float(np.nanmin(finite)),
        "max": float(np.nanmax(finite)),
        "std": float(np.nanstd(finite)),
    }


def crop_context_metrics(values: np.ndarray) -> dict[str, float | None]:
    stats = summarize_values(values)
    return {
        "crop_mean": stats["mean"],
        "crop_min": stats["min"],
        "crop_max": stats["max"],
        "crop_std": stats["std"],
    }


def neighborhood_metrics(
    values: np.ndarray,
    *,
    row: int,
    col: int,
    window_size: int,
    north_is_first: bool,
) -> dict[str, float | None]:
    radius = window_size // 2
    row_start = max(0, row - radius)
    row_end = min(values.shape[0], row + radius + 1)
    col_start = max(0, col - radius)
    col_end = min(values.shape[1], col + radius + 1)
    neighborhood = values[row_start:row_end, col_start:col_end]
    stats = summarize_values(neighborhood)

    west_mean = _mean_or_none(neighborhood[:, 0])
    east_mean = _mean_or_none(neighborhood[:, -1])
    top_mean = _mean_or_none(neighborhood[0, :])
    bottom_mean = _mean_or_none(neighborhood[-1, :])
    north_mean = top_mean if north_is_first else bottom_mean
    south_mean = bottom_mean if north_is_first else top_mean

    gradient_west_east = None
    if west_mean is not None and east_mean is not None:
        gradient_west_east = east_mean - west_mean

    gradient_south_north = None
    if north_mean is not None and south_mean is not None:
        gradient_south_north = north_mean - south_mean

    prefix = f"nb{window_size}"
    return {
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_min": stats["min"],
        f"{prefix}_max": stats["max"],
        f"{prefix}_std": stats["std"],
        f"{prefix}_gradient_west_east": gradient_west_east,
        f"{prefix}_gradient_south_north": gradient_south_north,
    }


def local_context_metrics(
    values: np.ndarray,
    *,
    row: int,
    col: int,
    north_is_first: bool,
    window_sizes: tuple[int, ...] = LOCAL_NEIGHBORHOOD_SIZES,
) -> dict[str, float | None]:
    sample_value = values[row, col]
    result: dict[str, float | None] = {
        "sample_value": None if not np.isfinite(sample_value) else float(sample_value),
    }
    for window_size in window_sizes:
        result.update(
            neighborhood_metrics(
                values,
                row=row,
                col=col,
                window_size=window_size,
                north_is_first=north_is_first,
            )
        )
    return result
