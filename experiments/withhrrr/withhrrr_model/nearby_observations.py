from __future__ import annotations

import datetime as dt
import re
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not token:
        raise ValueError("nearby station id produced an empty feature prefix")
    return token


def _numeric_or_none(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return None if pd.isna(numeric) else float(numeric)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def cutoff_timestamp(target_date_local: str, cutoff_local_time: str) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(dt.date.fromisoformat(target_date_local), dt.time.fromisoformat(cutoff_local_time), tzinfo=NY_TZ))


def _window_stats(history: pd.DataFrame, *, cutoff_ts: pd.Timestamp, hours: int, prefix: str) -> dict[str, float | None]:
    recent = history.loc[(history["valid_time_local"] > cutoff_ts - pd.Timedelta(hours=hours)) & (history["valid_time_local"] <= cutoff_ts)]
    compare = history.loc[history["valid_time_local"] <= cutoff_ts - pd.Timedelta(hours=hours)]
    compare_row = compare.iloc[-1] if not compare.empty else None
    last_row = history.iloc[-1] if not history.empty else None
    result: dict[str, float | None] = {f"{prefix}_temp_change_{hours}h_f": None}
    if last_row is not None and compare_row is not None:
        last_temp = _numeric_or_none(last_row.get("temp_f"))
        compare_temp = _numeric_or_none(compare_row.get("temp_f"))
        if last_temp is not None and compare_temp is not None:
            result[f"{prefix}_temp_change_{hours}h_f"] = float(last_temp - compare_temp)
    if hours == 3 and last_row is not None and compare_row is not None:
        last_dewpoint = _numeric_or_none(last_row.get("dewpoint_f"))
        compare_dewpoint = _numeric_or_none(compare_row.get("dewpoint_f"))
        if last_dewpoint is not None and compare_dewpoint is not None:
            result[f"{prefix}_dewpoint_change_3h_f"] = float(last_dewpoint - compare_dewpoint)
        last_pressure = _numeric_or_none(last_row.get("pressure_in"))
        compare_pressure = _numeric_or_none(compare_row.get("pressure_in"))
        if last_pressure is not None and compare_pressure is not None:
            result[f"{prefix}_pressure_change_3h"] = float(last_pressure - compare_pressure)
        result[f"{prefix}_wind_speed_mean_3h"] = _numeric_or_none(recent["wind_speed_mph"].mean()) if not recent.empty and "wind_speed_mph" in recent.columns else None
    if hours == 6:
        result[f"{prefix}_wind_gust_max_6h"] = _numeric_or_none(recent["wind_gust_mph"].max()) if not recent.empty and "wind_gust_mph" in recent.columns else None
        result[f"{prefix}_visibility_min_6h"] = _numeric_or_none(recent["visibility"].min()) if not recent.empty and "visibility" in recent.columns else None
        precip = _numeric_series(recent, "precip_hrly_in").fillna(0.0) if not recent.empty else pd.Series(dtype="float64")
        result[f"{prefix}_precip_total_6h"] = float(precip.sum()) if not precip.empty else None
    return result


def empty_nearby_features(target_dates: pd.Series, *, station_id: str = "KJRB", prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or f"nearby_{_safe_token(station_id)}"
    return pd.DataFrame(
        {
            "target_date_local": target_dates.astype(str),
            f"meta_{prefix}_obs_available": False,
            f"meta_{prefix}_last_obs_time_local": None,
        }
    )


def build_nearby_station_features(
    target_dates: pd.Series,
    obs_df: pd.DataFrame,
    *,
    station_id: str = "KJRB",
    cutoff_local_time: str = "00:05",
    max_obs_age_hours: float = 12.0,
    prefix: str | None = None,
) -> pd.DataFrame:
    prefix = prefix or f"nearby_{_safe_token(station_id)}"
    targets = pd.Series(target_dates, copy=True).astype(str).drop_duplicates().sort_values().reset_index(drop=True)
    if obs_df.empty:
        return empty_nearby_features(targets, station_id=station_id, prefix=prefix)

    obs = obs_df.copy()
    if "station_id" in obs.columns:
        obs = obs.loc[obs["station_id"].astype(str).str.upper() == station_id.upper()].copy()
    if obs.empty:
        return empty_nearby_features(targets, station_id=station_id, prefix=prefix)
    obs["valid_time_local"] = pd.to_datetime(obs["valid_time_local"], utc=True).dt.tz_convert(NY_TZ)
    obs = obs.sort_values("valid_time_local").reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for target_date_local in targets:
        cutoff_ts = cutoff_timestamp(target_date_local, cutoff_local_time)
        station_obs = obs.loc[
            (obs["valid_time_local"] > cutoff_ts - pd.Timedelta(hours=max_obs_age_hours))
            & (obs["valid_time_local"] <= cutoff_ts)
        ].copy()
        row: dict[str, object] = {
            "target_date_local": target_date_local,
            f"meta_{prefix}_obs_available": not station_obs.empty,
            f"meta_{prefix}_last_obs_time_local": None,
            f"{prefix}_last_temp_f": None,
            f"{prefix}_last_dewpoint_f": None,
            f"{prefix}_last_rh_pct": None,
            f"{prefix}_last_pressure_in": None,
            f"{prefix}_last_wind_speed_mph": None,
            f"{prefix}_last_wind_dir_deg": None,
            f"{prefix}_last_wind_gust_mph": None,
            f"{prefix}_last_visibility": None,
            f"{prefix}_last_precip_hrly_in": None,
            f"{prefix}_temp_change_1h_f": None,
            f"{prefix}_temp_change_3h_f": None,
            f"{prefix}_temp_change_6h_f": None,
            f"{prefix}_dewpoint_change_3h_f": None,
            f"{prefix}_pressure_change_3h": None,
            f"{prefix}_wind_speed_mean_3h": None,
            f"{prefix}_wind_gust_max_6h": None,
            f"{prefix}_visibility_min_6h": None,
            f"{prefix}_precip_total_6h": None,
        }
        if not station_obs.empty:
            last_obs = station_obs.iloc[-1]
            row.update(
                {
                    f"meta_{prefix}_last_obs_time_local": last_obs["valid_time_local"].isoformat(),
                    f"{prefix}_last_temp_f": _numeric_or_none(last_obs.get("temp_f")),
                    f"{prefix}_last_dewpoint_f": _numeric_or_none(last_obs.get("dewpoint_f")),
                    f"{prefix}_last_rh_pct": _numeric_or_none(last_obs.get("rh_pct")),
                    f"{prefix}_last_pressure_in": _numeric_or_none(last_obs.get("pressure_in")),
                    f"{prefix}_last_wind_speed_mph": _numeric_or_none(last_obs.get("wind_speed_mph")),
                    f"{prefix}_last_wind_dir_deg": _numeric_or_none(last_obs.get("wind_dir_deg")),
                    f"{prefix}_last_wind_gust_mph": _numeric_or_none(last_obs.get("wind_gust_mph")),
                    f"{prefix}_last_visibility": _numeric_or_none(last_obs.get("visibility")),
                    f"{prefix}_last_precip_hrly_in": _numeric_or_none(last_obs.get("precip_hrly_in")),
                }
            )
            for hours in (1, 3, 6):
                row.update(_window_stats(station_obs, cutoff_ts=cutoff_ts, hours=hours, prefix=prefix))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def add_nearby_station_derived_features(df: pd.DataFrame, *, station_id: str = "KJRB", prefix: str | None = None) -> pd.DataFrame:
    prefix = prefix or f"nearby_{_safe_token(station_id)}"
    out = df.copy()
    nearby_temp = _numeric_series(out, f"{prefix}_last_temp_f")
    klga_temp = _numeric_series(out, "wu_last_temp_f")
    out[f"{prefix}_minus_klga_last_temp_f"] = nearby_temp - klga_temp
    out[f"klga_minus_{prefix}_last_temp_f"] = klga_temp - nearby_temp
    for source_column, source_name in (
        ("nbm_tmax_open_f", "nbm"),
        ("nbm_native_tmax_2m_day_max_f", "native"),
        ("lamp_tmax_open_f", "lamp"),
        ("hrrr_tmax_open_f", "hrrr"),
    ):
        out[f"{prefix}_last_temp_minus_{source_name}_tmax_f"] = nearby_temp - _numeric_series(out, source_column)
    if f"{prefix}_temp_change_3h_f" in out.columns and "wu_temp_change_3h_f" in out.columns:
        out[f"{prefix}_minus_klga_temp_change_3h_f"] = _numeric_series(out, f"{prefix}_temp_change_3h_f") - _numeric_series(out, "wu_temp_change_3h_f")
    return out


def add_nearby_aggregate_features(df: pd.DataFrame, *, station_ids: list[str]) -> pd.DataFrame:
    out = df.copy()
    prefixes = [f"nearby_{_safe_token(station_id)}" for station_id in station_ids]
    temp_columns = [f"{prefix}_last_temp_f" for prefix in prefixes if f"{prefix}_last_temp_f" in out.columns]
    if not temp_columns:
        return out
    temps = out.loc[:, temp_columns].apply(pd.to_numeric, errors="coerce")
    out["nearby_station_available_count"] = temps.notna().sum(axis=1)
    out["nearby_station_last_temp_mean_f"] = temps.mean(axis=1)
    out["nearby_station_last_temp_median_f"] = temps.median(axis=1)
    out["nearby_station_last_temp_min_f"] = temps.min(axis=1)
    out["nearby_station_last_temp_max_f"] = temps.max(axis=1)
    out["nearby_station_last_temp_spread_f"] = out["nearby_station_last_temp_max_f"] - out["nearby_station_last_temp_min_f"]
    klga_temp = _numeric_series(out, "wu_last_temp_f")
    out["nearby_station_mean_minus_klga_last_temp_f"] = out["nearby_station_last_temp_mean_f"] - klga_temp
    out["klga_minus_nearby_station_mean_last_temp_f"] = klga_temp - out["nearby_station_last_temp_mean_f"]
    out["klga_warmest_vs_nearby_last_temp"] = (klga_temp > out["nearby_station_last_temp_max_f"]).astype("boolean")
    out["klga_coldest_vs_nearby_last_temp"] = (klga_temp < out["nearby_station_last_temp_min_f"]).astype("boolean")

    change_columns = [f"{prefix}_temp_change_3h_f" for prefix in prefixes if f"{prefix}_temp_change_3h_f" in out.columns]
    if change_columns:
        changes = out.loc[:, change_columns].apply(pd.to_numeric, errors="coerce")
        out["nearby_station_temp_change_3h_mean_f"] = changes.mean(axis=1)
        out["nearby_station_temp_change_3h_spread_f"] = changes.max(axis=1) - changes.min(axis=1)
        out["nearby_station_temp_change_3h_mean_minus_klga_f"] = out["nearby_station_temp_change_3h_mean_f"] - _numeric_series(out, "wu_temp_change_3h_f")
    return out


def add_nearby_station_feature_blocks(
    df: pd.DataFrame,
    nearby_obs_by_station: dict[str, pd.DataFrame],
    *,
    cutoff_local_time: str = "00:05",
    max_obs_age_hours: float = 12.0,
) -> pd.DataFrame:
    out = df.copy()
    station_ids = list(nearby_obs_by_station)
    for station_id, obs_df in nearby_obs_by_station.items():
        features = build_nearby_station_features(
            out["target_date_local"],
            obs_df,
            station_id=station_id,
            cutoff_local_time=cutoff_local_time,
            max_obs_age_hours=max_obs_age_hours,
        )
        out = out.merge(features, on="target_date_local", how="left")
        out = add_nearby_station_derived_features(out, station_id=station_id)
    return add_nearby_aggregate_features(out, station_ids=station_ids)
