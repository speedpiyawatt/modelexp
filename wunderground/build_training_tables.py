#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from zoneinfo import ZoneInfo

import pandas as pd


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_HISTORY_DIR = pathlib.Path("wunderground/output/history")
DEFAULT_OUTPUT_DIR = pathlib.Path("wunderground/output/tables")
DEFAULT_STATION_ID = "KLGA"


OBSERVATION_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("temp", "temp_f"),
    ("dewPt", "dewpoint_f"),
    ("rh", "rh_pct"),
    ("pressure", "pressure_in"),
    ("wspd", "wind_speed_mph"),
    ("wdir", "wind_dir_deg"),
    ("gust", "wind_gust_mph"),
    ("vis", "visibility"),
    ("clds", "cloud_cover_code"),
    ("wx_phrase", "wx_phrase"),
    ("precip_hrly", "precip_hrly_in"),
    ("snow_hrly", "snow_hrly_in"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build settlement-aligned Wunderground training tables for KLGA.")
    parser.add_argument("--history-dir", type=pathlib.Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument(
        "--file-glob",
        default=None,
        help="Optional JSON filename glob. Defaults to '<station_id>_9_US_*.json' and falls back to '*.json'.",
    )
    return parser.parse_args()


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _load_history_file(path: pathlib.Path, *, station_id: str) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    observations = payload.get("observations", [])
    if not observations:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for obs in observations:
        valid_time_utc = pd.to_datetime(obs["valid_time_gmt"], unit="s", utc=True)
        valid_time_local = valid_time_utc.tz_convert(NY_TZ)
        row: dict[str, object] = {
            "station_id": station_id,
            "source_file": path.name,
            "valid_time_utc": valid_time_utc.isoformat(),
            "valid_time_local": valid_time_local.isoformat(),
            "date_local": valid_time_local.date().isoformat(),
        }
        for source_name, output_name in OBSERVATION_COLUMN_MAP:
            row[output_name] = obs.get(source_name)
        rows.append(row)

    frame = pd.DataFrame.from_records(rows).sort_values("valid_time_utc").reset_index(drop=True)
    for _, output_name in OBSERVATION_COLUMN_MAP:
        if output_name in {"cloud_cover_code", "wx_phrase"}:
            continue
        frame[output_name] = _coerce_numeric(frame[output_name])
    return frame


def history_files(history_dir: pathlib.Path, *, station_id: str = DEFAULT_STATION_ID, file_glob: str | None = None) -> list[pathlib.Path]:
    if file_glob is not None:
        return sorted(history_dir.glob(file_glob))
    station_glob = f"{station_id}_9_US_*.json"
    paths = sorted(history_dir.glob(station_glob))
    return paths if paths else sorted(history_dir.glob("*.json"))


def build_wu_obs_intraday(
    history_dir: pathlib.Path,
    *,
    station_id: str = DEFAULT_STATION_ID,
    file_glob: str | None = None,
) -> pd.DataFrame:
    frames = [_load_history_file(path, station_id=station_id) for path in history_files(history_dir, station_id=station_id, file_glob=file_glob)]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(
            columns=[
                "station_id",
                "source_file",
                "valid_time_utc",
                "valid_time_local",
                "date_local",
                *(output_name for _, output_name in OBSERVATION_COLUMN_MAP),
                "max_so_far_f",
                "warming_rate_1h_f",
                "warming_rate_3h_f",
            ]
        )

    obs_df = pd.concat(frames, ignore_index=True)
    obs_df["valid_time_utc"] = pd.to_datetime(obs_df["valid_time_utc"], utc=True)
    obs_df["valid_time_local"] = pd.to_datetime(obs_df["valid_time_local"], utc=True).dt.tz_convert(NY_TZ)
    obs_df = obs_df.sort_values(["date_local", "valid_time_local"]).reset_index(drop=True)

    obs_df["max_so_far_f"] = obs_df.groupby("date_local")["temp_f"].cummax()

    for hours, output_name in ((1, "warming_rate_1h_f"), (3, "warming_rate_3h_f")):
        lagged_groups: list[pd.DataFrame] = []
        for _, group in obs_df.groupby("date_local", sort=True):
            history = group[["valid_time_local", "temp_f"]].rename(columns={"valid_time_local": "lag_time_local", "temp_f": "lag_temp_f"})
            targets = group[["valid_time_local", "temp_f"]].copy()
            targets["lookup_time_local"] = targets["valid_time_local"] - pd.Timedelta(hours=hours)
            merged = pd.merge_asof(
                targets.sort_values("lookup_time_local"),
                history.sort_values("lag_time_local"),
                left_on="lookup_time_local",
                right_on="lag_time_local",
                direction="backward",
            )
            merged[output_name] = merged["temp_f"] - merged["lag_temp_f"]
            lagged_groups.append(merged[["valid_time_local", output_name]])
        lagged_df = pd.concat(lagged_groups, ignore_index=True)
        obs_df = obs_df.merge(lagged_df, on="valid_time_local", how="left")

    obs_df["valid_time_utc"] = obs_df["valid_time_utc"].map(lambda value: value.isoformat())
    obs_df["valid_time_local"] = obs_df["valid_time_local"].map(lambda value: value.isoformat())
    return obs_df.reset_index(drop=True)


def build_labels_daily(obs_df: pd.DataFrame) -> pd.DataFrame:
    if obs_df.empty:
        return pd.DataFrame(
            columns=[
                "target_date_local",
                "station_id",
                "label_final_tmax_f",
                "label_final_tmin_f",
                "label_market_bin",
                "label_obs_count",
                "label_first_obs_time_local",
                "label_last_obs_time_local",
                "label_total_precip_in",
                "label_source_file",
            ]
        )

    labels = []
    grouped = obs_df.groupby(["date_local", "station_id"], sort=True)
    for (date_local, station_id), group in grouped:
        group = group.sort_values("valid_time_local")
        final_tmax = _coerce_numeric(group["temp_f"]).max()
        final_tmin = _coerce_numeric(group["temp_f"]).min()
        total_precip = _coerce_numeric(group["precip_hrly_in"]).fillna(0.0).sum()
        market_bin = None if pd.isna(final_tmax) else f"{int(round(float(final_tmax)))}F"
        labels.append(
            {
                "target_date_local": date_local,
                "station_id": station_id,
                "label_final_tmax_f": None if pd.isna(final_tmax) else float(final_tmax),
                "label_final_tmin_f": None if pd.isna(final_tmin) else float(final_tmin),
                "label_market_bin": market_bin,
                "label_obs_count": int(len(group)),
                "label_first_obs_time_local": str(group["valid_time_local"].iloc[0]),
                "label_last_obs_time_local": str(group["valid_time_local"].iloc[-1]),
                "label_total_precip_in": float(total_precip),
                "label_source_file": str(group["source_file"].iloc[0]),
            }
        )
    return pd.DataFrame.from_records(labels)


def build_training_tables(
    history_dir: pathlib.Path,
    *,
    station_id: str = DEFAULT_STATION_ID,
    file_glob: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    obs_df = build_wu_obs_intraday(history_dir, station_id=station_id, file_glob=file_glob)
    labels_df = build_labels_daily(obs_df)
    return labels_df, obs_df


def write_outputs(*, output_dir: pathlib.Path, labels_df: pd.DataFrame, obs_df: pd.DataFrame) -> tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "labels_daily.parquet"
    obs_path = output_dir / "wu_obs_intraday.parquet"
    labels_df.to_parquet(labels_path, index=False)
    obs_df.to_parquet(obs_path, index=False)
    return labels_path, obs_path


def main() -> int:
    args = parse_args()
    labels_df, obs_df = build_training_tables(args.history_dir, station_id=args.station_id, file_glob=args.file_glob)
    labels_path, obs_path = write_outputs(output_dir=args.output_dir, labels_df=labels_df, obs_df=obs_df)
    print(f"Wrote {labels_path}")
    print(f"Wrote {obs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
