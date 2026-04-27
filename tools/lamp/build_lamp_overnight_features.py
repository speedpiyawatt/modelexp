#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.progress import ProgressBar
from tools.lamp.parse_lamp_ascii import SOURCE_MODEL, SOURCE_PRODUCT, SOURCE_VERSION
from tools.weather.location_context import SETTLEMENT_LOCATION


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_FEATURES_ROOT = SCRIPT_DIR / "data" / "runtime" / "features"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "runtime" / "overnight"
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"
PROGRESS_RENDER_EVERY = 100
CHECKPOINT_HOURS = (6, 9, 12, 15, 18, 21)
REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "status",
    "wide_output_path",
    "init_time_utc",
    "init_time_local",
    "source_model",
    "source_product",
    "source_version",
    "station_id",
)
REQUIRED_WIDE_INPUT_COLUMNS: tuple[str, ...] = (
    "init_time_utc",
    "init_time_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "tmp",
    "dpt",
    "wsp",
    "wdr",
    "wgs",
    "cld",
    "cig",
    "vis",
    "obv",
    "typ",
    "p01",
    "p06",
    "p12",
    "pos",
    "poz",
)

CHECKPOINT_LABEL_SPECS: tuple[tuple[str, str, str], ...] = (
    ("tmp", "tmp_f_at_{hour:02d}", "numeric"),
    ("dpt", "dpt_f_at_{hour:02d}", "numeric"),
    ("wsp", "wsp_kt_at_{hour:02d}", "numeric"),
    ("wdr", "wdr_deg_at_{hour:02d}", "numeric"),
    ("wgs", "wgs_code_at_{hour:02d}", "string"),
    ("cld", "cld_code_at_{hour:02d}", "string"),
    ("cig", "cig_hundreds_ft_at_{hour:02d}", "numeric"),
    ("vis", "vis_miles_at_{hour:02d}", "numeric"),
    ("obv", "obv_code_at_{hour:02d}", "string"),
    ("typ", "typ_code_at_{hour:02d}", "string"),
)

OUTPUT_COLUMNS: tuple[str, ...] = (
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "target_date_local",
    "selected_init_time_utc",
    "selected_init_time_local",
    "previous_init_time_utc",
    "previous_init_time_local",
    "selection_cutoff_local",
    "coverage_complete",
    "missing_checkpoint_any",
    "missing_checkpoint_count",
    "revision_available",
    "forecast_hour_min",
    "forecast_hour_max",
    "target_day_row_count",
    "day_tmp_max_f_forecast",
    "day_tmp_min_f_forecast",
    "day_tmp_range_f_forecast",
    "day_tmp_argmax_local_hour",
    "morning_cld_mode",
    "morning_cig_min_hundreds_ft",
    "morning_vis_min_miles",
    "morning_obv_any",
    "morning_ifr_like_any",
    "day_p01_max_pct",
    "day_p06_max_pct",
    "day_p12_max_pct",
    "day_pos_max_pct",
    "day_poz_max_pct",
    "day_precip_type_any",
    "day_precip_type_mode",
    "rev_day_tmp_max_f",
    "rev_day_p01_max_pct",
    "rev_day_pos_max_pct",
    "rev_morning_cig_min_hundreds_ft",
    "rev_morning_vis_min_miles",
    *(template.format(hour=hour) for hour in CHECKPOINT_HOURS for _, template, _ in CHECKPOINT_LABEL_SPECS),
    *(f"rev_tmp_f_at_{hour:02d}" for hour in CHECKPOINT_HOURS),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build overnight LAMP summary features from issue-level KLGA LAMP parquet outputs.")
    parser.add_argument("--features-root", type=pathlib.Path, default=DEFAULT_FEATURES_ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-local-date", required=True, help="First target local date in YYYY-MM-DD.")
    parser.add_argument("--end-local-date", required=True, help="Last target local date in YYYY-MM-DD.")
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME, help="Latest allowed issue time in America/New_York, HH:MM.")
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def parse_local_time(value: str) -> dt.time:
    return dt.time.fromisoformat(value)


def iter_local_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    days = (end_date - start_date).days
    return [start_date + dt.timedelta(days=offset) for offset in range(days + 1)]


def output_paths(output_dir: pathlib.Path, target_date_local: dt.date) -> tuple[pathlib.Path, pathlib.Path]:
    root = output_dir / f"target_date_local={target_date_local.isoformat()}"
    root.mkdir(parents=True, exist_ok=True)
    return root / "lamp.overnight.parquet", root / "lamp.overnight.manifest.parquet"


def parse_local_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(NY_TZ)
    return timestamp.tz_convert(NY_TZ)


def parse_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def resolve_manifest_output_path(
    manifest_path: pathlib.Path,
    path_value: object,
    *,
    default_filename: str,
) -> pathlib.Path | None:
    if path_value is None or pd.isna(path_value):
        candidate = manifest_path.with_name(default_filename)
        return candidate if candidate.exists() else None

    candidate = pathlib.Path(str(path_value))
    if candidate.is_absolute():
        return candidate if candidate.exists() else None

    repo_relative_candidate = (REPO_ROOT / candidate).resolve()
    if repo_relative_candidate.exists():
        return repo_relative_candidate

    manifest_relative_candidate = (manifest_path.parent / candidate).resolve()
    if manifest_relative_candidate.exists():
        return manifest_relative_candidate

    return None


def discover_issue_catalog(features_root: pathlib.Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manifest_path in sorted(features_root.rglob("lamp.manifest.parquet")):
        manifest_df = pd.read_parquet(manifest_path, columns=REQUIRED_MANIFEST_COLUMNS)
        if manifest_df.empty:
            continue
        manifest_row = manifest_df.iloc[0]
        if str(manifest_row.get("status", "")).lower() != "ok":
            continue
        # LAMP manifests may store repo-relative output paths instead of
        # manifest-relative paths, so accept both contracts here.
        wide_path = resolve_manifest_output_path(
            manifest_path,
            manifest_row.get("wide_output_path"),
            default_filename="lamp.wide.parquet",
        )
        if wide_path is None:
            continue
        init_time_utc = parse_utc_timestamp(manifest_row["init_time_utc"])
        init_time_local = parse_local_timestamp(manifest_row["init_time_local"])
        rows.append(
            {
                "manifest_path": str(manifest_path),
                "wide_path": str(wide_path),
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "source_model": manifest_row.get("source_model", SOURCE_MODEL),
                "source_product": manifest_row.get("source_product", SOURCE_PRODUCT),
                "source_version": manifest_row.get("source_version", SOURCE_VERSION),
                "station_id": manifest_row.get("station_id", SETTLEMENT_LOCATION.station_id),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["manifest_path", "wide_path", "init_time_utc", "init_time_local", "source_model", "source_product", "source_version", "station_id"])
    return pd.DataFrame.from_records(rows).sort_values("init_time_local").reset_index(drop=True)


def selection_cutoff_timestamp(target_date_local: dt.date, cutoff_local_time: dt.time) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(target_date_local, cutoff_local_time, tzinfo=NY_TZ))


def select_issues_for_target(
    issue_catalog: pd.DataFrame,
    *,
    target_date_local: dt.date,
    cutoff_local_time: dt.time,
) -> tuple[pd.Series | None, pd.Series | None, pd.Timestamp]:
    cutoff_ts = selection_cutoff_timestamp(target_date_local, cutoff_local_time)
    eligible = issue_catalog.loc[issue_catalog["init_time_local"] <= cutoff_ts].sort_values("init_time_local")
    if eligible.empty:
        return None, None, cutoff_ts
    selected = eligible.iloc[-1]
    previous = eligible.iloc[-2] if len(eligible) > 1 else None
    return selected, previous, cutoff_ts


def available_parquet_columns(path: str | pathlib.Path) -> set[str]:
    return set(pq.read_schema(path).names)


def load_issue_frame(wide_path: str | pathlib.Path) -> pd.DataFrame:
    available_columns = available_parquet_columns(wide_path)
    projected_columns = [column for column in REQUIRED_WIDE_INPUT_COLUMNS if column in available_columns]
    df = pd.read_parquet(wide_path, columns=projected_columns).copy()
    if "init_time_utc" in df.columns:
        df["init_time_utc"] = df["init_time_utc"].map(parse_utc_timestamp)
    if "valid_time_utc" in df.columns:
        df["valid_time_utc"] = df["valid_time_utc"].map(parse_utc_timestamp)
    if "init_time_local" in df.columns:
        df["init_time_local"] = df["init_time_local"].map(parse_local_timestamp)
    if "valid_time_local" in df.columns:
        df["valid_time_local"] = df["valid_time_local"].map(parse_local_timestamp)
    return df.sort_values("valid_time_local").reset_index(drop=True)


def target_day_frame(issue_df: pd.DataFrame, *, target_date_local: dt.date) -> pd.DataFrame:
    if issue_df.empty:
        return issue_df.copy()
    return issue_df.loc[issue_df["valid_date_local"] == target_date_local.isoformat()].sort_values("valid_time_local").reset_index(drop=True)


def mode_or_none(series: pd.Series) -> object:
    non_null = series.dropna()
    if non_null.empty:
        return None
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return None
    return modes.iloc[0]


def boolish_any(series: pd.Series, *, missing_tokens: set[str] | None = None) -> bool:
    missing_tokens = missing_tokens or {"", "NG", "NONE"}
    non_null = series.dropna()
    if non_null.empty:
        return False
    normalized = {str(value).strip().upper() for value in non_null}
    return any(token not in missing_tokens for token in normalized)


def extract_checkpoint_values(day_df: pd.DataFrame) -> tuple[dict[str, object], dict[int, float | None], int]:
    values: dict[str, object] = {}
    tmp_by_hour: dict[int, float | None] = {}
    missing_count = 0
    if day_df.empty:
        for hour in CHECKPOINT_HOURS:
            for _, template, _ in CHECKPOINT_LABEL_SPECS:
                values[template.format(hour=hour)] = None
            tmp_by_hour[hour] = None
        return values, tmp_by_hour, len(CHECKPOINT_HOURS)

    day_df = day_df.copy()
    day_df["valid_local_hour"] = pd.to_datetime(day_df["valid_time_local"]).dt.hour
    for hour in CHECKPOINT_HOURS:
        row = day_df.loc[day_df["valid_local_hour"] == hour]
        if row.empty:
            missing_count += 1
            for _, template, _ in CHECKPOINT_LABEL_SPECS:
                values[template.format(hour=hour)] = None
            tmp_by_hour[hour] = None
            continue
        record = row.iloc[0]
        for source_column, template, kind in CHECKPOINT_LABEL_SPECS:
            value = record[source_column] if source_column in record.index else None
            if pd.isna(value):
                value = None
            elif kind == "string":
                value = str(value)
            elif kind == "numeric":
                value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                value = None if pd.isna(value) else value
            values[template.format(hour=hour)] = value
        tmp_by_hour[hour] = values[f"tmp_f_at_{hour:02d}"]
    return values, tmp_by_hour, missing_count


def summarize_day(day_df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "forecast_hour_min": None,
        "forecast_hour_max": None,
        "target_day_row_count": int(len(day_df)),
        "day_tmp_max_f_forecast": None,
        "day_tmp_min_f_forecast": None,
        "day_tmp_range_f_forecast": None,
        "day_tmp_argmax_local_hour": None,
        "morning_cld_mode": None,
        "morning_cig_min_hundreds_ft": None,
        "morning_vis_min_miles": None,
        "morning_obv_any": False,
        "morning_ifr_like_any": False,
        "day_p01_max_pct": None,
        "day_p06_max_pct": None,
        "day_p12_max_pct": None,
        "day_pos_max_pct": None,
        "day_poz_max_pct": None,
        "day_precip_type_any": False,
        "day_precip_type_mode": None,
    }
    if day_df.empty:
        return summary

    summary["forecast_hour_min"] = int(pd.to_numeric(day_df["forecast_hour"], errors="coerce").min())
    summary["forecast_hour_max"] = int(pd.to_numeric(day_df["forecast_hour"], errors="coerce").max())

    if "tmp" in day_df.columns:
        tmp_series = pd.to_numeric(day_df["tmp"], errors="coerce")
        if not tmp_series.dropna().empty:
            summary["day_tmp_max_f_forecast"] = float(tmp_series.max())
            summary["day_tmp_min_f_forecast"] = float(tmp_series.min())
            summary["day_tmp_range_f_forecast"] = float(tmp_series.max() - tmp_series.min())
            argmax_idx = tmp_series.idxmax()
            summary["day_tmp_argmax_local_hour"] = int(pd.Timestamp(day_df.loc[argmax_idx, "valid_time_local"]).hour)

    morning = day_df.loc[pd.to_datetime(day_df["valid_time_local"]).dt.hour.between(6, 12)].copy()
    if not morning.empty:
        summary["morning_cld_mode"] = mode_or_none(morning["cld"]) if "cld" in morning.columns else None
        cig_series = pd.to_numeric(morning["cig"], errors="coerce") if "cig" in morning.columns else pd.Series(dtype=float)
        vis_series = pd.to_numeric(morning["vis"], errors="coerce") if "vis" in morning.columns else pd.Series(dtype=float)
        if not cig_series.dropna().empty:
            summary["morning_cig_min_hundreds_ft"] = float(cig_series.min())
        if not vis_series.dropna().empty:
            summary["morning_vis_min_miles"] = float(vis_series.min())
        summary["morning_obv_any"] = boolish_any(morning["obv"]) if "obv" in morning.columns else False
        summary["morning_ifr_like_any"] = bool(
            (not cig_series.dropna().empty and float(cig_series.min()) <= 10.0)
            or (not vis_series.dropna().empty and float(vis_series.min()) <= 3.0)
        )

    for source_column, output_column in (
        ("p01", "day_p01_max_pct"),
        ("p06", "day_p06_max_pct"),
        ("p12", "day_p12_max_pct"),
        ("pos", "day_pos_max_pct"),
        ("poz", "day_poz_max_pct"),
    ):
        if source_column not in day_df.columns:
            continue
        numeric = pd.to_numeric(day_df[source_column], errors="coerce")
        if not numeric.dropna().empty:
            summary[output_column] = float(numeric.max())

    if "typ" in day_df.columns:
        typ_values = day_df["typ"].dropna().astype(str)
        filtered = typ_values.loc[~typ_values.str.strip().str.upper().isin({"", "NG", "NONE"})]
        summary["day_precip_type_any"] = not filtered.empty
        summary["day_precip_type_mode"] = mode_or_none(filtered) if not filtered.empty else None

    return summary


def build_revision_features(selected_row: dict[str, object], previous_day_df: pd.DataFrame | None) -> dict[str, object]:
    revisions: dict[str, object] = {
        "revision_available": previous_day_df is not None and not previous_day_df.empty,
        "rev_day_tmp_max_f": None,
        "rev_day_p01_max_pct": None,
        "rev_day_pos_max_pct": None,
        "rev_morning_cig_min_hundreds_ft": None,
        "rev_morning_vis_min_miles": None,
        **{f"rev_tmp_f_at_{hour:02d}": None for hour in CHECKPOINT_HOURS},
    }
    if previous_day_df is None or previous_day_df.empty:
        revisions["revision_available"] = False
        return revisions

    previous_summary = summarize_day(previous_day_df)
    if selected_row.get("day_tmp_max_f_forecast") is not None and previous_summary["day_tmp_max_f_forecast"] is not None:
        revisions["rev_day_tmp_max_f"] = float(selected_row["day_tmp_max_f_forecast"] - previous_summary["day_tmp_max_f_forecast"])
    if selected_row.get("day_p01_max_pct") is not None and previous_summary["day_p01_max_pct"] is not None:
        revisions["rev_day_p01_max_pct"] = float(selected_row["day_p01_max_pct"] - previous_summary["day_p01_max_pct"])
    if selected_row.get("day_pos_max_pct") is not None and previous_summary["day_pos_max_pct"] is not None:
        revisions["rev_day_pos_max_pct"] = float(selected_row["day_pos_max_pct"] - previous_summary["day_pos_max_pct"])
    if selected_row.get("morning_cig_min_hundreds_ft") is not None and previous_summary["morning_cig_min_hundreds_ft"] is not None:
        revisions["rev_morning_cig_min_hundreds_ft"] = float(
            selected_row["morning_cig_min_hundreds_ft"] - previous_summary["morning_cig_min_hundreds_ft"]
        )
    if selected_row.get("morning_vis_min_miles") is not None and previous_summary["morning_vis_min_miles"] is not None:
        revisions["rev_morning_vis_min_miles"] = float(selected_row["morning_vis_min_miles"] - previous_summary["morning_vis_min_miles"])

    previous_checkpoints, previous_tmp_by_hour, _ = extract_checkpoint_values(previous_day_df)
    del previous_checkpoints
    for hour in CHECKPOINT_HOURS:
        current_value = selected_row.get(f"tmp_f_at_{hour:02d}")
        previous_value = previous_tmp_by_hour.get(hour)
        if current_value is not None and previous_value is not None:
            revisions[f"rev_tmp_f_at_{hour:02d}"] = float(current_value - previous_value)
    return revisions


def build_target_row(
    *,
    target_date_local: dt.date,
    selected_issue: pd.Series,
    previous_issue: pd.Series | None,
    cutoff_ts: pd.Timestamp,
) -> dict[str, object]:
    selected_df = load_issue_frame(selected_issue["wide_path"])
    selected_day_df = target_day_frame(selected_df, target_date_local=target_date_local)
    checkpoint_values, _, missing_checkpoint_count = extract_checkpoint_values(selected_day_df)
    summary = summarize_day(selected_day_df)

    row: dict[str, object] = {column: None for column in OUTPUT_COLUMNS}
    row.update(
        {
            "source_model": selected_issue["source_model"],
            "source_product": selected_issue["source_product"],
            "source_version": selected_issue["source_version"],
            "station_id": selected_issue["station_id"],
            "target_date_local": target_date_local.isoformat(),
            "selected_init_time_utc": pd.Timestamp(selected_issue["init_time_utc"]).isoformat(),
            "selected_init_time_local": pd.Timestamp(selected_issue["init_time_local"]).isoformat(),
            "previous_init_time_utc": pd.Timestamp(previous_issue["init_time_utc"]).isoformat() if previous_issue is not None else None,
            "previous_init_time_local": pd.Timestamp(previous_issue["init_time_local"]).isoformat() if previous_issue is not None else None,
            "selection_cutoff_local": cutoff_ts.isoformat(),
            "coverage_complete": missing_checkpoint_count == 0,
            "missing_checkpoint_any": missing_checkpoint_count > 0,
            "missing_checkpoint_count": missing_checkpoint_count,
            **summary,
            **checkpoint_values,
        }
    )

    previous_day_df = None
    if previous_issue is not None:
        previous_df = load_issue_frame(previous_issue["wide_path"])
        previous_day_df = target_day_frame(previous_df, target_date_local=target_date_local)
    row.update(build_revision_features(row, previous_day_df))
    return row


def failure_manifest_row(*, target_date_local: dt.date, cutoff_ts: pd.Timestamp, overnight_path: pathlib.Path, manifest_path: pathlib.Path) -> dict[str, object]:
    return {
        "source_model": SOURCE_MODEL,
        "source_product": SOURCE_PRODUCT,
        "source_version": SOURCE_VERSION,
        "station_id": SETTLEMENT_LOCATION.station_id,
        "target_date_local": target_date_local.isoformat(),
        "status": "incomplete",
        "extraction_status": "no_qualifying_issue",
        "processed_timestamp_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "selection_cutoff_local": cutoff_ts.isoformat(),
        "selected_init_time_utc": None,
        "selected_init_time_local": None,
        "previous_init_time_utc": None,
        "previous_init_time_local": None,
        "revision_available": False,
        "coverage_complete": False,
        "missing_checkpoint_any": True,
        "missing_checkpoint_count": len(CHECKPOINT_HOURS),
        "warning": "No qualifying LAMP issue was available at or before the overnight local cutoff.",
        "failure_reason": "no_qualifying_issue",
        "overnight_output_path": str(overnight_path),
        "manifest_parquet_path": str(manifest_path),
        "row_count": 0,
    }


def success_manifest_row(
    *,
    target_date_local: dt.date,
    cutoff_ts: pd.Timestamp,
    overnight_path: pathlib.Path,
    manifest_path: pathlib.Path,
    row: dict[str, object],
) -> dict[str, object]:
    warnings: list[str] = []
    if row["missing_checkpoint_any"]:
        warnings.append(
            f"Missing target-day LAMP checkpoints: {row['missing_checkpoint_count']} of {len(CHECKPOINT_HOURS)}."
        )
    if not row["revision_available"]:
        warnings.append("No prior qualifying LAMP issue was available for revision features.")
    return {
        "source_model": row["source_model"],
        "source_product": row["source_product"],
        "source_version": row["source_version"],
        "station_id": row["station_id"],
        "target_date_local": target_date_local.isoformat(),
        "status": "ok",
        "extraction_status": "ok",
        "processed_timestamp_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "selection_cutoff_local": cutoff_ts.isoformat(),
        "selected_init_time_utc": row["selected_init_time_utc"],
        "selected_init_time_local": row["selected_init_time_local"],
        "previous_init_time_utc": row["previous_init_time_utc"],
        "previous_init_time_local": row["previous_init_time_local"],
        "revision_available": bool(row["revision_available"]),
        "coverage_complete": bool(row["coverage_complete"]),
        "missing_checkpoint_any": bool(row["missing_checkpoint_any"]),
        "missing_checkpoint_count": int(row["missing_checkpoint_count"]),
        "warning": "; ".join(warnings),
        "failure_reason": None,
        "overnight_output_path": str(overnight_path),
        "manifest_parquet_path": str(manifest_path),
        "row_count": 1,
    }


def build_for_date(
    *,
    issue_catalog: pd.DataFrame,
    target_date_local: dt.date,
    cutoff_local_time: dt.time,
    output_dir: pathlib.Path,
    progress: ProgressBar | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    overnight_path, manifest_path = output_paths(output_dir, target_date_local)
    if progress is not None:
        progress.update(stage="select", status=f"target_date_local={target_date_local.isoformat()}")
    selected_issue, previous_issue, cutoff_ts = select_issues_for_target(
        issue_catalog,
        target_date_local=target_date_local,
        cutoff_local_time=cutoff_local_time,
    )
    if selected_issue is None:
        if progress is not None:
            progress.update(stage="write", status=f"target_date_local={target_date_local.isoformat()} no_qualifying_issue")
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_parquet(overnight_path, index=False)
        pd.DataFrame.from_records(
            [failure_manifest_row(target_date_local=target_date_local, cutoff_ts=cutoff_ts, overnight_path=overnight_path, manifest_path=manifest_path)]
        ).to_parquet(manifest_path, index=False)
        return overnight_path, manifest_path

    if progress is not None:
        progress.update(stage="load", status=f"target_date_local={target_date_local.isoformat()} selected_issue")
    row = build_target_row(
        target_date_local=target_date_local,
        selected_issue=selected_issue,
        previous_issue=previous_issue,
        cutoff_ts=cutoff_ts,
    )
    if progress is not None:
        progress.update(stage="write", status=f"target_date_local={target_date_local.isoformat()} write_outputs")
    pd.DataFrame.from_records([row], columns=OUTPUT_COLUMNS).to_parquet(overnight_path, index=False)
    pd.DataFrame.from_records(
        [success_manifest_row(target_date_local=target_date_local, cutoff_ts=cutoff_ts, overnight_path=overnight_path, manifest_path=manifest_path, row=row)]
    ).to_parquet(manifest_path, index=False)
    return overnight_path, manifest_path


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


def main() -> int:
    args = parse_args()
    start_date = parse_local_date(args.start_local_date)
    end_date = parse_local_date(args.end_local_date)
    cutoff_local_time = parse_local_time(args.cutoff_local_time)
    issue_catalog = discover_issue_catalog(args.features_root)
    written: list[pathlib.Path] = []
    target_dates = iter_local_dates(start_date, end_date)
    progress = ProgressBar(len(target_dates), label="LAMP overnight", unit="date")
    for date_index, target_date_local in enumerate(target_dates, start=1):
        overnight_path, manifest_path = build_for_date(
            issue_catalog=issue_catalog,
            target_date_local=target_date_local,
            cutoff_local_time=cutoff_local_time,
            output_dir=args.output_dir,
            progress=progress,
        )
        written.extend([overnight_path, manifest_path])
        update_progress_sparse(
            progress,
            completed=date_index,
            total=len(target_dates),
            stage="complete",
            status=f"target_date_local={target_date_local.isoformat()} done",
        )
    progress.close(stage="finalize", status=f"wrote_files={len(written)}")
    print(f"[ok] files={len(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
