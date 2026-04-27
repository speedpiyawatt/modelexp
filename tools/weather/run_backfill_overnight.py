#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather import build_training_features_overnight as merged_builder
from tools.weather import build_training_features_overnight_normalized as normalized_builder
from tools.weather import filter_wu_training_tables as wu_filter


DEFAULT_RUN_ROOT = pathlib.Path("tools/weather/data/runtime/backfill_overnight")
DEFAULT_LABELS_PATH = pathlib.Path("wunderground/output/tables/labels_daily.parquet")
DEFAULT_OBS_PATH = pathlib.Path("wunderground/output/tables/wu_obs_intraday.parquet")
DEFAULT_STATION_ID = "KLGA"
DEFAULT_CUTOFF_LOCAL_TIME = "00:05"
DEFAULT_STAGE_ORDER = ("wu", "nbm", "lamp", "hrrr", "merged", "normalized")
PAUSE_SENTINEL = "PAUSE"


@dataclass(frozen=True)
class DayWindow:
    target_date_local: dt.date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue-from-disk overnight backfill runner.")
    parser.add_argument("--run-root", type=pathlib.Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--labels-path", type=pathlib.Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--label-history-path", type=pathlib.Path, default=None)
    parser.add_argument("--obs-path", type=pathlib.Path, default=DEFAULT_OBS_PATH)
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    parser.add_argument("--station-id", default=DEFAULT_STATION_ID)
    parser.add_argument("--cutoff-local-time", default=DEFAULT_CUTOFF_LOCAL_TIME)
    parser.add_argument("--nbm-workers", type=int, default=1)
    parser.add_argument("--nbm-lead-workers", type=int, default=4)
    parser.add_argument("--hrrr-workers", type=int, default=4)
    parser.add_argument("--stages", nargs="*", choices=DEFAULT_STAGE_ORDER, default=list(DEFAULT_STAGE_ORDER))
    parser.add_argument("--force-rebuild-stage", action="append", default=[])
    parser.add_argument("--force-rebuild-date", action="append", default=[])
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def iter_days(start_date: dt.date, end_date: dt.date) -> list[DayWindow]:
    return [DayWindow(start_date + dt.timedelta(days=offset)) for offset in range((end_date - start_date).days + 1)]


def day_token(day: DayWindow) -> str:
    return day.target_date_local.isoformat()


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")


def pause_requested(run_root: pathlib.Path) -> bool:
    return (run_root / PAUSE_SENTINEL).exists()


def clear_temp_files(root: pathlib.Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*.tmp"):
        path.unlink(missing_ok=True)


def delete_path(path: pathlib.Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _existing_paths(*paths: pathlib.Path) -> list[pathlib.Path]:
    return [path for path in paths if path.exists()]


def _max_mtime(paths: list[pathlib.Path]) -> float:
    if not paths:
        return -1.0
    return max(path.stat().st_mtime for path in paths)


def wu_paths(run_root: pathlib.Path, day: DayWindow) -> dict[str, pathlib.Path]:
    root = run_root / "wu" / f"target_date_local={day_token(day)}"
    return {
        "root": root,
        "labels": root / "labels_daily.parquet",
        "label_history": root / "labels_history.parquet",
        "obs": root / "wu_obs_intraday.parquet",
        "manifest": root / "manifest.json",
    }


def validate_wu_day(run_root: pathlib.Path, day: DayWindow) -> bool:
    paths = wu_paths(run_root, day)
    if not all(path.exists() for key, path in paths.items() if key != "root"):
        return False
    try:
        manifest = json.loads(paths["manifest"].read_text())
        labels_df = pd.read_parquet(paths["labels"])
        label_history_df = pd.read_parquet(paths["label_history"])
        obs_df = pd.read_parquet(paths["obs"])
    except Exception:
        return False
    token = day_token(day)
    if manifest.get("start_local_date") != token or manifest.get("end_local_date") != token:
        return False
    if int(manifest.get("label_row_count", -1)) != int(len(labels_df)):
        return False
    if int(manifest.get("label_history_row_count", -1)) != int(len(label_history_df)):
        return False
    if int(manifest.get("obs_row_count", -1)) != int(len(obs_df)):
        return False
    return True


def wu_dependency_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    paths = wu_paths(run_root, day)
    return _existing_paths(paths["labels"], paths["label_history"], paths["obs"], paths["manifest"])


def ensure_wu_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    paths = wu_paths(args.run_root, day)
    if not force and validate_wu_day(args.run_root, day):
        print(f"[skip] wu date={day_token(day)}")
        return
    delete_path(paths["root"])
    labels_df = pd.read_parquet(args.labels_path)
    label_history_source = pd.read_parquet(args.label_history_path or args.labels_path)
    obs_df = pd.read_parquet(args.obs_path)
    filtered_labels_df = wu_filter.filter_target_labels(labels_df, start_date=day.target_date_local, end_date=day.target_date_local)
    filtered_label_history_df = wu_filter.filter_label_history(
        label_history_source,
        start_date=day.target_date_local,
        end_date=day.target_date_local,
    )
    filtered_obs_df = wu_filter.filter_obs_support(obs_df, start_date=day.target_date_local, end_date=day.target_date_local)
    if "station_id" in filtered_labels_df.columns:
        filtered_labels_df = filtered_labels_df.loc[filtered_labels_df["station_id"].astype(str) == str(args.station_id)].reset_index(drop=True)
    if "station_id" in filtered_label_history_df.columns:
        filtered_label_history_df = filtered_label_history_df.loc[
            filtered_label_history_df["station_id"].astype(str) == str(args.station_id)
        ].reset_index(drop=True)
    if "station_id" in filtered_obs_df.columns:
        filtered_obs_df = filtered_obs_df.loc[filtered_obs_df["station_id"].astype(str) == str(args.station_id)].reset_index(drop=True)
    paths["root"].mkdir(parents=True, exist_ok=True)
    filtered_labels_df.to_parquet(paths["labels"], index=False)
    filtered_label_history_df.to_parquet(paths["label_history"], index=False)
    filtered_obs_df.to_parquet(paths["obs"], index=False)
    paths["manifest"].write_text(
        json.dumps(
            wu_filter.build_manifest(
                start_date=day.target_date_local,
                end_date=day.target_date_local,
                labels_df=filtered_labels_df,
                label_history_df=filtered_label_history_df,
                obs_df=filtered_obs_df,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    if not validate_wu_day(args.run_root, day):
        raise ValueError(f"WU validation failed for target_date_local={day_token(day)}")
    print(f"[done] wu date={day_token(day)}")


def validate_nbm_day(run_root: pathlib.Path, day: DayWindow) -> bool:
    root = run_root / "nbm_overnight" / f"target_date_local={day_token(day)}"
    output_path = root / "nbm.overnight.parquet"
    manifest_path = root / "nbm.overnight.manifest.parquet"
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest_df = pd.read_parquet(manifest_path)
        pd.read_parquet(output_path)
    except Exception:
        return False
    if manifest_df.empty:
        return False
    status = str(manifest_df.iloc[0].get("status", ""))
    return status in {"ok", "no_qualifying_issue"}


def nbm_dependency_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    root = run_root / "nbm_overnight" / f"target_date_local={day_token(day)}"
    return _existing_paths(root / "nbm.overnight.parquet", root / "nbm.overnight.manifest.parquet")


def ensure_nbm_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    if not force and validate_nbm_day(args.run_root, day):
        print(f"[skip] nbm date={day_token(day)}")
        return
    delete_path(args.run_root / "nbm_overnight" / f"target_date_local={day_token(day)}")
    tmp_root = args.run_root / "nbm_tmp" / day_token(day)
    delete_path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    raw_dir = tmp_root / "raw"
    scratch_dir = tmp_root / "scratch"
    run_command(
        [
            sys.executable,
            "tools/nbm/build_grib2_features.py",
            "--start-local-date",
            day_token(day),
            "--end-local-date",
            day_token(day),
            "--selection-mode",
            "overnight_0005",
            "--output-dir",
            str(raw_dir),
            "--scratch-dir",
            str(scratch_dir),
            "--workers",
            str(args.nbm_workers),
            "--lead-workers",
            str(args.nbm_lead_workers),
            "--overwrite",
        ]
    )
    run_command(
        [
            sys.executable,
            "tools/nbm/build_nbm_overnight_features.py",
            "--features-root",
            str(raw_dir),
            "--output-dir",
            str(args.run_root / "nbm_overnight"),
            "--start-local-date",
            day_token(day),
            "--end-local-date",
            day_token(day),
        ]
    )
    if not validate_nbm_day(args.run_root, day):
        raise ValueError(f"NBM validation failed for target_date_local={day_token(day)}")
    delete_path(tmp_root)
    print(f"[done] nbm date={day_token(day)}")


def validate_lamp_day(run_root: pathlib.Path, day: DayWindow) -> bool:
    root = run_root / "lamp_overnight" / f"target_date_local={day_token(day)}"
    output_path = root / "lamp.overnight.parquet"
    manifest_path = root / "lamp.overnight.manifest.parquet"
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest_df = pd.read_parquet(manifest_path)
        pd.read_parquet(output_path)
    except Exception:
        return False
    if manifest_df.empty:
        return False
    row = manifest_df.iloc[0]
    return str(row.get("status", "")) == "ok" and str(row.get("extraction_status", "")) == "ok"


def lamp_dependency_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    root = run_root / "lamp_overnight" / f"target_date_local={day_token(day)}"
    return _existing_paths(root / "lamp.overnight.parquet", root / "lamp.overnight.manifest.parquet")


def ensure_lamp_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    if not force and validate_lamp_day(args.run_root, day):
        print(f"[skip] lamp date={day_token(day)}")
        return
    delete_path(args.run_root / "lamp_overnight" / f"target_date_local={day_token(day)}")
    tmp_root = args.run_root / "lamp_tmp" / day_token(day)
    delete_path(tmp_root)
    raw_dir = tmp_root / "raw"
    cache_dir = tmp_root / "cache"
    features_dir = tmp_root / "features"
    run_command(
        [
            sys.executable,
            "tools/lamp/fetch_lamp.py",
            "archive",
            "--start-utc-date",
            day_token(day),
            "--end-utc-date",
            day_token(day),
            "--cycle",
            "0230",
            "--cycle",
            "0330",
            "--cycle",
            "0430",
            "--output-dir",
            str(raw_dir),
            "--cache-dir",
            str(cache_dir),
            "--overwrite",
        ]
    )
    run_command([sys.executable, "tools/lamp/build_lamp_klga_features.py", str(raw_dir), "--output-dir", str(features_dir)])
    run_command(
        [
            sys.executable,
            "tools/lamp/build_lamp_overnight_features.py",
            "--features-root",
            str(features_dir),
            "--output-dir",
            str(args.run_root / "lamp_overnight"),
            "--start-local-date",
            day_token(day),
            "--end-local-date",
            day_token(day),
        ]
    )
    if not validate_lamp_day(args.run_root, day):
        raise ValueError(f"LAMP validation failed for target_date_local={day_token(day)}")
    delete_path(tmp_root)
    print(f"[done] lamp date={day_token(day)}")


def hrrr_summary_paths(run_root: pathlib.Path, day: DayWindow) -> dict[str, pathlib.Path]:
    summary_root = run_root / "hrrr_summary" / f"target_date_local={day_token(day)}"
    state_root = run_root / "hrrr_summary_state" / f"target_date_local={day_token(day)}"
    return {
        "summary_root": summary_root,
        "state_root": state_root,
        "summary": summary_root / "hrrr.overnight.parquet",
        "manifest_json": state_root / "hrrr.manifest.json",
        "manifest_parquet": state_root / "hrrr.manifest.parquet",
    }


def validate_hrrr_day(run_root: pathlib.Path, day: DayWindow) -> bool:
    paths = hrrr_summary_paths(run_root, day)
    if not paths["summary"].exists() or not paths["manifest_json"].exists() or not paths["manifest_parquet"].exists():
        return False
    try:
        summary_df = pd.read_parquet(paths["summary"])
        manifest = json.loads(paths["manifest_json"].read_text())
        manifest_df = pd.read_parquet(paths["manifest_parquet"])
    except Exception:
        return False
    if summary_df.empty or len(summary_df) != 1:
        return False
    if str(summary_df.iloc[0].get("target_date_local")) != day_token(day):
        return False
    if not bool(manifest.get("complete")):
        return False
    expected_count = int(manifest.get("expected_task_count", -1))
    completed_count = len(manifest.get("completed_task_keys", []))
    if expected_count < 0 or expected_count != completed_count:
        return False
    return bool(manifest_df.empty or (manifest_df["status"] == "ok").all())


def hrrr_dependency_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    paths = hrrr_summary_paths(run_root, day)
    return _existing_paths(paths["summary"], paths["manifest_json"], paths["manifest_parquet"])


def ensure_hrrr_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    paths = hrrr_summary_paths(args.run_root, day)
    if not force and validate_hrrr_day(args.run_root, day):
        print(f"[skip] hrrr date={day_token(day)}")
        return
    delete_path(paths["summary_root"])
    delete_path(paths["state_root"])
    tmp_root = args.run_root / "hrrr_tmp" / day_token(day)
    delete_path(tmp_root)
    output_dir = tmp_root / "output"
    summary_dir = tmp_root / "summary"
    run_command(
        [
            sys.executable,
            "tools/hrrr/build_hrrr_klga_feature_shards.py",
            "--start-date",
            day_token(day),
            "--end-date",
            day_token(day),
            "--selection-mode",
            "overnight_0005",
            "--download-dir",
            str(tmp_root / "downloads"),
            "--reduced-dir",
            str(tmp_root / "reduced"),
            "--output-dir",
            str(output_dir),
            "--summary-output-dir",
            str(summary_dir),
            "--scratch-dir",
            str(tmp_root / "scratch"),
            "--max-workers",
            str(args.hrrr_workers),
            "--allow-partial",
        ]
    )
    month_id = day.target_date_local.strftime("%Y-%m")
    summary_month_path = summary_dir / f"{month_id}.parquet"
    manifest_json_path = output_dir / f"{month_id}.manifest.json"
    manifest_parquet_path = output_dir / f"{month_id}.manifest.parquet"
    if not summary_month_path.exists() or not manifest_json_path.exists() or not manifest_parquet_path.exists():
        raise ValueError(f"HRRR one-day run missing month outputs for target_date_local={day_token(day)}")
    summary_df = pd.read_parquet(summary_month_path)
    summary_df = summary_df.loc[summary_df["target_date_local"] == day_token(day)].reset_index(drop=True)
    if len(summary_df) != 1:
        raise ValueError(f"HRRR one-day summary row mismatch for target_date_local={day_token(day)}")
    manifest = json.loads(manifest_json_path.read_text())
    manifest_df = pd.read_parquet(manifest_parquet_path).copy()
    paths["summary_root"].mkdir(parents=True, exist_ok=True)
    paths["state_root"].mkdir(parents=True, exist_ok=True)
    summary_df.to_parquet(paths["summary"], index=False)
    manifest["target_date_local"] = day_token(day)
    manifest["summary_parquet_path"] = str(paths["summary"])
    manifest["manifest_json_path"] = str(paths["manifest_json"])
    manifest["manifest_parquet_path"] = str(paths["manifest_parquet"])
    for key in (
        "wide_parquet_path",
        "provenance_parquet_path",
        "summary_output_dir",
        "output_dir",
        "download_dir",
        "reduced_dir",
        "scratch_dir",
    ):
        if key in manifest:
            manifest[key] = None
    paths["manifest_json"].write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest_df.empty:
        replacement_paths = {
            "summary_parquet_path": str(paths["summary"]),
            "manifest_json_path": str(paths["manifest_json"]),
            "manifest_parquet_path": str(paths["manifest_parquet"]),
        }
        for column, value in replacement_paths.items():
            if column in manifest_df.columns:
                manifest_df[column] = value
        for column in (
            "wide_parquet_path",
            "provenance_parquet_path",
            "output_dir",
            "summary_output_dir",
            "download_dir",
            "reduced_dir",
            "scratch_dir",
        ):
            if column in manifest_df.columns:
                manifest_df[column] = pd.NA
    manifest_df.to_parquet(paths["manifest_parquet"], index=False)
    if not validate_hrrr_day(args.run_root, day):
        raise ValueError(f"HRRR validation failed for target_date_local={day_token(day)}")
    delete_path(tmp_root)
    print(f"[done] hrrr date={day_token(day)}")


def validate_merged_day(run_root: pathlib.Path, day: DayWindow, *, allow_empty: bool) -> bool:
    output_path, manifest_path = merged_builder.output_paths_for_date(run_root / "training_features_overnight", day_token(day))
    return merged_builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local=day_token(day),
        allow_empty=allow_empty,
    )


def merged_output_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    output_path, manifest_path = merged_builder.output_paths_for_date(run_root / "training_features_overnight", day_token(day))
    return _existing_paths(output_path, manifest_path)


def merged_dependencies_current(run_root: pathlib.Path, day: DayWindow) -> bool:
    outputs = merged_output_paths(run_root, day)
    if not outputs:
        return False
    dependencies = (
        wu_dependency_paths(run_root, day)
        + nbm_dependency_paths(run_root, day)
        + lamp_dependency_paths(run_root, day)
        + hrrr_dependency_paths(run_root, day)
    )
    if not dependencies:
        return False
    return _max_mtime(outputs) >= _max_mtime(dependencies)


def ensure_merged_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    if not force and validate_merged_day(args.run_root, day, allow_empty=False) and merged_dependencies_current(args.run_root, day):
        print(f"[skip] merged date={day_token(day)}")
        return
    delete_path(args.run_root / "training_features_overnight" / f"target_date_local={day_token(day)}")
    wu = wu_paths(args.run_root, day)
    run_command(
        [
            sys.executable,
            "tools/weather/build_training_features_overnight.py",
            "--labels-path",
            str(wu["labels"]),
            "--label-history-path",
            str(wu["label_history"]),
            "--obs-path",
            str(wu["obs"]),
            "--nbm-root",
            str(args.run_root / "nbm_overnight"),
            "--lamp-root",
            str(args.run_root / "lamp_overnight"),
            "--hrrr-root",
            str(args.run_root / "hrrr_summary"),
            "--output-dir",
            str(args.run_root / "training_features_overnight"),
            "--start-local-date",
            day_token(day),
            "--end-local-date",
            day_token(day),
            "--station-id",
            args.station_id,
            "--cutoff-local-time",
            args.cutoff_local_time,
        ]
    )
    if not validate_merged_day(args.run_root, day, allow_empty=False):
        raise ValueError(f"Merged validation failed for target_date_local={day_token(day)}")
    print(f"[done] merged date={day_token(day)}")


def validate_normalized_day(run_root: pathlib.Path, day: DayWindow, *, allow_empty: bool) -> bool:
    output_path, manifest_path = normalized_builder.output_paths_for_date(run_root / "training_features_overnight_normalized", day_token(day))
    return normalized_builder.validate_date_output(
        output_path=output_path,
        manifest_path=manifest_path,
        target_date_local=day_token(day),
        allow_empty=allow_empty,
    )


def normalized_output_paths(run_root: pathlib.Path, day: DayWindow) -> list[pathlib.Path]:
    output_path, manifest_path = normalized_builder.output_paths_for_date(
        run_root / "training_features_overnight_normalized", day_token(day)
    )
    return _existing_paths(output_path, manifest_path)


def normalized_dependencies_current(run_root: pathlib.Path, day: DayWindow) -> bool:
    outputs = normalized_output_paths(run_root, day)
    if not outputs:
        return False
    dependencies = merged_output_paths(run_root, day)
    if not dependencies:
        return False
    return _max_mtime(outputs) >= _max_mtime(dependencies)


def ensure_normalized_day(args: argparse.Namespace, day: DayWindow, *, force: bool) -> None:
    if not force and validate_normalized_day(args.run_root, day, allow_empty=False) and normalized_dependencies_current(args.run_root, day):
        print(f"[skip] normalized date={day_token(day)}")
        return
    delete_path(args.run_root / "training_features_overnight_normalized" / f"target_date_local={day_token(day)}")
    run_command(
        [
            sys.executable,
            "tools/weather/build_training_features_overnight_normalized.py",
            "--input-root",
            str(args.run_root / "training_features_overnight"),
            "--output-dir",
            str(args.run_root / "training_features_overnight_normalized"),
            "--start-local-date",
            day_token(day),
            "--end-local-date",
            day_token(day),
        ]
    )
    if not validate_normalized_day(args.run_root, day, allow_empty=False):
        raise ValueError(f"Normalized validation failed for target_date_local={day_token(day)}")
    print(f"[done] normalized date={day_token(day)}")


def selected_stages(args: argparse.Namespace) -> set[str]:
    return set(args.stages)


def force_rebuild(args: argparse.Namespace, *, stage: str, target_date_local: str) -> bool:
    return stage in set(args.force_rebuild_stage) or target_date_local in set(args.force_rebuild_date)


def stage_handlers() -> dict[str, Callable[[argparse.Namespace, DayWindow, bool], None]]:
    return {
        "wu": ensure_wu_day,
        "nbm": ensure_nbm_day,
        "lamp": ensure_lamp_day,
        "hrrr": ensure_hrrr_day,
        "merged": ensure_merged_day,
        "normalized": ensure_normalized_day,
    }


def main() -> int:
    args = parse_args()
    start_date = parse_local_date(args.start_local_date)
    end_date = parse_local_date(args.end_local_date)
    if end_date < start_date:
        raise ValueError("--end-local-date must be on or after --start-local-date")
    args.run_root.mkdir(parents=True, exist_ok=True)
    clear_temp_files(args.run_root)

    handlers = stage_handlers()
    enabled_stages = selected_stages(args)
    days = iter_days(start_date, end_date)
    for day in days:
        if pause_requested(args.run_root):
            print(f"[paused] target_date_local={day_token(day)} sentinel={args.run_root / PAUSE_SENTINEL}")
            return 0
        for stage in DEFAULT_STAGE_ORDER:
            if stage not in enabled_stages:
                continue
            handlers[stage](args, day, force=force_rebuild(args, stage=stage, target_date_local=day_token(day)))

    print(f"[done] start={start_date.isoformat()} end={end_date.isoformat()} run_root={args.run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
