#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import signal
import subprocess
import sys
import tempfile
from typing import Any

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.audit_training_features_overnight import build_audit as build_merged_audit
from tools.weather.audit_training_features_overnight_normalized import build_audit as build_normalized_audit
from tools.weather.build_training_features_overnight import build_training_features_overnight
from tools.weather.build_training_features_overnight_normalized import load_vocabularies, normalize_training_features_overnight
from wunderground.build_training_tables import build_training_tables


DEFAULT_HISTORY_DIR = REPO_ROOT / "wunderground" / "output" / "history"
DEFAULT_VOCAB_PATH = REPO_ROOT / "tools" / "weather" / "training_feature_vocabularies.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tools" / "weather" / "data" / "runtime" / "live_probes"
DEFAULT_RANDOM_SEED = 19
DEFAULT_TIMEOUT_SECONDS = 900


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run serialized random overnight-source probes for LAMP, NBM, and HRRR.")
    parser.add_argument("--history-dir", type=pathlib.Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--vocab-path", type=pathlib.Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--lamp-count", type=int, default=3)
    parser.add_argument("--nbm-count", type=int, default=3)
    parser.add_argument("--hrrr-count", type=int, default=2)
    return parser.parse_args()


def run_command(command: list[str], *, cwd: pathlib.Path, timeout_seconds: int) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return {
            "command": command,
            "returncode": None,
            "stdout": (_text(exc.stdout) + _text(stdout))[-4000:],
            "stderr": (_text(exc.stderr) + _text(stderr))[-4000:],
            "timed_out": True,
        }
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": (stdout or "")[-4000:],
        "stderr": (stderr or "")[-4000:],
        "timed_out": False,
    }


def choose_probe_dates(*, random_seed: int, lamp_count: int, nbm_count: int, hrrr_count: int) -> dict[str, list[str]]:
    rng = random.Random(random_seed)
    candidates = {
        "warm": ["2025-06-12", "2025-07-08", "2025-08-01"],
        "cold": ["2025-01-15", "2025-02-12", "2026-01-08"],
        "shoulder": ["2025-04-11", "2025-10-03", "2026-03-18"],
        "dst_adjacent": ["2025-03-09", "2025-11-02", "2026-03-08"],
    }
    return {
        "lamp": rng.sample(["2026-04-09", "2026-04-10", "2026-04-11"], k=min(lamp_count, 3)),
        "nbm": [rng.choice(candidates["warm"]), rng.choice(candidates["cold"]), rng.choice(candidates["dst_adjacent"])][:nbm_count],
        "hrrr": [rng.choice(candidates["shoulder"]), rng.choice(candidates["dst_adjacent"])][:hrrr_count],
    }


def load_single_parquet(root: pathlib.Path, pattern: str) -> pd.DataFrame:
    matches = sorted(root.rglob(pattern))
    frames = [pd.read_parquet(path) for path in matches if path.exists()]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_source_row(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"status": "missing_output"}
    row = df.iloc[0].to_dict()
    keep = {
        key: row.get(key)
        for key in (
            "target_date_local",
            "selected_init_time_utc",
            "selected_init_time_local",
            "previous_init_time_utc",
            "coverage_complete",
            "missing_checkpoint_count",
            "target_day_row_count",
            "selected_issue_age_minutes",
            "retained_cycle_count",
            "coverage_end_hour_local",
            "has_full_day_21_local_coverage",
        )
        if key in row
    }
    keep["status"] = "ok"
    return keep


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, pathlib.Path):
        return str(value)
    return value


def merged_probe_summary(*, labels_df: pd.DataFrame, obs_df: pd.DataFrame, vocabularies: dict[str, Any], target_date_local: str, nbm_df: pd.DataFrame, lamp_df: pd.DataFrame, hrrr_df: pd.DataFrame) -> dict[str, Any]:
    date_labels = labels_df.loc[labels_df["target_date_local"] == target_date_local].reset_index(drop=True)
    merged_df = build_training_features_overnight(
        labels_df=date_labels,
        obs_df=obs_df,
        nbm_daily_df=nbm_df,
        lamp_daily_df=lamp_df,
        hrrr_daily_df=hrrr_df,
        cutoff_local_time="00:05",
        station_id="KLGA",
    )
    merged_summary, _ = build_merged_audit(merged_df)
    normalized_df = normalize_training_features_overnight(merged_df, vocabularies)
    normalized_summary, _ = build_normalized_audit(normalized_df)
    return {
        "merged_row_count": int(len(merged_df)),
        "merged_checks": merged_summary["checks"],
        "normalized_row_count": int(len(normalized_df)),
        "normalized_checks": normalized_summary["checks"],
    }


def probe_lamp(date_local: str, *, root: pathlib.Path, labels_df: pd.DataFrame, obs_df: pd.DataFrame, vocabularies: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    date_utc = date_local
    raw_dir = root / "lamp_raw"
    features_dir = root / "lamp_features"
    overnight_dir = root / "lamp_overnight"
    steps = [
        run_command(["python3", "tools/lamp/fetch_lamp.py", "live", "--date-utc", date_utc, "--cycle", "0330", "--output-dir", str(raw_dir), "--overwrite"], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
        run_command(["python3", "tools/lamp/fetch_lamp.py", "live", "--date-utc", date_utc, "--cycle", "0430", "--output-dir", str(raw_dir), "--overwrite"], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
        run_command(["python3", "tools/lamp/build_lamp_klga_features.py", str(raw_dir), "--output-dir", str(features_dir)], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
        run_command(["python3", "tools/lamp/build_lamp_overnight_features.py", "--features-root", str(features_dir), "--output-dir", str(overnight_dir), "--start-local-date", date_local, "--end-local-date", date_local], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
    ]
    lamp_df = load_single_parquet(overnight_dir, "lamp.overnight.parquet")
    return {
        "steps": steps,
        "source_summary": summarize_source_row(lamp_df),
        "merge_summary": merged_probe_summary(labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, target_date_local=date_local, nbm_df=pd.DataFrame(), lamp_df=lamp_df, hrrr_df=pd.DataFrame()),
    }


def probe_nbm(date_local: str, *, root: pathlib.Path, labels_df: pd.DataFrame, obs_df: pd.DataFrame, vocabularies: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    features_dir = root / "nbm_features"
    overnight_dir = root / "nbm_overnight"
    steps = [
        run_command(["python3", "tools/nbm/build_grib2_features.py", "--start-local-date", date_local, "--end-local-date", date_local, "--output-dir", str(features_dir), "--workers", "1", "--overwrite"], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
        run_command(["python3", "tools/nbm/build_nbm_overnight_features.py", "--features-root", str(features_dir), "--output-dir", str(overnight_dir), "--start-local-date", date_local, "--end-local-date", date_local], cwd=REPO_ROOT, timeout_seconds=timeout_seconds),
    ]
    nbm_df = load_single_parquet(overnight_dir, "nbm.overnight.parquet")
    return {
        "steps": steps,
        "source_summary": summarize_source_row(nbm_df),
        "merge_summary": merged_probe_summary(labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, target_date_local=date_local, nbm_df=nbm_df, lamp_df=pd.DataFrame(), hrrr_df=pd.DataFrame()),
    }


def probe_hrrr(date_local: str, *, root: pathlib.Path, labels_df: pd.DataFrame, obs_df: pd.DataFrame, vocabularies: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    download_dir = root / "hrrr_downloads"
    reduced_dir = root / "hrrr_reduced"
    output_dir = root / "hrrr_features"
    summary_dir = root / "hrrr_summary"
    steps = [
        run_command(
            [
                "python3",
                "tools/hrrr/build_hrrr_klga_feature_shards.py",
                "--start-date",
                date_local,
                "--end-date",
                date_local,
                "--download-dir",
                str(download_dir),
                "--reduced-dir",
                str(reduced_dir),
                "--output-dir",
                str(output_dir),
                "--summary-output-dir",
                str(summary_dir),
                "--max-workers",
                "1",
                "--allow-partial",
            ],
            cwd=REPO_ROOT,
            timeout_seconds=timeout_seconds,
        ),
    ]
    hrrr_df = load_single_parquet(summary_dir, "*.parquet")
    hrrr_df = hrrr_df.loc[hrrr_df["target_date_local"] == date_local].reset_index(drop=True) if not hrrr_df.empty and "target_date_local" in hrrr_df.columns else hrrr_df
    return {
        "steps": steps,
        "source_summary": summarize_source_row(hrrr_df),
        "merge_summary": merged_probe_summary(labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, target_date_local=date_local, nbm_df=pd.DataFrame(), lamp_df=pd.DataFrame(), hrrr_df=hrrr_df),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels_df, obs_df = build_training_tables(args.history_dir)
    vocabularies = load_vocabularies(args.vocab_path)
    selected_dates = choose_probe_dates(
        random_seed=args.random_seed,
        lamp_count=args.lamp_count,
        nbm_count=args.nbm_count,
        hrrr_count=args.hrrr_count,
    )

    report: dict[str, Any] = {"selected_dates": selected_dates, "results": {"lamp": [], "nbm": [], "hrrr": []}}

    with tempfile.TemporaryDirectory(prefix="overnight-live-probes-") as temp_root:
        temp_root_path = pathlib.Path(temp_root)
        for date_local in selected_dates["lamp"]:
            report["results"]["lamp"].append({"target_date_local": date_local, **probe_lamp(date_local, root=temp_root_path / f"lamp_{date_local}", labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, timeout_seconds=args.timeout_seconds)})
        for date_local in selected_dates["nbm"]:
            report["results"]["nbm"].append({"target_date_local": date_local, **probe_nbm(date_local, root=temp_root_path / f"nbm_{date_local}", labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, timeout_seconds=args.timeout_seconds)})
        for date_local in selected_dates["hrrr"]:
            report["results"]["hrrr"].append({"target_date_local": date_local, **probe_hrrr(date_local, root=temp_root_path / f"hrrr_{date_local}", labels_df=labels_df, obs_df=obs_df, vocabularies=vocabularies, timeout_seconds=args.timeout_seconds)})

    output_path = args.output_dir / "live_probe_report.json"
    output_path.write_text(json.dumps(to_jsonable(report), indent=2, sort_keys=True))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
