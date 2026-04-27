#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import signal
import subprocess
import sys
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


DEFAULT_PROBE_TIMEOUT_SECONDS = 900
DEFAULT_SHORT_WINDOW_START = "2025-04-11"
DEFAULT_SHORT_WINDOW_END = "2025-04-13"
DEFAULT_VOCAB_PATH = REPO_ROOT / "tools" / "weather" / "training_feature_vocabularies.json"
LAMP_OVERNIGHT_ARCHIVE_CYCLES = ("0230", "0330", "0430")
STATUS_PASS = "pass"
STATUS_DEGRADED = "degraded"
STATUS_FAIL = "fail"

NETWORK_ERROR_TOKENS = (
    "404",
    "connection error",
    "connectionerror",
    "dns",
    "failed to resolve",
    "name resolution",
    "network is unreachable",
    "nodename nor servname provided",
    "source unavailable",
    "temporary failure in name resolution",
    "timed out while resolving",
    "upstream",
    "urlopen error",
)
ENVIRONMENT_ERROR_TOKENS = (
    "wgrib2",
    "no module named",
    "modulenotfounderror",
    "permission denied",
    "read-only file system",
)
INVENTORY_ERROR_TOKENS = (
    "apcp",
    "ambiguous",
    "inventory",
    "multiple records",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged server-oriented overnight source builds.")
    parser.add_argument("--stage", choices=["wu", "smoke", "short-window", "all"], required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--history-dir", type=pathlib.Path, required=True)
    parser.add_argument("--short-window-start", default=DEFAULT_SHORT_WINDOW_START)
    parser.add_argument("--short-window-end", default=DEFAULT_SHORT_WINDOW_END)
    parser.add_argument("--probe-timeout-seconds", type=int, default=DEFAULT_PROBE_TIMEOUT_SECONDS)
    parser.add_argument("--nbm-workers", type=int, default=1)
    parser.add_argument("--hrrr-workers", type=int, default=1)
    parser.add_argument("--allow-degraded-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True))


def load_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def stage_path(output_root: pathlib.Path, stage_name: str) -> pathlib.Path:
    return output_root / f"stage_{stage_name.replace('-', '_')}.json"


def run_summary_path(output_root: pathlib.Path) -> pathlib.Path:
    return output_root / "run_summary.json"


def run_command(command: list[str], *, cwd: pathlib.Path, timeout_seconds: int | None = None) -> dict[str, Any]:
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
            "stdout": f"{exc.stdout or ''}{stdout or ''}"[-4000:],
            "stderr": f"{exc.stderr or ''}{stderr or ''}"[-4000:],
            "timed_out": True,
        }
    return {
        "command": command,
        "returncode": process.returncode,
        "stdout": (stdout or "")[-4000:],
        "stderr": (stderr or "")[-4000:],
        "timed_out": False,
    }


def summarize_command_issue(result: dict[str, Any]) -> tuple[str, str | None]:
    if not result.get("timed_out") and result.get("returncode") == 0:
        return STATUS_PASS, None
    combined = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr"))
    lowered = combined.lower()
    if result.get("timed_out"):
        return STATUS_DEGRADED, "performance_blocker"
    if any(token in lowered for token in ENVIRONMENT_ERROR_TOKENS):
        return STATUS_FAIL, "environment_blocker"
    if any(token in lowered for token in INVENTORY_ERROR_TOKENS):
        return STATUS_FAIL, "inventory_mapping_bug"
    if any(token in lowered for token in NETWORK_ERROR_TOKENS):
        return STATUS_DEGRADED, "network_or_upstream_degraded"
    return STATUS_FAIL, "contract_bug"


def stage_result(
    stage_name: str,
    *,
    status: str,
    started_at_utc: str,
    details: dict[str, Any] | None = None,
    blocking: bool | None = None,
    reused: bool = False,
) -> dict[str, Any]:
    return {
        "stage": stage_name,
        "status": status,
        "blocking": status == STATUS_FAIL if blocking is None else blocking,
        "reused": reused,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now_iso(),
        "details": details or {},
    }


def aggregate_stage_status(statuses: list[str]) -> str:
    if any(status == STATUS_FAIL for status in statuses):
        return STATUS_FAIL
    if any(status == STATUS_DEGRADED for status in statuses):
        return STATUS_DEGRADED
    return STATUS_PASS


def update_run_summary(output_root: pathlib.Path, *, requested_stage: str) -> dict[str, Any]:
    summary = {
        "requested_stage": requested_stage,
        "updated_at_utc": utc_now_iso(),
        "stages": {},
    }
    statuses: list[str] = []
    for stage_name in ("wu", "smoke", "short-window"):
        payload = load_json(stage_path(output_root, stage_name))
        if payload is None:
            continue
        summary["stages"][stage_name] = payload
        statuses.append(str(payload.get("status")))
    summary["overall_status"] = aggregate_stage_status(statuses) if statuses else "not_run"
    write_json(run_summary_path(output_root), summary)
    return summary


def reuse_prior_stage(output_root: pathlib.Path, *, stage_name: str, acceptable_statuses: set[str], resume: bool) -> dict[str, Any] | None:
    if not resume:
        return None
    prior = load_json(stage_path(output_root, stage_name))
    if prior is None:
        return None
    if str(prior.get("status")) not in acceptable_statuses:
        return None
    reused = dict(prior)
    reused["reused"] = True
    return reused


def read_required_parquet(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_named_parquets(root: pathlib.Path, pattern: str) -> pd.DataFrame:
    if root.is_file():
        return pd.read_parquet(root)
    if not root.exists():
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in sorted(root.rglob(pattern))]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def filter_window(df: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty or "target_date_local" not in df.columns:
        return df.copy()
    return df.loc[(df["target_date_local"] >= start_date) & (df["target_date_local"] <= end_date)].reset_index(drop=True)


def write_review_artifacts(
    output_dir: pathlib.Path,
    *,
    merged_df: pd.DataFrame,
    merged_summary: dict[str, Any],
    merged_metrics: pd.DataFrame,
    normalized_df: pd.DataFrame,
    normalized_summary: dict[str, Any],
    normalized_metrics: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_path = output_dir / "training_features_overnight.parquet"
    normalized_path = output_dir / "training_features_overnight_normalized.parquet"
    merged_summary_path = output_dir / "training_features_overnight.audit.json"
    merged_metrics_path = output_dir / "training_features_overnight.column_metrics.csv"
    normalized_summary_path = output_dir / "training_features_overnight_normalized.audit.json"
    normalized_metrics_path = output_dir / "training_features_overnight_normalized.column_metrics.csv"
    merged_df.to_parquet(merged_path, index=False)
    normalized_df.to_parquet(normalized_path, index=False)
    write_json(merged_summary_path, merged_summary)
    merged_metrics.to_csv(merged_metrics_path, index=False)
    write_json(normalized_summary_path, normalized_summary)
    normalized_metrics.to_csv(normalized_metrics_path, index=False)
    return {
        "merged_path": str(merged_path),
        "normalized_path": str(normalized_path),
        "merged_audit_path": str(merged_summary_path),
        "normalized_audit_path": str(normalized_summary_path),
    }


def build_short_window_review(
    *,
    labels_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    nbm_df: pd.DataFrame,
    lamp_df: pd.DataFrame,
    hrrr_df: pd.DataFrame,
    vocabularies: dict[str, Any],
    output_dir: pathlib.Path,
    label_history_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    merged_df = build_training_features_overnight(
        labels_df=labels_df,
        label_history_df=label_history_df,
        obs_df=obs_df,
        nbm_daily_df=nbm_df,
        lamp_daily_df=lamp_df,
        hrrr_daily_df=hrrr_df,
        cutoff_local_time="00:05",
        station_id="KLGA",
    )
    merged_summary, merged_metrics = build_merged_audit(merged_df)
    normalized_df = normalize_training_features_overnight(merged_df, vocabularies)
    normalized_summary, normalized_metrics = build_normalized_audit(normalized_df)
    artifacts = write_review_artifacts(
        output_dir,
        merged_df=merged_df,
        merged_summary=merged_summary,
        merged_metrics=merged_metrics,
        normalized_df=normalized_df,
        normalized_summary=normalized_summary,
        normalized_metrics=normalized_metrics,
    )
    expected_rows = int(len(labels_df))
    return {
        "expected_label_rows": expected_rows,
        "source_row_count": int(len(nbm_df) + len(lamp_df) + len(hrrr_df)),
        "merged_row_count": int(len(merged_df)),
        "normalized_row_count": int(len(normalized_df)),
        "merged_checks": merged_summary["checks"],
        "normalized_checks": normalized_summary["checks"],
        "artifacts": artifacts,
        "label_spine_preserved": int(len(merged_df)) == expected_rows,
    }


def source_review_ok(review: dict[str, Any]) -> bool:
    merged_checks = review.get("merged_checks", {})
    normalized_checks = review.get("normalized_checks", {})
    return bool(review.get("label_spine_preserved")) and all(bool(value) for value in merged_checks.values()) and all(bool(value) for value in normalized_checks.values())


def classify_probe_entry(entry: dict[str, Any]) -> tuple[str, str | None]:
    steps = entry.get("steps", [])
    timeout_count = sum(1 for step in steps if step.get("timed_out"))
    if timeout_count > 1:
        return STATUS_FAIL, "performance_blocker"
    for step in steps:
        status, issue_class = summarize_command_issue(step)
        if status == STATUS_FAIL:
            return status, issue_class
        if status == STATUS_DEGRADED:
            review = entry.get("merge_summary", {})
            checks_ok = all(bool(value) for value in review.get("merged_checks", {}).values()) and all(
                bool(value) for value in review.get("normalized_checks", {}).values()
            )
            preserved = int(review.get("merged_row_count", 0)) > 0 and int(review.get("normalized_row_count", 0)) > 0
            return (STATUS_DEGRADED, issue_class) if checks_ok and preserved else (STATUS_FAIL, "contract_bug")
    review = entry.get("merge_summary", {})
    if int(review.get("merged_row_count", 0)) == 0 or int(review.get("normalized_row_count", 0)) == 0:
        return STATUS_FAIL, "contract_bug"
    if not all(bool(value) for value in review.get("merged_checks", {}).values()):
        return STATUS_FAIL, "contract_bug"
    if not all(bool(value) for value in review.get("normalized_checks", {}).values()):
        return STATUS_FAIL, "contract_bug"
    return STATUS_PASS, None


def run_wu_stage(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now_iso()
    output_dir = args.output_root / "wu_tables"
    validate_result = run_command(
        [sys.executable, "wunderground/validate_history.py", "--history-dir", str(args.history_dir)],
        cwd=REPO_ROOT,
    )
    validate_status, validate_issue_class = summarize_command_issue(validate_result)
    if validate_status != STATUS_PASS:
        return stage_result(
            "wu",
            status=STATUS_FAIL,
            started_at_utc=started_at,
            details={"validate_history": validate_result, "issue_class": validate_issue_class},
        )

    build_result = run_command(
        [
            sys.executable,
            "wunderground/build_training_tables.py",
            "--history-dir",
            str(args.history_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
    )
    build_status, build_issue_class = summarize_command_issue(build_result)
    labels_path = output_dir / "labels_daily.parquet"
    obs_path = output_dir / "wu_obs_intraday.parquet"
    labels_df = read_required_parquet(labels_path)
    obs_df = read_required_parquet(obs_path)

    checks = {
        "labels_written": labels_path.exists(),
        "obs_written": obs_path.exists(),
        "labels_non_empty": not labels_df.empty,
        "obs_non_empty": not obs_df.empty,
        "labels_unique_target_date_local": bool(not labels_df.empty and not labels_df.duplicated(subset=["target_date_local"]).any()),
    }
    status = STATUS_PASS if build_status == STATUS_PASS and all(checks.values()) else STATUS_FAIL
    details = {
        "validate_history": validate_result,
        "build_tables": build_result,
        "issue_class": build_issue_class,
        "checks": checks,
        "artifacts": {"labels_path": str(labels_path), "obs_path": str(obs_path)},
        "label_row_count": int(len(labels_df)),
        "obs_row_count": int(len(obs_df)),
    }
    return stage_result("wu", status=status, started_at_utc=started_at, details=details)


def run_smoke_stage(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now_iso()
    verification_dir = args.output_root / "verification"
    live_probe_dir = args.output_root / "live_probes"
    verification_result = run_command(
        [
            sys.executable,
            "tools/weather/run_verification_suite.py",
            "--output-dir",
            str(verification_dir),
            "--local-only",
        ],
        cwd=REPO_ROOT,
    )
    verification_summary = load_json(verification_dir / "summary.json") or {}
    local_lane_status = verification_summary.get("local_contract_lane", {}).get("status")
    verification_ok = local_lane_status == STATUS_PASS

    probe_result = run_command(
        [
            sys.executable,
            "tools/weather/run_live_overnight_probes.py",
            "--output-dir",
            str(live_probe_dir),
            "--history-dir",
            str(args.history_dir),
            "--timeout-seconds",
            str(args.probe_timeout_seconds),
            "--lamp-count",
            "1",
            "--nbm-count",
            "1",
            "--hrrr-count",
            "1",
        ],
        cwd=REPO_ROOT,
        timeout_seconds=args.probe_timeout_seconds * 3,
    )
    probe_report = load_json(live_probe_dir / "live_probe_report.json") or {}
    source_results: dict[str, dict[str, Any]] = {}
    aggregate_statuses = [STATUS_PASS if verification_ok else STATUS_FAIL]
    for source_name in ("lamp", "nbm", "hrrr"):
        entries = probe_report.get("results", {}).get(source_name, [])
        statuses: list[str] = []
        issue_classes: list[str] = []
        classified_entries: list[dict[str, Any]] = []
        for entry in entries:
            status, issue_class = classify_probe_entry(entry)
            statuses.append(status)
            if issue_class:
                issue_classes.append(issue_class)
            classified_entries.append({"target_date_local": entry.get("target_date_local"), "status": status, "issue_class": issue_class})
        source_status = aggregate_stage_status(statuses) if statuses else STATUS_FAIL
        source_results[source_name] = {
            "status": source_status,
            "issue_classes": sorted(set(issue_classes)),
            "entries": classified_entries,
        }
        aggregate_statuses.append(source_status)

    stage_status = aggregate_stage_status(aggregate_statuses)
    details = {
        "verification": {
            "command": verification_result,
            "summary_path": str(verification_dir / "summary.json"),
            "local_contract_lane_status": local_lane_status,
        },
        "live_probes": {
            "command": probe_result,
            "report_path": str(live_probe_dir / "live_probe_report.json"),
            "sources": source_results,
        },
    }
    if not verification_ok:
        details["issue_class"] = "contract_bug"
    return stage_result("smoke", status=stage_status, started_at_utc=started_at, details=details)


def source_stage_config(args: argparse.Namespace) -> list[dict[str, Any]]:
    lamp_fetch_command = [
        sys.executable,
        "tools/lamp/fetch_lamp.py",
        "archive",
        "--start-utc-date",
        args.short_window_start,
        "--end-utc-date",
        args.short_window_end,
    ]
    for cycle in LAMP_OVERNIGHT_ARCHIVE_CYCLES:
        lamp_fetch_command.extend(["--cycle", cycle])
    lamp_fetch_command.extend(
        [
            "--output-dir",
            str(args.output_root / "lamp" / "raw"),
            "--overwrite",
        ]
    )
    return [
        {
            "name": "nbm",
            "commands": [
                [
                    sys.executable,
                    "tools/nbm/build_grib2_features.py",
                    "--start-local-date",
                    args.short_window_start,
                    "--end-local-date",
                    args.short_window_end,
                    "--output-dir",
                    str(args.output_root / "nbm" / "features"),
                    "--workers",
                    str(args.nbm_workers),
                    "--overwrite",
                ],
                [
                    sys.executable,
                    "tools/nbm/build_nbm_overnight_features.py",
                    "--features-root",
                    str(args.output_root / "nbm" / "features"),
                    "--output-dir",
                    str(args.output_root / "nbm" / "overnight"),
                    "--start-local-date",
                    args.short_window_start,
                    "--end-local-date",
                    args.short_window_end,
                ],
            ],
            "loader": lambda: load_named_parquets(args.output_root / "nbm" / "overnight", "nbm.overnight.parquet"),
            "review_frames": lambda df: {"nbm_df": df, "lamp_df": pd.DataFrame(), "hrrr_df": pd.DataFrame()},
        },
        {
            "name": "lamp",
            "commands": [
                lamp_fetch_command,
                [
                    sys.executable,
                    "tools/lamp/build_lamp_klga_features.py",
                    str(args.output_root / "lamp" / "raw"),
                    "--output-dir",
                    str(args.output_root / "lamp" / "features"),
                ],
                [
                    sys.executable,
                    "tools/lamp/build_lamp_overnight_features.py",
                    "--features-root",
                    str(args.output_root / "lamp" / "features"),
                    "--output-dir",
                    str(args.output_root / "lamp" / "overnight"),
                    "--start-local-date",
                    args.short_window_start,
                    "--end-local-date",
                    args.short_window_end,
                ],
            ],
            "loader": lambda: load_named_parquets(args.output_root / "lamp" / "overnight", "lamp.overnight.parquet"),
            "review_frames": lambda df: {"nbm_df": pd.DataFrame(), "lamp_df": df, "hrrr_df": pd.DataFrame()},
        },
        {
            "name": "hrrr",
            "commands": [
                [
                    sys.executable,
                    "tools/hrrr/build_hrrr_klga_feature_shards.py",
                    "--start-date",
                    args.short_window_start,
                    "--end-date",
                    args.short_window_end,
                    "--download-dir",
                    str(args.output_root / "hrrr" / "downloads"),
                    "--reduced-dir",
                    str(args.output_root / "hrrr" / "reduced"),
                    "--output-dir",
                    str(args.output_root / "hrrr" / "features"),
                    "--summary-output-dir",
                    str(args.output_root / "hrrr" / "summary"),
                    "--max-workers",
                    str(args.hrrr_workers),
                    "--allow-partial",
                ],
            ],
            "loader": lambda: filter_window(load_named_parquets(args.output_root / "hrrr" / "summary", "*.parquet"), start_date=args.short_window_start, end_date=args.short_window_end),
            "review_frames": lambda df: {"nbm_df": pd.DataFrame(), "lamp_df": pd.DataFrame(), "hrrr_df": df},
        },
    ]


def run_short_window_stage(args: argparse.Namespace) -> dict[str, Any]:
    started_at = utc_now_iso()
    wu_labels = read_required_parquet(args.output_root / "wu_tables" / "labels_daily.parquet")
    wu_obs = read_required_parquet(args.output_root / "wu_tables" / "wu_obs_intraday.parquet")
    window_labels = filter_window(wu_labels, start_date=args.short_window_start, end_date=args.short_window_end)
    vocabularies = load_vocabularies(DEFAULT_VOCAB_PATH)
    source_summaries: list[dict[str, Any]] = []
    statuses: list[str] = []

    for config in source_stage_config(args):
        command_results: list[dict[str, Any]] = []
        command_statuses: list[str] = []
        issue_classes: list[str] = []
        for command in config["commands"]:
            result = run_command(command, cwd=REPO_ROOT, timeout_seconds=args.probe_timeout_seconds)
            status, issue_class = summarize_command_issue(result)
            command_results.append(result)
            command_statuses.append(status)
            if issue_class:
                issue_classes.append(issue_class)
            if status == STATUS_FAIL:
                break
        source_df = config["loader"]()
        review_dir = args.output_root / config["name"] / "review"
        review = build_short_window_review(
            labels_df=window_labels,
            label_history_df=wu_labels,
            obs_df=wu_obs,
            vocabularies=vocabularies,
            output_dir=review_dir,
            **config["review_frames"](source_df),
        )
        review_ok = source_review_ok(review)
        produced_rows = int(len(source_df)) > 0
        command_status = aggregate_stage_status(command_statuses) if command_statuses else STATUS_FAIL
        if command_status == STATUS_FAIL:
            source_status = STATUS_FAIL
            if not issue_classes:
                issue_classes.append("contract_bug")
        elif command_status == STATUS_DEGRADED:
            source_status = STATUS_DEGRADED if review_ok else STATUS_FAIL
        else:
            source_status = STATUS_PASS if produced_rows and review_ok else (STATUS_DEGRADED if review_ok else STATUS_FAIL)
            if source_status == STATUS_DEGRADED and not issue_classes:
                issue_classes.append("network_or_upstream_degraded")
        statuses.append(source_status)
        source_summaries.append(
            {
                "source": config["name"],
                "status": source_status,
                "issue_classes": sorted(set(issue_classes)),
                "commands": command_results,
                "review": review,
                "source_row_count": int(len(source_df)),
            }
        )

    details = {
        "short_window": {
            "start": args.short_window_start,
            "end": args.short_window_end,
            "label_row_count": int(len(window_labels)),
        },
        "sources": source_summaries,
    }
    return stage_result("short-window", status=aggregate_stage_status(statuses), started_at_utc=started_at, details=details)


def gate_or_fail(required_stage_name: str, *, current_stage_name: str, output_root: pathlib.Path, accepted_statuses: set[str]) -> dict[str, Any] | None:
    prior = load_json(stage_path(output_root, required_stage_name))
    if prior is None:
        return stage_result(
            current_stage_name,
            status=STATUS_FAIL,
            started_at_utc=utc_now_iso(),
            details={"issue_class": "contract_bug", "error": f"required prerequisite stage {required_stage_name} has not been run"},
        )
    if str(prior.get("status")) not in accepted_statuses:
        return stage_result(
            current_stage_name,
            status=STATUS_FAIL,
            started_at_utc=utc_now_iso(),
            details={"issue_class": "contract_bug", "error": f"required prerequisite stage {required_stage_name} status={prior.get('status')}"},
        )
    return None


def execute_requested_stages(args: argparse.Namespace) -> dict[str, Any]:
    args.output_root.mkdir(parents=True, exist_ok=True)
    requested = (
        ["wu", "smoke", "short-window"]
        if args.stage == "all"
        else [args.stage]
    )
    latest: dict[str, Any] | None = None

    for stage_name in requested:
        if stage_name == "wu":
            reused = reuse_prior_stage(args.output_root, stage_name="wu", acceptable_statuses={STATUS_PASS}, resume=args.resume)
            latest = reused if reused is not None else run_wu_stage(args)
        elif stage_name == "smoke":
            gating = gate_or_fail("wu", current_stage_name="smoke", output_root=args.output_root, accepted_statuses={STATUS_PASS})
            if gating is not None:
                latest = gating
            else:
                reused = reuse_prior_stage(args.output_root, stage_name="smoke", acceptable_statuses={STATUS_PASS, STATUS_DEGRADED}, resume=args.resume)
                latest = reused if reused is not None else run_smoke_stage(args)
        else:
            wu_gate = gate_or_fail("wu", current_stage_name="short-window", output_root=args.output_root, accepted_statuses={STATUS_PASS})
            smoke_gate = gate_or_fail("smoke", current_stage_name="short-window", output_root=args.output_root, accepted_statuses={STATUS_PASS, STATUS_DEGRADED})
            if wu_gate is not None:
                latest = wu_gate
            elif smoke_gate is not None:
                latest = smoke_gate
            else:
                reused = reuse_prior_stage(args.output_root, stage_name="short-window", acceptable_statuses={STATUS_PASS, STATUS_DEGRADED}, resume=args.resume)
                latest = reused if reused is not None else run_short_window_stage(args)
        assert latest is not None
        write_json(stage_path(args.output_root, stage_name), latest)
        update_run_summary(args.output_root, requested_stage=args.stage)
        if latest["blocking"]:
            break

    return update_run_summary(args.output_root, requested_stage=args.stage)


def exit_code_for_summary(summary: dict[str, Any], *, allow_degraded_live: bool) -> int:
    overall_status = str(summary.get("overall_status"))
    if overall_status == STATUS_FAIL:
        return 1
    if overall_status == STATUS_DEGRADED and not allow_degraded_live:
        return 2
    return 0


def main() -> int:
    args = parse_args()
    summary = execute_requested_stages(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return exit_code_for_summary(summary, allow_degraded_live=args.allow_degraded_live)


if __name__ == "__main__":
    raise SystemExit(main())
