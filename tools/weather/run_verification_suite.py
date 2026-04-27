#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
HRRR_DIR = REPO_ROOT / "tools" / "hrrr"
NBM_DIR = REPO_ROOT / "tools" / "nbm"
DEFAULT_OUTPUT_DIR = pathlib.Path(tempfile.gettempdir()) / "weather-verification"
DEFAULT_HRRR_DATE = "2025-04-11"
DEFAULT_HRRR_CYCLE = 12
DEFAULT_HRRR_FORECAST_HOUR = 0
DEFAULT_NBM_DATE = "2025-04-11"
DEFAULT_NBM_CYCLE = "04"
DEFAULT_NBM_FORECAST_HOUR = 1
PYTEST_TARGETS = ["tests"]
NETWORK_ERROR_TOKENS = (
    "nodename nor servname provided",
    "name resolution",
    "failed to resolve",
    "temporary failure in name resolution",
    "max retries exceeded",
    "connectionerror",
    "urlopen error",
)
WGRIB2_ENV_VAR = "WGRIB2_BINARY"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HRRR_DIR) not in sys.path:
    sys.path.insert(0, str(HRRR_DIR))
if str(NBM_DIR) not in sys.path:
    sys.path.insert(0, str(NBM_DIR))

from tools.weather import standardization_audit as audit
from tools.nbm import build_grib2_features as nbm_pipeline
from tools.nbm.fetch_nbm import S3HttpClient


@dataclasses.dataclass
class StageResult:
    name: str
    status: str
    elapsed_seconds: float
    command: list[str] | None = None
    artifacts: dict[str, str] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)
    error: str | None = None
    stdout: str = ""
    stderr: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full weather verification suite.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    parser.add_argument("--local-only", action="store_true", help="Run only the deterministic local-contract lane.")
    parser.add_argument("--live-only", action="store_true", help="Run only the live-source verification lane.")
    parser.add_argument("--hrrr-date", default=DEFAULT_HRRR_DATE)
    parser.add_argument("--hrrr-cycle", type=int, default=DEFAULT_HRRR_CYCLE)
    parser.add_argument("--hrrr-forecast-hour", type=int, default=DEFAULT_HRRR_FORECAST_HOUR)
    parser.add_argument("--nbm-date", default=DEFAULT_NBM_DATE)
    parser.add_argument("--nbm-cycle", default=DEFAULT_NBM_CYCLE)
    parser.add_argument("--nbm-forecast-hour", type=int, default=DEFAULT_NBM_FORECAST_HOUR)
    return parser.parse_args()


def run_command(command: list[str], *, cwd: pathlib.Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd or REPO_ROOT, capture_output=True, text=True, check=False)


def resolve_wgrib2_binary() -> str | None:
    candidate = shutil.which("wgrib2")
    if candidate:
        return candidate
    env_override_value = os.environ.get(WGRIB2_ENV_VAR)
    if env_override_value:
        env_override = pathlib.Path(env_override_value).expanduser()
        if env_override.exists():
            return str(env_override)
    executable_root = pathlib.Path(sys.executable).resolve().parent
    candidates = (
        executable_root / "wgrib2",
        executable_root.parent / "bin" / "wgrib2",
        pathlib.Path.home() / ".local" / "bin" / "wgrib2",
    )
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def network_error_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in NETWORK_ERROR_TOKENS)


def classify_live_stage_status(*, returncode: int, stdout: str, stderr: str, error: str | None = None) -> str:
    if returncode == 0 and not error:
        return "pass"
    combined = "\n".join(part for part in (stdout, stderr, error) if part)
    if network_error_text(combined):
        return "degraded"
    return "fail"


def lane_status(stages: list[StageResult]) -> str:
    statuses = [stage.status for stage in stages]
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "degraded" for status in statuses):
        return "degraded"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    return "pass"


def overall_status(*, local_status: str, live_status: str) -> str:
    if local_status != "pass":
        return "fail_local"
    if live_status == "pass":
        return "pass_both"
    return "pass_local"


def write_summary(path: pathlib.Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True))


def make_stage(
    name: str,
    start_time: float,
    *,
    status: str,
    command: list[str] | None = None,
    artifacts: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    stdout: str = "",
    stderr: str = "",
) -> StageResult:
    return StageResult(
        name=name,
        status=status,
        elapsed_seconds=round(time.perf_counter() - start_time, 3),
        command=command,
        artifacts=artifacts or {},
        warnings=warnings or [],
        error=error,
        stdout=stdout,
        stderr=stderr,
    )


def run_pytest_stage(output_dir: pathlib.Path) -> StageResult:
    del output_dir
    command = [sys.executable, "-m", "pytest", "-q", *PYTEST_TARGETS]
    start_time = time.perf_counter()
    result = run_command(command)
    return make_stage(
        "pytest_all",
        start_time,
        status="pass" if result.returncode == 0 else "fail",
        command=command,
        stdout=result.stdout,
        stderr=result.stderr,
        error=None if result.returncode == 0 else f"pytest exited with {result.returncode}",
    )


def run_audit_stage(output_dir: pathlib.Path) -> StageResult:
    stage_dir = output_dir / "local" / "audit"
    command = [sys.executable, str(SCRIPT_DIR / "standardization_audit.py"), "--output-dir", str(stage_dir)]
    start_time = time.perf_counter()
    result = run_command(command)
    matrix_path = stage_dir / audit.OUTPUT_MATRIX_NAME
    review_path = stage_dir / audit.OUTPUT_REVIEW_NAME
    error = None
    status = "pass"
    if result.returncode != 0:
        status = "fail"
        error = f"audit exited with {result.returncode}"
    elif not matrix_path.exists() or not review_path.exists():
        status = "fail"
        error = "audit did not write expected outputs"
    else:
        matrix = json.loads(matrix_path.read_text())
        if any(row.get("classification") == "drift" for row in matrix):
            status = "fail"
            error = "audit reported drift"
    return make_stage(
        "audit_smoke",
        start_time,
        status=status,
        command=command,
        artifacts={"matrix": str(matrix_path), "review": str(review_path)},
        error=error,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def write_compare_inputs(output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    compare_dir = output_dir / "local" / "compare_inputs"
    compare_dir.mkdir(parents=True, exist_ok=True)
    nbm_outputs = audit.build_nbm_sample_outputs()
    hrrr_outputs = audit.build_hrrr_sample_outputs()
    paths = {
        "nbm_wide": compare_dir / "nbm_wide.parquet",
        "nbm_provenance": compare_dir / "nbm_provenance.parquet",
        "nbm_manifest": compare_dir / "nbm_manifest.parquet",
        "hrrr_wide": compare_dir / "hrrr_wide.parquet",
        "hrrr_provenance": compare_dir / "hrrr_provenance.parquet",
        "hrrr_manifest": compare_dir / "hrrr_manifest.parquet",
    }
    nbm_outputs["wide_df"].to_parquet(paths["nbm_wide"], index=False)
    nbm_outputs["provenance_df"].to_parquet(paths["nbm_provenance"], index=False)
    pd.DataFrame.from_records([nbm_outputs["manifest_row"]]).to_parquet(paths["nbm_manifest"], index=False)
    hrrr_outputs["wide_df"].to_parquet(paths["hrrr_wide"], index=False)
    hrrr_outputs["provenance_df"].to_parquet(paths["hrrr_provenance"], index=False)
    pd.DataFrame.from_records([hrrr_outputs["manifest_row"]]).to_parquet(paths["hrrr_manifest"], index=False)
    return paths


def run_compare_stage(output_dir: pathlib.Path) -> StageResult:
    compare_inputs = write_compare_inputs(output_dir)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "compare_weather_contracts.py"),
        "--nbm-wide",
        str(compare_inputs["nbm_wide"]),
        "--nbm-provenance",
        str(compare_inputs["nbm_provenance"]),
        "--nbm-manifest",
        str(compare_inputs["nbm_manifest"]),
        "--hrrr-wide",
        str(compare_inputs["hrrr_wide"]),
        "--hrrr-provenance",
        str(compare_inputs["hrrr_provenance"]),
        "--hrrr-manifest",
        str(compare_inputs["hrrr_manifest"]),
        "--format",
        "json",
    ]
    start_time = time.perf_counter()
    result = run_command(command)
    error = None
    status = "pass"
    if result.returncode != 0:
        status = "fail"
        error = f"compare exited with {result.returncode}"
    else:
        report = json.loads(result.stdout)
        if report["verdict"]["overall_status"] != "synced" or not report["verdict"]["raw_runtime_contract_shared"]:
            status = "fail"
            error = "compare reported non-synced verdict"
    return make_stage(
        "compatibility_smoke",
        start_time,
        status=status,
        command=command,
        artifacts={key: str(value) for key, value in compare_inputs.items()},
        error=error,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_local_contract_lane(output_dir: pathlib.Path) -> list[StageResult]:
    return [
        run_pytest_stage(output_dir),
        run_audit_stage(output_dir),
        run_compare_stage(output_dir),
    ]


def run_hrrr_live_download_stage(output_dir: pathlib.Path, *, date: str, cycle: int, forecast_hour: int) -> StageResult:
    stage_dir = output_dir / "live" / "hrrr_download"
    command = [
        sys.executable,
        str(HRRR_DIR / "fetch_hrrr_records.py"),
        "--date",
        date.replace("-", ""),
        "--cycle",
        str(cycle),
        "--product",
        "surface",
        "--forecast-hours",
        str(forecast_hour),
        "--source",
        "aws",
        "--output-dir",
        str(stage_dir),
    ]
    start_time = time.perf_counter()
    result = run_command(command)
    subset_path = stage_dir / f"hrrr.t{cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2.subset.grib2"
    manifest_path = stage_dir / f"hrrr.t{cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2.manifest.csv"
    status = classify_live_stage_status(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    error = None
    if status == "pass":
        if not subset_path.exists() or not manifest_path.exists():
            status = "fail"
            error = "HRRR download stage did not write expected subset artifacts"
        else:
            wgrib2 = resolve_wgrib2_binary()
            if wgrib2 is None:
                status = "fail"
                error = f"wgrib2 is required for verification. Install it on PATH, set {WGRIB2_ENV_VAR}, or use an active environment that provides it."
            else:
                inspect_result = run_command([wgrib2, str(subset_path), "-s"])
                if inspect_result.returncode != 0 or not inspect_result.stdout.strip():
                    status = "fail"
                    error = "HRRR reduced GRIB is not parseable with wgrib2"
                else:
                    result = inspect_result
    return make_stage(
        "hrrr_live_download",
        start_time,
        status=status,
        command=command,
        artifacts={"subset_grib": str(subset_path), "manifest_csv": str(manifest_path)},
        error=error,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_hrrr_live_builder_stage(output_dir: pathlib.Path, *, date: str) -> StageResult:
    stage_root = output_dir / "live" / "hrrr_builder"
    command = [
        sys.executable,
        str(HRRR_DIR / "build_hrrr_klga_feature_shards.py"),
        "--start-date",
        date,
        "--end-date",
        date,
        "--download-dir",
        str(stage_root / "downloads"),
        "--reduced-dir",
        str(stage_root / "reduced"),
        "--output-dir",
        str(stage_root / "output"),
        "--max-workers",
        "1",
        "--limit-tasks",
        "1",
    ]
    start_time = time.perf_counter()
    result = run_command(command)
    manifest_json = stage_root / "output" / f"{date[:7]}.manifest.json"
    manifest_parquet = stage_root / "output" / f"{date[:7]}.manifest.parquet"
    status = classify_live_stage_status(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    error = None
    if status == "pass" and (not manifest_json.exists() or not manifest_parquet.exists()):
        status = "fail"
        error = "HRRR builder smoke did not write manifest outputs"
    elif status == "pass":
        manifest_df = pd.read_parquet(manifest_parquet)
        if manifest_df.empty:
            status = "fail"
            error = "HRRR builder smoke wrote an empty manifest parquet"
    return make_stage(
        "hrrr_live_builder_smoke",
        start_time,
        status=status,
        command=command,
        artifacts={"manifest_json": str(manifest_json), "manifest_parquet": str(manifest_parquet)},
        error=error,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_nbm_live_download_stage(output_dir: pathlib.Path, *, date: str, cycle: str, forecast_hour: int) -> StageResult:
    stage_dir = output_dir / "live" / "nbm_download"
    command = [
        sys.executable,
        str(NBM_DIR / "fetch_nbm.py"),
        "grib2",
        "--date",
        date,
        "--cycle",
        cycle,
        "--forecast-hour",
        str(forecast_hour),
        "--with-idx",
        "--output-dir",
        str(stage_dir),
        "--overwrite",
    ]
    start_time = time.perf_counter()
    result = run_command(command)
    stem = f"blend.t{cycle}z.core.f{forecast_hour:03d}.co.grib2"
    grib_path = stage_dir / stem
    idx_path = stage_dir / f"{stem}.idx"
    status = classify_live_stage_status(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    error = None
    if status == "pass":
        if not grib_path.exists() or not idx_path.exists():
            status = "fail"
            error = "NBM download stage did not write GRIB2 and idx outputs"
        else:
            first_lines = idx_path.read_text().splitlines()[:8]
            if not any(f"d={date.replace('-', '')}{cycle}" in line for line in first_lines):
                status = "fail"
                error = "NBM idx output does not match requested date/cycle"
    return make_stage(
        "nbm_live_download",
        start_time,
        status=status,
        command=command,
        artifacts={"grib2": str(grib_path), "idx": str(idx_path)},
        error=error,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def run_nbm_live_reduction_stage(output_dir: pathlib.Path, *, date: str, cycle: str, forecast_hour: int) -> StageResult:
    start_time = time.perf_counter()
    cycle_date = dt.datetime.strptime(date, "%Y-%m-%d").date()
    init_time_utc = dt.datetime.combine(cycle_date, dt.time(hour=int(cycle)), tzinfo=dt.timezone.utc)
    cycle_plan = nbm_pipeline.CyclePlan(
        init_time_utc=init_time_utc,
        init_time_local=init_time_utc.astimezone(nbm_pipeline.NY_TZ),
        cycle=cycle,
    )
    args = argparse.Namespace(
        region="co",
        output_dir=output_dir / "live" / "nbm_reduction",
        overwrite=True,
        left=nbm_pipeline.CROP_BOUNDS["left"],
        right=nbm_pipeline.CROP_BOUNDS["right"],
        bottom=nbm_pipeline.CROP_BOUNDS["bottom"],
        top=nbm_pipeline.CROP_BOUNDS["top"],
        keep_downloads=True,
        keep_reduced=True,
        write_long=False,
    )
    client = S3HttpClient(pool_maxsize=8)
    try:
        result = nbm_pipeline.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=forecast_hour)
    except Exception as exc:
        status = classify_live_stage_status(returncode=1, stdout="", stderr="", error=str(exc))
        return make_stage("nbm_live_reduction_smoke", start_time, status=status, error=str(exc))
    reduced_path = pathlib.Path(result.manifest_row["reduced_file_path"])
    status = "pass"
    error = None
    if result.manifest_row["extraction_status"] != "ok":
        error = str(result.manifest_row["extraction_status"])
        status = "degraded" if network_error_text(result.manifest_row.get("warnings", "")) else "fail"
    elif not reduced_path.exists():
        status = "fail"
        error = "NBM reduction smoke did not retain the reduced GRIB"
    else:
        wgrib2 = resolve_wgrib2_binary()
        if wgrib2 is None:
            status = "fail"
            error = f"wgrib2 is required for verification. Install it on PATH, set {WGRIB2_ENV_VAR}, or use an active environment that provides it."
        else:
            inspect_result = run_command([wgrib2, str(reduced_path), "-s"])
            if inspect_result.returncode != 0 or not inspect_result.stdout.strip():
                status = "fail"
                error = "NBM reduced GRIB is not parseable with wgrib2"
    return make_stage(
        "nbm_live_reduction_smoke",
        start_time,
        status=status,
        artifacts={"reduced_grib": str(reduced_path)},
        error=error,
        stdout="",
        stderr=result.manifest_row.get("warnings", ""),
    )


def run_live_source_lane(output_dir: pathlib.Path, args: argparse.Namespace) -> list[StageResult]:
    return [
        run_hrrr_live_download_stage(output_dir, date=args.hrrr_date, cycle=args.hrrr_cycle, forecast_hour=args.hrrr_forecast_hour),
        run_hrrr_live_builder_stage(output_dir, date=args.hrrr_date),
        run_nbm_live_download_stage(output_dir, date=args.nbm_date, cycle=args.nbm_cycle, forecast_hour=args.nbm_forecast_hour),
        run_nbm_live_reduction_stage(output_dir, date=args.nbm_date, cycle=args.nbm_cycle, forecast_hour=args.nbm_forecast_hour),
    ]


def build_summary(*, local_stages: list[StageResult], live_stages: list[StageResult], output_dir: pathlib.Path) -> dict[str, Any]:
    local_status = lane_status(local_stages)
    live_status = lane_status(live_stages)
    return {
        "local_contract_lane": {
            "status": local_status,
            "stages": [dataclasses.asdict(stage) for stage in local_stages],
        },
        "live_source_lane": {
            "status": live_status,
            "stages": [dataclasses.asdict(stage) for stage in live_stages],
        },
        "overall_status": overall_status(local_status=local_status, live_status=live_status),
        "summary_json_path": str(output_dir / "summary.json"),
    }


def print_human_summary(summary: dict[str, Any]) -> None:
    print(
        "\n".join(
            [
                f"overall={summary['overall_status']}",
                f"local_contract_lane={summary['local_contract_lane']['status']}",
                f"live_source_lane={summary['live_source_lane']['status']}",
                f"summary_json={summary['summary_json_path']}",
            ]
        )
    )


def main() -> int:
    args = parse_args()
    if args.local_only and args.live_only:
        raise SystemExit("--local-only and --live-only cannot be used together")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    local_stages = [] if args.live_only else run_local_contract_lane(output_dir)
    live_stages = [] if args.local_only else run_live_source_lane(output_dir, args)
    summary = build_summary(local_stages=local_stages, live_stages=live_stages, output_dir=output_dir)
    write_summary(output_dir / "summary.json", summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print_human_summary(summary)
    return 0 if summary["local_contract_lane"]["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
