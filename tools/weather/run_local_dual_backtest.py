#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.dual_guard import apply_guard_to_prediction_payloads


DEFAULT_OUTPUT_DIR = pathlib.Path("data/runtime/local_dual_backtest")
NEARBY_STATIONS = ("KJRB", "KJFK", "KEWR", "KTEB")
RUNNER_VERSION = "local_dual_backtest_v2"
DEFAULT_DUAL_GUARD_MANIFEST_PATH = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/dual_guard/dual_guard_manifest.json")


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; expected YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local no-HRRR/with-HRRR production inference over a date range with date-level concurrency."
    )
    parser.add_argument("date", nargs="?", type=parse_date, help="Optional single YYYY-MM-DD date.")
    parser.add_argument("--start-date", type=parse_date)
    parser.add_argument("--end-date", type=parse_date)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--day-workers", type=int, default=2, help="Number of target dates to process concurrently.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip dates that already completed successfully.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after failures.")
    parser.add_argument("--keep-artifacts", action="store_true", help="Keep heavy downloaded/intermediate artifacts for debugging.")
    parser.add_argument("--lamp-source", choices=("auto", "live", "archive", "iem"), default="archive")
    parser.add_argument("--event-bins-mode", choices=("polymarket", "standard"), default="polymarket")
    parser.add_argument("--nbm-lead-workers", type=int, default=8)
    parser.add_argument("--nbm-download-workers", type=int, default=6)
    parser.add_argument("--nbm-reduce-workers", type=int, default=4)
    parser.add_argument("--nbm-extract-workers", type=int, default=4)
    parser.add_argument("--nbm-batch-reduce-mode", choices=("off", "cycle"), default="off")
    parser.add_argument("--hrrr-max-workers", type=int, default=6)
    parser.add_argument("--hrrr-download-workers", type=int, default=6)
    parser.add_argument("--hrrr-reduce-workers", type=int, default=2)
    parser.add_argument("--hrrr-extract-workers", type=int, default=2)
    return parser.parse_args()


def date_range(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise SystemExit(f"end date {end.isoformat()} is before start date {start.isoformat()}")
    days: list[dt.date] = []
    current = start
    while current <= end:
        days.append(current)
        current += dt.timedelta(days=1)
    return days


def run_logged(command: list[str], log_path: pathlib.Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("+ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        return int(proc.returncode)


def start_logged(command: list[str], log_path: pathlib.Path) -> tuple[subprocess.Popen[str], Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    log.write("+ " + " ".join(command) + "\n")
    log.flush()
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc, log


def standard_event_bins_path(run_root: pathlib.Path) -> pathlib.Path:
    labels = [
        "30F or below",
        "31-32F",
        "33-34F",
        "35-36F",
        "37-38F",
        "39-40F",
        "41-42F",
        "43-44F",
        "45-46F",
        "47-48F",
        "49-50F",
        "51-52F",
        "53-54F",
        "55-56F",
        "57-58F",
        "59-60F",
        "61-62F",
        "63-64F",
        "65-66F",
        "67-68F",
        "69-70F",
        "71-72F",
        "73-74F",
        "75-76F",
        "77-78F",
        "79-80F",
        "81-82F",
        "83-84F",
        "85-86F",
        "87-88F",
        "89-90F",
        "91F or higher",
    ]
    path = run_root / "event_bins_standard.json"
    path.write_text(json.dumps({"labels": labels}, indent=2, sort_keys=True) + "\n")
    return path


def fetch_nearby_station(*, station: str, target_date: dt.date, root: pathlib.Path, log_path: pathlib.Path) -> pathlib.Path:
    history_dir = root / "history" / station
    tables_dir = root / "tables" / station
    start_date = target_date - dt.timedelta(days=1)
    fetch_cmd = [
        sys.executable,
        "wunderground/fetch_daily_history.py",
        "--location-id",
        f"{station}:9:US",
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        target_date.isoformat(),
        "--output-dir",
        str(history_dir),
        "--skip-http-status",
        "400",
        "--skip-http-status",
        "404",
        "--force",
    ]
    build_cmd = [
        sys.executable,
        "wunderground/build_training_tables.py",
        "--history-dir",
        str(history_dir),
        "--station-id",
        station,
        "--output-dir",
        str(tables_dir),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        for command in (fetch_cmd, build_cmd):
            log.write("+ " + " ".join(command) + "\n")
            log.flush()
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"nearby station {station} failed with returncode={proc.returncode}")
    obs_path = tables_dir / "wu_obs_intraday.parquet"
    if not obs_path.exists():
        raise RuntimeError(f"nearby station {station} did not produce {obs_path}")
    return obs_path


def find_event_bins_path(no_hrrr_runtime: pathlib.Path, standard_path: pathlib.Path | None) -> pathlib.Path:
    if standard_path is not None:
        return standard_path
    matches = sorted((no_hrrr_runtime / "polymarket").glob("event_slug=*/event_bins.json"))
    if not matches:
        raise RuntimeError(f"no event_bins.json found under {no_hrrr_runtime / 'polymarket'}")
    return matches[0]


def cleanup_paths(paths: list[pathlib.Path]) -> list[str]:
    deleted: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path))
    return deleted


def preserve_event_metadata(event_bins_path: pathlib.Path, run_root: pathlib.Path) -> pathlib.Path:
    preserved_root = run_root / "event_metadata"
    preserved_root.mkdir(parents=True, exist_ok=True)
    if event_bins_path.name == "event_bins.json":
        for name in ("event_bins.json", "event_bins.manifest.json", "polymarket_event.json"):
            source = event_bins_path.parent / name
            if source.exists():
                shutil.copy2(source, preserved_root / name)
        return preserved_root / "event_bins.json"
    destination = preserved_root / event_bins_path.name
    shutil.copy2(event_bins_path, destination)
    return destination


def existing_summary_matches(summary: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        summary.get("status") == "ok"
        and summary.get("event_bins_mode") == args.event_bins_mode
        and summary.get("lamp_source") == args.lamp_source
        and summary.get("runner_version") == RUNNER_VERSION
    )


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_comparison(*, run_root: pathlib.Path, target_date: dt.date, no_hrrr_prediction: pathlib.Path, with_hrrr_prediction: pathlib.Path) -> pathlib.Path:
    no_payload = load_json(no_hrrr_prediction)
    with_payload = load_json(with_hrrr_prediction)
    no_bins = {row["bin"]: float(row["probability"]) for row in no_payload.get("event_bins", [])}
    with_bins = {row["bin"]: float(row["probability"]) for row in with_payload.get("event_bins", [])}
    comparison = {
        "target_date_local": target_date.isoformat(),
        "local_run_root": str(run_root),
        "predictions": {
            "no_hrrr": {
                "path": str(no_hrrr_prediction),
                "status": no_payload.get("status"),
                "expected_final_tmax_f": no_payload.get("expected_final_tmax_f"),
                "anchor_tmax_f": no_payload.get("anchor_tmax_f"),
                "distribution_method": no_payload.get("distribution_method"),
                "final_tmax_quantiles_f": no_payload.get("final_tmax_quantiles_f"),
                "event_bins": no_payload.get("event_bins", []),
            },
            "with_hrrr": {
                "path": str(with_hrrr_prediction),
                "status": with_payload.get("status"),
                "expected_final_tmax_f": with_payload.get("expected_final_tmax_f"),
                "anchor_tmax_f": with_payload.get("anchor_tmax_f"),
                "distribution_method": with_payload.get("distribution_method"),
                "source_disagreement": with_payload.get("source_disagreement"),
                "calibration": with_payload.get("calibration"),
                "ladder_calibration": with_payload.get("ladder_calibration"),
                "final_tmax_quantiles_f": with_payload.get("final_tmax_quantiles_f"),
                "event_bins": with_payload.get("event_bins", []),
            },
        },
        "diff_with_hrrr_minus_no_hrrr": {
            "expected_final_tmax_f": float(with_payload["expected_final_tmax_f"]) - float(no_payload["expected_final_tmax_f"]),
            "anchor_tmax_f": float(with_payload.get("anchor_tmax_f", 0.0)) - float(no_payload.get("anchor_tmax_f", 0.0)),
            "event_bins": [
                {"bin": label, "probability_diff": with_bins[label] - no_bins[label]}
                for label in no_bins
                if label in with_bins
            ],
        },
    }
    if DEFAULT_DUAL_GUARD_MANIFEST_PATH.exists():
        manifest = load_json(DEFAULT_DUAL_GUARD_MANIFEST_PATH)
        candidate_id = str(manifest.get("selected_candidate_id") or "always_with_hrrr")
        guarded = apply_guard_to_prediction_payloads(no_payload, with_payload, candidate_id=candidate_id)
        guarded["manifest_path"] = str(DEFAULT_DUAL_GUARD_MANIFEST_PATH)
        comparison["guarded_recommendation"] = guarded
    path = run_root / "comparison.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    return path


def run_one(target_date: dt.date, args: argparse.Namespace) -> dict[str, Any]:
    started = dt.datetime.now(dt.timezone.utc)
    date_text = target_date.isoformat()
    run_root = args.output_dir / date_text
    runtime_root = run_root / "runtime"
    no_hrrr_runtime = runtime_root / "no_hrrr"
    with_hrrr_runtime = runtime_root / "with_hrrr"
    hrrr_run_root = runtime_root / "hrrr"
    nearby_root = runtime_root / "nearby"
    predictions_root = run_root / "predictions"
    no_pred_dir = predictions_root / "no_hrrr"
    with_pred_dir = predictions_root / "with_hrrr"
    logs_dir = run_root / "logs"
    no_pred = no_pred_dir / f"prediction_KLGA_{date_text}.json"
    with_pred = with_pred_dir / f"prediction_KLGA_{date_text}.json"
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.keep_artifacts:
        cleanup_paths([runtime_root])
    no_pred.unlink(missing_ok=True)
    with_pred.unlink(missing_ok=True)
    (run_root / "comparison.json").unlink(missing_ok=True)
    no_pred_dir.mkdir(parents=True, exist_ok=True)
    with_pred_dir.mkdir(parents=True, exist_ok=True)
    standard_bins = standard_event_bins_path(run_root) if args.event_bins_mode == "standard" else None

    no_cmd = [
        sys.executable,
        "-m",
        "experiments.no_hrrr_model.no_hrrr_model.run_online_inference",
        "--target-date-local",
        date_text,
        "--runtime-root",
        str(no_hrrr_runtime),
        "--prediction-output-dir",
        str(no_pred_dir),
        "--lamp-source",
        args.lamp_source,
        "--nbm-batch-reduce-mode",
        args.nbm_batch_reduce_mode,
        "--nbm-lead-workers",
        str(args.nbm_lead_workers),
        "--nbm-download-workers",
        str(args.nbm_download_workers),
        "--nbm-reduce-workers",
        str(args.nbm_reduce_workers),
        "--nbm-extract-workers",
        str(args.nbm_extract_workers),
        "--overwrite",
        "--keep-artifacts",
    ]
    if standard_bins is None:
        no_cmd.append("--polymarket-event-slug")
    else:
        no_cmd.extend(["--event-bins-path", str(standard_bins)])

    hrrr_cmd = [
        sys.executable,
        "tools/hrrr/run_hrrr_monthly_backfill.py",
        "--start-local-date",
        date_text,
        "--end-local-date",
        date_text,
        "--run-root",
        str(hrrr_run_root),
        "--selection-mode",
        "overnight_0005",
        "--batch-reduce-mode",
        "cycle",
        "--day-workers",
        "1",
        "--max-workers",
        str(args.hrrr_max_workers),
        "--download-workers",
        str(args.hrrr_download_workers),
        "--reduce-workers",
        str(args.hrrr_reduce_workers),
        "--extract-workers",
        str(args.hrrr_extract_workers),
        "--reduce-queue-size",
        "2",
        "--extract-queue-size",
        "2",
        "--range-merge-gap-bytes",
        "65536",
        "--crop-method",
        "auto",
        "--crop-grib-type",
        "same",
        "--wgrib2-threads",
        "1",
        "--extract-method",
        "wgrib2-bin",
        "--summary-profile",
        "overnight",
        "--skip-provenance",
        "--progress-mode",
        "log",
    ]

    no_proc, no_log = start_logged(no_cmd, logs_dir / "no_hrrr.log")
    hrrr_proc, hrrr_log = start_logged(hrrr_cmd, logs_dir / "hrrr.log")
    nearby_paths: dict[str, pathlib.Path] = {}
    nearby_error: str | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(NEARBY_STATIONS)) as nearby_executor:
        futures = {
            nearby_executor.submit(
                fetch_nearby_station,
                station=station,
                target_date=target_date,
                root=nearby_root,
                log_path=logs_dir / f"nearby_{station}.log",
            ): station
            for station in NEARBY_STATIONS
        }
        for future in concurrent.futures.as_completed(futures):
            station = futures[future]
            try:
                nearby_paths[station] = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve date-level failure context.
                nearby_error = f"{station}: {exc}"

    no_status = no_proc.wait()
    hrrr_status = hrrr_proc.wait()
    no_log.close()
    hrrr_log.close()
    if no_status != 0:
        raise RuntimeError(f"no-HRRR inference failed returncode={no_status}; see {logs_dir / 'no_hrrr.log'}")
    if hrrr_status != 0:
        raise RuntimeError(f"HRRR source build failed returncode={hrrr_status}; see {logs_dir / 'hrrr.log'}")
    if nearby_error is not None:
        raise RuntimeError(f"nearby fetch failed: {nearby_error}")
    if not no_pred.exists():
        raise RuntimeError(f"missing no-HRRR prediction: {no_pred}")
    event_bins_path = find_event_bins_path(no_hrrr_runtime, standard_bins)
    preserved_event_bins_path = preserve_event_metadata(event_bins_path, run_root)
    with_cmd = [
        sys.executable,
        "-m",
        "experiments.withhrrr.withhrrr_model.run_online_inference",
        "--target-date-local",
        date_text,
        "--station-id",
        "KLGA",
        "--runtime-root",
        str(with_hrrr_runtime),
        "--prediction-output-dir",
        str(with_pred_dir),
        "--wunderground-tables-dir",
        str(no_hrrr_runtime / "wunderground_tables" / f"target_date_local={date_text}"),
        "--nbm-root",
        str(no_hrrr_runtime / "nbm" / "nbm_overnight"),
        "--lamp-root",
        str(no_hrrr_runtime / "lamp_overnight"),
        "--hrrr-root",
        str(hrrr_run_root / "hrrr_summary"),
        "--event-bins-path",
        str(preserved_event_bins_path),
        "--skip-wunderground",
        "--skip-lamp",
        "--skip-nbm",
        "--skip-hrrr",
        "--skip-nearby-wunderground",
        "--overwrite",
    ]
    for station, path in sorted(nearby_paths.items()):
        with_cmd.extend(["--nearby-obs-path", f"{station}={path}"])
    with_status = run_logged(with_cmd, logs_dir / "with_hrrr.log")
    if with_status != 0:
        raise RuntimeError(f"with-HRRR inference failed returncode={with_status}; see {logs_dir / 'with_hrrr.log'}")
    if not with_pred.exists():
        raise RuntimeError(f"missing with-HRRR prediction: {with_pred}")
    comparison_path = write_comparison(
        run_root=run_root,
        target_date=target_date,
        no_hrrr_prediction=no_pred,
        with_hrrr_prediction=with_pred,
    )
    deleted: list[str] = []
    if not args.keep_artifacts:
        deleted = cleanup_paths(
            [
                no_hrrr_runtime / "wunderground_history",
                no_hrrr_runtime / "wunderground_tables",
                no_hrrr_runtime / "lamp_raw",
                no_hrrr_runtime / "lamp_features",
                no_hrrr_runtime / "lamp_overnight",
                no_hrrr_runtime / "nbm",
                no_hrrr_runtime / "polymarket",
                no_hrrr_runtime / "prediction_features",
                hrrr_run_root,
                nearby_root,
                with_hrrr_runtime / "prediction_features",
            ]
        )
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        "target_date_local": date_text,
        "status": "ok",
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "local_run_root": str(run_root),
        "event_bins_mode": args.event_bins_mode,
        "lamp_source": args.lamp_source,
        "runner_version": RUNNER_VERSION,
        "comparison_path": str(comparison_path),
        "no_hrrr_prediction_path": str(no_pred),
        "with_hrrr_prediction_path": str(with_pred),
        "event_bins_path": str(preserved_event_bins_path),
        "deleted_artifacts": deleted,
    }


def run_one_with_summary(target_date: dt.date, args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.output_dir / target_date.isoformat()
    try:
        summary = run_one(target_date, args)
    except Exception as exc:  # noqa: BLE001 - write per-date summary for later diagnosis.
        deleted: list[str] = []
        if not args.keep_artifacts:
            deleted = cleanup_paths([run_root / "runtime"])
        summary = {
            "target_date_local": target_date.isoformat(),
            "status": "failed",
            "message": str(exc),
            "local_run_root": str(run_root),
            "runner_version": RUNNER_VERSION,
            "deleted_artifacts": deleted,
        }
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    if args.date is not None:
        days = [args.date]
    else:
        if args.start_date is None or args.end_date is None:
            print("provide either a single date or both --start-date and --end-date", file=sys.stderr)
            return 2
        days = date_range(args.start_date, args.end_date)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    if manifest_path.exists() and not args.skip_existing:
        manifest_path.unlink()
    print(f"running {len(days)} dates locally: {days[0].isoformat()}..{days[-1].isoformat()}")
    print(f"local_output_dir={args.output_dir}")
    print(f"day_workers={args.day_workers}")
    run_rows_by_date: dict[str, dict[str, Any]] = {}
    pending_days: list[tuple[int, dt.date]] = []
    for index, day in enumerate(days, start=1):
        summary_path = args.output_dir / day.isoformat() / "summary.json"
        if args.skip_existing and summary_path.exists():
            existing = load_json(summary_path)
            if existing_summary_matches(existing, args):
                existing = {**existing, "skipped_existing": True}
                run_rows_by_date[day.isoformat()] = existing
                with manifest_path.open("a") as handle:
                    handle.write(json.dumps(existing, sort_keys=True) + "\n")
                print(f"[skip] {day.isoformat()} existing ok event_bins_mode={args.event_bins_mode} lamp_source={args.lamp_source}")
                continue
        pending_days.append((index, day))
    stopped_after_failure = False
    next_pending = 0
    running: dict[concurrent.futures.Future[dict[str, Any]], tuple[int, dt.date]] = {}
    max_workers = max(1, int(args.day_workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        while next_pending < len(pending_days) and len(running) < max_workers:
            index, day = pending_days[next_pending]
            print(f"[{index}/{len(days)}] {day.isoformat()} start", flush=True)
            running[executor.submit(run_one_with_summary, day, args)] = (index, day)
            next_pending += 1
        while running:
            done, _not_done = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                _index, day = running.pop(future)
                try:
                    summary = future.result()
                except Exception as exc:  # noqa: BLE001 - keep aggregate accounting intact.
                    summary = {
                        "target_date_local": day.isoformat(),
                        "status": "failed",
                        "message": str(exc),
                        "local_run_root": str(args.output_dir / day.isoformat()),
                        "runner_version": RUNNER_VERSION,
                    }
                    run_root = args.output_dir / day.isoformat()
                    run_root.mkdir(parents=True, exist_ok=True)
                    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
                run_rows_by_date[day.isoformat()] = summary
                with manifest_path.open("a") as handle:
                    handle.write(json.dumps(summary, sort_keys=True) + "\n")
                elapsed = summary.get("elapsed_seconds")
                elapsed_text = f"{float(elapsed):.1f}s" if elapsed is not None else "n/a"
                print(f"[done] {day.isoformat()} status={summary.get('status')} elapsed={elapsed_text}")
                if summary.get("status") != "ok" and not args.keep_going:
                    stopped_after_failure = True
            if not stopped_after_failure:
                while next_pending < len(pending_days) and len(running) < max_workers:
                    index, day = pending_days[next_pending]
                    print(f"[{index}/{len(days)}] {day.isoformat()} start", flush=True)
                    running[executor.submit(run_one_with_summary, day, args)] = (index, day)
                    next_pending += 1
    run_rows = [run_rows_by_date[day.isoformat()] for day in days if day.isoformat() in run_rows_by_date]
    aggregate = {
        "status": "ok" if all(row.get("status") == "ok" for row in run_rows) else "failed",
        "start_date": days[0].isoformat(),
        "end_date": days[-1].isoformat(),
        "date_count": len(days),
        "completed_count": len(run_rows),
        "ok_count": sum(1 for row in run_rows if row.get("status") == "ok"),
        "failed_count": sum(1 for row in run_rows if row.get("status") != "ok"),
        "runner_version": RUNNER_VERSION,
        "stopped_after_failure": stopped_after_failure,
        "runs": run_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(f"summary_path={args.output_dir / 'summary.json'}")
    print(f"status={aggregate['status']} ok={aggregate['ok_count']} failed={aggregate['failed_count']}")
    return 0 if aggregate["failed_count"] == 0 and not stopped_after_failure else 1


if __name__ == "__main__":
    raise SystemExit(main())
