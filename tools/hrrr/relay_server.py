#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_hrrr_klga_feature_shards as hrrr
from fetch_hrrr_records import download_subset_for_inventory_patterns
from tools.weather.progress import create_progress_reporter


DEFAULT_RUN_ROOT = Path("data/runtime/backfill_overnight")
DEFAULT_SELECTION_MODE = "overnight_0005"
DEFAULT_MAX_ATTEMPTS = 6
DEFAULT_PREPARE_WORKERS = 4
DEFAULT_TARGET_READY_TASKS = 64
DEFAULT_PREPARE_BATCH_SIZE = 16
DEFAULT_POLL_SECONDS = 5.0
DEFAULT_RECOVER_SECONDS = 30.0
DEFAULT_MIN_FREE_GB = 5.0
TASK_STATUSES = {
    "pending",
    "preparing",
    "ready",
    "leased",
    "result_uploaded",
    "completed",
    "retry_pending",
    "failed_terminal",
}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_text() -> str:
    return utc_now().isoformat()


def parse_utc(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def relay_root(run_root: Path) -> Path:
    return run_root / "hrrr_relay"


def db_path(run_root: Path) -> Path:
    return relay_root(run_root) / "queue.sqlite"


def pause_path(run_root: Path) -> Path:
    return relay_root(run_root) / "pause.flag"


def raw_root(run_root: Path) -> Path:
    return relay_root(run_root) / "raw"


def uploads_root(run_root: Path) -> Path:
    return relay_root(run_root) / "uploads"


def results_root(run_root: Path) -> Path:
    return relay_root(run_root) / "results"


def logs_root(run_root: Path) -> Path:
    return relay_root(run_root) / "logs"


def task_raw_dir(run_root: Path, task_id: str) -> Path:
    return raw_root(run_root) / task_id


def task_result_dir(run_root: Path, task_id: str) -> Path:
    return results_root(run_root) / task_id


def safe_task_id(task_key: str) -> str:
    return (
        task_key.replace(":", "")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "_")
    )


def connect(run_root: Path) -> sqlite3.Connection:
    path = db_path(run_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            target_date_local TEXT NOT NULL,
            task_json TEXT NOT NULL,
            status TEXT NOT NULL,
            attempt_count INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at TEXT,
            raw_dir TEXT,
            result_dir TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS days (
            target_date_local TEXT PRIMARY KEY,
            selection_mode TEXT NOT NULL,
            status TEXT NOT NULL,
            expected_task_count INTEGER NOT NULL,
            finalized_at TEXT,
            last_error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT,
            target_date_local TEXT,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_status_target
            ON tasks(status, target_date_local, task_id);
        CREATE INDEX IF NOT EXISTS idx_tasks_lease_expiry
            ON tasks(status, lease_expires_at);
        """
    )
    conn.commit()


def begin_immediate(conn: sqlite3.Connection) -> None:
    conn.execute("BEGIN IMMEDIATE")


def log_event(
    conn: sqlite3.Connection,
    *,
    task_id: str | None,
    target_date_local: str | None,
    event_type: str,
    details: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO events(task_id, target_date_local, timestamp, event_type, details_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (task_id, target_date_local, utc_now_text(), event_type, json.dumps(details or {}, sort_keys=True)),
    )


def task_from_row(row: sqlite3.Row) -> hrrr.TaskSpec:
    return hrrr.TaskSpec(**json.loads(str(row["task_json"])))


def task_paths(run_root: Path, task_id: str) -> dict[str, Path]:
    root = task_raw_dir(run_root, task_id)
    return {
        "root": root,
        "task_json": root / "task.json",
        "raw": root / "raw.grib2",
        "manifest": root / "raw.manifest.csv",
        "selection": root / "raw.selection.csv",
        "checksum": root / "checksum.sha256",
        "metadata": root / "metadata.json",
    }


def sha256_file(path: Path) -> str:
    return hrrr.sha256_file(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp_path.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
        handle.write(text)
    tmp_path.replace(path)


def write_parquet_atomic(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".parquet", dir=str(path.parent), delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        df.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def is_paused(run_root: Path) -> bool:
    return pause_path(run_root).exists()


def ensure_layout(run_root: Path) -> None:
    for path in (relay_root(run_root), raw_root(run_root), uploads_root(run_root), results_root(run_root), logs_root(run_root)):
        path.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def task_count_for_day(conn: sqlite3.Connection, target_date_local: str, status: str | None = None) -> int:
    if status is None:
        row = conn.execute("SELECT COUNT(*) AS count FROM tasks WHERE target_date_local = ?", (target_date_local,)).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM tasks WHERE target_date_local = ? AND status = ?",
            (target_date_local, status),
        ).fetchone()
    return int(row["count"])


def task_artifact_dirs(run_root: Path, row: sqlite3.Row) -> list[Path]:
    task_id = str(row["task_id"])
    paths = [task_raw_dir(run_root, task_id), task_result_dir(run_root, task_id)]
    if row["raw_dir"]:
        paths.append(Path(str(row["raw_dir"])))
    if row["result_dir"]:
        paths.append(Path(str(row["result_dir"])))
    return paths


def delete_task_artifacts(run_root: Path, row: sqlite3.Row) -> None:
    seen: set[str] = set()
    for path in task_artifact_dirs(run_root, row):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        delete_path(path)


def reconcile_day_tasks(
    conn: sqlite3.Connection,
    *,
    run_root: Path,
    target_date_local: str,
    expected_task_json: dict[str, str],
    now: str,
) -> bool:
    existing_rows = conn.execute(
        "SELECT * FROM tasks WHERE target_date_local = ?",
        (target_date_local,),
    ).fetchall()
    changed = False
    for row in existing_rows:
        task_id = str(row["task_id"])
        expected_json = expected_task_json.get(task_id)
        if expected_json is None:
            delete_task_artifacts(run_root, row)
            conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
            log_event(conn, task_id=task_id, target_date_local=target_date_local, event_type="task_removed_from_plan")
            changed = True
            continue
        if str(row["task_json"]) != expected_json:
            delete_task_artifacts(run_root, row)
            conn.execute(
                """
                UPDATE tasks
                SET task_json = ?, status = 'pending', attempt_count = 0, lease_owner = NULL,
                    lease_expires_at = NULL, raw_dir = ?, result_dir = ?, last_error = NULL, updated_at = ?
                WHERE task_id = ?
                """,
                (
                    expected_json,
                    str(task_raw_dir(run_root, task_id)),
                    str(task_result_dir(run_root, task_id)),
                    now,
                    task_id,
                ),
            )
            log_event(conn, task_id=task_id, target_date_local=target_date_local, event_type="task_replanned")
            changed = True
    return changed


def init_range(run_root: Path, *, start_local_date: str, end_local_date: str, selection_mode: str) -> dict[str, Any]:
    ensure_layout(run_root)
    conn = connect(run_root)
    start_date = dt.date.fromisoformat(start_local_date)
    end_date = dt.date.fromisoformat(end_local_date)
    if end_date < start_date:
        raise ValueError("--end-local-date must be on or after --start-local-date")

    inserted_tasks = 0
    inserted_days = 0
    now = utc_now_text()
    with conn:
        current = start_date
        while current <= end_date:
            target = current.isoformat()
            tasks = hrrr.build_tasks_for_target_date(pd.Timestamp(target), selection_mode=selection_mode)
            expected_task_json = {safe_task_id(task.key): json.dumps(asdict(task), sort_keys=True) for task in tasks}
            existing_day = conn.execute("SELECT * FROM days WHERE target_date_local = ?", (target,)).fetchone()
            if existing_day is None:
                inserted_days += 1
            day_changed = reconcile_day_tasks(
                conn,
                run_root=run_root,
                target_date_local=target,
                expected_task_json=expected_task_json,
                now=now,
            )
            if existing_day is not None:
                day_changed = (
                    day_changed
                    or str(existing_day["selection_mode"]) != str(selection_mode)
                    or int(existing_day["expected_task_count"]) != len(tasks)
                )
            conn.execute(
                """
                INSERT INTO days(target_date_local, selection_mode, status, expected_task_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(target_date_local) DO UPDATE SET
                    selection_mode = excluded.selection_mode,
                    expected_task_count = excluded.expected_task_count,
                    updated_at = excluded.updated_at
                """,
                (target, selection_mode, "pending", len(tasks), now, now),
            )
            if day_changed:
                paths = hrrr_summary_paths(run_root, target)
                delete_path(paths["summary_root"])
                delete_path(paths["state_root"])
                conn.execute(
                    """
                    UPDATE days
                    SET status = 'pending', finalized_at = NULL, last_error = NULL, updated_at = ?
                    WHERE target_date_local = ?
                    """,
                    (now, target),
                )
            for task in tasks:
                task_id = safe_task_id(task.key)
                task_json = expected_task_json[task_id]
                before = conn.total_changes
                conn.execute(
                    """
                    INSERT OR IGNORE INTO tasks(
                        task_id, target_date_local, task_json, status, attempt_count,
                        raw_dir, result_dir, created_at, updated_at
                    )
                    VALUES (?, ?, ?, 'pending', 0, ?, ?, ?, ?)
                    """,
                    (
                        task_id,
                        target,
                        task_json,
                        str(task_raw_dir(run_root, task_id)),
                        str(task_result_dir(run_root, task_id)),
                        now,
                        now,
                    ),
                )
                if conn.total_changes > before:
                    inserted_tasks += 1
                    log_event(conn, task_id=task_id, target_date_local=target, event_type="task_created")
            log_event(conn, task_id=None, target_date_local=target, event_type="day_initialized", details={"task_count": len(tasks)})
            current += dt.timedelta(days=1)

    return {"status": "ok", "inserted_days": inserted_days, "inserted_tasks": inserted_tasks}


def command_init(args: argparse.Namespace) -> int:
    payload = init_range(
        args.run_root,
        start_local_date=args.start_local_date,
        end_local_date=args.end_local_date,
        selection_mode=args.selection_mode,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


def claim_prepare_task(conn: sqlite3.Connection, *, max_attempts: int) -> sqlite3.Row | None:
    begin_immediate(conn)
    row = conn.execute(
        """
        SELECT * FROM tasks
        WHERE status IN ('pending', 'retry_pending') AND attempt_count < ?
        ORDER BY target_date_local, task_id
        LIMIT 1
        """,
        (max_attempts,),
    ).fetchone()
    if row is None:
        conn.commit()
        return None
    now = utc_now_text()
    conn.execute(
        """
        UPDATE tasks
        SET status = 'preparing', lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
        WHERE task_id = ?
        """,
        (now, row["task_id"]),
    )
    log_event(conn, task_id=row["task_id"], target_date_local=row["target_date_local"], event_type="prepare_claimed")
    conn.commit()
    return conn.execute("SELECT * FROM tasks WHERE task_id = ?", (row["task_id"],)).fetchone()


def mark_prepare_failure(conn: sqlite3.Connection, row: sqlite3.Row, *, error: str, max_attempts: int) -> None:
    begin_immediate(conn)
    current_attempts = int(row["attempt_count"])
    next_attempts = current_attempts + 1
    status = "failed_terminal" if next_attempts >= max_attempts else "retry_pending"
    now = utc_now_text()
    conn.execute(
        """
        UPDATE tasks
        SET status = ?, attempt_count = ?, last_error = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
        WHERE task_id = ?
        """,
        (status, next_attempts, error, now, row["task_id"]),
    )
    log_event(
        conn,
        task_id=row["task_id"],
        target_date_local=row["target_date_local"],
        event_type="prepare_failed",
        details={"status": status, "attempt_count": next_attempts, "error": error},
    )
    conn.commit()


def prepare_progress_callback(reporter: Any | None, worker_id: str | None):
    if reporter is None or worker_id is None:
        return lambda _event_name, _payload: None

    def callback(event_name: str, payload: dict[str, object]) -> None:
        try:
            if event_name == "start":
                file_label = str(payload.get("file_label") or "raw.grib2")
                total_bytes = int(payload.get("total_bytes") or 0) or None
                if hasattr(reporter, "update_worker"):
                    reporter.update_worker(worker_id, phase="prepare", details="download")
                if hasattr(reporter, "start_transfer"):
                    reporter.start_transfer(worker_id, file_label=file_label, total_bytes=total_bytes)
                return
            if event_name == "progress":
                total_bytes = int(payload.get("total_bytes") or 0) or None
                downloaded = int(payload.get("downloaded_bytes") or 0)
                if hasattr(reporter, "update_transfer"):
                    reporter.update_transfer(worker_id, bytes_downloaded=downloaded, total_bytes=total_bytes)
                return
            if event_name == "complete":
                if hasattr(reporter, "finish_transfer"):
                    reporter.finish_transfer(worker_id)
                if hasattr(reporter, "update_worker"):
                    reporter.update_worker(worker_id, phase="prepare", details="checksum")
        except Exception:
            # Progress reporting must never affect queue correctness.
            return

    return callback


def prepare_one(
    conn: sqlite3.Connection,
    run_root: Path,
    row: sqlite3.Row,
    *,
    max_attempts: int,
    progress_callback=None,
) -> dict[str, Any]:
    task_id = str(row["task_id"])
    task = task_from_row(row)
    paths = task_paths(run_root, task_id)
    delete_path(paths["root"])
    paths["root"].mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["task_json"], asdict(task))
    try:
        fetch_result = download_subset_for_inventory_patterns(
            date=task.run_date_utc.replace("-", ""),
            cycle=task.cycle_hour_utc,
            product="surface",
            forecast_hour=task.forecast_hour,
            source="google",
            patterns=[pattern for _, pattern in hrrr.inventory_selection_patterns()],
            subset_path=paths["raw"],
            manifest_path=paths["manifest"],
            selection_manifest_path=paths["selection"],
            range_merge_gap_bytes=hrrr.DEFAULT_RANGE_MERGE_GAP_BYTES,
            overwrite=True,
            progress_callback=progress_callback,
        )
        checksum = sha256_file(paths["raw"])
        expected_bytes = paths["raw"].stat().st_size
        write_text_atomic(paths["checksum"], f"{checksum}  {paths['raw'].name}\n")
        metadata = {
            "task_id": task_id,
            "task": asdict(task),
            "checksum_sha256": checksum,
            "expected_bytes": expected_bytes,
            "remote_file_size": fetch_result.remote_file_size,
            "selected_record_count": fetch_result.selected_record_count,
            "merged_range_count": fetch_result.merged_range_count,
            "downloaded_range_bytes": fetch_result.downloaded_range_bytes,
            "raw_path": str(paths["raw"]),
            "manifest_path": str(paths["manifest"]),
            "selection_manifest_path": str(paths["selection"]),
        }
        write_json_atomic(paths["metadata"], metadata)
    except Exception as exc:
        delete_path(paths["root"])
        mark_prepare_failure(conn, row, error=str(exc), max_attempts=max_attempts)
        return {"task_id": task_id, "status": "failed", "error": str(exc)}

    begin_immediate(conn)
    now = utc_now_text()
    conn.execute(
        """
        UPDATE tasks
        SET status = 'ready', raw_dir = ?, last_error = NULL, updated_at = ?
        WHERE task_id = ? AND status = 'preparing'
        """,
        (str(paths["root"]), now, task_id),
    )
    log_event(
        conn,
        task_id=task_id,
        target_date_local=str(row["target_date_local"]),
        event_type="prepared",
        details={"expected_bytes": expected_bytes, "checksum_sha256": checksum},
    )
    conn.commit()
    return {"task_id": task_id, "status": "ready", "expected_bytes": expected_bytes}


def command_prepare(args: argparse.Namespace) -> int:
    ensure_layout(args.run_root)
    conn = connect(args.run_root)
    if is_paused(args.run_root):
        print(json.dumps({"status": "paused", "prepared": 0}, sort_keys=True))
        return 0
    prepared: list[dict[str, Any]] = []
    for _ in range(max(0, int(args.limit))):
        if is_paused(args.run_root):
            break
        row = claim_prepare_task(conn, max_attempts=args.max_attempts)
        if row is None:
            break
        prepared.append(prepare_one(conn, args.run_root, row, max_attempts=args.max_attempts, progress_callback=lambda _event, _payload: None))
    print(json.dumps({"status": "ok", "prepared": len([item for item in prepared if item["status"] == "ready"]), "items": prepared}, sort_keys=True))
    return 0


def prepare_claimed_task(run_root: Path, *, max_attempts: int, worker_id: str | None = None, reporter: Any | None = None) -> dict[str, Any]:
    conn = connect(run_root)
    row = claim_prepare_task(conn, max_attempts=max_attempts)
    if row is None:
        return {"status": "no_work"}
    task_id = str(row["task_id"])
    if reporter is not None and worker_id is not None:
        reporter.start_worker(worker_id, label=task_id, phase="prepare", group_id="prepare")
    try:
        result = prepare_one(
            conn,
            run_root,
            row,
            max_attempts=max_attempts,
            progress_callback=prepare_progress_callback(reporter, worker_id),
        )
    finally:
        if reporter is not None and worker_id is not None:
            # Do not use complete_worker for dashboard totals; queue status is the source of truth.
            reporter.retire_worker(worker_id, message=f"prepared {task_id}")
    return result


def prepare_batch(
    run_root: Path,
    *,
    limit: int,
    workers: int,
    max_attempts: int,
    reporter: Any | None = None,
) -> list[dict[str, Any]]:
    task_limit = max(0, int(limit))
    worker_count = max(1, int(workers))
    if task_limit <= 0:
        return []
    if worker_count == 1:
        results: list[dict[str, Any]] = []
        for index in range(task_limit):
            result = prepare_claimed_task(run_root, max_attempts=max_attempts, worker_id=f"prepare-{index}", reporter=reporter)
            if result.get("status") == "no_work":
                break
            results.append(result)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hrrr-relay-prepare") as executor:
        futures = [
            executor.submit(
                prepare_claimed_task,
                run_root,
                max_attempts=max_attempts,
                worker_id=f"prepare-{index}",
                reporter=reporter,
            )
            for index in range(task_limit)
        ]
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result.get("status") != "no_work":
                results.append(result)
        return results


def prepare_attempt_number(row: sqlite3.Row) -> int:
    return int(row["attempt_count"]) + 1


def command_lease(args: argparse.Namespace) -> int:
    ensure_layout(args.run_root)
    conn = connect(args.run_root)
    if is_paused(args.run_root):
        print(json.dumps({"status": "paused"}, sort_keys=True))
        return 0
    begin_immediate(conn)
    row = conn.execute(
        """
        SELECT * FROM tasks
        WHERE status = 'ready'
        ORDER BY target_date_local, task_id
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        conn.commit()
        print(json.dumps({"status": "no_work"}, sort_keys=True))
        return 0
    lease_expires = utc_now() + dt.timedelta(minutes=args.lease_minutes)
    now = utc_now_text()
    conn.execute(
        """
        UPDATE tasks
        SET status = 'leased', lease_owner = ?, lease_expires_at = ?, updated_at = ?
        WHERE task_id = ? AND status = 'ready'
        """,
        (args.client_id, lease_expires.isoformat(), now, row["task_id"]),
    )
    log_event(
        conn,
        task_id=row["task_id"],
        target_date_local=row["target_date_local"],
        event_type="leased",
        details={"client_id": args.client_id, "lease_expires_at": lease_expires.isoformat()},
    )
    conn.commit()

    task_id = str(row["task_id"])
    metadata_path = task_paths(args.run_root, task_id)["metadata"]
    metadata = json.loads(metadata_path.read_text())
    payload = {
        "status": "leased",
        "task_id": task_id,
        "target_date_local": row["target_date_local"],
        "attempt_count": prepare_attempt_number(row),
        "lease_owner": args.client_id,
        "lease_expires_at": lease_expires.isoformat(),
        "raw_dir": str(task_raw_dir(args.run_root, task_id)),
        "raw_path": metadata["raw_path"],
        "manifest_path": metadata["manifest_path"],
        "selection_manifest_path": metadata["selection_manifest_path"],
        "checksum_sha256": metadata["checksum_sha256"],
        "expected_bytes": metadata["expected_bytes"],
        "task": metadata["task"],
        "upload_dir": str(uploads_root(args.run_root) / task_id / f"{args.client_id}-attempt-{prepare_attempt_number(row)}"),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_heartbeat(args: argparse.Namespace) -> int:
    conn = connect(args.run_root)
    begin_immediate(conn)
    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (args.task_id,)).fetchone()
    if row is None or row["status"] != "leased" or row["lease_owner"] != args.client_id:
        conn.commit()
        print(json.dumps({"status": "not_leased"}, sort_keys=True))
        return 1
    lease_expires = utc_now() + dt.timedelta(minutes=args.lease_minutes)
    conn.execute(
        "UPDATE tasks SET lease_expires_at = ?, updated_at = ? WHERE task_id = ?",
        (lease_expires.isoformat(), utc_now_text(), args.task_id),
    )
    log_event(conn, task_id=args.task_id, target_date_local=row["target_date_local"], event_type="heartbeat", details={"client_id": args.client_id})
    conn.commit()
    print(json.dumps({"status": "ok", "lease_expires_at": lease_expires.isoformat()}, sort_keys=True))
    return 0


def validate_upload(upload_dir: Path, task: hrrr.TaskSpec) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    complete_marker = upload_dir / "result.complete"
    row_path = upload_dir / "row.json"
    diagnostics_path = upload_dir / "diagnostics.json"
    provenance_path = upload_dir / "provenance.jsonl"
    if not complete_marker.exists():
        raise ValueError(f"missing result.complete in {upload_dir}")
    if not row_path.exists():
        raise ValueError(f"missing row.json in {upload_dir}")
    if not diagnostics_path.exists():
        raise ValueError(f"missing diagnostics.json in {upload_dir}")
    row = json.loads(row_path.read_text())
    diagnostics = json.loads(diagnostics_path.read_text())
    expected = {
        "task_key": task.key,
        "target_date_local": task.target_date_local,
        "run_date_utc": task.run_date_utc,
        "cycle_hour_utc": task.cycle_hour_utc,
        "forecast_hour": task.forecast_hour,
        "init_time_utc": task.init_time_utc,
        "valid_time_utc": task.valid_time_utc,
        "init_time_local": task.init_time_local,
        "valid_time_local": task.valid_time_local,
    }
    for key, expected_value in expected.items():
        actual_value = row.get(key)
        if str(actual_value) != str(expected_value):
            raise ValueError(f"row {key} mismatch: expected {expected_value}, got {actual_value}")
    provenance: list[dict[str, Any]] = []
    if provenance_path.exists():
        for line in provenance_path.read_text().splitlines():
            if line.strip():
                provenance.append(json.loads(line))
    return row, provenance, diagnostics


def promote_result(run_root: Path, task_id: str, upload_dir: Path, row: dict[str, Any], provenance: list[dict[str, Any]], diagnostics: dict[str, Any]) -> Path:
    result_dir = task_result_dir(run_root, task_id)
    tmp_dir = result_dir.with_name(result_dir.name + ".tmp")
    delete_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(tmp_dir / "row.json", row)
    write_json_atomic(tmp_dir / "diagnostics.json", diagnostics)
    if provenance:
        write_text_atomic(tmp_dir / "provenance.jsonl", "".join(json.dumps(item, sort_keys=True) + "\n" for item in provenance))
    else:
        write_text_atomic(tmp_dir / "provenance.jsonl", "")
    write_json_atomic(tmp_dir / "upload.json", {"source_upload_dir": str(upload_dir), "promoted_at": utc_now_text()})
    delete_path(result_dir)
    tmp_dir.replace(result_dir)
    return result_dir


def command_ack(args: argparse.Namespace) -> int:
    ensure_layout(args.run_root)
    conn = connect(args.run_root)
    begin_immediate(conn)
    task_row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (args.task_id,)).fetchone()
    if task_row is None or task_row["status"] != "leased" or task_row["lease_owner"] != args.client_id:
        conn.commit()
        print(json.dumps({"status": "not_leased"}, sort_keys=True))
        return 1
    conn.commit()

    task = task_from_row(task_row)
    row, provenance, diagnostics = validate_upload(args.upload_dir, task)
    result_dir = promote_result(args.run_root, args.task_id, args.upload_dir, row, provenance, diagnostics)

    begin_immediate(conn)
    cursor = conn.execute(
        """
        UPDATE tasks
        SET status = 'completed', result_dir = ?, lease_owner = NULL, lease_expires_at = NULL, last_error = NULL, updated_at = ?
        WHERE task_id = ? AND status = 'leased' AND lease_owner = ?
        """,
        (str(result_dir), utc_now_text(), args.task_id, args.client_id),
    )
    if cursor.rowcount != 1:
        conn.rollback()
        delete_path(result_dir)
        print(json.dumps({"status": "not_leased"}, sort_keys=True))
        return 1
    log_event(conn, task_id=args.task_id, target_date_local=task_row["target_date_local"], event_type="completed", details={"client_id": args.client_id})
    conn.commit()
    delete_path(task_raw_dir(args.run_root, args.task_id))

    try:
        finalize_result = finalize_day(args.run_root, str(task_row["target_date_local"]), allow_incomplete=True)
    except Exception as exc:
        finalize_result = {"status": "error", "error": str(exc)}
        with conn:
            conn.execute(
                "UPDATE days SET last_error = ?, updated_at = ? WHERE target_date_local = ?",
                (str(exc), utc_now_text(), task_row["target_date_local"]),
            )
            log_event(
                conn,
                task_id=None,
                target_date_local=task_row["target_date_local"],
                event_type="day_finalize_failed",
                details={"error": str(exc), "trigger_task_id": args.task_id},
            )
    print(json.dumps({"status": "ok", "task_id": args.task_id, "finalize": finalize_result}, sort_keys=True))
    return 0


def command_retry(args: argparse.Namespace) -> int:
    ensure_layout(args.run_root)
    conn = connect(args.run_root)
    begin_immediate(conn)
    row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (args.task_id,)).fetchone()
    if row is None:
        conn.commit()
        print(json.dumps({"status": "missing_task"}, sort_keys=True))
        return 1
    if row["status"] != "leased" or row["lease_owner"] != args.client_id:
        conn.commit()
        print(json.dumps({"status": "not_leased"}, sort_keys=True))
        return 1
    next_attempts = int(row["attempt_count"]) + 1
    status = "failed_terminal" if next_attempts >= args.max_attempts else "retry_pending"
    conn.execute(
        """
        UPDATE tasks
        SET status = ?, attempt_count = ?, last_error = ?, lease_owner = NULL, lease_expires_at = NULL, updated_at = ?
        WHERE task_id = ?
        """,
        (status, next_attempts, args.reason, utc_now_text(), args.task_id),
    )
    log_event(
        conn,
        task_id=args.task_id,
        target_date_local=row["target_date_local"],
        event_type="retry_requested",
        details={"client_id": args.client_id, "reason": args.reason, "status": status, "attempt_count": next_attempts},
    )
    conn.commit()
    delete_path(task_raw_dir(args.run_root, args.task_id))
    print(json.dumps({"status": status, "attempt_count": next_attempts}, sort_keys=True))
    return 0


def recover_row_status(row: sqlite3.Row, *, max_attempts: int) -> tuple[str, int]:
    next_attempts = int(row["attempt_count"]) + 1
    status = "failed_terminal" if next_attempts >= max_attempts else "retry_pending"
    return status, next_attempts


def recover_expired(
    run_root: Path,
    *,
    max_attempts: int,
    prepare_minutes: float,
    result_uploaded_minutes: float,
) -> list[str]:
    ensure_layout(run_root)
    conn = connect(run_root)
    now = utc_now()
    recovered: list[str] = []
    begin_immediate(conn)
    rows = conn.execute("SELECT * FROM tasks WHERE status IN ('leased', 'preparing', 'result_uploaded')").fetchall()
    for row in rows:
        updated_at = parse_utc(row["updated_at"])
        expires_at = parse_utc(row["lease_expires_at"])
        stale_preparing = row["status"] == "preparing" and updated_at is not None and updated_at < now - dt.timedelta(minutes=prepare_minutes)
        stale_uploaded = row["status"] == "result_uploaded" and updated_at is not None and updated_at < now - dt.timedelta(minutes=result_uploaded_minutes)
        expired_lease = row["status"] == "leased" and expires_at is not None and expires_at < now
        if not (stale_preparing or stale_uploaded or expired_lease):
            continue
        status, next_attempts = recover_row_status(row, max_attempts=max_attempts)
        reason = {
            "leased": "lease_expired",
            "preparing": "prepare_expired",
            "result_uploaded": "result_upload_expired",
        }[str(row["status"])]
        conn.execute(
            """
            UPDATE tasks
            SET status = ?, attempt_count = ?, lease_owner = NULL, lease_expires_at = NULL,
                last_error = ?, updated_at = ?
            WHERE task_id = ?
            """,
            (status, next_attempts, reason, utc_now_text(), row["task_id"]),
        )
        log_event(
            conn,
            task_id=row["task_id"],
            target_date_local=row["target_date_local"],
            event_type=reason,
            details={"status": status, "attempt_count": next_attempts, "previous_status": row["status"]},
        )
        if row["status"] in {"preparing", "result_uploaded"}:
            delete_path(task_raw_dir(run_root, str(row["task_id"])))
        recovered.append(str(row["task_id"]))
    conn.commit()
    return recovered


def command_recover_expired(args: argparse.Namespace) -> int:
    recovered = recover_expired(
        args.run_root,
        max_attempts=args.max_attempts,
        prepare_minutes=args.prepare_minutes,
        result_uploaded_minutes=args.result_uploaded_minutes,
    )
    print(json.dumps({"status": "ok", "recovered": recovered}, sort_keys=True))
    return 0


def hrrr_summary_paths(run_root: Path, target_date_local: str) -> dict[str, Path]:
    summary_root = run_root / "hrrr_summary" / f"target_date_local={target_date_local}"
    state_root = run_root / "hrrr_summary_state" / f"target_date_local={target_date_local}"
    return {
        "summary_root": summary_root,
        "state_root": state_root,
        "summary": summary_root / "hrrr.overnight.parquet",
        "manifest_json": state_root / "hrrr.manifest.json",
        "manifest_parquet": state_root / "hrrr.manifest.parquet",
    }


def load_result_rows(conn: sqlite3.Connection, run_root: Path, target_date_local: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[str]]:
    rows = conn.execute(
        "SELECT task_id, result_dir FROM tasks WHERE target_date_local = ? AND status = 'completed' ORDER BY task_id",
        (target_date_local,),
    ).fetchall()
    data_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, dict[str, Any]] = {}
    completed_keys: list[str] = []
    for row in rows:
        task_id = str(row["task_id"])
        result_dir = Path(str(row["result_dir"])) if row["result_dir"] else task_result_dir(run_root, task_id)
        data_rows.append(json.loads((result_dir / "row.json").read_text()))
        diagnostics[task_id] = json.loads((result_dir / "diagnostics.json").read_text())
        completed_keys.append(task_id)
    return data_rows, diagnostics, completed_keys


def finalize_day(run_root: Path, target_date_local: str, *, allow_incomplete: bool = False) -> dict[str, Any]:
    conn = connect(run_root)
    day = conn.execute("SELECT * FROM days WHERE target_date_local = ?", (target_date_local,)).fetchone()
    if day is None:
        return {"status": "missing_day", "target_date_local": target_date_local}
    expected_count = int(day["expected_task_count"])
    completed_count = task_count_for_day(conn, target_date_local, status="completed")
    if completed_count != expected_count:
        payload = {"status": "not_ready", "target_date_local": target_date_local, "completed": completed_count, "expected": expected_count}
        if allow_incomplete:
            return payload
        raise ValueError(f"target_date_local={target_date_local} completed={completed_count} expected={expected_count}")

    rows, diagnostics, completed_keys = load_result_rows(conn, run_root, target_date_local)
    if len(rows) != expected_count:
        raise ValueError(f"target_date_local={target_date_local} loaded_rows={len(rows)} expected={expected_count}")

    summary_row = hrrr.build_summary_row(target_date_local, rows)
    paths = hrrr_summary_paths(run_root, target_date_local)
    month_id = target_date_local[:7]
    manifest = {
        "month": month_id,
        "target_date_local": target_date_local,
        "selection_mode": str(day["selection_mode"]),
        "expected_task_count": expected_count,
        "expected_task_keys": sorted(completed_keys),
        "completed_task_keys": sorted(completed_keys),
        "failure_reasons": {},
        "missing_fields": {},
        "source_model": hrrr.DEFAULT_SOURCE_MODEL,
        "source_product": hrrr.DEFAULT_SOURCE_PRODUCT,
        "source_version": hrrr.DEFAULT_SOURCE_VERSION,
        "wide_parquet_path": None,
        "provenance_path": None,
        "summary_parquet_path": str(paths["summary"]),
        "manifest_json_path": str(paths["manifest_json"]),
        "manifest_parquet_path": str(paths["manifest_parquet"]),
        "keep_downloads": False,
        "keep_reduced": False,
        "task_diagnostics": diagnostics,
        "complete": True,
        "finalized_at": utc_now_text(),
    }
    write_parquet_atomic(paths["summary"], pd.DataFrame([summary_row]))
    write_json_atomic(paths["manifest_json"], manifest)
    manifest_df = pd.DataFrame.from_records(hrrr.manifest_records(month_id, manifest))
    manifest_df["target_date_local"] = target_date_local
    manifest_df["selection_mode"] = str(day["selection_mode"])
    write_parquet_atomic(paths["manifest_parquet"], manifest_df)

    with conn:
        conn.execute(
            "UPDATE days SET status = 'completed', finalized_at = ?, last_error = NULL, updated_at = ? WHERE target_date_local = ?",
            (manifest["finalized_at"], utc_now_text(), target_date_local),
        )
        log_event(conn, task_id=None, target_date_local=target_date_local, event_type="day_finalized", details={"completed": completed_count})
    return {"status": "ok", "target_date_local": target_date_local, "completed": completed_count}


def command_finalize(args: argparse.Namespace) -> int:
    result = finalize_day(args.run_root, args.target_local_date)
    print(json.dumps(result, sort_keys=True))
    return 0


def command_pause(args: argparse.Namespace) -> int:
    ensure_layout(args.run_root)
    write_text_atomic(pause_path(args.run_root), utc_now_text() + "\n")
    print(json.dumps({"status": "paused"}, sort_keys=True))
    return 0


def command_resume(args: argparse.Namespace) -> int:
    pause_path(args.run_root).unlink(missing_ok=True)
    print(json.dumps({"status": "resumed"}, sort_keys=True))
    return 0


def count_by_status(conn: sqlite3.Connection, target_date_local: str | None = None) -> dict[str, int]:
    if target_date_local is None:
        rows = conn.execute("SELECT status, COUNT(*) AS count FROM tasks GROUP BY status").fetchall()
    else:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM tasks WHERE target_date_local = ? GROUP BY status",
            (target_date_local,),
        ).fetchall()
    counts = {status: 0 for status in sorted(TASK_STATUSES)}
    counts.update({str(row["status"]): int(row["count"]) for row in rows})
    return counts


def command_status(args: argparse.Namespace) -> int:
    conn = connect(args.run_root)
    day_rows = conn.execute("SELECT status, COUNT(*) AS count FROM days GROUP BY status").fetchall()
    active_clients = [
        {"client_id": row["lease_owner"], "leased": int(row["count"])}
        for row in conn.execute(
            "SELECT lease_owner, COUNT(*) AS count FROM tasks WHERE status = 'leased' GROUP BY lease_owner"
        ).fetchall()
    ]
    payload = {
        "status": "ok",
        "paused": is_paused(args.run_root),
        "tasks": count_by_status(conn),
        "days": {str(row["status"]): int(row["count"]) for row in day_rows},
        "active_clients": active_clients,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def command_day_status(args: argparse.Namespace) -> int:
    conn = connect(args.run_root)
    day = conn.execute("SELECT * FROM days WHERE target_date_local = ?", (args.target_local_date,)).fetchone()
    if day is None:
        print(json.dumps({"status": "missing_day", "target_date_local": args.target_local_date}, sort_keys=True))
        return 1
    payload = {
        "status": "ok",
        "target_date_local": args.target_local_date,
        "day_status": day["status"],
        "selection_mode": day["selection_mode"],
        "expected_task_count": day["expected_task_count"],
        "finalized_at": day["finalized_at"],
        "last_error": day["last_error"],
        "tasks": count_by_status(conn, args.target_local_date),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def count_by_status_between(conn: sqlite3.Connection, *, start_local_date: str, end_local_date: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM tasks
        WHERE target_date_local BETWEEN ? AND ?
        GROUP BY status
        """,
        (start_local_date, end_local_date),
    ).fetchall()
    counts = {status: 0 for status in sorted(TASK_STATUSES)}
    counts.update({str(row["status"]): int(row["count"]) for row in rows})
    return counts


def day_counts_between(conn: sqlite3.Connection, *, start_local_date: str, end_local_date: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT status, COUNT(*) AS count
        FROM days
        WHERE target_date_local BETWEEN ? AND ?
        GROUP BY status
        """,
        (start_local_date, end_local_date),
    ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def active_clients_between(conn: sqlite3.Connection, *, start_local_date: str, end_local_date: str) -> list[dict[str, Any]]:
    return [
        {"client_id": row["lease_owner"], "leased": int(row["count"])}
        for row in conn.execute(
            """
            SELECT lease_owner, COUNT(*) AS count
            FROM tasks
            WHERE status = 'leased' AND target_date_local BETWEEN ? AND ?
            GROUP BY lease_owner
            """,
            (start_local_date, end_local_date),
        ).fetchall()
    ]


def recent_events_between(conn: sqlite3.Connection, *, start_local_date: str, end_local_date: str, limit: int = 8) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT task_id, target_date_local, timestamp, event_type, details_json
        FROM events
        WHERE target_date_local BETWEEN ? AND ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (start_local_date, end_local_date, int(limit)),
    ).fetchall()
    return [
        {
            "task_id": row["task_id"],
            "target_date_local": row["target_date_local"],
            "timestamp": row["timestamp"],
            "event_type": row["event_type"],
            "details": json.loads(row["details_json"] or "{}"),
        }
        for row in rows
    ]


def status_snapshot(run_root: Path, *, start_local_date: str, end_local_date: str) -> dict[str, Any]:
    ensure_layout(run_root)
    conn = connect(run_root)
    tasks = count_by_status_between(conn, start_local_date=start_local_date, end_local_date=end_local_date)
    days = day_counts_between(conn, start_local_date=start_local_date, end_local_date=end_local_date)
    total_tasks = sum(tasks.values())
    total_days = sum(days.values())
    completed_days = int(days.get("completed", 0))
    runnable = sum(int(tasks.get(status, 0)) for status in ("pending", "retry_pending", "preparing", "ready", "leased"))
    complete = total_days > 0 and completed_days == total_days
    all_tasks_terminal = total_tasks > 0 and int(tasks.get("completed", 0)) + int(tasks.get("failed_terminal", 0)) == total_tasks
    failed_stalled = not complete and runnable == 0 and all_tasks_terminal
    return {
        "status": "ok",
        "paused": is_paused(run_root),
        "tasks": tasks,
        "days": days,
        "active_clients": active_clients_between(conn, start_local_date=start_local_date, end_local_date=end_local_date),
        "recent_events": recent_events_between(conn, start_local_date=start_local_date, end_local_date=end_local_date),
        "free_gb": free_gb(run_root),
        "total_tasks": total_tasks,
        "total_days": total_days,
        "completed_days": completed_days,
        "complete": complete,
        "failed_stalled": failed_stalled,
    }


def finalize_ready_days(run_root: Path, *, start_local_date: str, end_local_date: str) -> list[dict[str, Any]]:
    conn = connect(run_root)
    days = conn.execute(
        """
        SELECT *
        FROM days
        WHERE status != 'completed' AND target_date_local BETWEEN ? AND ?
        ORDER BY target_date_local
        """,
        (start_local_date, end_local_date),
    ).fetchall()
    finalized: list[dict[str, Any]] = []
    for day in days:
        target = str(day["target_date_local"])
        expected = int(day["expected_task_count"])
        completed = task_count_for_day(conn, target, status="completed")
        if completed != expected:
            continue
        try:
            finalized.append(finalize_day(run_root, target))
        except Exception as exc:
            with conn:
                conn.execute(
                    "UPDATE days SET last_error = ?, updated_at = ? WHERE target_date_local = ?",
                    (str(exc), utc_now_text(), target),
                )
                log_event(conn, task_id=None, target_date_local=target, event_type="day_finalize_failed", details={"error": str(exc)})
    return finalized


def update_run_reporter(reporter: Any, snapshot: dict[str, Any]) -> None:
    tasks = snapshot["tasks"]
    days = snapshot["days"]
    reporter.set_total(int(snapshot["total_tasks"]))
    reporter.set_metrics(
        ready=tasks.get("ready", 0),
        pending=tasks.get("pending", 0),
        retry=tasks.get("retry_pending", 0),
        leased=tasks.get("leased", 0),
        completed=tasks.get("completed", 0),
        failed=tasks.get("failed_terminal", 0),
        clients=len(snapshot["active_clients"]),
        days=f"{snapshot['completed_days']}/{snapshot['total_days']}",
        free_gb=f"{float(snapshot['free_gb']):.1f}",
        paused=snapshot["paused"],
    )
    task_total = max(1, int(snapshot["total_tasks"]))
    for status in sorted(TASK_STATUSES):
        reporter.upsert_group(f"task:{status}", label=status, total=task_total, completed=int(tasks.get(status, 0)), status=f"{tasks.get(status, 0)}")
    day_total = max(1, int(snapshot["total_days"]))
    for status, count in sorted(days.items()):
        reporter.upsert_group(f"day:{status}", label=f"day {status}", total=day_total, completed=int(count), status=f"{count}")


def install_run_signal_handlers(stop_event: threading.Event, run_root: Path) -> None:
    def handle_signal(signum, _frame):
        write_text_atomic(pause_path(run_root), f"signal:{signum} {utc_now_text()}\n")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def command_run(args: argparse.Namespace) -> int:
    init_payload = init_range(
        args.run_root,
        start_local_date=args.start_local_date,
        end_local_date=args.end_local_date,
        selection_mode=args.selection_mode,
    )
    stop_event = threading.Event()

    def request_pause(*, reason: str = "operator") -> None:
        write_text_atomic(pause_path(args.run_root), f"{reason} {utc_now_text()}\n")
        stop_event.set()

    reporter = create_progress_reporter(
        "HRRR relay server",
        unit="task",
        mode=args.progress_mode,
        on_pause_request=request_pause,
        enable_dashboard_hotkeys=not bool(args.disable_dashboard_hotkeys),
        pause_control_file=args.pause_control_file,
    )
    reporter.log_event(f"initialized days={init_payload['inserted_days']} tasks={init_payload['inserted_tasks']}")
    install_run_signal_handlers(stop_event, args.run_root)

    last_recover_at = 0.0
    exit_status = 0
    final_status = "done"
    try:
        while True:
            finalized = finalize_ready_days(args.run_root, start_local_date=args.start_local_date, end_local_date=args.end_local_date)
            for item in finalized:
                reporter.log_event(f"finalized {item.get('target_date_local')}")

            snapshot = status_snapshot(args.run_root, start_local_date=args.start_local_date, end_local_date=args.end_local_date)
            update_run_reporter(reporter, snapshot)
            reporter.refresh()

            if snapshot["complete"]:
                final_status = "complete"
                break
            if snapshot["failed_stalled"]:
                exit_status = 1
                final_status = "failed_terminal"
                reporter.log_event("no runnable work remains and failed_terminal tasks exist", level="error")
                break
            if snapshot["paused"] or stop_event.is_set():
                reporter.mark_paused(reason="operator")
                final_status = "paused"
                break

            now = time.monotonic()
            if now - last_recover_at >= float(args.recover_seconds):
                recovered = recover_expired(
                    args.run_root,
                    max_attempts=args.max_attempts,
                    prepare_minutes=args.prepare_minutes,
                    result_uploaded_minutes=args.result_uploaded_minutes,
                )
                if recovered:
                    reporter.log_event(f"recovered={len(recovered)}")
                last_recover_at = now
                snapshot = status_snapshot(args.run_root, start_local_date=args.start_local_date, end_local_date=args.end_local_date)
                update_run_reporter(reporter, snapshot)

            ready = int(snapshot["tasks"].get("ready", 0))
            pending = int(snapshot["tasks"].get("pending", 0)) + int(snapshot["tasks"].get("retry_pending", 0))
            free = float(snapshot["free_gb"])
            if free < float(args.min_free_gb):
                reporter.log_event(f"low disk free_gb={free:.1f} min_free_gb={float(args.min_free_gb):.1f}", level="warn")
            elif ready < int(args.target_ready_tasks) and pending > 0:
                admission = min(
                    int(args.prepare_batch_size),
                    int(args.target_ready_tasks) - ready,
                    pending,
                )
                results = prepare_batch(
                    args.run_root,
                    limit=admission,
                    workers=args.prepare_workers,
                    max_attempts=args.max_attempts,
                    reporter=reporter,
                )
                prepared = len([item for item in results if item.get("status") == "ready"])
                failed = len([item for item in results if item.get("status") == "failed"])
                if prepared or failed:
                    reporter.log_event(f"prepared={prepared} failed={failed}")

            if args.once:
                final_status = "once"
                break
            stop_event.wait(float(args.poll_seconds))
    finally:
        reporter.close(status=final_status)
    return exit_status


def add_run_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Server-side HRRR selected-raw relay queue.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    add_run_root(init_parser)
    init_parser.add_argument("--start-local-date", required=True)
    init_parser.add_argument("--end-local-date", required=True)
    init_parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    init_parser.set_defaults(func=command_init)

    run_parser = subparsers.add_parser("run")
    add_run_root(run_parser)
    run_parser.add_argument("--start-local-date", required=True)
    run_parser.add_argument("--end-local-date", required=True)
    run_parser.add_argument("--selection-mode", default=DEFAULT_SELECTION_MODE)
    run_parser.add_argument("--prepare-workers", type=int, default=DEFAULT_PREPARE_WORKERS)
    run_parser.add_argument("--target-ready-tasks", type=int, default=DEFAULT_TARGET_READY_TASKS)
    run_parser.add_argument("--prepare-batch-size", type=int, default=DEFAULT_PREPARE_BATCH_SIZE)
    run_parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    run_parser.add_argument("--recover-seconds", type=float, default=DEFAULT_RECOVER_SECONDS)
    run_parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    run_parser.add_argument("--progress-mode", choices=("auto", "dashboard", "log"), default="auto")
    run_parser.add_argument("--pause-control-file")
    run_parser.add_argument("--disable-dashboard-hotkeys", action="store_true")
    run_parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    run_parser.add_argument("--prepare-minutes", type=float, default=30.0)
    run_parser.add_argument("--result-uploaded-minutes", type=float, default=30.0)
    run_parser.add_argument("--once", action="store_true")
    run_parser.set_defaults(func=command_run)

    prepare_parser = subparsers.add_parser("prepare")
    add_run_root(prepare_parser)
    prepare_parser.add_argument("--limit", type=int, default=8)
    prepare_parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    prepare_parser.set_defaults(func=command_prepare)

    lease_parser = subparsers.add_parser("lease")
    add_run_root(lease_parser)
    lease_parser.add_argument("--client-id", required=True)
    lease_parser.add_argument("--lease-minutes", type=float, default=30.0)
    lease_parser.set_defaults(func=command_lease)

    heartbeat_parser = subparsers.add_parser("heartbeat")
    add_run_root(heartbeat_parser)
    heartbeat_parser.add_argument("--task-id", required=True)
    heartbeat_parser.add_argument("--client-id", required=True)
    heartbeat_parser.add_argument("--lease-minutes", type=float, default=30.0)
    heartbeat_parser.set_defaults(func=command_heartbeat)

    ack_parser = subparsers.add_parser("ack")
    add_run_root(ack_parser)
    ack_parser.add_argument("--task-id", required=True)
    ack_parser.add_argument("--client-id", required=True)
    ack_parser.add_argument("--upload-dir", type=Path, required=True)
    ack_parser.set_defaults(func=command_ack)

    retry_parser = subparsers.add_parser("retry")
    add_run_root(retry_parser)
    retry_parser.add_argument("--task-id", required=True)
    retry_parser.add_argument("--client-id", required=True)
    retry_parser.add_argument("--reason", required=True)
    retry_parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    retry_parser.set_defaults(func=command_retry)

    recover_parser = subparsers.add_parser("recover-expired")
    add_run_root(recover_parser)
    recover_parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    recover_parser.add_argument("--prepare-minutes", type=float, default=30.0)
    recover_parser.add_argument("--result-uploaded-minutes", type=float, default=30.0)
    recover_parser.set_defaults(func=command_recover_expired)

    finalize_parser = subparsers.add_parser("finalize")
    add_run_root(finalize_parser)
    finalize_parser.add_argument("--target-local-date", required=True)
    finalize_parser.set_defaults(func=command_finalize)

    pause_parser = subparsers.add_parser("pause")
    add_run_root(pause_parser)
    pause_parser.set_defaults(func=command_pause)

    resume_parser = subparsers.add_parser("resume")
    add_run_root(resume_parser)
    resume_parser.set_defaults(func=command_resume)

    status_parser = subparsers.add_parser("status")
    add_run_root(status_parser)
    status_parser.set_defaults(func=command_status)

    day_status_parser = subparsers.add_parser("day-status")
    add_run_root(day_status_parser)
    day_status_parser.add_argument("--target-local-date", required=True)
    day_status_parser.set_defaults(func=command_day_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
