from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import pathlib
import sqlite3
import sys
from types import SimpleNamespace

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
HRRR_DIR = ROOT / "tools" / "hrrr"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HRRR_DIR) not in sys.path:
    sys.path.insert(0, str(HRRR_DIR))

RELAY_SERVER_PATH = HRRR_DIR / "relay_server.py"
HRRR_BACKFILL_PATH = HRRR_DIR / "run_hrrr_monthly_backfill.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


relay_server = load_module("hrrr_relay_server_test", RELAY_SERVER_PATH)
hrrr_backfill = load_module("hrrr_backfill_for_relay_test", HRRR_BACKFILL_PATH)


def run_root(tmp_path: pathlib.Path) -> pathlib.Path:
    return tmp_path / "runtime"


def init_one_day(tmp_path: pathlib.Path, target: str = "2026-04-12") -> pathlib.Path:
    root = run_root(tmp_path)
    args = argparse.Namespace(
        run_root=root,
        start_local_date=target,
        end_local_date=target,
        selection_mode="overnight_0005",
    )
    assert relay_server.command_init(args) == 0
    return root


def connection(root: pathlib.Path) -> sqlite3.Connection:
    return relay_server.connect(root)


class FakeReporter:
    def __init__(self, on_pause_request=None):
        self.on_pause_request = on_pause_request
        self.events: list[tuple[str, object]] = []
        self.metrics: dict[str, object] = {}
        self.closed_status: str | None = None
        self.paused = False

    def log_event(self, message, *, level="info"):
        self.events.append((level, message))

    def set_total(self, total):
        self.metrics["total"] = total

    def set_metrics(self, **metrics):
        self.metrics.update(metrics)

    def upsert_group(self, *_args, **_kwargs):
        return None

    def start_worker(self, *_args, **_kwargs):
        return None

    def retire_worker(self, *_args, **_kwargs):
        return None

    def refresh(self, *, force=False):
        return None

    def mark_paused(self, *, reason=None):
        self.paused = True
        self.events.append(("warn", f"paused:{reason}"))

    def close(self, *, status=None):
        self.closed_status = status


def run_args(tmp_path: pathlib.Path, **overrides):
    values = {
        "run_root": run_root(tmp_path),
        "start_local_date": "2026-04-12",
        "end_local_date": "2026-04-12",
        "selection_mode": "overnight_0005",
        "prepare_workers": 1,
        "target_ready_tasks": 1,
        "prepare_batch_size": 1,
        "poll_seconds": 0.01,
        "recover_seconds": 999.0,
        "min_free_gb": 0.0,
        "progress_mode": "log",
        "pause_control_file": None,
        "disable_dashboard_hotkeys": True,
        "max_attempts": 6,
        "prepare_minutes": 30.0,
        "result_uploaded_minutes": 30.0,
        "once": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def fake_download_subset_for_inventory_patterns(**kwargs):
    pathlib.Path(kwargs["subset_path"]).write_bytes(b"raw-bytes")
    pathlib.Path(kwargs["manifest_path"]).write_text("record_number,byte_start,byte_end,variable,level,valid_desc\n")
    pathlib.Path(kwargs["selection_manifest_path"]).write_text("selection_signature\nsig\n")
    return SimpleNamespace(
        remote_file_size=100,
        selected_record_count=2,
        merged_range_count=1,
        downloaded_range_bytes=9,
    )


def first_task(root: pathlib.Path):
    conn = connection(root)
    return conn.execute("SELECT * FROM tasks ORDER BY task_id LIMIT 1").fetchone()


def mark_task_ready(root: pathlib.Path, task_id: str) -> None:
    raw_dir = relay_server.task_raw_dir(root, task_id)
    raw_dir.mkdir(parents=True, exist_ok=True)
    relay_server.write_json_atomic(raw_dir / "metadata.json", {
        "task_id": task_id,
        "task": json.loads(first_task(root)["task_json"]),
        "checksum_sha256": "abc123",
        "expected_bytes": 12,
        "raw_path": str(raw_dir / "raw.grib2"),
        "manifest_path": str(raw_dir / "raw.manifest.csv"),
        "selection_manifest_path": str(raw_dir / "raw.selection.csv"),
    })
    (raw_dir / "raw.grib2").write_bytes(b"hello world\n")
    (raw_dir / "checksum.sha256").write_text("abc123  raw.grib2\n")
    with connection(root) as conn:
        conn.execute("UPDATE tasks SET status = 'ready', raw_dir = ? WHERE task_id = ?", (str(raw_dir), task_id))


def write_upload(upload_dir: pathlib.Path, task_id: str, *, target_date_local: str = "2026-04-12") -> None:
    upload_dir.mkdir(parents=True, exist_ok=True)
    task_json = connection(run_root_from_upload(upload_dir)).execute("SELECT task_json FROM tasks WHERE task_id = ?", (task_id,)).fetchone()["task_json"]
    task = relay_server.hrrr.TaskSpec(**json.loads(task_json))
    row = {
        "task_key": task_id,
        "target_date_local": target_date_local,
        "run_date_utc": task.run_date_utc,
        "cycle_hour_utc": task.cycle_hour_utc,
        "forecast_hour": task.forecast_hour,
        "init_time_utc": task.init_time_utc,
        "init_time_local": task.init_time_local,
        "valid_time_utc": task.valid_time_utc,
        "valid_time_local": task.valid_time_local,
        "init_date_local": task.init_date_local,
        "valid_date_local": task.valid_date_local,
        "init_hour_local": task.init_hour_local,
        "valid_hour_local": task.valid_hour_local,
        "cycle_rank_desc": task.cycle_rank_desc,
        "selected_for_summary": True,
        "anchor_cycle_candidate": task.anchor_cycle_candidate,
        "tmp_2m_k": 290.0,
    }
    (upload_dir / "row.json").write_text(json.dumps(row))
    (upload_dir / "diagnostics.json").write_text(json.dumps({"task_key": task_id}))
    (upload_dir / "provenance.jsonl").write_text("")
    (upload_dir / "result.complete").write_text("")


def run_root_from_upload(upload_dir: pathlib.Path) -> pathlib.Path:
    parts = upload_dir.parts
    idx = parts.index("hrrr_relay")
    return pathlib.Path(*parts[:idx])


def test_init_creates_expected_tasks_and_is_idempotent(tmp_path):
    root = init_one_day(tmp_path)
    conn = connection(root)
    first_count = conn.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]

    init_one_day(tmp_path)
    second_count = conn.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]
    day = conn.execute("SELECT * FROM days WHERE target_date_local = '2026-04-12'").fetchone()

    assert first_count > 0
    assert second_count == first_count
    assert day["selection_mode"] == "overnight_0005"
    assert day["expected_task_count"] == first_count


def test_init_reconciles_stale_tasks_for_existing_day(tmp_path):
    root = init_one_day(tmp_path)
    conn = connection(root)
    first_count = conn.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()["count"]
    task = first_task(root)
    stale_task_id = "stale-task"
    stale_raw_dir = relay_server.task_raw_dir(root, stale_task_id)
    stale_result_dir = relay_server.task_result_dir(root, stale_task_id)
    stale_raw_dir.mkdir(parents=True, exist_ok=True)
    stale_result_dir.mkdir(parents=True, exist_ok=True)
    (stale_raw_dir / "raw.grib2").write_bytes(b"stale")
    (stale_result_dir / "row.json").write_text("{}")
    with connection(root) as write_conn:
        write_conn.execute(
            """
            INSERT INTO tasks(
                task_id, target_date_local, task_json, status, attempt_count,
                raw_dir, result_dir, created_at, updated_at
            )
            VALUES (?, ?, ?, 'completed', 0, ?, ?, ?, ?)
            """,
            (
                stale_task_id,
                "2026-04-12",
                task["task_json"],
                str(stale_raw_dir),
                str(stale_result_dir),
                relay_server.utc_now_text(),
                relay_server.utc_now_text(),
            ),
        )

    init_one_day(tmp_path)

    rows = connection(root).execute("SELECT task_id FROM tasks WHERE target_date_local = ?", ("2026-04-12",)).fetchall()
    day = connection(root).execute("SELECT * FROM days WHERE target_date_local = ?", ("2026-04-12",)).fetchone()
    assert stale_task_id not in {row["task_id"] for row in rows}
    assert len(rows) == first_count
    assert day["expected_task_count"] == first_count
    assert not stale_raw_dir.exists()
    assert not stale_result_dir.exists()


def test_lease_is_atomic_and_returns_metadata(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    mark_task_ready(root, task["task_id"])

    args = argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)
    assert relay_server.command_lease(args) == 0
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-b", lease_minutes=30.0)) == 0

    conn = connection(root)
    leased = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task["task_id"],)).fetchone()
    assert leased["status"] == "leased"
    assert leased["lease_owner"] == "mac-a"
    assert relay_server.count_by_status(conn)["leased"] == 1


def test_prepare_writes_raw_files_and_marks_ready(tmp_path, monkeypatch):
    root = init_one_day(tmp_path)

    def fake_download_subset_for_inventory_patterns(**kwargs):
        pathlib.Path(kwargs["subset_path"]).write_bytes(b"raw-bytes")
        pathlib.Path(kwargs["manifest_path"]).write_text("record_number,byte_start,byte_end,variable,level,valid_desc\n")
        pathlib.Path(kwargs["selection_manifest_path"]).write_text("selection_signature\nsig\n")
        return SimpleNamespace(
            remote_file_size=100,
            selected_record_count=2,
            merged_range_count=1,
            downloaded_range_bytes=9,
        )

    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)

    assert relay_server.command_prepare(argparse.Namespace(run_root=root, limit=1, max_attempts=6)) == 0
    row = connection(root).execute("SELECT * FROM tasks WHERE status = 'ready'").fetchone()
    assert row is not None
    paths = relay_server.task_paths(root, row["task_id"])
    assert paths["raw"].read_bytes() == b"raw-bytes"
    assert paths["checksum"].exists()
    assert json.loads(paths["metadata"].read_text())["expected_bytes"] == 9


def test_ack_accepts_upload_deletes_raw_marks_completed_and_writes_event(tmp_path, monkeypatch):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    upload_dir = root / "hrrr_relay" / "uploads" / task_id / "mac-a-attempt-1"
    write_upload(upload_dir, task_id)
    monkeypatch.setattr(relay_server, "finalize_day", lambda *_args, **_kwargs: {"status": "not_ready"})

    assert relay_server.command_ack(argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", upload_dir=upload_dir)) == 0
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    events = connection(root).execute("SELECT event_type FROM events WHERE task_id = ? ORDER BY id", (task_id,)).fetchall()

    assert row["status"] == "completed"
    assert not relay_server.task_raw_dir(root, task_id).exists()
    assert (relay_server.task_result_dir(root, task_id) / "row.json").exists()
    assert "completed" in [event["event_type"] for event in events]


def test_ack_rechecks_lease_before_marking_completed(tmp_path, monkeypatch):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    upload_dir = root / "hrrr_relay" / "uploads" / task_id / "mac-a-attempt-1"
    write_upload(upload_dir, task_id)
    original_promote = relay_server.promote_result

    def promote_then_recover(run_root_arg, task_id_arg, upload_dir_arg, row, provenance, diagnostics):
        result_dir = original_promote(run_root_arg, task_id_arg, upload_dir_arg, row, provenance, diagnostics)
        with connection(root) as conn:
            conn.execute(
                "UPDATE tasks SET status = 'retry_pending', lease_owner = NULL, lease_expires_at = NULL WHERE task_id = ?",
                (task_id,),
            )
        return result_dir

    monkeypatch.setattr(relay_server, "promote_result", promote_then_recover)

    assert relay_server.command_ack(argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", upload_dir=upload_dir)) == 1
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "retry_pending"
    assert row["lease_owner"] is None
    assert not relay_server.task_result_dir(root, task_id).exists()


def test_retry_requeues_then_marks_terminal_after_max_attempts(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    assert relay_server.command_retry(
        argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", reason="checksum_mismatch", max_attempts=2)
    ) == 0
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "retry_pending"
    assert row["attempt_count"] == 1

    with connection(root) as conn:
        conn.execute("UPDATE tasks SET status = 'leased', lease_owner = 'mac-a' WHERE task_id = ?", (task_id,))
    assert relay_server.command_retry(
        argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", reason="process_error", max_attempts=2)
    ) == 0
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "failed_terminal"
    assert row["attempt_count"] == 2


def test_retry_rejects_completed_or_unowned_tasks(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)

    with connection(root) as conn:
        conn.execute("UPDATE tasks SET status = 'completed', lease_owner = NULL WHERE task_id = ?", (task_id,))
    assert relay_server.command_retry(
        argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", reason="late_retry", max_attempts=2)
    ) == 1
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "completed"
    assert row["attempt_count"] == 0

    with connection(root) as conn:
        conn.execute("UPDATE tasks SET status = 'leased', lease_owner = 'mac-b' WHERE task_id = ?", (task_id,))
    assert relay_server.command_retry(
        argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", reason="wrong_owner", max_attempts=2)
    ) == 1
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "leased"
    assert row["lease_owner"] == "mac-b"


def test_recover_expired_moves_stale_lease_to_retry_pending(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    expired = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()
    with connection(root) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'leased', lease_owner = 'mac-a', lease_expires_at = ? WHERE task_id = ?",
            (expired, task_id),
        )

    assert relay_server.command_recover_expired(
        argparse.Namespace(run_root=root, max_attempts=6, prepare_minutes=30.0, result_uploaded_minutes=30.0)
    ) == 0
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert row["status"] == "retry_pending"
    assert row["attempt_count"] == 1
    assert row["last_error"] == "lease_expired"


def test_recover_expired_requeues_stale_preparing_and_result_uploaded(tmp_path):
    root = init_one_day(tmp_path)
    rows = connection(root).execute("SELECT * FROM tasks ORDER BY task_id LIMIT 2").fetchall()
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=2)).isoformat()
    for status, row in zip(["preparing", "result_uploaded"], rows):
        relay_server.task_raw_dir(root, row["task_id"]).mkdir(parents=True, exist_ok=True)
        (relay_server.task_raw_dir(root, row["task_id"]) / "raw.grib2").write_bytes(b"stale")
        with connection(root) as conn:
            conn.execute("UPDATE tasks SET status = ?, updated_at = ? WHERE task_id = ?", (status, old, row["task_id"]))

    assert relay_server.command_recover_expired(
        argparse.Namespace(run_root=root, max_attempts=6, prepare_minutes=30.0, result_uploaded_minutes=30.0)
    ) == 0

    recovered = connection(root).execute("SELECT * FROM tasks WHERE task_id IN (?, ?) ORDER BY task_id", (rows[0]["task_id"], rows[1]["task_id"])).fetchall()
    assert [row["status"] for row in recovered] == ["retry_pending", "retry_pending"]
    assert [row["last_error"] for row in recovered] == ["prepare_expired", "result_upload_expired"]
    assert not relay_server.task_raw_dir(root, rows[0]["task_id"]).exists()
    assert not relay_server.task_raw_dir(root, rows[1]["task_id"]).exists()


def test_pause_blocks_prepare_and_lease_but_allows_ack(tmp_path, monkeypatch):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0
    assert relay_server.command_pause(argparse.Namespace(run_root=root)) == 0

    assert relay_server.command_prepare(argparse.Namespace(run_root=root, limit=1, max_attempts=6)) == 0
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-b", lease_minutes=30.0)) == 0

    upload_dir = root / "hrrr_relay" / "uploads" / task_id / "mac-a-attempt-1"
    write_upload(upload_dir, task_id)
    monkeypatch.setattr(relay_server, "finalize_day", lambda *_args, **_kwargs: {"status": "not_ready"})
    assert relay_server.command_ack(argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", upload_dir=upload_dir)) == 0
    assert connection(root).execute("SELECT status FROM tasks WHERE task_id = ?", (task_id,)).fetchone()["status"] == "completed"


def test_ack_returns_success_when_finalize_fails_after_completion(tmp_path, monkeypatch, capsys):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    upload_dir = root / "hrrr_relay" / "uploads" / task_id / "mac-a-attempt-1"
    write_upload(upload_dir, task_id)

    def raise_finalize(*_args, **_kwargs):
        raise RuntimeError("summary mismatch")

    monkeypatch.setattr(relay_server, "finalize_day", raise_finalize)
    assert relay_server.command_ack(argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", upload_dir=upload_dir)) == 0

    payload = json.loads(capsys.readouterr().out.splitlines()[-1])
    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    day = connection(root).execute("SELECT * FROM days WHERE target_date_local = ?", (task["target_date_local"],)).fetchone()
    assert payload["finalize"]["status"] == "error"
    assert row["status"] == "completed"
    assert day["last_error"] == "summary mismatch"


def test_ack_rejects_upload_with_mismatched_task_metadata(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    task_id = task["task_id"]
    mark_task_ready(root, task_id)
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    upload_dir = root / "hrrr_relay" / "uploads" / task_id / "mac-a-attempt-1"
    write_upload(upload_dir, task_id)
    row = json.loads((upload_dir / "row.json").read_text())
    row["forecast_hour"] = int(row["forecast_hour"]) + 1
    (upload_dir / "row.json").write_text(json.dumps(row))

    try:
        relay_server.command_ack(argparse.Namespace(run_root=root, task_id=task_id, client_id="mac-a", upload_dir=upload_dir))
    except ValueError as exc:
        assert "forecast_hour mismatch" in str(exc)
    else:
        raise AssertionError("expected upload validation failure")

    db_row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
    assert db_row["status"] == "leased"


def test_finalize_writes_daily_outputs_and_existing_validator_accepts(tmp_path, monkeypatch):
    root = init_one_day(tmp_path)
    conn = connection(root)
    target = "2026-04-12"
    tasks = conn.execute("SELECT * FROM tasks WHERE target_date_local = ? ORDER BY task_id", (target,)).fetchall()
    for task in tasks:
        result_dir = relay_server.task_result_dir(root, task["task_id"])
        result_dir.mkdir(parents=True, exist_ok=True)
        write_upload(result_dir, task["task_id"], target_date_local=target)
        with conn:
            conn.execute("UPDATE tasks SET status = 'completed', result_dir = ? WHERE task_id = ?", (str(result_dir), task["task_id"]))

    monkeypatch.setattr(relay_server.hrrr, "build_summary_row", lambda target_date_local, rows: {"target_date_local": target_date_local, "status": "ok"})

    result = relay_server.finalize_day(root, target)
    assert result["status"] == "ok"
    day = hrrr_backfill.DayWindow(dt.date.fromisoformat(target))
    assert hrrr_backfill.validate_hrrr_day(root, day, selection_mode="overnight_0005") is True


def test_status_and_day_status_are_read_only(tmp_path):
    root = init_one_day(tmp_path)
    before = connection(root).execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]

    assert relay_server.command_status(argparse.Namespace(run_root=root)) == 0
    assert relay_server.command_day_status(argparse.Namespace(run_root=root, target_local_date="2026-04-12")) == 0

    after = connection(root).execute("SELECT COUNT(*) AS count FROM events").fetchone()["count"]
    assert after == before


def test_run_once_initializes_and_prepares_bounded_work(tmp_path, monkeypatch):
    reporter = FakeReporter()
    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(relay_server, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)

    assert relay_server.command_run(run_args(tmp_path, target_ready_tasks=1, prepare_batch_size=4, prepare_workers=2)) == 0

    counts = relay_server.count_by_status(connection(run_root(tmp_path)))
    assert counts["ready"] == 1
    assert counts["pending"] > 0
    assert reporter.closed_status == "once"


def test_run_once_respects_target_ready_tasks(tmp_path, monkeypatch):
    reporter = FakeReporter()
    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(relay_server, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)

    assert relay_server.command_run(run_args(tmp_path, target_ready_tasks=3, prepare_batch_size=16, prepare_workers=4)) == 0

    counts = relay_server.count_by_status(connection(run_root(tmp_path)))
    assert counts["ready"] == 3


def test_run_once_skips_prepare_under_low_disk(tmp_path, monkeypatch):
    reporter = FakeReporter()
    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(relay_server, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(relay_server, "free_gb", lambda _path: 1.0)

    assert relay_server.command_run(run_args(tmp_path, min_free_gb=10.0, target_ready_tasks=3, prepare_batch_size=3)) == 0

    counts = relay_server.count_by_status(connection(run_root(tmp_path)))
    assert counts["ready"] == 0
    assert counts["pending"] > 0
    assert any("low disk" in str(message) for _level, message in reporter.events)


def test_concurrent_prepare_does_not_double_claim_tasks(tmp_path, monkeypatch):
    seen_subset_paths: list[str] = []

    def fake_download(**kwargs):
        seen_subset_paths.append(str(kwargs["subset_path"]))
        return fake_download_subset_for_inventory_patterns(**kwargs)

    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download)
    root = init_one_day(tmp_path)

    results = relay_server.prepare_batch(root, limit=8, workers=4, max_attempts=6)

    ready_results = [item for item in results if item["status"] == "ready"]
    ready_rows = connection(root).execute("SELECT task_id FROM tasks WHERE status = 'ready'").fetchall()
    assert len(ready_results) == 8
    assert len(ready_rows) == 8
    assert len(seen_subset_paths) == len(set(seen_subset_paths))


def test_run_pause_callback_stops_new_prepare_and_marks_paused(tmp_path, monkeypatch):
    reporter_holder: dict[str, FakeReporter] = {}

    def fake_create_reporter(*_args, **kwargs):
        reporter = FakeReporter(on_pause_request=kwargs.get("on_pause_request"))
        reporter_holder["reporter"] = reporter
        if reporter.on_pause_request is not None:
            reporter.on_pause_request(reason="test")
        return reporter

    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(relay_server, "create_progress_reporter", fake_create_reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)

    assert relay_server.command_run(run_args(tmp_path, once=False)) == 0

    root = run_root(tmp_path)
    counts = relay_server.count_by_status(connection(root))
    assert relay_server.pause_path(root).exists()
    assert counts["ready"] == 0
    assert reporter_holder["reporter"].paused is True
    assert reporter_holder["reporter"].closed_status == "paused"


def test_run_recovers_expired_work_during_loop(tmp_path, monkeypatch):
    reporter = FakeReporter()
    root = init_one_day(tmp_path)
    task = first_task(root)
    expired = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=5)).isoformat()
    with connection(root) as conn:
        conn.execute(
            "UPDATE tasks SET status = 'leased', lease_owner = 'mac-a', lease_expires_at = ? WHERE task_id = ?",
            (expired, task["task_id"]),
        )
    monkeypatch.setattr(relay_server, "download_subset_for_inventory_patterns", fake_download_subset_for_inventory_patterns)
    monkeypatch.setattr(relay_server, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)

    assert relay_server.command_run(run_args(tmp_path, recover_seconds=0.0, target_ready_tasks=0)) == 0

    row = connection(root).execute("SELECT * FROM tasks WHERE task_id = ?", (task["task_id"],)).fetchone()
    assert row["status"] == "retry_pending"
    assert row["last_error"] == "lease_expired"


def test_run_exits_complete_when_days_already_finalized(tmp_path, monkeypatch):
    reporter = FakeReporter()
    root = init_one_day(tmp_path)
    with connection(root) as conn:
        conn.execute("UPDATE tasks SET status = 'completed' WHERE target_date_local = ?", ("2026-04-12",))
        conn.execute("UPDATE days SET status = 'completed', finalized_at = ? WHERE target_date_local = ?", (relay_server.utc_now_text(), "2026-04-12"))
    monkeypatch.setattr(relay_server, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_server, "install_run_signal_handlers", lambda *_args, **_kwargs: None)

    assert relay_server.command_run(run_args(tmp_path, once=False)) == 0
    assert reporter.closed_status == "complete"


def test_status_snapshot_reports_counts_clients_events_and_disk(tmp_path):
    root = init_one_day(tmp_path)
    task = first_task(root)
    mark_task_ready(root, task["task_id"])
    assert relay_server.command_lease(argparse.Namespace(run_root=root, client_id="mac-a", lease_minutes=30.0)) == 0

    snapshot = relay_server.status_snapshot(root, start_local_date="2026-04-12", end_local_date="2026-04-12")

    assert snapshot["tasks"]["leased"] == 1
    assert snapshot["days"]["pending"] == 1
    assert snapshot["active_clients"] == [{"client_id": "mac-a", "leased": 1}]
    assert snapshot["recent_events"]
    assert snapshot["free_gb"] > 0
