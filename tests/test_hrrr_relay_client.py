from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import signal
import sys
import threading
from types import SimpleNamespace


ROOT = pathlib.Path(__file__).resolve().parents[1]
HRRR_DIR = ROOT / "tools" / "hrrr"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HRRR_DIR) not in sys.path:
    sys.path.insert(0, str(HRRR_DIR))

RELAY_CLIENT_PATH = HRRR_DIR / "relay_client.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


relay_client = load_module("hrrr_relay_client_test", RELAY_CLIENT_PATH)


def args(tmp_path: pathlib.Path, **overrides):
    values = {
        "server": "worker@example",
        "remote_repo": pathlib.Path("/srv/modelexp"),
        "remote_run_root": pathlib.Path("data/runtime/backfill_overnight"),
        "local_work_root": tmp_path / "client",
        "client_id": "mac-a",
        "workers": 1,
        "pull_workers": 1,
        "min_free_gb": 5.0,
        "lease_minutes": 30.0,
        "wgrib2_path": "wgrib2",
        "poll_seconds": 0.01,
        "heartbeat_seconds": 0.01,
        "rsync_path": "rsync",
        "ssh_path": "ssh",
        "max_local_task_seconds": 3600.0,
        "progress_mode": "log",
        "pause_control_file": None,
        "disable_dashboard_hotkeys": True,
        "server_status_seconds": 0.0,
        "keep_active_on_failure": False,
        "once": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def lease_payload(tmp_path: pathlib.Path, *, task_id: str = "task-1", raw_bytes: bytes = b"raw") -> dict[str, object]:
    raw_dir = tmp_path / "server_raw" / task_id
    upload_dir = tmp_path / "server_upload" / task_id / "mac-a-attempt-1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return {
        "status": "leased",
        "task_id": task_id,
        "target_date_local": "2026-04-12",
        "attempt_count": 1,
        "lease_owner": "mac-a",
        "lease_expires_at": "2026-04-22T00:00:00+00:00",
        "raw_dir": str(raw_dir),
        "raw_path": str(raw_dir / "raw.grib2"),
        "manifest_path": str(raw_dir / "raw.manifest.csv"),
        "selection_manifest_path": str(raw_dir / "raw.selection.csv"),
        "checksum_sha256": relay_client.hashlib.sha256(raw_bytes).hexdigest(),
        "expected_bytes": len(raw_bytes),
        "upload_dir": str(upload_dir),
        "task": {
            "target_date_local": "2026-04-12",
            "run_date_utc": "2026-04-12",
            "cycle_hour_utc": 0,
            "forecast_hour": 1,
            "init_time_utc": "2026-04-12T00:00:00+00:00",
            "init_time_local": "2026-04-11T20:00:00-04:00",
            "valid_time_utc": "2026-04-12T01:00:00+00:00",
            "valid_time_local": "2026-04-11T21:00:00-04:00",
            "init_date_local": "2026-04-11",
            "valid_date_local": "2026-04-11",
            "init_hour_local": 20,
            "valid_hour_local": 21,
            "cycle_rank_desc": 0,
            "selected_for_summary": True,
            "anchor_cycle_candidate": True,
        },
    }


class FakeReporter:
    def __init__(self, on_pause_request=None):
        self.on_pause_request = on_pause_request
        self.events: list[tuple[str, str]] = []
        self.metrics: dict[str, object] = {}
        self.workers: dict[str, dict[str, object]] = {}
        self.closed_status: str | None = None
        self.paused = False

    def log_event(self, message, *, level="info"):
        self.events.append((level, str(message)))

    def set_metrics(self, **metrics):
        self.metrics.update(metrics)

    def set_total(self, total):
        self.metrics["total"] = total

    def start_worker(self, worker_id, *, label, phase=None, group_id=None, details=None):
        self.workers[worker_id] = {"label": label, "phase": phase, "details": details}

    def update_worker(self, worker_id, *, phase=None, label=None, details=None):
        worker = self.workers.setdefault(worker_id, {})
        if phase is not None:
            worker["phase"] = phase
        if label is not None:
            worker["label"] = label
        if details is not None:
            worker["details"] = details

    def retire_worker(self, worker_id, *, message=None):
        self.workers.setdefault(worker_id, {})["retired"] = True

    def record_outcome(self, outcome="completed", *, count=1, message=None):
        self.events.append((outcome, str(message)))

    def start_transfer(self, worker_id, *, file_label, total_bytes=None):
        worker = self.workers.setdefault(worker_id, {})
        worker["transfer_file"] = file_label
        worker["transfer_total"] = total_bytes

    def update_transfer(self, worker_id, *, bytes_downloaded, total_bytes=None):
        worker = self.workers.setdefault(worker_id, {})
        worker["transfer_downloaded"] = bytes_downloaded
        if total_bytes is not None:
            worker["transfer_total"] = total_bytes

    def finish_transfer(self, worker_id):
        self.workers.setdefault(worker_id, {})["transfer_finished"] = True

    def mark_paused(self, *, reason=None):
        self.paused = True
        self.events.append(("warn", f"paused:{reason}"))

    def close(self, *, status=None):
        self.closed_status = status


def test_builds_ssh_command_for_server_actions(tmp_path):
    command = relay_client.build_remote_command(
        args(tmp_path),
        "lease",
        ["--client-id", "mac-a", "--lease-minutes", "30.0"],
    )

    assert command[0] == "ssh"
    assert command[1] == "worker@example"
    assert "cd /srv/modelexp &&" in command[2]
    assert "tools/hrrr/relay_server.py lease" in command[2]
    assert "--run-root data/runtime/backfill_overnight" in command[2]
    assert "--client-id mac-a" in command[2]


def test_parser_accepts_progress_flags_and_list_local(tmp_path):
    parser = relay_client.build_parser()

    run_args = parser.parse_args(
        [
            "run",
            "--server",
            "worker@example",
            "--remote-repo",
            "/srv/modelexp",
            "--progress-mode",
            "dashboard",
            "--pause-control-file",
            "/tmp/hrrr-client.pause",
            "--disable-dashboard-hotkeys",
            "--pull-workers",
            "2",
            "--server-status-seconds",
            "5",
            "--keep-active-on-failure",
            "--client-id",
            "mac-a",
        ]
    )
    list_args = parser.parse_args(["list-local", "--local-work-root", str(tmp_path / "client")])

    assert run_args.progress_mode == "dashboard"
    assert run_args.pause_control_file == "/tmp/hrrr-client.pause"
    assert run_args.disable_dashboard_hotkeys is True
    assert run_args.pull_workers == 2
    assert run_args.server_status_seconds == 5
    assert run_args.keep_active_on_failure is True
    assert list_args.func is relay_client.command_list_local


def test_builds_rsync_pull_and_upload_commands(tmp_path):
    client_args = args(tmp_path)

    pull = relay_client.build_rsync_pull_command(client_args, remote_dir="/remote/raw/task-1", local_dir=tmp_path / "raw")
    upload = relay_client.build_rsync_upload_command(
        client_args,
        local_upload_dir=tmp_path / "upload",
        remote_upload_dir="/remote/upload/task-1",
    )

    assert pull == ["rsync", "-aP", "worker@example:/remote/raw/task-1/", str(tmp_path / "raw") + "/"]
    assert upload == ["rsync", "-aP", str(tmp_path / "upload") + "/", "worker@example:/remote/upload/task-1/"]

    relative_pull = relay_client.build_rsync_pull_command(
        client_args,
        remote_dir="data/runtime/backfill_overnight/hrrr_relay/raw/task-1",
        local_dir=tmp_path / "raw",
    )
    relative_upload = relay_client.build_rsync_upload_command(
        client_args,
        local_upload_dir=tmp_path / "upload",
        remote_upload_dir="data/runtime/backfill_overnight/hrrr_relay/uploads/task-1/mac-a-attempt-1",
    )

    assert relative_pull[2] == "worker@example:/srv/modelexp/data/runtime/backfill_overnight/hrrr_relay/raw/task-1/"
    assert relative_upload[3] == "worker@example:/srv/modelexp/data/runtime/backfill_overnight/hrrr_relay/uploads/task-1/mac-a-attempt-1/"


def test_current_transfer_size_detects_rsync_hidden_temp_file(tmp_path):
    progress_path = tmp_path / "raw" / "raw.grib2"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    hidden = progress_path.parent / ".raw.grib2.ABC123"
    hidden.write_bytes(b"x" * 1234)

    assert relay_client.current_transfer_size(progress_path) == 1234


def test_prepare_local_task_dir_writes_lease_and_layout(tmp_path):
    lease = lease_payload(tmp_path)
    paths = relay_client.prepare_local_task_dir(tmp_path / "client", lease)

    assert (paths["root"] / "lease.json").exists()
    assert paths["raw"].is_dir()
    assert paths["reduced"].is_dir()
    assert paths["scratch"].is_dir()
    assert paths["upload"].is_dir()


def test_checksum_mismatch_calls_retry_and_skips_processing(tmp_path, monkeypatch):
    client_args = args(tmp_path)
    lease = lease_payload(tmp_path, raw_bytes=b"expected")
    calls: list[tuple[str, object]] = []

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"badbytes")

    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: calls.append(("process", None)))
    monkeypatch.setattr(relay_client, "server_retry", lambda _args, _lease, reason: calls.append(("retry", reason)) or {"status": "retry_pending"})

    assert relay_client.process_lease(client_args, lease) is False

    assert calls
    assert calls[0][0] == "retry"
    assert "sha256 mismatch" in str(calls[0][1])


def test_successful_processing_uploads_and_acks(tmp_path, monkeypatch):
    client_args = args(tmp_path)
    lease = lease_payload(tmp_path, raw_bytes=b"raw")
    calls: list[str] = []

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"raw")

    result = SimpleNamespace(
        ok=True,
        row={
            "task_key": "task-1",
            "target_date_local": "2026-04-12",
        },
        provenance_rows=[],
        diagnostics={"task_key": "task-1"},
        message=None,
    )

    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: result)
    monkeypatch.setattr(relay_client, "upload_result", lambda *_args, **_kwargs: calls.append("upload"))
    monkeypatch.setattr(relay_client, "server_ack", lambda *_args, **_kwargs: calls.append("ack") or {"status": "ok"})

    assert relay_client.process_lease(client_args, lease) is True

    upload_dir = tmp_path / "client" / "active" / "task-1" / "upload"
    assert calls == ["upload", "ack"]
    assert not upload_dir.exists()
    assert (tmp_path / "client" / "done" / "task-1.json").exists()


def test_successful_processing_reports_worker_phases(tmp_path, monkeypatch):
    client_args = args(tmp_path)
    lease = lease_payload(tmp_path, raw_bytes=b"raw")
    reporter = FakeReporter()
    reporter.start_worker("worker-0", label="worker-0", phase="idle")
    phases: list[str] = []

    def capture_update(worker_id, *, phase=None, label=None, details=None):
        if phase is not None:
            phases.append(phase)
        FakeReporter.update_worker(reporter, worker_id, phase=phase, label=label, details=details)

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"raw")

    result = SimpleNamespace(
        ok=True,
        row={"task_key": "task-1", "target_date_local": "2026-04-12"},
        provenance_rows=[],
        diagnostics={"task_key": "task-1"},
        message=None,
    )

    reporter.update_worker = capture_update
    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: result)
    monkeypatch.setattr(relay_client, "upload_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(relay_client, "server_ack", lambda *_args, **_kwargs: {"status": "ok"})

    assert relay_client.process_lease(client_args, lease, reporter=reporter, worker_id="worker-0") is True

    assert phases == ["pull", "verify", "process", "write_upload", "upload", "ack", "done"]
    assert any("done task_id=task-1" in message and "pull=" in message and "upload=" in message for _level, message in reporter.events)


def test_processing_failure_calls_retry_and_records_failure(tmp_path, monkeypatch):
    client_args = args(tmp_path)
    lease = lease_payload(tmp_path, raw_bytes=b"raw")
    reasons: list[str] = []

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"raw")

    result = SimpleNamespace(ok=False, row=None, provenance_rows=[], diagnostics={}, message="cfgrib failed")

    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: result)
    monkeypatch.setattr(relay_client, "server_retry", lambda _args, _lease, reason: reasons.append(reason) or {"status": "retry_pending"})

    assert relay_client.process_lease(client_args, lease) is False

    assert "cfgrib failed" in reasons[0]
    assert (tmp_path / "client" / "failed" / "task-1.json").exists()


def test_ack_failure_does_not_retry_and_preserves_active_upload(tmp_path, monkeypatch):
    client_args = args(tmp_path, heartbeat_seconds=999.0)
    lease = lease_payload(tmp_path, raw_bytes=b"raw")

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"raw")

    result = SimpleNamespace(
        ok=True,
        row={"task_key": "task-1", "target_date_local": "2026-04-12"},
        provenance_rows=[],
        diagnostics={"task_key": "task-1"},
        message=None,
    )

    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: result)
    monkeypatch.setattr(relay_client, "upload_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(relay_client, "server_ack", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("ssh dropped")))
    monkeypatch.setattr(relay_client, "server_retry", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retry must not run")))

    assert relay_client.process_lease(client_args, lease) is False

    active_upload = tmp_path / "client" / "active" / "task-1" / "upload"
    failed_marker = tmp_path / "client" / "failed" / "task-1.json"
    assert (active_upload / "result.complete").exists()
    assert "ack_ambiguous" in json.loads(failed_marker.read_text())["reason"]


def test_done_marker_failure_after_ack_does_not_retry(tmp_path, monkeypatch):
    client_args = args(tmp_path, heartbeat_seconds=999.0)
    lease = lease_payload(tmp_path, raw_bytes=b"raw")

    def fake_pull(_args, _lease, local_raw_dir, **_kwargs):
        local_raw_dir.mkdir(parents=True, exist_ok=True)
        (local_raw_dir / "raw.grib2").write_bytes(b"raw")

    result = SimpleNamespace(
        ok=True,
        row={"task_key": "task-1", "target_date_local": "2026-04-12"},
        provenance_rows=[],
        diagnostics={"task_key": "task-1"},
        message=None,
    )

    monkeypatch.setattr(relay_client, "pull_raw", fake_pull)
    monkeypatch.setattr(relay_client, "process_local_task", lambda **_kwargs: result)
    monkeypatch.setattr(relay_client, "upload_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(relay_client, "server_ack", lambda *_args, **_kwargs: {"status": "ok"})
    monkeypatch.setattr(relay_client, "record_done", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")))
    monkeypatch.setattr(relay_client, "server_retry", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retry must not run")))

    assert relay_client.process_lease(client_args, lease) is True


def test_startup_scan_deletes_stale_active_without_complete_upload(tmp_path):
    work_root = tmp_path / "client"
    active = work_root / "active" / "task-stale"
    active.mkdir(parents=True)
    relay_client.write_json_atomic(active / "lease.json", {"task_id": "task-stale"})

    state = relay_client.scan_startup_local_state(work_root)

    assert state["stale_deleted"] == ["task-stale"]
    assert not active.exists()
    assert (work_root / "failed" / "task-stale.json").exists()


def test_startup_scan_preserves_ack_ambiguous_active_upload(tmp_path):
    work_root = tmp_path / "client"
    active = work_root / "active" / "task-ambiguous" / "upload"
    active.mkdir(parents=True)
    (active / "result.complete").write_text("ok\n")

    state = relay_client.scan_startup_local_state(work_root)

    assert state["preserved_ack_ambiguous"] == ["task-ambiguous"]
    assert state["counts"]["ack_ambiguous"] == 1
    assert active.exists()


def test_list_local_reports_active_ack_ambiguous_done_and_failed(tmp_path, capsys):
    work_root = tmp_path / "client"
    (work_root / "active" / "task-active").mkdir(parents=True)
    upload = work_root / "active" / "task-ambiguous" / "upload"
    upload.mkdir(parents=True)
    (upload / "result.complete").write_text("ok\n")
    relay_client.write_json_atomic(work_root / "done" / "task-done.json", {"ok": True})
    relay_client.write_json_atomic(work_root / "failed" / "task-failed.json", {"ok": False})

    assert relay_client.command_list_local(argparse.Namespace(local_work_root=work_root)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["counts"] == {"active": 2, "ack_ambiguous": 1, "done": 1, "failed": 1}
    assert payload["ack_ambiguous"] == ["task-ambiguous"]


def test_heartbeat_loop_runs_until_stopped(tmp_path, monkeypatch):
    client_args = args(tmp_path, heartbeat_seconds=0.001)
    lease = lease_payload(tmp_path)
    stop_event = threading.Event()
    calls: list[str] = []

    def fake_run_server_json(_args, command, _extra):
        calls.append(command)
        stop_event.set()
        return {"status": "ok"}

    monkeypatch.setattr(relay_client, "run_server_json", fake_run_server_json)

    relay_client.heartbeat_loop(client_args, lease, stop_event)

    assert calls == ["heartbeat"]


def test_low_disk_prevents_leasing(tmp_path, monkeypatch):
    client_args = args(tmp_path, once=True, min_free_gb=10.0)
    stop_event = threading.Event()
    calls: list[str] = []

    monkeypatch.setattr(relay_client, "free_gb", lambda _path: 1.0)
    monkeypatch.setattr(relay_client, "lease_one", lambda _args: calls.append("lease") or {"status": "no_work"})

    assert relay_client.worker_loop(client_args, stop_event, worker_index=0) == 0
    assert calls == []


def test_once_processes_at_most_one_task(tmp_path, monkeypatch):
    client_args = args(tmp_path, once=True)
    stop_event = threading.Event()
    calls: list[str] = []

    monkeypatch.setattr(relay_client, "free_gb", lambda _path: 100.0)
    monkeypatch.setattr(relay_client, "lease_one", lambda _args: lease_payload(tmp_path))
    monkeypatch.setattr(relay_client, "process_lease", lambda _args, _lease: calls.append("process") or True)

    assert relay_client.worker_loop(client_args, stop_event, worker_index=0) == 1
    assert calls == ["process"]


def test_command_run_pause_callback_stops_new_leases_and_marks_paused(tmp_path, monkeypatch):
    reporter_holder: dict[str, FakeReporter] = {}

    def fake_create_reporter(*_args, **kwargs):
        reporter = FakeReporter(on_pause_request=kwargs.get("on_pause_request"))
        reporter_holder["reporter"] = reporter
        if reporter.on_pause_request is not None:
            reporter.on_pause_request(reason="test")
        return reporter

    monkeypatch.setattr(relay_client, "create_progress_reporter", fake_create_reporter)
    monkeypatch.setattr(relay_client, "install_signal_handlers", lambda _stop_event: None)
    monkeypatch.setattr(relay_client, "lease_one", lambda _args: (_ for _ in ()).throw(AssertionError("lease must not run")))

    assert relay_client.command_run(args(tmp_path, once=False, server_status_seconds=0.0)) == 0

    assert reporter_holder["reporter"].paused is True
    assert reporter_holder["reporter"].closed_status == "paused"


def test_server_status_poll_updates_metrics_and_tolerates_failure(tmp_path, monkeypatch):
    client_args = args(tmp_path, server_status_seconds=0.001)
    reporter = FakeReporter()
    stop_event = threading.Event()
    calls = 0

    def fake_run_server_json(_args, command, extra=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {
                "paused": False,
                "tasks": {"ready": 2, "leased": 1, "failed_terminal": 0},
                "active_clients": [{"client_id": "mac-a"}],
            }
        stop_event.set()
        raise RuntimeError("ssh failed")

    monkeypatch.setattr(relay_client, "run_server_json", fake_run_server_json)

    relay_client.server_status_loop(client_args, reporter, stop_event)

    assert reporter.metrics["server_ready"] == 2
    assert reporter.metrics["server_leased"] == 1
    assert reporter.metrics["server_clients"] == 1
    assert any("server status poll failed" in message for _level, message in reporter.events)


def test_worker_exception_stops_all_workers_and_returns_nonzero(tmp_path, monkeypatch):
    reporter = FakeReporter()
    monkeypatch.setattr(relay_client, "create_progress_reporter", lambda *_args, **_kwargs: reporter)
    monkeypatch.setattr(relay_client, "install_signal_handlers", lambda _stop_event: None)
    monkeypatch.setattr(relay_client, "scan_startup_local_state", lambda _root: {"counts": {"active": 0, "ack_ambiguous": 0, "done": 0, "failed": 0}, "stale_deleted": []})
    monkeypatch.setattr(relay_client, "worker_loop", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    assert relay_client.command_run(args(tmp_path, once=True, server_status_seconds=0.0)) == 1
    assert reporter.closed_status == "error"


def test_signal_handler_sets_stop_event(monkeypatch):
    handlers = {}

    def fake_signal(signum, handler):
        handlers[signum] = handler

    monkeypatch.setattr(relay_client.signal, "signal", fake_signal)
    stop_event = threading.Event()

    relay_client.install_signal_handlers(stop_event)
    handlers[signal.SIGINT](signal.SIGINT, None)

    assert stop_event.is_set()
