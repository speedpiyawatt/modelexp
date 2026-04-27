#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import build_hrrr_klga_feature_shards as hrrr
from tools.weather.progress import create_progress_reporter


DEFAULT_LOCAL_WORK_ROOT = Path("data/runtime/hrrr_relay_client")
DEFAULT_REMOTE_RUN_ROOT = Path("data/runtime/backfill_overnight")
DEFAULT_SERVER_STATUS_SECONDS = 30.0
DEFAULT_PULL_WORKERS = 2
REMOTE_PYTHON = ".venv/bin/python"
REMOTE_SERVER_SCRIPT = "tools/hrrr/relay_server.py"


class RelayClientError(RuntimeError):
    pass


def utc_time() -> float:
    return time.time()


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


def delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def ensure_local_layout(local_work_root: Path) -> None:
    local_work_root.mkdir(parents=True, exist_ok=True)
    for name in ("active", "done", "failed"):
        (local_work_root / name).mkdir(exist_ok=True)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_remote_command(args: argparse.Namespace, command: str, extra: list[str] | None = None) -> list[str]:
    # Quote command arguments but preserve shell operators.
    remote = (
        f"cd {shlex.quote(str(args.remote_repo))} && "
        f"{shlex.quote(REMOTE_PYTHON)} {shlex.quote(REMOTE_SERVER_SCRIPT)} "
        f"{shlex.quote(command)} --run-root {shlex.quote(str(args.remote_run_root))}"
    )
    if extra:
        remote += " " + shell_join(extra)
    return [str(args.ssh_path), str(args.server), remote]


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def current_transfer_size(progress_path: Path) -> int:
    if progress_path.exists():
        return progress_path.stat().st_size
    parent = progress_path.parent
    if not parent.exists():
        return 0
    candidates = [
        path
        for path in parent.iterdir()
        if path.is_file() and (
            path.name == progress_path.name
            or path.name.startswith(progress_path.name)
            or path.name.startswith(f".{progress_path.name}")
        )
    ]
    if not candidates:
        return 0
    return max(path.stat().st_size for path in candidates)


def run_command_with_file_progress(
    command: list[str],
    *,
    progress_path: Path | None = None,
    expected_bytes: int | None = None,
    reporter: Any | None = None,
    worker_id: str | None = None,
    file_label: str | None = None,
    phase: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if reporter is not None and worker_id is not None and progress_path is not None:
        reporter.start_transfer(worker_id, file_label=file_label or progress_path.name, total_bytes=expected_bytes)
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    try:
        while True:
            try:
                stdout_chunk, stderr_chunk = process.communicate(timeout=0.2)
                stdout_chunks.append(stdout_chunk or "")
                stderr_chunks.append(stderr_chunk or "")
                break
            except subprocess.TimeoutExpired as exc:
                if exc.stdout:
                    stdout_chunks.append(exc.stdout)
                if exc.stderr:
                    stderr_chunks.append(exc.stderr)
                if reporter is not None and worker_id is not None and progress_path is not None:
                    size = current_transfer_size(progress_path)
                    reporter.update_worker(worker_id, phase=phase)
                    reporter.update_transfer(worker_id, bytes_downloaded=size, total_bytes=expected_bytes)
        completed = subprocess.CompletedProcess(
            command,
            process.returncode,
            "".join(stdout_chunks),
            "".join(stderr_chunks),
        )
        if reporter is not None and worker_id is not None and progress_path is not None:
            size = current_transfer_size(progress_path)
            reporter.update_transfer(worker_id, bytes_downloaded=size, total_bytes=expected_bytes)
            reporter.finish_transfer(worker_id)
        if check and completed.returncode != 0:
            raise subprocess.CalledProcessError(completed.returncode, command, output=completed.stdout, stderr=completed.stderr)
        return completed
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def maybe_acquire(semaphore: threading.Semaphore | None):
    class _NullContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return semaphore if semaphore is not None else _NullContext()


def run_server_json(args: argparse.Namespace, command: str, extra: list[str] | None = None) -> dict[str, Any]:
    completed = run_command(build_remote_command(args, command, extra))
    stdout = completed.stdout.strip()
    if not stdout:
        raise RelayClientError(f"server command {command} returned empty stdout")
    try:
        return json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RelayClientError(f"server command {command} returned non-JSON stdout: {stdout}") from exc


def rsync_remote_source(server: str, remote_dir: str) -> str:
    return f"{server}:{remote_dir.rstrip('/')}/"


def remote_transfer_path(args: argparse.Namespace, remote_path: str) -> str:
    remote_text = str(remote_path)
    if remote_text.startswith("/"):
        return remote_text
    return str(PurePosixPath(str(args.remote_repo)) / remote_text)


def build_rsync_pull_command(args: argparse.Namespace, *, remote_dir: str, local_dir: Path) -> list[str]:
    transfer_dir = remote_transfer_path(args, remote_dir)
    return [
        str(args.rsync_path),
        "-aP",
        rsync_remote_source(str(args.server), transfer_dir),
        str(local_dir) + "/",
    ]


def build_rsync_upload_command(args: argparse.Namespace, *, local_upload_dir: Path, remote_upload_dir: str) -> list[str]:
    transfer_dir = remote_transfer_path(args, remote_upload_dir)
    return [
        str(args.rsync_path),
        "-aP",
        str(local_upload_dir) + "/",
        f"{args.server}:{transfer_dir.rstrip('/')}/",
    ]


def task_dirs(local_work_root: Path, task_id: str) -> dict[str, Path]:
    root = local_work_root / "active" / task_id
    return {
        "root": root,
        "raw": root / "raw",
        "reduced": root / "reduced",
        "scratch": root / "scratch",
        "upload": root / "upload",
        "done": local_work_root / "done" / f"{task_id}.json",
        "failed": local_work_root / "failed" / f"{task_id}.json",
    }


def prepare_local_task_dir(local_work_root: Path, lease: dict[str, Any]) -> dict[str, Path]:
    paths = task_dirs(local_work_root, str(lease["task_id"]))
    delete_path(paths["root"])
    for key in ("raw", "reduced", "scratch", "upload"):
        paths[key].mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["root"] / "lease.json", lease)
    return paths


def load_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def local_state(local_work_root: Path) -> dict[str, Any]:
    ensure_local_layout(local_work_root)
    active_root = local_work_root / "active"
    done_root = local_work_root / "done"
    failed_root = local_work_root / "failed"
    active: list[str] = []
    ack_ambiguous: list[str] = []
    for path in sorted(active_root.iterdir()):
        if not path.is_dir():
            continue
        task_id = path.name
        active.append(task_id)
        if (path / "upload" / "result.complete").exists():
            ack_ambiguous.append(task_id)
    done = sorted(path.stem for path in done_root.glob("*.json"))
    failed = sorted(path.stem for path in failed_root.glob("*.json"))
    return {
        "status": "ok",
        "active": active,
        "ack_ambiguous": ack_ambiguous,
        "done": done,
        "failed": failed,
        "counts": {
            "active": len(active),
            "ack_ambiguous": len(ack_ambiguous),
            "done": len(done),
            "failed": len(failed),
        },
    }


def scan_startup_local_state(local_work_root: Path) -> dict[str, Any]:
    ensure_local_layout(local_work_root)
    stale_deleted: list[str] = []
    preserved_ack_ambiguous: list[str] = []
    active_root = local_work_root / "active"
    for path in sorted(active_root.iterdir()):
        if not path.is_dir():
            continue
        task_id = path.name
        if (path / "upload" / "result.complete").exists():
            preserved_ack_ambiguous.append(task_id)
            continue
        lease = load_json_or_empty(path / "lease.json")
        failed_path = local_work_root / "failed" / f"{task_id}.json"
        write_json_atomic(
            failed_path,
            {
                "lease": lease,
                "reason": "stale_active_on_startup",
                "failed_at": utc_time(),
            },
        )
        delete_path(path)
        stale_deleted.append(task_id)
    state = local_state(local_work_root)
    state["stale_deleted"] = stale_deleted
    state["preserved_ack_ambiguous"] = preserved_ack_ambiguous
    return state


def verify_raw_file(raw_path: Path, *, expected_bytes: int, checksum_sha256: str) -> None:
    if not raw_path.exists():
        raise RelayClientError(f"missing raw file: {raw_path}")
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != int(expected_bytes):
        raise RelayClientError(f"raw byte count mismatch: expected {expected_bytes}, got {actual_bytes}")
    actual_sha = sha256_file(raw_path)
    if actual_sha != checksum_sha256:
        raise RelayClientError(f"raw sha256 mismatch: expected {checksum_sha256}, got {actual_sha}")


def task_from_lease(lease: dict[str, Any]) -> hrrr.TaskSpec:
    return hrrr.TaskSpec(**dict(lease["task"]))


def process_local_task(
    *,
    lease: dict[str, Any],
    raw_dir: Path,
    reduced_dir: Path,
    scratch_dir: Path,
    wgrib2_path: str,
    include_legacy_aliases: bool = False,
) -> hrrr.TaskResult:
    task = task_from_lease(lease)
    raw_path = raw_dir / "raw.grib2"
    reduced_path = hrrr.path_for_reduced(reduced_dir, task)
    diagnostics = hrrr.default_task_diagnostics(task)
    diagnostics.update(
        {
            "raw_file_path": str(raw_path),
            "raw_manifest_path": str(raw_dir / "raw.manifest.csv"),
            "raw_selection_manifest_path": str(raw_dir / "raw.selection.csv"),
            "reduced_file_path": str(reduced_path),
            "grib_url": hrrr.task_remote_url(task, "google"),
            "scratch_dir": str(scratch_dir),
            "downloaded_range_bytes": int(lease.get("expected_bytes") or 0),
        }
    )
    cfgrib_index_dir: Path | None = None
    try:
        reuse_signature = hrrr.build_reduced_reuse_signature(task=task, raw_path=raw_path)
        diagnostics["reduced_reuse_signature"] = reuse_signature
        reduced_result = hrrr.reduce_grib2(wgrib2_path, raw_path, reduced_path)
        if len(reduced_result) == 4:
            reduced_inventory, _, inventory_seconds, reduce_seconds = reduced_result
        else:
            reduced_inventory, _ = reduced_result
            inventory_seconds = 0.0
            reduce_seconds = 0.0
        hrrr.write_reduced_reuse_signature(reduced_path, reuse_signature)
        diagnostics["timing_wgrib_inventory_seconds"] = inventory_seconds
        diagnostics["timing_reduce_seconds"] = reduce_seconds
        cfgrib_parent = scratch_dir / "cfgrib_index"
        cfgrib_parent.mkdir(parents=True, exist_ok=True)
        cfgrib_index_dir = Path(tempfile.mkdtemp(prefix=f"hrrr_cfgrib_{task.forecast_hour:02d}_", dir=str(cfgrib_parent)))
        return hrrr.process_reduced_grib(
            reduced_path,
            reduced_inventory,
            task,
            hrrr.task_remote_url(task, "google"),
            cfgrib_index_dir=cfgrib_index_dir,
            diagnostics=diagnostics,
            include_legacy_aliases=include_legacy_aliases,
        )
    except Exception as exc:
        diagnostics["last_error_type"] = type(exc).__name__
        diagnostics["last_error_message"] = str(exc)
        return hrrr.TaskResult(False, task.key, None, [], [], str(exc), diagnostics)
    finally:
        if cfgrib_index_dir is not None:
            shutil.rmtree(cfgrib_index_dir, ignore_errors=True)
        reduced_path.unlink(missing_ok=True)
        hrrr.reduced_signature_path(reduced_path).unlink(missing_ok=True)


def write_upload_artifacts(upload_dir: Path, lease: dict[str, Any], result: hrrr.TaskResult) -> None:
    if not result.ok or result.row is None:
        raise RelayClientError(result.message or "task processing failed")
    delete_path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(upload_dir / "row.json", result.row)
    write_text_atomic(upload_dir / "provenance.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in result.provenance_rows))
    write_json_atomic(upload_dir / "diagnostics.json", dict(result.diagnostics))
    write_json_atomic(upload_dir / "task.json", dict(lease["task"]))
    write_text_atomic(upload_dir / "result.complete", "ok\n")


def heartbeat_loop(args: argparse.Namespace, lease: dict[str, Any], stop_event: threading.Event) -> None:
    task_id = str(lease["task_id"])
    while not stop_event.wait(float(args.heartbeat_seconds)):
        try:
            run_server_json(
                args,
                "heartbeat",
                ["--task-id", task_id, "--client-id", str(args.client_id), "--lease-minutes", str(args.lease_minutes)],
            )
        except Exception as exc:
            print(f"[heartbeat-error] task_id={task_id} error={exc}", file=sys.stderr, flush=True)


def reporter_update_worker(reporter: Any | None, worker_id: str | None, *, phase: str, label: str | None = None, details: str | None = None) -> None:
    if reporter is None or worker_id is None:
        return
    reporter.update_worker(worker_id, phase=phase, label=label, details=details)


def reporter_log(reporter: Any | None, message: str, *, level: str = "info") -> None:
    if reporter is not None:
        reporter.log_event(message, level=level)


def server_retry(args: argparse.Namespace, lease: dict[str, Any], reason: str) -> dict[str, Any]:
    return run_server_json(
        args,
        "retry",
        [
            "--task-id",
            str(lease["task_id"]),
            "--client-id",
            str(args.client_id),
            "--reason",
            reason[:500],
        ],
    )


def server_ack(args: argparse.Namespace, lease: dict[str, Any]) -> dict[str, Any]:
    return run_server_json(
        args,
        "ack",
        [
            "--task-id",
            str(lease["task_id"]),
            "--client-id",
            str(args.client_id),
            "--upload-dir",
            str(lease["upload_dir"]),
        ],
    )


def pull_raw(
    args: argparse.Namespace,
    lease: dict[str, Any],
    local_raw_dir: Path,
    *,
    reporter: Any | None = None,
    worker_id: str | None = None,
) -> None:
    local_raw_dir.mkdir(parents=True, exist_ok=True)
    with maybe_acquire(getattr(args, "pull_semaphore", None)):
        run_command_with_file_progress(
            build_rsync_pull_command(args, remote_dir=str(lease["raw_dir"]), local_dir=local_raw_dir),
            progress_path=local_raw_dir / "raw.grib2",
            expected_bytes=int(lease.get("expected_bytes") or 0),
            reporter=reporter,
            worker_id=worker_id,
            file_label=str(lease.get("task_id") or "raw.grib2"),
            phase="pull",
        )


def upload_result(
    args: argparse.Namespace,
    lease: dict[str, Any],
    local_upload_dir: Path,
    *,
    reporter: Any | None = None,
    worker_id: str | None = None,
) -> None:
    remote_upload_dir = remote_transfer_path(args, str(lease["upload_dir"]))
    run_command([str(args.ssh_path), str(args.server), f"mkdir -p {shlex.quote(remote_upload_dir)}"])
    total_bytes = 0
    for root, _dirs, files in os.walk(local_upload_dir):
        for name in files:
            total_bytes += (Path(root) / name).stat().st_size
    run_command_with_file_progress(
        build_rsync_upload_command(args, local_upload_dir=local_upload_dir, remote_upload_dir=str(lease["upload_dir"])),
        progress_path=local_upload_dir / "result.complete",
        expected_bytes=total_bytes or None,
        reporter=reporter,
        worker_id=worker_id,
        file_label=f"{lease.get('task_id')}:upload",
        phase="upload",
    )


def record_done(local_work_root: Path, lease: dict[str, Any], payload: dict[str, Any]) -> None:
    paths = task_dirs(local_work_root, str(lease["task_id"]))
    write_json_atomic(paths["done"], {"lease": lease, "result": payload, "completed_at": utc_time()})
    delete_path(paths["root"])


def record_failed(local_work_root: Path, lease: dict[str, Any], reason: str, *, keep_active: bool = False) -> None:
    paths = task_dirs(local_work_root, str(lease["task_id"]))
    write_json_atomic(paths["failed"], {"lease": lease, "reason": reason, "failed_at": utc_time()})
    if not keep_active:
        delete_path(paths["root"])


def retry_and_record_failure(args: argparse.Namespace, lease: dict[str, Any], reason: str, *, reporter: Any | None = None, worker_id: str | None = None) -> bool:
    reporter_update_worker(reporter, worker_id, phase="retry", details=reason[:120])
    try:
        server_retry(args, lease, reason)
    except Exception as retry_exc:
        reason = f"{reason}; retry_failed={type(retry_exc).__name__}: {retry_exc}"
    record_failed(args.local_work_root, lease, reason, keep_active=bool(getattr(args, "keep_active_on_failure", False)))
    reporter_log(reporter, f"retry task_id={lease['task_id']} reason={reason}", level="warn")
    print(f"[retry] task_id={lease['task_id']} reason={reason}", file=sys.stderr, flush=True)
    return False


def process_lease(args: argparse.Namespace, lease: dict[str, Any], *, reporter: Any | None = None, worker_id: str | None = None) -> bool:
    paths = prepare_local_task_dir(args.local_work_root, lease)
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(target=heartbeat_loop, args=(args, lease, heartbeat_stop), daemon=True)
    heartbeat_thread.start()
    started_at = utc_time()
    phase_started_at = started_at
    phase_seconds: dict[str, float] = {}
    task_id = str(lease["task_id"])
    reporter_update_worker(reporter, worker_id, phase="pull", label=task_id)
    try:
        try:
            pull_raw(args, lease, paths["raw"], reporter=reporter, worker_id=worker_id)
            phase_seconds["pull"] = utc_time() - phase_started_at
            phase_started_at = utc_time()
            reporter_update_worker(reporter, worker_id, phase="verify", label=task_id)
            verify_raw_file(
                paths["raw"] / "raw.grib2",
                expected_bytes=int(lease["expected_bytes"]),
                checksum_sha256=str(lease["checksum_sha256"]),
            )
            phase_seconds["verify"] = utc_time() - phase_started_at
            phase_started_at = utc_time()
            reporter_update_worker(reporter, worker_id, phase="process", label=task_id)
            result = process_local_task(
                lease=lease,
                raw_dir=paths["raw"],
                reduced_dir=paths["reduced"],
                scratch_dir=paths["scratch"],
                wgrib2_path=str(args.wgrib2_path),
            )
            phase_seconds["process"] = utc_time() - phase_started_at
            phase_started_at = utc_time()
            if float(args.max_local_task_seconds) > 0 and utc_time() - started_at > float(args.max_local_task_seconds):
                raise RelayClientError(f"local task exceeded {args.max_local_task_seconds}s")
            reporter_update_worker(reporter, worker_id, phase="write_upload", label=task_id)
            write_upload_artifacts(paths["upload"], lease, result)
            phase_seconds["write_upload"] = utc_time() - phase_started_at
            phase_started_at = utc_time()
            reporter_update_worker(reporter, worker_id, phase="upload", label=task_id)
            upload_result(args, lease, paths["upload"], reporter=reporter, worker_id=worker_id)
            phase_seconds["upload"] = utc_time() - phase_started_at
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            return retry_and_record_failure(args, lease, reason, reporter=reporter, worker_id=worker_id)

        try:
            phase_started_at = utc_time()
            reporter_update_worker(reporter, worker_id, phase="ack", label=task_id)
            ack_payload = server_ack(args, lease)
            phase_seconds["ack"] = utc_time() - phase_started_at
        except Exception as exc:
            reason = f"ack_ambiguous: {type(exc).__name__}: {exc}"
            reporter_update_worker(reporter, worker_id, phase="ack_ambiguous", label=task_id, details=reason[:120])
            record_failed(args.local_work_root, lease, reason, keep_active=True)
            reporter_log(reporter, f"ack_ambiguous task_id={task_id}", level="error")
            print(f"[ack-ambiguous] task_id={lease['task_id']} reason={reason}", file=sys.stderr, flush=True)
            return False

        try:
            reporter_update_worker(reporter, worker_id, phase="done", label=task_id)
            record_done(args.local_work_root, lease, ack_payload)
        except Exception as exc:
            reporter_log(reporter, f"done marker failed task_id={task_id}: {type(exc).__name__}: {exc}", level="warn")
            print(f"[done-record-error] task_id={lease['task_id']} error={type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return True
        reporter_log(
            reporter,
            f"done task_id={task_id} seconds={utc_time() - started_at:.1f} "
            f"pull={phase_seconds.get('pull', 0.0):.1f}s "
            f"verify={phase_seconds.get('verify', 0.0):.1f}s "
            f"process={phase_seconds.get('process', 0.0):.1f}s "
            f"write_upload={phase_seconds.get('write_upload', 0.0):.1f}s "
            f"upload={phase_seconds.get('upload', 0.0):.1f}s "
            f"ack={phase_seconds.get('ack', 0.0):.1f}s"
        )
        print(f"[done] task_id={lease['task_id']} finalize={ack_payload.get('finalize')}", flush=True)
        return True
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5)


def lease_one(args: argparse.Namespace) -> dict[str, Any]:
    return run_server_json(
        args,
        "lease",
        ["--client-id", str(args.client_id), "--lease-minutes", str(args.lease_minutes)],
    )


def worker_loop(args: argparse.Namespace, stop_event: threading.Event, worker_index: int, reporter: Any | None = None) -> int:
    processed = 0
    worker_id = f"worker-{worker_index}"
    if reporter is not None:
        reporter.start_worker(worker_id, label=worker_id, phase="idle")
    while not stop_event.is_set():
        local_free_gb = free_gb(args.local_work_root)
        if local_free_gb < float(args.min_free_gb):
            reporter_update_worker(reporter, worker_id, phase="low_disk", details=f"free_gb={local_free_gb:.2f}")
            print(f"[idle] worker={worker_index} low_disk free_gb={local_free_gb:.2f}", flush=True)
            if args.once:
                if reporter is not None:
                    reporter.retire_worker(worker_id, message="once low_disk")
                return processed
            stop_event.wait(float(args.poll_seconds))
            continue
        try:
            reporter_update_worker(reporter, worker_id, phase="lease")
            lease = lease_one(args)
        except Exception as exc:
            reporter_update_worker(reporter, worker_id, phase="idle", details="lease_error")
            reporter_log(reporter, f"lease-error worker={worker_index}: {exc}", level="warn")
            print(f"[lease-error] worker={worker_index} error={exc}", file=sys.stderr, flush=True)
            if args.once:
                if reporter is not None:
                    reporter.retire_worker(worker_id, message="once lease_error")
                return processed
            stop_event.wait(float(args.poll_seconds))
            continue
        status = str(lease.get("status"))
        if status == "leased":
            if reporter is None:
                ok = process_lease(args, lease)
            else:
                ok = process_lease(args, lease, reporter=reporter, worker_id=worker_id)
            processed += 1
            if reporter is not None:
                reporter.record_outcome("completed" if ok else "failed", message=f"task {lease.get('task_id')} {'ok' if ok else 'failed'}")
            if args.once:
                if reporter is not None:
                    reporter.retire_worker(worker_id, message="once done")
                return processed
            continue
        if status in {"paused", "no_work"}:
            reporter_update_worker(reporter, worker_id, phase="idle", details=status)
            print(f"[idle] worker={worker_index} status={status}", flush=True)
            if args.once:
                if reporter is not None:
                    reporter.retire_worker(worker_id, message=f"once {status}")
                return processed
            stop_event.wait(float(args.poll_seconds))
            continue
        raise RelayClientError(f"unexpected lease status: {status}")
    if reporter is not None:
        reporter.retire_worker(worker_id, message="drained")
    return processed


def install_signal_handlers(stop_event: threading.Event) -> None:
    def handle_signal(signum, _frame):
        print(f"[signal] received={signum} draining", file=sys.stderr, flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)


def update_local_metrics(reporter: Any | None, args: argparse.Namespace) -> None:
    if reporter is None:
        return
    state = local_state(args.local_work_root)
    counts = state["counts"]
    reporter.set_metrics(
        local_active=counts["active"],
        ack_ambiguous=counts["ack_ambiguous"],
        local_done=counts["done"],
        local_failed=counts["failed"],
        local_free_gb=f"{free_gb(args.local_work_root):.1f}",
    )


def server_status_loop(args: argparse.Namespace, reporter: Any, stop_event: threading.Event) -> None:
    interval = float(getattr(args, "server_status_seconds", 0.0))
    if interval <= 0:
        return
    while not stop_event.is_set():
        try:
            payload = run_server_json(args, "status")
            tasks = payload.get("tasks") or {}
            reporter.set_metrics(
                server_paused=payload.get("paused"),
                server_ready=tasks.get("ready", 0),
                server_leased=tasks.get("leased", 0),
                server_failed=tasks.get("failed_terminal", 0),
                server_clients=len(payload.get("active_clients") or []),
            )
        except Exception as exc:
            reporter.log_event(f"server status poll failed: {type(exc).__name__}: {exc}", level="warn")
        stop_event.wait(interval)


def command_run(args: argparse.Namespace) -> int:
    startup_state = scan_startup_local_state(args.local_work_root)
    stop_event = threading.Event()

    def request_pause(*, reason: str = "operator") -> None:
        stop_event.set()

    reporter = create_progress_reporter(
        "HRRR relay client",
        unit="task",
        mode=args.progress_mode,
        on_pause_request=request_pause,
        enable_dashboard_hotkeys=not bool(args.disable_dashboard_hotkeys),
        pause_control_file=args.pause_control_file,
    )
    reporter.log_event(
        "local state "
        f"active={startup_state['counts']['active']} "
        f"ack_ambiguous={startup_state['counts']['ack_ambiguous']} "
        f"stale_deleted={len(startup_state['stale_deleted'])}"
    )
    update_local_metrics(reporter, args)
    install_signal_handlers(stop_event)
    status_stop = threading.Event()
    status_thread = threading.Thread(target=server_status_loop, args=(args, reporter, status_stop), daemon=True)
    status_thread.start()
    worker_count = max(1, int(args.workers))
    pull_workers = max(1, int(getattr(args, "pull_workers", worker_count)))
    args.pull_semaphore = threading.BoundedSemaphore(min(worker_count, pull_workers))
    exit_status = 0
    final_status = "done"
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hrrr-relay") as executor:
        futures = [executor.submit(worker_loop, args, stop_event, index, reporter) for index in range(worker_count)]
        try:
            total = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    total += int(future.result())
                except Exception as exc:
                    exit_status = 1
                    final_status = "error"
                    reporter.log_event(f"worker failed: {type(exc).__name__}: {exc}", level="error")
                    stop_event.set()
                if not args.once and not stop_event.is_set() and exit_status == 0:
                    # Continuous mode workers should not normally return unless asked to stop.
                    exit_status = 1
                    final_status = "unexpected_worker_exit"
                    reporter.log_event("worker exited unexpectedly", level="error")
                    stop_event.set()
                update_local_metrics(reporter, args)
            print(f"[exit] processed={total}", flush=True)
        finally:
            stop_event.set()
            status_stop.set()
            status_thread.join(timeout=5)
            if stop_event.is_set() and final_status == "done" and exit_status == 0:
                final_status = "paused" if not args.once else "once"
                if not args.once:
                    reporter.mark_paused(reason="operator")
            reporter.close(status=final_status)
    return exit_status


def command_list_local(args: argparse.Namespace) -> int:
    print(json.dumps(local_state(args.local_work_root), sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client-side processor for the HRRR selected-raw relay queue.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--server", required=True)
    run_parser.add_argument("--remote-repo", type=Path, required=True)
    run_parser.add_argument("--remote-run-root", type=Path, default=DEFAULT_REMOTE_RUN_ROOT)
    run_parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT)
    run_parser.add_argument("--client-id", required=True)
    run_parser.add_argument("--workers", type=int, default=1)
    run_parser.add_argument("--pull-workers", type=int, default=DEFAULT_PULL_WORKERS)
    run_parser.add_argument("--min-free-gb", type=float, default=5.0)
    run_parser.add_argument("--lease-minutes", type=float, default=30.0)
    run_parser.add_argument("--wgrib2-path", default="wgrib2")
    run_parser.add_argument("--poll-seconds", type=float, default=10.0)
    run_parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    run_parser.add_argument("--rsync-path", default="rsync")
    run_parser.add_argument("--ssh-path", default="ssh")
    run_parser.add_argument("--max-local-task-seconds", type=float, default=3600.0)
    run_parser.add_argument("--progress-mode", choices=("auto", "dashboard", "log"), default="auto")
    run_parser.add_argument("--pause-control-file")
    run_parser.add_argument("--disable-dashboard-hotkeys", action="store_true")
    run_parser.add_argument("--server-status-seconds", type=float, default=DEFAULT_SERVER_STATUS_SECONDS)
    run_parser.add_argument("--keep-active-on-failure", action="store_true")
    run_parser.add_argument("--once", action="store_true")
    run_parser.set_defaults(func=command_run)

    list_parser = subparsers.add_parser("list-local")
    list_parser.add_argument("--local-work-root", type=Path, default=DEFAULT_LOCAL_WORK_ROOT)
    list_parser.set_defaults(func=command_list_local)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
