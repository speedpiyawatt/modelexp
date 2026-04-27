from __future__ import annotations

import io
import pathlib
import re
import sys
import time
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.weather.progress as progress_mod
from tools.weather.progress import ProgressBar, create_progress_reporter, format_progress_line


class TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)", "", text)


def last_frame(text: str, marker: str) -> str:
    index = text.rfind(marker)
    return text[index:] if index >= 0 else text


def test_format_progress_line_includes_label_counts_percent_and_bar():
    line = format_progress_line(
        2,
        5,
        label="NBM overnight",
        unit="date",
        stage="load",
        status="target_date_local=2026-04-12",
        parent_label="cycle",
        parent_completed=1,
        parent_total=3,
        parent_unit="lead",
        bar_width=10,
    )
    assert "[progress] NBM overnight" in line
    assert "2/5 date" in line
    assert "( 40.0%)" in line
    assert "[####------]" in line
    assert "parent=cycle:1/3 lead" in line
    assert "stage=load" in line
    assert "status=target_date_local=2026-04-12" in line


def test_progress_bar_close_finishes_at_100_percent_for_tty():
    stream = TtyStringIO()
    progress = ProgressBar(3, label="HRRR 2023-01", unit="task", stream=stream)

    progress.update(stage="submit", status="fh=00")
    progress.advance(stage="active", status="fh=00")
    progress.advance(stage="checkpoint", status="fh=01")
    progress.close()

    output = stream.getvalue()
    assert "\r" not in output
    assert "3/3 task" in output
    assert "(100.0%)" in output
    assert output.endswith("\n")


def test_progress_bar_defaults_to_stdout():
    progress = ProgressBar(1)
    assert progress.stream is sys.stdout
    progress.close()


def test_progress_bar_non_tty_update_is_silent_but_advance_and_close_emit_snapshots():
    stream = io.StringIO()
    progress = ProgressBar(2, label="LAMP overnight", unit="date", stream=stream, is_tty=False)

    progress.update(stage="select", status="target_date_local=2026-04-12")
    progress.advance(stage="write", status="target_date_local=2026-04-12")
    progress.advance(stage="complete", status="target_date_local=2026-04-12")
    progress.close(stage="finalize", status="target_date_local=2026-04-12 done")

    output = stream.getvalue()
    assert "\r" not in output
    assert output.count("\n") == 3
    assert "1/2 date" in output
    assert output.count("2/2 date") == 2
    assert "stage=select" not in output
    assert "stage=complete" in output
    assert "stage=finalize" in output


def test_progress_reporter_log_mode_emits_structured_events():
    stream = io.StringIO()
    reporter = create_progress_reporter("NBM build", unit="lead", total=10, mode="log", stream=stream, is_tty=False)

    reporter.set_metrics(cycles_total=2, active_cycles=1)
    reporter.upsert_group("20260101T1500Z", label="20260101T1500Z", total=36, status="queued")
    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download", group_id="20260101T1500Z")
    reporter.start_transfer("ThreadPoolExecutor-0_0", file_label="file.grib2", total_bytes=1024)
    reporter.update_transfer("ThreadPoolExecutor-0_0", bytes_downloaded=512, total_bytes=1024)
    reporter.complete_worker("ThreadPoolExecutor-0_0", message="lead ok")
    reporter.close(status="done")

    output = stream.getvalue()
    assert "event=run_start" in output
    assert "event=summary" in output
    assert "event=group" in output
    assert "event=worker_start" in output
    assert "event=transfer_progress" in output
    assert "event=worker_complete" in output
    assert "event=run_complete" in output


def test_progress_reporter_dashboard_mode_renders_worker_transfer_status():
    stream = TtyStringIO()
    reporter = create_progress_reporter("HRRR 2023-01", unit="task", total=2, mode="dashboard", stream=stream, is_tty=True)

    reporter.set_metrics(month="2023-01", retained_cycles=2, max_workers=2)
    reporter.upsert_group("2023-01", label="2023-01", total=2, status="queued")
    reporter.start_worker("ThreadPoolExecutor-0_0", label="2023-01-01 c00 f04", phase="download", group_id="2023-01")
    reporter.start_transfer("ThreadPoolExecutor-0_0", file_label="subset.grib2", total_bytes=2048)
    reporter.update_transfer("ThreadPoolExecutor-0_0", bytes_downloaded=1024, total_bytes=2048)
    reporter.log_event("2023-01-01 c00 f04 ok")
    reporter.close(status="done")

    output = stream.getvalue()
    plain = strip_ansi(output)
    assert "\x1b[?1049h" in output
    assert "\x1b[?25l" in output
    assert "\x1b[?25h" in output
    assert "\x1b[?1049l" in output
    assert "HRRR 2023-01" in plain
    assert "Workers 1" in plain
    assert "subset.grib2" not in plain
    assert "Groups:" not in plain
    assert "Alerts" in plain


def test_progress_reporter_dashboard_mode_shows_warn_error_footer_only():
    stream = TtyStringIO()
    reporter = create_progress_reporter("NBM build", unit="lead", total=3, mode="dashboard", stream=stream, is_tty=True)

    reporter.log_event("lead ok", level="info")
    reporter.log_event("lead warning", level="warn")
    reporter.log_event("lead failed", level="error")
    reporter.close(status="done")

    plain = strip_ansi(stream.getvalue())
    assert "lead warning" in plain
    assert "lead failed" in plain
    assert "lead ok" not in plain


def test_progress_reporter_dashboard_mode_width_limits_lines():
    stream = TtyStringIO()
    with mock.patch("tools.weather.progress.shutil.get_terminal_size", return_value=mock.Mock(columns=90, lines=18)):
        reporter = create_progress_reporter("NBM build", unit="lead", total=50, mode="dashboard", stream=stream, is_tty=True)
        reporter.set_metrics(cycles_total=10, completed_cycles=1)
        reporter.start_worker(
            "ThreadPoolExecutor-0_0",
            label="20260101T1500Z f001",
            phase="download",
            details="byte_range_download",
        )
        reporter.start_transfer("ThreadPoolExecutor-0_0", file_label="very-long-file-name.grib2", total_bytes=1000)
        reporter.update_transfer("ThreadPoolExecutor-0_0", bytes_downloaded=500, total_bytes=1000)
        reporter.close(status="done")

    plain = last_frame(strip_ansi(stream.getvalue()), "┌ Overview")
    lines = [line for line in plain.splitlines() if line]
    assert all(len(line) <= 90 for line in lines)
    assert "very-long-file-name.grib2" not in plain


def test_batch_dashboard_renders_header_day_board_and_timing():
    stream = TtyStringIO()
    with mock.patch("tools.weather.progress.shutil.get_terminal_size", return_value=mock.Mock(columns=120, lines=24)):
        reporter = create_progress_reporter(
            "NBM Batch Backfill",
            unit="day",
            total=4,
            mode="dashboard",
            stream=stream,
            is_tty=True,
            dashboard_kind="nbm_batch",
        )
        reporter.set_metrics(
            DateRange="2026-04-01..2026-04-04",
            BatchMode="cycle",
            PlanDays=4,
            PlanDownload=4,
            PlanReduce=1,
            PlanExtract=1,
            SelectionMode="overnight_0005",
            MetricProfile="overnight",
            Provenance="off",
            FreeGB="88.0",
            ActiveDays=1,
            QueuedDays=2,
        )
        reporter.upsert_batch_day(
            "2026-04-01",
            lifecycle_phase="complete",
            selected_issue="20260401T0500Z",
            downloaded_leads=23,
            expected_leads=23,
            batch_status="done",
            extract_status="done",
            overnight_status="done",
            status="complete",
            elapsed_seconds=120.0,
        )
        reporter.upsert_batch_day(
            "2026-04-02",
            lifecycle_phase="batch_extract",
            selected_issue="20260402T0500Z",
            downloaded_leads=18,
            expected_leads=23,
            batch_status="done",
            extract_status="rows",
            overnight_status="--",
            status="extract",
            started_at=time.perf_counter(),
        )
        reporter.record_batch_timing(
            full_day_seconds=120.0,
            raw_build_seconds=90.0,
            cleanup_seconds=5.0,
            timing_crop_seconds_median=3.0,
            timing_cfgrib_open_seconds_median=2.0,
            timing_row_metric_seconds_median=1.0,
        )
        reporter.log_event("2026-04-02 retry scheduled", level="warn")
        reporter.close(status="done")

    plain = strip_ansi(stream.getvalue())
    assert "NBM Batch Backfill | 2026-04-01..2026-04-04 | cycle | days=4 dl=4 reduce=1 extract=1" in plain
    assert "Day Pipeline" in plain
    assert "2026-04-02" in plain
    assert "18/23" in plain
    assert "rows" in plain
    assert "selection=overnight_0005" in plain
    assert "day med 02:00" in plain
    assert "2026-04-02 retry scheduled" in plain


def test_batch_dashboard_width_limits_lines():
    stream = TtyStringIO()
    with mock.patch("tools.weather.progress.shutil.get_terminal_size", return_value=mock.Mock(columns=84, lines=20)):
        reporter = create_progress_reporter(
            "NBM Batch Backfill",
            unit="day",
            total=1,
            mode="dashboard",
            stream=stream,
            is_tty=True,
            dashboard_kind="nbm_batch",
        )
        reporter.set_metrics(DateRange="2026-04-01..2026-04-14", BatchMode="cycle", PlanDays=4, PlanDownload=4, PlanReduce=1, PlanExtract=1)
        reporter.upsert_batch_day(
            "2026-04-01",
            lifecycle_phase="failed",
            selected_issue="20260401T0500Z",
            downloaded_leads=3,
            expected_leads=23,
            batch_status="err",
            extract_status="queued",
            status="failed_with_a_very_long_error_name",
        )
        reporter.close(status="failed")

    plain = last_frame(strip_ansi(stream.getvalue()), "┌ Overview")
    lines = [line for line in plain.splitlines() if line]
    assert lines
    assert all(len(line) <= 84 for line in lines)
    assert "3/23" in plain


def test_batch_dashboard_utilization_classifies_retry_workers_by_worker_name():
    reporter = create_progress_reporter(
        "NBM Batch Backfill",
        unit="day",
        total=1,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
        dashboard_kind="nbm_batch",
    )
    reporter.start_worker("2026-04-01/download_1", label="20260401T0500Z f019", phase="retry_wait")
    reporter.start_worker("2026-04-01/reduce_1", label="20260401T0500Z batch", phase="retry_wait")
    reporter.start_worker("2026-04-01/extract_1", label="20260401T0500Z batch", phase="retry_wait")

    rows = reporter.renderer._utilization_rows(reporter.state, width=120, height=4)
    reporter.close(status="done")

    assert any("downloads 1  crops 1  extracts 1" in row for row in rows)


def test_progress_reporter_dashboard_mode_shows_overflow_row():
    stream = TtyStringIO()
    with mock.patch("tools.weather.progress.shutil.get_terminal_size", return_value=mock.Mock(columns=120, lines=14)):
        reporter = create_progress_reporter("NBM build", unit="lead", total=20, mode="dashboard", stream=stream, is_tty=True)
        for index in range(12):
            reporter.start_worker(
                f"ThreadPoolExecutor-0_{index}",
                label=f"20260101T1500Z f{index + 1:03d}",
                phase="download",
                details="byte_range_download",
            )
        reporter.close(status="done")

    plain = strip_ansi(stream.getvalue())
    assert re.search(r"\+\d+ more active", plain)


def test_progress_reporter_dashboard_mode_keeps_worker_rows_stable():
    stream = TtyStringIO()
    reporter = create_progress_reporter("NBM build", unit="lead", total=4, mode="dashboard", stream=stream, is_tty=True)

    reporter.start_worker("ThreadPoolExecutor-0_1", label="20260101T1500Z f002", phase="download", details="byte_range_download")
    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="open", details="open_grouped_datasets")
    reporter.start_worker("ThreadPoolExecutor-0_2", label="20260101T1500Z f003", phase="extract", details="build_rows")

    reporter.refresh(force=True)
    before = last_frame(strip_ansi(stream.getvalue()), "┌ Overview")
    reporter.update_worker("ThreadPoolExecutor-0_2", phase="download", details="byte_range_download")
    reporter.update_worker("ThreadPoolExecutor-0_0", phase="cleanup", details="cleanup")
    reporter.update_transfer("ThreadPoolExecutor-0_2", bytes_downloaded=512, total_bytes=1024)
    reporter.refresh(force=True)
    after = last_frame(strip_ansi(stream.getvalue()), "┌ Overview")

    before_positions = [before.find("w1"), before.find("w0"), before.find("w2")]
    after_positions = [after.find("w1"), after.find("w0"), after.find("w2")]
    assert all(position >= 0 for position in before_positions)
    assert all(position >= 0 for position in after_positions)
    assert before_positions == sorted(before_positions)
    assert after_positions == sorted(after_positions)


def test_progress_reporter_dashboard_mode_eta_hidden_then_visible():
    stream = TtyStringIO()
    reporter = create_progress_reporter("NBM build", unit="lead", total=10, mode="dashboard", stream=stream, is_tty=True)
    initial = strip_ansi(stream.getvalue())
    assert "eta --" in initial

    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download")
    reporter.complete_worker("ThreadPoolExecutor-0_0", message="lead 1 ok")
    reporter.start_worker("ThreadPoolExecutor-0_1", label="20260101T1500Z f002", phase="download")
    reporter.complete_worker("ThreadPoolExecutor-0_1", message="lead 2 ok")
    reporter.close(status="done")

    final = strip_ansi(stream.getvalue())
    assert re.search(r"eta (?!\-\-)[0-9:]{5,8}", final)


def test_progress_reporter_retry_recovery_count_survives_worker_restart():
    reporter = create_progress_reporter("NBM build", unit="lead", total=1, mode="log", stream=io.StringIO(), is_tty=False)

    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download")
    reporter.schedule_retry(
        "ThreadPoolExecutor-0_0",
        attempt=2,
        max_attempts=3,
        delay_seconds=0.1,
        message="temporary failure",
        error_class="network",
    )
    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download")
    reporter.recover_worker("ThreadPoolExecutor-0_0", message="20260101T1500Z f001 recovered a2/3")

    assert reporter.state.recovered_count == 1


def test_progress_reporter_aggregate_speed_counts_only_active_download_workers():
    reporter = create_progress_reporter("NBM build", unit="lead", total=3, mode="log", stream=io.StringIO(), is_tty=False)

    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download")
    reporter.start_transfer("ThreadPoolExecutor-0_0", file_label="a.grib2", total_bytes=1024)
    reporter.update_transfer("ThreadPoolExecutor-0_0", bytes_downloaded=512, total_bytes=1024)

    reporter.start_worker("ThreadPoolExecutor-0_1", label="20260101T1500Z f002", phase="open")
    reporter.start_transfer("ThreadPoolExecutor-0_1", file_label="b.grib2", total_bytes=1024)
    reporter.update_transfer("ThreadPoolExecutor-0_1", bytes_downloaded=512, total_bytes=1024)

    active_rate = reporter.state.workers["ThreadPoolExecutor-0_0"].transfer.display_rate_bps()
    aggregate_rate = reporter.state.aggregate_transfer_bps

    assert active_rate is not None and active_rate > 0
    assert aggregate_rate is not None
    assert abs(aggregate_rate - active_rate) < 1e-6


def test_transfer_state_display_rate_drops_stale_bursts():
    with mock.patch("tools.weather.progress.time.perf_counter", side_effect=[100.0, 100.0, 100.5, 100.5]):
        from tools.weather.progress import TransferState

        transfer = TransferState()
        transfer.update(bytes_downloaded=1024, total_bytes=2048)

    assert transfer.display_rate_bps(now=101.0) and transfer.display_rate_bps(now=101.0) > 0
    assert transfer.display_rate_bps(now=105.0) == 0.0


def test_progress_reporter_retry_state_clears_transient_error_after_recovery():
    stream = TtyStringIO()
    reporter = create_progress_reporter("HRRR 2023-01", unit="task", total=1, mode="dashboard", stream=stream, is_tty=True)

    reporter.start_worker("ThreadPoolExecutor-0_0", label="2023-01-01 c05 f05", phase="download")
    reporter.set_worker_attempt("ThreadPoolExecutor-0_0", attempt=1, max_attempts=3)
    reporter.schedule_retry(
        "ThreadPoolExecutor-0_0",
        attempt=2,
        max_attempts=3,
        delay_seconds=2.0,
        message="BrokenPipeError: [Errno 32] Broken pipe",
        error_class="broken_pipe",
    )
    retry_frame = strip_ansi(stream.getvalue())
    assert "[RETRY]" in retry_frame
    assert "a2/3" in retry_frame

    reporter.start_retry("ThreadPoolExecutor-0_0", attempt=2, max_attempts=3)
    reporter.recover_worker("ThreadPoolExecutor-0_0", message="2023-01-01 c05 f05 recovered a2/3")
    reporter.complete_worker("ThreadPoolExecutor-0_0", message="2023-01-01 c05 f05 ok")
    reporter.close(status="done")

    final = strip_ansi(stream.getvalue())
    assert "BrokenPipeError" not in final
    assert "recovered a2/3" in final


def test_progress_reporter_refresh_rerenders_dashboard_retry_wait():
    stream = TtyStringIO()
    reporter = create_progress_reporter("HRRR 2023-01", unit="task", total=1, mode="dashboard", stream=stream, is_tty=True)

    reporter.start_worker("ThreadPoolExecutor-0_0", label="2023-01-01 c05 f05", phase="download")
    reporter.schedule_retry(
        "ThreadPoolExecutor-0_0",
        attempt=2,
        max_attempts=3,
        delay_seconds=2.0,
        message="BrokenPipeError: [Errno 32] Broken pipe",
        error_class="broken_pipe",
    )
    before = stream.getvalue()
    reporter.refresh(force=True)
    after = stream.getvalue()

    assert after != before
    assert "[RETRY]" in strip_ansi(after)
    assert "BrokenPipeError" not in strip_ansi(after)
    assert "retry in" in strip_ansi(after)


def test_progress_reporter_fail_worker_emits_standalone_error_line(capsys):
    reporter = create_progress_reporter("NBM build", unit="lead", total=1, mode="dashboard", stream=TtyStringIO(), is_tty=True)

    reporter.start_worker("ThreadPoolExecutor-0_0", label="20260101T1500Z f001", phase="download")
    reporter.fail_worker("ThreadPoolExecutor-0_0", message="20260101T1500Z f001 boom")

    captured = capsys.readouterr()
    assert "[error] 20260101T1500Z f001 boom" in captured.err


def test_progress_reporter_add_skipped_updates_counts():
    reporter = create_progress_reporter("HRRR 2023-01", unit="task", total=4, mode="log", stream=io.StringIO(), is_tty=False)

    reporter.add_skipped(2, message="skipped=2")

    assert reporter.state.skipped == 2
    assert reporter.state.queued == 2
    assert reporter.state.eta_completed == 0
    assert reporter.state.overall_eta_seconds is None


def test_progress_reporter_completed_outcomes_affect_eta():
    reporter = create_progress_reporter("HRRR 2023-01", unit="task", total=4, mode="log", stream=io.StringIO(), is_tty=False)

    reporter.record_outcome("completed", count=1, message="done=1")

    assert reporter.state.completed == 1
    assert reporter.state.eta_completed == 1
    assert reporter.state.overall_eta_seconds is not None


def test_progress_reporter_bulk_skips_do_not_make_eta_optimistic():
    reporter = create_progress_reporter("NBM monthly", unit="day", total=500, mode="log", stream=io.StringIO(), is_tty=False)
    reporter.add_skipped(400, message="valid_on_disk=400", affects_eta=False)
    reporter.record_outcome("completed", count=4, message="real_days=4")

    reporter.state.throughput_samples.clear()
    reporter.state.throughput_samples.append((0.0, 0))
    reporter.state.throughput_samples.append((60.0, 4))

    assert reporter.state.completed_total == 404
    assert reporter.state.eta_completed == 4
    assert reporter.state.overall_eta_seconds == 1440.0


def test_batch_dashboard_eta_uses_day_timing_and_parallelism():
    stream = TtyStringIO()
    reporter = create_progress_reporter(
        "NBM Batch Backfill",
        unit="day",
        total=500,
        mode="dashboard",
        stream=stream,
        is_tty=True,
        dashboard_kind="nbm_batch",
    )
    reporter.set_metrics(PlanDays=4, FreeGB="80.0")
    reporter.add_skipped(400, message="valid_on_disk=400", affects_eta=False)
    reporter.record_batch_timing(full_day_seconds=60.0)

    assert isinstance(reporter.renderer, progress_mod.BatchDashboardRenderer)
    assert reporter.renderer._batch_eta_seconds(reporter.state) == 1500.0


def test_progress_reporter_dashboard_hotkey_requests_pause_once():
    calls: list[str] = []
    reporter = create_progress_reporter(
        "NBM build",
        unit="lead",
        total=2,
        mode="dashboard",
        stream=TtyStringIO(),
        is_tty=True,
        on_pause_request=lambda reason="operator": calls.append(reason),
    )

    assert isinstance(reporter.renderer, progress_mod.LiveTerminalDashboardRenderer)
    reporter.renderer._handle_keypress("p")
    reporter.renderer._handle_keypress("p")

    assert calls == ["operator"]
    assert reporter.is_pause_requested() is True


def test_progress_reporter_dashboard_renders_pausing_and_paused_status():
    reporter = create_progress_reporter("NBM build", unit="lead", total=1, mode="dashboard", stream=TtyStringIO(), is_tty=True)

    reporter.request_pause(reason="operator")
    reporter.refresh(force=True)
    pausing = strip_ansi(reporter.stream.getvalue())
    assert "PAUSING" in pausing
    assert "Pause requested: draining admitted work" in pausing

    reporter.mark_paused(reason="operator")
    reporter.refresh(force=True)
    reporter.close(status="paused")
    paused = strip_ansi(reporter.stream.getvalue())
    assert "PAUSED" in paused
    assert "Paused: safe to exit" in paused


def test_progress_reporter_pause_control_file_requests_pause_and_removes_file(tmp_path):
    control_file = tmp_path / "pause.request"
    reporter = create_progress_reporter(
        "NBM build",
        unit="lead",
        total=1,
        mode="log",
        stream=io.StringIO(),
        is_tty=False,
        pause_control_file=str(control_file),
    )

    control_file.touch()
    deadline = time.time() + 2.0
    while time.time() < deadline and not reporter.is_pause_requested():
        time.sleep(0.05)

    reporter.close(status="paused")

    assert reporter.is_pause_requested() is True
    assert control_file.exists() is False
