from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "weather" / "run_server_overnight_stage.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("weather_run_server_overnight_stage_test", RUNNER_PATH)


def test_summarize_command_issue_marks_network_problems_degraded():
    status, issue_class = runner.summarize_command_issue(
        {
            "command": ["python3", "demo.py"],
            "returncode": 1,
            "stdout": "",
            "stderr": "ConnectionError: Failed to resolve source host",
            "timed_out": False,
        }
    )

    assert status == "degraded"
    assert issue_class == "network_or_upstream_degraded"


def test_classify_probe_entry_accepts_missing_output_when_audits_pass():
    status, issue_class = runner.classify_probe_entry(
        {
            "steps": [
                {
                    "command": ["python3", "ok.py"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "",
                    "timed_out": False,
                }
            ],
            "source_summary": {"status": "missing_output"},
            "merge_summary": {
                "merged_row_count": 1,
                "normalized_row_count": 1,
                "merged_checks": {
                    "non_empty_table": True,
                    "core_columns_present": True,
                    "registry_match": True,
                    "prefix_policy_ok": True,
                    "unique_target_date_station_key": True,
                },
                "normalized_checks": {
                    "non_empty_table": True,
                    "core_columns_present": True,
                    "registry_match": True,
                    "prefix_policy_ok": True,
                    "unique_target_date_station_key": True,
                    "no_string_default_model_inputs": True,
                    "unit_suffix_policy_ok": True,
                },
            },
        }
    )

    assert status == "pass"
    assert issue_class is None


def test_classify_probe_entry_fails_after_repeated_timeout():
    status, issue_class = runner.classify_probe_entry(
        {
            "steps": [
                {"command": ["python3"], "returncode": None, "stdout": "", "stderr": "", "timed_out": True},
                {"command": ["python3"], "returncode": None, "stdout": "", "stderr": "", "timed_out": True},
            ],
            "merge_summary": {
                "merged_row_count": 1,
                "normalized_row_count": 1,
                "merged_checks": {"non_empty_table": True},
                "normalized_checks": {"non_empty_table": True},
            },
        }
    )

    assert status == "fail"
    assert issue_class == "performance_blocker"


def test_execute_requested_stages_reuses_prior_successful_wu_stage(tmp_path, monkeypatch):
    args = runner.argparse.Namespace(
        stage="all",
        output_root=tmp_path,
        history_dir=tmp_path / "history",
        short_window_start="2025-04-11",
        short_window_end="2025-04-13",
        probe_timeout_seconds=900,
        nbm_workers=1,
        hrrr_workers=1,
        allow_degraded_live=False,
        resume=True,
    )
    runner.write_json(runner.stage_path(tmp_path, "wu"), runner.stage_result("wu", status="pass", started_at_utc="2026-04-14T00:00:00+00:00"))

    monkeypatch.setattr(runner, "run_wu_stage", lambda _args: (_ for _ in ()).throw(AssertionError("WU should be reused")))
    monkeypatch.setattr(runner, "run_smoke_stage", lambda _args: runner.stage_result("smoke", status="pass", started_at_utc="2026-04-14T00:00:01+00:00"))
    monkeypatch.setattr(runner, "run_short_window_stage", lambda _args: runner.stage_result("short-window", status="pass", started_at_utc="2026-04-14T00:00:02+00:00"))

    summary = runner.execute_requested_stages(args)

    assert summary["overall_status"] == "pass"
    assert summary["stages"]["wu"]["reused"] is True
    assert summary["stages"]["smoke"]["status"] == "pass"
    assert summary["stages"]["short-window"]["status"] == "pass"


def test_execute_requested_stages_blocks_short_window_without_smoke(tmp_path):
    args = runner.argparse.Namespace(
        stage="short-window",
        output_root=tmp_path,
        history_dir=tmp_path / "history",
        short_window_start="2025-04-11",
        short_window_end="2025-04-13",
        probe_timeout_seconds=900,
        nbm_workers=1,
        hrrr_workers=1,
        allow_degraded_live=False,
        resume=False,
    )
    runner.write_json(runner.stage_path(tmp_path, "wu"), runner.stage_result("wu", status="pass", started_at_utc="2026-04-14T00:00:00+00:00"))

    summary = runner.execute_requested_stages(args)
    stage_payload = json.loads((tmp_path / "stage_short_window.json").read_text())

    assert summary["overall_status"] == "fail"
    assert stage_payload["stage"] == "short-window"
    assert "required prerequisite stage smoke" in stage_payload["details"]["error"]


def test_source_stage_config_uses_short_window_lamp_archive_cycles(tmp_path):
    args = runner.argparse.Namespace(
        stage="short-window",
        output_root=tmp_path,
        history_dir=tmp_path / "history",
        short_window_start="2025-04-11",
        short_window_end="2025-04-13",
        probe_timeout_seconds=900,
        nbm_workers=1,
        hrrr_workers=1,
        allow_degraded_live=False,
        resume=False,
    )

    config = next(item for item in runner.source_stage_config(args) if item["name"] == "lamp")
    fetch_command = config["commands"][0]

    assert fetch_command[:3] == [runner.sys.executable, "tools/lamp/fetch_lamp.py", "archive"]
    assert "--start-utc-date" in fetch_command
    assert "--end-utc-date" in fetch_command
    assert "--year" not in fetch_command
    assert fetch_command.count("--cycle") == 3
    assert "0230" in fetch_command
    assert "0330" in fetch_command
    assert "0430" in fetch_command
