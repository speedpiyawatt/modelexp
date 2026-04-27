from __future__ import annotations

import importlib.util
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "weather" / "run_verification_suite.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verification_runner = load_module("weather_run_verification_suite_test", RUNNER_PATH)


def test_classify_live_stage_status_marks_network_errors_degraded():
    status = verification_runner.classify_live_stage_status(
        returncode=1,
        stdout="",
        stderr="ConnectionError: Failed to resolve source host",
    )
    assert status == "degraded"


def test_build_summary_combines_lane_statuses(tmp_path):
    local_stage = verification_runner.StageResult(name="pytest_all", status="pass", elapsed_seconds=0.1)
    live_stage = verification_runner.StageResult(name="nbm_live_download", status="degraded", elapsed_seconds=0.2)

    summary = verification_runner.build_summary(
        local_stages=[local_stage],
        live_stages=[live_stage],
        output_dir=tmp_path,
    )

    assert summary["local_contract_lane"]["status"] == "pass"
    assert summary["live_source_lane"]["status"] == "degraded"
    assert summary["overall_status"] == "pass_local"
    assert summary["summary_json_path"] == str(tmp_path / "summary.json")


def test_write_compare_inputs_writes_expected_parquet_files(tmp_path):
    paths = verification_runner.write_compare_inputs(tmp_path)
    for path in paths.values():
        assert path.exists()


def test_main_local_only_writes_summary_json(tmp_path, monkeypatch, capsys):
    local_stage = verification_runner.StageResult(name="pytest_all", status="pass", elapsed_seconds=0.1)

    monkeypatch.setattr(verification_runner, "run_local_contract_lane", lambda output_dir: [local_stage])
    monkeypatch.setattr(verification_runner, "run_live_source_lane", lambda output_dir, args: [])
    monkeypatch.setattr(
        verification_runner,
        "parse_args",
        lambda: verification_runner.argparse.Namespace(
            output_dir=tmp_path,
            json=True,
            local_only=True,
            live_only=False,
            hrrr_date="2025-04-11",
            hrrr_cycle=12,
            hrrr_forecast_hour=0,
            nbm_date="2025-04-11",
            nbm_cycle="04",
            nbm_forecast_hour=1,
        ),
    )

    exit_code = verification_runner.main()
    captured = capsys.readouterr()
    summary = json.loads(captured.out)

    assert exit_code == 0
    assert summary["local_contract_lane"]["status"] == "pass"
    assert summary["live_source_lane"]["status"] == "skipped"
    assert (tmp_path / "summary.json").exists()
