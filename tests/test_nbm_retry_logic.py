from __future__ import annotations

import argparse
import datetime as dt
import io
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
NBM_DIR = ROOT / "tools" / "nbm"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(NBM_DIR) not in sys.path:
    sys.path.insert(0, str(NBM_DIR))


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nbm_pipeline = load_module("nbm_pipeline_retry_test", NBM_DIR / "build_grib2_features.py")


def make_cycle_plan() -> nbm_pipeline.CyclePlan:
    init_time_utc = dt.datetime(2026, 1, 1, 5, 0, tzinfo=dt.timezone.utc)
    return nbm_pipeline.CyclePlan(
        init_time_utc=init_time_utc,
        init_time_local=init_time_utc.astimezone(nbm_pipeline.NY_TZ),
        cycle="05",
        selected_target_dates=(dt.date(2026, 1, 1),),
    )


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        top=nbm_pipeline.CROP_BOUNDS["top"],
        bottom=nbm_pipeline.CROP_BOUNDS["bottom"],
        left=nbm_pipeline.CROP_BOUNDS["left"],
        right=nbm_pipeline.CROP_BOUNDS["right"],
        region="co",
        overwrite=False,
        keep_downloads=False,
        keep_reduced=False,
        range_merge_gap_bytes=nbm_pipeline.DEFAULT_RANGE_MERGE_GAP_BYTES,
        max_task_attempts=3,
        retry_backoff_seconds=2.0,
        retry_max_backoff_seconds=30.0,
        scratch_dir=None,
    )


def test_process_unit_retries_transient_failure_then_recovers(monkeypatch):
    args = make_args()
    cycle_plan = make_cycle_plan()
    calls = {"count": 0}

    def fake_process_unit_once(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return nbm_pipeline.UnitResult(
                wide_row=None,
                wide_rows=[],
                long_rows=[],
                provenance_rows=[],
                manifest_row={
                    "extraction_status": "error:BrokenPipeError",
                    "warnings": "BrokenPipeError: [Errno 32] Broken pipe",
                    "attempt_count": 1,
                    "retried": False,
                    "retry_recovered": False,
                    "final_error_class": None,
                },
            )
        return nbm_pipeline.UnitResult(
            wide_row={"valid_time_utc": "2026-01-01T06:00:00+00:00"},
            wide_rows=[{"valid_time_utc": "2026-01-01T06:00:00+00:00"}],
            long_rows=[],
            provenance_rows=[],
            manifest_row={
                "extraction_status": "ok",
                "warnings": "",
                "attempt_count": 1,
                "retried": False,
                "retry_recovered": False,
                "final_error_class": None,
            },
        )

    monkeypatch.setattr(nbm_pipeline, "_process_unit_once", fake_process_unit_once)
    monkeypatch.setattr(nbm_pipeline.time, "sleep", lambda _seconds: None)
    reporter = nbm_pipeline.create_progress_reporter("NBM build", unit="lead", total=1, mode="log", stream=io.StringIO(), is_tty=False)

    result = nbm_pipeline.process_unit(
        args=args,
        client=object(),
        cycle_plan=cycle_plan,
        lead_hour=1,
        reporter=reporter,
    )

    assert calls["count"] == 2
    assert result.manifest_row["extraction_status"] == "ok"
    assert result.manifest_row["attempt_count"] == 2
    assert result.manifest_row["retry_recovered"] is True


def test_process_unit_records_failure_context(monkeypatch):
    args = make_args()
    cycle_plan = make_cycle_plan()

    def fake_process_unit_once(**kwargs):
        return nbm_pipeline.UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row={
                "extraction_status": "error:BrokenPipeError",
                "warnings": "BrokenPipeError: [Errno 32] Broken pipe",
                "attempt_count": 1,
                "retried": False,
                "retry_recovered": False,
                "final_error_class": None,
                "last_error_message": None,
                "raw_file_path": "/tmp/raw.grib2",
                "idx_file_path": "/tmp/raw.grib2.idx",
                "reduced_file_path": "/tmp/reduced.grib2",
            },
        )

    monkeypatch.setattr(nbm_pipeline, "_process_unit_once", fake_process_unit_once)
    monkeypatch.setattr(nbm_pipeline.time, "sleep", lambda _seconds: None)

    result = nbm_pipeline.process_unit(
        args=args,
        client=object(),
        cycle_plan=cycle_plan,
        lead_hour=1,
        reporter=None,
    )

    assert result.manifest_row["attempt_count"] == 3
    assert result.manifest_row["final_error_class"] == "broken_pipe"
    assert "BrokenPipeError" in result.manifest_row["last_error_message"]
    assert result.manifest_row["raw_file_path"] == "/tmp/raw.grib2"
