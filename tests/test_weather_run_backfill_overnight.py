from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from argparse import Namespace

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "weather" / "run_backfill_overnight.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("run_backfill_overnight_test", RUNNER_PATH)


def write_merged_day(root: pathlib.Path, target_date_local: str) -> tuple[pathlib.Path, pathlib.Path]:
    output_path, manifest_path = runner.merged_builder.output_paths_for_date(root / "training_features_overnight", target_date_local)
    df = runner.merged_builder.apply_registry_layout(
        pd.DataFrame([{"target_date_local": target_date_local, "station_id": "KLGA", "selection_cutoff_local": f"{target_date_local}T00:05:00-04:00"}])
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    manifest = runner.merged_builder.build_output_manifest_for_date(
        target_date_local=target_date_local,
        output_path=output_path,
        row_count=len(df),
        station_id="KLGA",
        cutoff_local_time="00:05",
        output_df=df,
    )
    manifest_path.write_text(json.dumps(manifest))
    return output_path, manifest_path


def write_normalized_day(root: pathlib.Path, target_date_local: str) -> tuple[pathlib.Path, pathlib.Path]:
    output_path, manifest_path = runner.normalized_builder.output_paths_for_date(
        root / "training_features_overnight_normalized", target_date_local
    )
    base = runner.merged_builder.apply_registry_layout(
        pd.DataFrame([{"target_date_local": target_date_local, "station_id": "KLGA", "selection_cutoff_local": f"{target_date_local}T00:05:00-04:00"}])
    )
    vocabularies = runner.normalized_builder.load_vocabularies(runner.normalized_builder.DEFAULT_VOCAB_PATH)
    df = runner.normalized_builder.normalize_training_features_overnight(base, vocabularies)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    manifest = runner.normalized_builder.build_date_manifest(
        target_date_local=target_date_local,
        output_path=output_path,
        output_df=df,
        vocabularies=vocabularies,
    )
    manifest_path.write_text(json.dumps(manifest))
    return output_path, manifest_path


def test_validate_wu_day_rejects_manifest_mismatch(tmp_path: pathlib.Path):
    day = runner.DayWindow(runner.parse_local_date("2026-04-12"))
    paths = runner.wu_paths(tmp_path, day)
    paths["root"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2026-04-12", "station_id": "KLGA"}]).to_parquet(paths["labels"])
    pd.DataFrame([{"target_date_local": "2026-04-11", "station_id": "KLGA"}]).to_parquet(paths["label_history"])
    pd.DataFrame([{"date_local": "2026-04-11", "station_id": "KLGA", "valid_time_local": "2026-04-11T23:00:00-04:00"}]).to_parquet(paths["obs"])
    paths["manifest"].write_text('{"start_local_date":"2026-04-13","end_local_date":"2026-04-13","label_row_count":1,"label_history_row_count":1,"obs_row_count":1}')

    assert not runner.validate_wu_day(tmp_path, day)


def test_ensure_wu_day_skips_existing_valid_day(tmp_path: pathlib.Path):
    day = runner.DayWindow(runner.parse_local_date("2026-04-12"))
    paths = runner.wu_paths(tmp_path, day)
    paths["root"].mkdir(parents=True, exist_ok=True)
    labels_df = pd.DataFrame([{"target_date_local": "2026-04-12", "station_id": "KLGA"}])
    label_history_df = pd.DataFrame([{"target_date_local": "2026-04-11", "station_id": "KLGA"}])
    obs_df = pd.DataFrame([{"date_local": "2026-04-11", "station_id": "KLGA", "valid_time_local": "2026-04-11T23:00:00-04:00"}])
    labels_df.to_parquet(paths["labels"])
    label_history_df.to_parquet(paths["label_history"])
    obs_df.to_parquet(paths["obs"])
    paths["manifest"].write_text('{"start_local_date":"2026-04-12","end_local_date":"2026-04-12","label_row_count":1,"label_history_row_count":1,"obs_row_count":1}')

    args = Namespace(
        run_root=tmp_path,
        labels_path=tmp_path / "missing_labels.parquet",
        label_history_path=None,
        obs_path=tmp_path / "missing_obs.parquet",
    )
    runner.ensure_wu_day(args, day, force=False)
    assert runner.validate_wu_day(tmp_path, day)


def test_main_honors_pause_file_after_current_day(tmp_path: pathlib.Path, monkeypatch):
    calls: list[tuple[str, str]] = []
    pause_path = tmp_path / runner.PAUSE_SENTINEL

    def fake_handler(_args, day, force):
        calls.append((runner.day_token(day), str(force)))
        if runner.day_token(day) == "2026-04-12":
            pause_path.write_text("pause")

    args = Namespace(
        run_root=tmp_path,
        labels_path=tmp_path / "labels.parquet",
        label_history_path=None,
        obs_path=tmp_path / "obs.parquet",
        start_local_date="2026-04-12",
        end_local_date="2026-04-13",
        station_id="KLGA",
        cutoff_local_time="00:05",
        nbm_workers=1,
        nbm_lead_workers=1,
        hrrr_workers=1,
        stages=["wu"],
        force_rebuild_stage=[],
        force_rebuild_date=[],
    )

    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "stage_handlers", lambda: {"wu": fake_handler})

    assert runner.main() == 0
    assert calls == [("2026-04-12", "False")]


def test_ensure_merged_day_rebuilds_when_upstream_is_newer(tmp_path: pathlib.Path, monkeypatch):
    day = runner.DayWindow(runner.parse_local_date("2026-04-12"))
    target = runner.day_token(day)
    wu = runner.wu_paths(tmp_path, day)
    wu["root"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": target, "station_id": "KLGA"}]).to_parquet(wu["labels"])
    pd.DataFrame([{"target_date_local": "2026-04-11", "station_id": "KLGA"}]).to_parquet(wu["label_history"])
    pd.DataFrame([{"date_local": "2026-04-11", "station_id": "KLGA"}]).to_parquet(wu["obs"])
    wu["manifest"].write_text(
        json.dumps({"start_local_date": target, "end_local_date": target, "label_row_count": 1, "label_history_row_count": 1, "obs_row_count": 1})
    )
    nbm_root = tmp_path / "nbm_overnight" / f"target_date_local={target}"
    nbm_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": target, "station_id": "KLGA"}]).to_parquet(nbm_root / "nbm.overnight.parquet")
    pd.DataFrame([{"status": "ok"}]).to_parquet(nbm_root / "nbm.overnight.manifest.parquet")
    lamp_root = tmp_path / "lamp_overnight" / f"target_date_local={target}"
    lamp_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": target, "station_id": "KLGA"}]).to_parquet(lamp_root / "lamp.overnight.parquet")
    pd.DataFrame([{"status": "ok", "extraction_status": "ok"}]).to_parquet(lamp_root / "lamp.overnight.manifest.parquet")
    hrrr = runner.hrrr_summary_paths(tmp_path, day)
    hrrr["summary_root"].mkdir(parents=True, exist_ok=True)
    hrrr["state_root"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": target}]).to_parquet(hrrr["summary"], index=False)
    hrrr["manifest_json"].write_text(json.dumps({"complete": True, "expected_task_count": 1, "completed_task_keys": ["a"]}))
    pd.DataFrame([{"status": "ok"}]).to_parquet(hrrr["manifest_parquet"])
    write_merged_day(tmp_path, target)
    output_path, manifest_path = runner.merged_builder.output_paths_for_date(tmp_path / "training_features_overnight", target)
    newer = output_path.stat().st_mtime + 10
    for path in (wu["manifest"], nbm_root / "nbm.overnight.manifest.parquet"):
        os.utime(path, (newer, newer))

    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        calls.append(command)
        write_merged_day(tmp_path, target)

    monkeypatch.setattr(runner, "run_command", fake_run)
    args = Namespace(run_root=tmp_path, station_id="KLGA", cutoff_local_time="00:05")
    runner.ensure_merged_day(args, day, force=False)

    assert calls
    assert manifest_path.exists()
    assert runner.validate_merged_day(tmp_path, day, allow_empty=False)


def test_ensure_normalized_day_rebuilds_when_merged_is_newer(tmp_path: pathlib.Path, monkeypatch):
    day = runner.DayWindow(runner.parse_local_date("2026-04-12"))
    target = runner.day_token(day)
    merged_output, _ = write_merged_day(tmp_path, target)
    normalized_output, normalized_manifest = write_normalized_day(tmp_path, target)
    newer = normalized_output.stat().st_mtime + 10
    os.utime(merged_output, (newer, newer))

    calls: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        calls.append(command)
        write_normalized_day(tmp_path, target)

    monkeypatch.setattr(runner, "run_command", fake_run)
    args = Namespace(run_root=tmp_path)
    runner.ensure_normalized_day(args, day, force=False)

    assert calls
    assert normalized_manifest.exists()
    assert runner.validate_normalized_day(tmp_path, day, allow_empty=False)


def test_ensure_hrrr_day_rewrites_manifest_paths_to_daily_outputs(tmp_path: pathlib.Path, monkeypatch):
    day = runner.DayWindow(runner.parse_local_date("2026-04-12"))
    target = runner.day_token(day)

    def fake_run(_command: list[str]) -> None:
        tmp_root = tmp_path / "hrrr_tmp" / target
        output_dir = tmp_root / "output"
        summary_dir = tmp_root / "summary"
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        month_id = day.target_date_local.strftime("%Y-%m")
        pd.DataFrame([{"target_date_local": target, "station_id": "KLGA"}]).to_parquet(summary_dir / f"{month_id}.parquet", index=False)
        (output_dir / f"{month_id}.manifest.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "expected_task_count": 1,
                    "completed_task_keys": ["task-1"],
                    "summary_parquet_path": str(summary_dir / f"{month_id}.parquet"),
                    "output_dir": str(output_dir),
                    "summary_output_dir": str(summary_dir),
                }
            )
        )
        pd.DataFrame(
            [
                {
                    "status": "ok",
                    "summary_parquet_path": str(summary_dir / f"{month_id}.parquet"),
                    "output_dir": str(output_dir),
                    "summary_output_dir": str(summary_dir),
                }
            ]
        ).to_parquet(output_dir / f"{month_id}.manifest.parquet", index=False)

    monkeypatch.setattr(runner, "run_command", fake_run)
    args = Namespace(run_root=tmp_path, hrrr_workers=1)
    runner.ensure_hrrr_day(args, day, force=False)

    paths = runner.hrrr_summary_paths(tmp_path, day)
    manifest = json.loads(paths["manifest_json"].read_text())
    manifest_df = pd.read_parquet(paths["manifest_parquet"])

    assert manifest["summary_parquet_path"] == str(paths["summary"])
    assert manifest["manifest_json_path"] == str(paths["manifest_json"])
    assert manifest["manifest_parquet_path"] == str(paths["manifest_parquet"])
    assert str(manifest_df.iloc[0]["summary_parquet_path"]) == str(paths["summary"])
    assert pd.isna(manifest_df.iloc[0]["output_dir"])
    assert not (tmp_path / "hrrr_tmp" / target).exists()
