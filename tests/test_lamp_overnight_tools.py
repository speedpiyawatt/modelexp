from __future__ import annotations

import datetime as dt
import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAMP_DIR = ROOT / "tools" / "lamp"
FIXTURES = LAMP_DIR / "data" / "fixtures" / "parser_samples"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LAMP_DIR) not in sys.path:
    sys.path.insert(0, str(LAMP_DIR))

BUILD_PATH = LAMP_DIR / "build_lamp_klga_features.py"
OVERNIGHT_PATH = LAMP_DIR / "build_lamp_overnight_features.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lamp_build = load_module("lamp_build_for_overnight_test", BUILD_PATH)
lamp_overnight = load_module("lamp_overnight_test", OVERNIGHT_PATH)


def write_issue_artifacts(
    root: pathlib.Path,
    *,
    init_time_utc: str,
    init_time_local: str,
    rows: list[dict[str, object]],
    manifest_wide_output_path: str | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    timestamp = pd.Timestamp(init_time_utc)
    issue_root = root / "station_id=KLGA" / f"date_utc={timestamp.date().isoformat()}" / f"cycle={timestamp.strftime('%H%M')}"
    issue_root.mkdir(parents=True, exist_ok=True)
    wide_path = issue_root / "lamp.wide.parquet"
    manifest_path = issue_root / "lamp.manifest.parquet"
    pd.DataFrame.from_records(rows).to_parquet(wide_path, index=False)
    pd.DataFrame.from_records(
        [
            {
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "station_id": "KLGA",
                "status": "ok",
                "extraction_status": "ok",
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "wide_output_path": manifest_wide_output_path if manifest_wide_output_path is not None else str(wide_path),
                "manifest_parquet_path": str(manifest_path),
            }
        ]
    ).to_parquet(manifest_path, index=False)
    return wide_path, manifest_path


def make_wide_rows(
    *,
    init_time_utc: str,
    init_time_local: str,
    target_date_local: str,
    tmp_offset: float = 0.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for hour, tmp, dpt, wsp, wdr, wgs, cld, cig, vis, obv, typ, p01, p06, p12, pos, poz in (
        (6, 54.0, 45.0, 8.0, 220.0, "15", "BKN", 12.0, 6.0, None, None, 10.0, 20.0, 30.0, 15.0, 5.0),
        (9, 60.0, 47.0, 10.0, 230.0, "18", "OVC", 8.0, 2.0, "HZ", "R", 20.0, 25.0, 35.0, 25.0, 10.0),
        (12, 68.0, 50.0, 12.0, 240.0, "20", "BKN", 10.0, 4.0, None, "R", 15.0, 30.0, 40.0, 30.0, 12.0),
        (15, 72.0, 52.0, 13.0, 250.0, "22", "SCT", 20.0, 6.0, None, None, 5.0, 10.0, 20.0, 8.0, 3.0),
        (18, 70.0, 51.0, 11.0, 255.0, "20", "SCT", 25.0, 6.0, None, None, 3.0, 8.0, 15.0, 5.0, 2.0),
        (21, 64.0, 49.0, 9.0, 245.0, "16", "FEW", 30.0, 6.0, None, None, 2.0, 5.0, 10.0, 4.0, 1.0),
    ):
        valid_local = pd.Timestamp(f"{target_date_local}T{hour:02d}:00:00-04:00")
        valid_utc = valid_local.tz_convert("UTC")
        rows.append(
            {
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "fallback_used_any": False,
                "missing_optional_any": False,
                "missing_optional_fields_count": 0,
                "station_id": "KLGA",
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "init_date_local": pd.Timestamp(init_time_local).date().isoformat(),
                "valid_time_utc": valid_utc.isoformat(),
                "valid_time_local": valid_local.isoformat(),
                "valid_date_local": target_date_local,
                "forecast_hour": int((valid_utc - pd.Timestamp(init_time_utc)).total_seconds() // 3600),
                "tmp": tmp + tmp_offset,
                "dpt": dpt,
                "wsp": wsp,
                "wdr": wdr,
                "wgs": wgs,
                "cld": cld,
                "cig": cig,
                "vis": vis,
                "obv": obv,
                "typ": typ,
                "p01": p01,
                "p06": p06,
                "p12": p12,
                "pos": pos,
                "poz": poz,
            }
        )
    return rows


def test_select_issues_for_target_uses_latest_issue_at_or_before_cutoff():
    issue_catalog = pd.DataFrame.from_records(
        [
            {
                "wide_path": "/tmp/a",
                "manifest_path": "/tmp/a.manifest",
                "init_time_utc": pd.Timestamp("2026-04-12T03:45:00+00:00"),
                "init_time_local": pd.Timestamp("2026-04-11T23:45:00-04:00"),
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "station_id": "KLGA",
            },
            {
                "wide_path": "/tmp/b",
                "manifest_path": "/tmp/b.manifest",
                "init_time_utc": pd.Timestamp("2026-04-12T04:00:00+00:00"),
                "init_time_local": pd.Timestamp("2026-04-12T00:00:00-04:00"),
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "station_id": "KLGA",
            },
            {
                "wide_path": "/tmp/c",
                "manifest_path": "/tmp/c.manifest",
                "init_time_utc": pd.Timestamp("2026-04-12T04:30:00+00:00"),
                "init_time_local": pd.Timestamp("2026-04-12T00:30:00-04:00"),
                "source_model": "LAMP",
                "source_product": "lav",
                "source_version": "lamp-station-ascii-public",
                "station_id": "KLGA",
            },
        ]
    )

    selected, previous, cutoff_ts = lamp_overnight.select_issues_for_target(
        issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=dt.time.fromisoformat("00:05"),
    )

    assert cutoff_ts.isoformat() == "2026-04-12T00:05:00-04:00"
    assert str(selected["wide_path"]) == "/tmp/b"
    assert str(previous["wide_path"]) == "/tmp/a"


def test_select_issues_for_target_accepts_tz_naive_local_manifest_times():
    issue_catalog = pd.DataFrame.from_records(
        [
            {"wide_path": "/tmp/a", "manifest_path": "/tmp/a.manifest", "init_time_utc": pd.Timestamp("2026-04-12T03:45:00+00:00"), "init_time_local": pd.Timestamp("2026-04-11T23:45:00"), "source_model": "LAMP", "source_product": "lav", "source_version": "lamp-station-ascii-public", "station_id": "KLGA"},
            {"wide_path": "/tmp/b", "manifest_path": "/tmp/b.manifest", "init_time_utc": pd.Timestamp("2026-04-12T04:00:00+00:00"), "init_time_local": pd.Timestamp("2026-04-12T00:00:00"), "source_model": "LAMP", "source_product": "lav", "source_version": "lamp-station-ascii-public", "station_id": "KLGA"},
        ]
    )

    issue_catalog["init_time_local"] = issue_catalog["init_time_local"].map(lamp_overnight.parse_local_timestamp)
    selected, previous, _ = lamp_overnight.select_issues_for_target(
        issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=dt.time.fromisoformat("00:05"),
    )

    assert str(selected["wide_path"]) == "/tmp/b"
    assert str(previous["wide_path"]) == "/tmp/a"


def test_build_from_inputs_raises_cleanly_when_no_station_rows_exist(tmp_path):
    empty_ascii = tmp_path / "empty.ascii"
    empty_ascii.write_text("")

    try:
        lamp_build.build_from_inputs([empty_ascii], station_id="KLGA", output_dir=tmp_path / "out", write_long=False)
    except SystemExit as exc:
        assert "No parsed LAMP rows found" in str(exc)
    else:
        raise AssertionError("Expected SystemExit for empty parsed LAMP input")


def test_build_for_date_writes_daily_summary_and_revision_features(tmp_path):
    features_root = tmp_path / "features"
    output_dir = tmp_path / "overnight"
    previous_rows = make_wide_rows(
        init_time_utc="2026-04-12T03:45:00+00:00",
        init_time_local="2026-04-11T23:45:00-04:00",
        target_date_local="2026-04-12",
        tmp_offset=-2.0,
    )
    selected_rows = make_wide_rows(
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        target_date_local="2026-04-12",
        tmp_offset=0.0,
    )
    write_issue_artifacts(features_root, init_time_utc="2026-04-12T03:45:00+00:00", init_time_local="2026-04-11T23:45:00-04:00", rows=previous_rows)
    write_issue_artifacts(features_root, init_time_utc="2026-04-12T04:00:00+00:00", init_time_local="2026-04-12T00:00:00-04:00", rows=selected_rows)

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = lamp_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=dt.time.fromisoformat("00:05"),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)
    row = overnight_df.iloc[0]
    manifest = manifest_df.iloc[0]

    assert len(overnight_df) == 1
    assert row["selected_init_time_local"] == "2026-04-12T00:00:00-04:00"
    assert row["previous_init_time_local"] == "2026-04-11T23:45:00-04:00"
    assert bool(row["coverage_complete"]) is True
    assert int(row["missing_checkpoint_count"]) == 0
    assert float(row["day_tmp_max_f_forecast"]) == 72.0
    assert float(row["day_tmp_min_f_forecast"]) == 54.0
    assert float(row["day_tmp_range_f_forecast"]) == 18.0
    assert int(row["day_tmp_argmax_local_hour"]) == 15
    assert row["morning_cld_mode"] == "BKN"
    assert float(row["morning_cig_min_hundreds_ft"]) == 8.0
    assert float(row["morning_vis_min_miles"]) == 2.0
    assert bool(row["morning_obv_any"]) is True
    assert bool(row["morning_ifr_like_any"]) is True
    assert float(row["day_p01_max_pct"]) == 20.0
    assert float(row["day_p06_max_pct"]) == 30.0
    assert float(row["day_p12_max_pct"]) == 40.0
    assert float(row["day_pos_max_pct"]) == 30.0
    assert float(row["day_poz_max_pct"]) == 12.0
    assert bool(row["day_precip_type_any"]) is True
    assert row["day_precip_type_mode"] == "R"
    assert float(row["tmp_f_at_06"]) == 54.0
    assert float(row["tmp_f_at_21"]) == 64.0
    assert bool(row["revision_available"]) is True
    assert float(row["rev_day_tmp_max_f"]) == 2.0
    assert float(row["rev_tmp_f_at_06"]) == 2.0
    assert float(row["rev_day_p01_max_pct"]) == 0.0
    assert manifest["status"] == "ok"
    assert manifest["warning"] == ""


def test_build_for_date_without_qualifying_issue_writes_incomplete_manifest(tmp_path):
    features_root = tmp_path / "features"
    output_dir = tmp_path / "overnight"
    future_rows = make_wide_rows(
        init_time_utc="2026-04-12T04:30:00+00:00",
        init_time_local="2026-04-12T00:30:00-04:00",
        target_date_local="2026-04-12",
    )
    write_issue_artifacts(features_root, init_time_utc="2026-04-12T04:30:00+00:00", init_time_local="2026-04-12T00:30:00-04:00", rows=future_rows)

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = lamp_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=dt.time.fromisoformat("00:05"),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest = pd.read_parquet(manifest_path).iloc[0]

    assert overnight_df.empty
    assert manifest["status"] == "incomplete"
    assert manifest["extraction_status"] == "no_qualifying_issue"
    assert manifest["failure_reason"] == "no_qualifying_issue"


def test_discover_issue_catalog_accepts_repo_relative_manifest_paths(tmp_path, monkeypatch):
    features_root = tmp_path / "features"
    monkeypatch.setattr(lamp_overnight, "REPO_ROOT", tmp_path)
    repo_relative_path = "features/station_id=KLGA/date_utc=2026-04-12/cycle=0400/lamp.wide.parquet"
    wide_path, _ = write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        rows=make_wide_rows(
            init_time_utc="2026-04-12T04:00:00+00:00",
            init_time_local="2026-04-12T00:00:00-04:00",
            target_date_local="2026-04-12",
        ),
        manifest_wide_output_path=repo_relative_path,
    )

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)

    assert len(issue_catalog) == 1
    assert pathlib.Path(issue_catalog.iloc[0]["wide_path"]) == wide_path.resolve()


def test_discover_issue_catalog_accepts_manifest_relative_paths(tmp_path):
    features_root = tmp_path / "features"
    wide_path, _ = write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        rows=make_wide_rows(
            init_time_utc="2026-04-12T04:00:00+00:00",
            init_time_local="2026-04-12T00:00:00-04:00",
            target_date_local="2026-04-12",
        ),
        manifest_wide_output_path="lamp.wide.parquet",
    )

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)

    assert len(issue_catalog) == 1
    assert pathlib.Path(issue_catalog.iloc[0]["wide_path"]) == wide_path.resolve()


def test_discover_issue_catalog_skips_missing_paths(tmp_path):
    features_root = tmp_path / "features"
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        rows=make_wide_rows(
            init_time_utc="2026-04-12T04:00:00+00:00",
            init_time_local="2026-04-12T00:00:00-04:00",
            target_date_local="2026-04-12",
        ),
        manifest_wide_output_path="missing/lamp.wide.parquet",
    )

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)

    assert issue_catalog.empty


def test_smoke_builds_one_target_day_row_from_fixture_derived_issue(tmp_path):
    features_root = tmp_path / "features"
    output_dir = tmp_path / "overnight"
    inputs = [
        FIXTURES / "current_standard_2026_0430.ascii",
        FIXTURES / "current_extended_2026_0430.ascii",
    ]
    lamp_build.build_from_inputs(inputs, station_id="KLGA", output_dir=features_root, write_long=True)

    issue_catalog = lamp_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = lamp_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-13").date(),
        cutoff_local_time=dt.time.fromisoformat("00:05"),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest = pd.read_parquet(manifest_path).iloc[0]
    row = overnight_df.iloc[0]

    assert len(overnight_df) == 1
    assert row["target_date_local"] == "2026-04-13"
    assert row["selected_init_time_local"] == "2026-04-12T00:30:00-04:00"
    assert bool(row["missing_checkpoint_any"]) is True
    assert int(row["missing_checkpoint_count"]) > 0
    assert float(row["tmp_f_at_06"]) == 52.0
    assert float(row["tmp_f_at_09"]) == 58.0
    assert float(row["tmp_f_at_12"]) == 70.0
    assert manifest["status"] == "ok"
    assert "Missing target-day LAMP checkpoints" in manifest["warning"]


def test_main_advances_progress_once_per_target_date(tmp_path, monkeypatch):
    progress_instances: list[object] = []

    class RecordingProgressBar:
        def __init__(self, total, *, label=None, unit="item", **_kwargs):
            self.total = total
            self.label = label
            self.unit = unit
            self.advances = 0
            self.closed = False
            self.updates: list[tuple[str | None, str | None]] = []
            progress_instances.append(self)

        def update(self, *, stage=None, status=None, **_kwargs):
            self.updates.append((stage, status))

        def advance(self, step: int = 1, *, stage=None, status=None, **_kwargs):
            self.advances += step
            self.updates.append((stage, status))

        def close(self, *, stage=None, status=None):
            self.closed = True
            self.updates.append((stage, status))

    def fake_build_for_date(**kwargs):
        target_date_local = kwargs["target_date_local"]
        root = kwargs["output_dir"] / f"target_date_local={target_date_local.isoformat()}"
        root.mkdir(parents=True, exist_ok=True)
        overnight_path = root / "lamp.overnight.parquet"
        manifest_path = root / "lamp.overnight.manifest.parquet"
        overnight_path.write_text("")
        manifest_path.write_text("")
        return overnight_path, manifest_path

    monkeypatch.setattr(lamp_overnight, "ProgressBar", RecordingProgressBar)
    monkeypatch.setattr(
        lamp_overnight,
        "parse_args",
        lambda: SimpleNamespace(
            features_root=tmp_path / "features",
            output_dir=tmp_path / "output",
            start_local_date="2026-04-12",
            end_local_date="2026-04-14",
            cutoff_local_time="00:05",
        ),
    )
    monkeypatch.setattr(lamp_overnight, "discover_issue_catalog", lambda _root: pd.DataFrame())
    monkeypatch.setattr(lamp_overnight, "build_for_date", fake_build_for_date)

    assert lamp_overnight.main() == 0
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.total == 3
    assert progress.label == "LAMP overnight"
    assert progress.unit == "date"
    assert progress.advances == 3
    assert progress.closed is True
    assert any(stage == "complete" for stage, _ in progress.updates)
    assert any(stage == "finalize" for stage, _ in progress.updates)
