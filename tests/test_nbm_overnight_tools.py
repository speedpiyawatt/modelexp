from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
NBM_OVERNIGHT_PATH = ROOT / "tools" / "nbm" / "build_nbm_overnight_features.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


nbm_overnight = load_module("nbm_overnight_test", NBM_OVERNIGHT_PATH)


def write_issue_artifacts(
    root: pathlib.Path,
    *,
    init_time_utc: str,
    init_time_local: str,
    target_date_local: str,
    wide_rows: list[dict[str, object]],
    processed_timestamp_utc: str | None = None,
) -> None:
    issue_root = root / "metadata" / "manifest" / f"init_date_local={pd.Timestamp(init_time_local).date().isoformat()}"
    issue_root.mkdir(parents=True, exist_ok=True)
    wide_path = root / "features" / "wide" / f"{pd.Timestamp(init_time_utc).strftime('%Y%m%dT%H%MZ')}.parquet"
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(wide_rows).to_parquet(wide_path, index=False)
    manifest_path = issue_root / f"cycle_{pd.Timestamp(init_time_utc).strftime('%Y%m%dT%H%MZ')}_manifest.parquet"
    pd.DataFrame.from_records(
        [
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "processed_timestamp_utc": processed_timestamp_utc or init_time_utc,
                "valid_date_local": target_date_local,
                "extraction_status": "ok",
                "wide_output_paths": str(wide_path),
            }
        ]
    ).to_parquet(manifest_path, index=False)


def make_wide_rows(*, init_time_utc: str, init_time_local: str, target_date_local: str, tmp_offset: float = 0.0) -> list[dict[str, object]]:
    rows = []
    for hour, tmp, dpt, rh, wind, wdir, gust, tcdc, dswrf, apcp, vis, ceil, cape, tmax, tmin, pcpdur, pwther, tstm, ptype, thunc, vrate in (
        (6, 280.0, 276.0, 60.0, 6.0, 210.0, 8.0, 50.0, 20.0, 0.0, 12000.0, 900.0, 10.0, 287.0, 274.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0),
        (9, 284.0, 278.0, 55.0, 8.0, 220.0, 10.0, 40.0, 200.0, 0.0, 12000.0, 1000.0, 20.0, 291.0, 275.0, 1.0, 3.0, 5.0, 1.0, 1.0, 150.0),
        (12, 289.0, 279.0, 50.0, 9.0, 230.0, 11.0, 35.0, 400.0, 0.1, 12000.0, 1100.0, 30.0, 294.0, 276.0, 2.0, 3.0, 15.0, 2.0, 2.0, 200.0),
        (15, 292.0, 280.0, 48.0, 10.0, 240.0, 12.0, 30.0, 500.0, 0.0, 12000.0, 1300.0, 50.0, 295.0, 277.0, 0.0, 2.0, 35.0, 2.0, 1.0, 250.0),
        (18, 290.0, 279.0, 52.0, 9.0, 245.0, 11.0, 45.0, 100.0, 0.0, 10000.0, 1400.0, 25.0, 293.0, 276.0, 0.0, 0.0, 10.0, 0.0, 0.0, 180.0),
        (21, 286.0, 278.0, 58.0, 7.0, 250.0, 9.0, 60.0, 0.0, 0.0, 9000.0, 1500.0, 5.0, 289.0, 275.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0),
    ):
        valid_local = pd.Timestamp(f"{target_date_local}T{hour:02d}:00:00-04:00")
        valid_utc = valid_local.tz_convert("UTC")
        rows.append(
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "valid_time_utc": valid_utc.isoformat(),
                "valid_time_local": valid_local.isoformat(),
                "init_date_local": pd.Timestamp(init_time_local).date().isoformat(),
                "valid_date_local": target_date_local,
                "forecast_hour": int((valid_utc - pd.Timestamp(init_time_utc)).total_seconds() // 3600),
                "tmp": tmp + tmp_offset,
                "dpt": dpt,
                "rh": rh,
                "wind": wind,
                "wdir": wdir,
                "gust": gust,
                "tcdc": tcdc,
                "dswrf": dswrf,
                "apcp": apcp,
                "vis": vis,
                "ceil": ceil,
                "cape": cape,
                "tmax": tmax + tmp_offset,
                "tmax_nb3_max": tmax + tmp_offset + 0.4,
                "tmax_nb7_max": tmax + tmp_offset + 0.8,
                "tmax_crop_max": tmax + tmp_offset + 1.2,
                "tmin": tmin + tmp_offset,
                "tmin_nb3_min": tmin + tmp_offset - 0.4,
                "tmin_nb7_min": tmin + tmp_offset - 0.8,
                "tmin_crop_min": tmin + tmp_offset - 1.2,
                "pcpdur": pcpdur,
                "pwther": pwther,
                "tstm": tstm,
                "ptype": ptype,
                "thunc": thunc,
                "vrate": vrate,
                "tmp_nb3_mean": tmp + 0.3,
                "tmp_nb7_mean": tmp + 0.6,
                "tmp_crop_mean": tmp + 1.0,
                "tcdc_crop_mean": tcdc + 5.0,
                "dswrf_crop_max": dswrf + 20.0,
                "wind_nb7_mean": wind + 0.5,
            }
        )
    return rows


def test_select_issue_for_target_uses_latest_cutoff_eligible_issue():
    issue_catalog = pd.DataFrame.from_records(
        [
            {"wide_path": "/tmp/a", "source_model": "NBM", "source_product": "grib2-core", "source_version": "nbm-grib2-core-public", "station_id": "KLGA", "init_time_utc": pd.Timestamp("2026-04-12T03:45:00+00:00"), "init_time_local": pd.Timestamp("2026-04-11T23:45:00-04:00"), "valid_date_local": "2026-04-12"},
            {"wide_path": "/tmp/b", "source_model": "NBM", "source_product": "grib2-core", "source_version": "nbm-grib2-core-public", "station_id": "KLGA", "init_time_utc": pd.Timestamp("2026-04-12T04:00:00+00:00"), "init_time_local": pd.Timestamp("2026-04-12T00:00:00-04:00"), "valid_date_local": "2026-04-12"},
            {"wide_path": "/tmp/c", "source_model": "NBM", "source_product": "grib2-core", "source_version": "nbm-grib2-core-public", "station_id": "KLGA", "init_time_utc": pd.Timestamp("2026-04-12T04:30:00+00:00"), "init_time_local": pd.Timestamp("2026-04-12T00:30:00-04:00"), "valid_date_local": "2026-04-12"},
        ]
    )

    selected, cutoff_ts = nbm_overnight.select_issue_for_target(
        issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
    )

    assert cutoff_ts.isoformat() == "2026-04-12T00:05:00-04:00"
    assert str(selected["wide_path"]) == "/tmp/b"


def test_select_issue_for_target_accepts_tz_naive_local_manifest_times():
    issue_catalog = pd.DataFrame.from_records(
        [
            {"wide_path": "/tmp/a", "source_model": "NBM", "source_product": "grib2-core", "source_version": "nbm-grib2-core-public", "station_id": "KLGA", "init_time_utc": pd.Timestamp("2026-04-12T03:45:00+00:00"), "init_time_local": pd.Timestamp("2026-04-11T23:45:00"), "valid_date_local": "2026-04-12"},
            {"wide_path": "/tmp/b", "source_model": "NBM", "source_product": "grib2-core", "source_version": "nbm-grib2-core-public", "station_id": "KLGA", "init_time_utc": pd.Timestamp("2026-04-12T04:00:00+00:00"), "init_time_local": pd.Timestamp("2026-04-12T00:00:00"), "valid_date_local": "2026-04-12"},
        ]
    )

    issue_catalog["init_time_local"] = issue_catalog["init_time_local"].map(nbm_overnight.parse_local_timestamp)
    selected, _ = nbm_overnight.select_issue_for_target(
        issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
    )

    assert str(selected["wide_path"]) == "/tmp/b"


def test_select_issue_for_target_rejects_issue_processed_after_cutoff():
    issue_catalog = pd.DataFrame.from_records(
        [
            {
                "wide_path": "/tmp/a",
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": pd.Timestamp("2026-04-12T03:45:00+00:00"),
                "init_time_local": pd.Timestamp("2026-04-11T23:45:00-04:00"),
                "available_time_local": pd.Timestamp("2026-04-11T23:55:00-04:00"),
                "valid_date_local": "2026-04-12",
            },
            {
                "wide_path": "/tmp/b",
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": pd.Timestamp("2026-04-12T04:00:00+00:00"),
                "init_time_local": pd.Timestamp("2026-04-12T00:00:00-04:00"),
                "available_time_local": pd.Timestamp("2026-04-12T00:12:00-04:00"),
                "valid_date_local": "2026-04-12",
            },
        ]
    )

    selected, _ = nbm_overnight.select_issue_for_target(
        issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
    )

    assert str(selected["wide_path"]) == "/tmp/a"


def test_build_for_date_writes_daily_summary(tmp_path):
    features_root = tmp_path / "nbm"
    output_dir = tmp_path / "overnight"
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T03:45:00+00:00",
        init_time_local="2026-04-11T23:45:00-04:00",
        target_date_local="2026-04-12",
        wide_rows=make_wide_rows(init_time_utc="2026-04-12T03:45:00+00:00", init_time_local="2026-04-11T23:45:00-04:00", target_date_local="2026-04-12", tmp_offset=-1.0),
    )
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        target_date_local="2026-04-12",
        wide_rows=make_wide_rows(init_time_utc="2026-04-12T04:00:00+00:00", init_time_local="2026-04-12T00:00:00-04:00", target_date_local="2026-04-12", tmp_offset=0.0),
    )

    issue_catalog = nbm_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = nbm_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)
    row = overnight_df.iloc[0]
    manifest = manifest_df.iloc[0]

    assert len(overnight_df) == 1
    assert row["selected_init_time_local"] == "2026-04-12T00:00:00-04:00"
    assert bool(row["coverage_complete"]) is True
    assert int(row["missing_checkpoint_count"]) == 0
    assert int(row["missing_required_feature_count"]) == 0
    assert float(row["nbm_temp_2m_day_max_k"]) == 292.0
    assert float(row["nbm_native_tmax_2m_day_max_k"]) == 295.0
    assert float(row["nbm_native_tmax_2m_nb7_day_max_k"]) == 295.8
    assert float(row["nbm_native_tmin_2m_day_min_k"]) == 274.0
    assert float(row["nbm_native_tmin_2m_crop_day_min_k"]) == 272.8
    assert float(row["nbm_temp_2m_09_local_k"]) == 284.0
    assert float(row["nbm_tcdc_day_mean_pct"]) == 43.333333333333336
    assert float(row["nbm_dswrf_crop_day_max_w_m2"]) == 520.0
    assert float(row["nbm_pcpdur_day_total_h"]) == 3.0
    assert float(row["nbm_pcpdur_morning_total_h"]) == 3.0
    assert float(row["nbm_pwther_code_day_mode"]) == 0.0
    assert int(row["nbm_pwther_nonzero_hour_count"]) == 3
    assert float(row["nbm_pwther_any_flag"]) == 1.0
    assert float(row["nbm_tstm_day_max_pct"]) == 35.0
    assert float(row["nbm_tstm_any_flag"]) == 1.0
    assert float(row["nbm_ptype_code_day_mode"]) == 0.0
    assert int(row["nbm_ptype_nonzero_hour_count"]) == 3
    assert float(row["nbm_thunc_day_max_code"]) == 2.0
    assert int(row["nbm_thunc_nonzero_hour_count"]) == 3
    assert float(row["nbm_vrate_day_max"]) == 250.0
    assert manifest["status"] == "ok"
    assert bool(manifest["coverage_complete"]) is True
    assert int(manifest["missing_required_feature_count"]) == 0


def test_build_for_date_excludes_post_cutoff_processed_issue(tmp_path):
    features_root = tmp_path / "nbm"
    output_dir = tmp_path / "overnight"
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T03:45:00+00:00",
        init_time_local="2026-04-11T23:45:00-04:00",
        processed_timestamp_utc="2026-04-12T03:55:00+00:00",
        target_date_local="2026-04-12",
        wide_rows=make_wide_rows(
            init_time_utc="2026-04-12T03:45:00+00:00",
            init_time_local="2026-04-11T23:45:00-04:00",
            target_date_local="2026-04-12",
            tmp_offset=-2.0,
        ),
    )
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        processed_timestamp_utc="2026-04-12T04:12:00+00:00",
        target_date_local="2026-04-12",
        wide_rows=make_wide_rows(
            init_time_utc="2026-04-12T04:00:00+00:00",
            init_time_local="2026-04-12T00:00:00-04:00",
            target_date_local="2026-04-12",
            tmp_offset=0.0,
        ),
    )

    issue_catalog = nbm_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = nbm_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)
    row = overnight_df.iloc[0]
    manifest = manifest_df.iloc[0]

    assert row["selected_init_time_local"] == "2026-04-11T23:45:00-04:00"
    assert float(row["nbm_temp_2m_day_max_k"]) == 290.0
    assert manifest["status"] == "ok"


def test_build_for_date_marks_missing_native_tmax_tmin_incomplete(tmp_path):
    features_root = tmp_path / "nbm"
    output_dir = tmp_path / "overnight"
    rows = make_wide_rows(
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        target_date_local="2026-04-12",
    )
    for row in rows:
        for column in (
            "tmax",
            "tmax_nb3_max",
            "tmax_nb7_max",
            "tmax_crop_max",
            "tmin",
            "tmin_nb3_min",
            "tmin_nb7_min",
            "tmin_crop_min",
        ):
            row.pop(column, None)
    write_issue_artifacts(
        features_root,
        init_time_utc="2026-04-12T04:00:00+00:00",
        init_time_local="2026-04-12T00:00:00-04:00",
        target_date_local="2026-04-12",
        wide_rows=rows,
    )

    issue_catalog = nbm_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = nbm_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2026-04-12").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
        output_dir=output_dir,
    )

    row = pd.read_parquet(overnight_path).iloc[0]
    manifest = pd.read_parquet(manifest_path).iloc[0]
    assert bool(row["coverage_complete"]) is False
    assert int(row["missing_checkpoint_count"]) == 0
    assert int(row["missing_required_feature_count"]) == 2
    assert pd.isna(row["nbm_native_tmax_2m_day_max_k"])
    assert pd.isna(row["nbm_native_tmin_2m_day_min_k"])
    assert bool(manifest["coverage_complete"]) is False
    assert int(manifest["missing_required_feature_count"]) == 2


def test_build_for_date_historical_replay_keeps_issue_when_manifest_uses_init_timestamp(tmp_path):
    features_root = tmp_path / "nbm"
    output_dir = tmp_path / "overnight"
    write_issue_artifacts(
        features_root,
        init_time_utc="2025-04-11T04:00:00+00:00",
        init_time_local="2025-04-11T00:00:00-04:00",
        target_date_local="2025-04-11",
        wide_rows=make_wide_rows(
            init_time_utc="2025-04-11T04:00:00+00:00",
            init_time_local="2025-04-11T00:00:00-04:00",
            target_date_local="2025-04-11",
            tmp_offset=0.0,
        ),
    )

    issue_catalog = nbm_overnight.discover_issue_catalog(features_root)
    overnight_path, manifest_path = nbm_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2025-04-11").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)

    assert len(overnight_df) == 1
    assert overnight_df.iloc[0]["selected_init_time_local"] == "2025-04-11T00:00:00-04:00"
    assert manifest_df.iloc[0]["status"] == "ok"


def test_discover_issue_catalog_prefers_wide_path_valid_date_partition(tmp_path):
    features_root = tmp_path / "nbm"
    output_dir = tmp_path / "overnight"
    init_time_utc = "2025-04-11T04:00:00+00:00"
    init_time_local = "2025-04-11T00:00:00-04:00"
    target_date_local = "2025-04-11"
    issue_root = features_root / "metadata" / "manifest" / "init_date_local=2025-04-11"
    issue_root.mkdir(parents=True, exist_ok=True)
    wide_path = (
        features_root
        / "features"
        / "wide"
        / "valid_date_local=2025-04-11"
        / "init_date_local=2025-04-11"
        / "mode=premarket"
        / "cycle_20250411T0400Z_wide.parquet"
    )
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        make_wide_rows(
            init_time_utc=init_time_utc,
            init_time_local=init_time_local,
            target_date_local=target_date_local,
            tmp_offset=0.0,
        )
    ).to_parquet(wide_path, index=False)
    manifest_path = issue_root / "cycle_20250411T0400Z_manifest.parquet"
    pd.DataFrame.from_records(
        [
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": init_time_utc,
                "init_time_local": init_time_local,
                "processed_timestamp_utc": init_time_utc,
                # Deliberately wrong; the builder should trust the wide-path partition.
                "valid_date_local": "2025-04-12",
                "extraction_status": "ok",
                "wide_output_paths": str(wide_path),
            }
        ]
    ).to_parquet(manifest_path, index=False)

    issue_catalog = nbm_overnight.discover_issue_catalog(features_root)
    assert issue_catalog.iloc[0]["valid_date_local"] == "2025-04-11"

    overnight_path, manifest_path = nbm_overnight.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=pd.Timestamp("2025-04-11").date(),
        cutoff_local_time=pd.Timestamp("00:05").time(),
        output_dir=output_dir,
    )

    overnight_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)
    assert len(overnight_df) == 1
    assert int(overnight_df.iloc[0]["target_day_row_count"]) == 6
    assert bool(overnight_df.iloc[0]["coverage_complete"]) is True
    assert manifest_df.iloc[0]["status"] == "ok"


def test_categorical_regime_summaries_are_stable_and_nullable():
    day_df = pd.DataFrame.from_records(
        [
            {"valid_time_local": "2026-04-12T06:00:00-04:00", "pwther": 2, "ptype": 5, "tstm": 0},
            {"valid_time_local": "2026-04-12T07:00:00-04:00", "pwther": 3, "ptype": 5, "tstm": 10},
            {"valid_time_local": "2026-04-12T08:00:00-04:00", "pwther": 2, "ptype": 4, "tstm": 20},
            {"valid_time_local": "2026-04-12T09:00:00-04:00", "pwther": 3, "ptype": 4, "tstm": 0},
        ]
    )

    assert nbm_overnight.stable_numeric_mode(day_df, "pwther") == 2.0
    assert nbm_overnight.stable_numeric_mode(day_df, "ptype") == 4.0
    assert nbm_overnight.nonzero_count(day_df, "tstm") == 2
    assert nbm_overnight.any_nonzero_flag(day_df, "tstm") == 1.0
    assert nbm_overnight.stable_numeric_mode(day_df, "missing") is None
    assert nbm_overnight.nonzero_count(day_df, "missing") is None
    assert nbm_overnight.any_nonzero_flag(day_df, "missing") is None


def test_target_day_row_count_counts_unique_forecast_hours():
    day_df = pd.DataFrame.from_records(
        [
            {"forecast_hour": 1, "valid_time_local": "2026-04-12T01:00:00-04:00"},
            {"forecast_hour": 1, "valid_time_local": "2026-04-12T01:00:00-04:00"},
            {"forecast_hour": 2, "valid_time_local": "2026-04-12T02:00:00-04:00"},
        ]
    )

    assert nbm_overnight.target_day_row_count(day_df) == 2


def test_checkpoint_features_uses_first_non_null_value_per_hour():
    day_df = pd.DataFrame.from_records(
        [
            {
                "valid_time_local": "2025-04-11T06:00:00-04:00",
                "tmp": None,
                "dpt": None,
                "rh": None,
                "wind": None,
                "wdir": None,
            },
            {
                "valid_time_local": "2025-04-11T06:00:00-04:00",
                "tmp": 280.5,
                "dpt": 276.0,
                "rh": 60.0,
                "wind": 6.0,
                "wdir": 210.0,
            },
            {
                "valid_time_local": "2025-04-11T09:00:00-04:00",
                "tmp": 284.0,
                "dpt": 278.0,
                "rh": 55.0,
                "wind": 8.0,
                "wdir": 220.0,
            },
            {
                "valid_time_local": "2025-04-11T12:00:00-04:00",
                "tmp": 289.0,
                "dpt": 279.0,
                "rh": 50.0,
                "wind": 9.0,
                "wdir": 230.0,
            },
            {
                "valid_time_local": "2025-04-11T15:00:00-04:00",
                "tmp": 292.0,
                "dpt": 280.0,
                "rh": 48.0,
                "wind": 10.0,
                "wdir": 240.0,
            },
            {
                "valid_time_local": "2025-04-11T18:00:00-04:00",
                "tmp": 290.0,
                "dpt": 279.0,
                "rh": 52.0,
                "wind": 9.0,
                "wdir": 245.0,
            },
            {
                "valid_time_local": "2025-04-11T21:00:00-04:00",
                "tmp": 286.0,
                "dpt": 278.0,
                "rh": 58.0,
                "wind": 7.0,
                "wdir": 250.0,
            },
        ]
    )

    outputs, missing = nbm_overnight.checkpoint_features(day_df)

    assert missing == 0
    assert outputs["nbm_temp_2m_06_local_k"] == 280.5
    assert outputs["nbm_dewpoint_2m_09_local_k"] == 278.0
    assert outputs["nbm_rh_2m_15_local_pct"] == 48.0
    assert outputs["nbm_wind_10m_direction_15_local_deg"] == 240.0


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
        overnight_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        overnight_path.write_text("")
        manifest_path.write_text("")
        return overnight_path, manifest_path

    monkeypatch.setattr(nbm_overnight, "ProgressBar", RecordingProgressBar)
    monkeypatch.setattr(
        nbm_overnight,
        "parse_args",
        lambda: SimpleNamespace(
            features_root=tmp_path / "features",
            output_dir=tmp_path / "output",
            start_local_date="2026-04-12",
            end_local_date="2026-04-14",
            cutoff_local_time="00:05",
        ),
    )
    monkeypatch.setattr(nbm_overnight, "discover_issue_catalog", lambda _root: pd.DataFrame())
    monkeypatch.setattr(nbm_overnight, "build_for_date", fake_build_for_date)

    assert nbm_overnight.main() == 0
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.total == 3
    assert progress.label == "NBM overnight"
    assert progress.unit == "date"
    assert progress.advances == 3
    assert progress.closed is True
    assert any(stage == "complete" for stage, _ in progress.updates)
    assert any(stage == "finalize" for stage, _ in progress.updates)
