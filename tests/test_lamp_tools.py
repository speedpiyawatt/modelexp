from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys
import tarfile

import pandas as pd


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAMP_DIR = ROOT / "tools" / "lamp"
FIXTURES = LAMP_DIR / "data" / "fixtures" / "parser_samples"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(LAMP_DIR) not in sys.path:
    sys.path.insert(0, str(LAMP_DIR))

PARSE_PATH = LAMP_DIR / "parse_lamp_ascii.py"
FETCH_PATH = LAMP_DIR / "fetch_lamp.py"
BUILD_PATH = LAMP_DIR / "build_lamp_klga_features.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


lamp_parse = load_module("lamp_parse_test", PARSE_PATH)
lamp_fetch = load_module("lamp_fetch_test", FETCH_PATH)
lamp_build = load_module("lamp_build_test", BUILD_PATH)


def test_live_file_names_accepts_real_nomads_issue_cadence():
    assert lamp_fetch.live_file_names("0400")[0][1] == "lmp.t0400z.lavtxt.ascii"
    assert lamp_fetch.live_file_names("0415")[0][1] == "lmp.t0415z.lavtxt.ascii"
    assert lamp_fetch.live_file_names("0430")[1][1] == "lmp.t0430z.lavtxt_ext.ascii"
    assert lamp_fetch.live_file_names("0445")[0][1] == "lmp.t0445z.lavtxt.ascii"


def test_parse_current_standard_fixture_emits_expected_klga_rows():
    df = lamp_parse.parse_bulletin_file(FIXTURES / "current_standard_2026_0430.ascii")

    klga = df.loc[df["station_id"] == "KLGA"].reset_index(drop=True)
    assert not klga.empty
    assert set(klga["bulletin_type"]) == {"standard"}
    assert klga["init_time_utc"].nunique() == 1
    assert klga["init_time_utc"].iloc[0] == "2026-04-12T04:30:00+00:00"
    assert klga["forecast_hour"].min() == 1
    assert klga["forecast_hour"].max() == 25

    tmp_f1 = klga.loc[(klga["raw_label"] == "TMP") & (klga["forecast_hour"] == 1), "value"].iloc[0]
    tmp_f25 = klga.loc[(klga["raw_label"] == "TMP") & (klga["forecast_hour"] == 25), "value"].iloc[0]
    wgs_f1 = klga.loc[(klga["raw_label"] == "WGS") & (klga["forecast_hour"] == 1), "value"].iloc[0]
    p06_hours = klga.loc[klga["raw_label"] == "P06", "forecast_hour"].tolist()
    p06_values = klga.loc[klga["raw_label"] == "P06", "value"].tolist()

    assert tmp_f1 == 50
    assert tmp_f25 == 52
    assert wgs_f1 == "NG"
    assert p06_hours == [8, 14, 20]
    assert p06_values == [0, 4, 19]


def test_parse_current_extended_fixture_preserves_hours_26_to_38():
    df = lamp_parse.parse_bulletin_file(FIXTURES / "current_extended_2026_0430.ascii")

    assert set(df["bulletin_type"]) == {"extended"}
    assert df["forecast_hour"].min() == 26
    assert df["forecast_hour"].max() == 38
    tmp_38 = df.loc[(df["raw_label"] == "TMP") & (df["forecast_hour"] == 38), "value"].iloc[0]
    valid_26 = df.loc[(df["raw_label"] == "TMP") & (df["forecast_hour"] == 26), "valid_time_utc"].iloc[0]

    assert tmp_38 == 77
    assert valid_26 == "2026-04-13T06:00:00+00:00"


def test_parse_archived_fixture_uses_explicit_hr_rows():
    df = lamp_parse.parse_bulletin_file(FIXTURES / "archived_standard_2024_2130.ascii")

    assert df["forecast_hour"].min() == 1
    assert df["forecast_hour"].max() == 10
    assert df.loc[(df["raw_label"] == "TMP") & (df["forecast_hour"] == 1), "valid_time_utc"].iloc[0] == "2024-01-11T22:00:00+00:00"
    assert df.loc[(df["raw_label"] == "TMP") & (df["forecast_hour"] == 3), "valid_time_utc"].iloc[0] == "2024-01-12T00:00:00+00:00"


def test_merge_station_frames_combines_standard_and_extended_hours():
    standard = lamp_parse.parse_bulletin_file(FIXTURES / "current_standard_2026_0430.ascii")
    extended = lamp_parse.parse_bulletin_file(FIXTURES / "current_extended_2026_0430.ascii")

    merged = lamp_parse.merge_station_frames([standard, extended])
    klga = merged.loc[(merged["station_id"] == "KLGA") & (merged["raw_label"] == "TMP")]

    assert klga["forecast_hour"].min() == 1
    assert klga["forecast_hour"].max() == 38
    assert len(klga["forecast_hour"].unique()) == 38


def test_archive_extraction_filters_members_by_issue_date_and_cycle(tmp_path):
    archive_path = tmp_path / "lmp_lavtxt.2026.tar"
    sample_a = FIXTURES / "current_standard_2026_0430.ascii"
    sample_b = FIXTURES / "archived_standard_2024_2130.ascii"

    with tarfile.open(archive_path, "w") as archive:
        archive.add(sample_a, arcname="weird/current_standard.ascii")
        archive.add(sample_b, arcname="weird/archived_standard.ascii")

    args = argparse.Namespace(
        start_utc_date="2026-04-12",
        end_utc_date="2026-04-12",
        cycle=["0430"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        keep_archive_tars=True,
        overwrite=False,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_tar = lamp_fetch.cache_path_for_year(2026, args.cache_dir)
    target_tar.write_bytes(archive_path.read_bytes())

    outputs = lamp_fetch.extract_archive_members(args)

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert "date_utc=2026-04-12" in str(outputs[0])
    assert "cycle=0430" in str(outputs[0])


def test_archive_extraction_redownloads_when_cached_tar_is_corrupt(tmp_path, monkeypatch):
    good_archive_path = tmp_path / "good_lmp_lavtxt.2026.tar"
    sample = FIXTURES / "current_standard_2026_0430.ascii"
    with tarfile.open(good_archive_path, "w") as archive:
        archive.add(sample, arcname="weird/current_standard.ascii")

    args = argparse.Namespace(
        start_utc_date="2026-04-12",
        end_utc_date="2026-04-12",
        cycle=["0430"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        keep_archive_tars=True,
        overwrite=False,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_tar = lamp_fetch.cache_path_for_year(2026, args.cache_dir)
    target_tar.write_bytes(b"not a real tar")

    calls: list[tuple[str, pathlib.Path, bool]] = []

    def fake_download(url: str, destination: pathlib.Path, *, overwrite: bool):
        calls.append((url, destination, overwrite))
        destination.write_bytes(good_archive_path.read_bytes())
        return destination

    monkeypatch.setattr(lamp_fetch, "download_with_progress", fake_download)

    outputs = lamp_fetch.extract_archive_members(args)

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert calls
    assert calls[-1] == (lamp_fetch.archive_url(2026), target_tar, True)


def test_archive_extraction_redownloads_when_tar_breaks_during_iteration(tmp_path, monkeypatch):
    good_archive_path = tmp_path / "good_lmp_lavtxt.2026.tar"
    sample = FIXTURES / "current_standard_2026_0430.ascii"
    with tarfile.open(good_archive_path, "w") as archive:
        archive.add(sample, arcname="weird/current_standard.ascii")

    args = argparse.Namespace(
        start_utc_date="2026-04-12",
        end_utc_date="2026-04-12",
        cycle=["0430"],
        output_dir=tmp_path / "raw",
        cache_dir=tmp_path / "cache",
        keep_archive_tars=True,
        overwrite=False,
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    target_tar = lamp_fetch.cache_path_for_year(2026, args.cache_dir)
    target_tar.write_bytes(good_archive_path.read_bytes())

    calls: list[tuple[str, pathlib.Path, bool]] = []
    real_extract = lamp_fetch._extract_members_from_tar
    state = {"first": True}

    def fake_download(url: str, destination: pathlib.Path, *, overwrite: bool):
        calls.append((url, destination, overwrite))
        destination.write_bytes(good_archive_path.read_bytes())
        return destination

    def flaky_extract(*args, **kwargs):
        if state["first"]:
            state["first"] = False
            raise tarfile.ReadError("unexpected end of data")
        return real_extract(*args, **kwargs)

    monkeypatch.setattr(lamp_fetch, "download_with_progress", fake_download)
    monkeypatch.setattr(lamp_fetch, "_extract_members_from_tar", flaky_extract)

    outputs = lamp_fetch.extract_archive_members(args)

    assert len(outputs) == 1
    assert outputs[0].exists()
    assert calls
    assert calls[-1] == (lamp_fetch.archive_url(2026), target_tar, True)


def test_build_klga_outputs_writes_wide_long_provenance_and_manifest(tmp_path):
    output_dir = tmp_path / "features"
    inputs = [
        FIXTURES / "current_standard_2026_0430.ascii",
        FIXTURES / "current_extended_2026_0430.ascii",
    ]

    written = lamp_build.build_from_inputs(inputs, station_id="KLGA", output_dir=output_dir, write_long=True)

    wide_paths = [path for path in written if path.name == "lamp.wide.parquet"]
    long_paths = [path for path in written if path.name == "lamp.long.parquet"]
    provenance_paths = [path for path in written if path.name == "lamp.provenance.parquet"]
    manifest_paths = [path for path in written if path.name == "lamp.manifest.parquet"]

    assert len(wide_paths) == 1
    assert len(long_paths) == 1
    assert len(provenance_paths) == 1
    assert len(manifest_paths) == 1

    wide_df = pd.read_parquet(wide_paths[0])
    long_df = pd.read_parquet(long_paths[0])
    provenance_df = pd.read_parquet(provenance_paths[0])
    manifest_df = pd.read_parquet(manifest_paths[0])

    assert len(wide_df) == 38
    assert wide_df["forecast_hour"].tolist() == list(range(1, 39))
    assert set(["source_model", "source_product", "source_version", "station_id", "init_time_utc", "valid_time_utc", "forecast_hour"]).issubset(wide_df.columns)
    assert "tmp" in wide_df.columns
    assert "dpt" in wide_df.columns
    assert "cld" in wide_df.columns
    assert "raw_label" in long_df.columns
    assert set(long_df["station_id"]) == {"KLGA"}
    assert set(provenance_df["feature_name"]).issuperset({"tmp", "dpt", "cld", "p01"})
    missing_optional = provenance_df.loc[provenance_df["missing_optional"] == True]
    assert not missing_optional.empty
    assert set(missing_optional["raw_feature_name"]).issuperset({"P06", "P12"})
    assert manifest_df["status"].iloc[0] == "ok"
    assert manifest_df["wide_row_count"].iloc[0] == 38
    assert manifest_df["wide_output_path"].iloc[0] == str(wide_paths[0])
    assert "Missing optional LAMP labels" in manifest_df["warnings"].iloc[0]


def test_build_from_inputs_creates_default_progress_for_parse_and_issue_writes(tmp_path, monkeypatch):
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

    monkeypatch.setattr(lamp_build, "ProgressBar", RecordingProgressBar)

    output_dir = tmp_path / "features"
    inputs = [
        FIXTURES / "current_standard_2026_0430.ascii",
        FIXTURES / "current_extended_2026_0430.ascii",
    ]
    written = lamp_build.build_from_inputs(inputs, station_id="KLGA", output_dir=output_dir, write_long=False)

    assert written
    assert [progress.label for progress in progress_instances] == ["LAMP parse", "LAMP features"]
    parse_progress, issue_progress = progress_instances
    assert parse_progress.total == 2
    assert parse_progress.closed is True
    assert any(stage == "parse" for stage, _ in parse_progress.updates)
    assert issue_progress.total == 1
    assert issue_progress.advances == 1
    assert issue_progress.closed is True
    assert any(stage == "build" for stage, _ in issue_progress.updates)
    assert any(stage == "write" for stage, _ in issue_progress.updates)
