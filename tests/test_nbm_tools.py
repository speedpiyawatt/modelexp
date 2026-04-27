from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import importlib.util
import io
import json
import pathlib
import sys
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import rasterio
import requests
import tools.weather.progress as progress_mod
import xarray as xr
from rasterio.transform import from_origin


ROOT = pathlib.Path(__file__).resolve().parents[1]
FETCH_PATH = ROOT / "tools" / "nbm" / "fetch_nbm.py"
SAMPLE_PATH = ROOT / "tools" / "nbm" / "sample_cog_point.py"
GRIB2_PIPELINE_PATH = ROOT / "tools" / "nbm" / "build_grib2_features.py"
BENCHMARK_PATH = ROOT / "tools" / "nbm" / "benchmark_nbm_sources.py"
MONTHLY_BACKFILL_PATH = ROOT / "tools" / "nbm" / "run_nbm_monthly_backfill.py"
NBM_OVERNIGHT_PATH = ROOT / "tools" / "nbm" / "build_nbm_overnight_features.py"
NBM_DIAGNOSTICS_SUMMARY_PATH = ROOT / "tools" / "nbm" / "summarize_nbm_diagnostics.py"
NBM_DEBUG_BOTTLENECKS_PATH = ROOT / "tools" / "nbm" / "debug_bottlenecks.py"
LOCATION_CONTEXT_PATH = ROOT / "tools" / "weather" / "location_context.py"


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fetch_nbm = load_module("fetch_nbm", FETCH_PATH)
location_context = load_module("location_context", LOCATION_CONTEXT_PATH)
sample_cog_point = load_module("sample_cog_point", SAMPLE_PATH)
build_grib2_features = load_module("build_grib2_features", GRIB2_PIPELINE_PATH)
benchmark_nbm_sources = load_module("benchmark_nbm_sources", BENCHMARK_PATH)
nbm_overnight_features = load_module("nbm_overnight_features", NBM_OVERNIGHT_PATH)
nbm_monthly_backfill = load_module("nbm_monthly_backfill", MONTHLY_BACKFILL_PATH)
nbm_diagnostics_summary = load_module("nbm_diagnostics_summary", NBM_DIAGNOSTICS_SUMMARY_PATH)
nbm_debug_bottlenecks = load_module("nbm_debug_bottlenecks", NBM_DEBUG_BOTTLENECKS_PATH)


class FakeClient:
    def __init__(self, cycles_by_prefix=None, keys_by_prefix=None, on_download=None):
        self.cycles_by_prefix = cycles_by_prefix or {}
        self.keys_by_prefix = keys_by_prefix or {}
        self.downloads: list[str] = []
        self.on_download = on_download

    def list_prefixes(self, base_url: str, prefix: str):
        return self.cycles_by_prefix.get(prefix, [])

    def list_keys(self, base_url: str, prefix: str, max_keys: int = 1000):
        result = self.keys_by_prefix.get(prefix, [])
        if callable(result):
            return result(prefix)
        return result

    def download_key(
        self,
        *,
        base_url: str,
        key: str,
        output_dir: pathlib.Path,
        overwrite: bool,
        preserve_tree: bool,
        progress_label: str | None = None,
    ):
        self.downloads.append(key)
        destination = output_dir / key if preserve_tree else output_dir / pathlib.Path(key).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("ok")
        if self.on_download:
            self.on_download(key)
        return destination


class FakeGrib2Client:
    def __init__(self, cycles_by_prefix=None, payloads=None):
        self.cycles_by_prefix = cycles_by_prefix or {}
        self.payloads = payloads or {}
        self.range_downloads: list[tuple[str, list[tuple[int, int]], str | None]] = []
        self.content_length_requests: list[str] = []

    @staticmethod
    def _key_from_url(url: str) -> str:
        return url.split(".amazonaws.com/", 1)[1]

    def list_prefixes(self, base_url: str, prefix: str):
        return self.cycles_by_prefix.get(prefix, [])

    def fetch_text(self, url: str):
        payload = self.payloads[self._key_from_url(url)]
        if isinstance(payload, bytes):
            return payload.decode("utf-8")
        return str(payload)

    def fetch_content_length(self, url: str):
        self.content_length_requests.append(self._key_from_url(url))
        payload = self.payloads[self._key_from_url(url)]
        if isinstance(payload, bytes):
            return len(payload)
        return len(str(payload).encode("utf-8"))

    def download_byte_ranges(self, *, url: str, ranges, destination: pathlib.Path, overwrite: bool, progress_label: str | None = None):
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads[self._key_from_url(url)]
        if not isinstance(payload, bytes):
            payload = str(payload).encode("utf-8")
        with destination.open("wb") as handle:
            for start, end in ranges:
                handle.write(payload[start : end + 1])
        self.range_downloads.append((self._key_from_url(url), list(ranges), progress_label))
        return destination

    def download_key(
        self,
        *,
        base_url: str,
        key: str,
        output_dir: pathlib.Path,
        overwrite: bool,
        preserve_tree: bool,
        progress_label: str | None = None,
    ):
        destination = output_dir / key if preserve_tree else output_dir / pathlib.Path(key).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.payloads.get(key, "")
        if isinstance(payload, bytes):
            destination.write_bytes(payload)
        else:
            destination.write_text(str(payload))
        return destination


def make_raw_grib_payload(size: int = 1400) -> bytes:
    chunks = bytearray()
    for index in range(size):
        chunks.append(index % 251)
    return bytes(chunks)


def make_required_idx_text() -> str:
    return "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"),
            make_inventory_line(3, "RH", "2 m above ground", "1 hour fcst"),
            make_inventory_line(4, "WIND", "10 m above ground", "1 hour fcst"),
            make_inventory_line(5, "WDIR", "10 m above ground", "1 hour fcst"),
            make_inventory_line(6, "GUST", "10 m above ground", "1 hour fcst"),
            make_inventory_line(7, "TCDC", "surface", "1 hour fcst"),
            make_inventory_line(8, "DSWRF", "surface", "1 hour fcst"),
            make_inventory_line(9, "APCP", "surface", "0-1 hour acc fcst"),
            make_inventory_line(10, "VIS", "surface", "1 hour fcst"),
            make_inventory_line(11, "CEIL", "cloud ceiling", "1 hour fcst"),
            make_inventory_line(12, "CAPE", "surface", "1 hour fcst"),
            make_inventory_line(13, "VRATE", "entire atmosphere (considered as a single layer)", "1 hour fcst"),
        ]
    )


def make_batch_args(tmp_path: pathlib.Path, **overrides):
    base = dict(
        start_date="2024-03-01",
        end_date="2024-03-02",
        cycle="12",
        domain="conus",
        version=None,
        preset="klga_surface",
        variables=["temp"],
        dry_run=True,
        max_files=None,
        workers=2,
        plan_workers=2,
        output_dir=tmp_path / "out",
        overwrite=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def make_test_tif(path: pathlib.Path, values, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        transform=from_origin(-75.0, 42.0, 1.0, 1.0),
        crs="EPSG:4326",
        nodata=nodata,
    ) as dataset:
        dataset.write(values, 1)


class FakeBenchmarkResponse:
    def __init__(self, payload: bytes, content_length: int | None = None):
        self.payload = payload
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size: int = 1024):
        for start in range(0, len(self.payload), chunk_size):
            yield self.payload[start : start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeBenchmarkSession:
    def __init__(self, payload_by_url, bad_length_urls=None):
        self.payload_by_url = payload_by_url
        self.bad_length_urls = set(bad_length_urls or [])

    def get(self, url: str, stream: bool = True, timeout=None):
        payload = self.payload_by_url[url]
        content_length = len(payload) + 1 if url in self.bad_length_urls else len(payload)
        return FakeBenchmarkResponse(payload, content_length=content_length)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeFTP:
    payload_by_path: dict[str, bytes] = {}
    requested_paths: list[str] = []

    def __init__(self, host: str):
        self.host = host

    def login(self):
        return None

    def voidcmd(self, command: str):
        return "200"

    def retrbinary(self, command: str, callback, blocksize: int = 8192):
        path = command.split(" ", 1)[1]
        FakeFTP.requested_paths.append(path)
        payload = self.payload_by_path[path]
        for start in range(0, len(payload), blocksize):
            callback(payload[start : start + blocksize])

    def quit(self):
        return None

    def close(self):
        return None


def test_resolve_cog_version_and_cycle_prefers_newest_available():
    client = FakeClient(
        cycles_by_prefix={
            "blendv4.3/conus/2024/08/01/": [],
            "blendv4.2/conus/2024/08/01/": ["blendv4.2/conus/2024/08/01/1200/"],
            "blendv4.1/conus/2024/08/01/": ["blendv4.1/conus/2024/08/01/1200/"],
        }
    )
    version, cycle = fetch_nbm.resolve_cog_version_and_cycle(
        day=fetch_nbm.normalize_date("2024-08-01"),
        domain="conus",
        version=None,
        cycle="12",
        client=client,
    )
    assert version == "blendv4.2"
    assert cycle == "12"


def test_benchmark_resolve_source_url_for_all_sources():
    key = "blend.20260411/11/core/blend.t11z.core.f001.co.grib2"
    assert benchmark_nbm_sources.resolve_source_url("aws", key) == f"{benchmark_nbm_sources.AWS_BASE}/{key}"
    assert benchmark_nbm_sources.resolve_source_url("noaa_https", key) == f"{benchmark_nbm_sources.NOAA_HTTPS_BASE}/{key}"
    assert benchmark_nbm_sources.resolve_source_url("noaa_ftp", key) == f"{benchmark_nbm_sources.NOAA_FTP_BASE}/{key}"


def test_benchmark_noaa_https_falls_back_to_inventory_page(monkeypatch):
    key = "blend.20260411/09/core/blend.t09z.core.f001.co.grib2"
    monkeypatch.setattr(benchmark_nbm_sources, "probe_http_url", lambda url, session_factory=requests.Session: False)
    url, content_kind = benchmark_nbm_sources.resolve_noaa_https_target(
        key=key,
        cycle="09",
        region="co",
    )
    assert content_kind == "inventory_html"
    assert url == "https://www.nco.ncep.noaa.gov/pmb/products/blend/conus/09/blend.t09z.core.f001.co.grib2.shtml"


def test_benchmark_parse_args_normalizes_sample_selector():
    args = benchmark_nbm_sources.parse_args(
        [
            "--date",
            "20260101",
            "--cycle",
            "6",
            "--forecast-hour",
            "1",
            "--region",
            "co",
            "--sources",
            "aws,noaa_https,noaa_ftp",
        ]
    )
    day = fetch_nbm.normalize_date(args.date)
    cycle = fetch_nbm.normalize_cycle(args.cycle)
    key = fetch_nbm.resolve_grib2_key(day, cycle, args.forecast_hour, args.region)
    assert cycle == "06"
    assert key == "blend.20260101/06/core/blend.t06z.core.f001.co.grib2"
    assert args.sources == ["aws", "noaa_https", "noaa_ftp"]


def test_benchmark_build_round_robin_schedule_is_deterministic():
    schedule = benchmark_nbm_sources.build_round_robin_schedule(
        sources=["aws", "noaa_https", "noaa_ftp"],
        warmup_runs=1,
        measured_runs=3,
    )
    assert schedule == [
        ("warmup", 1, "aws"),
        ("warmup", 1, "noaa_https"),
        ("warmup", 1, "noaa_ftp"),
        ("measured", 1, "aws"),
        ("measured", 1, "noaa_https"),
        ("measured", 1, "noaa_ftp"),
        ("measured", 2, "noaa_https"),
        ("measured", 2, "noaa_ftp"),
        ("measured", 2, "aws"),
        ("measured", 3, "noaa_ftp"),
        ("measured", 3, "aws"),
        ("measured", 3, "noaa_https"),
    ]


def test_benchmark_http_download_detects_content_length_mismatch(tmp_path):
    url = "https://example.test/file.grib2"
    payload = b"abc123"
    with pytest.raises(RuntimeError, match="Content-Length mismatch"):
        benchmark_nbm_sources.stream_http_download(
            url,
            tmp_path / "file.grib2",
            session_factory=lambda: FakeBenchmarkSession({url: payload}, bad_length_urls={url}),
        )


def test_benchmark_runner_and_summary_handle_success_failure_and_ranking(tmp_path):
    day = dt.date(2026, 1, 1)
    cycle = "06"
    key = fetch_nbm.resolve_grib2_key(day, cycle, 1, "co")
    payload = b"same-payload"
    sha256 = benchmark_nbm_sources.hashlib.sha256(payload).hexdigest()
    clock = iter([0.0, 1.0, 1.0, 3.0, 3.0, 6.0, 6.0, 8.0, 8.0, 11.0, 11.0, 11.5])

    def fake_perf_counter():
        return next(clock)

    def fake_download_once(*, source, url, destination, session_factory, ftp_factory, progress_callback=None, ftp_timeout_seconds=None):
        if progress_callback is not None:
            progress_callback(len(payload), len(payload), 0.5)
        if source == "noaa_ftp":
            FakeFTP.requested_paths.append("/pub/data/nccf/com/blend/prod/" + key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        return benchmark_nbm_sources.DownloadResult(bytes_downloaded=len(payload), sha256=sha256)

    monkeypatch = pytest.MonkeyPatch()
    FakeFTP.requested_paths = []
    monkeypatch.setattr(benchmark_nbm_sources.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(benchmark_nbm_sources, "download_once", fake_download_once)
    monkeypatch.setattr(
        benchmark_nbm_sources,
        "resolve_noaa_https_target",
        lambda *, key, cycle, region, session_factory=requests.Session: ("https://example.test/file.grib2", "grib2"),
    )
    try:
        rows = benchmark_nbm_sources.benchmark_sources(
            day=day,
            cycle=cycle,
            forecast_hour=1,
            region="co",
            sources=["aws", "noaa_https", "noaa_ftp"],
            measured_runs=2,
            warmup_runs=0,
            output_dir=tmp_path,
            session_factory=lambda: FakeBenchmarkSession({}),
            ftp_factory=FakeFTP,
        )
    finally:
        monkeypatch.undo()

    assert [row["source"] for row in rows] == [
        "aws",
        "noaa_https",
        "noaa_ftp",
        "noaa_https",
        "noaa_ftp",
        "aws",
    ]
    summary = benchmark_nbm_sources.summarize_results(rows, sources=["aws", "noaa_https", "noaa_ftp"], key=key)
    assert summary["benchmark_valid"] is True
    assert summary["sha256_agreement_status"] == "match"
    assert summary["ranked_fastest_by_median_wall_time"][0]["source"] == "aws"
    assert summary["sources"]["aws"]["median_wall_time_seconds"] == 0.75
    assert summary["sources"]["noaa_https"]["median_wall_time_seconds"] == 2.0
    assert summary["sources"]["noaa_ftp"]["median_wall_time_seconds"] == 3.0
    assert FakeFTP.requested_paths == ["/pub/data/nccf/com/blend/prod/" + key, "/pub/data/nccf/com/blend/prod/" + key]


def test_benchmark_summary_excludes_warmups_and_marks_hash_mismatch():
    key = "blend.20260101/06/core/blend.t06z.core.f001.co.grib2"
    rows = [
        {
            "source": "noaa_https",
            "url": "https",
            "date": "2026-01-01",
            "cycle": "06",
            "forecast_hour": 1,
            "region": "co",
            "run_index": 1,
            "phase": "warmup",
            "content_kind": "inventory_html",
            "bytes_downloaded": 100,
            "elapsed_seconds": 10.0,
            "throughput_mbps": 10.0,
            "success": True,
            "error_message": "",
            "sha256": "warmup-only",
            "local_path": "/tmp/https-warmup",
        },
        {
            "source": "noaa_https",
            "url": "https",
            "date": "2026-01-01",
            "cycle": "06",
            "forecast_hour": 1,
            "region": "co",
            "run_index": 1,
            "phase": "measured",
            "content_kind": "inventory_html",
            "bytes_downloaded": 100,
            "elapsed_seconds": 2.0,
            "throughput_mbps": 50.0,
            "success": True,
            "error_message": "",
            "sha256": "hash-b",
            "local_path": "/tmp/https-measured",
        },
        {
            "source": "noaa_ftp",
            "url": "ftp",
            "date": "2026-01-01",
            "cycle": "06",
            "forecast_hour": 1,
            "region": "co",
            "run_index": 1,
            "phase": "measured",
            "content_kind": "grib2",
            "bytes_downloaded": 100,
            "elapsed_seconds": 3.0,
            "throughput_mbps": 33.0,
            "success": True,
            "error_message": "",
            "sha256": "hash-c",
            "local_path": "/tmp/ftp-measured",
        },
    ]
    summary = benchmark_nbm_sources.summarize_results(rows, sources=["noaa_https", "noaa_ftp"], key=key)
    assert summary["benchmark_valid"] is False
    assert summary["sha256_agreement_status"] == "not_comparable"
    assert summary["sources"]["noaa_https"]["median_wall_time_seconds"] == 2.0
    assert summary["sources"]["noaa_ftp"]["median_wall_time_seconds"] == 3.0


def test_benchmark_summary_requires_all_requested_sources_to_succeed():
    key = "blend.20260101/06/core/blend.t06z.core.f001.co.grib2"
    rows = [
        {
            "source": "noaa_https",
            "url": "https",
            "date": "2026-01-01",
            "cycle": "06",
            "forecast_hour": 1,
            "region": "co",
            "run_index": 1,
            "phase": "measured",
            "content_kind": "inventory_html",
            "bytes_downloaded": 100,
            "elapsed_seconds": 2.0,
            "throughput_mbps": 50.0,
            "success": True,
            "error_message": "",
            "sha256": "hash-b",
            "local_path": "/tmp/https-measured",
        },
        {
            "source": "noaa_ftp",
            "url": "ftp",
            "date": "2026-01-01",
            "cycle": "06",
            "forecast_hour": 1,
            "region": "co",
            "run_index": 1,
            "phase": "measured",
            "content_kind": "grib2",
            "bytes_downloaded": 0,
            "elapsed_seconds": 5.0,
            "throughput_mbps": 0.0,
            "success": False,
            "error_message": "timed out",
            "sha256": "",
            "local_path": "/tmp/ftp-measured",
        },
    ]
    summary = benchmark_nbm_sources.summarize_results(rows, sources=["noaa_https", "noaa_ftp"], key=key)
    assert summary["benchmark_valid"] is False
    assert summary["sha256_agreement_status"] == "insufficient_successful_sources"
    assert summary["sources"]["noaa_https"]["median_wall_time_seconds"] == 2.0
    assert summary["sources"]["noaa_ftp"]["median_wall_time_seconds"] is None


def test_list_keys_handles_pagination_and_cache(monkeypatch):
    client = fetch_nbm.S3HttpClient(pool_maxsize=2)
    calls = []

    first_page = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>true</IsTruncated>
  <Contents><Key>a</Key></Contents>
  <Contents><Key>b</Key></Contents>
</ListBucketResult>"""
    second_page = b"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <IsTruncated>false</IsTruncated>
  <Contents><Key>c</Key></Contents>
</ListBucketResult>"""

    def fake_get_xml(url: str) -> bytes:
        calls.append(url)
        if "marker=b" in url:
            return second_page
        return first_page

    monkeypatch.setattr(client, "_get_xml", fake_get_xml)
    keys = client.list_keys(fetch_nbm.COG_BASE, "x/")
    keys_again = client.list_keys(fetch_nbm.COG_BASE, "x/")

    assert keys == ["a", "b", "c"]
    assert keys_again == ["a", "b", "c"]
    assert len(calls) == 2


def test_run_cog_batch_dry_run_respects_max_files(tmp_path, capsys):
    k1 = "blendv4.3/conus/2024/03/01/1200/temp/blendv4.3_conus_temp_2024-03-01T12:00_2024-03-01T13:00.tif"
    k2 = "blendv4.3/conus/2024/03/01/1200/temp/blendv4.3_conus_temp_2024-03-01T12:00_2024-03-01T14:00.tif"
    k3 = "blendv4.3/conus/2024/03/01/1200/temp/blendv4.3_conus_temp_2024-03-01T12:00_2024-03-01T15:00.tif"
    k4 = "blendv4.3/conus/2024/03/02/1200/temp/blendv4.3_conus_temp_2024-03-02T12:00_2024-03-02T13:00.tif"
    client = FakeClient(
        cycles_by_prefix={
            "blendv4.3/conus/2024/03/01/": ["blendv4.3/conus/2024/03/01/1200/"],
            "blendv4.3/conus/2024/03/02/": ["blendv4.3/conus/2024/03/02/1200/"],
        },
        keys_by_prefix={
            "blendv4.3/conus/2024/03/01/1200/temp/": [k1, k2, k3],
            "blendv4.3/conus/2024/03/02/1200/temp/": [k4],
        },
    )
    args = make_batch_args(tmp_path, max_files=2, dry_run=True)
    fetch_nbm.run_cog_batch(args, client=client)
    captured = capsys.readouterr()
    assert f"plan: {k1}" in captured.out
    assert f"plan: {k2}" in captured.out
    assert f"plan: {k3}" not in captured.out
    assert "planned_files=2" in captured.out


def test_run_cog_batch_streams_downloads_before_late_day_listing_finishes(tmp_path):
    download_started = threading.Event()
    allow_day_two = threading.Event()

    def key_lookup(prefix: str):
        if "2024/03/02" in prefix:
            allow_day_two.wait(timeout=2)
            return [
                "blendv4.3/conus/2024/03/02/1200/temp/blendv4.3_conus_temp_2024-03-02T12:00_2024-03-02T13:00.tif"
            ]
        return [
            "blendv4.3/conus/2024/03/01/1200/temp/blendv4.3_conus_temp_2024-03-01T12:00_2024-03-01T13:00.tif"
        ]

    client = FakeClient(
        cycles_by_prefix={
            "blendv4.3/conus/2024/03/01/": ["blendv4.3/conus/2024/03/01/1200/"],
            "blendv4.3/conus/2024/03/02/": ["blendv4.3/conus/2024/03/02/1200/"],
        },
        keys_by_prefix={
            "blendv4.3/conus/2024/03/01/1200/temp/": [
                "blendv4.3/conus/2024/03/01/1200/temp/blendv4.3_conus_temp_2024-03-01T12:00_2024-03-01T13:00.tif"
            ],
            "blendv4.3/conus/2024/03/02/1200/temp/": key_lookup,
        },
        on_download=lambda key: download_started.set(),
    )
    args = make_batch_args(tmp_path, dry_run=False, workers=1, plan_workers=2, max_files=1)

    result = {}

    def runner():
        result["value"] = fetch_nbm.run_cog_batch(args, client)

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    assert download_started.wait(timeout=1.0)
    allow_day_two.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert result["value"] == 0


def test_sample_file_extracts_row_col_and_value(tmp_path):
    tif_path = tmp_path / "blendv4.3_conus_temp_2026-04-11T11:00_2026-04-11T12:00.tif"
    values = np.array([[1, 2, 3], [4, 42, 6], [7, 8, 9]], dtype="float32")
    make_test_tif(tif_path, values)
    parsed = sample_cog_point.parse_file_metadata(tif_path)
    record = sample_cog_point.sample_file(parsed, "KLGA", 40.5, -73.5)
    assert record["grid_row"] == 1
    assert record["grid_col"] == 1
    assert record["sample_value"] == 42.0
    assert record["source_model"] == "NBM"
    assert record["source_product"] == "cog"
    assert record["source_version"] == "blendv4.3"
    assert record["fallback_used_any"] is False
    assert record["forecast_hour"] == pytest.approx(1.0)
    assert record["crop_mean"] == pytest.approx(82.0 / 9.0)
    assert record["nb3_mean"] == pytest.approx(82.0 / 9.0)
    assert record["nb3_std"] == pytest.approx(np.nanstd(values.astype(float)))
    assert record["nb3_gradient_west_east"] == pytest.approx(2.0)
    assert record["nb3_gradient_south_north"] == pytest.approx(-6.0)
    assert record["nb7_mean"] == pytest.approx(82.0 / 9.0)
    assert record["settlement_station_id"] == "KLGA"
    assert record["settlement_lat"] == pytest.approx(location_context.SETTLEMENT_LOCATION.lat)
    assert record["settlement_lon"] == pytest.approx(location_context.SETTLEMENT_LOCATION.lon)
    assert record["crop_left_lon"] == pytest.approx(282.5)
    assert record["crop_left"] == 282.5
    assert record["nearest_grid_lat"] == pytest.approx(40.5)
    assert record["nearest_grid_lon"] == pytest.approx(-73.5)


def test_sample_file_converts_nodata_to_none(tmp_path):
    tif_path = tmp_path / "blendv4.3_conus_temp_2026-04-11T11:00_2026-04-11T12:00.tif"
    values = np.array([[1, 2, 3], [4, -9999, 6], [7, 8, 9]], dtype="float32")
    make_test_tif(tif_path, values)
    parsed = sample_cog_point.parse_file_metadata(tif_path)
    record = sample_cog_point.sample_file(parsed, "KLGA", 40.5, -73.5)
    assert record["sample_value"] is None


def test_location_context_defaults_and_longitude_normalization():
    assert location_context.SETTLEMENT_LOCATION.station_id == "KLGA"
    assert location_context.SETTLEMENT_LOCATION.lat == pytest.approx(40.7769)
    assert location_context.SETTLEMENT_LOCATION.lon == pytest.approx(-73.8740)
    assert location_context.REGIONAL_CROP_BOUNDS.top == pytest.approx(43.5)
    assert location_context.REGIONAL_CROP_BOUNDS.bottom == pytest.approx(39.0)
    assert location_context.REGIONAL_CROP_BOUNDS.left == pytest.approx(282.5)
    assert location_context.REGIONAL_CROP_BOUNDS.right == pytest.approx(289.5)
    lon_grid = np.array([[285.5, 286.126, 286.7]])
    assert location_context.normalize_longitude_for_grid(lon_grid, -73.8740) == pytest.approx(286.126)


def test_validate_settlement_target_rejects_non_klga_args():
    args = argparse.Namespace(station_id="KJFK", lat=40.6413, lon=-73.7781)
    with pytest.raises(SystemExit):
        sample_cog_point.validate_settlement_target(args)


def test_local_context_metrics_support_3x3_7x7_and_edge_clipping():
    values = np.arange(1, 50, dtype=float).reshape(7, 7)
    metrics = location_context.local_context_metrics(values, row=3, col=3, north_is_first=True)
    assert metrics["sample_value"] == 25.0
    assert metrics["nb3_mean"] == pytest.approx(np.mean(values[2:5, 2:5]))
    assert metrics["nb7_mean"] == pytest.approx(np.mean(values))
    edge_metrics = location_context.local_context_metrics(values, row=0, col=0, north_is_first=True)
    assert edge_metrics["nb3_mean"] == pytest.approx(np.mean(values[0:2, 0:2]))
    assert edge_metrics["nb7_mean"] == pytest.approx(np.mean(values[0:4, 0:4]))


def test_local_context_metrics_handle_all_nan_neighborhoods_without_crashing():
    values = np.full((7, 7), np.nan, dtype=float)
    metrics = location_context.local_context_metrics(values, row=3, col=3, north_is_first=True)
    assert metrics["sample_value"] is None
    assert metrics["nb3_mean"] is None
    assert metrics["nb7_gradient_west_east"] is None


def test_crop_context_metrics_handle_all_nan_crops_without_crashing():
    values = np.full((7, 7), np.nan, dtype=float)
    metrics = location_context.crop_context_metrics(values)
    assert metrics["crop_mean"] is None
    assert metrics["crop_min"] is None
    assert metrics["crop_max"] is None
    assert metrics["crop_std"] is None


def test_sampler_csv_and_parquet_outputs_match(tmp_path):
    data_dir = tmp_path / "cog"
    temp_path = data_dir / "blendv4.3_conus_temp_2026-04-11T11:00_2026-04-11T12:00.tif"
    maxt_path = data_dir / "blendv4.3_conus_maxt_2026-04-11T11:00_2026-04-12T06:00.tif"
    make_test_tif(temp_path, np.array([[1, 2, 3], [4, 50, 6], [7, 8, 9]], dtype="float32"))
    make_test_tif(maxt_path, np.array([[1, 2, 3], [4, 60, 6], [7, 8, 9]], dtype="float32"))

    csv_args = argparse.Namespace(
        input_dir=data_dir,
        output=tmp_path / "features.csv",
        long_output=tmp_path / "features_long.csv",
        output_format="csv",
        station_id="KLGA",
        lat=40.5,
        lon=-73.5,
        variables=None,
        workers=2,
    )
    parquet_args = argparse.Namespace(
        input_dir=data_dir,
        output=tmp_path / "features.parquet",
        long_output=tmp_path / "features_long.parquet",
        output_format="parquet",
        station_id="KLGA",
        lat=40.5,
        lon=-73.5,
        variables=None,
        workers=2,
    )

    csv_files = sample_cog_point.discover_files(csv_args.input_dir, None)
    csv_chunks = sample_cog_point.collect_chunk_paths(csv_args, csv_files)
    sample_cog_point.write_outputs(csv_args, csv_chunks)

    parquet_files = sample_cog_point.discover_files(parquet_args.input_dir, None)
    parquet_chunks = sample_cog_point.collect_chunk_paths(parquet_args, parquet_files)
    sample_cog_point.write_outputs(parquet_args, parquet_chunks)

    csv_df = pd.read_csv(csv_args.output)
    parquet_df = pd.read_parquet(parquet_args.output)
    assert list(csv_df.columns) == list(parquet_df.columns)
    assert csv_df.fillna(-9999).to_dict(orient="records") == parquet_df.fillna(-9999).to_dict(orient="records")
    assert "source_model" in csv_df.columns
    assert "source_product" in csv_df.columns
    assert "source_version" in csv_df.columns
    assert "forecast_hour" in csv_df.columns
    assert "nearest_grid_lat" in csv_df.columns
    assert "nearest_grid_lon" in csv_df.columns
    assert "temp_crop_mean" in csv_df.columns
    assert "temp_nb7_mean" in csv_df.columns
    assert "temp_nb3_gradient_west_east" in csv_df.columns
    assert "temp_gradient_west_east" not in csv_df.columns
    assert "maxt_nb3_std" in csv_df.columns


def test_long_output_extension_validation(tmp_path):
    with pytest.raises(SystemExit):
        sample_cog_point.validate_output_path(tmp_path / "bad.parquet", "csv", "--long-output")


def make_inventory_line(record: int, short_name: str, level: str, step: str, extra: str = "") -> str:
    suffix = f":{extra}" if extra else ":"
    return f"{record}:{record * 100}:d=2026010100:{short_name}:{level}:{step}{suffix}"


def make_grib_dataset(var_name: str, short_name: str, values: np.ndarray) -> xr.Dataset:
    lat = np.array(
        [
            [41.2, 41.2, 41.2],
            [40.7769, 40.7769, 40.7769],
            [40.2, 40.2, 40.2],
        ]
    )
    lon = np.array(
        [
            [285.5, 286.126, 286.7],
            [285.5, 286.126, 286.7],
            [285.5, 286.126, 286.7],
        ]
    )
    data_array = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"latitude": (("y", "x"), lat), "longitude": (("y", "x"), lon)},
        attrs={
            "GRIB_shortName": short_name,
            "GRIB_typeOfLevel": "surface",
            "GRIB_stepType": "instant",
            "GRIB_name": short_name,
            "units": "1",
        },
    )
    return xr.Dataset({var_name: data_array})


def make_crop_args(
    tmp_path: pathlib.Path,
    *,
    crop_method: str = "auto",
    wgrib2_threads: int | None = None,
    workers: int = 1,
    reduce_workers: int = 1,
    region: str = "co",
) -> argparse.Namespace:
    return argparse.Namespace(
        region=region,
        crop_method=crop_method,
        crop_grib_type="same",
        wgrib2_threads=wgrib2_threads,
        workers=workers,
        reduce_workers=reduce_workers,
        scratch_dir=tmp_path / "scratch",
        output_dir=tmp_path / "output",
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
    )


def test_crop_selected_grib2_auto_uses_cached_ij_box_when_available(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    dataset = make_grib_dataset("t2m", "TMP", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    open_calls: list[pathlib.Path] = []
    commands: list[list[str]] = []

    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **_kwargs: open_calls.append(path) or [dataset],
    )

    def fake_run_command(command, *, input_text=None, env=None):
        del input_text
        commands.append(command)
        if "-grid" in command:
            return "1:0:grid_template=30: Lambert conformal grid:(3 x 3) units 1e-06"
        reduced_path.write_bytes(b"reduced")
        return "ok"

    monkeypatch.setattr(build_grib2_features, "run_command", fake_run_command)
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path)
    first = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )
    second = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )

    assert first.method_used == "ijsmall_grib"
    assert first.crop_grid_cache_hit is False
    assert second.method_used == "ijsmall_grib"
    assert second.crop_grid_cache_hit is True
    assert first.crop_ij_box == "1:3 1:3"
    assert len(open_calls) == 1
    assert any("-ijsmall_grib" in command for command in commands)


def test_read_wgrib2_grid_shape_accepts_lambert_style_output(tmp_path, monkeypatch):
    reduced_path = tmp_path / "reduced.grib2"
    reduced_path.write_bytes(b"reduced")

    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")
    monkeypatch.setattr(
        build_grib2_features,
        "run_command",
        lambda *_args, **_kwargs: (
            "1:0:grid_template=30:\n"
            "        Lambert Conformal: (47 x 19) input WE:SN output WE:SN\n"
            "        Latin 1 25.000000 Latin 2 25.000000\n"
        ),
    )

    assert build_grib2_features.read_wgrib2_grid_shape(reduced_path) == (47, 19)


def test_read_wgrib2_grid_shape_accepts_nx_ny_output(tmp_path, monkeypatch):
    reduced_path = tmp_path / "reduced.grib2"
    reduced_path.write_bytes(b"reduced")

    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")
    monkeypatch.setattr(
        build_grib2_features,
        "run_command",
        lambda *_args, **_kwargs: "1:0:grid_template=30: Lambert conformal nx=47 ny=19 input WE:SN output WE:SN",
    )

    assert build_grib2_features.read_wgrib2_grid_shape(reduced_path) == (47, 19)


def test_run_command_returns_stderr_when_stdout_empty(monkeypatch):
    class Completed:
        stdout = ""
        stderr = "metadata on stderr"

    monkeypatch.setattr(build_grib2_features.subprocess, "run", lambda *_args, **_kwargs: Completed())

    assert build_grib2_features.run_command(["tool", "-grid"]) == "metadata on stderr"


def test_crop_ij_box_from_north_first_partial_grid():
    lat_grid = np.array(
        [
            [43.0, 43.0, 43.0, 43.0],
            [42.0, 42.0, 42.0, 42.0],
            [41.0, 41.0, 41.0, 41.0],
            [40.0, 40.0, 40.0, 40.0],
        ]
    )
    lon_grid = np.array(
        [
            [283.0, 285.0, 287.0, 289.0],
            [283.0, 285.0, 287.0, 289.0],
            [283.0, 285.0, 287.0, 289.0],
            [283.0, 285.0, 287.0, 289.0],
        ]
    )

    north_box = build_grib2_features.crop_ij_box_from_grid(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        bounds=location_context.CropBounds(top=43.1, bottom=42.5, left=282.5, right=286.0),
        north_is_first=True,
        west_is_first=True,
    )
    middle_box = build_grib2_features.crop_ij_box_from_grid(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        bounds=location_context.CropBounds(top=42.2, bottom=40.8, left=284.0, right=288.0),
        north_is_first=True,
        west_is_first=True,
    )

    assert north_box.as_text() == "1:2 4:4"
    assert middle_box.as_text() == "2:3 2:3"


def test_crop_selected_grib2_auto_negative_caches_failed_ij_validation(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    dataset = make_grib_dataset("t2m", "TMP", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    commands: list[list[str]] = []
    open_calls: list[pathlib.Path] = []

    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **_kwargs: open_calls.append(path) or [dataset],
    )

    def fake_run_command(command, *, input_text=None, env=None):
        del input_text, env
        commands.append(command)
        if "-grid" in command:
            return "1:0:grid_template=30: Lambert conformal grid:(2 x 2) units 1e-06"
        reduced_path.write_bytes(b"reduced")
        return "ok"

    monkeypatch.setattr(build_grib2_features, "run_command", fake_run_command)
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path, crop_method="auto")
    first = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )
    second = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )

    assert first.method_used == "small_grib"
    assert second.method_used == "small_grib"
    assert sum(1 for command in commands if "-ijsmall_grib" in command) == 1
    assert sum(1 for command in commands if "-small_grib" in command) == 2
    assert len(open_calls) == 1


def test_crop_selected_grib2_auto_falls_back_to_small_grib_when_ij_unavailable(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    commands: list[list[str]] = []

    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no grid")))

    def fake_run_command(command, *, input_text=None, env=None):
        del input_text, env
        commands.append(command)
        reduced_path.write_bytes(b"reduced")
        return "ok"

    monkeypatch.setattr(build_grib2_features, "run_command", fake_run_command)
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path, crop_method="auto")
    result = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )

    assert result.method_used == "small_grib"
    assert result.crop_grid_cache_key is None
    assert commands[0][3] == "same"
    assert "-small_grib" in commands[0]


def test_crop_selected_grib2_forced_small_grib_bypasses_grid_cache(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")

    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected grid open")))
    monkeypatch.setattr(
        build_grib2_features,
        "run_command",
        lambda command, *, input_text=None, env=None: reduced_path.write_bytes(b"reduced") or "ok",
    )
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path, crop_method="small_grib")
    result = build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )

    assert result.method_used == "small_grib"
    assert result.crop_grid_cache_key is None
    assert result.crop_grid_cache_hit is False


def test_crop_selected_grib2_forced_ijsmall_grib_errors_when_ij_resolution_fails(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")

    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no grid")))
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path, crop_method="ijsmall_grib")
    with pytest.raises(RuntimeError):
        build_grib2_features.crop_selected_grib2(
            args=args,
            raw_path=raw_path,
            reduced_path=reduced_path,
            left=args.left,
            right=args.right,
            bottom=args.bottom,
            top=args.top,
        )


def test_crop_grid_cache_key_changes_with_region(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    dataset = make_grib_dataset("t2m", "TMP", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", lambda *_args, **_kwargs: [dataset])

    co_entry, _ = build_grib2_features.resolve_crop_grid_cache_entry(
        args=make_crop_args(tmp_path / "co", region="co"),
        raw_path=raw_path,
        reduced_path=reduced_path,
        bounds=location_context.CropBounds(top=43.5, bottom=39.0, left=282.5, right=289.5),
    )
    pr_entry, _ = build_grib2_features.resolve_crop_grid_cache_entry(
        args=make_crop_args(tmp_path / "pr", region="pr"),
        raw_path=raw_path,
        reduced_path=reduced_path,
        bounds=location_context.CropBounds(top=43.5, bottom=39.0, left=282.5, right=289.5),
    )

    assert co_entry.signature != pr_entry.signature


def test_crop_grid_identity_cache_avoids_probe_on_warm_hit(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    dataset = make_grib_dataset("t2m", "TMP", np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]))
    open_calls: list[pathlib.Path] = []

    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **_kwargs: open_calls.append(path) or [dataset],
    )

    args = make_crop_args(tmp_path)
    bounds = location_context.CropBounds(top=43.5, bottom=39.0, left=282.5, right=289.5)
    first, first_hit = build_grib2_features.resolve_crop_grid_cache_entry(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        bounds=bounds,
    )
    second, second_hit = build_grib2_features.resolve_crop_grid_cache_entry(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        bounds=bounds,
    )

    assert first.signature == second.signature
    assert first_hit is False
    assert second_hit is True
    assert len(open_calls) == 1


def test_crop_selected_grib2_passes_wgrib2_env(tmp_path, monkeypatch):
    raw_path = tmp_path / "raw.grib2"
    reduced_path = tmp_path / "reduced.grib2"
    raw_path.write_bytes(b"raw")
    seen_env: list[dict[str, str] | None] = []

    def fake_run_command(command, *, input_text=None, env=None):
        del command, input_text
        seen_env.append(env)
        reduced_path.write_bytes(b"reduced")
        return "ok"

    monkeypatch.setattr(build_grib2_features, "run_command", fake_run_command)
    monkeypatch.setattr(build_grib2_features, "resolve_wgrib2_binary", lambda: "wgrib2")

    args = make_crop_args(tmp_path, crop_method="small_grib", wgrib2_threads=3)
    build_grib2_features.crop_selected_grib2(
        args=args,
        raw_path=raw_path,
        reduced_path=reduced_path,
        left=args.left,
        right=args.right,
        bottom=args.bottom,
        top=args.top,
    )

    assert seen_env[0]["OMP_NUM_THREADS"] == "3"
    assert seen_env[0]["OMP_WAIT_POLICY"] == "PASSIVE"


def test_crop_wgrib2_thread_count_auto_policy():
    assert build_grib2_features.crop_wgrib2_thread_count(SimpleNamespace(wgrib2_threads=None, workers=1, reduce_workers=1)) == 2
    assert build_grib2_features.crop_wgrib2_thread_count(SimpleNamespace(wgrib2_threads=None, workers=2, reduce_workers=1)) == 1
    assert build_grib2_features.crop_wgrib2_thread_count(SimpleNamespace(wgrib2_threads=None, workers=1, reduce_workers=2)) == 1


def test_discover_cycle_plans_filters_by_local_date_and_derives_mode():
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": [f"blend.20260101/{hour:02d}/" for hour in (0, 12, 15, 23)],
            "blend.20260102/": ["blend.20260102/02/"],
        }
    )
    plans = build_grib2_features.discover_cycle_plans(
        dt.date(2026, 1, 1),
        dt.date(2026, 1, 1),
        client=client,
    )
    assert [plan.cycle for plan in plans] == ["12", "15", "23", "02"]
    assert [plan.mode for plan in plans] == ["premarket", "intraday", "intraday", "intraday"]


def test_discover_cycle_plans_overnight_mode_keeps_only_cutoff_eligible_cycles():
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": [f"blend.20260101/{hour:02d}/" for hour in (0, 6, 12, 18, 23)],
            "blend.20260102/": [f"blend.20260102/{hour:02d}/" for hour in (0, 2, 6)],
        }
    )
    plans = build_grib2_features.discover_cycle_plans(
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 2),
        client=client,
        selection_mode="overnight_0005",
    )
    assert [plan.cycle for plan in plans] == ["02"]
    assert [plan.init_time_local.isoformat() for plan in plans] == ["2026-01-01T21:00:00-05:00"]
    assert [target_date.isoformat() for target_date in plans[0].selected_target_dates] == ["2026-01-02"]


def test_discover_cycle_plans_overnight_mode_selects_latest_issue_per_target_day():
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": [f"blend.20260101/{hour:02d}/" for hour in (23,)],
            "blend.20260102/": [f"blend.20260102/{hour:02d}/" for hour in (0, 2, 4, 23)],
            "blend.20260103/": [f"blend.20260103/{hour:02d}/" for hour in (0, 3, 6)],
        }
    )
    plans = build_grib2_features.discover_cycle_plans(
        dt.date(2026, 1, 2),
        dt.date(2026, 1, 3),
        client=client,
        selection_mode="overnight_0005",
    )

    assert [plan.cycle_token for plan in plans] == ["20260102T0400Z", "20260103T0300Z"]
    assert [[target_date.isoformat() for target_date in plan.selected_target_dates] for plan in plans] == [
        ["2026-01-02"],
        ["2026-01-03"],
    ]


def test_lead_hours_for_cycle_defaults_to_full_lead_window():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 2, 4, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 23, tzinfo=build_grib2_features.NY_TZ),
        cycle="04",
    )

    assert build_grib2_features.lead_hours_for_cycle(cycle_plan) == build_grib2_features.LEAD_HOURS


def test_lead_hours_for_cycle_keeps_only_selected_target_day_leads():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 2, 2, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 21, tzinfo=build_grib2_features.NY_TZ),
        cycle="02",
        selected_target_dates=(dt.date(2026, 1, 2),),
    )

    assert build_grib2_features.lead_hours_for_cycle(cycle_plan) == list(range(3, 27))


def test_group_filters_split_height_above_ground_by_level():
    filters = [item for item in build_grib2_features.GROUP_FILTERS if item["typeOfLevel"] == "heightAboveGround"]
    assert {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2} in filters
    assert {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 10} in filters
    assert {"typeOfLevel": "heightAboveGround", "stepType": "max", "level": 2} in filters
    assert {"typeOfLevel": "heightAboveGround", "stepType": "min", "level": 2} in filters


def test_canonical_feature_name_maps_cfgrib_aliases():
    aliases = {
        "t2m": ("2t", "t2m", "2 metre temperature", "tmp"),
        "d2m": ("2d", "d2m", "2 metre dewpoint temperature", "dpt"),
        "r2": ("2r", "r2", "2 metre relative humidity", "rh"),
        "si10": ("10si", "si10", "10 metre wind speed", "wind"),
        "wdir10": ("10wdir", "wdir10", "10 metre wind direction", "wdir"),
        "i10fg": ("i10fg", "i10fg", "Instantaneous 10 metre wind gust", "gust"),
        "tcc": ("tcc", "tcc", "Total Cloud Cover", "tcdc"),
        "sdswrf": ("sdswrf", "sdswrf", "Surface downward short-wave radiation flux", "dswrf"),
        "tp": ("tp", "tp", "Total Precipitation", "apcp"),
    }
    for var_name, (grib_short_name, cf_var_name, grib_name, expected) in aliases.items():
        data_array = xr.DataArray(
            np.ones((2, 2), dtype=float),
            dims=("y", "x"),
            attrs={
                "GRIB_shortName": grib_short_name,
                "GRIB_cfVarName": cf_var_name,
                "GRIB_name": grib_name,
            },
        )
        assert build_grib2_features.canonical_feature_name(var_name, data_array) == expected


def test_record_matches_data_array_accepts_cfgrib_alias_short_names():
    aliases = [
        ("TMP", "2 m above ground", "1 hour fcst", "t2m", "2t", "2 metre temperature", "heightAboveGround", 2, "instant"),
        ("DPT", "2 m above ground", "1 hour fcst", "d2m", "2d", "2 metre dewpoint temperature", "heightAboveGround", 2, "instant"),
        ("RH", "2 m above ground", "1 hour fcst", "r2", "2r", "2 metre relative humidity", "heightAboveGround", 2, "instant"),
        ("WIND", "10 m above ground", "1 hour fcst", "si10", "10si", "10 metre wind speed", "heightAboveGround", 10, "instant"),
        ("WDIR", "10 m above ground", "1 hour fcst", "wdir10", "10wdir", "10 metre wind direction", "heightAboveGround", 10, "instant"),
        ("GUST", "10 m above ground", "1 hour fcst", "i10fg", "i10fg", "Instantaneous 10 metre wind gust", "heightAboveGround", 10, "instant"),
        ("TCDC", "surface", "1 hour fcst", "tcc", "tcc", "Total Cloud Cover", "surface", None, "instant"),
        ("DSWRF", "surface", "1 hour fcst", "sdswrf", "sdswrf", "Surface downward short-wave radiation flux", "surface", None, "instant"),
        ("APCP", "surface", "0-1 hour acc fcst", "tp", "tp", "Total Precipitation", "surface", None, "accum"),
    ]
    for short_name, level_text, step_text, var_name, data_short_name, grib_name, type_of_level, level, step_type in aliases:
        record = build_grib2_features.parse_idx_lines(make_inventory_line(1, short_name, level_text, step_text))[0]
        data_array = xr.DataArray(
            np.ones((2, 2), dtype=float),
            dims=("y", "x"),
            attrs={
                "GRIB_shortName": data_short_name,
                "GRIB_cfVarName": var_name,
                "GRIB_name": grib_name,
                "GRIB_typeOfLevel": type_of_level,
                "GRIB_stepType": step_type,
            },
        )
        if level is not None:
            data_array = data_array.assign_coords({type_of_level: level})
        assert build_grib2_features.record_matches_data_array(record, data_array) is True


def test_select_inventory_records_uses_fallback_levels_and_skips_stddev():
    inventory = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "WIND", "10 m above ground", "1 hour fcst"),
            make_inventory_line(3, "WDIR", "10 m above ground", "1 hour fcst"),
            make_inventory_line(4, "TCDC", "surface", "1 hour fcst"),
            make_inventory_line(5, "VRATE", "entire atmosphere (considered as a single layer)", "1 hour fcst"),
            make_inventory_line(6, "CAPE", "surface", "1 hour fcst"),
            make_inventory_line(7, "THUNC", "entire atmosphere", "1-2 hour missing fcst"),
            make_inventory_line(8, "PWTHER", "surface - reserved", "1 hour fcst"),
            make_inventory_line(9, "TMAX", "2 m above ground", "12-24 hour max fcst", "ens std dev"),
            make_inventory_line(10, "TMAX", "2 m above ground", "12-24 hour max fcst"),
        ]
    )
    records = build_grib2_features.parse_idx_lines(inventory)
    selected, warnings, missing_required = build_grib2_features.select_inventory_records(records)
    selected_by_name = {field.spec.short_name: field.records[0].level_text for field in selected if field.records}
    assert selected_by_name["TCDC"] == "surface"
    assert selected_by_name["VRATE"] == "entire atmosphere (considered as a single layer)"
    assert selected_by_name["CAPE"] == "surface"
    assert selected_by_name["THUNC"] == "entire atmosphere"
    assert selected_by_name["TMAX"] == "2 m above ground"
    assert any("missing field DPT" in warning for warning in missing_required)


def test_build_selected_ranges_uses_next_offset_and_content_length():
    inventory = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"),
            make_inventory_line(3, "RH", "2 m above ground", "1 hour fcst"),
            make_inventory_line(4, "WIND", "10 m above ground", "1 hour fcst"),
        ]
    )
    records = build_grib2_features.parse_idx_lines(inventory)
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[records[1]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
            records=[records[3]],
        ),
    ]

    ranges = build_grib2_features.build_selected_ranges(records, selected, content_length=550)

    assert [(item.record.short_name, item.byte_start, item.byte_end) for item in ranges] == [
        ("DPT", 200, 299),
        ("WIND", 400, 549),
    ]


def test_build_selected_ranges_preserves_inventory_order_for_long_fields():
    inventory = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "PTYPE", "surface", "1 hour fcst", "prob >=1 <2"),
            make_inventory_line(3, "WIND", "10 m above ground", "1 hour fcst"),
            make_inventory_line(4, "PTYPE", "surface", "1 hour fcst", "prob >=2 <3"),
        ]
    )
    records = build_grib2_features.parse_idx_lines(inventory)
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
            records=[records[3], records[1]],
        )
    ]

    ranges = build_grib2_features.build_selected_ranges(records, selected, content_length=520)

    assert [item.record.raw_line for item in ranges] == [records[1].raw_line, records[3].raw_line]


def test_merge_selected_ranges_merges_adjacent_and_small_gaps():
    ranges = [
        build_grib2_features.SelectedRange(record=object(), byte_start=100, byte_end=199),
        build_grib2_features.SelectedRange(record=object(), byte_start=200, byte_end=299),
        build_grib2_features.SelectedRange(record=object(), byte_start=320, byte_end=399),
        build_grib2_features.SelectedRange(record=object(), byte_start=600, byte_end=699),
    ]

    merged = build_grib2_features.merge_selected_ranges(ranges, max_gap_bytes=32)

    assert [(item.byte_start, item.byte_end, item.record_count) for item in merged] == [
        (100, 399, 3),
        (600, 699, 1),
    ]


def test_selected_ranges_require_content_length_only_when_last_record_is_selected():
    inventory = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"),
            make_inventory_line(3, "WIND", "10 m above ground", "1 hour fcst"),
        ]
    )
    records = build_grib2_features.parse_idx_lines(inventory)
    non_terminal = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[records[0]],
        )
    ]
    terminal = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
            records=[records[-1]],
        )
    ]

    assert build_grib2_features.selected_ranges_require_content_length(records, non_terminal) is False
    assert build_grib2_features.selected_ranges_require_content_length(records, terminal) is True


def test_should_reuse_cached_raw_grib_requires_existing_idx_and_exact_size(tmp_path):
    raw_path = tmp_path / "cached.grib2"
    raw_path.write_bytes(b"x" * 300)

    assert build_grib2_features.should_reuse_cached_raw_grib(
        raw_path=raw_path,
        idx_was_present=True,
        overwrite=True,
        expected_selected_bytes=300,
    ) is False
    assert build_grib2_features.should_reuse_cached_raw_grib(
        raw_path=raw_path,
        idx_was_present=False,
        overwrite=False,
        expected_selected_bytes=300,
    ) is False
    assert build_grib2_features.should_reuse_cached_raw_grib(
        raw_path=raw_path,
        idx_was_present=True,
        overwrite=False,
        expected_selected_bytes=301,
    ) is False
    assert build_grib2_features.should_reuse_cached_raw_grib(
        raw_path=raw_path,
        idx_was_present=True,
        overwrite=True,
        expected_selected_bytes=300,
    ) is False


def test_build_rows_from_datasets_creates_wide_long_and_provenance_rows():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "WIND", "10 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wdir", "WDIR", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(3, "WDIR", "10 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(4, "PTYPE", "surface", "1 hour fcst", "prob >=1 <2"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
        make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
        make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        make_grib_dataset("ptype", "PTYPE", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=float)),
    ]
    valid_time = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    for dataset in datasets:
        for data_array in dataset.data_vars.values():
            data_array.coords["valid_time"] = valid_time
    stats = {}
    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
        stats=stats,
    )
    assert len(wide_rows) == 1
    wide_row = wide_rows[0]
    assert wide_row["source_model"] == "NBM"
    assert wide_row["source_product"] == "grib2-core"
    assert wide_row["source_version"] == "nbm-grib2-core-public"
    assert wide_row["forecast_hour"] == 1
    assert wide_row["tmp"] == 14.0
    assert wide_row["tmp_crop_mean"] == 14.0
    assert wide_row["tmp_nb3_max"] == 18.0
    assert wide_row["tmp_nb3_std"] == pytest.approx(np.nanstd(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)))
    assert wide_row["tmp_nb3_gradient_west_east"] == pytest.approx(2.0)
    assert wide_row["tmp_nb3_gradient_south_north"] == pytest.approx(-6.0)
    assert "tmp_gradient_west_east" not in wide_row
    assert "tmp_gradient_south_north" not in wide_row
    assert wide_row["tmp_nb7_mean"] == pytest.approx(14.0)
    assert wide_row["tmp_nb7_std"] == pytest.approx(np.nanstd(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)))
    assert wide_row["wind"] == 11.0
    assert wide_row["wdir"] == 270.0
    assert wide_row["ptype"] == 4.0
    assert wide_row["ptype_nb3_max"] == 8.0
    assert wide_row["u10"] == pytest.approx(11.0, abs=1e-6)
    assert wide_row["v10"] == pytest.approx(0.0, abs=1e-6)
    assert wide_row["fallback_used_any"] is True
    assert wide_row["missing_optional_any"] is False
    assert wide_row["missing_optional_fields_count"] == 0
    assert wide_row["mode"] == "intraday"
    assert wide_row["valid_time_utc"] == "2026-01-01T16:00:00+00:00"
    assert wide_row["settlement_station_id"] == "KLGA"
    assert wide_row["nearest_grid_lat"] == pytest.approx(40.7769)
    assert wide_row["nearest_grid_lon"] == pytest.approx(-73.874)
    assert wide_row["crop_right_lon"] == pytest.approx(289.5)
    assert wide_row["crop_right"] == pytest.approx(289.5)
    assert len(long_rows) == 1
    assert long_rows[0]["feature_name"] == "ptype"
    assert long_rows[0]["valid_time_utc"] == "2026-01-01T16:00:00+00:00"
    assert long_rows[0]["nb3_gradient_west_east"] == pytest.approx(2.0)
    assert long_rows[0]["nb7_mean"] == pytest.approx(4.0)
    tmp_provenance = next(row for row in provenance_rows if row["feature_name"] == "tmp")
    assert tmp_provenance["source_version"] == "nbm-grib2-core-public"
    assert tmp_provenance["present_directly"] is True
    assert tmp_provenance["derived"] is False
    assert tmp_provenance["missing_optional"] is False
    assert json.loads(tmp_provenance["source_feature_names"]) == []
    assert tmp_provenance["fallback_used"] is True
    assert "mapped data variable" in tmp_provenance["fallback_source_description"]
    assert tmp_provenance["nearest_grid_lat"] == pytest.approx(40.7769)
    assert tmp_provenance["nearest_grid_lon"] == pytest.approx(-73.874)
    ptype_provenance = next(row for row in provenance_rows if row["feature_name"] == "ptype")
    assert ptype_provenance["present_directly"] is True
    assert ptype_provenance["derived"] is False
    assert ptype_provenance["grib_short_name"] == "PTYPE"
    assert ptype_provenance["grib_level_text"] == "surface"
    assert ptype_provenance["grib_step_text"] == "1 hour fcst"
    u10_provenance = next(row for row in provenance_rows if row["feature_name"] == "u10")
    assert u10_provenance["derived"] is True
    assert u10_provenance["present_directly"] is False
    assert json.loads(u10_provenance["source_feature_names"]) == ["wind", "wdir"]
    assert not any(row["feature_name"] == "tmax" for row in provenance_rows)
    assert stats["wide_row_count"] == len(wide_rows)
    assert stats["long_row_count"] == len(long_rows)
    assert stats["provenance_row_count"] == len(provenance_rows)
    assert stats["timing_row_geometry_seconds"] >= 0.0
    assert stats["timing_row_metric_seconds"] >= 0.0
    assert stats["timing_row_provenance_seconds"] >= 0.0


def test_build_rows_from_datasets_batch_mode_uses_matching_step_record_for_provenance():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[
                build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0],
                build_grib2_features.parse_idx_lines(make_inventory_line(2, "TMP", "2 m above ground", "2 hour fcst"))[0],
            ],
        ),
    ]
    dataset = make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float))
    dataset = dataset.assign_coords(valid_time=dt.datetime(2026, 1, 1, 17, tzinfo=dt.timezone.utc))

    wide_rows, _, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=[dataset],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=-1,
    )

    assert wide_rows[0]["forecast_hour"] == 2
    tmp_provenance = next(row for row in provenance_rows if row["feature_name"] == "tmp")
    assert tmp_provenance["grib_step_text"] == "2 hour fcst"
    assert tmp_provenance["inventory_line"] == make_inventory_line(2, "TMP", "2 m above ground", "2 hour fcst")


def test_validate_batch_row_forecast_hours_rejects_unadmitted_leads():
    with pytest.raises(ValueError, match="outside admitted leads"):
        build_grib2_features.validate_batch_row_forecast_hours(
            expected_leads={1, 2},
            wide_by_lead={1: [{"forecast_hour": 1}], 3: [{"forecast_hour": 3}]},
            long_by_lead={},
            provenance_by_lead={},
        )


def test_filter_rows_to_selected_target_dates_drops_spillover_rows():
    rows = [
        {"forecast_hour": 23, "valid_date_local": "2026-04-01"},
        {"forecast_hour": 24, "valid_date_local": "2026-04-02"},
    ]

    filtered = build_grib2_features.filter_rows_to_selected_target_dates(rows, (dt.date(2026, 4, 1),))

    assert filtered == [{"forecast_hour": 23, "valid_date_local": "2026-04-01"}]


def test_build_rows_from_datasets_can_skip_provenance():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "PTYPE", "surface", "1 hour fcst"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
        make_grib_dataset("ptype", "PTYPE", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=float)),
    ]
    valid_time = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    for dataset in datasets:
        for data_array in dataset.data_vars.values():
            data_array.coords["valid_time"] = valid_time

    stats = {}
    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
        write_provenance=False,
        stats=stats,
    )

    assert len(wide_rows) == 1
    assert len(long_rows) == 1
    assert provenance_rows == []
    assert stats["wide_row_count"] == 1
    assert stats["long_row_count"] == 1
    assert stats["provenance_row_count"] == 0
    assert stats["provenance_written"] is False
    assert stats["timing_row_provenance_seconds"] == 0.0


def test_build_rows_from_datasets_can_skip_unwritten_long_rows():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "PTYPE", "surface", "1 hour fcst"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("ptype", "PTYPE", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=float)),
    ]
    for dataset in datasets:
        for data_array in dataset.data_vars.values():
            data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)

    stats = {}
    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
        write_long=False,
        write_provenance=False,
        stats=stats,
    )

    assert len(wide_rows) == 1
    assert wide_rows[0]["ptype"] == 4.0
    assert long_rows == []
    assert provenance_rows == []
    assert stats["long_row_count"] == 0


def test_build_rows_from_datasets_overnight_metric_profile_limits_wide_metrics():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmax", "TMAX", ("2 m above ground",), "wide", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "TMAX", "2 m above ground", "1 hour max fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dswrf", "DSWRF", ("surface",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(3, "DSWRF", "surface", "1 hour avg fcst"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
        make_grib_dataset("tmax", "TMAX", np.array([[20, 21, 22], [23, 24, 25], [26, 27, 28]], dtype=float)),
        make_grib_dataset("dswrf", "DSWRF", np.array([[100, 110, 120], [130, 140, 150], [160, 170, 180]], dtype=float)),
    ]
    datasets[1]["tmax"].attrs["GRIB_stepType"] = "max"
    datasets[2]["dswrf"].attrs["GRIB_stepType"] = "avg"
    for dataset in datasets:
        for data_array in dataset.data_vars.values():
            data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)

    wide_rows, _, _ = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
        write_long=False,
        write_provenance=False,
        metric_profile="overnight",
    )

    wide_row = wide_rows[0]
    assert wide_row["tmp"] == 14.0
    assert wide_row["tmp_crop_mean"] == 14.0
    assert wide_row["tmp_nb3_mean"] == 14.0
    assert "tmp_nb3_std" not in wide_row
    assert wide_row["tmax_nb7_max"] == 28.0
    assert wide_row["tmax_crop_max"] == 28.0
    assert "tmax_crop_mean" not in wide_row
    assert wide_row["dswrf_crop_max"] == 180.0
    assert "dswrf_crop_mean" not in wide_row


def test_build_rows_from_datasets_mirrors_regime_long_features_to_wide_rows():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("ptype", "PTYPE", ("surface",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "PTYPE", "surface", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("pwther", "PWTHER", ("surface - reserved",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(3, "PWTHER", "surface - reserved", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tstm", "TSTM", ("surface",), "long", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(4, "TSTM", "surface", "1 hour fcst"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("tmp", "TMP", np.full((3, 3), 280.0)),
        make_grib_dataset("ptype", "PTYPE", np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=float)),
        make_grib_dataset("pwther", "PWTHER", np.array([[1, 1, 1], [2, 3, 4], [5, 5, 5]], dtype=float)),
        make_grib_dataset("tstm", "TSTM", np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)),
    ]
    valid_time = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    for dataset in datasets:
        for data_array in dataset.data_vars.values():
            data_array.coords["valid_time"] = valid_time

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert len(wide_rows) == 1
    wide_row = wide_rows[0]
    assert wide_row["ptype"] == 4.0
    assert wide_row["pwther"] == 3.0
    assert wide_row["tstm"] == 50.0
    assert wide_row["tstm_nb7_max"] == 90.0
    assert [row["feature_name"] for row in long_rows] == ["ptype", "pwther", "tstm"]
    assert {"ptype", "pwther", "tstm"} <= {row["feature_name"] for row in provenance_rows if row["present_directly"]}


def test_build_rows_from_datasets_keeps_alias_mapped_required_wide_features():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("rh", "RH", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(3, "RH", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(4, "WIND", "10 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wdir", "WDIR", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(5, "WDIR", "10 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("gust", "GUST", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(6, "GUST", "10 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec(
                "tcdc",
                "TCDC",
                ("surface", "entire atmosphere (considered as a single layer)", "reserved"),
                "wide",
            ),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(7, "TCDC", "surface", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dswrf", "DSWRF", ("surface",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(8, "DSWRF", "surface", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("apcp", "APCP", ("surface",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(9, "APCP", "surface", "0-1 hour acc fcst"))[0]],
        ),
    ]
    datasets = [
        make_grib_dataset("t2m", "2t", np.full((3, 3), 280.0)),
        make_grib_dataset("d2m", "2d", np.full((3, 3), 275.0)),
        make_grib_dataset("r2", "2r", np.full((3, 3), 55.0)),
        make_grib_dataset("si10", "10si", np.full((3, 3), 8.0)),
        make_grib_dataset("wdir10", "10wdir", np.full((3, 3), 225.0)),
        make_grib_dataset("i10fg", "i10fg", np.full((3, 3), 12.0)),
        make_grib_dataset("tcc", "tcc", np.full((3, 3), 70.0)),
        make_grib_dataset("sdswrf", "sdswrf", np.full((3, 3), 400.0)),
        make_grib_dataset("tp", "tp", np.full((3, 3), 1.25)),
    ]
    valid_time = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    for dataset in datasets[:3]:
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
            data_array.coords["heightAboveGround"] = 2
            data_array.coords["valid_time"] = valid_time
    for dataset in datasets[3:6]:
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
            data_array.coords["heightAboveGround"] = 10
            data_array.coords["valid_time"] = valid_time
    for dataset in datasets[6:8]:
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "surface"
            data_array.attrs["GRIB_stepType"] = "instant"
            data_array.coords["valid_time"] = valid_time
    for data_array in datasets[8].data_vars.values():
        data_array.attrs["GRIB_typeOfLevel"] = "surface"
        data_array.attrs["GRIB_stepType"] = "accum"
        data_array.coords["valid_time"] = valid_time

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert len(wide_rows) == 1
    wide_row = wide_rows[0]
    assert wide_row["tmp"] == 280.0
    assert wide_row["dpt"] == 275.0
    assert wide_row["rh"] == 55.0
    assert wide_row["wind"] == 8.0
    assert wide_row["wdir"] == 225.0
    assert wide_row["gust"] == 12.0
    assert wide_row["tcdc"] == 70.0
    assert wide_row["dswrf"] == 400.0
    assert wide_row["apcp"] == 1.25
    assert wide_row["u10"] == pytest.approx(5.65685424949)
    assert wide_row["v10"] == pytest.approx(5.65685424949)
    assert long_rows == []
    present_direct = {row["feature_name"] for row in provenance_rows if row["present_directly"]}
    assert {"tmp", "dpt", "rh", "wind", "wdir", "gust", "tcdc", "dswrf", "apcp"} <= present_direct


def test_build_rows_from_datasets_splits_wide_rows_by_actual_valid_time():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("thunc", "THUNC", ("entire atmosphere",), "wide", optional=True),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "THUNC", "entire atmosphere", "1-2 hour missing fcst"))[0]],
        ),
    ]
    tmp_dataset = make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float))
    thunc_dataset = make_grib_dataset("thunc", "THUNC", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
    for data_array in tmp_dataset.data_vars.values():
        data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    for data_array in thunc_dataset.data_vars.values():
        data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 17, tzinfo=dt.timezone.utc)

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=[tmp_dataset, thunc_dataset],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert len(wide_rows) == 2
    by_valid_time = {row["valid_time_utc"]: row for row in wide_rows}
    assert by_valid_time["2026-01-01T16:00:00+00:00"]["forecast_hour"] == 1
    assert by_valid_time["2026-01-01T16:00:00+00:00"]["tmp"] == 14.0
    assert by_valid_time["2026-01-01T16:00:00+00:00"]["missing_optional_any"] is True
    assert by_valid_time["2026-01-01T16:00:00+00:00"]["missing_optional_fields_count"] == 1
    assert "thunc" not in by_valid_time["2026-01-01T16:00:00+00:00"]
    assert by_valid_time["2026-01-01T17:00:00+00:00"]["forecast_hour"] == 2
    assert by_valid_time["2026-01-01T17:00:00+00:00"]["thunc"] == 5.0
    assert by_valid_time["2026-01-01T17:00:00+00:00"]["missing_optional_fields_count"] == 0
    assert "tmp" not in by_valid_time["2026-01-01T17:00:00+00:00"]
    assert long_rows == []
    assert {row["valid_time_utc"] for row in provenance_rows} == {
        "2026-01-01T16:00:00+00:00",
        "2026-01-01T17:00:00+00:00",
    }
    missing_thunc = [
        row for row in provenance_rows if row["feature_name"] == "thunc" and row["valid_time_utc"] == "2026-01-01T16:00:00+00:00"
    ]
    assert missing_thunc
    assert missing_thunc[0]["missing_optional"] is True


def test_build_rows_from_datasets_infers_unknown_long_field_from_selected_inventory():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    pwther_record = build_grib2_features.parse_idx_lines(
        make_inventory_line(8, "PWTHER", "surface - reserved", "1 hour fcst")
    )[0]
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("pwther", "PWTHER", ("surface - reserved",), "long", optional=True),
            records=[pwther_record],
        ),
    ]
    tmp_dataset = make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float))
    for data_array in tmp_dataset.data_vars.values():
        data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    values = np.array([[0, 1, 0], [2, 3, 4], [0, 0, 0]], dtype=float)
    unknown_array = xr.DataArray(
        values,
        dims=("y", "x"),
        coords={"latitude": (("y", "x"), np.array([[41.2, 41.2, 41.2], [40.7769, 40.7769, 40.7769], [40.2, 40.2, 40.2]])),
                "longitude": (("y", "x"), np.array([[285.5, 286.126, 286.7], [285.5, 286.126, 286.7], [285.5, 286.126, 286.7]]))},
        attrs={
            "GRIB_shortName": "unknown",
            "GRIB_cfVarName": "unknown",
            "GRIB_name": "unknown",
            "GRIB_typeOfLevel": "surface",
            "GRIB_stepType": "instant",
            "units": "unknown",
        },
    )
    unknown_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    datasets = [tmp_dataset, xr.Dataset({"unknown": unknown_array})]

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=datasets,
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert wide_rows[0]["missing_optional_any"] is False
    assert wide_rows[0]["missing_optional_fields_count"] == 0
    assert wide_rows[0]["pwther"] == 3.0
    assert wide_rows[0]["pwther_nb7_max"] == 4.0
    assert len(long_rows) == 1
    assert long_rows[0]["feature_name"] == "pwther"
    assert long_rows[0]["valid_time_utc"] == "2026-01-01T16:00:00+00:00"
    pwther_provenance = next(row for row in provenance_rows if row["feature_name"] == "pwther")
    assert pwther_provenance["present_directly"] is True
    assert pwther_provenance["missing_optional"] is False
    assert pwther_provenance["grib_short_name"] == "PWTHER"
    assert pwther_provenance["grib_level_text"] == "surface - reserved"
    assert pwther_provenance["grib_step_text"] == "1 hour fcst"
    assert pwther_provenance["fallback_used"] is True
    assert "unknown cfgrib variable" in pwther_provenance["fallback_source_description"]


def test_build_rows_from_datasets_skips_unselected_direct_feature_datasets(monkeypatch):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    tmp_dataset = make_grib_dataset("t2m", "TMP", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
    dpt_dataset = make_grib_dataset("d2m", "DPT", np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float))
    for dataset in (tmp_dataset, dpt_dataset):
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
            data_array.coords["heightAboveGround"] = 2
            data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)

    selected_slice_calls = []
    original_select_data_slice = build_grib2_features.select_data_slice

    def recording_select_data_slice(data_array):
        selected_slice_calls.append(str(data_array.name))
        return original_select_data_slice(data_array)

    monkeypatch.setattr(build_grib2_features, "select_data_slice", recording_select_data_slice)

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=[tmp_dataset, dpt_dataset],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert len(wide_rows) == 1
    assert wide_rows[0]["tmp"] == 5.0
    assert "dpt" not in wide_rows[0]
    assert long_rows == []
    assert [row["feature_name"] for row in provenance_rows if row["present_directly"]] == ["tmp"]
    assert selected_slice_calls == ["t2m"]


def test_build_rows_from_datasets_does_not_close_caller_owned_datasets(monkeypatch):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    for data_array in dataset.data_vars.values():
        data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
        data_array.coords["heightAboveGround"] = 2
        data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    close_calls = []
    monkeypatch.setattr(xr.Dataset, "close", lambda self: close_calls.append("closed"))

    build_grib2_features.build_rows_from_datasets(
        datasets=[dataset],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert close_calls == []


def test_build_rows_from_datasets_uses_per_variable_grid_context_for_mixed_coords():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"))[0]],
        ),
    ]
    valid_time = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    tmp_array = xr.DataArray(
        np.array([[10.0, 20.0], [30.0, 40.0]]),
        dims=("y", "x"),
        coords={
            "latitude": (("y", "x"), np.array([[40.7769, 40.7769], [39.0, 39.0]])),
            "longitude": (("y", "x"), np.array([[286.126, 288.0], [286.126, 288.0]])),
            "valid_time": valid_time,
        },
        attrs={"GRIB_shortName": "TMP", "GRIB_typeOfLevel": "heightAboveGround", "GRIB_stepType": "instant", "GRIB_level": 2},
    )
    dpt_array = xr.DataArray(
        np.array([[100.0, 200.0], [300.0, 400.0]]),
        dims=("lat", "lon"),
        coords={
            "lat": np.array([42.0, 40.7769]),
            "lon": np.array([284.0, 286.126]),
            "valid_time": valid_time,
        },
        attrs={"GRIB_shortName": "DPT", "GRIB_typeOfLevel": "heightAboveGround", "GRIB_stepType": "instant", "GRIB_level": 2},
    )

    wide_rows, _, _ = build_grib2_features.build_rows_from_datasets(
        datasets=[xr.Dataset({"t2m": tmp_array, "d2m": dpt_array})],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert wide_rows[0]["tmp"] == 10.0
    assert wide_rows[0]["dpt"] == 400.0


def test_build_rows_from_datasets_rejects_ambiguous_duplicate_selected_wide_feature():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset_a = make_grib_dataset("t2m", "TMP", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
    dataset_b = make_grib_dataset("t2m_dup", "TMP", np.array([[101, 102, 103], [104, 105, 106], [107, 108, 109]], dtype=float))
    for dataset in (dataset_a, dataset_b):
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
            data_array.coords["heightAboveGround"] = 2
            data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)

    with pytest.raises(ValueError, match="Ambiguous duplicate dataset for selected wide feature 'tmp'"):
        build_grib2_features.build_rows_from_datasets(
            datasets=[dataset_a, dataset_b],
            selected=selected,
            cycle_plan=cycle_plan,
            lead_hour=1,
        )


def test_build_rows_from_datasets_ignores_exact_duplicate_selected_wide_feature():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    values = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    dataset_a = make_grib_dataset("t2m", "TMP", values)
    dataset_b = make_grib_dataset("t2m_dup", "TMP", values.copy())
    for dataset in (dataset_a, dataset_b):
        for data_array in dataset.data_vars.values():
            data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
            data_array.coords["heightAboveGround"] = 2
            data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=[dataset_a, dataset_b],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert len(wide_rows) == 1
    assert long_rows == []
    assert len([row for row in provenance_rows if row["feature_name"] == "tmp"]) == 1


def test_build_rows_from_datasets_disables_unknown_long_inference_when_not_selected_only():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    pwther_record = build_grib2_features.parse_idx_lines(
        make_inventory_line(8, "PWTHER", "surface - reserved", "1 hour fcst")
    )[0]
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("pwther", "PWTHER", ("surface - reserved",), "long", optional=True),
            records=[pwther_record],
        ),
    ]
    tmp_dataset = make_grib_dataset("t2m", "TMP", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
    for data_array in tmp_dataset.data_vars.values():
        data_array.attrs["GRIB_typeOfLevel"] = "heightAboveGround"
        data_array.coords["heightAboveGround"] = 2
        data_array.coords["valid_time"] = dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc)
    unknown_array = xr.DataArray(
        np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=float),
        dims=("y", "x"),
        coords={
            "latitude": (("y", "x"), np.array([[41.2, 41.2, 41.2], [40.7769, 40.7769, 40.7769], [40.2, 40.2, 40.2]])),
            "longitude": (("y", "x"), np.array([[285.5, 286.126, 286.7], [285.5, 286.126, 286.7], [285.5, 286.126, 286.7]])),
            "valid_time": dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc),
        },
        attrs={
            "GRIB_shortName": "unknown",
            "GRIB_cfVarName": "unknown",
            "GRIB_name": "unknown",
            "GRIB_typeOfLevel": "surface",
            "GRIB_stepType": "instant",
            "units": "unknown",
        },
    )
    unknown_ds = xr.Dataset({"noise": unknown_array})

    wide_rows, long_rows, provenance_rows = build_grib2_features.build_rows_from_datasets(
        datasets=[tmp_dataset, unknown_ds],
        selected=selected,
        cycle_plan=cycle_plan,
        lead_hour=1,
        allow_unknown_long_inference=False,
    )

    assert len(wide_rows) == 1
    assert long_rows == []
    assert not any(row["feature_name"] == "pwther" and row["present_directly"] for row in provenance_rows)


def test_make_base_time_columns_uses_actual_crop_bounds():
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    crop_bounds = location_context.CropBounds(top=44.0, bottom=38.5, left=281.0, right=290.0)
    row = build_grib2_features.make_base_time_columns(
        cycle_plan,
        dt.datetime(2026, 1, 1, 16, tzinfo=dt.timezone.utc),
        1,
        crop_bounds=crop_bounds,
    )
    assert row["crop_top"] == pytest.approx(44.0)
    assert row["crop_bottom"] == pytest.approx(38.5)
    assert row["crop_left"] == pytest.approx(281.0)
    assert row["crop_right"] == pytest.approx(290.0)
    assert row["crop_top_lat"] == pytest.approx(44.0)
    assert row["crop_bottom_lat"] == pytest.approx(38.5)
    assert row["crop_left_lon"] == pytest.approx(281.0)
    assert row["crop_right_lon"] == pytest.approx(290.0)


def test_process_unit_writes_reduced_and_deletes_raw_inputs(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(
        payloads={
            key: make_raw_grib_payload(),
            f"{key}.idx": idx_text,
        }
    )
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(
        args=args,
        client=client,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert result.manifest_row["extraction_status"] == "ok"
    assert result.manifest_row["download_mode"] == "byte_range_selected_records"
    assert result.manifest_row["remote_file_size"] == len(client.payloads[key])
    assert result.manifest_row["selected_record_count"] == 13
    assert result.manifest_row["selected_download_bytes"] == 1300
    assert result.manifest_row["downloaded_range_bytes"] == 1300
    assert result.manifest_row["merged_range_count"] == 1
    assert result.manifest_row["head_used"] is True
    assert result.manifest_row["raw_file_size"] == 1300
    assert result.manifest_row["processed_timestamp_utc"] == "2026-01-01T15:00:00+00:00"
    assert result.manifest_row["timing_idx_fetch_seconds"] >= 0.0
    assert result.manifest_row["timing_idx_parse_seconds"] >= 0.0
    assert result.manifest_row["timing_head_seconds"] >= 0.0
    assert result.manifest_row["timing_range_download_seconds"] >= 0.0
    assert result.manifest_row["timing_crop_seconds"] >= 0.0
    assert result.manifest_row["timing_cfgrib_open_seconds"] >= 0.0
    assert result.manifest_row["timing_row_build_seconds"] >= 0.0
    assert result.manifest_row["timing_cleanup_seconds"] >= 0.0
    assert result.manifest_row["cfgrib_open_all_dataset_count"] == 0
    assert result.manifest_row["cfgrib_filtered_fallback_open_count"] == 0
    assert result.manifest_row["cfgrib_opened_dataset_count"] == 0
    assert result.manifest_row["wide_row_count"] == 1
    assert result.manifest_row["long_row_count"] == 0
    assert result.manifest_row["provenance_row_count"] == len(result.provenance_rows)
    assert not pathlib.Path(result.manifest_row["reduced_file_path"]).exists()
    assert result.manifest_row["raw_deleted"] is True
    assert result.manifest_row["idx_deleted"] is True
    assert result.manifest_row["reduced_deleted"] is True
    assert result.manifest_row["reduced_retained"] is False
    assert result.wide_row is not None
    assert len(result.wide_rows) == 1
    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    assert not raw_path.exists()
    assert not idx_path.exists()
    assert client.range_downloads == [
        (key, [(100, 1399)], "20260101T1500Z:f001")
    ]


def test_process_unit_fails_when_required_fields_missing_and_keeps_raw_inputs(tmp_path):
    idx_text = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "WIND", "10 m above ground", "1 hour fcst"),
        ]
    )
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(
        payloads={
            key: make_raw_grib_payload(),
            f"{key}.idx": idx_text,
        }
    )
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    result = build_grib2_features.process_unit(
        args=args,
        client=client,
        cycle_plan=cycle_plan,
        lead_hour=1,
    )

    assert result.manifest_row["extraction_status"] == "error:missing_required_fields"
    assert "missing field DPT" in result.manifest_row["warnings"]
    assert result.manifest_row["raw_file_size"] == 0
    assert result.manifest_row["selected_download_bytes"] == 0
    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    assert not raw_path.exists()
    assert idx_path.exists()
    assert client.range_downloads == []


def test_process_unit_passes_progress_labels_to_downloads(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    progress_calls: list[tuple[str, str | None]] = []

    class ProgressClient(FakeGrib2Client):
        def fetch_text(self, url: str):
            progress_calls.append((self._key_from_url(url), "idx-fetch"))
            return super().fetch_text(url)

        def download_byte_ranges(self, **kwargs):
            progress_calls.append((self._key_from_url(kwargs["url"]), kwargs.get("progress_label")))
            return super().download_byte_ranges(**kwargs)

    client = ProgressClient(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("d2m", "DPT", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)),
            make_grib_dataset("r2", "RH", np.array([[80, 82, 84], [86, 88, 90], [92, 94, 96]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
            make_grib_dataset("gust10m", "GUST", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("tcc", "TCDC", np.array([[50, 55, 60], [65, 70, 75], [80, 85, 90]], dtype=float)),
            make_grib_dataset("dswrf", "DSWRF", np.array([[100, 110, 120], [130, 140, 150], [160, 170, 180]], dtype=float)),
            make_grib_dataset("apcp", "APCP", np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=float)),
            make_grib_dataset("vrate", "VRATE", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)),
            make_grib_dataset("vis", "VIS", np.array([[1000, 1100, 1200], [1300, 1400, 1500], [1600, 1700, 1800]], dtype=float)),
            make_grib_dataset("ceil", "CEIL", np.array([[100, 110, 120], [130, 140, 150], [160, 170, 180]], dtype=float)),
            make_grib_dataset("cape", "CAPE", np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["extraction_status"] == "ok"
    assert progress_calls[0] == (f"{key}.idx", "idx-fetch")
    assert progress_calls[1][1] == "20260101T1500Z:f001"


def test_process_unit_skips_head_when_last_record_is_not_selected(tmp_path, monkeypatch):
    idx_text = "\n".join(
        [
            make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"),
            make_inventory_line(2, "DPT", "2 m above ground", "1 hour fcst"),
            make_inventory_line(3, "RH", "2 m above ground", "1 hour fcst"),
            make_inventory_line(4, "WIND", "10 m above ground", "1 hour fcst"),
            make_inventory_line(5, "WDIR", "10 m above ground", "1 hour fcst"),
        ]
    )
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(size=600), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
        range_merge_gap_bytes=32,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    monkeypatch.setattr(build_grib2_features, "select_inventory_records", lambda records: ([
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[records[0]],
        ),
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[records[1]],
        ),
    ], [], []))
    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", lambda **kwargs: kwargs["reduced_path"].write_bytes(b"reduced") or "crop cmd")
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("d2m", "DPT", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["extraction_status"] == "ok"
    assert result.manifest_row["head_used"] is False
    assert result.manifest_row["remote_file_size"] is None
    assert result.manifest_row["selected_download_bytes"] == 200
    assert result.manifest_row["downloaded_range_bytes"] == 200
    assert result.manifest_row["merged_range_count"] == 1
    assert client.content_length_requests == []
    assert client.range_downloads == [(key, [(100, 299)], "20260101T1500Z:f001")]


def test_parse_args_rejects_removed_show_progress_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_grib2_features.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--show-progress",
        ],
    )

    with pytest.raises(SystemExit):
        build_grib2_features.parse_args()


def test_parse_args_accepts_pause_control_file(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_grib2_features.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--pause-control-file",
            "/tmp/nbm.pause",
        ],
    )

    args = build_grib2_features.parse_args()

    assert args.pause_control_file == "/tmp/nbm.pause"


def test_parse_args_accepts_skip_provenance(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_grib2_features.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--skip-provenance",
        ],
    )

    args = build_grib2_features.parse_args()

    assert args.skip_provenance is True


def test_parse_args_accepts_crop_controls(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_grib2_features.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--crop-method",
            "ijsmall_grib",
            "--wgrib2-threads",
            "3",
        ],
    )

    args = build_grib2_features.parse_args()

    assert args.crop_method == "ijsmall_grib"
    assert args.wgrib2_threads == 3


def test_nbm_monthly_parse_args_accepts_disable_dashboard_hotkeys(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nbm_monthly_backfill.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--disable-dashboard-hotkeys",
        ],
    )

    args = nbm_monthly_backfill.parse_args()

    assert args.disable_dashboard_hotkeys is True


def test_nbm_debug_bottlenecks_parse_args_accepts_crop_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debug_bottlenecks.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--lead-workers",
            "2",
            "--crop-method",
            "ijsmall_grib",
            "--wgrib2-threads",
            "2",
            "--crop-packings",
            "same",
            "complex1",
        ],
    )

    args = nbm_debug_bottlenecks.parse_args()

    assert args.crop_method == "ijsmall_grib"
    assert args.wgrib2_threads == 2
    assert args.crop_packings == ["same", "complex1"]


def test_nbm_debug_bottlenecks_parse_args_accepts_matrix_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debug_bottlenecks.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--lead-workers",
            "2",
            "--auto-matrix",
            "--matrix-crop-methods",
            "small_grib",
            "auto",
            "--matrix-wgrib2-threads",
            "1",
            "2",
            "--matrix-crop-packings",
            "same",
            "complex3",
            "--matrix-repetitions",
            "2",
            "--matrix-max-runs",
            "12",
            "--matrix-baseline",
            "lw2_small_grib_t1_same",
            "--report-name",
            "campaign.md",
            "--write-json-summary",
        ],
    )

    args = nbm_debug_bottlenecks.parse_args()

    assert args.auto_matrix is True
    assert args.matrix_crop_methods == ["small_grib", "auto"]
    assert args.matrix_wgrib2_threads == [1, 2]
    assert args.matrix_crop_packings == ["same", "complex3"]
    assert args.matrix_repetitions == 2
    assert args.matrix_max_runs == 12
    assert args.matrix_baseline == "lw2_small_grib_t1_same"
    assert args.report_name == "campaign.md"
    assert args.write_json_summary is True


def test_nbm_debug_bottlenecks_build_command_includes_crop_flags(tmp_path):
    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        selection_mode="overnight_0005",
        workers=1,
        lead_hour_end=None,
        crop_method="ijsmall_grib",
        wgrib2_threads=2,
    )

    command = nbm_debug_bottlenecks.build_benchmark_command(
        args,
        lead_workers=4,
        crop_packing="complex1",
        output_dir=tmp_path / "output",
        scratch_dir=tmp_path / "scratch",
    )

    assert "--crop-method" in command
    assert "ijsmall_grib" in command
    assert "--wgrib2-threads" in command
    assert "2" in command
    assert "--crop-grib-type" in command
    assert "complex1" in command


def test_nbm_debug_bottlenecks_build_configs_auto_matrix_defaults():
    args = argparse.Namespace(
        auto_matrix=True,
        lead_workers=[8],
        matrix_lead_workers=None,
        matrix_crop_methods=None,
        matrix_wgrib2_threads=None,
        matrix_crop_packings=None,
        matrix_repetitions=1,
        matrix_max_runs=18,
    )

    configs = nbm_debug_bottlenecks.build_benchmark_configs(args)

    assert len(configs) == 18
    assert configs[0] == nbm_debug_bottlenecks.BenchmarkConfig(
        lead_workers=8,
        crop_method="small_grib",
        wgrib2_threads=1,
        crop_packing="same",
        repetition_index=1,
    )
    assert configs[-1] == nbm_debug_bottlenecks.BenchmarkConfig(
        lead_workers=8,
        crop_method="ijsmall_grib",
        wgrib2_threads=2,
        crop_packing="complex3",
        repetition_index=1,
    )


def test_nbm_debug_bottlenecks_build_configs_respects_max_runs():
    args = argparse.Namespace(
        auto_matrix=True,
        lead_workers=[8],
        matrix_lead_workers=None,
        matrix_crop_methods=None,
        matrix_wgrib2_threads=None,
        matrix_crop_packings=None,
        matrix_repetitions=1,
        matrix_max_runs=10,
    )

    with pytest.raises(SystemExit, match="exceeds --matrix-max-runs=10"):
        nbm_debug_bottlenecks.build_benchmark_configs(args)


def test_nbm_debug_bottlenecks_slugify_config_is_deterministic():
    config = nbm_debug_bottlenecks.BenchmarkConfig(
        lead_workers=8,
        crop_method="auto",
        wgrib2_threads=1,
        crop_packing="complex3",
        repetition_index=2,
    )

    assert nbm_debug_bottlenecks.slugify_config(config) == "lw8_auto_t1_complex3_r2"


def test_nbm_debug_bottlenecks_report_includes_crop_metadata(tmp_path):
    base = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="auto",
        wgrib2_threads=None,
        crop_packing="same",
        wall_seconds=10.0,
        sample_count=1,
        avg_cpu_pct=10.0,
        peak_cpu_pct=12.0,
        avg_rss_mb=100.0,
        peak_rss_mb=120.0,
        manifest_rows=5,
        ok_rows=5,
        error_rows=0,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=1.0,
        total_stage_seconds={column: (4.0 if column == "timing_crop_seconds" else 1.0) for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.5,
        wall_selected_mb_s=0.4,
        top_slowest_units=[],
        output_dir=str(tmp_path / "base"),
    )
    contender = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="ijsmall_grib",
        wgrib2_threads=1,
        crop_packing="complex1",
        wall_seconds=8.0,
        sample_count=1,
        avg_cpu_pct=11.0,
        peak_cpu_pct=13.0,
        avg_rss_mb=101.0,
        peak_rss_mb=121.0,
        manifest_rows=5,
        ok_rows=5,
        error_rows=0,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=0.8,
        total_stage_seconds={column: (2.0 if column == "timing_crop_seconds" else 1.0) for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.6,
        wall_selected_mb_s=0.5,
        top_slowest_units=[],
        output_dir=str(tmp_path / "contender"),
    )

    report_path = nbm_debug_bottlenecks.write_comparison_report(tmp_path, [base, contender])
    text = report_path.read_text()

    assert "crop_method: auto" in text
    assert "crop_packing: complex1" in text
    assert "crop_delta_seconds=-2.00" in text


def test_nbm_debug_bottlenecks_campaign_report_and_json_summary(tmp_path):
    base = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="small_grib",
        wgrib2_threads=1,
        crop_packing="same",
        wall_seconds=10.0,
        sample_count=1,
        avg_cpu_pct=10.0,
        peak_cpu_pct=12.0,
        avg_rss_mb=100.0,
        peak_rss_mb=120.0,
        manifest_rows=5,
        ok_rows=5,
        error_rows=0,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=1.0,
        total_stage_seconds={column: (4.0 if column == "timing_crop_seconds" else 1.0) for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.5,
        wall_selected_mb_s=0.4,
        top_slowest_units=[],
        output_dir=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "output"),
        config_slug="lw2_small_grib_t1_same",
        stdout_log=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "stdout.log"),
        stderr_log=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "stderr.log"),
    )
    contender = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="auto",
        wgrib2_threads=1,
        crop_packing="complex3",
        wall_seconds=8.0,
        sample_count=1,
        avg_cpu_pct=11.0,
        peak_cpu_pct=13.0,
        avg_rss_mb=101.0,
        peak_rss_mb=121.0,
        manifest_rows=5,
        ok_rows=5,
        error_rows=0,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=0.8,
        total_stage_seconds={column: (2.0 if column == "timing_crop_seconds" else 1.0) for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.6,
        wall_selected_mb_s=0.5,
        top_slowest_units=[],
        output_dir=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "output"),
        config_slug="lw2_auto_t1_complex3",
        stdout_log=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "stdout.log"),
        stderr_log=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "stderr.log"),
    )
    failed = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="ijsmall_grib",
        wgrib2_threads=2,
        crop_packing="same",
        wall_seconds=3.0,
        sample_count=1,
        avg_cpu_pct=9.0,
        peak_cpu_pct=10.0,
        avg_rss_mb=90.0,
        peak_rss_mb=95.0,
        manifest_rows=0,
        ok_rows=0,
        error_rows=0,
        total_downloaded_mb=0.0,
        total_selected_mb=0.0,
        total_reduced_mb=0.0,
        total_stage_seconds={column: 0.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 0.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 0.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=None,
        wall_selected_mb_s=None,
        top_slowest_units=[],
        output_dir=str(tmp_path / "runs" / "lw2_ijsmall_grib_t2_same" / "output"),
        config_slug="lw2_ijsmall_grib_t2_same",
        succeeded=False,
        return_code=1,
        stdout_log=str(tmp_path / "runs" / "lw2_ijsmall_grib_t2_same" / "stdout.log"),
        stderr_log=str(tmp_path / "runs" / "lw2_ijsmall_grib_t2_same" / "stderr.log"),
        failure_message="failed run",
    )

    report_path = nbm_debug_bottlenecks.write_campaign_report(
        tmp_path,
        [base, contender, failed],
        baseline_selector="lw2_small_grib_t1_same",
        matrix_meta={
            "matrix_lead_workers": [2],
            "matrix_crop_methods": ["small_grib", "auto", "ijsmall_grib"],
            "matrix_wgrib2_threads": [1, 2],
            "matrix_crop_packings": ["same", "complex3"],
            "matrix_repetitions": 1,
        },
        report_name="campaign.md",
    )
    summary_path = nbm_debug_bottlenecks.write_json_summary(tmp_path, [base, contender, failed])

    report_text = report_path.read_text()
    payload = json.loads(summary_path.read_text())

    assert report_path.name == "campaign.md"
    assert "Recommended Current Production Candidate" in report_text
    assert "lw2_auto_t1_complex3 vs lw2_small_grib_t1_same" in report_text
    assert "status: failed" in report_text
    assert payload[1]["config_slug"] == "lw2_auto_t1_complex3"
    assert payload[2]["succeeded"] is False


def test_nbm_debug_bottlenecks_write_matrix_metadata_single_run_uses_actual_dimensions(tmp_path):
    args = argparse.Namespace(
        auto_matrix=False,
        crop_method="auto",
        crop_packings=["complex3"],
        wgrib2_threads=2,
        lead_workers=[8],
        matrix_crop_methods=None,
        matrix_wgrib2_threads=None,
        matrix_crop_packings=None,
        matrix_lead_workers=None,
        matrix_repetitions=1,
    )
    configs = [
        nbm_debug_bottlenecks.BenchmarkConfig(
            lead_workers=8,
            crop_method="auto",
            wgrib2_threads=2,
            crop_packing="complex3",
        )
    ]

    matrix_path = nbm_debug_bottlenecks.write_matrix_metadata(tmp_path, args, configs)
    payload = json.loads(matrix_path.read_text())

    assert payload["matrix_crop_methods"] == ["auto"]
    assert payload["matrix_wgrib2_threads"] == [2]
    assert payload["matrix_crop_packings"] == ["complex3"]
    assert payload["matrix_lead_workers"] == [8]


def test_nbm_debug_bottlenecks_campaign_report_excludes_partial_error_runs_from_baseline_and_best(tmp_path):
    degraded = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="small_grib",
        wgrib2_threads=1,
        crop_packing="same",
        wall_seconds=5.0,
        sample_count=1,
        avg_cpu_pct=10.0,
        peak_cpu_pct=12.0,
        avg_rss_mb=100.0,
        peak_rss_mb=120.0,
        manifest_rows=5,
        ok_rows=4,
        error_rows=1,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=1.0,
        total_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.5,
        wall_selected_mb_s=0.4,
        top_slowest_units=[],
        output_dir=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "output"),
        config_slug="lw2_small_grib_t1_same",
        stdout_log=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "stdout.log"),
        stderr_log=str(tmp_path / "runs" / "lw2_small_grib_t1_same" / "stderr.log"),
    )
    clean = nbm_debug_bottlenecks.RunSummary(
        lead_workers=2,
        crop_method="auto",
        wgrib2_threads=1,
        crop_packing="complex3",
        wall_seconds=8.0,
        sample_count=1,
        avg_cpu_pct=11.0,
        peak_cpu_pct=13.0,
        avg_rss_mb=101.0,
        peak_rss_mb=121.0,
        manifest_rows=5,
        ok_rows=5,
        error_rows=0,
        total_downloaded_mb=5.0,
        total_selected_mb=4.0,
        total_reduced_mb=0.8,
        total_stage_seconds={column: (2.0 if column == "timing_crop_seconds" else 1.0) for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
        wall_download_mb_s=0.6,
        wall_selected_mb_s=0.5,
        top_slowest_units=[],
        output_dir=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "output"),
        config_slug="lw2_auto_t1_complex3",
        stdout_log=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "stdout.log"),
        stderr_log=str(tmp_path / "runs" / "lw2_auto_t1_complex3" / "stderr.log"),
    )

    report_path = nbm_debug_bottlenecks.write_campaign_report(
        tmp_path,
        [degraded, clean],
        baseline_selector=None,
        matrix_meta={
            "matrix_lead_workers": [2],
            "matrix_crop_methods": ["small_grib", "auto"],
            "matrix_wgrib2_threads": [1],
            "matrix_crop_packings": ["same", "complex3"],
            "matrix_repetitions": 1,
        },
        report_name="campaign.md",
    )
    text = report_path.read_text()

    assert "- baseline: lw2_auto_t1_complex3" in text
    assert "crop_method=small_grib" not in text.split("### Best By Crop Method", 1)[1].split("###", 1)[0]


def test_nbm_debug_bottlenecks_main_writes_isolated_run_outputs(tmp_path, monkeypatch):
    run_root = tmp_path / "bench"

    monkeypatch.setattr(
        nbm_debug_bottlenecks,
        "run_single_benchmark",
        lambda args, config: nbm_debug_bottlenecks.RunSummary(
            lead_workers=config.lead_workers,
            crop_method=config.crop_method,
            wgrib2_threads=config.wgrib2_threads,
            crop_packing=config.crop_packing,
            wall_seconds=10.0,
            sample_count=0,
            avg_cpu_pct=0.0,
            peak_cpu_pct=0.0,
            avg_rss_mb=0.0,
            peak_rss_mb=0.0,
            manifest_rows=1,
            ok_rows=1,
            error_rows=0,
            total_downloaded_mb=1.0,
            total_selected_mb=1.0,
            total_reduced_mb=1.0,
            total_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
            avg_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
            p95_stage_seconds={column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
            stage_pressure_x={column: 0.1 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
            wall_download_mb_s=0.1,
            wall_selected_mb_s=0.1,
            top_slowest_units=[],
            output_dir=str(run_root / "runs" / nbm_debug_bottlenecks.slugify_config(config) / "output"),
            config_slug=nbm_debug_bottlenecks.slugify_config(config),
            stdout_log=str(run_root / "runs" / nbm_debug_bottlenecks.slugify_config(config) / "stdout.log"),
            stderr_log=str(run_root / "runs" / nbm_debug_bottlenecks.slugify_config(config) / "stderr.log"),
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "debug_bottlenecks.py",
            "--start-local-date",
            "2026-01-01",
            "--end-local-date",
            "2026-01-01",
            "--lead-workers",
            "8",
            "--run-root",
            str(run_root),
            "--auto-matrix",
            "--matrix-crop-methods",
            "small_grib",
            "auto",
            "--matrix-wgrib2-threads",
            "1",
            "--matrix-crop-packings",
            "same",
            "--write-json-summary",
        ],
    )

    assert nbm_debug_bottlenecks.main() == 0
    assert (run_root / "report.md").exists()
    assert (run_root / "summary.json").exists()
    matrix_payload = json.loads((run_root / "matrix.json").read_text())
    assert matrix_payload["configs"][0]["config_slug"] == "lw8_small_grib_t1_same"
    assert matrix_payload["configs"][1]["config_slug"] == "lw8_auto_t1_same"


def test_nbm_debug_bottlenecks_summarize_run_uses_effective_crop_threads_from_manifest(tmp_path):
    manifest = pd.DataFrame(
        [
                {
                    "init_time_utc": "2026-01-01T05:00:00+00:00",
                    "valid_time_utc": "2026-01-01T06:00:00+00:00",
                    "forecast_hour": 1,
                    "extraction_status": "ok",
                    "downloaded_range_bytes": 1024,
                    "selected_download_bytes": 1024,
                    "selected_record_count": 10,
                    "reduced_file_size": 256,
                    "crop_wgrib2_threads": 1,
                    **{column: 1.0 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
                },
                {
                    "init_time_utc": "2026-01-01T05:00:00+00:00",
                    "valid_time_utc": "2026-01-01T07:00:00+00:00",
                    "forecast_hour": 2,
                    "extraction_status": "ok",
                    "downloaded_range_bytes": 1024,
                    "selected_download_bytes": 1024,
                    "selected_record_count": 10,
                    "reduced_file_size": 256,
                    "crop_wgrib2_threads": 1,
                    **{column: 1.5 for column in nbm_debug_bottlenecks.TIMING_COLUMNS},
                },
        ]
    )

    summary = nbm_debug_bottlenecks.summarize_run(
        lead_workers=2,
        crop_method="auto",
        wgrib2_threads=None,
        crop_packing="same",
        wall_seconds=10.0,
        samples=[],
        manifest=manifest,
        output_dir=tmp_path,
    )

    assert summary.wgrib2_threads == 1


def test_run_pipeline_legacy_pause_control_stops_additional_cycle_submission(tmp_path, monkeypatch):
    cycle_a = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 5, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 0, tzinfo=build_grib2_features.NY_TZ),
        cycle="05",
    )
    cycle_b = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 6, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 1, tzinfo=build_grib2_features.NY_TZ),
        cycle="06",
    )
    monkeypatch.setattr(build_grib2_features, "discover_cycle_plans", lambda *_args, **_kwargs: [cycle_a, cycle_b])
    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(build_grib2_features, "cycle_already_complete", lambda *_args, **_kwargs: False)

    processed_cycles: list[str] = []
    written_cycles: list[str] = []
    pause_once = {"value": False}

    def fake_process_cycle(*, args, client, cycle_plan, phase_limits=None, reporter=None):
        processed_cycles.append(cycle_plan.cycle_token)
        if reporter is not None and not pause_once["value"]:
            pause_once["value"] = True
            reporter.request_pause(reason="operator")
        return [
            build_grib2_features.UnitResult(
                wide_row={},
                wide_rows=[],
                long_rows=[],
                provenance_rows=[],
                manifest_row={"extraction_status": "ok"},
            )
            for _ in build_grib2_features.LEAD_HOURS
        ]

    monkeypatch.setattr(build_grib2_features, "process_cycle", fake_process_cycle)
    monkeypatch.setattr(
        build_grib2_features,
        "write_cycle_outputs",
        lambda _output_dir, cycle_plan, _results, **_kwargs: written_cycles.append(cycle_plan.cycle_token),
    )

    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        lead_workers=1,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        selection_mode="all",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        progress_mode="log",
        disable_dashboard_hotkeys=False,
        pause_control_file=str(tmp_path / "pause.request"),
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
        range_merge_gap_bytes=0,
    )

    assert build_grib2_features.run_pipeline(args, client=FakeGrib2Client(payloads={})) == 0
    assert processed_cycles == [cycle_a.cycle_token]
    assert written_cycles == [cycle_a.cycle_token]


def test_write_cycle_outputs_tracks_paths_per_unit(tmp_path):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    unit_ok = build_grib2_features.UnitResult(
        wide_row={
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "fallback_used_any": False,
            "init_time_utc": "2026-01-01T15:00:00+00:00",
            "init_time_local": "2026-01-01T10:00:00-05:00",
            "init_date_local": "2026-01-01",
            "valid_time_utc": "2026-01-01T16:00:00+00:00",
            "valid_time_local": "2026-01-01T11:00:00-05:00",
            "valid_date_local": "2026-01-01",
            "forecast_hour": 1,
            "lead_hour": 1,
            "mode": "intraday",
            "station_id": "KLGA",
            "station_lat": 40.7769,
            "station_lon": -73.8740,
            "settlement_lat": 40.7769,
            "settlement_lon": -73.8740,
            "crop_top_lat": 43.5,
            "crop_bottom_lat": 39.0,
            "crop_left_lon": 282.5,
            "crop_right_lon": 289.5,
            "nearest_grid_lat": 40.7769,
            "nearest_grid_lon": -73.874,
            "tmp": 14.0,
        },
        wide_rows=[
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "fallback_used_any": False,
                "init_time_utc": "2026-01-01T15:00:00+00:00",
                "init_time_local": "2026-01-01T10:00:00-05:00",
                "init_date_local": "2026-01-01",
                "valid_time_utc": "2026-01-01T16:00:00+00:00",
                "valid_time_local": "2026-01-01T11:00:00-05:00",
                "valid_date_local": "2026-01-01",
                "forecast_hour": 1,
                "lead_hour": 1,
                "mode": "intraday",
                "station_id": "KLGA",
                "station_lat": 40.7769,
                "station_lon": -73.8740,
                "settlement_lat": 40.7769,
                "settlement_lon": -73.8740,
                "crop_top_lat": 43.5,
                "crop_bottom_lat": 39.0,
                "crop_left_lon": 282.5,
                "crop_right_lon": 289.5,
                "nearest_grid_lat": 40.7769,
                "nearest_grid_lon": -73.874,
                "tmp": 14.0,
            }
        ],
        long_rows=[],
        provenance_rows=[
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "init_time_utc": "2026-01-01T15:00:00+00:00",
                "init_time_local": "2026-01-01T10:00:00-05:00",
                "init_date_local": "2026-01-01",
                "valid_time_utc": "2026-01-01T16:00:00+00:00",
                "valid_time_local": "2026-01-01T11:00:00-05:00",
                "valid_date_local": "2026-01-01",
                "forecast_hour": 1,
                "lead_hour": 1,
                "mode": "intraday",
                "station_id": "KLGA",
                "station_lat": 40.7769,
                "station_lon": -73.8740,
                "settlement_lat": 40.7769,
                "settlement_lon": -73.8740,
                "crop_top_lat": 43.5,
                "crop_bottom_lat": 39.0,
                "crop_left_lon": 282.5,
                "crop_right_lon": 289.5,
                "nearest_grid_lat": 40.7769,
                "nearest_grid_lon": -73.874,
                "feature_name": "tmp",
            }
        ],
        manifest_row={
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "init_time_utc": "2026-01-01T15:00:00+00:00",
            "init_time_local": "2026-01-01T10:00:00-05:00",
            "init_date_local": "2026-01-01",
            "valid_time_utc": "2026-01-01T16:00:00+00:00",
            "valid_time_local": "2026-01-01T11:00:00-05:00",
            "valid_date_local": "2026-01-01",
            "forecast_hour": 1,
            "lead_hour": 1,
            "mode": "intraday",
            "reduced_file_path": str(tmp_path / "reduced.grib2"),
            "reduced_retained": False,
            "extraction_status": "ok",
        },
    )
    unit_error = build_grib2_features.UnitResult(
        wide_row=None,
        wide_rows=[],
        long_rows=[],
        provenance_rows=[],
        manifest_row={
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "init_time_utc": "2026-01-01T15:00:00+00:00",
            "init_time_local": "2026-01-01T10:00:00-05:00",
            "init_date_local": "2026-01-01",
            "valid_time_utc": "2026-01-01T17:00:00+00:00",
            "valid_time_local": "2026-01-01T12:00:00-05:00",
            "valid_date_local": "2026-01-01",
            "forecast_hour": 2,
            "lead_hour": 2,
            "mode": "intraday",
            "reduced_file_path": str(tmp_path / "missing.grib2"),
            "reduced_retained": False,
            "extraction_status": "error:missing_required_fields",
        },
    )
    (tmp_path / "reduced.grib2").write_bytes(b"ok")
    build_grib2_features.write_cycle_outputs(tmp_path, cycle_plan, [unit_ok, unit_error], write_long=False)
    manifest = pd.read_parquet(build_grib2_features.manifest_path(tmp_path, cycle_plan))
    ok_row = manifest.loc[manifest["lead_hour"] == 1].iloc[0]
    err_row = manifest.loc[manifest["lead_hour"] == 2].iloc[0]
    assert ok_row["wide_output_paths"]
    assert ok_row["long_output_paths"] == ""
    assert ok_row["provenance_output_paths"]
    assert err_row["wide_output_paths"] == ""
    assert err_row["long_output_paths"] == ""
    assert err_row["provenance_output_paths"] == ""


def test_nbm_monthly_build_raw_command_passes_batch_reduce_mode(tmp_path):
    args = argparse.Namespace(
        selection_mode="overnight_0005",
        progress_mode="log",
        workers=1,
        lead_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        crop_method="small_grib",
        wgrib2_threads=1,
        crop_grib_type="complex3",
        keep_reduced=False,
        overnight_fast=True,
        metric_profile="overnight",
        batch_reduce_mode="cycle",
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        pause_control_file=None,
        overwrite=False,
    )

    command = nbm_monthly_backfill.build_raw_command(
        args,
        target_date_local="2026-04-11",
        raw_dir=tmp_path / "raw",
        scratch_dir=tmp_path / "scratch",
    )

    assert command[command.index("--batch-reduce-mode") + 1] == "cycle"
    assert command[command.index("--metric-profile") + 1] == "overnight"


def test_write_cycle_outputs_can_skip_provenance_outputs(tmp_path):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    unit = build_grib2_features.UnitResult(
        wide_row={
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "init_time_utc": "2026-01-01T15:00:00+00:00",
            "init_time_local": "2026-01-01T10:00:00-05:00",
            "init_date_local": "2026-01-01",
            "valid_time_utc": "2026-01-01T16:00:00+00:00",
            "valid_time_local": "2026-01-01T11:00:00-05:00",
            "valid_date_local": "2026-01-01",
            "forecast_hour": 1,
            "lead_hour": 1,
            "mode": "intraday",
            "tmp": 14.0,
        },
        wide_rows=[
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "init_time_utc": "2026-01-01T15:00:00+00:00",
                "init_time_local": "2026-01-01T10:00:00-05:00",
                "init_date_local": "2026-01-01",
                "valid_time_utc": "2026-01-01T16:00:00+00:00",
                "valid_time_local": "2026-01-01T11:00:00-05:00",
                "valid_date_local": "2026-01-01",
                "forecast_hour": 1,
                "lead_hour": 1,
                "mode": "intraday",
                "tmp": 14.0,
            }
        ],
        long_rows=[],
        provenance_rows=[
            {
                "valid_date_local": "2026-01-01",
                "init_date_local": "2026-01-01",
                "mode": "intraday",
                "feature_name": "tmp",
            }
        ],
        manifest_row={
            "source_model": "NBM",
            "source_product": "grib2-core",
            "source_version": "nbm-grib2-core-public",
            "init_time_utc": "2026-01-01T15:00:00+00:00",
            "init_time_local": "2026-01-01T10:00:00-05:00",
            "init_date_local": "2026-01-01",
            "valid_time_utc": "2026-01-01T16:00:00+00:00",
            "valid_time_local": "2026-01-01T11:00:00-05:00",
            "valid_date_local": "2026-01-01",
            "forecast_hour": 1,
            "lead_hour": 1,
            "mode": "intraday",
            "reduced_file_path": str(tmp_path / "reduced.grib2"),
            "reduced_retained": False,
            "extraction_status": "ok",
            "provenance_written": True,
        },
    )

    build_grib2_features.write_cycle_outputs(
        tmp_path,
        cycle_plan,
        [unit],
        write_long=False,
        write_provenance=False,
    )

    manifest = pd.read_parquet(build_grib2_features.manifest_path(tmp_path, cycle_plan))
    row = manifest.iloc[0]
    assert row["wide_output_paths"]
    assert row["long_output_paths"] == ""
    assert row["provenance_output_paths"] == ""
    assert row["provenance_written"] == False
    assert not (tmp_path / "metadata" / "provenance").exists()


def test_process_unit_keep_flags_preserve_artifacts(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=True,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["raw_deleted"] is False
    assert result.manifest_row["idx_deleted"] is False
    assert result.manifest_row["reduced_deleted"] is False
    assert result.manifest_row["reduced_retained"] is True
    assert result.manifest_row["raw_file_size"] == 1300
    assert result.manifest_row["reduced_file_size"] == len(b"reduced")
    assert pathlib.Path(result.manifest_row["reduced_file_path"]).exists()


def test_process_unit_reuses_reduced_grib_from_scratch_dir(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    scratch_dir = tmp_path / "scratch"
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path / "outputs",
        scratch_dir=scratch_dir,
        overwrite=False,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=True,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    crop_calls: list[pathlib.Path] = []

    def fake_crop_selected_grib2(**kwargs):
        crop_calls.append(kwargs["reduced_path"])
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    first = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)
    second = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    reduced_path = pathlib.Path(first.manifest_row["reduced_file_path"])
    assert first.manifest_row["reduced_reused"] is False
    assert second.manifest_row["reduced_reused"] is True
    assert second.manifest_row["crop_method_used"] == "reused"
    assert second.manifest_row["crop_grid_cache_key"] is None
    assert second.manifest_row["crop_grid_cache_hit"] is False
    assert second.manifest_row["scratch_dir"] == str(scratch_dir)
    assert second.manifest_row["cfgrib_index_strategy"] == "persistent_cache_per_reduced_signature"
    assert reduced_path.exists()
    assert build_grib2_features.reduced_signature_path(reduced_path).exists()
    assert len(crop_calls) == 1
    assert str(reduced_path).startswith(str(scratch_dir / "reduced"))
    assert (scratch_dir / "raw" / key).exists()


def test_nbm_reduced_reuse_signature_changes_with_selected_records_hash_or_size(tmp_path):
    raw_path = tmp_path / "raw.grib2"
    raw_path.write_bytes(b"a" * 8)
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    crop_bounds = build_grib2_features.CropBounds(top=43.5, bottom=39.0, left=282.5, right=289.5)
    first = build_grib2_features.build_reduced_reuse_signature(
        raw_path=raw_path,
        cycle_plan=cycle_plan,
        lead_hour=1,
        crop_bounds=crop_bounds,
        region="co",
        idx_sha256="idx-a",
        selected_records_hash="sel-a",
    )
    second = build_grib2_features.build_reduced_reuse_signature(
        raw_path=raw_path,
        cycle_plan=cycle_plan,
        lead_hour=1,
        crop_bounds=crop_bounds,
        region="co",
        idx_sha256="idx-a",
        selected_records_hash="sel-b",
    )
    raw_path.write_bytes(b"b" * 9)
    third = build_grib2_features.build_reduced_reuse_signature(
        raw_path=raw_path,
        cycle_plan=cycle_plan,
        lead_hour=1,
        crop_bounds=crop_bounds,
        region="co",
        idx_sha256="idx-a",
        selected_records_hash="sel-b",
    )
    assert first != second
    assert second != third


def test_nbm_process_unit_without_keep_reduced_skips_reuse_signature(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=False,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    build_calls = {"count": 0}

    def fake_build_reduced_reuse_signature(**_kwargs):
        build_calls["count"] += 1
        return "sig"

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "build_reduced_reuse_signature", fake_build_reduced_reuse_signature)
    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["extraction_status"] == "ok"
    assert build_calls["count"] == 0
    assert result.manifest_row["reduced_reuse_signature"] is None


def test_summarize_nbm_diagnostics_reports_timings_and_size_ratios(tmp_path, capsys):
    manifest_path = tmp_path / "cycle_20230101T0000Z_manifest.parquet"
    pd.DataFrame(
        [
            {
                "timing_idx_fetch_seconds": 1.0,
                "timing_idx_parse_seconds": 2.0,
                "timing_head_seconds": 0.1,
                "timing_range_download_seconds": 3.0,
                "timing_crop_seconds": 4.0,
                "timing_cfgrib_open_seconds": 5.0,
                "timing_row_build_seconds": 6.0,
                "timing_cleanup_seconds": 0.5,
                "selected_download_bytes": 80.0,
                "downloaded_range_bytes": 100.0,
                "raw_file_size": 200.0,
            },
            {
                "timing_idx_fetch_seconds": 3.0,
                "timing_idx_parse_seconds": 4.0,
                "timing_head_seconds": 0.2,
                "timing_range_download_seconds": 9.0,
                "timing_crop_seconds": 8.0,
                "timing_cfgrib_open_seconds": 7.0,
                "timing_row_build_seconds": 12.0,
                "timing_cleanup_seconds": 0.7,
                "selected_download_bytes": 100.0,
                "downloaded_range_bytes": 120.0,
                "raw_file_size": 400.0,
            },
        ]
    ).to_parquet(manifest_path, index=False)

    monkey_args = argparse.Namespace(path=tmp_path, glob="*.parquet")
    original_parse_args = nbm_diagnostics_summary.parse_args
    try:
        nbm_diagnostics_summary.parse_args = lambda: monkey_args
        assert nbm_diagnostics_summary.main() == 0
    finally:
        nbm_diagnostics_summary.parse_args = original_parse_args

    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_count"] == 1
    assert payload["metrics"]["timing_crop_seconds"]["median"] == 6.0
    assert payload["size_ratio"]["median_downloaded_to_raw"] == 0.4


def test_summarize_nbm_diagnostics_ignores_non_raw_manifest_parquet(tmp_path):
    raw_manifest = tmp_path / "cycle_20230101T0000Z_manifest.parquet"
    overnight_manifest = tmp_path / "nbm.overnight.manifest.parquet"
    pd.DataFrame([{"timing_crop_seconds": 1.0, "raw_file_size": 10.0, "selected_download_bytes": 5.0, "downloaded_range_bytes": 6.0}]).to_parquet(raw_manifest, index=False)
    pd.DataFrame([{"status": "ok"}]).to_parquet(overnight_manifest, index=False)

    discovered = nbm_diagnostics_summary.discover_manifest_paths(tmp_path, "*.parquet")

    assert discovered == [raw_manifest]


def test_process_unit_overwrite_disables_reduced_reuse(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    scratch_dir = tmp_path / "scratch"
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path / "outputs",
        scratch_dir=scratch_dir,
        overwrite=False,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=True,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    crop_calls: list[pathlib.Path] = []

    def fake_crop_selected_grib2(**kwargs):
        crop_calls.append(kwargs["reduced_path"])
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    first = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)
    args.overwrite = True
    second = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert first.manifest_row["reduced_reused"] is False
    assert second.manifest_row["reduced_reused"] is False
    assert len(crop_calls) == 2


def test_process_unit_range_download_failure_deletes_partial_raw(tmp_path):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")

    class FailingRangeClient(FakeGrib2Client):
        def download_byte_ranges(self, *, url: str, ranges, destination: pathlib.Path, overwrite: bool, progress_label: str | None = None):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"partial")
            raise RuntimeError("range failed")

    client = FailingRangeClient(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    assert result.manifest_row["extraction_status"] == "error:RuntimeError"
    assert result.manifest_row["raw_deleted"] is True
    assert result.manifest_row["idx_deleted"] is True
    assert not raw_path.exists()
    assert not idx_path.exists()


def test_process_unit_reuses_cached_raw_only_when_idx_already_existed(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=False,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    raw_path = tmp_path / "raw" / key
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(b"z" * 1300)

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["extraction_status"] == "ok"
    assert client.range_downloads
    assert raw_path.read_bytes() != b"z" * 1300


def test_process_unit_reuses_valid_cached_raw_when_idx_exists(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=False,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"q" * 1300
    raw_path.write_bytes(payload)
    idx_path.write_text(idx_text)
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})

    def fake_crop_selected_grib2(**kwargs):
        kwargs["reduced_path"].write_bytes(b"reduced")
        return "crop cmd"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda path, **kwargs: [
            make_grib_dataset("t2m", "TMP", np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=float)),
            make_grib_dataset("wind10m", "WIND", np.array([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=float)),
            make_grib_dataset("wdir10m", "WDIR", np.array([[180, 180, 180], [180, 270, 180], [180, 180, 180]], dtype=float)),
        ],
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    assert result.manifest_row["extraction_status"] == "ok"
    assert client.range_downloads == []
    assert raw_path.read_bytes() == payload


def test_process_unit_keep_downloads_preserves_raw_on_crop_failure(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=True,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    monkeypatch.setattr(
        build_grib2_features,
        "crop_selected_grib2",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crop failed")),
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    assert result.manifest_row["extraction_status"] == "error:RuntimeError"
    assert result.manifest_row["raw_deleted"] is False
    assert result.manifest_row["idx_deleted"] is False
    assert raw_path.exists()
    assert idx_path.exists()


def test_process_unit_without_keep_downloads_deletes_raw_on_crop_failure(tmp_path, monkeypatch):
    idx_text = make_required_idx_text()
    key = fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), "15", 1, "co")
    client = FakeGrib2Client(payloads={key: make_raw_grib_payload(), f"{key}.idx": idx_text})
    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
    )
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )

    monkeypatch.setattr(
        build_grib2_features,
        "crop_selected_grib2",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("crop failed")),
    )

    result = build_grib2_features.process_unit(args=args, client=client, cycle_plan=cycle_plan, lead_hour=1)

    raw_path = tmp_path / "raw" / key
    idx_path = tmp_path / "raw" / f"{key}.idx"
    assert result.manifest_row["extraction_status"] == "error:RuntimeError"
    assert result.manifest_row["raw_deleted"] is True
    assert result.manifest_row["idx_deleted"] is True
    assert not raw_path.exists()
    assert not idx_path.exists()


def test_fetch_range_rejects_non_partial_content_status(monkeypatch):
    client = fetch_nbm.S3HttpClient(pool_maxsize=2)

    class FakeResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            return None

    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: FakeResponse(200, b"abc"))

    with pytest.raises(RuntimeError, match="Expected HTTP 206"):
        client.fetch_range("https://example.com/file.grib2", start=0, end=2)


def test_fetch_range_rejects_short_payload(monkeypatch):
    client = fetch_nbm.S3HttpClient(pool_maxsize=2)

    class FakeResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content

        def raise_for_status(self):
            return None

    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: FakeResponse(206, b"ab"))

    with pytest.raises(RuntimeError, match="returned 2 bytes, expected 3"):
        client.fetch_range("https://example.com/file.grib2", start=0, end=2)


def test_cycle_already_complete_does_not_require_deleted_reduced_files(tmp_path):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    out_path = build_grib2_features.partition_file_path(
        root=tmp_path / "features" / "wide",
        cycle_plan=cycle_plan,
        suffix="wide",
        valid_date_local="2026-01-01",
        init_date_local="2026-01-01",
        mode="intraday",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"tmp": 14.0}]).to_parquet(out_path, index=False)
    prov_path = build_grib2_features.partition_file_path(
        root=tmp_path / "metadata" / "provenance",
        cycle_plan=cycle_plan,
        suffix="provenance",
        valid_date_local="2026-01-01",
        init_date_local="2026-01-01",
        mode="intraday",
    )
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"feature_name": "tmp"}]).to_parquet(prov_path, index=False)
    manifest_path = build_grib2_features.manifest_path(tmp_path, cycle_plan)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "lead_hour": 1,
                "extraction_status": "ok",
                "reduced_file_path": str(tmp_path / "deleted.grib2"),
                "reduced_retained": False,
                "wide_output_paths": str(out_path),
                "long_output_paths": "",
                "provenance_output_paths": str(prov_path),
            }
        ]
    ).to_parquet(manifest_path, index=False)
    assert build_grib2_features.cycle_already_complete(tmp_path, cycle_plan) is True


def test_run_pipeline_dry_run_keeps_all_36_leads(tmp_path, capsys):
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": ["blend.20260101/15/"],
        }
    )
    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        overwrite=False,
        dry_run=True,
    )
    assert build_grib2_features.run_pipeline(args, client=client) == 0
    captured = capsys.readouterr()
    assert "leads=f001-f036" in captured.out
    assert "planned_units=36" in captured.out


def test_run_pipeline_dry_run_overnight_mode_reports_selected_target_date(tmp_path, capsys):
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": [f"blend.20260101/{hour:02d}/" for hour in (23,)],
            "blend.20260102/": [f"blend.20260102/{hour:02d}/" for hour in (0, 2, 6)],
        }
    )
    args = argparse.Namespace(
        start_local_date="2026-01-02",
        end_local_date="2026-01-02",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        overwrite=False,
        dry_run=True,
        selection_mode="overnight_0005",
    )
    assert build_grib2_features.run_pipeline(args, client=client) == 0
    captured = capsys.readouterr()
    assert "target_date_local=2026-01-02" in captured.out
    assert "selection_mode=overnight_0005" in captured.out
    assert "mode=all" not in captured.out
    assert "leads=f003-f026" in captured.out
    assert "planned_units=24" in captured.out


def test_run_pipeline_staged_admits_only_selected_target_day_leads(tmp_path, monkeypatch):
    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(build_grib2_features, "LEAD_HOURS", [1, 2, 26])
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 2, 4, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 23, tzinfo=build_grib2_features.NY_TZ),
        cycle="04",
        selected_target_dates=(dt.date(2026, 1, 2),),
    )
    monkeypatch.setattr(build_grib2_features, "discover_cycle_plans", lambda *_args, **_kwargs: [cycle_plan])
    monkeypatch.setattr(build_grib2_features, "select_inventory_records", lambda _records: ([SimpleNamespace(records=[1])], [], []))
    monkeypatch.setattr(build_grib2_features, "selected_records_identity_hash", lambda _selected_fields: "selected")
    monkeypatch.setattr(
        build_grib2_features,
        "build_selected_ranges",
        lambda *_args, **_kwargs: [SimpleNamespace(byte_start=0, byte_end=9, byte_length=10)],
    )
    monkeypatch.setattr(build_grib2_features, "merge_selected_ranges", lambda ranges, **_kwargs: ranges)
    monkeypatch.setattr(build_grib2_features, "selected_ranges_require_content_length", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(build_grib2_features, "should_reuse_cached_raw_grib", lambda **_kwargs: False)

    def fake_crop_selected_grib2(*, raw_path, reduced_path, **_kwargs):
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_bytes(raw_path.read_bytes())
        return "crop"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda _path, *, index_dir, **_kwargs: [SimpleNamespace(close=lambda: None)],
    )
    monkeypatch.setattr(
        build_grib2_features,
        "build_rows_from_datasets",
        lambda **_kwargs: (
            [
                {
                    "valid_time_utc": "2026-01-02T05:00:00+00:00",
                    "valid_time_local": "2026-01-02T00:00:00-05:00",
                    "valid_date_local": "2026-01-02",
                    "fallback_used_any": False,
                }
            ],
            [],
            [],
        ),
    )

    written_leads: list[int] = []

    def fake_write_cycle_outputs(_output_dir, _cycle_plan, results, **_kwargs):
        written_leads.extend(result.manifest_row["lead_hour"] for result in results)

    monkeypatch.setattr(build_grib2_features, "write_cycle_outputs", fake_write_cycle_outputs)
    payloads = {}
    for lead_hour in [1, 2, 26]:
        key = build_grib2_features.resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, lead_hour, "co")
        payloads[key] = make_raw_grib_payload(32)
        payloads[f"{key}.idx"] = make_required_idx_text()
    client = FakeGrib2Client(payloads=payloads)
    args = argparse.Namespace(
        start_local_date="2026-01-02",
        end_local_date="2026-01-02",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        lead_workers=1,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        selection_mode="overnight_0005",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
    )

    assert build_grib2_features.run_pipeline(args, client=client) == 0

    assert written_leads == [1, 2]
    assert [
        pathlib.Path(key).name
        for key, _ranges, _progress_label in client.range_downloads
    ] == [
        pathlib.Path(build_grib2_features.resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, 1, "co")).name,
        pathlib.Path(build_grib2_features.resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, 2, "co")).name,
    ]


def test_run_pipeline_batch_reduce_mode_splits_rows_back_to_leads(tmp_path, monkeypatch):
    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(build_grib2_features, "LEAD_HOURS", [1, 2])
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 2, 4, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 23, tzinfo=build_grib2_features.NY_TZ),
        cycle="04",
        selected_target_dates=(dt.date(2026, 1, 2),),
    )
    monkeypatch.setattr(build_grib2_features, "discover_cycle_plans", lambda *_args, **_kwargs: [cycle_plan])
    selected_field = build_grib2_features.SelectedField(
        spec=build_grib2_features.FEATURE_SPEC_BY_NAME["tmp"],
        records=[SimpleNamespace(raw_line="selected-tmp")],
    )
    monkeypatch.setattr(build_grib2_features, "select_inventory_records", lambda _records: ([selected_field], [], []))
    monkeypatch.setattr(build_grib2_features, "selected_records_identity_hash", lambda _selected_fields: "selected")
    monkeypatch.setattr(
        build_grib2_features,
        "build_selected_ranges",
        lambda *_args, **_kwargs: [SimpleNamespace(byte_start=0, byte_end=9, byte_length=10)],
    )
    monkeypatch.setattr(build_grib2_features, "merge_selected_ranges", lambda ranges, **_kwargs: ranges)
    monkeypatch.setattr(build_grib2_features, "selected_ranges_require_content_length", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(build_grib2_features, "should_reuse_cached_raw_grib", lambda **_kwargs: False)

    crop_calls: list[pathlib.Path] = []

    def fake_crop_selected_grib2(*, raw_path, reduced_path, **_kwargs):
        crop_calls.append(raw_path)
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_bytes(raw_path.read_bytes())
        return build_grib2_features.CropExecutionResult(
            command="batch-crop",
            method_used="small_grib",
            crop_grid_cache_key=None,
            crop_grid_cache_hit=False,
            crop_ij_box=None,
            crop_wgrib2_threads=1,
        )

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)

    def fake_open_grouped_datasets(_path, *, index_dir, **_kwargs):
        values = np.zeros((2, 2, 2), dtype=float)
        data_array = xr.DataArray(
            values,
            dims=("step", "y", "x"),
            coords={
                "step": ("step", np.array([1.0, 2.0])),
                "valid_time": ("step", np.array(["2026-01-02T05:00:00", "2026-01-02T06:00:00"], dtype="datetime64[ns]")),
                "latitude": (("y", "x"), np.array([[41.0, 41.0], [40.0, 40.0]])),
                "longitude": (("y", "x"), np.array([[286.0, 287.0], [286.0, 287.0]])),
            },
            attrs={"GRIB_shortName": "TMP", "GRIB_typeOfLevel": "heightAboveGround", "GRIB_stepType": "instant"},
        )
        return [xr.Dataset({"t2m": data_array})]

    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", fake_open_grouped_datasets)

    observed_dataset_counts: list[int] = []

    def fake_build_rows_from_datasets(*, datasets, **_kwargs):
        observed_dataset_counts.append(len(datasets))
        for dataset in datasets:
            data_array = next(iter(dataset.data_vars.values()))
            assert data_array.ndim == 2
        return (
            [
                {
                    "valid_time_utc": "2026-01-02T05:00:00+00:00",
                    "valid_time_local": "2026-01-02T00:00:00-05:00",
                    "valid_date_local": "2026-01-02",
                    "forecast_hour": 1,
                    "lead_hour": 1,
                    "fallback_used_any": False,
                },
                {
                    "valid_time_utc": "2026-01-02T06:00:00+00:00",
                    "valid_time_local": "2026-01-02T01:00:00-05:00",
                    "valid_date_local": "2026-01-02",
                    "forecast_hour": 2,
                    "lead_hour": 2,
                    "fallback_used_any": False,
                },
            ],
            [],
            [],
        )

    monkeypatch.setattr(build_grib2_features, "build_rows_from_datasets", fake_build_rows_from_datasets)

    written_results: list[build_grib2_features.UnitResult] = []

    def fake_write_cycle_outputs(_output_dir, _cycle_plan, results, **_kwargs):
        written_results.extend(results)

    monkeypatch.setattr(build_grib2_features, "write_cycle_outputs", fake_write_cycle_outputs)
    payloads = {}
    for lead_hour in [1, 2]:
        key = build_grib2_features.resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, lead_hour, "co")
        payloads[key] = make_raw_grib_payload(32)
        payloads[f"{key}.idx"] = make_required_idx_text()
    client = FakeGrib2Client(payloads=payloads)
    args = argparse.Namespace(
        start_local_date="2026-01-02",
        end_local_date="2026-01-02",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        scratch_dir=tmp_path / "scratch",
        workers=1,
        lead_workers=1,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        selection_mode="overnight_0005",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        skip_provenance=True,
        metric_profile="overnight",
        crop_method="small_grib",
        crop_grib_type="complex3",
        wgrib2_threads=1,
        batch_reduce_mode="cycle",
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
    )

    assert build_grib2_features.run_pipeline(args, client=client) == 0

    assert len(crop_calls) == 1
    assert observed_dataset_counts == [2]
    assert [result.manifest_row["lead_hour"] for result in written_results] == [1, 2]
    assert [len(result.wide_rows) for result in written_results] == [1, 1]
    assert all(result.manifest_row["batch_reduce_mode"] == "cycle" for result in written_results)
    assert all(result.manifest_row["reduced_file_path"] == result.manifest_row["batch_reduced_file_path"] for result in written_results)
    assert sum(result.manifest_row["batch_crop_seconds"] for result in written_results) >= 0.0
    assert all(result.manifest_row["raw_deleted"] is True for result in written_results)


def test_run_pipeline_creates_default_cycle_progress(tmp_path, monkeypatch):
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": ["blend.20260101/15/"],
        }
    )
    reporter_instances: list[object] = []

    class RecordingReporter:
        def __init__(self, title, *, unit="task", total=None, mode="auto", stream=None, is_tty=None, **_kwargs):
            self.title = title
            self.unit = unit
            self.total = total
            self.mode = "log"
            self.metrics = []
            self.groups = []
            self.events = []
            self.closed = False
            reporter_instances.append(self)

        def set_metrics(self, **metrics):
            self.metrics.append(metrics)

        def upsert_group(self, group_id, **kwargs):
            self.groups.append((group_id, kwargs))

        def log_event(self, message, *, level="info"):
            self.events.append((level, message))

        def close(self, *, status=None):
            self.closed = True
            self.events.append(("close", status))

    monkeypatch.setattr(build_grib2_features, "create_progress_reporter", RecordingReporter)
    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(
        build_grib2_features,
        "process_cycle",
        lambda *, args, client, cycle_plan, phase_limits=None, reporter=None: [
            build_grib2_features.UnitResult(
                wide_row=None,
                wide_rows=[],
                long_rows=[],
                provenance_rows=[],
                manifest_row={"extraction_status": "ok", "lead_hour": 1},
            )
        ],
    )
    monkeypatch.setattr(build_grib2_features, "write_cycle_outputs", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        lead_workers=1,
        selection_mode="all",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        progress_mode="log",
    )

    assert build_grib2_features.run_pipeline(args, client=client) == 0
    assert len(reporter_instances) == 1
    reporter = reporter_instances[0]
    assert reporter.title == "NBM build"
    assert reporter.unit == "lead"
    assert reporter.total == len(build_grib2_features.LEAD_HOURS)
    assert reporter.closed is True
    assert any(metrics.get("cycles_total") == 1 for metrics in reporter.metrics)
    assert any(group_id == "20260101T1500Z" for group_id, _ in reporter.groups)


def test_run_pipeline_shares_phase_limits_across_cycles(tmp_path, monkeypatch):
    client = FakeGrib2Client(
        cycles_by_prefix={
            "blend.20260101/": ["blend.20260101/05/", "blend.20260101/15/"],
        }
    )
    phase_limit_ids: list[int] = []

    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)

    def fake_process_cycle(*, args, client, cycle_plan, phase_limits=None, reporter=None):
        phase_limit_ids.append(id(phase_limits))
        return [
            build_grib2_features.UnitResult(
                wide_row=None,
                wide_rows=[],
                long_rows=[],
                provenance_rows=[],
                manifest_row={"extraction_status": "ok", "lead_hour": 1},
            )
        ]

    monkeypatch.setattr(build_grib2_features, "process_cycle", fake_process_cycle)
    monkeypatch.setattr(build_grib2_features, "write_cycle_outputs", lambda *args, **kwargs: None)

    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=2,
        lead_workers=3,
        download_workers=None,
        reduce_workers=2,
        extract_workers=1,
        selection_mode="all",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        progress_mode="log",
    )

    assert build_grib2_features.run_pipeline(args, client=client) == 0
    assert len(phase_limit_ids) == 2
    assert len(set(phase_limit_ids)) == 1


def test_run_pipeline_pause_stops_new_cycle_activation_and_drains_active_cycle(tmp_path, monkeypatch):
    monkeypatch.setattr(build_grib2_features, "LEAD_HOURS", [1])
    monkeypatch.setattr(build_grib2_features, "ensure_runtime_dependencies", lambda: None)
    monkeypatch.setattr(build_grib2_features, "select_inventory_records", lambda _records: ([SimpleNamespace(records=[1])], [], []))
    monkeypatch.setattr(
        build_grib2_features,
        "build_selected_ranges",
        lambda *_args, **_kwargs: [SimpleNamespace(byte_start=0, byte_end=9, byte_length=10)],
    )
    monkeypatch.setattr(build_grib2_features, "merge_selected_ranges", lambda ranges, **_kwargs: ranges)
    monkeypatch.setattr(build_grib2_features, "should_reuse_cached_raw_grib", lambda **_kwargs: False)
    monkeypatch.setattr(build_grib2_features, "reduced_grib_reusable", lambda **_kwargs: False)

    def fake_crop_selected_grib2(*, raw_path, reduced_path, **_kwargs):
        reduced_path.parent.mkdir(parents=True, exist_ok=True)
        reduced_path.write_bytes(raw_path.read_bytes())
        return "crop"

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(
        build_grib2_features,
        "open_grouped_datasets",
        lambda _path, *, index_dir: [SimpleNamespace(close=lambda: None)],
    )
    monkeypatch.setattr(
        build_grib2_features,
        "build_rows_from_datasets",
        lambda **_kwargs: (
            [
                {
                    "valid_time_utc": "2026-01-01T16:00:00+00:00",
                    "valid_time_local": "2026-01-01T11:00:00-05:00",
                    "valid_date_local": "2026-01-01",
                    "fallback_used_any": False,
                }
            ],
            [],
            [],
        ),
    )

    written_cycles: list[str] = []
    monkeypatch.setattr(
        build_grib2_features,
        "write_cycle_outputs",
        lambda _output_dir, cycle_plan, _results, **_kwargs: written_cycles.append(cycle_plan.cycle_token),
    )

    cycle_a = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 5, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 0, tzinfo=build_grib2_features.NY_TZ),
        cycle="05",
    )
    cycle_b = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 6, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 1, tzinfo=build_grib2_features.NY_TZ),
        cycle="06",
    )
    monkeypatch.setattr(build_grib2_features, "discover_cycle_plans", lambda *_args, **_kwargs: [cycle_a, cycle_b])

    payloads = {}
    for cycle_plan in [cycle_a, cycle_b]:
        key = build_grib2_features.resolve_grib2_key(cycle_plan.init_time_utc.date(), cycle_plan.cycle, 1, "co")
        payloads[key] = make_raw_grib_payload(32)
        payloads[f"{key}.idx"] = make_required_idx_text()
    client = FakeGrib2Client(payloads=payloads)

    reporter_instances = []

    def auto_pause_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs)
        original_start_worker = reporter.start_worker
        triggered = {"value": False}

        def start_worker(*start_args, **start_kwargs):
            original_start_worker(*start_args, **start_kwargs)
            if not triggered["value"]:
                triggered["value"] = True
                reporter.request_pause(reason="operator")

        reporter.start_worker = start_worker  # type: ignore[method-assign]
        reporter_instances.append(reporter)
        return reporter

    monkeypatch.setattr(build_grib2_features, "create_progress_reporter", auto_pause_reporter)

    args = argparse.Namespace(
        start_local_date="2026-01-01",
        end_local_date="2026-01-01",
        region="co",
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path,
        workers=1,
        lead_workers=1,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        selection_mode="all",
        overwrite=True,
        dry_run=False,
        keep_downloads=False,
        keep_reduced=False,
        write_long=False,
        progress_mode="log",
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
        disable_dashboard_hotkeys=False,
    )

    assert build_grib2_features.run_pipeline(args, client=client) == 0
    assert written_cycles == [cycle_a.cycle_token]
    assert reporter_instances[0].is_paused() is True


def test_process_cycle_creates_per_cycle_lead_progress(tmp_path, monkeypatch):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    client = FakeGrib2Client()
    class RecordingReporter:
        mode = "log"

        def __init__(self):
            self.groups = []

        def upsert_group(self, group_id, **kwargs):
            self.groups.append((group_id, kwargs))

        def log_event(self, *args, **kwargs):
            return None

        def start_worker(self, *args, **kwargs):
            return None

        def update_worker(self, *args, **kwargs):
            return None

        def complete_worker(self, *args, **kwargs):
            return None

    reporter = RecordingReporter()
    monkeypatch.setattr(
        build_grib2_features,
        "process_unit",
        lambda *, args, client, cycle_plan, lead_hour, phase_limits=None, progress=None, reporter=None: build_grib2_features.UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row={"extraction_status": "ok", "lead_hour": lead_hour},
        ),
    )

    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
        lead_workers=1,
    )

    results = build_grib2_features.process_cycle(args=args, client=client, cycle_plan=cycle_plan, reporter=reporter)

    assert len(results) == len(build_grib2_features.LEAD_HOURS)
    assert any(group_id == cycle_plan.cycle_token for group_id, _ in reporter.groups)


def test_process_cycle_non_tty_uses_sparse_lead_checkpoints(tmp_path, monkeypatch, capsys):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    client = FakeGrib2Client()

    monkeypatch.setattr(
        build_grib2_features,
        "process_unit",
        lambda *, args, client, cycle_plan, lead_hour, phase_limits=None, progress=None, reporter=None: build_grib2_features.UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row={"extraction_status": "ok", "lead_hour": lead_hour},
        ),
    )

    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
        lead_workers=1,
    )

    results = build_grib2_features.process_cycle(args=args, client=client, cycle_plan=cycle_plan, reporter=None)

    assert len(results) == len(build_grib2_features.LEAD_HOURS)
    captured = capsys.readouterr()
    assert "[progress] NBM" not in captured.out
    assert captured.out.count("lead_checkpoint") == 6
    assert "completed=36/36" in captured.out


def test_build_phase_concurrency_limits_defaults_and_clamps():
    args = argparse.Namespace(download_workers=None, reduce_workers=None, extract_workers=None)
    limits = build_grib2_features.build_phase_concurrency_limits(lead_workers=6, args=args)
    assert limits.download_workers == 6
    assert limits.reduce_workers == 4
    assert limits.extract_workers == 4

    explicit = argparse.Namespace(download_workers=10, reduce_workers=0, extract_workers=3)
    limits = build_grib2_features.build_phase_concurrency_limits(lead_workers=5, args=explicit)
    assert limits.download_workers == 5
    assert limits.reduce_workers == 1
    assert limits.extract_workers == 3


def test_process_unit_phase_caps_limit_download_reduce_and_extract(tmp_path, monkeypatch):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 5, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 0, tzinfo=build_grib2_features.NY_TZ),
        cycle="05",
    )
    args = argparse.Namespace(
        top=43.5,
        bottom=39.0,
        left=282.5,
        right=289.5,
        output_dir=tmp_path / "out",
        scratch_dir=tmp_path / "scratch",
        region="co",
        overwrite=True,
        keep_downloads=False,
        keep_reduced=False,
        range_merge_gap_bytes=4096,
        max_task_attempts=1,
        retry_backoff_seconds=0.0,
        retry_max_backoff_seconds=0.0,
    )
    phase_limits = build_grib2_features.build_phase_concurrency_limits(
        lead_workers=3,
        args=argparse.Namespace(download_workers=2, reduce_workers=1, extract_workers=1),
    )
    inventory_record = build_grib2_features.InventoryRecord(
        record_id="1",
        offset=0,
        init_time_utc=cycle_plan.init_time_utc,
        short_name="TMP",
        level_text="2 m above ground",
        step_text="1 hour fcst",
        extra_text="",
        raw_line="1:0:d=2026010105:TMP:2 m above ground:1 hour fcst:",
    )
    client = FakeGrib2Client(
        payloads={
            f"{fetch_nbm.resolve_grib2_key(dt.date(2026, 1, 1), '05', lead_hour, 'co')}.idx": inventory_record.raw_line
            for lead_hour in (1, 2, 3)
        }
    )
    selected_range = build_grib2_features.SelectedRange(record=inventory_record, byte_start=0, byte_end=9)
    merged_range = build_grib2_features.MergedByteRange(byte_start=0, byte_end=9, record_count=1)

    phase_counts = defaultdict(int)
    phase_peaks = defaultdict(int)
    phase_lock = threading.Lock()

    def run_in_phase(phase_name: str, fn):
        with phase_lock:
            phase_counts[phase_name] += 1
            phase_peaks[phase_name] = max(phase_peaks[phase_name], phase_counts[phase_name])
        try:
            time.sleep(0.05)
            return fn()
        finally:
            with phase_lock:
                phase_counts[phase_name] -= 1

    monkeypatch.setattr(build_grib2_features, "parse_idx_lines", lambda _text: [inventory_record])
    monkeypatch.setattr(
        build_grib2_features,
        "select_inventory_records",
        lambda _records: ([SimpleNamespace(records=[inventory_record])], [], []),
    )
    monkeypatch.setattr(build_grib2_features, "selected_ranges_require_content_length", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(build_grib2_features, "build_selected_ranges", lambda *_args, **_kwargs: [selected_range])
    monkeypatch.setattr(build_grib2_features, "merge_selected_ranges", lambda *_args, **_kwargs: [merged_range])
    monkeypatch.setattr(build_grib2_features, "build_reduced_reuse_signature", lambda **_kwargs: "sig")
    monkeypatch.setattr(build_grib2_features, "reduced_grib_reusable", lambda **_kwargs: False)

    def fake_download_byte_ranges(*, destination: pathlib.Path, **_kwargs):
        def inner():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"0123456789")
            return destination
        return run_in_phase("download", inner)

    client.download_byte_ranges = fake_download_byte_ranges

    def fake_crop_selected_grib2(*, reduced_path: pathlib.Path, **_kwargs):
        def inner():
            reduced_path.parent.mkdir(parents=True, exist_ok=True)
            reduced_path.write_bytes(b"crop")
            return "crop_selected_grib2"
        return run_in_phase("reduce", inner)

    def fake_open_grouped_datasets(_reduced_path, *, index_dir, **_kwargs):
        def inner():
            index_dir.mkdir(parents=True, exist_ok=True)
            return [xr.Dataset()]
        return run_in_phase("extract", inner)

    monkeypatch.setattr(build_grib2_features, "crop_selected_grib2", fake_crop_selected_grib2)
    monkeypatch.setattr(build_grib2_features, "open_grouped_datasets", fake_open_grouped_datasets)
    monkeypatch.setattr(
        build_grib2_features,
        "build_rows_from_datasets",
        lambda **_kwargs: (
            [{"valid_time_utc": "2026-01-01T06:00:00+00:00", "valid_time_local": "2026-01-01T01:00:00-05:00", "valid_date_local": "2026-01-01", "fallback_used_any": False}],
            [],
            [],
        ),
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(
                build_grib2_features.process_unit,
                args=args,
                client=client,
                cycle_plan=cycle_plan,
                lead_hour=lead_hour,
                phase_limits=phase_limits,
                reporter=None,
            )
            for lead_hour in (1, 2, 3)
        ]
        results = [future.result() for future in futures]

    assert all(result.manifest_row["extraction_status"] == "ok" for result in results)
    assert phase_peaks["download"] <= 2
    assert phase_peaks["reduce"] <= 1
    assert phase_peaks["extract"] <= 1


def test_open_grouped_datasets_uses_filtered_fallback_for_uncovered_groups(tmp_path, monkeypatch):
    dataset_surface = make_grib_dataset("t2m", "TMP", np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float))
    dataset_surface["t2m"].attrs["GRIB_typeOfLevel"] = "surface"
    dataset_surface["t2m"].attrs["GRIB_stepType"] = "instant"
    dataset_hag = make_grib_dataset("wind10m", "WIND", np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=float))
    dataset_hag["wind10m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_hag["wind10m"].attrs["GRIB_stepType"] = "instant"
    dataset_hag["wind10m"].attrs["GRIB_level"] = 10

    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [dataset_surface])

    def fake_open_dataset(_path, engine, backend_kwargs):
        filter_keys = backend_kwargs["filter_by_keys"]
        if filter_keys == {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 10}:
            return dataset_hag
        raise RuntimeError("missing")

    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", fake_open_dataset)

    datasets = build_grib2_features.open_grouped_datasets(tmp_path / "reduced.grib2", index_dir=tmp_path / "idx")

    signatures = {
        tuple(sorted(dataset.data_vars))
        for dataset in datasets
    }
    assert ("t2m",) in signatures
    assert ("wind10m",) in signatures


def test_open_grouped_datasets_skips_irrelevant_filtered_fallback_groups(tmp_path, monkeypatch):
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("wind", "WIND", ("10 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "WIND", "10 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset_surface = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    dataset_hag = make_grib_dataset("wind10m", "WIND", np.ones((3, 3), dtype=float))
    dataset_hag["wind10m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_hag["wind10m"].attrs["GRIB_level"] = 10

    attempted_filters = []
    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [dataset_surface])

    def fake_open_dataset(_path, engine, backend_kwargs):
        attempted_filters.append(backend_kwargs["filter_by_keys"])
        if backend_kwargs["filter_by_keys"] == {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 10}:
            return dataset_hag
        raise AssertionError(f"unexpected fallback probe: {backend_kwargs['filter_by_keys']}")

    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", fake_open_dataset)

    stats = {}
    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=selected,
        stats=stats,
    )

    assert attempted_filters == [{"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 10}]
    assert {tuple(sorted(dataset.data_vars)) for dataset in datasets} == {("t2m",), ("wind10m",)}
    assert stats["cfgrib_open_all_dataset_count"] == 1
    assert stats["cfgrib_filtered_fallback_attempt_count"] == 1
    assert stats["cfgrib_filtered_fallback_open_count"] == 1
    assert stats["cfgrib_opened_dataset_count"] == 2


def test_open_grouped_datasets_fallback_runs_when_group_missing_selected_feature(tmp_path, monkeypatch):
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "DPT", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset_tmp = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    dataset_tmp["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_tmp["t2m"].attrs["GRIB_stepType"] = "instant"
    dataset_tmp["t2m"].attrs["GRIB_level"] = 2
    dataset_dpt = make_grib_dataset("d2m", "DPT", np.full((3, 3), 2.0, dtype=float))
    dataset_dpt["d2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_dpt["d2m"].attrs["GRIB_stepType"] = "instant"
    dataset_dpt["d2m"].attrs["GRIB_level"] = 2

    attempted_filters = []
    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [dataset_tmp])

    def fake_open_dataset(_path, engine, backend_kwargs):
        attempted_filters.append(backend_kwargs["filter_by_keys"])
        if backend_kwargs["filter_by_keys"] == {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2}:
            return dataset_dpt
        raise AssertionError(f"unexpected fallback probe: {backend_kwargs['filter_by_keys']}")

    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", fake_open_dataset)

    stats = {}
    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=selected,
        stats=stats,
    )

    assert attempted_filters == [{"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2}]
    assert {tuple(sorted(dataset.data_vars)) for dataset in datasets} == {("t2m",), ("d2m",)}
    assert stats["cfgrib_filtered_fallback_attempt_count"] == 1
    assert stats["cfgrib_filtered_fallback_open_count"] == 1


def test_open_grouped_datasets_fallback_runs_when_observed_feature_has_wrong_level(tmp_path, monkeypatch):
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("dpt", "DPT", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "DPT", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset_mixed = xr.Dataset(
        {
            "t2m": make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))["t2m"],
            "d2m": make_grib_dataset("d2m", "DPT", np.full((3, 3), 2.0, dtype=float))["d2m"],
        }
    )
    dataset_mixed["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_mixed["t2m"].attrs["GRIB_stepType"] = "instant"
    dataset_mixed["t2m"].attrs["GRIB_level"] = 2
    dataset_mixed["d2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_mixed["d2m"].attrs["GRIB_stepType"] = "instant"
    dataset_mixed["d2m"].attrs["GRIB_level"] = 10
    dataset_dpt_level_2 = make_grib_dataset("d2m", "DPT", np.full((3, 3), 3.0, dtype=float))
    dataset_dpt_level_2["d2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_dpt_level_2["d2m"].attrs["GRIB_stepType"] = "instant"
    dataset_dpt_level_2["d2m"].attrs["GRIB_level"] = 2

    attempted_filters = []
    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [dataset_mixed])

    def fake_open_dataset(_path, engine, backend_kwargs):
        attempted_filters.append(backend_kwargs["filter_by_keys"])
        if backend_kwargs["filter_by_keys"] == {"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2}:
            return dataset_dpt_level_2
        raise AssertionError(f"unexpected fallback probe: {backend_kwargs['filter_by_keys']}")

    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", fake_open_dataset)

    stats = {}
    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=selected,
        stats=stats,
    )

    assert attempted_filters == [{"typeOfLevel": "heightAboveGround", "stepType": "instant", "level": 2}]
    assert {tuple(sorted(dataset.data_vars)) for dataset in datasets} == {("d2m", "t2m"), ("d2m",)}
    assert stats["cfgrib_filtered_fallback_attempt_count"] == 1
    assert stats["cfgrib_filtered_fallback_open_count"] == 1


def test_open_grouped_datasets_skips_fallback_when_required_groups_are_covered(tmp_path, monkeypatch):
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    dataset_hag = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    dataset_hag["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_hag["t2m"].attrs["GRIB_level"] = 2

    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [dataset_hag])
    monkeypatch.setattr(
        build_grib2_features.xr,
        "open_dataset",
        lambda *_args, **_kwargs: pytest.fail("filtered fallback should not be called"),
    )

    stats = {}
    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=selected,
        stats=stats,
    )

    assert datasets == [dataset_hag]
    assert stats["cfgrib_open_all_dataset_count"] == 1
    assert stats["cfgrib_filtered_fallback_attempt_count"] == 0
    assert stats["cfgrib_filtered_fallback_open_count"] == 0
    assert stats["cfgrib_opened_dataset_count"] == 1


def test_open_grouped_datasets_stats_count_all_open_results_before_dedup(tmp_path, monkeypatch):
    dataset_primary = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    dataset_primary["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_primary["t2m"].attrs["GRIB_stepType"] = "instant"
    dataset_primary["t2m"].attrs["GRIB_level"] = 2
    dataset_duplicate = make_grib_dataset("t2m", "TMP", np.full((3, 3), 2.0, dtype=float))
    dataset_duplicate["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    dataset_duplicate["t2m"].attrs["GRIB_stepType"] = "instant"
    dataset_duplicate["t2m"].attrs["GRIB_level"] = 2
    dataset_empty = xr.Dataset()

    monkeypatch.setattr(
        build_grib2_features.cfgrib,
        "open_datasets",
        lambda *_args, **_kwargs: [dataset_primary, dataset_duplicate, dataset_empty],
    )
    monkeypatch.setattr(
        build_grib2_features.xr,
        "open_dataset",
        lambda *_args, **_kwargs: pytest.fail("filtered fallback should not be called"),
    )

    stats = {}
    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=[
            build_grib2_features.SelectedField(
                spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
                records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
            )
        ],
        stats=stats,
    )

    assert datasets == [dataset_primary]
    assert stats["cfgrib_open_all_dataset_count"] == 3
    assert stats["cfgrib_opened_dataset_count"] == 1
    assert stats["cfgrib_filtered_fallback_attempt_count"] == 0


def test_open_grouped_datasets_signature_keeps_same_var_with_different_step_or_level(tmp_path, monkeypatch):
    instant = make_grib_dataset("apcp", "APCP", np.ones((3, 3), dtype=float))
    instant["apcp"].attrs["GRIB_typeOfLevel"] = "surface"
    instant["apcp"].attrs["GRIB_stepType"] = "instant"
    accum = make_grib_dataset("apcp", "APCP", np.full((3, 3), 2.0, dtype=float))
    accum["apcp"].attrs["GRIB_typeOfLevel"] = "surface"
    accum["apcp"].attrs["GRIB_stepType"] = "accum"

    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [])

    def fake_open_dataset(_path, engine, backend_kwargs):
        filter_keys = backend_kwargs["filter_by_keys"]
        if filter_keys == {"typeOfLevel": "surface", "stepType": "instant"}:
            return instant
        if filter_keys == {"typeOfLevel": "surface", "stepType": "accum"}:
            return accum
        raise RuntimeError("missing")

    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", fake_open_dataset)

    datasets = build_grib2_features.open_grouped_datasets(tmp_path / "reduced.grib2", index_dir=tmp_path / "idx")

    assert [dataset["apcp"].attrs["GRIB_stepType"] for dataset in datasets] == ["instant", "accum"]


def test_open_grouped_datasets_signature_keeps_same_var_with_different_level(tmp_path, monkeypatch):
    selected = [
        build_grib2_features.SelectedField(
            spec=build_grib2_features.FieldSpec("tmp", "TMP", ("2 m above ground",), "wide"),
            records=[build_grib2_features.parse_idx_lines(make_inventory_line(1, "TMP", "2 m above ground", "1 hour fcst"))[0]],
        )
    ]
    level_2 = make_grib_dataset("t2m", "TMP", np.ones((3, 3), dtype=float))
    level_2["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    level_2["t2m"].attrs["GRIB_stepType"] = "instant"
    level_2["t2m"].attrs["GRIB_level"] = 2
    level_10 = make_grib_dataset("t2m", "TMP", np.full((3, 3), 2.0, dtype=float))
    level_10["t2m"].attrs["GRIB_typeOfLevel"] = "heightAboveGround"
    level_10["t2m"].attrs["GRIB_stepType"] = "instant"
    level_10["t2m"].attrs["GRIB_level"] = 10

    monkeypatch.setattr(build_grib2_features.cfgrib, "open_datasets", lambda *_args, **_kwargs: [level_2, level_10])
    monkeypatch.setattr(build_grib2_features.xr, "open_dataset", lambda *_args, **_kwargs: pytest.fail("fallback should not be called"))

    datasets = build_grib2_features.open_grouped_datasets(
        tmp_path / "reduced.grib2",
        index_dir=tmp_path / "idx",
        selected=selected,
    )

    assert [dataset["t2m"].attrs["GRIB_level"] for dataset in datasets] == [2, 10]


def test_process_cycle_parallel_lead_workers_preserves_lead_order(tmp_path, monkeypatch):
    cycle_plan = build_grib2_features.CyclePlan(
        init_time_utc=dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc),
        init_time_local=dt.datetime(2026, 1, 1, 10, tzinfo=build_grib2_features.NY_TZ),
        cycle="15",
    )
    client = FakeGrib2Client()

    monkeypatch.setattr(build_grib2_features, "stdout_is_tty", lambda: False)

    def fake_process_unit(*, args, client, cycle_plan, lead_hour, phase_limits=None, progress=None, reporter=None):
        return build_grib2_features.UnitResult(
            wide_row=None,
            wide_rows=[],
            long_rows=[],
            provenance_rows=[],
            manifest_row={"extraction_status": "ok", "lead_hour": lead_hour},
        )

    monkeypatch.setattr(build_grib2_features, "process_unit", fake_process_unit)

    args = argparse.Namespace(
        region="co",
        output_dir=tmp_path,
        overwrite=True,
        left=282.5,
        right=289.5,
        bottom=39.0,
        top=43.5,
        keep_downloads=False,
        keep_reduced=False,
        lead_workers=4,
        workers=1,
    )

    results = build_grib2_features.process_cycle(args=args, client=client, cycle_plan=cycle_plan, reporter=None)

    assert [result.manifest_row["lead_hour"] for result in results] == build_grib2_features.LEAD_HOURS


def test_nbm_monthly_backfill_iter_days():
    windows = nbm_monthly_backfill.iter_days("2024-02-28", "2024-03-01")

    assert [window.target_date_local.isoformat() for window in windows] == [
        "2024-02-28",
        "2024-02-29",
        "2024-03-01",
    ]


def test_nbm_monthly_backfill_runs_daily_commands_skips_valid_day_and_cleans_tmp(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    removed: list[pathlib.Path] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill.shutil, "rmtree", lambda path, ignore_errors=True: removed.append(path))

    valid_root = tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-01"
    valid_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01"}]).to_parquet(valid_root / "nbm.overnight.parquet", index=False)
    pd.DataFrame([{"status": "ok"}]).to_parquet(valid_root / "nbm.overnight.manifest.parquet", index=False)
    (valid_root / "nbm.resume.json").write_text(json.dumps({"target_date_local": "2024-01-01", "selection_mode": "overnight_0005"}))

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-02",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=6,
        reduce_workers=2,
        extract_workers=2,
        reduce_queue_size=5,
        extract_queue_size=7,
        crop_method="small_grib",
        wgrib2_threads=1,
        crop_grib_type="complex3",
        progress_mode="log",
        pause_control_file="/tmp/nbm.pause",
        max_task_attempts=5,
        retry_backoff_seconds=1.5,
        retry_max_backoff_seconds=9.0,
        overnight_fast=False,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert len(commands) == 1
    assert commands[0][1] == "tools/nbm/build_grib2_features.py"
    assert "--reduce-queue-size" in commands[0]
    assert "--extract-queue-size" in commands[0]
    assert "--max-task-attempts" in commands[0]
    assert "--retry-backoff-seconds" in commands[0]
    assert "--retry-max-backoff-seconds" in commands[0]
    assert commands[0][commands[0].index("--crop-method") + 1] == "small_grib"
    assert commands[0][commands[0].index("--wgrib2-threads") + 1] == "1"
    assert commands[0][commands[0].index("--crop-grib-type") + 1] == "complex3"
    assert "--skip-provenance" not in commands[0]
    assert "--pause-control-file" not in commands[0]
    assert (tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-02" / "nbm.overnight.parquet").exists()
    assert (tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-02" / "nbm.overnight.manifest.parquet").exists()
    performance = json.loads(
        (tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-02" / "nbm.performance.json").read_text()
    )
    assert performance["token"] == "2024-01-02"
    assert "full_day_seconds" in performance
    assert json.loads(
        (tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-02" / "nbm.resume.json").read_text()
    ) == {"selection_mode": "overnight_0005", "target_date_local": "2024-01-02"}
    assert removed == []


def test_nbm_monthly_backfill_rebuilds_day_when_selection_mode_changes(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)

    output_root = tmp_path / "runtime" / "nbm_overnight" / "target_date_local=2024-01-01"
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"target_date_local": "2024-01-01"}]).to_parquet(output_root / "nbm.overnight.parquet", index=False)
    pd.DataFrame([{"status": "ok"}]).to_parquet(output_root / "nbm.overnight.manifest.parquet", index=False)
    (output_root / "nbm.resume.json").write_text(json.dumps({"target_date_local": "2024-01-01", "selection_mode": "all"}))

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overnight_fast=False,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert [command[1] for command in commands] == ["tools/nbm/build_grib2_features.py"]


def test_nbm_monthly_backfill_resolves_relative_run_root_under_repo(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    relative_run_root = pathlib.Path("data/runtime/test_relative_nbm_review")
    absolute_run_root = nbm_monthly_backfill.REPO_ROOT / relative_run_root

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        assert raw_dir.is_absolute()
        assert overnight_root.is_absolute()
        assert raw_dir == absolute_run_root / "nbm_tmp" / target_date_local / "raw"
        assert overnight_root == absolute_run_root / "nbm_overnight"
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=relative_run_root,
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=True,
        keep_temp_on_failure=False,
    )

    try:
        assert nbm_monthly_backfill.run_backfill(args) == 0
        assert args.run_root == absolute_run_root
        assert pathlib.Path(commands[0][commands[0].index("--output-dir") + 1]) == absolute_run_root / "nbm_tmp" / "2024-01-01" / "raw"
        assert (absolute_run_root / "nbm_overnight" / "target_date_local=2024-01-01" / "nbm.resume.json").exists()
    finally:
        nbm_monthly_backfill.delete_path(absolute_run_root)


def test_nbm_monthly_backfill_in_process_failure_does_not_write_resume_and_obeys_temp_policy(tmp_path, monkeypatch):
    def fake_run(command, cwd, check):
        tmp_root = tmp_path / "runtime" / "nbm_tmp" / "2024-01-01"
        tmp_root.mkdir(parents=True, exist_ok=True)
        (tmp_root / "debug.txt").write_text("keep me when requested")
        return SimpleNamespace(returncode=0)

    def fail_build_overnight_in_process(**_kwargs):
        raise RuntimeError("overnight failed")

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fail_build_overnight_in_process)

    base_args = dict(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
    )

    args_delete = argparse.Namespace(**base_args, keep_temp_on_failure=False)
    with pytest.raises(RuntimeError, match="overnight failed"):
        nbm_monthly_backfill.run_backfill(args_delete)
    day = nbm_monthly_backfill.DayWindow(dt.date(2024, 1, 1))
    assert not nbm_monthly_backfill.nbm_day_metadata_path(args_delete.run_root, day).exists()
    assert not nbm_monthly_backfill.day_temp_root(args_delete.run_root, "2024-01-01").exists()

    args_keep = argparse.Namespace(**base_args, keep_temp_on_failure=True)
    with pytest.raises(RuntimeError, match="overnight failed"):
        nbm_monthly_backfill.run_backfill(args_keep)
    assert not nbm_monthly_backfill.nbm_day_metadata_path(args_keep.run_root, day).exists()
    assert (nbm_monthly_backfill.day_temp_root(args_keep.run_root, "2024-01-01") / "debug.txt").exists()


def test_nbm_monthly_backfill_can_use_overnight_subprocess_fallback(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        if command[1] == "tools/nbm/build_nbm_overnight_features.py":
            output_dir = pathlib.Path(command[command.index("--output-dir") + 1])
            target_date_local = command[command.index("--start-local-date") + 1]
            root = output_dir / f"target_date_local={target_date_local}"
            root.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(root / "nbm.overnight.parquet", index=False)
            pd.DataFrame([{"status": "ok"}]).to_parquet(root / "nbm.overnight.manifest.parquet", index=False)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(
        nbm_monthly_backfill,
        "build_overnight_in_process",
        lambda **_kwargs: pytest.fail("in-process overnight finalizer should not run"),
    )

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=True,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=True,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert [command[1] for command in commands] == [
        "tools/nbm/build_grib2_features.py",
        "tools/nbm/build_nbm_overnight_features.py",
    ]


def test_nbm_monthly_backfill_overnight_fast_passes_skip_provenance(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert "--skip-provenance" in commands[0]


def test_nbm_monthly_backfill_day_workers_can_run_two_days_concurrently(tmp_path, monkeypatch):
    active = 0
    max_active = 0
    active_lock = threading.Lock()
    overlap_seen = threading.Event()

    def fake_run(command, cwd, check):
        nonlocal active, max_active
        assert cwd == ROOT
        assert check is True
        if command[1] == "tools/nbm/build_grib2_features.py":
            with active_lock:
                active += 1
                max_active = max(max_active, active)
                if active >= 2:
                    overlap_seen.set()
            overlap_seen.wait(timeout=1.0)
            time.sleep(0.02)
            with active_lock:
                active -= 1
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-02",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert max_active == 2
    assert not nbm_monthly_backfill.day_temp_root(args.run_root, "2024-01-01").exists()
    assert not nbm_monthly_backfill.day_temp_root(args.run_root, "2024-01-02").exists()


def test_nbm_monthly_backfill_min_free_gb_blocks_extra_parallel_admission(tmp_path, monkeypatch):
    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def fake_run(command, cwd, check):
        nonlocal active, max_active
        assert cwd == ROOT
        assert check is True
        if command[1] == "tools/nbm/build_grib2_features.py":
            with active_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with active_lock:
                active -= 1
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    disk_values = iter([100.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: next(disk_values, 0.0))

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-02",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=5.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert max_active == 1


def test_nbm_monthly_backfill_min_free_gb_blocks_first_admission(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise AssertionError("raw builder should not start under low-disk admission block")

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 0.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=1,
        min_free_gb=5.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert not (args.run_root / "nbm_overnight" / "target_date_local=2024-01-01").exists()


def test_nbm_monthly_backfill_pause_control_file_drains_active_day_and_stops_new_admission(tmp_path, monkeypatch):
    started_tokens: list[str] = []
    pause_file = tmp_path / "nbm.pause"

    def fake_run(command, cwd, check):
        assert cwd == ROOT
        assert check is True
        if command[1] == "tools/nbm/build_grib2_features.py":
            token = command[command.index("--start-local-date") + 1]
            started_tokens.append(token)
            if token == "2024-01-01":
                pause_file.write_text("pause")
            time.sleep(0.05)
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    disk_values = iter([100.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: next(disk_values, 0.0))

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-03",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=5.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=str(pause_file),
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert started_tokens == ["2024-01-01"]
    assert (args.run_root / "nbm_overnight" / "target_date_local=2024-01-01" / "nbm.resume.json").exists()
    assert not (args.run_root / "nbm_overnight" / "target_date_local=2024-01-02").exists()


def test_nbm_monthly_backfill_parallel_auto_progress_is_forced_to_log(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="auto",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    progress_mode_index = commands[0].index("--progress-mode")
    assert commands[0][progress_mode_index + 1] == "log"


def test_nbm_monthly_backfill_dashboard_mode_supports_multi_day_parent_tui(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    reporters = []

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        commands.append(command)
        if command[1] == "tools/nbm/build_grib2_features.py":
            stdout_handler("[progress] NBM build event=worker_start worker=download_1 label=20240101T0500Z f019 phase=download details=byte_range_download")
            stdout_handler("[progress] NBM build event=worker_update worker=download_1 phase=download details=byte_range_download")
            stdout_handler("[progress] NBM build event=transfer_start worker=download_1 file=blend.t05z.core.f019.co.grib2 total_bytes=1024")
            stdout_handler("[progress] NBM build event=transfer_progress worker=download_1 file=blend.t05z.core.f019.co.grib2 bytes_downloaded=1024 total_bytes=1024 instant_bps=2048 average_bps=1024")
            stdout_handler("[progress] NBM build event=transfer_complete worker=download_1 file=blend.t05z.core.f019.co.grib2 bytes_downloaded=1024 total_bytes=1024")
            stdout_handler("[progress] NBM build event=worker_retire worker=download_1 label=20240101T0500Z f019")
            stdout_handler("child plain stdout")
            stderr_handler("child plain stderr")

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "stream_command", fake_stream_command)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=True,
        pause_control_file=str(tmp_path / "runner.pause"),
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert reporters[0].mode == "dashboard"
    progress_mode_index = commands[0].index("--progress-mode")
    assert commands[0][progress_mode_index + 1] == "log"
    assert "--pause-control-file" not in commands[0]
    worker = reporters[0].state.workers["2024-01-01/download_1"]
    assert worker.transfer is not None
    assert worker.transfer.bytes_downloaded == 1024
    assert reporters[0].state.groups["2024-01-01"].status == "complete"
    recent_messages = [event.message for event in reporters[0].state.recent_events]
    assert any("2024-01-01 stdout: child plain stdout" == message for message in recent_messages)
    assert any("2024-01-01 stderr: child plain stderr" == message for message in recent_messages)


def test_nbm_monthly_backfill_dashboard_disables_hotkeys_only_on_parent(tmp_path, monkeypatch):
    captured_kwargs = {}

    def fake_create_progress_reporter(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        if stdout_handler is not None:
            stdout_handler("[progress] NBM build event=worker_update worker=download_1 phase=download details=byte_range_download")

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "stream_command", fake_stream_command)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert captured_kwargs["enable_dashboard_hotkeys"] is False


def test_nbm_monthly_backfill_dashboard_overnight_subprocess_output_is_relayed(tmp_path, monkeypatch):
    reporters = []
    commands: list[list[str]] = []

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        commands.append(command)
        if command[1] == "tools/nbm/build_nbm_overnight_features.py":
            stdout_handler("overnight emitted path")
            stderr_handler("overnight warning")
        elif stdout_handler is not None:
            stdout_handler("[progress] NBM build event=worker_update worker=download_1 phase=crop details=crop_selected_grib2")

    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "stream_command", fake_stream_command)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    def fake_validate(*_args, **_kwargs):
        return True

    monkeypatch.setattr(nbm_monthly_backfill, "validate_nbm_day", fake_validate)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=True,
        overnight_fast=False,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=False,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=True,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert [command[1] for command in commands] == [
        "tools/nbm/build_grib2_features.py",
        "tools/nbm/build_nbm_overnight_features.py",
    ]
    recent_messages = [event.message for event in reporters[0].state.recent_events]
    assert any("2024-01-01 overnight stdout: overnight emitted path" == message for message in recent_messages)
    assert any("2024-01-01 overnight stderr: overnight warning" == message for message in recent_messages)


def test_nbm_monthly_backfill_dashboard_downgrades_futurewarning_stderr(tmp_path, monkeypatch):
    reporters = []

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        if command[1] == "tools/nbm/build_grib2_features.py":
            stderr_handler("/site-packages/cfgrib/xarray_store.py:51: FutureWarning: In a future version of xarray ...")
            stderr_handler("  o = xr.merge([o, ds], **kwargs)")

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "stream_command", fake_stream_command)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=2,
        min_free_gb=1.0,
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=False,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    events = list(reporters[0].state.recent_events)
    assert any(event.level == "warn" and "FutureWarning" in event.message for event in events)
    assert any(event.level == "warn" and "xr.merge" in event.message for event in events)


def test_nbm_monthly_backfill_dashboard_hides_noisy_cfgrib_grouping_stderr(tmp_path, monkeypatch):
    reporters = []

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    def fake_stream_command(command, *, stdout_handler=None, stderr_handler=None):
        if command[1] == "tools/nbm/build_grib2_features.py":
            stderr_handler('    dict_merge(variables, coord_vars)')
            stderr_handler('  File "/home/zeus/miniconda3/envs/cloudspace/lib/python3.12/site-packages/cfgrib/dataset.py", line 642, in dict_merge')
            stderr_handler("    raise DatasetBuildError(")
            stderr_handler("cfgrib.dataset.DatasetBuildError: key present and new value is different: key='step' value=Variable(dimensions=('step',), data=array([1., 2., 3.])) new_value=Variable(dimensions=(), data=np.float64(19.0))")

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "stream_command", fake_stream_command)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2023-01-07",
        end_local_date="2023-01-07",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        batch_reduce_mode="cycle",
        day_workers=4,
        min_free_gb=1.0,
        workers=1,
        lead_workers=4,
        download_workers=4,
        reduce_workers=1,
        extract_workers=1,
        reduce_queue_size=2,
        extract_queue_size=2,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        metric_profile="overnight",
        progress_mode="dashboard",
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
        smart_workers=False,
        cpu_cores=None,
        adaptive_workers=False,
        adaptive_sample_days=2,
        adaptive_cooldown_days=2,
        adaptive_min_day_workers=None,
        adaptive_max_day_workers=None,
        adaptive_min_reduce_workers=None,
        adaptive_max_reduce_workers=None,
        adaptive_min_extract_workers=None,
        adaptive_max_extract_workers=None,
        adaptive_min_lead_workers=None,
        adaptive_max_lead_workers=None,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    events = list(reporters[0].state.recent_events)
    assert events
    assert all(event.level == "info" for event in events if "cfgrib" in event.message or "DatasetBuildError" in event.message or "dict_merge" in event.message)
    footer = reporters[0].renderer._footer_rows(reporters[0].state, width=140, height=4)
    assert not any("DatasetBuildError" in row or "dict_merge" in row for row in footer)


def test_nbm_monthly_relay_child_progress_line_preserves_retry_delay(tmp_path):
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=1,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
    )
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    bridge = nbm_monthly_backfill.DayProgressBridge(reporter, day_workers=2, run_root=run_root)
    bridge.admit_day("2024-01-01")
    bridge.start_child_worker(
        "2024-01-01",
        child_worker="download_1",
        label="20240101T0500Z:f001",
        phase="download",
        details="byte_range_download",
    )

    before = time.perf_counter()
    nbm_monthly_backfill.relay_child_progress_line(
        bridge,
        "2024-01-01",
        "[progress] NBM build event=retry_scheduled worker=download_1 attempt=2 max_attempts=5 delay_seconds=12.5 message=temporary error_class=timeout",
        stream_name="stdout",
    )
    worker = reporter.state.workers["2024-01-01/download_1"]
    reporter.close(status="done")

    assert worker.phase == "retry_wait"
    assert worker.retry_scheduled_at is not None
    assert worker.retry_scheduled_at >= before + 10.0


def test_nbm_monthly_batch_bridge_maps_child_phases_to_day_pipeline(tmp_path):
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=1,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
        dashboard_kind="nbm_batch",
    )
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    bridge = nbm_monthly_backfill.DayProgressBridge(reporter, day_workers=4, run_root=run_root)
    bridge.admit_day("2026-04-01")

    nbm_monthly_backfill.relay_child_progress_line(
        bridge,
        "2026-04-01",
        "[progress] NBM build event=group group=20260401T0500Z label=20260401T0500Z completed=0 total=23 failed=0 status=submitted",
        stream_name="stdout",
    )
    nbm_monthly_backfill.relay_child_progress_line(
        bridge,
        "2026-04-01",
        "[progress] NBM build event=transfer_complete worker=download_1 file=blend.t05z.core.f019.co.grib2 bytes_downloaded=1024 total_bytes=1024",
        stream_name="stdout",
    )
    nbm_monthly_backfill.relay_child_progress_line(
        bridge,
        "2026-04-01",
        "[progress] NBM build event=worker_start worker=reduce_1 label=20260401T0500Z batch phase=reduce details=batch_crop leads=23",
        stream_name="stdout",
    )
    nbm_monthly_backfill.relay_child_progress_line(
        bridge,
        "2026-04-01",
        "[progress] NBM build event=worker_update worker=extract_1 label=20260401T0500Z batch phase=extract details=batch_build_rows",
        stream_name="stdout",
    )

    day = reporter.state.batch_days["2026-04-01"]
    reporter.close(status="done")

    assert day.selected_issue == "20260401T0500Z"
    assert day.expected_leads == 23
    assert day.downloaded_leads == 1
    assert day.batch_status == "done"
    assert day.extract_status == "rows"
    assert day.lifecycle_phase == "batch_extract"


def test_nbm_monthly_batch_bridge_counts_downloaded_leads_not_workers(tmp_path):
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=1,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
        dashboard_kind="nbm_batch",
    )
    run_root = tmp_path / "runtime"
    run_root.mkdir(parents=True, exist_ok=True)
    bridge = nbm_monthly_backfill.DayProgressBridge(reporter, day_workers=4, run_root=run_root)
    bridge.admit_day("2026-04-01")
    bridge.set_day_expected_leads("2026-04-01", issue="20260401T0500Z", expected_leads=3)

    for lead in (19, 20, 21):
        nbm_monthly_backfill.relay_child_progress_line(
            bridge,
            "2026-04-01",
            f"[progress] NBM build event=worker_start worker=download_1 label=20260401T0500Z f{lead:03d} phase=download details=byte_range_download",
            stream_name="stdout",
        )
        nbm_monthly_backfill.relay_child_progress_line(
            bridge,
            "2026-04-01",
            f"[progress] NBM build event=transfer_complete worker=download_1 file=blend.t05z.core.f{lead:03d}.co.grib2 bytes_downloaded=1024 total_bytes=1024",
            stream_name="stdout",
        )
        nbm_monthly_backfill.relay_child_progress_line(
            bridge,
            "2026-04-01",
            f"[progress] NBM build event=worker_retire worker=download_1 label=20260401T0500Z f{lead:03d}",
            stream_name="stdout",
        )

    day = reporter.state.batch_days["2026-04-01"]
    reporter.close(status="done")

    assert day.downloaded_leads == 3


def test_nbm_monthly_backfill_uses_batch_dashboard_only_for_cycle_mode(tmp_path, monkeypatch):
    captured_kinds: list[str] = []

    def fake_create_progress_reporter(*args, **kwargs):
        captured_kinds.append(kwargs.get("dashboard_kind", "generic"))
        return progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)

    def fake_process_day(args, day, *, bridge=None, worker_plan=None):
        token = day.target_date_local.isoformat()
        if bridge is not None:
            bridge.complete_day(token, performance=_adaptive_day_performance(token))
        return nbm_monthly_backfill.DayRunResult(token=token, performance=_adaptive_day_performance(token))

    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "process_day", fake_process_day)
    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2026-04-01",
        end_local_date="2026-04-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        batch_reduce_mode="cycle",
        day_workers=4,
        smart_workers=False,
        cpu_cores=None,
        adaptive_workers=False,
        adaptive_sample_days=2,
        adaptive_cooldown_days=2,
        adaptive_min_day_workers=None,
        adaptive_max_day_workers=None,
        adaptive_min_reduce_workers=None,
        adaptive_max_reduce_workers=None,
        adaptive_min_extract_workers=None,
        adaptive_max_extract_workers=None,
        adaptive_min_lead_workers=None,
        adaptive_max_lead_workers=None,
        min_free_gb=1.0,
        workers=1,
        lead_workers=4,
        download_workers=4,
        reduce_workers=1,
        extract_workers=1,
        reduce_queue_size=2,
        extract_queue_size=2,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        metric_profile="overnight",
        progress_mode="dashboard",
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert captured_kinds == ["nbm_batch"]


def test_nbm_monthly_backfill_resume_skips_do_not_affect_eta(tmp_path, monkeypatch):
    reporters = []

    def fake_create_progress_reporter(*args, **kwargs):
        reporter = progress_mod.create_progress_reporter(*args, **kwargs, stream=io.StringIO(), is_tty=True)
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(nbm_monthly_backfill, "create_progress_reporter", fake_create_progress_reporter)
    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "dashboard")
    monkeypatch.setattr(nbm_monthly_backfill, "validate_nbm_day", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2026-04-01",
        end_local_date="2026-04-02",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        batch_reduce_mode="cycle",
        day_workers=4,
        smart_workers=False,
        cpu_cores=None,
        adaptive_workers=False,
        adaptive_sample_days=2,
        adaptive_cooldown_days=2,
        adaptive_min_day_workers=None,
        adaptive_max_day_workers=None,
        adaptive_min_reduce_workers=None,
        adaptive_max_reduce_workers=None,
        adaptive_min_extract_workers=None,
        adaptive_max_extract_workers=None,
        adaptive_min_lead_workers=None,
        adaptive_max_lead_workers=None,
        min_free_gb=1.0,
        workers=1,
        lead_workers=4,
        download_workers=4,
        reduce_workers=1,
        extract_workers=1,
        reduce_queue_size=2,
        extract_queue_size=2,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        metric_profile="overnight",
        progress_mode="dashboard",
        disable_dashboard_hotkeys=True,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert len(reporters) == 1
    assert reporters[0].state.skipped == 2
    assert reporters[0].state.eta_completed == 0


def test_nbm_monthly_parse_args_accepts_adaptive_flags(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_nbm_monthly_backfill.py",
            "--start-local-date",
            "2024-01-01",
            "--end-local-date",
            "2024-01-02",
            "--adaptive-workers",
            "--adaptive-sample-days",
            "3",
            "--adaptive-cooldown-days",
            "4",
            "--adaptive-min-day-workers",
            "1",
            "--adaptive-max-day-workers",
            "4",
            "--adaptive-min-reduce-workers",
            "2",
            "--adaptive-max-reduce-workers",
            "6",
            "--adaptive-min-extract-workers",
            "2",
            "--adaptive-max-extract-workers",
            "6",
            "--adaptive-min-lead-workers",
            "4",
            "--adaptive-max-lead-workers",
            "12",
            "--crop-method",
            "small_grib",
            "--wgrib2-threads",
            "1",
            "--crop-grib-type",
            "complex3",
        ],
    )
    args = nbm_monthly_backfill.parse_args()

    assert args.adaptive_workers is True
    assert args.adaptive_sample_days == 3
    assert args.adaptive_cooldown_days == 4
    assert args.adaptive_max_day_workers == 4
    assert args.adaptive_max_lead_workers == 12
    assert args.crop_method == "small_grib"
    assert args.wgrib2_threads == 1
    assert args.crop_grib_type == "complex3"


def _adaptive_day_performance(
    token: str,
    *,
    cleanup_seconds: float = 5.0,
    full_day_seconds: float = 100.0,
    crop_median: float = 4.0,
    crop_p95: float = 4.5,
    cfgrib_open_median: float = 1.0,
    cfgrib_open_p95: float = 1.1,
    row_metric_median: float = 1.0,
    row_metric_p95: float = 1.1,
    row_provenance_median: float = 0.2,
    row_provenance_p95: float = 0.25,
) -> nbm_monthly_backfill.DayPerformance:
    return nbm_monthly_backfill.DayPerformance(
        token=token,
        lead_count_completed=20,
        raw_build_seconds=80.0,
        cleanup_seconds=cleanup_seconds,
        full_day_seconds=full_day_seconds,
        timing_crop_seconds_median=crop_median,
        timing_crop_seconds_p95=crop_p95,
        timing_cfgrib_open_seconds_median=cfgrib_open_median,
        timing_cfgrib_open_seconds_p95=cfgrib_open_p95,
        timing_row_metric_seconds_median=row_metric_median,
        timing_row_metric_seconds_p95=row_metric_p95,
        timing_row_provenance_seconds_median=row_provenance_median,
        timing_row_provenance_seconds_p95=row_provenance_p95,
        cfgrib_open_all_dataset_count_mean=3.0,
        cfgrib_filtered_fallback_attempt_count_mean=1.0,
        wide_row_count_mean=25.0,
        long_row_count_mean=5.0,
        provenance_row_count_mean=0.0,
    )


def _adaptive_args(**overrides):
    values = dict(
        adaptive_workers=True,
        adaptive_sample_days=2,
        adaptive_cooldown_days=2,
        adaptive_min_day_workers=None,
        adaptive_max_day_workers=None,
        adaptive_min_reduce_workers=None,
        adaptive_max_reduce_workers=None,
        adaptive_min_extract_workers=None,
        adaptive_max_extract_workers=None,
        adaptive_min_lead_workers=None,
        adaptive_max_lead_workers=None,
        batch_reduce_mode="off",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _adaptive_plan(**overrides):
    values = dict(
        day_workers=1,
        workers=1,
        lead_workers=8,
        download_workers=14,
        reduce_workers=4,
        extract_workers=4,
        reduce_queue_size=12,
        extract_queue_size=12,
    )
    values.update(overrides)
    return nbm_monthly_backfill.WorkerPlan(**values)


def test_nbm_adaptive_controller_seeds_from_explicit_non_download_overrides():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(
        _adaptive_args(),
        initial_plan=_adaptive_plan(lead_workers=10, reduce_workers=5, extract_workers=6, reduce_queue_size=15, extract_queue_size=18),
    )

    assert controller.current_plan.lead_workers == 10
    assert controller.current_plan.reduce_workers == 5
    assert controller.current_plan.extract_workers == 6
    assert controller.current_plan.download_workers == 14


def test_nbm_adaptive_controller_preserves_explicit_low_lead_seed():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(
        _adaptive_args(),
        initial_plan=_adaptive_plan(lead_workers=1, reduce_workers=1, extract_workers=1),
    )

    assert controller.current_plan.lead_workers == 1


def test_nbm_adaptive_controller_keeps_download_workers_fixed():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(), initial_plan=_adaptive_plan())
    controller.record_success(_adaptive_day_performance("2024-01-01"))
    controller.record_success(_adaptive_day_performance("2024-01-02"))

    assert controller.current_plan.download_workers == 14


def test_nbm_adaptive_controller_waits_for_sample_days():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(adaptive_sample_days=3), initial_plan=_adaptive_plan())

    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-01"))

    assert tuned is False
    assert "waiting_for_samples" in reason
    assert controller.current_plan.day_workers == 1


def test_nbm_adaptive_controller_cooldown_blocks_repeat_adjustments():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(adaptive_cooldown_days=2), initial_plan=_adaptive_plan())
    controller.record_success(_adaptive_day_performance("2024-01-01", cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=2.0, crop_p95=2.1, cfgrib_open_median=1.0, row_metric_median=0.8, row_provenance_median=0.1))
    tuned, _ = controller.record_success(_adaptive_day_performance("2024-01-02", cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=2.0, crop_p95=2.1, cfgrib_open_median=1.0, row_metric_median=0.8, row_provenance_median=0.1))
    assert tuned is True

    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-03"))

    assert tuned is False
    assert "cooldown" in reason


def test_nbm_adaptive_controller_reduces_day_workers_for_cleanup_heavy_window():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(), initial_plan=_adaptive_plan(day_workers=2))

    controller.record_success(_adaptive_day_performance("2024-01-01", cleanup_seconds=30.0, full_day_seconds=100.0))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-02", cleanup_seconds=28.0, full_day_seconds=100.0))

    assert tuned is True
    assert reason == "cleanup_share_high"
    assert controller.current_plan.day_workers == 1


def test_nbm_adaptive_controller_reduces_reduce_workers_for_crop_heavy_window():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(adaptive_max_day_workers=1), initial_plan=_adaptive_plan(day_workers=1, reduce_workers=5))
    controller.record_success(_adaptive_day_performance("stable-1", crop_median=2.0, crop_p95=2.2))
    controller.record_success(_adaptive_day_performance("stable-2", crop_median=2.1, crop_p95=2.3))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-03", crop_median=8.0, crop_p95=8.5, cfgrib_open_median=1.0, row_metric_median=1.0, row_provenance_median=0.1))

    assert tuned is True
    assert reason == "crop_p95_high"
    assert controller.current_plan.reduce_workers == 4


def test_nbm_adaptive_controller_reduces_extract_workers_for_extract_heavy_window():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(adaptive_max_day_workers=1), initial_plan=_adaptive_plan(day_workers=1, extract_workers=5))
    controller.record_success(_adaptive_day_performance("stable-1", crop_median=2.0, crop_p95=2.2, cfgrib_open_median=1.0, row_metric_median=0.8))
    controller.record_success(_adaptive_day_performance("stable-2", crop_median=2.1, crop_p95=2.3, cfgrib_open_median=1.0, row_metric_median=0.8))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-03", crop_median=2.0, crop_p95=2.1, cfgrib_open_median=3.0, cfgrib_open_p95=3.2, row_metric_median=2.0, row_metric_p95=2.2, row_provenance_median=1.5, row_provenance_p95=1.6))

    assert tuned is True
    assert reason == "extract_p95_high"
    assert controller.current_plan.extract_workers == 4


def test_nbm_adaptive_controller_increases_day_workers_for_healthy_window():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(), initial_plan=_adaptive_plan(day_workers=1))

    controller.record_success(_adaptive_day_performance("2024-01-01", cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=2.0, crop_p95=2.1, cfgrib_open_median=1.0, row_metric_median=0.8, row_provenance_median=0.1))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-02", cleanup_seconds=4.5, full_day_seconds=100.0, crop_median=2.1, crop_p95=2.2, cfgrib_open_median=1.0, row_metric_median=0.8, row_provenance_median=0.1))

    assert tuned is True
    assert reason == "cleanup_ok_extract_ok"
    assert controller.current_plan.day_workers == 2


def test_nbm_adaptive_controller_failed_days_do_not_trigger_upward_tuning():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(), initial_plan=_adaptive_plan(day_workers=1))
    controller.record_failure()
    controller.record_success(_adaptive_day_performance("2024-01-01", cleanup_seconds=4.0, full_day_seconds=100.0))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-02", cleanup_seconds=4.0, full_day_seconds=100.0))

    assert tuned is False
    assert controller.current_plan.day_workers == 1
    assert reason in {"window_stable_no_change", "waiting_for_repeated_failures"}


def test_nbm_adaptive_controller_metrics_include_plan_and_reason():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(_adaptive_args(), initial_plan=_adaptive_plan())
    metrics = controller.metrics()

    assert metrics["Adaptive"] == "on"
    assert metrics["PlanDays"] == 1
    assert metrics["PlanLead"] == 8
    assert "TuneReason" in metrics


def test_nbm_adaptive_controller_pins_stage_workers_for_batch_cycle():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(
        _adaptive_args(batch_reduce_mode="cycle"),
        initial_plan=_adaptive_plan(day_workers=2, lead_workers=4, reduce_workers=4, extract_workers=4),
    )

    assert controller.current_plan.lead_workers == 4
    assert controller.current_plan.reduce_workers == 1
    assert controller.current_plan.extract_workers == 1


def test_nbm_adaptive_controller_preserves_explicit_queue_caps_after_tune():
    controller = nbm_monthly_backfill.AdaptiveWorkerController(
        _adaptive_args(),
        initial_plan=_adaptive_plan(reduce_queue_size=1, extract_queue_size=1),
    )
    controller.record_success(_adaptive_day_performance("2024-01-01", cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=0.2, crop_p95=0.25, cfgrib_open_median=0.1, row_metric_median=0.05, row_provenance_median=0.05))
    tuned, reason = controller.record_success(_adaptive_day_performance("2024-01-02", cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=0.2, crop_p95=0.25, cfgrib_open_median=0.1, row_metric_median=0.05, row_provenance_median=0.05))

    assert tuned is True
    assert reason == "cleanup_ok_extract_ok"
    assert controller.current_plan.reduce_queue_size == 1
    assert controller.current_plan.extract_queue_size == 1


def test_nbm_monthly_run_backfill_uses_updated_plan_for_later_days(tmp_path, monkeypatch):
    submitted: list[tuple[str, nbm_monthly_backfill.WorkerPlan]] = []

    def fake_process_day(args, day, *, bridge=None, worker_plan=None):
        token = day.target_date_local.isoformat()
        submitted.append((token, worker_plan))
        return nbm_monthly_backfill.DayRunResult(
            token=token,
            performance=_adaptive_day_performance(token, cleanup_seconds=4.0, full_day_seconds=100.0, crop_median=2.0, crop_p95=2.1),
        )

    monkeypatch.setattr(nbm_monthly_backfill, "process_day", fake_process_day)
    monkeypatch.setattr(nbm_monthly_backfill, "resolve_progress_mode", lambda **_kwargs: "log")
    monkeypatch.setattr(nbm_monthly_backfill, "free_disk_gb", lambda _path: 100.0)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-04",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        day_workers=1,
        smart_workers=False,
        cpu_cores=None,
        adaptive_workers=True,
        adaptive_sample_days=2,
        adaptive_cooldown_days=2,
        adaptive_min_day_workers=None,
        adaptive_max_day_workers=None,
        adaptive_min_reduce_workers=None,
        adaptive_max_reduce_workers=None,
        adaptive_min_extract_workers=None,
        adaptive_max_extract_workers=None,
        adaptive_min_lead_workers=None,
        adaptive_max_lead_workers=None,
        min_free_gb=1.0,
        workers=1,
        lead_workers=8,
        download_workers=14,
        reduce_workers=4,
        extract_workers=4,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=False,
        progress_mode="log",
        disable_dashboard_hotkeys=False,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    assert submitted[0][1].day_workers == 1
    assert submitted[1][1].day_workers == 1
    assert submitted[2][1].day_workers == 2


def test_nbm_monthly_build_raw_command_smart_workers_plan_for_32_cores():
    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=ROOT / "data/runtime/test",
        selection_mode="overnight_0005",
        batch_reduce_mode="off",
        day_workers=1,
        smart_workers=True,
        cpu_cores=32,
        min_free_gb=1.0,
        workers=1,
        lead_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=False,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    plan = nbm_monthly_backfill.apply_smart_worker_plan(args)

    assert plan is not None
    assert plan.cpu_cores == 32
    assert args.day_workers == 2
    assert args.workers == 1
    assert args.lead_workers == 14
    assert args.download_workers == 14
    assert args.reduce_workers == 7
    assert args.extract_workers == 7


def test_nbm_monthly_smart_workers_batch_cycle_uses_day_parallel_profile():
    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-14",
        run_root=ROOT / "data/runtime/test",
        selection_mode="overnight_0005",
        batch_reduce_mode="cycle",
        day_workers=1,
        smart_workers=True,
        cpu_cores=4,
        min_free_gb=1.0,
        workers=1,
        lead_workers=4,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=False,
        overnight_subprocess=False,
        overnight_fast=True,
        progress_mode="dashboard",
        disable_dashboard_hotkeys=False,
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    plan = nbm_monthly_backfill.apply_smart_worker_plan(args)

    assert plan is not None
    assert plan.cpu_cores == 4
    assert args.day_workers == 4
    assert args.workers == 1
    assert args.lead_workers == 4
    assert args.download_workers == 4
    assert args.reduce_workers == 1
    assert args.extract_workers == 1
    assert args.reduce_queue_size == 2
    assert args.extract_queue_size == 2


def test_progress_dashboard_summary_rows_render_monthly_custom_metrics():
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=3,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
    )
    reporter.set_metrics(DayWorkers=2, FreeGB="42.0", Raw=1, Overnight=1, Validate=0, Cleanup=0)
    rows = reporter.renderer._summary_rows(reporter.state, width=140)
    reporter.close(status="done")

    assert any("DayWorkers 2" in row for row in rows)
    assert any("FreeGB 42.0" in row for row in rows)
    assert any("Raw 1" in row for row in rows)
    assert any("Overnight 1" in row for row in rows)


def test_progress_dashboard_group_rows_render_day_status():
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=3,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
    )
    reporter.upsert_group("2024-01-01", label="2024-01-01", total=1, completed=0, failed=0, active=1, status="raw:write_outputs")
    rows = reporter.renderer._group_rows(reporter.state, width=120, height=4)
    reporter.close(status="done")

    assert any("2024-01-01" in row for row in rows)
    assert any("write_outputs" in row for row in rows)


def test_progress_dashboard_footer_rows_render_info_events():
    reporter = progress_mod.create_progress_reporter(
        "NBM monthly backfill",
        unit="day",
        total=1,
        mode="dashboard",
        stream=io.StringIO(),
        is_tty=True,
    )
    reporter.log_event("2024-01-01 stdout: child plain stdout", level="info")
    rows = reporter.renderer._footer_rows(reporter.state, width=140, height=4)
    reporter.close(status="done")

    assert any("child plain stdout" in row for row in rows)


def test_nbm_overnight_build_range_matches_cli_main(tmp_path, monkeypatch):
    features_root = tmp_path / "features"
    valid_date = "2024-01-01"
    wide_path = (
        features_root
        / "features"
        / "wide"
        / f"valid_date_local={valid_date}"
        / "init_date_local=2023-12-31"
        / "mode=intraday"
        / "cycle_20240101T0400Z_wide.parquet"
    )
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": "2024-01-01T04:00:00+00:00",
                "valid_time_utc": "2024-01-01T12:00:00+00:00",
                "init_time_local": "2023-12-31T23:00:00-05:00",
                "valid_time_local": "2024-01-01T07:00:00-05:00",
                "valid_date_local": valid_date,
                "forecast_hour": 8,
                "tmp": 280.0,
                "tmax": 283.0,
                "tmin": 275.0,
            }
        ]
    ).to_parquet(wide_path, index=False)
    manifest_path = features_root / "metadata" / "manifest" / "init_date_local=2023-12-31" / "mode=intraday" / "cycle_20240101T0400Z_manifest.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "extraction_status": "ok",
                "wide_output_paths": str(wide_path),
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": "2024-01-01T04:00:00+00:00",
                "init_time_local": "2023-12-31T23:00:00-05:00",
                "processed_timestamp_utc": "2024-01-01T04:00:00+00:00",
                "valid_date_local": valid_date,
            }
        ]
    ).to_parquet(manifest_path, index=False)
    in_process_output = tmp_path / "in_process"
    cli_output = tmp_path / "cli"

    nbm_overnight_features.build_range(
        features_root=features_root,
        output_dir=in_process_output,
        start_local_date=dt.date(2024, 1, 1),
        end_local_date=dt.date(2024, 1, 1),
        cutoff_local_time=dt.time(0, 5),
        progress=None,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_nbm_overnight_features.py",
            "--features-root",
            str(features_root),
            "--output-dir",
            str(cli_output),
            "--start-local-date",
            valid_date,
            "--end-local-date",
            valid_date,
        ],
    )

    assert nbm_overnight_features.main() == 0

    in_process_df = pd.read_parquet(in_process_output / f"target_date_local={valid_date}" / "nbm.overnight.parquet")
    cli_df = pd.read_parquet(cli_output / f"target_date_local={valid_date}" / "nbm.overnight.parquet")
    pd.testing.assert_frame_equal(in_process_df, cli_df)
    in_process_manifest = pd.read_parquet(in_process_output / f"target_date_local={valid_date}" / "nbm.overnight.manifest.parquet")
    cli_manifest = pd.read_parquet(cli_output / f"target_date_local={valid_date}" / "nbm.overnight.manifest.parquet")
    in_process_manifest = in_process_manifest.drop(columns=["overnight_output_path", "manifest_output_path"])
    cli_manifest = cli_manifest.drop(columns=["overnight_output_path", "manifest_output_path"])
    pd.testing.assert_frame_equal(in_process_manifest, cli_manifest)


def test_nbm_overnight_load_issue_frame_projects_required_columns(tmp_path, monkeypatch):
    wide_path = tmp_path / "wide.parquet"
    pd.DataFrame(
        [
            {
                "init_time_utc": "2024-01-01T04:00:00+00:00",
                "valid_time_utc": "2024-01-01T12:00:00+00:00",
                "init_time_local": "2023-12-31T23:00:00-05:00",
                "valid_time_local": "2024-01-01T07:00:00-05:00",
                "valid_date_local": "2024-01-01",
                "forecast_hour": 8,
                "tmp": 280.0,
                "tmax": 283.0,
                "tmin": 275.0,
                "unused_extra_column": 999,
            }
        ]
    ).to_parquet(wide_path, index=False)

    captured_columns: list[str] = []
    real_read_parquet = nbm_overnight_features.pd.read_parquet

    def fake_read_parquet(path, *args, **kwargs):
        if pathlib.Path(path) == wide_path:
            captured_columns.extend(kwargs.get("columns") or [])
        return real_read_parquet(path, *args, **kwargs)

    monkeypatch.setattr(nbm_overnight_features.pd, "read_parquet", fake_read_parquet)

    frame = nbm_overnight_features.load_issue_frame(wide_path)

    assert "unused_extra_column" not in captured_columns
    assert "valid_time_local" in captured_columns
    assert "forecast_hour" in captured_columns
    assert "tmp" in captured_columns
    assert list(frame.columns) == [
        "init_time_utc",
        "valid_time_utc",
        "init_time_local",
        "valid_time_local",
        "valid_date_local",
        "forecast_hour",
        "tmp",
        "tmax",
        "tmin",
    ]


def test_nbm_overnight_build_for_date_handles_sparse_wide_columns_with_projection(tmp_path):
    features_root = tmp_path / "features"
    target_date_local = dt.date(2024, 1, 1)
    wide_path = (
        features_root
        / "features"
        / "wide"
        / "valid_date_local=2024-01-01"
        / "init_date_local=2023-12-31"
        / "mode=intraday"
        / "cycle_20240101T0400Z_wide.parquet"
    )
    wide_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "init_time_utc": "2024-01-01T04:00:00+00:00",
                "valid_time_utc": "2024-01-01T12:00:00+00:00",
                "init_time_local": "2023-12-31T23:00:00-05:00",
                "valid_time_local": "2024-01-01T07:00:00-05:00",
                "valid_date_local": "2024-01-01",
                "forecast_hour": 8,
                "tmp": 280.0,
                "tmax": 283.0,
                "tmin": 275.0,
            }
        ]
    ).to_parquet(wide_path, index=False)
    issue_catalog = pd.DataFrame.from_records(
        [
            {
                "wide_path": str(wide_path),
                "source_model": "NBM",
                "source_product": "grib2-core",
                "source_version": "nbm-grib2-core-public",
                "station_id": "KLGA",
                "init_time_utc": pd.Timestamp("2024-01-01T04:00:00+00:00"),
                "init_time_local": pd.Timestamp("2023-12-31T23:00:00-05:00"),
                "processed_timestamp_utc": pd.Timestamp("2024-01-01T04:00:00+00:00"),
                "valid_date_local": "2024-01-01",
                "available_time_local": pd.Timestamp("2023-12-31T23:00:00-05:00"),
            }
        ]
    )

    overnight_path, manifest_path = nbm_overnight_features.build_for_date(
        issue_catalog=issue_catalog,
        target_date_local=target_date_local,
        cutoff_local_time=dt.time(0, 5),
        output_dir=tmp_path / "overnight",
        progress=None,
    )

    summary_df = pd.read_parquet(overnight_path)
    manifest_df = pd.read_parquet(manifest_path)
    row = summary_df.iloc[0]
    assert row["nbm_temp_2m_day_max_k"] == 280.0
    assert row["nbm_native_tmax_2m_day_max_k"] == 283.0
    assert row["nbm_native_tmin_2m_day_min_k"] == 275.0
    assert pd.isna(row["nbm_tstm_day_max_pct"])
    assert manifest_df.iloc[0]["status"] == "ok"


def test_nbm_monthly_backfill_keep_reduced_preserves_scratch_but_cleans_raw(tmp_path, monkeypatch):
    commands: list[list[str]] = []

    def fake_run(command, cwd, check):
        commands.append(command)
        assert cwd == ROOT
        assert check is True
        if command[1] == "tools/nbm/build_grib2_features.py":
            scratch_dir = pathlib.Path(command[command.index("--scratch-dir") + 1])
            raw_dir = pathlib.Path(command[command.index("--output-dir") + 1])
            (scratch_dir / "reduced" / "init_date_local=2024-01-01").mkdir(parents=True, exist_ok=True)
            (scratch_dir / "reduced" / "init_date_local=2024-01-01" / "example.subset.grib2").write_bytes(b"reduced")
            (raw_dir / "blend.20240101" / "core").mkdir(parents=True, exist_ok=True)
            (raw_dir / "blend.20240101" / "core" / "raw.grib2").write_bytes(b"raw")
        return SimpleNamespace(returncode=0)

    def fake_build_overnight_in_process(*, target_date_local, raw_dir, overnight_root):
        root = overnight_root / f"target_date_local={target_date_local}"
        root.mkdir(parents=True, exist_ok=True)
        output_path = root / "nbm.overnight.parquet"
        manifest_path = root / "nbm.overnight.manifest.parquet"
        pd.DataFrame([{"target_date_local": target_date_local}]).to_parquet(output_path, index=False)
        pd.DataFrame([{"status": "ok"}]).to_parquet(manifest_path, index=False)
        return [output_path, manifest_path]

    monkeypatch.setattr(nbm_monthly_backfill.subprocess, "run", fake_run)
    monkeypatch.setattr(nbm_monthly_backfill, "build_overnight_in_process", fake_build_overnight_in_process)

    args = argparse.Namespace(
        start_local_date="2024-01-01",
        end_local_date="2024-01-01",
        run_root=tmp_path / "runtime",
        selection_mode="overnight_0005",
        workers=1,
        lead_workers=6,
        download_workers=None,
        reduce_workers=None,
        extract_workers=None,
        reduce_queue_size=None,
        extract_queue_size=None,
        keep_reduced=True,
        overnight_fast=False,
        progress_mode="log",
        pause_control_file=None,
        max_task_attempts=None,
        retry_backoff_seconds=None,
        retry_max_backoff_seconds=None,
        overwrite=False,
        keep_temp_on_failure=False,
    )

    assert nbm_monthly_backfill.run_backfill(args) == 0
    token = "2024-01-01"
    raw_root = nbm_monthly_backfill.day_raw_root(args.run_root, token)
    scratch_root = nbm_monthly_backfill.day_scratch_root(args.run_root, token)
    assert "--keep-reduced" in commands[0]
    assert not raw_root.exists()
    assert scratch_root.exists()
