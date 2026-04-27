#!/usr/bin/env python3
"""Download only the HRRR GRIB records needed for KLGA Tmax modeling.

This script uses the `.idx` sidecar file to find exact byte ranges for
recommended variables and levels, then downloads only those ranges from
the source `.grib2`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Pattern

from hrrr_fields import PRODUCT_FIELD_SPECS


PRODUCT_TO_TOKEN = {
    "surface": "wrfsfcf",
    "subhourly": "wrfsubhf",
}

BASE_URLS = {
    "aws": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
    "google": "https://storage.googleapis.com/high-resolution-rapid-refresh",
    "nomads": "https://nomads.ncep.noaa.gov/pub/data/nccf/com/hrrr/prod",
}

HTTP_TIMEOUT_SECONDS = 120
HTTP_MAX_RETRIES = 4
HTTP_RETRY_BACKOFF_SECONDS = 2.0
USER_AGENT = "hrrr-fetcher/1.0"
MERGE_GAP_BYTES = 64 * 1024


@dataclass
class IdxRecord:
    record_number: int
    byte_start: int
    run_stamp: str
    variable: str
    level: str
    valid_desc: str
    byte_end: int | None = None


@dataclass(frozen=True)
class ByteRangeSpan:
    start: int
    end: int

    @property
    def byte_length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class DownloadSubsetResult:
    subset_path: Path
    head_used: bool
    remote_file_size: int | None
    selected_record_count: int
    merged_range_count: int
    downloaded_range_bytes: int
    timing_idx_fetch_seconds: float = 0.0
    timing_idx_parse_seconds: float = 0.0
    timing_head_seconds: float = 0.0
    timing_range_download_seconds: float = 0.0

    def __fspath__(self) -> str:
        return os.fspath(self.subset_path)

    def __str__(self) -> str:
        return str(self.subset_path)

    def __getattr__(self, name: str):
        return getattr(self.subset_path, name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DownloadSubsetResult):
            return self.subset_path == other.subset_path
        if isinstance(other, Path):
            return self.subset_path == other
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Run date in YYYYMMDD.")
    parser.add_argument("--cycle", required=True, type=int, help="UTC cycle hour, 0-23.")
    parser.add_argument(
        "--product",
        required=True,
        choices=sorted(PRODUCT_TO_TOKEN),
        help="HRRR product family to pull.",
    )
    parser.add_argument(
        "--forecast-hours",
        required=True,
        help="Comma-separated forecast hours, e.g. 0,1,2,3,4 or 12,13,14.",
    )
    parser.add_argument(
        "--source",
        default="aws",
        choices=sorted(BASE_URLS),
        help="Archive source to use.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for subset GRIBs and CSV manifests.",
    )
    return parser.parse_args()


def forecast_hours_arg(value: str) -> list[int]:
    hours: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        hours.append(int(part))
    if not hours:
        raise ValueError("No forecast hours were provided.")
    return hours


def build_remote_paths(date: str, cycle: int, product: str, forecast_hour: int, source: str) -> tuple[str, str]:
    token = PRODUCT_TO_TOKEN[product]
    filename = f"hrrr.t{cycle:02d}z.{token}{forecast_hour:02d}.grib2"
    relpath = f"hrrr.{date}/conus/{filename}"
    base = BASE_URLS[source]
    grib_url = f"{base}/{relpath}"
    idx_url = f"{grib_url}.idx"
    return grib_url, idx_url


def _urlopen_with_retries(request: str | urllib.request.Request):
    last_error: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            return urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == HTTP_MAX_RETRIES:
                break
            time.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)
    assert last_error is not None
    raise last_error


def _retry_delay(attempt: int) -> None:
    time.sleep(HTTP_RETRY_BACKOFF_SECONDS * attempt)


def read_response_with_retries(request: urllib.request.Request) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                return response.read()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == HTTP_MAX_RETRIES:
                break
            _retry_delay(attempt)
    assert last_error is not None
    raise last_error


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return read_response_with_retries(request).decode("utf-8")


def fetch_content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with _urlopen_with_retries(request) as response:
        value = response.headers.get("Content-Length")
    if value is None:
        raise RuntimeError(f"Missing Content-Length for {url}")
    return int(value)


def parse_idx(idx_text: str, content_length: int) -> list[IdxRecord]:
    records: list[IdxRecord] = []
    starts: list[int] = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        records.append(
            IdxRecord(
                record_number=int(parts[0]),
                byte_start=int(parts[1]),
                run_stamp=parts[2],
                variable=parts[3],
                level=parts[4],
                valid_desc=parts[5],
            )
        )
        starts.append(int(parts[1]))

    for index, record in enumerate(records):
        next_start = starts[index + 1] if index + 1 < len(starts) else content_length
        record.byte_start = starts[index]
        record.byte_end = next_start - 1
    return records


def parse_idx_without_content_length(idx_text: str) -> list[IdxRecord]:
    records: list[IdxRecord] = []
    starts: list[int] = []
    for line in idx_text.splitlines():
        parts = line.split(":")
        if len(parts) < 6:
            continue
        records.append(
            IdxRecord(
                record_number=int(parts[0]),
                byte_start=int(parts[1]),
                run_stamp=parts[2],
                variable=parts[3],
                level=parts[4],
                valid_desc=parts[5],
            )
        )
        starts.append(int(parts[1]))

    for index, record in enumerate(records[:-1]):
        next_start = starts[index + 1]
        record.byte_start = starts[index]
        record.byte_end = next_start - 1
    return records


def wanted_records(records: Iterable[IdxRecord], product: str) -> list[IdxRecord]:
    wanted = []
    specs = PRODUCT_FIELD_SPECS[product]
    for record in records:
        valid_desc = f" {record.valid_desc.lower()} "
        for spec in specs:
            if spec.mode == "instant" and (" ave " in valid_desc or " acc " in valid_desc):
                continue
            if spec.mode == "accum" and " acc " not in valid_desc:
                continue
            if record.variable == spec.variable and record.level == spec.level:
                wanted.append(record)
                break
    return wanted


def wanted_records_by_patterns(
    records: Iterable[IdxRecord],
    patterns: Iterable[str | Pattern[str]],
) -> list[IdxRecord]:
    pattern_list = list(patterns)
    if all(isinstance(pattern, str) for pattern in pattern_list):
        compiled_patterns = list(_compile_patterns(tuple(str(pattern) for pattern in pattern_list)))
    else:
        compiled_patterns = [re.compile(pattern) if isinstance(pattern, str) else pattern for pattern in pattern_list]
    selected: list[IdxRecord] = []
    seen: set[tuple[int, int, int | None]] = set()
    for record in records:
        inventory_line = (
            f"{record.record_number}:{record.byte_start}:{record.run_stamp}:"
            f"{record.variable}:{record.level}:{record.valid_desc}"
        )
        for pattern in compiled_patterns:
            if not pattern.search(inventory_line):
                continue
            key = (record.record_number, record.byte_start, record.byte_end)
            if key not in seen:
                selected.append(record)
                seen.add(key)
            break
    return selected


@lru_cache(maxsize=64)
def _compile_patterns(patterns: tuple[str, ...]) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


def selected_ranges_require_content_length(records: Iterable[IdxRecord], selected: Iterable[IdxRecord]) -> bool:
    record_list = list(records)
    selected_list = list(selected)
    if not record_list or not selected_list:
        return False
    last_record = record_list[-1]
    return any(
        record.record_number == last_record.record_number and record.byte_start == last_record.byte_start
        for record in selected_list
    )


def download_range(
    url: str,
    start: int,
    end: int,
    *,
    chunk_callback: Callable[[int], None] | None = None,
) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": USER_AGENT,
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
                chunks: list[bytes] = []
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    if chunk_callback is not None:
                        chunk_callback(len(chunk))
                return b"".join(chunks)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt == HTTP_MAX_RETRIES:
                break
            _retry_delay(attempt)
    assert last_error is not None
    raise last_error


def merged_byte_ranges(records: Iterable[IdxRecord], *, max_gap_bytes: int = MERGE_GAP_BYTES) -> list[ByteRangeSpan]:
    spans = [
        ByteRangeSpan(start=record.byte_start, end=int(record.byte_end))
        for record in records
        if record.byte_end is not None
    ]
    if not spans:
        return []

    spans.sort(key=lambda item: item.start)
    merged: list[ByteRangeSpan] = [spans[0]]
    for span in spans[1:]:
        current = merged[-1]
        if span.start <= current.end + max_gap_bytes + 1:
            merged[-1] = ByteRangeSpan(start=current.start, end=max(current.end, span.end))
        else:
            merged.append(span)
    return merged


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination

    tmp_path = destination.with_name(f"{destination.name}.part")
    last_error: Exception | None = None
    for attempt in range(1, HTTP_MAX_RETRIES + 1):
        tmp_path.unlink(missing_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_path.replace(destination)
            return destination
        except urllib.error.HTTPError:
            tmp_path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            tmp_path.unlink(missing_ok=True)
            if attempt == HTTP_MAX_RETRIES:
                break
            _retry_delay(attempt)

    assert last_error is not None
    raise last_error


def write_manifest_csv(path: Path, grib_url: str, records: list[IdxRecord]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grib_url",
                "record_number",
                "byte_start",
                "byte_end",
                "variable",
                "level",
                "valid_desc",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    grib_url,
                    record.record_number,
                    record.byte_start,
                    record.byte_end,
                    record.variable,
                    record.level,
                    record.valid_desc,
                ]
            )


def read_manifest_records(path: Path) -> list[IdxRecord]:
    records: list[IdxRecord] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                IdxRecord(
                    record_number=int(row["record_number"]),
                    byte_start=int(row["byte_start"]),
                    run_stamp="",
                    variable=str(row["variable"]),
                    level=str(row["level"]),
                    valid_desc=str(row["valid_desc"]),
                    byte_end=int(row["byte_end"]) if row.get("byte_end") not in {None, ""} else None,
                )
            )
    return records


def manifest_inventory_lines(manifest_path: Path, *, run_stamp: str) -> list[str]:
    return [
        f"{record.record_number}:{record.byte_start}:{run_stamp}:{record.variable}:{record.level}:{record.valid_desc}"
        for record in read_manifest_records(manifest_path)
    ]


def selection_signature(*, grib_url: str, patterns: Iterable[str | Pattern[str]], selected: list[IdxRecord]) -> str:
    pattern_values = [pattern.pattern if hasattr(pattern, "pattern") else str(pattern) for pattern in patterns]
    selected_rows = [
        {
            "record_number": record.record_number,
            "byte_start": record.byte_start,
            "byte_end": record.byte_end,
            "variable": record.variable,
            "level": record.level,
            "valid_desc": record.valid_desc,
        }
        for record in selected
    ]
    payload = {
        "grib_url": grib_url,
        "patterns": pattern_values,
        "selected_rows": selected_rows,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def manifest_matches_selection(
    manifest_path: Path,
    *,
    expected_signature: str,
) -> bool:
    if not manifest_path.exists():
        return False
    try:
        with manifest_path.open(newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            row = next(reader, None)
    except Exception:
        return False
    if header != ["selection_signature"]:
        return False
    return bool(row) and len(row) == 1 and row[0] == expected_signature


def write_selection_manifest(path: Path, *, signature: str) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["selection_signature"])
        writer.writerow([signature])


def download_selected_records(
    *,
    grib_url: str,
    idx_url: str,
    subset_path: Path,
    manifest_path: Path,
    selection_manifest_path: Path,
    patterns: Iterable[str | Pattern[str]],
    selected: list[IdxRecord],
    range_merge_gap_bytes: int = MERGE_GAP_BYTES,
    overwrite: bool = False,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> DownloadSubsetResult:
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    signature = selection_signature(grib_url=grib_url, patterns=patterns, selected=selected)
    if (
        subset_path.exists()
        and manifest_path.exists()
        and selection_manifest_path.exists()
        and not overwrite
        and manifest_matches_selection(selection_manifest_path, expected_signature=signature)
    ):
        merged_spans = merged_byte_ranges(selected, max_gap_bytes=range_merge_gap_bytes)
        return DownloadSubsetResult(
            subset_path=subset_path,
            head_used=False,
            remote_file_size=None,
            selected_record_count=len(selected),
            merged_range_count=len(merged_spans),
            downloaded_range_bytes=sum(span.byte_length for span in merged_spans),
        )

    tmp_path = subset_path.with_name(f"{subset_path.name}.part")
    merged_spans = merged_byte_ranges(selected, max_gap_bytes=range_merge_gap_bytes)
    total_selected_bytes = sum(span.byte_length for span in merged_spans)
    downloaded_bytes = 0
    basename = Path(grib_url).name

    if progress_callback is None:
        print(
            f"[start] file={basename} records={len(selected)} spans={len(merged_spans)} "
            f"selected_mb={total_selected_bytes / (1024 * 1024):.1f}"
        )
    else:
        progress_callback(
            "start",
            {
                "file_label": basename,
                "selected_record_count": len(selected),
                "span_count": len(merged_spans),
                "total_bytes": total_selected_bytes,
            },
        )

    tmp_path.unlink(missing_ok=True)
    download_started_at = time.perf_counter()
    with tmp_path.open("wb") as handle:
        try:
            for index, span in enumerate(merged_spans, start=1):
                span_downloaded = 0

                def report_chunk(chunk_len: int) -> None:
                    nonlocal span_downloaded
                    span_downloaded += chunk_len
                    if progress_callback is None:
                        return
                    current_downloaded = downloaded_bytes + span_downloaded
                    progress_callback(
                        "progress",
                        {
                            "file_label": basename,
                            "span_index": index,
                            "span_total": len(merged_spans),
                            "downloaded_bytes": current_downloaded,
                            "total_bytes": total_selected_bytes,
                            "pct": ((current_downloaded / total_selected_bytes) * 100.0) if total_selected_bytes else 0.0,
                        },
                    )

                try:
                    payload = download_range(
                        grib_url,
                        span.start,
                        span.end,
                        chunk_callback=report_chunk if progress_callback is not None else None,
                    )
                except TypeError as exc:
                    if "chunk_callback" not in str(exc):
                        raise
                    payload = download_range(grib_url, span.start, span.end)
                    span_downloaded = len(payload)
                    if progress_callback is not None:
                        current_downloaded = downloaded_bytes + span_downloaded
                        progress_callback(
                            "progress",
                            {
                                "file_label": basename,
                                "span_index": index,
                                "span_total": len(merged_spans),
                                "downloaded_bytes": current_downloaded,
                                "total_bytes": total_selected_bytes,
                                "pct": ((current_downloaded / total_selected_bytes) * 100.0) if total_selected_bytes else 0.0,
                            },
                        )
                handle.write(payload)
                downloaded_bytes += len(payload)
                pct = (downloaded_bytes / total_selected_bytes) * 100.0 if total_selected_bytes else 0.0
                if progress_callback is None:
                    print(
                        f"[progress] file={basename} span={index}/{len(merged_spans)} "
                        f"downloaded_mb={downloaded_bytes / (1024 * 1024):.1f} pct={pct:.1f}"
                    )
                else:
                    progress_callback(
                        "progress",
                        {
                            "file_label": basename,
                            "span_index": index,
                            "span_total": len(merged_spans),
                            "downloaded_bytes": downloaded_bytes,
                            "total_bytes": total_selected_bytes,
                            "pct": pct,
                        },
                    )
        except Exception:
            tmp_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            selection_manifest_path.unlink(missing_ok=True)
            raise

    tmp_path.replace(subset_path)
    write_manifest_csv(manifest_path, grib_url, selected)
    write_selection_manifest(selection_manifest_path, signature=signature)
    if progress_callback is None:
        print(f"[ok] wrote {subset_path} and {manifest_path}")
    else:
        progress_callback(
            "complete",
            {
                "file_label": basename,
                "downloaded_bytes": downloaded_bytes,
                "total_bytes": total_selected_bytes,
            },
        )
    return DownloadSubsetResult(
        subset_path=subset_path,
        head_used=False,
        remote_file_size=None,
        selected_record_count=len(selected),
        merged_range_count=len(merged_spans),
        downloaded_range_bytes=total_selected_bytes,
        timing_range_download_seconds=time.perf_counter() - download_started_at,
    )


def download_subset_for_inventory_patterns(
    *,
    date: str,
    cycle: int,
    product: str,
    forecast_hour: int,
    source: str,
    patterns: Iterable[str | Pattern[str]],
    subset_path: Path,
    manifest_path: Path,
    selection_manifest_path: Path,
    range_merge_gap_bytes: int = MERGE_GAP_BYTES,
    overwrite: bool = False,
    progress_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> DownloadSubsetResult:
    grib_url, idx_url = build_remote_paths(date, cycle, product, forecast_hour, source)
    idx_fetch_started_at = time.perf_counter()
    idx_text = fetch_text(idx_url)
    idx_fetch_seconds = time.perf_counter() - idx_fetch_started_at
    idx_parse_started_at = time.perf_counter()
    records = parse_idx_without_content_length(idx_text)
    selected = wanted_records_by_patterns(records, patterns)
    idx_parse_seconds = time.perf_counter() - idx_parse_started_at
    if not selected:
        raise RuntimeError(f"No matching records found for {grib_url}")
    head_used = False
    remote_file_size: int | None = None
    head_seconds = 0.0
    if selected_ranges_require_content_length(records, selected):
        head_started_at = time.perf_counter()
        remote_file_size = fetch_content_length(grib_url)
        head_seconds = time.perf_counter() - head_started_at
        idx_parse_started_at = time.perf_counter()
        records = parse_idx(idx_text, remote_file_size)
        selected = wanted_records_by_patterns(records, patterns)
        idx_parse_seconds += time.perf_counter() - idx_parse_started_at
        head_used = True
    result = download_selected_records(
        grib_url=grib_url,
        idx_url=idx_url,
        subset_path=subset_path,
        manifest_path=manifest_path,
        selection_manifest_path=selection_manifest_path,
        patterns=patterns,
        selected=selected,
        range_merge_gap_bytes=range_merge_gap_bytes,
        overwrite=overwrite,
        progress_callback=progress_callback,
    )
    return DownloadSubsetResult(
        subset_path=result.subset_path,
        head_used=head_used,
        remote_file_size=remote_file_size,
        selected_record_count=result.selected_record_count,
        merged_range_count=result.merged_range_count,
        downloaded_range_bytes=result.downloaded_range_bytes,
        timing_idx_fetch_seconds=idx_fetch_seconds,
        timing_idx_parse_seconds=idx_parse_seconds,
        timing_head_seconds=head_seconds,
        timing_range_download_seconds=result.timing_range_download_seconds,
    )


def process_one(date: str, cycle: int, product: str, forecast_hour: int, source: str, output_dir: Path) -> bool:
    grib_url, idx_url = build_remote_paths(date, cycle, product, forecast_hour, source)
    try:
        idx_text = fetch_text(idx_url)
        content_length = fetch_content_length(grib_url)
    except urllib.error.HTTPError as exc:
        print(f"[skip] {grib_url} -> {exc}", file=sys.stderr)
        return False
    except Exception as exc:
        print(f"[skip] {grib_url} -> {exc}", file=sys.stderr)
        return False

    records = parse_idx(idx_text, content_length)
    selected = wanted_records(records, product)
    if not selected:
        print(f"[skip] no matching records for {grib_url}", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    basename = Path(grib_url).name
    subset_path = output_dir / f"{basename}.subset.grib2"
    manifest_path = output_dir / f"{basename}.manifest.csv"
    merged_spans = merged_byte_ranges(selected)
    total_selected_bytes = sum(span.byte_length for span in merged_spans)
    downloaded_bytes = 0

    print(
        f"[start] file={basename} records={len(selected)} spans={len(merged_spans)} "
        f"selected_mb={total_selected_bytes / (1024 * 1024):.1f}"
    )

    with subset_path.open("wb") as handle:
        try:
            for index, span in enumerate(merged_spans, start=1):
                payload = download_range(grib_url, span.start, span.end)
                handle.write(payload)
                downloaded_bytes += len(payload)
                pct = (downloaded_bytes / total_selected_bytes) * 100.0 if total_selected_bytes else 0.0
                print(
                    f"[progress] file={basename} span={index}/{len(merged_spans)} "
                    f"downloaded_mb={downloaded_bytes / (1024 * 1024):.1f} pct={pct:.1f}"
                )
        except Exception as exc:
            print(f"[skip] {grib_url} -> {exc}", file=sys.stderr)
            subset_path.unlink(missing_ok=True)
            manifest_path.unlink(missing_ok=True)
            return False

    write_manifest_csv(manifest_path, grib_url, selected)
    print(f"[ok] wrote {subset_path} and {manifest_path}")
    return True


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    try:
        forecast_hours = forecast_hours_arg(args.forecast_hours)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    wrote = 0
    failed = 0
    for forecast_hour in forecast_hours:
        ok = process_one(
            date=args.date,
            cycle=args.cycle,
            product=args.product,
            forecast_hour=forecast_hour,
            source=args.source,
            output_dir=output_dir,
        )
        if ok:
            wrote += 1
        else:
            failed += 1

    print(f"[done] wrote={wrote} failed={failed} output_dir={output_dir}")
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
