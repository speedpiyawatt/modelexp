#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_hrrr_klga_feature_shards import inventory_selection_patterns
from fetch_hrrr_records import (
    MERGE_GAP_BYTES,
    download_range,
    fetch_content_length,
    fetch_text,
    merged_byte_ranges,
    parse_idx,
    parse_idx_without_content_length,
    selected_ranges_require_content_length,
    wanted_records_by_patterns,
)


SOURCE_BASES = {
    "google": "https://storage.googleapis.com/high-resolution-rapid-refresh",
    "azure": "https://noaahrrr.blob.core.windows.net/hrrr",
}
DEFAULT_SOURCES = ("google", "azure")
DEFAULT_OUTPUT_ROOT = Path("tools/hrrr/data/runtime/source_benchmarks")


@dataclass(frozen=True)
class BenchmarkRun:
    source: str
    run_index: int
    grib_url: str
    idx_url: str
    selected_record_count: int
    merged_range_count: int
    downloaded_range_bytes: int
    head_used: bool
    timing_idx_fetch_seconds: float
    timing_idx_parse_seconds: float
    timing_head_seconds: float
    timing_range_download_seconds: float
    timing_total_seconds: float
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark isolated HRRR source fetch performance without changing production fetch defaults.")
    parser.add_argument("--date", required=True, help="Run date in YYYYMMDD format.")
    parser.add_argument("--cycle", required=True, type=int, help="UTC cycle hour, 0-23.")
    parser.add_argument("--forecast-hour", required=True, type=int, help="Forecast hour to benchmark.")
    parser.add_argument(
        "--product",
        default="surface",
        choices=("surface",),
        help="HRRR product family. The KLGA pipeline currently benchmarks the surface product.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(sorted(SOURCE_BASES)),
        default=list(DEFAULT_SOURCES),
        help="One or more remote sources to benchmark.",
    )
    parser.add_argument("--runs", type=int, default=3, help="Benchmark runs per source.")
    parser.add_argument(
        "--range-merge-gap-bytes",
        type=int,
        default=MERGE_GAP_BYTES,
        help="Merge adjacent selected ranges when the gap is at most this many bytes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for benchmark_runs.json, benchmark_runs.csv, and benchmark_summary.json.",
    )
    return parser.parse_args()


def build_remote_paths_for_source(date: str, cycle: int, product: str, forecast_hour: int, source: str) -> tuple[str, str]:
    if product != "surface":
        raise ValueError(f"Unsupported product for benchmark: {product}")
    base = SOURCE_BASES[source].rstrip("/")
    filename = f"hrrr.t{cycle:02d}z.wrfsfcf{forecast_hour:02d}.grib2"
    relpath = f"hrrr.{date}/conus/{filename}"
    grib_url = f"{base}/{relpath}"
    return grib_url, f"{grib_url}.idx"


def benchmark_one_run(
    *,
    source: str,
    run_index: int,
    date: str,
    cycle: int,
    product: str,
    forecast_hour: int,
    range_merge_gap_bytes: int,
) -> BenchmarkRun:
    grib_url, idx_url = build_remote_paths_for_source(date, cycle, product, forecast_hour, source)
    patterns = [pattern for _, pattern in inventory_selection_patterns()]

    started_at = time.perf_counter()

    idx_fetch_started_at = time.perf_counter()
    idx_text = fetch_text(idx_url)
    timing_idx_fetch_seconds = time.perf_counter() - idx_fetch_started_at

    idx_parse_started_at = time.perf_counter()
    records = parse_idx_without_content_length(idx_text)
    selected = wanted_records_by_patterns(records, patterns)
    timing_idx_parse_seconds = time.perf_counter() - idx_parse_started_at
    if not selected:
        raise RuntimeError(f"No selected records found for {grib_url}")

    head_used = False
    timing_head_seconds = 0.0
    if selected_ranges_require_content_length(records, selected):
        head_used = True
        head_started_at = time.perf_counter()
        content_length = fetch_content_length(grib_url)
        timing_head_seconds = time.perf_counter() - head_started_at
        idx_parse_started_at = time.perf_counter()
        records = parse_idx(idx_text, content_length)
        selected = wanted_records_by_patterns(records, patterns)
        timing_idx_parse_seconds += time.perf_counter() - idx_parse_started_at

    merged_ranges = merged_byte_ranges(selected, max_gap_bytes=max(0, int(range_merge_gap_bytes)))
    hasher = hashlib.sha256()
    timing_range_download_started_at = time.perf_counter()
    for span in merged_ranges:
        hasher.update(download_range(grib_url, span.start, span.end))
    timing_range_download_seconds = time.perf_counter() - timing_range_download_started_at
    timing_total_seconds = time.perf_counter() - started_at

    return BenchmarkRun(
        source=source,
        run_index=run_index,
        grib_url=grib_url,
        idx_url=idx_url,
        selected_record_count=len(selected),
        merged_range_count=len(merged_ranges),
        downloaded_range_bytes=sum(span.byte_length for span in merged_ranges),
        head_used=head_used,
        timing_idx_fetch_seconds=timing_idx_fetch_seconds,
        timing_idx_parse_seconds=timing_idx_parse_seconds,
        timing_head_seconds=timing_head_seconds,
        timing_range_download_seconds=timing_range_download_seconds,
        timing_total_seconds=timing_total_seconds,
        sha256=hasher.hexdigest(),
    )


def default_output_dir() -> Path:
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def write_outputs(output_dir: Path, runs: list[BenchmarkRun]) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [asdict(run) for run in runs]

    csv_path = output_dir / "benchmark_runs.csv"
    json_path = output_dir / "benchmark_runs.json"
    summary_path = output_dir / "benchmark_summary.json"

    if rows:
        header = list(rows[0].keys())
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("")
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    summary_by_source: list[dict[str, object]] = []
    grouped: dict[str, list[BenchmarkRun]] = {}
    for run in runs:
        grouped.setdefault(run.source, []).append(run)
    for source, source_runs in sorted(grouped.items()):
        total_seconds = [run.timing_total_seconds for run in source_runs]
        range_seconds = [run.timing_range_download_seconds for run in source_runs]
        bytes_per_second = [
            (run.downloaded_range_bytes / run.timing_range_download_seconds)
            for run in source_runs
            if run.timing_range_download_seconds > 0
        ]
        summary_by_source.append(
            {
                "source": source,
                "runs": len(source_runs),
                "median_total_seconds": statistics.median(total_seconds),
                "median_range_download_seconds": statistics.median(range_seconds),
                "median_download_bytes_per_second": statistics.median(bytes_per_second) if bytes_per_second else None,
                "sha256_values": sorted({run.sha256 for run in source_runs}),
            }
        )

    sha256_sets = {source: {run.sha256 for run in source_runs} for source, source_runs in grouped.items()}
    sha256_agreement = len({frozenset(values) for values in sha256_sets.values()}) <= 1
    summary = {
        "benchmark_target": {
            "date": runs[0].grib_url.split("/hrrr.", 1)[1].split("/", 1)[0] if runs else None,
            "forecast_hour": runs[0].grib_url.rsplit("wrfsfcf", 1)[1].split(".grib2", 1)[0] if runs else None,
        },
        "sources": summary_by_source,
        "sha256_agreement": sha256_agreement,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    runs: list[BenchmarkRun] = []

    for source in args.sources:
        for run_index in range(1, max(1, int(args.runs)) + 1):
            print(
                f"[benchmark] source={source} run={run_index} date={args.date} "
                f"cycle={args.cycle:02d} forecast_hour={args.forecast_hour:02d}"
            )
            run = benchmark_one_run(
                source=source,
                run_index=run_index,
                date=args.date,
                cycle=args.cycle,
                product=args.product,
                forecast_hour=args.forecast_hour,
                range_merge_gap_bytes=args.range_merge_gap_bytes,
            )
            runs.append(run)
            mb_per_second = (run.downloaded_range_bytes / run.timing_range_download_seconds) / (1024 * 1024)
            print(
                f"[result] source={source} run={run_index} total_seconds={run.timing_total_seconds:.3f} "
                f"range_seconds={run.timing_range_download_seconds:.3f} throughput_mb_s={mb_per_second:.2f} "
                f"selected_records={run.selected_record_count} merged_ranges={run.merged_range_count} sha256={run.sha256[:12]}"
            )

    summary = write_outputs(output_dir, runs)
    print(f"output_dir={output_dir}")
    for item in summary["sources"]:
        print(
            f"source={item['source']} median_total_seconds={item['median_total_seconds']:.3f} "
            f"median_range_download_seconds={item['median_range_download_seconds']:.3f} "
            f"median_download_bytes_per_second={item['median_download_bytes_per_second']}"
        )
    print(f"sha256_agreement={summary['sha256_agreement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
