#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = Path("data/runtime/download_speed_benchmarks")


def default_date_utc() -> str:
    # Use yesterday UTC so all forecast-hour files are likely published on mirrors.
    return (dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)).strftime("%Y%m%d")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portable NBM and HRRR public-source download benchmarks."
    )
    parser.add_argument(
        "--date",
        default=default_date_utc(),
        help="UTC initialization date, YYYYMMDD or YYYY-MM-DD. Default: yesterday UTC.",
    )
    parser.add_argument("--runs", type=int, default=2, help="Measured runs per source.")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="NBM warmup runs per source. HRRR benchmark does not have warmups.",
    )
    parser.add_argument("--nbm-cycle", default="04", help="NBM UTC cycle hour.")
    parser.add_argument("--nbm-forecast-hour", type=int, default=12, help="NBM forecast hour.")
    parser.add_argument("--nbm-region", default="co", help="NBM region code, usually co for CONUS.")
    parser.add_argument(
        "--nbm-sources",
        default="aws,noaa_https,noaa_ftp",
        help="Comma-separated NBM source ids.",
    )
    parser.add_argument("--hrrr-cycle", type=int, default=4, help="HRRR UTC cycle hour.")
    parser.add_argument("--hrrr-forecast-hour", type=int, default=12, help="HRRR forecast hour.")
    parser.add_argument(
        "--hrrr-sources",
        nargs="+",
        default=["google", "azure"],
        help="HRRR source ids.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "nbm", "hrrr"),
        default="all",
        help="Limit the benchmark to one product.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Default: timestamped data/runtime/download_speed_benchmarks run.",
    )
    return parser.parse_args()


def normalize_date_for_nbm(value: str) -> str:
    if "-" in value:
        return value
    return dt.datetime.strptime(value, "%Y%m%d").strftime("%Y-%m-%d")


def normalize_date_for_hrrr(value: str) -> str:
    if "-" not in value:
        return value
    return dt.datetime.strptime(value, "%Y-%m-%d").strftime("%Y%m%d")


def default_output_dir() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_ROOT / stamp


def run_command(label: str, command: list[str]) -> int:
    print(f"[{label}] command={' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    print(f"[{label}] exit_code={completed.returncode}", flush=True)
    return completed.returncode


def read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def print_combined_summary(output_dir: Path) -> None:
    print(f"output_dir={output_dir}")

    nbm_summary = read_json(output_dir / "nbm" / "benchmark_summary.json")
    if isinstance(nbm_summary, dict):
        print("[nbm]")
        ranking = nbm_summary.get("ranked_fastest_by_median_wall_time") or []
        for item in ranking:
            if not isinstance(item, dict):
                continue
            throughput = item.get("mean_throughput_mbps")
            throughput_text = f"{float(throughput):.2f}MiB/s" if throughput is not None else "n/a"
            print(
                f"source={item.get('source')} "
                f"median_seconds={float(item.get('median_wall_time_seconds')):.3f} "
                f"mean_throughput={throughput_text}"
            )
        print(
            f"benchmark_valid={nbm_summary.get('benchmark_valid')} "
            f"sha256_status={nbm_summary.get('sha256_agreement_status')}"
        )

    hrrr_summary = read_json(output_dir / "hrrr" / "benchmark_summary.json")
    if isinstance(hrrr_summary, dict):
        print("[hrrr]")
        for item in hrrr_summary.get("sources") or []:
            if not isinstance(item, dict):
                continue
            bps = item.get("median_download_bytes_per_second")
            mib = (float(bps) / (1024 * 1024)) if bps is not None else None
            throughput_text = f"{mib:.2f}MiB/s" if mib is not None else "n/a"
            print(
                f"source={item.get('source')} "
                f"median_total_seconds={float(item.get('median_total_seconds')):.3f} "
                f"median_range_seconds={float(item.get('median_range_download_seconds')):.3f} "
                f"median_throughput={throughput_text}"
            )
        print(f"sha256_agreement={hrrr_summary.get('sha256_agreement')}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    exit_codes: list[int] = []
    if args.only in {"all", "nbm"}:
        exit_codes.append(
            run_command(
                "nbm",
                [
                    sys.executable,
                    "tools/nbm/benchmark_nbm_sources.py",
                    "--date",
                    normalize_date_for_nbm(args.date),
                    "--cycle",
                    str(args.nbm_cycle),
                    "--forecast-hour",
                    str(args.nbm_forecast_hour),
                    "--region",
                    str(args.nbm_region),
                    "--sources",
                    str(args.nbm_sources),
                    "--runs",
                    str(args.runs),
                    "--warmup-runs",
                    str(args.warmup_runs),
                    "--output-dir",
                    str(output_dir / "nbm"),
                ],
            )
        )

    if args.only in {"all", "hrrr"}:
        exit_codes.append(
            run_command(
                "hrrr",
                [
                    sys.executable,
                    "tools/hrrr/benchmark_hrrr_sources.py",
                    "--date",
                    normalize_date_for_hrrr(args.date),
                    "--cycle",
                    str(args.hrrr_cycle),
                    "--forecast-hour",
                    str(args.hrrr_forecast_hour),
                    "--sources",
                    *[str(source) for source in args.hrrr_sources],
                    "--runs",
                    str(args.runs),
                    "--output-dir",
                    str(output_dir / "hrrr"),
                ],
            )
        )

    print_combined_summary(output_dir)
    return 1 if any(code != 0 for code in exit_codes) else 0


if __name__ == "__main__":
    raise SystemExit(main())
