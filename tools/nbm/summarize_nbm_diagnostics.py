#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_GLOB = "*.parquet"
RAW_MANIFEST_SUFFIX = "_manifest.parquet"
METRIC_COLUMNS = (
    "timing_idx_fetch_seconds",
    "timing_idx_parse_seconds",
    "timing_head_seconds",
    "timing_range_download_seconds",
    "timing_crop_seconds",
    "timing_cfgrib_open_seconds",
    "timing_row_build_seconds",
    "timing_row_geometry_seconds",
    "timing_row_metric_seconds",
    "timing_row_provenance_seconds",
    "timing_cleanup_seconds",
    "cfgrib_open_all_dataset_count",
    "cfgrib_filtered_fallback_open_count",
    "cfgrib_filtered_fallback_attempt_count",
    "cfgrib_opened_dataset_count",
    "wide_row_count",
    "long_row_count",
    "provenance_row_count",
    "selected_download_bytes",
    "downloaded_range_bytes",
    "raw_file_size",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NBM manifest timings and download sizes from parquet outputs."
    )
    parser.add_argument(
        "--path",
        type=Path,
        required=True,
        help="Manifest parquet file or directory to scan recursively.",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_GLOB,
        help="Glob to use when --path is a directory. Default: %(default)s",
    )
    return parser.parse_args()


def discover_manifest_paths(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob(pattern)
        if candidate.is_file()
        and candidate.name.startswith("cycle_")
        and candidate.name.endswith(RAW_MANIFEST_SUFFIX)
    )


def summarize_metric(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    if column not in df.columns:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    clean = pd.to_numeric(df.get(column), errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(clean.shape[0]),
        "median": float(clean.median()),
        "p95": float(clean.quantile(0.95)),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def build_summary(df: pd.DataFrame, manifest_paths: list[Path]) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": "ok",
        "manifest_count": len(manifest_paths),
        "row_count": int(len(df)),
        "metrics": {column: summarize_metric(df, column) for column in METRIC_COLUMNS},
    }
    if {"raw_file_size", "downloaded_range_bytes", "selected_download_bytes"}.issubset(df.columns):
        ratios = df.loc[
            pd.to_numeric(df["raw_file_size"], errors="coerce").gt(0)
            & pd.to_numeric(df["downloaded_range_bytes"], errors="coerce").ge(0)
            & pd.to_numeric(df["selected_download_bytes"], errors="coerce").ge(0)
        ].copy()
        if not ratios.empty:
            downloaded_to_raw = pd.to_numeric(ratios["downloaded_range_bytes"], errors="coerce") / pd.to_numeric(
                ratios["raw_file_size"], errors="coerce"
            )
            selected_to_raw = pd.to_numeric(ratios["selected_download_bytes"], errors="coerce") / pd.to_numeric(
                ratios["raw_file_size"], errors="coerce"
            )
            summary["size_ratio"] = {
                "count": int(ratios.shape[0]),
                "median_downloaded_to_raw": float(downloaded_to_raw.quantile(0.5)),
                "p95_downloaded_to_raw": float(downloaded_to_raw.quantile(0.95)),
                "median_selected_to_raw": float(selected_to_raw.quantile(0.5)),
                "p95_selected_to_raw": float(selected_to_raw.quantile(0.95)),
            }
    return summary


def main() -> int:
    args = parse_args()
    manifest_paths = discover_manifest_paths(args.path, args.glob)
    if not manifest_paths:
        print(json.dumps({"status": "no_manifests", "path": str(args.path)}, sort_keys=True))
        return 1
    frames = [pd.read_parquet(path) for path in manifest_paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    print(json.dumps(build_summary(df, manifest_paths), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
