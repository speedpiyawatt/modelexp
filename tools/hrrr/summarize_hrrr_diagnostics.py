#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_GLOB = "*.manifest.parquet"
METRIC_COLUMNS = (
    "timing_range_download_seconds",
    "timing_wgrib_inventory_seconds",
    "timing_reduce_seconds",
    "timing_cfgrib_open_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize HRRR task diagnostics from manifest parquet outputs."
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
    return sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())


def percentile(series: pd.Series, q: float) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(q))


def summarize_metric(df: pd.DataFrame, column: str) -> dict[str, float | int | None]:
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
    if {"raw_file_size", "reduced_file_size"}.issubset(df.columns):
        ratio_df = df.loc[
            pd.to_numeric(df["raw_file_size"], errors="coerce").gt(0)
            & pd.to_numeric(df["reduced_file_size"], errors="coerce").ge(0)
        ].copy()
        if not ratio_df.empty:
            ratio = pd.to_numeric(ratio_df["reduced_file_size"], errors="coerce") / pd.to_numeric(
                ratio_df["raw_file_size"], errors="coerce"
            )
            summary["size_ratio"] = {
                "count": int(ratio.shape[0]),
                "median_reduced_to_raw": percentile(ratio, 0.5),
                "p95_reduced_to_raw": percentile(ratio, 0.95),
                "median_raw_bytes": percentile(pd.to_numeric(ratio_df["raw_file_size"], errors="coerce"), 0.5),
                "median_reduced_bytes": percentile(pd.to_numeric(ratio_df["reduced_file_size"], errors="coerce"), 0.5),
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
