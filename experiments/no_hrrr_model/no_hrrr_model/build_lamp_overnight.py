from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Build experiment-local LAMP overnight summaries using the canonical selector.")
    parser.add_argument("--features-root", type=pathlib.Path, default=pathlib.Path("tools/lamp/data/runtime/features_full/station_id=KLGA"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("experiments/no_hrrr_model/data/runtime/lamp_overnight"))
    parser.add_argument("--start-local-date", required=True)
    parser.add_argument("--end-local-date", required=True)
    args = parser.parse_args()
    return subprocess.call(
        [
            sys.executable,
            "tools/lamp/build_lamp_overnight_features.py",
            "--features-root",
            str(args.features_root),
            "--output-dir",
            str(args.output_dir),
            "--start-local-date",
            args.start_local_date,
            "--end-local-date",
            args.end_local_date,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

