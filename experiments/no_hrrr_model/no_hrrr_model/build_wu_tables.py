from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Build experiment-local Wunderground tables using the canonical parser.")
    parser.add_argument("--history-dir", type=pathlib.Path, default=pathlib.Path("wunderground/output/history"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("experiments/no_hrrr_model/data/runtime/wunderground"))
    args = parser.parse_args()
    return subprocess.call(
        [
            sys.executable,
            "wunderground/build_training_tables.py",
            "--history-dir",
            str(args.history_dir),
            "--output-dir",
            str(args.output_dir),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())

