#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
from typing import Any


DEFAULT_SERVER = "root@198.199.64.163"
DEFAULT_REMOTE_REPO = "/root/modelexp"
DEFAULT_REMOTE_OUTPUT_ROOT = "data/runtime/server_dual_inference"
ANSI_RESET = "\033[0m"
HIGHLIGHT_STYLES = {
    "no_hrrr": {1: "1;34", 2: "36"},
    "with_hrrr": {1: "1;35", 2: "33"},
}


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"Invalid date {value!r}; expected YYYY-MM-DD") from exc


def run_ssh(server: str, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", server, "bash", "-lc", script],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def prediction_path(run_root: str, model_name: str, target_date: dt.date) -> str:
    return f"{run_root}/predictions/{model_name}/prediction_KLGA_{target_date.isoformat()}.json"


def remote_script(*, target_date: dt.date, remote_repo: str, output_root: str) -> str:
    date = target_date.isoformat()
    run_root = f"{output_root}/{date}"
    no_hrrr_runtime = f"{run_root}/runtime/no_hrrr"
    with_hrrr_runtime = f"{run_root}/runtime/with_hrrr"
    hrrr_run_root = f"{with_hrrr_runtime}/hrrr"
    no_hrrr_prediction_dir = f"{run_root}/predictions/no_hrrr"
    with_hrrr_prediction_dir = f"{run_root}/predictions/with_hrrr"
    no_hrrr_prediction = prediction_path(run_root, "no_hrrr", target_date)
    with_hrrr_prediction = prediction_path(run_root, "with_hrrr", target_date)
    no_hrrr_nbm_download_workers = os.environ.get("MODELEXP_NO_HRRR_NBM_DOWNLOAD_WORKERS", os.environ.get("MODELEXP_NBM_DOWNLOAD_WORKERS", "6"))
    no_hrrr_nbm_reduce_workers = os.environ.get("MODELEXP_NO_HRRR_NBM_REDUCE_WORKERS", os.environ.get("MODELEXP_NBM_REDUCE_WORKERS", "4"))
    no_hrrr_nbm_extract_workers = os.environ.get("MODELEXP_NO_HRRR_NBM_EXTRACT_WORKERS", os.environ.get("MODELEXP_NBM_EXTRACT_WORKERS", "4"))
    no_hrrr_nbm_lead_workers = os.environ.get("MODELEXP_NO_HRRR_NBM_LEAD_WORKERS", os.environ.get("MODELEXP_NBM_LEAD_WORKERS", "8"))
    no_hrrr_nbm_batch_reduce_mode = os.environ.get("MODELEXP_NO_HRRR_NBM_BATCH_REDUCE_MODE", os.environ.get("MODELEXP_NBM_BATCH_REDUCE_MODE", "off"))
    hrrr_max_workers = os.environ.get("MODELEXP_HRRR_MAX_WORKERS", "6")
    hrrr_download_workers = os.environ.get("MODELEXP_HRRR_DOWNLOAD_WORKERS", "6")
    hrrr_reduce_workers = os.environ.get("MODELEXP_HRRR_REDUCE_WORKERS", "2")
    hrrr_extract_workers = os.environ.get("MODELEXP_HRRR_EXTRACT_WORKERS", "2")
    return f"""
set -euo pipefail
cd {shlex.quote(remote_repo)}

DATE={shlex.quote(date)}
RUN_ROOT={shlex.quote(run_root)}
NO_HRRR_RUNTIME={shlex.quote(no_hrrr_runtime)}
WITH_HRRR_RUNTIME={shlex.quote(with_hrrr_runtime)}
HRRR_RUN_ROOT={shlex.quote(hrrr_run_root)}
NO_HRRR_PRED_DIR={shlex.quote(no_hrrr_prediction_dir)}
WITH_HRRR_PRED_DIR={shlex.quote(with_hrrr_prediction_dir)}
NO_HRRR_PRED={shlex.quote(no_hrrr_prediction)}
WITH_HRRR_PRED={shlex.quote(with_hrrr_prediction)}
NEARBY_PREFETCH_ROOT="$WITH_HRRR_RUNTIME/nearby_prefetch"
NEARBY_STATIONS=(KJRB KJFK KEWR KTEB)

mkdir -p "$NO_HRRR_PRED_DIR" "$WITH_HRRR_PRED_DIR"
rm -rf "$NO_HRRR_RUNTIME" "$WITH_HRRR_RUNTIME"
rm -f "$NO_HRRR_PRED" "$WITH_HRRR_PRED"

NEARBY_PIDS=()
for STATION in "${{NEARBY_STATIONS[@]}}"; do
  (
    set -euo pipefail
    HISTORY_DIR="$NEARBY_PREFETCH_ROOT/history/$STATION"
    TABLES_DIR="$NEARBY_PREFETCH_ROOT/tables/$STATION"
    .venv/bin/python wunderground/fetch_daily_history.py \\
      --location-id "$STATION:9:US" \\
      --start-date "$(python3 - <<PY
import datetime as dt
print((dt.date.fromisoformat("$DATE") - dt.timedelta(days=1)).isoformat())
PY
)" \\
      --end-date "$DATE" \\
      --output-dir "$HISTORY_DIR" \\
      --skip-http-status 400 \\
      --skip-http-status 404 \\
      --force
    .venv/bin/python wunderground/build_training_tables.py \\
      --history-dir "$HISTORY_DIR" \\
      --station-id "$STATION" \\
      --output-dir "$TABLES_DIR"
  ) > "$RUN_ROOT/nearby_$STATION.log" 2>&1 &
  NEARBY_PIDS+=("$!")
done

(
  set -euo pipefail
  .venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.run_online_inference \\
    --target-date-local "$DATE" \\
    --runtime-root "$NO_HRRR_RUNTIME" \\
    --prediction-output-dir "$NO_HRRR_PRED_DIR" \\
    --polymarket-event-slug \\
    --nbm-batch-reduce-mode {shlex.quote(no_hrrr_nbm_batch_reduce_mode)} \\
    --nbm-lead-workers {shlex.quote(no_hrrr_nbm_lead_workers)} \\
    --nbm-download-workers {shlex.quote(no_hrrr_nbm_download_workers)} \\
    --nbm-reduce-workers {shlex.quote(no_hrrr_nbm_reduce_workers)} \\
    --nbm-extract-workers {shlex.quote(no_hrrr_nbm_extract_workers)} \\
    --overwrite \\
    --keep-artifacts
) > "$RUN_ROOT/no_hrrr.log" 2>&1 &
NO_HRRR_PID=$!

(
  set -euo pipefail
  .venv/bin/python tools/hrrr/run_hrrr_monthly_backfill.py \\
    --start-local-date "$DATE" \\
    --end-local-date "$DATE" \\
    --run-root "$HRRR_RUN_ROOT" \\
    --selection-mode overnight_0005 \\
    --batch-reduce-mode cycle \\
    --day-workers 1 \\
    --max-workers {shlex.quote(hrrr_max_workers)} \\
    --download-workers {shlex.quote(hrrr_download_workers)} \\
    --reduce-workers {shlex.quote(hrrr_reduce_workers)} \\
    --extract-workers {shlex.quote(hrrr_extract_workers)} \\
    --reduce-queue-size 2 \\
    --extract-queue-size 2 \\
    --range-merge-gap-bytes 65536 \\
    --crop-method auto \\
    --crop-grib-type same \\
    --wgrib2-threads 1 \\
    --extract-method wgrib2-bin \\
    --summary-profile overnight \\
    --skip-provenance \\
    --progress-mode log
) > "$RUN_ROOT/hrrr.log" 2>&1 &
HRRR_PID=$!

set +e
wait "$NO_HRRR_PID"
NO_HRRR_STATUS=$?
wait "$HRRR_PID"
HRRR_STATUS=$?
NEARBY_STATUS=0
for idx in "${{!NEARBY_PIDS[@]}}"; do
  if ! wait "${{NEARBY_PIDS[$idx]}}"; then
    STATION="${{NEARBY_STATIONS[$idx]}}"
    echo "[warn] nearby Wunderground prefetch failed for $STATION; log follows" >&2
    tail -80 "$RUN_ROOT/nearby_$STATION.log" >&2 || true
    NEARBY_STATUS=1
  fi
done
set -e

if [ "$NO_HRRR_STATUS" -ne 0 ]; then
  echo "[error] no-HRRR inference failed; log follows" >&2
  tail -200 "$RUN_ROOT/no_hrrr.log" >&2 || true
  exit "$NO_HRRR_STATUS"
fi
if [ "$HRRR_STATUS" -ne 0 ]; then
  echo "[error] HRRR source build failed; log follows" >&2
  tail -200 "$RUN_ROOT/hrrr.log" >&2 || true
  exit "$HRRR_STATUS"
fi
if [ "$NEARBY_STATUS" -ne 0 ]; then
  echo "[error] one or more nearby Wunderground prefetches failed" >&2
  exit "$NEARBY_STATUS"
fi

EVENT_BINS_PATH=$(find "$NO_HRRR_RUNTIME/polymarket" -path '*/event_bins.json' -print -quit)
if [ -z "$EVENT_BINS_PATH" ]; then
  echo "[error] no-HRRR inference did not produce Polymarket event_bins.json" >&2
  tail -100 "$RUN_ROOT/no_hrrr.log" >&2 || true
  exit 1
fi

NEARBY_OBS_ARGS=()
for STATION in "${{NEARBY_STATIONS[@]}}"; do
  OBS_PATH="$NEARBY_PREFETCH_ROOT/tables/$STATION/wu_obs_intraday.parquet"
  if [ -f "$OBS_PATH" ]; then
    NEARBY_OBS_ARGS+=(--nearby-obs-path "$STATION=$OBS_PATH")
  fi
done

.venv/bin/python -m experiments.withhrrr.withhrrr_model.run_online_inference \\
  --target-date-local "$DATE" \\
  --station-id KLGA \\
  --runtime-root "$WITH_HRRR_RUNTIME" \\
  --prediction-output-dir "$WITH_HRRR_PRED_DIR" \\
  --wunderground-tables-dir "$NO_HRRR_RUNTIME/wunderground_tables/target_date_local=$DATE" \\
  --nbm-root "$NO_HRRR_RUNTIME/nbm/nbm_overnight" \\
  --lamp-root "$NO_HRRR_RUNTIME/lamp_overnight" \\
  --hrrr-root "$HRRR_RUN_ROOT/hrrr_summary" \\
  --event-bins-path "$EVENT_BINS_PATH" \\
  --skip-wunderground \\
  --skip-lamp \\
  --skip-nbm \\
  --skip-hrrr \\
  --skip-nearby-wunderground \\
  "${{NEARBY_OBS_ARGS[@]}}" \\
  --overwrite

rm -rf \\
  "$NO_HRRR_RUNTIME/wunderground_history" \\
  "$NO_HRRR_RUNTIME/wunderground_tables" \\
  "$NO_HRRR_RUNTIME/lamp_raw" \\
  "$NO_HRRR_RUNTIME/lamp_features" \\
  "$NO_HRRR_RUNTIME/lamp_overnight" \\
  "$NO_HRRR_RUNTIME/nbm" \\
  "$NO_HRRR_RUNTIME/polymarket" \\
  "$NO_HRRR_RUNTIME/prediction_features" \\
  "$NEARBY_PREFETCH_ROOT" \\
  "$HRRR_RUN_ROOT"

.venv/bin/python - <<'PY'
import json
from pathlib import Path

date = {date!r}
run_root = Path({run_root!r})
paths = {{
    "no_hrrr": Path({no_hrrr_prediction!r}),
    "with_hrrr": Path({with_hrrr_prediction!r}),
}}
data = {{name: json.loads(path.read_text()) for name, path in paths.items()}}

def bins(payload):
    return {{row["bin"]: float(row["probability"]) for row in payload.get("event_bins", [])}}

no_bins = bins(data["no_hrrr"])
with_bins = bins(data["with_hrrr"])
comparison = {{
    "target_date_local": date,
    "remote_run_root": str(run_root),
    "predictions": {{
        name: {{
            "path": str(paths[name]),
            "status": payload.get("status"),
            "expected_final_tmax_f": payload.get("expected_final_tmax_f"),
            "anchor_tmax_f": payload.get("anchor_tmax_f"),
            "distribution_method": payload.get("distribution_method"),
            "source_disagreement": payload.get("source_disagreement"),
            "ladder_calibration": payload.get("ladder_calibration"),
            "final_tmax_quantiles_f": payload.get("final_tmax_quantiles_f"),
            "event_bins": payload.get("event_bins", []),
        }}
        for name, payload in data.items()
    }},
    "diff_with_hrrr_minus_no_hrrr": {{
        "expected_final_tmax_f": data["with_hrrr"]["expected_final_tmax_f"] - data["no_hrrr"]["expected_final_tmax_f"],
        "anchor_tmax_f": data["with_hrrr"].get("anchor_tmax_f", 0.0) - data["no_hrrr"].get("anchor_tmax_f", 0.0),
        "event_bins": [
            {{"bin": label, "probability_diff": with_bins[label] - no_bins[label]}}
            for label in no_bins
            if label in with_bins
        ],
    }},
}}
comparison_path = run_root / "comparison.json"
comparison_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\\n")
print("RESULT_JSON_BEGIN")
print(json.dumps(comparison, sort_keys=True))
print("RESULT_JSON_END")
PY
"""


def extract_result(stdout: str) -> dict[str, Any]:
    start_marker = "RESULT_JSON_BEGIN"
    end_marker = "RESULT_JSON_END"
    if start_marker not in stdout or end_marker not in stdout:
        raise SystemExit("Remote inference completed without a result JSON block.")
    body = stdout.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
    return json.loads(body)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def color_enabled() -> bool:
    override = os.environ.get("MODELEXP_COLOR", "").strip().lower()
    if override in {"1", "always", "true", "yes", "on"}:
        return True
    if override in {"0", "never", "false", "no", "off"}:
        return False
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def colorize(text: str, style: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\033[{style}m{text}{ANSI_RESET}"


def top_bin_ranks(bin_probs: dict[str, float], *, count: int = 2) -> dict[str, int]:
    ranked = sorted(bin_probs.items(), key=lambda item: (-item[1], item[0]))
    return {label: rank for rank, (label, _) in enumerate(ranked[:count], start=1)}


def format_probability(
    model_name: str,
    probability: float,
    ranks: dict[str, int],
    label: str,
    *,
    enabled: bool,
) -> str:
    rank = ranks.get(label)
    text = f"{probability:6.4f}"
    if rank is None:
        return text
    marker = "1st" if rank == 1 else "2nd"
    text = f"{text} {marker}"
    return colorize(text, HIGHLIGHT_STYLES[model_name][rank], enabled=enabled)


def print_summary(result: dict[str, Any]) -> None:
    predictions = result["predictions"]
    no_hrrr = predictions["no_hrrr"]
    with_hrrr = predictions["with_hrrr"]
    diff = result["diff_with_hrrr_minus_no_hrrr"]
    use_color = color_enabled()
    print(f"date: {result['target_date_local']}")
    print(f"remote_run_root: {result['remote_run_root']}")
    print()
    print("model        expected_f  anchor_f  distribution")
    print(f"no_hrrr      {fmt(no_hrrr['expected_final_tmax_f']):>10}  {fmt(no_hrrr['anchor_tmax_f']):>8}  {no_hrrr['distribution_method']}")
    print(f"with_hrrr    {fmt(with_hrrr['expected_final_tmax_f']):>10}  {fmt(with_hrrr['anchor_tmax_f']):>8}  {with_hrrr['distribution_method']}")
    print(f"diff         {fmt(diff['expected_final_tmax_f']):>10}  {fmt(diff['anchor_tmax_f']):>8}")
    disagreement = with_hrrr.get("source_disagreement")
    if isinstance(disagreement, dict):
        regime = disagreement.get("source_disagreement_regime", "unknown")
        spread = disagreement.get("source_spread_f")
        warmest = disagreement.get("warmest_source", "unknown")
        coldest = disagreement.get("coldest_source", "unknown")
        print(f"with_hrrr regime={regime} spread_f={fmt(spread)} warmest={warmest} coldest={coldest}")
    ladder_calibration = with_hrrr.get("ladder_calibration")
    if isinstance(ladder_calibration, dict) and ladder_calibration.get("method_id"):
        print(f"with_hrrr ladder_adjustment={ladder_calibration.get('method_id')} enabled={ladder_calibration.get('enabled')}")
    print()
    print("event bins")
    no_bins = {row["bin"]: float(row["probability"]) for row in no_hrrr["event_bins"]}
    with_bins = {row["bin"]: float(row["probability"]) for row in with_hrrr["event_bins"]}
    no_ranks = top_bin_ranks(no_bins)
    with_ranks = top_bin_ranks(with_bins)
    print(
        "highlight: "
        f"{colorize('no_hrrr 1st', HIGHLIGHT_STYLES['no_hrrr'][1], enabled=use_color)}, "
        f"{colorize('no_hrrr 2nd', HIGHLIGHT_STYLES['no_hrrr'][2], enabled=use_color)}, "
        f"{colorize('with_hrrr 1st', HIGHLIGHT_STYLES['with_hrrr'][1], enabled=use_color)}, "
        f"{colorize('with_hrrr 2nd', HIGHLIGHT_STYLES['with_hrrr'][2], enabled=use_color)}"
    )
    for label, no_prob in no_bins.items():
        if label not in with_bins:
            continue
        with_prob = with_bins[label]
        no_text = format_probability("no_hrrr", no_prob, no_ranks, label, enabled=use_color)
        with_text = format_probability("with_hrrr", with_prob, with_ranks, label, enabled=use_color)
        print(f"{label:<18} no_hrrr={no_text:<10}  with_hrrr={with_text:<10}  diff={with_prob - no_prob:+7.4f}")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: tools/weather/run_server_dual_inference.py YYYY-MM-DD", file=sys.stderr)
        return 2
    target_date = parse_date(sys.argv[1])
    server = os.environ.get("MODELEXP_SERVER", DEFAULT_SERVER)
    remote_repo = os.environ.get("MODELEXP_REMOTE_REPO", DEFAULT_REMOTE_REPO)
    output_root = os.environ.get("MODELEXP_REMOTE_OUTPUT_ROOT", DEFAULT_REMOTE_OUTPUT_ROOT)

    completed = run_ssh(server, remote_script(target_date=target_date, remote_repo=remote_repo, output_root=output_root))
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        return completed.returncode
    result = extract_result(completed.stdout)
    print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
