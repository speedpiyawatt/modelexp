# Server Staging Plan for Overnight Source Builds

## Summary

Add one server-oriented orchestration entrypoint that runs these stages in order:

1. Wunderground table build and validation
2. source-specific smoke tests
3. short-window NBM/LAMP/HRRR builds
4. server-only issue triage and retry classification

The runner should be **staged and stateful**, not a blind full backfill. It must write a machine-readable run summary after each stage and stop before the next stage when a blocking condition is hit. The default server policy should be:

- fail fast on WU contract problems
- run live/source smokes serially
- run short-window source builds serially with conservative worker counts
- classify live network/source-availability issues as `degraded`
- classify contract drift, missing required artifacts, parse failures, and persistent timeouts as `fail`

## Key Changes

### 1. Add a single server orchestration entrypoint
Create one new runner under `tools/weather`, for example `run_server_overnight_stage.py`, with these CLI modes:

- `--stage wu`
- `--stage smoke`
- `--stage short-window`
- `--stage all`

Required arguments:

- `--output-root`
- `--history-dir`
- `--short-window-start`
- `--short-window-end`

Optional arguments with fixed defaults:

- `--probe-timeout-seconds 900`
- `--nbm-workers 1`
- `--hrrr-workers 1`
- `--allow-degraded-live`
- `--resume`

The runner writes:

- `run_summary.json`
- `stage_wu.json`
- `stage_smoke.json`
- `stage_short_window.json`
- one subdirectory per source under the chosen output root

Behavior:

- `wu` runs only WU validation and table build
- `smoke` requires WU stage success
- `short-window` requires WU success and at least non-failing smoke status
- `all` runs all stages in order and stops on first blocking failure

### 2. Freeze the exact stage behavior
WU stage:

- run `python3 wunderground/validate_history.py --history-dir <history_dir>`
- fail if no files, missing-day issues outside recent tolerance, timestamp issues, or non-200 status rows appear
- run `python3 wunderground/build_training_tables.py --history-dir <history_dir> --output-dir <output_root>/wu_tables`
- require both `labels_daily.parquet` and `wu_obs_intraday.parquet`
- require non-empty outputs and unique `target_date_local` rows in labels

Smoke stage:

- run `python3 tools/weather/run_verification_suite.py --output-dir <output_root>/verification --local-only`
- require local-contract lane `pass`
- run `python3 tools/weather/run_live_overnight_probes.py --output-dir <output_root>/live_probes --timeout-seconds <probe_timeout> --lamp-count 1 --nbm-count 1 --hrrr-count 1`
- parse `live_probe_report.json`
- pass criteria:
  - every probe preserves the merged label-spine row
  - merged and normalized audits both pass
  - a source may be `missing_output` and still be acceptable in smoke if the merge/audit layer passes
- degraded criteria:
  - 404s, source unavailability, or network-resolution failures
- fail criteria:
  - builder crashes
  - duplicate keys
  - empty merged/normalized rows for a probed date
  - audit failure
  - timeout on the same source more than once in the same smoke stage

Short-window stage:

- use one fixed validation window, not random dates
- default window:
  - start `2025-04-11`
  - end `2025-04-13`
- if the user later wants broader confidence, extend to a second fixed DST window:
  - `2025-03-08` through `2025-03-10`
- run sources one by one:
  - NBM raw build, then NBM overnight summarizer
  - LAMP fetch/build/overnight summarizer
  - HRRR shard builder with overnight summary output
- after each source build, build temporary merged + normalized overnight rows for just the short window and run both audits
- stage passes only if every source either:
  - produces contract-valid output for at least one date in the window, or
  - is explicitly classified `degraded` with preserved merged rows and passing audits

### 3. Define concrete server triage rules
When a server-only issue appears, do not patch blindly. Classify it first:

- `environment_blocker`
  - missing `wgrib2`
  - missing Python deps
  - filesystem permissions
- `network_or_upstream_degraded`
  - source 404
  - DNS / connection failures
  - source file temporarily missing
- `performance_blocker`
  - repeated timeout with `workers=1`
  - OOM / disk exhaustion
- `contract_bug`
  - missing manifest fields
  - empty output written as success
  - bad timestamp parsing
  - dropped label-spine rows
- `inventory_mapping_bug`
  - ambiguous HRRR or NBM field selection such as `APCP`

Fix policy by class:

- `environment_blocker`: fix on server and rerun same stage immediately
- `network_or_upstream_degraded`: record degraded status, do not patch code, continue only if audits still pass
- `performance_blocker`: increase timeout first, then rerun once; do not raise worker count above `2` in initial server stabilization
- `contract_bug`: stop the staged run and patch code before any broader backfill
- `inventory_mapping_bug`: stop the affected source, capture stderr/stdout and offending manifest/inventory snippet, then patch source selector logic before retry

### 4. Lock concrete source commands and acceptance checks
WU commands:

```bash
python3 wunderground/validate_history.py --history-dir wunderground/output/history
python3 wunderground/build_training_tables.py --history-dir wunderground/output/history --output-dir <output_root>/wu_tables
```

NBM short-window commands:

```bash
python3 tools/nbm/build_grib2_features.py --start-local-date <start> --end-local-date <end> --selection-mode overnight_0005 --output-dir <output_root>/nbm/features --workers 3 --lead-workers 4 --keep-downloads --overwrite
python3 tools/nbm/build_nbm_overnight_features.py --features-root <output_root>/nbm/features --output-dir <output_root>/nbm/overnight --start-local-date <start> --end-local-date <end>
```

NBM notes:

- `overnight_0005` restricts raw issue discovery to the latest issue eligible by the overnight `00:05 America/New_York` cutoff for each target local day
- that is a source-selection change, not just a throughput change
- `processed_timestamp_utc` in raw NBM manifests must represent issue availability time, not wall-clock rebuild time, so historical short-window replays remain cutoff-eligible
- `--lead-workers` improves within-cycle throughput; keep it bounded for server stability
- `--lead-workers` is a general throughput optimization and can still help outside short-window runs; `overnight_0005` is the short-window-specific part

LAMP short-window commands:

```bash
python3 tools/lamp/fetch_lamp.py archive --year 2025 --output-dir <output_root>/lamp/raw
python3 tools/lamp/build_lamp_klga_features.py <output_root>/lamp/raw --output-dir <output_root>/lamp/features
python3 tools/lamp/build_lamp_overnight_features.py --features-root <output_root>/lamp/features --output-dir <output_root>/lamp/overnight --start-local-date <start> --end-local-date <end>
```

HRRR short-window commands:

```bash
python3 tools/hrrr/build_hrrr_klga_feature_shards.py --start-date <start> --end-date <end> --download-dir <output_root>/hrrr/downloads --reduced-dir <output_root>/hrrr/reduced --output-dir <output_root>/hrrr/features --summary-output-dir <output_root>/hrrr/summary --max-workers 1 --allow-partial
```

HRRR notes:

- if using the fast overnight path, add `--selection-mode overnight_0005` to reduce retained cycles for short-window validation
- merged byte-range downloads and deferred month-manifest flushing are general HRRR throughput optimizations and are not limited to short-window runs

Post-source acceptance after each source:

- build merged rows from WU labels/obs plus that source’s daily output
- build normalized rows from merged rows
- require:
  - unique `target_date_local` + `station_id`
  - non-empty rows for the short-window dates covered by WU labels
  - no string default model inputs in normalized output
  - no audit failures
  - explicit availability/coverage flags when the source is missing or incomplete

## Test Plan

Run this exact sequence on the server:

1. `python3 tools/weather/run_verification_suite.py --output-dir <output_root>/verification_local --local-only`
2. WU validate + WU build
3. one-source live probe lane via `tools/weather/run_live_overnight_probes.py` with `1/1/1` counts
4. NBM short-window build and NBM overnight summarizer
5. LAMP short-window fetch/build/overnight summarizer
6. HRRR short-window shard build
7. after each source, run merged + normalized overnight audits for the same short window
8. write final `run_summary.json` with per-stage status `pass|degraded|fail`

Required success conditions:

- WU tables build successfully and are non-empty
- local verification lane passes
- every smoke probe preserves the label spine
- every short-window merged table keeps one row per WU label date
- normalized audit passes on every accepted short-window output
- degraded upstream issues are isolated to source availability, not contract correctness

## Assumptions And Defaults

- server OS is Linux and the repo is run from its root
- WU history JSON files already exist under `wunderground/output/history`; this plan does not include fetching more WU history
- `wgrib2` must be installed and discoverable on `PATH` or via `WGRIB2_BINARY`
- initial server stabilization is serialized with `NBM workers=1` and `HRRR workers=1`
- LAMP short-window uses archive fetches on the server for reproducibility; live LAMP is only for smoke
- degraded live-source status is acceptable only when merged and normalized audits still pass and label rows are preserved
- full-history backfills are out of scope for this stage and begin only after all three stages above pass cleanly
