# HRRR KLGA Feature Pipeline

This directory contains the HRRR feature pipeline for the KLGA settlement model.

The pipeline is not a generic archive mirror. It is a settlement-centered forecast feature builder with these fixed assumptions:

- settlement station: `KLGA`
- local timezone: `America/New_York`
- target use: modeling the finalized local-day high temperature at LaGuardia
- persistent outputs from the raw builder: monthly wide Parquet, monthly provenance Parquet, monthly manifest Parquet, and an auxiliary JSON month manifest

The main production workflow is still:

1. download full hourly HRRR surface GRIB2 files
2. subset and crop them locally with `wgrib2`
3. extract KLGA-centered features with `xarray` + `cfgrib`
4. write monthly Parquet outputs
5. delete raw and reduced GRIB files by default

Low-disk daily resume-from-disk backfill helper:

- `tools/hrrr/run_hrrr_monthly_backfill.py`

This wrapper runs `build_hrrr_klga_feature_shards.py` in one-target-day child jobs, scans existing daily outputs on every run, skips valid days already on disk, rebuilds invalid days, and deletes each successful day temp tree so only daily `hrrr_summary` and `hrrr_summary_state` outputs remain under the chosen run root. `--day-workers` controls how many target days are admitted at once; the default `1` preserves serial behavior.

Example:

```bash
source .venv/bin/activate
python tools/hrrr/run_hrrr_monthly_backfill.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --day-workers 4 \
  --max-workers 8 \
  --download-workers 8 \
  --reduce-workers 2 \
  --extract-workers 3 \
  --progress-mode dashboard \
  --pause-control-file data/runtime/backfill_overnight_2023_2026/hrrr.pause
```

Rerun/debug throughput profile:

```bash
source .venv/bin/activate
python tools/hrrr/run_hrrr_monthly_backfill.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --day-workers 4 \
  --max-workers 8 \
  --download-workers 8 \
  --reduce-workers 2 \
  --extract-workers 3 \
  --progress-mode dashboard \
  --keep-reduced
```

Use `--keep-reduced` when you want to preserve reduced GRIB2 artifacts for reruns or debugging. Default low-disk runs should continue deleting reduced files after successful extraction.

Monthly-helper operational notes:

- the parent monthly runner owns dashboard rendering and pause/drain
- when the parent dashboard is active, child raw builders are forced to `--progress-mode log`
- parent `--pause-control-file`, dashboard `p`, SIGINT, and SIGTERM stop new day admission and drain already admitted days
- the parent does not forward its `--pause-control-file` to child builders during normal monthly processing
- each completed target day writes `hrrr.performance.json` beside `hrrr.manifest.json` under `hrrr_summary_state/target_date_local=YYYY-MM-DD/`
- existing valid daily outputs remain skippable even if older runs do not have `hrrr.performance.json`

Diagnostics summary helper:

```bash
source .venv/bin/activate
python tools/hrrr/summarize_hrrr_diagnostics.py \
  --path data/runtime/backfill_overnight_2023_2026
```

This summarizes medians and p95 values for download, inventory, reduce, and cfgrib-open timings, plus reduced-to-raw size ratios when those diagnostics are available in manifest parquet outputs.

Server-owned selected-raw relay queue:

- `tools/hrrr/relay_server.py`

Use this when a Linux server can download HRRR selected raw subsets quickly, but a Mac or laptop should do the expensive `wgrib2` / `cfgrib` processing. The server remains the durable source of truth: it owns the SQLite queue, selected raw task directories, accepted per-task results, and final daily `hrrr_summary` plus `hrrr_summary_state` outputs. Processor clients are disposable and should only lease work, process local temp files, upload compact result artifacts, and acknowledge or retry through the server CLI.

Server runner example:

```bash
source .venv/bin/activate
python tools/hrrr/relay_server.py run \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --selection-mode overnight_0005 \
  --prepare-workers 8 \
  --target-ready-tasks 64 \
  --prepare-batch-size 16 \
  --min-free-gb 10 \
  --progress-mode dashboard \
  --pause-control-file /tmp/hrrr-relay.pause
```

The runner initializes the requested range, keeps a capped queue of `ready` selected-raw tasks, periodically recovers expired work, displays progress, and exits when all requested days finalize. The ready cap is intentional: selected raw subsets still consume server disk, so avoid preparing the entire history unless the server has enough space.

Client processing example:

```bash
source .venv/bin/activate
python tools/hrrr/relay_client.py run \
  --server s_01kp5eg1qmt0bsapwjsdz39qac@ssh.lightning.ai \
  --remote-repo /path/to/modelexp \
  --remote-run-root data/runtime/backfill_overnight_2023_2026 \
  --local-work-root data/runtime/hrrr_relay_client \
  --client-id macbook-m4 \
  --workers 4 \
  --min-free-gb 5 \
  --lease-minutes 30 \
  --progress-mode dashboard \
  --pause-control-file /tmp/hrrr-client.pause
```

Operator commands:

```bash
python tools/hrrr/relay_server.py status --run-root data/runtime/backfill_overnight_2023_2026
python tools/hrrr/relay_server.py day-status --run-root data/runtime/backfill_overnight_2023_2026 --target-local-date 2024-01-05
python tools/hrrr/relay_server.py pause --run-root data/runtime/backfill_overnight_2023_2026
python tools/hrrr/relay_server.py resume --run-root data/runtime/backfill_overnight_2023_2026
python tools/hrrr/relay_server.py recover-expired --run-root data/runtime/backfill_overnight_2023_2026
```

The relay queue stores state under:

```text
<run-root>/hrrr_relay/
  queue.sqlite
  pause.flag
  raw/<task_id>/
  uploads/<task_id>/
  results/<task_id>/
```

Each task is one selected raw HRRR subset keyed by the existing `TaskSpec.key`. The server writes `task.json`, `raw.grib2`, `raw.manifest.csv`, `raw.selection.csv`, `checksum.sha256`, and `metadata.json` under `raw/<task_id>/`. A client lease response returns the exact paths and checksum the client should use; clients should not infer path names.

Remote path rule:

- server commands run as `cd <remote-repo> && .venv/bin/python tools/hrrr/relay_server.py ...`, so `--remote-run-root data/runtime/...` is intentionally interpreted relative to the remote repo
- `rsync` and remote `mkdir` do not run under that `cd`, so the client resolves relative lease paths by prefixing `--remote-repo` before transferring files
- keep the original lease paths for server CLI calls such as `ack`; only transfer commands should use the resolved remote filesystem path

The client stores disposable local state under:

```text
<local-work-root>/
  active/<task_id>/
    lease.json
    raw/
    reduced/
    scratch/
    upload/
  done/<task_id>.json
  failed/<task_id>.json
```

Inspect local client state without contacting the server:

```bash
python tools/hrrr/relay_client.py list-local \
  --local-work-root data/runtime/hrrr_relay_client
```

Relay state and recovery rules:

- `pending` and `retry_pending` tasks are eligible for `prepare`
- `prepare` downloads the selected raw subset and moves the task to `ready`
- `run` is the preferred server entrypoint; it wraps `init`, bounded concurrent preparation, expired-work recovery, progress reporting, and pause/drain in one long-running command
- `lease` atomically assigns one `ready` task to one client
- `ack` accepts only the current lease owner, validates uploaded row metadata against the leased task, promotes results, rechecks ownership before marking the task `completed`, deletes the raw task directory, and then attempts day finalization
- `retry` accepts only the current lease owner and only for `leased` tasks
- `recover-expired` requeues expired `leased` tasks plus stale `preparing` and stale `result_uploaded` tasks, so interrupted server/client processes do not strand work
- re-running `init` for an existing day reconciles stale task rows and artifacts if the planned task set or selection mode changes, so finalization counts only the current expected tasks
- `pause` blocks new preparation and new leases; already leased tasks can still be acknowledged
- in `run`, dashboard `p` or `--pause-control-file` requests pause, stops new preparation, drains already admitted prepare work, then marks safe-to-exit
- `--target-ready-tasks` controls server prefetch depth and `--min-free-gb` stops new preparation when disk is low without blocking client ack/retry/status commands
- client `run` has its own dashboard/pause controls; dashboard `p`, `--pause-control-file`, SIGINT, or SIGTERM stop new leases while allowing active leased tasks to finish upload/ack or retry
- client progress reports phase timing (`lease`, `pull`, `verify`, `process`, `upload`, `ack`, `done`, `retry`, `ack_ambiguous`) but does not parse rsync byte progress in v1
- client `list-local` reports disposable local state, including preserved ack-ambiguous uploads
- if a client has uploaded artifacts but cannot determine whether `ack` succeeded, it records an `ack_ambiguous` local failure marker and preserves `active/<task_id>/upload/` instead of calling `retry`
- if finalization fails after an `ack`, the task remains completed and the command reports `finalize.status=error` while storing the day error for later inspection

When all tasks for a target local date are completed, the server atomically writes the normal daily outputs:

```text
<run-root>/hrrr_summary/target_date_local=YYYY-MM-DD/hrrr.overnight.parquet
<run-root>/hrrr_summary_state/target_date_local=YYYY-MM-DD/hrrr.manifest.json
<run-root>/hrrr_summary_state/target_date_local=YYYY-MM-DD/hrrr.manifest.parquet
```

## What To Run First

If you are new to the weather stack, start here:

1. raw HRRR extraction
   - `tools/hrrr/build_hrrr_klga_feature_shards.py`
2. canonical training normalization
   - `tools/weather/normalize_training_features.py`
3. local verification
   - `python3 tools/weather/run_verification_suite.py --local-only`

## Standardization Layers

HRRR is one of the two v1 standardized live runtime weather builders in this repo.

The standard now has three layers:

- raw runtime source contract
- canonical training schema
- operational metadata contract

That means raw HRRR output is intentionally source-aware. It should not be forced to become byte-for-byte identical to raw NBM wide output.

### Raw Runtime Source Contract

### Source Of Truth For Spatial Semantics

The canonical location and spatial-context logic now lives in:

- `tools/weather/location_context.py`

That shared module defines:

- settlement location
- crop bounds
- longitude normalization
- nearest-grid lookup
- `nb3` and `nb7` neighborhood metrics
- crop-wide metrics

HRRR uses that shared module directly. NBM uses the same shared implementation.

### Canonical Training Schema

The model-facing normalized schema now lives in:

- `tools/weather/canonical_feature_schema.py`
- `tools/weather/normalize_training_features.py`

Training code should consume those canonical normalized outputs instead of assuming raw HRRR and raw NBM wide tables share the same meteorological base names.

### Operational Metadata Contract

Provenance is standardized enough to answer where a feature came from. Manifests remain operational artifacts and do not need to become structurally identical across sources.

## What Is Committed vs Not Committed

Committed regression fixture:

- `tools/hrrr/data/fixtures/contract_smoke`

Generated runtime output:

- `tools/hrrr/data/runtime/...`

Ignored local scratch/output:

- `tools/hrrr/out`
- tool-local `data/runtime/...`

Do not treat the committed fixture as a production output directory. It exists to support contract tests and review tooling.

### Settlement And Crop Contract

The weather-feature contract is anchored to:

- station id: `KLGA`
- settlement latitude: `40.7769`
- settlement longitude: `-73.8740`
- settlement longitude in `0-360`: `286.1260`

The regional crop is:

- top latitude: `43.5`
- bottom latitude: `39.0`
- left longitude: `282.5`
- right longitude: `289.5`

### Canonical Spatial Semantics

For each feature family, HRRR computes:

- nearest-point value
- crop-wide mean
- crop-wide min
- crop-wide max
- crop-wide std
- `nb3` mean
- `nb3` min
- `nb3` max
- `nb3` std
- `nb3` gradient west-east
- `nb3` gradient south-north
- `nb7` mean
- `nb7` min
- `nb7` max
- `nb7` std
- `nb7` gradient west-east
- `nb7` gradient south-north

Canonical naming follows the NBM-style schema:

- nearest-point value uses the bare field prefix, for example `tmp_2m_k`
- crop context uses `_crop_mean`, `_crop_min`, `_crop_max`, `_crop_std`
- neighborhood context uses `_nb3_*` and `_nb7_*`

Concrete examples:

```text
tmp_2m_k
tmp_2m_k_crop_mean
tmp_2m_k_crop_min
tmp_2m_k_nb3_mean
tmp_2m_k_nb3_gradient_west_east
tmp_2m_k_nb7_mean
ugrd_10m_ms
ugrd_10m_ms_nb3_std
tcdc_entire_pct_crop_max
pwat_entire_atmosphere_kg_m2_nb7_mean
```

### Compatibility Aliases

HRRR no longer writes transition aliases in the default canonical wide output.

Compatibility aliases are available only when `--write-legacy-aliases` is enabled:

- `_nearest`
- `3x3`
- `7x7`

Examples:

- `tmp_2m_k_nearest`
- `tmp_2m_k_3x3_mean`
- `tmp_2m_k_7x7_mean`

Those aliases are compatibility-only. Canonical downstream code should use the `nb3` / `nb7` schema, and canonical compatibility checks should treat the alias columns as non-standard.

### Canonical Identity Columns

Every HRRR wide row is guaranteed to contain:

- `source_model`
- `source_product`
- `source_version`
- `fallback_used_any`
- `station_id`
- `forecast_hour`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`

Current values are:

- `source_model = "HRRR"`
- `source_product = "wrfsfcf"`
- `source_version = "hrrr-conus-wrfsfcf-public"`
- `fallback_used_any = False` for the current HRRR implementation because derived features are not treated as fallback mappings

### Canonical Spatial Metadata Columns

Every HRRR wide row is also guaranteed to contain:

- `settlement_lat`
- `settlement_lon`
- `crop_top_lat`
- `crop_bottom_lat`
- `crop_left_lon`
- `crop_right_lon`
- `nearest_grid_lat`
- `nearest_grid_lon`

Canonical longitude semantics follow the NBM contract:

- `nearest_grid_lon` uses WGS84-style `-180..180`
- `grid_lon` and `target_lon` may remain in the native `0-360` convention as compatibility metadata

Useful extra columns such as `task_key`, `run_date_utc`, `cycle_hour_utc`, `grid_row`, and `grid_col` remain available, but they are secondary to the canonical join contract above.

### Field Families

The current HRRR hourly surface build keeps these feature families when available.

Near surface:

- `tmp_2m_k`
- `dpt_2m_k`
- `rh_2m_pct`

Wind, pressure, visibility:

- `ugrd_10m_ms`
- `vgrd_10m_ms`
- `gust_surface_ms`
- `surface_pressure_pa`
- `mslma_pa`
- `visibility_m`

Cloud, radiation, precipitation:

- `lcdc_low_pct`
- `mcdc_mid_pct`
- `hcdc_high_pct`
- `tcdc_entire_pct`
- `dswrf_surface_w_m2`
- `dlwrf_surface_w_m2`
- `apcp_surface_kg_m2`
- `prate_surface_kg_m2_s`

Boundary layer and column:

- `hpbl_m`
- `pwat_entire_atmosphere_kg_m2`
- `cape_surface_j_kg`
- `cin_surface_j_kg`
- `refc_entire_atmosphere`
- `ltng_entire_atmosphere`

Upper-air levels at `1000`, `925`, `850`, and `700 mb`:

- `tmp_*mb_k`
- `ugrd_*mb_ms`
- `vgrd_*mb_ms`
- `rh_*mb_pct`
- `spfh_*mb_kg_kg`
- `hgt_*mb_gpm`

### Derived Feature Families

HRRR also writes derived feature families when source fields are available:

- Fahrenheit temperature/dewpoint families, for example `tmp_2m_f`
- wind speed families, for example `wind_10m_speed_ms`
- wind speed in mph, for example `wind_10m_speed_mph`
- wind direction families, for example `wind_10m_direction_deg`
- upper-air RH families derived from temperature and dew point when direct RH is absent

The derived families use the same canonical suffix contract:

- bare nearest value
- `_crop_*`
- `_nb3_*`
- `_nb7_*`

### Missing Optional Semantics

Optional field absence does not fail a task by itself.

Wide rows expose:

- `missing_optional_any`
- `missing_optional_fields_count`

The month manifest records missing optional fields by task, and the provenance artifact records missing feature families explicitly.

### Provenance Contract

Each month now includes a provenance artifact:

- `YYYY-MM.provenance.parquet`

It is row-granular at:

- one row per `task_key` x `feature_name`

Each provenance row describes where that feature family came from, including:

- source model, product, and source version
- canonical row identity timestamps
- forecast hour
- canonical nearest-grid identity
- feature name
- GRIB short name
- level text
- type of level
- step type
- step text
- original inventory line when direct
- units
- whether the feature was directly present
- whether the feature was derived
- derivation method
- source feature names used for derivation
- whether the feature family was missing optional data
- whether fallback mapping was used
- fallback source description
- notes

This is the audit trail that makes HRRR and NBM comparable in one downstream feature ecosystem.

## Current Retained HRRR Task Policy

The current HRRR monthly builder uses a local-day overnight slice anchored to `America/New_York`.

Planning unit:

- one `target_date_local`

Retained cycles:

- local init times from previous local day `18:00` through target day `00:00`
- only cycles available by target day `00:05` local
- retain the latest cycle in that window that can cover the target day through `21:00` local
- retain the newest `3` additional cycles for revision features
- total retained cycle count is normally `4`

Retained forecast rows:

- only valid times whose `valid_time_local` falls on the `target_date_local`
- only valid hours from `00:00` through `21:00` local

Optimization scope:

- `--selection-mode overnight_0005` is an overnight-validation and short-window replay optimization because it reduces the retained cycle set
- `--range-merge-gap-bytes` is a general fetch-layer optimization and lets HRRR trade a small amount of extra download size for fewer HTTP Range requests
- conditional `HEAD` requests avoid one remote metadata call when the selected records do not need the terminal file size
- deferred month-manifest flushing is a general month-builder optimization and applies beyond short-window runs

Execution controls:

- `--max-workers` sets the outer retained-task concurrency
- `--download-workers`, `--reduce-workers`, and `--extract-workers` optionally cap each pipeline phase independently
- `--reduce-queue-size` and `--extract-queue-size` bound the stage handoff queues
- `--progress-mode auto|dashboard|log` selects either the live terminal dashboard or structured progress logs
- in `--progress-mode dashboard`, press `p` to request a graceful drain-and-pause; once the dashboard shows `PAUSED`, it is safe to exit and rerun later
- `--disable-dashboard-hotkeys` disables interactive dashboard hotkeys without changing dashboard rendering
- `--pause-control-file /tmp/hrrr.pause` enables a non-TTY pause trigger; create that file with `touch /tmp/hrrr.pause` from another shell to request the same graceful drain-and-pause flow
- `--max-task-attempts`, `--retry-backoff-seconds`, and `--retry-max-backoff-seconds` control automatic task retries with backoff
- `--allow-partial` lets a degraded month finish with exit code `0` when some tasks fail
- `--limit-tasks` is the intended smoke-test throttle
- `--batch-reduce-mode cycle` groups downloaded forecast-hour subsets by retained HRRR cycle, concatenates them, crops once, opens the multi-step reduced GRIB, and extracts task rows from the matching forecast step; `off` keeps the legacy per-task crop/open path

The builder now runs a staged pipeline instead of a single uniform worker pool:

1. download and subset selected HRRR records
2. reduce them to the NYC crop
3. open/extract feature rows and summary rows

The dashboard/log view reports retained-cycle counts, phase-specific worker counts, queue depths, transfer speed, retry countdowns, and recovered tasks. Use `--progress-mode log` for CI, redirected output, or long append-only runs. In dashboard mode, `p` requests a graceful drain-and-pause instead of cancelling in-flight work. For remote terminals that do not deliver raw keypresses reliably, use `--pause-control-file` and trigger pause with `touch`.

The monthly helper has its own day-level orchestration controls:

- `--day-workers` parallelizes target days; default `1` preserves serial behavior
- `--progress-mode dashboard` renders one parent-owned monthly dashboard for all active target days
- child raw builders are forced to `--progress-mode log` under the parent dashboard so only one TUI owns the terminal
- the monthly runner owns pause/drain; `p`, `--pause-control-file`, SIGINT, or SIGTERM stop new day admission and drain already admitted days
- the monthly runner does not forward its parent pause-control file into child builders during normal monthly operation
- `hrrr.performance.json` is written beside each per-day manifest under `hrrr_summary_state/target_date_local=YYYY-MM-DD/`
- The monthly helper forwards `--batch-reduce-mode` to the raw HRRR builder.

Outputs:

- hourly/raw overnight-selected Parquet and provenance artifacts
- one monthly overnight summary Parquet with one row per `target_date_local`

This is a scheduling and data-retention choice, not the schema contract.

The row-join contract is driven by timestamps and spatial metadata, not by HRRR-specific scheduling semantics.

HRRR does **not** copy NBM’s `premarket` / `intraday` scheduler model. Joinability comes from the canonical row identity and time fields instead.

## Artifact Policy

HRRR remains Parquet-first.

Persistent outputs:

- `YYYY-MM.parquet`
- `YYYY-MM.provenance.parquet`
- `YYYY-MM.manifest.parquet`
- `YYYY-MM.manifest.json`
- sibling overnight summary shard under the summary output directory

Temporary outputs while a month is in progress:

- `YYYY-MM.rows.jsonl`
- `YYYY-MM.provenance.rows.jsonl`

Temporary processing artifacts:

- raw full GRIB2 downloads under `data/runtime/downloads`
- reduced regional GRIB2 files under `data/runtime/reduced`

Default behavior:

- raw downloads are deleted after successful extraction
- reduced GRIB2 files are deleted after successful extraction
- only wide Parquet, provenance Parquet, manifest Parquet, and the auxiliary JSON manifest remain

Debug-only flags:

- `--keep-downloads`
- `--keep-reduced`
- `--write-legacy-aliases`

Those flags exist for inspection and replay. They are not the production default.

### Month Manifest Description

`YYYY-MM.manifest.parquet` is the canonical month-level manifest artifact.

`YYYY-MM.manifest.json` remains as an auxiliary state description for operational resume/debugging.

The manifest surfaces record:

- expected task count
- expected task keys
- completed task keys
- failure reasons
- missing fields
- wide parquet path
- provenance path
- manifest parquet path
- row-buffer paths
- keep-flag policy
- month completion status
- per-task fetch/extract diagnostics such as `head_used`, selected/downloaded byte counts, and stage timings

Key timing fields currently include:

- `timing_idx_fetch_seconds`
- `timing_idx_parse_seconds`
- `timing_head_seconds`
- `timing_range_download_seconds`
- `timing_wgrib_inventory_seconds`
- `timing_reduce_seconds`
- `timing_cfgrib_open_seconds`
- `timing_row_build_seconds`
- `timing_cleanup_seconds`

It is not a feature table. It is the description of the month build.

The committed files under `tools/hrrr/data/fixtures/contract_smoke/` are regression fixtures for this live contract. Regenerate them whenever the runtime schema changes so review tooling does not drift behind the builder.

## Implementation Notes

### Main Entry Point

The production entrypoint is:

```bash
python3 build_hrrr_klga_feature_shards.py
```

Useful performance tuning flag:

```bash
python3 build_hrrr_klga_feature_shards.py \
  --range-merge-gap-bytes 65536
```

Useful execution-control pattern:

```bash
python3 build_hrrr_klga_feature_shards.py \
  --max-workers 4 \
  --download-workers 4 \
  --reduce-workers 3 \
  --extract-workers 4 \
  --progress-mode dashboard
```

### Pipeline Shape

For each retained file, the pipeline does:

1. build the Google HRRR URL for the hourly surface file
2. fetch only the required GRIB records with `.idx`, conditional `HEAD`, and coalesced HTTP byte-range requests
3. validate/reuse the cached subset only when its selection manifest matches the current inventory contract
4. inspect the subset inventory with `wgrib2`
5. crop to the NYC box
6. open the reduced GRIB2 with grouped `cfgrib` reads backed by reusable per-file temporary indexes
7. compute canonical nearest, crop, `nb3`, and `nb7` metrics
8. write monthly wide and provenance outputs
9. build one overnight summary row per local target day
10. delete temporary GRIB files unless keep flags are enabled

The coalesced range fetch path reduces HTTP round trips without changing selected-record manifest identity. The month builder treats the JSONL row/provenance buffers as the mid-run checkpoint layer, writes them once per completed task rather than once per row, and defers manifest rewrites until finalization or incomplete-month exit.

### Shared Spatial Logic

The builder no longer defines station/crop/window logic inline. Spatial semantics come from:

- `tools/weather/location_context.py`

That keeps HRRR and NBM aligned on:

- settlement point
- crop box
- nearest-grid semantics
- neighborhood calculations
- crop statistics

### Runtime Requirements

The production pipeline requires:

- Python 3
- `wgrib2`
- `eccodes`
- `cfgrib`
- `xarray`
- `numpy`
- `pandas`
- `pyarrow`

Install Python packages with:

```bash
python3 -m pip install eccodes cfgrib pyarrow xarray numpy pandas
```

Run from:

```bash
cd tools/hrrr
```

### Example Commands

Full backfill:

```bash
python3 build_hrrr_klga_feature_shards.py \
  --start-date 2023-01-01 \
  --end-date 2026-04-12 \
  --download-dir data/runtime/downloads \
  --reduced-dir data/runtime/reduced \
  --output-dir data/runtime/features/hourly_surface_klga \
  --summary-output-dir data/runtime/features/overnight_summary_klga \
  --max-workers 4 \
  --download-workers 4 \
  --reduce-workers 3 \
  --extract-workers 4
```

One-day production-path test:

```bash
python3 build_hrrr_klga_feature_shards.py \
  --start-date 2023-01-01 \
  --end-date 2023-01-01 \
  --download-dir data/runtime/daytest/downloads \
  --reduced-dir data/runtime/daytest/reduced \
  --output-dir data/runtime/daytest/features \
  --summary-output-dir data/runtime/daytest/summary \
  --max-workers 2 \
  --progress-mode log
```

One-task smoke test:

```bash
python3 build_hrrr_klga_feature_shards.py \
  --start-date 2025-04-11 \
  --end-date 2025-04-11 \
  --download-dir data/runtime/smoke/downloads \
  --reduced-dir data/runtime/smoke/reduced \
  --output-dir data/runtime/smoke/features \
  --summary-output-dir data/runtime/smoke/summary \
  --max-workers 1 \
  --progress-mode log \
  --limit-tasks 1
```

## Size Findings And Test Results

Measured raw-file size sample:

- one sampled hourly `wrfsfcf` file: `137,221,390` bytes
- about `131 MiB`

Observed one-day live test for `2023-01-01` under the older fixed-UTC slice:

- retained tasks: `49`
- completed tasks: `49`
- failed tasks: `0`
- finished wide rows: `49`
- finished wide columns in that test shard: `651`
- final Parquet size: `620,673` bytes
- about `606.13 KiB`

That observed one-day Parquet output was about:

- `221.1x` smaller than the sampled full raw HRRR file size above

Observed cleanup result for that one-day test:

- `0` files left in the temporary download directory after completion
- `0` files left in the temporary reduced directory after completion

This is why HRRR keeps the Parquet and provenance outputs as the product while treating GRIB files as temporary processing artifacts.

## Legacy Helpers

### `fetch_hrrr_records.py`

This helper still exists for subset-GRIB workflows that use `.idx` plus HTTP range requests.

Current behavior:

- counts successes and failures across requested forecast hours
- exits non-zero if any requested forecast hour was skipped
- writes subset GRIB plus CSV manifest for successful outputs
- production now uses the same subset-download machinery, but with the production inventory-selection contract rather than the helper's older field preset
- cached production subsets are guarded by a selection-signature sidecar so widening the required inventory forces a rebuild instead of silently reusing an older subset

It is not the main production feature-builder anymore.

### `extract_hrrr_point_timeseries.py`

This older script still exists for simpler point-only workflows.

It is useful for:

- quick validation
- simple one-row-per-file extraction
- exploratory point sampling

It is not the canonical historical feature pipeline for the KLGA model.
