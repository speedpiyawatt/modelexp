# NBM Tooling

Operational documentation for the NOAA National Blend of Models tooling in `tools/nbm`.

This directory exists to support one concrete use case:

- build settlement-aligned forecast features for the Polymarket KLGA temperature market

The tooling is not a generic weather platform. It is optimized for:

- downloading public NOAA NBM data
- reducing it to the KLGA / NYC problem
- extracting compact tabular features for modeling

## Standardization Layers

The repo now uses a three-layer weather standard instead of treating raw NBM as the one true schema.

### 1. Raw Runtime Source Contract

`build_grib2_features.py` is one of the two v1 standardized live runtime builders.

At this layer, NBM and HRRR must agree on:

- settlement station and crop logic from `tools/weather/location_context.py`
- shared runtime identity fields
- shared spatial metadata fields
- shared nearest / `nb3` / `nb7` / crop suffix semantics
- Parquet-first persistent outputs

### 2. Canonical Training Schema

The model-facing schema now lives downstream in:

- `tools/weather/canonical_feature_schema.py`
- `tools/weather/normalize_training_features.py`

Raw NBM output stays NBM-shaped. Raw HRRR output stays HRRR-shaped. Training code should read the canonical normalized outputs instead of assuming the raw wide tables are drop-in identical.

### 3. Operational Metadata Contract

Provenance should answer where each feature came from. Manifests remain operational artifacts for extraction/debugging and are not primary model-training inputs.

## Weather Stack Layout

The forecast tooling now lives side by side under `tools/`:

- `tools/nbm`
  - NBM ingestion and extraction
- `tools/hrrr`
  - HRRR ingestion and extraction
- `tools/weather`
  - shared location context, canonical schema, normalization, audits, and verification

Treat these as the canonical entrypoints:

- raw NBM runtime builder: `tools/nbm/build_grib2_features.py`
- raw HRRR runtime builder: `tools/hrrr/build_hrrr_klga_feature_shards.py`
- canonical training normalization: `tools/weather/normalize_training_features.py`
- verification runner: `tools/weather/run_verification_suite.py`

Data should be interpreted in two buckets:

- committed regression fixtures
  - example: `tools/hrrr/data/fixtures/contract_smoke`
- generated runtime output
  - tool-local `data/runtime/...` trees and temporary scratch/output directories

Do not treat committed fixtures as production outputs.

## What Is Here

The directory contains three main documented components:

- `fetch_nbm.py`
  - downloader for public NBM GRIB2 core data
- `build_grib2_features.py`
  - raw GRIB2 download, subset, crop, feature extraction, and parquet writer
- `location_context.py`
  - shared location and neighborhood logic used by the GRIB2 feature pipeline

Low-disk daily resume-from-disk backfill helper:

- `tools/nbm/run_nbm_monthly_backfill.py`

This wrapper runs the NBM overnight build one target day at a time, scans existing daily overnight outputs on every run, skips valid days already on disk, rebuilds invalid days, and deletes each successful day temp tree so only `nbm_overnight` remains under the chosen run root.

Example:

```bash
source .venv/bin/activate
python tools/nbm/run_nbm_monthly_backfill.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --selection-mode overnight_0005 \
  --batch-reduce-mode cycle \
  --smart-workers \
  --cpu-cores 4 \
  --min-free-gb 10 \
  --crop-method small_grib \
  --wgrib2-threads 1 \
  --crop-grib-type complex3 \
  --overnight-fast \
  --progress-mode dashboard \
  --pause-control-file /tmp/nbm.pause
```

For overnight backfills using `--batch-reduce-mode cycle`, the normal operating mode is static day-level parallelism. On the current 4-core server, benchmark runs over 2026-04-01 through 2026-04-14 were fastest with `--day-workers 4`, `--download-workers 4`, `--reduce-workers 1`, `--extract-workers 1`, and `--wgrib2-threads 1`. `--smart-workers --cpu-cores 4` now selects that batch profile automatically. Do not use `--adaptive-workers` for routine batch-cycle runs; it is mainly for experiments and legacy separated per-lead tuning.

If you want faster reruns or debugging and have disk available, retain reduced GRIBs:

```bash
source .venv/bin/activate
python tools/nbm/run_nbm_monthly_backfill.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --selection-mode overnight_0005 \
  --batch-reduce-mode cycle \
  --smart-workers \
  --cpu-cores 4 \
  --min-free-gb 10 \
  --keep-reduced
```

## Data Sources

The documented NBM source is the NOAA public GRIB2 core archive:

- GRIB2 core archive: `noaa-nbm-grib2-pds`

The GRIB2 path is the production path when you want to:

- keep only specific variables
- crop to a fixed NYC box
- write reduced intermediate GRIB2 files
- preserve exact GRIB provenance

## Current NBM Data Contract

The implemented NBM pipeline is:

```text
NOAA .idx inventory
-> selected GRIB2 byte-range download
-> selected-record raw subset GRIB
-> wgrib2 NYC/KLGA crop
-> cfgrib grouped decode
-> raw wide/provenance/manifest parquet
-> one-row-per-target-day nbm.overnight.parquet
```

For the overnight market-open model, `--selection-mode overnight_0005` selects one latest cutoff-eligible issue per `target_date_local`. The eligible init window is previous local day `18:00` through target day `00:05 America/New_York`. The raw builder processes NBM core leads `f001` through `f036`.

The low-disk wrapper stores only the daily overnight output durably by default:

- temp raw/crop/extract tree: `<run-root>/nbm_tmp/YYYY-MM-DD`
- durable daily output: `<run-root>/nbm_overnight/target_date_local=YYYY-MM-DD/nbm.overnight.parquet`
- durable daily manifest: `<run-root>/nbm_overnight/target_date_local=YYYY-MM-DD/nbm.overnight.manifest.parquet`

After a successful day, the wrapper removes the per-day temp tree unless an explicit debug/retention flag requires otherwise. Re-running the wrapper rescans daily overnight outputs and rebuilds missing or invalid days.

Current raw NBM fields:

- required wide fields: `tmp`, `dpt`, `rh`, `wind`, `wdir`, `gust`, `tcdc`, `dswrf`, `apcp`, `vrate`, `vis`, `ceil`, `cape`
- optional native/regime fields: `tmax`, `tmin`, `pcpdur`, `thunc`, `ptype`, `pwther`, `tstm`
- mirrored long-to-wide regime fields: `ptype`, `pwther`, `tstm`

`ptype`, `pwther`, and `tstm` are mirrored into the wide parquet even when `--write-long` is disabled. This is intentional: overnight summarization should not depend on long parquet output.

Current overnight NBM output includes:

- old checkpoint/day summaries such as `nbm_temp_2m_day_max_k`, local-hour checkpoint features, cloud/radiation/precip/visibility/ceiling/CAPE summaries, and neighborhood/crop summaries
- native daily guidance: `nbm_native_tmax_*` and `nbm_native_tmin_*` when available
- regime/suppression summaries: `nbm_pcpdur_*`, `nbm_pwther_*`, `nbm_tstm_*`
- optional lower-priority summaries: `nbm_ptype_*`, `nbm_thunc_*`, `nbm_vrate_*`
- coverage metadata: `missing_required_feature_count` and `coverage_complete`

Native `tmax` and `tmin` are currently counted by the coverage metadata. Missing values do not fail the day outright; they produce null native fields, increment `missing_required_feature_count`, and set `coverage_complete=false`.

For the `overnight_0005` daily-high workflow, native `TMIN` may be absent in the selected 04Z core source inventory while hourly `TMP` and native `TMAX` are present. This is visible before crop/extract in the source `.idx` files, so `nbm_native_tmin_* = null`, `missing_required_feature_count=1`, and `coverage_complete=false` are not by themselves evidence of a `--batch-reduce-mode cycle` crop/decode failure. Daily-high modeling should continue to use rows where the max-temperature signals are present, especially `nbm_temp_2m_day_max_k`, local-hour `nbm_temp_2m_*_local_k`, and `nbm_native_tmax_2m_day_max_k`.

## Location Model

The code uses one explicit three-level location model.

### 1. Settlement Location

This is the market settlement point:

- station: `KLGA`
- latitude: `40.7769`
- longitude: `-73.8740`

This point is used for:

- nearest-grid extraction
- label alignment
- final model-facing rows

Every feature row is ultimately “about KLGA.”

### 2. Regional Crop Location

This is the fixed NYC context box:

- top latitude: `43.5`
- bottom latitude: `39.0`
- left longitude: `282.5`
- right longitude: `289.5`

Important:

- longitudes are in the `0-360` system for the crop
- NYC around `-73.9` maps to about `286.1`

This crop is used for:

- GRIB2 spatial subsetting
- regional context statistics
- limiting storage versus full CONUS

### 3. Local Neighborhood Location

After locating the nearest grid point to KLGA, the tooling extracts:

- nearest point
- `3x3` neighborhood
- `7x7` neighborhood

The feature pipelines compute:

- nearest-point value
- neighborhood mean
- neighborhood min
- neighborhood max
- neighborhood std
- west-east gradient
- south-north gradient

The intent is:

- nearest point captures the airport signal
- `3x3` captures very local structure
- `7x7` captures slightly broader local context
- crop-wide stats capture the surrounding regional state

## Tooling Overview

### `fetch_nbm.py`

This script is documented here in its GRIB2 role:

- `grib2`
  - fetch a single GRIB2 forecast file

Key behavior:

- pooled HTTP requests
- optional `.idx` sidecar download
- byte-range helpers for selected-record GRIB2 fetches
- deterministic NOAA key resolution for GRIB2 core files

For byte-range GRIB2 fetches, the client now treats Range semantics strictly:

- the server must return `206 Partial Content`
- each response body must match the exact requested byte span

If either condition fails, the selected-record download is rejected instead of writing a corrupt local GRIB artifact.

### `build_grib2_features.py`

This is the more controlled feature pipeline.

It performs:

1. cycle discovery across a local-date window
2. `.idx` download
3. `.idx` parsing and record selection
4. conditional remote GRIB2 content-length lookup only when the final selected record needs file-end resolution
5. byte-range download of only the selected GRIB2 records into a local selected-record `.grib2`, with adjacent or near-adjacent spans merged to reduce HTTP round trips
6. crop to the NYC box
7. grouped `cfgrib` opens
8. feature extraction
9. parquet writes
10. cleanup of temporary GRIB processing artifacts after success

To inspect where time is actually going in NBM manifests:

```bash
source .venv/bin/activate
python tools/nbm/summarize_nbm_diagnostics.py \
  --path data/runtime/backfill_overnight_2023_2026
```

Persistent outputs by default:

- wide parquet
- provenance parquet
- manifest parquet

Optional persistent output:

- long parquet when `--write-long` is enabled

Temporary processing artifacts by default:

- raw selected-record `.grib2`
- raw `.idx`
- reduced cropped GRIB2

Debug / replay flags:

- `--keep-downloads`
- `--keep-reduced`

`--keep-downloads` now retains:

- the selected-record local `.grib2`
- the downloaded `.idx`

If a crop or downstream parse step fails and `--keep-downloads` is enabled, those retained files are intentionally kept for replay and debugging.

Execution controls:

- `--workers` controls how many NBM cycles can run concurrently
- `--smart-workers` is the normal startup planner; with `--batch-reduce-mode cycle`, it uses a batch-aware day-parallel profile instead of scaling per-day reduce/extract workers
- `--adaptive-workers` is opt-in dynamic tuning for future day admissions; avoid it for routine batch-cycle runs because the useful concurrency is already fixed at the day level
- `--day-workers`, `--lead-workers`, `--download-workers`, `--reduce-workers`, and `--extract-workers` are manual override knobs; with batch-cycle overnight runs on the current 4-core server, use `--day-workers 4 --download-workers 4 --reduce-workers 1 --extract-workers 1`
- `--reduce-queue-size` and `--extract-queue-size` bound the pipeline queues between download, reduce, and extract stages
- `--crop-method auto|small_grib|ijsmall_grib` controls the crop primitive; `auto` is the default and prefers cached `wgrib2 -ijsmall_grib` grid boxes when the raw subset grid is stable, then falls back to the legacy lat/lon `-small_grib` path when needed
- `--wgrib2-threads` optionally overrides crop subprocess OpenMP threads; when unset, the builder applies a safe automatic policy and always runs crop subprocesses with `OMP_WAIT_POLICY=PASSIVE`
- `--metric-profile full|overnight` controls row metric extraction; monthly overnight backfills default to `overnight`, which computes only nearest values plus the nb/crop metrics consumed by `nbm.overnight.parquet`
- `--batch-reduce-mode off|cycle` controls whether reduce/extract stays per-lead (`off`, default) or concatenates selected lead GRIBs for each cycle and performs one crop/cfgrib open before splitting rows back to per-lead outputs (`cycle`)
- `--selection-mode overnight_0005` now prunes per-cycle lead hours down to only the leads whose valid local date matches the selected target day; omit it for full raw all-cycle inventories
- `--skip-provenance` suppresses provenance-row construction and provenance parquet writes without changing wide or optional long outputs
- `--progress-mode auto|dashboard|log` selects either the live terminal dashboard or structured progress logs
- in `--progress-mode dashboard`, press `p` to request a graceful drain-and-pause; once the dashboard shows `PAUSED`, it is safe to exit and resume later from disk
- `--disable-dashboard-hotkeys` disables interactive dashboard hotkeys without changing dashboard rendering
- `--pause-control-file /tmp/nbm.pause` enables a non-TTY pause trigger; create that file with `touch /tmp/nbm.pause` from another shell to request the same graceful drain-and-pause flow
- `--max-task-attempts`, `--retry-backoff-seconds`, and `--retry-max-backoff-seconds` control automatic per-lead retries with backoff

Low-disk daily resume-from-disk backfill helper:

- `tools/nbm/run_nbm_monthly_backfill.py`

This wrapper now runs day by day, rescans disk on every run, skips valid daily overnight outputs, rebuilds stale or missing days, and deletes the per-day temp tree after success so only `nbm_overnight` is retained.

Example:

```bash
source .venv/bin/activate
python tools/nbm/run_nbm_monthly_backfill.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-03-31 \
  --run-root data/runtime/backfill_overnight_2023_2026 \
  --selection-mode overnight_0005 \
  --batch-reduce-mode cycle \
  --smart-workers \
  --cpu-cores 4 \
  --min-free-gb 10 \
  --crop-method small_grib \
  --wgrib2-threads 1 \
  --crop-grib-type complex3 \
  --overnight-fast \
  --progress-mode dashboard \
  --pause-control-file /tmp/nbm.pause
```

The worker model is no longer a single flat pool. The builder now runs a staged per-cycle pipeline:

1. download and plan selected-record GRIB content
2. reduce the selected-record GRIB to the NYC crop
3. open and extract Parquet rows

With `--batch-reduce-mode cycle`, step 1 remains per-lead, but steps 2 and 3 are performed once per selected issue/day by concatenating the selected raw lead GRIBs into a local multi-lead GRIB. The builder then expands cfgrib step/time dimensions back into 2D slices and emits the same per-lead wide/manifest contract. Final overnight parquet parity against legacy `--batch-reduce-mode off` has been validated over 2026-04-01 through 2026-04-14.

Batch mode has two guardrails to keep the per-lead contract explicit:

- after row extraction, every wide/long/provenance row must map by computed `forecast_hour` to one of the admitted leads for that cycle; unexpected forecast hours fail the batch instead of being silently dropped
- provenance and diagnostic inventory matching is forecast-hour aware in batch extraction, so a row for `f020` is matched back to the selected `f020` inventory record rather than the first selected record for the same feature

The live dashboard/log output exposes cycle counts, phase-specific worker counts, queue depths, transfer speed, retries, and recovered tasks. Use `--progress-mode log` in CI or when you want plain append-only logs instead of the full-screen dashboard. In dashboard mode, `p` requests a graceful drain-and-pause rather than an immediate cancel. For remote terminals that do not deliver raw keypresses reliably, use `--pause-control-file` and trigger pause with `touch`.

The monthly helper now has its own operational controls:

- `--day-workers` parallelizes target days conservatively; default `1` preserves serial behavior
- `--smart-workers` chooses an initial worker plan from available CPU cores; in batch-cycle mode, `--smart-workers --cpu-cores 4` selects the measured 4-core profile: `day-workers=4`, `download-workers=4`, `reduce-workers=1`, `extract-workers=1`, and queue sizes `2`
- `--adaptive-workers` enables admission-time runtime tuning for future days based on observed raw-build and cleanup timings; it never mutates already-running children and intentionally leaves `--download-workers` fixed for the whole run. Keep it off for normal batch-cycle runs.
- `--adaptive-sample-days` and `--adaptive-cooldown-days` control how quickly the controller can react; defaults are conservative (`2` successful days to learn, then `2` successful days between changes)
- `--adaptive-min-day-workers`, `--adaptive-max-day-workers`, `--adaptive-min-lead-workers`, `--adaptive-max-lead-workers`, `--adaptive-min-reduce-workers`, `--adaptive-max-reduce-workers`, `--adaptive-min-extract-workers`, and `--adaptive-max-extract-workers` bound what the controller is allowed to change
- `--crop-method`, `--wgrib2-threads`, and `--crop-grib-type` are passed to the child raw builder for every admitted day; monthly backfills default to `small_grib`, `1`, and `complex3` respectively because the benchmark harness showed that combination beating cached `ijsmall_grib` on the current NBM workload
- `--metric-profile overnight` is the monthly default because the durable output is the one-row daily overnight summary; use `--metric-profile full` only when you need the complete raw wide metric family for source debugging
- `--batch-reduce-mode off|cycle` is passed to the child raw builder; use `cycle` for overnight backfills so crop/cfgrib work is batched after selected-record downloads
- `--min-free-gb` blocks new day admission when free disk is below the threshold, including the first pending day
- `--overnight-fast` passes `--skip-provenance` through to the raw builder for overnight-only production runs
- in-process overnight finalization is the default handoff after raw cycle outputs are written
- `--overnight-subprocess` keeps the old second-process finalizer path available for debugging or parity checks
- `--progress-mode dashboard` now supports `--day-workers > 1` with one parent-owned monthly dashboard
- active target days are shown in a dedicated groups panel, while the worker table now shows the real underlying child download/reduce/extract workers plus monthly overnight/validate/cleanup workers
- child raw builders are still forced to `--progress-mode log` underneath the monthly runner so there is only one TUI on screen
- the monthly runner owns pause/drain; `p`, `--pause-control-file`, SIGINT, or SIGTERM stop new day admission and drain already admitted days instead of letting children manage the same pause file independently
- when adaptive tuning is enabled, the summary panel shows the live planned `day/lead/reduce/extract` settings plus the last tuning decision and reason; tuning only applies to newly admitted days

Measured batch-cycle profile on the current 4-core server:

```text
2026-04-01 through 2026-04-14, batch-reduce-mode=cycle
day-workers=2  real 4:32.22
day-workers=3  real 4:11.21
day-workers=4  real 3:18.68
day-workers=5  real 3:29.42
```

Use `day-workers=4` for this workload unless a larger benchmark shows different behavior.

Availability semantics:

- `processed_timestamp_utc` is written as the issue `init_time_utc`, not wall-clock script runtime
- this keeps historical replay builds eligible for the overnight `00:05 America/New_York` cutoff
- `build_nbm_overnight_features.py` treats that field as the issue availability timestamp when selecting the latest cutoff-eligible issue

Default runtime output root:

- `data/nbm/grib2`

Those outputs are generated runtime artifacts, not committed fixtures.

### `location_context.py`

Shared source of truth for:

- settlement location
- regional crop bounds
- neighborhood sizes
- nearest-cell lookup
- neighborhood stats
- crop stats

If the location model changes, update this module first.

## Variable Sets

### GRIB2 Field Set

The GRIB2 pipeline currently targets these fields when available:

- `TMP`, `DPT`, `RH`, `TMAX`, `TMIN`
- `WIND`, `WDIR`, `GUST`
- `TCDC`
- `DSWRF`, `APCP`, `VRATE`, `PCPDUR`, `VIS`
- `CEIL`
- `CAPE`, `THUNC`
- optional long-format fields such as `PTYPE`, `PWTHER`, `TSTM`

Some fields use nearest operational matches from NBM inventory text rather than perfect textual equality. Those choices are recorded in provenance output.

## Installation

From the repo root:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

The GRIB2 pipeline also requires `wgrib2` on `PATH`.

On this machine, the practical installation path is:

```bash
conda install -c conda-forge wgrib2
```

Useful checks:

```bash
.venv/bin/python -m py_compile tools/nbm/*.py
.venv/bin/python -m pytest -q tests/test_nbm_tools.py
wgrib2 -version
```

## Common Workflows

### 1. Fetch One GRIB2 File

```bash
python3 tools/nbm/fetch_nbm.py grib2 \
  --date 2026-04-11 \
  --cycle 11 \
  --forecast-hour 1 \
  --region co \
  --with-idx
```

### 2. Build GRIB2-Based Reduced Features

```bash
python3 tools/nbm/build_grib2_features.py \
  --start-local-date 2023-01-01 \
  --end-local-date 2026-04-12 \
  --workers 4 \
  --lead-workers 6 \
  --download-workers 6 \
  --reduce-workers 4 \
  --extract-workers 4 \
  --output-dir data/nbm/grib2
```

Write optional long parquet and keep reduced intermediates only when needed:

```bash
python3 tools/nbm/build_grib2_features.py \
  --start-local-date 2026-04-10 \
  --end-local-date 2026-04-12 \
  --write-long \
  --keep-reduced \
  --keep-downloads \
  --progress-mode log \
  --output-dir data/nbm/grib2_debug
```

Skip provenance when you only need overnight summary outputs:

```bash
python3 tools/nbm/build_grib2_features.py \
  --start-local-date 2026-04-10 \
  --end-local-date 2026-04-12 \
  --selection-mode overnight_0005 \
  --skip-provenance \
  --progress-mode log \
  --output-dir data/nbm/grib2_fast
```

Dry-run:

```bash
python3 tools/nbm/build_grib2_features.py \
  --start-local-date 2026-04-10 \
  --end-local-date 2026-04-12 \
  --progress-mode log \
  --dry-run
```

### 3. Fast Overnight Short-Window Raw Build

For direct raw-builder debugging, prefer the latest-issue overnight planner and bounded per-cycle lead concurrency:

```bash
python3 tools/nbm/build_grib2_features.py \
  --start-local-date 2025-04-11 \
  --end-local-date 2025-04-13 \
  --selection-mode overnight_0005 \
  --workers 3 \
  --lead-workers 4 \
  --download-workers 4 \
  --reduce-workers 2 \
  --extract-workers 2 \
  --range-merge-gap-bytes 4096 \
  --keep-downloads \
  --overwrite \
  --output-dir tools/weather/data/runtime/server_short_window/nbm/features
```

For production overnight backfills, prefer the monthly runner with `--batch-reduce-mode cycle` so concurrency is applied across target days rather than inside one selected issue.

Then build one-row-per-day overnight summaries:

```bash
python3 tools/nbm/build_nbm_overnight_features.py \
  --features-root tools/weather/data/runtime/server_short_window/nbm/features \
  --output-dir tools/weather/data/runtime/server_short_window/nbm/overnight \
  --start-local-date 2025-04-11 \
  --end-local-date 2025-04-13
```

`--selection-mode overnight_0005` keeps only the latest issue eligible by the overnight `00:05 America/New_York` cutoff for each target local day instead of all cycles in the local-date window. That changes source selection, not just throughput.

Use that selection mode when you intentionally want the overnight market-open issue set, including long historical overnight-model backfills. For raw all-cycle source exploration, omit `--selection-mode overnight_0005` and keep the default cycle planner.

### 3. Benchmark Public Download Sources

```bash
python3 tools/nbm/benchmark_nbm_sources.py \
  --date 2026-04-11 \
  --cycle 11 \
  --forecast-hour 1 \
  --region co \
  --sources aws,noaa_https,noaa_ftp \
  --runs 3 \
  --warmup-runs 1
```

This benchmark compares AWS, NOAA HTTPS, and NOAA FTP for the same canonical NBM GRIB2 core file.

This writes one runtime benchmark directory under `tools/nbm/data/runtime/download_benchmarks/<timestamp>/` with:

- `downloads/`
  - one unique file per source, phase, and run
- `benchmark_runs.csv`
  - per-run benchmark records
- `benchmark_runs.json`
  - the same per-run records as JSON
- `benchmark_summary.json`
  - aggregated success counts, wall-time statistics, throughput, SHA256 agreement, and source ranking

While it runs, the benchmark prints a terminal progress view with:

- overall run progress across all warmup and measured attempts
- per-transfer status including source, phase, run index, elapsed time, downloaded size, and transfer speed
- explicit completion or error status for each attempt

### 4. Benchmark Crop Variants

```bash
python3 tools/nbm/debug_bottlenecks.py \
  --start-local-date 2026-04-11 \
  --end-local-date 2026-04-12 \
  --selection-mode overnight_0005 \
  --lead-workers 8 \
  --crop-method auto \
  --wgrib2-threads 1 \
  --crop-packings same complex1 complex3
```

This reuses the NBM bottleneck harness for crop experiments. The per-run summaries and the final comparison report now include:

- crop primitive (`auto`, `small_grib`, `ijsmall_grib`)
- crop packing (`same`, `complex1`, `complex3`, etc.)
- effective `wgrib2` thread setting
- `timing_crop_seconds` totals, averages, and p95
- reduced-file size totals
- crop timing deltas versus the baseline run

Use this benchmark to decide whether a future default packing change is justified. The runtime builder still keeps its current `-set_grib_type same` default unless you explicitly benchmark alternatives.

## Output Structure

### GRIB2 Outputs

`build_grib2_features.py` writes:

- `features/wide/`
  - wide parquet per cycle partition
- `features/long/`
  - optional long parquet per cycle partition when `--write-long` is enabled
- `metadata/provenance/`
  - exact field provenance when provenance writing is enabled
- `metadata/manifest/`
  - processing audit trail

Reduced cropped GRIB2 files are temporary by default and are only retained when `--keep-reduced` is enabled.
Provenance parquet is optional and is skipped entirely when `--skip-provenance` is enabled.
When `--crop-method auto` uses cached `ijsmall_grib`, the manifest records which crop primitive actually ran, whether the crop-grid cache was hit, the cached ij box, and the effective crop subprocess thread count.

The raw `raw/` artifact is not the full NOAA object anymore. It is the concatenated selected-record GRIB built from `.idx`-planned byte ranges.

Partitions are keyed by:

- `valid_date_local`
- `init_date_local`
- `mode`

`mode` is derived metadata for modeling convenience. It is not the canonical row identity.

## Schema Notes

## Canonical Row Identity

Every NBM wide row must contain these canonical identity fields:

- `source_model`
- `source_product`
- `source_version`
- `fallback_used_any`
- `station_id`
- `init_time_utc`
- `valid_time_utc`
- `init_time_local`
- `valid_time_local`
- `init_date_local`
- `valid_date_local`
- `forecast_hour`
- `settlement_lat`
- `settlement_lon`
- `crop_top_lat`
- `crop_bottom_lat`
- `crop_left_lon`
- `crop_right_lon`
- `nearest_grid_lat`
- `nearest_grid_lon`

Compatibility columns such as `lead_hour`, `lead_hours`, `grid_lat`, `grid_lon`, and `crop_top` / `crop_bottom` / `crop_left` / `crop_right` may remain in outputs, but they are not the canonical contract.

Rows must be joinable without using `mode`.

### GRIB2 Wide Schema

The GRIB2 wide table contains:

- source metadata
- settlement metadata
- crop metadata
- init and valid timestamps
- `forecast_hour`
- derived `mode`
- nearest-point values
- crop-wide stats
- `nb3`
- `nb7`
- derived `u10`
- derived `v10`

For example:

- `tmp`
- `tmp_crop_mean`
- `tmp_nb3_mean`
- `tmp_nb3_gradient_west_east`
- `tmp_nb7_mean`
- `tmp_nb7_gradient_south_north`

The canonical spatial feature-family pattern for continuous numeric fields is:

- nearest
- `nb3`
- `nb7`
- crop

Future weather pipelines should align to this naming pattern.

### Provenance

The GRIB2 provenance table records the exact source field used for each extracted feature, including:

- feature name
- source product
- source version
- GRIB short name
- level text
- type of level
- step type
- step text
- inventory line
- GRIB variable name
- units
- whether fallback mapping was used
- fallback source description

This is the audit trail for field mapping and fallback behavior.

### Manifest

The GRIB2 manifest records one row per processed lead unit and includes:

- source URL
- download mode
- remote file size
- raw file name
- raw file size
- selected record count
- selected download bytes
- downloaded range bytes after merge compaction
- merged range count
- whether `HEAD` was used to resolve the terminal byte range
- reduced file path
- `.idx` hash
- extraction status
- output paths
- subset command
- per-stage timing fields for idx fetch/parse, HEAD, range download, crop, cfgrib open, row build, and cleanup
- cleanup status flags
- warnings
- processed timestamp

## Storage Expectations

There are two main storage tiers.

### 1. Reduced GRIB2 Files

These are much smaller than retaining full raw NBM forecast files because they keep:

- only selected variables
- only the NYC crop

The raw selected-record `.grib2` retained by `--keep-downloads` is also much smaller than the full NOAA source object because it only contains the records chosen from the `.idx`.

They are still not tiny when retained for debugging because the pipeline can keep:

- one reduced GRIB2 file per lead
- for every cycle

That means storage scales with:

- cycles per local day
- 36 leads per cycle

Measured local smoke findings from real reduced outputs:

- `tmp/grib2-smoke/.../blend.t14z.core.f001.co.subset.grib2`
  - `335,894` bytes
- `tmp/reduced_size_20260411/.../blend.t07z.core.f001.co.subset.grib2`
  - `350,584` bytes

Practical rule of thumb from those live files:

- one reduced lead is about `0.32-0.34 MB`
- one 36-lead cycle is about `11.5-12.2 MB`
- one active local day with many cycles can still land around `~200 MB` if every reduced lead file is retained

That estimate is specifically for a debug-style retained-intermediate workflow:

- keep one reduced GRIB2 file per lead
- keep every discovered cycle in the local date window
- keep parquet outputs in addition to reduced GRIB2

### 2. Final Feature Tables

This is the smallest and most model-friendly layer.

If long-term storage matters more than GRIB reprocessing, the best storage reduction is:

- keep parquet features
- drop reduced GRIB2 after successful extraction

That is now the default production policy.

## Performance Notes

### GRIB2

`build_grib2_features.py` supports:

- concurrent cycle workers
- concurrent lead workers within a cycle via `--lead-workers`
- batch-cycle reduce/extract via `--batch-reduce-mode cycle`, where a selected issue is concatenated, cropped, opened, and row-built once before rows are split back to per-lead outputs
- overnight-only latest-issue planning via `--selection-mode overnight_0005`
- `.idx`-driven selected-record byte-range downloads
- merged selected byte ranges via `--range-merge-gap-bytes`
- raw cleanup after successful extraction

The bottleneck is often:

- GRIB decode, crop, and parquet write work after download
- NOAA transfer latency for the selected records
- not just raw bandwidth

Optimization scope:

- `--selection-mode overnight_0005` is the overnight-model source selector because it changes which issues are included
- for legacy separated per-lead mode, `--lead-workers` is a general throughput optimization and can improve larger runs too, subject to local CPU and disk limits
- for batch-cycle overnight backfills, scale `--day-workers` in the monthly runner instead of scaling per-day `--reduce-workers` or `--extract-workers`; the measured 4-core profile is `day-workers=4`, `download-workers=4`, `reduce-workers=1`, `extract-workers=1`, `wgrib2-threads=1`
- `--range-merge-gap-bytes` is a general network-latency optimization that trades a small amount of extra downloaded data for fewer HTTP Range requests
- conditional `HEAD` requests reduce one remote metadata call for units whose selected records do not include the final inventory record

## Operational Guidance

Recommended order of use:

1. use `build_grib2_features.py --dry-run`
2. backfill a short date range
3. inspect output sizes
4. validate wide and long schemas
5. expand to a larger historical window

For model building:

- use KLGA settlement-aligned outputs
- prefer parquet for large backfills
- treat `mode` as derived metadata, not as the primary join key

## Troubleshooting

### `wgrib2` not found

Install:

Use a package source that provides `wgrib2` on your platform, then ensure it is on `PATH`.

Verify:

```bash
wgrib2 -version
```

### `cfgrib` / `eccodes` import failure

Reinstall Python requirements:

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
```

### Historical backfills are too large

Use one or more of:

- fewer variables
- fewer cycles
- `--selection-mode overnight_0005` for overnight validation runs
- `--batch-reduce-mode cycle` plus monthly `--day-workers` for overnight backfills
- bounded `--lead-workers` to use local CPU more effectively in separated per-lead or raw-debug runs
- shorter date windows
- parquet output
- keep reduced GRIB2 only for debugging or replay via `--keep-reduced`

For full-history overnight-model backfills, use `--selection-mode overnight_0005` with monthly `--batch-reduce-mode cycle`. For full-history raw all-cycle NBM inventories, keep the default cycle planner and only carry over the throughput-oriented knobs such as `--workers` and `--lead-workers`.

## Testing

Primary test file:

```bash
.venv/bin/python -m pytest -q tests/test_nbm_tools.py tests/test_nbm_overnight_tools.py
```

The tests currently cover:

- paginated bucket listing
- dry-run and streaming behavior
- location defaults
- longitude normalization
- GRIB2 record selection
- cfgrib alias mapping
- GRIB2 feature extraction
- cleanup behavior
- manifest path behavior
- canonical identity columns
- canonical wide-schema naming
- Parquet-first cleanup defaults
- overnight native `tmax` / `tmin` summaries, with native `tmin` allowed to be null when absent from the selected source inventory
- regime-field mirroring into wide rows
- categorical mode/count/any encoding for overnight features

## Short Summary

If you only need model-ready KLGA features:

- use `build_grib2_features.py` for controlled reduction and provenance

If storage matters:

- reduced GRIB2 is smaller
- parquet features are the smallest useful layer

## Shared Cross-Model Weather Feature Contract

Any weather feature pipeline in this repo should align to:

- settlement-locked station modeling
- shared raw runtime identity fields
- shared raw spatial feature families: nearest, `nb3`, `nb7`, crop
- Parquet-first persistent artifacts
- provenance-required extraction

For model training, the authoritative shared schema is the canonical normalized layer in `tools/weather/canonical_feature_schema.py`, not the raw NBM wide table.
