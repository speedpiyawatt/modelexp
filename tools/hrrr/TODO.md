# HRRR Binary Extraction TODO

This file is only for the new HRRR optimization path.

Future agents: update these checkboxes as work is completed or intentionally deferred. Do not add old completed phase history back into this file. The previous cfgrib/ecCodes/GRIB-repack paths are reference implementations for parity, not the optimization target.

## Current Goal

Build a new binary extraction path that avoids the current expensive sequence:

- selected-record download
- GRIB-to-GRIB crop/repack with `-small_grib` or `-ijsmall_grib`
- Python GRIB decode through cfgrib/ecCodes full-array reads

The target path should use `wgrib2` to dump selected HRRR messages directly to binary arrays, then compute KLGA nearest/neighborhood/crop stats in NumPy.

Keep the weather/model contract unchanged:

- settlement target remains final KLGA daily high
- station day remains `America/New_York`
- overnight cutoff remains `00:05 America/New_York`
- summary output remains the HRRR overnight feature table
- cfgrib/ecCodes outputs are parity references only

## Why This Is The Lead

Local smoke on cached `2023-02-04T05Z` reduced multi-forecast file:

- current ecCodes extractor: `23.72s` for 19 forecast-hour tasks
- `wgrib2 -no_header -bin ...` over the same reduced GRIB: about `0.33s` with `OMP_NUM_THREADS=1`
- NumPy read of the resulting `151 MB` binary: about `0.26s`

This is the first optimization lead that attacks the real bottleneck: GRIB repacking plus Python full-array GRIB decode.

## Step 1: Reduced-GRIB Binary Extractor

- [x] Add opt-in extraction backend `--extract-method wgrib2-bin`.
  - Keep current `cfgrib` and `eccodes` paths as reference implementations.
  - Do not make `wgrib2-bin` the default until parity and representative benchmarks pass.

- [x] Start from the current reduced multi-forecast GRIB.
  - Input: existing `selected_multiforecast.reduced.grib2`.
  - Run `wgrib2 selected_multiforecast.reduced.grib2 -no_header -bin <tmp.bin>` or equivalent.
  - Use `OMP_NUM_THREADS=1` unless a benchmark proves otherwise.
  - Preserve the reduced inventory used to create the binary dump.

- [x] Parse the binary dump in Python.
  - Determine dtype, endian, array shape, and message count from wgrib2/grid metadata.
  - Parse arrays by reduced-inventory message order.
  - Map each binary array back to the same feature prefix used by the current HRRR row builder.
  - Handle multi-forecast files and forecast-hour filtering without relying on cfgrib coordinates.

- [x] Reuse existing row math.
  - Reuse `feature_metrics()`.
  - Reuse temperature conversion logic.
  - Reuse derived RH logic.
  - Reuse derived wind speed/direction logic.
  - Reuse summary row generation.

- [x] Add diagnostics.
  - `extract_method=wgrib2-bin`
  - binary dump command
  - binary dump seconds
  - binary read seconds
  - row-build seconds
  - binary byte count
  - dtype/endian
  - grid shape
  - inventory message count
  - parsed array count
  - skipped/unknown message count

## Step 2: Parity Gates

- [x] Add a focused parity command.
  - Added `tools/hrrr/check_hrrr_wgrib2_bin_parity.py`.
  - Compares `wgrib2-bin` against cfgrib and/or ecCodes reference rows.
  - Asserts task/message alignment using reduced inventory order.
  - Writes a JSON report and exits nonzero on parity failure.

- [ ] Run the focused parity command for cached `2023-02-04T05Z`.
  - Use forecast hours `0-18`.
  - Record the JSON report path and max numeric drift.
  - Status: not run in this step because that cached artifact was not available in the current local scratch path.

- [x] Verify row-level parity on available local smoke artifacts.
  - Compare all shared numeric wide-row columns with documented tolerances.
  - Compare missing-field lists.
  - Compare grid identity: row/col, nearest grid lat/lon, crop metadata.
  - Verify forecast-hour filtering in multi-forecast batch files.
  - Verify duplicate-message handling, especially APCP.
  - Verify derived RH and wind fields.
  - Status: passed vs ecCodes at `1e-4` tolerance for `2026-04-11T04Z f00` narrow revision artifact and full 42-message artifact.
  - Reports: `/var/folders/yy/nsr07t_14b118f00wc0gd4km0000gn/T/hrrr_wgrib2_bin_parity_revision.json` and `/var/folders/yy/nsr07t_14b118f00wc0gd4km0000gn/T/hrrr_wgrib2_bin_parity_full.json`.
  - Max row drift: `3.4332275390625e-05` narrow, `6.201003812833505e-05` full.

- [x] Verify summary-level parity on available local smoke artifacts.
  - Run at least one narrow `--summary-profile overnight` smoke.
  - Run at least one full-profile smoke.
  - Confirm `hrrr.overnight.parquet` values are unchanged within expected numeric tolerance.
  - Status: passed vs ecCodes at `1e-4` tolerance for the same narrow and full `2026-04-11T04Z f00` artifacts.
  - Max summary drift: `0.0` narrow, `4.87795949197789e-05` full.

## Step 3: Bigger Win, Skip Reduced GRIB Creation

If Step 1 works, the bigger win is next:

- input: uncropped `selected_multiforecast.grib2`
- use cached full-grid `ijbox`
- skip reduced GRIB creation entirely

Implementation tasks:

- [x] Add cached full-grid `ijbox` support.
  - Infer the full-grid i/j rectangle needed for KLGA nearest/neighborhood/crop metrics.
  - Cache by HRRR grid signature and crop/stat bounds.
  - Validate grid shape and orientation against reference output.
  - Status: implemented for the opt-in `wgrib2-ijbox-bin` path using the existing crop-grid cache format and an ecCodes full-grid lat/lon context.

- [x] Dump binary arrays directly from the uncropped selected multi-forecast GRIB.
  - Input: `selected_multiforecast.grib2` from the existing `.idx` byte-range downloader.
  - Run `wgrib2 selected_multiforecast.grib2 -ijbox ... -no_header -bin <tmp.bin>` or equivalent.
  - Avoid `-small_grib` and `-ijsmall_grib`.
  - Avoid writing any reduced GRIB output.
  - Status: implemented as `--extract-method wgrib2-ijbox-bin` in batch-cycle mode. Non-batch mode still goes through the existing per-task reduce stage.

- [x] Add fallback behavior.
  - If direct `-ijbox ... -bin` fails validation, fall back to current reduced-GRIB path.
  - Record fallback reason in diagnostics.
  - Do not silently mix binary and reduced-GRIB outputs without diagnostics.
  - Status: batch-cycle reduction validates one direct `ijbox` binary dump per cycle. On validation failure it builds the reduced GRIB and sets `direct_ijbox_fallback_to_reduced=true` plus `direct_ijbox_fallback_reason`.

- [x] Add direct-path parity support.
  - `tools/hrrr/check_hrrr_wgrib2_bin_parity.py` now supports `--candidate-method wgrib2-ijbox-bin` and a separate `--reference-grib`.
  - Local parity passed for uncropped `2026-04-11T04Z f00` selected GRIB against the reduced ecCodes reference at `1e-4` tolerance.
  - Report: `/var/folders/yy/nsr07t_14b118f00wc0gd4km0000gn/T/hrrr_wgrib2_ijbox_bin_parity_full.json`.

## Step 4: Benchmarks

- [x] Add a reusable binary-extractor benchmark harness.
  - Added `tools/hrrr/benchmark_hrrr_binary_extractors.py`.
  - Compares `cfgrib`, `eccodes`, reduced `wgrib2-bin`, and direct `wgrib2-ijbox-bin` on cached GRIB artifacts.
  - Writes `benchmark_runs.json`, `benchmark_runs.csv`, and `benchmark_summary.json`.
  - Records wall time, CPU time, process max RSS, binary temp bytes observed, extractor timing breakouts, and parity drift.

- [ ] Benchmark reduced-GRIB `wgrib2-bin`.
  - Same cached `2023-02-04T05Z` multi-forecast file.
  - Same forecast hours `0-18`.
  - Compare against cfgrib and ecCodes references.
  - Record wall time, CPU time, max RSS, temp bytes, and output parity.
  - Smoke status: passed on available `2026-04-11T04Z f00` artifact.
  - Smoke report: `/tmp/hrrr_binary_benchmark_smoke_20260428T080315Z/benchmark_summary.json`.
  - Smoke result: ecCodes `0.348s`, reduced `wgrib2-bin` `0.080s`, max row drift `6.201003812833505e-05`.

- [ ] Benchmark direct `-ijbox ... -bin`.
  - Use the uncropped selected multi-forecast GRIB.
  - Include time saved by skipping reduced GRIB creation.
  - Break out download, binary dump, binary read, row-build, and summary timings.
  - Smoke status: passed on available uncropped `2026-04-11T04Z f00` selected artifact.
  - Smoke report: `/tmp/hrrr_binary_benchmark_smoke_20260428T080315Z/benchmark_summary.json`.
  - Smoke result: direct `wgrib2-ijbox-bin` `0.527s`, max row drift `6.201003812833505e-05`.

- [ ] Benchmark a representative overnight window.
  - Use the production-ish overnight settings.
  - Include anchor and lag/revision cycles.
  - Confirm the end-to-end monthly or multi-day backfill is materially faster.

## Definition Of Done

- [x] `--extract-method wgrib2-bin` exists and is opt-in.
- [x] Reduced-GRIB binary extraction passes row and summary parity.
- [x] Direct `-ijbox ... -bin` path skips reduced GRIB creation and passes parity.
- [x] Diagnostics clearly separate download, binary dump, binary read, row build, fallback, and summary costs.
- [ ] Representative overnight backfill is faster.
- [x] HRRR overnight output values are unchanged within expected numeric tolerance.
