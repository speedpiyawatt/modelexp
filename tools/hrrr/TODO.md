# HRRR Pipeline Optimization TODO

This TODO tracks optimization work for the KLGA overnight HRRR feature pipeline.

Future agents: update these checkboxes as work is completed or intentionally deferred. If a task changes scope, edit the note under that task rather than leaving stale boxes behind.

## Goal

Make the HRRR overnight backfill materially faster without changing the settlement target or model contract:

- target remains final KLGA daily high for the `America/New_York` local station day
- overnight cutoff remains `00:05 America/New_York`
- core output remains a stable weather target and `hrrr.overnight.parquet` summary features
- lag/revision cycles should do only the work required for revision features

## Phase 1: Low-Risk Production Wins

- [x] Port the NBM crop layer into HRRR.
  - Add cached `-ijsmall_grib` support for stable HRRR grids.
  - Keep `-small_grib` fallback for validation failures or unsupported grids.
  - Add `-set_grib_type` support.
  - Add crop-specific `OMP_NUM_THREADS` / `OMP_WAIT_POLICY` handling.
  - Record crop method, grid cache key, crop seconds, and fallback reason in diagnostics.

- [x] Add HRRR `--summary-profile overnight`.
  - Anchor cycle keeps the fields needed for the full overnight summary.
  - Lag/revision cycles fetch and extract only fields needed by `build_revision_features()`.
  - Keep an escape hatch such as `--summary-profile full` or `--full-raw` for diagnostics.
  - Status: implemented as a task-aware field profile. Under `--summary-profile overnight`, anchor tasks use the full field set and non-anchor revision tasks use the revision-only field set.

- [x] Add HRRR `--skip-provenance`.
  - Skip provenance row construction and provenance parquet writes for production overnight backfills.
  - Preserve current provenance behavior by default.
  - Record `provenance_written=false` in manifests when skipped.

- [x] Fix batch timing diagnostics.
  - Store cycle-level reduce/open timings separately from task-level timings.
  - Avoid stamping full cycle-level cfgrib open seconds onto every task as if it were per-task cost.
  - Keep legacy-compatible apportioned timing fields if existing analysis scripts need them.

## Phase 2: Narrow Lag-Cycle Work

- [x] Define an explicit lag-cycle field plan.
  - Required revision bases:
    - `hrrr_temp_2m_day_max_k`
    - `hrrr_temp_2m_09_local_k`
    - `hrrr_temp_2m_12_local_k`
    - `hrrr_temp_2m_15_local_k`
    - `hrrr_tcdc_day_mean_pct`
    - `hrrr_dswrf_day_max_w_m2`
    - `hrrr_pwat_day_mean_kg_m2`
    - `hrrr_hpbl_day_max_m`
    - `hrrr_mslp_day_mean_pa`

- [x] Use the lag-cycle field plan during selected-record byte-range fetch.
  - Do not download broad HRRR fields for lag cycles when only revision bases are needed.
  - Preserve current behavior for anchor cycles and full diagnostic runs.

- [x] Use the lag-cycle field plan during reduce/extract.
  - Avoid opening cfgrib groups that cannot contribute to revision features.
  - Avoid provenance and derived-field work that is irrelevant to revision deltas.

- [x] Add parity tests for anchor-plus-lag summary output.
  - Compare old full lag-cycle extraction against narrow lag-cycle extraction for a small fixed date window.
  - Assert unchanged final overnight summary columns and values, except expected diagnostics/provenance differences.
  - Status: focused parity smoke compared the six revision raw fields for `2026-04-11__2026-04-11_t04_f00` between full extraction and narrow revision extraction; all matched exactly.

## Phase 3: Faster Extraction Prototype

- [x] Prototype a direct `wgrib2` or ecCodes station-summary extractor.
  - Bypass cfgrib/xarray for the production summary path.
  - Extract only KLGA nearest point and required neighborhood/crop stats.
  - Compute daily summaries directly from selected messages.
  - Keep current cfgrib path as the reference implementation until parity is proven.
  - Status: implemented as opt-in `--extract-method eccodes`. It iterates GRIB messages directly with ecCodes, builds the same wide-row metrics, supports batch-cycle extraction, and keeps `--extract-method cfgrib` as the default reference path.

- [ ] Benchmark direct extraction against current cfgrib extraction.
  - Use the same dates, cycles, fields, crop bounds, and worker settings.
  - Compare wall time, CPU time, memory pressure, and output parity.
  - Record results in HRRR diagnostics or a short benchmark note.

## Phase 4: Cloud-Native Prototype

- [ ] Prototype HRRR-Zarr access for the 2023-2026 training window.
  - Verify required variables, levels, forecast hours, and archive coverage.
  - Confirm KLGA point extraction and local-day summaries match the current GRIB path.
  - Keep GRIB path as fallback for gaps.

- [ ] Evaluate Kerchunk only if HRRR-Zarr is insufficient.
  - Use Kerchunk to avoid repeated GRIB scans while preserving cloud/range access.
  - Promote only if it is simpler or faster than the direct extractor for this station-summary workload.

## Definition Of Done

- [ ] Monthly overnight HRRR backfill is faster on a representative month.
- [ ] `hrrr.overnight.parquet` values are unchanged within expected numeric tolerance.
- [ ] Diagnostics distinguish download, crop/reduce, cfgrib/open, row build, provenance, and summary costs.
- [ ] Production command examples are documented in `tools/hrrr/README.md`.
