# HRRR-Zarr Phase 4 Notes

This note tracks the cloud-native HRRR-Zarr prototype for the KLGA overnight pipeline.

## Source

The public HRRR-Zarr archive is the University of Utah/MesoWest `hrrrzarr` bucket:

- archive overview: `https://mesowest.utah.edu/html/hrrr/`
- variable/layout docs: `https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_variables.html`
- Python loading examples: `https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/python_data_loading.html`
- Herbie example: `https://herbie.readthedocs.io/en/2023.12.4/user_guide/_bonus_notebooks/zarr_HRRR.html`

The bucket layout mirrors GRIB groups:

```text
sfc/YYYYMMDD/YYYYMMDD_HHz_fcst.zarr/LEVEL/VARIABLE/LEVEL/VARIABLE/
```

Example:

```text
sfc/20230204/20230204_05z_fcst.zarr/2m_above_ground/TMP/2m_above_ground/TMP/
```

## Prototype Tool

Use the metadata-only probe:

```bash
source .venv/bin/activate
python tools/hrrr/probe_hrrr_zarr.py \
  --start-date 2023-02-04 \
  --end-date 2023-02-04 \
  --selection-mode overnight_0005 \
  --summary-profile overnight \
  --output-json tools/hrrr/data/runtime/zarr_probes/2023-02-04_overnight_0005.json
```

The default probe uses only `requests` and checks public S3 `.zarray` metadata. It verifies:

- mapped Zarr level/variable paths for the HRRR feature contract
- retained forecast-hour coverage for each task
- anchor vs revision-cycle field requirements under `--summary-profile overnight`

It does not yet decode Zarr chunks or compute KLGA point values. That should be the next step only after deciding how much fallback to GRIB is acceptable.

## Initial Findings

Probed dates:

- `2023-02-04` with `--selection-mode overnight_0005 --summary-profile overnight`
- `2026-04-11` with `--selection-mode overnight_0005 --summary-profile overnight`

Both dates showed the same structural gaps:

- upper-level direct `RH` arrays are absent for `1000mb`, `925mb`, `850mb`, and `700mb`
- upper-level direct `SPFH` arrays are absent for `1000mb`, `925mb`, `850mb`, and `700mb`
- `925mb/HGT` is absent
- the latest non-full overnight cycle has shape `[18, 1059, 1799]` for tested revision fields, so forecast hour `f18` is not covered in Zarr even though the GRIB path can have that task

These gaps mean HRRR-Zarr is not a drop-in replacement for the current GRIB feature contract.

## Current Recommendation

Keep the GRIB path as the production fallback. HRRR-Zarr can still be useful for a narrower cloud-native prototype if it:

- derives upper-level RH from `TMP` and `DPT` where direct `RH` is absent
- treats upper-level `SPFH` and `925mb/HGT` as unavailable from Zarr unless another archive group is identified
- falls back to GRIB for non-full-cycle forecast hours that exceed the Zarr time dimension
- handles documented HRRR-Zarr constant-field gaps, especially solar-radiation fields at night

Kerchunk should only be evaluated if these Zarr gaps make direct Zarr access insufficient but GRIB byte-range access can still be simplified.

## Kerchunk Evaluation

Kerchunk is installed as a development/prototype dependency and has a separate probe:

```bash
source .venv/bin/activate
python tools/hrrr/probe_hrrr_kerchunk.py \
  --start-date 2023-02-04 \
  --end-date 2023-02-04 \
  --selection-mode overnight_0005 \
  --summary-profile overnight \
  --max-tasks 1 \
  --skip-messages 20 \
  --output-json tools/hrrr/data/runtime/kerchunk_probes/2023-02-04_overnight_0005_smoke.json
```

Use `--skip-messages 0` for a full-file reference scan. The probe keeps the current GRIB path as the reference and measures whether Kerchunk can build reusable references for the same remote GRIB files selected by the overnight task planner. It also records the selected-record byte-range footprint from the existing `.idx` sidecar logic so Kerchunk scan cost can be compared against the current selected-record download path.

Initial result on 2026-04-27:

- command target: `2023-02-04` with `--selection-mode overnight_0005 --summary-profile overnight --max-tasks 1`
- quick smoke: `--skip-messages 20` scanned 20 message references in 13.71s
- full-file scan: `--skip-messages 0` scanned 170 message references in 90.51s
- same task's current `.idx` path selected 6 records in 6 merged byte ranges, with `.idx` fetch/parse around 0.11s before range download

That first measurement argues against replacing the current selected-record path with full-file Kerchunk reference generation for the station-summary workload. Kerchunk could still be useful only if references are generated once, cached durably, and reused enough times to pay back the initial scan cost.
