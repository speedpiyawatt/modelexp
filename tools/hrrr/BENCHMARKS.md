# HRRR Benchmarks

## 2026-04-27: cfgrib vs ecCodes extraction

Purpose: benchmark the opt-in direct ecCodes extractor against the current cfgrib reference path for the production HRRR overnight summary workload.

Input:

- cached reduced multi-forecast GRIB: `data/runtime/backfill_overnight_2023_2026/hrrr_tmp/2023-02-04/scratch/reduced/batch/hrrr.20230204.t05z/selected_multiforecast.reduced.grib2`
- target local date: `2023-02-04`
- retained cycle: `2023-02-04T05Z`
- forecast hours: `0` through `18`
- task count: `19`
- provenance disabled
- summary profile: `overnight`

Results:

| Extractor | Total seconds | Open seconds | Row seconds | Successful tasks |
| --- | ---: | ---: | ---: | ---: |
| cfgrib | 26.32 | 22.96 | 3.37 | 19 |
| ecCodes | 23.72 | 0.00 | 0.31 | 19 |

Total speedup: cfgrib/ecCodes = `1.11x`.

Separate-process resource measurements:

| Extractor | Wall seconds | User CPU seconds | System CPU seconds | Max RSS KB |
| --- | ---: | ---: | ---: | ---: |
| cfgrib | 25.73 | 25.05 | 0.63 | 190,144 |
| ecCodes | 23.75 | 22.85 | 0.75 | 148,856 |

Parity:

- compared values: `17,328`
- task keys equal: `true`
- max numeric diff: `0.0001202`
- values differing by more than `1e-4`: `12`
- remaining differences above `1e-4` were derived relative-humidity floating-point differences

Implementation note:

- The first benchmark exposed an ecCodes APCP mismatch because the reduced GRIB contains both cumulative `0-N hour acc` and hourly `N-1-N hour acc` records for the same forecast step. The cfgrib reference path used the cumulative record. The direct ecCodes extractor now keeps the first APCP record for each filtered task step instead of overwriting it with the later hourly record.

Runtime artifact:

- `tools/hrrr/data/runtime/extraction_benchmarks/20260427T225741Z/benchmark_summary.json`
