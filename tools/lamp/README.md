# LAMP Tools

Station-based NOAA LAMP ingestion and extraction for `KLGA`.

Primary entrypoints:

- `python3 tools/lamp/fetch_lamp.py live --date-utc 2026-04-12 --cycle 0430`
- `python3 tools/lamp/fetch_lamp.py archive --start-utc-date 2025-01-01 --end-utc-date 2025-01-07 --cycle 0230 --cycle 0330 --cycle 0430`
- `python3 tools/lamp/build_lamp_klga_features.py tools/lamp/data/runtime/raw --write-long`

Runtime layout:

- `tools/lamp/data/runtime/raw`
- `tools/lamp/data/runtime/parsed`
- `tools/lamp/data/runtime/features`

`fetch_lamp.py` targets the station ASCII `lavtxt` bulletins. v1 intentionally excludes BUFR decoding and gridded GLMP.

## Archive Cycle Notes

For temperature modeling, use the full `lavtxt` guidance cycles on `HH30`.

Observed archive behavior:

- `HH30` cycles such as `0230`, `0330`, and `0430` contain full KLGA guidance with `TMP` and 25 forecast hours
- quarter-hour/non-`HH30` cycles such as `0345`, `0400`, `0445`, and `0500` can be 3-hour `CIG/VIS/OBV`-only aviation updates with no `TMP`
- the overnight `00:05 America/New_York` workflow should fetch `0230`, `0330`, and `0430` UTC, then select the latest issue available by the local cutoff
- standard time typically selects `0430Z`; daylight time typically selects `0330Z`

Known upstream archive gaps in the rebuilt `0230/0330/0430` set:

- `2023-04-09`
- `2023-04-12`
- `2023-06-13`
- `2024-11-04`
- `2024-11-05`
- `2024-11-06`

Treat LAMP as unavailable for those dates. Do not classify them as local extraction or parser failures unless a separate source confirms the NOAA archive has been repaired.

## Fetch Modes

### `fetch_lamp.py live`

Live fetches download published `lavtxt` bulletin files from NOMADS for a single UTC issue date and cycle.

Current behavior:

- accepts cycle tokens in `HHMM` or `HH`
- normalizes `HH` to `HH00`
- enforces the published 15-minute live cadence: `HH00`, `HH15`, `HH30`, or `HH45`
- attempts both the standard and extended bulletin names
- treats a missing extended bulletin as a skip instead of a hard failure
- prints `[start]`, `[progress]`, `[skip]`, and `[ok]` status lines during download

Useful flags:

- `--output-dir`
- `--overwrite`

Live outputs are partitioned under:

- `tools/lamp/data/runtime/raw/source=nomads/date_utc=YYYY-MM-DD/cycle=HHMM/`

### `fetch_lamp.py archive`

Archive fetches materialize issue-level ASCII files under the same raw runtime tree.

Current behavior:

- accepts an optional repeatable `--cycle` filter
- uses a monthly cycle-gzip fast path when a cycle filter is provided and the requested window is `31` days or less
- falls back to yearly tar extraction when the fast path is not applicable or fails
- supports a local archive cache via `--cache-dir`
- removes downloaded yearly tar files by default when using the default cache location unless `--keep-archive-tars` is enabled
- prints explicit progress/status lines during archive downloads and extraction

Useful flags:

- `--cycle`
- `--cache-dir`
- `--keep-archive-tars`
- `--overwrite`

Archive outputs are partitioned under:

- `tools/lamp/data/runtime/raw/source=archive/date_utc=YYYY-MM-DD/cycle=HHMM/`

## Raw Feature Builder

`build_lamp_klga_features.py` builds issue-level Parquet artifacts from raw ASCII files or directories.

Current behavior:

- accepts one or more raw ASCII files or directories and discovers bulletin inputs recursively
- parses each input, merges station frames, and filters to the requested station id
- groups rows by `init_time_utc` so each issue writes its own artifact set
- emits progress bars for both parse and build/write phases
- writes wide, provenance, and manifest Parquet for every issue
- optionally writes the merged raw long artifact when `--write-long` is enabled

Useful flags:

- `--output-dir`
- `--station-id`
- `--write-long`

Per-issue outputs are partitioned under:

- `tools/lamp/data/runtime/features/station_id=KLGA/date_utc=YYYY-MM-DD/cycle=HHMM/lamp.wide.parquet`
- `tools/lamp/data/runtime/features/station_id=KLGA/date_utc=YYYY-MM-DD/cycle=HHMM/lamp.provenance.parquet`
- `tools/lamp/data/runtime/features/station_id=KLGA/date_utc=YYYY-MM-DD/cycle=HHMM/lamp.manifest.parquet`
- optional `tools/lamp/data/runtime/features/station_id=KLGA/date_utc=YYYY-MM-DD/cycle=HHMM/lamp.long.parquet`

The raw manifest records issue-level status, warning text, row counts, and the exact artifact paths written for that issue.

## Example Commands

Live fetch with overwrite:

```bash
python3 tools/lamp/fetch_lamp.py live \
  --date-utc 2026-04-12 \
  --cycle 0430 \
  --overwrite
```

Archive fetch using the monthly cycle fast path:

```bash
python3 tools/lamp/fetch_lamp.py archive \
  --start-utc-date 2025-01-01 \
  --end-utc-date 2025-01-07 \
  --cycle 0230 \
  --cycle 0330 \
  --cycle 0430 \
  --cache-dir tools/lamp/data/runtime/archive_cache
```

Build raw issue-level features:

```bash
python3 tools/lamp/build_lamp_klga_features.py \
  tools/lamp/data/runtime/raw \
  --output-dir tools/lamp/data/runtime/features \
  --write-long
```
