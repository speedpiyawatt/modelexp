# Wunderground KLGA History Dataset

This workspace contains a local dataset of historical airport observations for LaGuardia Airport (`KLGA`) pulled from the Weather.com historical observations endpoint that Wunderground uses to hydrate its `Daily Observations` table.

This README is intended to be future context for anyone working in this folder: what the data is, where it came from, how it is structured, what is normal vs anomalous, and how the helper scripts work.

## What This Dataset Is

- Source station: `KLGA`
- Source location id: `KLGA:9:US`
- Geographic context: New York / LaGuardia Airport
- Source timezone for interpretation: `America/New_York`
- Storage format: one JSON file per calendar day
- Coverage currently present: `2023-01-01` through `2026-04-11`

The dataset is not scraped from rendered HTML rows. It is downloaded directly from the historical observations API that the Wunderground history page requests client-side after hydration.

## Actual Source Endpoint

The relevant endpoint is:

```text
https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json
```

Typical query parameters:

```text
apiKey=<key>
units=e
startDate=YYYYMMDD
endDate=YYYYMMDD
```

Example:

```text
https://api.weather.com/v1/location/KLGA:9:US/observations/historical.json?apiKey=...&units=e&startDate=20250411&endDate=20250411
```

## Why This Matters

Earlier in the investigation, `v2/pws/history` looked relevant because Wunderground also references personal weather station data on the page. That was the wrong backend for the `Daily Observations` table.

The correct source for the table is the airport historical observations feed above. This is why:

- the values match the rendered table
- the timestamps match the table row times like `12:51 AM`, `1:51 AM`, etc.
- the personal weather station history feed did not match the page row-for-row

## Relationship To The Wunderground Page

The Wunderground page at:

```text
https://www.wunderground.com/history/daily/us/ny/new-york-city/KLGA/date/2025-4-11
```

does not fully server-render the `Daily Observations` rows in the initial HTML response.

Instead:

1. the first HTML response may show `No Data Recorded`
2. the page hydrates client-side
3. it requests the airport history API
4. it renders the table from that response

That is why inspecting only raw HTML can be misleading.

## File Layout

Daily files are stored in:

```text
output/history/
```

Naming convention:

```text
KLGA_9_US_YYYY-MM-DD.json
```

Examples:

- `output/history/KLGA_9_US_2023-01-01.json`
- `output/history/KLGA_9_US_2025-04-11.json`

There are also two helper scripts in the repo root:

- `fetch_daily_history.py`
- `validate_history.py`

## JSON Structure

Each file is a JSON object with two top-level keys:

```json
{
  "metadata": { ... },
  "observations": [ ... ]
}
```

### `metadata`

Typical metadata fields:

- `language`
- `transaction_id`
- `version`
- `location_id`
- `units`
- `expire_time_gmt`
- `status_code`

Important note:

- `transaction_id` and `expire_time_gmt` are response metadata and can differ between downloads even when the actual observations are the same.
- A byte-for-byte comparison against a live re-fetch may differ because of metadata ordering and response metadata churn, even when the observations themselves match.

### `observations`

`observations` is an array of observation objects for the given local calendar day.

Common fields include:

- `valid_time_gmt`: observation timestamp in UTC epoch seconds
- `obs_name`: station display name
- `temp`: temperature in Fahrenheit
- `dewPt`: dew point in Fahrenheit
- `rh`: relative humidity percent
- `pressure`: pressure in inches
- `wdir`: wind direction in degrees
- `wdir_cardinal`: wind direction as a cardinal string
- `wspd`: wind speed in mph
- `gust`: wind gust in mph, may be `null`
- `precip_hrly`: hourly precipitation in inches
- `wx_phrase`: weather condition phrase
- `vis`: visibility
- `wc`: wind chill / feels-like related field used by the source
- `uv_index`
- `clds`

Example observation shape:

```json
{
  "valid_time_gmt": 1744347060,
  "temp": 43,
  "dewPt": 28,
  "rh": 56,
  "pressure": 30.19,
  "wdir_cardinal": "E",
  "wspd": 12,
  "gust": null,
  "precip_hrly": 0,
  "wx_phrase": "Cloudy"
}
```

## How To Interpret Timestamps

The source timestamp is `valid_time_gmt`, which is UTC.

For human interpretation and alignment with the Wunderground page, convert it to:

```text
America/New_York
```

Example:

- `1744347060` UTC converts to `2025-04-11 12:51 AM` in `America/New_York`

That is why the page rows are typically aligned around `:51` past the hour.

## Why Observation Counts Vary By Day

This is expected.

The dataset is not a normalized `24 rows per day` product. It is the raw airport observation stream for the day.

Normal cases:

- `24` observations: roughly one hourly observation
- `23` observations: common on fall DST transition days
- `2` or `3` observations: spring DST anomaly from the upstream API

High-count cases:

- `>24` observations mean the airport published additional special observations inside the hour
- this usually happens during changing weather such as rain, snow, sleet, wind shifts, visibility changes, or pressure changes

Examples observed in this dataset:

- `2025-02-04`: exactly `24` rows, all hourly
- `2025-01-31`: `37` rows with intra-hour updates
- `2025-02-06`: `46` rows with many weather transition observations

This is normal for this source.

## Known Source Anomalies

These are source behaviors, not fetcher bugs.

### 1. Spring DST-start truncation

The following DST-start dates have extremely low counts:

- `2023-03-12`
- `2024-03-10`
- `2025-03-09`
- `2026-03-08`

Observed counts:

- `2` or `3` rows

This is an upstream API anomaly. The files are valid downloads, but the source itself returns a truncated day.

### 2. Fall DST-end count shifts

The fall DST-end dates tend to have `23` rows rather than a clean `24` or `25`.

Examples:

- `2023-11-05`
- `2024-11-03`
- `2025-11-02`

Again, this is source behavior.

### 3. Recent day incompleteness

Very recent dates may have low counts simply because the day was still in progress when fetched.

Example from the validator run:

- `2026-04-11` had `6` rows

That is expected if the fetch ran before the day completed.

## Dataset Quality Summary

The dataset has been validated with the included validator script.

Validation results at the time of writing:

- files: `1197`
- first day: `2023-01-01`
- last day: `2026-04-11`
- missing days: `0`
- min count: `2`
- median count: `25`
- max count: `59`
- status issues: none
- timestamp issues: none

So the dataset is structurally sound.

The only flagged issues are expected source-side anomalies:

- DST transition days
- one unusually low non-DST day
- high-density weather-event days
- very recent incomplete days

## Fetch Script

File:

```text
fetch_daily_history.py
```

Purpose:

- fetch one day at a time
- save one JSON file per day
- apply a conservative delay between requests
- retry `429` and `5xx` responses with exponential backoff and jitter
- skip already-downloaded files unless `--force` is passed

Default behavior:

- start date: `2025-01-01`
- end date: `today`
- delay: `1.5` seconds
- retries: `5`

Example usage:

```bash
python3 fetch_daily_history.py
python3 fetch_daily_history.py --start-date 2023-01-01
python3 fetch_daily_history.py --start-date 2023-01-01 --end-date 2024-12-31
python3 fetch_daily_history.py --force
```

## Validator Script

File:

```text
validate_history.py
```

Purpose:

- check missing dates
- check response status codes
- check timestamp ordering
- ensure observations stay within the day
- flag DST transition dates
- flag low-count and high-count days
- flag recent possibly incomplete days

Example usage:

```bash
python3 validate_history.py
python3 validate_history.py --recent-days 3
python3 validate_history.py --low-count-threshold 18 --high-count-threshold 50
```

## Practical Guidance

If you are using this dataset for analysis:

- treat each file as the raw source truth for that day
- do not assume exactly `24` rows
- convert timestamps to `America/New_York`
- handle DST dates as special cases
- treat the most recent date or two as potentially incomplete

If you need a normalized dataset:

- resample to one observation per hour
- or choose a rule such as “latest observation within each local hour”

That normalization is not stored in this repo yet. The current dataset is intentionally raw.

## Recommended Assumptions For Future Work

- The historical observations endpoint is the authoritative source for the Wunderground `Daily Observations` table for `KLGA`.
- Varying observation counts are normal.
- High-count days are usually weather-event days, not duplicate download errors.
- DST-start low-count days are upstream anomalies.
- Re-fetch comparisons should ignore metadata ordering and mutable metadata fields.

