# AGENTS.md

## Project Context

This workspace is for building a model and trading workflow for Polymarket weather events that resolve on the finalized KLGA daily high.

## Python Environment

This repo currently uses a local `uv`-managed virtual environment.

- prefer running Python and package commands through the repo-local `.venv`
- create or refresh the environment with `uv venv .venv --python 3.11`
- install dependencies with `uv pip install --python .venv/bin/python -r requirements-dev.txt`
- the checked-in dependency set lives in `requirements.txt` and `requirements-dev.txt`; runtime code currently requires `numpy` directly in addition to `pandas`, `xarray`, `cfgrib`, `eccodes`, `rasterio`, `pyarrow`, and `requests`
- do not treat `.venv` as portable across machines or operating systems; copy the repo, then recreate `.venv` on the destination host with the local Python there
- avoid relying on the global/default Python for repo work unless the user explicitly asks for that
- when giving commands to the user, prefer forms that activate `.venv` first or explicitly target `.venv/bin/python`

Reference event:

- `https://polymarket.com/event/highest-temperature-in-nyc-on-april-11-2026`

The practical goal is not generic weather prediction. The goal is to estimate the probability of each settlement bin for the relevant local day, compare those probabilities against market prices, and trade only when the modeled edge survives fees, spread, and execution risk.

## What The Market Resolves To

Treat the written settlement rule as the source of truth:

- "NYC" means the **LaGuardia Airport station (`KLGA`)**
- resolution uses the **highest temperature recorded at KLGA**
- the resolution source is the finalized **Wunderground daily history page for `KLGA`**
- resolution is in **whole degrees Fahrenheit**
- the market resolves to the **temperature bin** containing that final daily high
- revisions after the market's finalized timeframe should be ignored for settlement purposes

Do not optimize for Central Park, citywide averages, or vague "NYC weather" narratives. Optimize for **final KLGA settlement accuracy**.

## Current Primary Objective

The current primary modeling milestone is the **12:05 a.m. America/New_York overnight fair-value model**.

Its job is:

- at `12:05 a.m. America/New_York`, produce calibrated probabilities for the final KLGA daily max-temperature bin for that local day
- output expected final daily high at KLGA
- preserve a stable internal temperature target so event-specific Polymarket bins can be mapped at runtime

The intraday updater remains in scope, but it is a later layer built on top of the overnight baseline rather than the main current milestone.

## Separate Experiments

The folder `experiments/no_hrrr_model/` is a separate no-HRRR modeling experiment.

- treat it as independent from the main overnight model described in the root `MODEL.md`
- do not assume design choices in `experiments/no_hrrr_model/MODEL.md` apply to the canonical workflow
- do not remove HRRR from the main model contract because of this experiment
- keep experiment-specific artifacts, notes, and schema names inside the experiment unless the user explicitly asks to promote them
- the experiment still uses the same settlement target: final KLGA daily high for the `America/New_York` local station day

The folder `experiments/withhrrr/` is the HRRR-inclusive version of the no-HRRR experiment and is now the preferred experiment when the user asks for the best overnight fair-value model with current inputs.

- it keeps the same model family as `experiments/no_hrrr_model`: residual LightGBM quantile models, calibrated final-Tmax quantiles, a 1F temperature ladder, and an event-bin adapter
- it learns from Wunderground, NBM, LAMP, and HRRR source blocks; HRRR is both a learned input and part of the selected equal 3-way source anchor
- the current selected anchor is `equal_3way`: `(nbm_tmax_open_f + lamp_tmax_open_f + hrrr_tmax_open_f) / 3`; HRRR-only was not promoted because holdout metrics were worse than the residual model
- the current selected with-HRRR model uses the Source-Trust upgrade: model candidate `very_regularized_min_leaf70_lgbm_350`, feature profile `high_disagreement_weighted`, weight profile `high_disagreement_weighted`, quantile calibration `conformal_intervals`, distribution `normal_iqr`, and ladder reliability `bucket_reliability_s1_00`
- HRRR/source disagreement diagnostics are promoted as features, including HRRR-vs-LAMP/NBM deltas, native-NBM deltas, source ranks, warmest/coldest flags, WU-last-temp-vs-source deltas, and disagreement regimes
- current selected with-HRRR defaults are documented in `experiments/withhrrr/README.md`, `TODO.md`, and `IMPLEMENTATION_TODO.md`
- future agents must update both `experiments/withhrrr/TODO.md` and `experiments/withhrrr/IMPLEMENTATION_TODO.md` when changing selected model behavior, inference behavior, or reported metrics

## Event And Bin Architecture

Polymarket structures these weather contracts as **separate daily events**, and each event can have its own set of outcome markets.

Reference:

- Polymarket docs on markets and events: `https://docs.polymarket.com/concepts/markets-events`

That means:

- do **not** train the model directly on a single day's Polymarket bins as fixed class labels
- do **not** hardcode today's event ladder into the core weather model
- do hardcode the **settlement station and resolution logic**
- do treat the market bins as an **event-specific adapter layer**

The correct architecture is:

1. predict a stable weather target first
2. parse the current event's actual outcome bins
3. map the internal temperature distribution into those exact bins at runtime
4. compare mapped probabilities against market prices

The stable target should be one of:

- continuous final daily `KLGA` Tmax in Fahrenheit
- a fine-grained internal ladder you control, such as 1°F bins

The key rule is: **predict temperature first, bins second.**

## Modeling Policy

This is a **tabular forecasting + calibration** problem, not a "build a weather model from scratch" problem.

Priority order:

1. match settlement labels correctly
2. build the overnight market-open baseline from implemented weather sources
3. calibrate the overnight temperature distribution into fair values
4. add the intraday updater
5. paper trade before trusting live deployment

If there is a tradeoff between meteorological elegance and settlement-matched reliability, choose settlement-matched reliability.

## Overnight Model Contract

The overnight model is source-aware and stacked. It is not one giant flat table shoved into one learner.

The overnight source blocks are:

- Wunderground late-evening station state
- NBM overnight prior
- LAMP overnight station summary
- HRRR overnight summary and revisions

The hard as-of rule is:

- every overnight feature used for `target_date_local` must be available by `target_date_local 00:05 America/New_York`
- training and inference must use the same cutoff logic

The canonical overnight training grain is:

- one row per `target_date_local`
- keyed by `target_date_local`, `station_id`, and `selection_cutoff_local`

Detailed model design, feature families, and architecture live in `MODEL.md`.

## Current Repo Reality

Anchor planning and implementation to what already exists in this repo:

- Wunderground is the canonical settlement-aligned source for `final_tmax_f` labels and intraday observation state
- `tools/lamp/build_lamp_overnight_features.py` already produces one LAMP overnight summary row per `target_date_local`
- archived LAMP `lavtxt` temperature guidance is on full `HH30` cycles; quarter-hour/non-`HH30` archive cycles such as `0345`, `0400`, `0445`, and `0500` can be short `CIG/VIS/OBV`-only aviation updates and must not be used as the temperature source
- for the `00:05 America/New_York` overnight cutoff, fetch candidate full LAMP cycles `0230`, `0330`, and `0430` UTC, then let `build_lamp_overnight_features.py` select the latest issue at or before the local cutoff
- `tools/hrrr` already uses an overnight local-day summary design with a `00:05` cutoff and retained-cycle revision logic
- `tools/weather` already provides canonical normalization contracts for downstream training
- `experiments/no_hrrr_model/no_hrrr_model/run_online_inference.py` runs one-date no-HRRR online inference from Wunderground, LAMP, NBM, and Polymarket bins
- `experiments/withhrrr/withhrrr_model/run_online_inference.py` runs one-date with-HRRR online inference from Wunderground, LAMP, NBM, HRRR, and Polymarket bins
- `tools/weather/run_server_dual_inference.py` is the local one-argument command for asking the DigitalOcean server to run both no-HRRR and with-HRRR inference and print a comparison

Do not promise datasets or pipelines that are not currently present.

## KLGA-Specific Weather Framing

For `KLGA`, focus on:

- synoptic setup
- frontal timing
- cloud timing
- wind shifts
- dry vs humid air mass behavior
- rate of warming after sunrise
- whether the airport is running warmer or cooler than forecast

Potential later upgrades can include:

- nearby-station temperature gradients
- cloud products or satellite-derived timing features

Those are later extensions, not overnight-model blockers.

## Trading Principles

The model exists to support trading, so output must be decision-ready.

- always build the core model around a **temperature distribution**, not a fixed event ladder
- convert that temperature distribution into **today's actual event bins** before trading
- hard-zero bins that are already impossible because they are below `max_so_far`
- prefer calibrated probabilities over flashy model complexity
- compare model probabilities against actual market prices and liquidity
- respect fees, slippage, spread, and position sizing
- treat paper trading and backtesting as mandatory before any serious live use

## Non-Goals For The Overnight Baseline

Do not drift into these unless explicitly asked:

- building a weather simulator from scratch
- RTMA or URMA ingestion
- nearby-station datasets
- satellite or cloud-imagery pipelines
- market microstructure or orderbook features in the overnight baseline
- premature feature-store or infra complexity

## Working Rules For Future Agents

When contributing in this repo:

- keep the project centered on **Polymarket settlement accuracy for `KLGA`**
- prefer small, testable, reproducible pipelines over speculative architecture
- use absolute dates in notes and outputs when discussing a specific market day
- be explicit about timezone assumptions; the market day is tied to the local station day in `America/New_York`
- separate clearly between settlement labels, historical features, live features, model outputs, and trading decisions
- keep the core weather target stable even if Polymarket changes event bins from day to day
- ignore Polymarket's AI-generated market commentary when it conflicts with the written resolution rules; the written rules win

For download-heavy or long-running network steps:

- prefer giving the user the exact command to run locally instead of running the download directly from the agent
- wait for the user to confirm the download or fetch step is complete before continuing
- do not poll or heartbeat-check external downloads when the user has agreed to run them manually
- once the user confirms completion, continue with validation, parsing, merging, or downstream local processing
- for one-date online fair-value inference, prefer the server runner instead of doing heavy GRIB downloads locally:

```bash
.venv/bin/python tools/weather/run_server_dual_inference.py YYYY-MM-DD
```

- the server runner assumes SSH access to `root@198.199.64.163` and repo path `/root/modelexp`; override with `MODELEXP_SERVER`, `MODELEXP_REMOTE_REPO`, or `MODELEXP_REMOTE_OUTPUT_ROOT` only when needed
- as of 2026-04-29, `/root/modelexp` is expected to include commit `20330a1` or newer plus synced ignored with-HRRR runtime model/evaluation artifacts under `experiments/withhrrr/data/runtime/`; a plain Git pull is not enough to refresh those ignored artifacts
- the server runner runs no-HRRR and HRRR source work in parallel, reuses the no-HRRR Wunderground/LAMP/NBM artifacts to build the with-HRRR prediction, then returns a local text comparison
- server runner outputs are under `/root/modelexp/data/runtime/server_dual_inference/YYYY-MM-DD/` and retain final prediction JSONs, `comparison.json`, status manifests, and logs
- the server runner deletes downloaded/intermediate source artifacts after producing the predictions

Manual current-day Tmax requests:

- when the user asks for a manual/current Tmax forecast, first record the request time in both the machine timezone and `America/New_York`
- pull LAMP, NBM, and HRRR using the latest currently relevant issue/cycle, not the overnight `00:05` anchor, unless the user explicitly asks for the overnight model
- target only the plausible remaining Tmax window around KLGA peak heating, roughly late morning through late afternoon local, and avoid pulling the entire local day when the request is intraday
- for manual current-day NBM/HRRR runs, prefer a fresh `data/runtime/manual_tmax_YYYY_MM_DD_HHMM` run root and delete stale/partial artifacts before re-running
- on an M4-class local machine, prefer multiple workers for manual pulls: use at least 4 download workers when the tool supports it, keep `wgrib2` threads at `1`, and use bounded reduce/extract queues so downloads, crop, and extraction can overlap without exhausting disk
- after the user confirms the manual downloads are complete, analyze the latest-cycle raw rows directly if no current-day summary builder exists; do not reinterpret latest-cycle data as an overnight fair-value row

Optimization guidance:

- treat `--selection-mode overnight_0005` in NBM and HRRR as an overnight-validation or short-window replay optimization, not a full-history backfill default; for NBM it also narrows source selection to one latest cutoff-eligible issue per target day
- treat NBM `--lead-workers` as a general throughput optimization for separated per-lead/raw-debug runs when local CPU is available
- for NBM monthly overnight backfills with `--batch-reduce-mode cycle`, scale by target day instead of by lead; the measured 4-core profile is `--day-workers 4 --download-workers 4 --reduce-workers 1 --extract-workers 1 --reduce-queue-size 2 --extract-queue-size 2 --wgrib2-threads 1`
- do not use `--adaptive-workers` for routine NBM batch-cycle runs; use static workers or `--smart-workers --cpu-cores 4`
- for NBM overnight daily-high backfills, native `TMIN` may be absent in the selected 04Z core source inventory even when hourly `TMP` and native `TMAX` are present; this can produce `missing_required_feature_count=1` and `coverage_complete=false`, but by itself is not a batch-cropping failure and should not block daily-high modeling rows
- for LAMP overnight backfills, treat `0230`, `0330`, and `0430` UTC as the full-guidance candidate cycles. Do not backfill temperature features from `0345`, `0400`, `0445`, or `0500` UTC archive files; observed 2023-2025 archive files for those cycles were 3-hour `CIG/VIS/OBV`-only products with no `TMP`.
- known public LAMP archive gaps after the `HH30` rebuild: `2023-04-09`, `2023-04-12`, `2023-06-13`, `2024-11-04`, `2024-11-05`, and `2024-11-06` are absent from the `0230/0330/0430` monthly archive caches. Treat LAMP as unavailable for those target dates rather than as a local extraction failure.
- treat HRRR merged byte-range downloads and deferred month-manifest flushing as general throughput optimizations that apply beyond short-window runs
- for HRRR production overnight backfills, prefer the completed throughput options unless debugging requires the legacy path: `--batch-reduce-mode cycle --summary-profile overnight --crop-method auto --crop-grib-type same --range-merge-gap-bytes 65536 --wgrib2-threads 1 --extract-method wgrib2-bin`
- use HRRR `--skip-provenance` when the run only needs overnight summary features and provenance parquet output is not needed; manifests should then record `provenance_written=false`
- keep HRRR `--extract-method cfgrib` as the conservative reference backend, but production online inference now uses `--extract-method wgrib2-bin`
- `wgrib2-bin` dumps selected/reduced GRIB messages directly to binary arrays and computes features in NumPy; this was the first major promising extractor speedup compared with Python full-array ecCodes/cfgrib decode
- `wgrib2-ijbox-bin` is the next larger optimization direction for bypassing reduced GRIB creation from uncropped selected GRIB input; treat it as a separate future step unless the current code already selects it explicitly
- the HRRR monthly helper forwards the optimized raw-builder flags, including `--batch-reduce-mode`, `--range-merge-gap-bytes`, `--crop-method`, `--crop-grib-type`, `--wgrib2-threads`, `--extract-method`, `--summary-profile`, and `--skip-provenance`
- HRRR-Zarr Phase 4 has started but is not a drop-in GRIB replacement; `tools/hrrr/probe_hrrr_zarr.py` found missing upper-level direct `RH`, missing upper-level direct `SPFH`, missing `925mb/HGT`, and missing `f18` coverage on the latest non-full overnight cycle for tested dates. Keep GRIB fallback in any Zarr prototype.
- do not assume an optimization is short-window-only unless it changes source selection scope rather than execution throughput

Online inference worker defaults:

- no-HRRR online inference NBM uses `--download-workers 4 --reduce-workers 2 --extract-workers 2 --wgrib2-threads 1`
- with-HRRR online inference NBM uses `--download-workers 4 --reduce-workers 2 --extract-workers 2 --wgrib2-threads 1`
- with-HRRR online inference HRRR uses `--download-workers 4 --reduce-workers 2 --extract-workers 2 --wgrib2-threads 1`
- keep `wgrib2-threads=1` unless benchmarking proves otherwise; process-level parallelism has been more useful than multi-threading individual `wgrib2` calls

Online inference artifact policy:

- both online inference CLIs delete downloaded/intermediate runtime artifacts after a successful prediction by default
- final prediction JSONs and status manifests are retained
- pass `--keep-artifacts` when debugging and you need Wunderground/LAMP/NBM/HRRR source artifacts or prediction feature parquet retained
- cleanup only targets paths under `--runtime-root` that the wrapper created; external paths supplied through `--skip-*`/explicit root arguments must not be deleted

Server dual inference:

- use `.venv/bin/python tools/weather/run_server_dual_inference.py YYYY-MM-DD` from the local repo when the user wants one-date production-style inference and comparison
- the script starts no-HRRR online inference and HRRR source build concurrently on `root@198.199.64.163`, then builds the with-HRRR row from no-HRRR WU/LAMP/NBM artifacts plus HRRR summary
- the script prints expected Tmax, anchor, distribution method, event-bin probabilities, and with-HRRR minus no-HRRR probability differences
- a 2026-04-29 validation run for `2026-04-28` returned no-HRRR expected `65.07F`, with-HRRR expected `64.63F`, and finalized Wunderground/Synoptic peak near `64F`; with-HRRR was closer for that date
- if the script fails, inspect `/root/modelexp/data/runtime/server_dual_inference/YYYY-MM-DD/no_hrrr.log` and `hrrr.log`

Synoptic API spot checks:

- Synoptic Data timeseries endpoint can be used for KLGA observation spot checks around expected Tmax windows, but do not store API tokens in repo files or AGENTS.md
- endpoint shape: `https://api.synopticdata.com/v2/stations/timeseries?stid=KLGA&start=YYYYMMDDHHMM&end=YYYYMMDDHHMM&vars=air_temp&token=...`
- on 2026-04-29, a Synoptic check for `2026-04-28 10:00..18:00 America/New_York` returned max `64.4F`, matching the finalized `64F` settlement label behavior

## Weather Tool Layout

The canonical weather tooling layout is:

- `wunderground`
- `tools/nbm`
- `tools/lamp`
- `tools/hrrr`
- `tools/weather`

Canonical data split:

- committed Wunderground history files live under `wunderground/output/history`
- committed regression fixtures live under `tools/hrrr/data/fixtures`
- generated runtime outputs live under tool-local `data/runtime/...`

Do not treat committed fixtures as production outputs.

## Wunderground Settlement Dataset

The top-level `wunderground/` folder is the canonical settlement-aligned source in this repo.

- source station and resolution target: `KLGA`
- source location id: `KLGA:9:US`
- source timezone for interpretation: `America/New_York`
- storage contract: one JSON file per local calendar day under `wunderground/output/history`
- file naming contract: `KLGA_9_US_YYYY-MM-DD.json`

Working rules:

- prefer this dataset when building or validating `final_tmax_f` labels that must track Polymarket settlement behavior
- interpret `valid_time_gmt` in `America/New_York` before deriving local-day features or max-so-far state
- do not assume a normalized `24` rows per day; preserve the raw observation stream
- treat DST truncation days as known upstream anomalies, not automatic fetch failures
- ignore volatile response metadata like `transaction_id` and `expire_time_gmt` when comparing re-fetches

Canonical validation command:

- `python3 wunderground/validate_history.py --history-dir wunderground/output/history`

Canonical verification entrypoint:

- `python3 tools/weather/run_verification_suite.py`

Repository rule:

- new weather tooling belongs under `tools/`, not `research/`

## Immediate Build Order

1. create a settlement-matched KLGA label dataset with `final_tmax_f` as the core target
2. create `training_features_overnight` as the merged overnight feature table
3. train the stacked overnight model for continuous temperature output and bin probabilities
4. calibrate overnight fair values
5. build the intraday updater
6. build the event adapter and trading layer
7. paper trade against actual Polymarket prices
8. only then add nearby-station or cloud-timing upgrades

## One-Line Summary

We are building a **settlement-aware KLGA daily-high model** that predicts true final station temperature first and maps that distribution into each day's Polymarket bins second, with the first priority being **correct `KLGA` settlement alignment** and the current milestone being the `12:05 a.m. America/New_York` overnight fair-value model.
