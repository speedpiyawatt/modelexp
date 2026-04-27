# No-HRRR Experiment Implementation TODO

This file is the working implementation plan for `experiments/no_hrrr_model`.

Agents must keep this file current while working:

- Update task status as work progresses.
- Use `Status: done` only after implementation is complete and locally verified.
- Use `Status: not done` for pending work.
- Use `Status: blocked` only when the blocker is explicit and written down.
- Add a short note with date, command, or artifact path when marking a task done.
- Keep all no-HRRR experiment code, derived data, models, reports, and manifests under `experiments/no_hrrr_model/`.
- Existing source data may be read from canonical locations, but do not write experiment outputs back into canonical tool folders.
- Do not add HRRR inputs, `hrrr_` fields, or `meta_hrrr_*` placeholders to this experiment.

Long-running or download-heavy commands:

- Do not run long downloads or long backfills from the agent.
- Give the exact command to the user and wait for the user to say it is complete.
- After user confirmation, continue with validation, parsing, merging, training, or evaluation.
- Reasonable short local validation commands may be run by the agent.

## Source Inputs

Read-only source inputs currently expected:

- Wunderground history: `wunderground/output/history`
- NBM overnight data: `data/runtime/backfill_overnight/nbm_overnight`
- LAMP issue-level features: `tools/lamp/data/runtime/features_full/station_id=KLGA`

Known local coverage as of 2026-04-25:

- Wunderground history: `2023-01-01` through `2026-04-11`
- NBM overnight: `2023-01-01` through `2025-12-31`
- LAMP issue-level features: approximately `2023-01-01` through `2025-12-31`

Use `America/New_York` for the target local station day and the `00:05` local overnight cutoff.

## Target Layout

Status: done

Note 2026-04-25: created experiment package, WU/LAMP wrapper entrypoints, README, requirements file, runtime folders through local builds, and tests under `experiments/no_hrrr_model/tests/`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests`.

Create this experiment-local layout as needed:

```text
experiments/no_hrrr_model/
  MODEL.md
  IMPLEMENTATION_TODO.md
  README.md
  requirements.txt
  no_hrrr_model/
    __init__.py
    contracts.py
    build_wu_tables.py
    build_lamp_overnight.py
    build_training_features.py
    normalize_features.py
    train_quantile_models.py
    evaluate.py
    distribution.py
    event_bins.py
  data/runtime/
    wunderground/
    lamp_overnight/
    training/
    models/
    evaluation/
    predictions/
  tests/
```

## TODO

### 1. Build Experiment-Local Wunderground Tables

Status: done

Note 2026-04-25: ran `.venv/bin/python wunderground/build_training_tables.py --history-dir wunderground/output/history --output-dir experiments/no_hrrr_model/data/runtime/wunderground`. Outputs: `labels_daily.parquet` with 1,197 rows and `wu_obs_intraday.parquet` with 33,123 rows.

Goal:

- Build settlement labels and pre-cutoff observation inputs for this experiment.
- Output into `experiments/no_hrrr_model/data/runtime/wunderground/`.

Preferred short command:

```bash
.venv/bin/python wunderground/build_training_tables.py \
  --history-dir wunderground/output/history \
  --output-dir experiments/no_hrrr_model/data/runtime/wunderground
```

Expected outputs:

- `experiments/no_hrrr_model/data/runtime/wunderground/labels_daily.parquet`
- `experiments/no_hrrr_model/data/runtime/wunderground/wu_obs_intraday.parquet`

If additional Wunderground dates need to be downloaded, give this command to the user instead of running it:

```bash
.venv/bin/python wunderground/fetch_daily_history.py \
  --start-date YYYY-MM-DD \
  --end-date YYYY-MM-DD \
  --output-dir experiments/no_hrrr_model/data/runtime/wunderground/history
```

Then wait for the user to confirm completion.

### 2. Build Experiment-Local LAMP Overnight Summaries

Status: done

Note 2026-04-25: one-day validation completed for `2023-08-08` with `.venv/bin/python tools/lamp/build_lamp_overnight_features.py --features-root tools/lamp/data/runtime/features_full/station_id=KLGA --output-dir experiments/no_hrrr_model/data/runtime/lamp_overnight --start-local-date 2023-08-08 --end-local-date 2023-08-08`. Full-history build still needs to be run by the user because it is the long date-range step.

Note 2026-04-25: user completed the full-history build. Validated `1,096` `lamp.overnight.parquet` files under `experiments/no_hrrr_model/data/runtime/lamp_overnight`, covering `2023-01-01` through `2025-12-31`.

Goal:

- Use existing LAMP issue-level feature files as read-only inputs.
- Write selected overnight rows into the experiment folder.
- Preserve the `00:05 America/New_York` cutoff.
- Use full temperature-guidance cycles only; do not use short `CIG/VIS/OBV`-only cycles as temperature sources.

Potentially long command. Give this to the user if the date range is large:

```bash
.venv/bin/python tools/lamp/build_lamp_overnight_features.py \
  --features-root tools/lamp/data/runtime/features_full/station_id=KLGA \
  --output-dir experiments/no_hrrr_model/data/runtime/lamp_overnight \
  --start-local-date 2023-01-01 \
  --end-local-date 2025-12-31
```

Expected output shape:

```text
experiments/no_hrrr_model/data/runtime/lamp_overnight/
  target_date_local=YYYY-MM-DD/
    lamp.overnight.parquet
    lamp.overnight.manifest.parquet
```

After user confirms completion, validate row counts and date coverage locally.

### 3. Create No-HRRR Training Feature Builder

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/build_training_features.py`. Verified one-day output for `2023-08-08`: `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet` and manifest. Core row had `final_tmax_f=80.0`, `nbm_tmax_open_f=79.790022`, `lamp_tmax_open_f=82.0`, `anchor_tmax_f=80.895011`, `model_training_eligible=True`, and no HRRR columns. Missing-source rows are retained with `model_training_eligible=False`.

Note 2026-04-25: rebuilt full table for `2023-01-01` through `2025-12-31`. Output has `1,096` rows and `202` columns. Audit passed with no HRRR columns. `1,094` rows are model-eligible. `2024-11-05` and `2024-11-06` are retained but ineligible because `lamp_tmax_open_f` is missing.

Goal:

- Implement `experiments/no_hrrr_model/no_hrrr_model/build_training_features.py`.
- Merge Wunderground labels, Wunderground pre-cutoff features, NBM overnight rows, and LAMP overnight rows.
- Write only experiment-local outputs.

Inputs:

- `experiments/no_hrrr_model/data/runtime/wunderground/labels_daily.parquet`
- `experiments/no_hrrr_model/data/runtime/wunderground/wu_obs_intraday.parquet`
- `data/runtime/backfill_overnight/nbm_overnight`
- `experiments/no_hrrr_model/data/runtime/lamp_overnight`

Required output:

- `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet`
- `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.manifest.json`

Required core columns:

- `target_date_local`
- `station_id`
- `selection_cutoff_local`
- `final_tmax_f`
- `final_tmin_f`
- `nbm_tmax_open_f`
- `lamp_tmax_open_f`
- `anchor_tmax_f`
- `nbm_minus_lamp_tmax_f`
- `target_residual_f`

Required formulas:

```text
nbm_tmax_open_f = fahrenheit(nbm_temp_2m_day_max_k)
lamp_tmax_open_f = lamp_day_tmp_max_f_forecast
anchor_tmax_f = 0.5 * nbm_tmax_open_f + 0.5 * lamp_tmax_open_f
nbm_minus_lamp_tmax_f = nbm_tmax_open_f - lamp_tmax_open_f
target_residual_f = final_tmax_f - anchor_tmax_f
```

Rules:

- One row per `target_date_local`, `station_id`, and `selection_cutoff_local`.
- No HRRR columns.
- No fixed Polymarket event-bin labels as model targets.
- Keep missing-source flags explicit, e.g. `meta_nbm_available`, `meta_lamp_available`, `meta_wu_obs_available`.
- Keep rows with missing NBM or LAMP for coverage diagnostics, but mark them `model_training_eligible=False`; model training must filter to eligible rows.

### 4. Add No-HRRR Contract And Audit

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/contracts.py` with required-column, duplicate-key, HRRR-prefix, residual-formula, cutoff timestamp, and eligible-row checks. Verified by pytest and by the one-day builder audit.

Goal:

- Implement `contracts.py` for the no-HRRR schema.
- Add a small audit function or script to check required columns, duplicate keys, missing core fields, and accidental HRRR columns.

Required checks:

- Non-empty table.
- Unique `target_date_local` plus `station_id`.
- Required columns present.
- No columns beginning with `hrrr_`.
- No columns beginning with `meta_hrrr_`.
- `target_residual_f` equals `final_tmax_f - anchor_tmax_f` within tolerance.
- `selection_cutoff_local` is `00:05 America/New_York` for the target date.

### 5. Normalize Features For Modeling

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/normalize_features.py`. Verified on the one-day training artifact; output written to `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet` with manifest. Manifest includes categorical vocabularies for non-label categorical fields.

Note 2026-04-25: rebuilt full normalized table from the full training artifact. Output has `1,096` rows and `274` columns. Manifest records `41` non-label categorical vocabularies and no `label_*` vocabulary leakage.

Goal:

- Implement `normalize_features.py`.
- Convert the merged no-HRRR table into model-ready units and encodings.

Output:

- `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet`
- `experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.manifest.json`

Rules:

- Normalize NBM Kelvin temperatures to Fahrenheit.
- Normalize NBM m/s wind speeds to mph.
- Normalize LAMP knots to mph.
- Encode local categorical fields inside this experiment.
- Do not encode label-derived categoricals such as `label_market_bin` as model features.
- Do not depend on the canonical HRRR-aware normalized registry.

### 6. Train Residual Quantile Models

Status: done

Note 2026-04-25: entrypoint scaffold added at `experiments/no_hrrr_model/no_hrrr_model/train_quantile_models.py`, but LightGBM training is not implemented yet and experiment requirements have not been installed.

Note 2026-04-25: installed experiment dependencies into `.venv` with `uv pip install --python .venv/bin/python -r experiments/no_hrrr_model/requirements.txt`. Implemented and ran `experiments/no_hrrr_model/no_hrrr_model/train_quantile_models.py`. Outputs written under `experiments/no_hrrr_model/data/runtime/models/`: seven `residual_quantile_q*.txt` LightGBM models, `feature_manifest.json`, and `training_manifest.json`. Training used `1,094` eligible rows, with `875` train rows through `2025-05-26` and `219` validation rows from `2025-05-27`.

Note 2026-04-25: reviewed `feature_manifest.json` and removed exact time-code and source-code features from model selection. Retrained models with `180` selected features and no `time`, `source`, `label`, `target_date`, or `market` feature names. Updated single-holdout q50 MAE/RMSE is `1.2579/1.6153`.

Goal:

- Implement `train_quantile_models.py`.
- Train residual quantile models around the 50/50 NBM/LAMP anchor.

Primary target:

```text
target_residual_f
```

Primary quantiles:

- `0.05`
- `0.10`
- `0.25`
- `0.50`
- `0.75`
- `0.90`
- `0.95`

Preferred learner:

- LightGBM quantile regression.

Dependency note:

- `lightgbm` is not currently in root `requirements.txt`.
- If adding it, prefer experiment-local dependency documentation in `experiments/no_hrrr_model/requirements.txt` unless the user explicitly asks to promote it.

Expected outputs:

```text
experiments/no_hrrr_model/data/runtime/models/
  residual_quantile_q05.*
  residual_quantile_q10.*
  residual_quantile_q25.*
  residual_quantile_q50.*
  residual_quantile_q75.*
  residual_quantile_q90.*
  residual_quantile_q95.*
  feature_manifest.json
  training_manifest.json
```

If dependency installation is needed, ask the user to run it, for example:

```bash
uv pip install --python .venv/bin/python -r experiments/no_hrrr_model/requirements.txt
```

Then wait for confirmation.

### 7. Train And Compare Baselines

Status: done

Note 2026-04-25: implemented baseline comparison in `experiments/no_hrrr_model/no_hrrr_model/evaluate.py`. Validation MAE/RMSE: NBM-only `1.5735/2.0401`, LAMP-only `1.3744/1.8245`, fixed 50/50 anchor `1.3734/1.7833`, linear NBM/LAMP blend `1.4330/1.8317`, direct absolute LightGBM `1.4974/2.0774`, residual quantile q50 `1.2708/1.6291`.

Goal:

- Implement baselines in `train_quantile_models.py` or `evaluate.py`.

Required baselines:

- NBM-only anchor.
- LAMP-only anchor.
- Fixed 50/50 NBM/LAMP anchor.
- Linear NBM/LAMP blend.
- Direct absolute model predicting `final_tmax_f`, if dependency support is available.

### 8. Evaluate Model And Calibration

Status: done

Note 2026-04-25: entrypoint scaffold added at `experiments/no_hrrr_model/no_hrrr_model/evaluate.py`; implementation waits on trained quantile model artifacts.

Note 2026-04-25: implemented and ran evaluation. Outputs written under `experiments/no_hrrr_model/data/runtime/evaluation/`: `metrics_overall.json`, `baseline_comparison.csv`, `pinball_loss.csv`, `quantile_coverage.csv`, `quantile_crossing.csv`, `metrics_by_season.csv`, `metrics_by_nbm_lamp_disagreement.csv`, `validation_predictions.parquet`, and `evaluation_manifest.json`. Validation slice is `2025-05-27` through `2025-12-31` with `219` rows. After feature cleanup, residual q50 final-Tmax MAE/RMSE is `1.2579/1.6153`. q05-q95 coverage is `0.8904` with mean width `4.8870°F`.

Note 2026-04-25: added and ran `experiments/no_hrrr_model/no_hrrr_model/rolling_origin_evaluate.py`. Outputs: `rolling_origin_metrics.csv` and `rolling_origin_manifest.json`. 2024 holdout residual q50 MAE/RMSE is `1.4812/2.0691` versus anchor `1.5904/2.1703`; q05-q95 coverage `0.7225`. 2025 holdout residual q50 MAE/RMSE is `1.3720/1.9435` versus anchor `1.5987/2.1825`; q05-q95 coverage `0.7452`.

Goal:

- Implement `evaluate.py`.
- Evaluate the primary model against baselines and report decision-relevant diagnostics.

Required outputs:

```text
experiments/no_hrrr_model/data/runtime/evaluation/
  metrics_overall.json
  baseline_comparison.csv
  pinball_loss.csv
  quantile_coverage.csv
  metrics_by_season.csv
  metrics_by_nbm_lamp_disagreement.csv
  evaluation_manifest.json
```

Minimum metrics:

- MAE and RMSE for `final_tmax_f`.
- Residual MAE and RMSE around `anchor_tmax_f`.
- Pinball loss by residual quantile.
- Quantile coverage for final Tmax.
- Warm-season and cool-season slices.
- Missing LAMP slice.
- High NBM/LAMP disagreement slice.

### 9. Build Temperature Distribution Utilities

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/distribution.py` for adding residual quantiles to the anchor and converting final-Tmax quantiles into a 1°F probability ladder.

Goal:

- Implement `distribution.py`.
- Convert residual quantiles into final-Tmax quantiles and a controlled internal 1°F ladder.

Rules:

- Predict final temperature first.
- Map bins second.
- Keep the ladder independent of any one Polymarket event.
- Preserve expected final high and per-degree probabilities.

### 10. Build Event Bin Adapter

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/event_bins.py` with simple Polymarket-style bin parsing, ladder mapping, and `max_so_far_f` hard-zero behavior. Verified by pytest.

Goal:

- Implement `event_bins.py`.
- Parse event-specific bin definitions and map the internal 1°F distribution into those exact bins.

Rules:

- Do not train on event bins.
- Hard-zero bins below `max_so_far_f` when doing intraday/event-time adaptation.
- Keep trading and market-price comparison outside the core weather model.

### 11. Add Tests

Status: done

Note 2026-04-25: added focused tests in `experiments/no_hrrr_model/tests/test_no_hrrr_model.py`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests` (`11 passed` after adding Polymarket event parsing tests).

Goal:

- Add focused tests under `experiments/no_hrrr_model/tests/`.

Minimum tests:

- Training builder emits no HRRR columns.
- Anchor and residual formulas are correct.
- Duplicate target-date/station rows are rejected or audited.
- Missing LAMP or NBM rows are represented with availability flags.
- Normalizer produces expected units.
- Event-bin adapter maps a 1°F ladder into bins correctly.

Preferred command:

```bash
.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests
```

### 12. Add README Runbook

Status: done

Note 2026-04-25: added `experiments/no_hrrr_model/README.md` with experiment boundary, source paths, output paths, and commands for WU, LAMP, training feature build, normalization, prediction, Polymarket event-bin fetch, online inference, and tests.

Goal:

- Add `experiments/no_hrrr_model/README.md`.
- Document the exact local workflow from source inputs through evaluation.

README must include:

- Experiment boundary.
- Source input paths.
- Output artifact paths.
- Commands to build WU tables.
- Commands to build or validate LAMP overnight summaries.
- Commands to build training features.
- Commands to normalize features.
- Commands to train models.
- Commands to evaluate.
- Reminder that long-running downloads/backfills should be run by the user.

## Completed Prediction Adapter

Status: done

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/predict.py`. It loads `feature_manifest.json`, the seven residual quantile models, selects a normalized feature row by `target_date_local` and `station_id`, emits final-Tmax quantiles, converts them to a 1°F ladder, and optionally maps repeated `--event-bin` labels. Verified with `2025-08-15`; output written to `experiments/no_hrrr_model/data/runtime/predictions/prediction_KLGA_2025-08-15.json`.

## Completed Model Review

Status: done

Review the full no-HRRR experiment end to end before adding live adapters:

- Confirm feature selection in `feature_manifest.json` excludes leakage and unwanted date/source identifiers.
- Inspect validation predictions for quantile crossing before and after monotone rearrangement.
- Decide whether the residual q50 improvement over the 50/50 anchor is stable enough for paper-trading simulations.
- Add rolling-origin or yearly cross-validation if a single chronological validation split is not sufficient.
- Only after this review, build a live/current-day feature path and Polymarket event fetch/parse layer.

Note 2026-04-25: completed review. Feature selection now excludes date/time/source/label/market leakage-name fields. Raw quantile crossing diagnostics are written to `quantile_crossing.csv`; evaluation and prediction apply monotone rearrangement before coverage/bin mapping. Rolling-origin validation shows stable q50 point-error improvement over the fixed anchor, but q05-q95 intervals are under-covered and require calibration.

## Completed Calibration

Status: done

Implement calibration for decision-ready probabilities:

- Add calibration utilities under `experiments/no_hrrr_model/no_hrrr_model/`.
- Use validation and rolling-origin prediction residuals to widen or recalibrate quantile intervals before converting to the 1°F ladder.
- Write calibrated outputs under `experiments/no_hrrr_model/data/runtime/evaluation/` and update `predict.py` to optionally use calibrated quantiles.
- Re-evaluate q05-q95, q10-q90, and q25-q75 coverage after calibration.
- Do not start paper trading until calibrated interval/bin probabilities are reviewed.

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/calibrate_quantiles.py`, which fits empirical validation-slice final-quantile offsets and writes `calibration_manifest.json`, `calibrated_quantile_coverage.csv`, and `validation_predictions_calibrated.parquet`. Updated `predict.py` to apply `calibration_manifest.json` by default when present. Validation q05-q95 coverage changed from `0.8904` to `0.8995`; q10-q90 from `0.7671` to `0.7991`; q25-q75 from `0.4155` to `0.4977`.

Note 2026-04-25: added rolling-origin per-row predictions and `experiments/no_hrrr_model/no_hrrr_model/calibrate_rolling_origin.py`. Calibration offsets fit on the 2024 rolling holdout and tested on the 2025 rolling holdout improved q05-q95 coverage from `0.7452` to `0.8932`, q10-q90 from `0.6082` to `0.8466`, and q25-q75 from `0.3589` to `0.5726`. q05-q95 interval score improved from `9.3269` to `8.2491`; q50 MAE changed from `1.3720` to `1.3746`. Updated `predict.py` to prefer `rolling_origin_calibration_manifest.json` by default when present.

## Current Next Step

Status: done

Build the live/current-day no-HRRR feature path and event-bin ingestion:

- Add an inference feature-builder that creates one normalized no-HRRR row for a requested `target_date_local` using the same contracts as training.
- Reuse experiment-local categorical vocabularies from `training_features_overnight_no_hrrr_normalized.manifest.json`; unknown categories should encode as `-1`.
- Require NBM and LAMP anchor availability before model prediction; otherwise emit a clear unavailable status.
- Add a Polymarket event-bin parser/fetch layer that only maps model probabilities into event bins; do not add orderbook or trading execution yet.
- Keep live/prediction artifacts under `experiments/no_hrrr_model/data/runtime/predictions/`.

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/build_inference_features.py`. It builds one unnormalized and normalized inference feature row for a requested `target_date_local`, reuses training categorical vocabularies, checks NBM/LAMP anchor availability, and writes under `experiments/no_hrrr_model/data/runtime/predictions/features/target_date_local=YYYY-MM-DD/`.

Note 2026-04-25: extended `predict.py` and `event_bins.py` to ingest event bins from a text or JSON file via `--event-bins-path`, in addition to repeated `--event-bin` labels. Verified with `2025-08-15` local inference features and a JSON outcomes file.

## Completed Polymarket Event Metadata Adapter

Status: done

Add Polymarket event metadata fetch/parse:

- Fetch or load a Polymarket event and extract the written outcome/bin labels without using AI-generated commentary.
- Keep market-price/orderbook comparison separate from weather probability generation.

Note 2026-04-25: added `experiments/no_hrrr_model/no_hrrr_model/run_online_inference.py`, a date-scoped orchestrator that refreshes experiment-local Wunderground, LAMP, and NBM inputs from online sources, builds the normalized inference row, and runs prediction. It supports `--lamp-source auto|live|archive|iem`, repeated `--event-bin`, and `--event-bins-path`. The default `auto` mode probes live NOMADS first, then the public NOAA LAMP archive, then Iowa State IEM AFOS station-specific `LAVLGA` products. NBM remains download-heavy, so this should be used for one requested target date, not broad loops.

Note 2026-04-25: date-scoped online Wunderground history/table outputs and added per-target status manifests under `experiments/no_hrrr_model/data/runtime/online_inference/status/target_date_local=YYYY-MM-DD/online_inference.manifest.json`. Failed runs now preserve already-built artifacts and record the failure reason plus paths. Verified `2026-04-14` failure leaves Wunderground JSON/parquet artifacts under target-date folders while recording LAMP unavailability.

Note 2026-04-25: added `experiments/no_hrrr_model/no_hrrr_model/polymarket_event.py`, which uses the public Polymarket Gamma API by event slug, persists raw event metadata, extracts ordered temperature-bin markets, and writes `event_bins.json` with market ids, slugs, condition ids, token ids, outcome prices, outcomes, and question text. Integrated `run_online_inference.py --polymarket-event-slug SLUG` so the online run can fetch bins automatically. `--polymarket-event-slug` can also be passed without a value to derive the standard weather slug from `--target-date-local`, e.g. `2026-04-25` -> `highest-temperature-in-nyc-on-april-25-2026`. Live validation against `highest-temperature-in-nyc-on-april-11-2026` produced 11 bins from `59F or below` through `78F or higher` under `experiments/no_hrrr_model/data/runtime/polymarket/event_slug=highest-temperature-in-nyc-on-april-11-2026/`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests`.

## Completed Online Inference TUI

Status: done

Add a guided terminal UI for one-date online inference:

- Default to the current `America/New_York` target date.
- Let the user set target date, station, LAMP source, and optional max-so-far value.
- Run the existing online inference pipeline immediately from the TUI.
- Display expected final high, final-Tmax quantiles, and mapped Polymarket event-bin probabilities.
- Isolate TUI artifacts under `experiments/no_hrrr_model/data/runtime/tui_online_inference/`.
- Delete artifacts for failed TUI-launched runs after reading the failure manifest; preserve normal CLI failure artifacts unchanged.

Note 2026-04-25: implemented `experiments/no_hrrr_model/no_hrrr_model/tui.py` with a Textual-based guided runner. Added `textual` to the experiment requirements, added TUI helper tests for New York default-date behavior, prediction summary formatting, failure-manifest messages, safe cleanup, and run-scoped command construction, and documented the TUI command in `README.md`.

## Current Next Step

Status: not done

Build paper-trading simulation:

- Join mapped weather probabilities to provided or fetched market prices.
- Use the preserved Polymarket market ids, condition ids, and CLOB token ids from `event_bins.json` as the join surface.
- Include fees, spread, slippage, liquidity, and position sizing assumptions as explicit inputs.
- Emit a report with model probability, market-implied probability, edge after costs, and action/no-action status per bin.
- Do not add live order placement.
