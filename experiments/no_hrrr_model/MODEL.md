# No-HRRR Experiment MODEL.md

## Purpose

This folder is a separate experiment for building a KLGA daily-high model without HRRR data.

The motivation is practical: HRRR collection is currently difficult, so this experiment tests whether a usable overnight fair-value model can be built from the sources that are easier to obtain and maintain.

This is not the main overnight model spec. The canonical project model remains the root `MODEL.md`.

## Experiment Boundary

Keep this experiment isolated from the main workflow unless the user explicitly asks to merge ideas back.

- do not require HRRR inputs
- do not modify the root overnight model contract to remove HRRR
- do not treat results here as canonical production behavior
- do keep the same settlement target: final KLGA daily high in whole degrees Fahrenheit
- do keep the same market-day timezone: `America/New_York`
- do keep the same `00:05 America/New_York` overnight as-of rule

The experiment may reuse shared label, Wunderground, LAMP, NBM, and normalization utilities when doing so does not change their canonical contracts.

## Modeling Goal

At `target_date_local 00:05 America/New_York`, estimate a temperature distribution for the final KLGA daily maximum temperature for that local day, then map that distribution into event-specific Polymarket bins.

The core target remains Wunderground-settled temperature first, bins second:

1. use finalized Wunderground KLGA daily high as the label
2. build an NBM/LAMP forecast anchor available by the cutoff
3. train a residual distribution model around that anchor
4. add residual quantiles back to the anchor to get final Tmax quantiles
5. convert the final Tmax distribution into a controlled internal 1°F ladder
6. map the distribution into the current event's actual bin ladder
7. compare fair values against market prices only after settlement-aligned probabilities exist

## Label And Feature Boundary

Wunderground is the settlement reference. Treat finalized Wunderground KLGA daily high as the canonical label:

```text
label_final_tmax_f = finalized_wu_klga_tmax_f
```

Do not train against NBM Tmax, LAMP Tmax, a station other than `KLGA`, or a generic NYC temperature target.

Forecast features come from:

- NBM overnight prior
- LAMP overnight station summary
- optional Wunderground observations available before `00:05 America/New_York` for current station state and recent trends

This means Wunderground has two separate roles:

- finalized daily history after the event: label source
- observations before the cutoff: optional live-state features

HRRR fields, HRRR revision features, and HRRR availability flags should be excluded from this experiment's training table and model inputs.

## Primary Model Architecture

Use a residual model around an NBM/LAMP blended anchor.

Start with a fixed 50/50 anchor:

```text
anchor_tmax_f = 0.5 * nbm_tmax_open_f + 0.5 * lamp_tmax_open_f
target_residual_f = label_final_tmax_f - anchor_tmax_f
```

The model predicts the residual distribution, not the full temperature from scratch.

At inference:

```text
anchor_tmax_f = 0.5 * nbm_tmax_open_f + 0.5 * lamp_tmax_open_f
predicted_tmax_quantile_f = anchor_tmax_f + predicted_residual_quantile_f
```

This keeps the model centered on the two forecast systems while allowing it to learn systematic settlement error relative to Wunderground KLGA.

## Learner

Use LightGBM quantile regression as the first serious learner for this experiment.

Train separate residual quantile models, for example:

- `q05`
- `q10`
- `q25`
- `q50`
- `q75`
- `q90`
- `q95`

Each model predicts:

```text
target_residual_f = finalized_wu_klga_tmax_f - anchor_tmax_f
```

The median quantile is the point residual. The full quantile set is converted into a temperature distribution before Polymarket bin mapping.

CatBoost, absolute LightGBM, linear blends, and single-source residual baselines can be benchmarked later, but they are not the primary architecture for this experiment.

## Core Features

Minimum anchor features:

- `nbm_tmax_open_f`
- `lamp_tmax_open_f`
- `anchor_tmax_f`
- `nbm_minus_lamp_tmax_f`

`nbm_minus_lamp_tmax_f` is required:

```text
nbm_minus_lamp_tmax_f = nbm_tmax_open_f - lamp_tmax_open_f
```

It captures both disagreement direction and likely uncertainty. When NBM and LAMP agree, the model should generally be more confident. When they disagree, the residual distribution should be allowed to widen.

NBM feature families:

- target-day Tmax guidance
- local `06/09/12/15/18/21` temperature curve
- dewpoint and humidity checkpoints
- wind and gust features
- cloud-cover summaries
- precipitation features
- solar/radiation features when available

LAMP feature families:

- target-day station Tmax guidance
- local `06/09/12/15/18/21` temperature curve
- dewpoint checkpoints
- wind direction and speed checkpoints
- cloud, ceiling, visibility, obstruction, and precipitation-probability summaries
- revision features when available by cutoff

Optional Wunderground pre-cutoff feature families:

- latest KLGA observation before `00:05`
- recent temperature, dewpoint, pressure, wind, and visibility trends
- previous local-day finalized high and low

Season and solar context:

- day of year encoded cyclically
- month or season
- approximate daylight duration or solar elevation proxy if implemented locally

## As-Of Rule

Every feature used for `target_date_local` must be available by `target_date_local 00:05 America/New_York`.

Training and inference must use the same cutoff logic.

- Wunderground: observations at or before the cutoff
- NBM: the latest cutoff-eligible issue for the target local day
- LAMP: the latest full temperature-guidance issue at or before the cutoff

Do not fill missing HRRR-derived fields with placeholders just to match the main table schema. A no-HRRR table should have its own explicit schema.

## Training Table

Use one row per target local day.

Required identity fields:

- `target_date_local`
- `station_id`
- `selection_cutoff_local`

Required labels:

- `final_tmax_f`
- `target_residual_f`
- `anchor_tmax_f`
- optional `market_bin` only as an adapter/output field, not as the core model target

Recommended experiment artifacts:

- `training_features_overnight_no_hrrr`
- `training_features_overnight_no_hrrr_normalized`
- `model_no_hrrr_residual_quantile_q05`
- `model_no_hrrr_residual_quantile_q10`
- `model_no_hrrr_residual_quantile_q25`
- `model_no_hrrr_residual_quantile_q50`
- `model_no_hrrr_residual_quantile_q75`
- `model_no_hrrr_residual_quantile_q90`
- `model_no_hrrr_residual_quantile_q95`
- `calibration_no_hrrr_overnight`

These names are intentionally different from the canonical root model artifacts.

## Baselines To Backtest

Backtest the primary residual quantile model against simple baselines:

- NBM-only anchor
- LAMP-only anchor
- fixed 50/50 NBM/LAMP anchor
- linear-regression NBM/LAMP blend
- direct absolute LightGBM model predicting `final_tmax_f`

The primary model should remain:

```text
WU label -> NBM/LAMP blended anchor -> residual quantile LightGBM -> calibrated 1°F distribution -> Polymarket bin adapter
```

## Evaluation

Compare the no-HRRR experiment against the canonical model only when both are evaluated over the same target dates and cutoff rules.

Minimum evaluation slices:

- overall MAE/RMSE for `final_tmax_f`
- residual MAE/RMSE around the anchor
- pinball loss by residual quantile
- quantile coverage for the final Tmax distribution
- calibration by 1°F or market-bin probability buckets
- warm-season and cool-season splits
- days where LAMP is unavailable
- days where NBM and LAMP strongly disagree
- days with large observed forecast busts

The experiment is successful only if it produces decision-ready calibrated probabilities. A lower-maintenance pipeline is useful, but not if settlement-bin probabilities are unreliable.

## Non-Goals

Do not add these inside this experiment unless explicitly requested:

- HRRR ingestion or HRRR repair work
- nearby-station expansion
- satellite or cloud imagery
- market orderbook features
- changes to the canonical root model contract
