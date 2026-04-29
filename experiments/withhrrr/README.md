# With-HRRR KLGA Overnight Experiment

This experiment promotes the no-HRRR model pattern into the HRRR-inclusive overnight model.

The goal is not a staged ensemble. The goal is the same residual quantile LightGBM workflow used in `experiments/no_hrrr_model`, but trained on canonical Wunderground, NBM, LAMP, and HRRR overnight features.

## Contract

- Target: finalized Wunderground KLGA daily high.
- Target day: `America/New_York` local date.
- Cutoff: `00:05 America/New_York`.
- Row grain: one row per `target_date_local` and `station_id`.
- Prediction target: final temperature distribution first, Polymarket bins second.
- Initial residual target: `label_final_tmax_f - anchor_tmax_f`.
- Initial anchor: same NBM/LAMP anchor pattern as no-HRRR.
- HRRR role: direct model input features, not a separate main model.

## Working Docs

- `IMPLEMENTATION_TODO.md`: build plan and implementation checklist.
- `TODO.md`: probability-first optimization roadmap.

Future agents must keep both files updated as work progresses.

## Current Local Commands

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict --target-date-local 2025-12-31
```

Current selected candidate after rolling-origin model selection:

```text
very_regularized_min_leaf70_lgbm_350
```

Current selected calibration method:

```text
season_offsets
```
