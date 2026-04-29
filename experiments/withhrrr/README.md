# With-HRRR KLGA Overnight Experiment

This experiment is the HRRR-inclusive version of the KLGA overnight fair-value model. It keeps the same model family as `experiments/no_hrrr_model`: residual LightGBM quantile models, calibrated final-Tmax quantiles, a 1F temperature ladder, and an event-bin adapter.

The model predicts final KLGA temperature first. Polymarket bins are parsed and mapped at runtime.

## Contract

- Target: finalized Wunderground KLGA daily high.
- Target day: `America/New_York` local date.
- Cutoff: `00:05 America/New_York`.
- Row grain: one row per `target_date_local` and `station_id`.
- Anchor: equal 3-way blend, `(nbm_tmax_open_f + lamp_tmax_open_f + hrrr_tmax_open_f) / 3`.
- Residual target: `label_final_tmax_f - anchor_tmax_f`.
- HRRR role: direct model features, calibration/evaluation regimes, and the selected equal 3-way anchor.

HRRR-only is not good enough to be the whole model. On the 2025 holdout, HRRR-only remains materially worse than the final residual model, but rolling anchor-policy diagnostics selected the equal NBM/LAMP/HRRR anchor by calibrated event-bin NLL.

## Current Defaults

- Model candidate: `very_regularized_min_leaf70_lgbm_350`
- Anchor policy: `equal_3way`
- Quantile calibration: `hrrr_nbm_direction_offsets`
- Distribution method: `normal_iqr`
- Ladder reliability calibration: `bucket_reliability_s1_00`
- Feature count: `304`
- Training rows: `1,094` eligible rows from `2023-01-01` through `2025-12-31`

## Current Metrics

Single holdout, `2025-05-27..2025-12-31`, `219` rows:

| Metric | Value |
| --- | ---: |
| q50 MAE/RMSE | `1.2589 / 1.6305` |
| degree NLL/Brier/RPS | `2.182397 / 0.822290 / 0.010258` |
| event-bin NLL/Brier | `1.411969 / 0.620390` |

Rolling 2025 calibration test, `365` rows:

| Stage | Event NLL | Event Brier | Degree NLL | Degree RPS |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated quantiles + interpolation | `1.370969` | `0.622307` | `2.233741` | `0.010356` |
| conformal quantiles + interpolation | `1.274655` | `0.615411` | `2.071875` | `0.010205` |
| conformal quantiles + `normal_iqr` | `1.261539` | `0.616776` | `2.024162` | `0.009516` |
| selected HRRR-direction quantiles + `normal_iqr` + ladder reliability | `1.240912` | `0.603902` | `1.999327` | `0.009510` |

HRRR ablation, rolling `729` validation rows:

| Variant | Event NLL | Degree NLL | q50 MAE/RMSE |
| --- | ---: | ---: | ---: |
| with HRRR features | `1.515909` | `2.367784` | `1.4042 / 1.9819` |
| HRRR columns dropped | `1.594013` | `2.469751` | `1.4261 / 2.0042` |

HRRR improves the rolling probability metrics materially. Keep HRRR in the model.

## Runbook

Prepare the model table:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
```

Train the selected quantile model:

```bash
rm -rf experiments/withhrrr/data/runtime/models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
```

Evaluate the chronological holdout:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/full_holdout_local
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate \
  --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local
```

Run rolling-origin model selection:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/model_selection
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
```

Run HRRR ablation:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/hrrr_ablation
.venv/bin/python -m experiments.withhrrr.withhrrr_model.hrrr_ablation_diagnostics
```

Select quantile calibration:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/calibration_selection
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin
```

Select distribution method:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics
.venv/bin/python -m experiments.withhrrr.withhrrr_model.distribution_diagnostics
```

Select ladder reliability calibration:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/ladder_calibration
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_ladder
```

Run a prediction with local event bins:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bin "30F or below" \
  --event-bin "31-35F" \
  --event-bin "36F or higher"
```

Fetch a Polymarket event and map its bins:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.polymarket_event \
  --event-slug highest-temperature-in-nyc-on-april-11-2026

.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bins-path experiments/withhrrr/data/runtime/polymarket/event_slug=highest-temperature-in-nyc-on-april-11-2026/event_bins.json
```

Run tests:

```bash
.venv/bin/python -m pytest -q experiments/withhrrr/tests
```

## Key Artifacts

- Training table: `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet`
- Models: `experiments/withhrrr/data/runtime/models/`
- Model selection: `experiments/withhrrr/data/runtime/evaluation/model_selection/`
- Calibration selection: `experiments/withhrrr/data/runtime/evaluation/calibration_selection/`
- Distribution diagnostics: `experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics/`
- Ladder calibration: `experiments/withhrrr/data/runtime/evaluation/ladder_calibration/`
- HRRR ablation: `experiments/withhrrr/data/runtime/evaluation/hrrr_ablation/`
- Polymarket event metadata: `experiments/withhrrr/data/runtime/polymarket/`

## Working Docs

- `TODO.md`: optimization checklist with completed metrics.
- `IMPLEMENTATION_TODO.md`: implementation notes and command history.

Future agents must update both files when changing model behavior or selected artifacts.
