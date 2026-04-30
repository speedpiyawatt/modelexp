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

- Model candidate: `nearby_vreg_leaf100_lgbm_350`
- Anchor policy: `equal_3way`
- Feature/weight profile: `high_disagreement_weighted_nearby` / `high_disagreement_weighted`
- Nearby stations: `KJRB`, `KJFK`, `KEWR`, `KTEB`
- Quantile calibration: guarded `global_offsets_no_upper_tail`
- Distribution method: `normal_iqr`
- Ladder reliability calibration: `bucket_reliability_s1_00`
- Dual-model inference guard: `with_hrrr_except_native_cold_hrrr_warm`
- Feature count: `467`
- Training rows: `1,094` eligible rows from `2023-01-01` through `2025-12-31`

Source-trust upgrade status:

- Implemented and validated 2026-04-29.
- Rolling model selection can now evaluate dynamic anchors (`equal_3way`, 4-way native/NBM/LAMP/HRRR, median 4-way, trimmed-mean 4-way, fold-local ridge 4-way), source-trust feature profiles, weighted disagreement specialists, and an optional meta residual correction.
- Full rolling-origin validation selected the high-disagreement-weighted profile on the existing equal-3way anchor. Dynamic 4-way, median, trimmed, ridge, and meta residual variants were evaluated but not promoted.
- Prior deployment 2026-04-29: code was pushed in commit `20330a1` and pulled on `/root/modelexp` at `root@198.199.64.163` with ignored runtime artifacts synced separately.

Nearby-station upgrade status:

- Implemented and validated 2026-04-30.
- Added cutoff-safe Wunderground observation features for `KJRB`, `KJFK`, `KEWR`, and `KTEB`. Nearby observations are only used if they are available by `00:05 America/New_York` and within the configured stale-observation guard.
- Online inference fetches/builds those nearby Wunderground station rows by default and removes their downloaded/intermediate artifacts after a successful prediction unless `--keep-artifacts` is set.
- Review fixes pushed 2026-04-30: server dual inference now uses the with-HRRR online wrapper so nearby station features are fetched, non-nearby feature profiles exclude all nearby-derived columns, nearby-selected inference fails instead of silently predicting with all nearby features missing, guarded calibration selection is the current production default, and holdout evaluation scores the production calibrated stack.
- Server deployment status 2026-04-30: `/root/modelexp` has been synced with the latest pushed with-HRRR inference/evaluation fixes plus ignored with-HRRR runtime model/evaluation artifacts. Any later model artifact rebuild or code commit still needs an explicit server sync because Git does not track runtime artifacts.
- Full rolling-origin validation evaluated `47` candidate specs including nearby feature profiles and focused nearby LightGBM candidates. It selected `nearby_vreg_leaf100_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted_nearby__weights=high_disagreement_weighted`.
- Dynamic 4-way, median, trimmed, ridge, meta residual, non-nearby source-trust, and disagreement-widening variants remain evaluated fallbacks but are not the current production default.

## Current Metrics

Production-calibrated single holdout, `2025-05-27..2025-12-31`, `219` rows:

| Metric | Value |
| --- | ---: |
| raw q50 MAE/RMSE | `1.2612 / 1.6395` |
| scored q50 MAE/RMSE | `1.2657 / 1.6428` |
| degree NLL/Brier/RPS | `1.861432 / 0.811339 / 0.009238` |
| event-bin NLL/Brier | `1.133178 / 0.603449` |

Rolling 2025 calibration test, `365` rows:

| Stage | Event NLL | Event Brier | Degree NLL | Degree RPS |
| --- | ---: | ---: | ---: | ---: |
| uncalibrated quantiles + interpolation | `1.370969` | `0.622307` | `2.233741` | `0.010356` |
| conformal quantiles + interpolation | `1.274655` | `0.615411` | `2.071875` | `0.010205` |
| conformal quantiles + `normal_iqr` | `1.261539` | `0.616776` | `2.024162` | `0.009516` |
| selected source-trust quantiles + `normal_iqr` + ladder reliability | `1.250237` | `0.607134` | `2.008158` | `0.009521` |
| selected nearby source-trust quantiles + guarded `global_offsets_no_upper_tail` + `normal_iqr` + ladder reliability | `1.241810` | `0.606852` | `2.000715` | `0.009475` |

Source-disagreement robustness, rolling 2025 calibration test:

- Quantile calibration now evaluates `global_offsets`, `conformal_intervals`, `no_offsets`, `global_offsets_no_upper_tail`, `global_offsets_shrunk_50pct`, and conditional source-disagreement variants. Promotion is guarded against observed-bin-probability drops in `tight_consensus` and `moderate_disagreement` slices. `global_offsets` and `conformal_intervals` were rejected by this guard; `global_offsets_no_upper_tail` is selected.
- `global_offsets_no_upper_tail` applies the learned lower/median quantile offsets while zeroing the q75/q90/q95 offsets. This keeps the calibration benefit that improved validation NLL without pushing too much probability out of the correct upper event bins on tight/moderate source-consensus days.
- Prediction JSON includes `calibration_method_id`, legacy `calibration_offsets_f`, and structured `calibration` metadata. If a conditional source-disagreement calibration is selected in a future run, the structured metadata records the row regime, branch, branch method, and branch offsets.
- `source_disagreement_regime_offsets` remains evaluated as a diagnostic method but is no longer eligible for promotion unless the promotion family is deliberately expanded.
- Source-disagreement ladder widening was evaluated at `0.5F`, `1.0F`, and `1.5F`; none beat `bucket_reliability_s1_00`, so no widening is enabled by default.
- Diagnostics are written to `metrics_by_source_disagreement_regime.csv` and `ladder_calibration_disagreement_slices.csv`.

Dual-model guard, validation plus 2026 live backtest:

- `tools/weather/evaluate_dual_guard.py` evaluates no-HRRR, with-HRRR, regime switching, and blend candidates on the production holdout plus retained 2026 server-dual prediction JSONs.
- Selected guard: `with_hrrr_except_native_cold_hrrr_warm`. It uses with-HRRR normally, but falls back to no-HRRR probabilities and expected Tmax when the with-HRRR source regime is `native_cold_hrrr_warm`.
- Validation holdout, `219` rows: always with-HRRR event NLL/observed-bin probability/MAE `1.133178/0.385604/1.265688`; selected guard `1.137793/0.385989/1.250839`.
- 2026 live backtest, `86` successful rows: always with-HRRR event NLL/observed-bin probability/MAE/top-bin accuracy `1.589559/0.408022/2.085680/0.500`; selected guard `1.554194/0.411389/2.024112/0.500`.
- Guard artifacts live under `experiments/withhrrr/data/runtime/evaluation/dual_guard/`. Server dual inference reads `dual_guard_manifest.json` when present and prints a `guarded` recommendation alongside raw no-HRRR and with-HRRR outputs.

HRRR ablation, rolling `729` validation rows:

| Variant | Event NLL | Degree NLL | q50 MAE/RMSE |
| --- | ---: | ---: | ---: |
| with HRRR features | `1.515909` | `2.367784` | `1.4042 / 1.9819` |
| HRRR columns dropped | `1.594013` | `2.469751` | `1.4261 / 2.0042` |

HRRR improves the rolling probability metrics materially. Keep HRRR in the model.

## Runbook

Production-style one-date inference should use the server dual runner from the local repo:

```bash
.venv/bin/python tools/weather/run_server_dual_inference.py YYYY-MM-DD
```

The output includes raw `no_hrrr`, raw `with_hrrr`, and a `guarded` recommendation when `experiments/withhrrr/data/runtime/evaluation/dual_guard/dual_guard_manifest.json` exists. Rebuild the guard manifest after refreshing validation or 2026 backtest artifacts:

```bash
.venv/bin/python tools/weather/evaluate_dual_guard.py
```

The server runner uses `/root/modelexp` on `root@198.199.64.163`. For the with-HRRR side to use the nearby Source-Trust model, the server needs the latest pushed with-HRRR inference/evaluation fixes and the ignored runtime artifacts under `experiments/withhrrr/data/runtime/`; a Git pull alone does not refresh those model/evaluation artifacts. As of 2026-04-30, the server has been pulled/synced with both code and runtime artifacts.

The server runner overlaps no-HRRR source work, HRRR source work, and nearby Wunderground station prefetch. Its default server tuning uses per-lead NBM parallelism for one-date latency (`batch-reduce-mode=off`, `lead-workers=8`, `download-workers=6`, `reduce-workers=4`, `extract-workers=4`, `wgrib2-threads=1`) while keeping HRRR in optimized batch mode. On the server, a `2026-04-28` NBM benchmark took `27.91s` in per-lead mode versus `53.16s` in batch-cycle mode with material output parity. Override with environment variables such as `MODELEXP_NBM_BATCH_REDUCE_MODE`, `MODELEXP_NBM_LEAD_WORKERS`, `MODELEXP_NBM_DOWNLOAD_WORKERS`, `MODELEXP_HRRR_DOWNLOAD_WORKERS`, and `MODELEXP_HRRR_MAX_WORKERS` when benchmarking.

Prepare the model table:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
```

Train the selected quantile model:

```bash
rm -rf experiments/withhrrr/data/runtime/models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
```

Evaluate the chronological holdout with the production probability stack:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/full_holdout_local
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate \
  --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local
```

By default this loads the selected quantile calibration, distribution manifest, and ladder calibration manifest. It writes calibrated/scored rows to `validation_predictions.parquet` and raw model rows to `validation_predictions_raw.parquet`. Use `--no-calibration --distribution-method interpolation_tail --no-ladder-calibration` only when intentionally debugging raw model quantiles.

Run rolling-origin model selection:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/model_selection
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
```

After model selection, retrain production artifacts from the selected manifest:

```bash
rm -rf experiments/withhrrr/data/runtime/models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
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
