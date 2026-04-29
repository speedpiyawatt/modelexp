# With-HRRR Model Optimization TODO

This TODO tracks model-quality work for the HRRR-inclusive KLGA overnight experiment.

Future agents: update this document after finishing each task. Change `[ ]` to `[x]`, add the completion date, command(s), artifact paths, and a short result note. If a task is blocked, leave it unchecked and add a `Blocked:` note with the exact reason.

For every completed optimization task, also add an `Eval:` note. The note must include the validation slice, the command used, artifact path, and before/after or baseline/current metrics needed to judge whether quality improved. Prefer probability metrics first, then q50 MAE/RMSE:

- degree-ladder NLL, Brier score, ranked probability score, and observed-temp probability
- representative event-bin NLL, Brier score, and observed-bin probability
- PIT summary and interval coverage when calibration changed
- q50 MAE/RMSE as secondary diagnostics

## Scope Guardrails

- Keep this work inside `experiments/withhrrr/`.
- This is not a staged ensemble project unless the user explicitly asks for that later.
- Build the same model style as `experiments/no_hrrr_model`: residual LightGBM quantile models, calibrated whole-degree distribution, and event-bin adapter.
- The difference from no-HRRR is that the training row includes HRRR features and metadata from the canonical overnight table.
- HRRR must be real input features. Do not only compare against HRRR, and do not train a separate HRRR-only side model as the main deliverable.
- Use `America/New_York` local target dates and the `00:05` overnight cutoff.
- Predict final KLGA temperature first; map into Polymarket bins second.
- Prefer rolling-origin and out-of-time validation over random splits.
- Do not add live order placement in this phase.

## Primary Model Contract

Train one HRRR-inclusive residual quantile model family:

```text
canonical WU + NBM + LAMP + HRRR normalized row
-> LightGBM residual quantile models
-> calibrated final-Tmax quantiles
-> 1F probability ladder
-> event-bin adapter
```

Initial target contract:

```text
nbm_tmax_open_f = fahrenheit(nbm_temp_2m_day_max_k)
lamp_tmax_open_f = lamp_day_temp_max_f or canonical LAMP equivalent
anchor_tmax_f = 0.5 * nbm_tmax_open_f + 0.5 * lamp_tmax_open_f
target_residual_f = label_final_tmax_f - anchor_tmax_f
```

Important: HRRR enters as model input features. Do not replace the model with an HRRR anchor or staged HRRR ensemble unless a later evaluation proves that is better and the user approves it.

## Phase 1: Canonical HRRR-Inclusive Training Table

- [x] Pull or stage the completed HRRR overnight backfill into the local canonical runtime path.
  - Source server path from the completed run: `/root/modelexp/data/runtime/hrrr_backfill_overnight/hrrr_summary_state/`.
  - Local experiment should not write into canonical folders except when explicitly building canonical source tables.
  - Suggested local staging path: `experiments/withhrrr/data/runtime/source/hrrr_summary_state/`.
  - Eval: record date coverage, row count, manifest failure count, `extract_method`, `summary_profile`, and `selection_mode`.
  - Done 2026-04-29: copied HRRR state manifests to `experiments/withhrrr/data/runtime/source/hrrr_summary_state/` and actual summary rows to `experiments/withhrrr/data/runtime/source/hrrr_summary/`.
  - Eval 2026-04-29: validation command in `IMPLEMENTATION_TODO.md` showed `1,096` state days, `0` failures, `extract_method={'wgrib2-bin': 1096}`, `summary_profile={'overnight': 1096}`, `selection_mode={'overnight_0005': 1096}`. Actual summary parquet count is `1,096`, covering `2023-01-01` through `2025-12-31`.

- [x] Build the canonical merged table with HRRR included.
  - Use `tools/weather/build_training_features_overnight.py`.
  - Inputs should include Wunderground labels/obs, NBM overnight rows, LAMP overnight rows, and HRRR overnight summaries.
  - Output copy for this experiment should live under `experiments/withhrrr/data/runtime/training/`.
  - Eval: audit row count, date range, missing-source counts, and HRRR availability.
  - Done 2026-04-29: pulled NBM from Lightning path `/teamspace/studios/this_studio/modelexp/data/runtime/backfill_overnight/nbm_overnight/` into `data/runtime/backfill_overnight/nbm_overnight/`, pulled LAMP from Lightning path `/teamspace/studios/this_studio/modelexp/experiments/no_hrrr_model/data/runtime/lamp_overnight/` into `experiments/no_hrrr_model/data/runtime/lamp_overnight/`, rebuilt Wunderground tables with `.venv/bin/python wunderground/build_training_tables.py`, then ran `tools/weather/build_training_features_overnight.py`.
  - Eval 2026-04-29: canonical merged output at `experiments/withhrrr/data/runtime/training/training_features_overnight.parquet` has `1,096` rows, `316` columns, date range `2023-01-01` through `2025-12-31`, `1,096` HRRR-available rows, and `105` HRRR columns.

- [x] Build the canonical normalized HRRR-inclusive table.
  - Use `tools/weather/build_training_features_overnight_normalized.py`.
  - Output:
    - `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_normalized.parquet`
    - `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_normalized.manifest.json`
  - Eval: record normalized row count, column count, HRRR feature count, and categorical vocab use.
  - Done 2026-04-29: ran `.venv/bin/python tools/weather/build_training_features_overnight_normalized.py --input-root experiments/withhrrr/data/runtime/training/training_features_overnight --output-dir experiments/withhrrr/data/runtime/training/training_features_overnight_normalized --manifest-output-path experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.manifest.json --start-local-date 2023-01-01 --end-local-date 2025-12-31`, then combined daily parts into `experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.parquet`.
  - Eval 2026-04-29: normalized output has `1,096` rows, `316` columns, date range `2023-01-01` through `2025-12-31`, `1,096` HRRR-available rows, and `105` HRRR columns.

- [x] Add experiment-local contract checks.
  - Confirm one row per `target_date_local` + `station_id`.
  - Confirm required labels and source availability columns exist.
  - Confirm HRRR columns are present and populated for expected dates.
  - Confirm `selection_cutoff_local` is exactly `00:05 America/New_York`.
  - Eval: write a contract audit JSON under `experiments/withhrrr/data/runtime/audit/`.
  - Done 2026-04-29: implemented checks inside `prepare_training_features.py`, including required identity/label/anchor fields, optional HRRR date join for non-canonical inputs, residual formula check, HRRR availability, and model eligibility update.
  - Eval 2026-04-29: prepared canonical table at `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet` has `1,096` rows, `1,096` unique dates, `1,096` HRRR-available rows, `1,094` model-eligible rows, and `105` HRRR columns before feature selection.

## Phase 2: Port The No-HRRR Model Stack

- [x] Create the `experiments.withhrrr.withhrrr_model` package.
  - Start by copying/adapting the no-HRRR modules only where needed.
  - Keep module names parallel to no-HRRR for easy diffing:
    - `model_config.py`
    - `train_quantile_models.py`
    - `evaluate.py`
    - `distribution.py`
    - `event_bins.py`
    - `predict.py`
    - `calibrate_rolling_origin.py`
    - `distribution_diagnostics.py`
    - `ensemble_diagnostics.py`
  - Eval: import smoke and pytest smoke should pass.
  - Done 2026-04-29: added package plus `prepare_training_features.py`, `model_config.py`, and `train_quantile_models.py`. Verified by running both modules.

- [x] Add derived residual columns for the canonical table.
  - Derive `nbm_tmax_open_f`, `lamp_tmax_open_f`, `anchor_tmax_f`, `nbm_minus_lamp_tmax_f`, and `target_residual_f`.
  - Keep the no-HRRR anchor formula initially so the only major change is adding HRRR features.
  - Eval: verify `target_residual_f = label_final_tmax_f - anchor_tmax_f` on all model-eligible rows.
  - Done 2026-04-29: `prepare_training_features.py` preserves the no-HRRR anchor and residual formula, aliases `final_tmax_f` to `label_final_tmax_f`, and verifies residual parity before writing output.
  - Eval 2026-04-29: preparation command completed without residual mismatch; output manifest written to `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.manifest.json`.

- [x] Port no-HRRR feature selection with HRRR allowed.
  - Keep leakage exclusions for labels, exact dates, market bins, raw timestamps, and source identifier codes.
  - Unlike no-HRRR, allow `hrrr_` and `meta_hrrr_` numeric/bool columns.
  - Avoid duplicate raw/unit-converted pairs when canonical normalized columns provide converted versions.
  - Eval: write selected feature manifest and report counts by prefix: `wu_`, `nbm_`, `lamp_`, `hrrr_`, `meta_`.
  - Done 2026-04-29: `train_quantile_models.py` selects numeric/bool features, excludes label/date/market/time-code/source-code leakage fields, and permits HRRR features.
  - Eval 2026-04-29: canonical rerun `feature_manifest.json` selected `291` features: `hrrr_=105`, `lamp_=82`, `nbm_=59`, `wu_=23`, `meta_=21`.

- [x] Train the first HRRR-inclusive residual quantile model.
  - Reuse the no-HRRR default candidate family first: `very_regularized_min_leaf70_lgbm_350`.
  - Train quantiles `q05`, `q10`, `q25`, `q50`, `q75`, `q90`, `q95`.
  - Eval: compare single chronological holdout against the no-HRRR promoted stack.
  - Done 2026-04-29: ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models` after rebuilding from canonical WU/NBM/LAMP/HRRR inputs. Models and manifests are under `experiments/withhrrr/data/runtime/models/`.
  - Eval 2026-04-29: validation slice `2025-05-27..2025-12-31` with `219` rows. Canonical with-HRRR q50 MAE/RMSE `1.2723/1.6497`; existing no-HRRR reference on the same slice `1.3063/1.6855`. Validation pinball losses: q05 `0.173021`, q10 `0.290295`, q25 `0.503113`, q50 `0.636131`, q75 `0.497551`, q90 `0.279705`, q95 `0.170042`. Quick comparison artifact: `experiments/withhrrr/data/runtime/evaluation/quick_holdout_comparison.json`.

## Phase 3: Probability-First Evaluation

- [ ] Port degree-ladder scoring.
  - Compute NLL, Brier, RPS, observed-temp probability, modal accuracy, and interval coverage.
  - Eval: compare HRRR-inclusive stack vs no-HRRR stack on the same validation dates.
  - Progress 2026-04-29: reused the no-HRRR evaluator against with-HRRR paths to produce a full holdout report under `experiments/withhrrr/data/runtime/evaluation/full_holdout/`; code still needs an experiment-local with-HRRR evaluator module.
  - Eval 2026-04-29: validation `2025-05-27..2025-12-31`, `219` rows. With-HRRR degree-ladder NLL/Brier/RPS `2.147962/0.822671/0.010102`; no-HRRR reference `2.141472/0.833441/0.010164`.
  - Progress 2026-04-29: ported the evaluator to `experiments/withhrrr/withhrrr_model/evaluate.py` and verified it with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local`.

- [ ] Port representative event-bin scoring.
  - Use the same stable representative bins as no-HRRR so metrics are comparable.
  - Eval: report event-bin NLL, Brier, observed-bin probability, and reliability.
  - Progress 2026-04-29: reused the no-HRRR evaluator with the representative event-bin ladder for both with-HRRR and no-HRRR artifacts.
  - Eval 2026-04-29: validation `2025-05-27..2025-12-31`, `219` rows. With-HRRR event-bin NLL/Brier `1.344095/0.620545`; no-HRRR reference `1.320356/0.630038`. HRRR improved Brier and q50 MAE/RMSE but not event-bin NLL in this first uncalibrated run.
  - Progress 2026-04-29: local with-HRRR evaluator writes representative event-bin scores and reliability outputs under `experiments/withhrrr/data/runtime/evaluation/full_holdout_local/`.

- [ ] Port PIT/CDF calibration diagnostics.
  - Include overall, season, NBM-LAMP disagreement, and HRRR-availability/coverage slices.
  - Eval: record PIT mean/std, tail rates, and any HRRR-specific failure slices.

- [ ] Add HRRR-specific ablation diagnostics.
  - Compare:
    - no-HRRR promoted model artifacts
    - with-HRRR model using all features
    - with-HRRR training with HRRR columns dropped
  - Eval: require same folds and same date coverage for fair comparison.

## Phase 4: Rolling-Origin Model Selection

- [x] Port rolling-origin model selection.
  - Use expanding yearly folds:
    - train through `2023-12-31`, validate `2024-01-01..2024-12-31`
    - train through `2024-12-31`, validate `2025-01-01..2025-12-31`
  - Eval: write fold metrics and selected candidate manifest.
  - Done 2026-04-29: ported `rolling_origin_model_select.py` into `experiments/withhrrr/withhrrr_model/`, removed the no-HRRR ban on `hrrr_` and `meta_hrrr_` features, and ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select`.
  - Eval 2026-04-29: output under `experiments/withhrrr/data/runtime/evaluation/model_selection/`; leakage check passed with `291` features, `105` HRRR features included. Selected candidate remained `very_regularized_min_leaf70_lgbm_350` by weighted event-bin NLL then degree-ladder NLL. Weighted metrics: event-bin NLL `1.508995`, degree-ladder NLL `2.362184`, q50 MAE/RMSE `1.4410/2.0163` across `729` rolling validation rows.

- [ ] Reuse and retune the constrained LightGBM grid.
  - Start from no-HRRR candidates, including `very_regularized_min_leaf70_lgbm_350`.
  - Add only small, explicit HRRR-aware variants if needed after the baseline run.
  - Eval: select by event-bin NLL first, degree-ladder NLL second, q50 MAE/RMSE as secondary.
  - Progress 2026-04-29: reused the full no-HRRR candidate grid unchanged as the first HRRR-inclusive grid. No HRRR-specific candidates added yet.

- [ ] Add leakage and stability checks.
  - Confirm selected features exclude labels, market bins, exact target date, raw timestamp code fields, and source-id code fields.
  - Confirm HRRR features are included in the selected feature set.
  - Eval: write leakage findings and fold-to-fold stability metrics.
  - Progress 2026-04-29: rolling-origin manifest recorded `leakage_check.status=ok`, `finding_count=0`; stability CSV written to `experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_stability.csv`.

## Phase 5: Calibration And Distribution Defaults

- [x] Port rolling-origin quantile calibration.
  - Compare uncalibrated, global offsets, season offsets, NBM-LAMP disagreement offsets, and HRRR-coverage segments.
  - Eval: select only by out-of-time probability metrics.
  - Done 2026-04-29: ported `calibrate_quantiles.py` and `calibrate_rolling_origin.py`; ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin`.
  - Eval 2026-04-29: selected `season_offsets` by event-bin NLL then degree-ladder NLL using 2024 as calibration and 2025 as test. On 2025 test rows, uncalibrated event-bin NLL/Brier `1.412542/0.632488`; selected season offsets `1.291832/0.616072`. Degree-ladder NLL/Brier/RPS improved from `2.284547/0.846069/0.010629` to `2.132255/0.829765/0.010389`. Manifest: `experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json`.

- [ ] Test conformalized interval calibration.
  - Reuse the no-HRRR method and compare against additive offsets.
  - Eval: include interval score and q05-q95/q10-q90/q25-q75 coverage.

- [ ] Port distribution diagnostics.
  - Compare `interpolation_tail`, `interpolation_no_tail`, `smoothed_interpolation_tail`, and `normal_iqr`.
  - Eval: select by event-bin NLL and degree-ladder NLL on 2025 out-of-time validation.

- [ ] Keep monotone rearrangement as a safety guard.
  - Diagnose raw quantile crossings and repair outputs before ladder construction.
  - Eval: record crossing rates by fold and candidate.

- [ ] Decide whether ladder reliability calibration should be promoted.
  - Revalidate using the selected HRRR-inclusive quantile calibration and distribution method.
  - Eval: compare ladder-calibrated vs selected quantile/distribution default.

## Phase 6: Prediction And Event Adapter

- [x] Port prediction CLI.
  - Load HRRR-inclusive feature manifest and quantile model artifacts.
  - Emit expected final high, final-Tmax quantiles, degree ladder, calibration metadata, and optional event-bin probabilities.
  - Eval: smoke on a historical target date with fixed event bins.
  - Done 2026-04-29: ported `predict.py` with with-HRRR defaults and automatic use of `experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json` when present.
  - Eval 2026-04-29: smoke command `.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict --target-date-local 2025-12-31 --event-bin '50F or below' --event-bin '51-55F' --event-bin '56F or higher'` wrote `experiments/withhrrr/data/runtime/predictions/prediction_KLGA_2025-12-31.json`, with calibrated expected final high `33.18F`.

- [ ] Port Polymarket event-bin adapter.
  - Reuse no-HRRR event-bin parsing and mapping behavior.
  - Keep market prices/trading decisions outside the core weather model.
  - Eval: parse a known event slug and map probabilities.

- [ ] Build HRRR-inclusive online inference path.
  - This may be heavier than no-HRRR because HRRR live/current-day inputs are required.
  - For the first version, support historical/local feature rows before adding live HRRR fetching.
  - Eval: clearly mark unavailable status when HRRR is missing.

## Phase 7: Documentation And Tests

- [ ] Add README runbook.
  - Include exact commands for data staging, table build, normalization, training, evaluation, calibration, prediction, and tests.
  - Include comparison against no-HRRR metrics.

- [ ] Add tests.
  - Minimum tests:
    - residual formula
    - HRRR columns are allowed and selected
    - leakage columns are excluded
    - probability mass sums to 1
    - event-bin mapping is stable
    - calibration manifests are applied deterministically

- [ ] Update this TODO after each completed phase.
  - Future agents must keep tick boxes, dates, commands, artifact paths, and eval notes current.

## Later: Paper-Trading Simulation

- [ ] Build paper-trading simulation after HRRR-inclusive calibrated probability quality is reviewed.
  - Join mapped weather probabilities to provided or fetched market prices.
  - Use Polymarket market ids, condition ids, and CLOB token ids from `event_bins.json` as join keys.
  - Include fees, spread, slippage, liquidity, and position sizing assumptions as explicit inputs.
  - Emit model probability, market-implied probability, edge after costs, and action/no-action status per bin.
  - Do not add live order placement.
