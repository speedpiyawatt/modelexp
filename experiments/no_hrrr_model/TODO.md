# No-HRRR Model Optimization TODO

This TODO tracks model-quality improvements for the no-HRRR KLGA overnight experiment without adding new weather input data.

Future agents: update this document after finishing each task. Change `[ ]` to `[x]`, add the completion date, command(s), artifact paths, and a short result note. If a task is blocked, leave it unchecked and add a `Blocked:` note with the exact reason.

For every completed optimization task, also add an `Eval:` note. The note must include the validation slice, the command used, the artifact path, and the before/after or baseline/current metrics needed to judge whether quality improved. Prefer probability metrics first, then q50 MAE/RMSE:

- degree-ladder NLL, Brier score, ranked probability score, and observed-temp probability
- representative event-bin NLL, Brier score, and observed-bin probability
- PIT summary and interval coverage when calibration changed
- q50 MAE/RMSE as secondary diagnostics

Scope guardrails:

- Keep this work inside `experiments/no_hrrr_model/`.
- Do not add HRRR inputs or `hrrr_` / `meta_hrrr_` placeholder fields.
- Optimize for calibrated KLGA final-Tmax probability mass over whole-degree bins.
- Use `America/New_York` local target dates and the `00:05` overnight cutoff.
- Prefer rolling-origin and out-of-time validation over random splits.
- Do not add live order placement in this phase.

## Phase 1: Probability-First Evaluation

- [x] Add degree-ladder scoring to evaluation.
  - Score the internal 1°F ladder against observed `final_tmax_f`.
  - Include negative log likelihood, Brier score, ranked probability score or CRPS-style score, and observed-vs-predicted reliability buckets.
  - Write outputs under `experiments/no_hrrr_model/data/runtime/evaluation/`.
  - Done 2026-04-28: implemented in `experiments/no_hrrr_model/no_hrrr_model/evaluate.py`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests` and `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.evaluate`. Outputs include `degree_ladder_scores.csv`, `degree_ladder_reliability.csv`, and `degree_ladder_metrics.json`.
  - Eval 2026-04-28: validation slice `2025-05-27` through `2025-12-31`, `219` rows. Baseline metrics in `experiments/no_hrrr_model/data/runtime/evaluation/degree_ladder_metrics.json` after tail-mass fix: degree-ladder NLL `2.1122`, Brier `0.8278`, ranked probability score `0.01000`, mean observed-temp probability `0.1887`, modal accuracy `0.2374`, q05-q95 coverage `0.8904`, q10-q90 coverage `0.7671`, q25-q75 coverage `0.4155`. Zero observed-temp probability rows: `0 / 219`. This step adds evaluation coverage and fixes probability-ladder tail scoring; it does not change quantile model predictions.

- [x] Add event-bin probability scoring.
  - Map validation predictions into representative Polymarket-style bins before scoring.
  - Report per-bin Brier/log-loss style metrics and calibration tables.
  - Keep event bins as an adapter/evaluation layer, not a training target.
  - Done 2026-04-28: added representative event-bin scoring to `evaluate.py`. Validation outputs include `event_bin_scores.csv`, `event_bin_metrics.csv`, `event_bin_metrics.json`, and `event_bin_reliability.csv`.
  - Eval 2026-04-28: validation slice `2025-05-27` through `2025-12-31`, `219` rows. Baseline metrics in `experiments/no_hrrr_model/data/runtime/evaluation/event_bin_metrics.json` after stable-bin and tail-mass fixes: representative event-bin NLL `1.2835`, Brier `0.6109`, mean observed-bin probability `0.3915`, `31` fixed representative bins from `35F or below` through `94F or higher`. Zero observed-bin probability rows: `0 / 219`. This step adds stable evaluation coverage; it does not change quantile model predictions.

- [x] Add PIT or CDF calibration diagnostics.
  - Compute where observed final Tmax lands inside the predicted distribution.
  - Report overall, seasonal, and high-disagreement slices.
  - Done 2026-04-28: added PIT lower/mid/upper fields to `degree_ladder_scores.csv`, slice summaries to `pit_diagnostics.csv`, modal-confidence reliability to `degree_ladder_modal_reliability.csv`, and per-degree calibration reliability to `degree_ladder_reliability.csv`.
  - Eval 2026-04-28: validation slice `2025-05-27` through `2025-12-31`, `219` rows. Baseline PIT summary in `experiments/no_hrrr_model/data/runtime/evaluation/pit_diagnostics.csv`: overall PIT mean `0.4784`, PIT std `0.2821`, PIT below `0.1` rate `0.0868`, PIT above `0.9` rate `0.0776`; warm-season PIT mean `0.4660`; cool-season PIT mean `0.5107`; NBM-LAMP disagreement `2_to_5f` PIT mean `0.5703`. This step adds calibration diagnostics and per-degree reliability; it does not change quantile model predictions.

## Phase 2: Rolling-Origin Model Selection

- [x] Build `rolling_origin_model_select.py`.
  - Use expanding-window folds instead of a single chronological validation split.
  - Include at least yearly folds: train through `2023-12-31` -> validate `2024`, and train through `2024-12-31` -> validate `2025`.
  - Add optional shorter seasonal/monthly folds when enough rows are available.
  - Done 2026-04-28: implemented `experiments/no_hrrr_model/no_hrrr_model/rolling_origin_model_select.py`. Supports default yearly folds and optional custom split config via `--splits-path`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests` and `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.rolling_origin_model_select`. Outputs written under `experiments/no_hrrr_model/data/runtime/evaluation/model_selection/`.
  - Eval 2026-04-28: yearly expanding folds are `train<=2023-12-31 -> validate 2024-01-01..2024-12-31` and `train<=2024-12-31 -> validate 2025-01-01..2025-12-31`, `729` validation rows total. Fold metrics are in `rolling_origin_model_selection_fold_metrics.csv`. Current default fold 2024: event-bin NLL `1.7705`, degree-ladder NLL `2.6520`, q50 MAE/RMSE `1.4812/2.0691`, q05-q95 coverage `0.7225`. Current default fold 2025: event-bin NLL `1.5602`, degree-ladder NLL `2.5617`, q50 MAE/RMSE `1.3720/1.9435`, q05-q95 coverage `0.7452`.

- [x] Select models by probabilistic score.
  - Primary selection metric should be degree-ladder or event-bin probability quality.
  - Keep q50 MAE/RMSE as secondary diagnostics.
  - Persist candidate configs, fold metrics, and selected defaults in a manifest.
  - Done 2026-04-28: selector ranks candidates by validation-row-weighted `event_bin_nll`, then weighted `degree_ladder_nll`; selected candidate and config are written to `rolling_origin_model_selection_manifest.json`. Candidate IDs are validated as unique.
  - Eval 2026-04-28: only the current parameter candidate is included before Phase 4 tuning, so selected candidate is `current_lgbm_fixed_250_no_inner_es`. This name is intentionally precise: it uses current LightGBM params with fixed `250` rounds and no inner early stopping for leak-free rolling-origin scoring. Weighted rolling metrics in `rolling_origin_model_selection_summary.csv`: event-bin NLL `1.6652`, event-bin Brier `0.6855`, degree-ladder NLL `2.6068`, degree-ladder Brier `0.8800`, degree-ladder RPS `0.01135`, q50 MAE/RMSE `1.4265/2.0062`, q05-q95 coverage `0.7339`.

- [x] Add leakage and stability checks to model-selection output.
  - Confirm selected feature names still exclude label, market, exact target-date, raw timestamp-code, and source-identifier leakage.
  - Report fold-to-fold variance for q50 MAE, interval coverage, and probability scores.
  - Done 2026-04-28: added feature leakage checks for forbidden exact fields, `label_`, `hrrr_`, `meta_hrrr_`, market fields, raw timestamp-code fields, and source-code fields. Stability output is written to `rolling_origin_model_selection_stability.csv`.
  - Eval 2026-04-28: leakage check passed with `finding_count=0` across `180` selected features. Fold standard deviations: event-bin NLL `0.1051`, event-bin Brier `0.0375`, degree-ladder NLL `0.0451`, degree-ladder Brier `0.0053`, q50 MAE `0.0546`, q50 RMSE `0.0628`, q05-q95 coverage `0.0113`.

## Phase 3: Anchor Optimization

- [x] Tune fixed NBM/LAMP anchor weights.
  - Evaluate `anchor_tmax_f = w * nbm_tmax_open_f + (1 - w) * lamp_tmax_open_f`.
  - Test `w` from `0.0` to `1.0` on a small grid.
  - Select by rolling-origin probabilistic score, not in-sample error.
  - Done 2026-04-28: implemented `experiments/no_hrrr_model/no_hrrr_model/rolling_origin_anchor_select.py` with fixed NBM-weight grid `0.0` through `1.0` by `0.1`. Verified with `.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests` and `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.rolling_origin_anchor_select`. Outputs written under `experiments/no_hrrr_model/data/runtime/evaluation/anchor_selection/`.
  - Eval 2026-04-28: selected fixed anchor is `fixed_nbm_weight_0.6`, meaning `0.6 * NBM + 0.4 * LAMP`, by weighted event-bin NLL. Compared with current `fixed_nbm_weight_0.5`: event-bin NLL `1.6652 -> 1.6515` improved, event-bin Brier `0.6855 -> 0.6844` improved, degree-ladder NLL `2.6068 -> 2.6064` about flat, degree-ladder RPS `0.01135 -> 0.01128` improved, q50 MAE/RMSE `1.4265/2.0062 -> 1.4290/2.0088` slightly worse, q05-q95 coverage `0.7339 -> 0.7284` slightly worse. This is a probability-score improvement with small point-error and coverage tradeoffs.
  - Refresh 2026-04-28: after Phase 4 selected `very_regularized_lgbm_350`, reran anchor selection so artifacts use the new model default. New selected fixed anchor is `fixed_nbm_weight_0.7`, meaning `0.7 * NBM + 0.3 * LAMP`. Compared with the Phase 4 50/50 anchor: event-bin NLL `1.5667 -> 1.5518`, event-bin Brier `0.6670 -> 0.6601`, degree-ladder NLL `2.4605 -> 2.4372`, degree-ladder RPS `0.01131 -> 0.01120`, q50 MAE/RMSE `1.4581/2.0314 -> 1.4471/2.0246`, q05-q95 coverage `0.7805 -> 0.7929`.

- [x] Test simple segmented anchor weights.
  - Try month, warm/cool season, and high-vs-low `abs(nbm_minus_lamp_tmax_f)` segments.
  - Keep the segmentation small enough to avoid overfitting the limited history.
  - Done 2026-04-28: added fold-local train-MAE-selected segmented anchors for warm/cool season, calendar month, and NBM-LAMP disagreement buckets. Segment weights are learned only from each fold's training rows and written to `rolling_origin_anchor_selection_anchor_metadata.csv`.
  - Eval 2026-04-28: best segmented candidate was `segmented_disagreement_train_mae`: event-bin NLL `1.6593`, event-bin Brier `0.6875`, degree-ladder NLL `2.6233`, q50 MAE/RMSE `1.4318/2.0183`, q05-q95 coverage `0.7407`. It beats current 50/50 on event-bin NLL but does not beat tuned fixed `0.6` NBM weight. Season segmentation was weaker: event-bin NLL `1.6848`, q50 MAE/RMSE `1.4505/2.0291`. Month segmentation was weaker still: event-bin NLL `1.7129`, event-bin Brier `0.6956`, degree-ladder NLL `2.6472`, q50 MAE/RMSE `1.4767/2.0844`, q05-q95 coverage `0.7366`.
  - Review fix 2026-04-28: `load_splits` now rejects overlapping train/validation windows and reversed validation ranges; segmented anchor weight selection now falls back cleanly when a segment has no finite labels or anchor inputs. Verified with `.venv/bin/python -m pytest experiments/no_hrrr_model/tests/test_no_hrrr_model.py` and reran `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.rolling_origin_anchor_select`.

- [x] Test a regularized linear anchor baseline.
  - Use only safe anchor inputs such as NBM open Tmax, LAMP open Tmax, season/month, and NBM-LAMP disagreement.
  - Compare against the fixed 50/50 anchor and tuned fixed-weight anchor.
  - Done 2026-04-28: added fold-local ridge anchor with features `intercept`, `nbm_tmax_open_f`, `lamp_tmax_open_f`, `abs_nbm_minus_lamp_tmax_f`, `month_sin`, and `month_cos`; coefficients are fit only on each fold's training rows and written to `rolling_origin_anchor_selection_anchor_metadata.csv`.
  - Eval 2026-04-28: `ridge_linear_anchor` improved event-bin Brier `0.6855 -> 0.6765` and q50 MAE `1.4265 -> 1.4241` versus current 50/50, but event-bin NLL worsened `1.6652 -> 1.6831`, degree-ladder NLL worsened `2.6068 -> 2.6603`, and RMSE worsened `2.0062 -> 2.0113`. It is not selected by probability-first criteria.

## Phase 4: Constrained LightGBM Tuning

- [x] Add a small, explicit LightGBM search space.
  - Candidate ranges should emphasize regularization: `num_leaves`, `min_data_in_leaf`, `max_depth`, `lambda_l1`, `lambda_l2`, `feature_fraction`, and `bagging_fraction`.
  - Avoid broad AutoML over the small training set.
  - Done 2026-04-28: expanded `experiments/no_hrrr_model/no_hrrr_model/rolling_origin_model_select.py` from the current baseline to a six-candidate constrained grid: `current_lgbm_fixed_250_no_inner_es`, `regularized_shallow_lgbm_300`, `regularized_moderate_lgbm_300`, `high_min_leaf_lgbm_250`, `very_regularized_lgbm_350`, and `slower_regularized_lgbm_350`. Tuned candidates cap depth, raise `min_data_in_leaf`, add `lambda_l1`, strengthen `lambda_l2`, and use feature/bagging subsampling.
  - Eval 2026-04-28: baseline current config before tuning had weighted event-bin NLL/Brier `1.6652/0.6855`, degree-ladder NLL/Brier/RPS `2.6068/0.8800/0.01135`, q50 MAE/RMSE `1.4265/2.0062`, q05-q95 coverage `0.7339`.

- [x] Tune quantile models through rolling-origin folds.
  - Evaluate each candidate across all configured folds.
  - Record pinball loss by quantile, q50 MAE/RMSE, interval coverage, degree-ladder score, and event-bin score.
  - Done 2026-04-28: rolling-origin model selection now aggregates per-quantile pinball loss into `rolling_origin_model_selection_summary.csv` and stability output. Reran `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.rolling_origin_model_select`; outputs were written under `experiments/no_hrrr_model/data/runtime/evaluation/model_selection/`.
  - Eval 2026-04-28: best probability candidate was `very_regularized_lgbm_350`: weighted event-bin NLL/Brier `1.5667/0.6670`, degree-ladder NLL/Brier/RPS `2.4605/0.8611/0.01131`, q50 MAE/RMSE `1.4581/2.0314`, q05-q95 coverage `0.7805`, q50 pinball `0.7290`, event-bin NLL fold std `0.1055`, q50 MAE fold std `0.0426`. Runner-up `regularized_shallow_lgbm_300` had slightly better degree-ladder NLL `2.4474`, q50 MAE/RMSE `1.4367/2.0124`, and coverage `0.7901`, but worse event-bin NLL `1.5761`.

- [x] Select conservative default hyperparameters.
  - Prefer stable out-of-time probability quality over the best single-fold point error.
  - Write the selected config into the model manifest.
  - Done 2026-04-28: selected `very_regularized_lgbm_350` as the default probability-first config and wired it into `DEFAULT_MODEL_CANDIDATE_ID`, Phase 3 anchor selection's model candidate lookup, and `train_quantile_models.py`. Reran `.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.train_quantile_models`; `experiments/no_hrrr_model/data/runtime/models/training_manifest.json` now records `model_candidate_id=very_regularized_lgbm_350`, `num_boost_round=350`, and the selected LightGBM params.
  - Eval 2026-04-28: selected vs current improved event-bin NLL `1.6652 -> 1.5667`, event-bin Brier `0.6855 -> 0.6670`, degree-ladder NLL `2.6068 -> 2.4605`, degree-ladder Brier `0.8800 -> 0.8611`, and q05-q95 coverage `0.7339 -> 0.7805`. Accepted point-error regression for probability improvement: q50 MAE/RMSE worsened `1.4265/2.0062 -> 1.4581/2.0314`, and q50 pinball worsened `0.7128 -> 0.7290`.
  - Review fix 2026-04-28: centralized model candidates and the selected default in `experiments/no_hrrr_model/no_hrrr_model/model_config.py`, changed final training to use fixed `350` rounds with no inner early stopping, and regenerated model-selection, training, and anchor-selection manifests so they all reference `very_regularized_lgbm_350`.

## Phase 5: Calibration Improvements

- [ ] Add segmented quantile calibration.
  - Compare global offsets against month/season and NBM-LAMP-disagreement segments.
  - Require out-of-time improvement before making segmented calibration the default.
  - Eval: not started. Compare segmented calibration against current rolling-origin calibration on coverage, PIT, interval score, degree-ladder NLL/Brier/RPS, and event-bin NLL/Brier.

- [ ] Implement conformalized quantile calibration.
  - Use rolling-origin predictions as calibration data.
  - Preserve valid out-of-time interval behavior while keeping intervals as sharp as practical.
  - Compare against current additive quantile offsets.
  - Eval: not started. Record conformal vs additive calibration coverage, interval width/score, PIT, probability scores, and q50 MAE/RMSE.

- [ ] Calibrate the final 1°F probability ladder.
  - Adjust probability mass using validation reliability diagnostics without breaking total probability or monotonic CDF behavior.
  - Validate on held-out rolling-origin folds before updating `predict.py` defaults.
  - Eval: not started. Compare calibrated vs uncalibrated ladder NLL/Brier/RPS and event-bin NLL/Brier on held-out folds.

## Phase 6: Quantile Crossing And Distribution Shape

- [ ] Expand quantile-crossing diagnostics.
  - Report crossing frequency by fold, season, quantile pair, and candidate hyperparameter config.
  - Penalize configs with frequent crossing even if monotone rearrangement can repair outputs.
  - Eval: not started. Record crossing-rate reductions and whether probability metrics improve after penalizing crossing-heavy configs.

- [ ] Compare distribution-construction methods.
  - Compare the current interpolation-based 1°F ladder with alternative smooth CDF or kernelized residual methods.
  - Score methods by whole-degree probability calibration and event-bin metrics.
  - Eval: not started. Compare each method against current interpolation baseline using degree-ladder and event-bin probability metrics.

- [ ] Keep monotone rearrangement as a safety guard.
  - Do not emit non-monotone quantiles from prediction artifacts.
  - Document whether crossing was repaired and how often.
  - Eval: not started. Record repair frequency and any score differences before/after rearrangement.

## Phase 7: Small Ensembles

- [ ] Test a small ensemble of stable model variants.
  - Combine only a few strong rolling-origin candidates, such as best LightGBM configs or anchor variants.
  - Average final 1°F ladders or monotone quantile outputs.
  - Eval: not started. Compare ensemble vs selected single model on rolling-origin probability metrics, PIT, coverage, and q50 MAE/RMSE.

- [ ] Compare ensemble against the selected single model.
  - Require improvement in probability scores and calibration stability.
  - Avoid large ensembles that hide overfit candidate selection.
  - Eval: not started. Record whether ensemble improvement is large enough to justify added complexity.

## Phase 8: Prediction Defaults And Runbook Updates

- [ ] Update `predict.py` defaults only after out-of-time validation.
  - Use the selected anchor, model config, distribution method, and calibration manifest.
  - Preserve override flags for comparing old and new behavior.
  - Eval: not started. Record old-default vs new-default metrics and artifact paths.

- [ ] Update README and manifests.
  - Document the selected validation design, score targets, model-selection results, and current default artifacts.
  - Include exact commands to rerun selection and calibration.
  - Eval: not started. Include final before/after metric table in the README or linked report.

- [ ] Add regression tests for selected behavior.
  - Test probability mass sums to 1.
  - Test event-bin mapping remains stable.
  - Test calibration manifests are applied deterministically.
  - Test no HRRR columns enter feature selection.
  - Eval: not started. Record test command and passing test count.

## Later: Paper-Trading Simulation

- [ ] Build paper-trading simulation after calibrated probability quality is reviewed.
  - Join mapped weather probabilities to provided or fetched market prices.
  - Use Polymarket market ids, condition ids, and CLOB token ids from `event_bins.json` as join keys.
  - Include fees, spread, slippage, liquidity, and position sizing assumptions as explicit inputs.
  - Emit model probability, market-implied probability, edge after costs, and action/no-action status per bin.
  - Do not add live order placement.
  - Eval: not started. Record paper-trading hit rate, calibration, gross/net edge after costs, no-action rate, and drawdown by date range.
