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

Selected target contract:

```text
nbm_tmax_open_f = fahrenheit(nbm_temp_2m_day_max_k)
lamp_tmax_open_f = lamp_day_temp_max_f or canonical LAMP equivalent
hrrr_tmax_open_f = fahrenheit(hrrr_temp_2m_day_max_k)
anchor_tmax_f = (nbm_tmax_open_f + lamp_tmax_open_f + hrrr_tmax_open_f) / 3
target_residual_f = label_final_tmax_f - anchor_tmax_f
```

Important: HRRR enters as both the selected equal 3-way anchor component and model input features. Do not replace the model with an HRRR-only anchor or staged HRRR ensemble unless a later evaluation proves that is better and the user approves it.

## Latest Optimization Pass: Anchor Policy Selection

- [x] Test native NBM TMAX and HRRR-inclusive anchor policies.
  - Done 2026-04-29: `prepare_training_features.py` now supports `--anchor-policy` with `current_50_50`, `hourly_native_lamp`, `hourly_native_lamp_hrrr`, `native_lamp`, and `equal_3way`. It also derives native-NBM disagreement features such as `nbm_native_tmax_minus_anchor_f`, `nbm_native_tmax_minus_nbm_tmax_f`, and 2F direction flags.
  - Eval 2026-04-29: raw rolling-origin comparison across 729 validation rows wrote `experiments/withhrrr/data/runtime/anchor_experiments/anchor_policy_comparison.csv`. Event-bin NLL ranked: `equal_3way=1.486443`, `hourly_native_lamp_hrrr=1.497680`, `hourly_native_lamp=1.504711`, `native_lamp=1.516323`, `current_50_50=1.518486`.
  - Eval 2026-04-29: calibrated top-two comparison selected `equal_3way`. With `hrrr_nbm_direction_offsets`, `normal_iqr`, and `bucket_reliability_s1_00`, 2025 event-bin NLL/Brier is `1.240912/0.603902`; degree NLL/RPS is `1.999327/0.009510`. The HRRR+native blend was second with event-bin NLL/Brier `1.250025/0.606569`.
  - Decision 2026-04-29: promoted `equal_3way` as the default anchor and `very_regularized_min_leaf70_lgbm_350` as the default candidate because it has the best calibrated historical probability metrics. A real April 26, 2026 replay showed this can worsen a specific HRRR-cold/native-NBM-warm miss, but one live miss is not enough evidence to override the stronger 2025 rolling validation result. Native NBM TMAX remains a model feature and diagnostic, and the 4-way `hourly_native_lamp_hrrr` anchor remains the best fallback if later regime testing proves it generalizes better.

## Latest Optimization Pass: HRRR Disagreement Regimes

- [x] Add cutoff-safe nearby-station Wunderground context and rerun the full selection/calibration stack.
  - Done 2026-04-30: added `withhrrr_model/nearby_observations.py` and integrated `KJRB`, `KJFK`, `KEWR`, and `KTEB` Wunderground observation features into `prepare_training_features.py`. Nearby features are cutoff-safe at `00:05 America/New_York` and use a stale-observation guard so an old station report is not reused for a later target day.
  - Done 2026-04-30: generalized `wunderground/build_training_tables.py` beyond hardcoded `KLGA_9_US_*.json` and added `--skip-http-status` support to `wunderground/fetch_daily_history.py` so sparse/no-data nearby days do not abort a full station backfill.
  - Done 2026-04-30: added nearby-aware feature profiles and focused LightGBM candidates. Rolling selection now includes `source_trust_nearby_features`, `high_disagreement_weighted_nearby`, nearby specialist profiles, and nearby-tuned candidates around the previous winner.
  - Done 2026-04-30: updated online inference so live with-HRRR runs fetch/build nearby Wunderground rows for `KJRB`, `KJFK`, `KEWR`, and `KTEB`, pass those obs parquets into `build_inference_features.py`, and delete the nearby downloaded/intermediate artifacts with the rest of the runtime source data after successful prediction.
  - Eval 2026-04-30: rebuilt `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet`; table has `1,096` rows and `508` columns. Availability: NBM/LAMP/HRRR `1096/1096`; KJRB `1086/1096`, KJFK `1092/1096`, KEWR `1092/1096`, KTEB `1091/1096`; nearby station count is `4` on `968` rows, `3` on `124` rows, and `0` on `4` rows. Label parity check against `final_tmax_f` had max absolute diff `0.0`; leakage scan found `0` cutoff violations.
  - Eval 2026-04-30: full rolling-origin grid evaluated `47` candidate specs. Selected candidate: `nearby_vreg_leaf100_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted_nearby__weights=high_disagreement_weighted`; weighted rolling event-bin NLL `1.459584`, degree-ladder NLL `2.281087`, q50 MAE/RMSE `1.443267/1.996376`.
  - Eval 2026-04-30: refreshed calibration/distribution/ladder/final model artifacts. Selected quantile calibration `global_offsets`, distribution `normal_iqr`, ladder `bucket_reliability_s1_00`. Rolling 2025 ladder-calibrated event-bin NLL/Brier `1.244989/0.608950`; degree NLL/RPS `2.000670/0.009480`. Disagreement widening remained unpromoted.
  - Eval 2026-04-30: after diagnosing `2026-04-29`, added guarded calibration selection with candidates `global_offsets`, `conformal_intervals`, `no_offsets`, `global_offsets_no_upper_tail`, `global_offsets_shrunk_50pct`, and conditional source-disagreement variants. The guard rejects candidates that reduce observed-bin probability by more than `0.02` in `tight_consensus` or `moderate_disagreement`; `global_offsets` and `conformal_intervals` were rejected, and `global_offsets_no_upper_tail` was selected. Refreshed rolling ladder metrics: event-bin NLL/Brier `1.241810/0.606852`; degree NLL/RPS `2.000715/0.009475`.
  - Eval 2026-04-30: refreshed production-calibrated holdout `2025-05-27..2025-12-31`, `219` rows. Event-bin NLL/Brier `1.133178/0.603449`; degree NLL/RPS `1.861432/0.009238`; raw q50 MAE/RMSE `1.261223/1.639528`; scored q50 MAE/RMSE `1.265688/1.642842`.
  - Eval 2026-04-30: selected source slices from rolling validation: high-disagreement event-bin NLL/Brier `1.675893/0.710495`; `native_warm_hrrr_cold` `1.593042/0.722225`; `hrrr_cold_outlier` `1.503710/0.567461`; `tight_consensus` `1.450139/0.664654`.
  - Verification 2026-04-30: `.venv/bin/python -m py_compile wunderground/*.py experiments/withhrrr/withhrrr_model/*.py` and `.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py` passed (`27 passed`).
  - Smoke 2026-04-30: built one inference feature row for `2025-12-31` using existing local artifacts and nearby obs paths. The row had all four nearby stations available, `nearby_feature_count=106`, `feature_count=467`, and `model_prediction_available=true`; `predict.py` returned a valid event-bin distribution.
  - Review fixes 2026-04-30: fixed server dual inference to use the with-HRRR online wrapper so nearby station features are fetched; fixed feature-profile filtering so non-nearby profiles have `0` nearby-like columns; fixed nearby-selected inference to fail when no nearby station observations are available.
  - Verification 2026-04-30 after review fixes: `.venv/bin/python -m py_compile wunderground/*.py experiments/withhrrr/withhrrr_model/*.py tools/weather/run_server_dual_inference.py` passed; `.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py` passed (`30 passed`); all-nearby inference smoke passed; no-nearby inference smoke failed explicitly as intended.
  - Deploy status 2026-04-30: code pushed to GitHub `origin/main` and the DigitalOcean server `/root/modelexp` has been pulled/synced with ignored runtime artifacts; later code commits or artifact rebuilds still need explicit server sync.

- [x] Add source-trust dynamic anchor and specialist candidate system.
  - Done 2026-04-29: added `withhrrr_model/source_trust.py`, richer pairwise source deltas, source ranks, warmest/coldest flags, WU-last-temp-vs-source deltas, target month/day-of-year features, median 4-way and trimmed-mean 4-way anchors, fold-local ridge 4-way anchors, weighted source-trust candidate profiles, and an optional source-trust meta residual correction candidate.
  - Done 2026-04-29: `rolling_origin_model_select.py` now evaluates candidate specs as `(anchor_policy, model_candidate_id, feature_profile, weight_profile, meta_residual)` and writes `rolling_origin_model_selection_source_slices.csv` with high-disagreement and regime-specific probability metrics.
  - Done 2026-04-29: `train_quantile_models.py`, `build_inference_features.py`, `evaluate.py`, and `predict.py` now read the selected anchor/profile metadata from model-selection/training manifests. If a ridge anchor or meta residual correction is selected, the required metadata/artifact is saved and loaded explicitly.
  - Eval 2026-04-29: smoke validation over actual training parquet passed with a `source_median_4way` candidate and one rolling split. Commands: `.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features`, a temp one-candidate `rolling_origin_model_select`, `train_quantile_models --model-selection-manifest-path ...`, and `evaluate --models-dir ...`.
  - Eval 2026-04-29: local tests passed: `.venv/bin/python -m py_compile experiments/withhrrr/withhrrr_model/*.py tools/weather/run_server_dual_inference.py` and `.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py` (`26 passed`).
  - Eval 2026-04-29: full default rolling-origin grid evaluated `32` candidate specs. Selected candidate: `very_regularized_min_leaf70_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted__weights=high_disagreement_weighted`; weighted rolling event-bin NLL `1.462654`, degree-ladder NLL `2.325077`, q50 MAE `1.415372`.
  - Eval 2026-04-29: dynamic 4-way/native/median/trimmed/ridge anchors and the meta residual candidate were evaluated but not promoted. Top alternatives: `native_warm_hrrr_cold_specialist` event NLL `1.464717`, `hrrr_outlier_specialist` `1.471316`, `hourly_native_lamp_hrrr` anchor `1.488369`, `ridge_4way_anchor` `1.497106`, meta residual `1.496992`.
  - Eval 2026-04-29: refreshed calibration/distribution/ladder/final model artifacts. Selected quantile calibration `conformal_intervals`, distribution `normal_iqr`, ladder `bucket_reliability_s1_00`. Rolling 2025 ladder-calibrated event-bin NLL/Brier `1.250237/0.607134`; degree NLL/RPS `2.008158/0.009521`.
  - Eval 2026-04-29: refreshed holdout `2025-05-27..2025-12-31`, `219` rows. Event-bin NLL/Brier `1.372809/0.619639`; degree NLL/RPS `2.149011/0.010002`; q50 MAE/RMSE `1.252238/1.651186`.
  - Deploy 2026-04-29: pushed code as commit `20330a1` and pulled it on `/root/modelexp` at `root@198.199.64.163`. Synced ignored runtime artifacts for `models`, `model_selection`, `calibration_selection`, `distribution_diagnostics`, and `ladder_calibration`; server verification found feature manifest `equal_3way/high_disagreement_weighted/high_disagreement_weighted` with `350` features.

- [x] Add source-disagreement robustness layer.
  - Done 2026-04-29: added shared source-regime features for NBM hourly, native NBM TMAX, LAMP, and HRRR. Regimes include `native_warm_hrrr_cold`, `native_cold_hrrr_warm`, HRRR hot/cold outliers, broad/moderate disagreement, tight consensus, and unknown.
  - Done 2026-04-29: `calibrate_rolling_origin.py` now evaluates `source_disagreement_regime_offsets` with hierarchical shrinkage: segment offsets require at least `30` rows and shrink toward global with weight `min(1, count/120)`.
  - Done 2026-04-29: `calibrate_ladder.py` now evaluates source-disagreement widening candidates at `0.5F`, `1.0F`, and `1.5F`; `predict.py` can apply a selected widening method only on high-disagreement regimes.
  - Eval 2026-04-29: rolling 2025 calibration did not promote the new source-regime quantile method. `source_disagreement_regime_offsets` had event-bin NLL/Brier `1.370733/0.616891`, worse than selected `hrrr_nbm_direction_offsets` at `1.305226/0.606479`.
  - Eval 2026-04-29: widening did not promote. Best selected ladder remains `bucket_reliability_s1_00` with event-bin NLL/Brier `1.240912/0.603902`; widening candidates were `0.5F=1.244622`, `1.0F=1.248985`, `1.5F=1.257902` event-bin NLL.
  - Eval 2026-04-29: source-regime diagnostics now write `metrics_by_source_disagreement_regime.csv` and `ladder_calibration_disagreement_slices.csv`. Holdout high-disagreement slice has event-bin NLL/Brier `1.519170/0.652170`; `native_warm_hrrr_cold` has `1.255661/0.684633`.

- [x] Promote HRRR-vs-source disagreement into first-class model features.
  - Done 2026-04-29: `prepare_training_features.py` now derives `hrrr_tmax_open_f`, `hrrr_minus_lamp_tmax_f`, `hrrr_minus_nbm_tmax_f`, absolute disagreement columns, `anchor_equal_3way_tmax_f`, HRRR-outside-NBM/LAMP-range columns, and 3F hotter/colder boolean flags.
  - Eval 2026-04-29: rebuilt `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet`; table has `1,096` rows, `337` columns, and `1,094` eligible rows. Selected feature manifest now has `304` features.

- [x] Add HRRR disagreement and direction slices to evaluation.
  - Done 2026-04-29: `evaluate.py` now writes `metrics_by_hrrr_lamp_disagreement.csv`, `metrics_by_hrrr_nbm_disagreement.csv`, `metrics_by_hrrr_lamp_direction.csv`, and `metrics_by_hrrr_nbm_direction.csv`; PIT diagnostics also include HRRR-vs-LAMP/NBM disagreement and direction slices.
  - Eval 2026-04-29: holdout `2025-05-27..2025-12-31`, `219` rows. HRRR-hotter-than-LAMP-by-3F+ slice has q50 MAE/RMSE `1.4275/1.7435`; HRRR-hotter-than-NBM-by-3F+ slice has `1.5960/2.0167`; HRRR-within-3F-of-NBM slice has `1.1495/1.4927`.

- [x] Add HRRR-aware calibration candidates and online application.
  - Done 2026-04-29: `calibrate_rolling_origin.py` now evaluates HRRR-LAMP disagreement, HRRR-NBM disagreement, HRRR-LAMP direction, and HRRR-NBM direction segmented offsets. `predict.py` can apply any selected HRRR segment method at inference time.
  - Eval 2026-04-29: rolling calibration on 2024, tested on 2025. Selected method is `conformal_intervals`; event-bin NLL/Brier improved from uncalibrated `1.370969/0.622307` to `1.274655/0.615411`. HRRR-aware offsets were tested but did not beat conformal intervals on event-bin NLL.

- [x] Promote the rolling-origin-selected with-HRRR candidate.
  - Done 2026-04-29: `model_config.py` now sets `DEFAULT_MODEL_CANDIDATE_ID="regularized_shallow_lgbm_300"` for the with-HRRR experiment instead of inheriting the no-HRRR default.
  - Eval 2026-04-29: rolling-origin selected `regularized_shallow_lgbm_300` by weighted event-bin NLL then degree-ladder NLL. The retrained production artifact reports holdout q50 MAE/RMSE `1.2413/1.6092`, degree-ladder NLL/Brier/RPS `2.114456/0.824174/0.010029`, and event-bin NLL/Brier `1.321593/0.618214`.

- [x] Add HRRR anchor diagnostics without changing the residual target.
  - Done 2026-04-29: `evaluate.py` compares HRRR-only, fixed equal 3-way NBM/LAMP/HRRR anchor, and learned linear NBM/LAMP/HRRR blend baselines.
  - Eval 2026-04-29: holdout q50 MAE/RMSE: NBM-only `1.5735/2.0401`, LAMP-only `1.3744/1.8245`, HRRR-only `1.9757/2.5496`, fixed 50/50 NBM/LAMP anchor `1.3734/1.7833`, equal 3-way anchor `1.3726/1.7806`, linear 3-way blend `1.3990/1.7922`, final residual quantile model `1.2413/1.6092`. HRRR should remain a learned feature/regime signal, not the primary anchor, with the current input data.

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

- [x] Port degree-ladder scoring.
  - Compute NLL, Brier, RPS, observed-temp probability, modal accuracy, and interval coverage.
  - Eval: compare HRRR-inclusive stack vs no-HRRR stack on the same validation dates.
  - Progress 2026-04-29: reused the no-HRRR evaluator against with-HRRR paths to produce a full holdout report under `experiments/withhrrr/data/runtime/evaluation/full_holdout/`; code still needs an experiment-local with-HRRR evaluator module.
  - Eval 2026-04-29: validation `2025-05-27..2025-12-31`, `219` rows. With-HRRR degree-ladder NLL/Brier/RPS `2.147962/0.822671/0.010102`; no-HRRR reference `2.141472/0.833441/0.010164`.
  - Progress 2026-04-29: ported the evaluator to `experiments/withhrrr/withhrrr_model/evaluate.py` and verified it with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local`.
  - Eval 2026-04-29 optimized pass: retrained selected candidate with HRRR disagreement features. Validation `2025-05-27..2025-12-31`, `219` rows: degree-ladder NLL/Brier/RPS `2.114456/0.824174/0.010029`.

- [x] Port representative event-bin scoring.
  - Use the same stable representative bins as no-HRRR so metrics are comparable.
  - Eval: report event-bin NLL, Brier, observed-bin probability, and reliability.
  - Progress 2026-04-29: reused the no-HRRR evaluator with the representative event-bin ladder for both with-HRRR and no-HRRR artifacts.
  - Eval 2026-04-29: validation `2025-05-27..2025-12-31`, `219` rows. With-HRRR event-bin NLL/Brier `1.344095/0.620545`; no-HRRR reference `1.320356/0.630038`. HRRR improved Brier and q50 MAE/RMSE but not event-bin NLL in this first uncalibrated run.
  - Progress 2026-04-29: local with-HRRR evaluator writes representative event-bin scores and reliability outputs under `experiments/withhrrr/data/runtime/evaluation/full_holdout_local/`.
  - Eval 2026-04-29 optimized pass: representative event-bin NLL/Brier `1.321593/0.618214`, improving the earlier with-HRRR uncalibrated event-bin NLL `1.344095` and preserving the Brier gain.

- [x] Port PIT/CDF calibration diagnostics.
  - Include overall, season, NBM-LAMP disagreement, and HRRR-availability/coverage slices.
  - Eval: record PIT mean/std, tail rates, and any HRRR-specific failure slices.
  - Done 2026-04-29: PIT diagnostics include overall, season, NBM-LAMP disagreement, HRRR-LAMP disagreement, HRRR-NBM disagreement, and HRRR hotter/colder direction slices.

- [x] Add HRRR-specific ablation diagnostics.
  - Compare:
    - no-HRRR promoted model artifacts
    - with-HRRR model using all features
    - with-HRRR training with HRRR columns dropped
  - Eval: require same folds and same date coverage for fair comparison.
  - Done 2026-04-29: added `hrrr_ablation_diagnostics.py` and ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.hrrr_ablation_diagnostics`.
  - Eval 2026-04-29: rolling `729` validation rows. With HRRR features: event-bin NLL/Brier `1.515909/0.653115`, degree NLL/RPS `2.367784/0.011048`, q50 MAE/RMSE `1.4042/1.9819`. HRRR columns dropped: event-bin NLL/Brier `1.594013/0.666865`, degree NLL/RPS `2.469751/0.011270`, q50 MAE/RMSE `1.4261/2.0042`. HRRR has real out-of-sample value.

## Phase 4: Rolling-Origin Model Selection

- [x] Port rolling-origin model selection.
  - Use expanding yearly folds:
    - train through `2023-12-31`, validate `2024-01-01..2024-12-31`
    - train through `2024-12-31`, validate `2025-01-01..2025-12-31`
  - Eval: write fold metrics and selected candidate manifest.
  - Done 2026-04-29: ported `rolling_origin_model_select.py` into `experiments/withhrrr/withhrrr_model/`, removed the no-HRRR ban on `hrrr_` and `meta_hrrr_` features, and ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select`.
  - Eval 2026-04-29: output under `experiments/withhrrr/data/runtime/evaluation/model_selection/`; leakage check passed with `291` features, `105` HRRR features included. Selected candidate remained `very_regularized_min_leaf70_lgbm_350` by weighted event-bin NLL then degree-ladder NLL. Weighted metrics: event-bin NLL `1.508995`, degree-ladder NLL `2.362184`, q50 MAE/RMSE `1.4410/2.0163` across `729` rolling validation rows.

- [x] Reuse and retune the constrained LightGBM grid.
  - Start from no-HRRR candidates, including `very_regularized_min_leaf70_lgbm_350`.
  - Add only small, explicit HRRR-aware variants if needed after the baseline run.
  - Eval: select by event-bin NLL first, degree-ladder NLL second, q50 MAE/RMSE as secondary.
  - Progress 2026-04-29: reused the full no-HRRR candidate grid unchanged as the first HRRR-inclusive grid. No HRRR-specific candidates added yet.
  - Done 2026-04-29: added explicit HRRR-aware candidates `hrrr_shallow_stronger_l2_lgbm_350`, `hrrr_shallow_min_leaf45_lgbm_350`, and `hrrr_medium_min_leaf60_lgbm_300`; reran rolling selection over `19` candidates.
  - Eval 2026-04-29: selected candidate remained `regularized_shallow_lgbm_300` by weighted event-bin NLL then degree NLL. HRRR-specific variants were evaluated but did not beat the selected candidate.

- [x] Add leakage and stability checks.
  - Confirm selected features exclude labels, market bins, exact target date, raw timestamp code fields, and source-id code fields.
  - Confirm HRRR features are included in the selected feature set.
  - Eval: write leakage findings and fold-to-fold stability metrics.
  - Progress 2026-04-29: rolling-origin manifest recorded `leakage_check.status=ok`, `finding_count=0`; stability CSV written to `experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_stability.csv`.
  - Done 2026-04-29: rerun with HRRR disagreement features and HRRR-specific grid passed leakage checks with `304` selected features.

## Phase 5: Calibration And Distribution Defaults

- [x] Port rolling-origin quantile calibration.
  - Compare uncalibrated, global offsets, season offsets, NBM-LAMP disagreement offsets, and HRRR-coverage segments.
  - Eval: select only by out-of-time probability metrics.
  - Done 2026-04-29: ported `calibrate_quantiles.py` and `calibrate_rolling_origin.py`; ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin`.
  - Eval 2026-04-29: selected `season_offsets` by event-bin NLL then degree-ladder NLL using 2024 as calibration and 2025 as test. On 2025 test rows, uncalibrated event-bin NLL/Brier `1.412542/0.632488`; selected season offsets `1.291832/0.616072`. Degree-ladder NLL/Brier/RPS improved from `2.284547/0.846069/0.010629` to `2.132255/0.829765/0.010389`. Manifest: `experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json`.

- [x] Test conformalized interval calibration.
  - Reuse the no-HRRR method and compare against additive offsets.
  - Eval: include interval score and q05-q95/q10-q90/q25-q75 coverage.
  - Done 2026-04-29: `calibrate_rolling_origin.py` compares conformal intervals against global, season, NBM-LAMP disagreement, HRRR-LAMP disagreement/direction, HRRR-NBM disagreement/direction, and month offsets.
  - Eval 2026-04-29: selected `conformal_intervals`; 2025 event-bin NLL/Brier improved from uncalibrated `1.370969/0.622307` to `1.274655/0.615411`, with q05-q95 coverage `0.9479`.

- [x] Port distribution diagnostics.
  - Compare `interpolation_tail`, `interpolation_no_tail`, `smoothed_interpolation_tail`, and `normal_iqr`.
  - Eval: select by event-bin NLL and degree-ladder NLL on 2025 out-of-time validation.
  - Done 2026-04-29: ported `distribution_diagnostics.py` and ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.distribution_diagnostics`.
  - Eval 2026-04-29: selected `normal_iqr`; 2025 event-bin NLL/Brier `1.261539/0.616776`, degree NLL/RPS `2.024162/0.009516`, beating interpolation-tail event-bin NLL `1.274655`.

- [x] Keep monotone rearrangement as a safety guard.
  - Diagnose raw quantile crossings and repair outputs before ladder construction.
  - Eval: record crossing rates by fold and candidate.
  - Done 2026-04-29: prediction and evaluation continue applying monotone rearrangement before ladder construction; `distribution_diagnostics.py` writes `quantile_crossing_diagnostics.csv` and `quantile_crossing_candidate_summary.csv`.

- [x] Decide whether ladder reliability calibration should be promoted.
  - Revalidate using the selected HRRR-inclusive quantile calibration and distribution method.
  - Eval: compare ladder-calibrated vs selected quantile/distribution default.
  - Done 2026-04-29: ported `calibrate_ladder.py`, changed it to validate against selected `normal_iqr`, and wired `predict.py` to apply the selected ladder reliability manifest by default.
  - Eval 2026-04-29: selected `bucket_reliability_s1_00`; with conformal quantiles and `normal_iqr`, 2025 event-bin NLL/Brier improved `1.261539/0.616776 -> 1.259265/0.615309`, degree NLL/RPS improved `2.024162/0.009516 -> 2.016883/0.009493`.

## Phase 6: Prediction And Event Adapter

- [x] Port prediction CLI.
  - Load HRRR-inclusive feature manifest and quantile model artifacts.
  - Emit expected final high, final-Tmax quantiles, degree ladder, calibration metadata, and optional event-bin probabilities.
  - Eval: smoke on a historical target date with fixed event bins.
  - Done 2026-04-29: ported `predict.py` with with-HRRR defaults and automatic use of `experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json` when present.
  - Eval 2026-04-29: smoke command `.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict --target-date-local 2025-12-31 --event-bin '50F or below' --event-bin '51-55F' --event-bin '56F or higher'` wrote `experiments/withhrrr/data/runtime/predictions/prediction_KLGA_2025-12-31.json`, with calibrated expected final high `33.18F`.

- [x] Port Polymarket event-bin adapter.
  - Reuse no-HRRR event-bin parsing and mapping behavior.
  - Keep market prices/trading decisions outside the core weather model.
  - Eval: parse a known event slug and map probabilities.
  - Done 2026-04-29: ported `polymarket_event.py` and updated with-HRRR online inference to call the with-HRRR adapter.
  - Eval 2026-04-29: fetched `highest-temperature-in-nyc-on-april-11-2026`, wrote `experiments/withhrrr/data/runtime/polymarket/event_slug=highest-temperature-in-nyc-on-april-11-2026/event_bins.json`, then mapped it through `predict.py` for `2025-12-31`. Output used `normal_iqr` and ladder reliability.

- [x] Build HRRR-inclusive online inference path.
  - This may be heavier than no-HRRR because HRRR live/current-day inputs are required.
  - For the first version, support historical/local feature rows before adding live HRRR fetching.
  - Eval: clearly mark unavailable status when HRRR is missing.
  - Done 2026-04-29: `run_online_inference.py` already fetches WU, LAMP, NBM, and HRRR and now uses the with-HRRR Polymarket adapter. Earlier server smoke produced an OK manifest and prediction for `2026-04-29`.

## Phase 7: Documentation And Tests

- [x] Add README runbook.
  - Include exact commands for data staging, table build, normalization, training, evaluation, calibration, prediction, and tests.
  - Include comparison against no-HRRR metrics.
  - Done 2026-04-29: rewrote `experiments/withhrrr/README.md` with current defaults, metrics, artifact paths, and runbook commands.

- [x] Add tests.
  - Minimum tests:
    - residual formula
    - HRRR columns are allowed and selected
    - leakage columns are excluded
    - probability mass sums to 1
    - event-bin mapping is stable
    - calibration manifests are applied deterministically
  - Done 2026-04-29: added `experiments/withhrrr/tests/test_withhrrr_model.py`.
  - Eval 2026-04-29: `.venv/bin/python -m pytest -q experiments/withhrrr/tests` passed, `11` tests.

- [ ] Update this TODO after each completed phase.
  - Future agents must keep tick boxes, dates, commands, artifact paths, and eval notes current.

## Later: Paper-Trading Simulation

- [ ] Build paper-trading simulation after HRRR-inclusive calibrated probability quality is reviewed.
  - Join mapped weather probabilities to provided or fetched market prices.
  - Use Polymarket market ids, condition ids, and CLOB token ids from `event_bins.json` as join keys.
  - Include fees, spread, slippage, liquidity, and position sizing assumptions as explicit inputs.
  - Emit model probability, market-implied probability, edge after costs, and action/no-action status per bin.
  - Do not add live order placement.
