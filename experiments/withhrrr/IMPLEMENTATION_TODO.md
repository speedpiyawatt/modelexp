# With-HRRR Experiment Implementation TODO

This file is the working implementation plan for `experiments/withhrrr`.

Agents must keep this file current while working:

- Update task status as work progresses.
- Use `Status: done` only after implementation is complete and locally verified.
- Use `Status: not done` for pending work.
- Use `Status: blocked` only when the blocker is explicit and written down.
- Add a short note with date, command, or artifact path when marking a task done.
- Keep all with-HRRR experiment code, derived data, models, reports, and manifests under `experiments/withhrrr/`.
- Existing source data may be read from canonical locations, but do not write experiment outputs back into canonical tool folders unless explicitly building canonical source tables.
- HRRR is required as a real model input source for this experiment.
- Do not introduce a staged ensemble or separate HRRR-only main model unless the user explicitly asks for that later.

Long-running or download-heavy commands:

- Do not run long downloads or long backfills from the agent unless the user explicitly asks.
- Give the exact command to the user and wait for the user to say it is complete.
- After user confirmation, continue with validation, parsing, merging, training, or evaluation.
- Reasonable short local validation commands may be run by the agent.

## Source Inputs

Read-only source inputs expected:

- Wunderground history/tables: `wunderground/output/history` and `wunderground/output/tables/`
- NBM overnight data: canonical NBM overnight runtime output
- LAMP overnight data: canonical or experiment-local LAMP overnight summaries
- HRRR overnight data: completed HRRR overnight summary output from the optimized `wgrib2-bin` backfill

Known HRRR server coverage as of 2026-04-29:

- Source path: `/root/modelexp/data/runtime/hrrr_backfill_overnight/hrrr_summary_state/`
- Date coverage: `2023-01-01` through `2025-12-31`
- Days: `1,096`
- Manifest failures: `0`
- `extract_method`: `wgrib2-bin`
- `summary_profile`: `overnight`
- `selection_mode`: `overnight_0005`

Use `America/New_York` for the target local station day and the `00:05` local overnight cutoff.

## Target Layout

Status: done

Create this experiment-local layout as needed:

```text
experiments/withhrrr/
  IMPLEMENTATION_TODO.md
  TODO.md
  README.md
  requirements.txt
  withhrrr_model/
    __init__.py
    contracts.py
    prepare_training_features.py
    train_quantile_models.py
    evaluate.py
    distribution.py
    event_bins.py
    model_config.py
    calibrate_rolling_origin.py
    distribution_diagnostics.py
    predict.py
  data/runtime/
    source/
      hrrr_summary_state/
    training/
    audit/
    models/
    evaluation/
    predictions/
  tests/
```

Note 2026-04-29: created `experiments/withhrrr/withhrrr_model/` with `prepare_training_features.py`, `train_quantile_models.py`, and `model_config.py`. Runtime source/training/model folders were created through the first data-prep and training run. Later on 2026-04-29, canonical WU/NBM/LAMP/HRRR merged and normalized tables were rebuilt locally and the first model was retrained against that canonical input.

## TODO

### Source-Disagreement Robustness Layer

Status: done

Goal:

- Keep the base LightGBM residual quantile model, but add a top-layer evaluation path for source-conflict regimes.
- Do not promote a regime method unless rolling validation selects it.

Implemented 2026-04-29:

- Added shared source-disagreement regime logic under `withhrrr_model/source_disagreement.py`.
- Added `source_disagreement_regime_offsets` with hierarchical shrinkage to quantile calibration.
- Added source-disagreement ladder widening candidates to final ladder calibration.
- Added source-regime diagnostics to holdout evaluation and prediction JSON output.
- Updated server dual inference comparison to print the with-HRRR source regime and ladder adjustment.

Verification 2026-04-29:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin
.venv/bin/python -m experiments.withhrrr.withhrrr_model.distribution_diagnostics
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_ladder
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local
.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py
.venv/bin/python -m py_compile experiments/withhrrr/withhrrr_model/*.py tools/weather/run_server_dual_inference.py
```

Result:

- New quantile candidate `source_disagreement_regime_offsets` did not win rolling validation.
- New ladder widening candidates did not win rolling validation.
- Selected quantile calibration remains `hrrr_nbm_direction_offsets`.
- Selected ladder calibration remains `bucket_reliability_s1_00`.
- Diagnostics are available in `metrics_by_source_disagreement_regime.csv` and `ladder_calibration_disagreement_slices.csv`.

### Dual No-HRRR / With-HRRR Guard

Status: done

Goal:

- Keep raw no-HRRR and raw with-HRRR outputs visible, but add a validation-selected guarded recommendation for regimes where the with-HRRR source-trust model is historically risky.
- Use validation rows plus retained 2026 live backtest rows before promoting any guard.

Implemented 2026-04-30:

- Added `tools/weather/dual_guard.py` with shared guard weights and payload blending.
- Added `tools/weather/evaluate_dual_guard.py` to evaluate:
  - `always_no_hrrr`
  - `always_with_hrrr`
  - `with_hrrr_except_native_cold_hrrr_warm`
  - `with_hrrr_only_high_or_very_high_disagreement`
  - `probability_blend_by_regime`
  - `expected_tmax_blend_by_regime`
- Updated `tools/weather/run_server_dual_inference.py` to read `experiments/withhrrr/data/runtime/evaluation/dual_guard/dual_guard_manifest.json` and print a `guarded` recommendation with applied weights.

Verification 2026-04-30:

```bash
.venv/bin/python tools/weather/evaluate_dual_guard.py
.venv/bin/python -m py_compile tools/weather/dual_guard.py tools/weather/evaluate_dual_guard.py tools/weather/run_server_dual_inference.py experiments/withhrrr/withhrrr_model/source_disagreement.py experiments/withhrrr/withhrrr_model/predict.py
.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py
```

Result:

- Selected guard: `with_hrrr_except_native_cold_hrrr_warm`.
- Validation holdout, `219` rows: always with-HRRR event NLL/observed-bin probability/MAE `1.133178/0.385604/1.265688`; selected guard `1.137793/0.385989/1.250839`.
- 2026 live backtest, `86` successful rows: always with-HRRR event NLL/observed-bin probability/MAE/top-bin accuracy `1.589559/0.408022/2.085680/0.500`; selected guard `1.554194/0.411389/2.024112/0.500`.
- Artifact paths: `experiments/withhrrr/data/runtime/evaluation/dual_guard/dual_guard_manifest.json`, `dual_guard_summary.csv`, `dual_guard_slices.csv`, and `dual_guard_scored_rows.parquet`.
- Note: validation expected-Tmax metrics use q50 as a proxy because historical holdout prediction JSONs are not retained; 2026 live-backtest metrics use the retained `expected_final_tmax_f` from prediction JSONs.

### Source-Trust Model Upgrade

Status: done

Goal:

- Keep the base residual LightGBM quantile model family.
- Evaluate dynamic anchors, source-trust interaction features, weighted disagreement specialists, and an optional meta residual correction.
- Promote only if rolling-origin validation improves overall probability metrics or stays within `+0.005` overall event-bin NLL while improving high-disagreement event-bin NLL by at least `0.02`.

Implemented 2026-04-29:

- Added `withhrrr_model/source_trust.py` with richer source deltas, source ranks, WU-last-temp-vs-source deltas, median/trimmed anchors, fold-local ridge anchor helpers, and source-regime training weights.
- Extended `prepare_training_features.py` with `source_median_4way` and `source_trimmed_mean_4way` anchor policies and source-trust features.
- Extended `rolling_origin_model_select.py` so evaluated candidates are full specs: anchor policy, model candidate, feature profile, weight profile, and optional meta residual correction.
- Extended production training/inference so selected anchor/profile metadata is read from manifests. Ridge anchors and meta residual models require explicit saved artifacts when selected.

Verification 2026-04-29:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select --candidate-config-path /tmp/.../candidates.json --splits-path /tmp/.../splits.json --output-dir /tmp/.../out
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models --model-selection-manifest-path /tmp/.../out/rolling_origin_model_selection_manifest.json --output-dir /tmp/.../models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --models-dir /tmp/.../models --output-dir /tmp/.../eval
.venv/bin/python -m py_compile experiments/withhrrr/withhrrr_model/*.py tools/weather/run_server_dual_inference.py
.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py
```

Result:

- Smoke candidate used `source_median_4way` and completed rolling selection, production training, and holdout evaluation.
- Unit suite passed with `26` tests.
- Full default rolling-origin grid evaluated `32` candidate specs.
- Selected candidate: `very_regularized_min_leaf70_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted__weights=high_disagreement_weighted`.
- Selected model metadata: anchor `equal_3way`, feature profile `high_disagreement_weighted`, weight profile `high_disagreement_weighted`, no meta residual.
- Rolling selection metrics: event-bin NLL `1.462654`, degree-ladder NLL `2.325077`, q50 MAE `1.415372`.
- Refreshed selected downstream defaults: `conformal_intervals`, `normal_iqr`, `bucket_reliability_s1_00`.
- Refreshed holdout `2025-05-27..2025-12-31`: event-bin NLL/Brier `1.372809/0.619639`, degree NLL/RPS `2.149011/0.010002`, q50 MAE/RMSE `1.252238/1.651186`.
- Deployed to production server: code commit `20330a1` is pulled on `/root/modelexp` at `root@198.199.64.163`; ignored runtime artifacts were synced separately for `models`, `model_selection`, `calibration_selection`, `distribution_diagnostics`, and `ladder_calibration`. Server-side verification confirmed selected training spec `equal_3way`, feature profile `high_disagreement_weighted`, weight profile `high_disagreement_weighted`, no meta residual.

### Nearby-Station Source-Trust Upgrade

Status: done

Goal:

- Add nearby airport/harbor observation context without leaking past the `00:05 America/New_York` cutoff.
- Use Wunderground history for `KJRB`, `KJFK`, `KEWR`, and `KTEB`.
- Rerun the complete selection, calibration, ladder, final training, and holdout stack before promotion.

Implemented 2026-04-30:

- Added `withhrrr_model/nearby_observations.py`.
- Added optional nearby observation loading to `prepare_training_features.py`, including `--nearby-root`, repeatable `--nearby-station-id`, `--legacy-kjrb-obs-path`, `--disable-nearby-obs`, and `--nearby-max-obs-age-hours`.
- Added stale-observation protection: each nearby station feature block only uses observations at or before cutoff and not older than the configured max age.
- Added nearby feature profiles and focused nearby LightGBM candidates to `source_trust.py`, `model_config.py`, and `rolling_origin_model_select.py`.
- Updated `build_inference_features.py` and `run_online_inference.py` so production inference can fetch, build, pass, and clean up nearby Wunderground observation features instead of filling the selected nearby columns as all missing.
- Generalized Wunderground table building to station-specific file globs and added fetch-time HTTP skip handling for missing/no-data days.

Verification 2026-04-30:

```bash
.venv/bin/python wunderground/build_training_tables.py \
  --history-dir experiments/withhrrr/data/runtime/source/wunderground_kjrb/history \
  --station-id KJRB \
  --output-dir experiments/withhrrr/data/runtime/source/wunderground_kjrb
.venv/bin/python -m experiments.withhrrr.withhrrr_model.prepare_training_features
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin
.venv/bin/python -m experiments.withhrrr.withhrrr_model.distribution_diagnostics
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_ladder
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local
.venv/bin/python -m experiments.withhrrr.withhrrr_model.build_inference_features --target-date-local 2025-12-31 ...
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict --features-path /tmp/withhrrr_nearby_inference_smoke/.../withhrrr.inference_features_normalized.parquet ...
.venv/bin/python -m py_compile wunderground/*.py experiments/withhrrr/withhrrr_model/*.py
.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py
```

Result:

- Training table: `1,096` rows, `508` columns, `467` selected features, `106` nearby-derived columns in the preparation manifest.
- Coverage: `meta_nearby_kjrb_obs_available=1086`, `meta_nearby_kjfk_obs_available=1092`, `meta_nearby_kewr_obs_available=1092`, `meta_nearby_kteb_obs_available=1091`.
- Leakage/label checks: `0` cutoff violations found; `label_final_tmax_f` matched `final_tmax_f` with max absolute diff `0.0`.
- Rolling selection evaluated `47` candidate specs and selected `nearby_vreg_leaf100_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted_nearby__weights=high_disagreement_weighted`.
- Selected model metadata: anchor `equal_3way`, model candidate `nearby_vreg_leaf100_lgbm_350`, feature profile `high_disagreement_weighted_nearby`, weight profile `high_disagreement_weighted`, no meta residual.
- Rolling selection metrics: event-bin NLL `1.459584`, degree-ladder NLL `2.281087`, q50 MAE/RMSE `1.443267/1.996376`.
- Refreshed selected downstream defaults: `global_offsets`, `normal_iqr`, `bucket_reliability_s1_00`.
- Refreshed rolling 2025 ladder metrics: event-bin NLL/Brier `1.244989/0.608950`, degree NLL/RPS `2.000670/0.009480`.
- Calibration robustness update 2026-04-30: added candidate comparison for `global_offsets`, `conformal_intervals`, `no_offsets`, `global_offsets_no_upper_tail`, `global_offsets_shrunk_50pct`, and conditional source-disagreement methods. Promotion now rejects candidates that reduce observed-bin probability by more than `0.02` in `tight_consensus` or `moderate_disagreement`; selected quantile calibration changed to `global_offsets_no_upper_tail`. Refreshed rolling 2025 ladder metrics: event-bin NLL/Brier `1.241810/0.606852`, degree NLL/RPS `2.000715/0.009475`.
- Calibration implementation note: `global_offsets_no_upper_tail` applies the fitted q05/q10/q25/q50 offsets and explicitly sets q75/q90/q95 offsets to `0.0`; prediction output includes both the legacy `calibration_offsets_f` and structured `calibration` metadata so future conditional methods can expose the selected branch.
- Refreshed production-calibrated holdout `2025-05-27..2025-12-31`: event-bin NLL/Brier `1.133178/0.603449`, degree NLL/RPS `1.861432/0.009238`, raw q50 MAE/RMSE `1.261223/1.639528`, scored q50 MAE/RMSE `1.265688/1.642842`.
- Local inference smoke for `2025-12-31` produced an inference row with all four nearby stations available, `nearby_feature_count=106`, `feature_count=467`, and a valid prediction JSON.
- Unit suite passed with `27` tests.

Review fixes 2026-04-30:

- Fixed `tools/weather/run_server_dual_inference.py` so the remote with-HRRR side calls `experiments.withhrrr.withhrrr_model.run_online_inference` with reused KLGA WU/LAMP/NBM/HRRR artifacts. This lets the wrapper fetch/build nearby Wunderground station features instead of calling `build_inference_features.py` without nearby inputs.
- Fixed `source_trust.py` so non-nearby profiles exclude all nearby-derived columns, including `klga_minus_nearby_*` and `*_vs_nearby_*`.
- Fixed `build_inference_features.py` so a selected nearby feature profile requires at least one available nearby station and generated nearby columns; it no longer silently predicts with all nearby features missing.
- Reran rolling-origin model selection after the feature-filter fix. The selected candidate remained `nearby_vreg_leaf100_lgbm_350__anchor=equal_3way__features=high_disagreement_weighted_nearby__weights=high_disagreement_weighted`.
- Reran calibration, distribution diagnostics, ladder calibration, final training, and holdout evaluation. After fixing the evaluator to score the production calibrated stack, final holdout event-bin NLL/Brier is `1.133178/0.603449`; raw q50 MAE remains `1.261223`.
- Evaluator behavior: `withhrrr_model.evaluate` now loads selected calibration/distribution/ladder manifests by default. `validation_predictions.parquet` is the production-calibrated scoring frame; `validation_predictions_raw.parquet` preserves raw model quantiles for debugging.
- Verification: py_compile passed; `.venv/bin/python -m pytest experiments/withhrrr/tests/test_withhrrr_model.py` passed with `30` tests; all-nearby inference smoke passed; no-nearby inference smoke failed explicitly as intended.
- Deploy status: code was pushed to GitHub `origin/main` and synced to the DigitalOcean server with ignored runtime artifacts. Later code commits or runtime artifact rebuilds still need explicit server sync before server dual inference is current.

### 1. Stage HRRR Overnight Summary Data

Status: done

Goal:

- Make the completed HRRR overnight summary data available locally or in the run environment.
- Keep a copy under `experiments/withhrrr/data/runtime/source/hrrr_summary_state/` unless using a canonical runtime source path for a full build.

Suggested copy command from local machine:

```bash
rsync -avz root@198.199.64.163:/root/modelexp/data/runtime/hrrr_backfill_overnight/hrrr_summary_state/ \
  experiments/withhrrr/data/runtime/source/hrrr_summary_state/
```

Validation after copy:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import json
root = Path("experiments/withhrrr/data/runtime/source/hrrr_summary_state")
days = sorted(p for p in root.glob("target_date_local=*") if p.is_dir())
print("days", len(days), days[0].name if days else None, days[-1].name if days else None)
failures = []
methods = {}
profiles = {}
for day in days:
    manifest_path = day / "hrrr.manifest.json"
    if not manifest_path.exists():
        failures.append((day.name, "missing_manifest"))
        continue
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("failed", 0) not in (0, None):
        failures.append((day.name, manifest.get("failed")))
    methods[manifest.get("extract_method")] = methods.get(manifest.get("extract_method"), 0) + 1
    profiles[manifest.get("summary_profile")] = profiles.get(manifest.get("summary_profile"), 0) + 1
print("failures", len(failures), failures[:10])
print("extract_methods", methods)
print("summary_profiles", profiles)
PY
```

Expected:

- `days 1096 target_date_local=2023-01-01 target_date_local=2025-12-31`
- `failures 0`
- `extract_methods {'wgrib2-bin': 1096}`
- `summary_profiles {'overnight': 1096}`

Note 2026-04-29: copied both HRRR state manifests and actual HRRR summary parquet rows from `root@198.199.64.163`. Validated `1,096` state days covering `2023-01-01` through `2025-12-31`, `0` manifest failures, `extract_method={'wgrib2-bin': 1096}`, `summary_profile={'overnight': 1096}`, and `selection_mode={'overnight_0005': 1096}`. Actual summary output under `experiments/withhrrr/data/runtime/source/hrrr_summary/` has `1,096` `hrrr.overnight.parquet` files and `1,096` unique dates.

### 2. Build Canonical Merged Training Features With HRRR

Status: done

Goal:

- Build an HRRR-inclusive `training_features_overnight` table using the canonical weather builder.
- Write experiment-local outputs.

Command template:

```bash
.venv/bin/python tools/weather/build_training_features_overnight.py \
  --labels-path wunderground/output/tables/labels_daily.parquet \
  --obs-path wunderground/output/tables/wu_obs_intraday.parquet \
  --nbm-root data/runtime/backfill_overnight/nbm_overnight \
  --lamp-root experiments/no_hrrr_model/data/runtime/lamp_overnight \
  --hrrr-root experiments/withhrrr/data/runtime/source/hrrr_summary \
  --start-local-date 2023-01-01 \
  --end-local-date 2025-12-31 \
  --output-dir experiments/withhrrr/data/runtime/training/training_features_overnight \
  --manifest-output-path experiments/withhrrr/data/runtime/training/training_features_overnight.manifest.json
```

Validation:

- Run `tools/weather/audit_training_features_overnight.py` on the combined output if needed.
- Confirm `meta_hrrr_available=True` for expected rows.
- Confirm HRRR columns such as `hrrr_temp_2m_day_max_k` are present.

Note 2026-04-29: resolved the local source-root blocker by pulling NBM from Lightning path `/teamspace/studios/this_studio/modelexp/data/runtime/backfill_overnight/nbm_overnight/` into `data/runtime/backfill_overnight/nbm_overnight/` and LAMP from Lightning path `/teamspace/studios/this_studio/modelexp/experiments/no_hrrr_model/data/runtime/lamp_overnight/` into `experiments/no_hrrr_model/data/runtime/lamp_overnight/`. Rebuilt Wunderground tables with `.venv/bin/python wunderground/build_training_tables.py --history-dir wunderground/output/history --output-dir wunderground/output/tables`.

Done 2026-04-29: ran the command above with `--hrrr-root experiments/withhrrr/data/runtime/source/hrrr_summary`. Combined daily parts into `experiments/withhrrr/data/runtime/training/training_features_overnight.parquet`.

Eval 2026-04-29: merged output has `1,096` rows, `316` columns, date range `2023-01-01` through `2025-12-31`, `1,096` HRRR-available rows, and `105` HRRR columns.

### 3. Build Canonical Normalized Training Features

Status: done

Goal:

- Normalize the canonical merged table into model-ready units.
- Keep outputs under `experiments/withhrrr/data/runtime/training/`.

Command template:

```bash
.venv/bin/python tools/weather/build_training_features_overnight_normalized.py \
  --input-root experiments/withhrrr/data/runtime/training/training_features_overnight \
  --output-dir experiments/withhrrr/data/runtime/training/training_features_overnight_normalized \
  --manifest-output-path experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.manifest.json \
  --start-local-date 2023-01-01 \
  --end-local-date 2025-12-31
```

Then optionally combine daily parquet outputs into:

```text
experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_normalized.parquet
```

Validation:

- Count rows and dates.
- Count columns by prefix.
- Confirm `hrrr_` normalized columns are present and numeric.

Done 2026-04-29: ran the command above and combined daily parts into `experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.parquet`.

Eval 2026-04-29: normalized output has `1,096` rows, `316` columns, date range `2023-01-01` through `2025-12-31`, `1,096` HRRR-available rows, and `105` HRRR columns.

### 4. Prepare With-HRRR Model Training Table

Status: done

Goal:

- Add no-HRRR-style model target columns to the canonical normalized table.
- Keep HRRR features as model inputs.

Required derived columns:

```text
nbm_tmax_open_f
lamp_tmax_open_f
anchor_tmax_f
nbm_minus_lamp_tmax_f
hrrr_tmax_open_f
hrrr_minus_lamp_tmax_f
hrrr_minus_nbm_tmax_f
abs_hrrr_minus_lamp_tmax_f
abs_hrrr_minus_nbm_tmax_f
anchor_equal_3way_tmax_f
hrrr_outside_nbm_lamp_range_f
target_residual_f
model_training_eligible
```

Initial formulas:

```text
anchor_tmax_f = 0.5 * nbm_tmax_open_f + 0.5 * lamp_tmax_open_f
target_residual_f = label_final_tmax_f - anchor_tmax_f
model_training_eligible = finite(label_final_tmax_f, anchor_tmax_f) and meta_nbm_available and meta_lamp_available and meta_hrrr_available
```

Important:

- Do not make HRRR the anchor in the initial implementation.
- HRRR must be included as model input features so the model can learn from HRRR directly.

Expected output:

```text
experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet
experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.manifest.json
```

Done 2026-04-29: updated `prepare_training_features.py` so the default base input is `experiments/withhrrr/data/runtime/training/training_features_overnight_normalized.parquet`; it derives no-HRRR-style residual target columns from canonical normalized NBM/LAMP columns and only joins raw HRRR summaries when the base table does not already contain HRRR columns.

Eval 2026-04-29: prepared model table has `1,096` rows, `324` columns, `1,094` eligible rows, `1,096` HRRR-available rows, and `105` HRRR columns. Output manifest: `experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.manifest.json`.

Optimization 2026-04-29: added HRRR disagreement and direction features to `prepare_training_features.py`: `hrrr_tmax_open_f`, HRRR-vs-LAMP/NBM deltas, absolute deltas, equal 3-way anchor diagnostic, HRRR-outside-NBM/LAMP-range diagnostics, and 3F hotter/colder boolean flags.

Eval 2026-04-29: rebuilt prepared model table. Output now has `1,096` rows, `337` columns, `1,094` eligible rows, and `304` selected model features after retraining.

### 5. Port Train/Evaluate/Prediction Modules

Status: partially done

Goal:

- Reuse the no-HRRR implementation pattern with HRRR-enabled feature selection.
- Keep model behavior familiar and comparable.

Port/adapt from `experiments/no_hrrr_model/no_hrrr_model/`:

- `model_config.py`
- `train_quantile_models.py`
- `evaluate.py`
- `distribution.py`
- `event_bins.py`
- `predict.py`
- `calibrate_rolling_origin.py`
- `distribution_diagnostics.py`
- `ensemble_diagnostics.py`

Feature selection rules:

- Exclude `label_` columns.
- Exclude market/bin target fields.
- Exclude exact target-date fields and raw timestamp-code fields.
- Exclude source identifier code fields.
- Include numeric/bool `wu_`, `nbm_`, `lamp_`, `hrrr_`, and useful `meta_` availability/coverage fields.
- Report feature counts by prefix in `feature_manifest.json`.

Note 2026-04-29: added initial `model_config.py` and `train_quantile_models.py`. The trainer reuses no-HRRR model candidates and selected default `very_regularized_min_leaf70_lgbm_350`, but allows numeric/bool HRRR and HRRR metadata features while keeping label/date/market/source-code leakage exclusions.

Eval 2026-04-29: canonical rerun selected `291` features, including `105` `hrrr_` features. Prefix counts reported by `feature_manifest.json`: `hrrr_=105`, `lamp_=82`, `nbm_=59`, `wu_=23`, `meta_=21`.

Note 2026-04-29: ported `evaluate.py`, `distribution.py`, `event_bins.py`, `calibrate_quantiles.py`, `calibrate_rolling_origin.py`, `rolling_origin_model_select.py`, and `predict.py` into `experiments/withhrrr/withhrrr_model/`. The rolling-origin selector was adjusted so HRRR feature prefixes are allowed.

Optimization 2026-04-29: after rolling-origin selection with HRRR disagreement features, promoted the with-HRRR default candidate to `regularized_shallow_lgbm_300` in `model_config.py`. This is now intentionally different from the no-HRRR default.

### 6. Train First With-HRRR Quantile Model

Status: done

Goal:

- Train the same residual quantile LightGBM family as no-HRRR, now with HRRR features included.

Initial command shape:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models \
  --input-path experiments/withhrrr/data/runtime/training/training_features_overnight_withhrrr_model.parquet \
  --output-dir experiments/withhrrr/data/runtime/models
```

Expected outputs:

```text
experiments/withhrrr/data/runtime/models/
  residual_quantile_q05.txt
  residual_quantile_q10.txt
  residual_quantile_q25.txt
  residual_quantile_q50.txt
  residual_quantile_q75.txt
  residual_quantile_q90.txt
  residual_quantile_q95.txt
  feature_manifest.json
  training_manifest.json
```

Note 2026-04-29: ran `.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models` against the canonical prepared table. Outputs written under `experiments/withhrrr/data/runtime/models/`. The selected feature set has `291` features, including `105` `hrrr_` features. Prefix counts: `hrrr_=105`, `lamp_=82`, `nbm_=59`, `wu_=23`, `meta_=21`. Single chronological validation starts `2025-05-27` with `219` rows. Validation pinball losses: q05 `0.173021`, q10 `0.290295`, q25 `0.503113`, q50 `0.636131`, q75 `0.497551`, q90 `0.279705`, q95 `0.170042`.

Optimization 2026-04-29: retrained production artifacts with selected candidate `regularized_shallow_lgbm_300` after adding HRRR disagreement features. Command:

```bash
rm -rf experiments/withhrrr/data/runtime/models
.venv/bin/python -m experiments.withhrrr.withhrrr_model.train_quantile_models
```

Eval 2026-04-29: selected feature count `304`; validation pinball losses q05 `0.172547`, q10 `0.294406`, q25 `0.493671`, q50 `0.620629`, q75 `0.502329`, q90 `0.284522`, q95 `0.165234`.

### 7. Evaluate Against No-HRRR

Status: partially done

Goal:

- Compare the with-HRRR model against the promoted no-HRRR stack on identical dates.
- Do not compare mismatched validation windows.

Required comparisons:

- no-HRRR promoted model
- with-HRRR model using all selected features
- with-HRRR model with HRRR columns dropped as an ablation

Required metrics:

- event-bin NLL and Brier
- degree-ladder NLL, Brier, and RPS
- observed-temp and observed-bin probability
- q50 MAE/RMSE
- q05-q95, q10-q90, and q25-q75 coverage
- PIT diagnostics

Note 2026-04-29: completed a quick q50 sanity comparison on the same `2025-05-27..2025-12-31` validation slice. Canonical with-HRRR q50 MAE/RMSE: `1.2723/1.6497`. Existing no-HRRR promoted artifact q50 MAE/RMSE on the same rows: `1.3063/1.6855`. Quick comparison artifact: `experiments/withhrrr/data/runtime/evaluation/quick_holdout_comparison.json`.

Progress 2026-04-29: reused `experiments.no_hrrr_model.no_hrrr_model.evaluate` with explicit with-HRRR paths to produce probability-first holdout outputs under `experiments/withhrrr/data/runtime/evaluation/full_holdout/`, and ran the same evaluator for the no-HRRR reference under `experiments/withhrrr/data/runtime/evaluation/no_hrrr_reference_holdout/`.

Eval 2026-04-29: with-HRRR validation metrics on `2025-05-27..2025-12-31`: degree-ladder NLL/Brier/RPS `2.147962/0.822671/0.010102`, event-bin NLL/Brier `1.344095/0.620545`, q50 MAE/RMSE `1.2723/1.6497`. No-HRRR reference on the same slice: degree-ladder NLL/Brier/RPS `2.141472/0.833441/0.010164`, event-bin NLL/Brier `1.320356/0.630038`, q50 MAE/RMSE `1.3063/1.6855`. HRRR improves Brier and q50 error in this uncalibrated holdout, but not NLL yet.

Done 2026-04-29: verified the local with-HRRR evaluator with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.evaluate --output-dir experiments/withhrrr/data/runtime/evaluation/full_holdout_local`.

Optimization 2026-04-29: evaluator now reports HRRR anchor diagnostics and HRRR disagreement slices:

```text
metrics_by_hrrr_lamp_disagreement.csv
metrics_by_hrrr_nbm_disagreement.csv
metrics_by_hrrr_lamp_direction.csv
metrics_by_hrrr_nbm_direction.csv
```

Eval 2026-04-29: after retraining the selected candidate with HRRR disagreement features, validation `2025-05-27..2025-12-31` produced q50 MAE/RMSE `1.2413/1.6092`, degree-ladder NLL/Brier/RPS `2.114456/0.824174/0.010029`, and event-bin NLL/Brier `1.321593/0.618214`. Baseline q50 MAE/RMSE on the same slice: NBM-only `1.5735/2.0401`, LAMP-only `1.3744/1.8245`, HRRR-only `1.9757/2.5496`, 50/50 anchor `1.3734/1.7833`, equal 3-way anchor `1.3726/1.7806`, linear NBM/LAMP/HRRR blend `1.3990/1.7922`.

### 8. Rolling-Origin Selection And Calibration

Status: partially done

Goal:

- Port the no-HRRR probability-first optimization stack.
- Re-run it with HRRR features included.

Steps:

- Rolling-origin model selection.
- Constrained LightGBM tuning.
- Quantile calibration selection.
- Conformal interval comparison.
- Distribution method comparison.
- Optional ladder reliability calibration.
- Prediction default selection.

Evaluation outputs should mirror no-HRRR paths under:

```text
experiments/withhrrr/data/runtime/evaluation/
```

Done 2026-04-29: ran rolling-origin model selection with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select`. Outputs:

```text
experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_fold_metrics.csv
experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_summary.csv
experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_predictions.parquet
experiments/withhrrr/data/runtime/evaluation/model_selection/rolling_origin_model_selection_manifest.json
```

Eval 2026-04-29: selected candidate is `very_regularized_min_leaf70_lgbm_350`; leakage check passed with `291` selected features and HRRR included. Weighted rolling metrics across `729` validation rows: event-bin NLL `1.508995`, degree-ladder NLL `2.362184`, q50 MAE/RMSE `1.4410/2.0163`.

Optimization 2026-04-29: reran rolling-origin selection after adding HRRR disagreement features and promoting HRRR diagnostics. Command:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/model_selection
.venv/bin/python -m experiments.withhrrr.withhrrr_model.rolling_origin_model_select
```

Eval 2026-04-29: selected/default candidate is now `regularized_shallow_lgbm_300`; feature count `304`. Outputs refreshed under `experiments/withhrrr/data/runtime/evaluation/model_selection/`.

Optimization 2026-04-29: added HRRR-specific candidates to `model_config.py` and reran rolling-origin selection over `19` candidates. Added:

```text
hrrr_shallow_stronger_l2_lgbm_350
hrrr_shallow_min_leaf45_lgbm_350
hrrr_medium_min_leaf60_lgbm_300
```

Eval 2026-04-29: selected/default candidate remained `regularized_shallow_lgbm_300`; the HRRR-specific variants were evaluated but did not improve weighted event-bin NLL.

Optimization 2026-04-29: added and ran `hrrr_ablation_diagnostics.py`.

Eval 2026-04-29: rolling `729` validation rows. With HRRR features: event-bin NLL/Brier `1.515909/0.653115`, degree NLL/RPS `2.367784/0.011048`, q50 MAE/RMSE `1.4042/1.9819`. HRRR columns dropped: event-bin NLL/Brier `1.594013/0.666865`, degree NLL/RPS `2.469751/0.011270`, q50 MAE/RMSE `1.4261/2.0042`.

Done 2026-04-29: ran rolling-origin calibration with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin`. Outputs:

```text
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_summary.csv
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibrated_coverage.csv
```

Eval 2026-04-29: selected method is `season_offsets`. On the 2025 test split, event-bin NLL/Brier improved from uncalibrated `1.412542/0.632488` to `1.291832/0.616072`; degree-ladder NLL/Brier/RPS improved from `2.284547/0.846069/0.010629` to `2.132255/0.829765/0.010389`.

Optimization 2026-04-29: added HRRR-LAMP disagreement, HRRR-NBM disagreement, HRRR-LAMP direction, and HRRR-NBM direction offset candidates to `calibrate_rolling_origin.py`; updated `predict.py` so online inference can apply those segment names if selected. Reran calibration selection:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/calibration_selection
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin
```

Eval 2026-04-29: selected method is `conformal_intervals`. On the 2025 test split, event-bin NLL/Brier improved from uncalibrated `1.370969/0.622307` to `1.274655/0.615411`; degree-ladder NLL/Brier/RPS improved from `2.233741/0.836850/0.010356` to `2.071875/0.827982/0.010205`. HRRR-aware segmented offsets were evaluated but did not beat conformal intervals by event-bin NLL.

Optimization 2026-04-29: ported `distribution_diagnostics.py` and `calibrate_ladder.py`, then selected the final distribution and ladder defaults.

Commands:

```bash
rm -rf experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics
.venv/bin/python -m experiments.withhrrr.withhrrr_model.distribution_diagnostics

rm -rf experiments/withhrrr/data/runtime/evaluation/ladder_calibration
.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_ladder
```

Eval 2026-04-29: distribution selection chose `normal_iqr`; 2025 event-bin NLL/Brier `1.261539/0.616776`, degree NLL/RPS `2.024162/0.009516`, beating interpolation-tail event-bin NLL `1.274655`. Ladder reliability was revalidated using selected `normal_iqr` and selected `bucket_reliability_s1_00`; event-bin NLL/Brier improved to `1.259265/0.615309`, degree NLL/RPS improved to `2.016883/0.009493`.

Optimization 2026-04-29: wired `predict.py` to auto-load and apply:

```text
experiments/withhrrr/data/runtime/evaluation/distribution_diagnostics/distribution_diagnostics_manifest.json
experiments/withhrrr/data/runtime/evaluation/ladder_calibration/ladder_calibration_manifest.json
```

when present.

Done 2026-04-29: ported and smoke-tested prediction CLI. Command:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bin '50F or below' \
  --event-bin '51-55F' \
  --event-bin '56F or higher'
```

Eval 2026-04-29: prediction artifact written to `experiments/withhrrr/data/runtime/predictions/prediction_KLGA_2025-12-31.json`; calibrated expected final high `33.18F`.

Optimization smoke 2026-04-29: after the HRRR disagreement/calibration pass, ran:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bin '30F or below' \
  --event-bin '31-35F' \
  --event-bin '36F or higher' \
  --output-dir experiments/withhrrr/data/runtime/predictions/smoke_optimized
```

Eval 2026-04-29: prediction succeeded; expected final high `33.08F`, q05/q10/q25/q50/q75/q90/q95 `29.67/31.14/31.90/32.96/34.09/35.64/36.91`, event-bin probabilities `0.0783/0.8082/0.1135`.

Optimization 2026-04-29: ported `polymarket_event.py`, updated `run_online_inference.py` to call the with-HRRR adapter, and smoke-tested a real event:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.polymarket_event \
  --event-slug highest-temperature-in-nyc-on-april-11-2026

.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bins-path experiments/withhrrr/data/runtime/polymarket/event_slug=highest-temperature-in-nyc-on-april-11-2026/event_bins.json \
  --output-dir experiments/withhrrr/data/runtime/predictions/polymarket_smoke
```

Eval 2026-04-29: Polymarket fetch wrote `event_bins.json` with bins `59F or below` through `78F or higher`; prediction used `distribution_method=normal_iqr` and ladder calibration `bucket_reliability_s1_00`.

### 9. Tests And README

Status: done

Goal:

- Add a runbook and regression tests before using the model operationally.

Minimum tests:

- HRRR columns are allowed and selected.
- Leakage fields are excluded.
- Residual formula is correct.
- Probability ladder sums to `1.0`.
- Event-bin mapping is stable.
- Calibration manifest is applied deterministically.

Preferred command:

```bash
.venv/bin/python -m pytest -q experiments/withhrrr/tests
```

Done 2026-04-29: rewrote `README.md` with current defaults, metrics, command runbook, and artifact paths.

Done 2026-04-29: added focused tests in `experiments/withhrrr/tests/test_withhrrr_model.py`.

Eval 2026-04-29: `.venv/bin/python -m pytest -q experiments/withhrrr/tests` passed, `11` tests.

### 10. Keep TODO Current

Status: done

Future agents must update:

- `experiments/withhrrr/IMPLEMENTATION_TODO.md` while implementing.
- `experiments/withhrrr/TODO.md` after each optimization step.

Every completed item needs:

- status update or checkbox tick
- completion date
- command(s)
- artifact path(s)
- short result note
- `Eval:` note for model-quality tasks
