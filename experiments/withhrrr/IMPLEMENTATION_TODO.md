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

Status: partially done

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

Done 2026-04-29: ran rolling-origin calibration with `.venv/bin/python -m experiments.withhrrr.withhrrr_model.calibrate_rolling_origin`. Outputs:

```text
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_manifest.json
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibration_summary.csv
experiments/withhrrr/data/runtime/evaluation/calibration_selection/rolling_origin_calibrated_coverage.csv
```

Eval 2026-04-29: selected method is `season_offsets`. On the 2025 test split, event-bin NLL/Brier improved from uncalibrated `1.412542/0.632488` to `1.291832/0.616072`; degree-ladder NLL/Brier/RPS improved from `2.284547/0.846069/0.010629` to `2.132255/0.829765/0.010389`.

Done 2026-04-29: ported and smoke-tested prediction CLI. Command:

```bash
.venv/bin/python -m experiments.withhrrr.withhrrr_model.predict \
  --target-date-local 2025-12-31 \
  --event-bin '50F or below' \
  --event-bin '51-55F' \
  --event-bin '56F or higher'
```

Eval 2026-04-29: prediction artifact written to `experiments/withhrrr/data/runtime/predictions/prediction_KLGA_2025-12-31.json`; calibrated expected final high `33.18F`.

### 9. Tests And README

Status: not done

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

### 10. Keep TODO Current

Status: not done

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
