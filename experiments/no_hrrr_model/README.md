# No-HRRR KLGA Overnight Experiment

This experiment builds a separate no-HRRR overnight fair-value workflow for KLGA daily high settlement. It keeps the same Wunderground settlement target and `00:05 America/New_York` cutoff as the canonical model, but excludes all HRRR inputs and schema fields.

## Boundary

- Predict finalized Wunderground KLGA daily high first.
- Use NBM, LAMP, and optional pre-cutoff Wunderground observations.
- Keep all generated no-HRRR outputs under `experiments/no_hrrr_model/data/runtime/`.
- Do not promote no-HRRR design choices into the root model unless explicitly requested.

## Source Inputs

- Wunderground history: `wunderground/output/history`
- NBM overnight rows: `data/runtime/backfill_overnight/nbm_overnight`
- LAMP issue-level features: `tools/lamp/data/runtime/features_full/station_id=KLGA`

Current experiment-local artifacts as of 2026-04-25:

- Wunderground tables: `1,197` label rows and `33,123` intraday observation rows.
- LAMP overnight summaries: `1,096` daily files covering `2023-01-01` through `2025-12-31`.
- NBM overnight source rows: `1,096` daily files covering `2023-01-01` through `2025-12-31`.
- Training table: `1,096` rows, `202` columns, `1,094` rows with `model_training_eligible=True`.
- Normalized table: `1,096` rows, `274` columns, `41` non-label categorical vocabularies.
- Model artifacts: seven LightGBM residual quantile models under `experiments/no_hrrr_model/data/runtime/models/`; current feature manifest has `180` model features after excluding label, source, market, target-date, and exact-time code fields.
- Evaluation artifacts: validation reports under `experiments/no_hrrr_model/data/runtime/evaluation/`, including holdout, baseline, quantile-crossing, rolling-origin, and preliminary calibration diagnostics.
- Prediction adapter: `experiments.no_hrrr_model.no_hrrr_model.predict`, writing JSON outputs under `experiments/no_hrrr_model/data/runtime/predictions/`.
- Polymarket event-bin adapter: `experiments.no_hrrr_model.no_hrrr_model.polymarket_event`, writing fetched Gamma API metadata and parsed event bins under `experiments/no_hrrr_model/data/runtime/polymarket/`.
- Ineligible training dates: `2024-11-05` and `2024-11-06`, both due to missing `lamp_tmax_open_f`.

## Workflow

Build Wunderground settlement labels and intraday observation tables:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.build_wu_tables \
  --history-dir wunderground/output/history \
  --output-dir experiments/no_hrrr_model/data/runtime/wunderground
```

Build LAMP overnight summaries. This can be slow over the full history, so run it locally when needed:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.build_lamp_overnight \
  --features-root tools/lamp/data/runtime/features_full/station_id=KLGA \
  --output-dir experiments/no_hrrr_model/data/runtime/lamp_overnight \
  --start-local-date 2023-01-01 \
  --end-local-date 2025-12-31
```

Build the no-HRRR training table:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.build_training_features \
  --labels-path experiments/no_hrrr_model/data/runtime/wunderground/labels_daily.parquet \
  --obs-path experiments/no_hrrr_model/data/runtime/wunderground/wu_obs_intraday.parquet \
  --nbm-root data/runtime/backfill_overnight/nbm_overnight \
  --lamp-root experiments/no_hrrr_model/data/runtime/lamp_overnight \
  --start-local-date 2023-01-01 \
  --end-local-date 2025-12-31 \
  --output-path experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet \
  --manifest-output-path experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.manifest.json
```

Rows with missing NBM or LAMP anchors are retained for coverage diagnostics, but they are marked `model_training_eligible=False`. Residual quantile training should use only rows where `model_training_eligible=True`.

Normalize model-ready features:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.normalize_features \
  --input-path experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr.parquet \
  --output-path experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.parquet \
  --manifest-output-path experiments/no_hrrr_model/data/runtime/training/training_features_overnight_no_hrrr_normalized.manifest.json
```

The normalized manifest includes categorical vocabularies for non-label categorical fields so inference can reuse the same encodings. Label-derived categorical fields are not encoded as model features.

Train, evaluation, and prediction entrypoints write under:

- `experiments/no_hrrr_model/data/runtime/models/`
- `experiments/no_hrrr_model/data/runtime/evaluation/`
- `experiments/no_hrrr_model/data/runtime/predictions/`

Install experiment-only training dependencies when ready:

```bash
uv pip install --python .venv/bin/python -r experiments/no_hrrr_model/requirements.txt
```

Train residual quantile models:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.train_quantile_models
```

The training entrypoint trains LightGBM residual quantile models using `target_residual_f` on rows where `model_training_eligible=True`. It uses a chronological validation split and writes model artifacts plus manifests under `experiments/no_hrrr_model/data/runtime/models/`.

Evaluate model and baseline artifacts:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.evaluate
```

Current validation slice: `2025-05-27` through `2025-12-31`, `219` rows.

Current single holdout validation metrics after feature-leakage cleanup:

- Residual quantile q50 final-Tmax MAE/RMSE: `1.2579°F` / `1.6153°F`.
- Fixed 50/50 anchor MAE/RMSE: `1.3734°F` / `1.7833°F`.
- NBM-only MAE/RMSE: `1.5735°F` / `2.0401°F`.
- LAMP-only MAE/RMSE: `1.3744°F` / `1.8245°F`.
- q05-q95 interval coverage: `0.8904`, mean width `4.8870°F`.

Rolling-origin validation:

- Train through `2023-12-31`, validate `2024-01-01` through `2024-12-31`: residual q50 MAE/RMSE `1.4812°F` / `2.0691°F`; fixed anchor MAE/RMSE `1.5904°F` / `2.1703°F`; q05-q95 coverage `0.7225`.
- Train through `2024-12-31`, validate `2025-01-01` through `2025-12-31`: residual q50 MAE/RMSE `1.3720°F` / `1.9435°F`; fixed anchor MAE/RMSE `1.5987°F` / `2.1825°F`; q05-q95 coverage `0.7452`.
- Interpretation: the residual q50 improves point error over the 50/50 anchor in both yearly splits, but interval coverage is under-calibrated and needs calibration before decision-ready fair values.

Fit preliminary empirical quantile offsets from validation predictions:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.calibrate_quantiles
```

Current validation-slice calibration diagnostics:

- Uncalibrated q05-q95 coverage/width: `0.8904` / `4.8870°F`.
- Calibrated q05-q95 coverage/width: `0.8995` / `4.9693°F`.
- Uncalibrated q10-q90 coverage/width: `0.7671` / `4.0199°F`.
- Calibrated q10-q90 coverage/width: `0.7991` / `4.1586°F`.
- Uncalibrated q25-q75 coverage/width: `0.4155` / `1.6127°F`.
- Calibrated q25-q75 coverage/width: `0.4977` / `1.9600°F`.
- This is a first validation-offset calibration pass, not enough by itself for live trading. Review out-of-sample calibration before paper trading.

Out-of-sample rolling calibration:

- Fit offsets on the 2024 rolling holdout and test on the 2025 rolling holdout.
- q05-q95 coverage: `0.7452` uncalibrated -> `0.8932` calibrated.
- q10-q90 coverage: `0.6082` uncalibrated -> `0.8466` calibrated.
- q25-q75 coverage: `0.3589` uncalibrated -> `0.5726` calibrated.
- q05-q95 interval score: `9.3269` uncalibrated -> `8.2491` calibrated.
- q50 MAE: `1.3720°F` uncalibrated -> `1.3746°F` calibrated.

Generate a probability forecast from a normalized feature row:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.predict \
  --target-date-local YYYY-MM-DD \
  --event-bin "80 or below" \
  --event-bin "81-84" \
  --event-bin "85+"
```

The prediction output includes residual quantiles, final-Tmax quantiles, expected final high, a 1°F internal probability ladder, and optional mapped event-bin probabilities. It does not include market prices or trading decisions.

If `experiments/no_hrrr_model/data/runtime/evaluation/rolling_origin_calibration_manifest.json` exists, `predict.py` applies those out-of-sample-tested quantile offsets by default. Use `--calibration-path PATH` to choose a different calibration manifest.

Build a normalized inference feature row from local source artifacts:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.build_inference_features \
  --target-date-local YYYY-MM-DD
```

Then predict from that generated row:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.predict \
  --features-path experiments/no_hrrr_model/data/runtime/predictions/features/target_date_local=YYYY-MM-DD/no_hrrr.inference_features_normalized.parquet \
  --target-date-local YYYY-MM-DD \
  --event-bins-path path/to/event_bins.json
```

`--event-bins-path` accepts either a text file with one label per line or JSON containing a list, `bins`, `outcomes`, or `markets`. JSON objects can use `label`, `name`, `outcome`, or `title`.

Fetch a Polymarket event by slug and parse its temperature-bin markets automatically:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.polymarket_event \
  --event-slug highest-temperature-in-nyc-on-april-11-2026
```

This uses the public Polymarket Gamma API event endpoint and writes:

- `experiments/no_hrrr_model/data/runtime/polymarket/event_slug=SLUG/polymarket_event.json`
- `experiments/no_hrrr_model/data/runtime/polymarket/event_slug=SLUG/event_bins.json`
- `experiments/no_hrrr_model/data/runtime/polymarket/event_slug=SLUG/event_bins.manifest.json`

The event-bin JSON preserves the ordered bin labels plus market metadata such as market id, slug, condition id, CLOB token ids, outcomes, outcome prices, and question text. The labels can be passed directly to `predict.py` with `--event-bins-path`.

Fetch online source inputs and run inference for one target date:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.run_online_inference \
  --target-date-local YYYY-MM-DD \
  --event-bins-path path/to/event_bins.json
```

This date-scoped command refreshes experiment-local Wunderground, LAMP, and NBM artifacts before building the inference row. NBM is download-heavy; for large windows, keep using the user-run backfill commands rather than broad online inference loops. LAMP source selection defaults to `--lamp-source auto`: the runner checks live NOMADS first, then the public NOAA LAMP archive, then the Iowa State IEM AFOS archive for station-specific `LAVLGA` products. You can still force `--lamp-source archive`, `--lamp-source live`, or `--lamp-source iem`.

Online inference writes source artifacts by target date where practical, so partial runs remain inspectable even if a later source is unavailable. For example, Wunderground online artifacts for `2026-04-14` are written under:

- `experiments/no_hrrr_model/data/runtime/online_inference/wunderground_history/target_date_local=2026-04-14/`
- `experiments/no_hrrr_model/data/runtime/online_inference/wunderground_tables/target_date_local=2026-04-14/`

Each run also writes a status manifest:

- `experiments/no_hrrr_model/data/runtime/online_inference/status/target_date_local=YYYY-MM-DD/online_inference.manifest.json`

If the run fails, the manifest records the failure message plus the artifact paths that were already produced.

To have the same online inference command fetch the Polymarket bins itself, pass the event slug instead of a local event-bin file:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.run_online_inference \
  --target-date-local YYYY-MM-DD \
  --polymarket-event-slug highest-temperature-in-nyc-on-april-11-2026
```

For the standard NYC daily-high weather event slug, the runner can derive the slug from `--target-date-local`; pass `--polymarket-event-slug` without a value:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.run_online_inference \
  --target-date-local 2026-04-25 \
  --polymarket-event-slug
```

This derives `highest-temperature-in-nyc-on-april-25-2026`. If Polymarket uses a nonstandard slug for a specific event, pass the explicit slug value.

Launch the guided terminal UI for one-date online inference:

```bash
.venv/bin/python -m experiments.no_hrrr_model.no_hrrr_model.tui
```

The TUI defaults the target date to the current `America/New_York` date, derives the standard Polymarket weather-event slug from that date, and runs the same online inference command under the hood. It shows the expected final high, final-temperature quantiles, and mapped event-bin probabilities after a successful run. The runtime-root field controls where that TUI run writes artifacts. Mouse tracking is disabled by default to avoid terminal escape sequences leaking into inputs; use the keyboard to move between fields, press the Run button, and press `Ctrl+Q` or `Esc` to quit. If your terminal handles Textual mouse events correctly, add `--enable-mouse`.

TUI runs are isolated under:

- `experiments/no_hrrr_model/data/runtime/tui_online_inference/`

Each run gets its own `target_date_local=YYYY-MM-DD__run_id=...` directory with source artifacts, status manifest, and prediction JSON. If a TUI-launched run fails, the TUI deletes that run directory after reading the failure manifest. The plain `run_online_inference` CLI still preserves partial artifacts for inspection.

Run tests:

```bash
.venv/bin/python -m pytest -q experiments/no_hrrr_model/tests
```
