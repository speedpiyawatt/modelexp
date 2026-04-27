# tools/weather

This folder is the repo's shared weather-layer glue. It is not one single pipeline.

It currently does four different jobs:

1. define shared schemas and location helpers
2. normalize source-specific outputs into stable contracts
3. build the overnight merged and normalized training tables
4. run verification, staging, audit, and review workflows around those tables

That mix is why the folder feels bloated. The core path is fairly small, but it sits next to a lot of QA and review tooling.

## Keep / Importance

### Core pipeline files

These are the files most directly tied to the overnight KLGA training-table path.

- `build_training_features_overnight.py`
  Merges Wunderground labels and cutoff-state features with NBM, LAMP, and HRRR overnight daily outputs into one source-aware training table.
- `build_training_features_overnight_normalized.py`
  Converts the merged overnight table into the model-ready normalized table with standardized units and encoded categorical vocabularies.
- `training_features_overnight_contract.py`
  Frozen source-aware registry for the merged overnight table.
- `training_features_overnight_normalized_contract.py`
  Frozen registry for the normalized overnight table, including transform metadata and normalized units.
- `training_feature_vocabularies.json`
  Encoding maps for cloud cover, weather family, and precipitation type used by normalized training features.
- `location_context.py`
  Shared KLGA settlement-location and crop/neighborhood spatial helpers used across weather tooling.

### Shared normalization / schema files

These matter when comparing or standardizing raw NBM, HRRR, and LAMP outputs outside the overnight merged-table builder.

- `canonical_feature_schema.py`
  Defines the canonical cross-source wide/provenance schema and raw-to-canonical feature-name mappings.
- `normalize_training_features.py`
  Normalizes raw NBM, HRRR, or LAMP wide/provenance tables into the canonical cross-source schema.

### Important verification / execution entrypoints

These are not the core model builders, but they are useful operational entrypoints.

- `run_verification_suite.py`
  Canonical verification runner. Executes local contract checks and selected live-source checks, then writes a summary.
- `run_server_overnight_stage.py`
  Staged orchestration entrypoint for WU, smoke, and short-window overnight runs.
- `filter_wu_training_tables.py`
  Builds short-window Wunderground label/observation subsets for contract tests and stage reviews.
- `progress.py`
  Shared progress-bar and progress-line utilities used by weather tooling.

## Lower Importance

These files are useful, but they are more review, audit, or comparison tooling than core production logic.

- `audit_training_features_overnight.py`
  Audits a merged overnight table against the checked-in merged contract.
- `audit_training_features_overnight_normalized.py`
  Audits a normalized overnight table against the checked-in normalized contract.
- `compare_weather_contracts.py`
  Compares NBM and HRRR raw/canonical/manifest contract surfaces and emits a report.
- `standardization_audit.py`
  Synthetic comparison harness that builds compatibility outputs for HRRR vs NBM standardization.
- `run_month_contract_test.py`
  Builds a month-scoped short-window contract review from Wunderground plus existing source outputs.
- `run_overnight_review.py`
  Local review harness that samples dates, inspects merged rows, and fuzzes normalized-table behavior.
- `run_live_overnight_probes.py`
  Probe harness that runs live LAMP/NBM/HRRR commands on selected dates and summarizes results.

### Generated or report-style artifacts

These are documentation or generated review outputs, not core code.

- `compatibility_matrix.json`
  Structured output from the standardization/compatibility review.
- `compatibility_review.md`
  Human-readable summary of the compatibility review.
- `linux_portability_sweep_v2.md`
  Notes from a Linux portability cleanup sweep.

### Minimal package file

- `__init__.py`
  Package marker only.

## File-by-File Reference

### `__init__.py`

Purpose:
- Makes `tools.weather` importable as a package.

Importance:
- Low.

### `location_context.py`

Purpose:
- Defines settlement metadata for KLGA.
- Provides longitude normalization, nearest-grid lookup, crop summaries, and neighborhood metrics.

Used by:
- raw source pipelines
- standardization checks
- any code that needs one shared definition of the settlement location

Importance:
- High.

### `canonical_feature_schema.py`

Purpose:
- Declares the canonical raw-wide and provenance schemas for cross-source normalization.
- Contains source-to-canonical feature base mappings for NBM, HRRR, and LAMP.

Used by:
- `normalize_training_features.py`
- `standardization_audit.py`
- cross-source compatibility work

Importance:
- High for raw-source standardization.

### `normalize_training_features.py`

Purpose:
- Converts raw NBM, HRRR, or LAMP wide/provenance outputs into a shared canonical schema.
- Handles unit conversions needed to put source-specific outputs into the same raw training namespace.

Used by:
- compatibility and standardization tooling
- raw source comparison workflows

Importance:
- High if you are still maintaining shared raw-source contracts.
- Medium relative to the overnight merged-table path.

### `training_features_overnight_contract.py`

Purpose:
- Defines the frozen column registry for the merged overnight training table.
- Encodes identity, labels, metadata, and source-aware feature columns.
- Includes the current NBM native daily-guidance and regime fields, including `nbm_native_tmax_*`, `nbm_native_tmin_*`, `nbm_pcpdur_*`, `nbm_pwther_*`, `nbm_tstm_*`, optional `nbm_ptype_*` / `nbm_thunc_*` / `nbm_vrate_*`, and `meta_nbm_missing_required_feature_count`.

Used by:
- `build_training_features_overnight.py`
- merged-table audits
- tests that enforce column stability

Importance:
- High.

### `training_features_overnight_normalized_contract.py`

Purpose:
- Defines the frozen registry for the normalized overnight table.
- Maps source-aware columns into normalized names, transforms, and units.

Used by:
- `build_training_features_overnight_normalized.py`
- normalized-table audits
- tests that enforce model-input stability

Importance:
- High.

### `training_feature_vocabularies.json`

Purpose:
- Stable integer vocabularies for encoded categorical weather fields.

Used by:
- `build_training_features_overnight_normalized.py`

Importance:
- High.

### `build_training_features_overnight.py`

Purpose:
- Main merged-table builder for the overnight KLGA training dataset.
- Reads WU labels/observations plus NBM, LAMP, and HRRR overnight outputs.
- Builds WU cutoff features and merges all source blocks into one row per target date.
- Preserves registered NBM native daily guidance and regime features from `nbm.overnight.parquet`; unregistered source columns are intentionally dropped by the registry layout.

Role in project:
- This is a central file for the current overnight modeling milestone.

Importance:
- Very high.

### `build_training_features_overnight_normalized.py`

Purpose:
- Main normalized-table builder for the overnight KLGA training dataset.
- Converts merged-table values into stable model-ready units and vocab IDs.
- Writes a normalization manifest.

Role in project:
- This is the second core builder after merged-table assembly.

Importance:
- Very high.

### `audit_training_features_overnight.py`

Purpose:
- Checks a merged overnight table for registry drift, missing core columns, prefix violations, and duplicate keys.

Role in project:
- QA guardrail, not core feature generation.

Importance:
- Medium.

### `audit_training_features_overnight_normalized.py`

Purpose:
- Same audit concept as above, but for the normalized overnight table.
- Also checks normalized-unit suffix policy and model-input string leakage.

Role in project:
- QA guardrail, not core feature generation.

Importance:
- Medium.

### `filter_wu_training_tables.py`

Purpose:
- Creates date-windowed Wunderground subsets for short contract tests and staging workflows.

Role in project:
- Support tool for review/staging, not a core builder.

Importance:
- Medium.

### `progress.py`

Purpose:
- Small shared utility for human-readable progress output.

Role in project:
- Developer/operator convenience only.

Importance:
- Low to medium.

### `run_verification_suite.py`

Purpose:
- Main verification harness for the weather stack.
- Runs local contract checks and optional live-source checks, then writes a summarized result.

Role in project:
- Canonical repo verification entrypoint.

Importance:
- High for validation, but not part of the model itself.

### `run_server_overnight_stage.py`

Purpose:
- Staged runner for WU prep, smoke checks, and short-window overnight source builds.
- Produces stage JSON summaries and run summaries.

Role in project:
- Orchestration and operational validation.

Importance:
- High for staged execution.

### `run_month_contract_test.py`

Purpose:
- Builds a month-scoped contract/review window using WU tables plus source outputs already present on disk.

Role in project:
- Review harness.

Importance:
- Medium to low unless you actively use month-scope review workflows.

### `run_overnight_review.py`

Purpose:
- Local review utility for merged and normalized overnight tables.
- Samples dates, inspects row behavior, and fuzzes normalization behavior.

Role in project:
- Review and exploratory QA.

Importance:
- Medium to low.

### `run_live_overnight_probes.py`

Purpose:
- Runs live LAMP, NBM, and HRRR probe commands against selected dates and summarizes source availability plus merge behavior.

Role in project:
- Live-source smoke testing.

Importance:
- Medium to low unless you actively probe sources.

### `compare_weather_contracts.py`

Purpose:
- Compares NBM and HRRR raw-wide, provenance, and manifest contracts.
- Produces a structured report and markdown review.

Role in project:
- Cross-source schema comparison tooling.

Importance:
- Medium to low.

### `standardization_audit.py`

Purpose:
- Synthetic audit harness that fabricates NBM and HRRR sample outputs, runs standardization logic, and emits compatibility outputs.

Role in project:
- One layer deeper than `compare_weather_contracts.py`; mostly QA and documentation support.

Importance:
- Medium to low.

### `compatibility_matrix.json`

Purpose:
- Machine-readable compatibility output generated from standardization/comparison work.

Importance:
- Low as code, medium as documentation snapshot.

### `compatibility_review.md`

Purpose:
- Human-readable summary of the compatibility review.

Importance:
- Low as code, medium as historical documentation.

### `linux_portability_sweep_v2.md`

Purpose:
- Notes from a portability sweep, mostly documenting what was fixed and how it was checked.

Importance:
- Low.

## Recommended Mental Model

If you only care about the current overnight KLGA milestone, focus on this subset first:

- `location_context.py`
- `training_features_overnight_contract.py`
- `build_training_features_overnight.py`
- `training_features_overnight_normalized_contract.py`
- `training_feature_vocabularies.json`
- `build_training_features_overnight_normalized.py`
- `run_verification_suite.py`

If you want to slim this folder later, the safest archive-first candidates are:

- `run_live_overnight_probes.py`
- `run_month_contract_test.py`
- `run_overnight_review.py`
- `compare_weather_contracts.py`
- `standardization_audit.py`
- `compatibility_matrix.json`
- `compatibility_review.md`
- `linux_portability_sweep_v2.md`

I would not start by deleting the contract files or the two overnight builders. Those are the core of what this folder is doing.
