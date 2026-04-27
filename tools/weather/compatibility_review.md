# HRRR vs NBM Standardization Review

Verdict: **standardized with accepted differences**

## Direct Answers
- Shared raw runtime contract: yes
- Canonical training schema available: yes
- Provenance minimum synchronized: yes
- Operational manifest minimum synchronized: yes

## Synced Areas
- `raw_runtime_contract.shared_location_context`: Both pipelines should resolve spatial helpers from tools/weather/location_context.py.
- `raw_runtime_contract.runtime_identity_columns`: Both live raw pipelines must carry the shared runtime row-identity columns, including source_version.
- `raw_runtime_contract.source_version_field`: Both live raw pipelines now expose source_version on the wide rows.
- `raw_runtime_contract.runtime_spatial_metadata`: Spatial join metadata should be present on both wide outputs.
- `raw_runtime_contract.raw_spatial_suffix_contract`: Raw outputs keep source-specific base names but share the unsuffixed nearest plus crop, nb3, and nb7 spatial suffix pattern.
- `raw_runtime_contract.missing_optional_summary_columns`: Both raw pipelines now expose row-level missing-optional summaries.
- `canonical_training_schema.schema_module_present`: Canonical training wide columns are defined centrally in tools/weather/canonical_feature_schema.py.
- `canonical_training_schema.wide_column_compatibility`: The NBM and HRRR adapters now emit one stable canonical wide column layout with nullable passthrough metadata such as mode.
- `canonical_training_schema.provenance_column_compatibility`: The provenance adapters now emit one stable canonical lineage schema for downstream debugging and feature-traceability.
- `canonical_training_schema.shared_feature_mapping_examples`: Shared physical features such as temperature and 10 m wind now normalize into the same explicit-unit namespace.
- `provenance_minimum_contract.core_provenance_identity`: Both provenance outputs should expose feature identity, timestamps, and units.
- `provenance_minimum_contract.grib_source_metadata`: Both provenance tables expose GRIB short-name/level/step metadata, though field names differ slightly.
- `provenance_minimum_contract.direct_derived_missing_semantics`: Both raw provenance outputs now expose direct, derived, and missing-optional semantics.
- `provenance_minimum_contract.derivation_lineage_fields`: Both pipelines now expose derivation lineage and fallback descriptions through the raw provenance layer.
- `provenance_minimum_contract.inventory_line_reference`: Both pipelines retain a source inventory line reference for direct fields.
- `operational_manifest_contract.manifest_parquet_artifact`: Both live pipelines write a Parquet manifest artifact; HRRR also retains an auxiliary JSON sidecar for month-level recovery state.
- `operational_manifest_contract.tracks_failure_state`: Both manifests should expose failure/error state.
- `operational_manifest_contract.tracks_missing_fields`: Both manifests should expose missing/partial extraction information.
- `operational_manifest_contract.tracks_output_paths`: Both manifests should expose output artifact paths.
- `operational_manifest_contract.tracks_cleanup_state`: Both manifests should record or imply cleanup/retention policy.
- `artifact_policy.parquet_first_persistent_outputs`: Both production paths persist tabular outputs and treat GRIB as processing artifacts by default.
- `artifact_policy.keep_flags_debug_only`: Both pipelines expose keep flags for raw/reduced artifacts while defaulting to cleanup.

## Allowed Differences
- `raw_runtime_contract.source_model_product_values`: Model/product identifiers are expected to differ by source.
- `raw_runtime_contract.source_specific_mode_metadata`: NBM exposes premarket/intraday mode as optional metadata; HRRR intentionally keeps it out of raw runtime identity.
- `raw_runtime_contract.raw_namespace_is_source_aware`: Raw NBM and raw HRRR remain source-aware; the model-facing standard now lives in the canonical training schema instead of in the raw wide tables.
- `operational_manifest_contract.expected_completed_state`: HRRR manifest JSON tracks month-level expected/completed task state; NBM manifest rows remain per-unit operational records.
- `operational_manifest_contract.manifest_json_sidecar`: HRRR keeps an auxiliary JSON manifest sidecar for resumable month builds; NBM does not need the same recovery artifact.
- `artifact_policy.nbm_long_output`: NBM supports optional long-format output; HRRR intentionally keeps long output out of scope.

## Drift Requiring Follow-up
- none

## Summary
- synced: 22
- allowed_difference: 6
- drift: 0