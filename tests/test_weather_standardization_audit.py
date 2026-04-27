from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.weather import standardization_audit as audit


@pytest.fixture(scope="module")
def audit_result():
    matrix, review = audit.compare_contracts()
    return {
        "matrix": matrix,
        "review": review,
        "by_item": {row["item"]: row for row in matrix},
    }


def test_shared_location_context_source_matches():
    assert audit.same_location_context_source() is True
    assert audit.location_context.SETTLEMENT_LOCATION.station_id == "KLGA"
    assert audit.location_context.REGIONAL_CROP_BOUNDS.left == pytest.approx(282.5)


def test_deliberate_differences_are_classified_as_allowed(audit_result):
    by_item = audit_result["by_item"]
    assert by_item["source_model_product_values"]["classification"] == "allowed_difference"
    assert by_item["source_specific_mode_metadata"]["classification"] == "allowed_difference"
    assert by_item["raw_namespace_is_source_aware"]["classification"] == "allowed_difference"
    assert by_item["expected_completed_state"]["classification"] == "allowed_difference"
    assert by_item["manifest_json_sidecar"]["classification"] == "allowed_difference"


def test_runtime_and_canonical_contracts_are_now_synced(audit_result):
    by_item = audit_result["by_item"]
    assert by_item["source_version_field"]["classification"] == "synced"
    assert by_item["missing_optional_summary_columns"]["classification"] == "synced"
    assert by_item["wide_column_compatibility"]["classification"] == "synced"
    assert by_item["direct_derived_missing_semantics"]["classification"] == "synced"
    assert by_item["manifest_parquet_artifact"]["classification"] == "synced"
    assert "Both live raw pipelines now expose source_version" in by_item["source_version_field"]["notes"]


def test_manifest_normalizers_compare_semantics_not_file_format():
    hrrr_manifest_row = {
        "status": "ok",
        "missing_fields": "[]",
        "wide_parquet_path": "/tmp/hrrr.parquet",
        "provenance_path": "/tmp/hrrr.provenance.parquet",
        "keep_downloads": False,
        "keep_reduced": False,
    }
    hrrr_manifest_json = {
        "expected_task_count": 1,
        "completed_task_keys": ["task_a"],
        "failure_reasons": {},
        "missing_fields": {},
    }
    nbm_manifest = {
        "extraction_status": "ok",
        "warnings": "",
        "wide_output_paths": "/tmp/nbm.parquet",
        "provenance_output_paths": "/tmp/nbm.provenance.parquet",
        "raw_deleted": True,
        "reduced_deleted": True,
    }
    hrrr_semantics = audit.normalize_hrrr_manifest_semantics(hrrr_manifest_row, hrrr_manifest_json)
    nbm_semantics = audit.normalize_nbm_manifest_semantics(nbm_manifest)
    assert hrrr_semantics["tracks_output_paths"] is True
    assert nbm_semantics["tracks_output_paths"] is True
    assert hrrr_semantics["tracks_failure_state"] is True
    assert nbm_semantics["tracks_failure_state"] is True


def test_audit_writes_output_files(tmp_path, audit_result):
    matrix_path, review_path = audit.write_outputs(tmp_path, audit_result["matrix"], audit_result["review"])
    assert matrix_path.exists()
    assert review_path.exists()
    matrix = json.loads(matrix_path.read_text())
    review = review_path.read_text()
    assert any(row["classification"] == "allowed_difference" for row in matrix)
    assert not any(row["classification"] == "drift" for row in matrix)
    assert "Verdict: **standardized with accepted differences**" in review


def test_committed_hrrr_smoke_artifacts_match_current_contract():
    fixture_dir = ROOT / "tools" / "hrrr" / "data" / "fixtures" / "contract_smoke"
    manifest_json = json.loads((fixture_dir / "2025-04.manifest.json").read_text())
    manifest_df = pd.read_parquet(fixture_dir / "2025-04.manifest.parquet")
    wide_df = pd.read_parquet(fixture_dir / "2025-04.parquet")
    provenance_df = pd.read_parquet(fixture_dir / "2025-04.provenance.parquet")

    assert manifest_json["source_version"] == "hrrr-conus-wrfsfcf-public"
    assert "manifest_parquet_path" in manifest_json
    assert "source_version" in manifest_df.columns
    assert "source_version" in wide_df.columns
    assert "source_version" in provenance_df.columns
    assert not any("_nearest" in column or "_3x3_" in column or "_7x7_" in column for column in wide_df.columns)
