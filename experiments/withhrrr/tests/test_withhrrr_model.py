from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.withhrrr.withhrrr_model.calibrate_ladder import apply_bucket_reliability, fit_bucket_reliability, ladder_records
from experiments.withhrrr.withhrrr_model.calibrate_rolling_origin import calibration_sort_key
from experiments.withhrrr.withhrrr_model.distribution import degree_ladder_from_quantiles
from experiments.withhrrr.withhrrr_model.event_bins import EventBin, load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from experiments.withhrrr.withhrrr_model.hrrr_ablation_diagnostics import drop_hrrr_feature_columns
from experiments.withhrrr.withhrrr_model.model_config import DEFAULT_CANDIDATES, DEFAULT_MODEL_CANDIDATE_ID, HRRR_CANDIDATES, candidate_by_id
from experiments.withhrrr.withhrrr_model.polymarket_event import extract_event_bins, weather_event_slug_for_date
from experiments.withhrrr.withhrrr_model.predict import apply_ladder_calibration, calibration_offsets, selected_distribution_method
from experiments.withhrrr.withhrrr_model.prepare_training_features import build_training_table
from experiments.withhrrr.withhrrr_model.rolling_origin_model_select import leakage_findings
from experiments.withhrrr.withhrrr_model.run_online_inference import cleanup_runtime_artifacts
from experiments.withhrrr.withhrrr_model.train_quantile_models import select_feature_columns


def test_training_table_promotes_hrrr_disagreement_features() -> None:
    base = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "label_final_tmax_f": 70.0,
                "final_tmax_f": 70.0,
                "nbm_tmax_open_f": 68.0,
                "lamp_tmax_open_f": 65.0,
                "hrrr_temp_2m_day_max_f": 72.0,
                "meta_nbm_available": True,
                "meta_lamp_available": True,
                "meta_hrrr_available": True,
            }
        ]
    )
    out = build_training_table(base)
    assert float(out.loc[0, "anchor_tmax_f"]) == 66.5
    assert float(out.loc[0, "target_residual_f"]) == 3.5
    assert float(out.loc[0, "hrrr_tmax_open_f"]) == 72.0
    assert float(out.loc[0, "hrrr_minus_lamp_tmax_f"]) == 7.0
    assert float(out.loc[0, "hrrr_minus_nbm_tmax_f"]) == 4.0
    assert float(out.loc[0, "hrrr_outside_nbm_lamp_range_f"]) == 4.0
    assert bool(out.loc[0, "hrrr_hotter_than_lamp_3f"]) is True
    assert bool(out.loc[0, "model_training_eligible"]) is True


def test_feature_selection_allows_hrrr_but_excludes_leakage() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "target_residual_f": 1.0,
                "model_training_eligible": True,
                "label_final_tmax_f": 70.0,
                "meta_hrrr_available": True,
                "hrrr_tmax_open_f": 72.0,
                "hrrr_minus_lamp_tmax_f": 3.0,
                "meta_hrrr_anchor_init_time_utc_code": 1,
                "meta_hrrr_source_model_code": 2,
                "nbm_tmax_open_f": 68.0,
            }
        ]
    )
    assert select_feature_columns(df) == ["meta_hrrr_available", "hrrr_tmax_open_f", "hrrr_minus_lamp_tmax_f", "nbm_tmax_open_f"]
    findings = leakage_findings(["hrrr_tmax_open_f", "label_final_tmax_f", "meta_hrrr_source_model_code"])
    assert not [row for row in findings if row["feature"] == "hrrr_tmax_open_f"]
    assert [row for row in findings if row["feature"] == "label_final_tmax_f"]
    assert [row for row in findings if row["feature"] == "meta_hrrr_source_model_code"]


def test_hrrr_ablation_drops_hrrr_and_derived_features() -> None:
    full = [
        "wu_last_temp_f",
        "nbm_tmax_open_f",
        "lamp_tmax_open_f",
        "meta_hrrr_available",
        "hrrr_temp_2m_day_max_f",
        "hrrr_minus_lamp_tmax_f",
        "anchor_equal_3way_tmax_f",
    ]
    assert drop_hrrr_feature_columns(full) == ["wu_last_temp_f", "nbm_tmax_open_f", "lamp_tmax_open_f"]


def test_default_grid_contains_hrrr_specific_candidates() -> None:
    candidate_ids = [str(candidate["candidate_id"]) for candidate in DEFAULT_CANDIDATES]
    assert DEFAULT_MODEL_CANDIDATE_ID == "regularized_shallow_lgbm_300"
    assert len(candidate_ids) == len(set(candidate_ids))
    assert {str(candidate["candidate_id"]) for candidate in HRRR_CANDIDATES}.issubset(set(candidate_ids))
    assert candidate_by_id(DEFAULT_MODEL_CANDIDATE_ID)["candidate_id"] == DEFAULT_MODEL_CANDIDATE_ID


def test_prediction_reads_hrrr_segmented_calibration() -> None:
    payload = {
        "selected_method_id": "hrrr_lamp_direction_offsets",
        "methods": {
            "hrrr_lamp_direction_offsets": {
                "config": {
                    "segment_name": "hrrr_lamp_direction",
                    "global_offsets_f": {"q50": 0.0},
                    "segment_offsets_f": {"hrrr_hotter_than_lamp_3f": {"q50": -0.5}},
                }
            }
        },
    }
    row = pd.DataFrame([{"hrrr_minus_lamp_tmax_f": 4.0}])
    assert calibration_offsets(payload, row=row) == {"q50": -0.5}


def test_distribution_and_ladder_calibration_are_manifest_driven(tmp_path) -> None:
    method_id, path = selected_distribution_method("auto", manifest_path=tmp_path / "missing_distribution_manifest.json")
    assert method_id == "normal_iqr"
    assert path is None

    distribution_manifest = tmp_path / "distribution_manifest.json"
    distribution_manifest.write_text('{"selected_distribution_method_id": "normal_iqr"}')
    method_id, path = selected_distribution_method("auto", manifest_path=distribution_manifest)
    assert method_id == "normal_iqr"
    assert path == distribution_manifest

    reliability_path = tmp_path / "reliability.csv"
    reliability_path.write_text('probability_bucket,row_count,mean_predicted_probability,observed_frequency,raw_factor\n"(-0.001, 0.1]",10,0.05,0.10,2.0\n"(0.1, 0.2]",10,0.15,0.15,1.0\n')
    ladder_manifest = tmp_path / "ladder_manifest.json"
    ladder_manifest.write_text(
        json.dumps(
            {
                "selected_method_id": "bucket_reliability_s1_00",
                "bucket_count": 10,
                "bucket_reliability_path": str(reliability_path),
            }
        )
    )
    ladder = pd.DataFrame({"temp_f": [69, 70], "probability": [0.05, 0.15]})
    calibrated, metadata = apply_ladder_calibration(ladder, ladder_manifest)
    assert metadata["enabled"] is True
    assert round(float(calibrated["probability"].sum()), 6) == 1.0
    assert float(calibrated.loc[calibrated["temp_f"] == 69, "probability"].iloc[0]) > 0.25


def test_ladder_records_uses_selected_distribution_method() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-01-01",
                "station_id": "KLGA",
                "final_tmax_f": 70.0,
                "pred_tmax_q05_f": 65.0,
                "pred_tmax_q10_f": 66.0,
                "pred_tmax_q25_f": 68.0,
                "pred_tmax_q50_f": 70.0,
                "pred_tmax_q75_f": 72.0,
                "pred_tmax_q90_f": 74.0,
                "pred_tmax_q95_f": 75.0,
            }
        ]
    )
    records = ladder_records(df, distribution_method="normal_iqr")
    assert round(float(records["probability"].sum()), 6) == 1.0
    assert bool(records["observed"].any())


def test_ladder_bucket_reliability_preserves_probability_mass() -> None:
    records = pd.DataFrame(
        [
            {"target_date_local": "2025-01-01", "station_id": "KLGA", "temp_f": 69, "probability": 0.25, "observed": False},
            {"target_date_local": "2025-01-01", "station_id": "KLGA", "temp_f": 70, "probability": 0.50, "observed": True},
            {"target_date_local": "2025-01-01", "station_id": "KLGA", "temp_f": 71, "probability": 0.25, "observed": False},
            {"target_date_local": "2025-01-02", "station_id": "KLGA", "temp_f": 69, "probability": 0.20, "observed": True},
            {"target_date_local": "2025-01-02", "station_id": "KLGA", "temp_f": 70, "probability": 0.30, "observed": False},
            {"target_date_local": "2025-01-02", "station_id": "KLGA", "temp_f": 71, "probability": 0.50, "observed": False},
        ]
    )
    reliability = fit_bucket_reliability(records, bucket_count=5)
    calibrated = apply_bucket_reliability(records, reliability, bucket_count=5, shrinkage=0.5)
    assert calibrated.groupby(["target_date_local", "station_id"])["probability"].sum().round(6).tolist() == [1.0, 1.0]


def test_event_bin_adapter_and_polymarket_parser() -> None:
    ladder = pd.DataFrame({"temp_f": [68, 69, 70, 71], "probability": [0.25, 0.25, 0.25, 0.25]})
    bins = [parse_event_bin("68 or below"), EventBin("69-70", 69, 70), parse_event_bin("71+")]
    out = map_ladder_to_bins(ladder, bins, max_so_far_f=70.5)
    assert out.to_dict("records") == [
        {"bin": "68 or below", "probability": 0.0},
        {"bin": "69-70", "probability": 0.0},
        {"bin": "71+", "probability": 1.0},
    ]
    event = {
        "markets": [
            {"id": "1", "groupItemTitle": "59°F or below"},
            {"id": "2", "groupItemTitle": "60-61°F"},
            {"id": "3", "question": "Highest temperature in NYC on April 11? 78°F or higher"},
        ]
    }
    assert [row["label"] for row in extract_event_bins(event)] == ["59F or below", "60-61F", "78F or higher"]
    assert weather_event_slug_for_date(dt.date(2026, 4, 25)) == "highest-temperature-in-nyc-on-april-25-2026"


def test_load_event_bin_labels_and_distribution_mass(tmp_path) -> None:
    path = tmp_path / "bins.json"
    path.write_text('{"outcomes": [{"name": "80 or below"}, {"name": "81-84"}, {"name": "85+"}]}')
    assert load_event_bin_labels(path) == ["80 or below", "81-84", "85+"]
    ladder = degree_ladder_from_quantiles({0.05: 65.0, 0.1: 66.0, 0.25: 68.0, 0.5: 70.0, 0.75: 72.0, 0.9: 74.0, 0.95: 75.0}, method_id="normal_iqr")
    assert round(float(ladder["probability"].sum()), 6) == 1.0
    assert float(ladder["probability"].min()) >= 0.0


def test_calibration_sort_key_uses_probability_scores_before_uncalibrated_penalty() -> None:
    summary = pd.DataFrame(
        [
            {"method_id": "uncalibrated", "event_bin_nll": 1.0, "degree_ladder_nll": 2.0},
            {"method_id": "global_offsets", "event_bin_nll": 1.0, "degree_ladder_nll": 2.0},
            {"method_id": "worse", "event_bin_nll": 1.1, "degree_ladder_nll": 1.9},
        ]
    )
    assert list(calibration_sort_key(summary)["method_id"]) == ["global_offsets", "uncalibrated", "worse"]


def test_online_inference_cleanup_deletes_only_runtime_artifacts(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    status_dir = runtime_root / "status" / "target_date_local=2026-04-28"
    hrrr_dir = runtime_root / "hrrr"
    feature_file = runtime_root / "prediction_features" / "row.parquet"
    outside_dir = tmp_path / "outside"
    status_dir.mkdir(parents=True)
    hrrr_dir.mkdir(parents=True)
    feature_file.parent.mkdir(parents=True)
    feature_file.write_text("x")
    outside_dir.mkdir()

    deleted = cleanup_runtime_artifacts(runtime_root, [hrrr_dir, feature_file])

    assert deleted == [str(hrrr_dir), str(feature_file)]
    assert not hrrr_dir.exists()
    assert not feature_file.exists()
    assert status_dir.exists()

    try:
        cleanup_runtime_artifacts(runtime_root, [outside_dir])
    except SystemExit as exc:
        assert "outside runtime root" in str(exc)
    else:
        raise AssertionError("Expected cleanup outside runtime root to fail")
