from __future__ import annotations

import asyncio
import datetime as dt
import json
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.no_hrrr_model.no_hrrr_model.build_training_features import build_training_features
from experiments.no_hrrr_model.no_hrrr_model.build_inference_features import filter_label_history_for_inference
from experiments.no_hrrr_model.no_hrrr_model.contracts import audit_training_features
from experiments.no_hrrr_model.no_hrrr_model.distribution import degree_ladder_from_quantiles, quantiles_to_degree_ladder
from experiments.no_hrrr_model.no_hrrr_model.event_bins import EventBin, load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from experiments.no_hrrr_model.no_hrrr_model.evaluate import (
    build_degree_ladder_diagnostics,
    build_event_bin_diagnostics,
    DEFAULT_REPRESENTATIVE_EVENT_BINS,
    representative_event_bins,
)
from experiments.no_hrrr_model.no_hrrr_model.normalize_features import normalize_features
from experiments.no_hrrr_model.no_hrrr_model.polymarket_event import extract_event_bins, fallback_weather_event_slug, weather_event_slug_for_date, write_outputs
from experiments.no_hrrr_model.no_hrrr_model.predict import calibration_offsets, selected_distribution_method
from experiments.no_hrrr_model.no_hrrr_model.calibrate_rolling_origin import calibration_sort_key
from experiments.no_hrrr_model.no_hrrr_model.calibrate_ladder import apply_bucket_reliability, fit_bucket_reliability, selected_quantile_calibrated_frame
from experiments.no_hrrr_model.no_hrrr_model.distribution_diagnostics import crossing_diagnostics, normal_iqr_ladder, smooth_ladder
from experiments.no_hrrr_model.no_hrrr_model.ensemble_diagnostics import build_ensemble_frame, ensemble_specs, ranked_candidates
from experiments.no_hrrr_model.no_hrrr_model.rolling_origin_anchor_select import (
    anchor_candidate_specs,
    apply_anchor,
    best_weight_for_subset,
    fixed_anchor,
    fit_ridge_anchor,
    month_segment,
    transformed_fold_df,
)
from experiments.no_hrrr_model.no_hrrr_model.rolling_origin_model_select import DEFAULT_CANDIDATES, DEFAULT_MODEL_CANDIDATE_ID, candidate_by_id, leakage_findings, load_candidates, load_splits, summarize_candidates
from experiments.no_hrrr_model.no_hrrr_model.run_online_inference import cleanup_runtime_artifacts, resolve_lamp_source
from experiments.no_hrrr_model.no_hrrr_model.tui import (
    command_for_run,
    default_target_date,
    failure_message,
    format_prediction_summary,
    parse_args as parse_tui_args,
    parse_target_date,
    safe_delete_run_root,
    RunConfig,
)
from experiments.no_hrrr_model.no_hrrr_model.train_quantile_models import select_feature_columns


def test_training_builder_anchor_residual_and_no_hrrr_columns() -> None:
    labels = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "label_final_tmax_f": 70.0,
                "label_final_tmin_f": 50.0,
                "label_total_precip_in": 0.0,
            }
        ]
    )
    obs = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "valid_time_local": "2025-04-11T00:00:00-04:00",
                "temp_f": 51.0,
                "dewpoint_f": 45.0,
                "pressure_in": 30.0,
                "wind_speed_mph": 6.0,
                "wind_gust_mph": 8.0,
                "visibility": 10.0,
                "precip_hrly_in": 0.0,
            }
        ]
    )
    nbm = pd.DataFrame([{"target_date_local": "2025-04-11", "station_id": "KLGA", "nbm_temp_2m_day_max_k": 294.2611111111}])
    lamp = pd.DataFrame([{"target_date_local": "2025-04-11", "station_id": "KLGA", "day_tmp_max_f_forecast": 72.0}])

    out = build_training_features(labels_df=labels, obs_df=obs, nbm_df=nbm, lamp_df=lamp)

    assert not [column for column in out.columns if column.startswith(("hrrr_", "meta_hrrr_"))]
    assert round(float(out.loc[0, "nbm_tmax_open_f"]), 6) == 70.0
    assert float(out.loc[0, "lamp_tmax_open_f"]) == 72.0
    assert round(float(out.loc[0, "anchor_tmax_f"]), 6) == 71.0
    assert round(float(out.loc[0, "target_residual_f"]), 6) == -1.0
    assert bool(out.loc[0, "model_training_eligible"]) is True
    assert audit_training_features(out).ok


def test_audit_rejects_duplicate_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "selection_cutoff_local": "2025-04-11T00:05:00-04:00",
                "final_tmax_f": 70.0,
                "final_tmin_f": 50.0,
                "nbm_tmax_open_f": 70.0,
                "lamp_tmax_open_f": 72.0,
                "anchor_tmax_f": 71.0,
                "nbm_minus_lamp_tmax_f": -2.0,
                "target_residual_f": -1.0,
                "meta_nbm_available": True,
                "meta_lamp_available": True,
                "meta_wu_obs_available": True,
                "model_training_eligible": True,
            }
        ]
        * 2
    )
    result = audit_training_features(df)
    assert not result.ok
    assert any("duplicate" in error for error in result.errors)


def test_missing_sources_are_flagged() -> None:
    labels = pd.DataFrame([{"target_date_local": "2025-04-11", "station_id": "KLGA", "label_final_tmax_f": 70.0, "label_final_tmin_f": 50.0}])
    out = build_training_features(labels_df=labels, obs_df=pd.DataFrame(), nbm_df=pd.DataFrame(), lamp_df=pd.DataFrame())
    assert bool(out.loc[0, "meta_nbm_available"]) is False
    assert bool(out.loc[0, "meta_lamp_available"]) is False
    assert bool(out.loc[0, "meta_wu_obs_available"]) is False
    assert bool(out.loc[0, "model_training_eligible"]) is False
    assert audit_training_features(out).ok


def test_duplicate_source_rows_are_rejected() -> None:
    labels = pd.DataFrame([{"target_date_local": "2025-04-11", "station_id": "KLGA", "label_final_tmax_f": 70.0, "label_final_tmin_f": 50.0}])
    nbm = pd.DataFrame(
        [
            {"target_date_local": "2025-04-11", "station_id": "KLGA", "nbm_temp_2m_day_max_k": 294.0},
            {"target_date_local": "2025-04-11", "station_id": "KLGA", "nbm_temp_2m_day_max_k": 295.0},
        ]
    )
    try:
        build_training_features(labels_df=labels, obs_df=pd.DataFrame(), nbm_df=nbm, lamp_df=pd.DataFrame())
    except ValueError as exc:
        assert "NBM input has duplicate" in str(exc)
    else:
        raise AssertionError("duplicate NBM rows should be rejected")


def test_normalizer_units() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "selection_cutoff_local": "2025-04-11T00:05:00-04:00",
                "final_tmax_f": 70.0,
                "final_tmin_f": 50.0,
                "nbm_tmax_open_f": 70.0,
                "lamp_tmax_open_f": 72.0,
                "anchor_tmax_f": 71.0,
                "nbm_minus_lamp_tmax_f": -2.0,
                "target_residual_f": -1.0,
                "meta_nbm_available": True,
                "meta_lamp_available": True,
                "meta_wu_obs_available": True,
                "model_training_eligible": True,
                "nbm_temp_2m_15_local_k": 294.2611111111,
                "nbm_wind_10m_speed_15_local_ms": 10.0,
                "lamp_wsp_kt_at_15": 10.0,
            }
        ]
    )
    out, vocabularies = normalize_features(df)
    assert round(float(out.loc[0, "nbm_temp_2m_15_local_f"]), 6) == 70.0
    assert round(float(out.loc[0, "nbm_wind_10m_speed_15_local_mph"]), 6) == round(22.369362921, 6)
    assert round(float(out.loc[0, "lamp_wsp_mph_at_15"]), 6) == round(11.50779448, 6)
    assert isinstance(vocabularies, dict)


def test_event_bin_adapter_maps_ladder_and_max_so_far() -> None:
    ladder = pd.DataFrame({"temp_f": [68, 69, 70, 71], "probability": [0.25, 0.25, 0.25, 0.25]})
    bins = [parse_event_bin("68 or below"), EventBin("69-70", 69, 70), parse_event_bin("71+")]
    out = map_ladder_to_bins(ladder, bins, max_so_far_f=70.5)
    assert out.to_dict("records") == [
        {"bin": "68 or below", "probability": 0.0},
        {"bin": "69-70", "probability": 0.0},
        {"bin": "71+", "probability": 1.0},
    ]


def test_distribution_rearranges_crossing_quantiles() -> None:
    ladder = quantiles_to_degree_ladder({0.05: 70.0, 0.5: 60.0, 0.95: 72.0})
    assert round(float(ladder["probability"].sum()), 6) == 1.0
    try:
        quantiles_to_degree_ladder({0.05: 70.0, 0.5: 60.0, 0.95: 72.0}, rearrange_crossing=False)
    except ValueError as exc:
        assert "monotone" in str(exc)
    else:
        raise AssertionError("crossing quantiles should be rejected when rearrangement is disabled")


def test_distribution_allocates_nonzero_tail_probability() -> None:
    ladder = quantiles_to_degree_ladder({0.05: 70.0, 0.5: 75.0, 0.95: 80.0}, min_temp_f=60, max_temp_f=90)
    assert round(float(ladder["probability"].sum()), 6) == 1.0
    assert float(ladder.loc[ladder["temp_f"] == 60, "probability"].iloc[0]) > 0.0
    assert float(ladder.loc[ladder["temp_f"] == 90, "probability"].iloc[0]) > 0.0
    assert round(float(ladder.loc[ladder["temp_f"] < 70, "probability"].sum()), 6) == 0.05
    assert round(float(ladder.loc[ladder["temp_f"] > 80, "probability"].sum()), 6) == 0.05


def test_degree_ladder_diagnostics_score_probability_forecast() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "final_tmax_f": 70.0,
                "nbm_minus_lamp_tmax_f": 1.0,
                "pred_tmax_q05_f": 68.0,
                "pred_tmax_q10_f": 69.0,
                "pred_tmax_q25_f": 70.0,
                "pred_tmax_q50_f": 70.0,
                "pred_tmax_q75_f": 71.0,
                "pred_tmax_q90_f": 72.0,
                "pred_tmax_q95_f": 73.0,
            }
        ]
    )

    scores, modal_reliability, degree_reliability, pit, metrics = build_degree_ladder_diagnostics(df)

    assert len(scores) == 1
    assert 0.0 <= float(scores.loc[0, "observed_probability"]) <= 1.0
    assert 0.0 <= float(scores.loc[0, "pit_mid"]) <= 1.0
    assert float(scores.loc[0, "negative_log_likelihood"]) >= 0.0
    assert not modal_reliability.empty
    assert not degree_reliability.empty
    assert {"mean_predicted_probability", "observed_frequency"}.issubset(set(degree_reliability.columns))
    assert pit.loc[0, "slice"] == "overall"
    assert metrics["row_count"] == 1


def test_event_bin_diagnostics_scores_representative_bins() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "final_tmax_f": 70.0,
                "pred_tmax_q05_f": 68.0,
                "pred_tmax_q10_f": 69.0,
                "pred_tmax_q25_f": 70.0,
                "pred_tmax_q50_f": 70.0,
                "pred_tmax_q75_f": 71.0,
                "pred_tmax_q90_f": 72.0,
                "pred_tmax_q95_f": 73.0,
            },
            {
                "target_date_local": "2025-04-12",
                "station_id": "KLGA",
                "final_tmax_f": 74.0,
                "pred_tmax_q05_f": 72.0,
                "pred_tmax_q10_f": 73.0,
                "pred_tmax_q25_f": 74.0,
                "pred_tmax_q50_f": 74.0,
                "pred_tmax_q75_f": 75.0,
                "pred_tmax_q90_f": 76.0,
                "pred_tmax_q95_f": 77.0,
            },
        ]
    )

    bins = representative_event_bins()
    scores, bin_metrics, reliability, metrics = build_event_bin_diagnostics(df)

    assert [event_bin.name for event_bin in bins] == list(DEFAULT_REPRESENTATIVE_EVENT_BINS)
    assert bins[0].upper_f is not None
    assert bins[-1].lower_f is not None
    assert len(scores) == 2
    assert set(scores["observed_bin"]).issubset(set(metrics["bins"]))
    assert not bin_metrics.empty
    assert not reliability.empty
    assert metrics["bin_count"] == len(metrics["bins"])


def test_event_bin_strict_under_and_greater_than() -> None:
    assert parse_event_bin("under 70").upper_f == 69
    assert parse_event_bin("less than 70").upper_f == 69
    assert parse_event_bin("greater than 70").lower_f == 71
    assert parse_event_bin("70 or below").upper_f == 70


def test_load_event_bin_labels_from_json(tmp_path) -> None:
    path = tmp_path / "bins.json"
    path.write_text('{"outcomes": [{"name": "80 or below"}, {"name": "81-84"}, {"name": "85+"}]}')
    assert load_event_bin_labels(path) == ["80 or below", "81-84", "85+"]


def test_extract_polymarket_event_bins_from_markets() -> None:
    event = {
        "id": "event-1",
        "markets": [
            {"id": "1", "groupItemTitle": "59°F or below", "question": "Highest temperature in NYC on April 11?"},
            {"id": "2", "groupItemTitle": "60-61°F", "question": "Highest temperature in NYC on April 11?"},
            {"id": "3", "question": "Highest temperature in NYC on April 11? 78°F or higher"},
        ],
    }
    rows = extract_event_bins(event)
    assert [row["label"] for row in rows] == ["59F or below", "60-61F", "78F or higher"]


def test_weather_event_slug_for_date() -> None:
    assert weather_event_slug_for_date(dt.date(2026, 4, 25)) == "highest-temperature-in-nyc-on-april-25-2026"
    assert fallback_weather_event_slug("highest-temperature-in-nyc-on-january-1-2026") == "highest-temperature-in-nyc-on-january-1"


def test_polymarket_manifest_records_resolved_slug(tmp_path) -> None:
    _, bins_path, manifest_path = write_outputs(
        output_dir=tmp_path,
        slug="highest-temperature-in-nyc-on-january-1-2026",
        resolved_slug="highest-temperature-in-nyc-on-january-1",
        event={"id": "event-1", "markets": []},
        bins=[{"label": "30F or below"}],
    )
    manifest = json.loads(manifest_path.read_text())
    assert bins_path.parent.name == "event_slug=highest-temperature-in-nyc-on-january-1"
    assert manifest["event_slug"] == "highest-temperature-in-nyc-on-january-1-2026"
    assert manifest["resolved_event_slug"] == "highest-temperature-in-nyc-on-january-1"
    assert manifest["used_fallback_slug"] is True


def test_lamp_source_auto_uses_archive_for_past_cutoff_dates() -> None:
    def url_exists(url: str) -> bool:
        return "202512.0230z.gz" in url

    assert resolve_lamp_source("auto", dt.date(2025, 12, 31), url_exists=url_exists, iem_product_exists=lambda _: False) == "archive"
    assert resolve_lamp_source("live", dt.date(2026, 4, 21), url_exists=url_exists) == "live"


def test_lamp_source_auto_uses_live_when_nomads_exists() -> None:
    def url_exists(url: str) -> bool:
        return "lmp.20260425/lmp.t0230z" in url

    assert resolve_lamp_source("auto", dt.date(2026, 4, 25), url_exists=url_exists, iem_product_exists=lambda _: False) == "live"


def test_lamp_source_auto_uses_iem_when_nomads_and_noaa_archive_missing() -> None:
    assert resolve_lamp_source("auto", dt.date(2026, 4, 21), url_exists=lambda _: False, iem_product_exists=lambda _: True) == "iem"


def test_lamp_source_auto_fails_cleanly_when_no_source_exists() -> None:
    try:
        resolve_lamp_source("auto", dt.date(2026, 4, 21), url_exists=lambda _: False, iem_product_exists=lambda _: False)
    except SystemExit as exc:
        assert "LAMP unavailable" in str(exc)
    else:
        raise AssertionError("auto LAMP source should fail when live and archive are unavailable")


def test_feature_selection_excludes_time_source_and_label_codes() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-04-11",
                "station_id": "KLGA",
                "selection_cutoff_local": "2025-04-11T00:05:00-04:00",
                "target_residual_f": 1.0,
                "model_training_eligible": True,
                "label_market_bin_code": 70,
                "meta_nbm_selected_init_time_utc_code": 1,
                "meta_nbm_source_model_code": 1,
                "nbm_temp_2m_day_max_f": 70.0,
                "wu_last_temp_f": 50.0,
            }
        ]
    )
    assert select_feature_columns(df) == ["nbm_temp_2m_day_max_f", "wu_last_temp_f"]


def test_model_selection_leakage_checker_flags_forbidden_features() -> None:
    findings = leakage_findings(["wu_last_temp_f", "label_market_bin_code", "hrrr_temp_2m_day_max_k", "meta_nbm_source_model_code"])
    reasons = {(row["feature"], row["reason"]) for row in findings}
    assert ("label_market_bin_code", "forbidden prefix") in reasons
    assert ("label_market_bin_code", "forbidden substring market") in reasons
    assert ("hrrr_temp_2m_day_max_k", "forbidden prefix") in reasons
    assert ("meta_nbm_source_model_code", "forbidden substring _source_model_code") in reasons


def test_model_selection_summary_sorts_by_probability_scores() -> None:
    metrics = pd.DataFrame(
        [
            {"candidate_id": "worse", "validation_row_count": 100, "degree_ladder_nll": 2.0, "degree_ladder_brier": 0.9, "degree_ladder_rps": 0.1, "event_bin_nll": 1.5, "event_bin_brier": 0.7, "final_tmax_q50_mae_f": 1.0, "final_tmax_q50_rmse_f": 1.2, "q05_q95_coverage": 0.8, "q50_pinball_loss": 0.5},
            {"candidate_id": "better", "validation_row_count": 10, "degree_ladder_nll": 2.2, "degree_ladder_brier": 0.9, "degree_ladder_rps": 0.1, "event_bin_nll": 1.0, "event_bin_brier": 0.6, "final_tmax_q50_mae_f": 1.1, "final_tmax_q50_rmse_f": 1.3, "q05_q95_coverage": 0.8, "q50_pinball_loss": 0.4},
            {"candidate_id": "better", "validation_row_count": 90, "degree_ladder_nll": 2.2, "degree_ladder_brier": 0.9, "degree_ladder_rps": 0.1, "event_bin_nll": 1.0, "event_bin_brier": 0.6, "final_tmax_q50_mae_f": 1.1, "final_tmax_q50_rmse_f": 1.3, "q05_q95_coverage": 0.8, "q50_pinball_loss": 0.6},
        ]
    )
    summary = summarize_candidates(metrics)
    assert list(summary["candidate_id"]) == ["better", "worse"]
    assert "weighted_mean_event_bin_nll" in summary.columns
    assert round(float(summary.loc[summary["candidate_id"] == "better", "weighted_mean_q50_pinball_loss"].iloc[0]), 6) == 0.58


def test_default_model_selection_grid_is_constrained_and_regularized() -> None:
    assert 15 <= len(DEFAULT_CANDIDATES) <= 20
    candidate_ids = [str(candidate["candidate_id"]) for candidate in DEFAULT_CANDIDATES]
    assert "current_lgbm_fixed_250_no_inner_es" in candidate_ids
    assert DEFAULT_MODEL_CANDIDATE_ID == "very_regularized_min_leaf70_lgbm_350"
    assert candidate_by_id(DEFAULT_MODEL_CANDIDATE_ID)["candidate_id"] == DEFAULT_MODEL_CANDIDATE_ID
    assert len(candidate_ids) == len(set(candidate_ids))
    tuned_candidates = [candidate for candidate in DEFAULT_CANDIDATES if str(candidate["candidate_id"]) != "current_lgbm_fixed_250_no_inner_es"]
    for candidate in tuned_candidates:
        params = candidate["params"]
        assert isinstance(params, dict)
        assert "max_depth" in params
        assert "lambda_l1" in params
        assert "lambda_l2" in params
        assert int(candidate["num_boost_round"]) <= 450


def test_model_selection_rejects_duplicate_candidate_ids(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text('{"candidates": [{"candidate_id": "dup", "params": {}}, {"candidate_id": "dup", "params": {}}]}')
    try:
        load_candidates(path)
    except ValueError as exc:
        assert "duplicate candidate_id" in str(exc)
    else:
        raise AssertionError("duplicate candidate ids should be rejected")


def test_prediction_reads_selected_calibration_manifest_offsets() -> None:
    payload = {
        "selected_method_id": "global_offsets",
        "methods": {
            "global_offsets": {
                "config": {
                    "offsets_f": {
                        "q05": -1.0,
                        "q50": 0.25,
                        "q95": 0.5,
                    }
                }
            }
        },
    }
    assert calibration_offsets(payload) == {"q05": -1.0, "q50": 0.25, "q95": 0.5}


def test_prediction_reads_selected_conformal_manifest_median_offsets() -> None:
    payload = {
        "selected_method_id": "conformal_intervals",
        "methods": {
            "conformal_intervals": {
                "config": {
                    "median_offsets_f": {
                        "q05": -0.5,
                        "q50": 0.0,
                        "q95": 0.5,
                    },
                    "interval_adjustments_f": {"q05_q95": 1.0, "q10_q90": 0.5, "q25_q75": 0.25},
                }
            }
        },
    }
    assert calibration_offsets(payload) == {"q05": -1.0, "q10": -0.5, "q25": -0.25, "q50": 0.0, "q75": 0.25, "q90": 0.5, "q95": 1.0}


def test_prediction_reads_selected_segmented_manifest_offsets() -> None:
    payload = {
        "selected_method_id": "disagreement_offsets",
        "methods": {
            "disagreement_offsets": {
                "config": {
                    "segment_name": "disagreement",
                    "global_offsets_f": {"q50": 0.0},
                    "segment_offsets_f": {"2_to_5f": {"q50": 0.75}},
                }
            }
        },
    }
    row = pd.DataFrame([{"nbm_minus_lamp_tmax_f": 3.0}])
    assert calibration_offsets(payload, row=row) == {"q50": 0.75}


def test_prediction_auto_distribution_uses_selected_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "distribution_manifest.json"
    manifest_path.write_text('{"selected_distribution_method_id": "normal_iqr"}')
    method_id, path = selected_distribution_method("auto", manifest_path=manifest_path)
    assert method_id == "normal_iqr"
    assert path == manifest_path


def test_prediction_distribution_override_preserves_old_ladder_method() -> None:
    method_id, path = selected_distribution_method("interpolation_tail", manifest_path=pathlib.Path("/missing/manifest.json"))
    assert method_id == "interpolation_tail"
    assert path is None


def test_calibration_sort_key_uses_probability_scores_before_uncalibrated_penalty() -> None:
    summary = pd.DataFrame(
        [
            {"method_id": "uncalibrated", "event_bin_nll": 1.0, "degree_ladder_nll": 2.0},
            {"method_id": "global_offsets", "event_bin_nll": 1.0, "degree_ladder_nll": 2.0},
            {"method_id": "worse", "event_bin_nll": 1.1, "degree_ladder_nll": 1.9},
        ]
    )
    assert list(calibration_sort_key(summary)["method_id"]) == ["global_offsets", "uncalibrated", "worse"]


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
    totals = calibrated.groupby(["target_date_local", "station_id"])["probability"].sum().round(6).tolist()
    assert totals == [1.0, 1.0]
    assert float(calibrated["probability"].min()) >= 0.0


def test_ladder_uses_selected_quantile_calibration_manifest() -> None:
    df = pd.DataFrame(
        [
            {
                "target_date_local": "2025-01-01",
                "station_id": "KLGA",
                "pred_tmax_q05_f": 69.0,
                "pred_tmax_q10_f": 69.5,
                "pred_tmax_q25_f": 70.0,
                "pred_tmax_q50_f": 71.0,
                "pred_tmax_q75_f": 72.0,
                "pred_tmax_q90_f": 72.5,
                "pred_tmax_q95_f": 73.0,
            }
        ]
    )
    manifest = {"selected_method_id": "global_offsets", "methods": {"global_offsets": {"config": {"offsets_f": {"q05": 1.0, "q10": 1.0, "q25": 1.0, "q50": 1.0, "q75": 1.0, "q90": 1.0, "q95": 1.0}}}}}
    out = selected_quantile_calibrated_frame(df, manifest)
    assert float(out["pred_tmax_q50_f"].iloc[0]) == 72.0


def test_distribution_crossing_diagnostics_counts_raw_crossings() -> None:
    df = pd.DataFrame(
        [
            {
                "candidate_id": "candidate",
                "train_end": "2023-12-31",
                "valid_start": "2024-01-01",
                "valid_end": "2024-12-31",
                "target_date_local": "2024-01-01",
                "anchor_tmax_f": 70.0,
                "nbm_minus_lamp_tmax_f": 1.0,
                "pred_residual_q05_f": 1.0,
                "pred_residual_q10_f": 0.0,
                "pred_residual_q25_f": 2.0,
                "pred_residual_q50_f": 3.0,
                "pred_residual_q75_f": 4.0,
                "pred_residual_q90_f": 5.0,
                "pred_residual_q95_f": 6.0,
            }
        ]
    )
    diagnostics = crossing_diagnostics(df)
    row = diagnostics.loc[(diagnostics["slice"] == "overall") & (diagnostics["quantile_pair"] == "q05_q10")].iloc[0]
    assert int(row["crossing_count"]) == 1
    any_row = diagnostics.loc[(diagnostics["slice"] == "overall") & (diagnostics["quantile_pair"] == "any_adjacent")].iloc[0]
    assert int(any_row["crossing_count"]) == 1


def test_distribution_alternative_ladders_preserve_mass() -> None:
    ladder = pd.DataFrame({"temp_f": [69, 70, 71], "probability": [0.0, 1.0, 0.0]})
    smoothed = smooth_ladder(ladder)
    assert round(float(smoothed["probability"].sum()), 6) == 1.0
    row = pd.Series({"pred_tmax_q05_f": 65.0, "pred_tmax_q25_f": 68.0, "pred_tmax_q50_f": 70.0, "pred_tmax_q75_f": 72.0, "pred_tmax_q95_f": 75.0})
    normal = normal_iqr_ladder(row, min_temp_f=60, max_temp_f=80)
    assert round(float(normal["probability"].sum()), 6) == 1.0
    assert float(normal["probability"].min()) >= 0.0
    collapsed = normal_iqr_ladder(pd.Series({"pred_tmax_q05_f": 70.0, "pred_tmax_q25_f": 70.0, "pred_tmax_q50_f": 70.0, "pred_tmax_q75_f": 70.0, "pred_tmax_q95_f": 70.0}), min_temp_f=65, max_temp_f=75)
    assert round(float(collapsed["probability"].sum()), 6) == 1.0
    assert collapsed["probability"].isna().sum() == 0
    runtime = degree_ladder_from_quantiles({0.05: 65.0, 0.1: 66.0, 0.25: 68.0, 0.5: 70.0, 0.75: 72.0, 0.9: 74.0, 0.95: 75.0}, method_id="normal_iqr")
    assert round(float(runtime["probability"].sum()), 6) == 1.0
    assert float(runtime["probability"].min()) >= 0.0


def test_ensemble_specs_use_selected_single_and_ranked_top_sets() -> None:
    ranked = ["a", "b", "c", "d"]
    specs = ensemble_specs(ranked, selected_candidate_id="b", sizes=[2, 3])
    assert specs[0] == {"method_id": "selected_single", "member_candidate_ids": ["b"]}
    assert specs[1] == {"method_id": "top2_quantile_mean", "member_candidate_ids": ["a", "b"]}
    assert specs[2] == {"method_id": "top3_quantile_mean", "member_candidate_ids": ["a", "b", "c"]}


def test_ranked_candidates_sorts_by_probability_scores() -> None:
    summary = pd.DataFrame(
        [
            {"candidate_id": "b", "weighted_mean_event_bin_nll": 1.1, "weighted_mean_degree_ladder_nll": 2.0},
            {"candidate_id": "a", "weighted_mean_event_bin_nll": 1.0, "weighted_mean_degree_ladder_nll": 2.1},
            {"candidate_id": "c", "weighted_mean_event_bin_nll": 1.0, "weighted_mean_degree_ladder_nll": 1.9},
        ]
    )
    assert ranked_candidates(summary) == ["c", "a", "b"]


def test_build_ensemble_frame_averages_and_rearranges_quantiles() -> None:
    rows = []
    for candidate_id, q05, q10 in [("a", 70.0, 69.0), ("b", 72.0, 73.0)]:
        row = {
            "candidate_id": candidate_id,
            "target_date_local": "2025-01-01",
            "station_id": "KLGA",
            "train_end": "2024-12-31",
            "valid_start": "2025-01-01",
            "valid_end": "2025-12-31",
            "final_tmax_f": 71.0,
            "target_residual_f": 0.0,
            "anchor_tmax_f": 71.0,
            "nbm_tmax_open_f": 71.0,
            "lamp_tmax_open_f": 71.0,
            "nbm_minus_lamp_tmax_f": 0.0,
        }
        for quantile in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            tag = f"q{int(round(quantile * 100)):02d}"
            row[f"pred_tmax_{tag}_f"] = 71.0
        row["pred_tmax_q05_f"] = q05
        row["pred_tmax_q10_f"] = q10
        rows.append(row)
    out = build_ensemble_frame(pd.DataFrame(rows), member_candidate_ids=["a", "b"], method_id="test")
    assert len(out) == 1
    assert float(out["pred_tmax_q05_f"].iloc[0]) == 71.0
    assert float(out["pred_tmax_q10_f"].iloc[0]) == 71.0
    assert str(out["member_candidate_ids"].iloc[0]) == "a,b"


def test_build_ensemble_frame_rejects_missing_member_for_target_key() -> None:
    rows = []
    for candidate_id, target_date in [("a", "2025-01-01"), ("b", "2025-01-01"), ("a", "2025-01-02")]:
        row = {
            "candidate_id": candidate_id,
            "target_date_local": target_date,
            "station_id": "KLGA",
            "train_end": "2024-12-31",
            "valid_start": "2025-01-01",
            "valid_end": "2025-12-31",
            "final_tmax_f": 71.0,
            "target_residual_f": 0.0,
            "anchor_tmax_f": 71.0,
            "nbm_tmax_open_f": 71.0,
            "lamp_tmax_open_f": 71.0,
            "nbm_minus_lamp_tmax_f": 0.0,
        }
        for quantile in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            tag = f"q{int(round(quantile * 100)):02d}"
            row[f"pred_tmax_{tag}_f"] = 71.0
        rows.append(row)
    try:
        build_ensemble_frame(pd.DataFrame(rows), member_candidate_ids=["a", "b"], method_id="test")
    except ValueError as exc:
        assert "without all requested members" in str(exc)
    else:
        raise AssertionError("ensemble frame should reject target keys missing a requested member")


def test_model_selection_loads_custom_splits(tmp_path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('{"splits": [{"train_end": "2023-12-31", "valid_start": "2024-01-01", "valid_end": "2024-06-30"}]}')
    assert load_splits(path) == [("2023-12-31", "2024-01-01", "2024-06-30")]


def test_model_selection_rejects_overlapping_custom_splits(tmp_path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('{"splits": [{"train_end": "2024-01-01", "valid_start": "2024-01-01", "valid_end": "2024-06-30"}]}')
    try:
        load_splits(path)
    except ValueError as exc:
        assert "train_end < valid_start" in str(exc)
    else:
        raise AssertionError("overlapping train and validation splits should be rejected")


def test_model_selection_rejects_reversed_validation_window(tmp_path) -> None:
    path = tmp_path / "splits.json"
    path.write_text('{"splits": [{"train_end": "2023-12-31", "valid_start": "2024-07-01", "valid_end": "2024-06-30"}]}')
    try:
        load_splits(path)
    except ValueError as exc:
        assert "valid_start <= valid_end" in str(exc)
    else:
        raise AssertionError("reversed validation windows should be rejected")


def test_anchor_selection_fixed_anchor_and_residual() -> None:
    df = pd.DataFrame(
        [
            {"final_tmax_f": 72.0, "nbm_tmax_open_f": 70.0, "lamp_tmax_open_f": 74.0},
            {"final_tmax_f": 75.0, "nbm_tmax_open_f": 76.0, "lamp_tmax_open_f": 72.0},
        ]
    )
    anchor = fixed_anchor(df, 0.25)
    out = apply_anchor(df, anchor)
    assert list(anchor) == [73.0, 73.0]
    assert list(out["target_residual_f"]) == [-1.0, 2.0]


def test_anchor_selection_best_weight_prefers_train_subset() -> None:
    df = pd.DataFrame(
        [
            {"final_tmax_f": 70.0, "nbm_tmax_open_f": 70.0, "lamp_tmax_open_f": 80.0},
            {"final_tmax_f": 72.0, "nbm_tmax_open_f": 72.0, "lamp_tmax_open_f": 82.0},
        ]
    )
    assert best_weight_for_subset(df, fallback_weight=0.5) == 1.0


def test_anchor_selection_best_weight_uses_fallback_for_nonfinite_subset() -> None:
    df = pd.DataFrame(
        [
            {"final_tmax_f": pd.NA, "nbm_tmax_open_f": pd.NA, "lamp_tmax_open_f": pd.NA},
        ]
    )
    assert best_weight_for_subset(df, fallback_weight=0.3) == 0.3


def test_anchor_selection_includes_month_segment_candidate() -> None:
    ids = {str(candidate["anchor_candidate_id"]) for candidate in anchor_candidate_specs()}
    assert "segmented_month_train_mae" in ids
    df = pd.DataFrame([{"target_date_local": "2024-01-15"}, {"target_date_local": "2024-12-31"}])
    assert list(month_segment(df)) == ["month_01", "month_12"]


def test_anchor_selection_ridge_anchor_uses_train_fold_only() -> None:
    df = pd.DataFrame(
        [
            {"target_date_local": "2023-01-01", "final_tmax_f": 70.0, "nbm_tmax_open_f": 70.0, "lamp_tmax_open_f": 70.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2023-01-02", "final_tmax_f": 72.0, "nbm_tmax_open_f": 72.0, "lamp_tmax_open_f": 72.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2023-01-03", "final_tmax_f": 74.0, "nbm_tmax_open_f": 74.0, "lamp_tmax_open_f": 74.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2023-01-04", "final_tmax_f": 76.0, "nbm_tmax_open_f": 76.0, "lamp_tmax_open_f": 76.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2023-01-05", "final_tmax_f": 78.0, "nbm_tmax_open_f": 78.0, "lamp_tmax_open_f": 78.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2023-01-06", "final_tmax_f": 80.0, "nbm_tmax_open_f": 80.0, "lamp_tmax_open_f": 80.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2024-01-01", "final_tmax_f": 999.0, "nbm_tmax_open_f": 76.0, "lamp_tmax_open_f": 76.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2024-01-02", "final_tmax_f": 999.0, "nbm_tmax_open_f": 78.0, "lamp_tmax_open_f": 78.0, "nbm_minus_lamp_tmax_f": 0.0},
            {"target_date_local": "2024-01-03", "final_tmax_f": 999.0, "nbm_tmax_open_f": 80.0, "lamp_tmax_open_f": 80.0, "nbm_minus_lamp_tmax_f": 0.0},
        ]
    )
    coef = fit_ridge_anchor(df.loc[df["target_date_local"] < "2024-01-01"], l2=0.1)
    assert len(coef) == 6
    transformed, metadata = transformed_fold_df(
        df,
        {"anchor_candidate_id": "ridge_linear_anchor", "anchor_type": "ridge_linear"},
        train_end="2023-12-31",
        valid_start="2024-01-01",
        valid_end="2024-12-31",
    )
    assert "coefficients" in metadata
    assert transformed.loc[transformed["target_date_local"] >= "2024-01-01", "anchor_tmax_f"].max() < 999.0


def test_inference_label_history_excludes_target_day_label() -> None:
    label_history = pd.DataFrame(
        [
            {"target_date_local": "2026-04-24", "station_id": "KLGA", "label_final_tmax_f": 65.0},
            {"target_date_local": "2026-04-25", "station_id": "KLGA", "label_final_tmax_f": 999.0},
            {"target_date_local": "2026-04-24", "station_id": "KJFK", "label_final_tmax_f": 70.0},
        ]
    )

    out = filter_label_history_for_inference(label_history, target_date_local="2026-04-25", station_id="KLGA")

    assert out.to_dict("records") == [{"target_date_local": "2026-04-24", "station_id": "KLGA", "label_final_tmax_f": 65.0}]


def test_tui_default_date_uses_new_york_date() -> None:
    now = dt.datetime(2026, 4, 26, 4, 44, tzinfo=dt.timezone(dt.timedelta(hours=7)))
    assert default_target_date(now) == dt.date(2026, 4, 25)


def test_tui_parse_target_date_accepts_common_forms() -> None:
    assert parse_target_date("2026-04-25") == dt.date(2026, 4, 25)
    assert parse_target_date("2026-4-5") == dt.date(2026, 4, 5)
    assert parse_target_date("2026/04/25") == dt.date(2026, 4, 25)


def test_tui_parse_target_date_rejects_bad_values() -> None:
    for value in ["", "04-25-2026", "2026-13-25"]:
        try:
            parse_target_date(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {value!r}")


def test_tui_mouse_tracking_is_opt_in() -> None:
    assert not parse_tui_args([]).enable_mouse
    assert parse_tui_args(["--enable-mouse"]).enable_mouse


def test_tui_ctrl_q_quits_from_input_focus() -> None:
    from textual.widgets import Input

    from experiments.no_hrrr_model.no_hrrr_model.tui import NoHrrrInferenceTui

    async def run_check() -> None:
        app = NoHrrrInferenceTui()
        async with app.run_test() as pilot:
            app.query_one("#target_date", Input).focus()
            await pilot.press("ctrl+q")
            await pilot.pause()
            assert app._exit

    asyncio.run(run_check())


def test_tui_prediction_summary_formats_bins_and_tmax() -> None:
    payload = {
        "status": "ok",
        "target_date_local": "2026-04-24",
        "station_id": "KLGA",
        "expected_final_tmax_f": 64.765,
        "anchor_tmax_f": 64.43,
        "final_tmax_quantiles_f": {"0.05": 61.94, "0.5": 64.91, "0.95": 67.90},
        "event_bins": [{"bin": "64-65F", "probability": 0.4287}],
        "_prediction_path": "prediction.json",
    }
    text = format_prediction_summary(payload)
    assert "Expected final high: 64.77 F" in text
    assert "q50=64.91" in text
    assert "64-65F: 0.4287" in text
    assert "prediction.json" in text


def test_tui_failure_message_reads_manifest(tmp_path) -> None:
    manifest = tmp_path / "online_inference.manifest.json"
    manifest.write_text('{"message": "LAMP unavailable"}')
    assert failure_message(manifest, 1) == "LAMP unavailable"


def test_tui_safe_delete_run_root_only_deletes_tui_runs(tmp_path) -> None:
    run_root = tmp_path / "tui_online_inference" / "target_date_local=2026-04-24__run_id=abc12345"
    run_root.mkdir(parents=True)
    (run_root / "artifact.txt").write_text("x")
    assert safe_delete_run_root(run_root)
    assert not run_root.exists()
    unsafe = tmp_path / "target_date_local=2026-04-24__run_id=abc12345"
    unsafe.mkdir()
    assert not safe_delete_run_root(unsafe)
    assert unsafe.exists()
    custom_parent = tmp_path / "custom_runtime_root"
    custom_run = custom_parent / "target_date_local=2026-04-24__run_id=abc12345"
    custom_run.mkdir(parents=True)
    assert safe_delete_run_root(custom_run, allowed_parent=custom_parent)
    assert not custom_run.exists()
    outside_parent = tmp_path / "outside_runtime_root"
    outside_run = outside_parent / "target_date_local=2026-04-24__run_id=abc12345"
    outside_run.mkdir(parents=True)
    assert not safe_delete_run_root(outside_run, allowed_parent=custom_parent)
    assert outside_run.exists()


def test_online_inference_cleanup_deletes_only_runtime_artifacts(tmp_path) -> None:
    runtime_root = tmp_path / "runtime"
    status_dir = runtime_root / "status" / "target_date_local=2026-04-28"
    artifact_dir = runtime_root / "nbm"
    artifact_file = runtime_root / "prediction_features" / "row.parquet"
    outside_dir = tmp_path / "outside"
    status_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    artifact_file.parent.mkdir(parents=True)
    artifact_file.write_text("x")
    outside_dir.mkdir()

    deleted = cleanup_runtime_artifacts(runtime_root, [artifact_dir, artifact_file])

    assert deleted == [str(artifact_dir), str(artifact_file)]
    assert not artifact_dir.exists()
    assert not artifact_file.exists()
    assert status_dir.exists()

    try:
        cleanup_runtime_artifacts(runtime_root, [outside_dir])
    except SystemExit as exc:
        assert "outside runtime root" in str(exc)
    else:
        raise AssertionError("Expected cleanup outside runtime root to fail")


def test_tui_command_uses_run_scoped_runtime_and_prediction_dirs(tmp_path) -> None:
    config = RunConfig(
        target_date_local=dt.date(2026, 4, 24),
        station_id="KLGA",
        lamp_source="auto",
        max_so_far_f=None,
        runtime_root=tmp_path,
    )
    command = command_for_run(config, tmp_path / "run")
    assert "--runtime-root" in command
    assert str(tmp_path / "run" / "runtime") in command
    assert "--prediction-output-dir" in command
    assert str(tmp_path / "run" / "predictions") in command
    assert "--polymarket-event-slug" in command
