from __future__ import annotations

import asyncio
import datetime as dt
import pathlib
import sys

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.no_hrrr_model.no_hrrr_model.build_training_features import build_training_features
from experiments.no_hrrr_model.no_hrrr_model.build_inference_features import filter_label_history_for_inference
from experiments.no_hrrr_model.no_hrrr_model.contracts import audit_training_features
from experiments.no_hrrr_model.no_hrrr_model.distribution import quantiles_to_degree_ladder
from experiments.no_hrrr_model.no_hrrr_model.event_bins import EventBin, load_event_bin_labels, map_ladder_to_bins, parse_event_bin
from experiments.no_hrrr_model.no_hrrr_model.normalize_features import normalize_features
from experiments.no_hrrr_model.no_hrrr_model.polymarket_event import extract_event_bins, weather_event_slug_for_date
from experiments.no_hrrr_model.no_hrrr_model.run_online_inference import resolve_lamp_source
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
