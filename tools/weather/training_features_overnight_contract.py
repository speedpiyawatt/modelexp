from __future__ import annotations


CHECKPOINT_HOURS: tuple[int, ...] = (6, 9, 12, 15, 18, 21)
ALLOWED_PREFIXES: tuple[str, ...] = ("label_", "meta_", "wu_", "nbm_", "lamp_", "hrrr_")


def registry_row(
    column_name: str,
    *,
    block: str,
    role: str,
    dtype: str,
    nullable: bool,
    freeze_level: str,
    model_input_default: bool,
    description: str,
) -> dict[str, object]:
    return {
        "column_name": column_name,
        "block": block,
        "role": role,
        "dtype": dtype,
        "nullable": nullable,
        "freeze_level": freeze_level,
        "model_input_default": model_input_default,
        "description": description,
    }


REGISTRY_ROWS: list[dict[str, object]] = [
    registry_row(
        "target_date_local",
        block="identity",
        role="identity",
        dtype="string",
        nullable=False,
        freeze_level="core_required",
        model_input_default=False,
        description="KLGA local calendar day for the overnight training row.",
    ),
    registry_row(
        "station_id",
        block="identity",
        role="identity",
        dtype="string",
        nullable=False,
        freeze_level="core_required",
        model_input_default=False,
        description="Settlement station identifier.",
    ),
    registry_row(
        "selection_cutoff_local",
        block="identity",
        role="identity",
        dtype="string",
        nullable=False,
        freeze_level="core_required",
        model_input_default=False,
        description="As-of cutoff in America/New_York used to select overnight source state.",
    ),
]

REGISTRY_ROWS.extend(
    [
        registry_row("label_final_tmax_f", block="label", role="label", dtype="float64", nullable=False, freeze_level="core_required", model_input_default=False, description="Settlement-aligned final KLGA Tmax in Fahrenheit."),
        registry_row("label_final_tmin_f", block="label", role="label", dtype="float64", nullable=False, freeze_level="core_required", model_input_default=False, description="Settlement-aligned final KLGA Tmin in Fahrenheit."),
        registry_row("label_market_bin", block="label", role="label", dtype="string", nullable=False, freeze_level="core_required", model_input_default=False, description="Exact-degree proxy bin derived from the finalized daily Tmax."),
        registry_row("label_obs_count", block="label", role="label", dtype="int64", nullable=False, freeze_level="core_required", model_input_default=False, description="Observation count used to reconstruct the daily label."),
        registry_row("label_first_obs_time_local", block="label", role="label", dtype="string", nullable=False, freeze_level="core_required", model_input_default=False, description="First local observation timestamp retained for the label day."),
        registry_row("label_last_obs_time_local", block="label", role="label", dtype="string", nullable=False, freeze_level="core_required", model_input_default=False, description="Last local observation timestamp retained for the label day."),
        registry_row("label_total_precip_in", block="label", role="label", dtype="float64", nullable=False, freeze_level="core_required", model_input_default=False, description="Daily summed Wunderground hourly precipitation for the label day."),
    ]
)

for name, dtype, model_input, description in (
    ("meta_wu_obs_available", "bool", True, "Whether a Wunderground observation existed at or before the overnight cutoff."),
    ("meta_wu_last_obs_time_local", "string", False, "Latest Wunderground observation timestamp at or before the overnight cutoff."),
    ("meta_nbm_available", "bool", True, "Whether an NBM overnight-daily row was available for the target date."),
    ("meta_nbm_source_model", "string", False, "NBM source model name."),
    ("meta_nbm_source_product", "string", False, "NBM source product identifier."),
    ("meta_nbm_source_version", "string", False, "NBM source version identifier."),
    ("meta_nbm_selected_init_time_utc", "string", False, "UTC init time of the selected NBM issue."),
    ("meta_nbm_selected_init_time_local", "string", False, "Local init time of the selected NBM issue."),
    ("meta_nbm_selected_issue_age_minutes", "float64", False, "Minutes between the overnight cutoff and the selected NBM issue time."),
    ("meta_nbm_target_day_row_count", "int64", True, "Target-day hourly row count retained for the selected NBM issue."),
    ("meta_nbm_missing_checkpoint_count", "int64", True, "Missing required checkpoint count in the selected NBM overnight-daily row."),
    ("meta_nbm_missing_required_feature_count", "int64", True, "Missing native required NBM guidance feature count in the selected NBM overnight-daily row."),
    ("meta_nbm_coverage_complete", "bool", True, "Whether the selected NBM issue populated all required overnight checkpoints."),
    ("meta_lamp_available", "bool", True, "Whether a LAMP overnight-daily row was available for the target date."),
    ("meta_lamp_source_model", "string", False, "LAMP source model name."),
    ("meta_lamp_source_product", "string", False, "LAMP source product identifier."),
    ("meta_lamp_source_version", "string", False, "LAMP source version identifier."),
    ("meta_lamp_selected_init_time_utc", "string", False, "UTC init time of the selected LAMP issue."),
    ("meta_lamp_selected_init_time_local", "string", False, "Local init time of the selected LAMP issue."),
    ("meta_lamp_previous_init_time_utc", "string", False, "UTC init time of the previous LAMP issue used for revisions."),
    ("meta_lamp_previous_init_time_local", "string", False, "Local init time of the previous LAMP issue used for revisions."),
    ("meta_lamp_revision_available", "bool", True, "Whether the selected LAMP row includes a previous-issue revision comparison."),
    ("meta_lamp_missing_optional_any", "bool", True, "Whether any optional LAMP labels were missing in the selected issue."),
    ("meta_lamp_missing_optional_fields_count", "int64", True, "Count of optional LAMP labels missing from the selected issue."),
    ("meta_lamp_coverage_complete", "bool", True, "Whether the selected LAMP issue populated all required overnight checkpoints."),
    ("meta_lamp_missing_checkpoint_count", "int64", True, "Missing checkpoint count in the selected LAMP overnight row."),
    ("meta_hrrr_available", "bool", True, "Whether an HRRR overnight summary row was available for the target date."),
    ("meta_hrrr_source_model", "string", False, "HRRR source model name."),
    ("meta_hrrr_source_product", "string", False, "HRRR source product identifier."),
    ("meta_hrrr_source_version", "string", False, "HRRR source version identifier."),
    ("meta_hrrr_anchor_init_time_utc", "string", False, "UTC init time of the anchor HRRR cycle."),
    ("meta_hrrr_anchor_init_time_local", "string", False, "Local init time of the anchor HRRR cycle."),
    ("meta_hrrr_retained_cycle_count", "int64", True, "Number of retained HRRR cycles contributing to the overnight summary."),
    ("meta_hrrr_first_valid_hour_local", "int64", True, "First local valid hour covered by the anchor HRRR cycle."),
    ("meta_hrrr_last_valid_hour_local", "int64", True, "Last local valid hour covered by the anchor HRRR cycle."),
    ("meta_hrrr_covered_hour_count", "int64", True, "Distinct local valid-hour count covered by the anchor HRRR cycle."),
    ("meta_hrrr_covered_checkpoint_count", "int64", True, "Count of canonical overnight checkpoint hours covered by the anchor HRRR cycle."),
    ("meta_hrrr_coverage_end_hour_local", "int64", True, "Latest local valid hour covered by the anchor HRRR cycle."),
    ("meta_hrrr_has_full_day_21_local_coverage", "bool", True, "Whether the anchor HRRR cycle covers the target day through 21 local."),
    ("meta_hrrr_missing_checkpoint_count", "int64", True, "Missing checkpoint count in the anchor HRRR summary."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="metadata",
            role="metadata",
            dtype=dtype,
            nullable=True,
            freeze_level="core_required",
            model_input_default=model_input,
            description=description,
        )
    )

for name, dtype, description in (
    ("wu_last_temp_f", "float64", "Latest observed KLGA temperature before the overnight cutoff."),
    ("wu_last_dewpoint_f", "float64", "Latest observed KLGA dew point before the overnight cutoff."),
    ("wu_last_rh_pct", "float64", "Latest observed KLGA relative humidity before the overnight cutoff."),
    ("wu_last_pressure_in", "float64", "Latest observed KLGA pressure before the overnight cutoff."),
    ("wu_last_wind_speed_mph", "float64", "Latest observed KLGA wind speed before the overnight cutoff."),
    ("wu_last_wind_dir_deg", "float64", "Latest observed KLGA wind direction before the overnight cutoff."),
    ("wu_last_wind_gust_mph", "float64", "Latest observed KLGA wind gust before the overnight cutoff."),
    ("wu_last_visibility", "float64", "Latest observed KLGA visibility before the overnight cutoff."),
    ("wu_last_cloud_cover_code", "string", "Latest observed KLGA cloud cover code before the overnight cutoff."),
    ("wu_last_wx_phrase", "string", "Latest observed KLGA weather phrase before the overnight cutoff."),
    ("wu_last_precip_hrly_in", "float64", "Latest observed KLGA hourly precipitation before the overnight cutoff."),
    ("wu_prev_day_final_tmax_f", "float64", "Previous local day finalized KLGA Tmax."),
    ("wu_prev_day_final_tmin_f", "float64", "Previous local day finalized KLGA Tmin."),
    ("wu_prev_day_total_precip_in", "float64", "Previous local day summed hourly precipitation."),
    ("wu_temp_change_1h_f", "float64", "Temperature change from the latest pre-cutoff observation versus one hour earlier."),
    ("wu_temp_change_3h_f", "float64", "Temperature change from the latest pre-cutoff observation versus three hours earlier."),
    ("wu_temp_change_6h_f", "float64", "Temperature change from the latest pre-cutoff observation versus six hours earlier."),
    ("wu_dewpoint_change_3h_f", "float64", "Dew point change over the three hours preceding the overnight cutoff."),
    ("wu_pressure_change_3h", "float64", "Pressure change over the three hours preceding the overnight cutoff."),
    ("wu_wind_speed_mean_3h", "float64", "Mean observed wind speed over the three hours preceding the overnight cutoff."),
    ("wu_wind_gust_max_6h", "float64", "Maximum observed wind gust over the six hours preceding the overnight cutoff."),
    ("wu_visibility_min_6h", "float64", "Minimum observed visibility over the six hours preceding the overnight cutoff."),
    ("wu_precip_total_6h", "float64", "Summed hourly precipitation over the six hours preceding the overnight cutoff."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="wu",
            role="feature",
            dtype=dtype,
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=description,
        )
    )

for name, description in (
    ("nbm_temp_2m_day_max_k", "Maximum forecast 2m temperature across the target local day from the selected NBM issue."),
    ("nbm_temp_2m_day_mean_k", "Mean forecast 2m temperature across the target local day from the selected NBM issue."),
    ("nbm_native_tmax_2m_day_max_k", "Native NBM maximum 2m temperature guidance across the target local day."),
    ("nbm_native_tmax_2m_nb3_day_max_k", "Maximum native NBM 3x3-neighborhood maximum 2m temperature guidance across the target local day."),
    ("nbm_native_tmax_2m_nb7_day_max_k", "Maximum native NBM 7x7-neighborhood maximum 2m temperature guidance across the target local day."),
    ("nbm_native_tmax_2m_crop_day_max_k", "Maximum native NBM regional-crop maximum 2m temperature guidance across the target local day."),
    ("nbm_native_tmin_2m_day_min_k", "Native NBM minimum 2m temperature guidance across the target local day."),
    ("nbm_native_tmin_2m_nb3_day_min_k", "Minimum native NBM 3x3-neighborhood minimum 2m temperature guidance across the target local day."),
    ("nbm_native_tmin_2m_nb7_day_min_k", "Minimum native NBM 7x7-neighborhood minimum 2m temperature guidance across the target local day."),
    ("nbm_native_tmin_2m_crop_day_min_k", "Minimum native NBM regional-crop minimum 2m temperature guidance across the target local day."),
    ("nbm_gust_10m_day_max_ms", "Maximum forecast 10m gust across the target local day from the selected NBM issue."),
    ("nbm_tcdc_morning_mean_pct", "Mean forecast total cloud cover from 06-12 local from the selected NBM issue."),
    ("nbm_tcdc_day_mean_pct", "Mean forecast total cloud cover across the target local day from the selected NBM issue."),
    ("nbm_dswrf_day_max_w_m2", "Maximum forecast downward shortwave radiation across the target local day from the selected NBM issue."),
    ("nbm_apcp_day_total_kg_m2", "Summed forecast precipitation across the target local day from the selected NBM issue."),
    ("nbm_pcpdur_day_total_h", "Summed native NBM precipitation-duration guidance across the target local day."),
    ("nbm_pcpdur_day_max_h", "Maximum native NBM precipitation-duration guidance across the target local day."),
    ("nbm_pcpdur_morning_total_h", "Summed native NBM precipitation-duration guidance from 06-12 local."),
    ("nbm_visibility_day_min_m", "Minimum forecast visibility across the target local day from the selected NBM issue."),
    ("nbm_ceiling_morning_min_m", "Minimum forecast ceiling from 06-12 local from the selected NBM issue."),
    ("nbm_cape_day_max_j_kg", "Maximum forecast CAPE across the target local day from the selected NBM issue."),
    ("nbm_pwther_code_day_mode", "Mode native NBM present-weather code across the target local day, tie-broken by smaller numeric code."),
    ("nbm_pwther_nonzero_hour_count", "Count of target-day NBM rows with nonzero present-weather code."),
    ("nbm_pwther_any_flag", "Whether any target-day NBM row had nonzero present-weather code."),
    ("nbm_tstm_day_max_pct", "Maximum native NBM thunderstorm-probability guidance across the target local day."),
    ("nbm_tstm_day_mean_pct", "Mean native NBM thunderstorm-probability guidance across the target local day."),
    ("nbm_tstm_any_flag", "Whether any target-day NBM row had nonzero thunderstorm-probability guidance."),
    ("nbm_ptype_code_day_mode", "Mode native NBM precipitation-type code across the target local day, tie-broken by smaller numeric code."),
    ("nbm_ptype_nonzero_hour_count", "Count of target-day NBM rows with nonzero precipitation-type code."),
    ("nbm_ptype_any_flag", "Whether any target-day NBM row had nonzero precipitation-type code."),
    ("nbm_thunc_day_max_code", "Maximum native NBM thunderstorm-coverage code across the target local day."),
    ("nbm_thunc_day_mean_code", "Mean native NBM thunderstorm-coverage code across the target local day."),
    ("nbm_thunc_nonzero_hour_count", "Count of target-day NBM rows with nonzero thunderstorm-coverage code."),
    ("nbm_vrate_day_max", "Maximum native NBM ventilation-rate guidance across the target local day."),
    ("nbm_vrate_day_mean", "Mean native NBM ventilation-rate guidance across the target local day."),
    ("nbm_temp_2m_nb3_day_mean_k", "Mean selected-NBM 3x3 neighborhood temperature across the target day."),
    ("nbm_temp_2m_nb7_day_mean_k", "Mean selected-NBM 7x7 neighborhood temperature across the target day."),
    ("nbm_temp_2m_crop_day_mean_k", "Mean selected-NBM regional crop temperature across the target day."),
    ("nbm_tcdc_crop_day_mean_pct", "Mean selected-NBM regional crop total cloud cover across the target day."),
    ("nbm_dswrf_crop_day_max_w_m2", "Maximum selected-NBM regional crop downward shortwave radiation across the target day."),
    ("nbm_wind_10m_speed_nb7_day_mean_ms", "Mean selected-NBM 7x7 neighborhood wind speed across the target day."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="nbm",
            role="feature",
            dtype="float64",
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=description,
        )
    )

for hour, metric, dtype, description in (
    (6, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (9, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (12, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (15, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (18, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (21, "temp_2m", "float64", "Forecast 2m temperature checkpoint at {hour:02d} local from the selected NBM issue."),
    (9, "dewpoint_2m", "float64", "Forecast 2m dew point checkpoint at {hour:02d} local from the selected NBM issue."),
    (15, "dewpoint_2m", "float64", "Forecast 2m dew point checkpoint at {hour:02d} local from the selected NBM issue."),
    (9, "rh_2m", "float64", "Forecast 2m relative humidity checkpoint at {hour:02d} local from the selected NBM issue."),
    (15, "rh_2m", "float64", "Forecast 2m relative humidity checkpoint at {hour:02d} local from the selected NBM issue."),
    (9, "wind_10m_speed", "float64", "Forecast 10m wind speed checkpoint at {hour:02d} local from the selected NBM issue."),
    (15, "wind_10m_speed", "float64", "Forecast 10m wind speed checkpoint at {hour:02d} local from the selected NBM issue."),
    (9, "wind_10m_direction", "float64", "Forecast 10m wind direction checkpoint at {hour:02d} local from the selected NBM issue."),
    (15, "wind_10m_direction", "float64", "Forecast 10m wind direction checkpoint at {hour:02d} local from the selected NBM issue."),
):
    suffix = "k" if "temp" in metric or "dewpoint" in metric else "pct" if "rh" in metric else "ms" if "speed" in metric else "deg"
    REGISTRY_ROWS.append(
        registry_row(
            f"nbm_{metric}_{hour:02d}_local_{suffix}",
            block="nbm",
            role="feature",
            dtype=dtype,
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=description.format(hour=hour),
        )
    )

LAMP_CHECKPOINT_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("tmp_f_at_{hour:02d}", "float64", "lamp", "Temperature forecast checkpoint in Fahrenheit."),
    ("dpt_f_at_{hour:02d}", "float64", "lamp", "Dew point forecast checkpoint in Fahrenheit."),
    ("wsp_kt_at_{hour:02d}", "float64", "lamp", "Wind speed forecast checkpoint in knots."),
    ("wdr_deg_at_{hour:02d}", "float64", "lamp", "Wind direction forecast checkpoint in degrees."),
    ("cld_code_at_{hour:02d}", "string", "lamp", "Cloud cover code checkpoint."),
    ("cig_hundreds_ft_at_{hour:02d}", "float64", "lamp", "Ceiling checkpoint in hundreds of feet."),
    ("vis_miles_at_{hour:02d}", "float64", "lamp", "Visibility checkpoint in miles."),
    ("obv_code_at_{hour:02d}", "string", "lamp", "Observed weather code checkpoint."),
    ("typ_code_at_{hour:02d}", "string", "lamp", "Precipitation type code checkpoint."),
)

for hour in CHECKPOINT_HOURS:
    for template, dtype, block, description in LAMP_CHECKPOINT_FIELDS:
        column_name = f"lamp_{template.format(hour=hour)}"
        REGISTRY_ROWS.append(
            registry_row(
                column_name,
                block=block,
                role="feature",
                dtype=dtype,
                nullable=True,
                freeze_level="core_required",
                model_input_default=True,
                description=f"{description} Hour={hour:02d} local.",
            )
        )
    REGISTRY_ROWS.append(
        registry_row(
            f"lamp_rev_tmp_f_at_{hour:02d}",
            block="lamp",
            role="feature",
            dtype="float64",
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=f"Selected-minus-previous issue revision for LAMP temperature checkpoint at {hour:02d} local.",
        )
    )

for name, dtype, description in (
    ("lamp_day_tmp_max_f_forecast", "float64", "LAMP daily maximum temperature forecast."),
    ("lamp_day_tmp_min_f_forecast", "float64", "LAMP daily minimum temperature forecast."),
    ("lamp_day_tmp_range_f_forecast", "float64", "LAMP daily temperature range forecast."),
    ("lamp_day_tmp_argmax_local_hour", "int64", "Local hour of the LAMP daily maximum temperature forecast."),
    ("lamp_morning_cld_mode", "string", "Most common LAMP morning cloud cover code."),
    ("lamp_morning_cig_min_hundreds_ft", "float64", "Minimum LAMP morning ceiling in hundreds of feet."),
    ("lamp_morning_vis_min_miles", "float64", "Minimum LAMP morning visibility in miles."),
    ("lamp_morning_obv_any", "bool", "Whether any morning LAMP observation/weather code is present."),
    ("lamp_morning_ifr_like_any", "bool", "Whether any morning LAMP conditions indicate IFR-like constraints."),
    ("lamp_day_p01_max_pct", "float64", "Maximum LAMP hourly precipitation probability."),
    ("lamp_day_p06_max_pct", "float64", "Maximum LAMP 6-hour precipitation probability."),
    ("lamp_day_p12_max_pct", "float64", "Maximum LAMP 12-hour precipitation probability."),
    ("lamp_day_pos_max_pct", "float64", "Maximum LAMP precipitation occurrence probability."),
    ("lamp_day_poz_max_pct", "float64", "Maximum LAMP freezing precipitation occurrence probability."),
    ("lamp_day_precip_type_any", "bool", "Whether any precipitation type code is present in the LAMP day forecast."),
    ("lamp_day_precip_type_mode", "string", "Most common LAMP precipitation type code across the target day."),
    ("lamp_rev_day_tmp_max_f", "float64", "Selected-minus-previous issue revision for LAMP daily maximum temperature."),
    ("lamp_rev_day_p01_max_pct", "float64", "Selected-minus-previous issue revision for maximum hourly precipitation probability."),
    ("lamp_rev_day_pos_max_pct", "float64", "Selected-minus-previous issue revision for precipitation occurrence probability."),
    ("lamp_rev_morning_cig_min_hundreds_ft", "float64", "Selected-minus-previous issue revision for minimum morning ceiling."),
    ("lamp_rev_morning_vis_min_miles", "float64", "Selected-minus-previous issue revision for minimum morning visibility."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="lamp",
            role="feature",
            dtype=dtype,
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=description,
        )
    )

for name, dtype, description in (
    ("hrrr_temp_2m_06_local_k", "float64", "HRRR anchor-cycle 2m temperature at 06 local."),
    ("hrrr_temp_2m_09_local_k", "float64", "HRRR anchor-cycle 2m temperature at 09 local."),
    ("hrrr_temp_2m_12_local_k", "float64", "HRRR anchor-cycle 2m temperature at 12 local."),
    ("hrrr_temp_2m_15_local_k", "float64", "HRRR anchor-cycle 2m temperature at 15 local."),
    ("hrrr_temp_2m_18_local_k", "float64", "HRRR anchor-cycle 2m temperature at 18 local."),
    ("hrrr_dewpoint_2m_06_local_k", "float64", "HRRR anchor-cycle 2m dew point at 06 local."),
    ("hrrr_dewpoint_2m_09_local_k", "float64", "HRRR anchor-cycle 2m dew point at 09 local."),
    ("hrrr_dewpoint_2m_12_local_k", "float64", "HRRR anchor-cycle 2m dew point at 12 local."),
    ("hrrr_dewpoint_2m_15_local_k", "float64", "HRRR anchor-cycle 2m dew point at 15 local."),
    ("hrrr_dewpoint_2m_18_local_k", "float64", "HRRR anchor-cycle 2m dew point at 18 local."),
    ("hrrr_rh_2m_06_local_pct", "float64", "HRRR anchor-cycle 2m relative humidity at 06 local."),
    ("hrrr_rh_2m_09_local_pct", "float64", "HRRR anchor-cycle 2m relative humidity at 09 local."),
    ("hrrr_rh_2m_12_local_pct", "float64", "HRRR anchor-cycle 2m relative humidity at 12 local."),
    ("hrrr_rh_2m_15_local_pct", "float64", "HRRR anchor-cycle 2m relative humidity at 15 local."),
    ("hrrr_rh_2m_18_local_pct", "float64", "HRRR anchor-cycle 2m relative humidity at 18 local."),
    ("hrrr_u10m_09_local_ms", "float64", "HRRR anchor-cycle 10m U-wind component at 09 local."),
    ("hrrr_u10m_12_local_ms", "float64", "HRRR anchor-cycle 10m U-wind component at 12 local."),
    ("hrrr_u10m_15_local_ms", "float64", "HRRR anchor-cycle 10m U-wind component at 15 local."),
    ("hrrr_u10m_18_local_ms", "float64", "HRRR anchor-cycle 10m U-wind component at 18 local."),
    ("hrrr_v10m_09_local_ms", "float64", "HRRR anchor-cycle 10m V-wind component at 09 local."),
    ("hrrr_v10m_12_local_ms", "float64", "HRRR anchor-cycle 10m V-wind component at 12 local."),
    ("hrrr_v10m_15_local_ms", "float64", "HRRR anchor-cycle 10m V-wind component at 15 local."),
    ("hrrr_v10m_18_local_ms", "float64", "HRRR anchor-cycle 10m V-wind component at 18 local."),
    ("hrrr_wind_10m_09_local_speed_ms", "float64", "HRRR anchor-cycle 10m wind speed at 09 local."),
    ("hrrr_wind_10m_12_local_speed_ms", "float64", "HRRR anchor-cycle 10m wind speed at 12 local."),
    ("hrrr_wind_10m_15_local_speed_ms", "float64", "HRRR anchor-cycle 10m wind speed at 15 local."),
    ("hrrr_wind_10m_18_local_speed_ms", "float64", "HRRR anchor-cycle 10m wind speed at 18 local."),
    ("hrrr_wind_10m_09_local_direction_deg", "float64", "HRRR anchor-cycle 10m wind direction at 09 local."),
    ("hrrr_wind_10m_12_local_direction_deg", "float64", "HRRR anchor-cycle 10m wind direction at 12 local."),
    ("hrrr_wind_10m_15_local_direction_deg", "float64", "HRRR anchor-cycle 10m wind direction at 15 local."),
    ("hrrr_wind_10m_18_local_direction_deg", "float64", "HRRR anchor-cycle 10m wind direction at 18 local."),
    ("hrrr_mslp_09_local_pa", "float64", "HRRR anchor-cycle mean sea level pressure at 09 local."),
    ("hrrr_mslp_12_local_pa", "float64", "HRRR anchor-cycle mean sea level pressure at 12 local."),
    ("hrrr_mslp_15_local_pa", "float64", "HRRR anchor-cycle mean sea level pressure at 15 local."),
    ("hrrr_surface_pressure_09_local_pa", "float64", "HRRR anchor-cycle surface pressure at 09 local."),
    ("hrrr_temp_2m_day_max_k", "float64", "HRRR anchor-cycle daily maximum 2m temperature."),
    ("hrrr_temp_2m_day_mean_k", "float64", "HRRR anchor-cycle daily mean 2m temperature."),
    ("hrrr_rh_2m_day_min_pct", "float64", "HRRR anchor-cycle daily minimum 2m relative humidity."),
    ("hrrr_wind_10m_day_max_ms", "float64", "HRRR anchor-cycle daily maximum 10m wind speed."),
    ("hrrr_gust_day_max_ms", "float64", "HRRR anchor-cycle daily maximum gust."),
    ("hrrr_tcdc_day_mean_pct", "float64", "HRRR anchor-cycle daily mean total cloud cover."),
    ("hrrr_tcdc_morning_mean_pct", "float64", "HRRR anchor-cycle morning mean total cloud cover."),
    ("hrrr_tcdc_afternoon_mean_pct", "float64", "HRRR anchor-cycle afternoon mean total cloud cover."),
    ("hrrr_tcdc_day_max_pct", "float64", "HRRR anchor-cycle daily maximum total cloud cover."),
    ("hrrr_lcdc_morning_mean_pct", "float64", "HRRR anchor-cycle morning mean low-cloud cover."),
    ("hrrr_mcdc_day_mean_pct", "float64", "HRRR anchor-cycle daily mean mid-cloud cover."),
    ("hrrr_hcdc_day_mean_pct", "float64", "HRRR anchor-cycle daily mean high-cloud cover."),
    ("hrrr_mcdc_afternoon_mean_pct", "float64", "HRRR anchor-cycle afternoon mean mid-cloud cover."),
    ("hrrr_hcdc_afternoon_mean_pct", "float64", "HRRR anchor-cycle afternoon mean high-cloud cover."),
    ("hrrr_dswrf_day_max_w_m2", "float64", "HRRR anchor-cycle daily maximum downward shortwave radiation."),
    ("hrrr_dlwrf_night_mean_w_m2", "float64", "HRRR anchor-cycle night mean downward longwave radiation."),
    ("hrrr_apcp_day_total_kg_m2", "float64", "HRRR anchor-cycle daily total precipitation."),
    ("hrrr_cape_day_max_j_kg", "float64", "HRRR anchor-cycle daily maximum surface CAPE."),
    ("hrrr_cape_afternoon_max_j_kg", "float64", "HRRR anchor-cycle afternoon maximum surface CAPE."),
    ("hrrr_cin_day_min_j_kg", "float64", "HRRR anchor-cycle daily minimum surface CIN."),
    ("hrrr_cin_afternoon_min_j_kg", "float64", "HRRR anchor-cycle afternoon minimum surface CIN."),
    ("hrrr_refc_day_max", "float64", "HRRR anchor-cycle daily maximum composite reflectivity."),
    ("hrrr_ltng_day_max", "float64", "HRRR anchor-cycle daily maximum lightning signal."),
    ("hrrr_ltng_day_any", "boolean", "HRRR anchor-cycle daily lightning-any flag with missing propagation."),
    ("hrrr_mslp_day_mean_pa", "float64", "HRRR anchor-cycle daily mean mean sea level pressure."),
    ("hrrr_pwat_day_mean_kg_m2", "float64", "HRRR anchor-cycle daily mean precipitable water."),
    ("hrrr_hpbl_day_max_m", "float64", "HRRR anchor-cycle daily maximum planetary boundary layer height."),
    ("hrrr_temp_1000mb_day_mean_k", "float64", "HRRR anchor-cycle daily mean 1000 mb temperature."),
    ("hrrr_temp_925mb_day_mean_k", "float64", "HRRR anchor-cycle daily mean 925 mb temperature."),
    ("hrrr_temp_850mb_day_mean_k", "float64", "HRRR anchor-cycle daily mean 850 mb temperature."),
    ("hrrr_rh_925mb_day_mean_pct", "float64", "HRRR anchor-cycle daily mean 925 mb relative humidity."),
    ("hrrr_u925_day_mean_ms", "float64", "HRRR anchor-cycle daily mean 925 mb U-wind component."),
    ("hrrr_v925_day_mean_ms", "float64", "HRRR anchor-cycle daily mean 925 mb V-wind component."),
    ("hrrr_u850_day_mean_ms", "float64", "HRRR anchor-cycle daily mean 850 mb U-wind component."),
    ("hrrr_v850_day_mean_ms", "float64", "HRRR anchor-cycle daily mean 850 mb V-wind component."),
    ("hrrr_hgt_925_day_mean_gpm", "float64", "HRRR anchor-cycle daily mean 925 mb geopotential height."),
    ("hrrr_hgt_700_day_mean_gpm", "float64", "HRRR anchor-cycle daily mean 700 mb geopotential height."),
    ("hrrr_temp_2m_k_nb3_day_mean", "float64", "HRRR anchor-cycle daily mean 3x3 neighborhood 2m temperature."),
    ("hrrr_temp_2m_k_crop_day_mean", "float64", "HRRR anchor-cycle daily mean regional crop 2m temperature."),
    ("hrrr_tcdc_entire_pct_crop_day_mean", "float64", "HRRR anchor-cycle daily mean regional crop total cloud cover."),
    ("hrrr_dswrf_surface_w_m2_crop_day_max", "float64", "HRRR anchor-cycle daily maximum regional crop downward shortwave radiation."),
    ("hrrr_pwat_entire_atmosphere_kg_m2_nb7_day_mean", "float64", "HRRR anchor-cycle daily mean 7x7 neighborhood precipitable water."),
    ("hrrr_wind_850mb_speed_day_mean_ms", "float64", "HRRR anchor-cycle daily mean 850 mb wind speed."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="hrrr",
            role="feature",
            dtype=dtype,
            nullable=True,
            freeze_level="core_required",
            model_input_default=True,
            description=description,
        )
    )

for lag in (1, 2, 3):
    for base_name, description in (
        ("hrrr_temp_2m_day_max_k", "Selected-minus-prior-cycle revision for HRRR daily maximum 2m temperature."),
        ("hrrr_temp_2m_09_local_k", "Selected-minus-prior-cycle revision for the 09 local 2m temperature checkpoint."),
        ("hrrr_temp_2m_12_local_k", "Selected-minus-prior-cycle revision for the 12 local 2m temperature checkpoint."),
        ("hrrr_temp_2m_15_local_k", "Selected-minus-prior-cycle revision for the 15 local 2m temperature checkpoint."),
        ("hrrr_tcdc_day_mean_pct", "Selected-minus-prior-cycle revision for daily mean total cloud cover."),
        ("hrrr_dswrf_day_max_w_m2", "Selected-minus-prior-cycle revision for daily maximum downward shortwave radiation."),
        ("hrrr_pwat_day_mean_kg_m2", "Selected-minus-prior-cycle revision for daily mean precipitable water."),
        ("hrrr_hpbl_day_max_m", "Selected-minus-prior-cycle revision for daily maximum boundary layer height."),
        ("hrrr_mslp_day_mean_pa", "Selected-minus-prior-cycle revision for daily mean mean sea level pressure."),
    ):
        REGISTRY_ROWS.append(
            registry_row(
                f"{base_name}_rev_{lag}cycle",
                block="hrrr",
                role="feature",
                dtype="float64",
                nullable=True,
                freeze_level="core_required",
                model_input_default=True,
                description=f"{description} Lag={lag} cycle.",
            )
        )

for name, dtype, description in (
    ("nbm_minus_lamp_day_max_f", "float64", "NBM-minus-LAMP difference between daily maximum temperature forecasts after converting NBM Tmax to Fahrenheit."),
    ("nbm_minus_hrrr_day_max_k", "float64", "NBM-minus-HRRR difference between daily maximum temperature forecasts in Kelvin."),
):
    REGISTRY_ROWS.append(
        registry_row(
            name,
            block="nbm",
            role="feature",
            dtype=dtype,
            nullable=True,
            freeze_level="optional_candidate",
            model_input_default=False,
            description=description,
        )
    )


def registry_columns() -> list[str]:
    return [str(row["column_name"]) for row in REGISTRY_ROWS]


def registry_by_name() -> dict[str, dict[str, object]]:
    return {str(row["column_name"]): row for row in REGISTRY_ROWS}
