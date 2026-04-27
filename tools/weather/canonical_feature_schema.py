from __future__ import annotations

from typing import Iterable


RUNTIME_IDENTITY_COLUMNS: tuple[str, ...] = (
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "init_time_utc",
    "valid_time_utc",
    "init_time_local",
    "valid_time_local",
    "init_date_local",
    "valid_date_local",
    "forecast_hour",
)

RUNTIME_SPATIAL_COLUMNS: tuple[str, ...] = (
    "settlement_lat",
    "settlement_lon",
    "crop_top_lat",
    "crop_bottom_lat",
    "crop_left_lon",
    "crop_right_lon",
    "nearest_grid_lat",
    "nearest_grid_lon",
)

RUNTIME_METADATA_COLUMNS: tuple[str, ...] = (
    "fallback_used_any",
    "missing_optional_any",
    "missing_optional_fields_count",
)

OPTIONAL_PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "mode",
    "task_key",
    "run_date_utc",
    "cycle_hour_utc",
    "lead_hour",
    "requested_lead_hour",
    "target_date_local",
    "slice_policy",
    "init_hour_local",
    "valid_hour_local",
    "cycle_rank_desc",
    "selected_for_summary",
    "anchor_cycle_candidate",
)

SPATIAL_STAT_SUFFIXES: tuple[str, ...] = (
    "",
    "_crop_mean",
    "_crop_min",
    "_crop_max",
    "_crop_std",
    "_nb3_mean",
    "_nb3_min",
    "_nb3_max",
    "_nb3_std",
    "_nb3_gradient_west_east",
    "_nb3_gradient_south_north",
    "_nb7_mean",
    "_nb7_min",
    "_nb7_max",
    "_nb7_std",
    "_nb7_gradient_west_east",
    "_nb7_gradient_south_north",
)

PRESSURE_LEVELS: tuple[int, ...] = (1000, 925, 850, 700)

CANONICAL_SPATIAL_BASES: tuple[str, ...] = (
    "temp_2m_k",
    "dewpoint_2m_k",
    "rh_2m_pct",
    "temp_2m_f",
    "dewpoint_2m_f",
    "tmax_2m_k",
    "tmin_2m_k",
    "wind_10m_u_ms",
    "wind_10m_v_ms",
    "wind_10m_speed_ms",
    "wind_10m_direction_deg",
    "gust_10m_ms",
    "pressure_surface_pa",
    "mslp_pa",
    "visibility_m",
    "ceiling_m",
    "tcdc_pct",
    "lcdc_pct",
    "mcdc_pct",
    "hcdc_pct",
    "dswrf_surface_w_m2",
    "dlwrf_surface_w_m2",
    "apcp_surface_kg_m2",
    "prate_surface_kg_m2_s",
    "pcpdur_surface_h",
    "cape_surface_j_kg",
    "cin_surface_j_kg",
    "hpbl_m",
    "pwat_kg_m2",
    "vrate_atmosphere_m2_s",
    *(f"temp_{level}mb_k" for level in PRESSURE_LEVELS),
    *(f"rh_{level}mb_pct" for level in PRESSURE_LEVELS),
    *(f"wind_{level}mb_u_ms" for level in PRESSURE_LEVELS),
    *(f"wind_{level}mb_v_ms" for level in PRESSURE_LEVELS),
    *(f"spfh_{level}mb_kg_kg" for level in PRESSURE_LEVELS),
    *(f"hgt_{level}mb_gpm" for level in PRESSURE_LEVELS),
)

CANONICAL_NEAREST_ONLY_BASES: tuple[str, ...] = (
    "ptype_code",
    "pwther_code",
    "thunc_code",
    "tstm_prob_pct",
)


def spatial_family_columns(base_name: str) -> list[str]:
    return [f"{base_name}{suffix}" for suffix in SPATIAL_STAT_SUFFIXES]


CANONICAL_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    column
    for base_name in CANONICAL_SPATIAL_BASES
    for column in spatial_family_columns(base_name)
) + CANONICAL_NEAREST_ONLY_BASES

CANONICAL_WIDE_COLUMNS: tuple[str, ...] = (
    RUNTIME_IDENTITY_COLUMNS
    + RUNTIME_SPATIAL_COLUMNS
    + RUNTIME_METADATA_COLUMNS
    + CANONICAL_FEATURE_COLUMNS
)

CANONICAL_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_model",
    "source_product",
    "source_version",
    "station_id",
    "init_time_utc",
    "init_time_local",
    "init_date_local",
    "valid_time_utc",
    "valid_time_local",
    "valid_date_local",
    "forecast_hour",
    "nearest_grid_lat",
    "nearest_grid_lon",
    "feature_name",
    "raw_feature_name",
    "present_directly",
    "derived",
    "missing_optional",
    "derivation_method",
    "source_feature_names",
    "fallback_used",
    "fallback_source_description",
    "grib_short_name",
    "grib_level_text",
    "grib_type_of_level",
    "grib_step_type",
    "grib_step_text",
    "inventory_line",
    "units",
    "notes",
)

NBM_TO_CANONICAL_BASES: dict[str, str] = {
    "tmp": "temp_2m_k",
    "dpt": "dewpoint_2m_k",
    "rh": "rh_2m_pct",
    "tmax": "tmax_2m_k",
    "tmin": "tmin_2m_k",
    "u10": "wind_10m_u_ms",
    "v10": "wind_10m_v_ms",
    "wind": "wind_10m_speed_ms",
    "wdir": "wind_10m_direction_deg",
    "gust": "gust_10m_ms",
    "tcdc": "tcdc_pct",
    "dswrf": "dswrf_surface_w_m2",
    "apcp": "apcp_surface_kg_m2",
    "vrate": "vrate_atmosphere_m2_s",
    "pcpdur": "pcpdur_surface_h",
    "vis": "visibility_m",
    "ceil": "ceiling_m",
    "cape": "cape_surface_j_kg",
    "thunc": "thunc_code",
    "ptype": "ptype_code",
    "pwther": "pwther_code",
    "tstm": "tstm_prob_pct",
}

HRRR_TO_CANONICAL_BASES: dict[str, str] = {
    "tmp_2m_k": "temp_2m_k",
    "dpt_2m_k": "dewpoint_2m_k",
    "rh_2m_pct": "rh_2m_pct",
    "tmp_2m_f": "temp_2m_f",
    "dpt_2m_f": "dewpoint_2m_f",
    "ugrd_10m_ms": "wind_10m_u_ms",
    "vgrd_10m_ms": "wind_10m_v_ms",
    "wind_10m_speed_ms": "wind_10m_speed_ms",
    "wind_10m_direction_deg": "wind_10m_direction_deg",
    "gust_surface_ms": "gust_10m_ms",
    "surface_pressure_pa": "pressure_surface_pa",
    "mslma_pa": "mslp_pa",
    "visibility_m": "visibility_m",
    "lcdc_low_pct": "lcdc_pct",
    "mcdc_mid_pct": "mcdc_pct",
    "hcdc_high_pct": "hcdc_pct",
    "tcdc_entire_pct": "tcdc_pct",
    "dswrf_surface_w_m2": "dswrf_surface_w_m2",
    "dlwrf_surface_w_m2": "dlwrf_surface_w_m2",
    "apcp_surface_kg_m2": "apcp_surface_kg_m2",
    "prate_surface_kg_m2_s": "prate_surface_kg_m2_s",
    "hpbl_m": "hpbl_m",
    "pwat_entire_atmosphere_kg_m2": "pwat_kg_m2",
    "cape_surface_j_kg": "cape_surface_j_kg",
    "cin_surface_j_kg": "cin_surface_j_kg",
}
for _level in PRESSURE_LEVELS:
    HRRR_TO_CANONICAL_BASES.update(
        {
            f"tmp_{_level}mb_k": f"temp_{_level}mb_k",
            f"rh_{_level}mb_pct": f"rh_{_level}mb_pct",
            f"ugrd_{_level}mb_ms": f"wind_{_level}mb_u_ms",
            f"vgrd_{_level}mb_ms": f"wind_{_level}mb_v_ms",
            f"spfh_{_level}mb_kg_kg": f"spfh_{_level}mb_kg_kg",
            f"hgt_{_level}mb_gpm": f"hgt_{_level}mb_gpm",
        }
    )

SOURCE_TO_CANONICAL_BASES: dict[str, dict[str, str]] = {
    "nbm": NBM_TO_CANONICAL_BASES,
    "hrrr": HRRR_TO_CANONICAL_BASES,
    "lamp": {
        "tmp": "temp_2m_k",
        "dpt": "dewpoint_2m_k",
        "wdr": "wind_10m_direction_deg",
        "wsp": "wind_10m_speed_ms",
        "wgs": "gust_10m_ms",
        "cig": "ceiling_m",
        "vis": "visibility_m",
        "typ": "ptype_code",
    },
}


def source_base_mapping(source: str) -> dict[str, str]:
    key = source.strip().lower()
    if key not in SOURCE_TO_CANONICAL_BASES:
        raise KeyError(f"Unsupported weather source '{source}'.")
    return SOURCE_TO_CANONICAL_BASES[key]


def canonical_base_for(source: str, raw_base: str) -> str | None:
    return source_base_mapping(source).get(raw_base)


def canonical_wide_columns(*, passthrough_columns: Iterable[str] = OPTIONAL_PASSTHROUGH_COLUMNS) -> list[str]:
    ordered: list[str] = list(CANONICAL_WIDE_COLUMNS)
    for column in passthrough_columns:
        if column and column not in ordered:
            ordered.append(column)
    return ordered
