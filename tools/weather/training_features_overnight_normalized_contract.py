from __future__ import annotations

from tools.weather.training_features_overnight_contract import REGISTRY_ROWS as SOURCE_REGISTRY_ROWS


ALLOWED_PREFIXES: tuple[str, ...] = ("label_", "meta_", "wu_", "nbm_", "lamp_", "hrrr_")
NORMALIZATION_VERSION = "overnight-v1"


def registry_row(
    column_name: str,
    *,
    source_column: str,
    block: str,
    role: str,
    dtype: str,
    nullable: bool,
    freeze_level: str,
    model_input_default: bool,
    description: str,
    transform: str = "copy",
    units: str | None = None,
) -> dict[str, object]:
    return {
        "column_name": column_name,
        "source_column": source_column,
        "block": block,
        "role": role,
        "dtype": dtype,
        "nullable": nullable,
        "freeze_level": freeze_level,
        "model_input_default": model_input_default,
        "description": description,
        "transform": transform,
        "units": units,
    }


def _replace_unit_suffix(name: str, old: str, new: str) -> str:
    if name.endswith(old):
        return f"{name[:-len(old)]}{new}"
    return name.replace(old, new)


def _normalize_wu_name(name: str) -> tuple[str, str, str | None]:
    explicit = {
        "wu_last_cloud_cover_code": ("wu_last_cloud_cover_id", "cloud_cover_id", None),
        "wu_last_wx_phrase": ("wu_last_weather_family_id", "weather_family_id", None),
        "wu_last_visibility": ("wu_last_visibility_mi", "copy_numeric", "mi"),
        "wu_last_pressure_in": ("wu_last_pressure_inhg", "copy_numeric", "inhg"),
        "wu_pressure_change_3h": ("wu_pressure_change_3h_inhg", "copy_numeric", "inhg"),
    }
    if name in explicit:
        return explicit[name]
    return name, "copy_numeric", _units_for_name(name)


def _normalize_lamp_name(name: str) -> tuple[str, str, str | None]:
    replacements = {
        "lamp_day_tmp_max_f_forecast": ("lamp_day_temp_max_f", "copy_numeric", "f"),
        "lamp_day_tmp_min_f_forecast": ("lamp_day_temp_min_f", "copy_numeric", "f"),
        "lamp_day_tmp_range_f_forecast": ("lamp_day_temp_range_f", "copy_numeric", "f"),
        "lamp_day_tmp_argmax_local_hour": ("lamp_day_temp_argmax_local_hour", "copy_numeric", "hour"),
        "lamp_morning_cld_mode": ("lamp_morning_cloud_cover_mode_id", "cloud_cover_id", None),
        "lamp_morning_cig_min_hundreds_ft": ("lamp_morning_ceiling_min_ft", "hundreds_ft_to_ft", "ft"),
        "lamp_morning_vis_min_miles": ("lamp_morning_visibility_min_mi", "copy_numeric", "mi"),
        "lamp_day_precip_type_mode": ("lamp_day_precip_type_mode_id", "precip_type_id", None),
        "lamp_rev_day_tmp_max_f": ("lamp_day_temp_max_f_rev", "copy_numeric", "f"),
        "lamp_rev_morning_cig_min_hundreds_ft": ("lamp_morning_ceiling_min_ft_rev", "hundreds_ft_to_ft", "ft"),
        "lamp_rev_morning_vis_min_miles": ("lamp_morning_visibility_min_mi_rev", "copy_numeric", "mi"),
    }
    if name in replacements:
        return replacements[name]
    checkpoint_patterns = (
        ("lamp_tmp_f_at_", "lamp_temp_", "_local_f", "copy_numeric", "f"),
        ("lamp_dpt_f_at_", "lamp_dewpoint_", "_local_f", "copy_numeric", "f"),
        ("lamp_wsp_kt_at_", "lamp_wind_speed_", "_local_mph", "kt_to_mph", "mph"),
        ("lamp_wdr_deg_at_", "lamp_wind_direction_", "_local_deg", "copy_numeric", "deg"),
        ("lamp_cld_code_at_", "lamp_cloud_cover_", "_local_id", "cloud_cover_id", None),
        ("lamp_cig_hundreds_ft_at_", "lamp_ceiling_", "_local_ft", "hundreds_ft_to_ft", "ft"),
        ("lamp_vis_miles_at_", "lamp_visibility_", "_local_mi", "copy_numeric", "mi"),
        ("lamp_obv_code_at_", "lamp_weather_", "_local_id", "weather_family_id", None),
        ("lamp_typ_code_at_", "lamp_precip_type_", "_local_id", "precip_type_id", None),
        ("lamp_rev_tmp_f_at_", "lamp_temp_", "_local_f_rev", "copy_numeric", "f"),
    )
    for prefix, out_prefix, out_suffix, transform, units in checkpoint_patterns:
        if name.startswith(prefix):
            hour = name.removeprefix(prefix)
            return f"{out_prefix}{hour}{out_suffix}", transform, units
    if name.endswith("_hundreds_ft"):
        return name.replace("_hundreds_ft", "_ft"), "hundreds_ft_to_ft", "ft"
    if name.endswith("_miles"):
        return name.replace("_miles", "_mi"), "copy_numeric", "mi"
    return name, "copy_numeric", _units_for_name(name)


def _units_for_name(name: str) -> str | None:
    if name.endswith("_f"):
        return "f"
    if name.endswith("_pct"):
        return "pct"
    if name.endswith("_deg"):
        return "deg"
    if name.endswith("_w_m2"):
        return "w_m2"
    if name.endswith("_j_kg"):
        return "j_kg"
    if name.endswith("_kg_m2"):
        return "kg_m2"
    if name.endswith("_m"):
        return "m"
    if name.endswith("_in"):
        return "in"
    if name.endswith("_inhg"):
        return "inhg"
    if name.endswith("_ft"):
        return "ft"
    if name.endswith("_mi"):
        return "mi"
    if name.endswith("_mph"):
        return "mph"
    if name.endswith("_hour"):
        return "hour"
    return None


def _normalize_generic_name(name: str) -> tuple[str, str, str | None]:
    if ("_rev_" in name or "minus_" in name) and name.endswith("_k"):
        return _replace_unit_suffix(name, "_k", "_f"), "temp_delta_k_to_f", "f"
    if ("_rev_" in name or "minus_" in name) and "_k_" in name and ("temp_" in name or "dewpoint_" in name):
        return name.replace("_k_", "_f_"), "temp_delta_k_to_f", "f"
    if name.endswith("_k"):
        return _replace_unit_suffix(name, "_k", "_f"), "k_to_f", "f"
    if "_k_" in name and ("temp_" in name or "dewpoint_" in name):
        return name.replace("_k_", "_f_"), "k_to_f", "f"
    if name.endswith("_ms"):
        return _replace_unit_suffix(name, "_ms", "_mph"), "ms_to_mph", "mph"
    if name.endswith("_kg_m2") and "apcp" in name:
        return _replace_unit_suffix(name, "_kg_m2", "_in"), "kg_m2_to_in", "in"
    if name.endswith("_m") and "visibility" in name:
        return _replace_unit_suffix(name, "_m", "_mi"), "m_to_mi", "mi"
    if name.endswith("_m") and "ceiling" in name:
        return _replace_unit_suffix(name, "_m", "_ft"), "m_to_ft", "ft"
    return name, "copy_numeric", _units_for_name(name)


def normalized_feature_row(source_row: dict[str, object]) -> dict[str, object]:
    source_name = str(source_row["column_name"])
    dtype = str(source_row["dtype"])
    block = str(source_row["block"])
    description = str(source_row["description"])

    if block == "wu":
        output_name, transform, units = _normalize_wu_name(source_name)
    elif block == "lamp":
        output_name, transform, units = _normalize_lamp_name(source_name)
    else:
        output_name, transform, units = _normalize_generic_name(source_name)

    output_dtype = "int64" if transform.endswith("_id") else dtype
    normalized_description = description if units is None else f"{description} Normalized units={units}."
    return registry_row(
        output_name,
        source_column=source_name,
        block=block,
        role=str(source_row["role"]),
        dtype=output_dtype,
        nullable=bool(source_row["nullable"]),
        freeze_level=str(source_row["freeze_level"]),
        model_input_default=bool(source_row["model_input_default"]),
        description=normalized_description,
        transform=transform,
        units=units,
    )


REGISTRY_ROWS: list[dict[str, object]] = []

for source_row in SOURCE_REGISTRY_ROWS:
    role = str(source_row["role"])
    source_name = str(source_row["column_name"])
    if role in {"identity", "label", "metadata"}:
        REGISTRY_ROWS.append(
            registry_row(
                source_name,
                source_column=source_name,
                block=str(source_row["block"]),
                role=role,
                dtype=str(source_row["dtype"]),
                nullable=bool(source_row["nullable"]),
                freeze_level=str(source_row["freeze_level"]),
                model_input_default=bool(source_row["model_input_default"]),
                description=str(source_row["description"]),
                transform="copy",
                units=_units_for_name(source_name),
            )
        )
        continue
    REGISTRY_ROWS.append(normalized_feature_row(source_row))


def registry_columns() -> list[str]:
    return [str(row["column_name"]) for row in REGISTRY_ROWS]


def registry_by_name() -> dict[str, dict[str, object]]:
    return {str(row["column_name"]): row for row in REGISTRY_ROWS}


def registry_by_source_column() -> dict[str, dict[str, object]]:
    return {str(row["source_column"]): row for row in REGISTRY_ROWS}
