from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    column: str
    variable: str
    level: str
    description: str
    unit: str
    mode: str = "instant"


SURFACE_FIELD_SPECS = [
    FieldSpec("visibility_m", "VIS", "surface", "surface visibility", "m"),
    FieldSpec("gust_surface_ms", "GUST", "surface", "surface wind gust", "m s-1"),
    FieldSpec("surface_pressure_pa", "PRES", "surface", "surface pressure", "Pa"),
    FieldSpec("tmp_2m_k", "TMP", "2 m above ground", "2 m temperature", "K"),
    FieldSpec("dpt_2m_k", "DPT", "2 m above ground", "2 m dew point", "K"),
    FieldSpec("rh_2m_pct", "RH", "2 m above ground", "2 m relative humidity", "%"),
    FieldSpec("ugrd_10m_ms", "UGRD", "10 m above ground", "10 m u wind", "m s-1"),
    FieldSpec("vgrd_10m_ms", "VGRD", "10 m above ground", "10 m v wind", "m s-1"),
    FieldSpec("prate_surface_kg_m2_s", "PRATE", "surface", "precipitation rate", "kg m-2 s-1"),
    FieldSpec("apcp_surface_kg_m2", "APCP", "surface", "accumulated precipitation", "kg m-2", mode="accum"),
    FieldSpec("cape_surface_j_kg", "CAPE", "surface", "surface CAPE", "J kg-1"),
    FieldSpec("cin_surface_j_kg", "CIN", "surface", "surface CIN", "J kg-1"),
    FieldSpec(
        "pwat_entire_atmosphere_kg_m2",
        "PWAT",
        "entire atmosphere (considered as a single layer)",
        "precipitable water",
        "kg m-2",
    ),
    FieldSpec("tcdc_boundary_pct", "TCDC", "boundary layer cloud layer", "boundary layer cloud cover", "%"),
    FieldSpec("lcdc_low_pct", "LCDC", "low cloud layer", "low cloud cover", "%"),
    FieldSpec("mcdc_mid_pct", "MCDC", "middle cloud layer", "middle cloud cover", "%"),
    FieldSpec("hcdc_high_pct", "HCDC", "high cloud layer", "high cloud cover", "%"),
    FieldSpec("tcdc_entire_pct", "TCDC", "entire atmosphere", "total cloud cover", "%"),
    FieldSpec("ceiling_m", "HGT", "cloud ceiling", "cloud ceiling height", "m"),
    FieldSpec("cloud_base_m", "HGT", "cloud base", "cloud base height", "m"),
    FieldSpec("cloud_top_m", "HGT", "cloud top", "cloud top height", "m"),
    FieldSpec("cloud_base_pressure_pa", "PRES", "cloud base", "cloud base pressure", "Pa"),
    FieldSpec("cloud_top_pressure_pa", "PRES", "cloud top", "cloud top pressure", "Pa"),
    FieldSpec("dswrf_surface_w_m2", "DSWRF", "surface", "downward shortwave flux", "W m-2"),
    FieldSpec("dlwrf_surface_w_m2", "DLWRF", "surface", "downward longwave flux", "W m-2"),
    FieldSpec("uswrf_surface_w_m2", "USWRF", "surface", "upward shortwave flux", "W m-2"),
    FieldSpec("ulwrf_surface_w_m2", "ULWRF", "surface", "upward longwave flux", "W m-2"),
    FieldSpec("uswrf_toa_w_m2", "USWRF", "top of atmosphere", "upward shortwave flux at TOA", "W m-2"),
    FieldSpec("ulwrf_toa_w_m2", "ULWRF", "top of atmosphere", "upward longwave flux at TOA", "W m-2"),
    FieldSpec("hpbl_m", "HPBL", "surface", "planetary boundary layer height", "m"),
]

SUBHOURLY_FIELD_SPECS = [
    FieldSpec("visibility_m", "VIS", "surface", "surface visibility", "m"),
    FieldSpec("gust_surface_ms", "GUST", "surface", "surface wind gust", "m s-1"),
    FieldSpec("surface_pressure_pa", "PRES", "surface", "surface pressure", "Pa"),
    FieldSpec("tmp_2m_k", "TMP", "2 m above ground", "2 m temperature", "K"),
    FieldSpec("dpt_2m_k", "DPT", "2 m above ground", "2 m dew point", "K"),
    FieldSpec("ugrd_10m_ms", "UGRD", "10 m above ground", "10 m u wind", "m s-1"),
    FieldSpec("vgrd_10m_ms", "VGRD", "10 m above ground", "10 m v wind", "m s-1"),
    FieldSpec("prate_surface_kg_m2_s", "PRATE", "surface", "precipitation rate", "kg m-2 s-1"),
    FieldSpec("apcp_surface_kg_m2", "APCP", "surface", "accumulated precipitation", "kg m-2", mode="accum"),
    FieldSpec("ceiling_m", "HGT", "cloud ceiling", "cloud ceiling height", "m"),
    FieldSpec("cloud_base_m", "HGT", "cloud base", "cloud base height", "m"),
    FieldSpec("cloud_top_m", "HGT", "cloud top", "cloud top height", "m"),
    FieldSpec("dswrf_surface_w_m2", "DSWRF", "surface", "downward shortwave flux", "W m-2"),
    FieldSpec("dlwrf_surface_w_m2", "DLWRF", "surface", "downward longwave flux", "W m-2"),
    FieldSpec("uswrf_surface_w_m2", "USWRF", "surface", "upward shortwave flux", "W m-2"),
    FieldSpec("ulwrf_surface_w_m2", "ULWRF", "surface", "upward longwave flux", "W m-2"),
    FieldSpec("uswrf_toa_w_m2", "USWRF", "top of atmosphere", "upward shortwave flux at TOA", "W m-2"),
    FieldSpec("ulwrf_toa_w_m2", "ULWRF", "top of atmosphere", "upward longwave flux at TOA", "W m-2"),
]

PRODUCT_FIELD_SPECS = {
    "surface": SURFACE_FIELD_SPECS,
    "subhourly": SUBHOURLY_FIELD_SPECS,
}

