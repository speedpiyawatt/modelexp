# Data Source Inventory

This document compiles the data sources currently present or explicitly supported in this repo for the KLGA settlement model. It separates:

- settlement/label data
- forecast feature data
- raw runtime outputs
- canonical training outputs
- operational metadata

It is intentionally repo-specific. If a source is not listed here, it is not currently implemented or committed in this workspace.

## Executive Summary

Current source lanes in the repo:

1. Wunderground KLGA daily history
2. NOAA NBM forecast data
3. NOAA HRRR forecast data
4. NOAA LAMP station guidance
5. Canonical normalized training schema in `tools/weather`

Not currently present as a committed repo data source:

- Polymarket market/orderbook/trade history datasets
- ASOS/METAR historical tables outside the Wunderground history reconstruction
- NWS NDFD grids
- satellite/cloud products
- nearby-station datasets

That means the repo already covers:

- settlement-aligned historical labels from KLGA
- gridded forecast guidance from NBM and HRRR
- station-text guidance from LAMP

But it does not yet contain:

- market-price history
- event-bin snapshots
- execution/liquidity history

## 1. Wunderground Settlement Dataset

Location:

- `wunderground/output/history`

Purpose:

- canonical settlement-aligned historical observation source
- best current source for reconstructing `final_tmax_f`
- best current source for deriving intraday `max_so_far`

Granularity:

- one JSON file per local calendar day
- each file contains a raw observation stream for KLGA
- observation count varies by day; it is not forced to 24 rows

Coverage currently committed:

- `2023-01-01` through `2026-04-11`
- `1197` daily files

Top-level JSON structure:

```json
{
  "metadata": { "...": "..." },
  "observations": [{ "...": "..." }]
}
```

### 1.1 Wunderground metadata fields

Observed metadata keys in the committed dataset:

- `expire_time_gmt`
- `language`
- `location_id`
- `status_code`
- `transaction_id`
- `units`
- `version`

Notes:

- for overnight selection, `processed_timestamp_utc` is the source-availability timestamp used to determine cutoff eligibility
- historical rebuilds should preserve source availability semantics rather than writing wall-clock rebuild time into this field

- `transaction_id` and `expire_time_gmt` are volatile response metadata
- these should not be used as model features
- `location_id` should remain `KLGA:9:US`

### 1.2 Wunderground observation fields

Observed observation-key superset in the committed dataset:

- `blunt_phrase`
- `class`
- `clds`
- `day_ind`
- `dewPt`
- `expire_time_gmt`
- `feels_like`
- `gust`
- `heat_index`
- `icon_extd`
- `key`
- `max_temp`
- `min_temp`
- `obs_id`
- `obs_name`
- `precip_hrly`
- `precip_total`
- `pressure`
- `pressure_desc`
- `pressure_tend`
- `primary_swell_direction`
- `primary_swell_height`
- `primary_swell_period`
- `primary_wave_height`
- `primary_wave_period`
- `qualifier`
- `qualifier_svrty`
- `rh`
- `secondary_swell_direction`
- `secondary_swell_height`
- `secondary_swell_period`
- `snow_hrly`
- `temp`
- `terse_phrase`
- `uv_desc`
- `uv_index`
- `valid_time_gmt`
- `vis`
- `water_temp`
- `wc`
- `wdir`
- `wdir_cardinal`
- `wspd`
- `wx_icon`
- `wx_phrase`

### 1.3 Wunderground fields that matter most for the model

High-priority label and intraday fields:

- `valid_time_gmt`
- `temp`
- `max_temp`
- `min_temp`
- `dewPt`
- `rh`
- `pressure`
- `wdir`
- `wdir_cardinal`
- `wspd`
- `gust`
- `vis`
- `clds`
- `wx_phrase`
- `precip_hrly`
- `precip_total`
- `snow_hrly`

Useful interpretation fields:

- `obs_name`
- `pressure_desc`
- `pressure_tend`
- `day_ind`
- `feels_like`
- `heat_index`
- `wc`
- `uv_index`
- `uv_desc`
- `terse_phrase`
- `blunt_phrase`

Likely noise or non-core for V1:

- wave/swell fields
- `water_temp`
- most icon/qualifier keys

### 1.4 Recommended model artifacts derived from Wunderground

You will likely want to build these tables from the raw JSON:

`labels_daily`

- `date_local`
- `station_id`
- `final_tmax_f`
- `final_tmin_f`
- `obs_count`
- `first_obs_time_local`
- `last_obs_time_local`
- `source_file`

`observations_intraday`

- `date_local`
- `valid_time_utc`
- `valid_time_local`
- `temp_f`
- `dewpoint_f`
- `rh_pct`
- `pressure_in`
- `wind_dir_deg`
- `wind_speed_mph`
- `wind_gust_mph`
- `visibility`
- `cloud_cover_code`
- `wx_phrase`
- `precip_hrly_in`
- `snow_hrly_in`
- `max_so_far_f`
- `warming_rate_1h_f`
- `warming_rate_3h_f`

## 2. NOAA NBM Data

Repo locations:

- `tools/nbm/fetch_nbm.py`
- `tools/nbm/build_grib2_features.py`

Purpose:

- public forecast guidance for KLGA-centered feature generation
- both historical backfill and production-style feature extraction through the GRIB2 pipeline

Operational fetch contract:

- `build_grib2_features.py` downloads the small `.idx` first
- it plans selected GRIB2 record byte ranges from that inventory
- it downloads only those selected ranges into a local selected-record `.grib2`
- ranged GRIB2 responses must return `206 Partial Content` with the exact requested byte count

### 2.1 NBM GRIB2 source

Bucket:

- `noaa-nbm-grib2-pds`

This is the primary raw NBM runtime builder in the repo.

Configured NBM feature families in `build_grib2_features.py`:

- `tmp`
- `dpt`
- `rh`
- `tmax`
- `tmin`
- `wind`
- `wdir`
- `gust`
- `tcdc`
- `dswrf`
- `apcp`
- `vrate`
- `pcpdur`
- `vis`
- `ceil`
- `cape`
- `thunc`
- `ptype`
- `pwther`
- `tstm`

Optional NBM families:

- `tmax`
- `tmin`
- `pcpdur`
- `thunc`
- `ptype`
- `pwther`
- `tstm`

NBM field meanings:

- `tmp`: 2 m temperature
- `dpt`: 2 m dew point
- `rh`: 2 m relative humidity
- `tmax`: max 2 m temperature
- `tmin`: min 2 m temperature
- `wind`: 10 m wind speed
- `wdir`: 10 m wind direction
- `gust`: 10 m gust
- `tcdc`: total cloud cover
- `dswrf`: downward shortwave radiation
- `apcp`: accumulated precipitation
- `vrate`: ventilation rate
- `pcpdur`: precipitation duration
- `vis`: visibility
- `ceil`: ceiling
- `cape`: CAPE
- `thunc`: thunder coverage/code
- `ptype`: precipitation type code
- `pwther`: present weather code
- `tstm`: thunderstorm probability

### 2.2 NBM runtime output structure

The raw NBM GRIB2 builder writes:

- wide parquet
- provenance parquet
- manifest parquet
- optional long parquet

When `--keep-downloads` is enabled, the retained raw artifact is the selected-record local `.grib2`, not the full remote NOAA object.

Expected wide identity columns align to the shared runtime contract:

- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `init_time_utc`
- `valid_time_utc`
- `init_time_local`
- `valid_time_local`
- `init_date_local`
- `valid_date_local`
- `forecast_hour`

Expected raw NBM metadata columns:

- `fallback_used_any`
- `missing_optional_any`
- `missing_optional_fields_count`

Expected raw NBM spatial columns:

- `settlement_lat`
- `settlement_lon`
- `crop_top_lat`
- `crop_bottom_lat`
- `crop_left_lon`
- `crop_right_lon`
- `nearest_grid_lat`
- `nearest_grid_lon`

Likely additional operational columns in raw NBM wide/manifests:

- `task_key`
- `run_date_utc`
- `cycle_hour_utc`
- `lead_hour`
- `requested_lead_hour`
- GRIB URL/path columns
- inventory/debug columns

The actual meteorological columns expand by suffix:

- bare field: nearest value, for example `tmp`
- `_crop_mean`
- `_crop_min`
- `_crop_max`
- `_crop_std`
- `_nb3_mean`
- `_nb3_min`
- `_nb3_max`
- `_nb3_std`
- `_nb3_gradient_west_east`
- `_nb3_gradient_south_north`
- `_nb7_mean`
- `_nb7_min`
- `_nb7_max`
- `_nb7_std`
- `_nb7_gradient_west_east`
- `_nb7_gradient_south_north`

So `tmp` becomes:

- `tmp`
- `tmp_crop_mean`
- `tmp_crop_min`
- `tmp_crop_max`
- `tmp_crop_std`
- `tmp_nb3_mean`
- `tmp_nb3_min`
- `tmp_nb3_max`
- `tmp_nb3_std`
- `tmp_nb3_gradient_west_east`
- `tmp_nb3_gradient_south_north`
- `tmp_nb7_mean`
- `tmp_nb7_min`
- `tmp_nb7_max`
- `tmp_nb7_std`
- `tmp_nb7_gradient_west_east`
- `tmp_nb7_gradient_south_north`

Nearest-only code/probability fields are expected for:

- `ptype`
- `pwther`
- `thunc`
- `tstm`

## 3. NOAA HRRR Data

Repo locations:

- `tools/hrrr/build_hrrr_klga_feature_shards.py`
- `tools/hrrr/hrrr_fields.py`
- `tools/hrrr/data/fixtures/contract_smoke`

Purpose:

- higher-detail forecast guidance
- rich gridded feature source for same-day updates and forecast residual framing

Persistent runtime artifacts:

- monthly wide parquet
- monthly provenance parquet
- monthly manifest parquet
- monthly JSON manifest

### 3.1 HRRR core field families

Near-surface fields:

- `tmp_2m_k`
- `dpt_2m_k`
- `rh_2m_pct`

Wind, pressure, visibility:

- `ugrd_10m_ms`
- `vgrd_10m_ms`
- `gust_surface_ms`
- `surface_pressure_pa`
- `mslma_pa`
- `visibility_m`

Cloud, radiation, precipitation:

- `lcdc_low_pct`
- `mcdc_mid_pct`
- `hcdc_high_pct`
- `tcdc_entire_pct`
- `dswrf_surface_w_m2`
- `dlwrf_surface_w_m2`
- `apcp_surface_kg_m2`
- `prate_surface_kg_m2_s`

Boundary layer and column:

- `hpbl_m`
- `pwat_entire_atmosphere_kg_m2`
- `cape_surface_j_kg`
- `cin_surface_j_kg`
- `refc_entire_atmosphere`
- `ltng_entire_atmosphere`

Additional surface/subhourly fields defined in `hrrr_fields.py`:

- `tcdc_boundary_pct`
- `ceiling_m`
- `cloud_base_m`
- `cloud_top_m`
- `cloud_base_pressure_pa`
- `cloud_top_pressure_pa`
- `uswrf_surface_w_m2`
- `ulwrf_surface_w_m2`
- `uswrf_toa_w_m2`
- `ulwrf_toa_w_m2`

Upper-air fields at `1000`, `925`, `850`, and `700 mb`:

- `tmp_*mb_k`
- `ugrd_*mb_ms`
- `vgrd_*mb_ms`
- `rh_*mb_pct`
- `spfh_*mb_kg_kg`
- `hgt_*mb_gpm`

### 3.2 HRRR raw wide table shape

The committed fixture `tools/hrrr/data/fixtures/contract_smoke/2025-04.parquet` has:

- `1` row
- `905` columns

Raw wide identity and metadata columns observed in the fixture:

- `task_key`
- `source_model`
- `source_product`
- `source_version`
- `fallback_used_any`
- `station_id`
- `run_date_utc`
- `cycle_hour_utc`
- `forecast_hour`
- `grib_url`
- `settlement_station_id`
- `settlement_lat`
- `settlement_lon`
- `settlement_lon_360`
- `crop_top_lat`
- `crop_bottom_lat`
- `crop_left_lon`
- `crop_right_lon`
- `crop_top`
- `crop_bottom`
- `crop_left`
- `crop_right`
- `target_lat`
- `target_lon`
- `grid_row`
- `grid_col`
- `nearest_grid_lat`
- `nearest_grid_lon`
- `grid_lat`
- `grid_lon`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`

Every spatial HRRR base field expands by the same suffix family:

- bare nearest value
- `_crop_mean`
- `_crop_min`
- `_crop_max`
- `_crop_std`
- `_nb3_mean`
- `_nb3_min`
- `_nb3_max`
- `_nb3_std`
- `_nb3_gradient_west_east`
- `_nb3_gradient_south_north`
- `_nb7_mean`
- `_nb7_min`
- `_nb7_max`
- `_nb7_std`
- `_nb7_gradient_west_east`
- `_nb7_gradient_south_north`

Example HRRR column families:

- `tmp_2m_k`
- `tmp_2m_k_crop_mean`
- `tmp_2m_k_nb3_mean`
- `tmp_2m_k_nb7_gradient_west_east`
- `ugrd_10m_ms`
- `ugrd_10m_ms_nb3_std`
- `tcdc_entire_pct_crop_max`
- `pwat_entire_atmosphere_kg_m2_nb7_mean`

### 3.3 HRRR provenance table

Observed columns in the committed fixture provenance parquet:

- `task_key`
- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`
- `forecast_hour`
- `nearest_grid_lat`
- `nearest_grid_lon`
- `feature_name`
- `output_column_base`
- `grib_short_name`
- `grib_level_text`
- `grib_type_of_level`
- `grib_step_type`
- `grib_step_text`
- `source_inventory_line`
- `units`
- `present_directly`
- `derived`
- `derivation_method`
- `source_feature_names`
- `missing_optional`
- `fallback_used`
- `fallback_source_description`
- `notes`

This is useful for:

- feature lineage
- debugging missing fields
- understanding derivations and fallbacks

### 3.4 HRRR manifest table

Observed columns in the committed fixture manifest parquet:

- `month`
- `task_key`
- `status`
- `failure_reason`
- `missing_fields`
- `source_model`
- `source_product`
- `source_version`
- `wide_parquet_path`
- `provenance_path`
- `manifest_json_path`
- `keep_downloads`
- `keep_reduced`
- `complete`

## 4. NOAA LAMP Data

Repo locations:

- `tools/lamp/fetch_lamp.py`
- `tools/lamp/parse_lamp_ascii.py`
- `tools/lamp/build_lamp_klga_features.py`
- `tools/lamp/build_lamp_overnight_features.py`
- `tools/lamp/data/fixtures/parser_samples`

Purpose:

- station-based textual guidance tied directly to KLGA
- useful for short-range intraday updating
- lower-dimensional than NBM/HRRR but very operationally relevant
- now also includes a model-facing overnight summarizer built from raw issue-level LAMP feature parquet

### 4.1 LAMP raw parsed long table

Raw parsed long columns:

- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `bulletin_type`
- `bulletin_version`
- `bulletin_source_path`
- `bulletin_source_url`
- `archive_member`
- `raw_label`
- `raw_value`
- `value`
- `units`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`
- `forecast_hour`
- `utc_hour_token`
- `station_lat`
- `station_lon`
- `settlement_station_id`
- `settlement_lat`
- `settlement_lon`

This table is one row per:

- issue time
- valid hour
- raw LAMP label

### 4.2 LAMP raw labels currently curated into wide features

Curated labels:

- `TMP`
- `DPT`
- `WDR`
- `WSP`
- `WGS`
- `PPO`
- `P01`
- `P06`
- `P12`
- `PCO`
- `PC1`
- `LP1`
- `LC1`
- `CP1`
- `CC1`
- `POZ`
- `POS`
- `TYP`
- `CLD`
- `CIG`
- `CCG`
- `VIS`
- `CVS`
- `OBV`

Label meanings implied by parser/build code:

- `TMP`: temperature in F
- `DPT`: dew point in F
- `WDR`: wind direction in tens of degrees
- `WSP`: wind speed in kt
- `WGS`: wind gust in kt or code
- `PPO`: probability of precipitation occurrence
- `P01`: 1-hour precipitation probability
- `P06`: 6-hour precipitation probability
- `P12`: 12-hour precipitation probability
- `PCO`: conditional precip occurrence yes/no
- `PC1`: conditional 1-hour precip yes/no
- `LP1`: lightning probability
- `LC1`: lightning conditional yes/no
- `CP1`: conditional precip probability
- `CC1`: conditional cloud/ceiling style yes/no flag
- `POZ`: freezing precipitation style probability/code family
- `POS`: snow/sleet style probability/code family
- `TYP`: precipitation type code
- `CLD`: cloud cover code
- `CIG`: ceiling code
- `CCG`: categorical ceiling group/code
- `VIS`: visibility code
- `CVS`: categorical visibility group/code
- `OBV`: obstruction-to-vision code

Optional labels:

- `P06`
- `P12`
- `LP1`
- `LC1`
- `CP1`
- `CC1`
- `POZ`
- `POS`
- `TYP`

### 4.3 LAMP raw wide table

Observed columns from a built KLGA fixture path:

- `source_model`
- `source_product`
- `source_version`
- `fallback_used_any`
- `station_id`
- `station_lat`
- `station_lon`
- `settlement_station_id`
- `settlement_lat`
- `settlement_lon`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`
- `forecast_hour`
- `cc1`
- `ccg`
- `cig`
- `cld`
- `cp1`
- `cvs`
- `dpt`
- `lc1`
- `lp1`
- `obv`
- `p01`
- `pc1`
- `pco`
- `pos`
- `poz`
- `ppo`
- `tmp`
- `typ`
- `vis`
- `wdr`
- `wgs`
- `wsp`
- `missing_optional_any`
- `missing_optional_fields_count`
- `p06`

Note:

- `p12` is curated and can appear, but was not present in the sampled output shown above
- LAMP wide data is much narrower than NBM/HRRR and has no neighborhood/crop expansions
- raw LAMP wide rows now carry `missing_optional_any` and `missing_optional_fields_count`, and the raw manifest warns when optional labels are missing across the built issue rows

### 4.4 LAMP provenance table

Observed columns:

- `source_model`
- `source_product`
- `source_version`
- `fallback_used_any`
- `station_id`
- `station_lat`
- `station_lon`
- `settlement_station_id`
- `settlement_lat`
- `settlement_lon`
- `init_time_utc`
- `init_time_local`
- `init_date_local`
- `valid_time_utc`
- `valid_time_local`
- `valid_date_local`
- `forecast_hour`
- `feature_name`
- `raw_feature_name`
- `present_directly`
- `derived`
- `missing_optional`
- `derivation_method`
- `source_feature_names`
- `fallback_used`
- `fallback_source_description`
- `grib_short_name`
- `grib_level_text`
- `grib_type_of_level`
- `grib_step_type`
- `grib_step_text`
- `inventory_line`
- `units`
- `notes`
- `bulletin_type`
- `bulletin_version`
- `bulletin_source_path`
- `archive_member`

Notes:

- present curated labels emit direct provenance rows
- absent optional labels emit `missing_optional=True` provenance rows keyed to the same init/valid timestamps
- LAMP provenance is station-text lineage, not GRIB lineage, so the GRIB metadata fields are present but normally null

### 4.5 LAMP overnight summary table

Repo location:

- `tools/lamp/build_lamp_overnight_features.py`

Runtime output location:

- `tools/lamp/data/runtime/overnight/target_date_local=YYYY-MM-DD/lamp.overnight.parquet`

Purpose:

- one model-facing row per target local day for the overnight KLGA model
- built from the latest raw LAMP issue available by `00:05` local, plus the immediately prior issue for revision features when available

Observed columns from the current smoke output:

- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `target_date_local`
- `selected_init_time_utc`
- `selected_init_time_local`
- `previous_init_time_utc`
- `previous_init_time_local`
- `selection_cutoff_local`
- `coverage_complete`
- `missing_checkpoint_any`
- `missing_checkpoint_count`
- `revision_available`
- `forecast_hour_min`
- `forecast_hour_max`
- `target_day_row_count`
- `day_tmp_max_f_forecast`
- `day_tmp_min_f_forecast`
- `day_tmp_range_f_forecast`
- `day_tmp_argmax_local_hour`
- `morning_cld_mode`
- `morning_cig_min_hundreds_ft`
- `morning_vis_min_miles`
- `morning_obv_any`
- `morning_ifr_like_any`
- `day_p01_max_pct`
- `day_p06_max_pct`
- `day_p12_max_pct`
- `day_pos_max_pct`
- `day_poz_max_pct`
- `day_precip_type_any`
- `day_precip_type_mode`
- `rev_day_tmp_max_f`
- `rev_day_p01_max_pct`
- `rev_day_pos_max_pct`
- `rev_morning_cig_min_hundreds_ft`
- `rev_morning_vis_min_miles`
- checkpoint families at `06/09/12/15/18/21` local:
  - `tmp_f_at_HH`
  - `dpt_f_at_HH`
  - `wsp_kt_at_HH`
  - `wdr_deg_at_HH`
  - `wgs_code_at_HH`
  - `cld_code_at_HH`
  - `cig_hundreds_ft_at_HH`
  - `vis_miles_at_HH`
  - `obv_code_at_HH`
  - `typ_code_at_HH`
- revision checkpoint deltas:
  - `rev_tmp_f_at_06`
  - `rev_tmp_f_at_09`
  - `rev_tmp_f_at_12`
  - `rev_tmp_f_at_15`
  - `rev_tmp_f_at_18`
  - `rev_tmp_f_at_21`

Interpretation:

- this table is intentionally LAMP-specific and model-facing
- it preserves native LAMP units and codes instead of forcing canonical cross-source conversion
- it is not a replacement for the raw source-aware LAMP issue-level outputs

### 4.6 LAMP overnight manifest table

Runtime output location:

- `tools/lamp/data/runtime/overnight/target_date_local=YYYY-MM-DD/lamp.overnight.manifest.parquet`

Observed columns:

- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `target_date_local`
- `status`
- `extraction_status`
- `processed_timestamp_utc`
- `selection_cutoff_local`
- `selected_init_time_utc`
- `selected_init_time_local`
- `previous_init_time_utc`
- `previous_init_time_local`
- `revision_available`
- `coverage_complete`
- `missing_checkpoint_any`
- `missing_checkpoint_count`
- `warning`
- `failure_reason`
- `overnight_output_path`
- `manifest_parquet_path`
- `row_count`

Notes:

- success rows use `status=ok` / `extraction_status=ok`
- dates with no qualifying issue use `status=incomplete` / `extraction_status=no_qualifying_issue`
- successful manifests also warn when target-day checkpoints are sparse or no prior issue is available for revision deltas

## 5. Canonical Training Schema

Repo location:

- `tools/weather/canonical_feature_schema.py`
- `tools/weather/normalize_training_features.py`

Purpose:

- unify raw NBM, HRRR, and LAMP outputs into one model-facing schema

This is not a new datasource. It is the downstream standardized training layer.

### 5.1 Canonical identity columns

- `source_model`
- `source_product`
- `source_version`
- `station_id`
- `init_time_utc`
- `valid_time_utc`
- `init_time_local`
- `valid_time_local`
- `init_date_local`
- `valid_date_local`
- `forecast_hour`

### 5.2 Canonical spatial columns

- `settlement_lat`
- `settlement_lon`
- `crop_top_lat`
- `crop_bottom_lat`
- `crop_left_lon`
- `crop_right_lon`
- `nearest_grid_lat`
- `nearest_grid_lon`

### 5.3 Canonical metadata columns

- `fallback_used_any`
- `missing_optional_any`
- `missing_optional_fields_count`

Optional passthrough columns:

- `mode`
- `task_key`
- `run_date_utc`
- `cycle_hour_utc`
- `lead_hour`
- `requested_lead_hour`

### 5.4 Canonical meteorological feature bases

Spatially-expanded bases:

- `temp_2m_k`
- `dewpoint_2m_k`
- `rh_2m_pct`
- `temp_2m_f`
- `dewpoint_2m_f`
- `tmax_2m_k`
- `tmin_2m_k`
- `wind_10m_u_ms`
- `wind_10m_v_ms`
- `wind_10m_speed_ms`
- `wind_10m_direction_deg`
- `gust_10m_ms`
- `pressure_surface_pa`
- `mslp_pa`
- `visibility_m`
- `ceiling_m`
- `tcdc_pct`
- `lcdc_pct`
- `mcdc_pct`
- `hcdc_pct`
- `dswrf_surface_w_m2`
- `dlwrf_surface_w_m2`
- `apcp_surface_kg_m2`
- `prate_surface_kg_m2_s`
- `pcpdur_surface_h`
- `cape_surface_j_kg`
- `cin_surface_j_kg`
- `hpbl_m`
- `pwat_kg_m2`
- `vrate_atmosphere_m2_s`
- `temp_1000mb_k`
- `temp_925mb_k`
- `temp_850mb_k`
- `temp_700mb_k`
- `rh_1000mb_pct`
- `rh_925mb_pct`
- `rh_850mb_pct`
- `rh_700mb_pct`
- `wind_1000mb_u_ms`
- `wind_925mb_u_ms`
- `wind_850mb_u_ms`
- `wind_700mb_u_ms`
- `wind_1000mb_v_ms`
- `wind_925mb_v_ms`
- `wind_850mb_v_ms`
- `wind_700mb_v_ms`
- `spfh_1000mb_kg_kg`
- `spfh_925mb_kg_kg`
- `spfh_850mb_kg_kg`
- `spfh_700mb_kg_kg`
- `hgt_1000mb_gpm`
- `hgt_925mb_gpm`
- `hgt_850mb_gpm`
- `hgt_700mb_gpm`

Nearest-only bases:

- `ptype_code`
- `pwther_code`
- `thunc_code`
- `tstm_prob_pct`

Each spatial base expands with the same suffix family:

- bare nearest value
- `_crop_mean`
- `_crop_min`
- `_crop_max`
- `_crop_std`
- `_nb3_mean`
- `_nb3_min`
- `_nb3_max`
- `_nb3_std`
- `_nb3_gradient_west_east`
- `_nb3_gradient_south_north`
- `_nb7_mean`
- `_nb7_min`
- `_nb7_max`
- `_nb7_std`
- `_nb7_gradient_west_east`
- `_nb7_gradient_south_north`

### 5.5 Canonical source mappings

NBM to canonical:

- `tmp -> temp_2m_k`
- `dpt -> dewpoint_2m_k`
- `rh -> rh_2m_pct`
- `tmax -> tmax_2m_k`
- `tmin -> tmin_2m_k`
- `u10/wind family -> wind_*`
- `gust -> gust_10m_ms`
- `tcdc -> tcdc_pct`
- `dswrf -> dswrf_surface_w_m2`
- `apcp -> apcp_surface_kg_m2`
- `vrate -> vrate_atmosphere_m2_s`
- `pcpdur -> pcpdur_surface_h`
- `vis -> visibility_m`
- `ceil -> ceiling_m`
- `cape -> cape_surface_j_kg`
- `thunc -> thunc_code`
- `ptype -> ptype_code`
- `pwther -> pwther_code`
- `tstm -> tstm_prob_pct`

HRRR to canonical:

- `tmp_2m_k -> temp_2m_k`
- `dpt_2m_k -> dewpoint_2m_k`
- `rh_2m_pct -> rh_2m_pct`
- `tmp_2m_f -> temp_2m_f`
- `dpt_2m_f -> dewpoint_2m_f`
- `ugrd_10m_ms -> wind_10m_u_ms`
- `vgrd_10m_ms -> wind_10m_v_ms`
- `wind_10m_speed_ms -> wind_10m_speed_ms`
- `wind_10m_direction_deg -> wind_10m_direction_deg`
- `gust_surface_ms -> gust_10m_ms`
- `surface_pressure_pa -> pressure_surface_pa`
- `mslma_pa -> mslp_pa`
- `visibility_m -> visibility_m`
- `lcdc_low_pct -> lcdc_pct`
- `mcdc_mid_pct -> mcdc_pct`
- `hcdc_high_pct -> hcdc_pct`
- `tcdc_entire_pct -> tcdc_pct`
- `dswrf_surface_w_m2 -> dswrf_surface_w_m2`
- `dlwrf_surface_w_m2 -> dlwrf_surface_w_m2`
- `apcp_surface_kg_m2 -> apcp_surface_kg_m2`
- `prate_surface_kg_m2_s -> prate_surface_kg_m2_s`
- `hpbl_m -> hpbl_m`
- `pwat_entire_atmosphere_kg_m2 -> pwat_kg_m2`
- `cape_surface_j_kg -> cape_surface_j_kg`
- `cin_surface_j_kg -> cin_surface_j_kg`
- pressure-level fields map level-for-level

LAMP to canonical:

- `tmp -> temp_2m_k` after F-to-K conversion
- `dpt -> dewpoint_2m_k` after F-to-K conversion
- `wdr -> wind_10m_direction_deg`
- `wsp -> wind_10m_speed_ms` after kt-to-m/s conversion
- `wgs -> gust_10m_ms` after kt-to-m/s conversion
- `cig -> ceiling_m` after hundreds-of-feet to meters conversion
- `vis -> visibility_m` after miles-to-meters conversion
- `typ -> ptype_code`

## 6. Model-Planning Takeaways

For model design, the practical stack is:

Labels:

- Wunderground daily KLGA history

Intraday observed state:

- Wunderground observation stream

Forecast backbones:

- NBM
- HRRR
- LAMP

Canonical model input layer:

- normalized `tools/weather` schema

Overnight LAMP model adapter:

- `tools/lamp/build_lamp_overnight_features.py`
- produces one daily summary row from raw issue-level LAMP features for the overnight model

Still missing for full trading workflow:

- Polymarket event-bin parser data
- historical market-price snapshots
- liquidity/spread history
- fee/slippage execution dataset

## 7. Recommended V1 Training Tables

If you want a clean modeling plan, build these four tables first:

`labels_daily`

- one row per KLGA local day
- target: `final_tmax_f`

`obs_intraday_hourly`

- one row per observation timestamp
- derived from Wunderground
- includes `max_so_far` and warming-rate features

`forecast_features_runtime`

- raw NBM, HRRR, and LAMP rows keyed by:
  - `source_model`
  - `init_time_utc`
  - `valid_time_utc`
  - `forecast_hour`

`lamp_overnight_daily`

- one row per `target_date_local`
- built from the latest LAMP issue available by the overnight local cutoff
- preserves native LAMP units/codes and adds revision deltas versus the prior issue when available

`training_features_canonical`

- output of `tools/weather/normalize_training_features.py`
- one canonical schema regardless of source

Once those exist, the missing trading layer is separate:

- event outcome bins
- market prices
- trade simulator/backtest ledger
