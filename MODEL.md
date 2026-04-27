# MODEL.md

## Purpose

`MODEL.md` is the canonical detailed design spec for the overnight KLGA fair-value model.

The model's job is:

**At 12:05 a.m. America/New_York, produce calibrated probabilities for the final KLGA daily max temperature bin for that local day.**

This is a market-open fair-value model. It is not the intraday updater. The intraday updater is a later layer that should consume the same settlement-aligned target and build on top of the overnight baseline.

## Timing And As-Of Rule

One hard rule governs both training and inference:

**All overnight features must be available by `target_date_local 00:05 America/New_York`.**

Use that cutoff consistently.

- Wunderground: keep only observations at or before the cutoff
- LAMP: select the latest issue with `init_time_local <= 00:05`
- HRRR: restrict to cycles actually available by `00:05`, then retain the anchor cycle plus the configured revision cycles
- NBM: use the latest available issue by the same cutoff

Do not mix "best available later that night" data into rows for a `12:05 a.m. America/New_York` model.

NBM and HRRR are intentionally not symmetric here: NBM contributes one selected cutoff-eligible issue per `target_date_local`, while HRRR contributes a retained set of cutoff-eligible cycles because revision features are part of the overnight contract.

## Modeling Unit

Use **one row per `target_date_local`**.

That row is the overnight snapshot for the local market day.

Required identity fields:

- `target_date_local`
- `station_id`
- `selection_cutoff_local`

Required training labels:

- `label_final_tmax_f`
- `label_market_bin`

All source blocks collapse into this one row.

## Persisted Training Tables

Freeze two separate overnight artifacts:

- `training_features_overnight`: source-aware merged contract table
- `training_features_overnight_normalized`: model-ready normalized table built only from the merged contract table

Do not collapse these into one artifact.

`training_features_overnight` is the provenance-preserving join point. It keeps source-specific prefixes, source-native categorical strings, mixed units where needed, and explicit availability/coverage metadata.

`training_features_overnight_normalized` is the model-facing layer. It keeps the same row grain and labels, but applies frozen normalization policy:

- unified model-facing units for temperatures, winds, visibility, ceiling, precipitation depth, and pressure
- stable integer vocabularies for cloud cover, weather-family, and precipitation-type categories
- no raw freeform categorical strings in default model inputs
- no persisted numeric imputation in v1
- metadata retained for auditability, but raw timestamps are not default model inputs

Use these names consistently throughout the repo:

- `training_features_overnight` = merged contract table
- `training_features_overnight_normalized` = canonical model-facing table

## Canonical Labels

Use Wunderground-derived daily labels as the canonical target source.

Build and retain these fields:

- `final_tmax_f`
- `final_tmin_f`
- `market_bin`
- `obs_count`
- `first_obs_time_local`
- `last_obs_time_local`

Wunderground is the canonical settlement-aligned source in this repo for reconstructing `final_tmax_f` and the intraday observation state needed to derive settlement-matched labels.

## Source-Aware Feature Blocks

The overnight model should be a **stacked ensemble**, not one giant flat table shoved into one learner.

Use four source-aware feature blocks:

- Wunderground late-evening state
- NBM overnight prior
- LAMP overnight station summary
- HRRR overnight summary plus revisions

### 1. Wunderground Late-Evening State Block

This block answers: what is KLGA actually doing right now as the new local day begins?

Use only the latest available observations before `00:05 America/New_York`.

Raw features:

- last observed `temp_f`
- last observed `dewpoint_f`
- last observed `rh_pct`
- last observed `pressure_in`
- last observed `wind_speed_mph`
- last observed `wind_dir_deg`
- last observed `wind_gust_mph`
- last observed `visibility`
- last observed `cloud_cover_code`
- last observed `wx_phrase`
- last observed `precip_hrly_in`
- previous local-day `final_tmax_f`
- previous local-day `final_tmin_f`
- previous local-day total precip

Trend features over the last `1/3/6` hours before cutoff:

- `temp_change_1h_f`
- `temp_change_3h_f`
- `temp_change_6h_f`
- `dewpoint_change_3h_f`
- `pressure_change_3h`
- `wind_speed_mean_3h`
- `wind_gust_max_6h`
- `visibility_min_6h`
- `precip_total_6h`

### 2. NBM Overnight Prior Block

NBM is the **primary overnight forecast source** and the best blended overnight prior in the currently implemented source set.

Do not train on every raw valid hour directly. Summarize NBM into the target local day from the single latest NBM issue available by `00:05 America/New_York`. Do not blend checkpoints or day summaries across multiple NBM cycles inside one overnight row.

Must-use NBM features:

- `tmax_2m_k`
- `temp_2m_k` checkpoints at local `06/09/12/15/18/21`
- `dewpoint_2m_k` checkpoints at `09/15`
- `rh_2m_pct` checkpoints at `09/15`
- `wind_10m_speed_ms` checkpoints at `09/15`
- `wind_10m_direction_deg` checkpoints at `09/15`
- `gust_10m_ms` day max
- `tcdc_pct` morning mean and day mean
- `dswrf_surface_w_m2` day max
- `apcp_surface_kg_m2` day total
- `visibility_m` day min
- `ceiling_m` morning min
- `cape_surface_j_kg` day max when present

Native NBM `tmin` is useful when present, but it is not a blocker for the overnight daily-high model. In observed 04Z `overnight_0005` core backfill inventory, native `TMAX` and hourly `TMP` are present while native `TMIN` can be absent before crop/extract. Treat null `nbm_native_tmin_*` fields and `missing_required_feature_count=1` from missing native `tmin` as source/selection availability, not evidence of batch-crop corruption. Do not reject otherwise valid daily-high rows solely because native `tmin` is missing.

Spatial context from supported NBM expansions:

- nearest
- `nb3`
- `nb7`
- crop

High-value spatial features:

- `temp_2m_k_nb3_mean`
- `temp_2m_k_nb7_mean`
- `temp_2m_k_crop_mean`
- `tcdc_pct_crop_mean`
- `dswrf_surface_w_m2_crop_max`
- `apcp_surface_kg_m2_crop_sum` when derived
- `wind_10m_speed_ms_nb7_mean`

### 3. LAMP Overnight Station-Summary Block

LAMP should remain source-aware and station-specific.

Current repo contract:

- latest issue at or before `00:05` local
- prior issue only for revision features
- one row per `target_date_local`
- fixed checkpoints at `06/09/12/15/18/21`

Operational cycle selection:

- use full LAMP `lavtxt` guidance cycles on `HH30`
- for the `00:05 America/New_York` cutoff, backfill candidate UTC cycles `0230`, `0330`, and `0430`
- let the overnight builder select the latest issue with `init_time_local <= 00:05`; this selects `0430Z` during standard time and `0330Z` during daylight time
- do not use archive cycles such as `0345`, `0400`, `0445`, or `0500` as temperature guidance; observed 2023-2025 archive files for those cycles contain only 3-hour `CIG/VIS/OBV` aviation updates and no `TMP`
- known upstream missing full-cycle archive dates are `2023-04-09`, `2023-04-12`, `2023-06-13`, `2024-11-04`, `2024-11-05`, and `2024-11-06`; mark LAMP unavailable for those dates and do not treat them as local parse or extraction failures

Must-use LAMP checkpoint features:

- `tmp_f_at_06/09/12/15/18/21`
- `dpt_f_at_06/09/12/15/18/21`
- `wsp_kt_at_06/09/12/15/18/21`
- `wdr_deg_at_06/09/12/15/18/21`
- `cld_code_at_06/09/12/15/18/21`
- `cig_hundreds_ft_at_06/09/12/15/18/21`
- `vis_miles_at_06/09/12/15/18/21`
- `obv_code_at_06/09/12/15/18/21`
- `typ_code_at_06/09/12/15/18/21`

Must-use LAMP summary features:

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

Must-use LAMP revision features:

- `rev_day_tmp_max_f`
- `rev_tmp_f_at_06/09/12/15/18/21`
- `rev_day_p01_max_pct`
- `rev_day_pos_max_pct`
- `rev_morning_cig_min_hundreds_ft`
- `rev_morning_vis_min_miles`

Operational and missingness fields:

- `selected_issue_age_minutes`
- `previous_issue_age_minutes`
- `missing_optional_any`
- `missing_optional_fields_count`

### 4. HRRR Overnight Summary And Revision Block

HRRR is the most important redesign relative to the earlier model framing.

Current repo contract:

- planning by `target_date_local`
- overnight init window from previous local day `18:00` to target day `00:00`
- availability cutoff at `00:05` local
- retain the latest `4` reliable cycles
- keep valid hours on the target day from `00:00` to `21:00` local
- use the latest fully covering cycle as the anchor
- use older cycles only for revisions

Must-use HRRR anchor checkpoint features:

- `hrrr_temp_2m_06_local_k`
- `hrrr_temp_2m_09_local_k`
- `hrrr_temp_2m_12_local_k`
- `hrrr_temp_2m_15_local_k`
- `hrrr_temp_2m_18_local_k`
- `hrrr_dewpoint_2m_06_local_k`
- `hrrr_dewpoint_2m_09_local_k`
- `hrrr_dewpoint_2m_12_local_k`
- `hrrr_dewpoint_2m_15_local_k`
- `hrrr_dewpoint_2m_18_local_k`
- `hrrr_rh_2m_06_local_pct`
- `hrrr_rh_2m_09_local_pct`
- `hrrr_rh_2m_12_local_pct`
- `hrrr_rh_2m_15_local_pct`
- `hrrr_rh_2m_18_local_pct`
- `hrrr_u10m_09_local_ms`
- `hrrr_u10m_12_local_ms`
- `hrrr_u10m_15_local_ms`
- `hrrr_u10m_18_local_ms`
- `hrrr_v10m_09_local_ms`
- `hrrr_v10m_12_local_ms`
- `hrrr_v10m_15_local_ms`
- `hrrr_v10m_18_local_ms`
- `hrrr_wind_10m_09_local_speed_ms`
- `hrrr_wind_10m_12_local_speed_ms`
- `hrrr_wind_10m_15_local_speed_ms`
- `hrrr_wind_10m_18_local_speed_ms`
- `hrrr_wind_10m_09_local_direction_deg`
- `hrrr_wind_10m_12_local_direction_deg`
- `hrrr_wind_10m_15_local_direction_deg`
- `hrrr_wind_10m_18_local_direction_deg`
- `hrrr_mslp_09_local_pa`
- `hrrr_mslp_12_local_pa`
- `hrrr_mslp_15_local_pa`
- `hrrr_surface_pressure_09_local_pa`

Must-use HRRR day-window summaries:

- `hrrr_temp_2m_day_max_k`
- `hrrr_temp_2m_day_mean_k`
- `hrrr_rh_2m_day_min_pct`
- `hrrr_wind_10m_day_max_ms`
- `hrrr_gust_day_max_ms`
- `hrrr_tcdc_day_mean_pct`
- `hrrr_tcdc_morning_mean_pct`
- `hrrr_tcdc_afternoon_mean_pct`
- `hrrr_tcdc_day_max_pct`
- `hrrr_lcdc_morning_mean_pct`
- `hrrr_mcdc_day_mean_pct`
- `hrrr_hcdc_day_mean_pct`
- `hrrr_mcdc_afternoon_mean_pct`
- `hrrr_hcdc_afternoon_mean_pct`
- `hrrr_dswrf_day_max_w_m2`
- `hrrr_dlwrf_night_mean_w_m2`
- `hrrr_apcp_day_total_kg_m2`
- `hrrr_cape_day_max_j_kg`
- `hrrr_cape_afternoon_max_j_kg`
- `hrrr_cin_day_min_j_kg`
- `hrrr_cin_afternoon_min_j_kg`
- `hrrr_refc_day_max`
- `hrrr_ltng_day_max`
- `hrrr_ltng_day_any`
- `hrrr_mslp_day_mean_pa`
- `hrrr_pwat_day_mean_kg_m2`
- `hrrr_hpbl_day_max_m`

Must-use upper-air summaries:

- `hrrr_temp_1000mb_day_mean_k`
- `hrrr_temp_925mb_day_mean_k`
- `hrrr_temp_850mb_day_mean_k`
- `hrrr_rh_925mb_day_mean_pct`
- `hrrr_u925_day_mean_ms`
- `hrrr_v925_day_mean_ms`
- `hrrr_u850_day_mean_ms`
- `hrrr_v850_day_mean_ms`
- `hrrr_wind_850mb_speed_day_mean_ms`
- `hrrr_hgt_925_day_mean_gpm`
- `hrrr_hgt_700_day_mean_gpm`

Regional-context summaries should use selected nearest, `nb3`, `nb7`, and crop summaries for:

- temperature
- total cloud
- DSWRF
- PWAT

Revision features should compare the anchor versus previous `1/2/3` cycles for:

- day max `2m` temperature
- `09/12/15` local checkpoints
- day mean cloud
- day max DSWRF
- day mean PWAT
- day max HPBL
- day mean MSLP

Guardrails:

- morning and afternoon windows are half-open local-hour windows
  morning: `06 <= hour < 12`
  afternoon: `12 <= hour < 18`
- use canonical field names with explicit unit suffixes in the HRRR overnight summary output
- `hrrr_ltng_day_any` is `None` when all retained-cycle lightning inputs are missing
- do not add revision deltas for new humidity checkpoints, wind checkpoints, cloud-family summaries, convective summaries, upper-air vector means, or surface-pressure checkpoints in v1; the only new revision family added here is `hrrr_mslp_day_mean_pa_rev_{1,2,3}cycle`

Operational coverage fields:

- `retained_cycle_count`
- `coverage_end_hour_local`
- `has_full_day_21_local_coverage`
- `missing_checkpoint_count`
- `first_valid_hour_local`
- `last_valid_hour_local`
- `covered_hour_count`
- `covered_checkpoint_count`

## Merged Contract Table

Do **not** train directly on raw source-specific tables.

Create one merged contract table:

### `training_features_overnight`

Identity fields:

- `target_date_local`
- `station_id`
- `selection_cutoff_local`

Metadata:

- source versions
- fallback flags
- missing-optional flags
- coverage flags from LAMP and HRRR

Feature blocks:

- Wunderground late-evening state
- NBM overnight prior
- LAMP overnight summary
- HRRR overnight summary

Labels:

- `final_tmax_f`
- `market_bin`

`training_features_overnight` is the merged contract table used as the provenance-preserving join point for overnight modeling. It is merged, but it is not the final model-facing table. Source-aware raw tables stay upstream. The model should not train directly on raw source outputs.

Create one separate canonical model-facing table:

### `training_features_overnight_normalized`

`training_features_overnight_normalized` is built only from `training_features_overnight` and is the canonical model-facing training table for the stacked overnight model. It preserves the same row grain and labels while applying the frozen normalization and categorical encoding policy described above.

### Implemented Contract-Freeze Entrypoints

Use these scripts to build and audit the frozen overnight tables before any server backfill:

- Wunderground labels and intraday observations:
  - `python3 wunderground/build_training_tables.py`
- NBM overnight daily summaries:
  - `python3 tools/nbm/build_nbm_overnight_features.py --start-local-date YYYY-MM-DD --end-local-date YYYY-MM-DD`
- merged overnight training table:
  - `python3 tools/weather/build_training_features_overnight.py`
- merged-table contract audit:
  - `python3 tools/weather/audit_training_features_overnight.py`

The checked-in merged-table registry lives in:

- `tools/weather/training_features_overnight_contract.py`

The audit is expected to verify:

- one row per `target_date_local` + `station_id`
- registry-column match
- prefix-policy match
- null-rate / availability-rate / variance metrics for `optional_candidate` columns

## Model Architecture

Use a **three-stage stacked overnight model**.

### Stage 1: Source-Aware Base Models

Train one model per source block:

- `model_nbm`
- `model_lamp`
- `model_hrrr`
- optional `model_wu_state`

Each base model should predict:

- `final_tmax_f`
- optionally raw bin probabilities

Training input policy:

- Stage 1 base models may train from source-specific feature views or from source-isolated slices derived from `training_features_overnight`
- Stage 1 base models should remain source-aware and should not require unrelated source blocks to be present in order to score a row

Why:

- each source has different strengths and missingness patterns
- source-aware models are more robust than one giant mixed learner as the initial design

### Stage 2: Blended Overnight Meta-Model

Inputs:

- Stage 1 predictions
- selected coverage, confidence, and missingness fields
- selected cross-source agreement features

Training input policy:

- Stage 2 trains from `training_features_overnight_normalized`
- the meta-model consumes normalized metadata features together with Stage 1 prediction outputs

Examples:

- `nbm_minus_lamp_day_max`
- `nbm_minus_hrrr_day_max`
- `lamp_revision_available`
- `hrrr_has_full_day_21_local_coverage`

Output:

- continuous `final_tmax_f`

Missing-source policy:

- if a source block is unavailable on a date, keep the row
- if LAMP is missing, `model_lamp` output may be null and LAMP missingness flags must be exposed to the meta-model
- if HRRR lacks full `21:00` local coverage, keep the row and expose HRRR coverage fields rather than dropping the date
- the meta-model must learn from explicit availability, missingness, and coverage signals instead of filtering to perfect-source dates only

### Stage 3: Bin Probability Head Plus Calibration

Use either:

- a direct bin classifier
- or binned conversion from the continuous model

Training input policy:

- Stage 3 trains from `training_features_overnight_normalized`
- calibration is fit on out-of-time validation predictions generated from the same normalized-table contract

Then calibrate on out-of-time validation with:

- isotonic-style calibration
- Dirichlet-style calibration when justified

Final overnight outputs at `12:05 a.m. America/New_York`:

- calibrated bin probabilities
- expected Tmax
- optional confidence score

## Explicit Exclusions

The overnight baseline should **not** include:

- RTMA or URMA
- nearby stations
- satellite or cloud imagery
- market or orderbook features

Those can be evaluated later, but they are not part of the overnight baseline described here.

## Repo Alignment

This design is intentionally aligned to current repo reality:

- Wunderground is the canonical settlement-aligned label source
- LAMP already has an overnight daily summarizer keyed by `target_date_local`
- HRRR already has an overnight local-day summary design with a `00:05` cutoff and retained-cycle revision logic
- HRRR operational optimization options are documented in `tools/hrrr/README.md`; they do not change the model target or the overnight summary/revision contract
- canonical weather normalization already exists under `tools/weather`

This document does not imply that unimplemented datasets or trading datasets already exist.

## Final Design

**one row per target local day at 12:05 a.m., combining Wunderground late-evening station state, NBM as the main overnight prior, LAMP as the station short-range summary, and the redesigned HRRR overnight summary/revision block, merged into a canonical training table and blended through a stacked ensemble into calibrated final Tmax bin probabilities.**
