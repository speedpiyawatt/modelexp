#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import tempfile
import sys

import pandas as pd


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.training_features_overnight_contract import REGISTRY_ROWS as SOURCE_REGISTRY_ROWS
from tools.weather.training_features_overnight_normalized_contract import (
    NORMALIZATION_VERSION,
    REGISTRY_ROWS,
    registry_by_name,
    registry_columns,
)


DEFAULT_INPUT_PATH = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight.parquet")
DEFAULT_INPUT_ROOT = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight")
DEFAULT_OUTPUT_PATH = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight_normalized.parquet")
DEFAULT_OUTPUT_DIR = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight_normalized")
DEFAULT_MANIFEST_PATH = pathlib.Path("tools/weather/data/runtime/training/training_features_overnight_normalized.manifest.json")
DEFAULT_VOCAB_PATH = pathlib.Path("tools/weather/training_feature_vocabularies.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the model-ready normalized overnight KLGA training table from the frozen merged contract table.")
    parser.add_argument("--input-path", type=pathlib.Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-root", type=pathlib.Path, default=None)
    parser.add_argument("--output-path", type=pathlib.Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--manifest-path", type=pathlib.Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--manifest-output-path", type=pathlib.Path, default=None)
    parser.add_argument("--start-local-date", default=None)
    parser.add_argument("--end-local-date", default=None)
    parser.add_argument("--vocab-path", type=pathlib.Path, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--allow-empty", action="store_true", help="Allow empty inputs and emit an empty schema-only normalized output.")
    return parser.parse_args()


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def kelvin_to_f(series: pd.Series) -> pd.Series:
    numeric = _numeric(series)
    return (numeric - 273.15) * (9.0 / 5.0) + 32.0


def temp_delta_k_to_f(series: pd.Series) -> pd.Series:
    return _numeric(series) * (9.0 / 5.0)


def ms_to_mph(series: pd.Series) -> pd.Series:
    return _numeric(series) * 2.2369362920544


def kt_to_mph(series: pd.Series) -> pd.Series:
    return _numeric(series) * 1.150779448


def m_to_mi(series: pd.Series) -> pd.Series:
    return _numeric(series) / 1609.344


def m_to_ft(series: pd.Series) -> pd.Series:
    return _numeric(series) * 3.2808398950131


def hundreds_ft_to_ft(series: pd.Series) -> pd.Series:
    return _numeric(series) * 100.0


def kg_m2_to_in(series: pd.Series) -> pd.Series:
    return _numeric(series) / 25.4


def load_vocabularies(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _normalize_cloud_cover_token(value: object) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    token = str(value).strip().upper()
    if not token:
        return "__MISSING__"
    aliases = {"C": "CLR", "CLEAR": "CLR", "BK": "BKN", "OV": "OVC"}
    return aliases.get(token, token)


def _normalize_weather_family_token(value: object) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    token = str(value).strip().upper()
    if not token or token in {"NONE", "NO WX", "NO WEATHER"}:
        return "NONE"
    explicit = {
        "HZ": "HAZE",
        "BR": "MIST",
        "FG": "FOG",
        "FZFG": "FOG",
        "RA": "RAIN",
        "SHRA": "RAIN",
        "SN": "SNOW",
        "DZ": "RAIN",
        "FZDZ": "FREEZING",
        "FZRA": "FREEZING",
        "PL": "FREEZING",
        "TS": "THUNDER",
        "TSRA": "THUNDER",
    }
    if token in explicit:
        return explicit[token]
    for needle, family in (
        ("THUNDER", "THUNDER"),
        ("FREEZ", "FREEZING"),
        ("SNOW", "SNOW"),
        ("RAIN", "RAIN"),
        ("SHOWER", "RAIN"),
        ("DRIZZLE", "RAIN"),
        ("FOG", "FOG"),
        ("MIST", "MIST"),
        ("HAZE", "HAZE"),
        ("SMOKE", "HAZE"),
        ("CLOUD", "NONE"),
        ("FAIR", "NONE"),
        ("CLEAR", "NONE"),
    ):
        if needle in token:
            return family
    return "__UNK__"


def _normalize_precip_type_token(value: object) -> str:
    if value is None or pd.isna(value):
        return "__MISSING__"
    token = str(value).strip().upper()
    if not token:
        return "__MISSING__"
    if token in {"NONE", "NIL"}:
        return "NONE"
    aliases = {"RW": "R", "RS": "R", "SW": "S", "ZR": "ZR", "PL": "IP", "T": "T", "TS": "T"}
    return aliases.get(token, token)


def _encode_vocab(series: pd.Series, vocab: dict[str, int], normalizer) -> pd.Series:
    def encode(value: object) -> int:
        token = normalizer(value)
        return int(vocab.get(token, vocab["__UNK__"]))

    return series.map(encode).astype("Int64")


def validate_merged_input(df: pd.DataFrame, *, input_path: pathlib.Path, allow_empty: bool) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"merged overnight training input does not exist: {input_path}")
    if df.empty and not allow_empty:
        raise ValueError(f"merged overnight training input is empty: {input_path}")
    required_columns = [str(row["column_name"]) for row in SOURCE_REGISTRY_ROWS if str(row["freeze_level"]) == "core_required"]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"merged overnight training input is missing required core columns: {missing}")


def normalize_training_features_overnight(df: pd.DataFrame, vocabularies: dict[str, object]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=registry_columns())

    normalized_columns: dict[str, pd.Series] = {}
    for row in REGISTRY_ROWS:
        output_column = str(row["column_name"])
        source_column = str(row["source_column"])
        transform = str(row["transform"])
        source_series = df[source_column] if source_column in df.columns else pd.Series(pd.NA, index=df.index)

        if transform == "copy":
            normalized_columns[output_column] = source_series
        elif transform == "copy_numeric":
            normalized_columns[output_column] = _numeric(source_series)
        elif transform == "k_to_f":
            normalized_columns[output_column] = kelvin_to_f(source_series)
        elif transform == "temp_delta_k_to_f":
            normalized_columns[output_column] = temp_delta_k_to_f(source_series)
        elif transform == "ms_to_mph":
            normalized_columns[output_column] = ms_to_mph(source_series)
        elif transform == "kt_to_mph":
            normalized_columns[output_column] = kt_to_mph(source_series)
        elif transform == "m_to_mi":
            normalized_columns[output_column] = m_to_mi(source_series)
        elif transform == "m_to_ft":
            normalized_columns[output_column] = m_to_ft(source_series)
        elif transform == "hundreds_ft_to_ft":
            normalized_columns[output_column] = hundreds_ft_to_ft(source_series)
        elif transform == "kg_m2_to_in":
            normalized_columns[output_column] = kg_m2_to_in(source_series)
        elif transform == "cloud_cover_id":
            normalized_columns[output_column] = _encode_vocab(source_series, vocabularies["cloud_cover"], _normalize_cloud_cover_token)
        elif transform == "weather_family_id":
            normalized_columns[output_column] = _encode_vocab(source_series, vocabularies["weather_family"], _normalize_weather_family_token)
        elif transform == "precip_type_id":
            normalized_columns[output_column] = _encode_vocab(source_series, vocabularies["precip_type"], _normalize_precip_type_token)
        else:
            raise ValueError(f"unsupported normalization transform: {transform}")

    normalized = pd.DataFrame(normalized_columns, index=df.index)
    for row in REGISTRY_ROWS:
        column = str(row["column_name"])
        dtype = str(row["dtype"])
        if dtype == "bool" and column in normalized.columns:
            normalized[column] = normalized[column].astype("boolean")
        elif dtype == "int64" and column in normalized.columns and str(row["transform"]).endswith("_id"):
            normalized[column] = normalized[column].astype("Int64")
    return normalized.loc[:, registry_columns()].reset_index(drop=True)


def build_manifest(*, input_path: pathlib.Path, output_df: pd.DataFrame, vocabularies: dict[str, object]) -> dict[str, object]:
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "input_path": str(input_path),
        "row_count": int(len(output_df)),
        "column_count": int(len(output_df.columns)),
        "vocab_version": vocabularies.get("version"),
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def iter_local_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    return [start_date + dt.timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def output_paths_for_date(output_dir: pathlib.Path, target_date_local: str) -> tuple[pathlib.Path, pathlib.Path]:
    root = output_dir / f"target_date_local={target_date_local}"
    return root / "part.parquet", root / "manifest.json"


def _write_atomic_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
        temp_path = pathlib.Path(handle.name)
    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def _write_atomic_json(path: pathlib.Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent, delete=False, mode="w") as handle:
        temp_path = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
    try:
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def load_date_input(input_root: pathlib.Path, target_date_local: str) -> pd.DataFrame:
    path = input_root / f"target_date_local={target_date_local}" / "part.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def build_date_manifest(
    *,
    target_date_local: str,
    output_path: pathlib.Path,
    output_df: pd.DataFrame,
    vocabularies: dict[str, object],
) -> dict[str, object]:
    return {
        "status": "ok",
        "normalization_version": NORMALIZATION_VERSION,
        "target_date_local": target_date_local,
        "row_count": int(len(output_df)),
        "column_count": int(len(output_df.columns)),
        "output_path": str(output_path),
        "vocab_version": vocabularies.get("version"),
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def validate_date_output(
    *,
    output_path: pathlib.Path,
    manifest_path: pathlib.Path,
    target_date_local: str,
    allow_empty: bool,
) -> bool:
    if not output_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        df = pd.read_parquet(output_path)
    except Exception:
        return False
    if manifest.get("status") != "ok":
        return False
    if manifest.get("target_date_local") != target_date_local:
        return False
    if int(manifest.get("row_count", -1)) != int(len(df)):
        return False
    if list(df.columns) != registry_columns():
        return False
    if df.empty:
        return allow_empty
    if len(df) != 1:
        return False
    target_dates = df["target_date_local"].astype(str)
    if not (target_dates == target_date_local).all():
        return False
    return True


def main() -> int:
    args = parse_args()
    if args.output_dir is not None and args.output_path != DEFAULT_OUTPUT_PATH:
        raise ValueError("Use either --output-path or --output-dir, not both.")
    if args.input_root is not None and args.input_path != DEFAULT_INPUT_PATH:
        raise ValueError("Use either --input-path or --input-root, not both.")
    if (args.start_local_date is None) ^ (args.end_local_date is None):
        raise ValueError("Provide both --start-local-date and --end-local-date when using daily output mode.")
    vocabularies = load_vocabularies(args.vocab_path)
    if args.output_dir is None:
        input_df = pd.read_parquet(args.input_path) if args.input_path.exists() else pd.DataFrame()
        validate_merged_input(input_df, input_path=args.input_path, allow_empty=args.allow_empty)
        output_df = normalize_training_features_overnight(input_df, vocabularies)
        if output_df.empty and not args.allow_empty:
            raise ValueError("training_features_overnight_normalized build produced zero rows; rerun with --allow-empty only for schema/bootstrap workflows")
        manifest = build_manifest(input_path=args.input_path, output_df=output_df, vocabularies=vocabularies)
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(args.output_path, index=False)
        args.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        print(args.output_path)
        print(args.manifest_path)
        return 0

    if args.input_root is None:
        raise ValueError("--input-root is required when using --output-dir daily mode.")
    start_date = parse_local_date(args.start_local_date)
    end_date = parse_local_date(args.end_local_date)
    if end_date < start_date:
        raise ValueError("--end-local-date must be on or after --start-local-date")

    written: list[pathlib.Path] = []
    target_dates = iter_local_dates(start_date, end_date)
    for target_date in target_dates:
        target_token = target_date.isoformat()
        input_df = load_date_input(args.input_root, target_token)
        input_path = args.input_root / f"target_date_local={target_token}" / "part.parquet"
        validate_merged_input(input_df, input_path=input_path, allow_empty=args.allow_empty)
        output_df = normalize_training_features_overnight(input_df, vocabularies)
        if output_df.empty and not args.allow_empty:
            raise ValueError(
                f"training_features_overnight_normalized build produced zero rows for target_date_local={target_token}; "
                "rerun with --allow-empty only for schema/bootstrap workflows"
            )
        output_path, manifest_path = output_paths_for_date(args.output_dir, target_token)
        manifest = build_date_manifest(
            target_date_local=target_token,
            output_path=output_path,
            output_df=output_df,
            vocabularies=vocabularies,
        )
        _write_atomic_parquet(output_df, output_path)
        _write_atomic_json(manifest_path, manifest)
        if not validate_date_output(
            output_path=output_path,
            manifest_path=manifest_path,
            target_date_local=target_token,
            allow_empty=args.allow_empty,
        ):
            raise ValueError(f"daily normalized output validation failed for target_date_local={target_token}")
        written.extend([output_path, manifest_path])

    if args.manifest_output_path is not None:
        summary_payload = {
            "status": "ok",
            "start_local_date": start_date.isoformat(),
            "end_local_date": end_date.isoformat(),
            "target_date_count": len(target_dates),
            "written_paths": [str(path) for path in written],
            "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        _write_atomic_json(args.manifest_output_path, summary_payload)
        print(args.manifest_output_path)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
