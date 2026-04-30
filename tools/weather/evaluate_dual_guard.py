#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Any

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.weather.dual_guard import EPSILON, blend_event_bins, blend_probability, risk_level, with_hrrr_weight


DEFAULT_BACKTEST_ROOT = pathlib.Path("data/runtime/server_dual_backtest_2026_0101_0429")
DEFAULT_LABELS_PATH = pathlib.Path("wunderground/output/tables/labels_daily.parquet")
DEFAULT_WITH_HRRR_HOLDOUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/full_holdout_local")
DEFAULT_NO_HRRR_HOLDOUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/no_hrrr_reference_holdout")
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/evaluation/dual_guard")
GUARD_CANDIDATES = [
    "always_no_hrrr",
    "always_with_hrrr",
    "with_hrrr_except_native_cold_hrrr_warm",
    "with_hrrr_only_high_or_very_high_disagreement",
    "probability_blend_by_regime",
    "expected_tmax_blend_by_regime",
]


@dataclass(frozen=True)
class Bin:
    label: str
    lower: int | None = None
    upper: int | None = None

    def contains(self, temp_f: float) -> bool:
        value = int(round(float(temp_f)))
        if self.lower is not None and value < self.lower:
            return False
        if self.upper is not None and value > self.upper:
            return False
        return True


def parse_bin(label: str) -> Bin:
    numbers = [int(value) for value in re.findall(r"\d+", label)]
    if not numbers:
        raise ValueError(f"could not parse event bin label: {label!r}")
    lowered = label.lower()
    if "or below" in lowered or "or lower" in lowered:
        return Bin(label=label, upper=numbers[0])
    if "or higher" in lowered or "or above" in lowered:
        return Bin(label=label, lower=numbers[0])
    if len(numbers) >= 2:
        lo, hi = sorted(numbers[:2])
        return Bin(label=label, lower=lo, upper=hi)
    return Bin(label=label, lower=numbers[0], upper=numbers[0])


def actual_bin(labels: list[str], temp_f: float) -> str:
    for label in labels:
        if parse_bin(label).contains(temp_f):
            return label
    raise ValueError(f"temperature {temp_f} did not map to any event bin: {labels}")


def prob_for(event_bins: list[dict[str, Any]], label: str) -> float:
    for row in event_bins:
        if row.get("bin") == label:
            return float(row.get("probability", 0.0))
    return 0.0


def top_bin(event_bins: list[dict[str, Any]]) -> str | None:
    if not event_bins:
        return None
    return str(max(event_bins, key=lambda row: float(row.get("probability", 0.0))).get("bin"))


def expected_column(df: pd.DataFrame, *, preferred: str, fallback: str) -> str:
    return preferred if preferred in df.columns else fallback


def load_2026_rows(backtest_root: pathlib.Path, labels_path: pathlib.Path) -> pd.DataFrame:
    prediction_root = backtest_root / "remote_predictions"
    if not prediction_root.exists():
        prediction_root = backtest_root
    labels = pd.read_parquet(labels_path)
    labels["target_date_local"] = labels["target_date_local"].astype(str)
    label_map = dict(zip(labels["target_date_local"], labels["label_final_tmax_f"]))
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(backtest_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        day = str(summary.get("target_date_local") or summary_path.parent.name)
        if summary.get("status") != "ok" or day not in label_map:
            continue
        no_path = prediction_root / day / "predictions" / "no_hrrr" / f"prediction_KLGA_{day}.json"
        with_path = prediction_root / day / "predictions" / "with_hrrr" / f"prediction_KLGA_{day}.json"
        if not no_path.exists() or not with_path.exists():
            continue
        no_payload = json.loads(no_path.read_text())
        with_payload = json.loads(with_path.read_text())
        labels_for_day = [str(row["bin"]) for row in with_payload.get("event_bins", [])]
        observed_bin = actual_bin(labels_for_day, float(label_map[day]))
        disagreement = with_payload.get("source_disagreement") if isinstance(with_payload.get("source_disagreement"), dict) else {}
        regime = str(disagreement.get("source_disagreement_regime") or "unknown")
        spread = disagreement.get("source_spread_f")
        risk = str(disagreement.get("source_disagreement_risk_level") or risk_level(regime, spread))
        rows.append(
            {
                "dataset": "2026_live_backtest",
                "target_date_local": day,
                "station_id": "KLGA",
                "final_tmax_f": float(label_map[day]),
                "observed_bin": observed_bin,
                "source_disagreement_regime": regime,
                "source_disagreement_risk_level": risk,
                "source_spread_f": spread,
                "no_expected_final_tmax_f": float(no_payload["expected_final_tmax_f"]),
                "with_expected_final_tmax_f": float(with_payload["expected_final_tmax_f"]),
                "no_observed_bin_probability": prob_for(no_payload.get("event_bins", []), observed_bin),
                "with_observed_bin_probability": prob_for(with_payload.get("event_bins", []), observed_bin),
                "no_top_correct": top_bin(no_payload.get("event_bins", [])) == observed_bin,
                "with_top_correct": top_bin(with_payload.get("event_bins", [])) == observed_bin,
                "no_event_bins": no_payload.get("event_bins", []),
                "with_event_bins": with_payload.get("event_bins", []),
                "has_top_fields": True,
            }
        )
    return pd.DataFrame(rows)


def load_validation_rows(with_hrrr_dir: pathlib.Path, no_hrrr_dir: pathlib.Path) -> pd.DataFrame:
    with_predictions = pd.read_parquet(with_hrrr_dir / "validation_predictions.parquet")
    no_predictions = pd.read_parquet(no_hrrr_dir / "validation_predictions.parquet")
    with_scores = pd.read_csv(with_hrrr_dir / "event_bin_scores.csv")
    no_scores = pd.read_csv(no_hrrr_dir / "event_bin_scores.csv")
    key = ["target_date_local", "station_id", "final_tmax_f"]
    for df in (with_predictions, no_predictions, with_scores, no_scores):
        df["target_date_local"] = df["target_date_local"].astype(str)
        df["station_id"] = df["station_id"].astype(str)
    merged = with_scores.merge(
        no_scores[key + ["observed_bin", "observed_bin_probability"]].rename(
            columns={
                "observed_bin_probability": "no_observed_bin_probability",
                "observed_bin": "no_observed_bin",
            }
        ),
        on=key,
        how="inner",
    ).rename(columns={"observed_bin_probability": "with_observed_bin_probability"})
    with_expected_column = expected_column(
        with_predictions,
        preferred="calibrated_pred_tmax_q50_f",
        fallback="pred_tmax_q50_f",
    )
    no_expected_column = expected_column(
        no_predictions,
        preferred="calibrated_pred_tmax_q50_f",
        fallback="pred_tmax_q50_f",
    )
    merged = merged.merge(
        with_predictions[
            key
            + [
                "source_disagreement_regime",
                "source_spread_f",
                with_expected_column,
            ]
        ].rename(columns={with_expected_column: "with_expected_final_tmax_f"}),
        on=key,
        how="left",
    )
    merged = merged.merge(
        no_predictions[key + [no_expected_column]].rename(columns={no_expected_column: "no_expected_final_tmax_f"}),
        on=key,
        how="left",
    )
    merged["dataset"] = "validation_holdout"
    merged["source_disagreement_regime"] = merged["source_disagreement_regime"].fillna("unknown").astype(str)
    merged["source_disagreement_risk_level"] = [
        risk_level(regime, spread)
        for regime, spread in zip(merged["source_disagreement_regime"], merged["source_spread_f"])
    ]
    merged["no_top_correct"] = pd.NA
    merged["with_top_correct"] = pd.NA
    merged["no_event_bins"] = pd.NA
    merged["with_event_bins"] = pd.NA
    merged["has_top_fields"] = False
    merged["with_expected_source_column"] = with_expected_column
    merged["no_expected_source_column"] = no_expected_column
    return merged[
        [
            "dataset",
            "target_date_local",
            "station_id",
            "final_tmax_f",
            "observed_bin",
            "source_disagreement_regime",
            "source_disagreement_risk_level",
            "source_spread_f",
            "no_expected_final_tmax_f",
            "with_expected_final_tmax_f",
            "no_observed_bin_probability",
            "with_observed_bin_probability",
            "no_top_correct",
            "with_top_correct",
            "no_event_bins",
            "with_event_bins",
            "has_top_fields",
            "with_expected_source_column",
            "no_expected_source_column",
        ]
    ]


def score_candidate(rows: pd.DataFrame, candidate_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    scored = rows.copy()
    weights = [
        with_hrrr_weight(candidate_id, regime=regime, risk=risk)
        for regime, risk in zip(scored["source_disagreement_regime"], scored["source_disagreement_risk_level"])
    ]
    if candidate_id == "expected_tmax_blend_by_regime":
        probability_weights = [
            0.0 if str(regime) == "native_cold_hrrr_warm" else 1.0
            for regime in scored["source_disagreement_regime"]
        ]
        expected_weights = weights
    else:
        probability_weights = weights
        expected_weights = weights
    scored["with_hrrr_probability_weight"] = probability_weights
    scored["with_hrrr_expected_weight"] = expected_weights
    scored["observed_bin_probability"] = [
        blend_probability(no_probability, with_probability, weight=weight)
        for no_probability, with_probability, weight in zip(
            scored["no_observed_bin_probability"],
            scored["with_observed_bin_probability"],
            scored["with_hrrr_probability_weight"],
        )
    ]
    scored["expected_final_tmax_f"] = [
        blend_probability(no_expected, with_expected, weight=weight)
        for no_expected, with_expected, weight in zip(
            scored["no_expected_final_tmax_f"],
            scored["with_expected_final_tmax_f"],
            scored["with_hrrr_expected_weight"],
        )
    ]
    scored["abs_error_f"] = (pd.to_numeric(scored["expected_final_tmax_f"]) - pd.to_numeric(scored["final_tmax_f"])).abs()
    scored["negative_log_likelihood"] = -scored["observed_bin_probability"].clip(lower=EPSILON).map(math.log)
    top_correct_values = []
    for _, row in scored.iterrows():
        if not bool(row["has_top_fields"]):
            top_correct_values.append(pd.NA)
            continue
        weight = float(row["with_hrrr_probability_weight"])
        if weight == 1.0:
            top_correct_values.append(row["with_top_correct"])
        elif weight == 0.0:
            top_correct_values.append(row["no_top_correct"])
        else:
            no_bins = row.get("no_event_bins")
            with_bins = row.get("with_event_bins")
            if not isinstance(no_bins, list) or not isinstance(with_bins, list):
                top_correct_values.append(pd.NA)
                continue
            blended_top = top_bin(blend_event_bins(no_bins, with_bins, weight=weight))
            top_correct_values.append(blended_top == row["observed_bin"] if blended_top is not None else pd.NA)
    scored["top_correct"] = top_correct_values
    metrics = {
        "candidate_id": candidate_id,
        "row_count": int(len(scored)),
        "event_bin_nll": float(scored["negative_log_likelihood"].mean()),
        "event_bin_observed_probability": float(scored["observed_bin_probability"].mean()),
        "expected_tmax_mae_f": float(scored["abs_error_f"].mean()),
        "with_hrrr_probability_weight_mean": float(pd.to_numeric(scored["with_hrrr_probability_weight"]).mean()),
    }
    top = scored["top_correct"].dropna()
    metrics["top_bin_accuracy"] = float(top.astype(bool).mean()) if not top.empty else None
    return metrics, scored


def choose_candidate(summary: pd.DataFrame, *, selection_dataset: str, confirmation_dataset: str | None = None) -> str:
    selection = summary.loc[summary["dataset"] == selection_dataset].copy()
    baseline = selection.loc[selection["candidate_id"] == "always_with_hrrr"].iloc[0]
    candidates = selection.loc[
        (selection["candidate_id"] != "always_with_hrrr")
        & (selection["event_bin_nll"] <= float(baseline["event_bin_nll"]) + 0.01)
        & (selection["event_bin_observed_probability"] >= float(baseline["event_bin_observed_probability"]))
        & (selection["expected_tmax_mae_f"] <= float(baseline["expected_tmax_mae_f"]))
    ].copy()
    if confirmation_dataset is not None and confirmation_dataset in set(summary["dataset"]):
        confirmation = summary.loc[summary["dataset"] == confirmation_dataset].set_index("candidate_id")
        confirmation_baseline = confirmation.loc["always_with_hrrr"]
        keep_ids = []
        for candidate_id in candidates["candidate_id"]:
            if candidate_id not in confirmation.index:
                continue
            row = confirmation.loc[candidate_id]
            if float(row["event_bin_nll"]) > float(confirmation_baseline["event_bin_nll"]):
                continue
            if float(row["event_bin_observed_probability"]) < float(confirmation_baseline["event_bin_observed_probability"]):
                continue
            if float(row["expected_tmax_mae_f"]) > float(confirmation_baseline["expected_tmax_mae_f"]):
                continue
            top = row.get("top_bin_accuracy")
            baseline_top = confirmation_baseline.get("top_bin_accuracy")
            if pd.notna(top) and pd.notna(baseline_top) and float(top) < float(baseline_top):
                continue
            keep_ids.append(candidate_id)
        candidates = candidates.loc[candidates["candidate_id"].isin(keep_ids)].copy()
    if candidates.empty:
        return "always_with_hrrr"
    candidates["selection_rank"] = candidates["candidate_id"].map(
        {
            "always_with_hrrr": 0,
            "with_hrrr_except_native_cold_hrrr_warm": 1,
            "probability_blend_by_regime": 2,
            "expected_tmax_blend_by_regime": 3,
            "with_hrrr_only_high_or_very_high_disagreement": 4,
            "always_no_hrrr": 5,
        }
    ).fillna(99)
    selected = candidates.sort_values(
        ["event_bin_nll", "expected_tmax_mae_f", "selection_rank"], ascending=[True, True, True]
    ).iloc[0]
    return str(selected["candidate_id"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate no-HRRR/with-HRRR conditional guard candidates.")
    parser.add_argument("--backtest-root", type=pathlib.Path, default=DEFAULT_BACKTEST_ROOT)
    parser.add_argument("--labels-path", type=pathlib.Path, default=DEFAULT_LABELS_PATH)
    parser.add_argument("--with-hrrr-holdout-dir", type=pathlib.Path, default=DEFAULT_WITH_HRRR_HOLDOUT_DIR)
    parser.add_argument("--no-hrrr-holdout-dir", type=pathlib.Path, default=DEFAULT_NO_HRRR_HOLDOUT_DIR)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = []
    if args.with_hrrr_holdout_dir.exists() and args.no_hrrr_holdout_dir.exists():
        datasets.append(load_validation_rows(args.with_hrrr_holdout_dir, args.no_hrrr_holdout_dir))
    if args.backtest_root.exists() and args.labels_path.exists():
        backtest = load_2026_rows(args.backtest_root, args.labels_path)
        if not backtest.empty:
            datasets.append(backtest)
    if not datasets:
        raise SystemExit("no dual-guard evaluation rows were available")
    rows = pd.concat(datasets, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_scored = []
    summary_rows = []
    slice_rows = []
    for dataset_name, dataset_rows in rows.groupby("dataset", dropna=False):
        for candidate_id in GUARD_CANDIDATES:
            metrics, scored = score_candidate(dataset_rows, candidate_id)
            metrics["dataset"] = dataset_name
            summary_rows.append(metrics)
            scored["candidate_id"] = candidate_id
            all_scored.append(scored)
            for regime, group in scored.groupby("source_disagreement_regime", dropna=False):
                slice_rows.append(
                    {
                        "dataset": dataset_name,
                        "candidate_id": candidate_id,
                        "slice": f"regime:{regime}",
                        "row_count": int(len(group)),
                        "event_bin_nll": float(group["negative_log_likelihood"].mean()),
                        "event_bin_observed_probability": float(group["observed_bin_probability"].mean()),
                        "expected_tmax_mae_f": float(group["abs_error_f"].mean()),
                    }
                )
    summary = pd.DataFrame(summary_rows).sort_values(["dataset", "event_bin_nll", "expected_tmax_mae_f"])
    slices = pd.DataFrame(slice_rows).sort_values(["dataset", "slice", "event_bin_nll"])
    scored_rows = pd.concat(all_scored, ignore_index=True)
    summary.to_csv(args.output_dir / "dual_guard_summary.csv", index=False)
    slices.to_csv(args.output_dir / "dual_guard_slices.csv", index=False)
    scored_rows.to_parquet(args.output_dir / "dual_guard_scored_rows.parquet", index=False)
    selection_dataset = "validation_holdout" if "validation_holdout" in set(summary["dataset"]) else str(summary["dataset"].iloc[0])
    confirmation_dataset = "2026_live_backtest" if "2026_live_backtest" in set(summary["dataset"]) else None
    selected_candidate = choose_candidate(
        summary,
        selection_dataset=selection_dataset,
        confirmation_dataset=confirmation_dataset,
    )
    manifest = {
        "status": "ok",
        "selected_candidate_id": selected_candidate,
        "selection_dataset": selection_dataset,
        "confirmation_dataset": confirmation_dataset,
        "candidate_ids": GUARD_CANDIDATES,
        "summary_path": str(args.output_dir / "dual_guard_summary.csv"),
        "slice_metrics_path": str(args.output_dir / "dual_guard_slices.csv"),
        "scored_rows_path": str(args.output_dir / "dual_guard_scored_rows.parquet"),
        "selection_rule": {
            "primary": "event_bin_nll",
            "guards": [
                "candidate_event_bin_nll_within_0.01_of_always_with_hrrr",
                "observed_bin_probability_not_below_always_with_hrrr",
                "expected_tmax_mae_not_worse_than_always_with_hrrr",
                "when_2026_live_backtest_is_available_candidate_must_not_worsen_2026_nll_observed_probability_mae_or_top_bin_accuracy",
            ],
        },
        "notes": [
            "validation_holdout expected_tmax metrics prefer calibrated q50 when present and fall back to raw q50 because archived holdout prediction JSONs are not retained",
            "2026_live_backtest metrics use the retained live prediction JSON expected_final_tmax_f values and actual event bins",
        ],
    }
    (args.output_dir / "dual_guard_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"selected_candidate_id={selected_candidate}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
