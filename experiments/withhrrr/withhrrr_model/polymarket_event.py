from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import urllib.parse
import urllib.request
from typing import Any

from .event_bins import parse_event_bin


GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
DEFAULT_OUTPUT_DIR = pathlib.Path("experiments/withhrrr/data/runtime/polymarket")
WEATHER_EVENT_SLUG_PREFIX = "highest-temperature-in-nyc-on"
BIN_PATTERN = re.compile(
    r"(?P<label>\d+\s*(?:-|to|through)\s*\d+\s*°?\s*F|\d+\s*°?\s*F?\s*(?:or\s+below|or\s+lower|or\s+under|or\s+above|or\s+higher|\+)|(?:under|below|less than|over|above|greater than)\s+\d+\s*°?\s*F?)",
    re.IGNORECASE,
)
MONTH_SLUGS = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Polymarket event metadata and extract weather event-bin labels.")
    parser.add_argument("--event-slug", required=True, help="Polymarket event slug, e.g. highest-temperature-in-nyc-on-april-11-2026.")
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input-json", type=pathlib.Path, help="Optional local event JSON for offline parsing/tests.")
    return parser.parse_args()


def fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "withhrrr-polymarket-event-fetcher/1.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def weather_event_slug_for_date(target_date: dt.date) -> str:
    month = MONTH_SLUGS[target_date.month]
    return f"{WEATHER_EVENT_SLUG_PREFIX}-{month}-{target_date.day}-{target_date.year}"


def fetch_event_by_slug(slug: str) -> dict[str, Any]:
    quoted = urllib.parse.quote(slug, safe="")
    try:
        payload = fetch_json(f"{GAMMA_BASE_URL}/events/slug/{quoted}")
    except Exception:
        payload = fetch_json(f"{GAMMA_BASE_URL}/events?slug={quoted}")
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"no Polymarket event found for slug={slug!r}")
        return payload[0]
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected Polymarket event response type for slug={slug!r}: {type(payload).__name__}")
    return payload


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def candidate_market_text(market: dict[str, Any]) -> list[str]:
    fields = (
        "groupItemTitle",
        "groupItemThreshold",
        "question",
        "title",
        "slug",
        "description",
        "resolutionCriteria",
        "rules",
    )
    values = [str(market[field]) for field in fields if market.get(field)]
    for outcome in _json_array(market.get("outcomes")):
        if isinstance(outcome, str) and outcome.lower() not in {"yes", "no"}:
            values.append(outcome)
    return values


def normalize_bin_label(label: str) -> str:
    text = re.sub(r"\s+", " ", label.strip())
    text = text.replace("°", "")
    text = re.sub(r"\s*F\b", "F", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\bf$", "F", text)
    text = text.replace(" to ", "-")
    text = text.replace(" through ", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    if re.fullmatch(r"\d+-\d+\s*F?", text, re.IGNORECASE):
        return text.upper().replace("F", "") + "F"
    return text


def extract_bin_label_from_market(market: dict[str, Any]) -> str | None:
    for text in candidate_market_text(market):
        match = BIN_PATTERN.search(text)
        if not match:
            continue
        label = normalize_bin_label(match.group("label"))
        try:
            parse_event_bin(label)
        except ValueError:
            continue
        return label
    return None


def market_sort_key(label: str) -> tuple[float, float, str]:
    event_bin = parse_event_bin(label)
    lower = float("-inf") if event_bin.lower_f is None else float(event_bin.lower_f)
    upper = float("inf") if event_bin.upper_f is None else float(event_bin.upper_f)
    return lower, upper, label


def extract_event_bins(event: dict[str, Any]) -> list[dict[str, Any]]:
    markets = event.get("markets") or event.get("eventMarkets") or []
    if not isinstance(markets, list):
        raise ValueError("Polymarket event payload does not contain a markets list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for market in markets:
        if not isinstance(market, dict):
            continue
        label = extract_bin_label_from_market(market)
        if label is None or label in seen:
            continue
        seen.add(label)
        rows.append(
            {
                "label": label,
                "market_id": market.get("id"),
                "market_slug": market.get("slug"),
                "question": market.get("question") or market.get("title"),
                "condition_id": market.get("conditionId") or market.get("condition_id"),
                "clob_token_ids": market.get("clobTokenIds") or market.get("clob_token_ids"),
                "outcomes": market.get("outcomes"),
                "outcome_prices": market.get("outcomePrices") or market.get("outcome_prices"),
            }
        )
    rows.sort(key=lambda row: market_sort_key(str(row["label"])))
    if not rows:
        raise ValueError("no parseable temperature bin markets found in Polymarket event payload")
    return rows


def write_outputs(*, output_dir: pathlib.Path, slug: str, event: dict[str, Any], bins: list[dict[str, Any]]) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    root = output_dir / f"event_slug={slug}"
    root.mkdir(parents=True, exist_ok=True)
    event_path = root / "polymarket_event.json"
    bins_path = root / "event_bins.json"
    manifest_path = root / "event_bins.manifest.json"
    event_path.write_text(json.dumps(event, indent=2, sort_keys=True) + "\n")
    bins_path.write_text(json.dumps({"bins": bins, "labels": [row["label"] for row in bins]}, indent=2, sort_keys=True) + "\n")
    manifest = {
        "status": "ok",
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "event_slug": slug,
        "event_id": event.get("id"),
        "event_title": event.get("title") or event.get("question"),
        "bin_count": len(bins),
        "event_path": str(event_path),
        "bins_path": str(bins_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return event_path, bins_path, manifest_path


def main() -> int:
    args = parse_args()
    if args.input_json is not None:
        event = json.loads(args.input_json.read_text())
    else:
        event = fetch_event_by_slug(args.event_slug)
    bins = extract_event_bins(event)
    _, bins_path, manifest_path = write_outputs(output_dir=args.output_dir, slug=args.event_slug, event=event, bins=bins)
    print(bins_path)
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
