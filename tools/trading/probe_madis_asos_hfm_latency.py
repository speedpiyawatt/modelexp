#!/usr/bin/env python3
"""Probe MADIS public ASOS-HFM latency for a station.

The script polls the MADIS public surface dump endpoint, extracts the newest
ASOS-HFM temperature record, and prints one JSON object per poll with HTTP
latency, observation age, and decoded temperature.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any


MADIS_URL = "https://madis-data.ncep.noaa.gov/madisPublic/cgi-bin/madisXmlPublicDir"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure MADIS ASOS-HFM response time and observation staleness."
    )
    parser.add_argument("--station", default="KLGA", help="MADIS station ID, e.g. KLGA.")
    parser.add_argument("--state", default="NY", help="State filter for MADIS form.")
    parser.add_argument(
        "--provider",
        default="ASOS-HFM",
        help="Provider to extract from MADIS records.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=12,
        help="Number of polls to run. Use 0 to run until interrupted.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds between polls.",
    )
    parser.add_argument(
        "--lookback-minutes",
        type=int,
        default=20,
        help="MADIS lookback window in minutes.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--timefilter",
        choices=("obs", "receipt"),
        default="receipt",
        help="Use observation-time or receipt-time filtering.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSONL output path. Records are appended as they are collected.",
    )
    return parser.parse_args()


def build_url(args: argparse.Namespace) -> str:
    timefilter = "1" if args.timefilter == "receipt" else "0"
    params = {
        "time": "0",
        "minbck": f"-{abs(args.lookback_minutes)}",
        "minfwd": "0",
        "recwin": "3",
        "timefilter": timefilter,
        "dfltrsel": "3",
        "state": args.state,
        "stanam": args.station.upper(),
        "stasel": "1",
        "pvdrsel": "0",
        "varsel": "2",
        "qctype": "0",
        "qcsel": "1",
        "xml": "1",
        "csvmiss": "0",
    }
    return f"{MADIS_URL}?{urllib.parse.urlencode(params)}"


def kelvin_to_f(value: float) -> float:
    return (value - 273.15) * 9.0 / 5.0 + 32.0


def parse_ob_time(value: str) -> datetime:
    # MADIS public dump emits GMT/UTC timestamps like 2026-04-27T09:30.
    return datetime.strptime(value, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)


def extract_records(body: bytes, provider: str) -> dict[str, dict[str, Any]]:
    root = ET.fromstring(body)
    selected: dict[str, dict[str, Any]] = {}
    for elem in root.findall("record"):
        attrs = elem.attrib
        if attrs.get("provider") != provider:
            continue
        var = attrs.get("var")
        ob_time_raw = attrs.get("ObTime")
        data_value_raw = attrs.get("data_value")
        if not var or not ob_time_raw or data_value_raw is None:
            continue
        try:
            ob_time = parse_ob_time(ob_time_raw)
            data_value = float(data_value_raw)
        except ValueError:
            continue
        prior = selected.get(var)
        if prior is None or ob_time > prior["ob_time"]:
            selected[var] = {
                "ob_time": ob_time,
                "data_value": data_value,
                "qcd": attrs.get("QCD"),
                "qca": attrs.get("QCA"),
                "qcr": attrs.get("QCR"),
            }
    return selected


def poll_once(args: argparse.Namespace, url: str) -> dict[str, Any]:
    started_monotonic = time.perf_counter()
    request_utc = datetime.now(timezone.utc)
    try:
        with urllib.request.urlopen(url, timeout=args.timeout) as response:
            body = response.read()
            status = response.status
    except Exception as exc:  # noqa: BLE001 - include network errors in JSONL.
        finished_utc = datetime.now(timezone.utc)
        return {
            "ok": False,
            "station": args.station.upper(),
            "provider": args.provider,
            "request_utc": request_utc.isoformat(),
            "response_utc": finished_utc.isoformat(),
            "http_seconds": round(time.perf_counter() - started_monotonic, 3),
            "error": repr(exc),
        }

    finished_utc = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "ok": True,
        "station": args.station.upper(),
        "provider": args.provider,
        "request_utc": request_utc.isoformat(),
        "response_utc": finished_utc.isoformat(),
        "http_status": status,
        "http_seconds": round(time.perf_counter() - started_monotonic, 3),
        "bytes": len(body),
    }

    try:
        records = extract_records(body, args.provider)
    except ET.ParseError as exc:
        record.update({"ok": False, "error": f"XML parse error: {exc}"})
        return record

    temp = records.get("V-T")
    if temp is None:
        record.update(
            {
                "has_temperature": False,
                "available_vars": sorted(records),
            }
        )
        return record

    ob_time = temp["ob_time"]
    age_seconds = (finished_utc - ob_time).total_seconds()
    temp_k = temp["data_value"]
    record.update(
        {
            "has_temperature": True,
            "ob_time_utc": ob_time.isoformat(),
            "obs_age_seconds": round(age_seconds, 1),
            "obs_age_minutes": round(age_seconds / 60.0, 2),
            "temp_k": round(temp_k, 3),
            "temp_f": round(kelvin_to_f(temp_k), 2),
            "qcd": temp["qcd"],
            "qca": temp["qca"],
            "qcr": temp["qcr"],
        }
    )
    return record


def emit(record: dict[str, Any], output_path: str | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> int:
    args = parse_args()
    url = build_url(args)
    emit({"event": "start", "url": url}, args.output)

    i = 0
    while args.count == 0 or i < args.count:
        i += 1
        record = poll_once(args, url)
        record["poll_index"] = i
        emit(record, args.output)
        if args.count != 0 and i >= args.count:
            break
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
