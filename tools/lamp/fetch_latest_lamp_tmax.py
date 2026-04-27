#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from zoneinfo import ZoneInfo


NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/lmp/prod"
UTC = dt.timezone.utc
NY_TZ = ZoneInfo("America/New_York")

HEADER_RE = re.compile(
    r"^\s*(?P<station>[A-Z0-9]{4})\s+GFS LAMP GUIDANCE\s+"
    r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\s+(?P<hhmm>\d{4})\s+UTC\s*$"
)
DIR_DATE_RE = re.compile(r'href="lmp\.(?P<date>\d{8})/"')
FULL_CYCLE_RE = re.compile(r'href="lmp\.t(?P<cycle>\d{2}30)z\.lavtxt\.ascii"')


@dataclass(frozen=True)
class LampFile:
    url: str
    text: str
    last_modified: dt.datetime | None


@dataclass(frozen=True)
class ForecastPoint:
    forecast_hour: int
    valid_time_utc: dt.datetime
    valid_time_local: dt.datetime
    tmp_f: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch the latest full NOAA LAMP station text cycle and report Tmax guidance."
    )
    parser.add_argument("--station", default="KLGA", help="Station id, default KLGA.")
    parser.add_argument(
        "--target-date-local",
        default=None,
        help="Local station date to summarize in YYYY-MM-DD. Default: issue date in America/New_York.",
    )
    parser.add_argument(
        "--latest-before-local",
        default=None,
        help=(
            "Select the latest full HH30 issue strictly before this America/New_York local timestamp. "
            "Use YYYY-MM-DDTHH:MM, for example 2026-04-27T00:00."
        ),
    )
    parser.add_argument(
        "--date-utc",
        default=None,
        help="UTC LAMP directory date in YYYY-MM-DD or YYYYMMDD. Default: latest available NOMADS directory.",
    )
    parser.add_argument(
        "--cycle",
        default=None,
        help="UTC full LAMP cycle as HH30 or HH. Default: latest available HH30 cycle for the date.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0, help="Per-request network timeout.")
    parser.add_argument("--attempts", type=int, default=3, help="Fetch attempts per URL.")
    parser.add_argument("--show-hourly", action="store_true", help="Print the hourly TMP forecast rows.")
    return parser.parse_args()


def fetch_text(url: str, *, timeout: float, attempts: int) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "klga-lamp-tmax/1.0"})
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def fetch_lamp_file(url: str, *, timeout: float, attempts: int) -> LampFile:
    request = urllib.request.Request(url, headers={"User-Agent": "klga-lamp-tmax/1.0"})
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw_last_modified = response.headers.get("Last-Modified")
                last_modified = None
                if raw_last_modified:
                    parsed = email.utils.parsedate_to_datetime(raw_last_modified)
                    last_modified = parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
                text = response.read().decode("utf-8", errors="replace")
            return LampFile(url=url, text=text, last_modified=last_modified)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def normalize_date_token(value: str) -> str:
    cleaned = value.replace("-", "")
    dt.datetime.strptime(cleaned, "%Y%m%d")
    return cleaned


def normalize_cycle_token(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) == 2:
        digits = f"{digits}30"
    if not re.fullmatch(r"\d{2}30", digits):
        raise ValueError(f"Full LAMP station cycles use HH30 tokens; got {value!r}")
    hour = int(digits[:2])
    if hour > 23:
        raise ValueError(f"Invalid cycle hour in {value!r}")
    return digits


def parse_local_cutoff(value: str) -> dt.datetime:
    cleaned = value.strip().replace(" ", "T")
    timestamp = dt.datetime.fromisoformat(cleaned)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=NY_TZ)
    return timestamp.astimezone(NY_TZ)


def issue_tokens_before_local_cutoff(value: str) -> tuple[str, str]:
    cutoff_local = parse_local_cutoff(value)
    cutoff_utc = cutoff_local.astimezone(UTC)
    candidate = cutoff_utc.replace(minute=30, second=0, microsecond=0)
    if candidate >= cutoff_utc:
        candidate -= dt.timedelta(hours=1)
    return candidate.strftime("%Y%m%d"), candidate.strftime("%H%M")


def latest_date_token(*, timeout: float, attempts: int) -> str:
    listing = fetch_text(f"{NOMADS_BASE}/", timeout=timeout, attempts=attempts)
    dates = sorted({match.group("date") for match in DIR_DATE_RE.finditer(listing)})
    if not dates:
        raise RuntimeError("No lmp.YYYYMMDD directories found on NOMADS.")
    return dates[-1]


def latest_cycle_token(date_token: str, *, timeout: float, attempts: int) -> str:
    listing = fetch_text(f"{NOMADS_BASE}/lmp.{date_token}/", timeout=timeout, attempts=attempts)
    cycles = sorted({match.group("cycle") for match in FULL_CYCLE_RE.finditer(listing)})
    if not cycles:
        raise RuntimeError(f"No full HH30 LAMP lavtxt cycles found for lmp.{date_token}.")
    return cycles[-1]


def issue_time_from_header(line: str) -> dt.datetime | None:
    match = HEADER_RE.match(line.rstrip())
    if not match:
        return None
    parts = match.groupdict()
    hhmm = parts["hhmm"]
    return dt.datetime(
        int(parts["year"]),
        int(parts["month"]),
        int(parts["day"]),
        int(hhmm[:2]),
        int(hhmm[2:]),
        tzinfo=UTC,
    )


def parse_fixed_width_cells(line: str, count: int | None = None) -> tuple[str, list[str | None]]:
    normalized = line.rstrip("\n")
    label = normalized[:4].strip()
    body = normalized[5:] if len(normalized) > 5 else ""
    if count is None:
        width = len(body)
        if width % 3:
            width += 3 - (width % 3)
        body = body.ljust(width)
    else:
        body = body.ljust(count * 3)
        width = count * 3
    cells = [body[index : index + 3].strip() or None for index in range(0, width, 3)]
    return label, cells


def extract_station_block(text: str, station: str) -> tuple[str, list[str]]:
    current_header: str | None = None
    current_lines: list[str] = []
    station = station.upper()
    for line in text.splitlines():
        if HEADER_RE.match(line.rstrip()):
            if current_header and current_header.lstrip().startswith(station):
                return current_header, current_lines
            current_header = line
            current_lines = []
        elif current_header is not None:
            current_lines.append(line)
    if current_header and current_header.lstrip().startswith(station):
        return current_header, current_lines
    raise RuntimeError(f"Station {station} was not found in fetched LAMP bulletin.")


def parse_tmp_points(text: str, station: str) -> list[ForecastPoint]:
    header, lines = extract_station_block(text, station)
    issue_time = issue_time_from_header(header)
    if issue_time is None:
        raise RuntimeError("Could not parse LAMP issue time from station header.")

    row_map: dict[str, str] = {}
    for line in lines:
        label = line[:4].strip()
        if label:
            row_map[label] = line

    if "TMP" not in row_map:
        raise RuntimeError(f"Station {station} block has no TMP row.")
    if "HR" in row_map:
        _, hr_cells = parse_fixed_width_cells(row_map["HR"])
        forecast_hours = [int(token) for token in hr_cells if token]
    elif "UTC" in row_map:
        _, utc_cells = parse_fixed_width_cells(row_map["UTC"])
        forecast_hours = list(range(1, len([token for token in utc_cells if token]) + 1))
    else:
        raise RuntimeError(f"Station {station} block has no HR or UTC row.")

    _, tmp_cells = parse_fixed_width_cells(row_map["TMP"], count=len(forecast_hours))
    first_valid = issue_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    points: list[ForecastPoint] = []
    for index, forecast_hour in enumerate(forecast_hours):
        token = tmp_cells[index] if index < len(tmp_cells) else None
        if not token or set(token) == {"9"}:
            continue
        valid_utc = first_valid + dt.timedelta(hours=forecast_hour - 1)
        points.append(
            ForecastPoint(
                forecast_hour=forecast_hour,
                valid_time_utc=valid_utc,
                valid_time_local=valid_utc.astimezone(NY_TZ),
                tmp_f=int(token),
            )
        )
    return points


def group_points_by_local_date(points: list[ForecastPoint]) -> dict[dt.date, list[ForecastPoint]]:
    grouped: dict[dt.date, list[ForecastPoint]] = {}
    for point in points:
        grouped.setdefault(point.valid_time_local.date(), []).append(point)
    return grouped


def format_ts(timestamp: dt.datetime | None, tz: ZoneInfo = NY_TZ) -> str:
    if timestamp is None:
        return "unknown"
    return f"{timestamp.astimezone(UTC):%Y-%m-%d %H:%M:%S %Z} / {timestamp.astimezone(tz):%Y-%m-%d %I:%M:%S %p %Z}"


def main() -> int:
    args = parse_args()
    station = args.station.upper()
    attempts = max(1, int(args.attempts))
    timeout = max(1.0, float(args.timeout_seconds))
    if args.latest_before_local and (args.date_utc or args.cycle):
        raise ValueError("--latest-before-local cannot be combined with --date-utc or --cycle.")
    if args.latest_before_local:
        date_token, cycle_token = issue_tokens_before_local_cutoff(args.latest_before_local)
    else:
        date_token = normalize_date_token(args.date_utc) if args.date_utc else latest_date_token(timeout=timeout, attempts=attempts)
        cycle_token = normalize_cycle_token(args.cycle) if args.cycle else latest_cycle_token(
            date_token, timeout=timeout, attempts=attempts
        )

    base = f"{NOMADS_BASE}/lmp.{date_token}"
    standard = fetch_lamp_file(f"{base}/lmp.t{cycle_token}z.lavtxt.ascii", timeout=timeout, attempts=attempts)
    files = [standard]
    try:
        files.append(fetch_lamp_file(f"{base}/lmp.t{cycle_token}z.lavtxt_ext.ascii", timeout=timeout, attempts=attempts))
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise

    points_by_key: dict[tuple[int, dt.datetime], ForecastPoint] = {}
    for lamp_file in files:
        for point in parse_tmp_points(lamp_file.text, station):
            points_by_key[(point.forecast_hour, point.valid_time_utc)] = point
    points = sorted(points_by_key.values(), key=lambda point: point.forecast_hour)
    if not points:
        raise RuntimeError(f"No TMP forecast points found for {station}.")

    issue_time_utc = dt.datetime.strptime(date_token + cycle_token, "%Y%m%d%H%M").replace(tzinfo=UTC)
    target_date = (
        dt.date.fromisoformat(args.target_date_local)
        if args.target_date_local
        else issue_time_utc.astimezone(NY_TZ).date()
    )
    grouped = group_points_by_local_date(points)

    print(f"station={station}")
    print(f"issue={issue_time_utc:%Y-%m-%d %H:%M UTC}")
    print(f"standard_published={format_ts(standard.last_modified)}")
    if len(files) > 1:
        print(f"extended_published={format_ts(files[1].last_modified)}")
    print(f"source={standard.url}")

    target_points = grouped.get(target_date, [])
    if target_points:
        max_tmp = max(point.tmp_f for point in target_points)
        peak_points = [point for point in target_points if point.tmp_f == max_tmp]
        peak_times = ", ".join(point.valid_time_local.strftime("%Y-%m-%d %H:%M %Z") for point in peak_points)
        print(f"target_date_local={target_date.isoformat()}")
        print(f"predicted_tmax_f={max_tmp}")
        print(f"peak_time_local={peak_times}")
    else:
        print(f"target_date_local={target_date.isoformat()}")
        print("predicted_tmax_f=unavailable")

    print("all_local_date_tmax:")
    for local_date in sorted(grouped):
        day_points = grouped[local_date]
        max_tmp = max(point.tmp_f for point in day_points)
        peak = next(point for point in day_points if point.tmp_f == max_tmp)
        print(f"  {local_date.isoformat()}: {max_tmp}F at {peak.valid_time_local:%H:%M %Z}")

    if args.show_hourly:
        print("hourly_tmp:")
        for point in points:
            print(
                f"  f{point.forecast_hour:03d} "
                f"{point.valid_time_local:%Y-%m-%d %H:%M %Z} "
                f"{point.tmp_f}F"
            )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
