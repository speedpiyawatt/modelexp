#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import re
import statistics
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


ARCHIVE_BASE = "https://lamp.mdl.nws.noaa.gov/lamp/Data/archives"
UTC = dt.timezone.utc
NY_TZ = ZoneInfo("America/New_York")
DEFAULT_HISTORY_DIR = Path("wunderground/output/history")
DEFAULT_CACHE_DIR = Path("tools/lamp/data/runtime/archive_cycle_cache")
DEFAULT_OUTPUT_DIR = Path("tools/lamp/data/runtime/analysis")

HEADER_RE = re.compile(
    r"^\s*(?P<station>[A-Z0-9]{4})\s+GFS LAMP GUIDANCE\s+"
    r"(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>\d{4})\s+(?P<hhmm>\d{4})\s+UTC\s*$"
)


@dataclass(frozen=True)
class WuLabel:
    target_date_local: dt.date
    final_tmax_f: float
    final_tmax_time_local: dt.datetime


@dataclass(frozen=True)
class LampForecast:
    target_date_local: dt.date
    issue_time_utc: dt.datetime
    issue_time_local: dt.datetime
    issue_local_hour: int
    issue_minutes_local: int
    predicted_tmax_f: float
    forecast_valid_count: int
    first_valid_time_local: dt.datetime
    last_valid_time_local: dt.datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze LAMP issue-hour Tmax accuracy against Wunderground KLGA Tmax.")
    parser.add_argument("--start-date", default="2026-02-01", help="First target local date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default="2026-03-31", help="Last target local date, YYYY-MM-DD.")
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-lead-hours", type=float, default=6.0, help="Minimum issue-to-observed-Tmax lead for trusted summaries.")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def iter_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    return [start_date + dt.timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def iter_year_months(start_date: dt.date, end_date: dt.date) -> list[str]:
    months: list[str] = []
    current = dt.date(start_date.year, start_date.month, 1)
    final = dt.date(end_date.year, end_date.month, 1)
    while current <= final:
        months.append(current.strftime("%Y%m"))
        current = dt.date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
    return months


def cycle_tokens() -> list[str]:
    return [f"{hour:02d}30" for hour in range(24)]


def download_archive(year_month: str, cycle: str, cache_dir: Path, *, timeout: float, attempts: int) -> Path | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"lmp_lavtxt.{year_month}.{cycle}z.gz"
    if path.exists() and path.stat().st_size > 0 and gzip_is_readable(path):
        return path
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    url = f"{ARCHIVE_BASE}/lmp_lavtxt.{year_month}.{cycle}z.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "lamp-tmax-analysis/1.0"})
    last_error: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response, path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    handle.write(chunk)
            if gzip_is_readable(path):
                return path
            raise EOFError("Downloaded gzip did not pass validation.")
        except Exception as exc:
            last_error = exc
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    print(f"[warn] failed archive {url}: {last_error}", file=sys.stderr)
    return None


def gzip_is_readable(path: Path) -> bool:
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def load_wu_labels(history_dir: Path, target_dates: set[dt.date]) -> dict[dt.date, WuLabel]:
    labels: dict[dt.date, WuLabel] = {}
    for target_date in sorted(target_dates):
        path = history_dir / f"KLGA_9_US_{target_date.isoformat()}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        rows = []
        for obs in payload.get("observations", []):
            temp = obs.get("temp")
            valid_gmt = obs.get("valid_time_gmt")
            if temp is None or valid_gmt is None:
                continue
            valid_local = dt.datetime.fromtimestamp(int(valid_gmt), UTC).astimezone(NY_TZ)
            if valid_local.date() == target_date:
                rows.append((float(temp), valid_local))
        if not rows:
            continue
        max_temp = max(temp for temp, _ in rows)
        first_max_time = min(valid_time for temp, valid_time in rows if temp == max_temp)
        labels[target_date] = WuLabel(
            target_date_local=target_date,
            final_tmax_f=max_temp,
            final_tmax_time_local=first_max_time,
        )
    return labels


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
        width = count * 3
        body = body.ljust(width)
    return label, [body[index : index + 3].strip() or None for index in range(0, width, 3)]


def issue_time_from_header(line: str) -> dt.datetime | None:
    match = HEADER_RE.match(line.rstrip())
    if not match:
        return None
    info = match.groupdict()
    hhmm = info["hhmm"]
    return dt.datetime(
        int(info["year"]),
        int(info["month"]),
        int(info["day"]),
        int(hhmm[:2]),
        int(hhmm[2:]),
        tzinfo=UTC,
    )


def parse_klga_block(header: str, body: list[str], *, station: str, target_dates: set[dt.date]) -> list[LampForecast]:
    if not header.strip().startswith(station):
        return []
    issue_time_utc = issue_time_from_header(header)
    if issue_time_utc is None:
        return []
    issue_time_local = issue_time_utc.astimezone(NY_TZ)
    row_map: dict[str, str] = {}
    for line in body:
        label = line[:4].strip()
        if label:
            row_map[label] = line
    if "TMP" not in row_map:
        return []
    if "HR" in row_map:
        _, hr_cells = parse_fixed_width_cells(row_map["HR"])
        forecast_hours = [int(token) for token in hr_cells if token]
    elif "UTC" in row_map:
        _, utc_cells = parse_fixed_width_cells(row_map["UTC"])
        forecast_hours = list(range(1, len([token for token in utc_cells if token]) + 1))
    else:
        return []

    _, tmp_cells = parse_fixed_width_cells(row_map["TMP"], count=len(forecast_hours))
    first_valid = issue_time_utc.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    by_date: dict[dt.date, list[tuple[int, dt.datetime]]] = {}
    for index, forecast_hour in enumerate(forecast_hours):
        token = tmp_cells[index] if index < len(tmp_cells) else None
        if not token or set(token) == {"9"}:
            continue
        valid_utc = first_valid + dt.timedelta(hours=forecast_hour - 1)
        valid_local_date = valid_utc.astimezone(NY_TZ).date()
        if valid_local_date not in target_dates:
            continue
        by_date.setdefault(valid_local_date, []).append((int(token), valid_utc.astimezone(NY_TZ)))

    forecasts: list[LampForecast] = []
    for target_date, temp_times in by_date.items():
        if not temp_times:
            continue
        temps = [temp for temp, _ in temp_times]
        valid_times = [valid_time for _, valid_time in temp_times]
        forecasts.append(
            LampForecast(
                target_date_local=target_date,
                issue_time_utc=issue_time_utc,
                issue_time_local=issue_time_local,
                issue_local_hour=issue_time_local.hour,
                issue_minutes_local=issue_time_local.hour * 60 + issue_time_local.minute,
                predicted_tmax_f=float(max(temps)),
                forecast_valid_count=len(temp_times),
                first_valid_time_local=min(valid_times),
                last_valid_time_local=max(valid_times),
            )
        )
    return forecasts


def parse_archive(path: Path, *, station: str, target_dates: set[dt.date]) -> list[LampForecast]:
    forecasts: list[LampForecast] = []
    active_header: str | None = None
    active_body: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if HEADER_RE.match(line.rstrip()):
                if active_header is not None:
                    forecasts.extend(parse_klga_block(active_header, active_body, station=station, target_dates=target_dates))
                active_header = line
                active_body = []
            elif active_header is not None:
                active_body.append(line)
    if active_header is not None:
        forecasts.extend(parse_klga_block(active_header, active_body, station=station, target_dates=target_dates))
    return forecasts


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(rows: list[dict[str, object]], group_key: str) -> list[dict[str, object]]:
    grouped: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(row[group_key], []).append(row)
    summaries = []
    for key, group in sorted(grouped.items()):
        errors = [float(row["error_f"]) for row in group]
        abs_errors = [abs(value) for value in errors]
        summaries.append(
            {
                group_key: key,
                "n": len(group),
                "mae_f": mean(abs_errors),
                "rmse_f": math.sqrt(mean([value * value for value in errors])),
                "bias_f": mean(errors),
                "median_abs_error_f": statistics.median(abs_errors),
                "p75_abs_error_f": percentile(abs_errors, 0.75),
                "within_1f_rate": mean([1.0 if value <= 1.0 else 0.0 for value in abs_errors]),
                "within_2f_rate": mean([1.0 if value <= 2.0 else 0.0 for value in abs_errors]),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0].keys())
    lines = [",".join(columns)]
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                text = "" if value is None else str(value)
                cells.append('"' + text.replace('"', '""') + '"' if "," in text else text)
        lines.append(",".join(cells))
    path.write_text("\n".join(lines) + "\n")


def json_safe(value: object) -> object:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    start_date = dt.date.fromisoformat(args.start_date)
    end_date = dt.date.fromisoformat(args.end_date)
    target_dates = set(iter_dates(start_date, end_date))
    labels = load_wu_labels(args.history_dir, target_dates)
    if not labels:
        raise RuntimeError("No Wunderground labels loaded.")

    forecasts: list[LampForecast] = []
    months = iter_year_months(start_date, end_date)
    cycles = cycle_tokens()
    total = len(months) * len(cycles)
    done = 0
    for year_month in months:
        for cycle in cycles:
            done += 1
            print(f"[{done}/{total}] archive {year_month} {cycle}", file=sys.stderr)
            archive = download_archive(
                year_month,
                cycle,
                args.cache_dir,
                timeout=float(args.timeout_seconds),
                attempts=int(args.attempts),
            )
            if archive is None:
                continue
            forecasts.extend(parse_archive(archive, station=args.station.upper(), target_dates=target_dates))

    rows: list[dict[str, object]] = []
    for forecast in forecasts:
        label = labels.get(forecast.target_date_local)
        if label is None:
            continue
        if forecast.issue_time_local >= label.final_tmax_time_local:
            continue
        if forecast.last_valid_time_local < label.final_tmax_time_local:
            continue
        lead_hours = (label.final_tmax_time_local - forecast.issue_time_local).total_seconds() / 3600.0
        if lead_hours < 0:
            continue
        rows.append(
            {
                "target_date_local": forecast.target_date_local.isoformat(),
                "actual_tmax_f": label.final_tmax_f,
                "actual_tmax_time_local": label.final_tmax_time_local.isoformat(),
                "issue_time_utc": forecast.issue_time_utc.isoformat(),
                "issue_time_local": forecast.issue_time_local.isoformat(),
                "issue_local_hour": forecast.issue_local_hour,
                "predicted_tmax_f": forecast.predicted_tmax_f,
                "error_f": forecast.predicted_tmax_f - label.final_tmax_f,
                "abs_error_f": abs(forecast.predicted_tmax_f - label.final_tmax_f),
                "lead_hours_to_actual_tmax": lead_hours,
                "forecast_valid_count": forecast.forecast_valid_count,
                "first_valid_time_local": forecast.first_valid_time_local.isoformat(),
                "last_valid_time_local": forecast.last_valid_time_local.isoformat(),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "lamp_tmax_forecast_rows.csv", rows)
    write_csv(args.output_dir / "wunderground_tmax_labels.csv", [label.__dict__ for label in labels.values()])

    eligible = [row for row in rows if float(row["lead_hours_to_actual_tmax"]) >= float(args.min_lead_hours)]
    write_csv(args.output_dir / "lamp_issue_hour_summary_all.csv", summarize(rows, "issue_local_hour"))
    trusted_summary = summarize(eligible, "issue_local_hour")
    write_csv(args.output_dir / "lamp_issue_hour_summary_minlead.csv", trusted_summary)

    pre_midnight = [
        row
        for row in rows
        if dt.datetime.fromisoformat(str(row["issue_time_local"])).date()
        < dt.date.fromisoformat(str(row["target_date_local"]))
    ]
    write_csv(args.output_dir / "lamp_premidnight_rows.csv", pre_midnight)
    pre_midnight_summary = summarize(pre_midnight, "issue_local_hour")
    write_csv(args.output_dir / "lamp_premidnight_summary.csv", pre_midnight_summary)
    summary_payload = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "station": args.station.upper(),
        "min_lead_hours": float(args.min_lead_hours),
        "wunderground_label_days": len(labels),
        "forecast_rows_before_observed_tmax": len(rows),
        "forecast_rows_minlead": len(eligible),
        "best_issue_hours_minlead_by_mae": sorted(
            trusted_summary,
            key=lambda item: (float(item["mae_f"]), -int(item["n"])),
        )[:12],
        "issue_hour_summary_minlead": trusted_summary,
        "premidnight_summary": pre_midnight_summary,
        "output_files": {
            "forecast_rows_csv": str(args.output_dir / "lamp_tmax_forecast_rows.csv"),
            "wunderground_labels_csv": str(args.output_dir / "wunderground_tmax_labels.csv"),
            "issue_hour_summary_all_csv": str(args.output_dir / "lamp_issue_hour_summary_all.csv"),
            "issue_hour_summary_minlead_csv": str(args.output_dir / "lamp_issue_hour_summary_minlead.csv"),
            "premidnight_rows_csv": str(args.output_dir / "lamp_premidnight_rows.csv"),
            "premidnight_summary_csv": str(args.output_dir / "lamp_premidnight_summary.csv"),
            "report_md": str(args.output_dir / "lamp_tmax_accuracy_report.md"),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }
    write_json(args.output_dir / "summary.json", summary_payload)

    report_lines = [
        f"# LAMP Tmax Accuracy {start_date.isoformat()} to {end_date.isoformat()}",
        "",
        f"Wunderground label days: {len(labels)}",
        f"Forecast rows before observed Tmax: {len(rows)}",
        f"Forecast rows with lead >= {args.min_lead_hours:g}h: {len(eligible)}",
        "",
        "## Best Issue Hours, Lead Filtered",
        "",
        "| issue_local_hour | n | MAE F | RMSE F | bias F | within 2F |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(trusted_summary, key=lambda item: (float(item["mae_f"]), -int(item["n"])))[:12]:
        report_lines.append(
            f"| {row['issue_local_hour']} | {row['n']} | {float(row['mae_f']):.2f} | "
            f"{float(row['rmse_f']):.2f} | {float(row['bias_f']):.2f} | {float(row['within_2f_rate']):.0%} |"
        )
    report_lines.extend(
        [
            "",
            "## Pre-Midnight Issue Hours",
            "",
            "| issue_local_hour | n | MAE F | RMSE F | bias F | within 2F |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pre_midnight_summary:
        report_lines.append(
            f"| {row['issue_local_hour']} | {row['n']} | {float(row['mae_f']):.2f} | "
            f"{float(row['rmse_f']):.2f} | {float(row['bias_f']):.2f} | {float(row['within_2f_rate']):.0%} |"
        )
    (args.output_dir / "lamp_tmax_accuracy_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"[done] output_dir={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
