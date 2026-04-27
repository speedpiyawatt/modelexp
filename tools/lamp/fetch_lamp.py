#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import io
import pathlib
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lamp.parse_lamp_ascii import issue_time_from_header_line


NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/lmp/prod"
ARCHIVE_BASE = "https://lamp.mdl.nws.noaa.gov/lamp/Data/archives"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "runtime" / "raw"
FAST_ARCHIVE_MAX_DAYS = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download NOAA LAMP ASCII bulletins for KLGA workflows.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live = subparsers.add_parser("live", help="Download live hourly LAMP bulletins from NOMADS.")
    live.add_argument("--date-utc", required=True, help="UTC issue date in YYYY-MM-DD or YYYYMMDD.")
    live.add_argument("--cycle", required=True, help="Issue cycle token in HHMM or HH.")
    live.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    live.add_argument("--overwrite", action="store_true")

    archive = subparsers.add_parser("archive", help="Download and filter archived yearly LAMP tar bundles.")
    archive.add_argument("--start-utc-date", required=True, help="First UTC issue date in YYYY-MM-DD or YYYYMMDD.")
    archive.add_argument("--end-utc-date", required=True, help="Last UTC issue date in YYYY-MM-DD or YYYYMMDD.")
    archive.add_argument("--cycle", action="append", default=None, help="Optional HHMM cycle filter. Repeatable.")
    archive.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT_DIR)
    archive.add_argument("--cache-dir", type=pathlib.Path, default=None, help="Optional local cache directory for yearly tar files.")
    archive.add_argument("--keep-archive-tars", action="store_true")
    archive.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def normalize_date(value: str) -> dt.date:
    cleaned = value.replace("-", "")
    return dt.datetime.strptime(cleaned, "%Y%m%d").date()


def normalize_cycle_token(value: str) -> str:
    digits = "".join(char for char in value if char.isdigit())
    if len(digits) == 2:
        digits = f"{digits}00"
    if len(digits) != 4:
        raise ValueError(f"Cycle token must resolve to HHMM, got {value!r}")
    return digits


def live_file_names(cycle_token: str) -> list[tuple[str, str]]:
    if cycle_token[-2:] not in {"00", "15", "30", "45"}:
        raise ValueError("Live LAMP cycle token must use a published 15-minute issue cadence: HH00, HH15, HH30, or HH45.")
    return [
        ("standard", f"lmp.t{cycle_token}z.lavtxt.ascii"),
        ("extended", f"lmp.t{cycle_token}z.lavtxt_ext.ascii"),
    ]


def download_with_progress(url: str, destination: pathlib.Path, *, overwrite: bool) -> pathlib.Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[skip] {destination}")
        return destination
    request = urllib.request.Request(url, headers={"User-Agent": "codex-lamp-fetcher/1.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        total = response.headers.get("Content-Length")
        expected = int(total) if total and total.isdigit() else None
        written = 0
        print(f"[start] {url}")
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if expected:
                pct = (written / expected) * 100.0
                print(f"[progress] {destination.name} downloaded_mb={written / (1024 * 1024):.2f} pct={pct:.1f}")
            else:
                print(f"[progress] {destination.name} downloaded_mb={written / (1024 * 1024):.2f}")
    print(f"[ok] {destination}")
    return destination


def download_live(args: argparse.Namespace) -> list[pathlib.Path]:
    issue_date = normalize_date(args.date_utc)
    cycle_token = normalize_cycle_token(args.cycle)
    daily_dir = issue_date.strftime("%Y%m%d")
    outputs: list[pathlib.Path] = []
    for bulletin_type, filename in live_file_names(cycle_token):
        url = f"{NOMADS_BASE}/lmp.{daily_dir}/{filename}"
        destination = args.output_dir / "source=nomads" / f"date_utc={issue_date.isoformat()}" / f"cycle={cycle_token}" / filename
        try:
            outputs.append(download_with_progress(url, destination, overwrite=args.overwrite))
        except urllib.error.HTTPError as exc:
            if bulletin_type == "extended" and exc.code == 404:
                print(f"[skip] extended bulletin unavailable for cycle={cycle_token}")
                continue
            raise
    return outputs


def iter_utc_dates(start_date: dt.date, end_date: dt.date) -> list[dt.date]:
    days = (end_date - start_date).days
    return [start_date + dt.timedelta(days=offset) for offset in range(days + 1)]


def archive_url(year: int) -> str:
    return f"{ARCHIVE_BASE}/lmp_lavtxt.{year}.tar"


def archive_monthly_cycle_url(year_month: str, cycle_token: str) -> str:
    return f"{ARCHIVE_BASE}/lmp_lavtxt.{year_month}.{cycle_token}z.gz"


def cache_path_for_year(year: int, cache_dir: pathlib.Path | None) -> pathlib.Path:
    root = cache_dir or (DEFAULT_OUTPUT_DIR / "archive_cache")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"lmp_lavtxt.{year}.tar"


def cache_path_for_monthly_cycle(year_month: str, cycle_token: str, cache_dir: pathlib.Path | None) -> pathlib.Path:
    root = cache_dir or (DEFAULT_OUTPUT_DIR / "archive_cycle_cache")
    root.mkdir(parents=True, exist_ok=True)
    return root / f"lmp_lavtxt.{year_month}.{cycle_token}z.gz"


def header_bytes_issue_time(header_bytes: bytes) -> dt.datetime | None:
    first_line = header_bytes.decode("utf-8", errors="replace").splitlines()[0] if header_bytes else ""
    return issue_time_from_header_line(first_line)


def member_cycle_token(issue_time_utc: dt.datetime) -> str:
    return issue_time_utc.strftime("%H%M")


def iter_year_months(start_date: dt.date, end_date: dt.date) -> list[str]:
    values: list[str] = []
    current = dt.date(start_date.year, start_date.month, 1)
    end_month = dt.date(end_date.year, end_date.month, 1)
    while current <= end_month:
        values.append(current.strftime("%Y%m"))
        if current.month == 12:
            current = dt.date(current.year + 1, 1, 1)
        else:
            current = dt.date(current.year, current.month + 1, 1)
    return values


def should_use_monthly_cycle_fast_path(
    *,
    start_date: dt.date,
    end_date: dt.date,
    cycle_filter: set[str] | None,
) -> bool:
    if not cycle_filter:
        return False
    requested_days = (end_date - start_date).days + 1
    return requested_days <= FAST_ARCHIVE_MAX_DAYS


def archive_issue_destination(
    *,
    output_dir: pathlib.Path,
    issue_time_utc: dt.datetime,
    bulletin_type: str,
    source_name: str,
) -> pathlib.Path:
    cycle_token = member_cycle_token(issue_time_utc)
    issue_date = issue_time_utc.date().isoformat()
    filename = f"{bulletin_type}_{source_name}"
    return output_dir / "source=archive" / f"date_utc={issue_date}" / f"cycle={cycle_token}" / filename


def canonical_archive_member_name(*, issue_time_utc: dt.datetime, bulletin_type: str) -> str:
    suffix = "_ext" if bulletin_type == "extended" else ""
    return f"lmp_lavtxt.{issue_time_utc.strftime('%Y%m%d')}.{member_cycle_token(issue_time_utc)}z{suffix}.ascii"


def write_archive_issue_file(
    *,
    output_dir: pathlib.Path,
    issue_time_utc: dt.datetime,
    bulletin_type: str,
    source_name: str,
    lines: list[str],
    overwrite: bool,
) -> pathlib.Path:
    destination = archive_issue_destination(
        output_dir=output_dir,
        issue_time_utc=issue_time_utc,
        bulletin_type=bulletin_type,
        source_name=source_name,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        print(f"[skip] {destination}")
        return destination
    payload = "\n".join(lines).rstrip("\n") + "\n"
    destination.write_text(payload)
    print(f"[ok] {destination}")
    return destination


def _extract_matching_issues_from_monthly_cycle_gzip(
    gzip_path: pathlib.Path,
    *,
    start_date: dt.date,
    end_date: dt.date,
    cycle_filter: set[str],
    output_dir: pathlib.Path,
    overwrite: bool,
) -> list[pathlib.Path]:
    selected_outputs: list[pathlib.Path] = []
    with gzip.open(gzip_path, "rt", encoding="utf-8", errors="replace") as handle:
        active_issue_time: dt.datetime | None = None
        active_lines: list[str] = []
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            issue_time = issue_time_from_header_line(line)
            if issue_time is not None and active_issue_time is not None and issue_time != active_issue_time:
                cycle_token = member_cycle_token(active_issue_time)
                issue_date = active_issue_time.date()
                if start_date <= issue_date <= end_date and cycle_token in cycle_filter:
                    selected_outputs.append(
                        write_archive_issue_file(
                            output_dir=output_dir,
                            issue_time_utc=active_issue_time,
                            bulletin_type="standard",
                            source_name=canonical_archive_member_name(issue_time_utc=active_issue_time, bulletin_type="standard"),
                            lines=active_lines,
                            overwrite=overwrite,
                        )
                    )
                active_lines = []
            if issue_time is not None:
                active_issue_time = issue_time
            if active_issue_time is not None:
                active_lines.append(line)
        if active_issue_time is not None and active_lines:
            cycle_token = member_cycle_token(active_issue_time)
            issue_date = active_issue_time.date()
            if start_date <= issue_date <= end_date and cycle_token in cycle_filter:
                selected_outputs.append(
                    write_archive_issue_file(
                        output_dir=output_dir,
                        issue_time_utc=active_issue_time,
                        bulletin_type="standard",
                        source_name=canonical_archive_member_name(issue_time_utc=active_issue_time, bulletin_type="standard"),
                        lines=active_lines,
                        overwrite=overwrite,
                    )
                )
    return selected_outputs


def _extract_matching_issues_from_monthly_cycle_lines(
    lines,
    *,
    start_date: dt.date,
    end_date: dt.date,
    cycle_filter: set[str] | None,
    output_dir: pathlib.Path,
    overwrite: bool,
) -> list[pathlib.Path]:
    selected_outputs: list[pathlib.Path] = []
    active_issue_time: dt.datetime | None = None
    active_lines: list[str] = []

    def flush_active() -> None:
        nonlocal active_lines, active_issue_time
        if active_issue_time is None or not active_lines:
            return
        cycle_token = member_cycle_token(active_issue_time)
        issue_date = active_issue_time.date()
        if start_date <= issue_date <= end_date and (cycle_filter is None or cycle_token in cycle_filter):
            selected_outputs.append(
                write_archive_issue_file(
                    output_dir=output_dir,
                    issue_time_utc=active_issue_time,
                    bulletin_type="standard",
                    source_name=canonical_archive_member_name(issue_time_utc=active_issue_time, bulletin_type="standard"),
                    lines=active_lines,
                    overwrite=overwrite,
                )
            )
        active_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        issue_time = issue_time_from_header_line(line)
        if issue_time is not None and active_issue_time is not None and issue_time != active_issue_time:
            flush_active()
        if issue_time is not None:
            active_issue_time = issue_time
        if active_issue_time is not None:
            active_lines.append(line)
    flush_active()
    return selected_outputs


def promote_staged_archive_outputs(
    staged_root: pathlib.Path,
    final_root: pathlib.Path,
    *,
    overwrite: bool,
) -> list[pathlib.Path]:
    promoted: list[pathlib.Path] = []
    for staged_path in sorted(candidate for candidate in staged_root.rglob("*.ascii") if candidate.is_file()):
        relative = staged_path.relative_to(staged_root)
        destination = final_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not overwrite:
                print(f"[skip] {destination}")
                promoted.append(destination)
                continue
            destination.unlink()
        staged_path.replace(destination)
        print(f"[ok] {destination}")
        promoted.append(destination)
    return promoted


def extract_archive_members_fast(args: argparse.Namespace) -> list[pathlib.Path]:
    start_date = normalize_date(args.start_utc_date)
    end_date = normalize_date(args.end_utc_date)
    cycle_filter = {normalize_cycle_token(value) for value in args.cycle} if args.cycle else None
    if cycle_filter is None:
        return []

    final_archive_root = args.output_dir
    final_archive_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lamp-fast-", dir=final_archive_root) as tmp_dir:
        staged_output_dir = pathlib.Path(tmp_dir)
        for year_month in iter_year_months(start_date, end_date):
            for cycle_token in sorted(cycle_filter):
                gzip_path = cache_path_for_monthly_cycle(year_month, cycle_token, args.cache_dir)
                url = archive_monthly_cycle_url(year_month, cycle_token)
                download_with_progress(url, gzip_path, overwrite=args.overwrite)
                _extract_matching_issues_from_monthly_cycle_gzip(
                    gzip_path,
                    start_date=start_date,
                    end_date=end_date,
                    cycle_filter=cycle_filter,
                    output_dir=staged_output_dir,
                    overwrite=True,
                )
        return promote_staged_archive_outputs(staged_output_dir, final_archive_root, overwrite=args.overwrite)


def _extract_members_from_tar(
    tar_path: pathlib.Path,
    *,
    start_date: dt.date,
    end_date: dt.date,
    cycle_filter: set[str] | None,
    output_dir: pathlib.Path,
    overwrite: bool,
) -> list[pathlib.Path]:
    selected_outputs: list[pathlib.Path] = []
    with tarfile.open(tar_path) as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_name = pathlib.Path(member.name).name
            member_name_lower = member_name.lower()
            fileobj = archive.extractfile(member)
            if fileobj is None:
                continue
            if member_name_lower.endswith(".gz"):
                name_parts = member_name_lower.split(".")
                if len(name_parts) >= 3:
                    year_month = name_parts[1]
                    cycle_token = name_parts[2].removesuffix("z")
                    if cycle_filter is not None and cycle_token not in cycle_filter:
                        continue
                    if len(year_month) == 6 and year_month.isdigit():
                        month_start = dt.date(int(year_month[:4]), int(year_month[4:]), 1)
                        if month_start.month == 12:
                            next_month = dt.date(month_start.year + 1, 1, 1)
                        else:
                            next_month = dt.date(month_start.year, month_start.month + 1, 1)
                        month_end = next_month - dt.timedelta(days=1)
                        if month_end < start_date or month_start > end_date:
                            continue
                with gzip.GzipFile(fileobj=fileobj) as gzip_file:
                    text_handle = io.TextIOWrapper(gzip_file, encoding="utf-8", errors="replace")
                    selected_outputs.extend(
                        _extract_matching_issues_from_monthly_cycle_lines(
                            text_handle,
                            start_date=start_date,
                            end_date=end_date,
                            cycle_filter=cycle_filter,
                            output_dir=output_dir,
                            overwrite=overwrite,
                        )
                    )
                continue
            if not member_name_lower.endswith(".ascii"):
                continue
            header = fileobj.read(256)
            issue_time = header_bytes_issue_time(header)
            if issue_time is None:
                continue
            issue_date = issue_time.date()
            if issue_date < start_date or issue_date > end_date:
                continue
            cycle_token = member_cycle_token(issue_time)
            if cycle_filter is not None and cycle_token not in cycle_filter:
                continue
            bulletin_type = "extended" if "ext" in pathlib.Path(member.name).name.lower() else "standard"
            destination = archive_issue_destination(
                output_dir=output_dir,
                issue_time_utc=issue_time,
                bulletin_type=bulletin_type,
                source_name=canonical_archive_member_name(issue_time_utc=issue_time, bulletin_type=bulletin_type),
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not overwrite:
                print(f"[skip] {destination}")
                selected_outputs.append(destination)
                continue
            with destination.open("wb") as handle:
                handle.write(header)
                shutil.copyfileobj(fileobj, handle)
            print(f"[ok] {destination}")
            selected_outputs.append(destination)
    return selected_outputs


def _repair_archive(
    *,
    year: int,
    tar_path: pathlib.Path,
    overwrite: bool,
) -> None:
    attempts = 2
    for attempt in range(attempts):
        if not tar_path.exists():
            download_with_progress(archive_url(year), tar_path, overwrite=True)
        try:
            with tarfile.open(tar_path):
                return
        except tarfile.ReadError:
            if attempt == attempts - 1:
                raise
            print(f"[warn] corrupt archive detected, re-downloading {tar_path}")
            try:
                tar_path.unlink()
            except FileNotFoundError:
                pass
            download_with_progress(archive_url(year), tar_path, overwrite=True)


def extract_archive_members(args: argparse.Namespace) -> list[pathlib.Path]:
    start_date = normalize_date(args.start_utc_date)
    end_date = normalize_date(args.end_utc_date)
    cycle_filter = {normalize_cycle_token(value) for value in args.cycle} if args.cycle else None
    if should_use_monthly_cycle_fast_path(start_date=start_date, end_date=end_date, cycle_filter=cycle_filter):
        try:
            return extract_archive_members_fast(args)
        except Exception as exc:
            print(f"[warn] monthly-cycle archive fast path failed, falling back to yearly tar bundles: {exc}")

    selected_outputs: list[pathlib.Path] = []

    for year in sorted({day.year for day in iter_utc_dates(start_date, end_date)}):
        tar_path = cache_path_for_year(year, args.cache_dir)
        attempts = 2
        for attempt in range(attempts):
            _repair_archive(year=year, tar_path=tar_path, overwrite=args.overwrite)
            try:
                selected_outputs.extend(
                    _extract_members_from_tar(
                        tar_path,
                        start_date=start_date,
                        end_date=end_date,
                        cycle_filter=cycle_filter,
                        output_dir=args.output_dir,
                        overwrite=args.overwrite,
                    )
                )
                break
            except tarfile.ReadError:
                if attempt == attempts - 1:
                    raise
                print(f"[warn] archive read failed during extraction, re-downloading {tar_path}")
                try:
                    tar_path.unlink()
                except FileNotFoundError:
                    pass
                download_with_progress(archive_url(year), tar_path, overwrite=True)
        if not args.keep_archive_tars and args.cache_dir is None:
            try:
                tar_path.unlink()
            except FileNotFoundError:
                pass
    return selected_outputs


def main() -> int:
    args = parse_args()
    if args.command == "live":
        outputs = download_live(args)
    else:
        outputs = extract_archive_members(args)
    print(f"[done] files={len(outputs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
