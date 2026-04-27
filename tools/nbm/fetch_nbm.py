#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import concurrent.futures
import datetime as dt
import pathlib
import queue
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "fetch_nbm.py requires requests. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
GRIB2_BASE = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
COG_BASE = "https://noaa-nbm-pds.s3.amazonaws.com"
DEFAULT_OUTPUT = pathlib.Path("data/nbm")
COG_VERSION_PRIORITY = ["blendv4.3", "blendv4.2", "blendv4.1", "blendv4.0", "blendv3.2"]
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 120
LIST_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 0.5
STREAM_CHUNK_BYTES = 1024 * 1024
JOB_QUEUE_MAXSIZE = 128
DAY_PROGRESS_EVERY = 25
DOWNLOAD_PROGRESS_EVERY = 100
SINGLE_FILE_PROGRESS_EVERY_BYTES = 10 * 1024 * 1024
COG_PRESETS = {
    "klga_surface": [
        "temp",
        "dewpoint",
        "windspd",
        "winddir",
        "windgust",
        "sky",
        "ceil",
        "vis",
        "rh",
        "dswrf",
        "qpf01",
        "maxt",
        "mint",
    ],
    "klga_extended": [
        "temp",
        "tempstddev",
        "dewpoint",
        "tdstddev",
        "windspd",
        "windspdstddev",
        "winddir",
        "windgust",
        "sky",
        "skystddev",
        "ceil",
        "vis",
        "rh",
        "dswrf",
        "qpf01",
        "qpf06",
        "pop01",
        "pop06",
        "maxt",
        "maxtempstddev",
        "mint",
        "mintempstddev",
        "mixhgt",
        "wbgt",
        "hindex",
    ],
}


@dataclass
class DayResolution:
    day: dt.date
    version: str | None = None
    cycle: str | None = None
    error: str | None = None


@dataclass
class DayListing:
    day: dt.date
    version: str | None = None
    cycle: str | None = None
    variable_keys: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    remaining_variables: int = 0
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.error is not None or self.remaining_variables == 0


@dataclass
class DownloadProgress:
    start_time: float = field(default_factory=time.perf_counter)
    completed: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def increment(self) -> int:
        with self.lock:
            self.completed += 1
            return self.completed

    def elapsed(self) -> float:
        return max(0.001, time.perf_counter() - self.start_time)


class S3HttpClient:
    def __init__(
        self,
        *,
        pool_maxsize: int = 32,
        connect_timeout: int = CONNECT_TIMEOUT,
        read_timeout: int = READ_TIMEOUT,
        list_timeout: int = LIST_TIMEOUT,
        max_retries: int = MAX_RETRIES,
        backoff_factor: float = BACKOFF_FACTOR,
    ) -> None:
        self.timeout = (connect_timeout, read_timeout)
        self.list_timeout = (connect_timeout, list_timeout)
        self.session = self._build_session(
            pool_maxsize=pool_maxsize,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
        )
        self._content_length_cache: dict[str, int] = {}
        self._xml_cache: dict[str, bytes] = {}
        self._prefix_cache: dict[tuple[str, str], list[str]] = {}
        self._keys_cache: dict[tuple[str, str, int], list[str]] = {}
        self._cache_lock = threading.Lock()

    @staticmethod
    def _build_session(
        *,
        pool_maxsize: int,
        max_retries: int,
        backoff_factor: float,
    ) -> requests.Session:
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=pool_maxsize,
            pool_maxsize=pool_maxsize,
        )
        session = requests.Session()
        session.headers.update({"User-Agent": "nbm-fetcher/2.0"})
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _get_xml(self, url: str) -> bytes:
        with self._cache_lock:
            cached = self._xml_cache.get(url)
        if cached is not None:
            return cached

        response = self.session.get(url, timeout=self.list_timeout)
        response.raise_for_status()
        content = response.content
        with self._cache_lock:
            self._xml_cache[url] = content
        return content

    def list_prefixes(self, base_url: str, prefix: str) -> list[str]:
        cache_key = (base_url, prefix)
        with self._cache_lock:
            cached = self._prefix_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        params = {"prefix": prefix, "delimiter": "/"}
        response = self._get_xml(f"{base_url}/?{requests.compat.urlencode(params)}")
        root = ET.fromstring(response)
        prefixes = [
            elem.text or ""
            for elem in root.findall("s3:CommonPrefixes/s3:Prefix", S3_NS)
            if elem.text
        ]
        with self._cache_lock:
            self._prefix_cache[cache_key] = list(prefixes)
        return prefixes

    def list_keys(self, base_url: str, prefix: str, max_keys: int = 1000) -> list[str]:
        cache_key = (base_url, prefix, max_keys)
        with self._cache_lock:
            cached = self._keys_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        marker = None
        keys: list[str] = []
        while True:
            params = {"prefix": prefix, "max-keys": str(max_keys)}
            if marker:
                params["marker"] = marker
            xml_bytes = self._get_xml(f"{base_url}/?{requests.compat.urlencode(params)}")
            root = ET.fromstring(xml_bytes)
            page_keys = [
                elem.text or ""
                for elem in root.findall("s3:Contents/s3:Key", S3_NS)
                if elem.text
            ]
            keys.extend(page_keys)
            is_truncated = (
                root.findtext("s3:IsTruncated", default="false", namespaces=S3_NS) or ""
            ).lower()
            if is_truncated != "true" or not page_keys:
                break
            marker = page_keys[-1]

        with self._cache_lock:
            self._keys_cache[cache_key] = list(keys)
        return keys

    def fetch_text(self, url: str, *, timeout: tuple[int, int] | None = None) -> str:
        response = self.session.get(url, timeout=timeout or self.timeout)
        response.raise_for_status()
        return response.text

    def fetch_content_length(self, url: str) -> int:
        with self._cache_lock:
            cached = self._content_length_cache.get(url)
        if cached is not None:
            return cached
        response = self.session.head(url, timeout=self.timeout)
        response.raise_for_status()
        value = response.headers.get("Content-Length")
        if value is None:
            raise RuntimeError(f"Missing Content-Length for {url}")
        content_length = int(value)
        with self._cache_lock:
            self._content_length_cache[url] = content_length
        return content_length

    def fetch_range(self, url: str, *, start: int, end: int) -> bytes:
        response_cm = self.session.get(
            url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=self.timeout,
            stream=True,
        )
        response_context = response_cm if hasattr(response_cm, "__enter__") else contextlib.nullcontext(response_cm)
        with response_context as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206 for range {start}-{end}, got {response.status_code} from {url}"
                )
            chunks: list[bytes] = []
            if hasattr(response, "iter_content"):
                for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                    if not chunk:
                        continue
                    chunks.append(chunk)
            else:
                payload = getattr(response, "content", b"")
                if payload:
                    chunks.append(payload)
        payload = b"".join(chunks)
        expected_length = end - start + 1
        if len(payload) != expected_length:
            raise RuntimeError(
                f"Range {start}-{end} returned {len(payload)} bytes, expected {expected_length}"
            )
        return payload

    def download_byte_ranges(
        self,
        *,
        url: str,
        ranges: list[tuple[int, int]],
        destination: pathlib.Path,
        overwrite: bool,
        progress_label: str | None = None,
        progress_callback=None,
    ) -> pathlib.Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            return destination

        tmp_destination = destination.with_name(destination.name + ".part")
        if tmp_destination.exists():
            tmp_destination.unlink()

        total_bytes = sum(max(0, end - start + 1) for start, end in ranges)
        downloaded_bytes = 0
        next_progress_bytes = SINGLE_FILE_PROGRESS_EVERY_BYTES
        started_at = time.perf_counter()
        if progress_label:
            total_text = f"{total_bytes / (1024 * 1024):.1f}MB" if total_bytes else "unknown"
            print(f"download_start label={progress_label} file={destination.name} total={total_text}")
        if progress_callback:
            progress_callback(downloaded_bytes, total_bytes or None, 0.0)

        try:
            with tmp_destination.open("wb") as handle:
                for start, end in ranges:
                    with self.session.get(
                        url,
                        headers={"Range": f"bytes={start}-{end}"},
                        timeout=self.timeout,
                        stream=True,
                    ) as response:
                        response.raise_for_status()
                        if response.status_code != 206:
                            raise RuntimeError(
                                f"Expected HTTP 206 for range {start}-{end}, got {response.status_code} from {url}"
                            )
                        range_written = 0
                        for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            range_written += len(chunk)
                            downloaded_bytes += len(chunk)
                            elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
                            if progress_callback:
                                progress_callback(downloaded_bytes, total_bytes or None, elapsed_seconds)
                            if progress_label and (
                                downloaded_bytes >= next_progress_bytes or (total_bytes and downloaded_bytes >= total_bytes)
                            ):
                                mb_written = downloaded_bytes / (1024 * 1024)
                                if total_bytes:
                                    pct = (downloaded_bytes / total_bytes) * 100.0
                                    print(
                                        f"download_progress label={progress_label} file={destination.name} "
                                        f"downloaded_mb={mb_written:.1f} pct={pct:.1f}"
                                    )
                                else:
                                    print(
                                        f"download_progress label={progress_label} file={destination.name} "
                                        f"downloaded_mb={mb_written:.1f}"
                                    )
                                next_progress_bytes += SINGLE_FILE_PROGRESS_EVERY_BYTES
                    expected_length = end - start + 1
                    if range_written != expected_length:
                        raise RuntimeError(
                            f"Range {start}-{end} returned {range_written} bytes, expected {expected_length}"
                        )
            tmp_destination.replace(destination)
        except Exception:
            tmp_destination.unlink(missing_ok=True)
            raise

        if progress_label:
            final_mb = destination.stat().st_size / (1024 * 1024)
            print(f"download_complete label={progress_label} file={destination.name} size_mb={final_mb:.1f}")
        return destination

    def download_key(
        self,
        *,
        base_url: str,
        key: str,
        output_dir: pathlib.Path,
        overwrite: bool,
        preserve_tree: bool,
        progress_label: str | None = None,
    ) -> pathlib.Path:
        destination = output_dir / key if preserve_tree else output_dir / pathlib.Path(key).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            return destination

        tmp_destination = destination.with_name(destination.name + ".part")
        if tmp_destination.exists():
            tmp_destination.unlink()

        url = f"{base_url}/{key}"
        with self.session.get(url, stream=True, timeout=self.timeout) as response:
            response.raise_for_status()
            total_bytes = int(response.headers.get("Content-Length") or 0)
            written_bytes = 0
            next_progress_bytes = SINGLE_FILE_PROGRESS_EVERY_BYTES
            if progress_label:
                total_text = f"{total_bytes / (1024 * 1024):.1f}MB" if total_bytes else "unknown"
                print(f"download_start label={progress_label} file={destination.name} total={total_text}")
            with tmp_destination.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    written_bytes += len(chunk)
                    if progress_label and (written_bytes >= next_progress_bytes or (total_bytes and written_bytes >= total_bytes)):
                        mb_written = written_bytes / (1024 * 1024)
                        if total_bytes:
                            pct = (written_bytes / total_bytes) * 100.0
                            print(f"download_progress label={progress_label} file={destination.name} downloaded_mb={mb_written:.1f} pct={pct:.1f}")
                        else:
                            print(f"download_progress label={progress_label} file={destination.name} downloaded_mb={mb_written:.1f}")
                        next_progress_bytes += SINGLE_FILE_PROGRESS_EVERY_BYTES

        tmp_destination.replace(destination)
        if progress_label:
            final_mb = destination.stat().st_size / (1024 * 1024)
            print(f"download_complete label={progress_label} file={destination.name} size_mb={final_mb:.1f}")
        return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch public NOAA NBM GRIB2 or COG files.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    grib2 = subparsers.add_parser("grib2", help="Fetch one NBM GRIB2 core file")
    grib2.add_argument("--date", help="Initialization date in YYYY-MM-DD or YYYYMMDD")
    grib2.add_argument("--cycle", help="Initialization hour, e.g. 11")
    grib2.add_argument("--forecast-hour", type=int, required=True, help="Forecast hour")
    grib2.add_argument(
        "--region",
        default="co",
        choices=["ak", "co", "gu", "hi", "pr"],
        help="NBM regional grid code",
    )
    grib2.add_argument(
        "--with-idx",
        action="store_true",
        help="Also download the .idx sidecar file",
    )
    add_common_args(grib2)

    cog = subparsers.add_parser("cog", help="Fetch one NBM COG GeoTIFF file")
    cog.add_argument("--date", help="Initialization date in YYYY-MM-DD or YYYYMMDD")
    cog.add_argument("--cycle", help="Initialization hour, e.g. 11")
    cog.add_argument(
        "--domain",
        default="conus",
        choices=["alaska", "conus", "global", "guam", "hawaii", "oceanic", "puertoRico"],
        help="COG domain",
    )
    cog.add_argument(
        "--variable",
        required=True,
        help="COG variable folder, e.g. temp, maxt, dewpoint, windspd, sky",
    )
    cog.add_argument("--valid", required=True, help="Valid time in YYYY-MM-DDTHH:MM")
    cog.add_argument(
        "--version",
        help="COG dataset version. Omit to auto-detect by date.",
    )
    add_common_args(cog)

    cog_batch = subparsers.add_parser(
        "cog-batch",
        help="Backfill full COG variable folders across a date range",
    )
    cog_batch.add_argument("--start-date", required=True, help="First date in YYYY-MM-DD or YYYYMMDD")
    cog_batch.add_argument("--end-date", required=True, help="Last date in YYYY-MM-DD or YYYYMMDD")
    cog_batch.add_argument(
        "--cycle",
        help="Initialization hour, e.g. 00, 06, 12, 18. Omit for latest available on each day.",
    )
    cog_batch.add_argument(
        "--domain",
        default="conus",
        choices=["alaska", "conus", "global", "guam", "hawaii", "oceanic", "puertoRico"],
        help="COG domain",
    )
    cog_batch.add_argument(
        "--version",
        help="COG dataset version. Omit to auto-detect per day.",
    )
    cog_batch.add_argument(
        "--preset",
        choices=sorted(COG_PRESETS),
        default="klga_surface",
        help="Named multi-variable forecast block to fetch",
    )
    cog_batch.add_argument(
        "--variables",
        nargs="+",
        help="Explicit variable list. Overrides --preset.",
    )
    cog_batch.add_argument("--dry-run", action="store_true", help="Print planned downloads without fetching data")
    cog_batch.add_argument(
        "--max-files",
        type=int,
        help="Stop after planning or downloading this many files",
    )
    cog_batch.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent download workers for cog-batch",
    )
    cog_batch.add_argument(
        "--plan-workers",
        type=int,
        default=4,
        help="Concurrent planning workers for day-level resolution and listing",
    )
    add_common_args(cog_batch)

    return parser.parse_args()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT,
        help="Directory to place downloaded files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files")


def normalize_date(value: str | None) -> dt.date:
    if value is None:
        return dt.datetime.utcnow().date()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"Unsupported date format: {value}")


def iter_dates(start: dt.date, end: dt.date) -> list[dt.date]:
    if end < start:
        raise SystemExit("--end-date must be on or after --start-date")
    return [start + dt.timedelta(days=offset) for offset in range((end - start).days + 1)]


def normalize_cycle(value: str) -> str:
    token = value.strip()
    if token.endswith("00") and len(token) == 4:
        token = token[:2]
    if len(token) == 1:
        token = f"0{token}"
    if len(token) != 2 or not token.isdigit():
        raise SystemExit(f"Unsupported cycle format: {value}")
    return token


def list_grib2_cycles(day: dt.date, client: S3HttpClient) -> list[str]:
    prefix = f"blend.{day:%Y%m%d}/"
    return sorted(item.rstrip("/").split("/")[-1] for item in client.list_prefixes(GRIB2_BASE, prefix))


def list_cog_cycles(day: dt.date, version: str, domain: str, client: S3HttpClient) -> list[str]:
    prefix = f"{version}/{domain}/{day:%Y/%m/%d}/"
    return sorted(item.rstrip("/").split("/")[-1][:2] for item in client.list_prefixes(COG_BASE, prefix))


def find_latest_grib2_cycle(day: dt.date, client: S3HttpClient) -> tuple[dt.date, str]:
    for offset in range(0, 3):
        target = day - dt.timedelta(days=offset)
        cycles = list_grib2_cycles(target, client)
        if cycles:
            return target, cycles[-1]
    raise SystemExit("No GRIB2 cycles found in the last 3 days.")


def resolve_cog_version_and_cycle(
    *,
    day: dt.date,
    domain: str,
    version: str | None,
    cycle: str | None,
    client: S3HttpClient,
) -> tuple[str, str]:
    versions = [version] if version else COG_VERSION_PRIORITY
    normalized_cycle = normalize_cycle(cycle) if cycle else None
    for candidate in versions:
        cycles = list_cog_cycles(day, candidate, domain, client)
        if not cycles:
            continue
        if normalized_cycle:
            if normalized_cycle in cycles:
                return candidate, normalized_cycle
            continue
        return candidate, cycles[-1]

    if version and normalized_cycle:
        raise SystemExit(
            f"No COG data found for version={version}, date={day:%Y-%m-%d}, cycle={normalized_cycle}."
        )
    if version:
        raise SystemExit(f"No COG data found for version={version} on {day:%Y-%m-%d}.")
    if normalized_cycle:
        raise SystemExit(f"No COG data found on {day:%Y-%m-%d} for cycle {normalized_cycle}.")
    raise SystemExit(f"No COG data found on {day:%Y-%m-%d}.")


def resolve_grib2_key(day: dt.date, cycle: str, forecast_hour: int, region: str) -> str:
    return (
        f"blend.{day:%Y%m%d}/{cycle}/core/"
        f"blend.t{cycle}z.core.f{forecast_hour:03d}.{region}.grib2"
    )


def list_cog_variable_keys(
    *,
    day: dt.date,
    cycle: str,
    version: str,
    domain: str,
    variable: str,
    client: S3HttpClient,
) -> list[str]:
    prefix = f"{version}/{domain}/{day:%Y/%m/%d}/{cycle}00/{variable}/"
    return sorted(key for key in client.list_keys(COG_BASE, prefix) if key.endswith(".tif"))


def resolve_cog_key(
    *,
    day: dt.date,
    cycle: str,
    version: str,
    domain: str,
    variable: str,
    valid: str,
    client: S3HttpClient,
) -> str:
    valid_dt = dt.datetime.strptime(valid, "%Y-%m-%dT%H:%M")
    prefix = f"{version}/{domain}/{day:%Y/%m/%d}/{cycle}00/{variable}/"
    init_token = f"{day:%Y-%m-%d}T{cycle}:00"
    valid_token = valid_dt.strftime("%Y-%m-%dT%H:%M")
    pattern = re.compile(
        rf"^{re.escape(prefix)}{re.escape(version)}_{re.escape(domain)}_"
        rf"{re.escape(variable)}_{re.escape(init_token)}_{re.escape(valid_token)}\.tif$"
    )
    keys = list_cog_variable_keys(
        day=day,
        cycle=cycle,
        version=version,
        domain=domain,
        variable=variable,
        client=client,
    )
    matches = [key for key in keys if pattern.match(key)]
    if not matches:
        raise SystemExit(
            "No matching COG file found. Check the date, cycle, variable, and valid time."
        )
    if len(matches) > 1:
        raise SystemExit(f"Expected one COG file, found {len(matches)}.")
    return matches[0]


def resolve_batch_variables(args: argparse.Namespace) -> list[str]:
    if args.variables:
        seen: dict[str, None] = {}
        for variable in args.variables:
            seen[variable] = None
        return list(seen)
    return list(COG_PRESETS[args.preset])


def plan_day_resolution(
    day: dt.date,
    *,
    domain: str,
    version: str | None,
    cycle: str | None,
    client: S3HttpClient,
) -> DayResolution:
    try:
        version_name, cycle_name = resolve_cog_version_and_cycle(
            day=day,
            domain=domain,
            version=version,
            cycle=cycle,
            client=client,
        )
        return DayResolution(day=day, version=version_name, cycle=cycle_name)
    except SystemExit as exc:
        return DayResolution(day=day, error=str(exc))


def plan_variable_listing(
    *,
    day: dt.date,
    version: str,
    cycle: str,
    domain: str,
    variable: str,
    client: S3HttpClient,
) -> tuple[str, list[str], str | None]:
    keys = list_cog_variable_keys(
        day=day,
        cycle=cycle,
        version=version,
        domain=domain,
        variable=variable,
        client=client,
    )
    if not keys:
        return variable, [], (
            f"warn: no files for {day:%Y-%m-%d} version={version} cycle={cycle} variable={variable}"
        )
    return variable, keys, None


def download_worker(
    *,
    worker_id: int,
    client: S3HttpClient,
    download_queue: queue.Queue[str | None],
    output_dir: pathlib.Path,
    overwrite: bool,
    progress: DownloadProgress,
    errors: list[str],
    error_lock: threading.Lock,
) -> None:
    while True:
        item = download_queue.get()
        try:
            if item is None:
                return
            client.download_key(
                base_url=COG_BASE,
                key=item,
                output_dir=output_dir,
                overwrite=overwrite,
                preserve_tree=True,
            )
            completed = progress.increment()
            if completed % DOWNLOAD_PROGRESS_EVERY == 0:
                rate = completed / progress.elapsed()
                print(
                    f"download_progress completed={completed} rate_fps={rate:.2f}",
                    file=sys.stderr,
                )
        except Exception as exc:  # pragma: no cover - exercised by smoke path
            with error_lock:
                errors.append(f"download worker {worker_id} failed for {item}: {exc}")
        finally:
            download_queue.task_done()


def run_cog_batch(args: argparse.Namespace, client: S3HttpClient | None = None) -> int:
    start_date = normalize_date(args.start_date)
    end_date = normalize_date(args.end_date)
    variables = resolve_batch_variables(args)
    days = iter_dates(start_date, end_date)
    client = client or S3HttpClient(pool_maxsize=max(args.workers, args.plan_workers) + 8)
    day_states: list[DayListing | None] = [None] * len(days)
    day_future_map: dict[concurrent.futures.Future[DayResolution], int] = {}
    listing_future_map: dict[concurrent.futures.Future[tuple[str, list[str], str | None]], int] = {}
    next_emit_idx = 0
    planned_files = 0
    resolved_days = 0
    limit_reached = False
    planning_start = time.perf_counter()

    download_queue: queue.Queue[str | None] | None = None
    download_threads: list[threading.Thread] = []
    download_errors: list[str] = []
    error_lock = threading.Lock()
    progress = DownloadProgress()

    if not args.dry_run:
        download_queue = queue.Queue(maxsize=JOB_QUEUE_MAXSIZE)
        for worker_id in range(max(1, args.workers)):
            thread = threading.Thread(
                target=download_worker,
                kwargs={
                    "worker_id": worker_id,
                    "client": client,
                    "download_queue": download_queue,
                    "output_dir": args.output_dir,
                    "overwrite": args.overwrite,
                    "progress": progress,
                    "errors": download_errors,
                    "error_lock": error_lock,
                },
                daemon=True,
            )
            thread.start()
            download_threads.append(thread)

    planning_workers = max(1, args.plan_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=planning_workers) as executor:
        for idx, day in enumerate(days):
            future = executor.submit(
                plan_day_resolution,
                day,
                domain=args.domain,
                version=args.version,
                cycle=args.cycle,
                client=client,
            )
            day_future_map[future] = idx

        while day_future_map or listing_future_map:
            pending = list(day_future_map) + list(listing_future_map)
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                if future in day_future_map:
                    idx = day_future_map.pop(future)
                    resolution = future.result()
                    resolved_days += 1
                    if resolved_days % DAY_PROGRESS_EVERY == 0:
                        elapsed = max(0.001, time.perf_counter() - planning_start)
                        print(
                            f"planning_progress resolved_days={resolved_days}/{len(days)} rate_dps={resolved_days / elapsed:.2f}",
                            file=sys.stderr,
                        )
                    if resolution.error:
                        day_states[idx] = DayListing(day=resolution.day, error=resolution.error)
                        continue
                    day_states[idx] = DayListing(
                        day=resolution.day,
                        version=resolution.version,
                        cycle=resolution.cycle,
                        remaining_variables=len(variables),
                    )
                    for variable in variables:
                        listing_future = executor.submit(
                            plan_variable_listing,
                            day=resolution.day,
                            version=resolution.version or "",
                            cycle=resolution.cycle or "",
                            domain=args.domain,
                            variable=variable,
                            client=client,
                        )
                        listing_future_map[listing_future] = idx
                else:
                    idx = listing_future_map.pop(future)
                    variable, keys, warning = future.result()
                    state = day_states[idx]
                    if state is None:
                        continue
                    state.variable_keys[variable] = keys
                    if warning:
                        state.warnings.append(warning)
                    state.remaining_variables -= 1

            while next_emit_idx < len(day_states):
                state = day_states[next_emit_idx]
                if state is None or not state.ready:
                    break
                if state.error:
                    print(f"skip: {state.day:%Y-%m-%d} {state.error}", file=sys.stderr)
                    next_emit_idx += 1
                    continue

                print(
                    f"day={state.day:%Y-%m-%d} version={state.version} cycle={state.cycle} variables={','.join(variables)}"
                )
                for warning in state.warnings:
                    print(warning, file=sys.stderr)

                for variable in variables:
                    for key in state.variable_keys.get(variable, []):
                        if args.max_files is not None and planned_files >= args.max_files:
                            print(f"stopping at max_files={args.max_files}")
                            limit_reached = True
                            break
                        planned_files += 1
                        if args.dry_run:
                            print(f"plan: {key}")
                        else:
                            assert download_queue is not None
                            download_queue.put(key)
                    if limit_reached:
                        break
                next_emit_idx += 1
                if limit_reached:
                    break

            if limit_reached or download_errors:
                break

        if limit_reached:
            for future in list(day_future_map) + list(listing_future_map):
                future.cancel()

    if args.dry_run:
        print(f"planned_files={planned_files}")
        return 0

    assert download_queue is not None
    for _ in download_threads:
        download_queue.put(None)
    download_queue.join()
    for thread in download_threads:
        thread.join()

    if download_errors:
        raise SystemExit(download_errors[0])

    print(f"downloaded_files={progress.completed}")
    return 0


def main() -> int:
    args = parse_args()
    client = S3HttpClient(pool_maxsize=max(getattr(args, "workers", 4), getattr(args, "plan_workers", 1)) + 8)

    if args.mode == "grib2":
        day = normalize_date(args.date)
        cycle = normalize_cycle(args.cycle) if args.cycle else None
        if cycle is None:
            day, cycle = find_latest_grib2_cycle(day, client)
        key = resolve_grib2_key(day, cycle, args.forecast_hour, args.region)
        path = client.download_key(
            base_url=GRIB2_BASE,
            key=key,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            preserve_tree=False,
            progress_label="nbm-grib2",
        )
        if args.with_idx:
            client.download_key(
                base_url=GRIB2_BASE,
                key=f"{key}.idx",
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                preserve_tree=False,
                progress_label="nbm-grib2-idx",
            )
        print(f"init_date={day:%Y-%m-%d} cycle={cycle} file={path.name}")
        return 0

    if args.mode == "cog":
        day = normalize_date(args.date)
        version, cycle = resolve_cog_version_and_cycle(
            day=day,
            domain=args.domain,
            version=args.version,
            cycle=args.cycle,
            client=client,
        )
        key = resolve_cog_key(
            day=day,
            cycle=cycle,
            version=version,
            domain=args.domain,
            variable=args.variable,
            valid=args.valid,
            client=client,
        )
        path = client.download_key(
            base_url=COG_BASE,
            key=key,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            preserve_tree=False,
        )
        print(
            " ".join(
                [
                    f"init_date={day:%Y-%m-%d}",
                    f"version={version}",
                    f"cycle={cycle}",
                    f"variable={args.variable}",
                    f"file={path.name}",
                ]
            )
        )
        return 0

    if args.mode == "cog-batch":
        return run_cog_batch(args, client=client)

    raise SystemExit("Unsupported mode")


if __name__ == "__main__":
    sys.exit(main())
