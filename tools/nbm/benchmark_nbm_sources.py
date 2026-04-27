from __future__ import annotations

import argparse
import csv
import datetime as dt
import ftplib
import hashlib
import json
import pathlib
import socket
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, TextIO
from urllib.parse import urlparse

import requests

AWS_BASE = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
NOAA_HTTPS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
NOAA_FTP_BASE = "ftp://ftp.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
NOAA_HTTPS_RAW_CANDIDATE_ROOTS = (
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/v4.3",
    "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/v4.2",
)
NCO_INVENTORY_BASE = "https://www.nco.ncep.noaa.gov/pmb/products/blend"
DEFAULT_SOURCES = ("aws", "noaa_https", "noaa_ftp")
SOURCE_BASES = {
    "aws": AWS_BASE,
    "noaa_https": NOAA_HTTPS_BASE,
    "noaa_ftp": NOAA_FTP_BASE,
}
REGION_CHOICES = ("co", "ak", "gu", "hi", "pr", "oc", "global")
STREAM_CHUNK_BYTES = 1024 * 1024
PROGRESS_UPDATE_EVERY_BYTES = 10 * 1024 * 1024
DEFAULT_BAR_WIDTH = 24
_STREAM_RENDER_LOCK = threading.Lock()
DEFAULT_FTP_TIMEOUT_SECONDS = 10.0


@dataclass
class DownloadResult:
    bytes_downloaded: int
    sha256: str


ProgressCallback = Callable[[int, Optional[int], float], None]


def normalize_date(value: str | None) -> dt.date:
    if value is None:
        return dt.datetime.utcnow().date()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise SystemExit(f"Unsupported date format: {value}")


def normalize_cycle(value: str) -> str:
    token = value.strip()
    if token.endswith("00") and len(token) == 4:
        token = token[:2]
    if len(token) == 1:
        token = f"0{token}"
    if len(token) != 2 or not token.isdigit():
        raise SystemExit(f"Unsupported cycle format: {value}")
    return token


def resolve_grib2_key(day: dt.date, cycle: str, forecast_hour: int, region: str) -> str:
    return (
        f"blend.{day:%Y%m%d}/{cycle}/core/"
        f"blend.t{cycle}z.core.f{forecast_hour:03d}.{region}.grib2"
    )


def format_progress_line(
    completed: int,
    total: int,
    *,
    label: str | None = None,
    unit: str = "item",
    stage: str | None = None,
    status: str | None = None,
    parent_label: str | None = None,
    parent_completed: int | None = None,
    parent_total: int | None = None,
    parent_unit: str | None = None,
    bar_width: int = DEFAULT_BAR_WIDTH,
) -> str:
    safe_total = max(0, total)
    safe_completed = min(max(0, completed), safe_total) if safe_total else 0
    pct = 100.0 if safe_total == 0 else (safe_completed / safe_total) * 100.0
    filled = bar_width if safe_total == 0 else int(round((safe_completed / safe_total) * bar_width))
    bar = "#" * filled + "-" * max(0, bar_width - filled)
    parts = ["[progress]"]
    if label:
        parts.append(label)
    parts.append(f"{safe_completed}/{safe_total} {unit}")
    parts.append(f"({pct:5.1f}%)")
    parts.append(f"[{bar}]")
    if parent_label and parent_completed is not None and parent_total is not None:
        parent_unit_text = f" {parent_unit}" if parent_unit else ""
        parts.append(f"parent={parent_label}:{parent_completed}/{parent_total}{parent_unit_text}")
    if stage:
        parts.append(f"stage={stage}")
    if status:
        parts.append(f"status={status}")
    return " ".join(parts)


def emit_progress_message(message: str, *, stream: TextIO | None = None) -> None:
    target = stream if stream is not None else sys.stdout
    is_tty = bool(getattr(target, "isatty", lambda: False)())
    with _STREAM_RENDER_LOCK:
        if is_tty:
            target.write("\n")
        target.write(f"{message}\n")
        target.flush()


class ProgressBar:
    def __init__(
        self,
        total: int,
        *,
        label: str | None = None,
        unit: str = "item",
        stream: TextIO | None = None,
        is_tty: bool | None = None,
        bar_width: int = DEFAULT_BAR_WIDTH,
    ) -> None:
        self.total = max(0, int(total))
        self.label = label
        self.unit = unit
        self.stream = stream if stream is not None else sys.stdout
        self.bar_width = bar_width
        self.completed = 0
        self.stage: str | None = None
        self.status: str | None = None
        self.parent_label: str | None = None
        self.parent_completed: int | None = None
        self.parent_total: int | None = None
        self.parent_unit: str | None = None
        self._closed = False
        self._lock = threading.Lock()
        self._is_tty = bool(is_tty) if is_tty is not None else bool(getattr(self.stream, "isatty", lambda: False)())

    def update(
        self,
        *,
        completed: int | None = None,
        stage: str | None = None,
        status: str | None = None,
        parent_label: str | None = None,
        parent_completed: int | None = None,
        parent_total: int | None = None,
        parent_unit: str | None = None,
    ) -> int:
        with self._lock:
            if self._closed or self.total == 0:
                return self.completed
            if completed is not None:
                self.completed = min(self.total, max(0, int(completed)))
            self.stage = stage
            self.status = status
            self.parent_label = parent_label
            self.parent_completed = parent_completed
            self.parent_total = parent_total
            self.parent_unit = parent_unit
            return self.completed

    def advance(
        self,
        step: int = 1,
        *,
        stage: str | None = None,
        status: str | None = None,
        parent_label: str | None = None,
        parent_completed: int | None = None,
        parent_total: int | None = None,
        parent_unit: str | None = None,
    ) -> int:
        with self._lock:
            if self._closed or self.total == 0:
                return self.completed
            self.completed = min(self.total, self.completed + max(0, int(step)))
            self.stage = stage
            self.status = status
            self.parent_label = parent_label
            self.parent_completed = parent_completed
            self.parent_total = parent_total
            self.parent_unit = parent_unit
            self._render()
            return self.completed

    def close(self, *, stage: str | None = None, status: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if self.total:
                already_complete = self.completed >= self.total and stage is None and status is None
                self.completed = self.total
                if stage is not None:
                    self.stage = stage
                if status is not None:
                    self.status = status
                if not already_complete:
                    self._render()
                self.stream.flush()
            self._closed = True

    def _render(self) -> None:
        line = format_progress_line(
            self.completed,
            self.total,
            label=self.label,
            unit=self.unit,
            stage=self.stage,
            status=self.status,
            parent_label=self.parent_label,
            parent_completed=self.parent_completed,
            parent_total=self.parent_total,
            parent_unit=self.parent_unit,
            bar_width=self.bar_width,
        )
        with _STREAM_RENDER_LOCK:
            self.stream.write(f"{line}\n")
            self.stream.flush()


def default_output_dir() -> pathlib.Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return pathlib.Path("tools/nbm/data/runtime/download_benchmarks") / stamp


def parse_sources(value: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        if token not in SOURCE_BASES:
            raise argparse.ArgumentTypeError(f"Unsupported source: {token}")
        if token not in seen:
            seen.add(token)
            tokens.append(token)
    if not tokens:
        raise argparse.ArgumentTypeError("At least one source is required.")
    return tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one NBM GRIB2 core file across public download sources.")
    parser.add_argument("--date", required=True, help="Initialization date in YYYY-MM-DD or YYYYMMDD")
    parser.add_argument("--cycle", required=True, help="Initialization hour, e.g. 11")
    parser.add_argument("--forecast-hour", type=int, required=True, help="Forecast hour")
    parser.add_argument(
        "--region",
        default="co",
        choices=REGION_CHOICES,
        help="NBM regional grid code",
    )
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=list(DEFAULT_SOURCES),
        help="Comma-separated source ids: aws,noaa_https,noaa_ftp",
    )
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per source")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs per source")
    parser.add_argument(
        "--ftp-timeout-seconds",
        type=float,
        default=DEFAULT_FTP_TIMEOUT_SECONDS,
        help="Timeout for FTP control/data operations and preflight connect checks",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=default_output_dir(),
        help="Directory to place benchmark artifacts",
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if args.warmup_runs < 0:
        raise SystemExit("--warmup-runs must be 0 or greater")
    if args.forecast_hour < 0:
        raise SystemExit("--forecast-hour must be 0 or greater")
    if args.ftp_timeout_seconds <= 0:
        raise SystemExit("--ftp-timeout-seconds must be greater than 0")
    return args


def resolve_source_url(source: str, key: str) -> str:
    return f"{SOURCE_BASES[source]}/{key}"


def key_filename(key: str) -> str:
    return pathlib.Path(key).name


def inventory_url_for_key(*, key: str, cycle: str, region: str) -> str:
    region_dir_map = {
        "co": "conus",
        "ak": "alaska",
        "gu": "guam",
        "hi": "hawaii",
        "pr": "puertorico",
        "oc": "oceanic",
        "global": "global",
    }
    region_dir = region_dir_map[region]
    return f"{NCO_INVENTORY_BASE}/{region_dir}/{cycle}/{key_filename(key)}.shtml"


def probe_http_url(url: str, *, session_factory: Callable[[], requests.Session] = requests.Session) -> bool:
    session = session_factory()
    try:
        with session:
            response = session.head(url, timeout=(5, 20), allow_redirects=True)
            return response.status_code == 200
    except Exception:
        return False


def probe_tcp_connect(host: str, port: int, *, timeout_seconds: float) -> tuple[bool, str]:
    sock = socket.socket()
    sock.settimeout(timeout_seconds)
    try:
        sock.connect((host, port))
        return True, "ok"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sock.close()


def resolve_noaa_https_target(
    *,
    key: str,
    cycle: str,
    region: str,
    session_factory: Callable[[], requests.Session] = requests.Session,
) -> tuple[str, str]:
    for root in NOAA_HTTPS_RAW_CANDIDATE_ROOTS:
        candidate = f"{root}/{key}"
        if probe_http_url(candidate, session_factory=session_factory):
            return candidate, "grib2"
    return inventory_url_for_key(key=key, cycle=cycle, region=region), "inventory_html"


def build_round_robin_schedule(*, sources: list[str], warmup_runs: int, measured_runs: int) -> list[tuple[str, int, str]]:
    schedule: list[tuple[str, int, str]] = []
    for phase, total_rounds in (("warmup", warmup_runs), ("measured", measured_runs)):
        for round_index in range(total_rounds):
            ordered = [sources[(round_index + offset) % len(sources)] for offset in range(len(sources))]
            for source in ordered:
                schedule.append((phase, round_index + 1, source))
    return schedule


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def format_transfer_status(
    *,
    source: str,
    phase: str,
    run_index: int,
    bytes_downloaded: int,
    elapsed_seconds: float,
    total_bytes: int | None,
) -> str:
    speed_mib_per_second = (bytes_downloaded / elapsed_seconds) / (1024 * 1024) if elapsed_seconds > 0 else 0.0
    bytes_text = f"{bytes_downloaded / (1024 * 1024):.1f}MiB"
    total_text = f"/{total_bytes / (1024 * 1024):.1f}MiB" if total_bytes else ""
    return (
        f"source={source} phase={phase} run={run_index} "
        f"elapsed={elapsed_seconds:.1f}s downloaded={bytes_text}{total_text} "
        f"speed={speed_mib_per_second:.2f}MiB/s"
    )


def stream_http_download(
    url: str,
    destination: pathlib.Path,
    *,
    session_factory: Callable[[], requests.Session] = requests.Session,
    progress_callback: ProgressCallback | None = None,
) -> DownloadResult:
    digest = hashlib.sha256()
    downloaded = 0
    tmp_destination = destination.with_name(destination.name + ".part")
    tmp_destination.unlink(missing_ok=True)
    session = session_factory()
    started = time.perf_counter()
    try:
        with session:
            with session.get(url, stream=True, timeout=(5, 300)) as response:
                response.raise_for_status()
                content_length = response.headers.get("Content-Length")
                total_bytes = int(content_length) if content_length is not None else None
                next_progress_bytes = PROGRESS_UPDATE_EVERY_BYTES
                with tmp_destination.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        elapsed_seconds = max(time.perf_counter() - started, 1e-9)
                        if progress_callback and (
                            downloaded >= next_progress_bytes or (total_bytes is not None and downloaded >= total_bytes)
                        ):
                            progress_callback(downloaded, total_bytes, elapsed_seconds)
                            next_progress_bytes += PROGRESS_UPDATE_EVERY_BYTES
                if content_length is not None and int(content_length) != downloaded:
                    raise RuntimeError(
                        f"Content-Length mismatch for {url}: expected {content_length}, got {downloaded}"
                    )
        if progress_callback:
            progress_callback(downloaded, total_bytes, max(time.perf_counter() - started, 1e-9))
        tmp_destination.replace(destination)
        return DownloadResult(bytes_downloaded=downloaded, sha256=digest.hexdigest())
    except Exception:
        tmp_destination.unlink(missing_ok=True)
        raise


def stream_ftp_download(
    url: str,
    destination: pathlib.Path,
    *,
    ftp_factory: Callable[[str], ftplib.FTP] = ftplib.FTP,
    progress_callback: ProgressCallback | None = None,
    timeout_seconds: float = DEFAULT_FTP_TIMEOUT_SECONDS,
) -> DownloadResult:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.path:
        raise RuntimeError(f"Unsupported FTP url: {url}")
    ok, detail = probe_tcp_connect(parsed.hostname, 21, timeout_seconds=timeout_seconds)
    if not ok:
        raise TimeoutError(f"FTP control port 21 unreachable for {parsed.hostname}: {detail}")
    digest = hashlib.sha256()
    downloaded = 0
    tmp_destination = destination.with_name(destination.name + ".part")
    tmp_destination.unlink(missing_ok=True)
    if ftp_factory is ftplib.FTP:
        ftp = ftplib.FTP()
        ftp.connect(parsed.hostname, 21, timeout=timeout_seconds)
    else:
        ftp = ftp_factory(parsed.hostname)
    started = time.perf_counter()
    try:
        ftp.login()
        ftp.voidcmd("TYPE I")
        remote_path = parsed.path
        try:
            total_bytes = ftp.size(remote_path)
        except Exception:
            total_bytes = None
        next_progress_bytes = [PROGRESS_UPDATE_EVERY_BYTES]
        with tmp_destination.open("wb") as handle:
            def write_chunk(chunk: bytes) -> None:
                nonlocal downloaded
                handle.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                elapsed_seconds = max(time.perf_counter() - started, 1e-9)
                if progress_callback and (
                    downloaded >= next_progress_bytes[0] or (total_bytes is not None and downloaded >= total_bytes)
                ):
                    progress_callback(downloaded, total_bytes, elapsed_seconds)
                    next_progress_bytes[0] += PROGRESS_UPDATE_EVERY_BYTES

            ftp.retrbinary(f"RETR {remote_path}", write_chunk, blocksize=STREAM_CHUNK_BYTES)
        if progress_callback:
            progress_callback(downloaded, total_bytes, max(time.perf_counter() - started, 1e-9))
        tmp_destination.replace(destination)
        return DownloadResult(bytes_downloaded=downloaded, sha256=digest.hexdigest())
    except Exception:
        tmp_destination.unlink(missing_ok=True)
        raise
    finally:
        try:
            ftp.quit()
        except Exception:
            try:
                ftp.close()
            except Exception:
                pass


def download_once(
    *,
    source: str,
    url: str,
    destination: pathlib.Path,
    session_factory: Callable[[], requests.Session] = requests.Session,
    ftp_factory: Callable[[str], ftplib.FTP] = ftplib.FTP,
    progress_callback: ProgressCallback | None = None,
    ftp_timeout_seconds: float = DEFAULT_FTP_TIMEOUT_SECONDS,
) -> DownloadResult:
    if source in {"aws", "noaa_https"}:
        return stream_http_download(url, destination, session_factory=session_factory, progress_callback=progress_callback)
    if source == "noaa_ftp":
        return stream_ftp_download(
            url,
            destination,
            ftp_factory=ftp_factory,
            progress_callback=progress_callback,
            timeout_seconds=ftp_timeout_seconds,
        )
    raise RuntimeError(f"Unsupported source: {source}")


def benchmark_sources(
    *,
    day: dt.date,
    cycle: str,
    forecast_hour: int,
    region: str,
    sources: list[str],
    measured_runs: int,
    warmup_runs: int,
    output_dir: pathlib.Path,
    session_factory: Callable[[], requests.Session] = requests.Session,
    ftp_factory: Callable[[str], ftplib.FTP] = ftplib.FTP,
    ftp_timeout_seconds: float = DEFAULT_FTP_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    key = resolve_grib2_key(day, cycle, forecast_hour, region)
    schedule = build_round_robin_schedule(sources=sources, warmup_runs=warmup_runs, measured_runs=measured_runs)
    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    overall_progress = ProgressBar(len(schedule), label="NBM benchmark", unit="run") if schedule else None
    for phase, run_index, source in schedule:
        if source == "aws":
            url = resolve_source_url(source, key)
            content_kind = "grib2"
        elif source == "noaa_https":
            url, content_kind = resolve_noaa_https_target(
                key=key,
                cycle=cycle,
                region=region,
                session_factory=session_factory,
            )
        else:
            url = resolve_source_url(source, key)
            content_kind = "grib2"
        destination = downloads_dir / source / phase / f"run_{run_index:02d}" / pathlib.Path(key).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        if overall_progress is not None:
            overall_progress.update(
                stage="start",
                status=f"source={source} phase={phase} run={run_index} content={content_kind}",
            )
        row = {
            "source": source,
            "url": url,
            "date": day.isoformat(),
            "cycle": cycle,
            "forecast_hour": forecast_hour,
            "region": region,
            "run_index": run_index,
            "phase": phase,
            "content_kind": content_kind,
            "bytes_downloaded": 0,
            "elapsed_seconds": 0.0,
            "throughput_mbps": 0.0,
            "success": False,
            "error_message": "",
            "sha256": "",
            "local_path": str(destination),
        }
        transfer_progress: ProgressBar | None = None

        def on_progress(bytes_downloaded: int, total_bytes: int | None, elapsed_seconds: float) -> None:
            nonlocal transfer_progress
            status = format_transfer_status(
                source=source,
                phase=phase,
                run_index=run_index,
                bytes_downloaded=bytes_downloaded,
                elapsed_seconds=elapsed_seconds,
                total_bytes=total_bytes,
            )
            if total_bytes and total_bytes > 0:
                if transfer_progress is None:
                    transfer_progress = ProgressBar(
                        total_bytes,
                        label=f"{source} {phase}",
                        unit="byte",
                    )
                transfer_progress.update(completed=bytes_downloaded)
                transfer_progress.advance(
                    0,
                    stage="download",
                    status=status,
                    parent_label="NBM benchmark",
                    parent_completed=len(rows) + 1,
                    parent_total=len(schedule),
                    parent_unit="run",
                )
            else:
                emit_progress_message(f"[status] {status}")

        try:
            result = download_once(
                source=source,
                url=url,
                destination=destination,
                session_factory=session_factory,
                ftp_factory=ftp_factory,
                progress_callback=on_progress,
                ftp_timeout_seconds=ftp_timeout_seconds,
            )
            elapsed = time.perf_counter() - started
            row["bytes_downloaded"] = result.bytes_downloaded
            row["elapsed_seconds"] = elapsed
            row["throughput_mbps"] = (result.bytes_downloaded / elapsed) / (1024 * 1024) if elapsed > 0 else 0.0
            row["success"] = True
            row["sha256"] = result.sha256
            if transfer_progress is not None:
                transfer_progress.close(stage="complete", status=f"{format_transfer_status(source=source, phase=phase, run_index=run_index, bytes_downloaded=result.bytes_downloaded, elapsed_seconds=max(elapsed, 1e-9), total_bytes=result.bytes_downloaded)} sha256={result.sha256[:12]}")
            else:
                emit_progress_message(
                    f"[status] complete source={source} phase={phase} run={run_index} content={content_kind} "
                    f"elapsed={elapsed:.1f}s bytes={result.bytes_downloaded} sha256={result.sha256[:12]}"
                )
        except Exception as exc:
            row["elapsed_seconds"] = time.perf_counter() - started
            row["error_message"] = str(exc)
            if transfer_progress is not None:
                transfer_progress.close(stage="error", status=f"source={source} phase={phase} run={run_index} error={type(exc).__name__}")
            else:
                emit_progress_message(
                    f"[status] error source={source} phase={phase} run={run_index} content={content_kind} error={type(exc).__name__} detail={exc}"
                )
        rows.append(row)
        if overall_progress is not None:
            overall_progress.advance(
                stage="complete" if row["success"] else "error",
                status=f"source={source} phase={phase} run={run_index} content={content_kind} success={row['success']}",
            )
    if overall_progress is not None:
        overall_progress.close(stage="finalize", status=f"runs={len(rows)}")
    return rows


def summarize_results(rows: list[dict[str, object]], *, sources: list[str], key: str) -> dict[str, object]:
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    summary_sources: dict[str, dict[str, object]] = {}
    ranking: list[dict[str, object]] = []
    successful_sources = {
        str(row["source"])
        for row in measured_rows
        if row["success"]
    }
    successful_hashes: set[str] = {
        str(row["sha256"])
        for row in measured_rows
        if row["success"] and row["sha256"]
    }
    content_kinds = {
        str(row.get("content_kind", "grib2"))
        for row in measured_rows
        if row["success"]
    }
    if len(successful_sources) < len(sources):
        sha_status = "insufficient_successful_sources"
        benchmark_valid = False
    elif len(content_kinds) > 1:
        sha_status = "not_comparable"
        benchmark_valid = False
    elif len(successful_hashes) > 1:
        sha_status = "mismatch"
        benchmark_valid = False
    elif len(successful_hashes) == 1:
        sha_status = "match"
        benchmark_valid = True
    else:
        sha_status = "insufficient_data"
        benchmark_valid = False

    for source in sources:
        source_rows = [row for row in measured_rows if row["source"] == source]
        successes = [row for row in source_rows if row["success"]]
        elapsed_values = [float(row["elapsed_seconds"]) for row in successes]
        throughput_values = [float(row["throughput_mbps"]) for row in successes]
        payload_sizes = sorted({int(row["bytes_downloaded"]) for row in successes})
        source_summary = {
            "success_count": len(successes),
            "attempt_count": len(source_rows),
            "median_wall_time_seconds": statistics.median(elapsed_values) if elapsed_values else None,
            "mean_wall_time_seconds": statistics.mean(elapsed_values) if elapsed_values else None,
            "best_wall_time_seconds": min(elapsed_values) if elapsed_values else None,
            "mean_throughput_mbps": statistics.mean(throughput_values) if throughput_values else None,
            "payload_size_bytes": payload_sizes[0] if len(payload_sizes) == 1 else (payload_sizes if payload_sizes else None),
            "content_kind": successes[0]["content_kind"] if successes else None,
        }
        summary_sources[source] = source_summary
        if source_summary["median_wall_time_seconds"] is not None:
            ranking.append(
                {
                    "source": source,
                    "median_wall_time_seconds": source_summary["median_wall_time_seconds"],
                    "mean_throughput_mbps": source_summary["mean_throughput_mbps"],
                }
            )

    ranking.sort(key=lambda item: (float(item["median_wall_time_seconds"]), item["source"]))
    return {
        "sample_key": key,
        "benchmark_valid": benchmark_valid,
        "sha256_agreement_status": sha_status,
        "sources": summary_sources,
        "ranked_fastest_by_median_wall_time": ranking,
    }


def write_reports(output_dir: pathlib.Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_runs.csv"
    json_path = output_dir / "benchmark_runs.json"
    summary_path = output_dir / "benchmark_summary.json"
    fieldnames = [
        "source",
        "url",
        "date",
        "cycle",
        "forecast_hour",
        "region",
        "run_index",
        "phase",
        "content_kind",
        "bytes_downloaded",
        "elapsed_seconds",
        "throughput_mbps",
        "success",
        "error_message",
        "sha256",
        "local_path",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2))
    summary_path.write_text(json.dumps(summary, indent=2))


def print_summary(summary: dict[str, object]) -> None:
    print(f"sample_key={summary['sample_key']}")
    print(f"benchmark_valid={summary['benchmark_valid']} sha256_agreement_status={summary['sha256_agreement_status']}")
    ranking = summary["ranked_fastest_by_median_wall_time"]
    if not ranking:
        print("no_successful_measured_runs=true")
        return
    for item in ranking:
        mean_throughput = item["mean_throughput_mbps"]
        throughput_text = f"{mean_throughput:.2f}" if mean_throughput is not None else "n/a"
        print(
            f"source={item['source']} median_seconds={item['median_wall_time_seconds']:.3f} "
            f"mean_mib_per_second={throughput_text}"
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    day = normalize_date(args.date)
    cycle = normalize_cycle(args.cycle)
    key = resolve_grib2_key(day, cycle, args.forecast_hour, args.region)
    rows = benchmark_sources(
        day=day,
        cycle=cycle,
        forecast_hour=args.forecast_hour,
        region=args.region,
        sources=args.sources,
        measured_runs=args.runs,
        warmup_runs=args.warmup_runs,
        output_dir=args.output_dir,
        ftp_timeout_seconds=args.ftp_timeout_seconds,
    )
    if not any(row["success"] for row in rows):
        write_reports(
            args.output_dir,
            rows,
            summarize_results(rows, sources=args.sources, key=key),
        )
        raise SystemExit("All requested sources failed.")
    summary = summarize_results(rows, sources=args.sources, key=key)
    write_reports(args.output_dir, rows, summary)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
