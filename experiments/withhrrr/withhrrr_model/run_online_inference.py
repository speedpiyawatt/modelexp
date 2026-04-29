from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

from experiments.withhrrr.withhrrr_model.polymarket_event import weather_event_slug_for_date


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_RUNTIME_ROOT = pathlib.Path("experiments/withhrrr/data/runtime/online_inference")
CANDIDATE_LAMP_CYCLES = ("0230", "0330", "0430")
AUTO_POLYMARKET_EVENT_SLUG = "__auto__"
DEFAULT_NEARBY_STATIONS = ("KJRB", "KJFK", "KEWR", "KTEB")
LAMP_NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/lmp/prod"
LAMP_ARCHIVE_BASE = "https://lamp.mdl.nws.noaa.gov/lamp/Data/archives"
IEM_AFOS_BASE = "https://mesonet.agron.iastate.edu/p.php"
AFOS_PRE_RE = re.compile(r"<pre[^>]*class=[\"']afos-pre[\"'][^>]*>(?P<text>.*?)</pre>", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch online with-HRRR source inputs and run one target-date inference.")
    parser.add_argument("--target-date-local", required=True, help="KLGA local target date in YYYY-MM-DD.")
    parser.add_argument("--station-id", default="KLGA")
    parser.add_argument("--runtime-root", type=pathlib.Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--prediction-output-dir", type=pathlib.Path, default=pathlib.Path("experiments/withhrrr/data/runtime/predictions"))
    parser.add_argument("--wunderground-tables-dir", type=pathlib.Path, help="Use existing Wunderground labels/obs tables instead of the runtime-root default.")
    parser.add_argument("--lamp-root", type=pathlib.Path, help="Use existing LAMP overnight summaries instead of the runtime-root default.")
    parser.add_argument("--nbm-root", type=pathlib.Path, help="Use existing NBM overnight summaries instead of the runtime-root default.")
    parser.add_argument("--hrrr-root", type=pathlib.Path, help="Use existing HRRR overnight summaries instead of the runtime-root default.")
    parser.add_argument(
        "--nearby-obs-path",
        action="append",
        default=[],
        help="Existing nearby station obs parquet in STATION=PATH form. Can be repeated. Suppresses fetching for supplied stations.",
    )
    parser.add_argument(
        "--nearby-station-id",
        action="append",
        dest="nearby_station_ids",
        default=None,
        help="Nearby Wunderground station to fetch for source-trust features. Can be repeated. Defaults to KJRB/KJFK/KEWR/KTEB.",
    )
    parser.add_argument("--skip-nearby-wunderground", action="store_true")
    parser.add_argument("--lamp-source", choices=("auto", "live", "archive", "iem"), default="auto")
    parser.add_argument("--event-bin", action="append", default=[])
    parser.add_argument("--event-bins-path", type=pathlib.Path)
    parser.add_argument(
        "--polymarket-event-slug",
        nargs="?",
        const=AUTO_POLYMARKET_EVENT_SLUG,
        help="Fetch event metadata from Polymarket Gamma API and use its temperature bins. Omit the value to derive the standard weather-event slug from --target-date-local.",
    )
    parser.add_argument("--max-so-far-f", type=float)
    parser.add_argument("--skip-wunderground", action="store_true")
    parser.add_argument("--skip-lamp", action="store_true")
    parser.add_argument("--skip-nbm", action="store_true")
    parser.add_argument("--skip-hrrr", action="store_true")
    parser.add_argument("--nbm-download-workers", type=int, default=4)
    parser.add_argument("--nbm-reduce-workers", type=int, default=2)
    parser.add_argument("--nbm-extract-workers", type=int, default=2)
    parser.add_argument("--nbm-wgrib2-threads", type=int, default=1)
    parser.add_argument("--hrrr-max-workers", type=int, default=4)
    parser.add_argument("--hrrr-download-workers", type=int, default=4)
    parser.add_argument("--hrrr-reduce-workers", type=int, default=2)
    parser.add_argument("--hrrr-extract-workers", type=int, default=2)
    parser.add_argument("--hrrr-wgrib2-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-lamp-fetch-error", action="store_true")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep downloaded/intermediate runtime artifacts after a successful prediction. By default they are deleted.",
    )
    return parser.parse_args()


def parse_local_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def target_date_token(target_date: dt.date) -> str:
    return f"target_date_local={target_date.isoformat()}"


def cutoff_utc_date(target_date: dt.date) -> dt.date:
    cutoff_local = dt.datetime.combine(target_date, dt.time(0, 5), tzinfo=NY_TZ)
    return cutoff_local.astimezone(dt.timezone.utc).date()


def lamp_live_url(utc_date: dt.date, cycle: str) -> str:
    daily_dir = utc_date.strftime("%Y%m%d")
    return f"{LAMP_NOMADS_BASE}/lmp.{daily_dir}/lmp.t{cycle}z.lavtxt.ascii"


def lamp_archive_monthly_url(utc_date: dt.date, cycle: str) -> str:
    year_month = utc_date.strftime("%Y%m")
    return f"{LAMP_ARCHIVE_BASE}/lmp_lavtxt.{year_month}.{cycle}z.gz"


def lamp_archive_yearly_url(utc_date: dt.date) -> str:
    return f"{LAMP_ARCHIVE_BASE}/lmp_lavtxt.{utc_date.year}.tar"


def iem_lav_pil(station_id: str) -> str:
    station = station_id.upper()
    if len(station) == 4 and station.startswith("K"):
        station = station[1:]
    return f"LAV{station}"


def iem_lamp_pid(*, station_id: str, utc_date: dt.date, cycle: str) -> str:
    issue_hour = cycle[:2]
    return f"{utc_date.strftime('%Y%m%d')}{issue_hour}00-KWNO-FOUS11-{iem_lav_pil(station_id)}"


def iem_lamp_url(*, station_id: str, utc_date: dt.date, cycle: str) -> str:
    return f"{IEM_AFOS_BASE}?pid={iem_lamp_pid(station_id=station_id, utc_date=utc_date, cycle=cycle)}"


def remote_url_exists(url: str) -> bool:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "withhrrr-online-inference/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    except urllib.error.URLError:
        return False


def iem_lamp_product_exists(url: str) -> bool:
    request = urllib.request.Request(url, headers={"User-Agent": "withhrrr-online-inference/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            page = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError:
        return False
    match = AFOS_PRE_RE.search(page)
    return bool(match and "GFS LAMP GUIDANCE" in html.unescape(match.group("text")))


def lamp_live_available(utc_date: dt.date, *, url_exists=remote_url_exists) -> bool:
    return any(url_exists(lamp_live_url(utc_date, cycle)) for cycle in CANDIDATE_LAMP_CYCLES)


def lamp_archive_available(utc_date: dt.date, *, url_exists=remote_url_exists) -> bool:
    if any(url_exists(lamp_archive_monthly_url(utc_date, cycle)) for cycle in CANDIDATE_LAMP_CYCLES):
        return True
    return url_exists(lamp_archive_yearly_url(utc_date))


def iem_lamp_available(utc_date: dt.date, station_id: str, *, product_exists=iem_lamp_product_exists) -> bool:
    return any(product_exists(iem_lamp_url(station_id=station_id, utc_date=utc_date, cycle=cycle)) for cycle in CANDIDATE_LAMP_CYCLES)


def resolve_lamp_source(
    lamp_source: str,
    target_date: dt.date,
    *,
    station_id: str = "KLGA",
    url_exists=remote_url_exists,
    iem_product_exists=iem_lamp_product_exists,
) -> str:
    if lamp_source != "auto":
        return lamp_source
    utc_date = cutoff_utc_date(target_date)
    if lamp_live_available(utc_date, url_exists=url_exists):
        return "live"
    if lamp_archive_available(utc_date, url_exists=url_exists):
        return "archive"
    if iem_lamp_available(utc_date, station_id, product_exists=iem_product_exists):
        return "iem"
    raise SystemExit(
        "LAMP unavailable for "
        f"target_date_local={target_date.isoformat()} cutoff_utc_date={utc_date.isoformat()}: "
        "not found on live NOMADS, not found in the public LAMP archive, and not found in the IEM AFOS archive. "
        "This can happen for recent past dates after NOMADS live retention expires but before the archive is published. "
        "Try a current live date, a date with archived LAMP, or provide local LAMP artifacts and rerun with --skip-lamp."
    )


def run(command: list[str], *, continue_on_error: bool = False) -> None:
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command)
    if result.returncode != 0 and not continue_on_error:
        raise SystemExit(result.returncode)


def write_run_manifest(
    *,
    runtime_root: pathlib.Path,
    target_date: dt.date,
    station_id: str,
    status: str,
    message: str | None,
    paths: dict[str, str],
) -> pathlib.Path:
    root = runtime_root / "status" / target_date_token(target_date)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "online_inference.manifest.json"
    payload = {
        "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "target_date_local": target_date.isoformat(),
        "station_id": station_id,
        "status": status,
        "message": message,
        "paths": paths,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(path, flush=True)
    return path


def cleanup_runtime_artifacts(runtime_root: pathlib.Path, candidates: list[pathlib.Path]) -> list[str]:
    deleted: list[str] = []
    runtime_root_resolved = runtime_root.resolve()
    for candidate in candidates:
        path = candidate.resolve()
        if not path.exists():
            continue
        if path == runtime_root_resolved or runtime_root_resolved not in path.parents:
            raise SystemExit(f"Refusing to clean path outside runtime root: {candidate}")
        if path.name == "status" or any(parent.name == "status" for parent in path.parents):
            raise SystemExit(f"Refusing to clean status path: {candidate}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(candidate))
        print(f"[cleanup] deleted {candidate}", flush=True)
    return deleted


def fetch_iem_lamp_cycle(*, station_id: str, utc_date: dt.date, cycle: str, output_dir: pathlib.Path, overwrite: bool) -> pathlib.Path:
    destination = output_dir / "source=iem" / f"date_utc={utc_date.isoformat()}" / f"cycle={cycle}" / f"iem.{iem_lav_pil(station_id)}.{utc_date.strftime('%Y%m%d')}.{cycle}z.ascii"
    if destination.exists() and not overwrite:
        print(f"[skip] {destination}", flush=True)
        return destination
    url = iem_lamp_url(station_id=station_id, utc_date=utc_date, cycle=cycle)
    request = urllib.request.Request(url, headers={"User-Agent": "withhrrr-online-inference/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")
    match = AFOS_PRE_RE.search(page)
    if not match:
        raise SystemExit(f"IEM LAMP product not found: {url}")
    text = html.unescape(match.group("text")).strip() + "\n"
    if "GFS LAMP GUIDANCE" not in text:
        raise SystemExit(f"IEM LAMP response did not contain LAMP guidance: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    print(f"[ok] {destination}", flush=True)
    return destination


def fetch_wunderground(args: argparse.Namespace, target_date: dt.date) -> pathlib.Path:
    date_token = target_date_token(target_date)
    history_dir = args.runtime_root / "wunderground_history" / date_token
    tables_dir = args.runtime_root / "wunderground_tables" / date_token
    start_date = target_date - dt.timedelta(days=1)
    command = [
        sys.executable,
        "wunderground/fetch_daily_history.py",
        "--start-date",
        start_date.isoformat(),
        "--end-date",
        target_date.isoformat(),
        "--output-dir",
        str(history_dir),
    ]
    if args.overwrite:
        command.append("--force")
    run(command)
    run(
        [
            sys.executable,
            "wunderground/build_training_tables.py",
            "--history-dir",
            str(history_dir),
            "--output-dir",
            str(tables_dir),
        ]
    )
    return tables_dir


def fetch_nearby_wunderground(args: argparse.Namespace, target_date: dt.date) -> dict[str, pathlib.Path]:
    date_token = target_date_token(target_date)
    start_date = target_date - dt.timedelta(days=1)
    station_ids = [station.upper() for station in (args.nearby_station_ids or DEFAULT_NEARBY_STATIONS)]
    obs_paths: dict[str, pathlib.Path] = {}
    for station_id in station_ids:
        history_dir = args.runtime_root / "nearby_wunderground_history" / date_token / station_id
        tables_dir = args.runtime_root / "nearby_wunderground_tables" / date_token / station_id
        command = [
            sys.executable,
            "wunderground/fetch_daily_history.py",
            "--location-id",
            f"{station_id}:9:US",
            "--start-date",
            start_date.isoformat(),
            "--end-date",
            target_date.isoformat(),
            "--output-dir",
            str(history_dir),
            "--skip-http-status",
            "400",
            "--skip-http-status",
            "404",
        ]
        if args.overwrite:
            command.append("--force")
        run(command)
        run(
            [
                sys.executable,
                "wunderground/build_training_tables.py",
                "--history-dir",
                str(history_dir),
                "--station-id",
                station_id,
                "--output-dir",
                str(tables_dir),
            ]
        )
        obs_path = tables_dir / "wu_obs_intraday.parquet"
        if obs_path.exists():
            obs_paths[station_id] = obs_path
    return obs_paths


def parse_nearby_obs_paths(values: list[str]) -> dict[str, pathlib.Path]:
    obs_paths: dict[str, pathlib.Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--nearby-obs-path must be STATION=PATH, got {value!r}")
        station_id, path_text = value.split("=", 1)
        station_id = station_id.strip().upper()
        path = pathlib.Path(path_text)
        if not station_id:
            raise SystemExit(f"--nearby-obs-path has empty station id: {value!r}")
        if not path.exists():
            raise SystemExit(f"nearby obs path for {station_id} does not exist: {path}")
        obs_paths[station_id] = path
    return obs_paths


def fetch_lamp(args: argparse.Namespace, target_date: dt.date) -> pathlib.Path:
    date_token = target_date_token(target_date)
    raw_dir = args.runtime_root / "lamp_raw" / date_token
    features_dir = args.runtime_root / "lamp_features" / date_token
    overnight_dir = args.runtime_root / "lamp_overnight"
    utc_date = cutoff_utc_date(target_date)
    lamp_source = resolve_lamp_source(args.lamp_source, target_date, station_id=args.station_id)
    print(f"[info] lamp_source={lamp_source} target_date_local={target_date.isoformat()} cutoff_utc_date={utc_date.isoformat()}", flush=True)
    if lamp_source == "live":
        for cycle in CANDIDATE_LAMP_CYCLES:
            command = [
                sys.executable,
                "tools/lamp/fetch_lamp.py",
                "live",
                "--date-utc",
                utc_date.isoformat(),
                "--cycle",
                cycle,
                "--output-dir",
                str(raw_dir),
            ]
            if args.overwrite:
                command.append("--overwrite")
            run(command, continue_on_error=args.continue_on_lamp_fetch_error)
    elif lamp_source == "archive":
        command = [
            sys.executable,
            "tools/lamp/fetch_lamp.py",
            "archive",
            "--start-utc-date",
            utc_date.isoformat(),
            "--end-utc-date",
            utc_date.isoformat(),
            "--output-dir",
            str(raw_dir),
        ]
        for cycle in CANDIDATE_LAMP_CYCLES:
            command.extend(["--cycle", cycle])
        if args.overwrite:
            command.append("--overwrite")
        run(command)
    elif lamp_source == "iem":
        for cycle in CANDIDATE_LAMP_CYCLES:
            fetch_iem_lamp_cycle(station_id=args.station_id, utc_date=utc_date, cycle=cycle, output_dir=raw_dir, overwrite=args.overwrite)
    else:
        raise ValueError(f"Unsupported LAMP source: {lamp_source}")

    run(
        [
            sys.executable,
            "tools/lamp/build_lamp_klga_features.py",
            str(raw_dir),
            "--output-dir",
            str(features_dir),
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "experiments.no_hrrr_model.no_hrrr_model.build_lamp_overnight",
            "--features-root",
            str(features_dir / "station_id=KLGA"),
            "--output-dir",
            str(overnight_dir),
            "--start-local-date",
            target_date.isoformat(),
            "--end-local-date",
            target_date.isoformat(),
        ]
    )
    return overnight_dir


def fetch_nbm(args: argparse.Namespace, target_date: dt.date) -> pathlib.Path:
    run_root = args.runtime_root / "nbm"
    command = [
        sys.executable,
        "tools/nbm/run_nbm_monthly_backfill.py",
        "--start-local-date",
        target_date.isoformat(),
        "--end-local-date",
        target_date.isoformat(),
        "--run-root",
        str(run_root),
        "--selection-mode",
        "overnight_0005",
        "--day-workers",
        "1",
        "--download-workers",
        str(args.nbm_download_workers),
        "--reduce-workers",
        str(args.nbm_reduce_workers),
        "--extract-workers",
        str(args.nbm_extract_workers),
        "--reduce-queue-size",
        "2",
        "--extract-queue-size",
        "2",
        "--wgrib2-threads",
        str(args.nbm_wgrib2_threads),
        "--batch-reduce-mode",
        "cycle",
        "--progress-mode",
        "log",
    ]
    if args.overwrite:
        command.append("--overwrite")
    run(command)
    return run_root / "nbm_overnight"


def fetch_hrrr(args: argparse.Namespace, target_date: dt.date) -> pathlib.Path:
    run_root = args.runtime_root / "hrrr"
    command = [
        sys.executable,
        "tools/hrrr/run_hrrr_monthly_backfill.py",
        "--start-local-date",
        target_date.isoformat(),
        "--end-local-date",
        target_date.isoformat(),
        "--run-root",
        str(run_root),
        "--selection-mode",
        "overnight_0005",
        "--batch-reduce-mode",
        "cycle",
        "--day-workers",
        "1",
        "--max-workers",
        str(args.hrrr_max_workers),
        "--download-workers",
        str(args.hrrr_download_workers),
        "--reduce-workers",
        str(args.hrrr_reduce_workers),
        "--extract-workers",
        str(args.hrrr_extract_workers),
        "--reduce-queue-size",
        "2",
        "--extract-queue-size",
        "2",
        "--range-merge-gap-bytes",
        "65536",
        "--crop-method",
        "auto",
        "--crop-grib-type",
        "same",
        "--wgrib2-threads",
        str(args.hrrr_wgrib2_threads),
        "--extract-method",
        "wgrib2-bin",
        "--summary-profile",
        "overnight",
        "--skip-provenance",
        "--progress-mode",
        "log",
    ]
    run(command)
    return run_root / "hrrr_summary"


def main() -> int:
    args = parse_args()
    target_date = parse_local_date(args.target_date_local)
    args.runtime_root.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    date_token = target_date_token(target_date)
    wu_tables_dir = args.wunderground_tables_dir or (args.runtime_root / "wunderground_tables" / date_token)
    lamp_overnight_dir = args.lamp_root or (args.runtime_root / "lamp_overnight")
    nbm_overnight_dir = args.nbm_root or (args.runtime_root / "nbm" / "nbm_overnight")
    hrrr_summary_dir = args.hrrr_root or (args.runtime_root / "hrrr" / "hrrr_summary")
    paths.update(
        {
            "wunderground_history_dir": str(args.runtime_root / "wunderground_history" / date_token),
            "wunderground_tables_dir": str(wu_tables_dir),
            "wunderground_labels_path": str(wu_tables_dir / "labels_daily.parquet"),
            "wunderground_obs_path": str(wu_tables_dir / "wu_obs_intraday.parquet"),
            "lamp_raw_dir": str(args.runtime_root / "lamp_raw" / date_token),
            "lamp_features_dir": str(args.runtime_root / "lamp_features" / date_token),
            "lamp_overnight_dir": str(lamp_overnight_dir),
            "nbm_overnight_dir": str(nbm_overnight_dir),
            "hrrr_summary_dir": str(hrrr_summary_dir),
        }
    )

    try:
        cleanup_candidates: list[pathlib.Path] = []
        if not args.skip_wunderground:
            wu_tables_dir = fetch_wunderground(args, target_date)
            paths["wunderground_tables_dir"] = str(wu_tables_dir)
            paths["wunderground_labels_path"] = str(wu_tables_dir / "labels_daily.parquet")
            paths["wunderground_obs_path"] = str(wu_tables_dir / "wu_obs_intraday.parquet")
            cleanup_candidates.extend(
                [
                    args.runtime_root / "wunderground_history" / date_token,
                    args.runtime_root / "wunderground_tables" / date_token,
                ]
            )
        nearby_obs_paths: dict[str, pathlib.Path] = parse_nearby_obs_paths(args.nearby_obs_path)
        if not args.skip_nearby_wunderground:
            fetched_nearby_paths = fetch_nearby_wunderground(args, target_date)
            nearby_obs_paths = {**fetched_nearby_paths, **nearby_obs_paths}
            paths["nearby_wunderground_obs_paths"] = json.dumps({station: str(path) for station, path in nearby_obs_paths.items()}, sort_keys=True)
            cleanup_candidates.extend(
                [
                    args.runtime_root / "nearby_wunderground_history" / date_token,
                    args.runtime_root / "nearby_wunderground_tables" / date_token,
                ]
            )
        elif nearby_obs_paths:
            paths["nearby_wunderground_obs_paths"] = json.dumps({station: str(path) for station, path in nearby_obs_paths.items()}, sort_keys=True)
        if not args.skip_lamp:
            lamp_overnight_dir = fetch_lamp(args, target_date)
            paths["lamp_overnight_dir"] = str(lamp_overnight_dir)
            cleanup_candidates.extend(
                [
                    args.runtime_root / "lamp_raw" / date_token,
                    args.runtime_root / "lamp_features" / date_token,
                    args.runtime_root / "lamp_overnight",
                ]
            )
        if not args.skip_nbm:
            nbm_overnight_dir = fetch_nbm(args, target_date)
            paths["nbm_overnight_dir"] = str(nbm_overnight_dir)
            cleanup_candidates.append(args.runtime_root / "nbm")
        if not args.skip_hrrr:
            hrrr_summary_dir = fetch_hrrr(args, target_date)
            paths["hrrr_summary_dir"] = str(hrrr_summary_dir)
            cleanup_candidates.append(args.runtime_root / "hrrr")

        event_bins_path = args.event_bins_path
        if args.polymarket_event_slug:
            polymarket_event_slug = args.polymarket_event_slug
            if polymarket_event_slug == AUTO_POLYMARKET_EVENT_SLUG:
                polymarket_event_slug = weather_event_slug_for_date(target_date)
            polymarket_dir = args.runtime_root / "polymarket"
            run(
                [
                    sys.executable,
                    "-m",
                    "experiments.withhrrr.withhrrr_model.polymarket_event",
                    "--event-slug",
                    polymarket_event_slug,
                    "--output-dir",
                    str(polymarket_dir),
                ]
            )
            event_bins_path = polymarket_dir / f"event_slug={polymarket_event_slug}" / "event_bins.json"
            paths["event_bins_path"] = str(event_bins_path)
            cleanup_candidates.append(polymarket_dir)

        features_dir = args.runtime_root / "prediction_features"
        run(
            [
                sys.executable,
                "-m",
                "experiments.withhrrr.withhrrr_model.build_inference_features",
                "--target-date-local",
                target_date.isoformat(),
                "--station-id",
                args.station_id,
                "--label-history-path",
                str(wu_tables_dir / "labels_daily.parquet"),
                "--obs-path",
                str(wu_tables_dir / "wu_obs_intraday.parquet"),
                "--nbm-root",
                str(nbm_overnight_dir),
                "--lamp-root",
                str(lamp_overnight_dir),
                "--hrrr-root",
                str(hrrr_summary_dir),
                "--output-dir",
                str(features_dir),
            ]
            + [item for station, path in sorted(nearby_obs_paths.items()) for item in ("--nearby-obs-path", f"{station}={path}")]
        )

        normalized_path = features_dir / f"target_date_local={target_date.isoformat()}" / "withhrrr.inference_features_normalized.parquet"
        paths["prediction_features_path"] = str(normalized_path)
        predict_command = [
            sys.executable,
            "-m",
            "experiments.withhrrr.withhrrr_model.predict",
            "--features-path",
            str(normalized_path),
            "--output-dir",
            str(args.prediction_output_dir),
            "--target-date-local",
            target_date.isoformat(),
            "--station-id",
            args.station_id,
        ]
        for label in args.event_bin:
            predict_command.extend(["--event-bin", label])
        if event_bins_path is not None:
            predict_command.extend(["--event-bins-path", str(event_bins_path)])
        if args.max_so_far_f is not None:
            predict_command.extend(["--max-so-far-f", str(args.max_so_far_f)])
        run(predict_command)
        prediction_path = args.prediction_output_dir / f"prediction_{args.station_id}_{target_date.isoformat()}.json"
        paths["prediction_path"] = str(prediction_path)
        cleanup_candidates.append(features_dir)
        if args.keep_artifacts:
            paths["artifacts_cleanup"] = "skipped_keep_artifacts"
        else:
            paths["artifacts_cleanup_deleted"] = json.dumps(cleanup_runtime_artifacts(args.runtime_root, cleanup_candidates))
        write_run_manifest(runtime_root=args.runtime_root, target_date=target_date, station_id=args.station_id, status="ok", message=None, paths=paths)
        return 0
    except SystemExit as exc:
        message = str(exc)
        if message:
            print(message, file=sys.stderr, flush=True)
        write_run_manifest(runtime_root=args.runtime_root, target_date=target_date, station_id=args.station_id, status="failed", message=message or None, paths=paths)
        return int(exc.code) if isinstance(exc.code, int) else 1


if __name__ == "__main__":
    raise SystemExit(main())
