#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


GAMMA_BASE = "https://gamma-api.polymarket.com"
AWC_METAR_URL = "https://aviationweather.gov/api/data/metar"
NY_TZ = ZoneInfo("America/New_York")
THAI_TZ = ZoneInfo("Asia/Bangkok")
UTC = dt.timezone.utc


@dataclass(frozen=True)
class TempBin:
    name: str
    market_slug: str
    question: str
    low_f: int | None
    high_f: int | None
    yes_token_id: str
    no_token_id: str
    tick_size: str
    order_min_size: float


@dataclass(frozen=True)
class MetarObs:
    obs_time_utc: dt.datetime
    receipt_time_utc: dt.datetime | None
    raw: str
    temp_c: float
    temp_f: float
    rounded_temp_f: int
    max_temp_f: float | None = None
    rounded_max_temp_f: int | None = None


@dataclass(frozen=True)
class PositionView:
    bin: TempBin
    shares: float
    best_bid: float | None
    bid_depth_allowed: float
    planned_sell_shares: float
    expected_avg_price: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor KLGA METAR Tmax and market-sell YES positions in Polymarket "
            "temperature bins that are already below max-so-far."
        )
    )
    parser.add_argument("--target-date", required=True, help="NYC/KLGA local market date, YYYY-MM-DD.")
    parser.add_argument("--event-slug", help="Override Polymarket event slug. Defaults from --target-date.")
    parser.add_argument("--station", default="KLGA")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--init-hours", type=float, default=18.0, help="Recent METAR lookback used to initialize max-so-far.")
    parser.add_argument("--audit-log", type=Path, default=Path("tools/trading/data/runtime/metar_tmax_guard.jsonl"))
    parser.add_argument("--state-path", type=Path, default=Path("tools/trading/data/runtime/metar_tmax_guard_state.json"))
    parser.add_argument("--once", action="store_true", help="Run one poll cycle and exit.")
    parser.add_argument("--live", action="store_true", help="Actually submit market sell orders. Default is dry-run.")
    parser.add_argument("--max-sell-shares", type=float, default=None, help="Optional cap per token sell action.")
    parser.add_argument("--min-sell-shares", type=float, default=1.0, help="Skip sell actions below this share count.")
    parser.add_argument("--max-slippage", type=float, default=0.05, help="Sell only into bids no worse than best_bid minus this amount.")
    parser.add_argument("--min-avg-price", type=float, default=0.10, help="Skip/trim market sell if expected average price is below this.")
    parser.add_argument("--share-decimals", type=int, default=6, help="Conditional-token balance decimal places.")
    parser.add_argument("--chain-id", type=int, default=137)
    parser.add_argument("--clob-host", default="https://clob.polymarket.com")
    parser.add_argument("--user-agent", default="modelexp-metar-tmax-guard/0.1")
    parser.add_argument("--print-bins", action="store_true")
    parser.add_argument("--no-dashboard", action="store_true", help="Use line logs only; do not redraw terminal dashboard.")
    return parser.parse_args()


def request_json(url: str, *, user_agent: str, timeout: float = 20.0) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if response.status == 204:
            return []
        return json.loads(response.read().decode("utf-8", errors="replace"))


def event_slug_for_date(target_date: dt.date) -> str:
    month = target_date.strftime("%B").lower()
    return f"highest-temperature-in-nyc-on-{month}-{target_date.day}-{target_date.year}"


def parse_jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(item) for item in loaded]
        except json.JSONDecodeError:
            pass
    return []


def parse_bin(question: str) -> tuple[int | None, int | None] | None:
    q = question.replace("°", "")
    match = re.search(r"between\s+(-?\d+)\s*-\s*(-?\d+)\s*F", q, flags=re.I)
    if match:
        return int(match.group(1)), int(match.group(2))
    match = re.search(r"(-?\d+)\s*F\s+or\s+below", q, flags=re.I)
    if match:
        return None, int(match.group(1))
    match = re.search(r"(-?\d+)\s*F\s+or\s+higher", q, flags=re.I)
    if match:
        return int(match.group(1)), None
    return None


def fetch_event_bins(*, target_date: dt.date, event_slug: str | None, user_agent: str) -> tuple[dict[str, Any], list[TempBin]]:
    slug = event_slug or event_slug_for_date(target_date)
    url = f"{GAMMA_BASE}/events?slug={urllib.parse.quote(slug)}"
    events = request_json(url, user_agent=user_agent)
    if not events:
        raise RuntimeError(f"No Polymarket event found for slug={slug!r}")
    event = events[0]
    bins: list[TempBin] = []
    for market in event.get("markets", []):
        question = str(market.get("question") or "")
        parsed = parse_bin(question)
        if parsed is None:
            continue
        token_ids = parse_jsonish_list(market.get("clobTokenIds"))
        outcomes = parse_jsonish_list(market.get("outcomes"))
        if len(token_ids) < 2 or outcomes[:2] != ["Yes", "No"]:
            continue
        low, high = parsed
        name = (
            f"{high}F-or-below" if low is None else f"{low}F-or-higher" if high is None else f"{low}-{high}F"
        )
        bins.append(
            TempBin(
                name=name,
                market_slug=str(market.get("slug") or ""),
                question=question,
                low_f=low,
                high_f=high,
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1],
                tick_size=str(market.get("orderPriceMinTickSize") or "0.01"),
                order_min_size=float(market.get("orderMinSize") or 0.0),
            )
        )
    if not bins:
        raise RuntimeError(f"No temperature-bin markets with CLOB token IDs found for event slug={slug!r}")
    return event, sorted(bins, key=lambda item: (-10_000 if item.low_f is None else item.low_f))


def c_to_f(value: float) -> float:
    return value * 9.0 / 5.0 + 32.0


def round_f(value: float) -> int:
    # Polymarket settlement rules state whole degrees Fahrenheit. Use explicit
    # half-up rounding so edge cases are deterministic and auditable.
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def parse_awc_time(value: Any) -> dt.datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), UTC)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        try:
            parsed = dt.datetime.fromisoformat(cleaned)
            return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def temp_c_from_metar(row: dict[str, Any]) -> float | None:
    raw = str(row.get("rawOb") or "")
    # METAR RMK T snTTTsnDDD reports temperature/dewpoint in tenths C.
    match = re.search(r"\bT([01])(\d{3})([01])(\d{3})\b", raw)
    if match:
        sign = -1 if match.group(1) == "1" else 1
        return sign * (int(match.group(2)) / 10.0)
    value = row.get("temp")
    if value is None:
        return None
    return float(value)


def fetch_metars(*, station: str, hours: float, user_agent: str) -> list[MetarObs]:
    params = urllib.parse.urlencode({"ids": station.upper(), "format": "json", "hours": f"{hours:g}"})
    rows = request_json(f"{AWC_METAR_URL}?{params}", user_agent=user_agent)
    observations: list[MetarObs] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        temp_c = temp_c_from_metar(row)
        obs_time = parse_awc_time(row.get("obsTime"))
        if temp_c is None or obs_time is None:
            continue
        temp_f = c_to_f(temp_c)
        max_temp_f = None
        rounded_max_temp_f = None
        if row.get("maxT") is not None:
            max_temp_f = c_to_f(float(row["maxT"]))
            rounded_max_temp_f = round_f(max_temp_f)
        observations.append(
            MetarObs(
                obs_time_utc=obs_time,
                receipt_time_utc=parse_awc_time(row.get("receiptTime")),
                raw=str(row.get("rawOb") or ""),
                temp_c=temp_c,
                temp_f=temp_f,
                rounded_temp_f=round_f(temp_f),
                max_temp_f=max_temp_f,
                rounded_max_temp_f=rounded_max_temp_f,
            )
        )
    return sorted(observations, key=lambda obs: obs.obs_time_utc)


def impossible_bins(bins: list[TempBin], *, max_so_far_f: int) -> list[TempBin]:
    return [item for item in bins if item.high_f is not None and item.high_f < max_so_far_f]


def metar_day_max_candidates(obs: MetarObs, *, target_date: dt.date) -> list[int]:
    values = [obs.rounded_temp_f]
    # AWC maxT is a 6-hour max. Do not use it for early local-day METARs
    # where the 6-hour window reaches into the prior settlement day.
    if obs.rounded_max_temp_f is not None:
        obs_local = obs.obs_time_utc.astimezone(NY_TZ)
        if obs_local - dt.timedelta(hours=6) >= dt.datetime.combine(target_date, dt.time.min, tzinfo=NY_TZ):
            values.append(obs.rounded_max_temp_f)
    return values


def append_audit(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"already_sold_tokens": [], "seen_obs_times": [], "last_max_so_far_f": None}
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            data.setdefault("already_sold_tokens", [])
            data.setdefault("seen_obs_times", [])
            data.setdefault("last_max_so_far_f", None)
            return data
    except json.JSONDecodeError:
        pass
    backup = path.with_suffix(path.suffix + ".corrupt")
    path.replace(backup)
    return {"already_sold_tokens": [], "seen_obs_times": [], "last_max_so_far_f": None, "corrupt_backup": str(backup)}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def initialize_state_for_run(
    state: dict[str, Any], *, target_date: dt.date, event_slug: str | None, live: bool
) -> dict[str, Any]:
    expected_mode = "live" if live else "dry_run"
    if (
        state.get("target_date") != target_date.isoformat()
        or state.get("event_slug") != event_slug
        or state.get("mode") != expected_mode
    ):
        return {"already_sold_tokens": [], "seen_obs_times": [], "last_max_so_far_f": None}
    return state


def fmt_time(timestamp: dt.datetime | None, tz: ZoneInfo) -> str:
    if timestamp is None:
        return "unknown"
    return timestamp.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def build_clob_client(args: argparse.Namespace):
    try:
        from py_clob_client_v2 import ApiCreds, ClobClient
    except ImportError as exc:
        raise RuntimeError("Install py-clob-client-v2 in the runtime environment before using --live.") from exc
    creds = ApiCreds(
        api_key=os.environ["CLOB_API_KEY"],
        api_secret=os.environ["CLOB_SECRET"],
        api_passphrase=os.environ["CLOB_PASS_PHRASE"],
    )
    return ClobClient(host=args.clob_host, chain_id=args.chain_id, key=os.environ["PK"], creds=creds)


def conditional_balance_shares(client: Any, token_id: str, *, decimals: int) -> float:
    from py_clob_client_v2 import AssetType, BalanceAllowanceParams

    response = client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL, token_id=token_id)
    )
    raw_value = response.get("balance", "0") if isinstance(response, dict) else getattr(response, "balance", "0")
    raw_balance = Decimal(str(raw_value))
    return float(raw_balance / (Decimal(10) ** decimals))


def orderbook_bids(client: Any, token_id: str) -> list[tuple[float, float]]:
    book = client.get_order_book(token_id)
    bids = getattr(book, "bids", None)
    if bids is None and isinstance(book, dict):
        bids = book.get("bids")
    parsed: list[tuple[float, float]] = []
    for bid in bids or []:
        price = getattr(bid, "price", None)
        size = getattr(bid, "size", None)
        if isinstance(bid, dict):
            price = bid.get("price", price)
            size = bid.get("size", size)
        if price is None or size is None:
            continue
        parsed.append((float(price), float(size)))
    return sorted(parsed, key=lambda item: item[0], reverse=True)


def plan_orderbook_sell(
    bids: list[tuple[float, float]],
    *,
    desired_shares: float,
    max_slippage: float,
    min_avg_price: float,
) -> tuple[float, float | None, float | None, float]:
    if desired_shares <= 0 or not bids:
        return 0.0, None, None, 0.0
    best_bid = bids[0][0]
    min_price = max(0.0, best_bid - max_slippage)
    remaining = desired_shares
    filled = 0.0
    notional = 0.0
    for price, size in bids:
        if price < min_price:
            break
        take = min(size, remaining)
        if take <= 0:
            continue
        filled += take
        notional += take * price
        remaining -= take
        if remaining <= 1e-9:
            break
    if filled <= 0:
        return 0.0, best_bid, None, 0.0
    avg = notional / filled
    if avg < min_avg_price:
        return 0.0, best_bid, avg, filled
    return filled, best_bid, avg, filled


def market_sell(client: Any, token_id: str, *, shares: float, tick_size: str) -> Any:
    from py_clob_client_v2 import MarketOrderArgs, OrderType, PartialCreateOrderOptions, Side

    return client.create_and_post_market_order(
        order_args=MarketOrderArgs(
            token_id=token_id,
            amount=shares,
            side=Side.SELL,
            order_type=OrderType.FAK,
        ),
        options=PartialCreateOrderOptions(tick_size=tick_size),
        order_type=OrderType.FAK,
    )


def discover_positions(
    *,
    args: argparse.Namespace,
    bins: list[TempBin],
    client: Any | None,
    max_so_far_f: int | None,
) -> list[PositionView]:
    views: list[PositionView] = []
    for temp_bin in bins:
        shares = 0.0
        bids: list[tuple[float, float]] = []
        if args.live and client is not None:
            try:
                shares = conditional_balance_shares(client, temp_bin.yes_token_id, decimals=args.share_decimals)
            except Exception as exc:
                append_audit(args.audit_log, {
                    "ts_utc": dt.datetime.now(UTC).isoformat(),
                    "action": "balance_fetch_error",
                    "bin": temp_bin.name,
                    "yes_token_id": temp_bin.yes_token_id,
                    "error": str(exc),
                })
                shares = 0.0
            try:
                bids = orderbook_bids(client, temp_bin.yes_token_id)
            except Exception as exc:
                append_audit(args.audit_log, {
                    "ts_utc": dt.datetime.now(UTC).isoformat(),
                    "action": "orderbook_fetch_error",
                    "bin": temp_bin.name,
                    "yes_token_id": temp_bin.yes_token_id,
                    "error": str(exc),
                })
        else:
            shares = args.max_sell_shares if args.max_sell_shares is not None else 0.0
        desired = shares if args.max_sell_shares is None else min(shares, args.max_sell_shares)
        planned, best_bid, avg, allowed_depth = plan_orderbook_sell(
            bids,
            desired_shares=desired,
            max_slippage=args.max_slippage,
            min_avg_price=args.min_avg_price,
        )
        views.append(
            PositionView(
                bin=temp_bin,
                shares=shares,
                best_bid=best_bid,
                bid_depth_allowed=allowed_depth,
                planned_sell_shares=planned,
                expected_avg_price=avg,
            )
        )
    return views


def render_dashboard(
    *,
    args: argparse.Namespace,
    event: dict[str, Any],
    latest: MetarObs,
    max_so_far_f: int,
    positions: list[PositionView],
    impossible: list[TempBin],
    already_sold_tokens: set[str],
) -> None:
    if args.no_dashboard:
        return
    now = dt.datetime.now(UTC)
    latency_receipt = None if latest.receipt_time_utc is None else (now - latest.receipt_time_utc).total_seconds()
    latency_obs = (now - latest.obs_time_utc).total_seconds()
    impossible_ids = {item.yes_token_id for item in impossible}
    print("\033[2J\033[H", end="")
    print("KLGA METAR Tmax Position Guard")
    print(f"mode={'LIVE' if args.live else 'DRY-RUN'} event={event.get('slug')}")
    print(f"now_utc={fmt_time(now, UTC)}  now_thai={fmt_time(now, THAI_TZ)}")
    print(f"target_date={args.target_date} station={args.station.upper()} poll={args.poll_seconds:g}s")
    print()
    print(f"latest_metar_utc={fmt_time(latest.obs_time_utc, UTC)}")
    print(f"latest_metar_ny ={fmt_time(latest.obs_time_utc, NY_TZ)}")
    print(f"latest_metar_th ={fmt_time(latest.obs_time_utc, THAI_TZ)}")
    print(f"receipt_utc={fmt_time(latest.receipt_time_utc, UTC)} receipt_thai={fmt_time(latest.receipt_time_utc, THAI_TZ)}")
    print(f"latency_from_obs={latency_obs:.1f}s latency_from_receipt={latency_receipt:.1f}s" if latency_receipt is not None else f"latency_from_obs={latency_obs:.1f}s")
    print(f"latest_temp={latest.temp_f:.1f}F rounded={latest.rounded_temp_f}F settlement_rounding=half_up max_so_far={max_so_far_f}F")
    print(f"raw={latest.raw}")
    print()
    print(f"{'bin':<14} {'shares':>10} {'best_bid':>8} {'sell_plan':>10} {'avg':>7} {'state'}")
    for view in positions:
        state = "SOLD" if view.bin.yes_token_id in already_sold_tokens else "IMPOSSIBLE" if view.bin.yes_token_id in impossible_ids else "open"
        best_bid = "-" if view.best_bid is None else f"{view.best_bid:.3f}"
        avg = "-" if view.expected_avg_price is None else f"{view.expected_avg_price:.3f}"
        print(f"{view.bin.name:<14} {view.shares:>10.4f} {best_bid:>8} {view.planned_sell_shares:>10.4f} {avg:>7} {state}")
    print(flush=True)


def sell_impossible_positions(
    *,
    args: argparse.Namespace,
    positions: list[PositionView],
    bins_to_sell: list[TempBin],
    client: Any | None,
    already_sold_tokens: set[str],
    max_so_far_f: int,
    latest_obs: MetarObs,
) -> None:
    positions_by_token = {view.bin.yes_token_id: view for view in positions}
    for temp_bin in bins_to_sell:
        view = positions_by_token.get(temp_bin.yes_token_id)
        if view is None:
            continue
        response: Any = None
        sell_shares = view.planned_sell_shares if args.live else (
            args.max_sell_shares if args.max_sell_shares is not None else max(args.min_sell_shares, temp_bin.order_min_size)
        )
        action = {
            "ts_utc": dt.datetime.now(UTC).isoformat(),
            "ts_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
            "mode": "live" if args.live else "dry_run",
            "action": "market_sell_yes_position",
            "bin": temp_bin.name,
            "market_slug": temp_bin.market_slug,
            "yes_token_id": temp_bin.yes_token_id,
            "max_so_far_f": max_so_far_f,
            "latest_metar_obs_time_utc": latest_obs.obs_time_utc.isoformat(),
            "latest_metar_obs_time_thai": latest_obs.obs_time_utc.astimezone(THAI_TZ).isoformat(),
            "latest_metar_receipt_time_utc": None if latest_obs.receipt_time_utc is None else latest_obs.receipt_time_utc.isoformat(),
            "latest_metar_receipt_time_thai": None if latest_obs.receipt_time_utc is None else latest_obs.receipt_time_utc.astimezone(THAI_TZ).isoformat(),
            "latest_metar_raw": latest_obs.raw,
            "detected_shares": view.shares,
            "sell_shares": sell_shares,
            "best_bid": view.best_bid,
            "orderbook_allowed_depth": view.bid_depth_allowed,
            "expected_avg_price": view.expected_avg_price,
            "max_slippage": args.max_slippage,
            "min_avg_price": args.min_avg_price,
        }
        if sell_shares < args.min_sell_shares:
            action["action"] = "skip_no_liquidity_or_small_position"
            append_audit(args.audit_log, action)
            print(f"[skip] {temp_bin.name} shares={view.shares:.6f} planned={sell_shares:.6f}", flush=True)
            continue
        if args.live:
            try:
                response = market_sell(client, temp_bin.yes_token_id, shares=sell_shares, tick_size=temp_bin.tick_size)
                action["clob_response"] = response
            except Exception as exc:
                action["action"] = "market_sell_error"
                action["error"] = str(exc)
                append_audit(args.audit_log, action)
                print(f"[error] sell YES {temp_bin.name} failed: {exc}", flush=True)
                continue
        append_audit(args.audit_log, action)
        print(
            f"[{'LIVE' if args.live else 'DRY'}] sell YES {temp_bin.name} "
            f"shares={sell_shares:.6f} max_so_far={max_so_far_f}F",
            flush=True,
        )
        if args.live:
            try:
                remaining = conditional_balance_shares(client, temp_bin.yes_token_id, decimals=args.share_decimals)
                action_remaining = {
                    "ts_utc": dt.datetime.now(UTC).isoformat(),
                    "ts_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
                    "mode": "live",
                    "action": "post_sell_balance_check",
                    "bin": temp_bin.name,
                    "yes_token_id": temp_bin.yes_token_id,
                    "remaining_shares": remaining,
                }
                append_audit(args.audit_log, action_remaining)
                if remaining < args.min_sell_shares:
                    already_sold_tokens.add(temp_bin.yes_token_id)
            except Exception as exc:
                append_audit(
                    args.audit_log,
                    {
                        "ts_utc": dt.datetime.now(UTC).isoformat(),
                        "ts_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
                        "mode": "live",
                        "action": "post_sell_balance_check_error",
                        "bin": temp_bin.name,
                        "yes_token_id": temp_bin.yes_token_id,
                        "error": str(exc),
                    },
                )


def main() -> int:
    args = parse_args()
    target_date = dt.date.fromisoformat(args.target_date)
    event, bins = fetch_event_bins(target_date=target_date, event_slug=args.event_slug, user_agent=args.user_agent)
    print(f"event={event.get('slug')} title={event.get('title')}")
    if args.print_bins:
        for item in bins:
            print(f"bin={item.name} yes={item.yes_token_id} market={item.market_slug}")

    if args.live:
        missing = [name for name in ("PK", "CLOB_API_KEY", "CLOB_SECRET", "CLOB_PASS_PHRASE") if not os.environ.get(name)]
        if missing:
            raise RuntimeError(f"Missing required live trading env vars: {', '.join(missing)}")
    client = build_clob_client(args) if args.live else None
    if args.live:
        print("[live] market-sell mode enabled", flush=True)
    else:
        print("[dry-run] no orders will be submitted; pass --live to sell positions", flush=True)

    state = initialize_state_for_run(
        load_state(args.state_path), target_date=target_date, event_slug=str(event.get("slug")), live=args.live
    )
    already_sold_tokens: set[str] = set(str(value) for value in state.get("already_sold_tokens", []))
    seen_obs_times: set[str] = set(str(value) for value in state.get("seen_obs_times", []))
    max_so_far: int | None = state.get("last_max_so_far_f")
    while True:
        now_local = dt.datetime.now(UTC).astimezone(NY_TZ)
        midnight_local = dt.datetime.combine(target_date, dt.time.min, tzinfo=NY_TZ)
        hours_since_midnight = max(0.0, (now_local - midnight_local).total_seconds() / 3600.0)
        query_hours = max(args.init_hours, hours_since_midnight + 2.0)
        try:
            metars = fetch_metars(station=args.station, hours=query_hours, user_agent=args.user_agent)
        except Exception as exc:
            error_event = {
                "ts_utc": dt.datetime.now(UTC).isoformat(),
                "ts_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
                "mode": "live" if args.live else "dry_run",
                "action": "metar_fetch_error",
                "station": args.station.upper(),
                "target_date": target_date.isoformat(),
                "error": str(exc),
            }
            append_audit(args.audit_log, error_event)
            print(f"[warn] METAR fetch failed: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(args.poll_seconds)
            continue
        target_metars = [obs for obs in metars if obs.obs_time_utc.astimezone(NY_TZ).date() == target_date]
        if not target_metars:
            print("[warn] no target-day METARs returned yet", flush=True)
            if args.once:
                return 1
            time.sleep(args.poll_seconds)
            continue

        latest = target_metars[-1]
        current_max = max(max(metar_day_max_candidates(obs, target_date=target_date)) for obs in target_metars)
        max_changed = max_so_far is None or current_max > max_so_far
        max_so_far = current_max if max_so_far is None else max(max_so_far, current_max)
        obs_key = latest.obs_time_utc.isoformat()
        new_obs = obs_key not in seen_obs_times
        seen_obs_times.add(obs_key)

        status = {
            "ts_utc": dt.datetime.now(UTC).isoformat(),
            "ts_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
            "mode": "live" if args.live else "dry_run",
            "action": "poll",
            "station": args.station.upper(),
            "target_date": target_date.isoformat(),
            "latest_obs_time_utc": latest.obs_time_utc.isoformat(),
            "latest_obs_time_ny": latest.obs_time_utc.astimezone(NY_TZ).isoformat(),
            "latest_obs_time_thai": latest.obs_time_utc.astimezone(THAI_TZ).isoformat(),
            "latest_receipt_time_utc": None if latest.receipt_time_utc is None else latest.receipt_time_utc.isoformat(),
            "latest_receipt_time_thai": None if latest.receipt_time_utc is None else latest.receipt_time_utc.astimezone(THAI_TZ).isoformat(),
            "latency_seconds_from_obs": (dt.datetime.now(UTC) - latest.obs_time_utc).total_seconds(),
            "latency_seconds_from_receipt": None if latest.receipt_time_utc is None else (dt.datetime.now(UTC) - latest.receipt_time_utc).total_seconds(),
            "latest_temp_f": latest.temp_f,
            "latest_rounded_temp_f": latest.rounded_temp_f,
            "latest_max_temp_f": latest.max_temp_f,
            "latest_rounded_max_temp_f": latest.rounded_max_temp_f,
            "settlement_rounding_policy": "fahrenheit_half_up_to_whole_degree",
            "max_so_far_f": max_so_far,
            "new_obs": new_obs,
            "metar_query_hours": query_hours,
            "raw": latest.raw,
        }

        bins_to_sell = impossible_bins(bins, max_so_far_f=max_so_far)
        positions = discover_positions(args=args, bins=bins, client=client, max_so_far_f=max_so_far)
        status["positions"] = [
            {
                "bin": view.bin.name,
                "yes_token_id": view.bin.yes_token_id,
                "shares": view.shares,
                "best_bid": view.best_bid,
                "planned_sell_shares": view.planned_sell_shares,
                "expected_avg_price": view.expected_avg_price,
                "impossible": view.bin in bins_to_sell,
            }
            for view in positions
        ]
        append_audit(args.audit_log, status)
        if args.no_dashboard:
            print(
                f"[poll] latest={latest.obs_time_utc.isoformat()} "
                f"temp={latest.temp_f:.1f}F rounded={latest.rounded_temp_f}F max_so_far={max_so_far}F "
                f"{'NEW' if new_obs else 'same'}",
                flush=True,
            )
        else:
            render_dashboard(
                args=args,
                event=event,
                latest=latest,
                max_so_far_f=max_so_far,
                positions=positions,
                impossible=bins_to_sell,
                already_sold_tokens=already_sold_tokens,
            )
        if bins_to_sell and (new_obs or max_changed):
            sell_impossible_positions(
                args=args,
                positions=positions,
                bins_to_sell=bins_to_sell,
                client=client,
                already_sold_tokens=already_sold_tokens,
                max_so_far_f=max_so_far,
                latest_obs=latest,
            )
        state.update(
            {
                "updated_at_utc": dt.datetime.now(UTC).isoformat(),
                "updated_at_thai": dt.datetime.now(UTC).astimezone(THAI_TZ).isoformat(),
                "mode": "live" if args.live else "dry_run",
                "event_slug": event.get("slug"),
                "target_date": target_date.isoformat(),
                "last_obs_time_utc": latest.obs_time_utc.isoformat(),
                "last_obs_time_thai": latest.obs_time_utc.astimezone(THAI_TZ).isoformat(),
                "last_max_so_far_f": max_so_far,
                "already_sold_tokens": sorted(already_sold_tokens),
                "seen_obs_times": sorted(seen_obs_times)[-200:],
            }
        )
        if args.live:
            save_state(args.state_path, state)

        if args.once:
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stop] interrupted", file=sys.stderr)
        raise SystemExit(130)
