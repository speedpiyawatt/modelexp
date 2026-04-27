from __future__ import annotations

import contextlib
import os
import re
import select
import shutil
import sys
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, TextIO

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - non-POSIX fallback
    termios = None
    tty = None


DEFAULT_BAR_WIDTH = 24
DEFAULT_RECENT_EVENT_LIMIT = 8
DEFAULT_RENDER_INTERVAL_SECONDS = 0.2
DEFAULT_MAX_VISIBLE_WORKERS = 8
DEFAULT_FOOTER_EVENT_LIMIT = 5
DEFAULT_THROUGHPUT_SAMPLE_WINDOW_SECONDS = 300.0
DEFAULT_TRANSFER_STALE_SECONDS = 2.0
_STREAM_RENDER_LOCK = threading.Lock()
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(?:\x07|\x1b\\)")


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


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--:--"
    value = max(0, int(round(float(seconds))))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "?"
    value = float(num_bytes)
    units = ("B", "KB", "MB", "GB", "TB")
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    if unit_index == 0:
        return f"{int(value)}{units[unit_index]}"
    return f"{value:.1f}{units[unit_index]}"


def format_rate(bytes_per_second: float | None) -> str:
    if bytes_per_second is None or bytes_per_second <= 0:
        return "--"
    return f"{format_bytes(bytes_per_second)}/s"


def compact_worker_id(worker_id: str) -> str:
    token = worker_id.split("_")[-1]
    return f"w{token}" if token.isdigit() else worker_id[-10:]


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def ellipsize(text: str, width: int) -> str:
    if width <= 0:
        return ""
    plain = strip_ansi(text)
    if len(plain) <= width:
        return text
    if width == 1:
        return "…"
    return plain[: max(0, width - 1)] + "…"


def pad_cell(text: str, width: int, *, align: str = "left") -> str:
    clipped = ellipsize(text, width)
    plain_width = visible_len(clipped)
    padding = max(0, width - plain_width)
    if align == "right":
        return (" " * padding) + clipped
    if align == "center":
        left = padding // 2
        right = padding - left
        return (" " * left) + clipped + (" " * right)
    return clipped + (" " * padding)


def is_utf8_stream(stream: TextIO | None) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    return "utf" in encoding.lower() or encoding == ""


def supports_color(stream: TextIO | None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if stream not in {sys.stdout, sys.stderr}:
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def colorize(text: str, code: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    return format_duration(seconds)


def format_compact_count(label: str, value: object) -> str:
    return f"{label}:{value}"


def render_bar(
    completed: int,
    total: int,
    *,
    width: int,
    unicode_enabled: bool = True,
) -> str:
    safe_width = max(3, width)
    safe_total = max(0, total)
    safe_completed = min(max(0, completed), safe_total) if safe_total else 0
    if safe_total == 0:
        fill_ratio = 1.0
    else:
        fill_ratio = safe_completed / safe_total
    fill_count = int(round(fill_ratio * safe_width))
    full = "█" if unicode_enabled else "#"
    empty = "░" if unicode_enabled else "-"
    return full * fill_count + empty * max(0, safe_width - fill_count)


def render_pct_bar(
    pct: float | None,
    *,
    width: int,
    unicode_enabled: bool = True,
) -> str:
    safe_width = max(3, width)
    if pct is None:
        return ("·" if unicode_enabled else ".") * safe_width
    clipped = max(0.0, min(100.0, pct))
    fill_count = int(round((clipped / 100.0) * safe_width))
    full = "█" if unicode_enabled else "#"
    empty = "░" if unicode_enabled else "-"
    return full * fill_count + empty * max(0, safe_width - fill_count)


def compact_task_token(label: str) -> str:
    label = label.strip()
    nbm_match = re.match(r"(?P<date>\d{8})T\d{4}Z\s+f(?P<lead>\d{3})$", label)
    if nbm_match:
        date_token = nbm_match.group("date")
        return f"{date_token[4:6]}-{date_token[6:8]} f{nbm_match.group('lead')}"
    hrrr_match = re.match(r"(?P<date>\d{4}-\d{2}-\d{2})\s+c(?P<cycle>\d{2})\s+f(?P<lead>\d{2,3})$", label)
    if hrrr_match:
        date_token = hrrr_match.group("date")
        return f"{date_token[5:7]}-{date_token[8:10]} c{hrrr_match.group('cycle')} f{hrrr_match.group('lead')}"
    parts = label.split()
    if len(parts) >= 2 and re.match(r"f\d{2,3}", parts[-1]):
        return f"{parts[-2][-5:]} {parts[-1]}"
    return label[-18:]


def compact_status(details: str | None, phase: str | None) -> str:
    if details:
        aliases = {
            "byte_range_download": "range",
            "open_grouped_datasets": "cfgrib",
            "build_rows": "extract",
            "cleanup_scratch": "cleanup",
        }
        if details in aliases:
            return aliases[details]
        if "=" in details:
            return details.split("=", 1)[0]
        return details.replace("_", " ")
    if phase:
        return phase.replace("_", " ")
    return "-"


def format_retry_countdown(target_time: float | None) -> str:
    if target_time is None:
        return "--"
    return format_eta(max(0.0, target_time - time.perf_counter()))


def resolve_progress_mode(
    *,
    mode: str = "auto",
    stream: TextIO | None = None,
    is_tty: bool | None = None,
) -> str:
    if mode not in {"auto", "dashboard", "log"}:
        raise ValueError(f"Unknown progress mode: {mode}")
    target = stream if stream is not None else sys.stdout
    tty = bool(is_tty) if is_tty is not None else bool(getattr(target, "isatty", lambda: False)())
    if mode == "log":
        return "log"
    if mode == "dashboard":
        return "dashboard" if tty else "log"
    if tty and os.environ.get("TERM", "").lower() != "dumb":
        return "dashboard"
    return "log"


@dataclass
class TransferState:
    file_label: str | None = None
    total_bytes: int | None = None
    bytes_downloaded: int = 0
    started_at: float = field(default_factory=time.perf_counter)
    updated_at: float = field(default_factory=time.perf_counter)
    instant_bps: float | None = None
    average_bps: float | None = None
    sample_count: int = 0
    _last_sample_bytes: int = 0
    _last_sample_at: float = field(default_factory=time.perf_counter)

    def update(self, *, bytes_downloaded: int, total_bytes: int | None = None) -> None:
        now = time.perf_counter()
        delta_bytes = max(0, int(bytes_downloaded) - self._last_sample_bytes)
        delta_seconds = max(now - self._last_sample_at, 1e-9)
        self.bytes_downloaded = max(0, int(bytes_downloaded))
        if total_bytes is not None:
            self.total_bytes = max(0, int(total_bytes))
        self.updated_at = now
        self.instant_bps = (delta_bytes / delta_seconds) if delta_bytes else self.instant_bps
        total_elapsed = max(now - self.started_at, 1e-9)
        self.average_bps = self.bytes_downloaded / total_elapsed
        self.sample_count += 1
        self._last_sample_bytes = self.bytes_downloaded
        self._last_sample_at = now

    def display_rate_bps(self, *, now: float | None = None) -> float | None:
        current_time = time.perf_counter() if now is None else now
        sample_age = current_time - self._last_sample_at
        if self.instant_bps is not None and sample_age <= DEFAULT_TRANSFER_STALE_SECONDS:
            return self.instant_bps
        if sample_age <= (DEFAULT_TRANSFER_STALE_SECONDS * 2):
            return self.average_bps or self.instant_bps
        return 0.0


@dataclass
class WorkerState:
    worker_id: str
    label: str
    phase: str | None = None
    details: str | None = None
    group_id: str | None = None
    started_at: float = field(default_factory=time.perf_counter)
    updated_at: float = field(default_factory=time.perf_counter)
    active: bool = True
    transfer: TransferState | None = None
    attempt: int = 1
    max_attempts: int = 1
    retry_scheduled_at: float | None = None
    last_error_message: str | None = None
    transient_error_active: bool = False


@dataclass
class GroupState:
    group_id: str
    label: str
    total: int | None = None
    completed: int = 0
    failed: int = 0
    active: int | None = None
    status: str | None = None
    updated_at: float = field(default_factory=time.perf_counter)


@dataclass
class RecentEvent:
    level: str
    message: str
    created_at: float = field(default_factory=time.perf_counter)


@dataclass
class BatchDayState:
    target_date: str
    lifecycle_phase: str = "queued"
    selected_issue: str | None = None
    downloaded_leads: int = 0
    expected_leads: int | None = None
    retry_count: int = 0
    active_child_phase_counts: OrderedDict[str, int] = field(default_factory=OrderedDict)
    batch_status: str = "queued"
    extract_status: str = "queued"
    overnight_status: str = "--"
    status: str = "queued"
    started_at: float | None = None
    ended_at: float | None = None
    elapsed_seconds: float | None = None
    updated_at: float = field(default_factory=time.perf_counter)


@dataclass
class ProgressState:
    title: str
    unit: str
    total: int | None = None
    started_at: float = field(default_factory=time.perf_counter)
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    active: int = 0
    metrics: OrderedDict[str, str] = field(default_factory=OrderedDict)
    groups: OrderedDict[str, GroupState] = field(default_factory=OrderedDict)
    workers: OrderedDict[str, WorkerState] = field(default_factory=OrderedDict)
    recent_events: Deque[RecentEvent] = field(default_factory=lambda: deque(maxlen=DEFAULT_RECENT_EVENT_LIMIT))
    retry_events: OrderedDict[str, RecentEvent] = field(default_factory=OrderedDict)
    recovered_events: Deque[RecentEvent] = field(default_factory=lambda: deque(maxlen=3))
    throughput_samples: Deque[tuple[float, int]] = field(default_factory=lambda: deque(maxlen=64))
    eta_completed: int = 0
    final_status: str | None = None
    recovered_count: int = 0
    pause_requested: bool = False
    paused: bool = False
    pause_reason: str | None = None
    dashboard_hotkeys_enabled: bool = False
    batch_days: OrderedDict[str, BatchDayState] = field(default_factory=OrderedDict)
    batch_timing_samples: Deque[dict[str, float | None]] = field(default_factory=lambda: deque(maxlen=128))

    @property
    def completed_total(self) -> int:
        return self.completed + self.failed + self.skipped

    @property
    def queued(self) -> int | None:
        if self.total is None:
            return None
        return max(0, self.total - self.completed_total - self.active)

    @property
    def throughput_per_minute(self) -> float | None:
        if len(self.throughput_samples) >= 2:
            start_time, start_completed = self.throughput_samples[0]
            end_time, end_completed = self.throughput_samples[-1]
            delta_time = max(end_time - start_time, 1e-9)
            delta_completed = max(0, end_completed - start_completed)
            if delta_completed > 0:
                return (delta_completed / delta_time) * 60.0
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        if self.eta_completed <= 0:
            return None
        return (self.eta_completed / elapsed) * 60.0

    @property
    def overall_eta_seconds(self) -> float | None:
        if self.total is None:
            return None
        remaining = max(0, self.total - self.completed_total)
        if remaining <= 0:
            return 0.0
        rate_per_min = self.throughput_per_minute
        if rate_per_min is None or rate_per_min <= 0 or len(self.throughput_samples) < 2:
            return None
        return (remaining / rate_per_min) * 60.0

    @property
    def aggregate_transfer_bps(self) -> float | None:
        now = time.perf_counter()
        active_rates = []
        for worker in self.workers.values():
            if not worker.active or worker.transfer is None or worker.phase != "download":
                continue
            rate = worker.transfer.display_rate_bps(now=now)
            if rate is not None and rate > 0:
                active_rates.append(rate)
        if not active_rates:
            return None
        return sum(active_rates)

    @property
    def active_phase_counts(self) -> OrderedDict[str, int]:
        counts: OrderedDict[str, int] = OrderedDict()
        for worker in sorted(self.workers.values(), key=lambda item: (item.phase or "", item.updated_at)):
            if not worker.active:
                continue
            key = worker.phase or "idle"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def retrying_count(self) -> int:
        return sum(1 for worker in self.workers.values() if worker.active and worker.phase in {"retry_wait", "retrying"})


class StructuredLogRenderer:
    def __init__(self, *, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout

    def emit(self, state: ProgressState, event_name: str, payload: dict[str, object]) -> None:
        parts = ["[progress]", state.title, f"event={event_name}"]
        for key, value in payload.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        with _STREAM_RENDER_LOCK:
            self.stream.write(" ".join(parts) + "\n")
            self.stream.flush()

    def close(self, state: ProgressState) -> None:
        if state.final_status:
            self.emit(state, "run_complete", {"status": state.final_status})


class LiveTerminalDashboardRenderer:
    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        refresh_interval_seconds: float = DEFAULT_RENDER_INTERVAL_SECONDS,
        max_visible_workers: int = DEFAULT_MAX_VISIBLE_WORKERS,
        on_pause_request=None,
        enable_hotkeys: bool = True,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.refresh_interval_seconds = max(0.01, float(refresh_interval_seconds))
        self.max_visible_workers = max(1, int(max_visible_workers))
        self._last_render_at = 0.0
        self._entered_screen = False
        self._last_terminal_size: tuple[int, int] | None = None
        self._unicode = is_utf8_stream(self.stream)
        self._color = supports_color(self.stream)
        self._on_pause_request = on_pause_request
        self._enable_hotkeys = bool(enable_hotkeys)
        self._input_thread: threading.Thread | None = None
        self._input_stop = threading.Event()
        self._stdin_fd: int | None = None
        self._stdin_attrs = None
        self._hotkeys_started = False

    def emit(self, state: ProgressState, event_name: str, payload: dict[str, object], *, force: bool = False) -> None:
        self._ensure_input_listener(state)
        now = time.perf_counter()
        terminal_size = self._terminal_size()
        if not force and self._last_terminal_size == terminal_size and (now - self._last_render_at) < self.refresh_interval_seconds:
            return
        self._render(state, terminal_size=terminal_size)

    def close(self, state: ProgressState) -> None:
        self._render(state, terminal_size=self._terminal_size())
        self._stop_input_listener()
        with _STREAM_RENDER_LOCK:
            if self._entered_screen:
                self.stream.write("\x1b[?25h\x1b[?1049l")
                self._entered_screen = False
            self.stream.flush()

    def _terminal_size(self) -> tuple[int, int]:
        size = shutil.get_terminal_size(fallback=(120, 32))
        return max(80, size.columns), max(20, size.lines)

    def _hotkeys_available(self) -> bool:
        if not self._enable_hotkeys:
            return False
        if self.stream is not sys.stdout:
            return False
        if termios is None or tty is None:
            return False
        stdin = sys.stdin
        if not bool(getattr(stdin, "isatty", lambda: False)()):
            return False
        try:
            stdin.fileno()
        except Exception:
            return False
        return True

    def _ensure_input_listener(self, state: ProgressState) -> None:
        state.dashboard_hotkeys_enabled = self._hotkeys_available()
        if self._hotkeys_started or not state.dashboard_hotkeys_enabled:
            return
        try:
            self._stdin_fd = sys.stdin.fileno()
            self._stdin_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
        except Exception:
            self._stdin_fd = None
            self._stdin_attrs = None
            state.dashboard_hotkeys_enabled = False
            return
        self._input_stop.clear()
        self._input_thread = threading.Thread(target=self._input_loop, name="progress-hotkeys", daemon=True)
        self._input_thread.start()
        self._hotkeys_started = True

    def _stop_input_listener(self) -> None:
        self._input_stop.set()
        if self._input_thread is not None:
            self._input_thread.join(timeout=1.0)
            self._input_thread = None
        if self._stdin_fd is not None and self._stdin_attrs is not None and termios is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(self._stdin_fd, termios.TCSADRAIN, self._stdin_attrs)
        self._stdin_fd = None
        self._stdin_attrs = None
        self._hotkeys_started = False

    def _input_loop(self) -> None:
        while not self._input_stop.is_set():
            if self._stdin_fd is None:
                return
            try:
                ready, _, _ = select.select([self._stdin_fd], [], [], 0.1)
            except Exception:
                return
            if not ready:
                continue
            try:
                pressed = os.read(self._stdin_fd, 1)
            except Exception:
                return
            if not pressed:
                continue
            self._handle_keypress(pressed.decode(errors="ignore"))

    def _handle_keypress(self, key: str) -> None:
        if key.lower() == "p" and self._on_pause_request is not None:
            self._on_pause_request()

    def _render(self, state: ProgressState, *, terminal_size: tuple[int, int]) -> None:
        cols, rows = terminal_size
        lines = self._build_lines(state, width=cols, height=rows)
        padded_lines = lines[:rows]
        if len(padded_lines) < rows:
            padded_lines.extend([""] * (rows - len(padded_lines)))
        frame = "\n".join(self._fit_line(line, cols) for line in padded_lines)
        with _STREAM_RENDER_LOCK:
            if not self._entered_screen:
                self.stream.write("\x1b[?1049h\x1b[?25l")
                self._entered_screen = True
            self.stream.write("\x1b[H\x1b[2J")
            self.stream.write(frame)
            self.stream.write("\x1b[H")
            self.stream.flush()
        self._last_terminal_size = terminal_size
        self._last_render_at = time.perf_counter()

    def _fit_line(self, line: str, width: int) -> str:
        return pad_cell(line, width)

    def _border_chars(self) -> dict[str, str]:
        if self._unicode:
            return {
                "h": "─",
                "v": "│",
                "tl": "┌",
                "tr": "┐",
                "bl": "└",
                "br": "┘",
                "lt": "├",
                "rt": "┤",
            }
        return {"h": "-", "v": "|", "tl": "+", "tr": "+", "bl": "+", "br": "+", "lt": "+", "rt": "+"}

    def _panel(self, title: str, body_lines: list[str], *, width: int, height: int) -> list[str]:
        chars = self._border_chars()
        inner_width = max(1, width - 2)
        title_text = f" {title} " if title else ""
        top_fill = max(0, inner_width - visible_len(title_text))
        top = f"{chars['tl']}{title_text}{chars['h'] * top_fill}{chars['tr']}"
        content_rows = max(0, height - 2)
        lines = [top]
        for index in range(content_rows):
            body = body_lines[index] if index < len(body_lines) else ""
            lines.append(f"{chars['v']}{pad_cell(body, inner_width)}{chars['v']}")
        lines.append(f"{chars['bl']}{chars['h'] * inner_width}{chars['br']}")
        return lines[:height]

    def _status_badge(self, state: ProgressState) -> str:
        if state.paused or state.final_status == "paused":
            badge = "PAUSED"
            color = "33"
        elif state.pause_requested:
            badge = "PAUSING"
            color = "33"
        elif state.final_status is not None:
            badge = "DONE"
            color = "32"
        elif state.failed > 0:
            badge = "DEGRADED"
            color = "33"
        else:
            badge = "RUNNING"
            color = "36"
        return colorize(f"[{badge}]", color, enabled=self._color)

    def _summary_rows(self, state: ProgressState, *, width: int) -> list[str]:
        total_text = state.total if state.total is not None else "?"
        active_value = state.metrics.get("ActiveDays", state.active)
        queued_value = state.metrics.get("QueuedDays", state.queued if state.queued is not None else "?")
        row1_items = [
            format_compact_count("Total", total_text),
            format_compact_count("Done", state.completed),
            format_compact_count("Active", active_value),
            format_compact_count("Queued", queued_value),
            format_compact_count("Fail", state.failed),
            format_compact_count("Skip", state.skipped),
        ]
        row2_items = [
            f"Retrying {state.retrying_count}",
            f"Recovered {state.recovered_count}",
        ]
        if "cycles_total" in state.metrics and "completed_cycles" in state.metrics:
            row2_items.append(f"Cycles {state.metrics['completed_cycles']}/{state.metrics['cycles_total']}")
        if "month" in state.metrics:
            row2_items.append(f"Month {state.metrics['month']}")
        if "retained_cycles" in state.metrics:
            row2_items.append(f"Retained {state.metrics['retained_cycles']}")
        for key, value in state.metrics.items():
            if key in {"cycles_total", "completed_cycles", "month", "retained_cycles", "ActiveDays", "QueuedDays"}:
                continue
            row2_items.append(f"{key} {value}")
        throughput = state.throughput_per_minute
        if throughput is not None:
            row2_items.append(f"Rate {throughput:.1f}/{state.unit}m")
        aggregate_speed = state.aggregate_transfer_bps
        if aggregate_speed is not None:
            row2_items.append(f"Speed {format_rate(aggregate_speed)}")
        total = state.total or 0
        pct = 100.0 if total == 0 else (state.completed_total / total) * 100.0
        bar_width = max(10, width - 24)
        bar = render_pct_bar(pct, width=bar_width, unicode_enabled=self._unicode)
        row3 = f"{bar} {pct:5.1f}%  {state.completed_total}/{total_text}"
        return [
            "  ".join(row1_items),
            "  ".join(row2_items),
            row3,
        ]

    def _phase_strip(self, state: ProgressState, *, width: int) -> str:
        counts = state.active_phase_counts
        if not counts:
            return "No active workers"
        parts = [f"{phase}:{count}" for phase, count in counts.items()]
        return ellipsize(" | ".join(parts), width)

    def _worker_eta(self, worker: WorkerState) -> str:
        if worker.phase == "retry_wait":
            return format_retry_countdown(worker.retry_scheduled_at)
        transfer = worker.transfer
        if transfer is None or transfer.total_bytes is None or transfer.sample_count < 2:
            return "--"
        rate = transfer.display_rate_bps()
        if rate is None or rate <= 0:
            return "--"
        remaining = max(0, transfer.total_bytes - transfer.bytes_downloaded)
        return format_eta(remaining / rate if remaining > 0 else 0.0)

    def _worker_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        active_workers = [worker for worker in state.workers.values() if worker.active]
        active_workers.sort(
            key=lambda worker: (
                worker.started_at,
                worker.worker_id,
            )
        )
        header = "ID   Task         Try   Phase      Progress          Speed     Elapsed  ETA      Status"
        lines = [pad_cell(header, width)]
        visible_capacity = max(1, height - 1)
        visible_workers = active_workers[:visible_capacity]
        for worker in visible_workers:
            progress_pct = None
            speed = "--"
            if worker.transfer is not None:
                speed = format_rate(worker.transfer.display_rate_bps())
                if worker.transfer.total_bytes:
                    progress_pct = (worker.transfer.bytes_downloaded / worker.transfer.total_bytes) * 100.0
            if worker.phase == "retry_wait":
                speed = "--"
            columns = [
                pad_cell(compact_worker_id(worker.worker_id), 4),
                pad_cell(compact_task_token(worker.label), 12),
                pad_cell(f"a{worker.attempt}/{worker.max_attempts}", 5, align="right"),
                pad_cell((worker.phase or "-")[:10], 10),
                pad_cell(render_pct_bar(progress_pct, width=16, unicode_enabled=self._unicode), 16),
                pad_cell(speed, 9, align="right"),
                pad_cell(format_duration(time.perf_counter() - worker.started_at), 8, align="right"),
                pad_cell(self._worker_eta(worker), 8, align="right"),
                pad_cell(compact_status(worker.details, worker.phase), max(0, width - 80)),
            ]
            lines.append(pad_cell(" ".join(columns).rstrip(), width))
        hidden = max(0, len(active_workers) - len(visible_workers))
        if hidden:
            lines[-1] = pad_cell(f"+{hidden} more active", width)
        return lines[:height]

    def _group_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        groups = list(state.groups.values())
        groups.sort(key=lambda group: (-group.updated_at, group.label))
        header = "ID   Label         Done/Total  Fail  Status"
        lines = [pad_cell(header, width)]
        visible_capacity = max(1, height - 1)
        visible_groups = groups[:visible_capacity]
        for group in visible_groups:
            done_total = (
                f"{group.completed}/{group.total}"
                if group.total is not None
                else str(group.completed)
            )
            columns = [
                pad_cell(compact_task_token(group.group_id), 4),
                pad_cell(group.label, 12),
                pad_cell(done_total, 10, align="right"),
                pad_cell(str(group.failed), 5, align="right"),
                pad_cell(group.status or "-", max(0, width - 39)),
            ]
            lines.append(pad_cell(" ".join(columns).rstrip(), width))
        hidden = max(0, len(groups) - len(visible_groups))
        if hidden:
            lines[-1] = pad_cell(f"+{hidden} more groups", width)
        return lines[:height]

    def _footer_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        lines = []
        retry_workers = [
            worker
            for worker in state.workers.values()
            if worker.transient_error_active and worker.active and worker.phase in {"retry_wait", "retrying"}
        ]
        retry_workers.sort(key=lambda item: item.updated_at, reverse=True)
        for worker in retry_workers:
            prefix = colorize("[RETRY]", "36", enabled=self._color)
            attempt_text = f"a{worker.attempt}/{worker.max_attempts}"
            phase_text = "retrying now" if worker.phase == "retrying" else f"retry in {self._worker_eta(worker)}"
            error_text = worker.details or "transient_error"
            lines.append(ellipsize(f"{prefix} {worker.label} {attempt_text} {phase_text} ({error_text})", width))
        relevant = [event for event in list(state.recent_events) if event.level in {"warn", "error"}]
        for event in relevant:
            badge = "ERR" if event.level == "error" else "WARN"
            code = "31" if event.level == "error" else "33"
            prefix = colorize(f"[{badge}]", code, enabled=self._color)
            lines.append(ellipsize(f"{prefix} {event.message}", width))
        for event in list(state.recovered_events):
            prefix = colorize("[OK]", "32", enabled=self._color)
            lines.append(ellipsize(f"{prefix} {event.message}", width))
        if not lines and len(lines) < height:
            info_events = [event for event in list(state.recent_events) if event.level == "info"]
            info_capacity = max(0, height - len(lines))
            for event in info_events[-info_capacity:]:
                prefix = colorize("[INFO]", "36", enabled=self._color)
                lines.append(ellipsize(f"{prefix} {event.message}", width))
        if not lines:
            return ["No warnings or errors"]
        return lines[-height:]

    def _build_lines(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        now = time.perf_counter()
        elapsed = now - state.started_at
        header_left = state.title
        header_right = "  ".join(
            [
                f"elapsed {format_duration(elapsed)}",
                f"eta {format_eta(state.overall_eta_seconds)}",
                (
                    f"rate {state.throughput_per_minute:.1f}/{state.unit}m"
                    if state.throughput_per_minute is not None
                    else "rate --"
                ),
                (
                    f"speed {format_rate(state.aggregate_transfer_bps)}"
                    if state.aggregate_transfer_bps is not None
                    else "speed --"
                ),
                self._status_badge(state),
            ]
        )
        header_line = f"{header_left}  {header_right}"
        phase_text = self._phase_strip(state, width=max(10, width - 4))
        if state.paused:
            phase_line = ellipsize(f"{phase_text}  |  Paused: safe to exit", max(10, width - 4))
        elif state.pause_requested:
            phase_line = ellipsize(f"{phase_text}  |  Pause requested: draining admitted work", max(10, width - 4))
        elif state.dashboard_hotkeys_enabled:
            phase_line = ellipsize(f"{phase_text}  |  Press p to pause after drain", max(10, width - 4))
        else:
            phase_line = phase_text

        header_panel = self._panel("Overview", [header_line, phase_line], width=width, height=4)
        summary_panel = self._panel("Summary", self._summary_rows(state, width=max(10, width - 4)), width=width, height=5)

        remaining_height = max(8, height - len(header_panel) - len(summary_panel))
        footer_height = min(DEFAULT_FOOTER_EVENT_LIMIT + 2, max(4, remaining_height // 3))
        workers_height = max(5, remaining_height - footer_height)
        group_panel: list[str] = []
        group_count = len(state.groups)
        if group_count:
            groups_height = min(max(4, min(group_count + 2, 8)), max(4, remaining_height // 4))
            workers_height = max(5, remaining_height - footer_height - groups_height)
            group_panel = self._panel(
                f"Groups {group_count}",
                self._group_rows(state, width=max(10, width - 4), height=max(1, groups_height - 2)),
                width=width,
                height=groups_height,
            )
        else:
            workers_height = max(5, remaining_height - footer_height)
        active_worker_count = sum(1 for worker in state.workers.values() if worker.active)
        workers_panel = self._panel(
            f"Workers {active_worker_count}",
            self._worker_rows(state, width=max(10, width - 4), height=max(1, workers_height - 2)),
            width=width,
            height=workers_height,
        )
        footer_panel = self._panel(
            "Alerts",
            self._footer_rows(state, width=max(10, width - 4), height=max(1, footer_height - 2)),
            width=width,
            height=footer_height,
        )
        return header_panel + summary_panel + group_panel + workers_panel + footer_panel


class BatchDashboardRenderer(LiveTerminalDashboardRenderer):
    def _worker_profile(self, state: ProgressState) -> str:
        days = state.metrics.get("PlanDays") or state.metrics.get("DayWorkers") or "?"
        lead = state.metrics.get("PlanLead") or state.metrics.get("lead_workers") or "?"
        download = state.metrics.get("PlanDownload") or state.metrics.get("download_workers") or lead
        reduce_workers = state.metrics.get("PlanReduce") or state.metrics.get("reduce_workers") or "?"
        extract_workers = state.metrics.get("PlanExtract") or state.metrics.get("extract_workers") or "?"
        return f"days={days} dl={download} reduce={reduce_workers} extract={extract_workers}"

    def _percentile(self, values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        fraction = position - lower
        return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)

    def _timing_value(self, state: ProgressState, key: str, *, op: str = "median") -> float | None:
        values = [
            float(sample[key])
            for sample in state.batch_timing_samples
            if sample.get(key) is not None
        ]
        if not values:
            return None
        if op == "p95":
            return self._percentile(values, 0.95)
        return self._percentile(values, 0.50)

    def _batch_eta_seconds(self, state: ProgressState) -> float | None:
        if state.total is None:
            return None
        remaining = max(0, state.total - state.completed_total)
        if remaining <= 0:
            return 0.0
        median_day_seconds = self._timing_value(state, "full_day_seconds")
        if median_day_seconds is None or median_day_seconds <= 0:
            return state.overall_eta_seconds
        worker_value = state.metrics.get("PlanDays") or state.metrics.get("DayWorkers") or "1"
        try:
            effective_workers = max(1, int(float(worker_value)))
        except (TypeError, ValueError):
            effective_workers = 1
        return (remaining * median_day_seconds) / effective_workers

    def _stage_medians(self, state: ProgressState) -> dict[str, float | None]:
        return {
            "day": self._timing_value(state, "full_day_seconds"),
            "day_p95": self._timing_value(state, "full_day_seconds", op="p95"),
            "raw": self._timing_value(state, "raw_build_seconds"),
            "cleanup": self._timing_value(state, "cleanup_seconds"),
            "crop": self._timing_value(state, "timing_crop_seconds_median"),
            "cfgrib": self._timing_value(state, "timing_cfgrib_open_seconds_median"),
            "rows": self._timing_value(state, "timing_row_metric_seconds_median"),
            "prov": self._timing_value(state, "timing_row_provenance_seconds_median"),
        }

    def _summary_rows(self, state: ProgressState, *, width: int) -> list[str]:
        total_text = state.total if state.total is not None else "?"
        active_value = state.metrics.get("ActiveDays", state.active)
        queued_value = state.metrics.get("QueuedDays", state.queued if state.queued is not None else "?")
        row1_items = [
            f"done {state.completed}/{total_text}",
            f"active {active_value}",
            f"queued {queued_value}",
            f"failed {state.failed}",
            f"skipped {state.skipped}",
        ]
        throughput = state.throughput_per_minute
        retry_total = sum(day.retry_count for day in state.batch_days.values())
        row2_items = [
            f"rate {throughput:.2f} days/min" if throughput is not None else "rate --",
            f"retries {retry_total}",
            f"recovered {state.recovered_count}",
        ]
        if "FreeGB" in state.metrics:
            row2_items.append(f"disk {state.metrics['FreeGB']}GB")
        mode_items = []
        for key in ("SelectionMode", "MetricProfile", "Provenance", "BatchMode", "Adaptive"):
            value = state.metrics.get(key)
            if value is not None:
                label = {
                    "SelectionMode": "selection",
                    "MetricProfile": "metric",
                    "Provenance": "provenance",
                    "BatchMode": "mode",
                    "Adaptive": "adaptive",
                }[key]
                mode_items.append(f"{label}={value}")
        if not mode_items:
            mode_items.append("mode=cycle")
        return [
            ellipsize("  ".join(row1_items), width),
            ellipsize("  ".join(row2_items), width),
            ellipsize("  ".join(mode_items), width),
        ]

    def _ordered_days(self, state: ProgressState) -> list[BatchDayState]:
        days = list(state.batch_days.values())
        active = [day for day in days if day.status not in {"complete", "failed", "skipped"} and day.lifecycle_phase != "queued"]
        complete = [day for day in days if day.status in {"complete", "failed", "skipped"}]
        queued = [day for day in days if day.lifecycle_phase == "queued" and day.status == "queued"]
        active.sort(key=lambda item: (-item.updated_at, item.target_date))
        complete.sort(key=lambda item: (-(item.ended_at or item.updated_at), item.target_date))
        queued.sort(key=lambda item: item.target_date)
        return active + complete + queued

    def _download_text(self, day: BatchDayState) -> str:
        if day.expected_leads is None:
            return "--"
        return f"{day.downloaded_leads}/{day.expected_leads}"

    def _day_elapsed(self, day: BatchDayState) -> str:
        if day.elapsed_seconds is not None:
            return format_duration(day.elapsed_seconds)
        if day.started_at is not None:
            return format_duration(time.perf_counter() - day.started_at)
        return "--"

    def _day_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        status_width = max(8, width - 72)
        widths = [10, 12, 8, 8, 8, 9, status_width, 8]
        header_parts = [
            pad_cell("Date", widths[0]),
            pad_cell("Issue", widths[1]),
            pad_cell("Download", widths[2], align="right"),
            pad_cell("Batch", widths[3]),
            pad_cell("Extract", widths[4]),
            pad_cell("Overnight", widths[5]),
            pad_cell("Status", widths[6]),
            pad_cell("Time", widths[7], align="right"),
        ]
        lines = [pad_cell(" ".join(header_parts), width)]
        visible_capacity = max(1, height - 1)
        for day in self._ordered_days(state)[:visible_capacity]:
            issue = day.selected_issue or "--"
            if re.match(r"\d{8}T\d{4}Z$", issue):
                issue = f"{issue[4:8]} {issue[9:13]}Z"
            parts = [
                pad_cell(day.target_date, widths[0]),
                pad_cell(issue, widths[1]),
                pad_cell(self._download_text(day), widths[2], align="right"),
                pad_cell(day.batch_status or "--", widths[3]),
                pad_cell(day.extract_status or "--", widths[4]),
                pad_cell(day.overnight_status or "--", widths[5]),
                pad_cell(day.status or day.lifecycle_phase, widths[6]),
                pad_cell(self._day_elapsed(day), widths[7], align="right"),
            ]
            lines.append(pad_cell(" ".join(parts).rstrip(), width))
        hidden = max(0, len(state.batch_days) - (len(lines) - 1))
        if hidden:
            lines[-1] = pad_cell(f"+{hidden} more days", width)
        return lines[:height]

    def _utilization_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        active_days = int(state.metrics.get("ActiveDays", "0") or 0)
        day_workers = state.metrics.get("PlanDays") or state.metrics.get("DayWorkers") or "?"
        active_download = 0
        active_reduce = 0
        active_extract = 0
        for worker in state.workers.values():
            if not worker.active:
                continue
            worker_stage = self._batch_worker_stage(worker)
            if worker_stage == "download":
                active_download += 1
            elif worker_stage == "reduce":
                active_reduce += 1
            elif worker_stage == "extract":
                active_extract += 1
        rows = [
            f"active days {active_days}/{day_workers}",
            f"downloads {active_download}  crops {active_reduce}  extracts {active_extract}",
        ]
        timings = self._stage_medians(state)
        rows.append(
            "day med {med} p95 {p95}".format(
                med=format_duration(timings["day"]),
                p95=format_duration(timings["day_p95"]),
            )
        )
        stage_parts = [
            f"raw {format_duration(timings['raw'])}",
            f"cleanup {format_duration(timings['cleanup'])}",
            f"crop {format_duration(timings['crop'])}",
            f"cfgrib {format_duration(timings['cfgrib'])}",
            f"rows {format_duration(timings['rows'])}",
        ]
        if timings["prov"] is not None:
            stage_parts.append(f"prov {format_duration(timings['prov'])}")
        rows.append(ellipsize("  ".join(stage_parts), width))
        if state.final_status is not None:
            ok = state.completed
            failed = state.failed
            skipped = state.skipped
            slowest = max(
                ((name, value) for name, value in timings.items() if value is not None and name != "day_p95"),
                key=lambda item: item[1],
                default=None,
            )
            final = f"final ok={ok} failed={failed} skipped={skipped}"
            if slowest is not None:
                final += f" slowest={slowest[0]} {format_duration(slowest[1])}"
            rows.append(final)
        return [ellipsize(row, width) for row in rows[:height]]

    def _batch_worker_stage(self, worker: WorkerState) -> str | None:
        worker_name = worker.worker_id.rsplit("/", 1)[-1]
        if worker_name.startswith("download_"):
            return "download"
        if worker_name.startswith("reduce_"):
            return "reduce"
        if worker_name.startswith("extract_"):
            return "extract"
        phase = worker.phase or ""
        if phase in {"download", "download_wait", "idx", "parse"}:
            return "download"
        if phase in {"reduce", "crop", "reduce_wait"}:
            return "reduce"
        if phase in {"extract", "open", "extract_wait", "finalize"}:
            return "extract"
        return None

    def _footer_rows(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        lines = []
        retry_workers = [
            worker
            for worker in state.workers.values()
            if worker.transient_error_active and worker.active and worker.phase in {"retry_wait", "retrying"}
        ]
        retry_workers.sort(key=lambda item: item.updated_at, reverse=True)
        for worker in retry_workers:
            prefix = colorize("[RETRY]", "36", enabled=self._color)
            attempt_text = f"a{worker.attempt}/{worker.max_attempts}"
            phase_text = "retrying now" if worker.phase == "retrying" else f"retry in {self._worker_eta(worker)}"
            lines.append(ellipsize(f"{prefix} {worker.label} {attempt_text} {phase_text} ({worker.details or 'retry'})", width))
        for event in list(state.recent_events):
            if event.level not in {"warn", "error"}:
                continue
            badge = "ERR" if event.level == "error" else "WARN"
            code = "31" if event.level == "error" else "33"
            lines.append(ellipsize(f"{colorize(f'[{badge}]', code, enabled=self._color)} {event.message}", width))
        for event in list(state.recovered_events):
            lines.append(ellipsize(f"{colorize('[OK]', '32', enabled=self._color)} {event.message}", width))
        if not lines:
            return ["No warnings, errors, or retries"]
        return lines[-height:]

    def _build_lines(self, state: ProgressState, *, width: int, height: int) -> list[str]:
        now = time.perf_counter()
        elapsed = now - state.started_at
        date_range = state.metrics.get("DateRange") or ""
        mode = state.metrics.get("BatchMode") or "cycle"
        header_left = " | ".join(part for part in [state.title, date_range, mode, self._worker_profile(state)] if part)
        header_right = "  ".join(
            [
                f"elapsed {format_duration(elapsed)}",
                f"ETA {format_eta(self._batch_eta_seconds(state))}",
                f"disk {state.metrics.get('FreeGB', '?')}GB",
                self._status_badge(state),
            ]
        )
        phase_text = self._phase_strip(state, width=max(10, width - 4))
        if state.paused:
            phase_text = ellipsize(f"{phase_text}  |  Paused: safe to exit", max(10, width - 4))
        elif state.pause_requested:
            phase_text = ellipsize(f"{phase_text}  |  Pause requested: draining admitted work", max(10, width - 4))
        elif state.dashboard_hotkeys_enabled:
            phase_text = ellipsize(f"{phase_text}  |  Press p to pause after drain", max(10, width - 4))
        header_panel = self._panel("Overview", [f"{header_left}  {header_right}", phase_text], width=width, height=4)
        summary_panel = self._panel("Summary", self._summary_rows(state, width=max(10, width - 4)), width=width, height=5)

        remaining_height = max(8, height - len(header_panel) - len(summary_panel))
        alerts_height = min(DEFAULT_FOOTER_EVENT_LIMIT + 2, max(4, remaining_height // 4))
        util_height = min(7, max(5, remaining_height // 4))
        board_height = max(5, remaining_height - alerts_height - util_height)
        board_panel = self._panel(
            f"Day Pipeline {len(state.batch_days)}",
            self._day_rows(state, width=max(10, width - 4), height=max(1, board_height - 2)),
            width=width,
            height=board_height,
        )
        util_panel = self._panel(
            "Utilization And Timing",
            self._utilization_rows(state, width=max(10, width - 4), height=max(1, util_height - 2)),
            width=width,
            height=util_height,
        )
        alerts_panel = self._panel(
            "Recent Issues",
            self._footer_rows(state, width=max(10, width - 4), height=max(1, alerts_height - 2)),
            width=width,
            height=alerts_height,
        )
        return header_panel + summary_panel + board_panel + util_panel + alerts_panel


class ProgressReporter:
    def __init__(
        self,
        title: str,
        *,
        unit: str = "task",
        total: int | None = None,
        mode: str = "auto",
        stream: TextIO | None = None,
        is_tty: bool | None = None,
        on_pause_request=None,
        enable_dashboard_hotkeys: bool = True,
        pause_control_file: str | None = None,
        dashboard_kind: str = "generic",
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.mode = resolve_progress_mode(mode=mode, stream=self.stream, is_tty=is_tty)
        self.state = ProgressState(title=title, unit=unit, total=total)
        self._lock = threading.Lock()
        self._pause_callback = on_pause_request
        self._pause_control_file = os.fspath(pause_control_file) if pause_control_file else None
        self._pause_file_thread: threading.Thread | None = None
        self._pause_file_stop = threading.Event()
        if self.mode == "dashboard" and dashboard_kind == "nbm_batch":
            self.renderer = BatchDashboardRenderer(
                stream=self.stream,
                on_pause_request=self.request_pause,
                enable_hotkeys=enable_dashboard_hotkeys,
            )
        elif self.mode == "dashboard":
            self.renderer = LiveTerminalDashboardRenderer(
                stream=self.stream,
                on_pause_request=self.request_pause,
                enable_hotkeys=enable_dashboard_hotkeys,
            )
        else:
            self.renderer = StructuredLogRenderer(stream=self.stream)
        self._start_pause_file_watcher()
        self._record_progress_sample()
        self._emit("run_start", {"mode": self.mode, "total": total}, force=True)

    def request_pause(self, *, reason: str = "operator") -> None:
        callback = None
        with self._lock:
            if self.state.paused or self.state.pause_requested:
                return
            self.state.pause_requested = True
            self.state.pause_reason = reason
            self.state.recent_events.append(RecentEvent(level="warn", message=f"pause requested ({reason})"))
            self._emit("pause_requested", {"reason": reason}, force=True)
            callback = self._pause_callback
        if callback is not None:
            callback(reason=reason)

    def mark_paused(self, *, reason: str | None = None) -> None:
        with self._lock:
            self.state.pause_requested = True
            self.state.paused = True
            if reason is not None:
                self.state.pause_reason = reason
            self.state.recent_events.append(RecentEvent(level="warn", message="paused: safe to exit"))
            self._emit("paused", {"reason": self.state.pause_reason}, force=True)

    def is_pause_requested(self) -> bool:
        with self._lock:
            return self.state.pause_requested

    def is_paused(self) -> bool:
        with self._lock:
            return self.state.paused

    def set_total(self, total: int) -> None:
        with self._lock:
            self.state.total = max(0, int(total))
            self._emit("counts", {"total": self.state.total})

    def set_metrics(self, **metrics: object) -> None:
        with self._lock:
            for key, value in metrics.items():
                if value is None:
                    self.state.metrics.pop(key, None)
                else:
                    self.state.metrics[key] = str(value)
            self._emit("summary", dict(metrics))

    def log_event(self, message: str, *, level: str = "info") -> None:
        with self._lock:
            self.state.recent_events.append(RecentEvent(level=level, message=message))
            self._emit("message", {"level": level, "message": message}, force=level == "error")

    def add_skipped(self, count: int = 1, *, message: str | None = None, affects_eta: bool = False) -> None:
        skip_count = max(0, int(count))
        if skip_count <= 0:
            return
        with self._lock:
            self.state.skipped += skip_count
            if affects_eta:
                self.state.eta_completed += skip_count
            self._record_progress_sample()
            event_message = message or f"skipped={skip_count}"
            self.state.recent_events.append(RecentEvent(level="warn", message=event_message))
            self._emit("skipped", {"count": skip_count, "message": event_message})

    def upsert_batch_day(
        self,
        target_date: str,
        *,
        lifecycle_phase: str | None = None,
        selected_issue: str | None = None,
        downloaded_leads: int | None = None,
        expected_leads: int | None = None,
        retry_count: int | None = None,
        active_child_phase_counts: OrderedDict[str, int] | dict[str, int] | None = None,
        batch_status: str | None = None,
        extract_status: str | None = None,
        overnight_status: str | None = None,
        status: str | None = None,
        started_at: float | None = None,
        ended_at: float | None = None,
        elapsed_seconds: float | None = None,
    ) -> None:
        with self._lock:
            day = self.state.batch_days.get(target_date)
            if day is None:
                day = BatchDayState(target_date=target_date)
                self.state.batch_days[target_date] = day
            if lifecycle_phase is not None:
                day.lifecycle_phase = lifecycle_phase
            if selected_issue is not None:
                day.selected_issue = selected_issue
            if downloaded_leads is not None:
                day.downloaded_leads = max(0, int(downloaded_leads))
            if expected_leads is not None:
                day.expected_leads = max(0, int(expected_leads))
            if retry_count is not None:
                day.retry_count = max(0, int(retry_count))
            if active_child_phase_counts is not None:
                day.active_child_phase_counts = OrderedDict(
                    (str(key), max(0, int(value))) for key, value in active_child_phase_counts.items()
                )
            if batch_status is not None:
                day.batch_status = batch_status
            if extract_status is not None:
                day.extract_status = extract_status
            if overnight_status is not None:
                day.overnight_status = overnight_status
            if status is not None:
                day.status = status
            if started_at is not None:
                day.started_at = started_at
            if ended_at is not None:
                day.ended_at = ended_at
            if elapsed_seconds is not None:
                day.elapsed_seconds = max(0.0, float(elapsed_seconds))
            elif day.started_at is not None and day.ended_at is not None:
                day.elapsed_seconds = max(0.0, day.ended_at - day.started_at)
            day.updated_at = time.perf_counter()
            self._emit(
                "batch_day",
                {
                    "target_date": target_date,
                    "phase": day.lifecycle_phase,
                    "issue": day.selected_issue,
                    "downloaded": day.downloaded_leads,
                    "expected": day.expected_leads,
                    "status": day.status,
                },
            )

    def record_batch_timing(self, **timings: float | None) -> None:
        with self._lock:
            self.state.batch_timing_samples.append(dict(timings))
            self._emit("batch_timing", {key: value for key, value in timings.items() if value is not None})

    def upsert_group(
        self,
        group_id: str,
        *,
        label: str | None = None,
        total: int | None = None,
        completed: int | None = None,
        failed: int | None = None,
        active: int | None = None,
        status: str | None = None,
    ) -> None:
        with self._lock:
            group = self.state.groups.get(group_id)
            if group is None:
                group = GroupState(group_id=group_id, label=label or group_id)
                self.state.groups[group_id] = group
            if label is not None:
                group.label = label
            if total is not None:
                group.total = max(0, int(total))
            if completed is not None:
                group.completed = max(0, int(completed))
            if failed is not None:
                group.failed = max(0, int(failed))
            if active is not None:
                group.active = max(0, int(active))
            if status is not None:
                group.status = status
            group.updated_at = time.perf_counter()
            self._emit(
                "group",
                {
                    "group": group_id,
                    "label": group.label,
                    "completed": group.completed,
                    "total": group.total,
                    "failed": group.failed,
                    "status": group.status,
                },
            )

    def start_worker(
        self,
        worker_id: str,
        *,
        label: str,
        phase: str | None = None,
        group_id: str | None = None,
        details: str | None = None,
    ) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                worker = WorkerState(worker_id=worker_id, label=label)
                self.state.workers[worker_id] = worker
                self.state.active += 1
            elif not worker.active:
                self.state.active += 1
                worker.started_at = time.perf_counter()
            worker.active = True
            worker.label = label
            worker.phase = phase
            worker.group_id = group_id
            worker.details = details
            worker.updated_at = time.perf_counter()
            worker.transfer = None
            worker.retry_scheduled_at = None
            self._emit(
                "worker_start",
                {"worker": worker_id, "label": label, "phase": phase, "group": group_id, "details": details},
            )

    def set_worker_attempt(self, worker_id: str, *, attempt: int, max_attempts: int) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            worker.attempt = max(1, int(attempt))
            worker.max_attempts = max(1, int(max_attempts))
            worker.updated_at = time.perf_counter()
            self._emit(
                "worker_attempt",
                {"worker": worker_id, "attempt": worker.attempt, "max_attempts": worker.max_attempts},
            )

    def update_worker(
        self,
        worker_id: str,
        *,
        phase: str | None = None,
        label: str | None = None,
        details: str | None = None,
    ) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            if label is not None:
                worker.label = label
            if phase is not None:
                worker.phase = phase
            if details is not None:
                worker.details = details
            worker.updated_at = time.perf_counter()
            self._emit(
                "worker_update",
                {"worker": worker_id, "label": worker.label, "phase": worker.phase, "details": worker.details},
            )

    def start_transfer(self, worker_id: str, *, file_label: str, total_bytes: int | None = None) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            worker.transfer = TransferState(file_label=file_label, total_bytes=total_bytes)
            self._emit(
                "transfer_start",
                {"worker": worker_id, "file": file_label, "total_bytes": total_bytes},
            )

    def update_transfer(
        self,
        worker_id: str,
        *,
        bytes_downloaded: int,
        total_bytes: int | None = None,
    ) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            if worker.transfer is None:
                worker.transfer = TransferState(total_bytes=total_bytes)
            worker.transfer.update(bytes_downloaded=bytes_downloaded, total_bytes=total_bytes)
            worker.updated_at = time.perf_counter()
            self._emit(
                "transfer_progress",
                {
                    "worker": worker_id,
                    "file": worker.transfer.file_label,
                    "bytes_downloaded": worker.transfer.bytes_downloaded,
                    "total_bytes": worker.transfer.total_bytes,
                    "instant_bps": round(worker.transfer.instant_bps or 0.0, 2),
                    "average_bps": round(worker.transfer.average_bps or 0.0, 2),
                },
            )

    def finish_transfer(self, worker_id: str) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None or worker.transfer is None:
                return
            self._emit(
                "transfer_complete",
                {
                    "worker": worker_id,
                    "file": worker.transfer.file_label,
                    "bytes_downloaded": worker.transfer.bytes_downloaded,
                    "total_bytes": worker.transfer.total_bytes,
                },
            )

    def schedule_retry(
        self,
        worker_id: str,
        *,
        attempt: int,
        max_attempts: int,
        delay_seconds: float,
        message: str,
        error_class: str,
    ) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            worker.attempt = max(1, int(attempt))
            worker.max_attempts = max(1, int(max_attempts))
            worker.phase = "retry_wait"
            worker.retry_scheduled_at = time.perf_counter() + max(0.0, float(delay_seconds))
            worker.last_error_message = message
            worker.transient_error_active = True
            worker.details = error_class
            worker.transfer = None
            self.state.retry_events[worker_id] = RecentEvent(
                level="warn",
                message=f"{worker.label} a{attempt}/{max_attempts} retry in {format_eta(delay_seconds)} ({error_class})",
            )
            self._emit(
                "retry_scheduled",
                {
                    "worker": worker_id,
                    "attempt": worker.attempt,
                    "max_attempts": worker.max_attempts,
                    "delay_seconds": round(float(delay_seconds), 2),
                    "error_class": error_class,
                    "message": message,
                },
                force=True,
            )

    def start_retry(self, worker_id: str, *, attempt: int, max_attempts: int) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            worker.attempt = max(1, int(attempt))
            worker.max_attempts = max(1, int(max_attempts))
            worker.phase = "retrying"
            worker.retry_scheduled_at = None
            worker.updated_at = time.perf_counter()
            self._emit(
                "retry_started",
                {"worker": worker_id, "attempt": worker.attempt, "max_attempts": worker.max_attempts},
                force=True,
            )

    def recover_worker(self, worker_id: str, *, message: str) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None:
                return
            if worker.transient_error_active:
                self.state.recovered_count += 1
            worker.transient_error_active = False
            worker.retry_scheduled_at = None
            worker.last_error_message = None
            self.state.retry_events.pop(worker_id, None)
            self.state.recovered_events.append(RecentEvent(level="info", message=message))
            self._emit("retry_recovered", {"worker": worker_id, "message": message}, force=True)

    def complete_worker(self, worker_id: str, *, message: str | None = None, outcome: str = "completed") -> None:
        error_message: str | None = None
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None or not worker.active:
                return
            self.state.retry_events.pop(worker_id, None)
            worker.active = False
            worker.updated_at = time.perf_counter()
            worker.retry_scheduled_at = None
            worker.transient_error_active = False
            self.state.active = max(0, self.state.active - 1)
            if outcome == "failed":
                self.state.failed += 1
                level = "error"
            elif outcome == "skipped":
                self.state.skipped += 1
                level = "warn"
            else:
                self.state.completed += 1
                level = "info"
            self.state.eta_completed += 1
            self._record_progress_sample()
            event_message = message or f"{worker.label} {outcome}"
            self.state.recent_events.append(RecentEvent(level=level, message=event_message))
            self._emit(
                "worker_complete",
                {"worker": worker_id, "label": worker.label, "outcome": outcome, "message": event_message},
                force=outcome == "failed",
            )
            if outcome == "failed":
                error_message = event_message
        if error_message is not None:
            emit_progress_message(f"[error] {error_message}", stream=sys.stderr)

    def fail_worker(self, worker_id: str, *, message: str) -> None:
        self.complete_worker(worker_id, message=message, outcome="failed")

    def retire_worker(self, worker_id: str, *, message: str | None = None) -> None:
        with self._lock:
            worker = self.state.workers.get(worker_id)
            if worker is None or not worker.active:
                return
            self.state.retry_events.pop(worker_id, None)
            worker.active = False
            worker.updated_at = time.perf_counter()
            worker.retry_scheduled_at = None
            worker.transient_error_active = False
            self.state.active = max(0, self.state.active - 1)
            self._emit(
                "worker_retire",
                {"worker": worker_id, "label": worker.label, "message": message},
            )

    def record_outcome(
        self,
        outcome: str = "completed",
        *,
        count: int = 1,
        message: str | None = None,
        affects_eta: bool = True,
    ) -> None:
        amount = max(0, int(count))
        if amount <= 0:
            return
        with self._lock:
            if outcome == "failed":
                self.state.failed += amount
                level = "error"
            elif outcome == "skipped":
                self.state.skipped += amount
                level = "warn"
            else:
                self.state.completed += amount
                level = "info"
            if affects_eta:
                self.state.eta_completed += amount
            self._record_progress_sample()
            event_message = message or f"{outcome}={amount}"
            self.state.recent_events.append(RecentEvent(level=level, message=event_message))
            self._emit(
                "outcome",
                {"outcome": outcome, "count": amount, "message": event_message},
                force=outcome == "failed",
            )
        if outcome == "failed":
            emit_progress_message(f"[error] {event_message}", stream=sys.stderr)

    def refresh(self, *, force: bool = False) -> None:
        if not isinstance(self.renderer, LiveTerminalDashboardRenderer):
            return
        with self._lock:
            self._emit("refresh", {}, force=force)

    def close(self, *, status: str | None = None) -> None:
        with self._lock:
            self.state.final_status = status or "done"
        self._stop_pause_file_watcher()
        with self._lock:
            self.renderer.close(self.state)

    def _emit(self, event_name: str, payload: dict[str, object], *, force: bool = False) -> None:
        if isinstance(self.renderer, LiveTerminalDashboardRenderer):
            self.renderer.emit(self.state, event_name, payload, force=force)
        else:
            self.renderer.emit(self.state, event_name, payload)

    def _record_progress_sample(self) -> None:
        now = time.perf_counter()
        self.state.throughput_samples.append((now, self.state.eta_completed))
        cutoff = now - DEFAULT_THROUGHPUT_SAMPLE_WINDOW_SECONDS
        while len(self.state.throughput_samples) > 1 and self.state.throughput_samples[0][0] < cutoff:
            self.state.throughput_samples.popleft()

    def _start_pause_file_watcher(self) -> None:
        if not self._pause_control_file or self._pause_file_thread is not None:
            return
        self._pause_file_stop.clear()
        self._pause_file_thread = threading.Thread(
            target=self._pause_file_loop,
            name="progress-pause-file",
            daemon=True,
        )
        self._pause_file_thread.start()

    def _stop_pause_file_watcher(self) -> None:
        self._pause_file_stop.set()
        if self._pause_file_thread is not None:
            self._pause_file_thread.join(timeout=1.0)
            self._pause_file_thread = None

    def _pause_file_loop(self) -> None:
        while not self._pause_file_stop.wait(0.2):
            if not self._pause_control_file:
                return
            if not os.path.exists(self._pause_control_file):
                continue
            with contextlib.suppress(OSError):
                os.unlink(self._pause_control_file)
            self.request_pause(reason=f"control_file:{self._pause_control_file}")
            return


def create_progress_reporter(
    title: str,
    *,
    unit: str = "task",
    total: int | None = None,
    mode: str = "auto",
    stream: TextIO | None = None,
    is_tty: bool | None = None,
    on_pause_request=None,
    enable_dashboard_hotkeys: bool = True,
    pause_control_file: str | None = None,
    dashboard_kind: str = "generic",
) -> ProgressReporter:
    return ProgressReporter(
        title,
        unit=unit,
        total=total,
        mode=mode,
        stream=stream,
        is_tty=is_tty,
        on_pause_request=on_pause_request,
        enable_dashboard_hotkeys=enable_dashboard_hotkeys,
        pause_control_file=pause_control_file,
        dashboard_kind=dashboard_kind,
    )


@dataclass
class RunControl:
    _pause_requested: bool = False
    _paused: bool = False
    pause_completed_at: float | None = None
    pause_reason: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def request_pause(self, *, reason: str = "operator") -> None:
        with self._lock:
            if self._pause_requested:
                return
            self._pause_requested = True
            self.pause_reason = reason

    def pause_requested(self) -> bool:
        with self._lock:
            return self._pause_requested

    def mark_paused(self, *, reason: str | None = None) -> None:
        with self._lock:
            self._pause_requested = True
            self._paused = True
            if reason is not None:
                self.pause_reason = reason
            self.pause_completed_at = time.perf_counter()

    def is_paused(self) -> bool:
        with self._lock:
            return self._paused


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
