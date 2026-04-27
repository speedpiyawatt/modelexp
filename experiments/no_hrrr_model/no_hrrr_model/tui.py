from __future__ import annotations

import asyncio
import argparse
import datetime as dt
import json
import pathlib
import re
import shutil
import sys
import uuid
from dataclasses import dataclass
from zoneinfo import ZoneInfo

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, Footer, Header, Input, Label, Log, Select, Static
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI users without optional dependency.
    raise SystemExit("Textual is not installed. Run: uv pip install --python .venv/bin/python -r experiments/no_hrrr_model/requirements.txt") from exc

from .polymarket_event import weather_event_slug_for_date


NY_TZ = ZoneInfo("America/New_York")
DEFAULT_TUI_RUNTIME_ROOT = pathlib.Path("experiments/no_hrrr_model/data/runtime/tui_online_inference")
PREDICTION_OUTPUT_SUBDIR = "predictions"
RUN_STATUS_SUBDIR = "runtime/status"
SAFE_RUNTIME_PART = "tui_online_inference"
TARGET_DATE_RE = re.compile(r"^\s*(\d{4})[-/](\d{1,2})[-/](\d{1,2})\s*$")


@dataclass(frozen=True)
class RunConfig:
    target_date_local: dt.date
    station_id: str
    lamp_source: str
    max_so_far_f: float | None
    runtime_root: pathlib.Path


def default_target_date(now: dt.datetime | None = None) -> dt.date:
    timestamp = now or dt.datetime.now(dt.timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=dt.timezone.utc)
    return timestamp.astimezone(NY_TZ).date()


def parse_target_date(value: str) -> dt.date:
    match = TARGET_DATE_RE.match(value)
    if not match:
        raise ValueError("target date must be YYYY-MM-DD")
    year, month, day = (int(part) for part in match.groups())
    try:
        return dt.date(year, month, day)
    except ValueError as exc:
        raise ValueError(f"target date is invalid: {exc}") from exc


def run_id_for_date(target_date: dt.date) -> str:
    return f"target_date_local={target_date.isoformat()}__run_id={uuid.uuid4().hex[:8]}"


def prediction_path_for_run(run_root: pathlib.Path, station_id: str, target_date: dt.date) -> pathlib.Path:
    return run_root / PREDICTION_OUTPUT_SUBDIR / f"prediction_{station_id}_{target_date.isoformat()}.json"


def manifest_path_for_run(run_root: pathlib.Path, target_date: dt.date) -> pathlib.Path:
    return run_root / RUN_STATUS_SUBDIR / f"target_date_local={target_date.isoformat()}" / "online_inference.manifest.json"


def command_for_run(config: RunConfig, run_root: pathlib.Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.no_hrrr_model.no_hrrr_model.run_online_inference",
        "--target-date-local",
        config.target_date_local.isoformat(),
        "--station-id",
        config.station_id,
        "--lamp-source",
        config.lamp_source,
        "--runtime-root",
        str(run_root / "runtime"),
        "--prediction-output-dir",
        str(run_root / PREDICTION_OUTPUT_SUBDIR),
        "--polymarket-event-slug",
    ]
    if config.max_so_far_f is not None:
        command.extend(["--max-so-far-f", str(config.max_so_far_f)])
    return command


def format_prediction_summary(payload: dict[str, object]) -> str:
    lines = [
        f"Status: {payload.get('status', 'unknown')}",
        f"Date: {payload.get('target_date_local')}  Station: {payload.get('station_id')}",
        f"Expected final high: {float(payload.get('expected_final_tmax_f', float('nan'))):.2f} F",
        f"Anchor high: {float(payload.get('anchor_tmax_f', float('nan'))):.2f} F",
    ]
    final_quantiles = payload.get("final_tmax_quantiles_f", {})
    if isinstance(final_quantiles, dict):
        labels = (("0.05", "q05"), ("0.1", "q10"), ("0.25", "q25"), ("0.5", "q50"), ("0.75", "q75"), ("0.9", "q90"), ("0.95", "q95"))
        values = [f"{label}={float(final_quantiles[key]):.2f}" for key, label in labels if key in final_quantiles]
        if values:
            lines.append("Quantiles: " + "  ".join(values))
    event_bins = payload.get("event_bins", [])
    if isinstance(event_bins, list) and event_bins:
        lines.append("")
        lines.append("Event bins")
        for row in event_bins:
            if isinstance(row, dict):
                lines.append(f"  {row.get('bin')}: {float(row.get('probability', 0.0)):.4f}")
    output_path = payload.get("_prediction_path")
    if output_path:
        lines.append("")
        lines.append(f"Prediction JSON: {output_path}")
    return "\n".join(lines)


def failure_message(manifest_path: pathlib.Path, return_code: int) -> str:
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            payload = {}
        message = payload.get("message")
        if message:
            return str(message)
    return f"Online inference failed with return code {return_code}."


def safe_delete_run_root(run_root: pathlib.Path, allowed_parent: pathlib.Path | None = None) -> bool:
    resolved = run_root.resolve()
    if not resolved.name.startswith("target_date_local="):
        return False
    if allowed_parent is not None:
        try:
            resolved.relative_to(allowed_parent.resolve())
        except ValueError:
            return False
    else:
        parts = set(resolved.parts)
        if SAFE_RUNTIME_PART not in parts:
            return False
    if resolved.exists():
        shutil.rmtree(resolved)
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the no-HRRR KLGA online inference TUI.")
    parser.add_argument(
        "--runtime-root",
        type=pathlib.Path,
        default=DEFAULT_TUI_RUNTIME_ROOT,
        help="Root directory for TUI run artifacts.",
    )
    parser.add_argument(
        "--enable-mouse",
        action="store_true",
        help="Enable terminal mouse tracking. Disabled by default because some terminals leak mouse escape sequences into inputs.",
    )
    return parser.parse_args(argv)


class NoHrrrInferenceTui(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }
    #controls {
        height: 5;
        padding: 0 1;
    }
    .field_label {
        width: auto;
        padding: 1 0 0 1;
    }
    #path_controls {
        height: 3;
        padding: 0 1;
    }
    #results {
        height: 18;
        border: solid $primary;
        padding: 1;
    }
    #log {
        border: solid $secondary;
        height: 1fr;
    }
    Input {
        width: 24;
    }
    Select {
        width: 20;
    }
    Button {
        width: 16;
    }
    #runtime_root {
        width: 1fr;
    }
    """
    BINDINGS = [("ctrl+q", "quit", "Quit"), ("escape", "quit", "Quit")]

    def __init__(self, runtime_root: pathlib.Path = DEFAULT_TUI_RUNTIME_ROOT) -> None:
        super().__init__()
        self.runtime_root = runtime_root
        self.current_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        target_date = default_target_date().isoformat()
        yield Header()
        yield Vertical(
            Label("No-HRRR KLGA Online Inference"),
            Horizontal(
                Label("Date", classes="field_label"),
                Input(value=target_date, placeholder="YYYY-MM-DD", id="target_date"),
                Label("Station", classes="field_label"),
                Input(value="KLGA", placeholder="Station", id="station_id"),
                Label("LAMP", classes="field_label"),
                Select(
                    [("auto", "auto"), ("live", "live"), ("archive", "archive"), ("iem", "iem")],
                    value="auto",
                    id="lamp_source",
                    allow_blank=False,
                ),
                Label("Max", classes="field_label"),
                Input(value="", placeholder="max so far F (optional)", id="max_so_far"),
                Button("Run", id="run", variant="primary"),
                id="controls",
            ),
            Horizontal(
                Label("Runtime root"),
                Input(value=str(self.runtime_root), placeholder="TUI runtime artifact root", id="runtime_root"),
                id="path_controls",
            ),
            Static(self.initial_result_text(target_date), id="results"),
            Log(id="log", highlight=True),
        )
        yield Footer()

    def initial_result_text(self, target_date: str) -> str:
        slug = weather_event_slug_for_date(dt.date.fromisoformat(target_date))
        return f"Ready.\nTarget date defaults to New York today.\nPolymarket slug: {slug}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "run":
            return
        if self.current_task is not None and not self.current_task.done():
            self.query_one("#results", Static).update("A run is already active.")
            return
        try:
            config = self.config_from_inputs()
        except ValueError as exc:
            self.query_one("#results", Static).update(f"Input error: {exc}")
            return
        self.current_task = asyncio.create_task(self.run_inference(config))

    def config_from_inputs(self) -> RunConfig:
        target_date_input = self.query_one("#target_date", Input)
        target_date = parse_target_date(target_date_input.value)
        target_date_input.value = target_date.isoformat()
        station_id = self.query_one("#station_id", Input).value.strip() or "KLGA"
        lamp_source = str(self.query_one("#lamp_source", Select).value or "auto")
        max_text = self.query_one("#max_so_far", Input).value.strip()
        max_so_far = float(max_text) if max_text else None
        runtime_root_text = self.query_one("#runtime_root", Input).value.strip()
        if not runtime_root_text:
            raise ValueError("runtime root is required")
        runtime_root = pathlib.Path(runtime_root_text).expanduser()
        return RunConfig(
            target_date_local=target_date,
            station_id=station_id,
            lamp_source=lamp_source,
            max_so_far_f=max_so_far,
            runtime_root=runtime_root,
        )

    async def run_inference(self, config: RunConfig) -> None:
        run_root = config.runtime_root / run_id_for_date(config.target_date_local)
        prediction_path = prediction_path_for_run(run_root, config.station_id, config.target_date_local)
        manifest_path = manifest_path_for_run(run_root, config.target_date_local)
        command = command_for_run(config, run_root)
        log = self.query_one("#log", Log)
        results = self.query_one("#results", Static)
        log.clear()
        results.update(
            "Running online inference...\n"
            f"Date: {config.target_date_local.isoformat()}\n"
            f"Station: {config.station_id}\n"
            f"LAMP source: {config.lamp_source}\n"
            f"Runtime root: {config.runtime_root}\n"
            f"Polymarket slug: {weather_event_slug_for_date(config.target_date_local)}"
        )
        log.write_line("+ " + " ".join(command))
        process = await asyncio.create_subprocess_exec(*command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        assert process.stdout is not None
        async for raw_line in process.stdout:
            log.write_line(raw_line.decode("utf-8", errors="replace").rstrip())
        return_code = await process.wait()
        if return_code == 0 and prediction_path.exists():
            payload = json.loads(prediction_path.read_text())
            payload["_prediction_path"] = str(prediction_path)
            results.update(format_prediction_summary(payload))
            return
        message = failure_message(manifest_path, return_code)
        deleted = safe_delete_run_root(run_root, allowed_parent=config.runtime_root)
        cleanup = "Cleaned up TUI run artifacts." if deleted else "Did not delete artifacts; path did not pass safety checks."
        results.update(f"Run failed.\n\n{message}\n\n{cleanup}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    NoHrrrInferenceTui(runtime_root=args.runtime_root).run(mouse=args.enable_mouse)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
