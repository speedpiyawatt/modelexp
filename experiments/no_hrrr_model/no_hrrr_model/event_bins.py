from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class EventBin:
    name: str
    lower_f: int | None = None
    upper_f: int | None = None

    def contains(self, temp_f: int) -> bool:
        if self.lower_f is not None and temp_f < self.lower_f:
            return False
        if self.upper_f is not None and temp_f > self.upper_f:
            return False
        return True


def parse_event_bin(label: str) -> EventBin:
    text = label.strip()
    lower = re.search(r"(\d+)\s*(?:F|degrees?)?\s*(?:or|and)?\s*(?:above|higher|\+)", text, re.IGNORECASE)
    if lower and re.search(r"above|higher|\+", text, re.IGNORECASE):
        return EventBin(name=label, lower_f=int(lower.group(1)))
    upper = re.search(r"(\d+)\s*(?:F|degrees?)?\s*(?:or|and)?\s*(?:below|lower|under)", text, re.IGNORECASE)
    if upper:
        value = int(upper.group(1))
        strict = re.search(r"\b(?:under|below|less than)\b", text, re.IGNORECASE) and not re.search(r"\bor\s+(?:below|lower|under)\b", text, re.IGNORECASE)
        return EventBin(name=label, upper_f=value - 1 if strict else value)
    leading_upper = re.search(r"\b(?:under|below|less than)\s+(\d+)\s*(?:F|degrees?)?", text, re.IGNORECASE)
    if leading_upper:
        return EventBin(name=label, upper_f=int(leading_upper.group(1)) - 1)
    leading_lower = re.search(r"\b(?:over|above|greater than)\s+(\d+)\s*(?:F|degrees?)?", text, re.IGNORECASE)
    if leading_lower:
        return EventBin(name=label, lower_f=int(leading_lower.group(1)) + 1)
    rng = re.search(r"(\d+)\s*(?:-|to|through)\s*(\d+)", text, re.IGNORECASE)
    if rng:
        lo, hi = sorted((int(rng.group(1)), int(rng.group(2))))
        return EventBin(name=label, lower_f=lo, upper_f=hi)
    exact = re.fullmatch(r"\s*(\d+)\s*F?\s*", text, re.IGNORECASE)
    if exact:
        value = int(exact.group(1))
        return EventBin(name=label, lower_f=value, upper_f=value)
    raise ValueError(f"could not parse event bin label: {label!r}")


def load_event_bin_labels(path: str | Path) -> list[str]:
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text())
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = payload.get("labels") or payload.get("bins") or payload.get("outcomes") or payload.get("markets")
        else:
            items = None
        if not isinstance(items, list):
            raise ValueError(f"JSON event-bin file must contain a list, bins, outcomes, or markets: {source}")
        labels = []
        for item in items:
            if isinstance(item, str):
                labels.append(item)
            elif isinstance(item, dict):
                value = item.get("label") or item.get("name") or item.get("outcome") or item.get("title")
                if value is not None:
                    labels.append(str(value))
        if not labels:
            raise ValueError(f"no event-bin labels found in {source}")
        return labels
    labels = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"no event-bin labels found in {source}")
    return labels


def map_ladder_to_bins(
    ladder: pd.DataFrame,
    bins: list[EventBin],
    *,
    max_so_far_f: float | None = None,
) -> pd.DataFrame:
    work = ladder.copy()
    work["temp_f"] = pd.to_numeric(work["temp_f"]).astype(int)
    work["probability"] = pd.to_numeric(work["probability"], errors="coerce").fillna(0.0)
    if max_so_far_f is not None:
        work.loc[work["temp_f"] < float(max_so_far_f), "probability"] = 0.0
        total = work["probability"].sum()
        if total > 0:
            work["probability"] = work["probability"] / total
    rows = []
    for event_bin in bins:
        mask = work["temp_f"].map(event_bin.contains)
        rows.append({"bin": event_bin.name, "probability": float(work.loc[mask, "probability"].sum())})
    return pd.DataFrame(rows)
