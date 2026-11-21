"""Causal live observations and matured hourly labels for the paper trainer."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np

from tools.replay_tensor_network_stock import STOCK, _series
from tools.train_aapl_qualified_row import scaled, target_for


HOUR = timedelta(hours=1)


def timestamp(value: str) -> datetime:
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("market bar timestamp lacks a timezone")
    return instant.astimezone(timezone.utc)


def current_row(history: dict, latest: dict, policies, *, now: datetime) -> tuple[dict | None, str]:
    """Infer at the latest minute with prior completed hourly context; never label it."""
    bars = latest.get("bars", {})
    required = (STOCK, "SPY", "BTC/USD")
    if any(not isinstance(bars.get(symbol), dict) for symbol in required):
        return None, "missing"
    times = [timestamp(bars[symbol]["t"]) for symbol in required]
    if any(now - at > timedelta(minutes=3) or at > now for at in times):
        return None, "stale"
    if max(times) - min(times) > timedelta(minutes=2):
        return None, "unaligned"
    hour_start = times[0].replace(minute=0, second=0, microsecond=0)
    prefix = {}
    for symbol in required:
        prior = [bar for bar in _series(history, symbol) if timestamp(bar["t"]) < hour_start]
        current = bars[symbol]
        close = current.get("c")
        if (not isinstance(close, (int, float)) or isinstance(close, bool)
                or not math.isfinite(close) or close <= 0):
            return None, "invalid"
        prefix[symbol] = prior + [current]
    vector = policies.features({"bars": prefix}).get(STOCK)
    if vector is None:
        return None, "insufficient"
    row = scaled([{"decision_bar": bars[STOCK]["t"], "features": vector}])[0]
    return row, "current"


def matured_rows(history: dict, observations: list[dict], *, now: datetime) -> tuple[np.ndarray, np.ndarray, int]:
    """Label the exact features observed live only after four future hourly bars complete."""
    stock = _series(history, STOCK)
    by_time = {bar["t"]: index for index, bar in enumerate(stock)}
    by_hour = {timestamp(bar["t"]): index for index, bar in enumerate(stock)}
    complete_before = now - timedelta(minutes=20)
    rows = []
    for observed in observations:
        hour = timestamp(observed["t"]).replace(minute=0, second=0, microsecond=0)
        index = by_hour.get(hour)
        if (index is None or index + 4 >= len(stock)
                or timestamp(stock[index + 4]["t"]) + HOUR > complete_before):
            continue
        vector = np.asarray(observed["features"], dtype=np.float64)
        if vector.shape != (6,) or not np.isfinite(vector).all():
            raise ValueError("recorded live feature frame is invalid")
        rows.append({"decision_bar": stock[index]["t"],
                     "fill_open": stock[index + 1]["o"], "features": vector})
    rows = rows[-256:]
    if not rows:
        return np.empty((0, 6), dtype=np.float64), np.empty((0, 3), dtype=np.float64), 0
    return (np.stack([row["features"] for row in rows]),
            np.stack([target_for(row, by_time, stock) for row in rows]), len(rows))
