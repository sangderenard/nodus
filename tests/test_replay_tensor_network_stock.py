"""Historical decisions cannot read a later stock or linked-market bar."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from tools.replay_tensor_network_stock import causal_rows


class LastVisibleCloses:
    @staticmethod
    def features(history):
        bars = history["bars"]
        return {"AAPL": np.asarray([bars[symbol][-1]["c"]
                                    for symbol in ("AAPL", "SPY", "BTC/USD")])}


def test_next_open_is_execution_only_and_last_forming_bar_is_excluded():
    bars = [{"t": f"2026-01-01T{i:02d}:00:00Z", "o": float(i + 100),
             "c": float(i + 101)} for i in range(12)]
    history = {"bars": {symbol: deepcopy(bars) for symbol in ("AAPL", "SPY", "BTC/USD")}}
    first = causal_rows(history, LastVisibleCloses)
    assert len(first) == 2
    assert [(row["decision_bar"], row["fill_bar"]) for row in first] == [
        (bars[8]["t"], bars[9]["t"]), (bars[9]["t"], bars[10]["t"])]
    assert first[0]["features"].tolist() == [bars[8]["c"]] * 3
    assert first[0]["fill_open"] == bars[9]["o"]

    later = deepcopy(history)
    for symbol in later["bars"]:
        later["bars"][symbol][9]["c"] *= 100
        later["bars"][symbol][11]["c"] *= 100
    after = causal_rows(later, LastVisibleCloses)
    np.testing.assert_array_equal(first[0]["features"], after[0]["features"])
    assert after[0]["fill_open"] == first[0]["fill_open"]
