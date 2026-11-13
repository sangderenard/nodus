"""Snapshot a bounded year of Alpaca IEX AAPL/SPY and BTC bars for offline training."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--secrets-dir", type=Path, required=True)
    parser.add_argument("--account-id", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.turing_root.resolve() / "examples"))
    from meowdoku import alpaca_credentials, market_data

    if args.output.exists():
        raise FileExistsError("historical snapshot output already exists")
    keys = alpaca_credentials.load_paper(args.secrets_dir, args.account_id)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)
    params = {"start": start.isoformat().replace("+00:00", "Z"),
              "end": (end - timedelta(minutes=20)).isoformat().replace("+00:00", "Z"),
              "timeframe": "1Hour", "limit": "10000"}
    stocks, _ = market_data._history_pages("/v2/stocks/bars", {
        **params, "symbols": "AAPL,SPY", "feed": "iex"}, keys)
    btc_by_time = {}
    cursor = start
    while cursor < end:
        boundary = min(cursor + timedelta(days=30), end)
        crypto, _ = market_data._history_pages("/v1beta3/crypto/us/bars", {
            "start": cursor.isoformat().replace("+00:00", "Z"),
            "end": boundary.isoformat().replace("+00:00", "Z"),
            "timeframe": "1Hour", "limit": "10000", "symbols": "BTC/USD"}, keys)
        for bar in crypto.get("bars", {}).get("BTC/USD", []):
            btc_by_time[bar["t"]] = bar
        cursor = boundary
    history = {"asof": end.isoformat(), "stock": "AAPL", "etf_feed": "iex",
               "bars": {"AAPL": stocks.get("bars", {}).get("AAPL", []),
                        "SPY": stocks.get("bars", {}).get("SPY", []),
                        "BTC/USD": [btc_by_time[t] for t in sorted(btc_by_time)]}}
    payload = (json.dumps(history, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    print(json.dumps({"path": str(args.output.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
                      "bar_counts": {symbol: len(bars) for symbol, bars in history["bars"].items()},
                      "start": params["start"], "asof": history["asof"]}, indent=2))


if __name__ == "__main__":
    main()
