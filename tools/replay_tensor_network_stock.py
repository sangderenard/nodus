"""Replay completed stock bars through a compiled Nodus row and Meowdoku mock ledger."""

from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_DOWN
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from nodus_graph import Graph, compile_row_product_graph, pack_tensor_handle, unpack_tensor_handle
from tools.compile_tensor_network_row import SHAPES, request


OWNER = 1  # scratch database only; no live account state is opened
STOCK = "AAPL"


def load_history(turing_root: Path, *, history_file: Path | None,
                 secrets_dir: Path | None, account_id: int | None) -> dict:
    if history_file is not None:
        history = json.loads(history_file.read_text(encoding="utf-8"))
    else:
        if secrets_dir is None or account_id is None:
            raise ValueError("provide --history-file or both --secrets-dir and --account-id")
        sys.path.insert(0, str(turing_root / "examples"))
        from meowdoku import market_data
        history = market_data.history(secrets_dir, account_id, STOCK)
    if history.get("stock") != STOCK or not isinstance(history.get("bars"), dict):
        raise ValueError("expected the existing AAPL market history response")
    return history


def _series(history: dict, symbol: str) -> list[dict]:
    bars = history["bars"].get(symbol)
    if not isinstance(bars, list):
        raise ValueError(f"{symbol} history missing")
    times = [bar.get("t") for bar in bars]
    if any(not isinstance(t, str) for t in times) or times != sorted(set(times)):
        raise ValueError(f"{symbol} history timestamps must be unique and increasing")
    for bar in bars:
        if not all(isinstance(bar.get(k), (int, float)) and np.isfinite(bar[k]) and bar[k] > 0
                   for k in ("o", "c")):
            raise ValueError(f"{symbol} history has an invalid open or close")
    return bars


def causal_rows(history: dict, paper_policies) -> list[dict]:
    """Use bars through the decision hour; the next stock bar supplies execution."""
    stock = _series(history, STOCK)
    spy = _series(history, "SPY")
    btc = _series(history, "BTC/USD")
    if len(stock) < 11:
        raise ValueError("AAPL history needs at least eleven bars")
    rows = []
    # The final returned bar may still be forming. Exclude it and fill only at
    # the open of an earlier, completed bar.
    for index in range(8, len(stock) - 2):
        decision, fill = stock[index], stock[index + 1]
        cutoff = decision["t"]
        prefix = {STOCK: stock[:index + 1],
                  "SPY": [bar for bar in spy if bar["t"] <= cutoff],
                  "BTC/USD": [bar for bar in btc if bar["t"] <= cutoff]}
        if len(prefix["SPY"]) < 4 or len(prefix["BTC/USD"]) < 4:
            continue
        vector = paper_policies.features({"bars": prefix}).get(STOCK)
        if vector is None:
            continue
        rows.append({"decision_bar": cutoff, "fill_bar": fill["t"],
                     "fill_open": fill["o"], "features": vector})
    return rows


def probe_weights() -> dict[str, np.ndarray]:
    """Fixed sign-of-return plumbing probe, never presented as trained weights."""
    first = np.zeros(SHAPES["input_weight"], dtype=np.float64)
    first[1, 0] = 40.0
    first[1, 1] = -40.0
    second = np.zeros(SHAPES["output_weight"], dtype=np.float64)
    second[0, 0] = 1.0
    second[1, 1] = 1.0
    return {"input_weight": first,
            "hidden_bias": np.zeros(SHAPES["hidden_bias"], dtype=np.float64),
            "output_weight": second,
            "output_bias": np.zeros(SHAPES["output_bias"], dtype=np.float64)}


def infer(graph: Graph, rows: list[dict], weights: dict[str, np.ndarray]) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    for offset in range(0, len(rows), 2):
        pair = rows[offset:offset + 2]
        features = np.stack([row["features"] for row in pair] +
                            ([pair[-1]["features"]] if len(pair) == 1 else []))
        values = {"features": features, **weights}
        handles: list[int] = []
        try:
            for name, value in values.items():
                handle = graph.create_tensor(2, value.shape, value.tobytes(order="C"))
                handles.append(handle)
                if not graph.push_bytes(name, pack_tensor_handle(handle)).accepted:
                    raise RuntimeError(f"Nodus refused {name} input")
            graph.step()
            payload = graph.pull_bytes("prediction", 8)
            if payload is None:
                raise RuntimeError("compiled network published no prediction")
            result = unpack_tensor_handle(payload)
            handles.append(result)
            prediction = np.frombuffer(graph.read_tensor(result), dtype=np.float64).reshape(SHAPES["prediction"])
            if not np.isfinite(prediction).all():
                raise RuntimeError("compiled network returned nonfinite scores")
            outputs.extend(prediction[:len(pair)].copy())
        finally:
            for handle in reversed(handles):
                graph.destroy_tensor(handle)
    return outputs


def replay(db, rows: list[dict], predictions: list[np.ndarray], paper_arena, paper_mock,
           *, dataset_sha256: str, product_sha256: str, final_mark: Decimal,
           final_mark_bar: str, weights_label: str = "fixed untrained sign-of-return probe",
           record_events: bool = True) -> dict:
    if len(rows) != len(predictions):
        raise ValueError("every decision requires one compiled prediction")
    bot = paper_arena.register_bot(db, OWNER, {
        "request_id": "historical_aapl_compiled_bot", "name": "Historical AAPL compiled probe",
        "kind": "compiled", "source": {"ref": "product:" + product_sha256}})
    strategy = paper_arena.create_strategy(db, OWNER, {
        "request_id": "historical_aapl_cautious", "name": "Historical AAPL probe",
        "risk": "cautious", "bot_id": bot["id"], "policy": "tensor",
        "outcomes": ["fixed_horizon", "closed_trade", "fill_quality"]})
    strategy_id = strategy["id"]
    receipts = []
    for index, (row, score) in enumerate(zip(rows, predictions)):
        mark = Decimal(str(row["fill_open"]))
        ledger = paper_mock.ledger(db, OWNER, strategy_id)
        holding = next((item for item in ledger["holdings"] if item["symbol"] == STOCK), None)
        quantity = Decimal(holding["quantity"]) if holding else Decimal(0)
        cash = Decimal(ledger["cash_usd"])
        action, amount = "hold", None
        if float(score[0]) > 0.02 and quantity * mark < Decimal("0.00001") and cash >= Decimal("5.0025"):
            action, amount = "buy", Decimal("5")
        elif float(score[1]) > 0.02 and quantity * mark >= Decimal("0.00001"):
            execution = mark * (1 - (paper_mock.SPREAD_BPS + paper_mock.MARKUP_BPS) / paper_mock.DENOM)
            amount = min(Decimal("5"), (quantity * execution).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN))
            if amount > 0:
                action = "sell"
        receipt = None
        if action != "hold":
            order = {"request_id": f"historical-{index}", "strategy_id": strategy_id,
                     "symbol": STOCK, "side": action, "amount": format(amount, "f")}
            receipt = paper_mock.place(db, OWNER, order, mark)
            if record_events:
                paper_arena.record_activity(db, OWNER, bot["id"], "kernel", "historical mock fill",
                                            strategy_id, json.dumps({"venue": "internal_mock", "order_id": receipt["id"],
                                                                     "decision_bar": row["decision_bar"], "fill_bar": row["fill_bar"]}))
        marked = paper_mock.ledger(db, OWNER, strategy_id)
        marked_quantity = sum((Decimal(item["quantity"]) for item in marked["holdings"]
                               if item["symbol"] == STOCK), Decimal(0))
        marked_equity = Decimal(marked["cash_usd"]) + marked_quantity * mark
        receipts.append({"decision_bar": row["decision_bar"], "fill_bar": row["fill_bar"],
                         "fill_open": row["fill_open"], "features": row["features"].tolist(),
                         "scores": score.tolist(), "action": action, "order": receipt,
                         "cash_usd": marked["cash_usd"], "position_qty": str(marked_quantity),
                         "marked_equity_usd": str(marked_equity)})
    final_ledger = paper_mock.ledger(db, OWNER, strategy_id)
    holdings_value = sum((Decimal(item["quantity"]) * final_mark for item in final_ledger["holdings"]
                          if item["symbol"] == STOCK), Decimal(0))
    equity = Decimal(final_ledger["cash_usd"]) + holdings_value
    return {"schema": "nodus-paper-historical-replay-v1", "stock": STOCK,
            "venue": "internal_mock", "feed": "Alpaca IEX hourly history",
            "dataset_sha256": dataset_sha256, "product_sha256": product_sha256,
            "weights": weights_label, "risk": "cautious",
            "strategy_id": strategy_id, "bot_id": bot["id"], "decisions": receipts,
            "ledger": final_ledger, "final_mark_bar": final_mark_bar,
            "final_mark_close": str(final_mark),
            "ending_equity_usd": str(equity), "pnl_usd": str(equity - Decimal("50")),
            "trainer_events": paper_arena.trainer_events(db, 0, 500)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--native-nodus-root", type=Path)
    parser.add_argument("--history-file", type=Path)
    parser.add_argument("--secrets-dir", type=Path)
    parser.add_argument("--account-id", type=int)
    args = parser.parse_args()
    output, turing_root = args.output.resolve(), args.turing_root.resolve()
    if (output / "paper-arena.sqlite").exists():
        raise FileExistsError("use a new output directory for each replay")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(turing_root / "examples"))
    from meowdoku import paper_arena, paper_mock, paper_policies

    history = load_history(turing_root, history_file=args.history_file,
                           secrets_dir=args.secrets_dir, account_id=args.account_id)
    history_bytes = (json.dumps(history, sort_keys=True, separators=(",", ":")) + "\n").encode()
    history_path = output / "history.json"
    history_path.write_bytes(history_bytes)
    rows = causal_rows(history, paper_policies)
    if not rows:
        raise ValueError("no completed AAPL decision/fill pairs")

    nodus_root = Path(__file__).resolve().parents[1]
    native_root = (args.native_nodus_root or nodus_root).resolve()
    row_request = request(output, turing_root, nodus_root)
    request_path = output / "request.json"
    request_path.write_text(json.dumps(row_request, indent=2), encoding="utf-8")
    graph = Graph()
    loaded_id = ""
    try:
        _, adoption = compile_row_product_graph(
            graph, request_path, stable_id=0x4E45544E,
            python_executable=sys.executable, turing_root=turing_root,
            product_dir=row_request["output_dir"], nodus_root=native_root,
            module_library_root=output / "module-library")
        loaded_id = str(adoption["loaded_tool_id"])
        predictions = infer(graph, rows, probe_weights())
    finally:
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
    manifest_path = Path(row_request["output_dir"]) / "nodus-row-tool-product.json"
    product_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    db = paper_arena.connect(output / "paper-arena.sqlite")
    try:
        final_bar = history["bars"][STOCK][-2]
        result = replay(db, rows, predictions, paper_arena, paper_mock,
                        dataset_sha256=hashlib.sha256(history_bytes).hexdigest(),
                        product_sha256=product_hash,
                        final_mark=Decimal(str(final_bar["c"])), final_mark_bar=final_bar["t"])
    finally:
        db.close()
    (output / "replay.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "equity.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("decision_bar", "fill_bar", "fill_open", "action",
                                                    "buy_score", "sell_score", "cash_usd", "position_qty",
                                                    "marked_equity_usd"))
        writer.writeheader()
        for item in result["decisions"]:
            writer.writerow({"decision_bar": item["decision_bar"], "fill_bar": item["fill_bar"],
                             "fill_open": item["fill_open"], "action": item["action"],
                             "buy_score": item["scores"][0], "sell_score": item["scores"][1],
                             "cash_usd": item["cash_usd"], "position_qty": item["position_qty"],
                             "marked_equity_usd": item["marked_equity_usd"]})
    print(json.dumps({"stock": STOCK, "bars": len(history["bars"][STOCK]),
                      "decisions": len(rows), "compiled_batches": (len(rows) + 1) // 2,
                      "orders": sum(item["order"] is not None for item in result["decisions"]),
                      "ending_equity_usd": result["ending_equity_usd"],
                      "pnl_usd": result["pnl_usd"], "output": str(output / "replay.json")}, indent=2))


if __name__ == "__main__":
    main()
