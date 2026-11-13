"""Train one bounded AbstractTensor policy, then judge unseen AAPL mock windows."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from nodus_graph import Graph, compile_row_product_graph
from tools.compile_tensor_network_row import request
from tools.replay_tensor_network_stock import (
    OWNER, STOCK, _series, causal_rows, infer, replay,
)


FEATURE_SCALE = 100.0
MAX_EPOCHS = 160
CHECK_EPOCHS = (20, 40, 80, 160)
SEEDS = (0, 1, 2)
MARGIN = Decimal("0.01")  # 1% of the $5 maximum per-order capital at risk


def scaled(rows: list[dict]) -> list[dict]:
    return [{**row, "features": np.clip(row["features"] * FEATURE_SCALE, -5.0, 5.0)}
            for row in rows]


def target_for(row: dict, stock_by_time: dict[str, int], stock: list[dict]) -> np.ndarray:
    index = stock_by_time[row["decision_bar"]]
    forward = stock[index + 4]["c"] / row["fill_open"] - 1.0
    buy = math.tanh(20 * (forward - 0.004))
    sell = math.tanh(20 * (-forward - 0.004))
    return np.asarray((buy, sell, 0.0), dtype=np.float64)


class PolicyNetwork:
    """The exact dense/tanh graph used by the compiled Nodus inference row."""

    def __init__(self, AT, seed: int):
        rng = np.random.default_rng(seed)
        self.AT = AT
        with AT.use_backend("numpy"):
            self.parameters = [
                AT.tensor(rng.normal(0, 0.2, (6, 4))),
                AT.tensor(np.zeros((1, 4))),
                AT.tensor(rng.normal(0, 0.2, (4, 3))),
                AT.tensor(np.zeros((1, 3))),
            ]

    def step(self, x: np.ndarray, target: np.ndarray, learning_rate: float = 0.1) -> float:
        AT = self.AT
        with AT.use_backend("numpy"):
            for parameter in self.parameters:
                parameter.requires_grad_()
            w1, b1, w2, b2 = self.parameters
            prediction = ((AT.tensor(x) @ w1 + b1).tanh() @ w2 + b2).tanh()
            error = prediction - AT.tensor(target)
            loss = (error * error).mean()
            gradients = AT.autograd.grad(loss, self.parameters, retain_graph=False)
            with AT.autograd.no_grad():
                for parameter, gradient in zip(self.parameters, gradients):
                    AT.copyto(parameter, parameter - gradient * learning_rate)
            return float(np.asarray(loss.numpy()))

    def weights(self) -> dict[str, np.ndarray]:
        w1, b1, w2, b2 = (np.asarray(parameter.numpy(), dtype=np.float64)
                           for parameter in self.parameters)
        return {"input_weight": w1.copy(), "hidden_bias": np.repeat(b1, 2, axis=0),
                "output_weight": w2.copy(), "output_bias": np.repeat(b2, 2, axis=0)}

    def values(self, rows: list[dict]) -> list[np.ndarray]:
        x = np.stack([row["features"] for row in rows])
        w = self.weights()
        hidden = np.tanh(x @ w["input_weight"] + w["hidden_bias"][0])
        return list(np.tanh(hidden @ w["output_weight"] + w["output_bias"][0]))


def window_results(rows: list[dict], predictions: list[np.ndarray], stock_close: dict[str, Decimal],
                   paper_arena, paper_mock, dataset_hash: str, product_hash: str,
                   *, compiled: bool = False) -> tuple[list[str], list[dict]]:
    size = len(rows) // 3
    parts = [rows[:size], rows[size:2 * size], rows[2 * size:]]
    score_parts = [predictions[:size], predictions[size:2 * size], predictions[2 * size:]]
    pnls, summaries = [], []
    for window, scores in zip(parts, score_parts):
        if not window:
            raise ValueError("qualification windows must be nonempty")
        db = paper_arena.connect(":memory:")
        try:
            result = replay(db, window, scores, paper_arena, paper_mock,
                            dataset_sha256=dataset_hash, product_sha256=product_hash,
                            final_mark=stock_close[window[-1]["fill_bar"]],
                            final_mark_bar=window[-1]["fill_bar"],
                            weights_label="trained AbstractTensor policy", record_events=compiled)
            pnls.append(result["pnl_usd"])
            summaries.append({"first_decision": window[0]["decision_bar"],
                              "last_fill": window[-1]["fill_bar"],
                              "decisions": len(window),
                              "orders": sum(item["order"] is not None for item in result["decisions"]),
                              "sell_fills": sum(item["action"] == "sell" and item["order"] is not None
                                                for item in result["decisions"]),
                              "net_pnl_usd": result["pnl_usd"]})
        finally:
            db.close()
    return pnls, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--native-nodus-root", type=Path)
    parser.add_argument("--publish-db", type=Path,
                        help="shared Meowdoku paper_arena.sqlite for public progress")
    parser.add_argument("--worker", default="local-training-worker")
    args = parser.parse_args()
    if args.publish_db is None:
        return _run(args, None)
    sys.path.insert(0, str(args.turing_root.resolve() / "examples"))
    from meowdoku import qualification_runs
    with qualification_runs.publishing(args.publish_db, market="AAPL + SPY + BTC/USD",
                                       model="AbstractTensor 6→4→3", worker=args.worker,
                                       planned_epochs=len(SEEDS) * MAX_EPOCHS) as (db, run_id):
        _run(args, (qualification_runs, db, run_id))


def _run(args, publish) -> None:
    def announce(phase: str, completed: int, **metrics) -> None:
        if publish is not None:
            module, db, run_id = publish
            module.progress(db, run_id, phase=phase, completed_epochs=completed, **metrics)

    output, turing_root = args.output.resolve(), args.turing_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("training output directory must be fresh")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(turing_root))
    sys.path.insert(0, str(turing_root / "examples"))
    from src.common.tensors import AbstractTensor as AT
    from meowdoku import paper_arena, paper_mock, paper_policies

    history_bytes = args.history_file.read_bytes()
    history = json.loads(history_bytes)
    dataset_hash = hashlib.sha256(history_bytes).hexdigest()
    stock = _series(history, STOCK)
    all_rows = scaled(causal_rows(history, paper_policies))
    cutoff, validation_end = int(len(all_rows) * 0.60), int(len(all_rows) * 0.80)
    if cutoff < 50 or validation_end - cutoff < 30 or len(all_rows) - validation_end < 30:
        raise ValueError("history is too short for chronological qualification")
    train_rows = all_rows[:cutoff - 4]
    validation_rows = all_rows[cutoff:validation_end]
    holdout_rows = all_rows[validation_end:]
    stock_by_time = {bar["t"]: index for index, bar in enumerate(stock)}
    stock_close = {bar["t"]: Decimal(str(bar["c"])) for bar in stock}
    x = np.stack([row["features"] for row in train_rows])
    target = np.stack([target_for(row, stock_by_time, stock) for row in train_rows])

    candidates = []
    selected = None
    completed = 0
    for seed in SEEDS:
        network = PolicyNetwork(AT, seed)
        for epoch in range(1, MAX_EPOCHS + 1):
            loss = network.step(x, target)
            completed += 1
            if epoch % 5 == 0 and epoch not in CHECK_EPOCHS:
                announce("training", completed, seed=seed, epoch=epoch, loss=loss)
            if epoch not in CHECK_EPOCHS:
                continue
            scores = network.values(validation_rows)
            pnls, windows = window_results(validation_rows, scores, stock_close, paper_arena, paper_mock,
                                            dataset_hash, "0" * 64)
            returns = [Decimal(value) / Decimal("5") for value in pnls]
            mean = sum(returns) / len(returns)
            passed = (all(value > 0 for value in returns) and mean >= MARGIN
                      and all(window["sell_fills"] >= 1 for window in windows))
            candidate = {"seed": seed, "epoch": epoch, "train_loss": loss,
                         "validation_window_pnl_usd": pnls,
                         "validation_mean_return_on_order_cap": str(mean),
                         "validation_passed": passed, "validation_windows": windows}
            candidates.append(candidate)
            announce("validation", completed, seed=seed, epoch=epoch, loss=loss,
                     validation_mean=float(mean), validation_passed=passed)
            if selected is None or mean > selected[0]:
                selected = (mean, network.weights(), candidate)
            if passed:
                selected = (mean, network.weights(), candidate)
                break
        if selected and selected[2]["validation_passed"]:
            break
    assert selected is not None
    _, weights, chosen = selected
    np.savez(output / "weights.npz", **weights)
    announce("compiling", completed)

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
        holdout_scores = infer(graph, holdout_rows, weights)
        eager_x = np.stack([row["features"] for row in holdout_rows])
        eager_hidden = np.tanh(eager_x @ weights["input_weight"] + weights["hidden_bias"][0])
        eager_scores = np.tanh(eager_hidden @ weights["output_weight"] + weights["output_bias"][0])
        np.testing.assert_allclose(np.stack(holdout_scores), eager_scores, rtol=1e-10, atol=1e-12)
    finally:
        graph.close()
        if loaded_id:
            graph.unload_plugin(loaded_id)
    product_hash = hashlib.sha256((Path(row_request["output_dir"]) / "nodus-row-tool-product.json").read_bytes()).hexdigest()
    announce("holdout", completed)
    holdout_pnls, holdout_windows = window_results(
        holdout_rows, holdout_scores, stock_close, paper_arena, paper_mock,
        dataset_hash, product_hash, compiled=True)
    judge_db = paper_arena.connect(output / "qualification.sqlite")
    try:
        bot = paper_arena.register_bot(judge_db, OWNER, {
            "request_id": "trained_aapl_candidate", "name": "Trained AAPL candidate",
            "kind": "compiled", "source": {"ref": "product:" + product_hash}})
        qualification = paper_arena.record_historical_qualification(
            judge_db, OWNER, bot["id"], "cautious", dataset_hash, holdout_pnls,
            [window["sell_fills"] for window in holdout_windows],
            validation_passed=chosen["validation_passed"], margin=MARGIN)
    finally:
        judge_db.close()
    report = {"schema": "nodus-aapl-training-qualification-v1", "dataset_sha256": dataset_hash,
              "product_sha256": product_hash, "chronological_counts": {
                  "train": len(train_rows), "validation": len(validation_rows),
                  "holdout": len(holdout_rows)},
              "target": "realized next-three-bar AAPL price move after 0.4% estimated round-trip cost",
              "weights_file": "weights.npz", "chosen": chosen, "candidates": candidates,
              "holdout_windows": holdout_windows, "qualification": qualification,
              "adoption": adoption, "venue": "internal_mock"}
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if publish is not None:
        module, db, run_id = publish
        module.finish(db, run_id, status="passed_historical" if qualification["passed"]
                      else "unqualified", result={
                          "dataset_sha256": dataset_hash,
                          "validation_passed": chosen["validation_passed"],
                          "holdout_passed": qualification["passed"],
                          "holdout_mean_return_on_order_cap": qualification["mean_net_return"],
                          "holdout_window_pnl_usd": holdout_pnls})
    print(json.dumps({"chosen_seed": chosen["seed"], "chosen_epoch": chosen["epoch"],
                      "validation_passed": chosen["validation_passed"],
                      "holdout_passed": qualification["passed"],
                      "holdout_window_pnl_usd": holdout_pnls,
                      "holdout_mean_return_on_order_cap": qualification["mean_net_return"],
                      "report": str(output / "qualification.json")}, indent=2))


if __name__ == "__main__":
    main()
