"""Continuously train AAPL weights and expose the active Nodus inference canvas.

The historical holdout is deliberately untouched here. Repeated validation is
feedback for training, never a qualification verdict or an order permission.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nodus_graph import Graph, compile_row_product_graph
from tools.compile_tensor_network_row import request
from tools.live_market_frames import current_row, matured_rows
from tools.replay_tensor_network_stock import STOCK, _series, causal_rows, infer
from tools.run_canvas import ConsoleServer
from tools.train_aapl_qualified_row import PolicyNetwork, scaled, target_for, window_results

_stop = False
_PARAMETERS = ("w1", "b1", "w2", "b2")
_SHAPES = ((6, 4), (1, 4), (4, 3), (1, 3))


def _request_stop(_signum, _frame) -> None:
    global _stop
    _stop = True


def _native_pair() -> None:
    """Refuse a mixed tensor core before importing AbstractTensor."""
    headless = os.environ.get("NODUS_HEADLESS_TEST_DLL") or os.environ.get("NODUS_HEADLESS_LIB")
    core = os.environ.get("NODUS_TENSOR_CORE")
    if not headless or not core:
        raise ValueError("set NODUS_HEADLESS_LIB and NODUS_TENSOR_CORE to one matched build")
    headless_path, core_path = Path(headless).resolve(), Path(core).resolve()
    if (not headless_path.is_file() or not core_path.is_file()
            or headless_path.parent != core_path.parent):
        raise ValueError("Nodus headless and tensor core DLLs must exist in the same build directory")


def _load_weights(network: PolicyNetwork, path: Path, dataset_sha256: str) -> int:
    if not path.is_file():
        return 0
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["dataset_sha256"].item()) != dataset_sha256:
            raise ValueError("checkpoint dataset digest differs from the configured history")
        epoch = int(saved["epoch"].item())
        if epoch < 0:
            raise ValueError("checkpoint epoch is invalid")
        arrays = [np.asarray(saved[name], dtype=np.float64) for name in _PARAMETERS]
    if any(value.shape != shape or not np.isfinite(value).all()
           for value, shape in zip(arrays, _SHAPES)):
        raise ValueError("checkpoint tensor shape or value is invalid")
    AT = network.AT
    with AT.use_backend("numpy"), AT.autograd.no_grad():
        for parameter, value in zip(network.parameters, arrays):
            AT.copyto(parameter, AT.tensor(value))
    return epoch


def _save_weights(network: PolicyNetwork, path: Path, dataset_sha256: str, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    arrays = {name: np.asarray(parameter.numpy(), dtype=np.float64)
              for name, parameter in zip(_PARAMETERS, network.parameters)}
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays, epoch=np.int64(epoch), dataset_sha256=np.asarray(dataset_sha256))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _observations(path: Path, current: dict | None = None) -> list[dict]:
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"] if path.is_file() else []
    if not isinstance(frames, list):
        raise ValueError("live observations are invalid")
    if current is None:
        return frames
    if any(frame["t"] == current["decision_bar"] for frame in frames):
        return frames
    frames = (frames + [{"t": current["decision_bar"],
                         "features": current["features"].tolist()}])[-4096:]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump({"frames": frames}, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canvas", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=60)
    parser.add_argument("--console-port", type=int, required=True)
    parser.add_argument("--console-token-file", type=Path, required=True)
    parser.add_argument("--history-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turing-root", type=Path, required=True)
    parser.add_argument("--native-nodus-root", type=Path, required=True)
    parser.add_argument("--publish-db", type=Path, required=True)
    parser.add_argument("--worker", default="nodus-training")
    parser.add_argument("--autosave-seconds", type=float, default=60)
    parser.add_argument("--validation-epochs", type=int, default=100)
    parser.add_argument("--report-seconds", type=float, default=2)
    parser.add_argument("--market-owner", default="admin")
    parser.add_argument("--accounts-db", type=Path)
    parser.add_argument("--secrets-dir", type=Path)
    parser.add_argument("--market-poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if (args.rate <= 0 or args.autosave_seconds <= 0 or args.validation_epochs < 1
            or args.report_seconds <= 0 or args.market_poll_seconds < 15):
        parser.error("rate, autosave interval, validation epochs and report interval must be positive")
    if (args.accounts_db is None) != (args.secrets_dir is None):
        parser.error("provide both --accounts-db and --secrets-dir for live market intake")
    _native_pair()
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, _request_stop)
    args.output.mkdir(parents=True, exist_ok=True)
    args.canvas.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.turing_root.resolve()))
    sys.path.insert(0, str(args.turing_root.resolve() / "examples"))
    from src.common.tensors import AbstractTensor as AT
    from meowdoku import (alpaca_credentials, market_data, paper_arena, paper_mock,
                         paper_policies, qualification_runs)

    market_owner = None
    if args.accounts_db is not None:
        with sqlite3.connect(args.accounts_db) as accounts:
            found = accounts.execute("SELECT id FROM users WHERE username=? COLLATE NOCASE",
                                     (args.market_owner,)).fetchone()
        if found and alpaca_credentials.configured(args.secrets_dir, found[0]):
            market_owner = found[0]

    history_bytes = args.history_file.read_bytes()
    history = json.loads(history_bytes)
    dataset_sha256 = hashlib.sha256(history_bytes).hexdigest()
    stock = _series(history, STOCK)
    rows = scaled(causal_rows(history, paper_policies))
    cutoff, validation_end = int(len(rows) * .60), int(len(rows) * .80)
    if cutoff < 50 or validation_end - cutoff < 30:
        raise ValueError("history is too short for continuous training and validation")
    train_rows, validation_rows = rows[:cutoff - 4], rows[cutoff:validation_end]
    index = {bar["t"]: i for i, bar in enumerate(stock)}
    close = {bar["t"]: Decimal(str(bar["c"])) for bar in stock}
    x = np.stack([row["features"] for row in train_rows])
    targets = np.stack([target_for(row, index, stock) for row in train_rows])
    network = PolicyNetwork(AT, seed=0)
    checkpoint = args.output / "latest.npz"
    epoch = _load_weights(network, checkpoint, dataset_sha256)
    observation_path = args.output / "live-observations.json"
    observations = _observations(observation_path)
    live_x = np.empty((0, 6), dtype=np.float64)
    live_targets = np.empty((0, 3), dtype=np.float64)
    live_count = 0
    live_status = "waiting" if market_owner is not None else "unavailable"
    live_at = None
    rng = np.random.default_rng(0x4E4F445553 + epoch)
    with qualification_runs.publishing(args.publish_db, market="AAPL + SPY + BTC/USD",
                                       model="Nodus continuous 6→4→3", worker=args.worker,
                                       planned_epochs=0) as (db, run_id):
        graph = Graph()
        loaded_id = ""
        console = None
        try:
            product_output = args.output / "product"
            product_output.mkdir(exist_ok=True)
            row_request = request(args.output, args.turing_root.resolve(), Path(__file__).resolve().parents[1])
            request_path = args.output / "request.json"
            request_path.write_text(json.dumps(row_request, indent=2), encoding="utf-8")
            qualification_runs.progress(db, run_id, phase="compiling", completed_epochs=epoch)
            _, adoption = compile_row_product_graph(
                graph, request_path, stable_id=0x4E45544E,
                python_executable=sys.executable, turing_root=args.turing_root,
                product_dir=row_request["output_dir"], nodus_root=args.native_nodus_root,
                module_library_root=args.output / "module-library")
            loaded_id = str(adoption["loaded_tool_id"])
            graph.save(args.canvas)
            token = args.console_token_file.read_text(encoding="utf-8").splitlines()[0]
            if not token:
                raise ValueError("console token file is empty")
            console = ConsoleServer(token, args.canvas.name, lambda: epoch, str(args.canvas))
            port = console.start(args.console_port)
            console.set_context(args.rate, [])
            print(json.dumps({"canvas": str(args.canvas), "contract_ports": [],
                              "pid": os.getpid(), "console_port": port}), flush=True)
            qualification_runs.progress(db, run_id, phase="training", completed_epochs=epoch)
            last_report = last_save = time.monotonic()
            last_market_poll = float("-inf")
            while not _stop:
                now = time.monotonic()
                if market_owner is not None and now - last_market_poll >= args.market_poll_seconds:
                    last_market_poll = now
                    try:
                        recent = market_data.history(args.secrets_dir, market_owner, STOCK)
                        latest = market_data.now(args.secrets_dir, market_owner, STOCK)
                        current, live_status = current_row(
                            recent, latest, paper_policies, now=datetime.now(timezone.utc))
                        if current is not None:
                            live_at = current["decision_bar"]
                            observations = _observations(observation_path, current)
                            infer(graph, [current, current], network.weights())
                        if observations:
                            live_x, live_targets, live_count = matured_rows(
                                recent, observations, now=datetime.now(timezone.utc))
                    except (market_data.MarketError, alpaca_credentials.CredentialError,
                            ValueError, KeyError, OSError):
                        live_status = "unavailable"
                mixed = live_status == "current" and live_count > 0
                if mixed:
                    history_ids = rng.choice(len(x), size=32, replace=False)
                    live_ids = rng.choice(live_count, size=32, replace=live_count < 32)
                    batch_x = np.concatenate((x[history_ids], live_x[live_ids]))
                    batch_targets = np.concatenate((targets[history_ids], live_targets[live_ids]))
                    loss = network.step(batch_x, batch_targets, learning_rate=0.01)
                else:
                    loss = network.step(x, targets)
                epoch += 1
                console.service(graph)
                now = time.monotonic()
                if now - last_report >= args.report_seconds:
                    qualification_runs.live_state(db, run_id, source=live_status,
                                                  observed_at=live_at, matured_frames=live_count,
                                                  historical_batch=32 if mixed else len(x),
                                                  live_batch=32 if mixed else 0)
                if epoch % args.validation_epochs == 0:
                    qualification_runs.progress(db, run_id, phase="validation", completed_epochs=epoch,
                                                epoch=epoch, loss=loss)
                    predictions = infer(graph, validation_rows, network.weights())
                    pnls, windows = window_results(validation_rows, predictions, close,
                                                    paper_arena, paper_mock, dataset_sha256, "0" * 64)
                    returns = [Decimal(value) / Decimal("5") for value in pnls]
                    mean = sum(returns) / len(returns)
                    passed = all(value > 0 for value in returns) and mean >= Decimal("0.01") and all(
                        window["sell_fills"] >= 1 for window in windows)
                    qualification_runs.progress(db, run_id, phase="training", completed_epochs=epoch,
                                                epoch=epoch, loss=loss, validation_mean=float(mean),
                                                validation_passed=passed)
                    last_report = time.monotonic()
                elif now - last_report >= args.report_seconds:
                    # Execute the current weights on the same live Nodus row
                    # between full validation passes; this keeps the canvas
                    # active without repeatedly consulting the validation set.
                    infer(graph, train_rows[-2:], network.weights())
                    qualification_runs.progress(db, run_id, phase="training", completed_epochs=epoch,
                                                epoch=epoch, loss=loss)
                    last_report = now
                if now - last_save >= args.autosave_seconds:
                    qualification_runs.progress(db, run_id, phase="autosaving", completed_epochs=epoch)
                    _save_weights(network, checkpoint, dataset_sha256, epoch)
                    qualification_runs.progress(db, run_id, phase="training", completed_epochs=epoch)
                    last_save = time.monotonic()
            qualification_runs.progress(db, run_id, phase="autosaving", completed_epochs=epoch)
            _save_weights(network, checkpoint, dataset_sha256, epoch)
            qualification_runs.finish(db, run_id, status="stopped", result={"dataset_sha256": dataset_sha256})
            return 0
        finally:
            if console is not None:
                console.close()
            graph.close()
            if loaded_id:
                graph.unload_plugin(loaded_id)


if __name__ == "__main__":
    raise SystemExit(main())
