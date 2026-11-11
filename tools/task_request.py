"""Ask a task prefix to perform one named task now: one contract, one tick of the table, one answer.

The cron tool (include/cron_tool.h) is the engine that asks on a schedule. This is the engine for a single event:
anything that can reach the prefix's contract port writes the request file the hosts already understand and submits
one contract. The market row offers it, a host (tools/cron_host.py) performs the named task, the worker row routes
the answer back, and the engine commits. Nothing is scheduled and nothing is retried.

    outcome = submit(native, port=47615, engine_id=7202, jobs_dir=..., task="system.broadcast",
                     fields={"text": "...", "audience": "admin"}, timeout=5.0)
    outcome == {"outcome": "done", "status": 1, "contract": 1881...}

``native`` is the loaded ``contract_service_peer`` module. ``outcome`` is one of ``done`` (a host answered; ``status``
is its status), ``timeout`` (no answer in time; the task may still be performed later, the contract stays in the
table until a host takes it or the engine's registration ends), ``closed`` (the port closed the connection) or
``unreachable`` (no port). The request file is ``<jobs_dir>/<contract id>.request.json``; its ``task`` selects one
handler a host registered, and nothing in it is a credential.
"""
from __future__ import annotations

import json
from pathlib import Path
import secrets
import select
import time

SCHEMA = "nodus.task.request.v1"


class _Producer:
    def __init__(self, contract):
        self.contract, self.admitted_seen, self.result = contract, False, None

    def request(self):
        return None

    def admitted(self, identifier):
        if identifier == self.contract[0]:
            self.admitted_seen = True

    def route(self, result):
        if result[0] != self.contract[0]:
            return 0                                  # not ours: hold, never commit a foreign result
        self.result = tuple(result)
        return 1


def result(jobs_dir, contract) -> dict | None:
    """The host's record of a finished task (``<jobs_dir>/<contract>.result.json``): status, detail and, when the task
    has one, its ``output``. None when there is no readable result."""
    try:
        return json.loads((Path(jobs_dir) / f"{int(contract)}.result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def contract_id(now: float | None = None) -> int:
    """Unique beside the cron tool's ids, which are (seconds << 20) | a small serial: these set the high serial bit."""
    return (int(time.time() if now is None else now) << 20) | 0x80000 | secrets.randbits(19)


def submit(native, *, port, engine_id, jobs_dir, task, fields=None, timeout=5.0, connect=None) -> dict:
    identifier = contract_id()
    request = {"schema": SCHEMA, "contract": identifier, "task": str(task), "fired_at": int(time.time()),
               **{key: value for key, value in (fields or {}).items() if key not in ("schema", "contract", "task", "fired_at")}}
    jobs = Path(jobs_dir)
    jobs.mkdir(parents=True, exist_ok=True)
    (jobs / f"{identifier}.request.json").write_text(json.dumps(request), encoding="utf-8")
    contract = (identifier, int(engine_id), 1, 0)
    producer = _Producer(contract)
    try:
        connection = (connect or (lambda: native.connect_loopback(int(port), native.ROLE_ENGINE, int(engine_id))))()
    except OSError as error:
        return {"outcome": "unreachable", "status": None, "contract": identifier, "error": str(error)}
    peer = native.EnginePeer(connection, producer)
    try:
        if not peer.submit(contract):
            return {"outcome": "closed", "status": None, "contract": identifier}
        end = time.monotonic() + float(timeout)
        while producer.result is None:
            peer.poll()
            if peer.error is not None or peer.closed:
                return {"outcome": "closed", "status": None, "contract": identifier, "error": repr(peer.error)}
            if producer.result is not None:
                break
            if time.monotonic() >= end:
                return {"outcome": "timeout", "status": None, "contract": identifier, "admitted": producer.admitted_seen}
            try:
                select.select([peer.connection], [], [], 0.05)
            except (OSError, ValueError):
                time.sleep(0.05)
        for _ in range(20):                           # let the commit decision leave before the connection closes
            peer.poll()
            if not peer.outgoing:
                break
            time.sleep(0.01)
        return {"outcome": "done", "status": int(producer.result[7]), "contract": identifier}
    finally:
        peer.close()
