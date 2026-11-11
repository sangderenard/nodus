"""Host side of a cron prefix: perform the tasks the cron tool (include/cron_tool.h) asks for.

The cron tool is an engine of a contract prefix. For every run it writes
``<jobs_dir>/<contract id>.request.json`` and submits contract ``<id>``. A host on that prefix
takes the loan, reads the request by the contract's id, performs the named task and answers
with a status; the tool records the status and commits. This module is that host:

    host = CronHost(native, port=47615, host_id=7301, jobs_dir=..., handlers={"name": fn})
    ...
    host.close()

``native`` is the loaded ``contract_service_peer`` module (the caller chooses how to load
it). A handler takes the request (a dict: task, scheduled_for, fired_at, catch_up,
stands_for, reason) and returns ``DONE`` or ``UNCHANGED``, optionally with a sentence:
``return DONE`` or ``return UNCHANGED, "the manual has not changed"``. A handler that raises
is recorded as ``FAILED``; a task with no handler as ``UNKNOWN_TASK``. Nothing here retries:
the cron tool decides what runs again, from its ledger.

The outcome of each run is also written to ``<jobs_dir>/<contract id>.result.json``.
Request and result files older than ``keep_days`` are removed when a run is performed.
One thread owns the connection; a handler runs on a single worker so the socket is never
blocked by a slow task. No credentials belong in a request, and a request is data: a task
name selects one of the handlers this process registered, nothing else.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import select
import threading
import time

DONE, UNCHANGED = 1, 2
UNKNOWN_TASK, FAILED, REQUEST_MISSING = 16, 17, 18
STATUS_NAMES = {DONE: "done", UNCHANGED: "unchanged", UNKNOWN_TASK: "unknown task", FAILED: "failed",
                REQUEST_MISSING: "request missing"}


class CronHost:
    def __init__(self, native, *, port, host_id, jobs_dir, handlers, connect=None, log=print, keep_days=7):
        if not isinstance(handlers, dict) or not all(callable(fn) for fn in handlers.values()):
            raise TypeError("handlers must map task names to callables")
        self.native, self.port, self.host_id = native, int(port), int(host_id)
        self.jobs_dir = Path(jobs_dir)
        self.handlers, self.log, self.keep_seconds = dict(handlers), log, float(keep_days) * 86400.0
        self._connect = connect or (lambda: native.connect_loopback(self.port, native.ROLE_HOST, self.host_id))
        self.performed = []          # (contract id, task, status), newest last; bounded
        self.closed = False
        self._wake = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cron-task")
        self._thread = threading.Thread(target=self._run, name="cron-host", daemon=True)
        self._thread.start()

    def close(self):
        self.closed = True
        self._wake.set()
        self._thread.join(timeout=5)
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ---- one run ------------------------------------------------------------------------------------------
    def _prune(self):
        cutoff = time.time() - self.keep_seconds
        for pattern in ("*.request.json", "*.result.json"):
            for path in self.jobs_dir.glob(pattern):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    pass

    def _dispatch(self, loan) -> int:
        """Never raises: every outcome is a status, so one bad task cannot stop the host."""
        started, task, detail = time.time(), None, ""
        try:
            self._prune()
            try:
                request = json.loads((self.jobs_dir / f"{loan.contract}.request.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                request = None
            if not isinstance(request, dict) or not isinstance(request.get("task"), str):
                status = REQUEST_MISSING
            else:
                task = request["task"]
                handler = self.handlers.get(task)
                if handler is None:
                    status = UNKNOWN_TASK
                else:
                    try:
                        answer = handler(request)
                        status, detail = answer if isinstance(answer, tuple) else (answer, "")
                        if status not in (DONE, UNCHANGED):
                            status, detail = FAILED, f"the handler returned {status!r}, not DONE or UNCHANGED"
                    except Exception as error:   # the task's failure is its status
                        status, detail = FAILED, f"{type(error).__name__}: {error}"
            record = {"schema": "nodus.cron.result.v1", "contract": loan.contract, "task": task, "status": status,
                      "status_name": STATUS_NAMES[status], "detail": str(detail)[:2000],
                      "started": started, "finished": time.time()}
            try:
                (self.jobs_dir / f"{loan.contract}.result.json").write_text(json.dumps(record), encoding="utf-8")
            except OSError:
                pass
            self.performed.append((loan.contract, task, status))
            del self.performed[:-64]
            self.log(f"[cron] {task or '?'}: {STATUS_NAMES[status]}" + (f" ({detail})" if detail else ""))
            return status
        except Exception as error:               # belt and braces: still a status
            self.log(f"[cron] run of contract {getattr(loan, 'contract', '?')} failed: {type(error).__name__}: {error}")
            return FAILED

    # ---- the connection -----------------------------------------------------------------------------------
    def _run(self):
        peer, retry_at = None, 0.0
        while not self.closed:
            if peer is not None and (peer.closed or peer.error is not None):
                self.log(f"[cron] host peer ended ({peer.error!r}); reconnecting")
                peer.close()
                peer = None
            if peer is None and time.monotonic() >= retry_at:
                try:
                    peer = self.native.ServicePeer(self._connect(), self._executor, self._dispatch,
                                                   lambda loan, accepted: None, announce=False)
                except OSError as error:
                    retry_at = time.monotonic() + 5.0
                    self.log(f"[cron] port {self.port} is not reachable: {error}")
            if peer is None:
                self._wake.wait(0.5)
                continue
            peer.poll()
            try:
                select.select([peer.connection], [], [], 0.1)
            except (OSError, ValueError):
                time.sleep(0.1)
        if peer is not None:
            peer.close()
