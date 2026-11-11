# Nodus cron tool (2026-11-11)

Owner's request: "we should make a cron tool in nodus that can perform server scheduled tasks and reason about ones
that would've been missed through some gap."

## What it is

`include/cron_tool.h` (`nodus::cron::CronTool`) is a contract-negotiator ENGINE: `negotiator.add_engine(cron.producer())`.
It is the mirror of the Turing compile host, which is a HOST. It needs no row of its own: the existing
`<prefix>.worker` row asks every engine for work once per worker turn. When a task is due the tool writes
`<jobs_dir>/<contract id>.request.json` and offers contract `<id>` (payload 0, as the compile host's convention). Any
host on the prefix performs the task and answers with a status; the tool records it and commits. The tool performs
nothing itself and blocks nothing; it starts no thread.

`include/cron_schedule.h` is the pure core, header-only, standard library only: 5-field cron expressions in UTC, the
table, the missed-run reasoning (`decide`), and the ledger file. Time is always passed in.

`tools/cron_host.py` is the host side for Python services: a `ServicePeer` host with a table of handlers by task name.

## Files in the tool's directory

| File | Written by | Holds |
|---|---|---|
| `cron.table` | people | one task per line: `<name> <m> <h> <dom> <mon> <dow> [missed=skip\|once\|all] [max_late=S] [max_catch_up=N]` |
| `cron.ledger` | the tool, atomically | per task: handled-up-to, last started, last finished, last status, fired, skipped; and `alive` |
| `cron.log` | the tool | one line per decision with its reason |
| `<id>.request.json` | the tool | task, scheduled_for, fired_at, catch_up, stands_for, reason |
| `<id>.result.json` | `cron_host.py` | status, status_name, detail, started, finished |

## Missed runs

`decide(task, last_due, now)` takes every occurrence in `(last_due, now]`. One decided within 45 s of its time is on
time; older ones are missed. The tool does not ask why it was not looking: a stopped process, an earlier run still
out, and a prefix with no host are the same thing to it.

- `once` (default): one run stands for the gap. If an on-time occurrence is in the same decision it carries the count.
- `skip`: none run.
- `all`: each runs, oldest first, the most recent `max_catch_up` of them.
- `max_late`: a missed occurrence older than this never runs.
- A run started and never finished (`last_started > last_finished`) is run again once at the next attach, unless the
  policy is skip or it is too late.
- A task new to the table starts from when the tool first sees it.
- The ledger's `alive` stamp, refreshed every 30 s, is how a gap is measured; an attach more than 120 s after it logs
  the gap.

One run is out at a time. A decision that fires is written to the ledger when its contract is ADMITTED; with `all`
only the first run of a decision is taken and the ledger advances just past it, so the rest are decided again. The
tool retires its own committed contract on its next turn, so it never fills the negotiator.

## ABI and host

- `nodus_headless_attach_cron(prefix, engine_id, jobs_dir)` -> 1/0, `nodus_headless_detach_cron`, and
  `nodus_headless_cron_stats(prefix, engine_id, out8)` = fired, finished, skipped, interrupted, tasks, gap_seconds,
  in_flight, last_status. Releasing a contract pair removes its cron engines first.
- Python: `Graph.attach_cron / detach_cron / cron_stats`. A library built before the tool still loads; only these
  three are refused, by name.
- `tools/run_canvas.py --cron PREFIX:ENGINE_ID:JOBS_DIR` (PREFIX must also be acquired). A cron that cannot start (an
  unreadable table, an older library) is reported on stderr and in the first JSON line (`"cron": [...]`) and left
  off; the canvas host and its other services start regardless.

## Checks

- `cron_schedule_test`: parsing and refusals, next/last occurrence incl. leap day and the either-day rule, the table,
  every policy, an 80-day and a 200-day every-minute gap, the ledger.
- `cron_tool_test`: the real `ContractNegotiator`, a fake host, a fake clock: a new schedule fires nothing; an on-time
  run with its request file, status and retired contract; a three-day gap under once / skip / all with a cap; a run
  left unfinished and run again by the next process; a request offered unchanged until admitted.
- `tests/test_cron_tool.py`: `run_canvas.py --cron` on a temporary canvas, `cron_host.py` performing an every-minute
  task (waits for the next minute); a malformed table leaves the tool off and the host running.

## Not done

- Time comes from the system clock through `CronConfig::now`. When the privileged clock port exists it is bound there.
- The table is read at attach only; changing it takes a restart of the canvas host.
- The console protocol does not list cron tools or their stats yet.
- A task with no host stays out indefinitely (its loan is re-offered); there is no give-up time.
- The live service's library (`build-service`) has not been rebuilt, so the live canvas host has no cron yet.
