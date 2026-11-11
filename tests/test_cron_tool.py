"""The cron tool on a real canvas host: run_canvas --cron fires a due task and tools/cron_host.py performs it.

Needs the built headless library (as tests/test_run_canvas.py does). The schedule is every minute, so the test
waits for the next minute boundary: up to about 70 s.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import contract_service_peer as native  # noqa: E402
import cron_host  # noqa: E402

PREFIX, PORT = "py.cron", 47695


class CronToolOnACanvasHost(unittest.TestCase):
    def test_a_due_task_is_fired_performed_and_recorded(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = pathlib.Path(scratch)
            blank, canvas, jobs = scratch / "blank.canvas", scratch / "cron.canvas", scratch / "jobs"
            jobs.mkdir()
            blank.write_text("CANVAS V4\n1 1 56\nOFFSET 0 0\n\n", encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "tools" / "add_contract_port.py"), "--canvas", str(blank),
                            "--out", str(canvas), "--prefix", PREFIX, "--port", str(PORT), "--loan", "3600"],
                           check=True, capture_output=True, timeout=120)
            (jobs / "cron.table").write_text("# every minute\nbeat * * * * *\nnobody 0 0 1 1 * missed=skip\n", encoding="utf-8")
            # A table that cannot be read is never guessed: the tool stays off and says so, and the host still runs.
            bad = scratch / "bad"
            bad.mkdir()
            (bad / "cron.table").write_text("beat 61 * * * *\n", encoding="utf-8")
            refused = subprocess.run([sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas),
                                      "--acquire", f"{PREFIX}:4:3600", "--cron", f"{PREFIX}:9:{bad}", "--ticks", "5"],
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(refused.returncode, 0)
            self.assertIn("cron attach refused", refused.stderr)
            first = json.loads(refused.stdout.splitlines()[0])
            self.assertEqual([(c["attached"], c["prefix"]) for c in first["cron"]], [(False, PREFIX)])
            self.assertIn("attach refused", (bad / "cron.log").read_text(encoding="utf-8"))

            host = subprocess.Popen([sys.executable, str(ROOT / "tools" / "run_canvas.py"), "--canvas", str(canvas),
                                     "--rate", "60", "--acquire", f"{PREFIX}:4:3600", "--cron", f"{PREFIX}:9:{jobs}",
                                     "--ticks", "6000"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            seen = []
            performer = None
            try:
                for line in host.stdout:
                    if "contract_ports" in line:
                        break
                performer = cron_host.CronHost(
                    native, port=PORT, host_id=5, jobs_dir=jobs, log=lambda text: None,
                    handlers={"beat": lambda request: (seen.append(request) or (cron_host.DONE, "beat performed"))})
                end = time.time() + 80
                while not seen and time.time() < end:
                    time.sleep(0.25)
                self.assertTrue(seen, "the every-minute task did not fire within 80 s")
                request = seen[0]
                self.assertEqual(request["task"], "beat")
                self.assertEqual(request["scheduled_for"] % 60, 0)
                self.assertFalse(request["catch_up"])
                end = time.time() + 10
                ledger = ""
                while "beat" not in ledger or " 1 1 0" not in ledger.split("task beat")[-1].splitlines()[0]:
                    self.assertLess(time.time(), end, f"the ledger never recorded the finished run: {ledger!r}")
                    time.sleep(0.2)
                    ledger = (jobs / "cron.ledger").read_text(encoding="utf-8")
                result = json.loads((jobs / f"{request['contract']}.result.json").read_text(encoding="utf-8"))
                self.assertEqual((result["status"], result["status_name"], result["detail"]), (1, "done", "beat performed"))
                self.assertIn("finished with status 1", (jobs / "cron.log").read_text(encoding="utf-8"))
            finally:
                if performer is not None:
                    performer.close()
                host.kill()
                host.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
