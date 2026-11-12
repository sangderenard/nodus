import csv
import json
import platform
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: run_job_timing.py <binary-directory> <output-directory>")
source_root = Path(__file__).resolve().parents[2]
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
exe = Path(sys.argv[1]).resolve() / "nodus_job_timing_profile.exe"
revision = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
).strip()
cases = []
for scenario in ("independent1", "independent8"):
    for work in (0, 10, 100, 1000, 5000):
        for mode in ("sequential", "pooled", "slip", "free"):
            cases.append((scenario, mode, work, 2, 1))
for scenario in ("math_add", "math_multiply", "math_divide", "math_modulo", "math_chain64", "math_chain8"):
    for mode in ("sequential", "pooled", "slip", "free"):
        cases.append((scenario, mode, 0, 2, 1))
for scenario in ("pipeline", "fanin"):
    for work in (0, 100, 1000):
        for capacity in (2, 16):
            for mode in ("sequential", "pooled", "free"):
                cases.append((scenario, mode, work, capacity, 1))
for work in (100, 1000):
    for mode in ("pooled", "free"):
        cases.append(("pipeline_other", mode, work, 2, 1))
for scenario in ("independent1", "independent8"):
    for work in (0, 100):
        for mode in ("pooled", "free"):
            cases.append((scenario, mode, work, 2, 0))

header = (
    "repeat", "scenario", "mode", "work_us", "capacity", "timed", "module", "role",
    "elapsed_s", "runs", "row_wall_s", "timed_runs", "start_s", "first_tick_s", "stop_s",
)
rows = []
for repeat in range(3):
    for number, case in enumerate(cases, 1):
        scenario, mode, work, capacity, timed = case
        process = subprocess.run(
            [str(exe), scenario, mode, str(work), str(capacity), str(timed), str(repeat)],
            cwd=exe.parent, text=True, capture_output=True,
        )
        if process.returncode:
            raise RuntimeError(f"{case} repeat={repeat}: {process.stderr[-2000:]} {process.stdout[-2000:]}")
        data = [line.split(",") for line in process.stdout.splitlines()
                if line.startswith(f"{repeat},") and len(line.split(",")) == len(header)]
        if not data:
            raise RuntimeError(f"missing measurements: {case}: {process.stdout[-1000:]}")
        rows.extend(data)
        if number % 10 == 0 or number == len(cases):
            print(f"repeat {repeat + 1}/3: {number}/{len(cases)} cases", flush=True)

out = root / "nodus_job_timing_results.csv"
with out.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream)
    writer.writerow(header)
    writer.writerows(rows)

groups = defaultdict(list)
for line in rows:
    item = dict(zip(header, line))
    key = (item["scenario"], item["mode"], int(item["work_us"]),
           int(item["capacity"]), int(item["timed"]), item["role"])
    groups[key].append(item)
summary = []
for key, values in sorted(groups.items()):
    scenario, mode, work, capacity, timed, role = key
    throughput = [sum(int(v["runs"]) for v in values if v["repeat"] == repeat) /
                  float(next(v["elapsed_s"] for v in values if v["repeat"] == repeat))
                  for repeat in ("0", "1", "2")]
    per_module = [float(v["row_wall_s"]) * 1e6 / int(v["timed_runs"])
                  for v in values if int(v["timed_runs"]) > 0]
    summary.append({
        "scenario": scenario, "mode": mode, "work_us": work,
        "capacity": capacity, "timed": timed, "role": role,
        "modules": len(values) // 3,
        "median_total_rows_per_s": statistics.median(throughput),
        "median_module_us": statistics.median(per_module) if per_module else None,
        "median_start_us": statistics.median(float(v["start_s"]) * 1e6 for v in values),
        "median_first_tick_us": statistics.median(float(v["first_tick_s"]) * 1e6 for v in values),
        "median_stop_us": statistics.median(float(v["stop_s"]) * 1e6 for v in values),
    })
summary_out = root / "nodus_job_timing_summary.json"
summary_out.write_text(json.dumps({
    "revision": revision,
    "platform": platform.platform(),
    "build": "Release, NODUS_ENABLE_TORCH=OFF",
    "run_seconds_per_case": 0.5,
    "repetitions": 3,
    "summary": summary,
}, indent=2), encoding="utf-8")
print(out)
print(summary_out)
