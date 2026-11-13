import csv
import statistics
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: run_math_scaling.py <binary-directory> <output-directory>")
root = Path(sys.argv[2]).resolve()
root.mkdir(parents=True, exist_ok=True)
bin_dir = Path(sys.argv[1]).resolve()
job = bin_dir / "nodus_job_timing_profile.exe"
registry = bin_dir / "nodus_type_registry_profile.exe"

math_rows = []
for repeat in range(3):
    for tables in (1, 2, 4, 8):
        scenario = "math_chain64" if tables == 1 else f"math_chain{tables}"
        for mode in ("sequential", "pooled", "free"):
            process = subprocess.run(
                [str(job), scenario, mode, "0", "2", "1", str(repeat)],
                cwd=bin_dir, text=True, capture_output=True, check=True,
            )
            data = [line.split(",") for line in process.stdout.splitlines()
                    if line.startswith(f"{repeat},") and len(line.split(",")) == 15]
            if len(data) != tables:
                raise RuntimeError(f"{scenario} {mode}: expected {tables} rows, got {len(data)}")
            elapsed = float(data[0][8])
            total = sum(int(line[9]) for line in data)
            module_us = statistics.median(float(line[10]) * 1e6 / int(line[11]) for line in data)
            math_rows.append((repeat, tables, mode, elapsed, total, total / elapsed, module_us))
            print("math", repeat, tables, mode, round(total / elapsed), round(module_us, 2), flush=True)

registry_rows = []
for repeat in range(3):
    for threads in (1, 2, 4, 8, 16):
        process = subprocess.run(
            [str(registry), str(threads)], cwd=bin_dir,
            text=True, capture_output=True, check=True,
        )
        line = process.stdout.strip().split(",")
        if len(line) != 4: raise RuntimeError(process.stdout)
        registry_rows.append((repeat, *line))
        print("registry", repeat, threads, round(float(line[-1])), flush=True)

for name, header, data in (
    ("nodus_math_scaling.csv", ("repeat", "tables", "mode", "elapsed_s", "completed_tables", "tables_per_s", "median_module_us"), math_rows),
    ("nodus_type_registry_scaling.csv", ("repeat", "threads", "elapsed_s", "get_calls", "get_calls_per_s"), registry_rows),
):
    with (root / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(data)
    print(root / name)
