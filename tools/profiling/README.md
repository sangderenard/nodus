# Nodus table and row timing profile

This profile runs real Nodus `ThreadManager` table contracts, GraphRuntime rows,
tool stacks, and (for connected cases) table-edge FIFOs. The math cases use the
builtin Add, Multiply, Divide, and Modulo tool rows. The 64-Add case keeps 65
values on the real stack and verifies the final result. Pipeline and fan-in
cases verify FIFO values and order. Independent cases use a plugin row with
0, 10, 100, 1000, or 5000 microseconds of work.

The CSV reports end-to-end completed module iterations per second, per-module
wall time from `ThreadManager::get_module_timing`, and start, first-tick, and
stop times. It does not attribute time to individual rows within a multi-row
table. The direct registry microbenchmark is a diagnostic, not a substitute
for table measurements.

Build outside the checkout on Windows:

```powershell
$source = (Get-Location).Path
$build = 'C:\Users\alber\nodus-build\free-runtime-build'
$output = 'C:\Users\alber\nodus-build\profile\final'
cmake -S $source -B $build -DNODUS_ENABLE_TORCH=OFF -DCMAKE_PROJECT_INCLUDE="$source/tools/profiling/nodus_profile_targets.cmake"
cmake --build $build --config Release --target nodus_job_timing_profile nodus_type_registry_profile
python tools/profiling/run_job_timing.py "$build/Release" $output
python tools/profiling/run_math_scaling.py "$build/Release" $output
```

`run_job_timing.py` executes 112 configurations three times for 0.5 seconds
each. Outputs stay in the selected user directory. The benchmark executables
also accept one case at a time; run without arguments to see their usage.
