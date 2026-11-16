# Interpreted row refusal propagation

## Scope

Base: `6f3183e49d74d2392d889bfde740bafb72a8f680`.
This closes the generated-wrapper portion of item C in
`NODUS_MEOWDOKU_SERVICE_BACKBONE_AUDIT_2026-11-07.md`, not the other service
cutover recommendations.

`Interpreter::execute_tensor_stack` returns false and restores input tensors
on refusal. `ThreadManager` already checks `ToolStackContext.succeeded` and
stops before following Output rows when it is false. The generated Python
wrapper previously printed the error but left that flag at its default true.
The wrapper now records false before checking its stack and assigns the actual
interpreter result on invocation. Successful invocation resets a previously
false context. No interpreter, scheduler, FIFO, tensor, ABI or physics policy
is replaced.

Already-built interpreted wrappers must be regenerated and rebuilt to receive
this change. Existing compiled plugin binaries are not modified by the source
change.

## Verification actually run

In an isolated Linux cloud checkout, using the installed Python 3.12 headers
and library (no package installation):

```
c++ -std=c++20 -fpermissive -ffunction-sections -fdata-sections -Iinclude -Isrc $(python3-config --includes) tests/python_row_tool_test.cpp src/python_row_tool.cpp src/mem_backend_host.cpp src/spirv_translation.cpp -Wl,--gc-sections $(python3-config --embed --ldflags) -o /tmp/nodus_python_row_test
LD_LIBRARY_PATH=$(python3-config --prefix)/lib /tmp/nodus_python_row_test
```

- Before the fix, the added generated-source assertion fails because the
  wrapper contains no `c.succeeded = false;` (exit 134).
- After the fix, the complete existing `python_row_tool_test` reports OK. This
  exercises actual embedded CPython primitive calls/refusals and generated
  source assertions, not loaded generated tensor plugins.
- Generated a wrapper with the real `write_plugin_source`, then compiled it
  with `c++ -std=c++20 -Iinclude -Isrc -fsyntax-only`: passed.
- Built and ran `python_row_plugin_integration_test` on Linux: it explicitly
  reports SKIP because plugin loading is Windows-only. Its extended Windows
  test checks missing-stack refusal, successful result status, Python
  exception status and restoration of the exact original tensor handles.
- `git diff --check`: passed.

`-fpermissive` is needed for the pre-existing extern/static `g_cpu_vtable`
diagnostic. An attempted standalone tensor-linked wrapper harness was blocked
by existing `ThreadPool::Options` default-argument/default-member-initializer
errors; excluding that source left required thread-pool symbols unresolved.
These were not patched. CMake is unavailable here.

## Limits and next gate

The new loaded-plugin regression and a full scheduler graph with a failing
Python row followed by Output have not run. Run the existing Windows
`python_row_plugin_integration_test` after rebuilding, then the graph-level
acceptance recommended by the audit. No claim of full service readiness,
GPU validation, complete suite, merge or deployment. No user-computer or active
Studio changes.

The review checked current default heads and open proposals across the five
requested repositories. Nodus open PRs 60, 52 and 36 do not touch these files.
Active uncommitted Studio development and its distinct local state were
intentionally left alone.
