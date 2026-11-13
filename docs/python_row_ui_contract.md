# Python row UI contract boundary

Python row descriptors carry durable tool-definition metadata; a placed row
will carry graph-instance bindings.  Keeping these separate lets one retained
Python source and ABI descriptor be deployed through CPython, C, LLVM, or
Fortran without baking a particular canvas wiring into the tool.

## Descriptor-owned input presentation

Each input has a stable machine name and primitive type.  It may additionally
declare:

- `label`: the human-readable form label; an empty value means use the machine
  name.
- `form_control`: an open presentation hint such as `number` or `integer`; an
  empty value means derive the control from the primitive type.
- `form_required`: whether a form must supply the value when form input is in
  use.  It defaults to `true`.

The text descriptor stores non-default presentation metadata as:

```text
INPUT "gain" float64
INPUT_UI "gain" "Gain" "number" REQUIRED
```

These fields are metadata only.  They do not alter execution, port count,
stack order, compilation, or the `AbstractTensor` transport ABI.

## Reserved placed-row binding contract

The eventual UI binding belongs to a placed row instance, not to the reusable
tool descriptor.  Its semantic contract is reserved as follows:

- Inputs default to stack delivery in declaration order.
- Every input may select an explicit stack index instead of its declaration
  index.
- Stack delivery may be disabled independently for an input.
- When stack delivery is disabled, that input is supplied by its edge port.
- The UI must not silently consume both sources for one input.  A binding has
  one active source: `stack` or `edge`.

No instance-binding storage, canvas mutation, edge routing, or editor control
is implemented here.  When that work begins, it should consume this contract
through the existing row/canvas persistence authority rather than adding
routing behavior to the Python interpreter or Turing compiler adapters.


## Site-owned compiler artifacts

The site bridge supplies `site_artifact_context` in a row-tool request, populated
from its authenticated owner and existing workspace lease, never from submitted
client metadata:

```json
{
  "site_artifact_context": {
    "owner_id": 7,
    "workspace_id": "0123456789abcdef0123456789abcdef",
    "job_id": "divider-17",
    "job_directory": "C:/user-state/content/workspaces/compile-jobs/divider-17"
  }
}
```

`owner_id` is a positive integer, `workspace_id` is the existing 32-hex lease ID,
`job_id` is the divider ID, and `job_directory` is absolute. The native host's
existing `<jobs_dir>/<id>` output convention is unchanged. `output_dir` must
resolve to this exact directory. Repository outputs (including worktree `.git`
files), escaping existing symlinks/junctions, and composition directories outside
the job are rejected before compilation. This context is ownership metadata,
not an authentication token. The service/bridge must enforce site-only admission;
its optional absence remains supported for existing injected compiler tests and
legacy internal callers, not an authorized route around the site.

The dedicated job subprocess routes Turing's existing host SSA, identity spill,
SymPy, native law, law module, accelerator, GLSL and usage-trace directory knobs
under `compiler-work`. It also routes TEMP/TMP/TMPDIR, disables import bytecode
writes and restores environment, tempfile cache, and bytecode settings even on
failure. Compiler cwd changes remain inside existing adapter compiler phases.
The matching Turing change adds `TURING_COMPILER_ARTIFACT_ROOT`, used for the
previously repository-fixed compiler foundation, tensor kernel, and native shell
defaults. Deploy both sides together; unpatched Turing ignores that override.

`site-artifact-lifecycle.json` is written before the compiler is imported, retains
owner/workspace/job identity, start/end timestamps and completed/uncompiled/failed
status, and records exception details on failure. A killed process may leave
`running`, which must be reconciled against the site's job ledger. Successful
product manifests also retain the site context and enumerate compiler-work files.
No cleanup is automatic. The site's existing workspace accounting/retention gate
owns lifetime decisions; deleting or expiring a contract is not evidence that its
files are safe to erase.

These controls route trusted compiler outputs; they are not an OS sandbox for
arbitrary authored Python or a defense against concurrent filesystem mutation.
Verification uses injected fake compilers and pure path tests only. No real
compiler, native build, live service, or live database was exercised.


Site compilation also captures Python stdout/stderr, process file descriptors
1/2, and Windows standard handles into `compiler-work/stdout.log` and
`compiler-work/stderr.log`. Inheriting tool children use those job-owned logs.
The capture begins before the Turing boundary import, flushes before artifact
hashing, preserves failure tracebacks, and restores streams/descriptors/handles
before the CLI prints its final manifest path. Logs append on an explicit rerun.
Children that intentionally select other streams or explicit files remain
responsible for those destinations. Plain Python printing-child tests verify
inheritance without invoking any compiler.
