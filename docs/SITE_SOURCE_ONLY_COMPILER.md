# Site source-only compiler admission

The default tensor row request and its permissions are unchanged. The separate,
opt-in request to `tools/turing_row_tool_compiler.py` contains exactly:

- `unstable_source_only: true` (a JSON boolean)
- `source_names`: nonempty, unique authored function or qualified method names
- exactly one of `source_text` or `source_path`
- `backend`: `c`, `llvm`, or `fortran`
- `output_dir`, explicit `extraction_contract`, and `turing_root`
- `site_artifact_context`: existing owner/workspace/job/directory metadata

Site authenticates the owner and checks the unstable permission before dispatch.
The metadata is required here, but is not itself authentication. Do not expose
this trusted adapter as a service. Row ABI fields, wrapping, composition,
`product_name`, `allow_uncompiled`, and unknown request fields are rejected in
this mode. There is no invented tensor ABI or annotation rewriting.

Validation precedes artifact writes. Output must be empty (an existing empty
host-created directory is allowed); stale products are not deleted or reused.
Authored text, including CRLF, is retained unchanged. The existing isolated
compile-host calls `examples.meowdoku.source_divider._attempt_main`, which uses
its existing partition/emission implementation. The attempt spec supplies the
retained explicit extraction contract. `run_attempt` and its deletion path are
not used. No native backend build is requested.

The product remains `nodus-row-tool-product.json`, with schema
`nodus-turing-source-only-product-v1`, `unstable_source_only: true`, exact
`source_names`, `state: emitted-not-built` or `refused`, and the divider's
unaltered `attempt` record. `loadable` is false; `binary`, `buffer_abi`, and
`adoption_receipt` are null. Existing Site artifact accounting and diagnostics
retain all job outputs and lifecycle state. Compiler refusal details are taken
from the actual divider entry (type, stage, error), not replaced with a tensor
ABI admission error. A refusal exits the CLI unsuccessfully while retaining its
manifest and attempt.

Native `gp_turing_row_tool_compile_only` returns 3 for source-only emission;
`CompileStatus::EmittedNotBuilt` is 4. Neither adoption-enabled nor
auto-adoption-disabled hosts treat this as a compiled/runnable row. A forged
source manifest marked loadable, or accompanied by an adoption receipt, fails.
`compile_and_load` still requires return 1, and the adopter only accepts the
original row-product schema. Existing native status numbers do not change. Before the existing-product fast
path, source-only or malformed-mode requests are refused if a row receipt already
exists (including escaped field names and duplicate-mode fields). Legacy row
requests and a single explicit `unstable_source_only: false` without
`source_names` keep their restart behavior.
The native source-only manifest check requires `buffer_abi: null` and explicit
`loadable: false`.

Tests use only fake divider boundaries and stub compiler scripts. The compiler
itself, live Site service, backend build/link, and GPU render are not run by
these focused tests. A Site-authorized integration exercise remains a separate
step, including checking that Turing's optional extraction-contract argument is
present and that the Site recorder understands host status 4 and manifest
attempt diagnostics.
