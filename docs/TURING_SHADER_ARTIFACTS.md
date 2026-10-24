# Turing shader artifacts in Nodus

`tools/turing_shader_artifact.py` consumes shader source that Turing has
already emitted. It does not lower Python or SSA, and it does not use the
KernelIR `TranslationMatrix`.

Put the authored shader source, its Turing `.api.yaml`, and a JSON request in
one directory:

```json
{
  "schema": "nodus-turing-shader-artifact-v1",
  "source": "example.comp",
  "api": "example.api.yaml",
  "targets": ["glsl", "essl", "hlsl", "msl"]
}
```

Paths must be relative and cannot escape the request directory. Unknown fields,
targets, and schemas are errors. The API must be
`turing-compiled-program-api-v1`, contain a `turing.shader-component.v1`
component ABI, and agree with it on language, shader stage, and entrypoint.

Run:

```text
python tools/turing_shader_artifact.py path/to/request.json path/to/output
```

The output retains byte-identical source and API files, a validated SPIR-V
module, each requested SPIRV-Cross source, and
`shader-artifact.receipt.json`. The receipt includes SHA-256 hashes,
provenance, tool argv, exit codes, stdout, and stderr. External tools are
launched with an argv sequence and no shell.

Supported input languages are GLSL-family spellings published by Turing and
HLSL. Supported outputs are desktop GLSL, ESSL, HLSL, and MSL. WGSL input is
refused until a WGSL-to-SPIR-V compiler is configured; WGSL output is refused
because the installed SPIRV-Cross has no WGSL backend. These are explicit
diagnostics, not fallback translations.
