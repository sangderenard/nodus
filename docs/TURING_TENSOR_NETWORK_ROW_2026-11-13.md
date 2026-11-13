# Compiled AbstractTensor network as a Nodus tool row

`examples/tensor_network_row.py` is a two-layer, batched inference network:
six inputs, four hidden units, three outputs, with `tanh` after each dense
layer. Weights and biases are input tensors, so a trainer can publish new
parameters without recompiling the row. This proof covers inference only.

`tools/compile_tensor_network_row.py` uses the existing
`turing_row_tool_compiler` and Turing's public `lower_ast_source_to_ssa` entry.
It declares the exact float64 rank-two shapes in both the Nodus buffer columns
and Turing's `program_abi`. Omitting `shape` from the latter reproduced a
compiler refusal: `matmul ... has no settled dtype`. With the shapes declared,
the C emitter produced a loadable `nodus-turing-row-tool-product-v1` with six
buffers and a native DLL. The source, extraction contract, repository SSA,
buffer ABI descriptor, and adoption receipt remain in the output directory.

`tools/prove_tensor_network_row.py` calls the existing
`compile_row_product_graph` crossing. The Nodus module has Input, PluginTool,
and Output rows. It creates Nodus-owned tensor handles for every input, steps
the graph, reads the output handle, and compares it with the same eager
AbstractTensor network.

Measured on Windows, 2026-11-13, against Turing `9d96173c` and the installed
Nodus `free-runtime-build` headless DLL: adoption reported receipt validated,
wrapper imported, wrapper staged, plugin loaded; output shape `(2, 3)`;
maximum absolute difference `5.551115123125783e-17`. This is a local Nodus
graph proof, not a live Meowdoku row or a trained market model.

The installed Nodus build initially lacked `canvas_tables.lib`, so plugin
linking failed with unresolved tensor/memory ABI symbols. Building its existing
`canvas_tables` shared target and making the paired build directory available
as `<native-nodus-root>/build/Release` resolved that linker error. Use a
headless DLL, headers, and import libraries from the same Nodus build.

Run from the Nodus repository root with the paired native build available:

```powershell
$env:NODUS_HEADLESS_LIB='C:\Users\alber\nodus-build\free-runtime-build\Release\nodus_headless.dll'
python -m tools.prove_tensor_network_row C:\Users\alber\nodus-tensor-network-row-proof --turing-root C:\dev\Powershell\turing --native-nodus-root C:\Users\alber\nodus-build\free-runtime-src
```

The command creates a fresh product and module library under the chosen output
directory. Fixed shapes are part of this first row contract; changing batch or
layer widths requires a new compilation. No order submission or trading state
is connected to it.
