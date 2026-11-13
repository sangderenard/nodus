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
Nodus `free-runtime-build` headless DLL from source `bd6003f`: adoption reported receipt validated,
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
layer widths requires a new compilation. The standalone row proof does not
submit orders.

## Historical AAPL mock replay

`tools/replay_tensor_network_stock.py` connects this compiled row to the
existing Meowdoku `market_data.history`, `paper_policies.features`,
`paper_arena`, and `paper_mock` modules. It requests AAPL plus linked SPY and
BTC/USD hourly bars using a configured paper data credential, then writes a
public-bar snapshot and its SHA-256 digest. No credential enters the snapshot.
An existing history JSON file can be supplied instead of fetching again.

Each decision uses AAPL, SPY and BTC bars no later than its AAPL decision bar.
The following AAPL bar's open is the mock execution mark. The final potentially
forming bar is excluded. Two feature rows feed each compiled Nodus graph step;
the final batch duplicates its last input only to satisfy the fixed `(2, 6)`
row shape and ignores that duplicate prediction.

The row weights are a fixed, untrained sign-of-three-bar-return probe. They
exercise buy, sell and hold routing; they do not establish a learned or
profitable strategy. The replay creates a new scratch `paper_arena.sqlite`,
registers its compiled bot and cautious $5/$50 strategy, and sends admitted
orders through `paper_mock.place`. The mock venue applies its recorded spread,
markup and fee. `replay.json` retains every feature vector, prediction,
decision/fill timestamp, order receipt, ledger mark and trainer event;
`equity.csv` is the chronological equity trace. The scratch trainer events
label the venue `internal_mock`; these are never Alpaca paper success labels.

Measured 2026-11-13 from Alpaca IEX AAPL history (2026-10-14 through
2026-11-13): 178 AAPL bars, 168 causal decisions, 84 compiled graph steps,
53 mock orders, and final marked equity $49.20473120524958 from a $50 cautious
budget. The $0.79526879475042 loss is one deliberately untrained probe path,
not a backtest claim about a trained model. The snapshot digest is
`7e0b4c2e5900ff9225bd50aeced975bc6e1aaa23a00fbbb103773a32b84a2843`.

Run with a fresh output directory and the server's restricted state secrets
directory. The account ID is the credential owner; it is not used as the mock
strategy owner, which exists only in the scratch database:

```powershell
$env:NODUS_HEADLESS_LIB='C:\Users\alber\nodus-build\free-runtime-build\Release\nodus_headless.dll'
python -m tools.replay_tensor_network_stock C:\Users\alber\nodus-aapl-replay-new --turing-root C:\dev\Powershell\turing --native-nodus-root C:\Users\alber\nodus-build\free-runtime-src --secrets-dir C:\dev\Powershell\srv\state\secrets --account-id 3
```
