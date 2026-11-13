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

## Bounded training and historical qualification

`tools/fetch_aapl_training_history.py` creates a fresh one-year public-bar
snapshot from Alpaca IEX AAPL/SPY and BTC/USD hourly bars. BTC is fetched in
30-day windows to stay within the historical endpoint's page limit. On
2026-11-13 the snapshot contained 1,997 AAPL, 2,200 SPY, and 8,759 BTC bars;
SHA-256 `8be31249c828863a831c9c90236d943f7c025b6eb5290443e33aaaa618d8d7f9`.
It contains no credential.

`tools/train_aapl_qualified_row.py` trains the exact 6-to-4-to-3 `tanh` row
with Turing `AbstractTensor` autograd. It uses causal AAPL/SPY/BTC features
and a next-three-bar price-move target after an estimated 0.4% round-trip
cost. Chronological 60/20/20 train, validation, and untouched holdout splits
avoid using future bars in training. Three seeds and four checkpoints per seed
are bounded in advance; validation selects the checkpoint. The selected
weights are compiled through the existing Nodus row crossing, whose scores
are checked against eager inference before holdout mock replay. Each of three
validation and holdout windows starts a fresh cautious mock ledger. Admission
needs positive net P&L and a sell fill in every holdout window plus mean net
P&L of at least 1% of the $5 maximum per-order capital. This return
denominator is the order cap, not the $50 account budget.

The 2026-11-13 run selected seed 1, epoch 80. Validation failed: window net
P&Ls were $0.88925, -$0.49292, and $0.32989, with sell-fill counts 0/1/0.
Untouched holdout also failed: $0.12577, $0.17845, and -$0.09379, with sell
fills 0/0/2. The mean holdout P&L was about $0.07014, or 1.40% of the $5
order cap, but one window lost money and two had no sell fill. The candidate
remains unqualified. The holdout has now been inspected; further model
changes require a fresh future holdout for an honest admission decision.
These are internal mock fills, not Alpaca paper executions or evidence of
future profit.

Run with a fresh output directory and the server's restricted state secrets
directory. The account ID is the credential owner; it is not used as the mock
strategy owner, which exists only in the scratch database:

```powershell
$env:NODUS_HEADLESS_LIB='C:\Users\alber\nodus-build\free-runtime-build\Release\nodus_headless.dll'
python -m tools.replay_tensor_network_stock C:\Users\alber\nodus-aapl-replay-new --turing-root C:\dev\Powershell\turing --native-nodus-root C:\Users\alber\nodus-build\free-runtime-src --secrets-dir C:\dev\Powershell\srv\state\secrets --account-id 3
```
