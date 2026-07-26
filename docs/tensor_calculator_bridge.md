# Tensor Calculator bridge

Nodus's in-memory backend and Turing's C backend share the standalone
`tensor-calculator` operator engine. Nodus does not copy its tensor payloads
into the calculator: `InMemoryCalculator` maps an existing dense F32/F64
tensor, binds that pointer as an external calculator handle, and keeps the map
alive through synchronous execution or asynchronous job completion.

The bridge is intentionally outside `InMemoryBackend`'s allocator internals.
It therefore does not duplicate arena ownership, alter leases, or mix Nodus's
graph scheduler with the calculator's optional worker queue.

Use `CalculatorInstruction` arrays for equal-shaped elementwise regions:

```cpp
const CalculatorInstruction program[] = {
    {TC_MUL, &temporary, &input, nullptr, 2.0, true, false},
    {TC_ADD, &output, &temporary, nullptr, 1.0, true, false},
};
InMemoryCalculator::instance().execute(program);
```

`submit()` returns a move-only `CalculatorJob`. The job owns all mappings and
calculator bindings until it is destroyed; destruction waits for unfinished
work before unmapping. Callers must still respect ordinary Nodus tensor
lifetime and mutation ordering.

Registry mutation and synchronous prepared execution are lock-free by
contract. Mutexes exist only in the optional asynchronous queue and completion
wait. Environment configuration:

- `NODUS_CALCULATOR_WORKERS`
- `NODUS_CALCULATOR_QUEUE_CAPACITY`
- `NODUS_CALCULATOR_ASYNC_THRESHOLD`

The bridge compile-check target is `tensor_calculator_bridge_compile_check`.
The full runtime test currently reaches Nodus's pre-existing
`nodus_tensor_core` link blocker: unresolved `ThreadPool`/`JobBatch` symbols
from `tensor_math`. The bridge itself compiles cleanly independently of that
substrate issue.
