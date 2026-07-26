# In-memory tensor calculator

`InMemoryCalculator` is a persistent execution view over Nodus
`AbstractTensor` objects. It does not own a second tensor registry, copy or
rebind tensor payloads, or implement its own arithmetic.

Every `CalculatorInstruction` carries a generated
`nodus::ops::CanonicalOp`. Execution delegates exclusively to
`tensor_elementwise_unary`, `tensor_elementwise_binary`, or
`tensor_elementwise_scalar` in Nodus TensorMath. Those functions are the
in-memory CPU semantics shared by direct tensor programs, tools, and the
calculator.

```cpp
const CalculatorInstruction program[] = {
    {nodus::ops::CanonicalOp::MUL,
     &temporary, &input, nullptr, 2.0, true, false},
    {nodus::ops::CanonicalOp::ADD,
     &output, &temporary, nullptr, 1.0, true, false},
};
InMemoryCalculator::instance().execute(program);
```

The current `submit()` path deliberately completes inline. TensorMath may
parallelize an individual operation through Nodus's shared tensor thread pool,
but the calculator introduces no private worker queue, mutex, handle table, or
lifetime regime. A future free-running KPN executor can schedule the same
instruction arrays without changing their math or tensor ownership.

The old standalone `tensor-calculator` scalar switch is not linked into
`nodus_tensor_core`. Python can still use Turing's C backend on its own; when
Nodus ingestion is selected, Nodus owns and executes the tensor program.

Verification targets:

- `tensor_calculator_bridge_test`
- `nodus_precomposed_calculator_benchmark`
