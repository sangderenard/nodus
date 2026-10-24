# Eigen operator lane audit — 2026-10-24

## Finding

Nodus does not have a second, Eigen-owned tensor object system parallel to
`AbstractTensor`. The established implementation is better factored: Eigen is
one execution engine behind the existing `AbstractTensorHandle` row currency.
The graph, raw stack frame, tensor arena, ownership rules, and plugin ABI remain
the same as the in-memory engine.

Four similarly named pieces must not be conflated:

1. **Eigen vocabulary tools — real and executable.**
   `vocabulary_actualizer.cpp` emits standalone
   `eigen.abstract_tensor.<op>` plugins. They pop/push
   `VT_ABSTRACT_TENSOR` handles, map their payload through the tensor C ABI,
   and evaluate an Eigen array expression.
2. **`VT_EIGEN_TENSOR` / `eigen_tensor_view` — only a registered type.**
   No producer, consumer, stack helper, serializer, or test uses it. It is not
   an established alternative object representation.
3. **The `kernel_eigen` runner — a separate CPU reference path.**
   Despite its historical name, its own source states that it is a scalar
   KernelIR interpreter, deliberately not an Eigen-expression engine.
4. **`GenericObject` — a real schema/dispatch shell, but not automatically
   wired to Eigen.** Its test demonstrates that one class schema can be bound
   to two manually supplied callables called “inmemory” and “eigen.” It does
   not resolve a schema method's `body_reference` into the generated Eigen
   tool registry or execute an Eigen operator.

## What is established

The generated Eigen vocabulary currently covers 40 F32 unary/binary
elementwise operations:

```text
add sub mul truediv pow mod floordiv
sqrt exp log neg abs round trunc floor ceil
isfinite isnan isinf logical_not
less less_equal greater greater_equal equal not_equal maximum minimum
sin cos tan asin acos atan sinh cosh tanh asinh acosh atanh
```

The catalog gate admits 50 in-memory unary/binary operations. The ten admitted
operations without an Eigen expression are:

```text
sign invert bitand bitor bitxor shl shr logical_and logical_or sigmoid
```

`vocab_tool_dll_test` proves the lane end to end on Windows: it actualizes and
builds every generated plugin DLL, loads its plugin factory, executes it on
real tensor handles, checks an independent `<cmath>` reference, compares each
Eigen result with the in-memory engine, and proves the comparator rejects an
intentional add-versus-subtract mismatch. The measured manifest contains 50
in-memory and 40 Eigen tools; 80 per-engine/op substitution certificates are
retained for the 40 operations covered by the independent reference battery.

The focused measured gate was:

```text
vocab_tool_dll_test  passed
eigen_kernel_test    passed
generic_object_test  passed
```

## What is not established

The Eigen vocabulary implementation is deliberately narrow:

- F32 only;
- non-empty tensors only;
- unary and binary flat elementwise evaluation only;
- storage is obtained through the default Nodus tensor ABI and treated as a
  contiguous `Eigen::ArrayXf`;
- binary admission checks equal element count, not the full
  `AbstractTensor` rank/shape/stride/broadcast plan;
- no scalar broadcasting, general broadcasting, non-contiguous view
  semantics, reductions, contractions/matmul, remaps, generators, ordering,
  linalg, FFT, autograd, or the composed roughly-300-operation Tier-2 surface;
- no automatic graph-definition/backend-selection policy; callers must name
  the qualified Eigen plugin tool explicitly; and
- no automatic `GenericObject` binder from method `body_reference` to a
  selected backend group's registered tool.

Accordingly, Eigen is an admitted substitute for the tested F32 elementwise
slice. It is not yet a complete alternative implementation of AbstractTensor
semantics, and `eigen_tensor_view` should not be presented as if it were one.

## Priorities if this lane resumes

1. Preserve the one-currency design: extend Eigen behind
   `AbstractTensorHandle`; do not create Eigen-owned graph values.
2. Before adding operations, make the engine boundary enforce the same
   shape/stride/broadcast contract as the in-memory semantic anchor and add
   refusal/differential tests for mismatched shapes, broadcast inputs, views,
   empty tensors, and non-F32 dtypes.
3. If object dispatch is needed, implement the missing binder from a
   `GenericObject` method's `body_reference` plus backend choice to an actual
   registered tool. The shell itself is already adequate.
4. Widen dtype and operator coverage only from measured needs, keeping an
   independent reference and cross-engine certificate for every admitted op.
5. Resolve misleading names separately: either give `VT_EIGEN_TENSOR` a real
   contract and users or retire it; treat `kernel_eigen` as the scalar
   KernelIR reference it currently is rather than evidence of Eigen tensor
   execution.
