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
