#pragma once

#include "value_types.h"
#include <torch/torch.h>

// Simple helper: operate on a RawStackFrame of torch pointers.
// Behavior: try to pop an existing pointer (if present), delete it.
// Then allocate a new heap torch::Tensor (empty) and push its pointer.
// Returns 1 on success, 0 on failure (e.g., stack full for push).
int tensor_pointer_allocator_cycle(RawStackFrame& frame);
