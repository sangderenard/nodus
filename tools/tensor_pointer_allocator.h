#pragma once

#include "value_types.h"
#if defined(__has_include)
# if __has_include(<torch/torch.h>)
#  include <torch/torch.h>
#  define NODUS_HAVE_LIBTORCH 1
# else
#  define NODUS_HAVE_LIBTORCH 0
namespace torch { class Tensor; }
# endif
#else
# include <torch/torch.h>
# define NODUS_HAVE_LIBTORCH 1
#endif

// Simple helper: operate on a RawStackFrame of torch pointers.
// Behavior: try to pop an existing pointer (if present), delete it.
// Then allocate a new heap torch::Tensor (empty) and push its pointer.
// Returns 1 on success, 0 on failure (e.g., stack full for push).
int tensor_pointer_allocator_cycle(RawStackFrame& frame);
