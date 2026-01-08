#pragma once

#include <string_view>

#include "common/tensors/abstraction/tensor_backend.h"

namespace nodus::tensors {

// Registry does not take ownership; backend lifetime must outlive registry use.
void register_backend(TensorBackend* backend, bool make_default = false);
TensorBackend* find_backend(std::string_view name);
TensorBackend* default_backend();
void set_default_backend(TensorBackend* backend);

} // namespace nodus::tensors
