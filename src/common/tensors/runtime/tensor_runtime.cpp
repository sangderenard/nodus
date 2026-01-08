#include "common/tensors/runtime/tensor_runtime.h"

#include "common/tensors/abstraction/in_memory_backend.h"

namespace nodus::tensors {

Runtime::Runtime(const RuntimeConfig& config) {
    reset(config);
}

void Runtime::reset(const RuntimeConfig& /*config*/) {
    // Placeholder for runtime configuration once the tensor system is wired in.
    register_in_memory_backend(true);
}

Runtime& runtime_singleton() {
    static Runtime runtime{};
    return runtime;
}

} // namespace nodus::tensors
