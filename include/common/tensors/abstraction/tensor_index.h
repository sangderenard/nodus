#pragma once

#include "common/tensors/abstraction/abstract_tensor_handle.h"

#include <cstdint>
#include <variant>
#include <vector>

namespace nodus::tensors {

struct TensorSlice {
    int64_t start = 0;
    int64_t stop = 0;
    int64_t step = 1;
    bool is_all = true; // true means ":" (full dimension)

    static TensorSlice all() { return TensorSlice{}; }
};

using TensorIndex = std::variant<int64_t, TensorSlice, AbstractTensorHandle>;

struct TensorIndexSpec {
    std::vector<TensorIndex> dims;
};

} // namespace nodus::tensors
