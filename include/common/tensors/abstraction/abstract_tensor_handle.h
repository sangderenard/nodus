#pragma once

#include <cstdint>

namespace nodus::tensors {

struct AbstractTensorHandle {
    uint64_t id = 0;
};

inline bool operator==(const AbstractTensorHandle& a, const AbstractTensorHandle& b) {
    return a.id == b.id;
}

inline bool operator!=(const AbstractTensorHandle& a, const AbstractTensorHandle& b) {
    return a.id != b.id;
}

inline bool abstract_tensor_handle_is_valid(AbstractTensorHandle handle) {
    return handle.id != 0;
}

} // namespace nodus::tensors
