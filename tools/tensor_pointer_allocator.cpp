#include "tensor_pointer_allocator.h"
#include <new>

int tensor_pointer_allocator_cycle(RawStackFrame& frame) {
    // Pop existing pointer if present
    torch::Tensor* oldptr = nullptr;
    int popped = raw_stack_pop_torch(frame, oldptr);
    if (popped == 1 && oldptr) {
        // Caller asked for deletion: free heap tensor
        delete oldptr;
        oldptr = nullptr;
    }

    // Allocate a new tensor on the heap. For demo purposes create an empty tensor.
    // In practice constructor tool would create a tensor backed by allocator memory.
    torch::Tensor* newptr = nullptr;
    try {
        newptr = new torch::Tensor(torch::empty({0}, torch::kFloat32));
    } catch (...) {
        // Allocation failed; push nothing and return 0
        return 0;
    }

    // Push pointer back onto stack
    int pushed = raw_stack_push_torch(frame, newptr);
    if (!pushed) {
        // push failed: cleanup
        delete newptr;
        return 0;
    }
    return 1;
}
