#include "edge_tensor_fifo.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

bool fail(const char* message) {
    std::cerr << "edge_tensor_fifo_peek_test: " << message << '\n';
    return false;
}

bool seed_backend_context(gp_mem_backend_handle_t backend, const void* value, size_t bytes) {
    if (!gp_mem_backend_resize(backend, bytes)) return false;
    void* mapped = gp_mem_backend_map(backend);
    if (!mapped) return false;
    std::memcpy(mapped, value, bytes);
    gp_mem_backend_unmap(backend);
    return true;
}

bool test_pointer_peek_uses_ring_storage() {
    gp_mem_backend_handle_t backend = gp_mem_backend_create_host(0);
    if (!backend) return fail("could not create pointer host backend");

    EdgeTensorFifo fifo;
    fifo.set_backend(backend);
    fifo.configure({1}, 2, 0, sizeof(void*));
    constexpr uint64_t reader = 101;
    if (!fifo.subscribe(reader, false)) return fail("could not subscribe pointer reader");

    int expected_object = 7;
    int context_object = 11;
    void* expected = &expected_object;
    void* context_sentinel = &context_object;
    if (!seed_backend_context(backend, &context_sentinel, sizeof(context_sentinel))) {
        return fail("could not seed pointer backend context");
    }

    bool dropped = true;
    if (!fifo.push_ptr(1, 201, expected, &dropped) || dropped) {
        return fail("could not push pointer sample");
    }

    void* first_peek = nullptr;
    void* second_peek = nullptr;
    void* popped = nullptr;
    if (!fifo.peek_ptr(reader, &first_peek)) return fail("first pointer peek failed");
    if (!fifo.peek_ptr(reader, &second_peek)) return fail("second pointer peek failed");
    if (!fifo.pop_ptr(reader, &popped)) return fail("pointer pop after peeks failed");
    if (first_peek != expected || second_peek != expected || popped != expected) {
        return fail("pointer peeks and pop did not agree on ring storage");
    }

    gp_mem_backend_release(backend);
    return true;
}

bool test_byte_peek_uses_ring_storage() {
    gp_mem_backend_handle_t backend = gp_mem_backend_create_host(0);
    if (!backend) return fail("could not create byte host backend");

    EdgeTensorFifo fifo;
    fifo.set_backend(backend);
    fifo.configure({1}, 2, 0, sizeof(uint64_t));
    constexpr uint64_t reader = 102;
    if (!fifo.subscribe(reader, false)) return fail("could not subscribe byte reader");

    const uint64_t expected = UINT64_C(0x1122334455667788);
    const uint64_t context_sentinel = UINT64_C(0xaabbccddeeff0011);
    if (!seed_backend_context(backend, &context_sentinel, sizeof(context_sentinel))) {
        return fail("could not seed byte backend context");
    }

    bool dropped = true;
    if (!fifo.push(2, 202, &expected, sizeof(expected), &dropped) || dropped) {
        return fail("could not push byte sample");
    }

    uint64_t first_peek = 0;
    uint64_t second_peek = 0;
    uint64_t popped = 0;
    size_t written = 0;
    if (!fifo.peek(reader, &first_peek, sizeof(first_peek), written) || written != sizeof(first_peek)) {
        return fail("first byte peek failed");
    }
    if (!fifo.peek(reader, &second_peek, sizeof(second_peek), written) || written != sizeof(second_peek)) {
        return fail("second byte peek failed");
    }
    if (!fifo.pop(reader, &popped, sizeof(popped), written) || written != sizeof(popped)) {
        return fail("byte pop after peeks failed");
    }
    if (first_peek != expected || second_peek != expected || popped != expected) {
        return fail("byte peeks and pop did not agree on ring storage");
    }

    gp_mem_backend_release(backend);
    return true;
}

} // namespace

int main() {
    const bool pointer_ok = test_pointer_peek_uses_ring_storage();
    const bool byte_ok = test_byte_peek_uses_ring_storage();
    if (!pointer_ok || !byte_ok) return 1;
    std::cout << "edge_tensor_fifo_peek_test: PASS\n";
    return 0;
}
