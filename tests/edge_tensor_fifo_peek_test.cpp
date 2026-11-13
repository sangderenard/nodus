#include "edge_tensor_fifo.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

bool fail(const char* message) {
    std::cerr << "edge_tensor_fifo_peek_test: " << message << '\n';
    return false;
}

bool verify_pointer_sequence(
    EdgeTensorFifo& fifo,
    uint64_t reader,
    void* const* expected,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        bool dropped = true;
        if (!fifo.push_ptr(1, 201, expected[i], &dropped) || dropped) {
            return fail("could not push pointer sample");
        }

        void* first_peek = nullptr;
        void* second_peek = nullptr;
        void* popped = nullptr;
        if (!fifo.peek_ptr(reader, &first_peek)) return fail("first pointer peek failed");
        if (!fifo.peek_ptr(reader, &second_peek)) return fail("second pointer peek failed");
        if (!fifo.pop_ptr(reader, &popped)) return fail("pointer pop after peeks failed");
        if (first_peek != expected[i] || second_peek != expected[i] || popped != expected[i]) {
            return fail("pointer peeks and pop did not agree on ring storage");
        }
        if (fifo.peek_ptr(reader, &first_peek)) return fail("pointer peek succeeded after queue emptied");
        if (fifo.pop_ptr(reader, &popped)) return fail("pointer pop succeeded after queue emptied");
    }
    return true;
}

bool verify_byte_sequence(
    EdgeTensorFifo& fifo,
    uint64_t reader,
    const uint64_t* expected,
    size_t count
) {
    for (size_t i = 0; i < count; ++i) {
        bool dropped = true;
        if (!fifo.push(2, 202, &expected[i], sizeof(expected[i]), &dropped) || dropped) {
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
        if (first_peek != expected[i] || second_peek != expected[i] || popped != expected[i]) {
            return fail("byte peeks and pop did not agree on ring storage");
        }
        written = 99;
        if (fifo.peek(reader, &first_peek, sizeof(first_peek), written) || written != 0) {
            return fail("byte peek succeeded after queue emptied");
        }
        written = 99;
        if (fifo.pop(reader, &popped, sizeof(popped), written) || written != 0) {
            return fail("byte pop succeeded after queue emptied");
        }
    }
    return true;
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

    int expected_objects[] = {7, 8, 9};
    int context_object = 11;
    void* expected[] = {&expected_objects[0], &expected_objects[1], &expected_objects[2]};
    void* context_values[] = {expected[0], &context_object};
    if (!seed_backend_context(backend, context_values, sizeof(context_values))) {
        return fail("could not seed pointer backend context");
    }

    const bool sequence_ok = verify_pointer_sequence(fifo, reader, expected, 3);
    gp_mem_backend_release(backend);
    return sequence_ok;
}

bool test_byte_peek_uses_ring_storage() {
    gp_mem_backend_handle_t backend = gp_mem_backend_create_host(0);
    if (!backend) return fail("could not create byte host backend");

    EdgeTensorFifo fifo;
    fifo.set_backend(backend);
    fifo.configure({1}, 2, 0, sizeof(uint64_t));
    constexpr uint64_t reader = 102;
    if (!fifo.subscribe(reader, false)) return fail("could not subscribe byte reader");

    const uint64_t expected[] = {
        UINT64_C(0x1122334455667788),
        UINT64_C(0x2233445566778899),
        UINT64_C(0x33445566778899aa),
    };
    const uint64_t context_sentinel = UINT64_C(0xaabbccddeeff0011);
    const uint64_t context_values[] = {expected[0], context_sentinel};
    if (!seed_backend_context(backend, context_values, sizeof(context_values))) {
        return fail("could not seed byte backend context");
    }

    const bool sequence_ok = verify_byte_sequence(fifo, reader, expected, 3);
    gp_mem_backend_release(backend);
    return sequence_ok;
}

bool test_default_host_storage() {
    int pointer_objects[] = {21, 22, 23};
    void* pointers[] = {&pointer_objects[0], &pointer_objects[1], &pointer_objects[2]};
    EdgeTensorFifo pointer_fifo;
    pointer_fifo.configure({1}, 2, 0, sizeof(void*));
    constexpr uint64_t pointer_reader = 103;
    if (!pointer_fifo.subscribe(pointer_reader, false)) {
        return fail("could not subscribe default pointer reader");
    }
    if (!verify_pointer_sequence(pointer_fifo, pointer_reader, pointers, 3)) return false;

    const uint64_t bytes[] = {
        UINT64_C(0x445566778899aabb),
        UINT64_C(0x5566778899aabbcc),
        UINT64_C(0x66778899aabbccdd),
    };
    EdgeTensorFifo byte_fifo;
    byte_fifo.configure({1}, 2, 0, sizeof(uint64_t));
    constexpr uint64_t byte_reader = 104;
    if (!byte_fifo.subscribe(byte_reader, false)) {
        return fail("could not subscribe default byte reader");
    }
    return verify_byte_sequence(byte_fifo, byte_reader, bytes, 3);
}

} // namespace

int main() {
    const bool pointer_ok = test_pointer_peek_uses_ring_storage();
    const bool byte_ok = test_byte_peek_uses_ring_storage();
    const bool default_host_ok = test_default_host_storage();
    if (!pointer_ok || !byte_ok || !default_host_ok) return 1;
    std::cout << "edge_tensor_fifo_peek_test: PASS\n";
    return 0;
}
