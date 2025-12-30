#include "value_types.h"
#include <vector>
#include <cassert>
#include <iostream>

int main() {
    // Allocate and initialize source frame
    RawStackFrame* src = new RawStackFrame();
    if (!raw_stack_init_frame(*src, 64)) {
        std::cerr << "failed to init src frame\n";
        return 2;
    }

    // Push two int32 values (a then b)
    int32_t a = 0x12345678;
    int32_t b = 0x0badf00d;
    ValueTypeId tid_int32 = ValueTypeRegistry::global().builtin(VT_INT32);
    if (tid_int32 == kInvalidValueTypeId) {
        std::cerr << "int32 type id not available\n";
        raw_stack_destroy_frame(src);
        return 2;
    }
    int ok = raw_stack_push_typed(*src, &a, tid_int32);
    if (ok != 1) { std::cerr << "push a failed\n"; raw_stack_destroy_frame(src); return 2; }
    ok = raw_stack_push_typed(*src, &b, tid_int32);
    if (ok != 1) { std::cerr << "push b failed\n"; raw_stack_destroy_frame(src); return 2; }

    // Inspect frame state
    std::cerr << "src byte_count=" << src->byte_count << " byte_capacity=" << src->byte_capacity << "\n";
    size_t sz = sizeof(uint64_t) + src->byte_count + src->byte_count * sizeof(ValueTypeId);
    std::vector<uint8_t> buf(sz);
    size_t written = 0;
    if (!gp_raw_stack_frame_serialize(src, buf.data(), buf.size(), &written)) {
        std::cerr << "serialize failed\n";
        raw_stack_destroy_frame(src);
        return 5;
    }
    std::cerr << "serialized size need=" << sz << " wrote=" << written << "\n";
    std::cerr << "serialized size need=" << sz << " wrote=" << written << "\n";
    if (written >= 8) {
        uint64_t bc = 0;
        std::memcpy(&bc, buf.data(), sizeof(bc));
        std::cerr << "serialized byte_count=" << bc << "\n";
    }

    // Deserialize into a fresh frame
    RawStackFrame* dst = new RawStackFrame();
    if (!gp_raw_stack_frame_deserialize(dst, buf.data(), written)) {
        std::cerr << "deserialize failed\n";
        raw_stack_destroy_frame(src);
        return 3;
    }

    // Pop back and validate values (LIFO)
    int32_t out1 = 0;
    int32_t out2 = 0;
    if (!raw_stack_pop_typed(*dst, &out1, tid_int32)) {
        std::cerr << "pop out1 failed\n";
        raw_stack_destroy_frame(src);
        raw_stack_destroy_frame(dst);
        return 4;
    }
    if (!raw_stack_pop_typed(*dst, &out2, tid_int32)) {
        std::cerr << "pop out2 failed\n";
        raw_stack_destroy_frame(src);
        raw_stack_destroy_frame(dst);
        return 4;
    }
    if (out1 != b || out2 != a) {
        std::cerr << "mismatch: " << out1 << ", " << out2 << " expected " << b << ", " << a << "\n";
        raw_stack_destroy_frame(src);
        raw_stack_destroy_frame(dst);
        return 4;
    }

    raw_stack_destroy_frame(src);
    raw_stack_destroy_frame(dst);
    std::cout << "frame_serialization_test: PASS\n";
    return 0;
}
