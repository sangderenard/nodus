// tensor_handle_dup is failure-atomic (review 924fc2a7, research case 6393486c).
//
// A refused dup must leave the frame exactly as it was found: logical depth, top type and the original handle. Cases:
//   1. one tensor entry, a backend whose reads succeed and whose writes fail: refused, frame unchanged
//   2. a full frame (one tensor entry fills it): refused, frame unchanged
//   3. a non-tensor top: refused, frame unchanged
//   4. one tensor entry with room: succeeds, two entries naming the same handle
#define NODUS_PLUGIN_COMPOSITE 1
#include "module_library/source/tools/tensor_handle_dup.cpp"
#include "mem_backend.h"
#include "src/mem_backend_registry_internal.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// A backend whose reads come from `bytes` and whose writes are refused.
struct WriteRefusingBackend {
    uint64_t magic = 0;  // never the CPU backend's tag: the vtable comes from the registry
    std::vector<uint8_t> bytes;
};
int refusing_copy_to(gp_mem_backend_handle_t, size_t, const void*, size_t) { return 0; }
int refusing_copy_from(gp_mem_backend_handle_t h, size_t offset, void* dst, size_t n) {
    auto* b = static_cast<WriteRefusingBackend*>(h);
    if (offset + n > b->bytes.size()) return 0;
    std::memcpy(dst, b->bytes.data() + offset, n);
    return 1;
}

struct Snapshot {
    size_t depth = 0;
    ValueTypeId top = kInvalidValueTypeId;
    uint64_t handle = 0;
};
Snapshot snap(const RawStackFrame& f) {
    Snapshot s;
    s.depth = f.byte_count;
    raw_stack_peek_type(f, s.top);
    nodus::tensors::AbstractTensorHandle h{};
    if (raw_stack_peek_abstract_tensor(f, h)) s.handle = h.id;
    return s;
}
bool same(const Snapshot& a, const Snapshot& b) { return a.depth == b.depth && a.top == b.top && a.handle == b.handle; }

// raw_stack_destroy_frame deletes a heap frame; these frames are locals, so release their storage only.
void release_frame(RawStackFrame& frame) {
    raw_stack_free_mask(frame);
    if (frame.backend) gp_mem_backend_release(frame.backend);
    frame.backend = nullptr;
}

bool run_dup(RawStackFrame& frame) {
    ITool* tool = NODUS_PLUGIN_FACTORY_NAME();
    ToolStackContext ctx;
    ctx.stack.raw = &frame;
    tool->execute_stack(ctx);
    NODUS_PLUGIN_DESTROY_NAME(tool);
    return ctx.succeeded;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const ValueTypeId tensor_tid = ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
    const size_t entry = ValueTypeRegistry::global().get(tensor_tid)->size;
    const nodus::tensors::AbstractTensorHandle original{0x1234ABCDull};

    {  // 1. reads succeed, the first write fails
        RawStackFrame frame;
        raw_stack_init_frame(frame, entry * 4);
        raw_stack_push_abstract_tensor(frame, original);  // through the host backend
        WriteRefusingBackend refusing;
        refusing.bytes.resize(entry * 4);
        gp_mem_backend_get_vtable(frame.backend)->copy_from_backend(frame.backend, 0, refusing.bytes.data(), entry);
        static gp_mem_backend_vtable_t vt{};
        vt.copy_to_backend = &refusing_copy_to;
        vt.copy_from_backend = &refusing_copy_from;
        gp_mem_backend_handle_t host = frame.backend;
        nodus::mem_backend_internal::register_external_handle(&refusing, GP_MEM_BACKEND_CUSTOM, &vt, refusing.bytes.size(), 0);
        frame.backend = &refusing;
        const Snapshot before = snap(frame);
        check(before.handle == original.id && before.depth == entry, "case 1 setup: one entry, read through the refusing backend");
        const bool ok = run_dup(frame);
        check(!ok, "case 1: a failed first write is refused");
        check(same(before, snap(frame)), "case 1: depth, top type and original handle unchanged");
        nodus::mem_backend_internal::unregister_external_handle(&refusing);
        frame.backend = host;
        release_frame(frame);
    }
    {  // 2. full frame
        RawStackFrame frame;
        raw_stack_init_frame(frame, entry);
        raw_stack_push_abstract_tensor(frame, original);
        const Snapshot before = snap(frame);
        check(!run_dup(frame), "case 2: a full frame is refused");
        check(same(before, snap(frame)), "case 2: frame unchanged");
        release_frame(frame);
    }
    {  // 3. non-tensor top
        RawStackFrame frame;
        raw_stack_init_frame(frame, 64);
        const double v = 3.5;
        raw_stack_push_typed(frame, &v, ValueTypeRegistry::global().builtin(VT_FLOAT64));
        const Snapshot before = snap(frame);
        check(!run_dup(frame), "case 3: a non-tensor top is refused");
        check(same(before, snap(frame)), "case 3: frame unchanged");
        release_frame(frame);
    }
    {  // 4. success
        RawStackFrame frame;
        raw_stack_init_frame(frame, entry * 4);
        raw_stack_push_abstract_tensor(frame, original);
        check(run_dup(frame), "case 4: room for the second entry succeeds");
        nodus::tensors::AbstractTensorHandle a{}, b{};
        const bool two = raw_stack_pop_abstract_tensor(frame, a) && raw_stack_pop_abstract_tensor(frame, b) && frame.byte_count == 0;
        check(two && a.id == original.id && b.id == original.id, "case 4: two entries naming the original handle");
        release_frame(frame);
    }
    std::printf("%s\n", failures ? "FAILED" : "PASSED");
    return failures ? 1 : 0;
}
