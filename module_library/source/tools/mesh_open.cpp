// mesh_open: a row whose one duty is the mesh's open step (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 4).
// "NMV1" -> "NMO1" | writer lineage | generation | readable u8 | snapshot: opened with any key set this node still
// holds, or not readable (empty) for a node that is not a reader.
// The crypto is the headless image's (src/headless/nodus_mesh_crypto.cpp, the pinned OpenSSL); this row moves one U8
// tensor in and one out. It reads its input and never keeps or destroys it (a Borrower: in a table with a Returns row,
// the input goes back to the mint). A refusal puts the input back and reports succeeded = false; a result of nothing
// (a record applied) pushes nothing.
#include "tool_api.h"
#include "value_types.h"
#include "nodus_headless_abi.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace nodus::mesh_rows {

class MeshOpen final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "nodus.mesh.open"; }
    const char* name_cstr() const noexcept override { return "Mesh open"; }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    ToolTensorRole tensor_role() const override { return ToolTensorRole::Borrower; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t i) const override {
        if (i == 0) return ToolPortSpec{ToolPortKind::Argument, 1};
        if (i == 1) return ToolPortSpec{ToolPortKind::Return, 1};
        return ToolPortSpec{};
    }
    bool serialize(OutputArchive&) const override { return true; }       // no settings: the keys are the process's
    bool deserialize(InputArchive&) override { return true; }

    void execute_stack(ToolStackContext& c) override {
        if (!c.stack.raw) return;
        RawStackFrame& stack = *c.stack.raw;
        nodus::tensors::AbstractTensorHandle input{};
        if (!raw_stack_pop_abstract_tensor(stack, input)) { c.succeeded = false; return; }
        NodusTensorDesc d{};
        std::vector<uint8_t> in;
        bool ok = input.id && nodus_tensor_describe(input.id, &d) == NODUS_OK && d.dtype == NODUS_DTYPE_U8;
        if (ok) {
            in.resize(static_cast<size_t>(d.total_bytes));
            ok = !d.total_bytes || nodus_tensor_read(input.id, 0, in.data(), in.size()) == static_cast<int64_t>(in.size());
        }
        std::vector<uint8_t> out;
        size_t written = 0;
        if (ok) {
            out.resize(static_cast<size_t>(nodus_mesh_bound(NODUS_MESH_OPEN, in.data(), in.size())));
            ok = nodus_mesh_row(NODUS_MESH_OPEN, in.data(), in.size(), out.data(), out.size(), &written) == 1;
        }
        uint64_t result = 0;
        if (ok && written) {
            const uint64_t shape[1] = {static_cast<uint64_t>(written)};
            result = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
            ok = result && nodus_tensor_write(result, 0, out.data(), written) == static_cast<int64_t>(written) &&
                 raw_stack_push_abstract_tensor(stack, nodus::tensors::AbstractTensorHandle{result});
            if (!ok && result) nodus_tensor_destroy(result);
        }
        if (!ok) {
            raw_stack_push_abstract_tensor(stack, input);                  // the input goes back where it was
            c.succeeded = false;
        }
    }
};

}  // namespace nodus::mesh_rows

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::mesh_rows::MeshOpen(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
