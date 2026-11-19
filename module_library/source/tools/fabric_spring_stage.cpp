// fabric_spring_stage: the four host stage rows of the fabric spring row on the tensor state library
// (include/fabric_spring_state_tool.h): begin, prepare, feedback and commit, one class, the stage chosen by its settings.
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/fabric_spring_state_tool.cpp).
// Its settings (stage, id) are its saved state (serialize/deserialize); a row with no library refuses.
#include "fabric_spring_state_tool.h"

#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <limits>
#include <vector>

namespace nodus::gpu {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kStageSettingsMagic = 0x31535346, kStageSettingsVersion = 1;

// ---- the four host stages ---------------------------------------------------------------------------------------
class FabricStageTool final : public ITool {
public:
    FabricStageTool(TensorStateLibraryTool& library, FabricStage stage, std::string id) : library_(&library), stage_(stage), id_(std::move(id)) {}
    FabricStageTool() = default;           // no library: the row refuses until a host gives it one; settings from configure()/saved state
    void configure(FabricStage stage, std::string id) { stage_ = stage; id_ = std::move(id); }
    // Saved state: the row's settings (magic "FSS1", version 1, stage, id).
    bool serialize(OutputArchive& out) const override {
        auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
        auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
        auto str=[&](const std::string& v){return u32(static_cast<std::uint32_t>(v.size()))&&(v.empty()||out.write(v.data(),v.size()));};
        auto vec=[&](const auto& v){
            if(!u64(v.size()))return false;
            for(const auto x:v)if(!u64(static_cast<std::uint64_t>(x)))return false;
            return true;
        };
        (void)vec;
        return u32(kStageSettingsMagic) && u32(kStageSettingsVersion) && u64(static_cast<std::uint64_t>(stage_)) && str(id_);
    }
    bool deserialize(InputArchive& in) override {
        auto u32=[&](std::uint32_t& v){return in.read(&v,sizeof(v));};
        auto u64=[&](std::uint64_t& v){return in.read(&v,sizeof(v));};
        auto str=[&](std::string& v,std::uint32_t limit){
            std::uint32_t n=0;
            if(!u32(n)||n>limit)return false;
            v.resize(n);
            return n==0||in.read(v.data(),n);
        };
        auto vec=[&](auto& v,std::uint64_t limit){
            std::uint64_t n=0;
            if(!u64(n)||n>limit)return false;
            v.resize(static_cast<std::size_t>(n));
            for(auto& x:v){
                std::uint64_t raw=0;
                using Element=std::decay_t<decltype(x)>;
                if(!u64(raw)||raw>std::numeric_limits<Element>::max())return false;
                x=static_cast<Element>(raw);
            }
            return true;
        };
        (void)vec;
        std::uint32_t magic = 0, version = 0;
        std::uint64_t stage = 0;
        std::string id;
        if (!u32(magic) || !u32(version) || magic != kStageSettingsMagic || version != kStageSettingsVersion || !u64(stage) ||
            stage < std::uint64_t(FabricStage::Begin) || stage > std::uint64_t(FabricStage::Commit) || !str(id, 4096)) return false;
        configure(static_cast<FabricStage>(stage), std::move(id));
        return true;
    }
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return "Fabric spring stage"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t i) const override {
        if (i == 0) return {ToolPortKind::Argument, 1};
        if (i == 1) return {ToolPortKind::Return, 1};
        return {};
    }
    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) deserialize(in);
        }
    }
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (!context.stack.raw || !library_) return;
        auto& stack = *context.stack.raw;
        nodus::tensors::AbstractTensorHandle input{};
        if (!raw_stack_pop_abstract_tensor(stack, input)) return;
        std::array<std::uint64_t, 4> w{};
        NodusTensorDesc d{};
        const auto t0 = Clock::now();
        std::uint64_t status = 1;
        if (nodus_tensor_describe(input.id, &d) == NODUS_OK && d.dtype == NODUS_DTYPE_U64 && d.rank == 1 && d.shape[0] == 4 &&
            nodus_tensor_read(input.id, 0, w.data(), sizeof(w)) == sizeof(w) && w[0] && w[2] == std::uint64_t(kFabricBatch))
            status = run(w[0], w[1]) ? 0 : 2;
        const auto ns = std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        const std::uint64_t shape[1] = {9};
        const auto receipt = stage_receipt(status, w, ns);
        const auto out = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
        if (!out || nodus_tensor_write(out, 0, receipt.data(), sizeof(receipt)) != sizeof(receipt) || !raw_stack_push_abstract_tensor(stack, {out})) {
            if (out) nodus_tensor_destroy(out);
            raw_stack_push_abstract_tensor(stack, input); return;
        }
        context.succeeded = true;
    }

private:
    double* column(std::uint64_t card, std::uint64_t address, bool writable, std::vector<TensorStateLibraryTool::CpuLoan>& loans, std::size_t bytes) {
        auto loan = library_->borrow_cpu(card, address, writable);
        if (!loan || loan.bytes() < bytes) return nullptr;
        auto* p = static_cast<double*>(loan.data());
        loans.push_back(std::move(loan));
        return p;
    }
    bool run(std::uint64_t card, std::uint64_t base) {
        std::vector<TensorStateLibraryTool::CpuLoan> loans;
        const std::size_t lanes = std::size_t(kFabricBatch) * sizeof(double);
        auto col = [&](std::uint64_t i, bool w) { return column(card, base + i, w, loans, lanes); };
        switch (stage_) {
        case FabricStage::Begin: {
            double* p = col(17, true); double* v = col(23, true); const double* cp = col(kCommittedP, false); const double* cv = col(kCommittedV, false);
            if (!p || !v || !cp || !cv) return false;
            std::memcpy(p, cp, lanes); std::memcpy(v, cv, lanes); return true;
        }
        case FabricStage::Prepare: {
            auto contact_loan = library_->borrow_cpu(card, base + kContact, false);
            auto dt_loan = library_->borrow_cpu(card, base + kDt, false);
            auto table_loan = library_->borrow_cpu(card, base + kNeighbors, false);
            if (!contact_loan || contact_loan.bytes() < 8 * sizeof(double) || !dt_loan || dt_loan.bytes() < sizeof(double) ||
                !table_loan || table_loan.bytes() < std::size_t(kFabricNodeSlots) * 4 * sizeof(std::uint32_t)) return false;
            const double* contact = static_cast<const double*>(contact_loan.data());
            const double slice_dt = *static_cast<const double*>(dt_loan.data());
            const auto* table = static_cast<const std::uint32_t*>(table_loan.data());
            double* enabled = col(1, true); double* height = col(2, true); double* dt = col(4, true);
            const double* position = col(17, false);
            std::array<const double*, 4> mask{}; std::array<double*, 4> neighbor{};
            for (int e = 0; e < 4; ++e) { mask[e] = col(5 + e, false); neighbor[e] = col(11 + e, true); }
            if (!enabled || !height || !dt || !position) return false;
            for (int e = 0; e < 4; ++e) if (!mask[e] || !neighbor[e]) return false;
            const bool floor_on = contact[0] != 0, table_on = contact[2] != 0;
            std::fill(dt, dt + kFabricBatch, slice_dt);
            std::fill(enabled, enabled + kFabricBatch, 0.0);
            std::fill(height, height + kFabricBatch, 0.0);
            if (floor_on || table_on) for (int node = 0; node < kFabricNodeSlots; node++) {
                const int lane = node * kFabricAxes;
                if (floor_on) { enabled[lane + 1] = 1; height[lane + 1] = contact[1]; }
                if (table_on && position[lane] >= contact[4] && position[lane] <= contact[5] &&
                    position[lane + 2] >= contact[6] && position[lane + 2] <= contact[7]) { enabled[lane + 1] = 1; height[lane + 1] = contact[3]; }
            }
            for (int node = 0; node < kFabricNodeSlots; node++) for (int e = 0; e < 4; e++) {
                const int other = int(table[node * 4 + e]);
                for (int axis = 0; axis < kFabricAxes; axis++)
                    neighbor[e][node * kFabricAxes + axis] = mask[e][node * kFabricAxes + axis] ? position[other * kFabricAxes + axis] : 0;
            }
            return true;
        }
        case FabricStage::Feedback: {
            const double* np = col(24, false); const double* nv = col(25, false); double* p = col(17, true); double* v = col(23, true);
            if (!np || !nv || !p || !v) return false;
            for (int i = 0; i < kFabricBatch; i++) if (!std::isfinite(np[i]) || !std::isfinite(nv[i])) return false;
            std::memcpy(p, np, lanes); std::memcpy(v, nv, lanes); return true;
        }
        case FabricStage::Commit: {
            const double* p = col(17, false); const double* v = col(23, false); double* cp = col(kCommittedP, true); double* cv = col(kCommittedV, true);
            if (!p || !v || !cp || !cv) return false;
            std::memcpy(cp, p, lanes); std::memcpy(cv, v, lanes); return true;
        }
        }
        return false;
    }
    TensorStateLibraryTool* library_ = nullptr;
    FabricStage stage_ = FabricStage::Begin;
    std::string id_;
};

}  // namespace
}  // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::FabricStageTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
