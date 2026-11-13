#pragma once
// Inventory -> ordinary native tool factories and the EXISTING contract pair.
// No enumeration, context, process launch, queue or scheduler is created here.
#include "kernel_vulkan_runner.h"
#include "contract_negotiator_registry.h"
#include "opengl_gpu_graph_binding.h"
#include "module_library.h"
#include <optional>

namespace nodus::gpu {
struct InventoryRowConfig {
    kernels::VulkanDeviceInfo device; // caller's read-only snapshot; not GL proof
    uint64_t host_id=0; // caller's persisted contract participant identity
    // No config, missing Windows identity or missing node mask means a visible
    // unavailable row. Never infer node_mask, GL capability or device affinity.
    std::optional<ProtectedProcessOptions> process;
};
struct InventoryRow {
    kernels::VulkanDeviceInfo device;
    uint64_t host_id=0;
    std::string tool_id,unavailable_reason;
    std::shared_ptr<OpenGLGpuHost> host;
};
struct InventoryRowStatus {
    uint64_t host_id=0;
    std::array<uint8_t,16> uuid{};
    std::array<uint8_t,8> inventory_luid{};
    uint32_t vulkan_snapshot_index=0;
    bool provider_available=false,harness_busy=false,physical_utilization_available=false;
    std::string reason;
    Telemetry operation{};
};
class GpuInventoryRows {
public:
    using ProviderFactory=std::function<std::shared_ptr<ProtectedLinkProvider>(const ProtectedProcessOptions&)>;
    // Registration must occur in the graph's native image before binding/loading
    // rows. Input order remains the existing negotiator host-turn order.
    static std::unique_ptr<GpuInventoryRows> create(const std::string& prefix,
        const std::vector<InventoryRowConfig>& devices,
        std::function<std::shared_ptr<const Work>(const contracts::Contract&)> lookup,
        size_t contract_capacity=64,uint64_t loan_ticks=60,ProviderFactory factory={});
    ~GpuInventoryRows();
    GpuInventoryRows(const GpuInventoryRows&)=delete;
    const std::vector<InventoryRow>& rows() const {return rows_;}
    contracts::ContractNegotiator& negotiator() const;
    const InventoryRow* find_tool(const std::string& tool_id) const;
    InventoryRowStatus status(const std::string& tool_id) const;
    // Existing module-library configuration types/serializer, no GPU save format.
    GP_ModuleLibraryTool library_tool(const std::string& tool_id) const;
    GP_ModuleLibraryModule library_module(const std::string& tool_id,int module_index) const;
    bool bind_module(const std::string& saved_tool_id,GraphRuntime& graph,
        ThreadManager::ModuleContract& module,GP_TableContext* root,int control_edge,
        int telemetry_edge,int control_contact,int telemetry_contact,std::string* error=nullptr) const;
    // Explicit post-load restoration for a stopped, freshly loaded native graph.
    // Verifies the saved row program and FIFO types; does not reinterpret stale
    // numeric ValueType IDs or replace live tools. No samples may be in flight.
    bool restore_module(const std::string& saved_tool_id,GraphRuntime& graph,
        ThreadManager::ModuleContract& module,GP_TableContext* root,int control_edge,
        int telemetry_edge,int control_contact,int telemetry_contact,std::string* error=nullptr) const;
    // Only after bound graph/tool instances are destroyed, on the owning lane.
    // Unfinished contracts or native ownership refuse removal; no silent replay.
    bool close(std::string& reason);
private:
    struct InstanceBindings;
    GpuInventoryRows()=default;
    std::string prefix_;
    contracts::ContractNegotiator* negotiator_=nullptr;
    std::vector<InventoryRow> rows_;
    std::vector<std::shared_ptr<ProtectedLinkProvider>> providers_;
    std::shared_ptr<InstanceBindings> instances_;
};
} // namespace nodus::gpu
