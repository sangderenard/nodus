// The fabric spring rows in the tool registry (include/fabric_spring_state_tool.h). The rows themselves are the plugin sources
// module_library/source/tools/fabric_spring_stage.cpp and module_library/source/tools/fabric_spring_state.cpp, compiled into
// this image the way a composed plugin source is: NODUS_PLUGIN_COMPOSITE with each source's exports renamed (as
// src/tcp_byte_stream_tool.cpp does for its row). What stays here is the registration of the rows, the law rows and the pipelines.
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_fabric_spring_stage_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_fabric_spring_stage_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_fabric_spring_stage_source
#include "../module_library/source/tools/fabric_spring_stage.cpp"
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_fabric_spring_state_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_fabric_spring_state_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_fabric_spring_state_source
#include "../module_library/source/tools/fabric_spring_state.cpp"

#include "cpu_fabric_dispatch_tool.h"
#include "gpu_shader_job_tool.h"
#include "kernel_pipeline_tool.h"

#include <algorithm>
#include <memory>

namespace nodus::gpu {

bool register_fabric_spring(ToolRegistry& registry, TensorStateLibraryTool& library, const FabricSpringProducts& products,
                            const std::string& row_id, const std::string& backend, std::uint64_t card, std::uint64_t base,
                            std::vector<std::string>* registered) {
    std::vector<std::string> ids;
    auto undo = [&] { for (auto it = ids.rbegin(); it != ids.rend(); ++it) registry.unregister_tool(*it); return false; };
    const std::pair<FabricStage, const char*> stages[] = {{FabricStage::Begin, "nodus.fabric.spring.begin"}, {FabricStage::Prepare, "nodus.fabric.spring.prepare"},
                                                         {FabricStage::Feedback, "nodus.fabric.spring.feedback"}, {FabricStage::Commit, "nodus.fabric.spring.commit"}};
    for (const auto& [stage, id] : stages) {
        if (registry.find(id)) continue;                                   // shared by every fabric row of this library
        ToolRegistry::Entry e; e.id = id; e.name = "Fabric spring stage"; e.caps = ToolCaps::None;
        e.source_path = nodus_fabric_spring_stage_source();
        e.factory = [&library, stage = stage, id = std::string(id)] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new FabricStageTool(library, stage, id),
                                                                       [](ITool* t) { nodus_fabric_spring_stage_destroy(t); });
        };
        if (!registry.register_tool(std::move(e))) return undo();
        ids.push_back(id);
    }
    // The law: Turing's LLVM product as the existing CPU row, and its GLSL/glslang product as a shader job per device.
    if (!registry.find("nodus.fabric.dispatch.cpu")) {
        if (!register_cpu_fabric_dispatcher(registry, library, products.llvm_dll, products.llvm_entry, std::uint64_t(kFabricBatch), 1, kLawProgram)) return undo();
        ids.push_back("nodus.fabric.dispatch.cpu");
    }
    GpuShaderJobContract contract;
    contract.id = "nodus.shader.turing.spring_network_step"; contract.program_id = kLawProgram; contract.spirv = products.vulkan_spirv;
    for (int i = 0; i < 27; i++) contract.minimum_binding_bytes.push_back(i == 26 ? sizeof(std::uint32_t) : std::size_t(kFabricBatch) * sizeof(double));
    contract.writable.assign(27, 0); contract.writable[24] = contract.writable[25] = 1;
    contract.local_size_x = products.local_size_x; contract.maximum_invocations = std::uint32_t(kFabricBatch);
    const std::string gpu_first = contract.id + ".vulkan.0";
    if (!products.vulkan_spirv.empty() && library.device_count() && !registry.find(gpu_first)) {
        if (!register_gpu_shader_job(registry, library, contract)) return undo();
        for (std::size_t d = 0; d < library.device_count(); d++) ids.push_back(contract.id + ".vulkan." + std::to_string(d));
    }
    const bool gpu = backend.rfind("vulkan.", 0) == 0;
    const std::string law = gpu ? contract.id + "." + backend : std::string("nodus.fabric.dispatch.cpu");
    const auto cpu = TensorStateLibraryTool::ExecutionApi::Cpu;
    const std::string pipeline = "nodus.pipeline.fabric.spring." + backend;
    if (!registry.find(pipeline)) {
        if (!register_kernel_pipeline(registry, pipeline, {
                {"nodus.fabric.spring.begin", cpu, kStageProgram}, {"nodus.fabric.spring.prepare", cpu, kStageProgram},
                {law, gpu ? TensorStateLibraryTool::ExecutionApi::Vulkan : cpu, kLawProgram},
                {"nodus.fabric.spring.feedback", cpu, kStageProgram}, {"nodus.fabric.spring.commit", cpu, kStageProgram}})) return undo();
        ids.push_back(pipeline);
    }
    ToolRegistry::Entry row; row.id = row_id; row.name = "Batched spring sheets and ropes (state library)"; row.caps = ToolCaps::None;
    row.source_path = nodus_fabric_spring_state_source();
    row.factory = [&library, &registry, row_id, pipeline, gpu, card, base] {
        return std::unique_ptr<ITool, std::function<void(ITool*)>>(new FabricSpringRow(library, registry, row_id, pipeline, gpu, card, base),
                                                                   [](ITool* t) { nodus_fabric_spring_state_destroy(t); });
    };
    if (!registry.register_tool(std::move(row))) return undo();
    ids.push_back(row_id);
    if (registered) registered->insert(registered->end(), ids.begin(), ids.end());
    return true;
}

}  // namespace nodus::gpu
