// The volunteer cycle's rows in the tool registry (case a11b5c9b1be74c9b9fb684790f7fd5a3). The work row itself is the
// plugin source module_library/source/tools/gpu_work_row.cpp, compiled into this image the way a composed plugin
// source is (as src/gpu_shader_job_tool.cpp does for its row).
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_gpu_work_row_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_gpu_work_row_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_gpu_work_row_source
#include "../module_library/source/tools/gpu_work_row.cpp"

#include "tool_registry.h"

namespace nodus::gpu {

// Each device's work row wraps a fresh instance of that device's job row <job_id>.vulkan.<device>, which the session
// registered over its library (KernelLibrarySession::register_shader).
bool register_volunteer_work_rows(ToolRegistry& registry, const std::string& job_id, std::uint64_t program,
                                  std::size_t devices, std::size_t capacity, std::vector<std::string>* registered) {
    for (std::size_t device = 0; device < devices; ++device) {
        const std::string job = job_id + ".vulkan." + std::to_string(device);
        if (!registry.find(job)) return false;
        ToolRegistry::Entry item;
        item.id = "nodus.volunteer.work." + job;
        item.name = "GPU work row " + job;
        item.caps = ToolCaps::None;
        item.source_path = nodus_gpu_work_row_source();
        const std::string id = item.id;
        item.factory = [&registry, id, job, device, program, capacity] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(
                new GpuWorkRowTool(id, registry.create(job), device, program, capacity),
                [](ITool* value) { nodus_gpu_work_row_destroy(value); });
        };
        if (!registry.register_tool(std::move(item))) return false;
        if (registered) registered->push_back(id);
    }
    return true;
}

}  // namespace nodus::gpu
