#pragma once
#include "graph_runtime.h"
#include "opengl_gpu_row_tool.h"

namespace nodus::gpu {
// Bind an UNUSED module and existing typed FIFO edges through the actual native
// GraphRuntime/ThreadManager contract. Caller owns graph topology and lifecycle.
// Edges must already be configured losslessly, before any traffic is published.
// This helper creates no context, renderer, queue or independent scheduler.
inline bool bind_opengl_gpu_module(GraphRuntime& graph,ToolRegistry& registry,
    const std::string& tool_id,ThreadManager::ModuleContract& module,
    GP_TableContext* root_table,int control_edge,int telemetry_edge,
    int control_contact,int telemetry_contact,std::string* error=nullptr) {
    auto fail=[&](const char* why){if(error)*error=why;return false;};
    if(module.module_idx<0 || !root_table || control_edge<0 || telemetry_edge<0 ||
       control_edge==telemetry_edge || control_contact<0 || telemetry_contact<0 ||
       control_contact==telemetry_contact)return fail("GPU module/edge/contact identity is invalid");
    const auto types=register_port_types();
    GP_TableEdgeTensorSpecTyped control{},telemetry{};
    if(!gp_table_edge_get_tensor_spec(root_table,control_edge,&control) ||
       !gp_table_edge_get_tensor_spec(root_table,telemetry_edge,&telemetry))return fail("GPU FIFO edge is not configured");
    auto matches=[](const GP_TableEdgeTensorSpecTyped& spec,ValueTypeId type,size_t bytes) {
        return spec.dim_count==1 && spec.dims[0]==1 && spec.elem_size==int32_t(bytes) &&
            spec.type_id==type && spec.slots>0 && spec.top_k==0;
    };
    if(!matches(control,types.control,sizeof(Control)) || !matches(telemetry,types.telemetry,sizeof(Telemetry)))
        return fail("GPU FIFO must match exact typed Control/Telemetry and lossless capacity");
    const auto index=size_t(module.module_idx);
    if(index<graph.module_io_rows.size() && !graph.module_io_rows[index].empty())return fail("GPU module slot is already occupied");
    auto tool=registry.create(tool_id);
    if(!tool || dynamic_cast<OpenGLGpuTool*>(tool.get())==nullptr)return fail("GPU module requires its bound native OpenGL factory");
    // Grow only the selected slot; preserve every other module and tool.
    graph.module_io_rows.resize(std::max(graph.module_io_rows.size(),index+1));
    graph.module_plugin_instances.resize(std::max(graph.module_plugin_instances.size(),index+1));
    graph.module_stack_contract.resize(std::max(graph.module_stack_contract.size(),index+1),ModuleStackContract::Opaque);
    graph.module_exec_mode.resize(std::max(graph.module_exec_mode.size(),index+1),int(ThreadManager::ExecMode::Pooled));
    graph.module_io_in_count.resize(std::max(graph.module_io_in_count.size(),index+1),0);
    graph.module_io_out_count.resize(std::max(graph.module_io_out_count.size(),index+1),0);
    graph.module_io_rows[index]={{ModuleRowKind::Input,control_contact,ModuleToolKind::None,1},
        {ModuleRowKind::Tool,-1,ModuleToolKind::None,0,ModuleToolOrigin::Plugin,tool_id},
        {ModuleRowKind::Output,telemetry_contact,ModuleToolKind::None,1}};
    graph.module_plugin_instances[index].resize(3);
    graph.module_plugin_instances[index][1]=std::move(tool);
    graph.module_stack_contract[index]=ModuleStackContract::AbstractTensorPointer;
    graph.module_exec_mode[index]=int(ThreadManager::ExecMode::Sequential);
    graph.module_io_in_count[index]=1;graph.module_io_out_count[index]=1;
    module.in_count=1;module.out_count=1;
    module.stack_contract=ModuleStackContract::AbstractTensorPointer;
    module.exec_mode=ThreadManager::ExecMode::Sequential;
    if(error)error->clear();return true;
}
} // namespace nodus::gpu
