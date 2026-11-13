#pragma once

#include "gpu_shader_job_tool.h"
#include "kernel_pipeline_tool.h"
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace nodus::gpu {

// Process owner for the existing state-library row and its registered shader
// and pipeline rows. GPU row factories retain a reference to the library, so
// a host must keep this owner alive until all graph/table instances are gone.
// This class provides ownership only; scheduling remains with Nodus tables.
class KernelLibrarySession final {
public:
    explicit KernelLibrarySession(ToolRegistry& registry)
        :registry_(registry),row_(registry.create("nodus.tensor.state.library")){
        library_=row_?dynamic_cast<TensorStateLibraryTool*>(row_.get()):nullptr;
        if(library_)row_->initialize({});
    }
    KernelLibrarySession(const KernelLibrarySession&)=delete;
    KernelLibrarySession& operator=(const KernelLibrarySession&)=delete;
    ~KernelLibrarySession(){
        // Graph/table instances must be destroyed before this owner. Drop
        // factories before destroying the library referenced by those factories.
        for(auto it=registered_.rbegin();it!=registered_.rend();++it)
            registry_.unregister_tool(*it);
        if(library_)row_->shutdown();
    }

    explicit operator bool() const noexcept{return library_!=nullptr;}
    ITool& library_row() const noexcept{return *row_;}
    TensorStateLibraryTool& library() const noexcept{return *library_;}

    bool register_shader(const GpuShaderJobContract& contract){
        if(!library_||library_->device_count()==0)return false;
        std::vector<std::string> ids;
        for(std::size_t device=0;device<library_->device_count();device++){
            auto id=contract.id+".vulkan."+std::to_string(device);
            if(registry_.find(id))return false;
            ids.push_back(std::move(id));
        }
        if(!register_gpu_shader_job(registry_,*library_,contract)){
            // The registrar may have inserted earlier device rows before a
            // later registration failed. All ids were absent at preflight.
            for(const auto& id:ids)registry_.unregister_tool(id);
            return false;
        }
        registered_.insert(registered_.end(),ids.begin(),ids.end());
        return true;
    }

    bool register_pipeline(std::string id,
                           std::vector<KernelPipelineComponent> components){
        if(!library_||registry_.find(id)||
           !register_kernel_pipeline(registry_,id,std::move(components)))return false;
        registered_.push_back(std::move(id));
        return true;
    }

private:
    ToolRegistry& registry_;
    ToolRegistry::Factory::result_type row_;
    TensorStateLibraryTool* library_=nullptr;
    std::vector<std::string> registered_;
};

} // namespace nodus::gpu
