#pragma once

#include "gpu_shader_job_tool.h"
#include "gpu_fabric_dispatch_tool.h"
#include "fabric_spring_state_tool.h"
#include "fftfree_tool.h"
#include "wave_t4_state_tool.h"
#include "kernel_pipeline_tool.h"
#include "gpu_work_row_tool.h"
#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace nodus::gpu {

// A table row that is a session's own state library row: every call goes to that one row, and the session owns its
// lifetime (initialize and shutdown are the session's, never the table's).
class LibraryRowView final : public ITool {
public:
    explicit LibraryRowView(ITool& row):row_(row){}
    const char* id_cstr() const noexcept override{return "nodus.volunteer.library";}
    const char* name_cstr() const noexcept override{return row_.name_cstr();}
    ToolCaps caps() const override{return row_.caps();}
    void initialize(const ToolInitContext&) override{}
    void shutdown() override{}
    void tick(double dt,HostAPI& host) override{row_.tick(dt,host);}
    void render(RenderContext& context) override{row_.render(context);}
    int32_t port_count() const override{return row_.port_count();}
    ToolPortSpec port_spec(int32_t index) const override{return row_.port_spec(index);}
    void execute_stack(ToolStackContext& context) override{row_.execute_stack(context);}
    int32_t row_port_count() const override{return row_.row_port_count();}
    ToolRowPort row_port(int32_t index) const override{return row_.row_port(index);}
    ToolTensorRole tensor_role() const override{return row_.tensor_role();}
private:
    ITool& row_;
};

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

    // The fabric dispatch rows nodus.fabric.dispatch.vulkan.<device>, one per GPU bank, over this owner's library
    // (gpu_fabric_dispatch_tool.h); unregistered with the session like the other rows.
    bool register_fabric_dispatch(const std::vector<std::uint32_t>& shader,std::uint32_t local_size,
                                  GpuShaderWorkContract contract={}){
        if(!library_||library_->device_count()==0)return false;
        std::vector<std::string> ids;
        for(std::size_t device=0;device<library_->device_count();device++){
            auto id="nodus.fabric.dispatch.vulkan."+std::to_string(device);
            if(registry_.find(id))return false;
            ids.push_back(std::move(id));
        }
        if(!register_gpu_fabric_dispatchers(registry_,*library_,shader,local_size,contract)){
            for(const auto& id:ids)registry_.unregister_tool(id);
            return false;
        }
        registered_.insert(registered_.end(),ids.begin(),ids.end());
        return true;
    }

    // The fabric row on this library (fabric_spring_state_tool.h): its stage rows, law rows, pipeline and program row
    // `row_id` with backend "cpu" or "vulkan.<device>", its state at `base` on `card`.
    bool register_fabric_spring(const FabricSpringProducts& products,const std::string& row_id,
                                const std::string& backend,std::uint64_t card,std::uint64_t base){
        if(!library_||registry_.find(row_id))return false;
        return nodus::gpu::register_fabric_spring(registry_,*library_,products,row_id,backend,card,base,&registered_);
    }

    // The volunteer cycle's rows on this library (case a11b5c9b1be74c9b9fb684790f7fd5a3): the library row
    // nodus.volunteer.library, which IS this session's state library row (a table holding it acts on this library,
    // not a second one), registered once, and one work row per device for the shader job `contract` registered here
    // (gpu_work_row_tool.h): call once per job, so each device takes mixed programs. `capacity` bounds each work
    // row's runner queue.
    bool register_volunteer_rows(const GpuShaderJobContract& contract,std::size_t capacity=4){
        if(!library_||!registry_.find(contract.id+".vulkan.0"))return false;
        if(!register_volunteer_library())return false;
        return register_volunteer_work_rows(registry_,contract.id,contract.program_id,library_->device_count(),
                                            capacity,&registered_);
    }

    // The library row alone (nodus.volunteer.library, this session's own row), for a host whose work runs elsewhere:
    // the server's side of the cycle across the link. Registered once; true when it is registered.
    bool register_volunteer_library(){
        if(!library_)return false;
        if(registry_.find("nodus.volunteer.library"))return true;
        ToolRegistry::Entry item;
        item.id="nodus.volunteer.library";
        item.name="Volunteer state library";
        item.caps=ToolCaps::None;
        auto* row=row_.get();
        item.factory=[row]{
            return std::unique_ptr<ITool,std::function<void(ITool*)>>(new LibraryRowView(*row),
                [](ITool* value){delete value;});
        };
        if(!registry_.register_tool(std::move(item)))return false;
        registered_.push_back("nodus.volunteer.library");
        return true;
    }

    bool register_pipeline(std::string id,
                           std::vector<KernelPipelineComponent> components){
        if(!library_||registry_.find(id)||
           !register_kernel_pipeline(registry_,id,std::move(components)))return false;
        registered_.push_back(std::move(id));
        return true;
    }

    bool register_fftfree(const std::string& dll){
        if(!library_||registry_.find("nodus.fftfree")||
           !register_fftfree_tool(registry_,*library_,dll))return false;
        registered_.push_back("nodus.fftfree");
        return true;
    }

    bool register_wave_t4(const std::string& dll){
        if(!library_||registry_.find("nodus.wave.t4.state")||
           !register_wave_t4_state_tool(registry_,*library_,dll))return false;
        registered_.push_back("nodus.wave.t4.state");
        return true;
    }

private:
    ToolRegistry& registry_;
    ToolRegistry::Factory::result_type row_;
    TensorStateLibraryTool* library_=nullptr;
    std::vector<std::string> registered_;
};

} // namespace nodus::gpu
