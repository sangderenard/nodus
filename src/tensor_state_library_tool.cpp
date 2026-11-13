#include "tensor_state_library_tool.h"
#include "tool_registry.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <utility>

namespace nodus::gpu {

std::optional<TensorStateLibraryTool::ResidencyChoice>
TensorStateLibraryTool::choose_resident_locked(
    std::uint64_t card,std::size_t device,
    std::span<const Candidate> candidates) const {
    const auto client=cards_.find(card);
    if(!library_||client==cards_.end()||
       device>=client->second.gpu_quota.size())return {};
    std::optional<ResidencyChoice> best;
    for(std::size_t row=0;row<candidates.size();++row){
        const auto base=candidates[row].base,count=candidates[row].fields;
        if(!base||!count||count>256||base>UINT64_MAX-count+1)continue;
        std::uint64_t total=0,resident=0;
        bool owned=true;
        for(std::uint64_t field=0;field<count;++field){
            const auto address=base+field;
            const auto owner=address_card_.find(address);
            const auto size=address_bytes_.find(address);
            if(owner==address_card_.end()||owner->second!=card||
               size==address_bytes_.end()||size->second>UINT64_MAX-total){
                owned=false;break;
            }
            total+=size->second;
            resident+=library_->resident_bytes(address,device);
        }
        if(!owned||total>client->second.gpu_quota[device]-
                          client->second.gpu_leased[device])continue;
        const bool full=resident==total;
        const bool best_full=best&&best->resident_bytes==best->total_bytes;
        if(!best||(full&&!best_full)||
           (full==best_full&&resident>best->resident_bytes))
            best=ResidencyChoice{row,base,resident,total,count};
    }
    return best;
}

std::optional<TensorStateLibraryTool::ResidencyChoice>
TensorStateLibraryTool::choose_resident(
    std::uint64_t card,std::size_t device,
    std::span<const Candidate> candidates) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return choose_resident_locked(card,device,candidates);
}

void TensorStateLibraryTool::initialize(const ToolInitContext&) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(library_)return;
    std::vector<TensorStateLibrary::Device> devices;
    std::vector<kernels::VulkanDeviceInfo> inventory;
    std::string error;
    if(kernels::enumerate_vulkan_devices(&inventory,&error))
        for(const auto& item:inventory){
            if(!item.compute)continue;
            gp_mem_backend_vulkan_params_t params{item.index,GP_VULKAN_ALLOC_DEVICE_LOCAL};
            kernels::VulkanDeviceInfo selected;
            auto* context=kernels::create_vulkan_backend(&params,&selected,&error);
            if(!context)continue;
            if(item.uuid_valid&&(!selected.uuid_valid||item.uuid!=selected.uuid)){
                gp_mem_backend_release(context);continue;
            }
            const auto suggested=VulkanStateBank::suggested_budget(context,.5);
            const auto slot=devices.size();
            const auto budget=slot<device_budget_bytes_.size()&&
                              device_budget_bytes_[slot]
                ? std::min(suggested,device_budget_bytes_[slot]) : suggested;
            if(!budget){gp_mem_backend_release(context);continue;}
            contexts_.emplace_back(context,&gp_mem_backend_release);
            devices.push_back({context,budget});
        }
    library_=std::make_unique<TensorStateLibrary>(std::move(devices),ram_budget_);
    completion_windows_.resize(library_->device_count());
}

void TensorStateLibraryTool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!library_)return;
    if(inflight_gpu_loans_){
        std::fputs("Nodus tensor state library shutdown with GPU loans in flight\n",stderr);
        return;
    }
    if(!library_->flush_all()) {
        std::fputs("Nodus tensor state library shutdown with an active or unsaved lease\n",stderr);
        return;
    }
    for(const auto& [address,_]:address_card_)
        if(!library_->remove(address)){
            std::fputs("Nodus tensor state library could not release an address\n",stderr);
            return;
        }
    address_card_.clear();address_bytes_.clear();cards_.clear();
    completion_windows_.clear();work_windows_.clear();
    library_.reset();contexts_.clear();
}

void TensorStateLibraryTool::record_completion(
    std::size_t device,std::uint64_t lease_ns,
    std::uint64_t dispatch_ns,std::uint64_t total_ns){
    std::lock_guard<std::mutex> lock(mutex_);
    if(device>=completion_windows_.size())return;
    auto& window=completion_windows_[device];
    window.lease[window.next]=lease_ns;
    window.dispatch[window.next]=dispatch_ns;
    window.total[window.next]=total_ns;
    window.next=(window.next+1)%window.total.size();
    window.count=std::min(window.count+1,window.total.size());
}

std::optional<TensorStateLibraryTool::DevicePerformance>
TensorStateLibraryTool::device_performance(std::size_t device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!library_||device>=completion_windows_.size())return {};
    const auto cache=library_->stats();
    const auto& window=completion_windows_[device];
    DevicePerformance result{};
    result.samples=window.count;
    result.cache_resident_bytes=cache.vram[device].resident_bytes;
    result.cache_budget_bytes=cache.vram[device].budget_bytes;
    if(window.count){
        for(std::size_t i=0;i<window.count;i++){
            result.average_lease_ns+=window.lease[i]/window.count;
            result.average_dispatch_ns+=window.dispatch[i]/window.count;
            result.average_total_ns+=window.total[i]/window.count;
        }
    }
    return result;
}

void TensorStateLibraryTool::record_work_completion(
    ExecutionApi api,std::size_t device,std::uint64_t program,
    std::uint64_t lanes,std::uint64_t lease_ns,
    std::uint64_t dispatch_ns,std::uint64_t total_ns){
    if(!program||!lanes)return;
    std::lock_guard<std::mutex> lock(mutex_);
    if(!library_||
       (api==ExecutionApi::Vulkan&&device>=library_->device_count()))return;
    const WorkKey key{api,device,program,lanes};
    if(!work_windows_.contains(key)&&work_windows_.size()>=1024){
        auto oldest=std::min_element(work_windows_.begin(),work_windows_.end(),
            [](const auto& a,const auto& b){
                return a.second.last_update_ns<b.second.last_update_ns;
            });
        work_windows_.erase(oldest);
    }
    auto& window=work_windows_[key];
    window.lease[window.next]=lease_ns;
    window.dispatch[window.next]=dispatch_ns;
    window.total[window.next]=total_ns;
    window.next=(window.next+1)%window.total.size();
    window.count=std::min(window.count+1,window.total.size());
    window.last_update_ns=static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

std::optional<TensorStateLibraryTool::WorkPerformance>
TensorStateLibraryTool::work_performance(ExecutionApi api,std::size_t device,
    std::uint64_t program,std::uint64_t lanes) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found=work_windows_.find({api,device,program,lanes});
    if(found==work_windows_.end())return {};
    const auto& window=found->second;
    WorkPerformance result{};result.samples=window.count;
    for(std::size_t i=0;i<window.count;i++){
        result.average_lease_ns+=window.lease[i]/window.count;
        result.average_dispatch_ns+=window.dispatch[i]/window.count;
        result.average_total_ns+=window.total[i]/window.count;
    }
    return result;
}

TensorStateLibraryTool::Result TensorStateLibraryTool::handle(
    const std::array<std::uint64_t,kRequestWords>& request,std::uint64_t payload,
    std::array<std::uint64_t,kReceiptWords>& receipt) {
    if(!library_)return Result::LibraryUnavailable;
    const auto op=static_cast<Operation>(request[0]);
    const auto card_id=request[1],address=request[3];
    if(op==Operation::Card){
        if(!card_id||!request[2]||!request[4]||cards_.count(card_id))
            return Result::Invalid;
        const auto stats=library_->stats();
        std::uint64_t reserved=0;
        for(const auto& [_,card]:cards_)reserved+=card.ram_quota;
        if(request[4]>ram_budget_-reserved)return Result::Quota;
        Card card;card.process=request[2];card.ram_quota=request[4];
        card.gpu_quota.resize(stats.vram.size());
        card.gpu_leased.resize(stats.vram.size());
        for(std::size_t i=0;i<3;i++){
            const auto requested=request[5+i];
            if(i>=stats.vram.size()){
                if(requested)return Result::DeviceUnavailable;
                continue;
            }
            std::uint64_t taken=0;
            for(const auto& [_,earlier]:cards_)taken+=earlier.gpu_quota[i];
            if(requested>stats.vram[i].budget_bytes-taken)return Result::Quota;
            card.gpu_quota[i]=requested;
        }
        cards_.emplace(card_id,std::move(card));
        return Result::Ok;
    }
    if(op==Operation::Status)return Result::Ok;
    const auto found=cards_.find(card_id);
    if(found==cards_.end())return Result::CardUnknown;
    if(op==Operation::ChooseResident){
        const auto device=static_cast<std::size_t>(request[5]);
        if(device>=library_->device_count() ||
           device>=found->second.gpu_quota.size())return Result::DeviceUnavailable;
        NodusTensorDesc desc{};
        if(nodus_tensor_describe(payload,&desc)!=NODUS_OK ||
           desc.dtype!=NODUS_DTYPE_U64 || desc.rank!=2 ||
           desc.shape[0]<1 || desc.shape[0]>64 || desc.shape[1]!=2)
            return Result::Invalid;
        std::vector<Candidate> candidates(desc.shape[0]);
        if(nodus_tensor_read(payload,0,candidates.data(),
               candidates.size()*sizeof(Candidate))!=
           static_cast<std::int64_t>(candidates.size()*sizeof(Candidate)))
            return Result::Invalid;
        const auto choice=choose_resident_locked(card_id,device,candidates);
        if(!choice)return Result::NoCandidate;
        receipt[2]=choice->base;
        receipt[21]=choice->index;
        receipt[22]=choice->resident_bytes;
        receipt[23]=choice->total_bytes;
        receipt[24]=choice->fields;
        receipt[25]=device;
        return Result::Ok;
    }
    if(!address)return Result::Invalid;
    if(op==Operation::Put){
        if(address_card_.count(address))return Result::StateConflict;
        NodusTensorDesc desc{};
        if(nodus_tensor_describe(payload,&desc)!=NODUS_OK||!desc.rank||
           !desc.element_bytes||!desc.total_bytes||
           desc.total_bytes>std::numeric_limits<std::size_t>::max()||
           desc.dtype==NODUS_DTYPE_UNKNOWN)return Result::Invalid;
        const auto bytes=static_cast<std::size_t>(desc.total_bytes);
        if(bytes>found->second.ram_quota-found->second.ram_used)return Result::Quota;
        std::vector<std::uint8_t> values(bytes);
        if(nodus_tensor_read(payload,0,values.data(),bytes)!=static_cast<int64_t>(bytes))
            return Result::Invalid;
        TensorStateLibrary::Spec spec;
        spec.dtype=static_cast<nodus::tensors::TensorDType>(desc.dtype);
        spec.shape.assign(desc.shape,desc.shape+desc.rank);
        if(!library_->add(address,std::move(spec),values.data()))return Result::Quota;
        found->second.ram_used+=bytes;
        address_card_[address]=card_id;address_bytes_[address]=bytes;
        return Result::Ok;
    }
    if(op==Operation::Remove){
        if(!address_card_.count(address)||address_card_[address]!=card_id)
            return Result::CardUnknown;
        if(!library_->remove(address))return Result::StateConflict;
        found->second.ram_used-=address_bytes_[address];
        address_bytes_.erase(address);address_card_.erase(address);
        return Result::Ok;
    }
    return Result::Invalid;
}

void TensorStateLibraryTool::execute_stack(ToolStackContext& context) {
    context.succeeded=false;
    if(!context.stack.raw)return;
    auto& stack=*context.stack.raw;
    nodus::tensors::AbstractTensorHandle payload{},control{};
    if(!raw_stack_pop_abstract_tensor(stack,payload))return;
    if(!raw_stack_pop_abstract_tensor(stack,control)){
        raw_stack_push_abstract_tensor(stack,payload);return;
    }
    std::array<std::uint64_t,kRequestWords> request{};
    NodusTensorDesc desc{};
    Result result=Result::Invalid;
    std::array<std::uint64_t,kReceiptWords> receipt{};
    if(nodus_tensor_describe(control.id,&desc)==NODUS_OK&&
       desc.dtype==NODUS_DTYPE_U64&&desc.rank==1&&
       desc.shape[0]==kRequestWords&&
       nodus_tensor_read(control.id,0,request.data(),sizeof(request))==sizeof(request)){
        receipt[2]=request[3];
        std::lock_guard<std::mutex> lock(mutex_);
        result=handle(request,payload.id,receipt);
    }
    receipt[0]=static_cast<std::uint64_t>(result);
    receipt[1]=request[1];
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(library_){
            const auto stats=library_->stats();
            receipt[3]=stats.ram_bytes;receipt[4]=stats.ram_budget_bytes;
            receipt[5]=stats.vram.size();
            for(std::size_t i=0;i<stats.vram.size()&&i<3;i++){
                receipt[6+i*5]=stats.vram[i].budget_bytes;
                receipt[7+i*5]=stats.vram[i].resident_bytes;
                receipt[8+i*5]=stats.vram[i].evicted;
                receipt[9+i*5]=stats.vram[i].reused;
                receipt[10+i*5]=stats.vram[i].saved;
                receipt[26+i]=stats.vram[i].restored;
            }
        }
    }
    const std::uint64_t shape[1]={kReceiptWords};
    const auto output=nodus_tensor_create(NODUS_DTYPE_U64,1,shape);
    if(!output||nodus_tensor_write(output,0,receipt.data(),sizeof(receipt))!=sizeof(receipt)||
       !raw_stack_push_abstract_tensor(stack,{output})){
        if(output)nodus_tensor_destroy(output);
        raw_stack_push_abstract_tensor(stack,control);
        raw_stack_push_abstract_tensor(stack,payload);
        return;
    }
    context.succeeded=true;
}

TensorStateLibraryTool::GpuLoan TensorStateLibraryTool::borrow_gpu(
    std::uint64_t card,std::uint64_t address,std::size_t device,bool writable){
    std::uint64_t bytes=0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto client=cards_.find(card);
        auto owner=address_card_.find(address);
        if(!library_||client==cards_.end()||owner==address_card_.end()||
           owner->second!=card||device>=client->second.gpu_quota.size())return {};
        bytes=address_bytes_[address];
        if(bytes>client->second.gpu_quota[device]-client->second.gpu_leased[device])
            return {};
        client->second.gpu_leased[device]+=bytes;
        ++inflight_gpu_loans_;
    }
    auto lease=library_->lease_gpu(address,device,writable);
    if(!lease){
        std::lock_guard<std::mutex> lock(mutex_);
        cards_.at(card).gpu_leased[device]-=bytes;
        --inflight_gpu_loans_;
        return {};
    }
    return GpuLoan(this,card,device,bytes,std::move(lease));
}
TensorStateLibraryTool::CpuLoan TensorStateLibraryTool::borrow_cpu(
    std::uint64_t card,std::uint64_t address,bool writable){
    std::lock_guard<std::mutex> lock(mutex_);
    const auto owner=address_card_.find(address);
    if(!library_||!cards_.count(card)||owner==address_card_.end()||
       owner->second!=card)return {};
    return CpuLoan(library_->lease_cpu(address,writable));
}
std::optional<TensorStateLibrary::Spec> TensorStateLibraryTool::describe_state(
    std::uint64_t card,std::uint64_t address) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto owner=address_card_.find(address);
    if(!library_||!cards_.count(card)||owner==address_card_.end()||
       owner->second!=card)return {};
    return library_->describe(address);
}
std::uint64_t TensorStateLibraryTool::resident_bytes(
    std::uint64_t card,std::uint64_t address,std::size_t device) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto owner=address_card_.find(address);
    if(!library_||!cards_.count(card)||owner==address_card_.end()||
       owner->second!=card)return 0;
    return library_->resident_bytes(address,device);
}
void TensorStateLibraryTool::release_gpu_loan(
    std::uint64_t card,std::size_t device,std::uint64_t bytes,
    TensorStateLibrary::GpuLease& lease) noexcept {
    lease.reset();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it=cards_.find(card);
    if(it!=cards_.end()&&device<it->second.gpu_leased.size())
        it->second.gpu_leased[device]-=bytes;
    --inflight_gpu_loans_;
}
void TensorStateLibraryTool::GpuLoan::reset() noexcept {
    if(owner_)owner_->release_gpu_loan(card_,device_,bytes_,lease_);
    owner_=nullptr;
}
std::size_t TensorStateLibraryTool::device_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return library_?library_->device_count():0;
}

} // namespace nodus::gpu

using NodusTensorStateLibraryTool=nodus::gpu::TensorStateLibraryTool;
REGISTER_TOOL(NodusTensorStateLibraryTool,"nodus.tensor.state.library",
              "Tensor state library",ToolCaps::None);
