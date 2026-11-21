// tensor_state_library: the tensor state library row (include/tensor_state_library_tool.h).
// The row tool's source, in the module library: the Menu lists it, the canvas compiler inlines it, it builds as its own
// plugin DLL, and the image registers it from it (src/tensor_state_library_tool.cpp).
// Its settings (budgets) are its saved state (serialize/deserialize); a row made with no settings has the defaults.
#include "tensor_state_library_tool.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
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
        std::uint64_t total=0,resident=0,extra=0;
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
            if(!pins_.contains({address,device}))extra+=size->second;
        }
        if(!owned||extra>client->second.gpu_quota[device]-
                          client->second.gpu_pinned[device]-
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

void TensorStateLibraryTool::configure(std::uint64_t ram_budget_bytes,
    std::vector<std::uint64_t> device_budget_bytes,bool include_gpu){
    std::lock_guard<std::mutex> lock(mutex_);
    ram_budget_=ram_budget_bytes;
    device_budget_bytes_=std::move(device_budget_bytes);
    include_gpu_=include_gpu;
}
// Saved state: the row's settings (magic "TSL1", version 1, RAM budget, include_gpu, device budget count, device budgets).
namespace {
constexpr std::uint32_t kTensorLibrarySettingsMagic=0x314c5354,kTensorLibrarySettingsVersion=1;
}
bool TensorStateLibraryTool::serialize(OutputArchive& out) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto u32=[&](std::uint32_t v){return out.write(&v,sizeof(v));};
    auto u64=[&](std::uint64_t v){return out.write(&v,sizeof(v));};
    if(!u32(kTensorLibrarySettingsMagic)||!u32(kTensorLibrarySettingsVersion)||
       !u64(ram_budget_)||!u64(include_gpu_?1:0)||!u64(device_budget_bytes_.size()))return false;
    for(const auto budget:device_budget_bytes_)if(!u64(budget))return false;
    return true;
}
bool TensorStateLibraryTool::deserialize(InputArchive& in) {
    auto u32=[&](std::uint32_t& v){return in.read(&v,sizeof(v));};
    auto u64=[&](std::uint64_t& v){return in.read(&v,sizeof(v));};
    std::uint32_t magic=0,version=0;
    std::uint64_t ram=0,gpu=0,count=0;
    if(!u32(magic)||!u32(version)||magic!=kTensorLibrarySettingsMagic||
       version!=kTensorLibrarySettingsVersion||!u64(ram)||!u64(gpu)||gpu>1||
       !u64(count)||count>64)return false;
    std::vector<std::uint64_t> budgets(static_cast<std::size_t>(count));
    for(auto& budget:budgets)if(!u64(budget))return false;
    configure(ram,std::move(budgets),gpu!=0);
    return true;
}

void TensorStateLibraryTool::initialize(const ToolInitContext& ctx) {
    if(ctx.serialized_path&&ctx.serialized_path[0]!='\0'){
        FileInputArchive in(ctx.serialized_path);
        if(in.ok())deserialize(in);
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if(library_)return;
    std::vector<TensorStateLibrary::Device> devices;
    std::vector<kernels::VulkanDeviceInfo> inventory;
    std::string error;
    // One bank per physical device: the Vulkan inventory can list a device twice (two ICDs; e.g. each NVIDIA card on
    // this machine), and a second bank for the same device would duplicate its state and fall outside the card's
    // GPU quota words (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 4).
    std::vector<std::array<std::uint8_t,16>> opened;
    if(include_gpu_&&kernels::enumerate_vulkan_devices(&inventory,&error))
        for(const auto& item:inventory){
            if(!item.compute)continue;
            if(item.uuid_valid&&std::find(opened.begin(),opened.end(),item.uuid)!=opened.end())continue;
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
            if(item.uuid_valid)opened.push_back(item.uuid);
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
    address_card_.clear();address_bytes_.clear();cards_.clear();pins_.clear();
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
        card.gpu_pinned.resize(stats.vram.size());
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
    if(op==Operation::Pin||op==Operation::Unpin)
        return set_gpu_pin_locked(card_id,address,
            static_cast<std::size_t>(request[5]),op==Operation::Pin);
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
        for(auto it=pins_.begin();it!=pins_.end();){
            if(it->first.first==address){
                found->second.gpu_pinned[it->first.second]-=address_bytes_[address];
                it=pins_.erase(it);
            }else ++it;
        }
        found->second.ram_used-=address_bytes_[address];
        address_bytes_.erase(address);address_card_.erase(address);
        return Result::Ok;
    }
    return Result::Invalid;
}

TensorStateLibraryTool::Result TensorStateLibraryTool::set_gpu_pin_locked(
    std::uint64_t card,std::uint64_t address,std::size_t device,bool pinned){
    const auto owner=address_card_.find(address);
    auto client=cards_.find(card);
    if(!library_||client==cards_.end()||owner==address_card_.end()||
       owner->second!=card)return Result::CardUnknown;
    if(device>=client->second.gpu_quota.size())return Result::DeviceUnavailable;
    const auto key=std::make_pair(address,device);
    const auto existing=pins_.find(key);
    if(pinned){
        if(existing!=pins_.end())return Result::Ok;
        const auto bytes=address_bytes_.at(address);
        if(bytes>client->second.gpu_quota[device]-
                 client->second.gpu_pinned[device]-
                 client->second.gpu_leased[device])return Result::Quota;
        if(!library_->set_gpu_pin(address,device,true))return Result::StateConflict;
        client->second.gpu_pinned[device]+=bytes;
        pins_.emplace(key,card);
        return Result::Ok;
    }
    if(existing==pins_.end())return Result::Invalid;
    if(!library_->set_gpu_pin(address,device,false))return Result::StateConflict;
    client->second.gpu_pinned[device]-=address_bytes_.at(address);
    pins_.erase(existing);
    return Result::Ok;
}

bool TensorStateLibraryTool::pin_gpu(std::uint64_t card,std::uint64_t address,
                                     std::size_t device,bool pinned){
    std::lock_guard<std::mutex> lock(mutex_);
    return set_gpu_pin_locked(card,address,device,pinned)==Result::Ok;
}

// ---- The volunteer cycle (case a11b5c9b1be74c9b9fb684790f7fd5a3) -------------------------------------------------
// Slot kinds and layout: include/gpu_work_row_tool.h (VolunteerSlots). A target is device + 1; 0 = choose.
namespace {
constexpr std::size_t kSlotWords=10;
constexpr std::uint64_t kSlotNull=0,kSlotWork=7,kSlotReceipt=9;
}

TensorStateLibraryTool::CycleCounts TensorStateLibraryTool::cycle_counts() const {
    std::lock_guard<std::mutex> lock(cycle_mutex_);
    return {work_queue_.size(),outstanding_.size(),completed_,retried_,batches_.size(),completed_by_device_};
}

// Rule 1, by the library's metrics: the device whose measured completions of this program at this lane count are
// fastest (work_performance, which the job rows record), else its device-level measure, weighted by the work already
// sent to it; an unmeasured device counts as fast, so it is measured. Returns device + 1, or 0 with no device.
std::uint64_t TensorStateLibraryTool::choose_target_locked(std::uint64_t program,std::uint64_t lanes) {
    const std::size_t devices=device_count();
    if(!devices)return 0;
    if(inflight_by_device_.size()<devices)inflight_by_device_.resize(devices,0);
    std::size_t best=0;
    std::uint64_t best_score=UINT64_MAX;
    for(std::size_t d=0;d<devices;++d){
        std::uint64_t average=0;
        if(const auto work=program?work_performance(ExecutionApi::Vulkan,d,program,lanes):std::nullopt;
           work&&work->samples)average=work->average_total_ns;
        else if(const auto perf=device_performance(d);perf&&perf->samples)average=perf->average_total_ns;
        const std::uint64_t score=(inflight_by_device_[d]+1)*(average+1);
        if(score<best_score){best_score=score;best=d;}
    }
    return best+1;
}

// One batch: receipts finish (rule 2: success completes it; failure re-queues it, first), then null slots take work.
void TensorStateLibraryTool::fill_and_settle_locked(std::vector<std::uint64_t>& words) {
    const std::size_t slots=words.size()/kSlotWords;
    for(std::size_t s=0;s<slots;++s){
        std::uint64_t* slot=&words[s*kSlotWords];
        if(slot[0]!=kSlotReceipt)continue;
        const std::uint64_t* receipt=slot+1;  // the job row's U64[9]: status, device, ..., work id at 4, ok at 8
        const auto found=outstanding_.find(receipt[4]);
        if(found!=outstanding_.end()){
            const auto device=static_cast<std::size_t>(receipt[1]);
            if(device<inflight_by_device_.size()&&inflight_by_device_[device])--inflight_by_device_[device];
            if(receipt[0]==0&&receipt[8]==1){
                ++completed_;
                if(device<completed_by_device_.size())++completed_by_device_[device];
            }
            else{++retried_;work_queue_.push_front(found->second);}
            outstanding_.erase(found);
        }
        std::fill(slot,slot+kSlotWords,0);
    }
    for(std::size_t s=0;s<slots&&!work_queue_.empty();++s){
        std::uint64_t* slot=&words[s*kSlotWords];
        if(slot[0]!=kSlotNull)continue;
        const WorkPacket packet=work_queue_.front();
        const std::uint64_t target=packet[5]?packet[5]:choose_target_locked(packet[6],packet[2]);
        if(!target)break;  // no device: the work waits in the queue
        slot[0]=kSlotWork;
        std::copy(packet.begin(),packet.begin()+5,slot+1);
        slot[6]=target;
        slot[7]=packet[6];  // the program
        outstanding_[packet[3]]=packet;
        if(inflight_by_device_.size()<target)inflight_by_device_.resize(target,0);
        ++inflight_by_device_[target-1];
        work_queue_.pop_front();
    }
}

bool TensorStateLibraryTool::cycle_step(ToolStackContext& context) {
    auto& stack=*context.stack.raw;
    const ValueTypeId tensor_type=ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
    // 0 nothing on the stack, 1 work packet, 2 batch, 3 anything else (an ordinary library request), 4 the 0 an
    // Input row pushes for an empty edge (its pad: nothing arrived there; the mint reads a 0 the same way).
    auto top=[&](std::uint64_t& id)->int{
        ValueTypeId tid=kInvalidValueTypeId;
        if(!raw_stack_peek_type(stack,tid))return 0;
        if(tid!=tensor_type){
            const ValueType* vt=ValueTypeRegistry::global().get(tid);
            if(vt&&vt->size==sizeof(double)){
                double v=1.0;
                if(raw_stack_peek_known_type(stack,&v,tid,sizeof(v))&&v==0.0)return 4;
            }else if(vt&&vt->size==sizeof(float)){
                float v=1.0f;
                if(raw_stack_peek_known_type(stack,&v,tid,sizeof(v))&&v==0.0f)return 4;
            }
            return 3;
        }
        nodus::tensors::AbstractTensorHandle h{};
        NodusTensorDesc d{};
        if(!raw_stack_peek_abstract_tensor(stack,h))return 3;
        if(!h.id)return 4;  // in a tensor table the Input row's pad for an empty edge is the null handle
        if(nodus_tensor_describe(h.id,&d)!=NODUS_OK||d.dtype!=NODUS_DTYPE_U64)return 3;
        id=h.id;
        if(d.rank==1&&(d.shape[0]==6||d.shape[0]==7))return 1;  // U64[6]: no program word (program 0)
        if(d.rank==2&&d.shape[1]==kSlotWords&&d.shape[0])return 2;
        return 3;
    };
    std::uint64_t id=0;
    int kind=top(id);
    std::lock_guard<std::mutex> lock(cycle_mutex_);
    if(kind==3)return false;
    if(kind==0&&work_queue_.empty()&&batches_.empty()&&outstanding_.empty())return false;
    while(kind==1||kind==2||kind==4){
        if(kind==4){
            ValueTypeId tid=kInvalidValueTypeId;
            const ValueType* vt=raw_stack_peek_type(stack,tid)?ValueTypeRegistry::global().get(tid):nullptr;
            std::uint64_t scratch=0;
            if(!vt||!raw_stack_pop_known_type(stack,&scratch,tid,vt->size,true))break;
            kind=top(id);
            continue;
        }
        nodus::tensors::AbstractTensorHandle h{};
        if(!raw_stack_pop_abstract_tensor(stack,h))break;
        if(kind==1){
            WorkPacket packet{};  // a U64[6] packet leaves the program word 0
            NodusTensorDesc pd{};
            const auto bytes=nodus_tensor_describe(h.id,&pd)==NODUS_OK?
                static_cast<std::int64_t>(pd.shape[0]*sizeof(std::uint64_t)):0;
            if(bytes&&nodus_tensor_read(h.id,0,packet.data(),bytes)==bytes&&
               packet[0]&&packet[2]&&packet[3])work_queue_.push_back(packet);
            nodus_tensor_destroy(h.id);  // the packet is read into the queue; the library keeps nothing else of it
        }else batches_.push_back(h.id);
        kind=top(id);
    }
    // Work waits and no batch is in hand: borrow an empty one from the mint.
    if(batches_.empty()&&!work_queue_.empty()&&context.row_ports&&context.row_ports->consume){
        std::uint64_t blank=0;
        std::size_t written=0;
        if(context.row_ports->consume(context.row_ports->host,0,&blank,sizeof(blank),&written)&&
           written==sizeof(blank)&&blank){
            NodusTensorDesc d{};
            if(nodus_tensor_describe(blank,&d)==NODUS_OK&&d.dtype==NODUS_DTYPE_U64&&d.rank==2&&
               d.shape[1]==kSlotWords&&d.shape[0])batches_.push_back(blank);
            else nodus_tensor_destroy(blank);
        }
    }
    std::vector<std::uint64_t> carrying;  // per batch in hand: whether any slot is filled
    for(const auto batch:batches_){
        NodusTensorDesc d{};
        bool filled=false;
        if(nodus_tensor_describe(batch,&d)==NODUS_OK){
            std::vector<std::uint64_t> words(static_cast<std::size_t>(d.shape[0])*kSlotWords);
            const auto bytes=static_cast<std::int64_t>(words.size()*sizeof(std::uint64_t));
            if(nodus_tensor_read(batch,0,words.data(),bytes)==bytes){
                fill_and_settle_locked(words);
                nodus_tensor_write(batch,0,words.data(),bytes);
                for(std::size_t s=0;s<words.size();s+=kSlotWords)filled=filled||words[s]!=kSlotNull;
            }
        }
        carrying.push_back(filled?1:0);
    }
    // One batch leaves on the Output row: the first that carries work, else (while work is out) an empty one to
    // bring the receipts back. Empty batches beyond that go back to the mint.
    std::size_t leave=batches_.size();
    for(std::size_t i=0;i<batches_.size()&&leave==batches_.size();++i)if(carrying[i])leave=i;
    if(leave==batches_.size()&&!batches_.empty()&&(!outstanding_.empty()||!work_queue_.empty()))leave=0;
    std::deque<std::uint64_t> kept;
    for(std::size_t i=0;i<batches_.size();++i){
        const auto batch=batches_[i];
        if(i==leave){
            if(!raw_stack_push_abstract_tensor(stack,{batch}))kept.push_back(batch);
            continue;
        }
        if(!carrying[i]&&context.row_ports&&context.row_ports->publish&&
           context.row_ports->publish(context.row_ports->host,1,&batch,sizeof(batch)))continue;
        kept.push_back(batch);
    }
    batches_=std::move(kept);
    context.succeeded=true;
    return true;
}

void TensorStateLibraryTool::execute_stack(ToolStackContext& context) {
    context.succeeded=false;
    if(!context.stack.raw)return;
    if(cycle_step(context))return;
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
                receipt[29+i]=stats.vram[i].pinned_bytes;
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
    std::uint64_t bytes=0,charge=0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto client=cards_.find(card);
        auto owner=address_card_.find(address);
        if(!library_||client==cards_.end()||owner==address_card_.end()||
           owner->second!=card||device>=client->second.gpu_quota.size())return {};
        bytes=address_bytes_[address];
        charge=pins_.contains({address,device})?0:bytes;
        if(charge>client->second.gpu_quota[device]-
                  client->second.gpu_pinned[device]-client->second.gpu_leased[device])
            return {};
        client->second.gpu_leased[device]+=charge;
        ++inflight_gpu_loans_;
    }
    auto lease=library_->lease_gpu(address,device,writable);
    if(!lease){
        std::lock_guard<std::mutex> lock(mutex_);
        cards_.at(card).gpu_leased[device]-=charge;
        --inflight_gpu_loans_;
        return {};
    }
    return GpuLoan(this,card,device,charge,std::move(lease));
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

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::TensorStateLibraryTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
