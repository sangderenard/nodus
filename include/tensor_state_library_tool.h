#pragma once
#include "tensor_state_library.h"
#include "tool_api.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cstdint>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <unordered_map>
#include <vector>

namespace nodus::gpu {

// FIFO-facing tensor library row. Both inputs and the receipt are abstract
// tensor tokens, so ordinary Nodus table edges own sequencing and backpressure.
// No raw Vulkan handle is published through a graph edge.
class TensorStateLibraryTool final : public ITool {
public:
    explicit TensorStateLibraryTool(
        std::uint64_t ram_budget_bytes=1024ull*1024*1024,
        std::vector<std::uint64_t> device_budget_bytes={})
        :ram_budget_(ram_budget_bytes),
         device_budget_bytes_(std::move(device_budget_bytes)){}
    enum class Operation : std::uint64_t {
        Card=1, Put=2, Remove=3, Status=4, ChooseResident=5
    };
    enum class Result : std::uint64_t {
        Ok=0, Invalid=1, CardUnknown=2, Quota=3, StateConflict=4,
        DeviceUnavailable=5, LibraryUnavailable=6, NoCandidate=7
    };
    static constexpr std::size_t kRequestWords=12, kReceiptWords=32;
    // Request U64[12]: op, card, process, address, RAM quota, GPU0 quota,
    // GPU1 quota, GPU2 quota, flags, reserved... . Put reads its second
    // Argument port as an ordinary typed tensor. ChooseResident takes U64[N,2]
    // candidate rows {base_address, field_count}, device in request[5], and
    // selects the warmest admissible candidate; equal scores keep FIFO order.
    // Other operations ignore the payload.
    const char* id_cstr() const noexcept override {return "nodus.tensor.state.library";}
    const char* name_cstr() const noexcept override {return "Tensor state library";}
    ToolCaps caps() const override {return ToolCaps::None;}
    int32_t port_count() const override {return 3;}
    ToolPortSpec port_spec(int32_t index) const override {
        if(index==0||index==1)return {ToolPortKind::Argument,1};
        if(index==2)return {ToolPortKind::Return,1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override;
    void tick(double,HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;

    class GpuLoan {
    public:
        GpuLoan()=default;
        GpuLoan(TensorStateLibraryTool* owner,std::uint64_t card,
                std::size_t device,std::uint64_t bytes,
                TensorStateLibrary::GpuLease lease) noexcept
            :owner_(owner),card_(card),device_(device),bytes_(bytes),
             lease_(std::move(lease)){}
        GpuLoan(const GpuLoan&)=delete;
        GpuLoan& operator=(const GpuLoan&)=delete;
        GpuLoan(GpuLoan&& other) noexcept { *this=std::move(other); }
        GpuLoan& operator=(GpuLoan&& other) noexcept {
            if(this!=&other){reset();owner_=std::exchange(other.owner_,nullptr);
                card_=other.card_;device_=other.device_;bytes_=other.bytes_;
                lease_=std::move(other.lease_);}return *this;
        }
        ~GpuLoan(){reset();}
        void* handle() const noexcept {return lease_.handle();}
        std::size_t bytes() const noexcept {return lease_.bytes();}
        explicit operator bool() const noexcept {return static_cast<bool>(lease_);}
        void reset() noexcept;
    private:
        TensorStateLibraryTool* owner_=nullptr;
        std::uint64_t card_=0,bytes_=0;
        std::size_t device_=0;
        TensorStateLibrary::GpuLease lease_;
    };
    class CpuLoan {
    public:
        CpuLoan()=default;
        CpuLoan(TensorStateLibrary::CpuLease lease) noexcept
            :lease_(std::move(lease)){}
        CpuLoan(const CpuLoan&)=delete;
        CpuLoan& operator=(const CpuLoan&)=delete;
        CpuLoan(CpuLoan&&)=default;
        CpuLoan& operator=(CpuLoan&&)=default;
        void* data() const noexcept {return lease_.data();}
        std::size_t bytes() const noexcept {return lease_.bytes();}
        explicit operator bool() const noexcept {return static_cast<bool>(lease_);}
    private:
        TensorStateLibrary::CpuLease lease_;
    };
    // Same-process GPU rows call this only after a FIFO work token is theirs.
    // The library card limits simultaneously leased bytes for that client on each GPU.
    GpuLoan borrow_gpu(std::uint64_t card,std::uint64_t address,
                       std::size_t device,bool writable);
    CpuLoan borrow_cpu(std::uint64_t card,std::uint64_t address,bool writable);
    std::optional<TensorStateLibrary::Spec> describe_state(
        std::uint64_t card,std::uint64_t address) const;
    struct Candidate { std::uint64_t base=0, fields=0; };
    static_assert(sizeof(Candidate)==2*sizeof(std::uint64_t));
    struct ResidencyChoice {
        std::size_t index=0;
        std::uint64_t base=0,resident_bytes=0,total_bytes=0,fields=0;
    };
    // The manager can rank its ready work without transferring a tensor.
    std::optional<ResidencyChoice> choose_resident(
        std::uint64_t card,std::size_t device,
        std::span<const Candidate> candidates) const;
    std::uint64_t resident_bytes(std::uint64_t card,std::uint64_t address,
                                 std::size_t device) const;
    struct DevicePerformance {
        std::uint64_t samples=0,average_lease_ns=0,
                      average_dispatch_ns=0,average_total_ns=0;
        std::uint64_t cache_resident_bytes=0,cache_budget_bytes=0;
    };
    // Completion receipts feed the same library that owns the device cache.
    // These are observed host timings, not GPU hardware utilization.
    void record_completion(std::size_t device,std::uint64_t lease_ns,
                           std::uint64_t dispatch_ns,std::uint64_t total_ns);
    std::optional<DevicePerformance> device_performance(std::size_t device) const;
    enum class ExecutionApi : std::uint8_t { Cpu=0, Vulkan=1, OpenGL=2 };
    struct WorkPerformance {
        std::uint64_t samples=0,average_lease_ns=0,
                      average_dispatch_ns=0,average_total_ns=0;
    };
    // Exact program and lane shape, with API kept distinct even if two
    // runtimes select the same physical adapter. No cross-shape averages.
    void record_work_completion(ExecutionApi api,std::size_t device,
        std::uint64_t program,std::uint64_t lanes,
        std::uint64_t lease_ns,std::uint64_t dispatch_ns,
        std::uint64_t total_ns);
    std::optional<WorkPerformance> work_performance(ExecutionApi api,
        std::size_t device,std::uint64_t program,std::uint64_t lanes) const;
    std::size_t device_count() const;

private:
    struct Card {
        std::uint64_t process=0,ram_quota=0,ram_used=0;
        std::vector<std::uint64_t> gpu_quota,gpu_leased;
    };
    void release_gpu_loan(std::uint64_t card,std::size_t device,
                          std::uint64_t bytes,
                          TensorStateLibrary::GpuLease& lease) noexcept;
    Result handle(const std::array<std::uint64_t,kRequestWords>&,
                  std::uint64_t payload,
                  std::array<std::uint64_t,kReceiptWords>& receipt);
    std::optional<ResidencyChoice> choose_resident_locked(
        std::uint64_t card,std::size_t device,
        std::span<const Candidate> candidates) const;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<void,decltype(&gp_mem_backend_release)>> contexts_;
    std::unique_ptr<TensorStateLibrary> library_;
    std::unordered_map<std::uint64_t,Card> cards_;
    std::unordered_map<std::uint64_t,std::uint64_t> address_card_,address_bytes_;
    std::uint64_t ram_budget_;
    std::vector<std::uint64_t> device_budget_bytes_;
    std::uint64_t inflight_gpu_loans_=0;
    struct CompletionWindow {
        std::array<std::uint64_t,8> lease{},dispatch{},total{};
        std::size_t count=0,next=0;
        std::uint64_t last_update_ns=0;
    };
    std::vector<CompletionWindow> completion_windows_;
    using WorkKey=std::tuple<ExecutionApi,std::size_t,std::uint64_t,std::uint64_t>;
    std::map<WorkKey,CompletionWindow> work_windows_;
};

} // namespace nodus::gpu
