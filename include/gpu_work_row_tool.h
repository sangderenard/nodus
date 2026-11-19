#pragma once
// The row's source: module_library/source/tools/gpu_work_row.cpp. The volunteer cycle's work row
// (case a11b5c9b1be74c9b9fb684790f7fd5a3): "A GPU is a work row; adding a GPU is adding a row."
#include "tool_api.h"
#include "tool_registry.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nodus::gpu {

// The cycle's batch: U64[N,10], one slot per row. Word 0 of a slot is its kind (the payload's size); the other nine
// words are the payload. The kinds (owner-agreed 2026-11-18):
//   0  null sentinel (an empty slot)
//   6  work packet: card, base, invocations, work id, Y invocations (0 = one-dimensional), target
//   9  receipt: the job row's receipt U64[9], unchanged
// A target names a device as device + 1; 0 asks the library to choose (rule 1). The batch is a tensor the tensor
// mint lends; it travels library -> work -> library on two lossless edges.
struct VolunteerSlots {
    static constexpr std::size_t kWords = 10;
    static constexpr std::uint64_t kNull = 0, kWork = 6, kReceipt = 9;
};

// One GPU's work row. It wraps that device's job row (a registered shader job row <contract>.vulkan.<device>) and
// runs it on its own thread, so a dispatch never holds up the table. Per batch it is never blocked:
//   - it takes each work slot whose target is its own device while its runner has room, and leaves a null slot;
//     a slot it has no room for stays for a later pass;
//   - it writes each finished receipt into a null slot; receipts wait for a batch when none is passing.
class GpuWorkRowTool final : public ITool {
public:
    using JobRow = std::unique_ptr<ITool, std::function<void(ITool*)>>;
    GpuWorkRowTool(std::string id, JobRow job, std::size_t device, std::size_t capacity);
    GpuWorkRowTool();  // no job row: the row passes batches through untouched
    ~GpuWorkRowTool() override;
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return "GPU work row"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t index) const override {
        if (index == 0) return {ToolPortKind::Argument, 1};
        if (index == 1) return {ToolPortKind::Return, 1};
        return {};
    }
    void initialize(const ToolInitContext&) override;
    void shutdown() override;
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext&) override;

    struct Counts { std::uint64_t taken = 0, finished = 0, receipts_written = 0; };
    Counts counts() const;

private:
    using Packet = std::array<std::uint64_t, 6>;
    using Receipt = std::array<std::uint64_t, 9>;
    void run();
    Receipt dispatch(const Packet& packet);

    std::string id_ = "nodus.volunteer.work";
    JobRow job_;
    std::size_t device_ = 0, capacity_ = 4;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::deque<Packet> pending_;
    std::deque<Receipt> finished_;
    std::size_t running_ = 0;
    bool stop_ = false;
    Counts counts_;
    std::thread runner_;
};

// The cycle's rows on one KernelLibrarySession: the library row nodus.volunteer.library (the session's own state
// library) and, for each device, the work row nodus.volunteer.work.<job>.vulkan.<device> around <job>.vulkan.<device>.
bool register_volunteer_work_rows(ToolRegistry& registry, const std::string& job_id, std::size_t devices,
                                  std::size_t capacity, std::vector<std::string>* registered);

}  // namespace nodus::gpu
