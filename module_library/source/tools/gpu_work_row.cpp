// gpu_work_row: the volunteer cycle's work row, one per GPU (include/gpu_work_row_tool.h, case
// a11b5c9b1be74c9b9fb684790f7fd5a3). The row tool's source, in the module library; the image registers it from this
// source (src/gpu_work_row_tool.cpp), over a KernelLibrarySession's device job rows.
#include "gpu_work_row_tool.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <cstdio>
#include <vector>

namespace nodus::gpu {

GpuWorkRowTool::GpuWorkRowTool(std::string id, JobRow job, std::size_t device, std::uint64_t program,
                               std::size_t capacity)
    : id_(std::move(id)), job_(std::move(job)), device_(device), capacity_(capacity ? capacity : 1),
      program_(program) {}
GpuWorkRowTool::GpuWorkRowTool() = default;
GpuWorkRowTool::~GpuWorkRowTool() { shutdown(); }

void GpuWorkRowTool::initialize(const ToolInitContext&) {
    if (!job_ || runner_.joinable()) return;
    job_->initialize({});
    stop_ = false;
    runner_ = std::thread([this] { run(); });
}

void GpuWorkRowTool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    if (runner_.joinable()) runner_.join();
}

GpuWorkRowTool::Counts GpuWorkRowTool::counts() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return counts_;
}

// The runner thread: one dispatch at a time through the job row, on its own stack frame.
void GpuWorkRowTool::run() {
    for (;;) {
        Packet packet{};
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || !pending_.empty(); });
            if (stop_) return;
            packet = pending_.front();
            pending_.pop_front();
            ++running_;
        }
        const Receipt receipt = dispatch(packet);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_;
            finished_.push_back(receipt);
            ++counts_.finished;
        }
    }
}

// The job row's own work token (U64[4], or U64[5] with Y invocations) in, its receipt U64[9] out. A job row that
// refuses leaves its token on the stack; the receipt then says so (status 4) with the work id, so the library retries.
GpuWorkRowTool::Receipt GpuWorkRowTool::dispatch(const Packet& p) {
    Receipt refused{4, static_cast<std::uint64_t>(device_), p[2], 0, p[3], p[1], 0, 0, 0};
    const std::uint64_t words = p[4] ? 5 : 4;
    const std::uint64_t shape[1] = {words};
    const std::uint64_t token = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
    if (!token) return refused;
    if (nodus_tensor_write(token, 0, p.data(), words * sizeof(std::uint64_t)) !=
        static_cast<std::int64_t>(words * sizeof(std::uint64_t))) {
        nodus_tensor_destroy(token);
        return refused;
    }
    RawStackFrame frame;
    if (!raw_stack_init_frame(frame, 4096)) {
        nodus_tensor_destroy(token);
        return refused;
    }
    raw_stack_push_abstract_tensor(frame, {token});
    ToolStackContext context;
    context.stack.raw = &frame;
    job_->execute_stack(context);
    Receipt receipt = refused;
    nodus::tensors::AbstractTensorHandle out{};
    if (context.succeeded && raw_stack_pop_abstract_tensor(frame, out)) {
        NodusTensorDesc desc{};
        if (nodus_tensor_describe(out.id, &desc) == NODUS_OK && desc.dtype == NODUS_DTYPE_U64 && desc.rank == 1 &&
            desc.shape[0] == 9 && nodus_tensor_read(out.id, 0, receipt.data(), sizeof(receipt)) ==
                                      static_cast<std::int64_t>(sizeof(receipt))) {
        } else {
            receipt = refused;
        }
        nodus_tensor_destroy(out.id);
    }
    while (raw_stack_pop_abstract_tensor(frame, out)) {}  // a refusing row restored the token
    nodus_tensor_destroy(token);                           // the job row reads its token and never keeps it
    raw_stack_free_mask(frame);
    if (frame.backend) gp_mem_backend_release(frame.backend);
    return receipt;
}

void GpuWorkRowTool::execute_stack(ToolStackContext& context) {
    context.succeeded = true;  // a batch passes on whether or not this row took anything from it
    if (!context.stack.raw) return;
    RawStackFrame& stack = *context.stack.raw;
    nodus::tensors::AbstractTensorHandle batch{};
    if (!raw_stack_peek_abstract_tensor(stack, batch)) return;
    NodusTensorDesc desc{};
    if (nodus_tensor_describe(batch.id, &desc) != NODUS_OK || desc.dtype != NODUS_DTYPE_U64 || desc.rank != 2 ||
        desc.shape[1] != VolunteerSlots::kWords || !desc.shape[0])
        return;  // not a batch: leave it for the rows after this one
    const std::size_t slots = static_cast<std::size_t>(desc.shape[0]);
    std::vector<std::uint64_t> words(slots * VolunteerSlots::kWords);
    const auto bytes = static_cast<std::int64_t>(words.size() * sizeof(std::uint64_t));
    if (nodus_tensor_read(batch.id, 0, words.data(), bytes) != bytes) return;
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (std::size_t s = 0; s < slots; ++s) {
            std::uint64_t* slot = &words[s * VolunteerSlots::kWords];
            if (slot[0] != VolunteerSlots::kWork || slot[6] != device_ + 1) continue;
            if (slot[7] && slot[7] != program_) continue;  // another program's work on this device: its own row's
            if (!job_ || pending_.size() + running_ >= capacity_) break;  // no room: the slot stays for a later pass
            pending_.push_back({slot[1], slot[2], slot[3], slot[4], slot[5], slot[6], slot[7]});
            std::fill(slot, slot + VolunteerSlots::kWords, 0);
            ++counts_.taken;
            changed = true;
        }
        for (std::size_t s = 0; s < slots && !finished_.empty(); ++s) {
            std::uint64_t* slot = &words[s * VolunteerSlots::kWords];
            if (slot[0] != VolunteerSlots::kNull) continue;
            slot[0] = VolunteerSlots::kReceipt;
            std::copy(finished_.front().begin(), finished_.front().end(), slot + 1);
            finished_.pop_front();
            ++counts_.receipts_written;
            changed = true;
        }
    }
    if (changed) wake_.notify_all();
    if (changed && nodus_tensor_write(batch.id, 0, words.data(), bytes) != bytes)
        std::fprintf(stderr, "[%s] batch write failed\n", id_.c_str());
}

}  // namespace nodus::gpu

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::gpu::GpuWorkRowTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
