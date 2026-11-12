// exec_row_tool: a tool row that owns one external native program and passes byte spans through it.
//
// With a compressor as its program this is the byte-span compressor in waiting: a child is already started and
// waiting on its input, the row does nothing while its stack holds no span, and a span that arrives is handed to the
// waiting child at once while the next child is started behind it.
//
// Stack currency is the ordinary one for plugin rows: an AbstractTensorHandle (VT_ABSTRACT_TENSOR). A span is a
// tensor's payload, whatever its dtype and shape; what comes back is a rank-1 U8 tensor. As with every tensor row the
// input handle is taken from the stack and not destroyed (its producer owns it; the row copies the bytes in the same
// tick), and the output handle belongs to whoever takes it from the stack. For those handles to reach the row the
// module's stack contract must be AbstractTensorPointer and its edges typed, as for any tensor row.
//
// The row never blocks the tick. Each execute_stack:
//   1. collects what its children have written; a child that has exited makes one finished span,
//   2. takes every span handle on top of the stack (none is the normal case, and costs nothing),
//   3. pushes at most one finished span, oldest first (its one declared return).
// Spans come back in the order they went in, however long each took.
//
// The program is held to the rules in exec_pipe.h (never Git, never a working directory inside the core) against the
// core the HOST declared (exec_row.h), and a refused or missing program refuses the span: the handle is put back and
// ToolStackContext::succeeded is cleared, so the rows after this one do not take the stack for a result. A child that
// fails makes no span; the tick that finds it is refused the same way and the reason goes to the host log.
//
// Settings are the tool's own saved state (exec_row.h), read from ToolInitContext::serialized_path.
// Windows only, as exec_pipe.h is.
#include "exec_row.h"

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

namespace {

HostAPI* g_host = nullptr;

class ExecRowTool : public ITool {
public:
    const char* id_cstr() const noexcept override { return "exec_row"; }
    const char* name_cstr() const noexcept override { return "Exec"; }

    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) deserialize(in);
        }
    }

    void shutdown() override {
        waiting_.reset();
        working_.clear();
        queued_.clear();
        for (uint64_t handle : ready_) nodus_tensor_destroy(handle);
        ready_.clear();
    }

    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}

    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t idx) const override {
        if (idx == 0) return {ToolPortKind::Argument, 1};
        if (idx == 1) return {ToolPortKind::Return, 1};
        return {};
    }

    bool serialize(OutputArchive& out) const override { return nodus::exec::write_settings(out, settings_); }

    bool deserialize(InputArchive& in) override {
        nodus::exec::RowSettings read;
        if (!nodus::exec::read_settings(in, read)) return false;
        shutdown();
        settings_ = std::move(read);
        refused_.clear();
        keep_one_waiting();
        return true;
    }

    void execute_stack(ToolStackContext& ctx) override {
        if (!ctx.stack.raw) return;
        RawStackFrame& frame = *ctx.stack.raw;
        if (!collect()) ctx.succeeded = false;

        // Every span on top of the stack, in the order they were pushed.
        std::vector<nodus::tensors::AbstractTensorHandle> arrived;
        nodus::tensors::AbstractTensorHandle handle{};
        while (raw_stack_pop_abstract_tensor(frame, handle)) arrived.push_back(handle);
        if (!arrived.empty()) {
            // All of them or none: nothing is queued until every one has been read and a program may start.
            std::vector<std::string> spans(arrived.size());
            bool taken = keep_one_waiting();
            const bool startable = taken;
            for (size_t i = arrived.size(); taken && i-- > 0;) taken = read_span(arrived[i], spans[arrived.size() - 1 - i]);
            if (!taken) {
                for (size_t back = arrived.size(); back-- > 0;) raw_stack_push_abstract_tensor(frame, arrived[back]);   // as it was
                ctx.succeeded = false;
                report(startable ? "a handle on the stack is not a readable tensor" : refused_);
                return;
            }
            for (std::string& span : spans) queued_.push_back(std::move(span));
            feed();
        }

        if (ctx.succeeded && !ready_.empty()) {
            nodus::tensors::AbstractTensorHandle out{};
            out.id = ready_.front();
            if (raw_stack_push_abstract_tensor(frame, out)) ready_.pop_front();
        }
    }

private:
    struct Work {
        std::unique_ptr<nodus::exec::Child> child;
        std::string out;
    };

    // A child started and waiting on its input, so a span never waits for a process to start.
    bool keep_one_waiting() {
        if (waiting_) return true;
        nodus::exec::Program program;
        if (!nodus::exec::row_program(settings_, program, refused_)) return false;
        auto child = std::make_unique<nodus::exec::Child>();
        if (!child->start(program, refused_)) return false;
        waiting_ = std::move(child);
        return true;
    }

    static bool read_span(nodus::tensors::AbstractTensorHandle handle, std::string& span) {
        // The payload as the descriptor counts it. A mapping's length is the arena lease, which is rounded up.
        NodusTensorDesc desc{};
        if (handle.id == 0 || nodus_tensor_describe(handle.id, &desc) != NODUS_OK) return false;
        span.assign(static_cast<size_t>(desc.total_bytes), '\0');
        return span.empty() ||
               nodus_tensor_read(handle.id, 0, span.data(), span.size()) == static_cast<int64_t>(span.size());
    }

    // Queued spans go to children, oldest first, while fewer than most_at_once are working.
    void feed() {
        while (!queued_.empty() && working_.size() < settings_.most_at_once && keep_one_waiting()) {
            Work work;
            work.child = std::move(waiting_);
            work.child->send(std::move(queued_.front()));
            work.child->finish_input();
            queued_.pop_front();
            working_.push_back(std::move(work));
            keep_one_waiting();
        }
    }

    // False when a child failed: its span is lost and the reason is reported.
    bool collect() {
        bool all_good = true;
        for (Work& work : working_) work.child->take(work.out);
        while (!working_.empty() && working_.front().child->finished()) {
            Work& done = working_.front();
            done.child->take(done.out);
            const int code = done.child->exit_code();
            uint64_t made = 0;
            if (code == 0) {
                const uint64_t shape[1] = {static_cast<uint64_t>(done.out.size())};
                made = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
                if (made && !done.out.empty() &&
                    nodus_tensor_write(made, 0, done.out.data(), done.out.size()) != static_cast<int64_t>(done.out.size())) {
                    nodus_tensor_destroy(made);
                    made = 0;
                }
            }
            if (made) {
                ready_.push_back(made);
            } else {
                all_good = false;
                report(code == 0 ? "the result could not be made into a tensor"
                                 : "the program ended with exit code " + std::to_string(code) + "; its span is lost");
            }
            working_.pop_front();
        }
        feed();
        return all_good;
    }

    void report(const std::string& why) {
        if (g_host) g_host->log(("exec row: " + why).c_str());
    }

    nodus::exec::RowSettings settings_;
    std::unique_ptr<nodus::exec::Child> waiting_;
    std::deque<std::string> queued_;     // spans that arrived while most_at_once children were working
    std::deque<Work> working_;           // oldest first
    std::deque<uint64_t> ready_;         // finished spans not yet pushed, oldest first
    std::string refused_;
};

} // namespace

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new ExecRowTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
#if !defined(NODUS_PLUGIN_COMPOSITE)
extern "C" NODUS_PLUGIN_EXPORT int plugin_init(HostAPI* host) { g_host = host; return 1; }
extern "C" NODUS_PLUGIN_EXPORT void plugin_shutdown() { g_host = nullptr; }
#endif
