// exec_row_test <exec_row_tool.dll> <compressor> <scratch-dir> [core-root]
// Drives the exec tool row the way the thread manager drives any plugin row: a module stack that is reset every
// tick, a number left by an input row that had nothing unread, a tensor handle when a span arrived, one
// execute_stack per tick, and whatever the row left on top of the stack afterwards.
// The compressor must compress standard input to standard output with "-c" and decompress with "-d -c".
#include "exec_row.h"

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

using Handle = nodus::tensors::AbstractTensorHandle;
using Clock = std::chrono::steady_clock;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

struct RecordingHost : HostAPI {
    std::vector<std::string> lines;
    void log(const char* message) override { lines.emplace_back(message ? message : ""); }
    void draw_text(const TextRenderArgs&) override {}
    bool said(const char* part) const {
        for (const std::string& line : lines)
            if (line.find(part) != std::string::npos) return true;
        return false;
    }
};

static CreateToolFn create_tool = nullptr;
static void (*destroy_tool)(ITool*) = nullptr;

static uint64_t tensor_of(const std::string& bytes) {
    const uint64_t shape[1] = {bytes.size()};
    const uint64_t handle = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
    if (handle && !bytes.empty()) nodus_tensor_write(handle, 0, bytes.data(), bytes.size());
    return handle;
}

static std::string bytes_of(uint64_t handle) {
    NodusTensorDesc desc{};
    if (nodus_tensor_describe(handle, &desc) != NODUS_OK) return {};
    std::string out(static_cast<size_t>(desc.total_bytes), '\0');
    if (!out.empty()) nodus_tensor_read(handle, 0, out.data(), out.size());
    return out;
}

// One row in one module: its settings file, its stack, and the tick.
struct Row {
    ITool* tool = nullptr;
    RawStackFrame frame{};
    double longest_ms = 0.0;

    Row(const nodus::exec::RowSettings& settings, const std::string& file) {
        { FileOutputArchive out(file.c_str()); nodus::exec::write_settings(out, settings); }
        tool = create_tool();
        ToolInitContext init{};
        init.serialized_path = file.c_str();
        tool->initialize(init);
        raw_stack_init_frame(frame, 4096);
    }
    ~Row() {
        tool->shutdown();
        destroy_tool(tool);
        raw_stack_free_mask(frame);
    }

    // One tick. `spans` are what the input rows delivered (none: an input row leaves a number instead).
    // Returns the handle the row left on top of the stack, 0 when it left none.
    uint64_t tick(const std::vector<uint64_t>& spans, bool* succeeded = nullptr) {
        raw_stack_reset(frame);
        if (spans.empty()) {
            const double nothing = 0.0;
            raw_stack_push_typed(frame, &nothing, ValueTypeRegistry::global().builtin(VT_FLOAT64));
        }
        for (uint64_t span : spans) raw_stack_push_abstract_tensor(frame, Handle{span});
        ToolStackContext context{};
        context.stack.raw = &frame;
        context.stack.default_type = ValueTypeRegistry::global().builtin(VT_FLOAT64);
        const auto before = Clock::now();
        tool->execute_stack(context);
        const double took = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
        if (took > longest_ms) longest_ms = took;
        if (succeeded) *succeeded = context.succeeded;
        Handle top{};
        return raw_stack_pop_abstract_tensor(frame, top) ? top.id : 0;
    }

    // Tick with nothing arriving until the row leaves a span, or the limit passes.
    uint64_t wait_for_span(int limit_ms) {
        const auto deadline = Clock::now() + std::chrono::milliseconds(limit_ms);
        while (Clock::now() < deadline) {
            if (const uint64_t out = tick({})) return out;
            Sleep(1);
        }
        return 0;
    }
};

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: exec_row_test <exec_row_tool.dll> <compressor> <scratch-dir> [core-root]\n"); return 2; }
    const std::string dll = argv[1], compressor = argv[2], scratch = argv[3], core = argc > 4 ? argv[4] : "C:/dev/Powershell";
    HMODULE module = LoadLibraryA(dll.c_str());
    if (!module) { std::fprintf(stderr, "the row DLL did not load (error %lu)\n", GetLastError()); return 2; }
    create_tool = reinterpret_cast<CreateToolFn>(GetProcAddress(module, "create_tool"));
    destroy_tool = reinterpret_cast<void (*)(ITool*)>(GetProcAddress(module, "destroy_tool"));
    auto plugin_init = reinterpret_cast<int (*)(HostAPI*)>(GetProcAddress(module, "plugin_init"));
    if (!create_tool || !destroy_tool || !plugin_init) { std::fprintf(stderr, "the row DLL lacks the plugin exports\n"); return 2; }
    RecordingHost host;
    plugin_init(&host);
    const std::string file = scratch + "/exec_row_settings.bin";
    const nodus::exec::RowSettings pack{compressor, {"-c"}, scratch, 2}, unpack{compressor, {"-d", "-c"}, scratch, 2};

    // A host that declared no core gets a row that starts nothing.
    SetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", nullptr);
    {
        Row row(pack, file);
        const uint64_t span = tensor_of("abc");
        bool succeeded = true;
        const uint64_t top = row.tick({span}, &succeeded);
        check(!succeeded && top == span, "with no declared core the span is refused and left on the stack");
        check(host.said("declared no core"), "and the host log says why");
        nodus_tensor_destroy(span);
    }
    SetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", ("C:/nowhere-at-all;" + core).c_str());

    // Declared port layout.
    {
        Row row(pack, file);
        check(row.tool->port_count() == 2 && row.tool->port_spec(0).kind == ToolPortKind::Argument &&
              row.tool->port_spec(0).count == 1 && row.tool->port_spec(1).kind == ToolPortKind::Return &&
              row.tool->port_spec(1).count == 1, "the row declares one argument and one return");

        // In waiting: no span, nothing done, the stack as the input row left it.
        bool succeeded = false, quiet = true;
        for (int i = 0; i < 50; ++i) {
            const size_t before = 8;
            quiet = quiet && row.tick({}, &succeeded) == 0 && succeeded && row.frame.byte_count == before;
        }
        check(quiet, "with no span the row does nothing and leaves the stack alone");
        check(row.longest_ms < 5.0, "an idle tick costs next to nothing");
    }

    // A span through the compressor and back through a second row.
    std::string text;
    for (int i = 0; i < 200000; ++i) text += "triangle_map[" + std::to_string(i % 4096) + "] ";
    {
        Row packer(pack, file), unpacker(unpack, file);
        const uint64_t span = tensor_of(text);
        const auto started = Clock::now();
        bool succeeded = false;
        uint64_t packed = packer.tick({span}, &succeeded);
        check(succeeded, "a span is taken");
        nodus_tensor_destroy(span);                       // its producer owns it; the row already has the bytes
        if (!packed) packed = packer.wait_for_span(60000);
        const std::string squeezed = bytes_of(packed);
        check(packed != 0 && !squeezed.empty() && squeezed.size() < text.size() / 4, "the compressed span comes back, much smaller");
        NodusTensorDesc desc{};
        nodus_tensor_describe(packed, &desc);
        check(desc.dtype == NODUS_DTYPE_U8 && desc.rank == 1, "as a rank-1 U8 tensor");
        uint64_t back = unpacker.tick({packed});
        if (!back) back = unpacker.wait_for_span(60000);
        const double took = std::chrono::duration<double, std::milli>(Clock::now() - started).count();
        check(back != 0 && bytes_of(back) == text, "a second row returns the same bytes");
        std::printf("     %zu bytes -> %zu -> back in %.0f ms; longest tick %.2f ms (pack) %.2f ms (unpack)\n",
                    text.size(), squeezed.size(), took, packer.longest_ms, unpacker.longest_ms);
        check(packer.longest_ms < 50.0 && unpacker.longest_ms < 50.0, "no tick waited for the program");
        nodus_tensor_destroy(packed);
        nodus_tensor_destroy(back);
    }

    // Order: a large span, then small ones on the following ticks, two of them in one tick; more than most_at_once.
    {
        Row packer(pack, file), unpacker(unpack, file);
        std::vector<std::string> sent = {text, "one", std::string(300000, 'z'), "", "five", text.substr(0, 100000)};
        std::vector<uint64_t> inputs;
        for (const std::string& bytes : sent) inputs.push_back(tensor_of(bytes));
        std::vector<uint64_t> packed;
        auto note = [&](uint64_t out) { if (out) packed.push_back(out); };
        note(packer.tick({inputs[0]}));
        note(packer.tick({inputs[1]}));
        note(packer.tick({inputs[2], inputs[3]}));        // two input rows delivered in the same tick
        note(packer.tick({inputs[4]}));
        note(packer.tick({inputs[5]}));
        for (uint64_t input : inputs) nodus_tensor_destroy(input);
        const auto deadline = Clock::now() + std::chrono::seconds(60);
        while (packed.size() < sent.size() && Clock::now() < deadline) { note(packer.tick({})); Sleep(1); }
        check(packed.size() == sent.size(), "six spans in, six out");
        bool in_order = packed.size() == sent.size();
        for (size_t i = 0; in_order && i < packed.size(); ++i) {
            uint64_t back = unpacker.tick({packed[i]});
            if (!back) back = unpacker.wait_for_span(60000);
            in_order = back != 0 && bytes_of(back) == sent[i];
            if (back) nodus_tensor_destroy(back);
        }
        check(in_order, "and they come back in the order they went in, an empty span among them");
        for (uint64_t out : packed) nodus_tensor_destroy(out);
        check(packer.tick({}) == 0, "after which the row is in waiting again");
    }

    // The rules a row's program is held to.
    {
        const uint64_t span = tensor_of("abc");
        bool succeeded = true;
        host.lines.clear();
        { Row row({"C:/Program Files/Git/cmd/git.exe", {"--version"}, scratch, 1}, file);
          check(row.tick({span}, &succeeded) == span && !succeeded && host.said("never runs Git"), "a row told to run Git refuses the span"); }
        host.lines.clear();
        { Row row({compressor, {"-c"}, core + "/turing", 1}, file);
          check(row.tick({span}, &succeeded) == span && !succeeded && host.said("inside the core"), "so does a working directory inside the core"); }
        host.lines.clear();
        { Row row({scratch + "/no-such-program.exe", {}, scratch, 1}, file);
          check(row.tick({span}, &succeeded) == span && !succeeded && host.said("could not be started"), "and a program that is not there"); }
        host.lines.clear();
        { Row row(pack, file);
          const uint64_t other = tensor_of("def");
          nodus_tensor_destroy(other);
          check(row.tick({span, other}, &succeeded) == other && !succeeded, "a handle that is no tensor refuses the tick, the stack as it was");
          Handle beneath{};
          check(raw_stack_pop_abstract_tensor(row.frame, beneath) && beneath.id == span, "with the span beneath it still there"); }
        nodus_tensor_destroy(span);
    }

    // A program that fails loses its span, says so, and the row goes on working.
    {
        Row unpacker(unpack, file);
        host.lines.clear();
        const uint64_t junk = tensor_of("this is not a compressed span at all, not even a little");
        unpacker.tick({junk});
        nodus_tensor_destroy(junk);
        bool refused_once = false, succeeded = true, left_a_span = false;
        const auto deadline = Clock::now() + std::chrono::seconds(30);
        while (!refused_once && Clock::now() < deadline) {
            left_a_span = left_a_span || unpacker.tick({}, &succeeded) != 0;
            refused_once = !succeeded;
            if (!refused_once) Sleep(1);
        }
        check(!left_a_span, "a failed program leaves no span");
        check(refused_once && host.said("exit code"), "the tick that finds the failure is refused and the host log has the exit code");
        Row packer(pack, file);
        const uint64_t fresh = tensor_of("still working");
        uint64_t packed = packer.tick({fresh});
        nodus_tensor_destroy(fresh);
        if (!packed) packed = packer.wait_for_span(30000);
        uint64_t back = unpacker.tick({packed});
        if (!back) back = unpacker.wait_for_span(30000);
        check(back != 0 && bytes_of(back) == "still working", "the row that saw the failure handles the next span");
    }

    // The settings are the tool's saved state.
    {
        Row row(unpack, file);
        const std::string saved = scratch + "/exec_row_saved.bin";
        { FileOutputArchive out(saved.c_str()); check(row.tool->serialize(out), "the row saves its settings"); }
        FileInputArchive in(saved.c_str());
        nodus::exec::RowSettings read;
        check(nodus::exec::read_settings(in, read) && read.program == compressor && read.arguments == unpack.arguments &&
              read.working_dir == scratch && read.most_at_once == 2, "and they read back the same");
    }
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
