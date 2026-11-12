// exec_row_registry_test declare <file> <compressor> <scratch-dir> [core-root]
//     writes a declarations file (two good rows, two the rules refuse).
// exec_row_registry_test check <file> <compressor> <scratch-dir> [core-root]
//     run with NODUS_EXEC_ROWS=<file>, NODUS_EXEC_CORE_ROOTS=<core-root> and
//     NODUS_EXEC_WORK_ROOTS=<core-root>/srv/state/content/users in the environment, so the rows were
//     registered when this image loaded; finds them in the real tool registry, makes them the way a canvas does
//     (by id, with an empty init context), and passes a span through both.
#include "exec_row.h"
#include "exec_row_registry.h"

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "value_types.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

using Handle = nodus::tensors::AbstractTensorHandle;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

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

// One span through one row: ticks as the thread manager ticks, until the row leaves a span.
static uint64_t through(ITool& tool, uint64_t span) {
    RawStackFrame frame{};
    raw_stack_init_frame(frame, 4096);
    uint64_t out = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (bool first = true; !out && std::chrono::steady_clock::now() < deadline; first = false) {
        raw_stack_reset(frame);
        if (first) raw_stack_push_abstract_tensor(frame, Handle{span});
        ToolStackContext context{};
        context.stack.raw = &frame;
        context.stack.default_type = ValueTypeRegistry::global().builtin(VT_FLOAT64);
        tool.execute_stack(context);
        Handle top{};
        if (raw_stack_pop_abstract_tensor(frame, top) && top.id != span) out = top.id;
        if (!out) Sleep(1);
    }
    raw_stack_free_mask(frame);
    return out;
}

static void write_file(const std::string& path, const std::string& text) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: exec_row_registry_test declare|check <file> <compressor> <scratch-dir> [core-root]\n"); return 2; }
    const std::string mode = argv[1], file = argv[2], compressor = argv[3], scratch = argv[4];
    const std::string core = argc > 5 ? argv[5] : "C:/dev/Powershell";
    if (mode == "declare") {
        write_file(file,
            "# the byte-span compressor in waiting, and its other half\n"
            "row exec.zstd.pack\n  name Zstd pack\n  program " + compressor + "\n  argument -c\n  directory " + scratch + "\n\n"
            "row exec.zstd.unpack\n  name Zstd unpack\n  program " + compressor + "\n  argument -d\n  argument -c\n"
            "  directory " + scratch + "\n  most 1\n\n"
            "row exec.git\n  program C:/Program Files/Git/cmd/git.exe\n  argument --version\n  directory " + scratch + "\n\n"
            "row exec.inside\n  program " + compressor + "\n  argument -c\n  directory " + core + "/turing\n\n"
            "row exec.userfolder\n  program " + compressor + "\n  argument -c\n  directory " + core +
            "/srv/state/content/users/account/2/exec\n");
        return 0;
    }

    // What the image registered when it loaded.
    ToolRegistry& registry = tool_registry_global();
    const ToolRegistry::Entry* pack = registry.find("exec.zstd.pack");
    const ToolRegistry::Entry* unpack = registry.find("exec.zstd.unpack");
    check(pack && pack->name == "Zstd pack" && unpack && unpack->name == "Zstd unpack", "the declared rows are in the tool registry under their ids");
    check(pack && !pack->source_path.empty(), "with the row's source recorded");
    check(!registry.find("exec.git"), "a declared row that would run Git is not registered");
    check(!registry.find("exec.inside"), "nor one whose working directory is inside the core");
    check(registry.find("exec.userfolder") != nullptr,
          "but one in a declared user folder is, though that folder is inside the core");
    check(!registry.find("exec_row") && !registry.find("exec"), "and there is no exec row that was not declared");

    // Made the way a canvas makes a plugin row: by id, with an init context that carries no settings.
    if (pack && unpack) {
        auto packer = registry.create("exec.zstd.pack");
        auto unpacker = registry.create("exec.zstd.unpack");
        ToolInitContext init{};
        packer->initialize(init);
        unpacker->initialize(init);
        check(packer->port_count() == 2, "the row made from the registry declares its ports");
        std::string text;
        for (int i = 0; i < 200000; ++i) text += "triangle_map[" + std::to_string(i % 4096) + "] ";
        const uint64_t span = tensor_of(text);
        const auto started = std::chrono::steady_clock::now();
        const uint64_t packed = through(*packer, span);
        const size_t packed_size = bytes_of(packed).size();
        const uint64_t back = packed ? through(*unpacker, packed) : 0;
        const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        check(packed && packed_size && packed_size < text.size() / 4, "a span goes through the registered pack row");
        check(back && bytes_of(back) == text, "and back through the registered unpack row, the same bytes");
        std::printf("     %zu bytes -> %zu -> back in %lld ms\n", text.size(), packed_size, static_cast<long long>(took));
        packer->shutdown();
        unpacker->shutdown();
    }

    // Declarations that cannot be read register nothing and name the line.
    ToolRegistry local;
    std::string report;
    const std::string bad = scratch + "/exec_rows_bad.txt";
    write_file(bad, "row exec.one\n  program " + compressor + "\n  colour blue\n");
    check(nodus::exec::register_declared_rows(bad, local, report) == 0 && report.find("line 3") != std::string::npos,
          "a line that is not a declaration stops the file, by number");
    report.clear();
    write_file(bad, "row zstd\n  program " + compressor + "\n");
    check(nodus::exec::register_declared_rows(bad, local, report) == 0 && report.find("under 'exec.'") != std::string::npos,
          "an id outside 'exec.' is refused");
    report.clear();
    write_file(bad, "row exec.one\n  program " + compressor + "\n  directory " + scratch + "\nrow exec.one\n");
    check(nodus::exec::register_declared_rows(bad, local, report) == 0 && report.find("twice") != std::string::npos,
          "so is a row declared twice");
    report.clear();
    check(nodus::exec::register_declared_rows(scratch + "/no-such-declarations.txt", local, report) == 0 && !report.empty(),
          "and a file that is not there");

    // The same declarations with no declared core: nothing is registered.
    SetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", nullptr);
    report.clear();
    check(nodus::exec::register_declared_rows(file, local, report) == 0 && report.find("declared no core") != std::string::npos,
          "with no declared core no row is registered");
    SetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", core.c_str());
    report.clear();
    check(nodus::exec::register_declared_rows(file, local, report) == 3 && local.find("exec.zstd.pack") && !local.find("exec.git"),
          "with the core declared the three good rows are, into any registry");
    check(nodus::exec::register_declared_rows(file, local, report) == 0 && report.find("already taken") != std::string::npos,
          "and a second registration of the same ids is refused");
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
