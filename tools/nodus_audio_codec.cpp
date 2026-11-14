// One bounded audio span through a host-declared Nodus exec row. The host owns
// the FFmpeg declaration and both file paths; this program never invokes an
// undeclared executable or interprets a client-provided command line.
#include "exec_row.h"
#include "exec_row_registry.h"
#include "tool_registry.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "value_types.h"

#include <chrono>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr std::uint64_t kMaxSpan = 256ull * 1024ull * 1024ull;

void write_u32(std::vector<std::uint8_t>& bytes, std::size_t at, std::uint32_t value) {
    for (int i = 0; i != 4; ++i) bytes[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

std::uint32_t read_u32(const std::vector<std::uint8_t>& bytes, std::size_t at) {
    std::uint32_t value = 0;
    for (int i = 0; i != 4; ++i) value |= std::uint32_t(bytes[at + i]) << (8 * i);
    return value;
}

// FFmpeg writes 0xffffffff RIFF/data lengths when stdout is a pipe. Once the
// exec row has returned the complete span, replace those placeholders with
// the actual lengths so the file-source loader sees one finite WAV.
bool finish_streamed_wav(std::vector<std::uint8_t>& bytes) {
    if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) ||
        std::memcmp(bytes.data() + 8, "WAVE", 4)) return false;
    write_u32(bytes, 4, static_cast<std::uint32_t>(bytes.size() - 8));
    for (std::size_t at = 12; at + 8 <= bytes.size();) {
        const auto size = read_u32(bytes, at + 4);
        if (!std::memcmp(bytes.data() + at, "data", 4)) {
            if (size != 0xffffffffu && at + 8 + size != bytes.size()) return false;
            write_u32(bytes, at + 4, static_cast<std::uint32_t>(bytes.size() - at - 8));
            return true;
        }
        if (size > bytes.size() - at - 8) return false;
        at += 8 + size + (size & 1u);
    }
    return false;
}

bool inside_work_root(const std::filesystem::path& path) {
    const auto resolved = std::filesystem::weakly_canonical(path);
    for (const auto& name : nodus::exec::declared_work_roots()) {
        const auto root = std::filesystem::weakly_canonical(std::filesystem::u8path(name));
        auto a = root.begin(), b = resolved.begin();
        while (a != root.end() && b != resolved.end() && *a == *b) { ++a; ++b; }
        if (a == root.end() && b != resolved.end()) return true;
    }
    return false;
}

std::uint64_t step(ITool& tool, std::uint64_t input, bool& okay) {
    RawStackFrame frame{};
    if (!raw_stack_init_frame(frame, 8)) { okay = false; return 0; }
    if (input) raw_stack_push_abstract_tensor(frame, {input});
    ToolStackContext context{};
    context.stack.raw = &frame;
    tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle output{};
    raw_stack_pop_abstract_tensor(frame, output);
    okay = context.succeeded;
    raw_stack_free_mask(frame);
    if (frame.backend) gp_mem_backend_release(frame.backend);
    return output.id;
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::fputs("usage: nodus_audio_codec <rows-file> <row-id> <input> <output>\n", stderr);
        return 2;
    }
    try {
        const auto input_path = std::filesystem::u8path(argv[3]);
        const auto output_path = std::filesystem::u8path(argv[4]);
        if (!input_path.is_absolute() || !output_path.is_absolute() ||
            !inside_work_root(input_path) || !inside_work_root(output_path) ||
            !std::filesystem::is_regular_file(input_path) ||
            std::filesystem::exists(output_path)) return 3;
        const auto length = std::filesystem::file_size(input_path);
        if (!length || length > kMaxSpan) return 4;
        std::ifstream stream(input_path, std::ios::binary);
        std::vector<std::uint8_t> bytes(std::istreambuf_iterator<char>{stream}, {});
        if (bytes.size() != length) return 4;

        ToolRegistry registry;
        std::string report;
        if (!nodus::exec::register_declared_rows(argv[1], registry, report)) {
            std::fputs(report.c_str(), stderr); return 5;
        }
        auto tool = registry.create(argv[2]);
        if (!tool) return 5;
        const std::uint64_t shape[1] = {length};
        const auto tensor = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
        if (!tensor || nodus_tensor_write(tensor, 0, bytes.data(), bytes.size()) !=
                           static_cast<std::int64_t>(bytes.size())) return 6;
        bytes.clear();
        bool okay = true;
        auto result = step(*tool, tensor, okay);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
        while (okay && !result && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            result = step(*tool, 0, okay);
        }
        nodus_tensor_destroy(tensor);
        if (!okay || !result) { tool->shutdown(); return 7; }
        NodusTensorDesc desc{};
        if (nodus_tensor_describe(result, &desc) != NODUS_OK ||
            desc.dtype != NODUS_DTYPE_U8 || desc.rank != 1 ||
            !desc.total_bytes || desc.total_bytes > kMaxSpan) {
            nodus_tensor_destroy(result); tool->shutdown(); return 8;
        }
        bytes.resize(static_cast<std::size_t>(desc.total_bytes));
        if (nodus_tensor_read(result, 0, bytes.data(), bytes.size()) !=
            static_cast<std::int64_t>(bytes.size())) {
            nodus_tensor_destroy(result); tool->shutdown(); return 8;
        }
        nodus_tensor_destroy(result);
        tool->shutdown();
        const std::string row_id = argv[2];
        if (row_id.size() >= 7 && row_id.ends_with("_decode") && !finish_streamed_wav(bytes)) return 8;
        std::ofstream out(output_path, std::ios::binary | std::ios::out | std::ios::trunc);
        if (!out) return 9;
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        out.flush();
        if (!out) { out.close(); std::filesystem::remove(output_path); return 9; }
        std::printf("codec_bytes=%llu\n", static_cast<unsigned long long>(bytes.size()));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "audio codec refused: %s\n", error.what());
        return 10;
    }
}
