// file_read: a row whose one duty is a file source. Owner, 2026-11-21: "reading and writing files belongs to reading
// and writing file rows"; "single event file reads are fine for loading actual row content but not for runtime unless
// it's an intentional file cache or process destination or source"; "rows should not be doing double duty".
//
// Its setting (saved state, the canvas file's ROW_STATE) is a source folder. Each execution, when the stack has room,
// it takes the next file there by name, moves it into <folder>/sent first (so it is never offered twice; a name sent
// before is numbered, never overwritten), reads its
// bytes and pushes them onto the stack as one U8 [n] tensor. A file it cannot read, or a push the stack refuses, is
// moved back and offered again on the next execution. One file is on the stack at a time; the row belongs in a
// stack-loopback table, so a tensor its Output row could not move yet is kept rather than reset. Hidden files (a leading '.') and folders are left alone, so a
// writer can stage a file under a hidden name and rename it into place when it is whole.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace nodus::file_rows {

namespace fs = std::filesystem;

// UTF-8 text of a path and back. C++20 paths speak char8_t; the runtime plugin builder compiles rows as C++17.
#if defined(__cpp_char8_t)
inline std::string utf8_of(const fs::path& p) { const auto u = p.u8string(); return std::string(u.begin(), u.end()); }
inline fs::path path_of_utf8(const std::string& s) { return fs::path(std::u8string(s.begin(), s.end())); }
#else
inline std::string utf8_of(const fs::path& p) { return p.u8string(); }
inline fs::path path_of_utf8(const std::string& s) { return fs::u8path(s); }
#endif

class FileRead final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "nodus.file.read"; }
    const char* name_cstr() const noexcept override { return "File read"; }
    void initialize(const ToolInitContext& ctx) override {
        if (ctx.serialized_path && ctx.serialized_path[0] != '\0') {
            FileInputArchive in(ctx.serialized_path);
            if (in.ok()) deserialize(in);
        }
    }
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    int32_t port_count() const override { return 1; }
    ToolPortSpec port_spec(int32_t index) const override {
        if (index == 0) return {ToolPortKind::Return, 1};
        return {};
    }

    // Saved state: magic "NFR1", version 1, folder length (u32), folder (UTF-8), largest file (u64 bytes).
    bool serialize(OutputArchive& out) const override {
        if (folder_.empty()) return false;
        const std::string f = utf8_of(folder_);
        const uint32_t head[3] = {kMagic, kVersion, static_cast<uint32_t>(f.size())};
        return out.write(head, sizeof(head)) && out.write(f.data(), f.size()) && out.write(&largest_, sizeof(largest_));
    }
    bool deserialize(InputArchive& in) override {
        uint32_t head[3] = {};
        if (!in.read(head, sizeof(head)) || head[0] != kMagic || head[1] != kVersion || !head[2] || head[2] > 4096) return false;
        std::string f(head[2], '\0');
        uint64_t largest = 0;
        if (!in.read(f.data(), f.size()) || !in.read(&largest, sizeof(largest)) || !largest) return false;
        folder_ = path_of_utf8(f);
        largest_ = largest;
        return true;
    }

    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (folder_.empty() || !context.stack.raw) return;
        RawStackFrame& stack = *context.stack.raw;
        const ValueType* vt = ValueTypeRegistry::global().get(ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR));
        const size_t entry = vt ? vt->size : sizeof(nodus::tensors::AbstractTensorHandle);
        context.succeeded = true;
        if (stack.byte_capacity < stack.byte_count + entry) return;      // no room: the file waits
        // One file at a time: while the last one still waits on the stack (its Output edge full), the next waits in the
        // folder. Put this row in a stack-loopback table, so a tensor that has not gone out yet is kept, not reset.
        const ValueTypeId tensor_type = ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
        ValueTypeId top = kInvalidValueTypeId;
        nodus::tensors::AbstractTensorHandle waiting{};
        if (raw_stack_peek_type(stack, top) && top == tensor_type && raw_stack_peek_abstract_tensor(stack, waiting) && waiting.id)
            return;
        const fs::path next = next_file();
        if (next.empty()) return;
        std::error_code ec;
        const fs::path sent_dir = folder_ / "sent";
        fs::create_directories(sent_dir, ec);
        // A name sent before (each generation's records reuse 000000.rec, 000001.rec ...) gets the next free number:
        // nothing in sent is ever overwritten.
        fs::path taken = sent_dir / next.filename();
        for (uint64_t k = 1; fs::exists(taken, ec); ++k) {
            taken = sent_dir / path_of_utf8(utf8_of(next.filename()) + "." + std::to_string(k));
        }
        fs::rename(next, taken, ec);
        if (ec) return;
        const uint64_t t = tensor_of(taken);
        if (!t || !raw_stack_push_abstract_tensor(stack, nodus::tensors::AbstractTensorHandle{t})) {
            if (t) nodus_tensor_destroy(t);
            fs::rename(taken, next, ec);                                 // offered again next execution
            context.succeeded = false;
        }
    }

private:
    static constexpr uint32_t kMagic = 0x3152464E, kVersion = 1;

    fs::path next_file() const {
        std::error_code ec;
        std::vector<fs::path> names;
        for (fs::directory_iterator it(folder_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = utf8_of(it->path().filename());
            if (name.empty() || name[0] == '.' || !it->is_regular_file(ec)) continue;
            names.push_back(it->path());
        }
        if (names.empty()) return {};
        return *std::min_element(names.begin(), names.end());
    }
    uint64_t tensor_of(const fs::path& p) const {
        std::error_code ec;
        const uint64_t size = fs::file_size(p, ec);
        if (ec || !size || size > largest_) return 0;
        std::vector<uint8_t> bytes(static_cast<size_t>(size));
        {
            std::ifstream in(p, std::ios::binary);
            if (!in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) return 0;
        }
        const uint64_t shape[1] = {size};
        const uint64_t t = nodus_tensor_create(NODUS_DTYPE_U8, 1, shape);
        if (!t) return 0;
        if (nodus_tensor_write(t, 0, bytes.data(), bytes.size()) != static_cast<int64_t>(bytes.size())) {
            nodus_tensor_destroy(t);
            return 0;
        }
        return t;
    }

    fs::path folder_;
    uint64_t largest_ = uint64_t(256) << 20;
};

}  // namespace nodus::file_rows

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::file_rows::FileRead(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
