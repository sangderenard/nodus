// file_write: a row whose one duty is a file destination (see file_read.cpp for the owner's words on file rows).
//
// Its setting (saved state, the canvas file's ROW_STATE) is a destination folder. Each execution it takes the tensor
// waiting on top of the stack -- an Input row's pads (a scalar, or a null handle) are dropped -- and writes its bytes
// to a new file there: first under a hidden staging name, then renamed to <number>.arrival, the number one past the
// largest already in the folder. Nothing is ever overwritten: both names are created exclusively. A tensor it cannot
// write is restored to the stack with succeeded=false and offered again. A written tensor is the row's to destroy.
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
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

class FileWrite final : public ITool {
public:
    const char* id_cstr() const noexcept override { return "nodus.file.write"; }
    const char* name_cstr() const noexcept override { return "File write"; }
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
        if (index == 0) return {ToolPortKind::Argument, 1};
        return {};
    }

    // Saved state: magic "NFW1", version 1, folder length (u32), folder (UTF-8).
    bool serialize(OutputArchive& out) const override {
        if (folder_.empty()) return false;
        const std::string f = utf8_of(folder_);
        const uint32_t head[3] = {kMagic, kVersion, static_cast<uint32_t>(f.size())};
        return out.write(head, sizeof(head)) && out.write(f.data(), f.size());
    }
    bool deserialize(InputArchive& in) override {
        uint32_t head[3] = {};
        if (!in.read(head, sizeof(head)) || head[0] != kMagic || head[1] != kVersion || !head[2] || head[2] > 4096) return false;
        std::string f(head[2], '\0');
        if (!in.read(f.data(), f.size())) return false;
        folder_ = path_of_utf8(f);
        next_ = 0;                                                     // found again on the first write
        return true;
    }

    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (folder_.empty() || !context.stack.raw) return;
        RawStackFrame& stack = *context.stack.raw;
        const ValueTypeId tensor_type = ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR);
        nodus::tensors::AbstractTensorHandle input{};
        bool found = false;
        ValueTypeId top = kInvalidValueTypeId;
        while (raw_stack_peek_type(stack, top)) {
            if (top == tensor_type) {
                if (!raw_stack_pop_abstract_tensor(stack, input)) break;
                if (input.id) { found = true; break; }
                continue;
            }
            const ValueType* pad = ValueTypeRegistry::global().get(top);
            if (!pad || !pad->size) break;
            std::vector<uint8_t> discard(pad->size);
            if (!raw_stack_pop_typed(stack, discard.data(), top)) break;
        }
        context.succeeded = true;
        if (!found) return;
        if (!write(input.id)) {
            raw_stack_push_abstract_tensor(stack, input);              // refused: it stays on the stack
            context.succeeded = false;
            return;
        }
        nodus_tensor_destroy(input.id);
    }

private:
    static constexpr uint32_t kMagic = 0x3157464E, kVersion = 1;

    // One past the largest <number>.arrival already in the folder.
    uint64_t first_free() const {
        uint64_t next = 1;
        std::error_code ec;
        for (fs::directory_iterator it(folder_, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = utf8_of(it->path().filename());
            const auto dot = name.find(".arrival");
            if (dot == std::string::npos || dot == 0 || dot + 8 != name.size()) continue;
            uint64_t n = 0;
            bool digits = true;
            for (size_t i = 0; i < dot && digits; ++i) {
                digits = name[i] >= '0' && name[i] <= '9';
                n = n * 10 + uint64_t(name[i] - '0');
            }
            if (digits && n + 1 > next) next = n + 1;
        }
        return next;
    }
    static bool create_exclusive(const fs::path& p, const std::vector<uint8_t>& bytes) {
        std::FILE* f = std::fopen(p.string().c_str(), "wbx");
        if (!f) return false;
        const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
        return std::fclose(f) == 0 && ok;
    }
    bool write(uint64_t t) {
        NodusTensorDesc d{};
        if (nodus_tensor_describe(t, &d) != NODUS_OK || !d.total_bytes) return false;
        std::vector<uint8_t> bytes(static_cast<size_t>(d.total_bytes));
        if (nodus_tensor_read(t, 0, bytes.data(), bytes.size()) != static_cast<int64_t>(bytes.size())) return false;
        std::error_code ec;
        fs::create_directories(folder_, ec);
        if (!next_) next_ = first_free();
        for (int attempt = 0; attempt < 64; ++attempt, ++next_) {
            char name[48];
            std::snprintf(name, sizeof(name), "%08llu.arrival", static_cast<unsigned long long>(next_));
            const fs::path final_path = folder_ / name;
            const fs::path staging = folder_ / (std::string(".") + name + ".partial");
            if (fs::exists(final_path, ec) || fs::exists(staging, ec)) continue;
            if (!create_exclusive(staging, bytes)) { fs::remove(staging, ec); return false; }
            if (fs::exists(final_path, ec)) { fs::remove(staging, ec); continue; }
            fs::rename(staging, final_path, ec);
            if (ec) { fs::remove(staging, ec); return false; }
            ++next_;
            return true;
        }
        return false;
    }

    fs::path folder_;
    uint64_t next_ = 0;
};

}  // namespace nodus::file_rows

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new nodus::file_rows::FileWrite(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* tool) { delete tool; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
