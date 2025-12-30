#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <string>
#include "value_types.h"

struct ToolInitContext {
    const char* root_dir = nullptr;
    const char* serialized_path = nullptr;
    void* user = nullptr;
};

struct RenderContext {
    void* pixels = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    int32_t pitch = 0;
    float scale = 1.0f;
};

enum class ToolCaps : uint32_t {
    None = 0,
    Table = 1u << 0,
    Led = 1u << 1,
    Text = 1u << 2,
};

inline ToolCaps operator|(ToolCaps a, ToolCaps b) {
    return static_cast<ToolCaps>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline ToolCaps operator&(ToolCaps a, ToolCaps b) {
    return static_cast<ToolCaps>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline ToolCaps& operator|=(ToolCaps& a, ToolCaps b) {
    a = a | b;
    return a;
}

struct TableRenderArgs {
    int32_t rows = 0;
    int32_t cols = 0;
    float cell_w = 0.0f;
    float cell_h = 0.0f;
    void* frame_buffer = nullptr;
    int32_t pitch = 0;
};

struct LEDRenderArgs {
    int32_t connection_handle = -1;
    float region_x = 0.0f;
    float region_y = 0.0f;
    float region_w = 0.0f;
    float region_h = 0.0f;
    const float* palette = nullptr;
    int32_t palette_len = 0;
};

struct TextRenderArgs {
    int32_t font_id = 0;
    float size = 0.0f;
    float x = 0.0f;
    float y = 0.0f;
    const char* text = nullptr;
};

enum class ToolPortKind : uint8_t {
    Argument = 0,
    Return = 1,
    Internal = 2,
};

struct ToolPortSpec {
    ToolPortKind kind = ToolPortKind::Internal;
    int32_t count = 0;
};

struct ToolInputState {
    float mouse_x = 0.0f;
    float mouse_y = 0.0f;
    int32_t mouse_down = 0;
    int32_t mouse_up = 0;
    float mouse_dx = 0.0f;
    float mouse_dy = 0.0f;
    int32_t mouse_button = 0;
    float mouse_scroll = 0.0f;
    uint32_t mouse_button_mask_down = 0u;
    uint32_t mouse_button_mask_up = 0u;
    int32_t mouse_device_id = 0;
    int32_t key = 0;
    int32_t key_event = 0;
};

struct ToolStackFrame {
    float* values = nullptr;
    int32_t count = 0;
    int32_t capacity = 0;
};

// Raw (byte-oriented) stack frame support is provided in `value_types.h`.
// `RawStackFrame` and the `ValueTypeRegistry` are intended for use by
// table/edge code that must carry arbitrary typed payloads. The typed
// raw stack helpers allow single-op memcpy push/pop and are stride-aware.

struct ToolStackContext {
    ToolStackFrame stack{};
    const ToolInputState* input = nullptr;
};

inline float tool_stack_pop(ToolStackFrame& frame) {
    if (!frame.values || frame.count <= 0) return 0.0f;
    float v = frame.values[frame.count - 1];
    frame.count -= 1;
    return v;
}

inline void tool_stack_push(ToolStackFrame& frame, float v) {
    if (!frame.values || frame.count >= frame.capacity) return;
    frame.values[frame.count] = v;
    frame.count += 1;
}

// Pop up to `n` values from the stack into `out` preserving the
// semantics of successive single `tool_stack_pop` calls: i.e. the
// first element written to `out[0]` is the top-most stack element.
// Returns the number of values actually popped.
inline int tool_stack_pop_n(ToolStackFrame& frame, float* out, int n) {
    if (!frame.values || frame.count <= 0 || n <= 0) return 0;
    int avail = frame.count;
    int to = (n < avail) ? n : avail;
    // Write in pop order: out[0] = top, out[to-1] = bottom of popped block.
    for (int i = 0; i < to; ++i) {
        out[i] = frame.values[frame.count - 1 - i];
    }
    frame.count -= to;
    return to;
}

// Pop a contiguous block of up to `n` values from the stack into `out`
// using a single memcpy operation (i.e. preserves the underlying memory
// order from older->newer). This is useful when the consumer expects the
// block in the same order it was pushed. Returns the number of values
// actually popped.
inline int tool_stack_pop_block(ToolStackFrame& frame, float* out, int n) {
    if (!frame.values || frame.count <= 0 || n <= 0) return 0;
    int avail = frame.count;
    int to = (n < avail) ? n : avail;
    float* src = frame.values + (frame.count - to);
    std::memcpy(out, src, static_cast<size_t>(to) * sizeof(float));
    frame.count -= to;
    return to;
}

// Push up to `n` values from `in` onto the stack using a single memcpy
// operation. Returns the number of values actually pushed.
inline int tool_stack_push_n(ToolStackFrame& frame, const float* in, int n) {
    if (!frame.values || n <= 0) return 0;
    int free_space = frame.capacity - frame.count;
    if (free_space <= 0) return 0;
    int to = (n < free_space) ? n : free_space;
    std::memcpy(frame.values + frame.count, in, static_cast<size_t>(to) * sizeof(float));
    frame.count += to;
    return to;
}

// Integer stack helpers (32-bit signed). These use the float stack storage
// but present an explicit int API for built-ins that require integer types.
inline int32_t tool_stack_pop_int(ToolStackFrame& frame) {
    float v = tool_stack_pop(frame);
    return static_cast<int32_t>(std::lround(v));
}

inline void tool_stack_push_int(ToolStackFrame& frame, int32_t v) {
    tool_stack_push(frame, static_cast<float>(v));
}

struct OutputArchive {
    virtual ~OutputArchive() = default;
    virtual bool write(const void* data, size_t len) = 0;
};

struct InputArchive {
    virtual ~InputArchive() = default;
    virtual bool read(void* data, size_t len) = 0;
};

class FileInputArchive : public InputArchive {
public:
    explicit FileInputArchive(const char* path)
        : stream_(path, std::ios::binary) {}

    bool read(void* data, size_t len) override {
        if (!stream_.good()) return false;
        stream_.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(len));
        return stream_.good();
    }

    bool ok() const { return stream_.good(); }

private:
    std::ifstream stream_;
};

class FileOutputArchive : public OutputArchive {
public:
    explicit FileOutputArchive(const char* path)
        : stream_(path, std::ios::binary | std::ios::trunc) {}

    bool write(const void* data, size_t len) override {
        if (!stream_.good()) return false;
        stream_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
        return stream_.good();
    }

    bool ok() const { return stream_.good(); }

private:
    std::ofstream stream_;
};

class HostAPI {
public:
    virtual ~HostAPI() = default;
    virtual void log(const char* message) = 0;
    virtual void draw_text(const TextRenderArgs& args) = 0;
};

struct ITool {
    virtual ~ITool() noexcept = default;

    // Prefer stable C-style string accessors for cross-DLL safety.
    // Default implementations return empty C string and are noexcept.
    virtual const char* id_cstr() const noexcept { return ""; }
    virtual const char* name_cstr() const noexcept { return ""; }

    // Backwards-compatible string-returning accessors; they construct
    // an std::string from the safe C string accessors. These are
    // intentionally non-virtual so implementations in plugins should
    // only provide `id_cstr()`/`name_cstr()` to avoid cross-DLL
    // std::string construction/destruction issues.
    std::string id() const { return std::string(id_cstr()); }
    std::string name() const { return std::string(name_cstr()); }
    virtual ToolCaps caps() const { return ToolCaps::None; }

    virtual void initialize(const ToolInitContext& ctx) = 0;
    virtual void shutdown() = 0;
    virtual void tick(double dt, HostAPI& host) = 0;
    virtual void render(RenderContext& ctx) = 0;

    virtual void render_table(TableRenderArgs& args) { (void)args; }
    virtual void render_led_cell(LEDRenderArgs& args) { (void)args; }
    virtual void render_text(TextRenderArgs& args) { (void)args; }

    virtual int32_t port_count() const { return 0; }
    virtual ToolPortSpec port_spec(int32_t /*idx*/) const { return ToolPortSpec{}; }

    virtual void execute_stack(ToolStackContext& /*ctx*/) {}

    virtual bool serialize(OutputArchive& /*out*/) const { return false; }
    virtual bool deserialize(InputArchive& /*in*/) { return false; }
};

#if defined(_WIN32)
#if defined(NODUS_PLUGIN_COMPOSITE)
#define NODUS_PLUGIN_EXPORT
#else
#define NODUS_PLUGIN_EXPORT __declspec(dllexport)
#endif
#else
#define NODUS_PLUGIN_EXPORT
#endif

#ifndef NODUS_PLUGIN_FACTORY_NAME
#define NODUS_PLUGIN_FACTORY_NAME create_tool
#endif
#ifndef NODUS_PLUGIN_DESTROY_NAME
#define NODUS_PLUGIN_DESTROY_NAME destroy_tool
#endif
#ifndef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_SOURCE_NAME plugin_source_path
#endif

extern "C" {
    typedef ITool* (*CreateToolFn)();
}
