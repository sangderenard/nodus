#pragma once

#include <cstdint>
#include <cstddef>
#include <fstream>
#include <string>

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
    int32_t key = 0;
    int32_t key_event = 0;
};

struct ToolStackFrame {
    float* values = nullptr;
    int32_t count = 0;
    int32_t capacity = 0;
};

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

    virtual std::string id() const = 0;
    virtual std::string name() const = 0;
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

extern "C" {
    typedef ITool* (*CreateToolFn)();
}
