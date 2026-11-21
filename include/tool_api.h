#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
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
    RawStackFrame* raw = nullptr;
    ValueTypeId default_type = kInvalidValueTypeId;
};

// Raw (byte-oriented) stack frame support is provided in `value_types.h`.
// `RawStackFrame` and the `ValueTypeRegistry` are intended for use by
// table/edge code that must carry arbitrary typed payloads. The typed
// raw stack helpers allow single-op memcpy push/pop and are stride-aware.

// ---- A row's own surface (research case 075069677cb74e3f9794914c9009cec5) ------------------------------------------
// Table ports move the stack: N items on the stack with N output ports go out together. A port declared ON a row is
// the row's own channel: the row reads and writes it itself during its tick (ToolStackContext::row_ports). A row also
// declares where its ports sit and which controls it shows; its picture is the ITool paint half (render_table, with
// ToolCaps::Table). The canvas draws and routes exactly what the row declares: rectangles are also the click routes.
struct ToolRowPort {
    ToolPortKind kind = ToolPortKind::Return;  // Argument = into the row, Return = out of the row
    float x = 1.0f, y = 0.5f;                   // the port's LED centre, as fractions of the row's rectangle
    const char* name = "";                      // stable name; a port pair may name it (table.row:name)
    uint32_t token_bytes = 0;                   // bytes per token on its edge (0 = whatever the edge is configured to)
};

// The control kinds are the Pluck knob kinds (button, toggle, stepper, slider, choice) plus text.
enum class ToolRowControlKind : uint8_t { Text = 0, Button = 1, Toggle = 2, Stepper = 3, Slider = 4, Choice = 5 };
struct ToolRowControl {
    ToolRowControlKind kind = ToolRowControlKind::Text;
    int32_t id = 0;                             // handed back in on_row_control
    float x = 0.0f, y = 0.0f, w = 0.0f, h = 0.0f;  // fractions of the row's rectangle
    const char* label = "";
    float value = 0.0f, min = 0.0f, max = 1.0f;    // toggle: 0/1; stepper, slider: in [min, max]; choice: index
    int32_t choices = 0;                        // choice: how many
};
enum class ToolRowControlEventKind : uint8_t { Press = 1, Set = 2, Step = 3 };
struct ToolRowControlEvent {
    ToolRowControlEventKind kind = ToolRowControlEventKind::Press;
    float value = 0.0f;                         // Set: the new value; Step: +1 or -1
};

// The row's authority over its own ports during execute_stack. Port indices are the row's declared ports.
struct ToolRowPortAccess {
    void* host = nullptr;
    // One token onto the edge at this (output) port: 1 accepted; 0 no edge there, the edge is full, or wrong size.
    int32_t (*publish)(void* host, int32_t port, const void* bytes, size_t len) = nullptr;
    // One token from the edge at this (input) port: 1 read into out (*written bytes); 0 nothing waiting or no edge.
    int32_t (*consume)(void* host, int32_t port, void* out, size_t capacity, size_t* written) = nullptr;
    // 1 when the edge at this (output) port can take `count` more tokens without dropping; 0 otherwise or no edge.
    // Trailing: hosts that do not fill it leave it null.
    int32_t (*can_publish)(void* host, int32_t port, int32_t count) = nullptr;
};

enum class ToolTensorRole : uint8_t { Owner = 0, Borrower = 1, Returns = 2 };

// The table's used-up AbstractTensor handles, oldest first, offered to a Returns row (ITool::tensor_role). Every use
// of a duplicated handle (tensor_handle_dup) is a stack entry, so a handle appears here only once no entry names it.
// The row sets `taken` to how many it handed on, from the front; the host keeps the rest for the next tick.
struct ToolReturnedTensors {
    const uint64_t* ids = nullptr;
    size_t count = 0;
    size_t taken = 0;
};

struct ToolStackContext {
    ToolStackFrame stack{};
    const ToolInputState* input = nullptr;
    // Tools that restore their inputs after refusing an execution must clear
    // this flag.  The scheduler then leaves the restored stack available for
    // inspection without mistaking its tail for declared return values.
    // Kept as a trailing, default-true field so existing tools retain their
    // successful void execute_stack contract until they opt into refusal.
    bool succeeded = true;
    // The row's own ports (row_port_count() > 0), bound by the host for this execution; null otherwise. Trailing, as
    // `succeeded` is, so rows that declare no ports are unaffected.
    const ToolRowPortAccess* row_ports = nullptr;
    // For a Returns row (ITool::tensor_role): the table's used-up handles; null for every other row. Trailing.
    ToolReturnedTensors* returns = nullptr;
};

inline float tool_stack_pop(ToolStackFrame& frame) {
    if (!frame.raw || frame.default_type == kInvalidValueTypeId) return 0.0f;
    double tmp = 0.0;
    if (!raw_stack_pop_typed(*frame.raw, &tmp, frame.default_type)) return 0.0f;
    return static_cast<float>(tmp);
}

inline void tool_stack_push(ToolStackFrame& frame, float v) {
    if (!frame.raw || frame.default_type == kInvalidValueTypeId) return;
    double tmp = static_cast<double>(v);
    raw_stack_push_typed(*frame.raw, &tmp, frame.default_type);
}

// Pop up to `n` values from the stack into `out` preserving the
// semantics of successive single `tool_stack_pop` calls: i.e. the
// first element written to `out[0]` is the top-most stack element.
// Returns the number of values actually popped.
inline int tool_stack_pop_n(ToolStackFrame& frame, float* out, int n) {
    if (!frame.raw || frame.default_type == kInvalidValueTypeId || !out || n <= 0) return 0;
    int popped = 0;
    for (; popped < n; ++popped) {
        ValueTypeId top_tid = kInvalidValueTypeId;
        if (!raw_stack_peek_type(*frame.raw, top_tid) || top_tid != frame.default_type) break;
        out[popped] = tool_stack_pop(frame);
    }
    return popped;
}

// Pop a contiguous block of up to `n` values from the stack into `out`
// using a single memcpy operation (i.e. preserves the underlying memory
// order from older->newer). This is useful when the consumer expects the
// block in the same order it was pushed. Returns the number of values
// actually popped.
inline int tool_stack_pop_block(ToolStackFrame& frame, float* out, int n) {
    if (!frame.raw || frame.default_type == kInvalidValueTypeId || !out || n <= 0) return 0;
    const ValueType* vt = ValueTypeRegistry::global().get(frame.default_type);
    if (!vt) return 0;
    size_t per = vt->size;
    if (per == 0) return 0;
    std::vector<uint8_t> temp(static_cast<size_t>(n) * per);
    int popped = raw_stack_pop_block(*frame.raw, temp.data(), n, frame.default_type);
    for (int i = 0; i < popped; ++i) {
        float value = 0.0f;
        const uint8_t* ptr = temp.data() + static_cast<size_t>(i) * per;
        if (per == sizeof(float)) {
            float fv = 0.0f;
            std::memcpy(&fv, ptr, sizeof(float));
            value = fv;
        } else if (per == sizeof(double)) {
            double dv = 0.0;
            std::memcpy(&dv, ptr, sizeof(double));
            value = static_cast<float>(dv);
        }
        out[i] = value;
    }
    return popped;
}

// Push up to `n` values from `in` onto the stack using a single memcpy
// operation. Returns the number of values actually pushed.
inline int tool_stack_push_n(ToolStackFrame& frame, const float* in, int n) {
    if (!frame.raw || !in || n <= 0 || frame.default_type == kInvalidValueTypeId) return 0;
    int pushed = 0;
    for (int i = 0; i < n; ++i) {
        double tmp = static_cast<double>(in[i]);
        if (!raw_stack_push_typed(*frame.raw, &tmp, frame.default_type)) break;
        pushed += 1;
    }
    return pushed;
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

// A row's saved state held in memory: the canvas saves every row's serialize() output in its file (ROW_STATE records)
// and hands it back to deserialize() on load, so a row's settings travel with the canvas that holds it.
class MemoryOutputArchive : public OutputArchive {
public:
    bool write(const void* data, size_t len) override {
        const auto* b = static_cast<const uint8_t*>(data);
        bytes.insert(bytes.end(), b, b + len);
        return true;
    }
    std::vector<uint8_t> bytes;
};

class MemoryInputArchive : public InputArchive {
public:
    explicit MemoryInputArchive(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}
    bool read(void* data, size_t len) override {
        if (len > bytes_.size() - at_) return false;
        std::memcpy(data, bytes_.data() + at_, len);
        at_ += len;
        return true;
    }
    size_t remaining() const { return bytes_.size() - at_; }
private:
    const std::vector<uint8_t>& bytes_;
    size_t at_ = 0;
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

    // A row's own surface (see ToolRowPort above). Appended after the existing virtuals so their order is unchanged.
    // The row's picture is render_table (ToolCaps::Table): the host calls it with rows = cols = 1, cell_w/cell_h = the
    // row's pixel size and frame_buffer/pitch at the row's rectangle, then draws the declared ports and controls on it.
    virtual int32_t row_height_px() const { return 0; }                 // 0 = the host's tool row height
    virtual int32_t row_port_count() const { return 0; }
    virtual ToolRowPort row_port(int32_t /*idx*/) const { return ToolRowPort{}; }
    virtual int32_t row_control_count() const { return 0; }
    virtual ToolRowControl row_control(int32_t /*idx*/) const { return ToolRowControl{}; }
    virtual void on_row_control(int32_t /*id*/, const ToolRowControlEvent& /*event*/) {}

    // What a row does with the AbstractTensor handles it pops (research case 6393486c, the mint table). Appended after
    // the row-surface virtuals so their order is unchanged.
    //   Owner    : the row may keep what it pops (state libraries, negotiators); the host never returns it. Default.
    //   Borrower : the row only reads what it pops and never keeps or destroys it (compiled rows, tensor_handle_dup).
    //              A handle a Borrower row pops that no stack entry still names is used up: the host records it.
    //   Returns  : the row hands the table's used-up handles on (ToolStackContext::returns) to the mint table.
    virtual ToolTensorRole tensor_role() const { return ToolTensorRole::Owner; }
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
// Optional export: called once at DLL load time, before any create_tool call,
// so a plugin can register new ValueType structs (ValueTypeRegistry::global()
// .register_struct(...)) that its tools need the value stack/edges to carry.
// Absent in most plugins -- PluginLoader treats a missing export as a no-op.
//
// Scope note: ValueTypeRegistry::global() is a Meyer's singleton defined
// inline in value_types.h -- it is NOT shared across a real DLL boundary.
// A plugin built as its own standalone .dll and loaded via LoadLibrary (the
// repo-ingestion tier's shape, see repo_package.h) gets its own independent
// copy of the registry; register_struct() calls there are invisible to the
// host. This export only reaches the host's registry when the plugin source
// is compiled into the *same* binary as the host (the composite/inlined
// graph-collapse path in module_library_actualizer.cpp, or a tool linked
// directly into canvas_tables_static). For genuinely cross-DLL plugins,
// prefer keeping shared data opaque (VT_VOID_PTR, interpreted only by tools
// that agree on its shape at compile time) or tensor-shaped
// (VT_ABSTRACT_TENSOR) rather than registering a new central struct type --
// nodus staying the sole owner/mutator of its own registry is deliberate,
// not an oversight. A real cross-DLL registration path (plugin describes
// fields via a plain POD struct, host commits them into its own registry)
// is not designed yet; build it only once something concrete needs a
// struct's fields visible outside the package that defined it.
#ifndef NODUS_PLUGIN_REGISTER_TYPES_NAME
#define NODUS_PLUGIN_REGISTER_TYPES_NAME register_types
#endif

extern "C" {
    typedef ITool* (*CreateToolFn)();
    typedef void (*RegisterTypesFn)();
}

// Capability tags describing what a repo-ingested tool actually is, honestly,
// rather than gatekeeping on IR-translatability. A tool can be none of these
// (fully opaque, maximally caveated) or several. Numeric on purpose -- no
// general string-interning utility exists in nodus yet (see repo_package.h);
// this stays a plain closed enum until one does, without changing the shape
// of anything that carries these tags.
enum ToolCapabilityTag : uint32_t {
    CAP_NONE = 0,
    // Decomposes into nodus::spirv::KernelIR (src/kernel_isa.h) -- translatable
    // to any registered backend (SPIR-V, GLSL, CPU, ...) via TranslationMatrix.
    CAP_ISA = 1u << 0,
    // Wraps already-compiled opaque native code (a DLL/lib) with no source-level
    // decomposition offered -- CPU-only, not portable to other backends.
    CAP_BINARY = 1u << 1,
    // Supplies a gradient/backward path even if the forward pass is opaque, so
    // it can still participate in an autodiff graph.
    CAP_BACKWARD = 1u << 2,
};
