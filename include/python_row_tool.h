#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "value_types.h"

#if defined(_WIN32)
#  if defined(NODUS_PYTHON_ROW_BUILD)
#    define NODUS_PYTHON_ROW_API __declspec(dllexport)
#  else
#    define NODUS_PYTHON_ROW_API __declspec(dllimport)
#  endif
#else
#  define NODUS_PYTHON_ROW_API
#endif

namespace nodus::python_row {

// Version 2 adds the Bytes port (below); version 1 descriptors are still read and run unchanged.
inline constexpr uint32_t kPythonRowAbiVersion = 2;
inline constexpr uint32_t kPythonRowAbiVersionMin = 1;

// Interpreted and compiled rows deliberately have different execution modes.
// This implementation accepts Interpreted only; a compiler adapter must create
// a compiled plugin artifact explicitly and must not use this class as fallback.
enum class ExecutionMode : uint32_t {
    Interpreted = 0,
    CompiledSsa = 1,
};

// Version 1 admits only self-contained primitive values. Pointer and tensor
// carriers are intentionally absent: their ownership cannot cross CPython's
// object boundary through a numeric conversion without a separate contract.
// Version 2 is that contract for bytes (Bytes): an input is one U8 tensor of any shape, read whole into a Python
// `bytes`; the tensor stays the caller's (it is never kept or destroyed here). An output is a `bytes` (or bytearray /
// memoryview) the callable returns, copied into a new U8 [n] tensor that belongs to whoever pops it; an empty result
// pushes nothing for that port. No pointer crosses: only copies of the bytes. Bytes ports run only through
// execute_tensor_stack (the row stack carries tensors); the primitive execute() refuses them.
enum class PrimitiveType : uint32_t {
    Float32 = 0,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bytes,      // ABI v2: one U8 tensor <-> Python bytes (see above)
};

struct Port {
    std::string name;
    PrimitiveType type = PrimitiveType::Float64;
    // Optional presentation metadata.  Empty values mean that a future UI
    // derives the label and control from name/type.  These fields do not
    // participate in execution or change the row stack ABI.
    std::string label;
    std::string form_control;
    bool form_required = true;
};

struct Descriptor {
    uint32_t abi_version = kPythonRowAbiVersion;
    ExecutionMode mode = ExecutionMode::Interpreted;
    std::string tool_id;
    std::string tool_name;
    std::string source_path;
    std::string module_name;
    std::string callable_name;
    std::vector<Port> inputs;
    std::vector<Port> outputs;
};

NODUS_PYTHON_ROW_API ValueTypeBuiltin builtin_type(PrimitiveType type);
NODUS_PYTHON_ROW_API const char* primitive_type_name(PrimitiveType type);
NODUS_PYTHON_ROW_API bool validate_descriptor(const Descriptor& descriptor,
                                              std::string* error = nullptr);
NODUS_PYTHON_ROW_API bool write_descriptor(const Descriptor& descriptor,
                                           const std::string& output_path,
                                           std::string* error = nullptr);
NODUS_PYTHON_ROW_API bool read_descriptor(const std::string& input_path,
                                          Descriptor* descriptor,
                                          std::string* error = nullptr);

// Emits one ordinary ITool plugin source. The generated plugin calls the
// interpreter support below, exports plugin_source_path() for the retained
// .py file, and uses NODUS_PLUGIN_* export names so module composition can
// rename and inline it exactly like existing plugin sources.
NODUS_PYTHON_ROW_API bool generate_plugin_source(const Descriptor& descriptor,
                                                 std::string* source,
                                                 std::string* error = nullptr);
NODUS_PYTHON_ROW_API bool write_plugin_source(const Descriptor& descriptor,
                                              const std::string& output_path,
                                              std::string* error = nullptr);

class NODUS_PYTHON_ROW_API Interpreter {
public:
    explicit Interpreter(Descriptor descriptor);
    ~Interpreter();
    Interpreter(const Interpreter&) = delete;
    Interpreter& operator=(const Interpreter&) = delete;
    Interpreter(Interpreter&&) noexcept;
    Interpreter& operator=(Interpreter&&) noexcept;

    bool initialize(std::string* error = nullptr);
    void shutdown() noexcept;
    // Ordinary Nodus row plugins exchange AbstractTensor handles.  This
    // scalar bridge preserves that transport so an interpreted row and its
    // Turing-translated replacement have the same stack ABI.
    bool execute_tensor_stack(RawStackFrame& stack, std::string* error = nullptr);
    // Primitive execution remains available for embedding and focused ABI
    // tests; generated ITool plugins use execute_tensor_stack().
    bool execute(RawStackFrame& stack, std::string* error = nullptr);
    const Descriptor& descriptor() const noexcept;

private:
    bool execute_bytes(const std::vector<nodus::tensors::AbstractTensorHandle>& inputs, RawStackFrame& stack,
                       std::string* error);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodus::python_row
