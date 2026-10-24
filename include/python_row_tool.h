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

inline constexpr uint32_t kPythonRowAbiVersion = 1;

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
};

struct Port {
    std::string name;
    PrimitiveType type = PrimitiveType::Float64;
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
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodus::python_row
