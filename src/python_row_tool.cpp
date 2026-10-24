#include "python_row_tool.h"

#include <Python.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <utility>

namespace nodus::python_row {
namespace {

std::mutex g_interpreter_init_mutex;

void set_error(std::string* error, std::string message) {
    if (error) *error = std::move(message);
}

std::string python_error_text() {
    if (!PyErr_Occurred()) return "unknown Python error";
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);
    std::string out = "Python exception";
    PyObject* text = value ? PyObject_Str(value) : nullptr;
    if (text) {
        const char* utf8 = PyUnicode_AsUTF8(text);
        if (utf8) out = utf8;
    }
    Py_XDECREF(text);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
    PyErr_Clear();
    return out;
}

bool ensure_interpreter(std::string* error) {
    std::lock_guard<std::mutex> lock(g_interpreter_init_mutex);
    if (Py_IsInitialized()) return true;
    Py_Initialize();
    if (!Py_IsInitialized()) {
        set_error(error, "CPython initialization failed");
        return false;
    }
    // This library owns no process-wide finalization. Release the initial GIL
    // so Nodus worker threads can enter through PyGILState_Ensure().
    PyEval_SaveThread();
    return true;
}

bool valid_identifier(const std::string& text) {
    if (text.empty()) return false;
    const auto first = static_cast<unsigned char>(text.front());
    if (!(std::isalpha(first) || first == '_')) return false;
    return std::all_of(text.begin() + 1, text.end(), [](char c) {
        const auto uc = static_cast<unsigned char>(c);
        return std::isalnum(uc) || c == '_';
    });
}

bool supported_type(PrimitiveType type) {
    return type >= PrimitiveType::Float32 && type <= PrimitiveType::UInt64;
}

bool parse_primitive_type(const std::string& name, PrimitiveType& type) {
    for (uint32_t raw = static_cast<uint32_t>(PrimitiveType::Float32);
         raw <= static_cast<uint32_t>(PrimitiveType::UInt64); ++raw) {
        const auto candidate = static_cast<PrimitiveType>(raw);
        if (name == primitive_type_name(candidate)) {
            type = candidate;
            return true;
        }
    }
    return false;
}

const char* primitive_enum_name(PrimitiveType type) {
    switch (type) {
        case PrimitiveType::Float32: return "Float32";
        case PrimitiveType::Float64: return "Float64";
        case PrimitiveType::Int8: return "Int8";
        case PrimitiveType::Int16: return "Int16";
        case PrimitiveType::Int32: return "Int32";
        case PrimitiveType::Int64: return "Int64";
        case PrimitiveType::UInt8: return "UInt8";
        case PrimitiveType::UInt16: return "UInt16";
        case PrimitiveType::UInt32: return "UInt32";
        case PrimitiveType::UInt64: return "UInt64";
    }
    return "Invalid";
}

std::string cpp_escape(const std::string& input) {
    std::ostringstream out;
    for (unsigned char c : input) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20 || c >= 0x7f) {
                    char buf[5]{};
                    std::snprintf(buf, sizeof(buf), "\\x%02X", static_cast<unsigned>(c));
                    out << buf;
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    return out.str();
}

std::string cpp_identifier(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 1);
    if (input.empty() || std::isdigit(static_cast<unsigned char>(input.front()))) out.push_back('_');
    for (char c : input) {
        const auto uc = static_cast<unsigned char>(c);
        out.push_back((std::isalnum(uc) || c == '_') ? c : '_');
    }
    return out;
}

ValueTypeId registered_type(PrimitiveType type) {
    return ValueTypeRegistry::global().builtin(builtin_type(type));
}

template <typename T>
bool pop_exact(RawStackFrame& stack, ValueTypeId type, T& value) {
    return raw_stack_pop_typed(stack, &value, type) != 0;
}

template <typename T>
bool push_exact(RawStackFrame& stack, ValueTypeId type, const T& value) {
    return raw_stack_push_typed(stack, &value, type) != 0;
}

struct NativeValue {
    PrimitiveType type = PrimitiveType::Float64;
    std::array<unsigned char, sizeof(uint64_t)> bytes{};
};

template <typename T>
T native_load(const NativeValue& value) {
    T out{};
    std::memcpy(&out, value.bytes.data(), sizeof(out));
    return out;
}

template <typename T>
bool pop_native(RawStackFrame& stack, ValueTypeId tid, NativeValue& out) {
    T value{};
    if (!pop_exact(stack, tid, value)) return false;
    std::memcpy(out.bytes.data(), &value, sizeof(value));
    return true;
}

bool pop_value(RawStackFrame& stack, PrimitiveType type, NativeValue& out) {
    out.type = type;
    const ValueTypeId tid = registered_type(type);
    switch (type) {
        case PrimitiveType::Float32: return pop_native<float>(stack, tid, out);
        case PrimitiveType::Float64: return pop_native<double>(stack, tid, out);
        case PrimitiveType::Int8: return pop_native<int8_t>(stack, tid, out);
        case PrimitiveType::Int16: return pop_native<int16_t>(stack, tid, out);
        case PrimitiveType::Int32: return pop_native<int32_t>(stack, tid, out);
        case PrimitiveType::Int64: return pop_native<int64_t>(stack, tid, out);
        case PrimitiveType::UInt8: return pop_native<uint8_t>(stack, tid, out);
        case PrimitiveType::UInt16: return pop_native<uint16_t>(stack, tid, out);
        case PrimitiveType::UInt32: return pop_native<uint32_t>(stack, tid, out);
        case PrimitiveType::UInt64: return pop_native<uint64_t>(stack, tid, out);
    }
    return false;
}

bool push_value(RawStackFrame& stack, const NativeValue& value) {
    const ValueTypeId tid = registered_type(value.type);
    switch (value.type) {
        case PrimitiveType::Float32: { const auto v = native_load<float>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::Float64: { const auto v = native_load<double>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::Int8: { const auto v = native_load<int8_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::Int16: { const auto v = native_load<int16_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::Int32: { const auto v = native_load<int32_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::Int64: { const auto v = native_load<int64_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::UInt8: { const auto v = native_load<uint8_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::UInt16: { const auto v = native_load<uint16_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::UInt32: { const auto v = native_load<uint32_t>(value); return push_exact(stack, tid, v); }
        case PrimitiveType::UInt64: { const auto v = native_load<uint64_t>(value); return push_exact(stack, tid, v); }
    }
    return false;
}

PyObject* to_python(const NativeValue& value) {
    switch (value.type) {
        case PrimitiveType::Float32:
            return PyFloat_FromDouble(native_load<float>(value));
        case PrimitiveType::Float64:
            return PyFloat_FromDouble(native_load<double>(value));
        case PrimitiveType::Int8:
            return PyLong_FromLong(native_load<int8_t>(value));
        case PrimitiveType::Int16:
            return PyLong_FromLong(native_load<int16_t>(value));
        case PrimitiveType::Int32:
            return PyLong_FromLong(native_load<int32_t>(value));
        case PrimitiveType::Int64:
            return PyLong_FromLongLong(native_load<int64_t>(value));
        case PrimitiveType::UInt8:
            return PyLong_FromUnsignedLong(native_load<uint8_t>(value));
        case PrimitiveType::UInt16:
            return PyLong_FromUnsignedLong(native_load<uint16_t>(value));
        case PrimitiveType::UInt32:
            return PyLong_FromUnsignedLong(native_load<uint32_t>(value));
        case PrimitiveType::UInt64:
            return PyLong_FromUnsignedLongLong(native_load<uint64_t>(value));
    }
    return nullptr;
}

template <typename T, typename Wide>
bool checked_integer(Wide value, NativeValue& out, std::string* error) {
    if (PyErr_Occurred() || value < static_cast<Wide>(std::numeric_limits<T>::min()) ||
        value > static_cast<Wide>(std::numeric_limits<T>::max())) {
        PyErr_Clear();
        set_error(error, "Python integer result is outside the declared port type range");
        return false;
    }
    const T narrowed = static_cast<T>(value);
    std::memcpy(out.bytes.data(), &narrowed, sizeof(narrowed));
    return true;
}

bool from_python(PyObject* object, PrimitiveType type, NativeValue& out, std::string* error) {
    out.type = type;
    if (type == PrimitiveType::Float32 || type == PrimitiveType::Float64) {
        if (!PyFloat_Check(object) && !PyLong_Check(object)) {
            set_error(error, "Python result is not numeric for a declared floating-point port");
            return false;
        }
        const double value = PyFloat_AsDouble(object);
        if (PyErr_Occurred()) {
            set_error(error, python_error_text());
            return false;
        }
        if (type == PrimitiveType::Float32) {
            const float narrowed = static_cast<float>(value);
            std::memcpy(out.bytes.data(), &narrowed, sizeof(narrowed));
        } else {
            std::memcpy(out.bytes.data(), &value, sizeof(value));
        }
        return true;
    }
    if (!PyLong_Check(object)) {
        set_error(error, "Python result is not an int for a declared integer port");
        return false;
    }
    switch (type) {
        case PrimitiveType::Int8:
            return checked_integer<int8_t>(PyLong_AsLongLong(object), out, error);
        case PrimitiveType::Int16:
            return checked_integer<int16_t>(PyLong_AsLongLong(object), out, error);
        case PrimitiveType::Int32:
            return checked_integer<int32_t>(PyLong_AsLongLong(object), out, error);
        case PrimitiveType::Int64: {
            const long long value = PyLong_AsLongLong(object);
            if (PyErr_Occurred()) { set_error(error, python_error_text()); return false; }
            const int64_t narrowed = static_cast<int64_t>(value);
            std::memcpy(out.bytes.data(), &narrowed, sizeof(narrowed));
            return true;
        }
        case PrimitiveType::UInt8:
            return checked_integer<uint8_t>(PyLong_AsUnsignedLongLong(object), out, error);
        case PrimitiveType::UInt16:
            return checked_integer<uint16_t>(PyLong_AsUnsignedLongLong(object), out, error);
        case PrimitiveType::UInt32:
            return checked_integer<uint32_t>(PyLong_AsUnsignedLongLong(object), out, error);
        case PrimitiveType::UInt64: {
            const unsigned long long value = PyLong_AsUnsignedLongLong(object);
            if (PyErr_Occurred()) { set_error(error, python_error_text()); return false; }
            const uint64_t narrowed = static_cast<uint64_t>(value);
            std::memcpy(out.bytes.data(), &narrowed, sizeof(narrowed));
            return true;
        }
        case PrimitiveType::Float32:
        case PrimitiveType::Float64:
            break;
    }
    set_error(error, "unsupported Python row primitive type");
    return false;
}

bool restore_inputs(RawStackFrame& stack, const std::vector<NativeValue>& inputs) {
    for (const NativeValue& value : inputs) {
        if (!push_value(stack, value)) return false;
    }
    return true;
}

} // namespace

ValueTypeBuiltin builtin_type(PrimitiveType type) {
    switch (type) {
        case PrimitiveType::Float32: return VT_FLOAT32;
        case PrimitiveType::Float64: return VT_FLOAT64;
        case PrimitiveType::Int8: return VT_INT8;
        case PrimitiveType::Int16: return VT_INT16;
        case PrimitiveType::Int32: return VT_INT32;
        case PrimitiveType::Int64: return VT_INT64;
        case PrimitiveType::UInt8: return VT_UINT8;
        case PrimitiveType::UInt16: return VT_UINT16;
        case PrimitiveType::UInt32: return VT_UINT32;
        case PrimitiveType::UInt64: return VT_UINT64;
    }
    return VT_BUILTIN_COUNT;
}

const char* primitive_type_name(PrimitiveType type) {
    switch (type) {
        case PrimitiveType::Float32: return "float32";
        case PrimitiveType::Float64: return "float64";
        case PrimitiveType::Int8: return "int8";
        case PrimitiveType::Int16: return "int16";
        case PrimitiveType::Int32: return "int32";
        case PrimitiveType::Int64: return "int64";
        case PrimitiveType::UInt8: return "uint8";
        case PrimitiveType::UInt16: return "uint16";
        case PrimitiveType::UInt32: return "uint32";
        case PrimitiveType::UInt64: return "uint64";
    }
    return "invalid";
}

bool validate_descriptor(const Descriptor& descriptor, std::string* error) {
    if (descriptor.abi_version != kPythonRowAbiVersion) {
        set_error(error, "unsupported Python row ABI version");
        return false;
    }
    if (descriptor.mode != ExecutionMode::Interpreted) {
        set_error(error, "the interpreter lane accepts mode=Interpreted only; no compiled-SSA fallback is performed");
        return false;
    }
    if (descriptor.tool_id.empty() || descriptor.tool_name.empty()) {
        set_error(error, "tool_id and tool_name must not be empty");
        return false;
    }
    if (descriptor.source_path.empty()) {
        set_error(error, "source_path must name the retained Python source");
        return false;
    }
    if (!valid_identifier(descriptor.module_name) || !valid_identifier(descriptor.callable_name)) {
        set_error(error, "module_name and callable_name must be Python identifiers");
        return false;
    }
    auto validate_ports = [&](const std::vector<Port>& ports, const char* direction) {
        for (size_t i = 0; i < ports.size(); ++i) {
            if (ports[i].name.empty()) {
                set_error(error, std::string(direction) + " port " + std::to_string(i) + " has no name");
                return false;
            }
            if (!supported_type(ports[i].type)) {
                set_error(error, std::string(direction) + " port " + ports[i].name + " uses a pointer, tensor, dynamic, or unknown type");
                return false;
            }
            for (size_t j = 0; j < i; ++j) {
                if (ports[j].name == ports[i].name) {
                    set_error(error, std::string(direction) + " port name is duplicated: " + ports[i].name);
                    return false;
                }
            }
        }
        return true;
    };
    return validate_ports(descriptor.inputs, "input") && validate_ports(descriptor.outputs, "output");
}

bool write_descriptor(const Descriptor& descriptor, const std::string& output_path, std::string* error) {
    if (!validate_descriptor(descriptor, error)) return false;
    std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        set_error(error, "could not open Python row ABI descriptor path: " + output_path);
        return false;
    }
    out << "NODUSPYROW " << descriptor.abi_version << "\n";
    out << "MODE INTERPRETED\n";
    out << "TOOL_ID " << std::quoted(descriptor.tool_id) << "\n";
    out << "TOOL_NAME " << std::quoted(descriptor.tool_name) << "\n";
    out << "SOURCE " << std::quoted(descriptor.source_path) << "\n";
    out << "MODULE " << std::quoted(descriptor.module_name) << "\n";
    out << "CALLABLE " << std::quoted(descriptor.callable_name) << "\n";
    for (const Port& port : descriptor.inputs) {
        out << "INPUT " << std::quoted(port.name) << " " << primitive_type_name(port.type) << "\n";
    }
    for (const Port& port : descriptor.outputs) {
        out << "OUTPUT " << std::quoted(port.name) << " " << primitive_type_name(port.type) << "\n";
    }
    if (!out.good()) {
        set_error(error, "could not write Python row ABI descriptor path: " + output_path);
        return false;
    }
    return true;
}

bool read_descriptor(const std::string& input_path, Descriptor* descriptor, std::string* error) {
    if (!descriptor) {
        set_error(error, "descriptor output pointer is null");
        return false;
    }
    std::ifstream in(input_path, std::ios::binary);
    if (!in.good()) {
        set_error(error, "could not open Python row ABI descriptor path: " + input_path);
        return false;
    }
    Descriptor parsed;
    std::string line;
    if (!std::getline(in, line)) {
        set_error(error, "empty Python row ABI descriptor");
        return false;
    }
    {
        std::istringstream header(line);
        std::string magic;
        header >> magic >> parsed.abi_version;
        if (magic != "NODUSPYROW" || !header) {
            set_error(error, "invalid Python row ABI descriptor header");
            return false;
        }
    }
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::string tag;
        row >> tag;
        if (tag == "MODE") {
            std::string mode;
            row >> mode;
            if (mode != "INTERPRETED") {
                set_error(error, "Python row ABI descriptor requests a non-interpreted mode");
                return false;
            }
            parsed.mode = ExecutionMode::Interpreted;
        } else if (tag == "TOOL_ID") {
            row >> std::quoted(parsed.tool_id);
        } else if (tag == "TOOL_NAME") {
            row >> std::quoted(parsed.tool_name);
        } else if (tag == "SOURCE") {
            row >> std::quoted(parsed.source_path);
        } else if (tag == "MODULE") {
            row >> std::quoted(parsed.module_name);
        } else if (tag == "CALLABLE") {
            row >> std::quoted(parsed.callable_name);
        } else if (tag == "INPUT" || tag == "OUTPUT") {
            Port port;
            std::string type_name;
            row >> std::quoted(port.name) >> type_name;
            if (!row || !parse_primitive_type(type_name, port.type)) {
                set_error(error, "invalid primitive type in Python row ABI descriptor");
                return false;
            }
            (tag == "INPUT" ? parsed.inputs : parsed.outputs).push_back(std::move(port));
        } else {
            set_error(error, "unknown Python row ABI descriptor record: " + tag);
            return false;
        }
        if (!row) {
            set_error(error, "malformed Python row ABI descriptor record: " + tag);
            return false;
        }
    }
    if (!validate_descriptor(parsed, error)) return false;
    *descriptor = std::move(parsed);
    return true;
}

bool generate_plugin_source(const Descriptor& descriptor, std::string* source, std::string* error) {
    if (!source) {
        set_error(error, "source output pointer is null");
        return false;
    }
    if (!validate_descriptor(descriptor, error)) return false;
    const std::string class_name = "PythonRowTool_" + cpp_identifier(descriptor.tool_id);
    std::ostringstream out;
    out << "// Generated Nodus interpreted Python row; ABI v" << descriptor.abi_version << "\n";
    out << "#include \"python_row_tool.h\"\n";
    out << "#include \"tool_api.h\"\n";
    out << "#include <cstdio>\n#include <stdexcept>\n#include <string>\n\n";
    out << "namespace {\n";
    out << "nodus::python_row::Descriptor make_descriptor() {\n";
    out << "    nodus::python_row::Descriptor d;\n";
    out << "    d.abi_version = " << descriptor.abi_version << "u;\n";
    out << "    d.mode = nodus::python_row::ExecutionMode::Interpreted;\n";
    out << "    d.tool_id = \"" << cpp_escape(descriptor.tool_id) << "\";\n";
    out << "    d.tool_name = \"" << cpp_escape(descriptor.tool_name) << "\";\n";
    out << "    d.source_path = \"" << cpp_escape(descriptor.source_path) << "\";\n";
    out << "    d.module_name = \"" << cpp_escape(descriptor.module_name) << "\";\n";
    out << "    d.callable_name = \"" << cpp_escape(descriptor.callable_name) << "\";\n";
    for (const Port& port : descriptor.inputs) {
        out << "    d.inputs.push_back({\"" << cpp_escape(port.name)
            << "\", nodus::python_row::PrimitiveType::";
        out << primitive_enum_name(port.type) << "});\n";
    }
    for (const Port& port : descriptor.outputs) {
        out << "    d.outputs.push_back({\"" << cpp_escape(port.name)
            << "\", nodus::python_row::PrimitiveType::";
        out << primitive_enum_name(port.type) << "});\n";
    }
    out << "    return d;\n}\n\n";
    out << "class " << class_name << " final : public ITool {\npublic:\n";
    out << "    " << class_name << "() : interpreter_(make_descriptor()) {}\n";
    out << "    const char* id_cstr() const noexcept override { return \"" << cpp_escape(descriptor.tool_id) << "\"; }\n";
    out << "    const char* name_cstr() const noexcept override { return \"" << cpp_escape(descriptor.tool_name) << "\"; }\n";
    out << "    void initialize(const ToolInitContext&) override { std::string e; if (!interpreter_.initialize(&e)) throw std::runtime_error(e); }\n";
    out << "    void shutdown() override { interpreter_.shutdown(); }\n";
    out << "    void tick(double, HostAPI&) override {}\n";
    out << "    void render(RenderContext&) override {}\n";
    out << "    int32_t port_count() const override { return 2; }\n";
    out << "    ToolPortSpec port_spec(int32_t i) const override {\n";
    out << "        if (i == 0) return ToolPortSpec{ToolPortKind::Argument, " << static_cast<int32_t>(descriptor.inputs.size()) << "};\n";
    out << "        if (i == 1) return ToolPortSpec{ToolPortKind::Return, " << static_cast<int32_t>(descriptor.outputs.size()) << "};\n";
    out << "        return ToolPortSpec{};\n    }\n";
    out << "    void execute_stack(ToolStackContext& c) override {\n";
    out << "        if (!c.stack.raw) return; std::string e; if (!interpreter_.execute(*c.stack.raw, &e)) std::fprintf(stderr, \"python row "
        << cpp_escape(descriptor.tool_id) << ": %s\\n\", e.c_str());\n    }\n";
    out << "private:\n    nodus::python_row::Interpreter interpreter_;\n};\n} // namespace\n\n";
    out << "extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new " << class_name << "(); }\n";
    out << "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }\n";
    out << "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return \""
        << cpp_escape(descriptor.source_path) << "\"; }\n";
    *source = out.str();
    return true;
}

bool write_plugin_source(const Descriptor& descriptor, const std::string& output_path, std::string* error) {
    std::string source;
    if (!generate_plugin_source(descriptor, &source, error)) return false;
    std::ofstream stream(output_path, std::ios::binary | std::ios::trunc);
    if (!stream.good()) {
        set_error(error, "could not open generated plugin source path: " + output_path);
        return false;
    }
    stream.write(source.data(), static_cast<std::streamsize>(source.size()));
    if (!stream.good()) {
        set_error(error, "could not write generated plugin source path: " + output_path);
        return false;
    }
    return true;
}

struct Interpreter::Impl {
    explicit Impl(Descriptor d) : descriptor(std::move(d)) {}
    Descriptor descriptor;
    PyObject* module = nullptr;
    PyObject* callable = nullptr;
    bool initialized = false;
};

Interpreter::Interpreter(Descriptor descriptor) : impl_(new Impl(std::move(descriptor))) {}
Interpreter::~Interpreter() { shutdown(); }
Interpreter::Interpreter(Interpreter&&) noexcept = default;
Interpreter& Interpreter::operator=(Interpreter&&) noexcept = default;

bool Interpreter::initialize(std::string* error) {
    if (impl_->initialized) return true;
    if (!validate_descriptor(impl_->descriptor, error) || !ensure_interpreter(error)) return false;
    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* util = PyImport_ImportModule("importlib.util");
    PyObject* spec_factory = util ? PyObject_GetAttrString(util, "spec_from_file_location") : nullptr;
    PyObject* spec = spec_factory ? PyObject_CallFunction(spec_factory, "ss",
        impl_->descriptor.module_name.c_str(), impl_->descriptor.source_path.c_str()) : nullptr;
    PyObject* module_factory = util ? PyObject_GetAttrString(util, "module_from_spec") : nullptr;
    PyObject* module = (module_factory && spec) ? PyObject_CallOneArg(module_factory, spec) : nullptr;
    PyObject* loader = spec ? PyObject_GetAttrString(spec, "loader") : nullptr;
    PyObject* executed = (loader && module) ? PyObject_CallMethod(loader, "exec_module", "O", module) : nullptr;
    PyObject* callable = (executed && module)
        ? PyObject_GetAttrString(module, impl_->descriptor.callable_name.c_str()) : nullptr;
    bool ok = callable && PyCallable_Check(callable);
    if (!ok) set_error(error, python_error_text());
    Py_XDECREF(executed);
    Py_XDECREF(loader);
    Py_XDECREF(module_factory);
    Py_XDECREF(spec);
    Py_XDECREF(spec_factory);
    Py_XDECREF(util);
    if (ok) {
        impl_->module = module;
        impl_->callable = callable;
        impl_->initialized = true;
    } else {
        Py_XDECREF(callable);
        Py_XDECREF(module);
    }
    PyGILState_Release(gil);
    return ok;
}

void Interpreter::shutdown() noexcept {
    if (!impl_ || (!impl_->module && !impl_->callable)) return;
    if (!Py_IsInitialized()) {
        impl_->module = nullptr;
        impl_->callable = nullptr;
        impl_->initialized = false;
        return;
    }
    PyGILState_STATE gil = PyGILState_Ensure();
    Py_CLEAR(impl_->callable);
    Py_CLEAR(impl_->module);
    impl_->initialized = false;
    PyGILState_Release(gil);
}

bool Interpreter::execute(RawStackFrame& stack, std::string* error) {
    if (!initialize(error)) return false;
    const size_t input_count = impl_->descriptor.inputs.size();
    std::vector<NativeValue> inputs(input_count);
    // Stack top is the last declared argument. Validate all types before the
    // first pop so a type/arity refusal leaves the caller's frame untouched.
    size_t cursor = stack.byte_count;
    size_t output_bytes = 0;
    for (const Port& output : impl_->descriptor.outputs) {
        const ValueType* value_type = ValueTypeRegistry::global().get(registered_type(output.type));
        if (!value_type || value_type->size > stack.byte_capacity - output_bytes) {
            set_error(error, "invalid or oversized declared Python row output ABI");
            return false;
        }
        output_bytes += value_type->size;
    }
    if (output_bytes > stack.byte_capacity - cursor) {
        set_error(error, "Python row outputs cannot fit after consuming declared inputs");
        return false;
    }
    for (size_t reverse = 0; reverse < input_count; ++reverse) {
        const Port& port = impl_->descriptor.inputs[input_count - 1 - reverse];
        const ValueTypeId tid = registered_type(port.type);
        const ValueType* value_type = ValueTypeRegistry::global().get(tid);
        if (!value_type || cursor < value_type->size || !stack.types_per_byte ||
            stack.types_per_byte[cursor - 1] != tid) {
            set_error(error, "stack arity/type mismatch at input port " + port.name);
            return false;
        }
        const size_t start = cursor - value_type->size;
        for (size_t byte = start; byte < cursor; ++byte) {
            if (stack.types_per_byte[byte] != tid) {
                set_error(error, "non-contiguous stack type at input port " + port.name);
                return false;
            }
        }
        cursor = start;
    }
    size_t popped = 0;
    for (size_t reverse = 0; reverse < input_count; ++reverse) {
        const size_t index = input_count - 1 - reverse;
        if (!pop_value(stack, impl_->descriptor.inputs[index].type, inputs[index])) {
            for (size_t restore = input_count - popped; restore < input_count; ++restore) {
                push_value(stack, inputs[restore]);
            }
            set_error(error, "failed to pop validated Python row input");
            return false;
        }
        ++popped;
    }

    PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* args = PyTuple_New(static_cast<Py_ssize_t>(input_count));
    bool ok = args != nullptr;
    for (size_t i = 0; ok && i < input_count; ++i) {
        PyObject* value = to_python(inputs[i]);
        if (!value) { ok = false; break; }
        PyTuple_SET_ITEM(args, static_cast<Py_ssize_t>(i), value); // steals value
    }
    PyObject* result = ok ? PyObject_CallObject(impl_->callable, args) : nullptr;
    Py_XDECREF(args);
    if (!result) {
        set_error(error, python_error_text());
        PyGILState_Release(gil);
        restore_inputs(stack, inputs);
        return false;
    }

    const size_t output_count = impl_->descriptor.outputs.size();
    std::vector<NativeValue> outputs(output_count);
    if (output_count == 0) {
        ok = (result == Py_None);
        if (!ok) set_error(error, "zero-output Python row must return None");
    } else if (output_count == 1) {
        ok = from_python(result, impl_->descriptor.outputs[0].type, outputs[0], error);
    } else if (!PyTuple_Check(result) || PyTuple_GET_SIZE(result) != static_cast<Py_ssize_t>(output_count)) {
        ok = false;
        set_error(error, "Python row result tuple arity does not match declared outputs");
    } else {
        for (size_t i = 0; ok && i < output_count; ++i) {
            ok = from_python(PyTuple_GET_ITEM(result, static_cast<Py_ssize_t>(i)),
                             impl_->descriptor.outputs[i].type, outputs[i], error);
        }
    }
    Py_DECREF(result);
    PyGILState_Release(gil);
    if (!ok) {
        restore_inputs(stack, inputs);
        return false;
    }
    for (const NativeValue& output : outputs) {
        if (!push_value(stack, output)) {
            // Capacity was preflighted above, so this indicates an invariant
            // failure rather than ordinary user input.
            set_error(error, "Python row output does not fit the RawStackFrame");
            return false;
        }
    }
    return true;
}

const Descriptor& Interpreter::descriptor() const noexcept { return impl_->descriptor; }

} // namespace nodus::python_row
