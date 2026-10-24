#include "plugin_manager.h"
#include "python_row_tool.h"
#include "tool_registry.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using nodus::python_row::Descriptor;
using nodus::python_row::Port;
using nodus::python_row::PrimitiveType;

namespace {

bool write_text(const fs::path& path, const std::string& text) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream.good()) return false;
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
    return stream.good();
}

template <typename T>
bool push(RawStackFrame& stack, ValueTypeBuiltin builtin, const T& value) {
    return raw_stack_push_typed(
        stack, &value, ValueTypeRegistry::global().builtin(builtin));
}

template <typename T>
bool pop(RawStackFrame& stack, ValueTypeBuiltin builtin, T* value) {
    return raw_stack_pop_typed(
        stack, value, ValueTypeRegistry::global().builtin(builtin));
}

} // namespace

int main() {
#ifndef _WIN32
    std::cout << "python_row_plugin_integration_test: SKIP (plugin loading is Windows-only)\n";
    return 0;
#else
    const fs::path repo = fs::path(NODUS_REPO_ROOT);
    const fs::path root = fs::temp_directory_path() /
                          "nodus_python_row_plugin_integration_test";
    const fs::path source_dir = root / "module_library" / "source" / "tools";
    const fs::path binary_dir = root / "module_library" / "bin";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(source_dir, ec);
    fs::create_directories(binary_dir, ec);
    if (ec) {
        std::cerr << "could not create isolated module-library fixture\n";
        return 1;
    }

    // This is the authored program retained for later UI editing.  The C++
    // plugin below is only the ordinary Nodus ITool membrane that calls it.
    const fs::path python_source = source_dir / "interpreted_arithmetic.py";
    if (!write_text(
            python_source,
            "def calculate(left, right):\n"
            "    return left * 2.0 + right, int(left - right)\n")) {
        std::cerr << "could not retain authored Python source\n";
        return 1;
    }

    Descriptor descriptor;
    descriptor.tool_id = "python.interpreted_arithmetic";
    descriptor.tool_name = "Interpreted Python arithmetic";
    descriptor.source_path = fs::absolute(python_source).generic_string();
    descriptor.module_name = "interpreted_arithmetic_python_row";
    descriptor.callable_name = "calculate";
    descriptor.inputs = {
        Port{"left", PrimitiveType::Float64},
        Port{"right", PrimitiveType::Int32},
    };
    descriptor.outputs = {
        Port{"combined", PrimitiveType::Float64},
        Port{"difference", PrimitiveType::Int64},
    };

    std::string error;
    const fs::path abi = source_dir / "interpreted_arithmetic.pyabi";
    const fs::path wrapper = source_dir /
                             "interpreted_arithmetic_python_row.cpp";
    if (!nodus::python_row::write_descriptor(
            descriptor, abi.generic_string(), &error) ||
        !nodus::python_row::write_plugin_source(
            descriptor, wrapper.generic_string(), &error)) {
        std::cerr << "could not generate interpreted row membrane: " << error << "\n";
        return 1;
    }

    char loaded_id[1024] = {};
    if (!gp_plugin_build_module_and_load(
            wrapper.string().c_str(), repo.string().c_str(),
            binary_dir.string().c_str(), nullptr,
            loaded_id, static_cast<int>(sizeof(loaded_id)))) {
        std::cerr << "generated interpreted row did not build and load\n";
        return 1;
    }

    const fs::path retained_binary = binary_dir /
                                     "interpreted_arithmetic_python_row.dll";
    if (!fs::is_regular_file(python_source) || !fs::is_regular_file(abi) ||
        !fs::is_regular_file(wrapper) || !fs::is_regular_file(retained_binary)) {
        std::cerr << "module library did not retain Python, ABI, wrapper, and binary\n";
        return 1;
    }

    auto tool = tool_registry_global().create(loaded_id);
    if (!tool || tool->port_spec(0).count != 2 ||
        tool->port_spec(1).count != 2) {
        std::cerr << "loaded interpreted row was not registered with its ABI\n";
        return 1;
    }

    tool->initialize(ToolInitContext{});
    RawStackFrame raw{};
    if (!raw_stack_init_frame(raw, 256)) {
        std::cerr << "could not initialize row stack\n";
        return 1;
    }
    const double left = 9.5;
    const int32_t right = 4;
    if (!push(raw, VT_FLOAT64, left) || !push(raw, VT_INT32, right)) {
        std::cerr << "could not push interpreted row inputs\n";
        return 1;
    }
    ToolStackContext stack{};
    stack.stack.raw = &raw;
    tool->execute_stack(stack);

    int64_t difference = 0;
    double combined = 0.0;
    const bool executed = pop(raw, VT_INT64, &difference) &&
                          pop(raw, VT_FLOAT64, &combined) &&
                          raw.byte_count == 0 &&
                          difference == 5 &&
                          std::abs(combined - 23.0) < 1.0e-12;
    raw_stack_free_mask(raw);
    if (raw.backend) {
        gp_mem_backend_release(raw.backend);
        raw.backend = nullptr;
    }
    tool->shutdown();
    tool.reset();

    if (!executed) {
        std::cerr << "registered interpreted row returned the wrong values\n";
        return 1;
    }
    if (!gp_plugin_unload(loaded_id)) {
        std::cerr << "registered interpreted row could not be unloaded\n";
        return 1;
    }

    if (!std::getenv("NODUS_PYTHON_ROW_KEEP_TEMP")) fs::remove_all(root, ec);
    std::cout << "python_row_plugin_integration_test: OK\n";
    return 0;
#endif
}
