#include "plugin_manager.h"
#include "python_row_tool.h"
#include "python_row_workflow.h"
#include "tool_registry.h"

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

std::string read_text(const fs::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream),
                       std::istreambuf_iterator<char>());
}

uint64_t scalar_tensor(double value) {
    const uint64_t handle = nodus_tensor_create(NODUS_DTYPE_F64, 0, nullptr);
    if (!handle || nodus_tensor_write(handle, 0, &value, sizeof(value)) !=
                       static_cast<int64_t>(sizeof(value))) {
        if (handle) nodus_tensor_destroy(handle);
        return 0;
    }
    return handle;
}

bool execute_compiled(ITool& tool) {
    const uint64_t left = scalar_tensor(9.5);
    const uint64_t right = scalar_tensor(4.0);
    if (!left || !right) return false;
    RawStackFrame raw{};
    bool ok = raw_stack_init_frame(raw, 1024);
    nodus::tensors::AbstractTensorHandle left_handle{left};
    nodus::tensors::AbstractTensorHandle right_handle{right};
    ok = ok && raw_stack_push_abstract_tensor(raw, left_handle) &&
         raw_stack_push_abstract_tensor(raw, right_handle);
    ToolStackContext context{};
    context.stack.raw = &raw;
    if (ok) tool.execute_stack(context);
    nodus::tensors::AbstractTensorHandle output{};
    NodusTensorDesc output_desc{};
    double value = 0.0;
    ok = ok && raw_stack_pop_abstract_tensor(raw, output) &&
         nodus_tensor_describe(output.id, &output_desc) == NODUS_OK &&
         output_desc.dtype == NODUS_DTYPE_F64 && output_desc.rank == 0 &&
         nodus_tensor_read(output.id, 0, &value, sizeof(value)) ==
             static_cast<int64_t>(sizeof(value)) &&
         std::abs(value - 13.5) < 1.0e-12;
    if (output.id) nodus_tensor_destroy(output.id);
    nodus_tensor_destroy(left);
    nodus_tensor_destroy(right);
    raw_stack_free_mask(raw);
    if (raw.backend) gp_mem_backend_release(raw.backend);
    return ok;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cout << "python_row_workflow_test: SKIP (plugin loading is Windows-only)\n";
    return 0;
#else
    const fs::path nodus_root = fs::path(NODUS_REPO_ROOT);
    const fs::path turing_root = fs::path(TURING_REPO_ROOT);
    const fs::path python = fs::path(PYTHON_EXECUTABLE);
    const fs::path root = fs::temp_directory_path() / "nodus_python_row_workflow_test";
    const fs::path module_library = root / "module_library";
    const fs::path source = root / "prototype.py";
    const fs::path descriptor_path = root / "prototype.pyabi";
    const std::string authored =
        "def calculate(left, right):\n"
        "    return left + right\n";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    if (ec || !write_text(source, authored)) return 1;

    Descriptor descriptor;
    descriptor.tool_id = "python.upgradable_arithmetic";
    descriptor.tool_name = "Upgradable Python arithmetic";
    descriptor.source_path = fs::absolute(source).generic_string();
    descriptor.module_name = "upgradable_arithmetic";
    descriptor.callable_name = "calculate";
    descriptor.inputs = {
        Port{"left", PrimitiveType::Float64},
        Port{"right", PrimitiveType::Float64},
    };
    descriptor.inputs[0].label = "Left operand";
    descriptor.inputs[0].form_control = "number";
    descriptor.inputs[1].label = "Right operand";
    descriptor.inputs[1].form_control = "number";
    descriptor.outputs = {Port{"result", PrimitiveType::Float64}};
    std::string error;
    if (!nodus::python_row::write_descriptor(
            descriptor, descriptor_path.generic_string(), &error)) {
        std::cerr << error << '\n';
        return 1;
    }

    const fs::path contract = turing_root / "extraction_contracts" /
                              "program_extraction.yaml";
    const std::array<const char*, 4> targets = {
        "cpython", "c", "llvm", "fortran",
    };
    for (const char* target : targets) {
        const bool interpreted = std::string(target) == "cpython";
        const fs::path product = root / (std::string("product-") + target);
        GP_PythonRowDeploymentReport report{};
        char tool_id[1024] = {};
        char retained_python[4096] = {};
        char wrapper[4096] = {};
        if (!gp_python_row_deploy_and_load(
                descriptor_path.string().c_str(), target,
                interpreted ? nullptr : python.string().c_str(),
                interpreted ? nullptr : turing_root.string().c_str(),
                interpreted ? nullptr : contract.string().c_str(),
                interpreted ? nullptr : product.string().c_str(),
                nodus_root.string().c_str(), module_library.string().c_str(),
                &report, tool_id, sizeof(tool_id),
                retained_python, sizeof(retained_python),
                wrapper, sizeof(wrapper)) ||
            report.abi_loaded != 1 || report.source_retained != 1 ||
            report.wrapper_staged != 1 || report.plugin_loaded != 1 ||
            report.compiler_invoked != (interpreted ? 0 : 1) ||
            report.request_written != (interpreted ? 0 : 1)) {
            std::cerr << "deployment failed for target " << target << '\n';
            return 1;
        }
        if (read_text(retained_python) != authored ||
            !fs::is_regular_file(wrapper)) {
            std::cerr << "source or wrapper was not retained for " << target << '\n';
            return 1;
        }
        Descriptor retained_descriptor;
        const fs::path retained_abi = module_library / "source" / "tools" /
                                      "upgradable_arithmetic.pyabi";
        if (!nodus::python_row::read_descriptor(
                retained_abi.generic_string(), &retained_descriptor, &error) ||
            retained_descriptor.inputs.size() != 2 ||
            retained_descriptor.inputs[0].label != "Left operand" ||
            retained_descriptor.inputs[1].form_control != "number") {
            std::cerr << "input presentation metadata was not retained for "
                      << target << '\n';
            return 1;
        }
        if (!interpreted &&
            (read_text(product / "authored-source.py") != authored ||
             !fs::is_regular_file(product / "repository-ssa.pkl") ||
             !fs::is_regular_file(product / "nodus-row-tool-product.receipt"))) {
            std::cerr << "Turing products were not retained for " << target << '\n';
            return 1;
        }
        auto tool = tool_registry_global().create(tool_id);
        if (!tool || tool->port_spec(0).count != 2 ||
            tool->port_spec(1).count != 1) {
            std::cerr << "deployed tool ABI mismatch for " << target << '\n';
            return 1;
        }
        tool->initialize(ToolInitContext{});
        // Live CPython and every Turing backend deliberately share the same
        // AbstractTensor row-stack transport, so a translated tool can replace
        // the prototype without changing its surrounding row wiring.
        const bool executed = execute_compiled(*tool);
        tool->shutdown();
        tool.reset();
        if (!executed || !gp_plugin_unload(tool_id)) {
            std::cerr << "deployed tool did not execute/unload for " << target << '\n';
            return 1;
        }
    }

    // Native selection is strict: a bad target is refused, not redirected to
    // the live interpreter lane.
    GP_PythonRowDeploymentReport refused{};
    char stale_id[16] = "stale";
    if (gp_python_row_deploy_and_load(
            descriptor_path.string().c_str(), "not-a-backend", nullptr,
            nullptr, nullptr, nullptr, nodus_root.string().c_str(),
            module_library.string().c_str(), &refused,
            stale_id, sizeof(stale_id), nullptr, 0, nullptr, 0) ||
        stale_id[0] != '\0' || refused.plugin_loaded != 0) {
        std::cerr << "invalid deployment target did not fail closed\n";
        return 1;
    }

    refused = {};
    std::strcpy(stale_id, "stale");
    const fs::path refused_product = root / "refused-native-product";
    if (gp_python_row_deploy_and_load(
            descriptor_path.string().c_str(), "c",
            (root / "missing-python.exe").string().c_str(),
            turing_root.string().c_str(), contract.string().c_str(),
            refused_product.string().c_str(), nodus_root.string().c_str(),
            module_library.string().c_str(), &refused,
            stale_id, sizeof(stale_id), nullptr, 0, nullptr, 0) ||
        stale_id[0] != '\0' || refused.compiler_invoked != 1 ||
        refused.plugin_loaded != 0) {
        std::cerr << "failed native deployment fell back to CPython\n";
        return 1;
    }

    if (!std::getenv("NODUS_PYTHON_ROW_KEEP_TEMP")) fs::remove_all(root, ec);
    std::cout << "python_row_workflow_test: OK\n";
    return 0;
#endif
}
