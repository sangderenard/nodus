#include "turing_product_adopter.h"

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "plugin_manager.h"
#include "tool_registry.h"
#include "value_types.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return out.good();
}

bool run(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

uint64_t tensor_f64(const std::vector<double>& values) {
    const uint64_t shape[1] = {values.size()};
    const uint64_t handle = nodus_tensor_create(NODUS_DTYPE_F64, 1, shape);
    if (!handle) return 0;
    if (nodus_tensor_write(handle, 0, values.data(), values.size() * sizeof(double)) !=
        static_cast<int64_t>(values.size() * sizeof(double))) {
        nodus_tensor_destroy(handle);
        return 0;
    }
    return handle;
}

bool tensor_values(uint64_t handle, std::vector<double>* values) {
    NodusTensorDesc desc{};
    if (nodus_tensor_describe(handle, &desc) != NODUS_OK ||
        desc.dtype != NODUS_DTYPE_F64 || desc.rank != 1) return false;
    values->resize(static_cast<size_t>(desc.shape[0]));
    return nodus_tensor_read(handle, 0, values->data(), values->size() * sizeof(double)) ==
           static_cast<int64_t>(values->size() * sizeof(double));
}

bool equal(const std::vector<double>& left, const std::vector<double>& right) {
    if (left.size() != right.size()) return false;
    for (size_t i = 0; i < left.size(); ++i) {
        if (std::fabs(left[i] - right[i]) > 1e-12) return false;
    }
    return true;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cout << "turing_product_adopter_test skipped: fake DLL fixture is Windows-only\n";
    return 0;
#else
    const fs::path repo = fs::path(NODUS_REPO_ROOT);
    const fs::path test_root = repo / "module_library" / "test_build" / "turing_product_adopter_test";
    const fs::path product = test_root / "product";
    const fs::path module_library = test_root / "module_library";
    std::error_code ec;
    fs::remove_all(test_root, ec);
    fs::create_directories(product / "backend");

    const fs::path backend_source = test_root / "fake_product_backend.cpp";
    if (!write_text(backend_source, R"CPP(
#include <cstdint>
extern "C" __declspec(dllexport) void row_entry(void** buffers, int64_t* extents) {
    auto* input = static_cast<double*>(buffers[0]);
    auto* workspace = static_cast<double*>(buffers[1]);
    auto* output = static_cast<double*>(buffers[2]);
    for (int64_t i = 0; i < extents[0]; ++i) {
        workspace[i] = input[i] * 3.0;
        output[i] = workspace[i] + 2.0;
    }
}
)CPP")) return 1;

    const fs::path backend_project = test_root / "backend_project";
    const fs::path backend_build = backend_project / "build";
    std::ostringstream backend_cmake;
    backend_cmake << "cmake_minimum_required(VERSION 3.15)\n"
                     "project(fake_product_backend LANGUAGES CXX)\n"
                     "add_library(fake_product_backend SHARED \""
                  << backend_source.generic_string() << "\")\n"
                     "set_target_properties(fake_product_backend PROPERTIES CXX_STANDARD 17 "
                     "RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\" "
                     "RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
    if (!write_text(backend_project / "CMakeLists.txt", backend_cmake.str()) ||
        !run("cmake -S \"" + backend_project.string() + "\" -B \"" + backend_build.string() + "\"") ||
        !run("cmake --build \"" + backend_build.string() + "\" --config Release")) {
        std::cerr << "failed to build fake product backend\n";
        return 1;
    }
    const fs::path backend_dll = backend_build / "out" / "fake_product_backend.dll";
    const fs::path product_dll = product / "backend" / "fake_product_backend.dll";
    fs::copy_file(backend_dll, product_dll, fs::copy_options::overwrite_existing);

    const fs::path descriptor = product / "backend" / "adopter_fixture.nodus_buffer_abi.txt";
    if (!write_text(descriptor,
        "NODUS BUFFER ABI V1\n"
        "BACKEND \"c\"\n"
        "EXTENT_DTYPE \"i64\"\n"
        "TOOL \"imported.adopter_fixture.c\" \"Adopter Fixture C\"\n"
        "ENTRY \"row_entry\"\n"
        "VALUE INPUT \"%t1\" \"f64\" 1 3\n"
        "VALUE WORKSPACE \"%t2\" \"f64\" 1 3\n"
        "VALUE OUTPUT \"%t3\" \"f64\" 1 3\n"
        "BUFFER \"%t1\"\n"
        "BUFFER \"%t2\"\n"
        "BUFFER \"%t3\"\n"
        "EXTENT \"%t1\" 0\n") ||
        !write_text(product / "nodus-row-tool-product.json",
                    "{\n  \"schema\": \"nodus-turing-row-tool-product-v1\",\n"
                    "  \"backend\": \"c\",\n"
                    "  \"loadable\": true\n}\n") ||
        !write_text(product / "nodus-row-tool-product.receipt",
                    "NODUS ROW TOOL PRODUCT V1\n"
                    "BACKEND \"c\"\n"
                    "MANIFEST \"nodus-row-tool-product.json\"\n"
                    "ABI \"backend/adopter_fixture.nodus_buffer_abi.txt\"\n"
                    "BINARY \"backend/fake_product_backend.dll\"\n")) {
        return 1;
    }

    GP_TuringProductAdoptionReport report{};
    char loaded_id[1024] = {};
    char staged_source[2048] = {};
    if (!gp_turing_product_adopt_and_load(
            (product / "nodus-row-tool-product.receipt").string().c_str(),
            repo.string().c_str(), module_library.string().c_str(), &report,
            loaded_id, sizeof(loaded_id), staged_source, sizeof(staged_source)) ||
        report.receipt_validated != 1 || report.wrapper_imported != 1 ||
        report.wrapper_staged != 1 || report.plugin_loaded != 1) {
        std::cerr << "valid product was not adopted and loaded\n";
        return 1;
    }

    const fs::path expected_source = module_library / "source" / "tools" /
                                     "imported_adopter_fixture_c.cpp";
    const fs::path expected_binary = module_library / "bin" /
                                     "imported_adopter_fixture_c.dll";
    if (fs::path(staged_source) != expected_source || !fs::is_regular_file(expected_source) ||
        !fs::is_regular_file(expected_binary) || !fs::is_regular_file(descriptor) ||
        !fs::is_regular_file(product_dll)) {
        std::cerr << "adoption did not retain source, plugin binary, or products\n";
        return 1;
    }

    auto tool = tool_registry_global().create(loaded_id);
    if (!tool || tool->port_spec(0).count != 1 || tool->port_spec(1).count != 1) {
        std::cerr << "loaded row tool was not registered\n";
        return 1;
    }
    tool->initialize(ToolInitContext{});
    const uint64_t input = tensor_f64({1.0, -2.0, 4.0});
    RawStackFrame raw{};
    raw_stack_init_frame(raw, 2048);
    nodus::tensors::AbstractTensorHandle input_handle{};
    input_handle.id = input;
    raw_stack_push_abstract_tensor(raw, input_handle);
    ToolStackContext stack{};
    stack.stack.raw = &raw;
    tool->execute_stack(stack);
    nodus::tensors::AbstractTensorHandle output{};
    std::vector<double> values;
    const bool numeric_ok = raw_stack_pop_abstract_tensor(raw, output) &&
                            tensor_values(output.id, &values) &&
                            equal(values, {5.0, -4.0, 14.0});
    if (output.id) nodus_tensor_destroy(output.id);
    if (input) nodus_tensor_destroy(input);
    raw_stack_free_mask(raw);
    tool->shutdown();
    tool.reset();
    if (!numeric_ok) {
        std::cerr << "loaded row tool returned the wrong numerical result\n";
        return 1;
    }
    if (!gp_plugin_unload(loaded_id)) {
        std::cerr << "loaded row tool could not be unloaded\n";
        return 1;
    }

    // A receipt may not name a missing or escaping artifact, and refusal must
    // happen before a wrapper is imported.
    const fs::path refused_receipt = product / "refused.receipt";
    if (!write_text(refused_receipt,
                    "NODUS ROW TOOL PRODUCT V1\n"
                    "BACKEND \"c\"\n"
                    "MANIFEST \"nodus-row-tool-product.json\"\n"
                    "ABI \"../outside.nodus_buffer_abi.txt\"\n"
                    "BINARY \"backend/fake_product_backend.dll\"\n")) return 1;
    report = {};
    std::string stale_tool = "stale";
    std::string stale_source = "stale";
    char refused_tool[16] = {};
    char refused_source[16] = {};
    std::copy(stale_tool.begin(), stale_tool.end(), refused_tool);
    std::copy(stale_source.begin(), stale_source.end(), refused_source);
    if (gp_turing_product_adopt_and_load(refused_receipt.string().c_str(), repo.string().c_str(),
                                         module_library.string().c_str(), &report,
                                         refused_tool, sizeof(refused_tool),
                                         refused_source, sizeof(refused_source)) ||
        report.wrapper_imported != 0 || refused_tool[0] != '\0' || refused_source[0] != '\0') {
        std::cerr << "escaping product path was accepted\n";
        return 1;
    }

    if (!write_text(product / "nodus-row-tool-product.json",
                    "{\n  \"schema\": \"nodus-turing-row-tool-product-v1\",\n"
                    "  \"backend\": \"llvm\",\n"
                    "  \"loadable\": true\n}\n")) return 1;
    report = {};
    if (gp_turing_product_adopt_and_load(
            (product / "nodus-row-tool-product.receipt").string().c_str(),
            repo.string().c_str(), module_library.string().c_str(), &report,
            nullptr, 0, nullptr, 0) || report.receipt_validated != 0 ||
        report.wrapper_imported != 0) {
        std::cerr << "manifest/receipt backend mismatch was accepted\n";
        return 1;
    }

    if (!write_text(product / "nodus-row-tool-product.json",
                    "{\n  \"note\": \"schema nodus-turing-row-tool-product-v1 "
                    "and loadable true are not fields\",\n"
                    "  \"schema\": \"wrong\",\n  \"backend\": \"c\",\n"
                    "  \"loadable\": false\n}\n")) return 1;
    report = {};
    if (gp_turing_product_adopt_and_load(
            (product / "nodus-row-tool-product.receipt").string().c_str(),
            repo.string().c_str(), module_library.string().c_str(), &report,
            nullptr, 0, nullptr, 0) || report.receipt_validated != 0) {
        std::cerr << "manifest substrings were mistaken for exact fields\n";
        return 1;
    }

    std::cout << "turing_product_adopter_test passed\n";
    return 0;
#endif
}
