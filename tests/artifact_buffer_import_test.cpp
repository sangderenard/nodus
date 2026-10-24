// Standalone acceptance test for NODUS BUFFER ABI V1/V2 import.
// This file is intentionally self-contained so it can be added to the normal
// test manifest independently; it does not require a Turing checkout.
#include "artifact_importer.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

bool write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
    return out.good();
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool run(const std::string& command) {
    return std::system(command.c_str()) == 0;
}

fs::path find_core_lib(const fs::path& build) {
    for (const auto& path : {build / "Release" / "nodus_tensor_core.lib",
                             build / "Debug" / "nodus_tensor_core.lib",
                             build / "nodus_tensor_core.lib"}) {
        if (fs::exists(path)) return path;
    }
    return {};
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

bool tensor_values(uint64_t handle, std::vector<double>* out) {
    NodusTensorDesc desc{};
    if (nodus_tensor_describe(handle, &desc) != NODUS_OK ||
        desc.dtype != NODUS_DTYPE_F64 || desc.rank != 1) return false;
    out->resize(static_cast<size_t>(desc.shape[0]));
    return nodus_tensor_read(handle, 0, out->data(), out->size() * sizeof(double)) ==
           static_cast<int64_t>(out->size() * sizeof(double));
}

bool close(const std::vector<double>& got, const std::vector<double>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - want[i]) > 1e-12) return false;
    }
    return true;
}

bool execute_case(ITool& tool,
                  const std::vector<double>& x_values,
                  const std::vector<double>& y_values,
                  const std::vector<double>& want_sum,
                  const std::vector<double>& want_difference) {
    uint64_t x = tensor_f64(x_values);
    uint64_t y = tensor_f64(y_values);
    RawStackFrame raw{};
    bool ok = x && y && raw_stack_init_frame(raw, 4096);
    nodus::tensors::AbstractTensorHandle handle{};
    handle.id = x; ok = ok && raw_stack_push_abstract_tensor(raw, handle);
    handle.id = y; ok = ok && raw_stack_push_abstract_tensor(raw, handle);
    ToolStackContext stack{};
    stack.stack.raw = &raw;
    if (ok) tool.execute_stack(stack);

    nodus::tensors::AbstractTensorHandle difference{}, sum{};
    ok = ok && raw_stack_pop_abstract_tensor(raw, difference) &&
         raw_stack_pop_abstract_tensor(raw, sum);
    std::vector<double> got_sum, got_difference;
    ok = ok && tensor_values(sum.id, &got_sum) &&
         tensor_values(difference.id, &got_difference) &&
         close(got_sum, want_sum) && close(got_difference, want_difference);
    if (sum.id) nodus_tensor_destroy(sum.id);
    if (difference.id) nodus_tensor_destroy(difference.id);
    if (x) nodus_tensor_destroy(x);
    if (y) nodus_tensor_destroy(y);
    raw_stack_free_mask(raw);
    if (raw.backend) gp_mem_backend_release(raw.backend);
    return ok;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cout << "artifact_buffer_import_test skipped: fixture DLL path is Windows-only\n";
    return 0;
#else
    const fs::path repo = fs::path(NODUS_REPO_ROOT);
    const fs::path build = fs::path(NODUS_BUILD_DIR);
    const fs::path root = repo / "module_library" / "test_build" / "artifact_buffer_import_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    // The fake backend observes workspace initialization as well as buffer and
    // extent order. Nonzero workspace produces deliberately wrong results.
    const fs::path backend_src = root / "fake_backend.cpp";
    if (!write_text(backend_src, R"CPP(
#include <cstdint>
extern "C" __declspec(dllexport) void buffer_entry(void** buffers, int64_t* extents) {
    auto* x = static_cast<double*>(buffers[0]);
    auto* y = static_cast<double*>(buffers[1]);
    auto* workspace = static_cast<double*>(buffers[2]);
    auto* sum = static_cast<double*>(buffers[3]);
    auto* difference = static_cast<double*>(buffers[4]);
    const int64_t n = extents[0];
    for (int64_t i = 0; i < n; ++i) {
        const double dirty = workspace[i];
        workspace[i] = x[i] + y[i];
        sum[i] = workspace[i] + dirty * 1000.0;
        difference[i] = x[i] - y[i];
    }
}
)CPP")) return 1;

    const fs::path backend_project = root / "backend_project";
    const fs::path backend_build = backend_project / "build";
    std::ostringstream backend_cmake;
    backend_cmake << "cmake_minimum_required(VERSION 3.15)\n"
                     "project(fake_buffer_backend LANGUAGES CXX)\n"
                     "add_library(fake_buffer_backend SHARED \""
                  << backend_src.generic_string() << "\")\n"
                     "set_target_properties(fake_buffer_backend PROPERTIES CXX_STANDARD 17 "
                     "RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\" "
                     "RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
    if (!write_text(backend_project / "CMakeLists.txt", backend_cmake.str()) ||
        !run("cmake -S \"" + backend_project.string() + "\" -B \"" + backend_build.string() + "\"") ||
        !run("cmake --build \"" + backend_build.string() + "\" --config Release")) {
        std::cerr << "failed to build fake buffer backend\n";
        return 1;
    }
    fs::path backend_dll = backend_build / "out" / "fake_buffer_backend.dll";
    if (!fs::exists(backend_dll)) {
        std::cerr << "fake backend DLL not found\n";
        return 1;
    }

    const fs::path descriptor = root / "fixture.buffer_abi";
    const std::string valid =
        "NODUS BUFFER ABI V2\n"
        "BACKEND \"c\"\n"
        "EXTENT_DTYPE \"i64\"\n"
        "TOOL \"imported.buffer_fixture.c\" \"Buffer Fixture C\"\n"
        "ENTRY \"buffer_entry\"\n"
        "VALUE INPUT \"%x\" \"f64\" 1 ?\n"
        "VALUE INPUT \"%y\" \"f64\" 1 ?\n"
        "VALUE WORKSPACE \"%tmp\" \"f64\" 1 ?\n"
        "VALUE OUTPUT \"%sum\" \"f64\" 1 ?\n"
        "VALUE OUTPUT \"%difference\" \"f64\" 1 ?\n"
        "BUFFER \"%x\"\nBUFFER \"%y\"\nBUFFER \"%tmp\"\n"
        "BUFFER \"%sum\"\nBUFFER \"%difference\"\n"
        "EXTENT \"%x\" 0\n"
        "SHAPE \"%y\" 0 FROM \"%x\" 0\n"
        "SHAPE \"%tmp\" 0 FROM \"%x\" 0\n"
        "SHAPE \"%sum\" 0 FROM \"%x\" 0\n"
        "SHAPE \"%difference\" 0 FROM \"%x\" 0\n";
    if (!write_text(descriptor, valid)) return 1;

    GP_ArtifactImportReport report{};
    char wrapper_buf[1024] = {};
    char package_buf[1024] = {};
    if (!gp_artifact_import_buffer_abi(descriptor.string().c_str(), backend_dll.string().c_str(),
                                       root.string().c_str(), &report,
                                       wrapper_buf, sizeof(wrapper_buf),
                                       package_buf, sizeof(package_buf)) ||
        report.wrappers_written != 1 || report.shortfalls != 0) {
        std::cerr << "valid buffer descriptor was refused\n";
        return 1;
    }
    const fs::path wrapper_src(wrapper_buf);
    if (wrapper_src.parent_path() != root / "source" / "tools" ||
        wrapper_src.filename() != "imported_buffer_fixture_c.cpp" ||
        !fs::exists(package_buf)) {
        std::cerr << "wrapper was not staged in the direct row-tool source directory\n";
        return 1;
    }
    const std::string generated = read_text(wrapper_src);
    if (generated.find("using Extent = int64_t") == std::string::npos ||
        generated.find("NODUS_PLUGIN_FACTORY_NAME") == std::string::npos ||
        generated.find("kWorkspaces") == std::string::npos) {
        std::cerr << "generated C wrapper lost ABI or compositing declarations\n";
        return 1;
    }

    fs::path core_lib = find_core_lib(build);
    if (core_lib.empty()) {
        std::cerr << "nodus_tensor_core import library not found\n";
        return 1;
    }
    const fs::path wrapper_project = root / "wrapper_project";
    const fs::path wrapper_build = wrapper_project / "build";
    std::ostringstream wrapper_cmake;
    wrapper_cmake << "cmake_minimum_required(VERSION 3.15)\n"
                     "project(imported_buffer_fixture LANGUAGES CXX)\n"
                     "add_library(imported_buffer_fixture SHARED \""
                  << wrapper_src.generic_string() << "\")\n"
                  << "target_include_directories(imported_buffer_fixture PRIVATE \""
                  << repo.generic_string() << "\" \"" << (repo / "include").generic_string() << "\")\n"
                  << "target_link_libraries(imported_buffer_fixture PRIVATE \"" << core_lib.generic_string() << "\")\n"
                     "set_target_properties(imported_buffer_fixture PROPERTIES CXX_STANDARD 20 "
                     "RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\" "
                     "RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
    if (!write_text(wrapper_project / "CMakeLists.txt", wrapper_cmake.str()) ||
        !run("cmake -S \"" + wrapper_project.string() + "\" -B \"" + wrapper_build.string() + "\"") ||
        !run("cmake --build \"" + wrapper_build.string() + "\" --config Release")) {
        std::cerr << "failed to build generated wrapper\n";
        return 1;
    }
    fs::path wrapper_dll = wrapper_build / "out" / "imported_buffer_fixture.dll";
    HMODULE module = LoadLibraryA(wrapper_dll.string().c_str());
    if (!module) {
        std::cerr << "failed to load generated wrapper: " << GetLastError() << "\n";
        return 1;
    }
    using CreateFn = ITool* (*)();
    using DestroyFn = void (*)(ITool*);
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(module, "create_tool"));
    auto destroy = reinterpret_cast<DestroyFn>(GetProcAddress(module, "destroy_tool"));
    ITool* tool = create ? create() : nullptr;
    if (!tool || !destroy || tool->port_spec(0).count != 2 || tool->port_spec(1).count != 2) {
        std::cerr << "generated wrapper exports or multi-output ports are wrong\n";
        return 1;
    }
    tool->initialize(ToolInitContext{});

    bool ok = execute_case(
        *tool, {1.0, 2.0, 3.0, 4.0}, {0.5, -1.0, 2.0, 8.0},
        {1.5, 1.0, 5.0, 12.0}, {0.5, 3.0, 1.0, -4.0});
    ok = ok && execute_case(
        *tool, {10.0, -2.0, 0.25, 8.0, 3.0, -7.0},
        {1.0, 4.0, 0.75, -3.0, 2.0, 5.0},
        {11.0, 2.0, 1.0, 5.0, 5.0, -2.0},
        {9.0, -6.0, -0.5, 11.0, 1.0, -12.0});
    tool->shutdown();
    destroy(tool);
    FreeLibrary(module);
    if (!ok) {
        std::cerr << "buffer wrapper numerical result/order/workspace check failed\n";
        return 1;
    }

    // LLVM uses the actual i32 extent ABI, independently of the C wrapper.
    const fs::path llvm_descriptor = root / "fixture_llvm.buffer_abi";
    std::string llvm = valid;
    llvm.replace(llvm.find("BACKEND \"c\""), std::string("BACKEND \"c\"").size(), "BACKEND \"llvm\"");
    llvm.replace(llvm.find("EXTENT_DTYPE \"i64\""), std::string("EXTENT_DTYPE \"i64\"").size(),
                 "EXTENT_DTYPE \"i32\"");
    llvm.replace(llvm.find("imported.buffer_fixture.c"), std::string("imported.buffer_fixture.c").size(),
                 "imported.buffer_fixture.llvm");
    if (!write_text(llvm_descriptor, llvm) ||
        !gp_artifact_import_buffer_abi(llvm_descriptor.string().c_str(), backend_dll.string().c_str(),
                                       root.string().c_str(), &report,
                                       wrapper_buf, sizeof(wrapper_buf), nullptr, 0) ||
        read_text(wrapper_buf).find("using Extent = int32_t") == std::string::npos) {
        std::cerr << "LLVM wrapper did not select i32 extents\n";
        return 1;
    }

    auto expect_refusal = [&](const char* name, std::string text) {
        const fs::path path = root / name;
        write_text(path, text);
        GP_ArtifactImportReport refused{};
        return !gp_artifact_import_buffer_abi(path.string().c_str(), backend_dll.string().c_str(),
                                               root.string().c_str(), &refused,
                                               nullptr, 0, nullptr, 0) &&
               refused.shortfalls == 1 && refused.wrappers_written == 0;
    };
    std::string invalid_extent = valid;
    invalid_extent.replace(invalid_extent.find("1 ?\n"), 3, "1 -1\n");
    std::string v1_dynamic = valid;
    v1_dynamic.replace(v1_dynamic.find("NODUS BUFFER ABI V2"),
                       std::string("NODUS BUFFER ABI V2").size(),
                       "NODUS BUFFER ABI V1");
    std::string missing_shape_source = valid;
    missing_shape_source.erase(
        missing_shape_source.find("SHAPE \"%sum\" 0 FROM \"%x\" 0\n"),
        std::string("SHAPE \"%sum\" 0 FROM \"%x\" 0\n").size());
    std::string bad_dtype = valid;
    bad_dtype.replace(bad_dtype.find("\"f64\""), 5, "\"complex64\"");
    std::string missing_buffer = valid;
    missing_buffer.erase(missing_buffer.find("BUFFER \"%tmp\"\n"), std::string("BUFFER \"%tmp\"\n").size());
    std::string wrong_extent_dtype = valid;
    wrong_extent_dtype.replace(wrong_extent_dtype.find("EXTENT_DTYPE \"i64\""),
                               std::string("EXTENT_DTYPE \"i64\"").size(),
                               "EXTENT_DTYPE \"i32\"");
    if (!expect_refusal("invalid_extent.buffer_abi", invalid_extent) ||
        !expect_refusal("v1_dynamic.buffer_abi", v1_dynamic) ||
        !expect_refusal("missing_shape_source.buffer_abi", missing_shape_source) ||
        !expect_refusal("dtype.buffer_abi", bad_dtype) ||
        !expect_refusal("missing_buffer.buffer_abi", missing_buffer) ||
        !expect_refusal("extent_dtype.buffer_abi", wrong_extent_dtype)) {
        std::cerr << "one or more required refusal cases were admitted\n";
        return 1;
    }

    std::cout << "[BUFFER IMPORT] PASS: dynamic C multi-output/workspace, LLVM i32 extents, refusals\n";
    return 0;
#endif
}
