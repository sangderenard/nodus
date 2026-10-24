#include "python_row_workflow.h"

#include "plugin_manager.h"
#include "python_row_tool.h"
#include "turing_product_adopter.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

void copy_out(const std::string& value, char* output, int32_t capacity) {
    if (!output || capacity <= 0) return;
    std::strncpy(output, value.c_str(), static_cast<size_t>(capacity) - 1);
    output[capacity - 1] = '\0';
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string json_string(const std::string& value) {
    std::ostringstream output;
    output << std::quoted(value);
    return output.str();
}

const char* shared_library_extension() {
#ifdef _WIN32
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

bool retain_source(const fs::path& source, const fs::path& destination,
                   std::string* error) {
    std::error_code ec;
    const fs::path from = fs::absolute(source).lexically_normal();
    const fs::path to = fs::absolute(destination).lexically_normal();
    if (!fs::is_regular_file(from)) {
        *error = "Python row source is missing: " + from.generic_string();
        return false;
    }
    fs::create_directories(to.parent_path(), ec);
    if (ec) {
        *error = "cannot create retained Python source directory";
        return false;
    }
    if (from != to) {
        fs::copy_file(from, to, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            *error = "cannot retain Python source: " + ec.message();
            return false;
        }
    }
    return fs::is_regular_file(to);
}

bool write_turing_request(
    const nodus::python_row::Descriptor& descriptor,
    const fs::path& retained_source,
    const fs::path& extraction_contract,
    const fs::path& product,
    const std::string& backend,
    fs::path* request_path,
    std::string* error) {
    std::error_code ec;
    fs::create_directories(product, ec);
    if (ec) {
        *error = "cannot create Turing product directory";
        return false;
    }
    *request_path = product / "nodus-python-row-request.json";
    std::ofstream out(*request_path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        *error = "cannot write Turing row request";
        return false;
    }
    out << "{\n";
    out << "  \"source_path\": " << json_string(retained_source.generic_string()) << ",\n";
    out << "  \"row_entry\": " << json_string(descriptor.callable_name) << ",\n";
    out << "  \"product_name\": " << json_string(descriptor.module_name) << ",\n";
    out << "  \"backend\": " << json_string(backend) << ",\n";
    out << "  \"output_dir\": " << json_string(product.generic_string()) << ",\n";
    out << "  \"extraction_contract\": "
        << json_string(extraction_contract.generic_string()) << ",\n";
    out << "  \"allow_uncompiled\": false,\n";
    out << "  \"ssa_composition\": {\"directory\": "
        << json_string((product / "ssa-composition").generic_string())
        << ", \"options\": {}},\n";
    out << "  \"columns\": [\n";
    size_t column = 0;
    const size_t count = descriptor.inputs.size() + descriptor.outputs.size();
    auto emit = [&](const nodus::python_row::Port& port, const char* role) {
        out << "    {\"name\": " << json_string(port.name)
            << ", \"role\": " << json_string(role)
            << ", \"dtype\": "
            << json_string(nodus::python_row::primitive_type_name(port.type))
            << ", \"shape\": []}";
        if (++column != count) out << ',';
        out << '\n';
    };
    for (const auto& port : descriptor.inputs) emit(port, "input");
    for (const auto& port : descriptor.outputs) emit(port, "output");
    out << "  ]\n}\n";
    if (!out.good()) {
        *error = "failed while writing Turing row request";
        return false;
    }
    return true;
}

} // namespace

extern "C" int gp_python_row_deploy_and_load(
    const char* descriptor_path,
    const char* target,
    const char* python_executable,
    const char* turing_root,
    const char* extraction_contract,
    const char* product_dir,
    const char* nodus_root,
    const char* module_library_root,
    GP_PythonRowDeploymentReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_retained_python,
    int32_t out_retained_python_cap,
    char* out_staged_wrapper,
    int32_t out_staged_wrapper_cap) {
    GP_PythonRowDeploymentReport local{};
    if (report) *report = local;
    copy_out("", out_tool_id, out_tool_id_cap);
    copy_out("", out_retained_python, out_retained_python_cap);
    copy_out("", out_staged_wrapper, out_staged_wrapper_cap);
    if (!descriptor_path || !descriptor_path[0] || !target || !target[0] ||
        !nodus_root || !nodus_root[0] ||
        !module_library_root || !module_library_root[0]) return 0;

    const std::string selected = lowercase(target);
    if (selected != "cpython" && selected != "c" &&
        selected != "llvm" && selected != "fortran") {
        std::cerr << "python_row_workflow: unsupported deployment target\n";
        return 0;
    }

    nodus::python_row::Descriptor descriptor;
    std::string error;
    if (!nodus::python_row::read_descriptor(descriptor_path, &descriptor, &error)) {
        std::cerr << "python_row_workflow: " << error << '\n';
        return 0;
    }
    local.abi_loaded = 1;

    const fs::path module_root = fs::absolute(module_library_root);
    const fs::path tool_source_root = module_root / "source" / "tools";
    const fs::path retained_python = tool_source_root /
                                     (descriptor.module_name + ".py");
    if (!retain_source(descriptor.source_path, retained_python, &error)) {
        std::cerr << "python_row_workflow: " << error << '\n';
        if (report) *report = local;
        return 0;
    }
    descriptor.source_path = fs::absolute(retained_python).generic_string();
    const fs::path retained_abi = tool_source_root /
                                  (descriptor.module_name + ".pyabi");
    if (!nodus::python_row::write_descriptor(
            descriptor, retained_abi.generic_string(), &error)) {
        std::cerr << "python_row_workflow: " << error << '\n';
        if (report) *report = local;
        return 0;
    }
    local.source_retained = 1;
    copy_out(retained_python.generic_string(), out_retained_python,
             out_retained_python_cap);

    if (selected == "cpython") {
        const fs::path wrapper = tool_source_root /
                                 (descriptor.module_name + "_cpython.cpp");
        if (!nodus::python_row::write_plugin_source(
                descriptor, wrapper.generic_string(), &error)) {
            std::cerr << "python_row_workflow: " << error << '\n';
            if (report) *report = local;
            return 0;
        }
        local.wrapper_staged = 1;
        std::error_code ec;
        fs::create_directories(module_root / "bin", ec);
        char loaded_id[1024] = {};
        if (ec || !gp_plugin_build_module_and_load(
                wrapper.string().c_str(), nodus_root,
                (module_root / "bin").string().c_str(), nullptr,
                loaded_id, static_cast<int>(sizeof(loaded_id)))) {
            std::cerr << "python_row_workflow: CPython wrapper build/load failed\n";
            if (report) *report = local;
            return 0;
        }
        const fs::path binary = module_root / "bin" /
                                (wrapper.stem().string() + shared_library_extension());
        if (!fs::is_regular_file(binary)) {
            gp_plugin_unload(loaded_id);
            std::cerr << "python_row_workflow: CPython wrapper binary was not retained\n";
            if (report) *report = local;
            return 0;
        }
        local.plugin_loaded = 1;
        copy_out(loaded_id, out_tool_id, out_tool_id_cap);
        copy_out(wrapper.generic_string(), out_staged_wrapper,
                 out_staged_wrapper_cap);
        if (report) *report = local;
        return 1;
    }

    if (!python_executable || !python_executable[0] ||
        !turing_root || !turing_root[0] ||
        !extraction_contract || !extraction_contract[0] ||
        !product_dir || !product_dir[0]) {
        std::cerr << "python_row_workflow: native target requires compiler paths\n";
        if (report) *report = local;
        return 0;
    }
    const fs::path contract = fs::absolute(extraction_contract);
    const fs::path product = fs::absolute(product_dir);
    if (!fs::is_regular_file(contract)) {
        std::cerr << "python_row_workflow: extraction contract is missing\n";
        if (report) *report = local;
        return 0;
    }
    fs::path request;
    if (!write_turing_request(
            descriptor, retained_python, contract, product, selected,
            &request, &error)) {
        std::cerr << "python_row_workflow: " << error << '\n';
        if (report) *report = local;
        return 0;
    }
    local.request_written = 1;
    local.compiler_invoked = 1;
    GP_TuringProductAdoptionReport adoption{};
    char loaded_id[1024] = {};
    char staged_wrapper[4096] = {};
    if (!gp_turing_row_tool_compile_and_load(
            python_executable, request.string().c_str(), turing_root,
            product.string().c_str(), nodus_root, module_library_root,
            &adoption, loaded_id, static_cast<int>(sizeof(loaded_id)),
            staged_wrapper, static_cast<int>(sizeof(staged_wrapper)))) {
        if (report) *report = local;
        return 0;
    }
    local.wrapper_staged = adoption.wrapper_staged;
    local.plugin_loaded = adoption.plugin_loaded;
    copy_out(loaded_id, out_tool_id, out_tool_id_cap);
    copy_out(staged_wrapper, out_staged_wrapper, out_staged_wrapper_cap);
    if (report) *report = local;
    return local.plugin_loaded;
}
