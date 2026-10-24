#include "turing_product_adopter.h"

#include "artifact_importer.h"
#include "plugin_manager.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

struct ProductReceipt {
    std::string backend;
    std::string manifest;
    std::string abi;
    std::string binary;
};

void copy_out(const std::string& value, char* out, int32_t cap) {
    if (!out || cap <= 0) return;
    std::strncpy(out, value.c_str(), static_cast<size_t>(cap) - 1);
    out[cap - 1] = '\0';
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string sanitize(const std::string& value) {
    std::string out;
    for (char c : value) {
        out.push_back(std::isalnum(static_cast<unsigned char>(c)) ? c : '_');
    }
    if (out.empty() || std::isdigit(static_cast<unsigned char>(out.front()))) out.insert(out.begin(), '_');
    return out;
}

std::string qualified_stem(const std::string& tool_id, const std::string& backend) {
    std::string stem = sanitize(tool_id);
    const std::string suffix = "_" + sanitize(backend);
    if (stem.size() < suffix.size() ||
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) != 0) {
        stem += suffix;
    }
    return stem;
}

bool parse_receipt(const fs::path& path, ProductReceipt* out, std::string* error) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        *error = "cannot open product receipt: " + path.generic_string();
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        *error = "empty product receipt";
        return false;
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "NODUS ROW TOOL PRODUCT V1") {
        *error = "unsupported product receipt header";
        return false;
    }
    ProductReceipt receipt;
    bool saw_backend = false, saw_manifest = false, saw_abi = false, saw_binary = false;
    size_t line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        std::istringstream row(line);
        std::string tag;
        std::string value;
        row >> tag >> std::quoted(value);
        std::string trailing;
        if (!row || (row >> trailing)) {
            *error = "malformed receipt record at line " + std::to_string(line_no);
            return false;
        }
        bool* seen = nullptr;
        std::string* target = nullptr;
        if (tag == "BACKEND") { seen = &saw_backend; target = &receipt.backend; }
        else if (tag == "MANIFEST") { seen = &saw_manifest; target = &receipt.manifest; }
        else if (tag == "ABI") { seen = &saw_abi; target = &receipt.abi; }
        else if (tag == "BINARY") { seen = &saw_binary; target = &receipt.binary; }
        else {
            *error = "unknown receipt record '" + tag + "'";
            return false;
        }
        if (*seen || value.empty()) {
            *error = "duplicate or empty receipt record '" + tag + "'";
            return false;
        }
        *seen = true;
        *target = std::move(value);
    }
    if (!saw_backend || !saw_manifest || !saw_abi || !saw_binary) {
        *error = "receipt requires BACKEND, MANIFEST, ABI, and BINARY";
        return false;
    }
    if (receipt.backend != "c" && receipt.backend != "llvm" && receipt.backend != "fortran") {
        *error = "unsupported receipt backend '" + receipt.backend + "'";
        return false;
    }
    *out = std::move(receipt);
    return true;
}

bool resolve_product_file(const fs::path& product_root,
                          const std::string& relative,
                          fs::path* out,
                          std::string* error) {
    fs::path rel(relative);
    if (rel.is_absolute()) {
        *error = "receipt paths must be product-root relative: " + relative;
        return false;
    }
    for (const auto& component : rel) {
        if (component == "..") {
            *error = "receipt path escapes the product root: " + relative;
            return false;
        }
    }
    std::error_code ec;
    fs::path candidate = fs::weakly_canonical(product_root / rel, ec);
    if (ec || !fs::is_regular_file(candidate)) {
        *error = "receipt product is missing: " + relative;
        return false;
    }
    fs::path root = fs::weakly_canonical(product_root, ec);
    fs::path back = fs::relative(candidate, root, ec);
    if (ec || back.empty() || back.is_absolute() || *back.begin() == "..") {
        *error = "receipt product resolves outside the product root: " + relative;
        return false;
    }
    *out = std::move(candidate);
    return true;
}

bool top_level_json_value(const std::string& text, const std::string& key, size_t* value_pos) {
    int depth = 0;
    for (size_t pos = 0; pos < text.size();) {
        if (text[pos] == '{' || text[pos] == '[') { ++depth; ++pos; continue; }
        if (text[pos] == '}' || text[pos] == ']') { --depth; ++pos; continue; }
        if (text[pos] != '"') { ++pos; continue; }
        ++pos;
        bool escaped = false;
        std::string token;
        for (; pos < text.size(); ++pos) {
            const char c = text[pos];
            if (escaped) { token.push_back(c); escaped = false; continue; }
            if (c == '\\') { escaped = true; continue; }
            if (c == '"') break;
            token.push_back(c);
        }
        if (pos == text.size()) return false;
        size_t after = text.find_first_not_of(" \t\r\n", pos + 1);
        if (depth == 1 && token == key && after != std::string::npos && text[after] == ':') {
            after = text.find_first_not_of(" \t\r\n", after + 1);
            if (after == std::string::npos) return false;
            *value_pos = after;
            return true;
        }
        ++pos;
    }
    return false;
}

bool json_string_field(const std::string& text, const std::string& key, std::string* out) {
    size_t pos = 0;
    if (!top_level_json_value(text, key, &pos) || text[pos] != '"') return false;
    std::string value;
    bool escaped = false;
    for (++pos; pos < text.size(); ++pos) {
        const char c = text[pos];
        if (escaped) { value.push_back(c); escaped = false; continue; }
        if (c == '\\') { escaped = true; continue; }
        if (c == '"') { *out = std::move(value); return true; }
        value.push_back(c);
    }
    return false;
}

bool json_true_field(const std::string& text, const std::string& key) {
    size_t pos = 0;
    if (!top_level_json_value(text, key, &pos) || text.compare(pos, 4, "true") != 0) return false;
    const size_t end = pos + 4;
    return end == text.size() || std::isspace(static_cast<unsigned char>(text[end])) ||
           text[end] == ',' || text[end] == '}';
}

bool validate_product_manifest(const fs::path& path,
                               const std::string& receipt_backend,
                               std::string* error) {
    std::ifstream in(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::string schema;
    std::string backend;
    if (!json_string_field(text, "schema", &schema) ||
        schema != "nodus-turing-row-tool-product-v1") {
        *error = "product manifest has the wrong schema";
        return false;
    }
    if (!json_true_field(text, "loadable")) {
        *error = "product manifest is not loadable";
        return false;
    }
    if (!json_string_field(text, "backend", &backend) || backend != receipt_backend) {
        *error = "product manifest backend does not match receipt";
        return false;
    }
    return true;
}

bool read_package_tool(const fs::path& package_path,
                       std::string* tool_id,
                       fs::path* source_path,
                       std::string* error) {
    std::ifstream in(package_path);
    std::string line;
    if (!std::getline(in, line) || line.rfind("NODUSPKG", 0) != 0) {
        *error = "importer emitted an invalid package manifest";
        return false;
    }
    while (std::getline(in, line)) {
        std::istringstream row(line);
        std::string tag;
        row >> tag;
        if (tag != "TOOL") continue;
        std::string name;
        std::string source;
        row >> *tool_id >> std::quoted(name) >> std::quoted(source);
        if (!row || tool_id->empty() || source.empty()) {
            *error = "importer package has a malformed TOOL record";
            return false;
        }
        *source_path = fs::path(source);
        return true;
    }
    *error = "importer package has no TOOL record";
    return false;
}

std::string shared_library_extension() {
#ifdef _WIN32
    return ".dll";
#elif defined(__APPLE__)
    return ".dylib";
#else
    return ".so";
#endif
}

#ifdef _WIN32
std::wstring quote_windows_argument(const std::wstring& value) {
    if (value.find_first_of(L" \t\"") == std::wstring::npos) return value;
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        if (c == L'\"') {
            out.append(slashes * 2 + 1, L'\\');
            out.push_back(c);
        } else {
            out.append(slashes, L'\\');
            out.push_back(c);
        }
        slashes = 0;
    }
    out.append(slashes * 2, L'\\');
    out.push_back(L'\"');
    return out;
}
#endif

bool run_compiler_driver(const fs::path& python,
                         const fs::path& driver,
                         const fs::path& request,
                         const fs::path& turing_root,
                         const fs::path& product_dir,
                         const fs::path& nodus_root,
                         std::string* error) {
    const std::vector<fs::path> args = {
        python, driver, "--request", request, "--output-dir", product_dir,
        "--turing-root", turing_root,
    };
#ifdef _WIN32
    std::wstring command;
    for (const auto& arg : args) {
        if (!command.empty()) command.push_back(L' ');
        command += quote_windows_argument(arg.wstring());
    }
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    const std::wstring application = python.wstring();
    const std::wstring cwd = nodus_root.wstring();
    if (!CreateProcessW(application.c_str(), mutable_command.data(), nullptr, nullptr, FALSE,
                        0, nullptr, cwd.c_str(), &startup, &process)) {
        *error = "failed to launch compiler driver: " + std::to_string(GetLastError());
        return false;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD exit_code = 1;
    GetExitCodeProcess(process.hProcess, &exit_code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    if (exit_code != 0) {
        *error = "compiler driver failed with exit code " + std::to_string(exit_code);
        return false;
    }
    return true;
#else
    std::vector<std::string> storage;
    std::vector<char*> argv;
    for (const auto& arg : args) storage.push_back(arg.string());
    for (auto& arg : storage) argv.push_back(arg.data());
    argv.push_back(nullptr);
    const pid_t child = fork();
    if (child < 0) { *error = "fork failed"; return false; }
    if (child == 0) {
        chdir(nodus_root.string().c_str());
        execv(storage.front().c_str(), argv.data());
        _exit(127);
    }
    int status = 0;
    if (waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        *error = "compiler driver failed";
        return false;
    }
    return true;
#endif
}

} // namespace

extern "C" int gp_turing_product_adopt_and_load(
    const char* receipt_path,
    const char* nodus_root,
    const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_staged_source,
    int32_t out_staged_source_cap) {
    GP_TuringProductAdoptionReport local{};
    if (report) *report = local;
    copy_out("", out_tool_id, out_tool_id_cap);
    copy_out("", out_staged_source, out_staged_source_cap);
    if (!receipt_path || !receipt_path[0] || !nodus_root || !nodus_root[0] ||
        !module_library_root || !module_library_root[0]) return 0;
    const fs::path nodus = fs::absolute(nodus_root);
    if (!fs::is_regular_file(nodus / "include" / "tool_api.h")) {
        std::cerr << "turing_product_adopter: Nodus root has no include/tool_api.h\n";
        return 0;
    }
    std::string error;
    ProductReceipt receipt;
    const fs::path receipt_file = fs::absolute(receipt_path);
    if (!parse_receipt(receipt_file, &receipt, &error)) {
        std::cerr << "turing_product_adopter: " << error << "\n";
        if (report) *report = local;
        return 0;
    }
    const fs::path product_root = receipt_file.parent_path();
    fs::path manifest, abi, binary;
    if (!resolve_product_file(product_root, receipt.manifest, &manifest, &error) ||
        !resolve_product_file(product_root, receipt.abi, &abi, &error) ||
        !resolve_product_file(product_root, receipt.binary, &binary, &error) ||
        !validate_product_manifest(manifest, receipt.backend, &error)) {
        std::cerr << "turing_product_adopter: " << error << "\n";
        if (report) *report = local;
        return 0;
    }
    const std::string abi_name = lowercase(abi.filename().string());
    if ((receipt.backend == "fortran" &&
         (abi_name.size() < 9 || abi_name.substr(abi_name.size() - 9) != ".api.yaml")) ||
        (receipt.backend != "fortran" &&
         (abi_name.size() < 21 || abi_name.substr(abi_name.size() - 21) != ".nodus_buffer_abi.txt"))) {
        std::cerr << "turing_product_adopter: ABI descriptor extension does not match backend\n";
        if (report) *report = local;
        return 0;
    }
    local.receipt_validated = 1;

    const fs::path module_root = fs::absolute(module_library_root);
    std::error_code ec;
    fs::create_directories(module_root / "source" / "tools", ec);
    fs::create_directories(module_root / "bin", ec);
    if (ec) {
        std::cerr << "turing_product_adopter: cannot create module-library directories\n";
        if (report) *report = local;
        return 0;
    }

    GP_ArtifactImportReport import_report{};
    char wrapper_buffer[4096] = {};
    char package_buffer[4096] = {};
    int imported = 0;
    if (receipt.backend == "fortran") {
        imported = gp_artifact_import(abi.string().c_str(), binary.string().c_str(),
                                      module_root.string().c_str(), &import_report,
                                      package_buffer, sizeof(package_buffer));
    } else {
        imported = gp_artifact_import_buffer_abi(
            abi.string().c_str(), binary.string().c_str(), module_root.string().c_str(),
            &import_report, wrapper_buffer, sizeof(wrapper_buffer),
            package_buffer, sizeof(package_buffer));
    }
    if (!imported || import_report.wrappers_written != 1) {
        std::cerr << "turing_product_adopter: artifact importer refused product\n";
        if (report) *report = local;
        return 0;
    }
    local.wrapper_imported = 1;

    std::string package_tool_id;
    fs::path imported_source;
    if (!read_package_tool(package_buffer, &package_tool_id, &imported_source, &error)) {
        std::cerr << "turing_product_adopter: " << error << "\n";
        if (report) *report = local;
        return 0;
    }
    if (imported_source.is_relative()) imported_source = module_root / imported_source;
    if (!fs::is_regular_file(imported_source)) {
        std::cerr << "turing_product_adopter: imported wrapper source is missing\n";
        if (report) *report = local;
        return 0;
    }
    const std::string stem = qualified_stem(package_tool_id, receipt.backend);
    const fs::path staged_source = module_root / "source" / "tools" / (stem + ".cpp");
    if (fs::absolute(imported_source).lexically_normal() !=
        fs::absolute(staged_source).lexically_normal()) {
        fs::copy_file(imported_source, staged_source, fs::copy_options::overwrite_existing, ec);
        if (ec) {
            std::cerr << "turing_product_adopter: failed to stage direct wrapper source\n";
            if (report) *report = local;
            return 0;
        }
    }
    local.wrapper_staged = 1;

    char loaded_id[1024] = {};
    if (!gp_plugin_build_module_and_load(
            staged_source.string().c_str(), nodus.string().c_str(),
            (module_root / "bin").string().c_str(), nullptr,
            loaded_id, sizeof(loaded_id))) {
        std::cerr << "turing_product_adopter: wrapper build/load failed\n";
        if (report) *report = local;
        return 0;
    }
    const fs::path retained_binary = module_root / "bin" / (staged_source.stem().string() + shared_library_extension());
    if (!fs::is_regular_file(retained_binary)) {
        std::cerr << "turing_product_adopter: plugin loaded but retained same-stem binary is missing\n";
        gp_plugin_unload(loaded_id);
        if (report) *report = local;
        return 0;
    }
    local.plugin_loaded = 1;
    copy_out(loaded_id, out_tool_id, out_tool_id_cap);
    copy_out(staged_source.generic_string(), out_staged_source, out_staged_source_cap);
    if (report) *report = local;
    return 1;
}

extern "C" int gp_turing_row_tool_compile_and_load(
    const char* python_executable,
    const char* request_json,
    const char* turing_root,
    const char* product_dir,
    const char* nodus_root,
    const char* module_library_root,
    GP_TuringProductAdoptionReport* report,
    char* out_tool_id,
    int32_t out_tool_id_cap,
    char* out_staged_source,
    int32_t out_staged_source_cap) {
    if (report) *report = GP_TuringProductAdoptionReport{};
    copy_out("", out_tool_id, out_tool_id_cap);
    copy_out("", out_staged_source, out_staged_source_cap);
    if (!python_executable || !python_executable[0] || !request_json || !request_json[0] ||
        !turing_root || !turing_root[0] || !product_dir || !product_dir[0] ||
        !nodus_root || !nodus_root[0] || !module_library_root || !module_library_root[0]) return 0;
    const fs::path python = fs::absolute(python_executable);
    const fs::path request = fs::absolute(request_json);
    const fs::path turing = fs::absolute(turing_root);
    const fs::path product = fs::absolute(product_dir);
    const fs::path nodus = fs::absolute(nodus_root);
    const fs::path driver = nodus / "tools" / "turing_row_tool_compiler.py";
    if (!fs::is_regular_file(python) || !fs::is_regular_file(request) ||
        !fs::is_directory(turing) || !fs::is_directory(nodus) ||
        !fs::is_regular_file(driver)) {
        std::cerr << "turing_product_adopter: compile request names a missing executable, request, root, or driver\n";
        return 0;
    }
    std::error_code ec;
    fs::create_directories(product, ec);
    if (ec) return 0;
    std::string error;
    if (!run_compiler_driver(python, driver, request, turing, product, nodus, &error)) {
        std::cerr << "turing_product_adopter: " << error << "\n";
        return 0;
    }
    const fs::path receipt = product / "nodus-row-tool-product.receipt";
    if (!fs::is_regular_file(receipt)) {
        std::cerr << "turing_product_adopter: compiler completed without a product receipt\n";
        return 0;
    }
    return gp_turing_product_adopt_and_load(
        receipt.string().c_str(), nodus.string().c_str(), module_library_root,
        report, out_tool_id, out_tool_id_cap,
        out_staged_source, out_staged_source_cap);
}
