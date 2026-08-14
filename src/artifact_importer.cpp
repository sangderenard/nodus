#include "artifact_importer.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// turing-compiled-program-api-v1 subset reader.
//
// The descriptor is emitted by yaml.safe_dump(sort_keys=False) from
// turing/src/compiler/compiled_program_api.py. This reads exactly the shape
// that serializer produces (2-space block indentation, "- " list items,
// "key: value" scalars) for the fields wrapper generation needs; it is not a
// general YAML parser and refuses documents that don't carry the v1 schema
// line, so a format drift fails loudly here instead of mis-parsing.
// ---------------------------------------------------------------------------

struct ApiParameter {
    std::string name;
    std::string role;     // "extent" | "input" | "output"
    std::string dtype;
    std::string c_type;
    std::string passing;  // "value" | "reference"
    std::string source_name;
    std::vector<int64_t> shape;
};

struct ApiEntryPoint {
    std::string name;
    std::string symbol;
    std::string kind;     // "control" | "numerical" | "region"
    std::vector<ApiParameter> parameters;
};

struct ApiRuntimeDependency {
    std::string name;
    std::string path;
};

struct ApiDoc {
    std::string module;
    std::string language;
    std::string entry;
    std::vector<ApiEntryPoint> entry_points;
    // metadata.runtime_dependencies -- packed by the producer at compile
    // time (the toolchain knows its own support DLLs; error register E15).
    std::vector<ApiRuntimeDependency> runtime_dependencies;
};

size_t indent_of(const std::string& line) {
    size_t i = 0;
    while (i < line.size() && line[i] == ' ') ++i;
    return i;
}

std::string scalar_value(const std::string& line) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return {};
    std::string v = line.substr(pos + 1);
    while (!v.empty() && (v.front() == ' ')) v.erase(v.begin());
    while (!v.empty() && (v.back() == '\r' || v.back() == ' ')) v.pop_back();
    // strip a single layer of quoting if present
    if (v.size() >= 2 && ((v.front() == '\'' && v.back() == '\'') ||
                          (v.front() == '"' && v.back() == '"'))) {
        v = v.substr(1, v.size() - 2);
    }
    return v;
}

std::string scalar_key(const std::string& line) {
    auto pos = line.find(':');
    if (pos == std::string::npos) return {};
    std::string k = line.substr(0, pos);
    while (!k.empty() && k.front() == ' ') k.erase(k.begin());
    if (!k.empty() && k.rfind("- ", 0) == 0) k = k.substr(2);
    return k;
}

bool parse_api_yaml(const fs::path& path, ApiDoc* out, std::string* err) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        *err = "cannot open " + path.generic_string();
        return false;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) lines.push_back(line);
    if (lines.empty() || scalar_key(lines[0]) != "schema" ||
        scalar_value(lines[0]) != "turing-compiled-program-api-v1") {
        *err = "not a turing-compiled-program-api-v1 document";
        return false;
    }

    ApiDoc doc;
    // Pass 1: top-level scalars.
    for (const auto& l : lines) {
        if (indent_of(l) != 0) continue;
        const std::string key = scalar_key(l);
        if (key == "module") doc.module = scalar_value(l);
        else if (key == "language") doc.language = scalar_value(l);
        else if (key == "entry") doc.entry = scalar_value(l);
    }
    // Pass 1b: metadata.runtime_dependencies block:
    //   metadata:
    //     runtime_dependencies:
    //     - name: libgfortran-5.dll
    //       path: C:/msys64/mingw64/bin/libgfortran-5.dll
    {
        size_t m = 0;
        while (m < lines.size() &&
               !(indent_of(lines[m]) == 0 && scalar_key(lines[m]) == "metadata")) {
            ++m;
        }
        if (m < lines.size()) {
            ++m;
            bool in_deps = false;
            ApiRuntimeDependency* dep = nullptr;
            for (; m < lines.size(); ++m) {
                const std::string& l = lines[m];
                if (l.empty()) continue;
                const size_t ind = indent_of(l);
                if (ind == 0) break; // metadata block ended
                const bool is_item = l.compare(ind, 2, "- ") == 0;
                const std::string key = scalar_key(l);
                if (ind == 2 && !is_item) {
                    in_deps = (key == "runtime_dependencies");
                    dep = nullptr;
                    continue;
                }
                if (!in_deps) continue;
                if (ind == 2 && is_item) {
                    doc.runtime_dependencies.emplace_back();
                    dep = &doc.runtime_dependencies.back();
                    if (key == "name") dep->name = scalar_value(l);
                    else if (key == "path") dep->path = scalar_value(l);
                    continue;
                }
                if (ind == 4 && dep) {
                    if (key == "name") dep->name = scalar_value(l);
                    else if (key == "path") dep->path = scalar_value(l);
                }
            }
        }
    }
    // Pass 2: entry_points block. Layout emitted by safe_dump:
    //   entry_points:
    //   - name: ...            (indent 0, "- " item)
    //     symbol: ...          (indent 2)
    //     parameters:          (indent 2)
    //     - name: t0           (indent 2, "- " item)
    //       role: input        (indent 4)
    size_t i = 0;
    while (i < lines.size() && scalar_key(lines[i]) != "entry_points") ++i;
    ++i;
    ApiEntryPoint* ep = nullptr;
    ApiParameter* param = nullptr;
    bool in_parameters = false;
    for (; i < lines.size(); ++i) {
        const std::string& l = lines[i];
        if (l.empty()) continue;
        const size_t ind = indent_of(l);
        const bool is_item = l.compare(ind, 2, "- ") == 0;
        if (ind == 0 && !is_item) break; // next top-level key
        const std::string key = scalar_key(l);
        const std::string value = scalar_value(l);
        if (ind == 0 && is_item) {
            doc.entry_points.emplace_back();
            ep = &doc.entry_points.back();
            param = nullptr;
            in_parameters = false;
            if (key == "name") ep->name = value;
            continue;
        }
        if (!ep) continue;
        if (ind == 2 && !is_item) {
            in_parameters = (key == "parameters");
            param = nullptr;
            if (key == "name") ep->name = value;
            else if (key == "symbol") ep->symbol = value;
            else if (key == "kind") ep->kind = value;
            continue;
        }
        if (ind == 2 && is_item && in_parameters) {
            ep->parameters.emplace_back();
            param = &ep->parameters.back();
            if (key == "name") param->name = value;
            continue;
        }
        if (ind == 4 && param) {
            if (key == "name") param->name = value;
            else if (key == "role") param->role = value;
            else if (key == "dtype") param->dtype = value;
            else if (key == "c_type") param->c_type = value;
            else if (key == "passing") param->passing = value;
            else if (key == "source_name") param->source_name = value;
            else if (key == "shape") {
                // inline "[a, b]" form; block form arrives as "- N" at indent 4
                std::string v = value;
                if (!v.empty() && v.front() == '[') {
                    v = v.substr(1, v.size() - 2);
                    std::istringstream ss(v);
                    std::string tok;
                    while (std::getline(ss, tok, ',')) {
                        param->shape.push_back(std::stoll(tok));
                    }
                }
            }
            continue;
        }
        if (ind == 4 && is_item && param) {
            // block-form shape element: "    - 4"
            std::string v = l.substr(ind + 2);
            try { param->shape.push_back(std::stoll(v)); } catch (...) {}
            continue;
        }
    }
    *out = std::move(doc);
    return true;
}

// ---------------------------------------------------------------------------
// Wrapper generation
// ---------------------------------------------------------------------------

std::string sanitize_identifier(const std::string& id) {
    std::string out;
    for (char ch : id) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
            out.push_back(ch);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty() || (out[0] >= '0' && out[0] <= '9')) out.insert(out.begin(), '_');
    return out;
}

// Select the entry point a caller should invoke: the named `entry` if it
// exists, else the single control entry, else the single numerical one.
const ApiEntryPoint* select_entry(const ApiDoc& doc) {
    for (const auto& ep : doc.entry_points) {
        if (!doc.entry.empty() && doc.entry != "null" && ep.symbol == doc.entry) return &ep;
    }
    const ApiEntryPoint* control = nullptr;
    const ApiEntryPoint* numerical = nullptr;
    for (const auto& ep : doc.entry_points) {
        if (ep.kind == "control" && !control) control = &ep;
        if (ep.kind == "numerical" && !numerical) numerical = &ep;
    }
    return control ? control : numerical;
}

std::string escape_native_path(const fs::path& p) {
    std::string native = p.string();
    std::string escaped;
    for (char c : native) {
        if (c == '/' || c == '\\') escaped += "\\\\";
        else escaped += c;
    }
    return escaped;
}

std::string generate_wrapper_source(const ApiDoc& doc,
                                    const ApiEntryPoint& ep,
                                    const std::string& tool_id,
                                    const fs::path& artifact_path,
                                    std::string* shortfall) {
    std::vector<const ApiParameter*> inputs;
    const ApiParameter* output = nullptr;
    for (const auto& p : ep.parameters) {
        if (p.role == "input") inputs.push_back(&p);
        else if (p.role == "output") {
            if (output) { *shortfall = "more than one output parameter (v1 wraps exactly one)"; return {}; }
            output = &p;
        } else if (p.role == "extent") {
            *shortfall = "extent parameters not wired yet (v1 wraps static/scalar entries)";
            return {};
        }
    }
    if (!output) { *shortfall = "no output parameter"; return {}; }
    for (const auto& p : ep.parameters) {
        if (p.dtype != "float64") {
            *shortfall = "parameter '" + p.name + "' dtype '" + p.dtype +
                         "' (v1 wraps float64 only -- the C tensor layer is double-only)";
            return {};
        }
    }

    const std::string sym = sanitize_identifier(tool_id);
    std::ostringstream ss;
    ss << "// Generated by artifact_importer -- DO NOT EDIT.\n";
    ss << "// Artifact:  " << artifact_path.generic_string() << "\n";
    ss << "// Contract:  turing-compiled-program-api-v1, module " << doc.module
       << ", language " << doc.language << "\n";
    ss << "// Entry:     " << ep.symbol << " (kind " << ep.kind << ")\n";
    ss << "#include \"tool_api.h\"\n";
    ss << "#include \"value_types.h\"\n";
    ss << "#include \"common/tensors/abstraction/abstract_tensor_handle.h\"\n";
    ss << "#include \"common/tensors/abstraction/tensor_abi.h\"\n";
    ss << "#include <cstdio>\n";
    ss << "#ifdef _WIN32\n";
    ss << "#ifndef WIN32_LEAN_AND_MEAN\n#define WIN32_LEAN_AND_MEAN\n#endif\n";
    ss << "#include <windows.h>\n";
    ss << "#endif\n";
    ss << "\n";
    ss << "namespace {\n";
    // Entry signature: value doubles for by-value inputs, double* for
    // by-reference parameters, declared in the YAML's parameter order.
    ss << "using EntryFn = void (*)(";
    {
        bool first = true;
        for (const auto& p : ep.parameters) {
            if (!first) ss << ", ";
            first = false;
            ss << (p.passing == "value" ? "double" : "double*");
        }
    }
    ss << ");\n\n";
    ss << "class ImportedTool_" << sym << " final : public ITool {\n";
    ss << "public:\n";
    ss << "    const char* id_cstr() const noexcept override { return \"" << tool_id << "\"; }\n";
    ss << "    const char* name_cstr() const noexcept override { return \"imported "
       << doc.module << "\"; }\n";
    ss << "    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }\n";
    ss << "    void initialize(const ToolInitContext&) override {\n";
    ss << "#ifdef _WIN32\n";
    ss << "        if (!module_) {\n";
    // Contract-declared runtime dependencies (E15): register each declared
    // dependency's DIRECTORY with the loader (AddDllDirectory), which the
    // artifact load below honors through LOAD_LIBRARY_SEARCH_DEFAULT_DIRS.
    // Directory registration -- not file preloading -- is the mechanism that
    // resolves the dependencies' own inter-imports order-independently (a
    // preloaded libgfortran still needs the loader to find libquadmath for
    // itself; a registered directory serves every import that lives there).
    if (!doc.runtime_dependencies.empty()) {
        std::vector<std::string> dirs;
        for (const auto& dep : doc.runtime_dependencies) {
            std::string dir = escape_native_path(fs::path(dep.path).parent_path());
            if (std::find(dirs.begin(), dirs.end(), dir) == dirs.end()) {
                dirs.push_back(dir);
            }
        }
        ss << "        static const wchar_t* kRuntimeDepDirs[] = {\n";
        for (const auto& d : dirs) {
            ss << "            L\"" << d << "\",\n";
        }
        ss << "        };\n";
        ss << "        for (const wchar_t* dir : kRuntimeDepDirs) {\n";
        ss << "            if (!AddDllDirectory(dir)) {\n";
        ss << "                fprintf(stderr, \"[imported:" << sym
           << "] AddDllDirectory failed (%lu) for a declared dependency dir\\n\",\n";
        ss << "                        GetLastError());\n";
        ss << "            }\n";
        ss << "        }\n";
    }
    // LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: the artifact's own directory joins the
    // dependency search, so runtime DLLs placed beside it (or declared in the
    // contract's runtime_dependencies) resolve. Plain LoadLibraryA searches
    // the APPLICATION's directory for dependencies, not the artifact's --
    // which made a missing Fortran runtime fail silently (error register).
    // Native separators, escaped: the LOAD_LIBRARY_SEARCH_* flags only take
    // effect for a fully qualified path, and Windows' qualification check
    // wants backslashes (error register E15).
    ss << "            module_ = LoadLibraryExA(\"" << escape_native_path(artifact_path) << "\",\n";
    ss << "                nullptr,\n";
    ss << "                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);\n";
    ss << "            if (!module_) {\n";
    // Loud failure: a tool that cannot reach its engine must say so once,
    // not no-op forever (silence is the boundary's default failure mode).
    ss << "                fprintf(stderr, \"[imported:" << sym << "] LoadLibrary failed (%lu) for %s\\n\",\n";
    ss << "                        GetLastError(), \"" << artifact_path.generic_string() << "\");\n";
    ss << "            } else {\n";
    ss << "                entry_ = reinterpret_cast<EntryFn>(\n";
    ss << "                    GetProcAddress(static_cast<HMODULE>(module_), \"" << ep.symbol << "\"));\n";
    ss << "                if (!entry_) {\n";
    ss << "                    fprintf(stderr, \"[imported:" << sym << "] entry symbol '"
       << ep.symbol << "' not found\\n\");\n";
    ss << "                }\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "#endif\n";
    ss << "    }\n";
    ss << "    void shutdown() override {\n";
    ss << "#ifdef _WIN32\n";
    ss << "        if (module_) { FreeLibrary(static_cast<HMODULE>(module_)); module_ = nullptr; }\n";
    ss << "#endif\n";
    ss << "        entry_ = nullptr;\n";
    ss << "    }\n";
    ss << "    void tick(double, HostAPI&) override {}\n";
    ss << "    void render(RenderContext&) override {}\n";
    ss << "    int32_t port_count() const override { return 2; }\n";
    ss << "    ToolPortSpec port_spec(int32_t idx) const override {\n";
    ss << "        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, "
       << static_cast<int>(inputs.size()) << "};\n";
    ss << "        if (idx == 1) return ToolPortSpec{ToolPortKind::Return, 1};\n";
    ss << "        return ToolPortSpec{};\n";
    ss << "    }\n";
    ss << "    void execute_stack(ToolStackContext& ctx) override {\n";
    ss << "        if (!ctx.stack.raw || !entry_) return;\n";
    ss << "        RawStackFrame& frame = *ctx.stack.raw;\n";
    // Pop input handles in reverse of push order so h0..hN-1 follow the
    // contract's declared input order (feed order).
    ss << "        nodus::tensors::AbstractTensorHandle h["
       << inputs.size() << "]{};\n";
    ss << "        int popped = 0;\n";
    ss << "        for (int i = " << static_cast<int>(inputs.size()) << " - 1; i >= 0; --i) {\n";
    ss << "            if (!raw_stack_pop_abstract_tensor(frame, h[i])) break;\n";
    ss << "            ++popped;\n";
    ss << "        }\n";
    ss << "        auto restore = [&]() {\n";
    ss << "            for (int i = " << static_cast<int>(inputs.size())
       << " - popped; i < " << static_cast<int>(inputs.size()) << "; ++i) {\n";
    ss << "                raw_stack_push_abstract_tensor(frame, h[i]);\n";
    ss << "            }\n";
    ss << "        };\n";
    ss << "        if (popped != " << static_cast<int>(inputs.size()) << ") { restore(); return; }\n";
    // v1: every by-value input is a scalar read from element 0 of its tensor.
    ss << "        double in_vals[" << inputs.size() << "]{};\n";
    ss << "        for (int i = 0; i < " << static_cast<int>(inputs.size()) << "; ++i) {\n";
    ss << "            NodusTensorDesc d{};\n";
    ss << "            if (nodus_tensor_describe(h[i].id, &d) != NODUS_OK ||\n";
    ss << "                d.dtype != NODUS_DTYPE_F64 ||\n";
    ss << "                nodus_tensor_read(h[i].id, 0, &in_vals[i], sizeof(double)) !=\n";
    ss << "                    static_cast<int64_t>(sizeof(double))) {\n";
    ss << "                restore(); return;\n";
    ss << "            }\n";
    ss << "        }\n";
    ss << "        double out_val = 0.0;\n";
    // Call with arguments in the YAML's declared parameter order.
    ss << "        entry_(";
    {
        bool first = true;
        size_t input_index = 0;
        for (const auto& p : ep.parameters) {
            if (!first) ss << ", ";
            first = false;
            if (p.role == "output") ss << "&out_val";
            else ss << "in_vals[" << input_index++ << "]";
        }
    }
    ss << ");\n";
    ss << "        const uint64_t shape1[1] = {1};\n";
    ss << "        nodus::tensors::AbstractTensorHandle out{};\n";
    ss << "        out.id = nodus_tensor_create(NODUS_DTYPE_F64, 1, shape1);\n";
    ss << "        if (out.id == 0 ||\n";
    ss << "            nodus_tensor_write(out.id, 0, &out_val, sizeof(double)) !=\n";
    ss << "                static_cast<int64_t>(sizeof(double)) ||\n";
    ss << "            !raw_stack_push_abstract_tensor(frame, out)) {\n";
    ss << "            if (out.id) nodus_tensor_destroy(out.id);\n";
    ss << "            restore(); return;\n";
    ss << "        }\n";
    ss << "    }\n";
    ss << "private:\n";
    ss << "    void* module_ = nullptr;\n";
    ss << "    EntryFn entry_ = nullptr;\n";
    ss << "};\n";
    ss << "} // namespace\n";
    ss << "\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() {\n";
    ss << "    return new ImportedTool_" << sym << "();\n";
    ss << "}\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) {\n";
    ss << "    delete t;\n";
    ss << "}\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() {\n";
    ss << "    return __FILE__;\n";
    ss << "}\n";
    ss << "#if !defined(NODUS_PLUGIN_COMPOSITE)\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* nodus_tool_backend_group() {\n";
    ss << "    return \"compiled-" << doc.language << "\";\n";
    ss << "}\n";
    ss << "#endif\n";
    return ss.str();
}

bool write_file(const fs::path& target, const std::string& contents, std::string* err) {
    try {
        auto parent = target.parent_path();
        if (!parent.empty()) fs::create_directories(parent);
        std::ofstream ofs(target, std::ios::binary | std::ios::trunc);
        if (!ofs.good()) { *err = "cannot open " + target.generic_string(); return false; }
        ofs << contents;
        return ofs.good();
    } catch (const std::exception& e) {
        *err = e.what();
        return false;
    }
}

} // namespace

extern "C" int gp_artifact_import(const char* api_yaml_path,
                                  const char* artifact_path,
                                  const char* output_root,
                                  GP_ArtifactImportReport* report,
                                  char* out_package_path,
                                  int32_t out_package_cap) {
    if (!api_yaml_path || !artifact_path || !output_root) return 0;
    GP_ArtifactImportReport local{};

    ApiDoc doc;
    std::string err;
    if (!parse_api_yaml(api_yaml_path, &doc, &err)) {
        std::cerr << "artifact_importer: " << err << "\n";
        if (report) *report = local;
        return 0;
    }
    local.entry_points_seen = static_cast<int32_t>(doc.entry_points.size());

    const ApiEntryPoint* ep = select_entry(doc);
    if (!ep) {
        std::cerr << "artifact_importer: no callable entry point in " << api_yaml_path << "\n";
        if (report) *report = local;
        return 0;
    }

    const std::string tool_id = "imported." + doc.module;
    std::string shortfall;
    const std::string wrapper = generate_wrapper_source(
        doc, *ep, tool_id, fs::path(artifact_path), &shortfall);
    if (wrapper.empty()) {
        std::cerr << "artifact_importer: shortfall for entry '" << ep->symbol
                  << "': " << shortfall << "\n";
        ++local.shortfalls;
        if (report) *report = local;
        return 0;
    }

    const fs::path root = output_root;
    const fs::path wrapper_path =
        root / "source" / "tools" / "imported" / (sanitize_identifier(tool_id) + ".cpp");
    if (!write_file(wrapper_path, wrapper, &err)) {
        std::cerr << "artifact_importer: " << err << "\n";
        if (report) *report = local;
        return 0;
    }
    ++local.wrappers_written;

    // NODUSPKG manifest for the real ingest chassis (repo_package.cpp).
    std::ostringstream pkg;
    pkg << "NODUSPKG V1\n";
    pkg << "OWNER \"imported_" << sanitize_identifier(doc.module) << "\"\n";
    pkg << "VERSION \"1\"\n";
    pkg << "ROOT \"" << root.generic_string() << "\"\n";
    pkg << "TOOL " << tool_id << " \"imported " << doc.module << "\" \""
        << wrapper_path.generic_string() << "\" 0 0 \"backend-group compiled-"
        << doc.language << "; entry " << ep->symbol << "\"\n";
    const fs::path pkg_path = root / (sanitize_identifier(doc.module) + ".nodus_package.txt");
    if (!write_file(pkg_path, pkg.str(), &err)) {
        std::cerr << "artifact_importer: " << err << "\n";
        if (report) *report = local;
        return 0;
    }

    if (out_package_path && out_package_cap > 0) {
        const std::string s = pkg_path.generic_string();
        std::strncpy(out_package_path, s.c_str(), static_cast<size_t>(out_package_cap) - 1);
        out_package_path[out_package_cap - 1] = '\0';
    }
    if (report) *report = local;
    std::cerr << "artifact_importer: wrapped " << ep->symbol << " as " << tool_id
              << " (group compiled-" << doc.language << ")\n";
    return 1;
}
