#include "artifact_importer.h"

#include <algorithm>
#include <cctype>
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

struct BufferValue {
    enum class Role { Input, Output, Workspace };
    Role role = Role::Input;
    std::string ssa_id;
    std::string dtype;
    int32_t nodus_dtype = 0;
    std::vector<uint64_t> shape;
};

struct BufferExtent {
    std::string ssa_id;
    uint32_t axis = 0;
};

struct BufferAbiDoc {
    std::string backend;
    std::string extent_dtype;
    std::string tool_id;
    std::string tool_name;
    std::string entry;
    std::vector<BufferValue> values;
    std::vector<std::string> buffer_order;
    std::vector<BufferExtent> extent_order;
};

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

int32_t buffer_dtype_id(const std::string& spelling) {
    const std::string dtype = lowercase(spelling);
    if (dtype == "f32" || dtype == "float32") return 1;
    if (dtype == "f64" || dtype == "float64") return 2;
    if (dtype == "i8" || dtype == "int8") return 3;
    if (dtype == "i16" || dtype == "int16") return 4;
    if (dtype == "i32" || dtype == "int32") return 5;
    if (dtype == "i64" || dtype == "int64") return 6;
    if (dtype == "u8" || dtype == "uint8") return 7;
    if (dtype == "u16" || dtype == "uint16") return 12;
    if (dtype == "u32" || dtype == "uint32") return 13;
    if (dtype == "u64" || dtype == "uint64") return 14;
    if (dtype == "bool") return 15;
    return 0;
}

const BufferValue* find_buffer_value(const BufferAbiDoc& doc, const std::string& ssa_id) {
    for (const auto& value : doc.values) {
        if (value.ssa_id == ssa_id) return &value;
    }
    return nullptr;
}

bool parse_buffer_abi(const fs::path& path, BufferAbiDoc* out, std::string* err) {
    std::ifstream ifs(path);
    if (!ifs.good()) {
        *err = "cannot open " + path.generic_string();
        return false;
    }
    std::string line;
    if (!std::getline(ifs, line) || line != "NODUS BUFFER ABI V1") {
        *err = "not a NODUS BUFFER ABI V1 descriptor";
        return false;
    }
    BufferAbiDoc doc;
    size_t line_no = 1;
    while (std::getline(ifs, line)) {
        ++line_no;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        std::istringstream ss(line.substr(first));
        std::string tag;
        ss >> tag;
        if (tag == "BACKEND") {
            ss >> std::quoted(doc.backend);
            if (!ss) ss.clear(), ss.str(line.substr(first + tag.size())), ss >> doc.backend;
        } else if (tag == "EXTENT_DTYPE") {
            ss >> std::quoted(doc.extent_dtype);
        } else if (tag == "TOOL") {
            ss >> std::quoted(doc.tool_id) >> std::quoted(doc.tool_name);
        } else if (tag == "ENTRY") {
            ss >> std::quoted(doc.entry);
            if (!ss) ss.clear(), ss.str(line.substr(first + tag.size())), ss >> doc.entry;
        } else if (tag == "VALUE") {
            std::string role;
            BufferValue value;
            int64_t rank = -1;
            ss >> role >> std::quoted(value.ssa_id) >> std::quoted(value.dtype) >> rank;
            if (role == "INPUT") value.role = BufferValue::Role::Input;
            else if (role == "OUTPUT") value.role = BufferValue::Role::Output;
            else if (role == "WORKSPACE") value.role = BufferValue::Role::Workspace;
            else {
                *err = "line " + std::to_string(line_no) + ": VALUE role must be INPUT, OUTPUT, or WORKSPACE";
                return false;
            }
            if (!ss || value.ssa_id.empty() || rank < 0 || rank > 8) {
                *err = "line " + std::to_string(line_no) + ": malformed VALUE or unsupported rank";
                return false;
            }
            value.nodus_dtype = buffer_dtype_id(value.dtype);
            if (value.nodus_dtype == 0) {
                *err = "line " + std::to_string(line_no) + ": unsupported dtype '" + value.dtype + "'";
                return false;
            }
            for (int64_t i = 0; i < rank; ++i) {
                int64_t dim = 0;
                if (!(ss >> dim) || dim <= 0) {
                    *err = "line " + std::to_string(line_no) + ": dynamic or invalid extent refused";
                    return false;
                }
                value.shape.push_back(static_cast<uint64_t>(dim));
            }
            std::string trailing;
            if (ss >> trailing) {
                *err = "line " + std::to_string(line_no) + ": trailing VALUE fields";
                return false;
            }
            if (find_buffer_value(doc, value.ssa_id)) {
                *err = "line " + std::to_string(line_no) + ": duplicate SSA value '" + value.ssa_id + "'";
                return false;
            }
            doc.values.push_back(std::move(value));
        } else if (tag == "BUFFER") {
            std::string id;
            ss >> std::quoted(id);
            if (!ss || id.empty()) {
                *err = "line " + std::to_string(line_no) + ": malformed BUFFER";
                return false;
            }
            doc.buffer_order.push_back(std::move(id));
        } else if (tag == "EXTENT") {
            BufferExtent extent;
            int64_t axis = -1;
            ss >> std::quoted(extent.ssa_id) >> axis;
            if (!ss || extent.ssa_id.empty() || axis < 0) {
                *err = "line " + std::to_string(line_no) + ": malformed EXTENT";
                return false;
            }
            extent.axis = static_cast<uint32_t>(axis);
            doc.extent_order.push_back(std::move(extent));
        } else {
            *err = "line " + std::to_string(line_no) + ": unknown record '" + tag + "'";
            return false;
        }
    }

    doc.backend = lowercase(doc.backend);
    if (doc.backend != "c" && doc.backend != "llvm") {
        *err = "backend must be 'c' or 'llvm'";
        return false;
    }
    doc.extent_dtype = lowercase(doc.extent_dtype);
    const std::string required_extent_dtype = doc.backend == "llvm" ? "i32" : "i64";
    if (doc.extent_dtype != required_extent_dtype) {
        *err = "EXTENT_DTYPE must be '" + required_extent_dtype + "' for backend '" +
               doc.backend + "'";
        return false;
    }
    if (doc.tool_id.empty() || doc.tool_name.empty() || doc.entry.empty()) {
        *err = "descriptor requires BACKEND, TOOL id/name, and ENTRY";
        return false;
    }
    if (doc.tool_id.find_first_of(" \t\r\n\"") != std::string::npos) {
        *err = "TOOL id must be a non-whitespace manifest token";
        return false;
    }
    size_t input_count = 0;
    size_t output_count = 0;
    for (const auto& value : doc.values) {
        if (value.role == BufferValue::Role::Input) ++input_count;
        else if (value.role == BufferValue::Role::Output) ++output_count;
    }
    if (output_count == 0) {
        *err = "descriptor requires at least one OUTPUT value";
        return false;
    }
    if (doc.buffer_order.size() != doc.values.size()) {
        *err = "BUFFER order must name every VALUE exactly once";
        return false;
    }
    std::vector<std::string> seen;
    for (const auto& id : doc.buffer_order) {
        if (!find_buffer_value(doc, id)) {
            *err = "BUFFER names unknown SSA value '" + id + "'";
            return false;
        }
        if (std::find(seen.begin(), seen.end(), id) != seen.end()) {
            *err = "BUFFER repeats SSA value '" + id + "'";
            return false;
        }
        seen.push_back(id);
    }
    for (const auto& extent : doc.extent_order) {
        const BufferValue* value = find_buffer_value(doc, extent.ssa_id);
        if (!value || extent.axis >= value->shape.size()) {
            *err = "EXTENT references unknown value or invalid axis for '" + extent.ssa_id + "'";
            return false;
        }
    }
    (void)input_count;
    *out = std::move(doc);
    return true;
}

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

std::string backend_qualified_stem(const std::string& tool_id, const std::string& backend) {
    std::string stem = sanitize_identifier(tool_id);
    const std::string suffix = "_" + sanitize_identifier(backend);
    if (stem.size() < suffix.size() ||
        stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) != 0) {
        stem += suffix;
    }
    return stem;
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

std::string escape_cpp_string(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (char c : value) {
        if (c == '\\' || c == '"') escaped.push_back('\\');
        if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else escaped.push_back(c);
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

std::string generate_buffer_wrapper_source(const BufferAbiDoc& doc,
                                           const fs::path& artifact_path) {
    std::vector<size_t> input_indices;
    std::vector<size_t> output_indices;
    std::vector<size_t> workspace_indices;
    for (size_t i = 0; i < doc.values.size(); ++i) {
        if (doc.values[i].role == BufferValue::Role::Input) input_indices.push_back(i);
        else if (doc.values[i].role == BufferValue::Role::Output) output_indices.push_back(i);
        else workspace_indices.push_back(i);
    }
    std::vector<size_t> buffer_indices;
    for (const auto& id : doc.buffer_order) {
        for (size_t i = 0; i < doc.values.size(); ++i) {
            if (doc.values[i].ssa_id == id) {
                buffer_indices.push_back(i);
                break;
            }
        }
    }

    const std::string sym = backend_qualified_stem(doc.tool_id, doc.backend);
    std::ostringstream ss;
    ss << "// Generated by artifact_importer -- DO NOT EDIT.\n";
    ss << "// Contract: NODUS BUFFER ABI V1, backend " << doc.backend << "\n";
    ss << "// Artifact: " << artifact_path.generic_string() << "\n";
    ss << "// Entry: " << doc.entry << "\n";
    ss << "#include \"tool_api.h\"\n";
    ss << "#include \"value_types.h\"\n";
    ss << "#include \"common/tensors/abstraction/abstract_tensor_handle.h\"\n";
    ss << "#include \"common/tensors/abstraction/tensor_abi.h\"\n";
    ss << "#include <array>\n#include <cstdint>\n#include <cstdio>\n\n";
    ss << "#ifdef _WIN32\n#ifndef WIN32_LEAN_AND_MEAN\n#define WIN32_LEAN_AND_MEAN\n#endif\n#include <windows.h>\n#endif\n\n";
    ss << "namespace {\n";
    const char* extent_type = doc.extent_dtype == "i32" ? "int32_t" : "int64_t";
    ss << "using Extent = " << extent_type << ";\n";
    ss << "using EntryFn = void (*)(void**, Extent*);\n";
    ss << "class ImportedBufferTool_" << sym << " final : public ITool {\npublic:\n";
    ss << "    const char* id_cstr() const noexcept override { return \""
       << escape_cpp_string(doc.tool_id) << "\"; }\n";
    ss << "    const char* name_cstr() const noexcept override { return \""
       << escape_cpp_string(doc.tool_name) << "\"; }\n";
    ss << "    ToolCaps caps() const override { return static_cast<ToolCaps>(0); }\n";
    ss << "    void initialize(const ToolInitContext&) override {\n#ifdef _WIN32\n";
    ss << "        if (!module_) {\n";
    ss << "            module_ = LoadLibraryExA(\"" << escape_native_path(artifact_path)
       << "\", nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);\n";
    ss << "            if (!module_) {\n";
    ss << "                fprintf(stderr, \"[imported-buffer:" << sym
       << "] LoadLibrary failed (%lu)\\n\", GetLastError());\n";
    ss << "            } else {\n";
    ss << "                entry_ = reinterpret_cast<EntryFn>(GetProcAddress(static_cast<HMODULE>(module_), \""
       << escape_cpp_string(doc.entry) << "\"));\n";
    ss << "                if (!entry_) fprintf(stderr, \"[imported-buffer:" << sym
       << "] entry symbol not found: " << escape_cpp_string(doc.entry) << "\\n\");\n";
    ss << "            }\n        }\n#endif\n    }\n";
    ss << "    void shutdown() override {\n#ifdef _WIN32\n"
          "        if (module_) { FreeLibrary(static_cast<HMODULE>(module_)); module_ = nullptr; }\n"
          "#endif\n        entry_ = nullptr;\n    }\n";
    ss << "    void tick(double, HostAPI&) override {}\n";
    ss << "    void render(RenderContext&) override {}\n";
    ss << "    int32_t port_count() const override { return 2; }\n";
    ss << "    ToolPortSpec port_spec(int32_t idx) const override {\n";
    ss << "        if (idx == 0) return ToolPortSpec{ToolPortKind::Argument, " << input_indices.size() << "};\n";
    ss << "        if (idx == 1) return ToolPortSpec{ToolPortKind::Return, " << output_indices.size() << "};\n";
    ss << "        return ToolPortSpec{};\n    }\n";
    ss << "    void execute_stack(ToolStackContext& ctx) override {\n";
    ss << "        if (!ctx.stack.raw || !entry_) return;\n";
    ss << "        RawStackFrame& frame = *ctx.stack.raw;\n";
    ss << "        std::array<nodus::tensors::AbstractTensorHandle, " << doc.values.size() << "> handles{};\n";
    ss << "        constexpr std::array<int, " << input_indices.size() << "> kInputs = {";
    for (size_t i = 0; i < input_indices.size(); ++i) {
        if (i) ss << ", ";
        ss << input_indices[i];
    }
    ss << "};\n";
    ss << "        constexpr std::array<int, " << output_indices.size() << "> kOutputs = {";
    for (size_t i = 0; i < output_indices.size(); ++i) {
        if (i) ss << ", ";
        ss << output_indices[i];
    }
    ss << "};\n";
    ss << "        constexpr std::array<int, " << workspace_indices.size() << "> kWorkspaces = {";
    for (size_t i = 0; i < workspace_indices.size(); ++i) {
        if (i) ss << ", ";
        ss << workspace_indices[i];
    }
    ss << "};\n";
    ss << "        int popped = 0;\n";
    ss << "        for (int i = static_cast<int>(kInputs.size()) - 1; i >= 0; --i) {\n"
          "            if (!raw_stack_pop_abstract_tensor(frame, handles[static_cast<size_t>(kInputs[static_cast<size_t>(i)])])) break;\n"
          "            ++popped;\n        }\n";
    ss << "        auto restore_inputs = [&]() {\n"
          "            for (int idx : kInputs) raw_stack_push_abstract_tensor(frame, handles[static_cast<size_t>(idx)]);\n"
          "        };\n";
    ss << "        if (popped != static_cast<int>(kInputs.size())) {\n"
          "            for (int i = static_cast<int>(kInputs.size()) - popped; i < static_cast<int>(kInputs.size()); ++i)\n"
          "                raw_stack_push_abstract_tensor(frame, handles[static_cast<size_t>(kInputs[static_cast<size_t>(i)])]);\n"
          "            return;\n        }\n";
    ss << "        std::array<void*, " << doc.values.size() << "> data{};\n";
    ss << "        std::array<bool, " << doc.values.size() << "> mapped{};\n";
    ss << "        auto fail = [&]() {\n"
          "            for (size_t i = 0; i < handles.size(); ++i) if (mapped[i]) nodus_tensor_unmap(handles[i].id);\n"
          "            for (int idx : kOutputs) if (handles[static_cast<size_t>(idx)].id) nodus_tensor_destroy(handles[static_cast<size_t>(idx)].id);\n"
          "            for (int idx : kWorkspaces) if (handles[static_cast<size_t>(idx)].id) nodus_tensor_destroy(handles[static_cast<size_t>(idx)].id);\n"
          "            restore_inputs();\n"
          "        };\n";

    for (size_t i = 0; i < doc.values.size(); ++i) {
        const auto& value = doc.values[i];
        if (value.role == BufferValue::Role::Input) {
            ss << "        { NodusTensorDesc d{};\n";
            ss << "          if (nodus_tensor_describe(handles[" << i << "].id, &d) != NODUS_OK || d.dtype != "
               << value.nodus_dtype << " || d.rank != " << value.shape.size();
            for (size_t axis = 0; axis < value.shape.size(); ++axis) {
                ss << " || d.shape[" << axis << "] != " << value.shape[axis] << "ull";
            }
            ss << ") { fail(); return; } }\n";
        } else {
            if (value.shape.empty()) {
                ss << "        handles[" << i << "].id = nodus_tensor_create(" << value.nodus_dtype << ", 0, nullptr);\n";
            } else {
                ss << "        { static constexpr uint64_t shape[] = {";
                for (size_t axis = 0; axis < value.shape.size(); ++axis) {
                    if (axis) ss << ", ";
                    ss << value.shape[axis] << "ull";
                }
                ss << "}; handles[" << i << "].id = nodus_tensor_create(" << value.nodus_dtype << ", "
                   << value.shape.size() << ", shape); }\n";
            }
            ss << "        if (!handles[" << i << "].id) { fail(); return; }\n";
            if (value.role == BufferValue::Role::Workspace) {
                ss << "        if (nodus_tensor_zero(handles[" << i << "].id, 0, 0) != NODUS_OK) { fail(); return; }\n";
            }
        }
        ss << "        { uint64_t bytes = 0; if (nodus_tensor_map(handles[" << i
           << "].id, &data[" << i << "], &bytes) != NODUS_OK) { fail(); return; } mapped["
           << i << "] = true; }\n";
    }

    ss << "        std::array<void*, " << buffer_indices.size() << "> buffers = {";
    for (size_t i = 0; i < buffer_indices.size(); ++i) {
        if (i) ss << ", ";
        ss << "data[" << buffer_indices[i] << "]";
    }
    ss << "};\n";
    ss << "        std::array<Extent, " << doc.extent_order.size() << "> extents = {";
    for (size_t i = 0; i < doc.extent_order.size(); ++i) {
        if (i) ss << ", ";
        const auto* value = find_buffer_value(doc, doc.extent_order[i].ssa_id);
        ss << "static_cast<Extent>(" << value->shape[doc.extent_order[i].axis] << "ull)";
    }
    ss << "};\n";
    ss << "        entry_(buffers.data(), extents.empty() ? nullptr : extents.data());\n";
    ss << "        for (size_t i = 0; i < handles.size(); ++i) { if (mapped[i]) { nodus_tensor_unmap(handles[i].id); mapped[i] = false; } }\n";
    ss << "        for (int idx : kWorkspaces) { nodus_tensor_destroy(handles[static_cast<size_t>(idx)].id); handles[static_cast<size_t>(idx)].id = 0; }\n";
    ss << "        int pushed = 0;\n"
          "        for (int idx : kOutputs) {\n"
          "            if (!raw_stack_push_abstract_tensor(frame, handles[static_cast<size_t>(idx)])) {\n"
          "                for (int j = 0; j < pushed; ++j) { nodus::tensors::AbstractTensorHandle undo{}; if (raw_stack_pop_abstract_tensor(frame, undo)) nodus_tensor_destroy(undo.id); }\n"
          "                for (size_t j = static_cast<size_t>(pushed); j < kOutputs.size(); ++j) nodus_tensor_destroy(handles[static_cast<size_t>(kOutputs[j])].id);\n"
          "                restore_inputs(); return;\n"
          "            }\n"
          "            ++pushed;\n"
          "        }\n"
          "    }\n";
    ss << "private:\n    void* module_ = nullptr;\n    EntryFn entry_ = nullptr;\n};\n} // namespace\n\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new ImportedBufferTool_"
       << sym << "(); }\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }\n";
    ss << "extern \"C\" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }\n";
    ss << "#if !defined(NODUS_PLUGIN_COMPOSITE)\n"
          "extern \"C\" NODUS_PLUGIN_EXPORT const char* nodus_tool_backend_group() { return \"compiled-"
       << doc.backend << "\"; }\n#endif\n";
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

void copy_path_out(const fs::path& path, char* out, int32_t cap) {
    if (!out || cap <= 0) return;
    const std::string value = path.generic_string();
    std::strncpy(out, value.c_str(), static_cast<size_t>(cap) - 1);
    out[cap - 1] = '\0';
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

extern "C" int gp_artifact_import_buffer_abi(const char* descriptor_path,
                                               const char* artifact_path,
                                               const char* output_root,
                                               GP_ArtifactImportReport* report,
                                               char* out_wrapper_path,
                                               int32_t out_wrapper_cap,
                                               char* out_package_path,
                                               int32_t out_package_cap) {
    if (!descriptor_path || !artifact_path || !output_root) return 0;
    GP_ArtifactImportReport local{};
    local.entry_points_seen = 1;
    BufferAbiDoc doc;
    std::string err;
    if (!parse_buffer_abi(descriptor_path, &doc, &err)) {
        std::cerr << "artifact_importer: buffer ABI refusal: " << err << "\n";
        local.shortfalls = 1;
        if (report) *report = local;
        return 0;
    }

    const fs::path root = output_root;
    const std::string stem = backend_qualified_stem(doc.tool_id, doc.backend);
    const fs::path wrapper_path = root / "source" / "tools" / (stem + ".cpp");
    if (!write_file(wrapper_path, generate_buffer_wrapper_source(doc, fs::absolute(artifact_path)), &err)) {
        std::cerr << "artifact_importer: " << err << "\n";
        if (report) *report = local;
        return 0;
    }
    ++local.wrappers_written;

    std::ostringstream pkg;
    pkg << "NODUSPKG V1\n";
    pkg << "OWNER \"imported_" << stem << "\"\n";
    pkg << "VERSION \"1\"\n";
    pkg << "ROOT \"" << root.generic_string() << "\"\n";
    pkg << "TOOL " << doc.tool_id << " " << std::quoted(doc.tool_name) << " "
        << std::quoted(wrapper_path.generic_string()) << " 0 2 "
        << std::quoted("backend-group compiled-" + doc.backend +
                       "; buffer-abi; entry " + doc.entry) << "\n";
    const fs::path package_path = root / (stem + ".nodus_package.txt");
    if (!write_file(package_path, pkg.str(), &err)) {
        std::cerr << "artifact_importer: " << err << "\n";
        if (report) *report = local;
        return 0;
    }

    copy_path_out(wrapper_path, out_wrapper_path, out_wrapper_cap);
    copy_path_out(package_path, out_package_path, out_package_cap);
    if (report) *report = local;
    std::cerr << "artifact_importer: wrapped buffer entry " << doc.entry << " as "
              << doc.tool_id << " (group compiled-" << doc.backend << ")\n";
    return 1;
}
