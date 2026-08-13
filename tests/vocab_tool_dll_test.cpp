// vocab_tool_dll_test -- the vocabulary table checker.
//
// End-to-end proof of the vocabulary->tool pathway, one operation at a time:
//   1. gp_vocabulary_actualize_tools() emits per-op plugin sources for every
//      requested backend group (inmemory + eigen) from the canonical catalog.
//   2. One generated CMake project builds every source into its own tool DLL
//      (single configure, single parallel build).
//   3. Each DLL is loaded and its create_tool() factory invoked directly --
//      deliberately NOT through PluginLoader: this host links the shared
//      canvas_tables library so its nodus_tensor_* calls resolve into the
//      same image (and the same arena) the plugins use, and the C++
//      PluginLoader/ToolRegistry types are not part of that DLL's curated
//      export surface. create_tool() IS the plugin ABI, so calling it
//      directly tests exactly the boundary a host would use.
//   4. Every tool executes against real F32 tensors created through the
//      nodus_tensor_* transport, with operands passed as VT_ABSTRACT_TENSOR
//      handles on a raw typed stack -- the same currency the generated tools
//      speak.
//   5. Results are checked two ways: against an independent <cmath> reference
//      computed in this test, and (for the eigen group) against the inmemory
//      group's result for the same op, so the two engines must agree with the
//      reference AND each other.
//
// Per-op verdict lines go to stdout; exit code is nonzero if any op fails.
#include "vocabulary_actualizer.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
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

struct VocabEntry {
    std::string tool_id;
    std::string group;
    fs::path source;
    std::string op_name; // trailing segment of tool_id
};

struct RefSpec {
    int arity = 1;
    double a = 0.6;
    double b = -0.3;
    std::function<double(double, double)> fn;
};

// Independent reference semantics, computed in double. Domain-safe inputs per
// op; all tensor elements carry the same value so vectorized paths are
// exercised without leaving any op's domain.
std::map<std::string, RefSpec> make_reference() {
    std::map<std::string, RefSpec> ref;
    auto u = [](std::function<double(double)> f, double a = 0.6) {
        RefSpec s;
        s.arity = 1;
        s.a = a;
        s.fn = [f](double x, double) { return f(x); };
        return s;
    };
    auto b2 = [](std::function<double(double, double)> f, double a = 0.6, double b = -0.3) {
        RefSpec s;
        s.arity = 2;
        s.a = a;
        s.b = b;
        s.fn = std::move(f);
        return s;
    };
    ref["add"] = b2([](double x, double y) { return x + y; });
    ref["sub"] = b2([](double x, double y) { return x - y; });
    ref["mul"] = b2([](double x, double y) { return x * y; });
    ref["truediv"] = b2([](double x, double y) { return x / y; }, 7.0, 2.5);
    ref["pow"] = b2([](double x, double y) { return std::pow(x, y); }, 2.0, 3.0);
    ref["mod"] = b2([](double x, double y) { return std::fmod(x, y); }, 7.0, 2.5);
    ref["floordiv"] = b2([](double x, double y) { return std::floor(x / y); }, 7.0, 2.5);
    ref["sqrt"] = u([](double x) { return std::sqrt(x); }, 2.25);
    ref["exp"] = u([](double x) { return std::exp(x); });
    ref["log"] = u([](double x) { return std::log(x); }, 2.5);
    ref["neg"] = u([](double x) { return -x; });
    ref["abs"] = u([](double x) { return std::fabs(x); }, -0.6);
    ref["round"] = u([](double x) { return std::round(x); }, 0.6);
    ref["trunc"] = u([](double x) { return std::trunc(x); }, 1.7);
    ref["floor"] = u([](double x) { return std::floor(x); }, 1.7);
    ref["ceil"] = u([](double x) { return std::ceil(x); }, 1.2);
    ref["isfinite"] = u([](double x) { return std::isfinite(x) ? 1.0 : 0.0; });
    ref["isnan"] = u([](double x) { return std::isnan(x) ? 1.0 : 0.0; });
    ref["isinf"] = u([](double x) { return std::isinf(x) ? 1.0 : 0.0; });
    ref["logical_not"] = u([](double x) { return x == 0.0 ? 1.0 : 0.0; });
    ref["less"] = b2([](double x, double y) { return x < y ? 1.0 : 0.0; });
    ref["less_equal"] = b2([](double x, double y) { return x <= y ? 1.0 : 0.0; });
    ref["greater"] = b2([](double x, double y) { return x > y ? 1.0 : 0.0; });
    ref["greater_equal"] = b2([](double x, double y) { return x >= y ? 1.0 : 0.0; });
    ref["equal"] = b2([](double x, double y) { return x == y ? 1.0 : 0.0; });
    ref["not_equal"] = b2([](double x, double y) { return x != y ? 1.0 : 0.0; });
    ref["maximum"] = b2([](double x, double y) { return std::fmax(x, y); });
    ref["minimum"] = b2([](double x, double y) { return std::fmin(x, y); });
    ref["sin"] = u([](double x) { return std::sin(x); });
    ref["cos"] = u([](double x) { return std::cos(x); });
    ref["tan"] = u([](double x) { return std::tan(x); });
    ref["asin"] = u([](double x) { return std::asin(x); }, 0.4);
    ref["acos"] = u([](double x) { return std::acos(x); }, 0.4);
    ref["atan"] = u([](double x) { return std::atan(x); });
    ref["sinh"] = u([](double x) { return std::sinh(x); });
    ref["cosh"] = u([](double x) { return std::cosh(x); });
    ref["tanh"] = u([](double x) { return std::tanh(x); });
    ref["asinh"] = u([](double x) { return std::asinh(x); });
    ref["acosh"] = u([](double x) { return std::acosh(x); }, 1.5);
    ref["atanh"] = u([](double x) { return std::atanh(x); }, 0.4);
    return ref;
}

std::vector<VocabEntry> parse_manifest(const fs::path& manifest_path) {
    std::vector<VocabEntry> out;
    std::ifstream ifs(manifest_path);
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "VOCABTOOL") continue;
        VocabEntry e;
        ss >> e.tool_id >> e.group;
        std::string rest;
        std::getline(ss, rest);
        auto first = rest.find('"');
        auto last = rest.rfind('"');
        if (first == std::string::npos || last <= first) continue;
        e.source = rest.substr(first + 1, last - first - 1);
        auto dot = e.tool_id.rfind('.');
        e.op_name = (dot == std::string::npos) ? e.tool_id : e.tool_id.substr(dot + 1);
        out.push_back(std::move(e));
    }
    return out;
}

std::string sanitize(const std::string& id) {
    std::string out;
    for (char ch : id) {
        out.push_back((std::isalnum(static_cast<unsigned char>(ch))) ? ch : '_');
    }
    return out;
}

bool run_command(const std::string& cmd) {
    return std::system(cmd.c_str()) == 0;
}

// Generated tools and this host both link nodus_tensor_core -- the
// deliberately-shared home of the tensor singletons (see CMakeLists.txt) --
// so every image resolves to the same arena and a handle created here is
// valid inside a plugin.
fs::path find_tensor_core_import_lib(const fs::path& build_dir) {
    const fs::path release = build_dir / "Release" / "nodus_tensor_core.lib";
    if (fs::exists(release)) return release;
    const fs::path debug = build_dir / "Debug" / "nodus_tensor_core.lib";
    if (fs::exists(debug)) return debug;
    for (auto const& entry : fs::recursive_directory_iterator(build_dir)) {
        if (entry.is_regular_file() && entry.path().filename() == "nodus_tensor_core.lib")
            return entry.path();
    }
    return {};
}

uint64_t make_filled_tensor(double value, uint64_t n) {
    const uint64_t shape[1] = {n};
    uint64_t h = nodus_tensor_create(NODUS_DTYPE_F32, 1, shape);
    if (!h) return 0;
    std::vector<float> vals(static_cast<size_t>(n), static_cast<float>(value));
    if (nodus_tensor_write(h, 0, vals.data(), vals.size() * sizeof(float)) < 0) {
        nodus_tensor_destroy(h);
        return 0;
    }
    return h;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cerr << "vocab_tool_dll_test skipped: plugin DLL loading is Windows-only here.\n";
    return 0;
#else
    const fs::path repo_root = fs::path(NODUS_REPO_ROOT);
    const fs::path build_dir = fs::path(NODUS_BUILD_DIR);
    const fs::path test_root = repo_root / "module_library" / "test_build" / "vocab_tool_test";
    fs::create_directories(test_root);

    // 1) Actualize the vocabulary.
    GP_VocabActualizeReport report{};
    if (!gp_vocabulary_actualize_tools(test_root.string().c_str(),
                                       GP_VOCAB_GROUP_INMEMORY | GP_VOCAB_GROUP_EIGEN,
                                       &report)) {
        std::cerr << "[VOCAB] FAILED: actualization I/O failure\n";
        return 1;
    }
    std::cout << "[VOCAB] actualized: eligible=" << report.ops_eligible
              << " sources=" << report.sources_written
              << " shortfalls=" << report.shortfalls << "\n";
    if (report.sources_written == 0) {
        std::cerr << "[VOCAB] FAILED: no sources generated\n";
        return 1;
    }

    const auto entries = parse_manifest(test_root / "vocab_manifest.txt");
    if (entries.empty()) {
        std::cerr << "[VOCAB] FAILED: manifest empty or unreadable\n";
        return 1;
    }

    fs::path core_lib = find_tensor_core_import_lib(build_dir);
    if (core_lib.empty()) {
        std::cerr << "[VOCAB] FAILED: nodus_tensor_core.lib not found under " << build_dir << "\n";
        return 1;
    }

    // 2) One CMake project, one target per tool, single configure+build.
    const fs::path proj_dir = test_root / "cmake_vocab";
    fs::create_directories(proj_dir);
    {
        std::ostringstream cm;
        cm << "cmake_minimum_required(VERSION 3.15)\n";
        cm << "project(vocab_tools LANGUAGES CXX)\n";
        cm << "find_package(Eigen3 REQUIRED)\n";
        cm << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cm << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
        for (const auto& e : entries) {
            const std::string tgt = sanitize(e.group + "_" + e.tool_id);
            cm << "add_library(" << tgt << " SHARED \"" << e.source.generic_string() << "\")\n";
            cm << "target_include_directories(" << tgt << " PRIVATE \""
               << repo_root.generic_string() << "\" \""
               << (repo_root / "include").generic_string() << "\")\n";
            cm << "target_link_libraries(" << tgt << " PRIVATE \""
               << core_lib.generic_string() << "\"";
            if (e.group == "eigen") cm << " Eigen3::Eigen";
            cm << ")\n";
            cm << "set_target_properties(" << tgt << " PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)\n";
        }
        std::ofstream ofs(proj_dir / "CMakeLists.txt", std::ios::trunc);
        ofs << cm.str();
    }
    const fs::path proj_build = proj_dir / "build";
    const std::string cfg_cmd = "cmake -S \"" + proj_dir.string() + "\" -B \"" +
                                proj_build.string() + "\"";
    const std::string build_cmd = "cmake --build \"" + proj_build.string() +
                                  "\" --config Release --parallel";
    if (!run_command(cfg_cmd) || !run_command(build_cmd)) {
        std::cerr << "[VOCAB] FAILED: building generated tool DLLs\n";
        return 1;
    }

    // 3-5) Load each DLL, execute, verify.
    const auto reference = make_reference();
    const uint64_t kN = 4;
    int pass = 0, fail = 0, skip = 0;
    std::map<std::string, std::vector<float>> inmemory_results; // op -> values

    // Check inmemory first so eigen can cross-check against it.
    std::vector<const VocabEntry*> ordered;
    for (const auto& e : entries) if (e.group == "inmemory") ordered.push_back(&e);
    for (const auto& e : entries) if (e.group != "inmemory") ordered.push_back(&e);

    for (const auto* pe : ordered) {
        const auto& e = *pe;
        auto rit = reference.find(e.op_name);
        if (rit == reference.end()) {
            std::cout << "[VOCAB] SKIP " << e.group << " " << e.op_name
                      << " (no reference registered)\n";
            ++skip;
            continue;
        }
        const RefSpec& spec = rit->second;

        const std::string tgt = sanitize(e.group + "_" + e.tool_id);
        fs::path dll = proj_build / "out" / (tgt + ".dll");
        if (!fs::exists(dll)) dll = proj_build / "out" / "Release" / (tgt + ".dll");
        if (!fs::exists(dll)) dll = proj_build / "Release" / (tgt + ".dll");
        if (!fs::exists(dll)) {
            std::cout << "[VOCAB] FAIL " << e.group << " " << e.op_name
                      << " (DLL not built: " << tgt << ")\n";
            ++fail;
            continue;
        }

        HMODULE mod = LoadLibraryA(dll.string().c_str());
        if (!mod) {
            std::cout << "[VOCAB] FAIL " << e.group << " " << e.op_name
                      << " (LoadLibrary error " << GetLastError() << ")\n";
            ++fail;
            continue;
        }
        using CreateFn = ITool* (*)();
        using DestroyFn = void (*)(ITool*);
        using GroupFn = const char* (*)();
        auto create = reinterpret_cast<CreateFn>(GetProcAddress(mod, "create_tool"));
        auto destroy = reinterpret_cast<DestroyFn>(GetProcAddress(mod, "destroy_tool"));
        auto group_fn = reinterpret_cast<GroupFn>(GetProcAddress(mod, "nodus_tool_backend_group"));
        bool ok = create && destroy;
        std::string why;

        std::vector<float> out_vals;
        if (ok && group_fn && e.group != std::string(group_fn())) {
            ok = false;
            why = "backend group export mismatch";
        }
        if (ok) {
            ITool* tool = create();
            ok = tool != nullptr;
            if (!ok) why = "create_tool returned null";
            if (ok) {
                uint64_t left = make_filled_tensor(spec.a, kN);
                uint64_t right = (spec.arity == 2) ? make_filled_tensor(spec.b, kN) : 0;
                ok = left != 0 && (spec.arity == 1 || right != 0);
                if (!ok) why = "tensor allocation failed";
                if (ok) {
                    RawStackFrame raw{};
                    raw_stack_init_frame(raw, 4096);
                    nodus::tensors::AbstractTensorHandle lh{left}, rh{right};
                    raw_stack_push_abstract_tensor(raw, lh);
                    if (spec.arity == 2) raw_stack_push_abstract_tensor(raw, rh);
                    ToolStackContext tctx{};
                    tctx.stack.raw = &raw;
                    tool->execute_stack(tctx);

                    nodus::tensors::AbstractTensorHandle oh{};
                    ok = raw_stack_pop_abstract_tensor(raw, oh) && oh.id != 0;
                    if (!ok) why = "no output handle on stack";
                    if (ok) {
                        out_vals.resize(kN);
                        ok = nodus_tensor_read(oh.id, 0, out_vals.data(),
                                               out_vals.size() * sizeof(float)) ==
                             static_cast<int64_t>(out_vals.size() * sizeof(float));
                        if (!ok) why = "reading output payload failed";
                        nodus_tensor_destroy(oh.id);
                    }
                    raw_stack_free_mask(raw);
                }
                if (left) nodus_tensor_destroy(left);
                if (right) nodus_tensor_destroy(right);
                tool->shutdown();
                destroy(tool);
            }
        } else if (why.empty()) {
            why = "missing create_tool/destroy_tool export";
        }

        if (ok) {
            const double want = spec.fn(spec.a, spec.b);
            const double tol = std::max(1e-4, std::fabs(want) * 1e-3);
            for (uint64_t i = 0; ok && i < kN; ++i) {
                const double got = out_vals[static_cast<size_t>(i)];
                if (std::isnan(want) ? !std::isnan(got) : std::fabs(got - want) > tol) {
                    ok = false;
                    std::ostringstream w;
                    w << "value mismatch: got " << got << " want " << want;
                    why = w.str();
                }
            }
        }
        if (ok && e.group == "inmemory") {
            inmemory_results[e.op_name] = out_vals;
        }
        if (ok && e.group == "eigen") {
            auto im = inmemory_results.find(e.op_name);
            if (im != inmemory_results.end()) {
                for (uint64_t i = 0; ok && i < kN; ++i) {
                    const float x = im->second[static_cast<size_t>(i)];
                    const float y = out_vals[static_cast<size_t>(i)];
                    const bool same = (std::isnan(x) && std::isnan(y)) ||
                                      std::fabs(x - y) <= 1e-6f;
                    if (!same) {
                        ok = false;
                        std::ostringstream w;
                        w << "engine divergence: eigen " << y << " vs inmemory " << x;
                        why = w.str();
                    }
                }
            }
        }

        FreeLibrary(mod);
        if (ok) {
            std::cout << "[VOCAB] PASS " << e.group << " " << e.op_name << "\n";
            ++pass;
        } else {
            std::cout << "[VOCAB] FAIL " << e.group << " " << e.op_name
                      << " (" << why << ")\n";
            ++fail;
        }
    }

    std::cout << "[VOCAB] summary: pass=" << pass << " fail=" << fail
              << " skip=" << skip << "\n";
    return fail == 0 ? 0 : 1;
#endif
}
