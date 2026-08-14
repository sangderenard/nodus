// artifact_import_test -- Rung 2 of the boundary fluency campaign
// (BOUNDARY_FLUENCY_CAMPAIGN_2026-08-13.md).
//
// First admission of a turing-compiled artifact into the nodus swap pool
// through the substitution gate:
//   1. gp_artifact_import parses the fixture's turing-compiled-program-api-v1
//      contract and generates the ITool wrapper + NODUSPKG manifest.
//   2. The wrapper builds into a tool DLL (one generated CMake project),
//      linking nodus_tensor_core so host, wrapper, and artifact share one
//      arena (error register E5).
//   3. The tool executes against real F64 tensors pushed as
//      VT_ABSTRACT_TENSOR handles in the SELECTED ENTRY's declared input
//      order (E11: entry points of one artifact legitimately disagree on
//      parameter order; the contract's order is the call order).
//   4. The result is verified against an independently computed reference,
//      and a substitution certificate is written beside the package --
//      admission as a stored, auditable fact.
//
// Fixture: blend_signal(a, b, scale) = m - m^3/3, m = a*scale + b --
// boundary-clean pure arithmetic (its boundary-dirty twin is error E7).
#include "artifact_importer.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "tool_api.h"
#include "value_types.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace {

bool run_command(const std::string& cmd) {
    return std::system(cmd.c_str()) == 0;
}

fs::path find_tensor_core_import_lib(const fs::path& build_dir) {
    const fs::path release = build_dir / "Release" / "nodus_tensor_core.lib";
    if (fs::exists(release)) return release;
    const fs::path debug = build_dir / "Debug" / "nodus_tensor_core.lib";
    if (fs::exists(debug)) return debug;
    return {};
}

uint64_t make_scalar_f64(double value) {
    const uint64_t shape[1] = {1};
    uint64_t h = nodus_tensor_create(NODUS_DTYPE_F64, 1, shape);
    if (!h) return 0;
    if (nodus_tensor_write(h, 0, &value, sizeof(double)) !=
        static_cast<int64_t>(sizeof(double))) {
        nodus_tensor_destroy(h);
        return 0;
    }
    return h;
}

} // namespace

int main() {
#ifndef _WIN32
    std::cerr << "artifact_import_test skipped: Windows-only DLL loading here.\n";
    return 0;
#else
    const fs::path repo_root = fs::path(NODUS_REPO_ROOT);
    const fs::path build_dir = fs::path(NODUS_BUILD_DIR);
    const fs::path fixture_dir =
        repo_root / "module_library" / "test_build" / "artifact_import_fixture";
    const fs::path api_yaml = fixture_dir / "blend_signal.api.yaml";
    const fs::path artifact = fixture_dir / "blend_signal.dll";

    if (!fs::exists(api_yaml) || !fs::exists(artifact)) {
        std::cerr << "artifact_import_test skipped: fixture not present ("
                  << api_yaml.generic_string() << ").\n"
                  << "Regenerate with: cd turing && python -m "
                     "src.compiler.compile_section_to_dll "
                     "build/nodus_import_fixture/fixture_module.py blend_signal "
                  << fixture_dir.generic_string() << "\n";
        return 0;
    }

    const fs::path out_root = fixture_dir / "import_out";
    fs::create_directories(out_root);

    // 1) Import: contract -> wrapper + package.
    GP_ArtifactImportReport report{};
    char pkg_path[512] = {0};
    if (!gp_artifact_import(api_yaml.string().c_str(), artifact.string().c_str(),
                            out_root.string().c_str(), &report, pkg_path,
                            sizeof(pkg_path))) {
        std::cerr << "[IMPORT] FAILED: gp_artifact_import (entry_points="
                  << report.entry_points_seen << " shortfalls="
                  << report.shortfalls << ")\n";
        return 1;
    }
    std::cout << "[IMPORT] wrapper generated; package: " << pkg_path << "\n";

    const fs::path wrapper_src =
        out_root / "source" / "tools" / "imported" / "imported_blend_signal_fortran.cpp";
    if (!fs::exists(wrapper_src)) {
        std::cerr << "[IMPORT] FAILED: expected wrapper source not found: "
                  << wrapper_src.generic_string() << "\n";
        return 1;
    }

    // 2) Build the wrapper into a tool DLL.
    fs::path core_lib = find_tensor_core_import_lib(build_dir);
    if (core_lib.empty()) {
        std::cerr << "[IMPORT] FAILED: nodus_tensor_core.lib not found\n";
        return 1;
    }
    const fs::path proj_dir = out_root / "cmake_wrapper";
    fs::create_directories(proj_dir);
    {
        std::ostringstream cm;
        cm << "cmake_minimum_required(VERSION 3.15)\n";
        cm << "project(imported_blend_signal LANGUAGES CXX)\n";
        cm << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY \"${CMAKE_BINARY_DIR}/out\")\n";
        cm << "set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE \"${CMAKE_BINARY_DIR}/out\")\n";
        cm << "add_library(imported_blend_signal SHARED \""
           << wrapper_src.generic_string() << "\")\n";
        cm << "target_include_directories(imported_blend_signal PRIVATE \""
           << repo_root.generic_string() << "\" \""
           << (repo_root / "include").generic_string() << "\")\n";
        cm << "target_link_libraries(imported_blend_signal PRIVATE \""
           << core_lib.generic_string() << "\")\n";
        cm << "set_target_properties(imported_blend_signal PROPERTIES "
              "CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)\n";
        std::ofstream ofs(proj_dir / "CMakeLists.txt", std::ios::trunc);
        ofs << cm.str();
    }
    const fs::path proj_build = proj_dir / "build";
    if (!run_command("cmake -S \"" + proj_dir.string() + "\" -B \"" +
                     proj_build.string() + "\"") ||
        !run_command("cmake --build \"" + proj_build.string() +
                     "\" --config Release --parallel")) {
        std::cerr << "[IMPORT] FAILED: building the generated wrapper\n";
        return 1;
    }
    fs::path wrapper_dll = proj_build / "out" / "imported_blend_signal.dll";
    if (!fs::exists(wrapper_dll))
        wrapper_dll = proj_build / "out" / "Release" / "imported_blend_signal.dll";
    if (!fs::exists(wrapper_dll)) {
        std::cerr << "[IMPORT] FAILED: wrapper DLL not found under "
                  << proj_build.generic_string() << "\n";
        return 1;
    }

    // 3) Load and execute through the plugin ABI.
    HMODULE mod = LoadLibraryA(wrapper_dll.string().c_str());
    if (!mod) {
        std::cerr << "[IMPORT] FAILED: LoadLibrary error " << GetLastError() << "\n";
        return 1;
    }
    using CreateFn = ITool* (*)();
    using DestroyFn = void (*)(ITool*);
    auto create = reinterpret_cast<CreateFn>(GetProcAddress(mod, "create_tool"));
    auto destroy = reinterpret_cast<DestroyFn>(GetProcAddress(mod, "destroy_tool"));
    if (!create || !destroy) {
        std::cerr << "[IMPORT] FAILED: wrapper missing plugin exports\n";
        return 1;
    }
    ITool* tool = create();
    if (!tool) {
        std::cerr << "[IMPORT] FAILED: create_tool returned null (E2)\n";
        return 1;
    }
    ToolInitContext ictx{};
    tool->initialize(ictx); // loads the artifact + resolves the entry symbol

    const double a = 0.7, b = -0.3, scale = 1.4;
    const double mixed = a * scale + b;
    const double want = mixed - (mixed * mixed * mixed) / 3.0; // 0.5751893333
    const double tol = 1e-9;

    uint64_t ha = make_scalar_f64(a);
    uint64_t hb = make_scalar_f64(b);
    uint64_t hs = make_scalar_f64(scale);
    bool ok = ha && hb && hs;
    std::string why = ok ? "" : "tensor allocation failed";

    double got = 0.0;
    if (ok) {
        RawStackFrame raw{};
        raw_stack_init_frame(raw, 4096);
        // Push in the selected (control) entry's declared input order:
        // t0=a, t2=b, t1=scale (E11 -- order is the entry's, not intuition's).
        nodus::tensors::AbstractTensorHandle h{};
        h.id = ha; raw_stack_push_abstract_tensor(raw, h);
        h.id = hb; raw_stack_push_abstract_tensor(raw, h);
        h.id = hs; raw_stack_push_abstract_tensor(raw, h);
        ToolStackContext tctx{};
        tctx.stack.raw = &raw;
        tool->execute_stack(tctx);

        nodus::tensors::AbstractTensorHandle out{};
        ok = raw_stack_pop_abstract_tensor(raw, out) && out.id != 0;
        if (!ok) {
            why = "no output handle on stack (check E1/E4 before the wrapper)";
        } else if (out.id == ha || out.id == hb || out.id == hs) {
            // Identity check: a tool that bails without consuming leaves the
            // inputs on the stack, and a naive pop reads its own input back
            // as "output" -- exactly how a silent LoadLibrary failure first
            // presented (as got==scale). Never trust an output that is one
            // of the inputs.
            ok = false;
            why = "output handle IS an input handle -- wrapper did not execute "
                  "(likely artifact load/entry failure; see stderr)";
        } else {
            ok = nodus_tensor_read(out.id, 0, &got, sizeof(double)) ==
                 static_cast<int64_t>(sizeof(double));
            if (!ok) why = "reading output payload failed";
            nodus_tensor_destroy(out.id);
        }
        raw_stack_free_mask(raw);
    }
    if (ha) nodus_tensor_destroy(ha);
    if (hb) nodus_tensor_destroy(hb);
    if (hs) nodus_tensor_destroy(hs);

    if (ok && std::fabs(got - want) > tol) {
        ok = false;
        std::ostringstream w;
        w << "value mismatch: got " << got << " want " << want;
        why = w.str();
    }

    // Negative control: the gate must catch a wrong witness. Feed inputs whose
    // correct output differs, against the ORIGINAL expectation -- if this
    // "passes", the harness is broken, not the artifact.
    if (ok) {
        uint64_t na = make_scalar_f64(a + 1.0);
        uint64_t nb = make_scalar_f64(b);
        uint64_t ns = make_scalar_f64(scale);
        if (na && nb && ns) {
            RawStackFrame raw{};
            raw_stack_init_frame(raw, 4096);
            nodus::tensors::AbstractTensorHandle h{};
            h.id = na; raw_stack_push_abstract_tensor(raw, h);
            h.id = nb; raw_stack_push_abstract_tensor(raw, h);
            h.id = ns; raw_stack_push_abstract_tensor(raw, h);
            ToolStackContext tctx{};
            tctx.stack.raw = &raw;
            tool->execute_stack(tctx);
            nodus::tensors::AbstractTensorHandle out{};
            double wrong_got = 0.0;
            if (raw_stack_pop_abstract_tensor(raw, out) && out.id != 0 &&
                nodus_tensor_read(out.id, 0, &wrong_got, sizeof(double)) ==
                    static_cast<int64_t>(sizeof(double))) {
                if (std::fabs(wrong_got - want) <= tol) {
                    ok = false;
                    why = "negative control failed: perturbed inputs matched "
                          "the reference -- the harness cannot distinguish "
                          "right from wrong";
                }
                nodus_tensor_destroy(out.id);
            }
            raw_stack_free_mask(raw);
        }
        if (na) nodus_tensor_destroy(na);
        if (nb) nodus_tensor_destroy(nb);
        if (ns) nodus_tensor_destroy(ns);
    }

    tool->shutdown();
    destroy(tool);
    FreeLibrary(mod);

    if (!ok) {
        std::cout << "[IMPORT] FAIL blend_signal (" << why << ")\n";
        return 1;
    }

    // 4) Substitution certificate: admission as a stored fact.
    {
        std::ostringstream cert;
        cert << "SUBSTITUTION-CERTIFICATE V1\n";
        cert << "tool: imported.blend_signal_fortran\n";
        cert << "backend_group: compiled-fortran\n";
        cert << "artifact: " << artifact.generic_string() << "\n";
        cert << "artifact_bytes: " << fs::file_size(artifact) << "\n";
        cert << "contract: " << api_yaml.generic_string() << "\n";
        cert << "entry: blend_signal__blend_signal (control)\n";
        cert << "battery: a=0.7 b=-0.3 scale=1.4 (declared order t0,t2,t1)\n";
        cert << "reference: independent host arithmetic m-(m^3)/3, m=a*scale+b\n";
        cert << "expected: " << want << "\n";
        cert << "observed: " << got << "\n";
        cert << "tolerance: " << tol << "\n";
        cert << "negative_control: perturbed-input mismatch verified\n";
        std::ofstream ofs(out_root / "blend_signal.substitution_certificate.txt",
                          std::ios::trunc);
        ofs << cert.str();
    }

    std::cout << "[IMPORT] PASS blend_signal: got " << got << " (want " << want
              << "); certificate written\n";
    return 0;
#endif
}
