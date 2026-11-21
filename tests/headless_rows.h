// headless_rows.h: rows for a headless test, built and loaded inside the headless runtime the test runs on.
//
// Review f1a3eb8f (operator-codex): the service path is nodus_headless, so a service-path test builds its rows with
// nodus_headless_build_plugin_source, as volunteer_cycle_link_test does: the plugin is loaded by the same headless image
// and registered in the same tool registry the graph uses. A test never loads a plugin through the canvas library's own
// loader (gp_plugin_load_from_path), which nodus_headless.dll does not export.
#pragma once

#include "nodus_headless_abi.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace headless_rows {

// Builds one row source into <library_root>/bin and loads it; returns its tool id, empty when it did not build.
inline std::string build(const std::filesystem::path& source, const std::filesystem::path& library_root) {
    std::filesystem::create_directories(library_root);
    char id[512] = {};
    if (!nodus_headless_build_plugin_source(source.string().c_str(), NODUS_REPO_ROOT, library_root.string().c_str(), id,
                                            static_cast<int32_t>(sizeof(id)))) {
        std::fprintf(stderr, "row %s did not build\n", source.string().c_str());
        return {};
    }
    return id;
}

// A module library row by name: module_library/source/tools/<name>.cpp of this checkout.
inline std::string build_named(const char* name, const std::filesystem::path& library_root) {
    return build(std::filesystem::path(NODUS_REPO_ROOT) / "module_library" / "source" / "tools" / (std::string(name) + ".cpp"),
                 library_root);
}

}  // namespace headless_rows
