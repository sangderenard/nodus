#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Third tier of nodus tool creation: an external repo's whole coherent
// package -- tools *and* the type/edge contracts they need -- ingested as
// one named, listable, unloadable-as-a-unit bundle. Sibling to
// GP_ModuleLibrary (module_library.h), but not keyed by the closed
// ModuleToolKind enum: entries here are arbitrary hand-written tool sources
// or prebuilt DLLs, honestly tagged via ToolCapabilityTag (tool_api.h)
// rather than gated on IR-translatability.

struct GP_RepoPackageTool {
    std::string id;
    std::string name;
    std::string source_or_dll_path;   // hand-written .cpp, or a prebuilt DLL path
    uint32_t caps = 0;                // ToolCaps
    uint32_t capability_tags = 0;     // CAP_ISA | CAP_BINARY | CAP_BACKWARD | ... (ToolCapabilityTag)
    std::string notes;                // free-text caveats, e.g. "forward-only, single-threaded"
};

struct GP_RepoPackageContract {
    // A register_types() source (no create_tool required). Built and loaded
    // as its own standalone DLL, so register_types() only reaches *that
    // DLL's own* copy of ValueTypeRegistry, not the host's -- see the scope
    // note on NODUS_PLUGIN_REGISTER_TYPES_NAME in tool_api.h. Contracts are
    // for the rare case where a package's tools need to agree on a local
    // struct layout among themselves; ingestion runs register_types() (so a
    // contract can still self-register into its own DLL for its own tools'
    // internal use, or simply validate it loads and runs without throwing)
    // but does not assume or require that it becomes visible to the host's
    // registry. Prefer opaque (VT_VOID_PTR) or tensor-shaped
    // (VT_ABSTRACT_TENSOR) data over reaching for a contract at all.
    std::string source_path;
};

struct GP_RepoPackage {
    std::string owner;                // e.g. "spectral-analyzer"; the unit gp_repo_package_unload targets
    std::string version;
    std::string root_dir;             // root of the owning repo; relative tool/contract paths resolve against this
    std::vector<GP_RepoPackageTool> tools;
    std::vector<GP_RepoPackageContract> contracts;
};

// NODUSPKG V1 text format read/write, mirroring gp_module_library_read_from_file/write_to_file.
int gp_repo_package_write_to_file(const GP_RepoPackage& package, const char* path);
int gp_repo_package_read_from_file(const char* path, GP_RepoPackage* out_package);

// Build any tool/contract source entries that aren't already a prebuilt DLL
// (`source_or_dll_path`/`source_path` ending in ".dll"), load every
// resulting DLL, call every contract's register_types(), and track the
// whole bundle under `package.owner` as one named unit. `host` is forwarded
// to each tool the same way PluginLoader::load_module forwards it to
// plugin_init. Fails (returns 0) if `package.owner` is already ingested, or
// if any tool/contract fails to build or load -- whatever this call loaded
// itself is unwound before returning, so a failed ingest leaves nothing
// behind.
int gp_repo_package_ingest(const GP_RepoPackage& package, void* host = nullptr);
int gp_repo_package_ingest_from_file(const char* path, void* host = nullptr);

// Unload every tool/contract `owner`'s package loaded, together. Returns 1
// on success, 0 if `owner` isn't a currently-ingested package.
int gp_repo_package_unload(const std::string& owner);

// Owners of currently-ingested packages.
std::vector<std::string> gp_repo_package_list();
