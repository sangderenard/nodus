#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "repo_package.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "plugin_manager.h"

namespace fs = std::filesystem;

static bool write_text_file(const fs::path& path, const std::string& contents) {
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.good()) return false;
    ofs << contents;
    return ofs.good();
}

// register_types() here registers into *this DLL's own* copy of
// ValueTypeRegistry (a Meyer's singleton in an inline header function isn't
// shared across a real DLL boundary -- see tool_api.h's scope note on
// NODUS_PLUGIN_REGISTER_TYPES_NAME). That's still exercised below to prove
// the export loads and runs without throwing, but the host process can't
// observe it through ValueTypeRegistry::global(), so it also drops a marker
// file the test can check for from outside the DLL.
static const char* kContractSource = R"(
#include "tool_api.h"
#include <cstdlib>
#include <fstream>
#include <vector>

extern "C" NODUS_PLUGIN_EXPORT void register_types() {
    ValueTypeId f32 = ValueTypeRegistry::global().builtin(VT_FLOAT32);
    std::vector<ValueField> fields;
    fields.push_back(ValueField{"x", 0, static_cast<uint32_t>(sizeof(float)), f32});
    fields.push_back(ValueField{"y", static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(sizeof(float)), f32});
    ValueTypeRegistry::global().register_struct("RepoPackageTestPoint2D", fields, static_cast<uint32_t>(sizeof(float) * 2));

    const char* marker = std::getenv("REPO_PACKAGE_TEST_MARKER");
    if (marker) {
        std::ofstream ofs(marker, std::ios::binary | std::ios::trunc);
        ofs << "ran";
    }
}
)";

static const char* kToolSource = R"(
#include "tool_api.h"

class RepoPackageTestTool : public ITool {
public:
    const char* id_cstr() const noexcept override { return "repo_package_test_tool"; }
    const char* name_cstr() const noexcept override { return "Repo Package Test Tool"; }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
};

extern "C" NODUS_PLUGIN_EXPORT ITool* NODUS_PLUGIN_FACTORY_NAME() { return new RepoPackageTestTool(); }
extern "C" NODUS_PLUGIN_EXPORT void NODUS_PLUGIN_DESTROY_NAME(ITool* t) { delete t; }
extern "C" NODUS_PLUGIN_EXPORT const char* NODUS_PLUGIN_SOURCE_NAME() { return __FILE__; }
)";

int main() {
#ifndef _WIN32
    std::cerr << "repo_package_test skipped: repo package ingestion only implemented on Windows.\n";
    return 0;
#endif

    const fs::path repo_root = fs::path(NODUS_REPO_ROOT);
    fs::path test_root = repo_root / "module_library" / "test_build" / "repo_package_test";
    std::error_code ec;
    fs::remove_all(test_root, ec);
    fs::create_directories(test_root);

    if (!write_text_file(test_root / "contract.cpp", kContractSource)) {
        std::cerr << "failed to write fixture contract source\n";
        return 1;
    }
    if (!write_text_file(test_root / "tool.cpp", kToolSource)) {
        std::cerr << "failed to write fixture tool source\n";
        return 1;
    }

    // --- roundtrip test ---
    GP_RepoPackage package{};
    package.owner = "repo_package_test_owner";
    package.version = "0.1";
    package.root_dir = test_root.generic_string();
    GP_RepoPackageTool tool{};
    tool.id = "repo_package_test_tool";
    tool.name = "Repo Package Test Tool";
    tool.source_or_dll_path = "tool.cpp";
    tool.caps = 0;
    tool.capability_tags = CAP_BINARY;
    tool.notes = "test fixture, forward-only";
    package.tools.push_back(tool);
    GP_RepoPackageContract contract{};
    contract.source_path = "contract.cpp";
    package.contracts.push_back(contract);

    fs::path manifest_path = test_root / "nodus_package.txt";
    if (!gp_repo_package_write_to_file(package, manifest_path.string().c_str())) {
        std::cerr << "gp_repo_package_write_to_file failed\n";
        return 1;
    }

    GP_RepoPackage roundtripped{};
    if (!gp_repo_package_read_from_file(manifest_path.string().c_str(), &roundtripped)) {
        std::cerr << "gp_repo_package_read_from_file failed\n";
        return 1;
    }
    if (roundtripped.owner != package.owner || roundtripped.version != package.version ||
        roundtripped.root_dir != package.root_dir || roundtripped.tools.size() != 1 ||
        roundtripped.contracts.size() != 1) {
        std::cerr << "roundtripped package does not match original\n";
        return 1;
    }
    if (roundtripped.tools[0].id != tool.id || roundtripped.tools[0].name != tool.name ||
        roundtripped.tools[0].source_or_dll_path != tool.source_or_dll_path ||
        roundtripped.tools[0].capability_tags != tool.capability_tags ||
        roundtripped.tools[0].notes != tool.notes) {
        std::cerr << "roundtripped tool entry does not match original\n";
        return 1;
    }
    if (roundtripped.contracts[0].source_path != contract.source_path) {
        std::cerr << "roundtripped contract entry does not match original\n";
        return 1;
    }
    std::cerr << "roundtrip ok\n";

    // --- ingestion test ---
    fs::path marker_path = test_root / "contract_ran.marker";
    fs::remove(marker_path, ec);
#ifdef _WIN32
    _putenv_s("REPO_PACKAGE_TEST_MARKER", marker_path.string().c_str());
#endif

    if (!gp_repo_package_ingest_from_file(manifest_path.string().c_str())) {
        std::cerr << "gp_repo_package_ingest_from_file failed\n";
        return 1;
    }

    // The host process can't observe the contract's register_types() through
    // ValueTypeRegistry::global() -- see the comment on kContractSource.
    // Confirm the contract DLL actually loaded and ran via the marker file
    // it drops instead.
    if (!fs::exists(marker_path)) {
        std::cerr << "contract's register_types() did not run (marker file missing)\n";
        return 1;
    }

    auto owners = gp_repo_package_list();
    bool found_owner = false;
    for (const auto& o : owners) {
        if (o == package.owner) found_owner = true;
    }
    if (!found_owner) {
        std::cerr << "gp_repo_package_list() does not report ingested owner\n";
        return 1;
    }

    char id_buf[4096] = {0};
    int loaded_before = gp_plugin_list_ids(id_buf, sizeof(id_buf));
    if (loaded_before <= 0) {
        std::cerr << "expected at least one loaded plugin tool after ingest\n";
        return 1;
    }

    if (!gp_repo_package_unload(package.owner)) {
        std::cerr << "gp_repo_package_unload failed\n";
        return 1;
    }

    owners = gp_repo_package_list();
    for (const auto& o : owners) {
        if (o == package.owner) {
            std::cerr << "owner still listed after unload\n";
            return 1;
        }
    }

    if (gp_repo_package_unload(package.owner)) {
        std::cerr << "unloading an already-unloaded owner unexpectedly succeeded\n";
        return 1;
    }

    std::cerr << "ingestion ok\n";
    return 0;
}
