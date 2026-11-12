// The exec tool row in the tool registry.
//
// A canvas binds a tool row by its registry id and gives it no settings, so there is no one "exec" entry: every exec
// row the host declared (exec_row.h, the file named by NODUS_EXEC_ROWS) is its own entry, under its declared id, made
// with its declared program. A host that declared none has no exec rows at all.
//
// The row itself is the plugin source tools/exec_row/exec_row_tool.cpp, compiled into this image the way a composed
// plugin source is: NODUS_PLUGIN_COMPOSITE with its exports renamed.
//
// A declaration the rules refuse (Git, a working directory inside the core, no declared core) is not registered, and
// the reason is said once on the error stream; a canvas that names it then reports a row with no registered tool.
#if defined(_WIN32)

#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_exec_row_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_exec_row_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_exec_row_source
#include "../../tools/exec_row/exec_row_tool.cpp"

#include "exec_row_registry.h"
#include "tool_registry.h"

#include <cstdio>

namespace nodus::exec {

int register_declared_rows(const std::string& path, ToolRegistry& registry, std::string& report) {
    std::vector<DeclaredRow> rows;
    std::string why;
    if (!read_declared_rows(path, rows, why)) {
        report += "exec rows: " + why + "\n";
        return 0;
    }
    int registered = 0;
    for (const DeclaredRow& row : rows) {
        Program program;
        if (!row_program(row.settings, program, why)) {
            report += "exec rows: '" + row.id + "' is not registered: " + why + "\n";
            continue;
        }
        ToolRegistry::Entry entry;
        entry.id = row.id;
        entry.name = row.name;
        entry.caps = ToolCaps::None;
        entry.source_path = nodus_exec_row_source();
        const RowSettings settings = row.settings;
        entry.factory = [settings] {
            auto* tool = new ExecRowTool();
            tool->configure(settings);
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(tool, [](ITool* made) { nodus_exec_row_destroy(made); });
        };
        if (registry.register_tool(std::move(entry))) {
            ++registered;
            report += "exec rows: '" + row.id + "' registered (" + row.settings.program + ")\n";
        } else {
            report += "exec rows: '" + row.id + "' is not registered: the id is already taken\n";
        }
    }
    return registered;
}

} // namespace nodus::exec

namespace {

// At image load: the rows the host declared for this process.
struct DeclaredExecRows {
    DeclaredExecRows() {
        const std::string path = nodus::exec::declared_rows_file();
        if (path.empty()) return;
        std::string report;
        nodus::exec::register_declared_rows(path, tool_registry_global(), report);
        std::fputs(report.c_str(), stderr);
    }
};
const DeclaredExecRows g_declared_exec_rows;

} // namespace

#endif // _WIN32
