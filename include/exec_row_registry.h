// Registering the host's declared exec rows (exec_row.h) as tool registry entries. Done once at image load for the
// file named by NODUS_EXEC_ROWS; a host that keeps its declarations elsewhere calls this itself.
#pragma once

#include <string>

class ToolRegistry;

namespace nodus::exec {

// Registers every row the file declares that the exec rules allow. Returns how many were registered and appends one
// line per row (registered, or why not) to `report`.
int register_declared_rows(const std::string& path, ToolRegistry& registry, std::string& report);

} // namespace nodus::exec
