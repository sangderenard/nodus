// exec_row: what an exec tool row is told to run, and how that is saved.
//
// The row itself is tools/exec_row/exec_row_tool.cpp, an ordinary ITool plugin source. This header is the part the
// row and whoever places the row both need: the settings, their saved form (the tool's own serialize/deserialize
// archive, read from ToolInitContext::serialized_path), and the host's declaration of the core.
//
// The core roots are declared by the host, never by the row: they come from the environment variable
// NODUS_EXEC_CORE_ROOTS (paths separated by ';') that the process hosting Nodus is started with. A row's saved
// settings cannot add to them or leave them out, and a host that declared none gets a row that starts nothing.
#pragma once

#include "exec_pipe.h"
#include "tool_api.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace nodus::exec {

inline constexpr uint32_t kRowSettingsMagic = 0x57525845u;   // "EXRW"
inline constexpr uint32_t kRowSettingsVersion = 1;
inline constexpr uint32_t kRowSettingsLimit = 1u << 16;      // longest string and most arguments a saved form may hold

struct RowSettings {
    std::string program;                  // the executable
    std::vector<std::string> arguments;   // it must read standard input to the end and write its result to standard output
    std::string working_dir;              // outside the core
    uint32_t most_at_once = 2;            // children working at the same time; further spans wait their turn in order
};

inline bool write_string(OutputArchive& out, const std::string& text) {
    const uint32_t length = static_cast<uint32_t>(text.size());
    return out.write(&length, sizeof(length)) && (length == 0 || out.write(text.data(), length));
}

inline bool read_string(InputArchive& in, std::string& text) {
    uint32_t length = 0;
    if (!in.read(&length, sizeof(length)) || length > kRowSettingsLimit) return false;
    text.assign(length, '\0');
    return length == 0 || in.read(text.data(), length);
}

inline bool write_settings(OutputArchive& out, const RowSettings& settings) {
    const uint32_t head[3] = {kRowSettingsMagic, kRowSettingsVersion, static_cast<uint32_t>(settings.arguments.size())};
    if (!out.write(head, sizeof(head)) || !write_string(out, settings.program)) return false;
    for (const std::string& argument : settings.arguments)
        if (!write_string(out, argument)) return false;
    return write_string(out, settings.working_dir) && out.write(&settings.most_at_once, sizeof(settings.most_at_once));
}

inline bool read_settings(InputArchive& in, RowSettings& settings) {
    uint32_t head[3] = {0, 0, 0};
    if (!in.read(head, sizeof(head)) || head[0] != kRowSettingsMagic || head[1] != kRowSettingsVersion ||
        head[2] > kRowSettingsLimit)
        return false;
    RowSettings read;
    if (!read_string(in, read.program)) return false;
    read.arguments.resize(head[2]);
    for (std::string& argument : read.arguments)
        if (!read_string(in, argument)) return false;
    if (!read_string(in, read.working_dir) || !in.read(&read.most_at_once, sizeof(read.most_at_once))) return false;
    if (read.most_at_once == 0) read.most_at_once = 1;
    settings = std::move(read);
    return true;
}

// The core repositories as the host declared them for this process. Empty when the host declared none.
inline std::vector<std::string> declared_core_roots() {
    std::vector<std::string> roots;
    // The process environment itself, not a C runtime's copy of it: a row built as its own DLL has its own runtime.
    const DWORD needed = GetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", nullptr, 0);
    if (needed == 0) return roots;
    std::string all(needed, '\0');
    all.resize(GetEnvironmentVariableA("NODUS_EXEC_CORE_ROOTS", all.data(), needed));
    size_t start = 0;
    while (start <= all.size()) {
        const size_t end = all.find(';', start);
        const std::string one = all.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!one.empty()) roots.push_back(one);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return roots;
}

// The program a row with these settings would start, held to the host's declaration of the core.
inline bool row_program(const RowSettings& settings, Program& program, std::string& why) {
    program = Program{settings.program, settings.arguments, settings.working_dir, declared_core_roots()};
    if (settings.program.empty()) { why = "the row has no program"; return false; }
    if (program.forbidden_roots.empty()) {
        why = "the host declared no core (NODUS_EXEC_CORE_ROOTS); a row starts nothing until it does";
        return false;
    }
    why = refusal(program);
    return why.empty();
}

} // namespace nodus::exec
