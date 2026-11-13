// exec_row: what an exec tool row is told to run, and how that is saved.
//
// The row itself is tools/exec_row/exec_row_tool.cpp, an ordinary ITool plugin source. This header is the part the
// row and whoever places the row both need: the settings, their saved form (the tool's own serialize/deserialize
// archive, read from ToolInitContext::serialized_path), and the host's declaration of the core.
//
// The core roots are declared by the host, never by the row: they come from the environment variable
// NODUS_EXEC_CORE_ROOTS (paths separated by ';') that the process hosting Nodus is started with. A row's saved
// settings cannot add to them or leave them out, and a host that declared none gets a row that starts nothing.
//
// The places people are meant to work, the untracked user folders and checkout leases, are declared the same way in
// NODUS_EXEC_WORK_ROOTS. A row's working directory inside one of them is allowed though it is inside the core.
//
// Declared rows. A canvas binds a tool row by its registry id alone and hands it no settings, so an exec row that can
// be placed in a canvas is one the host declared: the environment variable NODUS_EXEC_ROWS names a text file, and
// each row in it becomes one registry entry (src/tools/exec_row_registered.cpp). The file is blocks of
// "key value" lines, the value running to the end of the line; a "row" line starts a block:
//
//   row exec.zstd.pack                 the registry id; always under "exec."
//   name Zstd pack                     shown in the tool menu (the id when left out)
//   program C:/tools/zstd/zstd.exe
//   argument -c                        any number, in order
//   directory C:/scratch/exec          the working directory, outside the core
//   most 2                             children working at once (2 when left out)
//   mode stream                        keep one child's stdin open across spans; no return tensor
//   queue_bytes 1048576                maximum pending input bytes in stream mode
//
// Blank lines and lines that start with '#' are skipped. Anything else stops the reading with the line named.
#pragma once

#include "exec_pipe.h"
#include "tool_api.h"

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace nodus::exec {

inline constexpr uint32_t kRowSettingsMagic = 0x57525845u;   // "EXRW"
inline constexpr uint32_t kRowSettingsVersion = 2;
inline constexpr uint32_t kRowSettingsLimit = 1u << 16;      // longest string and most arguments a saved form may hold

struct RowSettings {
    std::string program;                  // the executable
    std::vector<std::string> arguments;   // it must read standard input to the end and write its result to standard output
    std::string working_dir;              // outside the core
    uint32_t most_at_once = 2;            // children working at the same time; further spans wait their turn in order
    bool stream = false;                   // one child receives successive spans without stdin EOF
    uint32_t queue_bytes = 1u << 20;      // bounded pending input for a streaming child
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
    const uint32_t stream = settings.stream ? 1u : 0u;
    return write_string(out, settings.working_dir) && out.write(&settings.most_at_once, sizeof(settings.most_at_once)) &&
           out.write(&stream, sizeof(stream)) && out.write(&settings.queue_bytes, sizeof(settings.queue_bytes));
}

inline bool read_settings(InputArchive& in, RowSettings& settings) {
    uint32_t head[3] = {0, 0, 0};
    if (!in.read(head, sizeof(head)) || head[0] != kRowSettingsMagic || (head[1] != 1 && head[1] != kRowSettingsVersion) ||
        head[2] > kRowSettingsLimit)
        return false;
    RowSettings read;
    if (!read_string(in, read.program)) return false;
    read.arguments.resize(head[2]);
    for (std::string& argument : read.arguments)
        if (!read_string(in, argument)) return false;
    if (!read_string(in, read.working_dir) || !in.read(&read.most_at_once, sizeof(read.most_at_once))) return false;
    if (read.most_at_once == 0) read.most_at_once = 1;
    if (head[1] == kRowSettingsVersion) {
        uint32_t stream = 0;
        if (!in.read(&stream, sizeof(stream)) || stream > 1 || !in.read(&read.queue_bytes, sizeof(read.queue_bytes)) ||
            read.queue_bytes == 0) return false;
        read.stream = stream != 0;
    }
    settings = std::move(read);
    return true;
}

// Paths separated by ';' from one variable of the process environment. Empty when it is not set.
inline std::vector<std::string> environment_paths(const char* variable) {
    std::vector<std::string> roots;
    // The process environment itself, not a C runtime's copy of it: a row built as its own DLL has its own runtime.
    const DWORD needed = GetEnvironmentVariableA(variable, nullptr, 0);
    if (needed == 0) return roots;
    std::string all(needed, '\0');
    all.resize(GetEnvironmentVariableA(variable, all.data(), needed));
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

// The core repositories as the host declared them for this process. Empty when the host declared none.
inline std::vector<std::string> declared_core_roots() { return environment_paths("NODUS_EXEC_CORE_ROOTS"); }

// The user folders and checkout leases the host declared for this process.
inline std::vector<std::string> declared_work_roots() { return environment_paths("NODUS_EXEC_WORK_ROOTS"); }

struct DeclaredRow {
    std::string id;
    std::string name;
    RowSettings settings;
};

// The rows a declarations file holds. False, with the reason, when the file cannot be read as one.
inline bool read_declared_rows(const std::string& path, std::vector<DeclaredRow>& rows, std::string& why) {
    std::ifstream file(path);
    if (!file.good()) { why = "the declarations file cannot be opened: " + path; return false; }
    std::vector<DeclaredRow> read;
    std::string line;
    for (int number = 1; std::getline(file, line); ++number) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) line.pop_back();
        const size_t first = line.find_first_not_of(" \t");
        if (first == std::string::npos || line[first] == '#') continue;
        const size_t gap = line.find_first_of(" \t", first);
        const std::string key = line.substr(first, gap == std::string::npos ? std::string::npos : gap - first);
        const size_t at = gap == std::string::npos ? std::string::npos : line.find_first_not_of(" \t", gap);
        const std::string value = at == std::string::npos ? std::string() : line.substr(at);
        const std::string where = "line " + std::to_string(number) + " of " + path + ": ";
        if (value.empty()) { why = where + "'" + key + "' has no value"; return false; }
        if (key == "row") {
            if (value.rfind("exec.", 0) != 0 || value.size() == 5) { why = where + "a declared row's id is under 'exec.'"; return false; }
            for (const DeclaredRow& other : read)
                if (other.id == value) { why = where + "row '" + value + "' is declared twice"; return false; }
            read.push_back(DeclaredRow{value, value, RowSettings{}});
            continue;
        }
        if (read.empty()) { why = where + "'" + key + "' comes before any 'row'"; return false; }
        DeclaredRow& row = read.back();
        if (key == "name") row.name = value;
        else if (key == "program") row.settings.program = value;
        else if (key == "argument") row.settings.arguments.push_back(value);
        else if (key == "directory") row.settings.working_dir = value;
        else if (key == "mode") {
            if (value != "span" && value != "stream") { why = where + "'mode' is span or stream"; return false; }
            row.settings.stream = value == "stream";
        }
        else if (key == "queue_bytes") {
            char* end = nullptr;
            const unsigned long count = std::strtoul(value.c_str(), &end, 10);
            if (!end || *end != '\0' || count == 0 || count > (64ul << 20)) {
                why = where + "'queue_bytes' is from 1 to 67108864"; return false;
            }
            row.settings.queue_bytes = static_cast<uint32_t>(count);
        }
        else if (key == "most") {
            char* end = nullptr;
            const unsigned long most = std::strtoul(value.c_str(), &end, 10);
            if (!end || *end != '\0' || most == 0 || most > 64) { why = where + "'most' is a count from 1 to 64"; return false; }
            row.settings.most_at_once = static_cast<uint32_t>(most);
        } else { why = where + "'" + key + "' is not a declaration"; return false; }
    }
    rows = std::move(read);
    return true;
}

// The declarations file the host named for this process; empty when it named none.
inline std::string declared_rows_file() {
    const DWORD needed = GetEnvironmentVariableA("NODUS_EXEC_ROWS", nullptr, 0);
    if (needed == 0) return {};
    std::string path(needed, '\0');
    path.resize(GetEnvironmentVariableA("NODUS_EXEC_ROWS", path.data(), needed));
    return path;
}

// The program a row with these settings would start, held to the host's declaration of the core.
inline bool row_program(const RowSettings& settings, Program& program, std::string& why) {
    program = Program{settings.program, settings.arguments, settings.working_dir, declared_core_roots(),
                      declared_work_roots()};
    if (settings.program.empty()) { why = "the row has no program"; return false; }
    if (program.forbidden_roots.empty()) {
        why = "the host declared no core (NODUS_EXEC_CORE_ROOTS); a row starts nothing until it does";
        return false;
    }
    why = refusal(program);
    return why.empty();
}

} // namespace nodus::exec
