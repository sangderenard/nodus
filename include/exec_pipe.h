// exec_pipe: one external native program with bytes going in and bytes coming out, for a row that owns it.
//
// This is the exchange an exec tool row uses. The row hands the child a byte span and collects what the child
// writes; it never blocks the tick. Two helper threads move the bytes (one feeds the child's input, one drains its
// output), so a child that fills its output while its input is still being written cannot deadlock the row.
//
//   nodus::exec::Child child;
//   std::string why;
//   if (!child.start({program, {"-c", "-q", "5"}, working_dir, forbidden_roots}, why)) ...
//   child.send(bytes);          // any number of times; returns at once
//   child.finish_input();       // the child sees end of input once everything sent has been written
//   child.take(out);            // whatever has arrived so far; returns at once
//   child.running() / child.exit_code() / child.kill()
//
// Rules a row's program is held to, checked by start(): it is never Git (Git in the core is asked for through the
// service, turing examples/meowdoku/core_git.py holds the same rule), and its working directory is never inside one
// of the forbidden roots (the core repositories). The child is put in a job that dies with this object, gets no
// console window, inherits only its two pipe ends, and its error stream goes nowhere.
//
// Windows only for now; the POSIX form (pipe/fork/exec) is not written.
#pragma once

#ifndef _WIN32
#error "exec_pipe.h is implemented for Windows only"
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace nodus::exec {

struct Program {
    std::string path;                          // the executable, as declared for the row
    std::vector<std::string> arguments;
    std::string working_dir;                   // must exist; never inside a forbidden root
    std::vector<std::string> forbidden_roots;  // the core repositories
};

inline std::string normalized(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

inline std::string full_path(const std::string& path) {
    char buffer[4096];
    DWORD length = GetFullPathNameA(path.c_str(), sizeof(buffer), buffer, nullptr);
    return normalized(length && length < sizeof(buffer) ? std::string(buffer, length) : path);
}

// Why this program may not be started as asked; empty when it may.
inline std::string refusal(const Program& program) {
    std::string name = normalized(program.path);
    const size_t slash = name.find_last_of('/');
    if (slash != std::string::npos) name = name.substr(slash + 1);
    if (name == "git" || name == "git.exe" || name.rfind("git-", 0) == 0)
        return "a row never runs Git; Git in the core is asked for through the service";
    if (program.working_dir.empty()) return "a row's program needs a declared working directory";
    const std::string where = full_path(program.working_dir);
    for (const std::string& root : program.forbidden_roots) {
        const std::string held = full_path(root);
        if (where == held || where.rfind(held + "/", 0) == 0)
            return "a row's program never runs with its working directory inside the core (" + root + ")";
    }
    return {};
}

inline std::string quoted(const std::string& argument) {
    if (!argument.empty() && argument.find_first_of(" \t\"") == std::string::npos) return argument;
    std::string out = "\"";
    size_t backslashes = 0;
    for (char c : argument) {
        if (c == '\\') { ++backslashes; continue; }
        if (c == '"') { out.append(backslashes * 2 + 1, '\\'); out.push_back('"'); backslashes = 0; continue; }
        out.append(backslashes, '\\');
        backslashes = 0;
        out.push_back(c);
    }
    out.append(backslashes * 2, '\\');
    out.push_back('"');
    return out;
}

class Child {
public:
    Child() = default;
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() { stop(); }

    bool start(const Program& program, std::string& why) {
        stop();
        why = refusal(program);
        if (!why.empty()) return false;
        SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        HANDLE child_in = nullptr, child_out = nullptr, nowhere = nullptr;
        if (!CreatePipe(&child_in, &input_, &inherit, 1 << 16) || !CreatePipe(&output_, &child_out, &inherit, 1 << 16)) {
            why = "the pipes could not be made";
            close_all(child_in, child_out, nowhere);
            return false;
        }
        SetHandleInformation(input_, HANDLE_FLAG_INHERIT, 0);          // the parent's ends stay with the parent
        SetHandleInformation(output_, HANDLE_FLAG_INHERIT, 0);
        nowhere = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherit, OPEN_EXISTING, 0, nullptr);
        std::string command = quoted(program.path);
        for (const std::string& argument : program.arguments) command += " " + quoted(argument);
        HANDLE inherited[3] = {child_in, child_out, nowhere};
        SIZE_T size = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &size);
        std::vector<char> attributes(size);
        STARTUPINFOEXA startup{};
        startup.StartupInfo.cb = sizeof(startup);
        startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        startup.StartupInfo.hStdInput = child_in;
        startup.StartupInfo.hStdOutput = child_out;
        startup.StartupInfo.hStdError = nowhere;
        startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        PROCESS_INFORMATION made{};
        const bool listed = InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &size)
            && UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited,
                                         sizeof(inherited), nullptr, nullptr);
        job_ = CreateJobObjectA(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;      // the child dies with this object
        const bool ready = listed && job_ && SetInformationJobObject(job_, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
        std::vector<char> line(command.begin(), command.end());
        line.push_back('\0');
        const bool created = ready && CreateProcessA(program.path.c_str(), line.data(), nullptr, nullptr, TRUE,
                                                     CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT,
                                                     nullptr, program.working_dir.c_str(), &startup.StartupInfo, &made);
        if (listed) DeleteProcThreadAttributeList(startup.lpAttributeList);
        close_all(child_in, child_out, nowhere);                       // the child holds its own copies now
        if (!created) {
            why = "the program could not be started (error " + std::to_string(GetLastError()) + ")";
            release();
            return false;
        }
        AssignProcessToJobObject(job_, made.hProcess);
        ResumeThread(made.hThread);
        CloseHandle(made.hThread);
        process_ = made.hProcess;
        input_done_ = false;
        reader_ = std::thread([this] { read_loop(); });
        writer_ = std::thread([this] { write_loop(); });
        return true;
    }

    // Queue bytes for the child's input. Returns at once.
    void send(std::string bytes) {
        if (bytes.empty()) return;
        { std::lock_guard<std::mutex> lock(mutex_); to_child_.push_back(std::move(bytes)); }
        wake_.notify_all();
    }

    // After everything sent has been written, the child sees end of input.
    void finish_input() {
        { std::lock_guard<std::mutex> lock(mutex_); input_done_ = true; }
        wake_.notify_all();
    }

    // Move whatever the child has written so far into `out` (appended). Returns the number of bytes moved.
    size_t take(std::string& out) {
        std::lock_guard<std::mutex> lock(mutex_);
        const size_t moved = from_child_.size();
        out.append(from_child_);
        from_child_.clear();
        return moved;
    }

    bool running() const { return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT; }

    // True once the child has exited AND everything it wrote has been collected.
    bool finished() const { return process_ && !running() && output_closed_.load(); }

    int exit_code() const {
        DWORD code = 0;
        return process_ && GetExitCodeProcess(process_, &code) ? static_cast<int>(code) : -1;
    }

    void kill() { if (job_) TerminateJobObject(job_, 1); }

    void stop() {
        kill();
        finish_input();
        if (writer_.joinable()) writer_.join();
        if (reader_.joinable()) reader_.join();
        release();
    }

private:
    static void close_all(HANDLE& a, HANDLE& b, HANDLE& c) {
        for (HANDLE* handle : {&a, &b, &c})
            if (*handle && *handle != INVALID_HANDLE_VALUE) { CloseHandle(*handle); *handle = nullptr; }
    }

    void release() {
        for (HANDLE* handle : {&input_, &output_, &process_, &job_})
            if (*handle && *handle != INVALID_HANDLE_VALUE) { CloseHandle(*handle); *handle = nullptr; }
        std::lock_guard<std::mutex> lock(mutex_);
        to_child_.clear();
        from_child_.clear();
        output_closed_ = false;
    }

    void write_loop() {
        for (;;) {
            std::string chunk;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock, [this] { return !to_child_.empty() || input_done_; });
                if (to_child_.empty()) break;                           // input is finished and all of it is written
                chunk = std::move(to_child_.front());
                to_child_.pop_front();
            }
            size_t at = 0;
            while (at < chunk.size()) {
                DWORD wrote = 0;
                const DWORD want = static_cast<DWORD>(std::min<size_t>(chunk.size() - at, 1 << 16));
                if (!WriteFile(input_, chunk.data() + at, want, &wrote, nullptr)) return;      // the child is gone
                at += wrote;
            }
        }
        std::lock_guard<std::mutex> lock(handles_);
        if (input_) { CloseHandle(input_); input_ = nullptr; }         // end of input for the child
    }

    void read_loop() {
        char buffer[1 << 16];
        for (;;) {
            DWORD got = 0;
            if (!ReadFile(output_, buffer, sizeof(buffer), &got, nullptr) || got == 0) break;
            std::lock_guard<std::mutex> lock(mutex_);
            from_child_.append(buffer, got);
        }
        output_closed_ = true;
    }

    HANDLE input_ = nullptr, output_ = nullptr, process_ = nullptr, job_ = nullptr;
    std::thread reader_, writer_;
    mutable std::mutex mutex_;
    std::mutex handles_;
    std::condition_variable wake_;
    std::deque<std::string> to_child_;
    std::string from_child_;
    bool input_done_ = false;
    std::atomic<bool> output_closed_{false};
};

}  // namespace nodus::exec
