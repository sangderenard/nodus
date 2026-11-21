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
// of the forbidden roots (the core repositories) unless it is inside one of the work roots: the untracked user
// folders and checkout leases where people are meant to work, which sit inside the core's folders but are not it. The child is put in a job that dies with this object, gets no
// console window, inherits only its two pipe ends, and its error stream goes nowhere.
//
// Platform: every OS call lives in the one marked region below (namespace nodus::exec::platform). Windows uses
// CreateProcess plus a kill-on-close job object; POSIX uses posix_spawn with the child leading its own process group
// (killed with this object). The rest of the file is platform-free.
#pragma once

#include <algorithm>
#include <atomic>
#include <cctype>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <initializer_list>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ===================================================================================================================
// PLATFORM REGION BEGIN: all OS includes and calls for this header.
// ===================================================================================================================
#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#else  // POSIX

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char** environ;

#endif

namespace nodus::exec {

struct Program {
    std::string path;                          // the executable, as declared for the row
    std::vector<std::string> arguments;
    std::string working_dir;                   // must exist; never inside a forbidden root
    std::vector<std::string> forbidden_roots;  // the core repositories
    std::vector<std::string> work_roots;       // user folders and checkout leases: allowed even inside the core
};

namespace platform {

#if defined(_WIN32)

// A path made absolute and lexically cleaned (not case-folded); `path` itself when that cannot be done.
inline std::string absolute_path(const std::string& path) {
    char buffer[4096];
    DWORD length = GetFullPathNameA(path.c_str(), sizeof(buffer), buffer, nullptr);
    return length && length < sizeof(buffer) ? std::string(buffer, length) : path;
}

// Command-line quoting as consumed by CommandLineToArgvW / the MSVC runtime.
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

// One started child with its two pipe ends. The child dies with this object (job object, kill on close).
class Process {
public:
    static constexpr long kPending = -2;       // read_some: nothing yet, ask again (never returned on Windows)

    Process() = default;
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    bool launch(const Program& program, std::string& why) {
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
        return true;
    }

    // Bytes written (>= 0), or -1 when the child is gone.
    long write_some(const char* data, size_t count) {
        DWORD wrote = 0;
        if (!WriteFile(input_, data, static_cast<DWORD>(count), &wrote, nullptr)) return -1;
        return static_cast<long>(wrote);
    }

    // Bytes read (> 0), 0 at end of output or on error, kPending to ask again.
    long read_some(char* buffer, size_t count) {
        DWORD got = 0;
        if (!ReadFile(output_, buffer, static_cast<DWORD>(count), &got, nullptr) || got == 0) return 0;
        return static_cast<long>(got);
    }

    void close_input() {
        if (input_) { CloseHandle(input_); input_ = nullptr; }         // end of input for the child
    }

    bool started() const { return process_ != nullptr; }
    bool alive() const { return process_ && WaitForSingleObject(process_, 0) == WAIT_TIMEOUT; }

    int exit_code() const {
        DWORD code = 0;
        return process_ && GetExitCodeProcess(process_, &code) ? static_cast<int>(code) : -1;
    }

    void terminate() { if (job_) TerminateJobObject(job_, 1); }

    void release() {
        for (HANDLE* handle : {&input_, &output_, &process_, &job_})
            if (*handle && *handle != INVALID_HANDLE_VALUE) { CloseHandle(*handle); *handle = nullptr; }
    }

private:
    static void close_all(HANDLE& a, HANDLE& b, HANDLE& c) {
        for (HANDLE* handle : {&a, &b, &c})
            if (*handle && *handle != INVALID_HANDLE_VALUE) { CloseHandle(*handle); *handle = nullptr; }
    }

    HANDLE input_ = nullptr, output_ = nullptr, process_ = nullptr, job_ = nullptr;
};

#else  // POSIX

// A path made absolute and lexically cleaned (not case-folded); `path` itself when that cannot be done.
inline std::string absolute_path(const std::string& path) {
    std::error_code ec;
    const std::filesystem::path made = std::filesystem::absolute(std::filesystem::path(path), ec);
    if (ec || made.empty()) return path;
    return made.lexically_normal().string();
}

// One started child with its two pipe ends. The child leads its own process group, which is killed with this object.
class Process {
public:
    static constexpr long kPending = -2;       // read_some: nothing yet, ask again

    Process() = default;
    Process(const Process&) = delete;
    Process& operator=(const Process&) = delete;

    bool launch(const Program& program, std::string& why) {
        int to_child[2] = {-1, -1};            // [0] the child's input, [1] ours
        int from_child[2] = {-1, -1};          // [0] ours, [1] the child's output
        if (!make_pipe(to_child) || !make_pipe(from_child)) {
            why = "the pipes could not be made";
            close_fds({to_child[0], to_child[1], from_child[0], from_child[1]});
            return false;
        }
        const int nowhere = high_fd(::open("/dev/null", O_WRONLY | O_CLOEXEC));

        // argv is a vector, never a shell line.
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(program.path.c_str()));
        for (const std::string& argument : program.arguments) argv.push_back(const_cast<char*>(argument.c_str()));
        argv.push_back(nullptr);

        posix_spawn_file_actions_t actions;
        posix_spawnattr_t attributes;
        posix_spawn_file_actions_init(&actions);
        posix_spawnattr_init(&attributes);
        // The child inherits only its two pipe ends (every fd of ours is close-on-exec; dup2 clears that on 0/1/2) and
        // its error stream goes nowhere.
        posix_spawn_file_actions_adddup2(&actions, to_child[0], 0);
        posix_spawn_file_actions_adddup2(&actions, from_child[1], 1);
        if (nowhere >= 0) posix_spawn_file_actions_adddup2(&actions, nowhere, 2);
        else posix_spawn_file_actions_addclose(&actions, 2);
        if (!program.working_dir.empty())
            posix_spawn_file_actions_addchdir_np(&actions, program.working_dir.c_str());
        sigset_t none, defaults;
        sigemptyset(&none);
        sigemptyset(&defaults);
        sigaddset(&defaults, SIGPIPE);
        posix_spawnattr_setsigmask(&attributes, &none);
        posix_spawnattr_setsigdefault(&attributes, &defaults);
        posix_spawnattr_setpgroup(&attributes, 0);                     // its own group: killed as one with its children
        posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF);

        pid_t made = -1;
        const int rc = posix_spawn(&made, program.path.c_str(), &actions, &attributes, argv.data(), environ);
        posix_spawn_file_actions_destroy(&actions);
        posix_spawnattr_destroy(&attributes);
        close_fds({to_child[0], from_child[1], nowhere});              // the child holds its own copies now
        if (rc != 0) {
            why = "the program could not be started (error " + std::to_string(rc) + ")";
            close_fds({to_child[1], from_child[0]});
            return false;
        }
        input_ = to_child[1];
        output_ = from_child[0];
        pid_ = made;
        return true;
    }

    // Bytes written (>= 0), or -1 when the child is gone. A closed pipe is an error here, never a SIGPIPE.
    long write_some(const char* data, size_t count) {
        sigset_t block, previous;
        sigemptyset(&block);
        sigaddset(&block, SIGPIPE);
        pthread_sigmask(SIG_BLOCK, &block, &previous);
        ssize_t wrote;
        do { wrote = ::write(input_, data, count); } while (wrote < 0 && errno == EINTR);
        if (wrote < 0 && errno == EPIPE) {
            timespec zero{0, 0};
            sigtimedwait(&block, nullptr, &zero);                      // swallow the SIGPIPE this write raised
        }
        pthread_sigmask(SIG_SETMASK, &previous, nullptr);
        return wrote < 0 ? -1 : static_cast<long>(wrote);
    }

    // Bytes read (> 0), 0 at end of output or on error, kPending to ask again (the wait is bounded so a stop is noticed).
    long read_some(char* buffer, size_t count) {
        pollfd waiting{output_, POLLIN, 0};
        const int ready = ::poll(&waiting, 1, 100);
        if (ready == 0) return kPending;
        if (ready < 0) return errno == EINTR ? kPending : 0;
        const ssize_t got = ::read(output_, buffer, count);
        if (got < 0) return (errno == EINTR || errno == EAGAIN) ? kPending : 0;
        return static_cast<long>(got);
    }

    void close_input() {
        if (input_ >= 0) { ::close(input_); input_ = -1; }             // end of input for the child
    }

    bool started() const { return pid_ > 0; }
    bool alive() const { return pid_ > 0 && !exited(nullptr); }

    int exit_code() const {
        siginfo_t info{};
        if (pid_ <= 0 || !exited(&info)) return -1;
        if (info.si_code == CLD_EXITED) return info.si_status;
        return 128 + info.si_status;                                   // ended by a signal
    }

    void terminate() { if (pid_ > 0) ::kill(-pid_, SIGKILL); }

    void release() {
        close_input();
        if (output_ >= 0) { ::close(output_); output_ = -1; }
        if (pid_ > 0) {
            ::kill(-pid_, SIGKILL);                                    // the child dies with this object
            int status = 0;
            while (::waitpid(pid_, &status, 0) < 0 && errno == EINTR) {}
            pid_ = -1;
        }
    }

private:
    // True once the child has exited. The zombie is left in place (WNOWAIT) so its process group id stays ours until
    // release() reaps it.
    bool exited(siginfo_t* out) const {
        siginfo_t info{};
        if (::waitid(P_PID, static_cast<id_t>(pid_), &info, WEXITED | WNOHANG | WNOWAIT) < 0) return true;
        if (out) *out = info;
        return info.si_pid != 0;
    }

    static int high_fd(int fd) {                                       // keep our ends off 0/1/2
        if (fd < 0 || fd > 2) return fd;
        const int moved = ::fcntl(fd, F_DUPFD_CLOEXEC, 3);
        ::close(fd);
        return moved;
    }

    static bool make_pipe(int ends[2]) {
        int made[2];
#if defined(__linux__)
        if (::pipe2(made, O_CLOEXEC) != 0) return false;
#else
        if (::pipe(made) != 0) return false;
        ::fcntl(made[0], F_SETFD, FD_CLOEXEC);
        ::fcntl(made[1], F_SETFD, FD_CLOEXEC);
#endif
        ends[0] = high_fd(made[0]);
        ends[1] = high_fd(made[1]);
        return ends[0] >= 0 && ends[1] >= 0;
    }

    static void close_fds(std::initializer_list<int> fds) {
        for (int fd : fds) if (fd >= 0) ::close(fd);
    }

    int input_ = -1, output_ = -1;
    pid_t pid_ = -1;
};

#endif

}  // namespace platform

// ===================================================================================================================
// PLATFORM REGION END
// ===================================================================================================================

inline std::string normalized(std::string path) {
    std::replace(path.begin(), path.end(), '\\', '/');
    std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    return path;
}

inline std::string full_path(const std::string& path) {
    return normalized(platform::absolute_path(path));
}

// True when `path` (already a full path) is `root` or inside it.
inline bool within(const std::string& path, const std::string& root) {
    const std::string held = full_path(root);
    return path == held || path.rfind(held + "/", 0) == 0;
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
    for (const std::string& work : program.work_roots)
        if (within(where, work)) return {};
    for (const std::string& root : program.forbidden_roots)
        if (within(where, root))
            return "a row's program never runs with its working directory inside the core (" + root +
                   ") outside a user folder or checkout lease";
    return {};
}

class Child {
public:
    Child() = default;
    Child(const Child&) = delete;
    Child& operator=(const Child&) = delete;
    ~Child() { stop(); }

    // output_limit=0 preserves span mode. A positive limit backpressures stdout;
    // consumers must call take while the child runs, not wait for process exit.
    bool start(const Program& program, std::string& why, bool discard_output = false, size_t output_limit = 0) {
        stop();
        why = refusal(program);
        if (!why.empty()) return false;
        if (!process_.launch(program, why)) {
            release();
            return false;
        }
        input_done_ = false;
        discard_output_ = discard_output;
        output_limit_ = output_limit;
        stopping_ = false;
        reader_ = std::thread([this] { read_loop(); });
        writer_ = std::thread([this] { write_loop(); });
        return true;
    }

    // Queue bytes for the child's input. Returns at once.
    void send(std::string bytes) {
        if (bytes.empty()) return;
        { std::lock_guard<std::mutex> lock(mutex_); pending_bytes_ += bytes.size(); to_child_.push_back(std::move(bytes)); }
        wake_.notify_all();
    }

    // A continuous row queues one tick's chunks atomically and never blocks on a slow child.
    bool try_send_many(std::vector<std::string>& chunks, size_t limit) {
        if (!running()) return false;
        size_t total = 0;
        for (const std::string& chunk : chunks) {
            if (chunk.size() > limit - total) return false;
            total += chunk.size();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (input_done_ || pending_bytes_ > limit - total) return false;
            pending_bytes_ += total;
            for (std::string& chunk : chunks) if (!chunk.empty()) to_child_.push_back(std::move(chunk));
        }
        wake_.notify_all();
        return true;
    }

    // After everything sent has been written, the child sees end of input.
    void finish_input() {
        { std::lock_guard<std::mutex> lock(mutex_); input_done_ = true; }
        wake_.notify_all();
    }

    // Move whatever the child has written so far into `out` (appended). Returns the number of bytes moved.
    size_t take(std::string& out, size_t maximum = SIZE_MAX) {
        size_t moved;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            moved = std::min(maximum, from_child_.size());
            out.append(from_child_, 0, moved);
            from_child_.erase(0, moved);
        }
        output_space_.notify_all();
        return moved;
    }

    size_t buffered_output() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return from_child_.size();
    }

    // Distinguish process completion from delivery of the final buffered bytes.
    bool output_drained() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return output_closed_.load() && from_child_.empty();
    }

    bool running() const { return process_.alive(); }

    // True once the child has exited AND everything it wrote has been collected.
    bool finished() const { return process_.started() && !running() && output_closed_.load(); }

    int exit_code() const { return process_.exit_code(); }

    void kill() {
        { std::lock_guard<std::mutex> lock(mutex_); stopping_ = true; }
        output_space_.notify_all();
        process_.terminate();
    }

    void stop() {
        kill();
        finish_input();
        if (writer_.joinable()) writer_.join();
        if (reader_.joinable()) reader_.join();
        release();
    }

private:
    void release() {
        process_.release();
        std::lock_guard<std::mutex> lock(mutex_);
        to_child_.clear();
        from_child_.clear();
        pending_bytes_ = 0;
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
                const size_t want = std::min<size_t>(chunk.size() - at, 1 << 16);
                const long wrote = process_.write_some(chunk.data() + at, want);
                if (wrote < 0) return;                                  // the child is gone
                at += static_cast<size_t>(wrote);
            }
            { std::lock_guard<std::mutex> lock(mutex_); pending_bytes_ -= chunk.size(); }
        }
        std::lock_guard<std::mutex> lock(handles_);
        process_.close_input();                                         // end of input for the child
    }

    void read_loop() {
        char buffer[1 << 16];
        for (;;) {
            size_t want = sizeof(buffer);
            {
                std::unique_lock<std::mutex> lock(mutex_);
                output_space_.wait(lock, [this] {
                    return stopping_ || discard_output_ || !output_limit_ || from_child_.size() < output_limit_;
                });
                if (stopping_) break;
                if (!discard_output_ && output_limit_)
                    want = std::min<size_t>(want, output_limit_ - from_child_.size());
            }
            const long got = process_.read_some(buffer, want);
            if (got == platform::Process::kPending) continue;           // nothing yet; look at stopping_ again
            if (got <= 0) break;
            std::lock_guard<std::mutex> lock(mutex_);
            if (!discard_output_) from_child_.append(buffer, static_cast<size_t>(got));
        }
        output_closed_ = true;
    }

    platform::Process process_;
    std::thread reader_, writer_;
    mutable std::mutex mutex_;
    std::mutex handles_;
    std::condition_variable wake_, output_space_;
    std::deque<std::string> to_child_;
    std::string from_child_;
    size_t pending_bytes_ = 0;
    bool input_done_ = false;
    bool discard_output_ = false;
    bool stopping_ = false;
    size_t output_limit_ = 0;
    std::atomic<bool> output_closed_{false};
};

}  // namespace nodus::exec
