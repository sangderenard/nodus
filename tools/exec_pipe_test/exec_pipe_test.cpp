// exec_pipe_test <compressor> <scratch-dir> [forbidden-root]
// Drives a real native compressor through nodus::exec::Child: a byte span in, compressed bytes out, and back.
// The compressor must compress standard input to standard output with "-c" and decompress with "-d -c"
// (brotli, xz, gzip, bzip2 and zstd all do).
#include "exec_pipe.h"

#include <chrono>
#include <cstdio>
#include <string>

using nodus::exec::Child;
using nodus::exec::Program;

static int failures = 0;

static void check(bool ok, const char* what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

// One span through one child: send, finish, collect until it has exited. Never blocks longer than `limit_ms`.
static bool run_span(const Program& program, const std::string& input, std::string& output, int limit_ms, std::string& why) {
    Child child;
    if (!child.start(program, why)) return false;
    child.send(input);
    child.finish_input();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(limit_ms);
    while (!child.finished()) {
        child.take(output);
        if (std::chrono::steady_clock::now() > deadline) { why = "timed out"; child.kill(); return false; }
        Sleep(1);
    }
    child.take(output);
    if (child.exit_code() != 0) { why = "exit code " + std::to_string(child.exit_code()); return false; }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: exec_pipe_test <compressor> <scratch-dir> [forbidden-root]\n"); return 2; }
    const std::string compressor = argv[1], scratch = argv[2], core = argc > 3 ? argv[3] : "C:/dev/Powershell";
    std::string why;

    // The rules a row's program is held to.
    check(!nodus::exec::refusal({"C:/Program Files/Git/cmd/git.exe", {}, scratch, {core}}).empty(), "git is refused");
    check(!nodus::exec::refusal({"GIT.EXE", {}, scratch, {core}}).empty(), "git is refused whatever its case");
    check(!nodus::exec::refusal({"git-upload-pack", {}, scratch, {core}}).empty(), "a git helper is refused");
    check(!nodus::exec::refusal({compressor, {}, core, {core}}).empty(), "a working directory that is the core is refused");
    check(!nodus::exec::refusal({compressor, {}, core + "/turing/src", {core}}).empty(), "a working directory inside the core is refused");
    check(!nodus::exec::refusal({compressor, {}, "", {core}}).empty(), "no working directory is refused");
    check(nodus::exec::refusal({compressor, {}, scratch, {core}}).empty(), "the compressor outside the core is allowed");
    {
        Child child;
        check(!child.start({compressor, {"-c"}, core, {core}}, why) && !child.running(), "start() applies the rules");
    }

    // A byte span in, compressed bytes out, and back again.
    std::string span;
    for (int i = 0; i < 200000; ++i) span += "triangle_map[" + std::to_string(i % 4096) + "] ";
    std::string packed, unpacked;
    const auto started = std::chrono::steady_clock::now();
    check(run_span({compressor, {"-c"}, scratch, {core}}, span, packed, 60000, why), "compress one span");
    check(!packed.empty() && packed.size() < span.size() / 4, "the output is much smaller than the input");
    check(run_span({compressor, {"-d", "-c"}, scratch, {core}}, packed, unpacked, 60000, why), "decompress it");
    check(unpacked == span, "the round trip returns the same bytes");
    const auto took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
    std::printf("     %zu bytes -> %zu -> %zu in %lld ms\n", span.size(), packed.size(), unpacked.size(), static_cast<long long>(took));

    // Bytes that do not compress: the child's output fills while its input is still being written.
    std::string noise(8u << 20, '\0');
    unsigned state = 12345;
    for (char& c : noise) { state = state * 1664525u + 1013904223u; c = static_cast<char>(state >> 24); }
    std::string packed_noise, unpacked_noise;
    check(run_span({compressor, {"-c"}, scratch, {core}}, noise, packed_noise, 120000, why), "8 MiB of noise does not deadlock");
    check(run_span({compressor, {"-d", "-c"}, scratch, {core}}, packed_noise, unpacked_noise, 120000, why) && unpacked_noise == noise,
          "and comes back the same");

    // Nothing here blocks the caller.
    {
        Child child;
        check(child.start({compressor, {"-c"}, scratch, {core}}, why), "a child waiting for input starts");
        std::string nothing;
        const auto before = std::chrono::steady_clock::now();
        child.send("abc");
        child.take(nothing);
        const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - before).count();
        check(waited < 200 && child.running(), "send and take return at once while it waits");
        child.kill();
        for (int i = 0; i < 2000 && child.running(); ++i) Sleep(1);
        check(!child.running(), "kill ends it");
    }
    {
        Child child;
        child.start({compressor, {"-c"}, scratch, {core}}, why);
    }
    check(true, "a child still waiting is ended when its owner goes away");

    // A program that is not there.
    {
        Child child;
        check(!child.start({scratch + "/no-such-program.exe", {}, scratch, {core}}, why) && !why.empty(), "a missing program is reported");
    }
    std::printf("%s\n", failures ? "FAILED" : "all passed");
    return failures ? 1 : 0;
}
