// spirv_translation.cpp
// C++20: KernelIR -> GLSL compute -> SPIR-V (external glslangValidator OR optional shaderc)
//
// Contract assumptions:
// - There exists a *pure* Tier-0 KernelIR and a GLSL compute emitter that only understands Tier-0 ops.
// - Composites / BitStructType dispatch occur *before* this stage (Tier-1 lowering), not in the emitter.
//
// Default behavior:
// - Emit Vulkan-flavored GLSL (.comp), compile to SPIR-V (.spv), cache by content hash.
//
// Build:
// - C++20 required.
// - Optional: define NODUS_HAVE_SHADERC=1 and link shaderc for in-process compilation.
// - Otherwise: use glslangValidator (Vulkan SDK tools) either via PATH or an explicit path.
//
// Security note:
// - External compiler invocation uses argv vectors (no shell). On Windows we still must build a command line
//   for CreateProcessW, but args are quoted and not interpreted by a shell.

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring> // memcpy
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#if defined(_WIN32)
  #define NOMINMAX
  #include <windows.h>
#else
  #include <sys/types.h>
  #include <sys/wait.h>
  #include <unistd.h>
#endif

// Public translation interface + Tier-0 GLSL emitter
#include "spirv_translation.h"
#include "kernel_ir_glsl_emitter.h" // nodus::kernel::{KernelIR, GlslEmitter, GlslEmitOptions}

#if defined(NODUS_HAVE_SHADERC) && NODUS_HAVE_SHADERC
  #include <shaderc/shaderc.hpp>
#endif

namespace nodus::spirv {

// ------------------------------
// Hash utilities (cache keys)
// ------------------------------

static inline u64 fnv1a64(const u8* data, size_t size) {
  constexpr u64 kOffset = 1469598103934665603ull;
  constexpr u64 kPrime  = 1099511628211ull;
  u64 h = kOffset;
  for (size_t i = 0; i < size; ++i) {
    h ^= static_cast<u64>(data[i]);
    h *= kPrime;
  }
  return h;
}

static inline void hash_combine(u64& h, u64 v) {
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
}

static inline u64 hash_string(std::string_view s) {
  return fnv1a64(reinterpret_cast<const u8*>(s.data()), s.size());
}

static inline std::string sanitize_filename(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.') out.push_back(c);
    else out.push_back('_');
  }
  if (out.empty()) out = "kernel";
  return out;
}

struct CacheKey {
  u64 h = 0;

  std::string hex16() const {
    std::ostringstream ss;
    ss << std::hex << std::setw(16) << std::setfill('0') << h;
    return ss.str();
  }

  static CacheKey make(const nodus::kernel::KernelIR& k,
                       const SpirvCompileOptions& opt,
                       std::string_view glsl) {
    u64 h = 0;
    hash_combine(h, hash_string(k.name));
    hash_combine(h, static_cast<u64>(opt.env));
    hash_combine(h, static_cast<u64>(opt.glsl_version));
    hash_combine(h, static_cast<u64>(opt.enable_scalar_block_layout));
    hash_combine(h, static_cast<u64>(opt.emit_debug_comments));

    // Defines must affect cache key
    for (const auto& d : opt.defines) {
      hash_combine(h, hash_string(d.first));
      hash_combine(h, hash_string(d.second));
    }

    // GLSL source already captures local sizes, buffer layout, and kernel logic.
    hash_combine(h, hash_string(glsl));
    return CacheKey{h};
  }
};

// ------------------------------
// File IO helpers
// ------------------------------

static inline void write_text_file(const std::filesystem::path& p, std::string_view text) {
  std::filesystem::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open file for write: " + p.string());
  f.write(text.data(), static_cast<std::streamsize>(text.size()));
}

static inline SpirvBinary read_spirv_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("Failed to open SPIR-V file: " + p.string());
  f.seekg(0, std::ios::end);
  const std::streamsize n = f.tellg();
  f.seekg(0, std::ios::beg);
  if (n < 0 || (n % 4) != 0) throw std::runtime_error("SPIR-V file size invalid (not multiple of 4): " + p.string());

  SpirvBinary bin;
  bin.words.resize(static_cast<size_t>(n / 4));
  f.read(reinterpret_cast<char*>(bin.words.data()), n);
  if (!f) throw std::runtime_error("Failed to read SPIR-V file: " + p.string());
  return bin;
}

// ------------------------------
// Process invocation (external compiler)
// ------------------------------

struct ProcessResult {
  int exit_code = 1;
  std::string output;
};

#if defined(_WIN32)

static inline std::wstring widen(std::string_view s) {
  if (s.empty()) return {};
  int wlen = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
  if (wlen <= 0) return {};
  std::wstring w;
  w.resize((size_t)wlen);
  MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), w.data(), wlen);
  return w;
}

static inline std::wstring quote_windows_arg(const std::wstring& a) {
  // Minimal CreateProcess-compatible quoting.
  if (a.empty()) return L"\"\"";
  bool needs = false;
  for (wchar_t c : a) {
    if (c == L' ' || c == L'\t' || c == L'"') { needs = true; break; }
  }
  if (!needs) return a;

  std::wstring out = L"\"";
  for (wchar_t c : a) {
    if (c == L'"') out += L"\\\"";
    else out += c;
  }
  out += L"\"";
  return out;
}

static ProcessResult run_process_capture(const std::vector<std::wstring>& argv) {
  if (argv.empty()) throw std::runtime_error("run_process_capture: empty argv");

  // Build command line (CreateProcess requires mutable buffer).
  std::wstring cmd;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i) cmd.push_back(L' ');
    cmd += quote_windows_arg(argv[i]);
  }
  std::vector<wchar_t> cmdline(cmd.begin(), cmd.end());
  cmdline.push_back(L'\0');

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE out_read = nullptr, out_write = nullptr;
  if (!CreatePipe(&out_read, &out_write, &sa, 0)) {
    throw std::runtime_error("CreatePipe failed");
  }
  // Ensure read handle is not inherited.
  SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags = STARTF_USESTDHANDLES;
  si.hStdOutput = out_write;
  si.hStdError  = out_write;
  si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi{};
  BOOL ok = CreateProcessW(
    /*lpApplicationName*/ nullptr,
    /*lpCommandLine*/ cmdline.data(),
    /*lpProcessAttributes*/ nullptr,
    /*lpThreadAttributes*/ nullptr,
    /*bInheritHandles*/ TRUE,
    /*dwCreationFlags*/ CREATE_NO_WINDOW,
    /*lpEnvironment*/ nullptr,
    /*lpCurrentDirectory*/ nullptr,
    /*lpStartupInfo*/ &si,
    /*lpProcessInformation*/ &pi
  );

  CloseHandle(out_write); // parent closes write end

  if (!ok) {
    CloseHandle(out_read);
    throw std::runtime_error("CreateProcessW failed");
  }

  std::string output;
  output.reserve(4096);

  auto drain_pipe = [&]() {
    for (;;) {
      DWORD avail = 0;
      if (!PeekNamedPipe(out_read, nullptr, 0, nullptr, &avail, nullptr)) break;
      if (avail == 0) break;

      char buf[4096];
      DWORD nread = 0;
      if (!ReadFile(out_read, buf, (DWORD)std::min<DWORD>(avail, sizeof(buf)), &nread, nullptr)) break;
      if (nread == 0) break;
      output.append(buf, buf + nread);
    }
  };

  // Drain while running to avoid pipe deadlock.
  for (;;) {
    drain_pipe();
    DWORD w = WaitForSingleObject(pi.hProcess, 25);
    if (w == WAIT_OBJECT_0) break;
  }
  // Drain remaining
  for (;;) {
    DWORD avail = 0;
    if (!PeekNamedPipe(out_read, nullptr, 0, nullptr, &avail, nullptr)) break;
    if (avail == 0) break;
    char buf[4096];
    DWORD nread = 0;
    if (!ReadFile(out_read, buf, sizeof(buf), &nread, nullptr)) break;
    if (nread == 0) break;
    output.append(buf, buf + nread);
  }

  DWORD exit_code = 1;
  GetExitCodeProcess(pi.hProcess, &exit_code);

  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(out_read);

  ProcessResult r;
  r.exit_code = static_cast<int>(exit_code);
  r.output = std::move(output);
  return r;
}

#else

static ProcessResult run_process_capture(const std::vector<std::string>& argv) {
  if (argv.empty()) throw std::runtime_error("run_process_capture: empty argv");

  int pipefd[2];
  if (pipe(pipefd) != 0) throw std::runtime_error("pipe failed");

  pid_t pid = fork();
  if (pid == -1) throw std::runtime_error("fork failed");

  if (pid == 0) {
    // child
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[0]);
    close(pipefd[1]);

    std::vector<char*> args;
    args.reserve(argv.size() + 1);
    for (const auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
    args.push_back(nullptr);

    execvp(args[0], args.data());
    _exit(127);
  }

  // parent
  close(pipefd[1]);
  std::string output;
  output.reserve(4096);
  char buf[4096];
  ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
    output.append(buf, buf + n);
  }
  close(pipefd[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  int exit_code = (WIFEXITED(status) ? WEXITSTATUS(status) : 1);

  ProcessResult r;
  r.exit_code = exit_code;
  r.output = std::move(output);
  return r;
}

#endif

// ------------------------------
// Compiler interface
// ------------------------------

static inline const char* target_env_arg(TargetEnv env) {
  switch (env) {
    case TargetEnv::Vulkan_1_1: return "vulkan1.1";
    case TargetEnv::Vulkan_1_2: return "vulkan1.2";
    case TargetEnv::Vulkan_1_3: return "vulkan1.3";
    default: return "vulkan1.2";
  }
}

class ExternalGlslangValidatorCompiler final : public ISpirvCompiler {
public:
  SpirvBinary compile_glsl_compute(const std::string& glsl,
                                   const SpirvCompileOptions& opt,
                                   const std::filesystem::path& out_spv_path,
                                   const std::filesystem::path& out_glsl_path) override {
    std::filesystem::create_directories(out_spv_path.parent_path());

    write_text_file(out_glsl_path, glsl);

#if defined(_WIN32)
    const std::wstring exe = opt.glslang_validator_path.empty()
      ? L"glslangValidator.exe"
      : opt.glslang_validator_path.wstring();

    std::vector<std::wstring> argv;
    argv.push_back(exe);
    argv.push_back(L"-V");
    argv.push_back(L"--target-env");
    argv.push_back(widen(target_env_arg(opt.env)));
    // -D NAME=VALUE
    for (const auto& d : opt.defines) {
      std::wstring w = L"-D";
      w += widen(d.first);
      if (!d.second.empty()) {
        w += L"=";
        w += widen(d.second);
      }
      argv.push_back(std::move(w));
    }
    argv.push_back(out_glsl_path.wstring());
    argv.push_back(L"-o");
    argv.push_back(out_spv_path.wstring());

    auto pr = run_process_capture(argv);
#else
    const std::string exe = opt.glslang_validator_path.empty()
      ? "glslangValidator"
      : opt.glslang_validator_path.string();

    std::vector<std::string> argv;
    argv.push_back(exe);
    argv.push_back("-V");
    argv.push_back("--target-env");
    argv.push_back(target_env_arg(opt.env));
    for (const auto& d : opt.defines) {
      std::string s = "-D" + d.first;
      if (!d.second.empty()) s += "=" + d.second;
      argv.push_back(std::move(s));
    }
    argv.push_back(out_glsl_path.string());
    argv.push_back("-o");
    argv.push_back(out_spv_path.string());

    auto pr = run_process_capture(argv);
#endif

    if (opt.verbose) {
      std::fprintf(stderr, "[glslangValidator] exit=%d\n%s\n", pr.exit_code, pr.output.c_str());
    }
    if (pr.exit_code != 0) {
      throw std::runtime_error("glslangValidator failed:\n" + pr.output);
    }

    return read_spirv_file(out_spv_path);
  }
};

#if defined(NODUS_HAVE_SHADERC) && NODUS_HAVE_SHADERC
class ShadercCompiler final : public ISpirvCompiler {
public:
  SpirvBinary compile_glsl_compute(const std::string& glsl,
                                   const SpirvCompileOptions& opt,
                                   const std::filesystem::path& out_spv_path,
                                   const std::filesystem::path& out_glsl_path) override {
    // Optionally keep GLSL on disk for debugging parity with external compiler.
    if (opt.keep_intermediates) {
      write_text_file(out_glsl_path, glsl);
    }

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Target env
    switch (opt.env) {
      case TargetEnv::Vulkan_1_1: options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1); break;
      case TargetEnv::Vulkan_1_2: options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2); break;
      case TargetEnv::Vulkan_1_3: options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3); break;
      default: break;
    }

    // Defines
    for (const auto& d : opt.defines) {
      options.AddMacroDefinition(d.first, d.second);
    }

    auto result = compiler.CompileGlslToSpv(
      glsl.data(), glsl.size(),
      shaderc_compute_shader,
      /*input_file_name*/ opt.keep_intermediates ? out_glsl_path.string().c_str() : "kernel.comp",
      /*entry_point*/ "main",
      options
    );

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      throw std::runtime_error(std::string("shaderc failed:\n") + result.GetErrorMessage());
    }

    SpirvBinary bin;
    bin.words.assign(result.cbegin(), result.cend());

    if (opt.keep_intermediates) {
      std::filesystem::create_directories(out_spv_path.parent_path());
      std::ofstream f(out_spv_path, std::ios::binary);
      if (!f) throw std::runtime_error("Failed to open SPIR-V output file: " + out_spv_path.string());
      f.write(reinterpret_cast<const char*>(bin.words.data()),
              static_cast<std::streamsize>(bin.words.size() * sizeof(u32)));
    }

    return bin;
  }
};
#endif

// ------------------------------
// Translator (driver + cache)
// ------------------------------

SpirvTranslator::SpirvTranslator(SpirvCompileOptions opt,
                                 std::unique_ptr<ISpirvCompiler> compiler)
  : opt_(std::move(opt)),
    compiler_(std::move(compiler)) {

  std::filesystem::create_directories(opt_.cache_dir);

  if (!compiler_) {
#if defined(NODUS_HAVE_SHADERC) && NODUS_HAVE_SHADERC
    // Prefer in-process when available (you can invert this preference if you want toolchain parity).
    compiler_ = std::make_unique<ShadercCompiler>();
#else
    compiler_ = std::make_unique<ExternalGlslangValidatorCompiler>();
#endif
  }
}

SpirvCompileResult SpirvTranslator::translate_kernel_to_spirv(const KernelIR& k) {
  // 1) Emit GLSL from Tier-0 kernel IR
  nodus::kernel::GlslEmitOptions emit_opt{};
  emit_opt.glsl_version = opt_.glsl_version;
  emit_opt.enable_scalar_block_layout = opt_.enable_scalar_block_layout;
  emit_opt.emit_debug_comments = opt_.emit_debug_comments;

  nodus::kernel::GlslEmitter emitter(emit_opt);
  std::string glsl = emitter.emit(k);

  // 2) Cache key
  CacheKey key = CacheKey::make(k, opt_, glsl);
  const std::string base_name = sanitize_filename(k.name) + "_" + key.hex16();
  const auto base = opt_.cache_dir / base_name;
  const auto glsl_file = std::filesystem::path(base.string() + ".comp");
  const auto spv_file  = std::filesystem::path(base.string() + ".spv");

  // 3) Cache hit
  if (opt_.enable_cache && std::filesystem::exists(spv_file)) {
    SpirvCompileResult r;
    r.spirv = read_spirv_file(spv_file);
    if (opt_.keep_intermediates) r.glsl_source = std::move(glsl);
    r.spv_path = spv_file;
    return r;
  }

  // 4) Compile
  SpirvBinary bin = compiler_->compile_glsl_compute(glsl, opt_, spv_file, glsl_file);

  // 5) Cleanup intermediates if requested
  if (!opt_.keep_intermediates) {
    std::error_code ec;
    std::filesystem::remove(glsl_file, ec);
    // Keep SPV if cache enabled; otherwise remove too.
    if (!opt_.enable_cache) {
      std::filesystem::remove(spv_file, ec);
    }
  }

  SpirvCompileResult r;
  r.spirv = std::move(bin);
  r.glsl_source = opt_.keep_intermediates ? std::move(glsl) : std::string{};
  r.spv_path = (opt_.enable_cache || opt_.keep_intermediates) ? spv_file : std::filesystem::path{};
  return r;
}

} // namespace nodus::spirv
