// spirv_translation.cpp
// C++20 skeleton: IR -> (GLSL compute) -> SPIR-V via external compiler or shaderc.
// Drop-in translation pipeline with caching and specialization support.
//
// Philosophy:
// - Keep a tiny, explicit KernelIR that your SSA/graph can be adapted into.
// - Emit portable GLSL compute (vulkan/spirv) first; compile to SPIR-V.
// - Later: replace GLSL emission with a direct SPIR-V builder without changing the public API.
//
// Build:
// - C++20 required.
// - Optional: define NODUS_HAVE_SHADERC=1 and link shaderc if you embed compilation.
// - Otherwise: provide path to glslangValidator or ensure it is on PATH.
//
// Security note:
// - External compiler invocation uses args vector (no shell concatenation) on Windows;
//   on POSIX it uses fork/exec. Adjust to your platform policy.

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace nodus::spirv {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

static inline std::string to_string(std::filesystem::path p) {
  return p.u8string();
}

// ------------------------------
// Minimal, portable KernelIR view
// ------------------------------
//
// Adapt your SSA/function/region into this representation.
// It is intentionally small: enough to emit a compute kernel.

enum class ScalarType : u8 {
  I32,
  U32,
  F32,
  // TODO: F16, I16, U16, F64, etc.
};

struct TensorType {
  ScalarType scalar{};
  std::vector<u32> shape;         // Compile-time shape for v0; empty means "runtime" later.
  bool is_buffer = true;          // For v0: all tensors live in SSBO-like buffers.
  bool is_readonly = false;       // Inputs are typically readonly.
};

struct ValueRef {
  u32 id = 0; // index into KernelIR::values
};

enum class OpCode : u16 {
  // Core SSA-ish ops you likely already have.
  Const,
  Load,        // buffer + index -> scalar/vector
  Store,       // buffer + index + value -> void

  Add,
  Sub,
  Mul,
  Div,
  Neg,
  Min,
  Max,
  Clamp,

  // Comparisons/select
  CmpLT,
  CmpLE,
  CmpGT,
  CmpGE,
  CmpEQ,
  CmpNE,
  Select,      // (cond, a, b)

  // Elementary math (subset)
  Exp,
  Log,
  Sqrt,
  Rsqrt,

  // TODO: reductions, matmul, gather/scatter, etc.
};

struct Operand {
  // Either a ValueRef, or an immediate constant.
  // Extend as needed (e.g., small vector constants, strings, ids).
  std::variant<ValueRef, i32, u32, float> v;

  static Operand ref(ValueRef r) { return Operand{r}; }
  static Operand i(i32 x) { return Operand{x}; }
  static Operand u(u32 x) { return Operand{x}; }
  static Operand f(float x) { return Operand{x}; }
};

struct Instruction {
  OpCode op{};
  std::vector<Operand> inputs;
  std::vector<ValueRef> outputs; // Often 0 or 1 output for v0; keep vector for future.
};

struct ValueDef {
  TensorType type;                 // For scalar temps, shape.size()==0 and is_buffer=false.
  std::string debug_name;
};

// One compute kernel region.
struct KernelIR {
  std::string name;

  // Interface:
  // - buffers: “tensors” passed by binding index
  // - params: specialization/push-constant style inputs (for later)
  std::vector<ValueDef> values;           // includes buffers and temporaries
  std::vector<u32> buffer_value_ids;      // subset of values that are SSBOs
  std::vector<Instruction> instrs;

  // Work dispatch: global size known at runtime; local size (workgroup) chosen at compile.
  std::array<u32, 3> suggested_local_size{16, 16, 1};

  // For v0: 1D indexing (global invocation linear). Extend to 2D/3D.
  u32 element_count = 0; // How many logical elements to process; used for bounds checks.
};

// ------------------------------
// SPIR-V compilation outputs
// ------------------------------

struct SpirvBinary {
  std::vector<u32> words; // SPIR-V is u32 word stream
};

enum class TargetEnv : u8 {
  Vulkan_1_1,
  Vulkan_1_2,
  Vulkan_1_3,
  // TODO: OpenGL SPIR-V (ARB_gl_spirv) if you need it later.
};

struct SpirvCompileOptions {
  TargetEnv env = TargetEnv::Vulkan_1_2;

  // If using external compiler:
  std::filesystem::path glslang_validator_path; // empty => rely on PATH lookup

  // Caching:
  std::filesystem::path cache_dir = "shader_cache";
  bool enable_cache = true;

  // Debug:
  bool emit_debug_names = true;
  bool keep_intermediates = false;   // keep .comp and .spv artifacts
  bool verbose = false;

  // GLSL target:
  u32 glsl_version = 460;            // 450 or 460
};

struct SpirvCompileResult {
  SpirvBinary spirv{};
  std::string glsl_source;           // optionally retained
  std::filesystem::path spv_path;    // cache location if enabled
};

// ------------------------------
// Hash utilities (cache keys)
// ------------------------------

static inline u64 fnv1a64(std::span<const u8> data) {
  constexpr u64 kOffset = 1469598103934665603ull;
  constexpr u64 kPrime  = 1099511628211ull;
  u64 h = kOffset;
  for (u8 b : data) {
    h ^= static_cast<u64>(b);
    h *= kPrime;
  }
  return h;
}

static inline void hash_combine(u64& h, u64 v) {
  // Similar to boost hash combine.
  h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
}

static inline u64 hash_string(std::string_view s) {
  return fnv1a64(std::span<const u8>(reinterpret_cast<const u8*>(s.data()), s.size()));
}

// ------------------------------
// GLSL emission (compute shader)
// ------------------------------

class GlslEmitter {
public:
  explicit GlslEmitter(const SpirvCompileOptions& opt) : opt_(opt) {}

  std::string emit_compute_glsl(const KernelIR& k) const {
    std::ostringstream out;

    // Header and target. For Vulkan GLSL -> SPIR-V:
    out << "#version " << opt_.glsl_version << "\n";
    out << "#extension GL_EXT_scalar_block_layout : enable\n";
    out << "#extension GL_EXT_shader_explicit_arithmetic_types_int32 : enable\n";
    out << "#extension GL_EXT_shader_explicit_arithmetic_types_float32 : enable\n";
    out << "\n";

    // Workgroup size as compile-time layout.
    out << "layout(local_size_x=" << k.suggested_local_size[0]
        << ", local_size_y=" << k.suggested_local_size[1]
        << ", local_size_z=" << k.suggested_local_size[2] << ") in;\n\n";

    // Buffers: bind in order of k.buffer_value_ids
    // v0: scalar_block_layout to permit tight packing.
    for (u32 b = 0; b < static_cast<u32>(k.buffer_value_ids.size()); ++b) {
      const u32 vid = k.buffer_value_ids[b];
      const auto& v = k.values.at(vid);
      const char* glsl_ty = glsl_scalar_type(v.type.scalar);
      // We expose as an unsized array of scalars; higher-level shapes are address math.
      out << "layout(set=0, binding=" << b << ", scalar) "
          << (v.type.is_readonly ? "readonly " : "")
          << "buffer Buf" << b << " { " << glsl_ty << " data[]; } buf" << b << ";\n";
    }
    out << "\n";

    // Utility: linear global index (1D). Extend to 2D/3D when you add dims.
    out << "uint global_linear_id() {\n";
    out << "  uvec3 gid = gl_GlobalInvocationID;\n";
    out << "  // v0: treat x as the linear index.\n";
    out << "  return gid.x;\n";
    out << "}\n\n";

    // Main function prologue
    out << "void main() {\n";
    out << "  uint idx = global_linear_id();\n";
    out << "  if (idx >= " << k.element_count << "u) { return; }\n\n";

    // Emit temporaries for scalar values (buffers are accessed via bufN.data[...] directly).
    // We will map ValueRef ids to local variable names when not a buffer.
    std::vector<std::string> local_name(k.values.size());
    for (u32 i = 0; i < static_cast<u32>(k.values.size()); ++i) {
      if (is_buffer_value(k, i)) {
        local_name[i] = ""; // buffer
      } else {
        local_name[i] = make_local_name(i, k.values[i].debug_name);
      }
    }

    // Emit instructions
    for (const auto& ins : k.instrs) {
      emit_instruction(out, k, ins, local_name);
    }

    out << "}\n";
    return out.str();
  }

private:
  const SpirvCompileOptions& opt_;

  static const char* glsl_scalar_type(ScalarType t) {
    switch (t) {
      case ScalarType::I32: return "int";
      case ScalarType::U32: return "uint";
      case ScalarType::F32: return "float";
      default: return "uint";
    }
  }

  static bool is_buffer_value(const KernelIR& k, u32 value_id) {
    return std::find(k.buffer_value_ids.begin(), k.buffer_value_ids.end(), value_id) != k.buffer_value_ids.end();
  }

  static std::string make_local_name(u32 id, const std::string& dbg) {
    if (!dbg.empty()) {
      // Sanitize to a GLSL-friendly identifier.
      std::string s = dbg;
      for (char& c : s) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_';
      }
      return "v_" + std::to_string(id) + "_" + s;
    }
    return "v_" + std::to_string(id);
  }

  static std::string operand_expr(const KernelIR& k,
                                  const Operand& op,
                                  const std::vector<std::string>& local_name) {
    if (std::holds_alternative<ValueRef>(op.v)) {
      auto r = std::get<ValueRef>(op.v);
      // If it is a buffer, it must be accessed via Load/Store ops.
      // So here we assume ValueRef refers to a scalar temp.
      return local_name.at(r.id);
    }
    if (std::holds_alternative<i32>(op.v)) return std::to_string(std::get<i32>(op.v));
    if (std::holds_alternative<u32>(op.v)) return std::to_string(std::get<u32>(op.v)) + "u";
    if (std::holds_alternative<float>(op.v)) {
      std::ostringstream s; s << std::setprecision(9) << std::get<float>(op.v);
      return s.str();
    }
    return "0";
  }

  static void emit_instruction(std::ostringstream& out,
                               const KernelIR& k,
                               const Instruction& ins,
                               const std::vector<std::string>& local_name) {
    auto out0 = [&](u32 i = 0) -> std::string {
      if (ins.outputs.size() <= i) return "";
      return local_name.at(ins.outputs[i].id);
    };

    switch (ins.op) {
      case OpCode::Const: {
        // Const: outputs[0] = immediate
        // inputs[0] = immediate
        const auto dst = out0();
        const auto rhs = operand_expr(k, ins.inputs.at(0), local_name);
        out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst << " = " << rhs << ";\n";
      } break;

      case OpCode::Load: {
        // Load: (buffer_ref, index_expr) -> scalar
        // inputs[0] = ValueRef to buffer value id
        // inputs[1] = index (usually idx + offset)
        const auto dst = out0();
        auto buf = std::get<ValueRef>(ins.inputs.at(0).v).id;
        const auto ixs = operand_expr(k, ins.inputs.at(1), local_name);
        const u32 binding = buffer_binding_of(k, buf);
        out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
            << " = buf" << binding << ".data[" << ixs << "];\n";
      } break;

      case OpCode::Store: {
        // Store: (buffer_ref, index_expr, value_expr)
        auto buf = std::get<ValueRef>(ins.inputs.at(0).v).id;
        const auto ixs = operand_expr(k, ins.inputs.at(1), local_name);
        const auto vxs = operand_expr(k, ins.inputs.at(2), local_name);
        const u32 binding = buffer_binding_of(k, buf);
        out << "  buf" << binding << ".data[" << ixs << "] = " << vxs << ";\n";
      } break;

      // Binary ops:
      case OpCode::Add: emit_bin(out, k, ins, local_name, "+"); break;
      case OpCode::Sub: emit_bin(out, k, ins, local_name, "-"); break;
      case OpCode::Mul: emit_bin(out, k, ins, local_name, "*"); break;
      case OpCode::Div: emit_bin(out, k, ins, local_name, "/"); break;
      case OpCode::Min: emit_call2(out, k, ins, local_name, "min"); break;
      case OpCode::Max: emit_call2(out, k, ins, local_name, "max"); break;

      case OpCode::Neg: {
        const auto dst = out0();
        const auto a = operand_expr(k, ins.inputs.at(0), local_name);
        out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst << " = -" << a << ";\n";
      } break;

      case OpCode::Clamp: {
        const auto dst = out0();
        const auto x  = operand_expr(k, ins.inputs.at(0), local_name);
        const auto lo = operand_expr(k, ins.inputs.at(1), local_name);
        const auto hi = operand_expr(k, ins.inputs.at(2), local_name);
        out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
            << " = clamp(" << x << ", " << lo << ", " << hi << ");\n";
      } break;

      // Comparisons:
      case OpCode::CmpLT: emit_cmp(out, k, ins, local_name, "<"); break;
      case OpCode::CmpLE: emit_cmp(out, k, ins, local_name, "<="); break;
      case OpCode::CmpGT: emit_cmp(out, k, ins, local_name, ">"); break;
      case OpCode::CmpGE: emit_cmp(out, k, ins, local_name, ">="); break;
      case OpCode::CmpEQ: emit_cmp(out, k, ins, local_name, "=="); break;
      case OpCode::CmpNE: emit_cmp(out, k, ins, local_name, "!="); break;

      case OpCode::Select: {
        // Select: cond, a, b
        const auto dst = out0();
        const auto c = operand_expr(k, ins.inputs.at(0), local_name);
        const auto a = operand_expr(k, ins.inputs.at(1), local_name);
        const auto b = operand_expr(k, ins.inputs.at(2), local_name);
        out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
            << " = (" << c << ") ? (" << a << ") : (" << b << ");\n";
      } break;

      // Unary math:
      case OpCode::Exp:  emit_call1(out, k, ins, local_name, "exp"); break;
      case OpCode::Log:  emit_call1(out, k, ins, local_name, "log"); break;
      case OpCode::Sqrt: emit_call1(out, k, ins, local_name, "sqrt"); break;
      case OpCode::Rsqrt: {
        // GLSL has inversesqrt
        emit_call1(out, k, ins, local_name, "inversesqrt");
      } break;

      default:
        out << "  // TODO: unhandled opcode\n";
        break;
    }
  }

  static std::string glsl_decl_for_value(const KernelIR& k, ValueRef outv) {
    // For v0: scalar temps only. Extend to vectors.
    const auto& ty = k.values.at(outv.id).type;
    switch (ty.scalar) {
      case ScalarType::I32: return "int";
      case ScalarType::U32: return "uint";
      case ScalarType::F32: return "float";
      default: return "uint";
    }
  }

  static u32 buffer_binding_of(const KernelIR& k, u32 buffer_value_id) {
    auto it = std::find(k.buffer_value_ids.begin(), k.buffer_value_ids.end(), buffer_value_id);
    if (it == k.buffer_value_ids.end()) throw std::runtime_error("Load/Store buffer is not in buffer_value_ids");
    return static_cast<u32>(std::distance(k.buffer_value_ids.begin(), it));
  }

  static void emit_bin(std::ostringstream& out,
                       const KernelIR& k,
                       const Instruction& ins,
                       const std::vector<std::string>& local_name,
                       const char* op) {
    const auto dst = local_name.at(ins.outputs.at(0).id);
    const auto a = operand_expr(k, ins.inputs.at(0), local_name);
    const auto b = operand_expr(k, ins.inputs.at(1), local_name);
    out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
        << " = (" << a << ") " << op << " (" << b << ");\n";
  }

  static void emit_call1(std::ostringstream& out,
                         const KernelIR& k,
                         const Instruction& ins,
                         const std::vector<std::string>& local_name,
                         const char* fn) {
    const auto dst = local_name.at(ins.outputs.at(0).id);
    const auto a = operand_expr(k, ins.inputs.at(0), local_name);
    out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
        << " = " << fn << "(" << a << ");\n";
  }

  static void emit_call2(std::ostringstream& out,
                         const KernelIR& k,
                         const Instruction& ins,
                         const std::vector<std::string>& local_name,
                         const char* fn) {
    const auto dst = local_name.at(ins.outputs.at(0).id);
    const auto a = operand_expr(k, ins.inputs.at(0), local_name);
    const auto b = operand_expr(k, ins.inputs.at(1), local_name);
    out << "  " << glsl_decl_for_value(k, ins.outputs.at(0)) << " " << dst
        << " = " << fn << "(" << a << ", " << b << ");\n";
  }

  static void emit_cmp(std::ostringstream& out,
                       const KernelIR&,
                       const Instruction& ins,
                       const std::vector<std::string>& local_name,
                       const char* cmp) {
    // For v0: comparisons yield bool.
    const auto dst = local_name.at(ins.outputs.at(0).id);
    const auto a = operand_expr_placeholder(ins.inputs.at(0), local_name);
    const auto b = operand_expr_placeholder(ins.inputs.at(1), local_name);
    out << "  bool " << dst << " = (" << a << ") " << cmp << " (" << b << ");\n";
  }

  static std::string operand_expr_placeholder(const Operand& op,
                                              const std::vector<std::string>& local_name) {
    // same as operand_expr but without KernelIR
    if (std::holds_alternative<ValueRef>(op.v)) return local_name.at(std::get<ValueRef>(op.v).id);
    if (std::holds_alternative<i32>(op.v)) return std::to_string(std::get<i32>(op.v));
    if (std::holds_alternative<u32>(op.v)) return std::to_string(std::get<u32>(op.v)) + "u";
    if (std::holds_alternative<float>(op.v)) {
      std::ostringstream s; s << std::setprecision(9) << std::get<float>(op.v);
      return s.str();
    }
    return "0";
  }
};

// ------------------------------
// External compiler invocation
// ------------------------------

struct ProcessResult {
  int exit_code = -1;
  std::string stdout_text;
  std::string stderr_text;
};

#if defined(_WIN32)
#include <windows.h>

static ProcessResult run_process_capture(const std::vector<std::wstring>& argv) {
  // Minimal CreateProcessW + pipes skeleton.
  // NOTE: This is a skeleton; harden quoting and error paths for production.

  auto join_cmdline = [&]() -> std::wstring {
    std::wostringstream cmd;
    for (size_t i = 0; i < argv.size(); ++i) {
      if (i) cmd << L' ';
      // Quote each arg if it contains spaces.
      const auto& a = argv[i];
      const bool needs_quote = a.find(L' ') != std::wstring::npos || a.find(L'\t') != std::wstring::npos;
      if (needs_quote) cmd << L'"' << a << L'"';
      else cmd << a;
    }
    return cmd.str();
  };

  SECURITY_ATTRIBUTES sa{};
  sa.nLength = sizeof(sa);
  sa.bInheritHandle = TRUE;

  HANDLE out_read = nullptr, out_write = nullptr;
  if (!CreatePipe(&out_read, &out_write, &sa, 0)) {
    throw std::runtime_error("CreatePipe failed");
  }
  SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);

  STARTUPINFOW si{};
  si.cb = sizeof(si);
  si.dwFlags |= STARTF_USESTDHANDLES;
  si.hStdOutput = out_write;
  si.hStdError  = out_write;
  si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);

  PROCESS_INFORMATION pi{};
  std::wstring cmdline = join_cmdline();
  if (!CreateProcessW(nullptr, cmdline.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
    CloseHandle(out_read);
    CloseHandle(out_write);
    throw std::runtime_error("CreateProcessW failed");
  }

  CloseHandle(out_write); // parent reads

  std::string output;
  char buf[4096];
  DWORD read = 0;
  while (ReadFile(out_read, buf, sizeof(buf), &read, nullptr) && read > 0) {
    output.append(buf, buf + read);
  }

  WaitForSingleObject(pi.hProcess, INFINITE);

  DWORD exit_code = 1;
  GetExitCodeProcess(pi.hProcess, &exit_code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(out_read);

  ProcessResult r;
  r.exit_code = static_cast<int>(exit_code);
  r.stdout_text = output;
  return r;
}
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static ProcessResult run_process_capture(const std::vector<std::string>& argv) {
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
    for (auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
    args.push_back(nullptr);

    execvp(args[0], args.data());
    _exit(127);
  }

  // parent
  close(pipefd[1]);
  std::string output;
  char buf[4096];
  ssize_t n;
  while ((n = read(pipefd[0], buf, sizeof(buf))) > 0) {
    output.append(buf, buf + n);
  }
  close(pipefd[0]);

  int status = 0;
  waitpid(pid, &status, 0);
  int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;

  ProcessResult r;
  r.exit_code = exit_code;
  r.stdout_text = output;
  return r;
}
#endif

class ISpirvCompiler {
public:
  virtual ~ISpirvCompiler() = default;
  virtual SpirvBinary compile_glsl_compute(std::string_view glsl,
                                           const SpirvCompileOptions& opt,
                                           const std::filesystem::path& out_spv_path,
                                           const std::filesystem::path& out_glsl_path) = 0;
};

class ExternalGlslangValidatorCompiler final : public ISpirvCompiler {
public:
  SpirvBinary compile_glsl_compute(std::string_view glsl,
                                   const SpirvCompileOptions& opt,
                                   const std::filesystem::path& out_spv_path,
                                   const std::filesystem::path& out_glsl_path) override {
    std::filesystem::create_directories(out_spv_path.parent_path());

    // Write GLSL to disk
    {
      std::ofstream f(out_glsl_path, std::ios::binary);
      if (!f) throw std::runtime_error("Failed to open GLSL output file");
      f.write(glsl.data(), static_cast<std::streamsize>(glsl.size()));
    }

    // Invoke glslangValidator to emit SPIR-V
#if defined(_WIN32)
    std::wstring exe = opt.glslang_validator_path.empty()
      ? L"glslangValidator.exe"
      : opt.glslang_validator_path.wstring();

    std::vector<std::wstring> argv{
      exe,
      L"-V",
      out_glsl_path.wstring(),
      L"-o",
      out_spv_path.wstring(),
    };
    auto pr = run_process_capture(argv);
#else
    std::string exe = opt.glslang_validator_path.empty()
      ? "glslangValidator"
      : opt.glslang_validator_path.string();

    std::vector<std::string> argv{
      exe,
      "-V",
      out_glsl_path.string(),
      "-o",
      out_spv_path.string(),
    };
    auto pr = run_process_capture(argv);
#endif

    if (opt.verbose) {
      // NOTE: stdout_text may include both stdout/stderr on some implementations.
      std::fprintf(stderr, "[glslangValidator] exit=%d\n%s\n", pr.exit_code, pr.stdout_text.c_str());
    }

    if (pr.exit_code != 0) {
      throw std::runtime_error("glslangValidator failed:\n" + pr.stdout_text);
    }

    return read_spirv_file(out_spv_path);
  }

private:
  static SpirvBinary read_spirv_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open SPIR-V file: " + to_string(p));
    std::vector<u8> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() % 4 != 0) throw std::runtime_error("SPIR-V file size not multiple of 4");
    SpirvBinary bin;
    bin.words.resize(bytes.size() / 4);
    std::memcpy(bin.words.data(), bytes.data(), bytes.size());
    return bin;
  }
};

// ------------------------------
// Cache key + driver
// ------------------------------

struct CacheKey {
  u64 h = 0;

  static CacheKey from_kernel(const KernelIR& k, const SpirvCompileOptions& opt, std::string_view glsl) {
    u64 h = 0;
    hash_combine(h, hash_string(k.name));
    hash_combine(h, static_cast<u64>(opt.env));
    hash_combine(h, static_cast<u64>(opt.glsl_version));
    hash_combine(h, hash_string(glsl));
    return CacheKey{h};
  }

  std::string hex() const {
    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << h;
    return s.str();
  }
};

class SpirvTranslator {
public:
  explicit SpirvTranslator(SpirvCompileOptions opt,
                           std::unique_ptr<ISpirvCompiler> compiler = {})
    : opt_(std::move(opt)),
      compiler_(compiler ? std::move(compiler) : std::make_unique<ExternalGlslangValidatorCompiler>()),
      emitter_(opt_) {
    std::filesystem::create_directories(opt_.cache_dir);
  }

  SpirvCompileResult translate_kernel_to_spirv(const KernelIR& k) {
    // 1) Emit GLSL
    std::string glsl = emitter_.emit_compute_glsl(k);

    // 2) Cache key
    CacheKey key = CacheKey::from_kernel(k, opt_, glsl);
    const auto base = opt_.cache_dir / (k.name + "_" + key.hex());
    const auto glsl_path = base;
    const auto spv_path = base;
    auto glsl_file = glsl_path; glsl_file += ".comp";
    auto spv_file  = spv_path;  spv_file  += ".spv";

    // 3) Cache hit
    if (opt_.enable_cache && std::filesystem::exists(spv_file)) {
      SpirvCompileResult r;
      r.spirv = read_spirv_file(spv_file);
      if (opt_.keep_intermediates) r.glsl_source = std::move(glsl);
      r.spv_path = spv_file;
      return r;
    }

    // 4) Compile (external or embedded)
    SpirvBinary bin = compiler_->compile_glsl_compute(glsl, opt_, spv_file, glsl_file);

    // 5) Optionally clean intermediates
    if (!opt_.keep_intermediates) {
      std::error_code ec;
      std::filesystem::remove(glsl_file, ec);
    }

    SpirvCompileResult r;
    r.spirv = std::move(bin);
    r.glsl_source = opt_.keep_intermediates ? std::move(glsl) : std::string{};
    r.spv_path = spv_file;
    return r;
  }

private:
  SpirvCompileOptions opt_;
  std::unique_ptr<ISpirvCompiler> compiler_;
  GlslEmitter emitter_;

  static SpirvBinary read_spirv_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("Failed to open SPIR-V file: " + to_string(p));
    std::vector<u8> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (bytes.size() % 4 != 0) throw std::runtime_error("SPIR-V file size not multiple of 4");
    SpirvBinary bin;
    bin.words.resize(bytes.size() / 4);
    std::memcpy(bin.words.data(), bytes.data(), bytes.size());
    return bin;
  }
};

// ------------------------------
// CMake FetchContent snippet (copy-paste reference)
// ------------------------------
//
// You can either:
// (A) Use external glslangValidator (fastest): install Vulkan SDK or glslang tools and set path.
// (B) Fetch shaderc or glslang as build deps and compile in-process.
//
// This is a reference snippet; tailor to your repo layout.
//
// const char* kFetchContentSnippet = R"cmake(
// include(FetchContent)
//
// # Option 1: Shaderc (embedded GLSL->SPIR-V compiler)
// FetchContent_Declare(
//   shaderc
//   GIT_REPOSITORY https://github.com/google/shaderc.git
//   GIT_TAG v2024.3
// )
// set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
// set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
// FetchContent_MakeAvailable(shaderc)
// target_link_libraries(your_target PRIVATE shaderc)
//
// # Option 2: glslang (if you want to call into it directly)
// FetchContent_Declare(
//   glslang
//   GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
//   GIT_TAG 14.3.0
// )
// set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE)
// FetchContent_MakeAvailable(glslang)
// )cmake";

} // namespace nodus::spirv
