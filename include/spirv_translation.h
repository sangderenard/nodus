#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

namespace nodus::spirv {

using u8  = std::uint8_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;
using u64 = std::uint64_t;
using u16 = std::uint16_t;

// Forward declarations for key IR and translation types
struct KernelIR;

// Definition for SpirvBinary (was only forward declared)
struct SpirvBinary {
    std::vector<u32> words; // SPIR-V is u32 word stream
};

enum class TargetEnv : uint8_t {
    Vulkan_1_1,
    Vulkan_1_2,
    Vulkan_1_3,
};

// Shader stage is part of the compilation contract.  Do not infer it from a
// file extension: deployed shader source may be retained under an arbitrary
// source identity before Nodus translates it to another shader language.
enum class ShaderStage : uint8_t {
    Vertex,
    TessellationControl,
    TessellationEvaluation,
    Geometry,
    Fragment,
    Compute,
};

struct GlslCompileRequest {
    std::string source;
    ShaderStage stage = ShaderStage::Compute;
    std::string entry_point = "main";
    TargetEnv target_env = TargetEnv::Vulkan_1_2;
    std::string source_identity = "shader.comp";
    std::vector<std::pair<std::string, std::string>> defines;
};

struct SpirvCompileOptions {
    TargetEnv env = TargetEnv::Vulkan_1_2;
    std::filesystem::path glslang_validator_path; // empty => rely on PATH lookup
    std::filesystem::path cache_dir = "shader_cache";
    bool enable_cache = true;
    bool emit_debug_names = true;
    bool keep_intermediates = false;
    bool verbose = false;
    uint32_t glsl_version = 460;
    std::vector<std::pair<std::string, std::string>> defines;
    bool enable_scalar_block_layout = true;
    bool emit_debug_comments = false;
};

struct SpirvCompileResult {
    SpirvBinary spirv{};
    std::string glsl_source;
    std::filesystem::path spv_path;
};

class ISpirvCompiler {
public:
    virtual ~ISpirvCompiler() = default;
    virtual SpirvBinary compile_glsl_compute(const std::string& glsl, const SpirvCompileOptions& opt,
                                            const std::filesystem::path& spv_file,
                                            const std::filesystem::path& glsl_file) = 0;
    // General GLSL source seam.  The default adapter deliberately supports
    // compute/main only so existing injected compilers remain source
    // compatible while failing closed for contracts they cannot honor.
    virtual SpirvBinary compile_glsl_source(const GlslCompileRequest& request,
                                            const SpirvCompileOptions& opt,
                                            const std::filesystem::path& spv_file,
                                            const std::filesystem::path& glsl_file);
};

// Main translation interface
class SpirvTranslator {
public:
    explicit SpirvTranslator(SpirvCompileOptions opt = {}, std::unique_ptr<class ISpirvCompiler> compiler = {});
    SpirvCompileResult compile_glsl_to_spirv(const GlslCompileRequest& request);
    SpirvCompileResult translate_kernel_to_spirv(const KernelIR& k);

private:
    SpirvCompileOptions opt_;
    std::unique_ptr<ISpirvCompiler> compiler_;
};

// Optionally expose helpers for constructing IR, options, etc.

} // namespace nodus::spirv
