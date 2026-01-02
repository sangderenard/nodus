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
};

// Main translation interface
class SpirvTranslator {
public:
    explicit SpirvTranslator(SpirvCompileOptions opt = {}, std::unique_ptr<class ISpirvCompiler> compiler = {});
    SpirvCompileResult translate_kernel_to_spirv(const KernelIR& k);

private:
    SpirvCompileOptions opt_;
    std::unique_ptr<ISpirvCompiler> compiler_;
};

// Optionally expose helpers for constructing IR, options, etc.

} // namespace nodus::spirv
