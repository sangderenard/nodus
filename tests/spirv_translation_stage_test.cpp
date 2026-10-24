#include "spirv_translation.h"
#include "kernel_isa.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace nodus::spirv;

struct CapturingCompiler final : ISpirvCompiler {
    GlslCompileRequest captured;
    std::filesystem::path captured_spv_path;
    std::filesystem::path captured_glsl_path;
    bool called = false;

    SpirvBinary compile_glsl_source(
        const GlslCompileRequest& request,
        const SpirvCompileOptions&,
        const std::filesystem::path& spv_file,
        const std::filesystem::path& glsl_file) override {
        captured = request;
        captured_spv_path = spv_file;
        captured_glsl_path = glsl_file;
        called = true;
        return SpirvBinary{{0x07230203u, 0x00010000u}};
    }

    SpirvBinary compile_glsl_compute(
        const std::string&,
        const SpirvCompileOptions&,
        const std::filesystem::path&,
        const std::filesystem::path&) override {
        throw std::runtime_error("stage-aware path unexpectedly used legacy compile entry");
    }
};

struct LegacyCompiler final : ISpirvCompiler {
    SpirvCompileOptions captured_options;
    bool called = false;

    SpirvBinary compile_glsl_compute(
        const std::string&,
        const SpirvCompileOptions& options,
        const std::filesystem::path&,
        const std::filesystem::path&) override {
        captured_options = options;
        called = true;
        return SpirvBinary{{0x07230203u}};
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int run_as_fake_glslang(int argc, char** argv) {
    std::string stage;
    std::string entry_point;
    std::string target;
    std::filesystem::path output;
    bool saw_define = false;
    bool saw_spaced_define = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-S" || arg == "-e" || arg == "--target-env" || arg == "-o")
            && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "-S") stage = value;
            else if (arg == "-e") entry_point = value;
            else if (arg == "--target-env") target = value;
            else output = value;
        } else if (arg == "-DNODUS_MODE=7") {
            saw_define = true;
        } else if (arg == "-DNODUS_LABEL=two words") {
            saw_spaced_define = true;
        }
    }
    if (stage != "frag" || entry_point != "shade"
        || target != "vulkan1.3" || !saw_define || !saw_spaced_define
        || output.empty()) {
        return 41;
    }
    const std::vector<std::uint32_t> words{0x07230203u, 0x00010000u};
    std::ofstream stream(output, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(words.data()),
                 static_cast<std::streamsize>(words.size() * sizeof(words.front())));
    return stream ? 0 : 42;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "-V") {
        return run_as_fake_glslang(argc, argv);
    }
    const auto cache_dir = std::filesystem::temp_directory_path()
        / "nodus_spirv_translation_stage_test";
    std::error_code ec;
    std::filesystem::remove_all(cache_dir, ec);

    SpirvCompileOptions options;
    options.cache_dir = cache_dir;
    options.enable_cache = false;
    options.keep_intermediates = false;

    auto compiler = std::make_unique<CapturingCompiler>();
    auto* capture = compiler.get();
    SpirvTranslator translator(options, std::move(compiler));

    GlslCompileRequest fragment;
    fragment.source = "#version 460\nvoid shade() {}\n";
    fragment.stage = ShaderStage::Fragment;
    fragment.entry_point = "shade";
    fragment.target_env = TargetEnv::Vulkan_1_3;
    fragment.source_identity = "deployed/material.glsl";
    fragment.defines = {{"NODUS_MODE", "7"}, {"NODUS_LABEL", "two words"}};

    const auto result = translator.compile_glsl_to_spirv(fragment);
    require(capture->called, "stage-aware compiler was not called");
    require(capture->captured.stage == ShaderStage::Fragment, "stage was not preserved");
    require(capture->captured.entry_point == "shade", "entry point was not preserved");
    require(capture->captured.target_env == TargetEnv::Vulkan_1_3, "target was not preserved");
    require(capture->captured.source_identity == fragment.source_identity, "source identity was not preserved");
    require(capture->captured.defines == fragment.defines, "defines were not preserved");
    require(capture->captured_glsl_path.extension() == ".frag", "stage-specific source extension was not used");
    require(result.spirv.words.size() == 2, "compiler result was not returned");
    require(result.spv_path.empty(), "non-cached result exposed a stale path");

    KernelIR kernel;
    kernel.name = "existing_kernel_path";
    (void)translator.translate_kernel_to_spirv(kernel);
    require(capture->captured.stage == ShaderStage::Compute,
            "KernelIR translation no longer requests compute stage");
    require(capture->captured.entry_point == "main",
            "KernelIR translation no longer requests main entry point");
    require(capture->captured.source.find("void main()") != std::string::npos,
            "KernelIR GLSL emitter behavior changed");

    LegacyCompiler legacy;
    GlslCompileRequest compute;
    compute.source = "#version 460\nvoid main() {}\n";
    compute.target_env = TargetEnv::Vulkan_1_1;
    compute.defines = {{"LEGACY", "1"}};
    (void)legacy.compile_glsl_source(compute, options, {}, {});
    require(legacy.called, "legacy compute adapter was not called");
    require(legacy.captured_options.env == TargetEnv::Vulkan_1_1, "legacy target was not adapted");
    require(legacy.captured_options.defines == compute.defines, "legacy defines were not adapted");

    bool rejected = false;
    compute.stage = ShaderStage::Vertex;
    try {
        (void)legacy.compile_glsl_source(compute, options, {}, {});
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "legacy compiler accepted an unsupported stage");

    rejected = false;
    fragment.defines = {{"NOT-A-MACRO", "1"}};
    try {
        (void)translator.compile_glsl_to_spirv(fragment);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "invalid macro name was accepted");

    rejected = false;
    fragment.defines.clear();
    fragment.stage = static_cast<ShaderStage>(255);
    try {
        (void)translator.compile_glsl_to_spirv(fragment);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "unknown shader stage was accepted");

    // Exercise the real external-compiler adapter with this test executable
    // acting as a strict fake glslangValidator.  It exits nonzero unless the
    // stage, entry point, target environment, and define are separate argv
    // values with the expected contents.
    options.glslang_validator_path = std::filesystem::absolute(argv[0]);
    SpirvTranslator external_translator(options);
    fragment.stage = ShaderStage::Fragment;
    fragment.defines = {{"NODUS_MODE", "7"}, {"NODUS_LABEL", "two words"}};
    const auto external_result = external_translator.compile_glsl_to_spirv(fragment);
    require(external_result.spirv.words.size() == 2,
            "external compiler adapter did not return the fake SPIR-V");

    std::filesystem::remove_all(cache_dir, ec);
    std::cout << "spirv translation stage contract passed\n";
    return 0;
}
