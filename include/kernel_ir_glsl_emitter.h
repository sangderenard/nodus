#pragma once

#include <cstdint>
#include <sstream>
#include <string>

#include "kernel_isa.h"

namespace nodus {
namespace kernel {

// Expose KernelIR from the spirv namespace for GLSL emission.
using KernelIR = ::nodus::spirv::KernelIR;

struct GlslEmitOptions {
    uint32_t glsl_version = 460;
    bool enable_scalar_block_layout = true;
    bool emit_debug_comments = false;
};

// Minimal placeholder emitter that produces a valid GLSL compute shader body.
class GlslEmitter {
public:
    explicit GlslEmitter(GlslEmitOptions opt = {}) : opt_(opt) {}

    std::string emit(const KernelIR& k) const {
        std::ostringstream ss;
        ss << "#version " << opt_.glsl_version << " core\n";
        if (opt_.emit_debug_comments) {
            ss << "// kernel: " << k.name << "\n";
        }
        // Stub compute entry point; real implementation should lower KernelIR to GLSL.
        ss << "layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;\n";
        ss << "void main() { }\n";
        return ss.str();
    }

private:
    GlslEmitOptions opt_;
};

} // namespace kernel
} // namespace nodus
