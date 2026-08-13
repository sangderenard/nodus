// kir_to_spirv_main.cpp
// The membrane crossing as a command: read one KIRTEXT document (emitted by
// turing's kernel_ir_lowering.serialize_kernel_ir), assemble it directly to
// SPIR-V, and write the binary module. Shortfalls print by name and refuse
// the write — an incomplete kernel never masquerades as a lowered one.
//
//   kir_to_spirv <in.kir> <out.spv>

#include "kernel_ir_text.h"
#include "kernel_spirv_assembler.h"
#include "../src/kernel_isa.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: kir_to_spirv <in.kir> <out.spv>\n");
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    std::ostringstream text;
    text << in.rdbuf();

    try {
        const nodus::spirv::KernelIR kernel =
            nodus::kernels::parse_kernel_ir_text(text.str());
        const nodus::kernels::SpirvAssembly assembly =
            nodus::kernels::assemble_kernel_ir_to_spirv(kernel);
        for (const auto& reason : assembly.shortfalls) {
            std::fprintf(stderr, "shortfall: %s\n", reason.c_str());
        }
        if (!assembly.complete()) {
            std::fprintf(stderr, "REFUSED: kernel '%s', %zu shortfall(s)\n",
                         kernel.name.c_str(), assembly.shortfalls.size());
            return 1;
        }
        std::ofstream out(argv[2], std::ios::binary);
        if (!out) {
            std::fprintf(stderr, "cannot write %s\n", argv[2]);
            return 2;
        }
        out.write(
            reinterpret_cast<const char*>(assembly.binary.words.data()),
            static_cast<std::streamsize>(assembly.binary.words.size() * 4));
        std::printf("%s: %zu words, id bound %u -> %s\n",
                    kernel.name.c_str(), assembly.binary.words.size(),
                    assembly.id_bound, argv[2]);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
