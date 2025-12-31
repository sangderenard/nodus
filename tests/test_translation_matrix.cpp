#include "translation_matrix.h"
#include "kernel_isa.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <string>

using namespace nodus::kernels;
using namespace nodus::spirv;

// Dummy translation functions for demonstration
void dummy_cpu_translation(const KernelIR& ir) {
    std::cout << "CPU translation for kernel: " << ir.name << std::endl;
}
void dummy_spirv_translation(const KernelIR& ir) {
    std::cout << "SPIR-V translation for kernel: " << ir.name << std::endl;
}

int main() {
    // Register two backends
    TranslationMatrix::instance().register_backend("cpu", dummy_cpu_translation);
    TranslationMatrix::instance().register_backend("spirv", dummy_spirv_translation);

    // Create a dummy kernel IR
    KernelIR ir;
    ir.name = "aspirational_test_kernel";
    // ...populate IR with some nodes if needed...

    // Try all registered backends
    std::vector<std::string> backends = {"cpu", "spirv", "metal", "onnx"};
    int fail_count = 0;
    for (const auto& backend : backends) {
        auto fn = TranslationMatrix::instance().get(backend);
        if (fn) {
            std::cout << "Running translation for backend: " << backend << std::endl;
            fn(ir);
        } else {
            std::cout << "No translation registered for backend: " << backend << " (expected for some)" << std::endl;
            ++fail_count;
        }
    }
    // We expect at least one backend to be missing
    assert(fail_count > 0);
    std::cout << "Aspirational translation matrix test complete. Failures: " << fail_count << std::endl;
    return 0;
}
