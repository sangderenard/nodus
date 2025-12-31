// kernels/translation_matrix.h
// Registry for translation helpers between SSA/KernelIR and backend kernels
#pragma once
#include <string>
#include <unordered_map>
#include <functional>
#include "kernel_isa.h"

namespace nodus {
namespace kernels {

// Example: translation function signature
using TranslationFn = std::function<void(const spirv::KernelIR&)>;

// Registry for translation helpers (backend name -> function)
class TranslationMatrix {
public:
    static TranslationMatrix& instance() {
        static TranslationMatrix inst;
        return inst;
    }
    void register_backend(const std::string& name, TranslationFn fn) {
        registry_[name] = fn;
    }
    TranslationFn get(const std::string& name) const {
        auto it = registry_.find(name);
        if (it != registry_.end()) return it->second;
        return nullptr;
    }
private:
    std::unordered_map<std::string, TranslationFn> registry_;
};

} // namespace kernels
} // namespace nodus
