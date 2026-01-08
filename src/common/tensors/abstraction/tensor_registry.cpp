#include "common/tensors/abstraction/tensor_registry.h"

#include <mutex>
#include <string>
#include <unordered_map>

namespace nodus::tensors {

namespace {
std::mutex g_mu;
std::unordered_map<std::string, TensorBackend*> g_backends;
TensorBackend* g_default_backend = nullptr;
} // namespace

void register_backend(TensorBackend* backend, bool make_default) {
    if (!backend) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_backends[backend->name()] = backend;
    if (make_default || !g_default_backend) {
        g_default_backend = backend;
    }
}

TensorBackend* find_backend(std::string_view name) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_backends.find(std::string(name));
    return (it == g_backends.end()) ? nullptr : it->second;
}

TensorBackend* default_backend() {
    std::lock_guard<std::mutex> lk(g_mu);
    return g_default_backend;
}

void set_default_backend(TensorBackend* backend) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_default_backend = backend;
}

} // namespace nodus::tensors
