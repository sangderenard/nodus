#pragma once

namespace nodus::tensors {

struct RuntimeConfig {
    int reserved = 0;
};

class Runtime {
public:
    Runtime() = default;
    explicit Runtime(const RuntimeConfig& config);

    void reset(const RuntimeConfig& config);
};

Runtime& runtime_singleton();

} // namespace nodus::tensors
