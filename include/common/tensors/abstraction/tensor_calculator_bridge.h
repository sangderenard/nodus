#pragma once

#include "canonical_ops.h"
#include "common/tensors/abstraction/abstract_tensor.h"

#include <cstdint>
#include <span>

namespace nodus::tensors {

struct CalculatorInstruction {
    nodus::ops::CanonicalOp op = nodus::ops::CanonicalOp::ADD;
    AbstractTensor* output = nullptr;
    const AbstractTensor* left = nullptr;
    const AbstractTensor* right = nullptr;
    double right_scalar = 0.0;
    bool has_scalar = false;
    bool reverse = false;
};

struct CalculatorStats {
    uint64_t programs_executed = 0;
    uint64_t instructions_executed = 0;
    uint64_t inline_jobs = 0;
};

class CalculatorJob {
public:
    bool valid() const { return valid_; }
    bool ready() const { return valid_; }
    bool wait() const { return valid_ && succeeded_; }

private:
    CalculatorJob(bool valid, bool succeeded)
        : valid_(valid), succeeded_(succeeded) {}
    bool valid_ = false;
    bool succeeded_ = false;
    friend class InMemoryCalculator;
};

// Persistent calculator view over Nodus-owned AbstractTensors. Arithmetic is
// executed exclusively by tensor_elementwise_* in tensor_math.cpp; this class
// owns no alternate scalar operator implementation or shadow tensor registry.
class InMemoryCalculator {
public:
    static InMemoryCalculator& instance();

    bool available() const { return true; }
    bool execute(std::span<const CalculatorInstruction> instructions);

    // Submission is inline today. TensorMath may parallelize the individual
    // operations through Nodus's shared thread pool; no second worker queue,
    // mutex, or tensor-handle map is introduced here.
    CalculatorJob submit(std::span<const CalculatorInstruction> instructions);
    CalculatorStats stats() const { return stats_; }

private:
    CalculatorStats stats_{};
};

} // namespace nodus::tensors
