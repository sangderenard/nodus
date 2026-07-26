#pragma once

#include "canonical_ops.h"
#include "common/tensors/abstraction/abstract_tensor.h"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace nodus::tensors {

class InMemoryBackend;

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

struct FusedProgramStep {
    nodus::ops::CanonicalOp op = nodus::ops::CanonicalOp::ADD;
    uint64_t result_id = 0;
    std::vector<uint64_t> input_ids;
    double right_scalar = 0.0;
    bool has_scalar = false;
    bool reverse = false;
};

// Versioned wire view of Turing's equal-shape FusedProgram. This carries
// canonical names/value ids across a language boundary; it is not a second
// semantic graph or an alternate arithmetic implementation.
struct FusedProgramTransport {
    uint64_t version = 0;
    std::vector<uint64_t> feed_ids;
    std::vector<FusedProgramStep> steps;
    uint64_t output_id = 0;
    bool has_output = false;
};

bool parse_fused_program_transport(
    std::istream& stream,
    FusedProgramTransport* output,
    std::string* error = nullptr);

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

// Persistent binding of transport value ids to Nodus-owned tensors. Program
// preparation allocates once; execute() delegates every operation to the
// singleton TensorMath-backed calculator.
class PreparedCalculatorProgram {
public:
    static std::unique_ptr<PreparedCalculatorProgram> create(
        const FusedProgramTransport& program,
        const TensorDesc& desc,
        InMemoryBackend* backend,
        std::string* error = nullptr);

    AbstractTensor* feed(uint64_t value_id);
    const AbstractTensor* feed(uint64_t value_id) const;
    AbstractTensor* output() { return output_; }
    const AbstractTensor* output() const { return output_; }
    size_t instruction_count() const { return instructions_.size(); }
    bool execute();

private:
    std::vector<std::unique_ptr<AbstractTensor>> storage_;
    std::vector<std::pair<uint64_t, AbstractTensor*>> feeds_;
    std::vector<CalculatorInstruction> instructions_;
    AbstractTensor* output_ = nullptr;
};

} // namespace nodus::tensors
