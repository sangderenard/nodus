#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"
#include "tensor_calculator.h"

#include <memory>
#include <span>

namespace nodus::tensors {

struct CalculatorInstruction {
    tc_op op = TC_ADD;
    AbstractTensor* output = nullptr;
    const AbstractTensor* left = nullptr;
    const AbstractTensor* right = nullptr;
    double right_scalar = 0.0;
    bool has_scalar = false;
    bool reverse = false;
};

class CalculatorJob {
public:
    CalculatorJob() = default;
    CalculatorJob(CalculatorJob&&) noexcept;
    CalculatorJob& operator=(CalculatorJob&&) noexcept;
    ~CalculatorJob();

    CalculatorJob(const CalculatorJob&) = delete;
    CalculatorJob& operator=(const CalculatorJob&) = delete;

    bool valid() const;
    bool ready() const;
    bool wait();

private:
    struct Impl;
    explicit CalculatorJob(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    friend class InMemoryCalculator;
};

class InMemoryCalculator {
public:
    static InMemoryCalculator& instance();

    bool available() const;
    bool execute(std::span<const CalculatorInstruction> instructions);
    CalculatorJob submit(std::span<const CalculatorInstruction> instructions);
    tc_stats stats() const;

private:
    InMemoryCalculator();
    ~InMemoryCalculator();
    InMemoryCalculator(const InMemoryCalculator&) = delete;
    InMemoryCalculator& operator=(const InMemoryCalculator&) = delete;

    tc_calculator* calculator_ = nullptr;
};

} // namespace nodus::tensors
