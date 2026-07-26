#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include "common/tensors/abstraction/tensor_math.h"

namespace nodus::tensors {

InMemoryCalculator& InMemoryCalculator::instance() {
    static InMemoryCalculator calculator;
    return calculator;
}

bool InMemoryCalculator::execute(
    std::span<const CalculatorInstruction> instructions) {
    if (instructions.empty()) return false;
    uint64_t completed = 0;
    for (const auto& instruction : instructions) {
        if (!instruction.output || !instruction.left) return false;
        bool ok = false;
        if (instruction.right) {
            if (instruction.has_scalar) return false;
            ok = instruction.reverse
                ? tensor_elementwise_binary(
                    instruction.op,
                    *instruction.right,
                    *instruction.left,
                    instruction.output)
                : tensor_elementwise_binary(
                    instruction.op,
                    *instruction.left,
                    *instruction.right,
                    instruction.output);
        } else if (instruction.has_scalar) {
            ok = tensor_elementwise_scalar(
                instruction.op,
                *instruction.left,
                instruction.right_scalar,
                instruction.reverse,
                instruction.output);
        } else {
            ok = tensor_elementwise_unary(
                instruction.op, *instruction.left, instruction.output);
        }
        if (!ok) return false;
        ++completed;
    }
    ++stats_.programs_executed;
    stats_.instructions_executed += completed;
    return true;
}

CalculatorJob InMemoryCalculator::submit(
    std::span<const CalculatorInstruction> instructions) {
    const bool succeeded = execute(instructions);
    if (succeeded) ++stats_.inline_jobs;
    return CalculatorJob(true, succeeded);
}

} // namespace nodus::tensors
