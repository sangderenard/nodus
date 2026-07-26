#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"

#include <istream>
#include <unordered_map>
#include <unordered_set>

namespace nodus::tensors {

bool parse_fused_program_transport(
    std::istream& stream,
    FusedProgramTransport* output,
    std::string* error) {
    if (!output) return false;
    *output = {};
    std::string token;
    if (!(stream >> token >> output->version) || token != "fused_program" ||
        output->version != 1) {
        if (error) *error = "expected 'fused_program 1' header";
        return false;
    }

    std::unordered_set<uint64_t> defined;
    bool ended = false;
    while (stream >> token) {
        if (token == "feed") {
            uint64_t value_id = 0;
            if (!(stream >> value_id) || !defined.insert(value_id).second) {
                if (error) *error = "invalid or duplicate feed id";
                return false;
            }
            output->feed_ids.push_back(value_id);
            continue;
        }
        if (token == "step") {
            uint64_t step_id = 0;
            std::string op_name;
            FusedProgramStep step;
            size_t input_count = 0;
            int has_scalar = 0;
            int reverse = 0;
            if (!(stream >> step_id >> op_name >> step.result_id >>
                  input_count)) {
                if (error) *error = "malformed FusedProgram step";
                return false;
            }
            const auto* descriptor = nodus::ops::find_op(op_name);
            if (!descriptor || descriptor->ct_value < 0 ||
                (descriptor->arity != 1 && descriptor->arity != 2)) {
                if (error) *error = "unsupported canonical operation: " + op_name;
                return false;
            }
            step.op = static_cast<nodus::ops::CanonicalOp>(
                descriptor->canonical_id);
            step.input_ids.resize(input_count);
            for (auto& input_id : step.input_ids) {
                if (!(stream >> input_id) || !defined.contains(input_id)) {
                    if (error) *error = "step reads an unavailable value id";
                    return false;
                }
            }
            if (!(stream >> has_scalar >> step.right_scalar >> reverse) ||
                (has_scalar != 0 && has_scalar != 1) ||
                (reverse != 0 && reverse != 1)) {
                if (error) *error = "malformed FusedProgram operand flags";
                return false;
            }
            step.has_scalar = has_scalar != 0;
            step.reverse = reverse != 0;
            const bool valid_unary =
                descriptor->arity == 1 && input_count == 1 && !step.has_scalar;
            const bool valid_tensor_binary =
                descriptor->arity == 2 && input_count == 2 && !step.has_scalar;
            const bool valid_scalar_binary =
                descriptor->arity == 2 && input_count == 1 && step.has_scalar;
            if (!valid_unary && !valid_tensor_binary && !valid_scalar_binary) {
                if (error) *error = "operation arity does not match operands";
                return false;
            }
            if (!defined.insert(step.result_id).second) {
                if (error) *error = "duplicate FusedProgram result id";
                return false;
            }
            output->steps.push_back(std::move(step));
            continue;
        }
        if (token == "output") {
            if (output->has_output || !(stream >> output->output_id) ||
                !defined.contains(output->output_id)) {
                if (error) {
                    *error =
                        "duplicate output or unavailable output value id";
                }
                return false;
            }
            output->has_output = true;
            continue;
        }
        if (token == "end") {
            ended = true;
            break;
        }
        if (error) *error = "unknown FusedProgram transport token: " + token;
        return false;
    }
    if (!ended || output->feed_ids.empty() || output->steps.empty() ||
        !output->has_output) {
        if (error) {
            *error = "FusedProgram requires feeds, steps, one output, and end";
        }
        return false;
    }
    return true;
}

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

std::unique_ptr<PreparedCalculatorProgram> PreparedCalculatorProgram::create(
    const FusedProgramTransport& program,
    const TensorDesc& desc,
    InMemoryBackend* backend,
    std::string* error) {
    if (!backend || program.feed_ids.empty() || program.steps.empty() ||
        !program.has_output) {
        if (error) *error = "invalid calculator program or backend";
        return nullptr;
    }
    auto prepared = std::unique_ptr<PreparedCalculatorProgram>(
        new PreparedCalculatorProgram());
    prepared->storage_.reserve(
        program.feed_ids.size() + program.steps.size());
    std::unordered_map<uint64_t, AbstractTensor*> values;

    for (const uint64_t feed_id : program.feed_ids) {
        auto tensor = std::make_unique<AbstractTensor>(desc, backend);
        if (!tensor->valid() || values.contains(feed_id)) {
            if (error) *error = "failed to bind FusedProgram feed";
            return nullptr;
        }
        auto* pointer = tensor.get();
        values.emplace(feed_id, pointer);
        prepared->feeds_.emplace_back(feed_id, pointer);
        prepared->storage_.push_back(std::move(tensor));
    }

    prepared->instructions_.reserve(program.steps.size());
    for (const auto& step : program.steps) {
        if (step.input_ids.empty() || !values.contains(step.input_ids[0]) ||
            (step.input_ids.size() == 2 &&
             !values.contains(step.input_ids[1])) ||
            values.contains(step.result_id)) {
            if (error) *error = "FusedProgram contains unavailable value ids";
            return nullptr;
        }
        auto tensor = std::make_unique<AbstractTensor>(desc, backend);
        if (!tensor->valid()) {
            if (error) *error = "failed to allocate FusedProgram result";
            return nullptr;
        }
        CalculatorInstruction instruction{};
        instruction.op = step.op;
        instruction.output = tensor.get();
        instruction.left = values.at(step.input_ids[0]);
        if (step.input_ids.size() == 2)
            instruction.right = values.at(step.input_ids[1]);
        instruction.right_scalar = step.right_scalar;
        instruction.has_scalar = step.has_scalar;
        instruction.reverse = step.reverse;
        values.emplace(step.result_id, tensor.get());
        prepared->storage_.push_back(std::move(tensor));
        prepared->instructions_.push_back(instruction);
    }
    const auto output = values.find(program.output_id);
    if (output == values.end()) {
        if (error) *error = "FusedProgram output is unavailable";
        return nullptr;
    }
    prepared->output_ = output->second;
    return prepared;
}

AbstractTensor* PreparedCalculatorProgram::feed(uint64_t value_id) {
    for (const auto& [candidate, tensor] : feeds_) {
        if (candidate == value_id) return tensor;
    }
    return nullptr;
}

const AbstractTensor* PreparedCalculatorProgram::feed(
    uint64_t value_id) const {
    for (const auto& [candidate, tensor] : feeds_) {
        if (candidate == value_id) return tensor;
    }
    return nullptr;
}

bool PreparedCalculatorProgram::execute() {
    return InMemoryCalculator::instance().execute(instructions_);
}

} // namespace nodus::tensors
