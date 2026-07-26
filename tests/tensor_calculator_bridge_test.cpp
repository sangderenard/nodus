#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include <cassert>
#include <cmath>
#include <sstream>

using namespace nodus::tensors;

int main() {
    auto& backend = in_memory_backend_singleton();
    const TensorDesc desc{TensorDType::F32, {{128}}};
    AbstractTensor input(desc, &backend);
    AbstractTensor temporary(desc, &backend);
    AbstractTensor output(desc, &backend);

    void* raw = nullptr;
    size_t bytes = 0;
    const bool input_mapped = backend.map(input.handle(), &raw, &bytes);
    assert(input_mapped);
    if (!input_mapped) return 1;
    auto* values = static_cast<float*>(raw);
    for (uint32_t i = 0; i < 128; ++i) values[i] = static_cast<float>(i) / 32.0f;
    backend.unmap(input.handle());

    const CalculatorInstruction program[] = {
        {nodus::ops::CanonicalOp::MUL,
         &temporary, &input, nullptr, 2.0, true, false},
        {nodus::ops::CanonicalOp::ADD,
         &output, &temporary, nullptr, 1.0, true, false},
    };
    auto& calculator = InMemoryCalculator::instance();
    assert(calculator.available());
    if (!calculator.available()) return 2;
    const bool executed = calculator.execute(program);
    assert(executed);
    if (!executed) return 3;

    const bool output_mapped = backend.map(output.handle(), &raw, &bytes);
    assert(output_mapped);
    if (!output_mapped) return 4;
    values = static_cast<float*>(raw);
    const bool value_matches =
        std::abs(values[17] - (17.0f / 16.0f + 1.0f)) < 1e-6f;
    assert(value_matches);
    backend.unmap(output.handle());
    if (!value_matches) return 5;

    auto job = calculator.submit(program);
    assert(job.valid());
    if (!job.valid()) return 6;
    const bool completed = job.wait();
    assert(completed);
    if (!completed) return 7;

    std::istringstream wire(
        "fused_program 1\n"
        "feed 1\n"
        "step 1 mul 2 1 1 1 2 0\n"
        "step 2 add 3 1 2 1 1 0\n"
        "output 3\n"
        "end\n");
    FusedProgramTransport transported;
    std::string error;
    const bool parsed =
        parse_fused_program_transport(wire, &transported, &error);
    assert(parsed);
    if (!parsed) return 8;
    auto prepared = PreparedCalculatorProgram::create(
        transported, desc, &backend, &error);
    assert(prepared);
    if (!prepared) return 9;
    auto* prepared_input = prepared->feed(1);
    assert(prepared_input);
    if (!prepared_input ||
        !backend.map(prepared_input->handle(), &raw, &bytes))
        return 10;
    values = static_cast<float*>(raw);
    for (uint32_t i = 0; i < 128; ++i)
        values[i] = static_cast<float>(i) / 32.0f;
    backend.unmap(prepared_input->handle());
    const bool prepared_executed = prepared->execute();
    assert(prepared_executed);
    if (!prepared_executed) return 11;
    const bool prepared_output_mapped =
        backend.map(prepared->output()->handle(), &raw, &bytes);
    assert(prepared_output_mapped);
    if (!prepared_output_mapped) return 12;
    values = static_cast<float*>(raw);
    const bool prepared_matches =
        std::abs(values[17] - (17.0f / 16.0f + 1.0f)) < 1e-6f;
    backend.unmap(prepared->output()->handle());
    assert(prepared_matches);
    if (!prepared_matches) return 13;

    std::istringstream truncated(
        "fused_program 1\nfeed 1\noutput 1\n");
    FusedProgramTransport rejected;
    const bool accepted_truncated =
        parse_fused_program_transport(truncated, &rejected, &error);
    assert(!accepted_truncated);
    if (accepted_truncated) return 14;
    return 0;
}
