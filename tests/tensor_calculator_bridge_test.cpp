#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include <cassert>
#include <cmath>

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
        {TC_MUL, &temporary, &input, nullptr, 2.0, true, false},
        {TC_ADD, &output, &temporary, nullptr, 1.0, true, false},
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
    return 0;
}
