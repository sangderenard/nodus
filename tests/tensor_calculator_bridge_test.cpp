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
    assert(backend.map(input.handle(), &raw, &bytes));
    auto* values = static_cast<float*>(raw);
    for (uint32_t i = 0; i < 128; ++i) values[i] = static_cast<float>(i) / 32.0f;
    backend.unmap(input.handle());

    const CalculatorInstruction program[] = {
        {TC_MUL, &temporary, &input, nullptr, 2.0, true, false},
        {TC_ADD, &output, &temporary, nullptr, 1.0, true, false},
    };
    auto& calculator = InMemoryCalculator::instance();
    assert(calculator.available());
    assert(calculator.execute(program));

    assert(backend.map(output.handle(), &raw, &bytes));
    values = static_cast<float*>(raw);
    assert(std::abs(values[17] - (17.0f / 16.0f + 1.0f)) < 1e-6f);
    backend.unmap(output.handle());

    auto job = calculator.submit(program);
    assert(job.valid());
    assert(job.wait());
    return 0;
}
