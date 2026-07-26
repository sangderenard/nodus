#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "tensor_calculator.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using nodus::tensors::AbstractTensor;
using nodus::tensors::InMemoryBackend;
using nodus::tensors::TensorDType;
using nodus::tensors::TensorDesc;

struct Binding {
    InMemoryBackend* backend = nullptr;
    AbstractTensor* tensor = nullptr;
    tc_tensor_handle handle = 0;
    void* data = nullptr;

    Binding() = default;
    Binding(const Binding&) = delete;
    Binding& operator=(const Binding&) = delete;

    Binding(Binding&& other) noexcept
        : backend(other.backend),
          tensor(other.tensor),
          handle(other.handle),
          data(other.data) {
        other.backend = nullptr;
        other.tensor = nullptr;
        other.handle = 0;
        other.data = nullptr;
    }
};

uint64_t parse_u64(
    const char* value, const char* option, bool allow_zero = false) {
    errno = 0;
    char* end = nullptr;
    const auto parsed = std::strtoull(value, &end, 10);
    if (!value[0] || value[0] == '-' || (end && *end) || errno == ERANGE ||
        (!allow_zero && parsed == 0)) {
        std::cerr << option << " requires "
                  << (allow_zero ? "a nonnegative" : "a positive")
                  << " integer\n";
        std::exit(2);
    }
    return static_cast<uint64_t>(parsed);
}

bool bind_tensor(
    tc_calculator* calculator,
    InMemoryBackend& backend,
    AbstractTensor& tensor,
    Binding* output) {
    if (!output) return false;
    void* data = nullptr;
    size_t bytes = 0;
    if (!backend.map(tensor.handle(), &data, &bytes)) return false;
    tc_tensor_handle handle = 0;
    const tc_tensor_desc desc{TC_F64, tensor.desc().shape.element_count()};
    if (tc_tensor_bind_external(calculator, desc, data, bytes, &handle) != TC_OK) {
        backend.unmap(tensor.handle());
        return false;
    }
    output->backend = &backend;
    output->tensor = &tensor;
    output->handle = handle;
    output->data = data;
    return true;
}

void release_binding(tc_calculator* calculator, Binding& binding) {
    if (binding.handle) tc_tensor_release(calculator, binding.handle);
    if (binding.backend && binding.tensor)
        binding.backend->unmap(binding.tensor->handle());
    binding.backend = nullptr;
    binding.tensor = nullptr;
    binding.handle = 0;
    binding.data = nullptr;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = std::min(lower + 1, values.size() - 1);
    const double alpha = position - static_cast<double>(lower);
    return values[lower] * (1.0 - alpha) + values[upper] * alpha;
}

} // namespace

int main(int argc, char** argv) {
    uint64_t elements = 262144;
    uint64_t warmup = 10;
    uint64_t repeats = 50;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--elements" && i + 1 < argc)
            elements = parse_u64(argv[++i], "--elements");
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = parse_u64(argv[++i], "--warmup", true);
        else if (arg == "--repeats" && i + 1 < argc)
            repeats = parse_u64(argv[++i], "--repeats");
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--elements N] [--warmup N] [--repeats N]\n";
            return 2;
        }
    }
    if (elements > UINT32_MAX) {
        std::cerr << "--elements exceeds Nodus TensorShape's uint32 dimension\n";
        return 2;
    }

    const auto setup_started = std::chrono::steady_clock::now();
    auto& backend = nodus::tensors::in_memory_backend_singleton();
    const TensorDesc desc{TensorDType::F64, {{static_cast<uint32_t>(elements)}}};
    AbstractTensor input(desc, &backend);
    AbstractTensor negated(desc, &backend);
    AbstractTensor exponent(desc, &backend);
    AbstractTensor denominator(desc, &backend);
    AbstractTensor output(desc, &backend);
    if (!input.valid() || !negated.valid() || !exponent.valid() ||
        !denominator.valid() || !output.valid()) {
        std::cerr << "failed to allocate Nodus tensors\n";
        return 1;
    }

    tc_config config{};
    tc_calculator* calculator = tc_create(&config);
    if (!calculator) {
        std::cerr << "failed to create calculator\n";
        return 1;
    }

    std::vector<Binding> bindings(5);
    AbstractTensor* tensors[] = {
        &input, &negated, &exponent, &denominator, &output};
    for (size_t i = 0; i < bindings.size(); ++i) {
        if (!bind_tensor(calculator, backend, *tensors[i], &bindings[i])) {
            std::cerr << "failed to bind Nodus tensor " << i << "\n";
            for (auto& binding : bindings) release_binding(calculator, binding);
            tc_destroy(calculator);
            return 1;
        }
    }

    auto* input_values = static_cast<double*>(bindings[0].data);
    for (uint64_t i = 0; i < elements; ++i) {
        input_values[i] =
            static_cast<double>(static_cast<int64_t>(i % 1024) - 512) / 128.0;
    }

    const tc_instruction instructions[] = {
        {TC_NEG, bindings[1].handle, bindings[0].handle,
         TC_OPERAND_NONE, 0, 0.0, 0},
        {TC_EXP, bindings[2].handle, bindings[1].handle,
         TC_OPERAND_NONE, 0, 0.0, 0},
        {TC_ADD, bindings[3].handle, bindings[2].handle,
         TC_OPERAND_SCALAR, 0, 1.0, 0},
        {TC_DIV, bindings[4].handle, bindings[3].handle,
         TC_OPERAND_SCALAR, 0, 1.0, 1},
    };
    const tc_program program{
        instructions, static_cast<uint32_t>(std::size(instructions))};
    tc_prepared_program* prepared = nullptr;
    if (tc_prepare(calculator, &program, &prepared) != TC_OK) {
        std::cerr << "failed to prepare calculator program\n";
        for (auto& binding : bindings) release_binding(calculator, binding);
        tc_destroy(calculator);
        return 1;
    }
    const double setup_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_started).count();

    for (uint64_t i = 0; i < warmup; ++i) {
        if (tc_prepared_execute(prepared) != TC_OK) {
            std::cerr << "warmup execution failed\n";
            tc_prepared_release(prepared);
            for (auto& binding : bindings) release_binding(calculator, binding);
            tc_destroy(calculator);
            return 1;
        }
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(repeats));
    for (uint64_t i = 0; i < repeats; ++i) {
        const auto started = std::chrono::steady_clock::now();
        if (tc_prepared_execute(prepared) != TC_OK) {
            std::cerr << "timed execution failed\n";
            tc_prepared_release(prepared);
            for (auto& binding : bindings) release_binding(calculator, binding);
            tc_destroy(calculator);
            return 1;
        }
        const auto stopped = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration<double>(stopped - started).count());
    }

    const auto* result = static_cast<const double*>(bindings[4].data);
    double checksum = 0.0;
    double max_abs_error = 0.0;
    for (uint64_t i = 0; i < elements; ++i) {
        const double expected = 1.0 / (1.0 + std::exp(-input_values[i]));
        checksum += result[i];
        if (std::isfinite(result[i]))
            max_abs_error =
                std::max(max_abs_error, std::abs(result[i] - expected));
        else
            max_abs_error = std::numeric_limits<double>::infinity();
    }

    std::cout << std::setprecision(17)
              << "{\"backend\":\"nodus_calculator\","
              << "\"device\":\"cpu\","
              << "\"execution\":\"precomposed\","
              << "\"dtype\":\"float64\","
              << "\"elements\":" << elements << ","
              << "\"warmup\":" << warmup << ","
              << "\"repeats\":" << repeats << ","
              << "\"setup_sec\":" << setup_sec << ","
              << "\"median_sec\":" << percentile(timings, 0.5) << ","
              << "\"min_sec\":" << percentile(timings, 0.0) << ","
              << "\"max_sec\":" << percentile(timings, 1.0) << ","
              << "\"checksum\":" << checksum << ","
              << "\"first\":" << result[0] << ","
              << "\"middle\":" << result[elements / 2] << ","
              << "\"last\":" << result[elements - 1] << ","
              << "\"max_abs_error\":" << max_abs_error << "}\n";

    tc_prepared_release(prepared);
    for (auto& binding : bindings) release_binding(calculator, binding);
    tc_destroy(calculator);
    return std::isfinite(checksum) && max_abs_error <= 2e-12 ? 0 : 1;
}
