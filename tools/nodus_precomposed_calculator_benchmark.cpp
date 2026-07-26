#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

using nodus::tensors::AbstractTensor;
using nodus::tensors::FusedProgramTransport;
using nodus::tensors::InMemoryCalculator;
using nodus::tensors::PreparedCalculatorProgram;
using nodus::tensors::TensorDType;
using nodus::tensors::TensorDesc;

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
    std::string program_path;
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        if (arg == "--elements" && i + 1 < argc)
            elements = parse_u64(argv[++i], "--elements");
        else if (arg == "--warmup" && i + 1 < argc)
            warmup = parse_u64(argv[++i], "--warmup", true);
        else if (arg == "--repeats" && i + 1 < argc)
            repeats = parse_u64(argv[++i], "--repeats");
        else if (arg == "--program" && i + 1 < argc)
            program_path = argv[++i];
        else if (arg == "--output-bin" && i + 1 < argc)
            output_path = argv[++i];
        else {
            std::cerr << "usage: " << argv[0]
                      << " --program FILE [--output-bin FILE]"
                      << " [--elements N] [--warmup N] [--repeats N]\n";
            return 2;
        }
    }
    if (program_path.empty()) {
        std::cerr << "--program is required; the benchmark owns no workload\n";
        return 2;
    }
    if (elements > UINT32_MAX) {
        std::cerr << "--elements exceeds Nodus TensorShape's uint32 dimension\n";
        return 2;
    }

    FusedProgramTransport program;
    std::string parse_error;
    std::ifstream program_stream(program_path);
    if (!program_stream ||
        !nodus::tensors::parse_fused_program_transport(
            program_stream, &program, &parse_error)) {
        if (!program_stream)
            parse_error = "could not open FusedProgram transport: " + program_path;
        std::cerr << parse_error << "\n";
        return 2;
    }

    const auto setup_started = std::chrono::steady_clock::now();
    auto& backend = nodus::tensors::in_memory_backend_singleton();
    const TensorDesc desc{TensorDType::F64, {{static_cast<uint32_t>(elements)}}};
    auto prepared = PreparedCalculatorProgram::create(
        program, desc, &backend, &parse_error);
    if (!prepared) {
        std::cerr << parse_error << "\n";
        return 1;
    }
    for (const uint64_t feed_id : program.feed_ids) {
        AbstractTensor* tensor = prepared->feed(feed_id);
        void* raw = nullptr;
        size_t bytes = 0;
        if (!tensor || !backend.map(tensor->handle(), &raw, &bytes)) {
            std::cerr << "failed to map Nodus feed tensor\n";
            return 1;
        }
        auto* input = static_cast<double*>(raw);
        for (uint64_t i = 0; i < elements; ++i) {
            input[i] =
                static_cast<double>(static_cast<int64_t>(i % 1024) - 512) /
                128.0;
        }
        backend.unmap(tensor->handle());
    }
    AbstractTensor* output = prepared->output();
    const double setup_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - setup_started).count();

    for (uint64_t i = 0; i < warmup; ++i) {
        if (!prepared->execute()) {
            std::cerr << "warmup execution failed\n";
            return 1;
        }
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(repeats));
    for (uint64_t i = 0; i < repeats; ++i) {
        const auto started = std::chrono::steady_clock::now();
        if (!prepared->execute()) {
            std::cerr << "timed execution failed\n";
            return 1;
        }
        const auto stopped = std::chrono::steady_clock::now();
        timings.push_back(
            std::chrono::duration<double>(stopped - started).count());
    }

    void* raw = nullptr;
    size_t bytes = 0;
    if (!backend.map(output->handle(), &raw, &bytes)) {
        std::cerr << "failed to map Nodus output tensor\n";
        return 1;
    }
    const auto* result = static_cast<const double*>(raw);
    double checksum = 0.0;
    for (uint64_t i = 0; i < elements; ++i) checksum += result[i];

    if (!output_path.empty()) {
        std::ofstream binary(output_path, std::ios::binary);
        binary.write(
            reinterpret_cast<const char*>(result),
            static_cast<std::streamsize>(elements * sizeof(double)));
        if (!binary) {
            backend.unmap(output->handle());
            std::cerr << "failed to write Nodus output\n";
            return 1;
        }
    }

    std::cout << std::setprecision(17)
              << "{\"backend\":\"nodus_calculator\","
              << "\"device\":\"cpu\","
              << "\"execution\":\"captured_fused_program\","
              << "\"dtype\":\"float64\","
              << "\"program_steps\":" << prepared->instruction_count() << ","
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
              << "\"last\":" << result[elements - 1] << "}\n";

    backend.unmap(output->handle());
    return std::isfinite(checksum) ? 0 : 1;
}
