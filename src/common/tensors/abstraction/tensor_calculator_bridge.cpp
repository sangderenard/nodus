#include "common/tensors/abstraction/tensor_calculator_bridge.h"

#include "common/tensors/abstraction/in_memory_backend.h"

#include <cstdlib>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nodus::tensors {
namespace {

uint32_t env_u32(const char* name, uint32_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const auto result = std::strtoul(value, &end, 10);
    return end == value ? fallback : static_cast<uint32_t>(result);
}

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* value = std::getenv(name);
    if (!value || !*value) return fallback;
    char* end = nullptr;
    const auto result = std::strtoull(value, &end, 10);
    return end == value ? fallback : static_cast<uint64_t>(result);
}

bool calculator_desc(const TensorDesc& source, tc_tensor_desc* output) {
    if (!output || source.layout != TensorLayout::Dense) return false;
    if (source.dtype == TensorDType::F32) output->dtype = TC_F32;
    else if (source.dtype == TensorDType::F64) output->dtype = TC_F64;
    else return false;
    output->element_count = source.shape.element_count();
    return true;
}

struct BoundTensor {
    InMemoryBackend* backend = nullptr;
    AbstractTensorHandle nodus_handle{};
    tc_tensor_handle calculator_handle = 0;
};

struct BoundProgram {
    tc_calculator* calculator = nullptr;
    std::vector<BoundTensor> bindings;
    std::vector<tc_instruction> instructions;
    tc_job_handle job = 0;

    ~BoundProgram() {
        if (calculator && job) {
            tc_job_wait(calculator, job);
            tc_job_release(calculator, job);
        }
        for (auto& binding : bindings) {
            if (calculator && binding.calculator_handle)
                tc_tensor_release(calculator, binding.calculator_handle);
            if (binding.backend)
                binding.backend->unmap(binding.nodus_handle);
        }
    }
};

std::unique_ptr<BoundProgram> bind_program(
    tc_calculator* calculator,
    std::span<const CalculatorInstruction> source) {
    if (!calculator || source.empty()) return {};
    auto program = std::make_unique<BoundProgram>();
    program->calculator = calculator;
    std::unordered_map<uint64_t, tc_tensor_handle> handles;

    auto bind = [&](const AbstractTensor* tensor) -> tc_tensor_handle {
        if (!tensor || !tensor->valid()) return 0;
        const uint64_t id = tensor->handle().id;
        if (const auto found = handles.find(id); found != handles.end())
            return found->second;
        auto* backend = dynamic_cast<InMemoryBackend*>(tensor->backend());
        if (!backend) return 0;
        tc_tensor_desc desc{};
        if (!calculator_desc(tensor->desc(), &desc)) return 0;
        void* data = nullptr;
        size_t bytes = 0;
        if (!backend->map(tensor->handle(), &data, &bytes)) return 0;
        tc_tensor_handle calculator_handle = 0;
        const tc_status status = tc_tensor_bind_external(
            calculator, desc, data, bytes, &calculator_handle);
        if (status != TC_OK) {
            backend->unmap(tensor->handle());
            return 0;
        }
        handles.emplace(id, calculator_handle);
        program->bindings.push_back(
            {backend, tensor->handle(), calculator_handle});
        return calculator_handle;
    };

    program->instructions.reserve(source.size());
    for (const auto& item : source) {
        tc_instruction native{};
        native.op = item.op;
        native.output = bind(item.output);
        native.left = bind(item.left);
        if (!native.output || !native.left) return {};
        if (item.right) {
            native.right_kind = TC_OPERAND_TENSOR;
            native.right = bind(item.right);
            if (!native.right) return {};
        } else if (item.has_scalar) {
            native.right_kind = TC_OPERAND_SCALAR;
            native.right_scalar = item.right_scalar;
        } else {
            native.right_kind = TC_OPERAND_NONE;
        }
        native.reverse = item.reverse ? 1 : 0;
        program->instructions.push_back(native);
    }
    return program;
}

} // namespace

struct CalculatorJob::Impl {
    std::unique_ptr<BoundProgram> program;
};

CalculatorJob::CalculatorJob(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CalculatorJob::CalculatorJob(CalculatorJob&&) noexcept = default;
CalculatorJob& CalculatorJob::operator=(CalculatorJob&&) noexcept = default;
CalculatorJob::~CalculatorJob() = default;

bool CalculatorJob::valid() const {
    return impl_ && impl_->program && impl_->program->job;
}

bool CalculatorJob::ready() const {
    if (!valid()) return false;
    return tc_job_poll(
        impl_->program->calculator, impl_->program->job) != TC_PENDING;
}

bool CalculatorJob::wait() {
    if (!valid()) return false;
    return tc_job_wait(
        impl_->program->calculator, impl_->program->job) == TC_OK;
}

InMemoryCalculator& InMemoryCalculator::instance() {
    static InMemoryCalculator calculator;
    return calculator;
}

InMemoryCalculator::InMemoryCalculator() {
    tc_config config{};
    config.worker_count = env_u32("NODUS_CALCULATOR_WORKERS", 0);
    config.queue_capacity = env_u32("NODUS_CALCULATOR_QUEUE_CAPACITY", 1024);
    config.async_element_threshold =
        env_u64("NODUS_CALCULATOR_ASYNC_THRESHOLD", 4096);
    calculator_ = tc_create(&config);
}

InMemoryCalculator::~InMemoryCalculator() {
    tc_destroy(calculator_);
}

bool InMemoryCalculator::available() const { return calculator_ != nullptr; }

bool InMemoryCalculator::execute(
    std::span<const CalculatorInstruction> instructions) {
    auto bound = bind_program(calculator_, instructions);
    if (!bound) return false;
    const tc_program program{
        bound->instructions.data(),
        static_cast<uint32_t>(bound->instructions.size())};
    return tc_execute(calculator_, &program) == TC_OK;
}

CalculatorJob InMemoryCalculator::submit(
    std::span<const CalculatorInstruction> instructions) {
    auto bound = bind_program(calculator_, instructions);
    if (!bound) return {};
    const tc_program program{
        bound->instructions.data(),
        static_cast<uint32_t>(bound->instructions.size())};
    if (tc_submit(calculator_, &program, &bound->job) != TC_OK) return {};
    auto impl = std::make_unique<CalculatorJob::Impl>();
    impl->program = std::move(bound);
    return CalculatorJob(std::move(impl));
}

tc_stats InMemoryCalculator::stats() const {
    tc_stats result{};
    if (calculator_) tc_get_stats(calculator_, &result);
    return result;
}

} // namespace nodus::tensors
