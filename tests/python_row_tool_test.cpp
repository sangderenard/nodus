#include "python_row_tool.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using nodus::python_row::Descriptor;
using nodus::python_row::ExecutionMode;
using nodus::python_row::Interpreter;
using nodus::python_row::Port;
using nodus::python_row::PrimitiveType;

namespace {

Descriptor descriptor_for(const fs::path& source, const char* callable) {
    Descriptor descriptor;
    descriptor.tool_id = std::string("python_test_") + callable;
    descriptor.tool_name = std::string("Python test ") + callable;
    descriptor.source_path = source.generic_string();
    descriptor.module_name = std::string("nodus_python_row_test_") + callable;
    descriptor.callable_name = callable;
    descriptor.inputs = {Port{"left", PrimitiveType::Float64}, Port{"right", PrimitiveType::Int32}};
    descriptor.inputs[0].label = "Left operand";
    descriptor.inputs[0].form_control = "number";
    descriptor.inputs[1].label = "Whole-number offset";
    descriptor.inputs[1].form_control = "integer";
    descriptor.inputs[1].form_required = false;
    descriptor.outputs = {Port{"difference", PrimitiveType::Float64}, Port{"sum", PrimitiveType::Int64}};
    return descriptor;
}

template <typename T>
void push(RawStackFrame& stack, ValueTypeBuiltin builtin, const T& value) {
    const ValueTypeId type = ValueTypeRegistry::global().builtin(builtin);
    assert(raw_stack_push_typed(stack, &value, type));
}

template <typename T>
T pop(RawStackFrame& stack, ValueTypeBuiltin builtin) {
    T value{};
    const ValueTypeId type = ValueTypeRegistry::global().builtin(builtin);
    assert(raw_stack_pop_typed(stack, &value, type));
    return value;
}

void write_python_source(const fs::path& path) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    assert(stream.good());
    stream <<
        "def ordered(left, right):\n"
        "    return left - right, int(left + right)\n\n"
        "def raises(left, right):\n"
        "    raise RuntimeError('deliberate python row failure')\n\n"
        "def wrong_arity(left, right):\n"
        "    return left - right\n\n"
        "def wrong_type(left, right):\n"
        "    return 'not-a-number', int(left + right)\n";
    assert(stream.good());
}

void release_stack(RawStackFrame& stack) {
    raw_stack_free_mask(stack);
    if (stack.backend) {
        gp_mem_backend_release(stack.backend);
        stack.backend = nullptr;
    }
}

void test_descriptor_and_generation(const fs::path& source_path) {
    std::string error;
    Descriptor descriptor = descriptor_for(source_path, "ordered");
    assert(nodus::python_row::validate_descriptor(descriptor, &error));

    std::string generated;
    assert(nodus::python_row::generate_plugin_source(descriptor, &generated, &error));
    assert(generated.find("NODUS_PLUGIN_FACTORY_NAME") != std::string::npos);
    assert(generated.find("NODUS_PLUGIN_SOURCE_NAME") != std::string::npos);
    assert(generated.find(source_path.generic_string()) != std::string::npos);
    assert(generated.find("PrimitiveType::Int32") != std::string::npos);
    assert(generated.find("PrimitiveType::Int64") != std::string::npos);
    assert(generated.find("Left operand") != std::string::npos);
    const fs::path wrapper_path = source_path.parent_path() / "python_test_ordered.cpp";
    assert(nodus::python_row::write_plugin_source(descriptor, wrapper_path.generic_string(), &error));
    assert(fs::exists(wrapper_path));

    const fs::path abi_path = source_path.parent_path() / "ordered.pyabi";
    assert(nodus::python_row::write_descriptor(descriptor, abi_path.generic_string(), &error));
    Descriptor restored;
    assert(nodus::python_row::read_descriptor(abi_path.generic_string(), &restored, &error));
    assert(restored.abi_version == descriptor.abi_version);
    assert(restored.tool_id == descriptor.tool_id);
    assert(restored.source_path == descriptor.source_path);
    assert(restored.inputs.size() == 2 && restored.inputs[1].type == PrimitiveType::Int32);
    assert(restored.inputs[0].label == "Left operand");
    assert(restored.inputs[0].form_control == "number");
    assert(restored.inputs[0].form_required);
    assert(restored.inputs[1].label == "Whole-number offset");
    assert(restored.inputs[1].form_control == "integer");
    assert(!restored.inputs[1].form_required);
    assert(restored.outputs.size() == 2 && restored.outputs[1].type == PrimitiveType::Int64);

    descriptor.mode = ExecutionMode::CompiledSsa;
    assert(!nodus::python_row::validate_descriptor(descriptor, &error));
    assert(error.find("no compiled-SSA fallback") != std::string::npos);
    descriptor.mode = ExecutionMode::Interpreted;
    descriptor.inputs[0].type = static_cast<PrimitiveType>(999);
    assert(!nodus::python_row::validate_descriptor(descriptor, &error));
    assert(error.find("pointer, tensor, dynamic, or unknown") != std::string::npos);
}

void test_order_and_multiple_returns(const fs::path& source_path) {
    RawStackFrame stack{};
    assert(raw_stack_init_frame(stack, 256));
    const double left = 9.5;
    const int32_t right = 4;
    push(stack, VT_FLOAT64, left);
    push(stack, VT_INT32, right);

    Interpreter interpreter(descriptor_for(source_path, "ordered"));
    std::string error;
    assert(interpreter.execute(stack, &error));
    const int64_t sum = pop<int64_t>(stack, VT_INT64);
    const double difference = pop<double>(stack, VT_FLOAT64);
    assert(sum == 13);
    assert(std::abs(difference - 5.5) < 1e-12);
    assert(stack.byte_count == 0);
    release_stack(stack);
}

void test_stack_type_refusal_is_non_mutating(const fs::path& source_path) {
    RawStackFrame stack{};
    assert(raw_stack_init_frame(stack, 256));
    const int32_t wrong_left = 9;
    const int32_t right = 4;
    push(stack, VT_INT32, wrong_left);
    push(stack, VT_INT32, right);
    const size_t before = stack.byte_count;

    Interpreter interpreter(descriptor_for(source_path, "ordered"));
    std::string error;
    assert(!interpreter.execute(stack, &error));
    assert(error.find("left") != std::string::npos);
    assert(stack.byte_count == before);
    assert(pop<int32_t>(stack, VT_INT32) == right);
    assert(pop<int32_t>(stack, VT_INT32) == wrong_left);
    release_stack(stack);
}

void test_python_failure_restores_inputs(const fs::path& source_path, const char* callable,
                                         const char* expected_error) {
    RawStackFrame stack{};
    assert(raw_stack_init_frame(stack, 256));
    const double left = 8.0;
    const int32_t right = 3;
    push(stack, VT_FLOAT64, left);
    push(stack, VT_INT32, right);
    const size_t before = stack.byte_count;

    Interpreter interpreter(descriptor_for(source_path, callable));
    std::string error;
    assert(!interpreter.execute(stack, &error));
    assert(error.find(expected_error) != std::string::npos);
    assert(stack.byte_count == before);
    assert(pop<int32_t>(stack, VT_INT32) == right);
    assert(pop<double>(stack, VT_FLOAT64) == left);
    release_stack(stack);
}

} // namespace

int main() {
    const fs::path root = fs::temp_directory_path() / "nodus_python_row_tool_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
    const fs::path source = root / "row_functions.py";
    write_python_source(source);

    test_descriptor_and_generation(source);
    test_order_and_multiple_returns(source);
    test_stack_type_refusal_is_non_mutating(source);
    test_python_failure_restores_inputs(source, "raises", "deliberate python row failure");
    test_python_failure_restores_inputs(source, "wrong_arity", "tuple arity");
    test_python_failure_restores_inputs(source, "wrong_type", "not numeric");

    if (!std::getenv("NODUS_PYTHON_ROW_KEEP_TEMP")) fs::remove_all(root, ec);
    std::cout << "python_row_tool_test: OK\n";
    return 0;
}
