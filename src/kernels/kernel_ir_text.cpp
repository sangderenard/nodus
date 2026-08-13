// kernel_ir_text.cpp
// KIRTEXT v1 parser. See include/kernel_ir_text.h for the format contract.

#include "../kernel_isa.h"
#include "kernel_ir_text.h"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nodus {
namespace kernels {

namespace {

using spirv::KernelIR;
using spirv::OpCode;
using spirv::Operand;
using spirv::ScalarType;
using spirv::ValueRef;

[[noreturn]] void fail(std::size_t line_number, const std::string& reason) {
    throw std::runtime_error(
        "kirtext line " + std::to_string(line_number) + ": " + reason);
}

const std::unordered_map<std::string, OpCode>& opcode_names() {
    static const std::unordered_map<std::string, OpCode> names = {
        {"MODULE_BEGIN", OpCode::MODULE_BEGIN},
        {"KERNEL_ENTRY", OpCode::KERNEL_ENTRY},
        {"TYPE", OpCode::TYPE}, {"CONST", OpCode::CONST},
        {"SPEC_CONST", OpCode::SPEC_CONST}, {"VAR", OpCode::VAR},
        {"ADDR", OpCode::ADDR}, {"LOAD", OpCode::LOAD},
        {"STORE", OpCode::STORE}, {"MEMCPY", OpCode::MEMCPY},
        {"UNARY", OpCode::UNARY}, {"BINARY", OpCode::BINARY},
        {"TERNARY", OpCode::TERNARY}, {"CMP", OpCode::CMP},
        {"SELECT", OpCode::SELECT}, {"CAST", OpCode::CAST},
        {"EXTRACT", OpCode::EXTRACT}, {"INSERT", OpCode::INSERT},
        {"SHUFFLE", OpCode::SHUFFLE}, {"IF", OpCode::IF},
        {"BARRIER", OpCode::BARRIER}, {"ATOMIC", OpCode::ATOMIC},
        {"AND", OpCode::AND}, {"OR", OpCode::OR},
        {"NOT", OpCode::NOT}, {"XOR", OpCode::XOR},
    };
    return names;
}

} // namespace

spirv::KernelIR parse_kernel_ir_text(const std::string& text) {
    KernelIR kernel;
    std::istringstream stream(text);
    std::string line;
    std::size_t line_number = 0;
    bool saw_header = false;

    while (std::getline(stream, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream fields(line);
        std::string record;
        fields >> record;

        if (!saw_header) {
            std::string version;
            if (record != "kirtext" || !(fields >> version) || version != "1") {
                fail(line_number, "expected 'kirtext 1' header");
            }
            saw_header = true;
            continue;
        }

        if (record == "kernel") {
            if (!(fields >> kernel.name)) fail(line_number, "kernel needs a name");
        } else if (record == "local_size") {
            if (!(fields >> kernel.suggested_local_size[0]
                         >> kernel.suggested_local_size[1]
                         >> kernel.suggested_local_size[2])) {
                fail(line_number, "local_size needs three integers");
            }
        } else if (record == "element_count") {
            if (!(fields >> kernel.element_count)) {
                fail(line_number, "element_count needs an integer");
            }
        } else if (record == "value") {
            std::string scalar, storage, access;
            if (!(fields >> scalar >> storage >> access)) {
                fail(line_number, "value needs scalar, storage, access");
            }
            spirv::ValueDef def;
            if (scalar == "I32") def.type.scalar = ScalarType::I32;
            else if (scalar == "U32") def.type.scalar = ScalarType::U32;
            else if (scalar == "F32") def.type.scalar = ScalarType::F32;
            else fail(line_number, "unknown scalar type " + scalar);
            if (storage == "buffer") def.type.is_buffer = true;
            else if (storage != "plain") {
                fail(line_number, "storage must be buffer or plain");
            }
            if (access == "readonly") def.type.is_readonly = true;
            else if (access != "writable") {
                fail(line_number, "access must be readonly or writable");
            }
            fields >> def.debug_name; // optional
            kernel.values.push_back(std::move(def));
        } else if (record == "buffers") {
            std::uint32_t id = 0;
            while (fields >> id) kernel.buffer_value_ids.push_back(id);
        } else if (record == "instr") {
            std::string opcode_name, marker;
            std::int32_t sub_op = -1;
            if (!(fields >> opcode_name >> sub_op >> marker) || marker != "in") {
                fail(line_number, "instr needs '<OPCODE> <sub_op> in ...'");
            }
            auto found = opcode_names().find(opcode_name);
            if (found == opcode_names().end()) {
                fail(line_number, "unknown opcode " + opcode_name);
            }
            spirv::Instruction ins;
            ins.op = found->second;
            ins.sub_op = sub_op;
            std::string token;
            bool in_outputs = false;
            while (fields >> token) {
                if (token == "out") { in_outputs = true; continue; }
                if (in_outputs) {
                    ins.outputs.push_back(
                        ValueRef{static_cast<std::uint32_t>(std::stoul(token))});
                    continue;
                }
                const auto colon = token.find(':');
                if (colon == std::string::npos) {
                    fail(line_number, "operand token needs kind:value, got " + token);
                }
                const std::string kind = token.substr(0, colon);
                const std::string raw = token.substr(colon + 1);
                if (kind == "ref") {
                    ins.inputs.push_back(Operand::ref(
                        ValueRef{static_cast<std::uint32_t>(std::stoul(raw))}));
                } else if (kind == "i") {
                    ins.inputs.push_back(
                        Operand::i(static_cast<std::int32_t>(std::stol(raw))));
                } else if (kind == "u") {
                    ins.inputs.push_back(
                        Operand::u(static_cast<std::uint32_t>(std::stoul(raw))));
                } else if (kind == "f") {
                    ins.inputs.push_back(Operand::f(std::stof(raw)));
                } else {
                    fail(line_number, "unknown operand kind " + kind);
                }
            }
            kernel.instrs.push_back(std::move(ins));
        } else {
            fail(line_number, "unknown record " + record);
        }
    }
    if (!saw_header) fail(0, "empty document");
    return kernel;
}

} // namespace kernels
} // namespace nodus
