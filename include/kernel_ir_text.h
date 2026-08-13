// kernel_ir_text.h
// KIRTEXT v1: the line-oriented KernelIR interchange emitted by turing's
// src/compiler/kernel_ir_lowering.py (serialize_kernel_ir). Deliberately a
// tiny owned format — whitespace-separated records, no escaping, no external
// parser dependency — so the repo-SSA -> KernelIR membrane crosses the
// Python/C++ boundary with both sides fully inspectable.
//
// Records:
//   kirtext 1
//   kernel <name>
//   local_size <x> <y> <z>
//   element_count <n>
//   value <I32|U32|F32> <buffer|plain> <readonly|writable> [debug_name]
//   buffers <id> <id> ...
//   instr <OPCODE> <sub_op> in <ref:id|i:v|u:v|f:v>... out <id>...
//
// Value ids are implicit list positions, matching KernelIR::values.

#pragma once

#include <string>

namespace nodus {
namespace spirv {
struct KernelIR;
}

namespace kernels {

// Parse one KIRTEXT document into a KernelIR. Malformed input throws
// std::runtime_error naming the offending line — fail closed, never guess.
spirv::KernelIR parse_kernel_ir_text(const std::string& text);

} // namespace kernels
} // namespace nodus
