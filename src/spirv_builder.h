// spirv_builder.h
// Maximal SPIR-V builder tuned for KernelIR
#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdint>
#include "kernel_isa.h"

namespace nodus {
namespace spirv {

class SpirvBuilder {
public:
    SpirvBuilder();
    // Module construction
    void beginModule();
    void endModule();
    // Type and constant registration
    uint32_t declareType(const std::string& typeDesc);
    uint32_t declareConstant(uint32_t typeId, double value);
    // Variable and function registration
    uint32_t declareVariable(uint32_t typeId, const std::string& storageClass);
    uint32_t beginFunction(uint32_t returnType, const std::vector<uint32_t>& paramTypes);
    void endFunction();
    // Instruction emission
    uint32_t emitBinaryOp(const std::string& op, uint32_t typeId, uint32_t lhs, uint32_t rhs);
    uint32_t emitUnaryOp(const std::string& op, uint32_t typeId, uint32_t operand);
    uint32_t emitLoad(uint32_t ptrId);
    void emitStore(uint32_t ptrId, uint32_t valueId);
    uint32_t emitSelect(uint32_t condId, uint32_t trueId, uint32_t falseId);
    uint32_t emitCmp(const std::string& pred, uint32_t typeId, uint32_t lhs, uint32_t rhs);
    uint32_t emitCast(const std::string& kind, uint32_t srcId, uint32_t dstTypeId);
    // Control flow
    void emitIf(uint32_t condId, uint32_t thenLabel, uint32_t elseLabel);
    void emitLabel(uint32_t labelId);
    void emitBranch(uint32_t labelId);
    void emitReturn(uint32_t valueId = 0);
    // Barrier/atomic
    void emitBarrier(const std::string& kind, const std::string& scope, const std::string& semantics);
    void emitAtomic(const std::string& kind, uint32_t ptrId, uint32_t valueId);
    // Composite ops
    uint32_t emitExtract(uint32_t compositeId, uint32_t idx);
    uint32_t emitInsert(uint32_t compositeId, uint32_t idx, uint32_t valueId);
    uint32_t emitShuffle(uint32_t vecAId, uint32_t vecBId, const std::vector<uint32_t>& mask);
    // Module output
    std::vector<uint32_t> getBinary() const;
    std::string disassemble() const;
    // ID management
    uint32_t nextId();
private:
    std::vector<uint32_t> words_;
    uint32_t idCounter_;
    std::unordered_map<std::string, uint32_t> typeMap_;
    std::unordered_map<double, uint32_t> constMap_;
    // ...other maps for variables, functions, etc.
};

} // namespace spirv
} // namespace nodus
