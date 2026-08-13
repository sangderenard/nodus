// vocabulary_actualizer.h
//
// The vocabulary actualizer closes the gap between the canonical operator
// catalog (ops/canonical_ops.json -> include/canonical_ops.h, the shared
// source of truth for turing and nodus) and the module-library reduction
// pipeline (module_library_actualizer.cpp -> gp_plugin_build_module_and_load).
//
// Before this component, the pieces existed but were never wired together:
//   - the catalog knew every operation,
//   - register_abstract_tensor_tool_ir() could mint in-process functor tools
//     (but only in binaries that happened to link its object file),
//   - the module actualizer could inline a plugin's source into a reduced
//     table->tool composite (but only for tools with a registered
//     source_path -- which no vocabulary operation had).
//
// This component walks the catalog and emits one real, standalone,
// composable plugin source per (operation, backend group), following the
// NODUS_PLUGIN_* export convention (docs/BUILD_YOUR_OWN_PLUGIN.md) so each
// source both builds standalone into a tool DLL and inlines into generated
// composite tools. Every emitted source implements the same typed-stack
// contract as the vocabulary functors in abstract_tensor_graph_ir.cpp:
// operands are AbstractTensor* values (VT_VOID_PTR) popped right-then-left,
// restored on failure, with one output tensor pointer pushed on success.
//
// Backend groups differ only in the compute engine inside that contract:
//   - "inmemory": dispatches through tensor_elementwise_unary/binary
//     (tensor_math's plan executor) -- the semantic anchor.
//   - "eigen": maps the in-memory payload via InMemoryBackend::map() into
//     Eigen arrays and evaluates an op-specific Eigen expression.
//     float32, in-memory-backed tensors only for now; anything else is
//     restored untouched (mirrors tensor_math.h's deliberate
//     in-memory-first policy).
// Groups whose engines are not implemented yet (torch, onnx, ...) are not
// silently stubbed: requesting one yields per-op shortfall lines instead of
// generated sources.
#pragma once

#include <cstdint>

// Bitmask of backend groups to actualize.
enum GP_VocabBackendGroup : uint32_t {
    GP_VOCAB_GROUP_INMEMORY = 1u << 0,
    GP_VOCAB_GROUP_EIGEN    = 1u << 1,
};

struct GP_VocabActualizeReport {
    int32_t ops_considered = 0;   // catalog entries examined
    int32_t ops_eligible = 0;     // passed the membrane gate (ct_value present, arity 1..2)
    int32_t sources_written = 0;  // per-(op,group) source files emitted
    int32_t shortfalls = 0;       // (op,group) pairs skipped with a reported reason
    int32_t manifest_entries = 0; // tool entries appended to the manifest file
};

// Walk the canonical catalog and emit per-op plugin sources for every group
// set in `group_mask`, under:
//   <output_root>/source/tools/vocab/<group>/abstract_tensor_<name>.cpp
// and write a manifest fragment listing every generated tool
// (id, group, source path) to:
//   <output_root>/vocab_manifest.txt
//
// Tool ids follow the vocabulary convention "abstract_tensor.<name>" for the
// inmemory group and "<group>.abstract_tensor.<name>" for every other group,
// so group membership is legible in the id itself and the unqualified id
// keeps meaning what it always meant.
//
// Returns 1 on success (even with shortfalls; they are reported in `report`
// and on stderr), 0 on I/O failure. `report` may be null.
#ifdef __cplusplus
extern "C" {
#endif
int gp_vocabulary_actualize_tools(const char* output_root,
                                  uint32_t group_mask,
                                  GP_VocabActualizeReport* report);
#ifdef __cplusplus
}
#endif
