#pragma once

#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"

#include <string>
#include <string_view>

namespace nodus::tensors::kpath {

// RelGeo "operator table" for GraphIR.
// This is intentionally small; it demonstrates how a domain operator set can be
// embedded as a table that emits graph edits (graph operation sequences).
//
// In other words: these operators do NOT mutate a sidecar RelProgram. The graph
// is the source of truth; evaluation/compilation can be performed by interpreting
// the graph.
//
// Operators:
// - pt(x: f64, y: f64) -> u32 (RelPointId as raw)
// - lerp(a: u32, b: u32, u: f64) -> u32
// - offset(base: u32, dx: f64, dy: f64) -> u32
// - ccint(c0: u32, r0: f64, c1: u32, r1: f64, pick: string) -> u32
// - contour(p0: u32, p1: u32, ..., ["open"|"closed"]) -> void
GraphIrOperatorSet make_relgeo_ir_ops();

// Convenience helper: parse+eval into a RelProgram.
bool relgeo_program_from_ir(std::string_view src, RelProgram& out_program, std::string* out_error = nullptr);

} // namespace nodus::tensors::kpath
