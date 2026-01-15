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
// - free() -> u32 point
// - pt(x: f64, y: f64) -> u32 point
// - pi() -> f64
// - add(a: f64, b: f64) -> f64
// - sub(a: f64, b: f64) -> f64
// - mul(a: f64, b: f64) -> f64
// - div(a: f64, b: f64) -> f64
// - lerp(a: u32, b: u32, u: f64) -> u32 point
// - offset(base: u32, dx: f64, dy: f64) -> u32 point
// - ccint(c0: u32, r0: f64, c1: u32, r1: f64, pick: string) -> u32 point
// - line(a: u32, b: u32) -> u32 line
// - ray(origin: u32, through: u32) -> u32 ray
// - segment(a: u32, b: u32) -> u32 segment
// - angle(a: u32, v: u32, b: u32) -> u32 angle
// - llint(l0: u32, l1: u32) -> u32 point
// - param_line(start: u32, end: u32, u: f64[, dir: string|bool]) -> u32
// - param_quad(start: u32, ctrl: u32, end: u32, u: f64[, dir: string|bool]) -> u32
// - param_cubic(start: u32, ctrl0: u32, ctrl1: u32, end: u32, u: f64[, dir: string|bool]) -> u32
// - param_sin(start: u32, end: u32, u: f64, amplitude: f64, cycles: f64, phase: f64[, dir: string|bool]) -> u32
// - circle(center: u32, r: f64|u32) -> u32 circle
//     - if r is f64: constant radius
//     - if r is u32: interpreted as a point id "through" (radius is distance(center, through))
// - arc(circle: u32circle, a0: f64, a1: f64[, dir: string|bool]) -> u32 arc
// - arcse(circle: u32circle, start: u32, end: u32[, dir: string|bool]) -> u32 arc
// - arc3(p0: u32, p1: u32, p2: u32) -> u32 arc
// - bezier(p0: u32, c0: u32, c1: u32, p1: u32) -> u32 bezier
// - contour(p0: u32, p1: u32, ..., ["open"|"closed"], ["ccw"|"cw"|"outer"|"inner"]) -> void
// - circuit(p0: u32, p1: u32, ..., ["open"|"closed"], ["ccw"|"cw"|"outer"|"inner"]) -> void
// - ngon(center: u32, r: f64, sides: u32[, phase: f64][, winding: string]) -> void
// - star(center: u32, r_outer: f64, r_inner: f64, points: u32[, phase: f64][, winding: string]) -> void
// - spiral(center: u32, r: f64, radians: f64[, samples: u32][, phase: f64][, winding: string]) -> void
//     - if r < 0: spiral inward from |r| to 0
// - path_begin() -> u32
// - path_spiral(path: u32, center: u32, r: f64, radians: f64[, samples: u32][, phase: f64][, winding: string]) -> void
// - path_arc(path: u32, center: u32, r: f64, radians: f64[, samples: u32][, phase: f64][, winding: string]) -> void
// - path_end(path: u32[, "open"|"closed"][, "ccw"|"cw"|"outer"|"inner"]) -> void
// - parallel(l0: u32, l1: u32) -> void edge
// - perp(l0: u32, l1: u32) -> void edge
// - perp_at(v: u32, a: u32, b: u32) -> void edge
// - incident(p: u32, l: u32) -> void edge
// - coincident(p0: u32, p1: u32) -> void edge
// - point_on_circle(p: u32, circle: u32circle) -> void edge
// - tangent(line: u32line, circle: u32circle) -> void edge
// - fixed_radius(circle: u32circle, r: f64) -> void edge
// - arc_angle(arc: u32arc, angle: f64) -> void edge
// - collinear(a: u32, b: u32, c: u32) -> void edge
// - equal_dist(a: u32, b: u32, c: u32, d: u32) -> void edge
// - midpoint(m: u32, a: u32, b: u32) -> void edge
// - on_segment(p: u32, a: u32, b: u32) -> void edge
// - on_segment_ratio(p: u32, a: u32, b: u32[, ratio: f64]) -> void edge
// - inscribed(p: u32, a: u32, b: u32[, ratio: f64]) -> void edge
// - outscribed(p: u32, a: u32, b: u32[, ratio: f64]) -> void edge
// - parallel_pairs(a0: u32, a1: u32, b0: u32, b1: u32) -> void edge
// - equal_angle(a0: u32, a1: u32, b0: u32, b1: u32) -> void edge
// - triangle(a: u32, b: u32, c: u32) -> void edge
GraphIrOperatorSet make_relgeo_ir_ops();

// A "pure" subset of RelGeo IR intended to avoid numeric literals for *derived* geometry.
// Numeric literals are allowed for *reference/anchor* values (e.g. pt(x,y)) and are
// expected to be replaced/overridden by an external constraint-solve stage when desired.
//
// Pure-only operators add a small constructible-ratio vocabulary:
// - ratio(n: u32, d: u32) -> u32 ratio
// - lerp_ratio(a: u32, b: u32, u: u32ratio) -> u32 point
//
// The returned operator set intentionally omits float-based operators such as offset()/param_*/ccint().
// It also omits constraints that require floats (fixed_radius/arc_angle) and the angle-based arc() constructor.
GraphIrOperatorSet make_relgeo_ir_ops_pure();

// Validates that a source program only uses the pure operator subset.
bool relgeo_validate_pure_ir(std::string_view src, std::string* out_error = nullptr);

// Parse + evaluate a pure RelGeo IR program and return the emitted edit log.
// This is the intended input for a later constraint-solve stage.
bool relgeo_edits_from_pure_ir(std::string_view src,
							  std::vector<nodus::tensors::GraphEdit>& out_edits,
							  std::string* out_error = nullptr);

// Convenience helper: parse+eval into a RelProgram.
bool relgeo_program_from_ir(std::string_view src, RelProgram& out_program, std::string* out_error = nullptr);

} // namespace nodus::tensors::kpath
