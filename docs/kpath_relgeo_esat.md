# RelGeo e-saturation (e-graph) integration

This document describes how RelGeo evolves from “evaluate-with-anchors” into a symbolic relational geometry engine driven by equality saturation.

## Why e-saturation here

RelGeo already has:

- **Stable noun identities** (`RelPointId`, `RelLineId`, `RelCircleId`, …)
- **Definitions** (points, lines, circles, arcs, beziers)
- **Logical assertions** (incidence/coincidence/parallel/perp, point-on-circle, fixed-radius, …)

The missing piece is a way to **derive consequences** (equal lengths, equal angles, ratios, canonical forms) *without* choosing coordinates.

E-saturation provides:

- A compact representation of “many equivalent expressions” (an **e-graph**)
- A systematic way to apply rewrite rules to saturation
- Proof-by-derivation: facts are justified by rewrites, not numeric projection

## Non-goals (for now)

- Not a numeric constraint solver.
- Not a full synthetic-geometry theorem prover.
- Not trying to decide solvability/uniqueness.

## Term language (what lives in the e-graph)

Start small with a first-order, mostly-equational term language:

- `Point(p)`
- `Dist(p,q)` (a length term; symmetric)
- `Radius(circle)`
- `ConstLength(c)` (exact/quantized literal)

Then expand:

- `Dir(line)`
- `Angle(a,v,b)` and/or `AngleBetween(line0,line1)`
- `Midpoint(a,b)`
- `Lerp(a,b,u)` where `u` is a rational term
- `Ratio(len0,len1)`

## Facts vs equalities

We separate:

- **Equalities**: `t0 == t1` (union in the e-graph)
- **Predicates**: `OnLine(p,l)`, `OnCircle(p,c)`, `Parallel(l0,l1)`, `Perp(l0,l1)`, `Tangent(l,c)`

Predicates generate equalities via rewrite rules (when sound), but we do *not* collapse everything into equality.

Example: `PointOnCircle(p,C)` implies the equational fact:

- `Dist(center(C), p) == Radius(C)`

while `Tangent(line,C)` is existential without an explicit tangency point, so it stays as a predicate until we have more structure.

## Integration points in code

- IR emits a graph edit log.
- `relgeo_program_from_ir()` converts edits into a `RelProgram` holding noun defs + logical assertions.
- **New:** an e-saturation pass consumes `RelProgram` and produces derived equalities/facts.

The key constraint: **IDs must never be treated as numeric literals**.

## Immediate milestones

1. **Compile-only scaffold**: add an e-sat module with a minimal term set and a runner.
2. **First derived fact** (smoke test):
   - From `point_on_circle(p,C)` and `fixed_radius(C, r)` derive `Dist(center(C), p) == r`.
3. Add term expansions and safe rewrites:
   - symmetry/canonicalization of `Dist`
   - propagating equalities through constructors (congruence)
4. Add a “proof trace” option (record which rule created each union).

## Long-term (ambitious) targets

- Canonicalization for angles and directed lines
- Ratios and proportionality reasoning
- Similarity/congruence pattern rules (SSS/SAS/ASA) as rewrite schemas
- Query API: “prove or produce counterexample” (counterexample likely requires optional numeric embedding)

---

Status: this doc is the contract for the first e-sat scaffold + smoke test.
