// canonical_ops_test.cpp
//
// Consumer + invariants for the generated canonical operation table
// (include/canonical_ops.h, generated from ops/canonical_ops.json).
//
// Deliberately links nothing: canonical_ops.h is header-only and dependency-free, and
// kernel_isa.h is structs only. So this test builds and runs even while the
// canvas_tables amalgam is mid-refactor (see NODUS_TENSOR_CORE_EXTRACTION_HANDOFF.md).
//
// Rationale for the table itself: research/15_the_missing_function_table.md.

#include "canonical_ops.h"
#include "../src/kernel_isa.h"

#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <string_view>

using nodus::ops::CanonicalOp;
using nodus::ops::OpClass;
using nodus::ops::OpDesc;
using nodus::ops::find_op;
using nodus::ops::find_op_by_handler;
using nodus::ops::kOpCount;
using nodus::ops::kOps;

static int failures = 0;

static bool check(bool condition, const std::string& what) {
  if (!condition) {
    std::cerr << "[CANONICAL-OPS] FAILED: " << what << "\n";
    ++failures;
  }
  return condition;
}

// ---------------------------------------------------------------------------
// compile-time: the table is usable in constant expressions, so a lowering pass
// can switch on it without runtime cost.
// ---------------------------------------------------------------------------
static_assert(find_op("add") != nullptr, "add must exist");
static_assert(find_op("add")->ct_value == 0, "add must be CTensorOp ordinal 0");
static_assert(find_op("add")->op_class == OpClass::Binary, "add is binary");
static_assert(find_op("no_such_op") == nullptr, "unknown names must yield nullptr");
static_assert(find_op(CanonicalOp::MUL)->name == std::string_view("mul"),
              "reverse lookup by enum must round-trip");
static_assert(static_cast<int>(CanonicalOp::COUNT) == 66,
              "every catalog operation must have a canonical ID");

int main() {
  // 1) Every lowerable op names a KernelIR opcode; every non-lowerable names none.
  //    This is the honesty invariant: an op cannot claim portability it lacks.
  //    Tier discipline (docs/TIERS.md): lowerable means "expressible as ONE
  //    Tier-0 instruction". An opaque op is not -- a reduction genuinely
  //    cannot be one instruction (research/15, finding 6b) -- so it carries no
  //    kernel_op and names its Tier-1 composition family instead. Keeping that
  //    distinction is what stops an emitter from being handed a composite.
  for (size_t i = 0; i < kOpCount; ++i) {
    const OpDesc& o = kOps[i];
    check(o.lowerable == !o.kernel_op.empty(),
          std::string("lowerable/kernel_op disagree for '") + std::string(o.name) + "'");
    if (o.op_class == OpClass::Opaque) {
      check(!o.lowerable,
            std::string("opaque op '") + std::string(o.name) +
                "' must not claim to be one Tier-0 instruction");
      check(!o.tier1_class.empty(),
            std::string("opaque op '") + std::string(o.name) +
                "' must name its Tier-1 composition family");
    } else {
      check(o.tier1_class.empty(),
            std::string("Tier-0 op '") + std::string(o.name) +
                "' needs no Tier-1 family");
    }
  }

  // 2) ct_values form a gapless, unique 0..N-1 block mirroring turing's CTensorOp.
  {
    std::set<int32_t> seen;
    for (size_t i = 0; i < kOpCount; ++i) {
      if (kOps[i].ct_value < 0) continue;
      check(seen.insert(kOps[i].ct_value).second,
            std::string("duplicate ct_value for '") + std::string(kOps[i].name) + "'");
    }
    // 40 = the 28 original members plus the vendored trig/hyperbolic family
    // (sin..atanh). ops/verify_canonical_ops.py checks this count against
    // turing's live CTensorOp header, so this constant tracks that verifier.
    check(static_cast<int>(seen.size()) == 40,
          "the current CTensorOp subset has 40 members");
    int32_t expected = 0;
    for (int32_t v : seen) {
      check(v == expected, "ct_values must be gapless from 0");
      ++expected;
    }
  }

  // 3) An op with a C target must name the CTensorOp member, and vice versa.
  for (size_t i = 0; i < kOpCount; ++i) {
    const OpDesc& o = kOps[i];
    check((o.ct_value >= 0) == !o.ct_op.empty(),
          std::string("ct_value/ct_op disagree for '") + std::string(o.name) + "'");
  }

  // 4) THE COLLISION TEST. 'trunc' (float round-toward-zero, CTensorOp) and 'int_trunc'
  //    (turing Handler.Trunc, an integer width cast) are different operations that a
  //    naive name-union would merge. The table must keep them apart.
  {
    const OpDesc* ftrunc = find_op("trunc");
    const OpDesc* itrunc = find_op("int_trunc");
    check(ftrunc != nullptr && itrunc != nullptr, "both trunc variants must exist");
    if (ftrunc && itrunc) {
      check(ftrunc != itrunc, "float trunc and integer trunc must be distinct entries");
      check(ftrunc->op_class == OpClass::Unary, "float trunc is unary");
      check(itrunc->op_class == OpClass::Cast, "int_trunc is a cast");
      check(ftrunc->ct_value >= 0, "float trunc has a C target (CT_OP_TRUNC)");
      check(itrunc->ct_value < 0, "int_trunc has no C target");
      check(itrunc->handler == std::string_view("Trunc"),
            "int_trunc is the one carrying turing's Handler.Trunc");
      check(ftrunc->handler.empty(), "float trunc must not claim Handler.Trunc");
    }
  }

  // 5) maximum/minimum are binary arithmetic, not predicates -- they used to ride inside
  //    the old compare_value switch, and the CTensorOp enum promoted them out of it.
  for (const char* name : {"maximum", "minimum"}) {
    const OpDesc* o = find_op(name);
    if (check(o != nullptr, std::string("missing '") + name + "'")) {
      check(o->op_class == OpClass::Binary, std::string(name) + " is binary, not compare");
      check(!o->returns_bool, std::string(name) + " returns a value, not a bool");
    }
  }

  // 6) Comparisons return bools and lower to CMP.
  for (const char* name : {"less", "less_equal", "greater", "greater_equal",
                           "equal", "not_equal"}) {
    const OpDesc* o = find_op(name);
    if (check(o != nullptr, std::string("missing '") + name + "'")) {
      check(o->op_class == OpClass::Compare, std::string(name) + " is a compare");
      check(o->returns_bool, std::string(name) + " returns bool");
      check(o->kernel_op == std::string_view("CMP"), std::string(name) + " lowers to CMP");
    }
  }

  // 7) Reductions, contractions, shape-changing, creation, and ordering ops are not
  //    Tier-0 instructions. Each records HOW it composes (its Tier-1 family) without
  //    pretending an emitter can execute it directly, and none carries a CTensorOp
  //    code -- turing's dispatcher requires one equally-shaped slot per instruction.
  {
    const struct { const char* name; const char* family; } expected[] = {
        {"matmul", "contract"}, {"sum", "reduce"}, {"mean", "reduce"},
        {"log_softmax", "reduce"}, {"topk", "order"}, {"pad", "remap"},
        {"stack", "remap"}, {"cat", "remap"}, {"gather", "remap"},
        {"arange", "generate"},
    };
    for (const auto& e : expected) {
      const OpDesc* o = find_op(e.name);
      if (check(o != nullptr, std::string("missing '") + e.name + "'")) {
        check(!o->lowerable, std::string(e.name) + " is not one Tier-0 instruction");
        check(o->kernel_op.empty(), std::string(e.name) + " must carry no Tier-0 opcode");
        check(o->tier1_class == std::string_view(e.family),
              std::string(e.name) + " must name Tier-1 family " + e.family);
        check(o->ct_value < 0, std::string(e.name) + " must have no CTensorOp code");
      }
    }
  }

  // 8) Handler lookup resolves a uniquely-owned handler, and Call is documented as
  //    fanning out (so callers must not dispatch on it).
  {
    const OpDesc* neg = find_op_by_handler("Neg");
    check(neg != nullptr && neg->name == std::string_view("neg"),
          "Handler 'Neg' must resolve to canonical 'neg'");
    check(find_op_by_handler("NoSuchHandler") == nullptr,
          "unknown handler must yield nullptr");
    int call_count = 0;
    for (size_t i = 0; i < kOpCount; ++i) {
      if (kOps[i].handler == std::string_view("Call")) ++call_count;
    }
    check(call_count > 1, "Handler 'Call' is expected to fan out across many ops");
  }

  // 9) The reason this table exists: KernelIR's UNARY/BINARY/CMP/CAST can now actually be
  //    specified. Before the sub_op field, these opcodes had no operation selector and no
  //    emitter could lower them.
  {
    const OpDesc* sub = find_op("sub");
    check(sub != nullptr, "sub must exist");
    if (sub) {
      nodus::spirv::Instruction ins;
      ins.op = nodus::spirv::OpCode::BINARY;
      ins.sub_op = sub->canonical_id;
      ins.inputs = {nodus::spirv::Operand::ref({1}), nodus::spirv::Operand::ref({2})};
      ins.outputs = {{3}};

      check(ins.sub_op == static_cast<int32_t>(CanonicalOp::SUB),
            "instruction sub_op must equal the canonical enum value");
      const OpDesc* round_tripped = find_op(static_cast<CanonicalOp>(ins.sub_op));
      check(round_tripped == sub, "sub_op must round-trip back to the same descriptor");
      check(round_tripped->reflectable,
            "sub is non-commutative, so it must be marked reflectable");
      check(ins.inputs.size() == round_tripped->arity,
            "operand count must match the declared arity");
    }

    // Default-constructed instructions carry no selector.
    nodus::spirv::Instruction blank;
    check(blank.sub_op == -1, "sub_op must default to -1 (not applicable)");
  }

  if (failures == 0) {
    std::cout << "[CANONICAL-OPS] OK: " << kOpCount << " canonical IDs\n";
    return 0;
  }
  std::cerr << "[CANONICAL-OPS] " << failures << " failure(s)\n";
  return 1;
}
