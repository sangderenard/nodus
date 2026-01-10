#pragma once

#include "kpath_atlas.h"
#include "kpath_schema.h"
#include "kpath_shaper.h"
#include "kpath_tape.h"

#include <vector>

namespace nodus::tensors::kpath {

struct TokenLayoutPlan final {
  TokenId token{};
  std::vector<EdgeId> edges;
  std::vector<float> advance_x;
  std::vector<float> advance_y;
};

bool build_token_from_sequence(AtlasBuilder& builder,
                               Shaper& shaper,
                               const CodepointSequence& sequence,
                               TokenLayoutPlan& out_plan);

bool compile_token_to_tape(const Atlas& atlas,
                           const TokenLayoutPlan& plan,
                           const MetricSchema& schema,
                           StepTape& out_tape);

} // namespace nodus::tensors::kpath
