#pragma once

#include "kpath_atlas.h"
#include "kpath_schema.h"
#include "kpath_shaper.h"
#include "kpath_tape.h"
#include "kpath_tokenizer.h"

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

// Tokenizes a sequence and builds one atlas token per token span.
bool build_tokens_from_sequence(AtlasBuilder& builder,
                                Shaper& shaper,
                                const CodepointSequence& sequence,
                                std::vector<TokenLayoutPlan>& out_plans,
                                const TokenizeOptions& tokenize_opts = {});

// Injects an externally-authored glyph outline as its own token.
// This bypasses HarfBuzz shaping; the caller provides the outline.
bool build_external_glyph_token(AtlasBuilder& builder,
                                const GlyphOutline& outline,
                                TokenLayoutPlan& out_plan);

// Injects a codepoint->glyph edge using an externally-authored outline.
// This is the preferred injection path for PUA/custom codepoints that should
// participate in atlas postings and lookups.
bool build_codepoint_glyph_token(AtlasBuilder& builder,
                                 uint32_t codepoint,
                                 const GlyphOutline& outline,
                                 float advance_x,
                                 float advance_y,
                                 TokenLayoutPlan& out_plan);

bool compile_token_to_tape(const Atlas& atlas,
                           const TokenLayoutPlan& plan,
                           const MetricSchema& schema,
                           StepTape& out_tape);

} // namespace nodus::tensors::kpath
