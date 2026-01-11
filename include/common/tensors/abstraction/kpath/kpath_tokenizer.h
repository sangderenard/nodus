#pragma once

#include "kpath_shaper.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

enum class TokenKind : uint8_t {
  Word,
  Punctuation,
  Symbol
};

struct TokenSpan final {
  size_t start = 0;
  size_t count = 0;
  TokenKind kind = TokenKind::Word;
};

// Lightweight tokenization options.
// This is intentionally heuristic (no ICU dependency) but Unicode-aware via
// HarfBuzz's unicode property helpers.
struct TokenizeOptions final {
  // If true, emits punctuation/symbol tokens as single-codepoint spans.
  // If false, they are treated as hard boundaries and skipped.
  bool emit_nonword_tokens = false;

  // If true, keeps apostrophes (', U+2019, U+02BC) inside words when surrounded
  // by letters/numbers.
  bool keep_internal_apostrophes = true;

  // If true, keeps hyphens/dashes inside words when surrounded by
  // letters/numbers (e.g. co-operate, rock-n-roll).
  bool keep_internal_hyphens = true;
};

// Splits a sequence into token spans using Unicode general categories.
// - Always splits on whitespace separators.
// - Uses simple heuristics to keep internal apostrophes/hyphens.
// - Optionally emits punctuation/symbol tokens.
std::vector<TokenSpan> tokenize_codepoints(const CodepointSequence& seq,
                                           const TokenizeOptions& opts = {});

} // namespace nodus::tensors::kpath
