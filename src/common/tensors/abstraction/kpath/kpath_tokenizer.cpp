#include "common/tensors/abstraction/kpath/kpath_tokenizer.h"

#include <hb.h>

#include <algorithm>

namespace nodus::tensors::kpath {

namespace {

hb_unicode_general_category_t category_of(uint32_t cp) {
  hb_unicode_funcs_t* ufuncs = hb_unicode_funcs_get_default();
  return hb_unicode_general_category(ufuncs, cp);
}

bool is_whitespace(uint32_t cp) {
  if (cp == 0x09u || cp == 0x0Au || cp == 0x0Bu || cp == 0x0Cu || cp == 0x0Du) return true;
  const auto cat = category_of(cp);
  return cat == HB_UNICODE_GENERAL_CATEGORY_SPACE_SEPARATOR ||
         cat == HB_UNICODE_GENERAL_CATEGORY_LINE_SEPARATOR ||
         cat == HB_UNICODE_GENERAL_CATEGORY_PARAGRAPH_SEPARATOR;
}

bool is_letter_or_number_category(hb_unicode_general_category_t cat) {
  switch (cat) {
    case HB_UNICODE_GENERAL_CATEGORY_UPPERCASE_LETTER:
    case HB_UNICODE_GENERAL_CATEGORY_LOWERCASE_LETTER:
    case HB_UNICODE_GENERAL_CATEGORY_TITLECASE_LETTER:
    case HB_UNICODE_GENERAL_CATEGORY_MODIFIER_LETTER:
    case HB_UNICODE_GENERAL_CATEGORY_OTHER_LETTER:
    case HB_UNICODE_GENERAL_CATEGORY_DECIMAL_NUMBER:
    case HB_UNICODE_GENERAL_CATEGORY_LETTER_NUMBER:
    case HB_UNICODE_GENERAL_CATEGORY_OTHER_NUMBER:
      return true;
    default:
      return false;
  }
}

bool is_mark_category(hb_unicode_general_category_t cat) {
  switch (cat) {
    case HB_UNICODE_GENERAL_CATEGORY_NON_SPACING_MARK:
    case HB_UNICODE_GENERAL_CATEGORY_SPACING_MARK:
    case HB_UNICODE_GENERAL_CATEGORY_ENCLOSING_MARK:
      return true;
    default:
      return false;
  }
}

bool is_word_char(uint32_t cp) {
  const auto cat = category_of(cp);
  return is_letter_or_number_category(cat) || is_mark_category(cat);
}

bool is_punct_or_symbol(uint32_t cp) {
  const auto cat = category_of(cp);
  switch (cat) {
    case HB_UNICODE_GENERAL_CATEGORY_CONNECT_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_DASH_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_CLOSE_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_FINAL_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_INITIAL_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_OTHER_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_OPEN_PUNCTUATION:
    case HB_UNICODE_GENERAL_CATEGORY_MATH_SYMBOL:
    case HB_UNICODE_GENERAL_CATEGORY_CURRENCY_SYMBOL:
    case HB_UNICODE_GENERAL_CATEGORY_MODIFIER_SYMBOL:
    case HB_UNICODE_GENERAL_CATEGORY_OTHER_SYMBOL:
      return true;
    default:
      return false;
  }
}

bool is_apostrophe(uint32_t cp) {
  // ASCII apostrophe + common Unicode apostrophe forms.
  return cp == 0x27u || cp == 0x2019u || cp == 0x02BCu;
}

bool is_hyphen(uint32_t cp) {
  // Hyphen-minus, hyphen, non-breaking hyphen, figure dash, en dash.
  return cp == 0x2Du || cp == 0x2010u || cp == 0x2011u || cp == 0x2012u || cp == 0x2013u;
}

TokenKind classify_single(uint32_t cp) {
  const auto cat = category_of(cp);
  if (is_letter_or_number_category(cat) || is_mark_category(cat)) return TokenKind::Word;
  if (cat == HB_UNICODE_GENERAL_CATEGORY_MATH_SYMBOL ||
      cat == HB_UNICODE_GENERAL_CATEGORY_CURRENCY_SYMBOL ||
      cat == HB_UNICODE_GENERAL_CATEGORY_MODIFIER_SYMBOL ||
      cat == HB_UNICODE_GENERAL_CATEGORY_OTHER_SYMBOL) {
    return TokenKind::Symbol;
  }
  return TokenKind::Punctuation;
}

} // namespace

std::vector<TokenSpan> tokenize_codepoints(const CodepointSequence& seq, const TokenizeOptions& opts) {
  std::vector<TokenSpan> out;
  const auto& cps = seq.codepoints;
  const size_t n = cps.size();
  if (n == 0) return out;

  size_t i = 0;
  while (i < n) {
    while (i < n && is_whitespace(cps[i])) ++i;
    if (i >= n) break;

    // If we start on a non-word char, either emit it as its own token or skip it.
    if (!is_word_char(cps[i])) {
      if (opts.emit_nonword_tokens && is_punct_or_symbol(cps[i])) {
        out.push_back(TokenSpan{i, 1, classify_single(cps[i])});
      }
      ++i;
      continue;
    }

    const size_t start = i;
    TokenKind kind = TokenKind::Word;

    while (i < n && !is_whitespace(cps[i])) {
      const uint32_t cp = cps[i];
      if (is_word_char(cp)) {
        ++i;
        continue;
      }

      const bool prev_word = (i > start) ? is_word_char(cps[i - 1]) : false;
      const bool next_word = (i + 1 < n) ? is_word_char(cps[i + 1]) : false;

      if (opts.keep_internal_apostrophes && is_apostrophe(cp) && prev_word && next_word) {
        ++i;
        continue;
      }
      if (opts.keep_internal_hyphens && is_hyphen(cp) && prev_word && next_word) {
        ++i;
        continue;
      }

      // Hard boundary on other punctuation/symbol.
      break;
    }

    const size_t count = i - start;
    if (count > 0) {
      out.push_back(TokenSpan{start, count, kind});
    }

    // Handle boundary character (punct/symbol) as an optional standalone token.
    if (i < n && !is_whitespace(cps[i]) && !is_word_char(cps[i])) {
      if (opts.emit_nonword_tokens && is_punct_or_symbol(cps[i])) {
        out.push_back(TokenSpan{i, 1, classify_single(cps[i])});
      }
      ++i;
    }
  }

  return out;
}

} // namespace nodus::tensors::kpath
