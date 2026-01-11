#include "tool_api.h"
#include "tool_registry.h"

#include "common/tensors/abstraction/kpath/kpath_pipeline.h"
#include "common/tensors/abstraction/kpath/kpath_program.h"
#include "common/tensors/abstraction/kpath/kpath_toolpath_program.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using nodus::tensors::kpath::Atlas;
using nodus::tensors::kpath::AtlasBuilder;
using nodus::tensors::kpath::CodepointSequence;
using nodus::tensors::kpath::MetricSchema;
using nodus::tensors::kpath::Shaper;
using nodus::tensors::kpath::StepTape;
using nodus::tensors::kpath::TokenId;
using nodus::tensors::kpath::TokenLayoutPlan;
using nodus::tensors::kpath::ToolpathProgram;

namespace {

ValueTypeId vt(ValueTypeBuiltin b) {
  return ValueTypeRegistry::global().builtin(b);
}

bool is_numeric_type(ValueTypeId tid) {
  return tid == vt(VT_FLOAT32) || tid == vt(VT_FLOAT64) ||
         tid == vt(VT_INT8) || tid == vt(VT_INT16) || tid == vt(VT_INT32) || tid == vt(VT_INT64) ||
         tid == vt(VT_UINT8) || tid == vt(VT_UINT16) || tid == vt(VT_UINT32) || tid == vt(VT_UINT64);
}

bool pop_numeric_as_int64(RawStackFrame& frame, ValueTypeId tid, int64_t& out) {
  if (tid == vt(VT_FLOAT32)) {
    float v = 0.0f;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(std::lround(static_cast<double>(v)));
    return true;
  }
  if (tid == vt(VT_FLOAT64)) {
    double v = 0.0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(std::lround(v));
    return true;
  }
  if (tid == vt(VT_INT8)) {
    int8_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_INT16)) {
    int16_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_INT32)) {
    int32_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_INT64)) {
    int64_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = v;
    return true;
  }
  if (tid == vt(VT_UINT8)) {
    uint8_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_UINT16)) {
    uint16_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_UINT32)) {
    uint32_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  if (tid == vt(VT_UINT64)) {
    uint64_t v = 0;
    if (!raw_stack_pop_typed(frame, &v, tid)) return false;
    out = static_cast<int64_t>(v);
    return true;
  }
  return false;
}

std::string resolve_font_path_noexcept() {
  try {
    if (const char* env = std::getenv("KPATH_TEST_FONT")) {
      std::filesystem::path path(env);
      if (std::filesystem::exists(path)) return path.string();
    }
    const std::vector<std::filesystem::path> candidates = {
      "C:/Windows/Fonts/arial.ttf",
      "C:/Windows/Fonts/DejaVuSans.ttf",
      "C:/Windows/Fonts/seguisb.ttf"
    };
    for (const auto& candidate : candidates) {
      if (std::filesystem::exists(candidate)) return candidate.string();
    }
  } catch (...) {
  }
  return {};
}

} // namespace

class KpathUtf16StackToToolpathTool final : public ITool {
public:
  const char* id_cstr() const noexcept override { return "kpath_utf16_stack_to_toolpath"; }
  const char* name_cstr() const noexcept override { return "KPath UTF16 Stack To Toolpath"; }
  ToolCaps caps() const override { return ToolCaps::None; }

  void initialize(const ToolInitContext& ctx) override {
    (void)ctx;
    const std::string font_path = resolve_font_path_noexcept();
    if (!font_path.empty()) {
      font_loaded_ = shaper_.load_font(font_path, 18.0f);
    }
  }

  void shutdown() override {}
  void tick(double, HostAPI&) override {}
  void render(RenderContext&) override {}
  void render_text(TextRenderArgs&) override {}
  void render_table(TableRenderArgs&) override {}

  void execute_stack(ToolStackContext& ctx) override {
    if (!ctx.stack.raw) return;

    RawStackFrame& frame = *ctx.stack.raw;
    const ValueTypeId vt_ptr = vt(VT_VOID_PTR);

    // Optional: top-of-stack may hold a previous ToolpathProgram*; keep it alive
    // unless we successfully build a replacement.
    ToolpathProgram* previous = nullptr;
    {
      ValueTypeId top_tid = kInvalidValueTypeId;
      if (raw_stack_peek_type(frame, top_tid) && top_tid == vt_ptr) {
        void* candidate = nullptr;
        if (raw_stack_pop_typed(frame, &candidate, vt_ptr)) {
          ToolpathProgram* maybe_prog = static_cast<ToolpathProgram*>(candidate);
          if (maybe_prog && maybe_prog->magic == ToolpathProgram::kMagic) {
            previous = maybe_prog;
          } else {
            // Not ours; restore pointer exactly as-is.
            raw_stack_push_typed(frame, &candidate, vt_ptr);
          }
        }
      }
    }

    // Pop all numeric values and interpret as UTF-16 code units.
    std::vector<uint16_t> code_units;
    while (frame.byte_count > 0) {
      ValueTypeId top_tid = kInvalidValueTypeId;
      if (!raw_stack_peek_type(frame, top_tid)) break;
      if (!is_numeric_type(top_tid)) break;

      int64_t v = 0;
      if (!pop_numeric_as_int64(frame, top_tid, v)) break;
      code_units.push_back(static_cast<uint16_t>(static_cast<uint64_t>(v) & 0xFFFFu));
    }

    if (code_units.empty()) {
      // No input; restore previous if we claimed it.
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    // Popped from top => reverse to recover original push order.
    std::reverse(code_units.begin(), code_units.end());

    std::u16string utf16;
    utf16.reserve(code_units.size());
    for (uint16_t cu : code_units) utf16.push_back(static_cast<char16_t>(cu));

    if (!font_loaded_) {
      const std::string font_path = resolve_font_path_noexcept();
      if (!font_path.empty()) font_loaded_ = shaper_.load_font(font_path, 18.0f);
    }
    if (!font_loaded_) {
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    CodepointSequence sequence = nodus::tensors::kpath::codepoints_from_utf16(utf16);
    if (sequence.codepoints.empty()) {
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    TokenLayoutPlan plan;
    if (!build_token_from_sequence(atlas_builder_, shaper_, sequence, plan)) {
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    Atlas atlas = atlas_builder_.snapshot();
    MetricSchema schema = MetricSchema::MakeCoreV01();
    StepTape tape;
    if (!compile_token_to_tape(atlas, plan, schema, tape)) {
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    // Success: replace previous (if any) with a new program.
    if (previous) delete previous;

    ToolpathProgram* program = nullptr;
    try {
      program = new ToolpathProgram();
      program->schema = std::move(schema);
      program->tape = std::move(tape);
      program->atlas = std::move(atlas);
      program->token = plan.token;
    } catch (...) {
      delete program;
      if (previous) {
        void* p = previous;
        raw_stack_push_typed(frame, &p, vt_ptr);
      }
      return;
    }

    void* out_ptr = program;
    raw_stack_push_typed(frame, &out_ptr, vt_ptr);
  }

private:
  AtlasBuilder atlas_builder_{}; // stateful across calls
  Shaper shaper_{};
  bool font_loaded_ = false;
};

REGISTER_TOOL(KpathUtf16StackToToolpathTool,
              "kpath_utf16_stack_to_toolpath",
              "KPath UTF16 Stack To Toolpath",
              ToolCaps::None);
