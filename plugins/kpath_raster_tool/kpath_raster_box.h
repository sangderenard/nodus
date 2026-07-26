#pragma once

#include "tool_api.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"

// Opaque, magic-tagged handles carried as VT_VOID_PTR on the tool stack --
// same convention as ToolpathProgram in kpath_utf16_stack_to_toolpath_tool.cpp.
// This is the input/output contract for kpath's raster tools (this folder):
// a future ArmatureProgram-producing tool targets KpathArmatureProgramBox as
// its output, and anything downstream of the raster tools reads
// KpathRasterResultBox for the rasterized canvases. Neither box is a
// registered ValueType struct -- both sides just agree on the layout at
// compile time, per the guidance in tool_api.h's NODUS_PLUGIN_REGISTER_TYPES_NAME
// scope note: this data never needs to be understood outside the tools that
// produce/consume it, so it stays opaque instead of reaching for central
// type registration.

struct KpathArmatureProgramBox {
    static constexpr uint32_t kMagic = 0x4B415052u; // 'KAPR'
    uint32_t magic = kMagic;
    nodus::tensors::kpath::ArmatureProgram program;
};

struct KpathRasterResultBox {
    static constexpr uint32_t kMagic = 0x4B505252u; // 'KPRR'
    uint32_t magic = kMagic;
    nodus::tensors::kpath::TensorCanvas2D energy;
    nodus::tensors::kpath::TensorCanvas2D temp;
};

// Pops a KpathArmatureProgramBox* off the top of `frame` iff it's there and
// magic-tagged (transferring ownership to the caller). Returns nullptr and
// leaves the stack untouched otherwise.
inline KpathArmatureProgramBox* kpath_pop_armature_program_box(RawStackFrame& frame) {
    const ValueTypeId vt_ptr = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
    ValueTypeId top_tid = kInvalidValueTypeId;
    if (!raw_stack_peek_type(frame, top_tid) || top_tid != vt_ptr) return nullptr;
    void* candidate = nullptr;
    if (!raw_stack_pop_typed(frame, &candidate, vt_ptr)) return nullptr;
    auto* box = static_cast<KpathArmatureProgramBox*>(candidate);
    if (!box || box->magic != KpathArmatureProgramBox::kMagic) {
        // Not ours -- restore exactly as-is.
        raw_stack_push_typed(frame, &candidate, vt_ptr);
        return nullptr;
    }
    return box;
}

inline void kpath_push_raster_result_box(RawStackFrame& frame, KpathRasterResultBox* box) {
    const ValueTypeId vt_ptr = ValueTypeRegistry::global().builtin(VT_VOID_PTR);
    void* p = box;
    raw_stack_push_typed(frame, &p, vt_ptr);
}
