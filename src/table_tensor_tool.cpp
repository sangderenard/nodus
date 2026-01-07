#include "table_tensor_tool.h"

#include "common/tensors/runtime/tensor_runtime.h"

struct GP_TableTensorToolState {
    nodus::tensors::Runtime* runtime = nullptr;
};

GP_TableTensorToolState* gp_table_tensor_tool_create(const GP_TableTensorToolConfig* /*config*/) {
    auto* state = new GP_TableTensorToolState{};
    state->runtime = &nodus::tensors::runtime_singleton();
    return state;
}

void gp_table_tensor_tool_destroy(GP_TableTensorToolState* state) {
    delete state;
}

void gp_table_tensor_tool_tick(GP_TableTensorToolState* state, GP_TableContext* /*table*/, int /*module_idx*/, int /*row_idx*/) {
    if (!state || !state->runtime) {
        return;
    }
    // Placeholder: tie table tool execution into tensor runtime once defined.
}
