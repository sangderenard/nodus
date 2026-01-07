#pragma once

struct GP_TableContext;

struct GP_TableTensorToolConfig {
    int reserved = 0;
};

struct GP_TableTensorToolState;

GP_TableTensorToolState* gp_table_tensor_tool_create(const GP_TableTensorToolConfig* config);
void gp_table_tensor_tool_destroy(GP_TableTensorToolState* state);
void gp_table_tensor_tool_tick(GP_TableTensorToolState* state, GP_TableContext* table, int module_idx, int row_idx);
