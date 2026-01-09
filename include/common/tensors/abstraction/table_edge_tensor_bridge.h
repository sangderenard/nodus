#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_backend.h"

struct GP_TableContext;

namespace nodus::tensors {

int32_t set_table_edge_tensor_storage(GP_TableContext* ctx,
                                      int32_t edge_idx,
                                      AbstractTensorHandle handle,
                                      TensorBackend* backend);

} // namespace nodus::tensors
