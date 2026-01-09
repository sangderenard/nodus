#include "common/tensors/abstraction/table_edge_tensor_bridge.h"

#include "table_abi.h"

extern "C" int32_t gp_table_edge_set_tensor_storage(GP_TableContext* ctx,
                                                    int32_t edge_idx,
                                                    nodus::tensors::AbstractTensorHandle handle,
                                                    nodus::tensors::TensorBackend* backend);

namespace nodus::tensors {

int32_t set_table_edge_tensor_storage(GP_TableContext* ctx,
                                      int32_t edge_idx,
                                      AbstractTensorHandle handle,
                                      TensorBackend* backend) {
    return gp_table_edge_set_tensor_storage(ctx, edge_idx, handle, backend);
}

} // namespace nodus::tensors
