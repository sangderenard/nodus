"""abstract_tensor.fill: a returned batch, cleaned to zero (research case 6393486c28b044c3b47caa2457f63cae).

The row's source is this AbstractTensor program. Nodus compiles it through Turing
(tools/turing_row_tool_compiler.py, lower_ast_source_to_ssa, LLVM) and adopts the product as a plugin row. The
canonical operation is fill (op 81); zero-fill is the calloc case. The volunteer cycle's tensor mint uses it to
return every slot of a batch to the null sentinel before the batch goes out again.
"""


def abstract_tensor_fill(batch):
    return batch.zeros_like()
