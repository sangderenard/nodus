"""abstract_tensor_sum_keepdim_4x4: one AbstractTensor operation as a Nodus row, a row of the composed log_softmax table
(research case 6393486c). Compiled through Turing (LLVM) and adopted as a plugin row.
"""


def abstract_tensor_sum_keepdim_4x4(a):
    return a.sum(dim=-1, keepdim=True)
