"""abstract_tensor_truediv_4x4_4x1: one AbstractTensor operation as a Nodus row, a row of the composed log_softmax table
(research case 6393486c). Compiled through Turing (LLVM) and adopted as a plugin row.
"""


def abstract_tensor_truediv_4x4_4x1(a, b):
    return a / b
