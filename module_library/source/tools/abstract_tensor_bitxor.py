"""abstract_tensor.bitxor: one basic AbstractTensor operation as a Nodus row (research case 6393486c).

Compiled through Turing (tools/turing_row_tool_compiler.py, LLVM) and adopted as a plugin row; this program is
the row's source and its history.
"""


def abstract_tensor_bitxor(a, b):
    return a ^ b
