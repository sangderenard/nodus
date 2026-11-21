"""abstract_tensor.cat: one basic AbstractTensor operation as a Nodus row (research case 6393486c).

Compiled through Turing (tools/turing_row_tool_compiler.py, LLVM) and adopted as a plugin row; this program is
the row's source and its history.
"""
from src.common.tensors.abstraction import AbstractTensor


def abstract_tensor_cat(a, b):
    return AbstractTensor.cat([a, b], dim=0)
