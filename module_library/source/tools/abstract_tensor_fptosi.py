"""abstract_tensor.fptosi: one basic AbstractTensor operation as a Nodus row (research case 6393486c).

Compiled through Turing (tools/turing_row_tool_compiler.py, LLVM) and adopted as a plugin row; this program is
the row's source and its history.
"""


def abstract_tensor_fptosi(a):
    return a.to_dtype('int64')
