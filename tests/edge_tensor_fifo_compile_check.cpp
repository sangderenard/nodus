#include "edge_tensor_fifo.h"

#include <cstddef>
#include <cstdint>

// This object target deliberately links nothing. Its purpose is to prove that
// EdgeTensorFifo is a self-contained runtime header and does not receive table,
// canvas, rope, stage, or ThreadManager declarations transitively from the
// historical table_abi.cpp amalgam.
bool edge_tensor_fifo_compile_check(
    EdgeTensorFifo& fifo,
    uint64_t writer,
    uint64_t reader,
    const void* input,
    size_t input_bytes,
    void* output,
    size_t output_bytes
) {
    fifo.configure({static_cast<int32_t>(input_bytes)}, 4, 0);
    if (!fifo.subscribe(reader, true)) {
        return false;
    }
    bool dropped = false;
    if (!fifo.push(0, writer, input, input_bytes, &dropped) || dropped) {
        return false;
    }
    size_t written = 0;
    return fifo.pop(reader, output, output_bytes, written) &&
           written == input_bytes;
}
