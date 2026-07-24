#pragma once

#include "edge_tensor_fifo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

namespace nodus::runtime {

inline constexpr uint32_t kEdgeTransactionMagic = 0x5854464Eu; // "NFTX" LE
inline constexpr uint32_t kEdgeTransactionVersion = 1u;
inline constexpr size_t kEdgeTransactionHeaderBytes =
    sizeof(uint32_t) * 7u + sizeof(uint64_t) * 6u;

inline size_t edge_transaction_snapshot_size(const EdgeTensorFifo& fifo) {
    if (!fifo.impl || !fifo.impl->configured) return 0;
    size_t readers = 0;
    for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
        if (fifo.impl->readers[i].key.load(std::memory_order_acquire) != 0) {
            ++readers;
        }
    }
    return kEdgeTransactionHeaderBytes +
           fifo.impl->slots * sizeof(uint64_t) +
           readers * sizeof(uint64_t) * 2u +
           fifo.impl->stride * fifo.impl->slots * fifo.impl->elem_size;
}

inline bool edge_transaction_snapshot_fill(
    EdgeTensorFifo& fifo, void* out_buf, size_t out_len
) {
    if (!out_buf || !fifo.impl || !fifo.impl->configured) return false;
    auto* impl = fifo.impl.get();
    const size_t need = edge_transaction_snapshot_size(fifo);
    if (need == 0 || out_len < need) return false;

    uint32_t reader_count = 0;
    for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
        if (impl->readers[i].key.load(std::memory_order_acquire) != 0) {
            ++reader_count;
        }
    }
    auto* dst = static_cast<uint8_t*>(out_buf);
    size_t off = 0;
    auto put32 = [&](uint32_t value) {
        std::memcpy(dst + off, &value, sizeof(value));
        off += sizeof(value);
    };
    auto put64 = [&](uint64_t value) {
        std::memcpy(dst + off, &value, sizeof(value));
        off += sizeof(value);
    };

    std::unique_lock<std::mutex> lock(impl->cv_mu);
    put32(kEdgeTransactionMagic);
    put32(kEdgeTransactionVersion);
    put32(static_cast<uint32_t>(impl->type_id));
    put32(static_cast<uint32_t>(fifo.layout));
    put32(static_cast<uint32_t>(fifo.dtype));
    put32(reader_count);
    put32(0u);
    put64(static_cast<uint64_t>(impl->slots));
    put64(static_cast<uint64_t>(impl->stride));
    put64(static_cast<uint64_t>(impl->elem_size));
    put64(static_cast<uint64_t>(impl->top_k));
    put64(impl->writer.load(std::memory_order_acquire));
    put64(impl->write_seq.load(std::memory_order_acquire));
    for (size_t i = 0; i < impl->slots; ++i) {
        put64(impl->slot_seq[i].load(std::memory_order_acquire));
    }
    for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
        const uint64_t key =
            impl->readers[i].key.load(std::memory_order_acquire);
        if (key == 0) continue;
        put64(key);
        put64(impl->readers[i].seq.load(std::memory_order_acquire));
    }
    const size_t storage_bytes =
        impl->stride * impl->slots * impl->elem_size;
    const gp_mem_backend_vtable_t* vt =
        gp_mem_backend_get_vtable(impl->storage_handle);
    bool copied = false;
    if (vt && vt->copy_from_backend) {
        copied = vt->copy_from_backend(
            impl->storage_handle, 0, dst + off, storage_bytes
        ) != 0;
    }
    if (!copied) {
        void* mapped = gp_mem_backend_map_or_null(impl->storage_handle);
        if (!mapped) return false;
        std::memcpy(dst + off, mapped, storage_bytes);
        gp_mem_backend_unmap(impl->storage_handle);
    }
    off += storage_bytes;
    return off == need;
}

inline bool edge_transaction_snapshot_restore(
    EdgeTensorFifo& fifo, const void* buf, size_t buf_len
) {
    if (!buf || !fifo.impl || !fifo.impl->configured) return false;
    auto* impl = fifo.impl.get();
    if (buf_len < kEdgeTransactionHeaderBytes) return false;
    const auto* src = static_cast<const uint8_t*>(buf);
    size_t off = 0;
    auto get32 = [&]() {
        uint32_t value = 0;
        std::memcpy(&value, src + off, sizeof(value));
        off += sizeof(value);
        return value;
    };
    auto get64 = [&]() {
        uint64_t value = 0;
        std::memcpy(&value, src + off, sizeof(value));
        off += sizeof(value);
        return value;
    };

    if (get32() != kEdgeTransactionMagic ||
        get32() != kEdgeTransactionVersion) return false;
    const int32_t type_id = static_cast<int32_t>(get32());
    const auto layout = static_cast<nodus::tensors::TensorLayout>(get32());
    const auto dtype = static_cast<nodus::tensors::TensorDType>(get32());
    const uint32_t reader_count = get32();
    if (get32() != 0u || reader_count > EdgeTensorFifo::kMaxReaders) {
        return false;
    }
    const uint64_t slots = get64();
    const uint64_t stride = get64();
    const uint64_t elem_size = get64();
    const uint64_t top_k = get64();
    const uint64_t writer = get64();
    const uint64_t write_seq = get64();
    if (slots != impl->slots || stride != impl->stride ||
        elem_size != impl->elem_size || top_k != impl->top_k ||
        type_id != impl->type_id || layout != fifo.layout ||
        dtype != fifo.dtype) return false;

    const size_t expected =
        kEdgeTransactionHeaderBytes +
        impl->slots * sizeof(uint64_t) +
        static_cast<size_t>(reader_count) * sizeof(uint64_t) * 2u +
        impl->stride * impl->slots * impl->elem_size;
    if (buf_len != expected) return false;

    std::vector<uint64_t> slot_seq(impl->slots);
    for (size_t i = 0; i < impl->slots; ++i) {
        slot_seq[i] = get64();
        if (slot_seq[i] > write_seq) return false;
    }
    std::vector<std::pair<uint64_t, uint64_t>> readers;
    readers.reserve(reader_count);
    for (uint32_t i = 0; i < reader_count; ++i) {
        const uint64_t key = get64();
        const uint64_t seq = get64();
        if (key == 0 || seq > write_seq) return false;
        for (const auto& prior : readers) {
            if (prior.first == key) return false;
        }
        readers.emplace_back(key, seq);
    }
    for (const auto& reader : readers) {
        if (!fifo.find_reader(reader.first)) return false;
    }
    size_t current_readers = 0;
    for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
        if (impl->readers[i].key.load(std::memory_order_acquire) != 0) {
            ++current_readers;
        }
    }
    if (current_readers != readers.size()) return false;

    std::unique_lock<std::mutex> lock(impl->cv_mu);
    const size_t storage_bytes =
        impl->stride * impl->slots * impl->elem_size;
    const gp_mem_backend_vtable_t* vt =
        gp_mem_backend_get_vtable(impl->storage_handle);
    bool copied = false;
    if (vt && vt->copy_to_backend) {
        copied = vt->copy_to_backend(
            impl->storage_handle, 0, src + off, storage_bytes
        ) != 0;
    }
    if (!copied) {
        void* mapped = gp_mem_backend_map_or_null(impl->storage_handle);
        if (!mapped) return false;
        std::memcpy(mapped, src + off, storage_bytes);
        gp_mem_backend_unmap(impl->storage_handle);
    }
    for (size_t i = 0; i < impl->slots; ++i) {
        impl->slot_seq[i].store(slot_seq[i], std::memory_order_release);
    }
    uint64_t minimum = write_seq;
    for (const auto& reader : readers) {
        auto* entry = fifo.find_reader(reader.first);
        entry->seq.store(reader.second, std::memory_order_release);
        minimum = std::min(minimum, reader.second);
    }
    impl->writer.store(writer, std::memory_order_release);
    impl->write_seq.store(write_seq, std::memory_order_release);
    impl->last_write_seq.store(write_seq, std::memory_order_release);
    impl->last_read_seq.store(minimum, std::memory_order_release);
    impl->cv.notify_all();
    return true;
}

} // namespace nodus::runtime
