#pragma once

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace nodus::tensors {

using DyadicValue = double;

// ----------------------------
// Bit helpers on fixed-length big-endian byte strings
// Bit positions are LSB-based: bit 0 = least significant bit.
// ----------------------------
static inline uint32_t dy_get_nibble_be(const uint8_t* idx_be,
                                        uint32_t idx_bytes,
                                        uint32_t bitpos_lsb) {
    const uint32_t byte_from_end = bitpos_lsb >> 3;          // 0 = last byte
    const uint32_t bit_in_byte = bitpos_lsb & 7u;            // 0 = LSB
    const uint32_t bi = (idx_bytes - 1u) - byte_from_end;

    uint16_t two = 0u;
    if (bi < idx_bytes) {
        two = static_cast<uint16_t>(idx_be[bi]);
    }
    if (bi > 0) {
        two |= static_cast<uint16_t>(idx_be[bi - 1u]) << 8;
    }

    return static_cast<uint32_t>((two >> bit_in_byte) & 0xFu);
}

// ----------------------------
// Pool configuration
// ----------------------------
static inline AbstractTensorPool::Options dyadic_pool_options() {
    AbstractTensorPool::Options opt;
    opt.clear_on_release = false;
    opt.cache_handles = true;
    opt.enable_shape_bucketing = false;
    opt.max_cached_handles_total = 32;
    opt.max_cached_handles_per_key = 4;
    return opt;
}

static inline AbstractTensorPool& dyadic_pool() {
    static thread_local AbstractTensorPool pool(dyadic_pool_options());
    return pool;
}

// ----------------------------
// Staged log: chunked append-only tensors
// ----------------------------
struct DyadicChunk {
    AbstractTensorPool::PooledTensor idx;   // U8 [cap, idx_bytes]
    AbstractTensorPool::PooledTensor val;   // F64 [cap]
    uint32_t used = 0;

    uint8_t* idx_ptr = nullptr;
    double* val_ptr = nullptr;
};

struct DyadicLog {
    std::vector<DyadicChunk> chunks;
    uint64_t total = 0;
};

struct DyadicLeaf16 {
    uint16_t mask = 0;
    DyadicValue values[16];
};

struct DyadicBinsConfig {
    uint32_t idx_bytes = 0;    // fixed width of index byte strings
    uint32_t chunk_cap = 4096; // items per chunk
    uint32_t stage_bits = 4;   // digit width; fixed to 4 for dense-16 leaf
};

struct DyadicNode {
    enum class Kind : uint8_t {
        Log = 0,
        Split = 1,
        Leaf = 2
    };

    Kind kind = Kind::Log;
    uint32_t shift = 0; // LSB bit position for this stage (nibble)
    DyadicLog log;
    std::array<std::unique_ptr<DyadicNode>, 16> child;
    DyadicLeaf16 leaf;
};

class DyadicBins {
public:
    DyadicBins() = default;

    bool init(const DyadicBinsConfig& cfg, TensorBackend* backend) {
        cfg_ = cfg;
        backend_ = backend;
        if (!backend_ || cfg_.idx_bytes == 0 || cfg_.chunk_cap == 0) {
            return false;
        }
        if (cfg_.stage_bits != 4u) {
            return false;
        }

        max_bits_ = cfg_.idx_bytes * 8u;
        if (max_bits_ < cfg_.stage_bits) {
            return false;
        }

        root_ = std::make_unique<DyadicNode>();
        root_->kind = DyadicNode::Kind::Log;
        root_->shift = max_bits_ - cfg_.stage_bits;
        return true;
    }

    void close() {
        if (!backend_) return;
        auto* mem = dynamic_cast<InMemoryBackend*>(backend_);
        if (!mem) return;
        if (root_) {
            close_node(mem, root_.get());
        }
    }

    inline bool add(const uint8_t* idx_be, DyadicValue v) {
        if (!root_) return false;
        return append_to_log(root_->log, idx_be, v);
    }

    // Refine the entire tree to dense-16 leaves (nibble-by-nibble MSD radix).
    bool refine_all() {
        if (!root_) return false;
        return refine_node(root_.get());
    }

    DyadicNode* root() { return root_.get(); }
    const DyadicNode* root() const { return root_.get(); }

private:
    bool ensure_tail_chunk_mapped(DyadicLog& log) {
        if (!backend_) return false;
        auto* mem = dynamic_cast<InMemoryBackend*>(backend_);
        if (!mem) return false;

        if (!log.chunks.empty() && log.chunks.back().used < cfg_.chunk_cap) {
            DyadicChunk& ch = log.chunks.back();
            if (ch.idx_ptr && ch.val_ptr) return true;
        } else {
            log.chunks.emplace_back();
        }

        DyadicChunk& ch = log.chunks.back();
        if (!ch.idx.valid()) {
            TensorDesc id{};
            id.dtype = TensorDType::U8;
            id.layout = TensorLayout::Dense;
            id.shape.dims = {cfg_.chunk_cap, cfg_.idx_bytes};

            TensorDesc vd{};
            vd.dtype = TensorDType::F64;
            vd.layout = TensorLayout::Dense;
            vd.shape.dims = {cfg_.chunk_cap};

            ch.idx = dyadic_pool().acquire(id, backend_);
            ch.val = dyadic_pool().acquire(vd, backend_);
            if (!ch.idx.valid() || !ch.val.valid()) return false;
            ch.used = 0;
            ch.idx_ptr = nullptr;
            ch.val_ptr = nullptr;
        }

        if (!ch.idx_ptr) {
            void* p = nullptr;
            size_t bytes = 0;
            if (!mem->map(ch.idx.tensor().handle(), &p, &bytes)) return false;
            ch.idx_ptr = static_cast<uint8_t*>(p);
        }
        if (!ch.val_ptr) {
            void* p = nullptr;
            size_t bytes = 0;
            if (!mem->map(ch.val.tensor().handle(), &p, &bytes)) return false;
            ch.val_ptr = static_cast<double*>(p);
        }
        return true;
    }

    bool append_to_log(DyadicLog& log, const uint8_t* idx_be, DyadicValue v) {
        if (!ensure_tail_chunk_mapped(log)) return false;
        DyadicChunk& ch = log.chunks.back();
        const uint32_t pos = ch.used++;
        const size_t row_off = static_cast<size_t>(pos) * static_cast<size_t>(cfg_.idx_bytes);
        std::memcpy(ch.idx_ptr + row_off, idx_be, cfg_.idx_bytes);
        ch.val_ptr[pos] = v;
        ++log.total;
        return true;
    }

    template <typename Fn>
    bool for_each_entry(DyadicLog& log, Fn&& fn) {
        if (!backend_) return false;
        auto* mem = dynamic_cast<InMemoryBackend*>(backend_);
        if (!mem) return false;

        for (DyadicChunk& ch : log.chunks) {
            uint8_t* idx_ptr = ch.idx_ptr;
            double* val_ptr = ch.val_ptr;

            void* tmp = nullptr;
            size_t bytes = 0;
            bool mapped_idx = false;
            bool mapped_val = false;

            if (!idx_ptr) {
                if (!mem->map(ch.idx.tensor().handle(), &tmp, &bytes)) return false;
                idx_ptr = static_cast<uint8_t*>(tmp);
                mapped_idx = true;
            }
            if (!val_ptr) {
                if (!mem->map(ch.val.tensor().handle(), &tmp, &bytes)) {
                    if (mapped_idx) mem->unmap(ch.idx.tensor().handle());
                    return false;
                }
                val_ptr = static_cast<double*>(tmp);
                mapped_val = true;
            }

            for (uint32_t i = 0; i < ch.used; ++i) {
                const uint8_t* ix = idx_ptr + static_cast<size_t>(i) * cfg_.idx_bytes;
                const DyadicValue vv = val_ptr[i];
                fn(ix, vv);
            }

            if (mapped_idx) mem->unmap(ch.idx.tensor().handle());
            if (mapped_val) mem->unmap(ch.val.tensor().handle());
        }
        return true;
    }

    void clear_log(DyadicLog& log) {
        log.chunks.clear();
        log.total = 0;
    }

    bool refine_node(DyadicNode* node) {
        if (!node) return false;

        if (node->kind == DyadicNode::Kind::Leaf) {
            return true;
        }

        if (node->kind == DyadicNode::Kind::Split) {
            for (auto& c : node->child) {
                if (c && !refine_node(c.get())) return false;
            }
            return true;
        }

        // node->kind == Log
        if (node->shift == 0u) {
            node->kind = DyadicNode::Kind::Leaf;
            node->leaf.mask = 0u;
            auto ok = for_each_entry(node->log, [&](const uint8_t* ix, DyadicValue vv) {
                const uint32_t digit = dy_get_nibble_be(ix, cfg_.idx_bytes, 0u);
                const uint16_t bit = static_cast<uint16_t>(1u << digit);
                if (node->leaf.mask & bit) {
                    node->leaf.values[digit] += vv;
                } else {
                    node->leaf.mask |= bit;
                    node->leaf.values[digit] = vv;
                }
            });
            if (!ok) return false;
            clear_log(node->log);
            return true;
        }

        node->kind = DyadicNode::Kind::Split;
        const uint32_t next_shift = node->shift - cfg_.stage_bits;
        for (uint32_t i = 0; i < 16; ++i) {
            node->child[i] = std::make_unique<DyadicNode>();
            node->child[i]->kind = DyadicNode::Kind::Log;
            node->child[i]->shift = next_shift;
        }

        bool append_ok = true;
        auto ok = for_each_entry(node->log, [&](const uint8_t* ix, DyadicValue vv) {
            if (!append_ok) return;
            const uint32_t digit = dy_get_nibble_be(ix, cfg_.idx_bytes, node->shift);
            if (!append_to_log(node->child[digit]->log, ix, vv)) {
                append_ok = false;
            }
        });
        if (!ok || !append_ok) return false;

        clear_log(node->log);
        for (auto& c : node->child) {
            if (c && !refine_node(c.get())) return false;
        }
        return true;
    }

    void close_node(InMemoryBackend* mem, DyadicNode* node) {
        if (!node) return;
        if (node->kind == DyadicNode::Kind::Log) {
            for (auto& ch : node->log.chunks) {
                if (ch.idx_ptr) mem->unmap(ch.idx.tensor().handle());
                if (ch.val_ptr) mem->unmap(ch.val.tensor().handle());
                ch.idx_ptr = nullptr;
                ch.val_ptr = nullptr;
            }
        }
        if (node->kind == DyadicNode::Kind::Split) {
            for (auto& c : node->child) {
                if (c) close_node(mem, c.get());
            }
        }
    }

private:
    DyadicBinsConfig cfg_{};
    TensorBackend* backend_ = nullptr;
    uint32_t max_bits_ = 0;
    std::unique_ptr<DyadicNode> root_;
};

} // namespace nodus::tensors
