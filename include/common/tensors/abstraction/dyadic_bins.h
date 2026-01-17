#pragma once

#include "common/tensors/abstraction/abstract_tensor_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cstdint>
#include <cstring>
#include <new>

namespace nodus::tensors {

using DyadicValue = double;

static constexpr uint32_t kDyadicInvalid = 0xFFFFFFFFu;

// ----------------------------
// Bit helpers on fixed-length big-endian byte strings
// Bit positions are LSB-based: bit 0 = least significant bit.
// ----------------------------
static inline uint32_t dy_get_byte_be(const uint8_t* idx_be,
                                      uint32_t idx_bytes,
                                      uint32_t bitpos_lsb) {
    const uint32_t byte_from_end = bitpos_lsb >> 3;          // 0 = last byte
    const uint32_t bi = (idx_bytes - 1u) - byte_from_end;
    if (bi >= idx_bytes) return 0u;
    return static_cast<uint32_t>(idx_be[bi]);
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
struct DyadicLog {
    uint32_t head = kDyadicInvalid;
    uint32_t tail = kDyadicInvalid;
    uint32_t chunk_count = 0;
    uint64_t total = 0;
};

struct DyadicLeaf256 {
    uint64_t mask[4] = {0, 0, 0, 0};
    DyadicValue values[256];
};

struct DyadicBinsConfig {
    uint32_t idx_bytes = 0;    // fixed width of index byte strings
    uint32_t chunk_cap = 4096; // items per chunk
    uint32_t stage_bits = 8;   // digit width; fixed to 8 for dense-256 leaf
    uint32_t max_nodes = 0;    // max node pool size (preallocated)
    uint32_t max_chunks = 0;   // max chunk pool size (preallocated)
    bool preallocate_chunks = true; // preallocate chunk tensors at init
};

struct DyadicNode {
    enum class Kind : uint8_t {
        Log = 0,
        Split = 1,
        Leaf = 2
    };

    Kind kind = Kind::Log;
    uint32_t shift = 0; // LSB bit position for this stage (byte)
    uint32_t next_free = kDyadicInvalid;
    DyadicLog log;
    uint32_t child[256] = {
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
        kDyadicInvalid, kDyadicInvalid, kDyadicInvalid, kDyadicInvalid,
    };
    DyadicLeaf256 leaf;
};

class DyadicBins {
public:
    DyadicBins() = default;

    bool init(const DyadicBinsConfig& cfg, TensorBackend* backend) {
        cfg_ = cfg;
        backend_ = backend;
        mem_ = dynamic_cast<InMemoryBackend*>(backend_);
        if (!backend_ || !mem_ || cfg_.idx_bytes == 0 || cfg_.chunk_cap == 0) {
            return false;
        }
        if (cfg_.stage_bits != 8u) {
            return false;
        }
        if (cfg_.max_nodes == 0 || cfg_.max_chunks == 0) {
            return false;
        }

        max_bits_ = cfg_.idx_bytes * 8u;
        if (max_bits_ < cfg_.stage_bits) {
            return false;
        }

        if (!init_pools()) return false;

        root_index_ = alloc_node(max_bits_ - cfg_.stage_bits);
        if (root_index_ == kDyadicInvalid) return false;

        DyadicNode* root = node_at(root_index_);
        root->kind = DyadicNode::Kind::Log;
        root->shift = max_bits_ - cfg_.stage_bits;
        return true;
    }

    void close() {
        if (!mem_) return;
        if (root_index_ != kDyadicInvalid) {
            close_node(node_at(root_index_));
        }

        if (chunks_) {
            for (uint32_t i = 0; i < chunk_pool_cap_; ++i) {
                DyadicChunk& ch = chunks_[i];
                if (ch.idx_ptr) mem_->unmap(ch.idx.tensor().handle());
                if (ch.val_ptr) mem_->unmap(ch.val.tensor().handle());
                ch.idx_ptr = nullptr;
                ch.val_ptr = nullptr;
                ch.~DyadicChunk();
            }
        }

        if (chunks_mapped_) {
            mem_->unmap(chunks_buf_.tensor().handle());
            chunks_mapped_ = false;
        }
        if (nodes_mapped_) {
            mem_->unmap(nodes_buf_.tensor().handle());
            nodes_mapped_ = false;
        }

        chunks_ = nullptr;
        nodes_ = nullptr;
        chunk_pool_cap_ = 0;
        node_pool_cap_ = 0;
        chunk_free_head_ = kDyadicInvalid;
        node_free_head_ = kDyadicInvalid;
        root_index_ = kDyadicInvalid;
        backend_ = nullptr;
        mem_ = nullptr;
    }

    inline bool add(const uint8_t* idx_be, DyadicValue v) {
        DyadicNode* root = node_at(root_index_);
        if (!root) return false;
        return append_to_log(root->log, idx_be, v);
    }

    // Refine the entire tree to dense-256 leaves (byte-by-byte MSD radix).
    bool refine_all() {
        DyadicNode* root = node_at(root_index_);
        if (!root) return false;
        return refine_node(root);
    }

    DyadicNode* root() { return node_at(root_index_); }
    const DyadicNode* root() const { return node_at(root_index_); }

private:
    struct DyadicChunk {
        AbstractTensorPool::PooledTensor idx;   // U8 [cap, idx_bytes]
        AbstractTensorPool::PooledTensor val;   // F64 [cap]
        uint32_t used = 0;
        uint32_t next = kDyadicInvalid;

        uint8_t* idx_ptr = nullptr;
        double* val_ptr = nullptr;
    };

    bool init_pools() {
        idx_desc_ = {};
        idx_desc_.dtype = TensorDType::U8;
        idx_desc_.layout = TensorLayout::Dense;
        idx_desc_.shape.dims = {cfg_.chunk_cap, cfg_.idx_bytes};

        val_desc_ = {};
        val_desc_.dtype = TensorDType::F64;
        val_desc_.layout = TensorLayout::Dense;
        val_desc_.shape.dims = {cfg_.chunk_cap};

        node_buf_desc_ = {};
        node_buf_desc_.dtype = TensorDType::U8;
        node_buf_desc_.layout = TensorLayout::Dense;
        node_buf_desc_.shape.dims = {static_cast<uint32_t>(cfg_.max_nodes * sizeof(DyadicNode))};

        chunk_buf_desc_ = {};
        chunk_buf_desc_.dtype = TensorDType::U8;
        chunk_buf_desc_.layout = TensorLayout::Dense;
        chunk_buf_desc_.shape.dims = {static_cast<uint32_t>(cfg_.max_chunks * sizeof(DyadicChunk))};

        nodes_buf_ = dyadic_pool().acquire(node_buf_desc_, backend_);
        chunks_buf_ = dyadic_pool().acquire(chunk_buf_desc_, backend_);
        if (!nodes_buf_.valid() || !chunks_buf_.valid()) return false;

        void* nptr = nullptr;
        size_t nbytes = 0;
        if (!mem_->map(nodes_buf_.tensor().handle(), &nptr, &nbytes)) return false;
        nodes_ = static_cast<DyadicNode*>(nptr);
        nodes_mapped_ = true;

        void* cptr = nullptr;
        size_t cbytes = 0;
        if (!mem_->map(chunks_buf_.tensor().handle(), &cptr, &cbytes)) return false;
        chunks_ = static_cast<DyadicChunk*>(cptr);
        chunks_mapped_ = true;

        node_pool_cap_ = cfg_.max_nodes;
        chunk_pool_cap_ = cfg_.max_chunks;

        node_free_head_ = 0;
        for (uint32_t i = 0; i < node_pool_cap_; ++i) {
            DyadicNode& n = nodes_[i];
            n.kind = DyadicNode::Kind::Log;
            n.shift = 0;
            n.next_free = (i + 1 < node_pool_cap_) ? (i + 1) : kDyadicInvalid;
            n.log = {};
            for (uint32_t j = 0; j < 256; ++j) {
                n.child[j] = kDyadicInvalid;
            }
            n.leaf.mask[0] = 0u;
            n.leaf.mask[1] = 0u;
            n.leaf.mask[2] = 0u;
            n.leaf.mask[3] = 0u;
            std::memset(n.leaf.values, 0, sizeof(n.leaf.values));
        }

        chunk_free_head_ = 0;
        for (uint32_t i = 0; i < chunk_pool_cap_; ++i) {
            DyadicChunk* ch = ::new (&chunks_[i]) DyadicChunk();
            ch->used = 0;
            ch->next = (i + 1 < chunk_pool_cap_) ? (i + 1) : kDyadicInvalid;
            ch->idx_ptr = nullptr;
            ch->val_ptr = nullptr;
        }

        if (cfg_.preallocate_chunks) {
            dyadic_pool().preallocate(idx_desc_, backend_, cfg_.max_chunks);
            dyadic_pool().preallocate(val_desc_, backend_, cfg_.max_chunks);
        }

        return true;
    }

    DyadicNode* node_at(uint32_t idx) {
        if (idx == kDyadicInvalid || !nodes_) return nullptr;
        return &nodes_[idx];
    }

    const DyadicNode* node_at(uint32_t idx) const {
        if (idx == kDyadicInvalid || !nodes_) return nullptr;
        return &nodes_[idx];
    }

    uint32_t alloc_node(uint32_t shift) {
        if (node_free_head_ == kDyadicInvalid) return kDyadicInvalid;
        const uint32_t idx = node_free_head_;
        DyadicNode& n = nodes_[idx];
        node_free_head_ = n.next_free;
        n.kind = DyadicNode::Kind::Log;
        n.shift = shift;
        n.next_free = kDyadicInvalid;
        n.log = {};
        for (uint32_t j = 0; j < 256; ++j) {
            n.child[j] = kDyadicInvalid;
        }
        n.leaf.mask[0] = 0u;
        n.leaf.mask[1] = 0u;
        n.leaf.mask[2] = 0u;
        n.leaf.mask[3] = 0u;
        std::memset(n.leaf.values, 0, sizeof(n.leaf.values));
        return idx;
    }

    uint32_t alloc_chunk() {
        if (chunk_free_head_ == kDyadicInvalid) return kDyadicInvalid;
        const uint32_t idx = chunk_free_head_;
        DyadicChunk& ch = chunks_[idx];
        chunk_free_head_ = ch.next;
        ch.next = kDyadicInvalid;
        ch.used = 0;
        ch.idx_ptr = nullptr;
        ch.val_ptr = nullptr;
        return idx;
    }

    bool ensure_chunk_mapped(DyadicChunk& ch) {
        if (!ch.idx.valid()) {
            ch.idx = dyadic_pool().acquire(idx_desc_, backend_);
            if (!ch.idx.valid()) return false;
        }
        if (!ch.val.valid()) {
            ch.val = dyadic_pool().acquire(val_desc_, backend_);
            if (!ch.val.valid()) return false;
        }
        if (!ch.idx_ptr) {
            void* p = nullptr;
            size_t bytes = 0;
            if (!mem_->map(ch.idx.tensor().handle(), &p, &bytes)) return false;
            ch.idx_ptr = static_cast<uint8_t*>(p);
        }
        if (!ch.val_ptr) {
            void* p = nullptr;
            size_t bytes = 0;
            if (!mem_->map(ch.val.tensor().handle(), &p, &bytes)) return false;
            ch.val_ptr = static_cast<double*>(p);
        }
        return true;
    }

    bool ensure_tail_chunk_mapped(DyadicLog& log) {
        if (!mem_) return false;

        if (log.tail != kDyadicInvalid) {
            DyadicChunk& tail = chunks_[log.tail];
            if (tail.used < cfg_.chunk_cap) {
                return ensure_chunk_mapped(tail);
            }
        }

        const uint32_t new_idx = alloc_chunk();
        if (new_idx == kDyadicInvalid) return false;
        DyadicChunk& ch = chunks_[new_idx];

        if (log.head == kDyadicInvalid) {
            log.head = new_idx;
            log.tail = new_idx;
        } else {
            chunks_[log.tail].next = new_idx;
            log.tail = new_idx;
        }
        log.chunk_count++;
        return ensure_chunk_mapped(ch);
    }

    bool append_to_log(DyadicLog& log, const uint8_t* idx_be, DyadicValue v) {
        if (!ensure_tail_chunk_mapped(log)) return false;
        DyadicChunk& ch = chunks_[log.tail];
        const uint32_t pos = ch.used++;
        const size_t row_off = static_cast<size_t>(pos) * static_cast<size_t>(cfg_.idx_bytes);
        std::memcpy(ch.idx_ptr + row_off, idx_be, cfg_.idx_bytes);
        ch.val_ptr[pos] = v;
        ++log.total;
        return true;
    }

    template <typename Fn>
    bool for_each_entry(DyadicLog& log, Fn&& fn) {
        if (!mem_) return false;

        for (uint32_t ci = log.head; ci != kDyadicInvalid; ci = chunks_[ci].next) {
            DyadicChunk& ch = chunks_[ci];
            uint8_t* idx_ptr = ch.idx_ptr;
            double* val_ptr = ch.val_ptr;

            void* tmp = nullptr;
            size_t bytes = 0;
            bool mapped_idx = false;
            bool mapped_val = false;

            if (!idx_ptr) {
                if (!mem_->map(ch.idx.tensor().handle(), &tmp, &bytes)) return false;
                idx_ptr = static_cast<uint8_t*>(tmp);
                mapped_idx = true;
            }
            if (!val_ptr) {
                if (!mem_->map(ch.val.tensor().handle(), &tmp, &bytes)) {
                    if (mapped_idx) mem_->unmap(ch.idx.tensor().handle());
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

            if (mapped_idx) mem_->unmap(ch.idx.tensor().handle());
            if (mapped_val) mem_->unmap(ch.val.tensor().handle());
        }
        return true;
    }

    void clear_log(DyadicLog& log) {
        if (!mem_) return;
        uint32_t ci = log.head;
        while (ci != kDyadicInvalid) {
            DyadicChunk& ch = chunks_[ci];
            const uint32_t next = ch.next;
            if (ch.idx_ptr) mem_->unmap(ch.idx.tensor().handle());
            if (ch.val_ptr) mem_->unmap(ch.val.tensor().handle());
            ch.idx_ptr = nullptr;
            ch.val_ptr = nullptr;
            ch.used = 0;
            ch.next = chunk_free_head_;
            chunk_free_head_ = ci;
            ci = next;
        }
        log.head = kDyadicInvalid;
        log.tail = kDyadicInvalid;
        log.chunk_count = 0;
        log.total = 0;
    }

    bool refine_node(DyadicNode* node) {
        if (!node) return false;

        if (node->kind == DyadicNode::Kind::Leaf) {
            return true;
        }

        if (node->kind == DyadicNode::Kind::Split) {
            for (uint32_t i = 0; i < 256; ++i) {
                if (node->child[i] != kDyadicInvalid) {
                    if (!refine_node(node_at(node->child[i]))) return false;
                }
            }
            return true;
        }

        // node->kind == Log
        if (node->log.total == 0) {
            return true;
        }
        if (node->shift == 0u) {
            node->kind = DyadicNode::Kind::Leaf;
            node->leaf.mask[0] = 0u;
            node->leaf.mask[1] = 0u;
            node->leaf.mask[2] = 0u;
            node->leaf.mask[3] = 0u;
            auto ok = for_each_entry(node->log, [&](const uint8_t* ix, DyadicValue vv) {
                const uint32_t digit = dy_get_byte_be(ix, cfg_.idx_bytes, 0u);
                const uint32_t word = digit >> 6;
                const uint64_t bit = 1ull << (digit & 63u);
                if (node->leaf.mask[word] & bit) {
                    node->leaf.values[digit] += vv;
                } else {
                    node->leaf.mask[word] |= bit;
                    node->leaf.values[digit] = vv;
                }
            });
            if (!ok) return false;
            clear_log(node->log);
            return true;
        }

        node->kind = DyadicNode::Kind::Split;
        const uint32_t next_shift = node->shift - cfg_.stage_bits;

        bool append_ok = true;
        auto ok = for_each_entry(node->log, [&](const uint8_t* ix, DyadicValue vv) {
            if (!append_ok) return;
            const uint32_t digit = dy_get_byte_be(ix, cfg_.idx_bytes, node->shift);
            if (node->child[digit] == kDyadicInvalid) {
                const uint32_t child_idx = alloc_node(next_shift);
                if (child_idx == kDyadicInvalid) {
                    append_ok = false;
                    return;
                }
                node->child[digit] = child_idx;
                DyadicNode* child = node_at(child_idx);
                child->kind = DyadicNode::Kind::Log;
                child->shift = next_shift;
            }
            if (!append_to_log(node_at(node->child[digit])->log, ix, vv)) {
                append_ok = false;
            }
        });
        if (!ok || !append_ok) return false;

        clear_log(node->log);
        for (uint32_t i = 0; i < 256; ++i) {
            if (node->child[i] != kDyadicInvalid) {
                if (!refine_node(node_at(node->child[i]))) return false;
            }
        }
        return true;
    }

    void close_node(DyadicNode* node) {
        if (!node) return;
        if (node->kind == DyadicNode::Kind::Log) {
            clear_log(node->log);
        }
        if (node->kind == DyadicNode::Kind::Split) {
            for (uint32_t i = 0; i < 256; ++i) {
                if (node->child[i] != kDyadicInvalid) {
                    close_node(node_at(node->child[i]));
                }
            }
        }
    }

private:
    DyadicBinsConfig cfg_{};
    TensorBackend* backend_ = nullptr;
    InMemoryBackend* mem_ = nullptr;
    uint32_t max_bits_ = 0;

    AbstractTensorPool::PooledTensor nodes_buf_;
    AbstractTensorPool::PooledTensor chunks_buf_;
    DyadicNode* nodes_ = nullptr;
    DyadicChunk* chunks_ = nullptr;
    uint32_t node_pool_cap_ = 0;
    uint32_t chunk_pool_cap_ = 0;
    uint32_t node_free_head_ = kDyadicInvalid;
    uint32_t chunk_free_head_ = kDyadicInvalid;
    uint32_t root_index_ = kDyadicInvalid;
    bool nodes_mapped_ = false;
    bool chunks_mapped_ = false;

    TensorDesc idx_desc_{};
    TensorDesc val_desc_{};
    TensorDesc node_buf_desc_{};
    TensorDesc chunk_buf_desc_{};
};

} // namespace nodus::tensors
