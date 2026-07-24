// Edge list helpers
#include "console_logger.h"
#include "edge_tensor_fifo_transaction.h"
#include <cmath>
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
// Helper: determines whether a FIFO should be treated as BYREF-capable based
// on subgroup flags OR an attached memory backend that advertises BYREF safety.
static inline bool fifo_effective_byref(const EdgeTensorFifo& fifo, uint32_t flags) {
    if (flags_imply_byref(flags)) return true;
    gp_mem_backend_handle_t h = fifo.get_backend();
    if (!h) return false;
    return gp_mem_backend_supports_byref(h) != 0;
}

// Serialize a live EdgeTensorFifo snapshot into a contiguous buffer.
// Format:
// [uint64_t slots][uint64_t stride][uint64_t elem_size][int32_t type_id][uint32_t padding]
// [uint64_t write_seq]
// [uint64_t slot_seq[slots]]
// [byte storage (stride * slots * elem_size)]
extern "C" int32_t gp_table_edge_serialize_snapshot(GP_TableContext* ctx, int32_t edge_idx, void* out_buf, size_t out_len, size_t* out_written) {
    if (!ctx || !out_buf || !out_written) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edge_fifos.size())) return 0;
    auto &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!fifo.impl || !fifo.impl->configured) return 0;
    auto impl = fifo.impl.get();
    // compute sizes
    uint64_t slots = static_cast<uint64_t>(impl->slots);
    uint64_t stride = static_cast<uint64_t>(impl->stride);
    uint64_t elem_size = static_cast<uint64_t>(impl->elem_size);
    int32_t type_id = impl->type_id;
    uint64_t write_seq = impl->write_seq.load(std::memory_order_acquire);
    size_t storage_bytes = static_cast<size_t>(stride * slots * elem_size);
    size_t header_sz = sizeof(uint64_t) * 4 + sizeof(int32_t) + sizeof(uint32_t); // slots,stride,elem_size,write_seq,type_id+pad
    size_t slot_seq_sz = static_cast<size_t>(slots) * sizeof(uint64_t);
    size_t need = header_sz + slot_seq_sz + storage_bytes;
    if (out_len < need) return 0;

    uint8_t* dst = reinterpret_cast<uint8_t*>(out_buf);
    size_t off = 0;
    // write metadata under lock to ensure consistent snapshot
    std::unique_lock<std::mutex> lk(impl->cv_mu);
    std::memcpy(dst + off, &slots, sizeof(slots)); off += sizeof(slots);
    std::memcpy(dst + off, &stride, sizeof(stride)); off += sizeof(stride);
    std::memcpy(dst + off, &elem_size, sizeof(elem_size)); off += sizeof(elem_size);
    std::memcpy(dst + off, &type_id, sizeof(type_id)); off += sizeof(type_id);
    uint32_t pad = 0; std::memcpy(dst + off, &pad, sizeof(pad)); off += sizeof(pad);
    std::memcpy(dst + off, &write_seq, sizeof(write_seq)); off += sizeof(write_seq);

    // slot_seq array
    for (uint64_t i = 0; i < slots; ++i) {
        uint64_t s = impl->slot_seq[i].load(std::memory_order_acquire);
        std::memcpy(dst + off, &s, sizeof(s)); off += sizeof(s);
    }

    // storage bytes: copy from backend/storage_handle
    if (impl->storage_handle) {
        const gp_mem_backend_vtable_t* svt = gp_mem_backend_get_vtable(impl->storage_handle);
        bool ok = false;
        if (svt && svt->copy_from_backend) {
            if (svt->copy_from_backend(impl->storage_handle, 0, dst + off, storage_bytes)) ok = true;
        }
        if (!ok) {
            void* m = gp_mem_backend_map_or_null(impl->storage_handle);
            if (!m) return 0;
            std::memcpy(dst + off, m, storage_bytes);
            gp_mem_backend_unmap(impl->storage_handle);
        }
        off += storage_bytes;
    } else {
        // no storage handle: nothing to copy, fill zeros
        std::memset(dst + off, 0, storage_bytes);
        off += storage_bytes;
    }

    *out_written = off;
    return 1;
}

// Chunked serialization: writer callback will be invoked sequentially. The
// first write contains the header + slot_seq blob; subsequent writes are the
// storage bytes in order. The writer should return non-zero on success.
typedef int (*gp_snapshot_writer_fn)(void* user, const void* data, size_t data_len, int is_final);
extern "C" int32_t gp_table_edge_serialize_snapshot_chunked(GP_TableContext* ctx, int32_t edge_idx, size_t chunk_size, gp_snapshot_writer_fn writer, void* user) {
    if (!ctx || !writer) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edge_fifos.size())) return 0;
    if (chunk_size == 0) chunk_size = 4 * 1024 * 1024; // default 4MB
    auto &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!fifo.impl || !fifo.impl->configured) return 0;
    auto impl = fifo.impl.get();

    uint64_t slots = static_cast<uint64_t>(impl->slots);
    uint64_t stride = static_cast<uint64_t>(impl->stride);
    uint64_t elem_size = static_cast<uint64_t>(impl->elem_size);
    int32_t type_id = impl->type_id;
    uint64_t write_seq = impl->write_seq.load(std::memory_order_acquire);
    size_t storage_bytes = static_cast<size_t>(stride * slots * elem_size);
    size_t slot_seq_sz = static_cast<size_t>(slots) * sizeof(uint64_t);
    size_t header_sz = sizeof(uint64_t) * 3 + sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint64_t) + slot_seq_sz; // slots,stride,elem_size,type_id+pad,write_seq,slot_seq

    // Build header+slot_seq in a temporary buffer and send as first writer call
    std::vector<uint8_t> header;
    try { header.resize(header_sz); } catch (...) { return 0; }
    size_t hoff = 0;
    std::memcpy(header.data() + hoff, &slots, sizeof(slots)); hoff += sizeof(slots);
    std::memcpy(header.data() + hoff, &stride, sizeof(stride)); hoff += sizeof(stride);
    std::memcpy(header.data() + hoff, &elem_size, sizeof(elem_size)); hoff += sizeof(elem_size);
    std::memcpy(header.data() + hoff, &type_id, sizeof(type_id)); hoff += sizeof(type_id);
    uint32_t pad = 0; std::memcpy(header.data() + hoff, &pad, sizeof(pad)); hoff += sizeof(pad);
    std::memcpy(header.data() + hoff, &write_seq, sizeof(write_seq)); hoff += sizeof(write_seq);
    // slot_seq under lock
    {
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        for (uint64_t i = 0; i < slots; ++i) {
            uint64_t s = impl->slot_seq[i].load(std::memory_order_acquire);
            std::memcpy(header.data() + hoff, &s, sizeof(s)); hoff += sizeof(s);
        }
    }

    if (!writer(user, header.data(), header.size(), storage_bytes == 0 ? 1 : 0)) return 0;

    // If no storage bytes, we're done
    if (storage_bytes == 0) return 1;

    // Stream storage bytes in chunks using backend copy/map
    size_t remaining = storage_bytes;
    size_t off = 0;
    std::vector<uint8_t> tmp;
    size_t max_chunk = std::min<size_t>(chunk_size, 4 * 1024 * 1024);
    try { tmp.resize(max_chunk); } catch (...) { return 0; }

    if (impl->storage_handle) {
        const gp_mem_backend_vtable_t* svt = gp_mem_backend_get_vtable(impl->storage_handle);
        while (remaining > 0) {
            size_t cur = std::min<size_t>(remaining, tmp.size());
            bool ok_read = false;
            if (svt && svt->copy_from_backend) {
                ok_read = svt->copy_from_backend(impl->storage_handle, off, tmp.data(), cur) != 0;
            }
            if (!ok_read) {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return 0;
                std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + off, cur);
                gp_mem_backend_unmap(impl->storage_handle);
                ok_read = true;
            }
            if (!ok_read) return 0;
            remaining -= cur;
            off += cur;
            int is_final = remaining == 0 ? 1 : 0;
            if (!writer(user, tmp.data(), cur, is_final)) return 0;
        }
    } else {
        // no storage handle: send zeroed chunks
        std::vector<uint8_t> zeros;
        try { zeros.resize(tmp.size()); } catch (...) { return 0; }
        while (remaining > 0) {
            size_t cur = std::min<size_t>(remaining, zeros.size());
            int is_final = (remaining - cur) == 0 ? 1 : 0;
            if (!writer(user, zeros.data(), cur, is_final)) return 0;
            remaining -= cur;
        }
    }

    return 1;
}

// Deserialize a snapshot produced by gp_table_edge_serialize_snapshot and restore
// it into the target edge FIFO. The FIFO will be reconfigured if necessary.
extern "C" int32_t gp_table_edge_deserialize_snapshot(GP_TableContext* ctx, int32_t edge_idx, const void* buf, size_t buf_len) {
    if (!ctx || !buf) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx->edge_fifos.size())) return 0;
    auto &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!fifo.impl) return 0;
    auto impl = fifo.impl.get();
    const uint8_t* src = reinterpret_cast<const uint8_t*>(buf);
    size_t off = 0;
    if (buf_len < sizeof(uint64_t)*3 + sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint64_t)) return 0;
    uint64_t slots = 0; std::memcpy(&slots, src + off, sizeof(slots)); off += sizeof(slots);
    uint64_t stride = 0; std::memcpy(&stride, src + off, sizeof(stride)); off += sizeof(stride);
    uint64_t elem_size = 0; std::memcpy(&elem_size, src + off, sizeof(elem_size)); off += sizeof(elem_size);
    int32_t type_id = 0; std::memcpy(&type_id, src + off, sizeof(type_id)); off += sizeof(type_id);
    off += sizeof(uint32_t); // pad
    uint64_t write_seq = 0; std::memcpy(&write_seq, src + off, sizeof(write_seq)); off += sizeof(write_seq);
    size_t slot_seq_sz = static_cast<size_t>(slots) * sizeof(uint64_t);
    size_t storage_bytes = static_cast<size_t>(stride * slots * elem_size);
    size_t need = sizeof(uint64_t)*3 + sizeof(int32_t) + sizeof(uint32_t) + sizeof(uint64_t) + slot_seq_sz + storage_bytes;
    if (buf_len < need) return 0;

    // If config differs, reconfigure the fifo to match snapshot
    {
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        bool reconfig = false;
        if (impl->slots != static_cast<size_t>(slots) || impl->stride != static_cast<size_t>(stride) || impl->elem_size != static_cast<size_t>(elem_size) || impl->type_id != type_id) {
            fifo.configure(fifo.shape, static_cast<size_t>(slots), static_cast<size_t>(impl->top_k), static_cast<size_t>(elem_size), type_id);
            impl = fifo.impl.get(); // refresh pointer
        }

        // copy slot_seq
        for (uint64_t i = 0; i < slots; ++i) {
            uint64_t s = 0; std::memcpy(&s, src + off, sizeof(s)); off += sizeof(s);
            if (i < impl->slots) impl->slot_seq[i].store(s, std::memory_order_release);
        }

        // write storage bytes into impl->storage_handle
        if (impl->storage_handle) {
            const gp_mem_backend_vtable_t* svt = gp_mem_backend_get_vtable(impl->storage_handle);
            bool ok = false;
            if (svt && svt->copy_to_backend) {
                if (svt->copy_to_backend(impl->storage_handle, 0, src + off, storage_bytes)) ok = true;
            }
            if (!ok) {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return 0;
                std::memcpy(m, src + off, storage_bytes);
                gp_mem_backend_unmap(impl->storage_handle);
            }
            off += storage_bytes;
        } else {
            off += storage_bytes; // skip
        }

        // restore write_seq
        impl->write_seq.store(write_seq, std::memory_order_release);
        impl->last_write_seq.store(write_seq, std::memory_order_release);
        impl->cv.notify_all();
    }

    return 1;
}

// Versioned quiescent transaction snapshot. The byte format and state restore
// live with EdgeTensorFifo so table and runtime-only ABIs share one authority.
extern "C" int32_t gp_table_edge_transaction_snapshot_size(
    GP_TableContext* ctx, int32_t edge_idx, size_t* out_size
) {
    if (!ctx || !out_size) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    const size_t size = nodus::runtime::edge_transaction_snapshot_size(
        ctx->edge_fifos[static_cast<size_t>(edge_idx)]
    );
    if (size == 0) return 0;
    *out_size = size;
    return 1;
}

extern "C" int32_t gp_table_edge_transaction_snapshot_fill(
    GP_TableContext* ctx, int32_t edge_idx, void* out_buf, size_t out_len
) {
    if (!ctx || edge_idx < 0 ||
        edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    return nodus::runtime::edge_transaction_snapshot_fill(
        ctx->edge_fifos[static_cast<size_t>(edge_idx)], out_buf, out_len
    ) ? 1 : 0;
}

extern "C" int32_t gp_table_edge_transaction_snapshot_restore(
    GP_TableContext* ctx, int32_t edge_idx, const void* buf, size_t buf_len
) {
    if (!ctx || edge_idx < 0 ||
        edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    auto& fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!nodus::runtime::edge_transaction_snapshot_restore(fifo, buf, buf_len)) {
        return 0;
    }

    // The FIFO is authoritative. Keep the legacy coordinator mirror current
    // for table scheduling/diagnostics after the exact state is restored.
    ThreadManager* tm = ThreadManager::global();
    if (tm && static_cast<size_t>(edge_idx) < ctx->edge_subscriber_slots.size()) {
        auto& slots_by_key = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i) {
            const uint64_t key = fifo.impl->readers[i].key.load(std::memory_order_acquire);
            if (key == 0) continue;
            const auto found = slots_by_key.find(key);
            if (found != slots_by_key.end() && found->second > 0) {
                tm->update_reader_seq(
                    found->second,
                    fifo.impl->readers[i].seq.load(std::memory_order_acquire)
                );
            }
        }
    }
    return 1;
}
int32_t gp_table_add_edge(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    ctx->edges.emplace_back(a, b);
    // create and record a persistent unique id for this edge
    uint64_t uid = 0ull;
    uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
    ctx->edge_ids.push_back(uid);
    // ensure relax arrays stay in sync
    // start unrelaxed so the cable animates into place
    ctx->relax_value.push_back(0.0f);
    ctx->relax_vel.push_back(0.0f);
    // companion tensor FIFO for this edge
    EdgeTensorFifo fifo;
    fifo.configure_default();
    ctx->edge_fifos.push_back(std::move(fifo));
    ctx->edge_sparse_baseline.emplace_back();
    // reset prospective state when a real edge is added
    ctx->prospective_initialized = false;
    // create a rope entry in the simulator (if available)
    if (ctx->rope_sim) {
        // compute approximate endpoints in table-local coords
        int ax = 0, ay = 0, bx = 0, by = 0;
        std::vector<int> row_y0;
        std::vector<int> row_h;
        compute_row_layout(ctx->rows.data(), static_cast<int>(ctx->rows.size()), ctx->st, ctx->geom.height_px, row_y0, row_h);
        auto compute_center_local = [&](uint64_t key, int &outx, int &outy) {
            outx = -1; outy = -1;
            uint32_t r_orig = static_cast<uint32_t>(key >> 32);
            uint32_t c_idx = static_cast<uint32_t>((key >> 16) & 0xFFFFu);
            uint32_t led = static_cast<uint32_t>(key & 0xFFFFu);
            if (r_orig >= ctx->rows.size()) return;
            const GP_TableRow &row = ctx->rows[static_cast<size_t>(r_orig)];
            if (static_cast<int>(c_idx) < 0 || static_cast<int>(c_idx) >= row.cell_count) return;
            int col_x0[8] = {0}; int col_w[8] = {0};
            compute_columns(ctx->cols.data(), static_cast<int>(ctx->cols.size()), ctx->st.w, ctx->st.name_w, col_x0, col_w);
            const GP_TableCell &cell = row.cells[static_cast<int>(c_idx)];
            int x0 = col_x0[static_cast<int>(c_idx)];
            int cw = col_w[static_cast<int>(c_idx)];
            int y0 = (r_orig < row_y0.size()) ? row_y0[static_cast<size_t>(r_orig)] : (static_cast<int>(r_orig) * ctx->st.row_h);
            int rh = (r_orig < row_h.size()) ? row_h[static_cast<size_t>(r_orig)] : ctx->st.row_h;
            int led_count = 9;
            if (cell.kind == GP_TABLE_CELL_LEDS_ARG) {
                int count = std::max(0, std::min(32, static_cast<int>(cell.value)));
                if (count == 0) count = (cell.flags & 0xFF);
                if (count == 0) count = 12;
                led_count = count;
            } else if (cell.kind == GP_TABLE_CELL_LEDS_TABLE) {
                led_count = 8;
            }
            int eff_w = std::max(1, cw - 4);
            int radius = 4;
            int led_spacing = std::max(radius * 2 + 2, eff_w / std::max(1, led_count + 1));
            int cx0 = x0 + 2 + led_spacing;
            if (static_cast<int>(led) >= 0 && static_cast<int>(led) < led_count) {
                outx = cx0 + static_cast<int>(led) * led_spacing;
                const bool is_image_row = row_find_image_cell(row, nullptr);
                int band_h = is_image_row ? std::min(rh, row_image_header_h(ctx->st, row)) : rh;
                outy = y0 + band_h / 2;
            }
        };
        compute_center_local(a, ax, ay);
        compute_center_local(b, bx, by);
        if (ax == bx && ay == by) {
            // Guard against degenerate endpoints: derive a simple offset from contact ids.
            auto decode = [](uint64_t key, int &row, int &col, int &led) {
                row = static_cast<int>(static_cast<uint32_t>(key >> 32));
                col = static_cast<int>((static_cast<uint32_t>(key >> 16)) & 0xFFFFu);
                led = static_cast<int>(static_cast<uint32_t>(key & 0xFFFFu));
            };
            int ra = 0, ca = 0, la = 0;
            int rb = 0, cb = 0, lb = 0;
            decode(a, ra, ca, la);
            decode(b, rb, cb, lb);
            int row_h = ctx->st.row_h > 0 ? ctx->st.row_h : 22;
            int dx = (cb - ca) * 16 + (lb - la) * 8;
            int dy = (rb - ra) * std::max(10, row_h / 2);
            if (dx == 0 && dy == 0) dx = 24; // last resort nudge
            bx += dx;
            by += dy;
            printf("gp_table_add_edge: adjusted degenerate endpoints a_row=%d b_row=%d dx=%d dy=%d -> (%d,%d)->(%d,%d)\n",
                ra, rb, dx, dy, ax, ay, bx, by);
        }
        int segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
        float slack = 0.0f;
        float plug_z = -ctx->st.cable_plug_depth;
        int idx = rope_sim_add_rope3(ctx->rope_sim, static_cast<float>(ax), static_cast<float>(ay), plug_z, static_cast<float>(bx), static_cast<float>(by), plug_z, segs, slack);
        uint64_t rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
        ctx->rope_ids.push_back(rid);
        printf("gp_table_add_edge: ctx=%p rope_ids_count=%zu after push id=%llu\n", (void*)ctx, ctx->rope_ids.size(), (unsigned long long)rid);
        ctx->rope_id_to_sim_idx[rid] = idx;
        printf("gp_table_add_edge: ctx=%p added rope idx=%d id=%llu\n", (void*)ctx, idx, (unsigned long long)rid);
        {
            GP_CanvasContext* cvs = gp_canvas_get_singleton();
            if (cvs) {
                uint64_t tmp = rid;
                gp_canvas_register_table_rope_ids_from_array(cvs, ctx, &tmp, 1);
            }
        }
    } else {
        uint64_t rid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
        ctx->rope_ids.push_back(rid);
        printf("gp_table_add_edge: ctx=%p queued edge id=%llu (no sim)\n", (void*)ctx, (unsigned long long)rid);
        {
            GP_CanvasContext* cvs = gp_canvas_get_singleton();
            if (cvs) {
                uint64_t tmp = rid;
                gp_canvas_register_table_rope_ids_from_array(cvs, ctx, &tmp, 1);
            }
        }
    }
    sync_edge_tensor_for_idx(ctx, ctx->edges.size() - 1);
    int32_t edge_idx = static_cast<int32_t>(ctx->edges.size() - 1);
    return 1;
}

int32_t gp_table_clear_edges(GP_TableContext* ctx) {
    if (!ctx) return 0;
    ctx->edges.clear();
    ctx->rings.clear();
    ctx->edge_ids.clear();
    ctx->next_edge_id = 1;
    ctx->relax_value.clear();
    ctx->relax_vel.clear();
    ctx->edge_fifos.clear();
    ctx->edge_batch_metadata.clear();
    ctx->edge_subgroup_flags.clear();
    ctx->edge_subscriber_slots.clear();
    ctx->edge_sparse_baseline.clear();
    // reset rope simulator mapping and recreate sim to free resources
    ctx->rope_id_to_sim_idx.clear();
    if (!ctx->rope_ids.empty()) printf("gp_table_clear_edges: ctx=%p clearing %zu rope_ids\n", (void*)ctx, ctx->rope_ids.size());
    ctx->rope_ids.clear();
    ctx->next_rope_id = 1;
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        int max_ropes = 1024;
        int max_segs = (ctx->debug_flags & GP_CANVAS_DEBUG_SEGMENTS_1) ? 1 : std::max(2, ctx->st.cable_segments);
        ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    } else {
        // if rope_sim is external or null, leave it alone; indices already cleared
    }
    return 1;
}

extern "C" int32_t gp_table_register_ring_edge(GP_TableContext* ctx, int32_t ring_id, unsigned long long ring_key) {
    if (!ctx) return -1;
    GP_TableContext::RingEntry re;
    re.ring_id = ring_id;
    re.key = ring_key;
    re.fifo.configure_default();
    re.uid = gp_canvas_generate_id(gp_canvas_get_singleton(), 0ull);
    ctx->rings.push_back(std::move(re));
    return static_cast<int32_t>(ctx->rings.size() - 1);
}

extern "C" int32_t gp_table_unregister_ring_edge(GP_TableContext* ctx, int32_t ring_entry_idx) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    ctx->rings.erase(ctx->rings.begin() + ring_entry_idx);
    return 1;
}

extern "C" int32_t gp_table_get_ring_edge_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->rings.size());
}

extern "C" int32_t gp_table_get_ring_edge(const GP_TableContext* ctx, int32_t idx, int32_t* out_ring_id, unsigned long long* out_key) {
    if (!ctx || !out_ring_id || !out_key) return 0;
    if (idx < 0 || idx >= static_cast<int>(ctx->rings.size())) return 0;
    const auto &re = ctx->rings[static_cast<size_t>(idx)];
    *out_ring_id = re.ring_id;
    *out_key = re.key;
    return 1;
}

extern "C" int32_t gp_table_ring_set_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t flags) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    ctx->rings[static_cast<size_t>(ring_entry_idx)].subgroup_flags = flags;
    return 1;
}

extern "C" int32_t gp_table_ring_get_subgroup_flags(GP_TableContext* ctx, int32_t ring_entry_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    *out_flags = ctx->rings[static_cast<size_t>(ring_entry_idx)].subgroup_flags;
    return 1;
}

extern "C" int32_t gp_table_ring_set_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, const GP_TableEdgeTensorSpecTyped* spec) {
    if (!ctx || !spec) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    // Mirror gp_table_edge_set_tensor_spec behavior: disallow reconfigure after writes
    if (re.fifo.impl && re.fifo.impl->write_seq.load(std::memory_order_relaxed) != 0) return 0;
    std::vector<int32_t> dims;
    int dc = std::max(0, std::min(8, spec->dim_count));
    dims.reserve(static_cast<size_t>(dc));
    for (int i = 0; i < dc; ++i) {
        int32_t d = spec->dims[i];
        if (d < 1) d = 1;
        dims.push_back(d);
    }
    size_t slots = spec->slots > 0 ? static_cast<size_t>(spec->slots) : size_t(1);
    size_t topk = spec->top_k > 0 ? static_cast<size_t>(spec->top_k) : size_t(0);
    size_t elem_size = spec->elem_size > 0 ? static_cast<size_t>(spec->elem_size) : sizeof(float);
    int32_t type_id = spec->type_id;
    re.fifo.configure(dims, slots, topk, elem_size, type_id);
    re.batch_metadata = GP_TableEdgeBatchMetadata();
    return 1;
}

extern "C" int32_t gp_table_ring_get_tensor_spec(GP_TableContext* ctx, int32_t ring_entry_idx, GP_TableEdgeTensorSpecTyped* out_spec) {
    if (!ctx || !out_spec) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    *out_spec = edge_fifo_to_table_spec(
        ctx->rings[static_cast<size_t>(ring_entry_idx)].fifo
    );
    return 1;
}

extern "C" int32_t gp_table_ring_subscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    return gp_table_ring_subscribe_ex(ctx, ring_entry_idx, subscriber_key, /*start_at_head=*/1);
}

extern "C" int32_t gp_table_ring_subscribe_ex(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    bool ok = re.fifo.subscribe(subscriber_key, start_at_head != 0);
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) >= ctx->ring_subscriber_slots.size()) ctx->ring_subscriber_slots.resize(ctx->rings.size());
        auto &map = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
        if (map.find(subscriber_key) == map.end()) {
            uint64_t rid = re.uid;
            int slot = tm->register_reader_for_edge(rid);
            if (slot > 0) map[subscriber_key] = slot;
            auto *r = re.fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

extern "C" int32_t gp_table_ring_unsubscribe(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    re.fifo.unsubscribe(subscriber_key);
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) < ctx->ring_subscriber_slots.size()) {
            auto &m = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
            auto it = m.find(subscriber_key);
            if (it != m.end()) {
                int slot = it->second;
                if (slot > 0) tm->unregister_reader_slot(slot);
                m.erase(it);
            }
        }
    }
    return 1;
}

extern "C" int32_t gp_table_ring_publish(GP_TableContext* ctx, int32_t ring_entry_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample_bytes || sample_len_bytes < 0) return 0;
    if (ring_entry_idx < 0 || ring_entry_idx >= static_cast<int>(ctx->rings.size())) return 0;
    auto &re = ctx->rings[static_cast<size_t>(ring_entry_idx)];
    bool dropped = false;
    uint64_t rid = re.uid;
    uint32_t flags = re.subgroup_flags;
    bool ok = false;
    if (fifo_effective_byref(re.fifo, flags)) {
        size_t sb = static_cast<size_t>(sample_len_bytes);
        size_t alloc_sz = sizeof(BoxedSample) + sb;
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(alloc_sz));
        if (!buf) { if (out_dropped) *out_dropped = 1; return 0; }
        BoxedSample* box = reinterpret_cast<BoxedSample*>(buf);
        box->magic = BOXED_SAMPLE_MAGIC;
        box->sample_len = sample_len_bytes;
        uint8_t* payload = buf + sizeof(BoxedSample);
        std::memcpy(payload, sample_bytes, sb);
        ok = re.fifo.push_ptr(rid, writer_key, static_cast<void*>(box), &dropped);
    } else {
        ok = re.fifo.push(rid, writer_key, sample_bytes, static_cast<size_t>(sample_len_bytes), &dropped);
    }
    if (out_dropped && dropped) *out_dropped = 1;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        if (static_cast<size_t>(ring_entry_idx) < ctx->ring_subscriber_slots.size()) {
            auto &m = ctx->ring_subscriber_slots[static_cast<size_t>(ring_entry_idx)];
            std::vector<std::pair<uint64_t,int>> subs; subs.reserve(m.size());
            for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
            for (const auto &kv : subs) {
                uint64_t subscriber_key = kv.first; int slot = kv.second;
                if (slot <= 0) continue;
                auto *r = re.fifo.find_reader(subscriber_key);
                if (!r) continue;
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return ok ? 1 : 0;
}

int32_t gp_table_get_edge_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->edges.size());
}

int32_t gp_table_get_edge(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_a, unsigned long long* out_b) {
    if (!ctx || !out_a || !out_b) return 0;
    if (idx < 0 || idx >= static_cast<int32_t>(ctx->edges.size())) return 0;
    *out_a = ctx->edges[static_cast<size_t>(idx)].first;
    *out_b = ctx->edges[static_cast<size_t>(idx)].second;
    return 1;
}

int32_t gp_table_edge_set_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeTensorSpecTyped* spec) {
    if (!ctx || !spec) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    // Spec immutability: disallow reconfigure after any writes have occurred.
    // (Caller may reconfigure only while the edge is quiescent.)
    {
        auto& fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
        if (fifo.impl && fifo.impl->write_seq.load(std::memory_order_relaxed) != 0) {
            return 0;
        }
    }
    std::vector<int32_t> dims;
    int dc = std::max(0, std::min(8, spec->dim_count));
    dims.reserve(static_cast<size_t>(dc));
    for (int i = 0; i < dc; ++i) {
        int32_t d = spec->dims[i];
        if (d < 1) d = 1;
        dims.push_back(d);
    }
    size_t slots = spec->slots > 0 ? static_cast<size_t>(spec->slots) : size_t(1);
    size_t topk = spec->top_k > 0 ? static_cast<size_t>(spec->top_k) : size_t(0);
    size_t elem_size = spec->elem_size > 0 ? static_cast<size_t>(spec->elem_size) : sizeof(float);
    int32_t type_id = spec->type_id;
    nodus::tensors::TensorLayout layout = static_cast<nodus::tensors::TensorLayout>(spec->layout);
    nodus::tensors::TensorDType dtype = static_cast<nodus::tensors::TensorDType>(spec->dtype);
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].configure(dims, slots, topk, elem_size, type_id, layout, dtype);
    sync_edge_tensor_for_idx(ctx, static_cast<size_t>(edge_idx));
    // Invalidate type-id snapshot after edge reconfigure so readers will refresh.
    std::atomic_store(&ctx->type_ids_snapshot, std::shared_ptr<std::vector<int32_t>>(nullptr));
    // If a ThreadManager is present, update registered reader slots with
    // the freshly-initialized sequence (usually zero) so manager state
    // remains consistent after reconfigure.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
        for (const auto &kv : m) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return 1;
}

int32_t gp_table_edge_get_tensor_spec(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeTensorSpecTyped* out_spec) {
    if (!ctx || !out_spec) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    *out_spec = edge_fifo_to_table_spec(
        ctx->edge_fifos[static_cast<size_t>(edge_idx)]
    );
    return 1;
}

int32_t gp_table_edge_subscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    return gp_table_edge_subscribe_ex(ctx, edge_idx, subscriber_key, /*start_at_head=*/1);
}

int32_t gp_table_edge_subscribe_ex(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t start_at_head) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    bool ok = ctx->edge_fifos[static_cast<size_t>(edge_idx)].subscribe(subscriber_key, start_at_head != 0);
    if (!ok) return 0;
    // Register reader slot with ThreadManager global (if available).
    // Avoid re-registering if this subscriber already has a slot mapping.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &map = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        if (map.find(subscriber_key) == map.end()) {
            uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
            int slot = tm->register_reader_for_edge(edge_id);
            if (slot > 0) map[subscriber_key] = slot;
            // Notify manager of the starting sequence for this reader (if available).
            auto *r = ctx->edge_fifos[static_cast<size_t>(edge_idx)].find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_unsubscribe(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].unsubscribe(subscriber_key);
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            if (slot > 0) tm->unregister_reader_slot(slot);
            m.erase(it);
        }
    }
    return 1;
}

int32_t gp_table_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample_bytes || sample_len_bytes < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];

    // Inspect edge subgroup flags to determine publish behavior. We use a
    // simple switch so future policies can be tacked on easily.
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    bool effective_byref = fifo_effective_byref(fifo, flags);
    if (effective_byref) {
        // BYREF: box the raw sample bytes and publish its pointer instead so
        // consumers receive by-reference payloads. Box format: [BoxedSample][raw bytes]
        size_t sb = static_cast<size_t>(sample_len_bytes);
        size_t alloc_sz = sizeof(BoxedSample) + sb;
        uint8_t* buf = static_cast<uint8_t*>(std::malloc(alloc_sz));
        if (!buf) { if (out_dropped) *out_dropped = 1; return 0; }
        BoxedSample* box = reinterpret_cast<BoxedSample*>(buf);
        box->magic = BOXED_SAMPLE_MAGIC;
        box->sample_len = sample_len_bytes;
        uint8_t* payload = buf + sizeof(BoxedSample);
        std::memcpy(payload, sample_bytes, sb);
        ok = fifo.push_ptr(edge_id, writer_key, static_cast<void*>(box), &dropped);
    } else {
        // Normal byte-path
        ok = fifo.push(edge_id, writer_key, sample_bytes, static_cast<size_t>(sample_len_bytes), &dropped);
    }
    if (out_dropped && dropped) *out_dropped = 1;
    // After a publish, the FIFO implementation may have advanced reader sequences
    // (e.g., during top-k trimming). Ensure the ThreadManager has up-to-date
    // per-slot sequences for all registered subscribers on this edge.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        // Snapshot subscriber map to avoid iterator invalidation if
        // concurrent subscribe/unsubscribe mutates the map.
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return ok ? 1 : 0;
}

int32_t gp_table_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped, int32_t timeout_ms) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !sample_bytes || sample_len_bytes < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
    bool ok = fifo.push_blocking(edge_id, writer_key, sample_bytes, static_cast<size_t>(sample_len_bytes), &dropped, timeout_ms);
    if (out_dropped && dropped) *out_dropped = 1;
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return 1;
}

// Publish a typed element by popping it from a RawStackFrame. This wrapper
// forwards to the FIFO implementation which attempts optimized backend
// transfers when possible.
int32_t gp_table_edge_publish_from_frame(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, struct RawStackFrame* src_frame, int32_t type_id, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !src_frame) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    bool ok = fifo.push_from_frame(ctx->edge_ids[static_cast<size_t>(edge_idx)], writer_key, src_frame, type_id, &dropped);
    if (out_dropped && dropped) *out_dropped = 1;
    return ok ? 1 : 0;
}

// Pointer-oriented publish: publish an opaque pointer into the edge FIFO.
int32_t gp_table_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    bool dropped = false;
    uint64_t edge_id = ctx->edge_ids[static_cast<size_t>(edge_idx)];
    bool ok = fifo.push_ptr(edge_id, writer_key, ptr, &dropped);
    if (out_dropped && dropped) *out_dropped = 1;
    // Sync ThreadManager reader sequences similar to float publish.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        std::vector<std::pair<uint64_t,int>> subs;
        subs.reserve(m.size());
        for (const auto &kv : m) subs.emplace_back(kv.first, kv.second);
        for (const auto &kv : subs) {
            uint64_t subscriber_key = kv.first;
            int slot = kv.second;
            if (slot <= 0) continue;
            auto *r = fifo.find_reader(subscriber_key);
            if (!r) continue;
            uint64_t seq = r->seq.load(std::memory_order_relaxed);
            tm->update_reader_seq(slot, seq);
        }
    }
    return ok ? 1 : 0;
}

// Forward declarations for table-specific "many" pointer helpers so the
// generic gp_edge_* wrappers below can call them even though their
// implementations appear later in this file.
int32_t gp_table_edge_consume_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptrs, int32_t max_out);
int32_t gp_table_edge_publish_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void** ptrs, int32_t count, int32_t* out_dropped);

// Generic wrappers to provide a neutral edge API surface. These forward to
// the table-specific implementations so callers outside the table system can
// use a stable `gp_edge_*` API while we evolve internals.
int32_t gp_edge_publish(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped) {
    return gp_table_edge_publish(ctx, edge_idx, writer_key, sample_bytes, sample_len_bytes, out_dropped);
}
int32_t gp_edge_publish_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, const void* sample_bytes, int32_t sample_len_bytes, int32_t* out_dropped, int32_t timeout_ms) {
    return gp_table_edge_publish_blocking(ctx, edge_idx, writer_key, sample_bytes, sample_len_bytes, out_dropped, timeout_ms);
}
int32_t gp_edge_publish_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void* ptr, int32_t* out_dropped) {
    return gp_table_edge_publish_ptr(ctx, edge_idx, writer_key, ptr, out_dropped);
}
int32_t gp_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr) {
    return gp_table_edge_consume_ptr(ctx, edge_idx, subscriber_key, out_ptr);
}

int32_t gp_edge_consume_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptrs, int32_t max_out) {
    return gp_table_edge_consume_ptr_many(ctx, edge_idx, subscriber_key, out_ptrs, max_out);
}

int32_t gp_edge_publish_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void** ptrs, int32_t count, int32_t* out_dropped) {
    return gp_table_edge_publish_ptr_many(ctx, edge_idx, writer_key, ptrs, count, out_dropped);
}

// Forward declarations for many-pointer table-edge helpers. These are
// implemented later in this translation unit but declared here so the
// gp_edge_* wrapper functions above can call them.
int32_t gp_table_edge_consume_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptrs, int32_t max_out);
int32_t gp_table_edge_publish_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void** ptrs, int32_t count, int32_t* out_dropped);

int32_t gp_table_edge_consume_ptr(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptr) {
    if (out_ptr) *out_ptr = nullptr;
    if (!ctx || !out_ptr) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    void* p = nullptr;
    bool ok = fifo.pop_ptr(subscriber_key, &p);
    if (!ok) return 0;
    if (out_ptr) *out_ptr = p;
    // Notify ThreadManager of read advancement
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_consume_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void** out_ptrs, int32_t max_out) {
    if (out_ptrs) {
        for (int i = 0; i < max_out; ++i) out_ptrs[i] = nullptr;
    }
    if (!ctx || !out_ptrs || max_out <= 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    int consumed = 0;
    void* p = nullptr;
    for (int i = 0; i < max_out; ++i) {
        bool ok = fifo.pop_ptr(subscriber_key, &p);
        if (!ok) break;
        out_ptrs[consumed++] = p;
    }
    if (consumed == 0) return 0;
    // Notify ThreadManager of read advancement using reader seq from fifo
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return consumed;
}

int32_t gp_table_edge_publish_ptr_many(GP_TableContext* ctx, int32_t edge_idx, unsigned long long writer_key, void** ptrs, int32_t count, int32_t* out_dropped) {
    if (out_dropped) *out_dropped = 0;
    if (!ctx || !ptrs || count <= 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    int total_dropped = 0;
    for (int i = 0; i < count; ++i) {
        int dropped = 0;
        gp_table_edge_publish_ptr(ctx, edge_idx, writer_key, ptrs[i], &dropped);
        total_dropped += dropped;
    }
    if (out_dropped) *out_dropped = total_dropped;
    return count;
}

int32_t gp_table_edge_consume(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample_bytes || out_len_bytes < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    // If the edge is configured BYREF, the FIFO carries pointers to boxed
    // float samples. In that case, consume via pop_ptr and transparently
    // unbox into the caller-provided float buffer and free the boxed memory.
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    bool effective_byref = flags_imply_byref(flags);
    if (!effective_byref) {
        // check attached backend capability
        if (fifo.impl && fifo.impl->backend) {
            if (gp_mem_backend_supports_byref(fifo.impl->backend)) effective_byref = true;
        }
    }
    if (effective_byref) {
        // Peek first: if there's a pointer and it's a boxed sample, then
        // consume it; if it's a non-boxed pointer (e.g., EventPayload)
        // do not advance the reader here — let pointer-oriented APIs
        // (`gp_table_edge_consume_ptr`) handle it.
        void* ppeek = nullptr;
        if (fifo.peek_ptr(subscriber_key, &ppeek)) {
            if (!ppeek) return 0;
            BoxedSample* boxpeek = reinterpret_cast<BoxedSample*>(ppeek);
            if (boxpeek->magic == BOXED_SAMPLE_MAGIC) {
                // Now actually pop the pointer and unbox.
                void* p = nullptr;
                if (!fifo.pop_ptr(subscriber_key, &p)) return 0;
                if (!p) return 0;
                BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
                int need_bytes = box->sample_len;
                if (out_len_bytes < need_bytes) {
                    std::free(box);
                    return 0;
                }
                uint8_t* payload = reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample);
                std::memcpy(out_sample_bytes, payload, static_cast<size_t>(need_bytes));
                std::free(box);
                wrote = static_cast<size_t>(need_bytes);
                ok = true;
            } else {
                // A raw pointer was delivered; do not consume here.
                return 0;
            }
        } else {
            // No pointer available — try float path.
            ok = fifo.pop(subscriber_key, out_sample_bytes, static_cast<size_t>(out_len_bytes), wrote);
        }
    } else {
        ok = fifo.pop(subscriber_key, out_sample_bytes, static_cast<size_t>(out_len_bytes), wrote);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    // Notify ThreadManager of reader advancement, if registered.
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

int32_t gp_table_edge_peek(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample_bytes || out_len_bytes < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (fifo_effective_byref(fifo, flags)) {
        void* p = nullptr;
        if (!fifo.peek_ptr(subscriber_key, &p)) return 0;
        if (!p) return 0;
        BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
        if (box->magic != BOXED_SAMPLE_MAGIC) return 0;
        int need_bytes = box->sample_len;
        if (out_len_bytes < need_bytes) return 0;
        uint8_t* payload = reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample);
        std::memcpy(out_sample_bytes, payload, static_cast<size_t>(need_bytes));
        wrote = static_cast<size_t>(need_bytes);
        ok = true;
    } else {
        ok = fifo.peek(subscriber_key, out_sample_bytes, static_cast<size_t>(out_len_bytes), wrote);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    // Do NOT notify ThreadManager because we didn't advance the reader.
    return 1;
}

int32_t gp_table_edge_consume_blocking(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, void* out_sample_bytes, int32_t out_len_bytes, int32_t* out_written, int32_t timeout_ms) {
    if (out_written) *out_written = 0;
    if (!ctx || !out_sample_bytes || out_len_bytes < 0) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    size_t wrote = 0;
    uint32_t flags = 0;
    if (static_cast<size_t>(edge_idx) < ctx->edge_subgroup_flags.size()) flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    bool ok = false;
    if (flags_imply_byref(flags)) {
        // Wait for either a boxed pointer or a float sample. We'll loop until
        // the timeout expires. To avoid busy-waiting we sleep in short
        // intervals while checking for availability.
        using clock = std::chrono::steady_clock;
        auto start = clock::now();
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (start + std::chrono::milliseconds(timeout_ms));
        while (true) {
            void* ppeek = nullptr;
            if (fifo.peek_ptr(subscriber_key, &ppeek)) {
                if (!ppeek) return 0;
                BoxedSample* boxpeek = reinterpret_cast<BoxedSample*>(ppeek);
                if (boxpeek->magic == BOXED_SAMPLE_MAGIC) {
                    // consume boxed sample
                    void* p = nullptr;
                    if (!fifo.pop_ptr(subscriber_key, &p)) return 0;
                    if (!p) return 0;
                    BoxedSample* box = reinterpret_cast<BoxedSample*>(p);
                    int need_bytes = box->sample_len;
                    if (out_len_bytes < need_bytes) { std::free(box); return 0; }
                    uint8_t* payload = reinterpret_cast<uint8_t*>(box) + sizeof(BoxedSample);
                    std::memcpy(out_sample_bytes, payload, static_cast<size_t>(need_bytes));
                    std::free(box);
                    wrote = static_cast<size_t>(need_bytes);
                    ok = true;
                    break;
                } else {
                    // raw pointer delivered; don't consume here
                    return 0;
                }
            }
            // Try to pop a float sample with the remaining timeout
            if (timeout_ms == 0) break;
            auto now = clock::now();
            if (now >= deadline) break;
            int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
            if (remaining_ms <= 0) break;
            if (fifo.pop_blocking(subscriber_key, out_sample_bytes, static_cast<size_t>(out_len_bytes), wrote, remaining_ms)) {
                ok = true;
                break;
            }
            // brief sleep to yield CPU before re-checking
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } else {
        ok = fifo.pop_blocking(subscriber_key, out_sample_bytes, static_cast<size_t>(out_len_bytes), wrote, timeout_ms);
    }
    if (out_written) *out_written = static_cast<int32_t>(wrote);
    if (!ok) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        auto it = m.find(subscriber_key);
        if (it != m.end()) {
            int slot = it->second;
            auto *r = fifo.find_reader(subscriber_key);
            if (r && slot > 0) {
                uint64_t seq = r->seq.load(std::memory_order_relaxed);
                tm->update_reader_seq(slot, seq);
            }
        }
    }
    return 1;
}

// Attach or query a memory backend for an edge FIFO.
int32_t gp_table_edge_set_backend(GP_TableContext* ctx, int32_t edge_idx, gp_mem_backend_handle_t h) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    fifo.set_backend(h);
    return 1;
}

extern "C" int32_t gp_table_edge_set_tensor_storage(GP_TableContext* ctx,
                                                    int32_t edge_idx,
                                                    nodus::tensors::AbstractTensorHandle handle,
                                                    nodus::tensors::TensorBackend* backend) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    fifo.set_tensor_storage(handle, backend);
    return 1;
}

// Swap the backend attached to an edge FIFO at runtime. If `copy_over` is
// non-zero, attempt to copy existing FIFO storage from the old backend (or
// host storage) into the new backend. Returns 1 on success, 0 on failure.
extern "C" int32_t gp_table_edge_swap_backend(GP_TableContext* ctx, int32_t edge_idx, gp_mem_backend_handle_t new_h, int32_t copy_over) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!fifo.impl) { fifo.set_backend(new_h); return 1; }
    gp_mem_backend_handle_t old_h = fifo.get_backend();
    if (old_h == new_h) return 1; // fast-path

    if (!copy_over) {
        fifo.set_backend(new_h);
        return 1;
    }

    // Compute total storage size for the FIFO (host layout)
    size_t total_bytes = fifo.impl->stride * fifo.impl->slots * fifo.impl->elem_size;
    if (total_bytes == 0) { fifo.set_backend(new_h); return 1; }

    std::vector<uint8_t> tmp;
    try { tmp.resize(total_bytes); } catch (...) { return 0; }

    // Read from old backend if possible
    bool read_ok = false;
    if (old_h) {
        const gp_mem_backend_vtable_t* old_vt = gp_mem_backend_get_vtable(old_h);
        if (old_vt && old_vt->copy_from_backend) {
            if (old_vt->copy_from_backend(old_h, 0, tmp.data(), total_bytes)) read_ok = true;
        }
        if (!read_ok) {
            void* m = gp_mem_backend_map_or_null(old_h);
            if (m) {
                std::memcpy(tmp.data(), m, total_bytes);
                gp_mem_backend_unmap(old_h);
                read_ok = true;
            }
        }
    }
    // Fallback to FIFO's internal storage handle
    if (!read_ok && fifo.impl->storage_handle) {
        const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(fifo.impl->storage_handle);
        if (sh_vt && sh_vt->copy_from_backend) {
            if (sh_vt->copy_from_backend(fifo.impl->storage_handle, 0, tmp.data(), total_bytes)) read_ok = true;
        }
        if (!read_ok) {
            void* m = gp_mem_backend_map_or_null(fifo.impl->storage_handle);
            if (m) {
                std::memcpy(tmp.data(), m, total_bytes);
                gp_mem_backend_unmap(fifo.impl->storage_handle);
                read_ok = true;
            }
        }
    }
    if (!read_ok) return 0;

    // Attach new backend and attempt to write into it
    fifo.set_backend(new_h);
    if (new_h) {
        const gp_mem_backend_vtable_t* new_vt = gp_mem_backend_get_vtable(new_h);
        bool write_ok = false;
        if (new_vt && new_vt->copy_to_backend) {
            if (new_vt->copy_to_backend(new_h, 0, tmp.data(), total_bytes)) write_ok = true;
        }
        if (!write_ok) {
            // Try mapping
            void* m = gp_mem_backend_map_or_null(new_h);
            if (m) {
                std::memcpy(m, tmp.data(), total_bytes);
                gp_mem_backend_unmap(new_h);
                write_ok = true;
            }
        }
        if (!write_ok) {
            // Could not transfer into new backend; revert backend pointer and fail.
            fifo.set_backend(old_h);
            return 0;
        }
    }

    // Keep the FIFO internal storage in sync with the copied buffer.
    if (fifo.impl->storage_handle) {
        const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(fifo.impl->storage_handle);
        if (sh_vt && sh_vt->copy_to_backend) {
            sh_vt->copy_to_backend(fifo.impl->storage_handle, 0, tmp.data(), total_bytes);
        } else {
            void* m = gp_mem_backend_map_or_null(fifo.impl->storage_handle);
            if (m) {
                std::memcpy(m, tmp.data(), total_bytes);
                gp_mem_backend_unmap(fifo.impl->storage_handle);
            }
        }
    }
    return 1;
}

gp_mem_backend_handle_t gp_table_edge_get_backend(GP_TableContext* ctx, int32_t edge_idx) {
    if (!ctx) return nullptr;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return nullptr;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    return fifo.get_backend();
}

int32_t gp_table_edge_unread(GP_TableContext* ctx, int32_t edge_idx, unsigned long long subscriber_key, int32_t* out_count) {
    if (!ctx || !out_count) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    size_t cnt = ctx->edge_fifos[static_cast<size_t>(edge_idx)].unread(subscriber_key);
    *out_count = static_cast<int32_t>(std::min<size_t>(cnt, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
    return 1;
}

int32_t gp_table_edge_set_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, const GP_TableEdgeBatchMetadata* metadata) {
    if (!ctx || !metadata) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_batch_metadata.size())) return 0;
    ctx->edge_batch_metadata[static_cast<size_t>(edge_idx)] = *metadata;
    return 1;
}

int32_t gp_table_edge_get_batch_metadata(GP_TableContext* ctx, int32_t edge_idx, GP_TableEdgeBatchMetadata* out_metadata) {
    if (!ctx || !out_metadata) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_batch_metadata.size())) return 0;
    *out_metadata = ctx->edge_batch_metadata[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_edge_set_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t flags) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_subgroup_flags.size())) return 0;
    ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)] = flags;
    return 1;
}

extern "C" int32_t gp_table_get_edge_rope_index(const GP_TableContext* ctx, int32_t edge_idx, int32_t* out_rope_idx) {
    if (!ctx || !out_rope_idx) return 0;
    if (edge_idx < 0 || edge_idx >= static_cast<int>(ctx->rope_ids.size())) { *out_rope_idx = -1; return 0; }
    uint64_t id = ctx->rope_ids[static_cast<size_t>(edge_idx)];
    if (id == 0ull) { *out_rope_idx = -1; return 1; }
    auto it = ctx->rope_id_to_sim_idx.find(id);
    if (it == ctx->rope_id_to_sim_idx.end()) { *out_rope_idx = -1; return 1; }
    *out_rope_idx = it->second;
    return 1;
}

int32_t gp_table_edge_get_subgroup_flags(GP_TableContext* ctx, int32_t edge_idx, uint32_t* out_flags) {
    if (!ctx || !out_flags) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_subgroup_flags.size())) return 0;
    *out_flags = ctx->edge_subgroup_flags[static_cast<size_t>(edge_idx)];
    return 1;
}

int32_t gp_table_edge_index_for_key(GP_TableContext* ctx, unsigned long long led_key, int32_t* out_edge_idx) {
    if (!ctx || !out_edge_idx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto& e = ctx->edges[i];
        if (e.first == led_key || e.second == led_key) {
            *out_edge_idx = static_cast<int32_t>(i);
            return 1;
        }
    }
    return 0;
}

// Resolve a persistent rope id to the attached RopeSim index for this table.
// Returns -1 if not present.
extern "C" int gp_table_resolve_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id) {
    if (!ctx) return -1;
    if (id == 0ull) return -1;
    auto it = ctx->rope_id_to_sim_idx.find(id);
    if (it == ctx->rope_id_to_sim_idx.end()) return -1;
    return it->second;
}

// Bind a persistent rope id to a specific RopeSim index. Useful when a caller
// creates a rope externally and wants the table to resolve it later.
extern "C" int gp_table_bind_rope_id_to_sim_index(GP_TableContext* ctx, uint64_t id, int32_t rope_idx) {
    if (!ctx || id == 0ull || rope_idx < 0) return 0;
    ctx->rope_id_to_sim_idx[id] = rope_idx;
    if (static_cast<size_t>(rope_idx) >= ctx->rope_ids.size()) {
        ctx->rope_ids.resize(static_cast<size_t>(rope_idx) + 1, 0ull);
    }
    if (ctx->rope_ids[static_cast<size_t>(rope_idx)] == 0ull) {
        ctx->rope_ids[static_cast<size_t>(rope_idx)] = id;
    }
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
    }
    // Diagnostic: report binding activity and associated sim pointer
    {
        RopeSim* rs = gp_table_get_rope_sim(ctx);
        printf("gp_table_bind_rope_id_to_sim_index: ctx=%p id=%llu rope_idx=%d sim=%p\n", (void*)ctx, (unsigned long long)id, (int)rope_idx, (void*)rs);
    }
    return 1;
}

    // Allow external code (canvas) to set the table's persistent rope id list
    // prior to serialization so exported blobs include canonical ids.
extern "C" int gp_table_set_rope_ids_from_array(GP_TableContext* ctx, const uint64_t* ids, int count) {
    if (!ctx) return 0;
    if (!ids || count <= 0) {
        ctx->rope_ids.clear();
        if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
            gp_canvas_mark_rope_map_dirty(cvs);
        }
        return 1;
    }
    ctx->rope_ids.assign(ids, ids + count);
    // rebuild the lookup map so callers can resolve sim indices from the
    // provided persistent ids.
    ctx->rope_id_to_sim_idx.clear();
    for (int i = 0; i < count; ++i) {
        uint64_t uid = ids[static_cast<size_t>(i)];
        if (uid != 0ull) ctx->rope_id_to_sim_idx[uid] = i;
    }
    // set next_rope_id to one past maximum to avoid collisions
    uint64_t mx = 1;
    for (auto v : ctx->rope_ids) if (v >= mx) mx = v + 1;
    ctx->next_rope_id = mx;
    // Diagnostic: report table->rope_sim and a few sample id->index->vert counts
    {
        RopeSim* rs = gp_table_get_rope_sim(ctx);
        printf("gp_table_set_rope_ids_from_array: ctx=%p sim=%p count=%d\n", (void*)ctx, (void*)rs, count);
        int sample = std::min(count, 5);
        for (int i = 0; i < sample; ++i) {
            uint64_t uid = ids[static_cast<size_t>(i)];
            int ridx = gp_table_resolve_rope_id_to_sim_index(ctx, uid);
            int vc = -1;
            if (rs && ridx >= 0) vc = rope_sim_get_vertex_count(rs, ridx);
            printf("  id[%d]=%llu -> idx=%d verts=%d\n", i, (unsigned long long)uid, ridx, vc);
        }
    }
    if (GP_CanvasContext* cvs = gp_canvas_get_singleton()) {
        gp_canvas_mark_rope_map_dirty(cvs);
    }
    return 1;
    }

int32_t gp_table_edge_index_for_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b, int32_t* out_edge_idx) {
    if (!ctx || !out_edge_idx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto& e = ctx->edges[i];
        if (e.first == a && e.second == b) {
            *out_edge_idx = static_cast<int32_t>(i);
            return 1;
        }
    }
    return 0;
}

int32_t gp_table_edge_set_delta_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t delta_mode) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_delta_mode(delta_mode != 0);
    return 1;
}

int32_t gp_table_edge_set_delta_sparse_mode(GP_TableContext* ctx,
                                            int32_t edge_idx,
                                            int32_t delta_sparse_mode,
                                            int32_t accumulate_mode,
                                            float threshold) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    if (static_cast<size_t>(edge_idx) >= ctx->edge_sparse_baseline.size()) ctx->edge_sparse_baseline.resize(ctx->edge_fifos.size());
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_delta_sparse_mode(delta_sparse_mode != 0, accumulate_mode != 0, threshold);
    if (!delta_sparse_mode && static_cast<size_t>(edge_idx) < ctx->edge_sparse_baseline.size()) {
        ctx->edge_sparse_baseline[static_cast<size_t>(edge_idx)].clear();
    }
    return 1;
}

int32_t gp_table_edge_consume_sparse(GP_TableContext* ctx,
                                     int32_t edge_idx,
                                     unsigned long long subscriber_key,
                                     void** out_sparse) {
    if (!ctx || !out_sparse) return 0;
    *out_sparse = nullptr;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    EdgeTensorFifo &fifo = ctx->edge_fifos[static_cast<size_t>(edge_idx)];
    if (!fifo.impl || !fifo.impl->configured) return 0;
    if (!fifo.delta_sparse_enabled()) return 0;

    const size_t elem_size = fifo.impl->elem_size;
    const size_t stride = fifo.impl->stride;
    const size_t bytes = stride * elem_size;
    if (bytes == 0) return 0;

    if (static_cast<size_t>(edge_idx) >= ctx->edge_sparse_baseline.size()) ctx->edge_sparse_baseline.resize(ctx->edge_fifos.size());
    auto &baseline_map = ctx->edge_sparse_baseline[static_cast<size_t>(edge_idx)];
    std::vector<uint8_t> &baseline = baseline_map[subscriber_key];
    if (baseline.size() != bytes) baseline.assign(bytes, 0u);

    std::vector<uint8_t> sample(bytes);
    size_t wrote = 0;
    if (!fifo.pop(subscriber_key, sample.data(), bytes, wrote) || wrote != bytes) {
        // No token available: still return a valid empty sparse result.
        // This keeps the API convenient for callers that "drain" after
        // coalescing/accumulation (see edge_fifo_sparse_delta_test).
        nodus::tensors::COOMatrix sparse;
        const std::vector<uint32_t> linear;
        const std::vector<uint8_t> values;
        if (!fifo.build_sparse_from_linear(linear, values, &sparse)) return 0;
        auto* heap_sparse = new nodus::tensors::COOMatrix(std::move(sparse));
        *out_sparse = heap_sparse;
        return 1;
    }

    const float threshold = fifo.delta_sparse_threshold;
    const bool accumulate = fifo.delta_sparse_accumulate_enabled();
    std::unordered_map<uint32_t, std::vector<uint8_t>> accum;

    auto apply_diff = [&](const std::vector<uint8_t>& prev, const std::vector<uint8_t>& cur) {
        const bool use_f32 = elem_size == sizeof(float);
        const bool use_f64 = elem_size == sizeof(double);
        for (size_t i = 0; i < stride; ++i) {
            const size_t off = i * elem_size;
            bool changed = false;
            if (use_f32) {
                float a = 0.0f; float b = 0.0f;
                std::memcpy(&a, cur.data() + off, sizeof(float));
                std::memcpy(&b, prev.data() + off, sizeof(float));
                changed = std::fabs(static_cast<double>(a - b)) > static_cast<double>(threshold);
            } else if (use_f64) {
                double a = 0.0; double b = 0.0;
                std::memcpy(&a, cur.data() + off, sizeof(double));
                std::memcpy(&b, prev.data() + off, sizeof(double));
                changed = std::fabs(a - b) > static_cast<double>(threshold);
            } else {
                for (size_t b = 0; b < elem_size; ++b) {
                    if (prev[off + b] != cur[off + b]) { changed = true; break; }
                }
            }
            if (changed) {
                std::vector<uint8_t> buf(elem_size);
                std::memcpy(buf.data(), cur.data() + off, elem_size);
                accum[static_cast<uint32_t>(i)] = std::move(buf);
            }
        }
    };

    apply_diff(baseline, sample);

    if (accumulate) {
        std::vector<uint8_t> next(bytes);
        size_t wrote_next = 0;
        while (fifo.pop(subscriber_key, next.data(), bytes, wrote_next)) {
            if (wrote_next != bytes) break;
            apply_diff(sample, next);
            sample.swap(next);
        }
    }

    std::vector<uint32_t> linear;
    std::vector<uint8_t> values;
    linear.reserve(accum.size());
    values.reserve(accum.size() * elem_size);
    for (const auto& kv : accum) {
        linear.push_back(kv.first);
        values.insert(values.end(), kv.second.begin(), kv.second.end());
    }

    nodus::tensors::COOMatrix sparse;
    if (!fifo.build_sparse_from_linear(linear, values, &sparse)) return 0;
    auto* heap_sparse = new nodus::tensors::COOMatrix(std::move(sparse));
    *out_sparse = heap_sparse;

    baseline = std::move(sample);
    return 1;
}

void gp_table_edge_free_sparse(void* sparse) {
    if (!sparse) return;
    auto* ptr = reinterpret_cast<nodus::tensors::COOMatrix*>(sparse);
    delete ptr;
}

int32_t gp_table_edge_set_order_mode(GP_TableContext* ctx, int32_t edge_idx, int32_t order_mode) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_order_mode(order_mode);
    return 1;
}

int32_t gp_table_edge_set_dirty_grid(GP_TableContext* ctx, int32_t edge_idx, int32_t grid_x, int32_t grid_y, float threshold) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    ctx->edge_fifos[static_cast<size_t>(edge_idx)].set_dirty_grid(grid_x, grid_y, threshold);
    return 1;
}

int32_t gp_table_edge_get_dirty_mask(GP_TableContext* ctx, int32_t edge_idx, uint8_t* out_mask, int32_t out_len, int32_t* out_grid_x, int32_t* out_grid_y, uint64_t* out_seq) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edge_fifos.size())) return 0;
    return ctx->edge_fifos[static_cast<size_t>(edge_idx)].dirty_mask_copy(out_mask, out_len, out_grid_x, out_grid_y, out_seq);
}

int32_t gp_table_remove_edge(GP_TableContext* ctx, int32_t edge_idx) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    if (edge_idx < 0 || edge_idx >= static_cast<int32_t>(ctx->edges.size())) return 0;
    ThreadManager* tm = ThreadManager::global();
    if (tm && edge_idx < static_cast<int32_t>(ctx->edge_subscriber_slots.size())) {
        auto &m = ctx->edge_subscriber_slots[static_cast<size_t>(edge_idx)];
        for (const auto &kv : m) {
            int slot = kv.second;
            if (slot > 0) tm->unregister_reader_slot(slot);
        }
        m.clear();
    }
    auto erase_at = [&](auto &vec) {
        if (edge_idx >= 0 && edge_idx < static_cast<int32_t>(vec.size())) {
            vec.erase(vec.begin() + edge_idx);
        }
    };
    // capture id for removed edge so we can remove mapping entries
    uint64_t removed_id = 0ull;
    if (edge_idx >= 0 && edge_idx < static_cast<int32_t>(ctx->rope_ids.size())) removed_id = ctx->rope_ids[static_cast<size_t>(edge_idx)];
    erase_at(ctx->edges);
    erase_at(ctx->edge_fifos);
    erase_at(ctx->edge_batch_metadata);
    erase_at(ctx->edge_subgroup_flags);
    erase_at(ctx->edge_ids);
    erase_at(ctx->edge_subscriber_slots);
    erase_at(ctx->edge_sparse_baseline);
    erase_at(ctx->relax_value);
    erase_at(ctx->relax_vel);
    // remove any id->sim mapping for the removed edge id
    if (removed_id != 0ull) ctx->rope_id_to_sim_idx.erase(removed_id);
    return 1;
}

int32_t gp_table_remove_edge_pair(GP_TableContext* ctx, unsigned long long a, unsigned long long b) {
    if (!ctx) return 0;
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size(); ++i) {
        const auto &e = ctx->edges[i];
        if ((e.first == a && e.second == b) || (e.first == b && e.second == a)) {
            return gp_table_remove_edge(ctx, static_cast<int32_t>(i));
        }
    }
    return 0;
}

int32_t gp_table_bind_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext* stage, int32_t is_output, int32_t channel) {
    if (!ctx || !stage) return 0;
    StagePortBinding b;
    b.stage = stage;
    b.is_output = is_output ? 1 : 0;
    b.channel = channel;
    ctx->stage_ports[led_key] = b;
    sync_edge_tensors_for_key(ctx, led_key);
    return 1;
}

int32_t gp_table_unbind_stage_port(GP_TableContext* ctx, unsigned long long led_key) {
    if (!ctx) return 0;
    ctx->stage_ports.erase(led_key);
    ensure_edge_fifos(ctx);
    for (size_t i = 0; i < ctx->edges.size() && i < ctx->edge_fifos.size(); ++i) {
        auto &fifo = ctx->edge_fifos[i];
        if (fifo.impl) {
            uint64_t cur = fifo.impl->writer.load(std::memory_order_relaxed);
            if (cur == led_key) fifo.impl->writer.store(0, std::memory_order_relaxed);
        }
        fifo.unsubscribe(led_key);
    }
    sync_edge_tensors_for_key(ctx, led_key);
    return 1;
}

int32_t gp_table_get_stage_port(GP_TableContext* ctx, unsigned long long led_key, GP_StageContext** out_stage, int32_t* out_is_output, int32_t* out_channel) {
    if (!ctx) return 0;
    auto it = ctx->stage_ports.find(led_key);
    if (it == ctx->stage_ports.end()) return 0;
    if (out_stage) *out_stage = it->second.stage;
    if (out_is_output) *out_is_output = it->second.is_output;
    if (out_channel) *out_channel = it->second.channel;
    return 1;
}

int32_t gp_table_get_selected_count(const GP_TableContext* ctx) {
    if (!ctx) return 0;
    return static_cast<int32_t>(ctx->selected_leds.size());
}

int32_t gp_table_get_selected_key(const GP_TableContext* ctx, int32_t idx, unsigned long long* out_key) {
    if (!ctx || !out_key) return 0;
    if (idx < 0 || idx >= static_cast<int>(ctx->selected_leds.size())) return 0;
    auto it = ctx->selected_leds.begin();
    std::advance(it, idx);
    *out_key = *it;
    return 1;
}

int32_t gp_table_set_prospective_mode(GP_TableContext* ctx, int32_t enabled) {
    if (!ctx) return 0;
    ctx->prospective_mode = enabled ? 1 : 0;
    if (!ctx->prospective_mode) ctx->prospective_initialized = false;
    return 1;
}

int32_t gp_table_get_prospective_mode(GP_TableContext* ctx, int32_t* out_enabled) {
    if (!ctx || !out_enabled) return 0;
    *out_enabled = ctx->prospective_mode;
    return 1;
}

// IO / type hint APIs
int32_t gp_table_set_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t type_id, int32_t is_input, int32_t is_output) {
    if (!ctx) return 0;
    ctx->key_type_hint[key] = type_id;
    if (is_input) ctx->key_is_input.insert(key); else ctx->key_is_input.erase(key);
    if (is_output) ctx->key_is_output.insert(key); else ctx->key_is_output.erase(key);
    // Invalidate snapshot so readers rebuild without locking on next access.
    std::atomic_store(&ctx->type_ids_snapshot, std::shared_ptr<std::vector<int32_t>>(nullptr));
    return 1;
}

int32_t gp_table_get_key_type_hint(GP_TableContext* ctx, unsigned long long key, int32_t* out_type_id, int32_t* out_is_input, int32_t* out_is_output) {
    if (!ctx) return 0;
    auto it = ctx->key_type_hint.find(key);
    if (out_type_id) *out_type_id = (it != ctx->key_type_hint.end()) ? it->second : -1;
    if (out_is_input) *out_is_input = ctx->key_is_input.count(key) ? 1 : 0;
    if (out_is_output) *out_is_output = ctx->key_is_output.count(key) ? 1 : 0;
    return 1;
}

int32_t gp_table_has_io_sections(GP_TableContext* ctx) {
    if (!ctx) return 0;
    return (ctx->key_is_input.size() > 0 && ctx->key_is_output.size() > 0) ? 1 : 0;
}

int32_t gp_table_enumerate_io_keys(GP_TableContext* ctx, int32_t direction, unsigned long long* out_keys, int32_t cap) {
    if (!ctx || !out_keys || cap <= 0) return 0;
    int written = 0;
    if (direction == 0) {
        for (auto k : ctx->key_is_input) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    } else {
        for (auto k : ctx->key_is_output) {
            if (written >= cap) break;
            out_keys[written++] = k;
        }
    }
    return written;
}

int32_t gp_table_get_type_ids(GP_TableContext* ctx, int32_t* out_ids, int32_t cap) {
    if (!ctx || !out_ids || cap <= 0) return 0;
    // Try fast path: load existing snapshot atomically
    auto snap = std::atomic_load(&ctx->type_ids_snapshot);
    if (!snap) {
        // Build local snapshot (no locks), then install it atomically.
        std::vector<int32_t> types;
        for (const auto &kv : ctx->key_type_hint) {
            if (kv.second >= 0) {
                bool found = false;
                for (int32_t x : types) if (x == kv.second) { found = true; break; }
                if (!found) types.push_back(kv.second);
            }
        }
        for (const auto &ef : ctx->edge_fifos) {
            if (ef.impl && ef.impl->type_id >= 0) {
                int32_t t = ef.impl->type_id;
                bool found = false;
                for (int32_t x : types) if (x == t) { found = true; break; }
                if (!found) types.push_back(t);
            }
        }
        auto v = std::make_shared<std::vector<int32_t>>(types.begin(), types.end());
        // Install snapshot for readers. Use atomic_store overload for shared_ptr.
        std::atomic_store(&ctx->type_ids_snapshot, v);
        snap = v;
    }
    int written = 0;
    for (int32_t id : *snap) {
        if (written >= cap) break;
        out_ids[written++] = id;
    }
    return static_cast<int32_t>(snap->size());
}

int32_t gp_table_set_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t dir) {
    if (!ctx) return 0;
    if (side < 0 || side > 1) return 0;
    if (dir < 0 || dir > 3) return 0;
    ctx->side_reading_dir[side] = dir;
    return 1;
}

int32_t gp_table_get_side_reading_direction(GP_TableContext* ctx, int32_t side, int32_t* out_dir) {
    if (!ctx || !out_dir) return 0;
    if (side < 0 || side > 1) return 0;
    *out_dir = ctx->side_reading_dir[side];
    return 1;
}

int32_t gp_table_set_led_grid_preference(GP_TableContext* ctx, int32_t pref_cols, int32_t pref_rows, float pref_aspect) {
    if (!ctx) return 0;
    ctx->led_pref_cols = std::max(0, pref_cols);
    ctx->led_pref_rows = std::max(0, pref_rows);
    ctx->led_pref_aspect = pref_aspect;
    return 1;
}

int32_t gp_table_get_led_grid_preference(GP_TableContext* ctx, int32_t* out_pref_cols, int32_t* out_pref_rows, float* out_pref_aspect) {
    if (!ctx) return 0;
    if (out_pref_cols) *out_pref_cols = ctx->led_pref_cols;
    if (out_pref_rows) *out_pref_rows = ctx->led_pref_rows;
    if (out_pref_aspect) *out_pref_aspect = ctx->led_pref_aspect;
    return 1;
}

int32_t gp_table_prospective_set_params(GP_TableContext* ctx, int32_t max_history, float slack, float rope_length) {
    if (!ctx) return 0;
    ctx->prospective_max_history = std::max(1, max_history);
    ctx->prospective_slack = slack;
    ctx->prospective_rope_length = rope_length;
    // trim existing queue if needed
    if (static_cast<int>(ctx->prospective_targets.size()) > ctx->prospective_max_history) {
        ctx->prospective_targets.erase(ctx->prospective_targets.begin(), ctx->prospective_targets.begin() + (ctx->prospective_targets.size() - ctx->prospective_max_history));
    }
    return 1;
}

int32_t gp_table_prospective_get_params(GP_TableContext* ctx, int32_t* out_max_history, float* out_slack, float* out_rope_length) {
    if (!ctx) return 0;
    if (out_max_history) *out_max_history = ctx->prospective_max_history;
    if (out_slack) *out_slack = ctx->prospective_slack;
    if (out_rope_length) *out_rope_length = ctx->prospective_rope_length;
    return 1;
}

// Editable flag helpers
int32_t gp_table_set_editable(GP_TableContext* ctx, int32_t editable) {
    if (!ctx) return 0;
    ctx->editable = editable ? 1 : 0;
    return 1;
}

int32_t gp_table_get_editable(GP_TableContext* ctx, int32_t* out_editable) {
    if (!ctx || !out_editable) return 0;
    *out_editable = ctx->editable;
    return 1;
}

