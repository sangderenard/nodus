// Meta edges as listeners (included by src/table_abi.cpp after table_abi_edge_io.inl).
//
// A meta group "behaves like an edge" (GP_MetaGroup::fifo). Here it listens: once per scheduler tick
// (ThreadManager::run_loop, after the tick's rows have run) gp_table_meta_feed_tick appends one fixed-size record per
// member edge to the group FIFO. A record is an observation of the member's FIFO header (sequence numbers, occupancy,
// refusals); it never copies member data and nothing in this file writes to a member edge. The group FIFO is a ring that
// no reader holds back (it is never subscribed to by the feed), so it is drop-oldest in effect; if a consumer subscribes
// and lets it fill, the push is refused, counted in GP_MetaGroup::feed_refused, and the feed carries on.
// Record layout: include/table_abi.h (GP_MetaRecordWord).
#include <chrono>

namespace {

constexpr size_t kMetaFeedSlots = 1024;  // records kept per group FIFO

uint64_t meta_steady_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

bool meta_feed_configure(GP_MetaGroup* mg) {
    if (!mg) return false;
    EdgeTensorFifo& f = mg->fifo;
    if (f.impl && f.impl->configured && f.impl->stride == GP_META_RECORD_WORDS && f.impl->elem_size == sizeof(uint64_t)) return true;
    try {
        f.configure(std::vector<int32_t>{GP_META_RECORD_WORDS}, kMetaFeedSlots, /*topk=*/0, sizeof(uint64_t));
    } catch (...) { return false; }
    return f.impl && f.impl->configured;
}

// Derive the distinct member ropes of the group from its vertices (caller holds feed_mu).
void meta_sync_members(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (mg->members_built_for == mg->vertices.size()) return;
    std::vector<GP_MetaGroup::MetaMember> next;
    for (const auto& v : mg->vertices) {
        uint64_t rid = v.rope_id;
        if (rid == 0ull && v.rope_idx >= 0 && static_cast<size_t>(v.rope_idx) < ctx->rope_ids.size()) rid = ctx->rope_ids[static_cast<size_t>(v.rope_idx)];
        if (rid == 0ull) continue;
        bool seen = false;
        for (const auto& m : next) if (m.rope_id == rid) { seen = true; break; }
        if (seen) continue;
        GP_MetaGroup::MetaMember m;
        m.rope_id = rid;
        for (const auto& old : mg->members) if (old.rope_id == rid) { m = old; break; }   // keep running deltas
        next.push_back(m);
    }
    mg->members = std::move(next);
    mg->members_built_for = mg->vertices.size();
}

// The table edge a persistent rope id names (rope_ids is indexed by table edge), -1 when the edge is gone.
int32_t meta_member_edge(GP_TableContext* ctx, GP_MetaGroup::MetaMember& m) {
    const size_t n = ctx->rope_ids.size();
    if (m.edge_idx >= 0 && static_cast<size_t>(m.edge_idx) < n && ctx->rope_ids[static_cast<size_t>(m.edge_idx)] == m.rope_id) return m.edge_idx;
    for (size_t i = 0; i < n; ++i) if (ctx->rope_ids[i] == m.rope_id) { m.edge_idx = static_cast<int32_t>(i); return m.edge_idx; }
    m.edge_idx = -1;
    return -1;
}

bool meta_read_ring_record(EdgeTensorFifo& f, uint64_t seq, uint64_t* out) {
    if (!f.impl || !f.impl->configured || f.impl->stride != GP_META_RECORD_WORDS || f.impl->elem_size != sizeof(uint64_t)) return false;
    const size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(f.impl->slots));
    if (f.impl->slot_seq[slot].load(std::memory_order_acquire) != seq + 1) return false;
    const size_t bytes = GP_META_RECORD_WORDS * sizeof(uint64_t);
    gp_mem_backend_handle_t sh = f.impl->storage_handle;
    if (!sh) return false;
    const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(sh);
    if (vt && vt->copy_from_backend) {
        if (!vt->copy_from_backend(sh, slot * bytes, out, bytes)) return false;
    } else {
        void* m = gp_mem_backend_map_or_null(sh);
        if (!m) return false;
        std::memcpy(out, reinterpret_cast<const uint8_t*>(m) + slot * bytes, bytes);
        gp_mem_backend_unmap(sh);
    }
    return f.impl->slot_seq[slot].load(std::memory_order_acquire) == seq + 1;   // not overwritten while reading
}

}  // namespace

extern "C" int32_t gp_table_meta_feed_tick(GP_TableContext* ctx, uint64_t tick_id) {
    if (!ctx || ctx->meta_groups.empty()) return 0;   // no groups: the scheduler pays one branch
    const uint64_t now = meta_steady_ns();
    int32_t groups = 0;
    for (auto& up : ctx->meta_groups) {
        GP_MetaGroup* mg = up.get();
        if (!mg || mg->vertices.empty()) continue;
        std::lock_guard<std::mutex> lk(mg->feed_mu);
        meta_sync_members(ctx, mg);
        if (mg->members.empty() || !meta_feed_configure(mg)) continue;
        const uint64_t dt_ns = mg->feed_last_ns ? now - mg->feed_last_ns : 0;
        mg->feed_last_ns = now;
        uint32_t ordinal = 0;
        for (auto& m : mg->members) {
            uint64_t rec[GP_META_RECORD_WORDS] = {};
            const int32_t ei = meta_member_edge(ctx, m);
            uint64_t flags = 0, dropped_delta = 0;
            rec[GP_META_REC_TICK] = tick_id;
            rec[GP_META_REC_TIME_NS] = now;
            rec[GP_META_REC_DT_NS] = dt_ns;
            rec[GP_META_REC_ORDINAL] = ordinal++;
            rec[GP_META_REC_EDGE_IDX] = ei >= 0 ? static_cast<uint64_t>(ei) : ~0ull;
            rec[GP_META_REC_ROPE_ID] = m.rope_id;
            EdgeTensorFifo* ef = (ei >= 0 && static_cast<size_t>(ei) < ctx->edge_fifos.size()) ? &ctx->edge_fifos[static_cast<size_t>(ei)] : nullptr;
            if (!ef || !ef->impl) {
                flags |= GP_META_FLAG_LOST;
            } else if (ef->impl->configured) {
                auto* impl = ef->impl.get();
                flags |= GP_META_FLAG_CONFIGURED;
                const uint64_t head = impl->write_seq.load(std::memory_order_acquire);
                const uint64_t tail = ef->min_reader_seq(head);
                const uint64_t slots = static_cast<uint64_t>(impl->slots);
                const uint64_t occupancy = std::min<uint64_t>(head >= tail ? head - tail : 0, slots);
                const uint64_t refused = impl->refused_pushes.load(std::memory_order_relaxed);
                const uint64_t dropped = impl->dropped_pushes.load(std::memory_order_relaxed);
                bool readers = false;
                for (size_t i = 0; i < EdgeTensorFifo::kMaxReaders; ++i)
                    if (impl->readers[i].key.load(std::memory_order_relaxed) != 0) { readers = true; break; }
                if (readers) flags |= GP_META_FLAG_READERS;
                if (occupancy >= slots) flags |= GP_META_FLAG_FULL;
                const uint64_t wd = m.primed ? head - std::min(head, m.last_write_seq) : 0;
                const uint64_t rd = m.primed ? tail - std::min(tail, m.last_read_seq) : 0;
                rec[GP_META_REC_WRITE_SEQ] = head;
                rec[GP_META_REC_WRITE_DELTA] = wd;
                rec[GP_META_REC_READ_SEQ] = tail;
                rec[GP_META_REC_READ_DELTA] = rd;
                rec[GP_META_REC_OCCUPANCY] = occupancy;
                rec[GP_META_REC_SLOTS] = slots;
                rec[GP_META_REC_REFUSED_TOTAL] = refused;
                rec[GP_META_REC_REFUSED_DELTA] = m.primed ? refused - std::min(refused, m.last_refused) : 0;
                rec[GP_META_REC_BYTES_DELTA] = wd * static_cast<uint64_t>(impl->stride) * static_cast<uint64_t>(impl->elem_size);
                dropped_delta = m.primed ? dropped - std::min(dropped, m.last_dropped) : 0;
                m.last_write_seq = head; m.last_read_seq = tail; m.last_refused = refused; m.last_dropped = dropped;
                m.primed = true;
            }
            rec[GP_META_REC_FLAGS] = flags | (dropped_delta << 8);
            std::memcpy(m.latest, rec, sizeof(rec));
            m.has_latest = true;
            bool dropped_oldest = false;
            if (!mg->fifo.push(/*edge_id=*/0, /*writer_id=*/mg->id ? mg->id : 1ull, rec, sizeof(rec), &dropped_oldest)) ++mg->feed_refused;
        }
        ++groups;
    }
    return groups;
}

extern "C" GP_MetaGroup* gp_table_meta_create_over_edges(GP_TableContext* ctx, const int32_t* edge_indices, int32_t count, int32_t channel_group) {
    if (!ctx || !edge_indices || count <= 0) return nullptr;
    std::vector<uint64_t> rids;
    for (int32_t i = 0; i < count; ++i) {
        const int32_t ei = edge_indices[i];
        if (ei < 0 || static_cast<size_t>(ei) >= ctx->edges.size() || static_cast<size_t>(ei) >= ctx->rope_ids.size() || ctx->rope_ids[static_cast<size_t>(ei)] == 0ull) return nullptr;
        rids.push_back(ctx->rope_ids[static_cast<size_t>(ei)]);
    }
    GP_MetaGroup* mg = gp_table_meta_create(ctx);
    if (!mg) return nullptr;
    // Membership only: the vertices are recorded, never registered with the rope simulator, so a listening group
    // applies no confinement or spring force to the ropes it listens to.
    for (int32_t i = 0; i < count; ++i) {
        GP_MetaVertex mv{};
        mv.rope_id = rids[static_cast<size_t>(i)];
        auto it = ctx->rope_id_to_sim_idx.find(mv.rope_id);
        mv.rope_idx = it != ctx->rope_id_to_sim_idx.end() ? it->second : edge_indices[i];
        mv.vertex_idx = 0;
        mv.u = -1.0f;
        mg->vertices.push_back(mv);
    }
    mg->channel_group = channel_group;
    meta_feed_configure(mg);
    return mg;
}

extern "C" int32_t gp_table_meta_get_member_count(GP_TableContext* ctx, GP_MetaGroup* mg) {
    if (!ctx || !mg) return 0;
    std::lock_guard<std::mutex> lk(mg->feed_mu);
    meta_sync_members(ctx, mg);
    return static_cast<int32_t>(mg->members.size());
}

extern "C" int32_t gp_table_meta_get_member(GP_TableContext* ctx, GP_MetaGroup* mg, int32_t ordinal, int32_t* out_edge_idx, uint64_t* out_rope_id) {
    if (!ctx || !mg || ordinal < 0) return 0;
    std::lock_guard<std::mutex> lk(mg->feed_mu);
    meta_sync_members(ctx, mg);
    if (static_cast<size_t>(ordinal) >= mg->members.size()) return 0;
    auto& m = mg->members[static_cast<size_t>(ordinal)];
    if (out_edge_idx) *out_edge_idx = meta_member_edge(ctx, m);
    if (out_rope_id) *out_rope_id = m.rope_id;
    return 1;
}

// A replica's records (docs/MESH_CANVAS_UP_PORTS_2026-11-21.md): records another node's feed took of its own members,
// carried over the link with that node's canvas state, appended to this group as they were. Each record becomes its
// member's latest (by rope id) and enters the group FIFO in order, so every reader of this group (edge_introspect, the
// link metrics row, gp_table_meta_read_records) sees the other node's traffic. Nothing here touches a member edge. A
// replica is never stepped, so its own feed never mixes records of its idle copies in. Returns records appended.
extern "C" int32_t gp_table_meta_append_records(GP_TableContext* ctx, GP_MetaGroup* mg, const uint64_t* words, int32_t count) {
    if (!ctx || !mg || !words || count <= 0) return 0;
    std::lock_guard<std::mutex> lk(mg->feed_mu);
    meta_sync_members(ctx, mg);
    if (!meta_feed_configure(mg)) return 0;
    int32_t n = 0;
    for (int32_t i = 0; i < count; ++i) {
        const uint64_t* rec = words + static_cast<size_t>(i) * GP_META_RECORD_WORDS;
        for (auto& m : mg->members) {
            if (m.rope_id != rec[GP_META_REC_ROPE_ID]) continue;
            std::memcpy(m.latest, rec, sizeof(m.latest));
            m.has_latest = true;
            break;
        }
        bool dropped_oldest = false;
        if (!mg->fifo.push(/*edge_id=*/0, /*writer_id=*/mg->id ? mg->id : 1ull, rec, GP_META_RECORD_WORDS * sizeof(uint64_t),
                           &dropped_oldest)) { ++mg->feed_refused; continue; }
        ++n;
    }
    return n;
}

extern "C" int32_t gp_table_meta_get_latest_records(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t* out_words, int32_t max_records) {
    if (!ctx || !mg || !out_words || max_records <= 0) return 0;
    std::lock_guard<std::mutex> lk(mg->feed_mu);
    meta_sync_members(ctx, mg);
    int32_t n = 0;
    for (const auto& m : mg->members) {
        if (n >= max_records) break;
        if (!m.has_latest) continue;
        std::memcpy(out_words + static_cast<size_t>(n) * GP_META_RECORD_WORDS, m.latest, sizeof(m.latest));
        ++n;
    }
    return n;
}

extern "C" int32_t gp_table_meta_read_records(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t from_seq, uint64_t* out_words, int32_t max_records, uint64_t* out_write_seq) {
    if (!ctx || !mg) return 0;
    EdgeTensorFifo& f = mg->fifo;
    if (!f.impl || !f.impl->configured) { if (out_write_seq) *out_write_seq = 0; return 0; }
    const uint64_t head = f.impl->write_seq.load(std::memory_order_acquire);
    if (out_write_seq) *out_write_seq = head;
    if (!out_words || max_records <= 0) return 0;
    const uint64_t oldest = head > f.impl->slots ? head - f.impl->slots : 0;
    uint64_t seq = std::max(from_seq, oldest);
    int32_t n = 0;
    for (; seq < head && n < max_records; ++seq) {
        if (meta_read_ring_record(f, seq, out_words + static_cast<size_t>(n) * GP_META_RECORD_WORDS)) ++n;
    }
    return n;
}

// 64-bit snapshot of a group: GP_MetaSnapshotWord header, then each member's latest record (zeros before its first tick).
extern "C" int32_t gp_table_meta_snapshot_u64(GP_TableContext* ctx, GP_MetaGroup* mg, uint64_t* out_words, int32_t capacity) {
    if (!ctx || !mg || !out_words || capacity < GP_META_SNAPSHOT_HEADER_WORDS) return 0;
    std::lock_guard<std::mutex> lk(mg->feed_mu);
    meta_sync_members(ctx, mg);
    out_words[GP_META_SNAP_ID] = mg->id;
    out_words[GP_META_SNAP_CHANNEL] = static_cast<uint64_t>(static_cast<int64_t>(mg->channel_group));
    out_words[GP_META_SNAP_VERTICES] = mg->vertices.size();
    out_words[GP_META_SNAP_MEMBERS] = mg->members.size();
    out_words[GP_META_SNAP_RECORDS] = mg->fifo.impl ? mg->fifo.impl->write_seq.load(std::memory_order_acquire) : 0ull;
    out_words[GP_META_SNAP_FEED_REFUSED] = mg->feed_refused;
    out_words[GP_META_SNAP_SLOTS] = mg->fifo.impl ? mg->fifo.impl->slots : 0ull;
    out_words[GP_META_SNAP_RECORD_WORDS] = GP_META_RECORD_WORDS;
    int32_t pos = GP_META_SNAPSHOT_HEADER_WORDS;
    for (const auto& m : mg->members) {
        if (pos + GP_META_RECORD_WORDS > capacity) break;
        if (m.has_latest) std::memcpy(out_words + pos, m.latest, sizeof(m.latest));
        else {
            std::memset(out_words + pos, 0, sizeof(m.latest));
            out_words[pos + GP_META_REC_ROPE_ID] = m.rope_id;
            out_words[pos + GP_META_REC_EDGE_IDX] = m.edge_idx >= 0 ? static_cast<uint64_t>(m.edge_idx) : ~0ull;
        }
        pos += GP_META_RECORD_WORDS;
    }
    return pos;
}
