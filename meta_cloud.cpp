#include "meta_cloud.h"
#include "canvas_abi.h"
#include "table_abi.h"
#include <cstring>
#include <cassert>
#include <sstream>

MetaCloud::MetaCloud() = default;
MetaCloud::~MetaCloud() = default;

void MetaCloud::add_vertex(const VertexRef& v) {
    vertices_.push_back(v);
}

void MetaCloud::set_overlay(const OverlayBinding& o) {
    overlay_ = o;
}

void MetaCloud::add_ring(const RingRef& r) {
    rings_.push_back(r);
}

int MetaCloud::serialize(std::vector<char>& out) const {
    // Simple binary layout:
    // uint32_t version
    // uint32_t vertex_count
    // for each vertex: uint64_t rope_uid, int32_t vertex_idx
    // uint32_t ring_count
    // for each ring: uint64_t uid, uint64_t key, float u
    // overlay: uint64_t key_a, uint64_t key_b, uint64_t port_a, uint64_t port_b

    size_t need = 0;
    need += 4; // version
    need += 4; // vertex_count
    need += vertices_.size() * (8 + 4);
    need += 4; // ring_count
    need += rings_.size() * (8 + 8 + 4);
    need += 8*4; // overlay fields

    out.resize(need);
    char* p = out.data();

    uint32_t v = version_;
    memcpy(p, &v, 4); p += 4;
    uint32_t vc = static_cast<uint32_t>(vertices_.size());
    memcpy(p, &vc, 4); p += 4;
    for (const auto &vv : vertices_) {
        uint64_t ru = vv.rope_uid;
        int32_t vi = vv.vertex_idx;
        memcpy(p, &ru, 8); p += 8;
        memcpy(p, &vi, 4); p += 4;
    }
    uint32_t rc = static_cast<uint32_t>(rings_.size());
    memcpy(p, &rc, 4); p += 4;
    for (const auto &rr : rings_) {
        uint64_t uid = rr.uid;
        uint64_t key = rr.key;
        float u = rr.u;
        memcpy(p, &uid, 8); p += 8;
        memcpy(p, &key, 8); p += 8;
        memcpy(p, &u, 4); p += 4;
    }
    memcpy(p, &overlay_.key_a, 8); p += 8;
    memcpy(p, &overlay_.key_b, 8); p += 8;
    memcpy(p, &overlay_.port_uuid_a, 8); p += 8;
    memcpy(p, &overlay_.port_uuid_b, 8); p += 8;

    assert(static_cast<size_t>(p - out.data()) == need);
    return static_cast<int>(need);
}

int MetaCloud::deserialize(const char* data, int32_t len) {
    if (!data || len <= 0) return 0;
    const char* p = data;
    const char* end = data + len;
    if (p + 4 > end) return 0;
    uint32_t v = 0; memcpy(&v, p, 4); p += 4;
    version_ = v;
    if (p + 4 > end) return 0;
    uint32_t vc = 0; memcpy(&vc, p, 4); p += 4;
    vertices_.clear();
    for (uint32_t i = 0; i < vc; ++i) {
        if (p + 12 > end) return 0;
        uint64_t ru = 0; int32_t vi = -1;
        memcpy(&ru, p, 8); p += 8;
        memcpy(&vi, p, 4); p += 4;
        VertexRef vr; vr.rope_uid = ru; vr.vertex_idx = vi;
        vertices_.push_back(vr);
    }
    if (p + 4 > end) return 0;
    uint32_t rc = 0; memcpy(&rc, p, 4); p += 4;
    rings_.clear();
    for (uint32_t i = 0; i < rc; ++i) {
        if (p + 20 > end) return 0;
        uint64_t uid = 0; uint64_t key = 0; float u = 0.0f;
        memcpy(&uid, p, 8); p += 8;
        memcpy(&key, p, 8); p += 8;
        memcpy(&u, p, 4); p += 4;
        RingRef rr; rr.uid = uid; rr.key = key; rr.u = u;
        rings_.push_back(rr);
    }
    if (p + 32 > end) return 0;
    memcpy(&overlay_.key_a, p, 8); p += 8;
    memcpy(&overlay_.key_b, p, 8); p += 8;
    memcpy(&overlay_.port_uuid_a, p, 8); p += 8;
    memcpy(&overlay_.port_uuid_b, p, 8); p += 8;
    // Successfully parsed
    return 1;
}

GP_MetaGroup* MetaCloud::apply_to_table(GP_TableContext* ctx) const {
    if (!ctx) return nullptr;
    // Create the runtime meta-group container
    GP_MetaGroup* mg = gp_table_meta_create(ctx);
    if (!mg) return nullptr;

    GP_CanvasContext* cvs = gp_canvas_get_singleton();

    // Track created rings and ring-entries for cleanup on failure
    std::vector<int> created_ring_ids;
    std::vector<int> created_ring_entries;

    // 1) Create rings (by persisted rope UID) and register ring edges if a key is present
    for (const auto &r : rings_) {
        if (r.uid == 0ull) continue;
        int ring_id = gp_table_create_ring_by_uid(ctx, r.uid, r.u);
        if (ring_id < 0) {
            // UID->index resolution failure or create failure: abort
            for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
            gp_table_meta_destroy(ctx, mg);
            return 3;
        }
        created_ring_ids.push_back(ring_id);
        if (r.key != 0ull) {
            int entry = gp_table_register_ring_edge(ctx, ring_id, r.key);
            if (entry < 0) {
                // failed to register ring entry
                for (int eid : created_ring_entries) gp_table_unregister_ring_edge(ctx, eid);
                for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
                gp_table_meta_destroy(ctx, mg);
                return 3;
            }
            created_ring_entries.push_back(entry);
        }
    }

    // 2) Add vertices by rope UID (prefer canvas mapping, fallback to table-local mapping)
    for (const auto &v : vertices_) {
        if (v.rope_uid == 0ull) {
            // allow vertex_idx == -1 (placeholder) but otherwise treat as error
            if (v.vertex_idx >= 0) {
                // invalid persisted record
                for (int eid : created_ring_entries) gp_table_unregister_ring_edge(ctx, eid);
                for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
                gp_table_meta_destroy(ctx, mg);
                return 5;
            }
            continue;
        }
        int ok = 0;
        if (cvs) ok = gp_canvas_table_meta_add_vertex_by_uid(cvs, ctx, reinterpret_cast<void*>(mg), v.rope_uid, v.vertex_idx);
        if (!ok) {
            // fallback: find table-local rope index
            int rope_idx = -1;
            for (size_t ri = 0; ri < ctx->rope_uids.size(); ++ri) {
                if (ctx->rope_uids[ri] == v.rope_uid) { rope_idx = static_cast<int>(ri); break; }
            }
            if (rope_idx < 0) {
                for (int eid : created_ring_entries) gp_table_unregister_ring_edge(ctx, eid);
                for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
                gp_table_meta_destroy(ctx, mg);
                return 5;
            }
            int ok2 = gp_table_meta_add_vertex(ctx, mg, rope_idx, v.vertex_idx);
            if (!ok2) {
                for (int eid : created_ring_entries) gp_table_unregister_ring_edge(ctx, eid);
                for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
                gp_table_meta_destroy(ctx, mg);
                return 4;
            }
        }
    }

    // 3) Register overlay canonical instance (if keys present) then bind to meta-group
    if ((overlay_.key_a != 0ull || overlay_.key_b != 0ull) && cvs) {
        gp_canvas_register_table_overlay(cvs, overlay_.key_a, overlay_.key_b, overlay_.port_uuid_a, overlay_.port_uuid_b);
        int s = gp_canvas_set_overlay_meta(cvs, overlay_.key_a, overlay_.key_b, ctx, reinterpret_cast<void*>(mg));
        if (!s) {
            for (int eid : created_ring_entries) gp_table_unregister_ring_edge(ctx, eid);
            for (int rid : created_ring_ids) gp_table_destroy_ring(ctx, rid);
            gp_table_meta_destroy(ctx, mg);
            return 6;
        }
    } else if ((overlay_.key_a != 0ull || overlay_.key_b != 0ull)) {
        // No canvas available; at minimum persist keys onto the meta-group for later binding
        gp_table_meta_set_overlay_keys(ctx, mg, overlay_.key_a, overlay_.key_b);
    }

    return mg;
}

MetaCloud MetaCloud::from_table_snapshot(GP_TableContext* /*ctx*/, GP_MetaGroup* /*mg*/) {
    MetaCloud mc;
    // TODO: query GP_TableContext/GP_MetaGroup to populate vertices_, rings_, overlay_
    return mc;
}

bool MetaCloud::validate() const {
    // Basic validation: ensure ring u in [0..1] and vertex indices non-negative
    for (const auto &r : rings_) {
        if (!(r.u >= 0.0f && r.u <= 1.0f)) return false;
    }
    for (const auto &v : vertices_) {
        if (v.vertex_idx < -1) return false;
    }
    return true;
}

int MetaCloud::finalize_build() {
    // Compute derived/normalized state if needed. Currently a no-op.
    return 1;
}

std::string MetaCloud::to_string() const {
    std::ostringstream ss;
    ss << "MetaCloud[v=" << vertices_.size() << ", r=" << rings_.size() << "]";
    return ss.str();
}
