#include "rope_sim.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <cstring>
#include <cstddef>
#include <algorithm>
#include "console_logger.h"
#ifndef printf
#define printf(...) CONSOLE_PRINTF(__VA_ARGS__)
#endif
#ifndef fprintf
#define fprintf(file, ...) CONSOLE_PRINTF(__VA_ARGS__)
#endif

// Optional compile-time Eigen path: define EIGEN_SIM to enable Eigen-accelerated paths.
#ifdef EIGEN_SIM
#include <Eigen/Dense>
#endif

struct Rope {
    int segments = 0; // number of segments => vertices = segments + 1
    float slack = 0.0f; // fraction: 0 = tight, >0 allows longer rest length
    float rest_len = 0.0f; // per-segment rest length (fallback when per-seg not set)
    std::vector<float> rest_len_seg; // per-segment rest lengths (size == segments)
    // positions separated for easier vectorization
    std::vector<float> pos_x; // size = verts
    std::vector<float> pos_y;
    std::vector<float> pos_z;
    std::vector<float> prev_x;
    std::vector<float> prev_y;
    std::vector<float> prev_z;
    float ax = 0.0f, ay = 0.0f; // fixed endpoint A
    float bx = 0.0f, by = 0.0f; // fixed endpoint B
    float az = 0.0f, bz = 0.0f; // fixed endpoint depth
    // Track rope-level participation so rings and dense spring networks are
    // first-class aspects of the rope itself.
    std::vector<int> ring_ids;      // ring ids bound to this rope
    std::vector<int> meta_groups;   // meta-group indices this rope participates in
    int owned_meta_group = -1;      // if non-negative, this rope owns the dense network/meta-group
    bool is_meta_rope = false;      // marked as meta rope carrying lasso network
    std::vector<int> meta_vertices; // rope vertex indices participating in dense connection
};

struct RopeSim {
    int max_ropes = 0;
    int max_segments_per_rope = 0;
    std::vector<std::unique_ptr<Rope>> ropes;
};

// forward-declare internal augmented sim so creation can return the internal type
struct RopeSim_internal;

extern "C" {

RopeSim* rope_sim_create(int max_ropes, int max_segments_per_rope);

void rope_sim_destroy(RopeSim* s) {
    if (!s) return;
    delete s;
}

int rope_sim_add_rope(RopeSim* s, float ax, float ay, float bx, float by, int segments, float slack_fraction) {
    return rope_sim_add_rope3(s, ax, ay, 0.0f, bx, by, 0.0f, segments, slack_fraction);
}


int rope_sim_add_rope3(RopeSim* s, float ax, float ay, float az, float bx, float by, float bz, int segments, float slack_fraction) {
    if (!s) return -1;
    if (segments < 1) segments = 1;
    if (segments > s->max_segments_per_rope) segments = s->max_segments_per_rope;
    if ((int)s->ropes.size() >= s->max_ropes) return -1;
    auto r = std::make_unique<Rope>();
    r->segments = segments;
    r->slack = std::max(0.0f, slack_fraction);
    r->ax = ax; r->ay = ay; r->az = az;
    r->bx = bx; r->by = by; r->bz = bz;
    int verts = segments + 1;
    r->pos_x.resize(static_cast<std::size_t>(verts));
    r->pos_y.resize(static_cast<std::size_t>(verts));
    r->pos_z.resize(static_cast<std::size_t>(verts));
    r->prev_x.resize(static_cast<std::size_t>(verts));
    r->prev_y.resize(static_cast<std::size_t>(verts));
    r->prev_z.resize(static_cast<std::size_t>(verts));
    // initialize evenly spaced positions between endpoints, with a slight out-of-plane bow towards z=0
    for (int i = 0; i < verts; ++i) {
        float t = float(i) / float(segments);
        float x = ax + (bx - ax) * t;
        float y = ay + (by - ay) * t;
        float base_z = az + (bz - az) * t;
        float lift = std::max(0.0f, std::max(-az, -bz));
        float z = base_z + std::sin(3.14159265f * t) * lift; // bow out toward the viewer
        r->pos_x[i] = x;
        r->pos_y[i] = y;
        r->pos_z[i] = z;
        r->prev_x[i] = x;
        r->prev_y[i] = y;
        r->prev_z[i] = z;
    }
    // rest length per segment based on straight-line length * (1 + slack)
    float dx = bx - ax;
    float dy = by - ay;
    float dz = bz - az;
    float total = std::sqrt(dx*dx + dy*dy + dz*dz);
    float seglen = total / float(segments);
    r->rest_len = seglen * (1.0f + r->slack);
    r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    int idx = static_cast<int>(s->ropes.size());
    s->ropes.push_back(std::move(r));
    // debug: log rope creation for diagnostics (helps track requested segment counts)
    {
        Rope* created = s->ropes[static_cast<size_t>(idx)].get();
        if (created) {
            int verts = created->segments + 1;
            printf("[rope_sim] add rope idx=%d segments=%d verts=%d ax=%.2f,%.2f bx=%.2f,%.2f slack=%.3f rest_len=%.3f\n",
                idx, created->segments, verts, created->ax, created->ay, created->bx, created->by, created->slack, created->rest_len);
        }
    }
    return idx;
}

static inline void verlet_step_point(float &x, float &y, float &z, float &px, float &py, float &pz, float dt, float gx, float gy, float gz, float damping) {
    // Verlet: x_new = x + (x - px) * (1-damping) + a*dt*dt
    float vx = x - px;
    float nx = x + vx * (1.0f - damping) + gx * dt * dt;
    float vy = y - py;
    float ny = y + vy * (1.0f - damping) + gy * dt * dt;
    float vz = z - pz;
    float nz = z + vz * (1.0f - damping) + gz * dt * dt;
    px = x; py = y;
    pz = z;
    x = nx; y = ny; z = nz;
}

// Note: the detailed step implementation is defined later and integrates
// meta-group constraints. Remove earlier duplicate implementation to keep
// a single `rope_sim_step` body (the later definition is used).

int rope_sim_get_vertex_count(RopeSim* s, int rope_idx) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    return r->segments + 1;
}

int rope_sim_set_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    return rope_sim_set_endpoints3(s, rope_idx, ax, ay, r->az, bx, by, r->bz);
}

int rope_sim_set_endpoints3(RopeSim* s, int rope_idx, float ax, float ay, float az, float bx, float by, float bz) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    r->ax = ax; r->ay = ay; r->az = az; r->bx = bx; r->by = by; r->bz = bz;
    int verts = r->segments + 1;
    if (verts <= 0) return 0;
    // snap endpoints in current state so constraints anchor
    r->pos_x[0] = ax; r->pos_y[0] = ay; r->pos_z[0] = az;
    r->prev_x[0] = ax; r->prev_y[0] = ay; r->prev_z[0] = az;
    r->pos_x[verts-1] = bx; r->pos_y[verts-1] = by; r->pos_z[verts-1] = bz;
    r->prev_x[verts-1] = bx; r->prev_y[verts-1] = by; r->prev_z[verts-1] = bz;
    float dx = bx - ax;
    float dy = by - ay;
    float dz = bz - az;
    float total = std::sqrt(dx*dx + dy*dy + dz*dz);
    float seglen = total / std::max(1, r->segments);
    r->rest_len = seglen * (1.0f + r->slack);
    r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    return 1;
}

int rope_sim_move_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    return rope_sim_move_endpoints3(s, rope_idx, ax, ay, r->az, bx, by, r->bz);
}

int rope_sim_move_endpoints3(RopeSim* s, int rope_idx, float ax, float ay, float az, float bx, float by, float bz) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    int verts = r->segments + 1;
    if (verts <= 0) return 0;
    // preserve previous positions so integrator sees endpoint motion
    float old_ax = r->pos_x[0];
    float old_ay = r->pos_y[0];
    float old_az = r->pos_z[0];
    float old_bx = r->pos_x[verts-1];
    float old_by = r->pos_y[verts-1];
    float old_bz = r->pos_z[verts-1];
    r->ax = ax; r->ay = ay; r->az = az; r->bx = bx; r->by = by; r->bz = bz;
    // set prev to old pos so vx = pos - prev will be (new - old)
    r->prev_x[0] = old_ax; r->prev_y[0] = old_ay; r->prev_z[0] = old_az;
    r->pos_x[0] = ax; r->pos_y[0] = ay; r->pos_z[0] = az;
    r->prev_x[verts-1] = old_bx; r->prev_y[verts-1] = old_by; r->prev_z[verts-1] = old_bz;
    r->pos_x[verts-1] = bx; r->pos_y[verts-1] = by; r->pos_z[verts-1] = bz;
    float dx = bx - ax;
    float dy = by - ay;
    float dz = bz - az;
    float total = std::sqrt(dx*dx + dy*dy + dz*dz);
    float seglen = total / std::max(1, r->segments);
    r->rest_len = seglen * (1.0f + r->slack);
    float new_total = r->rest_len * static_cast<float>(r->segments);
    if (r->rest_len_seg.size() == static_cast<size_t>(r->segments)) {
        float old_total = 0.0f;
        for (float v : r->rest_len_seg) old_total += v;
        if (old_total > 1e-6f && std::isfinite(old_total)) {
            float scale = new_total / old_total;
            for (float &v : r->rest_len_seg) v *= scale;
        } else {
            r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
        }
    } else {
        r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    }
    return 1;
}

int rope_sim_get_vertices(RopeSim* s, int rope_idx, float* out_xy, int max_count) {
    if (!s || !out_xy) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    int verts = r->segments + 1;
    if (max_count < 2 * verts) return 0;
    // interleave x,y into out_xy
    for (int i = 0; i < verts; ++i) {
        out_xy[2*i+0] = r->pos_x[i];
        out_xy[2*i+1] = r->pos_y[i];
    }
    return verts;
}

int rope_sim_get_vertices3(RopeSim* s, int rope_idx, float* out_xyz, int max_count) {
    if (!s || !out_xyz) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<std::size_t>(rope_idx)].get();
    if (!r) return 0;
    int verts = r->segments + 1;
    if (max_count < 3 * verts) return 0;
    for (int i = 0; i < verts; ++i) {
        out_xyz[3*i+0] = r->pos_x[i];
        out_xyz[3*i+1] = r->pos_y[i];
        out_xyz[3*i+2] = r->pos_z[i];
    }
    return verts;
}

int rope_sim_set_vertex_position(RopeSim* s, int rope_idx, int vertex_idx, float x, float y, float z) {
    if (!s) return 0;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(s->ropes.size())) return 0;
    Rope* r = s->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    int verts = r->segments + 1;
    if (vertex_idx < 0 || vertex_idx >= verts) return 0;
    r->pos_x[static_cast<size_t>(vertex_idx)] = x;
    r->pos_y[static_cast<size_t>(vertex_idx)] = y;
    r->pos_z[static_cast<size_t>(vertex_idx)] = z;
    r->prev_x[static_cast<size_t>(vertex_idx)] = x;
    r->prev_y[static_cast<size_t>(vertex_idx)] = y;
    r->prev_z[static_cast<size_t>(vertex_idx)] = z;
    return 1;
}

// --- Meta-group (lightweight) implementation ---------------------------

// Each meta-group stores an ordered list of (rope_idx, vertex_idx) members
// and optional edge-springs connecting members according to a selected mode.
struct MetaGroup {
    float pressure = 0.0f;
    std::vector<std::pair<int,int>> members; // (rope_idx, vertex_idx)
    int mode = 0; // 0=ribbon,1=closed,2=dense
    bool edges_enabled = false;
    float min_rest = 0.0f;
    float reduce_rate = 0.0f;
    // edges stored as pairs of member indices (ia, ib)
    std::vector<std::pair<int,int>> edges;
    // per-edge rest lengths (same order as `edges`)
    std::vector<float> edge_rest;
    // per-edge stiffness multiplier (1.0 = default/full correction)
    std::vector<float> edge_strength;
    // optional rope index for a dangling/widget rope attached to this meta-group
    int dangling_rope = -1;
    // per-member parametric position along rope (0..1). Mirrors `members`.
    std::vector<float> member_u;
    // optional star center member index (used for mode=2 star topology)
    int star_center_idx = -1;
};

// augment RopeSim with meta_groups container
struct RopeSim_internal : public RopeSim {
    std::vector<std::unique_ptr<MetaGroup>> meta_groups;
    struct Ring {
        int rope_idx = -1;
        int meta_group_idx = -1;
        int member_idx = -1;
        int vertex_idx = -1;
        float pair_rest_total = 0.0f;
        float u = 0.0f;
        float u_vel = 0.0f;
        float x = 0.0f, y = 0.0f, z = 0.0f;
        float prev_x = 0.0f, prev_y = 0.0f, prev_z = 0.0f;
    };
    std::vector<std::unique_ptr<Ring>> rings;
};

// Helper to cast to internal
static RopeSim_internal* to_internal(RopeSim* s) {
    return reinterpret_cast<RopeSim_internal*>(s);
}

// Rope-level membership helpers: keep per-rope awareness of rings and meta-groups
static void rope_track_ring(RopeSim_internal* si, int rope_idx, int ring_id) {
    if (!si) return;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return;
    if (std::find(r->ring_ids.begin(), r->ring_ids.end(), ring_id) == r->ring_ids.end()) {
        r->ring_ids.push_back(ring_id);
    }
}
static void rope_untrack_ring(RopeSim_internal* si, int rope_idx, int ring_id) {
    if (!si) return;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return;
    r->ring_ids.erase(std::remove(r->ring_ids.begin(), r->ring_ids.end(), ring_id), r->ring_ids.end());
}
static void rope_track_meta_group(RopeSim_internal* si, int rope_idx, int group_idx) {
    if (!si) return;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return;
    if (group_idx < 0) return;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return;
    if (std::find(r->meta_groups.begin(), r->meta_groups.end(), group_idx) == r->meta_groups.end()) {
        r->meta_groups.push_back(group_idx);
    }
}
static void rope_untrack_meta_group(RopeSim_internal* si, int rope_idx, int group_idx) {
    if (!si) return;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return;
    if (group_idx < 0) return;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return;
    r->meta_groups.erase(std::remove(r->meta_groups.begin(), r->meta_groups.end(), group_idx), r->meta_groups.end());
}
static void rope_rebuild_memberships(RopeSim_internal* si) {
    if (!si) return;
    for (auto &rp : si->ropes) {
        if (!rp) continue;
        rp->ring_ids.clear();
        rp->meta_groups.clear();
    }
    // Rings -> rope
    for (size_t ri = 0; ri < si->rings.size(); ++ri) {
        auto *rg = si->rings[ri].get();
        if (!rg) continue;
        rope_track_ring(si, rg->rope_idx, static_cast<int>(ri));
    }
    // Meta-groups -> rope
    for (size_t gi = 0; gi < si->meta_groups.size(); ++gi) {
        MetaGroup* mg = si->meta_groups[gi].get();
        if (!mg) continue;
        for (auto &m : mg->members) {
            rope_track_meta_group(si, m.first, static_cast<int>(gi));
        }
        if (mg->dangling_rope >= 0) {
            rope_track_meta_group(si, mg->dangling_rope, static_cast<int>(gi));
        }
    }
}

// Ownership setter/query
int rope_sim_set_rope_meta_owner(RopeSim* s, int rope_idx, int group_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    r->owned_meta_group = group_idx;
    // also ensure rope is tracking membership list for the owner group
    rope_track_meta_group(si, rope_idx, group_idx);
    return 1;
}

int rope_sim_get_rope_meta_owner(RopeSim* s, int rope_idx, int* out_group_idx) {
    if (!s || !out_group_idx) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    *out_group_idx = r->owned_meta_group;
    return 1;
}

int rope_sim_get_rope_rings(RopeSim* s, int rope_idx, int* out_ids, int max_count) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    int count = static_cast<int>(r->ring_ids.size());
    if (!out_ids || max_count <= 0) return count;
    int write = std::min(count, max_count);
    for (int i = 0; i < write; ++i) out_ids[i] = r->ring_ids[static_cast<size_t>(i)];
    return write;
}

int rope_sim_get_rope_meta_groups(RopeSim* s, int rope_idx, int* out_groups, int max_count) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    int count = static_cast<int>(r->meta_groups.size());
    if (!out_groups || max_count <= 0) return count;
    int write = std::min(count, max_count);
    for (int i = 0; i < write; ++i) out_groups[i] = r->meta_groups[static_cast<size_t>(i)];
    return write;
}

int rope_sim_mark_meta_rope(RopeSim* s, int rope_idx, const int* vertex_indices, int count) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    r->is_meta_rope = true;
    r->meta_vertices.clear();
    if (vertex_indices && count > 0) {
        r->meta_vertices.assign(vertex_indices, vertex_indices + count);
    }
    return 1;
}

int rope_sim_is_meta_rope(RopeSim* s, int rope_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    return r->is_meta_rope ? 1 : 0;
}

int rope_sim_get_meta_vertices(RopeSim* s, int rope_idx, int* out_vertices, int max_count) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    int count = static_cast<int>(r->meta_vertices.size());
    if (!out_vertices || max_count <= 0) return count;
    int write = std::min(count, max_count);
    for (int i = 0; i < write; ++i) out_vertices[i] = r->meta_vertices[static_cast<size_t>(i)];
    return write;
}

// Now that RopeSim_internal is defined, provide the real create implementation
RopeSim* rope_sim_create(int max_ropes, int max_segments_per_rope) {
    if (max_ropes <= 0 || max_segments_per_rope <= 0) return nullptr;
    RopeSim_internal* si = new RopeSim_internal();
    RopeSim* s = reinterpret_cast<RopeSim*>(si);
    s->max_ropes = max_ropes;
    s->max_segments_per_rope = max_segments_per_rope;
    s->ropes.reserve(static_cast<std::size_t>(max_ropes));
    return s;
}

int rope_sim_create_meta_group(RopeSim* s, float pressure) {
    if (!s) return -1;
    RopeSim_internal* si = to_internal(s);
    auto mg = std::make_unique<MetaGroup>();
    mg->pressure = pressure;
    int idx = static_cast<int>(si->meta_groups.size());
    si->meta_groups.push_back(std::move(mg));
    // debug: create_meta_group logging removed to reduce output
    // printf("[rope_sim] create_meta_group %d pressure=%f\n", idx, pressure);
    return idx;
}

int rope_sim_destroy_meta_group(RopeSim* s, int group_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (mg) {
        for (const auto &m : mg->members) {
            rope_untrack_meta_group(si, m.first, group_idx);
        }
        if (mg->dangling_rope >= 0) {
            rope_untrack_meta_group(si, mg->dangling_rope, group_idx);
        }
    }
    // Clear ownership on any rope claiming this group
    for (auto &rp : si->ropes) {
        if (!rp) continue;
        if (rp->owned_meta_group == group_idx) rp->owned_meta_group = -1;
    }
    si->meta_groups[static_cast<size_t>(group_idx)].reset();
    return 1;
}

int rope_sim_meta_group_add(RopeSim* s, int group_idx, int rope_idx, int vertex_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    printf("rope_sim_meta_group_add: sim=%p group=%d rope=%d vert=%d ropes=%zu mg_count=%zu\n", (void*)s, group_idx, rope_idx, vertex_idx, si->ropes.size(), si->meta_groups.size());
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->members.emplace_back(rope_idx, vertex_idx);
    rope_track_meta_group(si, rope_idx, group_idx);
    // initialize member param `u` from vertex index if possible
    float u = 0.0f;
    if (rope_idx >= 0 && rope_idx < static_cast<int>(si->ropes.size())) {
        Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
        if (r) {
            int verts = r->segments + 1;
            if (verts > 1) u = static_cast<float>(vertex_idx) / static_cast<float>(verts - 1);
        }
    }
    mg->member_u.push_back(u);
    printf("  -> added member u=%.6f (computed from verts=%d)\n", u, (rope_idx >= 0 && rope_idx < static_cast<int>(si->ropes.size())) ? si->ropes[static_cast<size_t>(rope_idx)]->segments + 1 : 0);
    return 1;
}

int rope_sim_meta_group_has_member(RopeSim* s, int group_idx, int rope_idx, int vertex_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    for (size_t i = 0; i < mg->members.size(); ++i) {
        if (mg->members[i].first == rope_idx && mg->members[i].second == vertex_idx) return 1;
    }
    return 0;
}

int rope_sim_meta_group_insert(RopeSim* s, int group_idx, int after_rope_idx, int after_vertex_idx, int rope_idx, int vertex_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    printf("rope_sim_meta_group_insert: sim=%p group=%d after=(%d,%d) insert=(%d,%d) ropes=%zu mg_count=%zu\n", (void*)s, group_idx, after_rope_idx, after_vertex_idx, rope_idx, vertex_idx, si->ropes.size(), si->meta_groups.size());
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    // find after pair
    size_t pos = mg->members.size();
    for (size_t i = 0; i < mg->members.size(); ++i) {
        if (mg->members[i].first == after_rope_idx && mg->members[i].second == after_vertex_idx) {
            pos = i + 1;
            break;
        }
    }
    mg->members.insert(mg->members.begin() + static_cast<ptrdiff_t>(pos), std::make_pair(rope_idx, vertex_idx));
    rope_track_meta_group(si, rope_idx, group_idx);
    // insert parametric u at same position
    float u = 0.0f;
    if (rope_idx >= 0 && rope_idx < static_cast<int>(si->ropes.size())) {
        Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
        if (r) {
            int verts = r->segments + 1;
            if (verts > 1) u = static_cast<float>(vertex_idx) / static_cast<float>(verts - 1);
        }
    }
    mg->member_u.insert(mg->member_u.begin() + static_cast<ptrdiff_t>(pos), u);
    printf("  -> inserted member at pos=%zu u=%.6f (computed from verts=%d)\n", pos, u, (rope_idx >= 0 && rope_idx < static_cast<int>(si->ropes.size())) ? si->ropes[static_cast<size_t>(rope_idx)]->segments + 1 : 0);
    return 1;
}

int rope_sim_meta_group_set_pressure(RopeSim* s, int group_idx, float pressure) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->pressure = pressure;
    return 1;
}

int rope_sim_meta_group_set_star_center(RopeSim* s, int group_idx, int rope_idx, int vertex_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    int found = -1;
    for (size_t i = 0; i < mg->members.size(); ++i) {
        if (mg->members[i].first == rope_idx && mg->members[i].second == vertex_idx) {
            found = static_cast<int>(i);
            break;
        }
    }
    if (found < 0) return 0;
    mg->star_center_idx = found;
    return 1;
}

static void adjust_indices_for_rope(RopeSim_internal* si, int rope_idx, int start_idx, int delta, const RopeSim_internal::Ring* skip_ring) {
    if (!si) return;
    for (size_t gi = 0; gi < si->meta_groups.size(); ++gi) {
        MetaGroup* mg = si->meta_groups[gi].get();
        if (!mg) continue;
        for (size_t mi = 0; mi < mg->members.size(); ++mi) {
            auto &m = mg->members[mi];
            if (m.first != rope_idx) continue;
            if (m.second >= start_idx) m.second += delta;
        }
    }
    for (size_t ri = 0; ri < si->rings.size(); ++ri) {
        auto *rg = si->rings[ri].get();
        if (!rg || rg == skip_ring) continue;
        if (rg->rope_idx != rope_idx) continue;
        if (rg->vertex_idx >= start_idx) rg->vertex_idx += delta;
    }
}

static bool rope_vertex_in_use(const RopeSim_internal* si, int rope_idx, int vertex_idx, const RopeSim_internal::Ring* skip_ring, int skip_group_idx, int skip_member_idx) {
    if (!si) return false;
    for (size_t gi = 0; gi < si->meta_groups.size(); ++gi) {
        const MetaGroup* mg = si->meta_groups[gi].get();
        if (!mg) continue;
        for (size_t mi = 0; mi < mg->members.size(); ++mi) {
            if (static_cast<int>(gi) == skip_group_idx && static_cast<int>(mi) == skip_member_idx) continue;
            const auto &m = mg->members[mi];
            if (m.first == rope_idx && m.second == vertex_idx) return true;
        }
    }
    for (size_t ri = 0; ri < si->rings.size(); ++ri) {
        const auto *rg = si->rings[ri].get();
        if (!rg || rg == skip_ring) continue;
        if (rg->rope_idx == rope_idx && rg->vertex_idx == vertex_idx) return true;
    }
    return false;
}

static int rope_sim_remove_vertex_internal(RopeSim_internal* si, int rope_idx, int vertex_idx);

int rope_sim_meta_group_set_member_u(RopeSim* s, int group_idx, int rope_idx, int vertex_idx, float u) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    int found = -1;
    for (size_t i = 0; i < mg->members.size(); ++i) {
        if (mg->members[i].first == rope_idx && mg->members[i].second == vertex_idx) {
            found = static_cast<int>(i);
            break;
        }
    }
    if (found < 0) return 0;
    if (u < 0.0f) u = 0.0f;
    if (u > 1.0f) u = 1.0f;
    if (static_cast<size_t>(found) >= mg->member_u.size()) {
        mg->member_u.resize(mg->members.size(), 0.0f);
    }
    float prev = mg->member_u[static_cast<size_t>(found)];
    mg->member_u[static_cast<size_t>(found)] = u;
    if (std::getenv("NODUS_DEBUG_META")) {
        printf("rope_sim_meta_group_set_member_u: group=%d rope=%d vert=%d u=%.6f prev=%.6f\n",
            group_idx, rope_idx, vertex_idx, u, prev);
    }
    return 1;
}

int rope_sim_meta_group_set_mode(RopeSim* s, int group_idx, int mode) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->mode = mode;
    // debug: meta_group_set_mode logging removed
    // printf("[rope_sim] meta_group_set_mode: sim=%p group_idx=%d mode=%d\n", (void*)s, group_idx, mode);
    return 1;
}

int rope_sim_meta_group_get_mode(RopeSim* s, int group_idx, int* out_mode) {
    if (!s || !out_mode) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    *out_mode = mg->mode;
    return 1;
}

int rope_sim_find_meta_group_with_rope(RopeSim* s, int rope_idx) {
    if (!s) return -1;
    RopeSim_internal* si = to_internal(s);
    for (size_t gi = 0; gi < si->meta_groups.size(); ++gi) {
        MetaGroup* mg = si->meta_groups[gi].get();
        if (!mg) continue;
        for (size_t mi = 0; mi < mg->members.size(); ++mi) {
            if (mg->members[mi].first == rope_idx) return static_cast<int>(gi);
        }
    }
    return -1;
}

// --- Ring implementation: allow rings to slide freely along ropes by
// projecting a stored world position onto the updated rope geometry each step.
static int rope_find_vertex_for_u(const Rope* r, float u, float eps) {
    if (!r) return -1;
    int verts = r->segments + 1;
    if (verts < 2) return -1;
    if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
    for (int vi = 0; vi < verts; ++vi) {
        float vu = (verts > 1) ? (static_cast<float>(vi) / static_cast<float>(verts - 1)) : 0.0f;
        if (std::fabs(vu - u) <= eps) return vi;
    }
    return -1;
}

static void ring_sync_vertex_and_rest(RopeSim_internal* si, RopeSim_internal::Ring* rg, float target_u, int ring_id) {
    if (!si || !rg) return;
    if (rg->rope_idx < 0 || rg->rope_idx >= static_cast<int>(si->ropes.size())) return;
    Rope* r = si->ropes[static_cast<size_t>(rg->rope_idx)].get();
    if (!r) return;
    if (target_u < 0.0f) target_u = 0.0f;
    if (target_u > 1.0f) target_u = 1.0f;
    int verts = r->segments + 1;
    if (verts < 2) return;
    if (r->rest_len_seg.size() != static_cast<size_t>(r->segments)) {
        r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    }
    MetaGroup* bound_mg = nullptr;
    int bound_member = -1;
    if (rg->meta_group_idx >= 0 && rg->meta_group_idx < static_cast<int>(si->meta_groups.size())) {
        bound_mg = si->meta_groups[static_cast<size_t>(rg->meta_group_idx)].get();
        if (bound_mg && rg->member_idx >= 0 && rg->member_idx < static_cast<int>(bound_mg->members.size())) {
            bound_member = rg->member_idx;
        }
    }
    if (rg->vertex_idx < 0 || rg->vertex_idx >= verts) {
        int use_vid = -1;
        if (bound_mg && bound_member >= 0 && bound_mg->members[static_cast<size_t>(bound_member)].first == rg->rope_idx) {
            int candidate = bound_mg->members[static_cast<size_t>(bound_member)].second;
            if (candidate > 0 && candidate < verts - 1) use_vid = candidate;
        }
        if (use_vid < 0) {
            int found = rope_find_vertex_for_u(r, target_u, 1e-3f);
            if (found > 0 && found < verts - 1) use_vid = found;
        }
        if (use_vid < 0) {
            float fidx = target_u * static_cast<float>(verts - 1);
            int seg = static_cast<int>(std::floor(fidx));
            float t = fidx - static_cast<float>(seg);
            if (seg < 0) seg = 0;
            if (seg >= r->segments) { seg = r->segments - 1; t = 1.0f; }
            int inserted = rope_sim_insert_vertex(reinterpret_cast<RopeSim*>(si), rg->rope_idx, seg, t);
            if (inserted >= 0) {
                adjust_indices_for_rope(si, rg->rope_idx, inserted, 1, rg);
                rg->vertex_idx = inserted;
                if (bound_mg && bound_member >= 0) {
                    bound_mg->members[static_cast<size_t>(bound_member)].second = rg->vertex_idx;
                }
            }
        } else {
            rg->vertex_idx = use_vid;
            if (bound_mg && bound_member >= 0) {
                bound_mg->members[static_cast<size_t>(bound_member)].second = rg->vertex_idx;
            }
        }
    }
    verts = r->segments + 1;
    if (rg->vertex_idx <= 0 || rg->vertex_idx >= verts - 1) return;
    float u_left = static_cast<float>(rg->vertex_idx - 1) / static_cast<float>(verts - 1);
    float u_right = static_cast<float>(rg->vertex_idx + 1) / static_cast<float>(verts - 1);
    if (target_u < u_left || target_u > u_right) {
        bool in_use = rope_vertex_in_use(si, rg->rope_idx, rg->vertex_idx, rg, rg->meta_group_idx, rg->member_idx);
        if (!in_use) {
            if (bound_mg && bound_member >= 0) {
                bound_mg->members[static_cast<size_t>(bound_member)].second = -1;
            }
            int old_idx = rg->vertex_idx;
            rope_sim_remove_vertex_internal(si, rg->rope_idx, old_idx);
            adjust_indices_for_rope(si, rg->rope_idx, old_idx + 1, -1, rg);
            r = si->ropes[static_cast<size_t>(rg->rope_idx)].get();
            if (!r) return;
            verts = r->segments + 1;
            float fidx = target_u * static_cast<float>(verts - 1);
            int seg = static_cast<int>(std::floor(fidx));
            float t = fidx - static_cast<float>(seg);
            if (seg < 0) seg = 0;
            if (seg >= r->segments) { seg = r->segments - 1; t = 1.0f; }
            int inserted = rope_sim_insert_vertex(reinterpret_cast<RopeSim*>(si), rg->rope_idx, seg, t);
            if (inserted >= 0) {
                adjust_indices_for_rope(si, rg->rope_idx, inserted, 1, rg);
                rg->vertex_idx = inserted;
                if (bound_mg && bound_member >= 0) {
                    bound_mg->members[static_cast<size_t>(bound_member)].second = rg->vertex_idx;
                }
            }
        } else {
            if (target_u < u_left) target_u = u_left;
            if (target_u > u_right) target_u = u_right;
        }
    }
    verts = r->segments + 1;
    int v_idx = rg->vertex_idx;
    if (v_idx <= 0 || v_idx >= verts - 1) return;
    float u_left2 = static_cast<float>(v_idx - 1) / static_cast<float>(verts - 1);
    float u_right2 = static_cast<float>(v_idx + 1) / static_cast<float>(verts - 1);
    float local = (u_right2 > u_left2) ? ((target_u - u_left2) / (u_right2 - u_left2)) : 0.5f;
    if (local < 0.0f) local = 0.0f;
    if (local > 1.0f) local = 1.0f;
    int left_seg = v_idx - 1;
    int right_seg = v_idx;
    if (left_seg >= 0 && right_seg < r->segments && r->rest_len_seg.size() == static_cast<size_t>(r->segments)) {
        float total = rg->pair_rest_total;
        if (!std::isfinite(total) || total <= 0.0f) {
            total = r->rest_len_seg[static_cast<size_t>(left_seg)] + r->rest_len_seg[static_cast<size_t>(right_seg)];
        }
        if (!std::isfinite(total) || total <= 1e-6f) total = r->rest_len * 2.0f;
        r->rest_len_seg[static_cast<size_t>(left_seg)] = total * local;
        r->rest_len_seg[static_cast<size_t>(right_seg)] = total - r->rest_len_seg[static_cast<size_t>(left_seg)];
        rg->pair_rest_total = total;
    }
    int lv = v_idx - 1;
    int rv = v_idx + 1;
    float tx = r->pos_x[static_cast<size_t>(rv)] - r->pos_x[static_cast<size_t>(lv)];
    float ty = r->pos_y[static_cast<size_t>(rv)] - r->pos_y[static_cast<size_t>(lv)];
    float tz = r->pos_z[static_cast<size_t>(rv)] - r->pos_z[static_cast<size_t>(lv)];
    float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
    if (tlen > 1e-6f) {
        float inv_len = 1.0f / tlen;
        float ux = tx * inv_len;
        float uy = ty * inv_len;
        float uz = tz * inv_len;
        float vx = r->pos_x[static_cast<size_t>(v_idx)] - r->pos_x[static_cast<size_t>(lv)];
        float vy = r->pos_y[static_cast<size_t>(v_idx)] - r->pos_y[static_cast<size_t>(lv)];
        float vz = r->pos_z[static_cast<size_t>(v_idx)] - r->pos_z[static_cast<size_t>(lv)];
        float proj = vx * ux + vy * uy + vz * uz;
        float cur_local = proj * inv_len;
        if (cur_local < 0.0f) cur_local = 0.0f;
        if (cur_local > 1.0f) cur_local = 1.0f;
        float delta = local - cur_local;
        if (std::fabs(delta) > 1e-6f) {
            float oldx = r->pos_x[static_cast<size_t>(v_idx)];
            float oldy = r->pos_y[static_cast<size_t>(v_idx)];
            float oldz = r->pos_z[static_cast<size_t>(v_idx)];
            r->prev_x[static_cast<size_t>(v_idx)] = oldx;
            r->prev_y[static_cast<size_t>(v_idx)] = oldy;
            r->prev_z[static_cast<size_t>(v_idx)] = oldz;
            float shift = delta * tlen;
            r->pos_x[static_cast<size_t>(v_idx)] = oldx + ux * shift;
            r->pos_y[static_cast<size_t>(v_idx)] = oldy + uy * shift;
            r->pos_z[static_cast<size_t>(v_idx)] = oldz + uz * shift;
        }
    }
    rg->u = target_u;
    rg->x = r->pos_x[static_cast<size_t>(rg->vertex_idx)];
    rg->y = r->pos_y[static_cast<size_t>(rg->vertex_idx)];
    rg->z = r->pos_z[static_cast<size_t>(rg->vertex_idx)];
}
int rope_sim_create_ring(RopeSim* s, int rope_idx, float u) {
    if (!s) return -1;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return -1;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return -1;
    int verts = r->segments + 1;
    if (verts <= 0) return -1;
    if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
    auto ring = std::make_unique<RopeSim_internal::Ring>();
    ring->rope_idx = rope_idx;
    ring->u = u;
    // compute initial world pos by interpolating rope vertices
    float fidx = u * static_cast<float>(verts - 1);
    int i0 = static_cast<int>(std::floor(fidx));
    if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
    int i1 = i0 + 1;
    float local_t = fidx - static_cast<float>(i0);
    ring->x = r->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_x[static_cast<size_t>(i1)] * local_t;
    ring->y = r->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_y[static_cast<size_t>(i1)] * local_t;
    ring->z = r->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_z[static_cast<size_t>(i1)] * local_t;
    ring->prev_x = ring->x; ring->prev_y = ring->y; ring->prev_z = ring->z;
    ring->vertex_idx = rope_find_vertex_for_u(r, u, 1e-3f);
    if (ring->vertex_idx <= 0 || ring->vertex_idx >= verts - 1) {
        float fidx = u * static_cast<float>(verts - 1);
        int seg = static_cast<int>(std::floor(fidx));
        float t = fidx - static_cast<float>(seg);
        if (seg < 0) seg = 0;
        if (seg >= r->segments) { seg = r->segments - 1; t = 1.0f; }
        int inserted = rope_sim_insert_vertex(s, rope_idx, seg, t);
        if (inserted >= 0) {
            adjust_indices_for_rope(si, rope_idx, inserted, 1, nullptr);
            ring->vertex_idx = inserted;
        }
    }
    if (ring->vertex_idx > 0 && ring->vertex_idx < (r->segments)) {
        int left_seg = ring->vertex_idx - 1;
        int right_seg = ring->vertex_idx;
        if (r->rest_len_seg.size() == static_cast<size_t>(r->segments)) {
            ring->pair_rest_total = r->rest_len_seg[static_cast<size_t>(left_seg)] +
                                    r->rest_len_seg[static_cast<size_t>(right_seg)];
        }
    }
    int id = static_cast<int>(si->rings.size());
    si->rings.push_back(std::move(ring));
    rope_track_ring(si, rope_idx, id);
    // debug: create_ring logging removed
    // printf("[rope_sim] create_ring id=%d rope=%d u=%f\n", id, rope_idx, u);
    return id;
}

int rope_sim_destroy_ring(RopeSim* s, int ring_id) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
    auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
    if (rg) {
        rope_untrack_ring(si, rg->rope_idx, ring_id);
    }
    si->rings[static_cast<size_t>(ring_id)].reset();
    return 1;
}

int rope_sim_get_ring_u(RopeSim* s, int ring_id, float* out_u) {
    if (!s || !out_u) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
    auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
    if (!rg) return 0;
    *out_u = rg->u;
    return 1;
}

int rope_sim_get_ring_rope_index(RopeSim* s, int ring_id, int* out_rope_idx) {
    if (!s || !out_rope_idx) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
    auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
    if (!rg) return 0;
    *out_rope_idx = rg->rope_idx;
    return 1;
}

int rope_sim_get_ring_count(RopeSim* s) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    return static_cast<int>(si->rings.size());
}

int rope_sim_set_ring_target(RopeSim* s, int ring_id, float target_u, float speed) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
    auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
    if (!rg) return 0;
    if (target_u < 0.0f) target_u = 0.0f; if (target_u > 1.0f) target_u = 1.0f;
    rg->u = target_u;
    // update world position immediately to match target
    int rope_idx = rg->rope_idx;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 1;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 1;
    int verts = r->segments + 1;
    if (verts < 2) return 1;
    float fidx = rg->u * static_cast<float>(verts - 1);
    int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
    int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
    rg->x = r->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_x[static_cast<size_t>(i1)] * local_t;
    rg->y = r->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_y[static_cast<size_t>(i1)] * local_t;
    rg->z = r->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_z[static_cast<size_t>(i1)] * local_t;
    (void)speed;
    return 1;
}

int rope_sim_bind_ring_to_member(RopeSim* s, int ring_id, int group_idx, int member_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
    auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
    if (!rg) return 0;
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    if (member_idx < 0 || member_idx >= static_cast<int>(mg->members.size())) return 0;
    rg->meta_group_idx = group_idx;
    rg->member_idx = member_idx;
    rg->u_vel = 0.0f;
    if (mg->members[static_cast<size_t>(member_idx)].first == rg->rope_idx) {
        rg->vertex_idx = mg->members[static_cast<size_t>(member_idx)].second;
        Rope* r = si->ropes[static_cast<size_t>(rg->rope_idx)].get();
        if (r && r->rest_len_seg.size() == static_cast<size_t>(r->segments) &&
            rg->vertex_idx > 0 && rg->vertex_idx < r->segments) {
            int left_seg = rg->vertex_idx - 1;
            int right_seg = rg->vertex_idx;
            rg->pair_rest_total = r->rest_len_seg[static_cast<size_t>(left_seg)] +
                                  r->rest_len_seg[static_cast<size_t>(right_seg)];
        }
    }
    if (static_cast<size_t>(member_idx) >= mg->member_u.size()) {
        mg->member_u.resize(mg->members.size(), rg->u);
    }
    float mu = mg->member_u[static_cast<size_t>(member_idx)];
    if (!(mu >= 0.0f && mu <= 1.0f)) mu = rg->u;
    if (mu < 0.0f) mu = 0.0f;
    if (mu > 1.0f) mu = 1.0f;
    rg->u = mu;
    mg->member_u[static_cast<size_t>(member_idx)] = rg->u;
    // Update ring world position to match the bound member's u.
    if (rg->rope_idx >= 0 && rg->rope_idx < static_cast<int>(si->ropes.size())) {
        Rope* r = si->ropes[static_cast<size_t>(rg->rope_idx)].get();
        if (r) {
            int verts = r->segments + 1;
            if (verts >= 2) {
                float fidx = rg->u * static_cast<float>(verts - 1);
                int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
                int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
                rg->x = r->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_x[static_cast<size_t>(i1)] * local_t;
                rg->y = r->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_y[static_cast<size_t>(i1)] * local_t;
                rg->z = r->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_z[static_cast<size_t>(i1)] * local_t;
                rg->prev_x = rg->x; rg->prev_y = rg->y; rg->prev_z = rg->z;
            }
        }
    }
    ring_sync_vertex_and_rest(si, rg, rg->u, ring_id);
    return 1;
}

int rope_sim_meta_group_enable_edge_springs(RopeSim* s, int group_idx, float min_rest, float reduce_rate) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->edges.clear();
    size_t n = mg->members.size();
    if (n < 2) return 1; // nothing to connect
    if (mg->mode == 2) {
        // dense: if a star center is set, connect it to all other members
        // that are on different ropes (keeps stem on its own rope).
        if (mg->star_center_idx >= 0 && mg->star_center_idx < static_cast<int>(n)) {
            int center = mg->star_center_idx;
            for (size_t i = 0; i < n; ++i) {
                if (static_cast<int>(i) == center) continue;
                mg->edges.emplace_back(center, static_cast<int>(i));
            }
        } else {
            // fallback: full dense mesh
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = i + 1; j < n; ++j) mg->edges.emplace_back(static_cast<int>(i), static_cast<int>(j));
            }
        }
    } else {
        // chain connections
        for (size_t i = 0; i + 1 < n; ++i) mg->edges.emplace_back(static_cast<int>(i), static_cast<int>(i+1));
        if (mg->mode == 1) {
            // closed loop: add last->first
            mg->edges.emplace_back(static_cast<int>(n-1), 0);
        }
    }
    mg->edges_enabled = true;
    mg->min_rest = min_rest;
    mg->reduce_rate = reduce_rate;
    mg->edge_rest.clear();
    mg->edge_strength.clear();
    // print created springs for debug
    for (const auto &e : mg->edges) {
        int ia = e.first;
        int ib = e.second;
        if (ia < 0 || ib < 0) continue;
        if (ia >= static_cast<int>(mg->members.size()) || ib >= static_cast<int>(mg->members.size())) continue;
        auto a = mg->members[static_cast<size_t>(ia)];
        auto b = mg->members[static_cast<size_t>(ib)];
        float ax=0, ay=0, az=0, bx=0, by=0, bz=0;
        // sample world positions using stored member_u where possible
        if (a.first >= 0 && a.first < static_cast<int>(si->ropes.size())) {
            Rope* ra = si->ropes[static_cast<size_t>(a.first)].get();
            if (ra) {
                int verts = ra->segments + 1;
                float fu = 0.0f;
                if (static_cast<size_t>(ia) < mg->member_u.size()) fu = mg->member_u[static_cast<size_t>(ia)];
                float fidx = fu * static_cast<float>(std::max(1, verts - 1));
                int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
                int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
                ax = ra->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + ra->pos_x[static_cast<size_t>(i1)] * local_t;
                ay = ra->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + ra->pos_y[static_cast<size_t>(i1)] * local_t;
                az = ra->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + ra->pos_z[static_cast<size_t>(i1)] * local_t;
            }
        }
        if (b.first >= 0 && b.first < static_cast<int>(si->ropes.size())) {
            Rope* rb = si->ropes[static_cast<size_t>(b.first)].get();
            if (rb) {
                int verts = rb->segments + 1;
                float fu = 0.0f;
                if (static_cast<size_t>(ib) < mg->member_u.size()) fu = mg->member_u[static_cast<size_t>(ib)];
                float fidx = fu * static_cast<float>(std::max(1, verts - 1));
                int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
                int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
                bx = rb->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + rb->pos_x[static_cast<size_t>(i1)] * local_t;
                by = rb->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + rb->pos_y[static_cast<size_t>(i1)] * local_t;
                bz = rb->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + rb->pos_z[static_cast<size_t>(i1)] * local_t;
            }
        }
        float dx = bx - ax; float dy = by - ay; float dz = bz - az;
        float d = std::sqrt(dx*dx + dy*dy + dz*dz);
        // debug: created spring logging removed
        // printf("[rope_sim] created spring: %d.%d -> %d.%d rest=%f\n", a.first, a.second, b.first, b.second, d);
        bool is_star = (mg->mode == 2 && mg->star_center_idx >= 0);
        float init_rest = std::max(d, mg->min_rest);
        float strength = is_star ? 2.0f : 1.2f; // star links need stronger pull
        if (is_star) {
            // Start at actual geometric distance; decay handles the contraction.
            init_rest = std::max(mg->min_rest, d);
        }
        // If this edge touches the dangling/widget rope, give it a longer
        // initial rest and higher stiffness so widget pulls feel stronger.
        if (mg->dangling_rope >= 0) {
            if (a.first == mg->dangling_rope || b.first == mg->dangling_rope) {
                init_rest = std::max(init_rest * 1.5f, mg->min_rest);
                strength = 1.6f;
            }
        }
        mg->edge_rest.push_back(init_rest);
        mg->edge_strength.push_back(strength);
        if (is_star && std::getenv("NODUS_DEBUG_META")) {
            printf("rope_sim_meta_group_enable_edge_springs: group=%d edge=%zu dist=%.3f init_rest=%.3f min_rest=%.3f reduce_rate=%.3f\n",
                group_idx, mg->edge_rest.size() - 1, d, init_rest, mg->min_rest, mg->reduce_rate);
        }
    }
    return 1;
}

int rope_sim_meta_group_disable_edge_springs(RopeSim* s, int group_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->edges_enabled = false;
    mg->edges.clear();
    mg->edge_rest.clear();
    mg->edge_strength.clear();
    return 1;
}

int rope_sim_meta_group_get_member_world_pos(RopeSim* s, int group_idx, int member_idx, float* out_xyz) {
    if (!s || !out_xyz) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    if (member_idx < 0 || member_idx >= static_cast<int>(mg->members.size())) return 0;
    auto m = mg->members[static_cast<size_t>(member_idx)];
    int rope_idx = m.first;
    int vert_idx = m.second;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return 0;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return 0;
    int verts = r->segments + 1;
    if (verts < 2) return 0;
    float u = 0.0f;
    if (static_cast<size_t>(member_idx) < mg->member_u.size()) u = mg->member_u[static_cast<size_t>(member_idx)];
    else { if (vert_idx >= 0 && vert_idx < verts) u = static_cast<float>(vert_idx) / static_cast<float>(verts - 1); }
    if (u < 0.0f) u = 0.0f; if (u > 1.0f) u = 1.0f;
    float fidx = u * static_cast<float>(verts - 1);
    int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
    int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
    out_xyz[0] = r->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_x[static_cast<size_t>(i1)] * local_t;
    out_xyz[1] = r->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_y[static_cast<size_t>(i1)] * local_t;
    out_xyz[2] = r->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + r->pos_z[static_cast<size_t>(i1)] * local_t;
    return 1;
}

int rope_sim_meta_group_set_dangling_rope(RopeSim* s, int group_idx, int rope_idx) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->dangling_rope = rope_idx;
    rope_track_meta_group(si, rope_idx, group_idx);
    // make the dangling rope slightly elastic so it can deform naturally
    if (rope_idx >= 0 && rope_idx < static_cast<int>(si->ropes.size())) {
        Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
        if (r) {
            // ensure slight slack and recompute rest_len
            float new_slack = std::max(r->slack, 0.08f);
            if (new_slack > r->slack) {
                r->slack = new_slack;
                // recompute rest_len from endpoints
                float dx = r->bx - r->ax;
                float dy = r->by - r->ay;
                float dz = r->bz - r->az;
                float total = std::sqrt(dx*dx + dy*dy + dz*dz);
                int segs = std::max(1, r->segments);
                float seglen = total / float(segs);
                r->rest_len = seglen * (1.0f + r->slack);
                r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
            }
        }
    }
    return 1;
}

// --- Minimal stubs for remaining APIs (to satisfy linking) ----------
int rope_sim_insert_vertex(RopeSim* s, int rope_idx, int seg_index, float t) {
    if (!s) return -1;
    RopeSim_internal* si = to_internal(s);
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return -1;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return -1;
    if (r->segments < 1) return -1;
    if (seg_index < 0) seg_index = 0;
    if (seg_index >= r->segments) seg_index = r->segments - 1;
    if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;

    int i0 = seg_index;
    int i1 = seg_index + 1;
    float x0 = r->pos_x[static_cast<size_t>(i0)];
    float y0 = r->pos_y[static_cast<size_t>(i0)];
    float z0 = r->pos_z[static_cast<size_t>(i0)];
    float x1 = r->pos_x[static_cast<size_t>(i1)];
    float y1 = r->pos_y[static_cast<size_t>(i1)];
    float z1 = r->pos_z[static_cast<size_t>(i1)];
    float nx = x0 * (1.0f - t) + x1 * t;
    float ny = y0 * (1.0f - t) + y1 * t;
    float nz = z0 * (1.0f - t) + z1 * t;

    size_t insert_at = static_cast<size_t>(i1);
    auto insert_val = [&](std::vector<float>& v, float val) {
        if (insert_at >= v.size()) v.push_back(val);
        else v.insert(v.begin() + static_cast<ptrdiff_t>(insert_at), val);
    };
    insert_val(r->pos_x, nx); insert_val(r->pos_y, ny); insert_val(r->pos_z, nz);
    insert_val(r->prev_x, nx); insert_val(r->prev_y, ny); insert_val(r->prev_z, nz);

    // Update per-segment rest lengths: split the segment we inserted into.
    if (r->rest_len_seg.size() == static_cast<size_t>(r->segments)) {
        float base = r->rest_len_seg[static_cast<size_t>(seg_index)];
        float left = base * t;
        float right = base * (1.0f - t);
        r->rest_len_seg[static_cast<size_t>(seg_index)] = left;
        r->rest_len_seg.insert(r->rest_len_seg.begin() + static_cast<ptrdiff_t>(seg_index + 1), right);
    }
    r->segments += 1;
    if (r->rest_len_seg.size() != static_cast<size_t>(r->segments)) {
        // Recompute rest lengths uniformly if the per-seg array isn't aligned.
        float dx = r->bx - r->ax;
        float dy = r->by - r->ay;
        float dz = r->bz - r->az;
        float total = std::sqrt(dx*dx + dy*dy + dz*dz);
        float seglen = total / std::max(1, r->segments);
        r->rest_len = seglen * (1.0f + r->slack);
        r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    } else {
        // Update fallback rest_len to match the average segment length.
        float sum = 0.0f;
        for (float v : r->rest_len_seg) sum += v;
        r->rest_len = (r->segments > 0) ? (sum / static_cast<float>(r->segments)) : 0.0f;
    }

    return static_cast<int>(insert_at);
}

static int rope_sim_remove_vertex_internal(RopeSim_internal* si, int rope_idx, int vertex_idx) {
    if (!si) return -1;
    if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) return -1;
    Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
    if (!r) return -1;
    int verts = r->segments + 1;
    if (vertex_idx <= 0 || vertex_idx >= verts - 1) return -1;
    auto erase_at = [vertex_idx](std::vector<float>& v) {
        v.erase(v.begin() + static_cast<ptrdiff_t>(vertex_idx));
    };
    erase_at(r->pos_x); erase_at(r->pos_y); erase_at(r->pos_z);
    erase_at(r->prev_x); erase_at(r->prev_y); erase_at(r->prev_z);
    if (r->rest_len_seg.size() == static_cast<size_t>(r->segments)) {
        int seg_left = vertex_idx - 1;
        float merged = r->rest_len_seg[static_cast<size_t>(seg_left)];
        if (vertex_idx < static_cast<int>(r->rest_len_seg.size())) {
            merged += r->rest_len_seg[static_cast<size_t>(vertex_idx)];
        }
        r->rest_len_seg[static_cast<size_t>(seg_left)] = merged;
        if (vertex_idx < static_cast<int>(r->rest_len_seg.size())) {
            r->rest_len_seg.erase(r->rest_len_seg.begin() + static_cast<ptrdiff_t>(vertex_idx));
        }
    }
    r->segments -= 1;
    if (r->segments < 1) r->segments = 1;
    if (r->rest_len_seg.size() != static_cast<size_t>(r->segments)) {
        float dx = r->bx - r->ax;
        float dy = r->by - r->ay;
        float dz = r->bz - r->az;
        float total = std::sqrt(dx*dx + dy*dy + dz*dz);
        float seglen = total / std::max(1, r->segments);
        r->rest_len = seglen * (1.0f + r->slack);
        r->rest_len_seg.assign(static_cast<size_t>(r->segments), r->rest_len);
    } else {
        float sum = 0.0f;
        for (float v : r->rest_len_seg) sum += v;
        r->rest_len = (r->segments > 0) ? (sum / static_cast<float>(r->segments)) : 0.0f;
    }
    return vertex_idx;
}

int rope_sim_set_rope_radius(RopeSim* s, int rope_idx, float radius) {
    (void)s; (void)rope_idx; (void)radius; return 1;
}
int rope_sim_get_rope_radius(RopeSim* s, int rope_idx, float* out_radius) {
    if (!s || !out_radius) return 0; *out_radius = 0.0f; return 1;
}

int rope_sim_modify_rest_length(RopeSim* s, int rope_idx, float delta) {
    (void)s; (void)rope_idx; (void)delta; return 0;
}
int rope_sim_set_rope_rest_target(RopeSim* s, int rope_idx, float target_rest, float rate, float delay) {
    (void)s; (void)rope_idx; (void)target_rest; (void)rate; (void)delay; return 0;
}
int rope_sim_clear_rope_rest_target(RopeSim* s, int rope_idx) { (void)s; (void)rope_idx; return 0; }
int rope_sim_get_rope_rest_length(RopeSim* s, int rope_idx, float* out_rest) { if (!s || !out_rest) return 0; *out_rest = 0.0f; return 1; }

// ring stubs removed; real implementations are provided above

int rope_sim_create_dangling_widget(RopeSim* s, int rope_idx, int vertex_idx, unsigned int widget_type) { (void)s; (void)rope_idx; (void)vertex_idx; (void)widget_type; return -1; }
int rope_sim_destroy_dangling_widget(RopeSim* s, int widget_id) { (void)s; (void)widget_id; return 0; }
int rope_sim_get_widget_position(RopeSim* s, int widget_id, float* out_xyz) { if (!out_xyz) return 0; out_xyz[0]=out_xyz[1]=out_xyz[2]=0.0f; return 0; }
int rope_sim_set_widget_mass(RopeSim* s, int widget_id, float mass) { (void)s; (void)widget_id; (void)mass; return 0; }

// --- RopeSim serialization -------------------------------------------------
static bool rope_sim_read_bytes(const char* &p, const char* end, void* out, size_t len) {
    if (!out || p + len > end) return false;
    memcpy(out, p, len);
    p += static_cast<ptrdiff_t>(len);
    return true;
}

static bool rope_sim_write_bytes(char* &p, char* end, const void* src, size_t len) {
    if (!src || p + len > end) return false;
    memcpy(p, src, len);
    p += static_cast<ptrdiff_t>(len);
    return true;
}

int rope_sim_serialized_size(RopeSim* s) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    const int32_t rope_count = static_cast<int32_t>(s->ropes.size());
    int32_t need = 0;
    need += 8; // magic
    need += 4; // version
    need += 4; // max_ropes
    need += 4; // max_segments_per_rope
    need += 4; // rope_count
    for (int32_t i = 0; i < rope_count; ++i) {
        Rope* r = s->ropes[static_cast<size_t>(i)].get();
        if (!r) return 0;
        int32_t verts = r->segments + 1;
        need += 4; // segments
        need += 4; // slack
        need += 4; // rest_len
        need += static_cast<int32_t>(r->segments) * 4; // rest_len_seg
        need += 6 * 4; // endpoints
        need += 4; // owned_meta_group
        need += 4; // is_meta_rope flag
        need += 4; // meta_vertex_count
        need += static_cast<int32_t>(r->meta_vertices.size()) * 4;
        need += verts * 6 * 4; // pos_xyz + prev_xyz
    }
    int32_t mg_count = static_cast<int32_t>(si->meta_groups.size());
    need += 4; // meta_group_count
    for (int32_t gi = 0; gi < mg_count; ++gi) {
        MetaGroup* mg = si->meta_groups[static_cast<size_t>(gi)].get();
        need += 4; // present flag
        if (!mg) continue;
        int32_t member_count = static_cast<int32_t>(mg->members.size());
        int32_t edge_count = static_cast<int32_t>(mg->edges.size());
        need += 4; // pressure
        need += 4; // mode
        need += 4; // edges_enabled
        need += 4; // min_rest
        need += 4; // reduce_rate
        need += 4; // dangling_rope
        need += 4; // star_center_idx
        need += 4; // member_count
        need += member_count * (4 + 4); // members
        need += member_count * 4; // member_u
        need += 4; // edge_count
        need += edge_count * (4 + 4); // edges
        need += edge_count * 4; // edge_rest
        need += edge_count * 4; // edge_strength
    }
    int32_t ring_count = static_cast<int32_t>(si->rings.size());
    need += 4; // ring_count
    for (int32_t ri = 0; ri < ring_count; ++ri) {
        RopeSim_internal::Ring* ring = si->rings[static_cast<size_t>(ri)].get();
        need += 4; // present flag
        if (!ring) continue;
        need += 4; // rope_idx
        need += 4; // meta_group_idx
        need += 4; // member_idx
        need += 4; // vertex_idx
        need += 4; // u
        need += 4; // u_vel
        need += 4; // pair_rest_total
        need += 6 * 4; // x/y/z + prev_x/prev_y/prev_z
    }
    return need;
}

int rope_sim_serialize(RopeSim* s, char* out_buf, int out_len) {
    if (!s) return 0;
    int32_t need = rope_sim_serialized_size(s);
    if (!out_buf) return need;
    if (out_len < need) return 0;
    RopeSim_internal* si = to_internal(s);
    char* p = out_buf;
    char* end = out_buf + out_len;
    const char magic[8] = {'R','P','S','I','M','0','0','1'};
    const int32_t version = 7;
    if (!rope_sim_write_bytes(p, end, magic, sizeof(magic))) return 0;
    if (!rope_sim_write_bytes(p, end, &version, sizeof(version))) return 0;
    int32_t max_ropes = s->max_ropes;
    int32_t max_segments = s->max_segments_per_rope;
    int32_t rope_count = static_cast<int32_t>(s->ropes.size());
    if (!rope_sim_write_bytes(p, end, &max_ropes, sizeof(max_ropes))) return 0;
    if (!rope_sim_write_bytes(p, end, &max_segments, sizeof(max_segments))) return 0;
    if (!rope_sim_write_bytes(p, end, &rope_count, sizeof(rope_count))) return 0;
    for (int32_t i = 0; i < rope_count; ++i) {
        Rope* r = s->ropes[static_cast<size_t>(i)].get();
        if (!r) return 0;
        int32_t segments = r->segments;
        if (!rope_sim_write_bytes(p, end, &segments, sizeof(segments))) return 0;
        if (!rope_sim_write_bytes(p, end, &r->slack, sizeof(r->slack))) return 0;
        if (!rope_sim_write_bytes(p, end, &r->rest_len, sizeof(r->rest_len))) return 0;
        if (r->rest_len_seg.size() != static_cast<size_t>(segments)) {
            r->rest_len_seg.assign(static_cast<size_t>(segments), r->rest_len);
        }
        for (int32_t si = 0; si < segments; ++si) {
            float seg_len = r->rest_len_seg[static_cast<size_t>(si)];
            if (!rope_sim_write_bytes(p, end, &seg_len, sizeof(seg_len))) return 0;
        }
        float endpoints[6] = {r->ax, r->ay, r->az, r->bx, r->by, r->bz};
        if (!rope_sim_write_bytes(p, end, endpoints, sizeof(endpoints))) return 0;
        int32_t owned = r->owned_meta_group;
        if (!rope_sim_write_bytes(p, end, &owned, sizeof(owned))) return 0;
        int32_t meta_flag = r->is_meta_rope ? 1 : 0;
        if (!rope_sim_write_bytes(p, end, &meta_flag, sizeof(meta_flag))) return 0;
        int32_t meta_count = static_cast<int32_t>(r->meta_vertices.size());
        if (!rope_sim_write_bytes(p, end, &meta_count, sizeof(meta_count))) return 0;
        for (int32_t vi = 0; vi < meta_count; ++vi) {
            int32_t mv = r->meta_vertices[static_cast<size_t>(vi)];
            if (!rope_sim_write_bytes(p, end, &mv, sizeof(mv))) return 0;
        }
        int verts = r->segments + 1;
        for (int vi = 0; vi < verts; ++vi) {
            float vdata[6] = {r->pos_x[static_cast<size_t>(vi)], r->pos_y[static_cast<size_t>(vi)], r->pos_z[static_cast<size_t>(vi)],
                              r->prev_x[static_cast<size_t>(vi)], r->prev_y[static_cast<size_t>(vi)], r->prev_z[static_cast<size_t>(vi)]};
            if (!rope_sim_write_bytes(p, end, vdata, sizeof(vdata))) return 0;
        }
    }
    int32_t mg_count = static_cast<int32_t>(si->meta_groups.size());
    if (!rope_sim_write_bytes(p, end, &mg_count, sizeof(mg_count))) return 0;
    for (int32_t gi = 0; gi < mg_count; ++gi) {
        MetaGroup* mg = si->meta_groups[static_cast<size_t>(gi)].get();
        int32_t present = mg ? 1 : 0;
        if (!rope_sim_write_bytes(p, end, &present, sizeof(present))) return 0;
        if (!mg) continue;
        int32_t mode = mg->mode;
        int32_t edges_enabled = mg->edges_enabled ? 1 : 0;
        int32_t dangling_rope = mg->dangling_rope;
        int32_t member_count = static_cast<int32_t>(mg->members.size());
        int32_t edge_count = static_cast<int32_t>(mg->edges.size());
        if (!rope_sim_write_bytes(p, end, &mg->pressure, sizeof(mg->pressure))) return 0;
        if (!rope_sim_write_bytes(p, end, &mode, sizeof(mode))) return 0;
        if (!rope_sim_write_bytes(p, end, &edges_enabled, sizeof(edges_enabled))) return 0;
        if (!rope_sim_write_bytes(p, end, &mg->min_rest, sizeof(mg->min_rest))) return 0;
        if (!rope_sim_write_bytes(p, end, &mg->reduce_rate, sizeof(mg->reduce_rate))) return 0;
        if (!rope_sim_write_bytes(p, end, &dangling_rope, sizeof(dangling_rope))) return 0;
        int32_t star_center_idx = mg->star_center_idx;
        if (!rope_sim_write_bytes(p, end, &star_center_idx, sizeof(star_center_idx))) return 0;
        if (!rope_sim_write_bytes(p, end, &member_count, sizeof(member_count))) return 0;
        for (int32_t mi = 0; mi < member_count; ++mi) {
            int32_t rope_idx = mg->members[static_cast<size_t>(mi)].first;
            int32_t vertex_idx = mg->members[static_cast<size_t>(mi)].second;
            if (!rope_sim_write_bytes(p, end, &rope_idx, sizeof(rope_idx))) return 0;
            if (!rope_sim_write_bytes(p, end, &vertex_idx, sizeof(vertex_idx))) return 0;
        }
        for (int32_t mi = 0; mi < member_count; ++mi) {
            float u = (static_cast<size_t>(mi) < mg->member_u.size()) ? mg->member_u[static_cast<size_t>(mi)] : 0.0f;
            if (!rope_sim_write_bytes(p, end, &u, sizeof(u))) return 0;
        }
        if (!rope_sim_write_bytes(p, end, &edge_count, sizeof(edge_count))) return 0;
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            int32_t a = mg->edges[static_cast<size_t>(ei)].first;
            int32_t b = mg->edges[static_cast<size_t>(ei)].second;
            if (!rope_sim_write_bytes(p, end, &a, sizeof(a))) return 0;
            if (!rope_sim_write_bytes(p, end, &b, sizeof(b))) return 0;
        }
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            float rest = (static_cast<size_t>(ei) < mg->edge_rest.size()) ? mg->edge_rest[static_cast<size_t>(ei)] : 0.0f;
            if (!rope_sim_write_bytes(p, end, &rest, sizeof(rest))) return 0;
        }
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            float strength = (static_cast<size_t>(ei) < mg->edge_strength.size()) ? mg->edge_strength[static_cast<size_t>(ei)] : 0.0f;
            if (!rope_sim_write_bytes(p, end, &strength, sizeof(strength))) return 0;
        }
    }
    int32_t ring_count = static_cast<int32_t>(si->rings.size());
    if (!rope_sim_write_bytes(p, end, &ring_count, sizeof(ring_count))) return 0;
    for (int32_t ri = 0; ri < ring_count; ++ri) {
        RopeSim_internal::Ring* ring = si->rings[static_cast<size_t>(ri)].get();
        int32_t present = ring ? 1 : 0;
        if (!rope_sim_write_bytes(p, end, &present, sizeof(present))) return 0;
        if (!ring) continue;
        if (!rope_sim_write_bytes(p, end, &ring->rope_idx, sizeof(ring->rope_idx))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->meta_group_idx, sizeof(ring->meta_group_idx))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->member_idx, sizeof(ring->member_idx))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->vertex_idx, sizeof(ring->vertex_idx))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->u, sizeof(ring->u))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->u_vel, sizeof(ring->u_vel))) return 0;
        if (!rope_sim_write_bytes(p, end, &ring->pair_rest_total, sizeof(ring->pair_rest_total))) return 0;
        float rpos[6] = {ring->x, ring->y, ring->z, ring->prev_x, ring->prev_y, ring->prev_z};
        if (!rope_sim_write_bytes(p, end, rpos, sizeof(rpos))) return 0;
    }
    return need;
}

RopeSim* rope_sim_deserialize(const char* in_buf, int in_len) {
    if (!in_buf || in_len <= 0) return nullptr;
    const char* p = in_buf;
    const char* end = in_buf + in_len;
    char magic[8];
    int32_t version = 0;
    if (!rope_sim_read_bytes(p, end, magic, sizeof(magic))) return nullptr;
    if (memcmp(magic, "RPSIM001", 8) != 0) return nullptr;
    if (!rope_sim_read_bytes(p, end, &version, sizeof(version))) return nullptr;
    if (version != 7) return nullptr;
    int32_t max_ropes = 0;
    int32_t max_segments = 0;
    int32_t rope_count = 0;
    if (!rope_sim_read_bytes(p, end, &max_ropes, sizeof(max_ropes))) return nullptr;
    if (!rope_sim_read_bytes(p, end, &max_segments, sizeof(max_segments))) return nullptr;
    if (!rope_sim_read_bytes(p, end, &rope_count, sizeof(rope_count))) return nullptr;
    if (max_ropes <= 0 || max_segments <= 0 || rope_count < 0 || rope_count > max_ropes) return nullptr;
    RopeSim* sim = rope_sim_create(max_ropes, max_segments);
    if (!sim) return nullptr;
    RopeSim_internal* si = to_internal(sim);
    si->ropes.clear();
    si->ropes.reserve(static_cast<size_t>(max_ropes));
    for (int32_t i = 0; i < rope_count; ++i) {
        int32_t segments = 0;
        float slack = 0.0f;
        float rest_len = 0.0f;
        float endpoints[6] = {};
        if (!rope_sim_read_bytes(p, end, &segments, sizeof(segments))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &slack, sizeof(slack))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &rest_len, sizeof(rest_len))) { rope_sim_destroy(sim); return nullptr; }
        if (segments < 1 || segments > max_segments) { rope_sim_destroy(sim); return nullptr; }
        std::vector<float> rest_len_seg;
        rest_len_seg.resize(static_cast<size_t>(std::max(1, segments)));
        for (int32_t si = 0; si < segments; ++si) {
            float seg_len = 0.0f;
            if (!rope_sim_read_bytes(p, end, &seg_len, sizeof(seg_len))) { rope_sim_destroy(sim); return nullptr; }
            rest_len_seg[static_cast<size_t>(si)] = seg_len;
        }
        if (!rope_sim_read_bytes(p, end, endpoints, sizeof(endpoints))) { rope_sim_destroy(sim); return nullptr; }
        int32_t owned_meta_group = -1;
        if (version >= 3) {
            if (!rope_sim_read_bytes(p, end, &owned_meta_group, sizeof(owned_meta_group))) { rope_sim_destroy(sim); return nullptr; }
        }
        int32_t meta_flag = 0;
        int32_t meta_count = 0;
        std::vector<int32_t> meta_verts;
        if (version >= 4) {
            if (!rope_sim_read_bytes(p, end, &meta_flag, sizeof(meta_flag))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &meta_count, sizeof(meta_count))) { rope_sim_destroy(sim); return nullptr; }
            if (meta_count < 0) { rope_sim_destroy(sim); return nullptr; }
            meta_verts.resize(static_cast<size_t>(meta_count));
            for (int32_t mv = 0; mv < meta_count; ++mv) {
                if (!rope_sim_read_bytes(p, end, &meta_verts[static_cast<size_t>(mv)], sizeof(int32_t))) { rope_sim_destroy(sim); return nullptr; }
            }
        }
        if (segments < 1) segments = 1;
        if (segments > max_segments) segments = max_segments;
        auto r = std::make_unique<Rope>();
        r->segments = segments;
        r->slack = slack;
        if (rest_len_seg.size() != static_cast<size_t>(segments)) {
            rest_len_seg.assign(static_cast<size_t>(segments), rest_len);
        }
        r->rest_len_seg.assign(rest_len_seg.begin(), rest_len_seg.end());
        if (!r->rest_len_seg.empty()) {
            float sum = 0.0f;
            for (float v : r->rest_len_seg) sum += v;
            r->rest_len = (r->segments > 0) ? (sum / static_cast<float>(r->segments)) : rest_len;
        } else {
            r->rest_len = rest_len;
        }
        r->ax = endpoints[0]; r->ay = endpoints[1]; r->az = endpoints[2];
        r->bx = endpoints[3]; r->by = endpoints[4]; r->bz = endpoints[5];
        r->owned_meta_group = owned_meta_group;
        r->is_meta_rope = (meta_flag != 0);
        if (!meta_verts.empty()) {
            r->meta_vertices.assign(meta_verts.begin(), meta_verts.end());
        }
        int verts = segments + 1;
        r->pos_x.resize(static_cast<size_t>(verts));
        r->pos_y.resize(static_cast<size_t>(verts));
        r->pos_z.resize(static_cast<size_t>(verts));
        r->prev_x.resize(static_cast<size_t>(verts));
        r->prev_y.resize(static_cast<size_t>(verts));
        r->prev_z.resize(static_cast<size_t>(verts));
        for (int vi = 0; vi < verts; ++vi) {
            float vdata[6] = {};
            if (!rope_sim_read_bytes(p, end, vdata, sizeof(vdata))) { rope_sim_destroy(sim); return nullptr; }
            r->pos_x[static_cast<size_t>(vi)] = vdata[0];
            r->pos_y[static_cast<size_t>(vi)] = vdata[1];
            r->pos_z[static_cast<size_t>(vi)] = vdata[2];
            r->prev_x[static_cast<size_t>(vi)] = vdata[3];
            r->prev_y[static_cast<size_t>(vi)] = vdata[4];
            r->prev_z[static_cast<size_t>(vi)] = vdata[5];
        }
        si->ropes.push_back(std::move(r));
    }
    int32_t mg_count = 0;
    if (!rope_sim_read_bytes(p, end, &mg_count, sizeof(mg_count))) { rope_sim_destroy(sim); return nullptr; }
    if (mg_count < 0) { rope_sim_destroy(sim); return nullptr; }
    si->meta_groups.clear();
    si->meta_groups.reserve(static_cast<size_t>(mg_count));
    for (int32_t gi = 0; gi < mg_count; ++gi) {
        int32_t present = 0;
        if (!rope_sim_read_bytes(p, end, &present, sizeof(present))) { rope_sim_destroy(sim); return nullptr; }
        if (!present) {
            si->meta_groups.push_back(nullptr);
            continue;
        }
        auto mg = std::make_unique<MetaGroup>();
        int32_t mode = 0;
        int32_t edges_enabled = 0;
        int32_t dangling_rope = -1;
        int32_t star_center_idx = -1;
        int32_t member_count = 0;
        int32_t edge_count = 0;
        if (!rope_sim_read_bytes(p, end, &mg->pressure, sizeof(mg->pressure))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mode, sizeof(mode))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &edges_enabled, sizeof(edges_enabled))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mg->min_rest, sizeof(mg->min_rest))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mg->reduce_rate, sizeof(mg->reduce_rate))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &dangling_rope, sizeof(dangling_rope))) { rope_sim_destroy(sim); return nullptr; }
        if (version >= 5) {
            if (!rope_sim_read_bytes(p, end, &star_center_idx, sizeof(star_center_idx))) { rope_sim_destroy(sim); return nullptr; }
        }
        if (!rope_sim_read_bytes(p, end, &member_count, sizeof(member_count))) { rope_sim_destroy(sim); return nullptr; }
        if (member_count < 0) { rope_sim_destroy(sim); return nullptr; }
        mg->mode = mode;
        mg->edges_enabled = edges_enabled != 0;
        mg->dangling_rope = dangling_rope;
        mg->star_center_idx = star_center_idx;
        mg->members.resize(static_cast<size_t>(member_count));
        for (int32_t mi = 0; mi < member_count; ++mi) {
            int32_t rope_idx = -1;
            int32_t vertex_idx = -1;
            if (!rope_sim_read_bytes(p, end, &rope_idx, sizeof(rope_idx))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &vertex_idx, sizeof(vertex_idx))) { rope_sim_destroy(sim); return nullptr; }
            mg->members[static_cast<size_t>(mi)] = std::make_pair(rope_idx, vertex_idx);
        }
        mg->member_u.resize(static_cast<size_t>(member_count));
        for (int32_t mi = 0; mi < member_count; ++mi) {
            float u = 0.0f;
            if (!rope_sim_read_bytes(p, end, &u, sizeof(u))) { rope_sim_destroy(sim); return nullptr; }
            mg->member_u[static_cast<size_t>(mi)] = u;
        }
        if (!rope_sim_read_bytes(p, end, &edge_count, sizeof(edge_count))) { rope_sim_destroy(sim); return nullptr; }
        if (edge_count < 0) { rope_sim_destroy(sim); return nullptr; }
        mg->edges.resize(static_cast<size_t>(edge_count));
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            int32_t a = 0;
            int32_t b = 0;
            if (!rope_sim_read_bytes(p, end, &a, sizeof(a))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &b, sizeof(b))) { rope_sim_destroy(sim); return nullptr; }
            mg->edges[static_cast<size_t>(ei)] = std::make_pair(a, b);
        }
        mg->edge_rest.resize(static_cast<size_t>(edge_count));
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            float rest = 0.0f;
            if (!rope_sim_read_bytes(p, end, &rest, sizeof(rest))) { rope_sim_destroy(sim); return nullptr; }
            mg->edge_rest[static_cast<size_t>(ei)] = rest;
        }
        mg->edge_strength.resize(static_cast<size_t>(edge_count));
        for (int32_t ei = 0; ei < edge_count; ++ei) {
            float strength = 0.0f;
            if (!rope_sim_read_bytes(p, end, &strength, sizeof(strength))) { rope_sim_destroy(sim); return nullptr; }
            mg->edge_strength[static_cast<size_t>(ei)] = strength;
        }
        si->meta_groups.push_back(std::move(mg));
    }
    {
        int32_t ring_count = 0;
        if (!rope_sim_read_bytes(p, end, &ring_count, sizeof(ring_count))) { rope_sim_destroy(sim); return nullptr; }
        if (ring_count < 0) { rope_sim_destroy(sim); return nullptr; }
        si->rings.clear();
        si->rings.reserve(static_cast<size_t>(ring_count));
        for (int32_t ri = 0; ri < ring_count; ++ri) {
            int32_t present = 0;
            if (!rope_sim_read_bytes(p, end, &present, sizeof(present))) { rope_sim_destroy(sim); return nullptr; }
            if (!present) {
                si->rings.push_back(nullptr);
                continue;
            }
            auto ring = std::make_unique<RopeSim_internal::Ring>();
            if (!rope_sim_read_bytes(p, end, &ring->rope_idx, sizeof(ring->rope_idx))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->meta_group_idx, sizeof(ring->meta_group_idx))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->member_idx, sizeof(ring->member_idx))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->vertex_idx, sizeof(ring->vertex_idx))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->u, sizeof(ring->u))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->u_vel, sizeof(ring->u_vel))) { rope_sim_destroy(sim); return nullptr; }
            if (!rope_sim_read_bytes(p, end, &ring->pair_rest_total, sizeof(ring->pair_rest_total))) { rope_sim_destroy(sim); return nullptr; }
            float rpos[6] = {};
            if (!rope_sim_read_bytes(p, end, rpos, sizeof(rpos))) { rope_sim_destroy(sim); return nullptr; }
            ring->x = rpos[0]; ring->y = rpos[1]; ring->z = rpos[2];
            ring->prev_x = rpos[3]; ring->prev_y = rpos[4]; ring->prev_z = rpos[5];
            if (ring->rope_idx < 0 || ring->rope_idx >= static_cast<int32_t>(si->ropes.size())) {
                ring->rope_idx = -1;
                ring->vertex_idx = -1;
            } else {
                Rope* rr = si->ropes[static_cast<size_t>(ring->rope_idx)].get();
                int verts = rr ? (rr->segments + 1) : 0;
                if (ring->vertex_idx < 0 || ring->vertex_idx >= verts) {
                    ring->vertex_idx = -1;
                }
                if (ring->pair_rest_total <= 0.0f && rr && ring->vertex_idx > 0 && ring->vertex_idx < rr->segments &&
                    rr->rest_len_seg.size() == static_cast<size_t>(rr->segments)) {
                    int left_seg = ring->vertex_idx - 1;
                    int right_seg = ring->vertex_idx;
                    ring->pair_rest_total = rr->rest_len_seg[static_cast<size_t>(left_seg)] +
                                            rr->rest_len_seg[static_cast<size_t>(right_seg)];
                }
            }
            if (ring->u < 0.0f) ring->u = 0.0f;
            if (ring->u > 1.0f) ring->u = 1.0f;
            si->rings.push_back(std::move(ring));
        }
    }
    // Rebuild per-rope membership caches so rings/meta groups are first-class on ropes.
    rope_rebuild_memberships(si);
    return sim;
}

// Apply meta-group edge constraints after rope constraints pass
// Simple distance constraints between member vertex positions
static void apply_meta_group_edges(RopeSim_internal* si, float dt) {
    if (!si) return;
    for (size_t gi = 0; gi < si->meta_groups.size(); ++gi) {
        MetaGroup* mg = si->meta_groups[gi].get();
        if (!mg || !mg->edges_enabled) continue;
        // decay rest lengths toward min_rest
        if (mg->reduce_rate > 0.0f && dt > 0.0f) {
            for (size_t ei = 0; ei < mg->edge_rest.size(); ++ei) {
                float &r = mg->edge_rest[ei];
                r = std::max(mg->min_rest, r - mg->reduce_rate * dt);
            }
        }

        const float ring_u_stiff = 40.0f;
        const float ring_u_damp = 8.0f;
        std::vector<int> ring_for_member;
        if (!mg->members.empty()) {
            ring_for_member.assign(mg->members.size(), -1);
            for (size_t ri = 0; ri < si->rings.size(); ++ri) {
                auto *rg = si->rings[ri].get();
                if (!rg) continue;
                if (rg->meta_group_idx != static_cast<int>(gi)) continue;
                if (rg->member_idx < 0 || rg->member_idx >= static_cast<int>(mg->members.size())) continue;
                ring_for_member[static_cast<size_t>(rg->member_idx)] = static_cast<int>(ri);
                if (static_cast<size_t>(rg->member_idx) >= mg->member_u.size()) {
                    mg->member_u.resize(mg->members.size(), rg->u);
                }
                if (rg->u < 0.0f) rg->u = 0.0f;
                if (rg->u > 1.0f) rg->u = 1.0f;
                mg->member_u[static_cast<size_t>(rg->member_idx)] = rg->u;
            }
        }

        auto apply_member_u = [&](int member_idx, float target_u) {
            if (member_idx < 0) return;
            if (target_u < 0.0f) target_u = 0.0f;
            if (target_u > 1.0f) target_u = 1.0f;
            int ring_id = -1;
            if (!ring_for_member.empty() && member_idx < static_cast<int>(ring_for_member.size())) {
                ring_id = ring_for_member[static_cast<size_t>(member_idx)];
            }
            if (ring_id < 0) {
                if (static_cast<size_t>(member_idx) >= mg->member_u.size()) {
                    mg->member_u.resize(mg->members.size(), target_u);
                }
                mg->member_u[static_cast<size_t>(member_idx)] = target_u;
                return;
            }
            auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
            if (!rg) return;
            float du = target_u - rg->u;
            if (dt > 0.0f) {
                rg->u_vel += du * ring_u_stiff * dt;
                float damp = std::exp(-ring_u_damp * dt);
                rg->u_vel *= damp;
                rg->u += rg->u_vel * dt;
            } else {
                rg->u = target_u;
                rg->u_vel = 0.0f;
            }
            if (rg->u < 0.0f) rg->u = 0.0f;
            if (rg->u > 1.0f) rg->u = 1.0f;
            if (static_cast<size_t>(member_idx) >= mg->member_u.size()) {
                mg->member_u.resize(mg->members.size(), rg->u);
            }
            mg->member_u[static_cast<size_t>(member_idx)] = rg->u;
            ring_sync_vertex_and_rest(si, rg, rg->u, ring_id);
        };

        // helper: sample a rope world-position from parametric u and also
        // return integer vertex indices and local t for distributing vertex deltas
        auto sample_rope = [&](Rope* rr, float fu, float &outx, float &outy, float &outz,
                               int &out_i0, int &out_i1, float &out_local_t) {
            outx = outy = outz = 0.0f; out_i0 = out_i1 = 0; out_local_t = 0.0f;
            if (!rr) return;
            int verts = rr->segments + 1;
            if (verts < 2) return;
            if (fu < 0.0f) fu = 0.0f; if (fu > 1.0f) fu = 1.0f;
            float fidx = fu * static_cast<float>(verts - 1);
            int i0 = static_cast<int>(std::floor(fidx)); if (i0 < 0) i0 = 0; if (i0 >= verts-1) i0 = verts-2;
            int i1 = i0 + 1; float local_t = fidx - static_cast<float>(i0);
            out_i0 = i0; out_i1 = i1; out_local_t = local_t;
            outx = rr->pos_x[static_cast<size_t>(i0)] * (1.0f - local_t) + rr->pos_x[static_cast<size_t>(i1)] * local_t;
            outy = rr->pos_y[static_cast<size_t>(i0)] * (1.0f - local_t) + rr->pos_y[static_cast<size_t>(i1)] * local_t;
            outz = rr->pos_z[static_cast<size_t>(i0)] * (1.0f - local_t) + rr->pos_z[static_cast<size_t>(i1)] * local_t;
        };

        // helper: compute a unit tangent at parametric u by sampling +/- small du
        auto compute_tangent = [&](Rope* rr, float fu, float &tx, float &ty, float &tz) {
            tx = ty = tz = 0.0f; if (!rr) return;
            int verts = rr->segments + 1; if (verts < 2) return;
            float du = 1.0f / static_cast<float>(std::max(1, verts - 1));
            float a_fu = fu - 0.5f * du; float b_fu = fu + 0.5f * du;
            if (a_fu < 0.0f) a_fu = 0.0f; if (b_fu > 1.0f) b_fu = 1.0f;
            float ax,ay,az,bx,by,bz; int ia0,ia1,ib0,ib1; float lat, lbt;
            sample_rope(rr, a_fu, ax, ay, az, ia0, ia1, lat);
            sample_rope(rr, b_fu, bx, by, bz, ib0, ib1, lbt);
            tx = bx - ax; ty = by - ay; tz = bz - az;
            float llen = std::sqrt(tx*tx + ty*ty + tz*tz);
            if (llen > 1e-6f) { tx /= llen; ty /= llen; tz /= llen; } else { tx = ty = tz = 0.0f; }
        };

        auto apply_normal = [&](Rope* rr, int rope_idx, int ring_id, int i0, int i1, float local_t,
                                float nx, float ny, float nz) {
            if (!rr) return;
            int verts = rr->segments + 1;
            if (ring_id >= 0 && ring_id < static_cast<int>(si->rings.size())) {
                auto *rg = si->rings[static_cast<size_t>(ring_id)].get();
                if (rg && rg->rope_idx == rope_idx) {
                    int v_idx = rg->vertex_idx;
                    if (v_idx > 0 && v_idx < verts - 1) {
                        rr->pos_x[static_cast<size_t>(v_idx)] += nx;
                        rr->pos_y[static_cast<size_t>(v_idx)] += ny;
                        rr->pos_z[static_cast<size_t>(v_idx)] += nz;
                        return;
                    }
                }
            }
            rr->pos_x[static_cast<size_t>(i0)] += nx * (1.0f - local_t);
            rr->pos_y[static_cast<size_t>(i0)] += ny * (1.0f - local_t);
            rr->pos_z[static_cast<size_t>(i0)] += nz * (1.0f - local_t);
            rr->pos_x[static_cast<size_t>(i1)] += nx * local_t;
            rr->pos_y[static_cast<size_t>(i1)] += ny * local_t;
            rr->pos_z[static_cast<size_t>(i1)] += nz * local_t;
        };

        int stem_member = -1;
        if (mg->mode == 2 && mg->star_center_idx >= 0 && mg->star_center_idx < static_cast<int>(mg->members.size())) {
            int center = mg->star_center_idx;
            int center_rope = mg->members[static_cast<size_t>(center)].first;
            float best = 1e9f;
            for (size_t i = 0; i < mg->members.size(); ++i) {
                if (static_cast<int>(i) == center) continue;
                if (mg->members[i].first != center_rope) continue;
                float fu = 0.0f;
                if (i < mg->member_u.size()) fu = mg->member_u[i];
                else {
                    Rope* rr = si->ropes[static_cast<size_t>(center_rope)].get();
                    int verts = rr ? rr->segments + 1 : 0;
                    int vid = mg->members[i].second;
                    if (verts > 1) fu = static_cast<float>(vid) / static_cast<float>(verts - 1);
                }
                float du = std::fabs(fu - 0.5f);
                if (du < best) { best = du; stem_member = static_cast<int>(i); }
            }
            if (stem_member >= 0) {
                apply_member_u(stem_member, 0.5f);
            }
        }

        for (size_t ei = 0; ei < mg->edges.size(); ++ei) {
            const auto &e = mg->edges[ei];
            int ia = e.first; int ib = e.second;
            if (ia < 0 || ib < 0) continue;
            if (ia >= static_cast<int>(mg->members.size()) || ib >= static_cast<int>(mg->members.size())) continue;
            auto ma = mg->members[static_cast<size_t>(ia)];
            auto mb = mg->members[static_cast<size_t>(ib)];
            if (ma.first < 0 || mb.first < 0) continue;
            if (ma.first >= static_cast<int>(si->ropes.size()) || mb.first >= static_cast<int>(si->ropes.size())) continue;
            Rope* ra = si->ropes[static_cast<size_t>(ma.first)].get();
            Rope* rb = si->ropes[static_cast<size_t>(mb.first)].get();
            if (!ra || !rb) continue;

            // use stored member_u when available; fall back to vertex index -> u
            float fua = 0.0f; float fub = 0.0f;
            if (static_cast<size_t>(ia) < mg->member_u.size()) fua = mg->member_u[static_cast<size_t>(ia)];
            else { int va = ma.second; int verts = ra->segments + 1; if (verts > 1) fua = static_cast<float>(va) / static_cast<float>(verts - 1); }
            if (static_cast<size_t>(ib) < mg->member_u.size()) fub = mg->member_u[static_cast<size_t>(ib)];
            else { int vb = mb.second; int verts = rb->segments + 1; if (verts > 1) fub = static_cast<float>(vb) / static_cast<float>(verts - 1); }

            float ax,ay,az,bx,by,bz; int ai0,ai1,bi0,bi1; float alocal,blocal;
            sample_rope(ra, fua, ax, ay, az, ai0, ai1, alocal);
            sample_rope(rb, fub, bx, by, bz, bi0, bi1, blocal);

            float dx = bx - ax; float dy = by - ay; float dz = bz - az;
            float d = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (d <= 1e-6f) continue;
            float rest = (ei < mg->edge_rest.size()) ? mg->edge_rest[ei] : d;
            float strength = 1.0f;
            if (ei < mg->edge_strength.size()) strength = mg->edge_strength[ei];
            float diff = (d - rest) / d * strength;
            // full correction vector from A->B
            float cx = dx * diff; float cy = dy * diff; float cz = dz * diff;

            // determine if endpoints are fixed (u at rope ends)
            bool a_fixed = (fua <= 1e-6f) || (fua >= 1.0f - 1e-6f) || (ia == stem_member);
            bool b_fixed = (fub <= 1e-6f) || (fub >= 1.0f - 1e-6f) || (ib == stem_member);
            if (a_fixed && b_fixed) continue;

            // compute local tangents at each member
            float tax, tay, taz, tbx_, tby, tbz;
            compute_tangent(ra, fua, tax, tay, taz);
            compute_tangent(rb, fub, tbx_, tby, tbz);
            int ring_a = (static_cast<size_t>(ia) < ring_for_member.size()) ? ring_for_member[static_cast<size_t>(ia)] : -1;
            int ring_b = (static_cast<size_t>(ib) < ring_for_member.size()) ? ring_for_member[static_cast<size_t>(ib)] : -1;

            float dax = 0.0f, day = 0.0f, daz = 0.0f;
            float dbx = 0.0f, dby = 0.0f, dbz = 0.0f;
            if (a_fixed && !b_fixed) {
                dbx = -cx; dby = -cy; dbz = -cz;
            } else if (b_fixed && !a_fixed) {
                dax = cx; day = cy; daz = cz;
            } else {
                dax = 0.5f * cx; day = 0.5f * cy; daz = 0.5f * cz;
                dbx = -0.5f * cx; dby = -0.5f * cy; dbz = -0.5f * cz;
            }

            auto apply_endpoint = [&](int member_idx, Rope* rr, int rope_idx, float fu,
                                      int i0, int i1, float local_t,
                                      float tx, float ty, float tz,
                                      float dx_, float dy_, float dz_,
                                      bool fixed, int ring_id) {
                float tangential = dx_ * tx + dy_ * ty + dz_ * tz;
                if (!fixed) {
                    float seg_dx = rr->pos_x[static_cast<size_t>(i1)] - rr->pos_x[static_cast<size_t>(i0)];
                    float seg_dy = rr->pos_y[static_cast<size_t>(i1)] - rr->pos_y[static_cast<size_t>(i0)];
                    float seg_dz = rr->pos_z[static_cast<size_t>(i1)] - rr->pos_z[static_cast<size_t>(i0)];
                    float seg_len = std::sqrt(seg_dx*seg_dx + seg_dy*seg_dy + seg_dz*seg_dz);
                    if (seg_len > 1e-6f && rr->segments > 0) {
                        float delta_u = tangential / (seg_len * static_cast<float>(rr->segments));
                        float cur_u = fu;
                        if (static_cast<size_t>(member_idx) < mg->member_u.size()) {
                            cur_u = mg->member_u[static_cast<size_t>(member_idx)];
                        }
                        apply_member_u(member_idx, cur_u + delta_u);
                    }
                }
                float nx = dx_ - tx * tangential;
                float ny = dy_ - ty * tangential;
                float nz = dz_ - tz * tangential;
                apply_normal(rr, rope_idx, ring_id, i0, i1, local_t, nx, ny, nz);
            };

            if (!a_fixed || (dax != 0.0f || day != 0.0f || daz != 0.0f)) {
                apply_endpoint(ia, ra, ma.first, fua, ai0, ai1, alocal, tax, tay, taz, dax, day, daz, a_fixed, ring_a);
            }
            if (!b_fixed || (dbx != 0.0f || dby != 0.0f || dbz != 0.0f)) {
                apply_endpoint(ib, rb, mb.first, fub, bi0, bi1, blocal, tbx_, tby, tbz, dbx, dby, dbz, b_fixed, ring_b);
            }
        }
    }
}

// Wrap up: integrate meta-group enforcement into step. We call this at end of step.
// Note: original RopeSim pointer layout is compatible with reinterpret_cast above
int rope_sim_step(RopeSim* s, float dt, float gravity, int constraint_iters, float damping) {
    // call existing implementation by reusing previous function body name: we duplicate logic here
    if (!s) return 0;
    if (dt <= 0.0f) return 0;
    RopeSim_internal* si = to_internal(s);
    // acceleration per-step
    const float gx = 0.0f;
    const float gy = gravity >= 0.0f ? gravity : 0.0f;
    const float gz = 0.0f;

    // integrate all ropes (vertex-wise Verlet)
    for (auto &rp : si->ropes) {
        if (!rp) continue;
        int verts = rp->segments + 1;
        for (int vi = 1; vi < verts - 1; ++vi) {
            float &x = rp->pos_x[vi];
            float &y = rp->pos_y[vi];
            float &z = rp->pos_z[vi];
            float &px = rp->prev_x[vi];
            float &py = rp->prev_y[vi];
            float &pz = rp->prev_z[vi];
            verlet_step_point(x, y, z, px, py, pz, dt, gx, gy, gz, damping);
        }
    }

    // constraints: keep segment lengths close to rest_len
    for (int it = 0; it < std::max(1, constraint_iters); ++it) {
        for (auto &rp : si->ropes) {
            if (!rp) continue;
            int verts = rp->segments + 1;
            if (verts <= 1) continue;
            if (rp->rest_len_seg.size() != static_cast<size_t>(rp->segments)) {
                rp->rest_len_seg.assign(static_cast<size_t>(rp->segments), rp->rest_len);
            }
            rp->pos_x[0] = rp->ax; rp->pos_y[0] = rp->ay; rp->pos_z[0] = rp->az;
            rp->pos_x[verts-1] = rp->bx; rp->pos_y[verts-1] = rp->by; rp->pos_z[verts-1] = rp->bz;
            for (int si_i = 0; si_i < rp->segments; ++si_i) {
                int i0 = si_i;
                int i1 = si_i + 1;
                float x0 = rp->pos_x[i0];
                float y0 = rp->pos_y[i0];
                float z0 = rp->pos_z[i0];
                float x1 = rp->pos_x[i1];
                float y1 = rp->pos_y[i1];
                float z1 = rp->pos_z[i1];
                float dx = x1 - x0;
                float dy = y1 - y0;
                float dz = z1 - z0;
                float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (d <= 1e-6f) continue;
                float rl = rp->rest_len;
                if (rp->rest_len_seg.size() == static_cast<size_t>(rp->segments)) {
                    rl = rp->rest_len_seg[static_cast<size_t>(si_i)];
                }
                float diff = (d - rl) / d;
                bool a_fixed = (i0 == 0);
                bool b_fixed = (i1 == verts - 1);
                if (a_fixed && b_fixed) {
                    continue;
                } else if (a_fixed) {
                    rp->pos_x[i1] = x1 - dx * diff;
                    rp->pos_y[i1] = y1 - dy * diff;
                    rp->pos_z[i1] = z1 - dz * diff;
                } else if (b_fixed) {
                    rp->pos_x[i0] = x0 + dx * diff;
                    rp->pos_y[i0] = y0 + dy * diff;
                    rp->pos_z[i0] = z0 + dz * diff;
                } else {
                    rp->pos_x[i0] = x0 + dx * 0.5f * diff;
                    rp->pos_y[i0] = y0 + dy * 0.5f * diff;
                    rp->pos_z[i0] = z0 + dz * 0.5f * diff;
                    rp->pos_x[i1] = x1 - dx * 0.5f * diff;
                    rp->pos_y[i1] = y1 - dy * 0.5f * diff;
                    rp->pos_z[i1] = z1 - dz * 0.5f * diff;
                }
            }
            rp->pos_x[0] = rp->ax; rp->pos_y[0] = rp->ay; rp->pos_z[0] = rp->az;
            rp->pos_x[verts-1] = rp->bx; rp->pos_y[verts-1] = rp->by; rp->pos_z[verts-1] = rp->bz;
        }
    }

    // apply meta-group edge constraints
    apply_meta_group_edges(to_internal(s), dt);

    // Update rings: project stored ring world-position onto the updated rope
    for (size_t ri = 0; ri < si->rings.size(); ++ri) {
        auto *rg = si->rings[ri].get();
        if (!rg) continue;
        int rope_idx = rg->rope_idx;
        if (rope_idx < 0 || rope_idx >= static_cast<int>(si->ropes.size())) continue;
        Rope* r = si->ropes[static_cast<size_t>(rope_idx)].get();
        if (!r) continue;
        int verts = r->segments + 1;
        if (verts < 2) continue;
        if (rg->vertex_idx > 0 && rg->vertex_idx < verts - 1) {
            float old_rx = rg->x, old_ry = rg->y, old_rz = rg->z;
            rg->prev_x = old_rx; rg->prev_y = old_ry; rg->prev_z = old_rz;
            rg->x = r->pos_x[static_cast<size_t>(rg->vertex_idx)];
            rg->y = r->pos_y[static_cast<size_t>(rg->vertex_idx)];
            rg->z = r->pos_z[static_cast<size_t>(rg->vertex_idx)];
            if (rg->meta_group_idx >= 0 && rg->member_idx >= 0 &&
                rg->meta_group_idx < static_cast<int>(si->meta_groups.size())) {
                MetaGroup* mg = si->meta_groups[static_cast<size_t>(rg->meta_group_idx)].get();
                if (mg && rg->member_idx < static_cast<int>(mg->member_u.size())) {
                    mg->member_u[static_cast<size_t>(rg->member_idx)] = rg->u;
                }
            }
            continue;
        }
        // find nearest point on rope to previous ring world pos (rg->x,y,z)
        float best_d2 = std::numeric_limits<float>::infinity();
        int best_seg = 0; float best_t = 0.0f;
        for (int si_i = 0; si_i < verts - 1; ++si_i) {
            float x0 = r->pos_x[si_i]; float y0 = r->pos_y[si_i]; float z0 = r->pos_z[si_i];
            float x1 = r->pos_x[si_i+1]; float y1 = r->pos_y[si_i+1]; float z1 = r->pos_z[si_i+1];
            float vx = x1 - x0; float vy = y1 - y0; float vz = z1 - z0;
            float wx = rg->x - x0; float wy = rg->y - y0; float wz = rg->z - z0;
            float len2 = vx*vx + vy*vy + vz*vz;
            float t = 0.0f;
            if (len2 > 1e-9f) t = (vx*wx + vy*wy + vz*wz) / len2;
            if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
            float px = x0 + vx * t; float py = y0 + vy * t; float pz = z0 + vz * t;
            float dx = rg->x - px; float dy = rg->y - py; float dz = rg->z - pz;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d2) { best_d2 = d2; best_seg = si_i; best_t = t; }
        }
        float proj_u = (static_cast<float>(best_seg) + best_t) / static_cast<float>(verts - 1);
        if (proj_u < 0.0f) proj_u = 0.0f; if (proj_u > 1.0f) proj_u = 1.0f;

        // compute tangent and interpolated vertex velocity at the projected segment
        int seg_i = best_seg;
        int e0 = seg_i;
        int e1 = seg_i + 1;
        float x0 = r->pos_x[e0]; float y0 = r->pos_y[e0]; float z0 = r->pos_z[e0];
        float x1 = r->pos_x[e1]; float y1 = r->pos_y[e1]; float z1 = r->pos_z[e1];
        float vx0 = r->pos_x[static_cast<size_t>(e0)] - r->prev_x[static_cast<size_t>(e0)];
        float vy0 = r->pos_y[static_cast<size_t>(e0)] - r->prev_y[static_cast<size_t>(e0)];
        float vz0 = r->pos_z[static_cast<size_t>(e0)] - r->prev_z[static_cast<size_t>(e0)];
        float vx1 = r->pos_x[static_cast<size_t>(e1)] - r->prev_x[static_cast<size_t>(e1)];
        float vy1 = r->pos_y[static_cast<size_t>(e1)] - r->prev_y[static_cast<size_t>(e1)];
        float vz1 = r->pos_z[static_cast<size_t>(e1)] - r->prev_z[static_cast<size_t>(e1)];
        float interp_vx = vx0 * (1.0f - best_t) + vx1 * best_t;
        float interp_vy = vy0 * (1.0f - best_t) + vy1 * best_t;
        float interp_vz = vz0 * (1.0f - best_t) + vz1 * best_t;
        float tx = x1 - x0; float ty = y1 - y0; float tz = z1 - z0;
        float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);

        // ring world velocity (previous world pos -> current world pos) / dt
        float old_rx = rg->x, old_ry = rg->y, old_rz = rg->z;
        float world_vx = 0.0f, world_vy = 0.0f, world_vz = 0.0f;
        if (dt > 1e-9f) {
            world_vx = (old_rx - rg->prev_x) / dt;
            world_vy = (old_ry - rg->prev_y) / dt;
            world_vz = (old_rz - rg->prev_z) / dt;
        }

        // relative velocity of ring w.r.t. rope point
        float rel_vx = world_vx - interp_vx;
        float rel_vy = world_vy - interp_vy;
        float rel_vz = world_vz - interp_vz;

        float tangential_speed = 0.0f;
        if (tlen > 1e-6f) {
            tangential_speed = (rel_vx * tx + rel_vy * ty + rel_vz * tz) / tlen;
        }

         // free-u mode: set ring parameter directly to projection so `u` is a free axis
         float old_u_val = rg->u;
         float new_u = proj_u;
         rg->u = new_u;

         // debug log: show ring parameter changes so we can see `u` moving
         // debug: ring parameter logging removed
         // printf("[rope_sim] ring %d rope=%d old_u=%f proj_u=%f new_u=%f seg=%d best_t=%f\n",
         //     (int)ri, rope_idx, old_u_val, proj_u, new_u, best_seg, best_t);

        // recompute world pos from rope at updated u
        float fidx = rg->u * static_cast<float>(verts - 1);
        int pi0 = static_cast<int>(std::floor(fidx)); if (pi0 < 0) pi0 = 0; if (pi0 >= verts-1) pi0 = verts-2;
        int pi1 = pi0 + 1; float local_t2 = fidx - static_cast<float>(pi0);
        float new_rx = r->pos_x[static_cast<size_t>(pi0)] * (1.0f - local_t2) + r->pos_x[static_cast<size_t>(pi1)] * local_t2;
        float new_ry = r->pos_y[static_cast<size_t>(pi0)] * (1.0f - local_t2) + r->pos_y[static_cast<size_t>(pi1)] * local_t2;
        float new_rz = r->pos_z[static_cast<size_t>(pi0)] * (1.0f - local_t2) + r->pos_z[static_cast<size_t>(pi1)] * local_t2;

        // advance stored prev position and set new world pos
        rg->prev_x = old_rx; rg->prev_y = old_ry; rg->prev_z = old_rz;
        rg->x = new_rx; rg->y = new_ry; rg->z = new_rz;
    }

    return 1;
}

} // extern C
