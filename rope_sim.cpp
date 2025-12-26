#include "rope_sim.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <cstring>
#include <cstddef>

// Optional compile-time Eigen path: define EIGEN_SIM to enable Eigen-accelerated paths.
#ifdef EIGEN_SIM
#include <Eigen/Dense>
#endif

struct Rope {
    int segments = 0; // number of segments => vertices = segments + 1
    float slack = 0.0f; // fraction: 0 = tight, >0 allows longer rest length
    float rest_len = 0.0f; // per-segment rest length
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
    int idx = static_cast<int>(s->ropes.size());
    s->ropes.push_back(std::move(r));
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
};

// augment RopeSim with meta_groups container
struct RopeSim_internal : public RopeSim {
    std::vector<std::unique_ptr<MetaGroup>> meta_groups;
    struct Ring { int rope_idx = -1; float u = 0.0f; float x=0.0f,y=0.0f,z=0.0f; float prev_x=0.0f, prev_y=0.0f, prev_z=0.0f; };
    std::vector<std::unique_ptr<Ring>> rings;
};

// Helper to cast to internal
static RopeSim_internal* to_internal(RopeSim* s) {
    return reinterpret_cast<RopeSim_internal*>(s);
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
    int id = static_cast<int>(si->rings.size());
    si->rings.push_back(std::move(ring));
    // debug: create_ring logging removed
    // printf("[rope_sim] create_ring id=%d rope=%d u=%f\n", id, rope_idx, u);
    return id;
}

int rope_sim_destroy_ring(RopeSim* s, int ring_id) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (ring_id < 0 || ring_id >= static_cast<int>(si->rings.size())) return 0;
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

int rope_sim_meta_group_enable_edge_springs(RopeSim* s, int group_idx, float min_rest, float reduce_rate) {
    if (!s) return 0;
    RopeSim_internal* si = to_internal(s);
    if (group_idx < 0 || group_idx >= static_cast<int>(si->meta_groups.size())) return 0;
    MetaGroup* mg = si->meta_groups[static_cast<size_t>(group_idx)].get();
    if (!mg) return 0;
    mg->edges.clear();
    size_t n = mg->members.size();
    if (n < 2) return 1; // nothing to connect
    // chain connections
    for (size_t i = 0; i + 1 < n; ++i) mg->edges.emplace_back(static_cast<int>(i), static_cast<int>(i+1));
    if (mg->mode == 1) {
        // closed loop: add last->first
        mg->edges.emplace_back(static_cast<int>(n-1), 0);
    } else if (mg->mode == 2) {
        // dense: add all pairwise links
        for (size_t i = 0; i < n; ++i) for (size_t j = i+1; j < n; ++j) mg->edges.emplace_back(static_cast<int>(i), static_cast<int>(j));
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
        float init_rest = std::max(d, mg->min_rest);
        float strength = 1.2f; // default: somewhat stiff
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
            }
        }
    }
    return 1;
}

// --- Minimal stubs for remaining APIs (to satisfy linking) ----------
int rope_sim_insert_vertex(RopeSim* s, int rope_idx, int seg_index, float t) {
    (void)s; (void)rope_idx; (void)seg_index; (void)t; return -1; // not implemented
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
        need += 6 * 4; // endpoints
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
        need += 4; // u
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
    const int32_t version = 2;
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
        float endpoints[6] = {r->ax, r->ay, r->az, r->bx, r->by, r->bz};
        if (!rope_sim_write_bytes(p, end, endpoints, sizeof(endpoints))) return 0;
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
        if (!rope_sim_write_bytes(p, end, &ring->u, sizeof(ring->u))) return 0;
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
    if (version != 1 && version != 2) return nullptr;
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
        if (!rope_sim_read_bytes(p, end, endpoints, sizeof(endpoints))) { rope_sim_destroy(sim); return nullptr; }
        if (segments < 1) segments = 1;
        if (segments > max_segments) segments = max_segments;
        auto r = std::make_unique<Rope>();
        r->segments = segments;
        r->slack = slack;
        r->rest_len = rest_len;
        r->ax = endpoints[0]; r->ay = endpoints[1]; r->az = endpoints[2];
        r->bx = endpoints[3]; r->by = endpoints[4]; r->bz = endpoints[5];
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
        int32_t member_count = 0;
        int32_t edge_count = 0;
        if (!rope_sim_read_bytes(p, end, &mg->pressure, sizeof(mg->pressure))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mode, sizeof(mode))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &edges_enabled, sizeof(edges_enabled))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mg->min_rest, sizeof(mg->min_rest))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &mg->reduce_rate, sizeof(mg->reduce_rate))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &dangling_rope, sizeof(dangling_rope))) { rope_sim_destroy(sim); return nullptr; }
        if (!rope_sim_read_bytes(p, end, &member_count, sizeof(member_count))) { rope_sim_destroy(sim); return nullptr; }
        if (member_count < 0) { rope_sim_destroy(sim); return nullptr; }
        mg->mode = mode;
        mg->edges_enabled = edges_enabled != 0;
        mg->dangling_rope = dangling_rope;
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
    if (version >= 2) {
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
            if (!rope_sim_read_bytes(p, end, &ring->u, sizeof(ring->u))) { rope_sim_destroy(sim); return nullptr; }
            float rpos[6] = {};
            if (!rope_sim_read_bytes(p, end, rpos, sizeof(rpos))) { rope_sim_destroy(sim); return nullptr; }
            ring->x = rpos[0]; ring->y = rpos[1]; ring->z = rpos[2];
            ring->prev_x = rpos[3]; ring->prev_y = rpos[4]; ring->prev_z = rpos[5];
            if (ring->rope_idx < 0 || ring->rope_idx >= static_cast<int32_t>(si->ropes.size())) {
                ring->rope_idx = -1;
            }
            if (ring->u < 0.0f) ring->u = 0.0f;
            if (ring->u > 1.0f) ring->u = 1.0f;
            si->rings.push_back(std::move(ring));
        }
    } else {
        si->rings.clear();
    }
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

        // helper: project an arbitrary point onto the rope polyline and return
        // parametric u (0..1) and the segment/local t it lies on.
        auto project_point_onto_rope = [&](Rope* rr, float px, float py, float pz, float &out_u, int &out_i0, int &out_i1, float &out_local_t) {
            out_u = 0.0f; out_i0 = out_i1 = 0; out_local_t = 0.0f; if (!rr) return;
            int segs = rr->segments; if (segs < 1) return;
            float best_d2 = 1e30f; int best_seg = 0; float best_t = 0.0f;
            for (int si = 0; si < segs; ++si) {
                // segment endpoints
                float x0 = rr->pos_x[static_cast<size_t>(si)]; float y0 = rr->pos_y[static_cast<size_t>(si)]; float z0 = rr->pos_z[static_cast<size_t>(si)];
                float x1 = rr->pos_x[static_cast<size_t>(si+1)]; float y1 = rr->pos_y[static_cast<size_t>(si+1)]; float z1 = rr->pos_z[static_cast<size_t>(si+1)];
                float vx = x1 - x0; float vy = y1 - y0; float vz = z1 - z0;
                float len2 = vx*vx + vy*vy + vz*vz;
                float t = 0.0f;
                if (len2 > 1e-12f) {
                    t = ((px - x0)*vx + (py - y0)*vy + (pz - z0)*vz) / len2;
                    if (t < 0.0f) t = 0.0f; if (t > 1.0f) t = 1.0f;
                }
                float cx = x0 + vx * t; float cy = y0 + vy * t; float cz = z0 + vz * t;
                float ddx = px - cx; float ddy = py - cy; float ddz = pz - cz;
                float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                if (d2 < best_d2) { best_d2 = d2; best_seg = si; best_t = t; }
            }
            out_i0 = best_seg; out_i1 = best_seg + 1; out_local_t = best_t;
            out_u = (static_cast<float>(best_seg) + best_t) / static_cast<float>(segs);
            if (out_u < 0.0f) out_u = 0.0f; if (out_u > 1.0f) out_u = 1.0f;
        };

        auto project_perp = [&](float cx_, float cy_, float cz_, float nx, float ny, float nz, float &outx, float &outy, float &outz) {
            float dot = cx_*nx + cy_*ny + cz_*nz;
            outx = cx_ - dot * nx; outy = cy_ - dot * ny; outz = cz_ - dot * nz;
        };

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
            bool a_fixed = (fua <= 1e-6f) || (fua >= 1.0f - 1e-6f);
            bool b_fixed = (fub <= 1e-6f) || (fub >= 1.0f - 1e-6f);
            if (a_fixed && b_fixed) continue;

            // compute local tangents at each member
            float tax, tay, taz, tbx_, tby, tbz;
            compute_tangent(ra, fua, tax, tay, taz);
            compute_tangent(rb, fub, tbx_, tby, tbz);

            if (a_fixed && !b_fixed) {
                // move B only: remove tangential component along B's tangent
                float px, py, pz; project_perp(-cx, -cy, -cz, tbx_, tby, tbz, px, py, pz);
                // distribute perpendicular delta to rb vertices (bi0,bi1)
                rb->pos_x[static_cast<size_t>(bi0)] += px * (1.0f - blocal);
                rb->pos_y[static_cast<size_t>(bi0)] += py * (1.0f - blocal);
                rb->pos_z[static_cast<size_t>(bi0)] += pz * (1.0f - blocal);
                rb->pos_x[static_cast<size_t>(bi1)] += px * blocal;
                rb->pos_y[static_cast<size_t>(bi1)] += py * blocal;
                rb->pos_z[static_cast<size_t>(bi1)] += pz * blocal;
                // Compute a target world position for B by applying the full
                // correction and project that point back onto the rope to get
                // a robust parametric `u`. This avoids relying on single-segment
                // length division which can under-update across multiple segments.
                float target_bx = bx + (-cx);
                float target_by = by + (-cy);
                float target_bz = bz + (-cz);
                float proj_u = 0.0f; int p_i0 = 0, p_i1 = 0; float p_local = 0.0f;
                project_point_onto_rope(rb, target_bx, target_by, target_bz, proj_u, p_i0, p_i1, p_local);
                if (static_cast<size_t>(ib) >= mg->member_u.size()) mg->member_u.resize(mg->members.size(), fub);
                float oldu = mg->member_u[static_cast<size_t>(ib)];
                mg->member_u[static_cast<size_t>(ib)] = proj_u;
                // debug: meta_group member u logging removed
                // printf("[rope_sim] meta_group %zu member %d set u: %f -> %f\n", gi, ib, oldu, proj_u);
            } else if (b_fixed && !a_fixed) {
                // move A only: remove tangential component along A's tangent
                float px, py, pz; project_perp(cx, cy, cz, tax, tay, taz, px, py, pz);
                ra->pos_x[static_cast<size_t>(ai0)] += px * (1.0f - alocal);
                ra->pos_y[static_cast<size_t>(ai0)] += py * (1.0f - alocal);
                ra->pos_z[static_cast<size_t>(ai0)] += pz * (1.0f - alocal);
                ra->pos_x[static_cast<size_t>(ai1)] += px * alocal;
                ra->pos_y[static_cast<size_t>(ai1)] += py * alocal;
                ra->pos_z[static_cast<size_t>(ai1)] += pz * alocal;
                float target_ax = ax + cx;
                float target_ay = ay + cy;
                float target_az = az + cz;
                float proj_u_a = 0.0f; int pa0=0, pa1=0; float pa_local=0.0f;
                project_point_onto_rope(ra, target_ax, target_ay, target_az, proj_u_a, pa0, pa1, pa_local);
                if (static_cast<size_t>(ia) >= mg->member_u.size()) mg->member_u.resize(mg->members.size(), fua);
                float oldu_a = mg->member_u[static_cast<size_t>(ia)];
                mg->member_u[static_cast<size_t>(ia)] = proj_u_a;
                // debug: meta_group member u logging removed
                // printf("[rope_sim] meta_group %zu member %d set u: %f -> %f\n", gi, ia, oldu_a, proj_u_a);
            } else {
                // move both: split correction and convert tangential parts into u deltas
                float half_cx = 0.5f * cx; float half_cy = 0.5f * cy; float half_cz = 0.5f * cz;
                float pax, pay, paz; project_perp(half_cx, half_cy, half_cz, tax, tay, taz, pax, pay, paz);
                float pbx, pby, pbz; project_perp(-half_cx, -half_cy, -half_cz, tbx_, tby, tbz, pbx, pby, pbz);
                ra->pos_x[static_cast<size_t>(ai0)] += pax * (1.0f - alocal);
                ra->pos_y[static_cast<size_t>(ai0)] += pay * (1.0f - alocal);
                ra->pos_z[static_cast<size_t>(ai0)] += paz * (1.0f - alocal);
                ra->pos_x[static_cast<size_t>(ai1)] += pax * alocal;
                ra->pos_y[static_cast<size_t>(ai1)] += pay * alocal;
                ra->pos_z[static_cast<size_t>(ai1)] += paz * alocal;
                rb->pos_x[static_cast<size_t>(bi0)] += pbx * (1.0f - blocal);
                rb->pos_y[static_cast<size_t>(bi0)] += pby * (1.0f - blocal);
                rb->pos_z[static_cast<size_t>(bi0)] += pbz * (1.0f - blocal);
                rb->pos_x[static_cast<size_t>(bi1)] += pbx * blocal;
                rb->pos_y[static_cast<size_t>(bi1)] += pby * blocal;
                rb->pos_z[static_cast<size_t>(bi1)] += pbz * blocal;
                // tangential conversion to delta-u for A
                float dot_a = half_cx * tax + half_cy * tay + half_cz * taz;
                float seg_len_a = std::sqrt(
                    (ra->pos_x[static_cast<size_t>(ai1)] - ra->pos_x[static_cast<size_t>(ai0)]) * (ra->pos_x[static_cast<size_t>(ai1)] - ra->pos_x[static_cast<size_t>(ai0)]) +
                    (ra->pos_y[static_cast<size_t>(ai1)] - ra->pos_y[static_cast<size_t>(ai0)]) * (ra->pos_y[static_cast<size_t>(ai1)] - ra->pos_y[static_cast<size_t>(ai0)]) +
                    (ra->pos_z[static_cast<size_t>(ai1)] - ra->pos_z[static_cast<size_t>(ai0)]) * (ra->pos_z[static_cast<size_t>(ai1)] - ra->pos_z[static_cast<size_t>(ai0)])
                );
                // Project target point for A (half correction) to rope
                float target_ax = ax + half_cx;
                float target_ay = ay + half_cy;
                float target_az = az + half_cz;
                float proj_u_a = 0.0f; int pa0=0, pa1=0; float pa_local=0.0f;
                project_point_onto_rope(ra, target_ax, target_ay, target_az, proj_u_a, pa0, pa1, pa_local);
                if (static_cast<size_t>(ia) >= mg->member_u.size()) mg->member_u.resize(mg->members.size(), fua);
                float oldu_a = mg->member_u[static_cast<size_t>(ia)];
                mg->member_u[static_cast<size_t>(ia)] = proj_u_a;
                // debug: meta_group member u logging removed
                // printf("[rope_sim] meta_group %zu member %d set u: %f -> %f\n", gi, ia, oldu_a, proj_u_a);
                // tangential conversion to delta-u for B
                float dot_b = -half_cx * tbx_ + -half_cy * tby + -half_cz * tbz;
                float seg_len_b = std::sqrt(
                    (rb->pos_x[static_cast<size_t>(bi1)] - rb->pos_x[static_cast<size_t>(bi0)]) * (rb->pos_x[static_cast<size_t>(bi1)] - rb->pos_x[static_cast<size_t>(bi0)]) +
                    (rb->pos_y[static_cast<size_t>(bi1)] - rb->pos_y[static_cast<size_t>(bi0)]) * (rb->pos_y[static_cast<size_t>(bi1)] - rb->pos_y[static_cast<size_t>(bi0)]) +
                    (rb->pos_z[static_cast<size_t>(bi1)] - rb->pos_z[static_cast<size_t>(bi0)]) * (rb->pos_z[static_cast<size_t>(bi1)] - rb->pos_z[static_cast<size_t>(bi0)])
                );
                // Project target point for B (half correction)
                float target_bx = bx - half_cx;
                float target_by = by - half_cy;
                float target_bz = bz - half_cz;
                float proj_u_b = 0.0f; int pb0=0, pb1=0; float pb_local=0.0f;
                project_point_onto_rope(rb, target_bx, target_by, target_bz, proj_u_b, pb0, pb1, pb_local);
                if (static_cast<size_t>(ib) >= mg->member_u.size()) mg->member_u.resize(mg->members.size(), fub);
                float oldu_b = mg->member_u[static_cast<size_t>(ib)];
                mg->member_u[static_cast<size_t>(ib)] = proj_u_b;
                // debug: meta_group member u logging removed
                // printf("[rope_sim] meta_group %zu member %d set u: %f -> %f\n", gi, ib, oldu_b, proj_u_b);
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
            float rl = rp->rest_len;
            if (verts <= 1) continue;
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
