#include "rope_sim.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <cstring>

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

extern "C" {

RopeSim* rope_sim_create(int max_ropes, int max_segments_per_rope) {
    if (max_ropes <= 0 || max_segments_per_rope <= 0) return nullptr;
    RopeSim* s = new RopeSim();
    s->max_ropes = max_ropes;
    s->max_segments_per_rope = max_segments_per_rope;
    s->ropes.reserve(static_cast<std::size_t>(max_ropes));
    return s;
}

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

int rope_sim_step(RopeSim* s, float dt, float gravity, int constraint_iters, float damping) {
    if (!s) return 0;
    if (dt <= 0.0f) return 0;
    // acceleration per-step
    const float gx = 0.0f;
    const float gy = gravity >= 0.0f ? gravity : 0.0f;
    const float gz = 0.0f; // neutral along depth; endpoints encode plug depth

    // integrate all ropes (vertex-wise Verlet)
    for (auto &rp : s->ropes) {
        if (!rp) continue;
        int verts = rp->segments + 1;
        // endpoints are fixed (index 0 and verts-1)
#ifdef EIGEN_SIM
        // vectorized Verlet using Eigen maps
        using namespace Eigen;
        Map<VectorXf> px_map(rp->pos_x.data(), verts);
        Map<VectorXf> py_map(rp->pos_y.data(), verts);
        Map<VectorXf> pz_map(rp->pos_z.data(), verts);
        Map<VectorXf> prevx_map(rp->prev_x.data(), verts);
        Map<VectorXf> prevy_map(rp->prev_y.data(), verts);
        Map<VectorXf> prevz_map(rp->prev_z.data(), verts);
        if (verts > 2) {
            VectorXf temp_x = px_map.segment(1, verts - 2);
            VectorXf temp_y = py_map.segment(1, verts - 2);
            VectorXf temp_z = pz_map.segment(1, verts - 2);
            // vx = x - px
            VectorXf vx = temp_x - prevx_map.segment(1, verts - 2);
            VectorXf vy = temp_y - prevy_map.segment(1, verts - 2);
            VectorXf vz = temp_z - prevz_map.segment(1, verts - 2);
            temp_x = temp_x + vx * (1.0f - damping) + VectorXf::Constant(verts - 2, gx * dt * dt);
            temp_y = temp_y + vy * (1.0f - damping) + VectorXf::Constant(verts - 2, gy * dt * dt);
            temp_z = temp_z + vz * (1.0f - damping) + VectorXf::Constant(verts - 2, gz * dt * dt);
            prevx_map.segment(1, verts - 2) = px_map.segment(1, verts - 2);
            prevy_map.segment(1, verts - 2) = py_map.segment(1, verts - 2);
            prevz_map.segment(1, verts - 2) = pz_map.segment(1, verts - 2);
            px_map.segment(1, verts - 2) = temp_x;
            py_map.segment(1, verts - 2) = temp_y;
            pz_map.segment(1, verts - 2) = temp_z;
        }
#else
        for (int vi = 1; vi < verts - 1; ++vi) {
            float &x = rp->pos_x[vi];
            float &y = rp->pos_y[vi];
            float &z = rp->pos_z[vi];
            float &px = rp->prev_x[vi];
            float &py = rp->prev_y[vi];
            float &pz = rp->prev_z[vi];
            verlet_step_point(x, y, z, px, py, pz, dt, gx, gy, gz, damping);
        }
#endif
    }

    // constraints: keep segment lengths close to rest_len
    for (int it = 0; it < std::max(1, constraint_iters); ++it) {
        for (auto &rp : s->ropes) {
            if (!rp) continue;
            int verts = rp->segments + 1;
            float rl = rp->rest_len;
            // ensure endpoints fixed positions
            if (verts <= 1) continue;
#ifdef EIGEN_SIM
            using namespace Eigen;
            Map<VectorXf> px_map(rp->pos_x.data(), verts);
            Map<VectorXf> py_map(rp->pos_y.data(), verts);
            Map<VectorXf> pz_map(rp->pos_z.data(), verts);
            // compute differences for segments: dx = x[i+1]-x[i]
            VectorXf dx = px_map.segment(1, verts - 1) - px_map.segment(0, verts - 1);
            VectorXf dy = py_map.segment(1, verts - 1) - py_map.segment(0, verts - 1);
            VectorXf dz = pz_map.segment(1, verts - 1) - pz_map.segment(0, verts - 1);
            VectorXf d = (dx.array().square() + dy.array().square() + dz.array().square()).sqrt();
            // apply corrections segment-wise (avoid divide by zero)
            for (int si = 0; si < rp->segments; ++si) {
                float di = d[si];
                if (di <= 1e-6f) continue;
                float diff = (di - rl) / di;
                float cx = dx[si] * 0.5f * diff;
                float cy = dy[si] * 0.5f * diff;
                float cz = dz[si] * 0.5f * diff;
                px_map[si] += cx;
                py_map[si] += cy;
                pz_map[si] += cz;
                px_map[si+1] -= cx;
                py_map[si+1] -= cy;
                pz_map[si+1] -= cz;
            }
            // enforce endpoints
            px_map[0] = rp->ax; py_map[0] = rp->ay; pz_map[0] = rp->az;
            px_map[verts-1] = rp->bx; py_map[verts-1] = rp->by; pz_map[verts-1] = rp->bz;
#else
            rp->pos_x[0] = rp->ax; rp->pos_y[0] = rp->ay; rp->pos_z[0] = rp->az;
            rp->pos_x[verts-1] = rp->bx; rp->pos_y[verts-1] = rp->by; rp->pos_z[verts-1] = rp->bz;
            for (int si = 0; si < rp->segments; ++si) {
                int i0 = si;
                int i1 = si + 1;
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
            // re-enforce endpoints
            rp->pos_x[0] = rp->ax; rp->pos_y[0] = rp->ay; rp->pos_z[0] = rp->az;
            rp->pos_x[verts-1] = rp->bx; rp->pos_y[verts-1] = rp->by; rp->pos_z[verts-1] = rp->bz;
#endif
        }
    }

    return 1;
}

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

} // extern C
