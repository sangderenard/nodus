#include "subject_capture.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

struct GP_Subject {
    std::vector<GP_SubjectVertex> verts;
    std::vector<GP_SubjectTriangle> tris;
    std::vector<GP_SubjectRayEvent> events;
    // cached bounds
    float bmin[3] = {0,0,0};
    float bmax[3] = {0,0,0};
};

static void compute_bounds(GP_Subject* s) {
    if (!s || s->verts.empty()) return;
    s->bmin[0] = s->bmax[0] = s->verts[0].x;
    s->bmin[1] = s->bmax[1] = s->verts[0].y;
    s->bmin[2] = s->bmax[2] = s->verts[0].z;
    for (const auto& v : s->verts) {
        s->bmin[0] = std::min(s->bmin[0], v.x);
        s->bmin[1] = std::min(s->bmin[1], v.y);
        s->bmin[2] = std::min(s->bmin[2], v.z);
        s->bmax[0] = std::max(s->bmax[0], v.x);
        s->bmax[1] = std::max(s->bmax[1], v.y);
        s->bmax[2] = std::max(s->bmax[2], v.z);
    }
}

static void normalize_vertices(GP_Subject* s) {
    if (!s || s->verts.empty()) return;
    compute_bounds(s);
    float cx = 0.5f * (s->bmin[0] + s->bmax[0]);
    float cy = 0.5f * (s->bmin[1] + s->bmax[1]);
    float cz = 0.5f * (s->bmin[2] + s->bmax[2]);
    float sx = s->bmax[0] - s->bmin[0];
    float sy = s->bmax[1] - s->bmin[1];
    float sz = s->bmax[2] - s->bmin[2];
    float scale = std::max({sx, sy, sz});
    if (scale < 1e-6f) scale = 1.0f;
    for (auto& v : s->verts) {
        v.x = (v.x - cx) / scale;
        v.y = (v.y - cy) / scale;
        v.z = (v.z - cz) / scale;
    }
    compute_bounds(s);
}

static void compute_tri_normals(GP_Subject* s) {
    if (!s) return;
    for (auto& t : s->tris) {
        if (t.i0 >= s->verts.size() || t.i1 >= s->verts.size() || t.i2 >= s->verts.size()) continue;
        const auto& v0 = s->verts[t.i0];
        const auto& v1 = s->verts[t.i1];
        const auto& v2 = s->verts[t.i2];
        float e1x = v1.x - v0.x;
        float e1y = v1.y - v0.y;
        float e1z = v1.z - v0.z;
        float e2x = v2.x - v0.x;
        float e2y = v2.y - v0.y;
        float e2z = v2.z - v0.z;
        float nx = e1y * e2z - e1z * e2y;
        float ny = e1z * e2x - e1x * e2z;
        float nz = e1x * e2y - e1y * e2x;
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len < 1e-8f) { t.normal[0]=0; t.normal[1]=0; t.normal[2]=1; continue; }
        t.normal[0] = nx / len;
        t.normal[1] = ny / len;
        t.normal[2] = nz / len;
    }
}

extern "C" {

GP_Subject* gp_subject_create(void) {
    return new GP_Subject();
}

void gp_subject_destroy(GP_Subject* s) {
    delete s;
}

int32_t gp_subject_set_mesh(GP_Subject* s,
    const GP_SubjectVertex* verts, int32_t vert_count,
    const GP_SubjectTriangle* tris, int32_t tri_count,
    int32_t normalize) {
    if (!s) return 0;
    s->verts.clear();
    s->tris.clear();
    if (vert_count > 0 && verts) s->verts.assign(verts, verts + vert_count);
    if (tri_count > 0 && tris) {
        s->tris.assign(tris, tris + tri_count);
        for (auto& t : s->tris) {
            t.reflectivity = std::clamp(t.reflectivity, 0.0f, 1.0f);
        }
    }
    if (normalize) normalize_vertices(s);
    compute_tri_normals(s);
    return 1;
}

int32_t gp_subject_clear_events(GP_Subject* s) {
    if (!s) return 0;
    s->events.clear();
    return 1;
}

int32_t gp_subject_add_event(GP_Subject* s, const GP_SubjectRayEvent* ev) {
    if (!s || !ev) return 0;
    s->events.push_back(*ev);
    return 1;
}

int32_t gp_subject_copy_events(const GP_Subject* s, GP_SubjectRayEvent* out_events, int32_t max_events) {
    if (!s || !out_events || max_events <= 0) return 0;
    int32_t n = static_cast<int32_t>(std::min<size_t>(s->events.size(), static_cast<size_t>(max_events)));
    std::memcpy(out_events, s->events.data(), static_cast<size_t>(n) * sizeof(GP_SubjectRayEvent));
    return n;
}

int32_t gp_subject_get_counts(const GP_Subject* s, int32_t* out_vertices, int32_t* out_triangles, int32_t* out_events) {
    if (!s) return 0;
    if (out_vertices) *out_vertices = static_cast<int32_t>(s->verts.size());
    if (out_triangles) *out_triangles = static_cast<int32_t>(s->tris.size());
    if (out_events) *out_events = static_cast<int32_t>(s->events.size());
    return 1;
}

} // extern "C"

