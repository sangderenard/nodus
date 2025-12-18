#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GP_Subject GP_Subject;

typedef struct GP_SubjectVertex {
    float x, y, z;
} GP_SubjectVertex;

typedef struct GP_SubjectTriangle {
    uint32_t i0, i1, i2;   // vertex indices
    float normal[3];       // optional; auto-computed if zero
    float reflectivity;    // 0..1 (1 = mirror, 0 = absorb)
} GP_SubjectTriangle;

// One captured surface interaction.
typedef struct GP_SubjectRayEvent {
    float pos[3];      // world-space hit position
    float dir_in[3];   // normalized incoming direction
    float dir_out[3];  // normalized outgoing/reflected direction (zero if absorbed)
    float weight;      // contribution/intensity at hit
    int32_t bounce;    // bounce index
    float travel;      // cumulative travel length
} GP_SubjectRayEvent;

// Create/destroy.
GP_Subject* gp_subject_create(void);
void gp_subject_destroy(GP_Subject* s);

// Set mesh (copies data). If normalize != 0, the mesh is uniformly scaled/translated
// into a unit cube centered at origin (useful for canonical subjects).
int32_t gp_subject_set_mesh(GP_Subject* s,
    const GP_SubjectVertex* verts, int32_t vert_count,
    const GP_SubjectTriangle* tris, int32_t tri_count,
    int32_t normalize);

// Clear captured events.
int32_t gp_subject_clear_events(GP_Subject* s);

// Append a captured event (optional: future ray tracer can call this).
int32_t gp_subject_add_event(GP_Subject* s, const GP_SubjectRayEvent* ev);

// Copy events out (returns number of events copied).
int32_t gp_subject_copy_events(const GP_Subject* s, GP_SubjectRayEvent* out_events, int32_t max_events);

// Query counts.
int32_t gp_subject_get_counts(const GP_Subject* s, int32_t* out_vertices, int32_t* out_triangles, int32_t* out_events);

#ifdef __cplusplus
}
#endif

