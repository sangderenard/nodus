#pragma once
#include <cstdint>

extern "C" {

typedef struct RopeSim RopeSim;

// Create a simulator capable of holding up to max_ropes each with up to max_segments segments.
RopeSim* rope_sim_create(int max_ropes, int max_segments_per_rope);
void rope_sim_destroy(RopeSim* s);

// Add a rope with endpoints (ax,ay)-(bx,by). Returns rope index or -1 on error.
int rope_sim_add_rope(RopeSim* s, float ax, float ay, float bx, float by, int segments, float slack_fraction);
// 3D variant: endpoints (ax,ay,az)-(bx,by,bz). Returns rope index or -1 on error.
int rope_sim_add_rope3(RopeSim* s, float ax, float ay, float az, float bx, float by, float bz, int segments, float slack_fraction);

// Update endpoints for an existing rope (rope_idx returned from rope_sim_add_rope)
int rope_sim_set_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by);
int rope_sim_set_endpoints3(RopeSim* s, int rope_idx, float ax, float ay, float az, float bx, float by, float bz);

// Move endpoints but preserve previous positions so the integrator sees the endpoint velocity.
int rope_sim_move_endpoints(RopeSim* s, int rope_idx, float ax, float ay, float bx, float by);
int rope_sim_move_endpoints3(RopeSim* s, int rope_idx, float ax, float ay, float az, float bx, float by, float bz);

// Step the simulation. constraint_iters controls how many constraint passes are applied.
int rope_sim_step(RopeSim* s, float dt, float gravity, int constraint_iters, float damping);

// Query functions: get vertex count for rope, and copy XY vertices into out_xy (length must be >= 2*count)
int rope_sim_get_vertex_count(RopeSim* s, int rope_idx);
int rope_sim_get_vertices(RopeSim* s, int rope_idx, float* out_xy, int max_count);
int rope_sim_get_vertices3(RopeSim* s, int rope_idx, float* out_xyz, int max_count);
// Insert a vertex into a rope at segment seg_index (between seg_index and seg_index+1)
// at param t in [0,1]. Returns the new vertex index or -1 on error.
int rope_sim_insert_vertex(RopeSim* s, int rope_idx, int seg_index, float t);

// Per-rope radius for exclusion/pipe collision behavior.
int rope_sim_set_rope_radius(RopeSim* s, int rope_idx, float radius);
int rope_sim_get_rope_radius(RopeSim* s, int rope_idx, float* out_radius);

// Meta-group (confinement) APIs: create/destroy meta groups and add vertex
// bindings. A meta group represents a set of rope vertices that should be
// constrained to move toward each other while allowing sliding along rope
// tangents. Pressure is a soft parameter controlling the attraction strength.
int rope_sim_create_meta_group(RopeSim* s, float pressure);
int rope_sim_destroy_meta_group(RopeSim* s, int group_idx);
int rope_sim_meta_group_add(RopeSim* s, int group_idx, int rope_idx, int vertex_idx);
// Insert a member into a meta-group after a specified existing member.
// If the (after_rope_idx, after_vertex_idx) pair is not found, the new
// member is appended. Returns 1 on success.
int rope_sim_meta_group_insert(RopeSim* s, int group_idx, int after_rope_idx, int after_vertex_idx, int rope_idx, int vertex_idx);
// Check whether a meta-group already contains the given (rope_idx,vertex_idx) member.
int rope_sim_meta_group_has_member(RopeSim* s, int group_idx, int rope_idx, int vertex_idx);
int rope_sim_meta_group_set_pressure(RopeSim* s, int group_idx, float pressure);
// Edge-spring support for meta-groups: create springs connecting consecutive
// meta-group members. Springs have a rest length that can be reduced over
// time until a minimum. Returns 1 on success.
int rope_sim_meta_group_enable_edge_springs(RopeSim* s, int group_idx, float min_rest, float reduce_rate);
int rope_sim_meta_group_disable_edge_springs(RopeSim* s, int group_idx);
// Mark a dangling/widget rope index on the meta-group so edge creation
// can give it special rest-length and stiffness behavior.
int rope_sim_meta_group_set_dangling_rope(RopeSim* s, int group_idx, int rope_idx);
// Set ring creation mode for a meta-group (0=ribbon,1=closed,2=dense)
int rope_sim_meta_group_set_mode(RopeSim* s, int group_idx, int mode);
// Query meta-group mode (0/1/2). Returns 1 on success.
int rope_sim_meta_group_get_mode(RopeSim* s, int group_idx, int* out_mode);
// Find a meta-group that contains any member from `rope_idx`. Returns group_idx or -1 if not found.
int rope_sim_find_meta_group_with_rope(RopeSim* s, int rope_idx);

// Per-rope rest-length adjustments: modify immediately, or schedule a target
// rest length to be approached at a given linear rate after an optional delay.
int rope_sim_modify_rest_length(RopeSim* s, int rope_idx, float delta);
int rope_sim_set_rope_rest_target(RopeSim* s, int rope_idx, float target_rest, float rate, float delay);
int rope_sim_clear_rope_rest_target(RopeSim* s, int rope_idx);
int rope_sim_get_rope_rest_length(RopeSim* s, int rope_idx, float* out_rest);
// Parametric ring APIs: rings slide along a rope (u in [0,1]) and apply
// tangential forces to nearby vertices. Returns ring id >=0 on success.
// Declare ring API functions
int rope_sim_create_ring(RopeSim* s, int rope_idx, float u);
int rope_sim_destroy_ring(RopeSim* s, int ring_id);
int rope_sim_set_ring_target(RopeSim* s, int ring_id, float target_u, float speed);
int rope_sim_get_ring_u(RopeSim* s, int ring_id, float* out_u);
// Query which rope a ring is attached to. Returns 1 on success and sets out_rope_idx, 0 on failure.
int rope_sim_get_ring_rope_index(RopeSim* s, int ring_id, int* out_rope_idx);

// Dangling widget API: create a small widget attached to a rope vertex that
// follows the vertex position each simulation step. `widget_type` is one of
// the `LassoWidgetType` values defined in `lasso_config.h` (caller must
// ensure compatibility). Returns widget id >=0 on success or -1 on failure.
int rope_sim_create_dangling_widget(RopeSim* s, int rope_idx, int vertex_idx, unsigned int widget_type);
int rope_sim_destroy_dangling_widget(RopeSim* s, int widget_id);
// Query widget world position (x,y,z). out_xyz must point to an array of 3 floats.
int rope_sim_get_widget_position(RopeSim* s, int widget_id, float* out_xyz);
// Set the mass multiplier for a widget. Returns 1 on success.
int rope_sim_set_widget_mass(RopeSim* s, int widget_id, float mass);
// Query world position of a meta-group member sampled from its parametric `u`.
// out_xyz must point to 3 floats. Returns 1 on success, 0 on failure.
int rope_sim_meta_group_get_member_world_pos(RopeSim* s, int group_idx, int member_idx, float* out_xyz);
} // extern C
