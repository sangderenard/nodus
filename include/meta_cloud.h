#pragma once

#include <cstdint>
#include <vector>
#include <string>

// Forward declarations for ABI types used during integration.
#ifdef __cplusplus
extern "C" {
#endif
typedef struct GP_TableContext GP_TableContext;
typedef struct GP_MetaGroup GP_MetaGroup;
#ifdef __cplusplus
}
#endif

// MetaCloud: central coordinator for meta-group clouds (vertices, rings,
// lasso ropes, overlay bindings, and edge metadata). This is a scaffold
// to be incrementally wired into the table/canvas codebase.
class MetaCloud {
public:
    struct VertexRef {
        uint64_t rope_uid = 0ull; // persistent rope uid
        int32_t vertex_idx = -1;  // discrete vertex index on rope
    };

    struct RingRef {
        uint64_t uid = 0ull;              // local ring uid
        uint64_t key = 0ull;              // meta-group key (for registration)
        int32_t rope_idx = -1;            // runtime rope index (optional)
        float u = 0.0f;                   // parametric position along rope
    };

    struct OverlayBinding {
        uint64_t key_a = 0ull;
        uint64_t key_b = 0ull;
        uint64_t port_uuid_a = 0ull;
        uint64_t port_uuid_b = 0ull;
    };

    MetaCloud();
    ~MetaCloud();

    // Mutators
    void add_vertex(const VertexRef& v);
    void set_overlay(const OverlayBinding& o);
    void add_ring(const RingRef& r);

    // Build a MetaCloud from a live table/meta-group snapshot.
    // This captures the current meta-group contents (vertices, anchor,
    // ring registration key, overlay keys) into a self-contained object
    // that can be serialized or later re-applied to a table.
    static MetaCloud from_table_snapshot(GP_TableContext* ctx, GP_MetaGroup* mg);

    // Validate invariants (bounds, uuids, etc.). Returns true if OK.
    bool validate() const;

    // Finalize any derived state (compute member parametric u from indices,
    // normalize ring entries, deduplicate vertices). Called before serialize.
    int finalize_build();

    // Serialize into a self-contained binary blob. Returns number of bytes written.
    // Format is intentionally simple and versioned so it can evolve.
    int serialize(std::vector<char>& out) const;

    // Deserialize from blob produced by `serialize`. Returns 1 on success, 0 on failure.
    int deserialize(const char* data, int32_t len);

    // Instantiate/wire into a live table context (creates rings, lasso ropes,
    // registers overlays). This is the atomic operation used by the deserializer
    // to restore the MetaCloud into runtime structures.
    // NOTE: implementation is a stub until integration is completed.
    // Instantiate/wire into a live table context (creates rings, lasso ropes,
    // registers overlays). Returns the created `GP_MetaGroup*` on success or
    // nullptr on failure. Caller owns/uses the returned pointer via table APIs.
    GP_MetaGroup* apply_to_table(GP_TableContext* ctx) const;

    // Debug helper
    std::string to_string() const;

private:
    uint32_t version_ = 1;
    std::vector<VertexRef> vertices_;
    std::vector<RingRef> rings_;
    OverlayBinding overlay_{};
    // future: channel state, edge metadata, lasso ropes, dangling widgets, etc.
};
