// canvas_abi_canvas_ports.inl -- canvas ports: named contacts on the canvas's ROOT module (canvas_abi.h GP_CANVAS_PORT_*).
// docs/MESH_CANVAS_UP_PORTS_2026-11-21.md, stage 1. Owner, 2026-11-21: the bulk row's edge "would be going to a canvas
// port in the top tools area then you'd need an up option to the mesh". A port is a contact; the data rides the canvas
// edge that ends (OUT) or starts (IN) there, in the root table's FIFO, so a full port refuses its writer like any lossless
// edge and nothing is dropped. The scheduler's ROOT block (thread_manager.cpp, run_scheduled_tick) leaves this contact
// range alone; whoever is above the canvas is the port's only reader or writer.

static int canvas_port_index(const GP_CanvasContextImpl* c, const char* name) {
    if (!c || !name) return -1;
    for (size_t i = 0; i < c->canvas_ports.size(); ++i)
        if (c->canvas_ports[i].name == name) return static_cast<int>(i);
    return -1;
}

static bool canvas_port_name_ok(const char* name) {
    if (!name) return false;
    const size_t n = std::strlen(name);
    if (n == 0 || n > 63) return false;
    for (size_t i = 0; i < n; ++i)
        if (std::isspace(static_cast<unsigned char>(name[i])) || static_cast<unsigned char>(name[i]) < 0x21) return false;
    return true;
}

extern "C" int gp_canvas_root_module_of(GP_CanvasContext* ctx_) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c) return -1;
    if (c->root_module_idx < 0 || c->root_module_idx >= static_cast<int>(c->modules.size())) return -1;
    return c->root_module_idx;
}

extern "C" int gp_canvas_declare_port(GP_CanvasContext* ctx_, const char* name, int direction, int up) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c || !canvas_port_name_ok(name)) return -1;
    if (direction != GP_CANVAS_PORT_OUT && direction != GP_CANVAS_PORT_IN) return -1;
    const int existing = canvas_port_index(c, name);
    if (existing >= 0) {
        auto& p = c->canvas_ports[static_cast<size_t>(existing)];
        if (p.direction != direction) return -1;
        p.up = up ? 1 : 0;
        return p.contact;
    }
    if (static_cast<int>(c->canvas_ports.size()) >= GP_CANVAS_PORT_MAX) return -1;
    if (canvas_ensure_root_module(c) < 0) return -1;   // a headless canvas has none until its first port
    GP_CanvasContextImpl::CanvasPortDecl d;
    d.name = name;
    d.direction = direction;
    d.up = up ? 1 : 0;
    d.contact = GP_CANVAS_PORT_CONTACT_BASE + static_cast<int>(c->canvas_ports.size());
    c->canvas_ports.push_back(d);
    return d.contact;
}

extern "C" int gp_canvas_port_count(GP_CanvasContext* ctx_) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    return c ? static_cast<int>(c->canvas_ports.size()) : -1;
}

extern "C" int gp_canvas_port_at(GP_CanvasContext* ctx_, int ordinal, char* out_name, int out_name_capacity,
                                 int* out_direction, int* out_contact, int* out_up) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    if (!c || ordinal < 0 || ordinal >= static_cast<int>(c->canvas_ports.size())) return 0;
    const auto& p = c->canvas_ports[static_cast<size_t>(ordinal)];
    if (out_name && out_name_capacity > 0) {
        std::snprintf(out_name, static_cast<size_t>(out_name_capacity), "%s", p.name.c_str());
    }
    if (out_direction) *out_direction = p.direction;
    if (out_contact) *out_contact = p.contact;
    if (out_up) *out_up = p.up;
    return 1;
}

// The root-table FIFO of the canvas edge that ends at (OUT) or starts at (IN) this port, by the canonical composed
// pair canvas edge creation stores it under (canvas_abi_table_step.inl, root-table edge resolution).
static bool canvas_port_edge(GP_CanvasContextImpl* c, const GP_CanvasContextImpl::CanvasPortDecl& p,
                             GP_TableContext** out_table, int32_t* out_edge_idx, uint64_t* out_key) {
    const int root_mod = gp_canvas_root_module_of(reinterpret_cast<GP_CanvasContext*>(c));
    GP_TableContext* root = canvas_ensure_root_table(c);
    if (root_mod < 0 || !root) return false;
    for (const auto& e : c->edges) {
        const bool here = p.direction == GP_CANVAS_PORT_OUT
            ? (e.desc.b_module == root_mod && e.desc.b_contact_idx == p.contact)
            : (e.desc.a_module == root_mod && e.desc.a_contact_idx == p.contact);
        if (!here) continue;
        const uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_module)) << 32) |
                            (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.a_contact_idx)) << 16);
        const uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_module)) << 32) |
                            (static_cast<uint64_t>(static_cast<uint32_t>(e.desc.b_contact_idx)) << 16);
        int32_t idx = -1;
        if (!gp_table_edge_index_for_pair(root, ka, kb, &idx) || idx < 0) continue;
        *out_table = root;
        *out_edge_idx = idx;
        *out_key = p.direction == GP_CANVAS_PORT_OUT ? kb : ka;   // the port's own end: composed (ROOT, contact)
        return true;
    }
    return false;
}

extern "C" int gp_canvas_port_consume(GP_CanvasContext* ctx_, const char* name, void* out_bytes, int capacity,
                                      int* out_written) {
    if (out_written) *out_written = 0;
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    const int i = canvas_port_index(c, name);
    if (i < 0 || !out_bytes || capacity <= 0) return 0;
    const auto& p = c->canvas_ports[static_cast<size_t>(i)];
    if (p.direction != GP_CANVAS_PORT_OUT) return 0;
    GP_TableContext* table = nullptr;
    int32_t edge_idx = -1;
    uint64_t key = 0;
    if (!canvas_port_edge(c, p, &table, &edge_idx, &key)) return 0;
    gp_table_edge_subscribe_ex(table, edge_idx, key, /*start_at_head=*/1);
    int32_t got = 0;
    if (!gp_table_edge_consume(table, edge_idx, key, out_bytes, capacity, &got) || got <= 0) return 0;
    if (out_written) *out_written = got;
    return 1;
}

// Whether the port's edge carries AbstractTensor handles (its tensor spec names the tensor type): 1 yes, 0 no (raw
// tokens: scalars), -1 no such port or no edge yet. A row that holds a token of a tensor port across a save keeps the
// tensor's bytes, not the handle, which means nothing in another process.
extern "C" int gp_canvas_port_carries_tensors(GP_CanvasContext* ctx_, const char* name) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    const int i = canvas_port_index(c, name);
    if (i < 0) return -1;
    GP_TableContext* table = nullptr;
    int32_t edge_idx = -1;
    uint64_t key = 0;
    if (!canvas_port_edge(c, c->canvas_ports[static_cast<size_t>(i)], &table, &edge_idx, &key)) return -1;
    GP_TableEdgeTensorSpecTyped spec{};
    if (!gp_table_edge_get_tensor_spec(table, edge_idx, &spec)) return 0;
    return spec.type_id == static_cast<int32_t>(ValueTypeRegistry::global().builtin(VT_ABSTRACT_TENSOR)) ? 1 : 0;
}

extern "C" int gp_canvas_port_publish(GP_CanvasContext* ctx_, const char* name, const void* bytes, int len) {
    auto* c = reinterpret_cast<GP_CanvasContextImpl*>(ctx_);
    const int i = canvas_port_index(c, name);
    if (i < 0 || !bytes || len <= 0) return 0;
    const auto& p = c->canvas_ports[static_cast<size_t>(i)];
    if (p.direction != GP_CANVAS_PORT_IN) return 0;
    GP_TableContext* table = nullptr;
    int32_t edge_idx = -1;
    uint64_t key = 0;
    if (!canvas_port_edge(c, p, &table, &edge_idx, &key)) return 0;
    if (!gp_table_edge_can_publish(table, edge_idx, 1)) return 0;   // lossless: full means refused, never dropped
    int32_t dropped = 0;
    return gp_table_edge_publish(table, edge_idx, key, bytes, len, &dropped) ? 1 : 0;
}
