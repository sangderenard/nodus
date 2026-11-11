// Winsock headers must precede anything that may pull in <windows.h>.
#include "contract_port_canvas.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace nodus::contracts {

namespace {
constexpr unsigned long long kEmptyComposedReaderKey = 0x8000000000000000ULL;

// ThreadManager subscribers for ordinary scalar rows use composed contact
// identity; the all-zero contact has a reserved nonzero fallback because the
// FIFO rejects reader key zero. (Same rule the headless runtime always used.)
unsigned long long scalar_reader_key(int32_t module_idx, int32_t contact_idx) {
    unsigned long long key =
        (static_cast<unsigned long long>(static_cast<uint32_t>(module_idx)) << 32) |
        (static_cast<unsigned long long>(static_cast<uint32_t>(contact_idx)) << 16);
    return key == 0ull ? kEmptyComposedReaderKey : key;
}

bool resolve_table_edge(GP_TableContext* table, const GP_CanvasEdgeDesc& d, int32_t* out) {
    gp_table_apply_pending_ops(table);
    const uint64_t ka = (static_cast<uint64_t>(static_cast<uint32_t>(d.a_module)) << 32) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(d.a_contact_idx)) << 16);
    const uint64_t kb = (static_cast<uint64_t>(static_cast<uint32_t>(d.b_module)) << 32) |
                        (static_cast<uint64_t>(static_cast<uint32_t>(d.b_contact_idx)) << 16);
    return gp_table_edge_index_for_pair(table, ka, kb, out) && *out >= 0;
}

bool fail(std::string* error, const std::string& msg) {
    if (error) *error = msg;
    return false;
}

struct EdgeFifoRecord {
    GP_CanvasEdgeDesc desc{};
    int32_t token_bytes = 0, slots = 0, top_k = 0, type_id = 0, layout = 0, dtype = 0, backend_type = 0;
};
}  // namespace

struct CanvasContractPorts::Binding {
    std::string prefix;
    size_t capacity = 0;
    int32_t canvas_edge_idx = -1;
    GP_CanvasEdgeDesc desc{};
    uint16_t listen_port = 0;
    uint16_t requested_port = 0;
    // Cleared in the destructor, before the sockets die, so callbacks the
    // process-owned negotiator still holds become inert.
    std::shared_ptr<bool> alive = std::make_shared<bool>(true);
    std::unique_ptr<ContractListener> listener;
    std::vector<std::unique_ptr<ContractSocket>> hosts;
    std::vector<std::unique_ptr<EngineSocket>> engines;
    std::vector<uint64_t> host_ids, engine_ids;  // parallel to hosts / engines
    std::vector<std::pair<uint64_t, uint64_t>> rejected;  // (contract id, generation)
    ~Binding() { *alive = false; }
};

CanvasContractPorts::CanvasContractPorts() = default;

size_t CanvasContractPorts::size() const { return ports_.size(); }

CanvasContractPorts::~CanvasContractPorts() {
    clear();
    if (vulkan_backend_) gp_mem_backend_release(vulkan_backend_);
}

void CanvasContractPorts::clear() { ports_.clear(); }

bool CanvasContractPorts::declare(GP_CanvasContext* ctx, const std::string& prefix, size_t capacity,
                                  int canvas_edge_idx, uint16_t requested_port, std::string* error,
                                  uint64_t loan_ticks) {
    if (!ctx || prefix.empty() || capacity == 0 || capacity > INT32_MAX || canvas_edge_idx < 0)
        return fail(error, "contract port: invalid arguments");
    if (std::any_of(prefix.begin(), prefix.end(), [](unsigned char c) { return std::isspace(c) != 0; }))
        return fail(error, "contract port: prefix contains whitespace");
    GP_TableContext* table = gp_canvas_get_container_table(ctx);
    if (!table) return fail(error, "contract port: canvas has no container table");
    GP_CanvasEdgeDesc desc{};
    int type_id = 0;
    if (!gp_canvas_get_edge(ctx, canvas_edge_idx, &desc, &type_id))
        return fail(error, "contract port: unknown canvas edge");
    int32_t table_edge = -1;
    if (!resolve_table_edge(table, desc, &table_edge))
        return fail(error, "contract port: canvas edge has no table edge");
    GP_TableEdgeTensorSpecTyped spec{};
    if (!gp_table_edge_get_tensor_spec(table, table_edge, &spec) || spec.dim_count != 1 ||
        spec.elem_size <= 0 ||
        static_cast<int64_t>(spec.dims[0]) * spec.elem_size != static_cast<int64_t>(sizeof(Result)))
        return fail(error, "contract port: result edge is not configured for sizeof(Result) tokens");
    for (const auto& p : ports_)
        if (p->prefix == prefix || p->canvas_edge_idx == canvas_edge_idx)
            return fail(error, "contract port: prefix or edge already declared");
    try {
        auto b = std::make_unique<Binding>();
        b->listener = std::make_unique<ContractListener>(requested_port);
        ContractNegotiator& negotiator = acquire(prefix, capacity, loan_ticks);
        const std::shared_ptr<bool> alive = b->alive;
        const unsigned long long wkey = scalar_reader_key(desc.a_module, desc.a_contact_idx);
        const unsigned long long rkey = scalar_reader_key(desc.b_module, desc.b_contact_idx);
        if (!negotiator.bind_result_transport(
                [=](const Result& r) {
                    if (!*alive) return false;
                    int32_t dropped = 0;
                    const int32_t accepted = gp_table_edge_publish(table, table_edge, wkey, &r,
                                                                   static_cast<int32_t>(sizeof(r)), &dropped);
                    return accepted != 0 && !dropped;
                },
                [=](Result& r) {
                    if (!*alive) return false;
                    int32_t n = 0;
                    return gp_table_edge_consume(table, table_edge, rkey, &r,
                                                 static_cast<int32_t>(sizeof(r)), &n) != 0 &&
                           n == static_cast<int32_t>(sizeof(r));
                }))
            return fail(error, "contract port: negotiator result transport already bound");
        b->prefix = prefix;
        b->capacity = capacity;
        b->canvas_edge_idx = canvas_edge_idx;
        b->desc = desc;
        b->requested_port = requested_port;
        b->listen_port = b->listener->port();
        ports_.push_back(std::move(b));
        return true;
    } catch (const std::exception& e) {
        return fail(error, std::string("contract port: ") + e.what());
    } catch (...) {
        return fail(error, "contract port: unknown failure");
    }
}

// Accept new peers and pump engine sockets before the scheduled tick. Runs on
// the caller's lane; ThreadManager is idle between steps, so negotiator calls
// here and inside the tick never overlap.
void CanvasContractPorts::pre_step() {
    for (auto& up : ports_) {
        auto* cp = up.get();
        auto* negotiator = nodus::contracts::find(cp->prefix);
        if (!negotiator) continue;
        const std::shared_ptr<bool> alive = cp->alive;
        // Peers whose socket closed are removed first, so the same id can connect
        // again (a LIVE duplicate id is still refused by add_engine/add_host).
        for (size_t i = cp->engines.size(); i-- > 0;) {
            cp->engines[i]->poll();
            if (!cp->engines[i]->closed()) continue;
            negotiator->remove_engine(cp->engine_ids[i]);
            cp->engines.erase(cp->engines.begin() + i);
            cp->engine_ids.erase(cp->engine_ids.begin() + i);
        }
        for (size_t i = cp->hosts.size(); i-- > 0;) {
            if (!cp->hosts[i]->closed()) continue;
            negotiator->remove_host(cp->host_ids[i]);
            cp->hosts.erase(cp->hosts.begin() + i);
            cp->host_ids.erase(cp->host_ids.begin() + i);
        }
        cp->listener->poll([=](SOCKET s, uint64_t role, uint64_t id) {
            try {
                if (role == 2) {
                    cp->engines.push_back(std::make_unique<EngineSocket>(s));
                    auto* e = cp->engines.back().get();
                    auto* rejected = &cp->rejected;
                    if (!negotiator->add_engine({id,
                            [=](const Result& r) {
                                if (!*alive) return false;
                                const int decision = e->route(r);
                                if (decision == -1) rejected->emplace_back(r.loan.work.id, r.loan.generation);
                                return decision == 1;
                            },
                            [=](Contract& c) { return *alive && e->request(c); },
                            [=](uint64_t cid) { if (*alive) e->admitted(cid); }})) {
                        cp->engines.pop_back();  // duplicate id: socket closed by destructor
                    } else {
                        cp->engine_ids.push_back(id);
                    }
                } else {
                    cp->hosts.push_back(std::make_unique<ContractSocket>(s));
                    auto* h = cp->hosts.back().get();
                    // A closed host is removed in the next pre_step (its loans are re-offered).
                    if (!negotiator->add_host({id,
                            [=](const Loan* l) { return *alive ? h->exchange(l) : Exchange{}; },
                            [=](uint64_t gen, bool ok) { if (*alive) h->receipt(gen, ok); }})) {
                        cp->hosts.pop_back();
                    } else {
                        cp->host_ids.push_back(id);
                    }
                }
            } catch (...) {
                WireSocket::close_socket(s);  // constructor threw before adopting the socket
            }
        });
        for (auto& e : cp->engines) e->poll();
    }
}

void CanvasContractPorts::post_step() {
    for (auto& cp : ports_) {
        auto* negotiator = nodus::contracts::find(cp->prefix);
        if (negotiator)
            for (const auto& r : cp->rejected) negotiator->reject_pending(r.first, r.second);
        cp->rejected.clear();
    }
}

bool CanvasContractPorts::parse_records(const std::string& path, std::vector<Record>* out,
                                        std::string* error) {
    std::ifstream ifs(path);
    std::string line;
    while (std::getline(ifs, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag != "CONTRACT_PORT") continue;
        Record cp;
        if (!(ss >> cp.prefix >> cp.capacity >> cp.desc.a_module >> cp.desc.a_contact_idx >>
              cp.desc.b_module >> cp.desc.b_contact_idx >> cp.listen_port))
            return fail(error, "malformed record: " + line);
        uint64_t lt = 0;
        if (ss >> lt) cp.loan_ticks = lt;  // optional trailing field
        if (out) out->push_back(std::move(cp));
    }
    return true;
}

bool CanvasContractPorts::load_trailer(GP_CanvasContext* ctx, const std::string& path,
                                       std::string* error) {
    if (!ctx) return fail(error, "load_trailer: null canvas");
    GP_TableContext* table = gp_canvas_get_container_table(ctx);
    if (!table) return fail(error, "load_trailer: canvas has no container table");
    std::vector<EdgeFifoRecord> edges;
    std::vector<Record> records;
    {
        std::ifstream ifs(path);
        std::string line;
        while (std::getline(ifs, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            std::istringstream ss(line);
            std::string tag;
            ss >> tag;
            if (tag != "EDGE_FIFO") continue;
            EdgeFifoRecord e;
            if (!(ss >> e.desc.a_module >> e.desc.a_contact_idx >> e.desc.b_module >> e.desc.b_contact_idx >>
                  e.token_bytes >> e.slots >> e.top_k >> e.type_id >> e.layout >> e.dtype >> e.backend_type))
                return fail(error, "malformed record: " + line);
            edges.push_back(e);
        }
    }
    if (!parse_records(path, &records, error)) return false;
    if (records.empty()) return true;

    for (const auto& e : edges) {
        int32_t te = -1;
        if (gp_canvas_find_edge(ctx, &e.desc) < 0 || !resolve_table_edge(table, e.desc, &te))
            return fail(error, "cannot restore EDGE_FIFO: edge missing from canvas");
        if (e.token_bytes <= 0 || e.slots <= 0 || e.top_k < 0 || e.top_k > e.slots)
            return fail(error, "cannot restore EDGE_FIFO: invalid config");
        GP_TableEdgeTensorSpecTyped spec{};
        spec.dims[0] = e.token_bytes;
        spec.dim_count = 1;
        spec.slots = e.slots;
        spec.top_k = e.top_k;
        spec.elem_size = 1;
        spec.type_id = e.type_id;
        spec.layout = e.layout;
        spec.dtype = e.dtype;
        if (!gp_table_edge_set_tensor_spec(table, te, &spec))
            return fail(error, "cannot restore EDGE_FIFO: tensor spec refused");
        if (e.backend_type == GP_MEM_BACKEND_VULKAN) {
            if (!vulkan_backend_) vulkan_backend_ = gp_mem_backend_create_vulkan(nullptr);
            if (!vulkan_backend_ || !gp_table_edge_set_backend(table, te, vulkan_backend_))
                return fail(error, "cannot restore EDGE_FIFO: vulkan backend refused");
        } else if (e.backend_type != GP_MEM_BACKEND_CPU) {
            return fail(error, "cannot restore EDGE_FIFO: unsupported backend");
        }
        // Subscribe only after the spec rebuilt FIFO storage (start-at-head).
        if (!gp_table_edge_subscribe_ex(table, te, scalar_reader_key(e.desc.b_module, e.desc.b_contact_idx), 1))
            return fail(error, "cannot restore EDGE_FIFO: subscribe refused");
    }
    for (const auto& r : records) {
        const int canvas_edge = gp_canvas_find_edge(ctx, &r.desc);
        if (canvas_edge < 0 || r.capacity <= 0 || r.capacity > INT32_MAX || r.listen_port < 0 ||
            r.listen_port > 65535)
            return fail(error, "cannot restore CONTRACT_PORT " + r.prefix);
        std::string why;
        if (!declare(ctx, r.prefix, static_cast<size_t>(r.capacity), canvas_edge,
                     static_cast<uint16_t>(r.listen_port), &why, r.loan_ticks))
            return fail(error, "cannot restore CONTRACT_PORT " + r.prefix + ": " + why);
    }
    return true;
}

void CanvasContractPorts::write_trailer(std::ostream& os) const {
    for (const auto& cp : ports_) {
        os << "CONTRACT_PORT " << cp->prefix << " " << cp->capacity << " "
           << cp->desc.a_module << " " << cp->desc.a_contact_idx << " "
           << cp->desc.b_module << " " << cp->desc.b_contact_idx << " " << cp->requested_port;
        if (auto* n = nodus::contracts::find(cp->prefix); n && n->loan_ticks() != 60) os << " " << n->loan_ticks();
        os << "\n";
    }
}

bool CanvasContractPorts::info(size_t ordinal, PortInfo* out) const {
    if (ordinal >= ports_.size() || !out) return false;
    const auto& cp = *ports_[ordinal];
    out->prefix = cp.prefix;
    out->capacity = cp.capacity;
    out->canvas_edge_idx = cp.canvas_edge_idx;
    out->listen_port = cp.listen_port;
    out->requested_port = cp.requested_port;
    out->hosts = cp.hosts.size();
    out->engines = cp.engines.size();
    return true;
}

bool CanvasContractPorts::lookup(const std::string& prefix, PortInfo* out) const {
    for (size_t i = 0; i < ports_.size(); ++i)
        if (ports_[i]->prefix == prefix) return info(i, out);
    return false;
}

}  // namespace nodus::contracts
