#pragma once
// Contract ports for ONE canvas: canvas-owned loopback-socket transports bound
// to process-owned negotiator pairs (<prefix>.market/.worker rows, see
// contract_negotiator_registry.h) and the result edge between them. One
// implementation shared by the headless ABI (NodusHeadlessGraph) and the GUI
// shell (one instance per canvas head). Includes the winsock-bearing
// contract_listener.h first: include this header before <windows.h>.
#include "contract_listener.h"
#include "contract_negotiator_registry.h"
#include "canvas_abi.h"
#include "table_abi.h"
#include "mem_backend.h"
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace nodus::contracts {

class CanvasContractPorts {
public:
    // Snapshot of one declared port, for enumeration.
    struct PortInfo {
        std::string prefix;
        size_t capacity = 0;
        int32_t canvas_edge_idx = -1;
        uint16_t listen_port = 0;      // actual bound port
        uint16_t requested_port = 0;   // as declared (0 = ephemeral); what is saved
        size_t hosts = 0;
        size_t engines = 0;
    };
    // One CONTRACT_PORT trailer record.
    struct Record {
        std::string prefix;
        int64_t capacity = 0;
        GP_CanvasEdgeDesc desc{};
        int32_t listen_port = 0;
    };

    CanvasContractPorts();  // out of line as well: the defaulted body would instantiate the vector cleanup here
    ~CanvasContractPorts();  // listeners/sockets die first; negotiators stay process-owned
    CanvasContractPorts(const CanvasContractPorts&) = delete;
    CanvasContractPorts& operator=(const CanvasContractPorts&) = delete;

    // Acquires the pair from the registry, resolves canvas_edge_idx to its
    // table edge (container table, pair key (module<<32 | contact<<16)),
    // binds result transport publish/consume with the scalar reader keys and
    // opens the loopback listener. The edge must already hold a result-sized
    // FIFO spec (sizeof(Result)). Returns false (error filled) on refusal.
    bool declare(GP_CanvasContext* ctx, const std::string& prefix, size_t capacity,
                 int canvas_edge_idx, uint16_t requested_port, std::string* error);

    // Listener poll + Hello role dispatch + engine socket polls. Same lane as step.
    void pre_step();
    // reject_pending for rejections recorded during the tick.
    void post_step();

    // Apply EDGE_FIFO records (configure + subscribe the consumer reader) then
    // declare every CONTRACT_PORT record of a saved canvas file. The GUI's
    // gp_canvas_load_from_file skips both record kinds. A file with no
    // CONTRACT_PORT records is left untouched (returns true).
    bool load_trailer(GP_CanvasContext* ctx, const std::string& canvas_path, std::string* error);
    // CONTRACT_PORT lines only (EDGE_FIFO is written by the owner of edge state).
    void write_trailer(std::ostream& os) const;

    // Parse only the CONTRACT_PORT records of a file (false + error if malformed).
    static bool parse_records(const std::string& canvas_path, std::vector<Record>* out,
                              std::string* error);

    size_t size() const;  // out of line: Binding is complete only in the .cpp
    bool info(size_t ordinal, PortInfo* out) const;
    bool lookup(const std::string& prefix, PortInfo* out) const;
    // Drops all ports (sockets and listeners). Negotiators are NOT released.
    void clear();

private:
    struct Binding;
    std::vector<std::unique_ptr<Binding>> ports_;
    gp_mem_backend_handle_t vulkan_backend_ = nullptr;
};

}  // namespace nodus::contracts
