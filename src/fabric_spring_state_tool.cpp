// The fabric row on the tensor state library (include/fabric_spring_state_tool.h). The host code of
// turing src/compiler/spring_network_nodus_row.cpp.in is carried over unchanged in meaning: configure validation, the
// per-substep dt/contact/gather preparation, the finite check and feedback, freeze, substeps, the adaptive substep
// target, floor and tabletop contact, and NFAB UDP frames. What moved is where the state lives and who runs the law.
#include "fabric_spring_state_tool.h"

#include "cpu_fabric_dispatch_tool.h"
#include "gpu_shader_job_tool.h"
#include "kernel_pipeline_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace nodus::gpu {
namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kStageProgram = 0x66616272696331ull;   // "fabric1": the four host stages
constexpr std::uint64_t kLawProgram = 0x737072696e6731ull;     // "spring1": the SymPy law, CPU or GPU product
constexpr std::uint64_t kNeighbors = 27, kContact = 28, kDt = 29, kCommittedP = 30, kCommittedV = 31, kAddresses = 32;

std::array<std::uint64_t, 9> stage_receipt(std::uint64_t status, const std::array<std::uint64_t, 4>& w, std::uint64_t ns) {
    return {status, UINT64_MAX, w[2], ns, w[3], w[1], 0, ns, status == 0 ? 1ull : 0ull};
}

// ---- the four host stages ---------------------------------------------------------------------------------------
class FabricStageTool final : public ITool {
public:
    FabricStageTool(TensorStateLibraryTool& library, FabricStage stage, std::string id) : library_(library), stage_(stage), id_(std::move(id)) {}
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return "Fabric spring stage"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t i) const override {
        if (i == 0) return {ToolPortKind::Argument, 1};
        if (i == 1) return {ToolPortKind::Return, 1};
        return {};
    }
    void initialize(const ToolInitContext&) override {}
    void shutdown() override {}
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (!context.stack.raw) return;
        auto& stack = *context.stack.raw;
        nodus::tensors::AbstractTensorHandle input{};
        if (!raw_stack_pop_abstract_tensor(stack, input)) return;
        std::array<std::uint64_t, 4> w{};
        NodusTensorDesc d{};
        const auto t0 = Clock::now();
        std::uint64_t status = 1;
        if (nodus_tensor_describe(input.id, &d) == NODUS_OK && d.dtype == NODUS_DTYPE_U64 && d.rank == 1 && d.shape[0] == 4 &&
            nodus_tensor_read(input.id, 0, w.data(), sizeof(w)) == sizeof(w) && w[0] && w[2] == std::uint64_t(kFabricBatch))
            status = run(w[0], w[1]) ? 0 : 2;
        const auto ns = std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count());
        const std::uint64_t shape[1] = {9};
        const auto receipt = stage_receipt(status, w, ns);
        const auto out = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape);
        if (!out || nodus_tensor_write(out, 0, receipt.data(), sizeof(receipt)) != sizeof(receipt) || !raw_stack_push_abstract_tensor(stack, {out})) {
            if (out) nodus_tensor_destroy(out);
            raw_stack_push_abstract_tensor(stack, input); return;
        }
        context.succeeded = true;
    }

private:
    double* column(std::uint64_t card, std::uint64_t address, bool writable, std::vector<TensorStateLibraryTool::CpuLoan>& loans, std::size_t bytes) {
        auto loan = library_.borrow_cpu(card, address, writable);
        if (!loan || loan.bytes() < bytes) return nullptr;
        auto* p = static_cast<double*>(loan.data());
        loans.push_back(std::move(loan));
        return p;
    }
    bool run(std::uint64_t card, std::uint64_t base) {
        std::vector<TensorStateLibraryTool::CpuLoan> loans;
        const std::size_t lanes = std::size_t(kFabricBatch) * sizeof(double);
        auto col = [&](std::uint64_t i, bool w) { return column(card, base + i, w, loans, lanes); };
        switch (stage_) {
        case FabricStage::Begin: {
            double* p = col(17, true); double* v = col(23, true); const double* cp = col(kCommittedP, false); const double* cv = col(kCommittedV, false);
            if (!p || !v || !cp || !cv) return false;
            std::memcpy(p, cp, lanes); std::memcpy(v, cv, lanes); return true;
        }
        case FabricStage::Prepare: {
            auto contact_loan = library_.borrow_cpu(card, base + kContact, false);
            auto dt_loan = library_.borrow_cpu(card, base + kDt, false);
            auto table_loan = library_.borrow_cpu(card, base + kNeighbors, false);
            if (!contact_loan || contact_loan.bytes() < 8 * sizeof(double) || !dt_loan || dt_loan.bytes() < sizeof(double) ||
                !table_loan || table_loan.bytes() < std::size_t(kFabricNodeSlots) * 4 * sizeof(std::uint32_t)) return false;
            const double* contact = static_cast<const double*>(contact_loan.data());
            const double slice_dt = *static_cast<const double*>(dt_loan.data());
            const auto* table = static_cast<const std::uint32_t*>(table_loan.data());
            double* enabled = col(1, true); double* height = col(2, true); double* dt = col(4, true);
            const double* position = col(17, false);
            std::array<const double*, 4> mask{}; std::array<double*, 4> neighbor{};
            for (int e = 0; e < 4; ++e) { mask[e] = col(5 + e, false); neighbor[e] = col(11 + e, true); }
            if (!enabled || !height || !dt || !position) return false;
            for (int e = 0; e < 4; ++e) if (!mask[e] || !neighbor[e]) return false;
            const bool floor_on = contact[0] != 0, table_on = contact[2] != 0;
            std::fill(dt, dt + kFabricBatch, slice_dt);
            std::fill(enabled, enabled + kFabricBatch, 0.0);
            std::fill(height, height + kFabricBatch, 0.0);
            if (floor_on || table_on) for (int node = 0; node < kFabricNodeSlots; node++) {
                const int lane = node * kFabricAxes;
                if (floor_on) { enabled[lane + 1] = 1; height[lane + 1] = contact[1]; }
                if (table_on && position[lane] >= contact[4] && position[lane] <= contact[5] &&
                    position[lane + 2] >= contact[6] && position[lane + 2] <= contact[7]) { enabled[lane + 1] = 1; height[lane + 1] = contact[3]; }
            }
            for (int node = 0; node < kFabricNodeSlots; node++) for (int e = 0; e < 4; e++) {
                const int other = int(table[node * 4 + e]);
                for (int axis = 0; axis < kFabricAxes; axis++)
                    neighbor[e][node * kFabricAxes + axis] = mask[e][node * kFabricAxes + axis] ? position[other * kFabricAxes + axis] : 0;
            }
            return true;
        }
        case FabricStage::Feedback: {
            const double* np = col(24, false); const double* nv = col(25, false); double* p = col(17, true); double* v = col(23, true);
            if (!np || !nv || !p || !v) return false;
            for (int i = 0; i < kFabricBatch; i++) if (!std::isfinite(np[i]) || !std::isfinite(nv[i])) return false;
            std::memcpy(p, np, lanes); std::memcpy(v, nv, lanes); return true;
        }
        case FabricStage::Commit: {
            const double* p = col(17, false); const double* v = col(23, false); double* cp = col(kCommittedP, true); double* cv = col(kCommittedV, true);
            if (!p || !v || !cp || !cv) return false;
            std::memcpy(cp, p, lanes); std::memcpy(cv, v, lanes); return true;
        }
        }
        return false;
    }
    TensorStateLibraryTool& library_;
    FabricStage stage_;
    std::string id_;
};

// ---- the program row ----------------------------------------------------------------------------------------------
class FabricSpringRow final : public ITool {
public:
    FabricSpringRow(TensorStateLibraryTool& library, ToolRegistry& registry, std::string id, std::string pipeline, bool gpu,
                    std::uint64_t card, std::uint64_t base)
        : library_(library), registry_(registry), id_(std::move(id)), pipeline_id_(std::move(pipeline)), gpu_(gpu), card_(card), base_(base) {}
    ~FabricSpringRow() override { shutdown(); }
    const char* id_cstr() const noexcept override { return id_.c_str(); }
    const char* name_cstr() const noexcept override { return "Batched spring sheets and ropes (state library)"; }
    ToolCaps caps() const override { return ToolCaps::None; }
    int32_t port_count() const override { return 2; }
    ToolPortSpec port_spec(int32_t i) const override {
        if (i == 0) return {ToolPortKind::Argument, 1};
        if (i == 1) return {ToolPortKind::Return, 1};
        return {};
    }
    void initialize(const ToolInitContext&) override {
        pipeline_ = registry_.create(pipeline_id_);
        if (pipeline_) pipeline_->initialize({});
        std::fprintf(stderr, "[spring] state library, backend=%s\n", gpu_ ? "vulkan" : "llvm");
        WSADATA wsa{};
        if (WSAStartup(MAKEWORD(2, 2), &wsa) == 0) {
            winsock_ = true;
            const char* host = std::getenv("NODUS_FABRIC_UDP_HOST");
            const char* port = std::getenv("NODUS_FABRIC_UDP_PORT");
            addrinfo hints{}; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_DGRAM; hints.ai_protocol = IPPROTO_UDP;
            addrinfo* addresses = nullptr;
            if (getaddrinfo(host && *host ? host : "239.255.78.68", port && *port ? port : "48768", &hints, &addresses) == 0) {
                socket_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
                if (socket_ != INVALID_SOCKET && addresses->ai_addrlen <= sizeof(destination_)) {
                    std::memcpy(&destination_, addresses->ai_addr, addresses->ai_addrlen);
                    destination_bytes_ = int(addresses->ai_addrlen);
                    u_long nonblocking = 1; ioctlsocket(socket_, FIONBIO, &nonblocking);
                    int ttl = 4; setsockopt(socket_, IPPROTO_IP, IP_MULTICAST_TTL, reinterpret_cast<const char*>(&ttl), sizeof(ttl));
                }
                freeaddrinfo(addresses);
            }
        }
    }
    void shutdown() override {
        if (pipeline_) { pipeline_->shutdown(); pipeline_.reset(); }
        if (socket_ != INVALID_SOCKET) { closesocket(socket_); socket_ = INVALID_SOCKET; }
        if (winsock_) { WSACleanup(); winsock_ = false; }
        ready_ = false;
    }
    void tick(double, HostAPI&) override {}
    void render(RenderContext&) override {}
    void execute_stack(ToolStackContext& context) override {
        context.succeeded = false;
        if (!context.stack.raw || !pipeline_) return;
        auto& stack = *context.stack.raw;
        nodus::tensors::AbstractTensorHandle input{};
        if (!raw_stack_pop_abstract_tensor(stack, input)) return;
        auto restore = [&] { raw_stack_push_abstract_tensor(stack, input); };
        NodusTensorDesc description{};
        if (nodus_tensor_describe(input.id, &description) != NODUS_OK || description.dtype != NODUS_DTYPE_F64) { restore(); return; }
        std::uint64_t shape[3] = {kFabricNetworks, kFabricNodes, kFabricAxes};
        auto output = nodus::tensors::AbstractTensorHandle{nodus_tensor_create(NODUS_DTYPE_F64, 3, shape)};
        if (!output.id) { restore(); return; }
        bool okay = false, publish_frame = true;
        const bool config = description.rank == 3 && description.shape[0] == kFabricNetworks && description.shape[1] == kFabricNodes &&
                            description.shape[2] == kFabricFields;
        if (config) okay = configure(input.id, output.id);
        else if (description.rank == 0 && ready_) { publish_frame = !frozen_; okay = advance(input.id, output.id); }
        else if (description.rank == 1 && description.shape[0] == 2 && ready_) okay = control(input.id, output.id);
        else if (description.rank == 1 && description.shape[0] == 6 && ready_) { publish_frame = false; okay = table_contact(input.id, output.id); }
        if (!okay || !raw_stack_push_abstract_tensor(stack, output)) { nodus_tensor_destroy(output.id); restore(); return; }
        context.succeeded = true;
        if (publish_frame) publish();
    }

private:
    // The library row's own Card and Put requests, the same the shader and T4 rows' callers make.
    bool library_request(std::uint64_t op, std::uint64_t address, std::uint64_t payload) {
        std::array<std::uint64_t, 12> q{}; q[0] = op; q[1] = card_; q[2] = 1907; q[3] = address;
        if (op == 1) { q[4] = 64ull << 20; q[5] = q[6] = q[7] = 64ull << 20; }
        const std::uint64_t shape12[1] = {12}, shape1[1] = {1};
        const auto command = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape12);
        const bool own = !payload; if (own) payload = nodus_tensor_create(NODUS_DTYPE_U64, 1, shape1);
        if (!command || !payload) return false;
        nodus_tensor_write(command, 0, q.data(), sizeof(q));
        RawStackFrame frame{}; bool ok = false;
        if (raw_stack_init_frame(frame, 4096)) {
            raw_stack_push_abstract_tensor(frame, {command}); raw_stack_push_abstract_tensor(frame, {payload});
            ToolStackContext c{}; c.stack.raw = &frame; library_.execute_stack(c);
            nodus::tensors::AbstractTensorHandle r{};
            if (c.succeeded && raw_stack_pop_abstract_tensor(frame, r)) {
                std::uint64_t status = 1; nodus_tensor_read(r.id, 0, &status, sizeof(status)); ok = status == 0; nodus_tensor_destroy(r.id);
            }
            raw_stack_free_mask(frame); if (frame.backend) gp_mem_backend_release(frame.backend);
        }
        nodus_tensor_destroy(command); if (own) nodus_tensor_destroy(payload);
        return ok;
    }
    // An address is added once (the library refuses a second Put with StateConflict); later values are written
    // through a writable CPU loan, as the library's own rows do.
    bool put(std::uint64_t address, int32_t dtype, const void* data, std::uint64_t count, std::size_t element) {
        if (present_) {
            auto loan = library_.borrow_cpu(card_, base_ + address, true);
            if (!loan || loan.bytes() < count * element) return false;
            std::memcpy(loan.data(), data, count * element);
            return true;
        }
        const std::uint64_t shape[1] = {count};
        const auto t = nodus_tensor_create(dtype, 1, shape);
        if (!t) return false;
        const bool ok = nodus_tensor_write(t, 0, data, count * element) == int64_t(count * element) && library_request(2, base_ + address, t);
        nodus_tensor_destroy(t);
        return ok;
    }
    bool put_contact() {
        const double c[8] = {floor_enabled_ ? 1.0 : 0.0, floor_y_, tabletop_enabled_ ? 1.0 : 0.0, tabletop_y_,
                             table_min_x_, table_max_x_, table_min_z_, table_max_z_};
        return put(kContact, NODUS_DTYPE_F64, c, 8, sizeof(double));
    }
    std::vector<double> read(std::uint64_t address) {
        auto loan = library_.borrow_cpu(card_, base_ + address, false);
        if (!loan || loan.bytes() < std::size_t(kFabricBatch) * sizeof(double)) return {};
        const auto* p = static_cast<const double*>(loan.data());
        return {p, p + kFabricBatch};
    }
    bool write_committed_positions(std::uint64_t output) {
        auto p = read(kCommittedP);
        return p.size() == std::size_t(kFabricBatch) &&
               nodus_tensor_write(output, 0, p.data(), kFabricBatch * sizeof(double)) == kFabricBatch * int64_t(sizeof(double));
    }

    bool configure(std::uint64_t input, std::uint64_t output) {
        std::vector<double> source(std::size_t(kFabricNodeSlots) * kFabricFields);
        if (nodus_tensor_read(input, 0, source.data(), source.size() * sizeof(double)) != int64_t(source.size() * sizeof(double))) return false;
        double limit = 1e9;
        std::array<std::uint64_t, kFabricNetworks> fabric_ids{};
        for (int net = 0; net < kFabricNetworks; net++) {
            const double identity = source[(net * kFabricNodes) * kFabricFields + 37];
            if (!std::isfinite(identity) || identity < 0 || identity > 9007199254740991.0 || identity != std::floor(identity)) return false;
            fabric_ids[net] = std::uint64_t(identity);
            bool active = false; for (int node = 0; node < kFabricNodes; node++) active |= source[(net * kFabricNodes + node) * kFabricFields + 6] != 0;
            if (active) {
                if (!fabric_ids[net]) return false;
                for (int earlier = 0; earlier < net; earlier++) {
                    bool earlier_active = false;
                    for (int node = 0; node < kFabricNodes; node++) earlier_active |= source[(earlier * kFabricNodes + node) * kFabricFields + 6] != 0;
                    if (earlier_active && fabric_ids[earlier] == fabric_ids[net]) return false;
                }
            }
        }
        for (int node = 0; node < kFabricNodeSlots; node++) {
            const double* row = source.data() + node * kFabricFields;
            for (int field = 0; field < kFabricFields; field++) if (!std::isfinite(row[field])) return false;
            if ((row[6] != 0 && row[6] != 1) || (row[7] != 0 && row[7] != 1 && row[7] != 2) || row[11] < 0 || row[12] < 0 || row[13] < 0) return false;
            int degree = 0;
            for (int edge = 0; edge < 4; edge++) {
                if (row[33 + edge] != 0 && row[33 + edge] != 1) return false;
                if (row[33 + edge]) {
                    const int index = int(row[17 + edge]);
                    if (row[17 + edge] != index || index < 0 || index >= kFabricNodes) return false;
                    degree++;
                }
            }
            if (row[6] && !row[7] && degree && row[12] && row[11]) limit = std::min(limit, 1 / std::sqrt(2.0 * degree * row[12] * row[11]));
        }
        std::array<std::vector<double>, 26> columns{};
        for (auto& c : columns) c.assign(kFabricBatch, 0.0);
        std::vector<std::uint32_t> neighbors(std::size_t(kFabricNodeSlots) * 4);
        for (int node = 0; node < kFabricNodeSlots; node++) {
            const double* row = source.data() + node * kFabricFields;
            const int net = node / kFabricNodes;
            for (int edge = 0; edge < 4; edge++) neighbors[node * 4 + edge] = std::uint32_t(net * kFabricNodes + (row[33 + edge] ? int(row[17 + edge]) : 0));
            for (int axis = 0; axis < kFabricAxes; axis++) {
                const int lane = node * kFabricAxes + axis;
                columns[17][lane] = row[axis]; columns[23][lane] = row[3 + axis];
                columns[0][lane] = row[8 + axis]; columns[3][lane] = row[13];
                columns[9][lane] = row[14 + axis]; columns[10][lane] = row[11];
                columns[15][lane] = row[6]; columns[16][lane] = (row[7] == 2 ? (axis == 0 ? 0 : 1) : row[7]);
                columns[22][lane] = row[12];
                for (int edge = 0; edge < 4; edge++) { columns[5 + edge][lane] = row[33 + edge]; columns[18 + edge][lane] = row[21 + edge * 3 + axis]; }
            }
        }
        // The card is the one given at registration: created here if it is new, used as it is if its owner made it
        // already (the library answers Invalid for an existing card; a card that does not exist fails the Puts below).
        if (!carded_) { library_request(1, 0, 0); carded_ = true; }
        for (std::uint64_t i = 0; i < 26; i++) if (!put(i, NODUS_DTYPE_F64, columns[i].data(), kFabricBatch, sizeof(double))) return false;
        const std::uint32_t count = std::uint32_t(kFabricBatch); const double zero = 0.0;
        floor_enabled_ = tabletop_enabled_ = false;
        if (!put(26, NODUS_DTYPE_U32, &count, 1, sizeof(count)) || !put(kNeighbors, NODUS_DTYPE_U32, neighbors.data(), neighbors.size(), sizeof(std::uint32_t)) ||
            !put_contact() || !put(kDt, NODUS_DTYPE_F64, &zero, 1, sizeof(double)) ||
            !put(kCommittedP, NODUS_DTYPE_F64, columns[17].data(), kFabricBatch, sizeof(double)) ||
            !put(kCommittedV, NODUS_DTYPE_F64, columns[23].data(), kFabricBatch, sizeof(double))) return false;
        present_ = true;
        occupied_ = columns[15];
        if (nodus_tensor_write(output, 0, columns[17].data(), kFabricBatch * sizeof(double)) != kFabricBatch * int64_t(sizeof(double))) return false;
        max_dt_ = limit; fabric_ids_ = fabric_ids; frozen_ = false; ready_ = true;
        return true;
    }
    bool advance(std::uint64_t input, std::uint64_t output) {
        double dt = 0;
        int selected = substeps_;
        if (gpu_ && target_ms_ > 0) {
            selected = 1;
            if (step_samples_) {
                double total = 0; for (int i = 0; i < step_samples_; i++) total += recent_step_ms_[i];
                const double average = total / step_samples_;
                if (average > 0) selected = std::clamp(int(target_ms_ / average), 1, substeps_);
            }
        }
        if (nodus_tensor_read(input, 0, &dt, sizeof(dt)) != sizeof(dt) || !std::isfinite(dt) || dt <= 0 || dt / double(selected) > max_dt_) return false;
        if (frozen_) return write_committed_positions(output);
        const double slice_dt = dt / double(selected);
        if (!put(kDt, NODUS_DTYPE_F64, &slice_dt, 1, sizeof(double))) return false;
        // One schedule per tick: begin, (prepare, law, feedback) per substep, commit. Components: 0 begin, 1 prepare,
        // 2 law, 3 feedback, 4 commit (the pipeline registered for this row's backend).
        std::vector<std::array<std::uint64_t, 7>> rows;
        auto add = [&](std::uint64_t component, std::uint64_t api, std::uint64_t program) {
            rows.push_back({component, api, program, card_, base_, std::uint64_t(kFabricBatch), ++generation_});
        };
        const std::uint64_t cpu = std::uint64_t(TensorStateLibraryTool::ExecutionApi::Cpu);
        const std::uint64_t law_api = gpu_ ? std::uint64_t(TensorStateLibraryTool::ExecutionApi::Vulkan) : cpu;
        add(0, cpu, kStageProgram);
        for (int step = 0; step < selected; step++) { add(1, cpu, kStageProgram); add(2, law_api, kLawProgram); add(3, cpu, kStageProgram); }
        add(4, cpu, kStageProgram);
        const std::uint64_t shape[2] = {rows.size(), 7};
        const auto schedule = nodus_tensor_create(NODUS_DTYPE_U64, 2, shape);
        if (!schedule) return false;
        nodus_tensor_write(schedule, 0, rows.data(), rows.size() * sizeof(rows[0]));
        RawStackFrame frame{}; bool ok = false;
        std::vector<std::array<std::uint64_t, 9>> receipts(rows.size() + 1);
        if (raw_stack_init_frame(frame, 4096)) {
            raw_stack_push_abstract_tensor(frame, {schedule});
            ToolStackContext c{}; c.stack.raw = &frame; pipeline_->execute_stack(c);
            nodus::tensors::AbstractTensorHandle r{};
            if (c.succeeded && raw_stack_pop_abstract_tensor(frame, r)) {
                nodus_tensor_read(r.id, 0, receipts.data(), receipts.size() * sizeof(receipts[0]));
                ok = receipts[0][0] == 0 && receipts[0][3] == rows.size();
                nodus_tensor_destroy(r.id);
            }
            raw_stack_free_mask(frame); if (frame.backend) gp_mem_backend_release(frame.backend);
        }
        nodus_tensor_destroy(schedule);
        if (!ok) { std::fprintf(stderr, "[spring] tick refused by the state-library schedule; state kept\n"); return false; }
        for (std::size_t i = 0; i < rows.size(); i++) if (rows[i][0] == 2) {     // the law's own elapsed time per substep
            recent_step_ms_[step_window_next_] = double(receipts[i + 1][3]) / 1e6;
            step_window_next_ = (step_window_next_ + 1) % int(recent_step_ms_.size());
            step_samples_ = std::min(step_samples_ + 1, int(recent_step_ms_.size()));
        }
        last_law_ns_ = 0; for (std::size_t i = 0; i < rows.size(); i++) if (rows[i][0] == 2) last_law_ns_ += receipts[i + 1][3];
        return write_committed_positions(output);
    }
    bool control(std::uint64_t input, std::uint64_t output) {
        double command[2]{};
        if (nodus_tensor_read(input, 0, command, sizeof(command)) != sizeof(command)) return false;
        if (command[0] == 3) {
            if (!std::isfinite(command[1]) || command[1] < 1 || command[1] > 128 || command[1] != std::floor(command[1])) return false;
            if (!write_committed_positions(output)) return false;
            substeps_ = int(command[1]); return true;
        }
        if (command[0] == 4) {
            if (!std::isfinite(command[1]) || command[1] < 0 || command[1] > 8) return false;
            if (!write_committed_positions(output)) return false;
            target_ms_ = command[1]; step_samples_ = 0; step_window_next_ = 0; return true;
        }
        if (command[0] == 5) {
            if (!std::isfinite(command[1])) return false;
            if (!write_committed_positions(output)) return false;
            floor_y_ = command[1]; floor_enabled_ = true; return put_contact();
        }
        if (command[0] != 1 || (command[1] != 0 && command[1] != 1)) return false;
        if (!write_committed_positions(output)) return false;
        frozen_ = command[1] == 1; return true;
    }
    bool table_contact(std::uint64_t input, std::uint64_t output) {
        double command[6]{};
        if (nodus_tensor_read(input, 0, command, sizeof(command)) != sizeof(command) || command[0] != 2 ||
            !std::all_of(command + 1, command + 6, [](double x) { return std::isfinite(x); }) || !(command[2] < command[3]) || !(command[4] < command[5])) return false;
        if (!write_committed_positions(output)) return false;
        tabletop_y_ = command[1]; table_min_x_ = command[2]; table_max_x_ = command[3]; table_min_z_ = command[4]; table_max_z_ = command[5];
        tabletop_enabled_ = true;
        return put_contact();
    }
    void publish() {
        if (socket_ == INVALID_SOCKET || !destination_bytes_ || !ready_) return;
        const auto positions = read(kCommittedP);
        if (positions.size() != std::size_t(kFabricBatch)) return;
        // NFAB v1 little-endian: 32-byte header, then up to 48 (node id, xyz) pairs.
        ++frame_;
        for (int net = 0; net < kFabricNetworks; net++) {
            int active = 0; for (int node = 0; node < kFabricNodes; node++) if (occupied_[(net * kFabricNodes + node) * kFabricAxes]) active++;
            if (!active) continue;
            const std::uint16_t chunks = std::uint16_t((active + 47) / 48);
            int emitted = 0, chunk = 0;
            while (emitted < active) {
                std::array<std::uint8_t, 800> packet{}; std::size_t size = 0;
                auto append = [&](const void* data, std::size_t bytes) { std::memcpy(packet.data() + size, data, bytes); size += bytes; };
                const std::uint32_t magic = 0x4241464e; const std::uint16_t version = 1, header = 32;
                const std::uint16_t part = std::uint16_t(chunk++), reserved = frozen_ ? 1 : 0;
                append(&magic, 4); append(&version, 2); append(&header, 2); append(&frame_, 8);
                append(&fabric_ids_[net], 8); append(&part, 2); append(&chunks, 2);
                const std::size_t count_offset = size; append(&reserved, 2); append(&reserved, 2);
                std::uint16_t count = 0; int skip = emitted;
                for (int node = 0; node < kFabricNodes && count < 48; node++) {
                    const int lane = (net * kFabricNodes + node) * kFabricAxes;
                    if (!occupied_[lane]) continue;
                    if (skip > 0) { --skip; continue; }
                    const std::uint32_t id = std::uint32_t(node);
                    const float vector[3] = {float(positions[lane]), float(positions[lane + 1]), float(positions[lane + 2])};
                    append(&id, 4); append(vector, sizeof(vector)); count++;
                }
                std::memcpy(packet.data() + count_offset, &count, 2);
                emitted += count;
                sendto(socket_, reinterpret_cast<const char*>(packet.data()), int(size), 0, reinterpret_cast<const sockaddr*>(&destination_), destination_bytes_);
            }
        }
    }

    TensorStateLibraryTool& library_;
    ToolRegistry& registry_;
    std::string id_, pipeline_id_;
    bool gpu_;
    std::uint64_t card_, base_, generation_ = 0, last_law_ns_ = 0;
    std::unique_ptr<ITool, std::function<void(ITool*)>> pipeline_;
    bool carded_ = false, present_ = false, ready_ = false, frozen_ = false;
    double max_dt_ = 0; int substeps_ = 1;
    double target_ms_ = 0; std::array<double, 8> recent_step_ms_{}; int step_samples_ = 0, step_window_next_ = 0;
    std::vector<double> occupied_;
    std::array<std::uint64_t, kFabricNetworks> fabric_ids_{}; std::uint64_t frame_ = 0;
    bool winsock_ = false; SOCKET socket_ = INVALID_SOCKET;
    bool tabletop_enabled_ = false, floor_enabled_ = false; double floor_y_ = 0;
    double tabletop_y_ = 0, table_min_x_ = 0, table_max_x_ = 0, table_min_z_ = 0, table_max_z_ = 0;
    sockaddr_storage destination_{}; int destination_bytes_ = 0;
};

}  // namespace

bool register_fabric_spring(ToolRegistry& registry, TensorStateLibraryTool& library, const FabricSpringProducts& products,
                            const std::string& row_id, const std::string& backend, std::uint64_t card, std::uint64_t base,
                            std::vector<std::string>* registered) {
    std::vector<std::string> ids;
    auto undo = [&] { for (auto it = ids.rbegin(); it != ids.rend(); ++it) registry.unregister_tool(*it); return false; };
    const std::pair<FabricStage, const char*> stages[] = {{FabricStage::Begin, "nodus.fabric.spring.begin"}, {FabricStage::Prepare, "nodus.fabric.spring.prepare"},
                                                         {FabricStage::Feedback, "nodus.fabric.spring.feedback"}, {FabricStage::Commit, "nodus.fabric.spring.commit"}};
    for (const auto& [stage, id] : stages) {
        if (registry.find(id)) continue;                                   // shared by every fabric row of this library
        ToolRegistry::Entry e; e.id = id; e.name = "Fabric spring stage"; e.caps = ToolCaps::None;
        e.factory = [&library, stage = stage, id = std::string(id)] {
            return std::unique_ptr<ITool, std::function<void(ITool*)>>(new FabricStageTool(library, stage, id), [](ITool* t) { delete t; });
        };
        if (!registry.register_tool(std::move(e))) return undo();
        ids.push_back(id);
    }
    // The law: Turing's LLVM product as the existing CPU row, and its GLSL/glslang product as a shader job per device.
    if (!registry.find("nodus.fabric.dispatch.cpu")) {
        if (!register_cpu_fabric_dispatcher(registry, library, products.llvm_dll, products.llvm_entry, std::uint64_t(kFabricBatch), 1, kLawProgram)) return undo();
        ids.push_back("nodus.fabric.dispatch.cpu");
    }
    GpuShaderJobContract contract;
    contract.id = "nodus.shader.turing.spring_network_step"; contract.program_id = kLawProgram; contract.spirv = products.vulkan_spirv;
    for (int i = 0; i < 27; i++) contract.minimum_binding_bytes.push_back(i == 26 ? sizeof(std::uint32_t) : std::size_t(kFabricBatch) * sizeof(double));
    contract.writable.assign(27, 0); contract.writable[24] = contract.writable[25] = 1;
    contract.local_size_x = products.local_size_x; contract.maximum_invocations = std::uint32_t(kFabricBatch);
    const std::string gpu_first = contract.id + ".vulkan.0";
    if (!products.vulkan_spirv.empty() && library.device_count() && !registry.find(gpu_first)) {
        if (!register_gpu_shader_job(registry, library, contract)) return undo();
        for (std::size_t d = 0; d < library.device_count(); d++) ids.push_back(contract.id + ".vulkan." + std::to_string(d));
    }
    const bool gpu = backend.rfind("vulkan.", 0) == 0;
    const std::string law = gpu ? contract.id + "." + backend : std::string("nodus.fabric.dispatch.cpu");
    const auto cpu = TensorStateLibraryTool::ExecutionApi::Cpu;
    const std::string pipeline = "nodus.pipeline.fabric.spring." + backend;
    if (!registry.find(pipeline)) {
        if (!register_kernel_pipeline(registry, pipeline, {
                {"nodus.fabric.spring.begin", cpu, kStageProgram}, {"nodus.fabric.spring.prepare", cpu, kStageProgram},
                {law, gpu ? TensorStateLibraryTool::ExecutionApi::Vulkan : cpu, kLawProgram},
                {"nodus.fabric.spring.feedback", cpu, kStageProgram}, {"nodus.fabric.spring.commit", cpu, kStageProgram}})) return undo();
        ids.push_back(pipeline);
    }
    ToolRegistry::Entry row; row.id = row_id; row.name = "Batched spring sheets and ropes (state library)"; row.caps = ToolCaps::None;
    row.factory = [&library, &registry, row_id, pipeline, gpu, card, base] {
        return std::unique_ptr<ITool, std::function<void(ITool*)>>(new FabricSpringRow(library, registry, row_id, pipeline, gpu, card, base),
                                                                   [](ITool* t) { delete t; });
    };
    if (!registry.register_tool(std::move(row))) return undo();
    ids.push_back(row_id);
    if (registered) registered->insert(registered->end(), ids.begin(), ids.end());
    return true;
}

}  // namespace nodus::gpu
