#pragma once
// Every tensor state library job kind as a work-market contract (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 3).
//
// A contract carries ids only (the market's rule): Contract{id = work id, revision = definition id, payload = work id}.
//   Definition  one per job kind a host may run: shader job (binding contract, GLSL by digest), pipeline (components),
//               CPU law (Turing LLVM DLL), fftfree (fft_cffi DLL), wave T4 (wave_t4 DLL), fabric program row (stateful).
//               Products travel as Artifacts by SHA-256; a shader's Vulkan build follows stage 2 (compiled for the host's
//               settings, or by the host's own glslang).
//   WorkItem    the state to write into the host's card, the row inputs to push in order, the addresses to read back.
//   WorkOutputs the row's own output tensor and the read-back state, returned before the Result.
// LibraryRunner registers definitions in a KernelLibrarySession and runs work items on its rows; the volunteer, the
// server's local host and the server's re-check all use it. Existing library state is written through writable CPU
// loans (a second Put of an address is refused with StateConflict).
#include "nodus_link_settings.h"
#include "kernel_library_session.h"
#include "cpu_fabric_dispatch_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace nodus::link {
namespace fs = std::filesystem;

// ---- little-endian records ---------------------------------------------------------------------------------------
struct Writer {
    std::vector<uint8_t> b;
    void u(uint64_t v, int n = 8) { const size_t at = b.size(); b.resize(at + size_t(n)); put_le(b.data() + at, v, size_t(n)); }
    void blob(const void* p, size_t n) { u(n); const auto* c = static_cast<const uint8_t*>(p); b.insert(b.end(), c, c + n); }
    void blob(const std::vector<uint8_t>& v) { blob(v.data(), v.size()); }
    void str(const std::string& s) { blob(s.data(), s.size()); }
    void digest(const std::array<uint8_t, 32>& d) { b.insert(b.end(), d.begin(), d.end()); }
};
struct Reader {
    const std::vector<uint8_t>& b; size_t at = 0; bool ok = true;
    explicit Reader(const std::vector<uint8_t>& v) : b(v) {}
    uint64_t u(int n = 8) { if (at + size_t(n) > b.size()) { ok = false; return 0; } auto v = get_le(b.data() + at, size_t(n)); at += size_t(n); return v; }
    std::vector<uint8_t> blob() { const uint64_t n = u(); if (!ok || at + n > b.size()) { ok = false; return {}; } std::vector<uint8_t> v(b.begin() + ptrdiff_t(at), b.begin() + ptrdiff_t(at + n)); at += size_t(n); return v; }
    std::string str() { auto v = blob(); return std::string(v.begin(), v.end()); }
    std::array<uint8_t, 32> digest() { std::array<uint8_t, 32> d{}; if (at + 32 > b.size()) { ok = false; return d; } std::memcpy(d.data(), b.data() + at, 32); at += 32; return d; }
};

// A tensor as bytes: dtype, shape, contents.
struct TensorBytes { int32_t dtype = 0; std::vector<uint64_t> shape; std::vector<uint8_t> bytes; };
inline void write_tensor(Writer& w, const TensorBytes& t) { w.u(uint32_t(t.dtype), 4); w.u(t.shape.size(), 4); for (auto s : t.shape) w.u(s); w.blob(t.bytes); }
inline TensorBytes read_tensor(Reader& r) { TensorBytes t; t.dtype = int32_t(r.u(4)); const auto n = r.u(4); for (uint64_t i = 0; i < n && r.ok; ++i) t.shape.push_back(r.u()); t.bytes = r.blob(); return t; }
inline TensorBytes tensor_of(uint64_t handle) {
    TensorBytes t; NodusTensorDesc d{};
    if (nodus_tensor_describe(handle, &d) != NODUS_OK) return t;
    t.dtype = d.dtype; t.shape.assign(d.shape, d.shape + d.rank); t.bytes.resize(size_t(d.total_bytes));
    nodus_tensor_read(handle, 0, t.bytes.data(), t.bytes.size()); return t;
}
inline uint64_t handle_of(const TensorBytes& t) {
    const auto h = nodus_tensor_create(t.dtype, uint32_t(t.shape.size()), t.shape.empty() ? nullptr : t.shape.data());
    if (h && !t.bytes.empty()) nodus_tensor_write(h, 0, t.bytes.data(), t.bytes.size());
    return h;
}
template <class T> TensorBytes tensor1(int32_t dtype, const std::vector<T>& v) {
    TensorBytes t; t.dtype = dtype; t.shape = {v.size()}; t.bytes.resize(v.size() * sizeof(T)); std::memcpy(t.bytes.data(), v.data(), t.bytes.size()); return t;
}

// ---- definitions -------------------------------------------------------------------------------------------------
enum class JobKind : uint32_t { ShaderJob = 1, Pipeline = 2, CpuLaw = 3, Fftfree = 4, WaveT4 = 5, FabricProgram = 6 };
inline const char* kind_name(JobKind k) {
    switch (k) { case JobKind::ShaderJob: return "shader job"; case JobKind::Pipeline: return "pipeline"; case JobKind::CpuLaw: return "CPU law";
                 case JobKind::Fftfree: return "fftfree"; case JobKind::WaveT4: return "wave T4"; case JobKind::FabricProgram: return "fabric program"; }
    return "?";
}
struct Definition {
    uint64_t id = 0; JobKind kind = JobKind::ShaderJob; bool stateful = false;
    std::string name;                                 // the registered id: shader job id, pipeline id, fabric row id
    // shader job (and the GPU law of a fabric program)
    uint64_t program_id = 0; std::array<uint8_t, 32> glsl{}; bool has_glsl = false; ShaderNeeds needs;
    std::vector<uint64_t> binding_bytes; std::vector<uint8_t> writable; std::vector<uint64_t> offsets;
    uint32_t local_x = 0, max_x = 0;
    // pipeline: component rows; `device_bound` rows run on <tool>.vulkan.<device>
    struct Component { std::string tool; uint32_t api = 0; uint64_t program = 0; bool device_bound = false; };
    std::vector<Component> components;
    // CPU products (LLVM law, fftfree, T4, fabric's CPU law)
    std::array<uint8_t, 32> dll{}; bool has_dll = false; std::string entry; uint64_t batch = 0;
    // fabric program
    std::string backend; uint64_t card = 0, base = 0;
};
inline std::vector<uint8_t> encode_definition(const Definition& d) {
    Writer w; w.u(0x31464544, 4); w.u(d.id); w.u(uint32_t(d.kind), 4); w.u(d.stateful ? 1 : 0, 4); w.str(d.name);
    w.u(d.program_id); w.u(d.has_glsl, 4); w.digest(d.glsl); w.u(d.needs.float64, 4); w.u(d.needs.local_size_x, 4);
    w.u(d.binding_bytes.size()); for (auto v : d.binding_bytes) w.u(v);
    w.blob(d.writable); w.u(d.offsets.size()); for (auto v : d.offsets) w.u(v);
    w.u(d.local_x, 4); w.u(d.max_x, 4);
    w.u(d.components.size()); for (const auto& c : d.components) { w.str(c.tool); w.u(c.api, 4); w.u(c.program); w.u(c.device_bound, 4); }
    w.u(d.has_dll, 4); w.digest(d.dll); w.str(d.entry); w.u(d.batch); w.str(d.backend); w.u(d.card); w.u(d.base);
    return w.b;
}
inline bool decode_definition(const std::vector<uint8_t>& b, Definition& d) {
    Reader r(b); if (r.u(4) != 0x31464544) return false;
    d = {}; d.id = r.u(); d.kind = JobKind(r.u(4)); d.stateful = r.u(4) != 0; d.name = r.str();
    d.program_id = r.u(); d.has_glsl = r.u(4) != 0; d.glsl = r.digest(); d.needs.float64 = r.u(4) != 0; d.needs.local_size_x = uint32_t(r.u(4));
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) d.binding_bytes.push_back(r.u());
    d.writable = r.blob(); for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) d.offsets.push_back(r.u());
    d.local_x = uint32_t(r.u(4)); d.max_x = uint32_t(r.u(4));
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) { Definition::Component c; c.tool = r.str(); c.api = uint32_t(r.u(4)); c.program = r.u(); c.device_bound = r.u(4) != 0; d.components.push_back(c); }
    d.has_dll = r.u(4) != 0; d.dll = r.digest(); d.entry = r.str(); d.batch = r.u(); d.backend = r.str(); d.card = r.u(); d.base = r.u();
    return r.ok;
}

// ---- work ----------------------------------------------------------------------------------------------------------
struct WorkItem {
    uint64_t id = 0, definition = 0, card = 0;
    std::vector<std::pair<uint64_t, TensorBytes>> puts;   // library state written before the row runs
    std::vector<TensorBytes> inputs;                      // pushed in order (work token, then control, ...)
    std::vector<uint64_t> outputs;                        // addresses read back after the row
};
struct WorkOutputs {
    uint64_t id = 0, status = 1;                          // 0 = the row succeeded and its receipt said 0
    TensorBytes row_output;
    std::vector<std::pair<uint64_t, std::vector<uint8_t>>> state;
    double ms = 0;
    bool operator==(const WorkOutputs& o) const {
        if (status != o.status || row_output.dtype != o.row_output.dtype || row_output.shape != o.row_output.shape || state.size() != o.state.size()) return false;
        // a receipt carries timings: compare a U64 receipt's status word only, every other output byte for byte
        const bool receipt = row_output.dtype == NODUS_DTYPE_U64 && row_output.bytes.size() <= 32 * 8 * 16;
        if (receipt ? (row_output.bytes.size() < 8 || o.row_output.bytes.size() < 8 || std::memcmp(row_output.bytes.data(), o.row_output.bytes.data(), 8) != 0)
                    : row_output.bytes != o.row_output.bytes) return false;
        for (size_t i = 0; i < state.size(); ++i) if (state[i] != o.state[i]) return false;
        return true;
    }
};
inline std::vector<uint8_t> encode_work(const WorkItem& w) {
    Writer o; o.u(0x314B5257, 4); o.u(w.id); o.u(w.definition); o.u(w.card);
    o.u(w.puts.size()); for (const auto& [a, t] : w.puts) { o.u(a); write_tensor(o, t); }
    o.u(w.inputs.size()); for (const auto& t : w.inputs) write_tensor(o, t);
    o.u(w.outputs.size()); for (auto a : w.outputs) o.u(a);
    return o.b;
}
inline bool decode_work(const std::vector<uint8_t>& b, WorkItem& w) {
    Reader r(b); if (r.u(4) != 0x314B5257) return false;
    w = {}; w.id = r.u(); w.definition = r.u(); w.card = r.u();
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) { const auto a = r.u(); w.puts.push_back({a, read_tensor(r)}); }
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) w.inputs.push_back(read_tensor(r));
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) w.outputs.push_back(r.u());
    return r.ok;
}
inline std::vector<uint8_t> encode_outputs(const WorkOutputs& x) {
    Writer o; o.u(0x3154554F, 4); o.u(x.id); o.u(x.status); write_tensor(o, x.row_output);
    o.u(x.state.size()); for (const auto& [a, v] : x.state) { o.u(a); o.blob(v); }
    o.u(uint64_t(x.ms * 1e6));
    return o.b;
}
inline bool decode_outputs(const std::vector<uint8_t>& b, WorkOutputs& x) {
    Reader r(b); if (r.u(4) != 0x3154554F) return false;
    x = {}; x.id = r.u(); x.status = r.u(); x.row_output = read_tensor(r);
    for (uint64_t i = 0, n = r.u(); i < n && r.ok; ++i) { const auto a = r.u(); x.state.push_back({a, r.blob()}); }
    x.ms = double(r.u()) / 1e6;
    return r.ok;
}

// ---- the runner ----------------------------------------------------------------------------------------------------
class LibraryRunner {
public:
    LibraryRunner(fs::path work, std::string glslang, std::string target, uint32_t device = 0)
        : work_(std::move(work)), target_(std::move(target)), device_(device), builds_(glslang, work_ / "builds"), glslang_(!glslang.empty()) {
        std::error_code ec; fs::create_directories(work_ / "artifacts", ec);
        if (!raw_stack_init_frame(frame_, 1 << 16)) frame_ = {};
    }
    ~LibraryRunner() {
        raw_stack_free_mask(frame_); if (frame_.backend) gp_mem_backend_release(frame_.backend);
    }
    explicit operator bool() const { return bool(session_); }
    nodus::gpu::KernelLibrarySession& session() { return session_; }
    uint32_t device() const { return device_; }
    bool has_local_glslang() const { return glslang_; }

    // Products by digest. A Vulkan build arrives per (GLSL digest, target) from the server; a host with glslang builds its own.
    void add_artifact(const std::array<uint8_t, 32>& d, const std::vector<uint8_t>& bytes) { artifacts_[d] = bytes; }
    void add_vulkan(const std::array<uint8_t, 32>& glsl, const std::vector<uint8_t>& spirv) { offered_vulkan_[glsl] = spirv; }
    bool has_artifact(const std::array<uint8_t, 32>& d) const { return artifacts_.count(d) != 0; }
    uint64_t local_builds_matching() const { return local_matched_; }
    uint64_t local_builds_differing() const { return local_differed_; }

    // Registers a definition once its products are here. Returns "" when registered, "pending" while products or a
    // build are outstanding, otherwise the refusal.
    std::string register_definition(const Definition& d) {
        if (ready_.count(d.id)) return "";
        definitions_[d.id] = d;
        auto dll_path = [&](const std::array<uint8_t, 32>& digest, std::string& path) {
            auto it = artifacts_.find(digest); if (it == artifacts_.end()) return false;
            path = (work_ / "artifacts" / (hex(digest) + ".dll")).string();
            if (!fs::exists(path)) std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(it->second.data()), std::streamsize(it->second.size()));
            return true;
        };
        std::vector<uint32_t> spirv;
        if ((d.kind == JobKind::ShaderJob || (d.kind == JobKind::FabricProgram && d.backend.rfind("vulkan.", 0) == 0)) && d.has_glsl) {
            const std::string why = vulkan_for(d.glsl, spirv);
            if (!why.empty()) return why;
        }
        bool ok = false;
        switch (d.kind) {
        case JobKind::ShaderJob: {
            nodus::gpu::GpuShaderJobContract c; c.id = d.name; c.program_id = d.program_id; c.spirv = spirv;
            c.minimum_binding_bytes = d.binding_bytes; c.writable = d.writable; c.binding_offsets = d.offsets;
            c.local_size_x = d.local_x; c.maximum_invocations = d.max_x;
            ok = session_.register_shader(c);
            break;
        }
        case JobKind::Pipeline: {   // registered per device on first use (ensure_pipeline_on); its components must exist
            for (const auto& c : d.components) if (c.device_bound && !tool_registry_global().find(c.tool + ".vulkan.0")) return "pending";
            ok = true;
            break;
        }
        case JobKind::CpuLaw: {
            std::string path; if (!dll_path(d.dll, path)) return "pending";
            ok = tool_registry_global().find("nodus.fabric.dispatch.cpu") ||
                 nodus::gpu::register_cpu_fabric_dispatcher(tool_registry_global(), session_.library(), path, d.entry, d.batch, 1, d.program_id);
            break;
        }
        case JobKind::Fftfree: { std::string path; if (!dll_path(d.dll, path)) return "pending"; ok = session_.register_fftfree(path); break; }
        case JobKind::WaveT4: { std::string path; if (!dll_path(d.dll, path)) return "pending"; ok = session_.register_wave_t4(path); break; }
        case JobKind::FabricProgram: {
            nodus::gpu::FabricSpringProducts p; if (!dll_path(d.dll, p.llvm_dll)) return "pending";
            p.llvm_entry = d.entry; p.vulkan_spirv = spirv; p.local_size_x = d.local_x;
            ok = session_.register_fabric_spring(p, d.name, d.backend, d.card, d.base);
            break;
        }
        }
        if (!ok) return "registration refused";
        ready_.insert(d.id);
        return "";
    }
    bool ready(uint64_t definition) const { return ready_.count(definition) != 0; }

    // A lane is one market host's execution context: its own stack frame and row instances, on one device.
    // prepare() runs on the owning (main) thread: card, state writes, row instance (the tool registry has no lock);
    // execute() may run on the host's own thread: push the inputs, run the row, read the outputs back.
    struct Lane {
        uint32_t device = 0;
        RawStackFrame frame{};
        std::map<std::string, std::unique_ptr<ITool, std::function<void(ITool*)>>> rows;
        explicit Lane(uint32_t d = 0) : device(d) { raw_stack_init_frame(frame, 1 << 16); }
        ~Lane() { rows.clear(); raw_stack_free_mask(frame); if (frame.backend) gp_mem_backend_release(frame.backend); }
        Lane(const Lane&) = delete; Lane& operator=(const Lane&) = delete;
    };
    bool prepare(const WorkItem& w, Lane& lane) {
        auto def = definitions_.find(w.definition);
        if (def == definitions_.end() || !ready(w.definition) || !ensure_card(w.card)) return false;
        for (const auto& [address, t] : w.puts) if (!write_state(w.card, address, t)) return false;
        return row_for(def->second, lane) != nullptr;
    }
    WorkOutputs execute(const WorkItem& w, Lane& lane) {
        WorkOutputs out; out.id = w.id;
        const auto t0 = std::chrono::steady_clock::now();
        auto def = definitions_.find(w.definition);
        if (def == definitions_.end()) return out;
        auto it = lane.rows.find(row_id(def->second, lane.device));
        if (it == lane.rows.end()) return out;
        ITool* row = it->second.get();
        std::vector<uint64_t> handles;
        for (const auto& t : w.inputs) { handles.push_back(handle_of(t)); raw_stack_push_abstract_tensor(lane.frame, {handles.back()}); }
        ToolStackContext c{}; c.stack.raw = &lane.frame; row->execute_stack(c);
        nodus::tensors::AbstractTensorHandle r{};
        bool popped = false;
        if (c.succeeded && raw_stack_pop_abstract_tensor(lane.frame, r)) { out.row_output = tensor_of(r.id); nodus_tensor_destroy(r.id); popped = true; }
        while (raw_stack_pop_abstract_tensor(lane.frame, r)) {}      // a refused row restores its inputs; drop them
        for (auto h : handles) nodus_tensor_destroy(h);
        if (!popped) return out;
        // a library receipt (U64, status first) must say 0; a program row's output is its data
        const bool receipt = out.row_output.dtype == NODUS_DTYPE_U64 && out.row_output.bytes.size() >= 8;
        out.status = receipt ? get_le(out.row_output.bytes.data(), 8) : 0;
        for (auto a : w.outputs) {
            auto loan = session_.library().borrow_cpu(w.card, a, false);
            if (!loan) { out.status = 1; break; }
            const auto* p = static_cast<const uint8_t*>(loan.data());
            out.state.push_back({a, std::vector<uint8_t>(p, p + loan.bytes())});
        }
        out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        return out;
    }
    WorkOutputs run(const WorkItem& w) {                 // both, on the runner's own lane (the server's re-check)
        if (!prepare(w, main_lane_)) { WorkOutputs out; out.id = w.id; return out; }
        return execute(w, main_lane_);
    }
    std::string row_id(const Definition& d, uint32_t device) const {
        switch (d.kind) {
        case JobKind::ShaderJob: return d.name + ".vulkan." + std::to_string(device);
        case JobKind::Pipeline: return d.name + ".on." + std::to_string(device);
        case JobKind::FabricProgram: return d.name;
        case JobKind::CpuLaw: return "nodus.fabric.dispatch.cpu";
        case JobKind::Fftfree: return "nodus.fftfree";
        case JobKind::WaveT4: return "nodus.wave.t4.state";
        }
        return {};
    }

private:
    static std::string hex(const std::array<uint8_t, 32>& d) { static const char* x = "0123456789abcdef"; std::string s; for (auto c : d) { s += x[c >> 4]; s += x[c & 15]; } return s; }
    // The Vulkan build this host runs: its own glslang build of the GLSL for its target, else the server's build.
    std::string vulkan_for(const std::array<uint8_t, 32>& glsl_digest, std::vector<uint32_t>& spirv) {
        std::vector<uint8_t> bytes;
        auto offered = offered_vulkan_.find(glsl_digest);
        if (glslang_) {
            auto source = artifacts_.find(glsl_digest);
            if (source == artifacts_.end()) return "pending";
            bool failed = false;
            if (!builds_.build(source->second, target_, bytes, failed)) return failed ? "local glslang build failed" : "pending";
            if (offered != offered_vulkan_.end() && !compared_.count(glsl_digest)) {
                compared_.insert(glsl_digest); (offered->second == bytes ? local_matched_ : local_differed_)++;
            }
        } else {
            if (offered == offered_vulkan_.end()) return "pending";
            bytes = offered->second;
        }
        if (bytes.empty() || bytes.size() % 4) return "empty Vulkan build";
        spirv.resize(bytes.size() / 4); std::memcpy(spirv.data(), bytes.data(), bytes.size());
        return "";
    }
    bool library_request(uint64_t op, uint64_t card, uint64_t address, uint64_t payload) {
        std::array<uint64_t, 12> q{}; q[0] = op; q[1] = card; q[2] = 1907; q[3] = address;
        if (op == 1) { q[4] = 1024ull << 20; q[5] = q[6] = q[7] = 512ull << 20; }
        const uint64_t s12[1] = {12}, s1[1] = {1};
        const auto command = nodus_tensor_create(NODUS_DTYPE_U64, 1, s12);
        const bool own = !payload; if (own) payload = nodus_tensor_create(NODUS_DTYPE_U64, 1, s1);
        nodus_tensor_write(command, 0, q.data(), sizeof(q));
        raw_stack_push_abstract_tensor(frame_, {command}); raw_stack_push_abstract_tensor(frame_, {payload});
        ToolStackContext c{}; c.stack.raw = &frame_; session_.library_row().execute_stack(c);
        nodus::tensors::AbstractTensorHandle r{}; bool ok = false;
        if (c.succeeded && raw_stack_pop_abstract_tensor(frame_, r)) { uint64_t s = 1; nodus_tensor_read(r.id, 0, &s, 8); ok = s == 0; nodus_tensor_destroy(r.id); }
        while (raw_stack_pop_abstract_tensor(frame_, r)) {}
        nodus_tensor_destroy(command); if (own) nodus_tensor_destroy(payload);
        return ok;
    }
    bool ensure_card(uint64_t card) { if (cards_.count(card)) return true; if (!library_request(1, card, 0, 0)) return false; cards_.insert(card); return true; }
    bool write_state(uint64_t card, uint64_t address, const TensorBytes& t) {
        if (state_.count({card, address})) {
            auto loan = session_.library().borrow_cpu(card, address, true);
            if (!loan || loan.bytes() < t.bytes.size()) return false;
            std::memcpy(loan.data(), t.bytes.data(), t.bytes.size()); return true;
        }
        const auto h = handle_of(t); const bool ok = h && library_request(2, card, address, h); if (h) nodus_tensor_destroy(h);
        if (ok) state_.insert({card, address});
        return ok;
    }
    ITool* row_for(const Definition& d, Lane& lane) {
        const std::string id = row_id(d, lane.device);
        auto it = lane.rows.find(id);
        if (it == lane.rows.end()) {
            if (d.kind == JobKind::Pipeline && !ensure_pipeline_on(d, lane.device)) return nullptr;
            auto row = tool_registry_global().create(id);
            if (!row) return nullptr;
            row->initialize({});
            it = lane.rows.emplace(id, std::move(row)).first;
        }
        return it->second.get();
    }
    // A pipeline of device-bound components is registered once per device it runs on (<name>.on.<device>).
    bool ensure_pipeline_on(const Definition& d, uint32_t device) {
        const std::string id = row_id(d, device);
        if (tool_registry_global().find(id)) return true;
        std::vector<nodus::gpu::KernelPipelineComponent> parts;
        for (const auto& c : d.components) parts.push_back({c.device_bound ? c.tool + ".vulkan." + std::to_string(device) : c.tool,
                                                           nodus::gpu::TensorStateLibraryTool::ExecutionApi(c.api), c.program});
        return session_.register_pipeline(id, parts);
    }

    nodus::gpu::KernelLibrarySession session_{tool_registry_global()};
    fs::path work_;
    std::string target_;
    uint32_t device_;
    GlslangBuilds builds_;
    bool glslang_;
    RawStackFrame frame_{};
    std::map<std::array<uint8_t, 32>, std::vector<uint8_t>> artifacts_, offered_vulkan_;
    std::set<std::array<uint8_t, 32>> compared_;
    std::map<uint64_t, Definition> definitions_;
    std::set<uint64_t> ready_, cards_;
    std::set<std::pair<uint64_t, uint64_t>> state_;
    uint64_t local_matched_ = 0, local_differed_ = 0;
    Lane main_lane_{0};
};

}  // namespace nodus::link
