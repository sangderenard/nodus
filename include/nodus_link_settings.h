#pragma once
// Compiler settings and shader builds over the Nodus link (volunteer case a11b5c9b1be74c9b9fb684790f7fd5a3, stage 2).
// Owner, 2026-11-18: "Spir v isn't acceptable, the client should transmit their compiler settings needed and we will send
// glsl and Vulcan"; "they can check for a local toolchain".
//
//   1. On joining, a volunteer sends one SettingsRecord: per GPU the existing VulkanDeviceInfo inventory (with
//      shaderFloat64, work-group and subgroup limits), the CPU's target triple and feature bits, and which toolchains
//      (glslang, clang) it has.
//   2. The server builds a shader's GLSL with glslang for each device's target environment (from its API version), only
//      for devices that meet the shader's declared needs (float64, local size), caches the build by GLSL digest and
//      target, and sends the GLSL and that Vulkan build. No prebuilt or Turing-direct SPIR-V is ever sent.
//   3. A volunteer with glslang builds the GLSL itself for its own target and compares digests; it runs its own build.
// Large records travel as typed bulk streams: the Begin descriptor names the kind and id, so the receiver never pairs a
// stream with a control packet by arrival order. Nothing here blocks: glslang runs as a nodus::exec::Child.
#include "nodus_link.h"
#include "kernel_vulkan_runner.h"
#include "exec_pipe.h"

#include <intrin.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <set>

namespace nodus::link {

// ---- typed bulk streams ------------------------------------------------------------------------------------------
// Artifact/Definition/WorkItem/Outputs: the job layer (nodus_link_jobs.h, stage 3).
enum class BulkStreamKind : uint32_t { Settings = 1, Glsl = 2, Vulkan = 3, Tensor = 4, Artifact = 5, Definition = 6, WorkItem = 7, Outputs = 8 };
constexpr uint32_t kTypedMagic = 0x31595442;   // "BTY1"
// Descriptor: magic u32 | kind u32 | id u64 | aux [up to 4080 bytes, kind-defined]
inline std::vector<uint8_t> typed_descriptor(BulkStreamKind kind, uint64_t id, const std::vector<uint8_t>& aux = {}) {
    std::vector<uint8_t> d(16 + aux.size());
    put_le(d.data(), kTypedMagic, 4); put_le(d.data() + 4, uint32_t(kind), 4); put_le(d.data() + 8, id, 8);
    std::copy(aux.begin(), aux.end(), d.begin() + 16); return d;
}
struct TypedStream { BulkStreamKind kind{}; uint64_t id = 0; std::vector<uint8_t> aux, bytes; };

// Sends whole typed streams over a Link's bulk pair as the window allows; receives and reassembles them.
class BulkMux {
public:
    static constexpr size_t kMaxStreamBytes = size_t(256) << 20;
    explicit BulkMux(Link& link) : link_(link) {}
    void send(BulkStreamKind kind, uint64_t id, std::vector<uint8_t> bytes, std::vector<uint8_t> aux = {}) {
        out_.push_back({typed_descriptor(kind, id, aux), std::move(bytes), 0, 0});
    }
    void pump() {
        while (!out_.empty()) {                                   // send: begin, write what the window takes, end
            auto& o = out_.front();
            if (!o.stream && !(o.stream = link_.begin_bulk(o.descriptor.data(), o.descriptor.size(), o.bytes.size()))) break;
            if (o.written < o.bytes.size()) o.written += link_.write_bulk(o.stream, o.bytes.data() + o.written, o.bytes.size() - o.written);
            if (o.written < o.bytes.size() || !link_.end_bulk(o.stream)) break;
            out_.pop_front();
        }
        BulkEvent e;
        while (link_.receive_bulk(e)) {                           // receive: Begin names the kind and id
            if (e.kind == BulkKind::Begin) {
                TypedStream s;
                if (e.data.size() < 16 || get_le(e.data.data(), 4) != kTypedMagic || e.total > kMaxStreamBytes) { ++rejected_; partial_[e.stream] = {}; partial_[e.stream].kind = BulkStreamKind(0); continue; }
                s.kind = BulkStreamKind(get_le(e.data.data() + 4, 4)); s.id = get_le(e.data.data() + 8, 8);
                s.aux.assign(e.data.begin() + 16, e.data.end()); s.bytes.reserve(size_t(e.total));
                partial_[e.stream] = std::move(s);
            } else if (e.kind == BulkKind::Chunk) {
                auto it = partial_.find(e.stream);
                if (it != partial_.end() && uint32_t(it->second.kind)) it->second.bytes.insert(it->second.bytes.end(), e.data.begin(), e.data.end());
            } else {
                auto it = partial_.find(e.stream);
                if (it == partial_.end()) continue;
                if (e.kind == BulkKind::End && uint32_t(it->second.kind)) done_.push_back(std::move(it->second));
                partial_.erase(it);
            }
        }
    }
    bool take(BulkStreamKind kind, TypedStream& s) {
        pump();
        for (auto it = done_.begin(); it != done_.end(); ++it) if (it->kind == kind) { s = std::move(*it); done_.erase(it); return true; }
        return false;
    }
    bool idle() const { return out_.empty(); }
    uint64_t rejected() const { return rejected_; }

private:
    struct Out { std::vector<uint8_t> descriptor, bytes; uint64_t stream; size_t written; };
    Link& link_;
    std::deque<Out> out_;
    std::map<uint64_t, TypedStream> partial_;
    std::deque<TypedStream> done_;
    uint64_t rejected_ = 0;
};

// ---- the settings record -----------------------------------------------------------------------------------------
enum CpuFeature : uint64_t { kSse42 = 1, kAvx = 2, kAvx2 = 4, kFma = 8, kAvx512f = 16, kF16c = 32, kBmi2 = 64, kPopcnt = 128 };
struct GpuSettings {
    std::string name; uint32_t vendor = 0, device = 0, api_version = 0, driver_version = 0;
    std::array<uint8_t, 16> uuid{}; bool float64 = false;
    uint32_t max_invocations = 0; std::array<uint32_t, 3> max_size{}; uint32_t subgroup = 0;
    uint32_t max_storage_buffers = 0, max_storage_bytes = 0;
};
struct SettingsRecord {
    std::vector<GpuSettings> gpus;
    std::string cpu_triple, cpu_brand; uint64_t cpu_features = 0; uint32_t cpu_threads = 0;
    bool has_glslang = false, has_clang = false;
    std::string glslang_path, clang_path;          // where they were found (stay on the volunteer; not sent)
};

namespace detail {
inline void put_str(std::vector<uint8_t>& b, const std::string& s, size_t n) {
    const size_t at = b.size(); b.resize(at + n, 0); std::memcpy(b.data() + at, s.data(), std::min(n - 1, s.size()));
}
inline void put_u(std::vector<uint8_t>& b, uint64_t v, size_t n) { const size_t at = b.size(); b.resize(at + n); put_le(b.data() + at, v, n); }
inline std::string get_str(const uint8_t* p, size_t n) { size_t l = 0; while (l < n && p[l]) ++l; return std::string(reinterpret_cast<const char*>(p), l); }
inline std::string find_on_path(const char* exe) {
    char out[MAX_PATH] = {}; char* file = nullptr;
    return SearchPathA(nullptr, exe, nullptr, MAX_PATH, out, &file) ? std::string(out) : std::string();
}
}  // namespace detail

// Fixed layout (152-byte header): magic "NLS1" u32 | version u16 | gpu count u16 | cpu triple [64] | cpu brand [64] |
// cpu features u64 | cpu threads u32 | toolchains u32 (1 glslang, 2 clang), then 128 bytes per GPU.
constexpr uint32_t kSettingsMagic = 0x31534C4E;
constexpr size_t kGpuSettingsBytes = 64 + 4 * 4 + 16 + 4 + 4 + 12 + 4 + 4 + 4;
inline std::vector<uint8_t> encode_settings(const SettingsRecord& r) {
    std::vector<uint8_t> b;
    detail::put_u(b, kSettingsMagic, 4); detail::put_u(b, 1, 2); detail::put_u(b, r.gpus.size(), 2);
    detail::put_str(b, r.cpu_triple, 64); detail::put_str(b, r.cpu_brand, 64);
    detail::put_u(b, r.cpu_features, 8); detail::put_u(b, r.cpu_threads, 4);
    detail::put_u(b, (r.has_glslang ? 1u : 0u) | (r.has_clang ? 2u : 0u), 4);
    for (const auto& g : r.gpus) {
        detail::put_str(b, g.name, 64);
        detail::put_u(b, g.vendor, 4); detail::put_u(b, g.device, 4); detail::put_u(b, g.api_version, 4); detail::put_u(b, g.driver_version, 4);
        b.insert(b.end(), g.uuid.begin(), g.uuid.end());
        detail::put_u(b, g.float64 ? 1 : 0, 4); detail::put_u(b, g.max_invocations, 4);
        for (auto v : g.max_size) detail::put_u(b, v, 4);
        detail::put_u(b, g.subgroup, 4); detail::put_u(b, g.max_storage_buffers, 4); detail::put_u(b, g.max_storage_bytes, 4);
    }
    return b;
}
inline bool decode_settings(const std::vector<uint8_t>& b, SettingsRecord& r) {
    if (b.size() < 152 || get_le(b.data(), 4) != kSettingsMagic || get_le(b.data() + 4, 2) != 1) return false;
    const size_t n = size_t(get_le(b.data() + 6, 2));
    if (n > 64 || b.size() != 152 + n * kGpuSettingsBytes) return false;
    r = {};
    r.cpu_triple = detail::get_str(b.data() + 8, 64); r.cpu_brand = detail::get_str(b.data() + 72, 64);
    r.cpu_features = get_le(b.data() + 136, 8); r.cpu_threads = uint32_t(get_le(b.data() + 144, 4));
    const uint32_t tools = uint32_t(get_le(b.data() + 148, 4)); r.has_glslang = tools & 1; r.has_clang = tools & 2;
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* p = b.data() + 152 + i * kGpuSettingsBytes;
        GpuSettings g; g.name = detail::get_str(p, 64); p += 64;
        g.vendor = uint32_t(get_le(p, 4)); g.device = uint32_t(get_le(p + 4, 4)); g.api_version = uint32_t(get_le(p + 8, 4)); g.driver_version = uint32_t(get_le(p + 12, 4)); p += 16;
        std::memcpy(g.uuid.data(), p, 16); p += 16;
        g.float64 = get_le(p, 4) != 0; g.max_invocations = uint32_t(get_le(p + 4, 4)); p += 8;
        for (auto& v : g.max_size) { v = uint32_t(get_le(p, 4)); p += 4; }
        g.subgroup = uint32_t(get_le(p, 4)); g.max_storage_buffers = uint32_t(get_le(p + 4, 4)); g.max_storage_bytes = uint32_t(get_le(p + 8, 4));
        r.gpus.push_back(std::move(g));
    }
    return true;
}

// This machine's settings: the Vulkan inventory, the CPU from cpuid, the toolchains on PATH.
inline SettingsRecord collect_settings() {
    SettingsRecord r;
    std::vector<nodus::kernels::VulkanDeviceInfo> devices; std::string error;
    if (nodus::kernels::enumerate_vulkan_devices(&devices, &error)) for (const auto& d : devices) {
        if (!d.compute) continue;
        GpuSettings g; g.name = d.name; g.vendor = d.vendor_id; g.device = d.device_id; g.api_version = d.api_version;
        g.driver_version = d.driver_version; g.uuid = d.uuid; g.float64 = d.shader_float64; g.max_invocations = d.max_compute_invocations;
        g.max_size = d.max_compute_size; g.subgroup = d.subgroup_size; g.max_storage_buffers = d.max_storage_buffers;
        g.max_storage_bytes = d.max_storage_buffer_bytes; r.gpus.push_back(std::move(g));
    }
    SYSTEM_INFO si{}; GetNativeSystemInfo(&si);
    r.cpu_threads = si.dwNumberOfProcessors;
    r.cpu_triple = si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64 ? "x86_64-pc-windows-msvc"
                 : si.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_ARM64 ? "aarch64-pc-windows-msvc" : "unknown-pc-windows-msvc";
#if defined(_M_X64)
    int regs[4] = {}; char brand[49] = {};
    __cpuid(regs, 0x80000000);
    if (unsigned(regs[0]) >= 0x80000004u) for (int i = 0; i < 3; ++i) { __cpuid(regs, 0x80000002 + i); std::memcpy(brand + 16 * i, regs, 16); }
    r.cpu_brand = brand;
    __cpuid(regs, 1);
    if (regs[2] & (1 << 20)) r.cpu_features |= kSse42;
    if (regs[2] & (1 << 28)) r.cpu_features |= kAvx;
    if (regs[2] & (1 << 12)) r.cpu_features |= kFma;
    if (regs[2] & (1 << 29)) r.cpu_features |= kF16c;
    if (regs[2] & (1 << 23)) r.cpu_features |= kPopcnt;
    __cpuidex(regs, 7, 0);
    if (regs[1] & (1 << 5)) r.cpu_features |= kAvx2;
    if (regs[1] & (1 << 8)) r.cpu_features |= kBmi2;
    if (regs[1] & (1 << 16)) r.cpu_features |= kAvx512f;
#endif
    r.glslang_path = detail::find_on_path("glslangValidator.exe"); r.has_glslang = !r.glslang_path.empty();
    r.clang_path = detail::find_on_path("clang.exe"); r.has_clang = !r.clang_path.empty();
    return r;
}

// glslang's target environment for a device's Vulkan API version.
inline std::string vulkan_target(uint32_t api_version) {
    const uint32_t major = api_version >> 22, minor = (api_version >> 12) & 0x3FF;
    if (major != 1) return {};
    return "vulkan1." + std::to_string(std::min(minor, 3u));
}

// What a shader product declares it needs (from its own record, e.g. Turing's spring_network_step.glsl.json:
// numeric_dtype float64, local_size_x).
struct ShaderNeeds { bool float64 = false; uint32_t local_size_x = 0; };
inline const char* unmet_need(const GpuSettings& g, const ShaderNeeds& n) {
    if (vulkan_target(g.api_version).empty()) return "no Vulkan 1.x target";
    if (n.float64 && !g.float64) return "no shaderFloat64";
    if (n.local_size_x > g.max_invocations || n.local_size_x > g.max_size[0]) return "local size beyond its work-group limits";
    return nullptr;
}

// ---- glslang builds, without blocking ----------------------------------------------------------------------------
// Builds GLSL for a target with glslangValidator as a child process; finished builds are cached by (GLSL digest, target).
class GlslangBuilds {
public:
    GlslangBuilds(std::string glslang, std::filesystem::path work) : glslang_(std::move(glslang)), work_(std::move(work)) {
        std::error_code ec; std::filesystem::create_directories(work_, ec);
    }
    using Key = std::pair<std::array<uint8_t, 32>, std::string>;
    // Starts (or finds) the build; returns true and fills `spirv` once it is done, false while it runs. `failed` on error.
    bool build(const std::vector<uint8_t>& glsl, const std::string& target, std::vector<uint8_t>& spirv, bool& failed) {
        failed = false;
        Key key{sha256(glsl.data(), glsl.size()), target};
        if (auto it = cache_.find(key); it != cache_.end()) { spirv = it->second; return true; }
        if (refused_.count(key)) { failed = true; return false; }
        auto& job = jobs_[key];
        if (!job) {
            job = std::make_unique<Job>();
            job->stem = "b" + std::to_string(++serial_);
            std::ofstream(work_ / (job->stem + ".comp"), std::ios::binary).write(reinterpret_cast<const char*>(glsl.data()), std::streamsize(glsl.size()));
            nodus::exec::Program p; p.path = glslang_; p.working_dir = work_.string();
            p.arguments = {"-V", "--target-env", target, "-S", "comp", job->stem + ".comp", "-o", job->stem + ".spv"};
            std::string why;
            if (!job->child.start(p, why)) { refused_.insert(key); jobs_.erase(key); failed = true; return false; }
            ++started_;
            return false;
        }
        std::string sink; job->child.take(sink);
        if (!job->child.finished()) return false;
        const bool ok = job->child.exit_code() == 0;
        std::ifstream f(work_ / (job->stem + ".spv"), std::ios::binary);
        std::vector<uint8_t> out{std::istreambuf_iterator<char>(f), {}};
        jobs_.erase(key);
        if (!ok || out.empty() || out.size() % 4) { refused_.insert(key); failed = true; return false; }
        cache_[key] = out; spirv = std::move(out); return true;
    }
    uint64_t started() const { return started_; }
    size_t cached() const { return cache_.size(); }

private:
    struct Job { nodus::exec::Child child; std::string stem; };
    std::string glslang_;
    std::filesystem::path work_;
    std::map<Key, std::vector<uint8_t>> cache_;
    std::map<Key, std::unique_ptr<Job>> jobs_;
    std::set<Key> refused_;
    uint64_t serial_ = 0, started_ = 0;
};

}  // namespace nodus::link
