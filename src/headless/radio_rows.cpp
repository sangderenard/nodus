// The Radio rows' headless ABI (nodus_headless_radio_*) and their registration. The five rows themselves are plugin
// sources in module_library/source/tools/ (radio_source, radio_pcm_s16le, radio_fft_bands, radio_udp, radio_pcm_udp),
// compiled into this image the way composed plugin sources are: NODUS_PLUGIN_COMPOSITE with exports renamed per row.
// Shared state (radio_shared.h) stays here: the registration captures it in each row's factory.
#define NODUS_PLUGIN_COMPOSITE 1
#include "nodus_headless_abi.h"
#include "tool_api.h"
#include "tool_registry.h"
#include "fftfree_tool.h"
#include "common/tensors/abstraction/tensor_abi.h"
#include "common/tensors/abstraction/abstract_tensor_handle.h"

#if defined(_WIN32)
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_radio_source_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_radio_source_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_radio_source_source
#include "../../module_library/source/tools/radio_source.cpp"
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_radio_pcm_s16le_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_radio_pcm_s16le_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_radio_pcm_s16le_source
#include "../../module_library/source/tools/radio_pcm_s16le.cpp"
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_radio_fft_bands_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_radio_fft_bands_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_radio_fft_bands_source
#include "../../module_library/source/tools/radio_fft_bands.cpp"
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_radio_udp_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_radio_udp_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_radio_udp_source
#include "../../module_library/source/tools/radio_udp.cpp"
#undef NODUS_PLUGIN_FACTORY_NAME
#undef NODUS_PLUGIN_DESTROY_NAME
#undef NODUS_PLUGIN_SOURCE_NAME
#define NODUS_PLUGIN_FACTORY_NAME nodus_radio_pcm_udp_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_radio_pcm_udp_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_radio_pcm_udp_source
#include "../../module_library/source/tools/radio_pcm_udp.cpp"

using namespace nodus::radio;

namespace {
std::mutex config_mutex;
std::shared_ptr<Shared> configured;

ToolRegistry::Entry entry(const char* id, const char* name, const char* source_path,
                          ToolRegistry::Factory factory) {
    ToolRegistry::Entry result;
    result.id = id;
    result.name = name;
    result.caps = ToolCaps::Table;
    result.source_path = source_path;
    result.factory = std::move(factory);
    return result;
}
} // namespace

extern "C" int32_t nodus_headless_radio_configure(
    const wchar_t* scope_dll, const wchar_t* fftfree_dll,
    const char* destination_host, uint16_t destination_port,
    uint32_t fft_size, uint32_t overlap, uint32_t band_count,
    float min_hz, float max_hz) {
    if (!scope_dll || !*scope_dll || !fftfree_dll || !*fftfree_dll ||
        !destination_host || !*destination_host || !destination_port ||
        fft_size < 256 || (fft_size & (fft_size - 1)) ||
        !overlap || (overlap & (overlap - 1)) || overlap > fft_size ||
        !band_count || band_count > kMaxBands ||
        !std::isfinite(min_hz) || !std::isfinite(max_hz) ||
        min_hz <= 0 || max_hz <= min_hz) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (configured) return 0;
    auto shared = std::make_shared<Shared>();
    shared->config = {scope_dll, fftfree_dll, destination_host, destination_port,
                      fft_size, overlap, band_count, min_hz, max_hz};
    shared->fft_states = std::make_unique<nodus::gpu::TensorStateLibraryTool>(
        16ull << 20, std::vector<uint64_t>{}, false); // Radio FFT is CPU-only.
    shared->fft_states->initialize({});
    if (!radio_register_fft_states(*shared)) return 0;
    auto& registry = tool_registry_global();
    if (!nodus::gpu::register_fftfree_tool(registry, *shared->fft_states,
                                           std::filesystem::path(fftfree_dll).string())) return 0;
    if (!registry.register_tool(entry("radio.system_output", "Radio PCM source", nodus_radio_source_source(),
        [shared] { return ToolRegistry::Factory::result_type(new RadioSource(shared), [](ITool* p) { nodus_radio_source_destroy(p); }); }))) {
        registry.unregister_tool("nodus.fftfree"); return 0;
    }
    if (!registry.register_tool(entry("radio.fft_bands", "Radio fftfree band projection", nodus_radio_fft_bands_source(),
        [shared] { return ToolRegistry::Factory::result_type(new RadioFftBands(shared), [](ITool* p) { nodus_radio_fft_bands_destroy(p); }); }))) {
        registry.unregister_tool("radio.system_output");
        registry.unregister_tool("nodus.fftfree"); return 0;
    }
    if (!registry.register_tool(entry("radio.udp_broadcast", "Radio UDP broadcast", nodus_radio_udp_source(),
        [shared] { return ToolRegistry::Factory::result_type(new RadioUdp(shared), [](ITool* p) { nodus_radio_udp_destroy(p); }); }))) {
        registry.unregister_tool("radio.system_output");
        registry.unregister_tool("radio.fft_bands");
        registry.unregister_tool("nodus.fftfree");
        return 0;
    }
    configured = std::move(shared);
    return 1;
}

extern "C" int32_t nodus_headless_radio_file_source(const wchar_t* wav_path, int32_t loop) {
    if (!wav_path || !*wav_path) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->running.load() || configured->captured.load()) return 0;
    configured->config.wav = wav_path;
    configured->config.loop = loop != 0;
    auto shared = configured;
    return tool_registry_global().register_tool(entry("radio.file_output", "Radio WAV playback + fftfree", nodus_radio_source_source(),
        [shared] { return ToolRegistry::Factory::result_type(new RadioSource(shared), [](ITool* p) { nodus_radio_source_destroy(p); }); })) ? 1 : 0;
}

extern "C" int32_t nodus_headless_radio_analysis_file(const wchar_t* path) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !path || !*path || !std::filesystem::path(path).is_absolute() ||
        configured->config.wav.empty() || configured->config.loop || configured->config.pcm_port ||
        configured->running.load() || configured->captured.load()) return 0;
    configured->config.analysis_only = true;
    configured->config.analysis_file = path;
    return 1;
}

extern "C" int32_t nodus_headless_radio_band_scale(int32_t linear) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->running.load() || configured->captured.load() ||
        (linear != 0 && linear != 1)) return 0;
    configured->config.linear_bands = linear != 0;
    return 1;
}

extern "C" int32_t nodus_headless_radio_hold(int32_t held) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || configured->config.wav.empty()) return 0;
    configured->held.store(held != 0);
    return 1;
}

extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t port) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !port || configured->config.wav.empty() || configured->running.load() ||
        configured->config.fft_size / configured->config.overlap != 256) return 0;
    auto shared = configured;
    if (!tool_registry_global().register_tool(entry("radio.pcm_rtp", "Radio PCM RTP", nodus_radio_pcm_udp_source(),
        [shared] { return ToolRegistry::Factory::result_type(new RadioPcmUdp(shared), [](ITool* p) { nodus_radio_pcm_udp_destroy(p); }); }))) return 0;
    if (!tool_registry_global().register_tool(entry("radio.pcm_s16le", "Radio PCM s16le", nodus_radio_pcm_s16le_source(),
        [] { return ToolRegistry::Factory::result_type(new RadioPcmS16le(), [](ITool* p) { nodus_radio_pcm_s16le_destroy(p); }); }))) {
        tool_registry_global().unregister_tool("radio.pcm_rtp");
        return 0;
    }
    configured->config.pcm_port = port;
    return 1;
}
extern "C" int32_t nodus_headless_radio_pcm_bind_at(NodusHeadlessGraph* graph, int32_t module, int32_t row) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !configured->config.pcm_port || configured->captured.load() || row < 0) return 0;
    NodusHeadlessFifoConfig fifo{};
    fifo.token_bytes = sizeof(PcmBlock); fifo.slots = 16; fifo.top_k = 8;
    fifo.type_id = -1; fifo.layout = 2; fifo.dtype = 0; fifo.backend_type = GP_MEM_BACKEND_CPU;
    if (!nodus_headless_declare_output_port_ex(graph, kPcmPort, module, row, &fifo, 1)) return 0;
    configured->pcm_graph = graph;
    return 1;
}
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph* graph, int32_t module) {
    return nodus_headless_radio_pcm_bind_at(graph, module, 2);
}

extern "C" int32_t nodus_headless_radio_pcm_status(uint64_t* emitted, uint64_t* dropped) {
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured || !emitted || !dropped) return 0;
    *emitted = configured->pcm_emitted.load(); *dropped = configured->pcm_dropped.load();
    return 1;
}

extern "C" int32_t nodus_headless_radio_status(NodusRadioStatus* out_status) {
    if (!out_status) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured) return 0;
    out_status->captured_frames = configured->captured.load(std::memory_order_relaxed);
    out_status->emitted_packets = configured->emitted.load(std::memory_order_relaxed);
    out_status->dropped_packets = configured->dropped.load(std::memory_order_relaxed);
    out_status->sample_rate = configured->sample_rate.load(std::memory_order_relaxed);
    out_status->capture_running = configured->running.load(std::memory_order_relaxed);
    return 1;
}
extern "C" int32_t nodus_headless_radio_capture_overruns(uint64_t* lost_frames) {
    if (!lost_frames) return 0;
    std::lock_guard<std::mutex> lock(config_mutex);
    if (!configured) return 0;
    *lost_frames = configured->capture_overruns.load(std::memory_order_relaxed);
    return 1;
}
#else
extern "C" int32_t nodus_headless_radio_hold(int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_configure(
    const wchar_t*, const wchar_t*, const char*, uint16_t,
    uint32_t, uint32_t, uint32_t, float, float) { return 0; }
extern "C" int32_t nodus_headless_radio_file_source(const wchar_t*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_analysis_file(const wchar_t*) { return 0; }
extern "C" int32_t nodus_headless_radio_band_scale(int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_configure(uint16_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_bind(NodusHeadlessGraph*, int32_t) { return 0; }
extern "C" int32_t nodus_headless_radio_pcm_status(uint64_t*, uint64_t*) { return 0; }
extern "C" int32_t nodus_headless_radio_status(NodusRadioStatus*) { return 0; }
extern "C" int32_t nodus_headless_radio_capture_overruns(uint64_t*) { return 0; }
#endif
