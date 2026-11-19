#include "nodus_headless_abi.h"
#include "tool_registry.h"
#if defined(_WIN32)
// The Output row is the plugin source module_library/source/tools/audio_pa_sdl_output.cpp, compiled into this image the
// way a composed plugin source is: NODUS_PLUGIN_COMPOSITE with its exports renamed.
#define NODUS_PLUGIN_COMPOSITE 1
#define NODUS_PLUGIN_FACTORY_NAME nodus_audio_pa_sdl_output_create
#define NODUS_PLUGIN_DESTROY_NAME nodus_audio_pa_sdl_output_destroy
#define NODUS_PLUGIN_SOURCE_NAME nodus_audio_pa_sdl_output_source
#include "../../module_library/source/tools/audio_pa_sdl_output.cpp"
#include <io.h>
#include <fcntl.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <thread>

extern "C" int32_t nodus_headless_pcm_playback_stdin(uint32_t device_id) {
    // This process entry owns its complete graph. No shared service canvas.
    auto state = std::make_shared<State>(); state->device = device_id;
    state->graph = nodus_headless_create(); if (!state->graph) return 1;
    ToolRegistry::Entry entry;
    entry.id = kTool; entry.name = "Owned PCM SDL output"; entry.caps = ToolCaps::Table;
    entry.source_path = nodus_audio_pa_sdl_output_source();
    entry.factory = [state] { return ToolRegistry::Factory::result_type(new Output(state), [](ITool* p){nodus_audio_pa_sdl_output_destroy(p);}); };
    int result = 1;
    if (!tool_registry_global().register_tool(std::move(entry))) { nodus_headless_destroy(state->graph); return 1; }
    GP_CanvasModuleDesc desc{}; desc.w=240; desc.h=120; std::strcpy(desc.label,"owned_pcm_output");
    const int module = nodus_headless_add_module(state->graph, &desc);
    GP_CanvasModuleRowDesc row{1,0,1,1,kTool};
    NodusHeadlessFifoConfig fifo{}; fifo.token_bytes=sizeof(Block); fifo.slots=16;
    fifo.top_k=0; fifo.type_id=-1; fifo.layout=2; fifo.dtype=0; fifo.backend_type=GP_MEM_BACKEND_CPU;
    if (module >= 0 && nodus_headless_set_module_rows(state->graph,module,&row,1) &&
        nodus_headless_declare_output_port_ex(state->graph,kPort,module,0,&fifo,1) &&
        nodus_headless_set_module_stack_contract(state->graph,module,2) &&
        nodus_headless_set_module_exec_mode(state->graph,module,3) && nodus_headless_step(state->graph,1./60)) {
        const auto ready_deadline = std::chrono::steady_clock::now()+std::chrono::seconds(3);
        while (!state->ready.load() && !state->failed.load() && std::chrono::steady_clock::now()<ready_deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (state->ready.load()) {
            _setmode(_fileno(stdin), _O_BINARY);
            result = 0;
            while (!state->failed.load()) {
                Block block{};
                const size_t bytes = std::fread(block.samples.data(),1,sizeof(block.samples),stdin);
                if (bytes % kFrameBytes || std::ferror(stdin)) { result=2; break; }
                if (!bytes) break;
                block.frames=static_cast<uint32_t>(bytes/kFrameBytes);
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
                while (!state->failed.load()) {
                    NodusHeadlessPublishResult publish{};
                    if (nodus_headless_publish_output_token(state->graph,kPort,&block,sizeof(block),&publish) && publish.accepted) {
                        if (publish.dropped) { result=3; state->failed.store(true); }
                        break;
                    }
                    state->refused.fetch_add(1);
                    if (std::chrono::steady_clock::now()>deadline) { result=3; state->failed.store(true); break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            state->finishing.store(true);
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (!state->drained.load() && !state->failed.load() && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            if (!state->drained.load() || state->failed.load()) result=1;
        }
    }
    nodus_headless_destroy(state->graph);
    tool_registry_global().unregister_tool(kTool);
    std::printf("pcm_sink_done frames=%llu max_queue_frames=%d rate=%.8f backpressure=%llu empty_stream_events=%llu drained=%d result=%d\n",
        static_cast<unsigned long long>(state->frames.load()),state->max_queued.load(),state->ratio.load(),
        static_cast<unsigned long long>(state->refused.load()),static_cast<unsigned long long>(state->empty_events.load()),
        state->drained.load()?1:0,result);
    std::fflush(stdout);
    return result;
}
#else
extern "C" int32_t nodus_headless_pcm_playback_stdin(uint32_t) { return 1; }
#endif
