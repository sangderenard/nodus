#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <cstring>

#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_init.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_render.h>

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/kpath/kpath_raster_utils.h"

namespace nodus::tensors::kpath {
namespace {

constexpr uint32_t kWinW = 800;
constexpr uint32_t kWinH = 600;

struct FlipBuffer {
    AbstractTensorPool::PooledTensor tensor;
};

static ArmatureProgram make_demo_program() {
    ArmatureProgram program;
    const int turns = 5;
    const int steps = 1200;
    const float radius = 220.0f;
    const float z = 0.0f;
    program.points.reserve(static_cast<size_t>(steps) + 1);
    for (int i = 0; i <= steps; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(steps);
        const float ang = t * static_cast<float>(turns) * 2.0f * 3.14159265358979323846f;
        const float r = radius * (0.15f + 0.85f * t);
        ToolPoint p;
        p.x = r * std::cos(ang);
        p.y = r * std::sin(ang);
        p.z = z;
        p.engaged = true;
        program.points.push_back(p);
    }
    return program;
}

} // namespace
} // namespace nodus::tensors::kpath

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    using namespace nodus::tensors;
    using namespace nodus::tensors::kpath;

    register_in_memory_backend(true);

    // Required when SDL_MAIN_HANDLED is defined.
    SDL_SetMainReady();
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        const char* err = SDL_GetError();
        std::fprintf(stderr, "SDL_Init failed: %s\n", (err && err[0]) ? err : "<no error message>");
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("kpath raster tensor demo",
                                          static_cast<int>(kWinW),
                                          static_cast<int>(kWinH),
                                          0);
    if (!window) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (!renderer) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer,
                                             SDL_PIXELFORMAT_RGBA32,
                                             SDL_TEXTUREACCESS_STREAMING,
                                             static_cast<int>(kWinW),
                                             static_cast<int>(kWinH));
    if (!texture) {
        std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    TensorBackend* backend = in_memory_backend_singleton().name() ? static_cast<TensorBackend*>(&in_memory_backend_singleton()) : nullptr;
    if (!backend) {
        std::fprintf(stderr, "In-memory backend unavailable.\n");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    AbstractTensorPool tensor_pool;
    ArmatureProgram program = make_demo_program();
    MachineControlConfig machine;
    machine.step_px = 0.75f;
    machine.energy_per_px = 1.0f;
    machine.enable_thermal_guard = false;
    const ThermalSimConfig heat_cfg = thermal_sim_from_preset(ThermalMaterialPreset::SteelThin);
    machine.energy_to_temp = heat_cfg.heat_gain;

    GaussianToolParams tool;
    tool.sigma_px = 1.1f;

    TensorCanvas2D energy;
    TensorCanvas2D heat_energy;
    TensorCanvas2D temp;
    energy.resize(kWinW, kWinH, 0.0f);
    heat_energy.resize(kWinW, kWinH, 0.0f);
    temp.resize(kWinW, kWinH, 0.0f);
    FilmTensor2D film(kWinW, kWinH, 3);
    film.basis.name = "rgb";
    film.basis.bin_centers = {0.0f, 1.0f, 2.0f};
    film.layer_reactance = {1.0f, 1.0f, 1.0f};

    ProgramRasterTransform xform = plan_program_raster_transform_refined(program,
                                                                         machine,
                                                                         1.0f,
                                                                         8.0f,
                                                                         tool);
    if (xform.width_px != kWinW || xform.height_px != kWinH) {
        xform.width_px = kWinW;
        xform.height_px = kWinH;
        xform.scale = 1.0f;
        xform.shift_x = static_cast<float>(kWinW) * 0.5f;
        xform.shift_y = static_cast<float>(kWinH) * 0.5f;
    }

    ProgramScatterPlan scatter_plan;
    if (!plan_program_scatter(program, machine, xform, &scatter_plan, true, true, backend)) {
        std::fprintf(stderr, "Failed to plan scatter.\n");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    AbstractTensor dense_points = coo_points_to_dense_f32(scatter_plan.points, backend);
    if (!dense_points.valid()) {
        std::fprintf(stderr, "Failed to materialize scatter points.\n");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }
    const uint32_t head_count = 4;
    const std::array<float, 4> kernel_weights = {1.0f, 0.85f, 0.7f, 0.55f};
    auto kernel_bank = make_kernel_bank(tensor_pool, backend, kernel_weights);
    auto kernel_ids = make_kernel_ids_for_heads(tensor_pool, dense_points, kWinW, kWinH, head_count, backend);
    auto temp_kernel_ids = make_kernel_ids_for_heads(tensor_pool, dense_points, kWinW, kWinH, head_count, backend);
    auto diffusion_kernel = make_laplacian_kernel(tensor_pool, backend);
    AbstractTensor temp_state;

    bool use_scatter = false;

    FlipBuffer flip[2];
    flip[0].tensor = make_flip_tensor(tensor_pool, backend, kWinW, kWinH);
    flip[1].tensor = make_flip_tensor(tensor_pool, backend, kWinW, kWinH);
    if (!flip[0].tensor.valid() || !flip[1].tensor.valid()) {
        std::fprintf(stderr, "Failed to allocate flip buffers.\n");
        SDL_DestroyTexture(texture);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    bool running = true;
    uint64_t frame = 0;
    auto last_title = std::chrono::high_resolution_clock::now();
    double avg_raster_ms = 0.0;
    uint32_t avg_count = 0;

    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) running = false;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) running = false;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_SPACE) use_scatter = !use_scatter;
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        BeamToolParams gaussian_tool{};
        gaussian_tool.falloff = BeamFalloffKind::Gaussian;
        gaussian_tool.sigma_px = tool.sigma_px;
        BeamToolParams airy_tool{};
        airy_tool.falloff = BeamFalloffKind::Airy;
        if (!use_scatter) {
            rasterize_program_with_kernel_transformed(program,
                                                      energy,
                                                      temp,
                                                      machine,
                                                      gaussian_tool,
                                                      xform);
            rasterize_program_scatter_with_spatial_kernel(program,
                                                          heat_energy,
                                                          temp,
                                                          machine,
                                                          xform,
                                                          airy_tool,
                                                          &diffusion_kernel.tensor(),
                                                          heat_cfg,
                                                          &temp_state,
                                                          backend);
        } else {
            rasterize_program_scatter_with_spatial_kernel(program,
                                                          energy,
                                                          temp,
                                                          machine,
                                                          xform,
                                                          airy_tool,
                                                          &diffusion_kernel.tensor(),
                                                          heat_cfg,
                                                          &temp_state,
                                                          backend);
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        avg_raster_ms = (avg_raster_ms * avg_count + ms) / static_cast<double>(avg_count + 1);
        avg_count = std::min<uint32_t>(avg_count + 1, 120u);

        FlipBuffer& buf = flip[frame & 1u];
        const float film_decay = 0.94f;
        const BeamHistogram beam_hist{};
        BlackbodyResponseConfig heat_response{};
        heat_response.min_kelvin = 800.0f;
        heat_response.max_kelvin = 2200.0f;
        heat_response.intensity = 1.0f;
        fill_flip_rgba_dual(energy,
                            temp,
                            film,
                            buf.tensor.tensor(),
                            2.5f,
                            film_decay,
                            beam_hist,
                            heat_response,
                            true,
                            true);

        void* pixels = nullptr;
        int pitch = 0;
        if (SDL_LockTexture(texture, nullptr, &pixels, &pitch) == 0) {
            auto* mem = dynamic_cast<InMemoryBackend*>(buf.tensor->backend());
            if (mem) {
                void* src_v = nullptr;
                size_t src_bytes = 0;
                if (mem->map(buf.tensor->handle(), &src_v, &src_bytes)) {
                    const size_t row_bytes = static_cast<size_t>(kWinW) * 4;
                    for (uint32_t y = 0; y < kWinH; ++y) {
                        std::memcpy(static_cast<uint8_t*>(pixels) + static_cast<size_t>(y) * pitch,
                                    static_cast<uint8_t*>(src_v) + static_cast<size_t>(y) * row_bytes,
                                    row_bytes);
                    }
                    mem->unmap(buf.tensor->handle());
                }
            }
            SDL_UnlockTexture(texture);
        }

        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);

        frame++;
        const auto now = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double>(now - last_title).count();
        if (dt >= 0.5) {
            const double fps = static_cast<double>(frame) / std::chrono::duration<double>(now - last_title).count();
            char title[256];
            std::snprintf(title, sizeof(title),
                          "kpath raster tensor demo | %s | raster %.2f ms | fps %.1f",
                          use_scatter ? "phased mirror array" : "gaussian raster",
                          avg_raster_ms,
                          fps);
            SDL_SetWindowTitle(window, title);
            last_title = now;
            frame = 0;
        }
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
