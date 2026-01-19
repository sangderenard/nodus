#include "common/tensors/abstraction/kpath/kpath_raster_utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"

namespace nodus::tensors::kpath {

namespace {

static inline std::chrono::high_resolution_clock::time_point now_hr() {
    return std::chrono::high_resolution_clock::now();
}

static inline double elapsed_ms(std::chrono::high_resolution_clock::time_point a,
                                std::chrono::high_resolution_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

bool fill_kernel_bank(AbstractTensor& out, TensorBackend* backend, std::span<const float> weights) {
    const uint32_t bank = static_cast<uint32_t>(weights.size());
    if (!out.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;
    void* ptr_v = nullptr;
    size_t bytes = 0;
    if (!mem->map(out.handle(), &ptr_v, &bytes)) return false;
    float* ptr = static_cast<float*>(ptr_v);
    for (uint32_t i = 0; i < bank; ++i) ptr[i] = weights[i];
    mem->unmap(out.handle());
    return true;
}

bool fill_kernel_ids(AbstractTensor& out,
                     const AbstractTensor& points,
                     uint32_t width,
                     uint32_t height,
                     uint32_t heads,
                     TensorBackend* backend) {
    if (!out.valid() || !points.valid() || heads == 0) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;

    void* pts_ptr_v = nullptr;
    size_t pts_bytes = 0;
    if (!mem->map(points.handle(), &pts_ptr_v, &pts_bytes)) return false;
    const float* pts = static_cast<const float*>(pts_ptr_v);

    void* id_ptr_v = nullptr;
    size_t id_bytes = 0;
    if (!mem->map(out.handle(), &id_ptr_v, &id_bytes)) {
        mem->unmap(points.handle());
        return false;
    }
    uint32_t* ids = static_cast<uint32_t*>(id_ptr_v);
    const uint32_t count = points.desc().shape.dims[0];

    for (uint32_t i = 0; i < count; ++i) {
        const float x = pts[i * 2 + 0];
        const float y = pts[i * 2 + 1];
        const uint32_t quad_x = (x < static_cast<float>(width) * 0.5f) ? 0u : 1u;
        const uint32_t quad_y = (y < static_cast<float>(height) * 0.5f) ? 0u : 1u;
        const uint32_t quad = quad_y * 2u + quad_x;
        ids[i] = quad % heads;
    }

    mem->unmap(points.handle());
    mem->unmap(out.handle());
    return true;
}

bool fill_laplacian_kernel(AbstractTensor& out, TensorBackend* backend) {
    if (!out.valid()) return false;
    auto* mem = dynamic_cast<InMemoryBackend*>(backend);
    if (!mem) return false;
    void* ptr_v = nullptr;
    size_t bytes = 0;
    if (!mem->map(out.handle(), &ptr_v, &bytes)) return false;
    float* k = static_cast<float*>(ptr_v);
    k[0] = 0.0f;  k[1] = 1.0f;  k[2] = 0.0f;
    k[3] = 1.0f;  k[4] = -4.0f; k[5] = 1.0f;
    k[6] = 0.0f;  k[7] = 1.0f;  k[8] = 0.0f;
    mem->unmap(out.handle());
    return true;
}

static void blackbody_rgb(float normalized, const BlackbodyResponseConfig& cfg, float& out_r, float& out_g, float& out_b) {
    const float t = std::clamp(normalized, 0.0f, 1.0f);
    const float kelvin = cfg.min_kelvin + (cfg.max_kelvin - cfg.min_kelvin) * t;
    const float temp = std::clamp(kelvin, 1000.0f, 40000.0f) / 100.0f;

    float r = 255.0f;
    float g = 255.0f;
    float b = 255.0f;

    if (temp <= 66.0f) {
        r = 255.0f;
        g = 99.4708025861f * std::log(temp) - 161.1195681661f;
        if (temp <= 19.0f) {
            b = 0.0f;
        } else {
            b = 138.5177312231f * std::log(temp - 10.0f) - 305.0447927307f;
        }
    } else {
        const float t2 = temp - 60.0f;
        r = 329.698727446f * std::pow(t2, -0.1332047592f);
        g = 288.1221695283f * std::pow(t2, -0.0755148492f);
        b = 255.0f;
    }

    r = std::clamp(r, 0.0f, 255.0f);
    g = std::clamp(g, 0.0f, 255.0f);
    b = std::clamp(b, 0.0f, 255.0f);

    out_r = (r / 255.0f) * cfg.intensity;
    out_g = (g / 255.0f) * cfg.intensity;
    out_b = (b / 255.0f) * cfg.intensity;
}

} // namespace

AbstractTensor make_flip_tensor(TensorBackend* backend, uint32_t width, uint32_t height) {
    TensorDesc desc{};
    desc.dtype = TensorDType::U32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {height, width};
    return AbstractTensor(desc, backend);
}

AbstractTensorPool::PooledTensor make_flip_tensor(AbstractTensorPool& pool,
                                                  TensorBackend* backend,
                                                  uint32_t width,
                                                  uint32_t height) {
    TensorDesc desc{};
    desc.dtype = TensorDType::U32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {height, width};
    return pool.acquire(desc, backend);
}

AbstractTensor make_kernel_bank(TensorBackend* backend, std::span<const float> weights) {
    const uint32_t bank = static_cast<uint32_t>(weights.size());
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {bank, 1u, 1u};
    AbstractTensor out(desc, backend);
    if (!fill_kernel_bank(out, backend, weights)) return {};
    return out;
}

AbstractTensorPool::PooledTensor make_kernel_bank(AbstractTensorPool& pool,
                                                  TensorBackend* backend,
                                                  std::span<const float> weights) {
    const uint32_t bank = static_cast<uint32_t>(weights.size());
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {bank, 1u, 1u};
    auto out = pool.acquire(desc, backend);
    if (!fill_kernel_bank(*out, backend, weights)) return {};
    return out;
}

AbstractTensor make_kernel_ids_for_heads(const AbstractTensor& points,
                                         uint32_t width,
                                         uint32_t height,
                                         uint32_t heads,
                                         TensorBackend* backend) {
    if (!points.valid() || heads == 0) return {};
    const TensorDesc& pd = points.desc();
    if (pd.dtype != TensorDType::F32 || pd.layout != TensorLayout::Dense) return {};
    if (pd.shape.dims.size() != 2 || pd.shape.dims[1] < 2) return {};
    const uint32_t count = pd.shape.dims[0];
    TensorDesc desc{};
    desc.dtype = TensorDType::U32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {count};
    AbstractTensor out(desc, backend);
    if (!fill_kernel_ids(out, points, width, height, heads, backend)) return {};
    return out;
}

AbstractTensorPool::PooledTensor make_kernel_ids_for_heads(AbstractTensorPool& pool,
                                                           const AbstractTensor& points,
                                                           uint32_t width,
                                                           uint32_t height,
                                                           uint32_t heads,
                                                           TensorBackend* backend) {
    if (!points.valid() || heads == 0) return {};
    const TensorDesc& pd = points.desc();
    if (pd.dtype != TensorDType::F32 || pd.layout != TensorLayout::Dense) return {};
    if (pd.shape.dims.size() != 2 || pd.shape.dims[1] < 2) return {};
    const uint32_t count = pd.shape.dims[0];
    TensorDesc desc{};
    desc.dtype = TensorDType::U32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {count};
    auto out = pool.acquire(desc, backend);
    if (!fill_kernel_ids(*out, points, width, height, heads, backend)) return {};
    return out;
}

AbstractTensor make_laplacian_kernel(TensorBackend* backend) {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {3u, 3u};
    AbstractTensor out(desc, backend);
    if (!fill_laplacian_kernel(out, backend)) return {};
    return out;
}

AbstractTensorPool::PooledTensor make_laplacian_kernel(AbstractTensorPool& pool, TensorBackend* backend) {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {3u, 3u};
    auto out = pool.acquire(desc, backend);
    if (!fill_laplacian_kernel(*out, backend)) return {};
    return out;
}

bool fill_flip_rgba(const TensorCanvas2D& energy,
                    FilmTensor2D& film,
                    AbstractTensor& rgba_u32,
                    float exposure_gain,
                    float film_decay) {
    if (!rgba_u32.valid()) return false;
    const TensorDesc& desc = rgba_u32.desc();
    if (desc.dtype != TensorDType::U32 || desc.layout != TensorLayout::Dense) return false;
    if (desc.shape.dims.size() != 2) return false;
    if (desc.shape.dims[0] != energy.height || desc.shape.dims[1] != energy.width) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(rgba_u32.backend());
    if (!mem) return false;
    void* ptr_v = nullptr;
    size_t bytes = 0;
    if (!mem->map(rgba_u32.handle(), &ptr_v, &bytes)) return false;
    auto* dst = static_cast<uint32_t*>(ptr_v);

    const float min_v = energy.min_value();
    const float max_v = energy.max_value();
    const float denom = std::max(max_v - min_v, 1e-6f);
    const uint32_t w = energy.width;
    const uint32_t h = energy.height;
    const uint32_t check = 16;

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const float v = (energy.at(x, y) - min_v) / denom;
            const float t = std::clamp(v * exposure_gain, 0.0f, 1.0f);

            const float r = std::pow(t, 0.85f);
            const float g = std::pow(t, 1.05f);
            const float b = std::pow(t, 1.25f);

            film.at(x, y, 0) = film.at(x, y, 0) * film_decay + r;
            film.at(x, y, 1) = film.at(x, y, 1) * film_decay + g;
            film.at(x, y, 2) = film.at(x, y, 2) * film_decay + b;

            const bool odd = ((x / check) ^ (y / check)) & 1u;
            const float base = odd ? 0.18f : 0.28f;

            const uint8_t cr = static_cast<uint8_t>(std::lround(std::clamp(base + film.at(x, y, 0), 0.0f, 1.0f) * 255.0f));
            const uint8_t cg = static_cast<uint8_t>(std::lround(std::clamp(base + film.at(x, y, 1), 0.0f, 1.0f) * 255.0f));
            const uint8_t cb = static_cast<uint8_t>(std::lround(std::clamp(base + film.at(x, y, 2), 0.0f, 1.0f) * 255.0f));
            const uint8_t ca = 255;

            const uint32_t packed = (static_cast<uint32_t>(cr)) |
                                    (static_cast<uint32_t>(cg) << 8) |
                                    (static_cast<uint32_t>(cb) << 16) |
                                    (static_cast<uint32_t>(ca) << 24);
            dst[static_cast<size_t>(y) * w + x] = packed;
        }
    }

    mem->unmap(rgba_u32.handle());
    return true;
}

bool fill_flip_rgba_dual(const TensorCanvas2D& energy,
                         const TensorCanvas2D& heat,
                         FilmTensor2D& film,
                         AbstractTensor& rgba_u32,
                         float exposure_gain,
                         float film_decay,
                         const BeamHistogram& beam,
                         const BlackbodyResponseConfig& blackbody,
                         bool normalize_energy,
                         bool normalize_heat,
                         float energy_norm_max,
                         float heat_norm_max,
                         FilmTimingBreakdown* timing) {
    if (timing) *timing = {};
    const auto t_total_start = now_hr();

    if (!rgba_u32.valid()) return false;
    const TensorDesc& desc = rgba_u32.desc();
    if (desc.dtype != TensorDType::U32 || desc.layout != TensorLayout::Dense) return false;
    if (desc.shape.dims.size() != 2) return false;
    if (desc.shape.dims[0] != energy.height || desc.shape.dims[1] != energy.width) return false;
    if (heat.width != energy.width || heat.height != energy.height) return false;

    auto* mem = dynamic_cast<InMemoryBackend*>(rgba_u32.backend());
    if (!mem) return false;
    void* ptr_v = nullptr;
    size_t bytes = 0;
    struct UnmapGuard {
        const InMemoryBackend* mem = nullptr;
        AbstractTensorHandle handle{};
        bool mapped = false;
        ~UnmapGuard() {
            if (mapped && mem) mem->unmap(handle);
        }
    } unmap_guard;

    {
        const auto t0 = now_hr();
        if (!mem->map(rgba_u32.handle(), &ptr_v, &bytes)) return false;
        const auto t1 = now_hr();
        if (timing) timing->map_ms += elapsed_ms(t0, t1);
        unmap_guard.mem = mem;
        unmap_guard.handle = rgba_u32.handle();
        unmap_guard.mapped = true;
    }
    auto* dst = static_cast<uint32_t*>(ptr_v);

    float min_e = 0.0f;
    float denom_e = std::max(energy_norm_max, 1e-6f);
    float min_h = 0.0f;
    float denom_h = std::max(heat_norm_max, 1e-6f);
    if (normalize_energy) {
        const auto t0 = now_hr();
        min_e = energy.min_value();
        const float max_e = energy.max_value();
        denom_e = std::max(max_e - min_e, 1e-6f);
        const auto t1 = now_hr();
        if (timing) timing->normalize_energy_ms += elapsed_ms(t0, t1);
    }
    if (normalize_heat) {
        const auto t0 = now_hr();
        min_h = heat.min_value();
        const float max_h = heat.max_value();
        denom_h = std::max(max_h - min_h, 1e-6f);
        const auto t1 = now_hr();
        if (timing) timing->normalize_heat_ms += elapsed_ms(t0, t1);
    }

    // Keep legacy FilmTensor2D path for non-demo uses: use the existing scalar accessors.
    const float inv_denom_e = 1.0f / denom_e;
    const float inv_denom_h = 1.0f / denom_h;
    const uint32_t check_shift = 4; // log2(16)
    const uint32_t w = energy.width;
    const uint32_t h = energy.height;
    const uint32_t chans = film.channels;
    if (film.width != w || film.height != h || chans < 3) return false;

    {
        const auto t0 = now_hr();
        for (uint32_t y = 0; y < h; ++y) {
            const uint32_t ycheck = (y >> check_shift);
            for (uint32_t x = 0; x < w; ++x) {
                const size_t idx = static_cast<size_t>(y) * w + x;
                const float e_norm = (energy.at(x, y) - min_e) * inv_denom_e;
                const float h_norm = (heat.at(x, y) - min_h) * inv_denom_h;

                float e = e_norm * exposure_gain;
                if (e < 0.0f) e = 0.0f;
                if (e > 1.0f) e = 1.0f;

                const float beam_r = beam.r * e;
                const float beam_g = beam.g * e;
                const float beam_b = beam.b * e;

                float heat_r = 0.0f;
                float heat_g = 0.0f;
                float heat_b = 0.0f;
                blackbody_rgb(h_norm, blackbody, heat_r, heat_g, heat_b);

                const float r = beam_r + heat_r;
                const float g = beam_g + heat_g;
                const float b = beam_b + heat_b;

                film.at(x, y, 0) = film.at(x, y, 0) * film_decay + r;
                film.at(x, y, 1) = film.at(x, y, 1) * film_decay + g;
                film.at(x, y, 2) = film.at(x, y, 2) * film_decay + b;

                const uint32_t xcheck = (x >> check_shift);
                const bool odd = ((xcheck ^ ycheck) & 1u) != 0u;
                const float base = odd ? 0.18f : 0.28f;

                float pr = base + film.at(x, y, 0);
                float pg = base + film.at(x, y, 1);
                float pb = base + film.at(x, y, 2);
                if (pr < 0.0f) pr = 0.0f;
                if (pr > 1.0f) pr = 1.0f;
                if (pg < 0.0f) pg = 0.0f;
                if (pg > 1.0f) pg = 1.0f;
                if (pb < 0.0f) pb = 0.0f;
                if (pb > 1.0f) pb = 1.0f;

                const uint8_t cr = static_cast<uint8_t>(std::lround(pr * 255.0f));
                const uint8_t cg = static_cast<uint8_t>(std::lround(pg * 255.0f));
                const uint8_t cb = static_cast<uint8_t>(std::lround(pb * 255.0f));
                const uint8_t ca = 255;

                const uint32_t packed = (static_cast<uint32_t>(cr)) |
                                        (static_cast<uint32_t>(cg) << 8) |
                                        (static_cast<uint32_t>(cb) << 16) |
                                        (static_cast<uint32_t>(ca) << 24);
                dst[idx] = packed;
            }
        }
        const auto t1 = now_hr();
        if (timing) timing->pixel_loop_ms += elapsed_ms(t0, t1);
    }

    {
        const auto t0 = now_hr();
        unmap_guard.mapped = false;
        mem->unmap(rgba_u32.handle());
        const auto t1 = now_hr();
        if (timing) timing->unmap_ms += elapsed_ms(t0, t1);
    }

    if (timing) timing->total_ms = elapsed_ms(t_total_start, now_hr());
    return true;
}

bool fill_flip_rgba_dual_tensor(const TensorCanvas2D& energy,
                                const TensorCanvas2D& heat,
                                FilmTensor& film,
                                AbstractTensorPool::PooledTensor& film_exposures,
                                AbstractTensorPool::PooledTensor& film_exposures_scratch,
                                AbstractTensor& film_rgb_f32,
                                AbstractTensor& rgba_u32,
                                float exposure_gain,
                                float film_decay,
                                const BeamHistogram& beam,
                                const BlackbodyResponseConfig& blackbody,
                                bool normalize_energy,
                                bool normalize_heat,
                                float energy_norm_max,
                                float heat_norm_max,
                                FilmTimingBreakdown* timing) {
    auto log_fail = [](const char* msg) {
        std::cerr << "[KPATH-FILM] fill_flip_rgba_dual_tensor failed: " << msg << "\n";
        return false;
    };
    if (timing) *timing = {};
    const auto t_total_start = now_hr();

    if (!film.valid()) return log_fail("film invalid");
    if (!rgba_u32.valid() || !film_exposures.valid() || !film_exposures_scratch.valid() || !film_rgb_f32.valid()) {
        return log_fail("invalid tensor handle(s)");
    }
    auto* exposures_backend = film_exposures.tensor().backend();
    auto* scratch_backend = film_exposures_scratch.tensor().backend();
    if (exposures_backend != rgba_u32.backend() || scratch_backend != rgba_u32.backend() ||
        rgba_u32.backend() != film_rgb_f32.backend()) {
        return log_fail("backend mismatch between film/exposures/scratch/rgba");
    }

    const TensorDesc& out_desc = rgba_u32.desc();
    if (out_desc.dtype != TensorDType::U32 || out_desc.layout != TensorLayout::Dense) {
        return log_fail("rgba_u32 desc not U32 Dense");
    }
    if (out_desc.shape.dims.size() != 2) {
        return log_fail("rgba_u32 dims not 2D");
    }

    const TensorDesc& exposures_desc = film_exposures.tensor().desc();
    if (exposures_desc.dtype != TensorDType::F32 || exposures_desc.layout != TensorLayout::Dense) {
        return log_fail("exposures desc not F32 Dense");
    }
    if (exposures_desc.shape.dims.size() != 3) {
        return log_fail("exposures dims not 3D");
    }
    const uint32_t h = exposures_desc.shape.dims[0];
    const uint32_t w = exposures_desc.shape.dims[1];
    const uint32_t total_channels = exposures_desc.shape.dims[2];
    if (film.total_channels() != total_channels) {
        std::cerr << "[KPATH-FILM] channel mismatch: film=" << film.total_channels()
                  << " exposures=" << total_channels << "\n";
        return false;
    }
    if (film.width() != w || film.height() != h) {
        std::cerr << "[KPATH-FILM] film size mismatch: film=" << film.width() << "x" << film.height()
                  << " exposures=" << w << "x" << h << "\n";
        return false;
    }
    if (energy.width != w || energy.height != h) {
        std::cerr << "[KPATH-FILM] energy size mismatch: energy=" << energy.width << "x" << energy.height
                  << " exposures=" << w << "x" << h << "\n";
        return false;
    }
    if (heat.width != w || heat.height != h) {
        std::cerr << "[KPATH-FILM] heat size mismatch: heat=" << heat.width << "x" << heat.height
                  << " exposures=" << w << "x" << h << "\n";
        return false;
    }
    if (out_desc.shape.dims[0] != h || out_desc.shape.dims[1] != w) {
        std::cerr << "[KPATH-FILM] rgba size mismatch: rgba=" << out_desc.shape.dims[1]
                  << "x" << out_desc.shape.dims[0] << " exposures=" << w << "x" << h << "\n";
        return false;
    }

    const TensorDesc& scratch_desc = film_exposures_scratch.tensor().desc();
    if (scratch_desc.dtype != TensorDType::F32 || scratch_desc.layout != TensorLayout::Dense ||
        scratch_desc.shape.dims != exposures_desc.shape.dims) {
        return log_fail("exposures_scratch desc mismatch");
    }

    const TensorDesc& agg_desc = film_rgb_f32.desc();
    if (agg_desc.dtype != TensorDType::F32 || agg_desc.layout != TensorLayout::Dense) {
        return log_fail("film_rgb_f32 desc not F32 Dense");
    }
    if (agg_desc.shape.dims.size() != 3) {
        return log_fail("film_rgb_f32 dims not 3D");
    }
    if (agg_desc.shape.dims[0] != h || agg_desc.shape.dims[1] != w ||
        agg_desc.shape.dims[2] != film.histogram_count()) {
        std::cerr << "[KPATH-FILM] film_rgb_f32 shape mismatch: rgb="
                  << agg_desc.shape.dims[1] << "x" << agg_desc.shape.dims[0]
                  << "x" << agg_desc.shape.dims[2] << " expected="
                  << w << "x" << h << "x" << film.histogram_count() << "\n";
        return false;
    }

    auto* mem = dynamic_cast<InMemoryBackend*>(rgba_u32.backend());
    if (!mem) return log_fail("rgba_u32 backend is not InMemoryBackend");

    void* rgba_ptr_v = nullptr;
    size_t rgba_bytes = 0;
    {
        const auto t0 = now_hr();
        if (!mem->map(rgba_u32.handle(), &rgba_ptr_v, &rgba_bytes)) return log_fail("rgba_u32 map failed");
        const auto t1 = now_hr();
        if (timing) timing->map_ms += elapsed_ms(t0, t1);
    }

    struct UnmapGuard {
        const InMemoryBackend* mem = nullptr;
        AbstractTensorHandle handle{};
        bool mapped = false;
        ~UnmapGuard() {
            if (mapped && mem) mem->unmap(handle);
        }
    } rgba_guard{mem, rgba_u32.handle(), true};

    auto* dst = static_cast<uint32_t*>(rgba_ptr_v);

    float min_e = 0.0f;
    float denom_e = std::max(energy_norm_max, 1e-6f);
    float min_h = 0.0f;
    float denom_h = std::max(heat_norm_max, 1e-6f);
    if (normalize_energy) {
        const auto t0 = now_hr();
        min_e = energy.min_value();
        const float max_e = energy.max_value();
        denom_e = std::max(max_e - min_e, 1e-6f);
        const auto t1 = now_hr();
        if (timing) timing->normalize_energy_ms += elapsed_ms(t0, t1);
    }
    if (normalize_heat) {
        const auto t0 = now_hr();
        min_h = heat.min_value();
        const float max_h = heat.max_value();
        denom_h = std::max(max_h - min_h, 1e-6f);
        const auto t1 = now_hr();
        if (timing) timing->normalize_heat_ms += elapsed_ms(t0, t1);
    }

    const float inv_denom_e = 1.0f / denom_e;
    const float inv_denom_h = 1.0f / denom_h;

    const auto t_build_start = now_hr();
    const size_t total_points = static_cast<size_t>(h) * w;
    TensorDesc pts_desc{};
    pts_desc.dtype = TensorDType::F32;
    pts_desc.layout = TensorLayout::Dense;
    pts_desc.shape.dims = {static_cast<uint32_t>(total_points), 2u};

    TensorDesc vals_desc{};
    vals_desc.dtype = TensorDType::F32;
    vals_desc.layout = TensorLayout::Dense;
    vals_desc.shape.dims = {static_cast<uint32_t>(total_points), total_channels};

    AbstractTensor points = AbstractTensor::create(pts_desc, rgba_u32.backend());
    AbstractTensor values = AbstractTensor::create(vals_desc, rgba_u32.backend());
    if (!points.valid() || !values.valid()) return false;

    void* pts_map = nullptr;
    size_t pts_bytes = 0;
    if (!mem->map(points.handle(), &pts_map, &pts_bytes)) return false;

    void* vals_map = nullptr;
    size_t vals_bytes = 0;
    if (!mem->map(values.handle(), &vals_map, &vals_bytes)) {
        mem->unmap(points.handle());
        return false;
    }
    const auto t_build_end = now_hr();
    if (timing) timing->build_ms += elapsed_ms(t_build_start, t_build_end);

    auto* pts_ptr = static_cast<float*>(pts_map);
    auto* vals_ptr = static_cast<float*>(vals_map);

    const auto t0 = now_hr();
    size_t idx = 0;
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            const float e_norm = (energy.at(x, y) - min_e) * inv_denom_e;
            const float h_norm = (heat.at(x, y) - min_h) * inv_denom_h;

            float e = e_norm * exposure_gain;
            e = std::clamp(e, 0.0f, 1.0f);

            float heat_r = 0.0f;
            float heat_g = 0.0f;
            float heat_b = 0.0f;
            blackbody_rgb(h_norm, blackbody, heat_r, heat_g, heat_b);

            pts_ptr[idx * 2 + 0] = static_cast<float>(x);
            pts_ptr[idx * 2 + 1] = static_cast<float>(y);

            for (uint32_t ch = 0; ch < total_channels; ++ch) {
                float contrib = 0.0f;
                if (ch == 0) contrib = beam.r * e + heat_r;
                else if (ch == 1) contrib = beam.g * e + heat_g;
                else if (ch == 2) contrib = beam.b * e + heat_b;
                vals_ptr[idx * total_channels + ch] = contrib;
            }
            ++idx;
        }
    }
    const auto t1 = now_hr();
    if (timing) timing->pixel_loop_ms += elapsed_ms(t0, t1);

    mem->unmap(points.handle());
    mem->unmap(values.handle());

    const auto t_scatter_start = now_hr();
    AbstractTensor* scaled_base = &film_exposures_scratch.tensor();
    if (!tensor_copy_f32_into(film_exposures.tensor(), scaled_base)) return false;
    if (!tensor_axpby_f32(*scaled_base, film_decay, *scaled_base, 0.0f, scaled_base)) return false;
    // Skip scatter contributions for already-saturated film pixels.
    scaled_base->set_slice_saturate_threshold(1.0f);

    AbstractTensor scatter_out;
    TensorTransferConfig scatter_cfg{};
    scatter_cfg.premix_scatter = TensorMixPolicy::Add;
    scatter_cfg.postmix_scatter = TensorMixPolicy::Add;
    scatter_cfg.clamp = true;
    if (!tensor_scatter_nd(*scaled_base, points, values, &scatter_out, scatter_cfg)) return false;
    if (!scatter_out.valid()) return false;
    if (!tensor_copy_f32_into(scatter_out, &film_exposures.tensor())) return false;
    const auto t_scatter_end = now_hr();
    if (timing) timing->scatter_ms += elapsed_ms(t_scatter_start, t_scatter_end);

    const auto t_reduce_start = now_hr();
    if (!film.reduce_all(film_exposures.tensor(), &film_rgb_f32)) return false;
    const auto t_reduce_end = now_hr();
    if (timing) timing->reduce_ms += elapsed_ms(t_reduce_start, t_reduce_end);

    const uint32_t check_shift = 4; // log2(16)
    void* film_ptr_v = nullptr;
    size_t film_bytes = 0;
    if (!mem->map(film_rgb_f32.handle(), &film_ptr_v, &film_bytes)) return false;
    struct UnmapGuard film_guard{mem, film_rgb_f32.handle(), true};
    auto* film_ptr = static_cast<const float*>(film_ptr_v);
    {
        const auto t2 = now_hr();
        const uint32_t agg_channels = static_cast<uint32_t>(film_rgb_f32.desc().shape.dims[2]);
        const size_t row_stride = static_cast<size_t>(w) * agg_channels;
        for (uint32_t y = 0; y < h; ++y) {
            const uint32_t ycheck = (y >> check_shift);
            const float* film_row = film_ptr + static_cast<size_t>(y) * row_stride;
            for (uint32_t x = 0; x < w; ++x) {
                const size_t base = static_cast<size_t>(x) * agg_channels;
                const uint32_t xcheck = (x >> check_shift);
                const bool odd = ((xcheck ^ ycheck) & 1u) != 0u;
                const float bias = odd ? 0.18f : 0.28f;

                auto fetch_channel = [&](uint32_t channel) -> float {
                    if (channel < agg_channels) return film_row[base + channel];
                    return 0.0f;
                };

                float pr = bias + fetch_channel(0);
                float pg = bias + fetch_channel(1);
                float pb = bias + fetch_channel(2);
                pr = std::clamp(pr, 0.0f, 1.0f);
                pg = std::clamp(pg, 0.0f, 1.0f);
                pb = std::clamp(pb, 0.0f, 1.0f);

                const uint8_t cr = static_cast<uint8_t>(std::lround(pr * 255.0f));
                const uint8_t cg = static_cast<uint8_t>(std::lround(pg * 255.0f));
                const uint8_t cb = static_cast<uint8_t>(std::lround(pb * 255.0f));
                const uint8_t ca = 255;
                dst[static_cast<size_t>(y) * w + x] = (static_cast<uint32_t>(cr)) |
                                                      (static_cast<uint32_t>(cg) << 8) |
                                                      (static_cast<uint32_t>(cb) << 16) |
                                                      (static_cast<uint32_t>(ca) << 24);
            }
        }
        const auto t3 = now_hr();
        if (timing) timing->pixel_loop_ms += elapsed_ms(t2, t3);
    }

    {
        const auto t0 = now_hr();
        film_guard.mapped = false;
        mem->unmap(film_rgb_f32.handle());
        rgba_guard.mapped = false;
        mem->unmap(rgba_u32.handle());
        const auto t1 = now_hr();
        if (timing) timing->unmap_ms += elapsed_ms(t0, t1);
    }

    if (timing) timing->total_ms = elapsed_ms(t_total_start, now_hr());
    return true;
}

} // namespace nodus::tensors::kpath
