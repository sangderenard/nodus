// Draw a droopy blended cable between two points.
static void draw_cable_blend(uint8_t* img, int w, int h, int pitch, int ax, int ay, int bx, int by, int jacket_px, int jacket_border, Color core_col, int segments, float sag_factor, float relax_v) {
    if (!img) return;
    if (segments < 4) segments = 4;
    float dx = float(bx - ax);
    float dy = float(by - ay);
    float dist = std::sqrt(dx*dx + dy*dy);
    float sag = dist * sag_factor;
    // core color modulated by relax value but keep a minimum visibility so cables never fully disappear
    Color core = core_col;
    float rv = std::min(1.0f, std::max(0.0f, relax_v));
    const float min_vis = 0.18f;
    float vis = std::max(min_vis, rv);
    core.a = static_cast<uint8_t>(std::lround(core.a * vis));
    // jacket color: use faint neutral grey to avoid dark edges
    Color jacket{200,200,200,13};
    Color jacket_edge{200,200,200,20};
    // choose samples so blob spacing is <= ~0.6 * jacket_px to avoid visible gaps
    int min_seg_for_spacing = 1;
    if (jacket_px > 0) min_seg_for_spacing = static_cast<int>(std::ceil(dist / (std::max(1.0f, float(jacket_px) * 0.6f))));
    int use_segments = std::max(segments, std::max(4, min_seg_for_spacing));
    // sample points and draw overlapping blobs for a continuous tube
    for (int si = 0; si <= use_segments; ++si) {
        float t = float(si) / float(use_segments);
        float px = float(ax) + dx * t;
        float py = float(ay) + dy * t + sag * std::sin(3.14159265f * t);
        int ipx = static_cast<int>(std::lround(px));
        int ipy = static_cast<int>(std::lround(py));
        // smear segment from previous sample to this sample (avoids per-sample caps)
        if (si > 0) {
            float px0 = float(ax) + dx * float(si - 1) / float(use_segments);
            float py0 = float(ay) + dy * float(si - 1) / float(use_segments) + sag * std::sin(3.14159265f * (float(si - 1) / float(use_segments)));
            // outer jacket smear
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, jacket_px, jacket);
            // thin jacket edge smear
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, std::max(1, jacket_px - 1), jacket_edge);
            // core smear
            int core_r = std::max(1, jacket_px - jacket_border);
            Color corec = core;
            corec.a = static_cast<uint8_t>(std::lround(corec.a * 1.0f));
            draw_segment_smear(img, w, h, pitch, px0, py0, px, py, core_r, corec);
        }
    }
    // end plugs: blended caps so endpoints remain joined to smears
    draw_blob_blend(img, w, h, pitch, ax, ay, jacket_px, Color{200,200,200,13});
    draw_blob_blend(img, w, h, pitch, bx, by, jacket_px, Color{200,200,200,13});
    // inner core caps (blended)
    Color corecap = core_col;
    corecap.a = static_cast<uint8_t>(std::lround(corecap.a * 0.2f));
    draw_blob_blend(img, w, h, pitch, ax, ay, std::max(1, jacket_px - jacket_border), corecap);
    draw_blob_blend(img, w, h, pitch, bx, by, std::max(1, jacket_px - jacket_border), corecap);
}

// Helper: draw a blended thick segment with optional alpha scale.
static void draw_segment_blend(uint8_t* img, int w, int h, int pitch, float x1, float y1, float x2, float y2, int radius, Color col, float alpha_scale = 1.0f) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float len2 = dx*dx + dy*dy;
    float rplus = float(radius) + 1.0f; // allow soft edge
    int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
    int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
    int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
    int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
    minx = std::max(minx, 0);
    miny = std::max(miny, 0);
    maxx = std::min(maxx, w - 1);
    maxy = std::min(maxy, h - 1);
    for (int py = miny; py <= maxy; ++py) {
        for (int px = minx; px <= maxx; ++px) {
            float cx = px + 0.5f;
            float cy = py + 0.5f;
            float t = 0.0f;
            if (len2 > 1e-6f) {
                t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                if (t < 0.0f) t = 0.0f;
                else if (t > 1.0f) t = 1.0f;
            }
            float closestX = x1 + dx * t;
            float closestY = y1 + dy * t;
            float ddx = cx - closestX;
            float ddy = cy - closestY;
            float dist = std::sqrt(ddx*ddx + ddy*ddy);
            if (dist <= rplus) {
                float coverage = 1.0f - (dist / rplus);
                float a_scaled = float(col.a) * coverage * alpha_scale;
                if (a_scaled <= 0.0f) continue;
                Color cc = col;
                cc.a = static_cast<uint8_t>(std::lround(std::min(255.0f, a_scaled)));
                uint8_t* dstp = img + py * pitch + px * 4;
                blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
            }
        }
    }
}

// Orthographic projection of 3D rope verts into 2D table space with a subtle tilt.
static void project_rope_vertices_ortho(const float* verts3, int count, float tilt_x, float tilt_y, std::vector<float>& out_xy, std::vector<float>& out_z, float &out_min_z, float &out_max_z) {
    out_xy.resize(static_cast<std::size_t>(count) * 2);
    out_z.resize(static_cast<std::size_t>(count));
    out_min_z = std::numeric_limits<float>::max();
    out_max_z = std::numeric_limits<float>::lowest();
    for (int i = 0; i < count; ++i) {
        float x = verts3[3*i+0];
        float y = verts3[3*i+1];
        float z = verts3[3*i+2];
        out_z[static_cast<std::size_t>(i)] = z;
        out_xy[2*i+0] = x + tilt_x * z;
        out_xy[2*i+1] = y + tilt_y * z;
        out_min_z = std::min(out_min_z, z);
        out_max_z = std::max(out_max_z, z);
    }
    if (out_min_z > out_max_z) {
        out_min_z = out_max_z = 0.0f;
    }
}

// C API wrapper: return projected 2D verts for a rope known to this table.
extern "C" int32_t gp_table_get_projected_rope_vertices(GP_TableContext* ctx, int32_t rope_idx, float* out_xy, int32_t max_count) {
    if (!ctx || !out_xy) return 0;
    RopeSim* sim = gp_table_get_rope_sim(ctx);
    if (!sim) return 0;
    if (rope_idx < 0) return 0;
    int vc = rope_sim_get_vertex_count(sim, rope_idx);
    if (vc <= 0) return 0;
    if (max_count < 2 * vc) return 0;

    std::vector<float> verts3(static_cast<size_t>(vc) * 3);
    int got = rope_sim_get_vertices3(sim, rope_idx, verts3.data(), static_cast<int>(verts3.size()));
    if (got != vc) return 0;

    std::vector<float> proj_xy;
    std::vector<float> proj_z;
    float min_z = 0.0f, max_z = 0.0f;
    project_rope_vertices_ortho(verts3.data(), got, ctx->st.cable_tilt_x, ctx->st.cable_tilt_y, proj_xy, proj_z, min_z, max_z);

    // copy into out_xy interleaved
    for (int i = 0; i < vc; ++i) {
        out_xy[2*i+0] = proj_xy[2*i+0];
        out_xy[2*i+1] = proj_xy[2*i+1];
    }
    return vc;
}

extern "C" int32_t gp_table_set_prospective_rope_index(GP_TableContext* ctx, int32_t rope_idx) {
    if (!ctx) return 0;
    ctx->prospective_rope_idx = rope_idx;
    return 1;
}

// Build Catmull-Rom samples for a rope and optionally depth samples aligned with them.
static void build_rope_samples_with_depth(const float* verts2d, const float* depth_per_vert, int count, int jacket_px, std::vector<std::pair<float,float>>& samples, std::vector<float>* depth_samples) {
    samples.clear();
    if (depth_samples) depth_samples->clear();
    if (!verts2d || count < 2) return;

    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts2d[2*idx+0], verts2d[2*idx+1]);
    };
    auto get_depth = [&](int idx) {
        if (!depth_per_vert) return 0.0f;
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return depth_per_vert[idx];
    };

    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        float dz = 0.0f;
        if (depth_per_vert) {
            float z0 = get_depth(i-1);
            float z1 = get_depth(i+0);
            float z2 = get_depth(i+1);
            float z3 = get_depth(i+2);
            dz = 0.5f * ((2.0f * z1) + (-z0 + z2) * t + (2.0f*z0 - 5.0f*z1 + 4.0f*z2 - z3) * t2 + (-z0 + 3.0f*z1 - 3.0f*z2 + z3) * t3);
        }
        return std::tuple<float,float,float>(x,y,dz);
    };

    samples.reserve(static_cast<std::size_t>((count - 1) * 8));
    if (depth_samples) depth_samples->reserve(static_cast<std::size_t>((count - 1) * 8));
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // sample spacing: ~0.6*jacket_px to avoid visible gaps while following spline
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto [sx, sy, sz] = catmull(i, t);
            samples.emplace_back(sx, sy);
            if (depth_samples) depth_samples->push_back(sz);
        }
    }
}

static void draw_rope_fiber_overlay(uint8_t* img, int w, int h, int pitch, const std::vector<std::pair<float,float>>& samples, const std::vector<float>& depth_samples, int radius, Color glow_col, float depth_fade, float gain, float min_z, float max_z) {
    if (!img || samples.size() < 2) return;
    float depth_span = std::max(1e-3f, max_z - min_z);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        float fade = 1.0f;
        if (depth_samples.size() == samples.size()) {
            float z_avg = 0.5f * (depth_samples[i] + depth_samples[i+1]);
            float norm = (z_avg - min_z) / depth_span;
            norm = std::clamp(norm, 0.0f, 1.0f);
            fade = 1.0f - depth_fade * norm;
        }
        float alpha_scale = std::max(0.0f, gain * fade);
        draw_segment_blend(img, w, h, pitch, a.first, a.second, b.first, b.second, radius, glow_col, alpha_scale);
    }
}

// forward declaration: color interpolation used by several rope/rope-glow helpers
static inline Color lerp_color(Color a, Color b, float t);

static float draw_rope_diffused_glow(
    uint8_t* img,
    int w,
    int h,
    int pitch,
    const std::vector<std::pair<float,float>>& samples,
    const std::vector<float>& depth_samples,
    bool start_is_front,
    float start_glow,
    float depth_fade,
    float gain,
    Color glow_col,
    const std::vector<Color>* jacket_colors,
    float min_z,
    float max_z,
    int radius) {
    if (samples.size() < 2) return 0.0f;
    float clamped_glow = std::max(0.0f, start_glow);
    if (clamped_glow <= 0.0f) return 0.0f;
    float depth_span = std::max(1e-3f, max_z - min_z);

    std::vector<float> prefix(samples.size(), 0.0f);
    for (size_t i = 1; i < samples.size(); ++i) {
        float dx = samples[i].first - samples[i-1].first;
        float dy = samples[i].second - samples[i-1].second;
        prefix[i] = prefix[i-1] + std::sqrt(dx*dx + dy*dy);
    }
    float total_len = prefix.back();
    if (total_len <= 1e-5f) return 0.0f;

    auto depth_scale_at = [&](size_t idx) {
        if (depth_samples.size() != samples.size()) return 1.0f;
        float norm = (depth_samples[idx] - min_z) / depth_span;
        norm = std::clamp(norm, 0.0f, 1.0f);
        return 1.0f - depth_fade * norm;
    };
    auto dist_from_start = [&](size_t idx) {
        float d = prefix[idx];
        return start_is_front ? d : (total_len - d);
    };

    // modest diffusion: higher decay keeps glow tighter along cable
    const float decay = 1.1f;
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float da = dist_from_start(i);
        float db = dist_from_start(i + 1);
        float dist_mid = 0.5f * (da + db);
        float along = (total_len > 1e-4f) ? (dist_mid / total_len) : 0.0f;
        float atten = std::exp(-decay * along);
        float depth_scale = 0.5f * (depth_scale_at(i) + depth_scale_at(i + 1));
        float alpha_scale = gain * clamped_glow * atten * depth_scale;
        if (img && alpha_scale > 0.0f) {
            Color seg_glow = glow_col;
            if (jacket_colors && !jacket_colors->empty()) {
                float u = (total_len > 1e-4f) ? ((prefix[i] + prefix[i + 1]) * 0.5f / total_len) : 0.0f;
                int seg_idx = static_cast<int>(std::floor(u * static_cast<float>(jacket_colors->size())));
                seg_idx = std::clamp(seg_idx, 0, static_cast<int>(jacket_colors->size()) - 1);
                Color jacket_col = (*jacket_colors)[static_cast<size_t>(seg_idx)];
                if (jacket_col.a > 0) {
                    float jacket_weight = std::clamp(static_cast<float>(jacket_col.a) / 255.0f, 0.0f, 1.0f) * 0.4f;
                    seg_glow = lerp_color(seg_glow, jacket_col, jacket_weight);
                    seg_glow.a = glow_col.a;
                }
            }
            draw_segment_kernel_glow(img, w, h, pitch, samples[i].first, samples[i].second, samples[i+1].first, samples[i+1].second, float(radius), seg_glow, alpha_scale);
        }
    }

    size_t dest_idx = start_is_front ? samples.size() - 1 : 0;
    float transmitted = clamped_glow * std::exp(-decay) * depth_scale_at(dest_idx);
    return transmitted;
}

// Draw a smooth blended rope/tube along given interleaved vertices using Catmull-Rom
// verts: float array [x0,y0, x1,y1, ...], count = number of vertices
static void draw_rope_curve_blend(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, Color core_col, int samples_per_segment) {
    if (!img || !verts || count < 2) return;
    if (samples_per_segment < 2) samples_per_segment = 2;

    auto get = [&](int idx) {
        // clamp
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };

    auto catmull = [&](int i, float t) {
        // control points p0..p3 for segment between i and i+1
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        // Catmull-Rom with tension 0.5
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    // analytic Catmull-Rom derivative (tangent) for parametric cross-sections
    

    // Build dense samples along the spline then rasterize as thick blended segments
    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        // estimate chord length for this segment (p1-p2)
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // sample spacing: ~0.6*jacket_px to avoid visible gaps while following spline
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1; // avoid duplicate samples at segment boundaries
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }

    if (samples.empty()) return;

    // endpoint caps removed to avoid double-painted ends

    // draw jacket (slimmer and translucent so background shows through)
    int eff_jacket = std::max(1, jacket_px - 1);
    // use neutral grey jacket as requested
    Color jacket_col = Color{200,200,200, static_cast<uint8_t>(13)};
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_col, 1.0f);
    }

    // draw core (thinner, faint tinted core so cable looks clear)
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    Color corec = core_col;
    // keep inner core at requested alpha so producer color remains visible
    corec.a = static_cast<uint8_t>(std::clamp<int>(corec.a, 0, 255));
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, corec, 1.0f);
    }
}

static void build_rope_samples_with_tangents(const float* verts, int count, int jacket_px, std::vector<std::pair<float,float>>& samples, std::vector<std::pair<float,float>>& tangents) {
    samples.clear();
    tangents.clear();
    if (!verts || count < 2) return;
    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        int s_start = (i == 0) ? 0 : 1;
        for (int s = s_start; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
}

static void draw_rope_jacket_overlay(uint8_t* img, int w, int h, int pitch,
                                     const std::vector<std::pair<float,float>>& samples,
                                     const std::vector<std::pair<float,float>>& tangents,
                                     int jacket_px,
                                     const std::vector<Color>& jacket_colors) {
    if (!img || samples.size() < 2 || samples.size() != tangents.size() || jacket_colors.empty()) return;
    int eff_jacket = std::max(1, jacket_px - 1);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = (samples.size() > 1) ? (static_cast<float>(i) + 0.5f) / static_cast<float>(samples.size() - 1) : 0.0f;
        int seg_idx = static_cast<int>(std::floor(u * static_cast<float>(jacket_colors.size())));
        seg_idx = std::clamp(seg_idx, 0, static_cast<int>(jacket_colors.size()) - 1);
        Color col = jacket_colors[static_cast<size_t>(seg_idx)];
        if (col.a == 0) continue;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, col, 1.0f);
    }
}

static void draw_rope_core_gradient(uint8_t* img, int w, int h, int pitch,
                                    const std::vector<std::pair<float,float>>& samples,
                                    const std::vector<std::pair<float,float>>& tangents,
                                    int jacket_px,
                                    int jacket_border,
                                    Color col_a,
                                    Color col_b,
                                    float intensity) {
    if (!img || samples.size() < 2 || samples.size() != tangents.size()) return;
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    float base_alpha = 255.0f * std::clamp(intensity, 0.0f, 1.0f) * 0.18f;
    auto lerp_color = [&](const Color &a, const Color &b, float t) {
        float tt = std::clamp(t, 0.0f, 1.0f);
        Color out;
        out.r = static_cast<uint8_t>(std::lround(float(a.r) * (1.0f - tt) + float(b.r) * tt));
        out.g = static_cast<uint8_t>(std::lround(float(a.g) * (1.0f - tt) + float(b.g) * tt));
        out.b = static_cast<uint8_t>(std::lround(float(a.b) * (1.0f - tt) + float(b.b) * tt));
        out.a = 0;
        return out;
    };
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float u = (samples.size() > 1) ? static_cast<float>(i) / static_cast<float>(samples.size() - 1) : 0.0f;
        float alpha_mix = (1.0f - u) * float(col_a.a) + u * float(col_b.a);
        uint8_t alpha = static_cast<uint8_t>(std::lround(base_alpha * std::clamp(alpha_mix / 255.0f, 0.0f, 1.0f)));
        if (alpha == 0) continue;
        Color core_col = lerp_color(col_a, col_b, u);
        core_col.a = alpha;
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, core_col, 1.0f);
    }
}

// Convert HSV (h in 0..1, s 0..1, v 0..1) to Color (alpha=255)
static inline Color hsv_to_color(float h, float s, float v, uint8_t a=255) {
    h = h - std::floor(h);
    float hh = h * 6.0f;
    int i = static_cast<int>(std::floor(hh));
    float f = hh - float(i);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));
    float r=0,g=0,b=0;
    switch (i % 6) {
        case 0: r = v; g = t; b = p; break;
        case 1: r = q; g = v; b = p; break;
        case 2: r = p; g = v; b = t; break;
        case 3: r = p; g = q; b = v; break;
        case 4: r = t; g = p; b = v; break;
        case 5: r = v; g = p; b = q; break;
    }
    return Color{ static_cast<uint8_t>(std::lround(r * 255.0f)), static_cast<uint8_t>(std::lround(g * 255.0f)), static_cast<uint8_t>(std::lround(b * 255.0f)), a };
}

// Convert RGB Color to hue in [0..1]
static inline float rgb_to_hue(const Color &c) {
    float r = c.r / 255.0f;
    float g = c.g / 255.0f;
    float b = c.b / 255.0f;
    float mx = std::max(r, std::max(g, b));
    float mn = std::min(r, std::min(g, b));
    float d = mx - mn;
    if (d <= 1e-6f) return 0.0f;
    float h = 0.0f;
    if (mx == r) h = (g - b) / d + (g < b ? 6.0f : 0.0f);
    else if (mx == g) h = (b - r) / d + 2.0f;
    else h = (r - g) / d + 4.0f;
    h /= 6.0f;
    return h - std::floor(h);
}

static inline Color lerp_color(Color a, Color b, float t) {
    float tt = std::clamp(t, 0.0f, 1.0f);
    auto mix = [&](uint8_t ca, uint8_t cb) -> uint8_t {
        return static_cast<uint8_t>(std::lround(float(ca) * (1.0f - tt) + float(cb) * tt));
    };
    Color out;
    out.r = mix(a.r, b.r);
    out.g = mix(a.g, b.g);
    out.b = mix(a.b, b.b);
    out.a = mix(a.a, b.a);
    return out;
}

static inline Color tint_color_hue(Color base, float hue_shift, float tint_strength) {
    float hue = rgb_to_hue(base);
    Color shifted = hsv_to_color(hue + hue_shift, 1.0f, 1.0f, base.a);
    return lerp_color(base, shifted, tint_strength);
}

// Colored variant: `hues` is an optional array of per-vertex hue values in [0..1]. If null, falls back to neutral drawing.
static void draw_rope_curve_blend_colored(uint8_t* img, int w, int h, int pitch, const float* verts, int count, int jacket_px, int jacket_border, const float* hues, int hue_count, int samples_per_segment, float hue_intensity) {
    if (!hues || hue_count <= 0) {
        // fallback
        Color neutral{200,200,200,200};
        draw_rope_curve_blend(img, w, h, pitch, verts, count, jacket_px, jacket_border, neutral, samples_per_segment);
        return;
    }
    if (!img || !verts || count < 2) return;

    // Build samples along spline (same as non-colored variant)
    auto get = [&](int idx) {
        if (idx < 0) idx = 0;
        if (idx >= count) idx = count - 1;
        return std::pair<float,float>(verts[2*idx+0], verts[2*idx+1]);
    };
    auto catmull = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float t3 = t2 * t;
        float x = 0.5f * ((2.0f * p1.first) + (-p0.first + p2.first) * t + (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t2 + (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t3);
        float y = 0.5f * ((2.0f * p1.second) + (-p0.second + p2.second) * t + (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t2 + (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t3);
        return std::pair<float,float>(x,y);
    };

    // analytic Catmull-Rom derivative (tension 0.5)
    auto catmull_deriv = [&](int i, float t) {
        auto p0 = get(i-1);
        auto p1 = get(i+0);
        auto p2 = get(i+1);
        auto p3 = get(i+2);
        float t2 = t * t;
        float dx = 0.5f * ((-p0.first + p2.first) + 2.0f * (2.0f*p0.first - 5.0f*p1.first + 4.0f*p2.first - p3.first) * t + 3.0f * (-p0.first + 3.0f*p1.first - 3.0f*p2.first + p3.first) * t2);
        float dy = 0.5f * ((-p0.second + p2.second) + 2.0f * (2.0f*p0.second - 5.0f*p1.second + 4.0f*p2.second - p3.second) * t + 3.0f * (-p0.second + 3.0f*p1.second - 3.0f*p2.second + p3.second) * t2);
        return std::pair<float,float>(dx, dy);
    };

    std::vector<std::pair<float,float>> samples;
    std::vector<std::pair<float,float>> tangents;
    samples.reserve((count - 1) * 8);
    tangents.reserve((count - 1) * 8);
    for (int i = 0; i < count - 1; ++i) {
        auto p1 = get(i);
        auto p2 = get(i+1);
        float dx = p2.first - p1.first;
        float dy = p2.second - p1.second;
        float seglen = std::sqrt(dx*dx + dy*dy);
        // build hue samples spacing: ~0.6*jacket_px to align hue samples with visual samples
        float preferred_spacing = std::max(1.0f, float(jacket_px) * 0.6f);
        int n = std::max(2, static_cast<int>(std::ceil(seglen / preferred_spacing)));
        for (int s = 0; s <= n; ++s) {
            float t = float(s) / float(n);
            auto p = catmull(i, t);
            auto d = catmull_deriv(i, t);
            samples.emplace_back(p.first, p.second);
            tangents.emplace_back(d.first, d.second);
        }
    }
    if (samples.empty()) return;

    // build hue samples aligned with `samples` by interpolating provided `hues` across the rope
    std::vector<float> hue_samples;
    hue_samples.reserve(samples.size());
    for (size_t si = 0; si < samples.size(); ++si) {
        float u = float(si) / float(std::max<size_t>(1, samples.size() - 1));
        if (hue_count <= 1) {
            hue_samples.push_back(hues[0]);
            continue;
        }
        float v = u * float(hue_count - 1);
        int idx = static_cast<int>(std::floor(v));
        idx = std::clamp(idx, 0, hue_count - 2);
        float ft = v - float(idx);
        float h0 = hues[idx];
        float h1 = hues[idx + 1];
        hue_samples.push_back(h0 * (1.0f - ft) + h1 * ft);
    }

    // helper to draw segments (reuse draw_segment_blend lambda from earlier by reimplementing minimal inline)
    auto draw_segment_blend_local = [&](float x1, float y1, float x2, float y2, int radius, Color col) {
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        float rplus = float(radius) + 1.0f;
        int minx = static_cast<int>(std::floor(std::min(x1,x2) - rplus));
        int maxx = static_cast<int>(std::ceil(std::max(x1,x2) + rplus));
        int miny = static_cast<int>(std::floor(std::min(y1,y2) - rplus));
        int maxy = static_cast<int>(std::ceil(std::max(y1,y2) + rplus));
        minx = std::max(minx, 0);
        miny = std::max(miny, 0);
        maxx = std::min(maxx, w - 1);
        maxy = std::min(maxy, h - 1);
        for (int py = miny; py <= maxy; ++py) {
            for (int px = minx; px <= maxx; ++px) {
                float cx = px + 0.5f;
                float cy = py + 0.5f;
                float t = 0.0f;
                if (len2 > 1e-6f) {
                    t = ((cx - x1) * dx + (cy - y1) * dy) / len2;
                    if (t < 0.0f) t = 0.0f;
                    else if (t > 1.0f) t = 1.0f;
                }
                float closestX = x1 + dx * t;
                float closestY = y1 + dy * t;
                float ddx = cx - closestX;
                float ddy = cy - closestY;
                float dist = std::sqrt(ddx*ddx + ddy*ddy);
                if (dist <= rplus) {
                    float coverage = 1.0f - (dist / rplus);
                    uint8_t a = static_cast<uint8_t>(std::lround(float(col.a) * coverage));
                    if (a == 0) continue;
                    Color cc = col;
                    cc.a = a;
                    uint8_t* dstp = img + py * pitch + px * 4;
                    blend_pixel(dstp, cc.r, cc.g, cc.b, cc.a);
                }
            }
        }
    };

    // Draw jacket first (grey base tinted by LED hue via tint_strength)
    int eff_jacket = std::max(1, jacket_px - 1);
    uint8_t jacket_alpha = static_cast<uint8_t>(13);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float hue_mid = 0.5f * (hue_samples[i] + hue_samples[i+1]);
        Color jacket_hue = hsv_to_color(hue_mid, 1.0f, 1.0f, jacket_alpha);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, eff_jacket, jacket_hue, 1.0f, hue_intensity);
    }

    // Draw colored core using hue_samples but keep core faint so cable looks clear
    int core_r = std::max(1, jacket_px - jacket_border - 0);
    for (size_t i = 0; i + 1 < samples.size(); ++i) {
        float hue_mid = 0.5f * (hue_samples[i] + hue_samples[i+1]);
        // keep colored core very faint so the rope interior remains mostly clear (≈5%)
        uint8_t alpha = static_cast<uint8_t>(std::lround(255.0f * std::clamp(hue_intensity, 0.0f, 1.0f) * 0.18f));
        Color hc = hsv_to_color(hue_mid, 1.0f, 1.0f, alpha);
        auto &a = samples[i];
        auto &b = samples[i+1];
        auto &ta = tangents[i];
        auto &tb = tangents[i+1];
        draw_segment_parametric_sdf(img, w, h, pitch, a.first, a.second, b.first, b.second, ta.first, ta.second, tb.first, tb.second, core_r, hc, 1.0f, hue_intensity);
    }
    // note: endpoint rims already drawn by caller if desired
}


GP_TableContext* gp_table_create(const GP_TableStyle* style) {
    std::unique_ptr<GP_TableContext> ctx(new GP_TableContext());
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    ctx->cols.resize(0);
    ctx->rows.resize(0);
    recompute_geom(ctx.get());
    // create rope simulator with reasonable capacities (owned by default)
    int max_ropes = 1024;
    int max_segs = std::max(4, ctx->st.cable_segments);
    ctx->rope_sim = rope_sim_create(max_ropes, max_segs);
    ctx->rope_sim_owned = 1;
    ctx->rope_id_to_sim_idx.clear();
    return ctx.release();
}

void gp_table_destroy(GP_TableContext* ctx) {
    if (!ctx) return;
    if (ctx->rope_sim && ctx->rope_sim_owned) {
        rope_sim_destroy(ctx->rope_sim);
        ctx->rope_sim = nullptr;
    }
    delete ctx;
}

int32_t gp_table_set_style(GP_TableContext* ctx, const GP_TableStyle* style) {
    if (!ctx) return 0;
    if (style) {
        ctx->style_raw = *style;
    } else {
        ctx->style_raw = make_default_style();
    }
    ctx->st = load_style(&ctx->style_raw);
    recompute_geom(ctx);
    for (auto &fifo : ctx->edge_fifos) {
        fifo.set_friction_regions(ctx->st.cable_fifo_friction_regions);
    }
    return 1;
}

int32_t gp_table_set_columns(GP_TableContext* ctx, const GP_TableColumn* cols, int32_t col_count) {
    if (!ctx) return 0;
    if (col_count < 0 || col_count > 8) return 0;
    if (col_count > 0 && !cols) return 0;
    if (col_count == 0) {
        ctx->cols.clear();
    } else {
        ctx->cols.assign(cols, cols + col_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_set_rows(GP_TableContext* ctx, const GP_TableRow* rows, int32_t row_count) {
    if (!ctx) return 0;
    if (row_count < 0) return 0;
    if (row_count > 0 && !rows) return 0;
    if (row_count == 0) {
        ctx->rows.clear();
    } else {
        ctx->rows.assign(rows, rows + row_count);
    }
    recompute_geom(ctx);
    return 1;
}

int32_t gp_table_apply_object_def(GP_TableContext* ctx, const GP_TableObjectDef* def) {
    if (!ctx || !def) return 0;
    if (!gp_table_set_columns(ctx, def->cols, def->col_count)) return 0;
    if (!gp_table_set_rows(ctx, def->rows, def->row_count)) return 0;
    if (!gp_table_set_actions(ctx, def->actions, def->action_count)) return 0;
    return 1;
}

int32_t gp_table_get_geom(const GP_TableContext* ctx, GP_TableGeom* out_geom) {
    if (!ctx || !out_geom) return 0;
    *out_geom = ctx->geom;
    return 1;
}

static bool action_matches(const GP_TableAction& action, const GP_TableHitBox& hb) {
    if (action.row_idx != GP_TABLE_ACTION_ANY && action.row_idx != hb.row_idx) return false;
    if (action.col_idx != GP_TABLE_ACTION_ANY && action.col_idx != hb.col_idx) return false;
    if (action.part != GP_TABLE_ACTION_ANY && action.part != hb.part) return false;
    if (action.aux0 != GP_TABLE_ACTION_ANY && action.aux0 != hb.aux0) return false;
    return true;
}

static bool dispatch_action_list(GP_TableContext* ctx, const GP_TableHitBox& hb, const GP_TableAction* actions, int count) {
    if (!ctx || !ctx->action_callback || !actions || count <= 0) return false;
    bool matched = false;
    for (int i = 0; i < count; ++i) {
        if (action_matches(actions[i], hb)) {
            ctx->action_callback(ctx->action_user, actions[i].action_id, &hb);
            matched = true;
        }
    }
    return matched;
}

static void dispatch_actions(GP_TableContext* ctx, const GP_TableHitBox& hb) {
    if (!ctx || !ctx->action_callback) return;
    if (!ctx->actions.empty()) {
        dispatch_action_list(ctx, hb, ctx->actions.data(), static_cast<int>(ctx->actions.size()));
        return;
    }
    static const GP_TableAction kDefaultActions[] = {
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_EXPAND, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_EXPAND_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_SCROLL_UP, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_SCROLL_UP },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_SCROLL_DOWN, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_SCROLL_DOWN },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_ARG, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE },
        { GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_ANY, GP_TABLE_HIT_LED_TABLE, GP_TABLE_ACTION_ANY, GP_TABLE_ACTION_LED_TOGGLE }
    };
    dispatch_action_list(ctx, hb, kDefaultActions, static_cast<int>(sizeof(kDefaultActions) / sizeof(kDefaultActions[0])));
}

int32_t gp_table_dispatch_hit(GP_TableContext* ctx, const GP_TableHitBox* hit) {
    if (!ctx || !hit) return 0;
    if (ctx->actions.empty()) return 0;
    return dispatch_action_list(ctx, *hit, ctx->actions.data(), static_cast<int>(ctx->actions.size())) ? 1 : 0;
}

int32_t gp_table_on_click(GP_TableContext* ctx, int32_t x, int32_t y, GP_TableHitBox* out_hit) {
    if (!ctx) return 0;
    // Render into a temporary buffer to collect hitboxes
    GP_TableGeom geom{};
    gp_table_get_geom(ctx, &geom);
    const int w = geom.width_px;
    const int h = geom.height_px;
    if (w <= 0 || h <= 0) return 0;
    std::vector<uint8_t> tmp;
    tmp.resize(static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 4);
    const int hitcap = 4096;
    std::vector<GP_TableHitBox> hits(hitcap);
    int written = 0;
    int ok = gp_table_render_rgba_with_state(ctx, nullptr, tmp.data(), static_cast<int32_t>(tmp.size()), &geom, hits.data(), hitcap, &written);
    if (!ok) return 0;
    // find first hit that contains (x,y)
    for (int i = 0; i < written; ++i) {
        const GP_TableHitBox &hb = hits[i];
        if (x >= hb.x0 && x < hb.x1 && y >= hb.y0 && y < hb.y1) {
            // process default actions
            if (out_hit) *out_hit = hb;
            dispatch_actions(ctx, hb);
            // expand toggle
            if (hb.part == GP_TABLE_HIT_EXPAND && hb.row_idx >= 0 && hb.row_idx < static_cast<int>(ctx->rows.size())) {
                ctx->rows[hb.row_idx].expanded = ctx->rows[hb.row_idx].expanded ? 0 : 1;
                recompute_geom(ctx);
                return 1;
            }
            // scroll up/down
            if (hb.part == GP_TABLE_HIT_SCROLL_UP) {
                ctx->scroll_row_offset = std::max(0, ctx->scroll_row_offset - 1);
                return 1;
            }
            if (hb.part == GP_TABLE_HIT_SCROLL_DOWN) {
                ctx->scroll_row_offset = std::min<int>(std::max(0, static_cast<int>(ctx->rows.size()) - 1), ctx->scroll_row_offset + 1);
                return 1;
            }
            // LED click: toggle selection (separate from on/off state)
            if ((hb.part == GP_TABLE_HIT_LED || hb.part == GP_TABLE_HIT_LED_ARG || hb.part == GP_TABLE_HIT_LED_TABLE) && hb.row_idx >= 0 && hb.col_idx >= 0) {
                int r_idx = hb.row_idx;
                int c_idx = hb.col_idx;
                int led = hb.aux0;
                if (led >= 0) {
                    uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(r_idx)) << 32) | (static_cast<uint64_t>(static_cast<uint32_t>(c_idx)) << 16) | static_cast<uint64_t>(static_cast<uint32_t>(led));
                    auto it = ctx->selected_leds.find(key);
                    if (it == ctx->selected_leds.end()) ctx->selected_leds.insert(key);
                    else ctx->selected_leds.erase(it);
                            // If exactly one other LED was selected prior to this click, form an edge.
                            if (ctx->selected_leds.size() == 2) {
                                // grab the two keys
                                auto it2 = ctx->selected_leds.begin();
                                uint64_t k0 = *it2; ++it2; uint64_t k1 = *it2;
                                // add edge if allowed by node-group rules (directional: k0 -> k1)
                                int allowed = gp_table_node_group_is_edge_allowed(ctx, k0, k1);
                                if (allowed == 1) {
                                    // use API helper so relax arrays are kept in sync
                                    gp_table_add_edge(ctx, k0, k1);
                                }
                                // clear selections after attempting to form edge
                                ctx->selected_leds.clear();
                            }
                            return 1;
                }
            }
            // Other parts: no default action, but return hit
            return 1;
        }
    }
    return 0;
}

int32_t gp_table_set_scroll_fraction(GP_TableContext* ctx, float frac) {
    if (!ctx) return 0;
    ctx->scroll_frac = std::clamp(frac, 0.0f, 1.0f);
    // compute row offset based on fraction and available rows
    int visible_rows = std::max(1, ctx->geom.height_px / ctx->st.row_h);
    int total = static_cast<int>(ctx->rows.size());
    int max_off = std::max(0, total - visible_rows);
    ctx->scroll_row_offset = static_cast<int>(std::lround(ctx->scroll_frac * float(max_off)));
    return 1;
}

int32_t gp_table_set_scroll_fraction_xy(GP_TableContext* ctx, float frac_x, float frac_y) {
    if (!ctx) return 0;
    ctx->scroll_frac_x = std::clamp(frac_x, 0.0f, 1.0f);
    return gp_table_set_scroll_fraction(ctx, frac_y);
}

