#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_compare.h"
#include "common/tensors/abstraction/tensor_types.h"

#include "common/tensors/abstraction/kpath/kpath_atlas.h"
#include "common/tensors/abstraction/kpath/kpath_fill.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/kpath/kpath_shaper.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <span>
#include <vector>

using namespace nodus::tensors;
using namespace nodus::tensors::kpath;

static void rgb_histogram_u8(const std::vector<uint8_t>& r,
                             const std::vector<uint8_t>& g,
                             const std::vector<uint8_t>& b,
                             std::array<uint32_t, 256>& hr,
                             std::array<uint32_t, 256>& hg,
                             std::array<uint32_t, 256>& hb);

static std::string make_output_path_next_to_exe(const char* argv0, const char* filename) {
  try {
    std::filesystem::path exe_path(argv0);
    if (!exe_path.empty()) return (exe_path.parent_path() / filename).string();
  } catch (...) {
  }
  return std::string(filename);
}

static GlyphOutline translate_outline(const GlyphOutline& outline, float dx, float dy) {
  GlyphOutline out;
  out.glyph_id = outline.glyph_id;
  out.segments.reserve(outline.segments.size());
  for (const auto& seg : outline.segments) {
    OutlineSegment s = seg;
    s.x1 += dx; s.y1 += dy;
    s.x2 += dx; s.y2 += dy;
    s.x3 += dx; s.y3 += dy;
    out.segments.push_back(s);
  }
  return out;
}

static std::string resolve_font_path() {
  if (const char* env = std::getenv("KPATH_TEST_FONT")) {
    std::filesystem::path p(env);
    if (std::filesystem::exists(p)) return p.string();
  }
  const std::vector<std::filesystem::path> candidates = {
      "C:/Windows/Fonts/arial.ttf",
      "C:/Windows/Fonts/DejaVuSans.ttf",
      "C:/Windows/Fonts/seguisb.ttf"
  };
  for (const auto& c : candidates) if (std::filesystem::exists(c)) return c.string();
  throw std::runtime_error("No usable font found for kpath api test");
}

static AbstractTensor make_codepoint_tensor(const std::string& text, TensorBackend* backend) {
  std::vector<uint32_t> cps;
  cps.reserve(text.size());
  for (unsigned char ch : text) cps.push_back(static_cast<uint32_t>(ch));

  TensorDesc desc;
  desc.dtype = TensorDType::U32;
  desc.shape.dims = {static_cast<uint32_t>(cps.size())};
  desc.layout = TensorLayout::Dense;

  return tensor_from_bytes(desc, backend, cps.data(), cps.size() * sizeof(uint32_t));
}

static std::vector<uint32_t> tensor_to_codepoints(const AbstractTensor& t) {
  std::vector<uint32_t> out;
  if (!t.valid()) return out;
  auto* mem = dynamic_cast<InMemoryBackend*>(t.backend());
  if (!mem) return out;
  void* ptr = nullptr; size_t bytes = 0;
  if (!mem->map(t.handle(), &ptr, &bytes)) return out;
  const size_t count = bytes / sizeof(uint32_t);
  out.resize(count);
  std::memcpy(out.data(), ptr, std::min(bytes, count * sizeof(uint32_t)));
  mem->unmap(t.handle());
  return out;
}

static bool make_fill_from_token(const std::string& token,
                                 const Shaper& shaper,
                                 const MachineControlConfig& machine,
                                 const GaussianToolParams& tool,
                                 ArmatureProgram& fill_prog,
                                 TensorCanvas2D& out_energy,
                                 TensorCanvas2D& out_temp) {
  CodepointSequence seq;
  for (unsigned char ch : token) seq.codepoints.push_back(static_cast<uint32_t>(ch));

  std::vector<Cluster> clusters;
  if (!shaper.shape(seq, clusters) || clusters.empty()) return false;

  // Build a token atlas passively while also collecting translated outlines.
  AtlasBuilder atlas;
  std::vector<EdgeId> token_edges;
  std::vector<GlyphOutline> outlines;
  ArmatureProgram outline_program;
  float pen_x = 0.0f;
  for (size_t i = 0; i < clusters.size(); ++i) {
    const Cluster& cl = clusters[i];
    const uint32_t cp = seq.codepoints[std::min(cl.start, seq.codepoints.size() - 1)];
    NodeId cp_node = atlas.add_node(AtlasNode{AtlasNodeKind::Codepoint, cp, AtlasNode::kNoMetadata});

    GlyphOutline outline;
    if (!shaper.extract_outline(cl.glyph_ids.front(), outline)) return false;
    GlyphOutline placed = translate_outline(outline, pen_x, 0.0f);
    outlines.push_back(std::move(placed));
    append_glyph_outline_to_program(outline_program, outlines.back(), 0.0f, 0.0f);

    NodeMetadata meta;
    meta.outline = outlines.back();
    meta.winding = RotDir::Zero;
    uint32_t meta_idx = atlas.add_node_metadata(std::move(meta));
    NodeId glyph_node = atlas.add_node(AtlasNode{AtlasNodeKind::Glyph, cl.glyph_ids.front(), meta_idx});

    AtlasEdge e;
    e.src = cp_node;
    e.dst = glyph_node;
    e.label = 1;
    e.advance_x = cl.advance_x;
    e.advance_y = cl.advance_y;
    e.offset_x = pen_x;
    e.offset_y = 0.0f;
    e.cluster_id = static_cast<uint32_t>(cl.start);
    EdgeId edge_id = atlas.add_edge(e);
    token_edges.push_back(edge_id);

    pen_x += cl.advance_x;
  }

  TokenId tok = atlas.add_token(std::span<const EdgeId>(token_edges.data(), token_edges.size()));
  for (size_t i = 0; i < token_edges.size(); ++i) {
    atlas.add_posting(token_edges[i], Posting{tok, static_cast<uint32_t>(i), 1});
  }
  (void)atlas.finalize(); // passive build; not used further here.

  if (outlines.empty()) return false;

  // Choose a stable render scale for tests and derive outline-space tool width.
  // Sizing is driven from the refined toolpath polyline bounds.
  const float render_scale = 3.0f; // pixels per outline unit
  const float margin_px = 14.0f;

  ToolCalibration cal = calibrate_gaussian_tool_impulse(256, tool);
  float eff_radius_px = std::max(cal.radius_at_value_fraction(0.05f), 1.0f);
  float eff_radius_outline = eff_radius_px / std::max(render_scale, 1e-6f);

  FillPlanConfig fill_cfg;
  fill_cfg.rule = FillRule::EvenOdd;
  fill_cfg.pattern = FillPattern::Hatch;
  fill_cfg.kerf = KerfMode::Center;
  fill_cfg.tool.tool_width = 2.0f * eff_radius_outline;
  fill_cfg.tool.stepover = 0.0f;
  fill_cfg.tool.overlap = 0.85f;
  fill_cfg.tool.angle_degrees = 22.5f;
  fill_cfg.safe_z = machine.safe_z;
  fill_cfg.cut_z = machine.cut_z;

  fill_prog.points.clear();
  bool filled_any = false;
  for (const auto& outline : outlines) {
    ArmatureProgram glyph_fill;
    if (plan_fill_for_glyph_outline(outline, glyph_fill, fill_cfg)) {
      fill_prog.points.insert(fill_prog.points.end(), glyph_fill.points.begin(), glyph_fill.points.end());
      filled_any = true;
    }
  }
  if (!filled_any) return false;

  ArmatureProgram master;
  master.points.reserve(outline_program.points.size() + fill_prog.points.size());
  master.points.insert(master.points.end(), outline_program.points.begin(), outline_program.points.end());
  master.points.insert(master.points.end(), fill_prog.points.begin(), fill_prog.points.end());

  ProgramRasterTransform xform = plan_program_raster_transform_refined(master, machine, render_scale, margin_px, tool);
  out_energy.resize(xform.width_px, xform.height_px, 0.0f);
  out_temp.resize(xform.width_px, xform.height_px, 0.0f);
  rasterize_program_gaussian_with_thermal_transformed(fill_prog, out_energy, out_temp, machine, tool, xform);

  TensorCanvas2D ch_r(xform.width_px, xform.height_px);
  TensorCanvas2D ch_g(xform.width_px, xform.height_px);
  TensorCanvas2D ch_b(xform.width_px, xform.height_px);
  TensorCanvas2D tmp(xform.width_px, xform.height_px);
  rasterize_program_gaussian_with_thermal_transformed(outline_program, ch_r, tmp, machine, tool, xform);
  rasterize_program_gaussian_with_thermal_transformed(fill_prog, ch_g, tmp, machine, tool, xform);
  // Leave B empty for contrast.

  auto r_u8 = ch_r.to_u8_normalized();
  auto g_u8 = ch_g.to_u8_normalized();
  auto b_u8 = ch_b.to_u8_normalized();

  std::array<uint32_t, 256> hr, hg, hb;
  rgb_histogram_u8(r_u8, g_u8, b_u8, hr, hg, hb);

  std::vector<uint8_t> rgb;
  rgb.resize(static_cast<size_t>(ch_r.width) * ch_r.height * 3);
  const size_t px = static_cast<size_t>(ch_r.width) * ch_r.height;
  for (size_t i = 0; i < px; ++i) {
    rgb[3 * i + 0] = (i < r_u8.size()) ? r_u8[i] : 0;
    rgb[3 * i + 1] = (i < g_u8.size()) ? g_u8[i] : 0;
    rgb[3 * i + 2] = (i < b_u8.size()) ? b_u8[i] : 0;
  }

  const std::string out_path = "kpath_api_fill_rgb.png";
  (void)write_png_rgb_u8(out_path, ch_r.width, ch_r.height, rgb);
  std::cout << "[KPATH-API] wrote PNG: " << out_path << "\n";
  auto report_peak = [](const std::array<uint32_t, 256>& h) {
    uint32_t peak_v = 0; uint32_t peak_bin = 0;
    for (uint32_t i = 0; i < 256; ++i) {
      if (h[i] > peak_v) { peak_v = h[i]; peak_bin = i; }
    }
    return peak_bin;
  };
  std::cout << "[KPATH-API] histogram peaks: R=" << report_peak(hr)
            << " G=" << report_peak(hg)
            << " B=" << report_peak(hb) << "\n";

  return out_energy.max_value() > 0.0f;
}

static void rgb_histogram_u8(const std::vector<uint8_t>& r,
                              const std::vector<uint8_t>& g,
                              const std::vector<uint8_t>& b,
                              std::array<uint32_t, 256>& hr,
                              std::array<uint32_t, 256>& hg,
                              std::array<uint32_t, 256>& hb) {
  hr.fill(0); hg.fill(0); hb.fill(0);
  for (uint8_t v : r) ++hr[v];
  for (uint8_t v : g) ++hg[v];
  for (uint8_t v : b) ++hb[v];
}

int main(int argc, char** argv) {
  (void)argc; (void)argv;

  register_in_memory_backend(true);
  TensorBackend* backend = &in_memory_backend_singleton();

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& e) {
    std::cerr << "[KPATH-API] font error: " << e.what() << "\n";
    return 1;
  }
  if (!shaper.load_font(font_path, 18.0f)) {
    std::cerr << "[KPATH-API] load_font failed\n";
    return 1;
  }

  const std::string token_text = "atlas api demo";
  AbstractTensor token_tensor = make_codepoint_tensor(token_text, backend);
  if (!token_tensor.valid()) {
    std::cerr << "[KPATH-API] failed to build codepoint tensor\n";
    return 1;
  }

  std::vector<uint32_t> cps = tensor_to_codepoints(token_tensor);
  if (cps.empty()) {
    std::cerr << "[KPATH-API] empty tensor decode\n";
    return 1;
  }

  MachineControlConfig machine;
  machine.step_px = 0.65f;
  machine.energy_per_px = 1.0f;
  machine.enable_thermal_guard = true;
  machine.feed_rate_px_per_s = 650.0f;
  machine.cooling_tau_s = 0.18f;
  machine.energy_to_temp = 0.65f;
  machine.max_temp = 4.0f;

  GaussianToolParams tool;

  ArmatureProgram fill_prog;
  TensorCanvas2D energy;
  TensorCanvas2D temp;
  if (!make_fill_from_token(token_text, shaper, machine, tool, fill_prog, energy, temp)) {
    std::cerr << "[KPATH-API] failed to build fill from token\n";
    return 1;
  }
  std::cout << "[KPATH-API] fill simulation complete (energy_max=" << energy.max_value()
            << ", temp_max=" << temp.max_value() << ")\n";
  return 0;
}
