#include "common/tensors/abstraction/kpath/kpath_armature_graph.h"
#include "common/tensors/abstraction/kpath/kpath_atlas.h"
#include "common/tensors/abstraction/kpath/kpath_fill.h"
#include "common/tensors/abstraction/kpath/kpath_geoglyph.h"
#include "common/tensors/abstraction/kpath/kpath_kinematics.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_solve.h"
#include "common/tensors/abstraction/kpath/kpath_ui_layout.h"
#include "common/tensors/abstraction/kpath/kpath_shaper.h"
#include "common/tensors/abstraction/kpath/kpath_tokenizer.h"
#include "common/tensors/abstraction/kpath/kpath_pipeline.h"
#include "common/tensors/abstraction/kpath/kpath_program.h"
#include "common/tensors/abstraction/kpath/kpath_image_export.h"
#include "common/tensors/abstraction/abstract_tensor.h"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nodus::tensors::kpath;
using nodus::tensors::AbstractTensor;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[KPATH-FACILITIES] FAILED: " << what << "\n";
  return false;
}

static std::string resolve_font_path() {
  if (const char* env = std::getenv("KPATH_TEST_FONT")) {
    std::filesystem::path path(env);
    if (std::filesystem::exists(path)) return path.string();
  }
  const std::vector<std::filesystem::path> candidates = {
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/DejaVuSans.ttf",
    "C:/Windows/Fonts/seguisb.ttf"
  };
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) return candidate.string();
  }
  throw std::runtime_error("No usable font found for kpath layout test");
}

static std::string make_output_path_next_to_exe(const char* argv0, const char* filename) {
  try {
    std::filesystem::path exe_path(argv0);
    if (!exe_path.empty()) {
      return (exe_path.parent_path() / filename).string();
    }
  } catch (...) {
  }
  return std::string(filename);
}

static std::vector<std::string> split_words_ascii(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : s) {
    if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
      continue;
    }
    cur.push_back(ch);
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

static CodepointSequence to_codepoints_ascii(const std::string& s) {
  CodepointSequence seq;
  seq.codepoints.reserve(s.size());
  for (unsigned char ch : s) seq.codepoints.push_back(static_cast<uint32_t>(ch));
  return seq;
}

static GlyphOutline translate_outline(const GlyphOutline& outline, float dx, float dy) {
  GlyphOutline result;
  result.glyph_id = outline.glyph_id;
  result.segments.reserve(outline.segments.size());
  for (const auto& seg : outline.segments) {
    OutlineSegment translated = seg;
    translated.x1 += dx;
    translated.y1 += dy;
    translated.x2 += dx;
    translated.y2 += dy;
    translated.x3 += dx;
    translated.y3 += dy;
    result.segments.push_back(translated);
  }
  return result;
}

static bool log_cluster(const Cluster& cluster, size_t index) {
  std::cout << "[KPATH-FACILITIES] cluster " << index << " glyphs=" << cluster.glyph_ids.size()
            << " advance_x=" << cluster.advance_x << " advance_y=" << cluster.advance_y << '\n';
  return !cluster.glyph_ids.empty();
}

static bool validate_shaper_smoke() {
  std::cout << "[KPATH-FACILITIES] Faculty: font load + HarfBuzz shaping + FreeType outline" << std::endl;

  // Negative test: shaping without a loaded font must fail.
  {
    Shaper shaper;
    CodepointSequence seq;
    seq.codepoints = {static_cast<uint32_t>('A')};
    std::vector<Cluster> clusters;
    if (!require_or_report(!shaper.shape(seq, clusters), "shape() should fail when font not loaded")) return false;
  }

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }

  std::cout << "[KPATH-FACILITIES] Using font: " << font_path << std::endl;
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed")) return false;

  CodepointSequence seq;
  const std::string text = "AV";
  for (unsigned char ch : text) seq.codepoints.push_back(static_cast<uint32_t>(ch));

  std::vector<Cluster> clusters;
  if (!require_or_report(shaper.shape(seq, clusters), "shape() failed")) return false;
  if (!require_or_report(!clusters.empty(), "shape() returned zero clusters")) return false;

  float total_adv = 0.0f;
  size_t glyph_total = 0;
  for (size_t i = 0; i < clusters.size(); ++i) {
    glyph_total += clusters[i].glyph_ids.size();
    total_adv += clusters[i].advance_x;
    if (!require_or_report(log_cluster(clusters[i], i), "cluster contained no glyph ids")) return false;
  }
  if (!require_or_report(glyph_total >= seq.codepoints.size(), "expected >= codepoints glyph count (incl. ligatures/marks)")) return false;
  if (!require_or_report(total_adv > 0.0f, "expected positive total advance_x")) return false;

  // Outline extraction: pull the first glyph we got from HarfBuzz.
  const uint32_t glyph_id = clusters.front().glyph_ids.front();
  std::cout << "[KPATH-FACILITIES] Extracting outline for glyph id=" << glyph_id << std::endl;
  GlyphOutline outline;
  if (!require_or_report(shaper.extract_outline(glyph_id, outline), "extract_outline() failed")) return false;
  if (!require_or_report(outline.glyph_id == glyph_id, "outline.glyph_id mismatch")) return false;
  if (!require_or_report(!outline.segments.empty(), "outline.segments empty")) return false;

  bool has_move = false;
  bool has_draw = false;
  bool has_close = false;
  for (const auto& seg : outline.segments) {
    has_move |= (seg.op == OutlineOp::MoveTo);
    has_draw |= (seg.op == OutlineOp::LineTo) || (seg.op == OutlineOp::QuadTo) || (seg.op == OutlineOp::CubicTo);
    has_close |= (seg.op == OutlineOp::Close);
  }
  if (!require_or_report(has_move, "outline missing MoveTo")) return false;
  if (!require_or_report(has_draw, "outline missing any drawable segments")) return false;
  if (!require_or_report(has_close, "outline missing Close")) return false;

  std::cout << "[KPATH-FACILITIES] OK: shaping + outline extraction validated" << std::endl;
  return true;
}

static bool validate_atlas_from_shaping() {
  std::cout << "[KPATH-FACILITIES] Faculty: glyph atlas generation + validation from shaper output" << std::endl;

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed")) return false;

  CodepointSequence seq;
  const std::string text = "HELLO";
  for (unsigned char ch : text) seq.codepoints.push_back(static_cast<uint32_t>(ch));

  AtlasBuilder atlas_builder;
  TokenLayoutPlan plan;
  if (!require_or_report(build_token_from_sequence(atlas_builder, shaper, seq, plan),
                         "build_token_from_sequence() failed")) {
    return false;
  }
  Atlas atlas = atlas_builder.finalize();

  if (!require_or_report(!plan.edges.empty(), "plan edges empty")) return false;

  MetricSchema schema = MetricSchema::MakeCoreV01();
  StepTape tape;
  if (!require_or_report(compile_token_to_tape(atlas, plan, schema, tape),
                         "compile_token_to_tape() failed")) {
    return false;
  }

  auto axis0 = tape.axis_delta(0);
  auto axis1 = tape.axis_delta(1);
  if (!require_or_report(axis0.size() == plan.edges.size(), "axis count mismatch")) return false;
  for (size_t i = 0; i < plan.edges.size(); ++i) {
    const AtlasEdge& e = atlas.edge(plan.edges[i]);
    if (!require_or_report(axis0[i] == e.advance_x, "axis0 mismatch")) return false;
    if (!require_or_report(axis1[i] == e.advance_y, "axis1 mismatch")) return false;
  }

  std::cout << "[KPATH-FACILITIES] OK: atlas built+validated from shaping" << std::endl;
  return true;
}

static bool validate_armature_graph_facilities() {
  std::cout << "[KPATH-FACILITIES] Faculty: armature graph generation + validation" << std::endl;

  ArmatureGraphBuilder builder;

  ArmatureJointInfo root_joint;
  root_joint.kind = JointKind::Revolute;
  root_joint.length = 1.0f;
  NodeId root_id = builder.add_joint(root_joint);

  ArmatureJointInfo child_joint;
  child_joint.kind = JointKind::Prismatic;
  child_joint.length = 0.5f;
  NodeId child_id = builder.add_joint(child_joint);

  ArmaturePort from{root_id, 0, Vec3{0.0f, 0.0f, 0.0f}};
  ArmaturePort to{child_id, 0, Vec3{0.0f, 0.0f, 0.0f}};
  ArmatureConstraint constraint;
  constraint.type = ConstraintType::Pivot;
  constraint.stiffness = 10.0f;
  EdgeId edge_id = builder.add_constraint(from, to, constraint);

  std::vector<EdgeId> token_edges{edge_id};
  TokenId token = builder.add_token(std::span<const EdgeId>(token_edges.data(), token_edges.size()));
  builder.add_posting(edge_id, Posting{token, 0, 1});

  ArmatureGraph graph = builder.finalize();
  if (!require_or_report(graph.joint_info(root_id) != nullptr, "root joint info missing")) return false;
  if (!require_or_report(graph.joint_info(child_id) != nullptr, "child joint info missing")) return false;
  if (!require_or_report(graph.constraint_info(edge_id) != nullptr, "constraint info missing")) return false;
  if (!require_or_report(graph.atlas.token_edges(token).size() == 1, "armature graph atlas token_edges mismatch")) return false;

  const ArmatureConstraint* cinfo = graph.constraint_info(edge_id);
  if (!require_or_report(cinfo->type == ConstraintType::Pivot, "constraint type mismatch")) return false;
  if (!require_or_report(cinfo->stiffness == 10.0f, "constraint stiffness mismatch")) return false;

  std::cout << "[KPATH-FACILITIES] OK: armature graph built+validated" << std::endl;
  return true;
}

static bool validate_raster_png_facility(const char* argv0) {
  std::cout << "[KPATH-FACILITIES] Faculty: armature program -> Gaussian tensor raster -> PNG output" << std::endl;

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed")) return false;

  CodepointSequence seq;
  const std::string text = "S";
  for (unsigned char ch : text) seq.codepoints.push_back(static_cast<uint32_t>(ch));

  std::vector<Cluster> clusters;
  if (!require_or_report(shaper.shape(seq, clusters), "shape() failed")) return false;
  if (!require_or_report(!clusters.empty(), "shape() returned zero clusters")) return false;
  if (!require_or_report(!clusters.front().glyph_ids.empty(), "first cluster has no glyph ids")) return false;

  const uint32_t glyph_id = clusters.front().glyph_ids.front();
  GlyphOutline outline;
  if (!require_or_report(shaper.extract_outline(glyph_id, outline), "extract_outline() failed")) return false;
  if (!require_or_report(!outline.segments.empty(), "outline.segments empty")) return false;

  ArmatureProgram program = glyph_outline_to_armature_program(outline, /*nominal_z=*/0.0f, /*samples_per_segment=*/24);
  if (!require_or_report(!program.points.empty(), "ArmatureProgram contained no sampled points")) return false;

  TensorCanvas2D energy;
  TensorCanvas2D temp;
  MachineControlConfig machine;
  machine.step_px = 0.75f;
  machine.energy_per_px = 1.25f;
  machine.enable_thermal_guard = true;
  machine.feed_rate_px_per_s = 500.0f;
  machine.cooling_tau_s = 0.20f;
  machine.energy_to_temp = 0.75f;
  machine.max_temp = 4.0f;

  GaussianToolParams tool;

  const float render_scale = 4.0f; // pixels per outline unit
  const float margin_px = 12.0f;
  ProgramRasterTransform xform = plan_program_raster_transform_refined(program, machine, render_scale, margin_px, tool);
  const uint32_t max_keyframes = 100;
  uint32_t last_bucket = std::numeric_limits<uint32_t>::max();
  uint32_t saved_frames = 0;
  rasterize_program_gaussian_with_thermal_transformed_keyframes(
      program,
      energy,
      temp,
      machine,
      tool,
      xform,
      [&](uint32_t frame_index, uint32_t frame_total, const TensorCanvas2D& frame_energy, const TensorCanvas2D& /*frame_temp*/) {
        if (frame_total == 0) return;
        const uint32_t bucket = static_cast<uint32_t>((static_cast<uint64_t>(frame_index) * max_keyframes) / frame_total);
        if (bucket == last_bucket) return;
        last_bucket = bucket;

        std::ostringstream name;
        name << "kpath_glyph_gaussian_frame_" << std::setw(3) << std::setfill('0') << saved_frames << ".png";
        const std::string name_str = name.str();
        const std::string frame_path = make_output_path_next_to_exe(argv0, name_str.c_str());
        AbstractTensor frame_tensor = make_image_tensor_from_canvas(frame_energy);
        export_tensor_png(frame_tensor, frame_path, true);
        ++saved_frames;
      });
  float max_v = energy.max_value();
  if (!require_or_report(max_v > 0.001f, "rasterized canvas should have non-zero energy")) return false;

  wait_for_pending_image_saves();
  std::cout << "[KPATH-FACILITIES] wrote " << saved_frames << " keyframes (max=" << max_v << ", temp_max=" << temp.max_value() << ")" << std::endl;
  return true;
}

static bool validate_pangram_atlas_and_png(const char* argv0) {
  std::cout << "[KPATH-FACILITIES] Faculty: pangram atlas (per-word tokens) + PNG render" << std::endl;

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed")) return false;

  const std::string pangram = "the quick brown fox jumps over the lazy dog";
  const auto words = split_words_ascii(pangram);
  if (!require_or_report(!words.empty(), "pangram words empty")) return false;

  AtlasBuilder atlas_builder;
  ArmatureProgram program;
  std::vector<GlyphOutline> pangram_outlines;

  // Explicit page layout (in outline units), rendered with a fixed mapping so
  // additional shapes do not change the text scale.
  const float page_origin_x = 20.0f;
  const float page_origin_y = 32.0f;
  const float page_scale = 2.2f; // outline-units -> pixels
  const float page_margin_px = 14.0f;

  // Layout in font space: we treat advance_x as x translation.
  float pen_x = page_origin_x;
  float pen_y = page_origin_y;
  const float word_spacing = 8.0f;

  // Build each word as its own token: a list of edges for its clusters.
  for (size_t w = 0; w < words.size(); ++w) {
    CodepointSequence seq = to_codepoints_ascii(words[w]);
    std::vector<Cluster> clusters;
    if (!require_or_report(shaper.shape(seq, clusters), "shape() failed for word")) return false;
    if (!require_or_report(!clusters.empty(), "shape() returned zero clusters for word")) return false;

    std::vector<EdgeId> token_edges;
    token_edges.reserve(clusters.size());

    float word_advance = 0.0f;
    float local_x = pen_x;

    for (size_t i = 0; i < clusters.size(); ++i) {
      const Cluster& cl = clusters[i];

      const uint32_t cp = seq.codepoints[std::min(cl.start, seq.codepoints.size() - 1)];
      NodeId cp_node = atlas_builder.add_node(AtlasNode{AtlasNodeKind::Codepoint, cp, AtlasNode::kNoMetadata});

      const uint32_t glyph_id = cl.glyph_ids.front();
    GlyphOutline outline;
    if (!require_or_report(shaper.extract_outline(glyph_id, outline), "extract_outline() failed while building pangram atlas")) return false;

      // Metadata per glyph.
      NodeMetadata meta;
      meta.outline = outline;
      meta.winding = RotDir::Zero;
      const uint32_t meta_idx = atlas_builder.add_node_metadata(std::move(meta));

      NodeId glyph_node = atlas_builder.add_node(AtlasNode{AtlasNodeKind::Glyph, glyph_id, meta_idx});

      AtlasEdge e;
      e.src = cp_node;
      e.dst = glyph_node;
      e.label = 1;
      e.advance_x = cl.advance_x;
      e.advance_y = cl.advance_y;
      e.offset_x = 0.0f;
      e.offset_y = 0.0f;
      e.cluster_id = static_cast<uint32_t>(cl.start);
      EdgeId edge_id = atlas_builder.add_edge(e);
      token_edges.push_back(edge_id);

      // Add this glyph outline into the program at the current pen position.
    append_glyph_outline_to_program(program, outline, local_x, pen_y, 0.0f, /*samples*/16);
    pangram_outlines.push_back(translate_outline(outline, local_x, pen_y));

      // Advance: for clusters that contain multiple glyphs, distribute advance crudely.
      float per_glyph_adv = cl.advance_x;
      if (cl.glyph_ids.size() > 0) per_glyph_adv = cl.advance_x / static_cast<float>(cl.glyph_ids.size());
      local_x += per_glyph_adv;
      word_advance += per_glyph_adv;
    }

    TokenId token = atlas_builder.add_token(std::span<const EdgeId>(token_edges.data(), token_edges.size()));
    for (size_t i = 0; i < token_edges.size(); ++i) {
      atlas_builder.add_posting(token_edges[i], Posting{token, static_cast<uint32_t>(i), 1});
    }

    std::cout << "[KPATH-FACILITIES] token(word)='" << words[w] << "' edges=" << token_edges.size() << " adv=" << word_advance << "\n";
    pen_x = local_x + word_spacing;
  }

  // Inject geometric PUA glyphs (no font dependency) into the atlas/program.
  {
    const float glyph_line_y = pen_y + 34.0f;
    float gx = page_origin_x;
    for (uint32_t cp : default_geoglyph_codepoints()) {
      GeoGlyph g;
      if (!require_or_report(make_geoglyph(cp, /*size*/26.0f, g), "make_geoglyph() failed")) return false;

      TokenLayoutPlan plan;
      if (!require_or_report(build_codepoint_glyph_token(atlas_builder, cp, g.outline, g.advance_x, g.advance_y, plan),
                             "build_codepoint_glyph_token() failed")) {
        return false;
      }

      append_glyph_outline_to_program(program, g.outline, gx, glyph_line_y, 0.0f, /*samples*/16);
      pangram_outlines.push_back(translate_outline(g.outline, gx, glyph_line_y));
      gx += g.advance_x;
    }
  }

  {
    const float shape_line_y = pen_y + 118.0f;
    const float shape_spacing = 78.0f;
    float shape_x = page_origin_x;
    RelGlyph square_shape = make_rel_square_panel(0xE011u, 28.0f);
    if (!require_or_report(add_relglyph_shape(atlas_builder, program, pangram_outlines, square_shape, shape_x, shape_line_y, "param square"),
                           "param square glyph build failed")) {
      return false;
    }
    shape_x += shape_spacing;
    // A straight-sided hex (line midpoint) and a wavy-sided hex (sin midpoint).
    RelGlyph hex_straight = make_rel_hex_panel(0xE012u, 22.0f, /*waviness_amplitude*/0.0f);
    if (!require_or_report(add_relglyph_shape(atlas_builder, program, pangram_outlines, hex_straight, shape_x, shape_line_y, "param hex line"),
                           "param hex line glyph build failed")) {
      return false;
    }
    shape_x += shape_spacing;
    RelGlyph hex_wavy = make_rel_hex_panel(0xE013u, 22.0f, /*waviness_amplitude*/22.0f * 0.08f);
    if (!require_or_report(add_relglyph_shape(atlas_builder, program, pangram_outlines, hex_wavy, shape_x, shape_line_y, "param hex sin"),
                           "param hex sin glyph build failed")) {
      return false;
    }
  }

  // Inject a relational-geometry glyph: equilateral triangle built from circle intersections.
  {
    const uint32_t cp = 0xE010u; // PUA

    RelGlyph rg;
    rg.codepoint = cp;
    rg.glyph_id = 0xF2000000u | (cp & 0x0000FFFFu);
    rg.advance_y = 0.0f;

    const float s = 26.0f;
    RelPointId A = rg.program.add_point(RelPointFixed{0.0f, 0.0f});
    RelPointId B = rg.program.add_point(RelPointFixed{s, 0.0f});

    RelCircle ca;
    ca.center = A;
    ca.radius = RelRadiusDistance{A, B};
    RelCircle cb;
    cb.center = B;
    cb.radius = RelRadiusDistance{A, B};

    RelPointId C = rg.program.add_point(RelPointCircleCircleIntersection{ca, cb, RelPick::HigherY});
    rg.program.add_contour({A, B, C}, /*closed*/true, RelContourWinding::CCW);
    rg.advance_x = s + 0.15f * s;

    GlyphOutline outline;
    std::string err;
    if (!require_or_report(compile_relglyph_outline(rg, outline, 0.0f, 0.0f, 1.0f, &err), err.c_str())) return false;

    TokenLayoutPlan plan;
    if (!require_or_report(build_codepoint_glyph_token(atlas_builder, cp, outline, rg.advance_x, rg.advance_y, plan),
                           "build_codepoint_glyph_token() failed for relglyph")) {
      return false;
    }

    const float glyph_line_y = pen_y + 78.0f;
    const float gx = page_origin_x;
    append_glyph_outline_to_program(program, outline, gx, glyph_line_y, 0.0f, /*samples*/16);
    pangram_outlines.push_back(translate_outline(outline, gx, glyph_line_y));
  }

  // Inject a pure-IR + constraint-solved relational-geometry glyph: a panel diagram constructed
  // from midlines, parallels, and intersections ("word as geometry diagram" demo).
  {
    const uint32_t cp = 0xE014u; // PUA

    RelGlyph rg;
    rg.codepoint = cp;
    rg.glyph_id = 0xF2000000u | (cp & 0x0000FFFFu);
    rg.advance_y = 0.0f;

    const float s = 28.0f;
    // A square, its diagonals, and an interior "frame" built from constrained parallels.
    // Notes:
    // - free() points provide symbolic degrees of freedom.
    // - parallel() constraints orient helper lines; llint() turns them into constructed points.
    const std::string ir =
      "a = pt(0, 0);\n"
      "b = pt(" + std::to_string(s) + ", 0);\n"
      "c = pt(" + std::to_string(s) + ", " + std::to_string(s) + ");\n"
      "d = pt(0, " + std::to_string(s) + ");\n"
      "half = ratio(1, 2);\n"
      "mab = lerp_ratio(a, b, half);\n"
      "mbc = lerp_ratio(b, c, half);\n"
      "mcd = lerp_ratio(c, d, half);\n"
      "mda = lerp_ratio(d, a, half);\n"
      "lab = line(a, b);\n"
      "lbc = line(b, c);\n"
      "lac = line(a, c);\n"
      "lbd = line(b, d);\n"
      "o = llint(lac, lbd);\n"
      "u1 = free();\n"
      "u2 = free();\n"
      "u3 = free();\n"
      "u4 = free();\n"
      "lv0 = line(mab, u1);\n"
      "parallel(lv0, lbc);\n"
      "lh0 = line(mbc, u2);\n"
      "parallel(lh0, lab);\n"
      "p0 = llint(lv0, lh0);\n"
      "lv1 = line(mcd, u3);\n"
      "parallel(lv1, lbc);\n"
      "lh1 = line(mda, u4);\n"
      "parallel(lh1, lab);\n"
      "p1 = llint(lv1, lh1);\n"
      // Outer border (closed), plus interior construction strokes (open).
      "contour(a, b, c, d, \"closed\", \"ccw\");\n"
      "contour(a, c, \"open\");\n"
      "contour(b, d, \"open\");\n"
      "contour(mab, mcd, \"open\");\n"
      "contour(mbc, mda, \"open\");\n"
      "contour(p0, o, p1, \"open\");\n";

    RelGeoSolveInputs sin;
    sin.eps = 1e-4f;
    sin.max_passes = 64;

    std::string solve_err;
    if (!require_or_report(relgeo_program_from_pure_ir_solved(ir, sin, rg.program, &solve_err), solve_err.c_str())) return false;
    rg.advance_x = s + 0.25f * s;

    GlyphOutline outline;
    std::string err;
    if (!require_or_report(compile_relglyph_outline(rg, outline, 0.0f, 0.0f, 1.0f, &err), err.c_str())) return false;

    TokenLayoutPlan plan;
    if (!require_or_report(build_codepoint_glyph_token(atlas_builder, cp, outline, rg.advance_x, rg.advance_y, plan),
                           "build_codepoint_glyph_token() failed for pure-ir relglyph")) {
      return false;
    }

    const float glyph_line_y = pen_y + 78.0f;
    const float gx = page_origin_x + 92.0f; // place next to the triangle demo
    append_glyph_outline_to_program(program, outline, gx, glyph_line_y, 0.0f, /*samples*/16);
    pangram_outlines.push_back(translate_outline(outline, gx, glyph_line_y));
  }

  {
    const std::vector<std::string> panel_words = {"kpath", "geometry", "layout"};
    const float panel_x = page_origin_x + 360.0f;
    const float panel_y = pen_y + 12.0f;
    const float panel_size = 90.0f;
    if (!require_or_report(append_ui_panel_with_text(atlas_builder, program, pangram_outlines, shaper, panel_words, panel_x, panel_y, panel_size),
                           "UI panel text layout failed")) {
      return false;
    }
  }
  Atlas atlas = atlas_builder.finalize();
  if (!require_or_report(atlas.node(NodeId{0}).kind == AtlasNodeKind::Codepoint, "atlas sanity check (node0 kind)")) return false;
  if (!require_or_report(atlas.node(NodeId{0}).payload == static_cast<uint32_t>('t'), "atlas sanity check (node0 payload)")) return false;

  // Render the composed atlas program.
  TensorCanvas2D energy;
  TensorCanvas2D temp;

  MachineControlConfig machine;
  machine.step_px = 0.65f;
  machine.energy_per_px = 1.0f;
  machine.enable_thermal_guard = true;
  machine.feed_rate_px_per_s = 650.0f;
  machine.cooling_tau_s = 0.18f;
  machine.energy_to_temp = 0.65f;
  machine.max_temp = 4.0f;

  GaussianToolParams tool;

  ProgramRasterTransform page_xform = plan_program_raster_transform_refined(program, machine, page_scale, page_margin_px, tool);
  rasterize_program_gaussian_with_thermal_transformed(program, energy, temp, machine, tool, page_xform);
  if (!require_or_report(energy.max_value() > 0.001f, "pangram atlas raster energy should be non-zero")) return false;

  const std::string out_path = make_output_path_next_to_exe(argv0, "kpath_atlas_pangram.png");
  AbstractTensor energy_image = make_image_tensor_from_canvas(energy);
  if (!require_or_report(export_tensor_png(energy_image, out_path, true), "export_tensor_png energy failed")) return false;

  const std::string temp_path = make_output_path_next_to_exe(argv0, "kpath_atlas_pangram_temp.png");
  AbstractTensor temp_image = make_image_tensor_from_canvas(temp);
  require_or_report(export_tensor_png(temp_image, temp_path, true), "export_tensor_png temp failed");

  std::cout << "[KPATH-FACILITIES] wrote PNG: " << out_path << " (energy_max=" << energy.max_value() << ")\n";
  std::cout << "[KPATH-FACILITIES] wrote PNG: " << temp_path << " (temp_max=" << temp.max_value() << ")\n";

  // Optional: demonstrate a basic filled-toolpath plan for one glyph.
  std::cout << "[KPATH-FACILITIES] Faculty: fill planning (even-odd hatch)" << std::endl;
  {
    if (pangram_outlines.empty()) {
      std::cerr << "[KPATH-FACILITIES] FAILED: no outlines captured for pangram fill\n";
      return false;
    }

    ArmatureProgram fill_prog;
    FillPlanConfig fill_cfg;
    fill_cfg.rule = FillRule::EvenOdd;
    fill_cfg.pattern = FillPattern::Hatch;

    ToolCalibration cal = calibrate_gaussian_tool_impulse(128, tool);
    float r_val_05 = cal.radius_at_value_fraction(0.05f);
    float r_cum_95 = cal.radius_at_cumulative_fraction(0.95f);
    std::cout << "[KPATH-FACILITIES] tool calibration: sigma_px=" << tool.sigma_px
          << " peak=" << cal.peak_value
          << " r@5%=" << r_val_05
          << " r@95%energy=" << r_cum_95 << "\n";

    float eff_radius_px = std::max(r_val_05, 1.0f);
    float eff_radius_outline = eff_radius_px / std::max(page_scale, 1e-6f);
    fill_cfg.tool.tool_width = 2.0f * eff_radius_outline;
    fill_cfg.tool.stepover = 0.0f;
    fill_cfg.tool.overlap = 0.85f;
    fill_cfg.tool.angle_degrees = 25.0f;
    fill_cfg.safe_z = machine.safe_z;
    fill_cfg.cut_z = machine.cut_z;
    fill_cfg.kerf = KerfMode::Center;

    bool filled_any = false;
    for (const auto& outline : pangram_outlines) {
      ArmatureProgram glyph_fill;
      if (plan_fill_for_glyph_outline(outline, glyph_fill, fill_cfg)) {
        fill_prog.points.insert(fill_prog.points.end(), glyph_fill.points.begin(), glyph_fill.points.end());
        filled_any = true;
      }
    }
    if (!require_or_report(filled_any, "fill plan generation failed")) return false;

    TensorCanvas2D fill_energy;
    TensorCanvas2D fill_temp;
    rasterize_program_gaussian_with_thermal_transformed(fill_prog, fill_energy, fill_temp, machine, tool, page_xform);
    if (!require_or_report(fill_energy.max_value() > 0.001f, "fill raster energy should be non-zero")) return false;

    const std::string fill_path = make_output_path_next_to_exe(argv0, "kpath_pangram_fill.png");
    AbstractTensor fill_image = make_image_tensor_from_canvas(fill_energy);
    if (!require_or_report(export_tensor_png(fill_image, fill_path, true), "export_tensor_png fill failed")) return 1;
    std::cout << "[KPATH-FACILITIES] wrote PNG: " << fill_path << " (energy_max=" << fill_energy.max_value() << ")\n";

    struct Pt2 {
      float x = 0.0f;
      float y = 0.0f;
    };

    auto area2 = [](const std::vector<Pt2>& loop) -> float {
      if (loop.size() < 3) return 0.0f;
      float a = 0.0f;
      for (size_t i = 1; i < loop.size(); ++i) {
        const Pt2& p0 = loop[i - 1];
        const Pt2& p1 = loop[i];
        a += p0.x * p1.y - p1.x * p0.y;
      }
      return a;
    };

    auto emit_loop = [](ArmatureProgram& prog, const std::vector<Pt2>& loop, float safe_z, float cut_z) {
      if (loop.size() < 2) return;
      ToolPoint rapid;
      rapid.x = loop.front().x;
      rapid.y = loop.front().y;
      rapid.z = safe_z;
      rapid.engaged = false;
      prog.points.push_back(rapid);

      ToolPoint plunge = rapid;
      plunge.z = cut_z;
      plunge.engaged = true;
      prog.points.push_back(plunge);

      for (size_t i = 1; i < loop.size(); ++i) {
        ToolPoint p;
        p.x = loop[i].x;
        p.y = loop[i].y;
        p.z = cut_z;
        p.engaged = true;
        prog.points.push_back(p);
      }

      ToolPoint lift = prog.points.back();
      lift.z = safe_z;
      lift.engaged = false;
      prog.points.push_back(lift);
    };

    auto flatten_loops = [](const GlyphOutline& outline) -> std::vector<std::vector<Pt2>> {
      std::vector<std::vector<Pt2>> loops;
      std::vector<Pt2> cur_loop;
      Pt2 cur{0, 0};
      Pt2 start{0, 0};
      bool have_loop = false;

      auto push = [&](Pt2 p) {
        cur_loop.push_back(p);
        cur = p;
      };

      constexpr uint32_t steps = 20;
      for (const auto& seg : outline.segments) {
        switch (seg.op) {
          case OutlineOp::MoveTo: {
            if (!cur_loop.empty()) {
              loops.push_back(cur_loop);
              cur_loop.clear();
            }
            cur = Pt2{seg.x1, seg.y1};
            start = cur;
            have_loop = true;
            push(cur);
            break;
          }
          case OutlineOp::LineTo: {
            push(Pt2{seg.x3, seg.y3});
            break;
          }
          case OutlineOp::QuadTo: {
            Pt2 p0{seg.x1, seg.y1};
            Pt2 p1{seg.x2, seg.y2};
            Pt2 p2{seg.x3, seg.y3};
            for (uint32_t i = 1; i <= steps; ++i) {
              float t = static_cast<float>(i) / static_cast<float>(steps);
              float a = 1.0f - t;
              Pt2 p;
              p.x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
              p.y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
              push(p);
            }
            break;
          }
          case OutlineOp::CubicTo: {
            Pt2 p0{cur.x, cur.y};
            Pt2 p1{seg.x1, seg.y1};
            Pt2 p2{seg.x2, seg.y2};
            Pt2 p3{seg.x3, seg.y3};
            for (uint32_t i = 1; i <= steps; ++i) {
              float t = static_cast<float>(i) / static_cast<float>(steps);
              float a = 1.0f - t;
              Pt2 p;
              p.x = a * a * a * p0.x + 3.0f * a * a * t * p1.x + 3.0f * a * t * t * p2.x + t * t * t * p3.x;
              p.y = a * a * a * p0.y + 3.0f * a * a * t * p1.y + 3.0f * a * t * t * p2.y + t * t * t * p3.y;
              push(p);
            }
            break;
          }
          case OutlineOp::Arc: {
            const float cx = seg.x1;
            const float cy = seg.y1;
            const float r = std::max(seg.x2, 1e-6f);
            const float start = seg.y2;
            const float sweep = seg.x3;
            uint32_t arc_steps = std::max<uint32_t>(4, static_cast<uint32_t>(std::ceil(std::fabs(sweep) / (3.14159265f / 8.0f))));
            arc_steps = std::max<uint32_t>(arc_steps, steps);
            for (uint32_t i = 1; i <= arc_steps; ++i) {
              float t = static_cast<float>(i) / static_cast<float>(arc_steps);
              float ang = start + sweep * t;
              Pt2 p{cx + r * std::cos(ang), cy + r * std::sin(ang)};
              push(p);
            }
            break;
          }
          case OutlineOp::Sin: {
            Pt2 start_pt{cur.x, cur.y};
            Pt2 end_pt{seg.x1, seg.y1};
            float dx = end_pt.x - start_pt.x;
            float dy = end_pt.y - start_pt.y;
            float len = std::max(std::sqrt(dx * dx + dy * dy), 1e-6f);
            float amp = seg.x2;
            float cycles = seg.y2;
            float phase = seg.x3;
            uint32_t sin_steps = std::max<uint32_t>(steps, static_cast<uint32_t>(std::ceil(len / 0.75f) + std::fabs(cycles) * 4.0f));
            float tx_dir = dx / len;
            float ty_dir = dy / len;
            float nx_dir = -ty_dir;
            float ny_dir = tx_dir;
            for (uint32_t i = 1; i <= sin_steps; ++i) {
              float t = static_cast<float>(i) / static_cast<float>(sin_steps);
              float base_x = start_pt.x + dx * t;
              float base_y = start_pt.y + dy * t;
              float s = std::sin((2.0f * 3.14159265f * cycles * t) + phase);
              Pt2 p{base_x + nx_dir * amp * s, base_y + ny_dir * amp * s};
              push(p);
            }
            break;
          }
          case OutlineOp::Close: {
            if (have_loop) {
              if (cur_loop.size() >= 2) {
                Pt2 last = cur_loop.back();
                if (std::fabs(last.x - start.x) > 1e-6f || std::fabs(last.y - start.y) > 1e-6f) {
                  cur_loop.push_back(start);
                }
              }
              loops.push_back(cur_loop);
              cur_loop.clear();
              have_loop = false;
            }
            break;
          }
        }
      }
      if (!cur_loop.empty()) loops.push_back(cur_loop);
      return loops;
    };

    ArmatureProgram outline_cw;
    ArmatureProgram outline_ccw;
    for (const auto& outline : pangram_outlines) {
      auto loops = flatten_loops(outline);
      for (const auto& loop : loops) {
        float a = area2(loop);
        if (a < 0.0f) emit_loop(outline_cw, loop, machine.safe_z, machine.cut_z);
        else emit_loop(outline_ccw, loop, machine.safe_z, machine.cut_z);
      }
    }

    ArmatureProgram master;
    master.points.reserve(outline_cw.points.size() + outline_ccw.points.size() + fill_prog.points.size());
    master.points.insert(master.points.end(), outline_cw.points.begin(), outline_cw.points.end());
    master.points.insert(master.points.end(), outline_ccw.points.begin(), outline_ccw.points.end());
    master.points.insert(master.points.end(), fill_prog.points.begin(), fill_prog.points.end());

    ProgramRasterTransform xform = plan_program_raster_transform_refined(master, machine, page_scale, page_margin_px, tool);

    TensorCanvas2D ch_r;
    TensorCanvas2D ch_g;
    TensorCanvas2D ch_b;
    TensorCanvas2D tmp;
    rasterize_program_gaussian_with_thermal_transformed(outline_cw, ch_r, tmp, machine, tool, xform);
    rasterize_program_gaussian_with_thermal_transformed(fill_prog, ch_g, tmp, machine, tool, xform);
    rasterize_program_gaussian_with_thermal_transformed(outline_ccw, ch_b, tmp, machine, tool, xform);

    const std::string rgb_path = make_output_path_next_to_exe(argv0, "kpath_pangram_fill_rgb.png");
    AbstractTensor rgb_image = make_image_tensor_from_canvases_rgb(ch_r, ch_g, ch_b);
    if (!require_or_report(export_tensor_png(rgb_image, rgb_path, true),
                 "export_tensor_png failed")) return 1;
    std::cout << "[KPATH-FACILITIES] wrote PNG: " << rgb_path << " (R=CW outline, G=fill, B=CCW outline)\n";

    ArmatureProgram fill_inside;
    ArmatureProgram fill_outside;
    ArmatureProgram contour_inside;
    ArmatureProgram contour_center;
    ArmatureProgram contour_outside;
    OffsetContourBuffer contour_registry;
    BeamProgram beam_prog;
    {
      beam_prog.beams.reserve(fill_prog.points.size());
      const float standoff = 20.0f;
      for (const auto& p : fill_prog.points) {
        BeamPoint b;
        b.dx = 0.0f; b.dy = 0.0f; b.dz = -1.0f;
        b.ox = p.x;
        b.oy = p.y;
        b.oz = standoff;
        b.engaged = p.engaged;
        beam_prog.beams.push_back(b);
      }
    }
    ArmatureProgram gimbal_surface;
    (void)project_beam_program_to_plane(beam_prog, /*plane_z=*/0.0f, gimbal_surface);
    {
      FillPlanConfig cfg_in = fill_cfg;
      cfg_in.kerf = KerfMode::Inside;
      for (const auto& outline : pangram_outlines) {
        (void)plan_fill_for_glyph_outline(outline, fill_inside, cfg_in);
      }

      FillPlanConfig cfg_out = fill_cfg;
      cfg_out.kerf = KerfMode::Outside;
      for (const auto& outline : pangram_outlines) {
        (void)plan_fill_for_glyph_outline(outline, fill_outside, cfg_out);
      }
    }

    contour_registry.tool_width = fill_cfg.tool.tool_width;
    contour_registry.calibration_scale = eff_radius_outline;
    contour_registry.finishing_allowance = fill_cfg.tool.tool_width * 0.25f;

    OffsetContourOptions contour_opts;
    contour_opts.miter_limit = fill_cfg.tool.offset_miter_limit;
    contour_opts.lead_in_length = eff_radius_outline * 0.65f;
    contour_opts.lead_out_length = eff_radius_outline * 0.65f;
    contour_opts.lead_sweep_degrees = 22.5f;
    contour_opts.capture = &contour_registry;
    contour_opts.capture_tool_width = fill_cfg.tool.tool_width;
    contour_opts.capture_calibration_scale = eff_radius_outline;
    contour_opts.capture_finishing_allowance = contour_registry.finishing_allowance;

    for (const auto& outline : pangram_outlines) {
      plan_offset_contour_for_outline(outline, -eff_radius_outline, contour_inside, machine.safe_z, machine.cut_z, contour_opts);
      plan_offset_contour_for_outline(outline, 0.0f, contour_center, machine.safe_z, machine.cut_z, contour_opts);
      plan_offset_contour_for_outline(outline, +eff_radius_outline, contour_outside, machine.safe_z, machine.cut_z, contour_opts);
    }

    if (!require_or_report(!contour_registry.loops.empty(), "offset contour registry captured loops")) return false;

    ArmatureProgram master_kerf;
    master_kerf.points.reserve(contour_inside.points.size() + contour_center.points.size() + contour_outside.points.size() + fill_prog.points.size() + gimbal_surface.points.size());
    master_kerf.points.insert(master_kerf.points.end(), contour_inside.points.begin(), contour_inside.points.end());
    master_kerf.points.insert(master_kerf.points.end(), contour_center.points.begin(), contour_center.points.end());
    master_kerf.points.insert(master_kerf.points.end(), contour_outside.points.begin(), contour_outside.points.end());
    master_kerf.points.insert(master_kerf.points.end(), fill_prog.points.begin(), fill_prog.points.end());
    master_kerf.points.insert(master_kerf.points.end(), gimbal_surface.points.begin(), gimbal_surface.points.end());

    ProgramRasterTransform xform_kerf = plan_program_raster_transform_refined(master_kerf, machine, page_scale, page_margin_px, tool);

    TensorCanvas2D k_r;
    TensorCanvas2D k_g;
    TensorCanvas2D k_b;
    rasterize_program_gaussian_with_thermal_transformed(contour_inside, k_r, tmp, machine, tool, xform_kerf);
    rasterize_program_gaussian_with_thermal_transformed(fill_prog, k_g, tmp, machine, tool, xform_kerf);
    rasterize_program_gaussian_with_thermal_transformed(contour_outside, k_b, tmp, machine, tool, xform_kerf);

    const std::string kerf_path = make_output_path_next_to_exe(argv0, "kpath_pangram_fill_kerf_rgb.png");
    AbstractTensor kerf_image = make_image_tensor_from_canvases_rgb(k_r, k_g, k_b);
    if (!require_or_report(export_tensor_png(kerf_image, kerf_path, true),
                 "export_tensor_png kerf failed")) return 1;
    std::cout << "[KPATH-FACILITIES] wrote PNG: " << kerf_path << " (R=inside-contour, G=fill, B=outside-contour)\n";
  }

  return true;
}

static bool validate_program_api_facilities() {
  std::cout << "[KPATH-FACILITIES] Faculty: codepoint->program->tensor pipeline" << std::endl;

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed for program API")) return false;

  const std::string text = "NODUS";
  ProgramBuildParams build_params;
  build_params.samples_per_segment = 12;

  ArmatureProgram utf8_program;
  if (!require_or_report(build_armature_program_from_utf8(shaper, text, utf8_program, build_params),
                         "build_armature_program_from_utf8() failed")) {
    return false;
  }

  CodepointSequence seq = codepoints_from_utf8(text);
  ArmatureProgram seq_program;
  if (!require_or_report(build_armature_program_from_sequence(shaper, seq, seq_program, build_params),
                         "build_armature_program_from_sequence() failed")) {
    return false;
  }

  if (!require_or_report(!utf8_program.points.empty(), "utf8 program empty")) return false;
  if (!require_or_report(!seq_program.points.empty(), "sequence program empty")) return false;
  if (!require_or_report(utf8_program.points.size() == seq_program.points.size(),
                         "utf8/sequence program point counts diverged")) {
    return false;
  }

  ProgramRasterParams raster_params;
  raster_params.margin = 6.0f;
  raster_params.initialization_value = 0.2f;
  raster_params.deposition_value = -1.1f;

  MachineControlConfig machine;
  machine.step_px = 0.5f;
  machine.feed_rate_px_per_s = 550.0f;
  machine.enable_thermal_guard = false;
  GaussianToolParams tool;

  TensorCanvas2D tensor;
  if (!require_or_report(rasterize_program_into_minimal_tensor(utf8_program, tensor, machine, tool, raster_params),
                         "rasterize_program_into_minimal_tensor() failed")) {
    return false;
  }

  if (!require_or_report(tensor.width > 0 && tensor.height > 0, "tensor dimensions empty")) return false;

  bool has_deposition = false;
  for (float v : tensor.values) {
    if (std::fabs(v - raster_params.initialization_value) > 1e-4f) {
      has_deposition = true;
      break;
    }
  }
  if (!require_or_report(has_deposition, "tensor never deviated from initialization")) return false;

  std::cout << "[KPATH-FACILITIES] program tensor dims=" << tensor.width << "x" << tensor.height
            << " init=" << raster_params.initialization_value
            << " deposit=" << raster_params.deposition_value << std::endl;
  return true;
}

static bool validate_word_tokenization_and_external_glyph_token() {
  std::cout << "[KPATH-FACILITIES] Faculty: word-aware tokenization + external glyph token injection" << std::endl;

  Shaper shaper;
  std::string font_path;
  try {
    font_path = resolve_font_path();
  } catch (const std::exception& err) {
    std::cerr << "[KPATH-FACILITIES] FAILED: " << err.what() << std::endl;
    return false;
  }
  if (!require_or_report(shaper.load_font(font_path, 18.0f), "load_font() failed for tokenizer")) return false;

  // Tokenization heuristics: keep internal apostrophes/hyphens.
  const std::string sample = "don't co-operate rock-n-roll l'homme";
  CodepointSequence seq = codepoints_from_utf8(sample);
  TokenizeOptions opts;
  opts.emit_nonword_tokens = false;
  opts.keep_internal_apostrophes = true;
  opts.keep_internal_hyphens = true;

  const auto spans = tokenize_codepoints(seq, opts);
  if (!require_or_report(spans.size() == 4, "expected 4 word tokens from sample string")) return false;

  auto span_contains = [&](const TokenSpan& span, uint32_t cp) -> bool {
    const size_t end = std::min(seq.codepoints.size(), span.start + span.count);
    for (size_t i = span.start; i < end; ++i) {
      if (seq.codepoints[i] == cp) return true;
    }
    return false;
  };
  if (!require_or_report(span_contains(spans[0], static_cast<uint32_t>('\'')), "expected apostrophe to remain inside token")) return false;
  if (!require_or_report(span_contains(spans[1], static_cast<uint32_t>('-')), "expected hyphen to remain inside token")) return false;

  AtlasBuilder atlas_builder;
  std::vector<TokenLayoutPlan> plans;
  if (!require_or_report(build_tokens_from_sequence(atlas_builder, shaper, seq, plans, opts),
                         "build_tokens_from_sequence() failed")) {
    return false;
  }
  if (!require_or_report(plans.size() == 4, "expected 4 atlas token plans")) return false;

  // External glyph token: extract an outline once, then inject it as a standalone token.
  {
    CodepointSequence a_seq = codepoints_from_utf8("A");
    std::vector<Cluster> clusters;
    if (!require_or_report(shaper.shape(a_seq, clusters) && !clusters.empty() && !clusters.front().glyph_ids.empty(),
                           "shape() failed for external glyph injection")) {
      return false;
    }
    const uint32_t glyph_id = clusters.front().glyph_ids.front();
    GlyphOutline outline;
    if (!require_or_report(shaper.extract_outline(glyph_id, outline), "extract_outline() failed for injection")) return false;

    TokenLayoutPlan injected;
    if (!require_or_report(build_external_glyph_token(atlas_builder, outline, injected),
                           "build_external_glyph_token() failed")) {
      return false;
    }
    if (!require_or_report(injected.edges.size() == 1, "external glyph token should have exactly 1 edge")) return false;
  }

  Atlas atlas = atlas_builder.finalize();
  MetricSchema schema = MetricSchema::MakeCoreV01();

  // Compile one of the word tokens and make sure it yields tape.
  {
    StepTape tape;
    if (!require_or_report(compile_token_to_tape(atlas, plans.front(), schema, tape), "compile_token_to_tape() failed for word token")) {
      return false;
    }
    if (!require_or_report(tape.layout().step_count == plans.front().edges.size(), "word token tape length mismatch")) return false;
  }

  std::cout << "[KPATH-FACILITIES] OK: tokenization and external glyph injection validated" << std::endl;
  return true;
}

int main(int argc, char** argv) {
  if (!validate_shaper_smoke()) return 1;
  if (!validate_atlas_from_shaping()) return 1;
  if (!validate_armature_graph_facilities()) return 1;
  if (!validate_raster_png_facility(argc > 0 ? argv[0] : "kpath_toolpath_facilities_test")) return 1;
  if (!validate_pangram_atlas_and_png(argc > 0 ? argv[0] : "kpath_toolpath_facilities_test")) return 1;
  if (!validate_program_api_facilities()) return 1;
  if (!validate_word_tokenization_and_external_glyph_token()) return 1;
  std::cout << "[KPATH-FACILITIES] All faculties validated." << std::endl;
  return 0;
}
