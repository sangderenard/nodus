#include "common/tensors/abstraction/kpath/kpath_pipeline.h"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace nodus::tensors::kpath {

namespace {

struct Vec2 final {
  float x{};
  float y{};
};

static void sample_quadratic(const OutlineSegment& seg, std::vector<Vec2>& pts, int steps = 6) {
  Vec2 p0{seg.x1, seg.y1};
  Vec2 p1{seg.x2, seg.y2};
  Vec2 p2{seg.x3, seg.y3};
  for (int i = 1; i <= steps; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(steps);
    float a = 1.0f - t;
    Vec2 p;
    p.x = a * a * p0.x + 2.0f * a * t * p1.x + t * t * p2.x;
    p.y = a * a * p0.y + 2.0f * a * t * p1.y + t * t * p2.y;
    pts.push_back(p);
  }
}

static void sample_cubic(const OutlineSegment& seg, std::vector<Vec2>& pts, int steps = 8) {
  Vec2 p0{seg.x1, seg.y1};
  Vec2 p1{seg.x2, seg.y2};
  Vec2 p2{seg.x3, seg.y3};
  Vec2 p3{seg.x3, seg.y3};
  for (int i = 1; i <= steps; ++i) {
    float t = static_cast<float>(i) / static_cast<float>(steps);
    float a = 1.0f - t;
    Vec2 p;
    p.x = a * a * a * p0.x +
          3.0f * a * a * t * p1.x +
          3.0f * a * t * t * p2.x +
          t * t * t * p3.x;
    p.y = a * a * a * p0.y +
          3.0f * a * a * t * p1.y +
          3.0f * a * t * t * p2.y +
          t * t * t * p3.y;
    pts.push_back(p);
  }
}

static RotDir compute_outline_winding(const GlyphOutline& outline) {
  std::vector<Vec2> pts;
  for (const auto& seg : outline.segments) {
    switch (seg.op) {
      case OutlineOp::MoveTo:
        pts.push_back(Vec2{seg.x1, seg.y1});
        break;
      case OutlineOp::LineTo:
        pts.push_back(Vec2{seg.x3, seg.y3});
        break;
      case OutlineOp::QuadTo:
        sample_quadratic(seg, pts);
        break;
      case OutlineOp::CubicTo:
        sample_cubic(seg, pts);
        break;
      case OutlineOp::Close:
        if (!pts.empty()) pts.push_back(pts.front());
        break;
    }
  }
  if (pts.size() < 3) return RotDir::Zero;
  float area = 0.0f;
  for (size_t i = 1; i < pts.size(); ++i) {
    const Vec2& a = pts[i - 1];
    const Vec2& b = pts[i];
    area += a.x * b.y - b.x * a.y;
  }
  if (area > 0.0f) return RotDir::Pos;
  if (area < 0.0f) return RotDir::Neg;
  return RotDir::Zero;
}

} // namespace

static CodepointSequence slice_sequence(const CodepointSequence& seq, size_t start, size_t count) {
  CodepointSequence out;
  if (seq.codepoints.empty() || count == 0) return out;
  const size_t end = std::min(seq.codepoints.size(), start + count);
  if (start >= end) return out;
  out.codepoints.insert(out.codepoints.end(), seq.codepoints.begin() + start, seq.codepoints.begin() + end);
  return out;
}

bool build_token_from_sequence(AtlasBuilder& builder,
                               Shaper& shaper,
                               const CodepointSequence& sequence,
                               TokenLayoutPlan& out_plan) {
  if (sequence.codepoints.empty()) return false;
  std::vector<Cluster> clusters;
  if (!shaper.shape(sequence, clusters) || clusters.empty()) return false;

  std::vector<EdgeId> edges;
  edges.reserve(clusters.size());
  out_plan.advance_x.clear();
  out_plan.advance_y.clear();

  for (size_t i = 0; i < clusters.size(); ++i) {
    const Cluster& cluster = clusters[i];
    uint32_t cp_idx = std::min(cluster.start, sequence.codepoints.size() - 1);
    uint32_t cp = sequence.codepoints[cp_idx];
    NodeId cp_node = builder.add_node(AtlasNode{AtlasNodeKind::Codepoint, cp, AtlasNode::kNoMetadata});

    if (cluster.glyph_ids.empty()) return false;
    uint32_t glyph_id = cluster.glyph_ids.front();
    GlyphOutline outline;
    if (!shaper.extract_outline(glyph_id, outline)) return false;

    NodeMetadata meta;
    meta.outline = outline;
    meta.winding = compute_outline_winding(outline);
    uint32_t meta_idx = builder.add_node_metadata(std::move(meta));

    NodeId glyph_node = builder.add_node(AtlasNode{AtlasNodeKind::Glyph, glyph_id, meta_idx});

    AtlasEdge edge{};
    edge.src = cp_node;
    edge.dst = glyph_node;
    edge.label = 1;
    edge.advance_x = cluster.advance_x;
    edge.advance_y = cluster.advance_y;
    edge.offset_x = 0.0f;
    edge.offset_y = 0.0f;
    edge.cluster_id = static_cast<uint32_t>(cluster.start);
    EdgeId edge_id = builder.add_edge(edge);
    edges.push_back(edge_id);
    out_plan.advance_x.push_back(cluster.advance_x);
    out_plan.advance_y.push_back(cluster.advance_y);
  }

  TokenId token = builder.add_token(std::span<const EdgeId>(edges.data(), edges.size()));
  for (size_t i = 0; i < edges.size(); ++i) {
    builder.add_posting(edges[i], Posting{token, static_cast<uint32_t>(i), 1});
  }

  out_plan.token = token;
  out_plan.edges = std::move(edges);
  return true;
}

bool build_tokens_from_sequence(AtlasBuilder& builder,
                                Shaper& shaper,
                                const CodepointSequence& sequence,
                                std::vector<TokenLayoutPlan>& out_plans,
                                const TokenizeOptions& tokenize_opts) {
  out_plans.clear();
  if (sequence.codepoints.empty()) return false;

  const auto spans = tokenize_codepoints(sequence, tokenize_opts);
  if (spans.empty()) return false;

  out_plans.reserve(spans.size());
  for (const auto& span : spans) {
    if (span.count == 0) continue;
    if (span.kind != TokenKind::Word) continue; // for now, only build word tokens.
    CodepointSequence sub = slice_sequence(sequence, span.start, span.count);
    TokenLayoutPlan plan;
    if (!build_token_from_sequence(builder, shaper, sub, plan)) {
      return false;
    }
    out_plans.push_back(std::move(plan));
  }

  return !out_plans.empty();
}

bool build_external_glyph_token(AtlasBuilder& builder,
                                const GlyphOutline& outline,
                                TokenLayoutPlan& out_plan) {
  if (outline.segments.empty()) return false;

  NodeMetadata meta;
  meta.outline = outline;
  meta.winding = compute_outline_winding(outline);
  const uint32_t meta_idx = builder.add_node_metadata(std::move(meta));

  // Standalone glyph node.
  NodeId glyph_node = builder.add_node(AtlasNode{AtlasNodeKind::Glyph, outline.glyph_id, meta_idx});

  // Represent a "token unto itself" as a single self-edge.
  AtlasEdge edge{};
  edge.src = glyph_node;
  edge.dst = glyph_node;
  edge.label = 0;
  edge.advance_x = 0.0f;
  edge.advance_y = 0.0f;
  edge.offset_x = 0.0f;
  edge.offset_y = 0.0f;
  edge.cluster_id = 0;
  EdgeId edge_id = builder.add_edge(edge);

  const EdgeId edges[] = {edge_id};
  TokenId token = builder.add_token(std::span<const EdgeId>(edges, 1));
  builder.add_posting(edge_id, Posting{token, 0, 1});

  out_plan.token = token;
  out_plan.edges = {edge_id};
  out_plan.advance_x = {0.0f};
  out_plan.advance_y = {0.0f};
  return true;
}

bool build_codepoint_glyph_token(AtlasBuilder& builder,
                                 uint32_t codepoint,
                                 const GlyphOutline& outline,
                                 float advance_x,
                                 float advance_y,
                                 TokenLayoutPlan& out_plan) {
  if (outline.segments.empty()) return false;

  NodeId cp_node = builder.add_node(AtlasNode{AtlasNodeKind::Codepoint, codepoint, AtlasNode::kNoMetadata});

  NodeMetadata meta;
  meta.outline = outline;
  meta.winding = compute_outline_winding(outline);
  const uint32_t meta_idx = builder.add_node_metadata(std::move(meta));
  NodeId glyph_node = builder.add_node(AtlasNode{AtlasNodeKind::Glyph, outline.glyph_id, meta_idx});

  AtlasEdge edge{};
  edge.src = cp_node;
  edge.dst = glyph_node;
  edge.label = 1;
  edge.advance_x = advance_x;
  edge.advance_y = advance_y;
  edge.offset_x = 0.0f;
  edge.offset_y = 0.0f;
  edge.cluster_id = 0;
  const EdgeId edge_id = builder.add_edge(edge);

  const EdgeId edges[] = {edge_id};
  const TokenId token = builder.add_token(std::span<const EdgeId>(edges, 1));
  builder.add_posting(edge_id, Posting{token, 0, 1});

  out_plan.token = token;
  out_plan.edges = {edge_id};
  out_plan.advance_x = {advance_x};
  out_plan.advance_y = {advance_y};
  return true;
}

bool compile_token_to_tape(const Atlas& atlas,
                           const TokenLayoutPlan& plan,
                           const MetricSchema& schema,
                           StepTape& out_tape) {
  if (plan.edges.empty()) return false;

  StepTapeBuilder builder(schema, 2, static_cast<uint32_t>(plan.edges.size()));
  for (uint32_t idx = 0; idx < plan.edges.size(); ++idx) {
    const AtlasEdge& edge = atlas.edge(plan.edges[idx]);
    builder.set_axis_delta(0, idx, edge.advance_x);
    builder.set_axis_delta(1, idx, edge.advance_y);
    builder.set_channel_u32("tool_mode", idx, static_cast<uint32_t>(ToolMode::Travel));
    builder.set_channel_u32("interp_mode", idx, static_cast<uint32_t>(InterpMode::Linear));
    builder.set_channel_u32("path_id", idx, plan.token.v);
    builder.set_channel_u32("contour_id", idx, edge.cluster_id);
    const NodeMetadata* meta = atlas.node_metadata(edge.dst);
    RotDir winding = meta ? meta->winding : RotDir::Zero;
    builder.set_channel_i32("winding_dir", idx, static_cast<int32_t>(winding));
    builder.set_channel_u32("token_id", idx, plan.token.v);
    builder.set_channel_u32("token_edge_offset", idx, idx);

    StepProvenance prov{};
    prov.token = plan.token;
    prov.token_edge_offset = idx;
    prov.glyph_id = atlas.node(edge.dst).payload;
    prov.contour_id = edge.cluster_id;
    builder.set_provenance(idx, prov);
  }

  out_tape = builder.finalize();
  return true;
}

} // namespace nodus::tensors::kpath
