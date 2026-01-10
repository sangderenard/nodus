#include "common/tensors/abstraction/kpath/kpath_atlas.h"
#include "common/tensors/abstraction/kpath/kpath_armature_graph.h"
#include "common/tensors/abstraction/kpath/kpath_kinematics.h"
#include "common/tensors/abstraction/kpath/kpath_schema.h"
#include "common/tensors/abstraction/kpath/kpath_tape.h"
#include "common/tensors/abstraction/kpath/kpath_shaper.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace nodus::tensors::kpath;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "kpath_pipeline_test: failed: " << what << "\n";
  return false;
}

struct AtlasFixture {
  Atlas atlas;
  NodeId codepoint{};
  NodeId glyph{};
  EdgeId layout_edge{};
  TokenId token{};
};

static AtlasFixture build_simple_atlas() {
  AtlasBuilder builder;

  NodeMetadata glyph_meta{};
  glyph_meta.outline.glyph_id = 1337;
  glyph_meta.outline.segments.push_back({OutlineOp::MoveTo, 0, 0, 0, 0, 0, 0});
  glyph_meta.outline.segments.push_back({OutlineOp::LineTo, 12, 0, 12, 12, 0, 0});
  glyph_meta.winding = RotDir::Pos;
  auto meta_idx = builder.add_node_metadata(std::move(glyph_meta));

  AtlasNode cp_node{AtlasNodeKind::Codepoint, 0, AtlasNode::kNoMetadata};
  NodeId cp_id = builder.add_node(cp_node);

  AtlasNode glyph_node{AtlasNodeKind::Glyph, 0, meta_idx};
  NodeId glyph_id = builder.add_node(glyph_node);

  AtlasEdge layout_edge{cp_id, glyph_id, 1, 4.0f, 0.2f, -0.1f, 0.0f, 7};
  EdgeId edge_id = builder.add_edge(layout_edge);

  std::vector<EdgeId> token_edges{edge_id};
  TokenId token = builder.add_token(token_edges);
  builder.add_posting(edge_id, Posting{token, 0, 1});

  Atlas atlas = builder.finalize();
  return AtlasFixture{std::move(atlas), cp_id, glyph_id, edge_id, token};
}

static void log_result(const char* name, bool pass) {
  std::cout << "[kpath] " << name << ": " << (pass ? "pass" : "FAIL") << std::endl;
}

static bool test_atlas_construction() {
  auto fixture = build_simple_atlas();

  auto edges = fixture.atlas.token_edges(fixture.token);
  if (!require_or_report(edges.size() == 1, "token should refer to single edge")) {
    log_result("atlas_construction", false);
    return false;
  }
  auto postings = fixture.atlas.edge_postings(fixture.layout_edge);
  if (!require_or_report(postings.size() == 1, "edge should have one posting")) {
    log_result("atlas_construction", false);
    return false;
  }

  const NodeMetadata* meta = fixture.atlas.node_metadata(fixture.glyph);
  if (!require_or_report(meta != nullptr, "glyph metadata should exist")) {
    log_result("atlas_construction", false);
    return false;
  }
  if (!require_or_report(meta->outline.glyph_id == 1337, "glyph metadata glyph_id expected")) {
    log_result("atlas_construction", false);
    return false;
  }
  if (!require_or_report(meta->winding == RotDir::Pos, "glyph metadata winding expected")) {
    log_result("atlas_construction", false);
    return false;
  }

  bool result = true;
  log_result("atlas_construction", result);
  return result;
}

static bool test_token_edge_features() {
  auto fixture = build_simple_atlas();
  const auto& edge = fixture.atlas.edge(fixture.layout_edge);
  if (!require_or_report(edge.advance_x == 4.0f, "advance_x preserved")) {
    log_result("token_edge_features", false);
    return false;
  }
  if (!require_or_report(edge.offset_x == -0.1f, "offset_x preserved")) {
    log_result("token_edge_features", false);
    return false;
  }
  if (!require_or_report(edge.cluster_id == 7u, "cluster id preserved")) {
    log_result("token_edge_features", false);
    return false;
  }
  bool result = true;
  log_result("token_edge_features", result);
  return result;
}

static bool test_international_layout() {
  AtlasBuilder builder;
  AtlasNode base_cp{AtlasNodeKind::Codepoint, 0, AtlasNode::kNoMetadata};
  NodeId cp0 = builder.add_node(base_cp);
  AtlasNode glyph_a{AtlasNodeKind::Glyph, 0, AtlasNode::kNoMetadata};
  NodeId glyphA = builder.add_node(glyph_a);
  AtlasNode glyph_b{AtlasNodeKind::Glyph, 0, AtlasNode::kNoMetadata};
  NodeId glyphB = builder.add_node(glyph_b);

  AtlasEdge first_edge{cp0, glyphA, 0, 3.0f, 0.0f, 0.0f, 0.0f, 10};
  AtlasEdge second_edge{glyphA, glyphB, 1, 3.5f, 0.1f, 0.0f, 0.0f, 20};
  EdgeId first = builder.add_edge(first_edge);
  EdgeId second = builder.add_edge(second_edge);

  std::vector<EdgeId> token_edges{first, second};
  TokenId token = builder.add_token(token_edges);
  builder.add_posting(first, Posting{token, 0, 1});
  builder.add_posting(second, Posting{token, 1, 1});

  Atlas atlas = builder.finalize();
  auto edges = atlas.token_edges(token);
  if (!require_or_report(edges.size() == 2, "international token should have two edges")) {
    log_result("international_layout", false);
    return false;
  }
  if (!require_or_report(atlas.edge(first).cluster_id == 10, "first cluster id preserved")) {
    log_result("international_layout", false);
    return false;
  }
  if (!require_or_report(atlas.edge(second).cluster_id == 20, "second cluster id preserved")) {
    log_result("international_layout", false);
    return false;
  }
  bool result = true;
  log_result("international_layout", result);
  return result;
}

static bool test_font_expression_to_tool_path_ir() {
  MetricSchema schema;
  ChannelDef dt_def{};
  dt_def.name = "dt";
  dt_def.kind = ChannelKind::Kinematic;
  dt_def.dtype = ChannelDType::F32;
  dt_def.scope = ChannelScope::PerStep;
  dt_def.interp = ChannelInterp::Stepped;
  schema.add_channel(dt_def);

  StepTapeBuilder builder(schema, 2, 3);
  builder.set_axis_delta(0, 0, 1.0f);
  builder.set_axis_delta(0, 1, 0.5f);
  builder.set_axis_delta(1, 1, 2.0f);
  if (!builder.set_channel_f32(0, 0, 0.125f)) return false;
  if (!builder.set_channel_f32(0, 1, 0.250f)) return false;

  StepProvenance prov{};
  prov.token = TokenId{42};
  prov.token_edge_offset = 2;
  prov.glyph_id = 7;
  prov.contour_id = 1;
  builder.set_provenance(1, prov);

  StepTape tape = builder.finalize();
  auto axis0 = tape.axis_delta(0);
  if (!require_or_report(axis0[0] == 1.0f, "axis delta 0 step 0 must match")) {
    log_result("font_expression_to_tool_path_ir", false);
    return false;
  }
  if (!require_or_report(axis0[1] == 0.5f, "axis delta 0 step 1 must match")) {
    log_result("font_expression_to_tool_path_ir", false);
    return false;
  }
  auto channel = tape.ch_f32(0);
  if (!require_or_report(channel[1] == 0.250f, "channel value recorded")) {
    log_result("font_expression_to_tool_path_ir", false);
    return false;
  }
  const auto& recorded = tape.provenance()[1];
  if (!require_or_report(recorded.token == TokenId{42}, "provenance token preserved")) {
    log_result("font_expression_to_tool_path_ir", false);
    return false;
  }
  if (!require_or_report(recorded.contour_id == 1, "provenance contour preserved")) {
    log_result("font_expression_to_tool_path_ir", false);
    return false;
  }
  bool result = true;
  log_result("font_expression_to_tool_path_ir", result);
  return result;
}

static bool test_kinetic_configuration_tree() {
  ArmatureModel model;
  model.joints = {
      Joint{JointKind::Prismatic, Vec3{1, 0, 0}, RotDir::Pos, -1, {}, 0.0f, 1.0f, {}},
      Joint{JointKind::Revolute, Vec3{0, 1, 0}, RotDir::Pos, 0, {}, -1.0f, 1.0f, {}},
      Joint{JointKind::Revolute, Vec3{0, 0, 1}, RotDir::Pos, 0, {}, -1.0f, 1.0f, {}},
  };

  ArmatureConfigurationTree tree = ArmatureConfigurationTree::Build(model);
  if (!require_or_report(tree.nodes.size() == 3, "tree should contain three nodes")) {
    log_result("kinetic_configuration_tree", false);
    return false;
  }
  auto roots = tree.root_indices();
  if (!require_or_report(roots.size() == 1 && roots[0] == 0, "root should be joint 0")) {
    log_result("kinetic_configuration_tree", false);
    return false;
  }
  auto order = tree.traversal_order();
  if (!require_or_report(order.size() == 3, "traversal should visit three nodes")) {
    log_result("kinetic_configuration_tree", false);
    return false;
  }
  bool result = true;
  log_result("kinetic_configuration_tree", result);
  return result;
}

static bool test_solution_set_and_traversal() {
  ArmatureModel model;
  model.joints = {
      Joint{JointKind::Prismatic, Vec3{1, 0, 0}, RotDir::Pos, -1, {}, 0.0f, 1.0f, {}},
      Joint{JointKind::Revolute, Vec3{0, 1, 0}, RotDir::Pos, 0, {}, -1.0f, 1.0f, {}},
  };
  ArmatureSolver solver(model);

  std::vector<std::vector<float>> seeds = {{0.0f, 0.1f}, {1.0f, 2.0f}};
  ToolPathSolution solution = analyze_armature(model, solver, seeds);
  if (!require_or_report(solution.size() == seeds.size(), "solution set size matches seeds")) {
    log_result("solution_set_and_traversal", false);
    return false;
  }

  const ArmatureConfiguration* picked = pick_configuration(solution, 1);
  if (!require_or_report(picked != nullptr, "pick_configuration should find entry")) {
    log_result("solution_set_and_traversal", false);
    return false;
  }
  if (!require_or_report(picked->axis_mask == 0x3u, "axis mask expected")) {
    log_result("solution_set_and_traversal", false);
    return false;
  }

  const ArmatureConfiguration* by_mask = find_configuration_by_axis_mask(solution, 0x3u);
  if (!require_or_report(by_mask != nullptr, "find by axis mask should succeed")) {
    log_result("solution_set_and_traversal", false);
    return false;
  }
  bool result = true;
  log_result("solution_set_and_traversal", result);
  return result;
}

static bool test_joint_mass_properties() {
  JointMetrics metrics;
  metrics.mass = 2.0f;
  metrics.com_offset = Vec3{1.0f, 0.0f, 0.0f};
  metrics.inertia = InertiaTensor{1.0f, 2.0f, 3.0f, 0.0f, 0.0f, 0.0f};

  Vec3 torque = joint_torque(metrics, Vec3{0.0f, 1.0f, 0.0f});
  if (!require_or_report(torque.z == 1.0f, "torque should be r x F")) {
    log_result("joint_mass_properties", false);
    return false;
  }

  float inertia_x = joint_inertia_about_axis(metrics, Vec3{1.0f, 0.0f, 0.0f});
  if (!require_or_report(inertia_x == 1.0f, "inertia about x axis should match tensor")) {
    log_result("joint_mass_properties", false);
    return false;
  }

  float inertia_z = joint_inertia_about_axis(metrics, Vec3{0.0f, 0.0f, 1.0f});
  if (!require_or_report(inertia_z == 5.0f, "parallel axis term should apply")) {
    log_result("joint_mass_properties", false);
    return false;
  }

  bool result = true;
  log_result("joint_mass_properties", result);
  return result;
}

static bool test_step_tape_named_channels() {
  MetricSchema schema = MetricSchema::MakeCoreV01();
  StepTapeBuilder builder(schema, 1, 2);
  bool ok = builder.set_channel_f32("dt", 0, 0.5f) &&
            builder.set_channel_f32("dt", 1, 0.75f) &&
            builder.set_channel_u32("interp_mode", 0, static_cast<uint32_t>(InterpMode::Linear));
  if (!ok) {
    log_result("step_tape_named_channels", false);
    return false;
  }
  StepTape tape = builder.finalize();
  auto channel = tape.ch_f32(0);
  bool result = std::fabs(channel[0] - 0.5f) < 1e-6f && std::fabs(channel[1] - 0.75f) < 1e-6f;
  log_result("step_tape_named_channels", result);
  return result;
}

static bool test_armature_graph_builder() {
  ArmatureGraphBuilder builder;
  ArmatureJointInfo joint{};
  joint.kind = JointKind::Revolute;
  joint.mass = 1.2f;
  joint.axis = Vec3{0, 0, 1};
  NodeId joint_node = builder.add_joint(joint);

  ArmaturePort from{joint_node, 0, Vec3{0, 0, 0}};
  ArmaturePort to{joint_node, 1, Vec3{1, 0, 0}};
  ArmatureConstraint constraint{};
  constraint.type = ConstraintType::Pivot;
  EdgeId edge = builder.add_constraint(from, to, constraint);
  builder.add_posting(edge, Posting{TokenId{5}, 0, 1});

  ArmatureGraph graph = builder.finalize();
  bool ok = graph.joint_info(joint_node) && graph.constraint_info(edge);
  if (!ok) {
    log_result("armature_graph_builder", false);
    return false;
  }
  log_result("armature_graph_builder", true);
  return true;
}

int main() {
  bool ok = true;
  ok &= test_atlas_construction();
  ok &= test_token_edge_features();
  ok &= test_international_layout();
  ok &= test_font_expression_to_tool_path_ir();
  ok &= test_kinetic_configuration_tree();
  ok &= test_solution_set_and_traversal();
  ok &= test_joint_mass_properties();
  ok &= test_step_tape_named_channels();
  ok &= test_armature_graph_builder();
  return ok ? 0 : 1;
}
