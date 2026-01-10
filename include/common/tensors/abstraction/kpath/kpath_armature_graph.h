#pragma once

#include "kpath_atlas.h"
#include "kpath_kinematics.h"

#include <unordered_map>

namespace nodus::tensors::kpath {

enum class ConstraintType : uint8_t { Rigid = 0, Pivot = 1, Slider = 2, Parallel = 3 };

struct ArmaturePort final {
  NodeId node{};
  uint32_t index{};
  Vec3 offset{};
};

struct ArmatureJointInfo final {
  JointKind kind{JointKind::Revolute};
  Vec3 axis{};
  Vec3 com_offset{};
  float length = 0.0f;
  float mass = 0.0f;
  float torque_limit = 0.0f;
  BeamTheory beam{};
  RotDir direction{RotDir::Pos};
};

struct ArmatureConstraint final {
  ArmaturePort from{};
  ArmaturePort to{};
  ConstraintType type{ConstraintType::Rigid};
  float stiffness = 0.0f;
  float compliance = 0.0f;
  float range_min = 0.0f;
  float range_max = 0.0f;
};

template <typename IdT>
struct IdHash final {
  size_t operator()(IdT id) const noexcept { return static_cast<size_t>(id.v); }
};

struct ArmatureGraph final {
  Atlas atlas;
  std::unordered_map<NodeId, ArmatureJointInfo, IdHash<NodeId>> joints;
  std::unordered_map<EdgeId, ArmatureConstraint, IdHash<EdgeId>> constraints;

  const ArmatureJointInfo* joint_info(NodeId node) const {
    auto it = joints.find(node);
    return it == joints.end() ? nullptr : &it->second;
  }

  const ArmatureConstraint* constraint_info(EdgeId edge) const {
    auto it = constraints.find(edge);
    return it == constraints.end() ? nullptr : &it->second;
  }
};

class ArmatureGraphBuilder final {
 public:
  ArmatureGraphBuilder() = default;

  ArmatureGraphBuilder(const ArmatureGraphBuilder&) = delete;
  ArmatureGraphBuilder& operator=(const ArmatureGraphBuilder&) = delete;

  NodeId add_joint(const ArmatureJointInfo& info) {
    AtlasNode node{AtlasNodeKind::ArmatureJoint, 0, AtlasNode::kNoMetadata};
    NodeId id = atlas_builder_.add_node(node);
    joint_info_.emplace(id, info);
    return id;
  }

  EdgeId add_constraint(const ArmaturePort& from,
                        const ArmaturePort& to,
                        ArmatureConstraint constraint) {
    AtlasEdge edge{from.node, to.node, static_cast<uint32_t>(constraint.type), 0.0f, 0.0f, 0.0f, 0.0f, 0, 0, 0};
    EdgeId id = atlas_builder_.add_edge(edge);
    constraint.from = from;
    constraint.to = to;
    constraints_.emplace(id, std::move(constraint));
    return id;
  }

  void add_posting(EdgeId e, Posting p) {
    atlas_builder_.add_posting(e, std::move(p));
  }

  TokenId add_token(std::span<const EdgeId> edges) {
    return atlas_builder_.add_token(edges);
  }

  ArmatureGraph finalize() {
    ArmatureGraph graph;
    graph.atlas = atlas_builder_.finalize();
    graph.joints = std::move(joint_info_);
    graph.constraints = std::move(constraints_);
    return graph;
  }

 private:
  AtlasBuilder atlas_builder_;
  std::unordered_map<NodeId, ArmatureJointInfo, IdHash<NodeId>> joint_info_;
  std::unordered_map<EdgeId, ArmatureConstraint, IdHash<EdgeId>> constraints_;
};

} // namespace nodus::tensors::kpath
