#pragma once

#include "kpath_ids.h"

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nodus::tensors::kpath {

struct Vec3 { float x{}, y{}, z{}; };
struct Quat { float w{1.f}, x{}, y{}, z{}; };

struct InertiaTensor {
  float xx = 0.0f;
  float yy = 0.0f;
  float zz = 0.0f;
  float xy = 0.0f;
  float xz = 0.0f;
  float yz = 0.0f;
};

struct Pose final {
  Vec3 t{};
  Quat q{};
};

// Tool-local mount describing where a beam emitter sits relative to the tool frame.
// The mount rotation orients the local_axis into tool space; local_axis defaults
// to -Z to match BeamPoint's conventional dz = -1 direction.
struct ToolMount final {
  Vec3 offset{};
  Quat rotation{};
  Vec3 local_axis{0.0f, 0.0f, -1.0f};
};

// Beam aiming result: world-space beam plus derived gimbal angles (yaw/pitch).
struct BeamAimSolution final {
  BeamPoint beam{};
  float yaw_rad = 0.0f;
  float pitch_rad = 0.0f;
  bool valid = false;
};

struct BeamTheory {
  float youngs_modulus = 0.0f;
  float moment_of_inertia = 0.0f;
  float shear_modulus = 0.0f;
};

struct JointMetrics {
  float mass = 0.0f;
  float mechanical_slack = 0.0f;
  Vec3 com_offset{};
  InertiaTensor inertia{};
  BeamTheory beam{};
};

enum class JointKind : uint8_t { Prismatic, Revolute };

struct Joint final {
  JointKind kind{};
  Vec3 axis_local{};
  RotDir dir{RotDir::Pos};
  int32_t parent{-1};
  Pose T_parent_joint{};
  float q_min{}, q_max{};
  JointMetrics metrics;
};

struct ArmatureModel final {
  std::vector<Joint> joints;
  FrameId base_frame{};
  FrameId tool_frame{};
};

class ArmatureSolver {
 public:
  explicit ArmatureSolver(const ArmatureModel& model) : model_(&model) {}

  // Solver consumes vectorized AbstractTensor-like arrays (eps: placeholder)
  // so it can batch identical armatures and propagate autograd gradients.
  Pose solve(const float* q, uint32_t nq) const {
    // Stub: real kinematics would accumulate transforms along joints.
    Pose pose;
    if (model_ && nq >= model_->joints.size()) {
      pose.t.x = q[0];
      pose.t.y = q[1];
      pose.t.z = nq > 2 ? q[2] : 0.0f;
    }
    return pose;
  }

  void collect_joint_metrics(std::vector<JointMetrics>& out) const {
    if (!model_) return;
    out.clear();
    for (const auto& joint : model_->joints) out.push_back(joint.metrics);
  }

 private:
  const ArmatureModel* model_{};
};

struct BatchedArmatureKinematicsOutput final {
  // [B, 4, 4] world-from-tool transforms.
  AbstractTensor tool_transforms;

  // Optional: [B, J, 4, 4] world-from-joint transforms (J = joint count).
  AbstractTensor joint_transforms;
};

// Solve batched forward kinematics for many identical armatures using an
// AbstractTensor input (shape: [B, J], dtype: F32). Results are written into
// new tensors allocated on the provided backend (or inferred from `q`).
// Returns false on shape/dtype/backend mismatch.
bool solve_armature_batch_dense(const ArmatureModel& model,
                                const AbstractTensor& q,
                                BatchedArmatureKinematicsOutput* out,
                                bool emit_joint_transforms = true,
                                TensorBackend* backend_override = nullptr);

// Computes a beam aim solution from a tool pose and a tool-local mount.
BeamAimSolution solve_beam_aim_from_pose(const Pose& pose, const ToolMount& mount, bool engaged = true);

// Vectorized beam/plane intersection: origins and dirs are [B, 3] F32; dirs
// need not be unit length. Outputs: hits [B, 3] F32 and mask [B] Bool where
// the ray hits plane z = plane_z with t >= 0 and dir_z != 0.
bool intersect_beams_with_plane(const AbstractTensor& origins,
                                 const AbstractTensor& dirs,
                                 float plane_z,
                                 AbstractTensor* out_hits,
                                 AbstractTensor* out_mask);

struct ArmatureConfigurationNode final {
  int32_t joint_index{-1};
  int32_t parent{-1};
  std::vector<int32_t> children;
};

struct ArmatureConfigurationTree final {
  std::vector<ArmatureConfigurationNode> nodes;
  static ArmatureConfigurationTree Build(const ArmatureModel& model);
  std::vector<int32_t> root_indices() const;
  std::vector<int32_t> traversal_order() const;
};

struct ArmatureConfiguration final {
  std::vector<float> q;
  Pose pose;
  uint32_t axis_mask{};
};

struct ToolPathSolution final {
  std::vector<ArmatureConfiguration> configs;
  bool empty() const { return configs.empty(); }
  size_t size() const { return configs.size(); }
};

ToolPathSolution analyze_armature(const ArmatureModel& model,
                                  const ArmatureSolver& solver,
                                  const std::vector<std::vector<float>>& seeds);

const ArmatureConfiguration* pick_configuration(const ToolPathSolution& solution, size_t idx);
const ArmatureConfiguration* find_configuration_by_axis_mask(const ToolPathSolution& solution,
                                                              uint32_t mask);

float vec3_dot(const Vec3& a, const Vec3& b);
Vec3 vec3_cross(const Vec3& a, const Vec3& b);
Vec3 vec3_scale(const Vec3& v, float s);
Vec3 vec3_add(const Vec3& a, const Vec3& b);
float vec3_len_sq(const Vec3& v);

Vec3 torque_from_force(const Vec3& lever, const Vec3& force);
Vec3 joint_torque(const JointMetrics& metrics, const Vec3& force);
float joint_inertia_about_axis(const JointMetrics& metrics, const Vec3& axis_unit);

} // namespace nodus::tensors::kpath
