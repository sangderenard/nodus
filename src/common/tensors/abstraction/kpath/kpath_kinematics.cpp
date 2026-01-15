#include "common/tensors/abstraction/kpath/kpath_kinematics.h"
#include "common/tensors/abstraction/kpath/kpath_raster.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_math.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <algorithm>
#include <cmath>

namespace nodus::tensors::kpath {

namespace {

AbstractTensor make_f32_tensor_1d(TensorBackend* backend, const float* data, uint32_t count) {
  if (!backend || !data || count == 0) return {};
  TensorDesc desc{};
  desc.dtype = TensorDType::F32;
  desc.layout = TensorLayout::Dense;
  desc.shape.dims = {count};
  AbstractTensor out = AbstractTensor::create(desc, backend);
  if (!out.valid()) return {};
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return {};
  void* ptr_v = nullptr;
  size_t bytes = 0;
  if (!mem->map(out.handle(), &ptr_v, &bytes)) return {};
  float* ptr = static_cast<float*>(ptr_v);
  for (uint32_t i = 0; i < count; ++i) ptr[i] = data[i];
  mem->unmap(out.handle());
  return out;
}

AbstractTensor make_vec3_tensor(TensorBackend* backend, const Vec3& v) {
  const float data[3] = {v.x, v.y, v.z};
  return make_f32_tensor_1d(backend, data, 3u);
}

AbstractTensor make_quat_tensor(TensorBackend* backend, const Quat& q) {
  const float data[4] = {q.w, q.x, q.y, q.z};
  return make_f32_tensor_1d(backend, data, 4u);
}

bool copy_mat4_to(InMemoryBackend* mem, const AbstractTensor& mat, float* dst16) {
  if (!mem || !dst16 || !mat.valid()) return false;
  const TensorDesc& desc = mat.desc();
  if (desc.dtype != TensorDType::F32 || desc.layout != TensorLayout::Dense) return false;
  if (desc.shape.dims.size() != 2 || desc.shape.dims[0] != 4 || desc.shape.dims[1] != 4) return false;
  void* ptr_v = nullptr;
  size_t bytes = 0;
  if (!mem->map(mat.handle(), &ptr_v, &bytes)) return false;
  const float* src = static_cast<const float*>(ptr_v);
  for (int i = 0; i < 16; ++i) dst16[i] = src[i];
  mem->unmap(mat.handle());
  return true;
}

Quat quat_conj(const Quat& q) {
  return Quat{q.w, -q.x, -q.y, -q.z};
}

Quat quat_mul(const Quat& a, const Quat& b) {
  return Quat{
      a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
      a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
      a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
      a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

Vec3 quat_rotate(const Quat& q, const Vec3& v) {
  Quat vq{0.0f, v.x, v.y, v.z};
  Quat qi = quat_conj(q);
  Quat r = quat_mul(quat_mul(q, vq), qi);
  return Vec3{r.x, r.y, r.z};
}

} // namespace

ArmatureConfigurationTree ArmatureConfigurationTree::Build(const ArmatureModel& model) {
  ArmatureConfigurationTree tree;
  tree.nodes.resize(model.joints.size());
  for (size_t i = 0; i < model.joints.size(); ++i) {
    tree.nodes[i].joint_index = static_cast<int32_t>(i);
    tree.nodes[i].parent = model.joints[i].parent;
  }
  for (size_t i = 0; i < tree.nodes.size(); ++i) {
    int32_t parent = tree.nodes[i].parent;
    if (parent >= 0 && parent < static_cast<int32_t>(tree.nodes.size())) {
      tree.nodes[parent].children.push_back(static_cast<int32_t>(i));
    }
  }
  return tree;
}

std::vector<int32_t> ArmatureConfigurationTree::root_indices() const {
  std::vector<int32_t> roots;
  for (size_t i = 0; i < nodes.size(); ++i)
    if (nodes[i].parent < 0) roots.push_back(static_cast<int32_t>(i));
  return roots;
}

std::vector<int32_t> ArmatureConfigurationTree::traversal_order() const {
  std::vector<int32_t> order;
  std::vector<int32_t> stack = root_indices();
  while (!stack.empty()) {
    int32_t node = stack.back();
    stack.pop_back();
    order.push_back(node);
    for (int32_t child : nodes[node].children) stack.push_back(child);
  }
  return order;
}

ToolPathSolution analyze_armature(const ArmatureModel& model,
                                  const ArmatureSolver& solver,
                                  const std::vector<std::vector<float>>& seeds) {
  ToolPathSolution solution;
  solution.configs.reserve(seeds.size());
  for (const auto& q : seeds) {
    ArmatureConfiguration config;
    config.q = q;
    config.pose = solver.solve(q.empty() ? nullptr : q.data(),
                               static_cast<uint32_t>(q.size()));
    if (!q.empty()) {
      size_t bits = std::min<size_t>(q.size(), 32);
      if (bits == 32) {
        config.axis_mask = 0xFFFFFFFFu;
      } else {
        config.axis_mask = (1u << static_cast<uint32_t>(bits)) - 1u;
      }
    }
    solution.configs.push_back(std::move(config));
  }
  return solution;
}

const ArmatureConfiguration* pick_configuration(const ToolPathSolution& solution, size_t idx) {
  if (idx >= solution.configs.size()) return nullptr;
  return &solution.configs[idx];
}

const ArmatureConfiguration* find_configuration_by_axis_mask(const ToolPathSolution& solution,
                                                              uint32_t mask) {
  for (const auto& cfg : solution.configs)
    if (cfg.axis_mask == mask) return &cfg;
  return nullptr;
}

float vec3_dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 vec3_cross(const Vec3& a, const Vec3& b) {
  return Vec3{
      a.y * b.z - a.z * b.y,
      a.z * b.x - a.x * b.z,
      a.x * b.y - a.y * b.x,
  };
}

Vec3 vec3_scale(const Vec3& v, float s) {
  return Vec3{v.x * s, v.y * s, v.z * s};
}

Vec3 vec3_add(const Vec3& a, const Vec3& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

float vec3_len_sq(const Vec3& v) {
  return vec3_dot(v, v);
}

Vec3 torque_from_force(const Vec3& lever, const Vec3& force) {
  return vec3_cross(lever, force);
}

Vec3 joint_torque(const JointMetrics& metrics, const Vec3& force) {
  return torque_from_force(metrics.com_offset, force);
}

float joint_inertia_about_axis(const JointMetrics& metrics, const Vec3& axis_unit) {
  float axis_len_sq = vec3_len_sq(axis_unit);
  if (axis_len_sq <= 0.0f) return 0.0f;
  float inv_len = 1.0f / std::sqrt(axis_len_sq);
  Vec3 a{axis_unit.x * inv_len, axis_unit.y * inv_len, axis_unit.z * inv_len};

  const InertiaTensor& I = metrics.inertia;
  float I_com =
      a.x * (I.xx * a.x + I.xy * a.y + I.xz * a.z) +
      a.y * (I.xy * a.x + I.yy * a.y + I.yz * a.z) +
      a.z * (I.xz * a.x + I.yz * a.y + I.zz * a.z);

  float r2 = vec3_len_sq(metrics.com_offset);
  float ar = vec3_dot(a, metrics.com_offset);
  float parallel_axis = metrics.mass * (r2 - ar * ar);
  return I_com + parallel_axis;
}

bool solve_armature_batch_dense(const ArmatureModel& model,
                                const AbstractTensor& q,
                                BatchedArmatureKinematicsOutput* out,
                                bool emit_joint_transforms,
                                TensorBackend* backend_override) {
  if (!out) return false;
  out->tool_transforms.reset();
  out->joint_transforms.reset();

  if (!q.valid()) return false;
  const TensorDesc& qd = q.desc();
  if (qd.dtype != TensorDType::F32) return false;
  if (qd.shape.rank() != 2) return false;
  const uint32_t batch = qd.shape.dims[0];
  const uint32_t joints = qd.shape.dims[1];
  if (joints != model.joints.size()) return false;

  TensorBackend* backend = backend_override ? backend_override : q.backend();
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem) return false;

  TensorDesc tool_desc;
  tool_desc.dtype = TensorDType::F32;
  tool_desc.shape.dims = {batch, 4u, 4u};
  tool_desc.layout = TensorLayout::Dense;
  out->tool_transforms = AbstractTensor::create(tool_desc, backend);
  if (!out->tool_transforms.valid()) return false;

  if (emit_joint_transforms) {
    TensorDesc joint_desc;
    joint_desc.dtype = TensorDType::F32;
    joint_desc.shape.dims = {batch, joints, 4u, 4u};
    joint_desc.layout = TensorLayout::Dense;
    out->joint_transforms = AbstractTensor::create(joint_desc, backend);
    if (!out->joint_transforms.valid()) return false;
  }

  void* q_ptr_v = nullptr; size_t q_bytes = 0;
  if (!mem->map(q.handle(), &q_ptr_v, &q_bytes)) return false;
  float* q_ptr = static_cast<float*>(q_ptr_v);

  void* tool_ptr_v = nullptr; size_t tool_bytes = 0;
  if (!mem->map(out->tool_transforms.handle(), &tool_ptr_v, &tool_bytes)) {
    mem->unmap(q.handle());
    return false;
  }
  float* tool_ptr = static_cast<float*>(tool_ptr_v);

  void* joint_ptr_v = nullptr; size_t joint_bytes = 0;
  float* joint_ptr = nullptr;
  if (emit_joint_transforms) {
    if (!mem->map(out->joint_transforms.handle(), &joint_ptr_v, &joint_bytes)) {
      mem->unmap(q.handle());
      mem->unmap(out->tool_transforms.handle());
      return false;
    }
    joint_ptr = static_cast<float*>(joint_ptr_v);
  }

  std::vector<AbstractTensor> world(model.joints.size());
  for (uint32_t b = 0; b < batch; ++b) {
    AbstractTensor base = tensor_affine_identity_f32(backend);
    if (!base.valid()) {
      mem->unmap(q.handle());
      mem->unmap(out->tool_transforms.handle());
      if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
      return false;
    }
    // Compute per-joint world transforms.
    for (size_t j = 0; j < model.joints.size(); ++j) {
      const Joint& joint = model.joints[j];
      float qval = q_ptr[b * joints + j];

      AbstractTensor q_parent = make_quat_tensor(backend, joint.T_parent_joint.q);
      AbstractTensor t_parent = make_vec3_tensor(backend, joint.T_parent_joint.t);
      AbstractTensor T_parent_joint = tensor_quat_to_mat4_f32(q_parent, t_parent);
      if (!T_parent_joint.valid()) {
        mem->unmap(q.handle());
        mem->unmap(out->tool_transforms.handle());
        if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
        return false;
      }

      AbstractTensor T_motion;
      if (joint.kind == JointKind::Prismatic) {
        Vec3 t{joint.axis_local.x * qval, joint.axis_local.y * qval, joint.axis_local.z * qval};
        AbstractTensor t_motion = make_vec3_tensor(backend, t);
        T_motion = tensor_affine_translation_f32(t_motion);
      } else { // Revolute
        float angle = (joint.dir == RotDir::Neg) ? -qval : qval;
        AbstractTensor axis = make_vec3_tensor(backend, joint.axis_local);
        AbstractTensor q_motion = tensor_quat_from_axis_angle_f32(axis, angle);
        T_motion = tensor_quat_to_mat4_f32(q_motion, AbstractTensor{});
      }

      if (!T_motion.valid()) {
        mem->unmap(q.handle());
        mem->unmap(out->tool_transforms.handle());
        if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
        return false;
      }

      AbstractTensor local = tensor_matmul_f32(T_parent_joint, T_motion);
      if (!local.valid()) {
        mem->unmap(q.handle());
        mem->unmap(out->tool_transforms.handle());
        if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
        return false;
      }
      if (joint.parent >= 0) {
        const AbstractTensor& Pw = world[static_cast<size_t>(joint.parent)];
        world[j] = tensor_matmul_f32(Pw, local);
      } else {
        world[j] = tensor_matmul_f32(base, local);
      }

      if (!world[j].valid()) {
        mem->unmap(q.handle());
        mem->unmap(out->tool_transforms.handle());
        if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
        return false;
      }

      if (emit_joint_transforms && joint_ptr) {
        float* dst = joint_ptr + static_cast<size_t>(b) * joints * 16 + j * 16;
        if (!copy_mat4_to(mem, world[j], dst)) {
          mem->unmap(q.handle());
          mem->unmap(out->tool_transforms.handle());
          mem->unmap(out->joint_transforms.handle());
          return false;
        }
      }
    }

    // Tool = last joint or tool_frame if provided.
    size_t tool_joint = model.tool_frame ? static_cast<size_t>(model.tool_frame.v) : model.joints.size() - 1;
    tool_joint = std::min(tool_joint, model.joints.size() - 1);
    if (!copy_mat4_to(mem, world[tool_joint], tool_ptr + static_cast<size_t>(b) * 16)) {
      mem->unmap(q.handle());
      mem->unmap(out->tool_transforms.handle());
      if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
      return false;
    }
  }

  mem->unmap(q.handle());
  mem->unmap(out->tool_transforms.handle());
  if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
  return true;
}

BeamAimSolution solve_beam_aim_from_pose(const Pose& pose, const ToolMount& mount, bool engaged) {
  BeamAimSolution sol{};
  sol.beam.engaged = engaged;

  const Vec3 origin = vec3_add(pose.t, quat_rotate(pose.q, mount.offset));
  const Quat q_world = quat_mul(pose.q, mount.rotation);
  const Vec3 dir = quat_rotate(q_world, mount.local_axis);
  const float len_sq = vec3_len_sq(dir);
  if (len_sq <= 0.0f) {
    sol.valid = false;
    return sol;
  }

  const float inv_len = 1.0f / std::sqrt(len_sq);
  const Vec3 d{dir.x * inv_len, dir.y * inv_len, dir.z * inv_len};

  sol.beam.ox = origin.x;
  sol.beam.oy = origin.y;
  sol.beam.oz = origin.z;
  sol.beam.dx = d.x;
  sol.beam.dy = d.y;
  sol.beam.dz = d.z;

  // Right-handed, Z-up: yaw around +Z, pitch up/down.
  sol.yaw_rad = std::atan2(d.y, d.x);
  const float xy = std::sqrt(d.x * d.x + d.y * d.y);
  sol.pitch_rad = std::atan2(d.z, xy);
  sol.valid = true;
  return sol;
}

bool intersect_beams_with_plane(const AbstractTensor& origins,
                                 const AbstractTensor& dirs,
                                 float plane_z,
                                 AbstractTensor* out_hits,
                                 AbstractTensor* out_mask) {
  return tensor_intersect_plane_z_f32(origins, dirs, plane_z, out_hits, out_mask);
}

} // namespace nodus::tensors::kpath
