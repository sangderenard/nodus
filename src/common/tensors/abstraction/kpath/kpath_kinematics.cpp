#include "common/tensors/abstraction/kpath/kpath_kinematics.h"

#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <algorithm>
#include <cmath>

namespace nodus::tensors::kpath {

namespace {

constexpr float kEps = 1e-6f;

struct Mat4 {
  float m[16]; // row-major
};

Mat4 mat4_identity() {
  Mat4 M{};
  M.m[0] = M.m[5] = M.m[10] = M.m[15] = 1.0f;
  return M;
}

Mat4 mat4_from_pose(const Pose& p) {
  // Convert quaternion to rotation matrix; quaternion assumed normalized.
  const float w = p.q.w, x = p.q.x, y = p.q.y, z = p.q.z;
  Mat4 M{};
  M.m[0] = 1 - 2 * (y * y + z * z);
  M.m[1] = 2 * (x * y - z * w);
  M.m[2] = 2 * (x * z + y * w);
  M.m[3] = 0.0f;

  M.m[4] = 2 * (x * y + z * w);
  M.m[5] = 1 - 2 * (x * x + z * z);
  M.m[6] = 2 * (y * z - x * w);
  M.m[7] = 0.0f;

  M.m[8] = 2 * (x * z - y * w);
  M.m[9] = 2 * (y * z + x * w);
  M.m[10] = 1 - 2 * (x * x + y * y);
  M.m[11] = 0.0f;

  M.m[12] = p.t.x;
  M.m[13] = p.t.y;
  M.m[14] = p.t.z;
  M.m[15] = 1.0f;
  return M;
}

Mat4 mat4_translate(const Vec3& t) {
  Mat4 M = mat4_identity();
  M.m[12] = t.x;
  M.m[13] = t.y;
  M.m[14] = t.z;
  return M;
}

Mat4 mat4_axis_angle(const Vec3& axis, float angle) {
  float ax = axis.x, ay = axis.y, az = axis.z;
  float len = std::sqrt(ax * ax + ay * ay + az * az);
  if (len < kEps) return mat4_identity();
  float inv = 1.0f / len;
  ax *= inv; ay *= inv; az *= inv;
  float c = std::cos(angle);
  float s = std::sin(angle);
  float t = 1.0f - c;

  Mat4 M{};
  M.m[0] = t * ax * ax + c;
  M.m[1] = t * ax * ay - s * az;
  M.m[2] = t * ax * az + s * ay;
  M.m[3] = 0.0f;

  M.m[4] = t * ay * ax + s * az;
  M.m[5] = t * ay * ay + c;
  M.m[6] = t * ay * az - s * ax;
  M.m[7] = 0.0f;

  M.m[8] = t * az * ax - s * ay;
  M.m[9] = t * az * ay + s * ax;
  M.m[10] = t * az * az + c;
  M.m[11] = 0.0f;

  M.m[12] = 0.0f;
  M.m[13] = 0.0f;
  M.m[14] = 0.0f;
  M.m[15] = 1.0f;
  return M;
}

Mat4 mat4_mul(const Mat4& A, const Mat4& B) {
  Mat4 R{};
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      float v = 0.0f;
      for (int k = 0; k < 4; ++k) v += A.m[4 * r + k] * B.m[4 * k + c];
      R.m[4 * r + c] = v;
    }
  }
  return R;
}

// Writes a Mat4 into a contiguous float buffer at index idx (stride 16).
inline void store_mat4(float* dst16, const Mat4& M) {
  for (int i = 0; i < 16; ++i) dst16[i] = M.m[i];
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

  std::vector<Mat4> world(model.joints.size());
  for (uint32_t b = 0; b < batch; ++b) {
    Mat4 base = mat4_identity();
    // Compute per-joint world transforms.
    for (size_t j = 0; j < model.joints.size(); ++j) {
      const Joint& joint = model.joints[j];
      float qval = q_ptr[b * joints + j];

      Mat4 T_parent_joint = mat4_from_pose(joint.T_parent_joint);
      Mat4 T_motion = mat4_identity();
      if (joint.kind == JointKind::Prismatic) {
        Vec3 t{joint.axis_local.x * qval, joint.axis_local.y * qval, joint.axis_local.z * qval};
        T_motion = mat4_translate(t);
      } else { // Revolute
        float angle = (joint.dir == RotDir::Neg) ? -qval : qval;
        T_motion = mat4_axis_angle(joint.axis_local, angle);
      }

      Mat4 local = mat4_mul(T_parent_joint, T_motion);
      if (joint.parent >= 0) {
        const Mat4& Pw = world[static_cast<size_t>(joint.parent)];
        world[j] = mat4_mul(Pw, local);
      } else {
        world[j] = mat4_mul(base, local);
      }

      if (emit_joint_transforms && joint_ptr) {
        float* dst = joint_ptr + static_cast<size_t>(b) * joints * 16 + j * 16;
        store_mat4(dst, world[j]);
      }
    }

    // Tool = last joint or tool_frame if provided.
    size_t tool_joint = model.tool_frame ? static_cast<size_t>(model.tool_frame.v) : model.joints.size() - 1;
    tool_joint = std::min(tool_joint, model.joints.size() - 1);
    store_mat4(tool_ptr + static_cast<size_t>(b) * 16, world[tool_joint]);
  }

  mem->unmap(q.handle());
  mem->unmap(out->tool_transforms.handle());
  if (emit_joint_transforms && joint_ptr) mem->unmap(out->joint_transforms.handle());
  return true;
}

bool intersect_beams_with_plane(const AbstractTensor& origins,
                                 const AbstractTensor& dirs,
                                 float plane_z,
                                 AbstractTensor* out_hits,
                                 AbstractTensor* out_mask) {
  if (!out_hits || !out_mask) return false;
  out_hits->reset();
  out_mask->reset();

  if (!origins.valid() || !dirs.valid()) return false;
  const TensorDesc& od = origins.desc();
  const TensorDesc& dd = dirs.desc();
  if (od.dtype != TensorDType::F32 || dd.dtype != TensorDType::F32) return false;
  if (od.shape.rank() != 2 || dd.shape.rank() != 2) return false;
  if (od.shape.dims.size() < 2 || dd.shape.dims.size() < 2) return false;
  const uint32_t batch = od.shape.dims[0];
  if (dd.shape.dims[0] != batch) return false;
  if (od.shape.dims[1] != 3 || dd.shape.dims[1] != 3) return false;

  TensorBackend* backend = origins.backend();
  auto* mem = dynamic_cast<InMemoryBackend*>(backend);
  if (!mem || mem != dynamic_cast<InMemoryBackend*>(dirs.backend())) return false;

  TensorDesc hit_desc;
  hit_desc.dtype = TensorDType::F32;
  hit_desc.shape.dims = {batch, 3u};
  hit_desc.layout = TensorLayout::Dense;
  *out_hits = AbstractTensor::create(hit_desc, backend);
  if (!out_hits->valid()) return false;

  TensorDesc mask_desc;
  mask_desc.dtype = TensorDType::Bool;
  mask_desc.shape.dims = {batch};
  mask_desc.layout = TensorLayout::Dense;
  *out_mask = AbstractTensor::create(mask_desc, backend);
  if (!out_mask->valid()) return false;

  void* o_ptr_v = nullptr; size_t o_bytes = 0;
  void* d_ptr_v = nullptr; size_t d_bytes = 0;
  if (!mem->map(origins.handle(), &o_ptr_v, &o_bytes)) return false;
  if (!mem->map(dirs.handle(), &d_ptr_v, &d_bytes)) {
    mem->unmap(origins.handle());
    return false;
  }
  float* o_ptr = static_cast<float*>(o_ptr_v);
  float* d_ptr = static_cast<float*>(d_ptr_v);

  void* h_ptr_v = nullptr; size_t h_bytes = 0;
  void* m_ptr_v = nullptr; size_t m_bytes = 0;
  if (!mem->map(out_hits->handle(), &h_ptr_v, &h_bytes)) {
    mem->unmap(origins.handle());
    mem->unmap(dirs.handle());
    return false;
  }
  if (!mem->map(out_mask->handle(), &m_ptr_v, &m_bytes)) {
    mem->unmap(origins.handle());
    mem->unmap(dirs.handle());
    mem->unmap(out_hits->handle());
    return false;
  }
  float* h_ptr = static_cast<float*>(h_ptr_v);
  uint8_t* m_ptr = static_cast<uint8_t*>(m_ptr_v);

  for (uint32_t b = 0; b < batch; ++b) {
    const float ox = o_ptr[3 * b + 0];
    const float oy = o_ptr[3 * b + 1];
    const float oz = o_ptr[3 * b + 2];

    const float dx = d_ptr[3 * b + 0];
    const float dy = d_ptr[3 * b + 1];
    const float dz = d_ptr[3 * b + 2];

    bool ok = std::fabs(dz) > kEps;
    float t = ok ? (plane_z - oz) / dz : 0.0f;
    ok = ok && (t >= 0.0f);
    m_ptr[b] = ok ? 1 : 0;

    float hx = ok ? (ox + t * dx) : 0.0f;
    float hy = ok ? (oy + t * dy) : 0.0f;
    float hz = ok ? plane_z : 0.0f;
    h_ptr[3 * b + 0] = hx;
    h_ptr[3 * b + 1] = hy;
    h_ptr[3 * b + 2] = hz;
  }

  mem->unmap(origins.handle());
  mem->unmap(dirs.handle());
  mem->unmap(out_hits->handle());
  mem->unmap(out_mask->handle());
  return true;
}

} // namespace nodus::tensors::kpath
