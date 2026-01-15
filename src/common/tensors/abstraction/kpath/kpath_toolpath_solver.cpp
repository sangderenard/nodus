#include "common/tensors/abstraction/kpath/kpath_toolpath_solver.h"

#include <algorithm>
#include <cmath>

namespace nodus::tensors::kpath {

namespace {

float vec2_len(float x, float y) {
  return std::sqrt(x * x + y * y);
}

float clamp_abs(float v, float limit) {
  if (limit <= 0.0f) return v;
  if (v > limit) return limit;
  if (v < -limit) return -limit;
  return v;
}

void append_span_samples(const PlanarSpan& span,
                         float step_len,
                         std::vector<Vec2>& out_points,
                         std::vector<ToolMode>& out_modes) {
  if (span.points.empty()) return;
  const float step = std::max(step_len, 1e-3f);
  for (size_t i = 1; i < span.points.size(); ++i) {
    const Vec2& a = span.points[i - 1];
    const Vec2& b = span.points[i];
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len = vec2_len(dx, dy);
    if (len <= 1e-6f) continue;
    const int steps = std::max(1, static_cast<int>(std::ceil(len / step)));
    for (int s = 0; s <= steps; ++s) {
      const float t = static_cast<float>(s) / static_cast<float>(steps);
      Vec2 p{a.x + dx * t, a.y + dy * t};
      out_points.push_back(p);
      out_modes.push_back(span.tool_mode);
    }
  }
  if (span.closed && !span.points.empty()) {
    const Vec2& a = span.points.back();
    const Vec2& b = span.points.front();
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    const float len = vec2_len(dx, dy);
    if (len > 1e-6f) {
      const int steps = std::max(1, static_cast<int>(std::ceil(len / step)));
      for (int s = 0; s <= steps; ++s) {
        const float t = static_cast<float>(s) / static_cast<float>(steps);
        Vec2 p{a.x + dx * t, a.y + dy * t};
        out_points.push_back(p);
        out_modes.push_back(span.tool_mode);
      }
    }
  }
}

} // namespace

bool solve_planar_toolpath_to_armature(const ArmatureModel& model,
                                       const PlanarToolpath& path,
                                       const ToolpathSolveParams& params,
                                       ToolpathSolveOutput* out) {
  if (!out) return false;
  out->frames.clear();
  if (path.spans.empty()) return false;

  const size_t joint_count = model.joints.size();
  if (joint_count == 0) return false;

  const float base_dt = std::max(params.base_dt, 1e-5f);
  const float feed_rate = std::max(params.feed_rate, 1e-3f);
  const float step_len = feed_rate * base_dt;

  std::vector<Vec2> samples;
  std::vector<ToolMode> modes;
  for (const auto& span : path.spans) {
    append_span_samples(span, step_len, samples, modes);
  }
  if (samples.empty()) return false;

  std::vector<float> q(joint_count, 0.0f);
  std::vector<float> dq(joint_count, 0.0f);
  const float max_speed = std::max(params.max_speed, 1e-3f);
  const float max_accel = std::max(params.max_accel, 1e-3f);
  const float max_force = std::max(params.max_force, 1e-3f);

  float t = 0.0f;
  out->frames.reserve(samples.size());

  for (size_t i = 0; i < samples.size(); ++i) {
    ToolpathActuationFrame frame{};
    frame.tool_mode = modes[i];
    frame.target_pos = Vec3{samples[i].x, samples[i].y, params.nominal_z};
    frame.target_rot = Quat{};

    const bool desired_engaged = (frame.tool_mode != ToolMode::Travel);
    frame.tool_engaged = desired_engaged;

    std::vector<float> q_target(joint_count, 0.0f);
    if (joint_count >= 1) q_target[0] = frame.target_pos.x;
    if (joint_count >= 2) q_target[1] = frame.target_pos.y;
    if (joint_count >= 3) q_target[2] = frame.target_pos.z;

    float max_err = 0.0f;
    for (size_t j = 0; j < joint_count; ++j) {
      max_err = std::max(max_err, std::fabs(q_target[j] - q[j]));
    }

    float dt_scale = 1.0f;
    for (size_t j = 0; j < joint_count; ++j) {
      const float err = std::fabs(q_target[j] - q[j]);
      const float req_speed = err / base_dt;
      dt_scale = std::max(dt_scale, req_speed / max_speed);
    }

    const float dt = base_dt * dt_scale;
    frame.dt = dt;
    frame.dt_scale = dt_scale;
    frame.t = t;
    t += dt;

    if (desired_engaged && max_err > params.hold_error) {
      frame.tool_engaged = false;
      if (joint_count >= 3) {
        q_target[2] = params.lift_z;
        frame.target_pos.z = params.lift_z;
      }
    }

    std::vector<float> dq_target(joint_count, 0.0f);
    for (size_t j = 0; j < joint_count; ++j) {
      const float err = q_target[j] - q[j];
      dq_target[j] = clamp_abs(err / dt, max_speed);
    }

    std::vector<float> u(joint_count, 0.0f);
    for (size_t j = 0; j < joint_count; ++j) {
      const float err = q_target[j] - q[j];
      const float derr = dq_target[j] - dq[j];
      float force = params.kp * err + params.kd * derr;
      force = clamp_abs(force, max_force);
      force = clamp_abs(force, max_accel);
      u[j] = force;
    }

    for (size_t j = 0; j < joint_count; ++j) {
      dq[j] += u[j] * dt;
      dq[j] = clamp_abs(dq[j], max_speed);
      q[j] += dq[j] * dt;
    }

    frame.q = q;
    frame.dq = dq;
    frame.u = u;
    if (joint_count >= 32) frame.axis_mask = 0xFFFFFFFFu;
    else frame.axis_mask = (joint_count == 0) ? 0 : ((1u << static_cast<uint32_t>(joint_count)) - 1u);

    out->frames.push_back(std::move(frame));
  }

  return true;
}

} // namespace nodus::tensors::kpath
