#pragma once

#include "kpath_film.h"
#include "kpath_raster.h"
#include "kpath_kinematics.h"

#include <limits>
#include <functional>
#include <optional>
#include <vector>

namespace nodus::tensors::kpath {

// Scene tensors represent the environment being modified (additive/subtractive).
// The concrete contents are application-defined; these are optional references.
struct SceneTensors final {
  // Optional scalar or spectral fields representing the environment volume/surface.
  // For now these are placeholders to be wired to concrete tensors later.
  void* environment_field = nullptr;
};

// Sensor media are recording substrates placed in the scene (plates/films/holograms).
struct SensorMedia final {
  PlateTensor2D* plate = nullptr;
  FilmTensor2D* film = nullptr;
  HolographicPlateTensor2D* holographic = nullptr;
};

// Continuous ray/volume query handed into user hooks to derive intersection details.
struct RayIntersectionContext final {
  BeamPoint ray;              // origin+direction (+engaged flag)
  float t_min = 0.0f;         // lower bound for parametric interval
  float t_max = std::numeric_limits<float>::infinity(); // upper bound (volume extent)
};

using SceneFunction = std::function<void(const RayIntersectionContext&, SceneTensors&)>;
using SensorFunction = std::function<void(const RayIntersectionContext&, SceneTensors&, SensorMedia&)>;

// Functional classifications avoid any voxel/vertex/grid storage. Callers supply
// lambdas that answer membership for arbitrary points in space.
enum class LocationClass : uint8_t { Boundary, Inside, Outside, Unknown };

struct RegionClassifier final {
  // Return true if p is on the implicit boundary of the region.
  std::function<bool(const Vec3& p)> on_boundary;
  // Return true if p is strictly inside.
  std::function<bool(const Vec3& p)> is_inside;
  // Return true if p is strictly outside.
  std::function<bool(const Vec3& p)> is_outside;
};

inline LocationClass classify_point(const RegionClassifier& r, const Vec3& p) {
  if (r.on_boundary && r.on_boundary(p)) return LocationClass::Boundary;
  if (r.is_inside && r.is_inside(p)) return LocationClass::Inside;
  if (r.is_outside && r.is_outside(p)) return LocationClass::Outside;
  return LocationClass::Unknown;
}

// Functional scene: arbitrary implicit regions with lambdas, no grids.
struct FunctionalScene final {
  std::vector<RegionClassifier> regions;

  LocationClass classify(const Vec3& p) const {
    // First boundary wins, else first definitive inside/outside, else unknown.
    for (const auto& r : regions) {
      LocationClass c = classify_point(r, p);
      if (c != LocationClass::Unknown) return c;
    }
    return LocationClass::Unknown;
  }
};

// Functional sensor: same idea, plus an optional event logger to keep a flat list.
struct FunctionalSensor final {
  std::vector<RegionClassifier> regions;

  struct Event {
    RayIntersectionContext ctx;
    LocationClass scene_class = LocationClass::Unknown;
    LocationClass sensor_class = LocationClass::Unknown;
  };

  std::vector<Event>* event_log = nullptr; // optional external sink

  void log_event(const Event& e) const {
    if (event_log) event_log->push_back(e);
  }

  LocationClass classify(const Vec3& p) const {
    for (const auto& r : regions) {
      LocationClass c = classify_point(r, p);
      if (c != LocationClass::Unknown) return c;
    }
    return LocationClass::Unknown;
  }
};

// Program state handed to a device for a single step.
struct DeviceStepContext final {
  float t = 0.0f;     // current time
  float dt = 0.0f;    // time step

  // Optional motion/beam programs. Devices may choose to ignore them.
  const ArmatureProgram* armature_program = nullptr;
  const BeamProgram* beam_program = nullptr;

  // Optional user-supplied parametric callback to query motion/tool state.
  std::function<void(float t, float dt)> parametric_callback;

  // Optional hooks to derive intersection details from a ray/volume query.
  SceneFunction scene_fn;
  SensorFunction sensor_fn;

  // Optional functional scene/sensor (implicit surfaces/volumes, no grids).
  const FunctionalScene* functional_scene = nullptr;
  const FunctionalSensor* functional_sensor = nullptr;
};

// Abstract device: anything that, on step(), can emit ray strikes, modify scene tensors,
// and/or write to sensor media. This is not restricted to beams; additive/subtractive
// interactions can mutate the scene tensors directly.
struct Device final {
  // Device-local configuration/state can be stored via captures in the step_fn.
  std::function<void(const DeviceStepContext&, SceneTensors&, SensorMedia&)> step_fn;
};

inline void dispatch_intersection_hooks(const RayIntersectionContext& ctx,
                                        SceneTensors& scene,
                                        SensorMedia& media,
                                        const DeviceStepContext& step_ctx) {
  if (step_ctx.scene_fn) step_ctx.scene_fn(ctx, scene);
  if (step_ctx.sensor_fn) step_ctx.sensor_fn(ctx, scene, media);
}

inline void dispatch_functional_intersection(const RayIntersectionContext& ctx,
                                             const DeviceStepContext& step_ctx) {
  if (!step_ctx.functional_scene && !step_ctx.functional_sensor) return;

  FunctionalSensor::Event evt;
  evt.ctx = ctx;

  if (step_ctx.functional_scene) {
    Vec3 p_at_t{ctx.ray.ox + ctx.ray.dx * ctx.t_min,
                ctx.ray.oy + ctx.ray.dy * ctx.t_min,
                ctx.ray.oz + ctx.ray.dz * ctx.t_min};
    evt.scene_class = step_ctx.functional_scene->classify(p_at_t);
  }

  if (step_ctx.functional_sensor) {
    Vec3 p_at_t{ctx.ray.ox + ctx.ray.dx * ctx.t_min,
                ctx.ray.oy + ctx.ray.dy * ctx.t_min,
                ctx.ray.oz + ctx.ray.dz * ctx.t_min};
    evt.sensor_class = step_ctx.functional_sensor->classify(p_at_t);
    step_ctx.functional_sensor->log_event(evt);
  }
}

// Utility to let a device emit a batch of strikes and apply them to all configured media.
inline void apply_strikes_to_media(const RayStrikeBatch& batch, SensorMedia& media) {
  if (media.plate) apply_strikes(*media.plate, batch);
  if (media.film) apply_strikes(*media.film, batch);
  if (media.holographic) apply_strikes(*media.holographic, batch);
}

} // namespace nodus::tensors::kpath
