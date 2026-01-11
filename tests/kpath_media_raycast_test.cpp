#include "common/tensors/abstraction/kpath/kpath_device.h"
#include "common/tensors/abstraction/kpath/kpath_film.h"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

using namespace nodus::tensors::kpath;

static bool approx(float a, float b, float eps = 1e-5f) {
  return std::fabs(a - b) <= eps;
}

int main() {
  // Simple downward beam starting above the plane, projected via kpath helper.
  BeamProgram beams;
  beams.beams.push_back(BeamPoint{1.0f, 1.0f, 1.0f, 0.0f, 0.0f, -1.0f, true});

  ArmatureProgram hits;
  bool ok = project_beam_program_to_plane(beams, /*plane_z=*/0.0f, hits);
  assert(ok);
  assert(hits.points.size() == 1);

  const ToolPoint& hp = hits.points.front();
  float hit_x = hp.x;
  float hit_y = hp.y;

  // Recover parametric t along the original beam for functional classification.
  const BeamPoint& beam = beams.beams.front();
  float t_hit = (hp.z - beam.oz) / beam.dz;
  assert(t_hit > 0.0f);

  // Prepare strike batch (scalar intensity and wavelength placeholders).
  std::array<float, 1> xs{hit_x};
  std::array<float, 1> ys{hit_y};
  std::array<float, 1> ws{532.0f}; // wavelength nm placeholder
  std::array<float, 1> thetas{0.0f};
  std::array<float, 1> phis{0.0f};
  std::array<float, 1> intens{2.0f};
  RayStrikeBatch batch{std::span<const float>(xs), std::span<const float>(ys), std::span<const float>(ws),
                       std::span<const float>(thetas), std::span<const float>(phis), std::span<const float>(intens)};

  // Plate receiver: accumulates scalar intensity.
  PlateTensor2D plate(8, 8);
  apply_strikes(plate, batch);
  uint32_t px = clamp_pixel(hit_x, plate.width);
  uint32_t py = clamp_pixel(hit_y, plate.height);
  assert(approx(plate.at(px, py), 2.0f));

  // Film receiver: nearest-bin spectral accumulation.
  SpectrumBasis basis;
  basis.bin_centers = {400.0f, 532.0f, 700.0f};
  FilmTensor2D film(8, 8, 3, basis);
  apply_strikes(film, batch);
  assert(approx(film.at(px, py, 1), 2.0f));

  // Holographic receiver: per-hit stencil capture.
  HolographicPlateTensor2D holo;
  holo.resize(8, 8);
  apply_strikes(holo, batch);
  assert(holo.stencils.size() == 1);
  assert(holo.at(px, py).size() == 1);
  assert(approx(holo.stencils.front().at(0, 0, 0), 2.0f));

  // Functional scene/sensor: sphere (radius 2) and plane boundary z=0.
  FunctionalScene scene;
  RegionClassifier sphere;
  sphere.on_boundary = [](const Vec3& p) {
    float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return approx(r2, 4.0f, 1e-4f);
  };
  sphere.is_inside = [](const Vec3& p) {
    float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return r2 < 4.0f;
  };
  sphere.is_outside = [](const Vec3& p) {
    float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return r2 > 4.0f;
  };
  scene.regions.push_back(sphere);

  FunctionalSensor sensor;
  RegionClassifier plane;
  plane.on_boundary = [](const Vec3& p) { return std::fabs(p.z) < 1e-5f; };
  plane.is_inside = [](const Vec3& p) { return p.z < -1e-5f; };
  plane.is_outside = [](const Vec3& p) { return p.z > 1e-5f; };
  sensor.regions.push_back(plane);

  std::vector<FunctionalSensor::Event> events;
  sensor.event_log = &events;

  DeviceStepContext step{};
  step.functional_scene = &scene;
  step.functional_sensor = &sensor;

  RayIntersectionContext ctx{};
  ctx.ray = beam;
  ctx.t_min = t_hit; // parametric point at intersection with plane
  dispatch_functional_intersection(ctx, step);

  assert(events.size() == 1);
  assert(events[0].scene_class == LocationClass::Inside);
  assert(events[0].sensor_class == LocationClass::Boundary);

  return 0;
}
