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
  // Functional scene/sensor: sphere + plane, log a single hit.
  FunctionalScene scene;
  RegionClassifier sphere;
  sphere.on_boundary = [](const Vec3& p) {
    const float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return approx(r2, 1.0f, 1e-4f);
  };
  sphere.is_inside = [](const Vec3& p) {
    const float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return r2 < 1.0f;
  };
  sphere.is_outside = [](const Vec3& p) {
    const float r2 = p.x * p.x + p.y * p.y + p.z * p.z;
    return r2 > 1.0f;
  };
  scene.regions.push_back(sphere);

  FunctionalSensor sensor;
  RegionClassifier plane;
  plane.on_boundary = [](const Vec3& p) { return std::fabs(p.z) < 1e-5f; };
  plane.is_inside = [](const Vec3& p) { return p.z < -1e-5f; };
  plane.is_outside = [](const Vec3& p) { return p.z > 1e-5f; };
  sensor.regions.push_back(plane);
  std::vector<FunctionalSensor::Event> log;
  sensor.event_log = &log;

  DeviceStepContext step{};
  step.functional_scene = &scene;
  step.functional_sensor = &sensor;

  RayIntersectionContext ctx{};
  ctx.ray = BeamPoint{0.0f, 0.0f, 2.0f, 0.0f, 0.0f, -1.0f, true};
  ctx.t_min = 2.0f; // hits plane at z=0
  dispatch_functional_intersection(ctx, step);

  assert(log.size() == 1);
  assert(log[0].scene_class == LocationClass::Inside);
  assert(log[0].sensor_class == LocationClass::Boundary);

  // Plate: scalar accumulation at floored pixel.
  PlateTensor2D plate(4, 4);
  std::array<float, 1> xs{1.1f}, ys{2.2f}, ws{500.0f}, ts{0.0f}, ps{0.0f}, intens{3.0f};
  RayStrikeBatch batch{std::span<const float>(xs), std::span<const float>(ys), std::span<const float>(ws),
                       std::span<const float>(ts), std::span<const float>(ps), std::span<const float>(intens)};
  apply_strikes(plate, batch);
  assert(approx(plate.at(1, 2), 3.0f));

  // Film: spectral binning by nearest wavelength bin.
  SpectrumBasis basis;
  basis.bin_centers = {400.0f, 600.0f};
  FilmTensor2D film(4, 4, 2, basis);
  apply_strikes(film, batch);
  assert(approx(film.at(1, 2, 1), 3.0f));
  assert(approx(film.at(1, 2, 0), 0.0f));

  // Holographic: per-hit stencil appended.
  HolographicPlateTensor2D holo;
  holo.resize(4, 4);
  apply_strikes(holo, batch);
  assert(holo.stencils.size() == 1);
  assert(holo.at(1, 2).size() == 1);
  const auto& stencil = holo.stencils.front();
  assert(stencil.dim_theta == 1 && stencil.dim_phi == 1 && stencil.dim_rho == 1);
  assert(approx(stencil.at(0, 0, 0), 3.0f));

  return 0;
}
