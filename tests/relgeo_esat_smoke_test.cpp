#include "common/tensors/abstraction/kpath/kpath_relgeo.h"
#include "common/tensors/abstraction/kpath/kpath_relgeo_esat.h"

#include <iostream>

using namespace nodus::tensors::kpath;

static bool require_or_report(bool condition, const char* what) {
  if (condition) return true;
  std::cerr << "[REL-GEO-ESAT] FAILED: " << what << "\n";
  return false;
}

int main() {
  // Derive Dist(center(C), p) == r from PointOnCircle(p,C) + FixedRadius(C,r).
  RelProgram p;

  const RelPointId c = p.add_point(RelPointFixed{0.0f, 0.0f});
  const RelPointId on = p.add_point(RelPointFixed{5.0f, 0.0f});

  RelCircle circle;
  circle.center = c;
  circle.radius = RelRadiusDistance{c, on};
  const RelCircleId C = p.add_circle(circle);

  p.add_assertion(RelAssertPointOnCircle{on, C});
  p.add_assertion(RelAssertFixedRadius{C, 5.0f});

  const RelGeoEsatResult r = relgeo_esaturate(p);

  const RelGeoEsatTerm dist_center_on = RelGeoEsatTerm::dist(c, on);
  const RelGeoEsatTerm five = RelGeoEsatTerm::const_length(5.0f);

  if (!require_or_report(r.are_equal(dist_center_on, five), "expected dist(center,on) == 5")) return 1;

  // Parallel/Perp: derive angle-between-lines facts symbolically.
  {
    RelProgram lp;
    const RelPointId a0 = lp.add_point(RelPointFixed{0.0f, 0.0f});
    const RelPointId a1 = lp.add_point(RelPointFixed{1.0f, 0.0f});
    const RelPointId b0 = lp.add_point(RelPointFixed{0.0f, 1.0f});
    const RelPointId b1 = lp.add_point(RelPointFixed{1.0f, 1.0f});
    const RelPointId c0 = lp.add_point(RelPointFixed{0.0f, 0.0f});
    const RelPointId c1 = lp.add_point(RelPointFixed{0.0f, 1.0f});

    const RelLineId L0 = lp.add_line(a0, a1);
    const RelLineId L1 = lp.add_line(b0, b1);
    const RelLineId L2 = lp.add_line(c0, c1);

    lp.add_assertion(RelAssertParallelLines{L0, L1, 1e-3f});
    lp.add_assertion(RelAssertPerpendicularLines{L0, L2, 1e-3f});

    const RelGeoEsatResult lr = relgeo_esaturate(lp);

    const RelGeoEsatTerm ab01 = RelGeoEsatTerm::angle_between(L0, L1);
    const RelGeoEsatTerm zero = RelGeoEsatTerm::const_angle_pi(0, 1);
    if (!require_or_report(lr.are_equal(ab01, zero), "expected angle_between(L0,L1) == 0 from parallel")) return 1;

    const RelGeoEsatTerm ab02 = RelGeoEsatTerm::angle_between(L0, L2);
    const RelGeoEsatTerm half_pi = RelGeoEsatTerm::const_angle_pi(1, 2);
    if (!require_or_report(lr.are_equal(ab02, half_pi), "expected angle_between(L0,L2) == pi/2 from perp")) return 1;
  }

  std::cout << "[REL-GEO-ESAT] OK\n";
  return 0;
}
