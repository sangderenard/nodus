#include "common/tensors/abstraction/kpath/kpath_path_builder.h"

#include <cmath>
#include <utility>

namespace nodus::tensors::kpath {

void PathBuilder::clear() {
  segments_.clear();
  cur_x_ = 0.0f;
  cur_y_ = 0.0f;
  has_subpath_ = false;
}

void PathBuilder::ensure_subpath() {
  if (!has_subpath_) {
    move_to(cur_x_, cur_y_);
  }
}

PathBuilder& PathBuilder::move_to(float x, float y) {
  OutlineSegment seg{OutlineOp::MoveTo, x, y, 0.0f, 0.0f, 0.0f, 0.0f};
  segments_.push_back(seg);
  cur_x_ = x;
  cur_y_ = y;
  has_subpath_ = true;
  return *this;
}

PathBuilder& PathBuilder::line_to(float x, float y) {
  ensure_subpath();
  OutlineSegment seg{OutlineOp::LineTo, cur_x_, cur_y_, 0.0f, 0.0f, x, y};
  segments_.push_back(seg);
  cur_x_ = x;
  cur_y_ = y;
  return *this;
}

PathBuilder& PathBuilder::quad_to(float cx, float cy, float x, float y) {
  ensure_subpath();
  OutlineSegment seg{OutlineOp::QuadTo, cur_x_, cur_y_, cx, cy, x, y};
  segments_.push_back(seg);
  cur_x_ = x;
  cur_y_ = y;
  return *this;
}

PathBuilder& PathBuilder::cubic_to(float c1x, float c1y, float c2x, float c2y, float x, float y) {
  ensure_subpath();
  OutlineSegment seg{OutlineOp::CubicTo, c1x, c1y, c2x, c2y, x, y};
  segments_.push_back(seg);
  cur_x_ = x;
  cur_y_ = y;
  return *this;
}

PathBuilder& PathBuilder::arc(float cx, float cy, float radius, float start_angle_rad, float sweep_angle_rad) {
  ensure_subpath();
  OutlineSegment seg{OutlineOp::Arc, cx, cy, radius, start_angle_rad, sweep_angle_rad, 0.0f};
  segments_.push_back(seg);
  // Update current point to end of arc for chaining.
  float end_ang = start_angle_rad + sweep_angle_rad;
  cur_x_ = cx + radius * std::cos(end_ang);
  cur_y_ = cy + radius * std::sin(end_ang);
  return *this;
}

PathBuilder& PathBuilder::sin_to(float x, float y, float amplitude, float cycles, float phase_rad) {
  ensure_subpath();
  OutlineSegment seg{OutlineOp::Sin, x, y, amplitude, cycles, phase_rad, 0.0f};
  segments_.push_back(seg);
  cur_x_ = x;
  cur_y_ = y;
  return *this;
}

PathBuilder& PathBuilder::append_outline(const GlyphOutline& outline, float tx, float ty) {
  for (const auto& s : outline.segments) {
    OutlineSegment seg = s;
    seg.x1 += tx; seg.y1 += ty;
    seg.x2 += tx; seg.y2 += ty;
    seg.x3 += tx; seg.y3 += ty;
    segments_.push_back(seg);
  }
  // Best-effort to update cursor if the appended outline ends with a draw.
  if (!outline.segments.empty()) {
    const auto& last = outline.segments.back();
    if (last.op == OutlineOp::MoveTo) {
      cur_x_ = last.x1 + tx;
      cur_y_ = last.y1 + ty;
      has_subpath_ = true;
    } else {
      cur_x_ = last.x3 + tx;
      cur_y_ = last.y3 + ty;
      has_subpath_ = true;
    }
  }
  return *this;
}

PathBuilder& PathBuilder::close() {
  if (!has_subpath_) return *this;
  OutlineSegment seg{OutlineOp::Close, cur_x_, cur_y_, 0.0f, 0.0f, 0.0f, 0.0f};
  segments_.push_back(seg);
  return *this;
}

GlyphOutline PathBuilder::finalize(uint32_t glyph_id) const {
  GlyphOutline out;
  out.glyph_id = glyph_id;
  out.segments = segments_;
  return out;
}

PathBuilder& PathBuilder::warp(const std::function<void(float, float&, float&)>& warp_fn,
                               uint32_t samples_per_curve) {
  if (!warp_fn) return *this;
  samples_per_curve = std::max<uint32_t>(1, samples_per_curve);

  std::vector<OutlineSegment> rebuilt;
  rebuilt.reserve(segments_.size() * 2);

  auto emit_move = [&](float x, float y) {
    rebuilt.push_back({OutlineOp::MoveTo, x, y, 0.0f, 0.0f, 0.0f, 0.0f});
  };
  auto emit_line = [&](float x0, float y0, float x1, float y1) {
    rebuilt.push_back({OutlineOp::LineTo, x0, y0, 0.0f, 0.0f, x1, y1});
  };

  bool have_pos = false;
  float curx = 0.0f, cury = 0.0f;
  float startx = 0.0f, starty = 0.0f;

  auto apply_warp = [&](float u, float& x, float& y) {
    warp_fn(u, x, y);
  };

  for (const auto& seg : segments_) {
    switch (seg.op) {
      case OutlineOp::MoveTo: {
        float x = seg.x1;
        float y = seg.y1;
        apply_warp(0.0f, x, y);
        emit_move(x, y);
        curx = x; cury = y;
        startx = x; starty = y;
        have_pos = true;
        break;
      }
      case OutlineOp::LineTo: {
        if (!have_pos) {
          float x = seg.x1;
          float y = seg.y1;
          emit_move(x, y);
          curx = x; cury = y;
          startx = x; starty = y;
          have_pos = true;
        }
        float x0 = curx, y0 = cury;
        float x1 = seg.x3, y1 = seg.y3;
        for (uint32_t i = 1; i <= samples_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(samples_per_curve);
          float x = x0 + (x1 - x0) * t;
          float y = y0 + (y1 - y0) * t;
          float u = t;
          apply_warp(u, x, y);
          if (i == 1 && rebuilt.empty()) emit_move(x0, y0);
          emit_line(curx, cury, x, y);
          curx = x; cury = y;
        }
        break;
      }
      case OutlineOp::QuadTo:
      case OutlineOp::CubicTo:
      case OutlineOp::Arc:
      case OutlineOp::Sin: {
        if (!have_pos) {
          float x = seg.x1;
          float y = seg.y1;
          emit_move(x, y);
          curx = x; cury = y;
          startx = x; starty = y;
          have_pos = true;
        }
        for (uint32_t i = 1; i <= samples_per_curve; ++i) {
          float t = static_cast<float>(i) / static_cast<float>(samples_per_curve);
          float x = curx;
          float y = cury;
          switch (seg.op) {
            case OutlineOp::QuadTo: {
              float a = 1.0f - t;
              x = a * a * curx + 2.0f * a * t * seg.x2 + t * t * seg.x3;
              y = a * a * cury + 2.0f * a * t * seg.y2 + t * t * seg.y3;
              break;
            }
            case OutlineOp::CubicTo: {
              float a = 1.0f - t;
              x = a*a*a*curx + 3*a*a*t*seg.x1 + 3*a*t*t*seg.x2 + t*t*t*seg.x3;
              y = a*a*a*cury + 3*a*a*t*seg.y1 + 3*a*t*t*seg.y2 + t*t*t*seg.y3;
              break;
            }
            case OutlineOp::Arc: {
              float ang = seg.y2 + seg.x3 * t;
              float r = seg.x2;
              x = seg.x1 + r * std::cos(ang);
              y = seg.y1 + r * std::sin(ang);
              break;
            }
            case OutlineOp::Sin: {
              float dx = seg.x1 - curx;
              float dy = seg.y1 - cury;
              float len = std::sqrt(dx * dx + dy * dy);
              float tx = (len > 0.0f) ? dx / len : 0.0f;
              float ty = (len > 0.0f) ? dy / len : 0.0f;
              float nx = -ty;
              float ny = tx;
              float base_x = curx + dx * t;
              float base_y = cury + dy * t;
              float s = std::sin((2.0f * 3.14159265358979323846f * seg.y2 * t) + seg.x3);
              x = base_x + nx * seg.x2 * s;
              y = base_y + ny * seg.x2 * s;
              break;
            }
            default: break;
          }
          float u = t;
          apply_warp(u, x, y);
          emit_line(curx, cury, x, y);
          curx = x; cury = y;
        }
        break;
      }
      case OutlineOp::Close: {
        if (have_pos) {
          float x = startx;
          float y = starty;
          apply_warp(1.0f, x, y);
          emit_line(curx, cury, x, y);
          curx = x; cury = y;
        }
        break;
      }
    }
  }

  segments_.swap(rebuilt);
  if (!segments_.empty()) {
    const auto& s = segments_.back();
    cur_x_ = s.x3;
    cur_y_ = s.y3;
    has_subpath_ = true;
  }

  return *this;
}

} // namespace nodus::tensors::kpath
