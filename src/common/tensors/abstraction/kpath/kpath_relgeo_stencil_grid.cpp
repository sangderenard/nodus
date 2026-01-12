#include "common/tensors/abstraction/kpath/kpath_relgeo_stencil_grid.h"

#include <algorithm>
#include <functional>
#include <queue>
#include <unordered_map>

namespace nodus::tensors::kpath {

namespace {

struct AxisGraph final {
  std::unordered_map<uint32_t, uint32_t> parent;
  std::unordered_map<uint32_t, uint32_t> rank;

  uint32_t find(uint32_t x) {
    auto it = parent.find(x);
    if (it == parent.end()) {
      parent[x] = x;
      rank[x] = 0;
      return x;
    }
    if (it->second == x) return x;
    it->second = find(it->second);
    return it->second;
  }

  void unite(uint32_t a, uint32_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    uint32_t ra = rank[a];
    uint32_t rb = rank[b];
    if (ra < rb) std::swap(a, b);
    parent[b] = a;
    if (ra == rb) rank[a] = ra + 1;
  }
};

} // namespace

RelGeoStencilGrid relgeo_stencil_grid_from_stencil(const RelGeoStencil& stencil,
                                                   const RelGeoStencilGridOptions& options) {
  RelGeoStencilGrid out;
  out.issues = stencil.issues;

  if (stencil.frames.empty() || stencil.points.empty()) return out;

  // Pre-index points for fast iteration.
  std::vector<uint32_t> point_ids;
  point_ids.reserve(stencil.points.size());
  if (options.include_unconstrained) {
    for (const auto& p : stencil.points) {
      if (p) point_ids.push_back(p.v);
    }
  } else {
    std::unordered_map<uint32_t, bool> active;
    for (const auto& ord : stencil.axis_orders) {
      active[ord.a.v] = true;
      active[ord.b.v] = true;
    }
    for (const auto& sign : stencil.axis_signs) {
      active[sign.p.v] = true;
    }
    for (const auto& bt : stencil.betweens) {
      active[bt.a.v] = true;
      active[bt.mid.v] = true;
      active[bt.b.v] = true;
    }
    for (const auto& p : stencil.free_points) {
      active[p.v] = true;
    }
    for (const auto& kv : active) {
      point_ids.push_back(kv.first);
    }
    std::sort(point_ids.begin(), point_ids.end());
  }

  for (const auto& frame : stencil.frames) {
    for (uint32_t axis = 0; axis < frame.dims; ++axis) {
      AxisGraph dsu;
      std::unordered_map<uint32_t, std::vector<uint32_t>> edges;
      std::unordered_map<uint32_t, uint32_t> indeg;

      auto add_edge = [&](uint32_t a, uint32_t b) {
        if (a == b) return;
        edges[a].push_back(b);
        indeg[b] += 1;
      };

      for (uint32_t pid : point_ids) {
        dsu.find(pid);
        indeg[pid] += 0;
      }

      for (const auto& ord : stencil.axis_orders) {
        if (ord.frame != frame.id || ord.axis != axis) continue;
        const uint32_t a = ord.a.v;
        const uint32_t b = ord.b.v;
        if (ord.order == RelAxisOrder::Equal) {
          dsu.unite(a, b);
        }
      }

      for (const auto& sign : stencil.axis_signs) {
        if (sign.frame != frame.id || sign.axis != axis) continue;
        if (!frame.origin) continue;
        if (sign.sign == RelAxisSign::Zero) {
          dsu.unite(frame.origin.v, sign.p.v);
        }
      }

      for (const auto& ord : stencil.axis_orders) {
        if (ord.frame != frame.id || ord.axis != axis) continue;
        const uint32_t a = dsu.find(ord.a.v);
        const uint32_t b = dsu.find(ord.b.v);
        if (a == b) {
          if (ord.order == RelAxisOrder::Less || ord.order == RelAxisOrder::Greater) {
            out.issues.push_back(RelStencilIssue{RelStencilIssue::Kind::Conflict, "grid: order conflicts with equality"});
          }
          continue;
        }
        if (ord.order == RelAxisOrder::Less) add_edge(a, b);
        if (ord.order == RelAxisOrder::Greater) add_edge(b, a);
      }

      for (const auto& bt : stencil.betweens) {
        if (bt.frame != frame.id || bt.axis != axis) continue;
        const uint32_t a = dsu.find(bt.a.v);
        const uint32_t m = dsu.find(bt.mid.v);
        const uint32_t b = dsu.find(bt.b.v);
        if (a != m) add_edge(a, m);
        if (m != b) add_edge(m, b);
      }

      for (const auto& sign : stencil.axis_signs) {
        if (sign.frame != frame.id || sign.axis != axis) continue;
        if (!frame.origin) continue;
        const uint32_t o = dsu.find(frame.origin.v);
        const uint32_t p = dsu.find(sign.p.v);
        if (o == p) continue;
        if (sign.sign == RelAxisSign::Positive) add_edge(o, p);
        if (sign.sign == RelAxisSign::Negative) add_edge(p, o);
      }

      // Build group list.
      std::unordered_map<uint32_t, std::vector<uint32_t>> groups;
      for (uint32_t pid : point_ids) {
        groups[dsu.find(pid)].push_back(pid);
      }

      // Topological order over groups.
      std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> q;
      std::vector<uint32_t> group_ids;
      group_ids.reserve(groups.size());
      for (const auto& kv : groups) {
        group_ids.push_back(kv.first);
      }
      std::sort(group_ids.begin(), group_ids.end());
      for (uint32_t gid : group_ids) {
        if (indeg[gid] == 0) q.push(gid);
      }

      std::vector<uint32_t> topo;
      topo.reserve(group_ids.size());
      while (!q.empty()) {
        const uint32_t g = q.top();
        q.pop();
        topo.push_back(g);
        auto it = edges.find(g);
        if (it == edges.end()) continue;
        for (uint32_t nxt : it->second) {
          if (indeg[nxt] == 0) continue;
          indeg[nxt] -= 1;
          if (indeg[nxt] == 0) q.push(nxt);
        }
      }

      if (topo.size() != groups.size()) {
        out.issues.push_back(RelStencilIssue{RelStencilIssue::Kind::Conflict, "grid: cycle in axis order"});
        topo = group_ids;
      }

      uint32_t axis_size = static_cast<uint32_t>(topo.size());
      std::unordered_map<uint32_t, uint32_t> level;
      if (!options.unique_per_point) {
        for (uint32_t g : topo) level[g] = 0;
        for (uint32_t g : topo) {
          auto it = edges.find(g);
          if (it == edges.end()) continue;
          for (uint32_t nxt : it->second) {
            const uint32_t next_level = level[g] + 1;
            auto il = level.find(nxt);
            if (il == level.end() || il->second < next_level) level[nxt] = next_level;
          }
        }
        axis_size = 0;
        for (const auto& kv : level) axis_size = std::max(axis_size, kv.second + 1);
      }
      out.axes.push_back(RelGeoStencilGridAxis{frame.id, axis, axis_size});

      std::unordered_map<uint32_t, uint32_t> group_to_index;
      group_to_index.reserve(topo.size());
      for (uint32_t i = 0; i < topo.size(); ++i) {
        if (options.unique_per_point) {
          group_to_index[topo[i]] = i;
        } else {
          group_to_index[topo[i]] = level[topo[i]];
        }
      }

      for (const auto& kv : groups) {
        const uint32_t idx = group_to_index[kv.first];
        for (uint32_t pid : kv.second) {
          out.placements.push_back(RelGeoStencilGridPlacement{frame.id, axis, RelPointId(pid), idx});
        }
      }
    }
  }

  return out;
}

std::vector<GraphEdit> relgeo_stencil_grid_edits(const RelGeoStencil& stencil,
                                                 const RelGeoStencilGrid& grid) {
  GraphEditBuilder builder;

  std::unordered_map<uint32_t, uint32_t> frame_nodes;
  std::unordered_map<uint64_t, uint32_t> axis_nodes;
  std::unordered_map<uint32_t, uint32_t> point_nodes;

  auto axis_key = [](uint32_t frame, uint32_t axis) {
    return (static_cast<uint64_t>(frame) << 32) | static_cast<uint64_t>(axis);
  };

  for (const auto& frame : stencil.frames) {
    const uint32_t nid = builder.add_node("stencil.frame");
    frame_nodes[frame.id] = nid;
    builder.set_attr(nid, "stencil.kind", 1u);
    builder.set_attr(nid, "stencil.frame_id", frame.id);
    builder.set_attr(nid, "stencil.dims", frame.dims);
  }

  for (const auto& axis : grid.axes) {
    const uint32_t nid = builder.add_node("stencil.axis");
    axis_nodes[axis_key(axis.frame, axis.axis)] = nid;
    builder.set_attr(nid, "stencil.kind", 2u);
    builder.set_attr(nid, "stencil.frame_id", axis.frame);
    builder.set_attr(nid, "stencil.axis", axis.axis);
    builder.set_attr(nid, "stencil.size", axis.size);
  }

  for (const auto& p : stencil.points) {
    if (!p) continue;
    const uint32_t nid = builder.add_node("stencil.point");
    point_nodes[p.v] = nid;
    builder.set_attr(nid, "stencil.kind", 3u);
    builder.set_attr(nid, "stencil.point_id", p.v);
  }

  for (const auto& place : grid.placements) {
    const uint32_t nid = builder.add_node("stencil.place");
    builder.set_attr(nid, "stencil.kind", 4u);
    builder.set_attr(nid, "stencil.frame_id", place.frame);
    builder.set_attr(nid, "stencil.axis", place.axis);
    builder.set_attr(nid, "stencil.point_id", place.point.v);
    builder.set_attr(nid, "stencil.index", place.index);
  }

  return builder.edits();
}

} // namespace nodus::tensors::kpath
