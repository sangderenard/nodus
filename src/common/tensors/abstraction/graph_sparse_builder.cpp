#include "common/tensors/abstraction/graph_sparse_builder.h"

#include "common/tensors/abstraction/in_memory_backend.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_set>

namespace nodus::tensors {

namespace {

static bool map_write_u32(InMemoryBackend& backend, const AbstractTensor& t, const std::vector<uint32_t>& data) {
  void* raw = nullptr;
  size_t bytes = 0;
  if (!backend.map(t.handle(), &raw, &bytes)) return false;
  if (bytes < data.size() * sizeof(uint32_t)) {
    backend.unmap(t.handle());
    return false;
  }
  std::memcpy(raw, data.data(), data.size() * sizeof(uint32_t));
  backend.unmap(t.handle());
  return true;
}

static bool map_write_f64(InMemoryBackend& backend, const AbstractTensor& t, const std::vector<double>& data) {
  void* raw = nullptr;
  size_t bytes = 0;
  if (!backend.map(t.handle(), &raw, &bytes)) return false;
  if (bytes < data.size() * sizeof(double)) {
    backend.unmap(t.handle());
    return false;
  }
  std::memcpy(raw, data.data(), data.size() * sizeof(double));
  backend.unmap(t.handle());
  return true;
}

static bool graph_ir_value_to_f64(const GraphIrValue& v, double& out) {
  if (const auto* d = std::get_if<double>(&v)) {
    out = *d;
    return true;
  }
  if (const auto* u = std::get_if<uint32_t>(&v)) {
    out = static_cast<double>(*u);
    return true;
  }
  if (const auto* b = std::get_if<bool>(&v)) {
    out = *b ? 1.0 : 0.0;
    return true;
  }
  return false;
}

struct AttrRow {
  uint32_t obj_index = 0;
  uint32_t key_index = 0;
  double value = 0.0;
};

static COOMatrix make_table_u32(uint32_t rows, uint32_t cols, uint32_t nnz, TensorBackend* backend) {
  TensorShape shape{};
  shape.dims = {rows, cols};
  return COOMatrix::create(shape, nnz, TensorDType::U32, backend, TensorDType::U32, CooIndexLayout::RowMajor);
}

static COOMatrix make_flags_table_u32(uint32_t port_count, uint32_t nnz, TensorBackend* backend) {
  TensorShape shape{};
  shape.dims = {port_count};
  return COOMatrix::create(shape, nnz, TensorDType::U32, backend, TensorDType::U32, CooIndexLayout::RowMajor);
}

static COOMatrix make_attr_table_f64(uint32_t obj_count, uint32_t key_count, uint32_t nnz, TensorBackend* backend) {
  TensorShape shape{};
  shape.dims = {obj_count, key_count};
  return COOMatrix::create(shape, nnz, TensorDType::F64, backend, TensorDType::U32, CooIndexLayout::RowMajor);
}

static void fill_identity_table_indices(std::vector<uint32_t>& out_indices, uint32_t nnz, uint32_t rank) {
  out_indices.clear();
  out_indices.reserve(static_cast<size_t>(nnz) * rank);
  for (uint32_t i = 0; i < nnz; ++i) {
    // (row=i, col=0) for rank=2 tables
    out_indices.push_back(i);
    if (rank > 1) out_indices.push_back(0);
    for (uint32_t k = 2; k < rank; ++k) out_indices.push_back(0);
  }
}

} // namespace

SparseGraphEditLogBuilder::SparseGraphEditLogBuilder(TensorBackend* backend) : backend_(backend) {}

void SparseGraphEditLogBuilder::clear() {
  edits_.clear();
}

void SparseGraphEditLogBuilder::apply(const GraphEdit& e) {
  edits_.push_back(e);
}

void SparseGraphEditLogBuilder::apply_all(std::span<const GraphEdit> edits) {
  edits_.insert(edits_.end(), edits.begin(), edits.end());
}

SparseGraph SparseGraphEditLogBuilder::build(SparseGraphBuildError* out_error) const {
  SparseGraphBuildError dummy;
  if (!out_error) out_error = &dummy;
  out_error->message.clear();

  if (!backend_) {
    out_error->message = "SparseGraphEditLogBuilder: backend is null";
    return {};
  }

  auto* mem = dynamic_cast<InMemoryBackend*>(backend_);
  if (!mem) {
    out_error->message = "SparseGraphEditLogBuilder: requires InMemoryBackend (map/unmap)";
    return {};
  }

  // Collect nodes/edges/ports in insertion order.
  std::vector<uint32_t> node_ids;
  std::vector<uint32_t> edge_ids;
  std::vector<uint32_t> port_obj_ids;

  std::unordered_map<uint32_t, uint32_t> node_index;
  std::unordered_map<uint32_t, uint32_t> edge_index;
  std::unordered_map<uint32_t, uint32_t> port_index;

  struct PortRec {
    uint32_t port_obj_id = 0; // may include kind bits; treated as unique key
    uint32_t owner_node_id = 0;
    uint32_t flags = 0;
  };
  std::vector<PortRec> ports;

  struct ConnRec { uint32_t src = 0; uint32_t dst = 0; };
  std::vector<ConnRec> conns;

  // attrs
  std::unordered_map<std::string, uint32_t> key_index;
  std::vector<std::string> keys;

  std::vector<AttrRow> node_attrs;
  std::vector<AttrRow> edge_attrs;
  std::vector<AttrRow> port_attrs;

  auto intern_key = [&](const std::string& k) -> uint32_t {
    auto it = key_index.find(k);
    if (it != key_index.end()) return it->second;
    const uint32_t idx = static_cast<uint32_t>(keys.size());
    keys.push_back(k);
    key_index[k] = idx;
    return idx;
  };

  auto ensure_node = [&](uint32_t node_id) {
    if (node_index.find(node_id) != node_index.end()) return;
    const uint32_t idx = static_cast<uint32_t>(node_ids.size());
    node_ids.push_back(node_id);
    node_index[node_id] = idx;
  };
  auto ensure_edge = [&](uint32_t edge_id) {
    if (edge_index.find(edge_id) != edge_index.end()) return;
    const uint32_t idx = static_cast<uint32_t>(edge_ids.size());
    edge_ids.push_back(edge_id);
    edge_index[edge_id] = idx;
  };
  auto ensure_port = [&](uint32_t port_obj_id) {
    if (port_index.find(port_obj_id) != port_index.end()) return;
    const uint32_t idx = static_cast<uint32_t>(ports.size());
    port_index[port_obj_id] = idx;
    ports.push_back(PortRec{port_obj_id, 0, 0});
  };

  for (const auto& e : edits_) {
    switch (e.kind) {
      case GraphEditKind::AddNode:
        ensure_node(e.a);
        break;
      case GraphEditKind::AddEdge:
        ensure_edge(e.a);
        break;
      case GraphEditKind::AddPort: {
        // e.a = owner node id, e.b = port object id, e.value = flags (u32)
        ensure_node(e.a);
        ensure_port(e.b);
        const uint32_t pidx = port_index[e.b];
        ports[pidx].owner_node_id = e.a;
        if (const auto* uf = std::get_if<uint32_t>(&e.value)) ports[pidx].flags = *uf;
        break;
      }
      case GraphEditKind::Connect:
        ensure_port(e.a);
        ensure_port(e.b);
        conns.push_back(ConnRec{e.a, e.b});
        break;
      case GraphEditKind::SetAttr: {
        // Route by id kind.
        double fv = 0.0;
        if (!graph_ir_value_to_f64(e.value, fv)) {
          out_error->message = "SparseGraphEditLogBuilder: non-numeric attr '" + e.key + "'";
          return {};
        }
        const uint32_t kidx = intern_key(e.key);
        const GraphObjectKind kind = graph_id_kind(e.a);
        if (kind == GraphObjectKind::Node) {
          ensure_node(e.a);
          node_attrs.push_back(AttrRow{node_index[e.a], kidx, fv});
        } else if (kind == GraphObjectKind::Edge) {
          ensure_edge(e.a);
          edge_attrs.push_back(AttrRow{edge_index[e.a], kidx, fv});
        } else {
          // Ports may be plain indices or graph ids; treat as port obj id.
          ensure_port(e.a);
          port_attrs.push_back(AttrRow{port_index[e.a], kidx, fv});
        }
        break;
      }
      default:
        break;
    }
  }

  const uint32_t node_count = static_cast<uint32_t>(node_ids.size());
  const uint32_t edge_count = static_cast<uint32_t>(edge_ids.size());
  const uint32_t port_count = static_cast<uint32_t>(ports.size());
  const uint32_t key_count = static_cast<uint32_t>(keys.size());

  SparseGraph g;

  // Basic required tables.
  g.nodes = make_table_u32(node_count, 1, node_count, backend_);
  g.edges = make_table_u32(edge_count, 1, edge_count, backend_);
  // node_ports is an adjacency-style COO table: indices = (node_index, port_index), value = 1.
  g.node_ports = make_table_u32(node_count, port_count, port_count, backend_);

  // edge_ports: we don't have explicit edge->port ownership in edits; keep a valid empty table.
  g.edge_ports = make_table_u32(edge_count, port_count, 0, backend_);

  // edge_meta_ports: schema is currently under-specified; keep a valid empty table.
  g.edge_meta_ports = make_table_u32(edge_count, 2, 0, backend_);

  // connections: use port_count x port_count and nnz = number of connections.
  g.connections = make_table_u32(port_count, port_count, static_cast<uint32_t>(conns.size()), backend_);

  // port_flags
  g.port_flags = make_flags_table_u32(port_count, port_count, backend_);

  if (!g.nodes.valid() || !g.edges.valid() || !g.node_ports.valid() || !g.edge_meta_ports.valid() || !g.connections.valid() || !g.port_flags.valid()) {
    out_error->message = "SparseGraphEditLogBuilder: failed to allocate COO tables";
    return {};
  }

  // Fill nodes.
  {
    std::vector<uint32_t> idx;
    fill_identity_table_indices(idx, node_count, 2);
    if (!map_write_u32(*mem, g.nodes.indices, idx)) {
      out_error->message = "SparseGraphEditLogBuilder: write nodes.indices failed";
      return {};
    }
    if (!map_write_u32(*mem, g.nodes.values, node_ids)) {
      out_error->message = "SparseGraphEditLogBuilder: write nodes.values failed";
      return {};
    }
  }

  // Fill edges.
  {
    std::vector<uint32_t> idx;
    fill_identity_table_indices(idx, edge_count, 2);
    if (!map_write_u32(*mem, g.edges.indices, idx)) {
      out_error->message = "SparseGraphEditLogBuilder: write edges.indices failed";
      return {};
    }
    if (!map_write_u32(*mem, g.edges.values, edge_ids)) {
      out_error->message = "SparseGraphEditLogBuilder: write edges.values failed";
      return {};
    }
  }

  // Fill node_ports: indices = (node_index(owner), port_index), values = 1
  {
    std::vector<uint32_t> idx;
    idx.reserve(static_cast<size_t>(port_count) * 2);
    for (uint32_t p = 0; p < port_count; ++p) {
      const uint32_t owner = ports[p].owner_node_id;
      auto it = node_index.find(owner);
      if (it == node_index.end()) {
        out_error->message = "SparseGraphEditLogBuilder: node_ports refers to unknown owner node";
        return {};
      }
      idx.push_back(it->second);
      idx.push_back(p);
    }
    if (!map_write_u32(*mem, g.node_ports.indices, idx)) {
      out_error->message = "SparseGraphEditLogBuilder: write node_ports.indices failed";
      return {};
    }

    std::vector<uint32_t> vals(port_count, 1u);
    if (!map_write_u32(*mem, g.node_ports.values, vals)) {
      out_error->message = "SparseGraphEditLogBuilder: write node_ports.values failed";
      return {};
    }
  }

  // Fill port_flags.
  {
    std::vector<uint32_t> flag_idx;
    flag_idx.reserve(port_count);
    std::vector<uint32_t> flag_vals;
    flag_vals.reserve(port_count);
    for (uint32_t i = 0; i < port_count; ++i) {
      flag_idx.push_back(i);
      flag_vals.push_back(ports[i].flags);
    }
    if (!map_write_u32(*mem, g.port_flags.indices, flag_idx)) {
      out_error->message = "SparseGraphEditLogBuilder: write port_flags.indices failed";
      return {};
    }
    if (!map_write_u32(*mem, g.port_flags.values, flag_vals)) {
      out_error->message = "SparseGraphEditLogBuilder: write port_flags.values failed";
      return {};
    }
  }

  // Fill connections.
  {
    std::vector<uint32_t> conn_idx;
    conn_idx.reserve(conns.size() * 2);
    for (const auto& c : conns) {
      const uint32_t src = port_index[c.src];
      const uint32_t dst = port_index[c.dst];
      conn_idx.push_back(src);
      conn_idx.push_back(dst);
    }
    if (!map_write_u32(*mem, g.connections.indices, conn_idx)) {
      out_error->message = "SparseGraphEditLogBuilder: write connections.indices failed";
      return {};
    }
    if (!map_write_u32(*mem, g.connections.values, std::vector<uint32_t>(conns.size(), 0u))) {
      out_error->message = "SparseGraphEditLogBuilder: write connections.values failed";
      return {};
    }
  }

  // Optional attrs.
  if (key_count > 0) {
    if (!node_attrs.empty()) {
      g.node_attrs = make_attr_table_f64(node_count, key_count, static_cast<uint32_t>(node_attrs.size()), backend_);
      std::vector<uint32_t> idx;
      idx.reserve(node_attrs.size() * 2);
      std::vector<double> vals;
      vals.reserve(node_attrs.size());
      for (const auto& a : node_attrs) {
        idx.push_back(a.obj_index);
        idx.push_back(a.key_index);
        vals.push_back(a.value);
      }
      if (!map_write_u32(*mem, g.node_attrs.indices, idx) || !map_write_f64(*mem, g.node_attrs.values, vals)) {
        out_error->message = "SparseGraphEditLogBuilder: write node_attrs failed";
        return {};
      }
    }

    if (!edge_attrs.empty()) {
      g.edge_attrs = make_attr_table_f64(edge_count, key_count, static_cast<uint32_t>(edge_attrs.size()), backend_);
      std::vector<uint32_t> idx;
      idx.reserve(edge_attrs.size() * 2);
      std::vector<double> vals;
      vals.reserve(edge_attrs.size());
      for (const auto& a : edge_attrs) {
        idx.push_back(a.obj_index);
        idx.push_back(a.key_index);
        vals.push_back(a.value);
      }
      if (!map_write_u32(*mem, g.edge_attrs.indices, idx) || !map_write_f64(*mem, g.edge_attrs.values, vals)) {
        out_error->message = "SparseGraphEditLogBuilder: write edge_attrs failed";
        return {};
      }
    }

    if (!port_attrs.empty()) {
      g.port_attrs = make_attr_table_f64(port_count, key_count, static_cast<uint32_t>(port_attrs.size()), backend_);
      std::vector<uint32_t> idx;
      idx.reserve(port_attrs.size() * 2);
      std::vector<double> vals;
      vals.reserve(port_attrs.size());
      for (const auto& a : port_attrs) {
        idx.push_back(a.obj_index);
        idx.push_back(a.key_index);
        vals.push_back(a.value);
      }
      if (!map_write_u32(*mem, g.port_attrs.indices, idx) || !map_write_f64(*mem, g.port_attrs.values, vals)) {
        out_error->message = "SparseGraphEditLogBuilder: write port_attrs failed";
        return {};
      }
    }
  }

  return g;
}

SparseGraph sparse_graph_from_edit_log(std::span<const GraphEdit> edits,
                                      TensorBackend* backend,
                                      SparseGraphBuildError* out_error) {
  SparseGraphEditLogBuilder b(backend);
  b.apply_all(edits);
  return b.build(out_error);
}

} // namespace nodus::tensors
