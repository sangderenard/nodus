#pragma once

#include "common/tensors/abstraction/graph_ir.h"
#include "common/tensors/abstraction/graph_sparse.h"

#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace nodus::tensors {

// Builds a SparseGraph from an edit log.
//
// Important: `SparseGraph` uses COO tensors which are fixed-size buffers.
// This builder therefore accumulates edits/records and materializes the final
// tensors in `build()`.
//
// Current constraints:
// - Requires an InMemoryBackend-backed TensorBackend (because SparseGraph validation
//   and current utilities rely on map/unmap).
// - Attributes are stored as numeric `double` (F64) in *_attrs tables.
//   Non-numeric attribute values are rejected.

struct SparseGraphBuildError final {
  std::string message;
};

class SparseGraphEditLogBuilder final {
 public:
  explicit SparseGraphEditLogBuilder(TensorBackend* backend);

  void clear();
  void apply(const GraphEdit& e);
  void apply_all(std::span<const GraphEdit> edits);

  // Materialize COO tensors.
  // On failure returns an empty graph (tables invalid) and sets out_error.
  SparseGraph build(SparseGraphBuildError* out_error = nullptr) const;

 private:
  TensorBackend* backend_ = nullptr;
  std::vector<GraphEdit> edits_;
};

// Convenience: build in one go.
SparseGraph sparse_graph_from_edit_log(std::span<const GraphEdit> edits,
                                      TensorBackend* backend,
                                      SparseGraphBuildError* out_error = nullptr);

} // namespace nodus::tensors
