#pragma once

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/coo_matrix.h"

namespace nodus::tensors {

// Compare two tensors elementwise and write a Bool mask for |a-b| <= threshold.
// Returns false if shapes/dtypes unsupported or backends are incompatible.
bool tensor_compare_within_threshold(const AbstractTensor& a,
                                     const AbstractTensor& b,
                                     double threshold,
                                     AbstractTensor* out_mask);

// Compare two tensors elementwise and write a Bool mask for |a-b| > threshold.
// Returns false if shapes/dtypes unsupported or backends are incompatible.
bool tensor_compare_outside_threshold(const AbstractTensor& a,
                                      const AbstractTensor& b,
                                      double threshold,
                                      AbstractTensor* out_mask);

// Convenience helper: returns a sparse COO tensor of values copied from `a`
// at indices where |a-b| <= threshold. Returns empty COOMatrix on failure.
COOMatrix tensor_compare_within_threshold_sparse(const AbstractTensor& a,
                                                 const AbstractTensor& b,
                                                 double threshold,
                                                 TensorBackend* backend = nullptr);

// Convenience helper: returns a sparse COO tensor of values copied from `a`
// at indices where |a-b| > threshold. Returns empty COOMatrix on failure.
COOMatrix tensor_compare_outside_threshold_sparse(const AbstractTensor& a,
                                                  const AbstractTensor& b,
                                                  double threshold,
                                                  TensorBackend* backend = nullptr);

// Create a tensor from raw bytes using the backend (in-memory only).
AbstractTensor tensor_from_bytes(const TensorDesc& desc,
                                 TensorBackend* backend,
                                 const void* bytes,
                                 size_t bytes_len);

// Copy raw bytes into an existing tensor (in-memory only).
bool tensor_copy_from_bytes(AbstractTensor* dst, const void* bytes, size_t bytes_len);

// Extract linear indices and raw value bytes from a COO tensor.
bool coo_extract_linear_values(const COOMatrix& coo,
                               std::vector<uint32_t>& out_linear,
                               std::vector<uint8_t>& out_values_bytes);

} // namespace nodus::tensors
