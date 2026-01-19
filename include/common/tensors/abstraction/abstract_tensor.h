#pragma once

#include <cstdint>
#include <initializer_list>

#include "common/tensors/abstraction/abstract_tensor_handle.h"
#include "common/tensors/abstraction/tensor_index.h"
#include "common/tensors/abstraction/tensor_types.h"

namespace nodus::tensors {

class TensorBackend;
class AbstractTensorPool;
struct TensorTransferConfig;

// Move-only handle wrapper with optional ownership semantics.
class AbstractTensor {
public:
    AbstractTensor() = default;
    explicit AbstractTensor(const TensorDesc& desc, TensorBackend* backend = nullptr);
    AbstractTensor(const AbstractTensor&) = delete;
    AbstractTensor& operator=(const AbstractTensor&) = delete;

    AbstractTensor(AbstractTensor&& other) noexcept;
    AbstractTensor& operator=(AbstractTensor&& other) noexcept;
    ~AbstractTensor();

    static AbstractTensor create(const TensorDesc& desc, TensorBackend* backend = nullptr);
    static AbstractTensor wrap(AbstractTensorHandle handle,
                               const TensorDesc& desc,
                               TensorBackend* backend,
                               bool owns_handle);

    bool valid() const { return abstract_tensor_handle_is_valid(handle_); }
    const TensorDesc& desc() const { return desc_; }
    TensorBackend* backend() const { return backend_; }
    AbstractTensorHandle handle() const { return handle_; }

    void set_dtype(TensorDType dtype);
    void set_slice_affine_row_major(const float* affine16);
    void set_slice_saturate_threshold(float threshold);
    void reset();
    void release();
    bool refresh_desc();

    AbstractTensor slice(const TensorIndexSpec& index) const;
    bool set_slice(const TensorIndexSpec& index, const AbstractTensor& value);
    AbstractTensor operator()(const TensorIndexSpec& index) const;
    AbstractTensor operator()(std::initializer_list<TensorIndex> dims) const;
    bool set(std::initializer_list<TensorIndex> dims, const AbstractTensor& value);

    // Unified scatter/gather/transfer front-ends.
    bool gather(const AbstractTensor& input_indices,
                AbstractTensor& output,
                const AbstractTensor& output_indices,
                const TensorTransferConfig& config) const;
    bool scatter(const AbstractTensor& input_indices,
                 AbstractTensor& output,
                 const AbstractTensor& output_indices,
                 const TensorTransferConfig& config) const;
    bool transfer(const AbstractTensor& input_indices,
                  AbstractTensor& output,
                  const AbstractTensor& output_indices,
                  bool gather_first,
                  const TensorTransferConfig& config) const;

private:
    friend class AbstractTensorPool;
    static AbstractTensor create_raw(const TensorDesc& desc, TensorBackend* backend = nullptr);

    AbstractTensorHandle handle_{};
    TensorDesc desc_{};
    TensorBackend* backend_ = nullptr;
    bool owns_handle_ = true;
    AbstractTensorPool* pool_ = nullptr;
    TensorDesc pool_desc_{};
};

} // namespace nodus::tensors
