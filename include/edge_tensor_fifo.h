#pragma once

#include "common/tensors/abstraction/tensor_types.h"
#include "common/tensors/abstraction/coo_matrix.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_backend.h"
#include "mem_backend.h"
#include "value_types.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

struct EdgeTensorFifo {
    static constexpr size_t kMaxReaders = 64;
    static constexpr int32_t kMaxOrder = 4;

    struct ReaderEntry {
        std::atomic<uint64_t> key{0};
        std::atomic<uint64_t> seq{0}; // next sequence number to read
    };

    struct Impl {
        std::atomic<uint64_t> write_seq{0}; // next sequence to write
        std::atomic<uint64_t> writer{0};    // bound writer key (0 => unbound)
        std::unique_ptr<std::atomic<uint64_t>[]> slot_seq; // published tag per slot (seq+1), 0 => empty
        // Storage is now owned via a backend buffer handle. Backends may be
        // host-backed or device-backed; operations should map or use vtable
        // copy hooks when accessing data.
        gp_mem_backend_handle_t storage_handle{nullptr};
        // Typed view is not persisted; map on demand. Backends should be
        // accessed via map/copy helpers — no cached typed pointer is kept.
        size_t elem_size = sizeof(float);                  // bytes per element
        int32_t type_id = -1;                              // schema id
        std::unique_ptr<ReaderEntry[]> readers;
        size_t stride = 1;
        size_t slots = 1;
        size_t top_k = 0;
        std::atomic<float> write_friction{0.0f};
        std::atomic<float> read_friction{0.0f};
        std::atomic<uint64_t> last_write_seq{0};
        std::atomic<uint64_t> last_read_seq{0};
        std::atomic<int32_t> last_write_region{-1};
        std::atomic<int32_t> last_read_region{-1};
        std::atomic<float> write_phase{0.0f};
        std::atomic<float> read_phase{0.0f};
        std::atomic<int32_t> friction_regions{8};
        std::mutex cv_mu;
        std::condition_variable cv;
        bool configured = false;

        // Optional memory backend handle attached to this FIFO
        gp_mem_backend_handle_t backend{nullptr};

        Impl() : readers(new ReaderEntry[kMaxReaders]) {}

        void release_storage() noexcept {
            if (!storage_handle) return;
            const gp_mem_backend_vtable_t* vt =
                gp_mem_backend_get_vtable(storage_handle);
            if (vt && vt->free) {
                vt->free(storage_handle);
            } else {
                gp_mem_backend_release(storage_handle);
            }
            storage_handle = nullptr;
        }

        ~Impl() { release_storage(); }
    };

    std::vector<int32_t> shape;
    std::unique_ptr<Impl> impl;
    nodus::tensors::TensorLayout layout = nodus::tensors::TensorLayout::Dense;
    nodus::tensors::TensorDType dtype = nodus::tensors::TensorDType::Unknown;
    std::vector<float> last_sample;
    std::vector<uint8_t> last_sample_bytes;
    bool last_sample_valid = false;
    bool last_sample_bytes_valid = false;
    bool delta_mode = false;
    nodus::tensors::AbstractTensorHandle tensor_storage{};
    nodus::tensors::TensorBackend* tensor_storage_backend = nullptr;
    bool delta_sparse_mode = false;
    bool delta_sparse_accumulate = false;
    float delta_sparse_threshold = 0.0f;
    int active_sparse_accum = 0;
    struct SparseAccumState {
        std::unordered_map<uint32_t, std::vector<uint8_t>> values;
    };
    SparseAccumState sparse_accum[2];
    std::mutex sparse_mu;
    std::vector<float> order_history;
    std::vector<float> order_integrator;
    std::vector<float> scratch;
    int32_t order_mode = 0;
    int32_t order_history_count = 0;
    int32_t order_history_cursor = 0;
    // Dirty-grid state for delta-aware buffers (tile mask for recent write).
    int32_t dirty_grid_x_req = 0;
    int32_t dirty_grid_y_req = 0;
    int32_t dirty_grid_x = 0;
    int32_t dirty_grid_y = 0;
    float dirty_threshold = 0.0f;
    std::vector<uint8_t> dirty_mask;
    std::atomic<uint64_t> dirty_seq{0};
    std::mutex dirty_mu;

    EdgeTensorFifo() : impl(new Impl()) {}
    EdgeTensorFifo(EdgeTensorFifo&& other) noexcept
        : shape(std::move(other.shape)),
          impl(std::move(other.impl)),
          layout(other.layout),
          dtype(other.dtype),
          last_sample(std::move(other.last_sample)),
          last_sample_bytes(std::move(other.last_sample_bytes)),
          last_sample_valid(other.last_sample_valid),
          last_sample_bytes_valid(other.last_sample_bytes_valid),
          delta_mode(other.delta_mode),
          tensor_storage(std::move(other.tensor_storage)),
          tensor_storage_backend(other.tensor_storage_backend),
          delta_sparse_mode(other.delta_sparse_mode),
          delta_sparse_accumulate(other.delta_sparse_accumulate),
          delta_sparse_threshold(other.delta_sparse_threshold),
          active_sparse_accum(other.active_sparse_accum),
          sparse_accum{std::move(other.sparse_accum[0]), std::move(other.sparse_accum[1])},
          order_history(std::move(other.order_history)),
          order_integrator(std::move(other.order_integrator)),
          scratch(std::move(other.scratch)),
          order_mode(other.order_mode),
          order_history_count(other.order_history_count),
          order_history_cursor(other.order_history_cursor),
          dirty_grid_x_req(other.dirty_grid_x_req),
          dirty_grid_y_req(other.dirty_grid_y_req),
          dirty_grid_x(other.dirty_grid_x),
          dirty_grid_y(other.dirty_grid_y),
          dirty_threshold(other.dirty_threshold),
          dirty_mask(std::move(other.dirty_mask)),
          dirty_seq(other.dirty_seq.load(std::memory_order_relaxed)) {}

    EdgeTensorFifo& operator=(EdgeTensorFifo&& other) noexcept {
        if (this == &other) return *this;
        shape = std::move(other.shape);
        impl = std::move(other.impl);
        layout = other.layout;
        dtype = other.dtype;
        last_sample = std::move(other.last_sample);
        last_sample_bytes = std::move(other.last_sample_bytes);
        last_sample_valid = other.last_sample_valid;
        last_sample_bytes_valid = other.last_sample_bytes_valid;
        delta_mode = other.delta_mode;
        tensor_storage = std::move(other.tensor_storage);
        tensor_storage_backend = other.tensor_storage_backend;
        delta_sparse_mode = other.delta_sparse_mode;
        delta_sparse_accumulate = other.delta_sparse_accumulate;
        delta_sparse_threshold = other.delta_sparse_threshold;
        active_sparse_accum = other.active_sparse_accum;
        sparse_accum[0] = std::move(other.sparse_accum[0]);
        sparse_accum[1] = std::move(other.sparse_accum[1]);
        order_history = std::move(other.order_history);
        order_integrator = std::move(other.order_integrator);
        scratch = std::move(other.scratch);
        order_mode = other.order_mode;
        order_history_count = other.order_history_count;
        order_history_cursor = other.order_history_cursor;
        dirty_grid_x_req = other.dirty_grid_x_req;
        dirty_grid_y_req = other.dirty_grid_y_req;
        dirty_grid_x = other.dirty_grid_x;
        dirty_grid_y = other.dirty_grid_y;
        dirty_threshold = other.dirty_threshold;
        dirty_mask = std::move(other.dirty_mask);
        dirty_seq.store(other.dirty_seq.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
    EdgeTensorFifo(const EdgeTensorFifo&) = delete;
    EdgeTensorFifo& operator=(const EdgeTensorFifo&) = delete;

    void configure_default() { configure(std::vector<int32_t>{1}, 16, 0); }

    void configure(const std::vector<int32_t>& dims,
                   size_t slot_count,
                   size_t topk,
                   size_t elem_size_bytes = sizeof(float),
                   int32_t type_id_in = -1,
                   nodus::tensors::TensorLayout layout_in = nodus::tensors::TensorLayout::Dense,
                   nodus::tensors::TensorDType dtype_in = nodus::tensors::TensorDType::Unknown) {
        if (!impl) impl.reset(new Impl());
        layout = layout_in;
        dtype = dtype_in;
        shape = dims;
        if (shape.empty()) shape.push_back(1);
        size_t stride_local = 1;
        for (int32_t d : shape) {
            stride_local *= static_cast<size_t>(std::max<int32_t>(1, d));
        }
        impl->stride = std::max<size_t>(1, stride_local);
        impl->slots = std::max<size_t>(1, slot_count);
        impl->top_k = topk;
        impl->elem_size = std::max<size_t>(1, elem_size_bytes);
        impl->type_id = type_id_in;

        last_sample.assign(impl->stride, 0.0f);
        last_sample_valid = false;
        last_sample_bytes.assign(impl->stride * impl->elem_size, 0u);
        last_sample_bytes_valid = false;
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_mode = 0;
        order_history_count = 0;
        order_history_cursor = 0;
        refresh_dirty_grid();

        size_t total_bytes = impl->stride * impl->slots * impl->elem_size;
        impl->release_storage();
        // Allocate a host-backed storage buffer by default. If a backend was
        // previously attached via set_backend(), use that backend to host
        // the allocation if the backend exposes alloc, otherwise fall back
        // to creating a host buffer handle.
        if (impl->backend) {
            const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(impl->backend);
            if (bvt && bvt->alloc) {
                impl->storage_handle = bvt->alloc(impl->backend, total_bytes, /*alignment=*/0);
            } else {
                // backend doesn't provide alloc: create a generic host buffer
                impl->storage_handle = gp_mem_backend_create_host(total_bytes);
            }
        } else {
            impl->storage_handle = gp_mem_backend_create_host(total_bytes);
        }
        impl->slot_seq.reset(new std::atomic<uint64_t>[impl->slots]);
        if (impl->storage_handle) {
            const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(impl->storage_handle);
            if (sh_vt && sh_vt->copy_to_backend) {
                std::vector<uint8_t> zeros;
                try { zeros.resize(total_bytes); } catch(...) { }
                if (zeros.size() == total_bytes) sh_vt->copy_to_backend(impl->storage_handle, 0, zeros.data(), total_bytes);
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (m) {
                    std::memset(m, 0, total_bytes);
                    gp_mem_backend_unmap(impl->storage_handle);
                }
            }
        }
        for (size_t i = 0; i < impl->slots; ++i) impl->slot_seq[i].store(0, std::memory_order_relaxed);
        impl->write_seq.store(0, std::memory_order_relaxed);
        impl->writer.store(0, std::memory_order_relaxed);
        impl->write_friction.store(0.0f, std::memory_order_relaxed);
        impl->read_friction.store(0.0f, std::memory_order_relaxed);
        impl->last_write_seq.store(0, std::memory_order_relaxed);
        impl->last_read_seq.store(0, std::memory_order_relaxed);
        impl->last_write_region.store(-1, std::memory_order_relaxed);
        impl->last_read_region.store(-1, std::memory_order_relaxed);
        impl->write_phase.store(0.0f, std::memory_order_relaxed);
        impl->read_phase.store(0.0f, std::memory_order_relaxed);
        for (size_t i = 0; i < kMaxReaders; ++i) {
            impl->readers[i].key.store(0, std::memory_order_relaxed);
            impl->readers[i].seq.store(0, std::memory_order_relaxed);
        }
        impl->configured = true;
    }

    // Attach a memory backend to this FIFO. The FIFO will consult backend
    // capabilities (e.g. BYREF safety) when deciding how to expose pointer
    // semantics to consumers.
    void set_backend(gp_mem_backend_handle_t h) {
        if (!impl) return;
        impl->backend = h;
    }
    gp_mem_backend_handle_t get_backend() const {
        if (!impl) return nullptr;
        return impl->backend;
    }

    size_t elem_count() const { return impl ? impl->stride : 0; }

    void set_delta_mode(bool enabled) { delta_mode = enabled; }
    void set_delta_sparse_mode(bool enabled, bool accumulate, float threshold) {
        delta_sparse_mode = enabled;
        delta_sparse_accumulate = enabled && accumulate;
        delta_sparse_threshold = std::max(0.0f, threshold);
    }
    bool delta_sparse_enabled() const { return delta_sparse_mode; }
    bool delta_sparse_accumulate_enabled() const { return delta_sparse_accumulate; }
    void set_tensor_storage(nodus::tensors::AbstractTensorHandle handle,
                            nodus::tensors::TensorBackend* backend) {
        tensor_storage = handle;
        tensor_storage_backend = backend;
    }
    bool has_tensor_storage() const {
        return nodus::tensors::abstract_tensor_handle_is_valid(tensor_storage) && tensor_storage_backend != nullptr;
    }
    void set_order_mode(int32_t mode) {
        order_mode = std::clamp(mode, -kMaxOrder, kMaxOrder);
        order_history_count = 0;
        order_history_cursor = 0;
        std::fill(order_history.begin(), order_history.end(), 0.0f);
        std::fill(order_integrator.begin(), order_integrator.end(), 0.0f);
        last_sample_valid = false;
    }
    void set_dirty_grid(int32_t grid_x, int32_t grid_y, float threshold) {
        dirty_grid_x_req = grid_x;
        dirty_grid_y_req = grid_y;
        dirty_threshold = std::max(0.0f, threshold);
        refresh_dirty_grid();
    }
    int32_t dirty_mask_copy(uint8_t* out_mask, int32_t out_len, int32_t* out_grid_x, int32_t* out_grid_y, uint64_t* out_seq) {
        if (out_grid_x) *out_grid_x = dirty_grid_x;
        if (out_grid_y) *out_grid_y = dirty_grid_y;
        if (out_seq) *out_seq = dirty_seq.load(std::memory_order_relaxed);
        if (!out_mask || out_len <= 0) return 0;
        std::lock_guard<std::mutex> lock(dirty_mu);
        int32_t need = static_cast<int32_t>(dirty_mask.size());
        int32_t copy_len = std::min(out_len, need);
        if (copy_len > 0) {
            std::memcpy(out_mask, dirty_mask.data(), static_cast<size_t>(copy_len));
        }
        return copy_len;
    }

    void update_last_sample_bytes(const void* sample_bytes, size_t bytes) {
        if (!impl) return;
        size_t need = impl->stride * impl->elem_size;
        if (!sample_bytes || bytes < need) {
            last_sample_bytes_valid = false;
            last_sample_valid = false;
            return;
        }
        if (last_sample_bytes.size() != need) last_sample_bytes.resize(need);
        std::memcpy(last_sample_bytes.data(), sample_bytes, need);
        last_sample_bytes_valid = true;
        if (impl->elem_size == sizeof(float)) {
            if (last_sample.size() != impl->stride) last_sample.resize(impl->stride);
            std::memcpy(last_sample.data(), sample_bytes, impl->stride * sizeof(float));
            last_sample_valid = true;
        } else {
            last_sample_valid = false;
        }
    }

    bool compute_sparse_delta(const void* sample_bytes,
                              size_t bytes,
                              std::vector<uint32_t>& out_linear,
                              std::vector<uint8_t>& out_values) {
        if (!impl || !sample_bytes) return false;
        size_t elem_size = impl->elem_size;
        size_t count = impl->stride;
        size_t need = count * elem_size;
        if (bytes < need) return false;
        out_linear.clear();
        out_values.clear();
        out_linear.reserve(count);
        out_values.reserve(need);
        const uint8_t* cur = reinterpret_cast<const uint8_t*>(sample_bytes);
        const uint8_t* prev = last_sample_bytes_valid ? last_sample_bytes.data() : nullptr;
        bool use_f32 = (elem_size == sizeof(float));
        bool use_f64 = (elem_size == sizeof(double));
        for (size_t i = 0; i < count; ++i) {
            bool changed = false;
            const size_t off = i * elem_size;
            if (!prev) {
                changed = true;
            } else if (use_f32) {
                float a = 0.0f;
                float b = 0.0f;
                std::memcpy(&a, cur + off, sizeof(float));
                std::memcpy(&b, prev + off, sizeof(float));
                changed = std::fabs(static_cast<double>(a - b)) > static_cast<double>(delta_sparse_threshold);
            } else if (use_f64) {
                double a = 0.0;
                double b = 0.0;
                std::memcpy(&a, cur + off, sizeof(double));
                std::memcpy(&b, prev + off, sizeof(double));
                changed = std::fabs(a - b) > static_cast<double>(delta_sparse_threshold);
            } else {
                for (size_t b = 0; b < elem_size; ++b) {
                    if (cur[off + b] != prev[off + b]) {
                        changed = true;
                        break;
                    }
                }
            }
            if (changed) {
                out_linear.push_back(static_cast<uint32_t>(i));
                out_values.insert(out_values.end(), cur + off, cur + off + elem_size);
            }
        }
        return true;
    }

    bool build_sparse_from_linear(const std::vector<uint32_t>& linear,
                                  const std::vector<uint8_t>& values_bytes,
                                  nodus::tensors::COOMatrix* out_sparse) {
        if (!out_sparse || !impl) return false;
        const uint32_t nnz = static_cast<uint32_t>(linear.size());
        nodus::tensors::TensorShape tshape{};
        tshape.dims.reserve(shape.size());
        for (int32_t d : shape) tshape.dims.push_back(static_cast<uint32_t>(std::max(1, d)));
        auto* backend = &nodus::tensors::in_memory_backend_singleton();
        nodus::tensors::COOMatrix tmp = nodus::tensors::COOMatrix::create(
            tshape,
            nnz,
            dtype,
            backend,
            nodus::tensors::TensorDType::U32,
            nodus::tensors::CooIndexLayout::RowMajor);
        if (!tmp.valid()) return false;
        auto* mem_backend = backend;
        void* idx_data = nullptr;
        size_t idx_bytes = 0;
        if (!mem_backend->map(tmp.indices.handle(), &idx_data, &idx_bytes)) return false;
        void* val_data = nullptr;
        size_t val_bytes = 0;
        if (!mem_backend->map(tmp.values.handle(), &val_data, &val_bytes)) {
            mem_backend->unmap(tmp.indices.handle());
            return false;
        }
        auto* idx_out = static_cast<uint32_t*>(idx_data);
        auto* val_out = static_cast<uint8_t*>(val_data);
        const uint32_t rank = static_cast<uint32_t>(tshape.dims.size());
        std::vector<uint32_t> coords(rank);
        for (uint32_t i = 0; i < nnz; ++i) {
            uint32_t lin = linear[i];
            uint64_t idx = lin;
            for (uint32_t r = rank; r-- > 0;) {
                uint32_t dim = tshape.dims[r];
                coords[r] = dim ? static_cast<uint32_t>(idx % dim) : 0u;
                idx /= dim ? dim : 1u;
            }
            for (uint32_t r = 0; r < rank; ++r) {
                idx_out[i * rank + r] = coords[r];
            }
            size_t off = static_cast<size_t>(i) * impl->elem_size;
            if (off + impl->elem_size <= values_bytes.size()) {
                std::memcpy(val_out + off, values_bytes.data() + off, impl->elem_size);
            }
        }
        mem_backend->unmap(tmp.indices.handle());
        mem_backend->unmap(tmp.values.handle());
        *out_sparse = std::move(tmp);
        return true;
    }

    void accumulate_sparse_delta(const std::vector<uint32_t>& linear,
                                 const std::vector<uint8_t>& values_bytes) {
        if (!impl) return;
        std::lock_guard<std::mutex> lock(sparse_mu);
        SparseAccumState& acc = sparse_accum[active_sparse_accum];
        const size_t elem_size = impl->elem_size;
        for (size_t i = 0; i < linear.size(); ++i) {
            uint32_t lin = linear[i];
            std::vector<uint8_t> buf(elem_size);
            size_t off = i * elem_size;
            if (off + elem_size <= values_bytes.size()) {
                std::memcpy(buf.data(), values_bytes.data() + off, elem_size);
            }
            acc.values[lin] = std::move(buf);
        }
    }

    bool take_sparse_accum(nodus::tensors::COOMatrix* out_sparse) {
        if (!out_sparse || !impl) return false;
        SparseAccumState local;
        {
            std::lock_guard<std::mutex> lock(sparse_mu);
            int take_idx = active_sparse_accum;
            active_sparse_accum = 1 - active_sparse_accum;
            sparse_accum[active_sparse_accum].values.clear();
            local.values.swap(sparse_accum[take_idx].values);
        }
        std::vector<uint32_t> linear;
        std::vector<uint8_t> values_bytes;
        linear.reserve(local.values.size());
        values_bytes.reserve(local.values.size() * impl->elem_size);
        for (const auto& kv : local.values) {
            linear.push_back(kv.first);
            const auto& buf = kv.second;
            values_bytes.insert(values_bytes.end(), buf.begin(), buf.end());
        }
        return build_sparse_from_linear(linear, values_bytes, out_sparse);
    }

private:
    void refresh_dirty_grid() {
        int32_t gx = dirty_grid_x_req;
        int32_t gy = dirty_grid_y_req;
        if (gx <= 0 || gy <= 0 || !impl) {
            std::lock_guard<std::mutex> lock(dirty_mu);
            dirty_grid_x = 0;
            dirty_grid_y = 0;
            dirty_mask.clear();
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        size_t width = 0;
        if (!shape.empty()) width = static_cast<size_t>(std::max<int32_t>(1, shape.back()));
        if (width == 0) width = impl->stride;
        size_t height = (width > 0) ? ((impl->stride + width - 1) / width) : 0;
        int32_t max_x = static_cast<int32_t>(std::max<size_t>(1, width));
        int32_t max_y = static_cast<int32_t>(std::max<size_t>(1, height));
        int32_t min_x = (max_x >= 2) ? 2 : 1;
        int32_t min_y = (max_y >= 2) ? 2 : 1;
        gx = std::clamp(gx, min_x, max_x);
        gy = std::clamp(gy, min_y, max_y);
        std::lock_guard<std::mutex> lock(dirty_mu);
        dirty_grid_x = gx;
        dirty_grid_y = gy;
        dirty_mask.assign(static_cast<size_t>(gx) * static_cast<size_t>(gy), 0u);
        dirty_seq.fetch_add(1, std::memory_order_relaxed);
    }

    void update_dirty_mask(const void* current_sample, size_t bytes) {
        if (!impl) return;
        if (dirty_grid_x <= 0 || dirty_grid_y <= 0 || !current_sample) return;
        size_t width = 0;
        if (!shape.empty()) width = static_cast<size_t>(std::max<int32_t>(1, shape.back()));
        if (width == 0) width = impl->stride;
        size_t height = (width > 0) ? ((impl->stride + width - 1) / width) : 0;
        if (width == 0 || height == 0) return;
        int32_t gx = dirty_grid_x;
        int32_t gy = dirty_grid_y;
        std::lock_guard<std::mutex> lock(dirty_mu);
        if (dirty_mask.size() != static_cast<size_t>(gx) * static_cast<size_t>(gy)) {
            dirty_mask.assign(static_cast<size_t>(gx) * static_cast<size_t>(gy), 0u);
        }
        if (!last_sample_bytes_valid || last_sample_bytes.size() != bytes) {
            std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        const size_t elem_size = impl->elem_size;
        const bool use_float = (elem_size == sizeof(float));
        const float* cur_f = reinterpret_cast<const float*>(current_sample);
        const float* prev_f = nullptr;
        const uint8_t* cur_b = reinterpret_cast<const uint8_t*>(current_sample);
        const uint8_t* prev_b = reinterpret_cast<const uint8_t*>(last_sample_bytes.data());
        if (use_float) {
            if (!last_sample_valid || last_sample.size() != impl->stride) {
                std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
                dirty_seq.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            prev_f = last_sample.data();
        }
        auto seg_bounds = [](int32_t idx, int32_t count, size_t size, size_t* out_start, size_t* out_end) {
            if (!out_start || !out_end) return;
            if (count <= 0 || size == 0) {
                *out_start = 0;
                *out_end = 0;
                return;
            }
            size_t base = size / static_cast<size_t>(count);
            size_t rem = size % static_cast<size_t>(count);
            size_t start = static_cast<size_t>(idx) * base + static_cast<size_t>(std::min<int32_t>(idx, static_cast<int32_t>(rem)));
            size_t len = base + ((static_cast<size_t>(idx) < rem) ? 1u : 0u);
            *out_start = start;
            *out_end = start + len;
        };
        float threshold = std::max(0.0f, dirty_threshold);
        size_t stride_elems = impl->stride;
        for (int32_t sy = 0; sy < gy; ++sy) {
            size_t y0 = 0, y1 = 0;
            seg_bounds(sy, gy, height, &y0, &y1);
            for (int32_t sx = 0; sx < gx; ++sx) {
                size_t x0 = 0, x1 = 0;
                seg_bounds(sx, gx, width, &x0, &x1);
                bool dirty = false;
                for (size_t y = y0; y < y1 && !dirty; ++y) {
                    size_t row = y * width;
                    for (size_t x = x0; x < x1; ++x) {
                        size_t idx = row + x;
                        if (idx >= stride_elems) { dirty = false; break; }
                        if (use_float) {
                            float dv = std::fabs(cur_f[idx] - prev_f[idx]);
                            if (dv > threshold) { dirty = true; break; }
                        } else {
                            size_t off = idx * elem_size;
                            bool diff = false;
                            for (size_t b = 0; b < elem_size; ++b) {
                                if (cur_b[off + b] != prev_b[off + b]) { diff = true; break; }
                            }
                            if (diff) { dirty = true; break; }
                        }
                    }
                }
                dirty_mask[static_cast<size_t>(sy) * static_cast<size_t>(gx) + static_cast<size_t>(sx)] = dirty ? 1u : 0u;
            }
        }
        dirty_seq.fetch_add(1, std::memory_order_relaxed);
    }

    void mark_dirty_all() {
        if (dirty_grid_x <= 0 || dirty_grid_y <= 0) return;
        std::lock_guard<std::mutex> lock(dirty_mu);
        if (!dirty_mask.empty()) {
            std::fill(dirty_mask.begin(), dirty_mask.end(), static_cast<uint8_t>(1));
            dirty_seq.fetch_add(1, std::memory_order_relaxed);
        }
    }
public:

    void maybe_claim_writer(uint64_t key) {
        if (!impl) return;
        uint64_t prev = impl->writer.load(std::memory_order_relaxed);
        if (prev == 0 || prev == key) {
            (void)impl->writer.compare_exchange_strong(prev, key, std::memory_order_relaxed);
        }
    }

    ReaderEntry* find_reader(uint64_t key) {
        if (!impl) return nullptr;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == key) return &impl->readers[i];
        }
        return nullptr;
    }

    void set_friction_regions(int32_t regions) {
        if (!impl) return;
        int32_t clamped = std::max(1, regions);
        impl->friction_regions.store(clamped, std::memory_order_relaxed);
    }

    bool subscribe(uint64_t key, bool start_at_head) {
        if (!impl) return false;
        if (key == 0) return false;
        if (find_reader(key)) return true;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            uint64_t expected = 0;
            if (!impl->readers[i].key.compare_exchange_strong(expected, key, std::memory_order_acq_rel)) continue;
            uint64_t head = impl->write_seq.load(std::memory_order_acquire);
            uint64_t start = head;
            if (!start_at_head) {
                uint64_t cap = static_cast<uint64_t>(impl->slots);
                start = (head > cap) ? (head - cap) : 0;
            }
            impl->readers[i].seq.store(start, std::memory_order_release);
            uint64_t prev = impl->last_read_seq.load(std::memory_order_relaxed);
            if (prev == 0) {
                impl->last_read_seq.store(start, std::memory_order_relaxed);
                if (impl->slots > 0) {
                    float phase = static_cast<float>(start % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
                    impl->read_phase.store(phase, std::memory_order_relaxed);
                }
            }
            impl->cv.notify_all();
            return true;
        }
        return false;
    }

    void unsubscribe(uint64_t key) {
        if (!impl || key == 0) return;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) != key) continue;
            impl->readers[i].key.store(0, std::memory_order_release);
            impl->cv.notify_all();
            return;
        }
    }

    uint64_t min_reader_seq(uint64_t head) const {
        if (!impl) return head;
        uint64_t m = head;
        bool any = false;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == 0) continue;
            uint64_t s = impl->readers[i].seq.load(std::memory_order_relaxed);
            m = std::min<uint64_t>(m, s);
            any = true;
        }
        return any ? m : head;
    }

    static float atomic_add(std::atomic<float>& v, float delta) {
        float cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur + delta, std::memory_order_relaxed)) {}
        return cur + delta;
    }

    static void atomic_scale(std::atomic<float>& v, float factor) {
        float cur = v.load(std::memory_order_relaxed);
        while (!v.compare_exchange_weak(cur, cur * factor, std::memory_order_relaxed)) {}
    }

    void note_activity(uint64_t seq,
                       std::atomic<uint64_t>& last_seq,
                       std::atomic<int32_t>& last_region,
                       std::atomic<float>& friction,
                       std::atomic<float>& phase) {
        if (!impl || impl->slots == 0) return;
        uint64_t prev = last_seq.exchange(seq, std::memory_order_relaxed);
        if (seq <= prev) return;
        uint64_t delta = seq - prev;
        float inc = static_cast<float>(delta) / static_cast<float>(impl->slots);
        uint64_t slot = seq % static_cast<uint64_t>(impl->slots);
        int32_t regions = impl->friction_regions.load(std::memory_order_relaxed);
        if (regions > 0) {
            int32_t region = static_cast<int32_t>((slot * static_cast<uint64_t>(regions)) / static_cast<uint64_t>(impl->slots));
            int32_t prev_region = last_region.exchange(region, std::memory_order_relaxed);
            if (prev_region >= 0 && prev_region != region) {
                int32_t diff = std::abs(region - prev_region);
                if (diff > regions / 2) diff = regions - diff;
                inc += 2.0f * static_cast<float>(std::max(1, diff));
            }
        }
        float phase_val = static_cast<float>(slot) / static_cast<float>(impl->slots);
        phase.store(phase_val, std::memory_order_relaxed);
        atomic_add(friction, inc);
    }

    void tick_friction(float dt, float half_life) {
        if (!impl) return;
        if (half_life <= 0.0f || dt <= 0.0f) return;
        float factor = std::pow(0.5f, dt / half_life);
        atomic_scale(impl->write_friction, factor);
        atomic_scale(impl->read_friction, factor);
    }

    float write_friction() const { return impl ? impl->write_friction.load(std::memory_order_relaxed) : 0.0f; }
    float read_friction() const { return impl ? impl->read_friction.load(std::memory_order_relaxed) : 0.0f; }
    float write_phase() const { return impl ? impl->write_phase.load(std::memory_order_relaxed) : 0.0f; }
    float read_phase() const { return impl ? impl->read_phase.load(std::memory_order_relaxed) : 0.0f; }
    uint64_t write_seq() const { return impl ? impl->write_seq.load(std::memory_order_relaxed) : 0; }

    float phase_delta() const {
        if (!impl) return 0.0f;
        float w = impl->write_phase.load(std::memory_order_relaxed);
        float r = impl->read_phase.load(std::memory_order_relaxed);
        float delta = w - r;
        if (delta > 0.5f) delta -= 1.0f;
        if (delta < -0.5f) delta += 1.0f;
        return delta;
    }

    bool fill_state(uint64_t edge_id, float& out_fill, float& out_head_phase, float& out_tail_phase) const {
        (void)edge_id; // retained for source compatibility with table callers
        out_fill = 0.0f;
        out_head_phase = 0.0f;
        out_tail_phase = 0.0f;
        if (!impl || impl->slots == 0) return false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        uint64_t used = (head >= min_seq) ? (head - min_seq) : 0;
        float fill = static_cast<float>(used) / static_cast<float>(impl->slots);
        out_fill = std::clamp(fill, 0.0f, 1.0f);
        out_head_phase = static_cast<float>(head % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
        out_tail_phase = static_cast<float>(min_seq % static_cast<uint64_t>(impl->slots)) / static_cast<float>(impl->slots);
        return true;
    }

    bool is_full(uint64_t seq, uint64_t edge_id = 0) const {
        (void)edge_id; // retained for source compatibility with table callers
        if (!impl) return false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        uint64_t used = (seq >= min_seq) ? (seq - min_seq) : 0;
        return used >= static_cast<uint64_t>(impl->slots);
    }

    bool ensure_space_for_write(uint64_t seq, bool* out_dropped, uint64_t edge_id = 0) {
        (void)edge_id; // retained for source compatibility with table callers
        if (!impl) return false;
        if (out_dropped) *out_dropped = false;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t min_seq = min_reader_seq(head);
        uint64_t used = (seq >= min_seq) ? (seq - min_seq) : 0;
        if (used < static_cast<uint64_t>(impl->slots)) return true;
        if (impl->top_k == 0) return false;
        uint64_t window = std::min<uint64_t>(static_cast<uint64_t>(impl->top_k), static_cast<uint64_t>(impl->slots));
        uint64_t target_min = (window > 0 && seq >= (window - 1)) ? (seq - (window - 1)) : 0;
        if (target_min <= min_seq) return false;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == 0) continue;
            uint64_t s = impl->readers[i].seq.load(std::memory_order_relaxed);
            if (s < target_min) impl->readers[i].seq.store(target_min, std::memory_order_relaxed);
        }
        if (out_dropped) *out_dropped = true;
        min_seq = min_reader_seq(head);
        used = (seq >= min_seq) ? (seq - min_seq) : 0;
        return used < static_cast<uint64_t>(impl->slots);
    }

    bool push(uint64_t edge_id, uint64_t writer_id, const void* sample_bytes, size_t sample_len_bytes, bool* out_dropped) {
        if (!impl || !impl->configured) return false;
        if (!sample_bytes) return false;
        if (sample_len_bytes != impl->stride * impl->elem_size) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        const uint8_t* sample_bytes_u = reinterpret_cast<const uint8_t*>(sample_bytes);
        const float* effective_sample_f = nullptr;
        std::vector<float> temp_scratch;
        if (impl->elem_size == sizeof(float) && order_mode != 0 && scratch.size() == impl->stride) {
            const float* sample_f = reinterpret_cast<const float*>(sample_bytes);
            int32_t mode = std::clamp(order_mode, -kMaxOrder, kMaxOrder);
            if (mode > 0) {
                for (size_t i = 0; i < impl->stride; ++i) {
                    float acc = sample_f[i];
                    order_integrator[i] += acc;
                    for (int32_t level = 1; level < mode; ++level) {
                        size_t idx = static_cast<size_t>(level) * impl->stride + i;
                        size_t prev = static_cast<size_t>(level - 1) * impl->stride + i;
                        order_integrator[idx] += order_integrator[prev];
                    }
                    scratch[i] = order_integrator[static_cast<size_t>(mode - 1) * impl->stride + i];
                }
                effective_sample_f = scratch.data();
            } else {
                int32_t order = -mode;
                if (order_history_count >= order) {
                    for (size_t i = 0; i < impl->stride; ++i) {
                        float sum = sample_f[i];
                        int32_t coef = 1;
                        for (int32_t k = 1; k <= order; ++k) {
                            coef = (coef * (order - (k - 1))) / k;
                            int32_t idx = order_history_cursor - k;
                            if (idx < 0) idx += kMaxOrder;
                            float prev = order_history[static_cast<size_t>(idx) * impl->stride + i];
                            float sign = (k % 2 == 0) ? 1.0f : -1.0f;
                            sum += sign * static_cast<float>(coef) * prev;
                        }
                        scratch[i] = sum;
                    }
                    effective_sample_f = scratch.data();
                } else {
                    temp_scratch.resize(impl->stride);
                    std::memcpy(temp_scratch.data(), sample_bytes, impl->stride * impl->elem_size);
                    effective_sample_f = reinterpret_cast<const float*>(temp_scratch.data());
                }
            }
        }

        if (delta_mode && last_sample_valid) {
            size_t byte_count = impl->stride * impl->elem_size;
            if (last_sample.size() * sizeof(float) == byte_count) {
                if (std::memcmp(last_sample.data(), sample_bytes, byte_count) == 0) {
                    if (out_dropped) *out_dropped = 0;
                    return true;
                }
            }
        }

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 0;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        size_t bytes = impl->stride * impl->elem_size;
        const void* effective_sample_ptr = (effective_sample_f ? reinterpret_cast<const void*>(effective_sample_f) : reinterpret_cast<const void*>(sample_bytes_u));
        update_dirty_mask(effective_sample_ptr, bytes);
        // Write into the FIFO's backend-owned storage buffer.
        gp_mem_backend_handle_t sh = impl->storage_handle;
        if (!sh) {
            if (out_dropped) *out_dropped = 0;
            return false;
        }
        const gp_mem_backend_vtable_t* sh_vt = gp_mem_backend_get_vtable(sh);
        size_t offset = slot * bytes;
        if (sh_vt && sh_vt->copy_to_backend) {
            if (!sh_vt->copy_to_backend(sh, offset, effective_sample_ptr, bytes)) {
                if (out_dropped) *out_dropped = 0;
                return false;
            }
        } else {
            void* m = gp_mem_backend_map_or_null(sh);
            if (!m) {
                if (out_dropped) *out_dropped = 0;
                return false;
            }
            std::memcpy(reinterpret_cast<uint8_t*>(m) + offset, effective_sample_ptr, bytes);
            gp_mem_backend_unmap(sh);
        }
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        if (!order_history.empty()) {
            size_t base = static_cast<size_t>(order_history_cursor) * impl->stride;
            if (impl->elem_size == sizeof(float)) std::memcpy(order_history.data() + base, sample_bytes, impl->stride * impl->elem_size);
            order_history_cursor = (order_history_cursor + 1) % kMaxOrder;
            order_history_count = std::min(order_history_count + 1, kMaxOrder);
        }
        if (last_sample.size() == impl->stride) {
            if (impl->elem_size == sizeof(float)) {
                std::memcpy(last_sample.data(), (effective_sample_f ? effective_sample_f : reinterpret_cast<const float*>(sample_bytes)), impl->stride * sizeof(float));
                last_sample_valid = true;
            } else {
                // For non-float element sizes we keep last_sample invalid (or zeroed)
                last_sample_valid = false;
            }
        }
        if (last_sample_bytes.size() == impl->stride * impl->elem_size) {
            std::memcpy(last_sample_bytes.data(), effective_sample_ptr, impl->stride * impl->elem_size);
            last_sample_bytes_valid = true;
        } else {
            last_sample_bytes_valid = false;
        }
        impl->cv.notify_all();
        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    // Pointer-mode push: publish opaque pointer values into FIFO slots.
    bool push_ptr(uint64_t edge_id, uint64_t writer_id, void* ptr, bool* out_dropped) {
        if (!impl || !impl->configured) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 0;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        // Write the pointer bytes into the slot's byte storage so pointers travel
        // through the same stride-based FIFO storage as float samples. The
        // consumer is expected to interpret the slot according to the edge
        // metadata (pointer vs float).
        size_t bytes = impl->stride * impl->elem_size;
        // Write pointer bytes into storage buffer
        gp_mem_backend_handle_t sh_ptr = impl->storage_handle;
        if (!sh_ptr) { if (out_dropped) *out_dropped = 0; return false; }
        const gp_mem_backend_vtable_t* shp_vt = gp_mem_backend_get_vtable(sh_ptr);
        size_t poff = slot * bytes;
        size_t pbytes = std::min<size_t>(sizeof(void*), bytes);
        if (shp_vt && shp_vt->copy_to_backend) {
            if (!shp_vt->copy_to_backend(sh_ptr, poff, &ptr, pbytes)) { if (out_dropped) *out_dropped = 0; return false; }
        } else {
            void* m = gp_mem_backend_map_or_null(sh_ptr);
            if (!m) { if (out_dropped) *out_dropped = 0; return false; }
            std::memset(reinterpret_cast<uint8_t*>(m) + poff, 0, bytes);
            std::memcpy(reinterpret_cast<uint8_t*>(m) + poff, &ptr, pbytes);
            gp_mem_backend_unmap(sh_ptr);
        }
        impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
        impl->write_seq.store(seq + 1, std::memory_order_release);
        note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
        impl->cv.notify_all();
        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    // Push a single typed element into the FIFO by popping it from a
    // RawStackFrame source. This attempts an optimized backend-to-backend
    // transfer when both source frame and FIFO storage expose backends and
    // fallbacks to a host-mediated copy otherwise. On success the source
    // frame has the element removed (popped). Returns true on success.
    bool push_from_frame(uint64_t edge_id, uint64_t writer_id, RawStackFrame* src_frame, int32_t type_id, bool* out_dropped) {
        if (!impl || !impl->configured || !src_frame) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound == 0) {
            (void)impl->writer.compare_exchange_strong(bound, writer_id, std::memory_order_relaxed);
        }
        bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;

        uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
        bool dropped = false;
        if (!ensure_space_for_write(seq, &dropped, edge_id)) {
            if (out_dropped) *out_dropped = 0;
            return false;
        }

        size_t slot = static_cast<size_t>(seq % static_cast<uint64_t>(impl->slots));
        size_t bytes = impl->stride * impl->elem_size;
        size_t dst_offset = slot * bytes;

        // Try direct backend-to-backend via raw_stack helper which knows how
        // to pop from the frame and write into a destination backend.
        if (src_frame->backend && impl->storage_handle) {
            if (gp_raw_stack_frame_pop_into_backend(src_frame, impl->storage_handle, dst_offset, type_id)) {
                impl->slot_seq[slot].store(seq + 1, std::memory_order_release);
                impl->write_seq.store(seq + 1, std::memory_order_release);
                note_activity(seq + 1, impl->last_write_seq, impl->last_write_region, impl->write_friction, impl->write_phase);
                impl->cv.notify_all();
                mark_dirty_all();
                last_sample_valid = false;
                last_sample_bytes_valid = false;
                if (out_dropped) *out_dropped = dropped ? 1 : 0;
                return true;
            }
        }

        // Fallback host-mediated path: read element bytes into host tmp,
        // push them into FIFO storage via existing push(), and then pop the
        // element from the source frame. This duplicates a transfer but
        // keeps correctness when optimized path unavailable.
        const ValueType* vt = ValueTypeRegistry::global().get(type_id);
        if (!vt || vt->size == 0) return false;
        size_t need = vt->size;
        std::vector<uint8_t> tmp;
        try { tmp.resize(bytes); } catch(...) { return false; }
        std::memset(tmp.data(), 0, bytes);

        // Read source element into tmp_head without popping yet.
        bool read_ok = false;
        if (src_frame->backend) {
            const gp_mem_backend_vtable_t* src_vt = gp_mem_backend_get_vtable(src_frame->backend);
            size_t start = (src_frame->byte_count >= need) ? (src_frame->byte_count - need) : 0;
            if (src_vt && src_vt->copy_from_backend) {
                if (src_vt->copy_from_backend(src_frame->backend, start, tmp.data(), need)) read_ok = true;
            }
            if (!read_ok) {
                void* m = gp_mem_backend_map_or_null(src_frame->backend);
                if (m) {
                    std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(m) + start, need);
                    gp_mem_backend_unmap(src_frame->backend);
                    read_ok = true;
                }
            }
        }
        if (!read_ok) return false;

        // Use existing push() to write tmp into FIFO storage (will copy-to-backend)
        if (!push(edge_id, writer_id, tmp.data(), bytes, &dropped)) return false;

        // Now remove the element from source frame (pop) to reflect transfer.
        // Pop into a throwaway buffer.
        std::vector<uint8_t> throwaway;
        try { throwaway.resize(need); } catch(...) { return false; }
        if (!raw_stack_pop_block(*src_frame, throwaway.data(), 1, type_id)) return false;

        if (out_dropped) *out_dropped = dropped ? 1 : 0;
        return true;
    }

    bool pop_ptr(uint64_t reader_id, void** out_ptr) {
        if (!impl || !impl->configured) return false;
        if (!out_ptr) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        // Read pointer bytes from the slot's storage and return as opaque pointer.
        size_t bytes = impl->stride * impl->elem_size;
        void* p = nullptr;
        gp_mem_backend_handle_t shr = impl->storage_handle;
        if (!shr) return false;
        const gp_mem_backend_vtable_t* shr_vt = gp_mem_backend_get_vtable(shr);
        size_t rbytes = std::min<size_t>(sizeof(void*), bytes);
        if (shr_vt && shr_vt->copy_from_backend) {
            size_t roff = slot * bytes;
            if (!shr_vt->copy_from_backend(shr, roff, &p, rbytes)) return false;
        } else {
            void* m = gp_mem_backend_map_or_null(shr);
            if (!m) return false;
            std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, rbytes);
            gp_mem_backend_unmap(shr);
        }
        *out_ptr = p;
        r->seq.store(rseq + 1, std::memory_order_relaxed);
        note_activity(rseq + 1, impl->last_read_seq, impl->last_read_region, impl->read_friction, impl->read_phase);
        impl->cv.notify_all();
        return true;
    }

    // Non-destructive peek for pointer slots: read opaque pointer bytes
    // without advancing the reader sequence. Returns true if a pointer
    // was available and copied into out_ptr.
    bool peek_ptr(uint64_t reader_id, void** out_ptr) {
        if (!impl || !impl->configured) return false;
        if (!out_ptr) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        size_t bytes = impl->stride * impl->elem_size;
        void* p = nullptr;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                void* tmp = nullptr;
                if (!vt->copy_from_backend(impl->backend, offset, &tmp, std::min<size_t>(sizeof(void*), bytes))) return false;
                p = tmp;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, std::min<size_t>(sizeof(void*), bytes));
                gp_mem_backend_unmap(impl->storage_handle);
            }
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                std::memcpy(&p, reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size, std::min<size_t>(sizeof(void*), bytes));
                gp_mem_backend_unmap(impl->storage_handle);
            }
        *out_ptr = p;
        return true;
    }

    // Blocking pointer pop: wait until a pointer entry is available then
    // read and advance the reader. Returns true on success.
    bool pop_ptr_blocking(uint64_t reader_id, void** out_ptr, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!find_reader(reader_id)) return false;
        if (!out_ptr) return false;
        if (timeout_ms == 0) return pop_ptr(reader_id, out_ptr);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = pop_ptr(reader_id, out_ptr);
            lk.lock();
            if (ok) return true;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    bool pop(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written_bytes) {
        out_written_bytes = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        size_t bytes = impl->stride * impl->elem_size;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                if (!vt->copy_from_backend(impl->backend, offset, out_sample_bytes, bytes)) return false;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
                std::memcpy(out_sample_bytes, src, bytes);
                gp_mem_backend_unmap(impl->storage_handle);
            }
        } else {
            void* m = gp_mem_backend_map_or_null(impl->storage_handle);
            if (!m) return false;
            uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
            std::memcpy(out_sample_bytes, src, bytes);
            gp_mem_backend_unmap(impl->storage_handle);
        }
        r->seq.store(rseq + 1, std::memory_order_relaxed);
        note_activity(rseq + 1, impl->last_read_seq, impl->last_read_region, impl->read_friction, impl->read_phase);
        impl->cv.notify_all();
        out_written_bytes = impl->stride * impl->elem_size;
        return true;
    }

    // Non-destructive peek: copy the next available sample for `reader_id`
    // into `out_sample` without advancing the reader sequence. Returns true
    // if a sample was available and copied. Contract matches `pop` with the
    // requirement that out_cap >= impl->stride.
    bool peek(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written_bytes) {
        out_written_bytes = 0;
        if (!impl || !impl->configured) return false;
        ReaderEntry* r = find_reader(reader_id);
        if (!r) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;

        uint64_t rseq = r->seq.load(std::memory_order_relaxed);
        size_t slot = static_cast<size_t>(rseq % static_cast<uint64_t>(impl->slots));
        uint64_t observed = impl->slot_seq[slot].load(std::memory_order_acquire);
        if (observed != (rseq + 1)) return false;

        size_t bytes = impl->stride * impl->elem_size;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* vt = gp_mem_backend_get_vtable(impl->backend);
            if (vt && vt->copy_from_backend) {
                size_t offset = slot * bytes;
                if (!vt->copy_from_backend(impl->backend, offset, out_sample_bytes, bytes)) return false;
            } else {
                void* m = gp_mem_backend_map_or_null(impl->storage_handle);
                if (!m) return false;
                uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
                std::memcpy(out_sample_bytes, src, bytes);
                gp_mem_backend_unmap(impl->storage_handle);
            }
        } else {
            void* m = gp_mem_backend_map_or_null(impl->storage_handle);
            if (!m) return false;
            uint8_t* src = reinterpret_cast<uint8_t*>(m) + slot * impl->stride * impl->elem_size;
            std::memcpy(out_sample_bytes, src, bytes);
            gp_mem_backend_unmap(impl->storage_handle);
        }
        // Note: do NOT advance r->seq and do NOT call note_activity / notify.
        out_written_bytes = impl->stride * impl->elem_size;
        return true;
    }

    bool push_blocking(uint64_t edge_id, uint64_t writer_id, const void* sample_bytes, size_t sample_len_bytes, bool* out_dropped, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!sample_bytes || sample_len_bytes != impl->stride * impl->elem_size) return false;
        uint64_t bound = impl->writer.load(std::memory_order_relaxed);
        if (bound != 0 && bound != writer_id) return false;
        if (timeout_ms == 0) return push(edge_id, writer_id, sample_bytes, sample_len_bytes, out_dropped);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = push(edge_id, writer_id, sample_bytes, sample_len_bytes, out_dropped);
            lk.lock();
            if (ok) return true;
            uint64_t seq = impl->write_seq.load(std::memory_order_relaxed);
            if (!is_full(seq, edge_id)) return false;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    // Ensure the FIFO stride (in bytes) can accomodate at least `min_bytes` per-slot.
    // If the current stride is too small we grow the stride and re-layout existing
    // per-slot contents into the new stride. This attempts to preserve outstanding
    // samples so FIFOs can be reconfigured (e.g. when an edge becomes by-ref)
    // without destroying the FIFO object.
    bool ensure_stride_for_bytes(size_t min_bytes) {
        if (!impl || !impl->configured) return false;
        size_t cur_bytes = impl->stride * impl->elem_size;
        if (cur_bytes >= min_bytes) return true;
        size_t needed_elems = (min_bytes + impl->elem_size - 1) / impl->elem_size;
        size_t new_stride = std::max(impl->stride, needed_elems);

        // allocate a new backend buffer for the resized layout and copy
        size_t new_total = new_stride * impl->slots * impl->elem_size;
        gp_mem_backend_handle_t old_h = impl->storage_handle;
        gp_mem_backend_handle_t new_h = nullptr;
        if (impl->backend) {
            const gp_mem_backend_vtable_t* bvt = gp_mem_backend_get_vtable(impl->backend);
            if (bvt && bvt->alloc) new_h = bvt->alloc(impl->backend, new_total, /*alignment=*/0);
            else new_h = gp_mem_backend_create_host(new_total);
        } else {
            new_h = gp_mem_backend_create_host(new_total);
        }
        if (!new_h) return false;
        // zero initialize new buffer
        const gp_mem_backend_vtable_t* new_vt = gp_mem_backend_get_vtable(new_h);
        if (new_vt && new_vt->copy_to_backend) {
            std::vector<uint8_t> zeros;
            try { zeros.resize(new_total); } catch(...) { }
            if (zeros.size() == new_total) new_vt->copy_to_backend(new_h, 0, zeros.data(), new_total);
        } else {
            void* nm = gp_mem_backend_map_or_null(new_h);
            if (nm) { std::memset(nm, 0, new_total); gp_mem_backend_unmap(new_h); }
        }

        // copy per-slot from old_h -> new_h
        const gp_mem_backend_vtable_t* old_vt = gp_mem_backend_get_vtable(old_h);
        for (size_t s = 0; s < impl->slots; ++s) {
            size_t copy_bytes = impl->stride * impl->elem_size;
            size_t old_off = s * impl->stride * impl->elem_size;
            size_t new_off = s * new_stride * impl->elem_size;
            // attempt optimized transfer between buffers
            if (old_vt && new_vt && old_vt->copy_between_backends) {
                if (!old_vt->copy_between_backends(old_h, new_h, old_off, new_off, copy_bytes)) {
                    // fallback to staged per-slot copy
                    std::vector<uint8_t> tmp;
                    try { tmp.resize(copy_bytes); } catch(...) { gp_mem_backend_release(new_h); return false; }
                    if (old_vt && old_vt->copy_from_backend) {
                        if (!old_vt->copy_from_backend(old_h, old_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                    } else {
                        void* om = gp_mem_backend_map_or_null(old_h);
                        if (!om) { gp_mem_backend_release(new_h); return false; }
                        std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(om) + old_off, copy_bytes);
                        gp_mem_backend_unmap(old_h);
                    }
                    if (new_vt && new_vt->copy_to_backend) {
                        if (!new_vt->copy_to_backend(new_h, new_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                    } else {
                        void* nm = gp_mem_backend_map_or_null(new_h);
                        if (!nm) { gp_mem_backend_release(new_h); return false; }
                        std::memcpy(reinterpret_cast<uint8_t*>(nm) + new_off, tmp.data(), copy_bytes);
                        gp_mem_backend_unmap(new_h);
                    }
                }
            } else {
                // staged copy path
                std::vector<uint8_t> tmp;
                try { tmp.resize(copy_bytes); } catch(...) { gp_mem_backend_release(new_h); return false; }
                if (old_vt && old_vt->copy_from_backend) {
                    if (!old_vt->copy_from_backend(old_h, old_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                } else {
                    void* om = gp_mem_backend_map_or_null(old_h);
                    if (!om) { gp_mem_backend_release(new_h); return false; }
                    std::memcpy(tmp.data(), reinterpret_cast<uint8_t*>(om) + old_off, copy_bytes);
                    gp_mem_backend_unmap(old_h);
                }
                if (new_vt && new_vt->copy_to_backend) {
                    if (!new_vt->copy_to_backend(new_h, new_off, tmp.data(), copy_bytes)) { gp_mem_backend_release(new_h); return false; }
                } else {
                    void* nm = gp_mem_backend_map_or_null(new_h);
                    if (!nm) { gp_mem_backend_release(new_h); return false; }
                    std::memcpy(reinterpret_cast<uint8_t*>(nm) + new_off, tmp.data(), copy_bytes);
                    gp_mem_backend_unmap(new_h);
                }
            }
        }

        // free old storage handle and swap in new one
        if (old_vt && old_vt->free) old_vt->free(old_h); else gp_mem_backend_release(old_h);
        impl->storage_handle = new_h;
        impl->stride = new_stride;

        // update our cached scratch/last_sample sizes to match new stride
        last_sample.assign(impl->stride, 0.0f);
        scratch.assign(impl->stride, 0.0f);
        order_history.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        order_integrator.assign(static_cast<size_t>(kMaxOrder) * impl->stride, 0.0f);
        return true;
    }

    bool pop_blocking(uint64_t reader_id, void* out_sample_bytes, size_t out_cap_bytes, size_t& out_written, int timeout_ms) {
        if (!impl || !impl->configured) return false;
        if (!find_reader(reader_id)) return false;
        if (!out_sample_bytes || out_cap_bytes < impl->stride * impl->elem_size) return false;
        if (timeout_ms == 0) return pop(reader_id, out_sample_bytes, out_cap_bytes, out_written);
        using clock = std::chrono::steady_clock;
        auto deadline = (timeout_ms < 0) ? clock::time_point::max() : (clock::now() + std::chrono::milliseconds(timeout_ms));
        std::unique_lock<std::mutex> lk(impl->cv_mu);
        while (true) {
            lk.unlock();
            bool ok = pop(reader_id, out_sample_bytes, out_cap_bytes, out_written);
            lk.lock();
            if (ok) return true;
            if (timeout_ms < 0) {
                impl->cv.wait(lk);
            } else {
                if (impl->cv.wait_until(lk, deadline) == std::cv_status::timeout) return false;
            }
        }
    }

    uint64_t unread(uint64_t reader_id) const {
        if (!impl || !impl->configured) return 0;
        const ReaderEntry* r = nullptr;
        for (size_t i = 0; i < kMaxReaders; ++i) {
            if (impl->readers[i].key.load(std::memory_order_relaxed) == reader_id) { r = &impl->readers[i]; break; }
        }
        if (!r) return 0;
        uint64_t head = impl->write_seq.load(std::memory_order_acquire);
        uint64_t tail = r->seq.load(std::memory_order_relaxed);
        return (head >= tail) ? (head - tail) : 0;
    }

};
