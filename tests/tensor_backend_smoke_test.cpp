#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/in_memory_backend.h"
#include "common/tensors/abstraction/tensor_types.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

using namespace nodus::tensors;

static TensorDesc make_desc() {
    TensorDesc desc{};
    desc.dtype = TensorDType::F32;
    desc.shape.dims = {2, 3, 4};
    desc.layout = TensorLayout::Dense;
    return desc;
}

int main() {
    register_in_memory_backend(true);

    TensorDesc desc = make_desc();
    InMemoryBackend& backend = in_memory_backend_singleton();

    AbstractTensor t1 = AbstractTensor::create(desc, &backend);
    assert(t1.valid());
    assert(t1.backend() == &backend);
    assert(t1.desc().shape.element_count() == 24);

    InMemoryBackend::AllocationInfo info1{};
    assert(backend.get_allocation_info(t1.handle(), &info1));
    assert(info1.alive);
    assert(info1.data != nullptr);
    assert(info1.bucket != 0xFFFFu);

    t1.reset();
    assert(!t1.valid());

    AbstractTensor t2 = AbstractTensor::create(desc, &backend);
    assert(t2.valid());

    InMemoryBackend::AllocationInfo info2{};
    assert(backend.get_allocation_info(t2.handle(), &info2));
    assert(info2.alive);
    assert(info2.bucket == info1.bucket);
    assert(info2.data == info1.data);

    void* data = nullptr;
    size_t bytes = 0;
    assert(backend.map(t2.handle(), &data, &bytes));
    assert(data != nullptr);
    assert(bytes == info2.bytes);
    std::memset(data, 0xAB, bytes);
    backend.unmap(t2.handle());

    void* data2 = nullptr;
    size_t bytes2 = 0;
    assert(backend.map(t2.handle(), &data2, &bytes2));
    assert(bytes2 == bytes);
    const uint8_t* probe = static_cast<const uint8_t*>(data2);
    assert(probe[0] == 0xAB);
    assert(probe[bytes2 - 1] == 0xAB);
    backend.unmap(t2.handle());

    AbstractTensorHandle h = t2.handle();
    AbstractTensor wrapped = AbstractTensor::wrap(h, desc, &backend, false);
    assert(wrapped.valid());
    assert(wrapped.refresh_desc());

    std::cout << "tensor_backend_smoke_test: ok\n";
    return 0;
}
