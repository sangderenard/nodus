#include "common/thread_pool.h"
#include "common/tensors/abstraction/in_memory_backend.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using namespace nodus;
using namespace nodus::tensors;

static void write_job(const ThreadPool::Job& job, uint32_t /*worker_id*/) {
    auto* backend = static_cast<InMemoryBackend*>(job.arena.backend);
    void* data = nullptr;
    size_t bytes = 0;
    if (!backend || !backend->map(job.arena.handle, &data, &bytes)) return;
    if (job.arena.offset_bytes + sizeof(uint32_t) <= bytes) {
        auto* out = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(data) + job.arena.offset_bytes);
        *out = static_cast<uint32_t>(job.param0);
    }
    backend->unmap(job.arena.handle);
}

int main() {
    register_in_memory_backend(true);
    InMemoryBackend& backend = in_memory_backend_singleton();

    TensorDesc desc{};
    desc.dtype = TensorDType::U32;
    desc.layout = TensorLayout::Dense;
    desc.shape.dims = {16};
    AbstractTensor arena = AbstractTensor::create(desc, &backend);
    assert(arena.valid());

    ThreadPool pool(ThreadPool::Options{4, true});

    std::vector<ThreadPool::Job> jobs;
    jobs.reserve(16);
    for (uint32_t i = 0; i < 16; ++i) {
        ThreadPool::Job job{};
        job.fn = &write_job;
        job.param0 = i + 10;
        job.arena.handle = arena.handle();
        job.arena.backend = &backend;
        job.arena.offset_bytes = sizeof(uint32_t) * i;
        job.arena.size_bytes = sizeof(uint32_t);
        jobs.push_back(job);
    }

    auto batch = pool.submit_batch(jobs.data(), static_cast<uint32_t>(jobs.size()));
    batch->wait();

    void* data = nullptr;
    size_t bytes = 0;
    assert(backend.map(arena.handle(), &data, &bytes));
    assert(bytes >= sizeof(uint32_t) * 16);
    auto* values = static_cast<uint32_t*>(data);
    for (uint32_t i = 0; i < 16; ++i) {
        assert(values[i] == i + 10);
    }
    backend.unmap(arena.handle());

    std::cout << "thread_pool_test: ok\n";
    return 0;
}
