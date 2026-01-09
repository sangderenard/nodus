#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/tensors/abstraction/abstract_tensor.h"
#include "common/tensors/abstraction/tensor_backend.h"

namespace nodus {

struct TensorArenaView {
    nodus::tensors::AbstractTensorHandle handle{};
    nodus::tensors::TensorBackend* backend = nullptr;
    uint64_t offset_bytes = 0;
    uint64_t size_bytes = 0;
};

class JobBatch {
public:
    explicit JobBatch(uint32_t count);
    void wait();
    bool done() const;

private:
    friend class ThreadPool;
    void finish_one();

    std::atomic<uint32_t> remaining_{0};
    mutable std::mutex mu_;
    std::condition_variable cv_;
};

class ThreadPool {
public:
    struct Job;
    using JobFn = void (*)(const Job& job, uint32_t worker_id);

    struct Job {
        JobFn fn = nullptr;
        void* user = nullptr;
        uint64_t param0 = 0;
        uint64_t param1 = 0;
        const void* params = nullptr;
        uint32_t params_size = 0;
        TensorArenaView arena{};
        std::shared_ptr<JobBatch> batch;
    };

    struct Options {
        uint32_t thread_count = 0; // 0 = hardware_concurrency
        bool start_immediately = true;
    };

    explicit ThreadPool(const Options& options = Options{});
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void start();
    void stop();

    void enqueue(Job job);
    std::shared_ptr<JobBatch> submit_batch(const Job* jobs, uint32_t count);

    uint32_t thread_count() const { return thread_count_; }

private:
    void worker_loop(uint32_t worker_id);

    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};
    uint32_t thread_count_ = 0;
    std::vector<std::thread> workers_;
    std::deque<Job> queue_;
    std::mutex queue_mu_;
    std::condition_variable queue_cv_;
};

} // namespace nodus
