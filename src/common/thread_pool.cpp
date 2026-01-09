#include "common/thread_pool.h"

#include <algorithm>

namespace nodus {

JobBatch::JobBatch(uint32_t count) {
    remaining_.store(count, std::memory_order_relaxed);
}

void JobBatch::wait() {
    if (done()) return;
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&]() { return remaining_.load(std::memory_order_acquire) == 0; });
}

bool JobBatch::done() const {
    return remaining_.load(std::memory_order_acquire) == 0;
}

void JobBatch::finish_one() {
    if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        std::lock_guard<std::mutex> lk(mu_);
        cv_.notify_all();
    }
}

ThreadPool::ThreadPool(const Options& options) {
    thread_count_ = options.thread_count;
    if (thread_count_ == 0) {
        thread_count_ = std::max(1u, std::thread::hardware_concurrency());
    }
    if (options.start_immediately) start();
}

ThreadPool::~ThreadPool() {
    stop();
}

void ThreadPool::start() {
    if (running_.exchange(true, std::memory_order_acq_rel)) return;
    stop_requested_.store(false, std::memory_order_release);
    workers_.reserve(thread_count_);
    for (uint32_t i = 0; i < thread_count_; ++i) {
        workers_.emplace_back([this, i]() { worker_loop(i); });
    }
}

void ThreadPool::stop() {
    if (!running_.exchange(false, std::memory_order_acq_rel)) return;
    stop_requested_.store(true, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        queue_cv_.notify_all();
    }
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void ThreadPool::enqueue(Job job) {
    if (!job.fn) return;
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        queue_.push_back(std::move(job));
    }
    queue_cv_.notify_one();
}

std::shared_ptr<JobBatch> ThreadPool::submit_batch(const Job* jobs, uint32_t count) {
    auto batch = std::make_shared<JobBatch>(count);
    if (!jobs || count == 0) {
        batch->finish_one();
        return batch;
    }
    {
        std::lock_guard<std::mutex> lk(queue_mu_);
        for (uint32_t i = 0; i < count; ++i) {
            Job job = jobs[i];
            job.batch = batch;
            queue_.push_back(std::move(job));
        }
    }
    queue_cv_.notify_all();
    return batch;
}

void ThreadPool::worker_loop(uint32_t worker_id) {
    while (true) {
        Job job{};
        {
            std::unique_lock<std::mutex> lk(queue_mu_);
            queue_cv_.wait(lk, [&]() {
                return stop_requested_.load(std::memory_order_acquire) || !queue_.empty();
            });
            if (stop_requested_.load(std::memory_order_acquire) && queue_.empty()) break;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        if (job.fn) {
            job.fn(job, worker_id);
        }
        if (job.batch) {
            job.batch->finish_one();
        }
    }
}

} // namespace nodus
