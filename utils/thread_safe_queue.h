#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
    public:
        ThreadSafeQueue(size_t maxSize = 20) : max_size_(maxSize) {}

        void push(T value) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_push_.wait(lock, [this] { return queue_.size() < max_size_; });
            queue_.push(std::move(value));
            lock.unlock();
            cond_pop_.notify_one();
        }

        bool try_pop(T& value) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (queue_.empty()) {
                return false;
            }
            value = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            cond_push_.notify_one();
            return true;
        }

        void wait_and_pop(T& value) {
            std::unique_lock<std::mutex> lock(mutex_);
            cond_pop_.wait(lock, [this] { return !queue_.empty(); });
            value = std::move(queue_.front());
            queue_.pop();
            lock.unlock();
            cond_push_.notify_one();
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.empty();
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return queue_.size();
        }

    private:
        mutable std::mutex mutex_;
        std::queue<T> queue_;
        std::condition_variable cond_pop_;
        std::condition_variable cond_push_;
        size_t max_size_;
};