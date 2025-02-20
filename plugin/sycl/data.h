/*!
 * Copyright by Contributors 2017-2023
 */
#ifndef PLUGIN_SYCL_DATA_H_
#define PLUGIN_SYCL_DATA_H_

#include <cstddef>
#include <limits>
#include <mutex>
#include <vector>
#include <memory>
#include <algorithm>

#include "xgboost/base.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "xgboost/data.h"
#pragma GCC diagnostic pop
#include "xgboost/logging.h"
#include "xgboost/host_device_vector.h"

#include "../../src/common/threading_utils.h"

#include <sycl/sycl.hpp>

namespace xgboost {
namespace sycl {
template <typename T>
using AtomicRef = ::sycl::atomic_ref<T,
                                    ::sycl::memory_order::relaxed,
                                    ::sycl::memory_scope::device,
                                    ::sycl::access::address_space::ext_intel_global_device_space>;

enum class MemoryType { shared, on_device};

template <typename T>
class USMDeleter {
 public:
  explicit USMDeleter(::sycl::queue qu) : qu_(qu) {}

  void operator()(T* data) const {
    ::sycl::free(data, qu_);
  }

 private:
  ::sycl::queue qu_;
};

template <typename T, MemoryType memory_type = MemoryType::shared>
class USMVector {
  static_assert(std::is_standard_layout<T>::value, "USMVector admits only POD types");

  std::shared_ptr<T> allocate_memory_(::sycl::queue* qu, size_t size) {
    if constexpr (memory_type == MemoryType::shared) {
      auto* ptr = ::sycl::malloc_shared<T>(size_, *qu);
      CHECK_NE(ptr, nullptr) << "Faild to allocate memory";
      return std::shared_ptr<T>(ptr, USMDeleter<T>(*qu));
    } else {
      auto* ptr = ::sycl::malloc_device<T>(size_, *qu);
      CHECK_NE(ptr, nullptr) << "Faild to allocate memory";
      return std::shared_ptr<T>(ptr, USMDeleter<T>(*qu));
    }
  }

  void copy_vector_to_memory_(::sycl::queue* qu, const std::vector<T> &vec) {
    if constexpr (memory_type == MemoryType::shared) {
      std::copy(vec.begin(), vec.end(), data_.get());
    } else {
      qu->memcpy(data_.get(), vec.data(), size_ * sizeof(T));
    }
  }


 public:
  class ConstIterator {
    const T* ptr_;

   public:
    using difference_type = size_t;
    using value_type = T;
    using pointer = const T;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    ConstIterator() : ptr_(nullptr) {}
    explicit ConstIterator(const T* ptr = 0) : ptr_(ptr) {}
    inline ConstIterator& operator+=(difference_type rhs) {ptr_ += rhs; return *this;}
    inline ConstIterator& operator-=(difference_type rhs) {ptr_ -= rhs; return *this;}
    inline const T& operator*() const {return *ptr_;}
    inline const T* operator->() const {return ptr_;}
    inline const T& operator[](difference_type idx) const {return ptr_[idx];}

    inline ConstIterator& operator++() {++ptr_; return *this;}
    inline ConstIterator& operator--() {--ptr_; return *this;}

    inline difference_type operator-(const ConstIterator& rhs) const {return ptr_-rhs.ptr_;}
    inline ConstIterator operator+(difference_type rhs) const {return ConstIterator(ptr_+rhs);}
    inline ConstIterator operator-(difference_type rhs) const {return ConstIterator(ptr_-rhs);}
    friend inline ConstIterator operator+(difference_type lhs, const ConstIterator& rhs) {
      return ConstIterator(lhs+rhs.ptr_);
    }
    friend inline ConstIterator operator-(difference_type lhs, const ConstIterator& rhs) {
      return ConstIterator(lhs-rhs.ptr_);
    }

    inline bool operator==(const ConstIterator& rhs) const {return ptr_ == rhs.ptr_;}
    inline bool operator!=(const ConstIterator& rhs) const {return ptr_ != rhs.ptr_;}
    inline bool operator>(const ConstIterator& rhs) const {return ptr_ > rhs.ptr_;}
    inline bool operator<(const ConstIterator& rhs) const {return ptr_ < rhs.ptr_;}
    inline bool operator>=(const ConstIterator& rhs) const {return ptr_ >= rhs.ptr_;}
    inline bool operator<=(const ConstIterator& rhs) const {return ptr_ <= rhs.ptr_;}
  };

  class Iterator {
    T* ptr_;

   public:
    using difference_type = size_t;
    using value_type = T;
    using pointer = const T;
    using reference = const T&;
    using iterator_category = std::random_access_iterator_tag;

    Iterator() : ptr_(nullptr) {}
    explicit Iterator(T* ptr = 0) : ptr_(ptr) {}
    inline Iterator& operator+=(difference_type rhs) {ptr_ += rhs; return *this;}
    inline Iterator& operator-=(difference_type rhs) {ptr_ -= rhs; return *this;}
    inline T& operator*() const {return *ptr_;}
    inline T* operator->() const {return ptr_;}
    inline T& operator[](difference_type idx) const {return ptr_[idx];}

    inline Iterator& operator++() {++ptr_; return *this;}
    inline Iterator& operator--() {--ptr_; return *this;}

    inline difference_type operator-(const Iterator& rhs) const {return ptr_-rhs.ptr_;}
    inline Iterator operator+(difference_type rhs) const {return Iterator(ptr_+rhs);}
    inline Iterator operator-(difference_type rhs) const {return Iterator(ptr_-rhs);}
    friend inline Iterator operator+(difference_type lhs, const Iterator& rhs) {
      return Iterator(lhs+rhs.ptr_);
    }
    friend inline Iterator operator-(difference_type lhs, const Iterator& rhs) {
      return Iterator(lhs-rhs.ptr_);
    }

    inline bool operator==(const Iterator& rhs) const {return ptr_ == rhs.ptr_;}
    inline bool operator!=(const Iterator& rhs) const {return ptr_ != rhs.ptr_;}
    inline bool operator>(const Iterator& rhs) const {return ptr_ > rhs.ptr_;}
    inline bool operator<(const Iterator& rhs) const {return ptr_ < rhs.ptr_;}
    inline bool operator>=(const Iterator& rhs) const {return ptr_ >= rhs.ptr_;}
    inline bool operator<=(const Iterator& rhs) const {return ptr_ <= rhs.ptr_;}
  };

  ConstIterator Cbegin() { return ConstIterator(data_.get()); }
  ConstIterator Cend() { return ConstIterator(data_.get() + size_); }

  Iterator Begin() { return Iterator(data_.get()); }
  Iterator End() { return Iterator(data_.get() + size_); }

  USMVector() : size_(0), capacity_(0), data_(nullptr) {}

  USMVector(::sycl::queue* qu, size_t size) : size_(size), capacity_(size) {
    data_ = allocate_memory_(qu, size_);
  }

  USMVector(::sycl::queue* qu, size_t size, T v) : size_(size), capacity_(size) {
    data_ = allocate_memory_(qu, size_);
    qu->fill(data_.get(), v, size_).wait();
  }

  USMVector(::sycl::queue* qu, size_t size, T v,
            ::sycl::event* event) : size_(size), capacity_(size) {
    data_ = allocate_memory_(qu, size_);
    *event = qu->fill(data_.get(), v, size_, *event);
  }

  USMVector(::sycl::queue* qu, const std::vector<T> &vec) {
    size_ = vec.size();
    capacity_ = size_;
    data_ = allocate_memory_(qu, size_);
    copy_vector_to_memory_(qu, vec);
  }

  ~USMVector() {
  }

  USMVector(const USMVector&) = delete;

  USMVector<T>& operator=(const USMVector<T>& other) = delete;

  T* Data() { return data_.get(); }
  const T* DataConst() const { return data_.get(); }

  size_t Size() const { return size_; }

  size_t Capacity() const { return capacity_; }

  T& operator[] (size_t i) { return data_.get()[i]; }
  const T& operator[] (size_t i) const { return data_.get()[i]; }

  bool Empty() const { return (size_ == 0); }

  void Clear() {
    data_.reset();
    size_ = 0;
    capacity_ = 0;
  }

  void Resize(::sycl::queue* qu, size_t size_new) {
    if (size_new <= capacity_) {
      size_ = size_new;
    } else {
      size_t size_old = size_;
      auto data_old = data_;
      size_ = size_new;
      capacity_ = size_new;
      data_ = allocate_memory_(qu, size_);;
      if (size_old > 0) {
        qu->memcpy(data_.get(), data_old.get(), sizeof(T) * size_old).wait();
      }
    }
  }

  /* Resize without keeping the data*/
  void ResizeNoCopy(::sycl::queue* qu, size_t size_new) {
    if (size_new <= capacity_) {
      size_ = size_new;
    } else {
      size_ = size_new;
      capacity_ = size_new;
      data_ = allocate_memory_(qu, size_);
    }
  }

  void Resize(::sycl::queue* qu, size_t size_new, T v) {
    if (size_new <= size_) {
      size_ = size_new;
    } else if (size_new <= capacity_) {
      qu->fill(data_.get() + size_, v, size_new - size_).wait();
      size_ = size_new;
    } else {
      size_t size_old = size_;
      auto data_old = data_;
      size_ = size_new;
      capacity_ = size_new;
      data_ = allocate_memory_(qu, size_);
      if (size_old > 0) {
        qu->memcpy(data_.get(), data_old.get(), sizeof(T) * size_old).wait();
      }
      qu->fill(data_.get() + size_old, v, size_new - size_old).wait();
    }
  }

  void Resize(::sycl::queue* qu, size_t size_new, T v, ::sycl::event* event) {
    if (size_new <= size_) {
      size_ = size_new;
    } else if (size_new <= capacity_) {
      *event = qu->fill(data_.get() + size_, v, size_new - size_, *event);
      size_ = size_new;
    } else {
      size_t size_old = size_;
      auto data_old = data_;
      size_ = size_new;
      capacity_ = size_new;
      data_ = allocate_memory_(qu, size_);
      if (size_old > 0) {
        *event = qu->memcpy(data_.get(), data_old.get(), sizeof(T) * size_old, *event);
      }
      *event = qu->fill(data_.get() + size_old, v, size_new - size_old, *event);
    }
  }

  void ResizeAndFill(::sycl::queue* qu, size_t size_new, int v, ::sycl::event* event) {
    if (size_new <= size_) {
      size_ = size_new;
      *event = qu->memset(data_.get(), v, size_new * sizeof(T), *event);
    } else if (size_new <= capacity_) {
      size_ = size_new;
      *event = qu->memset(data_.get(), v, size_new * sizeof(T), *event);
    } else {
      size_t size_old = size_;
      auto data_old = data_;
      size_ = size_new;
      capacity_ = size_new;
      data_ = allocate_memory_(qu, size_);
      *event = qu->memset(data_.get(), v, size_new * sizeof(T), *event);
    }
  }

  ::sycl::event Fill(::sycl::queue* qu, T v) {
    return qu->fill(data_.get(), v, size_);
  }

  void Init(::sycl::queue* qu, const std::vector<T> &vec) {
    size_ = vec.size();
    capacity_ = size_;
    data_ = allocate_memory_(qu, size_);
    copy_vector_to_memory_(qu, vec);
  }

  using value_type = T;  // NOLINT

 private:
  size_t size_;
  size_t capacity_;
  std::shared_ptr<T> data_;
};

/* Wrapper for DMatrix which stores all batches in a single USM buffer */
struct DeviceMatrix {
  DMatrix* p_mat = nullptr;  // Pointer to the original matrix on the host
  ::sycl::queue qu_;
  USMVector<size_t, MemoryType::on_device> row_ptr;
  USMVector<Entry, MemoryType::on_device> data;
  size_t total_offset;
  bool is_from_cache = false;

  DeviceMatrix() = default;

  DeviceMatrix(const DeviceMatrix& other) = delete;

  DeviceMatrix& operator= (const DeviceMatrix& other) = delete;

  // During training the same dmatrix is used, so we don't need reload it on device
  bool ReinitializationRequired(DMatrix* dmat, bool training) {
    if (!training) return true;
    if (p_mat != dmat) return true;
    return false;
  }

  void Init(::sycl::queue qu, DMatrix* dmat, bool training = false) {
    qu_ = qu;
    if (!ReinitializationRequired(dmat, training)) {
      is_from_cache = true;
      return;
    }

    is_from_cache = false;

    p_mat = dmat;

    size_t num_row = 0;
    size_t num_nonzero = 0;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      num_nonzero += data_vec.size();
      num_row += batch.Size();
    }

    row_ptr.Resize(&qu_, num_row + 1);
    size_t* rows = row_ptr.Data();
    data.Resize(&qu_, num_nonzero);

    size_t data_offset = 0;
    ::sycl::event event;
    for (auto &batch : dmat->GetBatches<SparsePage>()) {
      const auto& data_vec = batch.data.HostVector();
      const auto& offset_vec = batch.offset.HostVector();
      size_t batch_size = batch.Size();
      if (batch_size > 0) {
        const auto base_rowid = batch.base_rowid;
        event = qu.memcpy(row_ptr.Data() + base_rowid, offset_vec.data(),
                          sizeof(size_t) * batch_size, event);
        if (base_rowid > 0) {
          qu.submit([&](::sycl::handler& cgh) {
            cgh.depends_on(event);
            cgh.parallel_for<>(::sycl::range<1>(batch_size), [=](::sycl::id<1> pid) {
              int row_id = pid[0];
              rows[row_id] += base_rowid;
            });
          });
        }
        event = qu.memcpy(data.Data() + data_offset, data_vec.data(),
                          sizeof(Entry) * offset_vec[batch_size], event);
        data_offset += offset_vec[batch_size];
        qu.wait();
      }
    }
    qu.submit([&](::sycl::handler& cgh) {
      cgh.depends_on(event);
      cgh.single_task<>([=] {
        rows[num_row] = data_offset;
      });
    });
    qu.wait();
    total_offset = data_offset;
  }

  ~DeviceMatrix() {
  }
};
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_DATA_H_
