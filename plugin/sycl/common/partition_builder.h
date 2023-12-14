/*!
 * Copyright 2017-2023 XGBoost contributors
 */
#ifndef PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_
#define PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/data.h>
#pragma GCC diagnostic pop
#include <algorithm>
#include <vector>
#include <utility>

#include "../data.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {
// The builder is required for samples partition to left and rights children for set of nodes
class PartitionBuilder {
 public:
  static constexpr size_t maxLocalSums = 256;
  static constexpr size_t subgroupSize = 16;


  template<typename Func>
  void Init(::sycl::queue* qu, size_t n_nodes, Func funcNTaks) {
    nodes_offsets_.resize(n_nodes+1);
    result_rows_.resize(2 * n_nodes);
    n_nodes_ = n_nodes;


    nodes_offsets_[0] = 0;
    for (size_t i = 1; i < n_nodes+1; ++i) {
      nodes_offsets_[i] = nodes_offsets_[i-1] + funcNTaks(i-1);
    }


    if (data_.Size() < nodes_offsets_[n_nodes]) {
      data_.Resize(qu, nodes_offsets_[n_nodes]);
    }
    prefix_sums_.Resize(qu, maxLocalSums);
  }


  xgboost::common::Span<size_t> GetData(int nid) {
    return { data_.Data() + nodes_offsets_[nid], nodes_offsets_[nid + 1] - nodes_offsets_[nid] };
  }


  xgboost::common::Span<size_t> GetPrefixSums() {
    return { prefix_sums_.Data(), prefix_sums_.Size() };
  }


  size_t GetLocalSize(const xgboost::common::Range1d& range) {
    size_t range_size = range.end() - range.begin();
    size_t local_subgroups = range_size / (maxLocalSums * subgroupSize) +
                             !!(range_size % (maxLocalSums * subgroupSize));
    return subgroupSize * local_subgroups;
  }


  size_t GetSubgroupSize() {
    return subgroupSize;
  }

  size_t* GetResultRowsPtr() {
    return result_rows_.data();
  }


  size_t GetNLeftElems(int nid) const {
    // return result_left_rows_[nid];
    return result_rows_[2 * nid];
  }


  size_t GetNRightElems(int nid) const {
    // return result_right_rows_[nid];
    return result_rows_[2 * nid + 1];
  }


  ::sycl::event MergeToArray(::sycl::queue* qu, size_t node_in_set,
                             size_t* data_result,
                             ::sycl::event priv_event) {
    size_t n_nodes_total = GetNLeftElems(node_in_set) + GetNRightElems(node_in_set);
    if (n_nodes_total > 0) {
      const size_t* data = data_.Data() + nodes_offsets_[node_in_set];
      return qu->memcpy(data_result, data, sizeof(size_t) * n_nodes_total, priv_event);
    } else {
      return ::sycl::event();
    }
  }

 protected:
  std::vector<size_t> nodes_offsets_;
  std::vector<size_t> result_rows_;
  size_t n_nodes_;

  USMVector<size_t, MemoryType::on_device> data_;
  USMVector<size_t> prefix_sums_;
};

}  // namespace common
}  // namespace sycl
}  // namespace xgboost


#endif  // PLUGIN_SYCL_COMMON_PARTITION_BUILDER_H_
