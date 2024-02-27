/**
 * Copyright 2020-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

#include <string>
#include <utility>
#include <vector>

#include "../../../plugin/sycl/common/partition_builder.h"
#include "../../../plugin/sycl/device_manager.h"
#include "../helpers.h"

namespace xgboost::sycl::common {

void TestPartitioning(float sparsity, int max_bins) {
  const size_t num_rows = 16;
  const size_t num_columns = 1;

  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(ctx.Device());

  auto p_fmat = RandomDataGenerator{num_rows, num_columns, sparsity}.GenerateDMatrix();
  sycl::DeviceMatrix dmat;
  dmat.Init(qu, p_fmat.get());

  common::GHistIndexMatrix gmat;
  gmat.Init(qu, &ctx, dmat, max_bins);

  RowSetCollection row_set_collection;
  auto& row_indices = row_set_collection.Data();
  row_indices.Resize(&qu, num_rows);
  size_t* p_row_indices = row_indices.Data();

  qu.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(num_rows),
                       [p_row_indices](::sycl::item<1> pid) {
      const size_t idx = pid.get_id(0);
      p_row_indices[idx] = idx;
    });
  }).wait_and_throw();
  row_set_collection.Init();

  RegTree tree;
  tree.ExpandNode(0, 0, 0, false, 0, 0, 0, 0, 0, 0, 0);

  const size_t n_nodes = row_set_collection.Size();
  PartitionBuilder partition_builder;
  partition_builder.Init(&qu, n_nodes, [&](size_t nid) {
    return row_set_collection[nid].Size();
  });

  std::vector<tree::ExpandEntry> nodes;
  nodes.emplace_back(tree::ExpandEntry(0, tree.GetDepth(0)));

  ::sycl::event event;
  std::vector<int32_t> split_conditions = {2};
  partition_builder.Partition(gmat, nodes, row_set_collection,
                    split_conditions, &tree, &event);
  qu.wait_and_throw();

  size_t* data_result = const_cast<size_t*>(row_set_collection[0].begin);
  partition_builder.MergeToArray(0, data_result, &event);
  qu.wait_and_throw();

  std::stringstream ss;
  bst_float split_pt = gmat.cut.Values()[split_conditions[0]];
  ss << "split_pt = " << split_pt << "\n";

  std::vector<size_t> ridx_left;
  std::vector<size_t> ridx_right;
  for (auto &batch : gmat.p_fmat->GetBatches<SparsePage>()) {
    const auto& data_vec = batch.data.HostVector();
    const auto& offset_vec = batch.offset.HostVector();

    size_t begin = offset_vec[0];
    for (size_t idx = 0; idx < offset_vec.size() - 1; ++idx) {
      size_t end = offset_vec[idx + 1];
      ss << "idx = " << idx << "\tbegin = " << begin << "\tend = " << end;
      if (begin < end) {
        const auto& entry = data_vec[begin];
        ss << "fvalue = " << entry.fvalue;
        if (entry.fvalue < split_pt) {
          ridx_left.push_back(idx);
        } else {
          ridx_right.push_back(idx);
        }
      } else {
        // missing value
        ss << "default_left = " << tree[0].DefaultLeft();
        if (tree[0].DefaultLeft()) {
          ridx_left.push_back(idx);
        } else {
          ridx_right.push_back(idx);
        }
      }
      ss << "\n";
      begin = end;
    }
  }

  std::vector<size_t> row_indices_host(num_rows);
  qu.memcpy(row_indices_host.data(), row_indices.Data(), num_rows * sizeof(size_t));
  qu.wait_and_throw();
  ss << "row_indices = ";
  for (auto idx : row_indices_host) {
    ss << idx << "\t";
  }

  ASSERT_EQ(ridx_left.size(),  partition_builder.GetNLeftElems(0)) << ss.str();
  for (size_t i = 0; i < ridx_left.size(); ++i) {
    ASSERT_EQ(ridx_left[i], row_indices_host[i]) << ss.str();
  }

  ASSERT_EQ(ridx_right.size(), partition_builder.GetNRightElems(0)) << ss.str();
  for (size_t i = 0; i < ridx_right.size(); ++i) {
    ASSERT_EQ(ridx_right[i], row_indices_host[num_rows - 1 - i]) << ss.str();
  }
}

TEST(SyclPartitionBuilder, BasicTest) {
  constexpr size_t kNodes = 5;
  // Number of rows for each node
  std::vector<size_t> rows = { 5, 5, 10, 1, 2 };

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SYCL_default());
  PartitionBuilder builder;
  builder.Init(&qu, kNodes, [&](size_t i) {
    return rows[i];
  });

  // We test here only the basics, thus syntetic partition builder is adopted
  // Number of rows to go left for each node.
  std::vector<size_t> rows_for_left_node = { 2, 0, 7, 1, 2 };

  size_t first_row_id = 0;
  for(size_t nid = 0; nid < kNodes; ++nid) {
    size_t n_rows_nodes = rows[nid];

    auto rid_buff = builder.GetData(nid);
    size_t rid_buff_size = rid_buff.size();
    auto* rid_buff_ptr = rid_buff.data();

    size_t n_left  = rows_for_left_node[nid];
    size_t n_right = rows[nid] - n_left;

    qu.submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(n_left), [=](::sycl::id<1> pid) {
        int row_id = first_row_id + pid[0];
        rid_buff_ptr[pid[0]] = row_id;
      });
    });
    qu.wait();
    first_row_id += n_left;

    // We are storing indexes for the right side in the tail of the array to save some memory
    qu.submit([&](::sycl::handler& cgh) {
      cgh.parallel_for<>(::sycl::range<1>(n_right), [=](::sycl::id<1> pid) {
        int row_id = first_row_id + pid[0];
        rid_buff_ptr[rid_buff_size - pid[0] - 1] = row_id;
      });
    });
    qu.wait();
    first_row_id += n_right;

    builder.SetNLeftElems(nid, n_left);
    builder.SetNRightElems(nid, n_right);
  }

  ::sycl::event event;
  std::vector<size_t> v(*std::max_element(rows.begin(), rows.end()));
  size_t row_id = 0;
  for(size_t nid = 0; nid < kNodes; ++nid) {
    builder.MergeToArray(nid, v.data(), &event);
    qu.wait();

    // Check that row_id for left side are correct
    for(size_t j = 0; j < rows_for_left_node[nid]; ++j) {
       ASSERT_EQ(v[j], row_id++);
    }

    // Check that row_id for right side are correct
    for(size_t j = 0; j < rows[nid] - rows_for_left_node[nid]; ++j) {
      ASSERT_EQ(v[rows[nid] - j - 1], row_id++);
    }

    // Check that number of left/right rows are correct
    size_t n_left  = builder.GetNLeftElems(nid);
    size_t n_right = builder.GetNRightElems(nid);
    ASSERT_EQ(n_left, rows_for_left_node[nid]);
    ASSERT_EQ(n_right, (rows[nid] - rows_for_left_node[nid]));
  }
}

TEST(SyclPartitionBuilder, PartitioningSparce) {
  TestPartitioning(0.3, 256);
}

TEST(SyclPartitionBuilder, PartitioningDence8Bits) {
  TestPartitioning(0.0, 256);
}

TEST(SyclPartitionBuilder, PartitioningDence16Bits) {
  TestPartitioning(0.0, 256 + 1);
}

TEST(SyclPartitionBuilder, PartitioningDence32Bits) {
  TestPartitioning(0.0, (1u << 16) + 1);
}

}  // namespace xgboost::sycl::common
