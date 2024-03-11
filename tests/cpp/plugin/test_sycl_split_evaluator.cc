/**
 * Copyright 2020-2023 by XGBoost contributors
 */
#include <gtest/gtest.h>

// #include <string>
// #include <utility>
#include <vector>
// #include <numeric>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include "../../../plugin/sycl/tree/split_evaluator.h"
#pragma GCC diagnostic pop

#include "../../../plugin/sycl/device_manager.h"
#include "../helpers.h"

namespace xgboost::sycl::tree {

template<typename GradientSumT>
void BasicTestSplitEvaluator(const std::string& monotone_constraints, bool has_constrains) {
  const size_t n_columns = 2;

  xgboost::tree::TrainParam param;
  param.UpdateAllowUnknown(Args{{"min_child_weight", "0"},
                                {"reg_lambda", "0"},
                                {"monotone_constraints", monotone_constraints}});

  DeviceManager device_manager;
  auto qu = device_manager.GetQueue(DeviceOrd::SYCL_default());

  TreeEvaluator<GradientSumT> tree_evaluator(qu, param, n_columns);
  ASSERT_EQ(tree_evaluator.HasConstraint(), has_constrains);
}

TEST(SyclSplitEvaluator, BasicTest) {
  BasicTestSplitEvaluator<float>("( 0,  0)", false);
  BasicTestSplitEvaluator<float>("( 1,  0)", true);
  BasicTestSplitEvaluator<float>("( 0,  1)", true);
  BasicTestSplitEvaluator<float>("(-1,  0)", true);
  BasicTestSplitEvaluator<float>("( 0, -1)", true);
  BasicTestSplitEvaluator<float>("( 1,  1)", true);
  BasicTestSplitEvaluator<float>("(-1, -1)", true);
  BasicTestSplitEvaluator<float>("( 1, -1)", true);
  BasicTestSplitEvaluator<float>("(-1,  1)", true);
}

}  // namespace xgboost::sycl::tree
