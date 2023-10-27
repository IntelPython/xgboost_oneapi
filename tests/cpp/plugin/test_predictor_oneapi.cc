/*!
 * Copyright 2017-2020 XGBoost contributors
 */
#include <gtest/gtest.h>
#include <xgboost/predictor.h>

#include "../../../src/data/adapter.h"
#include "../../../src/data/proxy_dmatrix.h"
#include "../../../src/gbm/gbtree_model.h"
#include "../filesystem.h"  // dmlc::TemporaryDirectory
#include "../helpers.h"
#include "../predictor/test_predictor.h"

namespace xgboost {
namespace {
void TestBasic(DMatrix* dmat) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  std::unique_ptr<Predictor> oneapi_predictor =
      std::unique_ptr<Predictor>(Predictor::Create("oneapi_predictor", &ctx));

  size_t const kRows = dmat->Info().num_row_;
  size_t const kCols = dmat->Info().num_col_;

  LearnerModelParam param(MakeMP(kCols, .0, 1));
  gbm::GBTreeModel model = CreateTestModel(&param, &ctx);

  // Test predict batch
  PredictionCacheEntry out_predictions;
  oneapi_predictor->InitOutPredictions(dmat->Info(), &out_predictions.predictions, model);
  oneapi_predictor->PredictBatch(dmat, &out_predictions, model, 0);

  std::vector<float>& out_predictions_h = out_predictions.predictions.HostVector();
  for (size_t i = 0; i < out_predictions.predictions.Size(); i++) {
    ASSERT_EQ(out_predictions_h[i], 1.5);
  }

  // Test predict instance
  auto const &batch = *dmat->GetBatches<xgboost::SparsePage>().begin();
  auto page = batch.GetView();
  for (size_t i = 0; i < batch.Size(); i++) {
    std::vector<float> instance_out_predictions;
    oneapi_predictor->PredictInstance(page[i], &instance_out_predictions, model);
    ASSERT_EQ(instance_out_predictions[0], 1.5);
  }

  // Test predict leaf
  HostDeviceVector<float> leaf_out_predictions;
  oneapi_predictor->PredictLeaf(dmat, &leaf_out_predictions, model);
  auto const& h_leaf_out_predictions = leaf_out_predictions.ConstHostVector();
  for (auto v : h_leaf_out_predictions) {
    ASSERT_EQ(v, 0);
  }

  // Test predict contribution
  HostDeviceVector<float> out_contribution_hdv;
  auto& out_contribution = out_contribution_hdv.HostVector();
  oneapi_predictor->PredictContribution(dmat, &out_contribution_hdv, model);
  ASSERT_EQ(out_contribution.size(), kRows * (kCols + 1));
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i+1) % (kCols+1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }

  // Test predict contribution (approximate method)
  oneapi_predictor->PredictContribution(dmat, &out_contribution_hdv, model, 0, nullptr, true);
  for (size_t i = 0; i < out_contribution.size(); ++i) {
    auto const& contri = out_contribution[i];
    // shift 1 for bias, as test tree is a decision dump, only global bias is filled with LeafValue().
    if ((i+1) % (kCols+1) == 0) {
      ASSERT_EQ(out_contribution.back(), 1.5f);
    } else {
      ASSERT_EQ(contri, 0);
    }
  }
}
}  // anonymous namespace

TEST(SyclPredictor, Basic) {
  size_t constexpr kRows = 5;
  size_t constexpr kCols = 5;
  auto dmat = RandomDataGenerator(kRows, kCols, 0).GenerateDMatrix();
  TestBasic(dmat.get());
}

TEST(SyclPredictor, ExternalMemory) {
  size_t constexpr kPageSize = 64, kEntriesPerCol = 3;
  size_t constexpr kEntries = kPageSize * kEntriesPerCol * 2;
  std::unique_ptr<DMatrix> dmat = CreateSparsePageDMatrix(kEntries);
  TestBasic(dmat.get());
}

TEST(SyclPredictor, InplacePredict) {
  bst_row_t constexpr kRows{128};
  bst_feature_t constexpr kCols{64};
  auto gen = RandomDataGenerator{kRows, kCols, 0.5}.Device(-1);
  {
    HostDeviceVector<float> data;
    gen.GenerateDense(&data);
    ASSERT_EQ(data.Size(), kRows * kCols);
    Context ctx;
    ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
    std::shared_ptr<data::DMatrixProxy> x{new data::DMatrixProxy{}};
    auto array_interface = GetArrayInterface(&data, kRows, kCols);
    std::string arr_str;
    Json::Dump(array_interface, &arr_str);
    x->SetArrayData(arr_str.data());
    TestInplacePrediction(&ctx, x, kRows, kCols);
  }
}
}  // namespace xgboost
