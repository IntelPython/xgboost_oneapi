/*!
 * Copyright 2017-2019 XGBoost contributors
 */
#include <gtest/gtest.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <xgboost/objective.h>
#pragma GCC diagnostic pop
#include <xgboost/context.h>

#include "../helpers.h"
#include "../objective/test_regression_obj.h"

namespace xgboost {

TEST(SyclObjective, LinearRegressionGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestLinearRegressionGPair(&ctx);
}

TEST(SyclObjective, SquaredLog) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestSquaredLog(&ctx);
}

TEST(SyclObjective, LogisticRegressionGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestLogisticRegressionGPair(&ctx);
}

TEST(SyclObjective, LogisticRegressionBasic) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});

  TestLogisticRegressionBasic(&ctx);
}

TEST(SyclObjective, LogisticRawGPair) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  TestsLogisticRawGPair(&ctx);
}

TEST(SyclObjective, CPUvsSycl) {
  Context ctx;
  ctx.UpdateAllowUnknown(Args{{"device", "sycl"}});
  ObjFunction * obj_sycl =
      ObjFunction::Create("reg:squarederror_sycl", &ctx);

  ctx = ctx.MakeCPU();
  ObjFunction * obj_cpu =
      ObjFunction::Create("reg:squarederror", &ctx);

  HostDeviceVector<GradientPair> cpu_out_preds;
  HostDeviceVector<GradientPair> sycl_out_preds;

  constexpr size_t kRows = 400;
  constexpr size_t kCols = 100;
  auto pdmat = RandomDataGenerator(kRows, kCols, 0).Seed(0).GenerateDMatrix();
  HostDeviceVector<float> preds;
  preds.Resize(kRows);
  auto& h_preds = preds.HostVector();
  for (size_t i = 0; i < h_preds.size(); ++i) {
    h_preds[i] = static_cast<float>(i);
  }
  auto& info = pdmat->Info();

  info.labels.Reshape(kRows, 1);
  auto& h_labels = info.labels.Data()->HostVector();
  for (size_t i = 0; i < h_labels.size(); ++i) {
    h_labels[i] = 1 / static_cast<float>(i+1);
  }

  {
    // CPU
    obj_cpu->GetGradient(preds, info, 0, &cpu_out_preds);
  }
  {
    // sycl
    obj_sycl->GetGradient(preds, info, 0, &sycl_out_preds);
  }

  auto& h_cpu_out = cpu_out_preds.HostVector();
  auto& h_sycl_out = sycl_out_preds.HostVector();

  float sgrad = 0;
  float shess = 0;
  for (size_t i = 0; i < kRows; ++i) {
    sgrad += std::pow(h_cpu_out[i].GetGrad() - h_sycl_out[i].GetGrad(), 2);
    shess += std::pow(h_cpu_out[i].GetHess() - h_sycl_out[i].GetHess(), 2);
  }
  ASSERT_NEAR(sgrad, 0.0f, kRtEps);
  ASSERT_NEAR(shess, 0.0f, kRtEps);

  delete obj_cpu;
  delete obj_sycl;
}

}  // namespace xgboost
