/*!
 * Copyright 2017-2024 by Contributors
 * \file updater_quantile_hist.cc
 */
#include <dmlc/timer.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>
#pragma GCC diagnostic pop

#include <utility>

#include "updater_quantile_hist.h"

namespace xgboost {
namespace sycl {
namespace tree {

DMLC_REGISTRY_FILE_TAG(updater_quantile_hist_sycl);

DMLC_REGISTER_PARAMETER(HistMakerTrainParam);

void QuantileHistMaker::Configure(const Args& args) {
  const DeviceOrd device_spec = ctx_->Device();
  qu_ = device_manager.GetQueue(device_spec);

  // initialize pruner
  if (!pruner_) {
    pruner_.reset(TreeUpdater::Create("prune", ctx_, task_));
  }
  pruner_->Configure(args);
  param_.UpdateAllowUnknown(args);
  hist_maker_param_.UpdateAllowUnknown(args);
}

template<typename GradientSumT>
void QuantileHistMaker::SetPimpl(std::unique_ptr<HistUpdater<GradientSumT>>* pimpl,
                                 DMatrix *dmat) {
  pimpl->reset(new HistUpdater<GradientSumT>(
                qu_,
                param_,
                std::move(pruner_),
                int_constraint_, dmat));
  if (rabit::IsDistributed()) {
    (*pimpl)->SetHistSynchronizer(new DistributedHistSynchronizer<GradientSumT>());
    (*pimpl)->SetHistRowsAdder(new DistributedHistRowsAdder<GradientSumT>());
  } else {
    (*pimpl)->SetHistSynchronizer(new BatchHistSynchronizer<GradientSumT>());
    (*pimpl)->SetHistRowsAdder(new BatchHistRowsAdder<GradientSumT>());
  }
}

template<typename GradientSumT>
void QuantileHistMaker::CallUpdate(
        const std::unique_ptr<HistUpdater<GradientSumT>>& pimpl,
        xgboost::tree::TrainParam const *param,
        HostDeviceVector<GradientPair> *gpair,
        DMatrix *dmat,
        xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
        const std::vector<RegTree *> &trees) {
  const std::vector<GradientPair>& gpair_h = gpair->ConstHostVector();
  gpair_device_.Resize(&qu_, gpair_h.size());
  qu_.memcpy(gpair_device_.Data(), gpair_h.data(), gpair_h.size() * sizeof(GradientPair));
  qu_.wait();

  for (auto tree : trees) {
    pimpl->Update(ctx_, param, gmat_, gpair, gpair_device_, dmat, out_position, tree);
  }
}

void QuantileHistMaker::Update(xgboost::tree::TrainParam const *param,
                               HostDeviceVector<GradientPair> *gpair,
                               DMatrix *dmat,
                               xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                               const std::vector<RegTree *> &trees) {
  if (dmat != p_last_dmat_ || is_gmat_initialized_ == false) {
    updater_monitor_.Start("DeviceMatrixInitialization");
    sycl::DeviceMatrix dmat_device;
    dmat_device.Init(qu_, dmat);
    updater_monitor_.Stop("DeviceMatrixInitialization");
    updater_monitor_.Start("GmatInitialization");
    gmat_.Init(qu_, ctx_, dmat_device, static_cast<uint32_t>(param_.max_bin));
    updater_monitor_.Stop("GmatInitialization");
    is_gmat_initialized_ = true;
  }
  // rescale learning rate according to size of trees
  float lr = param_.learning_rate;
  param_.learning_rate = lr / trees.size();
  int_constraint_.Configure(param_, dmat->Info().num_col_);
  // build tree
  bool has_double_support = qu_.get_device().has(::sycl::aspect::fp64);
  if (hist_maker_param_.single_precision_histogram || !has_double_support) {
    if (!hist_maker_param_.single_precision_histogram) {
      LOG(WARNING) << "Target device doesn't support fp64, using single_precision_histogram=True";
    }
    if (!pimpl_single) {
      SetPimpl(&pimpl_single, dmat);
    }
    CallUpdate(pimpl_single, param, gpair, dmat, out_position, trees);
  } else {
    if (!pimpl_double) {
      SetPimpl(&pimpl_double, dmat);
    }
    CallUpdate(pimpl_double, param, gpair, dmat, out_position, trees);
  }

  param_.learning_rate = lr;

  p_last_dmat_ = dmat;
}

bool QuantileHistMaker::UpdatePredictionCache(const DMatrix* data,
                                              linalg::MatrixView<float> out_preds) {
  if (param_.subsample < 1.0f) {
    return false;
  } else {
    bool has_double_support = qu_.get_device().has(::sycl::aspect::fp64);
    if ((hist_maker_param_.single_precision_histogram || !has_double_support) && pimpl_single) {
        return pimpl_single->UpdatePredictionCache(data, out_preds);
    } else if (pimpl_double) {
        return pimpl_double->UpdatePredictionCache(data, out_preds);
    } else {
       return false;
    }
  }
}

XGBOOST_REGISTER_TREE_UPDATER(QuantileHistMaker, "grow_quantile_histmaker_sycl")
.describe("Grow tree using quantized histogram with SYCL.")
.set_body(
    [](Context const* ctx, ObjInfo const * task) {
      return new QuantileHistMaker(ctx, task);
    });
}  // namespace tree
}  // namespace sycl
}  // namespace xgboost
