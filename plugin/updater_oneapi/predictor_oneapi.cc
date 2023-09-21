/*!
 * Copyright by Contributors 2017-2023
 */
#include <cstddef>
#include <limits>
#include <mutex>
#include <rabit/rabit.h>

#include "data_oneapi.h"

#include "xgboost/tree_model.h"
#include "xgboost/tree_updater.h"

#include "../../src/data/adapter.h"
#include "../../src/common/math.h"
#include "../../src/gbm/gbtree_model.h"

#include "CL/sycl.hpp"

namespace xgboost {
namespace predictor {

DMLC_REGISTRY_FILE_TAG(predictor_oneapi);

class PredictorOneAPI : public Predictor {
 public:
  explicit PredictorOneAPI(Context const* context) :
      Predictor::Predictor{context} {
    bool is_cpu = false;
    std::vector<sycl::device> devices = sycl::device::get_devices();
    for (size_t i = 0; i < devices.size(); i++)
    {
      LOG(INFO) << "device_id = " << i << ", name = " << devices[i].get_info<sycl::info::device::name>();
    }
    if (context->device_id != Context::kDefaultId) {
      int n_devices = (int)devices.size();
      CHECK_LT(context->device_id, n_devices);
      is_cpu = devices[context->device_id].is_cpu();
    }
    LOG(INFO) << "device_id = " << context->device_id << ", is_cpu = " << int(is_cpu);

    if (is_cpu)
    {
      predictor_backend_.reset(Predictor::Create("cpu_predictor", context));
    }
    else
    {
      predictor_backend_.reset(Predictor::Create("oneapi_predictor_gpu", context));
    }
  }

  void Configure(const std::vector<std::pair<std::string, std::string>>& args) override {
    if (predictor_backend_) {
      predictor_backend_->Configure(args);
    }
  }

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts,
                    const gbm::GBTreeModel &model, uint32_t tree_begin,
                    uint32_t tree_end = 0) const override {
    predictor_backend_->PredictBatch(dmat, predts, model, tree_begin, tree_end);
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m,
                      const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    return predictor_backend_->InplacePredict(p_m, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    predictor_backend_->PredictInstance(inst, out_preds, model, ntree_limit);
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    predictor_backend_->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           const std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) const override {
    predictor_backend_->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       const std::vector<bst_float>* tree_weights,
                                       bool approximate) const override {
    predictor_backend_->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate);
  }

 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    predictor_backend_->InitOutPredictions(info, out_preds, model);
  }

 private:
  std::unique_ptr<Predictor> predictor_backend_;
};

/* Wrapper for descriptor of a tree node */
struct DeviceNodeOneAPI {
  DeviceNodeOneAPI()
      : fidx(-1), left_child_idx(-1), right_child_idx(-1) {}

  union NodeValue {
    float leaf_weight;
    float fvalue;
  };

  int fidx;
  int left_child_idx;
  int right_child_idx;
  NodeValue val;

  DeviceNodeOneAPI(const RegTree::Node& n) {
    this->left_child_idx = n.LeftChild();
    this->right_child_idx = n.RightChild();
    this->fidx = n.SplitIndex();
    if (n.DefaultLeft()) {
      fidx |= (1U << 31);
    }

    if (n.IsLeaf()) {
      this->val.leaf_weight = n.LeafValue();
    } else {
      this->val.fvalue = n.SplitCond();
    }
  }

  bool IsLeaf() const { return left_child_idx == -1; }

  int GetFidx() const { return fidx & ((1U << 31) - 1U); }

  bool MissingLeft() const { return (fidx >> 31) != 0; }

  int MissingIdx() const {
    if (MissingLeft()) {
      return this->left_child_idx;
    } else {
      return this->right_child_idx;
    }
  }

  float GetFvalue() const { return val.fvalue; }

  float GetWeight() const { return val.leaf_weight; }
};

/* OneAPI implementation of a device model, storing tree structure in USM buffers to provide access from device kernels */
class DeviceModelOneAPI {
 public:
  sycl::queue qu_;
  USMVector<DeviceNodeOneAPI> nodes_;
  USMVector<size_t> tree_segments_;
  USMVector<int> tree_group_;
  size_t tree_beg_;
  size_t tree_end_;
  int num_group_;

  DeviceModelOneAPI() {}

  ~DeviceModelOneAPI() {}

  void Init(sycl::queue qu, const gbm::GBTreeModel& model, size_t tree_begin, size_t tree_end) {
    qu_ = qu;
    CHECK_EQ(model.param.size_leaf_vector, 0);

    tree_segments_.Resize(qu_, (tree_end - tree_begin) + 1);
    int sum = 0;
    tree_segments_[0] = sum;
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      sum += model.trees[tree_idx]->GetNodes().size();
      tree_segments_[tree_idx - tree_begin + 1] = sum;
    }

    nodes_.Resize(qu_, sum);
    for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
      auto& src_nodes = model.trees[tree_idx]->GetNodes();
      for (size_t node_idx = 0; node_idx < src_nodes.size(); node_idx++)
        nodes_[node_idx + tree_segments_[tree_idx - tree_begin]] = src_nodes[node_idx];
    }

    tree_group_.Resize(qu_, model.tree_info.size());
    for (size_t tree_idx = 0; tree_idx < model.tree_info.size(); tree_idx++)
      tree_group_[tree_idx] = model.tree_info[tree_idx];

    tree_beg_ = tree_begin;
    tree_end_ = tree_end;
    num_group_ = model.learner_model_param->num_output_group; 
  }
};

float GetFvalue(int ridx, int fidx, Entry* data, size_t* row_ptr, bool& is_missing) {
  // Binary search
  auto begin_ptr = data + row_ptr[ridx];
  auto end_ptr = data + row_ptr[ridx + 1];
  Entry* previous_middle = nullptr;
  while (end_ptr != begin_ptr) {
    auto middle = begin_ptr + (end_ptr - begin_ptr) / 2;
    if (middle == previous_middle) {
      break;
    } else {
      previous_middle = middle;
    }

    if (middle->index == fidx) {
      is_missing = false;
      return middle->fvalue;
    } else if (middle->index < fidx) {
      begin_ptr = middle;
    } else {
      end_ptr = middle;
    }
  }
  is_missing = true;
  return 0.0;
}

float GetLeafWeight(int ridx, const DeviceNodeOneAPI* tree, Entry* data, size_t* row_ptr) {
  DeviceNodeOneAPI n = tree[0];
  int node_id = 0;
  bool is_missing;
  while (!n.IsLeaf()) {
    float fvalue = GetFvalue(ridx, n.GetFidx(), data, row_ptr, is_missing);
    // Missing value
    if (is_missing) {
      n = tree[n.MissingIdx()];
    } else {
      if (fvalue < n.GetFvalue()) {
        node_id = n.left_child_idx;
        n = tree[n.left_child_idx];
      } else {
        node_id = n.right_child_idx;
        n = tree[n.right_child_idx];
      }
    }
  }
  return n.GetWeight();
}

void DevicePredictInternal(sycl::queue qu,
                           DeviceMatrixOneAPI* dmat,
                           HostDeviceVector<float>* out_preds,
                           const gbm::GBTreeModel& model,
                           size_t tree_begin,
                           size_t tree_end) {
  if (tree_end - tree_begin == 0) {
    return;
  }
  DeviceModelOneAPI device_model;
  device_model.Init(qu, model, tree_begin, tree_end);

  auto& out_preds_vec = out_preds->HostVector();

  DeviceNodeOneAPI* nodes = device_model.nodes_.Data();
  sycl::buffer<float, 1> out_preds_buf(out_preds_vec.data(), out_preds_vec.size());
  size_t* tree_segments = device_model.tree_segments_.Data();
  int* tree_group = device_model.tree_group_.Data();
  size_t* row_ptr = dmat->row_ptr.Data();
  Entry* data = dmat->data.Data();
  int num_features = dmat->p_mat->Info().num_col_;
  int num_rows = dmat->row_ptr.Size() - 1;
  int num_group = model.learner_model_param->num_output_group;

  qu.submit([&](sycl::handler& cgh) {
    auto out_predictions = out_preds_buf.template get_access<sycl::access::mode::read_write>(cgh);
    cgh.parallel_for<>(sycl::range<1>(num_rows), [=](sycl::id<1> pid) {
      int global_idx = pid[0];
      if (global_idx >= num_rows) return;
      if (num_group == 1) {
        float sum = 0.0;
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const DeviceNodeOneAPI* tree = nodes + tree_segments[tree_idx - tree_begin];
          sum += GetLeafWeight(global_idx, tree, data, row_ptr);
        }
        out_predictions[global_idx] += sum;
      } else {
        for (int tree_idx = tree_begin; tree_idx < tree_end; tree_idx++) {
          const DeviceNodeOneAPI* tree = nodes + tree_segments[tree_idx - tree_begin];
          int out_prediction_idx = global_idx * num_group + tree_group[tree_idx];
          out_predictions[out_prediction_idx] += GetLeafWeight(global_idx, tree, data, row_ptr);
        }
      }
    });
  }).wait();
}

class GPUPredictorOneAPI : public Predictor {
 protected:
  void InitOutPredictions(const MetaInfo& info,
                          HostDeviceVector<bst_float>* out_preds,
                          const gbm::GBTreeModel& model) const {
    CHECK_NE(model.learner_model_param->num_output_group, 0);
    size_t n = model.learner_model_param->num_output_group * info.num_row_;
    const auto& base_margin = info.base_margin_.Data()->HostVector();
    out_preds->Resize(n);
    std::vector<bst_float>& out_preds_h = out_preds->HostVector();
    if (base_margin.size() == n) {
      CHECK_EQ(out_preds->Size(), n);
      std::copy(base_margin.begin(), base_margin.end(), out_preds_h.begin());
    } else {
      auto base_score = model.learner_model_param->BaseScore(ctx_)(0);
      if (!base_margin.empty()) {
        std::ostringstream oss;
        oss << "Ignoring the base margin, since it has incorrect length. "
            << "The base margin must be an array of length ";
        if (model.learner_model_param->num_output_group > 1) {
          oss << "[num_class] * [number of data points], i.e. "
              << model.learner_model_param->num_output_group << " * " << info.num_row_
              << " = " << n << ". ";
        } else {
          oss << "[number of data points], i.e. " << info.num_row_ << ". ";
        }
        oss << "Instead, all data points will use "
            << "base_score = " << base_score;
        LOG(WARNING) << oss.str();
      }
      std::fill(out_preds_h.begin(), out_preds_h.end(), base_score);
    }
  }

 public:
  explicit GPUPredictorOneAPI(Context const* context) :
      Predictor::Predictor{context}, cpu_predictor(Predictor::Create("cpu_predictor", context)) {
    std::vector<sycl::device> devices = sycl::device::get_devices();
    if (context->device_id != Context::kDefaultId) {
      qu_ = sycl::queue(devices[context->device_id]);
    } else {
      // sycl::default_selector selector;
      // qu_ = sycl::queue(selector);
      if (rabit::IsDistributed()) {
        qu_ = sycl::queue(devices[rabit::GetRank()]);
      } else {
        qu_ = sycl::queue(sycl::default_selector());
    }
    }
  }

  void PredictBatch(DMatrix *dmat, PredictionCacheEntry *predts,
                    const gbm::GBTreeModel &model, uint32_t tree_begin,
                    uint32_t tree_end = 0) const override {
    // Existing caching approach is not valid due to the const modifier of the PredictBatch method
    /* 
    if (this->device_matrix_cache_.find(dmat) ==
        this->device_matrix_cache_.end()) {
      this->device_matrix_cache_.emplace(
          dmat, std::unique_ptr<DeviceMatrixOneAPI>(
                    new DeviceMatrixOneAPI(qu_, dmat)));
    }
    DeviceMatrixOneAPI* device_matrix = device_matrix_cache_.find(dmat)->second.get();
    */

    DeviceMatrixOneAPI device_matrix(qu_, dmat); // TODO: remove temporary workaround after cache fix

    auto* out_preds = &predts->predictions;
    if (tree_end == 0) {
      tree_end = model.trees.size();
    }

    if (tree_begin < tree_end) {
      DevicePredictInternal(qu_, &device_matrix, out_preds, model, tree_begin, tree_end);
    }
  }

  bool InplacePredict(std::shared_ptr<DMatrix> p_m,
                      const gbm::GBTreeModel &model, float missing,
                      PredictionCacheEntry *out_preds, uint32_t tree_begin,
                      unsigned tree_end) const override {
    return cpu_predictor->InplacePredict(p_m, model, missing, out_preds, tree_begin, tree_end);
  }

  void PredictInstance(const SparsePage::Inst& inst,
                       std::vector<bst_float>* out_preds,
                       const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    cpu_predictor->PredictInstance(inst, out_preds, model, ntree_limit);
  }

  void PredictLeaf(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_preds,
                   const gbm::GBTreeModel& model, unsigned ntree_limit) const override {
    cpu_predictor->PredictLeaf(p_fmat, out_preds, model, ntree_limit);
  }

  void PredictContribution(DMatrix* p_fmat, HostDeviceVector<float>* out_contribs,
                           const gbm::GBTreeModel& model, uint32_t ntree_limit,
                           const std::vector<bst_float>* tree_weights,
                           bool approximate, int condition,
                           unsigned condition_feature) const override {
    cpu_predictor->PredictContribution(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate, condition, condition_feature);
  }

  void PredictInteractionContributions(DMatrix* p_fmat, HostDeviceVector<bst_float>* out_contribs,
                                       const gbm::GBTreeModel& model, unsigned ntree_limit,
                                       const std::vector<bst_float>* tree_weights,
                                       bool approximate) const override {
    cpu_predictor->PredictInteractionContributions(p_fmat, out_contribs, model, ntree_limit, tree_weights, approximate);
  }

 private:
  sycl::queue qu_;

  std::unique_ptr<Predictor> cpu_predictor;

  std::unordered_map<DMatrix*, std::unique_ptr<DeviceMatrixOneAPI>> device_matrix_cache_;
};

XGBOOST_REGISTER_PREDICTOR(PredictorOneAPI, "oneapi_predictor")
.describe("Make predictions using DPC++.")
.set_body([](Context const* context) {
            return new PredictorOneAPI(context);
          });

XGBOOST_REGISTER_PREDICTOR(GPUPredictorOneAPI, "oneapi_predictor_gpu")
.describe("Make predictions using DPC++.")
.set_body([](Context const* context) {
            return new GPUPredictorOneAPI(context);
          });

}  // namespace predictor
}  // namespace xgboost
