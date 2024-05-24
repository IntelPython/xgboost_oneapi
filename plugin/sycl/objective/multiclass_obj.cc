/*!
 * Copyright 2015-2023 by Contributors
 * \file multiclass_obj.cc
 * \brief Definition of multi-class classification objectives.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#pragma GCC diagnostic ignored "-W#pragma-messages"
#include <rabit/rabit.h>
#pragma GCC diagnostic pop

#include <vector>
#include <algorithm>
#include <limits>
#include <utility>

#include "xgboost/parameter.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "xgboost/data.h"
#pragma GCC diagnostic pop
#include "xgboost/logging.h"
#include "xgboost/objective.h"
#include "xgboost/json.h"

#include "../device_manager.h"
#include "../data.h"
#include <CL/sycl.hpp>


namespace xgboost {
namespace sycl {
namespace obj {


DMLC_REGISTRY_FILE_TAG(multiclass_obj_sycl);


/*!
 * \brief Do inplace softmax transformaton on start to end
 *
 * \tparam Iterator Input iterator type
 *
 * \param start Start iterator of input
 * \param end end iterator of input
 */
template <typename Iterator>
inline void Softmax(Iterator start, Iterator end) {
  bst_float wmax = *start;
  for (Iterator i = start+1; i != end; ++i) {
    wmax = ::sycl::max(*i, wmax);
  }
  float wsum = 0.0f;
  for (Iterator i = start; i != end; ++i) {
    *i = ::sycl::exp(*i - wmax);
    wsum += *i;
  }
  for (Iterator i = start; i != end; ++i) {
    *i /= static_cast<float>(wsum);
  }
}


/*!
 * \brief Find the maximum iterator within the iterators
 * \param begin The begining iterator.
 * \param end The end iterator.
 * \return the iterator point to the maximum value.
 * \tparam Iterator The type of the iterator.
 */
template<typename Iterator>
inline Iterator FindMaxIndex(Iterator begin, Iterator end) {
  Iterator maxit = begin;
  for (Iterator it = begin; it != end; ++it) {
    if (*it > *maxit) maxit = it;
  }
  return maxit;
}


struct SoftmaxMultiClassParam : public XGBoostParameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};


class SoftmaxMultiClassObj : public ObjFunction {
  static constexpr size_t kBatchSize = 1u << 22;
  mutable bool are_buffs_init = false;

  void InitBuffers() const {
    if (!are_buffs_init) {
      events_.resize(5);
      in_buff1_.Resize(&qu_, kBatchSize);
      in_buff2_.Resize(&qu_, kBatchSize);
      in_buff3_.Resize(&qu_, kBatchSize);
      out_gpair_.Resize(&qu_, kBatchSize);
      are_buffs_init = true;
    }
  }

 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
  : output_prob_(output_prob) {}


  void Configure(Args const& args) override {
    param_.UpdateAllowUnknown(args);
    qu_ = device_manager.GetQueue(ctx_->Device());
  }


  void GetGradient(const HostDeviceVector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   HostDeviceVector<GradientPair>* out_gpair) override {
    InitBuffers();
    if (preds.Size() == 0) return;
    if (info.labels.Size() == 0) return;

    CHECK(preds.Size() == (static_cast<size_t>(param_.num_class) * info.labels.Size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match.\n"
        << "label.Size() * num_class: "
        << info.labels.Size() * static_cast<size_t>(param_.num_class) << "\n"
        << "num_class: " << param_.num_class << "\n"
        << "preds.Size(): " << preds.Size();


    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(preds.Size() / nclass);

    out_gpair->Resize(preds.Size());

    const bool is_null_weight = info.weights_.Size() == 0;
    if (!is_null_weight) {
      CHECK_EQ(info.weights_.Size(), ndata)
          << "Number of weights should be equal to number of data points.";
    }

    bst_float* preds_ptr = in_buff1_.Data();
    bst_float* labels_ptr = in_buff2_.Data();
    bst_float* weights_ptr = in_buff3_.Data();
    GradientPair* out_gpair_ptr = out_gpair_.Data();

    int flag = 1;
    int wg_size = 32;
    const size_t nBatch = ndata / kBatchSize + (ndata % kBatchSize > 0);
    {
      ::sycl::buffer<int, 1> flag_buf(&flag, 1);
      for (size_t batch = 0; batch < nBatch; ++batch) {
        const size_t begin = batch * kBatchSize;
        const size_t end = (batch == nBatch - 1) ? ndata : begin + kBatchSize;
        const size_t batch_size = end - begin;
        int nwgs = (batch_size / wg_size + (batch_size % wg_size > 0));

        events_[0] = qu_.memcpy(preds_ptr, preds.HostPointer() + begin * nclass,
                                batch_size * nclass * sizeof(bst_float), events_[3]);
        events_[1] = qu_.memcpy(labels_ptr, info.labels.Data()->HostPointer() + begin,
                               batch_size * sizeof(bst_float), events_[3]);
        if (!is_null_weight) {
          events_[2] = qu_.memcpy(weights_ptr, info.weights_.HostPointer() + begin,
                                 info.weights_.Size() * sizeof(bst_float), events_[3]);
        }


        events_[3] = qu_.submit([&](::sycl::handler& cgh) {
          cgh.depends_on(events_);
          auto flag_buf_acc  = flag_buf.get_access<::sycl::access::mode::write>(cgh);
          cgh.parallel_for_work_group<>(::sycl::range<1>(nwgs), ::sycl::range<1>(wg_size),
                                        [=](::sycl::group<1> group) {
            group.parallel_for_work_item([&](::sycl::h_item<1> item) {
              const size_t idx = item.get_global_id()[0];

              const bst_float* pred = preds_ptr + idx * nclass;

              // Part of Softmax function
              bst_float wmax = std::numeric_limits<bst_float>::min();
              for (int k = 0; k < nclass; k++) { wmax = ::sycl::max(pred[k], wmax); }
              bst_float wsum = 0.0f;
              for (int k = 0; k < nclass; k++) { wsum += ::sycl::exp(pred[k] - wmax); }
              bst_float label = labels_ptr[idx];

              if (label < 0 || label >= nclass) {
                AtomicRef<int> flag_ref(flag_buf_acc[0]);
                flag_ref = 0;
                label = 0;
              }

              bst_float wt = is_null_weight ? 1.0f : weights_ptr[idx];
              for (int k = 0; k < nclass; ++k) {
                bst_float p = expf(pred[k] - wmax) / static_cast<float>(wsum);
                const float eps = 1e-16f;
                const bst_float h = ::sycl::max(2.0f * p * (1.0f - p) * wt, eps);
                p = label == k ? p - 1.0f : p;
                out_gpair_ptr[idx * nclass + k] = GradientPair(p * wt, h);
              }
            });
          });
        });
        events_[4] = qu_.memcpy(out_gpair->HostPointer() + begin * nclass, out_gpair_ptr,
                                batch_size * nclass * sizeof(GradientPair), events_[3]);
      }
      qu_.wait_and_throw();
    }
    // flag_buf is destroyed, content is copyed to the "flag"

    if (flag == 0) {
      LOG(FATAL) << "SYCL::SoftmaxMultiClassObj: label must be in [0, num_class).";
    }
  }

  void PredTransform(HostDeviceVector<bst_float>* io_preds) const override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(HostDeviceVector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "mlogloss";
  }


  inline void Transform(HostDeviceVector<bst_float> *io_preds, bool prob) const {
    InitBuffers();
    if (io_preds->Size() == 0) return;
    const int nclass = param_.num_class;
    const auto ndata = static_cast<int64_t>(io_preds->Size() / nclass);

    ::sycl::event event;
    bst_float* preds_ptr = in_buff1_.Data();
    const size_t nBatch = ndata / kBatchSize + (ndata % kBatchSize > 0);
    for (size_t batch = 0; batch < nBatch; ++batch) {
      const size_t begin = batch * kBatchSize;
      const size_t end = (batch == nBatch - 1) ? ndata : begin + kBatchSize;
      const size_t batch_size = end - begin;
      event = qu_.memcpy(preds_ptr, io_preds->HostPointer() + begin * nclass,
                         batch_size * nclass * sizeof(bst_float), event);
      if (prob) {
        event = qu_.submit([&](::sycl::handler& cgh) {
          cgh.depends_on(event);
          cgh.parallel_for<>(::sycl::range<1>(batch_size), [=](::sycl::id<1> pid) {
            int idx = pid[0];
            bst_float* point = preds_ptr + idx * nclass;
            Softmax(point, point + nclass);
          });
        });
        event = qu_.memcpy(io_preds->HostPointer() + begin * nclass, preds_ptr,
                           batch_size * nclass * sizeof(bst_float), event);
      } else {
        bst_float* max_preds_ptr = in_buff2_.Data();
        event = qu_.submit([&](::sycl::handler& cgh) {
          cgh.depends_on(event);
          cgh.parallel_for<>(::sycl::range<1>(batch_size), [=](::sycl::id<1> pid) {
            int idx = pid[0];
            const bst_float* point = preds_ptr + idx * nclass;
            max_preds_ptr[idx] = FindMaxIndex(point, point + nclass) - point;
          });
        });
        event = qu_.memcpy(io_preds->HostPointer() + begin, max_preds_ptr,
                           batch_size * sizeof(bst_float), event);
      }
    }
    qu_.wait_and_throw();
    if (!prob) {
      io_preds->Resize(ndata);
    }
  }

  struct ObjInfo Task() const override {return {ObjInfo::kClassification}; }


  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    if (this->output_prob_) {
      out["name"] = String("multi:softprob_sycl");
    } else {
      out["name"] = String("multi:softmax_sycl");
    }
    out["softmax_multiclass_param"] = ToJson(param_);
  }


  void LoadConfig(Json const& in) override {
    FromJson(in["softmax_multiclass_param"], &param_);
  }


 private:
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
  sycl::DeviceManager device_manager;

  mutable ::sycl::queue qu_;
  mutable std::vector<::sycl::event> events_;
  // Buffers
  // kBatchSize * nclass
  mutable USMVector<bst_float, MemoryType::on_device> in_buff1_;
  // kBatchSize
  mutable USMVector<bst_float, MemoryType::on_device> in_buff2_;
  // kBatchSize
  mutable USMVector<bst_float, MemoryType::on_device> in_buff3_;
  // kBatchSize * nclass
  mutable USMVector<GradientPair, MemoryType::on_device> out_gpair_;
};


// register the objective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);


XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax_sycl")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassObj(false); });


XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob_sycl")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassObj(true); });


}  // namespace obj
}  // namespace sycl
}  // namespace xgboost
