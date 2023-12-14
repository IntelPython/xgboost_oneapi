/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist.h
 */
#ifndef PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
#define PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_

#include <dmlc/timer.h>
#include <rabit/rabit.h>
#include <xgboost/tree_updater.h>

#include <queue>
#include <utility>
#include <memory>
#include <vector>

#include "../common/hist_util.h"
#include "../common/row_set.h"
#include "../common/partition_builder.h"
#include "split_evaluator.h"
#include "../device_manager.h"

#include "xgboost/data.h"
#include "xgboost/json.h"
#include "../../src/tree/constraints.h"
#include "../../src/common/random.h"

namespace xgboost {
namespace sycl {
/*!
 * \brief A C-style array with in-stack allocation.
          As long as the array is smaller than MaxStackSize, it will be allocated inside the stack. Otherwise, it will be heap-allocated.
          Temporary copy of implementation to remove dependency on updater_quantile_hist.h
 */
template<typename T, size_t MaxStackSize>
class MemStackAllocator {
 public:
  explicit MemStackAllocator(size_t required_size): required_size_(required_size) {
  }

  T* Get() {
    if (!ptr_) {
      if (MaxStackSize >= required_size_) {
        ptr_ = stack_mem_;
      } else {
        ptr_ =  reinterpret_cast<T*>(malloc(required_size_ * sizeof(T)));
        do_free_ = true;
      }
    }

    return ptr_;
  }

  ~MemStackAllocator() {
    if (do_free_) free(ptr_);
  }


 private:
  T* ptr_ = nullptr;
  bool do_free_ = false;
  size_t required_size_;
  T stack_mem_[MaxStackSize];
};

namespace tree {

using xgboost::sycl::common::HistCollection;
using xgboost::sycl::common::GHistBuilder;
using xgboost::sycl::common::GHistIndexMatrix;
using xgboost::sycl::common::GHistRow;
using xgboost::sycl::common::RowSetCollection;
using xgboost::sycl::common::PartitionBuilder;

template <typename GradientSumT>
class HistSynchronizer;

template <typename GradientSumT>
class BatchHistSynchronizer;

template <typename GradientSumT>
class DistributedHistSynchronizer;

template <typename GradientSumT>
class HistRowsAdder;

template <typename GradientSumT>
class BatchHistRowsAdder;

template <typename GradientSumT>
class DistributedHistRowsAdder;

// training parameters specific to this algorithm
struct HistMakerTrainParam
    : public XGBoostParameter<HistMakerTrainParam> {
  bool single_precision_histogram = false;
  // declare parameters
  DMLC_DECLARE_PARAMETER(HistMakerTrainParam) {
    DMLC_DECLARE_FIELD(single_precision_histogram).set_default(false).describe(
        "Use single precision to build histograms.");
  }
};

// data structure
template<typename GradType>
struct NodeEntry {
  /*! \brief statics for node entry */
  GradStats<GradType> stats;
  /*! \brief loss of this node, without split */
  GradType root_gain;
  /*! \brief weight calculated related to current data */
  GradType weight;
  /*! \brief current best solution */
  SplitEntry<GradType> best;
  // constructor
  explicit NodeEntry(const xgboost::tree::TrainParam& param)
      : root_gain(0.0f), weight(0.0f) {}
};
// actual builder that runs the algorithm

/*! \brief construct a tree using quantized feature values with SYCL backend*/
class QuantileHistMaker: public TreeUpdater {
 public:
  QuantileHistMaker(Context const* ctx, ObjInfo const * task) :
                             TreeUpdater(ctx), ctx_(ctx), task_{task} {
    updater_monitor_.Init("SYCLQuantileHistMaker");
  }
  void Configure(const Args& args) override;

  void Update(xgboost::tree::TrainParam const *param,
              HostDeviceVector<GradientPair>* gpair,
              DMatrix* dmat,
              xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
              const std::vector<RegTree*>& trees) override;

  bool UpdatePredictionCache(const DMatrix* data,
                             linalg::MatrixView<float> out_preds) override;

  void LoadConfig(Json const& in) override {
    auto const& config = get<Object const>(in);
    FromJson(config.at("train_param"), &this->param_);
    try {
      FromJson(config.at("sycl_hist_train_param"), &this->hist_maker_param_);
    } catch (std::out_of_range& e) {
      // XGBoost model is from 1.1.x, so 'cpu_hist_train_param' is missing.
      // We add this compatibility check because it's just recently that we (developers) began
      // persuade R users away from using saveRDS() for model serialization. Hopefully, one day,
      // everyone will be using xgb.save().
      LOG(WARNING) << "Attempted to load interal configuration for a model file that was generated "
        << "by a previous version of XGBoost. A likely cause for this warning is that the model "
        << "was saved with saveRDS() in R or pickle.dump() in Python. We strongly ADVISE AGAINST "
        << "using saveRDS() or pickle.dump() so that the model remains accessible in current and "
        << "upcoming XGBoost releases. Please use xgb.save() instead to preserve models for the "
        << "long term. For more details and explanation, see "
        << "https://xgboost.readthedocs.io/en/latest/tutorials/saving_model.html";
      this->hist_maker_param_.UpdateAllowUnknown(Args{});
    }
  }
  void SaveConfig(Json* p_out) const override {
    auto& out = *p_out;
    out["train_param"] = ToJson(param_);
    out["sycl_hist_train_param"] = ToJson(hist_maker_param_);
  }

  char const* Name() const override {
    return "grow_quantile_histmaker_sycl";
  }

 protected:
  template <typename GradientSumT>
  friend class HistSynchronizer;
  template <typename GradientSumT>
  friend class BatchHistSynchronizer;
  template <typename GradientSumT>
  friend class DistributedHistSynchronizer;

  template <typename GradientSumT>
  friend class HistRowsAdder;
  template <typename GradientSumT>
  friend class BatchHistRowsAdder;
  template <typename GradientSumT>
  friend class DistributedHistRowsAdder;

  HistMakerTrainParam hist_maker_param_;
  // training parameter
  xgboost::tree::TrainParam param_;
  // quantized data matrix
  GHistIndexMatrix gmat_;
  // (optional) data matrix with feature grouping
  // column accessor
  DMatrix const* p_last_dmat_ {nullptr};
  bool is_gmat_initialized_ {false};

  template<typename GradientSumT>
  struct Builder {
   public:
    template <MemoryType memory_type = MemoryType::shared>
    using GHistRowT = GHistRow<GradientSumT, memory_type>;
    using GradientPairT = xgboost::detail::GradientPairInternal<GradientSumT>;
    // constructor
    explicit Builder(::sycl::queue qu,
                     const xgboost::tree::TrainParam& param,
                     std::unique_ptr<TreeUpdater> pruner,
                     FeatureInteractionConstraintHost int_constraints_,
                     DMatrix const* fmat)
      : qu_(qu), param_(param),
        tree_evaluator_(qu, param, fmat->Info().num_col_),
        pruner_(std::move(pruner)),
        interaction_constraints_{std::move(int_constraints_)},
        p_last_tree_(nullptr), p_last_fmat_(fmat),
        snode_(&qu, 1u << (param.max_depth + 1), NodeEntry<GradientSumT>(param)) {
      builder_monitor_.Init("SYCL::Quantile::Builder");
      kernel_monitor_.Init("SYCL::Quantile::Kernels");
    }
    // update one tree, growing
    void Update(Context const * ctx,
                xgboost::tree::TrainParam const *param,
                const GHistIndexMatrix &gmat,
                HostDeviceVector<GradientPair> *gpair,
                const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                DMatrix *p_fmat,
                xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                RegTree *p_tree);

    inline ::sycl::event BuildHist(
                          const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                          const RowSetCollection::Elem row_indices,
                          const GHistIndexMatrix& gmat,
                          GHistRowT<MemoryType::on_device>* hist,
                          GHistRowT<MemoryType::on_device>* hist_buffer,
                          ::sycl::event event_priv) {
      return hist_builder_.BuildHist(gpair_device, row_indices, gmat, hist,
                                     data_layout_ != kSparseData, hist_buffer, event_priv);
    }

    inline void SubtractionTrick(GHistRowT<MemoryType::on_device>* self,
                                 const GHistRowT<MemoryType::on_device>& sibling,
                                 const GHistRowT<MemoryType::on_device>& parent) {
      builder_monitor_.Start("SubtractionTrick");
      hist_builder_.SubtractionTrick(self, sibling, parent);
      builder_monitor_.Stop("SubtractionTrick");
    }

    bool UpdatePredictionCache(const DMatrix* data,
                               linalg::MatrixView<float> p_out_preds);
    void SetHistSynchronizer(HistSynchronizer<GradientSumT>* sync);
    void SetHistRowsAdder(HistRowsAdder<GradientSumT>* adder);

    // initialize temp data structure
    void InitData(Context const * ctx,
                  const GHistIndexMatrix& gmat,
                  const std::vector<GradientPair>& gpair,
                  const USMVector<GradientPair, MemoryType::on_device> &gpair_device,
                  const DMatrix& fmat,
                  const RegTree& tree);

   protected:
    friend class HistSynchronizer<GradientSumT>;
    friend class BatchHistSynchronizer<GradientSumT>;
    friend class DistributedHistSynchronizer<GradientSumT>;
    friend class HistRowsAdder<GradientSumT>;
    friend class BatchHistRowsAdder<GradientSumT>;
    friend class DistributedHistRowsAdder<GradientSumT>;

    /* tree growing policies */
    struct ExpandEntry {
      static const int kRootNid  = 0;
      static const int kEmptyNid = -1;
      int nid;
      int sibling_nid;
      int depth;
      bst_float loss_chg;
      unsigned timestamp;
      ExpandEntry(int nid, int sibling_nid, int depth, bst_float loss_chg,
                  unsigned tstmp)
          : nid(nid), sibling_nid(sibling_nid), depth(depth),
            loss_chg(loss_chg), timestamp(tstmp) {}

      bool IsValid(xgboost::tree::TrainParam const &param, int32_t num_leaves) const {
        bool ret = loss_chg <= kRtEps ||
                   (param.max_depth > 0 && this->depth == param.max_depth) ||
                   (param.max_leaves > 0 && num_leaves == param.max_leaves);
        return ret;
      }
    };

    struct SplitQuery {
      int nid;
      int fid;
      SplitEntry<GradientSumT> best;
      const GradientPairT* hist;
    };

    void InitSampling(const std::vector<GradientPair>& gpair,
                      const USMVector<GradientPair, MemoryType::on_device> &gpair_device,
                      const DMatrix& fmat, USMVector<size_t, MemoryType::on_device>* row_indices);

    void EvaluateSplits(const std::vector<ExpandEntry>& nodes_set,
                        const GHistIndexMatrix& gmat,
                        const HistCollection<GradientSumT, MemoryType::on_device>& hist,
                        const RegTree& tree);

    // Enumerate the split values of specific feature
    // Returns the sum of gradients corresponding to the data points that contains a non-missing
    // value for the particular feature fid.
    template <int d_step>
    static GradStats<GradientSumT> EnumerateSplit(
        const uint32_t* cut_ptr, const bst_float* cut_val, const bst_float* cut_minval,
        const GradientPairT* hist_data, const NodeEntry<GradientSumT> &snode,
        SplitEntry<GradientSumT>* p_best, bst_uint fid, bst_uint nodeID,
        typename TreeEvaluator<GradientSumT>::SplitEvaluator const &evaluator,
        const TrainParam& param);

    static GradStats<GradientSumT> EnumerateSplit(const ::sycl::sub_group& sg,
        const uint32_t* cut_ptr, const bst_float* cut_val, const GradientPairT* hist_data,
        const NodeEntry<GradientSumT> &snode, SplitEntry<GradientSumT>* p_best, bst_uint fid,
        bst_uint nodeID,
        typename TreeEvaluator<GradientSumT>::SplitEvaluator const &evaluator,
        const TrainParam& param);

    void ApplySplit(std::vector<ExpandEntry> nodes,
                        const GHistIndexMatrix& gmat,
                        const HistCollection<GradientSumT, MemoryType::on_device>& hist,
                        RegTree* p_tree);

    template <typename BinIdxType>
    ::sycl::event PartitionKernel(const size_t nid,
                         const int32_t split_cond,
                         const GHistIndexMatrix &gmat,
                         const RegTree::Node& node,
                         xgboost::common::Span<size_t>* rid_buf,
                         size_t* parts_size,
                         ::sycl::event priv_event);

    void AddSplitsToRowSet(const std::vector<ExpandEntry>& nodes, RegTree* p_tree);


    void FindSplitConditions(const std::vector<ExpandEntry>& nodes, const RegTree& tree,
                             const GHistIndexMatrix& gmat, std::vector<int32_t>* split_conditions);

    void InitNewNode(int nid,
                     const GHistIndexMatrix& gmat,
                     const std::vector<GradientPair>& gpair,
                     const DMatrix& fmat,
                     const RegTree& tree);

    // if sum of statistics for non-missing values in the node
    // is equal to sum of statistics for all values:
    // then - there are no missing values
    // else - there are missing values
    static bool SplitContainsMissingValues(const GradStats<GradientSumT>& e,
                                           const NodeEntry<GradientSumT>& snode);

    void ExpandWithDepthWise(const GHistIndexMatrix &gmat,
                             DMatrix *p_fmat,
                             RegTree *p_tree,
                             const std::vector<GradientPair> &gpair,
                             const USMVector<GradientPair, MemoryType::on_device> &gpair_device);

    void BuildLocalHistograms(const GHistIndexMatrix &gmat,
                              RegTree *p_tree,
                              const USMVector<GradientPair, MemoryType::on_device> &gpair_device);

    void BuildHistogramsLossGuide(
                        ExpandEntry entry,
                        const GHistIndexMatrix &gmat,
                        RegTree *p_tree,
                        const USMVector<GradientPair, MemoryType::on_device> &gpair_device);

    // Split nodes to 2 sets depending on amount of rows in each node
    // Histograms for small nodes will be built explicitly
    // Histograms for big nodes will be built by 'Subtraction Trick'
    void SplitSiblings(const std::vector<ExpandEntry>& nodes,
                   std::vector<ExpandEntry>* small_siblings,
                   std::vector<ExpandEntry>* big_siblings,
                   RegTree *p_tree);

    void ParallelSubtractionHist(const xgboost::common::BlockedSpace2d& space,
                                 const std::vector<ExpandEntry>& nodes,
                                 const RegTree * p_tree);

    void BuildNodeStats(const GHistIndexMatrix &gmat,
                        DMatrix *p_fmat,
                        RegTree *p_tree,
                        const std::vector<GradientPair> &gpair);

    void EvaluateAndApplySplits(const GHistIndexMatrix &gmat,
                                RegTree *p_tree,
                                int *num_leaves,
                                int depth,
                                unsigned *timestamp,
                                std::vector<ExpandEntry> *temp_qexpand_depth);

    void AddSplitsToTree(
              const GHistIndexMatrix &gmat,
              RegTree *p_tree,
              int *num_leaves,
              int depth,
              unsigned *timestamp,
              std::vector<ExpandEntry>* nodes_for_apply_split,
              std::vector<ExpandEntry>* temp_qexpand_depth);

    void ExpandWithLossGuide(const GHistIndexMatrix& gmat,
                             DMatrix* p_fmat,
                             RegTree* p_tree,
                             const std::vector<GradientPair> &gpair,
                             const USMVector<GradientPair, MemoryType::on_device>& gpair_device);

    void ReduceHists(const std::vector<int>& sync_ids, size_t nbins);

    inline static bool LossGuide(ExpandEntry lhs, ExpandEntry rhs) {
      if (lhs.loss_chg == rhs.loss_chg) {
        return lhs.timestamp > rhs.timestamp;  // favor small timestamp
      } else {
        return lhs.loss_chg < rhs.loss_chg;  // favor large loss_chg
      }
    }
    //  --data fields--
    const xgboost::tree::TrainParam& param_;
    // number of omp thread used during training
    int nthread_;
    xgboost::common::ColumnSampler column_sampler_;
    // the internal row sets
    RowSetCollection row_set_collection_;
    USMVector<SplitQuery> split_queries_device_;
    /*! \brief TreeNode Data: statistics for each constructed node */
    USMVector<NodeEntry<GradientSumT>> snode_;
    /*! \brief culmulative histogram of gradients. */
    HistCollection<GradientSumT, MemoryType::on_device> hist_;
    /*! \brief culmulative local parent histogram of gradients. */
    HistCollection<GradientSumT, MemoryType::on_device> hist_local_worker_;
    TreeEvaluator<GradientSumT> tree_evaluator_;
    /*! \brief feature with least # of bins. to be used for dense specialization
               of InitNewNode() */
    uint32_t fid_least_bins_;

    GHistBuilder<GradientSumT> hist_builder_;
    std::unique_ptr<TreeUpdater> pruner_;
    FeatureInteractionConstraintHost interaction_constraints_;

    PartitionBuilder partition_builder_;

    // back pointers to tree and data matrix
    const RegTree* p_last_tree_;
    DMatrix const* const p_last_fmat_;

    using ExpandQueue =
       std::priority_queue<ExpandEntry, std::vector<ExpandEntry>,
                           std::function<bool(ExpandEntry, ExpandEntry)>>;

    std::unique_ptr<ExpandQueue> qexpand_loss_guided_;
    std::vector<ExpandEntry> qexpand_depth_wise_;
    // key is the node id which should be calculated by Subtraction Trick, value is the node which
    // provides the evidence for substracts
    std::vector<ExpandEntry> nodes_for_subtraction_trick_;
    // list of nodes whose histograms would be built explicitly.
    std::vector<ExpandEntry> nodes_for_explicit_hist_build_;

    enum DataLayout { kDenseDataZeroBased, kDenseDataOneBased, kSparseData };
    DataLayout data_layout_;

    xgboost::common::Monitor builder_monitor_;
    xgboost::common::Monitor kernel_monitor_;
    constexpr static size_t kNumParallelBuffers = 1;
    std::array<common::ParallelGHistBuilder<GradientSumT>, kNumParallelBuffers> hist_buffers_;
    std::array<::sycl::event, kNumParallelBuffers> hist_build_events_;
    USMVector<size_t, MemoryType::on_device> parts_size_;
    std::vector<size_t> parts_size_cpu_;
    std::vector<::sycl::event> apply_split_events_;
    std::vector<::sycl::event> merge_to_array_events_;
    // rabit::op::Reducer<GradientPairT, GradientPairT::Reduce> histred_;
    std::unique_ptr<HistSynchronizer<GradientSumT>> hist_synchronizer_;
    std::unique_ptr<HistRowsAdder<GradientSumT>> hist_rows_adder_;

    ::sycl::queue qu_;
  };
  xgboost::common::Monitor updater_monitor_;

  template<typename GradientSumT>
  void SetBuilder(std::unique_ptr<Builder<GradientSumT>>*, DMatrix *dmat);

  template<typename GradientSumT>
  void CallBuilderUpdate(const std::unique_ptr<Builder<GradientSumT>>& builder,
                         xgboost::tree::TrainParam const *param,
                         HostDeviceVector<GradientPair> *gpair,
                         DMatrix *dmat,
                         xgboost::common::Span<HostDeviceVector<bst_node_t>> out_position,
                         const std::vector<RegTree *> &trees);

 protected:
  std::unique_ptr<Builder<float>> float_builder_;
  std::unique_ptr<Builder<double>> double_builder_;

  std::unique_ptr<TreeUpdater> pruner_;
  FeatureInteractionConstraintHost int_constraint_;

  ::sycl::queue qu_;
  DeviceManager device_manager;
  Context const* ctx_;
  ObjInfo const *task_{nullptr};
};

template <typename GradientSumT>
class HistSynchronizer {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;

  virtual void SyncHistograms(BuilderT* builder,
                              const std::vector<int>& sync_ids,
                              RegTree *p_tree) = 0;
  virtual ~HistSynchronizer() = default;
};

template <typename GradientSumT>
class BatchHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void SyncHistograms(BuilderT* builder,
                      const std::vector<int>& sync_ids,
                      RegTree *p_tree) override;

  std::vector<::sycl::event> GetEvents() const {
    return hist_sync_events_;
  }

 private:
  std::vector<::sycl::event> hist_sync_events_;
};

template <typename GradientSumT>
class DistributedHistSynchronizer: public HistSynchronizer<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  using ExpandEntryT = typename BuilderT::ExpandEntry;

  void SyncHistograms(BuilderT* builder,
                      const std::vector<int>& sync_ids,
                      RegTree *p_tree) override;

  void ParallelSubtractionHist(BuilderT* builder,
                               const std::vector<ExpandEntryT>& nodes,
                               const RegTree * p_tree);
};

template <typename GradientSumT>
class HistRowsAdder {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;

  virtual void AddHistRows(BuilderT* builder, std::vector<int>* sync_ids, RegTree *p_tree) = 0;
  virtual ~HistRowsAdder() = default;
};

template <typename GradientSumT>
class BatchHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, std::vector<int>* sync_ids, RegTree *p_tree) override;
};

template <typename GradientSumT>
class DistributedHistRowsAdder: public HistRowsAdder<GradientSumT> {
 public:
  using BuilderT = QuantileHistMaker::Builder<GradientSumT>;
  void AddHistRows(BuilderT*, std::vector<int>* sync_ids, RegTree *p_tree) override;
};


}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_UPDATER_QUANTILE_HIST_H_
