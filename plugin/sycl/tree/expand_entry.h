/*!
 * Copyright 2017-2021 by Contributors
 * \file updater_quantile_hist.h
 */
#ifndef PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
#define PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtautological-constant-compare"
#include "../../src/tree/constraints.h"
#pragma GCC diagnostic pop

namespace xgboost {
namespace sycl {
namespace tree {
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

}  // namespace tree
}  // namespace sycl
}  // namespace xgboost

#endif  // PLUGIN_SYCL_TREE_EXPAND_ENTRY_H_
