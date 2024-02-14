/*!
 * Copyright 2017-2023 by Contributors
 * \file hist_util.cc
 */
#include <vector>
#include <limits>
#include <algorithm>

#include "../data/gradient_index.h"
#include "hist_util.h"

#include <CL/sycl.hpp>

namespace xgboost {
namespace sycl {
namespace common {

template <typename BinIdxType>
void mergeSort(BinIdxType* begin, BinIdxType* end, BinIdxType* buf) {
  const size_t total_len = end - begin;
  for (size_t block_len = 1; block_len < total_len; block_len <<= 1) {
    for (size_t cur_block = 0; cur_block + block_len < total_len; cur_block += 2 * block_len) {
      size_t start = cur_block;
      size_t mid = start + block_len;
      size_t finish = mid + block_len < total_len ? mid + block_len : total_len;
      size_t left_pos = start;
      size_t right_pos = mid;
      size_t pos = start;
      while (left_pos < mid || right_pos < finish) {
        if (left_pos < mid && (right_pos == finish || begin[left_pos] < begin[right_pos])) {
          buf[pos++] = begin[left_pos++];
        } else {
          buf[pos++] = begin[right_pos++];
        }
      }
      for (size_t i = start; i < finish; i++) begin[i] = buf[i];
    }
  }
}

/*!
 * \brief Fill histogram with zeroes
 */
template<typename GradientSumT>
void InitHist(::sycl::queue qu, GHistRow<GradientSumT, MemoryType::on_device>* hist, size_t size) {
  qu.fill(hist->Begin(), xgboost::detail::GradientPairInternal<GradientSumT>(), size);
}
template void InitHist(::sycl::queue qu,
                       GHistRow<float,  MemoryType::on_device>* hist, size_t size);
template void InitHist(::sycl::queue qu,
                       GHistRow<double, MemoryType::on_device>* hist, size_t size);

/*!
 * \brief Copy histogram from src to dst
 */
template<typename GradientSumT>
void CopyHist(::sycl::queue qu,
              GHistRow<GradientSumT, MemoryType::on_device>* dst,
              const GHistRow<GradientSumT, MemoryType::on_device>& src,
              size_t size) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst->Data());
  const GradientSumT* psrc = reinterpret_cast<const GradientSumT*>(src.DataConst());

  qu.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [=](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc[i];
    });
  }).wait();
}
template void CopyHist(::sycl::queue qu,
                       GHistRow<float, MemoryType::on_device>* dst,
                       const GHistRow<float, MemoryType::on_device>& src,
                       size_t size);
template void CopyHist(::sycl::queue qu,
                       GHistRow<double, MemoryType::on_device>* dst,
                       const GHistRow<double, MemoryType::on_device>& src,
                       size_t size);

/*!
 * \brief Compute Subtraction: dst = src1 - src2
 */
template<typename GradientSumT>
::sycl::event SubtractionHist(::sycl::queue qu,
                            GHistRow<GradientSumT, MemoryType::on_device>* dst,
                            const GHistRow<GradientSumT, MemoryType::on_device>& src1,
                            const GHistRow<GradientSumT, MemoryType::on_device>& src2,
                            size_t size, ::sycl::event event_priv) {
  GradientSumT* pdst = reinterpret_cast<GradientSumT*>(dst->Data());
  const GradientSumT* psrc1 = reinterpret_cast<const GradientSumT*>(src1.DataConst());
  const GradientSumT* psrc2 = reinterpret_cast<const GradientSumT*>(src2.DataConst());

  auto event_final = qu.submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_priv);
    cgh.parallel_for<>(::sycl::range<1>(2 * size), [pdst, psrc1, psrc2](::sycl::item<1> pid) {
      const size_t i = pid.get_id(0);
      pdst[i] = psrc1[i] - psrc2[i];
    });
  });
  return event_final;
}
template ::sycl::event SubtractionHist(::sycl::queue qu,
                              GHistRow<float, MemoryType::on_device>* dst,
                              const GHistRow<float, MemoryType::on_device>& src1,
                              const GHistRow<float, MemoryType::on_device>& src2,
                              size_t size, ::sycl::event event_priv);
template ::sycl::event SubtractionHist(::sycl::queue qu,
                              GHistRow<double, MemoryType::on_device>* dst,
                              const GHistRow<double, MemoryType::on_device>& src1,
                              const GHistRow<double, MemoryType::on_device>& src2,
                              size_t size, ::sycl::event event_priv);

// Kernel with buffer using
template<typename FPType, typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue qu,
                            const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                            ::sycl::event event_priv) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist->Data());
  const size_t nbins = gmat.nbins;

  const size_t max_feat_local =
    qu.get_device().get_info<::sycl::info::device::max_work_group_size>();
  const size_t feat_local = n_columns < max_feat_local ? n_columns : max_feat_local;

  const size_t max_nblocks = hist_buffer->Size() / (nbins * 2);
  const size_t min_block_size = 128;
  size_t nblocks = std::min(max_nblocks, size / min_block_size + !!(size % min_block_size));
  const size_t block_size = size / nblocks + !!(size % nblocks);
  FPType* hist_buffer_data = reinterpret_cast<FPType*>(hist_buffer->Data());

  auto event_fill = qu.fill(hist_buffer_data, FPType(0), nblocks * nbins * 2, event_priv);
  auto event_main = qu.submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_fill);
    cgh.parallel_for<>(::sycl::nd_range<2>(::sycl::range<2>(nblocks, feat_local),
                                           ::sycl::range<2>(1, feat_local)),
                       [=](::sycl::nd_item<2> pid) {
      size_t block = pid.get_global_id(0);
      size_t feat = pid.get_global_id(1);

      FPType* hist_local = hist_buffer_data + block * nbins * 2;
      for (size_t idx = 0; idx < block_size; ++idx) {
        size_t i = block * block_size + idx;
        if (i < size) {
          const size_t icol_start = n_columns * rid[i];
          const size_t idx_gh = rid[i];

          pid.barrier(::sycl::access::fence_space::local_space);
          const BinIdxType* gr_index_local = gradient_index + icol_start;

          for (size_t j = feat; j < n_columns; j += feat_local) {
            uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
            if constexpr (isDense) {
              idx_bin += offsets[j];
            }
            if (idx_bin < nbins) {
              hist_local[2 * idx_bin]   += pgh[2 * idx_gh];
              hist_local[2 * idx_bin+1] += pgh[2 * idx_gh+1];
            }
          }
        }
      }
    });
  });

  auto event_save = qu.submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_main);
    cgh.parallel_for<>(::sycl::range<1>(nbins), [=](::sycl::item<1> pid) {
      size_t idx_bin = pid.get_id(0);

      FPType gsum = 0.0f;
      FPType hsum = 0.0f;

      for (size_t j = 0; j < nblocks; ++j) {
        gsum += hist_buffer_data[j * nbins * 2 + 2 * idx_bin];
        hsum += hist_buffer_data[j * nbins * 2 + 2 * idx_bin + 1];
      }

      hist_data[2 * idx_bin] = gsum;
      hist_data[2 * idx_bin + 1] = hsum;
    });
  });
  return event_save;
}

// Kernel with atomic using
template<typename FPType, typename BinIdxType, bool isDense>
::sycl::event BuildHistKernel(::sycl::queue qu,
                            const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            ::sycl::event event_priv) {
  const size_t size = row_indices.Size();
  const size_t* rid = row_indices.begin;
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const float* pgh = reinterpret_cast<const float*>(gpair_device.DataConst());
  const BinIdxType* gradient_index = gmat.index.data<BinIdxType>();
  const uint32_t* offsets = gmat.index.Offset();
  FPType* hist_data = reinterpret_cast<FPType*>(hist->Data());
  const size_t nbins = gmat.nbins;

  const size_t max_feat_local =
    qu.get_device().get_info<::sycl::info::device::max_work_group_size>();
  const size_t feat_local = n_columns < max_feat_local ? n_columns : max_feat_local;

  auto event_fill = qu.fill(hist_data, FPType(0), nbins * 2, event_priv);
  auto event_main = qu.submit([&](::sycl::handler& cgh) {
    cgh.depends_on(event_fill);
    cgh.parallel_for<>(::sycl::range<2>(size, feat_local),
                      [=](::sycl::item<2> pid) {
      size_t i = pid.get_id(0);
      size_t feat = pid.get_id(1);

      const size_t icol_start = n_columns * rid[i];
      const size_t idx_gh = rid[i];

      const BinIdxType* gr_index_local = gradient_index + icol_start;

      for (size_t j = feat; j < n_columns; j += feat_local) {
        uint32_t idx_bin = static_cast<uint32_t>(gr_index_local[j]);
        if constexpr (isDense) {
          idx_bin += offsets[j];
        }
        if (idx_bin < nbins) {
          AtomicRef<FPType> gsum(hist_data[2 * idx_bin]);
          AtomicRef<FPType> hsum(hist_data[2 * idx_bin + 1]);
          gsum.fetch_add(pgh[2 * idx_gh]);
          hsum.fetch_add(pgh[2 * idx_gh + 1]);
        }
      }
    });
  });
  return event_main;
}

template<typename FPType, typename BinIdxType>
::sycl::event BuildHistDispatchKernel(
                ::sycl::queue qu,
                const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                const RowSetCollection::Elem& row_indices,
                const GHistIndexMatrix& gmat,
                GHistRow<FPType, MemoryType::on_device>* hist,
                bool isDense,
                GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                ::sycl::event events_priv) {
  const size_t size = row_indices.Size();
  const size_t n_columns = isDense ? gmat.nfeatures : gmat.row_stride;
  const size_t nbins = gmat.nbins;

  const size_t max_feat_local =
    qu.get_device().get_info<::sycl::info::device::max_work_group_size>();
  const size_t feat_local = n_columns < max_feat_local ? n_columns : max_feat_local;

  // max cycle size, while atomics are still effective
  const size_t max_cycle_size_atomics = nbins;
  const size_t cycle_size = size;
  if (cycle_size > max_cycle_size_atomics) {
    if (isDense) {
      return BuildHistKernel<FPType, BinIdxType, true>(qu, gpair_device, row_indices,
                                                       gmat, hist, hist_buffer,
                                                       events_priv);
    } else {
      return BuildHistKernel<FPType, uint32_t, false>(qu, gpair_device, row_indices,
                                                      gmat, hist, hist_buffer,
                                                      events_priv);
    }
  } else {
    if (isDense) {
      return BuildHistKernel<FPType, BinIdxType, true>(qu, gpair_device, row_indices,
                                                       gmat, hist, events_priv);
    } else {
      return BuildHistKernel<FPType, uint32_t, false>(qu, gpair_device, row_indices,
                                                      gmat, hist, events_priv);
    }
  }
}

template<typename FPType>
::sycl::event BuildHistKernel(::sycl::queue qu,
                            const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
                            const RowSetCollection::Elem& row_indices,
                            const GHistIndexMatrix& gmat, const bool isDense,
                            GHistRow<FPType, MemoryType::on_device>* hist,
                            GHistRow<FPType, MemoryType::on_device>* hist_buffer,
                            ::sycl::event event_priv) {
  const bool is_dense = isDense;
  switch (gmat.index.GetBinTypeSize()) {
    case BinTypeSize::kUint8BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint8_t>(qu, gpair_device, row_indices,
                                                      gmat, hist, is_dense, hist_buffer,
                                                      event_priv);
      break;
    case BinTypeSize::kUint16BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint16_t>(qu, gpair_device, row_indices,
                                                       gmat, hist, is_dense, hist_buffer,
                                                       event_priv);
      break;
    case BinTypeSize::kUint32BinsTypeSize:
      return BuildHistDispatchKernel<FPType, uint32_t>(qu, gpair_device, row_indices,
                                                       gmat, hist, is_dense, hist_buffer,
                                                       event_priv);
      break;
    default:
      CHECK(false);  // no default behavior
  }
}

template <typename GradientSumT>
::sycl::event GHistBuilder<GradientSumT>::BuildHist(
              const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix &gmat,
              GHistRowT<MemoryType::on_device>* hist,
              bool isDense,
              GHistRowT<MemoryType::on_device>* hist_buffer,
              ::sycl::event event_priv) {
  return BuildHistKernel<GradientSumT>(qu_, gpair_device, row_indices, gmat,
                                       isDense, hist, hist_buffer, event_priv);
}

template
::sycl::event GHistBuilder<float>::BuildHist(
              const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix& gmat,
              GHistRow<float, MemoryType::on_device>* hist,
              bool isDense,
              GHistRow<float, MemoryType::on_device>* hist_buffer,
              ::sycl::event event_priv);
template
::sycl::event GHistBuilder<double>::BuildHist(
              const USMVector<GradientPair, MemoryType::on_device>& gpair_device,
              const RowSetCollection::Elem& row_indices,
              const GHistIndexMatrix& gmat,
              GHistRow<double, MemoryType::on_device>* hist,
              bool isDense,
              GHistRow<double, MemoryType::on_device>* hist_buffer,
              ::sycl::event event_priv);

template<typename GradientSumT>
void GHistBuilder<GradientSumT>::SubtractionTrick(GHistRowT<MemoryType::on_device>* self,
                                                  const GHistRowT<MemoryType::on_device>& sibling,
                                                  const GHistRowT<MemoryType::on_device>& parent) {
  const size_t size = self->Size();
  CHECK_EQ(sibling.Size(), size);
  CHECK_EQ(parent.Size(), size);

  SubtractionHist(qu_, self, parent, sibling, size, ::sycl::event());
}
template
void GHistBuilder<float>::SubtractionTrick(GHistRow<float, MemoryType::on_device>* self,
                                           const GHistRow<float, MemoryType::on_device>& sibling,
                                           const GHistRow<float, MemoryType::on_device>& parent);
template
void GHistBuilder<double>::SubtractionTrick(GHistRow<double, MemoryType::on_device>* self,
                                            const GHistRow<double, MemoryType::on_device>& sibling,
                                            const GHistRow<double, MemoryType::on_device>& parent);
}  // namespace common
}  // namespace sycl
}  // namespace xgboost