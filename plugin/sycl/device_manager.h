/*!
 * Copyright 2017-2022 by Contributors
 * \file device_manager.h
 */
#ifndef XGBOOST_DEVICE_MANAGER_SYCL_H_
#define XGBOOST_DEVICE_MANAGER_SYCL_H_

#include <vector>
#include <mutex>
#include <string>

#include "CL/sycl.hpp"
#include "xgboost/context.h"

namespace xgboost {
namespace sycl {

class DeviceManager {
 public:
  ::sycl::queue GetQueue(const DeviceOrd& device_spec) const;

  ::sycl::device GetDevice(const DeviceOrd& device_spec) const;

 private:
  using QueueRegister_t = std::unordered_map<std::string, ::sycl::queue>;

  struct DeviceRegister {
    std::vector<::sycl::device> devices;
    std::vector<::sycl::device> cpu_devices;
    std::vector<::sycl::device> gpu_devices;
  };

  QueueRegister_t& GetQueueRegister() const;

  DeviceRegister& GetDevicesRegister() const;

  mutable std::mutex queue_registering_mutex;
  mutable std::mutex device_registering_mutex;
};

}  // namespace sycl
}  // namespace xgboost

#endif  // XGBOOST_DEVICE_MANAGER_SYCL_H_