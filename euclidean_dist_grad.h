#ifndef TENSORFLOW_KERNELS_EUCLIDEAN_DIST_OP_H_
#define TENSORFLOW_KERNELS_EUCLIDEAN_DIST_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchEuclideanDistGrad;

template <typename T>
struct LaunchEuclideanDistGradGPU
{
  static void launch(
      OpKernelContext*, OpKernel*, const Tensor&, const Tensor&,
      const Tensor&, const Tensor&, Tensor*, Tensor*);
};
extern template struct LaunchEuclideanDistGradGPU<double>;
extern template struct LaunchEuclideanDistGradGPU<float>;
extern template struct LaunchEuclideanDistGradGPU<int>;

}  // end namespace tensorflow

#endif  // TENSORFLOW_KERNELS_EUCLIDEAN_DIST_OP_H_
