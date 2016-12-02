//#include <cmath>

#if GOOGLE_CUDA

#include "euclidean_dist.h"

//#include "cuda/include/cuda.h"
#include "cuda.h"
//#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

template <typename T>
__global__ void EuclideanDist(const T* in1, const T* in2, const int n, const int d, const int k, T* out) {
  for (int in = threadIdx.x; in < n; in += blockDim.x) {
    for (int ik = blockIdx.x; ik < k; ik += gridDim.x) {
      out[in*k+ik] = 0;
      for (int id = 0; id < d; id ++) {
        T temp = in1[in*d+id] - in2[id*k+ik];
        out[in*k+ik] += temp*temp;
      }
      out[in*k+ik] = sqrt((float) out[in*k+ik]);
    }
  }
}

/* // Not available to user kernels
namespace {
template <typename T>
perftools::gputools::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  perftools::gputools::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  perftools::gputools::DeviceMemory<T> typed(wrapped);
  return typed;
}
}  // namespace
*/

template <typename T>
void LaunchEuclideanDistGPU<T>::launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      Tensor* out) {
  const uint64 n = a.dim_size(0);
  const uint64 d = a.dim_size(1);
  const uint64 k = b.dim_size(1);

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  //Not available to user kernels.
  //auto a_ptr = AsDeviceMemory(a.template flat<T>().data());
  //auto b_ptr = AsDeviceMemory(b.template flat<T>().data());
  //auto c_ptr = AsDeviceMemory(out->template flat<T>().data());
  auto a_ptr = a.template flat<T>().data();
  auto b_ptr = b.template flat<T>().data();
  auto c_ptr = out->template flat<T>().data();

  EuclideanDist<T><<<32, 256>>>(a_ptr, b_ptr, n, d, k, c_ptr);
}

template struct LaunchEuclideanDistGPU<double>;
template struct LaunchEuclideanDistGPU<float>;
template struct LaunchEuclideanDistGPU<int>;

}  // namespace tensorflow

#endif
