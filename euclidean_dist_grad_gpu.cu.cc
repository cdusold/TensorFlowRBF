//#include <cmath>

#if GOOGLE_CUDA

#include "euclidean_dist_grad.h"

//#include "cuda/include/cuda.h"
#include "cuda.h"
//#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

template <typename T>
__global__ void EuclideanDistGrad(const T* in1, const T* in2, const T* in3, const T* in4, const int n, const int d, const int k, T* out1, T* out2) {
  for (int in = blockIdx.x; in < n; in += gridDim.x) {
    for (int id = threadIdx.x; id < d; id += blockDim.x) {
      out1[in*d+id] = 0;
    }
  }
  for (int id = blockIdx.x; id < d; id += gridDim.x) {
    for (int ik = threadIdx.x; ik < k; ik += blockDim.x) {
      out2[id*k+ik] = 0;
    }
  }
  for (int in = blockIdx.x; in < n; in += gridDim.x) {
    for (int ik = 0; ik < k; ik ++) {
      T tempMultiplicand = in4[in*k+ik]/in3[in*k+ik];
      for (int id = threadIdx.x; id < d; id += blockDim.x) {
        T temp = (in1[in*d+id] - in2[id*k+ik])*tempMultiplicand;
        out1[in*d+id]+=temp;
      }
    }
  }
  for (int in = 0; in < n; in ++) {
    for (int ik = threadIdx.x; ik < k; ik += blockDim.x) {
      T tempMultiplicand = in4[in*k+ik]/in3[in*k+ik];
      for (int id = blockIdx.x; id < d; id += gridDim.x) {
        T temp = (in1[in*d+id] - in2[id*k+ik])*tempMultiplicand;
        out2[id*k+ik]-=temp;
      }
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
void LaunchEuclideanDistGradGPU<T>::launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      const Tensor& d, const Tensor& g, Tensor* xout, Tensor* cout) {
  const uint64 r1 = a.dim_size(0);
  const uint64 c1 = a.dim_size(1);
  const uint64 c2 = b.dim_size(1);

  auto* stream = ctx->op_device_context()->stream();
  OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

  //Not available to user kernels.
  //auto a_ptr = AsDeviceMemory(a.template flat<T>().data());
  //auto b_ptr = AsDeviceMemory(b.template flat<T>().data());
  //auto d_ptr = AsDeviceMemory(d.template flat<T>().data());
  //auto g_ptr = AsDeviceMemory(g.template flat<T>().data());
  //auto x_ptr = AsDeviceMemory(xout->template flat<T>().data());
  //auto c_ptr = AsDeviceMemory(out->template flat<T>().data());
  auto a_ptr = a.template flat<T>().data();
  auto b_ptr = b.template flat<T>().data();
  auto d_ptr = d.template flat<T>().data();
  auto g_ptr = g.template flat<T>().data();
  auto x_ptr = xout->template flat<T>().data();
  auto c_ptr = cout->template flat<T>().data();

  EuclideanDistGrad<T><<<32, 256>>>(a_ptr, b_ptr, d_ptr, g_ptr, r1, c1, c2, x_ptr, c_ptr);
}

template struct LaunchEuclideanDistGradGPU<double>;
template struct LaunchEuclideanDistGradGPU<float>;
template struct LaunchEuclideanDistGradGPU<int>;

}  // namespace tensorflow

#endif
