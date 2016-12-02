#include "euclidean_dist.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
// Not available for user made kernels
//#include "tensorflow/core/kernels/fill_functor.h"
#include <cmath>

namespace tensorflow {

template <typename T>
struct LaunchEuclideanDistCPU {    
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      Tensor* out) {
      
    const int r1 = a.dim_size(0);
    const int c1 = a.dim_size(1);
    const int c2 = b.dim_size(1);
    
    auto in1 = a.shaped<T, 2>({r1,c1});
    auto in2 = b.shaped<T, 2>({c1,c2});
    auto output = out->shaped<T, 2>({r1,c2});
    
    for(int n=0; n<r1; n++){
      for (int k=0; k<c2; k++){
        output(n,k) = 0;
        for (int d=0; d<c1; d++){
          output(n,k)+=pow(in1(n,d)-in2(d,k),2);
        }
        output(n,k)=sqrt(output(n,k));
      }
    }
  }
};

template <typename T>
struct LaunchEuclideanDist<CPUDevice, T> : public LaunchEuclideanDistCPU<T> {};

template <typename T>
struct LaunchEuclideanDist<GPUDevice, T> : public LaunchEuclideanDistGPU<T> {};

REGISTER_OP("EuclideanDist")
    .Input("data: T")
    .Input("clusters: T")
    .Output("distances: T")
    .Attr("T: {half, float, double, int32, int64, complex64, complex128}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      ::tensorflow::shape_inference::ShapeHandle a;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

      ::tensorflow::shape_inference::ShapeHandle b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

      ::tensorflow::shape_inference::DimensionHandle output_rows = c->Dim(a, 0);
      ::tensorflow::shape_inference::DimensionHandle output_cols = c->Dim(b, 1);

      // Validate that the inner shapes are compatible.
      ::tensorflow::shape_inference::DimensionHandle inner_a = c->Dim(a, 1);
      ::tensorflow::shape_inference::DimensionHandle inner_b = c->Dim(b, 0);
      ::tensorflow::shape_inference::DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));

      c->set_output(0, c->Matrix(output_rows, output_cols));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the euclidean distance from every vector in "data" to every
vector in "clusters". The inputs must be two-dimensional matrices and
the inner dimension of "data" must match the inner dimension of "clusters".
)doc");

template <typename Device, typename T>
class EuclideanDistOp : public OpKernel {
 public:
  EuclideanDistOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));

    OP_REQUIRES(ctx,
                a.dim_size(1) == b.dim_size(0),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));
    TensorShape out_shape({a.dim_size(0), b.dim_size(1)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }
    if (a.NumElements() == 0 || b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we return
      // the output with zeros.
    
      auto output = out->shaped<T, 2>({a.shape().dim_size(0),b.shape().dim_size(1)});
    
      // Not available for user made kernels
      //functor::SetZeroFunctor<Device, T> f;
      //f(ctx->eigen_device<Device>(), out->flat<T>());
      for (int k=0; k<b.dim_size(1); k++){
        for(int n=0; n<a.dim_size(0); n++){
          output(n,k) = 0;
        }
      }
      return;
    }

    LaunchEuclideanDist<Device, T>::launch(ctx, this, a, b, out);
    
  }
};

#define REGISTER_KERNEL(type)          \
  REGISTER_KERNEL_BUILDER(             \
    Name("EuclideanDist")              \
    .Device(DEVICE_CPU)                \
    .TypeConstraint<type>("T"),        \
    EuclideanDistOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(             \
    Name("EuclideanDist")              \
    .Device(DEVICE_GPU)                \
    .TypeConstraint<type>("T"),        \
    EuclideanDistOp<GPUDevice, type>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

}  // namespace tensorflow
