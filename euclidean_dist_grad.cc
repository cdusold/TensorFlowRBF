#include "euclidean_dist_grad.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
// Not available for user made kernels
//#include "tensorflow/core/kernels/fill_functor.h"
#include <cmath>
#include <thread>

namespace tensorflow {

template <typename T>
static void threadcompute(OpKernelContext* context, Tensor* xoutput_tensor, Tensor* coutput_tensor, int i, int numThreads){

  const Tensor& input_tensor1 = context->input(0);
  const Tensor& input_tensor2 = context->input(1);
  const Tensor& input_tensor3 = context->input(2);
  const Tensor& input_tensor4 = context->input(3);

  const int r1=input_tensor1.shape().dim_size(0); //num of data = n
  const int c1=input_tensor1.shape().dim_size(1); //dimensions = d
  const int c2=input_tensor2.shape().dim_size(1); //clusters = k


  auto in1 = input_tensor1.shaped<double, 2>({r1,c1});
  auto in2 = input_tensor2.shaped<double, 2>({c1,c2});
  auto in3 = input_tensor3.shaped<double, 2>({r1,c2});
  auto in4 = input_tensor4.shaped<double, 2>({r1,c2});
  
  auto xgrad = xoutput_tensor->shaped<double, 2>({r1,c1});
  auto cgrad = coutput_tensor->shaped<double, 2>({c1,c2});

  //printf("i: %d  r1: %d  numThreads: %d  \n",i,r1,numThreads);
  
  for(int in=i*r1/float(numThreads); in<(i+1)*r1/float(numThreads); in++){
    for (int id=0; id<c1; id++){
      xgrad(in,id) = 0;
    }
  }
  
  for(int ik=i*c2/float(numThreads); ik<(i+1)*c2/float(numThreads); ik++){
    for (int id=0; id<c1; id++){
      cgrad(id,ik) = 0;
    }
  }
  
  //printf("%f\n",(double)gradients(0,0));
  //printf("%f\n",(double)output(0,0));
  
  if (numThreads == 1) {
    for(int in=0; in<r1; in++){
      for (int ik=0; ik<c2; ik++){
      
        // Points where x==c cannot be differentiated.
        //  Often those are points that don't want to be
        //  moved anyway, since that's typically the "best"
        //  cluster representative input.
        if (in3(in,ik)!=0){
          auto tempMultiplicand = in4(in,ik)/in3(in,ik);
          for (int id=0; id<c1; id++){
            auto temp = (in1(in,id)-in2(id,ik))*tempMultiplicand;
            xgrad(in,id) += temp;
            cgrad(id,ik) -= temp;
          }
        }
      }
    }
  } else { // In multithreading, the two gradients have to be computed apart.
    for(int in=i*r1/float(numThreads); in<(i+1)*r1/float(numThreads); in++){
      for (int ik=0; ik<c2; ik++){
        if (in3(in,ik)!=0){
          auto tempMultiplicand = in4(in,ik)/in3(in,ik);
          for (int id=0; id<c1; id++){
            xgrad(in,id) += (in1(in,id)-in2(id,ik))*tempMultiplicand;
          }
        }
      }
    }
    for(int ik=i*c2/float(numThreads); ik<(i+1)*c2/float(numThreads); ik++){
      for (int in=0; in<r1; in++){
        if (in3(in,ik)!=0){
          auto tempMultiplicand = in4(in,ik)/in3(in,ik);
          for (int id=0; id<c1; id++){
            cgrad(id,ik) -= (in1(in,id)-in2(id,ik))*tempMultiplicand;
          }
        }
      }
    }
  }

}

template <typename Device, typename T>
class EuclideanDistGradOp;

template <typename T>
struct LaunchEuclideanDistGradCPU {
  static void launch(
      OpKernelContext* ctx, OpKernel* kernel, const Tensor& a, const Tensor& b,
      const Tensor& d, const Tensor& g, Tensor* xout, Tensor* cout) {
    
    int numThreads = (( EuclideanDistGradOp<CPUDevice, T>*) kernel)->get_number_of_threads();
    
    std::thread myThreads[numThreads-1];

    for (int id=0;id<numThreads-1;id++){
      myThreads[id]=std::thread(threadcompute<T>, ctx, xout, cout, id, numThreads);
    }
    
    // Remember that the thread launching these threads is a usuable thread as well.
    threadcompute<T>(ctx, xout, cout, numThreads-1, numThreads);
    
    for (int id=1; id<numThreads-1; id++){
      myThreads[id].join();
    }
  }
};

template <typename T>
struct LaunchEuclideanDistGrad<CPUDevice, T> : public LaunchEuclideanDistGradCPU<T> {};

template <typename T>
struct LaunchEuclideanDistGrad<GPUDevice, T> : public LaunchEuclideanDistGradGPU<T> {};

REGISTER_OP("EuclideanDistGrad")
    .Input("data: T")
    .Input("clusters: T")
    .Input("distances: T")
    .Input("gradients: T")
    .Output("xgrad: T")
    .Output("cgrad: T")
    .Attr("number_of_threads: int >= 1 = 1")
    .Attr("T: {half, float, double, int32, int64, complex64, complex128}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

      ::tensorflow::shape_inference::ShapeHandle a;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &a));

      ::tensorflow::shape_inference::ShapeHandle b;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &b));

      ::tensorflow::shape_inference::ShapeHandle d;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &d));

      ::tensorflow::shape_inference::ShapeHandle g;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 2, &g));

      ::tensorflow::shape_inference::DimensionHandle xoutput_rows = c->Dim(a, 0);
      ::tensorflow::shape_inference::DimensionHandle xoutput_cols = c->Dim(a, 1);

      ::tensorflow::shape_inference::DimensionHandle coutput_rows = c->Dim(b, 0);
      ::tensorflow::shape_inference::DimensionHandle coutput_cols = c->Dim(b, 1);

      // Validate that the inner shapes are compatible.
      ::tensorflow::shape_inference::DimensionHandle inner_a = c->Dim(a, 1);
      ::tensorflow::shape_inference::DimensionHandle inner_b = c->Dim(b, 0);
      ::tensorflow::shape_inference::DimensionHandle merged;
      TF_RETURN_IF_ERROR(c->Merge(inner_a, inner_b, &merged));
      // I actually don't know how to check the shapes of d and g here.
      // I don't know if that's necessary.
      // Documentation on shape_inference is sparse.

      c->set_output(0, c->Matrix(xoutput_rows, xoutput_cols));
      c->set_output(1, c->Matrix(coutput_rows, coutput_cols));
      return Status::OK();
    })
    .Doc(R"doc(
Computes the gradient of the pairwise Euclidean distance calculation in
the euclidean distance op with respect to the original inputs, the ouput,
and the backpropagated gradients. The inputs must be two-dimensional matrices,
the inner dimension of "data" must match the inner dimension of "clusters",
the outer dimensions of "data" and "clusters" must match the dimensions of
both "distances" and "gradients".
)doc");

template <typename Device, typename T>
class EuclideanDistGradOp : public OpKernel {
 public:
  EuclideanDistGradOp(OpKernelConstruction* context) : OpKernel(context) {
    // Get the number of the threads to use
    OP_REQUIRES_OK(context,
                   context->GetAttr("number_of_threads", &number_of_threads_));
    // Check that number_of_threads is strictly positive
    OP_REQUIRES(context, number_of_threads_ >= 1,
                errors::InvalidArgument("Need number_of_threads >= 1, got ",
                                        number_of_threads_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    const Tensor& d = ctx->input(2);
    const Tensor& g = ctx->input(3);

    // Check that the dimensions of the four matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(d.shape()),
                errors::InvalidArgument("In[2] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(g.shape()),
                errors::InvalidArgument("In[3] is not a matrix"));

    OP_REQUIRES(ctx,
                a.dim_size(1) == b.dim_size(0),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[1]: ",
                                        b.shape().DebugString()));
    OP_REQUIRES(ctx,
                a.dim_size(0) == d.dim_size(0),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[2]: ",
                                        d.shape().DebugString()));
    OP_REQUIRES(ctx,
                b.dim_size(1) == d.dim_size(1),
                errors::InvalidArgument("Matrix size-incompatible: In[1]: ",
                                        b.shape().DebugString(), ", In[2]: ",
                                        d.shape().DebugString()));
    OP_REQUIRES(ctx,
                a.dim_size(0) == g.dim_size(0),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(), ", In[3]: ",
                                        g.shape().DebugString()));
    OP_REQUIRES(ctx,
                b.dim_size(1) == g.dim_size(1),
                errors::InvalidArgument("Matrix size-incompatible: In[1]: ",
                                        b.shape().DebugString(), ", In[3]: ",
                                        g.shape().DebugString()));
    TensorShape xout_shape({a.dim_size(0), a.dim_size(1)});
    TensorShape cout_shape({b.dim_size(0), b.dim_size(1)});
    Tensor* xout = nullptr;
    Tensor* cout = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, xout_shape, &xout));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, cout_shape, &cout));

    if (a.NumElements() == 0) {
      // If a has zero elements, the first output shape is
      // a 0-element matrix, so there is nothing to do for it.
      if (b.NumElements() != 0) {
        // But if b has non-zero elements, we have to
        // set all of the second output to zero.
        auto output = cout->shaped<T, 2>({b.shape().dim_size(0),b.shape().dim_size(1)});
    
        // Not available for user made kernels
        //functor::SetZeroFunctor<Device, T> f;
        //f(ctx->eigen_device<Device>(), out->flat<T>());
        for (int k=0; k<b.dim_size(1); k++){
          for(int n=0; n<b.dim_size(0); n++){
            output(n,k) = 0;
          }
        }
      }
      return;
    }
    if (b.NumElements() == 0) {
      // If b has zero elements, and a doesn't then we
      // have to set all of the first output to zero.
      auto output = xout->shaped<T, 2>({a.shape().dim_size(0),a.shape().dim_size(1)});
    
      // Not available for user made kernels
      //functor::SetZeroFunctor<Device, T> f;
      //f(ctx->eigen_device<Device>(), out->flat<T>());
      for (int k=0; k<a.dim_size(1); k++){
        for(int n=0; n<a.dim_size(0); n++){
          output(n,k) = 0;
        }
      }
      return;
    }

    LaunchEuclideanDistGrad<Device, T>::launch(ctx, this, a, b, d, g, xout, cout);
    
  }
  
  int get_number_of_threads() {
    return number_of_threads_;
  }
 private:
  int number_of_threads_;
};

#define REGISTER_KERNEL(type)              \
  REGISTER_KERNEL_BUILDER(                 \
    Name("EuclideanDistGrad")              \
    .Device(DEVICE_CPU)                    \
    .TypeConstraint<type>("T"),            \
    EuclideanDistGradOp<CPUDevice, type>); \
  REGISTER_KERNEL_BUILDER(                 \
    Name("EuclideanDistGrad")              \
    .Device(DEVICE_GPU)                    \
    .TypeConstraint<type>("T"),            \
    EuclideanDistGradOp<GPUDevice, type>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

}  // namespace tensorflow
