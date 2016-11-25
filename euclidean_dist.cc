#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>

using namespace tensorflow;

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

template <typename T>
class EuclideanDistOp : public OpKernel {
 public:
  EuclideanDistOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);

    const int r1=input_tensor1.shape().dim_size(0); //num of data
    const int c1=input_tensor1.shape().dim_size(1); //dimensions

    const int r2=input_tensor2.shape().dim_size(0); //dimensions
    const int c2=input_tensor2.shape().dim_size(1); //clusters


    auto input1 = input_tensor1.shaped<T, 2>({input_tensor1.shape().dim_size(0),input_tensor1.shape().dim_size(1)});
    auto input2 = input_tensor2.shaped<T, 2>({input_tensor2.shape().dim_size(0),input_tensor2.shape().dim_size(1)});


    //printf("%d",input1(0,0));
    //printf("%d",input2(0,0));

    // Create an output tensor
    Tensor* output_tensor = NULL;

    TensorShape out_shape= TensorShape({r1,c2});

    //out_shape.add_dim(r1);
    //out_shape.add_dim(c2);


    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));

    auto output = output_tensor->shaped<T, 2>({r1,c2});

    // Set all but the first element of the output tensor to 0.
    //const int r1 = input1.size();
    //const int c1 = input1.size();

    /* Standard Matrix Multiplication */
    /*	
    for (int i = 0; i < r1; i++) {
	for (int j=0; j< c2; j++){
		for (int k=0; k<c2; k++){			
			output(i,j) += input1(i,k)*input2(k,j);
		}
	}
    }
    */

    // Zero out the vectors initially.
    for (int k=0; k<c2; k++){
        for(int n=0; n<r1; n++){
            output(n,k) = 0;
        }
	}

    
    for(int n=0; n<r1; n++){
	for (int k=0; k<c2; k++){
		for (int d=0; d<c1; d++){
			output(n,k)+=pow((input1(n,d)-input2(d,k)),2);
		}
	}
    }

    for(int n=0; n<r1; n++){
	for(int k=0; k<c2; k++){
		output(n,k)=sqrt(output(n,k));
	}
    }

    
  }
};

REGISTER_OP("EuclideanDistGrad")
    .Input("data: T")
    .Input("clusters: T")
    .Input("distances: T")
    .Input("gradients: T")
    .Output("xgrad: T")
    .Output("cgrad: T")
    .Attr("T: {half, float, double, int32, int64, complex64, complex128}")
    .Doc(R"doc(
Computes the gradient of the pairwise Euclidean distance calculation in
the euclidean distance op with respect to the original inputs, the ouput,
and the backpropagated gradients. The inputs must be two-dimensional matrices,
the inner dimension of "data" must match the inner dimension of "clusters",
the outer dimensions of "data" and "clusters" must match the dimensions of
both "distances" and "gradients".
)doc");

template <typename T>
class EuclideanDistGradOp : public OpKernel {
 public:
  EuclideanDistGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);
    const Tensor& input_tensor3 = context->input(2);
    const Tensor& input_tensor4 = context->input(3);

    const int r1=input_tensor1.shape().dim_size(0); //num of data
    const int c1=input_tensor1.shape().dim_size(1); //dimensions

    const int r2=input_tensor2.shape().dim_size(0); //dimensions
    const int c2=input_tensor2.shape().dim_size(1); //clusters


    auto x = input_tensor1.shaped<T, 2>({input_tensor1.shape().dim_size(0),input_tensor1.shape().dim_size(1)});
    auto c = input_tensor2.shaped<T, 2>({input_tensor2.shape().dim_size(0),input_tensor2.shape().dim_size(1)});
    auto output = input_tensor3.shaped<T, 2>({input_tensor3.shape().dim_size(0),input_tensor3.shape().dim_size(1)});
    auto gradients = input_tensor4.shaped<T, 2>({input_tensor4.shape().dim_size(0),input_tensor4.shape().dim_size(1)});


    //printf("%d",input1(0,0));
    //printf("%d",input2(0,0));

    // Create an output tensor
    Tensor* output_tensor1 = NULL;
    Tensor* output_tensor2 = NULL;

    TensorShape out_shape1= TensorShape({r1,c1});
    TensorShape out_shape2= TensorShape({r2,c2});

    //out_shape.add_dim(r1);
    //out_shape.add_dim(c2);


    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape1,
                                                     &output_tensor1));
    OP_REQUIRES_OK(context, context->allocate_output(1, out_shape2,
                                                     &output_tensor2));

    auto xGradients = output_tensor1->shaped<T, 2>({r1,c1});
    auto cGradients = output_tensor2->shaped<T, 2>({r2,c2});

    // Set all but the first element of the output tensor to 0.
    //const int r1 = input1.size();
    //const int c1 = input1.size();

    /* Standard Matrix Multiplication */
    /*	
    for (int i = 0; i < r1; i++) {
	for (int j=0; j< c2; j++){
		for (int k=0; k<c2; k++){			
			output(i,j) += input1(i,k)*input2(k,j);
		}
	}
    }
    */

    // Zero out the vectors initially.
    for (int d=0; d<c1; d++){
        for(int n=0; n<r1; n++){
            xGradients(n,d) = 0;
        }
	    for (int k=0; k<c2; k++){
	        cGradients(d,k) = 0;
	    }
	}
	
	//printf("%f\n",(double)gradients(0,0));
	//printf("%f\n",(double)output(0,0));
    
    for(int n=0; n<r1; n++){
	for (int k=0; k<c2; k++){
	    // Points where x==c cannot be differentiated.
	    //  Often those are points that don't want to be
	    //  moved anyway, since that's typically the "best"
	    //  cluster representative input.
	    if (output(n,k)!=0){
		auto tempMultiplicand = gradients(n,k)/output(n,k);
		for (int d=0; d<c1; d++){
		    auto temp = (x(n,d)-c(d,k))*tempMultiplicand;
			xGradients(n,d)+=temp;
			cGradients(d,k)-=temp;
		}
	    }
	}
    }

    
  }
};

#define REGISTER_KERNEL(type)   \
  REGISTER_KERNEL_BUILDER(      \
    Name("EuclideanDist")       \
    .Device(DEVICE_CPU)         \
    .TypeConstraint<type>("T"), \
    EuclideanDistOp<type>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

#define REGISTER_KERNEL(type)   \
  REGISTER_KERNEL_BUILDER(      \
    Name("EuclideanDistGrad")   \
    .Device(DEVICE_CPU)         \
    .TypeConstraint<type>("T"), \
    EuclideanDistGradOp<type>);

REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
