#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>

#include <thread>

using namespace tensorflow;

REGISTER_OP("EuclideanDistMultiThread")
    .Input("data: double")
    .Input("clusters: double")
    .Output("distances: double");



class EuclideanDistMultiThreadOp : public OpKernel {
 public:
  explicit EuclideanDistMultiThreadOp(OpKernelConstruction* context) : OpKernel(context) {}

  static void threadcompute(OpKernelContext* context, Tensor* output_tensor, int i, int numThreads){

    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);

    const int r1=input_tensor1.shape().dim_size(0); //num of data
    const int c1=input_tensor1.shape().dim_size(1); //dimensions

    const int r2=input_tensor2.shape().dim_size(0); //dimensions
    const int c2=input_tensor2.shape().dim_size(1); //clusters


    auto in1 = input_tensor1.shaped<double, 2>({input_tensor1.shape().dim_size(0),input_tensor1.shape().dim_size(1)});
    auto in2 = input_tensor2.shaped<double, 2>({input_tensor2.shape().dim_size(0),input_tensor2.shape().dim_size(1)});
    

    auto output = output_tensor->shaped<double, 2>({r1,c2});

    printf("i: %d  r1: %d  numThreads: %d  \n",i,r1,numThreads);
    
    for(int n=i*r1/float(numThreads); n<(i+1)*r1/float(numThreads); n++){
      for (int k=0; k<c2; k++){
        output(n,k) = 0;
        for (int d=0; d<c1; d++){
          output(n,k)+=pow(in1(n,d)-in2(d,k),2);
        }
        output(n,k)=sqrt(output(n,k));
      }
    }	

  }

  
  void Compute(OpKernelContext* context) override {

    const Tensor& input_tensor1 = context->input(0);
    const Tensor& input_tensor2 = context->input(1);  

    const int c2=input_tensor2.shape().dim_size(1);    

    Tensor* output_tensor = NULL;

    TensorShape out_shape=  input_tensor1.shape();

    out_shape.set_dim(1, c2);


    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape,
                                                     &output_tensor));   


    int numThreads=2;

    std::thread myThreads[numThreads];    

    for (int id=0;id<numThreads;id++){
	
	myThreads[id]=std::thread(threadcompute, context, output_tensor, id, numThreads);
	
    }

    for (int id=0; id<numThreads; id++){

	myThreads[id].join();
    }
    

    
  }
};


REGISTER_KERNEL_BUILDER(Name("EuclideanDistMultiThread").Device(DEVICE_CPU), EuclideanDistMultiThreadOp);
