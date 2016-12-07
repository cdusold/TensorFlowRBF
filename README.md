# TensorFlowRBF
This repository explores the design of a Radial Basis Function and related functions (like K-Means) for use with TensorFlow.

# To compile
Note: You must have a GPU compatible TensorFlow installed on your system before compiling this project.

Run the following in a terminal from the folder you checked out into: (use python2 instead of python3 for if you are using Python 2 for some reason.)

    TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    nvcc -std=c++11 -c -o euclidean_dist_gpu.cu.o euclidean_dist_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
    nvcc -std=c++11 -c -o euclidean_dist_grad_gpu.cu.o euclidean_dist_grad_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
    g++ -std=c++11 -shared euclidean_dist.cc euclidean_dist_grad.cc euclidean_dist_gpu.cu.o euclidean_dist_grad_gpu.cu.o -o euclidean_dist.so -fPIC -I $TF_INC

# To load the layer in python
From python running in the same folder as you checked out run the following code to load the layer from the compiled library:

    import tensorflow as tf
    euclidean_dist_module = tf.load_op_library("euclidean_dist.so")
    euclidean_dist = euclidean_dist_module.euclidean_dist
    euclidean_dist_grad = euclidean_dist_module.euclidean_dist_grad
    
    from tensorflow.python.framework import ops
    @ops.RegisterGradient("EuclideanDist")
    def _EuclideanDistGrad(op, grad):
        a = op.inputs[0]
        b = op.inputs[1]
        y = op.outputs[0]  # y = 0.5 * b / conj(a)
        xGrad, cGrad = euclidean_dist_grad(a,b,y,grad)
        return xGrad, cGrad

# To run the example code which generated our plots for the paper
Open a terminal, and run `python3 TestCPUImplementation.py`
Then run, `python3 TestGPUImplementation.py`
And finally run `python3 PlotTestData.py`
