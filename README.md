# TensorFlowRBF
This repository explores the design of a Radial Basis Function and related functions (like K-Means) for use with TensorFlow.

# To compile
run the following in browser

    TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    nvcc -std=c++11 -c -o euclidean_dist_gpu.cu.o euclidean_dist_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
    nvcc -std=c++11 -c -o euclidean_dist_grad_gpu.cu.o euclidean_dist_grad_gpu.cu.cc -I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
    g++ -std=c++11 -shared euclidean_dist.cc euclidean_dist_grad.cc euclidean_dist_gpu.cu.o euclidean_dist_grad_gpu.cu.o -o euclidean_dist.so -fPIC -I $TF_INC
