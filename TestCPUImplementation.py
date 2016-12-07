import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

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

import time
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def RBFEuclidean(x, C, number_of_threads=1):
    """Computes distance from cluster centers defined in input C
    
    Both outdim and indim should be integers.
    """
    return -euclidean_dist(x,C,number_of_threads)

repeat_i_times = 10

sess = tf.InteractiveSession()

try:
    results = np.load("feedforwardtimings.npy")
    results[:5,:] = np.zeros([5,3])
except:
    results = np.zeros([6,3])

for i in range(repeat_i_times):
    for n in range(0,5):
        for k in range(1,4):
            x = tf.placeholder(tf.float32, shape=[None, 10**k])
            y_ = tf.placeholder(tf.float32, shape=[None, 10**k])

            W = weight_variable([10**k, 10**k])

            x_in = np.random.normal(0,1,[10**k,10**k])
            W_in = np.random.normal(0,1,[10**k,10**k])

            y_conv = RBFEuclidean(x, W, 2**n)
            if True:
                start = time.time()
                y_conv.eval({x:x_in, W:W_in})
                elapsed_time = time.time()-start
                results[n,k-1] += elapsed_time/repeat_i_times
                print("\nFeed-forward took {} with {} threads and {} datapoints".format(elapsed_time, 2**n, 10**k))

np.save("feedforwardtimings.npy",results)

try:
    results = np.load("backproptimings.npy")
    results[:5,:] = np.zeros([5,3])
except:
    results = np.zeros([6,3])

for i in range(repeat_i_times):
    for n in range(0,5):
        for k in range(1,4):
            x = tf.placeholder(tf.float32, shape=[None, 10**k])
            y_ = tf.placeholder(tf.float32, shape=[None, 10**k])

            W = weight_variable([10**k, 10**k])

            x_in = np.random.normal(0,1,[10**k,10**k])
            W_in = np.random.normal(0,1,[10**k,10**k])

            y_conv = RBFEuclidean(x, W, 2**n)
        
            grad = tf.gradients(y_conv,[x,W])
            if True:
                start = time.time()
                tf.get_default_session().run(grad, feed_dict={x:x_in, W:W_in})
                elapsed_time = time.time()-start
                results[n,k-1] += elapsed_time/repeat_i_times
                print("\nBackprop took {} with {} threads and {} datapoints".format(elapsed_time, 2**n, 10**k))

np.save("backproptimings.npy",results)
