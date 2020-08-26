import numpy as np
import random
import sys

import numpy as np
import random
import sys


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def sigmoid_derivative(values):
    return values * (1 - values)

def tanh_normal(values):
    return np.tanh(values)

def tanh_derivative(values):
    return 1. - np.multiply(values,values)

def softmax(X):
    exp_X = np.exp(X)
    exp_X_sum = np.sum(exp_X,axis=1).reshape(-1,1)
    exp_X = exp_X/exp_X_sum
    return exp_X

def relu(X):
   return np.maximum(0,X)

def relu_derivative(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

#    X is the output from fully connected layer (num_examples x num_classes)
#    y is labels (num_examples x 1)
#   y is not one-hot encoded vector
def delta_cross_entropy(X,y):
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad




# createst uniform random array w/ values in [a,b) and shape args
def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a


# Creates a list of tuples wit mini batches for X and Y
def get_mini_batches(X, y, batch_size):
    random_idxs = np.random.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs, :, :]
    y_shuffled = y[random_idxs,:]
    mini_batches = [(X_shuffled[i:i + batch_size, :,:], y_shuffled[i:i + batch_size,:]) for i in range(0, len(y), batch_size)]
    return mini_batches


def get_index_batch(batch,range_low, range_high):
    seedValue = (random.randrange(sys.maxsize))//2**32
    np.random.seed(seedValue)
    return np.random.randint(low=range_low, high=range_high, size=(batch,))

