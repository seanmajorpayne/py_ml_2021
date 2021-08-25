import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def sigmoid(Z):
    return 1 / 1 + np.exp(-Z)


def softmax(A):
    exp_a = np.exp(A)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def tanh(Z):
    exp_z = np.exp(Z)
    exp_neg_z = np.exp(-Z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)


def relu(Z):
    Z[Z < 0] = 0
    return Z


def forward_ann(X, W1, b1, W2, b2):
    Z = relu(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)
    return Y, Z


def forward_log_regression(X, W, b):
    return relu(X.dot(W) + b)


def gradient_w2(Z, Y, T):
    return Z.T.dot(Y - T)


def gradient_b2(Y, T):
    return (Y - T).sum(axis=0)


def gradient_w1(X, Z, Y, T, W2):
    # Sigmoid/Tanh
    # return (Y - T).dot(W2.T) * Z * (1 - Z)

    # Relu
    dZ = (Y - T).dot(W2.T) * (Z > 0)
    return X.T.dot(dZ)


def gradient_b1(T, Y, W2, Z):
    # Sigmoid/Tanh
    # return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)

    #Relu
    return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)
