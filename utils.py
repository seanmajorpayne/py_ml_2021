import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def sigmoid(Z):
    """
    Takes a Matrix and runs it through the sigmoid activation
    :param Z: Numpy Matrix
    :return: Numpy Matrix
    """
    return 1 / 1 + np.exp(-Z)


def softmax(A):
    """
    Takes a Matrix and runs it through the softmax activation
    :param A: Numpy Matrix
    :return: Numpy Matrix
    """
    exp_a = np.exp(A)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


def tanh(Z):
    """
    Takes a Matrix and runs it through the tanh activation
    :param Z: Numpy Matrix
    :return: Numpy Matrix
    """
    exp_z = np.exp(Z)
    exp_neg_z = np.exp(-Z)
    return (exp_z - exp_neg_z) / (exp_z + exp_neg_z)


def relu(Z):
    """
    Takes a Matrix and runs it through the relu activation
    :param Z: Numpy Matrix
    :return: Numpy Matrix
    """
    Z[Z < 0] = 0
    return Z


def forward_ann(X, W1, b1, W2, b2):
    """
    Runs the feed forward activation functions for a 3-layer ANN.
    (Input, Hidden, Output)
    :param X: Input Numpy Matrix
    :param W1: Layer 1 Weights Numpy Matrix
    :param b1: Layer 1 Bias Numpy Matrix
    :param W2: Layer 2 Weights Numpy Matrix
    :param b2: Layer 2 Bias Numpy Matrix
    :return: Output Matrix and Hidden Layer Matrix
    """
    Z = relu(X.dot(W1) + b1)
    Y = softmax(Z.dot(W2) + b2)
    return Y, Z


def forward_log_regression(X, W, b):
    """
    Runs the activation function for a logistic regression unit.
    :param X: Input Numpy Matrix
    :param W: Weights Numpy Matrix
    :param b: Bias Numpy Array
    :return: Output Prediction Numpy Matrix
    """
    return sigmoid(X.dot(W) + b)


def gradient_w2(Z, Y, T):
    """
    Computes the gradient of the weights for the hidden layer to output layer.
    :param Z: Hidden Layer Activation Numpy Matrix
    :param Y: Prediction Numpy Matrix
    :param T: Targets Numpy Matrix
    :return: Gradient Matrix gW2
    """
    return Z.T.dot(Y - T)


def gradient_b2(Y, T):
    """
    Computes the gradient of the bias for the output layer.
    :param Y: Prediction Numpy Matrix
    :param T: Target Numpy Matrix
    :return: Gradient Matrix gb2
    """
    return (Y - T).sum(axis=0)


def gradient_w1(X, Z, Y, T, W2):
    """
    Computes the gradient of the weights for the input layer to hidden layer.
    :param X: Inputs Numpy Matrix
    :param Z: Input Layer Activation Numpy Matrix
    :param Y: Prediction Numpy Matrix
    :param T: Targets Numpy Matrix
    :param W2: Hidden to Output Weight Numpy Matrix
    :return: Gradient Matrix gW1
    """
    # Sigmoid/Tanh
    # return (Y - T).dot(W2.T) * Z * (1 - Z)

    # Relu
    dZ = (Y - T).dot(W2.T) * (Z > 0)
    return X.T.dot(dZ)


def gradient_b1(Z, Y, T, W2):
    """
    Computes the gradient of the bias for the hidden layer.
    :param Z: Input Layer Activation Numpy Matrix
    :param Y: Prediction Numpy Matrix
    :param T: Targets Numpy Matrix
    :param W2: Hidden to Output Weight Numpy Matrix
    :return:
    """
    # Sigmoid/Tanh
    # return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)

    #Relu
    return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)


def cross_entropy_cost(Y, T):
    total = T * np.log(Y)
    return total.sum()

