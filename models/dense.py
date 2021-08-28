import numpy as np
from utils.utils import softmax, sigmoid, tanh, relu


class Dense:

    def __init__(self, input_size, size, activation='relu'):
        """
        Creates a dense layer for use in a Neural Network.

        :param input_size: Int - # of nodes in previous layer (or data dimensionality if first layer)
        :param size: Int - # of nodes for this layer
        :param activation: String - ["softmax", "sigmoid", "tanh", "relu"]
        """
        self.W = np.random.randn(input_size, size) / np.sqrt(input_size)
        self.b = np.zeros(size)
        self.Z_in = None
        self.Z_out = None
        self.activation = activation

    def forward(self, input):
        """
        Feed forward operation.

        Computes a linear regression for the layer input using the current weights and bias
        terms of the dense layer and feeds this into an activation function that is supplied
        during initialization.

        Used to create a layer output that can be fed into the next layer, or predict at the
        final layer.

        :param input:
        :return: Z - The output (predictions) as a Numpy Matrix
        """
        self.Z_in = input
        if self.activation == 'softmax':
            self.Z_out = softmax(input.dot(self.W) + self.b)

        elif self.activation == 'sigmoid':
            self.Z_out = sigmoid(input.dot(self.W) + self.b)

        elif self.activation == 'tanh':
            self.Z_out = tanh(input.dot(self.W) + self.b)

        else:
            self.Z_out = relu(input.dot(self.W) + self.b)

        return self.Z_out

    def gradient_descent(self, delta, lr, reg):
        """
        Performs gradient descent on the weights and bias of the dense node.

        The weights/bias are returned so they can be used to perform gradient
        descent for the previous layer that inputs to the current layer.

        :param inputs: Numpy Matrix - Previous Layer Output
        :param delta: Numpy Matrix - Delta from Next Layer
        :param lr: Float - learning rate
        :param reg: Float - Regularization term
        :return W: Numpy Matrix
        """
        self.W -= lr * self.Z_in.T.dot(delta) + reg * self.W
        self.b -= lr * delta.sum(axis=0) + reg * self.b
        return self.W

    def calculate_delta(self, delta, W):
        """
        Calculates a new delta value for the (i) dense layer based on the
        delta and weights found from the (i + 1) layer.

        :param delta: Numpy Matrix - (i + 1) layer
        :param W: Numpy Matrix - (i + 1) layer
        :return: Numpy Matrix - delta for i layer
        """
        return delta.dot(W.T) * (self.Z_out > 0)




