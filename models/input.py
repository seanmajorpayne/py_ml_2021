import numpy as np
from utils.utils import softmax, sigmoid, tanh, relu

class Input:

    def __init__(self, input_size, size, activation='relu'):
        """
        Creates a dense layer for use in a Neural Network.

        :param input_size: Int - # of nodes in previous layer (or data dimensionality if first layer)
        :param size: Int - # of nodes for this layer
        :param activation: String - ["softmax", "sigmoid", "tanh", "relu"]
        """
        self.W = np.random.randn(input_size, size) / np.sqrt(input_size)
        self.b = np.zeros(size)
        self.Z = None
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
        if self.activation == 'softmax':
            self.Z = softmax(input.dot(self.W) + self.b)

        elif self.activation == 'sigmoid':
            self.Z = sigmoid(input.dot(self.W) + self.b)

        elif self.activation == 'tanh':
            self.Z = tanh(input.dot(self.W) + self.b)

        else:
            self.Z = relu(input.dot(self.W) + self.b)

        return input, self.Z

    def gradient_descent(self, inputs, delta, lr, reg):
        """
        Performs gradient descent on the weights and bias of the dense node.

        The weights/bias are returned so they can be used to perform gradient
        descent for the previous layer that inputs to the current layer.

        :param inputs: Numpy Matrix - Previous Layer Output
        :param delta: Numpy Matrix - Delta from Next Layer
        :param lr: Float - learning rate
        :param reg: Float - Regularization term
        :return Z, W, b: Numpy Matrix, Numpy Matrix, Numpy Array
        """
        self.W -= lr * inputs.T.dot(delta) + reg * self.W
        self.b -= lr * delta.sum(axis=0) + reg * self.b
        return self.Z, self.W, self.b