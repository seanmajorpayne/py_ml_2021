import numpy as np
from models.model import Model
from utils.utils import relu, softmax


class NeuralNetwork(Model):

    def __init__(self):
        super().__init__()

    def forward(self, X, W1, b1, W2, b2):
        hidden = relu(X.dot(W1) + b1)
        output = softmax(hidden.dot(W2) + b2)
        return hidden, output

    def fit(self, x_train, y_train, validation_data, epochs=10000, learning_rate=10e-6):


    def grad_descent(self):
        reg = 0.01
        for epoch in range(epochs):
            hidden, output = forward(X, W1, b1, W2, b2)