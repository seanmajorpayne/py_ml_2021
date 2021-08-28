import numpy as np
from models.model import Model
from utils.utils import relu, softmax


class ANN(Model):

    def __init__(self, layers):
        self.layers = layers
        super().__init__()

    def fit(self, x_train, y_train, validation_data, epochs=500, learning_rate=10e-7):
        x_test, y_test = validation_data
        train_costs = []
        test_costs = []
        reg = 0.001
        for epoch in range(epochs):
            # Feed Forward
            p_y_train = self.predict(x_train)

            delta = p_y_train - y_train
            W = self.layers[-1].gradient_descent(delta, learning_rate, reg)
            for i in reversed(range(1, len(self.layers) - 1)):
                delta = self.layers[i].calculate_delta(delta, W)
                W = self.layers[i].gradient_descent(delta, learning_rate, reg)
            delta = self.layers[0].calculate_delta(delta, W)
            self.layers[0].gradient_descent(delta, learning_rate, reg)

            if epoch % 10 == 0:
                p_y_train = self.predict(x_train)
                train_cost = self.cost(p_y_train, y_train)
                train_costs.append(train_cost)

                p_y_test = self.predict(x_test)
                test_cost = self.cost(p_y_test, y_test)
                test_costs.append(test_cost)
                print(epoch, train_cost, test_cost)

        self.predictions["train"] = p_y_train
        self.predictions["test"] = p_y_test
        self.history["train costs"] = train_costs
        self.history["test costs"] = test_costs

    def predict(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
