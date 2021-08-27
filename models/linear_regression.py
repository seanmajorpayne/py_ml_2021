import numpy as np
from models.model import Model


class LinearRegression(Model):

    def __init__(self):
        super().__init__()

    def fit(self, x_train, y_train, validation_data, epochs=100, learning_rate=0.00000001):
        x_test, y_test = validation_data
        N, D = x_train.shape
        self.W = np.random.randn(D) / np.sqrt(D)

        train_costs = []
        test_costs = []
        l2 = 10
        for i in range(epochs):
            p_y_train = x_train.dot(self.W)
            train_cost = self.cost(p_y_train, y_train)
            train_costs.append(train_cost)

            p_y_test = x_test.dot(self.W)
            test_cost = self.cost(p_y_test, y_test)
            test_costs.append(test_cost)

            if i % 10 == 0:
                print(i, train_cost, test_cost)

            self.W -= learning_rate * (x_train.T.dot(p_y_train - y_train))
