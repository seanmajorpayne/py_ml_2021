import numpy as np
from utils.utils import sigmoid
from models.model import Model


class LogisticRegression(Model):
    """
    An implementation of Logistic Regression without ML libraries.
    Best used with the Binary Cross Entropy Loss Function.
    No regularization used. For regularization, see the Lasso or Ridge
    models.
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, X):
        """
        Runs the activation function for a logistic regression unit.
        :param X: Input - Numpy Matrix
        :return: Output Prediction - Numpy Matrix
        """
        return sigmoid(X.dot(self.W) + self.b)

    def fit(self, x_train, y_train, validation_data, epochs=1000, learning_rate=0.1):
        x_test, y_test = validation_data
        _, D = x_train.shape
        self.W = np.random.randn(D) / np.sqrt(D)

        train_costs = []
        test_costs = []
        for i in range(epochs):
            # Compute and store train costs
            p_y_train = self.forward(x_train)
            train_cost = self.cost(p_y_train, y_train)
            train_costs.append(train_cost)

            # Compute and store test costs
            p_y_test = self.forward(x_test)
            test_cost = self.cost(p_y_test, y_test)
            test_costs.append(test_cost)

            # Adjust the weights/bias
            self.W -= learning_rate * x_train.T.dot(p_y_train - y_train)
            self.b -= learning_rate * (p_y_train - y_train).sum()

            if i % 500 == 0:
                print(i, train_cost, test_cost)

        self.store_costs(train_costs, test_costs)
        self.store_predictions(p_y_train, p_y_test)

