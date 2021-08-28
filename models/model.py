import numpy as np
from utils.loss import Loss
from utils.utils import classification_rate

class Model:
    """
    Base class for Machine Learning Models
    """
    def __init__(self):
        self.W = None
        self.b = 0
        self.history = {}
        self.predictions = {}
        pass

    def compile(self, optimizer=None, loss=Loss.MSE):
        self.loss = loss
        pass

    def fit(self, x_train, y_train, validation_data, epochs=1000):
        pass

    def cost(self, P, Y):
        """
        Takes predictions and targets and returns the error based
        on the chosen loss function provided at the compile stage.
        :param P: Predictions - Numpy Matrix
        :param Y: Targets - Numpy Matrix
        :return: Cost - Float
        """

        # Mean Squared Error Standardized
        if self.loss == Loss.MSE:
            delta = P - Y
            return delta.dot(delta)

        if self.loss == Loss.RSQUARED:
            ssr = Y - P
            sst = Y - np.mean(Y)
            r2 = 1 - (ssr.dot(ssr) / sst.dot(sst))
            return r2

        # Binary Crossentropy
        elif self.loss == Loss.BCE:
            return -np.mean(Y * np.log(P) + (1 - Y) * np.log(1 - P))

        # Categorical Crossentropy
        elif self.loss == Loss.CC:
            total = Y * np.log(P)
            return -total.mean()

    def store_costs(self, train_costs, test_costs):
        self.history["train_costs"] = train_costs
        self.history["test_costs"] = test_costs

    def store_predictions(self, train_predictions, test_predictions):
        self.predictions["Training Predictions"] = train_predictions
        self.predictions["Test Predictions"] = test_predictions

    def score(self, y_train, y_test):
        p_train = np.round(self.predictions["Training Predictions"])
        p_test = np.round(self.predictions["Test Predictions"])
        train_classification = classification_rate(p_train, y_train)
        test_classification = classification_rate(p_test, y_test)
        print("Final train classification rate: {}".format(train_classification))
        print("Final test classification rate: {}".format(test_classification))


