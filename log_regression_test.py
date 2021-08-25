from models.logistic_regression import LogisticRegression
from utils.loss import Loss
import matplotlib.pyplot as plt
from utils.utils import get_data, get_binary_data, normalize_data

"""
A demonstration of the LogisticRegression Model from the models directory.
Uses the MNIST dataset with the 0 and 1 class label data to perform binary
predictions.

Even with this simple model, the costs converge near 0 for train and
test sets and the classification rate is nearly 100% for both.
"""

def main():
    # Load in data for classes 0 & 1 and normalize
    Xtrain, Xtest, Ytrain, Ytest = get_data()
    Xtrain, Ytrain = get_binary_data(Xtrain, Ytrain)
    Xtest, Ytest = get_binary_data(Xtest, Ytest)
    Xtrain, Xtest = normalize_data(Xtrain, Xtest)

    model = LogisticRegression()
    model.compile(loss=Loss.BCE)
    model.fit(Xtrain, Ytrain, validation_data=(Xtest, Ytest), epochs=5000, learning_rate=0.00001)

    # Plot the costs over training epochs
    plt.plot(model.history["train_costs"], label="train_costs")
    plt.plot(model.history["test_costs"], label="test_costs")
    plt.legend()
    plt.show()

    # Log the classification rate
    model.score(Ytrain, Ytest)


if __name__ == '__main__':
    main()
