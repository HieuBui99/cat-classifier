from sklearn.base import BaseEstimator, ClassifierMixin
from helpers import *


class NeuralNet(BaseEstimator, ClassifierMixin):
    def __init__(self, layers_dims, learning_rate=0.0075, num_iterations=3000):
        self.parameters = {}
        self.layers_dims = layers_dims
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        self.parameters = initialize_parameters(self.layers_dims)

        for i in range(0, self.num_iterations):
            # Forward propagation
            AL, caches = forward_propagation(X, self.parameters)
            # Compute cost.
            cost = compute_cost(AL, y)
            # Backward propagation
            grads = backward_propagation(AL, y, caches)
            # Update parameters.
            self.parameters = update_parameters(self.parameters, grads,
                                                self.learning_rate)

            if i % 100 == 0:
                print("Cost after iteration {0}: {1}".format(i, cost))

        return self

    def predict(self, X):
        m = X.shape[1]
        p = np.zeros((1, m))
        proba, _ = forward_propagation(X, self.parameters)
        for i in range(0, proba.shape[1]):
            if proba[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        return p
