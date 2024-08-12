import numpy as np

# Binary Perceptron class
class Perceptron:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros(num_features)
        self.bias = 0

    def predict(self, x):
        z = np.dot(self.weights, x) + self.bias
        return 1 if z >= 0 else 0

    def train(self, x_train, y_train, num_iterations, learning_rate):
        for _ in range(num_iterations):
            for x, y in zip(x_train, y_train):
                prediction = self.predict(x)
                if prediction != y:
                    self.weights += learning_rate * (y - prediction) * x
                    self.bias += learning_rate * y