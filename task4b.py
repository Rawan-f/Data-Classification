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

# Load train and test data
def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            if values[-1] in ['class-2', 'class-3']:
                data.append([float(val) for val in values[:-1]])
                labels.append(1 if values[-1] == 'class-2' else 0)
    return np.array(data), np.array(labels)

# Train and evaluate binary Perceptron for one pair of classes
def train_and_evaluate_pair(classifier_name, train_data, train_labels, test_data, test_labels):
    num_features = train_data.shape[1]
    perceptron = Perceptron(num_features)
    num_iterations = 20
    learning_rate = 0.1

    # Training
    perceptron.train(train_data, train_labels, num_iterations, learning_rate)

    # Training accuracy
    train_predictions = [perceptron.predict(x) for x in train_data]
    train_accuracy = np.mean(train_predictions == train_labels)

    # Test accuracy
    test_predictions = [perceptron.predict(x) for x in test_data]
    test_accuracy = np.mean(test_predictions == test_labels)

    print(f"{classifier_name} - Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}")

# Load train and test data for each pair of classes
train_data, train_labels = load_data("train.data")
test_data, test_labels = load_data("test.data")

# Task 4 - Train and evaluate classifiers for three pairs of classes (b)
train_and_evaluate_pair("Class 2 vs. Class 3", train_data, train_labels, test_data, test_labels)