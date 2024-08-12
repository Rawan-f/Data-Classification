# Data Classification - Implementing Perceptron Algorithm

## Introduction
This Python codes implements the Perceptron algorithm by implementing a binary Perceptron for iris dataset, and multi-class classification using the 1-vs-rest approach.

## Prerequisites
Before running the code, ensure you have the following prerequisites installed:
- Python (>= 3.6)
- NumPy library (you can install it using `pip install numpy`)

## Getting Started
Follow these steps to set up and run the binary Perceptron classifier:

1.Clone or Download the Repository: If you haven't already, clone or download the code repository to your local machine.

2. Open the Python script `task3.py`,`task4a.py`,`task4b.py`,`task4c.py`,`task5.py`,`task6a.py`,`task6b.py`,`task6c.py`,`task7a.py`,`task7b.py`,`task7c.py` in your preferred code editor or IDE.

3. Review the code to understand its structure and functionality.

4. Prepare your dataset in CSV format with the following characteristics:
   - Each row represents a data point.
   - The last column contains binary class labels (e.g., 'class-1' and 'class-2').

5. Update the file paths in the `load_data` function to load your own training and test data files.

6. Save your changes.

7. Run the script:


## Usage

The scripts will train and evaluate the Perceptron classifier for the 'Class 1, Class 2, Class 3' 

## Results

The script will display the following results:

- Training Accuracy: The accuracy of the model on the training dataset.
- Test Accuracy: The accuracy of the model on the test dataset.
- The most discriminative feature used by the classifier based on the learned weights.
- The most difficult pair to separate of classes.
- Train and test accuracy using the 1-vs-rest approach with regularization

