import pandas as pd
import numpy as np

"""
sigmoid(z)
params:
    z - array of 1D
returns array of 1D applying sigmoid function element wise
on z.
sigmoid(x) = 1 / (1 + exp(-x))
where x is a variable.
Remember value of sigmoid(x) is always between 0 and 1
"""
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


"""
predict_abs(theta, X):
params:
    theta - array of 1D containing coefficients of data
    X - data
Applies sigmoid function to hypothesis
"""
def predict_abs(theta, X):
    return sigmoid(np.dot(X, theta))


"""
predict(theta, X)
params:
    theta - array of 1D containing coefficients of data
    X - data
Classifies to True or False, i.e. if probability >= 0.5 then True
else False
"""
def predict(theta, X):
    preds = predict_abs(theta, X)
    return preds >= 0.5


"""
compute_error_for_separator_given_data(theta, X, y)
params:
    theta - array of 1D containing coefficients of test data
    X - test data
    y - actual labels
Computes Least Mean Square Error between Predictions and actual values.
"""
def compute_error_for_separator_given_data(theta, X, y):
    preds = predict_abs(theta, X)
    total_error = np.sum((y - preds) ** 2)
    return total_error / float(len(y))


"""
feature_scaling(X):
params:
    X - data
Scales data using formula:
new_X = (old_X - min(old_X)) / (max(old_X) - min(old_X))
"""
def feature_scaling(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))
    return X


"""
step_gradient(theta_current, X, y, learning_rate)
params:
    theta_current - current value of theta for hypothesis
    X - data
    y - actual labels
    learning_rate - Hyperparameter to tune speed for calculating gradients.
Computes new values of theta for a given iteration of gradient_descent_runner
"""
def step_gradient(theta_current, X, y, learning_rate):
    preds = predict_abs(theta_current, X)
    theta_gradient = -(2 / len(y)) * np.dot(X.T, (y - preds))
    theta = theta_current - learning_rate * theta_gradient
    return theta


"""
gradient_descent_runner(X, y, initial_theta, learning_ratem num_iters)
params:
    X - data
    y - actual labels
    initial_theta - initial_theta value of coefficients of hypothesis
    learning_rate - Hyperparameter to tune speed for calculating gradients.
    num_iters - number of times you want to run gradient descent
Finds the best separator to the data.
"""
def gradient_descent_runner(X, y, initial_theta, learning_rate, num_iters):
    theta = initial_theta
    for i in range(num_iters):
        theta = step_gradient(theta, X, y, learning_rate)
    return theta


"""
accuracy(theta, X, y)
params:
    theta - value of coefficients of hypothesis
    X - data
    y - actual labels
Calculates accuracy of our hypothesis
"""
def accuracy(theta, X, y):
    preds = predict(theta, X)
    preds.astype(int)
    total = len(y)
    eq = np.equal(preds, y)
    vals, count = np.unique(eq, return_counts = True)
    d = dict(zip(vals, count))
    return d[True] / total


"""
This function sets up our data and goes through each step of the gradient descent.
"""
def run():
    #Collect data
    dataframe = pd.read_csv('binary.csv', sep = ',')
    data = dataframe.as_matrix()
    y = data[:, 0]
    X = data[:, 1:]
    
    #Scale Data
    X = feature_scaling(X)
    
    #Step 2 - define hyperparameters
    learning_rate = 3
    num_iters = 1000
    initial_theta = np.random.random(3)
    
    #train our model
    print('Starting gradient descent at theta = {0}, error = {1}, accuracy = {2}'
    .format(initial_theta, compute_error_for_separator_given_data(initial_theta, X, y), accuracy(initial_theta, X, y)))
    
    theta = gradient_descent_runner(X, y, initial_theta, learning_rate, num_iters)
    
    print('Ending gradient descent at Iteration = {0} theta = {1}, error = {2}, accuracy = {3}'
    .format(num_iters, theta, compute_error_for_separator_given_data(theta, X, y), accuracy(theta, X, y)))
    



if __name__ == '__main__':
    run()