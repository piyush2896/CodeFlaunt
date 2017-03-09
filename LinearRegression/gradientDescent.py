import numpy as np
from matplotlib import pyplot as plt

#y = mx + b (slope formula of line)
"""
compute_error_for_line_given_points(b, m, points)-
params:
    b - y intercept of the line
    m - slope of the line
    points - list of points [x, y]
Computes Least Mean Square Error between predictions and actual values.
"""
def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for ip in range(len(points)):
        x = points[ip, 0]
        y = points[ip, 1]
        total_error += (y - (m * x + b )) ** 2
    return total_error / float(len(points))

"""
step_gradient(points, b_current, m_current, learning_rate)
params:
    points - list of points [x, y]
    b_current - current value of y intercept
    m_current - current value of slope of the line
    learning_rate - Hyperparameter to tune speed for calculating gradients. 
Computes new values of b and m for a given iteration of gradient_descent_runner
"""
def step_gradient(points, b_current, m_current, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - learning_rate * b_gradient
    new_m = m_current - learning_rate * m_gradient
    return [new_b, new_m]
    
"""
gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
params:
    points - list of points [x, y]
    initial_b - initial value of the y intercept
    initial_m - initial value of slope of the line
    learning_rate - Hyperparameter to tune speed for calculating gradients.
    num_iterations - number of times you want to run gradient descent
Finds the best fit line to the data.
"""
def gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations):
    b = initial_b
    m = initial_m
    for i in range(num_iterations):
        b, m = step_gradient(points, b, m, learning_rate)
    return [b, m]

"""
This function sets up our data and goes through each step of the gradient descent.
"""
def run():
    #Step 1 - Collect the data
    points = np.genfromtxt('data.csv', delimiter = ',')
    
    #Step 2 - define hyperparameters
    learning_rate = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 1000
    
    #Step 3 - train our model
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'
    .format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    
    print('Ending gradient descent at Iteration = {0} b = {1}, m = {2}, error = {3}'
    .format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    
    #Step 4 - Visualization
    x = [ix[0] for ix in points]
    y = [iy[1] for iy in points]
    y_predict = [m * ix + b for ix in x]
    
    plt.figure(0)
    plt.scatter(x, y)
    plt.plot(x, y_predict)
    plt.show()
    


if __name__ == '__main__':
    run()