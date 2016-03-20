# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:56:48 2016

@author: Tyler Klement

Based on assignment given by Andrew Ng on OpenClassroom, in Machine Learning
http://openclassroom.stanford.edu/MainFolder/HomePage.php

Requires Numpy, Matplotlib, and the ex2Data folder downloadable on the website,
or available on my GitHub.
"""
import numpy as np
from matplotlib import pyplot as plt

def h_theta(theta, x):
    '''
    Non vectorized function for returning the Y value given a vector of Thetas 
    and an X feature vector
    '''
    y = 0
    for i in range(len(x)):
        y += theta[i] * x[i]
        
    return y
    
    
def h_theta_vec(theta, X):
    '''
    Vectorized function for returning the Y vector given a vector of Thetas and
    X's
    '''
    Y = []
    for i in range(len(X)):
        Y.append(0)
        for j in range(len(X[i])):
            Y[i] += theta[j] * X[i][j]
        
    return Y
    
    
def gradient_descent(theta, x, y, alpha):
    '''
    param theta: an array of thetas
    param x: an array of feature vectors
    param y: an array of outputs
    return new_theta: the adjusted thetas
    
    Performs gradient descent and returns the adjusted thetas
    '''
    m = len(x)
    num_features = len(x[0])
    new_theta = []
    for i in range(num_features):
        m_sum = 0
        for j in range(m):
            m_sum += (h_theta(theta, x[j]) - y[j]) * x[j][i]
                
        new_theta.append(theta[i] - alpha * (1 / m) * m_sum)
        
    return new_theta


def main():
    # variables
    theta = [] # feature coefficients
    alpha = 0.07 # learning rate
    
    # get x and y data
    x = np.loadtxt('ex2Data/ex2x.dat')[np.newaxis].T
    y = np.loadtxt('ex2Data/ex2y.dat')[np.newaxis].T
    
    # plot initial data
    f, ax = plt.subplots()
    plt.plot(x, y, 'o', label='Training data')
    plt.xlabel('Age in years')
    plt.ylabel('Height in meters')
    
    # adding ones to x feature vectors for bias/y-intercept
    m = len(x)
    bias = np.ones((m, 1))
    x = np.hstack((bias, x))
    
    num_features = len(x[0])
    
    # initialize thetas as 0
    for i in range(num_features):
        theta.append(0)
    
    # the amount we step with each descent. 1000 is a dummy value, it gets adjusted
    # step is only used as a stopping function.
    step = 1000
    iteration = 1 # used for reporting later
    
    # BATCH GRADIENT DESCENT
    # if steps are less than the below value, assume we have reached a local
    # minimum
    while step > 0.0000001:
        # update thetas
        new_theta = gradient_descent(theta, x, y, alpha)
        updates = np.subtract(np.asarray(theta), np.asarray(new_theta))
        theta = new_theta
        
        # see much we updated/stepped
        step = 0
        for elem in updates:
            step += elem[0]
        step = abs(step)
        
        iteration += 1
        print('Iteration: ' + str(iteration) + '\n' + \
                'Theta: ' + str(theta) + '\n' + \
                'Step: ' + str(step) + '\n' + \
                '-------------------------------------')
    
    # get regression line values
    newY = h_theta_vec(theta, x)
        
    # plot line
    plt.plot(x, newY, 'r-', label='Linear regression')
    
    # plot legend (necessary to remove duplicates from legend for matplotlib)
    handles, labels = ax.get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list, loc='upper left', numpoints=1)
    
    plt.show()
    
    # Now predict the height of two boys, age 3.5 and age 7:
    ages = [[1, 3.5], [1, 7]] # include 1s for intercepts
    heights = h_theta_vec(theta, ages)
    
    print('The heights of two boys with ages 3.5 and 7 are: ' + \
            str(round(heights[0][0], 2)) + ' and ' + str(round(heights[1][0], 2)))

if __name__ == "__main__":
    main()