Write a function to generate an m+1 dimensional data set, of size n, consisting of m continuous independent
variables (X) and one dependent variable (Y) defined as
yi = xiβ + e
where,
• e is a Gaussuan distribution with mean 0 and standard deviation (σ), representing the unexplained
variation in Y
• β is a random vector of dimensionality m + 1, representing the coefficients of the linear relationship
between X and Y, and
• ∀i ∈ [1, n], xi0 = 1
The function should take the following parameters:
• σ: The spread of noise in the output variable
• n: The size of the data set
• m: The number of indepedent variables
Output from the function should be:
• X: An n × m numpy array of independent variable values (with a 1 in the first column)
• Y : The n × 1 numpy array of output values
• β: The random coefficients used to generatre Y from X
2 Write a function that learns the parameters of a linear regression line given inputs
• X: An n × m numpy array of independent variable values
• Y : The n × 1 numpy array of output values
• k: the number of iteractions (epochs)
• τ : the threshold on change in Cost function value from the previous to current iteration
The function should implement the Gradient Descent algorithm as discussed in class that initialises β with
random values and then updates these values in each iteraction by moving in the the direction defined by
the partial derivative of the cost function with respect to each of the coefficients. The function should use
only one loop that ends after a number of iterations (k) or a threshold on the change in cost function value
(τ ).
The output should be an m + 1 dimensional vector of coefficients and the final cost function value.

**************************************************************************************************************
import random
import numpy as np
from matplotlib import pyplot as plt

def generate_data(sigma,n,m):
    e = random.gauss(0, sigma)
    x = np.random.rand(n , m+1)
    for i in range(0, len(x)):
        x[i][0] = 1
    beta = np.random.rand(m+1)
    x_beta = x @ beta
    y = x_beta + e

    return (x,y,beta)

def gaussian_descent(X,Y,k,tou,alpha):
    m = 6
    beta = np.random.rand(m+1)
    E = Y - X @ beta
    curr_cost = np.dot(E,E)
    def_cost = -(2*E @ X)

    for _ in range(0,1000):
        plt.plot(beta,curr_cost)
        E = Y -X @ beta
        cost = np.dot(E,E)
        def_cost = -(2*E @ X)
        if abs(curr_cost - cost).all() == tou:
            break
        else:
            beta = beta - alpha*def_cost
        curr_cost = cost
        plt.plot(np.reshape(beta,(1,-len(beta))),curr_cost)
    plt.show()
    return (beta,curr_cost)

X, Y, beta = generate_data(0.5,100,6)
print("Original data : ",beta)
gaussian_beta, cost = gaussian_descent(X,Y,100,0.01,0.001)
print("Predicted Beta : ",gaussian_beta)
print("cost",cost)
