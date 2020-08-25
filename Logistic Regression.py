Write a function to generate an m+1 dimensional data set, of size n, consisting of m continuous independent
variables (X) and one dependent binary variable (Y) defined as
Y =
(
1 if p(y = 1|~x) = 1
1+exp−~x.β~ > 0.5
0 otherwise
where,
• β is a random vector of dimensionality m + 1, representing the coefficients of the linear relationship
between X and Y, and
• ∀i ∈ [1, n], xi0 = 1
To add noise to the labels (Y) generated, we assume a Bernoulli distribution with probability of success, θ,
that determines whether or not the label generated, as above, is to be flipped. The larger the value of θ, the
greater is the noise.
The function should take the following parameters:
• θ: The probability of flipping the label, Y
• n: The size of the data set
• m: The number of indepedent variables
Output from the function should be:
• X: An n × m numpy array of independent variable values (with a 1 in the first column)
• Y : The n × 1 binary numpy array of output values
• β: The random coefficients used to generate Y from X
2 Write a function that learns the parameters of a logistic regression function given inputs
• X: An n × m numpy array of independent variable values
• Y : The n × 1 binary numpy array of output values
• k: the number of iteractions (epochs)
• τ : the threshold on change in Cost function value from the previous to current iteration
• λ: the learning rate for Gradient Descent
The function should implement the Gradient Descent algorithm as discussed in class that initialises β with
random values and then updates these values in each iteraction by moving in the the direction defined by
the partial derivative of the cost function with respect to each of the coefficients. The function should use
only one loop that ends after a number of iterations (k) or a threshold on the change in cost function value
(τ ).
The output should be a m + 1 dimensional vector of coefficients and the final cost function value.

*************************************************************************************************************

import math
import numpy as np
from matplotlib import pyplot as plt
import random

def generate_data(theta,n,m):
	list = []
	#e = np.random.binomial(0, theta)
	x = np.random.randn(n, m+1)
	for i in range(0, len(x)):
		x[i][0] = 1
	beta = np.random.randn(m+1)
	x_beta = x @ beta
	y = 1/(1 + np.exp(-x_beta))
	#for j in y:
	y = np.where(y>0.5,1,0)
	list.append(y)
	X = np.array(x)
	Y = np.array([y])
	return (X,Y,beta)

def gradient_descent(k,tou,lamda):
	list1 = []    #to add iterations
	list2 = []    #to add cost
	n = 10
	m = 2
	#k = 5
	#lamda = 2
	#tou = 0.5
	x = np.random.randn(n,m+1)
	for i in range(0, len(x)):
		x[i][0] = 1
	beta = np.random.randn(m+1)
	x_beta = x @ beta
	y = 1/(1 + np.exp(-x_beta))
	Y = np.array([y])
	print("Random Beta chosen : ",beta)
	cost = -(np.sum(Y*np.log(y) + (1-Y) *np.log(1-y)))/m
	print("Initial cost : ", cost)
	prev_cost = float('inf')
	for j in range(0,k):
		list1.append(j)
		X = np.array(x)
		X_beta = X @ beta
		y = 1/(1+np.exp(-X_beta))
		a1 = np.where(y>0.5,1,0)
		cost = -(np.sum(Y*np.log(y) + (1-Y) *np.log(1-y))) / m
		list2.append(cost)
		Y = np.reshape(10,1)
		def_cost = 1/n * np.dot(X.T, y-Y)
		beta = beta - lamda*def_cost
		if abs(prev_cost - cost) <tou:
			print("Total Iterations ", j)

			break
		prev_cost = cost
	print("Predicted beta : ",beta)
	print("Predicted y : ", a1)
	print("Final cost : ", cost)


print(generate_data(0.5,100,2))
print(gradient_descent(2,0.5,0.01))
