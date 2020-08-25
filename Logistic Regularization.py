Add L1 and L2 regularization to the Logistic Regression cost function. How does this impact the models
learnt? How does the choice of regularization constant impact the β vector learned?

***************************************************************************************

mport numpy as np
import random as rd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def randomVar(n, m, theta):
    x1= np.random.randn(n, m)
    unit = np.ones([n,1], dtype = int)
    x= np.append(unit, x1, axis=1)

    # Bias
    b= np.random.randn(m+1)
    b1= np.dot(x,b)
    prob= sigmoid(b1)
    print("Original Beta", b)
    # Output
    y= np.where(prob>=theta,  1,  0)
    return x, y, b
def accuracy(b, x, y):
    b1= np.dot(x,b)
    prob= sigmoid(b1)
    pre_y= np.where(prob>=0.5,  1,  0)
    n= y.shape[0]
    error= np.where(y!=pre_y, 0, 1)
    return np.count_nonzero(error==1)*100/n

def Cost(x, y, lasso, epochs=100, th=0.001, lr=0.01):
    p=float('inf')
    n=x.shape[0]
    m=x.shape[1]
    b=np.random.randn(m)
    tuningPar=0.01
    penalty1=penalty = 0
    for i in range(epochs):
        if lasso==True:
            penalty=tuningPar*np.sum(b[1:n]*b[1:n])
            penalty1=2*tuningPar*np.sum(b[1:n])
        elif lasso==False:
            penalty=np.sum(abs(b[1:n]))
            penalty1=tuningPar

        b1=np.dot(x,b)
        predictedProb= sigmoid(b1)
        cost= -(np.sum(y*np.log(predictedProb)+(1-y)*np.log(1-predictedProb)))/n+penalty
        if abs(p-cost)<=th:
            return cost, b
        p = cost
        cfunc = -1/n * np.dot(x.T, y-predictedProb)+penalty1
        b-=lr*cfunc
    return cost, b
def logRegression(x, y):
    return Cost(x, y,-1)
def lassoReg(x, y):
    return Cost(x, y, True)
def ridgeReg(x, y):
    return Cost(x, y, False)

data=randomVar(100, 3, 0.5)

n=data[1].shape[0]
trainLen=int(0.4*data[0].shape[0])

logisticCoef = logRegression(x=data[0][0:trainLen], y=data[1][0:trainLen])
print("Logistic Regression coefficients=", logisticCoef)
lassoCoef = lassoReg(x=data[0][0:trainLen], y=data[1][0:trainLen])
print("Lasso Regression coefficients=",lassoCoef)
ridgeCoef = ridgeReg(x=data[0][0:trainLen], y=data[1][0:trainLen])
print("Ridge Regression coefficients=" ,ridgeCoef)

logAcc =  accuracy(logisticCoef[1],data[0][trainLen:n],data[1][trainLen:n])
print("Logistic Regression's Accuracy=", logAcc)
lassoAcc = accuracy(lassoCoef[1],data[0][trainLen:n],data[1][trainLen:n])
print("Lasso Regression's accuracy=", lassoAcc)
ridgeAcc = accuracy(ridgeCoef[1],data[0][trainLen:n],data[1][trainLen:n])
print("Ridge Regression's accuracy=", ridgeAcc )

