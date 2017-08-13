import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def sigmoid(x):
    # Sigmoid of a input x is given by :
    # sig(x) = 1/1+e^-x
    return 1/(1+np.exp(-x))
def sigmoid_prime(x):
    # Derivate of sigmoid has pretty good formula
    # Denote derivative of sigmoid function by sigmoid'
    # Formula goes as : sigmoid(x)' = sigmoid(x) * (1-sigmoid(x))
    return sigmoid(x)*(1-sigmoid(x))
def prediction(X, W, b):
    # prediction i.e. y_hat = sigmoid(Wx+b)
    return sigmoid(np.matmul(X,W)+b)
def error_vector(y, y_hat):
    # Error or error vector E from lecture of cross entropy is given by:
    # E = - summation((yi*log(pi)) + (1-yi)*log(1-pi)) summation over i=1 to m
    # or this can be written as - from class of gradient descent
    # E = - Summation( (yi * log(y_hat)) + ((1-yi)*log(1-y_hat))) summation over i=1 to m
    #       because Pi = y_hat i.e. prediction
    return [-y[i]*np.log(y_hat[i]) - (1-y[i])*np.log(1-y_hat[i]) for i in range(len(y))]
def error(y, y_hat):
    # Error is just the average over the summation vector as described in class of LR
    ev = error_vector(y, y_hat)
    return sum(ev)/len(ev)

# Code below is to calculate the gradient of the error function.
# The result should be a list of three lists:
# The first list should contain the gradient (partial derivatives) with respect to w1
# The second list should contain the gradient (partial derivatives) with respect to w2
# The third list should contain the gradient (partial derivatives) with respect to b
def dErrors(X, y, y_hat):
    # The gradient of error function is given by dE/dwi = (y-y_hat)(xi) derivative over weights
    # and dE/db = (y-y_hat) derivative over bias
    DErrorsDx1 = [(y[i]-y_hat[i])*X[i][0] for i in range(len(y))]
    DErrorsDx2 = [(y[i]-y_hat[i])*X[i][1] for i in range(len(y))]
    DErrorsDb = [(y[i]-y_hat[i]) for i in range(len(y))]
    return DErrorsDx1, DErrorsDx2, DErrorsDb

# the code below to implement the gradient descent step.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b.
# It should calculate the prediction, the gradients, and use them to
# update the weights and bias W, b. Then return W and b.
# The error e will be calculated and returned for you, for plotting purposes.
def gradientDescentStep(X, y, W, b, learn_rate = 0.01):
    # as per the gradient descent pseudo code:
    '''1. Start with random weights i.e. w1, w2, w3,......, wn
    2. for every point in (x1, x2, x3,......, xn)
    for i in 1,2, 3,.....n:
    update wi' <-- wi - alpha*(y-y_hat)(xi)
    and update bias b' <-- b - alpha*(y-y_hat)
    '''
    # First Calculate the prediction i.e. y_hat
    y_hat = prediction(X, W, b)
    # Second Calculate the gradient
    gradient_result = dErrors(X, y , y_hat)
    W[0] +=  sum(gradient_result[0])*learn_rate
    W[1] +=  sum(gradient_result[1])*learn_rate
    b += sum(gradient_result[2])*learn_rate
    # calculate the error
    e = error(y, y_hat)
    return W, b, e

# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainLR(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    # Initialize the weights randomly
    W = np.array(np.random.rand(2,1))*2 -1
    b = np.random.rand(1)[0]*2 - 1
    # These are the solution lines that get plotted below.
    boundary_lines = []
    errors = []
    for i in range(num_epochs):
        # In each epoch, we apply the gradient descent step.
        W, b, error = gradientDescentStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
        errors.append(error)
    return boundary_lines, errors
