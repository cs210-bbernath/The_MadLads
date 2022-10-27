import numpy as np
import matplotlib as plt

def sigmoid(t):
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The logistic regression algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of logistic regression
        gamma: a scalar denoting the stepsize, aka learning rate
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of logistic_regression
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), 
            for each iteration of Logistic regression 
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    N = len(y)
    
    for n_iter in range(max_iters):
        # compute gradient
        gradient = 2/N * tx.T.dot(sigmoid(tx.dot(w)) - y)
        
        # update w by gradient
        gradient = logistic_gradient(y, tx, w, lambda_=0)
        w = w - (gamma * gradient)
        
        y_pred = sigmoid(tx.dot(w))
        # compute loss
        loss = -np.sum(np.dot(y.T,np.log(y_pred)+ np.dot((1-y).T, np.log(1-y_pred)))) /(len(y_pred))
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Logistic regression iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
         #     bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def logistic_gradient(y, tx, w, lambda_=0):
    return (1 / len(y)) * (tx.T @ (sigmoid(tx @ w) - y)) + 2 * lambda_ * w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The reg logistic regression algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: regularization factor
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of reg logistic regression
        gamma: a scalar denoting the stepsize, aka learning rate
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of reg logistic_regression
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), 
            for each iteration of reg Logistic regression 
    """
    ws = []
    losses = []
    w = initial_w
    N = len(y)
    
    for n_iter in range(max_iters):
        # compute gradient
        gradient = 2/N * tx.T.dot(sigmoid(tx@w) - y) + (lambda_)*w
        
        # update w by gradient
        w = w - (gamma * gradient)
        
        y_pred = sigmoid(tx.dot(w))
        
        # compute loss
        loss = -np.sum(np.dot(y.T,np.log(y_pred)+ np.dot((1-y).T, np.log(1-y_pred)))) /(len(y_pred)) + (lambda_/2)*(w.T.dot(w))
        # store w and loss
        ws.append(w)
        losses.append(loss)
        #print("Reg Logistic regression iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws