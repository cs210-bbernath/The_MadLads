import numpy as np
import matplotlib as plt

from helper import *
from cross_validation import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient
        gradient = compute_gradient(y, tx, w)
        # compute loss
        loss = compute_mse(y, tx, w)
        # update w by gradient
        w = w - (gamma * gradient)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    w = ws[-1]
    return w, loss




def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=1):
            # compute stoch_gradient
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            # compute loss
            loss = compute_mse(y, tx, w)
            # update w by gradient
            w = w - gamma * stoch_gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
    loss = losses[-1]
    w = ws[-1]
    return w, loss



def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    """
    # returns mse, and optimal weights
    Xt_X_inv = np.linalg.inv(tx.T@tx)
    w = Xt_X_inv@(tx.T@(y))
    loss = compute_mse(y, tx, w)
    return w, loss



def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    """
    N, D = tx.shape
    w = np.linalg.solve(np.dot(tx.T, tx) + 2 * lambda_ * N * np.eye(D), np.dot(tx.T, y))
    loss = compute_mse(y, tx, w)
    return w, loss



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """The logistic regression algorithm.  
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of logistic regression
        gamma: a scalar denoting the stepsize, aka learning rate
    """
    ws = [initial_w]
    losses = []
    w = initial_w
    N = len(y)
    for n_iter in range(max_iters):
        # compute gradient        
        # update w by gradient
        gradient = 2/N * tx.T.dot(sigmoid(tx@w) - y)
        w = w - (gamma * gradient)
        y_pred = sigmoid(tx.dot(w))
        # compute loss
        loss = log_loss(y, tx, w, lambda_=0)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    w = ws[-1]
    return w, loss




def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """The reg logistic regression algorithm.
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        lambda_: regularization factor
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of reg logistic regression
        gamma: a scalar denoting the stepsize, aka learning rate
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
        # compute loss
        loss = log_loss(y,tx, w, lambda_=0)
        # store w and loss
        ws.append(w)
        losses.append(loss)
    loss = losses[-1]
    w = ws[-1]
    return w, loss
