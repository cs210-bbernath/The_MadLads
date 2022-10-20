import numpy as np
import matplotlib as plt
from helper import *

from costs import compute_mse
from ridge_regression import ridge_regression
from build_polynomial import build_poly
from plots import cross_validation_visualization



def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)



def cross_validation(y, tx, func, k_indices = 0, k = 0, lambda_ = 0, degree = 0, initial_w = 0, max_iters = 0, gamma = 0, batch_size = 0):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        tx:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    # get k'th subgroup in test, others in train: TODO

    test_x = []
    test_y = []
    train_x = []
    train_y = []
    
    indices = k_indices[k]
    
    print("1")
    ind = np.arange(y.shape[0])
    train_ind = ~np.isin(ind, indices)
    test_x = tx[indices]
    test_y = y[indices]
    train_x = tx[train_ind]
    train_y = y[train_ind]

    print("2")

    # form data with polynomial degree:
    #poly_tr = build_poly(train_x, degree)
    #poly_te = build_poly(test_x, degree)
    model_tr = np.c_[train_x]
    print("3")
    model_te= np.c_[test_x]
    
    print("3,5")

    # ridge regression:
    if(func == 'ridge_regression'):
        w = ridge_regression(train_y, model_tr, lambda_)
    elif func == 'reg_logistic_regression':
        l, ws = reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
        w = ws[max_iters-1]
    elif func == 'logistic_regression':
        l, ws = logistic_regression(y, tx, initial_w, max_iters, gamma)
        w = ws[max_iters-1]
    elif func == 'logistic_regression':
        w, l = least_squares(y, tx)
    elif func == 'stochastic_gradient_descent':
        l, ws = stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma)
        w = ws[max_iters-1]
    elif func == 'gradient_descent':
        l, ws = gradient_descent(y, tx, initial_w, max_iters, gamma)
        w = ws[max_iters-1]

    print("4")

    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2*compute_mse(train_y, model_tr, w))
    print("4,5")
    loss_te = np.sqrt(2*compute_mse(test_y, model_te, w))
    
    return loss_tr, loss_te


from plots import cross_validation_visualization

def cross_validation_demo(y, tx, func, k_fold, lambdas=[], k_indices = 0, degree = 0, initial_w = 0, max_iters = 0, gammas = [], batch_size = 0):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    # cross validation over lambdas:
    if lambdas != [] and gammas != []:
        for ind, lambda_ in enumerate(lambdas):
            for ind, gamma in enumerate(gammas):
                rmse_tr_tmp = []
                rmse_te_tmp = []
                for k in range(k_fold):
                    loss_tr, loss_te = cross_validation(y=y, tx=tx, func=func, k_indices=k_indices, k=k, lambda_=lambda_, degree=degree, initial_w=initial_w, max_iters=max_iters, gamma=gamma, batch_size=batch_size)
                    rmse_tr_tmp.append(loss_tr)
                    rmse_te_tmp.append(loss_te)
                rmse_tr.append(np.mean(rmse_tr_tmp))
                rmse_te.append(np.mean(rmse_te_tmp))
    if lambdas == [] and gammas != []:
        for ind, gamma in enumerate(gammas):
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y=y, tx=tx, func=func, k_indices=k_indices, k=k, degree=degree, initial_w=initial_w, max_iters=max_iters, gamma=gamma, batch_size=batch_size)
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
    
    if lambdas != [] and gammas == []:
        for ind, lambda_ in enumerate(lambdas):
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y=y, tx=tx, func=func, k_indices=k_indices, k=k, lambda_=lambda_, degree=degree, initial_w=inital_w, max_iters=max_iters, batch_size=batch_size)
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
    if lambdas == [] and gammas == []:
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y=y, tx=tx, func=func, k_indices=k_indices, k=k, degree=degree, initial_w=initial_w, max_iters=max_iters, batch_size=batch_size)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))
    if lambdas != [] and gammas == []:
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    if lambdas ==[] and gammas !=[]:
        cross_validation_visualization(gammas, rmse_tr, rmse_te)
        
    best_rmse = min(rmse_te)
    best_lambda = lambdas[rmse_te.index(best_rmse)]
    
    print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse

def cross_validation_gradient_descent(y, tx, func, k_fold, initial_w, max_iters, gammas):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 12
    k_fold = k_fold
    gammas = gammas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    # cross validation over lambdas:
    for ind, gamma in enumerate(gammas):
        rmse_tr_tmp = []
        rmse_te_tmp = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y, tx, func = func, k_indices= k_indices, k=k, initial_w=initial_w, max_iters= max_iters, gamma= gamma)
            rmse_tr_tmp.append(loss_tr)
            rmse_te_tmp.append(loss_te)
        rmse_tr.append(np.mean(rmse_tr_tmp))
        rmse_te.append(np.mean(rmse_te_tmp))

    cross_validation_visualization(gammas, rmse_tr, rmse_te)
    
    best_rmse = min(rmse_te)
    best_gamma = gammas[rmse_te.index(best_rmse)]
    
    print("For polynomial expansion up to degree, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (best_gamma, best_rmse))
    return best_gamma, best_rmse


def best_degree_selection(degrees, k_fold, lambdas, seed, y, tx):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        best_rmse : value of the rmse for the couple (best_degree, best_lambda)
        
    >>> best_degree_selection(np.arange(2,11), 4, np.logspace(-4, 0, 30))
    (7, 0.004520353656360241, 0.28957280566456634)
    """
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation over degrees and lambdas:
    best_degrees = []
    best_lambdas = []
    best_rmses = []
    for deg in degrees:  
        rmse_tr = []
        rmse_te = []
        for ind, lambda_ in enumerate(lambdas):
            rmse_tr_tmp = []
            rmse_te_tmp = []
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y, tx, k_indices, k, lambda_, deg)
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)
            rmse_tr.append(np.mean(rmse_tr_tmp))
            rmse_te.append(np.mean(rmse_te_tmp))
        
        
        best_rmse = min(rmse_te)
        best_lambda = lambdas[rmse_te.index(best_rmse)]
        
        best_lambdas.append(best_lambda)
        best_rmses.append(best_rmse)
        
    best_rmse = min(best_rmses)
    best_lambda = lambdas[best_rmses.index(best_rmse)]
    best_degree = degrees[best_rmses.index(best_rmse)]
    
    return best_degree, best_lambda, best_rmse
