import numpy as np
import matplotlib as plt
from hilp import load_csv_data
from helper2 import *
from costs import *
from least_squares import *
from ridge_regression import *
from logistic_regression import *
from cross_validation import *
from implementations import *


# TODO Return type: Note that all functions should return: (w, loss), which is the last weight vector of the
# method, and the corresponding loss value (cost function). Note that while in previous labs you might
# have kept track of all encountered w for iterative methods, here we only want the last one. Moreover, the
# loss returned by the regularized methods (ridge regression and reg logistic regression) should
# not include the penalty term TODO TODO TODO

# TODO check tests in https://github.com/epfml/ML_course/tree/master/projects/project1/grading_tests 

# DONE change entete des fonctions + modifier output pour que elles return toutes (w, loss) DONE


# run script that execute our pipeline

# load the data
y_tr, x_tr, ids_tr = load_csv_data('data/train.csv')
y_te, x_te, ids_te = load_csv_data('data/test.csv')

# data cleaning
factor = 0.8
tx_tr, y_tr = clean_standardize(x_tr, y_tr, factor)
tx_te, y_te = clean_standardize(x_te, y_te, factor)

#range if parameters 
k_fold = 4
max_iters = 300
lambdas = [0.01, 0.001]
gammas = [0.08, 0.02, 0.1]
degrees = [1, 2]
initial_w = np.random.uniform(-1, 1, np.shape(x_tr)[1])

# best gamma for GRADIENT descent mean squared error using cross validation
best_gamma, best_rmse, acc, deg = cross_validation_gradient_descent(y_tr, tx_tr, k_fold, max_iters, gammas, degrees)
w_gradient_descent, loss = mean_squared_error_gd(y_tr, tx_tr, initial_w, max_iters, best_gamma)
y_pred = predict_y(tx_te, w_gradient_descent)
name = 'gradient_descent_submission'
create_csv_submission(ids_te, y_pred, name)


# For SGD, you must use the standard mini-batch-size 1
batch_size = 1
# best gamma for STOCHASTIC GRADIENT descent mean squared error using cross validation
best_gamma, best_rmse = cross_validation_stochastic_gradient_descent(y_tr, tx_tr, k_fold, initial_w, max_iters, gammas, batch_size)
w_stoch_gradient_descent, loss = mean_squared_error_sgd(y_tr, tx_tr, initial_w, max_iters, best_gamma)
y_pred = predict_y(tx_te, w_stoch_gradient_descent)
name = 'stoch_gradient_descent_submission'
create_csv_submission(ids_te, y_pred, name)


# LEAST SQUARES using cross validation
optimal_weights, mse_loss = least_squares(y_tr, tx_tr)
w_least_squares = optimal_weights
y_pred = predict(tx_te, w_least_squares)
name = 'least_squares_submission'
create_csv_submission(ids_te, y_pred, name)


# best lambda for RIDGE REGRESSION (least squared with lambda) using cross validation
best_lambda, best_rmse, acc, deg = cross_validation_ridge_regression(y_tr, tx_tr, k_fold, lambdas, degrees)
w_ridge_regression, loss = ridge_regression(y_tr, tx_tr, lambda_)
y_pred = predict(tx_te, w_ridge_regression)
name = 'ridge_regression_submission'
create_csv_submission(ids_te, y_pred, name)


# change logisitic
initial_w = np.random.uniform(0, 1, np.shape(x_tr)[1])
y_tr[np.where(y_tr == -1)] = 0
y_te[np.where(y_te == -1)] = 0 


# best gamma for LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
best_gamma, best_rmse, acc, deg = cross_validation_logistic_regression(y_tr, tx_tr, k_fold, max_iters, gammas, degrees)
w_log_regression, loss = logistic_regression(y_tr, tx_tr, initial_w, max_iters, best_gamma)
y_pred = predict_logistic(tx_te, w_log_regression)
name = 'logistic_regression_submission'
create_csv_submission(ids_te, y_pred, name)


# best gamma AND lambda for REGULARIZED LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
best_gamma, best_lambda, best_rmse, tmp, deg = cross_validation_reg_logistic_regression(y_tr, tx_tr, k_fold, max_iters, lambdas, gammas, degrees)
w_reg_log_regression, loss = reg_logistic_regression(y_tr, tx_tr, best_lambda, initial_w, max_iters, best_gamma)
y_pred = predict_logistic(tx_te, w_reg_log_regression)
name = 'reg_logistic_regression_submission'
create_csv_submission(ids_te, y_pred, name)


# convert the results into the file that we can put on the website challenge






