import numpy as np
import matplotlib as plt
from hilp import load_csv_data
from helper2 import *
from costs import *
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

print("Load the data...")
y_tr, x_tr, ids_tr = load_csv_data('data/train.csv')
y_te, x_te, ids_te = load_csv_data('data/test.csv')

print("Data cleaning...")
factor = 0.6
tx_tr, y_tr = clean_standardize(x_tr, y_tr, factor)
tx_te, y_te = clean_standardize(x_te, y_te, factor)
print("------------------------------------------")

#range if parameters 
k_fold = 4
max_iters = 300
lambdas = [0.01, 0.001]
gammas = [0.08, 0.02, 0.1]
degrees = [1, 2]

#print("Gradient descent cross validation")
# best gamma for GRADIENT descent mean squared error using cross validation
#best_gamma, best_rmse, acc, deg = cross_validation_gradient_descent(y_tr, tx_tr, k_fold, max_iters, gammas, degrees)
print("Gradient descent computing...")
tx_tr = build_poly(tx_tr, 1)
tx_te = build_poly(tx_te, 1)
initial_w = np.random.uniform(-1,1,tx.shape[1])
w, loss = mean_squared_error_gd(y_tr, tx_tr, initial_w, 300, 0.08)
w_gradient_descent, loss = mean_squared_error_gd(y_tr, tx_tr, initial_w, 300, 0.08)
y_pred_te = predict_y(tx_te, w_gradient_descent)
y_pred_tr = predict_y(tx_tr, w_gradient_descent)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'gradient_descent_submission'
create_csv_submission(ids_te, y_pred_te, name)
print("------------------------------------------")



# For SGD, you must use the standard mini-batch-size 1
batch_size = 1
# best gamma for STOCHASTIC GRADIENT descent mean squared error using cross validation
#print("Stochastic Gradient descent cross validation")
#best_gamma, best_rmse = cross_validation_stochastic_gradient_descent(y_tr, tx_tr, k_fold, initial_w, max_iters, gammas, batch_size)
print("Stochastic Gradient descent computing...")

w_stoch_gradient_descent, loss = mean_squared_error_sgd(y_tr, tx_tr, initial_w, 10, 0.009)
y_pred_te = predict_y(tx_te, w_stoch_gradient_descent)
y_pred_tr = predict_y(tx_tr, w_stoch_gradient_descent)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'stoch_gradient_descent_submission'
create_csv_submission(ids_te, y_pred_te, name)
print("------------------------------------------")

# LEAST SQUARES using cross validation
print("Least Squares computing...")
w_least_squares, loss = least_squares(y_tr, tx_tr)
y_pred_te = predict_y(tx_te, w_least_squares)
y_pred_tr = predict_y(tx_tr, w_least_squares)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'least_squares_submission'
create_csv_submission(ids_te, y_pred_te, name)
print("------------------------------------------")


# best lambda for RIDGE REGRESSION (least squared with lambda) using cross validation
print("Ridge Regression computing...")
#best_lambda, best_rmse, acc, deg = cross_validation_ridge_regression(y_tr, tx_tr, k_fold, lambdas, degrees)
w_ridge_regression, loss = ridge_regression(y_tr, tx_tr, 0.01)
y_pred_te = predict_y(tx_te, w_ridge_regression)
y_pred_tr = predict_y(tx_tr, w_ridge_regression)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'ridge_regression_submission'
create_csv_submission(ids_te, y_pred, name)
print("------------------------------------------")


# change logisitic
print("Change logistic...")
initial_w = np.random.uniform(0, 1, np.shape(tx_tr)[1])
y_tr[np.where(y_tr == -1)] = 0
y_te[np.where(y_te == -1)] = 0 
print("------------------------------------------")


# best gamma for LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
print("Logistic Regression...")
#best_gamma, best_rmse, acc, deg = cross_validation_logistic_regression(y_tr, tx_tr, k_fold, max_iters, gammas, degrees)
w_log_regression, loss = logistic_regression(y_tr, tx_tr, initial_w, 1000, 0.08)
y_pred_te = predict_logistic(tx_te, w_log_regression)
y_pred_tr = predict_logistic(tx_tr, w_log_regression)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'logistic_regression_submission'
create_csv_submission(ids_te, y_pred_te, name)
print("------------------------------------------")


# best gamma AND lambda for REGULARIZED LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
print("Regularized Logistic Regression...")
#best_gamma, best_lambda, best_rmse, tmp, deg = cross_validation_reg_logistic_regression(y_tr, tx_tr, k_fold, max_iters, lambdas, gammas, degrees)
w_reg_log_regression, loss = reg_logistic_regression(y_tr, tx_tr, 0.05, initial_w, 1000, 0.08)
y_pred_te = predict_logistic(tx_te, w_reg_log_regression)
y_pred_tr = predict_logistic(tx_tr, w_reg_log_regression)
print("Accuracy: ", compute_accuracy(y_pred_tr, y_tr))
name = 'reg_logistic_regression_submission'
create_csv_submission(ids_te, y_pred_te, name)
print("------------------------------------------")


# convert the results into the file that we can put on the website challenge
print("DONE")





