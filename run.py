import numpy as np
from helper import *
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

# this is for the correlation between two features : we remove 1 of the two feature if they have more than 0.5 correlation.
factor = 0.5
tx_tr, y_tr = clean_standardize(x_tr, y_tr, factor)
tx_te, y_te = clean_standardize(x_te, y_te, factor)
print("------------------------------------------")

######################################################################

# best gamma for GRADIENT descent mean squared error using cross validation
print("Gradient descent computing...")

# from cross validation have that the best combo is :
degree_gd = 1
max_iters_gd = 1000
gamma_gd = 0.08

# polynomial expansion only of degree 1
tx_tr_1 = build_poly(tx_tr, degree_gd)
tx_te_1 = build_poly(tx_te, degree_gd)

# creation of inital weights
initial_w_gd = np.random.uniform(-1,1,tx_tr_1.shape[1])

# run the gradient descent function with optimal parameters
w_gradient_descent, loss = mean_squared_error_gd(y_tr, tx_tr_1, initial_w_gd, max_iters_gd, gamma_gd)

# predict on the test set
y_pred_te_gd = predict_y(tx_te_1, w_gradient_descent)

# creation of the gradient descent file
name = 'gradient_descent_submission'
create_csv_submission(ids_te, y_pred_te_gd, name)
print("Gradient descent submission file created !")
print("------------------------------------------")

######################################################################

# For SGD, you must use the standard mini-batch-size 1
# best gamma for STOCHASTIC GRADIENT descent mean squared error using cross validation
print("Stochastic Gradient descent computing...")

# from cross validation have that the best combo is :
degree_sgd = 1
max_iters_sgd = 1000
gamma_sgd = 0.009

# creation of inital weights
initial_w_sgd = np.random.uniform(-1,1,tx_tr_1.shape[1])

# run the stochastic gradient descent function with optimal parameters (note that we use tx_tr_1 as the degree is still 1)
w_stoch_gradient_descent, loss = mean_squared_error_sgd(y_tr, tx_tr_1, initial_w_sgd, max_iters_sgd, gamma_sgd)

# predict on the test set
y_pred_te_sgd = predict_y(tx_te_1, w_stoch_gradient_descent)

# creation of the stochastic gradient descent file
name = 'stoch_gradient_descent_submission'
create_csv_submission(ids_te, y_pred_te_sgd, name)
print("Stochastic gradient descent submission file created !")
print("------------------------------------------")

######################################################################

# LEAST SQUARES using cross validation
print("Least Squares computing...")

# from cross validation have that the best degree is :
degree_ls = 6
lambda_ls = 0 # this is for the call to ridge regression

# polynomial expansion of degree 6
tx_tr_6 = build_poly(tx_tr, degree_ls)
tx_te_6 = build_poly(tx_te, degree_ls)

# run the least square function with optimal degree
w_least_squares, loss = ridge_regression(y_tr, tx_tr_6, lambda_ls)

# predict on the test set
y_pred_te_ls = predict_y(tx_te_6, w_least_squares)

# creation of the least square file
name = 'least_squares_submission'
create_csv_submission(ids_te, y_pred_te_ls, name)
print("Least Square submission file created !")
print("------------------------------------------")

######################################################################

# best lambda for RIDGE REGRESSION (least squared with lambda) using cross validation
print("Ridge Regression computing...")

# from cross validation have that the best combo is :
degree_ridge = 15
lambda_ridge = 0.00005

# polynomial expansion of degree 6
tx_tr_15 = build_poly(tx_tr, degree_ridge)
tx_te_15 = build_poly(tx_te, degree_ridge)

# run the ridge regression function with optimal parameters
w_ridge_regression, loss = ridge_regression(y_tr, tx_tr_15, lambda_ridge)

# predict on the test set
y_pred_te_ridge = predict_y(tx_te_15, w_ridge_regression)

# creation of the ridge regression file
name = 'ridge_regression_submission'
create_csv_submission(ids_te, y_pred_te_ridge, name)
print("Ridge regression submission file created !")
print("------------------------------------------")

# change logisitic
print("Changing to logistic adaptation labels...")
y_tr[np.where(y_tr == -1)] = 0
y_te[np.where(y_te == -1)] = 0 
print("Changed done !")
print("------------------------------------------")

######################################################################

# best gamma for LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
print("Logistic Regression...")

# from cross validation have that the best combo is :
degree_log = 1
gamma_log = 0.08
max_iters_log = 1000
lambda_log = 0 # this is for the call to regularized logistic regression

# creation of inital weights
initial_w_log = np.random.uniform(-1,1,tx_tr_1.shape[1])

# run the logistic regression function with optimal parameters (note that we use tx_tr_1 as the degree is still 1)
w_log_regression, loss = reg_logistic_regression(y_tr, tx_tr_1, lambda_log, initial_w_log, max_iters_log, gamma_log)

# predict on the test set
y_pred_te_log = predict_logistic(tx_te_1, w_log_regression)

# changing back to the correct label
y_pred_te_log[np.where(y_pred_te_log == 0)] = -1 

# creation of the logistic regression file
name = 'logistic_regression_submission'
create_csv_submission(ids_te, y_pred_te_log, name)
print("Logistic regression submission file created !")
print("------------------------------------------")

######################################################################

# best gamma AND lambda for REGULARIZED LOGISTIC REGRESSION (stochastic) gradient descent using cross validation
print("Regularized Logistic Regression...")

# from cross validation have that the best combo is :
degree_reg = 1
gamma_reg = 0.085
lambda_reg = 0.001
max_iters_reg = 1000

# creation of inital weights
initial_w_reg = np.random.uniform(-1,1,tx_tr_1.shape[1])

# run the regularized logistic regression function with optimal parameters (note that we use tx_tr_1 as the degree is still 1)
w_reg_log_regression, loss = reg_logistic_regression(y_tr, tx_tr_1, lambda_reg, initial_w_reg, max_iters_reg, gamma_reg)


# predict on the test set
y_pred_te_reg = predict_logistic(tx_te_1, w_reg_log_regression)

# changing back to the correct label
y_pred_te_reg[np.where(y_pred_te_reg == 0)] = -1 

# creation of the regularized logistic regression file
name = 'reg_logistic_regression_submission'
create_csv_submission(ids_te, y_pred_te_reg, name)
print("Regularized logistic regression submission file created !")
print("------------------------------------------")

######################################################################

# convert the results into the file that we can put on the website challenge
print("ALL DONE !")
