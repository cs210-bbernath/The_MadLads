{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "39ff88b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "a1d7fed9",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = np.genfromtxt(\n",
    "        \"data/train.csv\", delimiter=\",\", skip_header=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "5ffc1743",
   "metadata": {},
   "outputs": [],
   "source": [
    "def sigmoid(t):\n",
    "    return 1 / (1 + np.exp(-t))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "6918e8bf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def logistic_regression(y, tx, initial_w, max_iters, gamma):\n",
    "    \"\"\"The logistic regression algorithm.\n",
    "        \n",
    "    Args:\n",
    "        y: numpy array of shape=(N, )\n",
    "        tx: numpy array of shape=(N,2)\n",
    "        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters\n",
    "        max_iters: a scalar denoting the total number of iterations of logistic regression\n",
    "        gamma: a scalar denoting the stepsize, aka learning rate\n",
    "        \n",
    "    Returns:\n",
    "        losses: a list of length max_iters containing the loss value (scalar) for each iteration of logistic_regression\n",
    "        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), \n",
    "            for each iteration of Logistic regression \n",
    "    \"\"\"\n",
    "    ws = [initial_w]\n",
    "    losses = []\n",
    "    w = initial_w\n",
    "    N = len(y)\n",
    "    \n",
    "    for n_iter in range(max_iters):\n",
    "        # compute gradient\n",
    "        gradient = 2/N * tx.T.dot(sigmoid(tx.dot(w)) - y)\n",
    "        \n",
    "        # update w by gradient\n",
    "        w = w - (gamma * gradient)\n",
    "        \n",
    "        y_pred = sigmoid(tx.dot(w))\n",
    "        \n",
    "        # compute loss\n",
    "        loss = -np.sum(np.dot(y.T,np.log(y_pred)+ np.dot((1-y).T, np.log(1-y_pred)))) /(len(y_pred))\n",
    "        \n",
    "        # store w and loss\n",
    "        ws.append(w)\n",
    "        losses.append(loss)\n",
    "        print(\"Logistic regression iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}\".format(\n",
    "              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))\n",
    "\n",
    "    return losses, ws"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ada0d53f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def reg_logistic_regression(y, tx, lambda_ initial_w, max_iters, gamma):\n",
    "    \"\"\"The reg logistic regression algorithm.\n",
    "        \n",
    "    Args:\n",
    "        y: numpy array of shape=(N, )\n",
    "        tx: numpy array of shape=(N,2)\n",
    "        lambda_: regularization factor\n",
    "        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters\n",
    "        max_iters: a scalar denoting the total number of iterations of reg logistic regression\n",
    "        gamma: a scalar denoting the stepsize, aka learning rate\n",
    "        \n",
    "    Returns:\n",
    "        losses: a list of length max_iters containing the loss value (scalar) for each iteration of reg logistic_regression\n",
    "        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), \n",
    "            for each iteration of reg Logistic regression \n",
    "    \"\"\"\n",
    "    ws = [initial_w]\n",
    "    losses = []\n",
    "    w = initial_w\n",
    "    N = len(y)\n",
    "    \n",
    "    for n_iter in range(max_iters):\n",
    "        # compute gradient\n",
    "        gradient = 2/N * tx.T.dot(sigmoid(tx.dot(w)) - y) + (lambda_/N)*w\n",
    "        \n",
    "        # update w by gradient\n",
    "        w = w - (gamma * gradient)\n",
    "        \n",
    "        y_pred = sigmoid(tx.dot(w))\n",
    "        \n",
    "        # compute loss\n",
    "        loss = -np.sum(np.dot(y.T,np.log(y_pred)+ np.dot((1-y).T, np.log(1-y_pred)))) /(len(y_pred)) + (lambda_/(2*N))*(w.T.dot(w))\n",
    "        \n",
    "        # store w and loss\n",
    "        ws.append(w)\n",
    "        losses.append(loss)\n",
    "        print(\"Reg Logistic regression iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}\".format(\n",
    "              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))\n",
    "\n",
    "    return losses, ws"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
