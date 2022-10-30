import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using either MSE or MAE.
    
    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    error = y - tx.dot(w)
    return 1/2 * np.mean(error**2)

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def compute_ridge_mse(y, tx, w):
    pred = predict_y(tx, w)
    error = y-pred
    return 1/2*np.mean(error**2)
    

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        An numpy array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N = len(y)
    error = y - tx.dot(w)
    gradient = -(1/N) * tx.T.dot(error)
    
    return gradient

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD 
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
        #print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.
        
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.
        
    Returns:
        A numpy array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    N = len(y)
    error = y - tx.dot(w)
    stoch_gradient = -(1/N) * tx.T.dot(error)
    
    return stoch_gradient

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).
            
    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        
    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD 
    """
    
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size=batch_size):
            # compute stoch_gradient
            stoch_gradient = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            
            # compute loss
            loss = compute_mse(y, tx, w)
            
            # update w by gradient
            w = w - gamma * stoch_gradient
            
            # store w and loss
            ws.append(w)
            losses.append(loss)
        #print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
        #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
          
 #------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
            
def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def create_cor_matrix(cleaned_columns, fact):
    cor_matrix = np.corrcoef(cleaned_columns)
    upper_tri = np.triu(cor_matrix)
    for i in range(upper_tri.shape[0]):
        upper_tri[i][i]= 0
    return list(zip(*np.where(upper_tri > fact)))


def count(zipped):
    c = [item for z in zipped for item in z]
    return [(x,c.count(x)) for x in set(c)]


def del_biggest_cor(cleaned_columns, fact):
    zipped = create_cor_matrix(cleaned_columns, fact)
    while len(zipped) > 0 :
        biggest = (-1,-1)
        cnt = count(zipped)
        #print(cnt)
        for c in cnt:
            if c[1] > biggest[1]:
                biggest = c
        del cleaned_columns[biggest[0]]
        #print(np.shape(cleaned_columns))
        zipped = create_cor_matrix(cleaned_columns, fact)
    return cleaned_columns

def clean_standardize(input_data, yb, fact=1, center= False, logistic = False):
    if center:
        diff = yb.sum()
        seed = 12
        indices = [i for i, x in enumerate(yb) if x == -1]
        np.random.shuffle(indices)
        index = np.arange(len(yb))
        ind = indices[:int(np.abs(diff))]
        ind_keep = ~np.isin(index, ind)
        input_data = input_data[ind_keep]
        yb = yb[ind_keep]
    cleaned_columns = [c for c in input_data.T if (c==-999).sum()/len(c) < 0.2]
    for c in cleaned_columns:
        numb_of_nan = (c==-999).sum()
        median = np.median(list(filter(lambda x : x!= -999, c)))
        c[c == -999] = median
    del_biggest_cor(cleaned_columns, fact)
    std_data, mean, std = standardize(np.transpose(cleaned_columns))
    if logistic:
        yb[np.where(yb == -1)] = 0
    return std_data, yb

# build model data when not using polynomial expansion
def build_model_data(x, y):
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx,y
def compute_accuracy(y_pred, y):
    N = len(y)
    return 100*((y_pred==y).sum())/N

def predict_y(tx, w):
    y_pred = tx@w
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred
def predict_logistic(x: np.ndarray, w: np.ndarray):
    y_pred = sigmoid(x @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred


def sigmoid(t):
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))
