import numpy as np
import csv
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

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

def compute_mae(y, tx, w):
    """Calculate the loss using MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #compute loss by MAE
    error = y - tx.dot(w)
    return np.mean(np.abs(error))

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# compute ridge mse, predict is for the clipping
def compute_ridge_mse(y, tx, w):
    pred = predict_y(tx, w)
    error = y-pred
    return 1/2*np.mean(error**2)

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
    

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
    """Create correlation matrix"""
    cor_matrix = np.corrcoef(cleaned_columns)
    upper_tri = np.triu(cor_matrix)
    for i in range(upper_tri.shape[0]):
        upper_tri[i][i]= 0
    return list(zip(*np.where(upper_tri > fact)))


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def count(zipped):
    """count the number of features each feature is higly correlated with"""
    c = [item for z in zipped for item in z]
    return [(x,c.count(x)) for x in set(c)]


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def del_biggest_cor(cleaned_columns, fact):
    """deletes the features that are too correlated"""
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

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def clean_standardize(input_data, yb, fact=1, center= False, logistic = False):
    """Pipeline for pre-process the data, with multiple parameters.
    
    Args:
        input_data: input dataset
        yb: labels of the dataset
        fact:threshold for the suppression of the correlation between features 
        center: boolean : balance the data
        logistic: boolean : change the label -1 to 0 (adaptation for the logistic functions)
        
    Returns:
        The cleaned data set
    """
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

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

# build model data when not using polynomial expansion
def build_model_data(x, y):
    """Add the bias column"""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return tx,y

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def compute_accuracy(y_pred, y):
    """Compute the accuracy of the predictions y_pred"""
    N = len(y)
    return 100*((y_pred==y).sum())/N

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def predict_y(tx, w):
    """Predict the labels of the dataset given the weights"""
    y_pred = tx@w
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    return y_pred

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def predict_logistic(x: np.ndarray, w: np.ndarray):
    """Predict the labels of the dataset given the weights, for the logistic methods"""
    y_pred = sigmoid(x @ w)
    y_pred[np.where(y_pred <= 0.5)] = 0
    y_pred[np.where(y_pred > 0.5)] = 1
    return y_pred

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def sigmoid(t):
    """Compute the sigmoid function"""
    t = np.clip(t, -20, 20)
    return 1 / (1 + np.exp(-t))


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def build_poly(x, degree):
    """Compute the polynomial expansion given the data and the degree.
       Note that it adds the bias term"""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------

def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, rmse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    #plt.xlim(1e-4, 1)
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
    
#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------


def bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    plt.plot(
        degrees,
        rmse_tr.T,
        linestyle="-",
        color=([0.7, 0.7, 1]),
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        linestyle="-",
        color=[1, 0.7, 0.7],
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    plt.xlim(1, 9)
    plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.legend(loc=1)
    plt.title("Bias-Variance Decomposition")
    plt.savefig("bias_variance")


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
