# -*- coding: utf-8 -*-


import csv
import numpy as np


"""---------------DATA PRE-PROCESSING FUNCTION---------------"""


def standardize(x, mean_x = None, std_x = None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis = 0)
    x = x / std_x
    return x, mean_x, std_x


def handle_missing(x, means=None):
    """Handle NaN data"""
    if means is None:
        means = []
        for i in range(x.shape[1]):
            nan = (x[:, i] == -999)
            valid = (x[:, i] != -999)
            mean_i = np.mean(x[valid, i])
            x[nan, i] = mean_i
            means.append(mean_i)
    else:
        for i in range(x.shape[1]):
            nan = (x[:, i] == -999)
            x[nan, i] = means[i]
    return x, means


def correlated(y, x, threshold=0):
    """
    compute the correlation between with each feature
    return the index of the feature with correlation bigger than threshold
    """
    print('y shape', y.shape)
    cor = np.corrcoef(y.T, x.T)
    corr_degree = cor[0, 1:]
    print("All correlation\n", corr_degree)
    select_corr = corr_degree[np.abs(corr_degree) >= threshold]
    print("select_correlation\n", select_corr)
    sorted_index = np.argsort(np.abs(corr_degree))[::-1]
    print("sorted_index\n", sorted_index[:len(select_corr)])
    return sorted_index[:len(select_corr)]


def feature_cross_prod(x, corr_ind):
    prod_x = []
    for i in range(len(corr_ind)):
        for j in range(i+1, len(corr_ind)):
            m = x[:, corr_ind[i]] * x[:, corr_ind[j]]
            prod_x.append(m)
    prod_x = np.asarray(prod_x)
    return prod_x.T


"""---------------HELPER FUNCTIONS FOR LEAST SQUARES GD---------------"""


def build_model_data(y, x):
    """Form (y,tX) to get regression data in matrix form."""
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad


"""---------------HELPER FUNCTIONS FOR LEAST SQUARES SGD---------------"""


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


"""---------------HELPER FUNCTIONS FOR RIDGE REGRESSION---------------"""


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


"""---------------COMMON HELPER FUNCTIONS FOR 6 METHODS---------------"""


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = e.dot(e) / (2 * len(e))
    return mse


def split_data(y, x, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return y_tr, y_te, x_tr, x_te


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def degree_of_accuracy(y, x, w):
    """ return the percentage of right prediction"""
    y_pred = np.dot(x, w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    right = np.sum(y_pred == y)
    accuracy = float(right) / float(y.shape[0])

    return accuracy


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})